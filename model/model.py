import torch
from torch import nn
from third_party.mamba import Mamba, MambaConfig
from model.model_utils import smooth, resolve_device
from typing import Tuple, Union
# from model.kernels import *

import numpy as np
from tqdm import tqdm

# plt.rcParams['text.usetex'] = True
import gc

"""
these classes are the body of the Ouroboros method. An Ouroboros consists of three parallel Mamba encoders:
an Omega, a Gamma, and a kernel encoder. The output of each encoder is linearly mapped to our latent features.
These features are then used to reconstruct the estimated second derivative of the audio. See figure 1 of our 
paper for a visual illustration of the procedure.

Models are implemented using the mamba.py package (https://github.com/alxndrTL/mamba.py/),
with elements used included in `third_party` for convenience.

"""


class Ouroboros(nn.Module):
    def __init__(
        self,
        d_data: int,
        kernel: nn.Module,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 1,
        device: Union[str, torch.device] = "cuda",
        tau: float = 1 / 10000,
        smooth_len: float = 0.001,
    ):

        super().__init__()

        device = resolve_device(device)
        ## if stacking on data dimension, d should be 4* d_data (y,dy,rev y, rev dy)
        ## if stacking on time dimension, d should be 2*d_data (y,dy). We are stacking on the time dimension.
        omegaConfig = MambaConfig(
            d_model=2 * d_data,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )
        gammaConfig = MambaConfig(
            d_model=2 * d_data,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )

        self.omega_mamba = Mamba(omegaConfig).to(device)
        self.gamma_mamba = Mamba(gammaConfig).to(device)

        self.omega_net = nn.Linear(
            in_features=2 * d_data, out_features=d_data, device=device
        )  # output unconstrained
        self.gamma_net = nn.Linear(
            in_features=2 * d_data, out_features=d_data, device=device
        )  # output unconstrained

        kernelConfig = MambaConfig(
            d_model=2 * d_data,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )

        self.kernel_mamba = Mamba(kernelConfig).to(device)

        self.tau = tau
        self.smooth_len = smooth_len
        self.kernel = kernel.to(device)
        self.kernel.device = device
        self.kernel.tau = self.tau
        self.names = [r"$\omega$", r"$\gamma$", "weighted kernels", "states"]

    @property
    def device(self) -> torch.device:
        """device the model currently lives on (stays correct after `.to(...)`)"""
        return next(self.parameters()).device

    def forward(
        self,
        x: torch.FloatTensor,
        dxdt: torch.FloatTensor,
        dt: float,
        smoothing: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        predicts second derivative at time t.
        all other predictions should be done in the train loop (train_utils.py)

        inputs
        ------
            - x: cleaned audio segment
            - dxdt: first derivative estimate, scaled by sample interval dt
            - dt: sample interval
            - smoothing: whether we smooth functions during training. We do not, but you can

        outputs
        ------
            - yhat: model predicted second derivative, scaled by model time constant tau^2
            - weights: kernel weights over whole segment. these are regularized towards simplicity during training
        """

        # scale without modifying the caller's tensor in place
        dxdt = dxdt * self.tau / dt  # this is now \tau dx

        B, L, D = x.shape

        # x: x_0, x_dt, x_2dt,...
        smooth_len = int(round(self.smooth_len / dt))

        z = torch.cat([x, dxdt], dim=-1)
        L = z.shape[1]

        # we feed the audio and its first derivative, along with a time-reversed version,
        # to each encoder
        x_in = torch.cat([torch.flip(z, [1]), z], dim=1)  # stack on time dimension

        omegaControl = self.omega_mamba(x_in)[:, L:, :]
        gammaControl = self.gamma_mamba(x_in)[:, L:, :]
        kernelControl = self.kernel_mamba(x_in)[:, L:, :]

        omega = self.omega_net(
            omegaControl
        ).abs()  # Since we take omega^2 anyway, we take the absolute value to prevent things from switching around too much
        gamma = self.gamma_net(gammaControl)
        if smoothing:
            # smooth our model functions, if we choose to do so. I do not.
            omega = smooth(omega, smooth_len)
            gamma = smooth(gamma, smooth_len)
        weighted_kernels, weights = self.kernel(z, kernelControl)

        z1 = z[:, :, :1]
        z2 = z[:, :, 1:]

        yhat = -(omega**2) * z1 - gamma * z2 - weighted_kernels

        return yhat, weights

    def get_funcs(
        self,
        x: torch.FloatTensor,
        dxdt: torch.FloatTensor,
        dt: float,
        smoothing: bool = False,
        max_len_t: int = 4,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        Union[list, torch.FloatTensor],
    ]:
        """
        given data, returns learned model functions --- latent features

        inputs
        ------
            - x: audio segment
            - dxdt: first derivative estimate, scaled by timestep dt
            - dt: sampling timestep
            - smoothing: whether to smooth model functions. Here, we typically smooth omega & gamma
            - max_len_t: maximum audio length that we will process simultaneously. triggers sequential processing, if x is too long

        returns
        ------
            - omega: instantaneous frequency term
            - gamma: instantaneous damping termp
            - weighted_kernels: nonlinearity term
            - weights: weights on kernels in nonlinearity
            - states: control signals for omega, gamma, kernels. I typically do not use these.
        """

        B, L, D = x.shape

        ## as in forward (not in place, so the caller's dxdt is left alone)
        dxdt = dxdt * self.tau / dt

        max_len_s = int(round(max_len_t / dt))

        smooth_len = int(
            round(self.smooth_len / dt)
        )  # convert smooth len from seconds to samples

        z = torch.cat([x, dxdt], dim=-1)
        L = z.shape[1]
        if L > max_len_s:
            # if a function is too long, use sequential processing to retrieve latent features
            print("using step by step functions")
            yhat, omega, gamma, weighted_kernels, weights = self.funcs_by_step(
                z, dt, smoothing=smoothing, step_size=max_len_s
            )  # update for chunked steps
            return omega, gamma, weighted_kernels, weights, []

        x_in = torch.cat([torch.flip(z, [1]), z], dim=1)  ## stack on time dimension

        omegaControl = self.omega_mamba(x_in)[:, L:, :]
        gammaControl = self.gamma_mamba(x_in)[:, L:, :]
        kernelControl = self.kernel_mamba(x_in)[:, L:, :]

        omega = self.omega_net(omegaControl).abs()
        gamma = self.gamma_net(gammaControl)

        if smoothing:
            omega = smooth(omega.abs(), smooth_len)
            gamma = smooth(gamma, smooth_len)

        weighted_kernels, weights = self.kernel(z, kernelControl)

        return (
            omega,
            gamma,
            weighted_kernels,
            weights,
            torch.cat([omegaControl, gammaControl, kernelControl], dim=-1),
        )

    def integrate(
        self,
        x,
        dxdt,
        dx2,
        dt,
        method="RK45",
        st=0.05,
        scaled=True,
        with_residual=False,
        smoothing=True,
        strategy="interp",
        oversample_prop=1,
    ):
        """
        don't use this.
        """
        print(
            "Don't use this to integrate. Instead, use the integration methods in train/eval.py"
        )
        return

    def funcs_by_step(
        self,
        z: torch.FloatTensor,
        dt: float,
        smoothing: bool = False,
        step_size: int = 10000,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        """
        if your audio segment is too long, we'll just get these functions step by step
        rather than all at once. The Mamba encoders carry their state across chunks, so
        the result matches processing the whole segment at once (up to numerical error).
        This function assumes you do NOT need to backprop at all. That would probably take forever.
        So I would recommend keeping that as is.

        inputs
        ------
            - z: stacked x, \tau dx -- input to mamba models
            - dt: sampling timestep
            - smoothing: whether we smooth the latent features
            - step_size: maximum number of samples to run through the model at once

        returns
        ------
            - yhat: model estimated second derivative, scaled by tau^2 (as in `forward`)
            - omegas: instantaneous frequency (B x L x 1)
            - gammas: instantaneous damping (B x L x 1)
            - kernel: nonlinearity (B x L x 1)
            - weights: kernel weights for nonlinearity (B x L x P x P)
        """

        smooth_len = int(round(self.smooth_len / dt))
        B, L, D = z.shape

        def init_cache(mamba):
            # (hidden state, last d_conv - 1 conv inputs) for each layer. a hidden
            # state of None means "start from zero"
            cfg = mamba.config
            return [
                (
                    None,
                    torch.zeros(
                        (B, cfg.d_inner, cfg.d_conv - 1),
                        device=z.device,
                    ),
                )
                for _ in mamba.layers
            ]

        omega_cache = init_cache(self.omega_mamba)
        gamma_cache = init_cache(self.gamma_mamba)
        kernel_cache = init_cache(self.kernel_mamba)

        # as in `forward`: time-reversed context, followed by the data
        x_in = torch.cat([torch.flip(z, [1]), z], dim=1)
        L_new = x_in.shape[1]

        omegas, gammas, weights, kernel = [], [], [], []
        with torch.no_grad():
            for ii in tqdm(
                range(0, L_new, step_size),
                total=int(np.ceil(L_new / step_size)),
                desc=f"iterating through segment of length {L}",
            ):
                if np.mod(ii, 10000) == 0:
                    ## clean up stuff every so often
                    gc.collect()
                end_ind = min(L_new, ii + step_size)
                s = x_in[:, ii:end_ind, :]

                # each encoder keeps its own state, and has to see the whole input
                # (including the reversed half), even though we discard the outputs there
                omega_ctrl, omega_cache = self.omega_mamba.chunk_step(s, omega_cache)
                gamma_ctrl, gamma_cache = self.gamma_mamba.chunk_step(s, gamma_cache)
                kernel_ctrl, kernel_cache = self.kernel_mamba.chunk_step(
                    s, kernel_cache
                )

                if end_ind > L:
                    # only outputs for the second (forward) half of x_in are kept.
                    # local_start indexes into this chunk; data_start indexes into z
                    local_start = max(0, L - ii)
                    data_start = max(0, ii - L)
                    data_end = end_ind - L

                    omega = self.omega_net(omega_ctrl[:, local_start:]).abs()
                    gamma = self.gamma_net(gamma_ctrl[:, local_start:])
                    weighted_kernels, w = self.kernel(
                        z[:, data_start:data_end, :], kernel_ctrl[:, local_start:]
                    )
                    omegas.append(omega)
                    gammas.append(gamma)
                    kernel.append(weighted_kernels)
                    weights.append(w)

        omegas = torch.cat(omegas, dim=1)
        gammas = torch.cat(gammas, dim=1)
        kernel = torch.cat(kernel, dim=1)
        weights = torch.cat(weights, dim=1)

        if smoothing:
            omegas = smooth(omegas.abs(), smooth_len)
            gammas = smooth(gammas, smooth_len)

        # same form as `forward`
        yhat = -(omegas**2) * z[:, :, :1] - gammas * z[:, :, 1:] - kernel

        return yhat, omegas, gammas, kernel, weights
