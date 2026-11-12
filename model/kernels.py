import numpy as np
from abc import abstractmethod
import torch
from torch import nn
from model.model_utils import smooth, resolve_device
from typing import Tuple

"""
These kernel functions define the nonlinearity of our model. They are separate to make analysis
a little easier and code a little cleaner.
In general, these require a number of terms, an x dim, a z dim, and an activation function.
The activation function we typically use is the identity, but one could further constrain 
the structure of the nonlinearity by changing the activation

For all experiments, we use the FullPolyModule.
This can be extended to other kernels, but if you wanto to use them you can implement them yourselves
"""


class kernelModule(nn.Module):
    def __init__(self, nTerms, device, x_dim, z_dim, activation):

        super().__init__()

        self.nTerms = nTerms
        self.device = resolve_device(device)
        self.d = x_dim
        self.n = z_dim
        self.activation = activation
        self.tau = 1
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def get_weights(self, z, smooth_len):

        return smooth(self.activation(self.weights(z)), smooth_len)


class fullPolyModule(kernelModule):
    """
    uses products of polynomials to estimate functions. Takes control
    functions from mamba encoders and projects to weights, then calculates
    kernel based on those weights.
    """

    nTerms: int  # number of polynomial terms included
    device: str  # cuda or cpu. falls back to cpu if cuda is requested but unavailable
    x_dim: int  # data dimension
    z_dim: int  # number of inputs
    lam: float  # base regularization weight on polynomial weights
    activation: callable  # activation function for weights

    def __init__(self, nTerms, device, x_dim, z_dim, lam=0.01, activation=lambda x: x):

        super().__init__(nTerms, device, x_dim, z_dim, activation)

        self.poly_dim = nTerms

        self.weights = nn.Linear(self.n, (self.poly_dim + 1) ** 2).to(self.device)
        # a (non-persistent) buffer, so that .to(device) moves it along with the weights
        self.register_buffer(
            "powers",
            torch.arange(0, self.poly_dim + 1, device=self.device),
            persistent=False,
        )
        self.lam = lam

    def forward(
        self, x: torch.FloatTensor, z: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        computes weights and polynomial kernel

        inputs:
        -----
            - x: input data (audio and first derivative)
            - z: input control weights (from mamba encoder)
        returns:
        -----
            - computed kernel: weighted sum of product of polynomials
            - weights: learned weights on polynomial terms
        """

        B, L, d = x.shape
        _, _, n = z.shape
        weights = self.activation(self.weights(z))

        weights = weights.view(B, L, self.poly_dim + 1, self.poly_dim + 1)
        ### constant term
        weights[:, :, 0, 0] = weights[:, :, 0, 0] * 0
        ### y, ydot terms
        weights[:, :, 1, 0] = weights[:, :, 1, 0] * 0
        weights[:, :, 0, 1] = weights[:, :, 0, 1] * 0
        power_mat = (x[:, :, :, None].expand(-1, -1, -1, self.poly_dim + 1)).pow(
            self.powers
        )
        ## power mat is now: B x L x d x p
        ## we want to turn it into a B x L x 1 x p x p matrix

        z1 = power_mat[:, :, :1, :]
        z2 = power_mat[:, :, 1:, :]
        power_mat = torch.einsum("bldp,bldk -> blpk", z1, z2)
        ###
        ## batch x time x 1 (y [z1] or dy [z2]) x n poly
        ## -> batch x time x y degree x dy degree

        x = torch.einsum("blpd,blpd -> bl", weights, power_mat)

        return x[:, :, None], weights

    def forward_given_weights(
        self, x: torch.FloatTensor, weights: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        computes polynomial kernel given weights on polynomial terms

        inputs:
        -----
            - x: input data (audio and first derivative)
            - weights: weights on polynomial terms
        returns:
        -----
            - computed kernel: weighted sum of product of polynomials
        """

        B, L, d = x.shape
        weights = weights.view(B, L, self.poly_dim + 1, self.poly_dim + 1)
        ### constant term
        weights[:, :, 0, 0] = weights[:, :, 0, 0] * 0
        ### y, ydot terms
        weights[:, :, 1, 0] = weights[:, :, 1, 0] * 0
        weights[:, :, 0, 1] = weights[:, :, 0, 1] * 0
        power_mat = (x[:, :, :, None].expand(-1, -1, -1, self.poly_dim + 1)).pow(
            self.powers
        )

        z1 = power_mat[:, :, :1, :]
        z2 = power_mat[:, :, 1:, :]
        power_mat = torch.einsum("bldp,bldk -> blpk", z1, z2)
        x = torch.einsum("blpd,blpd -> bl", weights, power_mat)

        return x[:, :, None]

    def forward_given_weights_numpy(
        self, x: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """
        computes polynomial kernel given weights on polynomial terms.
        same as the above method, but using numpy instead of torch

        inputs:
        -----
            - x: input data (audio and first derivative)
            - weights: weights on polynomial terms
        returns:
        -----
            - computed kernel: weighted sum of product of polynomials
        """

        powers = self.powers.detach().cpu().numpy()
        if len(x.shape) != 3:
            x = np.reshape(x, (x.shape[0], -1, 2 * self.d))
        B, L, d = x.shape
        # if weights.shape != (B,L,self.d,self.poly_dim-1):
        weights = np.reshape(weights, (B, L, self.poly_dim + 1, self.poly_dim + 1))

        # constant term
        weights[:, :, 0, 0] = weights[:, :, 0, 0] * 0
        ### y, ydot terms
        weights[:, :, 1, 0] = weights[:, :, 1, 0] * 0
        weights[:, :, 0, 1] = weights[:, :, 0, 1] * 0

        power_mat = np.power(
            np.tile(x[:, :, :, None], (1, 1, 1, self.poly_dim + 1)), powers
        )
        z1 = power_mat[:, :, :1, :]
        z2 = power_mat[:, :, 1:, :]
        power_mat = np.einsum("bldp,bldk->blpk", z1, z2)
        x = np.einsum("blpd,blpd->bl", weights, power_mat)

        return x[:, :, None]
