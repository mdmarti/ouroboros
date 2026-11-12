"""
A standard convolutional variational autoencoder (VAE) for comparison against Ouroboros.

The architecture and objective follow the VAE from Autoencoded Vocal Analysis (AVA;
Goffinet et al., 2021, https://github.com/pearsonlab/autoencoded-vocal-analysis, MIT license):
a 7-layer convolutional encoder over 128 x 128 (mel-frequency x time) log-spectrograms, a
low-rank multivariate normal posterior, a mirrored transposed-convolutional decoder, and a
spherical Gaussian observation model with a fixed precision. Layer names match AVA's, so the
per-layer weights of an AVA checkpoint can be loaded into `VAE` layer by layer.

This file mirrors the structure of the main Ouroboros code, so that the two can be used
interchangeably in analyses:

    Ouroboros                                comparisons.vae
    ---------                                ---------------
    model.model.Ouroboros                    VAE
    model.get_funcs(...)                     VAE.get_latents(...)
    data.data_utils.get_loaders              get_spectrogram_loaders
    train.train.train                        train
    train.train.save_model / load_model      save_model / load_model
    train.eval.eval_model_error              eval_model_error
    train.model_cv.model_cv_lambdas          model_cv_precisions
    train.train_model.train_model            train_model

The differences are in the data. Ouroboros takes fixed-length chunks of raw audio, whereas the VAE
takes one spectrogram per vocalization (or per chunk, if `context_len` is given), each resized to
128 x 128. By default vocalizations of different lengths are time-stretched onto the same
grid, which is exactly the step that Ouroboros avoids.
"""

import glob
import os
import re
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from torch.distributions import LowRankMultivariateNormal
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.load_data import get_segmented_audio
from model.model_utils import resolve_device

X_SHAPE = (128, 128)
"""Spectrogram shape the network is built for: [frequency bins, time bins]"""
X_DIM = int(np.prod(X_SHAPE))

EPSILON = 1e-9

SPEC_PARAMS = {
    "min_freq": 500.0,  # lowest frequency in the spectrogram (Hz)
    "max_freq": 18000.0,  # highest frequency (Hz); clipped to the Nyquist frequency
    "nperseg": 512,  # STFT window length (samples)
    "noverlap": 256,  # STFT window overlap (samples)
    "num_freq_bins": X_SHAPE[0],
    "num_time_bins": X_SHAPE[1],
    "mel": True,  # mel-spaced frequency bins
    "time_stretch": True,  # widen the image for short vocalizations less than `max_dur` would (as in AVA)
    "max_dur": None,  # seconds. None: the vocalization's own duration (no stretching)
    "spec_min_val": 2.0,  # log-spectrogram value mapped to 0
    "spec_max_val": 4.0,  # log-spectrogram value mapped to 1
    "audio_scale": 2.0**15,  # AVA's thresholds assume int16-scaled audio; ours is in [-1, 1]
}
"""
spectrogram parameters. These are AVA's defaults. The thresholds `spec_min_val` and `spec_max_val`
depend on the recording gain, so they should be tuned per dataset (see `estimate_spec_range`).
"""

# --------------------------------------------------------------------------- spectrograms


def _mel(f):
    return 1127 * np.log(1 + np.asarray(f, dtype=float) / 700)


def _inv_mel(m):
    return 700 * (np.exp(np.asarray(m, dtype=float) / 1127) - 1)


def _as_1d(audio: np.ndarray) -> np.ndarray:
    return np.asarray(audio, dtype=np.float64).reshape(-1)


def _log_stft(
    audio: np.ndarray, fs: float, p: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """log |STFT| of a vocalization, with its DC offset removed. returns (freqs, times, log spec)"""
    audio = _as_1d(audio)
    audio = (audio - audio.mean()) * p["audio_scale"]
    f, t, spec = stft(audio, fs=fs, nperseg=p["nperseg"], noverlap=p["noverlap"])
    return f, t, np.log(np.abs(spec) + EPSILON)


def make_spectrogram(
    audio: np.ndarray,
    fs: float,
    p: Optional[dict] = None,
    max_dur: Optional[float] = None,
) -> np.ndarray:
    """
    make a spectrogram of a single vocalization, in the style of AVA: log-magnitude STFT,
    resampled onto mel-spaced frequency bins and a fixed number of time bins, then
    clipped and scaled to [0, 1].

    The vocalization is centered in the image. Without `p["time_stretch"]` the image spans
    `max_dur` seconds, so short vocalizations occupy a small part of it. With time stretching (as in
    AVA's implementation) the padding on each side is set from sqrt(duration * max_dur) instead,
    so the image spans duration + max_dur - sqrt(duration * max_dur) seconds: between the
    vocalization's own duration and `max_dur`. Short vocalizations then take up more of the image,
    without every vocalization being stretched to the same length.

    inputs
    -----
        - audio: the vocalization (any shape that flattens to one channel of samples)
        - fs: sampling rate, in Hz
        - p: spectrogram parameters. defaults to `SPEC_PARAMS`
        - max_dur: length (s) of the window shown in the spectrogram. Overrides `p["max_dur"]`;
            if both are None, the vocalization's own duration is used (no padding)

    returns
    -----
        - a spectrogram, of shape (num_freq_bins, num_time_bins), with values in [0, 1].
            Vocalizations shorter than one STFT window give all zeros.
    """
    p = {**SPEC_PARAMS, **(p or {})}
    audio = _as_1d(audio)
    shape = (p["num_freq_bins"], p["num_time_bins"])
    if len(audio) < p["nperseg"]:
        return np.zeros(shape, dtype=np.float32)

    duration = len(audio) / fs
    if max_dur is None:
        max_dur = p["max_dur"]
    if max_dur is None:
        max_dur = duration

    f, t, logspec = _log_stft(audio, fs, p)

    max_freq = min(p["max_freq"], fs / 2)
    if p["mel"]:
        target_freqs = _inv_mel(
            np.linspace(_mel(p["min_freq"]), _mel(max_freq), p["num_freq_bins"])
        )
    else:
        target_freqs = np.linspace(p["min_freq"], max_freq, p["num_freq_bins"])

    shown = np.sqrt(duration * max_dur) if p["time_stretch"] else duration
    shoulder = 0.5 * (max_dur - shown)
    target_times = np.linspace(-shoulder, duration + shoulder, p["num_time_bins"])

    interp = RegularGridInterpolator(
        (f, t), logspec, method="linear", bounds_error=False, fill_value=-1e9
    )
    grid = np.stack(np.meshgrid(target_freqs, target_times, indexing="ij"), axis=-1)
    spec = interp(grid)

    spec = (spec - p["spec_min_val"]) / (p["spec_max_val"] - p["spec_min_val"])
    return np.clip(spec, 0.0, 1.0).astype(np.float32)


def estimate_spec_range(
    audio: Sequence[np.ndarray],
    fs: float,
    p: Optional[dict] = None,
    lo_q: float = 40.0,
    hi_q: float = 99.5,
) -> Tuple[float, float]:
    """
    choose `spec_min_val` and `spec_max_val` from the data, as percentiles of the pooled log
    spectrogram values. Spectrogram values are dominated by background, so `lo_q` is near the noise
    floor and `hi_q` near the loudest vocal energy. This is a heuristic stand-in for the by-eye tuning
    done in AVA, and the percentiles can be changed.

    inputs
    -----
        - audio: list of vocalizations
        - fs: sampling rate
        - p: spectrogram parameters
        - lo_q, hi_q: percentiles for the lower and upper bounds

    returns
    -----
        - (spec_min_val, spec_max_val)
    """
    p = {**SPEC_PARAMS, **(p or {})}
    values = []
    for a in audio:
        a = _as_1d(a)
        if len(a) < p["nperseg"]:
            continue
        _, _, ls = _log_stft(a, fs, p)
        values.append(ls.reshape(-1))
    if not values:
        raise ValueError("no vocalization is long enough to compute a spectrogram")
    values = np.concatenate(values)
    lo, hi = np.percentile(values, [lo_q, hi_q])
    if hi - lo < 1e-6:  # constant spectrograms
        hi = lo + 1.0
    return float(lo), float(hi)


def resolve_spec_params(
    audio: Sequence[np.ndarray],
    fs: float,
    p: Optional[dict] = None,
    tune_range: bool = False,
) -> dict:
    """
    fill in the data-dependent spectrogram parameters: the window length `max_dur` (the longest
    vocalization, unless given), and optionally the intensity range (see `estimate_spec_range`)

    returns a new parameter dictionary
    """
    p = {**SPEC_PARAMS, **(p or {})}
    if p["max_dur"] is None:
        p["max_dur"] = max(len(_as_1d(a)) for a in audio) / fs
    if tune_range:
        p["spec_min_val"], p["spec_max_val"] = estimate_spec_range(audio, fs, p)
    return p


class SpectrogramDataset(Dataset):
    """a dataset of spectrograms; each item is a float32 tensor of shape (128, 128)"""

    def __init__(self, specs: np.ndarray):
        self.specs = torch.from_numpy(np.asarray(specs, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx):
        return self.specs[idx]


def get_spectrogram_loaders(
    audio: Sequence[np.ndarray],
    fs: float,
    num_workers: int = 4,
    batch_size: int = 32,
    train_size: float = 0.8,
    cv: bool = False,
    seed: Optional[int] = None,
    p: Optional[dict] = None,
) -> dict:
    """
    the VAE's counterpart of `data.data_utils.get_loaders`. converts vocalizations to spectrograms,
    then splits them at random into train/test (and validation, if `cv`) dataloaders. With
    `cv=True` the data not used for training is divided equally between validation and test,
    exactly as in `get_loaders`.

    inputs
    -----
        - audio: list of vocalizations (e.g., from `data.load_data.get_segmented_audio`)
        - fs: sampling rate
        - num_workers: dataloader workers
        - batch_size: batch size
        - train_size: fraction of data to train on
        - cv: whether to make a validation set as well
        - seed: random seed for the split
        - p: spectrogram parameters. `max_dur` defaults to the longest vocalization

    returns
    -----
        - dictionary of dataloaders with keys 'train', 'test', and 'val' (if `cv`)
    """
    p = resolve_spec_params(audio, fs, p)
    specs = np.stack([make_spectrogram(a, fs, p) for a in audio], axis=0)

    X_train, X_test = train_test_split(specs, test_size=1 - train_size, random_state=seed)
    dls = {}
    if cv:
        X_val, X_test = train_test_split(X_test, test_size=0.5, random_state=seed)
        dls["val"] = DataLoader(
            SpectrogramDataset(X_val),
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=False,
        )
    dls["train"] = DataLoader(
        SpectrogramDataset(X_train),
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )
    dls["test"] = DataLoader(
        SpectrogramDataset(X_test),
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )
    return dls


# -------------------------------------------------------------------------------- model


class VAE(nn.Module):
    """
    convolutional VAE for 128 x 128 spectrograms, with the architecture of AVA's VAE.

    The approximate posterior is q(z|x) = N(mu, u u^T + diag(d)) (rank-1 plus diagonal), the prior is
    a spherical unit normal, and the observation model is p(x|z) = N(decoder(z), I / model_precision).
    The network is trained to maximize the ELBO.

    inputs
    -----
        - z_dim: latent dimension. The layers constrain this to roughly 8-64
        - model_precision: precision of the observation model. Larger values weight
            reconstruction more heavily relative to the prior
        - device: 'cuda', 'cpu', etc. Falls back to cpu (with a warning) if cuda is requested
            but unavailable
        - spec_params: the spectrogram parameters the model was trained with. Stored (and saved
            with checkpoints) so new audio can be turned into spectrograms in the same way
    """

    def __init__(
        self,
        z_dim: int = 32,
        model_precision: float = 10.0,
        device: Union[str, torch.device] = "cuda",
        spec_params: Optional[dict] = None,
    ):
        super().__init__()
        device = resolve_device(device)
        self.z_dim = z_dim
        self.model_precision = model_precision
        self.spec_params = {**SPEC_PARAMS, **(spec_params or {})}
        self.names = ["latent mean", "latent covariance factor", "latent diagonal"]
        self._build_network()
        self.to(device)

    def _build_network(self):
        # encoder
        self.conv1 = nn.Conv2d(1, 8, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, 2, padding=1)
        self.conv5 = nn.Conv2d(16, 24, 3, 1, padding=1)
        self.conv6 = nn.Conv2d(24, 24, 3, 2, padding=1)
        self.conv7 = nn.Conv2d(24, 32, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(16)
        self.bn6 = nn.BatchNorm2d(24)
        self.bn7 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc31 = nn.Linear(256, 64)
        self.fc32 = nn.Linear(256, 64)
        self.fc33 = nn.Linear(256, 64)
        self.fc41 = nn.Linear(64, self.z_dim)
        self.fc42 = nn.Linear(64, self.z_dim)
        self.fc43 = nn.Linear(64, self.z_dim)
        # decoder
        self.fc5 = nn.Linear(self.z_dim, 64)
        self.fc6 = nn.Linear(64, 256)
        self.fc7 = nn.Linear(256, 1024)
        self.fc8 = nn.Linear(1024, 8192)
        self.convt1 = nn.ConvTranspose2d(32, 24, 3, 1, padding=1)
        self.convt2 = nn.ConvTranspose2d(24, 24, 3, 2, padding=1, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(24, 16, 3, 1, padding=1)
        self.convt4 = nn.ConvTranspose2d(16, 16, 3, 2, padding=1, output_padding=1)
        self.convt5 = nn.ConvTranspose2d(16, 8, 3, 1, padding=1)
        self.convt6 = nn.ConvTranspose2d(8, 8, 3, 2, padding=1, output_padding=1)
        self.convt7 = nn.ConvTranspose2d(8, 1, 3, 1, padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.bn9 = nn.BatchNorm2d(24)
        self.bn10 = nn.BatchNorm2d(24)
        self.bn11 = nn.BatchNorm2d(16)
        self.bn12 = nn.BatchNorm2d(16)
        self.bn13 = nn.BatchNorm2d(8)
        self.bn14 = nn.BatchNorm2d(8)

    @property
    def device(self) -> torch.device:
        """device the model currently lives on (stays correct after `.to(...)`)"""
        return next(self.parameters()).device

    def encode(
        self, x: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        compute the approximate posterior q(z|x) = N(mu, u u^T + diag(d))

        inputs
        -----
            - x: spectrograms, B x 128 x 128

        returns
        -----
            - mu: posterior mean, B x z_dim
            - u: posterior covariance factor, B x z_dim x 1
            - d: positive posterior diagonal, B x z_dim
        """
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(self.bn1(x)))
        x = F.relu(self.conv2(self.bn2(x)))
        x = F.relu(self.conv3(self.bn3(x)))
        x = F.relu(self.conv4(self.bn4(x)))
        x = F.relu(self.conv5(self.bn5(x)))
        x = F.relu(self.conv6(self.bn6(x)))
        x = F.relu(self.conv7(self.bn7(x)))
        x = x.reshape(-1, 8192)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc41(F.relu(self.fc31(x)))
        u = self.fc42(F.relu(self.fc32(x))).unsqueeze(-1)  # rank-1 covariance factor
        d = torch.exp(self.fc43(F.relu(self.fc33(x))))  # exponentiated, so it is positive
        return mu, u, d

    def decode(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """
        compute the mean of p(x|z)

        inputs
        -----
            - z: latent samples, B x z_dim

        returns
        -----
            - reconstructions, flattened: B x 16384
        """
        z = F.relu(self.fc5(z))
        z = F.relu(self.fc6(z))
        z = F.relu(self.fc7(z))
        z = F.relu(self.fc8(z))
        z = z.reshape(-1, 32, 16, 16)
        z = F.relu(self.convt1(self.bn8(z)))
        z = F.relu(self.convt2(self.bn9(z)))
        z = F.relu(self.convt3(self.bn10(z)))
        z = F.relu(self.convt4(self.bn11(z)))
        z = F.relu(self.convt5(self.bn12(z)))
        z = F.relu(self.convt6(self.bn13(z)))
        z = self.convt7(self.bn14(z))
        return z.reshape(-1, X_DIM)

    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        encode, sample a latent, decode, and compute the negative ELBO. The
        reconstruction uses a single latent sample, so is stochastic

        inputs
        -----
            - x: spectrograms, B x 128 x 128

        returns
        -----
            - x_rec: reconstructed spectrograms, B x 128 x 128
            - loss: negative ELBO, summed over the batch (so, not divided by batch size)
        """
        mu, u, d = self.encode(x)
        latent_dist = LowRankMultivariateNormal(mu, u, d)
        z = latent_dist.rsample()
        x_rec = self.decode(z)

        # E_q[log p(z)]. (As in AVA, the normalizing constant is counted once, not per sample;
        # it does not affect the gradients.)
        elbo = -0.5 * (torch.sum(torch.pow(z, 2)) + self.z_dim * np.log(2 * np.pi))
        # E_q[log p(x|z)]
        pxz_term = -0.5 * X_DIM * np.log(2 * np.pi / self.model_precision)
        l2s = torch.sum(torch.pow(x.reshape(x.shape[0], -1) - x_rec, 2), dim=1)
        elbo = elbo + pxz_term - 0.5 * self.model_precision * torch.sum(l2s)
        # entropy of q(z|x)
        elbo = elbo + torch.sum(latent_dist.entropy())

        return x_rec.reshape(-1, *X_SHAPE), -elbo

    def reconstruct(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """deterministic reconstruction: decode the posterior mean. B x 128 x 128"""
        mu, _, _ = self.encode(x)
        return self.decode(mu).reshape(-1, *X_SHAPE)

    def get_latents(
        self, x: torch.FloatTensor, return_posterior: bool = False
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        """
        the VAE's analogue of `Ouroboros.get_funcs`: given spectrograms, return the latent features.
        This does not disable gradients; wrap it in `torch.no_grad()` and call `model.eval()` for analysis

        inputs
        -----
            - x: spectrograms, B x 128 x 128
            - return_posterior: also return the posterior covariance terms

        returns
        -----
            - mu: latent means, B x z_dim; or (mu, u, d) if `return_posterior`
        """
        mu, u, d = self.encode(x)
        if return_posterior:
            return mu, u, d
        return mu


def get_latents_from_loader(model: VAE, loader: DataLoader) -> np.ndarray:
    """
    latent means for every spectrogram in a loader. Pass an unshuffled loader
    (e.g. 'val' or 'test') if you want to match the output to labels

    returns
    -----
        - latent means, N x z_dim
    """
    model.eval()
    latents = []
    with torch.no_grad():
        for x in loader:
            latents.append(model.get_latents(x.to(model.device).float()).cpu().numpy())
    return np.concatenate(latents, axis=0)


def get_latents_from_audio(
    model: VAE, audio: Sequence[np.ndarray], fs: float, batch_size: int = 32
) -> np.ndarray:
    """
    latent means for new vocalizations, turned into spectrograms with the model's `spec_params`

    returns
    -----
        - latent means, N x z_dim
    """
    p = model.spec_params
    specs = np.stack([make_spectrogram(a, fs, p) for a in audio], axis=0)
    loader = DataLoader(SpectrogramDataset(specs), batch_size=batch_size, shuffle=False)
    return get_latents_from_loader(model, loader)


# ------------------------------------------------------------------------ save and load


def _checkpoint_epoch(path: str) -> int:
    match = re.search(r"checkpoint_(\d+)\.tar$", os.path.basename(path))
    if match is None:
        raise ValueError(f"not a checkpoint file name: {path}")
    return int(match.group(1))


def save_model(
    model: VAE,
    opt: torch.optim.Optimizer,
    location: str,
    max_saved: int = 5,
):
    """
    save a model to `location`, which should be named checkpoint_{epochs completed}.tar. Also
    removes older checkpoints from that folder, so that only the `max_saved` most recent remain.

    inputs
    -----
        - model: VAE to be saved
        - opt: optimizer for that model
        - location: filename to save to
        - max_saved: maximum number of checkpoints to keep in the folder
    """
    folder = os.path.dirname(location) or "."
    os.makedirs(folder, exist_ok=True)

    others = [f for f in glob.glob(os.path.join(folder, "checkpoint_*.tar")) if f != location]
    others.sort(key=_checkpoint_epoch)
    n_remove = len(others) - (max_saved - 1)
    for f in others[: max(n_remove, 0)]:
        os.remove(f)

    torch.save(
        {
            "vae": model.state_dict(),
            "opt": opt.state_dict(),
            "z_dim": model.z_dim,
            "model_precision": model.model_precision,
            "spec_params": model.spec_params,
        },
        location,
    )


def load_model(
    location: str, device: Union[str, torch.device] = "cuda"
) -> Tuple[VAE, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
    """
    load the most recent checkpoint in a folder

    inputs
    -----
        - location: folder containing checkpoint_{epochs completed}.tar files
        - device: device to load onto (falls back to cpu if cuda is unavailable)

    returns
    -----
        - model: VAE with the saved structure and weights
        - opt: its optimizer
        - scheduler: a fresh learning rate scheduler for that optimizer
        - epoch: number of epochs completed at the checkpoint
    """
    device = resolve_device(device)
    files = glob.glob(os.path.join(location, "checkpoint_*.tar"))
    if not files:
        raise FileNotFoundError(f"no checkpoints found in {location}")
    latest = max(files, key=_checkpoint_epoch)
    print(f"loading from {latest}")

    sd = torch.load(latest, weights_only=False, map_location=device)
    model = VAE(
        z_dim=sd["z_dim"],
        model_precision=sd["model_precision"],
        device=device,
        spec_params=sd.get("spec_params"),
    )
    model.load_state_dict(sd["vae"])
    opt = Adam(model.parameters(), lr=1e-3)
    opt.load_state_dict(sd["opt"])
    scheduler = ReduceLROnPlateau(opt, factor=0.75, patience=5, min_lr=1e-10)
    return model, opt, scheduler, _checkpoint_epoch(latest)


# ----------------------------------------------------------------------- train and eval


def plot_reconstructions(
    model: VAE, loader: DataLoader, save_path: str, n_specs: int = 5
) -> None:
    """save a figure of spectrograms (top) and their deterministic reconstructions (bottom)"""
    model.eval()
    x = next(iter(loader))[:n_specs].to(model.device).float()
    with torch.no_grad():
        rec = model.reconstruct(x).cpu().numpy()
    x = x.cpu().numpy()
    fig, axs = plt.subplots(2, len(x), figsize=(2.5 * len(x), 5), squeeze=False)
    for i in range(len(x)):
        for row, img in enumerate((x[i], rec[i])):
            axs[row, i].imshow(img, origin="lower", aspect="auto", vmin=0, vmax=1)
            axs[row, i].axis("off")
    axs[0, 0].set_title("data")
    axs[1, 0].set_title("reconstruction")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def train(
    model: VAE,
    optimizer: torch.optim.Optimizer,
    loaders: dict,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    nEpochs: int = 100,
    val_freq: int = 1,
    runDir: str = ".",
    vis_freq: int = 0,
    start_epoch: int = 0,
    save_freq: int = 0,
) -> Tuple[list, list, VAE, torch.optim.Optimizer]:
    """
    main train loop for a VAE, with the same shape as `train.train.train`. The loss is the
    negative ELBO, reported per spectrogram. There is no `loss_fn` or `dt`, since the objective is
    fixed and the inputs are spectrograms.

    inputs
    -----
        - model: a VAE
        - optimizer: optimizer for that VAE
        - loaders: dictionary of dataloaders with 'train' and 'val' entries
        - scheduler: learning rate scheduler, stepped on the validation loss. optional
        - nEpochs: total number of passes through the training data (training continues from
            `start_epoch`)
        - val_freq: how often (in epochs) to compute the validation loss
        - runDir: directory for tensorboard logs, checkpoints and reconstruction plots
        - vis_freq: how often (in epochs) to save a reconstruction plot. 0 for never
        - start_epoch: epoch to start at. 0 for a new model; the number of epochs already
            completed when resuming from a checkpoint
        - save_freq: how often (in epochs) to save a checkpoint. 0 for never. checkpoints are named
            by the number of epochs completed

    returns
    -----
        - train_losses: training loss per gradient step
        - val_losses: list of tuples (gradient step, validation loss)
        - the model
        - the optimizer
    """
    os.makedirs(runDir, exist_ok=True)
    writer = SummaryWriter(log_dir=runDir)
    train_losses, val_losses = [], []
    n_val = len(loaders["val"].dataset)

    for epoch in tqdm(range(start_epoch, nEpochs), desc="training model"):
        model.train()
        for idx, x in enumerate(loaders["train"], start=epoch * len(loaders["train"])):
            x = x.to(model.device).float()
            optimizer.zero_grad()
            _, loss = model(x)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item() / len(x))
            writer.add_scalar("Loss/train", train_losses[-1], idx)

        if epoch % val_freq == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x in loaders["val"]:
                    _, loss = model(x.to(model.device).float())
                    val_loss += loss.item()
            val_loss /= n_val
            if scheduler:
                scheduler.step(val_loss)
            val_losses.append((epoch * len(loaders["train"]), val_loss))
            writer.add_scalar("Loss/validation", val_loss, epoch * len(loaders["train"]))

        if vis_freq > 0 and epoch % vis_freq == 0:
            plot_reconstructions(
                model, loaders["val"], os.path.join(runDir, f"reconstructions_{epoch}.svg")
            )
        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            save_model(model, optimizer, os.path.join(runDir, f"checkpoint_{epoch + 1}.tar"))
    writer.close()
    return train_losses, val_losses, model, optimizer


def eval_model_error(
    dls: dict, model: VAE, comparison: str = "val"
) -> Tuple[
    Tuple[float, float], Tuple[float, float], Tuple[np.ndarray, np.ndarray]
]:
    """
    reconstruction error in terms of R^2 per spectrogram (decoding the posterior mean),
    in the same format as `train.eval.eval_model_error`. Note this is R^2 of the *spectrogram*, not of
    the audio's second derivative, so it is not comparable in value to Ouroboros's R^2

    inputs
    -----
        - dls: dictionary of dataloaders, containing 'train' and `comparison`
        - model: a trained VAE
        - comparison: 'val' or 'test'. the split to evaluate on, besides the train set

    returns
    -----
        (train, comparison) mean R^2
        (train, comparison) sd of R^2
        (train, comparison) R^2 for every spectrogram
    """
    model.eval()

    def r2_of(loader):
        r2s = []
        with torch.no_grad():
            for x in loader:
                x = x.to(model.device).float()
                rec = model.reconstruct(x)
                sse = ((x - rec) ** 2).sum(dim=(1, 2))
                sst = ((x - x.mean(dim=(1, 2), keepdim=True)) ** 2).sum(dim=(1, 2))
                r2 = torch.where(sst > 0, 1 - sse / sst.clamp_min(1e-12), torch.nan)  # undefined for blank spectrograms
                r2s.append(r2.cpu().numpy())
        return np.concatenate(r2s)

    train_r2, test_r2 = r2_of(dls["train"]), r2_of(dls[comparison])
    mu = (np.nanmean(train_r2), np.nanmean(test_r2))
    sd = (np.nanstd(train_r2), np.nanstd(test_r2))
    print(f"Train r2: {mu[0]} +- {sd[0]}")
    print(f"{comparison} r2: {mu[1]} +- {sd[1]}")
    return mu, sd, (train_r2, test_r2)


def model_cv_precisions(
    dls: dict,
    n_epochs: int = 100,
    lr: float = 1e-3,
    z_dim: int = 32,
    precisions: Sequence[float] = (1.0, 10.0, 100.0),
    model_path: str = "",
    save_freq: int = 5,
    device: str = "cuda",
    spec_params: Optional[dict] = None,
) -> VAE:
    """
    the counterpart of `train.model_cv.model_cv_lambdas`: trains a VAE for each value of
    the one hyperparameter that matters for the objective, the observation model's precision, and keeps
    the best by validation R^2 of the reconstruction. (ELBOs with different precisions are not
    comparable, so R^2 is used.) Models that have already been trained are loaded and resumed.

    inputs
    -----
        - dls: dictionary of dataloaders with 'train', 'val' and 'test'
        - n_epochs: number of epochs to train each model for
        - lr: learning rate
        - z_dim: latent dimension
        - precisions: values of `model_precision` to try
        - model_path: folder to save all the models, and a csv of the results
        - save_freq: how often (in epochs) to save checkpoints
        - device: device to train on
        - spec_params: the spectrogram parameters the data were made with. Saved with the models

    returns
    -----
        - the best model (by validation R^2), in eval mode
    """
    os.makedirs(model_path, exist_ok=True)
    rows = []
    for precision in precisions:
        print(f"model precision {precision}")
        run_dir = os.path.join(model_path, f"vae_precision_{precision}")
        if glob.glob(os.path.join(run_dir, "checkpoint_*.tar")):
            model, opt, scheduler, start_epoch = load_model(run_dir, device=device)
        else:
            model = VAE(z_dim, precision, device=device, spec_params=spec_params)
            opt = Adam(model.parameters(), lr=lr)
            scheduler = ReduceLROnPlateau(
                opt, factor=0.5, patience=max(n_epochs // 25, 2), min_lr=1e-10
            )
            start_epoch = 0

        if start_epoch < n_epochs:
            train(
                model,
                opt,
                dls,
                scheduler=scheduler,
                nEpochs=n_epochs,
                val_freq=1,
                runDir=run_dir,
                vis_freq=max(n_epochs // 10, 1),
                start_epoch=start_epoch,
                save_freq=save_freq,
            )
            save_model(model, opt, os.path.join(run_dir, f"checkpoint_{n_epochs}.tar"))

        (train_r2, val_r2), *_ = eval_model_error(dls, model, comparison="val")
        rows.append({"precision": precision, "train R2": train_r2, "val R2": val_r2})

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(model_path, "cv_errs.csv"))
    best = float(results.loc[results["val R2"].idxmax(), "precision"])
    print(f"best precision: {best}")

    model, *_ = load_model(os.path.join(model_path, f"vae_precision_{best}"), device=device)
    model.eval()
    return model


def train_model(
    audio_dirs: list,
    seg_dirs: list,
    model_dir: str,
    max_vocs: int = 5000,
    context_len: Optional[float] = None,
    seed: Optional[int] = 1234,
    shuffle_order: bool = False,
    max_jobs: int = 4,
    batch_size: int = 32,
    n_epochs: int = 100,
    save_freq: int = 5,
    device: str = "cuda",
    z_dim: int = 32,
    precisions: Sequence[float] = (10.0,),
    lr: float = 1e-3,
    spec_params: Optional[dict] = None,
    tune_spec_range: bool = True,
) -> VAE:
    """
    train a VAE on audio, with the same call as `train.train_model.train_model`. Audio is taken from
    `audio_dirs` and onsets and offsets from `seg_dirs`. By default each segmented vocalization becomes
    one spectrogram, as is standard for VAEs. To train on the same fixed-length chunks as Ouroboros
    instead, pass `context_len`.

    inputs
    -----
        - audio_dirs: list of folders with audio
        - seg_dirs: list of folders with segmentation files
        - model_dir: where to save model checkpoints and plots
        - max_vocs: max number of spectrograms to train on
        - context_len: if given, train on chunks this long (s), as Ouroboros does; otherwise on whole
            segmented vocalizations
        - seed: random seed
        - shuffle_order: whether to shuffle the audio files when gathering data
        - max_jobs: max number of dataloader workers
        - batch_size: batch size
        - n_epochs: number of passes through the data
        - save_freq: how often (in epochs) to checkpoint
        - device: device to train on. falls back to cpu if cuda is unavailable
        - z_dim: latent dimension
        - precisions: observation precisions to cross-validate over. By default a single value
        - lr: learning rate
        - spec_params: overrides for the spectrogram parameters (see `SPEC_PARAMS`)
        - tune_spec_range: choose the spectrogram intensity range from the data (see
            `estimate_spec_range`) rather than using AVA's fixed defaults

    returns
    -----
        - the best model after cross-validating over `precisions`, in eval mode
    """
    n_cpu = os.cpu_count()
    n_jobs = min(max_jobs, n_cpu if n_cpu is not None else 0)

    assert len(audio_dirs) == len(seg_dirs), "Need the same number of audio dirs as segment dirs!"
    os.makedirs(model_dir, exist_ok=True)

    per_dir = max_vocs // len(audio_dirs)
    audio = []
    sr = None
    for audio_dir, seg_dir in tqdm(
        zip(audio_dirs, seg_dirs), desc="Gathering training data", total=len(audio_dirs)
    ):
        chunks, sr = get_segmented_audio(
            audio_dir,
            seg_dir,
            max_vocs=per_dir,
            context_len=context_len if context_len is not None else 0.3,
            seed=seed,
            training=context_len is not None,
            extend=True,
            shuffle_order=shuffle_order,
        )
        audio += chunks
    if len(audio) == 0:
        raise ValueError("no vocalizations were found")
    print(f"Gathered {len(audio)}/{max_vocs} allowed vocalizations")

    p = resolve_spec_params(audio, sr, spec_params, tune_range=tune_spec_range)
    dataloaders = get_spectrogram_loaders(
        audio,
        sr,
        num_workers=n_jobs,
        batch_size=batch_size,
        train_size=0.6,
        cv=True,
        seed=seed,
        p=p,
    )

    return model_cv_precisions(
        dataloaders,
        n_epochs=n_epochs,
        lr=lr,
        z_dim=z_dim,
        precisions=precisions,
        model_path=model_dir,
        save_freq=save_freq,
        device=device,
        spec_params=p,
    )


if __name__ == "__main__":
    pass
