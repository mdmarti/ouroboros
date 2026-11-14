"""
Baseline representations to compare against Ouroboros' omega/gamma (plan item A2).

Every analysis script that has a `--regressor` option uses these:

    ouro      Ouroboros' omega and gamma (the representation under test)
    acoustic  traditional acoustic features from VocalPy's SAT implementation: amplitude, pitch, goodness of
              pitch, frequency modulation, amplitude modulation and Wiener entropy. As time series
              (one value per ms) for the budgie spike regression, and summarized over each call
              (mean, sd, plus duration; `comparisons.acoustic_features`) for the marmoset decoding
    umap      a UMAP embedding of the spectrogram. Per 1 ms time frame (a log spectrogram frame) for the
              budgie regression, per call (an AVA-style spectrogram image; `comparisons.vae`) for decoding

UMAP is unsupervised and is fit once on all of the data, as is common for UMAP-based analyses (and as
the PCA step is for Ouroboros in the marmoset decoding), not within each train/test split.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import vocalpy as voc
from scipy.signal import stft

from comparisons.acoustic_features import SAT_FEATURES

REGRESSORS = ("ouro", "acoustic", "umap")


def check_regressor(regressor: str):
    if regressor not in REGRESSORS:
        raise ValueError(f"--regressor must be one of {REGRESSORS}, got {regressor!r}")


def sat_time_series(
    audio: np.ndarray, sr: float, hop_s: float = 0.001, n_fft: int = 512
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    SAT feature time series for one signal. Returns the frame times (s, from the start of `audio`)
    and a dict of feature name -> values (NaN where undefined, e.g. pitch in silence).
    """
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    hop = max(int(round(hop_s * sr)), 1)
    ds = voc.feature.sat(voc.Sound(data=audio, samplerate=int(round(sr))), n_fft=n_fft, hop_length=hop).data
    times = ds["time"].values
    return times, {f: ds[f].values.reshape(-1) for f in SAT_FEATURES}


def resample_to(times: np.ndarray, values: np.ndarray, new_times: np.ndarray) -> np.ndarray:
    """linear interpolation of `values` onto `new_times`, ignoring NaNs; all-NaN in, all-NaN out"""
    ok = np.isfinite(values)
    if ok.sum() < 2:
        return np.full(len(new_times), np.nan)
    return np.interp(new_times, times[ok], values[ok], left=np.nan, right=np.nan)


def log_spectral_frames(
    audio: np.ndarray,
    sr: float,
    hop_s: float = 0.001,
    win_s: float = 0.008,
    n_bins: int = 64,
    min_freq: float = 300.0,
    max_freq: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    log-magnitude spectrogram frames of one signal, pooled into `n_bins` linearly spaced bands between
    `min_freq` and `max_freq` (default: Nyquist). Returns the frame times (s) and an
    (n_frames, n_bins) array.
    """
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    nperseg = max(int(round(win_s * sr)), 16)
    hop = max(int(round(hop_s * sr)), 1)
    f, t, Z = stft(audio, fs=sr, nperseg=nperseg, noverlap=nperseg - hop, boundary="even", padded=False)
    mag = np.log(np.abs(Z) + 1e-8)
    max_freq = sr / 2 if max_freq is None else min(max_freq, sr / 2)
    edges = np.linspace(min_freq, max_freq, n_bins + 1)
    out = np.empty((len(t), n_bins))
    for i in range(n_bins):
        band = (f >= edges[i]) & (f < edges[i + 1] if i < n_bins - 1 else f <= edges[i + 1])
        out[:, i] = mag[band].mean(axis=0) if band.any() else mag.min()
    return t, out


def fit_umap(
    X: np.ndarray,
    n_components: int = 2,
    max_fit: int = 20000,
    seed: int = 0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
):
    """
    fits UMAP to (a random subsample of) the rows of X, and returns an object with `transform` (the
    embedding of all of X). Uses at most `max_fit` rows for fitting, since UMAP is slow on very many rows.
    """
    import umap  # imported here: slow to import and only needed for this baseline

    X = np.asarray(X, dtype=np.float32)
    rng = np.random.default_rng(seed)
    fit_rows = np.arange(len(X)) if len(X) <= max_fit else rng.choice(len(X), max_fit, replace=False)
    reducer = umap.UMAP(
        n_components=n_components, n_neighbors=min(n_neighbors, max(len(fit_rows) - 1, 2)),
        min_dist=min_dist, random_state=seed,
    )
    reducer.fit(X[fit_rows])
    return reducer


def embed_umap(X: np.ndarray, reducer, batch: int = 50000) -> np.ndarray:
    """UMAP embedding of every row of X, in batches"""
    X = np.asarray(X, dtype=np.float32)
    return np.concatenate([reducer.transform(X[i : i + batch]) for i in range(0, len(X), batch)], axis=0)
