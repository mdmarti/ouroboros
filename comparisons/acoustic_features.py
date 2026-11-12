"""
Traditional acoustic features, computed with VocalPy, for comparison against Ouroboros.

Like `comparisons.vae`, this mirrors the main code: `get_acoustic_features` takes the same
`audio_dirs`/`seg_dirs` inputs as `train.train_model.train_model`, and `features_from_audio`
plays the role of `Ouroboros.get_funcs` for audio you have already loaded.

Two feature sets are available, both from `vocalpy.feature`:

    - "sat": the Sound Analysis Toolbox features used by SAP/SAT: amplitude, pitch, goodness
        of pitch, frequency modulation, amplitude modulation, and Wiener entropy. These
        are time series, so each vocalization is summarized by statistics over time
        (by default, mean and standard deviation), plus its duration. Fast.
    - "biosound": the 21 predefined acoustic features of soundsig/BioSound (temporal and
        spectral moments, quartiles, fundamental frequency statistics). These are one
        number per vocalization. Slow (several seconds per vocalization), and it needs
        audio sampled above 20 kHz (soundsig computes spectral shape below 10 kHz).

Long recordings are handled one audio file at a time: files are memory-mapped, only the
segments that are needed are read, and the audio is released once its features are computed, so that
memory use is set by the largest file's segments rather than the whole dataset. `iter_acoustic_features`
exposes this directly, and `cache_dir` saves the features of each file as they are finished, so a long
run can be interrupted and resumed.

Features that are undefined (for example, from a vocalization too short to analyze) are NaN. Use
`make_feature_pipeline` to impute, standardize and, if wanted, reduce the dimensionality of the features
(e.g. to match the dimensionality of an Ouroboros or VAE latent space).
"""

import hashlib
import json
import os
import warnings
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import vocalpy as voc
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data.load_data import list_audio_files, load_file_segments

SAT_FEATURES = (
    "amplitude",
    "pitch",
    "goodness_of_pitch",
    "frequency_modulation",
    "amplitude_modulation",
    "entropy",
)
"""time series computed by `vocalpy.feature.sat`"""

BIOSOUND_FEATURES = (
    "mean_t", "std_t", "skew_t", "kurtosis_t", "entropy_t", "max_amp",
    "mean_s", "std_s", "skew_s", "kurtosis_s", "entropy_s",
    "q1", "q2", "q3",
    "mean_f0", "mean_sal", "second_v", "pk2", "max_fund", "min_fund", "cv_fund",
)  # fmt: skip
"""features computed by `vocalpy.feature.biosound`"""

SUMMARIES = {
    "mean": np.nanmean,
    "std": np.nanstd,
    "min": np.nanmin,
    "max": np.nanmax,
    "median": np.nanmedian,
}
"""ways of summarizing a feature time series over a vocalization"""

METHODS = ("sat", "biosound")
BIOSOUND_MIN_FS = 20000  # soundsig looks for spectral shape below 10 kHz, so needs a higher Nyquist frequency


def feature_names(
    method: str = "sat",
    summaries: Sequence[str] = ("mean", "std"),
    include_duration: bool = True,
) -> list:
    """
    names of the features returned by `extract_features`, in order

    inputs
    -----
        - method: 'sat' or 'biosound'
        - summaries: summary statistics used for the time series ('sat' only)
        - include_duration: whether the duration of the vocalization is included
    """
    if method == "sat":
        _check_summaries(summaries)
        names = [f"{feat}_{s}" for feat in SAT_FEATURES for s in summaries]
    elif method == "biosound":
        names = list(BIOSOUND_FEATURES)
    else:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if include_duration:
        names.append("duration")
    return names


def _check_summaries(summaries: Sequence[str]):
    unknown = [s for s in summaries if s not in SUMMARIES]
    if unknown:
        raise ValueError(f"unknown summaries {unknown}; choose from {list(SUMMARIES)}")
    if len(summaries) == 0:
        raise ValueError("need at least one summary statistic")


def _summarize(values: np.ndarray, how: str) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if np.all(np.isnan(values)):
        return np.nan
    return float(SUMMARIES[how](values))


def extract_features(
    audio: np.ndarray,
    fs: float,
    method: str = "sat",
    summaries: Sequence[str] = ("mean", "std"),
    include_duration: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, list]:
    """
    acoustic features of a single vocalization, using VocalPy

    inputs
    -----
        - audio: the vocalization. any shape that flattens to a single channel of samples. Must be finite
        - fs: sampling rate, in Hz
        - method: 'sat' or 'biosound' (see the module docstring)
        - summaries: for 'sat', the statistics used to summarize each time series over the vocalization.
            Any of 'mean', 'std', 'min', 'max', 'median'. Statistics ignore NaNs
        - include_duration: append the vocalization's duration (s) as a feature
        - kwargs: passed to the VocalPy feature function (e.g. `n_fft`, `hop_length`, `min_freq` for 'sat')

    returns
    -----
        - features: 1-D array of features (NaN where undefined)
        - names: the name of each feature
    """
    names = feature_names(method, summaries, include_duration)
    if method == "biosound" and fs <= BIOSOUND_MIN_FS:
        raise ValueError(
            f"biosound features need a sampling rate above {BIOSOUND_MIN_FS} Hz, got {fs}"
        )
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(audio)):
        raise ValueError("audio contains NaN or infinite values")

    sound = voc.Sound(data=audio, samplerate=int(round(fs)))

    if method == "sat":
        ds = voc.feature.sat(sound, **kwargs).data
        values = [
            _summarize(ds[feat].values, s) for feat in SAT_FEATURES for s in summaries
        ]
    else:
        ds = voc.feature.biosound(sound, **kwargs).data
        values = [_summarize(ds[feat].values, "mean") for feat in BIOSOUND_FEATURES]

    if include_duration:
        values.append(len(audio) / fs)
    return np.array(values, dtype=np.float64), names


def features_from_audio(
    audio: Sequence[np.ndarray],
    fs: float,
    method: str = "sat",
    summaries: Sequence[str] = ("mean", "std"),
    include_duration: bool = True,
    n_jobs: int = 1,
    verbose: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, list]:
    """
    acoustic features for a list of vocalizations. A vocalization that cannot be analyzed gives a row
    of NaNs and a warning, rather than stopping the whole batch

    inputs
    -----
        - audio: list of vocalizations
        - fs: sampling rate
        - method, summaries, include_duration, kwargs: see `extract_features`
        - n_jobs: number of parallel jobs (`joblib`). BioSound is slow enough that this is worth it
        - verbose: show a progress bar (serial processing only)

    returns
    -----
        - features: N x n_features array
        - names: feature names
    """
    names = feature_names(method, summaries, include_duration)

    def one(a):
        try:
            return extract_features(
                a, fs, method, summaries, include_duration, **kwargs
            )[0]
        except Exception as e:  # keep going on a bad vocalization, but say so
            warnings.warn(f"could not extract features ({type(e).__name__}: {e})")
            return np.full(len(names), np.nan)

    if n_jobs == 1:
        rows = [one(a) for a in tqdm(audio, disable=not verbose, desc="extracting features")]
    else:
        rows = Parallel(n_jobs=n_jobs)(delayed(one)(a) for a in audio)

    if len(rows) == 0:
        return np.zeros((0, len(names))), names
    return np.stack(rows, axis=0), names


def _file_stats(path: str) -> list:
    """size and modification time of a file, to notice when a cached result is out of date"""
    try:
        st = os.stat(path)
        return [st.st_size, st.st_mtime_ns]
    except OSError:
        return []


def _cache_path(cache_dir: str, audio_file: str) -> str:
    key = hashlib.sha1(os.path.abspath(audio_file).encode()).hexdigest()[:10]
    return os.path.join(cache_dir, f"{os.path.basename(audio_file)}.{key}.npz")


def _cache_meta(audio_file, seg_file, settings: dict) -> str:
    return json.dumps(
        {
            **settings,
            "audio": _file_stats(audio_file),
            "segments": _file_stats(seg_file),
        },
        sort_keys=True,
        default=repr,
    )


def _load_cache(path: str, meta: str):
    """cached (features, sr) if there is a cache file made with exactly these settings and files"""
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as d:
            if str(d["meta"]) != meta:
                return None
            return d["features"], float(d["sr"])
    except (OSError, ValueError, KeyError):  # unreadable or incomplete: recompute
        return None


def _save_cache(path: str, meta: str, features: np.ndarray, sr: float):
    # write to a temporary file first, so that an interrupted run never leaves a half-written cache
    tmp = path + ".tmp.npz"
    np.savez(tmp, features=features, sr=sr, meta=meta)
    os.replace(tmp, path)


def iter_acoustic_features(
    audio_dir: str,
    seg_dir: str,
    max_vocs: Optional[int] = None,
    context_len: Optional[float] = None,
    seed: Optional[int] = 1234,
    shuffle_order: bool = False,
    n_jobs: int = 1,
    method: str = "sat",
    summaries: Sequence[str] = ("mean", "std"),
    include_duration: bool = True,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Iterator[Tuple[str, np.ndarray, float]]:
    """
    compute acoustic features one audio file at a time, so that only one file's segments are ever in
    memory. This is the streaming form of `get_acoustic_features` for a single directory. Features are
    yielded as soon as each file is done, so you can also stop early or write them out as you go.

    inputs
    -----
        - audio_dir, seg_dir: folder of audio files and folder of segmentation files
        - max_vocs: stop after this many rows in total (the last file's rows are cut to fit). None for no limit
        - context_len: if given, use fixed-length chunks of this length (s) instead of whole vocalizations
        - seed, shuffle_order: as in `data.load_data.get_segmented_audio`
        - n_jobs: parallel jobs used within each file
        - method, summaries, include_duration, kwargs: see `extract_features`
        - cache_dir: if given, the features of each finished file are saved here, and used instead of recomputing
            when the same settings and unchanged files are seen again. The cache is checked before the audio is
            read, so resumed runs skip finished files entirely

    yields
    -----
        - audio_file: path of the audio file
        - features: n_segments x n_features array for that file
        - sr: sampling rate of that file
    """
    feature_names(method, summaries, include_duration)  # validate the options now, not at the first next()
    return _iter_acoustic_features(
        audio_dir, seg_dir, max_vocs, context_len, seed, shuffle_order, n_jobs,
        method, summaries, include_duration, cache_dir, kwargs,
    )  # fmt: skip


def _iter_acoustic_features(
    audio_dir, seg_dir, max_vocs, context_len, seed, shuffle_order, n_jobs,
    method, summaries, include_duration, cache_dir, kwargs,
):  # fmt: skip
    audio_files, seg_files = list_audio_files(
        audio_dir, seg_dir, ".wav", seed, shuffle_order
    )
    training = context_len is not None
    settings = {
        "method": method,
        "summaries": list(summaries) if method == "sat" else None,
        "include_duration": include_duration,
        "kwargs": kwargs,
        "context_len": context_len,
    }
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    remaining = max_vocs
    for audio_file, seg_file in zip(audio_files, seg_files):
        if remaining is not None and remaining <= 0:
            return
        if not (os.path.isfile(audio_file) and os.path.isfile(seg_file)):
            continue

        cached, path, meta = None, None, None
        if cache_dir is not None:
            path = _cache_path(cache_dir, audio_file)
            meta = _cache_meta(audio_file, seg_file, settings)
            cached = _load_cache(path, meta)  # before reading any audio

        if cached is not None:
            features, sr = cached
            if remaining is not None:
                features = features[:remaining]
        else:
            segments, sr = load_file_segments(
                audio_file,
                seg_file,
                context_len=context_len if training else 0.3,
                training=training,
                extend=True,
            )
            if len(segments) == 0:
                continue
            truncated = remaining is not None and len(segments) > remaining
            if truncated:
                segments = segments[:remaining]
            features, _ = features_from_audio(
                segments, sr, method=method, summaries=summaries,
                include_duration=include_duration, n_jobs=n_jobs, **kwargs,
            )  # fmt: skip
            del segments  # release this file's audio before moving on
            if path is not None and not truncated:  # only complete files are cached
                _save_cache(path, meta, features, sr)

        if len(features) == 0:
            continue
        if remaining is not None:
            remaining -= len(features)
        yield audio_file, features, sr


def get_acoustic_features(
    audio_dirs: list,
    seg_dirs: list,
    max_vocs: int = 5000,
    context_len: Optional[float] = None,
    seed: Optional[int] = 1234,
    shuffle_order: bool = False,
    max_jobs: int = 4,
    method: str = "sat",
    summaries: Sequence[str] = ("mean", "std"),
    include_duration: bool = True,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[np.ndarray, list, float]:
    """
    gather vocalizations and compute acoustic features for them, with the same data arguments as
    `train.train_model.train_model` and `comparisons.vae.train_model`, so the three can be run
    on identical data. Audio is taken from `audio_dirs` and onsets and offsets from `seg_dirs`. Each
    segmented vocalization gives one row, or, if `context_len` is given, each
    fixed-length chunk does (the chunks Ouroboros is trained on).

    Audio is processed one file at a time (see `iter_acoustic_features`), so memory use does not grow with
    the number or length of the recordings; only the (small) feature table is kept.

    inputs
    -----
        - audio_dirs: list of folders with audio
        - seg_dirs: list of folders with segmentation files
        - max_vocs: max number of rows, split evenly across the directories
        - context_len: if given, use chunks of this length (s) instead of whole vocalizations
        - seed: random seed (used when shuffling files)
        - shuffle_order: whether to shuffle the audio files when gathering data
        - max_jobs: max number of parallel jobs
        - method, summaries, include_duration, kwargs: see `extract_features`
        - cache_dir: save each file's features here, and reuse them on later runs (see `iter_acoustic_features`)

    returns
    -----
        - features: N x n_features array
        - names: feature names
        - sr: sampling rate of the audio
    """
    assert len(audio_dirs) == len(seg_dirs), "Need the same number of audio dirs as segment dirs!"

    n_cpu = os.cpu_count()
    n_jobs = max(1, min(max_jobs, n_cpu if n_cpu is not None else 1))
    names = feature_names(method, summaries, include_duration)

    per_dir = max_vocs // len(audio_dirs)
    rows, sr = [], None
    for audio_dir, seg_dir in zip(audio_dirs, seg_dirs):
        for _, features, file_sr in tqdm(
            iter_acoustic_features(
                audio_dir, seg_dir, max_vocs=per_dir, context_len=context_len, seed=seed,
                shuffle_order=shuffle_order, n_jobs=n_jobs, method=method, summaries=summaries,
                include_duration=include_duration, cache_dir=cache_dir, **kwargs,
            ),
            desc=f"Extracting features from {audio_dir}",
        ):  # fmt: skip
            rows.append(features)
            sr = file_sr
    if len(rows) == 0:
        raise ValueError("no vocalizations were found")
    features = np.concatenate(rows, axis=0)
    print(f"Gathered {len(features)}/{max_vocs} allowed vocalizations")
    return features, names, sr


def make_feature_pipeline(n_components: Optional[int] = None) -> Pipeline:
    """
    a scikit-learn pipeline that prepares acoustic features for analysis: fill NaNs
    with the column median, z-score each feature, and optionally project onto the first
    `n_components` principal components (for example, to match the dimensionality of a latent
    space). Fit it on the training data only, then `transform` held-out data

    inputs
    -----
        - n_components: number of principal components to keep. None to skip PCA

    returns
    -----
        - an unfitted `sklearn.pipeline.Pipeline`
    """
    steps = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ]
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components)))
    return Pipeline(steps)


if __name__ == "__main__":
    pass
