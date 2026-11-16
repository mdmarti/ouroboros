"""
Budgerigar: how well do spikes predict Ouroboros' latents, and how does that compare with baselines?

Converted from notebooks/budgie_analysis.ipynb (the sliding-window regression, with the
`spike_function_design` helper from cosyne_plots.ipynb), and extended with baseline regressors.

For each vocalization, spike counts (1 ms bins) in a window of the previous `win_length` bins are the
regressors for a time-varying target at the end of that window. `--regressor` picks the target(s):

    ouro      Ouroboros' omega(t) and gamma(t) (needs --model-path)
    acoustic  SAT acoustic features: amplitude, pitch, goodness of pitch, frequency modulation, amplitude
              modulation, entropy (VocalPy; 1 ms frames; NaNs, e.g. pitch in silence, are left out of the fit)
    umap      a UMAP embedding (--umap-dims, default 2) of 1 ms log-spectrogram frames

A LassoLarsCV regression is fit with `k_folds`-fold cross-validation (fit on k-1 folds, scored (R^2) on the
held-out fold), repeated `n_runs` times with fresh random folds, against
two nulls:

    - "local null": regressors are spike trains shuffled in time within each neuron/vocalization
    - "global null": real regressors, but the targets are shuffled across time points
    - "circular null": real regressors, but each vocalization's target is circularly shifted in time by a
      random lag (fresh lags every run). This keeps the target's autocorrelation, and only breaks its
      alignment with the spikes; the other nulls destroy the temporal structure of the spikes / targets

All regressors use the same spike designs and the same cross-validation folds, so their R^2s are directly
comparable. Targets are smoothed with the same Gaussian (`--spike-smooth`).

The data folder holds one `*_cleaned.wav`, one `*.txt` (onset/offset) and one `*.mat` (`spikeTimes`) file.

    python -m analysis_scripts.budgie_spike_regression run --regressor ouro \
        --model-path MODELS/budgies --data-dir DATA/Budgie --save-dir OUT
    python -m analysis_scripts.budgie_spike_regression run --regressor acoustic --data-dir DATA/Budgie --save-dir OUT
    python -m analysis_scripts.budgie_spike_regression run --regressor umap     --data-dir DATA/Budgie --save-dir OUT
    python -m analysis_scripts.budgie_spike_regression compare --save-dir OUT     # all regressors, side by side

Saved under --save-dir:
    designs.pkl                    spike design matrices (shared by all regressors)
    <regressor>/targets.pkl        the targets for each vocalization
    <regressor>/r2s.pkl            train/test R^2 for every run, target and condition
    <regressor>/weights.npz        regression weights for every run
    <regressor>/summary.json       median and IQR of test R^2 (over runs and folds)
    <regressor>/*.svg              weight maps, R^2 vs nulls, example predictions
    comparison.csv/.json/.svg      (from `compare`) every regressor and target together

Differences from the notebook
-----------------------------
    - the gamma global null shuffles gamma (the notebook shuffled omega, and scored the omega global null
      model with the local null spike design); global null R^2s are therefore not comparable to old numbers
    - `--split-by voc` splits train/test by vocalization. The notebook's behavior (`timepoint`, the default)
      puts neighboring time bins from the same call in both train and test, which inflates R^2
    - results and weights are cached per run as npz/pkl instead of pickling sklearn models
    - UMAP is fit once on the frames of all vocalizations (unsupervised), not within each split
"""

import gc
import glob
import json
import os
import pickle

import fire
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import make_interp_spline
from scipy.io import loadmat, wavfile
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LassoLarsCV
from tqdm import tqdm

from analysis_scripts.baselines import (
    check_regressor,
    embed_umap,
    fit_umap,
    log_spectral_frames,
    resample_to,
    sat_time_series,
)
from analysis_scripts.common import format_axis, load_best_model, plt, resolve_model_path
from utils import deriv_approx_dy

SPIKE_BIN_SIZE = 0.001
CONDITIONS = ["real", "local null", "global null", "circular null"]



# ------------------------------------------------------------------ data


def load_recording(data_dir):
    wavpath = glob.glob(os.path.join(data_dir, "*_cleaned.wav"))[0]
    segpath = glob.glob(os.path.join(data_dir, "*.txt"))[0]
    spikepath = glob.glob(os.path.join(data_dir, "*.mat"))[0]
    sr, audio = wavfile.read(wavpath)
    spiketimes = loadmat(spikepath)["spikeTimes"][0]
    onoffs = np.loadtxt(segpath, usecols=(0, 1))
    return sr, audio, spiketimes, np.atleast_2d(onoffs)


def _geometry(onset, offset, sr, L, win_length):
    """sample/bin bookkeeping for one vocalization, with `win_length` bins of context before onset"""
    padding = win_length * SPIKE_BIN_SIZE
    aud_on, aud_off = int(round(onset * sr)), int(round(offset * sr))
    on_pad = min(aud_on, int(round(padding * sr)))
    off_pad = min(L - aud_off, int(round(padding * sr)))
    spike_t = np.arange(onset, offset + SPIKE_BIN_SIZE / 2, SPIKE_BIN_SIZE)
    return aud_on, aud_off, on_pad, off_pad, spike_t


def build_designs(rec, win_length=20, seed=92):
    """
    spike design matrices for every vocalization: (T, n_neurons, win_length) arrays of binned spikes,
    and the same with each neuron's spikes shuffled in time ("nulls"). Independent of the regressor.
    """
    sr, audio, spiketimes, onoffs = rec
    gen = np.random.default_rng(seed=seed)
    L = len(audio)
    voc_spikes, voc_nulls = [], []
    for onset, offset in tqdm(onoffs, desc="building spike designs"):
        _, _, on_pad, off_pad, _ = _geometry(onset, offset, sr, L, win_length)
        spikebins = np.arange(
            onset - on_pad / sr, offset + off_pad / sr + SPIKE_BIN_SIZE / 2, SPIKE_BIN_SIZE
        )
        spike_off_pad = int(round(off_pad / (sr * SPIKE_BIN_SIZE)))
        spikes, null_spikes = [], []
        for s in spiketimes:
            in_window = (s >= (onset - on_pad / sr)) * (s <= (offset + off_pad / sr))
            binned, _ = np.histogram(s[in_window], bins=spikebins)
            spikes.append(binned)
            null_spikes.append(binned[gen.permutation(len(binned))])
        spikes, null_spikes = np.vstack(spikes), np.vstack(null_spikes)
        if spike_off_pad > 0:
            spikes, null_spikes = spikes[:, :-spike_off_pad], null_spikes[:, :-spike_off_pad]
        ons = np.arange(0, spikes.shape[1] - win_length)
        voc_spikes.append(np.stack([spikes[:, o : o + win_length] for o in ons], axis=0))
        voc_nulls.append(np.stack([null_spikes[:, o : o + win_length] for o in ons], axis=0))
    return {"spikes": voc_spikes, "nulls": voc_nulls}


# ------------------------------------------------------------------ targets


def _nan_smooth(v, sigma):
    """gaussian smoothing that ignores NaNs (NaN where there is no valid data nearby)"""
    ok = np.isfinite(v)
    if sigma <= 0 or not ok.any():
        return v
    num = gaussian_filter(np.where(ok, v, 0.0), sigma=sigma)
    den = gaussian_filter(ok.astype(float), sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 1e-3] = np.nan
    out[~ok] = np.nan  # keep missing values missing
    return out


def ouro_targets(rec, designs, model, win_length, model_smooth, spike_smooth, max_len_t):
    """omega(t) and gamma(t) on the spike bins, one array per vocalization"""
    sr, audio, _, onoffs = rec
    dt, L = 1 / sr, len(audio)
    old_smooth_len = model.smooth_len
    model.smooth_len = model_smooth
    bin_func_smooth = int(round(spike_smooth * sr))
    out = {"omega": [], "gamma": []}
    for (onset, offset), design in tqdm(zip(onoffs, designs["spikes"]), total=len(onoffs),
                                        desc="omega/gamma"):
        aud_on, aud_off, on_pad, off_pad, spike_t = _geometry(onset, offset, sr, L, win_length)
        x = audio[aud_on - on_pad : aud_off + off_pad][None, :, None]
        dxdt = deriv_approx_dy(x)
        x = torch.from_numpy(x).to(model.device).to(torch.float32)
        dxdt = torch.from_numpy(dxdt).to(model.device).to(torch.float32)
        with torch.no_grad():
            omega, gamma, *_ = model.get_funcs(x, dxdt, dt, smoothing=False, max_len_t=max_len_t)
        vals = []
        for f in (omega, gamma):
            f = gaussian_filter(f.detach().cpu().numpy().squeeze(), sigma=bin_func_smooth)
            f = f[on_pad:]
            if off_pad > 0:
                f = f[:-off_pad]
            vals.append(make_interp_spline(np.linspace(onset, offset, len(f)), f)(spike_t))
        out["omega"].append(vals[0][: len(design)])
        out["gamma"].append(vals[1][: len(design)])
    model.smooth_len = old_smooth_len
    return out


def acoustic_targets(rec, designs, win_length, spike_smooth):
    """SAT acoustic feature time series on the spike bins, one array per vocalization"""
    sr, audio, _, onoffs = rec
    L = len(audio)
    sigma = spike_smooth / SPIKE_BIN_SIZE
    out = None
    for (onset, offset), design in tqdm(zip(onoffs, designs["spikes"]), total=len(onoffs),
                                        desc="acoustic features"):
        aud_on, aud_off, on_pad, off_pad, spike_t = _geometry(onset, offset, sr, L, win_length)
        x = audio[aud_on - on_pad : aud_off + off_pad].astype(np.float64)
        if np.issubdtype(audio.dtype, np.integer):
            x = x / np.iinfo(audio.dtype).max
        times, series = sat_time_series(x, sr)
        if out is None:
            out = {k: [] for k in series}
        abs_t = onset - on_pad / sr + times
        for k, v in series.items():
            out[k].append(_nan_smooth(resample_to(abs_t, v, spike_t), sigma)[: len(design)])
    return out


def umap_targets(rec, designs, win_length, spike_smooth, umap_dims, seed, max_fit):
    """UMAP embedding of 1 ms log-spectrogram frames, on the spike bins, one array per vocalization"""
    sr, audio, _, onoffs = rec
    L = len(audio)
    sigma = spike_smooth / SPIKE_BIN_SIZE
    frames = []
    for (onset, offset), design in tqdm(zip(onoffs, designs["spikes"]), total=len(onoffs),
                                        desc="spectrogram frames"):
        aud_on, aud_off, on_pad, off_pad, spike_t = _geometry(onset, offset, sr, L, win_length)
        x = audio[aud_on - on_pad : aud_off + off_pad].astype(np.float64)
        if np.issubdtype(audio.dtype, np.integer):
            x = x / np.iinfo(audio.dtype).max
        t, F = log_spectral_frames(x, sr)
        abs_t = onset - on_pad / sr + t
        frames.append(
            np.stack([np.interp(spike_t, abs_t, F[:, j]) for j in range(F.shape[1])], axis=1)[: len(design)]
        )
    all_frames = np.concatenate(frames, axis=0)
    mu, sd = all_frames.mean(0), all_frames.std(0) + 1e-8
    print(f"fitting UMAP to {len(all_frames)} frames...")
    reducer = fit_umap((all_frames - mu) / sd, umap_dims, max_fit=max_fit, seed=seed)
    out = {f"umap_{i + 1}": [] for i in range(umap_dims)}
    for fr in tqdm(frames, desc="UMAP transform"):
        emb = embed_umap((fr - mu) / sd, reducer)
        for i in range(umap_dims):
            out[f"umap_{i + 1}"].append(gaussian_filter(emb[:, i], sigma=sigma) if sigma > 0 else emb[:, i])
    return out


# ------------------------------------------------------------------ regression


def _fit_score(train_x, train_y, test_x, test_y):
    lr = LassoLarsCV().fit(train_x, train_y)
    return lr, lr.score(train_x, train_y), lr.score(test_x, test_y)


def run(
    data_dir,
    save_dir,
    regressor="ouro",
    model_path="",
    umap_dims=2,
    umap_max_fit=20000,
    model_smooth=0.001,
    spike_smooth=0.005,
    win_length=20,
    seed=92,
    k_folds=5,
    n_runs=10,
    split_by="timepoint",
    n_vocs_vis=3,
    max_len_t=4,
    min_shift_frac=0.1,
    device="cuda",
):
    """
    data_dir: folder with the budgie audio, segments and spikes
    save_dir: where everything is saved (see module docstring). Outputs for this regressor go in save_dir/<regressor>
    regressor: ouro, acoustic or umap
    model_path: (ouro only) a trained model: the folder from `model_cv_lambdas`, or a single model's folder.
        This script does not train models
    umap_dims, umap_max_fit: (umap only) embedding dimension, and the max number of frames UMAP is fit on
    k_folds: number of cross-validation folds. Each run is one k-fold CV with a fresh random partition,
        so there are n_runs * k_folds R^2 values per target and condition
    split_by: "timepoint" (random time bins, as in the notebook) or "voc" (whole vocalizations)
    min_shift_frac: circular null: each vocalization's target is rolled by a lag drawn uniformly from
        [min_lag, L - min_lag] bins, with min_lag the larger of min_shift_frac * L and 10 spike-smoothing lengths
        (for vocalizations too short for that, the target is rolled by L // 2)
    """
    check_regressor(regressor)
    assert split_by in ("timepoint", "voc")
    assert k_folds >= 2, "k_folds must be at least 2"
    out_dir = os.path.join(save_dir, regressor)
    os.makedirs(out_dir, exist_ok=True)

    rec = load_recording(data_dir)
    n_vocs_total = len(rec[3])
    gen = np.random.default_rng(seed=seed)
    design_path = os.path.join(save_dir, "designs.pkl")
    if os.path.isfile(design_path):
        with open(design_path, "rb") as f:
            designs = pickle.load(f)
        assert len(designs["spikes"]) == n_vocs_total, "designs.pkl does not match this data; delete it"
    else:
        designs = build_designs(rec, win_length, seed)
        with open(design_path, "wb") as f:
            pickle.dump(designs, f)

    target_path = os.path.join(out_dir, "targets.pkl")
    if os.path.isfile(target_path):
        with open(target_path, "rb") as f:
            targets = pickle.load(f)
    else:
        if regressor == "ouro":
            assert model_path, "--model-path is required for --regressor ouro"
            model = load_best_model(resolve_model_path(model_path), device=device)
            targets = ouro_targets(rec, designs, model, win_length, model_smooth, spike_smooth, max_len_t)
            del model
            torch.cuda.empty_cache()
        elif regressor == "acoustic":
            targets = acoustic_targets(rec, designs, win_length, spike_smooth)
        else:
            targets = umap_targets(rec, designs, win_length, spike_smooth, umap_dims, seed, umap_max_fit)
        with open(target_path, "wb") as f:
            pickle.dump(targets, f)
    gc.collect()

    spike_designs, null_designs = designs["spikes"], designs["nulls"]
    N = spike_designs[0].shape[1]
    X = np.vstack([x.reshape(x.shape[0], -1) for x in spike_designs])
    X_local_null = np.vstack([x.reshape(x.shape[0], -1) for x in null_designs])
    
    Y = {k: np.hstack(v) for k, v in targets.items()}
    voc_id = np.concatenate([np.full(len(x), i) for i, x in enumerate(spike_designs)])
    n_total = len(X)
    names = list(Y)
    if split_by == "voc":
        assert k_folds <= len(spike_designs), "k_folds cannot exceed the number of vocalizations"
    print(f"{regressor}: {len(spike_designs)} vocalizations, {n_total} time bins, {N} neurons, targets {names}")

    # splits and shuffles come from their own generators, so every regressor sees the same splits
    split_gen = np.random.default_rng(seed)
    shuffle_gen = np.random.default_rng(seed + 1)
    shift_gen = np.random.default_rng(seed + 3)
    voc_len = np.array([len(x) for x in spike_designs])
    voc_start = np.concatenate([[0], np.cumsum(voc_len)[:-1]])
    min_lag_smooth = int(np.ceil(10 * spike_smooth / SPIKE_BIN_SIZE))

    r2 = {t: {c: {"train": [], "test": []} for c in CONDITIONS} for t in names}
    weights = {t: [] for t in names}
    last_models = {}
    n_valid = {}

    for run_ in range(1, n_runs + 1):
        print(f"run {run_}/{n_runs}")
        # circular null targets: every vocalization's targets rolled by a random lag (same lag for all
        # targets of a vocalization, fixed across the folds of this run)
        lags = np.zeros(len(spike_designs), dtype=int)
        for v, L in enumerate(voc_len):
            min_lag = max(min_lag_smooth, int(min_shift_frac * L))
            lags[v] = shift_gen.integers(min_lag, L - min_lag + 1) if L > 2 * min_lag else L // 2
        Y_circ = {
            k: np.concatenate([np.roll(y[s : s + L], lag)
                               for s, L, lag in zip(voc_start, voc_len, lags)])
            for k, y in Y.items()
        }

        # a fresh random k-fold partition each run; every time bin is in the test set once per run
        if split_by == "timepoint":
            folds = np.array_split(split_gen.permutation(n_total), k_folds)
        else:
            voc_folds = np.array_split(split_gen.permutation(len(spike_designs)), k_folds)
            folds = [np.flatnonzero(np.isin(voc_id, v)) for v in voc_folds]

        for fold, te in enumerate(folds):
            tr = np.setdiff1d(np.arange(n_total), te)
            for name in names:
                y = Y[name]
                tr_v, te_v = tr[np.isfinite(y[tr])], te[np.isfinite(y[te])]
                n_valid[name] = int(np.isfinite(y).sum())
                if len(tr_v) < 100 or len(te_v) < 100:
                    print(f"  fold {fold + 1}/{k_folds} {name}: too few valid time bins, skipping")
                    for c in CONDITIONS:
                        r2[name][c]["train"].append(np.nan)
                        r2[name][c]["test"].append(np.nan)
                    weights[name].append(np.full((N, win_length), np.nan))
                    continue
                lr, a, b = _fit_score(X[tr_v], y[tr_v], X[te_v], y[te_v])
                r2[name]["real"]["train"].append(a)
                r2[name]["real"]["test"].append(b)
                weights[name].append(lr.coef_.reshape(N, win_length))
                last_models[name] = lr

                _, a, b = _fit_score(X_local_null[tr_v], y[tr_v], X_local_null[te_v], y[te_v])
                r2[name]["local null"]["train"].append(a)
                r2[name]["local null"]["test"].append(b)

                y_tr = y[tr_v][shuffle_gen.permutation(len(tr_v))]
                y_te = y[te_v][shuffle_gen.permutation(len(te_v))]
                _, a, b = _fit_score(X[tr_v], y_tr, X[te_v], y_te)
                r2[name]["global null"]["train"].append(a)
                r2[name]["global null"]["test"].append(b)

                y_c = Y_circ[name]
                tr_c, te_c = tr[np.isfinite(y_c[tr])], te[np.isfinite(y_c[te])]
                _, a, b = _fit_score(X[tr_c], y_c[tr_c], X[te_c], y_c[te_c])
                r2[name]["circular null"]["train"].append(a)
                r2[name]["circular null"]["test"].append(b)
                print(f"  fold {fold + 1}/{k_folds} {name}: test R2 {r2[name]['real']['test'][-1]:.3f} "
                      f"(local {r2[name]['local null']['test'][-1]:.3f}, global null {r2[name]['global null']['test'][-1]:.3f}, "
                      f"circular null {r2[name]['circular null']['test'][-1]:.3f})")
            gc.collect()

    # ---- save results
    weights = {k: np.stack(v) for k, v in weights.items()}
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)
    with open(os.path.join(out_dir, "r2s.pkl"), "wb") as f:
        pickle.dump(r2, f)

    def stats(v):
        v = np.asarray(v, dtype=float)
        if np.all(np.isnan(v)):
            return {"median": None, "iqr": None}
        return {"median": float(np.nanmedian(v)),
                "iqr": [float(x) for x in np.nanpercentile(v, [25, 75])]}

    summary = {
        "regressor": regressor,
        "n_vocalizations": len(spike_designs),
        "n_neurons": N,
        "n_time_bins": int(n_total),
        "n_valid_time_bins": n_valid,
        "split_by": split_by,
        "k_folds": k_folds,
        "n_runs": n_runs,
        "seed": seed,
        "test_r2": {t: {c: stats(r2[t][c]["test"]) for c in CONDITIONS} for t in names},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["test_r2"], indent=2))

    # ---- figures
    mean_w = {k: np.nanmean(v, axis=0) for k, v in weights.items()}
    ncol = min(len(names), 3)
    nrow = int(np.ceil(len(names) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5 * nrow), squeeze=False, sharey=True)
    step = max(win_length // 4, 1)
    for ax, name in zip(axs.ravel(), names):
        g = ax.matshow(mean_w[name], aspect="auto", vmin=-0.01, vmax=0.01, cmap="RdBu_r", origin="lower")
        format_axis(ax, f"{name} weights (mean over runs)", "Lag (ms)", "Neuron number")
        ax.set_xticks(range(0, win_length, step))
        ax.set_xticklabels(range(-win_length, 0, step))
        ax.xaxis.tick_bottom()
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    fig.colorbar(g, ax=axs, label="Weight (a.u.)")
    fig.savefig(os.path.join(out_dir, "average_weights.svg"), transparent=True)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(6, 2.9 * len(names)), 4))
    labels = [f"{t}\n{c}" for t in names for c in CONDITIONS]
    def _clean(v):
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        return v if len(v) else np.array([np.nan])

    ax.boxplot([_clean(r2[t][c]["test"]) for t in names for c in CONDITIONS], tick_labels=labels)
    ax.tick_params(axis="x", labelsize=7)
    format_axis(ax, ylabel=r"Test $R^2$")
    fig.savefig(os.path.join(out_dir, "test_r2_vs_nulls.svg"), transparent=True)
    plt.close(fig)

    vis_gen = np.random.default_rng(seed + 2)
    choices = vis_gen.choice(len(spike_designs), min(n_vocs_vis, len(spike_designs)), replace=False)
    for jj, c in enumerate(choices):
        flat = spike_designs[c].reshape(spike_designs[c].shape[0], -1)
        t_ms = np.arange(len(flat)) * SPIKE_BIN_SIZE * 1000
        fig, axs = plt.subplots(len(names), 1, figsize=(10, 2.6 * len(names)), sharex=True, squeeze=False)
        for ax, name in zip(axs.ravel(), names):
            if name not in last_models:
                continue
            ax.plot(t_ms, targets[name][c][: len(flat)], label=regressor)
            ax.plot(t_ms, last_models[name].predict(flat), label="predicted")
            format_axis(ax, ylabel=name)
        axs.ravel()[-1].set_xlabel("Time (ms)")
        axs.ravel()[0].legend()
        fig.savefig(os.path.join(out_dir, f"example_prediction_{jj}.svg"), transparent=True)
        plt.close(fig)


def compare(save_dir):
    """collects the `summary.json` of every regressor under save_dir into one table and figure"""
    rows = []
    for path in sorted(glob.glob(os.path.join(save_dir, "*", "summary.json"))):
        s = json.load(open(path))
        if "test_r2" not in s:
            continue
        for target, conds in s["test_r2"].items():
            row = {"regressor": s["regressor"], "target": target, "split_by": s["split_by"]}
            for c in CONDITIONS:
                cs = conds.get(c, {"median": None, "iqr": None})  # summaries from before a null was added
                row[f"{c}_median"] = cs["median"]
                row[f"{c}_q25"], row[f"{c}_q75"] = cs["iqr"] if cs["iqr"] else (None, None)
            rows.append(row)
    assert rows, f"no */summary.json under {save_dir}; run `run` first"
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "comparison.csv"), index=False)
    with open(os.path.join(save_dir, "comparison.json"), "w") as f:
        json.dump(rows, f, indent=2)
    if df["split_by"].nunique() > 1:
        print("WARNING: regressors were run with different --split-by; R^2s are not comparable")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(df)), 4))
    x = np.arange(len(df))
    w = 0.8 / len(CONDITIONS)
    for k, c in enumerate(CONDITIONS):
        ax.bar(x + (k - (len(CONDITIONS) - 1) / 2) * w, df[f"{c}_median"].astype(float), w, label=c,
               yerr=[(df[f"{c}_median"] - df[f"{c}_q25"]).astype(float),
                     (df[f"{c}_q75"] - df[f"{c}_median"]).astype(float)], capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n{t}" for r, t in zip(df.regressor, df.target)], fontsize=7)
    ax.legend(frameon=False)
    format_axis(ax, ylabel=r"Test $R^2$ (median, IQR over runs)")
    fig.savefig(os.path.join(save_dir, "comparison.svg"), transparent=True, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fire.Fire({"run": run, "compare": compare})
