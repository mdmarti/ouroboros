"""
Marmosets: call-type spectra of omega/gamma, and decoding of caller identity from different representations.

Converted from notebooks/marmoset_plots.ipynb, with baseline representations added. Commands, each caching
its output so later steps can be re-run quickly:

    embed     gather vocalizations for every marmoset and (if --model-path is given) compute omega and gamma
              with a trained model              -> SAVE/<marmoset>_embeddings.pkl
    spectra   smoothed log-|FFT| of omega and gamma per vocalization on a common frequency grid
                                                -> SAVE/all_{omega,gamma}_ffts.pkl, mean spectra plots
    features  (acoustic and umap only) per-call baseline features
                                                -> SAVE/features_<regressor>.pkl
    decode    per call type, decode which marmoset made the call with a multinomial logistic regression
              from the representation chosen by --regressor
                                                -> SAVE/decoding_summary_<regressor>.json, confusion matrices
    compare   every regressor's decoding results side by side -> SAVE/decoding_comparison.csv/.svg

--regressor chooses the representation that is decoded from (see `analysis_scripts/baselines.py`):

    ouro      PCA (--n-pcs components each) of the omega and gamma spectra of that call type
    acoustic  SAT acoustic feature summaries of each call (mean and sd of amplitude, pitch, goodness of pitch,
              FM, AM, entropy, plus duration); imputed, z-scored and, with --acoustic-pcs, reduced by PCA
    umap      UMAP (--umap-dims components) of each call's AVA-style spectrogram image

    python -m analysis_scripts.marmoset_decoding embed    --model-path M --marmo-dir DATA --save-dir OUT
    python -m analysis_scripts.marmoset_decoding spectra  --save-dir OUT
    python -m analysis_scripts.marmoset_decoding features --regressor acoustic --save-dir OUT
    python -m analysis_scripts.marmoset_decoding decode   --regressor ouro --save-dir OUT
    python -m analysis_scripts.marmoset_decoding compare  --save-dir OUT

Expected data layout: DATA/<marmoset>/vocs/s*/wavfiles/ holds segment files `<tag>.txt` and
`double_denoised/<tag>_cleaned.wav`, where the final `_`-separated piece of <tag> is the call type.
`embed` without --model-path saves just the audio, which is all the baselines need.

Differences from the notebook
-----------------------------
    - call types come from each audio file's own name (the notebook paired glob order with segments,
      which is wrong when a file has several segments or a file has none)
    - decoding also fits a label-shuffled null and reports its accuracy (the notebook built shuffled
      labels but never used them), and can repeat the random train/test split (`--n-repeats`)
    - the 3d scatter of gamma used omega's third PC (typo); fixed
    - PCA uses sklearn's randomized solver instead of dask
For all regressors, the unsupervised dimensionality reduction (PCA for ouro, UMAP for umap) is fit on all calls of a type,
test calls included, as in the notebook; the scaler and classifier see only training calls. Calls from one
session can land in both train and test.
"""

import gc
import glob
import json
import os
import pickle

import fire
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from analysis_scripts.baselines import REGRESSORS, check_regressor, embed_umap, fit_umap
from analysis_scripts.common import (
    embed_vocalizations, format_axis, load_best_model, plt, resolve_model_path,
)
from comparisons.acoustic_features import features_from_audio, make_feature_pipeline
from data.load_data import iter_segmented_audio

SKIP_KEYS = ("sr", "freqs")


def _marmoset_dirs(marmo_dir):
    return sorted(d for d in glob.glob(os.path.join(marmo_dir, "[a-z]*")) if os.path.isdir(d))


def _embedding_files(save_dir):
    files = sorted(glob.glob(os.path.join(save_dir, "*_embeddings.pkl")))
    assert files, f"no *_embeddings.pkl in {save_dir}; run `embed` first"
    return files


def _name(path):
    return os.path.basename(path).replace("_embeddings.pkl", "")


# ------------------------------------------------------------------ embed


def embed(
    marmo_dir,
    save_dir,
    model_path="",
    max_vocs_per_session=100,
    padding=0.1,
    smooth_len_s=0.025,
    max_len_t=8,
    seed=1900,
    device="cuda",
):
    """
    embeds up to `max_vocs_per_session` vocalizations per session for each marmoset. Without `model_path` (a trained model: the folder from `model_cv_lambdas`, or a single
    model's folder; this script does not train models),
    only the audio, call types and sample rate are saved (enough for the baselines).
    """
    os.makedirs(save_dir, exist_ok=True)
    model = load_best_model(resolve_model_path(model_path), device=device) if model_path else None

    for m in _marmoset_dirs(marmo_dir):
        name = os.path.basename(m)
        save_path = os.path.join(save_dir, name + "_embeddings.pkl")
        existing = None
        if os.path.isfile(save_path):
            with open(save_path, "rb") as f:
                existing = pickle.load(f)
            if model is None or "omega" in existing:
                print(f"{name}: already done, skipping")
                continue

        if existing is not None:  # have the audio, just need the model's output
            vocs, voctypes, sessions, sr = (existing[k] for k in ("audio", "voctypes", "sessions", "sr"))
        else:
            seg_dirs = glob.glob(os.path.join(m, "vocs", "s*", "wavfiles"))
            vocs, voctypes, sessions = [], [], []
            sr = None
            for seg_dir in tqdm(seg_dirs, desc=f"{name}: gathering vocs"):
                aud_dir = os.path.join(seg_dir, "double_denoised")
                if not os.path.isdir(aud_dir):
                    continue
                n_session = 0
                for audio_file, segs, sr in iter_segmented_audio(
                    aud_dir, seg_dir, audio_id="_cleaned.wav", seed=seed,
                    training=False, padding=padding, shuffle_order=False,
                ):
                    call = os.path.basename(audio_file).split("_cleaned.wav")[0].split("_")[-1]
                    for s in segs:
                        if n_session >= max_vocs_per_session:
                            break
                        vocs.append(s)
                        voctypes.append(call)
                        sessions.append(seg_dir)
                        n_session += 1
                    if n_session >= max_vocs_per_session:
                        break
        if len(vocs) == 0:
            continue
        print(f"{name}: {len(vocs)} vocalizations")
        data = {"audio": vocs, "voctypes": voctypes, "sessions": sessions, "sr": sr}
        if model is not None:
            data["omega"], data["gamma"] = embed_vocalizations(
                model, vocs, sr, smooth_len_s, max_len_t, desc=f"{name}: embedding"
            )
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        del vocs, data
        gc.collect()


# ------------------------------------------------------------------ spectra (ouro)


def get_fft(data, sr, smooth_len=5):
    """frequencies and moving-average smoothed log-|rfft| of a signal"""
    freqs = np.fft.rfftfreq(len(data), d=1 / sr)
    logabs = np.log(np.abs(np.fft.rfft(data)))
    cumsum = np.cumsum(np.hstack([[logabs[0]] * smooth_len, logabs]))
    smoothed = (cumsum[smooth_len:] - cumsum[:-smooth_len]) / smooth_len
    return freqs, smoothed


def _fft_on_grid(signal, sr, freq_ax, min_freq, smooth_len):
    freqs, fft = get_fft(signal, sr, smooth_len)
    pos = freqs >= min_freq
    return make_interp_spline(freqs[pos], fft[pos])(freq_ax)


def spectra(save_dir, min_freq=5, smooth_len=5, plot=True):
    """spectra of omega and gamma, per marmoset and call type, from the `embed` outputs"""
    ffts = {"omega": {}, "gamma": {}}
    sr = None
    for path in _embedding_files(save_dir):
        name = _name(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        assert "omega" in data, f"{name} has no omega/gamma; re-run `embed` with --model-path"
        sr = data["sr"]
        freq_ax = np.arange(min_freq, int(sr // 2))
        voctypes = np.array(data["voctypes"])
        for key in ("omega", "gamma"):
            ffts[key][name] = {}
        for call in tqdm(np.unique(voctypes), desc=name):
            idx = np.flatnonzero(voctypes == call)
            for key in ("omega", "gamma"):
                ffts[key][name][call] = np.stack(
                    [_fft_on_grid(data[key][i], sr, freq_ax, min_freq, smooth_len) for i in idx]
                )
        del data
        gc.collect()

    for key in ("omega", "gamma"):
        ffts[key]["sr"] = sr
        ffts[key]["freqs"] = freq_ax
        with open(os.path.join(save_dir, f"all_{key}_ffts.pkl"), "wb") as f:
            pickle.dump(ffts[key], f)

    if plot:
        plot_dir = os.path.join(save_dir, "figs")
        os.makedirs(plot_dir, exist_ok=True)
        calls = np.unique(np.hstack([list(ffts["omega"][m]) for m in ffts["omega"] if m not in SKIP_KEYS]))
        cmap = plt.get_cmap("tab20")
        for marmo in (k for k in ffts["omega"] if k not in SKIP_KEYS):
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            for ii, call in enumerate(calls):
                if call not in ffts["omega"][marmo]:
                    continue
                for ax, key in zip(axs, ("omega", "gamma")):
                    x = ffts[key][marmo][call]
                    mu, sd = np.nanmean(x, 0), np.nanstd(x, 0)
                    ax.plot(freq_ax, mu, label=call, color=cmap(ii))
                    ax.fill_between(freq_ax, mu - sd, mu + sd, alpha=0.25, color=cmap(ii))
            format_axis(axs[0], "Omega", "Freq (Hz)", "log power")
            format_axis(axs[1], "Gamma", "Freq (Hz)")
            axs[0].legend(frameon=False)
            fig.suptitle(marmo)
            fig.savefig(os.path.join(plot_dir, f"all_vocs_ffts_{marmo}.svg"), transparent=True)
            plt.close(fig)


# ------------------------------------------------------------------ baseline features


def features(
    save_dir,
    regressor="acoustic",
    n_jobs=4,
    spec_sample_per_marmoset=20,
):
    """
    per-call baseline features, from the audio saved by `embed`.
      acoustic: SAT summaries (N x 13) per call
      umap:     flattened AVA-style spectrograms (N x 16384) per call; UMAP itself is fit in `decode`,
                separately for each call type
    saved as {marmoset: {call type: array}} with the sample rate and feature names.
    """
    assert regressor in ("acoustic", "umap"), "`features` is only needed for the acoustic and umap regressors"
    from comparisons.acoustic_features import feature_names

    files = _embedding_files(save_dir)
    spec_p = None
    if regressor == "umap":
        from comparisons.vae import resolve_spec_params

        longest, sample, sr = 0, [], None
        for path in files:  # first pass: window length and intensity range, from the data
            with open(path, "rb") as f:
                d = pickle.load(f)
            sr = d["sr"]
            longest = max(longest, max(len(a.reshape(-1)) for a in d["audio"]))
            sample += [d["audio"][i] for i in
                       np.random.default_rng(0).choice(len(d["audio"]),
                                                       min(spec_sample_per_marmoset, len(d["audio"])),
                                                       replace=False)]
            del d
        spec_p = resolve_spec_params(sample + [np.zeros(longest)], sr, tune_range=True)
        print(f"spectrogram window {spec_p['max_dur']:.2f}s, range {spec_p['spec_min_val']:.2f}-{spec_p['spec_max_val']:.2f}")

    out = {}
    for path in files:
        name = _name(path)
        with open(path, "rb") as f:
            d = pickle.load(f)
        sr = d["sr"]
        voctypes = np.array(d["voctypes"])
        out[name] = {}
        for call in np.unique(voctypes):
            idx = np.flatnonzero(voctypes == call)
            audio = [d["audio"][i].reshape(-1) for i in idx]
            if regressor == "acoustic":
                X, _ = features_from_audio(audio, sr, method="sat", n_jobs=n_jobs)
            else:
                from comparisons.vae import make_spectrogram

                X = np.stack([make_spectrogram(a, sr, spec_p).reshape(-1) for a in
                              tqdm(audio, desc=f"{name} {call}")])
            out[name][call] = X
        print(f"{name}: {sum(len(v) for v in out[name].values())} calls")
        del d
        gc.collect()
    out["sr"] = sr
    out["feature_names"] = feature_names("sat") if regressor == "acoustic" else None
    with open(os.path.join(save_dir, f"features_{regressor}.pkl"), "wb") as f:
        pickle.dump(out, f)


# ------------------------------------------------------------------ decoding


def _load_representation(save_dir, regressor):
    """per-marmoset, per-call arrays for the regressor, and the list of marmosets"""
    if regressor == "ouro":
        with open(os.path.join(save_dir, "all_omega_ffts.pkl"), "rb") as f:
            o = pickle.load(f)
        with open(os.path.join(save_dir, "all_gamma_ffts.pkl"), "rb") as f:
            g = pickle.load(f)
        marmos = [k for k in o if k not in SKIP_KEYS]
        return {"omega": o, "gamma": g}, marmos
    with open(os.path.join(save_dir, f"features_{regressor}.pkl"), "rb") as f:
        feats = pickle.load(f)
    marmos = [k for k in feats if k not in SKIP_KEYS + ("feature_names",)]
    return {"features": feats}, marmos


def _call_data(regressor, rep, owners, call, n_pcs, umap_dims, umap_max_fit, seed):
    """
    the (n_calls x n_dims) representation of every call of one type, across `owners`, the marmoset label
    of each call, and the processing to apply within the train split (fit on train only)
    """
    if regressor == "ouro":
        projs = {}
        for key in ("omega", "gamma"):
            all_x = np.concatenate([rep[key][m][call] for m in owners], axis=0)
            ok = np.isfinite(all_x).all(1)
            pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=0).fit(all_x[ok])
            projs[key] = np.concatenate([pca.transform(np.nan_to_num(rep[key][m][call])) for m in owners])
        data = np.concatenate([projs["omega"], projs["gamma"]], axis=-1)
        labels = np.concatenate([np.full(len(rep["omega"][m][call]), i) for i, m in enumerate(owners)])
        return data, labels, projs, "scale"
    feats = rep["features"]
    X = np.concatenate([feats[m][call] for m in owners], axis=0)
    labels = np.concatenate([np.full(len(feats[m][call]), i) for i, m in enumerate(owners)])
    if regressor == "acoustic":
        return X, labels, None, "acoustic"
    reducer = fit_umap(X, umap_dims, max_fit=umap_max_fit, seed=seed)
    return embed_umap(X, reducer), labels, None, "scale"


def decode(
    save_dir,
    regressor="ouro",
    n_pcs=10,
    acoustic_pcs=None,
    umap_dims=10,
    umap_max_fit=20000,
    min_calls=50,
    train_prop=0.75,
    n_repeats=1,
    seed=12345,
    plot=True,
):
    """
    For each call type, decode caller identity from the `regressor` representation (see module docstring).
    Marmosets with fewer than `min_calls` calls of that type are left out. Accuracy is weighted so that
    every marmoset counts equally. Reports test accuracy, accuracy with shuffled labels, and the chance
    level 1/n_marmosets.
    """
    check_regressor(regressor)
    rep, marmos = _load_representation(save_dir, regressor)
    first = rep["omega"] if regressor == "ouro" else rep["features"]
    calls = np.unique(np.hstack([list(first[m]) for m in marmos]))
    rng = np.random.default_rng(seed)
    plot_dir = os.path.join(save_dir, "figs", regressor)
    os.makedirs(plot_dir, exist_ok=True)
    results = {}

    for call in calls:
        owners = [m for m in marmos if call in first[m] and len(first[m][call]) > min_calls]
        if len(owners) < 2:
            continue
        data, labels, projs, prep = _call_data(regressor, rep, owners, call, n_pcs, umap_dims, umap_max_fit, seed)
        N = len(labels)
        weights = np.array([1 / np.sum(labels == v) for v in labels])

        accs, null_accs, last = [], [], None
        for _ in range(n_repeats):
            order = rng.permutation(N)
            n_train = int(N * train_prop)
            tr, te = order[:n_train], order[n_train:]
            if len(np.unique(labels[tr])) < len(owners):
                continue
            pre = make_feature_pipeline(acoustic_pcs) if prep == "acoustic" else Pipeline([("scale", StandardScaler())])
            x_tr = pre.fit_transform(data[tr])
            x_te = pre.transform(data[te])
            w_te = weights[te] / weights[te].sum()
            lr = LogisticRegression(max_iter=400).fit(x_tr, labels[tr])
            accs.append(lr.score(x_te, labels[te], w_te))
            lr_null = LogisticRegression(max_iter=400).fit(x_tr, labels[tr][rng.permutation(len(tr))])
            null_accs.append(lr_null.score(x_te, labels[te], w_te))
            last = (lr, x_te, labels[te])

        if not accs:
            continue
        results[call] = {
            "regressor": regressor,
            "marmosets": owners,
            "n_calls": {m: int(len(first[m][call])) for m in owners},
            "n_dims": int(data.shape[1] if prep != "acoustic" or acoustic_pcs is None else acoustic_pcs),
            "test_accuracy_mean": float(np.mean(accs)),
            "test_accuracy_sd": float(np.std(accs)),
            "shuffled_label_accuracy_mean": float(np.mean(null_accs)),
            "chance": 1 / len(owners),
        }
        print(f"[{regressor}] {call}: acc {np.mean(accs):.3f}  shuffled {np.mean(null_accs):.3f}  "
              f"chance {1/len(owners):.3f}")

        if plot and last is not None:
            lr, x_te, y_te = last
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_estimator(
                lr, x_te, y_te, ax=ax, normalize="true", im_kw={"vmin": 0, "vmax": 1},
                display_labels=owners,
            )
            ax.set_title(f"{regressor}, {call}: test")
            fig.savefig(os.path.join(plot_dir, f"confusion_{call}.svg"), transparent=True)
            plt.close(fig)

            cmap = plt.get_cmap("tab10")
            if projs is not None:  # omega and gamma PCs, 3d
                fig = plt.figure(figsize=(12, 5))
                for k, key in enumerate(("omega", "gamma"), start=1):
                    ax = fig.add_subplot(1, 2, k, projection="3d")
                    for i, m in enumerate(owners):
                        p = projs[key][labels == i]
                        ax.scatter(p[:, 0], p[:, 1], p[:, 2], color=cmap(i), s=12, label=m)
                    ax.set_title(key)
                    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_zlabel("PC 3")
                fig.axes[0].legend(frameon=True)
            else:  # first two dimensions of the (preprocessed) baseline representation
                fig, ax = plt.subplots(figsize=(6, 5))
                for i, m in enumerate(owners):
                    ax.scatter(data[labels == i, 0], data[labels == i, 1], color=cmap(i), s=12, label=m)
                ax.legend(frameon=True)
                format_axis(ax, xlabel="dim 1", ylabel="dim 2")
            fig.suptitle(f"{regressor}, {call}")
            fig.savefig(os.path.join(plot_dir, f"projections_{call}.svg"), transparent=True)
            plt.close(fig)

    with open(os.path.join(save_dir, f"decoding_summary_{regressor}.json"), "w") as f:
        json.dump(results, f, indent=2)


def compare(save_dir):
    """test accuracy of every regressor that has a `decoding_summary_<regressor>.json`, per call type"""
    rows = []
    for reg in REGRESSORS:
        path = os.path.join(save_dir, f"decoding_summary_{reg}.json")
        if not os.path.isfile(path):
            continue
        for call, r in json.load(open(path)).items():
            rows.append({"regressor": reg, "call": call, "n_marmosets": len(r["marmosets"]),
                         "n_dims": r["n_dims"], "accuracy": r["test_accuracy_mean"],
                         "accuracy_sd": r["test_accuracy_sd"],
                         "shuffled": r["shuffled_label_accuracy_mean"], "chance": r["chance"]})
    assert rows, f"no decoding_summary_*.json in {save_dir}; run `decode` first"
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "decoding_comparison.csv"), index=False)
    print(df.to_string(index=False))

    piv = df.pivot(index="call", columns="regressor", values="accuracy")
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(piv)), 4))
    piv.plot.bar(ax=ax, rot=30)
    chance = df.groupby("call")["chance"].first().reindex(piv.index)
    ax.plot(np.arange(len(piv)), chance.values, "k_", markersize=25, label="chance")
    ax.legend(frameon=False)
    format_axis(ax, ylabel="Test accuracy (marmoset identity)")
    fig.savefig(os.path.join(save_dir, "decoding_comparison.svg"), transparent=True, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fire.Fire({"embed": embed, "spectra": spectra, "features": features, "decode": decode, "compare": compare})
