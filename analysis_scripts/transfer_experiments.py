"""
High-quality simulated data: can Ouroboros transfer well to small datasets without overfitting
using pre-training on zebra finch data, and do the learned parameters transfer to `true` generating
mechanisms.

This is in comparison to a model trained on just four sampled vocalizations, from the same dataset.

The data folder holds 8 `.inp` files and 8 `.dat` files. Requires a `pretrained-dir` argument (used directly for fine-tuning, and for matching
architecture and sampling rate when training from scratch), and an `updated-model-path` argument, to store either the fine-tuned models or the fully-trained, 4 voc models.
In this `updated-model-path`, checks to see if any models have already been  trained; if so, loads directly.

`run-type`: 'finetune` or `scratch`

Saved under --save-dir:
    <run-type>/corrs.pkl / corrs.csv   per train set and recording: split (train/test), the correlation of
                                       5 ms-smoothed omega and gamma with the true flow, the median
                                       circular-shift null correlation, and its empirical p-value
    <run-type>/nulls.pkl               every shifted-null correlation: {(train_inds, split, recording): ...}
    <run-type>/summary.json            median and IQR of the correlations (over train sets and recordings)
    <run-type>/corrs_vs_null.svg       real vs null correlations, train and test
    comparison.csv/.json/.svg          (from `compare`) finetune and scratch together
"""



import gc
import glob
import json
import os
import pickle
from itertools import combinations

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
from torch.utils.data import DataLoader

from analysis_scripts.common import (
    embed_vocalizations,
    find_best_checkpoint_dir,
    format_axis,
    load_best_model,
    plt,
    resolve_model_path,
)
from data.data_utils import aud_neur_ds
from train.model_cv import model_cv_lambdas
from train.train import save_model, train
from utils import deriv_approx_dy, sse

DESCRIPTIONS = {'3':'400Hz: (1) spike train, constant interval',
               '4.2':'400Hz: (3) no spike after 10ms',
               '4': '400Hz: (2) no spike 15-20ms',
               '5': '400Hz: (4) varied ISI: … 10, 11, 15… 25, 26, 30 … ',
               '6': '200Hz: (1) spike train, constant interval',
               '7': '200Hz: (2) no spike 15-20ms',
               '8': '200Hz: (3) varied ISI: … 10, 12, 20… 30, 32, 40 ',
               '9':'50Hz, spike train, constant interval'
               }

def load_coen_data(data_path,target_fs=0):

    ######### TO DO: ADD IN DOWNSAMPLING FOR THESE DATA #########

    audio_files = glob.glob(os.path.join(data_path,'UTF-8*.dat'))
    audio_files.sort()
    sim_nums = [a.split('_')[-1].split('.dat')[0] for a in audio_files]
    inputs = [os.path.join(data_path,s + '.inp') for s in sim_nums]
    flows = [os.path.join(data_path,f'qc_t1_0.5_1.1_0vs_{s}.dat') for s in sim_nums]

    for inp,flow in zip(inputs,flows):
        assert os.path.isfile(inp), print(f"{inp} not found, check that your directory ({data_path}) is correct!")
        assert os.path.isfile(flow), print(f"{flow} not found, check that your directory ({data_path}) is correct!")
    descs = [DESCRIPTIONS[s] for s in sim_nums]
    ad = [np.genfromtxt(fn) for fn in audio_files]
    ts = [d[:,0]/1000 for d in ad]
    ys = [d[:,1] for d in ad]
    flws = [np.genfromtxt(fn) for fn in flows]
    spikes = [np.genfromtxt(fn,skip_header=2) for fn in inputs]
    srs = [int(round(1/(t[1] - t[0]))) for t in ts]
    if target_fs > 0:
        new_ts,new_ys = [],[]
        for t,y in zip(ts,ys):

            new_t = np.arange(t[0],t[-1],1/target_fs)
            yspl = make_interp_spline(t,y)
            new_y = yspl(new_t)
            new_ts.append(new_t)
            new_ys.append(new_y)
        return new_ts,new_ys,flws,spikes,target_fs,descs
    else:
        return ts,ys,flws,spikes,srs[0],descs

SMOOTH_LEN_S = 0.005  # smoothing timescale of model functions when comparing to flows
TARGETS = ["omega", "gamma"]
SPLITS = ["train", "test"]
CONDITIONS = ["real", "null"]


def _summarize(df):
    """median and IQR of the correlations, over train sets and recordings: {target: {split: {cond: ...}}}"""
    out = {}
    for target in TARGETS:
        out[target] = {}
        for split in SPLITS:
            out[target][split] = {}
            for cond in CONDITIONS:
                v = df.loc[df.split == split, f"corr_{target}" + ("_null" if cond == "null" else "")]
                v = v.dropna().to_numpy(dtype=float)
                out[target][split][cond] = {
                    "median": float(np.median(v)) if len(v) else None,
                    "iqr": [float(x) for x in np.percentile(v, [25, 75])] if len(v) else None,
                    "n": int(len(v)),
                }
    return out


def checkpoint_arch(model_dir):
    """architecture info (as `train` / `save_model` expect it) of the latest checkpoint in `model_dir`"""
    files = glob.glob(os.path.join(model_dir, "*.tar"))
    latest = max(files, key=lambda f: int(f.split("checkpoint_")[-1].split(".tar")[0]))
    sd = torch.load(latest, weights_only=False, map_location="cpu")
    return {
        "n layers": sd["n_layers"],
        "d state": sd["d_state"],
        "d conv": sd["d_conv"],
        "expand factor": sd["expand_factor"],
        "n kernels": sd["n_kernel"],
    }


def make_loaders(audios, batch_size=4, val_frac=0.2):
    """
    train/val/test loaders from whole recordings. Each recording is split in time (first 1-val_frac
    for training, the rest for validation), and pieces are cropped to a common length so they can be
    stacked. The validation set is what selects lambda (scratch) / is monitored (finetune); the held-out
    *recordings* are never touched here. `test` is aliased to `val` since `model_cv_lambdas` asks for it.
    """
    tr = [a[: int(len(a) * (1 - val_frac))] for a in audios]
    va = [a[int(len(a) * (1 - val_frac)) :] for a in audios]

    def loader(pieces, shuffle):
        L = min(len(p) for p in pieces)
        data = np.stack([p[:L, None] for p in pieces], axis=0).astype(np.float64)
        return DataLoader(aud_neur_ds(data), batch_size=batch_size, shuffle=shuffle, num_workers=0)

    val = loader(va, False)
    return {"train": loader(tr, True), "val": val, "test": val}


def finetune_model(
    ouroboros,
    dls,
    dt,
    num_epochs,
    run_dir,
    model_info,
    lr=1e-4,
    lam=1.0,
):
    """
    fine-tune a loaded Ouroboros on `dls` (same objective as pre-training: MSE on the scaled second
    derivative, plus the weight penalty). `lam` is the polynomial-degree penalty base; it is not saved
    with trained models (they load with lam=1), so set it here if you want to match pre-training.
    Saves the result to `run_dir/checkpoint_<num_epochs>.tar` and returns the model.
    """
    os.makedirs(run_dir, exist_ok=True)
    ouroboros.kernel.lam = lam
    optimizer = torch.optim.Adam(ouroboros.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=max(num_epochs // 25, 2), min_lr=1e-10
    )
    _, _, ouroboros, optimizer = train(
        ouroboros,
        optimizer,
        loss_fn=lambda y, yhat: sse(yhat, y, reduction="mean"),
        loaders=dls,
        scheduler=scheduler,
        nEpochs=num_epochs,
        val_freq=1,
        runDir=run_dir,
        dt=dt,
        vis_freq=0,
        smoothing=False,
        reg_weights=True,
        save_freq=0,
        model_info=model_info,
    )
    save_model(
        ouroboros,
        optimizer,
        os.path.join(run_dir, f"checkpoint_{num_epochs}.tar"),
        n_layers=model_info["n layers"],
        d_state=model_info["d state"],
        d_conv=model_info["d conv"],
        expand_factor=model_info["expand factor"],
    )
    ouroboros.eval()
    return ouroboros


def _corr(a, b):
    """Pearson correlation, truncating to the shorter of the two (recordings can differ in length)"""
    n = min(len(a), len(b))
    return np.corrcoef(a[:n], b[:n])[0, 1]


def flow_correlations(model, ts, audios, flows, sr, smooth_len_s=SMOOTH_LEN_S, n_shifts=200,
                      min_shift_frac=0.1, min_shift_smooth_mult=2.0, seed=1111):
    """
    Pearson correlation, per recording, between the model's omega / gamma (smoothed on
    `smooth_len_s`) and the true flow (column 1 of the flow file; column 0 is time in ms).
    Flows are interpolated onto the audio time grid.

    Null: circular time shift. The model function is rolled by a random lag relative to the flow (within
    the same recording), `n_shifts` times. This keeps both signals' autocorrelation and marginals and breaks
    only their alignment. Lags are drawn without replacement from [min_lag, L - min_lag], where min_lag is
    the larger of `min_shift_frac` * L and `min_shift_smooth_mult` smoothing lengths. Shorter minimum lags
    leave the shifted function partly correlated with its original alignment (conservative p-values), and
    nearby lags are strongly correlated, so the effective number of independent shifts is roughly
    (L - 2 min_lag) / smoothing length, far fewer than n_shifts; treat p-values as coarse for short recordings.
    (Circular wrapping puts one discontinuity at the seam of the shifted function; for periodic stimuli
    a lag near a multiple of the period realigns the signals, which makes the null conservative.)

    returns omega corr, gamma corr (arrays, one entry per recording), and the null correlations
    for omega and gamma (lists of arrays of length n_shifts, one per recording)
    """
    gen = np.random.default_rng(seed)
    vocs = [a[None, :, None] for a in audios]
    omegas, gammas = embed_vocalizations(
        model, vocs, sr, smooth_len_s=smooth_len_s, desc="model functions"
    )
    flows_t = [np.interp(t, flw[:, 0] / 1000, flw[:, 1]) for t, flw in zip(ts, flows)]

    c_omega, c_gamma, null_omega, null_gamma = [], [], [], []
    for omega, gamma, flow in zip(omegas, gammas, flows_t):
        L = min(len(omega), len(flow))
        omega, gamma, flow = omega[:L], gamma[:L], flow[:L]
        min_lag = max(int(np.ceil(min_shift_smooth_mult * smooth_len_s * sr)), int(min_shift_frac * L))
        assert L > 2 * min_lag, f"recording too short ({L} samples) for circular-shift null"
        all_lags = np.arange(min_lag, L - min_lag + 1)
        lags = gen.choice(all_lags, size=min(n_shifts, len(all_lags)), replace=False)
        c_omega.append(_corr(omega, flow))
        c_gamma.append(_corr(gamma, flow))
        null_omega.append(np.array([_corr(np.roll(omega, k), flow) for k in lags]))
        null_gamma.append(np.array([_corr(np.roll(gamma, k), flow) for k in lags]))
    return np.array(c_omega), np.array(c_gamma), null_omega, null_gamma


def shift_p(real, null):
    """two-sided empirical p of |real| against the |null| distribution (with the +1 correction)"""
    return (1 + np.sum(np.abs(null) >= np.abs(real))) / (1 + len(null))


def run(
    data_dir: str,
    save_dir: str,
    updated_model_path: str,
    pretrained_dir: str,
    run_type: str = "finetune",
    train_size: int = 4,
    device: str = "cuda",
    num_epochs: int = 500,
    lr: float = 1e-4,
    batch_size: int = 4,
    target_fs: int = 0,
):
    """
    For every choice of `train_size` recordings out of the data folder, either fine-tune the pre-trained
    model at `pretrained_dir` (`run_type='finetune'`) or train a model from scratch with lambda cross-validation
    (`run_type='scratch'`) on just those recordings. Scratch models use the pre-trained model's architecture
    (n_layers, d_state, d_conv, expand_factor, n_kernels) and sampling rate, so the two are matched. Each model is scored on train and held-out recordings by the correlation of
    its 5 ms-smoothed omega and gamma with the true flow.

    target_fs: resample the recordings to this rate first. The default (0) resamples to the pre-trained
        model's rate (1 / tau), since tau is tied to the sampling interval at pre-training.
    """
    assert run_type in ("finetune", "scratch"), f"unknown run_type {run_type!r}"
    pretrained_dir = resolve_model_path(pretrained_dir)
    arch = checkpoint_arch(find_best_checkpoint_dir(pretrained_dir)[0])
    if target_fs <= 0:
        target_fs = int(round(1 / load_best_model(pretrained_dir, device=device)[0].tau))

    times, audios, flows, spikes, sr, descriptions = load_coen_data(data_dir, target_fs=target_fs)
    dt = 1 / sr
    n = len(times)
    run_root = os.path.join(updated_model_path, run_type)
    out_dir = os.path.join(save_dir, run_type)
    os.makedirs(out_dir, exist_ok=True)

    records, nulls = [], {}
    for train_inds in tqdm(list(combinations(range(n), train_size)), desc="train sets"):
        train_inds = list(train_inds)
        test_inds = [i for i in range(n) if i not in train_inds]
        run_dir = os.path.join(run_root, "train_" + "-".join(map(str, train_inds)))
        os.makedirs(run_dir, exist_ok=True)

        if run_type == "finetune":
            if glob.glob(os.path.join(run_dir, "*.tar")):
                model,best_lambda = load_best_model(run_dir, device=device)  # already fine-tuned
            else:
                dls = make_loaders([audios[i] for i in train_inds], batch_size=batch_size)
                model,best_lambda = load_best_model(pretrained_dir, device=device)
                model = finetune_model(
                    model, dls, dt, num_epochs, run_dir,
                    model_info=arch,
                    lr=lr, lam=best_lambda,
                )
        else:
            if os.path.isfile(os.path.join(run_dir, "cv_errs.csv")):
                model,best_lambda = load_best_model(run_dir, device=device)  # cv already done
            else:
                dls = make_loaders([audios[i] for i in train_inds], batch_size=batch_size)
                model = model_cv_lambdas(
                    dls=dls, dt=dt, n_epochs=num_epochs, lr=1e-3, n_kernels=arch["n kernels"],
                    expand_factor=arch["expand factor"], n_layers=arch["n layers"],
                    d_state=arch["d state"], d_conv=arch["d conv"], tau=dt, model_path=run_dir, device=device,
                )
        model.eval()

        for split, inds in (("train", train_inds), ("test", test_inds)):
            c_omega, c_gamma, null_omega, null_gamma = flow_correlations(
                model,
                [times[i] for i in inds],
                [audios[i] for i in inds],
                [flows[i] for i in inds],
                sr,
            )
            for k, i in enumerate(inds):
                key = (tuple(train_inds), split, i)
                nulls[key] = {"omega": null_omega[k], "gamma": null_gamma[k]}
                records.append(
                    {"train_inds": tuple(train_inds), "split": split, "recording": i,
                     "description": descriptions[i],
                     "corr_omega": c_omega[k], "corr_gamma": c_gamma[k],
                     "corr_omega_null": np.median(null_omega[k]),
                     "corr_gamma_null": np.median(null_gamma[k]),
                     "p_omega": shift_p(c_omega[k], null_omega[k]),
                     "p_gamma": shift_p(c_gamma[k], null_gamma[k])}
                )
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # saved after every train set so that a long run can be inspected (and resumed) early
        df = pd.DataFrame(records)
        with open(os.path.join(out_dir, "corrs.pkl"), "wb") as f:
            pickle.dump(df, f)
        df.to_csv(os.path.join(out_dir, "corrs.csv"), index=False)
        with open(os.path.join(out_dir, "nulls.pkl"), "wb") as f:
            pickle.dump(nulls, f)

    summary = {"run_type": run_type, "train_size": train_size, "smooth_len_s": SMOOTH_LEN_S,
               "n_train_sets": len(set(df["train_inds"])), "corr": _summarize(df)}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["corr"], indent=2))

    fig, axs = plt.subplots(1, len(TARGETS), figsize=(4 * len(TARGETS), 4), sharey=True, squeeze=False)
    for ax, target in zip(axs.ravel(), TARGETS):
        cols = [(sp, c) for sp in SPLITS for c in CONDITIONS]
        ax.boxplot(
            [df.loc[df.split == sp, f"corr_{target}" + ("_null" if c == "null" else "")].dropna()
             for sp, c in cols],
            tick_labels=[f"{sp}\n{c}" for sp, c in cols],
        )
        format_axis(ax, title=target, ylabel="Correlation with flow" if target == TARGETS[0] else "")
    fig.savefig(os.path.join(out_dir, "corrs_vs_null.svg"), transparent=True)
    plt.close(fig)

    return df


def compare(save_dir):
    """collects the `summary.json` of every run type under save_dir into one table and figure"""
    rows = []
    for path in sorted(glob.glob(os.path.join(save_dir, "*", "summary.json"))):
        s = json.load(open(path))
        for target, splits in s["corr"].items():
            for split, conds in splits.items():
                row = {"run_type": s["run_type"], "target": target, "split": split,
                       "train_size": s["train_size"]}
                for c in CONDITIONS:
                    row[f"{c}_median"] = conds[c]["median"]
                    row[f"{c}_q25"], row[f"{c}_q75"] = conds[c]["iqr"] if conds[c]["iqr"] else (None, None)
                rows.append(row)
    assert rows, f"no */summary.json under {save_dir}; run `run` first"
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "comparison.csv"), index=False)
    with open(os.path.join(save_dir, "comparison.json"), "w") as f:
        json.dump(rows, f, indent=2)
    if df["train_size"].nunique() > 1:
        print("WARNING: run types were run with different --train-size; correlations are not comparable")
    print(df.to_string(index=False))

    fig, axs = plt.subplots(1, len(TARGETS), figsize=(max(5, 1.6 * len(df) / len(TARGETS)) * len(TARGETS) / 2, 4),
                            sharey=True, squeeze=False)
    w = 0.38
    for ax, target in zip(axs.ravel(), TARGETS):
        d = df[df.target == target].reset_index(drop=True)
        x = np.arange(len(d))
        for k, c in enumerate(CONDITIONS):
            ax.bar(x + (k - 0.5) * w, d[f"{c}_median"].astype(float), w, label=c,
                   yerr=[(d[f"{c}_median"] - d[f"{c}_q25"]).astype(float),
                         (d[f"{c}_q75"] - d[f"{c}_median"]).astype(float)], capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}\n{sp}" for r, sp in zip(d.run_type, d.split)], fontsize=8)
        format_axis(ax, title=target,
                    ylabel="Correlation with flow (median, IQR)" if target == TARGETS[0] else "")
    axs.ravel()[-1].legend(frameon=False)
    fig.savefig(os.path.join(save_dir, "comparison.svg"), transparent=True, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fire.Fire({"run": run, "compare": compare})
