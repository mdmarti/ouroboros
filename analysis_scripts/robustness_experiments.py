"""
Robustness of the learned latents (omega, gamma) to random seed, polynomial order K and regularization lambda.

Ported from scripts/assess_consistency.py (which trained `n_models` models and compared R^2, and whose
imports no longer exist) to the current model API, and extended to compare the latents themselves,
which is what the identifiability argument needs. Plan item A5.

    1. train:   train one model for every (K, lambda, seed) combination on the same pool of audio
                (a fixed hold-out set is removed first, so that no model has seen it). Seeds change the
                train/val/test split and the model initialization. Models land in SAVE/K{K}_lam{lambda}_seed{seed}
                and SAVE/manifest.json lists them.
    2. compare: on the shared hold-out set, report each model's R^2 for the second derivative, and the
                per-chunk Pearson correlation of omega and gamma between pairs of models:
                  - same K and lambda, different seeds   -> seed robustness
                  - same seed, different K or lambda      -> hyperparameter sensitivity
                -> SAVE/model_r2.csv, SAVE/latent_correlations.csv, SAVE/summary.json, figures

    python -m analysis_scripts.robustness_experiments train \
        --audio_dir A --seg_dir S --save-dir SAVE --seeds 1,2,3 --n_kernels 10 --lambdas 1.5
    python -m analysis_scripts.robustness_experiments compare --save-dir SAVE   # [--model-path EARLIER_SAVE_DIR]

Notes
-----
    - unlike `model_cv_lambdas`, each model here is trained at a single, given lambda (not cross-validated
      over lambda), so that seeds and hyperparameters can be compared directly
    - comparison of omega/gamma between models only makes sense for models trained on the same audio
    - models from different seeds with the same split seed would differ only in initialization; here the split
      seed and the initialization seed are the same number
"""

import itertools
import json
import os
import random
from itertools import combinations

import fire
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from analysis_scripts.common import embed_vocalizations, format_axis, plt, resolve_model_path
from data.data_utils import aud_neur_ds, get_loaders
from data.load_data import get_segmented_audio
from model.kernels import fullPolyModule
from model.model import Ouroboros
from train.eval import eval_model_error
from train.train import load_model, save_model, train
from utils import sse
from visualization.model_vis import loss_plot

# visualization.model_vis turns on LaTeX rendering at import; mathtext is enough here and needs no LaTeX install
plt.rcParams["text.usetex"] = False


def _as_list(x, cast):
    if isinstance(x, (list, tuple)):
        return [cast(v) for v in x]
    return [cast(v) for v in str(x).split(",")]


def _model_name(k, lam, seed):
    return f"K{k}_lam{lam}_seed{seed}"


def train_single(
    dls, dt, lam, n_kernels, model_path, n_epochs=100, lr=1e-3,
    expand_factor=10, n_layers=4, d_state=1, d_conv=4, smooth_len=0.001,
    save_freq=5, device="cuda",
):
    """trains one model at a fixed lambda; mirrors the inner loop of `model_cv_lambdas`"""
    kernel = fullPolyModule(nTerms=n_kernels, device=device, x_dim=1, z_dim=2,
                            activation=lambda x: x, lam=lam)
    model = Ouroboros(d_data=1, n_layers=n_layers, d_state=d_state, d_conv=d_conv,
                      expand_factor=expand_factor, tau=dt, smooth_len=smooth_len,
                      kernel=kernel, device=device)
    opt = Adam(model.parameters(), lr=lr)
    sched = ReduceLROnPlateau(opt, factor=0.5, patience=max(n_epochs // 25, 2), min_lr=1e-10)
    info = {"n layers": n_layers, "d state": d_state, "d conv": d_conv,
            "expand factor": expand_factor}
    tl, vl, model, opt = train(
        model, opt, loss_fn=lambda y, yhat: sse(yhat, y, reduction="mean"),
        loaders=dls, scheduler=sched, nEpochs=n_epochs, val_freq=1, runDir=model_path,
        dt=dt, vis_freq=max(n_epochs // 10, 1), smoothing=False, reg_weights=True,
        start_epoch=0, save_freq=save_freq, model_info=info,
    )
    loss_plot(tl, vl, save_loc=model_path, show=False)
    save_model(model, opt, os.path.join(model_path, f"checkpoint_{n_epochs}.tar"),
               n_layers=n_layers, d_state=d_state, expand_factor=expand_factor, d_conv=d_conv)
    return model


def _load_pool(audio_dir, seg_dir, audio_id, context_len, max_vocs, n_holdout, holdout_seed):
    chunks, sr = get_segmented_audio(
        audio_dir, seg_dir, audio_id=audio_id, max_vocs=max_vocs, context_len=context_len,
        seed=holdout_seed, training=True, extend=True, shuffle_order=True,
    )
    chunks = np.stack(chunks, axis=0)
    order = np.random.default_rng(holdout_seed).permutation(len(chunks))
    holdout, pool = chunks[order[:n_holdout]], chunks[order[n_holdout:]]
    return pool, holdout, sr


def train_models(
    audio_dir, seg_dir, save_dir, seeds="1,2,3", n_kernels="10", lambdas="1.5",
    audio_id=".wav", context_len=0.15, max_vocs=5000, n_holdout=500, holdout_seed=0,
    n_epochs=100, lr=1e-3, batch_size=32, expand_factor=10, n_layers=4, save_freq=5,
    device="cuda",
):
    """trains a model for every combination of `seeds`, `n_kernels` and `lambdas` (comma separated)"""
    os.makedirs(save_dir, exist_ok=True)
    seeds, ks, lams = _as_list(seeds, int), _as_list(n_kernels, int), _as_list(lambdas, float)
    pool, holdout, sr = _load_pool(audio_dir, seg_dir, audio_id, context_len, max_vocs,
                                   n_holdout, holdout_seed)
    dt = 1 / sr
    np.save(os.path.join(save_dir, "holdout.npy"), holdout)
    print(f"{len(pool)} training chunks, {len(holdout)} hold-out chunks, sr={sr}")

    manifest_path = os.path.join(save_dir, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.isfile(manifest_path) else []
    manifest = [m for m in manifest if os.path.isdir(os.path.join(save_dir, m["name"]))]
    done = {m["name"] for m in manifest}

    for seed, k, lam in itertools.product(seeds, ks, lams):
        name = _model_name(k, lam, seed)
        if name in done:
            print(f"{name}: already trained, skipping")
            continue
        print(f"=== training {name}")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        dls = get_loaders(pool, num_workers=4, batch_size=batch_size, train_size=0.6,
                          cv=True, seed=seed, dt=dt)
        model = train_single(dls, dt, lam, k, os.path.join(save_dir, name), n_epochs=n_epochs,
                             lr=lr, expand_factor=expand_factor, n_layers=n_layers,
                             save_freq=save_freq, device=device)
        manifest.append({"name": name, "seed": seed, "n_kernels": k, "lambda": lam,
                         "sr": sr, "dt": dt})
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        del model, dls
        torch.cuda.empty_cache()


def _pair_corr(a_list, b_list):
    cs = []
    for a, b in zip(a_list, b_list):
        n = min(len(a), len(b))
        if n > 2 and np.std(a[:n]) > 0 and np.std(b[:n]) > 0:
            cs.append(np.corrcoef(a[:n], b[:n])[0, 1])
    return np.array(cs)


def compare(save_dir, model_path="", smooth_len_s=0.005, device="cuda"):
    """
    R^2 and latent agreement between all models on the shared hold-out set. The models are the ones `train`
    saved in `save_dir`, unless `model_path` points to the folder of an earlier `train` run (the one holding
    `manifest.json` and `holdout.npy`); results are written to `save_dir` either way.
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = resolve_model_path(model_path, save_dir)
    manifest = json.load(open(os.path.join(model_path, "manifest.json")))
    holdout = np.load(os.path.join(model_path, "holdout.npy"))
    dt, sr = manifest[0]["dt"], manifest[0]["sr"]
    dl = DataLoader(aud_neur_ds(holdout), batch_size=32, shuffle=False)
    dls = {"train": dl, "val": dl, "test": dl}

    r2_rows, latents = [], {}
    for m in manifest:
        model, *_ = load_model(os.path.join(model_path, m["name"]), device=device)
        model.eval()
        with torch.no_grad():
            _, _, (_, r2s) = eval_model_error(dls, model, dt=dt)
        r2s = np.asarray(r2s).ravel()
        r2_rows.append({**m, "r2_median": float(np.median(r2s)), "r2_mean": float(np.mean(r2s))})
        omegas, gammas = embed_vocalizations(model, [h[None] for h in holdout], sr,
                                             smooth_len_s, max_len_t=4,
                                             desc=f"embedding {m['name']}")
        latents[m["name"]] = {"omega": omegas, "gamma": gammas}
        print(f"{m['name']}: median R2 {r2_rows[-1]['r2_median']:.3f}")
        del model
        torch.cuda.empty_cache()

    pd.DataFrame(r2_rows).to_csv(os.path.join(save_dir, "model_r2.csv"), index=False)

    rows = []
    for a, b in combinations(manifest, 2):
        same_cfg = (a["n_kernels"], a["lambda"]) == (b["n_kernels"], b["lambda"])
        same_seed = a["seed"] == b["seed"]
        if same_cfg and not same_seed:
            kind = "seed"
        elif same_seed and not same_cfg:
            kind = "hyperparameter"
        else:
            continue
        for key in ("omega", "gamma"):
            cs = _pair_corr(latents[a["name"]][key], latents[b["name"]][key])
            rows.append({"kind": kind, "latent": key, "model_a": a["name"], "model_b": b["name"],
                         "median_corr": float(np.median(cs)) if len(cs) else np.nan,
                         "q25": float(np.percentile(cs, 25)) if len(cs) else np.nan,
                         "q75": float(np.percentile(cs, 75)) if len(cs) else np.nan,
                         "n_chunks": int(len(cs))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "latent_correlations.csv"), index=False)

    summary = {"model_r2": {r["name"]: r["r2_median"] for r in r2_rows}}
    if len(df):
        summary["latent_agreement"] = {
            f"{kind}/{key}": {"median_of_pairs": float(g.median_corr.median()),
                              "min": float(g.median_corr.min()), "n_pairs": int(len(g))}
            for (kind, key), g in df.groupby(["kind", "latent"])
        }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([r["name"] for r in r2_rows], [r["r2_median"] for r in r2_rows], color="tab:blue")
    ax.tick_params(axis="x", rotation=90)
    format_axis(ax, ylabel=r"Hold-out $R^2$ (median over chunks)")
    fig.savefig(os.path.join(save_dir, "model_r2.svg"), transparent=True, bbox_inches="tight")
    plt.close(fig)

    if len(df):
        groups = [(f"{k}\n{lat}", g.median_corr.to_numpy()) for (k, lat), g in df.groupby(["kind", "latent"])]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot([v for _, v in groups], tick_labels=[l for l, _ in groups])
        format_axis(ax, ylabel="Median per-chunk correlation between models")
        ax.set_ylim(min(0, np.nanmin(df.median_corr)), 1)
        fig.savefig(os.path.join(save_dir, "latent_agreement.svg"), transparent=True, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    fire.Fire({"train": train_models, "compare": compare})
