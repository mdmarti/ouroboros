"""
Pupil-tutor comparison: do juvenile-turned-adult zebra finches' motifs look like their tutors' in omega/gamma?

Rewritten for the current codebase (the old version imported modules that no longer exist, and stopped before
extracting any model functions). Trajectory extraction and plots follow the "Figure Adult pupil/tutor"
section of notebooks/cosyne_plots.ipynb. Three commands, each caching its output:

    train    train one model (cross-validated over lambda) on motifs from all pupils and all tutors
             -> SAVE/model/
    embed    omega and gamma for every motif of every pupil and tutor, each resampled to `n_interp_points`
             points on normalized time -> SAVE/functions/<bird>_functions.pkl, <bird>_tutor_functions.pkl
    compare  mean omega/gamma trajectories of each pupil against each tutor
             -> plots per pair, SAVE/pupil_tutor_distances.csv, SAVE/pupil_tutor_summary.json

    python -m analysis_scripts.run_pupil_tutor_experiments train   --data-dir DATA --save-dir OUT
    python -m analysis_scripts.run_pupil_tutor_experiments embed   --data-dir DATA --save-dir OUT
    python -m analysis_scripts.run_pupil_tutor_experiments compare --save-dir OUT

`embed` uses the model trained by `train` (saved in SAVE/model) unless --model-path points to another trained model,
so `train` can be skipped if you already have one.

Expected layout (`DATA` holds one folder per pupil, and one `<pupil>_tutor` folder per tutor):

    DATA/<bird>/motif_audio/double_denoised/<tag>_cleaned.wav        DATA/<bird>/motif_txt/<tag>.txt
    DATA/<bird>_tutor/motif_audio_tutor/double_denoised/<tag>_cleaned.wav
    DATA/<bird>_tutor/motif_txt_tutor/<tag>.txt

The subfolder names are arguments. Birds are found automatically (folders matching `[b,g,p]*[0-9]` that have a
`_tutor` partner), or given with --birds blk411,blk415,...

`compare` (new; plan item B4, a summary across pairs): the distance between a pupil's and a tutor's mean
trajectories is the root-mean-square difference of their mean omega and gamma over normalized time, each latent
divided by its standard deviation across all motifs. For every pupil it reports the rank of the pupil's own
tutor among all tutors (1 = closest), and whether the own tutor is the nearest. A permutation test (random
reassignment of tutors to pupils) gives a p-value for the mean own-tutor distance. With few birds this is an
existence check, not a precise estimate. Time normalization removes tempo, so motif durations are saved and
reported separately.

Differences from the notebook: motifs longer than --max-len-s are skipped and none are truncated (the notebook
also truncated motifs to the first --max-len-s, 0.9 s in the figure; use --truncate-s for that); and the padding
the data loader adds before onset is removed after embedding.
"""

import gc
import glob
import json
import os
import pickle
from itertools import product

import fire
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import make_interp_spline
from tqdm import tqdm

from analysis_scripts.common import (
    embed_vocalizations, format_axis, load_best_model, plt, resolve_model_path,
)
from data.data_utils import get_loaders
from data.load_data import get_segmented_audio
from train.model_cv import model_cv_lambdas

# visualization.model_vis turns on LaTeX rendering at import; mathtext is enough here and needs no LaTeX install
plt.rcParams["text.usetex"] = False

ROLES = ("pupil", "tutor")


# ------------------------------------------------------------------ data layout


def _dirs(data_dir, bird, role, audio_subdir, seg_subdir, tutor_audio_subdir, tutor_seg_subdir):
    if role == "pupil":
        base = os.path.join(data_dir, bird)
        return os.path.join(base, audio_subdir), os.path.join(base, seg_subdir)
    base = os.path.join(data_dir, bird + "_tutor")
    return os.path.join(base, tutor_audio_subdir), os.path.join(base, tutor_seg_subdir)


def find_birds(data_dir, birds=None):
    """pupils with a tutor folder; `birds` (comma separated or a list) restricts the set"""
    if birds:
        birds = birds.split(",") if isinstance(birds, str) else list(birds)
        return [str(b) for b in birds]
    found = sorted(os.path.basename(p) for p in glob.glob(os.path.join(data_dir, "[b,g,p]*[0-9]")))
    found = [b for b in found if os.path.isdir(os.path.join(data_dir, b + "_tutor"))]
    assert found, f"no pupil/tutor folders found in {data_dir}"
    return found


# ------------------------------------------------------------------ train


def train(
    data_dir,
    save_dir,
    birds=None,
    max_vocs=3000,
    context_len=0.15,
    n_epochs=200,
    lr=1e-3,
    n_kernels=10,
    n_layers=3,
    expand_factor=8,
    batch_size=32,
    seed=92,
    audio_subdir="motif_audio/double_denoised",
    seg_subdir="motif_txt",
    tutor_audio_subdir="motif_audio_tutor/double_denoised",
    tutor_seg_subdir="motif_txt_tutor",
    device="cuda",
):
    """trains on `max_vocs` chunks, split evenly over all pupils and tutors"""
    birds = find_birds(data_dir, birds)
    model_dir = os.path.join(save_dir, "model")  # the trained model is saved here
    os.makedirs(model_dir, exist_ok=True)

    n_per = max(max_vocs // (2 * len(birds)), 1)
    chunks, sr = [], None
    for b, role in product(birds, ROLES):
        a_dir, s_dir = _dirs(data_dir, b, role, audio_subdir, seg_subdir, tutor_audio_subdir, tutor_seg_subdir)
        print(f"loading up to {n_per} chunks from {b} ({role})")
        c, this_sr = get_segmented_audio(
            a_dir, s_dir, audio_id="_cleaned.wav", max_vocs=n_per, context_len=context_len,
            seed=seed, training=True, extend=True, shuffle_order=True,
        )
        if len(c) == 0:
            print(f"  WARNING: nothing found in {a_dir}")
            continue
        assert sr is None or this_sr == sr, f"sample rates differ ({this_sr} vs {sr})"
        sr = this_sr
        chunks += c
    assert chunks, "no audio found"
    print(f"{len(chunks)} chunks, sr={sr}")

    dt = 1 / sr
    dls = get_loaders(np.stack(chunks, axis=0), batch_size=batch_size, train_size=0.6, cv=True,
                      seed=seed, dt=dt)
    model_cv_lambdas(
        dls, dt, n_epochs=n_epochs, lr=lr, n_kernels=n_kernels, n_layers=n_layers,
        expand_factor=expand_factor, tau=dt, model_path=model_dir, device=device,
    )
    with open(os.path.join(save_dir, "train_info.json"), "w") as f:
        json.dump({"birds": birds, "n_chunks": len(chunks), "sr": sr, "context_len": context_len,
                   "n_epochs": n_epochs, "n_kernels": n_kernels, "seed": seed}, f, indent=2)


# ------------------------------------------------------------------ embed


def _bird_functions(model, a_dir, s_dir, max_vocs, seed, padding, smooth_len, n_interp_points,
                    max_len_s, min_len_s, truncate_s):
    """omega and gamma of every motif in a folder, resampled to normalized time"""
    motifs, sr = get_segmented_audio(
        a_dir, s_dir, audio_id="_cleaned.wav", max_vocs=max_vocs, seed=seed,
        training=False, padding=padding, shuffle_order=True,
    )
    if len(motifs) == 0:
        return None
    pad = int(round(padding * sr))
    keep, durations = [], []
    for m in motifs:
        dur = (m.shape[1] - pad) / sr
        if dur > max_len_s or dur < min_len_s:
            continue
        if truncate_s:
            m = m[:, : pad + int(round(truncate_s * sr)), :]
            dur = min(dur, truncate_s)
        keep.append(m)
        durations.append(dur)
    omegas, gammas = embed_vocalizations(model, keep, sr, smooth_len, max_len_t=max_len_s + padding + 1,
                                         desc="embedding motifs")
    t_eval = np.linspace(0, 1, n_interp_points)
    om, ga = [], []
    for o, g in zip(omegas, gammas):
        o, g = o[pad:], g[pad:]  # drop the context added before onset
        if len(o) < 5:
            om.append(np.full(n_interp_points, np.nan))
            ga.append(np.full(n_interp_points, np.nan))
            continue
        t = np.linspace(0, 1, len(o))
        om.append(make_interp_spline(t, o)(t_eval))
        ga.append(make_interp_spline(t, g)(t_eval))
    return {"omegas": np.stack(om), "gammas": np.stack(ga), "durations": np.array(durations), "sr": sr,
            "n_skipped": len(motifs) - len(keep)}


def embed(
    data_dir,
    save_dir,
    birds=None,
    model_path="",
    max_vocs_per_bird=1000,
    padding=0.05,
    smooth_len=0.005,
    n_interp_points=1000,
    max_len_s=2.0,
    min_len_s=0.1,
    truncate_s=0.0,
    seed=152,
    audio_subdir="motif_audio/double_denoised",
    seg_subdir="motif_txt",
    tutor_audio_subdir="motif_audio_tutor/double_denoised",
    tutor_seg_subdir="motif_txt_tutor",
    device="cuda",
):
    """
    embeds motifs of every pupil and tutor with the model at `model_path` (the folder from `model_cv_lambdas`,
    or a single model's folder). Defaults to the model that `train` saved in <save-dir>/model
    """
    birds = find_birds(data_dir, birds)
    fnc_dir = os.path.join(save_dir, "functions")
    os.makedirs(fnc_dir, exist_ok=True)
    todo = [(b, r) for b, r in product(birds, ROLES)
            if not os.path.isfile(os.path.join(fnc_dir, _fn(b, r)))]
    model = (load_best_model(resolve_model_path(model_path, os.path.join(save_dir, "model")), device=device)
             if todo else None)

    for b, role in todo:
        a_dir, s_dir = _dirs(data_dir, b, role, audio_subdir, seg_subdir, tutor_audio_subdir, tutor_seg_subdir)
        print(f"embedding {b} ({role})")
        out = _bird_functions(model, a_dir, s_dir, max_vocs_per_bird, seed, padding, smooth_len,
                              n_interp_points, max_len_s, min_len_s, truncate_s)
        if out is None:
            print(f"  WARNING: no motifs found in {a_dir}")
            continue
        with open(os.path.join(fnc_dir, _fn(b, role)), "wb") as f:
            pickle.dump(out, f)
        print(f"  {len(out['omegas'])} motifs ({out['n_skipped']} skipped for length)")
        gc.collect()
        torch.cuda.empty_cache()


def _fn(bird, role):
    return f"{bird}_functions.pkl" if role == "pupil" else f"{bird}_tutor_functions.pkl"


# ------------------------------------------------------------------ compare


def _load_functions(save_dir):
    funcs, birds = {}, []
    for path in sorted(glob.glob(os.path.join(save_dir, "functions", "*_functions.pkl"))):
        name = os.path.basename(path).replace("_functions.pkl", "")
        with open(path, "rb") as f:
            funcs[name] = pickle.load(f)
    for name in funcs:
        if not name.endswith("_tutor") and name + "_tutor" in funcs:
            birds.append(name)
    assert birds, f"no pupil/tutor pairs in {save_dir}/functions; run `embed` first"
    return funcs, birds


def compare(save_dir, n_perm=10000, seed=0, plot=True):
    """mean trajectories, pupil-tutor distances, own-tutor rank and a permutation test"""
    funcs, birds = _load_functions(save_dir)
    rng = np.random.default_rng(seed)

    # each latent is scaled by its sd across all motifs, so omega and gamma count equally
    allo = np.concatenate([f["omegas"].ravel() for f in funcs.values()])
    allg = np.concatenate([f["gammas"].ravel() for f in funcs.values()])
    scale = np.array([np.nanstd(allo), np.nanstd(allg)])

    def mean_traj(name):
        f = funcs[name]
        return np.nanmean(np.stack([f["omegas"], f["gammas"]], axis=-1), axis=0) / scale  # (T, 2)

    pupils = {b: mean_traj(b) for b in birds}
    tutors = {b: mean_traj(b + "_tutor") for b in birds}
    D = np.array([[np.sqrt(np.nanmean((pupils[p] - tutors[t]) ** 2)) for t in birds] for p in birds])
    dist = pd.DataFrame(D, index=[f"{b} (pupil)" for b in birds], columns=[f"{b}_tutor" for b in birds])
    dist.to_csv(os.path.join(save_dir, "pupil_tutor_distances.csv"))
    print(dist.round(3).to_string())

    n = len(birds)
    own = np.diag(D)
    ranks = np.array([1 + np.sum(D[i] < D[i, i]) for i in range(n)])
    perm_means = np.array([np.mean(D[np.arange(n), rng.permutation(n)]) for _ in range(n_perm)])
    p_value = float((np.sum(perm_means <= own.mean()) + 1) / (n_perm + 1))

    durations = {b: {"pupil_mean_s": float(np.mean(funcs[b]["durations"])),
                     "tutor_mean_s": float(np.mean(funcs[b + "_tutor"]["durations"])),
                     "n_pupil_motifs": int(len(funcs[b]["omegas"])),
                     "n_tutor_motifs": int(len(funcs[b + "_tutor"]["omegas"]))} for b in birds}
    summary = {
        "birds": birds,
        "own_tutor_distance": {b: float(d) for b, d in zip(birds, own)},
        "own_tutor_rank": {b: int(r) for b, r in zip(birds, ranks)},
        "n_own_tutor_nearest": int(np.sum(ranks == 1)),
        "chance_expected_n_nearest": 1.0,
        "mean_own_distance": float(own.mean()),
        "mean_other_distance": float(np.mean(D[~np.eye(n, dtype=bool)])),
        "permutation_p_value": p_value,
        "n_permutations": n_perm,
        "latent_scale_sd": {"omega": float(scale[0]), "gamma": float(scale[1])},
        "motifs": durations,
    }
    with open(os.path.join(save_dir, "pupil_tutor_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "motifs"}, indent=2))

    if not plot:
        return
    plot_dir = os.path.join(save_dir, "figs")
    os.makedirs(plot_dir, exist_ok=True)
    cmap = plt.get_cmap("tab20")
    t = np.linspace(0, 1, funcs[birds[0]]["omegas"].shape[1])
    for ii, b in enumerate(birds):
        fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        for name, label, color in ((b, "pupil", cmap((2 * ii + 1) / 20)),
                                   (b + "_tutor", "tutor", cmap(2 * ii / 20))):
            for ax, key, nm in zip(axs, ("omegas", "gammas"), ("omega", "gamma")):
                x = funcs[name][key]
                mu, sd = np.nanmean(x, 0), np.nanstd(x, 0)
                ax.plot(t, mu, color=color, label=f"{label} (n={len(x)})")
                ax.fill_between(t, mu - sd, mu + sd, color=color, alpha=0.2)
                format_axis(ax, ylabel=nm)
        axs[1].set_xlabel("Time within motif (normalized)")
        axs[0].legend(frameon=False)
        axs[0].set_title(f"{b}: own-tutor rank {ranks[ii]}/{n}")
        fig.savefig(os.path.join(plot_dir, f"{b}_pupil_vs_tutor.svg"), transparent=True)
        plt.close(fig)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(t, pupils[b][:, 0] * scale[0], pupils[b][:, 1] * scale[1], c=cmap((2 * ii + 1) / 20), label="pupil")
        ax.plot(t, tutors[b][:, 0] * scale[0], tutors[b][:, 1] * scale[1], c=cmap(2 * ii / 20), label="tutor")
        ax.set_xlabel("Time within motif"); ax.set_ylabel("omega"); ax.set_zlabel("gamma")
        ax.legend()
        fig.savefig(os.path.join(plot_dir, f"{b}_mean_trajectories_3d.svg"), transparent=True)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(0.8 * n + 3, 0.8 * n + 2))
    g = ax.imshow(D, cmap="viridis_r")
    ax.set_xticks(range(n)); ax.set_xticklabels([f"{b}\ntutor" for b in birds], fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels([f"{b} pupil" for b in birds], fontsize=7)
    fig.colorbar(g, ax=ax, label="trajectory distance (sd units)")
    ax.set_title(f"pupil-tutor distances (permutation p = {p_value:.3f})")
    fig.savefig(os.path.join(plot_dir, "distance_matrix.svg"), transparent=True, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fire.Fire({"train": train, "embed": embed, "compare": compare})
