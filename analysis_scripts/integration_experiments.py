"""
Integration experiments: how well does integrating a method's predicted second derivative reproduce the audio?

    compare_external
        Converted from notebooks/integration_comparison_me_ziyi.ipynb. Integrates the omega/gamma
        trajectories from three methods (Ouroboros, "direct NP" = node perturbation, "ours") for the
        same vocalizations, saves the integrated audio and a spectrogram comparison figure.
        Inputs are the two pickles from the notebook: one with the original audio and Ouroboros'
        'audio', 'omegas', 'gammas'; one with gen['direct NP'] / gen['ours'] entries holding 'omega', 'gamma'.

    model_vs_baselines
        NEW (not in a notebook; plan item A6, a quantitative integration metric). For a trained model and a
        folder of segmented audio, integrates (a) the model's second derivative, (b) a null model with
        omega = gamma = 1, and (c) the finite-difference second derivative, and reports each one's
        correlation and R^2 against the original audio (high-pass filtered the same way the integration
        is, since `integrate_second_deriv` removes slow trends), plus log-spectrogram distance.

    python -m analysis_scripts.integration_experiments compare_external \
        --orig_pkl example_data.pkl --gen_pkl signal_task_final_gen.pkl --save-dir SAVE
    python -m analysis_scripts.integration_experiments model_vs_baselines \
        --model-path M --audio-dir A --seg-dir S --save-dir SAVE
"""

import json
import os
import pickle

import fire
import numpy as np
import pandas as pd
import torch

from analysis_scripts.common import load_best_model, plt, resolve_model_path
from data.load_data import get_segmented_audio
from model.model_utils import resolve_device
from train.eval import (
    correct,
    integrate_estimated_d2,
    integrate_model_d2,
    integrate_second_deriv,
)
from utils import deriv_approx_d2y, deriv_approx_dy, get_spec

SPEC_KW = dict(shoulder=0.0, interp=False, win_len=1028, normalize=False,
               min=-2, max=3.5, spec_type="log")


def _r2_corr(target, pred):
    n = min(len(target), len(pred))
    target, pred = target[:n], pred[:n]
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    return 1 - ss_res / ss_tot, np.corrcoef(target, pred)[0, 1]


def _log_spec_distance(a, b, fs):
    """mean absolute difference between the log spectrograms of two signals"""
    L = min(len(a), len(b))
    sa, *_ = get_spec(a[:L], fs, onset=0, offset=L / fs, **SPEC_KW)
    sb, *_ = get_spec(b[:L], fs, onset=0, offset=L / fs, **SPEC_KW)
    return float(np.mean(np.abs(sa - sb)))


def compare_external(
    orig_pkl,
    gen_pkl,
    save_dir,
    fs=44100,
    max_vocs=10,
    max_time_plot=0.15,
    method="rk4",
    device="cuda",
):
    """integrates three methods' omega/gamma for the same vocalizations; saves audio and a spectrogram figure"""
    os.makedirs(save_dir, exist_ok=True)
    device = resolve_device(device)
    with open(orig_pkl, "rb") as f:
        orig = pickle.load(f)
    with open(gen_pkl, "rb") as f:
        gen = pickle.load(f)

    dt = tau = 1 / fs
    methods = {
        "Ouroboros": (orig["omegas"], orig["gammas"]),
        "Node perturbation": (gen["direct NP"]["omega"], gen["direct NP"]["gamma"]),
        "Ours": (gen["ours"]["omega"], gen["ours"]["gamma"]),
    }
    audios = orig["audio"][:max_vocs]
    n = len(audios)
    fig, axs = plt.subplots(4, n, figsize=(7 * n, 12), squeeze=False)
    rows = []

    for ii, y in enumerate(audios):
        print(f"Integrating voc {ii + 1}/{n}")
        L = len(y)
        dy = deriv_approx_dy(y[None, :, None]).squeeze() / dt * tau
        ic = torch.from_numpy(np.hstack([y[0], dy[0] / tau])).to(device)
        eval_times = np.arange(0, L * dt + dt / 2, dt)[:L]

        integrated = {"original": y}
        for name, (omegas, gammas) in methods.items():
            d2 = (-omegas[ii] ** 2 * y - gammas[ii] * dy) / tau**2
            integrated[name] = integrate_second_deriv(d2, ic, eval_times, method=method, verbose=False)
            r2, corr = _r2_corr(correct(y), integrated[name])
            rows.append({"voc": ii, "method": name, "r2": r2, "corr": corr})

        with open(os.path.join(save_dir, f"vocalization_{ii + 1}_audios.pkl"), "wb") as f:
            pickle.dump(integrated, f)

        for row, (name, sig) in enumerate(integrated.items()):
            spec, times, freqs, _ = get_spec(sig, fs, onset=0, offset=L * dt, **SPEC_KW)
            cutoff = int(np.argmax(times > max_time_plot)) or spec.shape[-1]
            axs[row, ii].imshow(
                spec.squeeze()[:, :cutoff], origin="lower", aspect="auto", cmap="viridis",
                extent=[times[0], times[cutoff - 1], freqs[0] / 1000, freqs[-1] / 1000], vmin=-5.5,
            )
            axs[row, ii].set_title(name)
            if row < 3:
                axs[row, ii].set_xticks([])
            else:
                axs[row, ii].set_xlabel("Time (s)")
            if ii == 0:
                axs[row, ii].set_ylabel("Freq (kHz)")

    fig.savefig(os.path.join(save_dir, "all_comparisons.svg"), transparent=True, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "integration_metrics.csv"), index=False)
    print(df.groupby("method")[["r2", "corr"]].agg(["median", "mean"]))


def model_vs_baselines(
    audio_dir,
    seg_dir,
    save_dir,
    model_path="",
    audio_id=".wav",
    n_vocs=50,
    padding=0.05,
    max_len_s=0.5,
    seed=92,
    method="rk4",
    device="cuda",
):
    """
    integration error of the model (`model_path`: a trained model's folder; this script does not train one)
    against a null model (omega = gamma = 1) and the finite-difference
    second derivative. Vocalizations longer than `max_len_s` are skipped (integration is slow).
    """
    os.makedirs(save_dir, exist_ok=True)
    model = load_best_model(resolve_model_path(model_path), device=device)
    audios, sr = get_segmented_audio(
        audio_dir, seg_dir, audio_id=audio_id, max_vocs=n_vocs * 3, seed=seed,
        training=False, padding=padding, shuffle_order=True,
    )
    dt = 1 / sr
    rows = []
    n_done = 0
    for ii, a in enumerate(audios):
        if n_done >= n_vocs:
            break
        y = a.squeeze()
        if len(y) > max_len_s * sr or len(y) < 50:
            continue
        target = correct(y)
        preds = {
            "model": integrate_model_d2(model, y, dt, method=method, verbose=False),
            "null (omega=gamma=1)": integrate_model_d2(model, y, dt, method=method, null_comparison=True, verbose=False),
            "finite-difference d2": integrate_estimated_d2(y, dt, method=method, verbose=False, device=device),
        }
        for name, p in preds.items():
            r2, corr = _r2_corr(target, p)
            rows.append({"voc": ii, "method": name, "r2": r2, "corr": corr,
                         "log_spec_dist": _log_spec_distance(y, p, sr), "n_samples": len(y)})
        n_done += 1

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "integration_vs_baselines.csv"), index=False)
    summary = {
        m: {k: {"median": float(g[k].median()), "iqr": [float(v) for v in g[k].quantile([0.25, 0.75])]}
            for k in ("r2", "corr", "log_spec_dist")}
        for m, g in df.groupby("method")
    }
    summary["n_vocalizations"] = int(df["voc"].nunique())
    with open(os.path.join(save_dir, "integration_vs_baselines.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    order = list(df["method"].unique())
    for ax, k, lab in zip(axs, ("r2", "log_spec_dist"), (r"$R^2$ vs. original", "log-spectrogram distance")):
        ax.boxplot([df[df.method == m][k] for m in order], tick_labels=order)
        ax.set_ylabel(lab)
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(axis="x", rotation=20)
    fig.savefig(os.path.join(save_dir, "integration_vs_baselines.svg"), transparent=True, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fire.Fire({"compare_external": compare_external, "model_vs_baselines": model_vs_baselines})
