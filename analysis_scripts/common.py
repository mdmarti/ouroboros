"""
Helpers shared by the analysis scripts in this folder.

Run the scripts from the repository root so that `train`, `data`, `model` etc. are importable, e.g.

    python -m analysis_scripts.budgie_spike_regression run --model-path ... --data-dir ... --save-dir ...
"""

import glob
import os
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # scripts save figures, they never open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from train.train import load_model
from utils import deriv_approx_dy

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["text.usetex"] = False


def format_axis(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def find_best_checkpoint_dir(model_dir: str) -> str:
    """
    `model_cv_lambdas` writes one folder per regularization strength plus `cv_errs.csv`
    (whose "test MSE" column is actually test R^2, so the best model is the argmax).
    Returns the folder of the best model. If there is no `cv_errs.csv`, `model_dir`
    is assumed to already be a single model's folder.
    """
    csv = os.path.join(model_dir, "cv_errs.csv")
    if not os.path.isfile(csv):
        return model_dir
    df = pd.read_csv(csv)
    best_lambda = df["lambdas"].iloc[int(np.argmax(df["test MSE"].to_numpy()))]
    # folders are named with the full-precision lambda; csv values may be rounded
    candidates = glob.glob(
        os.path.join(model_dir, "kernelborous_poly_end_to_end_lambda_*")
    )
    assert len(candidates) > 0, f"no model folders found in {model_dir}"
    lams = [float(c.split("_lambda_")[-1]) for c in candidates]
    return candidates[int(np.argmin(np.abs(np.array(lams) - best_lambda)))]


def resolve_model_path(model_path: str = "", fallback_dir: str = "") -> str:
    """
    the folder of a trained model: `model_path` if given, else `fallback_dir` (typically `<save-dir>/model`,
    where a script that trains its own model saves it) if that exists. Raises a helpful error otherwise.
    `model_path` may be the folder written by `model_cv_lambdas` (the best model is chosen from `cv_errs.csv`)
    or a single model's folder holding `checkpoint_*.tar` files.
    """
    if model_path:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"--model-path {model_path!r} is not a folder")
        return model_path
    if fallback_dir and os.path.isdir(fallback_dir):
        return fallback_dir
    raise ValueError(
        "no trained model: pass --model-path" + (f" (or train one first; looked in {fallback_dir})" if fallback_dir else "")
    )


def load_best_model(model_dir: str, device: str = "cuda"):
    """loads the best cross-validated model from `model_dir` (see `find_best_checkpoint_dir`)"""
    path = find_best_checkpoint_dir(model_dir)
    model, *_ = load_model(path, device=device)
    model.eval()
    return model


def embed_vocalizations(
    model,
    vocalizations: Sequence[np.ndarray],
    sr: float,
    smooth_len_s: float = 0.025,
    max_len_t: float = 8,
    desc: str = "Embedding vocalizations",
):
    """
    omega and gamma for each vocalization. `vocalizations` are (1, L, 1) arrays.
    Returns two lists of 1d arrays.
    """
    old_smooth_len = model.smooth_len
    model.smooth_len = smooth_len_s
    dt = 1 / sr
    omegas, gammas = [], []
    try:
        for v in tqdm(vocalizations, desc=desc):
            dvdt = deriv_approx_dy(v)
            v_t = torch.from_numpy(v).to(model.device).to(torch.float32)
            dv_t = torch.from_numpy(dvdt).to(model.device).to(torch.float32)
            with torch.no_grad():
                omega, gamma, *_ = model.get_funcs(
                    v_t, dv_t, dt=dt, smoothing=True, max_len_t=max_len_t
                )
            omegas.append(omega.detach().cpu().numpy().squeeze())
            gammas.append(gamma.detach().cpu().numpy().squeeze())
    finally:
        model.smooth_len = old_smooth_len
    return omegas, gammas
