from train.train import train, save_model, load_model
from model.kernels import fullPolyModule
from model.model import Ouroboros
from utils import sse
from visualization.model_vis import loss_plot
from train.eval import eval_model_error

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

import seaborn as sns


def model_cv_lambdas(
    dls: dict,
    dt: float,
    n_epochs: int = 100,
    lr: float = 1e-3,
    n_kernels: int = 15,
    expand_factor: int = 10,
    n_layers: int = 4,
    d_state: int = 1,
    d_conv: int = 4,
    tau: float = 1 / 1000,
    smooth_len: float = 0.001,
    model_path: str = "",
    save_freq: int = 5,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    This function trains models and cross-validates across regularization strengths.
    We pick the minimum regularization strength to be 1.01 (to more heavily penalize
    more complex nonlinearities) and the largest to be 10**(4/(2*n_kernels)) (so that the regularization weight
    on the most complex term is 10**4). Saves these in a larger folder,
    alongside train stats, training plots, etc.

    inputs
    -----
        - dls: dictionary of dataloaders, train and test
        - dt: spacing between audio samples, in seconds
        - nEpochs: number of full passes through the dataset to train the model for
        - lr: learning rate
        - n_kernels: maximum polynomial degree for the nonlinearity
        - expand_factor: expansion from audio to mamba input
        - n_layers: number of mamba layers in encoder
        - d_state: internal state size of mamba model
        - d_conv: length of internal convolution of mamba model
        - tau: timescale for model decoder, to de-dimensionalize the data
        - smooth_len: smoothing length for model functions. not used in training
        - model_path: place to save all the models
        - save_freq: how often to save out our models
        - device: device to train on. falls back to cpu if cuda is requested but unavailable

    returns
    -----
        - the best model (on the test set)
    """

    model_info = {
        "n layers": n_layers,
        "d state": d_state,
        "d conv": d_conv,
        "expand factor": expand_factor,
    }

    min_lambda = 1.01
    max_lambda = 10 ** (4 / (2 * n_kernels))

    lambdas = np.linspace(min_lambda, max_lambda, 7)
    #
    lam_train_cv_err = []
    lam_test_cv_err = []

    lam_train_cv_sd = []
    lam_test_cv_sd = []

    lam_train_cv_r2 = []
    lam_test_cv_r2 = []

    for ii, lam in enumerate(lambdas):
        print(f"Regularizing with lambda={lam}")

        kernel = fullPolyModule(
            nTerms=n_kernels,
            device=device,
            x_dim=1,
            z_dim=2,
            activation=lambda x: x,
            lam=lam,
        )
        reg_weights = True
        full_model_poly = Ouroboros(
            d_data=1,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            tau=tau,
            smooth_len=smooth_len,
            kernel=kernel,
            device=device,
        )

        full_opt_poly = Adam(full_model_poly.parameters(), lr=lr)
        full_scheduler_poly = ReduceLROnPlateau(
            full_opt_poly, factor=0.5, patience=max(n_epochs // 25, 2), min_lr=1e-10
        )
        model_path_full_poly = (
            model_path + f"/kernelborous_poly_end_to_end_lambda_{lam}"
        )
        save_loc_poly = model_path_full_poly + f"/checkpoint_{n_epochs}.tar"
        save_files = glob.glob(os.path.join(model_path_full_poly, "*.tar"))

        start_epoch = 0
        if len(save_files) > 0:
            full_model_poly, full_opt_poly, full_scheduler_poly, start_epoch = (
                load_model(model_path_full_poly, device=device)
            )

        if start_epoch < n_epochs:
            tl, vl, full_model_poly, full_opt_poly = train(
                full_model_poly,
                full_opt_poly,
                loss_fn=lambda y, yhat: sse(yhat, y, reduction="mean"),
                loaders=dls,
                scheduler=full_scheduler_poly,
                nEpochs=n_epochs,
                val_freq=1,
                runDir=model_path_full_poly,
                dt=dt,
                vis_freq=max(n_epochs // 10, 1),
                smoothing=False,
                reg_weights=reg_weights,
                start_epoch=start_epoch,
                save_freq=save_freq,
                model_info=model_info,
            )

            loss_plot(tl, vl, save_loc=model_path_full_poly, show=False)

            save_model(
                full_model_poly,
                full_opt_poly,
                save_loc_poly,
                n_layers=n_layers,
                d_state=d_state,
                expand_factor=expand_factor,
                d_conv=d_conv,
            )

        full_model_poly.eval()
        with torch.no_grad():
            (train_mu, test_mu), (train_sd, test_sd), (train_r2, test_r2) = (
                eval_model_error(dls, full_model_poly, dt=dt)
            )
        lam_train_cv_err.append(train_mu)
        lam_test_cv_err.append(test_mu)

        lam_train_cv_sd.append(train_sd)
        lam_test_cv_sd.append(test_sd)

        lam_train_cv_r2.append(train_r2)
        lam_test_cv_r2.append(test_r2)

    splits = ["train"] * len(lam_train_cv_err) + ["val"] * len(lam_test_cv_err)
    lambdas_stacked = np.round(np.hstack([lambdas, lambdas]), 3)
    errs = np.hstack([lam_train_cv_err, lam_test_cv_err])
    df = pd.DataFrame({"lam": lambdas_stacked, "split": splits, "R2": errs})

    min_err_ind = np.argmax(lam_test_cv_err)  # argmax, since 'err' is actually r2
    print(f"best R2 alpha for {n_kernels} kernels: {lambdas[min_err_ind]}")
    ax = plt.gca()

    sns.boxplot(
        data=df,
        x="lam",
        y="R2",
        hue="split",
        hue_order=["train", "test"],
        ax=ax,
        gap=0.1,
    )

    ax.set_xlabel("Polynomial degree penalty")
    ax.set_ylabel(r"$R^2$")
    ylim = ax.get_ylim()
    ylim = (min(ylim[0], 0), max(ylim[-1], 1.01))
    ax.set_ylim(ylim)
    ax.legend()
    plt.savefig(
        os.path.join(model_path, "train_test_error_kernel_poly_nkernels_30.svg")
    )
    plt.close()

    model_path_best = (
        model_path + f"/kernelborous_poly_end_to_end_lambda_{lambdas[min_err_ind]}"
    )

    full_model_poly, full_opt_poly, full_scheduler_poly, _ = load_model(model_path_best, device=device)
    full_model_poly.eval()
    with torch.no_grad():
        (train_mu, test_mu), (train_sd, test_sd), (train_r2, test_r2) = (
            eval_model_error(dls, full_model_poly, dt=dt, comparison="test")
        )

    data_df = pd.DataFrame(
        {"lambdas": lambdas, "train MSE": lam_train_cv_err, "test MSE": lam_test_cv_err}
    )
    data_df.to_csv(os.path.join(model_path, "cv_errs.csv"))

    return full_model_poly
