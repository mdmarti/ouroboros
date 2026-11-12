import glob
import os

import numpy as np
import pytest
import torch
from torch.optim import Adam

from data.data_utils import get_loaders
from model.kernels import fullPolyModule
from train.eval import (
    correct,
    eval_model_error,
    integrate_estimated_d2,
    integrate_model_d2,
    integrate_second_deriv,
)
from train.train import load_model, save_model, train
from utils import deriv_approx_d2y, deriv_approx_dy, sse

MODEL_INFO = {"n layers": 1, "d state": 2, "d conv": 4, "expand factor": 2}


@pytest.fixture
def audio():
    """a few short noisy chirp-ish chunks: N x L x 1"""
    t = np.arange(300) / 20000
    rng = np.random.default_rng(0)
    return np.stack(
        [
            np.sin(2 * np.pi * (800 + 200 * i) * t + rng.uniform(0, 6))[:, None]
            + 0.01 * rng.standard_normal((300, 1))
            for i in range(20)
        ]
    )


@pytest.fixture
def loaders(audio):
    return get_loaders(audio, num_workers=0, batch_size=4, train_size=0.6, cv=True, seed=0)


@pytest.fixture
def small(make_model):
    return make_model(n_layers=1, d_state=2, n_terms=3)


# ------------------------------------------------------------------ derivative inputs


def test_derivative_stencils_exact_on_polynomials():
    t = np.arange(60, dtype=float)
    y = (0.01 * t**3 - 0.2 * t**2 + t)[None, :, None]
    dy_true = 0.03 * t**2 - 0.4 * t + 1
    d2y_true = 0.06 * t - 0.4
    inner = slice(4, -4)  # edges use reflection padding
    np.testing.assert_allclose(deriv_approx_dy(y)[0, inner, 0], dy_true[inner], atol=1e-8)
    np.testing.assert_allclose(deriv_approx_d2y(y)[0, inner, 0], d2y_true[inner], atol=1e-8)


def test_derivative_stencils_on_a_sinusoid():
    w = 0.3
    t = np.arange(200, dtype=float)
    y = np.sin(w * t)[None, :, None]
    inner = slice(5, -5)
    np.testing.assert_allclose(
        deriv_approx_dy(y)[0, inner, 0], w * np.cos(w * t)[inner], atol=1e-6
    )
    np.testing.assert_allclose(
        deriv_approx_d2y(y)[0, inner, 0], -(w**2) * np.sin(w * t)[inner], atol=1e-6
    )


def test_derivative_output_keeps_shape():
    y = np.random.randn(2, 50, 1)
    assert deriv_approx_dy(y).shape == y.shape
    assert deriv_approx_d2y(y).shape == y.shape


# ------------------------------------------------------------------- eval_model_error


def test_eval_model_error(small, loaders, dt):
    (tr_mu, te_mu), (tr_sd, te_sd), (tr_r2, te_r2) = eval_model_error(loaders, small, dt)
    assert np.isfinite([tr_mu, te_mu, tr_sd, te_sd]).all()
    assert len(tr_r2) == 12 and len(te_r2) == 4  # 60/20/20 split of 20 chunks; 'val' used
    assert np.isclose(tr_r2.mean(), tr_mu) and np.isclose(te_r2.mean(), te_mu)
    assert (tr_r2 <= 1).all()  # r2 can be very negative for an untrained model, never > 1


def test_eval_model_error_can_use_test_split(small, loaders, dt):
    *_, (_, val_r2) = eval_model_error(loaders, small, dt, comparison="val")
    *_, (_, test_r2) = eval_model_error(loaders, small, dt, comparison="test")
    assert len(val_r2) == len(test_r2) == 4
    assert not np.allclose(val_r2, test_r2)


def test_eval_model_error_is_one_for_a_perfect_prediction(loaders, dt):
    # r2 formula check, via a stub that returns the exact scaled second derivative
    class Perfect(torch.nn.Module):
        tau = dt

        def forward(self, x, dxdt, dt_):
            d2 = torch.from_numpy(deriv_approx_d2y(x.numpy().astype(float))).float()
            return d2 / dt_**2 * self.tau**2, None

        @property
        def device(self):
            return torch.device("cpu")

    (_, te_mu), *_ = eval_model_error(loaders, Perfect(), dt)
    assert te_mu == pytest.approx(1.0, abs=1e-4)


# ------------------------------------------------------------------------ train


def test_train_runs_and_improves(small, loaders, dt, tmp_path):
    small.train()
    opt = Adam(small.parameters(), lr=1e-2)
    run_dir = str(tmp_path / "run")
    tl, vl, model, _ = train(
        small,
        opt,
        loss_fn=lambda y, yhat: sse(yhat, y, reduction="mean"),
        loaders=loaders,
        nEpochs=6,
        val_freq=1,
        runDir=run_dir,
        dt=dt,
        vis_freq=0,
        reg_weights=True,
        save_freq=2,
        model_info=MODEL_INFO,
    )
    assert model is small
    assert len(tl) == 6 * len(loaders["train"])
    assert len(vl) == 6 and all(len(v) == 3 for v in vl)  # (step, val loss, penalty)
    assert np.isfinite(tl).all()
    assert np.mean(tl[-3:]) < np.mean(tl[:3])
    assert glob.glob(os.path.join(run_dir, "checkpoint_*.tar"))


def test_train_with_default_save_freq_does_not_crash(small, loaders, dt, tmp_path):
    # save_freq=0 means "never checkpoint"; used to raise ZeroDivisionError
    train(
        small,
        Adam(small.parameters(), lr=1e-3),
        loss_fn=lambda y, yhat: sse(yhat, y, reduction="mean"),
        loaders=loaders,
        nEpochs=1,
        val_freq=1,
        runDir=str(tmp_path),
        dt=dt,
        vis_freq=0,
    )


def test_regularisation_penalises_complex_terms(small, loaders, dt, tmp_path):
    # with the penalty on, validation penalty is reported and is non-negative
    _, vl, *_ = train(
        small,
        Adam(small.parameters(), lr=1e-3),
        loss_fn=lambda y, yhat: sse(yhat, y, reduction="mean"),
        loaders=loaders,
        nEpochs=1,
        val_freq=1,
        runDir=str(tmp_path),
        dt=dt,
        vis_freq=0,
        reg_weights=True,
    )
    assert vl[0][2] > 0


# ---------------------------------------------------------------- save/load models


def test_save_and_load_model_roundtrip(small, batch, dt, tmp_path):
    opt = Adam(small.parameters(), lr=1e-3)
    save_model(
        small, opt, str(tmp_path / "checkpoint_7.tar"), n_layers=1, d_state=2, d_conv=4, expand_factor=2
    )
    loaded, _, _, epoch = load_model(str(tmp_path), device="cpu")
    loaded.eval()
    assert epoch == 7
    assert loaded.kernel.nTerms == small.kernel.nTerms
    assert loaded.tau == small.tau
    assert loaded.device == torch.device("cpu")
    x, dxdt = batch
    a, _ = small(x, dxdt, dt)
    b, _ = loaded(x, dxdt, dt)
    torch.testing.assert_close(a, b)


def test_load_model_picks_the_latest_checkpoint(small, dt, tmp_path):
    opt = Adam(small.parameters(), lr=1e-3)
    for e in (3, 12, 5):
        save_model(
            small, opt, str(tmp_path / f"checkpoint_{e}.tar"), n_layers=1, d_state=2, d_conv=4, expand_factor=2
        )
    *_, epoch = load_model(str(tmp_path), device="cpu")
    assert epoch == 12


def test_save_model_keeps_only_recent_checkpoints(small, tmp_path):
    opt = Adam(small.parameters(), lr=1e-3)
    for e in range(1, 9):
        save_model(
            small, opt, str(tmp_path / f"checkpoint_{e}.tar"), n_layers=1, d_state=2,
            d_conv=4, expand_factor=2, max_saved=3,
        )
    kept = sorted(int(os.path.basename(f)[11:-4]) for f in glob.glob(str(tmp_path / "*.tar")))
    assert kept == [6, 7, 8]


def test_load_model_falls_back_to_cpu_without_cuda(small, tmp_path, monkeypatch):
    save_model(
        small, Adam(small.parameters()), str(tmp_path / "checkpoint_1.tar"),
        n_layers=1, d_state=2, d_conv=4, expand_factor=2,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(RuntimeWarning):
        model, *_ = load_model(str(tmp_path), device="cuda")
    assert model.device == torch.device("cpu")


# --------------------------------------------------------------------- integration


def test_correct_removes_slow_trends_and_keeps_fast_oscillations():
    n = 4000
    t = np.arange(n) / n
    fast = np.sin(2 * np.pi * 400 * t)  # 400 cycles per window
    drift = 5 * t**2 + 3
    out = correct(fast + drift)
    assert out.shape == (n,)
    mid = slice(500, -500)  # filter edge effects
    assert np.corrcoef(out[mid], fast[mid])[0, 1] > 0.99
    assert abs(out[mid].mean()) < 0.1


def test_integrate_second_deriv_recovers_a_sinusoid():
    dt, n, f = 1 / 20000, 2000, 2000.0  # 200 cycles in the window
    t = np.arange(n) * dt
    w = 2 * np.pi * f
    y = np.sin(w * t)
    d2y = -(w**2) * y
    ic = torch.tensor([y[0], w * np.cos(0.0)], dtype=torch.float32)
    out = integrate_second_deriv(d2y, ic, t, method="rk4", verbose=False)
    assert out.shape == (n,)
    mid = slice(200, -200)
    assert np.corrcoef(out[mid], y[mid])[0, 1] > 0.99


def test_integrate_estimated_d2_matches_the_audio():
    dt, n, f = 1 / 20000, 2000, 2000.0
    t = np.arange(n) * dt
    y = np.sin(2 * np.pi * f * t).astype(np.float32)
    out = integrate_estimated_d2(y, dt, verbose=False, device="cpu")
    mid = slice(200, -200)
    assert np.corrcoef(out[mid], y[mid])[0, 1] > 0.95


def test_integrate_model_d2_runs_and_respects_switches(small, dt):
    n = 1500
    t = np.arange(n) * dt
    y = (0.1 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
    full = integrate_model_d2(small, y, dt, verbose=False)
    no_nl = integrate_model_d2(small, y, dt, use_nonlinearity=False, verbose=False)
    null = integrate_model_d2(small, y, dt, null_comparison=True, verbose=False)
    for o in (full, no_nl, null):
        assert o.shape == (n,) and np.isfinite(o).all()
    assert not np.allclose(full, no_nl)
    assert not np.allclose(full, null)
