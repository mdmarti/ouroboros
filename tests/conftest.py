"""
shared fixtures for the Ouroboros tests. Everything runs on the cpu with tiny models,
so the whole suite is fast and needs no gpu. Tests that need a gpu are skipped without one.
"""

import numpy as np
import pytest
import torch

from model.kernels import fullPolyModule
from model.model import Ouroboros

DT = 1 / 20000
N_TERMS = 4  # polynomial degree of the test kernels


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def dt():
    return DT


@pytest.fixture
def make_model():
    """factory for small, cpu Ouroboros models"""

    def _make(
        n_terms=N_TERMS, lam=1.1, n_layers=2, d_state=4, expand_factor=2, tau=DT, **kw
    ):
        kernel = fullPolyModule(
            nTerms=n_terms, device="cpu", x_dim=1, z_dim=2, lam=lam
        )
        model = Ouroboros(
            d_data=1,
            kernel=kernel,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=4,
            expand_factor=expand_factor,
            device="cpu",
            tau=tau,
            **kw,
        )
        return model.eval()

    return _make


@pytest.fixture
def model(make_model):
    return make_model()


@pytest.fixture
def batch():
    """(x, dxdt) shaped like model inputs: B x L x 1, dxdt in per-sample units"""
    x = torch.randn(2, 600, 1) * 0.1
    dxdt = torch.randn(2, 600, 1) * 0.01
    return x, dxdt
