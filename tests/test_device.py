import warnings

import pytest
import torch

from model.kernels import fullPolyModule
from model.model import Ouroboros
from model.model_utils import resolve_device

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a gpu"
)


def test_cpu_is_always_available():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning for an explicit cpu request
        assert resolve_device("cpu") == torch.device("cpu")
        assert resolve_device(torch.device("cpu")) == torch.device("cpu")


def test_cuda_unavailable_warns_and_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(RuntimeWarning, match="falling back to cpu"):
        assert resolve_device("cuda") == torch.device("cpu")
    with pytest.warns(RuntimeWarning):
        assert resolve_device("cuda:1") == torch.device("cpu")


def test_cuda_available_is_returned_unchanged(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_device("cuda") == torch.device("cuda")
    assert resolve_device("cuda:1") == torch.device("cuda", 1)


def test_cuda_index_out_of_range_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="only 1 cuda device"):
        resolve_device("cuda:3")


def test_default_device_is_cuda_but_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(RuntimeWarning):
        kernel = fullPolyModule(nTerms=3, device="cuda", x_dim=1, z_dim=2)
    with pytest.warns(RuntimeWarning):
        model = Ouroboros(d_data=1, kernel=kernel, n_layers=1, d_state=2)
    assert model.device == torch.device("cpu")


def test_model_and_kernel_share_the_requested_device(model):
    assert model.device == torch.device("cpu")
    assert model.kernel.device == torch.device("cpu")
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_kernel_is_moved_to_the_model_device(monkeypatch):
    # a kernel built somewhere else must end up where the model lives
    kernel = fullPolyModule(nTerms=3, device="cpu", x_dim=1, z_dim=2)
    model = Ouroboros(d_data=1, kernel=kernel, n_layers=1, d_state=2, device="cpu")
    assert model.kernel is kernel
    assert kernel.powers.device == model.device


def test_powers_buffer_follows_module_but_not_checkpoints():
    kernel = fullPolyModule(nTerms=3, device="cpu", x_dim=1, z_dim=2)
    # a buffer (so .to() moves it), but not saved, so old checkpoints still load
    assert "powers" in dict(kernel.named_buffers())
    assert "powers" not in kernel.state_dict()


@needs_cuda
def test_model_on_gpu_runs(make_model, batch, dt):
    kernel = fullPolyModule(nTerms=3, device="cuda", x_dim=1, z_dim=2)
    model = Ouroboros(
        d_data=1, kernel=kernel, n_layers=1, d_state=2, device="cuda", tau=dt
    ).eval()
    x, dxdt = batch
    yhat, weights = model(x.cuda(), dxdt.cuda(), dt)
    assert yhat.device.type == "cuda" and weights.device.type == "cuda"
    assert model.device.type == "cuda"
