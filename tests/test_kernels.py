import numpy as np
import pytest
import torch

from model.kernels import fullPolyModule

P = 4  # polynomial degree used here


@pytest.fixture
def kernel():
    return fullPolyModule(nTerms=P, device="cpu", x_dim=1, z_dim=2, lam=1.5)


@pytest.fixture
def xz():
    x = torch.randn(3, 50, 2)  # (y, tau * ydot)
    z = torch.randn(3, 50, 2)  # encoder control signal
    return x, z


def one_hot_weights(B, L, p, k):
    w = torch.zeros(B, L, P + 1, P + 1)
    w[:, :, p, k] = 1.0
    return w


def test_forward_shapes(kernel, xz):
    x, z = xz
    out, weights = kernel(x, z)
    assert out.shape == (3, 50, 1)
    assert weights.shape == (3, 50, P + 1, P + 1)
    assert torch.isfinite(out).all()


def test_constant_and_linear_terms_are_removed(kernel, xz):
    # w00 (offset), w10 (y), w01 (ydot) are fixed to zero: omega and gamma own those
    x, z = xz
    _, weights = kernel(x, z)
    assert torch.all(weights[:, :, 0, 0] == 0)
    assert torch.all(weights[:, :, 1, 0] == 0)
    assert torch.all(weights[:, :, 0, 1] == 0)
    assert weights[:, :, 2:, :].abs().sum() > 0  # others are free


@pytest.mark.parametrize("p,k", [(2, 0), (0, 2), (1, 1), (3, 2), (2, 2), (4, 4)])
def test_single_term_is_product_of_powers(kernel, xz, p, k):
    # weight on [p, k] multiplies y^p * (tau ydot)^k
    x, _ = xz
    out = kernel.forward_given_weights(x, one_hot_weights(3, 50, p, k))
    expected = x[:, :, 0] ** p * x[:, :, 1] ** k
    torch.testing.assert_close(out[:, :, 0], expected)


@pytest.mark.parametrize("p,k", [(0, 0), (1, 0), (0, 1)])
def test_removed_terms_do_not_contribute(kernel, xz, p, k):
    x, _ = xz
    out = kernel.forward_given_weights(x, one_hot_weights(3, 50, p, k))
    assert torch.all(out == 0)


def test_output_is_linear_in_weights(kernel, xz):
    x, _ = xz
    w1, w2 = torch.randn(3, 50, P + 1, P + 1), torch.randn(3, 50, P + 1, P + 1)
    a, b = 2.0, -0.5
    lhs = kernel.forward_given_weights(x, (a * w1 + b * w2).clone())
    rhs = a * kernel.forward_given_weights(x, w1.clone()) + b * kernel.forward_given_weights(
        x, w2.clone()
    )
    torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-4)


def test_forward_matches_forward_given_weights(kernel, xz):
    x, z = xz
    out, weights = kernel(x, z)
    again = kernel.forward_given_weights(x, weights.detach().clone())
    torch.testing.assert_close(out, again)


def test_weights_are_a_linear_map_of_the_control(kernel, xz):
    x, z = xz
    _, weights = kernel(x, z)
    expected = kernel.weights(z).view(3, 50, P + 1, P + 1)
    keep = torch.ones(P + 1, P + 1, dtype=torch.bool)
    keep[0, 0] = keep[1, 0] = keep[0, 1] = False
    torch.testing.assert_close(weights[:, :, keep], expected[:, :, keep])


def test_numpy_version_matches_torch(kernel, xz):
    x, _ = xz
    w = torch.randn(3, 50, P + 1, P + 1)
    torch_out = kernel.forward_given_weights(x, w.clone()).numpy()
    np_out = kernel.forward_given_weights_numpy(x.numpy().copy(), w.numpy().copy())
    np.testing.assert_allclose(np_out, torch_out, rtol=1e-4, atol=1e-5)


def test_numpy_version_accepts_flattened_weights(kernel, xz):
    x, _ = xz
    w = torch.randn(3, 50, P + 1, P + 1)
    ref = kernel.forward_given_weights_numpy(x.numpy().copy(), w.numpy().copy())
    flat = kernel.forward_given_weights_numpy(
        x.numpy().copy(), w.numpy().reshape(3, 50, -1).copy()
    )
    np.testing.assert_allclose(flat, ref)


def test_gradients_reach_the_weight_projection(kernel, xz):
    x, z = xz
    out, _ = kernel(x, z)
    out.sum().backward()
    g = kernel.weights.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_get_weights_smooths_causally(kernel):
    z = torch.randn(1, 40, 2)
    w = kernel.get_weights(z, smooth_len=5)
    assert w.shape == (1, 40, (P + 1) ** 2)
