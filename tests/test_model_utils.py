import torch

from model.model_utils import smooth


def test_smooth_preserves_shape():
    x = torch.randn(2, 100, 3)
    assert smooth(x, 7).shape == x.shape


def test_smooth_is_trailing_mean_with_zero_prepad():
    x = torch.arange(1.0, 11.0).view(1, 10, 1)
    out = smooth(x, 4)[0, :, 0]
    # steady state: mean of the last 4 samples
    torch.testing.assert_close(out[5], x[0, 2:6, 0].mean())
    # start-up: missing history counts as zeros
    torch.testing.assert_close(out[0], x[0, 0, 0] / 4)
    torch.testing.assert_close(out[1], x[0, :2, 0].sum() / 4)


def test_smooth_is_causal():
    x = torch.randn(1, 60, 1)
    y = x.clone()
    y[:, 30:] += 100  # change the future
    a, b = smooth(x, 5), smooth(y, 5)
    torch.testing.assert_close(a[:, :30], b[:, :30])
    assert not torch.allclose(a[:, 30:], b[:, 30:])


def test_smooth_keeps_constants_after_warmup():
    x = torch.full((1, 50, 1), 3.0)
    out = smooth(x, 8)
    torch.testing.assert_close(out[:, 8:], x[:, 8:])


def test_smooth_length_one_is_identity():
    x = torch.randn(1, 20, 2)
    assert smooth(x, 1) is x
