"""the encoder is the bundled third_party Mamba: check it matches the reference behaviour"""

import pytest
import torch

from third_party.mamba import Mamba, MambaConfig


@pytest.fixture
def mamba():
    return Mamba(MambaConfig(d_model=2, n_layers=2, d_state=4, d_conv=4, expand_factor=2)).eval()


def init_cache(m, B):
    return [
        (None, torch.zeros(B, m.config.d_inner, m.config.d_conv - 1)) for _ in m.layers
    ]


def test_forward_shape(mamba):
    assert mamba(torch.randn(3, 50, 2)).shape == (3, 50, 2)


def test_is_causal(mamba):
    x = torch.randn(1, 80, 2)
    y = x.clone()
    y[:, 40:] += 5
    with torch.no_grad():
        a, b = mamba(x), mamba(y)
    torch.testing.assert_close(a[:, :40], b[:, :40], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("chunk", [1, 5, 17, 80])
def test_chunk_step_matches_forward(mamba, chunk):
    x = torch.randn(2, 80, 2)
    with torch.no_grad():
        full = mamba(x)
        cache, outs = init_cache(mamba, 2), []
        for i in range(0, 80, chunk):
            out, cache = mamba.chunk_step(x[:, i : i + chunk], cache)
            outs.append(out)
    torch.testing.assert_close(torch.cat(outs, dim=1), full, atol=1e-5, rtol=1e-4)


def test_state_dict_matches_mambapy_layout():
    # old checkpoints were trained with mambapy: parameter names/shapes must still line up
    mambapy = pytest.importorskip("mambapy.mamba")
    kw = dict(d_model=2, n_layers=3, d_state=1, d_conv=4, expand_factor=10)
    ours = Mamba(MambaConfig(**kw))
    ref = mambapy.Mamba(mambapy.MambaConfig(**kw))
    assert {k: v.shape for k, v in ours.state_dict().items()} == {
        k: v.shape for k, v in ref.state_dict().items()
    }
    ours.load_state_dict(ref.state_dict())
    x = torch.randn(2, 60, 2)
    with torch.no_grad():
        torch.testing.assert_close(ours(x), ref(x))
