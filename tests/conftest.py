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


@pytest.fixture
def wav_dataset(tmp_path):
    """
    a small on-disk dataset in the format the loaders expect: one folder of .wav files and
    one folder of .txt files giving onsets and offsets (s) of vocalizations. 8 files of
    3 tonal vocalizations each (24 total), 0.1 s long, at 20 kHz
    """
    from scipy.io import wavfile

    fs = 20000
    audio_dir, seg_dir = tmp_path / "audio", tmp_path / "segs"
    audio_dir.mkdir()
    seg_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(8):
        y = 0.002 * rng.standard_normal(fs)  # 1 s of quiet background
        onoffs = []
        for j in range(3):
            on = 0.1 + 0.3 * j
            t = np.arange(int(0.1 * fs)) / fs
            f = 2000 + 500 * j + 100 * i
            voc = 0.3 * np.sin(2 * np.pi * f * t) * np.hanning(len(t))
            start = int(on * fs)
            y[start : start + len(voc)] += voc
            onoffs.append((on, on + 0.1))
        wavfile.write(str(audio_dir / f"rec{i}.wav"), fs, (y * 32767).astype(np.int16))
        np.savetxt(str(seg_dir / f"rec{i}.txt"), np.array(onoffs), fmt="%.4f")
    return {"audio_dir": str(audio_dir), "seg_dir": str(seg_dir), "fs": fs, "n_vocs": 24}


@pytest.fixture
def tones():
    """24 tonal vocalizations of 0.1 s at 20 kHz, with distinct frequencies"""
    fs = 20000
    t = np.arange(int(0.1 * fs)) / fs
    audio = [
        0.3 * np.sin(2 * np.pi * (1500 + 250 * i) * t) * np.hanning(len(t)) for i in range(24)
    ]
    return audio, fs
