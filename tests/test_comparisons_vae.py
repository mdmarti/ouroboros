import glob
import os
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch
from torch.distributions import LowRankMultivariateNormal, Normal
from torch.optim import Adam

import comparisons.vae as cv
from comparisons.vae import (
    SPEC_PARAMS,
    VAE,
    SpectrogramDataset,
    estimate_spec_range,
    eval_model_error,
    get_latents_from_audio,
    get_latents_from_loader,
    get_spectrogram_loaders,
    load_model,
    make_spectrogram,
    model_cv_precisions,
    resolve_spec_params,
    save_model,
    train,
    train_model,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a gpu")

# a fixed intensity range, so that spectrograms do not depend on estimate_spec_range
P = {"spec_min_val": -5.0, "spec_max_val": 8.0}


@pytest.fixture
def vae():
    return VAE(z_dim=8, device="cpu")


@pytest.fixture
def specs():
    g = torch.Generator().manual_seed(0)
    return torch.rand(4, 128, 128, generator=g)


@pytest.fixture
def loaders(tones):
    audio, fs = tones
    return get_spectrogram_loaders(
        audio, fs, num_workers=0, batch_size=8, train_size=0.6, cv=True, seed=0, p=P
    )


# ------------------------------------------------------------------ spectrograms


def test_spectrogram_shape_dtype_range(tones):
    audio, fs = tones
    s = make_spectrogram(audio[0], fs, P)
    assert s.shape == (128, 128) and s.dtype == np.float32
    assert 0 <= s.min() and s.max() <= 1
    assert s.max() > 0.5  # the vocalization is visible


def test_spectrogram_accepts_any_single_channel_shape(tones):
    audio, fs = tones
    a = audio[0]
    ref = make_spectrogram(a, fs, P)
    for shaped in (a[None, :, None], a[:, None], a[None, :]):
        np.testing.assert_allclose(make_spectrogram(shaped, fs, P), ref)


def test_spectrogram_is_zero_for_audio_shorter_than_one_window():
    s = make_spectrogram(np.random.randn(300), 20000, P)
    assert s.shape == (128, 128) and not s.any()


def test_spectrogram_peak_is_at_the_tone_frequency():
    fs, f0 = 20000, 4000.0
    t = np.arange(2000) / fs
    s = make_spectrogram(0.3 * np.sin(2 * np.pi * f0 * t), fs, P)
    # mel-spaced rows, clipped to the Nyquist frequency
    top = min(SPEC_PARAMS["max_freq"], fs / 2)
    mels = np.linspace(cv._mel(SPEC_PARAMS["min_freq"]), cv._mel(top), 128)
    expected_row = np.argmin(np.abs(cv._inv_mel(mels) - f0))
    assert abs(np.argmax(s.mean(axis=1)) - expected_row) <= 2


def test_spectrogram_ignores_dc_offset(tones):
    audio, fs = tones
    np.testing.assert_allclose(
        make_spectrogram(audio[0], fs, P), make_spectrogram(audio[0] + 0.2, fs, P), atol=1e-5
    )


def test_louder_audio_gives_brighter_spectrogram(tones):
    audio, fs = tones
    assert make_spectrogram(4 * audio[0], fs, P).mean() > make_spectrogram(audio[0], fs, P).mean()


def occupied_columns(spec, thresh=0.05):
    cols = np.where(spec.max(axis=0) > thresh)[0]
    return cols.min(), cols.max()


def test_time_stretch_places_short_vocalization_in_the_middle(tones):
    audio, fs = tones
    dur = len(audio[0]) / fs
    # without stretching the image spans max_dur: a vocalization a quarter of that fills the middle quarter
    s = make_spectrogram(audio[0], fs, {**P, "time_stretch": False}, max_dur=4 * dur)
    lo, hi = occupied_columns(s)
    assert 40 <= lo and hi <= 88
    # with stretching, as in AVA, the image spans dur + max_dur - sqrt(dur * max_dur) = 3 dur,
    # so the vocalization takes up a third of the image: wider, and still centered
    s = make_spectrogram(audio[0], fs, {**P, "time_stretch": True}, max_dur=4 * dur)
    lo2, hi2 = occupied_columns(s)
    assert (hi2 - lo2) > (hi - lo) * 1.2
    assert abs((lo2 + hi2) / 2 - 63.5) < 4
    assert 0.25 < (hi2 - lo2 + 1) / 128 < 0.45


def test_spectrogram_without_max_dur_fills_the_window(tones):
    audio, fs = tones
    lo, hi = occupied_columns(make_spectrogram(audio[0], fs, P))
    assert lo < 15 and hi > 112


def test_estimate_spec_range(tones):
    audio, fs = tones
    lo, hi = estimate_spec_range(audio, fs)
    assert np.isfinite([lo, hi]).all() and lo < hi
    # constant audio has a constant spectrogram: still a valid range
    lo, hi = estimate_spec_range([np.ones(3000) * 0.1], fs)
    assert lo < hi
    with pytest.raises(ValueError):
        estimate_spec_range([np.zeros(10)], fs)


def test_tuned_range_makes_use_of_the_full_scale(tones):
    audio, fs = tones
    p = resolve_spec_params(audio, fs, tune_range=True)
    s = np.stack([make_spectrogram(a, fs, p) for a in audio])
    assert s.max() == pytest.approx(1.0) and s.min() == 0.0
    assert 0.0 < (s > 0).mean() < 1.0  # some background, some signal


def test_resolve_spec_params(tones):
    audio, fs = tones
    longer = np.zeros(4000)
    p = resolve_spec_params(audio + [longer], fs)
    assert p["max_dur"] == pytest.approx(4000 / fs)
    assert SPEC_PARAMS["max_dur"] is None  # the module defaults are not modified
    assert resolve_spec_params(audio, fs, {"max_dur": 0.5})["max_dur"] == 0.5


# ---------------------------------------------------------------------- loaders


def test_loaders_split_like_get_loaders(loaders):
    assert set(loaders) == {"train", "val", "test"}
    sizes = {k: len(v.dataset) for k, v in loaders.items()}
    assert sizes == {"train": 14, "val": 5, "test": 5}  # 60 / 20 / 20 of 24, as in get_loaders


def test_loaders_without_cv_have_no_validation_set(tones):
    audio, fs = tones
    dls = get_spectrogram_loaders(audio, fs, num_workers=0, train_size=0.75, p=P)
    assert set(dls) == {"train", "test"}
    assert len(dls["train"].dataset) == 18 and len(dls["test"].dataset) == 6


def test_loader_batches(loaders):
    x = next(iter(loaders["train"]))
    assert x.shape == (8, 128, 128) and x.dtype == torch.float32


def test_loader_split_is_reproducible_by_seed(tones):
    audio, fs = tones
    kw = dict(num_workers=0, cv=True, train_size=0.6, p=P)
    a = get_spectrogram_loaders(audio, fs, seed=3, **kw)
    b = get_spectrogram_loaders(audio, fs, seed=3, **kw)
    c = get_spectrogram_loaders(audio, fs, seed=4, **kw)
    assert torch.equal(a["test"].dataset.specs, b["test"].dataset.specs)
    assert not torch.equal(a["test"].dataset.specs, c["test"].dataset.specs)


def test_only_the_training_loader_shuffles(loaders):
    assert isinstance(loaders["train"].sampler, torch.utils.data.RandomSampler)
    for k in ("val", "test"):
        assert isinstance(loaders[k].sampler, torch.utils.data.SequentialSampler)


def test_loaders_fill_in_max_dur_from_the_data(tones):
    audio, fs = tones
    long = np.zeros(3000)
    dls = get_spectrogram_loaders(audio + [long], fs, num_workers=0, p=P)  # no error
    assert len(dls["train"].dataset) + len(dls["test"].dataset) == 25


def test_spectrogram_dataset():
    ds = SpectrogramDataset(np.random.rand(5, 128, 128))
    assert len(ds) == 5 and ds[2].shape == (128, 128) and ds[2].dtype == torch.float32


# ------------------------------------------------------------------ architecture


def test_matches_the_ava_architecture(vae):
    convs = [vae.conv1, vae.conv2, vae.conv3, vae.conv4, vae.conv5, vae.conv6, vae.conv7]
    assert [(c.in_channels, c.out_channels) for c in convs] == [
        (1, 8), (8, 8), (8, 16), (16, 16), (16, 24), (24, 24), (24, 32),
    ]  # fmt: skip
    assert [c.stride[0] for c in convs] == [1, 2, 1, 2, 1, 2, 1]
    assert all(c.kernel_size == (3, 3) and c.padding == (1, 1) for c in convs)
    convts = [vae.convt1, vae.convt2, vae.convt3, vae.convt4, vae.convt5, vae.convt6, vae.convt7]
    assert [(c.in_channels, c.out_channels) for c in convts] == [
        (32, 24), (24, 24), (24, 16), (16, 16), (16, 8), (8, 8), (8, 1),
    ]  # fmt: skip
    assert [c.stride[0] for c in convts] == [1, 2, 1, 2, 1, 2, 1]
    z = vae.z_dim
    fcs = [vae.fc1, vae.fc2, vae.fc31, vae.fc32, vae.fc33, vae.fc41, vae.fc42, vae.fc43,
           vae.fc5, vae.fc6, vae.fc7, vae.fc8]  # fmt: skip
    assert [(f.in_features, f.out_features) for f in fcs] == [
        (8192, 1024), (1024, 256), (256, 64), (256, 64), (256, 64), (64, z), (64, z), (64, z),
        (z, 64), (64, 256), (256, 1024), (1024, 8192),
    ]  # fmt: skip
    # 7 batch norms in the encoder (the first on the input image), 7 in the decoder
    assert [vae.get_submodule(f"bn{i}").num_features for i in range(1, 15)] == [
        1, 8, 8, 16, 16, 24, 24, 32, 24, 24, 16, 16, 8, 8,
    ]  # fmt: skip


def test_parameter_count_is_stable():
    # guards against accidental changes to the network; 17.4M parameters at z_dim = 8
    assert sum(p.numel() for p in VAE(z_dim=8, device="cpu").parameters()) == 17_420_107


@pytest.mark.parametrize("z_dim", [8, 32, 64])
def test_encode_decode_shapes(z_dim, specs):
    m = VAE(z_dim=z_dim, device="cpu")
    mu, u, d = m.encode(specs)
    assert mu.shape == (4, z_dim) and u.shape == (4, z_dim, 1) and d.shape == (4, z_dim)
    assert (d > 0).all()  # a valid covariance
    assert m.decode(mu).shape == (4, 128 * 128)


def test_forward_shapes_and_finite_loss(vae, specs):
    rec, loss = vae(specs)
    assert rec.shape == specs.shape
    assert loss.shape == () and torch.isfinite(loss)


def test_forward_is_stochastic_but_encoding_is_not(vae, specs):
    vae.eval()
    a, la = vae(specs)
    b, lb = vae(specs)
    assert not torch.allclose(a, b)  # a fresh latent sample each time
    mu1 = vae.get_latents(specs)
    mu2 = vae.get_latents(specs)
    torch.testing.assert_close(mu1, mu2)
    torch.testing.assert_close(vae.reconstruct(specs), vae.reconstruct(specs))


# -------------------------------------------------------------------------- ELBO


def reference_neg_elbo(vae, x):
    """the ELBO from torch.distributions, for a fixed random seed"""
    vae.eval()
    with torch.no_grad():
        mu, u, d = vae.encode(x)
        torch.manual_seed(5)
        dist = LowRankMultivariateNormal(mu, u, d)
        z = dist.rsample()
        rec = vae.decode(z)
        prior = Normal(0.0, 1.0).log_prob(z).sum()
        lik = Normal(rec, 1 / np.sqrt(vae.model_precision)).log_prob(x.reshape(len(x), -1)).sum()
        return -(prior + lik + dist.entropy().sum()), z.shape


def test_loss_is_the_negative_elbo_for_one_sample(vae, specs):
    x = specs[:1]
    ref, _ = reference_neg_elbo(vae, x)
    vae.eval()
    torch.manual_seed(5)
    with torch.no_grad():
        _, loss = vae(x)
    torch.testing.assert_close(loss, ref, rtol=1e-5, atol=1e-2)


def test_loss_matches_ava_convention_for_batches(vae, specs):
    # as in AVA, the normalizing constants of the prior and of the observation model are counted
    # once per batch, not once per sample. They do not affect the gradients
    ref, (B, z_dim) = reference_neg_elbo(vae, specs)
    vae.eval()
    torch.manual_seed(5)
    with torch.no_grad():
        _, loss = vae(specs)
    constants = 0.5 * z_dim * np.log(2 * np.pi) + 0.5 * 128 * 128 * np.log(
        2 * np.pi / vae.model_precision
    )
    torch.testing.assert_close(loss, ref - (B - 1) * constants, rtol=1e-5, atol=1e-2)


def test_higher_precision_weights_reconstruction_more(specs):
    losses = []
    for prec in (1.0, 100.0):
        m = VAE(z_dim=8, model_precision=prec, device="cpu")
        torch.manual_seed(0)
        m.load_state_dict(VAE(z_dim=8, device="cpu").state_dict())  # identical weights
        m.eval()
        torch.manual_seed(1)
        with torch.no_grad():
            _, loss = m(specs)
        losses.append(loss.item())
    assert losses[0] != losses[1]


def test_gradients_reach_every_parameter(vae, specs):
    vae.train()
    _, loss = vae(specs)
    loss.backward()
    for name, p in vae.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


def test_get_latents_return_posterior(vae, specs):
    vae.eval()
    mu, u, d = vae.get_latents(specs, return_posterior=True)
    torch.testing.assert_close(mu, vae.get_latents(specs))
    assert u.shape == (4, 8, 1) and (d > 0).all()


# ----------------------------------------------------------------------- devices


def test_device_property_and_fallback(monkeypatch):
    m = VAE(z_dim=4, device="cpu")
    assert m.device == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(RuntimeWarning, match="falling back to cpu"):
        m = VAE(z_dim=4)  # default is cuda
    assert m.device == torch.device("cpu")


@needs_cuda
def test_vae_runs_on_gpu(specs):
    m = VAE(z_dim=4, device="cuda")
    rec, loss = m(specs.cuda())
    assert rec.device.type == "cuda" and m.device.type == "cuda"


# ---------------------------------------------------------------------- training


def test_train_runs_and_improves(vae, loaders, tmp_path):
    opt = Adam(vae.parameters(), lr=1e-3)
    run_dir = str(tmp_path / "run")
    tl, vl, model, _ = train(vae, opt, loaders, nEpochs=6, runDir=run_dir, save_freq=3)
    assert model is vae
    assert len(tl) == 6 * len(loaders["train"])
    assert len(vl) == 6 and all(len(v) == 2 for v in vl)  # (gradient step, val loss)
    assert np.isfinite(tl).all() and np.isfinite([v[1] for v in vl]).all()
    assert np.mean(tl[-3:]) < np.mean(tl[:3])
    assert sorted(os.path.basename(f) for f in glob.glob(run_dir + "/*.tar")) == [
        "checkpoint_3.tar",
        "checkpoint_6.tar",
    ]


def test_validation_frequency_and_scheduler(vae, loaders, tmp_path):
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    opt = Adam(vae.parameters(), lr=1e-3)
    sched = ReduceLROnPlateau(opt, factor=0.5, patience=0)
    _, vl, *_ = train(
        vae, opt, loaders, scheduler=sched, nEpochs=4, val_freq=2, runDir=str(tmp_path)
    )
    assert len(vl) == 2  # epochs 0 and 2
    assert [v[0] for v in vl] == [0, 2 * len(loaders["train"])]


def test_visualization_is_saved(vae, loaders, tmp_path):
    train(vae, Adam(vae.parameters()), loaders, nEpochs=2, runDir=str(tmp_path), vis_freq=1)
    assert len(glob.glob(str(tmp_path / "reconstructions_*.svg"))) == 2


def test_training_can_be_resumed_from_a_checkpoint(vae, loaders, tmp_path):
    run_dir = str(tmp_path)
    train(vae, Adam(vae.parameters(), lr=1e-3), loaders, nEpochs=2, runDir=run_dir, save_freq=1)
    model, opt, sched, epoch = load_model(run_dir, device="cpu")
    assert epoch == 2  # epochs completed
    tl, vl, *_ = train(model, opt, loaders, nEpochs=3, runDir=run_dir, start_epoch=epoch)
    assert len(tl) == len(loaders["train"]) and len(vl) == 1  # only the one new epoch


def test_no_checkpoints_by_default(vae, loaders, tmp_path):
    train(vae, Adam(vae.parameters()), loaders, nEpochs=1, runDir=str(tmp_path))
    assert not glob.glob(str(tmp_path / "*.tar"))


# ----------------------------------------------------------------- save and load


def test_save_and_load_roundtrip(loaders, specs, tmp_path):
    m = VAE(z_dim=6, model_precision=3.0, device="cpu", spec_params={**P, "max_dur": 0.2})
    opt = Adam(m.parameters(), lr=1e-3)
    train(m, opt, loaders, nEpochs=1, runDir=str(tmp_path))  # so optimizer state exists
    save_model(m, opt, str(tmp_path / "checkpoint_9.tar"))
    loaded, opt2, _, epoch = load_model(str(tmp_path), device="cpu")
    assert epoch == 9
    assert (loaded.z_dim, loaded.model_precision) == (6, 3.0)
    assert loaded.spec_params == m.spec_params and loaded.spec_params["max_dur"] == 0.2
    assert loaded.device == torch.device("cpu")
    assert len(opt2.state_dict()["state"]) == len(opt.state_dict()["state"]) > 0
    m.eval()
    loaded.eval()
    torch.testing.assert_close(m.reconstruct(specs), loaded.reconstruct(specs))


def test_load_picks_the_latest_checkpoint(vae, tmp_path):
    opt = Adam(vae.parameters())
    for e in (3, 12, 5):
        save_model(vae, opt, str(tmp_path / f"checkpoint_{e}.tar"))
    assert load_model(str(tmp_path), device="cpu")[3] == 12


def test_save_keeps_only_recent_checkpoints(vae, tmp_path):
    opt = Adam(vae.parameters())
    for e in range(1, 9):
        save_model(vae, opt, str(tmp_path / f"checkpoint_{e}.tar"), max_saved=3)
    kept = sorted(
        int(os.path.basename(f)[11:-4]) for f in glob.glob(str(tmp_path / "*.tar"))
    )
    assert kept == [6, 7, 8]


def test_load_without_checkpoints_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path), device="cpu")


def test_load_falls_back_to_cpu_without_cuda(vae, tmp_path, monkeypatch):
    save_model(vae, Adam(vae.parameters()), str(tmp_path / "checkpoint_1.tar"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(RuntimeWarning):
        m, *_ = load_model(str(tmp_path), device="cuda")
    assert m.device == torch.device("cpu")


def test_checkpoint_layer_names_match_ava(vae, tmp_path):
    # state dict keys are AVA's layer names, so AVA's per-layer weights can be loaded layer by layer
    layers = {k.split(".")[0] for k in vae.state_dict()}
    expected = (
        {f"conv{i}" for i in range(1, 8)}
        | {f"convt{i}" for i in range(1, 8)}
        | {f"bn{i}" for i in range(1, 15)}
        | {"fc1", "fc2", "fc31", "fc32", "fc33", "fc41", "fc42", "fc43", "fc5", "fc6", "fc7", "fc8"}
    )
    assert layers == expected


# ------------------------------------------------------------------- evaluation


def test_eval_model_error_format(vae, loaders):
    (tr_mu, te_mu), (tr_sd, te_sd), (tr_r2, te_r2) = eval_model_error(loaders, vae)
    assert np.isfinite([tr_mu, te_mu, tr_sd, te_sd]).all()
    assert len(tr_r2) == 14 and len(te_r2) == 5  # train set, and 'val'
    assert np.isclose(tr_r2.mean(), tr_mu) and (tr_r2 <= 1).all()


def test_eval_model_error_comparison_split(vae, loaders):
    *_, (_, val_r2) = eval_model_error(loaders, vae, comparison="val")
    *_, (_, test_r2) = eval_model_error(loaders, vae, comparison="test")
    assert len(val_r2) == len(test_r2) == 5
    assert not np.allclose(val_r2, test_r2)


def test_eval_model_error_is_one_for_a_perfect_model(loaders):
    class Perfect(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        @property
        def device(self):
            return self.p.device

        def reconstruct(self, x):
            return x

    (tr, te), *_ = eval_model_error(loaders, Perfect())
    assert tr == pytest.approx(1.0) and te == pytest.approx(1.0)


def test_eval_model_error_ignores_constant_spectrograms(vae):
    from torch.utils.data import DataLoader

    specs = np.random.rand(6, 128, 128).astype(np.float32)
    specs[0] = 0  # R^2 is undefined for a blank spectrogram
    dl = DataLoader(SpectrogramDataset(specs), batch_size=3)
    (_, mu), _, (_, r2) = eval_model_error({"train": dl, "val": dl}, vae)
    assert np.isnan(r2[0]) and np.isfinite(mu)


# ---------------------------------------------------------------------- latents


def test_latents_from_loader(vae, loaders):
    z = get_latents_from_loader(vae, loaders["val"])
    assert z.shape == (5, 8)
    x = loaders["val"].dataset.specs
    vae.eval()
    with torch.no_grad():
        torch.testing.assert_close(torch.from_numpy(z), vae.get_latents(x), rtol=1e-4, atol=1e-5)


def test_latents_from_audio_use_the_models_spectrogram_parameters(tones):
    audio, fs = tones
    m = VAE(z_dim=8, device="cpu", spec_params=P)
    z = get_latents_from_audio(m, audio[:5], fs)
    assert z.shape == (5, 8)
    specs = torch.from_numpy(np.stack([make_spectrogram(a, fs, P) for a in audio[:5]]))
    m.eval()
    with torch.no_grad():
        torch.testing.assert_close(torch.from_numpy(z), m.get_latents(specs), rtol=1e-4, atol=1e-5)
    # a different intensity range gives different latents
    m2 = VAE(z_dim=8, device="cpu", spec_params={**P, "spec_max_val": 4.0})
    m2.load_state_dict(m.state_dict())
    assert not np.allclose(get_latents_from_audio(m2, audio[:5], fs), z)


# -------------------------------------------------------------- cross-validation


def test_cv_over_precisions(loaders, tmp_path):
    model = model_cv_precisions(
        loaders,
        n_epochs=2,
        z_dim=4,
        precisions=(1.0, 100.0),
        model_path=str(tmp_path),
        save_freq=0,  # checkpoints are ~200 MB; the final one is always saved
        device="cpu",
        spec_params=P,
    )
    assert isinstance(model, VAE) and not model.training
    assert model.spec_params["spec_min_val"] == P["spec_min_val"]
    df = __import__("pandas").read_csv(tmp_path / "cv_errs.csv")
    assert list(df["precision"]) == [1.0, 100.0]
    best = df.loc[df["val R2"].idxmax(), "precision"]
    assert model.model_precision == best
    assert os.path.isdir(tmp_path / "vae_precision_1.0")


def test_cv_resumes_without_retraining(loaders, tmp_path, monkeypatch):
    kw = dict(
        n_epochs=1, z_dim=4, precisions=(10.0,), model_path=str(tmp_path), device="cpu"
    )
    model_cv_precisions(loaders, **kw)

    def fail(*a, **k):
        raise AssertionError("trained again")

    monkeypatch.setattr(cv, "train", fail)
    assert isinstance(model_cv_precisions(loaders, **kw), VAE)


# ------------------------------------------------------------------- end to end


def test_train_model_end_to_end(wav_dataset, tmp_path):
    model_dir = str(tmp_path / "vae")
    model = train_model(
        [wav_dataset["audio_dir"]],
        [wav_dataset["seg_dir"]],
        model_dir,
        n_epochs=2,
        batch_size=4,
        max_jobs=1,
        save_freq=0,
        device="cpu",
        z_dim=4,
    )
    assert isinstance(model, VAE) and model.z_dim == 4 and model.device.type == "cpu"
    # one spectrogram per vocalization, which are 0.1 s long
    assert model.spec_params["max_dur"] == pytest.approx(0.1, abs=1e-3)
    # the intensity range was tuned to the data, rather than left at AVA's defaults
    assert (model.spec_params["spec_min_val"], model.spec_params["spec_max_val"]) != (2.0, 4.0)
    assert glob.glob(os.path.join(model_dir, "vae_precision_10.0", "checkpoint_*.tar"))
    assert os.path.isfile(os.path.join(model_dir, "cv_errs.csv"))


def test_train_model_on_fixed_length_chunks(wav_dataset, tmp_path):
    model = train_model(
        [wav_dataset["audio_dir"]],
        [wav_dataset["seg_dir"]],
        str(tmp_path),
        context_len=0.05,  # the 0.1 s vocalizations are cut into two chunks each
        n_epochs=1,
        batch_size=8,
        max_jobs=1,
        device="cpu",
        z_dim=4,
    )
    assert model.spec_params["max_dur"] == pytest.approx(0.05, abs=1e-3)


def test_train_model_can_keep_the_default_spectrogram_range(wav_dataset, tmp_path):
    model = train_model(
        [wav_dataset["audio_dir"]],
        [wav_dataset["seg_dir"]],
        str(tmp_path),
        n_epochs=1,
        batch_size=4,
        max_jobs=1,
        device="cpu",
        z_dim=4,
        tune_spec_range=False,
        spec_params={"spec_min_val": 1.0, "spec_max_val": 5.0},
    )
    assert (model.spec_params["spec_min_val"], model.spec_params["spec_max_val"]) == (1.0, 5.0)


def test_train_model_input_checks(wav_dataset, tmp_path, monkeypatch):
    with pytest.raises(AssertionError):
        train_model([wav_dataset["audio_dir"]], [], str(tmp_path), device="cpu")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no vocalizations"):
        train_model([str(empty)], [str(empty)], str(tmp_path / "m"), device="cpu")
