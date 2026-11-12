import warnings

import numpy as np
import pytest

from comparisons.acoustic_features import (
    BIOSOUND_FEATURES,
    SAT_FEATURES,
    SUMMARIES,
    extract_features,
    feature_names,
    features_from_audio,
    get_acoustic_features,
    make_feature_pipeline,
)

FS = 20000


def tone(freq=3000.0, amp=0.3, dur=0.1, fs=FS):
    t = np.arange(int(dur * fs)) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def noise(amp=0.1, dur=0.1, fs=FS, seed=0):
    return amp * np.random.default_rng(seed).standard_normal(int(dur * fs))


def feat(audio, names, **kw):
    values, got = extract_features(audio, FS, **kw)
    assert got == names
    return dict(zip(names, values))


# ----------------------------------------------------------------------- names


def test_default_names():
    names = feature_names()
    assert len(names) == len(set(names)) == len(SAT_FEATURES) * 2 + 1
    assert names[0] == "amplitude_mean" and names[1] == "amplitude_std"
    assert names[-1] == "duration"


def test_names_follow_the_options():
    assert feature_names("sat", ("median",), include_duration=False) == [
        f"{f}_median" for f in SAT_FEATURES
    ]
    assert feature_names("biosound", include_duration=False) == list(BIOSOUND_FEATURES)
    assert len(BIOSOUND_FEATURES) == 21


def test_invalid_options_are_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        feature_names("mfcc")
    with pytest.raises(ValueError, match="unknown summaries"):
        feature_names("sat", ("mean", "mode"))
    with pytest.raises(ValueError, match="at least one"):
        feature_names("sat", ())
    with pytest.raises(ValueError):
        extract_features(tone(), FS, method="mfcc")


# -------------------------------------------------------------------- extraction


def test_extract_features_shape_and_names():
    values, names = extract_features(tone(), FS)
    assert values.shape == (len(names),) == (13,)
    assert values.dtype == np.float64 and np.isfinite(values).all()
    assert len(set(names)) == len(names)


def test_pitch_of_a_tone():
    f = feat(tone(3000.0), feature_names())
    assert f["pitch_mean"] == pytest.approx(3000.0, rel=0.02)
    assert f["pitch_std"] < 50  # steady


def test_pitch_follows_frequency():
    names = feature_names()
    low, high = feat(tone(2000.0), names), feat(tone(4000.0), names)
    assert low["pitch_mean"] == pytest.approx(2000.0, rel=0.03)
    assert high["pitch_mean"] == pytest.approx(4000.0, rel=0.03)


def test_amplitude_increases_with_loudness():
    names = feature_names()
    assert feat(tone(amp=0.6), names)["amplitude_mean"] > feat(tone(amp=0.1), names)["amplitude_mean"]


def test_wiener_entropy_separates_tones_from_noise():
    names = feature_names()
    # entropy is far lower (more negative) for a pure tone than for broadband noise
    assert feat(tone(), names)["entropy_mean"] < feat(noise(), names)["entropy_mean"] - 2


def test_duration_feature():
    assert feat(tone(dur=0.1), feature_names())["duration"] == pytest.approx(0.1)
    assert feat(tone(dur=0.25), feature_names())["duration"] == pytest.approx(0.25)
    _, names = extract_features(tone(), FS, include_duration=False)
    assert "duration" not in names


def test_input_shape_dtype_and_scale_conventions():
    a = tone()
    ref, _ = extract_features(a, FS)
    for other in (a[None, :, None], a[:, None], a.astype(np.float32)):
        got, _ = extract_features(other, FS)
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)


def test_summaries_are_computed_over_time():
    names = feature_names("sat", ("mean", "std", "min", "max", "median"))
    f = feat(tone() * np.hanning(2000), names, summaries=("mean", "std", "min", "max", "median"))
    for feature in SAT_FEATURES:
        assert f[f"{feature}_min"] <= f[f"{feature}_median"] <= f[f"{feature}_max"]
        assert f[f"{feature}_min"] <= f[f"{feature}_mean"] <= f[f"{feature}_max"]
        assert f[f"{feature}_std"] >= 0
    assert f["amplitude_std"] > 1  # a windowed vocalization has a changing amplitude (dB)
    assert f["amplitude_max"] - f["amplitude_min"] > f["amplitude_std"]


def test_all_listed_summaries_exist():
    assert set(SUMMARIES) == {"mean", "std", "min", "max", "median"}


def test_deterministic():
    a, _ = extract_features(noise(), FS)
    b, _ = extract_features(noise(), FS)
    np.testing.assert_array_equal(a, b)


def test_kwargs_reach_vocalpy():
    default, _ = extract_features(tone(), FS)
    narrow, _ = extract_features(tone(), FS, n_fft=200, hop_length=20)
    assert not np.allclose(default, narrow)


def test_very_short_audio_still_gives_features():
    values, names = extract_features(np.random.randn(50) * 0.1, FS)
    assert values.shape == (len(names),)


def test_non_finite_audio_is_rejected():
    a = tone()
    a[10] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        extract_features(a, FS)


# ---------------------------------------------------------------- batches of audio


def test_features_from_audio():
    audio = [tone(2000.0), tone(3000.0), noise()]
    X, names = features_from_audio(audio, FS)
    assert X.shape == (3, len(names))
    np.testing.assert_allclose(X[1], extract_features(audio[1], FS)[0])
    assert X[0, names.index("pitch_mean")] < X[1, names.index("pitch_mean")]


def test_features_from_empty_list():
    X, names = features_from_audio([], FS)
    assert X.shape == (0, len(names))


def test_a_bad_vocalization_gives_nan_row_and_warning():
    bad = tone()
    bad[5] = np.inf
    with pytest.warns(UserWarning, match="could not extract features"):
        X, _ = features_from_audio([tone(), bad, tone()], FS)
    assert np.isnan(X[1]).all() and np.isfinite(X[[0, 2]]).all()


def test_batch_options_match_single_calls():
    audio = [tone(2500.0), noise()]
    X, names = features_from_audio(audio, FS, summaries=("median",), include_duration=False)
    assert names == feature_names("sat", ("median",), False)
    np.testing.assert_allclose(
        X[0], extract_features(audio[0], FS, summaries=("median",), include_duration=False)[0]
    )


@pytest.mark.slow
def test_parallel_jobs_give_the_same_features():
    audio = [tone(2000.0 + 300 * i) for i in range(4)]
    serial, _ = features_from_audio(audio, FS, n_jobs=1)
    parallel, _ = features_from_audio(audio, FS, n_jobs=2)
    np.testing.assert_allclose(serial, parallel)


def test_biosound_needs_a_high_enough_sampling_rate():
    with pytest.raises(ValueError, match="sampling rate above"):
        extract_features(tone(), FS, method="biosound")  # 20 kHz
    # the batch function turns this into NaN rows and warnings, so check the single call


@pytest.mark.slow
def test_biosound_features():
    fs = 44100
    values, names = extract_features(tone(fs=fs, dur=0.15), fs, method="biosound")
    assert names == list(BIOSOUND_FEATURES) + ["duration"]
    assert values.shape == (22,)
    f = dict(zip(names, values))
    assert f["duration"] == pytest.approx(0.15)
    finite = np.isfinite(values).sum()
    assert finite >= 19  # a few (e.g. pk2) can legitimately be undefined


# ------------------------------------------------------------- end to end from disk


def test_get_acoustic_features_from_files(wav_dataset):
    X, names, sr = get_acoustic_features(
        [wav_dataset["audio_dir"]], [wav_dataset["seg_dir"]], max_jobs=1
    )
    assert sr == wav_dataset["fs"]
    assert X.shape == (wav_dataset["n_vocs"], len(names))
    assert np.isfinite(X).all()
    assert np.allclose(X[:, names.index("duration")], 0.1)
    # vocalizations were made at 2000-3700 Hz
    pitch = X[:, names.index("pitch_mean")]
    assert pitch.min() > 1800 and pitch.max() < 4000


def test_get_acoustic_features_respects_max_vocs(wav_dataset):
    X, _, _ = get_acoustic_features(
        [wav_dataset["audio_dir"]], [wav_dataset["seg_dir"]], max_vocs=10, max_jobs=1
    )
    assert len(X) == 10


def test_get_acoustic_features_on_fixed_length_chunks(wav_dataset):
    # the same chunking that Ouroboros is trained with: two 0.05 s chunks per 0.1 s vocalization
    X, names, _ = get_acoustic_features(
        [wav_dataset["audio_dir"]], [wav_dataset["seg_dir"]], context_len=0.05, max_jobs=1
    )
    assert X.shape == (2 * wav_dataset["n_vocs"], len(names))
    assert np.allclose(X[:, names.index("duration")], 0.05)


def test_get_acoustic_features_input_checks(wav_dataset, tmp_path):
    with pytest.raises(AssertionError):
        get_acoustic_features([wav_dataset["audio_dir"]], [], max_jobs=1)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no vocalizations"):
        get_acoustic_features([str(empty)], [str(empty)], max_jobs=1)


# ---------------------------------------------------------------------- pipeline


def test_pipeline_imputes_and_standardizes():
    rng = np.random.default_rng(0)
    X = rng.normal(5, 3, size=(100, 4))
    X[3, 1] = np.nan
    X[:, 2] = np.nan  # a feature that is never defined
    out = make_feature_pipeline().fit_transform(X)
    assert out.shape == X.shape and np.isfinite(out).all()
    np.testing.assert_allclose(out[:, [0, 1, 3]].mean(axis=0), 0, atol=1e-8)
    np.testing.assert_allclose(out[:, [0, 1, 3]].std(axis=0), 1, atol=1e-2)


def test_pipeline_reduces_dimensionality():
    X = np.random.default_rng(1).normal(size=(50, 6))
    out = make_feature_pipeline(n_components=2).fit_transform(X)
    assert out.shape == (50, 2)
    assert abs(np.corrcoef(out.T)[0, 1]) < 1e-8  # principal components are uncorrelated


def test_pipeline_uses_training_statistics_for_new_data():
    rng = np.random.default_rng(2)
    train, test = rng.normal(0, 1, (80, 3)), rng.normal(10, 1, (20, 3))
    pipe = make_feature_pipeline().fit(train)
    # held-out data are not re-centered
    assert pipe.transform(test).mean() > 5
    test[0, 0] = np.nan
    assert np.isfinite(pipe.transform(test)).all()  # filled with the *training* median


def test_features_feed_the_pipeline(wav_dataset):
    X, names, _ = get_acoustic_features(
        [wav_dataset["audio_dir"]], [wav_dataset["seg_dir"]], max_jobs=1
    )
    Z = make_feature_pipeline(n_components=4).fit_transform(X)
    assert Z.shape == (len(X), 4) and np.isfinite(Z).all()
