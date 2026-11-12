"""acoustic features are computed one audio file at a time, optionally with a resumable on-disk cache"""

import glob
import os
import shutil

import numpy as np
import pytest
from scipy.io import wavfile

import comparisons.acoustic_features as af
import data.load_data as ld
from comparisons.acoustic_features import (
    features_from_audio,
    get_acoustic_features,
    iter_acoustic_features,
)
from data.load_data import get_segmented_audio


def reference(dirs_a, dirs_s, max_vocs=5000, context_len=None, **kw):
    """what get_acoustic_features did before: gather all the audio, then extract features"""
    per_dir = max_vocs // len(dirs_a)
    audio, sr = [], None
    for a, s in zip(dirs_a, dirs_s):
        chunks, sr = get_segmented_audio(
            a, s, max_vocs=per_dir, context_len=context_len if context_len is not None else 0.3,
            seed=1234, training=context_len is not None, extend=True, shuffle_order=False,
        )  # fmt: skip
        audio += chunks
    X, names = features_from_audio(audio, sr, **kw)
    return X, names, sr


@pytest.fixture
def second_dataset(tmp_path):
    """a second directory of recordings, with different vocalizations"""
    fs = 20000
    a, s = tmp_path / "audio2", tmp_path / "segs2"
    a.mkdir()
    s.mkdir()
    rng = np.random.default_rng(7)
    for i in range(3):
        y = 0.002 * rng.standard_normal(fs)
        t = np.arange(int(0.1 * fs)) / fs
        y[4000 : 4000 + len(t)] += 0.3 * np.sin(2 * np.pi * (5000 + 300 * i) * t) * np.hanning(len(t))
        wavfile.write(str(a / f"b{i}.wav"), fs, (y * 32767).astype(np.int16))
        np.savetxt(str(s / f"b{i}.txt"), np.array([[0.2, 0.3]]), fmt="%.4f")
    return {"audio_dir": str(a), "seg_dir": str(s)}


# ----------------------------------------------------------- same features as before


def test_get_acoustic_features_is_unchanged(wav_dataset):
    want, names, sr = reference([wav_dataset["audio_dir"]], [wav_dataset["seg_dir"]])
    got, got_names, got_sr = get_acoustic_features(
        [wav_dataset["audio_dir"]], [wav_dataset["seg_dir"]], max_jobs=1
    )
    assert got_names == names and got_sr == sr
    np.testing.assert_array_equal(got, want)


def test_unchanged_on_chunks_and_with_options(wav_dataset):
    kw = dict(context_len=0.05, method="sat", summaries=("mean", "max"), include_duration=False)
    a, s = [wav_dataset["audio_dir"]], [wav_dataset["seg_dir"]]
    want, *_ = reference(a, s, **kw)
    got, names, _ = get_acoustic_features(a, s, max_jobs=1, **kw)
    assert len(names) == 12
    np.testing.assert_array_equal(got, want)


def test_unchanged_across_directories_and_budgets(wav_dataset, second_dataset):
    a = [wav_dataset["audio_dir"], second_dataset["audio_dir"]]
    s = [wav_dataset["seg_dir"], second_dataset["seg_dir"]]
    for max_vocs in (5000, 10, 7):  # the budget is split evenly between directories
        want, *_ = reference(a, s, max_vocs=max_vocs)
        got, _, _ = get_acoustic_features(a, s, max_vocs=max_vocs, max_jobs=1)
        np.testing.assert_array_equal(got, want)
        assert len(got) == min(24, max_vocs // 2) + min(3, max_vocs // 2)


# -------------------------------------------------------------------- streaming


def test_iteration_is_per_file(wav_dataset):
    out = list(iter_acoustic_features(wav_dataset["audio_dir"], wav_dataset["seg_dir"]))
    assert len(out) == 8
    files, feats, srs = zip(*out)
    assert all(f.endswith(".wav") for f in files) and set(srs) == {wav_dataset["fs"]}
    assert all(x.shape == (3, 13) for x in feats)


def test_iteration_reads_one_file_at_a_time(wav_dataset, monkeypatch):
    reads = []
    real = wavfile.read

    def spy(path, *a, **k):
        reads.append(path)
        return real(path, *a, **k)

    monkeypatch.setattr(ld.wavfile, "read", spy)
    gen = iter_acoustic_features(wav_dataset["audio_dir"], wav_dataset["seg_dir"])
    assert reads == []
    next(gen)
    assert len(reads) == 1
    next(gen)
    assert len(reads) == 2


def test_options_are_checked_immediately(wav_dataset):
    with pytest.raises(ValueError, match="unknown method"):
        iter_acoustic_features(wav_dataset["audio_dir"], wav_dataset["seg_dir"], method="mfcc")
    with pytest.raises(ValueError, match="unknown summaries"):
        iter_acoustic_features(
            wav_dataset["audio_dir"], wav_dataset["seg_dir"], summaries=("mode",)
        )


def test_max_vocs_cuts_the_last_file(wav_dataset):
    full = np.concatenate(
        [x for _, x, _ in iter_acoustic_features(wav_dataset["audio_dir"], wav_dataset["seg_dir"])]
    )
    cut = list(iter_acoustic_features(wav_dataset["audio_dir"], wav_dataset["seg_dir"], max_vocs=7))
    assert [len(x) for _, x, _ in cut] == [3, 3, 1]  # stops reading files once it has enough
    np.testing.assert_array_equal(np.concatenate([x for _, x, _ in cut]), full[:7])


def test_max_vocs_zero_reads_nothing(wav_dataset, monkeypatch):
    monkeypatch.setattr(ld.wavfile, "read", lambda *a, **k: pytest.fail("read a file"))
    assert list(iter_acoustic_features(wav_dataset["audio_dir"], wav_dataset["seg_dir"], max_vocs=0)) == []


def test_files_without_segments_are_skipped(wav_dataset):
    os.remove(os.path.join(wav_dataset["seg_dir"], "rec3.txt"))
    open(os.path.join(wav_dataset["seg_dir"], "rec4.txt"), "w").close()
    out = list(iter_acoustic_features(wav_dataset["audio_dir"], wav_dataset["seg_dir"]))
    assert len(out) == 6


def test_no_data_raises(tmp_path):
    with pytest.raises(ValueError, match="no vocalizations"):
        get_acoustic_features([str(tmp_path)], [str(tmp_path)], max_jobs=1)


# ------------------------------------------------------------------------ cache


def count_extractions(monkeypatch):
    calls = []
    real = af.features_from_audio

    def counting(audio, *a, **k):
        calls.append(len(audio))
        return real(audio, *a, **k)

    monkeypatch.setattr(af, "features_from_audio", counting)
    return calls


def run(d, cache, **kw):
    return list(
        iter_acoustic_features(d["audio_dir"], d["seg_dir"], cache_dir=str(cache), **kw)
    )


def test_cache_is_written_and_reused(wav_dataset, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    calls = count_extractions(monkeypatch)
    first = run(wav_dataset, cache)
    assert len(calls) == 8 and len(glob.glob(str(cache / "*.npz"))) == 8
    assert not glob.glob(str(cache / "*.tmp*"))  # no half-written files left behind

    # a finished cache is used without reading any audio, or extracting anything
    monkeypatch.setattr(ld.wavfile, "read", lambda *a, **k: pytest.fail("read audio"))
    second = run(wav_dataset, cache)
    assert len(calls) == 8
    for (f1, x1, s1), (f2, x2, s2) in zip(first, second):
        assert f1 == f2 and s1 == s2
        np.testing.assert_array_equal(x1, x2)


def test_cached_features_equal_uncached(wav_dataset, tmp_path):
    want, names, sr = get_acoustic_features(
        [wav_dataset["audio_dir"]], [wav_dataset["seg_dir"]], max_jobs=1
    )
    kw = dict(max_jobs=1, cache_dir=str(tmp_path / "c"))
    for _ in range(2):  # fresh, then from the cache
        got, got_names, got_sr = get_acoustic_features(
            [wav_dataset["audio_dir"]], [wav_dataset["seg_dir"]], **kw
        )
        assert got_names == names and got_sr == sr
        np.testing.assert_array_equal(got, want)


def test_interrupted_run_resumes(wav_dataset, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    run(wav_dataset, cache)
    for f in sorted(glob.glob(str(cache / "*.npz")))[:2]:
        os.remove(f)  # as if the run had stopped part way
    calls = count_extractions(monkeypatch)
    run(wav_dataset, cache)
    assert calls == [3, 3]  # only the two missing files are redone


def test_cache_is_invalidated_by_changed_settings(wav_dataset, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    run(wav_dataset, cache)
    calls = count_extractions(monkeypatch)
    other = run(wav_dataset, cache, summaries=("median",))
    assert len(calls) == 8 and other[0][1].shape == (3, 7)
    run(wav_dataset, cache, summaries=("median",))
    assert len(calls) == 8  # and the new settings are cached in turn
    run(wav_dataset, cache, context_len=0.05)
    assert len(calls) == 16
    run(wav_dataset, cache, n_fft=200)  # options for VocalPy count too
    assert len(calls) == 24


def test_cache_is_invalidated_by_changed_files(wav_dataset, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    first = run(wav_dataset, cache)
    seg = os.path.join(wav_dataset["seg_dir"], os.path.basename(first[0][0])[:-4] + ".txt")
    np.savetxt(seg, np.array([[0.1, 0.2]]), fmt="%.4f")  # now one vocalization, not three
    calls = count_extractions(monkeypatch)
    second = run(wav_dataset, cache)
    assert calls == [1]
    assert second[0][1].shape[0] == 1 and sum(len(x) for _, x, _ in second) == 22

    audio = first[1][0]
    sr, y = wavfile.read(audio)
    wavfile.write(audio, sr, (y // 2).astype(y.dtype))  # re-recorded
    calls.clear()
    run(wav_dataset, cache)
    assert calls == [3]


def test_truncated_files_are_not_cached_but_cached_files_can_be_cut(wav_dataset, tmp_path):
    cache = tmp_path / "cache"
    run(wav_dataset, cache, max_vocs=7)  # reads 3 files; the third is cut, so is not cached
    assert len(glob.glob(str(cache / "*.npz"))) == 2
    full = run(wav_dataset, cache)
    assert len(glob.glob(str(cache / "*.npz"))) == 8
    cut = run(wav_dataset, cache, max_vocs=5)  # now served from the cache, cut to fit
    np.testing.assert_array_equal(
        np.concatenate([x for _, x, _ in cut]), np.concatenate([x for _, x, _ in full])[:5]
    )


def test_corrupt_cache_files_are_recomputed(wav_dataset, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    first = run(wav_dataset, cache)
    bad = sorted(glob.glob(str(cache / "*.npz")))[0]
    with open(bad, "wb") as f:
        f.write(b"not an npz file")
    calls = count_extractions(monkeypatch)
    again = run(wav_dataset, cache)
    assert calls == [3]
    for (_, x1, _), (_, x2, _) in zip(first, again):
        np.testing.assert_array_equal(x1, x2)


def test_cache_dir_is_shared_between_directories(wav_dataset, second_dataset, tmp_path):
    kw = dict(max_jobs=1, cache_dir=str(tmp_path / "c"))
    a = [wav_dataset["audio_dir"], second_dataset["audio_dir"]]
    s = [wav_dataset["seg_dir"], second_dataset["seg_dir"]]
    X1, *_ = get_acoustic_features(a, s, **kw)
    assert len(glob.glob(str(tmp_path / "c" / "*.npz"))) == 11
    X2, *_ = get_acoustic_features(a, s, **kw)
    np.testing.assert_array_equal(X1, X2)
    assert len(X1) == 27
