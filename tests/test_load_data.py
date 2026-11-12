"""
the loaders memory-map recordings and read only the segments they need. These tests check that
this gives exactly what the original read-everything loaders gave, and that it uses much less memory.
"""

import glob
import os
import tracemalloc
import warnings

import numpy as np
import pytest
from scipy.io import wavfile

import data.load_data as ld
from data.load_data import (
    get_audio_analysis,
    get_audio_training,
    get_segmented_audio,
    iter_segmented_audio,
    list_audio_files,
    load_file_segments,
)

# ------------------------------------------------- the original (read-everything) loaders


def reference_read(path):
    sr, aud = wavfile.read(path)
    if aud.dtype == np.int16:
        aud = aud / -np.iinfo(aud.dtype).min
    return sr, aud


def reference_training(audio_files, seg_files, max_vocs=5000, context_len=0.3, extend=True):
    audio, sr = [], None
    for a, s in zip(audio_files, seg_files):
        if os.path.isfile(a) and os.path.isfile(s):
            sr, aud = reference_read(a)
            chunk_len = int(round(context_len * sr))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                seg_onoffs = np.loadtxt(s, usecols=(0, 1))
            if len(seg_onoffs) == 0:
                continue
            if len(seg_onoffs.shape) == 1:
                seg_onoffs = seg_onoffs[None, :]
            for on, off in seg_onoffs:
                difference = (off - on) - context_len
                if (difference <= 0) and extend:
                    on += difference
                on_ind, off_ind = int(round(on * sr)), int(round(off * sr))
                aud_sample = aud[on_ind:off_ind]
                cut_len = np.mod(len(aud_sample), chunk_len)
                if cut_len > 0:
                    aud_sample = aud_sample[:-cut_len]
                aud_sample = aud_sample.reshape(-1, chunk_len, 1)
                audio += list(aud_sample)
                if len(audio) >= max_vocs:
                    return audio[:max_vocs], sr
    return audio, sr


def reference_analysis(audio_files, seg_files, max_vocs=5000, padding=0.1):
    audio, sr = [], None
    for a, s in zip(audio_files, seg_files):
        if os.path.isfile(a) and os.path.isfile(s):
            sr, aud = reference_read(a)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                seg_onoffs = np.loadtxt(s, usecols=(0, 1))
            if len(seg_onoffs) == 0:
                continue
            if len(seg_onoffs.shape) == 1:
                seg_onoffs = seg_onoffs[None, :]
            for on, off in seg_onoffs:
                on = max(0.0, on - padding)
                on_ind, off_ind = int(round(on * sr)), int(round(off * sr))
                audio.append(aud[on_ind:off_ind][None, :, None])
                if len(audio) >= max_vocs:
                    return audio[:max_vocs], sr
    return audio, sr


def assert_same(got, want):
    (got_audio, got_sr), (want_audio, want_sr) = got, want
    assert got_sr == want_sr
    assert len(got_audio) == len(want_audio)
    for g, w in zip(got_audio, want_audio):
        assert g.shape == w.shape and g.dtype == w.dtype
        np.testing.assert_array_equal(g, w)


def files_of(d):
    audio_files = sorted(glob.glob(os.path.join(d["audio_dir"], "*.wav")))
    seg_files = [os.path.join(d["seg_dir"], os.path.basename(a)[:-4] + ".txt") for a in audio_files]
    return audio_files, seg_files


@pytest.fixture
def odd_dataset(tmp_path):
    """wavs of several dtypes, with segments that start near 0, run past the end, or are tiny"""
    fs = 20000
    rng = np.random.default_rng(1)
    (tmp_path / "a").mkdir()
    (tmp_path / "s").mkdir()
    segs = np.array([[0.0, 0.2], [0.02, 0.07], [0.3, 0.55], [0.9, 1.3], [0.5, 0.5005]])
    for name, make in {
        "i16": lambda x: (x * 20000).astype(np.int16),
        "f32": lambda x: x.astype(np.float32),
        "f64": lambda x: x.astype(np.float64),
        "i32": lambda x: (x * 1e8).astype(np.int32),
    }.items():
        wavfile.write(str(tmp_path / "a" / f"{name}.wav"), fs, make(rng.standard_normal(fs)))
        np.savetxt(str(tmp_path / "s" / f"{name}.txt"), segs, fmt="%.5f")
    return {"audio_dir": str(tmp_path / "a"), "seg_dir": str(tmp_path / "s")}


# --------------------------------------------------------------------- same results


@pytest.mark.parametrize("extend", [True, False])
@pytest.mark.parametrize("context_len", [0.05, 0.1])
def test_training_audio_is_unchanged(wav_dataset, extend, context_len):
    a, s = files_of(wav_dataset)
    assert_same(
        get_audio_training(a, s, context_len=context_len, extend=extend),
        reference_training(a, s, context_len=context_len, extend=extend),
    )


@pytest.mark.parametrize("padding", [0.0, 0.05])
def test_analysis_audio_is_unchanged(wav_dataset, padding):
    a, s = files_of(wav_dataset)
    assert_same(
        get_audio_analysis(a, s, padding=padding), reference_analysis(a, s, padding=padding)
    )


@pytest.mark.parametrize("max_vocs", [1, 7, 24, 1000])
def test_max_vocs_is_respected_as_before(wav_dataset, max_vocs):
    a, s = files_of(wav_dataset)
    assert_same(
        get_audio_analysis(a, s, max_vocs=max_vocs), reference_analysis(a, s, max_vocs=max_vocs)
    )
    assert_same(
        get_audio_training(a, s, context_len=0.05, max_vocs=max_vocs),
        reference_training(a, s, context_len=0.05, max_vocs=max_vocs),
    )


def test_other_dtypes_and_awkward_segments_are_unchanged(odd_dataset):
    # int16 is scaled, other dtypes are not; segments at the start, past the end of the file, and
    # shorter than a chunk behave as they always did
    a, s = files_of(odd_dataset)
    assert_same(get_audio_analysis(a, s), reference_analysis(a, s))
    assert_same(
        get_audio_training(a, s, context_len=0.05), reference_training(a, s, context_len=0.05)
    )
    got, _ = get_audio_analysis(a, s)
    assert {x.dtype for x in got} == {np.dtype("float64"), np.dtype("float32"), np.dtype("int32")}


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("shuffle", [True, False])
def test_get_segmented_audio_is_unchanged(wav_dataset, training, shuffle):
    got = get_segmented_audio(
        wav_dataset["audio_dir"], wav_dataset["seg_dir"], context_len=0.05, seed=3,
        training=training, shuffle_order=shuffle,
    )  # fmt: skip
    # same file order as before (listing, then the seeded shuffle), same audio as the original loaders
    audio_files, seg_files = list_audio_files(
        wav_dataset["audio_dir"], wav_dataset["seg_dir"], ".wav", 3, shuffle
    )
    want = (
        reference_training(audio_files, seg_files, context_len=0.05)
        if training
        else reference_analysis(audio_files, seg_files, padding=0.0)
    )
    assert_same(got, want)


def test_no_audio_gives_empty_list_and_no_sampling_rate(tmp_path):
    assert get_segmented_audio(str(tmp_path), str(tmp_path)) == ([], None)


# --------------------------------------------------------------------- memory mapping


def test_recordings_are_memory_mapped(wav_dataset, monkeypatch):
    calls = []
    real = wavfile.read

    def spy(path, *args, **kwargs):
        calls.append(kwargs.get("mmap", False))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(ld.wavfile, "read", spy)
    a, s = files_of(wav_dataset)
    get_audio_analysis(a, s)
    get_audio_training(a, s, context_len=0.05)
    assert calls and all(calls)


def test_falls_back_to_a_normal_read_when_mapping_is_impossible(wav_dataset, monkeypatch):
    real = wavfile.read

    def no_mmap(path, *args, **kwargs):
        if kwargs.get("mmap"):
            raise ValueError("mmap is not supported for this file")
        return real(path, *args, **kwargs)

    a, s = files_of(wav_dataset)
    want = reference_analysis(a, s, padding=0.0)
    monkeypatch.setattr(ld.wavfile, "read", no_mmap)
    assert_same(get_audio_analysis(a, s, padding=0.0), want)
    assert_same(
        get_audio_training(a, s, context_len=0.05), reference_training(a, s, context_len=0.05)
    )


def test_returned_audio_does_not_depend_on_the_file(wav_dataset):
    a, s = files_of(wav_dataset)
    audio, _ = get_audio_analysis(a, s)
    want = [x.copy() for x in audio]
    assert not any(isinstance(x, np.memmap) or isinstance(x.base, np.memmap) for x in audio)
    for f in a:  # overwrite every recording on disk
        sr, x = wavfile.read(f)
        wavfile.write(f, sr, np.zeros_like(x))
    for g, w in zip(audio, want):
        np.testing.assert_array_equal(g, w)


def test_reading_a_few_segments_does_not_load_the_whole_recording(tmp_path):
    # a 60 s recording is 21 MB as float64. Reading one short segment of it should allocate almost nothing
    fs = 44100
    path = str(tmp_path / "long.wav")
    wavfile.write(path, fs, (np.random.default_rng(0).standard_normal(60 * fs) * 3000).astype(np.int16))
    np.savetxt(str(tmp_path / "long.txt"), np.array([[30.0, 30.1]]))
    whole_file_float64 = 60 * fs * 8

    tracemalloc.start()
    ld.get_audio_analysis([path], [str(tmp_path / "long.txt")])
    _, peak_new = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    reference_analysis([path], [str(tmp_path / "long.txt")], padding=0.0)
    _, peak_old = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak_old > 0.9 * whole_file_float64  # the test would notice the old behaviour
    assert peak_new < 0.05 * whole_file_float64


# ------------------------------------------------------------- streaming, file by file


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("shuffle", [True, False])
def test_streaming_gives_the_same_audio_in_the_same_order(wav_dataset, training, shuffle):
    kw = dict(context_len=0.05, seed=5, training=training, shuffle_order=shuffle)
    whole = get_segmented_audio(
        wav_dataset["audio_dir"], wav_dataset["seg_dir"], max_vocs=10**6, **kw
    )
    streamed, files = [], []
    for f, segments, sr in iter_segmented_audio(
        wav_dataset["audio_dir"], wav_dataset["seg_dir"], **kw
    ):
        streamed += segments
        files.append(f)
        assert sr == wav_dataset["fs"]
    assert_same((streamed, wav_dataset["fs"]), whole)
    assert len(files) == 8  # one item per file


def test_streaming_reads_one_file_at_a_time(wav_dataset, monkeypatch):
    reads = []
    real = wavfile.read

    def spy(path, *args, **kwargs):
        reads.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(ld.wavfile, "read", spy)
    gen = iter_segmented_audio(wav_dataset["audio_dir"], wav_dataset["seg_dir"])
    assert reads == []  # nothing happens until you ask
    next(gen)
    assert len(reads) == 1
    next(gen)
    assert len(reads) == 2


def test_streaming_skips_files_without_segments(wav_dataset):
    os.remove(os.path.join(wav_dataset["seg_dir"], "rec3.txt"))
    open(os.path.join(wav_dataset["seg_dir"], "rec4.txt"), "w").close()  # no segments
    files = [f for f, _, _ in iter_segmented_audio(wav_dataset["audio_dir"], wav_dataset["seg_dir"])]
    assert len(files) == 6
    assert not any(f.endswith(("rec3.wav", "rec4.wav")) for f in files)


def test_load_file_segments(wav_dataset):
    a, s = files_of(wav_dataset)
    segs, sr = load_file_segments(a[0], s[0])
    assert len(segs) == 3 and sr == wav_dataset["fs"] and segs[0].shape == (1, 2000, 1)
    chunks, _ = load_file_segments(a[0], s[0], training=True, context_len=0.05)
    assert len(chunks) == 6 and chunks[0].shape == (1000, 1)
    assert load_file_segments(a[0], "missing.txt") == ([], None)
