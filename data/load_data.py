import numpy as np
import warnings

import os
import glob
from scipy.io import wavfile


from typing import Iterator, Tuple, Union


def _read_wav(path: str) -> Tuple[int, np.ndarray]:
    """
    read a wav file, memory-mapping the samples where possible, so that only the
    parts of a (possibly very long) recording that are actually used are read from disk. Files that cannot
    be memory-mapped (anything other than uncompressed PCM) are read normally.

    returns
    -----
        - the sampling rate
        - the samples: a read-only memory map, or an array. Pass slices to `_to_float`
            before keeping them
    """
    try:
        return wavfile.read(path, mmap=True)
    except ValueError:
        return wavfile.read(path)


def _to_float(samples: np.ndarray) -> np.ndarray:
    """
    copy a slice of a recording out of the memory map, as an in-memory array. int16 audio is
    converted to floats in [-1, 1); other dtypes are kept as they are. Converting slice by slice
    (rather than the whole recording first) gives the same values with a fraction of the memory.
    """
    samples = np.asarray(samples)
    if samples.dtype == np.int16:  # ints are TOO BIG!! turn into floats
        return samples / -np.iinfo(samples.dtype).min
    return np.array(samples)


def get_audio_training(
    audio_files: list[str],
    seg_files: list[str],
    max_vocs: int = 5000,
    context_len: float = 0.3,
    extend: bool = True,
):
    """
    Takes a list of wav files, containing audio, and a list
    of txt files, containing the onsets and offsets of vocalizations.
    Using these, splits the audio files into chunks of
    `context_len` long vocalizations, returning at most
    `max_vocs` vocalizations. Used for generating model train/test data

    Inputs
    -----
        - audio_files: a list of .wav files
        - seg_files: a list of .txt files, containing onsets and offsets of vocalizations
        - max_vocs: maximum number of vocalizations to grab from the audio, total
        - context_len: length of segmented audio in seconds, when full_vocs = False
        - extend: whether to extend the onset of vocalizations to avoid cutting off the
        end when chunking

    Returns
    -----
        - a list of collected audio
        - the sample rate of the audio
    """

    audio = []
    sr = None  # stays None if no file is found

    for a, s in zip(audio_files, seg_files):
        if os.path.isfile(a) and os.path.isfile(s):
            sr, aud = _read_wav(a)  # here, we assume
            # all loaded files have the same sample rate

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
                    # extend from the beginning, if shorter than context_len
                    on += difference
                on_ind, off_ind = int(round(on * sr)), int(round(off * sr))
                aud_sample = _to_float(aud[on_ind:off_ind])

                cut_len = np.mod(len(aud_sample), chunk_len)
                if cut_len > 0:
                    aud_sample = aud_sample[:-cut_len]

                aud_sample = aud_sample.reshape(-1, chunk_len, 1)

                audio += list(aud_sample)

                if len(audio) >= max_vocs:
                    return audio[:max_vocs], sr

    return audio, sr


def get_audio_analysis(
    audio_files: list[str],
    seg_files: list[str],
    max_vocs: int = 5000,
    padding: float = 0.1,
):
    """
    Takes a list of wav files, containing audio, and a list
    of txt files, containing the onsets and offsets of vocalizations.
    Using these, extracts vocalizations, returning at most
    `max_vocs` of them. Used for generating data for analysis

    Inputs
    -----
        - audio_files: a list of .wav files
        - seg_files: a list of .txt files, containing onsets and offsets of vocalizations
        - max_vocs: maximum number of vocalizations to grab from the audio, total
        - padding: amount of time (in seconds) to pad the onsets of vocalizations wtih

    Returns
    -----
        - a list of collected audio
        - the sample rate of the audio
    """

    audio = []
    sr = None  # stays None if no file is found

    for a, s in zip(audio_files, seg_files):
        if os.path.isfile(a) and os.path.isfile(s):
            sr, aud = _read_wav(a)  # here, we assume
            # all loaded files have the same sample rate

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
                audio.append(_to_float(aud[on_ind:off_ind])[None, :, None])

                if len(audio) >= max_vocs:
                    return audio[:max_vocs], sr

    return audio, sr


def list_audio_files(
    audio_path: str,
    seg_path: str,
    audio_id: str,
    seed: Union[None, int],
    shuffle_order: bool,
) -> Tuple[list, list]:
    """the audio files in `audio_path`, in the (optionally shuffled) order they are used, with their segment files"""
    gen = np.random.default_rng(seed=seed)

    audio_files = glob.glob(os.path.join(audio_path, "*" + audio_id))

    if shuffle_order:
        order = gen.choice(len(audio_files), len(audio_files), replace=False)
        audio_files = [audio_files[o] for o in order]

    audio_tags = [a.split("/")[-1].split(audio_id)[0] for a in audio_files]
    seg_files = [os.path.join(seg_path, a + ".txt") for a in audio_tags]
    return audio_files, seg_files


def get_segmented_audio(
    audio_path: str,
    seg_path: str,
    audio_id: str = ".wav",
    max_vocs: int = 5000,
    context_len: float = 0.3,
    seed: Union[None, int] = None,
    training: bool = False,
    extend: bool = True,
    padding: float = 0.0,
    shuffle_order: bool = True,
) -> Tuple[list, int]:
    """
    Takes a path to audio files and a path to segment files.
    returns segmented audio and the sample rate of the audio.
    Audio files are memory-mapped, so only the segments are read into memory (but all of the segments
    returned are held in memory at once: see `iter_segmented_audio` for one file at a time).

    Inputs
    -----
        - audio_path: path to a set of audio files
        - seg_path: path to a set of .txt files with vocalization onsets and offsets
        - audio_id: common ending to all audio files. Used to get
            .txt filenames
        - max_vocs: number of vocalizations to extract
        - context_len: length of vocal chunks. used for training
        - seed: random seed used when shuffling filenames
        - training: whether to generate audio for training or analysis
        - extend: whether to extend onsets for training chunks
        - padding: how much to pad onsets of vocalizations for analysis chunks
        - shuffle_order: whether to shuffle filenames before extracting vocalizations
    Returns
    -----
        - a list of collected audio
        - the sample rate of the audio
    """

    audio_files, seg_files = list_audio_files(audio_path, seg_path, audio_id, seed, shuffle_order)

    if training:
        audio_segments, sr = get_audio_training(
            audio_files,
            seg_files,
            max_vocs=max_vocs,
            context_len=context_len,
            extend=extend,
        )

    else:
        audio_segments, sr = get_audio_analysis(
            audio_files, seg_files, max_vocs=max_vocs, padding=padding
        )

    return audio_segments, sr


def load_file_segments(
    audio_file: str,
    seg_file: str,
    context_len: float = 0.3,
    training: bool = False,
    extend: bool = True,
    padding: float = 0.0,
) -> Tuple[list, Union[int, None]]:
    """
    the segmented audio of a single file (the unit that `iter_segmented_audio` streams). Arguments are as in
    `get_segmented_audio`. Returns the list of audio segments (empty if the audio or segment file is
    missing, or there are no segments) and the sampling rate (None if the files are missing)
    """
    if training:
        return get_audio_training(
            [audio_file],
            [seg_file],
            max_vocs=np.inf,
            context_len=context_len,
            extend=extend,
        )
    return get_audio_analysis([audio_file], [seg_file], max_vocs=np.inf, padding=padding)


def iter_segmented_audio(
    audio_path: str,
    seg_path: str,
    audio_id: str = ".wav",
    context_len: float = 0.3,
    seed: Union[None, int] = None,
    training: bool = False,
    extend: bool = True,
    padding: float = 0.0,
    shuffle_order: bool = True,
) -> Iterator[Tuple[str, list, int]]:
    """
    the streaming version of `get_segmented_audio`: yields the segmented audio of one file at a time,
    so that the audio of only one file need be in memory at once. Arguments are as in
    `get_segmented_audio`, except there is no `max_vocs`; stop iterating when you have enough.
    Files are visited in the same order as `get_segmented_audio` visits them, and give the same audio, so
    concatenating the segments from every file gives what `get_segmented_audio` returns with no limit.
    Audio files with no segment file, or with no segments, are skipped.

    Yields
    -----
        - audio_file: path of the audio file
        - segments: list of the audio segments from that file
        - sr: the sampling rate
    """
    audio_files, seg_files = list_audio_files(audio_path, seg_path, audio_id, seed, shuffle_order)

    for a, sf in zip(audio_files, seg_files):
        segments, sr = load_file_segments(
            a, sf, context_len=context_len, training=training, extend=extend, padding=padding
        )
        if len(segments) > 0:
            yield a, segments, sr
