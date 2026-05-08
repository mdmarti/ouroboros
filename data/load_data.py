import numpy as np
import warnings

import os
import glob
from scipy.io import wavfile


from typing import Tuple, Union


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

    for a, s in zip(audio_files, seg_files):
        if os.path.isfile(a) and os.path.isfile(s):
            sr, aud = wavfile.read(a)  # here, we assume
            # all loaded files have the same sample rate
            if aud.dtype == np.int16:  # ints are TOO BIG!! turn into floats
                aud = aud / -np.iinfo(aud.dtype).min

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
                aud_sample = aud[on_ind:off_ind]

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

    for a, s in zip(audio_files, seg_files):
        if os.path.isfile(a) and os.path.isfile(s):
            sr, aud = wavfile.read(a)  # here, we assume
            # all loaded files have the same sample rate

            if aud.dtype == np.int16:  # ints are TOO BIG!! turn into floats
                aud = aud / -np.iinfo(aud.dtype).min

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

    gen = np.random.default_rng(seed=seed)

    audio_files = glob.glob(os.path.join(audio_path, "*" + audio_id))

    if shuffle_order:
        order = gen.choice(len(audio_files), len(audio_files), replace=False)
        audio_files = [audio_files[o] for o in order]
        # seg_files = [seg_files[o] for o in order]

    audio_tags = [a.split("/")[-1].split(audio_id)[0] for a in audio_files]
    seg_files = [os.path.join(seg_path, a + ".txt") for a in audio_tags]

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
