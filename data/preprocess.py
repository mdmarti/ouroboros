from ssqueezepy import ssq_cwt, issq_cwt, ssq_stft
from ssqueezepy.visuals import imshow, plot

from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
import glob
from utils import butter_filter
import noisereduce as nr

from typing import Union, Tuple

plt.rcParams

# Synchrosqueezing preprocessing code is based on , please check out this
# library! it's very good

### wavelets:
# morlet: higher mu -> greater frequency, lesser time resolution. for mu > 6
# wavelet almost exactly gaussian. mu=13.4: matches generalized morse wavelets (3,60)
# generalized morse wavelet (gmw): options are time & frequency spread
# bump: wider variance in time, narrower variance in frequency
# cmhat: complex mexican hat. wider frequency variance, narrower time variance
# hhhat: hilbert analytic function of hermitian hat. no idea what this does, don't use it
#
HP_DICT = {
    "chunk length": 10000,
    "wavelet": "morlet",  # options are gmw, morlet, bump, cmhat,hhhat
    "band min": 1000.0,
    "band max": 10000.0,
    "nv": 32,  # number of voices (wavelets per octave),
    "scales": "log-piecewise",
    "order": 5,  # polynomial order for band-pass filter,
    "prop_reduce": 1.0,  # proportion of noise to reduce, if reducing noise
    "time_constant_s": 0.4,  # time_constant for assuming stationarity of noise. increase if noise is more consistent
    "squeeze_freqs": True,
}

WAVELET_HP_DICT = {
    "morlet": {"mu": 13.4},
    "bump": {"mu": 5, "s": 1, "om": 0},
    "cmhat": {"mu": 1, "s": 1},
    "hhhat": {"mu": 5},
    "gmw": {
        "gamma": 6,
        "beta": 60,
    },
}

FILTER_DICT = {"chunk length": 10000, "band min": 1000.0, "band max": 10000.0}


def viz(x, Tx, Wx, vmin=None, vmax=None, axs=True):
    """
    visualization function from ssqueezepy
    """
    ax = plt.gca()
    if vmin is not None and vmax is not None:
        plt.imshow(np.abs(Wx), aspect="auto", cmap="turbo", vmin=vmin, vmax=vmax)
    else:
        plt.imshow(np.abs(Wx), aspect="auto", cmap="turbo")
    if not axs:
        print("removing ticks")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def lin_band(
    Tx: np.ndarray, slope: float, offset: float, bw: float, show: bool = True, **kw
) -> Tuple[np.ndarray, np.ndarray]:
    """
    create a linear band of frequencies (or scales, on the scaleogram
    from a SSQ transform) to preserve. Adapted from
    ssqueezepy.

    Inputs
    -----
        - Tx: frequency or scaleogram
        - slope: slope of linear band
        - offset: vertical offset (in frequency of scale space)
            of linear band
        - bw: width of linear band, as a portion of frequencies/scales
            ranges between 0,1
        - show: whether to plot and show results immediately
        - **kw: visualization keywords

    Returns
    -----
        Cs: lower bound of frequency band, in terms of scaelogram
        freqband: full frequency band
    """

    na, N = Tx.shape
    tcs = np.linspace(0, 1, N)
    Cs = (slope * tcs + offset) * na
    freqband = bw * na * np.ones(N)
    Cs, freqband = Cs.astype("int32"), freqband.astype("int32")
    # print(Cs[0],freqband[0])
    # print("")
    if show:
        imshow(Tx, abs=1, aspect="auto", show=0, **kw)
        plot(Cs + freqband, color="r")
        plot(Cs - freqband, color="r", show=1)
    return Cs, freqband


def band_pass_preprocess(
    data: np.ndarray,
    chunk_len: float,
    low_cut: float,
    high_cut: float,
    fs: float,
    kw: dict,
    tn: np.ndarray,
    return_full_ssq: bool = True,
    order: int = 5,
) -> Tuple[np.ndarray, Union[np.ndarray, None], Union[np.ndarray, None]]:
    """
    Remove certain frequencies using a band-pass filter

    Inputs
    -----
        - data: data to filter
        - chunk_len: length of audio to filter at once
        - low_cut: low frequency to cut
        - high_cut: high frequency to cut
        - fs: samplerate of audio
        - kw: synchrosqueezing keywords
        - tn: sample timepoints of audio
        - return_full_ssq: whether to return original frequencies, scaleogram
            in addition to processed audio
        - order: order of butterworth filter for band-passing

    Returns
    -----
        - full_rec: filtered data
        If return_full_ssq:
            - frequencies for scaleogram
            - original scaleogram
    """

    print("using band-pass preprocessing")

    full_rec = []
    chunk_ons = np.arange(0, len(data), chunk_len)
    for ii, on in enumerate(chunk_ons):
        off = min(len(data), on + chunk_len)
        filtered = butter_filter(
            data[on:off], np.array([low_cut, high_cut]), fs, order=order, btype="band"
        )
        full_rec.append(filtered)
    full_rec = np.hstack(full_rec)
    if return_full_ssq:
        # print(full_rec.shape)

        full_ssq, _, full_freqs, full_scales, *_ = ssq_cwt(data, t=tn, **kw)
        return full_rec, full_freqs, full_ssq
    return full_rec, None, None


def ssq_preprocess(
    data: np.ndarray,
    tn: np.ndarray,
    kw: dict,
    chunk_len: float,
    show: bool = True,
    min_band: float = 0.01,
    max_band: float = 0.35,
    return_full_ssq: bool = True,
) -> Tuple[np.ndarray, Union[np.ndarray, None], Union[np.ndarray, None]]:
    """
    Remove certain frequencies using a linear band on synchrosqueezed scaleogram

    Inputs
    -----
        - data: data to filter
        - tn: sample timepoints of audio
        - kw: synchrosqueezing keywords
        - chunk_len: length of audio to filter at once
        - show: whether to directly plot outputs
        - min_band: bottom of retained frequency band
        - max_band: top of retained frequency band
        - return_full_ssq: whether to return original frequencies, scaleogram
            in addition to processed audio

    Returns
    -----
        - full_rec: filtered data
        If return_full_ssq:
            - frequencies for spectrogram
            - original spectrogram
    """

    full_rec = []
    slopen = -0.00  # flat band across time

    chunk_ons = np.arange(0, len(data), chunk_len)
    # print(data.shape,tn.shape)

    for ii, on in enumerate(chunk_ons):
        off = min(len(data), on + chunk_len)
        nrec = np.zeros(data.shape)
        Tn, _, ssq_freqs, scales, *_ = ssq_cwt(data[on:off], t=tn[on:off], **kw)
        if max_band > 1:
            nf = Tn.shape[0]

            max_band_new = np.sum(ssq_freqs >= min_band) / nf
            min_band_new = np.sum(ssq_freqs >= max_band) / nf

            min_band = min_band_new
            max_band = max_band_new

        bwn = (max_band - min_band) / 2
        offsetn = (min_band + max_band) / 2
        Csn, freqbandn = lin_band(Tn, slopen, offsetn, bwn, norm=(0, 4e-1), show=show)
        nrec = issq_cwt(Tn, kw["wavelet"], Csn, freqbandn)[0]
        full_rec.append(nrec)

    full_rec = np.hstack(full_rec)
    print("", end="\r", flush=True)
    if show:
        ax = plt.gca()
        ax.plot(data[1200:1400], label="data")
        ax.plot(full_rec[1200:1400], alpha=0.25, label="reconstruction")
        plt.legend()
        plt.show()
        plt.close()
        Tsxo, Sxo, *_ = ssq_stft(data)
        viz(
            data,
            np.flipud(Tsxo),
            np.flipud(Sxo),
            vmin=np.amin(np.abs(Sxo)),
            vmax=np.amax(np.abs(Sxo)),
        )
        Tsx, Sx, *_ = ssq_stft(full_rec)
        viz(
            full_rec,
            np.flipud(Tsx),
            np.flipud(Sx),
            vmin=np.amin(np.abs(Sxo)),
            vmax=np.amax(np.abs(Sxo)),
        )
    if return_full_ssq:
        full_ssq, _, full_freqs, full_scales, *_ = ssq_cwt(data, t=tn, **kw)

        return full_rec, full_freqs, full_ssq
    else:
        return full_rec, ssq_freqs, None


def check_valid(data, dtype, default=""):
    """
    checks if inputs (from terminal) are valid,
    based on expected dtype.

    Inputs
    -----
        - data: data to check dtype of
        - dtype: expected dtype
        - default: default value of input

    Returns
    -----
        input as expected dtype
        flag of whether data was valid or not
    """

    if data == default:
        return data, True
    try:
        d = dtype(data)
        return d, True
    except ValueError:
        return data, False


def _tune_input_helper(p: dict) -> dict:
    """Get parameter adjustments from the user.

    Inputs
    -----
        p: dictionary of tuning parameters
    Returns
    -----
        updated tuning parameters

    """
    for key in p.keys():
        curr_dtype = type(p[key])
        # temp = 'not (number or empty)'
        temp = input("Set value for " + key + ": [" + str(p[key]) + "] ")
        temp, valid = check_valid(temp, curr_dtype)
        while not valid:
            temp = input("Set value for " + key + ": [" + str(p[key]) + "] ")
            temp, valid = check_valid(temp, curr_dtype)
        if temp != "":
            p[key] = temp
    return p


def _tuning_plot(
    orig_spec: np.ndarray,
    denoised_spec: np.ndarray,
    cleaned_spec: np.ndarray,
    ts: np.ndarray,
    spec_freqs: np.ndarray,
    scale_freqs: np.ndarray,
    scaleogram: np.ndarray,
    min_band: float,
    max_band: float,
    vmin: float = -0.5,
    vmax: float = 1.5,
    save_loc: str = "./pp.pdf",
):
    """
    Make a plot to assess parameter tuning

    Inputs
    -----
        - orig_spec: original spectrogram
        - denoised_spec: denoised spectrogram (using noisereduce)
        - cleaned_spec: cleaned spectrogram (removing certain frequency bands)
        - ts: audio timepoints
        - spec_freqs: spectrogram frequencies
        - scale_freqs: scaleogram frequencies
        - scaleogram: scaleogram
        - min_band: minimum of linear band of retained frequencies
        - max_band: maximum of lienar band of retained frequencies
        - vmin: minimum value for plots
        - vmax: maximum value for plots
        - save_loc: location to save out the tuning plot
    Returns
    -----
        nothing
    """

    nf, T = scaleogram.shape
    # assert len(scale_freqs) == ns, print(ns,len(scale_freqs))
    # scale_extent[2] = 0
    # scale_extent[3] = ns
    if max_band > 1:
        max_band_new = np.sum(scale_freqs >= min_band) / nf
        min_band_new = np.sum(scale_freqs >= max_band) / nf

        min_band = min_band_new
        max_band = max_band_new
    bw = (max_band - min_band) / 2
    offset = (max_band + min_band) / 2
    # print(offset,bw)

    band = bw * np.ones(T) * nf
    Cs = offset * nf * np.ones(T)
    # print(Cs[0])
    # print(band[0])

    vmin_scale = np.amin(scaleogram)
    vmax_scale = np.amax(scaleogram)
    vmax_scale = (vmax_scale - vmin_scale) / 50 + vmin_scale
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(13, 4))
    spec_extent = [ts[0], ts[-1], spec_freqs[0], spec_freqs[-1]]
    scale_extent = [ts[0], ts[-1], nf, 0]
    axs[0].imshow(
        orig_spec,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        extent=spec_extent,
    )
    axs[1].imshow(
        denoised_spec,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        extent=spec_extent,
    )
    axs[2].imshow(
        cleaned_spec,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        extent=spec_extent,
    )
    axs[3].imshow(
        scaleogram,
        aspect="auto",
        cmap="bone",
        vmin=vmin_scale,
        vmax=vmax_scale,
        extent=scale_extent,
    )
    # axs[2].set_xticks(np.arange(0,len(ts),len(ts)/10),ts[::int(round(len(ts)/10))])
    axs[3].set_yticks(
        np.arange(0, nf, 25), np.round(scale_freqs[::25]).astype(np.int32)
    )
    axs[3].plot(ts, Cs - band, color="r")
    axs[3].plot(ts, Cs + band, color="r")
    axs[0].set_ylabel("Frequency (Hz)")
    axs[0].set_title("Original spectrogram")
    axs[1].set_title("Denoised spectrogram")
    axs[2].set_title("Reconstructed spectrogram (after ssq)")
    axs[3].set_title("CWT with reconstruction boundaries")
    for ax in axs:
        ax.set_xlabel("Time (s)")
    plt.tight_layout
    plt.savefig(save_loc, transparent=True, bbox_inches="tight")
    plt.close("all")


def tune_preprocessing(
    audio_dirs: list[str],
    segment_dirs: list[str],
    hp_dict: dict,
    preprocess_type: str = "ssq",
    img_fn: str = "./pp.pdf",
    reduce_noise: bool = True,
) -> dict:
    """
    tune preprocessing parameters for denoising spectrograms

    Inputs
    ----------
        - audio_files : list of str
            Audio files
        - seg_files : list of str
            Segment files
        - hp_dict : dict
            Preprocessing parameters
        - preprocess_type: ssq or band-pass
        - img_fn: save filename for tuning plot
        - reduce noise: whether to use noise_reduce prior to linear filtering

    Returns
    -------
    p : dict
        Adjusted preprocessing parameters.

    """

    assert len(audio_dirs) == len(segment_dirs), print(
        "number of audio files is not same as number of segments!"
    )

    audio_files = [glob.glob(os.path.join(ad, "*.wav")) for ad in audio_dirs]
    segment_files = [
        [os.path.join(sd, af.split("/")[-1].split(".wav")[0] + ".txt") for af in ad]
        for ad, sd in zip(audio_files, segment_dirs)
    ]

    audio_files = sum(audio_files, [])
    segment_files = sum(segment_files, [])

    while True:
        p = _tune_input_helper(hp_dict)
        resp = "nothing yet"
        while resp != "s" and resp != "r":
            ind = np.random.choice(len(audio_files))
            audio_fn = audio_files[ind]
            seg_fn = segment_files[ind]
            print(audio_fn, seg_fn)

            onoffs = np.loadtxt(seg_fn, usecols=(0, 1))
            while len(onoffs) == 0:
                ind = np.random.choice(len(audio_files))
                audio_fn = audio_files[ind]
                seg_fn = segment_files[ind]
                print(audio_fn, seg_fn)
                onoffs = np.loadtxt(seg_fn, usecols=(0, 1))
            if len(onoffs.shape) == 1:
                onoffs = onoffs[None, :]

            a_ind = np.random.choice(len(onoffs))
            on, off = onoffs[a_ind, 0], onoffs[a_ind, 1]

            print(on, off)
            if ".wav" in audio_fn:
                sr, a = wavfile.read(audio_fn)

            elif ".flac" in audio_fn:
                a, sr = sf.read(audio_fn)

            orig_dtype = a.dtype
            on_ind, off_ind = int(round(on * sr)), int(round(off * sr))

            curr_len = off_ind - on_ind
            if len(a) < p["chunk length"]:
                print("short audio,skipping")
                continue
            if curr_len < p["chunk length"]:
                diff = p["chunk length"] - curr_len
                off_ind += diff
                off += diff / sr

                print(f"extending segment by {diff / sr:.2f}s")

            orig_audio = a[on_ind:off_ind]
            # print(orig_audio.shape)
            _, sx_orig, _, sx_freqs, *_ = ssq_stft(orig_audio, fs=sr)
            orig_spec = np.abs(sx_orig)
            vmin = np.amin(orig_spec)
            vmax = np.amax(orig_spec)
            vmax = (vmax - vmin) / 10 + vmin

            if reduce_noise:
                noise_reduced_chunk_on = max(0, on_ind - sr)
                on_diff = on_ind - noise_reduced_chunk_on
                noise_reduced_chunk_off = min(len(a), off_ind + sr)
                off_diff = noise_reduced_chunk_off - off_ind
                nyquist = sr // 2
                freq_mask_smooth_hz = int(
                    round(2.5 * nyquist / 100)
                )  # default: 500, scale by frequency range (let's do 2.5% of frequency range)
                a = nr.reduce_noise(
                    y=a[noise_reduced_chunk_on:noise_reduced_chunk_off],
                    sr=sr,
                    prop_decrease=p["prop_reduce"],
                    time_constant_s=p["time_constant_s"],
                    stationary=False,
                    freq_mask_smooth_hz=freq_mask_smooth_hz,
                )
                orig_audio = a[on_diff:-off_diff]
            # print(orig_audio.shape)
            # print(len(orig_audio)/sr, off-on)

            t = np.arange(0, off - on, 1 / sr)[: len(orig_audio)]

            _, sx_denoised, *_ = ssq_stft(orig_audio, fs=sr)
            denoised_spec = np.abs(sx_denoised)
            orig_spec = np.abs(sx_orig)
            cwt_kws = {
                "wavelet": (p["wavelet"], WAVELET_HP_DICT[p["wavelet"]]),
                "nv": p["nv"],
                "scales": p["scales"],
            }

            if p["squeeze_freqs"]:
                band_min = p["band min"]
                band_max = p["band max"]
            else:
                band_min = 0
                band_max = int(round(sr) / 2)
            if preprocess_type == "ssq":
                # processing ssq cwt
                recon_a, cwt_freqs, ssq_scaleogram = ssq_preprocess(
                    orig_audio,
                    t,
                    cwt_kws,
                    p["chunk length"],
                    show=False,
                    min_band=band_min,
                    max_band=band_max,
                    return_full_ssq=True,
                )
            else:
                recon_a, cwt_freqs, ssq_scaleogram = band_pass_preprocess(
                    orig_audio,
                    p["chunk length"],
                    low_cut=band_min,
                    high_cut=band_max,
                    fs=sr,
                    return_full_ssq=True,
                    kw=cwt_kws,
                    tn=t,
                    order=p["order"],
                )
            recon_a = recon_a.astype(orig_dtype)
            wavfile.write("./test_wav.wav", rate=sr, data=recon_a)
            _, sx_recon, *_ = ssq_stft(recon_a, fs=sr)
            recon_spec = np.abs(sx_recon)
            scaleogram = np.abs(ssq_scaleogram)

            _tuning_plot(
                orig_spec=orig_spec,
                denoised_spec=denoised_spec,
                cleaned_spec=recon_spec,
                ts=t,
                spec_freqs=sx_freqs,
                scale_freqs=cwt_freqs,
                scaleogram=scaleogram,
                min_band=p["band min"],
                max_band=p["band max"],
                vmin=vmin,
                vmax=vmax,
                save_loc=img_fn,
            )

            resp = input("Continue? [y] or [s]top tuning or [r]etune params: ")
            if resp == "s":
                return p


def filter_by_tags(audio_files, seg_files, audio_tags, seg_tags):

    ### assumes tags, files are all sorted already

    # filtered_audio_files = []
    # filtered_seg_files = []

    all_endings = set(audio_tags).intersection(seg_tags)

    filtered_audio_files = [
        w for w, t in zip(audio_files, audio_tags) if t in all_endings
    ]
    filtered_seg_files = [s for s, t in zip(seg_files, seg_tags) if t in all_endings]

    return filtered_audio_files, filtered_seg_files


def preprocess_helper(
    audio_file: str,
    out_dir: str,
    hyperparameters: dict,
    reprocess: bool,
    preprocess_type: str,
    reduce_noise: bool,
):
    """
    helper function for preprocessing audio

    Inputs
    -----
        - audio_file:
        - out_dir: location to put processed audio file
        - hyperparameters: preprocessing hyperparams
        - audio_id: extension of original audio files
        - reprocess: whether to reprocess, if there are existing files
        - preprocess_type: ssq or band-pass
        - reduce_noise: whether to use noise_reduce
    """

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    new_fn = audio_file.split("/")[-1]
    new_fn = os.path.join(out_dir, new_fn)
    if os.path.isfile(new_fn) and not reprocess:
        return

    if ".wav" in audio_file:
        sr, orig_audio = wavfile.read(audio_file)
    elif ".flac" in audio_file:
        orig_audio, sr = sf.read(audio_file)

    orig_dtype = orig_audio.dtype
    if reduce_noise:
        nyquist = sr // 2
        freq_mask_smooth_hz = int(
            round(2.5 * nyquist / 100)
        )  # default: 500, scale by frequency range (let's do 2.5% of frequency range)

        orig_audio = nr.reduce_noise(
            y=orig_audio,
            sr=sr,
            prop_decrease=hyperparameters["prop_reduce"],
            time_constant_s=hyperparameters["time_constant_s"],
            stationary=False,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
        )

    cwt_kws = {
        "wavelet": (
            hyperparameters["wavelet"],
            WAVELET_HP_DICT[hyperparameters["wavelet"]],
        ),
        "nv": hyperparameters["nv"],
        "scales": hyperparameters["scales"],
    }
    try:
        t = np.arange(0, len(orig_audio) / sr, 1 / sr)[: len(orig_audio)]
        if hyperparameters["squeeze_freqs"]:
            if preprocess_type == "ssq":
                recon_a, *_ = ssq_preprocess(
                    orig_audio,
                    t,
                    cwt_kws,
                    hyperparameters["chunk length"],
                    show=False,
                    min_band=hyperparameters["band min"],
                    max_band=hyperparameters["band max"],
                    return_full_ssq=False,
                )
            elif preprocess_type == "band-pass":
                recon_a, *_ = band_pass_preprocess(
                    orig_audio,
                    hyperparameters["chunk length"],
                    low_cut=hyperparameters["band min"],
                    high_cut=hyperparameters["band max"],
                    fs=sr,
                    return_full_ssq=True,
                    kw=cwt_kws,
                    tn=t,
                )
        else:
            recon_a = orig_audio

    except Exception:
        print(f"error in processing {audio_file}")
        print(t.shape)
        print(orig_audio.shape)
        raise
    recon_a = recon_a.astype(orig_dtype)
    wavfile.write(new_fn, rate=sr, data=recon_a)


def preprocess(
    audio_dir: str,
    out_dir: str,
    hp_dict: dict,
    reprocess: bool = True,
    preprocess_type: str = "ssq",
    reduce_noise: bool = True,
):
    """
    function for preprocessing your audio

    Inputs
    -----
        - audio_files: audio file directories to process
        - out_dir: locations to put processed audio files
        - hyperparameters: preprocessing hyperparams
        - audio_ext: extension of original audio files
        - parallel: whether to process in parallel
        - reprocess: whether to reprocess, if there are existing files
        - preprocess_type: ssq or band-pass
        - reduce_noise: whether to use noise_reduce
    """

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    print(f"cleaning {len(audio_files)} files in {audio_dir}")
    for in_file in audio_files:
        preprocess_helper(
            in_file, out_dir, hp_dict, reprocess, preprocess_type, reduce_noise
        )
    print(f"done cleaning {len(audio_files)} files in {audio_dir}")
