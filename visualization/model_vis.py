import numpy as np
from utils import get_spec, deriv_approx_d2y, deriv_approx_dy, from_numpy
import torch
import matplotlib.pyplot as plt
import gc
from matplotlib.animation import FuncAnimation

# import librosa
import os
import matplotlib as mpl
import seaborn as sns

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 22


def format_axes(ax, xlabel="", ylabel="", xlims=(), ylims=()):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", which="both", direction="in")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(xlims) == 2:
        ax.set_xlim(xlims)
    if len(ylims) == 2:
        ax.set_ylim(ylims)
    return


def r2_plot(means, sds, labels=["train,test"], saveloc="", show=False):

    cm = mpl.colormaps["tab10"]
    ax = plt.gca()
    bars = ax.bar(
        np.arange(0, len(means) // 2 + 0.5, 0.5)[: len(means)], means, width=0.35
    )
    ebs = ax.errorbar(
        np.arange(0, len(means) // 2 + 0.5, 0.5)[: len(means)],
        means,
        yerr=sds,
        linestyle=" ",
        color="k",
        capsize=20,
    )
    ax.set_xticks(
        np.arange(0, len(means) // 2 + 0.5, 0.5)[: len(means)], labels, rotation=35
    )
    for ii, b in enumerate(bars):
        b.set_facecolor(cm(ii))

    plt.savefig(os.path.join(saveloc, "r2s.svg"))
    if show:
        plt.show()
    plt.close


def loss_plot(train_loss, val_loss, save_loc="", show=True):

    train_loss, val_loss = np.array(train_loss), np.array(val_loss)

    ax = plt.gca()
    ax.plot(np.log10(train_loss), color="tab:blue", label="Train loss")
    ax.plot(
        val_loss[:, 0],
        np.log10(val_loss[:, 1]),
        color="tab:orange",
        label="Validation loss",
    )
    # ax.set_xlabel("Gradient steps")
    # ax.set_ylabel("Loss (MSE)")

    format_axes(ax, xlabel="Gradient steps", ylabel="Model performance (log10 MSE)")
    plt.legend()
    plt.savefig(os.path.join(save_loc, "train_test_loss.svg"))
    if show:
        plt.show()

    plt.close()


def get_kernel_grid(weights, grid_points, kernel):

    # weights should be 1 x L x P x P
    # grid points should be B x 1 x D
    # kernel should be a kernel
    _, L, P, P = weights.shape
    B, _, d = grid_points.shape

    kerns = []
    for ind in range(L):
        w = weights[:, ind : ind + 1, :, :]
        w = np.tile(w, [B, 1, 1, 1])
        kern = kernel.forward_given_weights_numpy(grid_points, w)
        kerns.append(kern.squeeze())  # should append a B member

    return np.stack(kerns, axis=0)  # should now be L x B


def visualize_kernel(model, y, dy, dt):

    # y: B x L x d
    y = y[None, :, :]
    smooth_len = int(round(model.smooth_len / dt))
    # dy = deriv_approx_dy(y)
    z = torch.cat([y, dy], dim=-1)
    x_in = torch.cat([z, torch.flip(z, [1])], dim=-1)

    weight_control = model.kernel_mamba(x_in)
    weighted_kernels, weights = model.kernel(z, weight_control, smooth_len) * model.tau
    y, dy = y.detach().cpu().numpy().squeeze(), dy.detach().cpu().numpy().squeeze() / dt
    spec, t, f, _ = get_spec(
        y,
        int(round(1 / dt)),
        onset=0,
        offset=len(y) * dt,
        shoulder=0.0,
        interp=False,
        win_len=1028,
        min=-2,
        max=3.5,
    )
    weighted_kernels = weighted_kernels.detach().cpu().numpy().squeeze()
    weights = weights.detach().cpu().numpy().squeeze()  # should be 1 x L x P x P
    assert weights.shape == (
        1,
        z.shape[1],
        model.kernel.nTerms + 1,
        model.kernel.nTerms + 1,
    ), print(weights.shape)

    ygrid = np.linspace(-1, 1, 50)
    dygrid = np.linspace(-1, 1, 50) / dt
    ygrid, dygrid = np.meshgrid(ygrid, dygrid)
    ygrid, dygrid = ygrid.flatten(), dygrid.flatten()
    zgrid = np.stack([ygrid, dygrid], axis=-1)[:, None, :]  # now B x 1x 2d
    kerns = get_kernel_grid(weights, zgrid, model.kernel)
    kerns = np.reshape(kerns, [kerns.shape[0], 50, 50])
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224)

    kmax = np.amax(kerns.abs())

    def init():

        s = ax1.imshow(
            spec, aspect="auto", origin="lower", extent=[t[0], t[-1], f[0], f[-1]]
        )
        (l,) = ax1.vlines(0, ymin=f[0], ymax=f[-1], color="tab:red")
        ax1.set_title("Song")
        w = ax2.matshow(weights[0, 0, :, :])
        ax2.set_title("Weights at time 0.0")
        k = ax3.imshow(kerns[0, :, :], vmin=-kmax, vmax=kmax)
        ax3.set_title("Kernel at time 0.0")
        return s, l, w, k

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))
    g = axs[1].scatter(y, dy, c=weighted_kernels)
    axs[1].set_xlabel("y")
    axs[1].set_ylabel("dy")
    cb = plt.colorbar(g, ax=axs[1])
    cb.set_label("Weighted kernel value", rotation=270, labelpad=20)
    axs[0].imshow(
        spec,
        vmin=0,
        vmax=1,
        origin="lower",
        extent=[t[0], t[-1], f[0], f[-1]],
        aspect="auto",
    )
    plt.show()
    plt.close()


def quantities_figure(
    model, audioData, sr, nSamples=10, save=False, filename="./quants.svg"
):

    samples = np.random.choice(len(audioData), nSamples)
    dt = 1 / sr
    for jj, s in enumerate(samples):
        aud = audioData[s]
        spec, spec_t, freq, _ = get_spec(
            aud[:, -1],
            sr,
            onset=4 * dt,
            offset=(len(aud) - 4) / sr,
            shoulder=0.0,
            interp=False,
        )

        aud = aud[None, :, :]
        dy = deriv_approx_dy(aud)
        d2y = deriv_approx_d2y(aud)
        aud = aud[:, 4:-4, :]
        L = aud.shape[1]

        terms = model.get_funcs(from_numpy(aud), dt)
        terms = [t.detach().cpu().numpy() for t in terms]
        omega, gamma, b, b_out, states = model.get_funcs(from_numpy(aud), dt)
        omega, gamma, b, b_out, states = (
            omega.detach().cpu().numpy(),
            gamma.detach().cpu().numpy(),
            b.detach().cpu().numpy(),
            b_out.detach().cpu().numpy(),
            states.detach().cpu().numpy(),
        )

        t = np.arange(4 * dt, len(aud) * dt + 4 * dt, dt)[:L]
        fig, axs = plt.subplots(2, 1, figsize=(20, 20))

        on, off = 0.1, 0.2
        onInd = int(round(on * sr))
        offInd = int(round(off * sr))
        onIndSpec, offIndSpec = (
            np.searchsorted(spec_t, on),
            np.searchsorted(spec_t, off),
        )

        fig, axs = plt.subplots(
            nrows=len(model.names), ncols=1, figsize=(10, 3 * len(model.names))
        )
        for ax, term, name in zip(axs[1:], terms, model.names):
            ax.plot(t[onInd:offInd], term.squeeze()[onInd:offInd], label=name)
            ax.legend(frameon=False)
            format_axes(ax, xticks=ax.get_xticks(), yticks=ax.get_yticks())

        axs[0].imshow(
            spec[:, onIndSpec:offIndSpec],
            extent=[on, off, freq[0], freq[-1]],
            origin="lower",
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        format_axes(axs[0])

    plt.savefig(filename)
    plt.show()
    plt.close()


#### gif plot of y vs ydot, omega vs gamma, avged kernel


def grab_segments(seg_list, *args, fs=42000):

    #### IMPLEMENT
    output = [[] for _ in args]
    for ii, onoffs in enumerate(seg_list):
        # onoffs = np.loadtxt(seg_file)
        if np.ndim(onoffs) == 1:
            onoffs = onoffs[None, :]
        for onInd, offInd in onoffs:
            # onInd, offInd = int(round(on*fs)),int(round(off*fs))
            for jj, inputs in enumerate(args):
                # print(inputs[ii].squeeze()[onInd:offInd].shape)
                output[jj].append(inputs[ii][onInd:offInd])

        # print(len(output[0]))
        # assert False
        # print(output)
    return output


def pad_with_nans(x, target_length, axis=0):

    currlen = x.shape[axis]
    currshape = list(x.shape)
    diff = target_length - currlen
    currshape[axis] = diff

    if diff > 0:
        padding = np.full(currshape, np.nan)

        return np.concatenate([x, padding], axis=axis)
    else:
        return x

"""
def make_gif(
    model,
    syllable,
    ages,
    audio_paths,
    segment_paths,
    audio_filetype=".wav",
    seg_filetype=".txt",
    save_loc="",
):

    # assumes all ages, audio paths are already sorted by age
    all_days_os = []
    all_days_gs = []
    assert len(ages) == len(audio_paths), print(
        "need to have the same number of ages as audio paths!"
    )
    for ap, sp in zip(audio_paths, segment_paths):
        (mean_omega, mean_gamma), _ = analysis.assess_variability(
            model, ap, sp, audio_filetype, seg_filetype
        )
        all_days_os.append(mean_omega)
        all_days_gs.append(mean_gamma)

    gc.collect()
    max_len = np.amax(list(map(len, all_days_os)))
    padder = lambda x: pad_with_nans(x, target_length=max_len, axis=0)
    o1 = np.stack(list(map(padder, all_days_os)), axis=0)
    g1 = np.stack(list(map(padder, all_days_gs)), axis=0)
    max_o = np.nanmax(o1)
    min_o = np.nanmin(o1)
    min_g = np.nanmin(g1)
    max_g = np.nanmax(g1)

    cs = sns.color_palette("viridis", len(o1))
    cm = sns.color_palette("viridis", as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=ages[0], vmax=ages[-1]))

    def init(ax):
        dayLines = []
        for ii, (omega, gamma) in enumerate(zip(o1, g1)):
            (l,) = ax.plot(gamma[0], omega[0], color=cs[ii])
            dayLines.append(l)

        format_axes(
            ax,
            ylabel=r"$\omega$",
            xlabel=r"$\gamma$",
            ylims=(min_o - 1000, max_o + 1000),
            xlims=(min_g - 1000, max_g + 1000),
        )
        cb1 = plt.colorbar(sm, ax=ax)
        cb1.set_label("Age of bird (days post hatch)", rotation=270, labelpad=80)

        return dayLines

    fig = plt.figure(figsize=(15, 15))
    ax = plt.gca()
    lines = init(ax)
    step = 5
    nFrames = o1.shape[1] // step + 400 // step

    def animate(i):
        print(f"{round(i / nFrames * 100)}%", end="\r")
        for l, omega, gamma in zip(lines, o1, g1):
            plot_end = min(i * step, len(omega))
            l.set_data(gamma[:plot_end], omega[:plot_end])

        return (lines,)

    ani = FuncAnimation(fig, func=animate, frames=nFrames, interval=10)
    ani.save(
        os.path.join(save_loc, f"anim_test_{syllable}_transparent.mp4"),
        writer="ffmpeg",
        savefig_kwargs={"transparent": True},
    )
    plt.close()

    return
"""