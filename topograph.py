import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from matplotlib import patches
from matplotlib.colors import Normalize
from scipy import signal


def get_psds_theta(data, fs=250, f_range=[4, 8]):
    """
    Calculate signal power using Welch method.
    Input: data- mxn matrix (m: number of channels, n: samples of signals)
           fs- Sampling frequency (default 128Hz)
           f_range- Frequency range (default 0.5Hz to 30Hz)
    Output: Power values and PSD values
    """
    powers = []
    psds = list()
    for sig in data:
        freq, psd = signal.welch(sig, fs)
        idx = np.logical_and(freq >= f_range[0], freq <= f_range[1])
        powers = np.append(powers, sum(psd[idx]))
        psds.append(psd[idx])

    return powers, psds


def get_psds_alpha(data, fs=250, f_range=[8, 12]):
    """
    Calculate signal power using Welch method.
    Input: data- mxn matrix (m: number of channels, n: samples of signals)
           fs- Sampling frequency (default 128Hz)
           f_range- Frequency range (default 0.5Hz to 30Hz)
    Output: Power values and PSD values
    """
    powers = []
    psds = list()
    for sig in data:
        freq, psd = signal.welch(sig, fs)
        idx = np.logical_and(freq >= f_range[0], freq <= f_range[1])
        powers = np.append(powers, sum(psd[idx]))
        psds.append(psd[idx])

    return powers, psds


def get_psds_beta(data, fs=250, f_range=[12, 30]):
    """
    Calculate signal power using Welch method.
    Input: data- mxn matrix (m: number of channels, n: samples of signals)
           fs- Sampling frequency (default 128Hz)
           f_range- Frequency range (default 0.5Hz to 30Hz)
    Output: Power values and PSD values
    """
    powers = []
    psds = list()
    for sig in data:
        freq, psd = signal.welch(sig, fs)
        idx = np.logical_and(freq >= f_range[0], freq <= f_range[1])
        powers = np.append(powers, sum(psd[idx]))
        psds.append(psd[idx])

    return powers, psds


def plot_topomap(data, ax, fig, draw_cbar=True):
    """
    Plot topographic plot of EEG data. This specialy design for Emotiv 14 electrode data.
    This can be change for any other arrangement by changing ch_pos (channel position array)
    Input: data- 1D array 14 data values
           ax- Matplotlib subplot object to be plotted every thing
           fig- Matplot lib figure object to draw colormap
           draw_cbar- Visualize color bar in the plot
    """
    N = 300
    xy_center = [2, 2]
    radius = 2

    # 下記がその順番
    # 'Fp1','Fp2','C3','C4','O1','O2','T3','T4','F7','F8','T5','T6'
    # T3=T7,T4=T8,T5=P7,T6=P8
    # 全部で１６電極
    ch_pos = [
        [1.5, 4.2],
        [2.5, 4.2],
        [0.95, 2],
        [3.05, 2],
        [1.5, 0],
        [2.5, 0],
        [-0.1, 2],
        [4.1, 2],
        [0.1, 3],
        [3.9, 3],
        [0.4, 0.4],
        [3.6, 0.4],
    ]
    x, y = [], []
    for i in ch_pos:
        x.append(i[0])
        y.append(i[1])

    xi = np.linspace(-2, 6, N)
    yi = np.linspace(-2, 6, N)
    zi = scipy.interpolate.griddata(
        (x, y), data, (xi[None, :], yi[:, None]), method="cubic"
    )

    dr = xi[1] - xi[0]
    for i in range(N):
        for j in range(N):
            r = np.sqrt((xi[i] - xy_center[0]) ** 2 + (yi[j] - xy_center[1]) ** 2)
            if (r - dr / 2) > radius:
                zi[j, i] = "nan"

    dist = ax.contourf(
        xi,
        yi,
        zi,
        60,
        cmap=plt.get_cmap("bwr"),
        zorder=1,
        norm=Normalize(vmin=-2.5, vmax=2.5),
    )
    ax.contour(xi, yi, zi, 15, linewidths=0.5, colors="grey", zorder=2)

    if draw_cbar:
        cbar = fig.colorbar(dist, ax=ax, format="%.1f")
        cbar.ax.tick_params(labelsize=8)

    ax.scatter(x, y, marker="o", c="b", s=15, zorder=3)
    circle = patches.Circle(
        xy=xy_center, radius=radius, edgecolor="k", facecolor="none", zorder=4
    )
    ax.add_patch(circle)

    for loc, spine in ax.spines.items():
        spine.set_linewidth(0)

    ax.set_xticks([])
    ax.set_yticks([])

    circle = patches.Ellipse(
        xy=[0, 2],
        width=0.4,
        height=1.0,
        angle=0,
        edgecolor="k",
        facecolor="w",
        zorder=0,
    )
    ax.add_patch(circle)
    circle = patches.Ellipse(
        xy=[4, 2],
        width=0.4,
        height=1.0,
        angle=0,
        edgecolor="k",
        facecolor="w",
        zorder=0,
    )
    ax.add_patch(circle)

    xy = [[1.6, 3.6], [2, 4.3], [2.4, 3.6]]
    polygon = patches.Polygon(xy=xy, edgecolor="k", facecolor="w", zorder=0)
    ax.add_patch(polygon)

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)

    return ax


# if __name__ == "__main__":
#     data = mne.io.read_raw_edf("1.edf")
#     raw_data = data.get_data()
#     ch_data = raw_data[2:16, :]
#     pwrs, _ = get_psds(ch_data)

#     fig, ax = plt.subplots(figsize=(10, 8))
#     plot_topomap(pwrs, ax, fig)
#     plt.show()
#     fig.savefig("topograph.png", bbox_inches="tight")
