"""Static figures for phasecorr/ (phase correlation).

Run from the repository root:  python3 -I tools/figs/phasecorr/figs.py
Writes phasecorr/fig-*.svg|png. The correlation surfaces come from the phasecorr demo's own
model (tools/figs/phasecorr/data.ts, run with node), so peaks and heights match the demo.
"""
import json, os, subprocess, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image

plt.rcParams.update({"font.size": 16, "axes.titlesize": 16, "svg.fonttype": "none",
                     "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, RED, GREEN, GREY, PURPLE = "#1f77b4", "#ff7f0e", "#d62728", "#2a9d4b", "#888888", "#cc79a7"
CELL = np.asarray(Image.open("demos/src/fourier2d/cell256.png")).astype(float)
OUT = "phasecorr/"


def demo_data():
    with tempfile.TemporaryDirectory() as d:
        cell, js = os.path.join(d, "cell.f64"), os.path.join(d, "pc.json")
        CELL.astype("<f8").tofile(cell)
        subprocess.run("node_modules/.bin/esbuild tools/figs/phasecorr/data.ts --bundle --platform=node "
                       "--log-level=warning | node - %s %s" % (cell, js), shell=True, check=True)
        return json.load(open(js))


def centred(surface, n):
    """Circular surface (index = shift mod n) -> array with shift (0, 0) in the middle, extent."""
    s = np.array(surface).reshape(n, n)
    return np.fft.fftshift(s), [-n // 2 - 0.5, n // 2 - 0.5, n // 2 - 0.5, -n // 2 - 0.5]


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def surf_panel(ax, d, title, vmax=None, zoom=None, cross=True):
    n = d["n"]; s, ext = centred(d["surface"], n)
    v = vmax if vmax is not None else np.abs(s).max()
    ax.imshow(s, cmap="RdBu_r", vmin=-v, vmax=v, extent=ext)
    tx, ty = d["trueShift"]
    ax.add_patch(Circle((tx, ty), 6, fill=False, ec=GREEN, lw=2.5))
    if cross:
        ax.plot([d["tx"]], [d["ty"]], "+", color="k", ms=18, mew=2.5)
    ax.set_title(title)
    ax.set_xticks([-50, 0, 50]); ax.set_yticks([-50, 0, 50])
    ax.tick_params(labelsize=14)
    if zoom:
        ax.set_xlim(-zoom, zoom); ax.set_ylim(zoom, -zoom)
        ax.set_xticks([-(zoom // 2), 0, zoom // 2]); ax.set_yticks([-(zoom // 2), 0, zoom // 2])


# ------------------------------------------------------------ CC: broad, moved by a ramp
def cc_trouble(D):
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.7), gridspec_kw={"wspace": 0.38})
    surf_panel(ax[0], D["cc"], "CC")
    surf_panel(ax[1], D["ramp_cc"], "CC, ramp on B")
    for a in ax:
        a.set_xlabel("shift $t_x$")
    ax[0].set_ylabel("shift $t_y$")
    fig.savefig(OUT + "fig-cc-trouble.png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    n = D["cc"]["n"]; s = np.array(D["cc"]["surface"]).reshape(n, n)
    print("cc (20,10)/peak:", s[10, 20] / s[13, 21])


# ------------------------------------------------------------ unit arrows
def arrows():
    N, t0 = 16, 3
    x = np.arange(N)
    A = 2 * np.exp(-(x - 4) ** 2 / 3) + 1.2 * np.exp(-(x - 10) ** 2 / 1.5) + 0.7 * np.exp(-(x - 13) ** 2 / 4)
    B = np.roll(A, -t0)
    P = np.fft.fft(A) * np.conj(np.fft.fft(B))
    R = P / np.abs(P)
    ks = range(1, 7)
    fig, axs = plt.subplots(2, 6, figsize=(9.6, 3.9), gridspec_kw={"hspace": 0.45, "wspace": 0.1})
    for row, t in enumerate((1, t0)):
        for i, k in enumerate(ks):
            a = axs[row, i]
            z = R[k] * np.exp(2j * np.pi * k * t / N)
            a.add_patch(Circle((0, 0), 1, fill=False, ec=GREY, lw=1.2))
            col = GREEN if z.real > 0.9 else ORANGE
            a.annotate("", xy=(z.real, z.imag), xytext=(0, 0),
                       arrowprops=dict(arrowstyle="-|>", color=col, lw=3, mutation_scale=18))
            a.set_xlim(-1.25, 1.25); a.set_ylim(-1.25, 1.25); a.set_aspect("equal"); off(a)
            if row == 0:
                a.set_title("k = %d" % k, fontsize=15)
        r = np.real(np.sum(R * np.exp(2j * np.pi * np.arange(N) * t / N))) / N
        axs[row, 0].text(-1.6, 0, "t = %d" % t, ha="right", va="center", fontsize=16,
                         color=GREEN if t == t0 else "k")
        axs[row, 5].text(1.5, 0, "sum: %.2f" % r, ha="left", va="center", fontsize=16,
                         color=GREEN if t == t0 else "k")
    fig.savefig(OUT + "fig-unit-arrows.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------ whitening, brightness
def whiten(D):
    d = D["ramp_phase"]; n = d["n"]
    a = np.array(D["phase"]["a"]).reshape(n, n); b = np.array(d["b"]).reshape(n, n)
    w = np.outer(np.hanning(n), np.hanning(n))   # numpy hanning = 0.5(1 - cos(2πk/(n-1)))
    Fa = np.fft.fft2((a - a.mean()) * w)
    wa = np.real(np.fft.ifft2(Fa / (np.abs(Fa) + 1e-9)))
    fig, ax = plt.subplots(1, 4, figsize=(13.2, 3.9), gridspec_kw={"wspace": 0.12})
    ax[0].imshow(a, cmap="gray", vmin=0, vmax=255); ax[0].set_title("crop A")
    ax[1].imshow(wa, cmap="gray", vmin=-np.percentile(np.abs(wa), 99.5), vmax=np.percentile(np.abs(wa), 99.5))
    ax[1].set_title("A, phases only")
    ax[2].imshow(b, cmap="gray"); ax[2].set_title("crop B + ramp 500")
    for i in range(3):
        off(ax[i])
    surf_panel(ax[3], d, "phase corr.: %.2f" % d["height"], vmax=0.3, zoom=40, cross=False)
    ax[3].set_xlabel("shift $t_x$", fontsize=14)
    fig.savefig(OUT + "fig-whiten.png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------ signal vs noise per frequency
def noise_spectrum(D):
    n = D["phase"]["n"]
    a = np.array(D["phase"]["a"]).reshape(n, n)
    w = np.outer(np.hanning(n), np.hanning(n))
    F = np.abs(np.fft.fft2((a - a.mean()) * w))
    sigma = 120
    # expected |noise| per frequency: sigma * sqrt(sum w^2) (times sqrt(pi)/2 for the mean of a Rayleigh)
    noise = sigma * np.sqrt((w ** 2).sum()) * np.sqrt(np.pi) / 2
    fy = np.fft.fftfreq(n)[:, None]; fx = np.fft.fftfreq(n)[None, :]
    r = np.hypot(fx, fy)
    edges = np.linspace(0, 0.5, 26); mid = 0.5 * (edges[1:] + edges[:-1])
    med = np.array([np.median(F[(r >= lo) & (r < hi)]) for lo, hi in zip(edges[:-1], edges[1:])])
    count = np.array([((r >= lo) & (r < hi)).sum() for lo, hi in zip(edges[:-1], edges[1:])])
    fig, ax = plt.subplots(figsize=(5.3, 4.2))
    ax.semilogy(mid, med, "o-", color=BLUE, lw=2.4, label="signal (crop A)")
    ax.axhline(noise, color=ORANGE, lw=2.4, ls="--", label="noise, σ = 120")
    cross = mid[np.argmax(med < noise)]
    ax.axvspan(cross, 0.5, color=ORANGE, alpha=0.12)
    ax.text(0.5 * (cross + 0.5), noise * 0.75, "noise wins:\n%d %% of\nall frequencies" %
            round(100 * (r >= cross).sum() / r.size), ha="center", va="top", fontsize=15, color="#a0522d")
    ax.set_xlabel("frequency (cycles per pixel)"); ax.set_ylabel("typical size |F|")
    ax.legend(frameon=False, fontsize=14, loc="lower left")
    ax.set_xlim(0, 0.5)
    fig.savefig(OUT + "fig-noise-spectrum.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("noise crossover at", cross, "cycles/px")


# ------------------------------------------------------------ taper
def taper():
    n, ax0, ay0, grad = 128, 40, 60, 600
    Y, X = np.mgrid[0:256, 0:256]
    img = CELL + grad * (X + Y) / 510
    a = img[ay0:ay0 + n, ax0:ax0 + n]
    a0 = a - a.mean()
    w = np.outer(np.hanning(n), np.hanning(n))
    S1 = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(a0))))
    S2 = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(a0 * w))))
    tiled = np.tile(a, (2, 2))
    fig, ax = plt.subplots(1, 4, figsize=(10.4, 3.1), gridspec_kw={"wspace": 0.12})
    ax[0].imshow(tiled, cmap="gray"); ax[0].set_title("crop, repeated")
    ax[0].axhline(n - 0.5, color=ORANGE, lw=1.5, ls=":"); ax[0].axvline(n - 0.5, color=ORANGE, lw=1.5, ls=":")
    lo, hi = np.percentile(S1, [5, 99.8])
    ax[1].imshow(S1, cmap="magma", vmin=lo, vmax=hi); ax[1].set_title("its spectrum")
    ax[2].imshow(a0 * w, cmap="gray"); ax[2].set_title("Hann taper")
    ax[3].imshow(S2, cmap="magma", vmin=np.percentile(S2, 5), vmax=np.percentile(S2, 99.8)); ax[3].set_title("its spectrum")
    for x in ax:
        off(x)
    fig.savefig(OUT + "fig-taper.png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------ periodic texture
def periodic(D):
    d = D["periodic"]; n = d["n"]
    a = np.array(d["a"]).reshape(n, n)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.5), gridspec_kw={"wspace": 0.3})
    ax[0].imshow(a, cmap="gray"); ax[0].set_title("crop A: a lattice"); off(ax[0])
    surf_panel(ax[1], d, "phase correlation", vmax=d["height"], zoom=32)
    for (tx, ty, v), (dx, dy, ha) in zip(d["peaks"][:3], ((3, -4, "left"), (-3, 7, "right"), (-3, -4, "right"))):
        ax[1].text(tx + dx, ty + dy, "%.2f" % v, fontsize=16, ha=ha)
    ax[1].text(d["trueShift"][0], d["trueShift"][1] - 8, "truth %.2f" % d["atTrue"], fontsize=15, color=GREEN,
               ha="center", va="bottom")
    ax[1].set_xlabel("shift $t_x$")
    fig.savefig(OUT + "fig-periodic.png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("periodic peaks", d["peaks"][:3], "truth", d["atTrue"], "ratio", d["ratio"])


if __name__ == "__main__":
    D = demo_data()
    cc_trouble(D); arrows(); whiten(D); noise_spectrum(D); taper(); periodic(D)
