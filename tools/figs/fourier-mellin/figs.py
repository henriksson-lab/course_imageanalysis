"""Static figures for fourier-mellin/ (rotation and scale become shifts: Fourier–Mellin).

Run from the repository root:  python3 -I tools/figs/fourier-mellin/figs.py
Writes fourier-mellin/fig-*.svg|png. Every image and number comes from the fmellin demo's own
model (tools/figs/fourier-mellin/data.ts, run with node), so the figures match the demo.
"""
import json, os, subprocess, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams.update({"font.size": 16, "axes.titlesize": 16, "svg.fonttype": "none",
                     "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, RED, GREEN, GREY, PURPLE = "#1f77b4", "#ff7f0e", "#d62728", "#2a9d4b", "#888888", "#cc79a7"
OUT = "fourier-mellin/"
N = 128


def demo_data():
    with tempfile.TemporaryDirectory() as d:
        js = os.path.join(d, "fm.json")
        subprocess.run("node_modules/.bin/esbuild tools/figs/fourier-mellin/data.ts --bundle --platform=node "
                       "--log-level=warning | node - %s" % js, shell=True, check=True)
        return json.load(open(js))


def arr(v, w, h=None):
    return np.array(v, dtype=float).reshape(h or w, w)


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def grey(ax, im, title, lo=None, hi=None):
    ax.imshow(im, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title(title); off(ax)


def spec(ax, im, title, aspect="equal"):
    ax.imshow(im, cmap="inferno", interpolation="nearest", aspect=aspect)
    ax.set_title(title); off(ax)


def centred(s, w, h):
    return np.fft.fftshift(arr(s, w, h))


def overlay(a, b):
    na = (a - a.min()) / (np.ptp(a) or 1); nb = (b - b.min()) / (np.ptp(b) or 1)
    return np.dstack([na, nb, na])


def save(fig, name, **kw):
    fig.savefig(OUT + name, bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)


# ------------------------------------------------------------ the problem
def problem(D):
    d = D["problem"]
    a, b = arr(d["a"], N), arr(d["b"], N)
    fig, ax = plt.subplots(1, 4, figsize=(14, 4.3), gridspec_kw={"wspace": 0.3})
    grey(ax[0], a, "view A", 0, 255)
    grey(ax[1], b, "view B:\nturned 30°, shifted", 0, 255)
    for k, (key, pk, ratio, title) in enumerate([("surface0", d["peak0"], d["ratio0"], "B only shifted"), ("surface", d["peak"], d["ratio"], "B turned 30° too")]):
        s = centred(d[key], N, N)
        ax[2 + k].imshow(s, cmap="inferno", vmin=0, vmax=0.25, extent=[-64.5, 63.5, 63.5, -64.5])
        ax[2 + k].plot([pk[0]], [pk[1]], "o", ms=16, mfc="none", mec="#00e5ff", mew=2)
        ax[2 + k].set_title("phase correlation,\n%s: peak %.2f" % (title, pk[2]), fontsize=15)
        ax[2 + k].set_xlim(-40, 40); ax[2 + k].set_ylim(40, -40)
        ax[2 + k].set_xticks([-30, 0, 30]); ax[2 + k].set_yticks([-30, 0, 30]); ax[2 + k].tick_params(labelsize=13)
    save(fig, "fig-problem.png", dpi=100)


# ------------------------------------------------------------ polar unwrap about a known centre
def polar(D):
    d = D["known"]; n = d["size"]; nT, nR = d["nTheta"], d["nR"]
    a, b = arr(d["a"], n), arr(d["b"], n)
    pa, pb = arr(d["pa"], nT, nR), arr(d["pb"], nT, nR)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    fig = plt.figure(figsize=(12.5, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 2.6], wspace=0.12, hspace=0.45)
    for k, (im, t) in enumerate([(a, "disc A"), (b, "disc B: A turned 50°")]):
        ax = fig.add_subplot(gs[:, k])
        grey(ax, im, t, lo, hi)
        c = n / 2
        ax.add_patch(Circle((c - 0.5, c - 0.5), 48, fill=False, ec="#ffd400", lw=2))
        for th in np.radians([0, 90]):
            ax.plot([c - 0.5, c - 0.5 + 48 * np.cos(th)], [c - 0.5, c - 0.5 + 48 * np.sin(th)], color=ORANGE, lw=2)
    for k, (im, t) in enumerate([(pa, "A unwrapped: angle →, radius ↓"), (pb, "B unwrapped: the same, shifted by 50°")]):
        ax = fig.add_subplot(gs[k, 2])
        ax.imshow(im, cmap="gray", vmin=lo, vmax=hi, aspect="auto", extent=[0, 360, nR, 0], interpolation="nearest")
        ax.set_title(t, fontsize=15); ax.set_yticks([])
        ax.set_xticks([0, 90, 180, 270, 360]); ax.tick_params(labelsize=13)
        if k == 1:
            ax.annotate("", xy=(50, nR * 0.5), xytext=(0, nR * 0.5), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=3, mutation_scale=20))
    save(fig, "fig-polar.png", dpi=100)


def rotcorr(D):
    d = D["known"]; nT = d["nTheta"]
    s = np.array(d["score"]); ang = np.arange(nT) * 360 / nT
    ang = np.where(ang > 180, ang - 360, ang); o = np.argsort(ang)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(ang[o], s[o], color=BLUE, lw=2.4)
    ax.axvline(d["trueAngle"], color=GREEN, ls="--", lw=2)
    ax.annotate("peak at %.0f°" % d["angle"], xy=(d["angle"], 1), xytext=(95, 0.8), fontsize=15,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_xlabel("angle of B relative to A (°)")
    ax.set_ylabel("score (peak = 1)")
    save(fig, "fig-rotcorr.svg")


# ------------------------------------------------------------ magnitude spectra
def magnitude(D):
    d = D["magnitude"]
    fig, ax = plt.subplots(1, 4, figsize=(13.2, 3.7), gridspec_kw={"wspace": 0.08})
    grey(ax[0], arr(d["a"], N), "view A", 0, 255)
    grey(ax[1], arr(d["b"], N), "B: turned 30°, shifted", 0, 255)
    ma, mb = arr(d["ma"], N), arr(d["mb"], N)
    hi = max(ma.max(), mb.max())
    ax[2].imshow(ma, cmap="inferno", vmin=0, vmax=hi); ax[2].set_title("log |F| of A"); off(ax[2])
    ax[3].imshow(mb, cmap="inferno", vmin=0, vmax=hi); ax[3].set_title("log |F| of B"); off(ax[3])
    # a guide line along the same direction, turned by 30° in B
    for k, th in [(2, 0), (3, 30)]:
        t = np.radians(th)
        ax[k].plot([64 - 60 * np.cos(t), 64 + 60 * np.cos(t)], [64 - 60 * np.sin(t), 64 + 60 * np.sin(t)], color="#00e5ff", lw=1.5, ls="--")
    save(fig, "fig-magnitude.png", dpi=100)


# ------------------------------------------------------------ log-polar: rotation and scale as shifts (schematic)
def logpolar_schematic():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"wspace": 0.35, "width_ratios": [1, 1.25]})
    a = ax[0]
    rs = 2 * (32 / 2) ** (np.arange(9) / 8)
    for r in rs:
        a.add_patch(Circle((0, 0), r, fill=False, ec=GREY, lw=0.8))
    for th in np.radians(np.arange(0, 360, 22.5)):
        a.plot([0, 32 * np.cos(th)], [0, 32 * np.sin(th)], color=GREY, lw=0.8)
    # a feature: a small blob of points at r ≈ 14, θ ≈ 20°; its copy scaled by 1.5 and turned by 40°
    def blob(r0, t0):
        t = np.radians(np.linspace(-10, 10, 30)); rr = r0 * np.exp(np.linspace(-0.15, 0.15, 30))
        T, Rr = np.meshgrid(t + np.radians(t0), rr)
        return Rr, T
    for (r0, t0, col, lab) in [(14, 20, BLUE, "A"), (21, 60, RED, "B")]:
        Rr, T = blob(r0, t0)
        a.fill(np.r_[Rr[0] * np.cos(T[0]), Rr[-1, ::-1] * np.cos(T[-1, ::-1])],
               np.r_[Rr[0] * np.sin(T[0]), Rr[-1, ::-1] * np.sin(T[-1, ::-1])], color=col, alpha=0.85)
        a.text(r0 * 1.45 * np.cos(np.radians(t0)), r0 * 1.45 * np.sin(np.radians(t0)), lab, color=col, fontsize=17, ha="center", va="center", weight="bold")
    a.set_xlim(-33, 33); a.set_ylim(33, -33); a.set_aspect("equal"); off(a)
    a.set_title("spectrum: B = A turned 40°,\nradius × 1.5", fontsize=14)
    b = ax[1]
    for (r0, t0, col, lab) in [(14, 20, BLUE, "A"), (21, 60, RED, "B")]:
        b.add_patch(plt.Rectangle((t0 - 10, np.log(r0) - 0.15), 20, 0.30, color=col, alpha=0.85))
        b.text(t0, np.log(r0) + 0.3, lab, color=col, fontsize=17, ha="center", weight="bold")
    b.annotate("", xy=(52, np.log(21)), xytext=(28, np.log(14)), arrowprops=dict(arrowstyle="-|>", color="k", lw=2, mutation_scale=18))
    b.text(72, np.log(15), "+40°, + ln 1.5", ha="left", fontsize=15)
    b.set_xlim(0, 180); b.set_ylim(np.log(60), np.log(2))
    b.set_xticks([0, 45, 90, 135, 180]); b.set_xlabel("angle θ (°)")
    b.set_yticks(np.log([2, 5, 10, 20, 50])); b.set_yticklabels(["2", "5", "10", "20", "50"]); b.set_ylabel("radius (log scale)")
    b.set_title("log-polar grid: a plain shift", fontsize=14)
    b.grid(color=GREY, lw=0.5, alpha=0.5)
    save(fig, "fig-logpolar.svg")


# ------------------------------------------------------------ the pipeline
def pipeline(D):
    d = D["pipeline"]; nT, nR = d["nTheta"], d["nR"]
    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 2, 1.2], hspace=0.35, wspace=0.18)
    grey(fig.add_subplot(gs[0, 0]), arr(d["a"], N), "A", 0, 255)
    grey(fig.add_subplot(gs[1, 0]), arr(d["b"], N), "B", 0, 255)
    ma, mb = arr(d["ma"], N), arr(d["mb"], N)
    spec(fig.add_subplot(gs[0, 1]), ma, "|F| of A"); spec(fig.add_subplot(gs[1, 1]), mb, "|F| of B")
    la, lb = arr(d["lpA"], nT, nR), arr(d["lpB"], nT, nR)
    hi = max(la.max(), lb.max())
    for k, (im, t) in enumerate([(la, "log-polar of |F_A|"), (lb, "log-polar of |F_B|")]):
        ax = fig.add_subplot(gs[k, 2])
        ax.imshow(im, cmap="inferno", vmin=0, vmax=hi, aspect="auto", extent=[0, 180, 62, 2])
        ax.set_title(t); ax.set_xticks([0, 45, 90, 135, 180]); ax.set_yticks([]); ax.tick_params(labelsize=13)
    ax = fig.add_subplot(gs[0, 3])
    s = centred(d["surface"], nT, nR)
    ax.imshow(s, cmap="inferno", aspect="auto", extent=[-nT / 2 - 0.5, nT / 2 - 0.5, nR / 2 - 0.5, -nR / 2 - 0.5])
    ax.plot([d["peak"][0]], [d["peak"][1]], "+", color="#00e5ff", ms=18, mew=2.5)
    ax.set_title("peak: %.1f°, ×%.2f" % (d["found"][0], d["found"][1]), fontsize=15)
    ax.set_xticks([-128, 0, 128]); ax.set_xticklabels(["−90°", "0", "90°"]); ax.set_yticks([]); ax.tick_params(labelsize=13)
    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(overlay(arr(d["a"], N), arr(d["aligned"], N))); off(ax)
    ax.set_title("A and B aligned", fontsize=15)
    save(fig, "fig-pipeline.png", dpi=100)


# ------------------------------------------------------------ the 180° ambiguity
def ambiguity(D):
    d = D["ambiguity"]
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 4.0), gridspec_kw={"wspace": 0.12})
    grey(ax[0], arr(d["b"], N), "B: A turned 200°", 0, 255)
    for k, c in enumerate(d["cands"]):
        ax[1 + k].imshow(overlay(arr(d["a"], N), arr(c["unwarped"], N)))
        off(ax[1 + k])
        ax[1 + k].set_title("undo %.0f°: shift peak %.2f" % (c["angle"] if c["angle"] > 0 else c["angle"] + 360, c["height"]), fontsize=15)
    save(fig, "fig-ambiguity.png", dpi=100)


# ------------------------------------------------------------ taper: the cross
def cross(D):
    d = D["cross"]
    fig, ax = plt.subplots(2, 3, figsize=(12.6, 6.4), gridspec_kw={"wspace": 0.15, "hspace": 0.35, "width_ratios": [1, 1, 2]})
    for row, (key, lab) in enumerate([("off", "no taper"), ("on", "Hann taper")]):
        e = d[key]
        grey(ax[row, 0], arr(e["a"], N), "A, uneven light" if row == 0 else "A, tapered")
        if row == 1:
            w = np.outer(np.hanning(N), np.hanning(N)); a = arr(e["a"], N)
            ax[row, 0].images[0].set_data((a - a.mean()) * w); ax[row, 0].images[0].autoscale()
        spec(ax[row, 1], arr(e["ma"], N), "|F|, " + lab)
        s = centred(e["surface"], 256, 128)
        ax[row, 2].imshow(s, cmap="inferno", aspect="auto", extent=[-128.5, 127.5, 63.5, -64.5])
        ax[row, 2].plot([e["peak"][0]], [e["peak"][1]], "+", color="#00e5ff", ms=18, mew=2.5)
        ax[row, 2].add_patch(Circle((d["trueBins"][0], d["trueBins"][1]), 7, fill=False, ec="#2ca02c", lw=2.5))
        ax[row, 2].set_title("(θ, log r) surface: peak at %.0f°" % abs(round(e["peak"][0] * 180 / 256)), fontsize=15)
        ax[row, 2].set_xticks([-128, -64, 0, 64, 128]); ax[row, 2].set_xticklabels(["−90°", "−45°", "0", "45°", "90°"])
        ax[row, 2].set_yticks([]); ax[row, 2].tick_params(labelsize=13)
    save(fig, "fig-cross.png", dpi=100)


# ------------------------------------------------------------ where the log-polar samples fall
def samples(D):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.25})
    nT, nR, r0, r1 = 32, 16, 2, 62
    rs = r0 * (r1 / r0) ** (np.arange(nR) / (nR - 1)); th = np.pi * np.arange(nT) / nT
    for k, lim in enumerate([64, 8]):
        a = ax[k]
        for v in np.arange(-lim, lim + 1, 4 if k == 0 else 1):
            a.axhline(v, color="#dddddd", lw=0.6, zorder=0); a.axvline(v, color="#dddddd", lw=0.6, zorder=0)
        R, T = np.meshgrid(rs, th)
        a.scatter((R * np.cos(T)).ravel(), (R * np.sin(T)).ravel(), s=10 if k == 0 else 24, color=BLUE, zorder=2)
        a.set_xlim(-lim, lim); a.set_ylim(lim, -2 if k == 1 else -4); a.set_aspect("equal")
        a.set_title("half a turn, radii 2 to 62" if k == 0 else "zoom on the centre (pixel grid)", fontsize=15)
        a.tick_params(labelsize=12)
    save(fig, "fig-samples.svg")


# ------------------------------------------------------------ averaging rotated noisy copies
def average(D):
    d = D["average"]; n = d["size"]
    fig, ax = plt.subplots(1, 9, figsize=(15, 2.5), gridspec_kw={"wspace": 0.06})
    ref = arr(d["ref"], n); lo, hi = np.percentile(ref, 1), np.percentile(ref, 99)
    for k in range(6):
        grey(ax[k], arr(d["copies"][k], n), "copy %d" % (k + 1), lo - 120, hi + 120)
    grey(ax[6], arr(d["plain"], n), "average", lo, hi)
    grey(ax[7], arr(d["aligned"], n), "aligned, average", lo, hi)
    grey(ax[8], ref, "truth", lo, hi)
    for k in (6, 7, 8):
        ax[k].title.set_fontsize(14)
    save(fig, "fig-average.png", dpi=100)


if __name__ == "__main__":
    D = demo_data()
    problem(D); polar(D); rotcorr(D); magnitude(D); logpolar_schematic(); pipeline(D)
    ambiguity(D); cross(D); samples(D); average(D)
    print("wrote", OUT)
