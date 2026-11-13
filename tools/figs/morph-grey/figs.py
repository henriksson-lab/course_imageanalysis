"""Figures for morph-grey/lec.qmd. Run from the repo root:
    python3 -I tools/figs/morph-grey/figs.py <data dir>
<data dir> holds spots.json from tools/figs/morph-reconstruct/data.ts (the top-hat demo's
synthetic image and its openings by discs of radius 2, 6, 20, computed with the demo's code).
The micrograph is demos/src/fourier2d/cell256.png (Robert M. Hunt, CC BY 3.0, Wikimedia Commons).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

OUT = "morph-grey"
DATA = sys.argv[1] if len(sys.argv) > 1 else "."
OBJ, ORANGE, GREEN, BLUE, RED, GREY = "#29334d", "#ff7f0e", "#2ca02c", "#1f77b4", "#d62728", "#9aa0a8"
plt.rcParams.update({"font.size": 18, "svg.hashsalt": "morph", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})


def disc(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def cell():
    a = plt.imread("demos/src/fourier2d/cell256.png")
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a * 255.0


def show(ax, img, title, lo=0, hi=255, fs=20):
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title(title, fontsize=fs)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig_minmax_cell():
    """Min and max filter (erosion, dilation) and opening of the micrograph, disc radius 3."""
    I = cell()
    se = disc(3)
    e = ndi.grey_erosion(I, footprint=se, mode="nearest")
    d = ndi.grey_dilation(I, footprint=se, mode="nearest")
    o = ndi.grey_dilation(e, footprint=se, mode="nearest")
    c = I[40:200, 40:200]
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 7.1))
    for ax, img, t in zip(axs.flat, [I, e, d, o], ["input", "erosion (min)", "dilation (max)", "opening"]):
        show(ax, img[40:200, 40:200], t, fs=25)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
    fig.savefig(f"{OUT}/fig-minmax-cell.png", dpi=110)
    plt.close(fig)


def flat_open(f, w):
    return ndi.grey_opening(f, size=w, mode="nearest")


def flat_close(f, w):
    return ndi.grey_closing(f, size=w, mode="nearest")


def profile():
    x = np.arange(140)
    f = 40 + 30 * np.sin(x / 140 * np.pi)            # slow background
    f = f + 55 * (np.abs(x - 25) <= 1)                # narrow peak (3 wide)
    f = f + 40 * ((x >= 52) & (x <= 74))              # wide plateau (23 wide)
    f = f - 35 * ((x >= 96) & (x <= 98))              # narrow valley (3 wide)
    f = f + 30 * (np.abs(x - 118) <= 2)               # peak 5 wide
    return x, f


def fig_landscape():
    """A profile, its opening (bar from below) and closing (bar from above), width 11."""
    x, f = profile()
    w = 11
    o, c = flat_open(f, w), flat_close(f, w)
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.fill_between(x, 0, f, step="mid", color="#e3e3e3")
    ax.step(x, f, where="mid", color="#555", lw=1.8, label="image I")
    ax.step(x, o, where="mid", color=ORANGE, lw=3, label="opening (bar from below)")
    ax.step(x, c, where="mid", color=BLUE, lw=2.5, ls="--", label="closing (bar from above)")
    # the bar at two positions: under the narrow peak (from below) and over the valley (from above)
    for xc, y in [(25, o[25]), (63, o[63])]:
        ax.add_patch(plt.Rectangle((xc - w / 2, y - 6), w, 6, color=ORANGE, alpha=0.9))
    ax.add_patch(plt.Rectangle((97 - w / 2, c[97]), w, 6, color=BLUE, alpha=0.9))
    ax.set_xlim(0, 139); ax.set_ylim(0, 150)
    ax.set_xlabel("position (pixels)")
    ax.set_ylabel("grey level")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=16, frameon=False, ncol=3, handlelength=1.5, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{OUT}/fig-landscape.svg")
    plt.close(fig)


def load_spots():
    d = json.load(open(os.path.join(DATA, "spots.json")))
    n = d["N"]
    g = lambda k: np.array(d[k], dtype=float).reshape(n, n)
    return g("img"), {2: g("open2"), 6: g("open6"), 20: g("open20")}


def overlay(I, mask):
    g = np.clip(I, 0, 255) / 255 * 0.8 + 30 / 255
    rgb = np.stack([g, g, g], -1)
    rgb[mask] = [1.0, 120 / 255, 20 / 255]
    return rgb


def fig_tophat2d():
    """Top-hat demo: thresholding the input fails, thresholding the top-hat works."""
    I, op = load_spots()
    th = I - op[6]
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 6.9))
    show(axs[0, 0], I, "input", fs=25)
    axs[0, 1].imshow(overlay(I, I > 150)); show(axs[0, 1], overlay(I, I > 150), "input > 150", fs=25)
    show(axs[1, 0], th, "top-hat, r = 6", 0, max(40, th.max()), fs=25)
    show(axs[1, 1], overlay(I, th > 28), "top-hat > 28", fs=25)
    fig.tight_layout(pad=0.3, h_pad=1.2, w_pad=0.4)
    fig.savefig(f"{OUT}/fig-tophat2d.png", dpi=110)
    plt.close(fig)


def fig_se_size():
    """White top-hat of the demo image with discs of radius 2, 6 and 20."""
    I, op = load_spots()
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.9))
    for ax, r, t in zip(axs, [2, 6, 20], ["r = 2: too small", "r = 6", "r = 20: too large"]):
        th = I - op[r]
        show(ax, th, t, 0, 90)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(f"{OUT}/fig-se-size.png", dpi=110)
    plt.close(fig)


def fig_ball():
    """Opening by a flat bar vs by a ball (a non-flat SE) on the same profile."""
    x = np.arange(140)
    f = 40 + 30 * np.sin(x / 140 * np.pi)
    f = f + np.clip(60 - 6 * np.abs(x - 28), 0, None)       # triangular peak (steep ridge)
    f = f + 40 * ((x >= 60) & (x <= 84))                     # plateau, 25 wide
    f = f + np.clip(45 - 9 * np.abs(x - 115), 0, None)      # narrower triangular peak
    R = 9
    b = np.arange(-R, R + 1)
    s = np.sqrt(R * R - b * b) * 3.0      # ball, stretched vertically (grey levels vs pixels)
    s = s - s.max()
    o_flat = flat_open(f, 2 * R + 1)
    o_ball = ndi.grey_opening(f, structure=s, mode="nearest")
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.fill_between(x, 0, f, step="mid", color="#e3e3e3")
    ax.step(x, f, where="mid", color="#555", lw=1.8, label="image I")
    ax.step(x, o_flat, where="mid", color=ORANGE, lw=3, label="opening, flat bar")
    ax.plot(x, o_ball, color=GREEN, lw=2.8, ls="--", label="opening, ball")
    for xc in (28, 115):
        print("ball vs flat at", xc, round(o_ball[xc], 1), round(o_flat[xc], 1), "image", round(f[xc], 1))
    ax.set_xlim(0, 139); ax.set_ylim(0, 150)
    ax.set_xlabel("position (pixels)")
    ax.set_ylabel("grey level")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=16, frameon=False, ncol=3, handlelength=1.5, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{OUT}/fig-ball.svg")
    plt.close(fig)


def fig_gradient():
    """Morphological gradient (3 x 3 square... disc r = 1) vs Sobel magnitude on the micrograph."""
    I = cell()
    se = disc(1)
    g = ndi.grey_dilation(I, footprint=se, mode="nearest") - ndi.grey_erosion(I, footprint=se, mode="nearest")
    sx, sy = ndi.sobel(I, 1, mode="nearest"), ndi.sobel(I, 0, mode="nearest")
    sob = np.hypot(sx, sy) / 4
    c = (slice(40, 200), slice(40, 200))
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.7))
    show(axs[0], g[c], "gradient, r = 1", 0, np.percentile(g, 99.5))
    show(axs[1], sob[c], "Sobel magnitude", 0, np.percentile(sob, 99.5))
    fig.tight_layout(pad=0.3, w_pad=0.5)
    fig.savefig(f"{OUT}/fig-gradient.png", dpi=110)
    plt.close(fig)


def fig_rank():
    """The rank-filter example of the slide: min, median, max over a window of 3."""
    v = np.array([5, 5, 9, 5, 5, 0, 5, 5, 5, 9, 0, 5], float)
    mn = ndi.minimum_filter1d(v, 3, mode="nearest")
    md = ndi.median_filter(v, 3, mode="nearest")
    mx = ndi.maximum_filter1d(v, 3, mode="nearest")
    assert (md == 5).all()
    rows = [("I", v), ("min", mn), ("median", md), ("max", mx)]
    cell, lab = 46, 120
    W, H = lab + 12 * cell + 10, 4 * (cell + 12) + 10
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="system-ui, sans-serif" role="img" aria-label="min, median and max over a window of 3 of the example signal">',
             f'<rect width="{W}" height="{H}" fill="#fff"/>']
    for k, (name, r) in enumerate(rows):
        y = 6 + k * (cell + 12)
        parts.append(f'<text x="{lab - 12}" y="{y + cell * 0.66:.1f}" text-anchor="end" font-size="22" font-weight="600">{name}</text>')
        for i, val in enumerate(r):
            g = int(255 - val / 9 * 200)
            fill = f"rgb({g},{g},{g})"
            tc = "#fff" if val >= 6 else "#222"
            parts.append(f'<rect x="{lab + i * cell}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#bbb"/>')
            parts.append(f'<text x="{lab + (i + 0.5) * cell:.1f}" y="{y + cell * 0.68:.1f}" text-anchor="middle" font-size="22" fill="{tc}">{int(val)}</text>')
        parts.append(f'<rect x="{lab}" y="{y}" width="{12 * cell}" height="{cell}" fill="none" stroke="#444" stroke-width="1.5"/>')
    parts.append("</svg>")
    open(f"{OUT}/fig-rank.svg", "w").write("\n".join(parts) + "\n")
    return mn.astype(int).tolist(), mx.astype(int).tolist()


if __name__ == "__main__":
    fig_minmax_cell()
    fig_landscape()
    fig_tophat2d()
    fig_se_size()
    fig_ball()
    fig_gradient()
    print("rank min/max:", fig_rank())
