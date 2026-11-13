"""Figures for morph-openclose/lec.qmd. Run from the repo root:
    python3 -I tools/figs/morph-openclose/figs.py <data dir>
where <data dir> holds twosizes.json written by tools/figs/morph-reconstruct/data.ts (the
granulometry demo's image and opening function, computed with the course's demo code).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "morph-binary"))
from gridfig import Svg, grid, erode, dilate, diff_colours, SQUARE, OBJ, RED, GREEN, ORANGE

OUT = "morph-openclose"
DATA = sys.argv[1] if len(sys.argv) > 1 else "."
plt.rcParams.update({"font.size": 17, "axes.titlesize": 18, "svg.hashsalt": "morph", "font.family": "DejaVu Sans"})


def disc(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def fig_rolling():
    """Opening = disc rolled inside, closing = disc rolled outside, on a smooth shape."""
    h, w = 150, 210
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.zeros((h, w), bool)
    a |= (xx > 30) & (xx < 120) & (yy > 35) & (yy < 115)               # main block
    a |= (xx >= 120) & (xx < 185) & (yy > 60) & (yy < 68)                # thin arm (8 px)
    a |= (xx - 165) ** 2 + (yy - 100) ** 2 < 24 ** 2                     # round lobe
    a &= ~((xx > 60) & (xx < 70) & (yy > 35) & (yy < 80))                # narrow bay (10 px)
    a &= ~((xx - 90) ** 2 + (yy - 95) ** 2 < 6 ** 2)                     # small hole
    a |= (xx > 22) & (xx < 31) & (yy > 70) & (yy < 76)                   # small bump
    r = 12
    se = disc(r)
    op = ndi.binary_opening(a, se)
    pad = 30
    ap = np.pad(a, pad)
    cl = ndi.binary_closing(ap, se)[pad:-pad, pad:-pad]
    cm = ListedColormap(["#ffffff", OBJ, RED, GREEN])
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, res, name, inside in [(axs[0], op, "opening: disc rolled inside", True),
                                  (axs[1], cl, "closing: disc rolled outside", False)]:
        lab = np.zeros((h, w), int)
        lab[a & res] = 1
        lab[a & ~res] = 2
        lab[~a & res] = 3
        ax.imshow(lab, cmap=cm, vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(name, fontsize=23)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#888")
        centres = [(55, 75), (105, 100), (166, 99)] if inside else [(65, 22), (140, 40), (100, 130)]
        for cx, cy in centres:
            ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec=ORANGE, lw=2.5))
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{OUT}/fig-rolling.svg")
    plt.close(fig)
    return int(a.sum()), int(op.sum()), int(cl.sum())


def fig_noisy():
    """The demo's noisy mask: open then close vs close then open (3 x 3 square, border ignored)."""
    a = grid(["00000000000000", "00000000000000", "00111111100000", "00111011100100", "00111111100000",
              "00111111111000", "00101111011000", "00111111111000", "00000011111000", "00100011011000",
              "00000011111000", "00000000000000", "00000000000000", "00000000000000"])
    op = lambda m: dilate(erode(m, SQUARE, 1), SQUARE, 0)
    cl = lambda m: erode(dilate(m, SQUARE, 0), SQUARE, 1)
    oc, co = cl(op(a)), op(cl(a))
    assert (a.sum(), oc.sum(), co.sum(), (oc != co).sum()) == (61, 36, 67, 31)
    cell = 20
    pw = 14 * cell
    gap = 40
    W, H = 3 * pw + 2 * gap + 40, pw + 70
    s = Svg(W, H, "a noisy mask; open then close gives 36 pixels, close then open gives 67")
    for k, (nm, v, c) in enumerate([("noisy mask: 61", a, None), ("open, then close: 36", oc, diff_colours(a, oc)),
                                    ("close, then open: 67", co, diff_colours(a, co))]):
        x = 20 + k * (pw + gap)
        s.text(x + pw / 2, 34, nm, size=22, weight="600")
        s.grid(x, 50, v, cell, colours=c)
    s.save(f"{OUT}/fig-noisy.svg")


def fig_granulo():
    """The granulometry demo's image coloured by the opening function, and its pattern spectrum."""
    d = json.load(open(os.path.join(DATA, "twosizes.json")))
    n = d["N"]
    of = np.array(d["of"]).reshape(n, n)
    spec = np.array(d["spectrum"])
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 4.4), gridspec_kw={"width_ratios": [1, 1.2]})
    cmap = plt.get_cmap("viridis").copy()
    img = np.ma.masked_less(of.astype(float), 0)
    cmap.set_bad("white")
    im = axs[0].imshow(img, cmap=cmap, vmin=0, vmax=14, interpolation="nearest")
    axs[0].set_xticks([]); axs[0].set_yticks([])
    axs[0].set_title("size of each pixel", fontsize=20)
    cb = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.03, ticks=[0, 5, 12])
    cb.ax.tick_params(labelsize=19)
    rr = np.arange(len(spec))
    axs[1].bar(rr[:16], spec[:16], color=[plt.get_cmap("viridis")(min(r, 14) / 14) for r in rr[:16]], edgecolor="#333", lw=0.5)
    axs[1].set_xlabel("disc radius r", fontsize=20)
    axs[1].set_ylabel("pixels removed", fontsize=20)
    axs[1].set_title("pattern spectrum", fontsize=20)
    axs[1].set_xticks([0, 5, 12])
    axs[1].set_yticks([0, 2000, 4000])
    axs[1].tick_params(labelsize=19)
    axs[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{OUT}/fig-granulo.svg")
    plt.close(fig)
    return list(spec[:14])


def blob():
    h, w = 120, 170
    yy, xx = np.mgrid[0:h, 0:w]
    a = (xx - 60) ** 2 / 48 ** 2 + (yy - 62) ** 2 / 40 ** 2 < 1
    a |= (xx - 122) ** 2 + (yy - 55) ** 2 < 30 ** 2
    a |= (xx > 40) & (xx < 150) & (yy > 92) & (yy < 104)
    return a


def fig_dt_levels():
    """Euclidean DT of a blob; its thresholds DT > r are the erosions by discs of radius r."""
    a = blob()
    dt = ndi.distance_transform_edt(a)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    cmap = plt.get_cmap("Blues").copy()
    im = ax.imshow(np.ma.masked_equal(dt, 0), cmap=cmap, vmin=0, vmax=dt.max() * 1.05, interpolation="nearest")
    ax.contour(a.astype(float), levels=[0.5], colors=[OBJ], linewidths=2)
    cs = ax.contour(dt, levels=[6, 14, 22], colors=[ORANGE], linewidths=2.2)
    ax.clabel(cs, fmt=lambda v: f"r = {v:.0f}", fontsize=16, inline=True)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, orientation="horizontal")
    cb.set_label("distance to the background (pixels)", fontsize=16)
    cb.ax.tick_params(labelsize=15)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{OUT}/fig-dt-levels.svg")
    plt.close(fig)


def fig_dt_touching():
    """Two touching round cells: the DT has one peak per cell and a ridge between them."""
    h, w = 90, 150
    yy, xx = np.mgrid[0:h, 0:w]
    a = ((xx - 52) ** 2 + (yy - 45) ** 2 < 34 ** 2) | ((xx - 104) ** 2 + (yy - 47) ** 2 < 28 ** 2)
    dt = ndi.distance_transform_edt(a)
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.0), gridspec_kw={"width_ratios": [1, 1.1]})
    axs[0].imshow(np.ma.masked_equal(dt, 0), cmap="Blues", vmin=0, vmax=dt.max() * 1.05, interpolation="nearest")
    axs[0].contour(a.astype(float), levels=[0.5], colors=[OBJ], linewidths=2)
    axs[0].axhline(46, color=ORANGE, lw=2, ls="--")
    axs[0].set_xticks([]); axs[0].set_yticks([])
    for s in axs[0].spines.values():
        s.set_visible(False)
    axs[0].set_title("DT of two touching cells", fontsize=17)
    axs[1].plot(dt[46], color=OBJ, lw=2.5)
    axs[1].set_xlabel("position along the dashed line")
    axs[1].set_ylabel("DT")
    axs[1].spines[["top", "right"]].set_visible(False)
    axs[1].set_title("two peaks, one per cell", fontsize=17)
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{OUT}/fig-dt-touching.svg")
    plt.close(fig)


if __name__ == "__main__":
    print("rolling (A, open, close):", fig_rolling())
    fig_noisy()
    print("spectrum:", fig_granulo())
    fig_dt_levels()
    fig_dt_touching()
