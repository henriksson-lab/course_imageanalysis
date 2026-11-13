"""Figures for morph-reconstruct/lec.qmd. Run from the repo root:
    python3 -I tools/figs/morph-reconstruct/figs.py <data dir>
<data dir> holds blobs.json written by data.ts in this folder (the reconstruction demo's mask
and its opening, opening by reconstruction, clear border and fill holes, computed with the
demo's own code). The 1D figures use the 12-sample signal of the lecture.
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
from gridfig import Svg, grid, OBJ, RED, GREEN, ORANGE

OUT = "morph-reconstruct"
DATA = sys.argv[1] if len(sys.argv) > 1 else "."
GREYBAR, BLUE, DRED = "#9aa0a8", "#4d8fd6", "#d62728"
plt.rcParams.update({"font.size": 18, "font.family": "DejaVu Sans"})
SIG = np.array([7, 1, 4, 3, 4, 6, 5, 1, 6, 4, 2, 3], float)


# ---------------------------------------------------------------- 1D reconstruction helpers
def rec_dilate(marker, mask):
    m = np.minimum(marker, mask).astype(float)
    while True:
        d = np.minimum(ndi.maximum_filter1d(m, 3, mode="nearest"), mask)
        if (d == m).all():
            return m
        m = d


def rec_erode(marker, mask):
    m = np.maximum(marker, mask).astype(float)
    while True:
        e = np.maximum(ndi.minimum_filter1d(m, 3, mode="nearest"), mask)
        if (e == m).all():
            return m
        m = e


def regional(f, minima):
    """Regional minima (or maxima) of a 1D signal: plateaus whose neighbours are all higher (lower)."""
    out = np.zeros(len(f), bool)
    i = 0
    while i < len(f):
        j = i
        while j + 1 < len(f) and f[j + 1] == f[i]:
            j += 1
        nb = [f[k] for k in (i - 1, j + 1) if 0 <= k < len(f)]
        if all((v > f[i]) if minima else (v < f[i]) for v in nb):
            out[i:j + 1] = True
        i = j + 1
    return out


# ---------------------------------------------------------------- bar-chart SVG (style of fig-hmax.svg)
class Bars:
    def __init__(self, label, n=12, vmax=8, extra_w=0, top=40):
        self.x0, self.step, self.bw = 60, 54, 46
        self.base, self.unit = top + vmax * 30, 30
        self.W = self.x0 + n * self.step + 10 + extra_w
        self.H = self.base + 34
        self.s = Svg(self.W, self.H, label)
        for v in range(0, vmax + 1, 2):
            y = self.base - v * self.unit
            self.s.add(f'<line x1="{self.x0}" x2="{self.x0 + n * self.step}" y1="{y}" y2="{y}" stroke="#e3e3e3"/>')
            self.s.text(self.x0 - 8, y + 6, str(v), size=18, anchor="end")
        self.n = n

    def xl(self, i):
        return self.x0 + 4 + i * self.step

    def y(self, v):
        return self.base - v * self.unit

    def bars(self, f, colour=GREYBAR, opacity=1.0, labels=True):
        for i, v in enumerate(f):
            self.s.add(f'<rect x="{self.xl(i)}" y="{self.y(v)}" width="{self.bw}" height="{v * self.unit}" fill="{colour}" fill-opacity="{opacity}"/>')
            if labels:
                self.s.text(self.xl(i) + self.bw / 2, self.base + 24, str(int(v)), size=18)

    def band(self, lo, hi, colour, opacity=0.85):
        for i in range(self.n):
            if hi[i] > lo[i]:
                self.s.add(f'<rect x="{self.xl(i)}" y="{self.y(hi[i])}" width="{self.bw}" height="{(hi[i] - lo[i]) * self.unit}" fill="{colour}" fill-opacity="{opacity}"/>')

    def level(self, f, colour, width=4, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        for i, v in enumerate(f):
            self.s.add(f'<line x1="{self.xl(i) - 4}" x2="{self.xl(i) + self.bw + 4}" y1="{self.y(v)}" y2="{self.y(v)}" stroke="{colour}" stroke-width="{width}"{d}/>')

    def dots(self, mask, f, colour, dy=-10):
        for i in np.nonzero(mask)[0]:
            self.s.add(f'<circle cx="{self.xl(i) + self.bw / 2}" cy="{self.y(f[i]) + dy}" r="7" fill="{colour}"/>')

    def axis(self):
        self.s.add(f'<line x1="{self.x0}" x2="{self.x0 + self.n * self.step}" y1="{self.base}" y2="{self.base}" stroke="#444" stroke-width="1.5"/>')

    def legend(self, items, x=None, y=60):
        x = self.x0 + self.n * self.step + 24 if x is None else x
        for k, (kind, colour, text) in enumerate(items):
            yy = y + k * 34
            if kind == "box":
                self.s.add(f'<rect x="{x}" y="{yy - 15}" width="22" height="18" fill="{colour}"/>')
            elif kind == "line":
                self.s.add(f'<line x1="{x - 2}" x2="{x + 24}" y1="{yy - 6}" y2="{yy - 6}" stroke="{colour}" stroke-width="4"/>')
            else:
                self.s.add(f'<circle cx="{x + 11}" cy="{yy - 6}" r="7" fill="{colour}"/>')
            self.s.text(x + 32, yy, text, size=19, anchor="start")


def fig_grey_recon():
    """One seed of height 5 at sample 5, reconstructed under the 12-sample signal."""
    M = np.zeros(12)
    M[5] = 5
    R = rec_dilate(M, SIG)
    b = Bars("a seed of height 5 floods under the 12-sample signal and is held down by the passes", extra_w=230)
    b.bars(SIG)
    b.band(np.zeros(12), R, ORANGE, 0.9)
    b.s.add(f'<circle cx="{b.xl(5) + b.bw / 2}" cy="{b.y(5)}" r="8" fill="{DRED}" stroke="#fff" stroke-width="2"/>')
    b.axis()
    b.legend([("box", GREYBAR, "image I"), ("dot", DRED, "seed, height 5"), ("box", ORANGE, "reconstruction")])
    b.s.save(f"{OUT}/fig-grey-recon.svg")
    return R.astype(int).tolist()


def fig_hmin():
    """h-minima transform of the 12-sample signal, h = 2: water poured into the basins."""
    H = rec_erode(SIG + 2, SIG)
    m0, m1 = regional(SIG, True), regional(H, True)
    b = Bars("the 12-sample signal and its h-minima transform for h = 2")
    b.bars(SIG)
    b.band(SIG, H, BLUE, 0.75)
    b.dots(m1, H, DRED, dy=12)
    b.axis()
    b.s.save(f"{OUT}/fig-hmin.svg")
    assert (ndi.label(m0)[1], ndi.label(m1)[1]) == (4, 3)
    return ndi.label(m0)[1], ndi.label(m1)[1], H.astype(int).tolist()


def fig_impose():
    """Minima imposition on the 12-sample signal with markers at samples 1 and 10."""
    mk = np.zeros(12, bool)
    mk[[1, 10]] = True
    M = np.where(mk, 0, SIG.max() + 2)
    G = np.minimum(SIG + 1, M)
    J = rec_erode(M, G)
    m = regional(J, True)
    assert (m == mk).all()
    b = Bars("minima imposition: with markers at samples 1 and 10 the result has minima only there", vmax=8, extra_w=230)
    b.bars(SIG)
    b.level(J, ORANGE, width=5)
    b.dots(mk, np.zeros(12), DRED, dy=-12)
    b.axis()
    b.legend([("box", GREYBAR, "image I"), ("dot", DRED, "markers"), ("line", ORANGE, "imposed image")])
    b.s.save(f"{OUT}/fig-impose.svg")
    return J.astype(int).tolist(), ndi.label(regional(SIG, True))[1]


def fig_geodesic():
    """Geodesic dilation steps from one marker pixel inside a mask (8-connected)."""
    a = grid(["0000000000000",
              "0111111100110",
              "0100000100110",
              "0101110100000",
              "0101010100000",
              "0101000100110",
              "0101111100110",
              "0000000000000"])
    h, w = a.shape
    steps = -np.ones_like(a, dtype=int)
    seed = (4, 5)
    steps[seed] = 0
    frontier = [seed]
    k = 0
    while frontier:
        k += 1
        nxt = []
        for (y, x) in frontier:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and a[yy, xx] and steps[yy, xx] < 0:
                        steps[yy, xx] = k
                        nxt.append((yy, xx))
        frontier = nxt
    cmap = plt.get_cmap("YlOrBr")
    kmax = steps.max()
    cols = [["" for _ in range(w)] for _ in range(h)]
    nums = [["" for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if a[y, x] and steps[y, x] >= 0:
                r, g, bb, _ = cmap(0.15 + 0.7 * steps[y, x] / kmax)
                cols[y][x] = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(bb * 255))
                nums[y][x] = str(steps[y, x])
            elif a[y, x]:
                cols[y][x] = "#c9d0df"
    cell = 40
    s = Svg(w * cell + 40, h * cell + 30, "geodesic dilation from one marker pixel: each mask pixel numbered by the step at which the marker reaches it; the separate component is never reached")
    s.grid(20, 15, a, cell, colours=cols, numbers=nums, num_size=19)
    s.outline(20, 15, [seed], cell, colour=DRED, width=4)
    s.save(f"{OUT}/fig-geodesic.svg")
    return int(kmax)


def blobs():
    d = json.load(open(os.path.join(DATA, "blobs.json")))
    n = d["N"]
    g = lambda k: np.array(d[k], dtype=np.uint8).reshape(n, n).astype(bool)
    return d, g


def diffimg(a, res):
    lab = np.zeros(a.shape, int)
    lab[a & res] = 1
    lab[a & ~res] = 2
    lab[~a & res] = 3
    return lab


CM = ListedColormap(["#ffffff", OBJ, RED, GREEN])


def panel(ax, lab, title, marker=None):
    ax.imshow(lab, cmap=CM, vmin=0, vmax=3, interpolation="nearest")
    if marker is not None:
        rgba = np.zeros(marker.shape + (4,))
        rgba[marker] = [1.0, 0.5, 0.05, 1.0]
        ax.imshow(rgba, interpolation="nearest")
    ax.set_title(title, fontsize=21)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#888")


def fig_open_vs_rec():
    d, g = blobs()
    a, op, rec, ero = g("mask"), g("open"), g("openrec"), g("ero")
    assert d["rmOpen"] == 203 and d["rmRec"] == 60
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.6))
    panel(axs[0], diffimg(a, op), "opening, r = 4")
    panel(axs[1], diffimg(a, rec), "opening by reconstruction", marker=ero)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(f"{OUT}/fig-open-vs-rec.png", dpi=120)
    plt.close(fig)


def fig_tools():
    d, g = blobs()
    a, cl, fi = g("mask"), g("clear"), g("fill")
    assert d["rmClear"] == 4165 and d["added"] == 556
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.6))
    panel(axs[0], diffimg(a, cl), "clear border")
    panel(axs[1], diffimg(a, fi), "fill holes")
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(f"{OUT}/fig-tools.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    print("grey recon:", fig_grey_recon())
    print("hmin (minima before, after, H):", fig_hmin())
    print("impose (J, minima of I):", fig_impose())
    print("geodesic max step:", fig_geodesic())
    fig_open_vs_rec()
    fig_tools()
