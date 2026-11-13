"""Figures for coloc/lec.qmd from the data written by data.ts (run that first; see its header).
Usage: python3 -I tools/figs/coloc/figs.py <datadir>   (from the repo root)"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colocfig import BLUE, ORANGE, VERM, GREY, fig, save, density, merge, noaxis, hexrgb, num, np  # noqa: E402

D = json.load(open(os.path.join(sys.argv[1], "coloc.json")))
OUT = "coloc"


def inside(d, key="cells"):
    m = np.asarray(d[key]) > 0
    return np.asarray(d["a"])[m], np.asarray(d["b"])[m]


# 1. scatter plots of the three merge panels (inside the cells)
f, axs = fig(920, 320, 1, 3, sharey=True)
for ax, (k, title) in zip(axs, [("perfect", "B has A's structures"), ("independent", "independent structures"), ("bleed", "independent + 40 % leak")]):
    a, b = inside(D[k])
    density(ax, a, b)
    ax.set_title(f"{title}\nr = {num(D[k]['r'])}", fontsize=18)
    ax.set_xlabel("channel A")
axs[0].set_ylabel("channel B")
f.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.82, wspace=0.1)
save(f, f"{OUT}/fig-scatter.svg")

# 2. the region: whole image vs inside the cells
d = D["region"]
f, axs = fig(500, 330, 1, 2, sharey=True)
a, b = np.asarray(d["a"]), np.asarray(d["b"])
density(axs[0], a, b, lim=(0, 120), bins=60)
axs[0].set_title(f"whole image\nr = {d['rAll']:.2f}")
a2, b2 = inside(d)
density(axs[1], a2, b2, lim=(0, 120), bins=60)
axs[1].set_title(f"inside the cells\nr = {num(d['rCells'])}")
axs[0].annotate("background", xy=(12, 12), xytext=(50, 14), fontsize=16, color=VERM, bbox=dict(fc="white", ec="none", pad=1), arrowprops=dict(arrowstyle="->", color=VERM), va="center")
for ax in axs:
    ax.set_xlabel("channel A"); ax.set_xticks([0, 50, 100])
axs[0].set_ylabel("channel B"); axs[0].set_yticks([0, 50, 100])
f.subplots_adjust(left=0.15, right=0.98, bottom=0.2, top=0.8, wspace=0.12)
save(f, f"{OUT}/fig-region.svg")

# 3. one bright outlier: values and ranks
d = D["outlier"]
a, b = np.asarray(d["a"], float), np.asarray(d["b"], float)


def ranks(x):
    o = np.argsort(x, kind="stable"); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        r[x == v] = r[x == v].mean()
    return r


ra, rb = ranks(a), ranks(b)
f, axs = fig(500, 340, 1, 2)
out = np.zeros(36, bool); out[14] = True
for ax, (x, y, t, lim) in zip(axs, [(a, b, f"values\nPearson r = {d['r']:.2f}", 21), (ra, rb, f"ranks\nSpearman = {num(d['rho'])}", 37)]):
    # jitter equal points slightly so overlapping pairs stay visible
    rng = np.random.default_rng(1)
    jx, jy = (rng.uniform(-0.25, 0.25, 36), rng.uniform(-0.25, 0.25, 36)) if lim == 21 else (0, 0)
    ax.scatter((x + jx)[~out], (y + jy)[~out], s=36, color=BLUE)
    ax.scatter(x[out], y[out], s=90, color=VERM, marker="D")
    ax.set_xlim(-1, lim); ax.set_ylim(-1, lim); ax.set_aspect("equal")
    ax.set_title(t)
    ax.set_xlabel("A" if lim == 21 else "rank of A")
    ax.set_ylabel("B" if lim == 21 else "rank of B")
axs[0].set_xticks([0, 10, 20]); axs[0].set_yticks([0, 10, 20])
axs[1].set_xticks([1, 18, 36]); axs[1].set_yticks([1, 18, 36])
axs[0].annotate("one pixel\nat (20, 20)", xy=(20, 20), xytext=(6, 14), fontsize=16, color=VERM, arrowprops=dict(arrowstyle="->", color=VERM))
f.subplots_adjust(left=0.12, right=0.98, bottom=0.2, top=0.8, wspace=0.45)
save(f, f"{OUT}/fig-outlier.svg")

# 4. Manders, A inside B: which of A's above-threshold pixels lie where B is above too
d = D["inside"]
A, B, cells = (np.asarray(d[k], float).reshape(128, 128) for k in ("a", "b", "cells"))
tA, tB = d["tA"], d["tB"]
f, axs = fig(500, 340, 1, 2)
axs[0].imshow(merge(A, B), interpolation="nearest")
axs[0].set_title("merge (A green, B magenta)", fontsize=16)
cls = np.ones((128, 128, 3))
cls[(B > tB) & (cells > 0)] = (0.85, 0.85, 0.85)
cls[(A > tA) & (B <= tB) & (cells > 0)] = hexrgb(ORANGE)
cls[(A > tA) & (B > tB) & (cells > 0)] = hexrgb(BLUE)
axs[1].imshow(cls, interpolation="nearest")
axs[1].contour(cells, [0.5], colors=[GREY], linewidths=1)
axs[1].set_title(f"$t_A$ = {tA:.0f}, $t_B$ = {tB:.0f}", fontsize=16)
for ax in axs:
    noaxis(ax)
from matplotlib.patches import Patch  # noqa: E402
f.legend(handles=[Patch(color=BLUE, label="A and B above"), Patch(color=ORANGE, label="A above, B below"), Patch(color="#d9d9d9", label="B above only")],
         loc="lower center", ncol=2, frameon=False, fontsize=16, handlelength=1.2, columnspacing=1)
f.subplots_adjust(left=0.01, right=0.99, bottom=0.25, top=0.9, wspace=0.06)
save(f, f"{OUT}/fig-manders.svg")

# 5. Costes: the walk down the regression line and the trace
d = D["costes"]
p = D["perfect"]
a, b = inside(p)
f, axs = fig(440, 340, 1, 2)
density(axs[0], a, b, lim=(0, 200), bins=64, cmap="Greys")
xs = np.array([0, 200])
axs[0].plot(xs, d["slope"] * xs + d["intercept"], color=BLUE, lw=2.5)
tr = d["trace"]
pts = [t for k, t in enumerate(tr) if k % 12 == 0]
axs[0].scatter([t["tA"] for t in pts], [t["tB"] for t in pts], color=ORANGE, s=30, zorder=3)
axs[0].scatter([d["tA"]], [d["tB"]], color=VERM, s=80, marker="D", zorder=4)
axs[0].annotate("stop", xy=(d["tA"], d["tB"]), xytext=(60, 8), color=VERM, fontsize=16, arrowprops=dict(arrowstyle="->", color=VERM))
axs[0].set_xlabel("channel A"); axs[0].set_ylabel("channel B")
axs[0].set_title("walk down")
axs[1].plot([t["tA"] for t in tr], [t["r"] for t in tr], color=BLUE, lw=2.5)
axs[1].axhline(0, color=GREY, lw=1)
axs[1].scatter([d["tA"]], [tr[-1]["r"]], color=VERM, s=80, marker="D", zorder=4)
axs[1].set_xlim(max(t["tA"] for t in tr) + 5, 0)
axs[1].set_ylim(-0.15, 1.05)
axs[1].set_xlabel("$t_A$ (lowered →)"); axs[1].set_ylabel("r of pixels below")
axs[1].set_title(f"stop at $t_A$ = {d['tA']:.1f}")
f.subplots_adjust(left=0.18, right=0.97, bottom=0.2, top=0.86, wspace=0.8)
save(f, f"{OUT}/fig-costes.svg")
