"""Extra slide figures for dtree/lec.qmd.

Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/dtree/data.ts
     FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/pixelclass/data.ts   (for fig-leak.png)
     python3 -I tools/figs/dtree/figs.py <dir>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "dtree")
TMP = sys.argv[1]
D = json.load(open(os.path.join(TMP, "dtree.json")))
C0, C1 = "#1f77b4", "#ff7f0e"
plt.rcParams.update({"font.size": 16, "font.family": "DejaVu Sans", "svg.fonttype": "none", "axes.linewidth": 1.0})


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), facecolor="white", **kw)
    plt.close(fig)
    print("wrote", name)


def points(ax, P, s=26):
    P = np.array(P)
    for c, col, m in [(0, C0, "o"), (1, C1, "^")]:
        q = P[P[:, 2] == c]
        ax.scatter(q[:, 0], q[:, 1], s=s, color=col, marker=m, edgecolors="white", linewidths=0.5, zorder=3)


def square(ax, xlabel="feature 1", ylabel="feature 2"):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)


def leaf_rects(ax, R, alpha=0.32):
    for x0, x1, y0, y1, p, n in R:
        col = np.array(to_rgb(C1)) * p + np.array(to_rgb(C0)) * (1 - p)
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color=col, alpha=alpha, lw=0))
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#444", lw=0.8))


# ---------------------------------------------------------------- nearest neighbour colours the plane
P = np.array(D["blobs"])
g = np.linspace(0, 1, 300)
X, Y = np.meshgrid(g, g)
d2 = (X[..., None] - P[:, 0]) ** 2 + (Y[..., None] - P[:, 1]) ** 2
lab = P[np.argmin(d2, axis=2), 2]
fig, ax = plt.subplots(figsize=(4.6, 4.6))
rgb = np.where(lab[..., None] == 1, to_rgb(C1), to_rgb(C0))
ax.imshow(0.3 * rgb + 0.7, origin="lower", extent=(0, 1, 0, 1))
points(ax, P)
new = (0.47, 0.73)
ax.scatter(*new, s=260, marker="*", color="black", zorder=4)
ax.annotate("new point", new, xytext=(0.06, 0.9), fontsize=16, arrowprops=dict(arrowstyle="->", lw=1.2))
square(ax, "feature 1 (e.g. intensity)", "feature 2 (e.g. texture)")
fig.tight_layout()
save(fig, "fig-nn.svg")
i = np.argmin((P[:, 0] - new[0]) ** 2 + (P[:, 1] - new[1]) ** 2)
print("new point's nearest neighbour has class", int(P[i, 2]))

# ---------------------------------------------------------------- XOR
fig, (a0, a1) = plt.subplots(1, 2, figsize=(6.6, 3.5), gridspec_kw={"width_ratios": [1, 1.2]})
leaf_rects(a0, D["xorRects"])
points(a0, D["xor"], s=22)
square(a0, "x", "y")
a0.set_title("after three cuts", fontsize=16)
for sc, st, lab in [(D["xorScan"][0], "-", "cut along x"), (D["xorScan"][1], "--", "cut along y")]:
    a1.plot(sc["t"], sc["g"], st, color="#333" if st == "-" else "#888", lw=2, label=lab)
a1.set_ylim(0, 0.5)
a1.set_xlim(0, 1)
a1.set_xlabel("threshold")
a1.set_ylabel("Gini decrease")
a1.set_title("first cut", fontsize=16)
a1.axhline(0.5, color="#bbb", lw=1, ls=":")
a1.text(0.02, 0.47, "largest possible", fontsize=15, color="#666", va="top")
best = max(max(s["g"]) for s in D["xorScan"])
a1.text(0.02, 0.12, f"best: {best:.3f}", fontsize=15)
a1.legend(frameon=False, loc="center right", fontsize=15, handlelength=1.4)
a1.set_xticks([0, 0.5, 1])
a1.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "fig-xor.svg")

# ---------------------------------------------------------------- impurity curves
p = np.linspace(0, 1, 401)
with np.errstate(divide="ignore", invalid="ignore"):
    H = -(np.where(p > 0, p * np.log2(p), 0) + np.where(p < 1, (1 - p) * np.log2(1 - p), 0))
fig, ax = plt.subplots(figsize=(4.6, 3.8))
ax.plot(p, 2 * p * (1 - p), color="#0072B2", lw=2.5, label="Gini  2p(1−p)")
ax.plot(p, H / 2, color="#D55E00", lw=2.5, ls="--", label="entropy / 2")
ax.plot(p, np.minimum(p, 1 - p), color="#555", lw=2, ls=":", label="misclassification")
ax.set_xlabel("share p of one class in the node")
ax.set_ylabel("impurity")
ax.set_xlim(0, 1); ax.set_ylim(0, 0.8); ax.set_yticks([0, 0.25, 0.5])
ax.legend(frameon=False, fontsize=15, loc="upper left", bbox_to_anchor=(0.0, 1.04), handlelength=1.6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "fig-impurity.svg")

# ---------------------------------------------------------------- error against depth
tr, te = np.array(D["depthTrain"]), np.array(D["depthTest"])
dd = np.arange(1, 13)
fig = plt.figure(figsize=(6.0, 6.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
a = fig.add_subplot(gs[0, :])
a.plot(dd, 100 * tr, "-o", color="#0072B2", lw=2, ms=5, label="training")
a.plot(dd, 100 * te, "--s", color="#D55E00", lw=2, ms=5, label="test")
a.set_xlabel("maximum depth"); a.set_ylabel("error (%)")
a.set_ylim(0, 32); a.set_xticks([1, 3, 6, 9, 12])
a.legend(frameon=False, fontsize=15, loc="lower left")
a.spines[["top", "right"]].set_visible(False)
a.annotate(f"best test: {100 * te[2]:.1f} %", (3, 100 * te[2]), xytext=(4.3, 29), fontsize=14,
           arrowprops=dict(arrowstyle="->", lw=1))
for k, (key, t) in enumerate([("ovRects3", "depth 3"), ("ovRects12", "grown fully")]):
    ax = fig.add_subplot(gs[1, k])
    leaf_rects(ax, D[key], alpha=0.4)
    points(ax, D["overlap"], s=10)
    ax.set_xlim(0.05, 0.95); ax.set_ylim(0.05, 0.95); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(t, fontsize=16)
fig.tight_layout()
save(fig, "fig-depth.svg")

# ---------------------------------------------------------------- staircase on a diagonal boundary
fig, ax = plt.subplots(figsize=(4.4, 4.4))
leaf_rects(ax, D["diagRects"])
points(ax, D["diag"], s=18)
ax.plot([0, 1], [0, 1], color="black", lw=2, ls="--")
square(ax, "x", "y")
ax.text(0.97, 0.05, f"{D['diagLeaves']} leaves", ha="right", fontsize=16,
        bbox=dict(fc="white", ec="none", alpha=0.85))
fig.tight_layout()
save(fig, "fig-staircase.svg")

# ---------------------------------------------------------------- split by pixel vs by region
PC = json.load(open(os.path.join(TMP, "pixelclass.json")))
S = PC["size"]
img = np.array(PC["image"]).reshape(S, S) / 255.0
labels = np.array(PC["heldout"]["labels"]).reshape(S, S)
TR, VA = np.array(to_rgb("#5e3c99")), np.array(to_rgb("#e66101"))
rng = np.random.default_rng(3)
fig, (a0, a1) = plt.subplots(1, 2, figsize=(6.0, 3.6))
for ax, mode in [(a0, "pixel"), (a1, "region")]:
    rgb = np.dstack([img] * 3) * 0.85
    m = labels > 0
    if mode == "pixel":
        coin = rng.random((S, S)) < 0.5
    else:
        coin = np.tile(np.arange(S) < int(round(PC["holdout"] * S)), (S, 1))
    rgb[m & coin] = TR
    rgb[m & ~coin] = VA
    ax.imshow(rgb, interpolation="nearest")
    if mode == "region":
        ax.axvline(PC["holdout"] * S - 0.5, color="white", lw=2, ls="--")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title("split by pixel" if mode == "pixel" else "split by region", fontsize=16)
# zoom inset on one stroke in the pixel split
ins = a0.inset_axes([0.52, 0.0, 0.48, 0.48])
ys, xs = np.nonzero(labels == 2)
cy, cx = int(np.median(ys[xs < S * 0.6])), int(np.median(xs[xs < S * 0.6]))
r = 14
ins.imshow(a0.images[0].get_array()[cy - r:cy + r, cx - r:cx + r], interpolation="nearest")
ins.set_xticks([]); ins.set_yticks([])
for s_ in ins.spines.values():
    s_.set_edgecolor("white"); s_.set_linewidth(2)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=TR, label="training pixels"), Patch(color=VA, label="validation pixels")],
           loc="lower center", ncol=2, frameon=False, fontsize=16, handlelength=1.2, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.11, 1, 1))
fig.savefig(os.path.join(OUT, "fig-leak.png"), dpi=150, facecolor="white")
plt.close(fig)
print("wrote fig-leak.png")
