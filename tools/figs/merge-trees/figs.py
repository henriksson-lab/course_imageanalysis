"""Figures for merge-trees/. Run from the repository root after compute.ts (see there):

    python3 -I tools/figs/merge-trees/figs.py <rawdir>

The micrograph is demos/src/fourier2d/cell256.png (Robert M. Hunt, CC BY 3.0, via Wikimedia
Commons); everything else is synthetic.
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 17, "font.family": "DejaVu Sans", "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
raw = sys.argv[1]
OUT = "merge-trees/"
D = json.load(open(f"{raw}/mt.json"))


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def meanimg(I, lab):
    I = np.asarray(I, float); lab = np.asarray(lab)
    out = np.zeros_like(I)
    for l in np.unique(lab):
        m = lab == l
        out[m] = I[m].mean()
    return out


def bnd(lab, n=256):
    l = np.asarray(lab).reshape(n, n)
    b = np.zeros_like(l, bool)
    b[:, :-1] |= l[:, :-1] != l[:, 1:]
    b[:-1, :] |= l[:-1, :] != l[1:, :]
    return b


def show_part(ax, I, lab, title):
    m = meanimg(I, lab).reshape(256, 256)
    rgb = np.stack([m, m, m], -1) / 255
    rgb[bnd(lab)] = (1, 0.88, 0)
    ax.imshow(rgb); ax.set_title(title, fontsize=17); clean(ax)


# ---- fig-scales.png: one hierarchy, three cuts ----
Nn = D["nuclei"]
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.8))
for ax, (k, title) in zip(axs, [("60", "60 regions: over-segmented"), ("19", "19: nuclei, touching pairs"), ("8", "8: only the deepest")]):
    show_part(ax, Nn["I"], Nn["cuts"][k], title)
fig.tight_layout(); fig.savefig(OUT + "fig-scales.png", dpi=85); plt.close(fig)

# ---- fig-rag.png: the region adjacency graph of the basins ----
fig, ax = plt.subplots(figsize=(5.6, 5.6))
I = np.asarray(Nn["I"], float).reshape(256, 256)
rgb = np.stack([I, I, I], -1) / 255
rgb[bnd(Nn["leaves"])] = (1, 0.88, 0)
ax.imshow(rgb)
cx, cy = Nn["cx"], Nn["cy"]
for a, b in zip(Nn["a"], Nn["b"]):
    ax.plot([cx[a], cx[b]], [cy[a], cy[b]], color="#d62728", lw=1)
ax.scatter(cx, cy, s=14, c="#1f77b4", zorder=3)
ax.set_xlim(0, 255); ax.set_ylim(255, 0); clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-rag.png", dpi=100); plt.close(fig)

# ---- fig-ucm.png: UCM of the micrograph and two thresholds ----
C = D["cell"]
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.8))
u = np.asarray(C["ucm"], float).reshape(256, 256) / C["merges"]
axs[0].imshow(u ** 1.5, cmap="inferno"); axs[0].set_title("UCM (bright = lasting)", fontsize=17); clean(axs[0])
show_part(axs[1], C["I"], C["c40"], "threshold: 60 regions")
show_part(axs[2], C["I"], C["c10"], "threshold: 20 regions")
fig.tight_layout(); fig.savefig(OUT + "fig-ucm.png", dpi=85); plt.close(fig)

# ---- fig-leakmean.png ----
L = D["leak"]
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.8))
axs[0].imshow(np.asarray(L["I"]).reshape(256, 256), cmap="gray", vmin=0, vmax=255); axs[0].set_title("bright disc with one weak spot", fontsize=16); clean(axs[0])
show_part(axs[1], L["I"], L["alt"], "altitude (cheapest pass), 2 regions")
show_part(axs[2], L["I"], L["mean"], "mean difference, 2 regions")
fig.tight_layout(); fig.savefig(OUT + "fig-leakmean.png", dpi=85); plt.close(fig)

# ---- fig-basins.svg: the five-basin landscape ----
# basins a..e: minima 1, 3, 0, 4, 2; passes a|b 5, b|c 4, c|d 7, d|e 6; widths 10, 2, 8, 3, 6
names = "abcde"
minima = [1, 3, 0, 4, 2]
passes = [5, 4, 7, 6]
widths = [10, 2, 8, 3, 6]
xs, ys = [0], [6.5]
x = 0
for i in range(5):
    w = widths[i]
    left = ys[-1]
    right = passes[i] if i < 4 else 6.5
    xs += [x + w * 0.3, x + w * 0.7, x + w]
    ys += [minima[i], minima[i], right]
    x += w
fig, ax = plt.subplots(figsize=(9.4, 3.9))
ax.fill_between(xs, ys, 8, color="#c9b38a", alpha=0.6)
ax.plot(xs, ys, color="#5a4a2a", lw=2.5)
x = 0
for i in range(5):
    w = widths[i]
    ax.text(x + w / 2, minima[i] - 0.7, f"{names[i]}\nmin {minima[i]}", ha="center", va="top", fontsize=14)
    if i < 4:
        ax.text(x + w, passes[i] + 0.3, f"{passes[i]}", ha="center", fontsize=15, color="#d62728")
    ax.text(x + w / 2, 7.5, f"width {w}", ha="center", fontsize=13, color="#555")
    x += w
ax.set_ylim(-2.3, 8.2); ax.set_xlim(-0.5, x + 0.5); clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-basins.svg"); plt.close(fig)

# ---- fig-trees.svg: three merge trees of the five basins ----
def dendro(ax, merges, order, title, ylabel):
    pos = {n: i for i, n in enumerate(order)}
    height = {n: 0 for n in order}
    groups = {n: n for n in order}
    for k, (a, b, lv) in enumerate(merges):
        ga, gb = groups[a], groups[b]
        xa, xb = pos[ga], pos[gb]
        ax.plot([xa, xa, xb, xb], [height[ga], lv, lv, height[gb]], color="#333", lw=2.5)
        ax.text((xa + xb) / 2, lv + 0.15, f"{lv:g}", ha="center", fontsize=14, color="#d62728")
        new = ga + gb
        pos[new] = (xa + xb) / 2; height[new] = lv
        for n in new:
            groups[n] = new
    ax.set_xticks(range(5)); ax.set_xticklabels(list(order), fontsize=16)
    ax.set_title(title, fontsize=16, loc="left"); ax.set_ylabel(ylabel, fontsize=12)
    ax.spines["left"].set_visible(True)


fig, axs = plt.subplots(3, 1, figsize=(5.6, 6.6))
dendro(axs[0], [("b", "c", 4), ("a", "b", 5), ("d", "e", 6), ("c", "d", 7)], "abcde", "altitude", "pass height")
dendro(axs[1], [("b", "c", 1), ("d", "e", 2), ("a", "b", 4), ("c", "d", 5)], "abcde", "dynamics", "depth of the shallower")
dendro(axs[2], [("b", "c", 2), ("d", "e", 3), ("c", "d", 9), ("a", "b", 10)], "abcde", "area", "area of the smaller")
fig.tight_layout(); fig.savefig(OUT + "fig-trees.svg"); plt.close(fig)

# ---- fig-component.svg: the component tree of a small 1D signal ----
sig = [1, 4, 2, 5, 6, 3, 1, 3, 2, 0]
fig, axs = plt.subplots(2, 1, figsize=(6.4, 6.2), gridspec_kw={"height_ratios": [1, 1.5]})
axs[0].bar(range(10), sig, color="#bbb")
for t, c in [(1, "#1f77b4"), (3, "#2ca02c"), (5, "#d62728")]:
    axs[0].axhline(t - 0.5, color=c, lw=1.5, ls="--")
    axs[0].text(9.6, t - 0.4, f"≥ {t}", color=c, fontsize=15)
axs[0].set_xticks(range(10)); axs[0].set_xticklabels([str(i + 1) for i in range(10)], fontsize=14)
axs[0].set_yticks([])
ax = axs[1]
nodes = {"1–9 (≥1)": (1.5, 0), "2–6 (≥2)": (0.8, 0.9), "8–9 (≥2)": (2.7, 0.9), "2 (≥3)": (0.0, 1.9), "4–6 (≥3)": (1.4, 1.9), "8 (≥3)": (2.7, 1.9),
         "4–5 (≥4)": (1.4, 2.8), "5 (≥6)": (1.4, 3.7)}
links = [("1–9 (≥1)", "2–6 (≥2)"), ("1–9 (≥1)", "8–9 (≥2)"), ("2–6 (≥2)", "2 (≥3)"), ("2–6 (≥2)", "4–6 (≥3)"), ("8–9 (≥2)", "8 (≥3)"),
         ("4–6 (≥3)", "4–5 (≥4)"), ("4–5 (≥4)", "5 (≥6)")]
for a, b in links:
    ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color="#555", lw=2)
for k, (x, y) in nodes.items():
    ax.text(x, y, k, ha="center", va="center", fontsize=15, bbox=dict(boxstyle="round,pad=0.25", fc="#e8f0fa", ec="#1f77b4"))
ax.set_xlim(-0.6, 3.4); ax.set_ylim(-0.4, 4.1); clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-component.svg"); plt.close(fig)
print("done")
