"""Figures for random-walker/. Run from the repository root (see compute.ts for the order):

    python3 -I tools/figs/random-walker/figs.py <rawdir> --export-cell   # writes <rawdir>/cell256.json
    (run compute.ts with <rawdir>)
    python3 -I tools/figs/random-walker/figs.py <rawdir>                 # writes the figures

The micrograph is demos/src/fourier2d/cell256.png (Robert M. Hunt, CC BY 3.0, via Wikimedia
Commons); everything else is synthetic.
"""
import json
import sys
from fractions import Fraction

import numpy as np

raw = sys.argv[1]
if "--export-cell" in sys.argv:
    from PIL import Image
    im = np.asarray(Image.open("demos/src/fourier2d/cell256.png").convert("RGB"), dtype=float)
    grey = 0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2]
    json.dump(grey.ravel().tolist(), open(f"{raw}/cell256.json", "w"))
    sys.exit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({"font.size": 17, "font.family": "DejaVu Sans", "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
OUT = "random-walker/"
D = json.load(open(f"{raw}/rw.json"))
OBJ, BG, L3 = "#ff7f0e", "#1f77b4", "#2ca02c"
DIV = LinearSegmentedColormap.from_list("div", ["#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b"])


def img(a, n=128):
    return np.asarray(a, dtype=float).reshape(n, n)


def seeds_rgba(s, n=128):
    s = np.asarray(s).reshape(n, n)
    out = np.zeros((n, n, 4))
    for v, c in [(1, OBJ), (2, BG), (3, L3)]:
        rgb = matplotlib.colors.to_rgb(c)
        out[s == v] = (*rgb, 1)
    return out


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ---- fig-graph.svg: the pixel graph with contrast weights and seeds (recall) ----
fig, ax = plt.subplots(figsize=(5.2, 5.2))
I = np.array([[60, 70, 190, 200], [65, 180, 195, 185], [60, 75, 190, 70], [55, 60, 65, 70]], float)
sig = 30
for r in range(4):
    for c in range(4):
        for dr, dc in [(0, 1), (1, 0)]:
            r2, c2 = r + dr, c + dc
            if r2 < 4 and c2 < 4:
                w = np.exp(-(I[r, c] - I[r2, c2]) ** 2 / (2 * sig ** 2))
                ax.plot([c, c2], [r, r2], color="#333", lw=0.6 + 9 * w, solid_capstyle="round", zorder=1)
for r in range(4):
    for c in range(4):
        g = I[r, c] / 255
        ec, lw = "#555", 1.5
        if (r, c) == (1, 2):
            ec, lw = OBJ, 5
        if (r, c) == (3, 0):
            ec, lw = BG, 5
        ax.add_patch(plt.Circle((c, r), 0.28, facecolor=(g, g, g), edgecolor=ec, lw=lw, zorder=2))
ax.text(2, 1, "1", ha="center", va="center", color="k", fontsize=18, zorder=3, fontweight="bold")
ax.text(0, 3, "0", ha="center", va="center", color="w", fontsize=18, zorder=3, fontweight="bold")
ax.set_xlim(-0.5, 3.5); ax.set_ylim(3.6, -0.5); ax.set_aspect("equal"); clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-graph.svg"); plt.close(fig)

# ---- fig-walks.png: 20 random walks from one pixel ----
W = D["walks"]
n = W["w"]
fig, ax = plt.subplots(figsize=(5.4, 5.4))
ax.imshow(img(W["I"], n), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
ax.imshow(seeds_rgba(W["seeds"], n), interpolation="nearest")
seeds = np.asarray(W["seeds"])
rng_ = np.random.default_rng(3)
for path in W["paths"]:
    p = np.asarray(path)
    jitter = rng_.uniform(-0.18, 0.18, 2)
    ax.plot(p[:400] % n + jitter[0], p[:400] // n + jitter[1], color="#ffd400", lw=0.8, alpha=0.6)
    e = p[-1]
    ax.plot(e % n, e // n, "o", ms=9, mfc=OBJ if seeds[e] == 1 else BG, mec="k")
s = W["start"]
ax.plot(s % n, s // n, "s", ms=11, mfc="none", mec="#d62728", mew=3)
clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-walks.png", dpi=110); plt.close(fig)

# ---- fig-chain.svg: the 1 x 5 chain as a resistor ladder with its voltages ----
def chain_fig(weights, v, fname, seeds, cut=None, labels=None, height=4.2):
    nn = len(v)
    fig, ax = plt.subplots(figsize=(1.25 * nn + 1.2, height))
    for i, w in enumerate(weights):
        ax.plot([i, i + 1], [1.35, 1.35], color="#333", lw=1.5 + 9 * w / max(weights), solid_capstyle="butt", zorder=1)
        ax.text(i + 0.5, 1.55, f"w = {Fraction(w).limit_denominator(100)}", ha="center", fontsize=15)
        ax.text(i + 0.5, 1.12, f"R = {Fraction(1 / w).limit_denominator(100)}", ha="center", fontsize=15, color="#666")
    for i in range(nn):
        fc = OBJ if seeds.get(i) == 1 else BG if seeds.get(i) == 0 else "white"
        ax.add_patch(plt.Circle((i, 1.35), 0.17, facecolor=fc, edgecolor="#333", lw=1.5, zorder=2))
        ax.text(i, 1.35, str(i + 1), ha="center", va="center", fontsize=15, color="w" if i in seeds else "k", zorder=3)
        col = OBJ if seeds.get(i) == 1 else BG if seeds.get(i) == 0 else ("#fdae6b" if v[i] > 0.5 else "#9ecae1")
        ax.bar(i, v[i] * 0.85, width=0.4, color=col, bottom=0)
        lab = labels[i] if labels else f"{v[i]:.2f}"
        ax.text(i, v[i] * 0.85 + 0.03, lab, ha="center", fontsize=15)
    ax.axhline(0.5 * 0.85, color="#888", ls="--", lw=1)
    ax.text(-0.45, 0.5 * 0.85 + 0.02, "0.5", color="#888", fontsize=14)
    for i in range(nn - 1):
        a, b = v[i] - 0.5, v[i + 1] - 0.5
        if a * b < 0:
            xc = i + (0.5 - v[i]) / (v[i + 1] - v[i])
            ax.plot([xc, xc], [-0.02, 0.9], color="#ff7f0e", lw=4)
            ax.text(xc, -0.12, "walker", color="#e6550d", ha="center", fontsize=14)
    if cut is not None:
        ax.plot([cut + 0.5, cut + 0.5], [1.15, 1.75], color="#d62728", lw=3, ls=(0, (4, 3)))
        ax.text(cut + 0.5, 1.82, "min cut", color="#d62728", ha="center", fontsize=14)
    ax.set_xlim(-0.6, nn - 0.4); ax.set_ylim(-0.2, 1.95); clean(ax)
    fig.tight_layout(); fig.savefig(OUT + fname); plt.close(fig)


chain_fig([1, 1, 0.25, 1], [1, 6 / 7, 5 / 7, 1 / 7, 0], "fig-chain.svg", {0: 1, 4: 0}, labels=["1", "6/7", "5/7", "1/7", "0"])
chain_fig([0.2, 1, 1, 1, 1, 0.25], [1, 8 / 13, 7 / 13, 6 / 13, 5 / 13, 4 / 13, 0], "fig-seed-c.svg", {0: 1, 6: 0}, cut=0,
          labels=["1", "8/13", "7/13", "6/13", "5/13", "4/13", "0"])
chain_fig([0.2, 1, 1, 1, 1, 0.25], [1, 1, 1, 6 / 7, 5 / 7, 4 / 7, 0], "fig-seed-d.svg", {2: 1, 6: 0}, cut=5,
          labels=["1", "1", "1", "6/7", "5/7", "4/7", "0"])

# ---- fig-cell.png: micrograph, two strokes, probability map with the 0.5 contour ----
C = D["cell"]
fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.9))
axs[0].imshow(img(C["I"]), cmap="gray", vmin=0, vmax=255)
axs[0].imshow(seeds_rgba(C["seeds"]))
axs[0].contour(img(C["p"]), [0.5], colors=["#ffe000"], linewidths=2.5)
axs[0].set_title("seeds and the 0.5 contour", fontsize=17)
im_ = axs[1].imshow(img(C["p"]), cmap=DIV, vmin=0, vmax=1)
axs[1].set_title("p(object)", fontsize=17)
for a in axs:
    clean(a)
cb = fig.colorbar(im_, ax=axs[1], fraction=0.046, pad=0.03, ticks=[0, 0.5, 1])
fig.tight_layout(); fig.savefig(OUT + "fig-cell.png", dpi=100); plt.close(fig)

# ---- fig-three.png: three labels as colours ----
T = D["three"]
P = [img(p) for p in T["p"]]
rgb = np.zeros((128, 128, 3))
for p, c in zip(P, [OBJ, BG, L3]):
    rgb += p[..., None] * np.array(matplotlib.colors.to_rgb(c))
fig, axs = plt.subplots(1, 2, figsize=(9.0, 4.7))
axs[0].imshow(img(T["I"]), cmap="gray", vmin=0, vmax=255); axs[0].imshow(seeds_rgba(T["seeds"]))
axs[0].set_title("image and seeds", fontsize=17)
axs[1].imshow(np.clip(rgb, 0, 1)); axs[1].set_title("p as colours", fontsize=17)
for a in axs:
    clean(a)
fig.tight_layout(); fig.savefig(OUT + "fig-three.png", dpi=100); plt.close(fig)

# ---- fig-fade.png: far from seeds, the probabilities fade towards an even split ----
F = D["fade"]
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
axs[0].imshow(img(F["p"]), cmap=DIV, vmin=0, vmax=1); axs[0].imshow(seeds_rgba(F["seeds"]))
axs[0].plot([0, 127], [0, 127], color="k", lw=1, ls=":")
axs[0].set_title("p(object), no edges", fontsize=17); clean(axs[0])
diag = [img(F["p"])[i, i] for i in range(128)]
axs[1].plot(range(128), diag, color="#333", lw=2.5)
axs[1].axhline(0.5, color="#888", ls="--", lw=1)
axs[1].set_xlabel("position along the diagonal"); axs[1].set_ylabel("p(object)"); axs[1].set_ylim(0, 1)
fig.tight_layout(); fig.savefig(OUT + "fig-fade.png", dpi=100); plt.close(fig)

# ---- fig-priors.png: no edge between two brightnesses: walker vs walker with priors ----
Pr = D["priors"]
fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.6))
axs[0].imshow(img(Pr["I"]), cmap="gray", vmin=0, vmax=255); axs[0].imshow(seeds_rgba(Pr["seeds"]))
axs[0].axvline(63.5, color="#2ca02c", ls="--", lw=2); axs[0].set_title("image (true split: green)", fontsize=16)
for a, key, title in [(axs[1], "p0", "walker"), (axs[2], "p1", "walker with priors")]:
    a.imshow(img(Pr[key]), cmap=DIV, vmin=0, vmax=1)
    a.contour(img(Pr[key]), [0.5], colors=["k"], linewidths=2)
    a.axvline(63.5, color="#2ca02c", ls="--", lw=2)
    a.set_title(title, fontsize=16)
for a in axs:
    clean(a)
fig.tight_layout(); fig.savefig(OUT + "fig-priors.png", dpi=100); plt.close(fig)

# ---- fig-gcrw.png: graph cut (seeds only) vs random walker on two scenes ----
G = D["gcrw"]
fig, axs = plt.subplots(2, 2, figsize=(8.2, 8.4))
for row, kind in enumerate(["objects", "gap"]):
    g = G[kind]
    for col, (key, title) in enumerate([("gc", "graph cut"), ("rw", "random walker")]):
        a = axs[row, col]
        a.imshow(img(g["I"]), cmap="gray", vmin=0, vmax=255)
        lab = img(g[key])
        mask = (lab == 1)
        ov = np.zeros((128, 128, 4)); ov[mask] = (*matplotlib.colors.to_rgb(OBJ), 0.45)
        a.imshow(ov)
        a.contour(mask.astype(float), [0.5], colors=["#ffe000"], linewidths=2)
        a.imshow(seeds_rgba(g["seeds"]))
        if row == 0:
            a.set_title(title, fontsize=18)
        clean(a)
fig.tight_layout(); fig.savefig(OUT + "fig-gcrw.png", dpi=95); plt.close(fig)

# ---- fig-cg.svg: CG residual against iteration ----
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for (sg, h), c in zip(D["cg"].items(), ["#d62728", "#ff7f0e", "#1f77b4"]):
    ax.semilogy(np.arange(len(h)) * 5, h, color=c, lw=2.5, label=f"σ = {sg}")
ax.set_xlabel("CG iteration"); ax.set_ylabel("relative residual")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT + "fig-cg.svg"); plt.close(fig)
print("done")
