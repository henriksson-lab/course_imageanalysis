"""Figures for superpixels/. Run from the repository root after compute.ts (see there):

    python3 -I tools/figs/superpixels/figs.py <rawdir>

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
OUT = "superpixels/"
D = json.load(open(f"{raw}/sp.json"))
N = 256
RGB = np.stack([np.asarray(c, float).reshape(N, N) for c in D["rgb"]], -1) / 255


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def bnd(lab):
    l = np.asarray(lab).reshape(N, N)
    b = np.zeros_like(l, bool)
    b[:, :-1] |= l[:, :-1] != l[:, 1:]
    b[:-1, :] |= l[:-1, :] != l[1:, :]
    return b


def overlay(base, lab, col=(1, 0.9, 0)):
    img = base.copy()
    img[bnd(lab)] = col
    return img


def grey(I):
    g = np.asarray(I, float).reshape(N, N) / 255
    return np.stack([g, g, g], -1)


# ---- fig-why.png: pixels vs superpixels with the RAG ----
C = D["cell"]
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.9))
axs[0].imshow(grey(C["I"])); axs[0].set_title("65 536 pixels", fontsize=17); clean(axs[0])
axs[1].imshow(overlay(grey(C["I"]) * 0.8 + 0.1, C["labels"], (1, 0.85, 0)))
for a, b in zip(C["a"], C["b"]):
    axs[1].plot([C["cx"][a], C["cx"][b]], [C["cy"][a], C["cy"][b]], color="#d62728", lw=0.7)
axs[1].scatter(C["cx"], C["cy"], s=5, c="#1f77b4", zorder=3)
axs[1].set_xlim(0, 255); axs[1].set_ylim(255, 0)
axs[1].set_title(f"{C['count']} superpixels, {len(C['a'])} edges", fontsize=17); clean(axs[1])
fig.tight_layout(); fig.savefig(OUT + "fig-why.png", dpi=90); plt.close(fig)

# ---- fig-iters.png: iterations 0, 1, 3, 10 (with the grid start and one window) ----
It = D["iters"]
S = It["S"]
fig, axs = plt.subplots(2, 2, figsize=(8.4, 8.8))
for ax, lab, cen, it in zip(axs.ravel(), It["labels"], It["centres"], [0, 1, 3, 10]):
    ax.imshow(RGB if it == 0 else overlay(RGB, lab))
    c = np.asarray(cen)
    ax.plot(c[:, 0], c[:, 1], "o", ms=3.5, color="#d62728")
    if it == 0:
        k = 45
        ax.add_patch(plt.Rectangle((c[k, 0] - S, c[k, 1] - S), 2 * S, 2 * S, fill=False, ec="#d62728", lw=2.5))
        ax.plot(c[k, 0], c[k, 1], "o", ms=8, color="k")
        ax.set_title(f"start: grid, S = {S:.1f}", fontsize=17)
    else:
        ax.set_title(f"after {it} iteration{'s' if it > 1 else ''}", fontsize=16)
    clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-iters.png", dpi=80); plt.close(fig)

# ---- fig-compact.png ----
Cm = D["compact"]
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.8))
for ax, m in zip(axs, ["5", "20", "80"]):
    v = Cm[m]
    ax.imshow(overlay(RGB, v["labels"])); ax.set_title(f"m = {m}: recall {v['br']:.2f}", fontsize=17); clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-compact.png", dpi=85); plt.close(fig)

# ---- fig-orphans.png ----
O = D["orphans"]
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.9))
crop = (slice(30, 130), slice(20, 130))
for ax, key, title in [(axs[0], "before", f"no clean-up: {O['labelsBefore']} labels, {O['pieces']} pieces"), (axs[1], "after", f"connectivity: {O['after_count']} superpixels")]:
    ax.imshow(overlay(RGB, O[key])[crop]); ax.set_title(title, fontsize=14); clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-orphans.png", dpi=90); plt.close(fig)

# ---- fig-methods.png: SLIC, FH, watershed at about K = 200 (grey test scene) ----
Mt = D["methods"]
g = grey(Mt["I"])
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.9))
for ax, key, name in [(axs[0], "slic", "SLIC"), (axs[1], "fh", "FH"), (axs[2], "ws", "watershed (grid markers)")]:
    br, ue, asa, n = Mt["metrics"][key]
    ax.imshow(overlay(g, Mt[key])); ax.set_title(f"{name}: {n}, recall {br:.2f}", fontsize=15); clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-methods.png", dpi=85); plt.close(fig)

# ---- fig-metrics.svg ----
Cv = D["curves"]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
for key, name, c in [("slic", "SLIC", "#d62728"), ("fh", "FH", "#2ca02c"), ("ws", "watershed", "#1f77b4")]:
    a = np.asarray(Cv[key])
    axs[0].semilogx(a[:, 0], a[:, 1], "o-", color=c, lw=2.5, label=name)
    axs[1].semilogx(a[:, 0], a[:, 2], "o-", color=c, lw=2.5, label=name)
axs[0].set_ylabel("boundary recall (2 px)"); axs[1].set_ylabel("under-segmentation error")
for ax in axs:
    ax.set_xlabel("number of superpixels")
axs[0].legend(frameon=False, fontsize=14)
fig.tight_layout(); fig.savefig(OUT + "fig-metrics.svg"); plt.close(fig)

# ---- fig-gc.png: graph cut on pixels vs superpixels ----
G = D["gc"]
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.9))
seeds = np.asarray(G["seeds"]).reshape(N, N)
for ax, key, title in [(axs[0], "pix", f"pixel graph: {N * N} nodes, {G['tPix']:.0f} ms"), (axs[1], "sp", f"{G['nodes']} superpixels: {G['tSp']:.0f} ms")]:
    img = RGB * 0.6 + 0.2
    m = np.asarray(G[key]).reshape(N, N) == 1
    img[m] = img[m] * 0.5 + np.array([1, 0.5, 0.05]) * 0.5
    if key == "sp":
        img[bnd(G["spLabels"])] = (0.3, 0.3, 0.3)
    img[seeds == 1] = (0.9, 0.3, 0); img[seeds == -1] = (0.12, 0.47, 0.71)
    ax.imshow(img); ax.set_title(title, fontsize=15); clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-gc.png", dpi=90); plt.close(fig)

# ---- fig-thin.png ----
T = D["thin"]
crop = (slice(115, 200), slice(130, 245))
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.2))
for ax, key, k, br in [(axs[0], "k60", 60, T["br60"]), (axs[1], "k400", 400, T["br400"])]:
    ax.imshow(overlay(grey(T["I"]), T[key])[crop]); ax.set_title(f"K = {k}: recall {br:.2f}", fontsize=16); clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-thin.png", dpi=90); plt.close(fig)
print("done")
