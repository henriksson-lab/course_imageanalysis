"""Figures for mst-seg/. Run from the repository root after compute.ts (see there):

    python3 -I tools/figs/mst-seg/figs.py <rawdir>

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
OUT = "mst-seg/"
D = json.load(open(f"{raw}/mst.json"))


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def colours(lab, seed=2):
    lab = np.asarray(lab)
    rng = np.random.default_rng(seed)
    pal = rng.uniform(0.25, 0.95, (lab.max() + 1, 3))
    return pal[lab]


def bounds(lab, n):
    l = np.asarray(lab).reshape(n, n)
    b = np.zeros_like(l, bool)
    b[:, :-1] |= l[:, :-1] != l[:, 1:]
    b[:-1, :] |= l[:-1, :] != l[1:, :]
    return b


# ---- fig-dissim.svg: a 4 x 4 image with |ΔI| on its edges ----
I = np.array([[12, 14, 13, 40], [11, 15, 41, 44], [13, 38, 42, 45], [12, 39, 43, 41]])
fig, ax = plt.subplots(figsize=(5.2, 5.2))
for r in range(4):
    for c in range(4):
        g = 0.35 + 0.6 * (I[r, c] - 10) / 36
        ax.add_patch(plt.Rectangle((c - 0.3, r - 0.3), 0.6, 0.6, facecolor=(g, g, g), edgecolor="#444", lw=1.2))
        ax.text(c, r, str(I[r, c]), ha="center", va="center", fontsize=17, color="k" if g > 0.55 else "w")
        for dr, dc in [(0, 1), (1, 0)]:
            r2, c2 = r + dr, c + dc
            if r2 < 4 and c2 < 4:
                d = abs(int(I[r, c]) - int(I[r2, c2]))
                ax.text((c + c2) / 2, (r + r2) / 2, str(d), ha="center", va="center", fontsize=15, color="#d62728" if d > 10 else "#1f77b4",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
ax.set_xlim(-0.5, 3.5); ax.set_ylim(3.5, -0.5); ax.set_aspect("equal"); clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-dissim.svg"); plt.close(fig)

# ---- fig-kruskal.svg: Kruskal on six nodes, four frames ----
pos = {"A": (0, 1), "B": (1.2, 1.6), "C": (1.0, 0.2), "D": (2.4, 1.2), "E": (2.2, -0.3), "F": (3.4, 0.4)}
edges = [("A", "B", 4), ("A", "C", 1), ("B", "C", 3), ("B", "D", 2), ("C", "D", 5), ("C", "E", 7), ("D", "F", 6), ("E", "F", 3.5)]
order = sorted(edges, key=lambda e: e[2])
frames = [(2, "AC 1, BD 2: join"), (4, "BC 3, EF 3.5: join"), (6, "AB 4, CD 5: skip (cycle)"), (8, "DF 6: join; CE 7: skip")]
fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.4))
for ax, (k, title) in zip(axs.ravel(), frames):
    parent = {v: v for v in pos}
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    status = {}
    for i, (a, b, w) in enumerate(order[:k]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb; status[(a, b)] = "tree"
        else:
            status[(a, b)] = "skip"
    last = {(a, b) for a, b, _ in order[max(0, k - 2):k]}
    for a, b, w in edges:
        st = status.get((a, b))
        col = "#222" if st == "tree" else "#ccc"
        lw = 5 if st == "tree" else 1.5
        ls = "--" if st == "skip" else "-"
        if (a, b) in last:
            col = "#ff7f0e" if st == "tree" else "#d62728"
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.plot([x1, x2], [y1, y2], color=col, lw=lw, ls=ls, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{w:g}", fontsize=15, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"), zorder=2)
    for v, (x, y) in pos.items():
        ax.add_patch(plt.Circle((x, y), 0.2, facecolor="#9ecae1", edgecolor="#333", lw=1.5, zorder=3))
        ax.text(x, y, v, ha="center", va="center", fontsize=16, zorder=4)
    ax.set_title(title, fontsize=15)
    ax.set_xlim(-0.4, 3.8); ax.set_ylim(-0.7, 2.0); ax.set_aspect("equal"); clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-kruskal.svg"); plt.close(fig)

# ---- fig-mst.png: the MST of a 16 x 16 crop of the micrograph ----
C = D["crop"]
n = 16
fig, ax = plt.subplots(figsize=(5.4, 5.4))
ax.imshow(np.asarray(C["I"]).reshape(n, n), cmap="gray", vmin=0, vmax=255)
d = np.asarray(C["d"])
cm = plt.get_cmap("plasma")
for p, q, dd, t in zip(C["p"], C["q"], d, C["tree"]):
    if t:
        ax.plot([p % n, q % n], [p // n, q // n], color=cm(min(1, dd / 40)), lw=3)
clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-mst.png", dpi=110); plt.close(fig)

# ---- fig-single.png and fig-fhk.png: the micrograph ----
CE = D["cell"]
for key, name, fmt in [("sl", "fig-single.png", "t = {}"), ("fh", "fig-fhk.png", "k = {}")]:
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.9))
    for ax, (k, v) in zip(axs, CE[key].items()):
        ax.imshow(colours(v["labels"]).reshape(256, 256, 3))
        ax.set_title(f"{fmt.format(k)}: {v['count']} regions", fontsize=17); clean(ax)
    fig.tight_layout(); fig.savefig(OUT + name, dpi=90); plt.close(fig)

# ---- fig-scene.png: single linkage cannot hold texture and keep the faint step; FH can ----
S = D["scene"]
fig, axs = plt.subplots(1, 4, figsize=(16, 4.6))
axs[0].imshow(np.asarray(S["I"]).reshape(256, 256), cmap="gray", vmin=0, vmax=255); axs[0].set_title("test scene", fontsize=17)
for ax, key, title in [(axs[1], "sl5", "single linkage t = 5"), (axs[2], "sl9", "single linkage t = 9"), (axs[3], "fh3000", "FH k = 3000")]:
    ax.imshow(colours(S[key]["labels"], 5).reshape(256, 256, 3)); ax.set_title(f"{title}: {S[key]['count']}", fontsize=17)
for ax in axs:
    clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-scene.png", dpi=85); plt.close(fig)

# ---- fig-kscale.png: large k keeps small high-contrast spots; fig-leak.png: a narrow gap ----
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.9))
for ax, key, title in [(axs[0], "fh300", "k = 300"), (axs[1], "fh10000", "k = 10 000")]:
    ax.imshow(colours(S[key]["labels"], 5).reshape(256, 256, 3)); ax.set_title(f"FH {title}: {S[key]['count']} regions", fontsize=17); clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-kscale.png", dpi=90); plt.close(fig)

img = np.asarray(S["I"]).reshape(256, 256)
lab = np.asarray(S["fh3000"]["labels"]).reshape(256, 256)
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.4))
sl = (slice(120, 195), slice(140, 240))
axs[0].imshow(img[sl], cmap="gray", vmin=0, vmax=255); axs[0].set_title("two bright halves, a dark line with a 3-px gap", fontsize=14)
axs[1].imshow(colours(lab.ravel(), 5).reshape(256, 256, 3)[sl]); axs[1].set_title("FH k = 3000: one region", fontsize=15)
for ax in axs:
    clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-leak.png", dpi=90); plt.close(fig)

# ---- fig-oned.svg: the 1D example: texture + faint step ----
O = D["oned"]
b = np.array(O["b"])
fig, axs = plt.subplots(4, 1, figsize=(8.4, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1, 1]})
axs[0].bar(range(12), b, color="#888")
for i in range(11):
    axs[0].text(i + 0.5, max(b[i], b[i + 1]) + 2, str(abs(b[i + 1] - b[i])), ha="center", fontsize=14, color="#d62728")
axs[0].set_ylim(0, 33); axs[0].set_ylabel("value")
pal = ["#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]
for ax, key, title in [(axs[1], "sl5", "single linkage, t = 5"), (axs[2], "sl7", "single linkage, t = 7"), (axs[3], "fh8", "FH, k = 8")]:
    labs = O[key]
    for i, l in enumerate(labs):
        ax.add_patch(plt.Rectangle((i - 0.45, 0), 0.9, 1, color=pal[(l - 1) % len(pal)]))
    ax.set_ylim(0, 1); ax.set_yticks([]); ax.text(-0.6, 0.5, title, ha="right", va="center", fontsize=15)
    for sp in ax.spines.values():
        sp.set_visible(False)
axs[3].set_xticks(range(12)); axs[3].set_xticklabels([str(i + 1) for i in range(12)])
fig.tight_layout(); fig.savefig(OUT + "fig-oned.svg"); plt.close(fig)
print("done")
