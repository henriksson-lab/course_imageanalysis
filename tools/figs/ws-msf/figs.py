"""Figures for ws-msf/. Run from the repository root after compute.ts (see there):

    python3 -I tools/figs/ws-msf/figs.py <rawdir>

All images are synthetic.
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 17, "font.family": "DejaVu Sans", "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
raw = sys.argv[1]
OUT = "ws-msf/"
D = json.load(open(f"{raw}/ws.json"))
A_COL, B_COL = "#ff7f0e", "#1f77b4"


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ---- fig-landscape.svg: a 1D landscape and its edge altitudes ----
f = [4, 2, 1, 3, 6, 5, 2, 0, 3, 7, 4, 2, 3]
fig, ax = plt.subplots(figsize=(9.5, 4.2))
ax.bar(range(len(f)), f, color="#bbb", width=0.8)
for i in range(len(f) - 1):
    a = max(f[i], f[i + 1])
    ax.plot([i + 0.1, i + 0.9], [a + 0.6, a + 0.6], color="#333", lw=2)
    ax.text(i + 0.5, a + 0.9, str(a), ha="center", fontsize=15, color="#d62728")
for i, v in enumerate(f):
    ax.text(i, v / 2, str(v), ha="center", va="center", fontsize=14)
for m, c, lab in [(2, A_COL, "A"), (7, B_COL, "B"), (11, "#2ca02c", "C")]:
    ax.plot(m, -0.6, "^", ms=14, color=c); ax.text(m, -1.5, lab, ha="center", fontsize=15, color=c)
ax.set_ylim(-2, 9); ax.set_yticks([]); ax.set_xticks([])
ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "fig-landscape.svg"); plt.close(fig)

# ---- fig-prim.svg: Prim from two markers on the 4 x 4 grid, four frames ----
F = np.array([1, 3, 9, 4, 2, 5, 8, 6, 3, 7, 2, 7, 4, 9, 6, 0]).reshape(4, 4)
edges = []
for r in range(4):
    for c in range(4):
        if c < 3: edges.append(((r, c), (r, c + 1)))
        if r < 3: edges.append(((r, c), (r + 1, c)))
alt = {e: max(F[e[0]], F[e[1]]) for e in edges}
# replay Prim with FIFO among ties (as lib/mst.ts primForest)
import heapq
lab = {(0, 0): 1, (3, 3): 2}
counter = 0
heap = []
queued = set()
def push_edges(v):
    global counter
    for e in edges:
        if v in e and e not in queued:
            queued.add(e); heapq.heappush(heap, (alt[e], counter, e)); counter += 1
for v in [(0, 0), (3, 3)]:
    push_edges(v)
steps = []
while heap:
    a, _, e = heapq.heappop(heap)
    u, v = e
    if u in lab and v in lab:
        steps.append((e, "cut" if lab[u] != lab[v] else "skip", dict(lab)))
        continue
    src, dst = (u, v) if u in lab else (v, u)
    lab[dst] = lab[src]
    steps.append((e, "grow", dict(lab)))
    push_edges(dst)
grow_idx = [i for i, s in enumerate(steps) if s[1] == "grow"]
frames = [grow_idx[1], grow_idx[5], grow_idx[9], len(steps) - 1]
fig, axs = plt.subplots(1, 4, figsize=(15.5, 4.4))
for ax, fi in zip(axs, frames):
    labs = steps[fi][2]
    state = {}
    for e, kind, _ in steps[:fi + 1]:
        state[e] = kind
    for e in edges:
        (r1, c1), (r2, c2) = e
        k = state.get(e)
        col = "#222" if k == "grow" else "#d62728" if k == "cut" else "#ccc"
        lw = 6 if k in ("grow", "cut") else 1.5
        ax.plot([c1, c2], [r1, r2], color=col, lw=lw, ls="--" if k == "cut" else "-", zorder=1)
        ax.text((c1 + c2) / 2, (r1 + r2) / 2, str(alt[e]), ha="center", va="center", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"), zorder=2)
    for r in range(4):
        for c in range(4):
            l = labs.get((r, c), 0)
            fc = A_COL if l == 1 else B_COL if l == 2 else "white"
            ax.add_patch(plt.Rectangle((c - 0.27, r - 0.27), 0.54, 0.54, facecolor=fc, edgecolor="#444", lw=3 if (r, c) in [(0, 0), (3, 3)] else 1.2, zorder=3))
            ax.text(c, r, str(F[r, c]), ha="center", va="center", fontsize=15, zorder=4, color="w" if l else "k")
    e, kind, _ = steps[fi]
    ax.set_title(f"step {fi + 1}: edge {alt[e]} ({kind})", fontsize=15)
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(3.5, -0.5); ax.set_aspect("equal"); clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-prim.svg"); plt.close(fig)

# ---- fig-root.svg: the virtual root ----
fig, ax = plt.subplots(figsize=(8.4, 3.8))
xs = list(range(7)); w = [2, 5, 1, 3, 6, 2]
for i, ww in enumerate(w):
    ax.plot([i, i + 1], [0, 0], color="#222" if i != 4 else "#d62728", lw=4, ls="-" if i != 4 else "--")
    ax.text(i + 0.5, 0.18, str(ww), ha="center", fontsize=16)
for i in xs:
    fc = A_COL if i == 0 else B_COL if i == 6 else "white"
    ax.add_patch(plt.Circle((i, 0), 0.17, facecolor=fc, edgecolor="#333", lw=1.5, zorder=3))
    ax.text(i, -0.45, "abcdefg"[i], ha="center", fontsize=16)
ax.add_patch(plt.Circle((3, 1.6), 0.2, facecolor="#ddd", edgecolor="#333", lw=1.5, zorder=3))
ax.text(3, 1.6, "r", ha="center", va="center", fontsize=16, zorder=4)
for i in [0, 6]:
    ax.plot([3, i], [1.6, 0], color="#888", lw=2, ls=":")
    ax.text((3 + i) / 2 + (0.35 if i == 0 else -0.35), 0.95, "−∞", fontsize=16, color="#555", ha="center")
ax.set_xlim(-0.6, 6.6); ax.set_ylim(-0.7, 2.0); ax.set_aspect("equal"); clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-root.svg"); plt.close(fig)

# ---- fig-three.png: the three seeded methods on three scenes ----
C = D["cases"]
rows = [("gap", "gap in a wall, noise 16"), ("ramp", "no edge, small seed"), ("objects", "weak boundary")]
fig, axs = plt.subplots(3, 3, figsize=(10.2, 10.4))
for r, (key, rowt) in enumerate(rows):
    c = C[key]
    I = np.asarray(c["I"]).reshape(128, 128)
    seeds = np.asarray(c["seeds"]).reshape(128, 128)
    for col, (m, title) in enumerate([("rw", "random walker (squares)"), ("gc", "graph cut (sum)"), ("ws", "watershed (max)")]):
        ax = axs[r, col]
        ax.imshow(I, cmap="gray", vmin=0, vmax=255)
        mask = np.asarray(c[m]).reshape(128, 128) == 1
        ov = np.zeros((128, 128, 4)); ov[mask] = (1, 0.5, 0.05, 0.42)
        ax.imshow(ov)
        ax.contour(mask.astype(float), [0.5], colors=["#ffe000"], linewidths=1.8)
        sv = np.zeros((128, 128, 4)); sv[seeds == 1] = (0.9, 0.3, 0, 1); sv[seeds == 2] = (0.12, 0.47, 0.71, 1)
        ax.imshow(sv)
        if r == 0:
            ax.set_title(title, fontsize=16)
        ax.text(2, 124, f"{int(mask.sum())} px", color="w", fontsize=14)
        clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-three.png", dpi=85); plt.close(fig)

# ---- fig-chain3.svg: the 7-node chain under the three methods ----
fig, ax = plt.subplots(figsize=(7.2, 3.6))
w = [0.2, 1, 1, 1, 1, 0.25]
for i, ww in enumerate(w):
    ax.plot([i, i + 1], [0, 0], color="#333", lw=1.5 + 7 * ww, solid_capstyle="butt")
    ax.text(i + 0.5, 0.17, f"d {1 / ww:g}", ha="center", fontsize=17)
for i in range(7):
    fc = A_COL if i == 0 else B_COL if i == 6 else "white"
    ax.add_patch(plt.Circle((i, 0), 0.17, facecolor=fc, edgecolor="#333", lw=1.5, zorder=3))
    ax.text(i, 0, str(i + 1), ha="center", va="center", fontsize=15, zorder=4, color="w" if i in (0, 6) else "k")
for x, y, lab, c in [(3.5, -0.5, "random walker", "#e6550d"), (0.5, -0.5, "graph cut", "#d62728"), (0.5, -0.95, "watershed", "#2ca02c")]:
    ax.plot([x, x], [y + 0.12, -0.22], color=c, lw=3)
    ax.text(x + 0.08, y - 0.05, lab, color=c, fontsize=17)
ax.set_xlim(-0.4, 6.6); ax.set_ylim(-1.15, 0.42); clean(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-chain3.svg"); plt.close(fig)

# ---- fig-timing.svg ----
T = D["timing"]
fig, ax = plt.subplots(figsize=(6.6, 4.4))
labels = ["watershed / MSF", "random walker (CG)", "graph cut (max-flow)"]
cols = ["#2ca02c", "#ff7f0e", "#d62728"]
for j, (lab, c) in enumerate(zip(labels, cols)):
    for i, n in enumerate(["128", "256"]):
        v = T[n][j]
        ax.bar(i * 4 + j, v, color=c, label=lab if i == 0 else None)
        ax.text(i * 4 + j, v * 1.15, f"{v:.0f}", ha="center", fontsize=14)
ax.set_yscale("log"); ax.set_ylim(5, 10000)
ax.set_xticks([1, 5]); ax.set_xticklabels(["128 × 128", "256 × 256"])
ax.set_ylabel("ms (node, one core)")
ax.legend(frameon=False, fontsize=13, loc="upper left")
fig.tight_layout(); fig.savefig(OUT + "fig-timing.svg"); plt.close(fig)
print("done", [(s[0], s[1]) for s in steps])
