# Figures for ws-flooding/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/ws-flooding/figs.py
# The flooding traces come from the course's lib/watershed.ts (data.ts, via node).
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ws-landscape"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from wsstyle import *

setup()
OUT = "ws-flooding"
D = run_ts(HERE)
LETTER = " AB"


def grid(ax, rows, cols, f, labels=None, markers=None, queue=None, popped=None, nb=(), title=None, fs=18, small=14):
    """A grid of heights; labels 1/2 as basin fills, -1/0-after-pop as line (black), markers framed."""
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c; v = f[i]
            L = labels[i] if labels is not None else 0
            if L > 0: fc, tc = BASIN_FILL[L - 1], INK
            elif L < 0: fc, tc = DAM, "white"
            else:
                g = 1 - 0.06 * v; fc, tc = (g, g, g), INK
            ax.add_patch(Rectangle((c, r), 1, 1, fc=fc, ec="white", lw=2))
            ax.text(c + 0.5, r + 0.47, str(v), ha="center", va="center", fontsize=fs, color=tc)
            if markers is not None and markers[i]:
                ax.add_patch(Rectangle((c + 0.07, r + 0.07), 0.86, 0.86, fc="none", ec=BASIN_INK[markers[i] - 1], lw=3.5))
    if queue:
        for q in queue:
            r, c = divmod(q["p"], cols)
            ax.add_patch(Rectangle((c + 0.07, r + 0.07), 0.86, 0.86, fc="none", ec="#555", lw=1.6, ls=(0, (3, 2))))
            ax.text(c + 0.93, r + 0.93, f"#{q['arrival']}", ha="right", va="bottom", fontsize=small, color="#444")
    for p in nb:
        r, c = divmod(p, cols)
        ax.add_patch(Rectangle((c + 0.12, r + 0.12), 0.76, 0.76, fc="none", ec=BLUE, lw=2.2, ls=(0, (4, 2))))
    if popped is not None:
        r, c = divmod(popped, cols)
        ax.add_patch(Rectangle((c + 0.04, r + 0.04), 0.92, 0.92, fc="none", ec=RED, lw=3.5))
    ax.set_xlim(-0.05, cols + 0.05); ax.set_ylim(rows + 0.05, -0.05); ax.set_aspect("equal"); ax.axis("off")
    if title: ax.set_title(title, fontsize=17, pad=4)


def state(T, k):
    """Labels (line = -1) and queue after pop k (k = 0: before the first pop)."""
    if k == 0: return T["init"]["labels"], T["init"]["queue"]
    s = T["steps"][k - 1]
    return s["labels"], s["queue"]


H = D["hand4"]; R, C = H["rows"], H["cols"]

# --- 1. neighbourhoods and a regional minimum on a plateau -------------------------------------
f = plt.figure(figsize=(560 / 72, 240 / 72))
for k, (name, offs) in enumerate([("4 neighbours", [(0, -1), (-1, 0), (1, 0), (0, 1)]),
                                   ("8 neighbours", [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy])]):
    ax = f.add_axes([0.0 + 0.21 * k, 0.14, 0.16, 0.6])
    for y in range(3):
        for x in range(3):
            on = (x - 1, y - 1) in offs
            ax.add_patch(Rectangle((x, y), 1, 1, fc=(BLUE if on else "white"), alpha=0.55 if on else 1, ec=GRID, lw=1.5))
    ax.add_patch(Rectangle((1, 1), 1, 1, fc=INK, ec=GRID))
    ax.set_xlim(0, 3); ax.set_ylim(3, 0); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(name, fontsize=17)
ex = [[5, 5, 6, 6, 7, 7], [5, 1, 1, 4, 3, 7], [6, 1, 2, 4, 3, 6], [6, 5, 4, 4, 5, 6]]
ax = f.add_axes([0.44, 0.0, 0.56, 1.0])
grid(ax, 4, 6, sum(ex, []), fs=20)
for cells, col in [([(1, 1), (1, 2), (2, 1)], BASIN_INK[0]), ([(1, 4), (2, 4)], BASIN_INK[1])]:
    for (r, c) in cells:
        for (dr, dc, x0, y0, x1, y1) in [(-1, 0, 0, 0, 1, 0), (1, 0, 0, 1, 1, 1), (0, -1, 0, 0, 0, 1), (0, 1, 1, 0, 1, 1)]:
            if (r + dr, c + dc) not in cells:
                ax.plot([c + x0, c + x1], [r + y0, r + y1], color=col, lw=4, solid_capstyle="round")
ax.set_title("two regional minima", fontsize=18, pad=4)
save(f, f"{OUT}/fig-neighbours.svg")

# --- 2. one moment of Meyer's algorithm: after pop 6 -----------------------------------------
K = 6
lab, q = state(H, K)
nxt = q[0]
f = plt.figure(figsize=(560 / 72, 330 / 72))
ax = f.add_axes([0.0, 0.0, 0.56, 0.92])
grid(ax, R, C, H["f"], lab, H["markers"], q, popped=nxt["p"], fs=20, small=16,
     nb=[p for p in range(R * C) if lab[p] > 0 and abs(p // C - nxt["p"] // C) + abs(p % C - nxt["p"] % C) == 1])
ax.set_title(f"after {K} pops: water level {H['steps'][K - 1]['level']:g}", fontsize=17, pad=4)
ax = f.add_axes([0.6, 0.0, 0.4, 0.92]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(len(q) + 1.2, 0)
ax.text(0, 0.5, "queue (height, #)", fontsize=17, va="center", fontweight="bold")
for j, it in enumerate(q):
    r, c = divmod(it["p"], C)
    col = RED if j == 0 else INK
    ax.text(0.02, j + 1.5, f"{it['value']}  #{it['arrival']:<3d} ({r},{c})", fontsize=18, family="monospace", va="center", color=col)
ax.text(1.02, 1.5, "← next", fontsize=17, color=RED, va="center", ha="right")
save(f, f"{OUT}/fig-meyer.svg")

# --- 3. the water rises evenly: snapshots at the end of each level ----------------------------
ends = {}
for k, s in enumerate(H["steps"]):
    ends[s["level"]] = k + 1
snaps = [(ends[1], "level 1"), (ends[2], "level 2"), (ends[3], "level 3"), (len(H["steps"]), "end")]
f, axs = plt.subplots(1, 4, figsize=(660 / 72, 220 / 72))
for ax, (k, name) in zip(axs, snaps):
    lab, q = state(H, k)
    grid(ax, R, C, H["f"], lab, H["markers"], fs=14, title=f"{name}\n{k} pops")
f.subplots_adjust(wspace=0.08)
save(f, f"{OUT}/fig-levels.svg")

# --- 4. plateau: FIFO vs LIFO -----------------------------------------------------------------
f, axs = plt.subplots(2, 1, figsize=(420 / 72, 400 / 72))
for ax, key, name in [(axs[0], "plateauFifo", "first in, first out"), (axs[1], "plateauLifo", "last in, first out")]:
    T = D[key]; lab = [(-1 if v == 0 else v) for v in T["labels"]]
    nl = sum(1 for v in T["labels"] if v == 0)
    grid(ax, T["rows"], T["cols"], T["f"], lab, T["markers"], fs=16, title=f"{name}: {nl} line pixels")
f.subplots_adjust(hspace=0.25)
save(f, f"{OUT}/fig-plateau.svg")

# --- 5. the same queue as buckets ---------------------------------------------------------------
lab, q = state(H, K)
f, ax = plt.subplots(figsize=(380 / 72, 330 / 72))
L = 7
for v in range(L):
    y = v
    ax.add_patch(Rectangle((0, y), 0.9, 0.9, fc="#eee", ec=GRID))
    ax.text(0.45, y + 0.45, str(v), ha="center", va="center", fontsize=17)
    items = [it for it in q if it["value"] == v]
    for j, it in enumerate(items):
        ax.add_patch(Rectangle((1.3 + 1.25 * j, y + 0.05), 1.15, 0.8, fc=BASIN_FILL[it["from"] - 1], ec="white"))
        ax.text(1.3 + 1.25 * j + 0.575, y + 0.45, f"#{it['arrival']}", ha="center", va="center", fontsize=16)
ax.text(0.45, -0.3, "height", ha="center", va="bottom", fontsize=16)
c = min(it["value"] for it in q)
ax.annotate("c", xy=(-0.05, c + 0.45), xytext=(-1.0, c + 0.45), fontsize=18, color=RED, va="center",
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
ax.set_xlim(-1.2, 6.3); ax.set_ylim(L, -1.0); ax.axis("off")
save(f, f"{OUT}/fig-buckets.svg")
print("ws-flooding figures written")
