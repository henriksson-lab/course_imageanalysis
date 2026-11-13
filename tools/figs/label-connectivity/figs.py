# Figures for label-connectivity/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/label-connectivity/figs.py
# The flood-fill orders come from the demo's own generator (data.ts, via node).
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, RegularPolygon
from labstyle import setup, fig, save, run_ts, GRID, INK, PIX, ORANGE, BLUE, GREEN, RED

setup()
OUT = "label-connectivity"
D = run_ts(HERE)
w, h = D["w"], D["h"]

# --- 1. flood fill: queue rings vs stack arms --------------------------------------------
f, axs = plt.subplots(1, 2, figsize=(560 / 72, 330 / 72))
cmap = plt.get_cmap("viridis")
for ax, kind, name, big in [(axs[0], "bfs", "queue", "rings"), (axs[1], "dfs", "stack", "arms")]:
    order = D[kind]["order"]
    rank = {p: k for k, p in enumerate(order)}
    n = len(order)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            k = rank[y * w + x]
            ax.add_patch(Rectangle((x, y), 1, 1, fc=cmap(k / (n - 1)), ec="white", lw=1))
    # the path of the first 12 pixels taken, to show the direction of growth
    pts = np.array([[p % w + 0.5, p // w + 0.5] for p in order[:13]])
    if kind == "dfs":
        ax.plot(pts[:, 0], pts[:, 1], "-", color=ORANGE, lw=2.5)
    ax.plot(pts[0, 0], pts[0, 1], "o", color=ORANGE, ms=9)
    ax.set_xlim(1, w - 1); ax.set_ylim(h - 1, 1); ax.set_aspect("equal"); ax.axis("off")
    mx = D[kind]["note"].split("Largest ")[1].split(":")[1].strip().split(" ")[0]
    ax.set_title(f"{name}: {big}", fontsize=19, pad=6)
    ax.text((w) / 2, h - 0.6, f"largest {name}: {mx} pixels", ha="center", va="top", fontsize=17)
f.subplots_adjust(wspace=0.12)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 49))
cb = f.colorbar(sm, ax=axs, orientation="horizontal", fraction=0.06, pad=0.2, aspect=30)
cb.set_label("order in which pixels are taken out (1 = seed)", fontsize=17)
cb.set_ticks([1, 10, 20, 30, 40, 49]); cb.ax.tick_params(labelsize=16)
save(f, f"{OUT}/fig-floodfill.svg")

# --- 2. hexagonal grid and 3D neighbourhoods ---------------------------------------------
f = plt.figure(figsize=(520 / 72, 470 / 72))
ax = f.add_axes([0.0, 0.52, 1.0, 0.46])
r = 1.0
for q in range(-2, 3):
    for rr in range(-2, 3):
        d = max(abs(q), abs(rr), abs(q + rr))
        if d > 2: continue
        cx, cy = np.sqrt(3) * r * (q + rr / 2), 1.5 * r * rr
        fc, ec = (INK, "white") if d == 0 else (PIX, "white") if d == 1 else ("white", GRID)
        ax.add_patch(RegularPolygon((cx, cy), 6, radius=r * 0.97, orientation=0, fc=fc, ec=ec, lw=1.5))
ax.set_xlim(-4.4, 12.5); ax.set_ylim(-4.0, 4.0); ax.set_aspect("equal"); ax.axis("off")
ax.text(4.8, 1.1, "hexagonal grid:", fontsize=22, va="center")
ax.text(4.8, 0.0, "6 neighbours, all", fontsize=22, va="center")
ax.text(4.8, -1.1, "share an edge", fontsize=22, va="center")

def cube(ax3, which, title):
    g = np.zeros((3, 3, 3), bool); col = np.empty((3, 3, 3), object)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                d = abs(i - 1) + abs(j - 1) + abs(k - 1)
                if d == 0: g[i, j, k] = True; col[i, j, k] = INK
                elif d <= which: g[i, j, k] = True; col[i, j, k] = PIX
    ax3.voxels(g, facecolors=col, edgecolors="#555", linewidth=0.5, alpha=0.95)
    ax3.set_box_aspect((1, 1, 1)); ax3.set_axis_off(); ax3.view_init(elev=22, azim=-58)
    ax3.set_title(title, fontsize=22, y=-0.1)

for k, (which, t) in enumerate([(1, "$N_6$: faces"), (2, "$N_{18}$: + edges"), (3, "$N_{26}$: + corners")]):
    ax3 = f.add_axes([k / 3, 0.0, 1 / 3, 0.5], projection="3d")
    cube(ax3, which, t)
save(f, f"{OUT}/fig-neighbours3d.svg")
