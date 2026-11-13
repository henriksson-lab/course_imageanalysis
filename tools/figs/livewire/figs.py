"""Figures for livewire/lec.qmd. Run from the repo root after data.ts (see there):
   python3 -I tools/figs/livewire/figs.py <tmp>/lw-data.json
Figure size in inches = display size / 100 px, fonts >= 12 pt."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "livewire")
plt.rcParams.update({"font.size": 13, "svg.fonttype": "none", "font.family": "DejaVu Sans"})
ORANGE, BLUE, GREY = "#ff7f0e", "#1f77b4", "#7f7f7f"
FINAL = "#9ecae1"  # light blue: final (as in the dpgrid demo)
d = json.load(open(sys.argv[1]))


def grey(c):
    return 0.93 - (c - 1) / 8 * 0.72  # light = cheap, as in the dp-paths figures


def draw_costs(ax, C, rows, cols, fs=14, fill=None, show=True):
    for i in range(rows):
        for j in range(cols):
            c = C[i * cols + j]; g = grey(c)
            fc = fill(i * cols + j) if fill else None
            ax.add_patch(Rectangle((j, i), 1, 1, fc=fc or str(g), ec="white", lw=2))
            if show:
                ax.text(j + 0.5, i + 0.5, str(c), ha="center", va="center", fontsize=fs, zorder=8,
                        color="black" if (fc or g > 0.55) else "white")
    ax.set_xlim(-0.05, cols + 0.05); ax.set_ylim(rows + 0.05, -0.05); ax.set_aspect("equal"); ax.axis("off")


def path_xy(path, cols):
    return [p % cols + 0.5 for p in path], [p // cols + 0.5 for p in path]


def fig_detour():
    t = d["detour"]; rows, cols, C = t["rows"], t["cols"], t["C"]
    fig, ax = plt.subplots(figsize=(4.4, 5.1))
    draw_costs(ax, C, rows, cols, fs=13, show=False)
    x, y = path_xy(t["djPath"], cols); ax.plot(x, y, color=ORANGE, lw=5, solid_capstyle="round", zorder=5)
    x, y = path_xy(t["dpPath"], cols); ax.plot([v + 0.12 for v in x], y, color=BLUE, lw=3.5, ls="--", zorder=6)
    ax.plot([], [], color=ORANGE, lw=5, label=f"following the channel: {t['djCost']:.0f}")
    ax.plot([], [], color=BLUE, lw=3.5, ls="--", label=f"best monotone path: {t['dpCost']:.0f}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1, frameon=False, fontsize=13)
    fig.tight_layout(pad=0.2); fig.savefig(os.path.join(OUT, "fig-detour.svg")); plt.close(fig)


def fig_graph():
    """3x3 pixels as a graph: nodes at pixel centres, edges to 4 neighbours; moves out of the
    centre are labelled with the cost of the pixel entered."""
    C = [3, 1, 4, 2, 5, 9, 6, 2, 1]
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 3.3))
    draw_costs(axs[0], C, 3, 3, fs=15)
    axs[0].set_title("pixels and costs", fontsize=13)
    ax = axs[1]
    pos = {(i, j): (j + 0.5, i + 0.5) for i in range(3) for j in range(3)}
    for (i, j), (x, y) in pos.items():
        for di, dj in ((0, 1), (1, 0)):
            if (i + di, j + dj) in pos:
                x2, y2 = pos[(i + di, j + dj)]
                ax.plot([x, x2], [y, y2], color="#bbbbbb", lw=2, zorder=1)
    for (i, j), (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.17, fc="white", ec="black", lw=1.5, zorder=4))
    cx, cy = pos[(1, 1)]
    ax.add_patch(Circle((cx, cy), 0.17, fc=ORANGE, ec="black", lw=1.5, zorder=5))
    for (i, j) in ((0, 1), (1, 0), (1, 2), (2, 1)):
        x, y = pos[(i, j)]
        ax.add_patch(FancyArrowPatch((cx, cy), (x, y), arrowstyle="-|>", mutation_scale=16, color=BLUE, lw=2,
                                     shrinkA=9, shrinkB=9, zorder=3))
        mx, my = (cx + x) / 2, (cy + y) / 2
        ox, oy = (0.17, 0) if x == cx else (0, -0.17)
        ax.text(mx + ox, my + oy, str(C[i * 3 + j]), fontsize=14, color=BLUE, ha="center", va="center",
                fontweight="bold")
    ax.set_xlim(-0.05, 3.05); ax.set_ylim(3.05, -0.05); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("graph: weight = cost entered", fontsize=13)
    fig.tight_layout(pad=0.3, w_pad=1.5); fig.savefig(os.path.join(OUT, "fig-graph.svg")); plt.close(fig)


def fig_front():
    """The by-hand grid after 6 steps: final pixels (order and d), tentative ones (d), unreached (∞)."""
    t = d["exercise"]; rows, cols, C = t["rows"], t["cols"], t["C"]
    k = 6
    ev = t["events"][k - 1]
    final = {e["pop"]: (n + 1, e["d"]) for n, e in enumerate(t["events"][:k])}
    tent = {p: v for p, v in ev["queue"]}
    nxt = ev["queue"][0][0]
    fig, axs = plt.subplots(1, 2, figsize=(4.6, 2.65))
    draw_costs(axs[0], C, rows, cols, fs=15); axs[0].set_title("costs", fontsize=13)
    ax = axs[1]
    def fill(p):
        return FINAL if p in final else ("#ffd9b3" if p in tent else "#f4f4f4")
    draw_costs(ax, C, rows, cols, fill=fill, show=False)
    for p in range(rows * cols):
        i, j = divmod(p, cols)
        if p in final:
            n, v = final[p]
            ax.text(j + 0.5, i + 0.55, f"{v:.0f}", ha="center", va="center", fontsize=15, fontweight="bold")
            ax.text(j + 0.12, i + 0.2, str(n), ha="left", va="center", fontsize=11, color=GREY)
        elif p in tent:
            ax.text(j + 0.5, i + 0.55, f"{tent[p]:.0f}", ha="center", va="center", fontsize=15)
        else:
            ax.text(j + 0.5, i + 0.55, "∞", ha="center", va="center", fontsize=15, color=GREY)
    i, j = divmod(nxt, cols)
    ax.add_patch(Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, fc="none", ec=ORANGE, lw=3, zorder=9))
    ax.set_title(f"d after {k} steps", fontsize=13)
    fig.tight_layout(pad=0.3, w_pad=1.5); fig.savefig(os.path.join(OUT, "fig-front.svg")); plt.close(fig)
    return [(p, final[p]) for p in final], tent, nxt


def fig_tree():
    t = d["wall"]; rows, cols, C = t["rows"], t["cols"], t["C"]
    fig, ax = plt.subplots(figsize=(4.0, 4.3))
    draw_costs(ax, C, rows, cols, show=False)
    for q, p in enumerate(t["pred"]):
        if p < 0: continue
        i, j = divmod(q, cols); i2, j2 = divmod(p, cols)
        ax.add_patch(FancyArrowPatch((j + 0.5, i + 0.5), (j2 + 0.5, i2 + 0.5), arrowstyle="-|>", mutation_scale=11,
                                     color="#555555", lw=1.3, shrinkA=8, shrinkB=8, zorder=4))
    x, y = path_xy(t["path"], cols)
    ax.plot(x, y, color=ORANGE, lw=4, alpha=0.85, zorder=3, solid_capstyle="round")
    ax.add_patch(Circle((0.5, 0.5), 0.2, fc="#d62728", ec="white", lw=1.5, zorder=6))
    tq = t["target"]; i, j = divmod(tq, cols)
    ax.add_patch(Rectangle((j + 0.3, i + 0.3), 0.4, 0.4, fc="white", ec="black", lw=2, zorder=6))
    ax.plot([], [], color=ORANGE, lw=4, label=f"backtracked path: {t['dist'][tq]:.0f}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=13)
    fig.tight_layout(pad=0.2); fig.savefig(os.path.join(OUT, "fig-tree.svg")); plt.close(fig)
    return t["dist"][tq]


def fig_negative():
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    P = {"s": (0, 0), "a": (2.4, 0), "b": (1.2, -1.5)}
    for n, (x, y) in P.items():
        ax.add_patch(Circle((x, y), 0.3, fc="white", ec="black", lw=1.8, zorder=4))
        ax.text(x, y, n, ha="center", va="center", fontsize=15, style="italic", zorder=5)
    def e(a, b, w, col="black", off=(0, 0.2)):
        (x1, y1), (x2, y2) = P[a], P[b]
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, color=col, lw=2,
                                     shrinkA=14, shrinkB=14, zorder=3))
        ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], w, ha="center", va="center", fontsize=14, color=col)
    e("s", "a", "2"); e("s", "b", "3", off=(-0.3, -0.05)); e("b", "a", "−2", col="#d62728", off=(0.4, -0.05))
    ax.text(2.4, 0.55, "final at 2,\nbut s→b→a = 1", ha="center", va="bottom", fontsize=12, color="#d62728")
    ax.set_xlim(-0.5, 3.4); ax.set_ylim(-1.95, 1.3); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(pad=0.1); fig.savefig(os.path.join(OUT, "fig-negative.svg")); plt.close(fig)


if __name__ == "__main__":
    fig_detour(); fig_graph()
    print("front", fig_front())
    print("tree target", fig_tree())
    fig_negative()
