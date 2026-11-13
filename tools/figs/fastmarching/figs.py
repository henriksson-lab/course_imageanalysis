"""Figures for fastmarching/lec.qmd. Run from the repo root after data.ts (see there):
   python3 -I tools/figs/fastmarching/figs.py <tmp>/fm-data.json
Figure size in inches = display size / 100 px, fonts >= 12 pt."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "fastmarching")
plt.rcParams.update({"font.size": 13, "svg.fonttype": "none", "font.family": "DejaVu Sans"})
ORANGE, BLUE, GREY, RED = "#ff7f0e", "#1f77b4", "#7f7f7f", "#d62728"


def eikonal(a, b, P=1.0):
    if not abs(a - b) < P:
        return min(a, b) + P
    return (a + b + np.sqrt(2 * P * P - (a - b) ** 2)) / 2


def fig_arrival(path):
    d = json.load(open(path))
    W, H = d["W"], d["H"]
    T = np.array(d["T"]).reshape(H, W)
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.imshow(T, cmap="viridis", origin="upper")
    ax.contour(T, levels=np.arange(10, T.max(), 10), colors="white", linewidths=1.2)
    c = d["disc"]
    ax.add_patch(Circle((c["x"], c["y"]), c["r"], fc="none", ec=ORANGE, lw=2.5, ls="--"))
    ax.plot(*d["seed"], "o", ms=9, mfc=RED, mec="white", mew=1.5)
    ax.text(c["x"], c["y"] + c["r"] + 10, f"P = {d['slow']:g} in the disc, 1 outside", ha="center",
            va="center", fontsize=15, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.6))
    ax.axis("off")
    fig.tight_layout(pad=0.1); fig.savefig(os.path.join(OUT, "fig-arrival.png"), dpi=200); plt.close(fig)


def fig_update_idea():
    """Left: Dijkstra, the value comes along one grid edge. Right: fast marching, a straight front
    through both known neighbours."""
    a, b = 2.0, 2.5
    T = eikonal(a, b)
    n = np.array([T - a, T - b]); n = n / np.linalg.norm(n)  # unit normal (x right, y down)
    fig, axs = plt.subplots(1, 2, figsize=(5.8, 3.4))
    for ax, mode in zip(axs, ("dijkstra", "fm")):
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc="#f2f2f2", ec="white", lw=3))
        # pixel centres: left neighbour (0,1), upper neighbour (1,0), v (1,1)
        for (x, y), lab, col in (((0, 1), "a", "black"), ((1, 0), "b", "black"), ((1, 1), "v", ORANGE)):
            ax.add_patch(Circle((x, y), 0.12, fc=col if lab == "v" else "white", ec="black", lw=1.5, zorder=5))
            ax.text(x - 0.2, y - 0.2, lab, fontsize=15, style="italic", ha="center", va="center")
        if mode == "dijkstra":
            ax.add_patch(FancyArrowPatch((0, 1), (1, 1), arrowstyle="-|>", mutation_scale=18, color=BLUE, lw=2.5,
                                         shrinkA=8, shrinkB=8, zorder=4))
            ax.set_title("Dijkstra: along one edge", fontsize=13)
        else:
            t = np.array([-n[1], n[0]])
            for p0 in (np.array([0, 1.0]), np.array([1.0, 0])):
                ax.plot(*np.c_[p0 - 1.4 * t, p0 + 1.4 * t], color=BLUE, lw=1.8)
            pv = np.array([1.0, 1.0])
            ax.plot(*np.c_[pv - 1.4 * t, pv + 1.4 * t], color=BLUE, lw=1.8, ls="--")
            ax.add_patch(FancyArrowPatch((0.35, 0.55), (0.35 + 0.55 * n[0], 0.55 + 0.55 * n[1]), arrowstyle="-|>",
                                         mutation_scale=16, color=BLUE, lw=2, zorder=4))
            ax.set_title("fast marching: a straight front", fontsize=13)
        ax.set_xlim(-0.55, 1.55); ax.set_ylim(1.55, -0.55); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(pad=0.3, w_pad=1.5); fig.savefig(os.path.join(OUT, "fig-update.svg")); plt.close(fig)


def fig_byhand():
    """The by-hand example: a = 2 (left), b = 2.5 (above), P = 1 -> T = 2.911 (Dijkstra: 3)."""
    a, b = 2.0, 2.5
    T = eikonal(a, b)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    vals = {(0, 1): f"{a:g}", (1, 0): f"{b:g}", (1, 1): f"{T:.3f}"}
    for x in range(3):
        for y in range(3):
            fc = "#9ecae1" if (x, y) in ((0, 1), (1, 0)) else ("#ffd9b3" if (x, y) == (1, 1) else "#f2f2f2")
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc=fc, ec="white", lw=3))
            if (x, y) in vals:
                ax.text(x, y, vals[(x, y)], ha="center", va="center", fontsize=17,
                        fontweight="bold" if (x, y) == (1, 1) else "normal")
    ax.add_patch(Rectangle((0.5, 0.5), 1, 1, fc="none", ec=ORANGE, lw=3))
    ax.text(1, 1.32, "Dijkstra: 3", ha="center", va="center", fontsize=12, color=GREY)
    ax.text(-0.5, 2.75, "P = 1", fontsize=13, ha="left", va="center")
    ax.set_xlim(-0.55, 2.55); ax.set_ylim(2.95, -0.55); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(pad=0.1); fig.savefig(os.path.join(OUT, "fig-byhand.svg")); plt.close(fig)
    return T


if __name__ == "__main__":
    fig_arrival(sys.argv[1]); fig_update_idea()
    print("T", fig_byhand(), "T11", eikonal(1, 1))
