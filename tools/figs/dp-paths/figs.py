"""Figures for dp-paths/lec.qmd. Run: python3 -I tools/figs/dp-paths/figs.py (from the repo root).
Figure size in inches = display size / 100 px, fonts >= 12 pt (>= 16 px on the slide)."""
import os, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "dp-paths")
plt.rcParams.update({"font.size": 13, "svg.fonttype": "none", "font.family": "DejaVu Sans"})
ORANGE, BLUE, GREY = "#ff7f0e", "#1f77b4", "#7f7f7f"

C = np.array([[7, 4, 8, 8, 3], [2, 9, 3, 8, 2], [3, 1, 3, 6, 8], [5, 1, 7, 3, 2], [9, 2, 3, 9, 6]])


def dp(C):
    n, m = C.shape
    M = C.astype(float).copy(); B = np.zeros_like(C)
    for i in range(1, n):
        for j in range(m):
            best = None
            for d in (0, -1, 1):
                if 0 <= j + d < m and (best is None or M[i - 1, j + d] < M[i - 1, j + best]):
                    best = d
            M[i, j] += M[i - 1, j + best]; B[i, j] = best
    j = int(np.argmin(M[-1])); path = [j]
    for i in range(n - 1, 0, -1):
        j += B[i, j]; path.append(j)
    return M, path[::-1]


def greedy(C):
    j = int(np.argmin(C[0])); path = [j]
    for i in range(1, C.shape[0]):
        cand = [j + d for d in (0, -1, 1) if 0 <= j + d < C.shape[1]]
        j = min(cand, key=lambda k: C[i, k]); path.append(j)
    return path


def draw_grid(ax, C, fs=15):
    n, m = C.shape
    for i in range(n):
        for j in range(m):
            g = 0.93 - (C[i, j] - 1) / 8 * 0.72  # light = cheap
            ax.add_patch(Rectangle((j, i), 1, 1, fc=str(g), ec="white", lw=2))
            ax.text(j + 0.5, i + 0.5, str(C[i, j]), ha="center", va="center", fontsize=fs, zorder=8,
                    color="black" if g > 0.55 else "white",
                    bbox=dict(boxstyle="circle,pad=0.15", fc=str(g), ec="none"))
    ax.set_xlim(0, m); ax.set_ylim(n, 0); ax.set_aspect("equal"); ax.axis("off")


def draw_path(ax, path, color, ls="-", off=0.0, lw=4):
    xs = [j + 0.5 + off for j in path]; ys = [i + 0.5 for i in range(len(path))]
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=5)


def fig_question():
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    draw_grid(ax, C)
    # the three moves from one cell
    i, j = 1, 2
    ax.add_patch(Rectangle((j, i), 1, 1, fc="none", ec=BLUE, lw=3, zorder=4))
    for d in (-1, 0, 1):
        ax.add_patch(FancyArrowPatch((j + 0.5, i + 0.75), (j + d + 0.5, i + 1.3), arrowstyle="-|>",
                                     mutation_scale=18, color=BLUE, lw=2.5, zorder=6))
    ax.text(-0.15, 0.5, "row 0", ha="right", va="center", fontsize=12, color=GREY)
    ax.text(-0.15, 4.5, "row 4", ha="right", va="center", fontsize=12, color=GREY)
    ax.set_xlim(-0.9, 5)
    fig.tight_layout(pad=0.2); fig.savefig(os.path.join(OUT, "fig-question.svg")); plt.close(fig)


def fig_greedy():
    M, best = dp(C); g = greedy(C)
    cb = sum(C[i, j] for i, j in enumerate(best)); cg = sum(C[i, j] for i, j in enumerate(g))
    assert (cb, cg) == (10, 19), (cb, cg)
    fig, ax = plt.subplots(figsize=(4.4, 4.6))
    draw_grid(ax, C)
    draw_path(ax, g, BLUE, ls="--")
    draw_path(ax, best, ORANGE)
    ax.plot([], [], color=ORANGE, lw=4, label=f"best path: {cb}")
    ax.plot([], [], color=BLUE, lw=4, ls="--", label=f"greedy: {cg}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=13,
              handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(pad=0.2); fig.savefig(os.path.join(OUT, "fig-greedy.svg")); plt.close(fig)


def fig_subpath():
    """Schematic: the best path to the bottom passes p; a cheaper way to p would make it cheaper."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.add_patch(Rectangle((0, 0), 10, 10, fc="#f4f4f4", ec=GREY, lw=1.5))
    ax.text(10.2, 0.4, "top row", va="center", fontsize=12, color=GREY)
    ax.text(10.2, 9.6, "bottom row", va="center", fontsize=12, color=GREY)
    t = np.linspace(0, 1, 50)
    # best path: top (3,0) -> p (5,5) -> bottom (6,10)
    x1 = 3 + 2 * t + 1.2 * np.sin(np.pi * t); y1 = 5 * t
    x2 = 5 + 1 * t - 0.8 * np.sin(np.pi * t); y2 = 5 + 5 * t
    ax.plot(x1, y1, color=ORANGE, lw=4); ax.plot(x2, y2, color=ORANGE, lw=4)
    # a cheaper way to p (hypothetical)
    x3 = 7 - 2 * t - 0.9 * np.sin(np.pi * t); y3 = 5 * t
    ax.plot(x3, y3, color=BLUE, lw=3, ls="--")
    ax.add_patch(Circle((5, 5), 0.32, fc="white", ec="black", lw=2, zorder=5))
    ax.text(5.5, 5.3, "p", fontsize=16, style="italic", va="center")
    ax.text(1.0, 2.6, "best path\nto p", fontsize=12, color=ORANGE, ha="center")
    ax.text(8.6, 2.6, "cheaper\nway to p?", fontsize=12, color=BLUE, ha="center")
    ax.text(7.2, 8.0, "rest of\nthe path", fontsize=12, color=ORANGE, ha="center")
    ax.set_xlim(-0.2, 12.6); ax.set_ylim(10.3, -0.3); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(pad=0.1); fig.savefig(os.path.join(OUT, "fig-subpath.svg")); plt.close(fig)


def fig_trellis():
    """Two-label chain as a path through a 2-state trellis (the segchain example)."""
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    n = 5
    labs = [0, 0, 1, 1, 1]
    for p in range(n):
        for s in (0, 1):
            ax.add_patch(Circle((p, s), 0.2, fc="white", ec="black", lw=1.5, zorder=4))
            if p < n - 1:
                for t in (0, 1):
                    ax.plot([p, p + 1], [s, t], color="#cccccc" if s == t else "#e3e3e3", lw=1.5,
                            ls="-" if s == t else ":", zorder=1)
    for p in range(n - 1):
        ax.plot([p, p + 1], [labs[p], labs[p + 1]], color=ORANGE, lw=4, zorder=3)
    for p in range(n):
        ax.add_patch(Circle((p, labs[p]), 0.2, fc=ORANGE, ec="black", lw=1.5, zorder=5))
        ax.text(p, 1.55, f"pixel {p + 1}", ha="center", fontsize=12)
    ax.text(-0.45, 0, "label 0", ha="right", va="center", fontsize=13)
    ax.text(-0.45, 1, "label 1", ha="right", va="center", fontsize=13)
    ax.annotate("label changes:\npay λ", xy=(1.5, 0.5), xytext=(2.35, -0.55), fontsize=12, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5), ha="left", va="center")
    ax.set_xlim(-1.6, n - 0.6); ax.set_ylim(1.8, -0.95); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(pad=0.1); fig.savefig(os.path.join(OUT, "fig-trellis.svg")); plt.close(fig)


def fig_steps():
    """Three step sets: up to δ = 2 sideways with penalty α|d|; → ↓; → ↓ ↘."""
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.4), gridspec_kw={"width_ratios": [5, 2, 2]})
    def cellgrid(ax, n, m, hi):
        for i in range(n):
            for j in range(m):
                ax.add_patch(Rectangle((j, i), 1, 1, fc="#f0f0f0", ec="white", lw=2))
        ax.add_patch(Rectangle(hi, 1, 1, fc="#ffd9b3", ec=ORANGE, lw=2.5, zorder=3))
        ax.set_xlim(-0.05, m + 0.05); ax.set_ylim(n + 0.05, -0.05); ax.set_aspect("equal"); ax.axis("off")
    def arr(ax, a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=BLUE, lw=2, zorder=5,
                                     shrinkA=0, shrinkB=2))
    ax = axs[0]; cellgrid(ax, 2, 5, (2, 1))
    for d in range(-2, 3):
        arr(ax, (2 + d + 0.5, 0.62), (2.5, 1.45))
    for d in (-2, -1, 1, 2):
        ax.text(2 + d + 0.5, 0.3, f"+{abs(d)}α", ha="center", va="center", fontsize=12, color="black")
    ax.set_title("up to δ = 2 sideways", fontsize=13)
    ax = axs[1]; cellgrid(ax, 2, 2, (1, 1))
    arr(ax, (0.5, 1.5), (1.45, 1.5)); arr(ax, (1.5, 0.5), (1.5, 1.45))
    ax.set_title("right or down", fontsize=13)
    ax = axs[2]; cellgrid(ax, 2, 2, (1, 1))
    arr(ax, (0.5, 1.5), (1.45, 1.5)); arr(ax, (1.5, 0.5), (1.5, 1.45)); arr(ax, (0.5, 0.5), (1.42, 1.42))
    ax.set_title("+ diagonal", fontsize=13)
    fig.tight_layout(pad=0.3, w_pad=2.0); fig.savefig(os.path.join(OUT, "fig-steps.svg")); plt.close(fig)


def fig_chamfer():
    """One forward raster pass of the city-block distance transform vs the true distance."""
    n = 6
    obj = np.ones((n, n), bool); obj[0, 0] = False; obj[4, 4] = False  # two background pixels
    INF = 99
    D = np.where(obj, INF, 0)
    for i in range(n):
        for j in range(n):
            if i > 0: D[i, j] = min(D[i, j], D[i - 1, j] + 1)
            if j > 0: D[i, j] = min(D[i, j], D[i, j - 1] + 1)
    true = np.fromfunction(lambda i, j: np.minimum(i + j, abs(i - 4) + abs(j - 4)), (n, n), dtype=int)
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.5))
    for ax, A, title in ((axs[0], D, "one pass (→, then ↓)"), (axs[1], true, "true distance")):
        for i in range(n):
            for j in range(n):
                wrong = A is D and A[i, j] != true[i, j]
                fc = "#222222" if not obj[i, j] else ("#ffd9b3" if wrong else "#f0f0f0")
                ax.add_patch(Rectangle((j, i), 1, 1, fc=fc, ec="white", lw=2))
                s = "∞" if A[i, j] >= INF else str(A[i, j])
                ax.text(j + 0.5, i + 0.5, s, ha="center", va="center", fontsize=12,
                        color="white" if not obj[i, j] else "black")
        ax.set_xlim(0, n); ax.set_ylim(n, 0); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=13)
    fig.tight_layout(pad=0.3, w_pad=2); fig.savefig(os.path.join(OUT, "fig-chamfer.svg")); plt.close(fig)
    return D, true


def fig_graphs():
    """Chain, tree, grid with loops."""
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.6))
    def node(ax, x, y): ax.add_patch(Circle((x, y), 0.17, fc="white", ec="black", lw=1.5, zorder=4))
    def edge(ax, a, b, c="black"): ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=2, zorder=2)
    ax = axs[0]
    pts = [(k * 0.75, 1) for k in range(4)]
    for a, b in zip(pts, pts[1:]): edge(ax, a, b)
    for p in pts: node(ax, *p)
    ax.set_title("chain: DP exact", fontsize=13)
    ax = axs[1]
    T = {(1.5, 0): [(0.75, 1), (2.25, 1)], (0.75, 1): [(0.3, 2), (1.2, 2)], (2.25, 1): [(2.25, 2)]}
    for a, bs in T.items():
        for b in bs: edge(ax, a, b)
    for p in [(1.5, 0), (0.75, 1), (2.25, 1), (0.3, 2), (1.2, 2), (2.25, 2)]: node(ax, *p)
    ax.set_title("tree: DP exact", fontsize=13)
    ax = axs[2]
    G = [(x * 0.9, y * 0.9) for y in range(3) for x in range(3)]
    for (x, y) in G:
        if x + 0.9 < 2: edge(ax, (x, y), (x + 0.9, y))
        if y + 0.9 < 2: edge(ax, (x, y), (x, y + 0.9))
    for e in [((0, 0), (0.9, 0)), ((0.9, 0), (0.9, 0.9)), ((0.9, 0.9), (0, 0.9)), ((0, 0.9), (0, 0))]:
        edge(ax, *e, c=ORANGE)
    for p in G: node(ax, *p)
    ax.set_title("grid: loops", fontsize=13)
    for ax in axs:
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-0.3, 2.55); ax.set_ylim(2.3, -0.3)
    fig.tight_layout(pad=0.3, w_pad=1.0); fig.savefig(os.path.join(OUT, "fig-graphs.svg")); plt.close(fig)


if __name__ == "__main__":
    fig_question(); fig_greedy(); fig_subpath(); fig_trellis(); fig_steps()
    D, true = fig_chamfer(); fig_graphs()
    M, best = dp(C)
    print("M last row", M[-1], "best", best, "greedy", greedy(C))
    print("pass1\n", D, "\ntrue\n", true)
