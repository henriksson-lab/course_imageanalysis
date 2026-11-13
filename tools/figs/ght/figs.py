"""Figures for ght/lec.qmd.

Run from the repository root:
    tools/figs/hough-lines/capture.sh /tmp/houghstills      # demo screenshots (needs docs/demos/dist)
    python3 -I tools/figs/ght/figs.py /tmp/houghstills
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../hough-lines"))
from stills import plt, panel, grid_of_panels, BLUE, ORANGE, GREEN, RED, PURPLE, GREY  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = os.path.join(REPO, "ght")
SH = sys.argv[1] if len(sys.argv) > 1 else None


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200 if name.endswith(".png") else None,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ---------------------------------------------------------------- a shape without a formula
def fig_shape():
    t = np.linspace(0, 2 * np.pi, 400)
    r = 1 + 0.32 * np.cos(2 * t) + 0.18 * np.sin(3 * t)
    x, y = r * np.cos(t), r * np.sin(t)
    ref = np.array([0.05, -0.05])
    dx, dy = np.gradient(x), np.gradient(y)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.set_aspect("equal")
    ax.fill(x, y, color="#e8e8e8")
    ax.plot(x, y, color="k", lw=1.5)
    ax.plot(*ref, "+", color="k", ms=18, mew=3)
    ax.text(ref[0] - 0.1, ref[1] - 0.22, "reference", fontsize=13, ha="right", va="top")
    cols = [BLUE, ORANGE, GREEN, PURPLE]
    for k, c in zip([20, 95, 180, 300], cols):
        p = np.array([x[k], y[k]])
        n = np.array([dy[k], -dx[k]])
        n /= np.linalg.norm(n)
        if np.dot(n, p - ref) > 0:          # inward normal = gradient of a bright shape
            n = -n
        ax.annotate("", xy=p + 0.35 * n, xytext=p,
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.5, mutation_scale=15))
        ax.annotate("", xy=ref, xytext=p,
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4, ls="--", mutation_scale=12,
                                    shrinkA=0, shrinkB=4))
        ax.plot(*p, "o", color=c, ms=8, zorder=5)
    ax.text(1.05, -1.25, "solid: gradient\ndashed: vector v", fontsize=13, ha="center")
    ax.axis("off")
    fig.tight_layout()
    save(fig, "fig-shape.svg")


# ---------------------------------------------------------------- the 4 x 4 square example
def square_example():
    pts = {"down": [(1, 0), (2, 0), (3, 0)], "up": [(1, 4), (2, 4), (3, 4)],
           "left": [(0, 1), (0, 2), (0, 3)], "right": [(4, 1), (4, 2), (4, 3)]}
    ref = np.array([2, 2])
    table = {d: [tuple(ref - np.array(p)) for p in ps] for d, ps in pts.items()}
    shift = np.array([3, 2])                         # translated copy in a 10 x 8 scene
    H = np.zeros((9, 11), int)
    for d, ps in pts.items():
        for p in ps:
            for v in table[d]:
                q = np.array(p) + shift + np.array(v)
                H[q[1], q[0]] += 1
    return pts, ref, table, shift, H


def fig_square():
    pts, ref, table, shift, H = square_example()
    cols = {"down": BLUE, "right": RED, "up": GREEN, "left": ORANGE}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.3, 5.6))
    a1.set_aspect("equal")
    a1.add_patch(plt.Rectangle((0, 0), 4, 4, fc="#e8e8e8", ec="k", lw=1.5))
    for d, ps in pts.items():
        for p in ps:
            a1.annotate("", xy=ref, xytext=p, arrowprops=dict(arrowstyle="-|>", color=cols[d], lw=1.3,
                                                             shrinkA=0, shrinkB=3, mutation_scale=10))
            a1.plot(*p, "o", color=cols[d], ms=8)
    a1.plot(*ref, "+", color="k", ms=14, mew=3)
    a1.set_xlim(-0.5, 4.5)
    a1.set_ylim(-0.5, 4.5)
    a1.set_xticks(range(5))
    a1.set_yticks(range(5))
    a1.set_title("R-table: 4 bins × 3", fontsize=14)
    ys, xs = np.nonzero(H)
    Hc = H[ys.min() - 1:ys.max() + 2, xs.min() - 1:xs.max() + 2]
    a2.imshow(Hc, origin="lower", cmap="Blues", vmin=0, vmax=14, interpolation="nearest")
    for (yy, xx), v in np.ndenumerate(Hc):
        if v:
            a2.text(xx, yy, str(v), ha="center", va="center", fontsize=13,
                    color="white" if v > 8 else "k", fontweight="bold" if v > 8 else "normal")
    a2.set_title(f"votes of a shifted copy: {H.sum()}", fontsize=14)
    a2.set_xticks([])
    a2.set_yticks([])
    for s in a2.spines.values():
        s.set_visible(True)
        s.set_color("#999")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig-squarevotes.svg")
    srt = np.sort(H.ravel())[::-1]
    return H.sum(), srt[0], srt[1]


def fig_stills():
    grid_of_panels([panel(SH, "ght_circle", 0), panel(SH, "ght_circle", 1)],
                   ["R-table: all vectors r = 18", "both discs found"],
                   os.path.join(OUT, "fig-circletable.png"), ncols=2, w=4.6)
    grid_of_panels([panel(SH, "ght_rotmiss", 1), panel(SH, "ght_rotmiss", 2)],
                   ["rotated key: missed", "no peak for it"],
                   os.path.join(OUT, "fig-rotmiss.png"), ncols=2, w=4.6)
    grid_of_panels([panel(SH, "ght_occluded", 1), panel(SH, "ght_similar", 1)],
                   ["occluded: 0.58, found", "similar: 0.31, false +"],
                   os.path.join(OUT, "fig-occluded.png"), ncols=2, w=4.6)


if __name__ == "__main__":
    fig_shape()
    print("square example: total votes, max, second:", fig_square())
    if SH:
        fig_stills()
