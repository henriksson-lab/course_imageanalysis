"""Static figures for the feature-matching/ deck.

Run from the repository root (after tools/figs/keypoints/export.ts, see its header):
    python3 -I tools/figs/feature-matching/figs.py <data-dir>
Writes feature-matching/fig-*.svg / *.png. Matching from demos/src/lib/featurematch.ts.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "keypoints"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import kplib
from kplib import load, show, save, side_by_side, BLUE, ORANGE, GREEN, RED, PURPLE, CYAN, YELLOW, GREY, FONT

kplib.DATA = sys.argv[1]
OUT = "feature-matching"


def draw_matches(ax, d, ms, title, lw=1.0):
    im, off = side_by_side(d["v1"], d["v2"])
    show(ax, im)
    nc = 0
    for m in ms:
        i, j, ok = m[0], m[1], m[-1]
        a, b = d["k1"][i], d["k2"][j]
        ax.plot([a["x"], b["x"] + off], [a["y"], b["y"]], color=GREEN if ok else RED, lw=lw, alpha=0.8)
        nc += ok
    ax.set_title(f"{title}: {len(ms)} matches, {nc} correct ({100 * nc / max(1, len(ms)):.0f} %)", fontsize=FONT - 3)
    return nc


def fig_allnn():
    d = load("matching")
    fig, ax = plt.subplots(figsize=(10, 5.0))
    draw_matches(ax, d, [[m[0], m[1], m[4]] for m in d["nn"]], "nearest neighbour for every point", lw=0.8)
    save(fig, OUT, "fig-allnn.png", dpi=100)


def fig_stages():
    d = load("matching")
    fig, axs = plt.subplots(1, 3, figsize=(19, 3.9))
    draw_matches(axs[0], d, [[m[0], m[1], m[4]] for m in d["nn"]], "nearest only", lw=0.7)
    draw_matches(axs[1], d, d["ratio"], "ratio test 0.8")
    draw_matches(axs[2], d, d["mutual"], "ratio + mutual")
    save(fig, OUT, "fig-stages.png", dpi=90)


def fig_hists():
    d = load("matching")
    nn = np.array(d["nn"])
    ok = nn[:, 4] > 0.5
    d1 = nn[:, 2]; r = nn[:, 2] / nn[:, 3]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.linspace(0, 1.2, 31)
    ax.hist(d1[ok], bins=bins, color=GREEN, alpha=0.65, label=f"correct ({ok.sum()})")
    ax.hist(d1[~ok], bins=bins, color=RED, alpha=0.55, label=f"wrong ({(~ok).sum()})")
    ax.set_xlabel("distance to the nearest, d1"); ax.set_ylabel("matches"); ax.legend(frameon=False)
    save(fig, OUT, "fig-d1hist.svg")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.linspace(0.2, 1.0, 33)
    ax.hist(r[ok], bins=bins, color=GREEN, alpha=0.65, label="correct")
    ax.hist(r[~ok], bins=bins, color=RED, alpha=0.55, label="wrong")
    ax.axvline(0.8, color="k", ls="--", lw=2); ax.text(0.79, ax.get_ylim()[1] * 0.92, "0.8", ha="right")
    ax.set_xlabel("ratio d1 / d2"); ax.set_ylabel("matches"); ax.legend(frameon=False, loc="upper left")
    save(fig, OUT, "fig-ratiohist.svg")
    print(f"  NN: {ok.sum()} correct of {len(ok)}; ratio<0.8 keeps {100 * (r[ok] < 0.8).mean():.0f}% correct, {100 * (r[~ok] < 0.8).mean():.0f}% wrong")


def fig_pr():
    d = load("matching")
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for key, col, lab in [("pr", BLUE, "ratio test"), ("prMut", PURPLE, "ratio + mutual")]:
        ax.plot([p["recall"] for p in d[key]], [p["precision"] for p in d[key]], color=col, lw=2.6, label=lab)
    p8 = min(d["pr"], key=lambda p: abs(p["t"] - 0.8))
    ax.plot(p8["recall"], p8["precision"], "o", color="k", ms=9); ax.annotate("t = 0.8", (p8["recall"], p8["precision"]), xytext=(-70, -30), textcoords="offset points", arrowprops=dict(arrowstyle="-"))
    pl = d["pr"][-1]
    ax.plot(pl["recall"], pl["precision"], "s", color=ORANGE, ms=9, label="nearest only")
    ax.set_xlim(0.4, 1); ax.set_ylim(0, 1.05); ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.legend(frameon=False, loc="lower left")
    save(fig, OUT, "fig-pr.svg")


def fig_mutual():
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    A = [(0, 3), (0, 2), (0, 1), (0, 0)]; B = [(3, 3.3), (3, 2.3), (3, 1.3), (3, 0.3)]
    for k, p in enumerate(A):
        ax.add_patch(Circle(p, 0.22, fc=BLUE)); ax.text(p[0] - 0.4, p[1], f"{k + 1}", ha="right", va="center")
    for k, p in enumerate(B):
        ax.add_patch(Circle(p, 0.22, fc=ORANGE)); ax.text(p[0] + 0.4, p[1], chr(97 + k), va="center")
    def arr(p, q, col, rad):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=18, color=col, lw=2, connectionstyle=f"arc3,rad={rad}", shrinkA=10, shrinkB=10))
    arr(A[0], B[0], GREEN, 0.12); arr(B[0], A[0], GREEN, 0.12)
    arr(A[1], B[1], GREEN, 0.12); arr(B[1], A[1], GREEN, 0.12)
    arr(A[2], B[1], RED, 0.0); arr(A[3], B[1], RED, -0.1)
    arr(B[3], A[3], GREY, 0.12)
    ax.text(1.5, -0.75, "green: nearest both ways (kept)\nred: 3 and 4 also pick b, but b picks 2", ha="center", fontsize=FONT - 3)
    ax.set_xlim(-1.8, 4.8); ax.set_ylim(-1.5, 4.1); ax.set_aspect("equal"); ax.axis("off")
    ax.text(0, 3.65, "image 1", ha="center"); ax.text(3, 3.85, "image 2", ha="center")
    save(fig, OUT, "fig-mutual.svg")


def fig_grid():
    d = load("gridmatch")
    fig, ax = plt.subplots(figsize=(10, 5.0))
    draw_matches(ax, d, d["m"], "repetitive wells, ratio 0.8")
    save(fig, OUT, "fig-grid.png", dpi=100)


def fig_kdtree():
    rng = np.random.default_rng(3)
    P = rng.uniform(0, 10, (40, 2)); q = np.array([6.3, 4.1])
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    def split(idx, box, depth):
        if len(idx) <= 4:
            return
        x0, x1, y0, y1 = box
        dim = 0 if (x1 - x0) >= (y1 - y0) else 1
        v = np.median(P[idx, dim])
        if dim == 0:
            ax.plot([v, v], [y0, y1], color=GREY, lw=2.5 - 0.4 * depth)
            split(idx[P[idx, 0] < v], (x0, v, y0, y1), depth + 1); split(idx[P[idx, 0] >= v], (v, x1, y0, y1), depth + 1)
        else:
            ax.plot([x0, x1], [v, v], color=GREY, lw=2.5 - 0.4 * depth)
            split(idx[P[idx, 1] < v], (x0, x1, y0, v), depth + 1); split(idx[P[idx, 1] >= v], (x0, x1, v, y1), depth + 1)
    split(np.arange(40), (0, 10, 0, 10), 0)
    ax.plot(P[:, 0], P[:, 1], "o", color=BLUE, ms=6)
    j = np.argmin(np.hypot(*(P - q).T)); r = np.hypot(*(P[j] - q))
    ax.plot(*q, "*", color=RED, ms=16); ax.add_patch(Circle(q, r, fill=False, ec=RED, ls="--", lw=1.5))
    ax.plot(*P[j], "o", color=RED, ms=9)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("k-d tree: split at medians,\nsearch the query's cell first", fontsize=FONT - 2)
    save(fig, OUT, "fig-kdtree.svg")


if __name__ == "__main__":
    for f in [fig_allnn, fig_stages, fig_hists, fig_pr, fig_mutual, fig_grid, fig_kdtree]:
        f()
