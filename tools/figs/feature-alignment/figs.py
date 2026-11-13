"""Static figures for the feature-alignment/ deck.

Run from the repository root (after tools/figs/keypoints/export.ts, see its header):
    python3 -I tools/figs/feature-alignment/figs.py <data-dir>
Writes feature-alignment/fig-*.svg / *.png. Fits and RANSAC from demos/src/lib/transforms2d.ts.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "keypoints"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
import kplib
from kplib import load, show, save, arr, apply, side_by_side, BLUE, ORANGE, GREEN, RED, PURPLE, CYAN, YELLOW, GREY, FONT

kplib.DATA = sys.argv[1]
OUT = "feature-alignment"


def checker(a, b, s=32):
    out = a.copy()
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    m = ((xx // s + yy // s) % 2 == 1) & np.isfinite(b)
    out[m] = b[m]
    return out


def fig_overlay():
    d = load("align")
    im, off = side_by_side(d["v1"], d["v2"])
    fig = plt.figure(figsize=(14, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.05, 1, 1])
    ax = fig.add_subplot(gs[0]); show(ax, im)
    inl = set(d["inl"])
    for k, (p, q) in enumerate(zip(d["src"], d["dst"])):
        ax.plot([p["x"], q["x"] + off], [p["y"], q["y"]], color=GREEN if k in inl else RED, lw=1.0, alpha=0.8)
    ax.set_title(f"{len(d['src'])} matches, {len(inl)} RANSAC inliers", fontsize=FONT - 2)
    a = arr(d["v1"])
    for g, key, t in [(gs[1], "w2l", "least squares on all"), (gs[2], "w2r", "RANSAC + refit")]:
        x = fig.add_subplot(g); show(x, checker(a, arr(d[key]).astype(float)), lo=0, hi=255); x.set_title(t, fontsize=FONT - 2)
    save(fig, OUT, "fig-overlay.png", dpi=95)


def fig_zoo():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], float)
    grid = [np.c_[np.full(11, t), np.linspace(0, 1, 11)] for t in np.linspace(0, 1, 5)] + [np.c_[np.linspace(0, 1, 11), np.full(11, t)] for t in np.linspace(0, 1, 5)]
    def H(M):
        def f(P):
            Q = np.c_[P, np.ones(len(P))] @ np.array(M).reshape(3, 3).T
            return Q[:, :2] / Q[:, 2:]
        return f
    c, s = np.cos(0.4), np.sin(0.4)
    models = [("translation\n2 numbers, 1 pair", [1, 0, 0.4, 0, 1, 0.2, 0, 0, 1]),
              ("rigid\n3 numbers, 2 pairs", [c, -s, 0.5, s, c, 0, 0, 0, 1]),
              ("similarity\n4 numbers, 2 pairs", [0.7 * c, -0.7 * s, 0.6, 0.7 * s, 0.7 * c, 0.1, 0, 0, 1]),
              ("affine\n6 numbers, 3 pairs", [1.1, 0.5, 0.1, 0.1, 0.7, 0.2, 0, 0, 1]),
              ("homography\n8 numbers, 4 pairs", [1.0, 0.2, 0.1, 0.0, 0.9, 0.2, 0.6, 0.3, 1])]
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.6))
    for ax, (t, M) in zip(axs, models):
        for g in grid:
            ax.plot(g[:, 0], g[:, 1], color="#bbb", lw=1)
            w = H(M)(g); ax.plot(w[:, 0], w[:, 1], color=BLUE, lw=1.6)
        ax.set_title(t, fontsize=FONT - 3); ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.2, 1.9); ax.set_ylim(-0.2, 1.7)
    save(fig, OUT, "fig-zoo.svg")


def fig_ls_outliers():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 10, 20); A = np.array([[1.1, 0.3], [-0.2, 0.9]]); t = np.array([1, 2])
    P = np.c_[x, rng.uniform(0, 10, 20)]; Q = P @ A.T + t + rng.normal(0, 0.15, (20, 2))
    Q[:3] += [[6, -5], [-5, 6], [7, 4]]
    M = np.c_[P, np.ones(20)]
    sol = np.linalg.lstsq(M, Q, rcond=None)[0]
    good = np.linalg.lstsq(M[3:], Q[3:], rcond=None)[0]
    fig, ax = plt.subplots(figsize=(6.4, 5))
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    for S, col, lab in [(good, GREEN, "fit without the 3 outliers"), (sol, RED, "least squares on all 20")]:
        w = np.c_[sq, np.ones(4)] @ S
        ax.add_patch(Polygon(w, fill=False, ec=col, lw=2.5, label=lab))
    ax.plot(Q[3:, 0], Q[3:, 1], "o", color=BLUE, ms=6, label="matched points")
    ax.plot(Q[:3, 0], Q[:3, 1], "x", color=RED, ms=11, mew=3, label="3 wrong matches")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=False, fontsize=FONT - 4, loc="upper left", bbox_to_anchor=(1, 1))
    ax.autoscale_view()
    save(fig, OUT, "fig-ls-outliers.svg")


def fig_camera():
    d = load("camera")
    a = arr(d["v1"])
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.1))
    show(axs[0], d["v1"]); axs[0].set_title("view 1", fontsize=FONT - 2)
    show(axs[1], d["v2"]); axs[1].set_title("view 2: camera turned\nabout its centre", fontsize=FONT - 3)
    for ax, key, t, n in [(axs[2], "wa", "affine", d["na"]), (axs[3], "wh", "homography", d["nh"])]:
        b = arr(d[key]); diff = np.where(np.isfinite(b), 128 + 1.5 * (a - b), 128)
        show(ax, diff); ax.set_title(f"difference, {t}\n{n} of {d['n']} inliers", fontsize=FONT - 3)
    save(fig, OUT, "fig-camera.png", dpi=95)


def fig_mosaic():
    d = load("mosaic")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, k, t in zip(axs, ["none", "feather", "multiband"], ["no blending: seams", "feathering", "multi-band"]):
        a = arr(d["blends"][k]); a = np.where(np.isfinite(a), a, 255)
        show(ax, a); ax.set_title(t)
        if k == "none":
            for c, col in zip(d["tiles"], [BLUE, RED, GREEN, ORANGE]):
                ax.add_patch(Polygon(c, fill=False, ec=col, lw=1.6))
    save(fig, OUT, "fig-mosaic.png", dpi=85)


def fig_parallax():
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.add_patch(Rectangle((-3, 3.5), 6, 0.15, fc="#888")); ax.text(0, 3.9, "far wall", ha="center")
    ax.add_patch(Rectangle((-0.6, 1.8), 1.2, 0.15, fc=RED)); ax.text(0.75, 1.85, "near object", color=RED, va="center")
    for cx, lab in [(-2.2, "camera 1"), (2.2, "camera 2")]:
        ax.add_patch(Polygon([[cx, 0], [cx - 0.35, -0.4], [cx + 0.35, -0.4]], fc="#333"))
        ax.text(cx, -0.75, lab, ha="center")
        ax.plot([cx, 0], [0, 1.9], color=RED, lw=1.5); ax.plot([cx, cx * -0.3], [0, 3.55], color=GREY, lw=1.2, ls="--")
    ax.text(0, -1.9, "the near object moves against the wall between the views:\nno single 2D transform fits both (parallax)", ha="center", fontsize=FONT - 3)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-2.6, 4.3); ax.axis("off"); ax.set_aspect("equal")
    save(fig, OUT, "fig-parallax.svg")


def fig_pipeline():
    steps = [("detect", "200 + 200\nkeypoints"), ("describe", "128 numbers\neach"), ("match", "89 after ratio\n+ mutual"), ("fit", "RANSAC:\n79 inliers"), ("warp", "true error\n0.07 px")]
    fig, ax = plt.subplots(figsize=(13, 2.4))
    for k, (a, b) in enumerate(steps):
        x = k * 2.6
        ax.add_patch(Rectangle((x, 0), 2.0, 1.2, fc="#e8eef6", ec=BLUE, lw=2))
        ax.text(x + 1.0, 0.82, a, ha="center", va="center", fontweight="bold")
        ax.text(x + 1.0, 0.35, b, ha="center", va="center", fontsize=FONT - 4)
        if k:
            ax.add_patch(FancyArrowPatch((x - 0.55, 0.6), (x - 0.05, 0.6), arrowstyle="->", mutation_scale=18, lw=2))
    ax.set_xlim(-0.1, 12.6); ax.set_ylim(-0.1, 1.3); ax.axis("off")
    save(fig, OUT, "fig-pipeline.svg")


if __name__ == "__main__":
    for f in [fig_overlay, fig_zoo, fig_ls_outliers, fig_camera, fig_mosaic, fig_parallax, fig_pipeline]:
        f()
