"""Figures for hough-circles/lec.qmd.

Run from the repository root:
    tools/figs/hough-lines/capture.sh /tmp/houghstills      # demo screenshots (needs docs/demos/dist)
    python3 -I tools/figs/hough-circles/figs.py /tmp/houghstills
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../hough-lines"))
from stills import plt, panel, grid_of_panels, BLUE, ORANGE, GREEN, RED, PURPLE, GREY  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = os.path.join(REPO, "hough-circles")
SH = sys.argv[1] if len(sys.argv) > 1 else None


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200 if name.endswith(".png") else None,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ---------------------------------------------------------------- the cone of votes
def fig_cone():
    r0 = 3.0
    angs = np.radians([90, 210, 330])
    P = np.c_[r0 * np.cos(angs), r0 * np.sin(angs)]
    cols = [BLUE, ORANGE, GREEN]
    fig = plt.figure(figsize=(4.6, 4.3))
    ax = fig.add_subplot(projection="3d")
    t = np.linspace(0, 2 * np.pi, 120)
    for (px, py), c in zip(P, cols):
        for r in np.arange(0.5, 5.01, 0.5):
            ax.plot(px + r * np.cos(t), py + r * np.sin(t), np.full_like(t, r), color=c, lw=1.0, alpha=0.75)
        ax.plot([px], [py], [0], "o", color=c, ms=7)
    # the true circle in the image plane (r = 0) and the common point (0, 0, r0)
    ax.plot(r0 * np.cos(t), r0 * np.sin(t), 0 * t, color="k", lw=1.2, ls="--")
    ax.plot([0, 0], [0, 0], [0, r0], color="k", lw=1, ls=":")
    ax.scatter([0], [0], [r0], s=160, facecolors="none", edgecolors=RED, linewidths=2.5, depthshade=False)
    ax.text(3.2, -3.2, r0 + 0.3, "(0, 0, 3)", color=RED, fontsize=14)
    ax.set_xlabel("a", labelpad=-6)
    ax.set_ylabel("b", labelpad=-6)
    ax.set_zlabel("r", labelpad=-6)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticks([0, 1, 2, 3, 4, 5])
    ax.tick_params(axis="z", labelsize=11, pad=0)
    ax.view_init(elev=14, azim=-60)
    ax.set_box_aspect((1, 1, 0.75))
    save(fig, "fig-cone.svg")


# ---------------------------------------------------------------- gradient points at the centre
def fig_gradcentre():
    p, u, r = np.array([3.0, 4.0]), np.array([0.6, 0.8]), 5.0
    cb, cd = p + r * u, p - r * u
    fig, ax = plt.subplots(figsize=(4.0, 4.8))
    ax.set_aspect("equal")
    t = np.linspace(0, 2 * np.pi, 200)
    ax.fill(cb[0] + r * np.cos(t), cb[1] + r * np.sin(t), color="#e8e8e8", zorder=0)
    ax.plot(cb[0] + r * np.cos(t), cb[1] + r * np.sin(t), color=BLUE, lw=2, ls="--")
    ax.plot(cd[0] + r * np.cos(t), cd[1] + r * np.sin(t), color=ORANGE, lw=2, ls="--")
    ax.annotate("", xy=p + 2.2 * u, xytext=p,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=2.5, mutation_scale=18))
    ax.plot(*p, "o", color="k", ms=9, zorder=5)
    ax.plot(*cb, "+", color=BLUE, ms=16, mew=3)
    ax.plot(*cd, "+", color=ORANGE, ms=16, mew=3)
    ax.text(p[0] + 0.5, p[1] - 0.5, "(3, 4)", fontsize=14, va="center")
    ax.text(p[0] - 0.3, p[1] + 1.1, "gradient", fontsize=13, ha="right")
    ax.text(cb[0] + 0.7, cb[1] + 0.6, "(6, 8)", fontsize=14, color=BLUE)
    ax.text(cb[0], cb[1] + 6.0, "bright disc: p + r u", fontsize=13, color=BLUE, ha="center")
    ax.text(cd[0] - 0.7, cd[1] - 1.3, "(0, 0)", fontsize=14, color=ORANGE, ha="right")
    ax.text(cd[0], cd[1] - 6.6, "dark disc: p − r u", fontsize=13, color=ORANGE, ha="center")
    ax.plot([cd[0], cb[0]], [cd[1], cb[1]], color=GREY, lw=1, ls=":", zorder=1)
    ax.set_xlim(-6, 12)
    ax.set_ylim(-7.4, 14.6)
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0, 5, 10])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    save(fig, "fig-gradcentre.svg")


# ---------------------------------------------------------------- distance transform of touching discs
def fig_alternatives():
    n = 120
    yy, xx = np.mgrid[:n, :n]
    discs = [(42, 60, 26), (82, 60, 22)]
    mask = np.zeros((n, n), bool)
    for cx, cy, r in discs:
        mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    dt = ndi.distance_transform_edt(mask)
    mx = (dt == ndi.maximum_filter(dt, size=15)) & (dt > 5)
    lab, nl = ndi.label(mx)
    pk = np.array(ndi.center_of_mass(mx, lab, range(1, nl + 1)))
    vals = [dt[int(round(y)), int(round(x))] for y, x in pk]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.4, 2.9))
    a1.imshow(mask, cmap="gray", interpolation="nearest")
    a1.set_title("binary mask", fontsize=14)
    im = a2.imshow(dt, cmap="viridis", interpolation="nearest")
    for (y, x), v in zip(pk, vals):
        a2.plot(x, y, "+", color=RED, ms=14, mew=3)
        a2.text(x, y + 13, f"{v:.0f}", color="white", fontsize=14, ha="center")
    a2.set_title("distance transform", fontsize=14)
    for a in (a1, a2):
        a.axis("off")
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=11)
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig-alternatives.png")
    return [(round(x), round(y), round(v, 1)) for (y, x), v in zip(pk, vals)]


# ---------------------------------------------------------------- ellipse: pairs with parallel gradients
def fig_ellipse():
    A, B, phi, c = 5.0, 3.0, np.radians(25), np.array([0.0, 0.0])
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

    def pt(t):
        return c + R @ np.array([A * np.cos(t), B * np.sin(t)])

    def nrm(t):
        v = R @ np.array([np.cos(t) / A, np.sin(t) / B])
        return v / np.linalg.norm(v)
    fig, ax = plt.subplots(figsize=(4.6, 3.7))
    ax.set_aspect("equal")
    T = np.linspace(0, 2 * np.pi, 300)
    E = np.array([pt(t) for t in T])
    ax.fill(E[:, 0], E[:, 1], color="#e8e8e8", zorder=0)
    ax.plot(E[:, 0], E[:, 1], color="k", lw=1.5)
    cols = [BLUE, ORANGE, GREEN, PURPLE]
    for t, col in zip(np.radians([20, 70, 115, 160]), cols):
        p, q = pt(t), pt(t + np.pi)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=1.2, ls="--")
        for s, tt in ((p, t), (q, t + np.pi)):
            ax.annotate("", xy=s - 1.3 * nrm(tt), xytext=s,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2, mutation_scale=14))
            ax.plot(*s, "o", color=col, ms=7)
    ax.plot(*c, "o", ms=14, mfc="none", mec=RED, mew=2.5)
    ax.annotate("centre = every midpoint", xy=c + np.array([0.15, -0.3]), xytext=(0.3, -4.6), color=RED, fontsize=13,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5))
    ax.axis("off")
    fig.tight_layout()
    save(fig, "fig-ellipse-pairs.svg")


def fig_stills():
    grid_of_panels([panel(SH, "hc_plate", 0), panel(SH, "hc_plate", 1)],
                   ["42 wells found", "accumulator, r = 8"],
                   os.path.join(OUT, "fig-plate.png"), ncols=2, w=4.6)
    grid_of_panels([panel(SH, "hc_false", 0), panel(SH, "hc_ellipse", 2)],
                   ["threshold 0.30: 18 found", "ellipse: smeared peak"],
                   os.path.join(OUT, "fig-wrong.png"), ncols=2, w=4.8)


if __name__ == "__main__":
    fig_cone()
    fig_gradcentre()
    print("distance transform peaks (x, y, value):", fig_alternatives())
    fig_ellipse()
    if SH:
        fig_stills()
