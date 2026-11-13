"""Figures for surface-measure/lec.qmd. Run from the repo root (data.ts first, see there):
   python3 -I tools/figs/surface-measure/figs.py <tmp>/sm-data.json
2D contours by marching squares (scikit-image find_contours: linear interpolation on cell
edges, as lib/marching.ts). Colours: orange = voxels / pixels, blue = marching squares/cubes,
black dashed = the true shape. Fonts >= 15 pt on the slide."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Ellipse
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from scipy.special import erf
from skimage.measure import find_contours

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "surface-measure")
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
ORANGE, BLUE, GREEN, RED, PURPLE, GREY = "#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, facecolor="white")
    if os.environ.get("FIGPREVIEW"):
        fig.savefig(os.path.join(os.environ["FIGPREVIEW"], name.replace(".svg", ".png")), facecolor="white", dpi=100)
    plt.close(fig)
    print("wrote", p)


def staircase_segments(mask):
    segs = []
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            if y == 0 or not mask[y - 1, x]: segs.append(((x - .5, y - .5), (x + .5, y - .5)))
            if y == h - 1 or not mask[y + 1, x]: segs.append(((x - .5, y + .5), (x + .5, y + .5)))
            if x == 0 or not mask[y, x - 1]: segs.append(((x - .5, y - .5), (x - .5, y + .5)))
            if x == w - 1 or not mask[y, x + 1]: segs.append(((x + .5, y - .5), (x + .5, y + .5)))
    return segs


def contour_length(cs):
    return sum(np.sum(np.hypot(*np.diff(c, axis=0).T)) for c in cs)


# 1. the 2D version of the volume formula: signed triangles from the origin
def fig_signed():
    th = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    r = 1.0 + 0.25 * np.cos(3 * th) + 0.1 * np.sin(2 * th)
    P = np.c_[3.2 + 1.3 * r * np.cos(th), 2.0 + 1.0 * r * np.sin(th)]   # counter-clockwise
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    tot = 0
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        s = 0.5 * (a[0] * b[1] - b[0] * a[1])
        tot += s
        ax.add_patch(Polygon([(0, 0), a, b], closed=True, fc=BLUE if s > 0 else RED, alpha=0.22, ec="none"))
        ax.plot([0, a[0]], [0, a[1]], color="#bbb", lw=0.8)
    Pc = np.vstack([P, P[:1]])
    ax.plot(Pc[:, 0], Pc[:, 1], color="black", lw=2.5)
    for i in range(0, len(P), 2):
        a, b = P[i], P[(i + 1) % len(P)]
        ax.annotate("", xy=tuple(a + 0.6 * (b - a)), xytext=tuple(a + 0.4 * (b - a)),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=20))
    ax.plot(0, 0, "o", color="black", ms=8)
    ax.text(0.12, -0.12, "origin", va="top", fontsize=16)
    ax.text(4.9, 3.5, "+ facing away", color=BLUE, fontsize=16, ha="center")
    ax.text(1.0, 2.9, "− facing\nthe origin", color=RED, fontsize=16, ha="center")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.4, 5.6); ax.set_ylim(-0.5, 3.9)
    fig.tight_layout()
    save(fig, "fig-signed.svg")
    # check: the signed sum is the polygon area (shoelace)
    x, y = P[:, 0], P[:, 1]
    assert abs(tot - 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) < 1e-9


# 2. pixel edges vs marching squares on a partial-volume disk of radius 6
def fig_staircase():
    R, n, ss = 6.0, 17, 16
    c = (n - 1) / 2 + 0.3
    yy, xx = np.mgrid[0:n * ss, 0:n * ss]
    sub = (np.hypot((xx + 0.5) / ss - 0.5 - c, (yy + 0.5) / ss - 0.5 - c) < R).astype(float)
    frac = sub.reshape(n, ss, n, ss).mean(axis=(1, 3))        # fraction of each pixel inside
    mask = frac > 0.5
    segs = staircase_segments(mask)
    stair = len(segs)
    cs = find_contours(np.pad(frac, 1), 0.5)
    cs = [c_[:, ::-1] - 1 for c_ in cs]
    ms = contour_length(cs)
    true = 2 * np.pi * R
    fig, ax = plt.subplots(figsize=(4.6, 4.9))
    ax.imshow(frac, cmap="gray_r", vmin=0, vmax=2.2)
    for a, b in segs:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=ORANGE, lw=3, solid_capstyle="round")
    for c_ in cs:
        ax.plot(c_[:, 0], c_[:, 1], color=BLUE, lw=2.5)
    ax.add_patch(Circle((c, c), R, fc="none", ec="black", lw=1.5, ls="--"))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xlabel(f"true 2πR = {true:.1f}\npixel edges {stair} (+{100 * (stair / true - 1):.0f} %)\n"
                  f"marching squares {ms:.1f} ({100 * (ms / true - 1):+.1f} %)", fontsize=16, loc="left")
    ax.xaxis.label.set_color("#333")
    fig.tight_layout()
    save(fig, "fig-staircase.svg")
    return stair, ms, true


# 3. sphere and torus: V - E + F
def mesh_stats(P, T):
    E = set()
    for a, b, c in T:
        for u, v in ((a, b), (b, c), (c, a)):
            E.add((min(u, v), max(u, v)))
    return len(P), len(E), len(T)


def shade(ax, P, T, color=(0.62, 0.79, 0.88)):
    tri = P[T]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    L = np.array([-0.4, -0.5, 0.75]); L /= np.linalg.norm(L)
    lam = 0.35 + 0.65 * np.abs(nrm @ L)
    cols = np.clip(np.array(color)[None, :] * lam[:, None] + 0.12, 0, 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=cols, edgecolors="#1f4e79", linewidths=0.3))


def fig_topology(d):
    fig = plt.figure(figsize=(4.0, 5.0))
    out = {}
    for i, (key, elev) in enumerate((("sphere", 20), ("torus", 38))):
        P = np.array(d[key]["p"]).reshape(-1, 3); T = np.array(d[key]["t"]).reshape(-1, 3)
        V, E, F = mesh_stats(P, T)
        out[key] = (V, E, F, V - E + F)
        y0 = 0.5 * (1 - i)
        ax = fig.add_axes([-0.04, y0 + 0.02, 0.56, 0.46], projection="3d")
        shade(ax, P, T)
        c = P.mean(0); h = np.ptp(P, axis=0).max() / 2 * 0.8
        ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_zlim(c[2] - h, c[2] + h)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=-60); ax.axis("off")
        fig.text(0.53, y0 + 0.36, f"V = {V}\nE = {E}\nF = {F}", ha="left", va="top", fontsize=16, linespacing=1.3)
        fig.text(0.53, y0 + 0.12, f"χ = {V - E + F}, g = {(2 - (V - E + F)) // 2}", ha="left", fontsize=17,
                 color=BLUE, fontweight="bold")
    save(fig, "fig-topology.svg")
    return out


# 4. choosing c: profiles across the boundary of a blurred step and of blurred balls
def ball_profile(r, R, s):
    """A ball of radius R (value 1) blurred by a 3D Gaussian of width s, at distance r from the centre."""
    r = np.maximum(r, 1e-9)
    a = 0.5 * (erf((R - r) / (np.sqrt(2) * s)) + erf((R + r) / (np.sqrt(2) * s)))
    b = s / (r * np.sqrt(2 * np.pi)) * (np.exp(-(R - r) ** 2 / (2 * s * s)) - np.exp(-(R + r) ** 2 / (2 * s * s)))
    return a - b


def fig_halfway():
    s = 1.0
    u = np.linspace(-3, 3, 400)          # distance from the true boundary, outwards (voxels)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    step = 0.5 * (1 - erf(u / (np.sqrt(2) * s)))
    ax.plot(u, step, color="black", lw=2.5, label="flat edge")
    vals = {}
    for R, col in ((4, BLUE), (2, RED)):
        p = ball_profile(R + u, R, s)
        ok = u > -R + 0.05                     # inside the ball only down to its centre
        ax.plot(u[ok], p[ok], color=col, lw=2.5, label=f"ball, R = {R}σ")
        vals[R] = ball_profile(np.array([R]), R, s)[0]
        ax.plot(0, vals[R], "o", color=col, ms=9, mec="black")
        # where the surface at c = 0.5 lands
        k = np.argmin(np.where(ok, np.abs(p - 0.5), 9))
        ax.plot([u[k], u[k]], [0, 0.5], color=col, lw=1.2, ls=":")
    ax.axhline(0.5, color=GREY, lw=1.5, ls="--")
    ax.axvline(0, color=GREY, lw=1)
    ax.text(-2.95, 0.47, "c = (a + b)/2", color="#555", ha="left", va="top", fontsize=16)
    ax.text(0, 1.04, "true boundary", color="#555", fontsize=16, va="bottom", ha="center")
    ax.set_xlabel("distance from the true boundary (σ)")
    ax.set_ylabel("blurred value")
    ax.set_xlim(-3, 3); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=15, loc="upper right")
    fig.tight_layout()
    save(fig, "fig-halfway.svg")
    return vals


# 5. surfacing a mask: terraces at c = 0.5, smooth after a slight blur (a 2D slice)
def fig_mask():
    n = 26
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    a, b, th = 9.5, 5.5, np.deg2rad(25)
    cx, cy = (n - 1) / 2, (n - 1) / 2
    X = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
    Y = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
    mask = ((X / a) ** 2 + (Y / b) ** 2 < 1).astype(float)
    fig, axs = plt.subplots(2, 1, figsize=(3.4, 5.4))
    for ax, img, title in ((axs[0], mask, "mask, c = 0.5"), (axs[1], gaussian_filter(mask, 1.0), "mask blurred, σ = 1")):
        ax.imshow(mask, cmap="gray_r", vmin=0, vmax=2.6)
        for c_ in find_contours(np.pad(img, 1), 0.5):
            c_ = c_ - 1
            ax.plot(c_[:, 1], c_[:, 0], color=BLUE, lw=3)
        ax.add_patch(Ellipse((cx, cy), 2 * a, 2 * b, angle=np.rad2deg(th), fc="none", ec="black", lw=1.3, ls="--"))
        ax.set_title(title, fontsize=16)
        ax.set_xlim(2.5, n - 3.5); ax.set_ylim(n - 6.5, 5.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
    fig.tight_layout()
    save(fig, "fig-mask.svg")


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    fig_signed(); print(fig_staircase()); print(fig_topology(data)); print(fig_halfway()); fig_mask()
