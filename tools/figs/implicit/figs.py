"""Figures for implicit/lec.qmd (all analytic, no data). Run from the repo root:
   python3 -I tools/figs/implicit/figs.py
Colours as in the demos: φ < 0 (inside) blue, φ > 0 red, zero contour black.
SVG sizes: 1 pt in the figure = 1 px on the slide at the width given in lec.qmd."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "implicit")
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, ORANGE, SKY, PURPLE, GREY = "#0072b2", "#d55e00", "#009e73", "#e69f00", "#56b4e9", "#cc79a7", "#777777"


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), facecolor="white", bbox_inches="tight", pad_inches=0.04, **kw); plt.close(fig); print("wrote", name)


def phimap(ax, X, Y, phi, m, levels=None, lw0=2.5):
    ax.pcolormesh(X, Y, np.clip(phi, -m, m), cmap="RdBu_r", vmin=-m, vmax=m, shading="auto", rasterized=True)
    if levels is not None:
        ax.contour(X, Y, phi, levels=[l for l in levels if abs(l) > 1e-9], colors="#555", linewidths=0.8)
    ax.contour(X, Y, phi, levels=[0], colors="black", linewidths=lw0)
    ax.set_aspect("equal")


def blob(t):
    """The example shape of the deck: a rounded, three-lobed curve."""
    return 1.0 + 0.2 * np.cos(3 * t) + 0.08 * np.sin(2 * t)


def blob_phi(X, Y):
    """Signed distance to the blob (distance transforms of inside and outside, grid units -> x units)."""
    from scipy.ndimage import distance_transform_edt
    inside = np.hypot(X, Y) < blob(np.arctan2(Y, X))
    h = X[0, 1] - X[0, 0]
    return h * (distance_transform_edt(~inside) - distance_transform_edt(inside))


# ------------------------------------------------------------------ explicit vs implicit
def fig_twoways():
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3))
    t = np.linspace(0, 2 * np.pi, 400)
    tp = np.linspace(0, 2 * np.pi, 19)[:-1]
    for ax in axs:
        ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.55, 1.55); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    ax = axs[0]
    x, y = blob(tp) * np.cos(tp), blob(tp) * np.sin(tp)
    ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], "-", color=GREY, lw=1.5)
    ax.plot(x, y, "o", color=BLUE, ms=7)
    for k in (0, 1, 2):
        ax.text(x[k] * 1.14, y[k] * 1.14, f"$(x_{k + 1}, y_{k + 1})$", fontsize=14, ha="left", va="center")
    ax.set_title("explicit: a list of points", fontsize=16)
    ax = axs[1]
    g = np.linspace(-1.75, 1.75, 300)
    X, Y = np.meshgrid(g, g)
    phimap(ax, X, Y, blob_phi(X, Y), 1.0, levels=np.arange(-1.0, 1.01, 0.25))
    box = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
    ax.text(-0.15, 0, "φ < 0", ha="center", va="center", fontsize=16, color="black", bbox=box)
    ax.text(1.3, 1.3, "φ > 0", ha="center", va="center", fontsize=16, color="black", bbox=box)
    ax.set_title("implicit: φ on the whole grid", fontsize=16)
    fig.tight_layout(pad=0.6, w_pad=1.0); save(fig, "fig-twoways.svg", dpi=150)


# ------------------------------------------------------------------ the circle by formula
def fig_circle():
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    g = np.linspace(-2.5, 2.5, 300)
    X, Y = np.meshgrid(g, g)
    phi = np.hypot(X, Y) - 1.5
    phimap(ax, X, Y, phi, 1.6, levels=np.arange(-1.5, 2.01, 0.5))
    ax.plot([1], [0], "o", ms=9, color="black", mfc=ORANGE)
    ax.annotate("(1, 0): φ = −0.5", xy=(1, 0), xytext=(0.0, -0.7), fontsize=15, ha="center",
                arrowprops=dict(arrowstyle="->", lw=1.5), bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.annotate("r = 1.5", xy=(-1.06, 1.06), xytext=(-2.35, 2.05), fontsize=15,
                arrowprops=dict(arrowstyle="->", lw=1.5), bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_xticks([-2, -1, 0, 1, 2]); ax.set_yticks([-2, -1, 0, 1, 2])
    fig.tight_layout(pad=0.2); save(fig, "fig-circle.svg", dpi=150)


# ------------------------------------------------------------------ set operations
def fig_setops():
    g = np.linspace(-2.2, 2.2, 300)
    X, Y = np.meshgrid(g, g)
    p1 = np.hypot(X + 0.55, Y) - 1
    p2 = np.hypot(X - 0.55, Y) - 1
    panels = [("A and B", None), ("union: min(φA, φB)", np.minimum(p1, p2)),
              ("intersection: max(φA, φB)", np.maximum(p1, p2)), ("difference: max(φA, −φB)", np.maximum(p1, -p2))]
    fig, axs = plt.subplots(2, 2, figsize=(5.4, 4.2))
    for ax, (title, phi) in zip(axs.flat, panels):
        ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if phi is not None:
            ax.contourf(X, Y, phi, levels=[-10, 0], colors=[SKY], alpha=0.6)
            ax.contour(X, Y, phi, levels=[0], colors="black", linewidths=2.5, zorder=3)
        for p, lab, xl in ((p1, "A", -1.25), (p2, "B", 1.25)):
            ax.contour(X, Y, p, levels=[0], colors=GREY, linewidths=1.2, linestyles="--", zorder=2)
            if phi is None:
                ax.text(xl, 0, lab, fontsize=18, ha="center", va="center")
        ax.set_title(title, fontsize=14)
    fig.tight_layout(pad=0.2, h_pad=0.4, w_pad=0.6); save(fig, "fig-setops.svg", dpi=150)


# ------------------------------------------------------------------ normals and curvature
def fig_normals():
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    t = np.linspace(0, 2 * np.pi, 600)
    r = 1.0 + 0.2 * np.cos(3 * t)
    x, y = r * np.cos(t), r * np.sin(t)
    ax.fill(x, y, color=SKY, alpha=0.35, lw=0)
    ax.plot(x, y, color="black", lw=2.5)

    def geom(t):
        r, r1, r2 = 1 + 0.2 * np.cos(3 * t), -0.6 * np.sin(3 * t), -1.8 * np.cos(3 * t)
        p = np.array([r * np.cos(t), r * np.sin(t)])
        d = np.array([r1 * np.cos(t) - r * np.sin(t), r1 * np.sin(t) + r * np.cos(t)])
        d /= np.linalg.norm(d)
        n = np.array([d[1], -d[0]])  # outward for a counter-clockwise curve
        k = (r * r + 2 * r1 * r1 - r * r2) / (r * r + r1 * r1) ** 1.5
        return p, n, k

    for tt in np.linspace(0, 2 * np.pi, 25)[:-1]:
        p, n, _ = geom(tt)
        ax.annotate("", xy=p + 0.35 * n, xytext=p, arrowprops=dict(arrowstyle="->", lw=1.8, color=GREEN))
    for tt, lab, off in ((0.0, "bump", (0.1, -0.85)), (np.pi / 3, "dent", (1.05, 0.5))):
        p, n, k = geom(tt)
        c = p - n / k
        ax.add_patch(Circle(c, abs(1 / k), fc="none", ec=ORANGE if k > 0 else PURPLE, lw=2, ls="--"))
        ax.plot(*p, "o", color="black", ms=6)
        ax.text(p[0] + off[0], p[1] + off[1], f"{lab}: κ = {k:+.1f}".replace("-", "−"), fontsize=15,
                color=ORANGE if k > 0 else PURPLE, ha="left", fontweight="bold")
        print(lab, "kappa", round(k, 3), "radius", round(1 / abs(k), 3))
    ax.set_xlim(-1.45, 2.9); ax.set_ylim(-1.9, 1.95); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(-1.4, -1.75, "arrows: normal n = ∇φ/|∇φ|", color=GREEN, fontsize=15)
    fig.tight_layout(pad=0.1); save(fig, "fig-normals.svg", dpi=150)


# ------------------------------------------------------------------ signed distance of a square
def sdf_box(X, Y, h):
    dx, dy = np.abs(X) - h, np.abs(Y) - h
    return np.hypot(np.maximum(dx, 0), np.maximum(dy, 0)) + np.minimum(np.maximum(dx, dy), 0)


def fig_sdf():
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={"width_ratios": [1, 1.15]})
    g = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(g, g)
    phi = sdf_box(X, Y, 1.0)
    ax = axs[0]
    phimap(ax, X, Y, phi, 1.0, levels=np.arange(-1.0, 1.01, 0.25))
    ax.plot([-1, 1], [-1, 1], ":", color="black", lw=2)
    ax.plot([-1, 1], [1, -1], ":", color="black", lw=2)
    ax.axhline(0, color=ORANGE, lw=2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("level lines every 0.25", fontsize=15)
    ax = axs[1]
    x = np.linspace(-2, 2, 401)
    ax.plot(x, sdf_box(x, 0 * x, 1.0), color=ORANGE, lw=2.5)
    ax.axhline(0, color="#999", lw=1)
    ax.axhspan(-1.2, 0, color=SKY, alpha=0.12, lw=0)
    ax.set_ylim(-1.2, 1.2); ax.set_xlim(-2, 2)
    ax.set_xlabel("x (along the orange line)"); ax.set_ylabel("φ")
    ax.text(1.5, 0.15, "slope 1", fontsize=15, ha="center", rotation=0)
    ax.annotate("kink on the ridge", xy=(0, -1), xytext=(0, -0.45), ha="center", fontsize=15,
                arrowprops=dict(arrowstyle="->", lw=1.5))
    fig.tight_layout(pad=0.3, w_pad=1.2); save(fig, "fig-sdf.svg", dpi=150)


# ------------------------------------------------------------------ same zero set, different φ
def fig_profiles():
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    x = np.linspace(-2.2, 2.2, 441)
    r = 1.5
    ax.plot(x, np.abs(x) - r, color=BLUE, lw=2.5, label="distance: |x| − 1.5")
    ax.plot(x, x * x - r * r, color=RED, lw=2.5, label="x² − 1.5²")
    ax.plot(x, 3 * (np.abs(x) - r), color=GREEN, lw=2.5, ls="--", label="3 × distance")
    ax.axhline(0, color="#999", lw=1)
    ax.plot([-r, r], [0, 0], "o", color="black", ms=8, zorder=5)
    ax.set_ylim(-3, 3); ax.set_xlim(-2.2, 2.2)
    ax.set_xlabel("x (a line through the centre)"); ax.set_ylabel("φ")
    ax.legend(loc="upper center", frameon=False, fontsize=14, bbox_to_anchor=(0.5, 1.25), ncol=1)
    fig.tight_layout(pad=0.2); save(fig, "fig-profiles.svg", dpi=150)


# ------------------------------------------------------------------ the 5 × 5 grid
def fig_grid():
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    xs = np.arange(-2, 3)
    phi = lambda x, y: np.hypot(x, y) - 1.5
    for x in xs:
        ax.plot([x, x], [-2, 2], color="#ccc", lw=1)
        ax.plot([-2, 2], [x, x], color="#ccc", lw=1)
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(1.5 * np.cos(t), 1.5 * np.sin(t), "--", color=GREY, lw=1.5)
    # crossings on cell edges (linear interpolation), joined cell by cell
    pts = {}
    for x in xs:
        for y in xs:
            for dx, dy in ((1, 0), (0, 1)):
                x2, y2 = x + dx, y + dy
                if x2 > 2 or y2 > 2:
                    continue
                a, b = phi(x, y), phi(x2, y2)
                if (a < 0) != (b < 0):
                    s = a / (a - b)
                    pts[(x, y, dx, dy)] = (x + s * dx, y + s * dy)
    segs = []
    for x in xs[:-1]:
        for y in xs[:-1]:
            edges = [(x, y, 1, 0), (x, y + 1, 1, 0), (x, y, 0, 1), (x + 1, y, 0, 1)]
            c = [pts[e] for e in edges if e in pts]
            if len(c) == 2:
                segs.append(c)
    for a, b in segs:
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=2.5, zorder=5)
    for p in pts.values():
        ax.plot(*p, "o", color="black", ms=5, zorder=6)
    for x in xs:
        for y in xs:
            v = phi(x, y)
            ax.plot(x, y, "o", ms=25, color=SKY if v < 0 else "#f4b6a0", mec="none", zorder=3, alpha=0.9)
            ax.text(x, y, f"{v:.2f}".replace("-", "−"), ha="center", va="center", fontsize=10.5, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.08", fc=SKY if v < 0 else "#f4b6a0", ec="none", alpha=0.9))
    ax.set_xlim(-2.45, 2.45); ax.set_ylim(-2.45, 2.45); ax.set_aspect("equal")
    ax.set_xticks(xs); ax.set_yticks(xs)
    ax.tick_params(labelsize=13)
    ax.set_xlabel("x", fontsize=13); ax.set_ylabel("y", fontsize=13)
    fig.tight_layout(pad=0.2); save(fig, "fig-grid.svg", dpi=150)
    print("crossings", len(pts), "segments", len(segs))


if __name__ == "__main__":
    fig_twoways(); fig_circle(); fig_setops(); fig_normals(); fig_sdf(); fig_profiles(); fig_grid()
