# Static figures for pipeline3d/lec.qmd. Run from the repo root:
#   python3 -I tools/figs/pipeline3d/figs.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(R, "pipeline3d")
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans"})
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#888888"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


def proj(p, az=np.radians(35), el=np.radians(22)):
    """Simple orthographic view of a 3D point (y up)."""
    x, y, z = p
    c, s = np.cos(az), np.sin(az)
    x1, z1 = c * x - s * z, s * x + c * z
    ce, se = np.cos(el), np.sin(el)
    return np.array([x1, ce * y - se * z1])


def arrow2(ax, a, b, col, lw=2.5, ms=18, ls="-"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=lw, color=col, mutation_scale=ms, linestyle=ls, shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- 1. indexed cube
V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
fig = plt.figure(figsize=(5.8, 6.4))
ax = fig.add_axes([0.15, 0.5, 0.7, 0.5])
ax.set_aspect("equal"); ax.axis("off")
P = np.array([proj(v - 0.5) for v in V])
front = [4, 5, 6, 7]
ax.add_patch(Polygon(P[[4, 5, 6]], closed=True, fc="#ffd9a8", ec=ORANGE, lw=2.5, zorder=1))
ax.add_patch(Polygon(P[[4, 6, 7]], closed=True, fc="#fff0dd", ec=ORANGE, lw=1.5, zorder=1))
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
for i, j in edges:
    hidden = 0 in (i, j)
    ax.plot(*P[[i, j]].T, color="#999" if hidden else "k", lw=1.5 if hidden else 2.2, ls="--" if hidden else "-", zorder=2)
for k, q in enumerate(P):
    d = q - P.mean(0); d = d / np.linalg.norm(d) * 0.12
    ax.plot(*q, "o", color=BLUE, ms=7, zorder=3)
    ax.text(*(q + d), str(k), fontsize=17, ha="center", va="center", color=BLUE, weight="bold")
ax.set_xlim(P[:, 0].min() - 0.25, P[:, 0].max() + 0.25); ax.set_ylim(P[:, 1].min() - 0.25, P[:, 1].max() + 0.25)
ax2 = fig.add_axes([0, 0, 1, 0.5]); ax2.axis("off")
ax2.text(0, 0.97, "vertex array (8 × 3 floats)", fontsize=17, weight="bold", va="top")
rows = [f"{k}: ({int(v[0])}, {int(v[1])}, {int(v[2])})" for k, v in enumerate(V)]
for k, r in enumerate(rows):
    ax2.text(0.02 + 0.5 * (k // 4), 0.84 - 0.11 * (k % 4), r, fontsize=15, va="top", family="DejaVu Sans Mono")
ax2.text(0, 0.36, "triangle list (12 × 3 indices)", fontsize=17, weight="bold", va="top")
ax2.text(0.02, 0.22, "(4, 5, 6)", fontsize=15, va="top", family="DejaVu Sans Mono", color=ORANGE, weight="bold")
ax2.text(0.34, 0.22, "(4, 6, 7)", fontsize=15, va="top", family="DejaVu Sans Mono", color=ORANGE)
ax2.text(0.66, 0.22, "(1, 0, 3) …", fontsize=15, va="top", family="DejaVu Sans Mono")
ax2.text(0.02, 0.09, "orange: the front face, two triangles\nsharing vertices 4 and 6", fontsize=14, va="top", color="#444")
save(fig, "fig-mesh.svg")

# ---------------------------------------------------------------- 2. winding, normal, non-manifold
fig = plt.figure(figsize=(6.2, 6.0))
axs = [fig.add_axes([0.05, 0.47, 0.9, 0.5]), fig.add_axes([0.0, 0.0, 0.48, 0.42]), fig.add_axes([0.52, 0.0, 0.48, 0.42])]
ax = axs[0]
a, b, c = np.array([0, 0, 0.]), np.array([1.6, 0, 0.2]), np.array([0.5, 0, 1.4])
a, b, c = [np.array([q[0], q[2] * 0.0 + 0, -q[2]]) for q in (a, b, c)]
# draw flat on the x-z plane seen from above-front
A, B, C = proj(a), proj(b), proj(c)
ax.add_patch(Polygon([A, B, C], closed=True, fc="#d4efd0", ec=GREEN, lw=2.5))
for P_, n in [(A, "a"), (B, "b"), (C, "c")]:
    ax.text(*(P_ + (P_ - (A + B + C) / 3) * 0.25), n, fontsize=19, weight="bold", ha="center", va="center")
cen = (A + B + C) / 3
ax.add_patch(FancyArrowPatch(cen + [0.25, -0.05], cen + [-0.25, -0.05], connectionstyle="arc3,rad=1.0", arrowstyle="-|>", mutation_scale=16, lw=2, color=GREEN))
nvec = np.cross(b - a, c - a); nvec = nvec / np.linalg.norm(nvec)
cen3 = (a + b + c) / 3
arrow2(ax, proj(cen3), proj(cen3 + 1.0 * nvec), "k", lw=3)
ax.text(*(proj(cen3 + 1.08 * nvec) + [0.08, 0]), "n = (b − a) × (c − a)", fontsize=17, va="bottom", ha="center")
ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.5, 2.0); ax.set_ylim(-0.6, 1.5)
ax.set_title("counter-clockwise from the front", fontsize=17)
ax = axs[1]
e0, e1 = np.array([0, 0, 0.]), np.array([0, 1.2, 0.])
for ang, col in [(0, BLUE), (120, ORANGE), (240, PURPLE)]:
    t = np.radians(ang); p = np.array([np.cos(t), 0.6, np.sin(t)])
    ax.add_patch(Polygon([proj(e0), proj(e1), proj(p)], closed=True, fc=col, alpha=0.35, ec=col, lw=2))
ax.plot(*np.array([proj(e0), proj(e1)]).T, color=RED, lw=4)
ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.7, 1.5)
ax.set_title("not manifold: 3 triangles\non one edge (red)", fontsize=16)
ax = axs[2]
cube = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
for off, col in [(np.array([-1, -1, 0.]), BLUE), (np.array([0, 0, 0.]), ORANGE)]:
    Q = np.array([proj((v + off) * 0.75) for v in cube])
    for f in [[4, 5, 6, 7], [1, 2, 6, 5], [2, 3, 7, 6]]:
        ax.add_patch(Polygon(Q[f], closed=True, fc=col, alpha=0.3, ec=col, lw=1.8))
ax.plot(*proj([0, 0, 0]), "o", color=RED, ms=10)
ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.5, 1.3); ax.set_ylim(-1.3, 1.1)
ax.set_title("not manifold: two cubes\ntouching at one corner", fontsize=16)
save(fig, "fig-winding.svg")

# ---------------------------------------------------------------- 3. T R vs R T
fig, ax = plt.subplots(figsize=(5.6, 4.8))
p = np.array([1, 0]); Rp = np.array([0, 1])
ax.axhline(0, color="#ccc", lw=1); ax.axvline(0, color="#ccc", lw=1)
ax.plot(*p, "ko", ms=9); ax.text(1.05, -0.32, "(1, 0)", fontsize=16)
th = np.linspace(0, np.pi / 2, 50)
ax.plot(np.cos(th), np.sin(th), color=BLUE, lw=2.5, ls="--")
arrow2(ax, np.array([np.cos(th[-3]), np.sin(th[-3])]), Rp, BLUE, lw=2.5)
arrow2(ax, Rp, Rp + [1, 0], BLUE, lw=3)
ax.plot(*(Rp + [1, 0]), "o", color=BLUE, ms=10)
ax.text(1.05, 1.12, "TR: (1, 1)", color=BLUE, fontsize=17, weight="bold")
arrow2(ax, p, p + [1, 0], ORANGE, lw=3)
th2 = np.linspace(0, np.pi / 2, 50)
ax.plot(2 * np.cos(th2), 2 * np.sin(th2), color=ORANGE, lw=2.5, ls="--")
arrow2(ax, np.array([2 * np.cos(th2[-3]), 2 * np.sin(th2[-3])]), [0, 2], ORANGE, lw=2.5)
ax.plot(0, 2, "o", color=ORANGE, ms=10)
ax.text(0.12, 2.12, "RT: (0, 2)", color=ORANGE, fontsize=17, weight="bold")
ax.text(1.35, 0.35, "R: rotate 90° about z\nT: move by (1, 0, 0)", fontsize=14, color="#444")
ax.set_xlim(-0.4, 2.5); ax.set_ylim(-0.5, 2.5); ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig-order.svg")

# ---------------------------------------------------------------- 4. look-at camera axes
pl = lambda q: proj(q, az=np.radians(-75), el=np.radians(18))
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.set_aspect("equal"); ax.axis("off")
e = np.array([2.6, 1.2, 1.4]); g = np.array([0, 0.3, 0]); up = np.array([0, 1, 0.])
f = (g - e) / np.linalg.norm(g - e); r = np.cross(f, up); r /= np.linalg.norm(r); u2 = np.cross(r, f)
# world axes and a target object (a small cube)
for v, lab in [((1, 0, 0), "x"), ((0, 1, 0), "y"), ((0, 0, 1), "z")]:
    arrow2(ax, pl([0, 0, 0]), pl(np.array(v) * 0.9), GREY, lw=1.8, ms=14)
    ax.text(*(pl(np.array(v) * 1.0) + [0.02, 0.02]), lab, color=GREY, fontsize=15)
Q = np.array([pl((v - 0.5) * 0.6 + g) for v in cube])
for fc in [[4, 5, 6, 7], [1, 2, 6, 5], [2, 3, 7, 6]]:
    ax.add_patch(Polygon(Q[fc], closed=True, fc="#eef3fa", ec=BLUE, lw=1.5))
ax.plot(*pl(g), "o", color=BLUE, ms=8, zorder=5); ax.text(*(pl(g) + [0.0, -0.55]), "target g", color=BLUE, fontsize=16, ha="center")
ax.plot(*pl(e), "o", color=RED, ms=10, zorder=5); ax.text(*(pl(e) + [0.12, -0.3]), "eye e", color=RED, fontsize=16)
ax.plot(*np.array([pl(e), pl(g)]).T, color=RED, lw=1.2, ls=":")
L = 1.0
for v, col, lab in [(f, ORANGE, "f (forward = −z)"), (r, GREEN, "r (= +x)"), (u2, PURPLE, "u′ (= +y)")]:
    arrow2(ax, pl(e), pl(e + L * v), col, lw=3)
    ax.text(*(pl(e + 1.12 * L * v) + ([-0.1, 0.32] if col == GREEN else [0.0, 0.02])), lab, color=col, fontsize=16, ha="center", va="bottom" if col == PURPLE else "top")
pts = np.array([pl(q) for q in [e, e + f, e + r, e + u2, g, [0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]])
ax.set_xlim(pts[:, 0].min() - 0.6, pts[:, 0].max() + 0.9); ax.set_ylim(pts[:, 1].min() - 0.5, pts[:, 1].max() + 0.4)
save(fig, "fig-lookat.svg")

# ---------------------------------------------------------------- 5. perspective vs orthographic (top view)
fig, axs = plt.subplots(2, 1, figsize=(6.0, 6.2))
objs = [(3.0, ORANGE), (6.0, BLUE)]  # distance, colour; both 1.0 tall
for ax, kind in zip(axs, ["perspective", "orthographic"]):
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.5, 7); ax.set_ylim(-1.5, 1.6)
    ax.plot([1, 1], [-1.3, 1.3], color="#444", lw=3)
    ax.text(1, -1.45, "image plane", ha="center", va="top", fontsize=14, color="#444")
    if kind == "perspective":
        ax.plot(0, 0, "o", color=RED, ms=9); ax.text(-0.1, -0.35, "eye", color=RED, fontsize=14)
    for d, col in objs:
        ax.plot([d, d], [0, 1.0], color=col, lw=5)
        if kind == "perspective":
            ax.plot([0, d], [0, 1.0], color=col, lw=1.3, ls="--"); ax.plot([0, d], [0, 0], color=col, lw=1.3, ls="--")
            ax.plot([1, 1], [0, 1.0 / d], color=col, lw=6, alpha=0.85)
        else:
            ax.plot([1, d], [1.0, 1.0], color=col, lw=1.3, ls="--"); ax.plot([1, d], [0, 0], color=col, lw=1.3, ls="--")
            off = -0.06 if col == ORANGE else 0.06
            ax.plot([1 + off, 1 + off], [0, 1.0], color=col, lw=5, alpha=0.85)
    ax.text(3.6, -0.75, "two objects of the same size,\nat distance 3 and 6", fontsize=14, ha="center", va="top", color="#444") if kind == "perspective" else ax.text(3.6, -0.75, "parallel rays:\nsame size on the image", fontsize=14, ha="center", va="top", color="#444")
    ax.set_title(kind + (": far one half as big" if kind == "perspective" else ""), fontsize=16)
save(fig, "fig-ortho.svg")

# ---------------------------------------------------------------- 6. frustum -> NDC cube (side view), a vertex behind the camera
n, fz = 1.0, 3.0
fov = np.radians(60); s = 1 / np.tan(fov / 2)
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.9), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axs[0]
ax.set_aspect("equal"); ax.axis("off")
tn, tf = n / s, fz / s
fr = np.array([[-n, -tn], [-fz, -tf], [-fz, tf], [-n, tn]])
ax.add_patch(Polygon(fr, closed=True, fc="#eef3fa", ec=BLUE, lw=2.5))
ax.plot(0, 0, "o", color=RED, ms=9); ax.text(0.05, -0.32, "eye", color=RED, fontsize=14)
ax.plot([0, -fz], [0, tf], color=BLUE, lw=1, ls=":"); ax.plot([0, -fz], [0, -tf], color=BLUE, lw=1, ls=":")
ax.text(-n, -tn - 0.15, "near", ha="center", va="top", fontsize=14); ax.text(-fz, -tf - 0.15, "far", ha="center", va="top", fontsize=14)
arrow2(ax, [0.6, -1.9], [-0.6, -1.9], "#444", lw=1.5, ms=14); ax.text(0.65, -1.9, "−z", va="center", fontsize=14)
pts = {"A": (-2.0, 0.6, GREEN), "B": (-1.5, -0.3, ORANGE), "C": (0.4, 0.3, PURPLE)}
for k, (z, y, col) in pts.items():
    ax.plot(z, y, "o", color=col, ms=10); ax.text(z + 0.08, y + 0.08, k, color=col, fontsize=17, weight="bold")
ax.text(0.4, 0.55, "behind\nthe camera", color=PURPLE, fontsize=14, ha="center", va="bottom")
ax.set_xlim(-3.3, 1.0); ax.set_ylim(-2.1, 2.0)
ax.set_title("view space (side view)", fontsize=16)
ax = axs[1]
ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Polygon([[-1, -1], [1, -1], [1, 1], [-1, 1]], closed=True, fc="#eef3fa", ec=BLUE, lw=2.5))
A_, B_ = (fz + n) / (fz - n), 2 * fz * n / (fz - n)
for k, (z, y, col) in pts.items():
    w = -z
    yn = s * y / w; zn = (A_ * z + B_) / w * -1  # z_clip = -A z - B ; zn = z_clip / w
    zn = (-A_ * z - B_) / w
    print(f"{k}: view (y={y}, z={z}) -> w={w:.2f}, y_ndc={yn:.3f}, z_ndc={zn:.3f}")
    X, Y = zn, yn  # plot depth horizontally (far = right? keep -z direction to the left)
    X = -zn
    if abs(Y) < 2.2 and abs(X) < 2.2:
        ax.plot(X, Y, "o", color=col, ms=10); ax.text(X + 0.07, Y + 0.07, k, color=col, fontsize=17, weight="bold")
    else:
        ax.text(0, -2.0, f"C: w = {w:.1f} < 0,\ny/w = {yn:.2f}: flipped", color=col, fontsize=15, ha="center", va="top")
ax.text(1, -1.45, "near", ha="center", fontsize=14); ax.text(-1, -1.45, "far", ha="center", fontsize=14)
ax.text(1.12, 0, "−1", fontsize=14, va="center"); ax.text(-1.12, 0, "+1", fontsize=14, va="center", ha="right")
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-2.9, 1.6)
ax.set_title("NDC, after ÷ w", fontsize=16)
save(fig, "fig-frustum.svg")

# ---------------------------------------------------------------- 7. a pixel is a ray
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.set_aspect("equal"); ax.axis("off")
ax.plot(0, 0, "o", color=RED, ms=10); ax.text(0, -0.3, "eye", color=RED, fontsize=16, ha="center", va="top")
ax.plot([1.0, 1.0], [-1.0, 1.4], color="#444", lw=3)
ax.add_patch(Polygon([[0.94, 0.25], [1.06, 0.25], [1.06, 0.5], [0.94, 0.5]], closed=True, fc=ORANGE, ec="k", zorder=4))
ax.text(1.0, 1.5, "screen", ha="center", fontsize=16)
ax.text(1.12, -0.25, "pixel", ha="left", fontsize=15, color=ORANGE)
dirv = np.array([1.0, 0.375])
ax.plot([0, 8 * dirv[0]], [0, 8 * dirv[1]], color=ORANGE, lw=2.5)
nearp, farp = 2.7 * dirv, 7.0 * dirv
ax.plot(*nearp, "o", color=BLUE, ms=10); ax.text(nearp[0], nearp[1] - 0.35, "depth −1\n(near)", color=BLUE, fontsize=15, ha="center", va="top")
ax.plot(*farp, "o", color=BLUE, ms=10); ax.text(farp[0], farp[1] - 0.35, "depth +1\n(far)", color=BLUE, fontsize=15, ha="center", va="top")
for t, col in [(3.6, GREEN), (5.2, PURPLE)]:
    ax.plot(*(t * dirv), "s", color=col, ms=10)
ax.text(4.4 * dirv[0] - 0.4, 4.4 * dirv[1] + 0.55, "every point on the ray\nlands on the same pixel", fontsize=15, ha="center")
ax.set_xlim(-0.4, 8.2); ax.set_ylim(-1.2, 3.6)
save(fig, "fig-pixelray.svg")
