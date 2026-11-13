# Static figures for the geometry-from-images day (concept pictures and print fallbacks).
# Run from the repository root:
#   python3 -I tools/figs/multiview/figs.py
# Writes SVGs into cameras/, two-view/, stereo/, sfm/. No external data; fixed seed.
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams.update({"font.size": 15, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
C = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)
    print("wrote", path)


# 1. cameras/fig-pinhole.svg: similar triangles, the image plane behind and the virtual one in front
fig, ax = plt.subplots(figsize=(10, 4.2))
f, Z, Y = 1.0, 4.0, 1.4
ax.axhline(0, color="#888", lw=1, ls="--")
ax.plot([0, Z], [0, Y], color=C[0], lw=2.2)
ax.plot([-f, 0], [-Y * f / Z, 0], color=C[0], lw=2.2, ls=":")
ax.plot([-f, -f], [-0.9, 0.9], color="#555", lw=4)
ax.plot([f, f], [-0.9, 0.9], color=C[2], lw=4)
ax.plot([Z], [Y], "o", color=C[1], ms=10)
ax.plot([0], [0], "o", color="#111", ms=8)
ax.plot([f], [Y * f / Z], "o", color=C[0], ms=8)
ax.plot([-f], [-Y * f / Z], "o", color=C[0], ms=8)
ax.annotate("", xy=(Z, -0.25), xytext=(0, -0.25), arrowprops=dict(arrowstyle="<->", color="#333"))
ax.text(Z / 2, -0.45, "depth Z", ha="center")
ax.annotate("", xy=(f, -1.05), xytext=(0, -1.05), arrowprops=dict(arrowstyle="<->", color="#333"))
ax.text(f / 2, -1.28, "f", ha="center")
ax.text(Z + 0.1, Y, "point (X, Y, Z)", va="center")
ax.text(0.05, 0.12, "pinhole", fontsize=14)
ax.text(-f - 0.08, 1.0, "sensor\n(image upside down)", ha="center", fontsize=13)
ax.text(f + 0.05, 1.0, "virtual image plane", ha="left", fontsize=13, color=C[2])
ax.text(f + 0.08, Y * f / Z - 0.05, "y = f·Y / Z", fontsize=15, color=C[0], va="top")
ax.set_xlim(-1.6, 6.2); ax.set_ylim(-1.45, 1.6); ax.set_aspect("equal"); ax.axis("off")
save(fig, "cameras/fig-pinhole.svg")

# 2. cameras/fig-distortion.svg: a grid without, with barrel and with pincushion distortion
fig, axs = plt.subplots(1, 3, figsize=(11, 3.9))
for ax, k1, title in zip(axs, (0, -0.25, 0.25), ("no distortion", "barrel (k₁ < 0)", "pincushion (k₁ > 0)")):
    t = np.linspace(-0.6, 0.6, 200)
    for g in np.linspace(-0.6, 0.6, 9):
        for xs, ys in ((t, np.full_like(t, g)), (np.full_like(t, g), t)):
            r2 = xs ** 2 + ys ** 2
            ax.plot(xs * (1 + k1 * r2), ys * (1 + k1 * r2), color=C[0], lw=1.6)
    ax.set_title(title); ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.75, 0.75); ax.set_ylim(-0.75, 0.75)
save(fig, "cameras/fig-distortion.svg")

# 3. two-view/fig-epipolar.svg: the epipolar plane through C, C' and X cuts both image planes in lines
fig, ax = plt.subplots(figsize=(10, 5.2))
Cl, Cr, X = np.array([0.0, 0.0]), np.array([6.0, 0.0]), np.array([3.4, 4.2])
ax.add_patch(Polygon([Cl, Cr, X], closed=True, color=C[3], alpha=0.18))
L1 = [np.array(p) for p in ((-0.6, 1.0), (1.6, 1.6), (1.9, 0.2), (-0.3, -0.4))]
L2 = [np.array(p) for p in ((4.4, 1.6), (6.6, 1.0), (6.3, -0.4), (4.1, 0.2))]
for P in (L1, L2):
    ax.add_patch(Polygon(P, closed=True, fill=False, ec="#444", lw=2))
ax.plot([Cl[0], X[0]], [Cl[1], X[1]], color=C[0], lw=2.2)
ax.plot([Cr[0], X[0]], [Cr[1], X[1]], color=C[0], lw=2.2, ls="--")
ax.plot([Cl[0], Cr[0]], [Cl[1], Cr[1]], color="#333", lw=2)
for P, lab in ((Cl, "C"), (Cr, "C′")):
    ax.plot(*P, "o", color="#111", ms=9)
    ax.text(P[0], P[1] - 0.45, lab, ha="center", fontsize=17)
ax.plot(*X, "o", color=C[1], ms=11); ax.text(X[0] + 0.15, X[1], "X", fontsize=17, va="center")
# image points, epipoles, epipolar lines (by hand, on the plane's intersection with each image)
x1 = Cl + 0.33 * (X - Cl); x2 = Cr + 0.37 * (X - Cr)
e1 = Cl + 0.27 * (Cr - Cl); e2 = Cr + 0.27 * (Cl - Cr)
for x, e in ((x1, e1), (x2, e2)):
    ax.plot([e[0], x[0] + 0.5 * (x[0] - e[0])], [e[1], x[1] + 0.5 * (x[1] - e[1])], color=C[2], lw=3)
    ax.plot(*x, "o", color=C[0], ms=8)
    ax.plot(*e, "s", color=C[4], ms=9)
ax.text(x1[0] - 0.35, x1[1] + 0.1, "x", fontsize=16, color=C[0])
ax.text(x2[0] + 0.15, x2[1] + 0.1, "x′", fontsize=16, color=C[0])
ax.text(e1[0], e1[1] - 0.45, "e", fontsize=16, color=C[4], ha="center")
ax.text(e2[0], e2[1] - 0.45, "e′", fontsize=16, color=C[4], ha="center")
ax.text(3.0, -0.35, "baseline", ha="center", fontsize=14)
ax.text(5.0, 2.45, "epipolar line of x", color=C[2], fontsize=14)
ax.text(2.2, 2.2, "epipolar\nplane", color="#b5651d", fontsize=14, ha="center")
ax.set_xlim(-1, 7.2); ax.set_ylim(-0.9, 4.7); ax.set_aspect("equal"); ax.axis("off")
save(fig, "two-view/fig-epipolar.svg")

# 4. two-view/fig-depth-error.svg: the ±1 px wedges of two 1D cameras meet in a quadrilateral
def ray(c, th, u, fpx=200.0, cpx=79.5):
    x = (u - cpx) / fpx
    F = np.array([math.cos(th), math.sin(th)]); R = np.array([math.sin(th), -math.cos(th)])
    d = F + x * R
    return c, d / np.linalg.norm(d)

def meet(r1, r2):
    (o1, d1), (o2, d2) = r1, r2
    A = np.array([d1, -d2]).T
    s = np.linalg.solve(A, o2 - o1)
    return o1 + s[0] * d1

def proj(c, th, p, fpx=200.0, cpx=79.5):
    d = p - c; F = np.array([math.cos(th), math.sin(th)]); R = np.array([math.sin(th), -math.cos(th)])
    return fpx * (d @ R) / (d @ F) + cpx

fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
P = np.array([0.0, 0.4])
for ax, b, title in zip(axs, (0.25, 2.2), ("short baseline", "wide baseline")):
    cams = [np.array([-b, -3.0]), np.array([b, -3.0])]
    ths = [math.atan2(P[1] - c[1], P[0] - c[0]) for c in cams]
    us = [proj(c, t, P) for c, t in zip(cams, ths)]
    for k, (c, t, u) in enumerate(zip(cams, ths, us)):
        for s in (-1, 1):
            o, d = ray(c, t, u + 8 * s)
            ax.plot([o[0], o[0] + 6 * d[0]], [o[1], o[1] + 6 * d[1]], color=C[k], lw=1.2, alpha=0.7)
        ax.plot(*c, "o", color=C[k], ms=9)
    q = [meet(ray(cams[0], ths[0], us[0] + 8 * s1), ray(cams[1], ths[1], us[1] + 8 * s2)) for s1, s2 in ((-1, -1), (-1, 1), (1, 1), (1, -1))]
    ax.add_patch(Polygon(q, closed=True, color="#f2b705", alpha=0.8))
    ax.plot(*P, "o", color="#111", ms=6)
    ax.set_title(title + " (±8 px wedges)")
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-3.2, 1.9); ax.set_aspect("equal"); ax.axis("off")
save(fig, "two-view/fig-depth-error.svg")

# 5. stereo/fig-disparity-depth.svg: Z = f B / d with the depth range of d ± 1 px
fpx, B = 1000.0, 0.12
d = np.linspace(2.2, 80, 400)
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.fill_between(d, fpx * B / (d + 1), fpx * B / (d - 1), color=C[0], alpha=0.2, label="d ± 1 px")
ax.plot(d, fpx * B / d, color=C[0], lw=2.5, label="Z = f B / d")
for dd in (60, 6):
    ax.plot([dd], [fpx * B / dd], "o", color=C[1], ms=8)
    ax.text(dd + 1.5, fpx * B / dd + 1.5, f"d = {dd} px: Z = {fpx * B / dd:.0f} m", color=C[1])
ax.set_xlabel("disparity d (pixels)"); ax.set_ylabel("depth Z (m)")
ax.set_ylim(0, 40); ax.set_xlim(0, 80)
ax.legend(frameon=False)
ax.set_title("f = 1000 px, B = 0.12 m", fontsize=14)
save(fig, "stereo/fig-disparity-depth.svg")

# 6. sfm/fig-sparsity.svg: the Jacobian of a small bundle adjustment and its normal matrix
rng = np.random.default_rng(3)
nc, npnt = 4, 10
obs = []
for i in range(npnt):
    cams = sorted(rng.choice(nc, size=rng.integers(2, 4), replace=False))
    obs += [(j, i) for j in cams]
J = np.zeros((2 * len(obs), 6 * nc + 3 * npnt))
for r, (j, i) in enumerate(obs):
    J[2 * r:2 * r + 2, 6 * j:6 * j + 6] = 1
    J[2 * r:2 * r + 2, 6 * nc + 3 * i:6 * nc + 3 * i + 3] = 1
A = (J.T @ J > 0).astype(float)
fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1, 1.1]})
axs[0].imshow(J, cmap="Blues", aspect="auto", interpolation="nearest", vmin=0, vmax=1.3)
axs[0].axvline(6 * nc - 0.5, color=C[1], lw=2)
axs[0].set_title(f"J: {J.shape[0]} residuals × {J.shape[1]} unknowns")
axs[0].set_xlabel("cameras (6 each) | points (3 each)"); axs[0].set_ylabel("observations (2 rows each)")
axs[1].imshow(A, cmap="Blues", interpolation="nearest", vmin=0, vmax=1.3)
for v in (6 * nc - 0.5,):
    axs[1].axvline(v, color=C[1], lw=2); axs[1].axhline(v, color=C[1], lw=2)
axs[1].text(6 * nc / 2, 6 * nc / 2, "U", ha="center", va="center", fontsize=20, color=C[1])
axs[1].text(6 * nc + 1.5 * npnt, 6 * nc / 2, "W", ha="center", va="center", fontsize=20, color=C[1])
axs[1].text(6 * nc + 1.5 * npnt, 6 * nc + 1.5 * npnt, "V (block-diagonal)", ha="center", va="center", fontsize=15, color=C[1])
axs[1].set_title("JᵀJ: arrowhead pattern")
for ax in axs:
    ax.set_xticks([]); ax.set_yticks([])
save(fig, "sfm/fig-sparsity.svg")
