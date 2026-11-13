# Figures for moments-basis/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/moments-basis/figs.py
# Shapes follow the moments demo (L of the presets, pearPoly of mode-higher.ts); the
# micrograph is demos/src/fourier2d/cell256.png (Robert M. Hunt, CC BY 3.0, Wikimedia Commons).
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "label-connectivity"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Rectangle, Circle
from labstyle import setup, fig, save, BLUE, ORANGE, GREEN, RED, GREY, PIX, INK

setup()
OUT = "moments-basis"
FILL = "#a6c4f4"


# ---------------------------------------------------------------- polygon moments (Green)
def poly_moments(P):
    """Area, centroid and covariance S of a polygon (exact, counter-clockwise or not)."""
    P = np.asarray(P, float); x0, y0 = P[:, 0], P[:, 1]; x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    c = x0 * y1 - x1 * y0
    A = c.sum() / 2
    cx = ((x0 + x1) * c).sum() / (6 * A); cy = ((y0 + y1) * c).sum() / (6 * A)
    sxx = ((x0**2 + x0 * x1 + x1**2) * c).sum() / (12 * A) - cx**2
    syy = ((y0**2 + y0 * y1 + y1**2) * c).sum() / (12 * A) - cy**2
    sxy = ((x0 * y1 + 2 * x0 * y0 + 2 * x1 * y1 + x1 * y0) * c).sum() / (24 * A) - cx * cy
    return abs(A), np.array([cx, cy]), np.array([[sxx, sxy], [sxy, syy]])


def pear(cx, cy, s, n=160):
    rB, b = 0.6 * s, 1.05 * s
    t = 2 * np.pi * np.arange(n) / n
    d = rB**2 - b**2 * np.sin(t)**2
    far = np.where(d >= 0, b * np.cos(t) + np.sqrt(np.maximum(d, 0)), -np.inf)
    r = np.maximum(s, far)
    return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]


def msqrt(S):
    w, V = np.linalg.eigh(S); return V @ np.diag(np.sqrt(w)) @ V.T


def equiv_ellipse(ax, c, S, col=ORANGE):
    w, V = np.linalg.eigh(S)
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    ax.add_patch(Ellipse(c, 4 * np.sqrt(w[1]), 4 * np.sqrt(w[0]), angle=ang, fill=False, ec=col, lw=3, ls="--"))
    ax.plot(*c, "+", color=INK, ms=16, mew=2.5)


# ---------------------------------------------------------------- 1. same ellipse
# the L of the demo presets (lib/shapecanvas.ts SHAPE_PRESETS, s = 64)
L = 64 * np.array([[0.25, 0.2], [0.45, 0.2], [0.45, 0.6], [0.75, 0.6], [0.75, 0.8], [0.25, 0.8]])
AL, cL, SL = poly_moments(L)
P = pear(0, 0, 12)
AP, cP, SP = poly_moments(P)
T = msqrt(SL) @ np.linalg.inv(msqrt(SP))          # maps the pear's covariance onto the L's
P2 = (P - cP) @ T.T + cL
# check
_, c2, S2 = poly_moments(P2)
assert np.allclose(S2, SL) and np.allclose(c2, cL)
w, V = np.linalg.eigh(SL)
t = np.linspace(0, 2 * np.pi, 200)
E = cL + (np.c_[2 * np.sqrt(w[1]) * np.cos(t), 2 * np.sqrt(w[0]) * np.sin(t)] @ np.c_[V[:, 1], V[:, 0]].T)
_, c3, S3 = poly_moments(E)
print("same-ellipse: S_L", SL.round(2), "ellipse S", S3.round(2))

f, axs = fig(780, 280, ncols=3)
for ax, poly, name in zip(axs, [L, P2, E], ["an L", "a pear", "a solid ellipse"]):
    ax.add_patch(Polygon(poly, closed=True, fc=FILL, ec=BLUE, lw=2))
    equiv_ellipse(ax, cL, SL)
    ax.set_xlim(8, 56); ax.set_ylim(60, 4); ax.set_aspect("equal"); ax.axis("off")
    ax.text(32, 62, name, ha="center", va="top", fontsize=20)
f.text(0.5, -0.06, "dashed: the equivalent ellipse, the same for all three", ha="center", fontsize=18, color="#555")
save(f, f"{OUT}/fig-sameellipse.svg")


# ---------------------------------------------------------------- 2. marginals of an image
def raster(poly, n=64):
    from matplotlib.path import Path
    yy, xx = np.mgrid[0:n, 0:n] + 0.5
    return Path(poly).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(n, n).astype(float)


M = raster(L)
xs = np.arange(64) + 0.5
gx, gy = M.sum(0), M.sum(1)
m00 = M.sum(); mx = (gx * xs).sum() / m00; my = (gy * xs).sum() / m00
sx = np.sqrt((gx * (xs - mx)**2).sum() / m00); sy = np.sqrt((gy * (xs - my)**2).sum() / m00)
skx = (gx * (xs - mx)**3).sum() / m00 / sx**3; sky = (gy * (xs - my)**3).sum() / m00 / sy**3
print("marginals: mean", mx, my, "sd", sx, sy, "skew", skx, sky)
f = plt.figure(figsize=(430 / 72, 430 / 72))
a0 = f.add_axes([0.0, 0.32, 0.66, 0.66]); a1 = f.add_axes([0.0, 0.05, 0.66, 0.25]); a2 = f.add_axes([0.69, 0.32, 0.25, 0.66])
from matplotlib.colors import ListedColormap
a0.imshow(M, cmap=ListedColormap(["white", FILL]), vmin=0, vmax=1, extent=(0, 64, 64, 0), interpolation="nearest")
a0.set_xlim(10, 54); a0.set_ylim(56, 8); a0.set_xticks([]); a0.set_yticks([])
for s in a0.spines.values(): s.set_visible(True); s.set_color("#bbb")
a0.plot(mx, my, "+", color=INK, ms=18, mew=3)
a1.bar(xs, gx, width=1, color=RED, alpha=0.45); a1.set_xlim(10, 54); a1.set_ylim(0, gx.max() * 1.1)
a1.axvline(mx, color=RED, lw=2); a1.plot([mx - sx, mx + sx], [gx.max() * 1.02] * 2, color=RED, lw=2)
a1.set_yticks([]); a1.set_xticks([]); a1.set_xlabel("column sums: the x-marginal", labelpad=4)
a2.barh(xs, gy, height=1, color=GREEN, alpha=0.45); a2.set_ylim(56, 8); a2.set_xlim(0, gy.max() * 1.1)
a2.axhline(my, color=GREEN, lw=2); a2.plot([gy.max() * 1.02] * 2, [my - sy, my + sy], color=GREEN, lw=2)
a2.set_xticks([]); a2.set_yticks([]); a2.set_ylabel("row sums", labelpad=4); a2.yaxis.set_label_position("right")
a1.text(53, gx.max() * 0.95, f"skewness {skx:+.2f}", ha="right", va="top", fontsize=17, color="#8b1a1a")
save(f, f"{OUT}/fig-marginals.svg")


# ---------------------------------------------------------------- 3. kurtosis: square, disc, Gaussian with the same S
a = 1.0
r = a / np.sqrt(3); sg = a / np.sqrt(12)
u = np.linspace(-1.1, 1.1, 801)
gsq = np.where(abs(u) <= a / 2, 1 / a, 0)
gdi = np.where(abs(u) <= r, 2 / (np.pi * r**2) * np.sqrt(np.maximum(r**2 - u**2, 0)), 0)
gga = np.exp(-u**2 / (2 * sg**2)) / np.sqrt(2 * np.pi) / sg
for g, n in [(gsq, "square"), (gdi, "disc"), (gga, "gauss")]:
    du = u[1] - u[0]; m0 = (g * du).sum(); v = (g * u**2 * du).sum() / m0; k = (g * u**4 * du).sum() / m0 / v**2
    print("kurtosis", n, "var", round(v, 4), "kappa", round(k, 3))
f, axs = plt.subplots(2, 3, figsize=(600 / 72, 390 / 72), gridspec_kw=dict(height_ratios=[1.15, 1], hspace=0.08, wspace=0.08))
yy, xx = np.mgrid[-0.9:0.9:300j, -0.9:0.9:300j]
for j, (name, kap, g) in enumerate([("square", "1.8", gsq), ("disc", "2.0", gdi), ("Gaussian blob", "3", gga)]):
    ax = axs[0, j]
    if j == 0: ax.add_patch(Rectangle((-a / 2, -a / 2), a, a, fc=FILL, ec=BLUE, lw=2))
    elif j == 1: ax.add_patch(Circle((0, 0), r, fc=FILL, ec=BLUE, lw=2))
    else: ax.imshow(np.exp(-(xx**2 + yy**2) / (2 * sg**2)), cmap="Blues", vmin=-0.15, vmax=1.6, extent=(-0.9, 0.9, -0.9, 0.9))
    ax.add_patch(Circle((0, 0), 2 * sg, fill=False, ec=ORANGE, lw=2.5, ls="--"))
    ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.9, 0.9); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(name, fontsize=19, pad=2)
    b = axs[1, j]
    b.fill_between(u, g, color=RED, alpha=0.3, lw=0); b.plot(u, g, color=RED, lw=2.2)
    b.set_xlim(-0.9, 0.9); b.set_ylim(0, 1.8); b.set_yticks([]); b.set_xticks([])
    b.text(0, 1.78, f"κ = {kap}", ha="center", va="top", fontsize=20)
axs[1, 0].set_ylabel("marginal")
save(f, f"{OUT}/fig-kurtosis.svg")


# ---------------------------------------------------------------- 4. Taylor series of the spectrum at 0
# 1D box of width 1: F(u) = sin(pi u) / (pi u); moments m_k = (1/2)^k / (k+1) for even k.
from math import factorial
uu = np.linspace(0, 2.5, 600)
F = np.sinc(uu)
def taylor(N):
    s = np.zeros_like(uu, dtype=complex)
    for k in range(0, N + 1):
        mk = (0.5**k / (k + 1)) if k % 2 == 0 else 0.0
        s += (-2j * np.pi * uu)**k / factorial(k) * mk
    return s.real
f, ax = fig(500, 330)
ax.axhline(0, color="#ccc", lw=1)
ax.plot(uu, F, color=INK, lw=3.5, label="spectrum of a bar")
cols = ["#E69F00", "#009E73", "#0072B2"]
for N, c in zip([2, 6, 12], cols):
    ax.plot(uu, taylor(N), color=c, lw=2.4, ls="--", label=f"from moments ≤ order {N}")
ax.set_ylim(-0.5, 1.15); ax.set_xlim(0, 2.5)
ax.set_xlabel("frequency u (cycles per bar length)"); ax.set_ylabel("F(u)")
ax.legend(loc="upper right", frameon=False, fontsize=15)
save(f, f"{OUT}/fig-taylor.svg")


# ---------------------------------------------------------------- 5. the matrix sandwich
from skimage import io, transform
cell = io.imread("demos/src/fourier2d/cell256.png", as_gray=True).astype(float)
if cell.max() > 1: cell = cell / 255
I = transform.resize(cell, (64, 64), anti_aliasing=True)
N = 4
xv = np.linspace(-1, 1, 64)
V = np.stack([xv**p for p in range(N + 1)], 1)       # 64 x 5
Mm = V.T @ I @ V
FW, FH = 760, 230
f = plt.figure(figsize=(FW / 72, FH / 72))
def box(x, y, w, h):   # pixel coordinates, origin top left
    return f.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH])
D, T = 170, 55        # image side and thickness of the 5-column matrices, in px
top = 30
def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
ax1 = box(0, top, D, T); ax1.imshow(V.T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"); clean(ax1)
ax1.set_title("$V_x^\\top$: rows $x^0 \\dots x^4$", fontsize=16, pad=4)
f.text((D + 12) / FW, 1 - (top + D / 2) / FH, "×", fontsize=26, va="center")
ax2 = box(D + 40, top, D, D); ax2.imshow(I, cmap="gray"); clean(ax2); ax2.set_title("image $I$", fontsize=16, pad=4)
f.text((2 * D + 52) / FW, 1 - (top + D / 2) / FH, "×", fontsize=26, va="center")
ax3 = box(2 * D + 80, top, T, D); ax3.imshow(V, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"); clean(ax3)
ax3.set_title("$V_y$", fontsize=16, pad=4)
f.text((2 * D + 80 + T + 12) / FW, 1 - (top + T / 2) / FH, "=", fontsize=26, va="center")
x4 = 2 * D + 80 + T + 45
ax4 = box(x4, top, T, T)
vm = np.abs(Mm).max()
ax4.imshow(Mm, cmap="RdBu_r", vmin=-vm, vmax=vm); clean(ax4)
ax4.set_title("$M$", fontsize=16, pad=4)
f.text((x4 + T + 12) / FW, 1 - (top + T / 2) / FH, "all $m_{pq}$,\n$p, q \\leq 4$", fontsize=16, va="center")
f.text((D + 40) / FW, 1 - (top + D + 6) / FH, "64 × 64 pixels, $x, y$ in [−1, 1]", fontsize=15, color="#555", va="top")
f.text(0, 1 - (top + T + 14) / FH, "red +, blue −", fontsize=15, color="#555", va="top")
save(f, f"{OUT}/fig-sandwich.svg")


# ---------------------------------------------------------------- 6. Tsai's moment-preserving threshold
g = np.clip(np.round(cell * 255), 0, 255).astype(int)
h = np.bincount(g.ravel(), minlength=256) / g.size
z = np.arange(256)
m1, m2, m3 = (h * z).sum(), (h * z**2).sum(), (h * z**3).sum()
cd = m2 - m1**2
c0 = (m1 * m3 - m2**2) / cd; c1 = (m1 * m2 - m3) / cd
z0 = 0.5 * (-c1 - np.sqrt(c1**2 - 4 * c0)); z1 = 0.5 * (-c1 + np.sqrt(c1**2 - 4 * c0))
p0 = (z1 - m1) / (z1 - z0)
thr = int(np.searchsorted(np.cumsum(h), p0))
print(f"Tsai: z0={z0:.1f} z1={z1:.1f} p0={p0:.3f} threshold={thr}")
# check the three moments of the two-level image
for k in (1, 2, 3):
    print("  moment", k, (h * z**k).sum(), p0 * z0**k + (1 - p0) * z1**k)
f = plt.figure(figsize=(560 / 72, 470 / 72))
ah = f.add_axes([0.1, 0.56, 0.86, 0.4])
ah.bar(z, h * 100, width=1, color=GREY, alpha=0.7, label="histogram")
ymax = h.max() * 100 * 1.25
ah.vlines([z0, z1], 0, ymax * 0.9, color=[BLUE, ORANGE], lw=4)
ah.text(z0, ymax * 0.93, f"$z_0$ ({p0 * 100:.0f} %)", ha="center", va="bottom", color=BLUE, fontsize=16)
ah.text(z1, ymax * 0.93, f"$z_1$ ({(1 - p0) * 100:.0f} %)", ha="center", va="bottom", color=ORANGE, fontsize=16)
ah.axvline(thr, color=INK, lw=2, ls="--"); ah.text(thr + 3, ymax * 0.62, f"threshold {thr}", fontsize=16, bbox=dict(fc="white", ec="none", pad=1))
ah.set_xlim(0, 256); ah.set_xticks([0, 50, 100, 150, 200, 250]); ah.set_ylim(0, ymax * 1.12); ah.set_yticks([])
ah.set_xlabel("grey level", labelpad=2); ah.set_ylabel("count")
for k, (img, ttl) in enumerate([(g, "micrograph"), (np.where(g > thr, z1, z0), "two levels $z_0$, $z_1$")]):
    ax = f.add_axes([0.06 + 0.47 * k, 0.0, 0.42, 0.42])
    ax.imshow(img, cmap="gray", vmin=0, vmax=255); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(ttl, fontsize=16, pad=3)
save(f, f"{OUT}/fig-tsai.svg")


# ---------------------------------------------------------------- 7. the weights x^p y^q (p, q <= 2) on [-1, 1]^2, y down
t = np.linspace(-1, 1, 120)
X, Y = np.meshgrid(t, t)          # row index = y, increasing downwards in the image
f, axs = fig(380, 420, nrows=3, ncols=3, gridspec_kw=dict(wspace=0.08, hspace=0.32))
names = {0: "", 1: "{}", 2: "{}^2"}
for q in range(3):
    for p in range(3):
        ax = axs[q, p]
        ax.imshow(X**p * Y**q, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        lab = ("x" + ("²" if p == 2 else "") if p else "") + ("y" + ("²" if q == 2 else "") if q else "")
        ax.set_title(f"$m_{{{p}{q}}}$: {lab or '1'}", fontsize=17, pad=3)
f.text(0.5, 0.06, "red +, blue −; y points down", ha="center", va="top", fontsize=16, color="#555")
save(f, f"{OUT}/fig-monomials.svg")
