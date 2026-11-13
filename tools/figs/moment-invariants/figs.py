# Figures for moment-invariants/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/moment-invariants/figs.py
# The L and its Hu invariants come from the moments demo's code (data.ts, via node).
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "label-connectivity"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Ellipse
from matplotlib.path import Path
from labstyle import setup, fig, save, run_ts, BLUE, ORANGE, GREEN, RED, GREY, INK

setup()
OUT = "moment-invariants"
FILL = "#a6c4f4"
D = run_ts(HERE)
L = np.array(D["L"], float)


def moments_of(poly, n=600, pad=0.0):
    """Area, centroid and central moments mu[p][q] (p+q <= 3) by fine sampling (y as given)."""
    x0, y0 = poly.min(0) - 1; x1, y1 = poly.max(0) + 1
    h = max(x1 - x0, y1 - y0) / n
    xs = np.arange(x0, x1, h) + h / 2; ys = np.arange(y0, y1, h) + h / 2
    X, Y = np.meshgrid(xs, ys)
    m = Path(poly).contains_points(np.c_[X.ravel(), Y.ravel()]).astype(float) * h * h
    X, Y = X.ravel(), Y.ravel()
    A = m.sum(); cx = (m * X).sum() / A; cy = (m * Y).sum() / A
    mu = {(p, q): (m * (X - cx)**p * (Y - cy)**q).sum() for p in range(4) for q in range(4) if p + q <= 3}
    return A, np.array([cx, cy]), mu


def rot(P, a, c=(0, 0)):
    c = np.asarray(c); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (P - c) @ R.T + c


def flat(ax):
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


# ---------------------------------------------------------------- 1. translation
Ly = np.c_[L[:, 0], 64 - L[:, 1]]          # draw with y up
A, c, _ = moments_of(Ly)
f, ax = fig(420, 330)
for k, (d, col) in enumerate([((0, 0), BLUE), ((34, -14), ORANGE)]):
    P = Ly + d; cc = c + d
    ax.add_patch(Polygon(P, closed=True, fc=col, alpha=0.25, ec=col, lw=2))
    ax.plot(*cc, "+", color=INK, ms=16, mew=2.5)
    corner = P[0]
    ax.annotate("", xy=corner, xytext=cc, arrowprops=dict(arrowstyle="-|>", color=INK, lw=2, mutation_scale=16))
ax.annotate("", xy=(c[0] + 30, c[1] - 10), xytext=(c[0] + 6, c[1] + 2),
            arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2.5, ls="--", mutation_scale=18))
ax.text(c[0] + 18, c[1] + 2, "move", color=GREY, fontsize=17, rotation=-25)
ax.text(46, 52, "same offsets from\nthe centroid (+)", fontsize=17, ha="left", va="top")
ax.set_xlim(12, 86); ax.set_ylim(-4, 54); flat(ax)
save(f, f"{OUT}/fig-translate.svg")


# ---------------------------------------------------------------- 2. scale
f, ax = fig(560, 260)
x = 0
for (a, b) in [(4, 2), (8, 4), (12, 6)]:
    for i in range(a):
        for j in range(b):
            ax.add_patch(Rectangle((x + i, j), 1, 1, fc=FILL, ec="white", lw=1))
    ax.add_patch(Rectangle((x, 0), a, b, fill=False, ec=BLUE, lw=2))
    mu20 = a**3 * b / 12; eta = mu20 / (a * b)**2
    ax.text(x + a / 2, -0.8, f"{a} × {b}", ha="center", va="top", fontsize=18)
    ax.text(x + a / 2, -2.6, f"$\\mu_{{20}}$ = {mu20:.3g}", ha="center", va="top", fontsize=17)
    ax.text(x + a / 2, -4.4, f"$\\eta_{{20}}$ = {eta:.3f}", ha="center", va="top", fontsize=17, color=RED)
    print("scale", a, b, mu20, eta)
    x += a + 3
ax.set_xlim(-1.5, x - 1.5); ax.set_ylim(-6.2, 6.5); flat(ax)
save(f, f"{OUT}/fig-scale.svg")


# ---------------------------------------------------------------- 3. rotation at order 2
M = np.array([[10, 3], [3, 4.0]])
f, axs = fig(500, 330, ncols=2, gridspec_kw=dict(wspace=0.05))
for ax, a in zip(axs, [0, 30]):
    t = np.radians(a); Q = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    R = Q @ M @ Q.T
    w, V = np.linalg.eigh(R)
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    ax.add_patch(Ellipse((0, 0), 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]), angle=ang, fc=FILL, ec=BLUE, lw=2))
    for k, col in [(1, RED), (0, GREEN)]:
        v = V[:, k] * np.sqrt(w[k])
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5, mutation_scale=16))
    ax.axhline(0, color="#ccc", lw=1, zorder=0); ax.axvline(0, color="#ccc", lw=1, zorder=0)
    ax.set_xlim(-3.8, 3.8); ax.set_ylim(-3.8, 3.8); flat(ax)
    ax.set_title("as given" if a == 0 else "rotated by 30°", fontsize=18, pad=2)
    ax.text(0, -4.0, f"$\\mu$: {R[0,0]:.2f}, {R[0,1]:.2f}, {R[1,1]:.2f}", ha="center", va="top", fontsize=17)
    ax.text(0, -5.2, f"trace {np.trace(R):.0f}", ha="center", va="top", fontsize=17, color=RED)
    ax.text(0, -6.4, f"$\\lambda$: {w[1]:.2f}, {w[0]:.2f}", ha="center", va="top", fontsize=17, color=RED)
    print("rot2", a, R.round(3), w)
save(f, f"{OUT}/fig-rot2.svg")


# ---------------------------------------------------------------- 4. mirror: three points
pts = np.array([[3, 1], [-1, 2], [-2, -3]], float)
z = pts[:, 0] + 1j * pts[:, 1]
sets = [("original", z, BLUE, "o"), ("rotated 90°", 1j * z, ORANGE, "s"), ("mirrored", np.conj(z), GREEN, "^")]
f, axs = fig(520, 300, ncols=2)
ax = axs[0]
for name, zz, col, mk in sets:
    ax.scatter(zz.real, zz.imag, s=110, color=col, marker=mk, label=name, zorder=3, edgecolor="white", lw=1)
ax.axhline(0, color="#ccc", lw=1); ax.axvline(0, color="#ccc", lw=1)
ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6); flat(ax)
ax.set_title("three points", fontsize=18, pad=4)
ax.legend(loc="lower center", bbox_to_anchor=(1.05, -0.32), ncol=3, frameon=False, fontsize=17, handletextpad=0.1, columnspacing=0.8)
ax = axs[1]
for name, zz, col, mk in sets:
    c30 = (zz**3).sum()
    ax.annotate("", xy=(c30.real, c30.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=3, mutation_scale=18))
    off = {"original": (8, 6), "rotated 90°": (8, -4), "mirrored": (8, -4)}[name]
    ax.text(c30.real + off[0] * 0, c30.imag + off[1], f"{c30.real:.0f} {'+' if c30.imag >= 0 else '−'} {abs(c30.imag):.0f}i",
            color=col, fontsize=18, ha="center" if name == "rotated 90°" else "left", va="center")
    print("c30", name, c30, "c20", (zz**2).sum())
ax.axhline(0, color="#ccc", lw=1); ax.axvline(0, color="#ccc", lw=1)
ax.set_xlim(-30, 170); ax.set_ylim(-95, 45); flat(ax)
ax.set_title("$c_{30}$", fontsize=19, pad=4)
save(f, f"{OUT}/fig-mirror.svg")


# ---------------------------------------------------------------- 5. log scale of Hu's invariants
hu = np.array(D["hu"]); lg = np.array(D["log"]); lgr = np.array(D["logRot"])
print("hu", hu, "log", lg.round(2))
f, axs = fig(450, 560, nrows=2, gridspec_kw=dict(hspace=0.45))
k = np.arange(1, 8)
ax = axs[0]
ax.bar(k, hu, color=BLUE, width=0.6); ax.axhline(0, color=INK, lw=1)
ax.set_xticks(k); ax.set_xticklabels([f"φ{i}" for i in k]); ax.set_ylabel("φ")
ax.text(1.4, hu[0] * 0.95, f"{hu[0]:.2f}", fontsize=16, va="top")
ax.text(4.8, hu[0] * 0.45, "φ2 … φ7 hardly visible", fontsize=17, ha="center")
ax.set_title("raw values", fontsize=17)
ax = axs[1]
ax.bar(k - 0.18, lg, width=0.36, color=BLUE, label="L")
ax.bar(k + 0.18, lgr, width=0.36, color=ORANGE, label="L rotated 50°")
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(k); ax.set_xticklabels([f"φ{i}" for i in k]); ax.set_ylabel("−sign(φ) log₁₀|φ|")
ax.text(1, lg[0] + 0.25, f"{lg[0]:.2f}", ha="center", fontsize=15)
ax.text(5, lg[4] - 0.25, f"{lg[4]:.2f}", ha="center", va="top", fontsize=15)
ax.set_ylim(-7.6, 6.2)
ax.legend(loc="upper right", frameon=False, fontsize=14)
ax.set_title("log scale", fontsize=17)
save(f, f"{OUT}/fig-logscale.svg")


# ---------------------------------------------------------------- 6. pose normalisation in steps
def pear(cx, cy, sz, n=160):   # pearPoly of demos/src/moments/mode-higher.ts
    rB, b = 0.6 * sz, 1.05 * sz
    t = 2 * np.pi * np.arange(n) / n
    d = rB**2 - b**2 * np.sin(t)**2
    far = np.where(d >= 0, b * np.cos(t) + np.sqrt(np.maximum(d, 0)), -np.inf)
    r = np.maximum(sz, far)
    return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]


P0 = rot(pear(0, 0, 12), np.radians(-140)) + np.array([40, 34])
A, c, mu = moments_of(P0)
P1 = P0 - c
P2 = P1 / np.sqrt(A)
_, _, mu2 = moments_of(P2)
th = 0.5 * np.arctan2(2 * mu2[(1, 1)], mu2[(2, 0)] - mu2[(0, 2)])
P3 = rot(P2, -th)
_, _, mu3 = moments_of(P3)
P4 = rot(P3, np.pi) if mu3[(3, 0)] < 0 else P3
_, _, mu4 = moments_of(P4)
print("pose: area", A, "theta", np.degrees(th), "mu30 after rot", mu3[(3, 0)], "final mu30", mu4[(3, 0)], "mu11", mu4[(1, 1)])
assert mu3[(3, 0)] < 0      # the example should need the last step
steps = [("1  centroid to 0", P0, P1, None), ("2  area 1", None, P2, None),
         ("3  major axis on x", P2, P3, None), ("4  flip: $\\mu_{30}$ > 0", P3, P4, None)]
f, axs = fig(470, 450, nrows=2, ncols=2, gridspec_kw=dict(hspace=0.22, wspace=0.08))
for k, (ax, (ttl, prev, cur, _)) in enumerate(zip(axs.ravel(), steps)):
    if prev is not None:
        ax.add_patch(Polygon(prev, closed=True, fill=False, ec=GREY, lw=1.6, ls="--"))
    ax.add_patch(Polygon(cur, closed=True, fc=FILL, ec=BLUE, lw=2))
    ax.axhline(0, color="#bbb", lw=1, zorder=0); ax.axvline(0, color="#bbb", lw=1, zorder=0)
    if k == 0:
        ax.plot(*c, "+", color=INK, ms=14, mew=2.5)
        ax.set_xlim(-25, 60); ax.set_ylim(-30, 55)
    else:
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    if k == 1:
        ax.text(0.98, -0.98, "scale ÷ $\\sqrt{\\mathrm{area}}$", ha="right", va="bottom", fontsize=15, color="#555")
    flat(ax)
    ax.set_title(ttl, fontsize=16, loc="left", pad=3)
save(f, f"{OUT}/fig-pose.svg")


# ---------------------------------------------------------------- 7. overview: which numbers survive which transformation
Lc = Ly - np.array([32, 32])
views = [("original", Lc, "reference\n(dashed)"),
         ("translated", Lc + np.array([7, -6]), "m changes\nμ, η, φ same"),
         ("scaled", Lc * 1.25, "m, μ change\nη, φ same"),
         ("rotated 50°", rot(Lc, np.radians(50)), "m, μ, η change\nφ same"),
         ("mirrored", np.c_[-Lc[:, 0], Lc[:, 1]], "m, μ, η change\nφ₇ flips sign")]
f, axs = fig(900, 290, ncols=5, gridspec_kw=dict(wspace=0.1))
for ax, (name, P, txt) in zip(axs, views):
    ax.add_patch(Polygon(P, closed=True, fc=FILL, ec=BLUE, lw=2))
    if name != "original":
        ax.add_patch(Polygon(Lc, closed=True, fill=False, ec=ORANGE, lw=2, ls="--"))
    ax.set_xlim(-30, 30); ax.set_ylim(-30, 30); flat(ax)
    ax.set_title(name, fontsize=19, pad=2)
    ax.text(0, -32, txt, ha="center", va="top", fontsize=17)
save(f, f"{OUT}/fig-invariants.svg")


# ---------------------------------------------------------------- 8. complex moments turn (p - q) times as fast
def blob(n=240):   # blobPoly of demos/src/zernikebasis/compute.ts, centred, y up
    t = 2 * np.pi * np.arange(n) / n
    r = 17 * (1 + 0.28 * np.cos(t - 2.6) + 0.16 * np.cos(2 * t + 0.9) + 0.12 * np.cos(3 * t - 0.4))
    return np.c_[r * np.cos(t), r * np.sin(t)]


def cmom(P, p, q, n=500):
    x0, y0 = P.min(0) - 1; x1, y1 = P.max(0) + 1
    h = max(x1 - x0, y1 - y0) / n
    X, Y = np.meshgrid(np.arange(x0, x1, h) + h / 2, np.arange(y0, y1, h) + h / 2)
    m = Path(P).contains_points(np.c_[X.ravel(), Y.ravel()]).astype(float)
    X, Y = X.ravel(), Y.ravel()
    cx, cy = (m * X).sum() / m.sum(), (m * Y).sum() / m.sum()
    z = (X - cx) + 1j * (Y - cy)
    return (m * z**p * np.conj(z)**q).sum()


B = blob()
angles = [0, 20, 40, 60, 80]
cols = plt.cm.viridis(np.linspace(0, 0.9, len(angles)))
f, axs = fig(860, 300, ncols=3, gridspec_kw=dict(wspace=0.15))
for ax, (p, q, ttl) in zip(axs, [(2, 0, "$c_{20}$ turns by 2α"), (2, 1, "$c_{21}$ turns by α"), (1, 1, "$c_{11}$ stays")]):
    cs = [cmom(rot(B, np.radians(a)), p, q) for a in angles]
    sc = max(abs(c) for c in cs)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ec="#ccc", lw=1))
    ax.axhline(0, color="#ddd", lw=1, zorder=0); ax.axvline(0, color="#ddd", lw=1, zorder=0)
    for a, c, col in zip(angles, cs, cols):
        v = c / sc
        ax.annotate("", xy=(v.real, v.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5, mutation_scale=16))
        if (p, q) != (1, 1):
            ax.text(1.22 * v.real, 1.22 * v.imag, f"{a}°", ha="center", va="center", fontsize=15, color=col)
    if (p, q) == (1, 1):
        ax.text(0, -0.45, "all five angles", ha="center", fontsize=15, color="#555")
    print("phase", p, q, [f"{np.degrees(np.angle(c)):.0f}" for c in cs])
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); flat(ax)
    ax.set_title(ttl, fontsize=19, pad=2)
# (a blob rotated by 0°, 20°, …, 80°; arrows scaled to the largest per dial: said in the alt text)
save(f, f"{OUT}/fig-phase.svg")
