"""Slide figures for classify/lec.qmd.

Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/classify/data.ts
     python3 -I tools/figs/classify/figs.py <dir>
Points come from the linclass demo's presets (dumped by data.ts), so the pictures and the
numbers match the demo. Figure sizes are the display size in px / 96 (13 pt text ≈ 17 px).
The micrograph: Transmission electron micrograph of a mesenchymal stem cell, Robert M. Hunt,
CC BY 3.0, via Wikimedia Commons (demos/src/fourier2d/cell256.png).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle, Ellipse, Circle, FancyBboxPatch
from scipy import ndimage as ndi
from skimage import filters, measure

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(R, "classify")
D = json.load(open(os.path.join(sys.argv[1], "classify.json")))
plt.rcParams.update({"font.size": 15, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#222", "xtick.color": "#555",
                     "ytick.color": "#555", "text.color": "#222", "svg.hashsalt": "classify"})
BLUE, RED, AMBER, GREY, INK = "#1f77b4", "#d62728", "#e6a700", "#888", "#222"
LBLUE, LRED = "#dce9f5", "#f8dede"


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), facecolor="white", **kw)
    plt.close(fig)
    print("wrote", name)


def pts(pid):
    P = D[pid]["points"]
    return np.array([[p["x"], p["y"]] for p in P]), np.array([p["c"] for p in P])


def scatter(ax, X, c, s=46):
    ax.scatter(X[c == 0, 0], X[c == 0, 1], s=s, c=BLUE, edgecolors="white", linewidths=0.8, zorder=3)
    ax.scatter(X[c == 1, 0], X[c == 1, 1], s=s, c=RED, edgecolors="white", linewidths=0.8, zorder=3)


def shade(ax, w, b, lo, hi, n=400):
    g = np.linspace(lo, hi, n)
    xx, yy = np.meshgrid(g, g)
    f = w[0] * xx + w[1] * yy + b
    ax.contourf(xx, yy, f, levels=[-1e9, 0, 1e9], colors=[LBLUE, LRED], zorder=0)
    ax.contour(xx, yy, f, levels=[0], colors=INK, linewidths=2, zorder=2)


def frame(ax, lo, hi, labels=("feature 1", "feature 2"), ticks=True):
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    if labels:
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
    if not ticks:
        ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def best_line(X, c, n_ang=1440):
    """Line with the fewest training errors (exhaustive over directions and thresholds)."""
    y = np.where(c == 1, 1, -1)
    best = (len(c) + 1, None)
    for a in np.linspace(0, np.pi, n_ang, endpoint=False):
        w = np.array([np.cos(a), np.sin(a)])
        p = X @ w
        o = np.argsort(p); ps, ys = p[o], y[o]
        for sgn in (1, -1):
            # threshold between consecutive projections: errors = (+ left of t) + (− right of t)
            ysg = ys * sgn
            left_pos = np.concatenate([[0], np.cumsum(ysg > 0)])
            right_neg = np.concatenate([np.cumsum((ysg < 0)[::-1])[::-1], [0]])
            e = left_pos + right_neg
            i = int(np.argmin(e))
            if e[i] < best[0]:
                t = ps[0] - 1 if i == 0 else (ps[-1] + 1 if i == len(ps) else (ps[i - 1] + ps[i]) / 2)
                best = (int(e[i]), (sgn * w, -sgn * t))
    return best


# ------------------------------------------------------------ 1. a classifier colours the feature space
X, c = pts("objects")
m = D["objects"]["mean"]
fig, ax = plt.subplots(figsize=(470 / 96, 430 / 96))
shade(ax, m["w"], m["b"], 0, 10)
scatter(ax, X, c)
frame(ax, 0, 10, ("area (100 px)", "mean intensity (×25)"))
ax.text(0.4, 9.3, "classified as nucleus (+1)", color=RED, fontsize=15, va="top")
ax.text(9.6, 0.5, "classified as debris (−1)", color=BLUE, fontsize=15, ha="right")
fig.tight_layout(pad=0.3)
save(fig, "fig-colour-space.svg")

# ------------------------------------------------------------ 2. pixel, object, window on the micrograph
cell = plt.imread(os.path.join(R, "demos/src/fourier2d/cell256.png")).astype(float)
if cell.ndim == 3:
    cell = cell[..., 0]
sm = filters.gaussian(cell, 2)
dark = sm < filters.threshold_otsu(sm)
lab = measure.label(ndi.binary_opening(dark, iterations=2))
props = sorted(measure.regionprops(lab), key=lambda r: -r.area)
# a medium-sized object away from the border
obj = next(r for r in props if 150 < r.area < 2500 and r.solidity > 0.85 and min(r.bbox[:2]) > 8 and r.bbox[2] < 248 and r.bbox[3] < 248)
fig, ax = plt.subplots(figsize=(470 / 96, 470 / 96))
ax.imshow(cell, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax.contour(lab == obj.label, levels=[0.5], colors=AMBER, linewidths=3)
oy, ox = obj.centroid
ax.text(obj.bbox[3] + 4, obj.bbox[0] - 4, "object", color="white", fontsize=17, fontweight="bold",
        bbox=dict(fc="#000000aa", ec="none", pad=2))
# window: a 48 × 48 box elsewhere
wy, wx = 30, 160
if abs(wy - oy) < 60 and abs(wx - ox) < 60:
    wy = 180
ax.add_patch(Rectangle((wx, wy), 48, 48, fill=False, ec="#56b4e9", lw=3))
ax.text(wx, wy - 5, "window", color="white", fontsize=17, fontweight="bold", bbox=dict(fc="#000000aa", ec="none", pad=2))
# pixel: crosshair
py, px = 200, 60
if abs(py - oy) < 40 and abs(px - ox) < 40:
    py = 120
ax.add_patch(Rectangle((px - 3, py - 3), 6, 6, fc="#ff4fa0", ec="white", lw=1.5))
ax.annotate("pixel", (px, py), (px + 22, py - 22), color="white", fontsize=17, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="white", lw=2), bbox=dict(fc="#000000aa", ec="none", pad=2))
ax.set_xticks([]); ax.set_yticks([])
fig.subplots_adjust(0, 0, 1, 1)
save(fig, "fig-three-ways.png", dpi=96)

# ------------------------------------------------------------ 3. a threshold is a linear classifier
rng = np.random.default_rng(3)
fig = plt.figure(figsize=(460 / 96, 560 / 96))
a1 = fig.add_axes([0.06, 0.82, 0.9, 0.09])
xs_b = rng.normal(65, 16, 14); xs_o = rng.normal(180, 18, 14); t = 120
xs_b = np.clip(xs_b, 5, 110); xs_o = np.clip(xs_o, 130, 250)
a1.axvspan(0, t, color=LBLUE, zorder=0); a1.axvspan(t, 255, color=LRED, zorder=0)
a1.scatter(xs_b, np.zeros_like(xs_b), c=BLUE, s=40, edgecolors="white", zorder=3)
a1.scatter(xs_o, np.zeros_like(xs_o), c=RED, s=40, edgecolors="white", zorder=3)
a1.axvline(t, color=INK, lw=2)
a1.text(t + 4, 0.55, "t", fontsize=16)
a1.text(25, 0.5, "f < 0", color=BLUE, fontsize=15, va="center")
a1.text(230, 0.5, "f > 0", color=RED, fontsize=15, va="center", ha="right")
a1.set_xlim(0, 255); a1.set_ylim(-0.6, 1.0); a1.set_yticks([])
a1.set_xlabel("1 feature (intensity x): a threshold", labelpad=2)
for s in ("top", "right", "left"):
    a1.spines[s].set_visible(False)
a2 = fig.add_axes([0.08, 0.05, 0.88, 0.6])
X2b = rng.normal([2.5, 2.5], 0.9, (18, 2)); X2o = rng.normal([6.5, 6.0], 0.9, (18, 2))
shade(a2, [1, 1.3], -10.5, 0, 9.5)
a2.scatter(X2b[:, 0], X2b[:, 1], c=BLUE, s=40, edgecolors="white", zorder=3)
a2.scatter(X2o[:, 0], X2o[:, 1], c=RED, s=40, edgecolors="white", zorder=3)
a2.text(0.3, 9.0, "f > 0", color=RED, fontsize=15, va="top")
a2.text(9.2, 0.3, "f < 0", color=BLUE, fontsize=15, ha="right")
a2.text(1.6, 7.1, "f = 0", fontsize=15, rotation=-37, rotation_mode="anchor")
frame(a2, 0, 9.5, None, ticks=False)
a2.set_xlabel("x₁", labelpad=1); a2.set_ylabel("x₂", labelpad=1)
a2.set_title("2 features: a line", fontsize=15, pad=4)
save(fig, "fig-threshold.svg")

# ------------------------------------------------------------ 4. scaling (w, b): the same line, f doubles
fig, axs = plt.subplots(1, 2, figsize=(600 / 96, 330 / 96))
g = np.linspace(-0.5, 4, 300)
xx, yy = np.meshgrid(g, g)
for ax, k in zip(axs, (1, 2)):
    f = k * (xx + 2 * yy - 4)
    lv = [-2, -1, 1, 2]
    cs = ax.contour(xx, yy, f, levels=lv, colors=GREY, linewidths=1.2, linestyles="dashed")
    ax.contour(xx, yy, f, levels=[0], colors=INK, linewidths=2.2)
    ax.plot(1, 1, "o", c=BLUE, ms=8, mec="white"); ax.plot(3, 2, "o", c=RED, ms=8, mec="white")
    fa, fb = k * (1 + 2 - 4), k * (3 + 4 - 4)
    ax.text(1.0, 0.75, f"A: f = {fa:g}".replace("-", "−"), fontsize=14.5, ha="center", va="top", bbox=dict(fc="white", ec="none", pad=1))
    ax.text(3.0, 2.2, f"B: f = {fb:g}", fontsize=14.5, ha="center", bbox=dict(fc="white", ec="none", pad=1))
    frame(ax, -0.5, 4, None, ticks=False)
    lab_ = "(w, b) = ((1, 2), −4)" if k == 1 else "× 2: ((2, 4), −8)"
    ax.set_title(lab_, fontsize=17, pad=4)
fig.tight_layout(pad=0.4, w_pad=1.0)
save(fig, "fig-scaling.svg")

# ------------------------------------------------------------ 5. distance to a hyperplane (math)
fig, ax = plt.subplots(figsize=(400 / 96, 360 / 96))
w = np.array([1.0, 2.0]); u = w / np.linalg.norm(w)
x0 = np.array([2.0, 1.0])               # on x1 + 2 x2 = 4
x = np.array([3.0, 2.0])                # point B
foot = x - (u @ (x - x0)) * u
L = np.array([[-0.5, 2.25], [4.5, -0.25]])
ax.plot(L[:, 0], L[:, 1], c=INK, lw=2.2)
ax.text(4.2, -0.05, "f = 0", fontsize=15, ha="right", va="top")
ax.add_patch(FancyArrowPatch(x0, x, arrowstyle="-|>", mutation_scale=16, color=GREY, lw=1.6))
ax.add_patch(FancyArrowPatch(x0, x0 + 0.9 * u, arrowstyle="-|>", mutation_scale=18, color=AMBER, lw=2.6))
ax.text(*(x0 + 0.9 * u + np.array([-0.55, 0.05])), "w/‖w‖", color="#9a7000", fontsize=15)
ax.plot(*zip(foot, x), c=RED, lw=2.4, ls=(0, (4, 2)))
ax.plot(*x0, "o", c=INK, ms=7); ax.text(x0[0] + 0.1, x0[1] - 0.35, "x₀", fontsize=16)
ax.plot(*x, "o", c=RED, ms=9, mec="white"); ax.text(x[0] + 0.12, x[1] + 0.05, "x", fontsize=16)
ax.plot(*foot, "s", c=INK, ms=5)
mid = (foot + x) / 2
ax.text(mid[0] + 0.12, mid[1] - 0.25, "f(x)/‖w‖", color=RED, fontsize=15)
ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 3.2); ax.set_aspect("equal"); ax.axis("off")
fig.subplots_adjust(0, 0, 1, 1)
save(fig, "fig-distance.svg")

# ------------------------------------------------------------ 6. many zero-error lines; one grazes a point
X, c = pts("many")
y = np.where(c == 1, 1, -1)
fig, ax = plt.subplots(figsize=(480 / 96, 440 / 96))
g = np.linspace(0, 10, 2)
zero = []
for a in np.linspace(-1.2, 0.2, 60):                # directions of the normal
    wv = np.array([np.cos(a + np.pi / 4), np.sin(a + np.pi / 4)])
    p = X @ wv
    lo, hi = p[c == 0].max(), p[c == 1].min()
    if hi > lo:
        zero.append((wv, lo, hi))
sel = zero[::max(1, len(zero) // 5)]
for wv, lo, hi in sel:
    for tt in (lo + 0.3 * (hi - lo), lo + 0.7 * (hi - lo)):
        xx_ = np.linspace(0, 10, 50); yy_ = (tt - wv[0] * xx_) / wv[1]
        ax.plot(xx_, yy_, c="#bbbbbb", lw=1.3, zorder=1)
# a grazing line: the middle direction, threshold just above the closest blue point
wv, lo, hi = zero[len(zero) // 2]
tt = lo + 0.04 * (hi - lo)
xx_ = np.linspace(0, 10, 50); yy_ = (tt - wv[0] * xx_) / wv[1]
ax.plot(xx_, yy_, c=AMBER, lw=3, zorder=2)
scatter(ax, X, c)
ib = np.argmax(np.where(c == 0, X @ wv, -np.inf))
newp = X[ib] + 0.45 * wv
ax.scatter(*newp, s=70, facecolors="white", edgecolors=BLUE, linewidths=2.2, zorder=4)
ax.annotate("new point\nnear a blue one:\nwrong side", newp, newp + np.array([0.6, -3.1]), fontsize=14.5,
            arrowprops=dict(arrowstyle="-", color=INK, lw=1))
nz = sum(1 for _ in zero)
ax.text(0.3, 9.7, "grey: lines with 0 errors\nyellow: 0 errors, but it grazes a point", fontsize=14.5, va="top", bbox=dict(fc="white", ec="none", alpha=0.9, pad=2), zorder=6)
frame(ax, 0, 10, ("feature 1", "feature 2"), ticks=False)
fig.tight_layout(pad=0.3)
save(fig, "fig-many-lines.svg")

# ------------------------------------------------------------ 7. nearest class mean, the worked example
mp, mm = np.array([4.0, 3.0]), np.array([1.0, 1.0])
w = mp - mm; mid = (mp + mm) / 2; b = -(w @ mid)
assert np.isclose(b, -11.5)
fig, ax = plt.subplots(figsize=(440 / 96, 400 / 96))
shade(ax, w, b, -0.5, 5.5)
rng = np.random.default_rng(5)
cp = mp + rng.normal(0, 0.55, (12, 2)); cm = mm + rng.normal(0, 0.55, (12, 2))
ax.scatter(cp[:, 0], cp[:, 1], c=RED, s=30, alpha=0.5, edgecolors="none", zorder=2)
ax.scatter(cm[:, 0], cm[:, 1], c=BLUE, s=30, alpha=0.5, edgecolors="none", zorder=2)
ax.plot(*zip(mm, mp), c=GREY, lw=1.5, ls="dashed", zorder=2)
ax.plot(*mp, "P", c=RED, ms=14, mec="white", zorder=4); ax.plot(*mm, "P", c=BLUE, ms=14, mec="white", zorder=4)
ax.plot(*mid, "o", c=INK, ms=7, zorder=4)
ax.text(mp[0] + 0.25, mp[1] - 0.15, "μ₊ (4, 3)", fontsize=15, color=RED, va="top")
ax.text(mm[0] - 0.2, mm[1] - 0.55, "μ₋ (1, 1)", fontsize=15, color=BLUE)
ax.text(mid[0] + 0.15, mid[1] - 0.45, "midpoint (2.5, 2)", fontsize=14.5)
ax.add_patch(FancyArrowPatch(mid, mid + 0.3 * w, arrowstyle="-|>", mutation_scale=18, color=AMBER, lw=2.6, zorder=5))
ax.text(*(mid + 0.3 * w + np.array([-2.0, 0.45])), "w = (3, 2)", color="#9a7000", fontsize=15)
frame(ax, -0.5, 5.5, ("x₁", "x₂"))
fig.tight_layout(pad=0.3)
save(fig, "fig-nearest-mean.svg")

# ------------------------------------------------------------ 8. generative vs discriminative
rng = np.random.default_rng(8)
Sig = np.array([[1.0, 0.5], [0.5, 0.8]])
Lc = np.linalg.cholesky(Sig)
mu_m, mu_p = np.array([3.0, 3.2]), np.array([6.2, 5.6])
Am = mu_m + rng.normal(size=(40, 2)) @ Lc.T; Ap = mu_p + rng.normal(size=(40, 2)) @ Lc.T
Si = np.linalg.inv(Sig); wl = Si @ (mu_p - mu_m); bl = -0.5 * (mu_p @ Si @ mu_p - mu_m @ Si @ mu_m)
fig, axs = plt.subplots(1, 2, figsize=(620 / 96, 340 / 96))
g = np.linspace(0, 9.5, 300); xx, yy = np.meshgrid(g, g)
for k, ax in enumerate(axs):
    ax.scatter(Am[:, 0], Am[:, 1], c=BLUE, s=22, edgecolors="none", alpha=0.85, zorder=3)
    ax.scatter(Ap[:, 0], Ap[:, 1], c=RED, s=22, edgecolors="none", alpha=0.85, zorder=3)
    if k == 0:
        for mu, col in ((mu_m, BLUE), (mu_p, RED)):
            d = np.stack([xx - mu[0], yy - mu[1]], -1)
            q = np.einsum("...i,ij,...j->...", d, Si, d)
            ax.contour(xx, yy, q, levels=[1, 4, 9], colors=col, linewidths=1.4, zorder=2)
        ax.set_title("generative", fontsize=17, pad=4)
    else:
        f = wl[0] * xx + wl[1] * yy + bl
        ax.contourf(xx, yy, f, levels=[-1e9, 0, 1e9], colors=[LBLUE, LRED], zorder=0)
        ax.contour(xx, yy, f, levels=[0], colors=INK, linewidths=2, zorder=2)
        ax.set_title("discriminative", fontsize=17, pad=4)
    frame(ax, 0, 9.5, None, ticks=False)
f = wl[0] * xx + wl[1] * yy + bl
axs[0].contour(xx, yy, f, levels=[0], colors=INK, linewidths=1.6, linestyles="dashed", zorder=2)
fig.tight_layout(pad=0.6, w_pad=0.8)
save(fig, "fig-generative.svg")

# ------------------------------------------------------------ 9. one neuron = a linear classifier
fig, ax = plt.subplots(figsize=(520 / 96, 270 / 96))
ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off"); ax.set_aspect("equal")
ins = [("x₁", 4.3, "w₁"), ("x₂", 2.6, "w₂"), ("1", 0.9, "b")]
S = np.array([5.0, 2.6])
for name, yy_, wn in ins:
    ax.add_patch(Circle((1.0, yy_), 0.45, fc="white", ec=INK, lw=1.5))
    ax.text(1.0, yy_, name, ha="center", va="center", fontsize=17)
    ax.add_patch(FancyArrowPatch((1.45, yy_), S + np.array([-0.75, (yy_ - S[1]) * 0.2]), arrowstyle="-|>",
                                 mutation_scale=14, color=GREY, lw=1.5))
    mx = 1.45 + 0.42 * (S[0] - 0.75 - 1.45)
    ax.text(mx, yy_ + 0.42 * (S[1] - yy_) + 0.25, wn, fontsize=16, color="#9a7000", ha="center")
ax.add_patch(Circle(S, 0.75, fc="#f3f3f3", ec=INK, lw=1.8))
ax.text(*S, "Σ", ha="center", va="center", fontsize=22)
ax.text(S[0], S[1] - 1.25, "w·x + b", ha="center", fontsize=15)
ax.add_patch(FancyArrowPatch(S + np.array([0.75, 0]), (7.0, S[1]), arrowstyle="-|>", mutation_scale=14, color=GREY, lw=1.5))
ax.add_patch(FancyBboxPatch((7.05, S[1] - 0.55), 1.3, 1.1, boxstyle="round,pad=0.05", fc="white", ec=INK, lw=1.5))
xs_ = np.array([7.25, 7.7, 7.7, 8.15]); ys_ = np.array([S[1] - 0.3, S[1] - 0.3, S[1] + 0.3, S[1] + 0.3])
ax.plot(xs_, ys_, c=INK, lw=2)
ax.text(7.7, S[1] - 0.95, "sign", ha="center", fontsize=15)
ax.add_patch(FancyArrowPatch((8.4, S[1]), (9.2, S[1]), arrowstyle="-|>", mutation_scale=14, color=GREY, lw=1.5))
ax.text(9.3, S[1], "±1", va="center", fontsize=17)
fig.subplots_adjust(0, 0, 1, 1)
save(fig, "fig-neuron.svg")

# ------------------------------------------------------------ 10. when no line works: XOR, ring
fig, axs = plt.subplots(1, 2, figsize=(600 / 96, 320 / 96))
for ax, pid, lab_ in zip(axs, ("xor", "ring"), ("XOR", "ring")):
    X, c = pts(pid)
    e, (wv, b) = best_line(X, c)
    print(pid, "best line errors", e, "of", len(c))
    shade(ax, wv, b, 0, 10)
    y = np.where(c == 1, 1, -1)
    wrong = y * (X @ wv + b) <= 0
    scatter(ax, X, c, s=28)
    ax.scatter(X[wrong, 0], X[wrong, 1], s=110, facecolors="none", edgecolors=INK, linewidths=1.3, zorder=4)
    frame(ax, 0, 10, None, ticks=False)
    ax.set_title(f"{lab_}: {e} of {len(c)} wrong", fontsize=17, pad=4)
fig.tight_layout(pad=0.6, w_pad=0.8)
save(fig, "fig-no-line.svg")
