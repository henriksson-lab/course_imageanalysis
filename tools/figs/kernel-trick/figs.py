"""Slide figures for kernel-trick/lec.qmd.

Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/kernel-trick/data.ts
     python3 -I tools/figs/kernel-trick/figs.py <dir>
Points, planes, α values and the (γ, C) grid come from the demos' presets and the course's
SVM solver (data.ts). Figure sizes are the display size in px / 96.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "kernel-trick"))
D = json.load(open(os.path.join(sys.argv[1], "kernel-trick.json")))
plt.rcParams.update({"font.size": 15, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#222", "xtick.color": "#555",
                     "ytick.color": "#555", "text.color": "#222", "svg.hashsalt": "kernel-trick"})
BLUE, RED, AMBER, GREY, INK = "#1f77b4", "#d62728", "#e6a700", "#888", "#222"
LBLUE, LRED, STREET = "#dce9f5", "#f8dede", "#e4e4e4"


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), facecolor="white", **kw)
    plt.close(fig)
    print("wrote", name)


def arr(P):
    return np.array([[p["x"], p["y"]] for p in P]), np.array([p["c"] for p in P])


def frame(ax, lo, hi):
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------ 1. lift the ring onto the paraboloid, cut with a plane
L = D["lift"]; X, c = arr(L["points"]); w = np.array(L["w"]); b = L["b"]
z = (X ** 2).sum(1)
fig = plt.figure(figsize=(640 / 96, 330 / 96))
a1 = fig.add_axes([0.02, 0.04, 0.38, 0.8])
g = np.linspace(-1, 1, 400); xx, yy = np.meshgrid(g, g)
F = w[0] * xx + w[1] * yy + w[2] * (xx ** 2 + yy ** 2) + b
a1.contourf(xx, yy, F, levels=[-1e9, 0, 1e9], colors=[LBLUE, LRED], zorder=0)
a1.contour(xx, yy, F, levels=[0], colors=INK, linewidths=2.2)
a1.scatter(X[c == 0, 0], X[c == 0, 1], s=22, c=BLUE, edgecolors="white", linewidths=0.5, zorder=3)
a1.scatter(X[c == 1, 0], X[c == 1, 1], s=22, c=RED, edgecolors="white", linewidths=0.5, zorder=3)
frame(a1, -1, 1)
a1.set_title("in 2D: a circle", fontsize=17, pad=4)
a2 = fig.add_axes([0.40, -0.04, 0.6, 0.98], projection="3d")
u = np.linspace(0, 1, 18); t = np.linspace(0, 2 * np.pi, 48)
U, T = np.meshgrid(u, t)
a2.plot_wireframe(U * np.cos(T), U * np.sin(T), U ** 2, color="#c8c8c8", lw=0.5, rstride=3, cstride=3)
# the plane w·(x1, x2, x3) + b = 0, i.e. x3 = −(w1 x1 + w2 x2 + b)/w3
P1, P2 = np.meshgrid(np.linspace(-1, 1, 2), np.linspace(-1, 1, 2))
P3 = -(w[0] * P1 + w[1] * P2 + b) / w[2]
a2.plot_surface(P1, P2, P3, color=AMBER, alpha=0.28, linewidth=0)
# cut: points on the paraboloid where F = 0
cs = plt.figure().add_subplot().contour(xx, yy, F, levels=[0]).allsegs[0]
plt.close()
for seg in cs:
    a2.plot(seg[:, 0], seg[:, 1], (seg ** 2).sum(1), c=INK, lw=2.2)
a2.scatter(X[c == 0, 0], X[c == 0, 1], z[c == 0], s=18, c=BLUE, depthshade=False)
a2.scatter(X[c == 1, 0], X[c == 1, 1], z[c == 1], s=18, c=RED, depthshade=False)
a2.view_init(elev=14, azim=-60)
a2.set_box_aspect(None, zoom=1.25)
a2.set_xticks([]); a2.set_yticks([]); a2.set_zticks([])
a2.set_xlabel("x₁", labelpad=-12); a2.set_ylabel("x₂", labelpad=-12); a2.set_zlabel("x₁² + x₂²", labelpad=-10)
fig.text(0.70, 0.90, "lifted: a flat plane", ha="center", fontsize=17)
save(fig, "fig-lift.svg")

# ------------------------------------------------------------ 2. f is a weighted sum over the support vectors
B = D["byhand"]; X, c = arr(B["points"]); al = np.array(B["alpha"]); w = np.array(B["w"]); b = B["b"]
fig, ax = plt.subplots(figsize=(420 / 96, 400 / 96))
g = np.linspace(-1, 4, 400); xx, yy = np.meshgrid(g, g)
F = w[0] * xx + w[1] * yy + b
ax.contourf(xx, yy, F, levels=[-1e9, -1, 1, 1e9], colors=[LBLUE, STREET, LRED], zorder=0)
ax.contour(xx, yy, F, levels=[0], colors=INK, linewidths=2.4)
ax.contour(xx, yy, F, levels=[-1], colors=BLUE, linewidths=1.8, linestyles=[(0, (6, 4))])
ax.contour(xx, yy, F, levels=[1], colors=RED, linewidths=1.8, linestyles=[(0, (6, 4))])
for (px, py), cc, a in zip(X, c, al):
    ax.scatter(px, py, s=80, c=RED if cc == 1 else BLUE, edgecolors="white", zorder=4)
    if a > 1e-6:
        ax.scatter(px, py, s=260, facecolors="none", edgecolors=INK, linewidths=1.8, zorder=5)
    ax.text(px + 0.18, py + 0.2, f"α = {a:.2g}" if a > 1e-6 else "α = 0", fontsize=15, zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
frame(ax, -1, 4)
fig.tight_layout(pad=0.3)
save(fig, "fig-alpha.svg")

# ------------------------------------------------------------ 3. the kernel trick: two routes to the same number
x = np.array([1.0, 2.0]); zv = np.array([3.0, 1.0])
phi = lambda v: np.array([v[0] ** 2, np.sqrt(2) * v[0] * v[1], v[1] ** 2])
lifted = phi(x) @ phi(zv); direct = (x @ zv) ** 2
assert np.isclose(lifted, 25) and np.isclose(direct, 25)
fig, ax = plt.subplots(figsize=(600 / 96, 300 / 96))
ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")


def box(xc, yc, txt, w_=3.4, h=1.15, fc="white", ec=INK, fs=15):
    ax.add_patch(FancyBboxPatch((xc - w_ / 2, yc - h / 2), w_, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.6))
    ax.text(xc, yc, txt, ha="center", va="center", fontsize=fs)


def arrow(p, q, label="", col=GREY, above=True):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, color=col, lw=1.8))
    if label:
        m = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + (0.35 if above else -0.35))
        ax.text(*m, label, ha="center", va="bottom" if above else "top", fontsize=13.5, color="#555")


box(1.35, 3.0, "x = (1, 2)\nz = (3, 1)", w_=2.3, h=1.6)
box(6.4, 4.9, "φ(x) = (1, 2√2, 4)\nφ(z) = (9, 3√2, 1)", w_=4.4, h=1.5, fc="#f3f3f3")
box(10.65, 4.9, "9 + 12 + 4\n= 25", w_=2.4, h=1.5, fc="#f3f3f3")
box(6.4, 1.1, "x·z = 3 + 2 = 5", w_=4.0, fc="#fdf1e2", ec=AMBER)
box(10.6, 1.1, "5² = 25", w_=2.2, fc="#fdf1e2", ec=AMBER)
arrow((2.55, 3.85), (4.1, 4.75), "lift")
arrow((8.65, 4.9), (9.45, 4.9))
arrow((2.55, 2.15), (4.35, 1.2), "")
arrow((8.45, 1.1), (9.45, 1.1))
ax.text(7.6, 3.65, "dot product up there: 6 multiplications", ha="center", fontsize=13.5, color="#555")
ax.text(7.6, 2.05, "kernel function (x·z)²: 2 multiplications", ha="center", fontsize=13.5, color="#9a7000")
fig.subplots_adjust(0, 0, 1, 1)
save(fig, "fig-two-routes.svg")

# ------------------------------------------------------------ 4. RBF: a signed sum of Gaussian bumps (1D)
gam = 1.0
sv = [(-3.0, +1.0), (-1.2, -1.0), (0.4, -0.8), (2.0, +1.2), (3.4, +0.7)]   # (position, α y)
b1 = 0.1
t = np.linspace(-5, 5, 800)
fig, ax = plt.subplots(figsize=(520 / 96, 300 / 96))
f = np.full_like(t, b1)
for p, ay in sv:
    bump = ay * np.exp(-gam * (t - p) ** 2)
    f += bump
    ax.plot(t, bump, c=RED if ay > 0 else BLUE, lw=1.4, ls=(0, (4, 3)))
    ax.plot(p, 0, "o", c=RED if ay > 0 else BLUE, ms=9, mec="white", zorder=5)
ax.fill_between(t, -1.4, 1.4, where=f > 0, color=LRED, zorder=0, step="mid")
ax.fill_between(t, -1.4, 1.4, where=f <= 0, color=LBLUE, zorder=0, step="mid")
ax.plot(t, f, c=INK, lw=2.6, label="f(x)")
ax.axhline(0, color=GREY, lw=1)
ax.annotate("", (sv[3][0] + 1 / np.sqrt(gam), -1.15), (sv[3][0], -1.15), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
ax.text(sv[3][0] + 0.5, -1.05, "1/√γ", ha="center", fontsize=14)
ax.set_xlim(-5, 5); ax.set_ylim(-1.4, 1.4)
ax.set_xlabel("feature x"); ax.set_yticks([-1, 0, 1])
ax.text(-4.9, 1.3, "solid: f = sum of the bumps + b", fontsize=13.5, va="top")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
save(fig, "fig-bumps.svg")

# ------------------------------------------------------------ 5. γ and C: four settings on the same data
G = D["grid"]; X, c = arr(G["points"]); R = G["R"]
gg = np.linspace(G["lo"], G["hi"], R); xx, yy = np.meshgrid(gg, gg)
fig, axs = plt.subplots(2, 2, figsize=(500 / 96, 540 / 96))
for k, cell in enumerate(G["cells"]):
    ax = axs[k // 2, k % 2]
    F = np.array(cell["F"]).reshape(R, R)
    ax.contourf(xx, yy, F, levels=[-1e9, 0, 1e9], colors=[LBLUE, LRED], zorder=0)
    ax.contour(xx, yy, F, levels=[0], colors=INK, linewidths=1.8)
    ax.scatter(X[c == 0, 0], X[c == 0, 1], s=12, c=BLUE, zorder=3)
    ax.scatter(X[c == 1, 0], X[c == 1, 1], s=12, c=RED, zorder=3)
    frame(ax, -5, 5)
    ax.set_title(f"{cell['nSV']} SV, {cell['errors']} wrong", fontsize=15, pad=3)
    if k % 2 == 0:
        ax.set_ylabel(f"γ = {cell['gamma']:g}", fontsize=16)
    if k // 2 == 1:
        ax.set_xlabel(f"C = {cell['C']:g}", fontsize=16)
fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.6)
save(fig, "fig-gamma-c.svg")
