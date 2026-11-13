"""Slide figures for maxmargin/lec.qmd.

Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/maxmargin/data.ts
     python3 -I tools/figs/maxmargin/figs.py <dir>
Points, lines, widths and errors come from the svm2d demo's presets and solver (data.ts).
Figure sizes are the display size in px / 96.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "maxmargin"))
D = json.load(open(os.path.join(sys.argv[1], "maxmargin.json")))
plt.rcParams.update({"font.size": 15, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#222", "xtick.color": "#555",
                     "ytick.color": "#555", "text.color": "#222", "svg.hashsalt": "maxmargin"})
BLUE, RED, AMBER, GREY, INK = "#1f77b4", "#d62728", "#e6a700", "#888", "#222"
LBLUE, LRED, STREET = "#dce9f5", "#f8dede", "#e4e4e4"


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), facecolor="white", **kw)
    plt.close(fig)
    print("wrote", name)


def arr(r):
    X = np.array([[p["x"], p["y"]] for p in r["points"]])
    c = np.array([p["c"] for p in r["points"]])
    return X, c


def street(ax, w, b, lo, hi, n=500, edges=True, fill=True):
    g = np.linspace(lo, hi, n)
    xx, yy = np.meshgrid(g, g)
    f = w[0] * xx + w[1] * yy + b
    if fill:
        ax.contourf(xx, yy, f, levels=[-1e9, -1, 1, 1e9], colors=[LBLUE, STREET, LRED], zorder=0)
    ax.contour(xx, yy, f, levels=[0], colors=INK, linewidths=2.4, zorder=2)
    if edges:
        ax.contour(xx, yy, f, levels=[-1], colors=BLUE, linewidths=1.8, linestyles=[(0, (6, 4))], zorder=2)
        ax.contour(xx, yy, f, levels=[1], colors=RED, linewidths=1.8, linestyles=[(0, (6, 4))], zorder=2)


def points(ax, X, c, sv=(), s=60, ring=True):
    ax.scatter(X[c == 0, 0], X[c == 0, 1], s=s, c=BLUE, edgecolors="white", linewidths=0.8, zorder=4)
    ax.scatter(X[c == 1, 0], X[c == 1, 1], s=s, c=RED, edgecolors="white", linewidths=0.8, zorder=4)
    if ring and len(sv):
        ax.scatter(X[sv, 0], X[sv, 1], s=s * 3.2, facecolors="none", edgecolors=INK, linewidths=1.8, zorder=5)


def frame(ax, lo, hi, ticks=False):
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    if not ticks:
        ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------ 1. moving a non-support vector changes nothing
fig, axs = plt.subplots(1, 2, figsize=(620 / 96, 370 / 96))
for ax, key, title in zip(axs, ("nonsv", "nonsvMoved"), ("before", "two points moved")):
    r = D[key]; X, c = arr(r)
    street(ax, r["w"], r["b"], -1, 4)
    points(ax, X, c, r["sv"], s=50)
    frame(ax, -1, 4)
    ax.set_title(f"{title}\nwidth {r['width']:.2f}", fontsize=17, pad=4)
X0, _ = arr(D["nonsv"])
for i, x, y in D["moves"]:
    axs[1].add_patch(FancyArrowPatch(X0[int(i)], (x, y), arrowstyle="-|>", mutation_scale=16, color=AMBER, lw=2.4, zorder=3))
    axs[1].scatter(*X0[int(i)], s=50, facecolors="none", edgecolors=GREY, linestyles="dotted", zorder=3)
assert np.allclose(D["nonsv"]["w"], D["nonsvMoved"]["w"], atol=1e-3) and abs(D["nonsv"]["b"] - D["nonsvMoved"]["b"]) < 1e-3
fig.tight_layout(pad=0.4, w_pad=1.0, rect=(0, 0, 1, 0.97))
save(fig, "fig-sv-move.svg")

# ------------------------------------------------------------ 2. why a wide street: noisy points
r = D["byhand"]; X, c = arr(r)
rad = 0.6                                   # how far a point may move
fig, axs = plt.subplots(1, 2, figsize=(620 / 96, 340 / 96))
# another line with zero errors, but close to (0, 2)
wn = np.array([1.0, 0.5]); bn = -1.3
margins = (np.where(c == 1, 1, -1) * (X @ wn + bn)) / np.linalg.norm(wn)
assert margins.min() > 0
for ax, (w, b), title in zip(axs, ((wn, bn), (np.array(r["w"]), r["b"])),
                            (f"margin {margins.min():.2f}", "widest: margin 1")):
    street(ax, w, b, -1, 4, edges=False, fill=False)
    g = np.linspace(-1, 4, 300); xx, yy = np.meshgrid(g, g)
    ax.contourf(xx, yy, w[0] * xx + w[1] * yy + b, levels=[-1e9, 0, 1e9], colors=[LBLUE, LRED], zorder=0)
    for (px, py), cc in zip(X, c):
        d = abs(w @ np.array([px, py]) + b) / np.linalg.norm(w)
        cross = d < rad
        ax.add_patch(Circle((px, py), rad, fc="none", ec=AMBER if cross else GREY, lw=2.4 if cross else 1.4,
                            ls="-" if cross else (0, (3, 3)), zorder=3))
    points(ax, X, c, ring=False)
    frame(ax, -1, 4)
    ax.set_title(title, fontsize=17, pad=4)
fig.tight_layout(pad=0.4, w_pad=1.0)
save(fig, "fig-robust.svg")

# ------------------------------------------------------------ 3. fixing the scale: shrink ‖w‖ until the band touches
fig, axs = plt.subplots(1, 2, figsize=(620 / 96, 380 / 96))
for ax, k in zip(axs, (2, 1)):
    w = np.array([k, 0.0]); b = -k
    street(ax, w, b, -1, 4)
    points(ax, X, c, s=55, ring=False)
    yf = np.where(c == 1, 1, -1) * (X @ w + b)
    for (px, py), v in zip(X, yf):
        ax.text(px + 0.12, py + 0.18, f"{v:g}", fontsize=14, zorder=6)
    frame(ax, -1, 4)
    ax.set_title(f"w = ({k}, 0)\nwidth {2 / k:g}", fontsize=17, pad=4)
    ax.text(1, -0.8, "f = 0", ha="center", fontsize=14, zorder=6)
axs[0].text(-0.9, 3.85, "all y f ≥ 1,\nbut ‖w‖ can shrink", fontsize=15, va="top", zorder=7, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
axs[1].text(-0.9, 3.85, "smallest ‖w‖:\nthree points at y f = 1", fontsize=15, va="top", zorder=7, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
fig.tight_layout(pad=0.4, w_pad=1.0)
save(fig, "fig-scale.svg")

# ------------------------------------------------------------ 4. hard vs soft margin on the outlier preset
fig, axs = plt.subplots(1, 2, figsize=(620 / 96, 380 / 96))
for ax, key, C in zip(axs, ("outlierHard", "outlierSoft"), ("1000", "1")):
    r = D[key]; X, c = arr(r)
    w, b = np.array(r["w"]), r["b"]
    street(ax, w, b, 0, 10)
    y = np.where(c == 1, 1, -1)
    # slack: segment from a violating point to its edge f = y
    nw2 = w @ w
    for (px, py), yy_, v in zip(X, y, r["yf"]):
        if v < 1 - 1e-6:
            fx = w @ np.array([px, py]) + b
            t = (yy_ - fx) / nw2
            ax.plot([px, px + t * w[0]], [py, py + t * w[1]], c=AMBER, lw=3.2, solid_capstyle="round", zorder=3)
    points(ax, X, c, r["sv"], s=44)
    frame(ax, 1.5, 8.5)
    ax.set_title(f"C = {C}\nwidth {r['width']:.2f}, {r['errors']} error{'s' if r['errors'] != 1 else ''}", fontsize=17, pad=4)
fig.tight_layout(pad=0.4, w_pad=1.0)
save(fig, "fig-soft.svg")

# ------------------------------------------------------------ 5. the four losses against y f
m = np.linspace(-1.6, 2.6, 600)
L = {"0–1": (m <= 0).astype(float), "hinge": np.maximum(0, 1 - m), "logistic": np.log2(1 + np.exp(-m)), "squared": (1 - m) ** 2}
col = {"0–1": INK, "hinge": "#0072b2", "logistic": "#009e73", "squared": "#d55e00"}
fig, ax = plt.subplots(figsize=(500 / 96, 300 / 96))
for k, v in L.items():
    if k == "0–1":
        ax.plot(m[m <= 0], v[m <= 0], c=col[k], lw=2.2); ax.plot(m[m > 0], v[m > 0], c=col[k], lw=2.2, label=k)
    else:
        ax.plot(m, v, c=col[k], lw=2.6 if k == "hinge" else 2, label=k)
for q in D["losses"]:
    ax.axvline(q["m"], color="#cccccc", lw=1, zorder=0)
    ax.plot(q["m"], q["hinge"], "o", c=col["hinge"], ms=7, zorder=5)
ax.set_xlim(-1.6, 2.6); ax.set_ylim(0, 3.2)
ax.set_xlabel("y f(x)"); ax.set_ylabel("loss")
ax.legend(frameon=False, fontsize=13.5, loc="upper right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
save(fig, "fig-losses.svg")
