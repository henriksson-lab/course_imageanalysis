"""Extra slide figures for forest/lec.qmd.

Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/forest/data.ts
     python3 -I tools/figs/forest/figs.py <dir>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "forest")
D = json.load(open(os.path.join(sys.argv[1], "forest.json")))
RES = D["RES"]
C0, C1 = "#1f77b4", "#ff7f0e"
plt.rcParams.update({"font.size": 16, "font.family": "DejaVu Sans", "svg.fonttype": "none", "axes.linewidth": 1.0})


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), facecolor="white", **kw)
    plt.close(fig)
    print("wrote", name)


def points(ax, P, s=12):
    P = np.array(P)
    for c, col, m in [(0, C0, "o"), (1, C1, "^")]:
        q = P[P[:, 2] == c]
        ax.scatter(q[:, 0], q[:, 1], s=s, color=col, marker=m, edgecolors="#222", linewidths=0.5, zorder=3)


def probmap(ax, p1, strength=True):
    p1 = np.asarray(p1).reshape(RES, RES)
    col = p1[..., None] * np.array(to_rgb(C1)) + (1 - p1[..., None]) * np.array(to_rgb(C0))
    conf = np.abs(2 * p1 - 1)[..., None] if strength else 1.0
    a = 0.08 + 0.42 * conf
    ax.imshow(1 - a * (1 - col), origin="lower", extent=(0, 1, 0, 1), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")


# ---------------------------------------------------------------- bootstrap trees and their average
B = D["bagged"]
fig, axs = plt.subplots(1, 4, figsize=(11.0, 3.4))
for b in range(3):
    probmap(axs[b], B["trees"][b], strength=False)
    inb = np.array(B["inbag"][b]) > 0
    points(axs[b], np.array(B["points"])[inb], s=9)
    axs[b].set_title(f"tree on sample {b + 1}", fontsize=16)
P1 = np.array(B["P"]).reshape(-1, B["k"])[:, 1]
probmap(axs[3], P1)
points(axs[3], B["points"], s=9)
axs[3].set_title("average of 100 trees", fontsize=16)
fig.tight_layout()
save(fig, "fig-bagtrees.png", dpi=150)

# ---------------------------------------------------------------- bootstrap matrix (illustration, n = 12)
n, T = 12, 6
rng = np.random.default_rng(25)
M = np.zeros((T, n), int)
for t in range(T):
    for i in rng.integers(0, n, n):
        M[t, i] += 1
print("bootstrap matrix: out-of-bag fraction %.2f; per sample" % np.mean(M == 0), (M == 0).sum(1))


def matrix(ax, highlight=None):
    for t in range(T):
        for i in range(n):
            c = M[t, i]
            if c == 0:
                fc = "#e8e8e8"
            else:
                fc = ["#ffffff", "#c6dbef", "#6baed6", "#2171b5"][min(c, 3)]
            ec = "#999"
            ax.add_patch(plt.Rectangle((i, T - 1 - t), 1, 1, fc=fc, ec=ec, lw=1))
            if c:
                ax.text(i + 0.5, T - 0.5 - t, str(c), ha="center", va="center", fontsize=15,
                        color="white" if c >= 3 else "black")
            elif highlight is None or i != highlight:
                ax.text(i + 0.5, T - 0.5 - t, "–", ha="center", va="center", fontsize=15, color="#888")
    if highlight is not None:
        ax.add_patch(plt.Rectangle((highlight, 0), 1, T, fill=False, ec="#d62728", lw=3))
        for t in range(T):
            if M[t, highlight] == 0:
                ax.text(highlight + 0.5, T - 0.5 - t, "✓", ha="center", va="center", fontsize=20, color="#d62728", fontweight="bold")
    ax.set_xlim(-1.9, n + 0.1); ax.set_ylim(-0.9, T + 0.1); ax.set_aspect("equal"); ax.axis("off")
    for t in range(T):
        ax.text(-0.15, T - 0.5 - t, f"tree {t + 1}", ha="right", va="center", fontsize=15)
    for i in range(n):
        ax.text(i + 0.5, -0.3, str(i + 1), ha="center", va="top", fontsize=13, color="#555")
    ax.text(n / 2, -0.9, "training point", ha="center", va="top", fontsize=15)


fig, ax = plt.subplots(figsize=(6.3, 3.8))
matrix(ax)
fig.tight_layout()
save(fig, "fig-bootstrap.svg")
j = int(np.nonzero((M == 0).sum(0) == 2)[0][-1])
fig, ax = plt.subplots(figsize=(6.3, 3.8))
matrix(ax, highlight=j)
fig.tight_layout()
save(fig, "fig-oob.svg")
print("OOB column", j + 1, "left out by", int((M[:, j] == 0).sum()), "of", T, "trees")

# ---------------------------------------------------------------- random feature subsets
feats = ["x", "y", "x + y", "x − y"]
fig, ax = plt.subplots(figsize=(5.4, 3.2))
picks = [[0, 2], [1, 3], [2, 3]]
best = [2, 1, 3]
for r, (pk, bst) in enumerate(zip(picks, best)):
    y0 = 2 - r
    ax.text(-0.2, y0 + 0.35, f"split {r + 1}", ha="right", va="center", fontsize=16)
    for j, f in enumerate(feats):
        on = j in pk
        ax.add_patch(plt.Rectangle((j * 1.25, y0), 1.1, 0.7, fc=("#cfe3f3" if on else "#f2f2f2"),
                                   ec=("#0072B2" if j == bst else ("#777" if on else "#ccc")), lw=(3 if j == bst else 1)))
        ax.text(j * 1.25 + 0.55, y0 + 0.35, f, ha="center", va="center", fontsize=16, color=("black" if on else "#aaa"))
ax.set_xlim(-1.6, 5.0); ax.set_ylim(-0.1, 2.8); ax.axis("off")
ax.text(2.4, -0.05, "shaded: m = 2 drawn;  bold: the best", ha="center", va="top", fontsize=15, color="#444")
fig.tight_layout()
save(fig, "fig-mtry.svg")

# ---------------------------------------------------------------- probability map
Bd = D["boundary"]
Pb = np.array(Bd["P"]).reshape(-1, Bd["k"])[:, 1]
fig, ax = plt.subplots(figsize=(4.6, 4.6))
probmap(ax, Pb)
points(ax, Bd["points"], s=14)
Pm = Pb.reshape(RES, RES)
print("boundary preset: P(class 1) at the four corners", Pm[0, 0], Pm[0, -1], Pm[-1, 0], Pm[-1, -1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("pale: uncertain; strong: confident", fontsize=15)
fig.tight_layout()
save(fig, "fig-prob.png", dpi=150)

# ---------------------------------------------------------------- OOB and test error against the number of trees
o = np.array([np.nan if v is None else v for v in B["oob"]])
t = np.array(B["test"])
b = np.arange(1, len(t) + 1)
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.plot(b, 100 * o, color="#0072B2", lw=2.2, label="OOB error")
ax.plot(b, 100 * t, color="#D55E00", lw=2.2, ls="--", label="test error")
ax.set_xlabel("number of trees")
ax.set_ylabel("error (%)")
ax.set_ylim(15, 40); ax.set_xlim(1, 100)
ax.legend(frameon=False, fontsize=15)
ax.spines[["top", "right"]].set_visible(False)
ax.text(100, 100 * o[-1] + 0.8, f"{100 * o[-1]:.1f} %", ha="right", va="bottom", fontsize=15, color="#0072B2")
ax.text(100, 100 * t[-1] - 0.8, f"{100 * t[-1]:.1f} %", ha="right", va="top", fontsize=15, color="#D55E00")
fig.tight_layout()
save(fig, "fig-oobcurve.svg")
print("first finite OOB at B =", int(np.argmax(np.isfinite(o))) + 1, "; error at B = 10, 30:", 100 * o[9], 100 * t[9], 100 * o[29], 100 * t[29])
