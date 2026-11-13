"""Extra slide figures for forest-objects/lec.qmd.

Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/forest-objects/data.ts
     python3 -I tools/figs/forest-objects/figs.py <dir>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "forest-objects")
D = json.load(open(os.path.join(sys.argv[1], "forest-objects.json")))
C0, C1 = "#1f77b4", "#ff7f0e"     # interphase, mitotic
MDI, PERM = "#0072B2", "#E69F00"  # the demo's bar colours: impurity blue, permutation orange
plt.rcParams.update({"font.size": 16, "font.family": "DejaVu Sans", "svg.fonttype": "none", "axes.linewidth": 1.0})
cols = [np.array(c) for c in D["cols"]]
y = np.array(D["y"])


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), facecolor="white", **kw)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- objects -> table
rows = [0, 200, 1, 201, 2, 202]
known = {0: "interphase", 200: "mitotic", 1: "interphase", 201: "mitotic", 2: "?", 202: "?"}
S = 300
img = np.full((S, S), 40.0)
lab = np.zeros((S, S), int)
rng = np.random.default_rng(4)
centres = [(68, 62), (180, 58), (255, 150), (62, 215), (160, 180), (250, 255)]
radii = []
yy, xx = np.mgrid[0:S, 0:S]
for k, (i, (cx, cy)) in enumerate(zip(rows, centres)):
    area = cols[0][i]
    r = np.sqrt(area / np.pi) * 3.0          # pixels per µm (illustration)
    radii.append(r)
    ang = rng.uniform(0, np.pi)
    e = 1.0 + 0.25 * rng.random()
    u = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
    v = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
    m = (u / (r * np.sqrt(e))) ** 2 + (v / (r / np.sqrt(e))) ** 2 <= 1
    tex = ndi.gaussian_filter(rng.standard_normal((S, S)), 1.2)
    tex /= tex.std()
    img[m] = cols[3][i] + 4.0 * cols[4][i] * tex[m]
    lab[m] = k + 1
img = ndi.gaussian_filter(img, 0.8) + 4 * rng.standard_normal((S, S))
fig = plt.figure(figsize=(10.4, 3.9))
ax = fig.add_axes([0.0, 0.02, 0.36, 0.96])
ax.imshow(np.clip(img, 0, 255), cmap="gray", vmin=0, vmax=220)
edge = (lab > 0) & ~ndi.binary_erosion(lab > 0)
ov = np.zeros((S, S, 4)); ov[edge] = (1.0, 0.85, 0.0, 1.0)
ax.imshow(ov)
for k, (cx, cy) in enumerate(centres):
    ax.text(cx - radii[k] * 0.75 - 6, cy - radii[k] * 0.75 - 6, str(k + 1), color="#ffd900", fontsize=18, fontweight="bold", ha="center", va="center")
ax.axis("off")
ta = fig.add_axes([0.38, 0.0, 0.62, 1.0]); ta.axis("off")
hdr = ["#", "area", "perimeter", "intensity", "texture", "class"]
xs = [0.02, 0.12, 0.30, 0.52, 0.72, 0.88]
for x, h in zip(xs, hdr):
    ta.text(x, 0.92, h, fontsize=16, fontweight="bold", ha="left" if h in ("#", "class") else "center")
ta.plot([0, 1.04], [0.875, 0.875], color="#444", lw=1)
for r_, i in enumerate(rows):
    yv = 0.78 - 0.125 * r_
    cls = known[i]
    vals = [str(r_ + 1), f"{cols[0][i]:.0f}", f"{cols[1][i]:.0f}", f"{cols[3][i]:.0f}", f"{cols[4][i]:.1f}"]
    for x, v in zip(xs, vals):
        ta.text(x, yv, v, fontsize=16, ha="left" if x == xs[0] else "center")
    ta.text(xs[5] - 0.06, yv, cls, fontsize=16, color=(C0 if cls == "interphase" else C1 if cls == "mitotic" else "#777"),
            fontweight="bold" if cls != "?" else "normal")
ta.text(0.5, 0.0, "one row per object (µm², µm, grey levels)", fontsize=14, color="#555", ha="center")
ta.set_xlim(0, 1.05); ta.set_ylim(-0.05, 1)
fig.savefig(os.path.join(OUT, "fig-table.png"), dpi=150, facecolor="white")
plt.close(fig)
print("wrote fig-table.png; rows", [(round(cols[0][i]), round(cols[4][i], 1), int(y[i])) for i in rows])

# ---------------------------------------------------------------- permutation: shuffle one column
A = D["all"]
oob = A["oob"]
tex_perm = A["perm"][A["names"].index("texture")]
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.axis("off")
vals = [[cols[0][i], cols[4][i]] for i in rows[:5]]
perm = [2, 4, 0, 1, 3]
for t, (x0, title, sh) in enumerate([(0.3, "original", False), (4.6, "texture shuffled", True)]):
    ax.text(x0 + 1.25, 6.25, title, ha="center", fontsize=16, fontweight="bold", color="#b35806" if sh else "black")
    for c, nm in enumerate(["area", "texture"]):
        xc = x0 + c * 1.3
        ax.text(xc + 0.6, 5.5, nm, ha="center", fontsize=15)
        for r_ in range(5):
            v = vals[perm[r_] if (sh and c == 1) else r_][c]
            ax.add_patch(plt.Rectangle((xc, 4.4 - r_), 1.2, 0.9, fc="#fde9c9" if c == 1 else "#f4f4f4",
                                       ec="#b35806" if (sh and c == 1) else "#999"))
            ax.text(xc + 0.6, 4.85 - r_, f"{v:.0f}" if c == 0 else f"{v:.1f}", ha="center", va="center", fontsize=16)
ax.annotate("", (4.4, 2.85), (3.1, 2.85), arrowprops=dict(arrowstyle="->", lw=2.5, color="#b35806"))
ax.text(3.55, -0.3, f"OOB error {100 * oob:.1f} % → {100 * (oob + tex_perm):.1f} %", ha="center", fontsize=16)
ax.text(3.55, -0.95, f"texture's permutation importance: +{100 * tex_perm:.1f} points", ha="center", fontsize=15, color="#444")
ax.set_xlim(0, 7.2); ax.set_ylim(-1.2, 6.7)
fig.tight_layout()
save(fig, "fig-perm.svg")

# ---------------------------------------------------------------- the small tree by hand
fig, ax = plt.subplots(figsize=(6.2, 4.0))
ax.axis("off")


def node(x, yv, counts, label=None, leaf=False):
    a, b = counts
    w = 1.5
    ax.add_patch(plt.Rectangle((x - w / 2, yv - 0.32), w, 0.64, fc="white", ec="#444", lw=1.5,
                               ls="-" if not leaf else (0, (3, 2))))
    ax.text(x, yv, f"{a} | {b}", ha="center", va="center", fontsize=16)
    if label:
        ax.text(x, yv + 0.46, label, ha="center", va="bottom", fontsize=15, fontweight="bold", color="#0072B2")


def edge(x0, y0, x1, y1):
    ax.plot([x0, x1], [y0 - 0.32, y1 + 0.32], color="#666", lw=1.4)


node(3.5, 3.0, (50, 50), "texture")
node(1.8, 1.55, (45, 10), "area"); node(5.2, 1.55, (5, 40), "texture")
edge(3.5, 3.0, 1.8, 1.55); edge(3.5, 3.0, 5.2, 1.55)
for x, c in [(0.9, (40, 2)), (2.7, (5, 8))]:
    node(x, 0.2, c, leaf=True); edge(1.8, 1.55, x, 0.2)
for x, c in [(4.3, (5, 5)), (6.1, (0, 35))]:
    node(x, 0.2, c, leaf=True); edge(5.2, 1.55, x, 0.2)
ax.text(0.0, 3.0, "interphase | mitotic", fontsize=14, color="#555", va="center")
ax.set_xlim(-0.1, 7.0); ax.set_ylim(-0.3, 3.8)
fig.tight_layout()
save(fig, "fig-handtree.svg")

# ---------------------------------------------------------------- correlated size features
Bk = D["block"]
fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.4, 4.4), gridspec_kw={"width_ratios": [1, 1.15]})
for c, col, m, nm in [(0, C0, "o", "interphase"), (1, C1, "^", "mitotic")]:
    q = y == c
    a0.scatter(cols[0][q], cols[1][q], s=12, color=col, marker=m, alpha=0.7, linewidths=0, label=nm)
r = np.corrcoef(cols[0], cols[1])[0, 1]
a0.set_xlabel("area (µm²)"); a0.set_ylabel("perimeter (µm)")
a0.text(0.04, 0.95, f"r = {r:.2f}", transform=a0.transAxes, fontsize=16, va="top")
a0.legend(frameon=False, fontsize=15, loc="lower right", markerscale=1.8, handletextpad=0.2)
a0.spines[["top", "right"]].set_visible(False)
names = Bk["names"]
idx = [names.index(n) for n in ["area", "perimeter", "eq. diam."]]
yy_ = np.arange(3)
a1.barh(yy_ - 0.2, [100 * Bk["mdi"][j] for j in idx], 0.38, color=MDI, label="impurity (% of total)")
a1.barh(yy_ + 0.2, [100 * Bk["perm"][j] for j in idx], 0.38, color=PERM, label="permutation (points)")
a1.barh(3.4, 100 * Bk["groupPerm"], 0.38, color=PERM, hatch="//", ec="white")
a1.set_yticks(list(yy_) + [3.4], ["area", "perimeter", "eq. diam.", "all three\nat once"])
a1.invert_yaxis()
a1.axvline(0, color="#444", lw=1)
a1.set_xlim(-2, 26)
a1.set_xlabel("importance")
fig.legend(*a1.get_legend_handles_labels(), frameon=False, fontsize=15, loc="lower center", ncol=2)
a1.spines[["top", "right"]].set_visible(False)
for j, v in zip(idx, [Bk["perm"][j] for j in idx]):
    pass
a1.text(100 * Bk["groupPerm"] + 0.5, 3.4, f"+{100 * Bk['groupPerm']:.1f}", va="center", fontsize=15)
fig.tight_layout(rect=(0, 0.08, 1, 1))
save(fig, "fig-corr.svg")

# ---------------------------------------------------------------- both importances, all features
nm = A["names"]
order = list(range(len(nm)))
fig, (a0, a1) = plt.subplots(1, 2, figsize=(6.8, 4.6), sharey=True)
yy_ = np.arange(len(nm))
a0.barh(yy_, [100 * A["mdi"][j] for j in order], color=MDI)
a1.barh(yy_, [100 * A["perm"][j] for j in order], color=PERM)
a0.set_yticks(yy_, [nm[j] for j in order]); a0.invert_yaxis()
a0.set_xlabel("impurity (% of total)"); a1.set_xlabel("permutation (points)")
a1.axvline(0, color="#444", lw=1)
for a in (a0, a1):
    a.spines[["top", "right"]].set_visible(False)
for k, j in enumerate(order):
    if nm[j] in ("noise", "coin"):
        a0.text(100 * A["mdi"][j] + 0.8, k, f"{100 * A['mdi'][j]:.0f} %", va="center", fontsize=15)
        a1.text(1.0, k, f"{100 * A['perm'][j]:+.1f}", va="center", fontsize=15)
fig.tight_layout()
save(fig, "fig-imp.svg")
