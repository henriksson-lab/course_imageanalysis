"""Slide figures for pixel-classify/lec.qmd.

Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/pixel-classify/data.ts
     python3 -I tools/figs/pixel-classify/figs.py <dir>
Uses the pixelclass demo's synthetic test image, strokes, feature stack, SVM plug-in and the
svmcv tab's cross-validation (numbers printed by data.ts). Figure sizes: display px / 96.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb, ListedColormap
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "pixel-classify"))
D = json.load(open(os.path.join(sys.argv[1], "pixel-classify.json")))
S = D["size"]
IMG = np.array(D["image"]).reshape(S, S)
TRUTH = np.array(D["truth"]).reshape(S, S)
LAB = np.array(D["labels"]).reshape(S, S)
COL = ["#1f77b4", "#ff7f0e", "#2ca02c"]          # background, nucleus, granular (the demo's colours)
NAMES = ["background", "nucleus", "granular cell"]
plt.rcParams.update({"font.size": 15, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#222", "xtick.color": "#555",
                     "ytick.color": "#555", "text.color": "#222", "svg.hashsalt": "pixel-classify"})
INK, AMBER, GREY = "#222", "#e6a700", "#888"


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), facecolor="white", **kw)
    plt.close(fig)
    print("wrote", name)


def grey_rgb(img):
    g = np.clip(img / 255.0, 0, 1)
    return np.dstack([g, g, g])


def with_strokes(labels):
    rgb = grey_rgb(IMG)
    for c in range(3):
        rgb[labels == c + 1] = to_rgb(COL[c])
    return rgb


def classes_rgb(pred):
    rgb = np.zeros((S, S, 3))
    for c in range(3):
        rgb[pred == c] = to_rgb(COL[c])
    return rgb


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])


# ------------------------------------------------------------ 1. paint → SVM result
svm = D["svm"]; pred = np.array(svm["pred"]).reshape(S, S)
fig, axs = plt.subplots(1, 2, figsize=(520 / 96, 315 / 96))
axs[0].imshow(with_strokes(LAB), interpolation="nearest"); axs[0].set_title("painted strokes", fontsize=17, pad=4)
axs[1].imshow(classes_rgb(pred), interpolation="nearest")
axs[1].set_title(f"SVM: {100 * svm['acc']:.1f} % correct", fontsize=17, pad=4)
for ax in axs:
    off(ax)
handles = [Rectangle((0, 0), 1, 1, fc=COL[c]) for c in range(3)]
fig.legend(handles, NAMES, loc="lower center", ncol=3, frameon=False, fontsize=14, handlelength=1, columnspacing=0.8, handletextpad=0.4)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.12, top=0.89, wspace=0.04)
save(fig, "fig-svm-result.png", dpi=96 * 1.5)

# ------------------------------------------------------------ 2. standardise: raw vs z-scores
Ch = D["chans"]; a = np.array(Ch["a"]); b = np.array(Ch["b"]); t = TRUTH.ravel() - 1    # truth: 1, 2, 3
rng = np.random.default_rng(1)
sel = rng.choice(S * S, 1500, replace=False)
fig, axs = plt.subplots(1, 2, figsize=(470 / 96, 330 / 96))
za, zb = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
for ax, (xa, xb), title in zip(axs, ((a, b), (za, zb)), ("raw, equal axes", "z-scores")):
    ax.scatter(xa[sel], xb[sel], s=7, c=[COL[k] for k in t[sel]], alpha=0.6, edgecolors="none")
    ax.set_title(title, fontsize=15.5, pad=4)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
axs[0].set_aspect("equal", adjustable="datalim")
axs[0].set_xlabel(f"A (sd {Ch['sd'][0]:.1f})"); axs[0].set_ylabel(f"B (sd {Ch['sd'][1]:.0f})", labelpad=0)
axs[0].set_yticks([0, 500, 1000]); axs[0].set_xticks([-300, 300])
axs[1].set_aspect("equal"); axs[1].set_xlim(-3, 6); axs[1].set_ylim(-3, 6)
axs[1].set_xlabel("A (z)"); axs[1].set_ylabel("B (z)", labelpad=0); axs[1].set_xticks([0, 5]); axs[1].set_yticks([0, 5])
print("channels:", Ch["names"], [round(v, 2) for v in Ch["sd"]])
fig.tight_layout(pad=0.3, w_pad=0.8)
save(fig, "fig-standardise.png", dpi=96 * 1.5)

# ------------------------------------------------------------ 3. the (C, γ) grid, folds by stroke
cv = D["cv"]; Cs, gs = cv["Cs"], cv["gs"]
fmt = lambda v: str(int(round(v))) if v >= 1 else f"{v:g}"


def heat(ax, mode, title):
    E = np.array(cv[mode]["E"]).reshape(len(gs), len(Cs))
    ax.imshow(E, cmap="viridis", vmin=0, vmax=0.5)
    for i in range(len(gs)):
        for j in range(len(Cs)):
            v = E[i, j]
            ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=14, color="white" if v < 0.3 else INK)
    bi = cv[mode]["best"]
    ax.add_patch(Rectangle((bi % len(Cs) - 0.5, bi // len(Cs) - 0.5), 1, 1, fill=False, ec=AMBER, lw=4))
    ax.set_xticks(range(len(Cs))); ax.set_xticklabels([fmt(c) for c in Cs])
    ax.set_yticks(range(len(gs))); ax.set_yticklabels([fmt(g) for g in gs])
    ax.set_xlabel("C"); ax.set_ylabel("γ")
    ax.set_title(title, fontsize=15.5, pad=4)
    return E


fig, ax = plt.subplots(figsize=(380 / 96, 380 / 96))
E = heat(ax, "stroke", "CV error (%), by stroke")
fig.tight_layout(pad=0.3)
save(fig, "fig-cv-grid.svg")

# ------------------------------------------------------------ 4. folds by stroke vs by pixel
pix = np.array(cv["pix"])
FOLDC = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#d55e00"]
fig, axs = plt.subplots(1, 2, figsize=(520 / 96, 315 / 96))
for ax, mode, title in zip(axs, ("stroke", "pixel"), ("by stroke", "by pixel")):
    rgb = grey_rgb(IMG) * 0.55 + 0.45
    fo = np.array(cv[mode]["folds"])
    yy, xx = np.divmod(pix, S)
    ax.imshow(rgb, interpolation="nearest")
    for f in range(cv[mode]["nF"]):
        m = fo == f
        ax.scatter(xx[m], yy[m], s=9, c=FOLDC[f], edgecolors="none")
    Em = np.array(cv[mode]["E"])
    ax.set_title(f"{title}\nbest cell {100 * Em.min():.0f} %", fontsize=15.5, pad=4)
    off(ax)
fig.text(0.5, 0.015, "colour = fold of each pixel", ha="center", fontsize=14)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.08, top=0.83, wspace=0.04)
save(fig, "fig-folds.png", dpi=96 * 1.5)

# ------------------------------------------------------------ 5. class imbalance: equal vs balanced weights
I = D["imbalance"]; P = np.array([[q["x"], q["y"]] for q in I["points"]]); c = np.array([q["c"] for q in I["points"]])
fig, axs = plt.subplots(1, 2, figsize=(520 / 96, 310 / 96))
g = np.linspace(0, 10, 300); xx, yy = np.meshgrid(g, g)
for ax, r in zip(axs, I["res"]):
    w, b0 = np.array(r["w"]), r["b"]
    F = w[0] * xx + w[1] * yy + b0
    ax.contourf(xx, yy, F, levels=[-1e9, 0, 1e9], colors=["#dce9f5", "#fde6cc"], zorder=0)
    ax.contour(xx, yy, F, levels=[0], colors=INK, linewidths=2.2)
    ax.contour(xx, yy, F, levels=[-1, 1], colors=GREY, linewidths=1.2, linestyles="dashed")
    ax.scatter(P[c == 0, 0], P[c == 0, 1], s=9, c=COL[0], alpha=0.7, edgecolors="none")
    ax.scatter(P[c == 1, 0], P[c == 1, 1], s=26, c=COL[1], edgecolors="white", linewidths=0.5, zorder=3)
    ax.set_xlim(0.5, 9.5); ax.set_ylim(1.5, 8.5); ax.set_aspect("equal"); off(ax)
    lab = "balanced weights" if r["balanced"] else "equal C"
    ax.set_title(f"{lab}\nfound {r['tp']}/{I['nPos']}, false {r['fp']}", fontsize=15, pad=4)
fig.tight_layout(pad=0.3, w_pad=1.0)
save(fig, "fig-imbalance.svg")

# ------------------------------------------------------------ 6. from pixels to objects
prob = np.array(svm["prob"]).reshape(S, S)
mask = pred == 1
MIN = 30
lab0, n0 = ndi.label(mask)
sizes = ndi.sum(mask, lab0, range(1, n0 + 1))
keep = np.isin(lab0, 1 + np.flatnonzero(sizes >= MIN))
lab, nobj = ndi.label(keep)
print("nucleus components:", n0, "of at least", MIN, "px:", nobj)
cm = plt.get_cmap("tab20")
obj_rgb = np.ones((S, S, 3))
for k in range(1, nobj + 1):
    obj_rgb[lab == k] = cm((k * 7) % 20)[:3]
fig, axs = plt.subplots(1, 3, figsize=(700 / 96, 280 / 96))
axs[0].imshow(prob, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); axs[0].set_title("score: nucleus", fontsize=16, pad=4)
axs[1].imshow(mask, cmap="gray", interpolation="nearest"); axs[1].set_title("class = nucleus", fontsize=16, pad=4)
axs[2].imshow(obj_rgb, interpolation="nearest"); axs[2].set_title(f"{nobj} objects (≥ {MIN} px)", fontsize=16, pad=4)
for ax in axs:
    off(ax)
fig.tight_layout(pad=0.3, w_pad=0.6)
save(fig, "fig-objects.png", dpi=96 * 1.5)
