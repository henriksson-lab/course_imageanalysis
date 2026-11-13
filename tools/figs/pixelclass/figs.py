"""Extra slide figures for pixelclass/lec.qmd.

Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/pixelclass/data.ts
     python3 -I tools/figs/pixelclass/figs.py <dir>
Uses the demo's synthetic test image, presets and forest (numbers printed by data.ts).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "pixelclass")
D = json.load(open(os.path.join(sys.argv[1], "pixelclass.json")))
S = D["size"]
IMG = np.array(D["image"]).reshape(S, S)
TRUTH = np.array(D["truth"]).reshape(S, S)
COL = ["#1f77b4", "#ff7f0e", "#2ca02c"]          # background, nucleus, granular (the demo's colours)
NAMES = ["background", "nucleus", "granular cell"]
plt.rcParams.update({"font.size": 16, "font.family": "DejaVu Sans", "axes.linewidth": 1.0})


def grey_rgb(img):
    g = np.clip(img / 255.0, 0, 1)
    return np.dstack([g, g, g])


def with_strokes(labels, img=IMG):
    rgb = grey_rgb(img)
    for c in range(3):
        m = labels == c + 1
        rgb[m] = to_rgb(COL[c])
    return rgb


def seg_rgb(pred, img=IMG, alpha=0.55):
    rgb = grey_rgb(img)
    col = np.array([to_rgb(c) for c in COL])[pred]
    return (1 - alpha) * rgb + alpha * col


def arr(key, name):
    return np.array(D[key][name]).reshape(S, S)


def panel(ax, im, cap, **kw):
    ax.imshow(im, interpolation="nearest", **kw)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(cap, fontsize=16, pad=6)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def legend_cell(ax, classes):
    from matplotlib.patches import Patch
    ax.axis("off")
    h = [Patch(color=COL[c], label=NAMES[c]) for c in classes]
    ax.legend(handles=h, loc="center", frameon=False, fontsize=17, handlelength=1.4)


def legend_row(fig, classes, y=0.01, ncol=None, fs=16):
    from matplotlib.patches import Patch
    h = [Patch(color=COL[c], label=NAMES[c]) for c in classes]
    fig.legend(handles=h, loc="lower center", ncol=ncol or len(classes), frameon=False, fontsize=fs, bbox_to_anchor=(0.5, y), handlelength=1.2, columnspacing=1.0)


# ---------------------------------------------------------------- every pixel is a point
ci = np.array(D["chInt"]).reshape(S, S)
ct = np.array(D["chTex"]).reshape(S, S)
inner = [ndi.binary_erosion(TRUTH == c + 1, iterations=2) for c in range(3)]
rs = np.random.default_rng(1)
fig, (a0, a1) = plt.subplots(1, 2, figsize=(6.8, 3.7), gridspec_kw={"width_ratios": [1, 1.15]})
panel(a0, grey_rgb(IMG), "test image")
for c in [0, 1, 2]:
    yy, xx = np.nonzero(inner[c])
    pick = rs.choice(len(yy), 400, replace=False)
    a1.scatter(ci[yy[pick], xx[pick]], ct[yy[pick], xx[pick]], s=7, color=COL[c], alpha=0.55, label=NAMES[c], linewidths=0)
a1.set_yscale("symlog", linthresh=10)
a1.set_xlabel("brightness (smoothed, σ = 5)")
a1.set_ylabel("texture (σ = 1.6)")
a1.spines[["top", "right"]].set_visible(False)
a1.set_ylim(-0.6, 4000)
for txt, (x, y), c in [("background", (58, 6.5), 0), ("nucleus", (128, 6.5), 1), ("granular cell", (112, 1300), 2)]:
    a1.text(x, y, txt, color=COL[c], fontsize=16, fontweight="bold", va="bottom")
fig.tight_layout()
save(fig, "fig-pixelpoint.png")

# ---------------------------------------------------------------- the loop
U, F = D["uncertain"], D["full"]
fig, axs = plt.subplots(2, 2, figsize=(6.0, 6.4))
panel(axs[0, 0], with_strokes(arr("uncertain", "labels")), "1  paint two classes")
panel(axs[0, 1], seg_rgb(arr("uncertain", "pred")), f"2  train, look: {100 * U['acc']:.1f} % right")
panel(axs[1, 0], with_strokes(arr("full", "labels")), "3  correct: paint more")
panel(axs[1, 1], seg_rgb(arr("full", "pred")), f"4  train again: {100 * F['acc']:.1f} %")
fig.tight_layout(rect=(0, 0.07, 1, 1))
legend_row(fig, [0, 1, 2], ncol=3, fs=15)
save(fig, "fig-loop.png")

# ---------------------------------------------------------------- trap 1: held-out region
H = D["heldout"]
x0 = int(round(D["holdout"] * S))
fig, (a0, a1) = plt.subplots(1, 2, figsize=(6.2, 3.8))
im = with_strokes(arr("heldout", "labels"))
im[:, x0:] = 0.75 * im[:, x0:] + 0.25 * np.array(to_rgb("#ffd700"))
panel(a0, im, "strokes; held-out region")
a0.axvline(x0 - 0.5, color="#e6b800", lw=3)
a0.text(x0 / 2, S + 4, f"OOB: {100 * H['oobAcc']:.1f} %", ha="center", va="top", fontsize=15,
        color="#222")
a0.text((x0 + S) / 2, S + 4, f"{100 * H['heldPainted']:.1f} %", ha="center", va="top", fontsize=15,
        color="#222")
seg = seg_rgb(arr("heldout", "pred"))
panel(a1, seg, "result")
a1.axvline(x0 - 0.5, color="#e6b800", lw=3)
a1.text((x0 + S) / 2, S + 4, f"all px: {100 * H['heldTruth']:.1f} %", ha="center", va="top", fontsize=15,
        color="#222")
fig.tight_layout()
save(fig, "fig-heldout.png")

# ---------------------------------------------------------------- trap 2: imbalance
A, B = D["imb"], D["imbBal"]
fig, axs = plt.subplots(2, 2, figsize=(6.4, 6.6))
axs = axs.ravel()
panel(axs[0], with_strokes(arr("imb", "labels")), "huge background strokes")
panel(axs[1], seg_rgb(arr("imb", "pred")), f"plain: nucleus Dice {A['dice'][1]:.2f}")
panel(axs[2], seg_rgb(arr("imbBal", "pred")), f"balanced: {B['dice'][1]:.2f}")
legend_cell(axs[3], [0, 1, 2])
fig.tight_layout()
save(fig, "fig-imbalance.png")
frac = lambda R: [np.mean(np.array(R["pred"]) == c) for c in range(3)]
print("imbalance predicted fractions", frac(A), frac(B))

# ---------------------------------------------------------------- trap 3: only what it was shown
Bp = arr("bright", "pred")
dimmask = np.zeros((S, S), bool)
lab, n = ndi.label(TRUTH == 2)
for e, kind in zip(D["ellipses"], D["kinds"]):
    if kind == "dim":
        l = lab[int(round(e["y"])), int(round(e["x"]))]
        dimmask |= lab == l
print("dim nuclei pixels predicted background (bright preset): %.1f %%" % (100 * np.mean(Bp[dimmask] == 0)))
Sm, Un = arr("small", "pred"), arr("uncertain", "pred")
print("granular pixels called nucleus without a granular stroke: %.1f %%" % (100 * np.mean(Un[TRUTH == 3] == 1)))
fig, axs = plt.subplots(2, 2, figsize=(6.4, 6.6))
axs = axs.ravel()
panel(axs[0], seg_rgb(Bp), "only bright nuclei painted")
for e, kind in zip(D["ellipses"], D["kinds"]):
    if kind == "dim":
        axs[0].add_patch(plt.Circle((e["x"], e["y"]), e["a"] + 7, fill=False, ec="#d62728", lw=2.5, ls="--"))
panel(axs[1], seg_rgb(Un), "granular never painted")
panel(axs[2], seg_rgb(Sm), "small scales only")
legend_cell(axs[3], [0, 1, 2])
fig.tight_layout()
save(fig, "fig-trap3.png")

# ---------------------------------------------------------------- from probabilities to objects
Pn = arr("full", "prob")
th = Pn > 0.5
cl = ndi.binary_opening(th, structure=np.ones((3, 3)), iterations=1)
cl = ndi.binary_fill_holes(cl)
lab, n = ndi.label(cl)
sizes = ndi.sum(cl, lab, range(1, n + 1))
keep = np.isin(lab, 1 + np.nonzero(sizes >= 30)[0])
lab, n = ndi.label(keep)
nt = ndi.label(TRUTH == 2)[1]
print(f"objects: {n} nucleus objects after clean-up (truth: {nt} nuclei); raw threshold pieces {ndi.label(th)[1]}")
pal = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3]
lrgb = np.ones((S, S, 3))
for i in range(1, n + 1):
    lrgb[lab == i] = pal[(i * 7) % 20]
fig, axs = plt.subplots(1, 4, figsize=(11.6, 3.5))
panel(axs[0], Pn, "P(nucleus)", cmap="viridis", vmin=0, vmax=1)
panel(axs[1], th, f"> 0.5: {ndi.label(th)[1]} pieces", cmap="gray_r")
panel(axs[2], keep, "open, fill, remove small", cmap="gray_r")
panel(axs[3], lrgb, f"label: {n} nuclei")
for ax in axs[1:4]:
    ax.add_patch(plt.Rectangle((-0.5, -0.5), S, S, fill=False, ec="#888", lw=1))
fig.tight_layout()
save(fig, "fig-objects.png")
