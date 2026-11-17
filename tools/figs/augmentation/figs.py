# Figures for the augmentation unit (augmentation/lec.qmd). Run (from the repo root) after the steps in
# the header of dump.ts:
#   python3 -I tools/figs/augmentation/figs.py
# demos/src/augmentlab/curves.json: the crop experiment recorded with the demo's own code;
# crops.json: some of its crops; code/out/augmentation/*.png: the micrograph augmented by the course
# code (`cargo run -p training --features solution -- augmentation`, run in code/).
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/augmentation/'
C = json.load(open(R + '/demos/src/augmentlab/curves.json'))
K = json.load(open(R + '/tools/figs/augmentation/crops.json'))
IMG = R + '/code/out/augmentation/'
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, GREY = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c", "#8a96a3"
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)
def run(i): return next(c for c in C if c["id"] == i)
def grey(path):
    a = imread(path)
    return a[..., :3].mean(axis=2) if a.ndim == 3 else a

# 1. the micrograph and four augmentations (course code)
cell = grey(R + '/demos/src/fourier2d/cell256.png')
pics = [(cell, "original"), (grey(IMG + "cell-flipped.png"), "flipped"), (grey(IMG + "cell-rotated30.png"), "rotated 30°"),
        (grey(IMG + "cell-elastic.png"), "elastic (4 px)"), (grey(IMG + "cell-intensity.png"), "gain, offset, γ 0.6")]
fig, axs = plt.subplots(1, 5, figsize=(880 / 96, 215 / 96))
for ax, (a, t) in zip(axs, pics):
    ax.imshow(a, cmap="gray", vmin=0, vmax=1); ax.set_title(t, fontsize=13); ax.axis("off")
fig.tight_layout(w_pad=0.4); save(fig, "fig-copies.png", dpi=130)

# 2. six random copies of one crop (course code: flips, any angle, gain ±30 %, noise 0.05)
fig, axs = plt.subplots(1, 6, figsize=(800 / 96, 160 / 96))
for k, ax in enumerate(axs):
    ax.imshow(grey(IMG + f"random-{k}.png"), cmap="gray", vmin=0, vmax=1); ax.axis("off")
fig.tight_layout(w_pad=0.3); save(fig, "fig-random.png", dpi=130)

# 3. training and validation crops (demo code)
fig, axs = plt.subplots(2, 10, figsize=(620 / 96, 160 / 96))
for row, key, lab in [(0, "train", "training"), (1, "val", "validation")]:
    for k in range(10):
        ax = axs[row, k]; ax.imshow(np.array(K[key][k]).reshape(16, 16), cmap="gray", vmin=0, vmax=1.1); ax.set_xticks([]); ax.set_yticks([])
    axs[row, 0].set_ylabel(lab, fontsize=11, rotation=0, ha="right", va="center")
fig.tight_layout(w_pad=0.2, h_pad=0.3); save(fig, "fig-crops.png", dpi=130)

# 4. curves without and with flips + rotations
def curves(ax, ids, title):
    for i, (ct, cv) in zip(ids, [(BLUE, RED), (GREEN, ORANGE)]):
        r = run(i); lab = "no augmentation" if i == "none" else r["label"]
        ax.plot(r["train"], color=ct, lw=2.2, label=f"{lab}: training")
        ax.plot(r["val"], color=cv, lw=2.2, label=f"{lab}: validation")
    ax.set_ylim(0, 3.2); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10.5, loc="upper left")
fig, ax = plt.subplots(figsize=(470 / 96, 320 / 96))
curves(ax, ["none"], "20 training crops, no augmentation")
fig.tight_layout(); save(fig, "fig-noaug.svg")
fig, ax = plt.subplots(figsize=(470 / 96, 320 / 96))
curves(ax, ["none", "geom"], "with and without flips + rotations")
fig.tight_layout(); save(fig, "fig-augcurves.svg")

# 5. final validation accuracy of every setting
items = [("none", "none"), ("decay", "weight decay"), ("intensity", "gain + noise"), ("geom", "flips + rotations"),
         ("all", "+ gain + noise"), ("elastic", "+ elastic"), (None, ""), ("o-none", "none"), ("o-flips", "flips"), ("o-rot", "rotations")]
fig, ax = plt.subplots(figsize=(560 / 96, 340 / 96))
x, ticks, labs = 0, [], []
for i, lab in items:
    if i is None: x += 0.6; continue
    v = 100 * run(i)["valAcc"][300]
    c = GREY if i in ("none", "o-none") else (RED if i in ("o-rot", "intensity", "decay") else GREEN)
    ax.bar(x, v, color=c, width=0.75); ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=11)
    ticks.append(x); labs.append(lab); x += 1
ax.set_xticks(ticks); ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=11)
ax.set_ylim(40, 115); ax.set_yticks([50, 75, 100]); ax.set_ylabel("validation % correct")
ax.text(2.5, 110, "round vs elongated", ha="center", fontsize=12); ax.text(7.6, 110, "lying vs standing", ha="center", fontsize=12)
ax.axhline(50, color=GREY, ls="--", lw=1)
fig.tight_layout(); save(fig, "fig-accuracy.svg")
# 6. elastic deformation: a grid and the micrograph (course code), and the field drawn by numpy
from scipy.ndimage import gaussian_filter, map_coordinates
g = np.ones((128, 128)); g[::16, :] = 0; g[:, ::16] = 0
rs = np.random.default_rng(3)
fig, axs = plt.subplots(1, 3, figsize=(620 / 96, 230 / 96))
for ax, (alpha, sigma, t) in zip(axs[:2], [(4, 10, "4 px, smooth"), (6, 3, "6 px, rough: torn")]):
    f = [gaussian_filter(rs.normal(size=g.shape), sigma) for _ in range(2)]
    u, v = [alpha * a / np.sqrt((a ** 2).mean()) for a in f]
    yy, xx = np.mgrid[0:128, 0:128]
    ax.imshow(map_coordinates(g, [yy + v, xx + u], order=1, mode="mirror"), cmap="gray", vmin=0, vmax=1); ax.set_title(t, fontsize=12); ax.axis("off")
axs[2].imshow(grey(IMG + "cell-elastic.png"), cmap="gray", vmin=0, vmax=1); axs[2].set_title("micrograph, 4 px", fontsize=12); axs[2].axis("off")
fig.tight_layout(w_pad=0.4); save(fig, "fig-elastic.png", dpi=130)
print("written")
