# Figures for the pretext unit (pretext/lec.qmd). Run from the repository root after dump.ts
# (see its header):  python3 -I tools/figs/pretext/figs.py DIR/pt.json
# Images: synthetic nuclei (the maskfill demo) and the course micrograph (transmission electron
# micrograph of a mesenchymal stem cell, Robert M. Hunt, CC BY 3.0, via Wikimedia Commons).
import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/pretext/'
D = json.load(open(sys.argv[1]))
N = D["N"]
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, GREY = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c", "#9aa5b1"
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)
def img(a):
    a = np.array(a, dtype=float).reshape(N, N); return np.ma.masked_less(a, 0)
cmap = plt.cm.gray.copy(); cmap.set_bad("#f2c48d")
cell = np.asarray(Image.open(R + '/demos/src/fourier2d/cell256.png').convert('L'), dtype=float)

# ------------------------------------------------------------ 1. pretext tasks on a micrograph crop
crop = cell[40:136, 120:216]
fig, axs = plt.subplots(1, 6, figsize=(1050 / 96, 230 / 96))
for k, ax in enumerate(axs[:4]):
    ax.imshow(np.rot90(crop, k), cmap="gray", vmin=0, vmax=255); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(["0°?", "90°?", "180°?", "270°?"][k], fontsize=13, color=BLUE)
rng = np.random.default_rng(2)
masked = crop.copy().astype(float); hid = np.zeros_like(masked, dtype=bool)
for gy in range(6):
    for gx in range(6):
        if rng.random() < 0.75: hid[gy * 16:(gy + 1) * 16, gx * 16:(gx + 1) * 16] = True
axs[4].imshow(np.ma.masked_where(hid, masked), cmap=cmap, vmin=0, vmax=255); axs[4].set_title("75 % hidden", fontsize=13, color=ORANGE)
axs[5].imshow(crop, cmap="gray", vmin=0, vmax=255); axs[5].set_title("target", fontsize=13, color=ORANGE)
for ax in axs[4:]: ax.set_xticks([]); ax.set_yticks([])
fig.text(0.33, 0.02, "predict the rotation", ha="center", color=BLUE, fontsize=14)
fig.text(0.83, 0.02, "predict the hidden patches", ha="center", color=ORANGE, fontsize=14)
fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.14, wspace=0.08); save(fig, "fig-pretext.png", dpi=110)

# ------------------------------------------------------------ 2. the demo's fills (nuclei, 75 % of the patches)
def fills(scene, kind, name):
    S = D[scene]; F = S[kind]
    fig, axs = plt.subplots(1, 4, figsize=(820 / 96, 245 / 96))
    for ax, (t, a) in zip(axs, [("image", S["test"]), ("what the model sees", F["seen"]), (f"interpolation\n{F['pi']:.2f} dB", F["interp"]), (f"learnt (PCA, k = 32)\n{F['pp']:.2f} dB", F["pca"])]):
        ax.imshow(img(a), cmap=cmap, vmin=0, vmax=255, interpolation="nearest"); ax.axis("off"); ax.set_title(t, fontsize=12.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.8, bottom=0.01, wspace=0.05); save(fig, name, dpi=110)
fills("nuclei", "patches", "fig-fill-nuclei.png")
fills("cell", "patches", "fig-fill-cell.png")

# ------------------------------------------------------------ 3. PSNR against the fraction hidden
fig, axs = plt.subplots(1, 2, figsize=(760 / 96, 330 / 96), sharey=True)
for ax, sc, title in [(axs[0], "nuclei", "synthetic nuclei"), (axs[1], "cell", "micrograph")]:
    c = np.array(D[sc]["curve"]); r = 100 * c[:, 0]
    ax.plot(r, c[:, 1], "-o", color=GREY, lw=2, label="patches: interpolation")
    ax.plot(r, c[:, 2], "-o", color=ORANGE, lw=2, label="patches: learnt")
    ax.plot(r, c[:, 3], "--o", color=GREY, lw=1.5, ms=4, label="pixels: interpolation")
    ax.plot(r, c[:, 4], "--o", color=ORANGE, lw=1.5, ms=4, label="pixels: learnt")
    ax.set_title(title, fontsize=13); ax.set_xlabel("hidden (%)")
axs[0].set_ylabel("PSNR on hidden pixels (dB)")
fig.legend(*axs[0].get_legend_handles_labels(), fontsize=11, frameon=False, loc="lower center", ncol=4)
fig.tight_layout(rect=(0, 0.1, 1, 1)); save(fig, "fig-ratio.svg")

# ------------------------------------------------------------ 4. the learnt components
fig, axs = plt.subplots(2, 8, figsize=(760 / 96, 230 / 96))
for r, (sc, lab) in enumerate([("nuclei", "nuclei"), ("cell", "micrograph")]):
    for k in range(8):
        c = np.array(D[sc]["comps"][k]).reshape(18, 18); mx = np.abs(c).max()
        axs[r, k].imshow(c, cmap="RdBu_r", vmin=-mx, vmax=mx); axs[r, k].set_xticks([]); axs[r, k].set_yticks([])
    axs[r, 0].set_ylabel(lab, fontsize=12)
fig.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.03, wspace=0.08, hspace=0.08); save(fig, "fig-components.png", dpi=110)

# ------------------------------------------------------------ 5. masked autoencoder diagram
fig, ax = plt.subplots(figsize=(900 / 96, 270 / 96))
ax.set_xlim(0, 18); ax.set_ylim(0, 5); ax.axis("off")
def box(x, y, w, h, text, fc, fs=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2", fc=fc, ec=INK, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
def arrow(x0, x1, y=2.5):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
for i in range(4):
    for j in range(4):
        hidden = (i * 7 + j * 3) % 4 != 0
        ax.add_patch(plt.Rectangle((0.2 + j * 0.62, 1.2 + i * 0.62), 0.56, 0.56, fc="#f2c48d" if hidden else "#8fb3cc", ec="white"))
ax.text(1.4, 0.7, "image in patches,\n75 % hidden", ha="center", va="top", fontsize=12)
arrow(2.8, 3.6)
box(3.7, 1.6, 2.6, 1.8, "encoder\n(visible\npatches only)", "#cfe0ec")
arrow(6.4, 7.2)
box(7.3, 1.6, 2.6, 1.8, "features\n+ placeholders\nfor the hidden", "#e8eef3", fs=12)
arrow(10.0, 10.8)
box(10.9, 1.6, 2.4, 1.8, "small\ndecoder", "#f6e0c3")
arrow(13.4, 14.2)
box(14.3, 1.6, 3.4, 1.8, "predicted patches;\nloss on the hidden\nones only", "#f6e0c3", fs=12)
ax.text(5.0, 4.3, "kept after training", ha="center", color=BLUE, fontsize=12.5)
ax.text(13.6, 4.3, "thrown away", ha="center", color=RED, fontsize=12.5)
fig.tight_layout(pad=0.2); save(fig, "fig-mae.svg")
print("wrote", OUT)
