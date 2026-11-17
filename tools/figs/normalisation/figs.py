# Figures for the normalisation unit (normalisation/lec.qmd). Run (from the repo root) after dump.ts
# and the course program (see the header of dump.ts):
#   python3 -I tools/figs/normalisation/figs.py
# data.json: signal and gradient sizes per layer from the deepstack demo's own code.
# plain-vs-residual.csv: training losses printed by `cargo run -p training -- normalisation`.
# The trap figure uses the course micrograph (demos/src/fourier2d/cell256.png, credit in the caption).
import json, csv
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/normalisation/'
D = json.load(open(R + '/tools/figs/normalisation/data.json'))
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, PURPLE, GREY = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c", "#7b5ea7", "#8a96a3"
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)
L = np.arange(1, 51)

# 1. size of the signal per layer
fig, ax = plt.subplots(figsize=(480 / 96, 320 / 96))
for k, c, lab in [("relu2", RED, "gain 2: explodes"), ("relu_he", GREEN, "gain √2: steady"), ("relu1", BLUE, "gain 1: fades"), ("bn2", ORANGE, "gain 2 + batch norm")]:
    ax.plot(np.r_[0, L], np.r_[D[k]["input"], D[k]["rms"]], color=c, lw=2.5, label=lab, ls="--" if k == "bn2" else "-")
ax.set_yscale("log"); ax.set_ylim(1e-8, 1e8); ax.set_yticks([1e-8, 1e-4, 1, 1e4, 1e8])
ax.set_xlabel("layer"); ax.set_ylabel("size of the signal (rms)")
ax.legend(fontsize=11, loc="upper left"); ax.set_title("50 ReLU layers of 64 units", fontsize=13)
fig.tight_layout(); save(fig, "fig-signal.svg")

# 2. size of the gradient per layer
fig, ax = plt.subplots(figsize=(480 / 96, 320 / 96))
for k, c, lab, ls in [("relu1", BLUE, "ReLU, gain 1", "-"), ("relu_he", GREEN, "ReLU, gain √2", "-"), ("sigmoid1", PURPLE, "sigmoid", "-"), ("ressig", PURPLE, "sigmoid, residual", "--")]:
    ax.plot(L, D[k]["grad"], color=c, lw=2.5, label=lab, ls=ls)
ax.set_yscale("log"); ax.set_ylim(1e-20, 1e2); ax.set_yticks([1e-20, 1e-15, 1e-10, 1e-5, 1])
ax.set_xlabel("layer (the loss is above layer 50)"); ax.set_ylabel("size of the gradient")
ax.legend(fontsize=11, loc="lower right")
fig.tight_layout(); save(fig, "fig-grads.svg")

# 3. batch norm vs layer norm on the batch matrix
fig, axs = plt.subplots(1, 2, figsize=(560 / 96, 260 / 96))
rng = np.random.default_rng(2)
for ax, kind in zip(axs, ["batch", "layer"]):
    M = rng.normal(size=(6, 8))
    ax.imshow(M, cmap="Greys", vmin=-2.5, vmax=2.5)
    if kind == "batch":
        ax.add_patch(Rectangle((2.5, -0.5), 1, 6, fill=False, ec=RED, lw=3))
        ax.set_title("batch norm: each unit\nover the examples", fontsize=13)
    else:
        ax.add_patch(Rectangle((-0.5, 1.5), 8, 1, fill=False, ec=BLUE, lw=3))
        ax.set_title("layer norm: each example\nover the units", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("units of the layer"); ax.set_ylabel("examples")
    for sp in ax.spines.values(): sp.set_visible(True)
fig.tight_layout(); save(fig, "fig-bnln.svg")

# 4. the trap: the same cells under another microscope
from matplotlib.image import imread
im = imread(R + '/demos/src/fourier2d/cell256.png')
if im.ndim == 3: im = im[..., :3].mean(axis=2)
other = np.minimum(1.6 * im + 0.15, 1.0)
fig, axs = plt.subplots(1, 2, figsize=(560 / 96, 300 / 96))
for ax, a, t in zip(axs, [im, other], ["training microscope\nafter batch norm:\nmean 0.00, sd 1.00", "another microscope (×1.6 + 0.15)\nwith the stored statistics:\nmean 1.15, sd 0.61"]):
    ax.imshow(a, cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=12)
fig.tight_layout(); save(fig, "fig-trap.png", dpi=150)

# 5. plain block vs residual block, and a U-Net skip connection
def box(ax, x, y, w, h, text, fc="#eef2f6", ec=INK, fs=12):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.5))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)
def arrow(ax, x0, y0, x1, y1, c=INK, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8, ls=ls))
fig, axs = plt.subplots(1, 3, figsize=(660 / 96, 300 / 96))
for ax in axs: ax.set_xlim(0, 4); ax.set_ylim(0, 5); ax.axis("off")
ax = axs[0]; ax.set_title("plain block", fontsize=13)
box(ax, 2, 0.5, 1.2, 0.5, "x"); box(ax, 2, 1.8, 2.2, 0.6, "layer"); box(ax, 2, 3.0, 2.2, 0.6, "layer"); box(ax, 2, 4.4, 1.8, 0.5, "y = F(x)")
arrow(ax, 2, 0.75, 2, 1.5); arrow(ax, 2, 2.1, 2, 2.7); arrow(ax, 2, 3.3, 2, 4.15)
ax = axs[1]; ax.set_title("residual block", fontsize=13)
box(ax, 2, 0.5, 1.2, 0.5, "x"); box(ax, 1.6, 1.8, 2.0, 0.6, "layer"); box(ax, 1.6, 3.0, 2.0, 0.6, "layer")
ax.add_patch(plt.Circle((2, 3.85), 0.22, fc="white", ec=RED, lw=2)); ax.text(2, 3.85, "+", ha="center", va="center", color=RED, fontsize=16)
box(ax, 2, 4.6, 2.6, 0.5, "y = x + F(x)")
arrow(ax, 1.8, 0.75, 1.6, 1.5); arrow(ax, 1.6, 2.1, 1.6, 2.7); arrow(ax, 1.6, 3.3, 1.9, 3.65)
ax.plot([2.3, 3.3, 3.3], [0.5, 0.5, 3.85], color=RED, lw=2); arrow(ax, 3.3, 3.85, 2.22, 3.85, RED)
ax.text(3.45, 2.2, "identity", color=RED, fontsize=11, rotation=90, va="center")
arrow(ax, 2, 4.07, 2, 4.37)
ax = axs[2]; ax.set_title("U-Net skip connection", fontsize=13)
box(ax, 0.8, 4.0, 1.5, 0.7, "encoder\nmap", fs=10); box(ax, 0.8, 2.6, 1.5, 0.5, "smaller", fs=10); box(ax, 3.2, 2.6, 1.5, 0.5, "decoder", fs=10)
box(ax, 3.2, 4.0, 1.5, 0.7, "concat-\nenate", fs=10)
arrow(ax, 0.8, 3.65, 0.8, 2.85); arrow(ax, 1.55, 2.6, 2.45, 2.6); arrow(ax, 3.2, 2.85, 3.2, 3.65)
arrow(ax, 1.55, 4.0, 2.45, 4.0, BLUE); ax.text(2.0, 4.45, "copy", color=BLUE, ha="center", fontsize=11)
ax.text(2.0, 1.4, "carries detail across\nthe U; joins, not adds", ha="center", fontsize=10, color=GREY)
fig.tight_layout(); save(fig, "fig-blocks.svg")

# 6. plain vs residual training (the course code)
rows = list(csv.DictReader(open(R + '/tools/figs/normalisation/plain-vs-residual.csv')))
ep = [int(r["epoch"]) for r in rows]
fig, ax = plt.subplots(figsize=(480 / 96, 320 / 96))
for k, c, ls, lab in [("plain2", GREEN, "-", "2 blocks, plain"), ("plain50", RED, "-", "50 blocks, plain"), ("residual50", BLUE, "-", "50 blocks, residual")]:
    ax.plot(ep, [float(r[k]) for r in rows], color=c, ls=ls, lw=2.3, label=lab)
ax.axhline(np.log(3), color=GREY, ls="--", lw=1); ax.text(60, np.log(3) + 0.04, "ln 3: guessing", color=GREY, fontsize=11, ha="right")
ax.set_ylim(0, 1.35); ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
ax.legend(fontsize=11, loc="center right", bbox_to_anchor=(1, 0.6))
fig.tight_layout(); save(fig, "fig-plain-vs-residual.svg")

# 7. ResNet in one picture
fig, ax = plt.subplots(figsize=(820 / 96, 300 / 96))
ax.set_xlim(0, 18); ax.set_ylim(0, 5); ax.axis("off")
box(ax, 0.9, 2.5, 1.6, 1.6, "image\n224²", fs=13)
box(ax, 3.1, 2.5, 2.3, 1.6, "stem:\n7×7 conv,\npool", fs=12, fc="#f6efe4")
x0 = 5.0
stages = [("stage 1", "56²\n64 ch", 3), ("stage 2", "28²\n128 ch", 4), ("stage 3", "14²\n256 ch", 6), ("stage 4", "7²\n512 ch", 3)]
x = 5.5
for i, (name, size, n) in enumerate(stages):
    h = 2.6 - 0.45 * i
    for b in range(n):
        ax.add_patch(Rectangle((x + b * 0.32, 2.5 - h / 2), 0.26, h, fc="#dbe7f0", ec=BLUE, lw=1.2))
    ax.text(x + n * 0.16, 2.5 + h / 2 + 0.35, name, ha="center", fontsize=13)
    ax.text(x + n * 0.16, 2.5 - h / 2 - 0.2, size + f"\n{n} blocks", ha="center", va="top", fontsize=12)
    x += n * 0.32 + 0.6
box(ax, x + 0.9, 2.5, 2.0, 1.2, "average\npool", fs=12, fc="#f6efe4"); box(ax, x + 3.3, 2.5, 2.4, 1.2, "classes\n(or a head)", fs=12, fc="#f6efe4")
ax.text(9.5, 4.9, "each bar = one residual block; each stage halves the size, doubles the channels", ha="center", fontsize=12, color=GREY)
fig.tight_layout(); save(fig, "fig-resnet.svg")
print("written")
