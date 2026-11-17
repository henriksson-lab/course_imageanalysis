# Diagram for the slide "The autoencoder as a tokenizer" (vae-gen/lec.qmd): a trained autoencoder
# as the compression stage of other generators. Run from the repo root:
#   python3 -I tools/figs/vae-gen/tokenizer.py
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

R = '/data/henriksson/github/course/imageanalysis'
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none", "text.color": "#1b2a3a"})
INK, BLUE, RED, GREEN, GREY = "#1b2a3a", "#2a6f97", "#bc4749", "#3a8a5c", "#9aa3ab"

def box(ax, x, y, w, h, text, col, fs=13):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=col, ec=INK, lw=1.5))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)
def arrow(ax, a, b, col=INK):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=col, lw=2, mutation_scale=16))

fig, ax = plt.subplots(figsize=(980 / 96, 360 / 96))
ax.set_xlim(0, 14); ax.set_ylim(0, 5.2); ax.axis("off")
# an image: a smooth synthetic cell, 64 x 64
yy, xx = np.mgrid[0:64, 0:64]
cell = np.exp(-(((xx - 32) / 20) ** 2 + ((yy - 30) / 14) ** 2) ** 2) * 0.7 + 0.5 * np.exp(-(((xx - 34) / 7) ** 2 + ((yy - 31) / 6) ** 2))
ax.imshow(cell, cmap="gray", extent=[0.2, 2.2, 1.6, 3.6], vmin=0, vmax=1.1)
ax.text(1.2, 1.25, "image\n256 × 256 × 3", ha="center", va="top", fontsize=12)
box(ax, 3.4, 2.6, 1.6, 1.0, "encoder", "#d7e6f2", 12)
# the latent grid: 8 x 8 cells drawn (standing for 32 x 32 x 4)
rng = np.random.default_rng(1)
for i in range(8):
    for j in range(8):
        ax.add_patch(Rectangle((4.8 + i * 0.16, 2.0 + j * 0.16), 0.15, 0.15, fc=plt.cm.viridis(rng.uniform(0.2, 0.9)), ec="white", lw=0.4))
ax.text(5.3, 1.25, "codes (\"tokens\")\n32 × 32 × 4\n48× fewer numbers", ha="center", va="top", fontsize=11)
box(ax, 8.6, 4.3, 3.9, 0.8, "latent diffusion, drifting", "#eef6ee", 11)
box(ax, 8.6, 2.6, 3.9, 1.0, "codebook → discrete tokens\n→ transformer, like words", "#f4ecdc", 11)
box(ax, 8.6, 0.9, 3.9, 0.8, "any model of the codes", "#eef2f6", 11)
for y in [4.3, 2.6, 0.9]:
    arrow(ax, (6.15, 2.65), (6.6, y))
    arrow(ax, (10.6, y), (10.98, 2.65))
box(ax, 11.85, 2.6, 1.7, 1.0, "decoder", "#d7e6f2", 12)
ax.imshow(cell[:, ::-1], cmap="gray", extent=[12.75, 13.95, 2.0, 3.2], vmin=0, vmax=1.1)
ax.text(13.35, 1.75, "pixels", ha="center", va="top", fontsize=12)
arrow(ax, (2.25, 2.6), (2.6, 2.6)); arrow(ax, (4.2, 2.6), (4.75, 2.6)); arrow(ax, (12.7, 2.6), (12.78, 2.6))
ax.text(7.0, 5.05, "trained once, then frozen: the same encoder and decoder under every model built on top", ha="center", fontsize=12, color=RED)
fig.savefig(R + "/vae-gen/fig-tokenizer.svg", facecolor="white"); plt.close(fig)
print("wrote vae-gen/fig-tokenizer.svg;", 256 * 256 * 3, "/", 32 * 32 * 4, "=", 256 * 256 * 3 / (32 * 32 * 4))
