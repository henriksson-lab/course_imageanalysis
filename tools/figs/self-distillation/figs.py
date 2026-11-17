# Figures for the self-distillation unit (self-distillation/lec.qmd). Run from the repository root:
#   python3 -I tools/figs/self-distillation/figs.py
# The entropies of fig-centring come from the unit's program, part 1:
#   cargo run --manifest-path code/Cargo.toml -p selfsup --features solution -- self-distillation
# (64 synthetic cells, 8 prototypes, an untrained network plus a common drift towards prototype 0).
# The moving-average curves are 1 − m^n, the same as part 2 of that program.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/self-distillation/'
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, GREY = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c", "#9aa5b1"
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)

def box(ax, x, y, w, h, text, fc, fs=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2", fc=fc, ec=INK, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
def arrow(ax, x0, y0, x1, y1, col=INK, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=col, lw=1.6, linestyle=ls))

# ------------------------------------------------------------ 1. student and teacher
fig, ax = plt.subplots(figsize=(980 / 96, 330 / 96))
ax.set_xlim(0, 20); ax.set_ylim(0, 6.4); ax.axis("off")
box(ax, 0.2, 2.5, 2.2, 1.4, "image", "#e8eef3")
box(ax, 3.4, 4.2, 2.4, 1.3, "view 1", "#e8eef3"); box(ax, 3.4, 0.9, 2.4, 1.3, "view 2", "#e8eef3")
arrow(ax, 2.4, 3.4, 3.4, 4.8); arrow(ax, 2.4, 3.0, 3.4, 1.6)
box(ax, 6.8, 4.2, 3.0, 1.3, "teacher", "#f6e0c3"); box(ax, 6.8, 0.9, 3.0, 1.3, "student", "#cfe0ec")
arrow(ax, 5.8, 4.85, 6.8, 4.85); arrow(ax, 5.8, 1.55, 6.8, 1.55)
box(ax, 10.8, 4.2, 4.0, 1.3, "centre, sharpen,\nsoftmax", "#f6e0c3", fs=12); box(ax, 10.8, 0.9, 4.0, 1.3, "softmax", "#cfe0ec", fs=12)
arrow(ax, 9.8, 4.85, 10.8, 4.85); arrow(ax, 9.8, 1.55, 10.8, 1.55)
box(ax, 15.8, 2.5, 3.9, 1.4, "cross-entropy:\nstudent ≈ teacher", "#ffffff", fs=12)
arrow(ax, 14.8, 4.85, 16.6, 3.9); arrow(ax, 14.8, 1.55, 16.6, 2.5)
ax.text(12.8, 5.85, "no gradient (stop)", ha="center", color=RED, fontsize=12)
arrow(ax, 8.3, 2.2, 8.3, 4.2, col=GREEN, ls="--")
ax.text(8.5, 3.2, "moving average\nof the weights", color=GREEN, fontsize=11.5, va="center")
ax.text(12.8, 0.35, "gradient (backpropagation)", ha="center", color=BLUE, fontsize=12)
fig.tight_layout(pad=0.2); save(fig, "fig-studentteacher.svg")

# ------------------------------------------------------------ 2. centring and sharpening (numbers from the Rust program)
rows = [("plain softmax", 1.981, 1.987, 1), ("sharpened only", 0.000, 0.000, 1), ("centred only", 2.073, 2.079, 7), ("centred and\nsharpened", 0.897, 1.932, 7)]
fig, ax = plt.subplots(figsize=(560 / 96, 330 / 96))
x = np.arange(len(rows)); w = 0.36
ax.bar(x - w / 2, [r[1] for r in rows], w, color=BLUE, label="entropy per cell (low: confident)")
ax.bar(x + w / 2, [r[2] for r in rows], w, color=ORANGE, label="entropy of the batch mean (low: all alike)")
ax.axhline(np.log(8), color=INK, ls="--", lw=1); ax.text(3.6, np.log(8) + 0.05, "ln 8", ha="right", fontsize=11)
for i, r in enumerate(rows): ax.text(i, -0.42, f"{r[3]} of 8 used", ha="center", fontsize=10.5, color=GREY)
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=11)
ax.set_ylim(-0.5, 2.9); ax.set_ylabel("entropy"); ax.legend(fontsize=10.5, frameon=False, loc="upper left")
fig.tight_layout(); save(fig, "fig-centring.svg")

# ------------------------------------------------------------ 3. the moving-average teacher
fig, ax = plt.subplots(figsize=(600 / 96, 290 / 96))
n = np.arange(0, 1001)
ax.plot([0, 0, 1000], [0, 1, 1], color=BLUE, lw=2, label="student")
for m, col in [(0.9, GREEN), (0.99, ORANGE), (0.996, RED)]:
    ax.plot(n, 1 - m ** n, color=col, lw=2, label=f"teacher, m = {m}")
ax.set_xscale("symlog", linthresh=1); ax.set_xlim(0, 1000); ax.set_ylim(-0.05, 1.1)
ax.set_xlabel("steps after the jump"); ax.set_ylabel("weight"); ax.legend(fontsize=10.5, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout(); save(fig, "fig-ema.svg")

# ------------------------------------------------------------ 4. image-text matching (schematic)
fig, ax = plt.subplots(figsize=(520 / 96, 360 / 96))
caps = ["nucleus in\nmitosis", "apoptotic\ncell", "two nuclei", "empty field"]
S = np.array([[0.62, 0.21, 0.30, 0.05], [0.18, 0.55, 0.12, 0.10], [0.33, 0.15, 0.58, 0.04], [0.04, 0.12, 0.06, 0.49]])
ax.imshow(S, cmap="Oranges", vmin=0, vmax=0.7)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=12, fontweight="bold" if i == j else "normal")
ax.set_xticks(range(4)); ax.set_xticklabels(caps, fontsize=10.5)
ax.set_yticks(range(4)); ax.set_yticklabels([f"image {k + 1}" for k in range(4)], fontsize=11)
ax.set_title("similarity of image and caption embeddings\n(illustration)", fontsize=12)
for sp in ax.spines.values(): sp.set_visible(False)
fig.tight_layout(); save(fig, "fig-imagetext.svg")
print("wrote", OUT)
