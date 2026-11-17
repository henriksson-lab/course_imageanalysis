# Figures for the contrastive unit (contrastive/lec.qmd). Run from the repository root after
# dump.ts (see its header):  python3 -I tools/figs/contrastive/figs.py DIR/ct.json
# Synthetic 16 x 16 cells of the contrastive demo; embeddings after 600 steps of its training.
import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/contrastive/'
D = json.load(open(sys.argv[1]))
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, GREY = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c", "#9aa5b1"
SHAPE = ["#1f77b4", "#d62728", "#2ca02c"]
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)

# ------------------------------------------------------------ 1. a cell and four augmented views
fig, axs = plt.subplots(3, 5, figsize=(470 / 96, 300 / 96))
for r, v in enumerate(D["views"]):
    for c, a in enumerate([v["clean"]] + v["views"]):
        ax = axs[r, c]
        ax.imshow(np.array(a).reshape(16, 16), cmap="gray", vmin=0, vmax=1.3, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(c == 0); sp.set_color(ORANGE); sp.set_linewidth(2.5)
axs[0, 0].set_title("cell", fontsize=13); axs[0, 2].set_title("four augmented views", fontsize=13, loc="left")
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.08, hspace=0.08); save(fig, "fig-views.png", dpi=110)

# ------------------------------------------------------------ 2. pull together, push apart (schematic on the circle)
fig, ax = plt.subplots(figsize=(420 / 96, 380 / 96))
t = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(t), np.sin(t), color=GREY, lw=1.5)
a, p = 0.9, 1.25
negs = [2.3, 3.4, 4.4, 5.4]
ax.scatter([np.cos(a)], [np.sin(a)], s=160, color=ORANGE, zorder=5, edgecolor="white")
ax.scatter([np.cos(p)], [np.sin(p)], s=160, color=ORANGE, zorder=5, marker="s", edgecolor="white")
ax.annotate("", xy=(np.cos(a + 0.12), np.sin(a + 0.12)), xytext=(np.cos(p - 0.04), np.sin(p - 0.04)), arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.5))
ax.text(1.05 * np.cos(1.1) + 0.05, 1.05 * np.sin(1.1) + 0.12, "two views of\none cell: pull", color=GREEN, fontsize=13, ha="left")
for n in negs:
    ax.scatter([np.cos(n)], [np.sin(n)], s=110, color=BLUE, zorder=5, edgecolor="white")
    d = n + 0.25 if (n - a) % (2 * np.pi) < np.pi else n - 0.25
    ax.annotate("", xy=(np.cos(d) * 0.995, np.sin(d) * 0.995), xytext=(np.cos(n), np.sin(n)), arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
ax.text(0.05, -0.1, "views of other\ncells: push", color=RED, fontsize=13, ha="center")
ax.set_xlim(-1.35, 1.6); ax.set_ylim(-1.3, 1.45); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout(); save(fig, "fig-pullpush.svg")

# ------------------------------------------------------------ 3. InfoNCE by hand: scores -> probabilities
fig, axs = plt.subplots(1, 2, figsize=(560 / 96, 280 / 96), sharey=True)
cos = np.array([0.9, 0.1, -0.5])
for ax, tau in zip(axs, [0.5, 0.1]):
    s = cos / tau; p = np.exp(s - s.max()); p /= p.sum()
    ax.bar([0, 1, 2], p, color=[ORANGE, BLUE, BLUE], width=0.6)
    for k in range(3): ax.text(k, p[k] + 0.03, f"{p[k]:.3f}" if p[k] > 0.0005 else f"{p[k]:.0e}", ha="center", fontsize=12)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["partner\ncos 0.9", "cos 0.1", "cos −0.5"], fontsize=11)
    ax.set_title(f"τ = {tau}: loss {-np.log(p[0]):.4f}", fontsize=13); ax.set_ylim(0, 1.15)
axs[0].set_ylabel("probability")
fig.tight_layout(); save(fig, "fig-infonce.svg")

# ------------------------------------------------------------ 4. embeddings after training, four settings
def circle(ax, run, by, title):
    t = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(t), np.sin(t), color=GREY, lw=1)
    z = np.array(run["z"])
    if by == "shape": cols = [SHAPE[c["shape"]] for c in D["cells"]]
    else: cols = [plt.cm.viridis((c["bright"] - 0.35) / 0.65) for c in D["cells"]]
    rng = np.random.default_rng(1)
    jit = 1 + 0.06 * rng.standard_normal(len(z))   # radial jitter so that equal points stay visible
    ax.scatter(z[:, 0] * jit, z[:, 1] * jit, c=cols, s=38, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=12.5)
    ax.text(0, -1.55, f"shape {100 * run['shape']:.0f} %, bright {100 * run['bright']:.0f} %", ha="center", fontsize=11.5)
Rn = D["runs"]
fig, axs = plt.subplots(1, 4, figsize=(1000 / 96, 300 / 96))
circle(axs[0], Rn["all"], "shape", "all augmentations\n(colour: shape)")
circle(axs[1], Rn["all"], "bright", "all augmentations\n(colour: brightness)")
circle(axs[2], Rn["nobright"], "bright", "no brightness augm.\n(colour: brightness)")
circle(axs[3], Rn["nobright"], "shape", "no brightness augm.\n(colour: shape)")
for ax in axs: ax.set_ylim(-1.7, 1.3)
fig.tight_layout(pad=0.4); save(fig, "fig-embeddings.svg")

# ------------------------------------------------------------ 5. collapse and the trivial task
fig, axs = plt.subplots(1, 2, figsize=(520 / 96, 300 / 96))
circle(axs[0], Rn["noneg"], "shape", "no negatives:\ncollapse")
circle(axs[1], Rn["none"], "shape", "no augmentation:\nspread, but unsorted")
for ax in axs: ax.set_ylim(-1.7, 1.3)
fig.tight_layout(pad=0.4); save(fig, "fig-collapse.svg")

# ------------------------------------------------------------ 6. loss curves
fig, ax = plt.subplots(figsize=(520 / 96, 300 / 96))
for id_, lab, col in [("all", "all augmentations", ORANGE), ("nobright", "no brightness augm.", BLUE), ("none", "no augmentation", GREY), ("tau1", "τ = 1", GREEN)]:
    L = np.array(Rn[id_]["losses"]); k = 10
    ax.plot(np.arange(len(L) - k + 1) + k, np.convolve(L, np.ones(k) / k, "valid"), color=col, lw=2, label=lab)
ax.axhline(np.log(64), color=INK, ls="--", lw=1); ax.text(600, np.log(64) + 0.06, "ln 64: guessing", ha="right", fontsize=11.5)
ax.set_xlabel("step"); ax.set_ylabel("InfoNCE loss"); ax.set_ylim(1.4, 4.6); ax.legend(fontsize=11, frameon=False, loc="lower center", ncol=2)
fig.tight_layout(); save(fig, "fig-loss.svg")
print("wrote", OUT)
