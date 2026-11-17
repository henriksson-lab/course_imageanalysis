# Figures for the unit gan (gan/lec.qmd, gan/exercise.qmd). Run (from the repo root) after dump.ts
# (see its header):  python3 -I tools/figs/gan/figs.py
# data.json holds runs of the gan1d demo's line GAN (data N(1, 0.5^2), start N(-1, 1)).
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/gan/'
D = json.load(open(R + '/tools/figs/gan/data.json'))
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, GREY = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c", "#9aa3ab"
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)
xs = np.array(D["xs"])

# ------------------------------------------------------------ 1. the game (diagram)
def box(ax, x, y, w, h, text, col, fs=15):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=col, ec=INK, lw=1.5))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)
def arrow(ax, a, b, col=INK):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=col, lw=2, mutation_scale=18))
fig, ax = plt.subplots(figsize=(720 / 96, 320 / 96))
ax.set_xlim(0, 10); ax.set_ylim(-0.4, 4.2); ax.axis("off")
box(ax, 1.0, 1.2, 1.8, 1.0, "noise z\n~ N(0, I)", "#eef2f6")
box(ax, 3.4, 1.2, 1.9, 1.0, "generator\nG", "#d7e6f2")
box(ax, 5.9, 1.2, 2.2, 1.0, "generated\nimage G(z)", "#eef2f6")
box(ax, 5.9, 3.5, 2.2, 1.0, "real image\nx", "#eef2f6")
box(ax, 8.6, 2.35, 2.4, 1.1, "discriminator\nD", "#f4dede")
arrow(ax, (1.9, 1.2), (2.45, 1.2)); arrow(ax, (4.35, 1.2), (4.8, 1.2))
arrow(ax, (7.0, 1.4), (7.4, 1.95)); arrow(ax, (7.0, 3.3), (7.4, 2.75))
ax.text(8.6, 1.4, "P(real)", ha="center", fontsize=14)
ax.annotate("", xy=(3.4, 0.55), xytext=(8.6, 1.2), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2, ls="--", connectionstyle="arc3,rad=-0.25", mutation_scale=18))
ax.text(5.6, -0.3, "its gradient tells G how to look more real", color=RED, fontsize=13, ha="center")
save(fig, "fig-game.svg")

# ------------------------------------------------------------ 2. snapshots of training
snaps = D["converge"]["snaps"]
fig, axs = plt.subplots(2, 4, figsize=(1000 / 96, 380 / 96), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1]))
for j, s in enumerate(snaps):
    ax = axs[0, j]
    ax.fill_between(xs, 0, D["one"], color=GREY, alpha=0.6, lw=0)
    ax.plot(xs, np.minimum(s["gen"], 1.6), color=BLUE, lw=2.5)
    ax.set_ylim(0, 1.6); ax.set_title(f"step {s['step']}: a {s['a']:.2f}, b {s['b']:.2f}", fontsize=13)
    if j == 0: ax.set_ylabel("density")
    else: ax.set_yticklabels([])
    ax2 = axs[1, j]
    ax2.plot(xs, s["disc"], color=RED, lw=2.5); ax2.axhline(0.5, color=GREY, ls="--", lw=1)
    ax2.set_ylim(0, 1); ax2.set_xlabel("x")
    if j == 0: ax2.set_ylabel("D(x)")
    else: ax2.set_yticklabels([])
axs[0, 0].text(1.7, 1.25, "data", color="#666", fontsize=13); axs[0, 0].text(-2.9, 0.5, "generator", color=BLUE, fontsize=13)
fig.tight_layout(); save(fig, "fig-snapshots.svg")

# ------------------------------------------------------------ 3. the best discriminator
fig, ax = plt.subplots(figsize=(470 / 96, 300 / 96))
p = np.array(D["one"]); q = np.exp(-0.5 * ((xs + 0.0) / 1.0) ** 2) / np.sqrt(2 * np.pi)
ax.fill_between(xs, 0, p, color=GREY, alpha=0.6, lw=0, label="data p")
ax.plot(xs, q, color=BLUE, lw=2.5, label="generator q = N(0, 1)")
ax.plot(xs, p / (p + q), color=RED, lw=2.5, label="best D* = p / (p + q)")
ax.axhline(0.5, color=GREY, ls="--", lw=1)
ax.set_xlabel("x"); ax.set_ylim(0, 1.05); ax.legend(fontsize=12, loc="upper left", frameon=False)
fig.tight_layout(); save(fig, "fig-optimal.svg")

# ------------------------------------------------------------ 4. saturating vs non-saturating
fig, axs = plt.subplots(1, 2, figsize=(820 / 96, 300 / 96))
d = np.linspace(0.001, 0.999, 400)
axs[0].plot(d, np.log(1 - d), color=GREY, lw=2.5, label="original: log(1 − D)")
axs[0].plot(d, -np.log(d), color=BLUE, lw=2.5, label="non-saturating: −log D")
axs[0].set_xlabel("D(G(z)): how real the sample looks"); axs[0].set_ylabel("generator loss"); axs[0].set_ylim(-4, 5)
axs[0].legend(fontsize=12, frameon=False)
axs[1].plot(d, d, color=GREY, lw=2.5, label="original: D")
axs[1].plot(d, 1 - d, color=BLUE, lw=2.5, label="non-saturating: 1 − D")
axs[1].scatter([0.01, 0.01], [0.01, 0.99], color=[GREY, BLUE], zorder=3, s=40)
axs[1].text(0.05, 0.78, "0.99", color=BLUE); axs[1].text(0.04, 0.06, "0.01", color="#777")
axs[1].set_xlabel("D(G(z))"); axs[1].set_ylabel("push on the logit"); axs[1].legend(fontsize=12, frameon=False, loc="center right")
axs[1].text(0.02, 1.08, "early: D(G(z)) ≈ 0.01", fontsize=12, color=INK)
fig.tight_layout(); save(fig, "fig-saturate.svg")

# ------------------------------------------------------------ 5. parameter paths
fig, ax = plt.subplots(figsize=(420 / 96, 330 / 96))
for key, col, lab in [("converge", BLUE, "non-saturating"), ("minimax", GREY, "original (minimax)")]:
    P = np.array(D[key]["path"]); ax.plot(P[:, 1], P[:, 0], color=col, lw=2, label=lab)
    for s in [25, 100]:
        ax.scatter([P[s, 1]], [P[s, 0]], color=col, s=30, zorder=3); ax.text(P[s, 1] + 0.05, P[s, 0] + 0.03, str(s), fontsize=11, color=col)
ax.scatter([-1], [1], color=INK, s=40, zorder=4); ax.text(-0.95, 1.05, "start", fontsize=12)
ax.scatter([1], [0.5], marker="*", s=220, color=INK, zorder=4); ax.text(1.05, 0.55, "data", fontsize=12)
ax.set_xlabel("b (centre)"); ax.set_ylabel("a (spread)"); ax.set_xlim(-1.3, 1.5); ax.set_ylim(0, 1.5)
ax.legend(fontsize=12, frameon=False, loc="lower left")
fig.tight_layout(); save(fig, "fig-path.svg")
print("wrote gan figures")
