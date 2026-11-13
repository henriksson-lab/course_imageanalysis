"""Slide figures of guidance/lec.qmd (diffusion-models day).

Run from the repository root:  python3 -I tools/figs/guidance/figs.py

fig-cfg: the classifier-free guidance example of the slide (unconditional 0.2, conditional 0.5,
  w = 3 -> 1.4) on a number line.
fig-sharpen: the sharpened density p(x | A) p(A | x)^w for two classes N(+-1, 1) with equal
  weights, class A at +1, w = 0, 4, 8; ratio of the density at x = 2 to x = 1 as on the slide.
fig-latent.svg is hand-written (guidance/fig-latent.svg).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "guidance/"
BLUE, ORANGE, VERM, GREEN, GREY = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#8a8a8a"
plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "legend.fontsize": 15, "legend.frameon": False})


def save(fig, name):
    fig.savefig(OUT + name, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# 1. classifier-free guidance on a number line -----------------------------------------------
u, c, w = 0.2, 0.5, 3
g = (1 + w) * c - w * u
fig, ax = plt.subplots(figsize=(5.6, 2.3))
ax.axhline(0, color="k", lw=1.2)
ax.annotate("", xy=(c, 0), xytext=(u, 0), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2.5, mutation_scale=20))
ax.annotate("", xy=(g, 0), xytext=(c, 0), arrowprops=dict(arrowstyle="-|>", color=VERM, lw=2.5, ls="--", mutation_scale=20))
for x, col, lab, up in [(u, GREY, "no label\n0.2", False), (c, BLUE, "label y\n0.5", True), (g, VERM, "guided, w = 3\n1.4", True)]:
    ax.plot([x], [0], "o", color=col, ms=11, zorder=5)
    ax.text(x, 0.12 if up else -0.1, lab, ha="center", va="bottom" if up else "top", fontsize=15, color="k")
ax.text((c + g) / 2 + 0.05, -0.1, "+ w × (label − no label)", ha="center", va="top", fontsize=14, color=VERM)
ax.set_xlim(0, 1.6); ax.set_ylim(-0.42, 0.55)
ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("predicted noise ε̂ (one pixel)")
save(fig, "fig-cfg.svg")

# 2. the sharpened density for w = 0, 4, 8 --------------------------------------------------
x = np.linspace(-3, 6, 1200)
pA = np.exp(-0.5 * (x - 1) ** 2)
post = 1 / (1 + np.exp(-2 * x))                       # p(A | x)
fig, ax = plt.subplots(figsize=(5.4, 4.0))
ax.plot(x, np.exp(-0.5 * (x + 1) ** 2) / np.sqrt(2 * np.pi), color=GREY, lw=1.8, ls=":", label="class B (w = 0)")
for wv, col, ls in [(0, BLUE, "-"), (4, ORANGE, "--"), (8, VERM, "-.")]:
    d = pA * post ** wv
    d /= np.trapezoid(d, x)
    r = (np.exp(-0.5) * ((1 + np.exp(-2)) / (1 + np.exp(-4))) ** wv)
    ax.plot(x, d, color=col, lw=2.6, ls=ls, label=f"w = {wv}:  p(2)/p(1) = {r:.2f}")
ax.axvline(1, color="k", lw=1, alpha=0.5)
ax.axvline(2, color="k", lw=1, alpha=0.5)
ax.text(1, 0.62, "class\nmean", ha="right", va="top", fontsize=14)
ax.text(2.05, 0.62, "x = 2", ha="left", va="top", fontsize=14)
ax.set_xlim(-3, 5); ax.set_ylim(0, 0.65)
ax.set_xlabel("x"); ax.set_ylabel("density of class A (at +1)")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), fontsize=15)
save(fig, "fig-sharpen.svg")
