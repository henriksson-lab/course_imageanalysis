"""Slide figures of diffusion-sampling/lec.qmd (diffusion-models day).

Run from the repository root:
  node_modules/.bin/esbuild tools/figs/diffusion-sampling/data.ts --bundle --platform=node --log-level=warning | node > /tmp/ds.json
  python3 -I tools/figs/diffusion-sampling/figs.py /tmp/ds.json

fig-steps: Euler steps of the probability-flow ODE for Gaussian data N(0, 1) from sigma = 3 (analytic;
  numbers as on the slide "Why steps matter").
fig-forget: 30 paths of the samplerpaths demo's "forget" preset (three unequal modes, start noise
  0.2 times too small, 20 steps), deterministic and ancestral, from the demo's model code.
fig-frechet: data at +-2 (sd 0.5) and the Gaussian N(0, 4.25) with the same mean and variance.
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

OUT = "diffusion-sampling/"
BLUE, ORANGE, VERM, GREEN, GREY = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#8a8a8a"
plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "legend.fontsize": 15, "legend.frameon": False})


def save(fig, name):
    fig.savefig(OUT + name, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# 1. why steps matter --------------------------------------------------------------------------
def euler(grid, x):
    xs = [x]
    for a, b in zip(grid[:-1], grid[1:]):
        x = x * (1 - (a - b) * a / (1 + a * a))
        xs.append(x)
    return np.array(xs)


x0 = np.sqrt(10)                                     # a path starting one spread out at sigma = 3
fig, (ax, bx) = plt.subplots(2, 1, figsize=(5.2, 5.6), gridspec_kw={"hspace": 0.55})
s = np.linspace(3, 0, 200)
ax.plot(s, np.sqrt(1 + s ** 2), color="k", lw=2.5, label="true path")
for g, c, ls, lab in [([3, 0], VERM, "--", "1 step"), ([3, 1.5, 0], ORANGE, "-.", "2 steps"),
                      (np.linspace(3, 0, 11), BLUE, "-", "10 steps")]:
    g = np.asarray(g, float)
    ax.plot(g, euler(g, x0), color=c, ls=ls, lw=2.2, marker="o", ms=5, label=lab)
ax.set_xlim(3.05, -0.05); ax.set_ylim(0, 3.4)
ax.set_xlabel("noise level σ"); ax.set_ylabel("position")
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), handlelength=1.4, fontsize=15, ncol=2, columnspacing=0.8)
N = np.array([1, 2, 3, 5, 10, 20, 50, 100])
spr = [euler(np.linspace(3, 0, n + 1), x0)[-1] for n in N]
bx.semilogx(N, spr, color=BLUE, marker="o", lw=2.2)
bx.axhline(1, color=GREY, ls=":", lw=1.5)
bx.text(100, 1.03, "exact 1", ha="right", va="bottom", fontsize=15, color="#555")
for n, v in zip(N, spr):
    if n in (1, 2, 10, 100):
        bx.annotate(f"{v:.2f}", (n, v), xytext=(6, -16), textcoords="offset points", fontsize=14)
bx.set_ylim(0, 1.15); bx.set_xlim(0.8, 130)
bx.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
bx.set_xlabel("equal steps"); bx.set_ylabel("spread of samples")
save(fig, "fig-steps.svg")

# 2. stochastic forgets a bad start ----------------------------------------------------------
d = json.load(open(sys.argv[1]))
mix = d["mix"]
mean = sum(w * m for w, m in zip(mix["w"], mix["mu"]))
sd_data = np.sqrt(sum(w * (v + m * m) for w, m, v in zip(mix["w"], mix["mu"], mix["v"])) - mean ** 2)
fig, axs = plt.subplots(2, 1, figsize=(5.4, 5.4), sharex=True, gridspec_kw={"hspace": 0.32})
for ax, key, title in [(axs[0], "eta0", "deterministic"), (axs[1], "eta1", "stochastic (fresh noise)")]:
    r = d[key]
    for m, v in zip(mix["mu"], mix["v"]):
        ax.axhspan(m - 2 * np.sqrt(v), m + 2 * np.sqrt(v), xmin=0.93, color=GREEN, alpha=0.35, lw=0)
    for p in r["paths"]:
        ax.plot(r["taus"], p, color=BLUE, lw=1, alpha=0.7)
    ax.set_xlim(1, -0.08); ax.set_ylim(-2.6, 2.6)
    ax.set_ylabel("position")
    ax.set_title(f"{title}: sd {r['sd']:.2f}", fontsize=16, loc="left")
axs[1].set_xlabel("time τ (noise → data)")
axs[0].text(-0.075, 2.35, "data", color=GREEN, ha="right", fontsize=14)
save(fig, "fig-forget.svg")
print("data sd", sd_data)

# 3. the Frechet distance is blind to shape -------------------------------------------------
x = np.linspace(-6, 6, 600)
pd = 0.5 * norm.pdf(x, -2, 0.5) + 0.5 * norm.pdf(x, 2, 0.5)
pg = norm.pdf(x, 0, np.sqrt(4.25))
fig, ax = plt.subplots(figsize=(4.6, 3.9))
ax.axvspan(-1, 1, color=GREY, alpha=0.15, lw=0)
ax.fill_between(x, pd, color=BLUE, alpha=0.25, lw=0)
ax.plot(x, pd, color=BLUE, lw=2.5, label="data: modes at ±2")
ax.plot(x, pg, color=ORANGE, lw=2.5, ls="--", label="generator: one Gaussian")
fd = 2 * (norm.cdf(1, 2, 0.5) - norm.cdf(-1, 2, 0.5)) / 2
fg = norm.cdf(1, 0, np.sqrt(4.25)) - norm.cdf(-1, 0, np.sqrt(4.25))
ax.text(0, 0.74, f"in [−1, 1]:\ngenerator {100 * fg:.0f} %\ndata {100 * fd:.0f} %", ha="center", va="top", fontsize=14)
ax.set_xlim(-5, 5); ax.set_ylim(0, 0.75)
ax.set_xlabel("value"); ax.set_ylabel("density")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=1, fontsize=15)
save(fig, "fig-frechet.svg")
