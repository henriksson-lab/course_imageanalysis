"""Slide figures of diffusion-inverse/lec.qmd (diffusion-models day).

Run from the repository root:
  node_modules/.bin/esbuild tools/figs/diffusion-inverse/data.ts --bundle --platform=node --log-level=warning | node > /tmp/di.json
  python3 -I tools/figs/diffusion-inverse/figs.py /tmp/di.json

fig-answers: MAP, posterior mean and posterior samples for one value, Gaussian prior against a
  two-mode prior (analytic).
fig-detail: the posterior1d demo's "below" preset (from the demo's model code, data.ts).
fig-coverage: coverage of the 90 % band for the three priors of posterior1d (data.ts).
fig-postsample.svg is hand-written (diffusion-inverse/fig-postsample.svg).
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

OUT = "diffusion-inverse/"
BLUE, ORANGE, VERM, GREEN, GREY = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#8a8a8a"
plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "legend.fontsize": 15, "legend.frameon": False})


def save(fig, name):
    fig.savefig(OUT + name, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# 1. three answers: Gaussian prior against a two-mode prior ----------------------------------
x = np.linspace(-1.2, 3.2, 1500)
sn, y = 0.5, 1.1
lik = norm.pdf(y, x, sn)
fig, axs = plt.subplots(2, 1, figsize=(5.6, 4.6), sharex=True, gridspec_kw={"hspace": 0.45})
rng = np.random.default_rng(4)
priors = [("Gaussian prior", [1.0], [1.0], [1.0]),
          ("two-mode prior (spot off / on)", [0.5, 0.5], [0.0, 2.0], [0.15 ** 2, 0.15 ** 2])]
for ax, (title, w, mu, v) in zip(axs, priors):
    w, mu, v = map(np.asarray, (w, mu, v))
    prior = sum(wk * norm.pdf(x, m, np.sqrt(vk)) for wk, m, vk in zip(w, mu, v))
    # exact posterior: a mixture again
    pw = w * norm.pdf(y, mu, np.sqrt(v + sn ** 2)); pw /= pw.sum()
    pm = mu + v / (v + sn ** 2) * (y - mu); pv = v * sn ** 2 / (v + sn ** 2)
    post = sum(a * norm.pdf(x, m, np.sqrt(s)) for a, m, s in zip(pw, pm, pv))
    mean = (pw * pm).sum()
    xmap = x[np.argmax(post)]
    sc = 1 / post.max()
    ax.plot(x, prior / prior.max(), color=GREY, lw=2, ls=":", label="prior")
    ax.plot(x, lik / lik.max(), color=ORANGE, lw=2, ls="--", label="fit to the data")
    ax.fill_between(x, post * sc, color=BLUE, alpha=0.2, lw=0)
    ax.plot(x, post * sc, color=BLUE, lw=2.5, label="posterior")
    k = rng.choice(len(pw), 8, p=pw)
    smp = pm[k] + np.sqrt(pv[k]) * rng.standard_normal(8)
    ax.plot(smp, np.full(8, -0.1), "|", color=BLUE, ms=16, mew=2.5, clip_on=False)
    ax.axvline(mean, color=VERM, lw=2.2)
    ax.axvline(xmap, color="k", lw=1.6, ls="-.")
    ax.set_ylim(-0.18, 1.3); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(title, fontsize=16, loc="left")
    if len(w) > 1:
        ax.text(mean - 0.06, 1.17, "mean", color=VERM, ha="right", fontsize=15)
        ax.text(xmap + 0.06, 1.17, "MAP", ha="left", fontsize=15)
        ax.text(2.4, -0.1, "samples", color=BLUE, fontsize=15, va="center")
    else:
        ax.text(mean + 0.06, 1.17, "mean = MAP", color=VERM, fontsize=15)
axs[0].legend(loc="lower left", bbox_to_anchor=(-0.02, 1.12), ncol=3, fontsize=15, handlelength=1.3, columnspacing=0.8)
axs[1].set_xlabel(f"value x   (measured y = {y}, noise {sn})")
save(fig, "fig-answers.svg")

# 2. where generated detail lives (posterior1d, "below") ------------------------------------
d = json.load(open(sys.argv[1]))
pos = np.arange(64)
pick = [i for i, c in enumerate(d["counts"]) if c == 2][:3] + [i for i, c in enumerate(d["counts"]) if c == 1][:2]
fig, axs = plt.subplots(3, 1, figsize=(4.4, 5.6), sharex=True, gridspec_kw={"hspace": 0.42, "height_ratios": [1, 1, 0.8]})
ax = axs[0]
ax.plot(pos, d["meas"], color=GREY, lw=3, label="measured")
ax.plot(pos, d["mean"], color=BLUE, lw=2.2, label="mean")
ax.plot(pos, d["truth"], color="k", lw=1.6, ls="--", label="truth")
ax.set_title("truth: two spots, 4 apart", fontsize=16, loc="left")
ax.legend(loc="upper left", fontsize=14, handlelength=1.0, borderaxespad=0, ncol=3, columnspacing=0.6, handletextpad=0.4)
ax = axs[1]
ax.fill_between(pos, d["lo"], d["hi"], color=GREY, alpha=0.3, lw=0, label="90 % band")
for i in pick:
    c = BLUE if d["counts"][i] == 2 else VERM
    ax.plot(pos, d["samples"][i], color=c, lw=1.6)
ax.set_title("samples: 24 of 32 show two", fontsize=16, loc="left")
ax = axs[2]
for i in pick:
    c = BLUE if d["counts"][i] == 2 else VERM
    ax.plot(pos, d["nullS"][i], color=c, lw=1.6)
ax.plot(pos, d["nullT"], color="k", lw=1.6, ls="--")
ax.set_title("unseen part: prior only", fontsize=16, loc="left")
ax.set_xlim(14, 50)
ax.set_xlabel("position (samples)")
for ax in axs[:2]:
    ax.set_ylim(-0.1, 1.15); ax.set_yticks([0, 1])
axs[0].set_ylim(-0.1, 1.5)
axs[2].set_ylim(-0.6, 0.6); axs[2].set_yticks([-0.5, 0, 0.5])
save(fig, "fig-detail.svg")
print("picked", pick)

# 3. coverage of the 90 % band --------------------------------------------------------------
cov = d["coverage"]
labels = ["matched prior", "no close pairs", "wider spots"]
vals = [100 * cov["matched"], 100 * cov["noclose"], 100 * cov["wide"]]
fig, ax = plt.subplots(figsize=(3.4, 2.7))
for i, (v, c) in enumerate(zip(vals, [BLUE, ORANGE, VERM])):
    ax.plot([60, v], [i, i], color=c, lw=1.5, alpha=0.5)
    ax.plot([v], [i], "o", color=c, ms=13)
    ax.text(v - 1.2, i - 0.32, f"{v:.1f} %", ha="right", va="center", fontsize=15)
ax.axvline(90, color="k", ls="--", lw=1.6)
ax.text(89.5, -1.0, "promised 90 %", fontsize=15, va="center", ha="right")
ax.set_yticks(range(3)); ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlim(60, 100); ax.set_ylim(2.5, -1.25)
ax.set_xlabel("truth inside band (%)")
save(fig, "fig-coverage.svg")
