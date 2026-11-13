"""Slide figures of score-matching/lec.qmd (diffusion-models day).

Run from the repository root:  python3 -I tools/figs/score-matching/figs.py

All numbers are analytic and match the slide text: schedules of lib/diffusion.ts
(linear: abar = exp(-(0.1 t + 9.95 t^2)); cosine with s = 0.008), the power spectrum
1000/f^2 with noise floor 1/SNR, and the two-mode data of the scorematch1d demo
(modes at +-1.5, sd 0.25). The micrograph is demos/src/fourier2d/cell256.png
(Robert M. Hunt, CC BY 3.0, via Wikimedia Commons).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from PIL import Image

OUT = "score-matching/"
BLUE, ORANGE, VERM, GREEN, GREY = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#8a8a8a"
plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "legend.fontsize": 15, "legend.frameon": False})


def save(fig, name):
    fig.savefig(OUT + name, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# 1. noise level = SNR: the micrograph at sigma 0.5, 1, 2, and log-SNR of the two schedules --
cell = np.asarray(Image.open("demos/src/fourier2d/cell256.png").convert("L"), float) / 255
cell = 2 * cell - 1                                    # data scaled to [-1, 1]
rng = np.random.default_rng(5)
eps = rng.standard_normal(cell.shape)
fig = plt.figure(figsize=(5.6, 5.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.3, wspace=0.06)
for k, (ab, s) in enumerate([(0.8, 0.5), (0.5, 1), (0.2, 2)]):
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(cell + s * eps, cmap="gray", vmin=-2.5, vmax=2.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"σ = {s:g}\nSNR {ab / (1 - ab):g}", fontsize=16)
ax = fig.add_subplot(gs[1, :])
t = np.linspace(0.001, 0.999, 500)
ab_lin = np.exp(-(0.1 * t + 9.95 * t ** 2))
sc = 0.008
f = lambda u: np.cos((u + sc) / (1 + sc) * np.pi / 2) ** 2
ab_cos = f(t) / f(0)
ax.plot(t, np.log(ab_cos / (1 - ab_cos)), color=BLUE, lw=2.5, label="cosine")
ax.plot(t, np.log(ab_lin / (1 - ab_lin)), color=ORANGE, lw=2.5, ls="--", label="linear")
ax.axhline(0, color=GREY, lw=1)
for tt, c in [(0.259, ORANGE), (0.50, BLUE)]:
    ax.plot([tt], [0], "o", color=c, ms=8, zorder=5)
    ax.annotate(f"τ = {tt:.2f}", (tt, 0), xytext=(tt + 0.03, 2.2), fontsize=15, color="k")
ax.text(0.99, 0.35, "SNR 1", ha="right", va="bottom", fontsize=15, color="#555")
ax.set_xlim(0, 1); ax.set_ylim(-8, 6)
ax.set_xlabel("time τ"); ax.set_ylabel("log SNR")
ax.legend(loc="lower left")
save(fig, "fig-snr.svg")

# 2. what noise destroys first: power 1000/f^2 against three noise floors -------------------
fig, ax = plt.subplots(figsize=(5.6, 4.1))
fr = np.logspace(0, np.log10(128), 200)
ax.loglog(fr, 1000 / fr ** 2, color="k", lw=2.5, label="image power 1000/f²")
for snr, c, ls in [(1, BLUE, "-"), (0.1, ORANGE, "--"), (0.01, VERM, ":")]:
    floor = 1 / snr
    fs = np.sqrt(1000 * snr)
    ax.axhline(floor, color=c, lw=2, ls=ls)
    ax.plot([fs], [floor], "o", color=c, ms=8)
    ax.text(100, floor * 1.25, f"SNR {snr:g}", color=c, ha="right", va="bottom", fontsize=15)
    ax.annotate(f"{fs:.1f}", (fs, floor), xytext=(fs * 0.8, floor * 0.22), ha="center", fontsize=15, color=c)
ax.set_xlim(1, 128); ax.set_ylim(0.05, 2000)
ax.set_xlabel("frequency f (cycles per image)")
ax.set_ylabel("power")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.text(1.15, 0.08, "power below a floor is lost", fontsize=15, color="#555")
save(fig, "fig-spectrum.svg")

# 3. best denoiser for the two-mode data at sigma = 0.6: conditional mean against Wiener -----
mu, v, sig = np.array([-1.5, 1.5]), 0.25 ** 2, 0.6


def cond_mean(x, s):
    c = v + s * s
    lw = -0.5 * (x[:, None] - mu) ** 2 / c
    r = np.exp(lw - lw.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
    return (r * (mu + v / c * (x[:, None] - mu))).sum(1)


P = 1.5 ** 2 + v                                       # data variance 2.3125
x = np.linspace(-3.5, 3.5, 400)
fig, ax = plt.subplots(figsize=(5.4, 4.3))
ax.plot(x, x, color=GREY, lw=1.2, ls=":", label="no denoising")
ax.plot(x, P / (P + sig ** 2) * x, color=ORANGE, lw=2.5, ls="--", label=f"Wiener (×{P / (P + sig ** 2):.2f})")
ax.plot(x, cond_mean(x, sig), color=BLUE, lw=3, label="best: conditional mean")
for m in mu:
    ax.axhspan(m - 0.25, m + 0.25, color=GREEN, alpha=0.15, lw=0)
ax.text(-3.4, 1.5, "data modes (shaded)", color=GREEN, va="center", ha="left", fontsize=15)
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-4.2, 3)
ax.set_xlabel("noisy value x  (σ = 0.6)")
ax.set_ylabel("denoised value")
ax.legend(loc="lower right")
save(fig, "fig-denoiser.svg")

# 4. denoising score matching: pairs, their binned average and the exact denoiser -----------
rng = np.random.default_rng(11)
n = 400
x0 = rng.choice(mu, n) + 0.25 * rng.standard_normal(n)
xn = x0 + sig * rng.standard_normal(n)
fig, ax = plt.subplots(figsize=(5.6, 4.3))
ax.scatter(xn, x0, s=10, color=GREY, alpha=0.55, lw=0, label="pairs (noisy, clean)")
edges = np.linspace(-3.5, 3.5, 21)
mids, avg = [], []
for a, b in zip(edges[:-1], edges[1:]):
    sel = (xn >= a) & (xn < b)
    if sel.sum() >= 3:
        mids.append((a, b)); avg.append(x0[sel].mean())
for (a, b), m in zip(mids, avg):
    ax.plot([a, b], [m, m], color=GREEN, lw=3.5, solid_capstyle="butt")
ax.plot([], [], color=GREEN, lw=3.5, label="average per bin")
ax.plot(x, cond_mean(x, sig), color=BLUE, lw=2, label="exact denoiser")
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-2.6, 2.6)
ax.set_xlabel("noisy value x = x₀ + σε  (σ = 0.6)")
ax.set_ylabel("clean value x₀")
ax.legend(loc="upper left", handlelength=1.5)
save(fig, "fig-dsm.svg")
print("D(1.5) at sigma 2.5:", cond_mean(np.array([1.5]), 2.5))
