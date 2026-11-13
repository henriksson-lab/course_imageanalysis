# Slide figures for track-analysis/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/track-analysis/figs.py
# Numbers follow the slides and the msdplot demo (D = 0.1 µm²/s, Δt = 0.1 s; σ = 30 nm,
# D = 0.01, Δt = 0.05 s; persistent walk s = 0.2 µm/s, P = 10 s). Sizes are display px / 96.
# (fig-tracks and fig-msd-shapes come from tools/figs/tracking/figs.ts.)
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(R, "track-analysis") + "/"
plt.rcParams.update({"font.size": 17, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "track-analysis"})
INK, BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#1b2a3a", "#1f77b4", "#e69f00", "#2a9d4b", "#d62728", "#cc79a7", "#888"
def save(fig, name):
    fig.savefig(OUT + name, facecolor="white", metadata={"Date": None}); plt.close(fig); print("wrote", OUT + name)

# ------------------------------------------------------------ 1. the speed trap
D = 0.1; dt = np.logspace(-3, 0.3, 200)
fig, ax = plt.subplots(figsize=(520 / 96, 340 / 96))
ax.loglog(dt, np.sqrt(np.pi * D / dt), color=BLUE, lw=3, label="random walk, D = 0.1")
sig = 0.03
ax.loglog(dt, np.sqrt(np.pi * (D * dt + sig ** 2)) / dt, color=ORANGE, lw=2.5, ls="--", label="+ localisation error 30 nm")
for t_, v in [(0.1, 1.77), (0.01, 5.60)]:
    ax.plot(t_, np.sqrt(np.pi * D / t_), "o", color=INK, ms=9)
    ax.text(t_ * 1.25, v * 1.08, f"{v:.2f} µm/s", fontsize=17)
ax.set_xlabel("frame interval Δt (s)"); ax.set_ylabel("apparent speed (µm/s)")
ax.set_xlim(1e-3, 2); ax.set_ylim(0.3, 60)
ax.legend(frameon=False, fontsize=16, loc="upper right")
fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.19)
save(fig, "fig-speedtrap.svg")

# ------------------------------------------------------------ 2. MSD worked example (positions 0, 1, 3, 2, 4)
x = np.array([0, 1, 3, 2, 4])
fig, axs = plt.subplots(1, 2, figsize=(480 / 96, 280 / 96))
axs[0].plot(range(5), x, "-o", color=BLUE, lw=2.5, ms=8)
for t0 in range(3):
    axs[0].annotate("", (t0 + 2, x[t0 + 2]), (t0, x[t0]), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2, connectionstyle="arc3,rad=-0.35"))
axs[0].set_xlabel("frame"); axs[0].set_ylabel("position"); axs[0].set_xticks(range(5)); axs[0].set_yticks([0, 2, 4])
axs[0].text(4.1, -0.2, "lag 2", color=ORANGE, fontsize=17, ha="right")
axs[0].set_ylim(-0.4, 5.0)
msd = [2.5, 11 / 3, 6.5, 16]
axs[1].plot([1, 2, 3, 4], msd, "o", color=INK, ms=9)
axs[1].set_xlabel("lag"); axs[1].set_ylabel("MSD"); axs[1].set_xticks([1, 2, 3, 4]); axs[1].set_ylim(0, 19); axs[1].set_xlim(0.4, 4.5)
fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.22, wspace=0.45)
save(fig, "fig-msd-worked.svg")

# ------------------------------------------------------------ 3. four MSD shapes (log-log)
tau = np.logspace(-1, 1.3, 200)
curves = [(4 * D * tau, "free: slope 1", BLUE, "-"),
          (4 * D * tau + 0.5 ** 2 * tau ** 2, "directed: → slope 2", RED, "-"),
          (0.25 * (1 - np.exp(-4 * D * tau / 0.25)), "confined: plateau R²", GREEN, "-"),
          (4 * D * tau ** 0.5, "anomalous: α = 0.5", PURPLE, "--")]
fig, ax = plt.subplots(figsize=(520 / 96, 360 / 96))
for y, lab, c, ls in curves: ax.loglog(tau, y, color=c, lw=3, ls=ls, label=lab)
ax.axhline(0.25, color=GREEN, lw=1, ls=":")
ax.set_xlabel("lag τ (s)"); ax.set_ylabel("MSD (µm²)"); ax.set_ylim(0.02, 3000)
ax.legend(frameon=False, fontsize=16, loc="upper left")
fig.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.18)
save(fig, "fig-msdmodels.svg")

# ------------------------------------------------------------ 4. localisation error: the offset
n = np.arange(0, 9); Dl, dtl, s_ = 0.01, 0.05, 0.03
true = 4 * Dl * dtl * n + 4 * s_ ** 2
k = np.arange(1, 5); slope0 = np.sum(k * true[1:5]) / np.sum(k * k)
fig, ax = plt.subplots(figsize=(520 / 96, 340 / 96))
ax.plot(n[1:] * dtl, true[1:], "o", color=INK, ms=8, label="measured MSD")
ax.plot(n * dtl, true, color=BLUE, lw=2.5, label=f"with offset: D = {Dl}")
ax.plot(n * dtl, slope0 * n, color=RED, lw=2.5, ls="--", label=f"through 0: D = {slope0 / (4 * dtl):.3f}")
ax.annotate("offset 4σ² = 0.0036", (0, 4 * s_ ** 2), (0.12, 0.0012), arrowprops=dict(arrowstyle="->", color=INK), fontsize=17)
ax.set_xlabel("lag τ (s)"); ax.set_ylabel("MSD (µm²)"); ax.set_xlim(0, 0.42); ax.set_ylim(0, 0.031)
ax.legend(frameon=False, fontsize=16, loc="upper left")
fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.19)
save(fig, "fig-offset.svg")
print("forced D", slope0 / (4 * dtl))

# ------------------------------------------------------------ 5. ensemble vs single tracks: two populations
rng = np.random.default_rng(7)
N, dt5, ntr = 200, 0.1, 300
Ds = np.where(rng.random(ntr) < 0.3, 0.01, 0.1)
def track_D(Dt):
    steps = rng.normal(0, np.sqrt(2 * Dt * dt5), (N - 1, 2)); r = np.vstack([[0, 0], np.cumsum(steps, 0)])
    m = [np.mean(np.sum((r[l:] - r[:-l]) ** 2, 1)) for l in range(1, 5)]
    return np.sum(np.arange(1, 5) * m) / np.sum(np.arange(1, 5) ** 2) / (4 * dt5), r
fits = [track_D(d) for d in Ds]
est = np.array([f[0] for f in fits])
ens = np.mean([[np.sum((f[1][l:] - f[1][:-l]) ** 2, 1).mean() for l in range(1, 5)] for f in fits], 0)
Dens = np.sum(np.arange(1, 5) * ens) / np.sum(np.arange(1, 5) ** 2) / (4 * dt5)
fig, ax = plt.subplots(figsize=(520 / 96, 340 / 96))
bins = np.logspace(-2.6, -0.4, 34)
ax.hist(est, bins=bins, color=BLUE, alpha=0.75)
ax.set_xscale("log"); ax.axvline(Dens, color=RED, lw=3, ls="--")
ax.text(Dens / 1.12, ax.get_ylim()[1] * 0.98, f"ensemble\nD = {Dens:.3f}", color=RED, fontsize=17, va="top", ha="right")
ax.text(0.0105, ax.get_ylim()[1] * 0.6, "30 % slow\n(D = 0.01)", ha="center", fontsize=16)
ax.set_xlim(2.5e-3, 0.9); ax.text(0.15, ax.get_ylim()[1] * 0.7, "70 % free\n(D = 0.1)", ha="left", fontsize=16, va="top")
ax.set_xlabel("D of single tracks (µm²/s)"); ax.set_ylabel("tracks")
fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.19)
save(fig, "fig-ensemble.svg")
print("ensemble D", Dens)

# ------------------------------------------------------------ 6. persistent random walk
s, P = 0.2, 10.0
tau = np.logspace(-0.5, 3, 300)
msd = 2 * s * s * P * (tau - P * (1 - np.exp(-tau / P)))
fig, ax = plt.subplots(figsize=(520 / 96, 340 / 96))
ax.loglog(tau, msd, color=BLUE, lw=3, label="MSD")
ax.loglog(tau[tau < 6], (s * tau[tau < 6]) ** 2 * 1.6, color=GREY, lw=2, ls=":")
ax.loglog(tau[tau > 30], 2 * s * s * P * tau[tau > 30] * 0.55, color=GREY, lw=2, ls=":")
ax.text(0.35, 1.0, "slope 2\n(ballistic)", fontsize=16, color=GREY, va="bottom")
ax.text(250, 8, "slope 1\n(random)", fontsize=16, color=GREY, ha="center")
ax.axvline(P, color=RED, lw=2, ls="--"); ax.text(P / 1.15, 300, "P = 10 s", ha="right", color=RED, fontsize=17)
ax.set_xlabel("lag τ (s)"); ax.set_ylabel("MSD (µm²)")
fig.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.19)
save(fig, "fig-persistent.svg")
