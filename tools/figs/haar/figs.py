# Extra slide figures for haar/lec.qmd. Run: python3 -I tools/figs/haar/figs.py
# Shares the helpers (filters, dwt1, colours, rcParams) of tools/figs/wavelets/figs.py.
import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "..", "wavelets", "figs.py")).read().split("# ---- haar/fig-spike-step.svg ----")[0])
from PIL import Image

# ---- haar/fig-global.svg: Fourier basis waves span the whole signal; an image row is flat + jumps ----
cell = np.array(Image.open(R + "demos/src/fourier2d/cell256.png")).astype(float)
row = 160
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw={"width_ratios": [1.15, 0.8, 1.35]})
n = 64; t = np.arange(n)
for i, k in enumerate([1, 4, 12]):
    ax[0].plot(t, 2.6 * i + np.cos(2 * np.pi * k * t / n), color=[BLUE, PURPLE, RED][i], lw=2.2)
    ax[0].text(n + 1, 2.6 * i, f"k = {k}", va="center", fontsize=15, color=[BLUE, PURPLE, RED][i])
ax[0].set_xlim(0, n + 9); ax[0].set_yticks([]); ax[0].spines["left"].set_visible(False)
ax[0].set_xlabel("position n"); ax[0].set_title("three Fourier basis waves", fontsize=15)
ax[1].imshow(cell, cmap="gray", vmin=0, vmax=255); ax[1].axhline(row, color=ORANGE, lw=2.5); ax[1].axis("off")
ax[1].set_title("micrograph", fontsize=15)
ax[2].plot(cell[row], color="k", lw=1.6); ax[2].set_xlim(0, 255); ax[2].set_ylim(0, 270)
ax[2].set_xlabel("position along the orange line"); ax[2].set_ylabel("grey value")
ax[2].set_title("an image row: plateaus and jumps", fontsize=15)
fig.tight_layout(); save(fig, "haar/fig-global.svg")

# ---- haar/fig-atoms.svg: STFT atoms (same window) vs wavelet atoms (same number of cycles) ----
tt = np.linspace(0, 1, 2000)
fig, ax = plt.subplots(1, 2, figsize=(6.6, 5.0), sharey=True)
freqs = [3, 8, 24]; cols = [BLUE, PURPLE, RED]; names = ["low", "medium", "high"]
for i, (f, c) in enumerate(zip(freqs, cols)):
    y0 = 2.6 * i
    sig = 0.09  # STFT: same Gaussian window
    ax[0].plot(tt, y0 + np.exp(-(tt - 0.5) ** 2 / (2 * sig ** 2)) * np.cos(2 * np.pi * f * (tt - 0.5)), color=c, lw=2)
    ax[0].plot(tt, y0 + np.exp(-(tt - 0.5) ** 2 / (2 * sig ** 2)), color=GREY, lw=1, ls="--")
    sw = 0.8 / f  # wavelet: window width proportional to 1/f (same number of cycles)
    ax[1].plot(tt, y0 + np.exp(-(tt - 0.5) ** 2 / (2 * sw ** 2)) * np.cos(2 * np.pi * f * (tt - 0.5)), color=c, lw=2)
    ax[1].plot(tt, y0 + np.exp(-(tt - 0.5) ** 2 / (2 * sw ** 2)), color=GREY, lw=1, ls="--")
    ax[0].text(0.08, y0, names[i], ha="right", va="center", fontsize=15, color=c)
for a in ax: a.set_yticks([]); a.spines["left"].set_visible(False); a.set_xticks([]); a.set_xlabel("position")
ax[0].set_title("short-time\nFourier", fontsize=15); ax[1].set_title("wavelets", fontsize=15)
ax[0].set_xlim(0.1, 0.9); ax[1].set_xlim(0.1, 0.9)
fig.tight_layout(); save(fig, "haar/fig-atoms.svg")

# ---- haar/fig-byhand.svg: the board example, averages level by level ----
x = np.array([4, 6, 10, 12, 8, 6, 5, 5], float)
fig, ax = plt.subplots(4, 1, figsize=(5.4, 6.6), sharex=True)
plt.rcParams["font.size"] = 17
cur = x.copy(); labels = ["x", "level 1", "level 2", "level 3"]
for r in range(4):
    a = ax[r]
    a.bar(np.arange(8), x, width=0.85, color="#d9d9d9", ec="#bdbdbd")
    if r > 0:
        s = cur.reshape(-1, 2); d = (s[:, 0] - s[:, 1]) / 2; cur = s.mean(axis=1)
        w = 8 // len(cur)
        for i, v in enumerate(cur):
            a.plot([i * w - 0.45, (i + 1) * w - 0.55], [v, v], color=BLUE, lw=3.5)
            a.text(i * w + w / 2 - 0.5, v + 0.8, f"{v:g}", ha="center", fontsize=17, color=BLUE)
        a.text(7.8, 7, "d: " + ", ".join(f"{v:g}" for v in d), fontsize=17, color=RED, va="center")
    else:
        for i, v in enumerate(x): a.text(i, v + 0.8, f"{v:g}", ha="center", fontsize=17)
    a.set_ylim(0, 16); a.set_yticks([]); a.spines["left"].set_visible(False)
    a.text(-0.9, 8, labels[r], ha="right", va="center", fontsize=17, color=BLUE if r else "k")
ax[-1].set_xticks(range(8)); ax[-1].set_xticklabels([str(i + 1) for i in range(8)]); ax[-1].set_xlabel("sample")
ax[-1].set_xlim(-0.6, 12.2)
fig.subplots_adjust(left=0.2, right=0.99, hspace=0.25); save(fig, "haar/fig-byhand.svg"); plt.rcParams["font.size"] = 15

# ---- haar/fig-sparse.svg: Haar coefficients of a piecewise-constant signal, a spike and a ramp ----
n = 32; L = 5; t = np.arange(n)
pc = np.where(t < 9, 1.0, 3.0); pc[t >= 22] = 2.0
spike = np.zeros(n); spike[13] = 3.0
ramp = t * 3.0 / (n - 1)
lev = np.zeros(n, int); lev[0] = L + 1
for j in range(1, L + 1): lev[n >> j: n >> (j - 1)] = j
lcol = {6: GREY, 5: "#8c564b", 4: PURPLE, 3: GREEN, 2: ORANGE, 1: BLUE}
fig, ax = plt.subplots(3, 2, figsize=(8.4, 6.4), gridspec_kw={"width_ratios": [1, 1.25]})
for r, (sig, name) in enumerate([(pc, "piecewise constant"), (spike, "spike"), (ramp, "ramp")]):
    ax[r, 0].bar(t, sig, color="#bdbdbd", width=0.85); ax[r, 0].set_ylim(0, 4.3); ax[r, 0].set_yticks([])
    ax[r, 0].text(0.02, 0.97, name, transform=ax[r, 0].transAxes, fontsize=15, va="top")
    c = dwt1(sig, H["haar"], L); nz = np.abs(c) > 1e-9
    for i in range(n):
        if nz[i]: ax[r, 1].plot([i, i], [0, c[i]], color=lcol[lev[i]], lw=2.5); ax[r, 1].plot(i, c[i], "o", color=lcol[lev[i]], ms=5)
        else: ax[r, 1].plot(i, 0, "o", mfc="white", mec="#bbbbbb", ms=4)
    ax[r, 1].axhline(0, color="k", lw=0.8)
    for j in range(1, L + 1): ax[r, 1].axvline((n >> j) - 0.5, color="#dddddd", lw=1, zorder=0)
    ax[r, 1].text(0.99, 0.97, f"{int(nz.sum())} nonzero", transform=ax[r, 1].transAxes, ha="right", va="top", fontsize=15)
    lim = np.abs(c).max() * 1.15; ax[r, 1].set_ylim(-lim, lim); ax[r, 1].set_yticks([])
    print(name, "nonzero", int(nz.sum()))
ax[0, 1].set_title("Haar coefficients", fontsize=15)
ax[0, 0].set_title("signal (32 samples)", fontsize=15)
for a in ax.ravel(): a.set_xticks([0, 16, 31])
ax[2, 0].set_xlabel("sample"); ax[2, 1].set_xlabel("coarse → fine")
fig.tight_layout(); save(fig, "haar/fig-sparse.svg")
