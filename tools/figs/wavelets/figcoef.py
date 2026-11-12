# shrinkage/fig-coefficients.svg: a piecewise-constant signal with noise, and its Haar detail coefficients
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open("figs.py").read().split("# ---- haar/fig-spike-step.svg ----")[0])
n = 512; t = np.arange(n) / n
T = [0.1, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81]; Hh = [4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2]
f = sum(h * (t >= tj) for h, tj in zip(Hh, T)); f = 7 * (f - f.mean()) / f.std()
rng = np.random.default_rng(5); y = f + rng.standard_normal(n)
L = 5; cf = dwt1(f, H["haar"], L); cy = dwt1(y, H["haar"], L)
a0 = n >> L
u = np.sqrt(2 * np.log(n))
fig, ax = plt.subplots(2, 1, figsize=(12, 6.4))
ax[0].plot(t * n, y, ".", color=GREY, ms=3, label="noisy"); ax[0].plot(t * n, f, "k", lw=1.5, label="clean"); ax[0].set_title("signal: piecewise constant + noise (σ = 1)"); ax[0].legend(frameon=False, loc="upper right", ncol=2)
idx = np.arange(a0, n)
ax[1].plot(idx, cy[a0:], ".", color=PURPLE, ms=4, label="noisy signal")
big = np.abs(cf[a0:]) > 1e-9
ax[1].plot(idx[big], cf[a0:][big], "o", mfc="none", mec="k", ms=7, label="clean signal (nonzero only)")
ax[1].axhline(u, color=RED, ls="--"); ax[1].axhline(-u, color=RED, ls="--")
ax[1].text(n - 4, u + 0.6, "±σ√(2 ln N)", color=RED, ha="right", fontsize=14)
for j in range(1, L + 1):
    ax[1].axvline(n >> j, color="#bbbbbb", lw=1)
    if j <= 3: ax[1].text(1.5 * (n >> j), 17, f"level {j}", ha="center", fontsize=13, color=GREY)
ax[1].set_ylim(-16, 20); ax[1].set_xlim(a0, n); ax[1].set_title("its Haar detail coefficients, coarse (left) to fine (right)"); ax[1].set_xlabel("coefficient index")
ax[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=13)
fig.tight_layout(); fig.savefig(R + "shrinkage/fig-coefficients.svg", bbox_inches="tight", pad_inches=0.05)
print("details: clean nonzero", int(big.sum()), "of", n - a0, "; noisy above u:", int(np.sum(np.abs(cy[a0:]) > u)), "; clean-zero positions above u:", int(np.sum((np.abs(cy[a0:]) > u) & ~big)), "u", u)
print("sd of noisy details where clean is zero", cy[a0:][~big].std())
