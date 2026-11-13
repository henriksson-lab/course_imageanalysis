"""RMS error of the Monte Carlo estimate of the integral of x^2 on [0, 1] against the number of
samples N (pathtrace/fig-mcnoise.svg): 4000 repeated estimates per N, uniform samples, against
sigma/sqrt(N) with sigma = sqrt(1/5 - 1/9) = 0.298. Run from the repo root."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 17, "svg.fonttype": "none", "font.family": "sans-serif"})
rng = np.random.default_rng(3)
sigma = np.sqrt(1 / 5 - 1 / 9)
Ns = 2 ** np.arange(0, 13)
rms = []
for n in Ns:
    est = (rng.random((4000, n)) ** 2).mean(axis=1)
    rms.append(np.sqrt(np.mean((est - 1 / 3) ** 2)))
rms = np.array(rms)
for n, r in zip(Ns, rms):
    print(n, round(r, 4), round(sigma / np.sqrt(n), 4))
fig, ax = plt.subplots(figsize=(6.0, 4.4))
ax.loglog(Ns, sigma / np.sqrt(Ns), color="#0072B2", lw=2.5, label="σ/√N, σ = 0.298")
ax.loglog(Ns, rms, "o", color="#D55E00", ms=7, label="measured (4000 repeats)")
i16, i64 = list(Ns).index(16), list(Ns).index(64)
ax.annotate("", xy=(64, rms[i64]), xytext=(16, rms[i16]), arrowprops=dict(arrowstyle="->", lw=2))
ax.text(30, rms[i16] * 1.15, "4× samples:\nhalf the error", fontsize=16)
ax.set_xlabel("samples N")
ax.set_ylabel("RMS error of the estimate")
ax.set_xticks([1, 4, 16, 64, 256, 1024, 4096])
ax.set_xticklabels(["1", "4", "16", "64", "256", "1024", "4096"])
ax.legend(loc="lower left", frameon=False, fontsize=15)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("pathtrace/fig-mcnoise.svg", facecolor="white")
print("wrote pathtrace/fig-mcnoise.svg")
