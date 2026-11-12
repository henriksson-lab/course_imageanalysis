# shrinkage/fig-priors.svg: one noisy step-and-ramp profile, denoised by total variation and by
# wavelet soft thresholding (db4, 5 levels, 16 shifts); each with the weight that minimises the error.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open("figs.py").read().split("# ---- haar/fig-spike-step.svg ----")[0])
n = 256; i = np.arange(n)
truth = np.where(i < 64, 0.0, 1.0); truth = np.where(i >= 128, np.clip(1 - (i - 128) / 96, 0, 1), truth)
rng = np.random.default_rng(3); sigma = 0.1; y = truth + sigma * rng.standard_normal(n)
def tv1d(y, lam, iters=4000):
    # dual projected gradient (Chambolle): min 1/2||y-x||^2 + lam * sum |x[k+1]-x[k]|
    p = np.zeros(len(y) - 1); tau = 0.25
    for _ in range(iters):
        x = y + np.r_[p[0], np.diff(p), -p[-1]] * 1.0  # x = y - D^T p with D x = diff(x): D^T p = -(diff with padding)
        g = np.diff(x)
        p = np.clip(p + tau * g, -lam, lam)
    return y + np.r_[p[0], np.diff(p), -p[-1]]
# check objective decreases sensible: compare with lam small
def shrink(y, t, h="db4", L=5, spins=16):
    out = np.zeros_like(y)
    for s in range(spins):
        c = dwt1(np.roll(y, s), H[h], L); a0 = len(y) >> L
        c[a0:] = np.sign(c[a0:]) * np.maximum(np.abs(c[a0:]) - t, 0)
        out += np.roll(idwt1(c, H[h], L), -s) / spins
    return out
rmse = lambda x: np.sqrt(np.mean((x - truth) ** 2))
best_tv = min(((rmse(tv1d(y, l)), l) for l in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]))
best_w = min(((rmse(shrink(y, t)), t) for t in [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3]))
print("noisy rmse", rmse(y), "TV best", best_tv, "wavelet best", best_w, "sigma*sqrt(2ln n)", sigma * np.sqrt(2 * np.log(n)))
xtv = tv1d(y, best_tv[1]); xw = shrink(y, best_w[1])
# TV objective check: should be <= objective at y and at truth
obj = lambda x, l: 0.5 * np.sum((y - x) ** 2) + l * np.sum(np.abs(np.diff(x)))
print("objective TV sol", obj(xtv, best_tv[1]), "at truth", obj(truth, best_tv[1]), "at y", obj(y, best_tv[1]))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
for a, x, ttl, c in [(ax[0], xtv, f"total variation (λ = {best_tv[1]}): RMSE {best_tv[0]:.3f}", RED), (ax[1], xw, f"wavelet soft threshold (t = {best_w[1]}): RMSE {best_w[0]:.3f}", BLUE)]:
    a.plot(i, y, ".", color=GREY, ms=3); a.plot(i, truth, "k", lw=1.3); a.plot(i, x, color=c, lw=2.5); a.set_title(ttl, fontsize=14); a.set_xlabel("position")
ax[0].set_ylabel("value")
fig.tight_layout(); fig.savefig(R + "shrinkage/fig-priors.svg", bbox_inches="tight", pad_inches=0.05); print("wrote")
