# Static figures for decayfit/lec.qmd (in addition to tools/figs/flim/figs.py, which writes
# fig-twoexp.svg). Run from the repository root:
#   python3 -I tools/figs/decayfit/figs.py <mc.json>
# where mc.json comes from tools/figs/decayfit/mc.ts (the photonbudget demo's Monte Carlo, same
# seeds and estimators); without it, fig-costs.svg and fig-photons.svg are skipped. The other
# figures simulate their own histograms (fixed seeds) with the slides' numbers.
import json, math, sys
import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import erf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 18, "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans", "svg.hashsalt": "decayfit"})
C = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]   # as in tools/figs/flim/figs.py
GREY = "#777"
OUT = "decayfit/"

def save(fig, name):
    fig.savefig(OUT + name, format="svg", facecolor="white"); plt.close(fig); print("wrote", OUT + name)

T = 12.5
def axis(nb):
    e = np.linspace(0, T, nb + 1); return e, 0.5 * (e[:-1] + e[1:])

def unit_decay(tau, nb, sigma=0.15, t0=1.0):
    """Expected fraction of photons per bin: periodic exponential convolved with a Gaussian IRF."""
    e, c = axis(nb)
    fine = np.linspace(0, T, nb * 16, endpoint=False) + T / (nb * 32)
    dec = np.exp(-fine / tau)
    irf = np.exp(-0.5 * (((fine - t0 + T / 2) % T - T / 2) / sigma) ** 2)
    m = np.real(np.fft.ifft(np.fft.fft(irf) * np.fft.fft(dec)))
    m = np.clip(m, 0, None).reshape(nb, 16).sum(1)
    return m / m.sum()

def poisson_dev(y, mu):
    mu = np.maximum(mu, 1e-12)
    return 2 * np.sum(np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0) - (y - mu))

# 1. fig-reconv.svg: deconvolving the data vs fitting through the forward model
nb = 128; e, c = axis(nb)
rng = np.random.default_rng(7)
tau0, N = 2.5, 2000
y = rng.poisson(N * unit_decay(tau0, nb)).astype(float)
irf = unit_decay(1e-3, nb)                                      # the IRF alone (zero lifetime)
H = np.fft.fft(irf)
dec = np.real(np.fft.ifft(np.fft.fft(y) / H))
fit = minimize(lambda p: poisson_dev(y, p[0] * unit_decay(p[1], nb)), [y.sum(), 2.0], method="Nelder-Mead")
A_f, tau_f = fit.x
fig, axs = plt.subplots(2, 1, figsize=(5.8, 5.6))
ax = axs[0]
ax.plot(c, dec, color=GREY, lw=1.2)
ax.set_title("deconvolve the data: noise explodes", fontsize=17)
ax.set_xlim(0, T); ax.set_xlabel("time (ns)"); ax.set_yticks([0]); ax.axhline(0, color="#bbb", lw=1)
ax = axs[1]
ax.plot(c, y, "o", ms=3.5, color=C[0], label="data (2 000 photons)")
ax.plot(c, A_f * unit_decay(tau_f, nb), color=C[1], lw=2.5, label=f"model ∗ IRF, τ = {tau_f:.2f} ns")
ax.plot(c, irf / irf.max() * y.max(), "--", color=C[3], lw=1.8, label="IRF (scaled)")
ax.set_title("reconvolution: blur the model instead", fontsize=17)
ax.set_xlim(0, T); ax.set_xlabel("time (ns)"); ax.set_ylabel("photons")
ax.legend(fontsize=14, frameon=False, loc="upper right")
fig.tight_layout(h_pad=1.0)
save(fig, "fig-reconv.svg")
print("reconvolution fit tau", tau_f)

# 2. fig-landscape.svg: cost landscapes, one lifetime (bowl) and two lifetimes (curved valley)
nb = 64
y1 = np.random.default_rng(3).poisson(1000 * unit_decay(2.5, nb)).astype(float)
taus = np.linspace(1.6, 3.6, 81); amps = np.linspace(800, 1200, 81)
B = {t: unit_decay(t, nb) for t in taus}
Z1 = np.array([[poisson_dev(y1, a * B[t]) for t in taus] for a in amps])
# a few Gauss-Newton / Levenberg-Marquardt steps (Poisson weights) from the mean arrival time
def lm_path(y, p, steps=6, lam=1e-2):
    path = [p.copy()]
    for _ in range(steps):
        mu = p[0] * unit_decay(p[1], nb)
        J = np.stack([unit_decay(p[1], nb), p[0] * (unit_decay(p[1] + 1e-4, nb) - unit_decay(p[1] - 1e-4, nb)) / 2e-4], 1)
        W = 1 / np.maximum(mu, 1e-9)
        A = J.T @ (J * W[:, None]); g = J.T @ (W * (y - mu))
        while True:
            q = p + np.linalg.solve(A + lam * np.diag(np.diag(A)), g)
            if q[1] > 0 and poisson_dev(y, q[0] * unit_decay(q[1], nb)) < poisson_dev(y, mu): p = q; lam /= 3; break
            lam *= 4
            if lam > 1e8: break
        path.append(p.copy())
    return np.array(path)
path = lm_path(y1, np.array([900.0, 1.8]))
y2 = np.random.default_rng(4).poisson(3000 * (0.4 * unit_decay(0.8, nb) + 0.6 * unit_decay(3.0, nb))).astype(float)
t1s = np.linspace(0.3, 1.8, 76); t2s = np.linspace(2.0, 5.0, 76)
U = {t: unit_decay(t, nb) for t in np.concatenate([t1s, t2s])}
def profiled(t1, t2):
    Bm = np.stack([U[t1], U[t2]], 1); w = 1 / np.sqrt(np.maximum(y2, 1))
    a, _ = nnls(Bm * w[:, None], y2 * w)
    for _ in range(3):                                         # reweight with the model (Pearson / Poisson)
        mu = np.maximum(Bm @ a, 1e-6); w = 1 / np.sqrt(mu); a, _ = nnls(Bm * w[:, None], y2 * w)
    return poisson_dev(y2, Bm @ a)
Z2 = np.array([[profiled(a, b) for a in t1s] for b in t2s])
fig, axs = plt.subplots(2, 1, figsize=(5.4, 6.6))
for ax, X, Yv, Z, title in ((axs[0], taus, amps, Z1, "one lifetime: a round bowl"), (axs[1], t1s, t2s, Z2, "two lifetimes: a long valley")):
    Zr = Z - Z.min()
    ax.contourf(X, Yv, np.log10(Zr + 1), levels=14, cmap="viridis_r")
    ax.contour(X, Yv, Zr, levels=[1, 4, 16, 64], colors="white", linewidths=1.2)
    ax.set_title(title, fontsize=17)
axs[0].plot(path[:, 1], path[:, 0], "o-", color=C[1], ms=6, lw=2, label="fit steps")
axs[0].plot(path[0, 1], path[0, 0], "s", color=C[1], ms=9)
axs[0].legend(fontsize=14, loc="upper right", framealpha=0.9)
axs[0].set_xlabel("τ (ns)"); axs[0].set_ylabel("amplitude (photons)")
axs[1].plot(0.8, 3.0, "*", color=C[1], ms=16, label="true (0.8, 3.0 ns)")
axs[1].legend(fontsize=14, loc="upper right", framealpha=0.9)
axs[1].set_xlabel("τ₁ (ns)"); axs[1].set_ylabel("τ₂ (ns)")
fig.tight_layout(h_pad=1.0)
save(fig, "fig-landscape.svg")

# 3. fig-reporting.svg: amplitude vs photon fractions; the two mean lifetimes
al, ta = np.array([0.5, 0.5]), np.array([0.5, 3.0])
f = al * ta / (al * ta).sum()
tau_a, tau_f = (al * ta).sum(), (f * ta).sum()
t = np.linspace(0, 10, 400)
fig, axs = plt.subplots(2, 1, figsize=(5.6, 5.4), gridspec_kw={"height_ratios": [2.2, 1]})
ax = axs[0]
for k, col in ((0, C[3]), (1, C[0])):
    yk = al[k] * np.exp(-t / ta[k])
    ax.plot(t, yk, color=col, lw=2.5)
    ax.fill_between(t, yk, color=col, alpha=0.25, lw=0)
ax.text(0.9, 0.42, f"0.5 ns: half the molecules,\n{100 * f[0]:.0f} % of the photons", color=C[3], fontsize=15)
ax.text(3.2, 0.2, f"3 ns: half the molecules,\n{100 * f[1]:.0f} % of the photons", color=C[0], fontsize=15)
ax.set_xlim(0, 10); ax.set_ylim(0, 0.62); ax.set_yticks([]); ax.set_xlabel("time (ns)"); ax.set_ylabel("emission")
ax = axs[1]
ax.axhline(0, color="#444", lw=1.5)
for v, lab, col, dy, ha, mk in ((0.5, "τ₁", C[3], 1, "center", "o"), (3.0, "τ₂", C[0], -1, "center", "o"),
                               (tau_a, r"$\bar\tau_a$" + f" = {tau_a:.2f}", "black", -1, "center", "D"),
                               (tau_f + 0.1, r"$\bar\tau_f$" + f" = {tau_f:.2f}", C[1], 1, "right", "D")):
    ax.plot(min(v, tau_f) if ha == "right" else v, 0, mk, color=col, ms=10)
    ax.text(v, 0.55 * dy, lab, ha=ha, va="center", fontsize=16, color=col)
ax.set_ylim(-1, 1); ax.set_xlim(0, 3.6); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("lifetime (ns)")
fig.tight_layout(h_pad=0.8)
save(fig, "fig-reporting.svg")
print("photon fractions", f, "tau_a", tau_a, "tau_f", tau_f)

# 4. fig-global.svg: the decay shapes are the columns; one pixel's amplitudes by NNLS
nb = 64; e, c = axis(nb)
b1, b2 = unit_decay(0.6, nb, 0.1, 0.5), unit_decay(3.5, nb, 0.1, 0.5)   # flimcell: IRF σ 0.1 ns at 0.5 ns
yp = np.random.default_rng(9).poisson(400 * (0.45 * b1 + 0.55 * b2)).astype(float)
def amounts(Bm, y, it=8):
    """Non-negative amplitudes, reweighted with the model (weights 1/μ, as for Poisson counts)."""
    w = 1 / np.sqrt(np.maximum(y, 1)); a, _ = nnls(Bm * w[:, None], y * w)
    for _ in range(it):
        w = 1 / np.sqrt(np.maximum(Bm @ a, 1e-6)); a, _ = nnls(Bm * w[:, None], y * w)
    return a
Bm = np.stack([b1, b2], 1); a = amounts(Bm, yp)
fig, axs = plt.subplots(2, 1, figsize=(5.6, 5.4))
ax = axs[0]
ax.semilogy(c, b1, color=C[3], lw=2.5, label="0.6 ns")
ax.semilogy(c, b2, color=C[0], lw=2.5, label="3.5 ns")
ax.set_title("decay shapes, fitted once to all pixels", fontsize=16)
ax.set_ylim(1e-4, 0.3); ax.set_yticks([]); ax.minorticks_off(); ax.legend(fontsize=15, frameon=False); ax.set_xlim(0, T)
ax.set_ylabel("shape (log)", fontsize=16)
ax = axs[1]
ax.bar(c, yp, width=T / nb, color="#bbb", lw=0, label="one pixel, 400 photons")
ax.plot(c, a[0] * b1, color=C[3], lw=2.2)
ax.plot(c, a[1] * b2, color=C[0], lw=2.2)
ax.plot(c, Bm @ a, color=C[1], lw=2.5, label=f"sum: {a[0]:.0f} + {a[1]:.0f} photons")
ax.set_title("per pixel: only the amounts (linear)", fontsize=16)
ax.set_xlim(0, T); ax.set_xlabel("time (ns)"); ax.set_ylabel("photons"); ax.legend(fontsize=14, frameon=False)
fig.tight_layout(h_pad=1.0)
save(fig, "fig-global.svg")
print("pixel amplitudes", a)

# 5. fig-residuals.svg: one-exponential fit to two-lifetime data (wavy residuals) vs two
nb = 64; e, c = axis(nb)
u1, u2 = unit_decay(0.8, nb), unit_decay(3.0, nb)
yr = np.random.default_rng(18).poisson(10000 * (0.4 * u1 + 0.6 * u2)).astype(float)
m1 = minimize(lambda p: poisson_dev(yr, p[0] * unit_decay(p[1], nb)), [yr.sum(), 2.0], method="Nelder-Mead")
mu1 = m1.x[0] * unit_decay(m1.x[1], nb)
def mu_two(tt):
    Bm = np.stack([unit_decay(tt[0], nb), unit_decay(tt[1], nb)], 1); return Bm @ amounts(Bm, yr)
m2 = minimize(lambda tt: poisson_dev(yr, mu_two(tt)), [0.7, 3.2], method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-8})
p2 = m2.x; mu2 = mu_two(p2)
def redchi(mu, k): return np.sum((yr - mu) ** 2 / mu) / (nb - k)
fig, axs = plt.subplots(3, 1, figsize=(5.8, 5.8), sharex=True, gridspec_kw={"height_ratios": [1.6, 1, 1]})
axs[0].semilogy(c, yr, "o", ms=3.5, color=C[0])
axs[0].semilogy(c, mu1, color=C[1], lw=2.2, label=f"one τ = {m1.x[1]:.2f} ns")
axs[0].semilogy(c, mu2, "--", color=C[2], lw=2.2, label="two τ")
axs[0].set_ylim(5, yr.max() * 2); axs[0].set_yticks([]); axs[0].minorticks_off(); axs[0].legend(fontsize=14, frameon=False)
axs[0].set_ylabel("photons (log)", fontsize=16)
for ax, mu, col, lab in ((axs[1], mu1, C[1], f"one τ: wavy, reduced χ² = {redchi(mu1, 2):.1f}"), (axs[2], mu2, C[2], f"two τ: noise, reduced χ² = {redchi(mu2, 4):.2f}")):
    r = (yr - mu) / np.sqrt(mu)
    ax.bar(c, r, width=T / nb, color=col, lw=0)
    ax.axhline(0, color="#444", lw=1); ax.set_ylim(-9, 9); ax.set_yticks([-5, 0, 5])
    ax.set_title(lab, fontsize=15, pad=2)
axs[1].set_ylabel("residual / √μ", fontsize=15)
axs[2].set_xlabel("time (ns)"); axs[2].set_xlim(0, T)
fig.tight_layout(h_pad=0.4)
save(fig, "fig-residuals.svg")
print("one-exp tau", m1.x[1], "redchi", redchi(mu1, 2), "two-exp", p2, redchi(mu2, 4))

# 6/7. from the photonbudget Monte Carlo
if len(sys.argv) > 1:
    R = json.load(open(sys.argv[1]))
    est = R["few100"]
    names = [("least squares", 2, C[2]), ("χ² with 1/y", 3, C[3]), ("Poisson", 4, C[0])]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    jit = np.random.default_rng(1)
    for row, (lab, j, col) in enumerate(names):
        v = np.array([x for x in est[j] if x is not None and np.isfinite(x)])
        ax.plot(v, row + jit.uniform(-0.28, 0.28, len(v)), "o", ms=3, color=col, alpha=0.45)
        m, sd = v.mean(), v.std(ddof=1)
        ax.errorbar(m, row, xerr=sd, fmt="D", color="black", ms=7, capsize=6, lw=2)
        ax.text(4.15, row, f"{m:.2f} ± {sd:.2f}", va="center", fontsize=15)
        print(lab, m, sd, 100 * (m / 2.5 - 1), "%")
    ax.axvline(2.5, color=C[1], lw=2, ls="--")
    ax.set_yticks(range(3)); ax.set_yticklabels([n[0] for n in names], fontsize=16)
    ax.set_xlim(1.2, 5.3); ax.set_xticks([1.5, 2.5, 3.5]); ax.set_xlabel("estimated τ (ns), true 2.5 ns")
    ax.set_title("400 histograms of 100 photons", fontsize=16)
    fig.tight_layout()
    save(fig, "fig-costs.svg")

    Ns = np.array(R["N_LIST"])
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.loglog(Ns, 100 / np.sqrt(Ns), "--", color=GREY, lw=2, label="1/√N (ideal)")
    ax.loglog(Ns, 100 * np.array(R["vsMono"][4]), "o-", color=C[0], lw=2.5, ms=7, label="one τ, Poisson fit")
    bi = np.array([np.nan if v is None else v for v in R["vsBi"][0]]), np.array([np.nan if v is None else v for v in R["vsBi"][1]])
    ax.loglog(Ns, 100 * bi[0], "s-", color=C[3], lw=2.5, ms=7, label="two τ: 0.8 ns")
    ax.loglog(Ns, 100 * bi[1], "^-", color=C[4], lw=2.5, ms=7, label="two τ: 3 ns")
    ax.axhline(10, color="#bbb", lw=1)
    ax.text(1.2e5, 10, "10 %", fontsize=15, color="#555", va="center")
    ax.set_xlabel("photons N"); ax.set_ylabel("relative RMS error (%)")
    ax.set_ylim(0.25, 60); ax.set_xticks([100, 1000, 10000, 100000])
    ax.legend(fontsize=15, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, columnspacing=1.0, handlelength=1.6)
    fig.tight_layout()
    save(fig, "fig-photons.svg")
