# Static figures for the FLIM day (fallbacks for print). Run from the repository root:
#   python3 -I tools/figs/flim/figs.py
# Writes SVGs into lifetime/, decayfit/, phasor/, flim-fret/. No external data; fixed seed.
import math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
C = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
W = 2 * math.pi * 80e-3  # rad/ns at 80 MHz

def save(fig, path):
    fig.tight_layout(); fig.savefig(path, format="svg"); plt.close(fig); print("wrote", path)

def semi(ax, ticks=(0.25, 0.5, 1, 2, 4, 8)):
    a = np.linspace(0, math.pi, 200)
    ax.plot(0.5 + 0.5 * np.cos(a), 0.5 * np.sin(a), color="#444", lw=1.8)
    for t in ticks:
        g, s = ph(t)
        ax.plot(g, s, "o", color="#444", ms=4)
        d = np.array([g - 0.5, s]); d /= np.linalg.norm(d)
        ax.text(g + 0.07 * d[0], s + 0.07 * d[1], f"{t:g}", ha="center", va="center", fontsize=13)
    ax.set_aspect("equal"); ax.set_xlim(-0.08, 1.08); ax.set_ylim(-0.03, 0.62)
    ax.set_xlabel("g"); ax.set_ylabel("s")

def ph(t):
    q = 1 + (W * t) ** 2
    return 1 / q, W * t / q

# 1. lifetime/fig-decay.svg: one decay, linear and log axes
t = np.linspace(0, 12, 400); tau = 2.5
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, log in zip(axs, (False, True)):
    ax.plot(t, np.exp(-t / tau), color=C[0], lw=2.5)
    for k, lab in ((1, "37 %"), (3, "5 %")):
        ax.axvline(k * tau, color="#888", ls="--", lw=1)
        ax.text(k * tau + 0.15, math.exp(-k) * (1.3 if log else 1) + (0 if log else 0.05), f"{k}τ: {lab} still to come", fontsize=12)
    ax.set_xlabel("time after the pulse (ns)")
    if log: ax.set_yscale("log"); ax.set_ylim(5e-3, 1.5); ax.set_title("log axis: a straight line, slope −1/τ")
    else: ax.set_ylim(0, 1.05); ax.set_title("linear axis: e^(−t/τ), τ = 2.5 ns")
    ax.set_ylabel("emission (relative)")
save(fig, "lifetime/fig-decay.svg")

# 2. lifetime/fig-tcspc.svg: pulses with photons, and the histogram with IRF
rng = np.random.default_rng(3)
fig, axs = plt.subplots(2, 1, figsize=(10, 4.6), gridspec_kw={"height_ratios": [1, 2.4]})
ax = axs[0]
T = 12.5
for k in range(9): ax.axvline(k * T, color=C[3], lw=3)
for k in (1, 4, 5, 7):
    ax.plot(k * T + 1 + rng.exponential(2.5), 0.5, "o", color=C[0], ms=9)
ax.set_xlim(0, 8 * T); ax.set_yticks([]); ax.set_xlabel("time (ns): laser pulses every 12.5 ns, at most one photon timed per pulse")
for s in ("left",): ax.spines[s].set_visible(False)
ax = axs[1]
edges = np.linspace(0, T, 129); cen = (edges[:-1] + edges[1:]) / 2
irf = np.exp(-0.5 * ((cen - 1) / 0.15) ** 2); irf /= irf.sum()
dec = np.where(cen >= 0, np.exp(-cen / 2.5), 0)
model = np.real(np.fft.ifft(np.fft.fft(irf) * np.fft.fft(dec))); model = model / model.sum() * 20000 + 2
y = rng.poisson(model)
ax.bar(cen, y, width=edges[1] - edges[0], color=C[0], alpha=0.45, label="photon counts")
ax.plot(cen, model, color=C[1], lw=2, label="expected: decay ∗ IRF + background")
ax.plot(cen, irf / irf.max() * y.max() * 0.9, "--", color=C[3], lw=2, label="IRF (scaled)")
ax.set_yscale("log"); ax.set_ylim(1, y.max() * 2); ax.set_xlim(0, T)
ax.set_xlabel("arrival time after the pulse (ns)"); ax.set_ylabel("photons per bin"); ax.legend(fontsize=12, loc="upper right")
save(fig, "lifetime/fig-tcspc.svg")

# 3. decayfit/fig-twoexp.svg: two different bi-exponentials whose curves agree within the noise
t = np.linspace(0, 12, 300)
a = 0.4 * np.exp(-t / 0.8) + 0.6 * np.exp(-t / 3.0)
# weighted least-squares match (weights 1/a, like Poisson noise) of a second pair, τ = 1.0 and 3.15 ns
B = np.stack([np.exp(-t / 1.0), np.exp(-t / 3.15)], 1)
wts = 1 / np.sqrt(a)
coef, *_ = np.linalg.lstsq(B * wts[:, None], a * wts, rcond=None)
b = B @ coef
diff = np.abs(b - a).max() / a.max() * 100
# noise for 1 000 photons in 64 bins over 12.5 ns: relative SD of the peak bin
peak_counts = 1000 * a.max() * (12.5 / 64) / (0.4 * 0.8 + 0.6 * 3.0)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
axs[0].semilogy(t, a, color=C[0], lw=3, label="τ = 0.8 and 3.0 ns")
axs[0].semilogy(t, b, "--", color=C[1], lw=2.5, label="τ = 1.0 and 3.15 ns")
axs[0].set_xlabel("time (ns)"); axs[0].set_ylabel("decay"); axs[0].legend(fontsize=12)
axs[0].set_title("two different pairs of lifetimes")
axs[1].plot(t, (b - a) / a.max() * 100, color=C[1], lw=2)
axs[1].axhline(0, color="#888", lw=1)
axs[1].set_xlabel("time (ns)"); axs[1].set_ylabel("difference (% of the peak)")
axs[1].set_title(f"at most {diff:.1f} % of the peak apart")
save(fig, "decayfit/fig-twoexp.svg")
print("max diff % of peak", diff, "amplitudes", coef, "peak bin counts (1000 photons)", peak_counts, "rel SD %", 100 / math.sqrt(peak_counts))

# 4. phasor/fig-semicircle.svg: single exponentials, a mixture on the chord
fig, ax = plt.subplots(figsize=(6.4, 3.9))
semi(ax)
p1, p2 = np.array(ph(0.4)), np.array(ph(2.5))
ax.plot(*zip(p1, p2), color=C[0], lw=2)
ax.plot(*p1, "o", color=C[2], ms=10); ax.plot(*p2, "o", color=C[4], ms=10)
m = 0.5 * p1 + 0.5 * p2
ax.plot(*m, "o", color=C[1], ms=10)
ax.text(m[0] + 0.03, m[1] - 0.06, "50/50 photons", color=C[1], fontsize=13)
ax.text(p1[0] - 0.02, p1[1] + 0.06, "0.4 ns", color=C[2], fontsize=13, ha="right")
ax.text(p2[0] - 0.02, p2[1] + 0.05, "2.5 ns", color=C[4], fontsize=13, ha="right")
ax.set_title("80 MHz: single lifetimes on the semicircle, mixtures on chords", fontsize=13)
save(fig, "phasor/fig-semicircle.svg")

# 5. flim-fret/fig-fret-phasor.svg: donor point, FRET states, line to a 30 % mixture
fig, ax = plt.subplots(figsize=(6.4, 3.9))
semi(ax)
tD = 2.5
traj = np.array([ph(tD * (1 - e)) for e in np.linspace(0, 0.99, 100)])
pD, pDA = np.array(ph(tD)), np.array(ph(1.0))
ax.plot(*zip(pD, pDA), "--", color=C[1], lw=2)
f = 0.3 * 1.0 / (0.3 * 1.0 + 0.7 * tD)
mix = (1 - f) * pD + f * pDA
ax.plot(*pD, "o", color=C[2], ms=10); ax.plot(*pDA, "o", color=C[1], ms=10); ax.plot(*mix, "o", color=C[0], ms=9)
ax.text(pD[0] - 0.03, pD[1] + 0.05, "donor alone 2.5 ns", ha="right", color=C[2], fontsize=12)
ax.text(pDA[0] + 0.03, pDA[1] + 0.03, "FRET state 1.0 ns (E = 0.6)", color=C[1], fontsize=12)
ax.annotate("30 % of donors interact:\nphoton fraction 0.15", xy=mix, xytext=(0.32, 0.18), fontsize=12, arrowprops=dict(arrowstyle="->", color=C[0]), color=C[0])
save(fig, "flim-fret/fig-fret-phasor.svg")
