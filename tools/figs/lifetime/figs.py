# Static figures for lifetime/lec.qmd (in addition to tools/figs/flim/figs.py, which writes
# fig-decay.svg and fig-tcspc.svg). Run from the repository root:
#   python3 -I tools/figs/lifetime/figs.py <scene.json>
# where scene.json comes from tools/figs/lifetime/scene.ts (the flimcell demo's quencher scene);
# without it, fig-display.png is skipped. All other curves are computed from the formulas of the
# slides; synthetic data only, fixed seeds.
import json, math, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.rcParams.update({"font.size": 18, "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans", "svg.hashsalt": "lifetime"})
C = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]   # as in tools/figs/flim/figs.py
GREY = "#777"
OUT = "lifetime/"
VIR = plt.get_cmap("viridis")

def save(fig, name, **kw):
    fig.savefig(OUT + name, format=name[-3:], facecolor="white", **kw); plt.close(fig); print("wrote", OUT + name)

def merge(tau, I, imax, lo=0.0, hi=4.0, mask=None):
    """Hue = lifetime (viridis, 0-4 ns as in the flimcell demo), brightness = intensity."""
    rgb = VIR(Normalize(lo, hi, clip=True)(np.nan_to_num(tau, nan=0)))[..., :3]
    b = np.clip(I / imax, 0, 1)[..., None]
    out = rgb * b
    if mask is not None: out[~mask] = 0
    out[~np.isfinite(tau)] = 0
    return out

# 1. fig-brighter.svg: two cells, one twice as bright. Two possible lifetime images.
n = 120
yy, xx = np.mgrid[0:n, 0:2 * n]
def cell(cx, cy, rx, ry):
    return np.clip(1.6 * (1 - ((xx - cx) / rx) ** 2 - ((yy - cy) / ry) ** 2), 0, 1)
A, B = cell(62, 60, 46, 38), cell(178, 62, 44, 40)
I = 0.5 * A + 1.0 * B                                   # the right cell twice as bright
tau_dye = np.where(A > 0, 2.5, np.where(B > 0, 2.5, np.nan))
tau_q = np.where(A > 0, 1.25, np.where(B > 0, 2.5, np.nan))
fig = plt.figure(figsize=(6.2, 6.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.38, wspace=0.08)
ax = fig.add_subplot(gs[0, :])
ax.imshow(I, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
ax.set_title("intensity: right cell twice as bright", fontsize=18)
ax.text(62, 60, "1×", color="white", ha="center", va="center", fontsize=22)
ax.text(178, 62, "2×", color="black", ha="center", va="center", fontsize=22)
for k, (tau, title) in enumerate(((tau_dye, "more dye:\nboth 2.5 ns"), (tau_q, "left one quenched:\n1.25 vs 2.5 ns"))):
    ax2 = fig.add_subplot(gs[1, k])
    ax2.imshow(merge(tau, I, 1.0), interpolation="bilinear")
    ax2.set_title(title, fontsize=17)
for a in fig.axes: a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
cax = fig.add_axes([0.18, 0.15, 0.64, 0.025])
cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(0, 4), VIR), cax=cax, orientation="horizontal")
cb.set_label("lifetime (ns); brightness = intensity", fontsize=16); cb.ax.tick_params(labelsize=16)
fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.27)
save(fig, "fig-brighter.svg")

# 2. fig-quench.svg: a reference dye (4 ns, Q = 0.8), twice the dye, and a quencher that halves tau
t = np.linspace(0, 12, 400)
cases = [("reference: τ = 4 ns", 1.0, 4.0, C[0], "-"),
         ("twice the dye: τ = 4 ns", 2.0, 4.0, C[2], "--"),
         ("quencher: τ = 2 ns", 1.0, 2.0, C[1], "-")]
fig, axs = plt.subplots(2, 1, figsize=(5.6, 5.8), sharex=True)
for lab, amp, tau, col, ls in cases:
    y = amp * np.exp(-t / tau) / 4.0                      # same k_r: initial emission ∝ molecules
    axs[0].plot(t, y, color=col, lw=3, ls=ls, label=lab)
    axs[1].semilogy(t, y, color=col, lw=3, ls=ls)
axs[0].fill_between(t, np.exp(-t / 2) / 4, color=C[1], alpha=0.18, lw=0)
axs[0].annotate("half the area\n= half the photons", xy=(1.2, 0.06), xytext=(5.6, 0.17), color=C[1], fontsize=16, arrowprops=dict(arrowstyle="->", color=C[1]))
axs[0].set_ylabel("emission"); axs[0].set_ylim(0, 0.55); axs[0].set_yticks([])
axs[0].legend(fontsize=15, frameon=False, loc="upper right", bbox_to_anchor=(1.04, 1.15))
axs[1].set_ylabel("emission (log)"); axs[1].set_ylim(5e-4, 1); axs[1].set_yticks([])
axs[1].minorticks_off()
axs[1].text(6.2, 0.18, "same slope", color=C[2], fontsize=16)
axs[1].text(4.0, 1.4e-3, "steeper: shorter τ", color=C[1], fontsize=16)
axs[1].set_xlabel("time after the pulse (ns)")
fig.tight_layout(h_pad=0.6)
save(fig, "fig-quench.svg")

# 3. fig-forward.svg: IRF * decay + background = measured histogram (12.5 ns, 64 bins)
T, nb = 12.5, 64
edges = np.linspace(0, T, nb + 1); cen = (edges[:-1] + edges[1:]) / 2
irf = np.exp(-0.5 * ((cen - 1.0) / 0.15) ** 2); irf /= irf.sum()
tau = 2.5
dec = np.exp(-cen / tau) / (1 - math.exp(-T / tau)); dec /= dec.sum()     # periodic decay
mod = np.real(np.fft.ifft(np.fft.fft(irf) * np.fft.fft(dec)))
N, bg = 3000, 2.0
mu = N * mod / mod.sum() + bg
y = np.random.default_rng(5).poisson(mu)
fig, axs = plt.subplots(1, 4, figsize=(12, 3.0), gridspec_kw={"wspace": 0.55})
axs[0].plot(cen, irf / irf.max(), color=C[3], lw=3); axs[0].set_title("IRF (σ = 0.15 ns)", fontsize=17)
axs[1].plot(cen, dec / dec.max(), color=GREY, lw=3); axs[1].set_title("decay, τ = 2.5 ns", fontsize=17)
axs[2].plot(cen, np.full_like(cen, 0.08), color=C[4], lw=3); axs[2].set_ylim(0, 1.05); axs[2].set_title("background", fontsize=17)
axs[3].bar(cen, y, width=T / nb, color=C[0], alpha=0.5, lw=0)
axs[3].plot(cen, mu, color=C[1], lw=2.5); axs[3].set_title("measured", fontsize=17)
for a in axs:
    a.set_xlim(0, T); a.set_xticks([0, 5, 10]); a.set_yticks([]); a.set_xlabel("time (ns)")
axs[0].set_ylim(0, 1.05); axs[1].set_ylim(0, 1.05)
for a, sym in zip(axs[:3], ("∗", "+", "=")):
    a.annotate(sym, xy=(1.27, 0.5), xycoords="axes fraction", fontsize=34, ha="center", va="center")
fig.subplots_adjust(left=0.02, right=0.99, top=0.84, bottom=0.25)
save(fig, "fig-forward.svg")

# 4. fig-pulsetrain.svg: wrap-around (tau = 8 ns at 12.5 ns) and pile-up (60 % of pulses)
fig, axs = plt.subplots(2, 1, figsize=(5.8, 6.0))
ax = axs[0]; Tp, tau = 12.5, 8.0
tt = np.linspace(0, 4 * Tp, 1000)
tot = np.zeros_like(tt)
for k in range(-6, 4):
    yk = np.where(tt >= k * Tp, np.exp(-(tt - k * Tp) / tau), 0)
    tot += yk
    if 0 <= k <= 2: ax.plot(tt, yk, color=C[0], lw=1.6, alpha=0.8)
ax.plot(tt, tot, color=C[1], lw=2.8)
for k in range(4): ax.axvline(k * Tp, color=C[3], lw=2.5, ymax=0.62)
ax.text(0.5, 2.0, "red: the sum the detector sees", color=C[1], fontsize=15)
ax.annotate(f"{100 * math.exp(-Tp / tau):.0f} % still to come", xy=(Tp - 0.2, math.exp(-Tp / tau)), xytext=(15, 1.45),
            fontsize=15, arrowprops=dict(arrowstyle="->", color=C[0]), color=C[0])
ax.set_xlim(0, 3.2 * Tp); ax.set_ylim(0, 2.35); ax.set_yticks([])
ax.set_title("wrap-around: τ = 8 ns, pulses every 12.5 ns", fontsize=16)
ax.set_xlabel("time (ns)")
ax = axs[1]; tau = 2.5
tb = np.linspace(0, 10, 300)
mu_p = -math.log(1 - 0.6)                                   # 60 % of pulses give >= 1 photon
p = np.exp(-tb / tau) / tau; F = 1 - np.exp(-tb / tau)
first = mu_p * p * np.exp(-mu_p * F) / (1 - math.exp(-mu_p))  # density of the first photon
ax.semilogy(tb, p, color=GREY, lw=3, label="true decay, 2.5 ns")
ax.semilogy(tb, first, color=C[1], lw=3, ls="--", label="first photon only (60 % of pulses)")
ax.set_ylim(2e-4, 0.8); ax.set_yticks([]); ax.minorticks_off()
ax.set_xlabel("arrival time (ns)"); ax.set_ylabel("histogram (log)")
ax.set_title("pile-up: early photons over-counted", fontsize=16)
ax.legend(fontsize=14, frameon=False, loc="lower left")
fig.tight_layout(h_pad=1.2)
save(fig, "fig-pulsetrain.svg")
print("pile-up: P(>=2 | >=1) =", (1 - math.exp(-mu_p) - mu_p * math.exp(-mu_p)) / 0.6)

# 5. fig-meanarrival.svg and fig-gates.svg: the two worked examples of the slide
tt = np.linspace(0, 12, 1200); dt = tt[1] - tt[0]
irf = np.exp(-0.5 * ((tt - 0.6) / 0.15) ** 2); irf /= irf.sum()
d = np.exp(-tt / 2.5); d /= d.sum()
meas = np.convolve(irf, d)[:len(tt)]; meas /= meas.sum()
print("means: IRF", (tt * irf).sum(), "measured (window 12 ns)", (tt * meas).sum(), "exact 0.6 + 2.5 = 3.1")
fig, ax = plt.subplots(figsize=(4.8, 2.25))
ax.plot(tt, irf / irf.max(), color=C[3], lw=2.5, label="IRF")
ax.plot(tt, meas / meas.max(), color=C[0], lw=2.5, label="measured")
for m, col, lab, x, yv in ((0.6, C[3], "IRF mean 0.6 ns", 0.75, 1.62), (3.1, C[0], "measured mean 3.1 ns", 3.3, 0.6)):
    ax.plot([m, m], [0, 1.75 if m < 1 else 1.3], color=col, ls="--", lw=1.5)
    ax.text(x, yv, lab, color=col, fontsize=16)
ax.annotate("", xy=(3.1, 1.25), xytext=(0.6, 1.25), arrowprops=dict(arrowstyle="<->", lw=1.8))
ax.text(3.3, 1.25, "τ ≈ 2.5 ns", ha="left", va="center", fontsize=16)
ax.set_xlim(0, 10); ax.set_ylim(0, 1.95); ax.set_yticks([]); ax.set_xlabel("time (ns)")
fig.tight_layout()
save(fig, "fig-meanarrival.svg")

Np, tau, w = 1000, 2.0, 0.25
e = np.arange(0, 10 + 1e-9, w); c = 0.5 * (e[:-1] + e[1:])
cnt = Np * (np.exp(-e[:-1] / tau) - np.exp(-e[1:] / tau))
D1 = Np * (1 - math.exp(-2 / tau)); D2 = Np * (math.exp(-2 / tau) - math.exp(-4 / tau))
fig, ax = plt.subplots(figsize=(4.8, 2.25))
col = np.where(c < 2, C[0], np.where(c < 4, C[3], "#bbbbbb"))
ax.bar(c, cnt, width=w * 0.95, color=col, lw=0)
ax.text(2.2, 106, f"D₁ = {D1:.0f}", ha="left", fontsize=16, color=C[0])
ax.text(4.1, 20, f"D₂ = {D2:.0f}", ha="left", fontsize=16, color=C[3])
ax.text(5.6, 48, "τ = 2/ln(632/233)\n   = 2.0 ns", fontsize=16); ax.set_ylim(0, 128)
ax.set_xlim(0, 10); ax.set_xticks([0, 2, 4, 6, 8, 10]); ax.set_xlabel("time after the IRF (ns)"); ax.set_ylabel("photons")
fig.tight_layout()
save(fig, "fig-gates.svg")
print("gates", D1, D2)

# 6. fig-display.png: the flimcell quencher scene at 60 photons, shown three ways
if len(sys.argv) > 1:
    S = json.load(open(sys.argv[1])); n = S["n"]
    I0 = np.array(S["I0"]).reshape(n, n); I1 = np.array(S["I1"]).reshape(n, n)
    t0 = np.array([np.nan if v is None else v for v in S["tau0"]]).reshape(n, n)
    t1 = np.array([np.nan if v is None else v for v in S["tau1"]]).reshape(n, n)
    imax0 = np.sort(I0.ravel())[int(0.995 * (n * n - 1))]; imax1 = np.sort(I1.ravel())[int(0.995 * (n * n - 1))]
    flat = VIR(Normalize(0, 4, clip=True)(np.nan_to_num(t0, nan=0)))[..., :3]
    panels = [(flat, "lifetime only:\nnoise and background loud"),
              (merge(t0, I0, imax0, mask=I0 >= 20), "brightness = intensity,\nmasked below 20 photons"),
              (merge(t1, I1, imax1, mask=I1 >= 20), "3 × 3 binned:\n9× the photons")]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 4.3))
    for a, (im, title) in zip(axs, panels):
        a.imshow(im, interpolation="nearest"); a.set_title(title, fontsize=17); a.set_xticks([]); a.set_yticks([])
        [s.set_visible(False) for s in a.spines.values()]
    cax = fig.add_axes([0.3, 0.19, 0.4, 0.04])
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(0, 4), VIR), cax=cax, orientation="horizontal")
    cb.set_label("τ by mean arrival (ns)", fontsize=16); cb.ax.tick_params(labelsize=16)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.3, wspace=0.05)
    save(fig, "fig-display.png", dpi=110)
