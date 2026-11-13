# Static figures for phasor/lec.qmd (in addition to tools/figs/flim/figs.py, which writes
# fig-semicircle.svg). Run from the repository root:  python3 -I tools/figs/phasor/figs.py
# All curves are computed from the formulas of the slides (80 MHz unless stated); the noisy
# clouds use simulated photon arrival times (fixed seed).
import math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 20, "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})
C = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]   # as in tools/figs/flim/figs.py
EXC, EMI = "#ff7f0e", "#1f77b4"                               # excitation orange, emission blue (as in the demo)
W = 2 * math.pi * 80e-3                                        # rad/ns at 80 MHz
OUT = "phasor/"

def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT + name, format="svg")
    if os.environ.get("PREVIEW"): fig.savefig(os.path.join(os.environ["PREVIEW"], name[:-4] + ".png"), dpi=96)
    plt.close(fig); print("wrote", OUT + name)

def ph(t, w=W):
    q = 1 + (w * t) ** 2
    return 1 / q, w * t / q

def semi(ax, ticks=(0.5, 1, 2, 4, 8), lab=True):
    a = np.linspace(0, math.pi, 200)
    ax.plot(0.5 + 0.5 * np.cos(a), 0.5 * np.sin(a), color="#444", lw=1.8)
    for t in ticks:
        g, s = ph(t)
        ax.plot(g, s, "o", color="#444", ms=4)
        if lab:
            d = np.array([g - 0.5, s]); d /= np.linalg.norm(d)
            ax.text(g + 0.08 * d[0], s + 0.08 * d[1], f"{t:g}", ha="center", va="center", fontsize=18, color="#444")
    ax.set_aspect("equal"); ax.set_xlim(-0.06, 1.08); ax.set_ylim(-0.03, 0.66)
    ax.set_xlabel("g"); ax.set_ylabel("s")

# 1. fig-waves.svg: excitation and emission at 80 MHz for tau = 2 ns
tau = 2.0; phi = math.atan(W * tau); m = 1 / math.sqrt(1 + (W * tau) ** 2)
t = np.linspace(0, 25, 600)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(t, 1 + np.cos(W * t), color=EXC, lw=2.5, label="excitation")
ax.plot(t, 1 + m * np.cos(W * t - phi), color=EMI, lw=2.5, label="emission")
tp = phi / W                                                   # the first emission peak
ax.annotate("", xy=(12.5 + tp, 2.12), xytext=(12.5, 2.12), arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
ax.text(12.5 + tp / 2, 2.2, f"delay φ = {math.degrees(phi):.1f}°", ha="center", va="bottom", fontsize=19)
ax.axvline(12.5, color="#bbb", lw=1, ls=":", ymax=2.1 / 3.5); ax.axvline(12.5 + tp, color="#bbb", lw=1, ls=":", ymax=2.1 / 3.5)
tp2 = tp                                                       # the first emission peak
ax.annotate("", xy=(tp2, 1 + m), xytext=(tp2, 1), arrowprops=dict(arrowstyle="<->", color=EMI, lw=1.5))
ax.text(tp2 + 0.5, 1 + m / 2, f"m = {m:.3f}", color=EMI, fontsize=19, va="center", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
ax.axhline(1, color="#bbb", lw=1, ls=":")
ax.set_xlabel("time (ns)"); ax.set_ylabel("intensity"); ax.set_ylim(0, 3.5); ax.set_xlim(0, 25)
ax.legend(frameon=False, fontsize=18, loc="upper left", ncol=2, bbox_to_anchor=(-0.01, 1.04))
print("waves: phi", round(math.degrees(phi), 1), "m", round(m, 3))
save(fig, "fig-waves.svg")

# 2. fig-frequency.svg: phase and modulation against the lifetime at three frequencies
taus = np.linspace(0, 10, 300)
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3))
for f, col in zip((10, 80, 200), (C[2], C[0], C[1])):
    w = 2 * math.pi * f * 1e-3
    axs[0].plot(taus, np.degrees(np.arctan(w * taus)), color=col, lw=2.5, label=f"{f} MHz")
    axs[1].plot(taus, 1 / np.sqrt(1 + (w * taus) ** 2), color=col, lw=2.5)
axs[0].set_ylabel("phase φ (°)"); axs[0].set_ylim(0, 90); axs[0].set_yticks([0, 45, 90])
axs[1].set_ylabel("modulation m"); axs[1].set_ylim(0, 1.02)
for ax in axs: ax.set_xlabel("lifetime τ (ns)"); ax.set_xlim(0, 10)
axs[1].legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=17, loc="center right", bbox_to_anchor=(1.1, 0.66), handlelength=1.2, labelspacing=0.3)
save(fig, "fig-frequency.svg")

# 3. fig-gs.svg: a TCSPC histogram and the cosine and sine it is projected onto
T = 12.5; nb = 64; tc = (np.arange(nb) + 0.5) * T / nb; tau = 2.5
y = np.exp(-tc / tau); y /= y.sum()
g = (y * np.cos(W * tc)).sum(); s = (y * np.sin(W * tc)).sum()
fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.bar(tc, y / y.max(), width=T / nb, color="#9ecae1", label="histogram")
ax.plot(tc, np.cos(W * tc), color=C[1], lw=2, label="cos ωt")
ax.plot(tc, np.sin(W * tc), color=C[2], lw=2, ls="--", label="sin ωt")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("time in the laser period (ns)"); ax.set_xlim(0, T); ax.set_ylim(-1.1, 1.15)
ax.legend(frameon=False, fontsize=18, loc="upper center", ncol=3, bbox_to_anchor=(0.45, -0.36), columnspacing=0.6, handlelength=1.3, handletextpad=0.4)
ax.set_title(f"τ = 2.5 ns:  g = {g:.2f},  s = {s:.2f}", fontsize=19)
print("gs", round(g, 3), round(s, 3), "exact", [round(v, 3) for v in ph(2.5)])
save(fig, "fig-gs.svg")

# 4. fig-chord.svg: two species on the semicircle, the chord, photon fractions along it
tA, tB = 0.4, 4.0; PA, PB = np.array(ph(tA)), np.array(ph(tB))
fA = tA / (tA + tB); fB = tB / (tA + tB)              # photon fractions for equal molecules
fig, ax = plt.subplots(figsize=(5.6, 3.6)); semi(ax, ticks=(1, 2, 8))
ax.plot(*np.vstack([PA, PB]).T, color=C[0], lw=2.2)
for f in (0.25, 0.5, 0.75):
    q = (1 - f) * PA + f * PB; ax.plot(*q, "|", color=C[0], ms=10, mew=2)
ax.plot(*PA, "o", color=C[3], ms=10); ax.plot(*PB, "o", color=C[4], ms=10)
ax.text(PA[0] + 0.04, PA[1] - 0.04, "0.4 ns", color=C[3], fontsize=19, ha="center", va="top")
ax.text(PB[0] - 0.03, PB[1] + 0.03, "4 ns", color=C[4], fontsize=19, ha="right")
mix = fA * PA + fB * PB; ax.plot(*mix, "s", color=C[1], ms=9)
ax.annotate(f"equal molecules:\n{fB * 100:.0f} % of photons\nfrom 4 ns", xy=mix, xytext=(0.3, 0.03), fontsize=17, color=C[1],
            arrowprops=dict(arrowstyle="->", color=C[1]))
print("chord: photon fraction long", round(fB, 3))
save(fig, "fig-chord.svg")

# 5. fig-calibration.svg: the IRF rotates and shrinks the phasors; one complex factor undoes it
rot, shrink = math.radians(11.5), 0.956                # the demo's uncalibrated IRF (sigma 0.6 ns, delay 0.4 ns)
fac = shrink * np.exp(1j * rot)                        # measured = true x IRF phasor: turned to a larger phase, shrunk
fig, ax = plt.subplots(figsize=(5.6, 3.6)); semi(ax, ticks=(0.5, 1, 2, 4, 8), lab=False)
a = np.linspace(0, math.pi, 200); circ = (0.5 + 0.5 * np.cos(a)) + 1j * 0.5 * np.sin(a)
ax.plot((circ * fac).real, (circ * fac).imag, color=C[1], lw=1.5, ls="--")
for tt in (0.5, 1, 2, 4):
    P = complex(*ph(tt)); Q = P * fac
    ax.annotate("", xy=(P.real, P.imag), xytext=(Q.real, Q.imag), arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    ax.plot(Q.real, Q.imag, "o", color=C[1], ms=6); ax.plot(P.real, P.imag, "o", color="#444", ms=6)
ax.text(0.5, 0.13, "true", color="#444", fontsize=19, ha="center")
ax.text(0.5, 0.04, "measured", color=C[1], fontsize=19, ha="center")
save(fig, "fig-calibration.svg")

# 6. fig-noise.svg: 100 and 1000 photons of a 2.5 ns decay, and a pixel with flat background
rng = np.random.default_rng(2)
def cloud(n, k=300, bg=0.0):
    out = []
    for _ in range(k):
        nb_ = rng.binomial(n, bg); ts = np.concatenate([rng.exponential(2.5, n - nb_) % T, rng.uniform(0, T, nb_)])
        out.append((np.cos(W * ts).mean(), np.sin(W * ts).mean()))
    return np.array(out)
fig, ax = plt.subplots(figsize=(5.6, 3.6)); semi(ax, ticks=(0.5, 1, 2, 4, 8), lab=False)
c100, c1k = cloud(100), cloud(1000)
ax.plot(c100[:, 0], c100[:, 1], ".", color=C[0], ms=3, alpha=0.5, label="100 photons")
ax.plot(c1k[:, 0], c1k[:, 1], ".", color=C[2], ms=3, alpha=0.7, label="1000 photons")
P = np.array(ph(2.5)); ax.plot([0, P[0]], [0, P[1]], color=C[3], lw=1.8, ls="--")
for f in (0.25, 0.5):
    ax.plot(*((1 - f) * P), "o", color=C[3], ms=7)
ax.text(0.07, 0.24, "+ background", color=C[3], fontsize=19, ha="center", rotation=math.degrees(math.atan2(P[1], P[0])))
ax.legend(frameon=False, fontsize=17, loc="lower center", markerscale=4, handletextpad=0.2, bbox_to_anchor=(0.66, -0.03))
print("noise: sd per coordinate at 100 photons", c100.std(0).round(3), "1000:", c1k.std(0).round(3))
save(fig, "fig-noise.svg")
