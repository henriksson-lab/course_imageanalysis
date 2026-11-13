# Static figures for flim-fret/lec.qmd (in addition to tools/figs/flim/figs.py, which writes
# fig-fret-phasor.svg). Run from the repository root:  python3 -I tools/figs/flim-fret/figs.py
# Everything is computed from the formulas of the slides at 80 MHz; the extras of the pitfalls
# figure use the fretphasor demo's constants (autofluorescence 0.4 ns, acceptor 3.0 ns).
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 18, "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans", "svg.hashsalt": "flim-fret"})
C = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]   # as in tools/figs/flim/figs.py
GREY = "#777"
OUT = "flim-fret/"
W = 2 * math.pi * 80e-3

def save(fig, name, tight=False):
    fig.savefig(OUT + name, format="svg", facecolor="white", **({"bbox_inches": "tight", "pad_inches": 0.08} if tight else {})); plt.close(fig); print("wrote", OUT + name)

def ph(t):
    return np.array([1 / (1 + (W * t) ** 2), W * t / (1 + (W * t) ** 2)])

def semi(ax, ticks=(0.5, 1, 2, 4, 8), lab=True, fs=15):
    a = np.linspace(0, math.pi, 200)
    ax.plot(0.5 + 0.5 * np.cos(a), 0.5 * np.sin(a), color="#444", lw=1.8)
    for t in ticks:
        g, s = ph(t)
        ax.plot(g, s, "o", color="#444", ms=4)
        if lab:
            d = np.array([g - 0.5, s]); d /= np.linalg.norm(d)
            ax.text(g + 0.07 * d[0], s + 0.07 * d[1], f"{t:g}", ha="center", va="center", fontsize=fs, color="#444")
    ax.set_aspect("equal"); ax.set_xlim(-0.06, 1.08); ax.set_ylim(-0.03, 0.64)
    ax.set_xlabel("g"); ax.set_ylabel("s")

def mix(f, taus):
    """Phasor of photon fractions f over lifetimes taus."""
    return sum(fi * ph(t) for fi, t in zip(f, taus))

# 1. fig-efficiency.svg: E(r) with R0 = 5 nm
r = np.linspace(0, 12, 400); R0 = 5.0
E = 1 / (1 + (r / R0) ** 6)
fig, ax = plt.subplots(figsize=(5.4, 3.9))
ax.plot(r, E, color=C[0], lw=3)
ax.axvspan(0, 10, color=C[2], alpha=0.08, lw=0)
ax.plot([R0, R0], [0, 0.5], ":", color="#444", lw=1.5); ax.plot([0, R0], [0.5, 0.5], ":", color="#444", lw=1.5)
ax.text(R0 + 0.3, 0.52, "R₀ = 5 nm: E = 0.5", fontsize=16)
ax.text(10 - 0.2, 0.86, "FRET reports\n“within ~10 nm”", ha="right", fontsize=16, color=C[2])
ax.text(11.8, 0.1, f"10 nm:\nE = {1 / (1 + 2 ** 6):.3f}", ha="right", fontsize=15, color="#444")
ax.set_xlim(0, 12); ax.set_ylim(0, 1.05); ax.set_xlabel("donor–acceptor distance r (nm)"); ax.set_ylabel("efficiency E")
fig.tight_layout()
save(fig, "fig-efficiency.svg")

# 2. fig-donor-decay.svg: donor alone (2.5 ns) and with acceptor (1.5 ns): E = 0.4
t = np.linspace(0, 12, 400)
fig, ax = plt.subplots(figsize=(5.4, 3.9))
ax.semilogy(t, np.exp(-t / 2.5), color=C[2], lw=3, label=r"donor alone: $\tau_D$ = 2.5 ns")
ax.semilogy(t, np.exp(-t / 1.5), color=C[1], lw=3, ls="--", label=r"with acceptor: $\tau_{DA}$ = 1.5 ns")
ax.text(0.4, 2.2e-3, "E = 1 − 1.5/2.5 = 0.4", fontsize=17)
ax.set_ylim(1e-3, 1.6); ax.set_yticks([]); ax.minorticks_off()
ax.set_xlabel("time after the pulse (ns)"); ax.set_ylabel("donor emission (log)")
ax.legend(fontsize=15, frameon=False, loc="upper right")
fig.tight_layout()
save(fig, "fig-donor-decay.svg")

# 3. fig-e-vs-alpha.svg: all donors at E = 0.3 vs 5/7 of donors at E = 0.6: same mean lifetime
tD = 2.5
alpha = 5 / 7; tDA = tD * (1 - 0.6)
f = alpha * tDA / (alpha * tDA + (1 - alpha) * tD)                  # photon fraction of the FRET state
mean_b = f * tDA + (1 - f) * tD
fig, axs = plt.subplots(2, 1, figsize=(5.4, 6.2), gridspec_kw={"height_ratios": [1, 1.25]})
ax = axs[0]
ax.semilogy(t, np.exp(-t / 1.75), color=C[0], lw=3, label="all donors, E = 0.3")
ax.semilogy(t, (1 - alpha) * np.exp(-t / tD) + alpha * np.exp(-t / tDA), color=C[3], lw=3, ls="--", label="71 % of donors, E = 0.6")
ax.set_ylim(1e-3, 1.6); ax.set_yticks([]); ax.minorticks_off(); ax.set_xlim(0, 12)
ax.set_xlabel("time (ns)"); ax.set_ylabel("decay (log)")
ax.legend(fontsize=15, frameon=False, loc="upper right")
ax.set_title(f"both: intensity-weighted mean {mean_b:.2f} ns", fontsize=16)
ax = axs[1]; semi(ax, lab=False)
pD, pDA = ph(tD), ph(tDA)
ax.plot(*zip(pD, pDA), ":", color=GREY, lw=1.8)
a = ph(1.75); b = (1 - f) * pD + f * pDA
ax.plot(*pD, "o", color=C[2], ms=10, ls="none", label="donor 2.5 ns")
ax.plot(*pDA, "o", color=C[1], ms=10, ls="none", label="FRET state 1.0 ns")
ax.plot(*a, "o", color=C[0], ms=12, ls="none", label="E = 0.3: on the circle")
ax.plot(*b, "D", color=C[3], ms=11, ls="none", label="71 % at E = 0.6: inside")
ax.set_ylim(-0.03, 0.56)
ax.legend(fontsize=15, loc="lower center", ncol=1, frameon=False, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.1)
fig.tight_layout(h_pad=1.0)
save(fig, "fig-e-vs-alpha.svg")
print("photon fraction", f, "mean", mean_b, "phasors", a, b)

# 4. fig-pitfalls.svg: what the extras do to a pixel (alpha = 0.5, E = 0.5, tau_D = 2.5 ns)
tDA = 1.25
fq = 0.5 * tDA / (0.5 * tDA + 0.5 * tD)
P = (1 - fq) * ph(tD) + fq * ph(tDA)
Pa = 0.8 * P + 0.2 * ph(0.4)                                          # 20 % autofluorescence
Pb = 0.7 * P                                                          # 30 % flat background
pdA, pA = ph(tDA), ph(3.0)
leak = np.array([pdA[0] * pA[0] - pdA[1] * pA[1], pdA[0] * pA[1] + pdA[1] * pA[0]])   # P_DA x P_A
Pl = 0.3 * pdA + 0.7 * leak                                           # alpha = 1, 70 % leak
fig, ax = plt.subplots(figsize=(5.4, 4.5))
semi(ax, lab=False)
ax.plot(*zip(ph(tD), ph(tDA)), "--", color=GREY, lw=2)
for Q in (Pa, Pb): ax.annotate("", xy=Q, xytext=P, arrowprops=dict(arrowstyle="->", color="#444", lw=1.8, shrinkA=6, shrinkB=6))
ax.plot(*ph(tD), "o", color=C[2], ms=10, ls="none", label="donor 2.5 ns")
ax.plot(*ph(tDA), "o", color=C[1], ms=10, ls="none", label="FRET state 1.25 ns")
ax.plot(*P, "o", color="black", ms=11, ls="none", label="clean pixel")
ax.plot(*Pa, "s", color=C[3], ms=10, ls="none", label="+20 % autofluorescence")
ax.plot(*Pb, "s", color=C[4], ms=10, ls="none", label="+30 % background")
ax.plot(*Pl, "s", color=C[0], ms=10, ls="none", label="acceptor leak")
ax.set_ylim(-0.03, 0.56)
ax.legend(fontsize=15, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=1, frameon=False, handletextpad=0.3, columnspacing=0.8)
fig.tight_layout()
save(fig, "fig-pitfalls.svg", tight=True)
print("clean", P, "auto", Pa, "bg", Pb, "leak", Pl, "|leak-0.5|", np.hypot(Pl[0] - 0.5, Pl[1]))

# 5. fig-nadh.svg: free (0.4 ns) and bound (2.5 ns) NADH; a cell at 40 % bound photons
pf, pb = ph(0.4), ph(2.5)
cellp = 0.6 * pf + 0.4 * pb
fig, ax = plt.subplots(figsize=(5.4, 4.1))
semi(ax, lab=False)
ax.plot(*zip(pf, pb), color=C[0], lw=2.2)
for q in (0.2, 0.4, 0.6, 0.8):
    m = (1 - q) * pf + q * pb; ax.plot(*m, "o", color=C[0], ms=4)
ax.plot(*pf, "o", color=C[3], ms=11, ls="none", label=f"free 0.4 ns ({pf[0]:.2f}, {pf[1]:.2f})")
ax.plot(*pb, "o", color=C[4], ms=11, ls="none", label=f"bound 2.5 ns ({pb[0]:.2f}, {pb[1]:.2f})")
ax.plot(*cellp, "D", color=C[1], ms=11, ls="none", label=f"cell ({cellp[0]:.2f}, {cellp[1]:.2f}): 40 % bound photons")
ax.text(0.5 * (pf[0] + pb[0]) - 0.06, 0.5 * (pf[1] + pb[1]) - 0.17, "dots: 20 % steps", color=C[0], fontsize=15, ha="center")
ax.set_ylim(-0.03, 0.56)
ax.legend(fontsize=15, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=1, frameon=False, handletextpad=0.3, columnspacing=0.8)
fig.tight_layout()
save(fig, "fig-nadh.svg", tight=True)

# 6. fig-triangle.svg: free, bound and a label (4.5 ns); a pixel inside the triangle
pl = ph(4.5)
fr = np.array([0.5, 0.3, 0.2])
px = fr[0] * pf + fr[1] * pb + fr[2] * pl
d = pb - pf; q = np.dot(px - pf, d) / np.dot(d, d)                    # two-species reading: lever rule along the line
proj = pf + q * d
fig, ax = plt.subplots(figsize=(5.4, 4.7))
semi(ax, lab=False)
tri = np.array([pf, pb, pl, pf])
ax.fill(tri[:, 0], tri[:, 1], color=C[0], alpha=0.12, lw=0); ax.plot(tri[:, 0], tri[:, 1], color=C[0], lw=2)
ax.plot(*zip(pf, pb), ":", color=C[0], lw=1)
ax.plot(*pf, "o", color=C[3], ms=11, ls="none", label="free 0.4 ns")
ax.plot(*pb, "o", color=C[4], ms=11, ls="none", label="bound 2.5 ns")
ax.plot(*pl, "o", color=C[2], ms=11, ls="none", label="label 4.5 ns")
ax.plot(*px, "D", color=C[1], ms=11, ls="none", label="pixel: 50 / 30 / 20 %")
ax.plot(*zip(px, proj), "-", color="black", lw=1.5)
ax.plot(*proj, "x", color="black", ms=11, mew=2.5, ls="none", label=f"two-species reading: {100 * q:.0f} % bound")
ax.set_ylim(-0.03, 0.56)
ax.legend(fontsize=15, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=1, frameon=False, handletextpad=0.3, columnspacing=0.8)
fig.tight_layout()
save(fig, "fig-triangle.svg", tight=True)
print("pixel", px, "two-species bound fraction", q)
