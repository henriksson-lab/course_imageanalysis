"""Figures for tv-flow/lec.qmd. Run from the repository root:
    python3 -I tools/figs/tv-flow/figs.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "heat-equation"))
from pslib import *  # noqa: E402,F401,F403

OUT = "tv-flow"
D = run_node("tv-flow")

# ---- fig-three-penalties.svg: Psi(s) and g(s) = Psi'(s)/s for heat, TV, PM (K = 2) ----
s = np.linspace(0.001, 10, 400)
K = 2
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
a1.plot(s, s ** 2 / 2, color=ORANGE, lw=3, label="heat: s²/2")
a1.plot(s, s, color=BLUE, lw=3, label="TV: s")
a1.plot(s, K * K / 2 * np.log(1 + (s / K) ** 2), color=GREEN, lw=3, label="Perona–Malik: (K²/2) ln(1 + s²/K²)")
a1.set_ylim(0, 12); a1.set_xlabel("gradient size s"); a1.set_ylabel("penalty Ψ(s)")
a1.legend(loc="upper left", frameon=False, fontsize=FONT - 4); a1.set_title("what an edge of slope s costs", fontsize=FONT - 1)
a2.plot(s, np.ones_like(s), color=ORANGE, lw=3, label="heat: 1")
a2.plot(s, 1 / s, color=BLUE, lw=3, label="TV: 1/s")
a2.plot(s, 1 / (1 + (s / K) ** 2), color=GREEN, lw=3, label="PM: 1/(1 + s²/K²)")
a2.set_ylim(0, 2.2); a2.set_xlabel("gradient size s"); a2.set_ylabel("conductivity g = Ψ′(s)/s")
a2.legend(loc="upper right", frameon=False, fontsize=FONT - 4); a2.set_title("how much heat it lets through", fontsize=FONT - 1)
fig.tight_layout()
save(fig, f"{OUT}/fig-three-penalties.svg")

# ---- fig-energy-descent.svg: a noisy step under the three flows; energy against iteration ----
rng = np.random.default_rng(2)
n = 64
y = np.where(np.arange(n) >= 32, 10.0, 0.0) + 0.8 * rng.standard_normal(n)


def flow1(u, kind, dt, eps=0.05):
    d = np.diff(u)
    if kind == "heat":
        f = d
    elif kind == "tv":
        f = d / np.sqrt(d * d + eps * eps)
    else:
        f = d / (1 + (d / K) ** 2)
    out = u.copy(); out[:-1] += dt * f; out[1:] -= dt * f
    return out


def energy(u, kind):
    d = np.diff(u)
    return {"heat": np.sum(d * d / 2), "tv": np.sum(np.abs(d)), "pm": np.sum(K * K / 2 * np.log(1 + (d / K) ** 2))}[kind]


fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
a1.plot(y, color=GREY, lw=1.5, ls=":", label="start")
for kind, col, dt, it, lab in (("heat", ORANGE, 0.4, 25, "heat"), ("tv", BLUE, 0.02, 500, "TV flow"), ("pm", GREEN, 0.4, 25, "Perona–Malik")):
    u = y.copy(); E = [energy(u, kind)]
    for _ in range(it):
        u = flow1(u, kind, dt); E.append(energy(u, kind))
    a1.plot(u, color=col, lw=2.5, label=lab)
    tt = np.arange(it + 1) * dt
    a2.plot(tt[1:], np.array(E[1:]) / E[0], color=col, lw=3, label=lab)
a1.set_xlabel("position x"); a1.set_ylabel("value"); a1.legend(loc="upper left", frameon=False, fontsize=FONT - 4)
a1.set_title("after t = 10", fontsize=FONT - 2)
a2.set_xlabel("time t"); a2.set_ylabel("own energy / start"); a2.set_xscale("log"); a2.legend(loc="lower left", frameon=False, fontsize=FONT - 4)
a2.set_title("each flow lowers its own energy", fontsize=FONT - 2)
fig.tight_layout()
save(fig, f"{OUT}/fig-energy-descent.svg")

# ---- fig-level-lines.png: iso-contours before / after TV flow ----
d0 = arr(D["discs0"]); S = D["snaps"]
fig, axs = plt.subplots(1, 3, figsize=(11, 4.1))
for ax, img, t in zip(axs, (d0, arr(S["t40"]), arr(S["t120"])), (0, 40, 120)):
    show(ax, img, 50, 110, title=f"TV flow, t = {t}")
    ax.contour(img, levels=[70, 80, 90], colors=[CYAN, YELLOW, RED], linewidths=1.4)
fig.tight_layout()
save(fig, f"{OUT}/fig-level-lines.png", dpi=110)

# ---- fig-discs-flow.png: three discs flowing + contrast plot ----
fig = plt.figure(figsize=(11, 4.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.4])
for k, (img, t) in enumerate(((arr(S["t40"]), 40), (arr(S["t120"]), 120))):
    ax = fig.add_subplot(gs[0, k]); show(ax, img, 50, 110, title=f"t = {t}")
ax = fig.add_subplot(gs[0, 2])
T = np.array(D["times"]); C = np.array(D["contrast"])
for j, (r_, col) in enumerate(((4, RED), (8, ORANGE), (16, BLUE))):
    ax.plot(T, C[:, j], "o-", color=col, lw=2.5, ms=4, label=f"r = {r_}")
    ax.plot([0, 20 * r_], [40, 0], ls="--", color=col, lw=1.5)
ax.set_ylim(-5, 44); ax.set_xlabel("time t"); ax.set_ylabel("contrast")
ax.legend(loc="upper right", frameon=False, fontsize=FONT - 3)
ax.set_title("dashed: 40 − 2t/r", fontsize=FONT - 2)
fig.tight_layout()
save(fig, f"{OUT}/fig-discs-flow.png", dpi=110)

# ---- fig-tv-byhand.svg: (0,0,0,6,6,0,0,0), ROF with lambda = 2 and 6 ----
y8 = np.array([0, 0, 0, 6, 6, 0, 0, 0.])
fig, ax = plt.subplots(figsize=(6.4, 4.0))
x = np.arange(1, 9)
ax.plot(x, y8, "s", color=GREY, ms=11, label="data y")
ax.plot(x, D["rof2"], "o-", color=BLUE, lw=3, ms=8, label="λ = 2 (= flow time 2)")
ax.plot(x, D["rof6"], "^--", color=RED, lw=2.5, ms=8, label="λ = 6")
for i, v in enumerate(D["rof2"]):
    ax.text(x[i], v + 0.25, f"{v:.2f}" if abs(v - 4) > 0.01 else "4", ha="center", color=BLUE, fontsize=FONT - 4)
ax.set_ylim(-0.5, 7); ax.set_xlabel("pixel"); ax.set_ylabel("value"); ax.set_xticks(x)
ax.legend(loc="upper left", frameon=False, fontsize=FONT - 4)
save(fig, f"{OUT}/fig-tv-byhand.svg")

# ---- fig-corner-round.png: the square under TV flow ----
fig, axs = plt.subplots(1, 3, figsize=(11, 4.0))
for ax, img, t in zip(axs, (d0, arr(S["t120"]), arr(S["t240"])), (0, 120, 240)):
    show(ax, img[56:120, 16:80], 50, 110, title=f"square, t = {t}")
    ax.contour(img[56:120, 16:80], levels=[80], colors=[RED], linewidths=1.5)
fig.tight_layout()
save(fig, f"{OUT}/fig-corner-round.png", dpi=110)

# ---- fig-stiff.svg: steps needed against epsilon ----
eps = np.logspace(-3, 1, 100)
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.loglog(eps, 10 / (eps / 4), color=RED, lw=3, label="TV flow to t = 10: Δt ≤ ε/4")
ax.axhline(10 / 0.25, color=ORANGE, lw=3, ls="--", label="heat to t = 10: Δt ≤ 1/4")
ax.set_xlabel("ε"); ax.set_ylabel("explicit steps"); ax.legend(loc="upper right", frameon=False, fontsize=FONT - 4)
ax.plot([0.1], [400], "o", color=RED); ax.annotate("ε = 0.1: 400 steps", (0.1, 400), (0.15, 3000), fontsize=FONT - 2, arrowprops=dict(arrowstyle="->"))
save(fig, f"{OUT}/fig-stiff.svg")

# ---- fig-tv-pm-heat.png: three results at equal flat-region noise, with profiles ----
c = D["cmp"]
fig = plt.figure(figsize=(11, 6.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.62])
for k, (key, lab) in enumerate((("tv", f"TV, λ = {c['lam']:.1f}"), ("pm", f"Perona–Malik, t = {c['pmT']:.1f}"), ("heat", f"heat, σ = {c['sig']:.2f}"))):
    ax = fig.add_subplot(gs[0, k]); show(ax, arr(c[key]), title=f"{lab}: {c['psnr'][k]:.1f} dB")
ax = fig.add_subplot(gs[1, :])
xs = np.arange(128); row = 112
ax.plot(xs, arr(c["truth"])[row], color="k", lw=1.5, ls="--", label="truth")
for key, col, lab in (("heat", ORANGE, "heat"), ("pm", GREEN, "Perona–Malik"), ("tv", BLUE, "TV")):
    ax.plot(xs, arr(c[key])[row], color=col, lw=2.2, label=lab)
ax.set_xlabel("row 112: step, ramp, 1-px line, soft step"); ax.set_ylabel("value")
ax.legend(loc="upper right", ncol=4, frameon=False, fontsize=FONT - 4)
fig.tight_layout()
save(fig, f"{OUT}/fig-tv-pm-heat.png", dpi=100)
print("cmp", c["lam"], c["sig"], c["pmT"], c["sd"], c["psnr"])
print("contrast", [[round(v, 1) for v in cc] for cc in D["contrast"]])
