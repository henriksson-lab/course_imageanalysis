"""Figures for perona-malik/lec.qmd. Run from the repository root:
    python3 -I tools/figs/perona-malik/figs.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "heat-equation"))
from pslib import *  # noqa: E402,F401,F403
from matplotlib.patches import FancyArrowPatch  # noqa: E402

OUT = "perona-malik"
D = run_node("perona-malik")

# ---- fig-conductivity.png: noisy card, g from the raw and from the pre-smoothed image ----
fig, axs = plt.subplots(1, 3, figsize=(11, 4.0))
show(axs[0], arr(D["noisy"]), title="noisy test card (σn = 12)")
show(axs[1], arr(D["gRaw"]), 0, 1, title="g(|∇u|), K = 10")
show(axs[2], arr(D["g"]), 0, 1, title="g(|∇(G₁ * u)|), K = 10")
fig.tight_layout()
save(fig, f"{OUT}/fig-conductivity.png", dpi=110)

# ---- fig-g-functions.svg ----
s = np.linspace(0, 4, 300)
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.plot(s, np.ones_like(s), color=GREY, lw=3, label="linear: g = 1")
ax.plot(s, np.exp(-s ** 2), color=BLUE, lw=3, label="g₁ = exp(−(s/K)²)")
ax.plot(s, 1 / (1 + s ** 2), color=ORANGE, lw=3, ls="--", label="g₂ = 1/(1 + (s/K)²)")
ax.axvline(1, color=RED, lw=1.5, ls=":")
ax.text(1.04, 0.9, "s = K", color=RED, fontsize=FONT - 1)
ax.set_xlabel("gradient size s / K"); ax.set_ylabel("conductivity g")
ax.set_ylim(0, 1.15); ax.legend(loc="upper right", frameon=False, fontsize=FONT - 3)
save(fig, f"{OUT}/fig-g-functions.svg")


def rowbars(ax, u, title, color=BLUE, ymax=55):
    x = np.arange(len(u))
    ax.bar(x, u, color=color, width=0.75)
    for i, v in enumerate(u):
        ax.text(i, v + 1.2, f"{v:.2f}".rstrip("0").rstrip("."), ha="center", fontsize=FONT - 3)
    ax.set_ylim(0, ymax); ax.set_xticks([]); ax.set_yticks([])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=FONT - 1)


# ---- fig-pm-byhand.svg: (10, 12, 10, 50, 50), K = 5, dt = 0.25 ----
row = np.array([10, 12, 10, 50, 50.])
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1))
rowbars(axs[0], row, "start", GREY)
rowbars(axs[1], heat1(row, 0.25), "heat, Δt = 0.25", ORANGE)
rowbars(axs[2], pm1(row, 0.25, 5), "Perona–Malik, K = 5", BLUE)
fig.tight_layout()
save(fig, f"{OUT}/fig-pm-byhand.svg")

# ---- fig-pm-vs-heat.png: heat vs PM at the same time ----
S = D["snaps"]
fig, axs = plt.subplots(2, 3, figsize=(10.5, 7.4))
for c, k in enumerate((8, 60, 400)):
    show(axs[0, c], arr(S[f"h{k}"]), title=f"heat, t = {0.225 * k:g}")
    show(axs[1, c], arr(S[f"p{k}"]), title=f"Perona–Malik, t = {0.225 * k:g}")
fig.tight_layout()
save(fig, f"{OUT}/fig-pm-vs-heat.png", dpi=110)

# ---- fig-flux.svg: flux phi(s) = s g2(s) with forward/backward ranges, and a gradient histogram ----
K = 5
s = np.linspace(0, 30, 400)
rng = np.random.default_rng(3)
prof = np.where(np.arange(200) >= 100, 60.0, 20.0) + 2.5 * rng.standard_normal(200)
d = np.abs(np.diff(prof))
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.2, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
ax.axvspan(0, K, color=GREEN, alpha=0.12); ax.axvspan(K, 30, color=RED, alpha=0.10)
ax.plot(s, s / (1 + (s / K) ** 2), color=BLUE, lw=3)
ax.text(1.0, 0.4, "forward:\nsmooths", color="#1e7a1e", fontsize=FONT - 2)
ax.text(17, 2.0, "backward:\nsharpens", color=RED, fontsize=FONT - 2)
ax.set_ylabel("flux φ(s) = s·g₂(s)"); ax.set_ylim(0, 2.9)
ax.axvline(K, color=RED, lw=1.5, ls=":"); ax.text(K + 0.3, 2.65, "K = 5", color=RED, fontsize=FONT - 2)
ax2.hist(d, bins=np.arange(0, 31, 1), color=GREY)
ax2.set_xlabel("size of a difference s"); ax2.set_ylabel("count")
ax2.text(12, 40, "noise differences: small\nthe step: 40", fontsize=FONT - 3)
fig.tight_layout()
save(fig, f"{OUT}/fig-flux.svg")

# ---- fig-sharpen.svg: (0, 2, 6, 10, 12), K = 2, one step; and a soft edge (max slope 20), K = 8 ----
u0 = np.array([0, 2, 6, 10, 12.])
u1 = pm1(u0, 0.25, 2)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1, 1.4]})
x = np.arange(5)
ax.plot(x, u0, "o-", color=GREY, lw=2, ms=8, label="start")
ax.plot(x, u1, "s-", color=BLUE, lw=3, ms=8, label="one step, K = 2")
for i in range(5):
    ax.text(i + 0.1, u1[i] - 1.0, f"{u1[i]:g}", color=BLUE, fontsize=FONT - 3)
ax.set_xticks(x); ax.set_xlabel("pixel"); ax.set_ylabel("value")
ax.legend(loc="upper left", frameon=False, fontsize=FONT - 3)
ax.set_title("by hand: shoulders flatten,\nthe middle stays steep", fontsize=FONT - 2)
xx = np.arange(60)
soft = 60 + 120 / (1 + np.exp(-(xx - 30) / 1.5))
u = soft.copy()
cols = {0: GREY, 20: BLUE, 150: RED}
for k in range(151):
    if k in cols:
        ax2.plot(xx, u, color=cols[k], lw=1.5 if k == 0 else 3, ls=":" if k == 0 else "-", label="start: soft edge, max slope 20" if k == 0 else f"{k} steps, K = 8")
    u = pm1(u, 0.45, 8)
hs = 60 + 120 / (1 + np.exp(-(xx - 30) / 1.5))
ax2.plot(xx, gauss1(hs, np.sqrt(2 * 0.45 * 150)), color=ORANGE, lw=2, ls="--", label="heat, 150 steps")
ax2.set_xlim(15, 45); ax2.set_xlabel("position x")
ax2.legend(loc="upper left", frameon=False, fontsize=FONT - 4)
ax2.set_title("a soft edge becomes a step", fontsize=FONT - 2)
fig.tight_layout()
save(fig, f"{OUT}/fig-sharpen.svg")
print("soft edge max slope after 150", np.abs(np.diff(u)).max())

# ---- fig-staircase.svg: a ramp of slope 4, K = 3, small noise ----
rng = np.random.default_rng(5)
n = 64
ramp = np.clip(4.0 * (np.arange(n) - 12), 0, 160) + 1.0 * rng.standard_normal(n)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
u = ramp.copy()
cols = {0: GREY, 50: BLUE, 500: RED}
for k in range(501):
    if k in cols:
        ax.plot(np.arange(n), u, color=cols[k], lw=2.5 if k else 1.5, label=f"{k} steps" if k else "start (slope 4, noise 1)")
    u = pm1(u, 0.45, 3)
ax.set_xlabel("position x"); ax.set_ylabel("value"); ax.legend(loc="upper left", frameon=False, fontsize=FONT - 3)
ax.set_title("ramp steeper than K = 3: terraces", fontsize=FONT - 1)
save(fig, f"{OUT}/fig-staircase.svg")


# ---- fig-spike.svg: a spike under g1 (K = 10), with and without pre-smoothing ----
def pm1_pre(u, dt, K, kind, sig):
    v = gauss1(u, sig) if sig > 0 else u
    d = np.diff(u); dv = np.abs(np.diff(v))
    f = g_of(kind, dv, K) * d
    out = u.copy(); out[:-1] += dt * f; out[1:] -= dt * f
    return out


base = np.full(40, 60.0); base[20] = 100.0
fig, ax = plt.subplots(figsize=(6.2, 3.9))
ax.plot(base, color=GREY, lw=1.5, ls=":", label="start: a spike of 40")
for sig, col, lab in ((0, BLUE, "classical, 800 steps"), (1.5, RED, "pre-smoothed σ = 1.5, 800 steps")):
    u = base.copy()
    for _ in range(800):
        u = pm1_pre(u, 0.45, 10, "g1", sig)
    ax.plot(u, color=col, lw=3 if sig == 0 else 2.5, ls="-" if sig == 0 else "--", label=lab)
    print("spike", sig, u[20])
ax.set_ylim(30, 108); ax.set_xlabel("position x"); ax.set_ylabel("value")
ax.legend(loc="lower right", frameon=False, fontsize=FONT - 3)
save(fig, f"{OUT}/fig-spike.svg")

# ---- fig-psnr-time.svg ----
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ts, ph, pp = np.array(D["ts"]), np.array(D["ph"]), np.array(D["pp"])
ax.plot(ts, pp, color=BLUE, lw=3, label="Perona–Malik (g₂, K = 10)")
ax.plot(ts, ph, color=ORANGE, lw=3, ls="--", label="heat")
for y_, c in ((pp, BLUE), (ph, ORANGE)):
    i = int(np.argmax(y_)); ax.plot(ts[i], y_[i], "o", color=c, ms=9)
ax.set_xscale("log"); ax.set_xlabel("time t"); ax.set_ylabel("PSNR (dB)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=FONT - 3)
save(fig, f"{OUT}/fig-psnr-time.svg")
i, j = int(np.argmax(pp)), int(np.argmax(ph))
print("psnr best PM", pp[i], ts[i], "heat", ph[j], ts[j], "noisy", pp[0], "PM at end", pp[-1])

# ---- fig-xi-eta.svg: gradient direction eta and level-line direction xi at an edge pixel ----
fig, ax = plt.subplots(figsize=(4.8, 4.2))
X, Y = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200))
Z = 1 / (1 + np.exp(-(X + 0.5 * Y) * 6))
ax.imshow(Z, cmap="gray", extent=(-1, 1, -1, 1), origin="lower", vmin=-0.2, vmax=1.2)
ax.contour(X, Y, Z, levels=[0.2, 0.5, 0.8], colors=[CYAN], linewidths=1.5)
e = np.array([1, 0.5]) / np.hypot(1, 0.5); xi = np.array([-e[1], e[0]])
ax.add_patch(FancyArrowPatch((0, 0), tuple(0.7 * e), arrowstyle="-|>", mutation_scale=22, color=RED, lw=3))
ax.add_patch(FancyArrowPatch((0, 0), tuple(0.7 * xi), arrowstyle="-|>", mutation_scale=22, color=YELLOW, lw=3))
ax.text(0.72 * e[0], 0.72 * e[1] - 0.15, "η (across)", color=RED, fontsize=FONT, fontweight="bold")
ax.text(0.7 * xi[0] - 0.35, 0.7 * xi[1] + 0.07, "ξ (along)", color=YELLOW, fontsize=FONT, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])
save(fig, f"{OUT}/fig-xi-eta.svg")
