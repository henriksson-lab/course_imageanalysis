"""Figures for heat-equation/lec.qmd. Run from the repository root:
    python3 -I tools/figs/heat-equation/figs.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pslib import *  # noqa: E402,F401,F403
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

OUT = "heat-equation"
D = run_node("heat-equation")


def bars(ax, u, title=None, ymax=None, color=BLUE, labels=True):
    x = np.arange(len(u))
    cols = [color if v >= 0 else RED for v in u]
    ax.bar(x, u, width=0.8, color=cols)
    ax.axhline(0, color="#444", lw=1)
    if labels:
        for i, v in enumerate(u):
            ax.text(i, v + (0.15 if v >= 0 else -0.15) * (ymax or 4) / 2, f"{v:g}", ha="center", va="bottom" if v >= 0 else "top", fontsize=FONT - 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    if ymax:
        ax.set_ylim(-ymax, ymax * 1.25) if min(u) < 0 else ax.set_ylim(0, ymax * 1.25)
    if title:
        ax.set_title(title, fontsize=FONT)


# ---- fig-average-repeat.svg: a row, three steps at r = 1/4, arrows of heat flow ----
u = np.array([0, 0, 0, 8, 0, 0, 0], float)
fig, axs = plt.subplots(3, 1, figsize=(6.2, 5.6))
for k, ax in enumerate(axs):
    bars(ax, np.round(u, 3), f"after {k} step{'s' if k != 1 else ''}" if k else "start", ymax=8)
    d = np.diff(u)
    for i, di in enumerate(d):
        if abs(di) > 1e-9:
            a, b = (i, i + 1) if di < 0 else (i + 1, i)
            y = 6.5
            ax.add_patch(FancyArrowPatch((a + (0.15 if b > a else -0.15), y), (b - (0.15 if b > a else -0.15), y), arrowstyle="-|>", mutation_scale=16, color=ORANGE, lw=2))
            ax.text((a + b) / 2, y + 0.6, f"{0.25 * abs(di):.2g}", ha="center", fontsize=FONT - 3, color="#b06d00")
    u = heat1(u, 0.25)
fig.tight_layout()
save(fig, f"{OUT}/fig-average-repeat.svg")

# ---- fig-flux-divergence.svg: flux arrows between a pixel and its 4 neighbours ----
fig, ax = plt.subplots(figsize=(5.0, 4.6))
vals = {(0, 0): 10, (1, 0): 6, (-1, 0): 8, (0, 1): 4, (0, -1): 9}
for (i, j), v in vals.items():
    ax.add_patch(Rectangle((i - 0.45, j - 0.45), 0.9, 0.9, fc="#dfe9f5" if (i, j) != (0, 0) else "#f6d9b8", ec="#555", lw=1.5))
    ax.text(i, j, str(v), ha="center", va="center", fontsize=FONT + 6)
for (i, j), v in vals.items():
    if (i, j) == (0, 0):
        continue
    f = 10 - v  # flow out of the centre
    a = (0.5 * i, 0.5 * j)
    ax.add_patch(FancyArrowPatch((0.28 * i, 0.28 * j), (0.72 * i, 0.72 * j), arrowstyle="-|>", mutation_scale=18, color=ORANGE, lw=2.5))
    ax.text(a[0] + 0.18 * (j != 0), a[1] + 0.16 * (i != 0), f"{f}", fontsize=FONT, color="#b06d00", ha="left", va="bottom")
ax.text(0, -1.85, "net outflow = 4 + 2 + 6 + 1 = 13\n= 4·10 − (6 + 8 + 4 + 9) = −∇²u", ha="center", va="center", fontsize=FONT - 1)
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-2.4, 1.6); ax.set_aspect("equal"); ax.axis("off")
save(fig, f"{OUT}/fig-flux-divergence.svg")

# ---- fig-heat-vs-gauss.png: explicit steps vs Gaussian, and their difference ----
fig, axs = plt.subplots(3, 3, figsize=(8.4, 8.6))
for c, e in enumerate(D["heat"]):
    h, g = arr(e["h"]), arr(e["g"])
    t = 0.2 * e["n"]
    show(axs[0, c], h, title=f"{e['n']} steps of Δt = 0.2")
    show(axs[1, c], g, title=f"Gaussian σ = √(2t) = {np.sqrt(2 * t):g}")
    show(axs[2, c], h - g, -3, 3, title=f"difference (max {np.abs(h - g).max():.2f})")

fig.tight_layout()
save(fig, f"{OUT}/fig-heat-vs-gauss.png", dpi=110)

# ---- fig-steps-sigma.svg: cost of the explicit scheme vs a separable Gaussian ----
s = np.linspace(0.5, 12, 200)
fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.plot(s, 6 * 2 * s ** 2, color=RED, lw=3, label="explicit steps: 2σ² steps × 6")
ax.plot(s, 2 * (2 * np.ceil(4 * s) + 1), color=BLUE, lw=3, label="separable Gaussian: 2 × (8σ + 1)")
ax.set_yscale("log"); ax.set_xlabel("σ (pixels)"); ax.set_ylabel("operations per pixel")
ax.plot([10], [1200], "o", color=RED); ax.annotate("σ = 10: 200 steps", (10, 1200), (4.2, 2500), fontsize=FONT - 1, arrowprops=dict(arrowstyle="->"))
ax.legend(loc="lower right", frameon=False)
save(fig, f"{OUT}/fig-steps-sigma.svg")

# ---- fig-handstep.svg: (0, 0, 4, 0, 0) after one step at r = 1/4, 1/2, 1 ----
fig, axs = plt.subplots(1, 4, figsize=(10.5, 2.9))
bars(axs[0], [0, 0, 4, 0, 0], "start", ymax=4)
for ax, r in zip(axs[1:], (0.25, 0.5, 1)):
    v = heat1([0, 0, 4, 0, 0], r)
    bars(ax, [round(x, 3) for x in v], f"r = {r:g}: total {v.sum():g}", ymax=4)
fig.tight_layout()
save(fig, f"{OUT}/fig-handstep.svg")

# ---- fig-checkerboard.png: 1.2 × the limit on the noisy test card ----
fig, axs = plt.subplots(1, 4, figsize=(11, 3.2))
for k, (ax, img) in enumerate(zip(axs, D["checker"])):
    show(ax, arr(img), title="start (noisy test card)" if k == 0 else f"{4 * k} steps at Δt = 0.3")
fig.tight_layout()
save(fig, f"{OUT}/fig-checkerboard.png", dpi=110)

# ---- fig-amplification.svg: A(ω) = 1 − 4r sin²(ω/2) ----
w = np.linspace(0, np.pi, 300)
fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.axhspan(-1, 1, color=GREEN, alpha=0.12)
for r, c in ((0.25, BLUE), (0.5, ORANGE), (0.6, RED)):
    ax.plot(w, 1 - 4 * r * np.sin(w / 2) ** 2, lw=3, color=c, label=f"explicit, r = {r:g}")
ax.plot(w, 1 / (1 + 4 * 2 * np.sin(w / 2) ** 2), lw=3, color=PURPLE, ls="--", label="implicit, r = 2")
ax.set_xticks([0, np.pi / 2, np.pi]); ax.set_xticklabels(["0", "π/2", "π\n(checkerboard)"])
ax.set_xlabel("frequency ω"); ax.set_ylabel("amplification A(ω)")
ax.text(2.05, 0.75, "|A| ≤ 1: stable", color="#1e7a1e", fontsize=FONT - 1)
ax.legend(loc="lower left", frameon=False, fontsize=FONT - 4)
ax.set_ylim(-1.55, 1.1)
save(fig, f"{OUT}/fig-amplification.svg")

# ---- fig-cfl-table.svg: allowed Δt against the pixel size, fronts vs diffusion ----
dx = np.array([0.25, 0.5, 1, 2])
fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.loglog(dx, dx / np.sqrt(2), "o-", color=BLUE, lw=3, ms=8, label="moving front, speed 1: Δx/√2")
ax.loglog(dx, dx ** 2 / 4, "s-", color=RED, lw=3, ms=8, label="diffusion (2D): Δx²/4")
ax.minorticks_off(); ax.set_xticks(dx); ax.set_xticklabels(["1/4", "1/2", "1", "2"])
ax.set_yticks([0.01, 0.1, 1]); ax.set_yticklabels(["0.01", "0.1", "1"])
ax.set_xlabel("pixel size Δx"); ax.set_ylabel("largest stable Δt")
ax.annotate("halve Δx: 2× the steps", (0.5, 0.354), (0.55, 0.16), fontsize=FONT - 3, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE))
ax.annotate("halve Δx: 4× the steps", (0.5, 0.0625), (0.62, 0.025), fontsize=FONT - 3, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
ax.set_ylim(0.008, 3)
ax.legend(loc="upper left", frameon=False, fontsize=FONT - 4)
save(fig, f"{OUT}/fig-cfl-table.svg")

# ---- fig-implicit.svg: explicit vs implicit with Δt = 2 on a step ----
n = 40
u0 = np.where(np.arange(n) >= 20, 100.0, 0.0)
L = np.diag(np.r_[1, 2 * np.ones(n - 2), 1]) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)
ue, ui = u0.copy(), u0.copy()
for _ in range(3):
    ue = heat1(ue, 2.0)
    ui = np.linalg.solve(np.eye(n) + 2.0 * L, ui)
x = np.arange(n)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(x, u0, color=GREY, lw=1.5, ls=":", label="start")
ax.plot(x, np.clip(ue, -400, 500), color=RED, lw=2.5, label="explicit, 3 steps of Δt = 2")
ax.plot(x, ui, color=BLUE, lw=3.5, label="implicit, 3 steps of Δt = 2")
ax.plot(x, gauss1(u0, np.sqrt(12)), color="k", lw=1.8, ls="--", label="Gaussian, t = 6")
ax.set_ylim(-420, 520); ax.set_xlabel("position x"); ax.set_ylabel("value")
ax.legend(loc="upper left", frameon=False, fontsize=FONT - 4)
save(fig, f"{OUT}/fig-implicit.svg")
print("explicit Δt=2 max", np.abs(ue).max(), "implicit vs gauss max diff", np.abs(ui - gauss1(u0, np.sqrt(12))).max())
