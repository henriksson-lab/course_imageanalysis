"""Figures for snake-matrix/lec.qmd (new ones; fig-modes.svg and fig-gvf.png are older).
Run from the repo root (see data.ts for the JSON):
   python3 -I tools/figs/snake-matrix/figs.py <tmp>/sm-data.json
The amplification plots use λ_k = 4α sin²(πk/n) + 16β sin⁴(πk/n) for unit 1's snake
(n = 40, α = 0.1, β = 0.05); the λ_k are those of lib/snake.ts (data.ts) and checked here.
The snake pictures are runs of the `snake` demo's code with its presets.
SVG sizes: 1 pt in the figure = 1 px on the slide at the width given in lec.qmd."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "snake-matrix")
plt.rcParams.update({"font.size": 18, "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, ORANGE, SKY, PURPLE, GREY = "#0072b2", "#d55e00", "#009e73", "#e69f00", "#56b4e9", "#cc79a7", "#777777"
SNAKE, START = "#ffd000", "#56b4e9"
N_, A_, B_ = 40, 0.1, 0.05


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), facecolor="white", **kw); plt.close(fig)
    print("wrote", name)


def lam(n=N_, a=A_, b=B_):
    s = np.sin(np.pi * np.arange(n) / n)
    return 4 * a * s ** 2 + 16 * b * s ** 4


# ------------------------------------------------------------------ six beads, one matrix
def fig_hexagon():
    n = 6
    ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    P = np.c_[np.cos(ang), np.sin(ang)]
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    for i in range(n):
        j = (i + 1) % n
        wrap = (i == n - 1)
        ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color=ORANGE if wrap else GREY, lw=4 if wrap else 2.5,
                solid_capstyle="round", zorder=1)
    ax.plot(P[:, 0], P[:, 1], "o", color="black", ms=12, zorder=3)
    for i in range(n):
        ax.text(1.27 * P[i, 0], 1.22 * P[i, 1], f"$x_{i}$", ha="center", va="center", fontsize=20)
    m = (P[0] + P[n - 1]) / 2
    ax.text(m[0] - 0.12, m[1] + 0.08, "wrap-\naround", ha="right", va="bottom", fontsize=18, color="#b07800")
    ax.text(0, 0, "row 0:\n$2x_0 - x_1 - x_5$", ha="center", va="center", fontsize=18)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.4, 1.45)
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, "fig-hexagon.svg")


# ------------------------------------------------------------------ amplification factors
def fig_factors(D, implicit=False):
    L = np.array(D["lam40"])
    assert np.allclose(L, lam(), atol=1e-9), "λ_k differ from lib/snake.ts"
    k = np.arange(N_ // 2 + 1)
    l = L[: N_ // 2 + 1]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axhspan(-1, 1, color=GREEN, alpha=0.08, lw=0)
    ax.axhline(1, color=GREEN, lw=1, ls=":"); ax.axhline(-1, color=GREEN, lw=1, ls=":")
    ax.axhline(0, color="#999", lw=1)
    if not implicit:
        for tau, col in ((1, BLUE), (2, RED)):
            f = 1 - tau * l
            ax.plot(k, f, "o-", color=col, ms=5, lw=2, label=f"explicit, τ = {tau}")
        ax.annotate(f"{1 - 2 * l[-1]:.1f}".replace("-", "−"), xy=(k[-1], 1 - 2 * l[-1]), xytext=(k[-1] - 4.5, -1.5),
                    fontsize=18, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
        ax.text(0.6, 1.08, "stable: between −1 and 1", color=GREEN, fontsize=17, va="bottom")
        ax.set_ylim(-1.65, 1.4)
    else:
        ax.plot(k, 1 - 2 * l, "--", color=GREY, lw=1.8, label="explicit, τ = 2")
        for tau, col in ((2, BLUE), (50, PURPLE)):
            f = 1 / (1 + tau * l)
            ax.plot(k, f, "o-", color=col, ms=5, lw=2, label=f"semi-implicit, τ = {tau}")
        ax.set_ylim(-1.65, 1.4)
    ax.set_xlabel("mode k (wiggles around the curve)")
    ax.set_ylabel("factor per step")
    ax.set_xticks([0, 5, 10, 15, 20]); ax.set_xlim(-0.5, 20.8)
    ax.legend(loc="lower left", fontsize=16, frameon=False)
    fig.tight_layout(pad=0.3)
    save(fig, "fig-semi-factors.svg" if implicit else "fig-explicit-factors.svg")


# ------------------------------------------------------------------ snake pictures
def im(ax, a, N=128):
    ax.imshow(np.array(a).reshape(N, N), cmap="gray", vmin=-0.1, vmax=1.1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_color("#999")
    ax.set_xlim(-0.5, N - 0.5); ax.set_ylim(N - 0.5, -0.5)


def curve(ax, c, **kw):
    ax.plot(np.r_[c["x"], c["x"][0]], np.r_[c["y"], c["y"][0]], **kw)


def field(ax, R, N=128, step=10):
    fx = np.array(R["fx"]).reshape(N, N); fy = np.array(R["fy"]).reshape(N, N)
    yy, xx = np.mgrid[5:N:step, 5:N:step]
    u, v = fx[5:N:step, 5:N:step], fy[5:N:step, 5:N:step]
    Lm = np.hypot(u, v).max()
    ax.quiver(xx, yy, u, v, color="#5ee36a", angles="xy", scale_units="xy", scale=Lm / 8, width=0.008, headwidth=3.5)


def fig_capture(D):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.55))
    for ax, R, lab in zip(axs, (D["far"], D["fargvf"]), ("edge force", "GVF")):
        im(ax, R["img"]); field(ax, R)
        curve(ax, R["snaps"][0]["c"], color=START, lw=1.6, ls="--")
        curve(ax, R["snaps"][-1]["c"], color=SNAKE, lw=2.4)
        ax.set_title(f"{lab}: r = {R['snaps'][-1]['meanR']:.1f}", fontsize=21)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.89, wspace=0.05)
    save(fig, "fig-capture.png", dpi=120)


def fig_balloon(D):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.55))
    for ax, R, idx, lab, cols in ((axs[0], D["balloon"], [0, 1, 2, 4], "κ = 0.3: stops", None),
                                  (axs[1], D["leak"], [0, 1, 2, 3], "weak edge: κ = 0.6", None)):
        im(ax, R["img"])
        curve(ax, R["snaps"][0]["c"], color=START, lw=1.6, ls="--")
        for j, k in enumerate(idx[1:]):
            last = (j == len(idx) - 2)
            curve(ax, R["snaps"][k]["c"], color=SNAKE if last else ORANGE, lw=2.4 if last else 1.4)
        ax.set_title(lab, fontsize=21)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.89, wspace=0.05)
    save(fig, "fig-balloon.png", dpi=120)


def fig_blobs(D):
    R = D["blobs"]
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    im(ax, R["img"])
    curve(ax, R["snaps"][0]["c"], color=START, lw=1.6, ls="--")
    curve(ax, R["snaps"][-1]["c"], color=SNAKE, lw=2.4)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    save(fig, "fig-blobs.png", dpi=120)


if __name__ == "__main__":
    D = json.load(open(sys.argv[1]))
    fig_hexagon(); fig_factors(D); fig_factors(D, implicit=True)
    fig_capture(D); fig_balloon(D); fig_blobs(D)
