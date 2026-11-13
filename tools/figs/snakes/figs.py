"""Figures for snakes/lec.qmd. Run from the repo root (see data.ts for the JSON):
   python3 -I tools/figs/snakes/figs.py <tmp>/snakes-data.json
The bead diagram and the square are computed here from the force formula of the slides;
the image figures are snake runs of the `snake` demo's code (data.ts), same presets.
Colours as in the demo: internal force blue, image force red, snake yellow/orange.
SVG sizes: 1 pt in the figure = 1 px on the slide at the width given in lec.qmd."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "snakes")
plt.rcParams.update({"font.size": 17, "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, ORANGE, SKY, PURPLE, GREY = "#0072b2", "#d55e00", "#009e73", "#e69f00", "#56b4e9", "#cc79a7", "#777777"
SNAKE, START = "#ffd000", "#56b4e9"


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), facecolor="white", **kw); plt.close(fig)
    print("wrote", name)


def arrow(ax, p, v, col, lw=2.5, **kw):
    ax.annotate("", xy=(p[0] + v[0], p[1] + v[1]), xytext=(p[0], p[1]),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=col, shrinkA=0, shrinkB=0, mutation_scale=18), **kw)


def internal(P, i, alpha, beta):
    n = len(P)
    m2, m1, p1, p2 = P[(i - 2) % n], P[(i - 1) % n], P[(i + 1) % n], P[(i + 2) % n]
    el = alpha * (m1 - 2 * P[i] + p1)
    rg = -beta * (m2 - 4 * m1 + 6 * P[i] - 4 * p1 + p2)
    return el, rg


# ------------------------------------------------------------------ a closed chain of beads
def fig_beads():
    # 10 beads around an object (grey blob); bead 2 sticks out (a corner)
    ang = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    rad = np.array([2.6, 2.5, 3.5, 2.6, 2.7, 2.6, 2.5, 2.7, 2.6, 2.5])
    P = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    t = np.linspace(0, 2 * np.pi, 200)
    ax.fill(1.9 * np.cos(t), 1.75 * np.sin(t), color="#d9d9d9", zorder=0)
    ax.plot(1.9 * np.cos(t), 1.75 * np.sin(t), color=GREY, lw=1.5, ls="--", zorder=1)
    ax.text(0, 0, "object", ha="center", va="center", color="#555", fontsize=17)
    Q = np.vstack([P, P[:1]])
    ax.plot(Q[:, 0], Q[:, 1], color=ORANGE, lw=2.5, zorder=2)
    ax.plot(P[:, 0], P[:, 1], "o", color="black", ms=9, zorder=3)
    i = 2
    a, b = P[i - 1], P[i + 1]
    mid = (a + b) / 2
    ax.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=1.2, ls=":", zorder=2)
    el, rg = internal(P, i, 1.0, 0.2)
    arrow(ax, P[i], 0.5 * el, BLUE)            # half the elastic force lands on the midpoint
    ax.plot(*mid, "o", mfc="white", mec=BLUE, ms=8, mew=2, zorder=4)
    ax.text(mid[0] - 0.2, mid[1] - 0.05, "midpoint", color=BLUE, fontsize=17, ha="right", va="center")
    ax.text(P[i][0] - 0.45, P[i][1] - 0.6, "elastic", color=BLUE, fontsize=17, ha="right", va="center")
    arrow(ax, P[i] + np.array([0.12, 0.0]), rg * 0.9, PURPLE)
    ax.text(P[i][0] + 0.45, P[i][1] - 0.35, "rigid", color=PURPLE, fontsize=17, ha="left", va="center")
    ax.text(P[i][0], P[i][1] + 0.3, "bead $i$", fontsize=17, ha="center", va="bottom")
    # image force at bead 7: towards the edge of the object
    j = 7
    e = np.array([1.9 * np.cos(ang[j]), 1.75 * np.sin(ang[j])])
    arrow(ax, P[j], (e - P[j]) * 0.9, RED)
    ax.text(P[j][0], P[j][1] - 0.25, "image: towards\nthe edge", color=RED, fontsize=17, ha="center", va="top")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-3.0, 3.4); ax.set_ylim(-4.0, 4.0)
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, "fig-beads.svg")


# ------------------------------------------------------------------ the square by hand
def fig_square():
    P = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 4.0))
    for ax, (alpha, beta, lab) in zip(axs, ((1, 0, "α = 1, β = 0"), (1, 1, "α = β = 1"))):
        Q = np.vstack([P, P[:1]])
        ax.plot(Q[:, 0], Q[:, 1], color=ORANGE, lw=2.5)
        ax.plot(P[:, 0], P[:, 1], "o", color="black", ms=9, zorder=3)
        new = []
        for i in range(4):
            F = sum(internal(P, i, alpha, beta))
            arrow(ax, P[i], 0.1 * F, BLUE, lw=2.5)
            new.append(P[i] + 0.1 * F)
        new = np.array(new)
        N = np.vstack([new, new[:1]])
        ax.plot(N[:, 0], N[:, 1], color=ORANGE, lw=2, ls="--")
        F0 = sum(internal(P, 0, alpha, beta))
        ax.text(0, -1.22, f"force ({F0[0]:g}, {F0[1]:g})", fontsize=18, color=BLUE, ha="center", va="top")
        ax.text(0, -1.62, f"corner → ({new[0][0]:.1f}, {new[0][1]:.1f})", ha="center", va="top", fontsize=18)
        ax.set_title(lab, fontsize=18)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-2.15, 1.2)
    fig.subplots_adjust(0.01, 0.0, 0.99, 0.9, wspace=0.12)
    save(fig, "fig-square.svg")


# ------------------------------------------------------------------ image figures
def im(ax, a, N=128, cmap="gray", vmin=0, vmax=1):
    ax.imshow(np.array(a).reshape(N, N), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_color("#999")


def curve(ax, c, **kw):
    x = np.r_[c["x"], c["x"][0]]; y = np.r_[c["y"], c["y"][0]]
    ax.plot(x, y, **kw)


def fig_edgemap(D):
    """Edge map of the noisy disk (noise 0.25) at a small and a large σ, with the force ∇f."""
    N = D["N"]
    fig, axs = plt.subplots(2, 1, figsize=(3.3, 6.6))
    yy, xx = np.mgrid[4:N:8, 4:N:8]
    for ax, E in zip(axs, D["edge"]):
        f = np.array(E["f"]).reshape(N, N)
        im(ax, f, vmax=max(0.3, f.max() * 0.8), cmap="magma")
        fx = np.array(E["fx"]).reshape(N, N)[4:N:8, 4:N:8]
        fy = np.array(E["fy"]).reshape(N, N)[4:N:8, 4:N:8]
        L = np.hypot(fx, fy)
        keep = L > 0.02 * L.max()
        ax.quiver(xx[keep], yy[keep], fx[keep], fy[keep], color="#5ee36a", angles="xy", scale_units="xy",
                  scale=L.max() / 7, width=0.012, headwidth=3.5)
        ax.set_title(f"σ = {E['sigma']:g}", fontsize=17)
        ax.set_xlim(-0.5, N - 0.5); ax.set_ylim(N - 0.5, -0.5)
    fig.subplots_adjust(0.01, 0.005, 0.99, 0.95, hspace=0.12)
    save(fig, "fig-edgemap.png", dpi=110)


def fig_explode(D):
    N = D["N"]
    cases = [(D["tau1"], 2, "τ = 1: 100 steps"), (D["tau2"], 2, "τ = 2: 30 steps"),
             (D["tau2"], 3, "τ = 2: 35 steps"), (D["tau2"], 5, "τ = 2: 40 steps")]
    fig, axs = plt.subplots(1, 4, figsize=(12.4, 3.6))
    for ax, (R, k, lab) in zip(axs, cases):
        im(ax, R["img"], vmin=-0.1, vmax=1.1)
        s = R["snaps"][k]
        curve(ax, R["snaps"][0]["c"], color=START, lw=1.5, ls="--")
        curve(ax, s["c"], color=SNAKE, lw=2)
        ax.plot(s["c"]["x"], s["c"]["y"], "o", color=SNAKE, ms=3)
        ax.set_title(lab, fontsize=20)
        ax.set_xlim(-0.5, N - 0.5); ax.set_ylim(N - 0.5, -0.5)
    fig.subplots_adjust(0.005, 0.01, 0.995, 0.88, wspace=0.05)
    save(fig, "fig-explode.png", dpi=105)


def fig_knobs(D):
    N = D["N"]
    cases = [(D["alpha"], "α = 1: shrinks"), (D["beta"], "β = 2: rounds"),
             (D["sigma"], "σ = 0.5: noise"), (D["far"], "too far")]
    fig, axs = plt.subplots(2, 2, figsize=(6.0, 6.4))
    for ax, (R, lab) in zip(axs.ravel(), cases):
        im(ax, R["img"], vmin=-0.1, vmax=1.1)
        curve(ax, R["snaps"][0]["c"], color=START, lw=1.5, ls="--")
        c = R["snaps"][-1]["c"]
        if R["snaps"][-1]["meanR"] < 1:
            ax.plot(np.mean(c["x"]), np.mean(c["y"]), "o", color=SNAKE, ms=7)
        else:
            curve(ax, c, color=SNAKE, lw=2.2)
        ax.set_title(lab, fontsize=21)
        ax.set_xlim(-0.5, N - 0.5); ax.set_ylim(N - 0.5, -0.5)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.93, wspace=0.06, hspace=0.18)
    save(fig, "fig-knobs.png", dpi=110)


if __name__ == "__main__":
    D = json.load(open(sys.argv[1]))
    fig_beads(); fig_square(); fig_edgemap(D); fig_explode(D); fig_knobs(D)
