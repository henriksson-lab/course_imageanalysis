"""Figures for shape-models/lec.qmd (new ones; fig-hand-modes.svg is older).
Run from the repo root (see data.ts for the JSON):
   python3 -I tools/figs/shape-models/figs.py <tmp>/shm-data.json
Hands, GPA leaves and variance shares come from the course's shape code (data.ts); the
Procrustes triangle is the by-hand example of the slides (a = 1.2, b = 0.9), computed here.
Colours as in the procrustes demo: target set blue, moving set orange.
SVG sizes: 1 pt in the figure = 1 px on the slide at the width given in lec.qmd."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "shape-models")
plt.rcParams.update({"font.size": 18, "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, ORANGE, SKY, PURPLE, GREY = "#0072b2", "#d55e00", "#009e73", "#e69f00", "#56b4e9", "#cc79a7", "#777777"
ANCH = [0, 4, 9, 14, 19, 24, 28]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), facecolor="white"); plt.close(fig)
    print("wrote", name)


def xy(s):
    s = np.asarray(s, float)
    return s[0::2], s[1::2]


def poly(ax, s, col, lw=2, **kw):
    x, y = xy(s)
    ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], color=col, lw=lw, solid_joinstyle="round", **kw)


def sim(s, a, b, tx=0.0, ty=0.0, reflect=False):
    x, y = xy(s)
    if reflect: x = -x
    out = np.empty(2 * len(x)); out[0::2] = a * x - b * y + tx; out[1::2] = b * x + a * y + ty
    return out


# ------------------------------------------------------------------ landmarks on two hands
def fig_landmarks(D):
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 3.9))
    for ax, s in zip(axs, D["hands"][:2]):
        x, y = xy(s)
        poly(ax, s, GREY, 1.6)
        ax.plot(x, y, "o", color="black", ms=4)
        ax.plot(x[ANCH], y[ANCH], "o", color="black", ms=7)
        ax.plot(x[9], y[9], "o", color=RED, ms=12, zorder=5)
        ax.text(x[9] - 0.06, y[9] - 0.07, "index tip", color=RED, ha="center", va="bottom", fontsize=18)
        for i, dx, dy in ((0, -0.04, 0.1), (28, 0.04, 0.1)):
            ax.text(x[i] + dx, y[i] + dy, str(i), ha="right" if dx < 0 else "left", va="center", fontsize=17)
        ax.set_aspect("equal"); ax.axis("off"); ax.invert_yaxis()
        ax.set_xlim(-1.02, 0.72); ax.set_ylim(0.8, -1.04)
    fig.subplots_adjust(0, 0, 1, 1, wspace=0.02)
    save(fig, "fig-landmarks.svg")


# ------------------------------------------------------------------ same shape, other pose
def fig_sameshape(D):
    s = np.array(D["hands"][2])
    # y down in the data: flip to y up for maths-style rotation
    up = s.copy(); up[1::2] *= -1
    th = np.deg2rad(30)
    A = sim(up, 1.0, 0.0, 0, 0)
    B = sim(up, 2 * np.cos(th), 2 * np.sin(th), 2.9, -0.2)
    C = sim(up, 1.0, 0.0, 5.6, 0.0, reflect=True)
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    for sh, col in ((A, BLUE), (B, BLUE), (C, RED)):
        x, y = xy(sh)
        ax.fill(x, y, color=col, alpha=0.12, lw=0)
        poly(ax, sh, col, 2.2)
    ax.text(0, -1.25, "a hand", ha="center", va="top", fontsize=18)
    ax.text(2.9, -2.35, "×2, turned 30°, moved:\nsame shape", ha="center", va="top", fontsize=18, color=BLUE)
    ax.text(5.6, -1.25, "mirror image:\nnot the same", ha="center", va="top", fontsize=18, color=RED)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.8, 6.5); ax.set_ylim(-3.15, 2.15)
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, "fig-sameshape.svg")


# ------------------------------------------------------------------ Procrustes in three steps
def fig_procrustes():
    TRI = np.array([0, 0, 2, 0, 0, 1], float)
    P = TRI + np.array([0.6, 0.5] * 3)                         # orange, as drawn in the demo
    Q = sim(TRI, 1.2, 0.9, -1, -1.5)                           # blue
    cP = np.array([P[0::2].mean(), P[1::2].mean()]); cQ = np.array([Q[0::2].mean(), Q[1::2].mean()])
    p0 = P - np.tile(cP, 3); q0 = Q - np.tile(cQ, 3)
    a = (p0 @ q0) / (p0 @ p0)
    b = sum(p0[2 * i] * q0[2 * i + 1] - p0[2 * i + 1] * q0[2 * i] for i in range(3)) / (p0 @ p0)
    assert abs(a - 1.2) < 1e-12 and abs(b - 0.9) < 1e-12
    s = np.hypot(a, b)
    stages = [("as drawn", P),
              ("1. centre", p0 + np.tile(cQ, 3)),
              (f"2. scale ×{s:.1f}", s * p0 + np.tile(cQ, 3)),
              (f"3. rotate {np.degrees(np.arctan2(b, a)):.1f}°", sim(p0, a, b, *cQ))]
    fig, axs = plt.subplots(2, 2, figsize=(5.4, 5.2))
    for ax, (lab, S) in zip(axs.ravel(), stages):
        for i in range(3):
            ax.plot([S[2 * i], Q[2 * i]], [S[2 * i + 1], Q[2 * i + 1]], color=GREY, lw=1, ls=":")
        poly(ax, Q, BLUE, 2.5); poly(ax, S, ORANGE, 2.5)
        ax.plot(Q[0::2], Q[1::2], "o", color=BLUE, ms=7); ax.plot(S[0::2], S[1::2], "o", color=ORANGE, ms=7)
        ax.set_title(lab, fontsize=19)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-2.4, 2.9); ax.set_ylim(-1.9, 2.0)
    fig.subplots_adjust(0, 0, 1, 0.93, wspace=0.05, hspace=0.15)
    save(fig, "fig-procrustes-steps.svg")


# ------------------------------------------------------------------ GPA
def fig_gpa(D):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.6))
    for s in D["raw"]:
        poly(axs[0], s, GREY, 1.4)
    axs[0].set_title("10 leaves as given", fontsize=18)
    G = D["gpa"][2]
    k = 1.25 * np.sqrt(12)
    for s in G["al"]:
        poly(axs[1], np.array(s) * k, GREY, 1.4)
    poly(axs[1], np.array(G["mean"]) * k, "black", 3)
    axs[1].set_title("after 2 GPA rounds", fontsize=18)
    for ax in axs:
        ax.set_aspect("equal"); ax.axis("off")
    for ax in axs:
        ax.relim(); ax.autoscale_view(); ax.margins(0.08)
    fig.subplots_adjust(0, 0, 1, 0.9, wspace=0.05)
    save(fig, "fig-gpa.svg")


# ------------------------------------------------------------------ variance shares
def fig_shares(D):
    g, b = D["shares"]["good"]["share"][:6], D["shares"]["bad"]["share"][:6]
    k = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    ax.bar(k - 0.2, g, 0.4, color=BLUE, label="correct numbering")
    ax.bar(k + 0.2, b, 0.4, color=RED, label=f"{D['shares']['nbad']} of 40 shifted by one")
    for x, v, c in ((1 - 0.2, g[0], BLUE), (1 + 0.2, b[0], RED), (2 - 0.2, g[1], BLUE)):
        ax.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=17, color=c)
    ax.set_xticks(k); ax.set_xlabel("mode"); ax.set_ylabel("% of the variance")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=17, loc="upper right")
    fig.tight_layout(pad=0.3)
    save(fig, "fig-shares.svg")


if __name__ == "__main__":
    D = json.load(open(sys.argv[1]))
    fig_landmarks(D); fig_sameshape(D); fig_procrustes(); fig_gpa(D); fig_shares(D)
