"""Figures for asm/lec.qmd (new ones; fig-asm-steps.png is older).
Run from the repo root (see data.ts for the JSON):
   python3 -I tools/figs/asm/figs.py <tmp>/asm-data.json
Everything except the by-hand profile comes from runs of the asm demo's code (data.ts),
with its presets; the by-hand profile is the table of the slide "A profile by hand".
Colours as in the demo: model shape blue, suggested points red, search normals light blue,
snake green, true outline dashed white.
SVG sizes: 1 pt in the figure = 1 px on the slide at the width given in lec.qmd."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "asm")
plt.rcParams.update({"font.size": 18, "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, ORANGE, SKY, PURPLE, GREY = "#0072b2", "#d55e00", "#009e73", "#e69f00", "#56b4e9", "#cc79a7", "#777777"
SHAPE, SUGG, SNAKE = "#3aa0ff", "#ff4040", "#5ee36a"


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), facecolor="white", **kw); plt.close(fig)
    print("wrote", name)


def xy(s):
    s = np.asarray(s, float)
    return s[0::2], s[1::2]


def im(ax, I, crop=None):
    a = np.array(I["d"]).reshape(I["h"], I["w"])
    ax.imshow(a, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_color("#999")
    if crop:
        ax.set_xlim(crop[0], crop[1]); ax.set_ylim(crop[3], crop[2])


def poly(ax, s, col, lw=2, **kw):
    x, y = xy(s)
    ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], color=col, lw=lw, **kw)


CROP = (14, 114, 22, 106)   # x0, x1, y0, y1: the nucleus and its surroundings


# ------------------------------------------------------------------ looking along the normal
def fig_normals(D):
    R = D["normals"]; m = D["m"]
    X = np.array(R["X"]); Nn = np.array(R["N"]); L = R["L"]
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]
    im(ax, R["img"], CROP)
    x, y = xy(X); nx, ny = xy(Nn)
    for i in range(len(x)):
        ax.plot([x[i] - m * nx[i], x[i] + m * nx[i]], [y[i] - m * ny[i], y[i] + m * ny[i]], color=SKY, lw=1.2)
    poly(ax, X, SHAPE, 2)
    ax.plot(x, y, "o", color=SHAPE, ms=3)
    ax.plot([x[L] - m * nx[L], x[L] + m * nx[L]], [y[L] - m * ny[L], y[L] + m * ny[L]], color=ORANGE, lw=3)
    ax.plot(x[L], y[L], "o", color=ORANGE, ms=8)
    ax.set_title("normals at the landmarks", fontsize=18)
    ax = axs[1]
    t = np.arange(-9, 10)
    p = np.array(R["prof"])
    ax.axvspan(-m - 0.5, m + 0.5, color=SKY, alpha=0.15, lw=0)
    ax.axhline(0, color="#999", lw=1)
    ax.plot(t, p, "o-", color="black", lw=2, ms=5)
    best = int(np.argmin(R["costs"])) - m
    ax.axvline(best, color=RED, lw=2, ls="--")
    ax.text(best + 0.4, 0.07, f"best match: {best}".replace("-", "−"), color=RED, fontsize=17, va="bottom", ha="left")
    ax.text(2.5, -0.145, f"search ±{m}", color=BLUE, ha="center", fontsize=17)
    ax.set_xlabel("offset along the normal (px)")
    ax.set_ylabel("grey-level change")
    ax.set_xticks([-9, -6, -3, 0, 3, 6, 9]); ax.set_ylim(-0.16, 0.1)
    fig.subplots_adjust(0.005, 0.17, 0.99, 0.91, wspace=0.42)
    save(fig, "fig-normals.png", dpi=110)


# ------------------------------------------------------------------ the profile by hand
def fig_profile():
    off = np.arange(-4, 5)
    v = np.array([0.0, 0.6, 0.0, 0.1, 0.3, 0.5, 0.2, 0.0, 0.1])
    model = np.array([0.3, 0.5, 0.2])
    centres = np.arange(-3, 4)
    ssd = np.array([((v[c + 4 - 1:c + 4 + 2] - model) ** 2).sum() for c in centres])
    assert abs(ssd[0] - 0.14) < 1e-12 and abs(ssd[4]) < 1e-12
    fig, axs = plt.subplots(2, 1, figsize=(5.4, 5.0), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]})
    ax = axs[0]
    ax.bar(off, v, 0.7, color="#bbbbbb", label="profile")
    ax.plot([0, 1, 2], model, "o-", color=BLUE, lw=2.5, ms=8, label="model, centred at +1")
    ax.plot([-4, -3, -2], model, "o--", color=ORANGE, lw=2, ms=7, label="centred at −3")
    ax.set_ylabel("derivative"); ax.set_ylim(0, 0.98)
    ax.legend(frameon=False, fontsize=15, loc="upper right")
    ax = axs[1]
    ax.bar(centres, ssd, 0.6, color=[BLUE if c == 1 else (ORANGE if c == -3 else GREY) for c in centres])
    for c, s in zip(centres, ssd):
        ax.text(c, s + 0.01, f"{s:.2f}".rstrip("0").rstrip(".") if s > 0 else "0", ha="center", va="bottom", fontsize=15)
    ax.set_ylabel("mismatch"); ax.set_ylim(0, 0.5)
    ax.set_xlabel("offset"); ax.set_xticks(off)
    fig.subplots_adjust(0.17, 0.12, 0.99, 0.98, hspace=0.12)
    save(fig, "fig-profile.svg")
    return ssd


# ------------------------------------------------------------------ one iteration, step by step
def fig_loop(D):
    R = D["loop"]
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.75))
    panels = [("suggest", "start", True), ("fit the pose", "pose", True), ("project, clamp", "clamp", False)]
    for ax, (lab, key, sugg) in zip(axs, panels):
        im(ax, R["img"], CROP)
        if key != "start":
            poly(ax, R["start"], SHAPE, 1.2, ls=":")
        poly(ax, R[key], SHAPE, 2.2)
        if sugg:
            x, y = xy(R["Y"]); ax.plot(x, y, "o", color=SUGG, ms=5)
        ax.set_title(lab, fontsize=24)
    fig.subplots_adjust(0.005, 0.01, 0.995, 0.88, wspace=0.04)
    save(fig, "fig-loop.png", dpi=110)


# ------------------------------------------------------------------ occlusion
def fig_occlusion(D):
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.75))
    for ax, R, lab in ((axs[0], D["occ"], "ASM"), (axs[1], D["occFree"], "no constraint")):
        im(ax, R["img"], CROP)
        poly(ax, R["truth"], "white", 1.3, ls="--")
        poly(ax, R["end"], SHAPE, 2.4)
        ax.set_title(f"{lab}: {R['e']:.1f} px", fontsize=20.5)
    fig.subplots_adjust(0.005, 0.01, 0.995, 0.88, wspace=0.04)
    save(fig, "fig-occlusion.png", dpi=110)


# ------------------------------------------------------------------ coarse to fine
def fig_pyramid(D):
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.75))
    for ax, R, lab in ((axs[0], D["far1"], "1 level"), (axs[1], D["far3"], "3 levels")):
        im(ax, R["img"], (4, 124, 4, 124))
        poly(ax, R["start"], SKY, 1.5, ls="--")
        poly(ax, R["end"], SHAPE, 2.4)
        ax.set_title(f"{lab}: {R['e']:.1f} px", fontsize=23)
    fig.subplots_adjust(0.005, 0.01, 0.995, 0.88, wspace=0.04)
    save(fig, "fig-pyramid.png", dpi=110)


# ------------------------------------------------------------------ outside the training set
def fig_outside(D):
    R = D["outside"]
    fig, ax = plt.subplots(figsize=(3.7, 3.7))
    im(ax, R["img"], CROP)
    poly(ax, R["end"], SHAPE, 2.4)
    sx, sy = np.array(R["snake"]["x"]), np.array(R["snake"]["y"])
    ax.plot(np.r_[sx, sx[0]], np.r_[sy, sy[0]], color=SNAKE, lw=2)
    ax.text(CROP[0] + 3, CROP[3] - 3, "snake", color=SNAKE, fontsize=19, va="bottom", fontweight="bold")
    ax.text(CROP[0] + 3, CROP[3] - 14, "ASM", color=SHAPE, fontsize=19, va="bottom", fontweight="bold")
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    save(fig, "fig-outside.png", dpi=120)


if __name__ == "__main__":
    D = json.load(open(sys.argv[1]))
    fig_normals(D); print("SSD", fig_profile()); fig_loop(D); fig_occlusion(D); fig_pyramid(D); fig_outside(D)
