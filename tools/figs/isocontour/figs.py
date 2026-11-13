"""Figures for isocontour/lec.qmd. Run from the repo root (data.ts first, see there):
   python3 -I tools/figs/isocontour/figs.py <tmp>/iso-data.json
Contours come from the course's lib/marching.ts (via data.ts). Colours as in fig-cases.svg:
orange = above c, blue = contour. Fonts >= 15 pt at the size the slides show them."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "isocontour")
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
ORANGE, BLUE, GREEN, RED, PURPLE, GREY = "#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, facecolor="white")
    if os.environ.get("FIGPREVIEW"):
        fig.savefig(os.path.join(os.environ["FIGPREVIEW"], name.replace(".svg", ".png")), facecolor="white", dpi=100)
    plt.close(fig)
    print("wrote", p)


def draw_lines(ax, lines, **kw):
    for L in lines:
        xy = np.array(L["xy"])
        if L["closed"]:
            xy = np.vstack([xy, xy[:1]])
        ax.plot(xy[:, 0], xy[:, 1], **kw)


def staircase(ax, mask, **kw):
    """Outline of the pixels where mask is true (pixel squares centred on integer coordinates)."""
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            if y == 0 or not mask[y - 1, x]: ax.plot([x - .5, x + .5], [y - .5, y - .5], **kw)
            if y == h - 1 or not mask[y + 1, x]: ax.plot([x - .5, x + .5], [y + .5, y + .5], **kw)
            if x == 0 or not mask[y, x - 1]: ax.plot([x - .5, x - .5], [y - .5, y + .5], **kw)
            if x == w - 1 or not mask[y, x + 1]: ax.plot([x + .5, x + .5], [y - .5, y + .5], **kw)


# 1. pixels above c (staircase) vs the level set between the pixel centres
def fig_levelset(d):
    s = d["small"]; n = s["n"]
    img = np.array(s["img"], float).reshape(n, n)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    for y in range(n):
        for x in range(n):
            v = img[y, x]
            ax.text(x, y, f"{v:.0f}", ha="center", va="center", fontsize=14, color="black" if v > 110 else "white")
    staircase(ax, img > 120, color=ORANGE, lw=3, solid_capstyle="round")
    draw_lines(ax, s["linear"]["lines"], color=BLUE, lw=3.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    fig.tight_layout()
    save(fig, "fig-levelset.svg")


# 2. the crossing point on one edge
def fig_edge():
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot([0, 1], [40, 200], color="#333", lw=2.5)
    ax.plot([0, 1], [40, 200], "o", color="#333", ms=9)
    ax.text(0.05, 40, "$I_a$ = 40", va="center", ha="left", fontsize=16)
    ax.text(0.97, 200, "$I_b$ = 200", va="center", ha="right", fontsize=16)
    for c, col in ((120, BLUE), (80, GREEN)):
        t = (c - 40) / (200 - 40)
        ax.axhline(c, color=col, lw=1.5, ls="--")
        ax.plot([t, t], [0, c], color=col, lw=1.5, ls=":")
        ax.plot(t, c, "o", color=col, ms=11, mec="black")
        ax.text(1.0, c + 4, f"c = {c}", color=col, ha="right", va="bottom", fontsize=16)
        ax.text(t + 0.03, c - 8, f"t = {t:g}", color=col, ha="left", va="top", fontsize=16)
    ax.axvline(0.5, color=GREY, lw=0)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(0, 220)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["a", "0.25", "0.5", "0.75", "b"])
    ax.set_xlabel("position t along the edge"); ax.set_ylabel("grey value")
    fig.tight_layout()
    save(fig, "fig-edge.svg")


# 3. marching over a grid: shared crossing points, closed and oriented curves (a bright ring)
def fig_orient(d):
    r = d["ring"]; n = r["n"]
    img = np.array(r["img"], float).reshape(n, n)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    for y in range(n):
        ax.plot([0, n - 1], [y, y], color="#ccc", lw=1)
    for x in range(n):
        ax.plot([x, x], [0, n - 1], color="#ccc", lw=1)
    yy, xx = np.mgrid[0:n, 0:n]
    above = img > 100
    ax.plot(xx[above], yy[above], "o", color=ORANGE, ms=8)
    ax.plot(xx[~above], yy[~above], "o", color="#bbb", ms=4)
    for L in r["lines"]:
        xy = np.array(L["xy"])
        col = BLUE if L["area"] > 0 else PURPLE
        xyc = np.vstack([xy, xy[:1]])
        ax.plot(xyc[:, 0], xyc[:, 1], color=col, lw=3)
        ax.plot(xy[:, 0], xy[:, 1], ".", color="black", ms=6)
        k = len(xy)
        for i in range(0, k, max(1, k // 4)):
            a, b = xy[i], xy[(i + 1) % k]
            m = (a + b) / 2
            ax.add_patch(FancyArrowPatch(tuple(a + 0.2 * (b - a)), tuple(m + 0.25 * (b - a)), arrowstyle="-|>",
                                         mutation_scale=32, color=col, lw=0))
    outer = [L for L in r["lines"] if L["area"] > 0][0]["area"]
    hole = [L for L in r["lines"] if L["area"] < 0][0]["area"]
    ax.text(-0.4, n - 0.3, f"outer: area {outer:+.1f}", color=BLUE, fontsize=16, va="top")
    ax.text(n - 0.6, n - 0.3, f"hole: {hole:+.1f}", color=PURPLE, fontsize=16, va="top", ha="right")
    ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(n + 0.6, -0.6)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    save(fig, "fig-orient.svg")
    return outer, hole


# 4. the asymptotic decider: bilinear interpolant, its level curve at c, and the saddle
def fig_decider():
    c = 5.0
    cases = [((8, 4, 7, 3), "join"), ((6, 4, 6.5, 3), "separate")]   # TL, TR, BR, BL
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 3.6))
    xs = np.linspace(0, 1, 301)
    X, Y = np.meshgrid(xs, xs)
    for ax, ((tl, tr, br, bl), verdict) in zip(axs, cases):
        I00, I10, I11, I01 = tl, tr, br, bl          # first index x, y down
        F = I00 * (1 - X) * (1 - Y) + I10 * X * (1 - Y) + I01 * (1 - X) * Y + I11 * X * Y
        ax.contourf(X, Y, F > c, levels=[-0.5, 0.5, 1.5], colors=["#f4f4f4", "#ffd9b3"])
        ax.contour(X, Y, F, levels=[c], colors=[BLUE], linewidths=3)
        D = I00 + I11 - I10 - I01
        fs = (I00 * I11 - I10 * I01) / D
        xs_, ys_ = -(I01 - I00) / D, -(I10 - I00) / D        # saddle: x* = -C/D, y* = -B/D with B = I10 - I00, C = I01 - I00
        ax.axvline(xs_, color=GREY, lw=1, ls="--"); ax.axhline(ys_, color=GREY, lw=1, ls="--")
        ax.plot(xs_, ys_, "X", color=RED, ms=13, mec="black")
        for (x, y), v in {(0, 0): tl, (1, 0): tr, (1, 1): br, (0, 1): bl}.items():
            ax.plot(x, y, "o", ms=16, color=ORANGE if v > c else "#ccc", mec="#333", clip_on=False)
            ax.text(x + (-0.09 if x == 0 else 0.09), y + (-0.09 if y == 0 else 0.09), f"{v:g}", fontsize=19,
                    ha="right" if x == 0 else "left", va="bottom" if y == 0 else "top", fontweight="bold")
        rel = ">" if fs > c else "<"
        ax.set_xlabel(f"$f^*$ = {fs:.2f} {rel} 5: {verdict}", fontsize=18, color=RED, labelpad=34)
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#333")
    fig.tight_layout(w_pad=3)
    save(fig, "fig-decider.svg")


# 5. noise vs blur first on the demo's test image (presets "noise" and "blur first", c = 120)
def fig_noise(d):
    N = d["N"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.9))
    for ax, key, title in ((axs[0], "noise", "noise σ = 25"), (axs[1], "blur", "then blur σ = 1.5")):
        img = np.array(d[key]["img"], float).reshape(N, N)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=(-0.5, N - 0.5, N - 0.5, -0.5))
        draw_lines(ax, d[key]["lines"], color="#ffd400", lw=1.6)
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(f"{d[key]['count']} contours, {d[key]['length']:.0f} px", fontsize=20)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
    fig.tight_layout()
    save(fig, "fig-noise.svg")


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    fig_levelset(data); fig_edge(); print(fig_orient(data)); fig_decider(); fig_noise(data)
