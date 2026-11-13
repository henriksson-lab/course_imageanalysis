"""Static figures for the canny/ deck (slide pictures).

Run from the repository root:  python3 -I tools/figs/canny/figs.py
Writes canny/fig-gauss-sigma.svg, fig-separable.png, fig-gradient.png, fig-nms.svg,
fig-hysteresis.png, fig-sigma-choice.png. Kernels and Canny as in demos/src/lib/edges.ts
(numpy port in edgelib.py).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from edgelib import (gauss_kernel, blur, sep, gradient, screen_angle, canny, cell, show,
                     colour_wheel, hsv_to_rgb, BLUE, ORANGE, GREEN, RED, CYAN, GREY, PURPLE, FONT)

OUT = "canny"


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.05, **kw)
    plt.close(fig)
    print("wrote", name)


# 1. The Gaussian of width sigma ------------------------------------------------------
def fig_gauss_sigma():
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.linspace(-14, 14, 801)
    for s, c in [(1, BLUE), (2, RED), (4, GREEN)]:
        g = np.exp(-x * x / (2 * s * s)) / (np.sqrt(2 * np.pi) * s)
        ax.plot(x, g, color=c, lw=2.2, label=f"σ = {s}")
    s = 2
    k = gauss_kernel(s)
    r = len(k) // 2
    xs = np.arange(-r, r + 1)
    ax.plot(xs, k, "o", color=RED, ms=5)
    ax.vlines(xs, 0, k, color=RED, lw=1)
    xx = np.linspace(-s, s, 100)
    ax.fill_between(xx, np.exp(-xx ** 2 / (2 * s * s)) / (np.sqrt(2 * np.pi) * s), color=RED, alpha=0.18, lw=0)
    ax.annotate("68 % of the\nweight within ±σ", xy=(1.2, 0.06), xytext=(3.2, 0.2), color=RED, fontsize=FONT - 2,
                arrowprops=dict(arrowstyle="-", color=RED, lw=1))
    for xv in (-4 * s, 4 * s):
        ax.axvline(xv, color=GREY, ls="--", lw=1.2)
    ax.text(4 * s + 0.3, 0.33, "cut at 4σ:\n17 taps\nfor σ = 2", color=GREY, fontsize=FONT - 3, va="top")
    ax.set_xlabel("offset (pixels)")
    ax.set_ylabel("weight")
    ax.set_xlim(-14, 14)
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig-gauss-sigma.svg")
    print("  sigma=2 taps", len(k), "weight within +-2:", k[r - 2:r + 3].sum())


# 2. Separability --------------------------------------------------------------------
def fig_separable():
    img = cell()[40:200, 40:200]
    s = 4
    k = gauss_kernel(s)
    rows = sep(img, k, np.array([1.0]))
    both = sep(rows, np.array([1.0]), k)
    # direct 2D kernel, replicate border
    r = len(k) // 2
    K = np.outer(k, k)
    a = np.pad(img, r, mode="edge")
    direct = np.zeros_like(img)
    for i in range(len(k)):
        for j in range(len(k)):
            direct += K[i, j] * a[i:i + img.shape[0], j:j + img.shape[1]]
    diff = np.abs(direct - both).max()
    print("  separable max diff", diff, "taps", len(k))
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.6), gridspec_kw={"wspace": 0.16})
    show(axs[0], img, "input")
    show(axs[1], rows, "1D blur along rows (33 taps)")
    show(axs[2], both, "… then along columns")
    show(axs[3], direct, "2D kernel (33 × 33)")
    axs[3].text(0.5, -0.08, "differs by < 10⁻¹² grey levels", transform=axs[3].transAxes,
                ha="center", va="top", fontsize=FONT - 2)
    axs[0].text(0, -0.08, "σ = 4.  " + "Micrograph: R. M. Hunt, CC BY 3.0", transform=axs[0].transAxes,
                ha="left", va="top", fontsize=FONT - 4, color=GREY)
    for a0, a1, t in [(axs[0], axs[1], ""), (axs[1], axs[2], "")]:
        fig.add_artist(FancyArrowPatch((a0.get_position().x1 - 0.005, 0.5), (a1.get_position().x0 + 0.005, 0.5),
                                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=22, color="k"))
    fig.text((axs[2].get_position().x1 + axs[3].get_position().x0) / 2, 0.5, "=", ha="center", va="center", fontsize=30)
    save(fig, "fig-separable.png", dpi=110)


# 3. Gradient: magnitude and direction -----------------------------------------------
def fig_gradient():
    h = w = 128
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w), 60.0)
    img[np.hypot(xx - 40, yy - 44) < 24] = 180
    img[70:112, 70:112] = 140
    img = img + 4 * np.random.default_rng(1).standard_normal((h, w))
    gx, gy, mag = gradient(img, 2)
    ang = np.mod(screen_angle(gx, gy), 2 * np.pi) / (2 * np.pi)
    v = np.clip(mag / np.percentile(mag, 99.5), 0, 1)
    rgb = hsv_to_rgb(ang, np.ones_like(ang), v)
    fig = plt.figure(figsize=(5.0, 3.0))
    ax1 = fig.add_axes([0.0, 0.0, 0.46, 0.86])
    ax2 = fig.add_axes([0.5, 0.0, 0.46, 0.86])
    show(ax1, img, "∇L (σ = 2)")
    show(ax2, rgb, "direction")
    ys, xs = np.mgrid[2:128:3, 2:128:3]
    m = (mag[ys, xs] > 0.4 * mag.max())
    pts = list(zip(ys[m], xs[m]))[::5]
    py, px = np.array(pts).T
    # imshow has y down, so the image-coordinate vector (gx, gy) is drawn as it is
    ax1.quiver(px, py, gx[py, px], gy[py, px], color=ORANGE, angles="xy", scale_units="xy",
               scale=1.3, width=0.012)
    axw = fig.add_axes([0.97, 0.2, 0.3, 0.5])
    colour_wheel(axw)
    for t, (dx, dy) in [("0°", (1.6, 0)), ("90°", (0, 1.45)), ("180°", (-1.95, 0)), ("270°", (0, -1.45))]:
        axw.text(dx, dy, t, ha="center", va="center", fontsize=FONT)
    axw.set_xlim(-2.8, 2.3); axw.set_ylim(-1.8, 1.8)
    save(fig, "fig-gradient.png", dpi=130)


# 4. Non-maximum suppression -----------------------------------------------------------
def fig_nms():
    fig = plt.figure(figsize=(16, 3.9))
    # (a) magnitude profile across an edge (step 120, sigma 2)
    ax = fig.add_axes([0.04, 0.17, 0.32, 0.7])
    s = 2
    x = np.arange(-5, 6)
    g = 120 * np.exp(-x * x / (2 * s * s)) / (np.sqrt(2 * np.pi) * s)
    cols = [GREEN if xi == 0 else "#bbbbbb" for xi in x]
    ax.bar(x, g, color=cols, width=0.8)
    ax.set_xlabel("pixel, across the edge")
    ax.set_ylabel("‖∇L‖")
    ax.set_xticks(x)
    ax.set_title("ridge ≈ 2σ wide → keep the crest", fontsize=FONT)
    ax.text(0, g[5] + 1, "kept", ha="center", va="bottom", color=GREEN, fontsize=FONT - 2)
    ax.set_ylim(0, 30)
    # (b) the 3x3 example
    ax = fig.add_axes([0.42, 0.02, 0.22, 0.86])
    vals = [[4, 9, 5], [5, 10, 6], [6, 8, 7]]
    for i in range(3):
        for j in range(3):
            fc = "white"
            if (i, j) == (1, 1):
                fc = "#c9ead9"
            elif (i, j) in [(1, 0), (1, 2)]:
                fc = "#cde8f7"
            ax.add_patch(Rectangle((j, 2 - i), 1, 1, fc=fc, ec="k", lw=1.5))
            ax.text(j + 0.5, 2 - i + 0.5, str(vals[i][j]), ha="center", va="center",
                    fontsize=FONT + 5, fontweight="bold" if (i, j) == (1, 1) else "normal")
    ax.annotate("", xy=(3.0, -0.3), xytext=(0.0, -0.3),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.5, mutation_scale=22))
    ax.text(1.5, -0.45, "gradient, θ = 0°", ha="center", va="top", color=RED, fontsize=FONT - 2)
    ax.set_xlim(-0.05, 3.05); ax.set_ylim(-0.95, 3.05); ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title("10 beats 5 and 6: kept", fontsize=FONT)
    # (c) the four rounded directions
    ax = fig.add_axes([0.68, 0.0, 0.32, 0.92])
    for k, (ang, lab) in enumerate([(0, "0°"), (45, "45°"), (90, "90°"), (135, "135°")]):
        cx = k * 3.6
        for i in range(3):
            for j in range(3):
                ax.add_patch(Rectangle((cx + j, i), 1, 1, fc="white", ec="#999999", lw=1))
        dx, dy = int(round(np.cos(np.radians(ang)))), int(round(np.sin(np.radians(ang))))
        for sgn in (1, -1):
            ax.add_patch(Rectangle((cx + 1 + sgn * dx, 1 + sgn * dy), 1, 1, fc="#cde8f7", ec="#999999", lw=1))
        ax.add_patch(Rectangle((cx + 1, 1), 1, 1, fc="#c9ead9", ec="#999999", lw=1))
        ax.annotate("", xy=(cx + 1.5 + 1.25 * np.cos(np.radians(ang)), 1.5 + 1.25 * np.sin(np.radians(ang))),
                    xytext=(cx + 1.5 - 1.25 * np.cos(np.radians(ang)), 1.5 - 1.25 * np.sin(np.radians(ang))),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=2, mutation_scale=16))
        ax.text(cx + 1.5, -0.45, lab, ha="center", va="top", fontsize=FONT)
    ax.set_xlim(-0.1, 3 * 3.6 + 3.1); ax.set_ylim(-1.3, 3.1); ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title("θ rounded: the two neighbours compared", fontsize=FONT)
    save(fig, "fig-nms.svg")
    print("  ridge profile", np.round(g, 1))


# 5. Hysteresis ------------------------------------------------------------------------
def hyst_scene():
    rng = np.random.default_rng(3)
    h = w = 112
    img = np.full((h, w), 100.0)
    y0, y1, x0, x1 = 14, 98, 24, 88
    for y in range(y0, y1):
        c = 80 + (12 - 80) * (y - y0) / (y1 - 1 - y0)   # contrast fades 80 -> 12
        img[y, x0:x1] += c
    return img + 12 * rng.standard_normal((h, w))


def fig_hysteresis():
    img = hyst_scene()
    lo, hi = 3, 10
    C = canny(img, 2, lo, hi)
    n = C["nms"]
    one_hi = n >= hi
    one_lo = n >= lo
    e = C["edges"]
    print("  hyst: strong", C["strong"].sum(), "weak", C["weak"].sum(), "kept", e.sum(),
          "only-high", one_hi.sum(), "only-low", one_lo.sum())

    def overlay(mask_r, mask_o=None):
        rgb = np.zeros(img.shape + (3,))
        rgb[mask_r] = [0.84, 0.37, 0.0]
        if mask_o is not None:
            rgb[mask_o] = [0.94, 0.89, 0.26]
        return rgb

    fig, axs = plt.subplots(2, 2, figsize=(7.4, 8.0), gridspec_kw={"wspace": 0.1, "hspace": 0.2})
    show(axs[0, 0], img, "contrast fades ↓")
    show(axs[0, 1], overlay(C["strong"], C["weak"] & (n > 0)), "strong red, weak yellow")
    show(axs[1, 0], overlay(one_hi), f"one threshold: {hi}")
    show(axs[1, 1], overlay(e), f"hysteresis: {lo} and {hi}")
    save(fig, "fig-hysteresis.png", dpi=110)


# 6. Choosing sigma ----------------------------------------------------------------------
def fig_sigma_choice():
    img = cell()[48:208, 48:208]
    sets = [(1, 4, 10), (2, 2.5, 6), (4, 1.2, 3)]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 7.6), gridspec_kw={"wspace": 0.05, "hspace": 0.18})
    show(axs[0, 0], img, "micrograph")
    for ax, (s, lo, hi) in zip([axs[0, 1], axs[1, 0], axs[1, 1]], sets):
        C = canny(img, s, lo, hi)
        rgb = np.stack([img / 255 * 0.55] * 3, -1)
        rgb[C["edges"]] = [0.84, 0.37, 0.0]
        show(ax, rgb, f"σ = {s}, t = {lo:g} / {hi:g}")
        print("  sigma", s, "edge px", C["edges"].sum())
    fig.text(0.13, 0.065, "Micrograph: R. M. Hunt, CC BY 3.0, Wikimedia Commons", fontsize=FONT - 5, color=GREY)
    save(fig, "fig-sigma-choice.png", dpi=110)


if __name__ == "__main__":
    fig_gauss_sigma()
    fig_separable()
    fig_gradient()
    fig_nms()
    fig_hysteresis()
    fig_sigma_choice()
