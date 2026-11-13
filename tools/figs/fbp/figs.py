"""Static figures for the fbp/ deck (slide pictures).

Run from the repository root:  python3 -I tools/figs/fbp/figs.py
Writes fbp/fig-pointramp.png, fig-ramp.svg, fig-boxfilter.svg, fig-streaks.png, fig-lemon.png.
Reconstructions use the demos' own library (tools/figs/radon/compute.ts): the same
projector, filters and test objects as the backproj and tomosampling demos.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "radon"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from common import FONT, BLUE, ORANGE, GREEN, RED, PURPLE, GREY, save, load, data, show, N

UNIT = "fbp"
BIG = {"font.size": 21, "xtick.labelsize": 19, "ytick.labelsize": 19, "axes.labelsize": 21, "legend.fontsize": 19}
NORMAL = {"font.size": FONT, "xtick.labelsize": FONT - 2, "ytick.labelsize": FONT - 2, "axes.labelsize": FONT,
          "legend.fontsize": FONT - 2}


def ramlak(k):
    k = np.asarray(k)
    h = np.where(k % 2 == 1, -1 / (np.pi * np.where(k == 0, 1, k)) ** 2, 0.0)
    return np.where(k == 0, 0.25, h)


# 1. Filter first: the filtered projection of a point and the 16-angle back-projections ----
def fig_pointramp():
    plt.rcParams.update(BIG)
    d = data()
    p = np.array(d["ptProj"]); q = np.array(d["ptProjRamp"])
    c = (N - 1) / 2
    b = np.arange(N) - c
    plain = load("pt_plain_16"); ramp = load("pt_ramp_16")
    sl = slice(32, 96)
    fig = plt.figure(figsize=(7.4, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1], hspace=0.38, wspace=0.08)
    axs = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ax = axs[0]
    m = np.abs(b) <= 6
    ax.axhline(0, color="black", lw=0.8)
    ax.plot(b[m], p[m] / p.max(), "-o", color=GREY, lw=2, ms=6, label="projection")
    ax.plot(b[m], q[m] / q.max(), "-o", color=RED, lw=2.4, ms=6, label="ramp-filtered")
    ax.set_xlabel("ρ (pixels)")
    ax.set_yticks([0, 1]); ax.set_ylim(-0.75, 1.25)
    ax.text(-5.8, 0.85, "projection", color=GREY, fontsize=20)
    ax.text(-5.8, 0.55, "ramp-filtered", color=RED, fontsize=20)
    ax.text(2.2, -0.5, "negative sidelobes", color=RED, fontsize=19, va="center")
    pv = plain[sl, sl]
    show(axs[1], pv, 0, np.percentile(pv, 99.5))
    axs[1].set_title("plain, 16 angles")
    rv = ramp[sl, sl]
    lim = np.percentile(np.abs(rv), 99)
    axs[2].imshow(rv, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    axs[2].set_xticks([]); axs[2].set_yticks([])
    for s in axs[2].spines.values():
        s.set_visible(False)
    axs[2].set_title("ramp, 16 angles")
    save(fig, UNIT, "fig-pointramp.png", dpi=110)
    plt.rcParams.update(NORMAL)


# 2. Why a ramp: spoke density 1/|w| times the ramp |w| = uniform -----------------------
def fig_ramp():
    plt.rcParams.update({"font.size": 24, "xtick.labelsize": 22, "ytick.labelsize": 22, "axes.labelsize": 24})
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(6.6, 9.4), gridspec_kw={"height_ratios": [1, 1], "hspace": 0.3})
    ns = 16
    for k in range(ns):
        th = np.pi * k / ns
        a0.plot([-np.cos(th), np.cos(th)], [-np.sin(th), np.sin(th)], color=ORANGE, lw=1.4)
    for r, c in [(0.3, BLUE), (0.95, PURPLE)]:
        a0.add_patch(Circle((0, 0), r, fill=False, ec=c, lw=1.4, ls="--"))
        for k in range(2 * ns):
            th = np.pi * k / ns
            a0.plot(r * np.cos(th), r * np.sin(th), "o", color=c, ms=5)
    a0.set_aspect("equal"); a0.set_xlim(-1.05, 1.05); a0.set_ylim(-1.05, 1.05); a0.axis("off")
    w = np.linspace(0.08, 2.0, 400)
    a1.plot(w, 1 / w, color=BLUE, lw=2.8)
    a1.plot(w, w, color=RED, lw=2.8)
    a1.plot(w, np.ones_like(w), color="black", lw=2.2, ls="--")
    a1.text(0.2, 2.2, "sample density ∝ 1/|ω|", color=BLUE, fontsize=22, va="top")
    a1.text(1.4, 1.55, "ramp |ω|", color=RED, fontsize=22)
    a1.text(0.12, 0.88, "product = 1", color="black", fontsize=22, va="top")
    a1.set_xlim(0, 2); a1.set_ylim(0, 2.3)
    a1.set_xlabel("|ω|, distance from the centre")
    a1.set_ylabel("weight")
    a1.set_xticks([0, 1, 2]); a1.set_yticks([0, 1, 2])
    save(fig, UNIT, "fig-ramp.svg")
    plt.rcParams.update(NORMAL)


# 3. By hand: the ramp kernel on the projection of a box -----------------------------------
def fig_boxfilter():
    plt.rcParams.update(BIG)
    x = np.arange(-4, 11)
    box = ((x >= 2) & (x <= 4)).astype(float)
    k3 = np.array([-0.1, 0.25, -0.1])
    out3 = np.convolve(box, k3, mode="same")
    K = np.arange(-60, 61)
    xs = np.arange(-70, 80)
    boxl = ((xs >= 2) & (xs <= 4)).astype(float)
    full = np.convolve(boxl, ramlak(K), mode="same")
    full = full[(xs >= -4) & (xs <= 10)]
    print("  3-tap", out3[(x >= 0) & (x <= 6)], " full", np.round(full[(x >= 0) & (x <= 6)], 3))
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(6.6, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.5], "hspace": 0.18})
    a0.bar(x, box, width=0.6, color=GREY)
    a0.set_ylabel("box"); a0.set_yticks([0, 1]); a0.set_ylim(0, 1.25)
    a1.axhline(0, color="black", lw=0.8)
    a1.bar(x - 0.17, out3, width=0.32, color=RED, label="3-tap kernel")
    a1.bar(x + 0.17, full, width=0.32, color=BLUE, label="full Ram–Lak kernel")
    for xi, v in zip(x, out3):
        if abs(v) > 1e-9 and 0 <= xi <= 6:
            a1.text(xi - 0.17, v + (0.012 if v > 0 else -0.012), f"{v:.2f}".replace("-", "−"), ha="center",
                    va="bottom" if v > 0 else "top", fontsize=17, color=RED)
    a1.set_ylabel("filtered")
    a1.set_ylim(-0.16, 0.24)
    a1.set_xlabel("detector bin")
    a1.set_xticks(range(0, 7))
    a1.set_xlim(-1.6, 7.6)
    a1.text(5.4, 0.2, "3-tap kernel", color=RED, fontsize=19)
    a1.text(5.4, 0.15, "full Ram–Lak", color=BLUE, fontsize=19)
    save(fig, UNIT, "fig-boxfilter.svg")
    plt.rcParams.update(NORMAL)


# 4. Too few angles: streaks ---------------------------------------------------------------
def fig_streaks():
    plt.rcParams.update(BIG)
    d = data()
    truth = load("shepp"); r18 = load("fbp18"); r200 = load("fbp200")
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.6), gridspec_kw={"wspace": 0.08})
    for ax, im, t in [(axs[0], r18, f"18 angles\nRMSE {d['rmse18']:.3f}"), (axs[1], r200, f"200 angles\nRMSE {d['rmse200']:.3f}")]:
        show(ax, im, 0, 1.0 * truth.max() * 0.35)
        ax.set_title(t, fontsize=30)
    save(fig, UNIT, "fig-streaks.png", dpi=110)
    plt.rcParams.update(NORMAL)


# 5. Limited angle: the missing wedge and the lemon ----------------------------------------
def fig_lemon():
    plt.rcParams.update(BIG)
    lemon = load("lemon")
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.6), gridspec_kw={"wspace": 0.08})
    ax = axs[0]
    for deg in np.arange(-60, 61, 2):
        t = np.deg2rad(deg)
        ax.plot([-np.cos(t), np.cos(t)], [-np.sin(t), np.sin(t)], color=ORANGE, lw=1.1)
    ax.add_patch(Wedge((0, 0), 1.0, 60, 120, color=GREY, alpha=0.25, lw=0))
    ax.add_patch(Wedge((0, 0), 1.0, 240, 300, color=GREY, alpha=0.25, lw=0))
    ax.text(0, 0.62, "missing\nwedge", ha="center", va="center", fontsize=23)
    ax.set_aspect("equal"); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.axis("off")
    ax.set_title("61 spokes, ±60°", fontsize=24)
    c = (N - 1) / 2
    show(axs[1], lemon, 0, lemon.max())
    axs[1].add_patch(Circle((c + 10, c - 6), 30, fill=False, ec="#56B4E9", lw=2, ls="--"))
    axs[1].set_title("disc → lemon", fontsize=24)
    for a in axs[1:2]:
        a.annotate("", xy=(0.04, 0.95), xytext=(0.04, 0.72), xycoords="axes fraction",
                   arrowprops=dict(arrowstyle="-|>", color="white", lw=2))
        a.text(0.08, 0.83, "beam", color="white", transform=a.transAxes, fontsize=22, va="center")
    save(fig, UNIT, "fig-lemon.png", dpi=110)
    plt.rcParams.update(NORMAL)


if __name__ == "__main__":
    fig_pointramp()
    fig_ramp()
    fig_boxfilter()
    fig_streaks()
    fig_lemon()
