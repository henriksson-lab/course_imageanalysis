"""Static figures for the backprojection/ deck (slide pictures).

Run from the repository root:  python3 -I tools/figs/backprojection/figs.py
Writes backprojection/fig-byhand.svg, fig-halo.svg, fig-slice.png, fig-spokes.svg.
The halo profile uses the demos' own projector/back-projector (tools/figs/radon/compute.ts).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "radon"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from common import FONT, BLUE, ORANGE, GREEN, RED, PURPLE, GREY, save, load, N

UNIT = "backprojection"
PTS = [("A", 2, 0, 1, BLUE), ("B", 0, 1, 2, ORANGE), ("C", -1, -1, 1, GREEN)]


# 1. By hand: three points, two angles, 5 x 5 grid ---------------------------------------
def fig_byhand():
    xs = np.arange(-2, 3)
    ys = np.arange(2, -3, -1)
    p0 = {x: sum(m for _, px, py, m, _ in PTS if px == x) for x in xs}
    p90 = {y: sum(m for _, px, py, m, _ in PTS if py == y) for y in ys}
    B = np.array([[p0[x] + p90[y] for x in xs] for y in ys])
    fig = plt.figure(figsize=(6.4, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 3.2], height_ratios=[1, 3.2], wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 1]); at = fig.add_subplot(gs[0, 1]); al = fig.add_subplot(gs[1, 0])
    ax.imshow(B, cmap="Greys", vmin=-0.5, vmax=6, extent=[-2.5, 2.5, -2.5, 2.5])
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            ax.text(x, y, str(B[i, j]), ha="center", va="center", fontsize=FONT + 3,
                    color="white" if B[i, j] >= 3 else "black", fontweight="bold" if B[i, j] >= 3 else "normal")
    for name, x, y, m, c in PTS:
        ax.add_patch(Circle((x, y), 0.42, fill=False, ec=c, lw=3))
        ax.text(x + 0.3, y + 0.28, name, color=c, fontsize=FONT - 1, fontweight="bold")
    for x, y in [(0, 0), (-1, 1), (2, 1), (0, -1)]:
        ax.add_patch(Rectangle((x - 0.42, y - 0.42), 0.84, 0.84, fill=False, ec=RED, lw=2, ls="--"))
    ax.set_xticks(xs); ax.set_yticks([]); ax.set_xlabel("x")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    at.bar(xs, [p0[x] for x in xs], width=0.5, color=GREY)
    at.set_xlim(-2.5, 2.5); at.set_ylim(0, 2.6); at.set_xticks([]); at.set_yticks([])
    at.spines["left"].set_visible(False)
    for x in xs:
        at.text(x, p0[x] + 0.1, str(p0[x]), ha="center", va="bottom", fontsize=FONT - 1)
    at.text(-2.45, 2.3, "$p_{0°}$", fontsize=FONT + 1, va="top")
    al.barh(ys, [p90[y] for y in ys], height=0.5, color=GREY)
    al.set_ylim(-2.5, 2.5); al.set_xlim(2.8, 0); al.set_yticks(ys); al.set_xticks([])
    al.spines["bottom"].set_visible(False); al.spines["right"].set_visible(True); al.spines["left"].set_visible(False)
    al.yaxis.tick_right(); al.tick_params(axis="y", length=0, labelright=False)
    for y in ys:
        al.text(p90[y] + 0.12, y, str(p90[y]), ha="right", va="center", fontsize=FONT - 1)
    al.text(2.7, 2.35, "$p_{90°}$", fontsize=FONT + 1, va="top")
    al.set_ylabel("y")
    al.yaxis.set_label_position("left")
    fig.text(0.13, 0.02, "○ true points    ⬚ ghosts", fontsize=FONT - 2, color="#333333")
    save(fig, UNIT, "fig-byhand.svg")


# 2. Why it blurs: lines through a point, and the 1/r profile -----------------------------
def fig_halo():
    plt.rcParams.update({"font.size": 23, "xtick.labelsize": 21, "ytick.labelsize": 21, "axes.labelsize": 23, "legend.fontsize": 21})
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.0, 3.9), gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.32})
    nl = 12
    for k in range(nl):
        th = np.pi * k / nl
        d = np.array([np.cos(th), np.sin(th)])
        a0.plot([-3.2 * d[0], 3.2 * d[0]], [-3.2 * d[1], 3.2 * d[1]], color=ORANGE, lw=1.4)
    for r, c in [(1, BLUE), (2.6, PURPLE)]:
        a0.add_patch(Circle((0, 0), r, fill=False, ec=c, lw=2))
        for k in range(2 * nl):
            th = np.pi * k / nl
            a0.plot(r * np.cos(th), r * np.sin(th), "o", color=c, ms=5)
    a0.plot(0, 0, "o", color="black", ms=7)
    a0.set_aspect("equal"); a0.set_xlim(-3.3, 3.3); a0.set_ylim(-3.3, 3.3); a0.axis("off")
    # radial profile of the back-projected centre point (180 angles), demo projector
    img = load("pt_plain_180")
    c = (N - 1) / 2
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.hypot(xx - c, yy - c)
    rb = np.arange(1, 40)
    prof = np.array([img[(r >= q - 0.5) & (r < q + 0.5)].mean() for q in rb])
    ref = prof[9] * 10 / rb   # 1/r through r = 10
    a1.loglog(rb, prof, "o", color=BLUE, ms=5, label="back-projected point")
    a1.loglog(rb, ref, color=RED, lw=2, ls="--", label="∝ 1/r")
    a1.set_xlabel("distance r (pixels)")
    a1.set_ylabel("brightness")
    a1.text(12, ref[11] * 1.6, "∝ 1/r", color=RED, fontsize=23, ha="left", va="bottom")
    a1.text(1.05, prof[-1] * 1.1, "back-projected\npoint (dots)", color=BLUE, fontsize=21, ha="left", va="bottom")
    a1.set_xticks([1, 2, 5, 10, 20]); a1.set_xticklabels(["1", "2", "5", "10", "20"])
    save(fig, UNIT, "fig-halo.svg")
    plt.rcParams.update({"font.size": FONT, "xtick.labelsize": FONT - 2, "ytick.labelsize": FONT - 2, "axes.labelsize": FONT,
                         "legend.fontsize": FONT - 2})
    print("  halo ratio prof(5)/prof(20)", prof[4] / prof[19])


# 3. Fourier slice by hand: stripes, their 2D spectrum, the two projection spectra --------
def fig_slice():
    plt.rcParams.update({"font.size": 21, "xtick.labelsize": 19, "ytick.labelsize": 19, "axes.labelsize": 21, "legend.fontsize": 19})
    n = 64
    x = np.arange(n)
    I = 0.5 + 0.5 * np.cos(2 * np.pi * x / 8)[None, :] * np.ones((n, 1))
    F = np.fft.fftshift(np.fft.fft2(I))
    p0 = I.sum(0); p90 = I.sum(1)
    P0 = np.fft.fftshift(np.fft.fft(p0)); P90 = np.fft.fftshift(np.fft.fft(p90))
    k = np.arange(-n // 2, n // 2)
    print("  2D DFT centre", abs(F[32, 32]), "at +8", abs(F[32, 40]), "P0 dc", abs(P0[32]), "P0 8", abs(P0[40]), "P90 dc", abs(P90[32]))
    fig = plt.figure(figsize=(7.0, 7.4))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.45, height_ratios=[1, 1])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    axs[0].imshow(I, cmap="gray", vmin=0, vmax=1, extent=[0, n, n, 0], interpolation="nearest")
    axs[0].set_xticks([0, 64]); axs[0].set_yticks([0, 64])
    axs[0].set_xlabel("x"); axs[0].set_ylabel("y")
    A = np.log10(1 + np.abs(F))
    axs[1].imshow(A, cmap="gray_r", extent=[-32.5, 31.5, -32.5, 31.5], origin="lower", interpolation="nearest")
    axs[1].axhline(0, color=RED, lw=2, alpha=0.8)
    axs[1].axvline(0, color=BLUE, lw=2, ls="--", alpha=0.8)
    for (u, v), t in [((0, 0), "2048"), ((8, 0), "1024"), ((-8, 0), "1024")]:
        axs[1].add_patch(Circle((u, v), 1.6, fill=False, ec="black", lw=1.5))
    axs[1].text(8, 2.2, "1024", ha="center", fontsize=17)
    axs[1].text(-8, 2.2, "1024", ha="center", fontsize=17)
    axs[1].text(1.0, -2.2, "2048", ha="left", fontsize=18, va="top")
    axs[1].set_xlim(-12, 12); axs[1].set_ylim(-12, 12)
    axs[1].set_xlabel("u"); axs[1].set_ylabel("v")
    a0, a9 = np.abs(P0), np.abs(P90)
    s0 = a0 > 1e-6; s9 = a9 > 1e-6
    axs[2].vlines(k[s0], 0, a0[s0], color=RED, lw=2.5)
    axs[2].plot(k[s0], a0[s0], "o", color=RED, ms=9, label="projection at 0°")
    axs[2].plot(k[s9], a9[s9], "s", color=BLUE, ms=15, mfc="none", mew=2.2, label="projection at 90°")
    axs[2].axhline(0, color="black", lw=0.8)
    axs[2].set_xlim(-12, 12); axs[2].set_ylim(0, 2400)
    axs[2].set_xlabel("frequency (cycles per 64 pixels)")
    axs[2].set_ylabel("|1D DFT|")
    axs[2].set_yticks([0, 1024, 2048])
    axs[2].text(8.6, 1150, "0°", color=RED, fontsize=19)
    axs[2].text(-1.0, 2200, "0° and 90°", color="black", fontsize=19, ha="right")
    save(fig, UNIT, "fig-slice.png", dpi=110)
    plt.rcParams.update({"font.size": FONT, "xtick.labelsize": FONT - 2, "ytick.labelsize": FONT - 2, "axes.labelsize": FONT,
                         "legend.fontsize": FONT - 2})


# 4. Direct Fourier reconstruction: spokes vs the Cartesian grid ---------------------------
def fig_spokes():
    R = 64
    nsp = 90
    fig, az = plt.subplots(figsize=(5.8, 5.6))
    ax = az.inset_axes([0.62, 0.02, 0.36, 0.36])
    for k in range(nsp):
        th = np.pi * k / nsp
        if th > np.pi / 2 + 1e-9:
            continue
        ax.plot([0, R * np.cos(th)], [0, R * np.sin(th)], color=ORANGE, lw=0.7)
    # zoom window at the outer ring around 40 degrees
    cx, cy, w = R * np.cos(np.deg2rad(40)), R * np.sin(np.deg2rad(40)), 5
    x0, y0 = np.floor(cx - w), np.floor(cy - w)
    ax.add_patch(Rectangle((x0, y0), 2 * w, 2 * w, fill=False, ec=RED, lw=2))
    ax.set_aspect("equal"); ax.set_xlim(-1, R + 2); ax.set_ylim(-1, R + 2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.patch.set_facecolor("white"); ax.patch.set_alpha(1); ax.set_zorder(5)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(1)
    gx = np.arange(x0, x0 + 2 * w + 1)
    GX, GY = np.meshgrid(gx, np.arange(y0, y0 + 2 * w + 1))
    az.plot(GX, GY, "s", color=GREY, ms=7, mfc="none", mew=1.4)
    for k in range(nsp):
        th = np.pi * k / nsp
        r = np.arange(0, R + 1)
        px, py = r * np.cos(th), r * np.sin(th)
        az.plot(px, py, color=ORANGE, lw=1)
        az.plot(px, py, "o", color=ORANGE, ms=6)
    az.add_patch(Circle((0, 0), R, fill=False, ec=BLUE, lw=1.4, ls="--"))
    th1, th2 = np.deg2rad(40), np.deg2rad(42)
    gap = R * (th2 - th1)
    print("  outer gap", gap)
    p1 = R * np.array([np.cos(th1), np.sin(th1)]); p2 = R * np.array([np.cos(th2), np.sin(th2)])
    az.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
    az.text(*(p1 + p2) / 2 + np.array([0.3, 0.6]), f"{gap:.1f}", color=RED, fontsize=FONT + 1, fontweight="bold")
    az.set_xlim(x0 - 0.5, x0 + 2 * w + 0.5); az.set_ylim(y0 - 0.5, y0 + 2 * w + 0.5)
    az.set_aspect("equal")
    az.set_ylabel("v")
    for sp in az.spines.values():
        sp.set_visible(True); sp.set_color(RED); sp.set_linewidth(2)
    az.set_xlabel("u (frequency steps)")
    save(fig, UNIT, "fig-spokes.svg")


if __name__ == "__main__":
    fig_byhand()
    fig_halo()
    fig_slice()
    fig_spokes()
