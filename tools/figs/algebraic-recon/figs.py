"""Static figures for the algebraic-recon/ deck (slide pictures).

Run from the repository root:  python3 -I tools/figs/algebraic-recon/figs.py
Writes algebraic-recon/fig-raygrid.svg, fig-nullspace.svg, fig-artlines.svg, fig-semiconv.svg.
The two-pixel ART paths follow demos/src/artsmall/model.ts (project2, artPath2); the
semi-convergence curves come from the iterrecon demo's solver (tools/figs/radon/compute.ts).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "radon"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from common import FONT, BLUE, ORANGE, GREEN, RED, PURPLE, GREY, save, data

UNIT = "algebraic-recon"
BIG = {"font.size": 21, "xtick.labelsize": 19, "ytick.labelsize": 19, "axes.labelsize": 21, "legend.fontsize": 19}
NORMAL = {"font.size": FONT, "xtick.labelsize": FONT - 2, "ytick.labelsize": FONT - 2, "axes.labelsize": FONT,
          "legend.fontsize": FONT - 2}


# 1. One row of A: the length of the ray inside each pixel --------------------------------
def fig_raygrid():
    n = 6
    p0 = np.array([0.0, 1.3]); d = np.array([np.cos(np.deg2rad(28)), np.sin(np.deg2rad(28))])
    ts = np.linspace(-2, 10, 240001)
    pts = p0[None, :] + ts[:, None] * d[None, :]
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < n) & (pts[:, 1] >= 0) & (pts[:, 1] < n)
    L = np.zeros((n, n))
    dt = ts[1] - ts[0]
    cols = np.floor(pts[inside, 0]).astype(int); rows = np.floor(pts[inside, 1]).astype(int)
    np.add.at(L, (rows, cols), dt)
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    mx = L.max()
    for r in range(n):
        for c in range(n):
            v = L[r, c]
            ax.add_patch(Rectangle((c, r), 1, 1, fc=(0.0, 0.447, 0.698, 0.08 + 0.6 * v / mx) if v > 0.005 else "white",
                                   ec=GREY, lw=1))
            if v > 0.005:
                ax.text(c + 0.5, r + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=FONT - 1, zorder=6,
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))
    a = p0 - 0.3 * d; b = p0 + 7.2 * d
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=22, color=RED, lw=2.6, zorder=5))
    ax.set_xlim(-0.4, n + 0.6); ax.set_ylim(-0.1, n + 0.1)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, UNIT, "fig-raygrid.svg")
    print("  ray length in grid", L.sum(), "pixels hit", (L > 0.005).sum())


# 2. Two images with the same data: the null image ---------------------------------------
def grid2(ax, x0, M, title, col="black", sums=True, diag=False):
    for r in range(2):
        for c in range(2):
            ax.add_patch(Rectangle((x0 + c, 1 - r), 1, 1, fc="white", ec=GREY, lw=1.4))
            v = M[r][c]
            ax.text(x0 + c + 0.5, 1.5 - r, (f"{v:+d}".replace("-", "−")) if col != "black" else str(v), ha="center", va="center",
                    fontsize=30, color=col)
    ax.text(x0 + 1, 2.25, title, ha="center", va="bottom", fontsize=24)
    if sums:
        for r in range(2):
            ax.text(x0 + 2.12, 1.5 - r, str(sum(M[r])), color=BLUE, fontsize=24, va="center")
        for c in range(2):
            ax.text(x0 + c + 0.5, -0.12, str(M[0][c] + M[1][c]), color=ORANGE, fontsize=24, ha="center", va="top")
    if diag:
        ax.plot([x0 + 0.15, x0 + 1.85], [1.85, 0.15], color=PURPLE, lw=1.6, ls="--")
        ax.text(x0 + 1, -0.75, f"a + d = {M[0][0] + M[1][1]}", color=PURPLE, fontsize=22, va="top", ha="center")


def fig_nullspace():
    plt.rcParams.update({"font.size": 24})
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    grid2(ax, 0, [[1, 2], [3, 4]], "image 1", diag=True)
    ax.text(3.35, 1, "+", fontsize=FONT + 10, ha="center", va="center")
    grid2(ax, 4.0, [[1, -1], [-1, 1]], "null image", col=RED, sums=False)
    ax.text(6.65, 1, "=", fontsize=FONT + 10, ha="center", va="center")
    grid2(ax, 7.3, [[2, 1], [2, 5]], "image 2", diag=True)
    ax.text(5.0, -0.12, "rows, columns\nsum to 0", color=RED, fontsize=22, va="top", ha="center")
    ax.set_xlim(-0.2, 9.9); ax.set_ylim(-1.45, 2.9)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, UNIT, "fig-nullspace.svg")
    plt.rcParams.update(NORMAL)


# 3. Two pixels: ART as alternating projections onto lines ---------------------------------
def project2(a, b, x, om=1.0):
    a = np.asarray(a, float)
    r = (b - a @ x) / (a @ a)
    return x + om * r * a


def art_path(lines, sweeps, om=1.0):
    x = np.zeros(2); P = [x]
    for _ in range(sweeps):
        for a, b in lines:
            x = project2(a, b, x, om); P.append(x)
    return np.array(P)


def demo_line(px, py, deg):
    t = np.deg2rad(deg)
    a = np.array([-np.sin(t), np.cos(t)])
    return a, a @ np.array([px, py])


def lsq(lines):
    A = np.array([a for a, _ in lines]); b = np.array([b for _, b in lines])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def draw_lines(ax, lines, lim):
    xs = np.linspace(lim[0], lim[1], 2)
    for a, b in lines:
        if abs(a[1]) > 1e-9:
            ax.plot(xs, (b - a[0] * xs) / a[1], color=BLUE, lw=2)
        else:
            ax.axvline(b / a[0], color=BLUE, lw=2)


def fig_artlines():
    plt.rcParams.update({"font.size": 23, "xtick.labelsize": 20, "ytick.labelsize": 20, "axes.labelsize": 23})
    fig, axs2 = plt.subplots(2, 2, figsize=(9.0, 9.6), gridspec_kw={"wspace": 0.3, "hspace": 0.38})
    axs = axs2.ravel()
    # perpendicular
    L1 = [((1, 1), 5), ((1, -1), -1)]
    P = art_path(L1, 1)
    print("  perpendicular path", P.round(3).tolist())
    # nearly parallel
    L2 = [((1, 1), 5), ((1, 1.2), 5.6)]
    P2 = art_path(L2, 30)
    sol2 = np.linalg.solve(np.array([a for a, _ in L2], float), np.array([b for _, b in L2], float))
    dist = np.linalg.norm(P2[-1] - sol2)
    ang = np.rad2deg(np.arctan2(1.2, 1) - np.arctan2(1, 1))
    print("  nearly parallel: solution", sol2, "after 30 sweeps distance", dist, "angle", ang)
    # noisy (demo preset "noisy")
    L3 = [demo_line(2, 3.8, 0), demo_line(2.9, 3, 75), demo_line(1.2, 3, 130)]
    P3 = art_path(L3, 30, 1.0)
    P3r = art_path(L3, 60, 0.3)
    star = lsq(L3)
    print("  noisy: lsq", star, "omega .3 end dist", np.linalg.norm(P3r[-1] - star))
    for ax, lines, path, lim, ttl in [(axs[0], L1, P, (-0.3, 5), "perpendicular"),
                                      (axs[1], L2, P2, (-0.3, 5), "nearly parallel"),
                                      (axs[2], L3, None, (0.8, 4.2), "noisy, ω = 1"),
                                      (axs[3], L3, None, (0.8, 4.2), "noisy, ω = 0.3")]:
        draw_lines(ax, lines, lim)
        if path is not None:
            ax.plot(path[:, 0], path[:, 1], "-o", color=ORANGE, lw=1.8, ms=4)
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.set_title(ttl, fontsize=22)
        ax.plot(0, 0, "s", color="black", ms=7)
        if lim[1] - lim[0] < 4:
            ax.set_xticks([1, 2, 3, 4]); ax.set_yticks([1, 2, 3, 4])
    axs[1].plot(*sol2, "*", color=RED, ms=18, zorder=5)
    axs[1].annotate(f"30 sweeps:\nstill {dist:.2f} away", xy=P2[-1], xytext=(0.0, 0.25), fontsize=21,
                    arrowprops=dict(arrowstyle="-", color="black", lw=1))
    axs[0].plot(2, 3, "*", color=RED, ms=18, zorder=5)
    tail = P3[-12:]
    axs[2].plot(P3[:, 0], P3[:, 1], "-", color=ORANGE, lw=1, alpha=0.6)
    axs[2].plot(tail[:, 0], tail[:, 1], "-o", color=ORANGE, lw=2.4, ms=5)
    axs[3].plot(P3r[:, 0], P3r[:, 1], "-o", color=GREEN, lw=2, ms=3)
    for a in axs[2:]:
        a.plot(*star, "*", color=RED, ms=18, zorder=5)
    save(fig, UNIT, "fig-artlines.svg")
    plt.rcParams.update(NORMAL)


# 4. Semi-convergence (iterrecon preset "noisy: stop early") -------------------------------
def fig_semiconv():
    plt.rcParams.update(BIG)
    d = data()
    e = np.array(d["semiErr"]); r = np.array(d["semiRes"]); k = np.arange(len(e))
    kb = int(d["semiBestK"])
    print("  best", kb, e[kb], "at 30", e[30], "fbp", d["semiFbp"])
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.semilogy(k[1:], e[1:], "-o", color=RED, lw=2.4, ms=4)
    ax.semilogy(k[1:], r[1:], "-o", color=BLUE, lw=2.4, ms=4)
    ax.axhline(d["semiFbp"], color=RED, lw=1.6, ls="--")
    ax.plot(kb, e[kb], "o", ms=14, mfc="none", mec="black", mew=2)
    ax.text(kb + 0.6, e[kb] * 0.74, f"lowest error\n{e[kb]:.3f} at sweep {kb}", fontsize=18, va="top")
    ax.text(30, e[30] * 1.08, f"{e[30]:.3f}", color=RED, fontsize=18, ha="right", va="bottom")
    ax.text(30, d["semiFbp"] * 1.08, f"FBP (Hann) {d['semiFbp']:.3f}", color=RED, fontsize=18, ha="right", va="bottom")
    ax.text(16, r[16] * 1.25, "data residual", color=BLUE, fontsize=19)
    ax.text(15, e[15] * 0.9, "error vs truth (RMSE)", color=RED, fontsize=19, ha="center", va="top")
    ax.set_yticks([0.01, 0.02, 0.04, 0.06]); ax.set_yticklabels(["0.01", "0.02", "0.04", "0.06"])
    ax.minorticks_off()
    ax.set_xlabel("ART sweep")
    ax.set_xlim(0, 31)
    save(fig, UNIT, "fig-semiconv.svg")
    plt.rcParams.update(NORMAL)


if __name__ == "__main__":
    fig_raygrid()
    fig_nullspace()
    fig_artlines()
    fig_semiconv()
