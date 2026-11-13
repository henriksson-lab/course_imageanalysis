"""Static figures for the radon/ deck (slide pictures).

Run from the repository root:  python3 -I tools/figs/radon/figs.py
Writes radon/fig-beer.svg, fig-projection.svg, fig-threepoints.svg, fig-shift.png,
fig-joseph.svg, fig-fivebinary.svg. Conventions as demos/src/lib/radon.ts (origin at the
centre, y up, line x cos(theta) + y sin(theta) = rho).
"""
import os
import sys
import itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc, FancyArrowPatch
from common import FONT, BLUE, ORANGE, GREEN, RED, PURPLE, GREY, save

UNIT = "radon"


# 1. Beer-Lambert: fractions multiply, logs add ------------------------------------------
def fig_beer():
    z = np.linspace(0, 20, 401)
    mu_t = np.full_like(z, 0.2)
    mu_b = np.where((z >= 9) & (z < 11), 0.5, 0.2)
    dz = z[1] - z[0]
    L_t = np.concatenate([[0], np.cumsum(mu_t[:-1] * dz)])
    L_b = np.concatenate([[0], np.cumsum(mu_b[:-1] * dz)])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.2, 5.6), sharex=True, gridspec_kw={"hspace": 0.18})
    for ax in (a1, a2):
        ax.axvspan(9, 11, color=GREY, alpha=0.18, lw=0)
    a2.text(10, 0.15, "2 cm\nbone", ha="center", va="bottom", fontsize=FONT - 3, color="#444444")
    a1.plot(z, np.exp(-L_t), color=BLUE, lw=2.6, label="20 cm tissue")
    a1.plot(z, np.exp(-L_b), color=RED, lw=2.6, ls="--", label="18 cm tissue + 2 cm bone")
    a1.set_ylabel("$I / I_0$")
    a1.set_ylim(0, 1.05)
    a1.legend(loc="upper right", frameon=False, fontsize=FONT - 3)
    a1.annotate(f"{np.exp(-4):.3f}", xy=(20, np.exp(-4)), xytext=(16.3, 0.3), color=BLUE, fontsize=FONT - 2,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1))
    a2.plot(z, L_t, color=BLUE, lw=2.6)
    a2.plot(z, L_b, color=RED, lw=2.6, ls="--")
    a2.text(20.2, 4.0, "4.0", color=BLUE, va="center", fontsize=FONT - 1)
    a2.text(20.2, 4.6, "4.6", color=RED, va="center", fontsize=FONT - 1)
    a2.set_ylabel(r"$-\ln(I / I_0)$")
    a2.set_xlabel("depth along the ray (cm)")
    a2.set_xlim(0, 21.5)
    a2.set_ylim(0, 5)
    a2.text(0.5, 4.3, "after the log: a straight sum,\nslope μ", fontsize=FONT - 3, va="top", color="black")
    save(fig, UNIT, "fig-beer.svg")


# 2. A projection: parallel rays at angle theta, profile on the detector -----------------
DISCS = [(-26, 14, 16, 1.0), (22, -18, 22, 0.6)]   # demo object "two discs" (x, y, R, value)


def disc_proj(rho, th):
    n = np.array([np.cos(th), np.sin(th)])
    p = np.zeros_like(rho)
    for x, y, R, v in DISCS:
        d = rho - (x * n[0] + y * n[1])
        p += 2 * v * np.sqrt(np.clip(R * R - d * d, 0, None))
    return p


def fig_projection():
    th = np.deg2rad(30)
    n = np.array([np.cos(th), np.sin(th)])
    t = np.array([-np.sin(th), np.cos(th)])   # ray direction
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.add_patch(Rectangle((-64, -64), 128, 128, fill=False, ec=GREY, lw=1.2))
    for x, y, R, v in DISCS:
        ax.add_patch(Circle((x, y), R, color=str(1 - 0.75 * v), ec="black", lw=1, zorder=2))
    # rays
    for rho in np.arange(-84, 85, 12):
        a, b = rho * n + 92 * t, rho * n - 112 * t
        ax.plot([a[0], b[0]], [a[1], b[1]], color=ORANGE, lw=1, alpha=0.75, zorder=1)
    # one highlighted ray and its rho
    rho0 = 44
    a, b = rho0 * n + 92 * t, rho0 * n - 112 * t
    ax.plot([a[0], b[0]], [a[1], b[1]], color=RED, lw=2.4, zorder=3)
    ax.annotate("", xy=rho0 * n, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2), zorder=4)
    ax.text(*(rho0 * n * 0.5 + 7 * t), "ρ", color=BLUE, fontsize=FONT + 4, ha="center", va="center", zorder=5,
            bbox=dict(fc="white", ec="none", pad=0.5))
    ax.add_patch(Arc((0, 0), 50, 50, theta1=0, theta2=30, color=BLUE, lw=1.6))
    ax.plot([0, 34], [0, 0], color=BLUE, lw=1.2, ls=":")
    ax.text(29 * np.cos(np.deg2rad(13)), 29 * np.sin(np.deg2rad(13)) - 1, "θ", color=BLUE, fontsize=FONT + 3, va="center",
            bbox=dict(fc="white", ec="none", pad=0.5))
    ax.plot(0, 0, "o", color="black", ms=4, zorder=5)
    # detector and profile
    D = 112
    r = np.linspace(-90, 90, 721)
    base = r[:, None] * n + (-D) * t
    p = disc_proj(r, th)
    tip = base - (1.0 * p)[:, None] * t
    ax.plot(base[[0, -1], 0], base[[0, -1], 1], color=BLUE, lw=4, solid_capstyle="butt", zorder=3)
    ax.plot(tip[:, 0], tip[:, 1], color="black", lw=2, zorder=2)
    ax.fill(np.concatenate([base[:, 0], tip[::-1, 0]]), np.concatenate([base[:, 1], tip[::-1, 1]]), color=GREY, alpha=0.25, lw=0)
    ax.text(*(base[-1] + 6 * n + 4 * t), "detector", color=BLUE, fontsize=FONT - 1, rotation=30, ha="left", va="center")
    pk = base[np.argmax(p)] - 1.0 * p.max() * t
    ax.text(pk[0] + 6, pk[1] - 8, r"$p_\theta(\rho)$", fontsize=FONT + 2, ha="left", va="top")
    ax.text(-62, 60, "rays", color=ORANGE, fontsize=FONT, va="top")
    ax.set_aspect("equal")
    ax.set_xlim(-70, 140)
    ax.set_ylim(-185, 70)
    ax.axis("off")
    save(fig, UNIT, "fig-projection.svg")
    print("  projection total", np.trapezoid(p, r), "mass", sum(np.pi * R * R * v for *_, R, v in DISCS))


# 3. Three points by hand ----------------------------------------------------------------
PTS = [("A", 2, 0, 1, BLUE), ("B", 0, 1, 2, ORANGE), ("C", -1, -1, 1, GREEN)]


def fig_threepoints():
    fig, axs = plt.subplots(2, 2, figsize=(7.6, 5.2), gridspec_kw={"hspace": 0.62, "wspace": 0.35})
    ax = axs[0, 0]
    ax.axhline(0, color=GREY, lw=0.8); ax.axvline(0, color=GREY, lw=0.8)
    for name, x, y, m, c in PTS:
        ax.plot(x, y, "o", color=c, ms=8 + 5 * m)
        ax.text(x + 0.25, y + 0.25, name, color=c, fontsize=FONT + 1, fontweight="bold")
    ax.annotate("", xy=(2.1, 2.1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4, ls="--"))
    ax.text(1.0, 2.15, "45°", color=GREY, fontsize=FONT - 2)
    ax.set_xlim(-2.6, 2.9); ax.set_ylim(-2.2, 2.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_xticks([-2, -1, 0, 1, 2]); ax.set_yticks([-2, -1, 0, 1, 2])
    for ax, deg in zip([axs[0, 1], axs[1, 0], axs[1, 1]], [0, 45, 90]):
        th = np.deg2rad(deg)
        for name, x, y, m, c in PTS:
            rho = x * np.cos(th) + y * np.sin(th)
            ax.bar(rho, m, width=0.22, color=c)
            ax.text(rho, m + 0.08, f"{rho:.2f}".rstrip("0").rstrip(".") if abs(rho) > 1e-9 else "0",
                    ha="center", va="bottom", fontsize=FONT - 4, color=c)
        ax.set_xlim(-2.4, 2.4); ax.set_ylim(0, 2.7)
        ax.set_xlabel("ρ")
        ax.set_yticks([0, 1, 2])
        ax.text(0.02, 1.02, f"θ = {deg}°", transform=ax.transAxes, fontsize=FONT, va="bottom")
    save(fig, UNIT, "fig-threepoints.svg")


# 4. Shift and symmetry: sinograms of a centred and a shifted disc over 360 deg ----------
def disc_sino(cx, cy, R, th, rho):
    d = rho[:, None] - (cx * np.cos(th) + cy * np.sin(th))[None, :]
    return 2 * np.sqrt(np.clip(R * R - d * d, 0, None))


def disc_img(cx, cy, R, n=128):
    c = (n - 1) / 2
    x = np.arange(n) - c
    X, Y = np.meshgrid(x, -x)
    return ((X - cx) ** 2 + (Y - cy) ** 2 <= R * R).astype(float)


def fig_shift():
    plt.rcParams.update({"font.size": 21, "xtick.labelsize": 19, "ytick.labelsize": 19, "axes.labelsize": 21})
    th = np.deg2rad(np.arange(0, 360, 0.5))
    rho = np.arange(-63.5, 64, 1.0)
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 5.8), gridspec_kw={"width_ratios": [1, 2.2], "hspace": 0.35, "wspace": 0.32})
    for row, (cx, cy, lab) in enumerate([(0, 0, "centred disc"), (20, 12, "shifted")]):
        a0, a1 = axs[row]
        a0.imshow(disc_img(cx, cy, 30), cmap="gray_r", extent=[-64, 64, -64, 64], vmin=0, vmax=1.6)
        a0.plot(0, 0, "+", color=RED, ms=12, mew=2)
        a0.set_xticks([]); a0.set_yticks([])
        a0.set_xlabel(lab, fontsize=20)
        s = disc_sino(cx, cy, 30, th, rho)
        a1.imshow(s, cmap="gray_r", aspect="auto", extent=[0, 360, -64, 64], origin="lower", vmin=0, vmax=1.6 * s.max())
        a1.axvline(180, color=RED, ls="--", lw=1.6)
        a1.set_xticks([0, 90, 180, 270, 360])
        a1.set_ylabel("ρ")
        if row == 1:
            a1.set_xlabel("θ (degrees)")
        a1.plot(np.rad2deg(th), cx * np.cos(th) + cy * np.sin(th), color=BLUE, lw=1.6, ls=":")
    axs[0, 1].text(184, 50, "180°: ρ flipped", color=RED, fontsize=19, va="center")
    axs[1, 1].text(5, -56, "20 cos θ + 12 sin θ", color=BLUE, fontsize=19, va="center")
    save(fig, UNIT, "fig-shift.png", dpi=110)
    plt.rcParams.update({"font.size": FONT, "xtick.labelsize": FONT - 2, "ytick.labelsize": FONT - 2, "axes.labelsize": FONT})


# 5. On a pixel grid: one sample per column, linear interpolation between pixel centres ----
def fig_joseph():
    nx, ny = 7, 5
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for i in range(nx + 1):
        ax.plot([i - 0.5, i - 0.5], [-0.5, ny - 0.5], color=GREY, lw=0.8)
    for j in range(ny + 1):
        ax.plot([-0.5, nx - 0.5], [j - 0.5, j - 0.5], color=GREY, lw=0.8)
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    ax.plot(X, Y, "o", color=GREY, ms=4)
    # ray y = 0.6 + 0.42 x (shallower than 45 deg): one sample per column
    k, y0 = 0.42, 0.6
    xs = np.array([-0.5, nx - 0.5])
    ax.plot(xs, y0 + k * xs, color=RED, lw=2.4)
    for i in range(nx):
        y = y0 + k * i
        j0 = int(np.floor(y)); f = y - j0
        ax.plot(i, y, "o", color=RED, ms=8, zorder=5)
        if j0 + 1 < ny:
            ax.plot([i, i], [j0, j0 + 1], color=BLUE, lw=2.2, alpha=0.8)
        if i == 4:
            ax.plot([i, i + 0.0], [j0, j0], "o", color=BLUE, ms=9)
            ax.plot([i], [j0 + 1], "o", color=BLUE, ms=9)
            ax.text(i - 0.14, j0 + 1, f"{f:.2f}", color=BLUE, fontsize=FONT - 1, va="center", ha="right")
            ax.text(i + 0.14, j0, f"{1 - f:.2f}", color=BLUE, fontsize=FONT - 1, va="center", ha="left")
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, nx - 0.4); ax.set_ylim(-0.6, ny - 0.4)
    ax.axis("off")
    save(fig, UNIT, "fig-joseph.svg")


# 6. Five binary images with the same row and column sums ---------------------------------
def fig_fivebinary():
    target = (1, 2, 1)
    sols = []
    for bits in itertools.product([0, 1], repeat=9):
        m = np.array(bits).reshape(3, 3)
        if tuple(m.sum(1)) == target and tuple(m.sum(0)) == target:
            sols.append(m)
    print("  binary solutions:", len(sols))
    fig, axs = plt.subplots(1, len(sols), figsize=(2.5 * len(sols), 2.9), gridspec_kw={"wspace": 0.35})
    for ax, m in zip(axs, sols):
        for r in range(3):
            for c in range(3):
                ax.add_patch(Rectangle((c, 2 - r), 1, 1, fc="black" if m[r, c] else "white", ec=GREY, lw=1.2))
        for r in range(3):
            ax.text(3.2, 2.5 - r, str(m[r].sum()), color=BLUE, fontsize=FONT, va="center")
        for c in range(3):
            ax.text(c + 0.5, -0.25, str(m[:, c].sum()), color=ORANGE, fontsize=FONT, ha="center", va="top")
        ax.set_xlim(-0.1, 3.7); ax.set_ylim(-0.9, 3.1)
        ax.set_aspect("equal"); ax.axis("off")
    axs[-1].text(3.7, 3.25, "row sums", color=BLUE, fontsize=FONT - 3, ha="right", va="bottom")
    axs[0].text(-0.1, -0.95, "column sums", color=ORANGE, fontsize=FONT - 3, ha="left", va="top")
    save(fig, UNIT, "fig-fivebinary.svg")


if __name__ == "__main__":
    fig_beer()
    fig_projection()
    fig_threepoints()
    fig_shift()
    fig_joseph()
    fig_fivebinary()
