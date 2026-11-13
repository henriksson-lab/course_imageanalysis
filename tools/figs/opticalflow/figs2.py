"""Second batch of slide figures for the optical flow day (pictures next to the text).

Run from the repository root, after compute.ts (see its header) has written flow.json:
    python3 -I tools/figs/opticalflow/figs2.py <dir with flow.json>
Writes SVG/PNG figures into flow-constraint/, lucas-kanade/, horn-schunck/, nonrigid-reg/.
Maps and error numbers come from compute.ts, which runs the demos' own code (lib/flow.ts,
lib/deform.ts) with the demos' scenes and seeds; the diagrams are computed here.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figs import flow_rgb, wheel_inset, derivs, CELL, CREDIT  # noqa: E402

plt.rcParams.update({"font.size": 16, "axes.titlesize": 16, "axes.labelsize": 16, "legend.fontsize": 15,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "figure.dpi": 100,
                     "savefig.facecolor": "white", "axes.facecolor": "white", "svg.fonttype": "path"})
# Okabe–Ito (colour-blind safe)
BLUE, ORANGE, GREEN, VERM, SKY, PURPLE, YELLOW, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#F0E442", "#666666"
D = json.load(open(os.path.join(sys.argv[1], "flow.json")))
N = 128


def A(key):
    return np.array(key, dtype=float).reshape(N, N)


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", path)


def arrow(ax, p, q, color, lw=2.5, ms=18, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


def vel_axes(ax, lim=(-0.5, 3.2, -0.6, 2.4)):
    """Velocity space as in the demos: u to the right, v downwards (screen convention)."""
    ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[3], lim[2]); ax.set_aspect("equal")
    ax.axhline(0, color="#999", lw=0.8); ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("u (px/frame)"); ax.set_ylabel("v (down)"); ax.grid(alpha=0.3)


# ================================================================ flow-constraint
def fig_bleach():
    x = np.arange(0, 41)
    f1 = 20 + 100 * np.exp(-(x - 20) ** 2 / (2 * 4 ** 2))
    f2 = 0.9 * f1                      # 10 % bleaching, nothing moves
    ix = np.gradient(f1); it = f2 - f1
    u = np.where(np.abs(ix) > 2, -it / ix, np.nan)
    fig, ax = plt.subplots(2, 1, figsize=(5.2, 5.6), sharex=True)
    ax[0].plot(x, f1, "-", color=BLUE, lw=3, label="frame 1")
    ax[0].plot(x, f2, "--", color=ORANGE, lw=3, label="frame 2: 10 % darker")
    ax[0].set_ylabel("grey value"); ax[0].legend(loc="upper left", fontsize=14); ax[0].set_ylim(0, 215)
    ax[0].set_title("bleaching: the spot does not move")
    ax[1].axhline(0, color=GREEN, lw=3, ls="--", label="true speed 0")
    ax[1].plot(x, u, "o", color=VERM, ms=7)
    ax[1].annotate("seems to move right", (12, u[12]), xytext=(-1, 2.55), fontsize=15, arrowprops=dict(arrowstyle="->", color="k"))
    ax[1].annotate("seems to move left", (28, u[28]), xytext=(13, -2.15), fontsize=15, arrowprops=dict(arrowstyle="->", color="k"))
    ax[1].set_ylim(-2.4, 3.4); ax[1].set_xlabel("position x (pixels)"); ax[1].set_ylabel("−I$_t$ / I$_x$ (px/frame)")
    ax[1].text(40, 0.25, "true speed 0", color=GREEN, fontsize=15, ha="right", va="bottom")
    ax[1].set_title("read as motion: the flanks move inwards")
    fig.tight_layout(); save(fig, "flow-constraint/fig-bleach.svg")


def fig_aperture_hole():
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    R = 3.0
    n = np.array([np.cos(np.radians(30)), -np.sin(np.radians(30))])   # edge normal (screen: x right, y up here)
    t = np.array([-n[1], n[0]])
    th = np.linspace(0, 2 * np.pi, 400)
    # dark side / bright side of the edge inside the hole
    gx, gy = np.meshgrid(np.linspace(-R, R, 300), np.linspace(-R, R, 300))
    side = (gx * n[0] + gy * n[1]) < 0
    img = np.where(side, 0.35, 0.85); img[gx ** 2 + gy ** 2 > R * R] = np.nan
    ax.imshow(img, extent=(-R, R, -R, R), origin="lower", cmap="gray", vmin=0, vmax=1)
    ax.add_patch(Circle((0, 0), R, fill=False, ec="k", lw=2))
    L = np.sqrt(R * R)
    ax.plot([-t[0] * L, t[0] * L], [-t[1] * L, t[1] * L], color="k", lw=2.5)
    d = 1.3
    h = np.sqrt(R * R - d * d)
    ax.plot([-t[0] * h + d * n[0], t[0] * h + d * n[0]], [-t[1] * h + d * n[1], t[1] * h + d * n[1]], color="k", lw=2, ls="--")
    for s, c in [(-1.9, SKY), (0.9, SKY), (2.3, SKY)]:
        q = d * n + s * t
        arrow(ax, (0, 0), tuple(q), c, lw=2.5)
    arrow(ax, (0, 0), tuple(d * n), VERM, lw=4, ms=22)
    ax.annotate("normal\nflow", d * n, xytext=(3.2, -1.6), color=VERM, fontsize=16, fontweight="bold", arrowprops=dict(arrowstyle="->", color=VERM, lw=1.5))
    ax.text(-2.9, -3.55, "solid: edge now; dashed: one frame later", fontsize=15)
    ax.text(-2.9, 3.25, "blue: three true motions,\nall give the same picture", fontsize=15, color="#005a8c", va="bottom")
    ax.set_xlim(-3.3, 4.6); ax.set_ylim(-3.8, 4.1); ax.set_aspect("equal"); ax.axis("off")
    save(fig, "flow-constraint/fig-aperture-hole.svg")


# ================================================================ lucas-kanade
def rect_scene():
    n = 96
    Y, X = np.mgrid[0:n, 0:n].astype(float)
    sm = lambda z: 1 / (1 + np.exp(-z))
    return lambda x, y: 60 + 140 * sm((x - 30) / 1.5) * sm((70 - x) / 1.5) * sm((y - 30) / 1.5) * sm((70 - y) / 1.5), X, Y


def fig_st_ellipses():
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(4)
    Y, X = np.mgrid[-12:13, -12:13].astype(float)
    a = np.radians(30)
    tex = gaussian_filter(rng.standard_normal((25, 25)), 1.5); tex = 128 + 40 * tex / tex.std()
    scenes = {"flat": 100 + 0 * X, "edge": 60 + 140 / (1 + np.exp(-(X * np.cos(a) - Y * np.sin(a)) / 1.5)), "texture": tex}
    spots = [("flat", GREY), ("edge", BLUE), ("texture", VERM)]
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    for name, c in spots:
        I = scenes[name] + 2 * rng.standard_normal(X.shape)
        ix, iy, _ = derivs(I, I, 1.0)
        g = np.array([(ix[12 + dy, 12 + dx], iy[12 + dy, 12 + dx]) for dy in range(-6, 7) for dx in range(-6, 7) if dx * dx + dy * dy <= 36])
        J = g.T @ g / len(g)
        mu, V = np.linalg.eigh(J)
        ax.plot(g[:, 0], g[:, 1], "o", color=c, ms=5, alpha=0.65)
        pos = {"flat": (-31, -25), "edge": (-31, 27), "texture": (9, 27)}[name]
        ax.text(*pos, f"{name}\nμ₁ {mu[1]:.0f}, μ₂ {mu[0]:.0f}", color=c if name != "texture" else "#a04000", fontsize=15, fontweight="bold")
        ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        ax.add_patch(Ellipse((0, 0), 2 * 2 * np.sqrt(mu[1]) + 0.6, 2 * 2 * np.sqrt(mu[0]) + 0.6, angle=ang, fill=False, ec=c, lw=2.5))
    ax.set_xlim(-32, 32); ax.set_ylim(33, -31); ax.set_aspect("equal")
    ax.axhline(0, color="#999", lw=0.8); ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("I$_x$"); ax.set_ylabel("I$_y$ (down)")
    ax.set_title("gradients in three windows\nand the ellipse of J", fontsize=16)
    save(fig, "lucas-kanade/fig-st-ellipses.svg")


def fig_lk_confidence():
    c = D["conf"]
    I1, mu2, u, v = A(c["I1"]), A(c["mu2"]), A(c["u"]), A(c["v"])
    vmax = 1.5
    fig, ax = plt.subplots(2, 2, figsize=(5.0, 5.6))
    ax[0, 0].imshow(I1, cmap="gray", vmin=0, vmax=255); ax[0, 0].set_title("frame 1 (1.5 px)", fontsize=17)
    ax[0, 1].imshow(np.log10(1 + mu2), cmap="viridis", vmin=0, vmax=np.log10(401))
    ax[0, 1].set_title("μ₂ (log; dark = low)", fontsize=17)
    ax[0, 1].plot(85, 114, "o", ms=11, mfc="none", mec=VERM, mew=2.5)
    ax[1, 0].imshow(flow_rgb(u, v, vmax)); ax[1, 0].set_title("LK flow, ρ = 2", fontsize=17)
    rgb = flow_rgb(u, v, vmax); rgb[mu2 < 20] = 0.55
    ax[1, 1].imshow(rgb); ax[1, 1].set_title("grey: μ₂ < 20", fontsize=17)
    for a in ax.ravel():
        a.axis("off")
    w = fig.add_axes([0.0, 0.0, 0.13, 0.13]); wheel_inset(w, 1)
    fig.text(0.15, 0.05, "wheel edge 1.5 px/frame;\nred circle: membrane (85, 114)", fontsize=15)
    fig.text(0.15, 0.0, CREDIT, fontsize=10, color="#555")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.15, hspace=0.18, wspace=0.05)
    save(fig, "lucas-kanade/fig-lk-confidence.png")


def lines_panel(ax, cons, title, sol=None):
    for (a, b, c), col in cons:
        if abs(b) < 1e-9:
            ax.axvline(-c / a, color=col, lw=6, alpha=0.55)
        else:
            ax.axhline(-c / b, color=col, lw=6, alpha=0.55)
    vel_axes(ax)
    ax.set_title(title)


def fig_byhand():
    fig, ax = plt.subplots(2, 1, figsize=(3.4, 5.3))
    lines_panel(ax[0], [((1, 0, -2), BLUE), ((0, 1, -1), ORANGE)], "u = 2 and v = 1")
    ax[0].plot(2, 1, "o", color=VERM, ms=14); ax[0].annotate("(2, 1)", (2, 1), xytext=(0.4, 1.8), fontsize=15, color=VERM)
    lines_panel(ax[1], [((1, 0, -2), BLUE)], "all four: u = 2")
    ax[1].annotate("", (2, -0.45), (2, 2.3), arrowprops=dict(arrowstyle="<->", color=VERM, lw=2.5))
    ax[1].text(1.9, 1.85, "v unknown", ha="right", color=VERM, fontsize=15)
    fig.tight_layout(); save(fig, "lucas-kanade/fig-byhand.svg")


def fig_window_size():
    r = D["rho"]
    rho = np.array([e["rho"] for e in r]); n12 = np.array([e["noise12"] for e in r]); bd = np.array([e["boundary"] for e in r])
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(rho, n12, "o-", color=BLUE, lw=3, ms=8, label="noise 12, one motion")
    ax.plot(rho, bd, "s--", color=VERM, lw=3, ms=8, label="near a motion boundary")
    for k in (1.5, 3, 6):
        i = list(rho).index(k)
        ax.annotate(f"{n12[i]:.2f}", (k, n12[i]), xytext=(4, 8), textcoords="offset points", fontsize=15, color=BLUE)
    ax.set_xlabel("window radius ρ (px)"); ax.set_ylabel("mean endpoint error (px)")
    ax.set_ylim(0, 1.2); ax.set_xlim(0.5, 8.5); ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=14)
    ax.set_title("blue: micrograph, 1 px, noise 12\nred: cell over still substrate, 1.5 px", fontsize=15)
    fig.tight_layout(); save(fig, "lucas-kanade/fig-window-size.svg")


def fig_large_motion():
    s = D["shifts"]
    s = [e for e in s if e["s"] >= 1]
    x = np.array([e["s"] for e in s]); one = np.array([e["one"] for e in s]); pyr = np.array([e["pyr"] for e in s])
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.plot(x, x, ":", color=GREY, lw=2.5, label="guessing zero")
    ax.plot(x, one, "o-", color=VERM, lw=3, ms=8, label="one solve")
    ax.plot(x, pyr, "s-", color=BLUE, lw=3, ms=8, label="pyramid (4 levels × 3)")
    for k in (1, 2, 4, 8):
        i = list(x).index(k)
        lab = {1: f"{one[i]:.2f}", 2: f"{one[i]:.2f}", 4: f"{one[i]:.1f}", 8: f"{one[i]:.2f}"}[k]
        ax.annotate(lab, (k, one[i]), xytext=(-14, -26) if k == 8 else (-30, 6), textcoords="offset points", fontsize=15, color=VERM)
    ax.annotate(f"{pyr[-1]:.2f}", (8, pyr[-1]), xytext=(-14, 10), textcoords="offset points", fontsize=15, color=BLUE)
    ax.set_xlabel("shift of the micrograph (px)"); ax.set_ylabel("mean endpoint error (px)")
    ax.set_xlim(0, 8.6); ax.set_ylim(0, 11.5); ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=15)
    fig.tight_layout(); save(fig, "lucas-kanade/fig-large-motion.svg")


# ================================================================ horn-schunck
def fig_lk_vs_hs():
    r = D["ring"]
    I1 = A(r["I1"]); vmax = 1.5
    lk_rgb = flow_rgb(A(r["lk"]["u"]), A(r["lk"]["v"]), vmax); lk_rgb[A(r["lk"]["mu2"]) < 20] = 0.55
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 3.8))
    ax[0].imshow(I1, cmap="gray", vmin=0, vmax=255); ax[0].set_title("frame 1: flat disc in a\ntextured ring (moves 1.5 px)")
    ax[1].imshow(lk_rgb); ax[1].set_title("Lucas–Kanade: holes\n(grey: μ₂ < 20)")
    ax[2].imshow(flow_rgb(A(r["hs2000"]["u"]), A(r["hs2000"]["v"]), vmax)); ax[2].set_title("Horn–Schunck, λ = 100:\ndense, filled in")
    for a in ax:
        a.axis("off")
    w = fig.add_axes([0.905, 0.12, 0.09, 0.25]); wheel_inset(w, 1); w.set_title("1.5 px", fontsize=15)
    fig.subplots_adjust(left=0.0, right=0.9, wspace=0.06)
    save(fig, "horn-schunck/fig-lk-vs-hs.png")


def fig_hs_update():
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    vel_axes(ax, (-0.4, 3.3, -0.95, 0.9))
    ax.axvline(2, color=BLUE, lw=6, alpha=0.55)
    ax.text(1.95, -0.7, "constraint line u = 2", color="#005a8c", fontsize=15, ha="right")
    ax.plot(0, 0, "o", color=GREY, ms=13); ax.text(-0.35, 0.5, "neighbour\naverage", fontsize=15, color=GREY)
    for lam, col, xy in [(3, PURPLE, (0.1, -0.35)), (1, ORANGE, (0.6, 0.6)), (0, VERM, (2.08, -0.35))]:
        u = 2 / (lam + 1)
        ax.plot(u, 0, "o", color=col, ms=12, zorder=5)
        ax.text(*xy, f"λ = {lam}: ({u:g}, 0)", fontsize=15, color=col)
    arrow(ax, (0.08, 0), (1.9, 0), "k", lw=1.8, ms=14)
    ax.set_title("(I$_x$, I$_y$, I$_t$) = (2, 0, −4): step u = 2 / (λ + 1)", fontsize=15)
    fig.tight_layout(); save(fig, "horn-schunck/fig-hs-update.svg")


def ae(u, v, ug, vg):
    c = (u * ug + v * vg + 1) / (np.sqrt(u * u + v * v + 1) * np.sqrt(ug * ug + vg * vg + 1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def fig_epe_ae():
    fig, ax = plt.subplots(2, 1, figsize=(4.0, 5.5), gridspec_kw={"height_ratios": [1.1, 1]})
    a = ax[0]
    for u, col, lab, x0, ha in [(0.1, ORANGE, "estimate\n(0.1, 0, 1)", 0.0, "right"), (0.6, BLUE, "truth\n(0.6, 0, 1)", 0.7, "left")]:
        arrow(a, (0, 0), (u, 1), col, lw=3)
        a.text(x0, 0.98, lab, color=col, fontsize=15, ha=ha, va="center")
    a.annotate("", (0.1, 1.12), (0.6, 1.12), arrowprops=dict(arrowstyle="<->", color="k", lw=1.5))
    a.text(0.35, 1.18, "EPE 0.5", fontsize=15, ha="center")
    th = np.linspace(np.arctan2(1, 0.6), np.arctan2(1, 0.1), 30)
    a.plot(0.45 * np.cos(th), 0.45 * np.sin(th), color=VERM, lw=2)
    a.text(0.42, 0.42, f"AE {ae(0.1, 0, 0.6, 0):.0f}°", fontsize=15, color=VERM)
    a.set_xlim(-1.0, 1.5); a.set_ylim(0, 1.4); a.set_aspect("equal")
    a.set_xlabel("u (px/frame)"); a.set_ylabel("time (1 frame)"); a.set_title("flows as space–time directions")
    b = ax[1]
    s = np.linspace(0.5, 4, 200)
    b.plot(s, ae(s - 0.5, 0, s, 0), color=VERM, lw=3)
    b.plot(0.6, ae(0.1, 0, 0.6, 0), "o", color=ORANGE, ms=11)
    b.plot(np.hypot(2.5, 1), ae(2, 1, 2.5, 1), "o", color=BLUE, ms=11)
    b.annotate(f"(2, 1) vs (2.5, 1): {ae(2, 1, 2.5, 1):.1f}°", (np.hypot(2.5, 1), ae(2, 1, 2.5, 1)), xytext=(1.25, 13), fontsize=15, color="#005a8c",
               arrowprops=dict(arrowstyle="->", color="#005a8c"))
    b.set_xlabel("true speed (px/frame)"); b.set_ylabel("angular error (°)")
    b.set_title("same EPE 0.5, falling AE"); b.grid(alpha=0.3); b.set_ylim(0, 28)
    fig.tight_layout(); save(fig, "horn-schunck/fig-epe-ae.svg")


def fig_residual():
    h = D["ring"]["hs60"]
    u, v, epe, res = A(h["u"]), A(h["v"]), A(h["epe"]), A(h["resid"])
    fig, ax = plt.subplots(1, 3, figsize=(8.0, 3.7))
    ax[0].imshow(flow_rgb(u, v, 1.5)); ax[0].set_title("HS, 60 sweeps:\ncentre still 0")
    im1 = ax[1].imshow(epe, cmap="inferno", vmin=0, vmax=1.5); ax[1].set_title("endpoint error:\nlarge in the centre")
    im2 = ax[2].imshow(res, cmap="gray", vmin=-40, vmax=40)
    ax[2].set_title(f"residual: small there\n(|r| {h['residCentre']:.1f} vs {h['residMean']:.1f} overall)")
    for a in ax:
        a.axis("off")
    for im, a, lab, tk in [(im1, ax[1], "px", [0, 1.5]), (im2, ax[2], "grey levels", [-40, 40])]:
        cb = fig.colorbar(im, ax=a, orientation="horizontal", fraction=0.05, pad=0.03, ticks=tk); cb.set_label(lab, fontsize=15); cb.ax.tick_params(labelsize=14)
    fig.subplots_adjust(left=0.0, right=1, wspace=0.25)
    save(fig, "horn-schunck/fig-residual.png")


# ================================================================ nonrigid-reg
def b3(t):
    a = np.abs(t)
    return np.where(a < 1, 2 / 3 - a ** 2 + 0.5 * a ** 3, np.where(a < 2, (2 - a) ** 3 / 6, 0))


def fig_folding():
    x = np.linspace(0, 40, 801)
    fig, ax = plt.subplots(2, 1, figsize=(4.3, 5.4), sharex=True)
    for c, col, ls in [(6, BLUE, "-"), (15, ORANGE, "--"), (25, VERM, "-.")]:
        d = c * b3((x - 20) / 10)
        ax[0].plot(x, d, ls, color=col, lw=3, label=f"coefficient {c}")
        y = x + d
        ax[1].plot(x, y, ls, color=col, lw=3)
        if c == 25:
            fold = np.gradient(y, x) < 0
            ax[1].fill_between(x, 0, 70, where=fold, color=VERM, alpha=0.15, lw=0)
            ax[1].text(1, 44, "shaded: folds,\nthe map runs backwards", color=VERM, fontsize=15)
    ax[0].set_ylabel("d(x) (px)"); ax[0].legend(loc="upper left"); ax[0].grid(alpha=0.3)
    ax[0].set_title("one control point (x = 20), knots every 10 px", fontsize=15)
    ax[1].set_ylabel("x + d(x) (px)"); ax[1].set_xlabel("x (px)"); ax[1].grid(alpha=0.3); ax[1].set_ylim(0, 65)
    ax[1].set_title("steepest slope 0.4 at c = 6; folds above c = 15", fontsize=15)
    fig.tight_layout(); save(fig, "nonrigid-reg/fig-folding.svg")


def fig_measures():
    j = D["jointhist"]
    fig, ax = plt.subplots(2, 1, figsize=(4.5, 5.4), sharex=True)
    for a, key, title in [(ax[0], "inverted", "inverted contrast"), (ax[1], "nonmonotone", "non-monotone (a hump)")]:
        c = j[key]; xs = np.array(c["xs"])
        for name, col, ls, lab in [("ss", ORANGE, "--", "SSD (negated)"), ("nc", BLUE, "-.", "NCC"), ("mi", VERM, "-", "MI")]:
            y = np.array(c[name]); y = (y - y.min()) / (y.max() - y.min())
            a.plot(xs, y, ls, color=col, lw=2.8, label=lab)
            k = int(np.argmax(y)); a.plot(xs[k], y[k], "o", color=col, ms=9)
        a.axvline(0, color=GREEN, lw=2, ls=":")
        a.set_title(f"{title}: true shift 0 (green)"); a.set_ylabel("score (scaled)"); a.set_ylim(-0.05, 1.12); a.set_xlim(-84, 84)
        a.grid(alpha=0.3)
    ax[0].annotate("SSD best at −27", (-27, 1.0), xytext=(-80, 0.15), fontsize=15, color="#9a5f00", arrowprops=dict(arrowstyle="->", color="#9a5f00"))
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=15, bbox_to_anchor=(0.5, 1.07), frameon=False)
    ax[1].set_xlabel("x shift of the moving image (px)")
    fig.text(0.01, -0.01, CREDIT, fontsize=11, color="#555")
    fig.tight_layout(); save(fig, "nonrigid-reg/fig-measures.svg")


def fig_entropy():
    cases = [([1], "always the same", 0), ([0.5, 0.5], "two equally likely", 1), ([0.5, 0.25, 0.25], "½, ¼, ¼", 1.5), ([0.25] * 4, "four equally likely", 2)]
    fig, ax = plt.subplots(1, 4, figsize=(9, 2.9), sharey=True)
    for a, (p, title, H) in zip(ax, cases):
        a.bar(range(len(p)), p, color=BLUE, width=0.7)
        a.set_xlim(-0.6, 3.6); a.set_xticks(range(4)); a.set_xticklabels(["a", "b", "c", "d"])
        a.set_title(f"{title}\nH = {H:g} bit" + ("s" if H != 1 else ""), fontsize=15)
    ax[0].set_ylabel("probability"); ax[0].set_ylim(0, 1.05)
    fig.tight_layout(); save(fig, "nonrigid-reg/fig-entropy.svg")


def fig_landscape():
    F = CELL[80:176, 80:176] / 255
    from scipy.ndimage import gaussian_filter
    xs = np.arange(-40, 41)
    fig, ax = plt.subplots(figsize=(5.0, 4.1))
    for sig, col, ls, lab in [(0, VERM, "-", "full resolution"), (4, BLUE, "--", "blurred (σ = 4 px)")]:
        G = gaussian_filter(CELL / 255, sig) if sig else CELL / 255
        Fi = G[80:176, 80:176]
        cost = [np.mean((Fi - G[80:176, 80 + t:176 + t]) ** 2) for t in xs]
        cost = np.array(cost) / max(cost)
        ax.plot(xs, cost, ls, color=col, lw=3, label=lab)
    ax.plot(-22, np.interp(-22, xs, cost) + 0.04, "o", color=GREY, ms=16, zorder=5)
    ax.set_xlabel("x shift of the moving image (px)"); ax.set_ylabel("SSD cost (scaled)")
    ax.set_title("blurred first: a wider, smoother valley", fontsize=15)
    ax.grid(alpha=0.3); ax.legend(loc="lower right"); ax.set_ylim(0, 1.1)
    fig.text(0.01, -0.02, CREDIT, fontsize=11, color="#555")
    fig.tight_layout(); save(fig, "nonrigid-reg/fig-landscape.svg")


if __name__ == "__main__":
    fig_bleach(); fig_aperture_hole()
    fig_st_ellipses(); fig_lk_confidence(); fig_byhand(); fig_window_size(); fig_large_motion()
    fig_lk_vs_hs(); fig_hs_update(); fig_epe_ae(); fig_residual()
    fig_folding(); fig_measures(); fig_entropy(); fig_landscape()
