# Static figures for rasterise/lec.qmd. Run from the repo root:
#   node_modules/.bin/esbuild tools/figs/rasterise/render.ts --bundle --platform=node --outfile=$TMP/r.js
#   node $TMP/r.js $TMP/data && python3 -I tools/figs/rasterise/figs.py $TMP/data
# The renders (render.ts) use the course's software rasteriser (demos/src/lib/raster.ts).
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
D = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join(R, "rasterise")
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans"})
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#888888"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", dpi=110)
    plt.close(fig)


def load(name):
    d = json.load(open(os.path.join(D, name + ".json")))
    return np.array(d["rows"], dtype=float).reshape(d["h"], d["w"], 3)


def E(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


a, b, c = (1, 1), (7, 2), (3, 6)
p = (4.5, 3.5)
Es = (E(b, c, p), E(c, a, p), E(a, b, p))
A2 = E(a, b, c)
print("E_bc, E_ca, E_ab at p:", Es, "2*area:", A2, "bary:", [round(e / A2, 3) for e in Es])


def grid(ax, W, H):
    for x in range(W + 1):
        ax.plot([x, x], [0, H], color="#ddd", lw=1, zorder=0)
    for y in range(H + 1):
        ax.plot([0, W], [y, y], color="#ddd", lw=1, zorder=0)
    ax.set_xlim(-0.2, W + 0.2); ax.set_ylim(-0.2, H + 0.2); ax.set_aspect("equal"); ax.axis("off")


# ---------------------------------------------------------------- 1. edge functions: which pixels
fig, ax = plt.subplots(figsize=(6.4, 5.6))
W, H = 8, 7
grid(ax, W, H)
# half-plane left of a->b
ax.add_patch(Polygon([(-0.2, 1 - 1.2 / 6), (8.2, 1 + 7.2 / 6), (8.2, 7.2), (-0.2, 7.2)], closed=True, fc="#e8f1fb", ec="none", zorder=0.5))
ins = 0
for i in range(W):
    for j in range(H):
        q = (i + 0.5, j + 0.5)
        inside = E(a, b, q) > 0 and E(b, c, q) > 0 and E(c, a, q) > 0
        ins += inside
        if inside:
            ax.add_patch(Rectangle((i, j), 1, 1, fc="#ffd9a8", ec="none", zorder=0.6))
        ax.plot(*q, "o", ms=6 if inside else 4, color=ORANGE if inside else "#999", zorder=3)
for x in range(W + 1):
    ax.plot([x, x], [0, H], color="#ccc", lw=1, zorder=0.7)
for y in range(H + 1):
    ax.plot([0, W], [y, y], color="#ccc", lw=1, zorder=0.7)
ax.add_patch(Polygon([a, b, c], closed=True, fill=False, ec="k", lw=2.5, zorder=2))
ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=3, color=BLUE, mutation_scale=22), zorder=4)
for P, n, off in [(a, "a", (-0.45, -0.35)), (b, "b", (0.15, -0.4)), (c, "c", (-0.1, 0.2))]:
    ax.text(P[0] + off[0], P[1] + off[1], n, fontsize=20, weight="bold")
ax.text(6.0, 6.3, "$E_{ab} > 0$\n(left of a→b)", color=BLUE, ha="center", fontsize=16)
ax.text(5.6, 0.35, "$E_{ab} < 0$", color=GREY, ha="center", fontsize=16)
print("pixels inside:", ins)
save(fig, "fig-edgefn.svg")

# ---------------------------------------------------------------- 2. barycentric areas
fig, ax = plt.subplots(figsize=(6.6, 5.4))
cols = {"a": "#cfe3f5", "b": "#ffe0bf", "c": "#d4efd0"}
ax.add_patch(Polygon([b, c, p], closed=True, fc=cols["a"], ec="k", lw=1))
ax.add_patch(Polygon([c, a, p], closed=True, fc=cols["b"], ec="k", lw=1))
ax.add_patch(Polygon([a, b, p], closed=True, fc=cols["c"], ec="k", lw=1))
ax.add_patch(Polygon([a, b, c], closed=True, fill=False, ec="k", lw=2.5))
ax.plot(*p, "ko", ms=8)
ax.text(p[0] - 0.1, p[1] - 0.25, "p", fontsize=20, weight="bold", ha="right", va="top")
for P, n, off in [(a, "a", (-0.4, -0.3)), (b, "b", (0.12, -0.25)), (c, "c", (-0.1, 0.15))]:
    ax.text(P[0] + off[0], P[1] + off[1], n, fontsize=20, weight="bold")


def cen(*pts):
    return np.mean(np.array(pts), axis=0)


for (pts, lab, e, pos) in [((b, c, p), "$b_a$", Es[0], (6.6, 5.4)), ((c, a, p), "$b_b$", Es[1], None), ((a, b, p), "$b_c$", Es[2], None)]:
    q = cen(*pts)
    if pos is None:
        ax.text(q[0], q[1], f"{lab} = {e:g}/{A2:g}\n= {e / A2:.3f}", ha="center", va="center", fontsize=16)
    else:
        ax.annotate(f"{lab} = {e:g}/{A2:g}\n= {e / A2:.3f}", xy=q, xytext=pos, ha="center", va="center", fontsize=16, arrowprops=dict(arrowstyle="-", color="#555"))
ax.set_xlim(0.3, 7.9); ax.set_ylim(0.4, 6.6); ax.set_aspect("equal"); ax.axis("off")
save(fig, "fig-bary.svg")

# ---------------------------------------------------------------- 3. shared edge + back face
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axs[0]
W, H = 7, 5
grid(ax, W, H)
t1 = [(0.5, 0.5), (6.5, 0.5), (0.5, 4.5)]
t2 = [(6.5, 0.5), (6.5, 4.5), (0.5, 4.5)]
ax.add_patch(Polygon(t1, closed=True, fc="#cfe3f5", ec=BLUE, lw=2, alpha=0.8))
ax.add_patch(Polygon(t2, closed=True, fc="#ffe0bf", ec=ORANGE, lw=2, alpha=0.8))
for i in range(W):
    for j in range(H):
        q = (i + 0.5, j + 0.5)
        e = E(t2[2], t2[0], q)  # diagonal from (0.5,4.5) to (6.5,0.5)
        if abs(e) < 1e-9 and 0.5 < q[0] < 6.5:
            ax.plot(*q, "s", ms=11, color=PURPLE, zorder=4)
        else:
            ax.plot(*q, "o", ms=4, color="#777", zorder=3)
ax.text(3.5, -0.75, "a centre on the diagonal (purple)\nis drawn by exactly one triangle", ha="center", va="top", fontsize=15)
ax.set_ylim(-1.9, 5.2)
ax = axs[1]
ax.set_xlim(-0.3, 7.3); ax.set_ylim(-1.9, 5.2); ax.set_aspect("equal"); ax.axis("off")
F = [(0.3, 0.8), (3.0, 0.8), (1.4, 4.0)]
B = [(4.2, 0.8), (5.6, 4.0), (6.9, 0.8)]
for T, fc, ec, lab, sgn in [(F, "#d4efd0", GREEN, "front: a→b→c\ncounter-clockwise\narea > 0, drawn", 1), (B, "#eeeeee", GREY, "back: a→b→c\nclockwise\narea < 0, culled", -1)]:
    ax.add_patch(Polygon(T, closed=True, fc=fc, ec=ec, lw=2.5))
    for k, n in enumerate("abc"):
        P = T[k]; q = cen(*T)
        d = np.array(P) - q; d = d / np.linalg.norm(d) * 0.35
        ax.text(P[0] + d[0], P[1] + d[1], n, ha="center", va="center", fontsize=17, weight="bold")
    q = cen(*T)
    ax.add_patch(FancyArrowPatch((q[0] + 0.45, q[1] - 0.15), (q[0] - 0.45, q[1] - 0.15), connectionstyle=f"arc3,rad={0.9 * sgn}", arrowstyle="-|>", mutation_scale=18, lw=2, color=ec))
    ax.text(q[0], -0.25, lab, ha="center", va="top", fontsize=14)
save(fig, "fig-sharedcull.svg")

# ---------------------------------------------------------------- 4. z-buffer: colour and depth
col, dep = load("intersect-colour"), load("intersect-depth")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
for ax, im, t in [(axs[0], col, "colour buffer"), (axs[1], dep, "depth (dark = near)")]:
    ax.imshow(np.clip(im, 0, 1), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(t, fontsize=24)
    for s in ax.spines.values():
        s.set_color("#999")
fig.tight_layout()
save(fig, "fig-zbuffer.svg") if False else None
fig.savefig(os.path.join(OUT, "fig-zbuffer.png"), bbox_inches="tight", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- 5. depth precision
f = 100.0
d = np.linspace(0.01, 100, 4000)
fig, ax = plt.subplots(figsize=(6.6, 4.6))
for n, colr in [(0.01, ORANGE), (1.0, BLUE)]:
    A, Bc = (f + n) / (f - n), 2 * f * n / (f - n)
    z = A - Bc / d
    m = z >= -1
    ax.plot(d[m], z[m], color=colr, lw=3, label=f"near n = {n:g}")
    half = Bc / A  # z = 0
    print(f"n={n}: z_ndc=0 at d={half:.4f}; z(50)={A - Bc / 50:.7f} z(51)={A - Bc / 51:.7f} diff steps 16bit={((Bc / 50 - Bc / 51)) / (2 / 65535):.2f}")
ax.axhline(0, color="#bbb", lw=1)
ax.set_xlabel("distance from the camera d")
ax.set_ylabel("stored depth $z_{ndc}$")
ax.set_xlim(0, 100); ax.set_ylim(-1.05, 1.05)
ax.legend(loc="lower right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.annotate("d = 50 … 51:\nalmost the same depth", xy=(50.5, 0.99), xytext=(35, 0.35), fontsize=15, arrowprops=dict(arrowstyle="->", color="#444"))
save(fig, "fig-depthprec.svg")

# ---------------------------------------------------------------- 6. affine vs perspective-correct floor
fa, fp = load("floor-affine"), load("floor-persp")
fig, axs = plt.subplots(2, 1, figsize=(5.6, 5.6))
for ax, im, t in [(axs[0], fa, "affine (screen weights)"), (axs[1], fp, "perspective-correct")]:
    ax.imshow(np.clip(im, 0, 1)[70:, :], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(t, fontsize=17)
    for s in ax.spines.values():
        s.set_color("#999")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig-perspcorrect.png"), bbox_inches="tight", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- 7. aliasing: 1 sample vs 4 samples
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.3))
W, H = 10, 6
ta, tb, tc = (0.3, 0.6), (9.7, 2.6), (2.0, 5.7)
offs1 = [(0.5, 0.5)]
offs4 = [(0.375, 0.125), (0.875, 0.375), (0.125, 0.625), (0.625, 0.875)]  # rotated grid
for ax, offs, t in [(axs[0], offs1, "1 sample per pixel: all or nothing"), (axs[1], offs4, "4 samples: coverage 0, ¼, ½, ¾, 1")]:
    grid(ax, W, H)
    for i in range(W):
        for j in range(H):
            cov = np.mean([E(ta, tb, (i + ox, j + oy)) > 0 and E(tb, tc, (i + ox, j + oy)) > 0 and E(tc, ta, (i + ox, j + oy)) > 0 for ox, oy in offs])
            if cov > 0:
                g = 1 - 0.75 * cov
                ax.add_patch(Rectangle((i, j), 1, 1, fc=(g, g * 0.95 + 0.03, 1 - 0.25 * cov), ec="none", zorder=0.5))
            for ox, oy in offs:
                ax.plot(i + ox, j + oy, ".", ms=3, color="#666", zorder=2)
    ax.add_patch(Polygon([ta, tb, tc], closed=True, fill=False, ec=RED, lw=2, zorder=3))
    ax.set_title(t, fontsize=16)
save(fig, "fig-aliasing.svg")

# ---------------------------------------------------------------- 8. shading vectors
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.set_xlim(-2.6, 2.6); ax.set_ylim(-1.6, 2.5); ax.set_aspect("equal"); ax.axis("off")
ax.fill_between([-2.4, 2.4], -0.5, 0, color="#e6e6e6")
ax.plot([-2.4, 2.4], [0, 0], color="k", lw=2)
O = np.array([0, 0])


def arrow(v, colr, lab, off=(0, 0), ls="-"):
    ax.annotate("", xy=O + 2 * np.array(v), xytext=O, arrowprops=dict(arrowstyle="-|>", lw=3, color=colr, mutation_scale=22, linestyle=ls))
    ax.text(*(O + 2.12 * np.array(v) + np.array(off)), lab, color=colr, fontsize=20, ha="center", va="center", weight="bold")


th_l, th_v = np.radians(-50), np.radians(25)  # angle from the normal; l to the left
l = np.array([np.sin(th_l), np.cos(th_l)]); v = np.array([np.sin(th_v), np.cos(th_v)])
n = np.array([0, 1]); r = 2 * (n @ l) * n - l; h = (l + v) / np.linalg.norm(l + v)
arrow(n, "k", "n", (0, 0.05))
arrow(l, ORANGE, "l")
arrow(v, BLUE, "v", (0.12, 0))
arrow(r, PURPLE, "r", (0.05, 0.1), ls="--")
arrow(h, GREEN, "h", (0.15, 0.05))
ax.text(0, -0.65, "l: to the light, v: to the eye,\nr: mirror of l, h: halfway between l and v", fontsize=15, va="top", ha="center", color="#333")
save(fig, "fig-vectors.svg")

# ---------------------------------------------------------------- 9. flat / Gouraud / per pixel
ims = [load("shade-flat"), load("shade-gouraud"), load("shade-pixel")]
fig, axs = plt.subplots(1, 3, figsize=(12, 4.4))
for ax, im, t in zip(axs, ims, ["flat", "Gouraud", "per pixel"]):
    ax.imshow(np.clip(im[10:350, 10:350], 0, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(t, fontsize=18)
    for s in ax.spines.values():
        s.set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig-where.png"), bbox_inches="tight", dpi=100)
plt.close(fig)
