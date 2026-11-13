# Figures for integral-image/lec.qmd. Run: python3 -I tools/figs/integral-image/figs.py
# from the repository root. Writes integral-image/fig-*.svg.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from matplotlib.patches import Rectangle, Polygon
from style import setup, fig, save, BLUE, ORANGE, GREEN, RED, GREY

setup()
OUT = "integral-image"

# --- A. a range sum on the 8-pixel row -------------------------------------------
f = np.array([3, 1, 4, 1, 5, 9, 2, 6])
S = np.concatenate([[0], np.cumsum(f)])
fg, ax = fig(520, 300)
a, b = 3, 7
cols = [ORANGE if a <= i <= b else "#cccccc" for i in range(1, 9)]
ax.bar(range(1, 9), f, color=cols, edgecolor="#444", width=0.85)
for i, v in enumerate(f, 1):
    ax.text(i, v + 0.2, str(v), ha="center", va="bottom", fontsize=18)
ax.set_xticks(range(1, 9))
ax.set_xlabel("pixel $i$")
ax.set_ylabel("value $f(i)$")
ax.set_ylim(0, 12)
ax.set_yticks([0, 5, 10])
ax.text(5, 11.2, "4 + 1 + 5 + 9 + 2 = 21", ha="center", fontsize=18, color="#b35900")
save(fg, f"{OUT}/fig-rangesum.svg")

# --- B. calculus picture: area under f = rise of F --------------------------------
x = np.linspace(0, 10, 400)
fx = 1.2 + np.sin(x * 0.9) * 0.7 + 0.08 * x
Fx = np.concatenate([[0], np.cumsum((fx[1:] + fx[:-1]) / 2 * np.diff(x))])
xa, xb = 3.0, 7.0
fg, (a1, a2) = fig(520, 470, nrows=2, sharex=True)
a1.plot(x, fx, color=BLUE, lw=3)
m = (x >= xa) & (x <= xb)
a1.fill_between(x[m], 0, fx[m], color=ORANGE, alpha=0.5)
a1.set_ylabel("$f$")
a1.set_ylim(0, 2.6)
a1.set_yticks([])
a1.text(5, 0.5, "area", ha="center", fontsize=18)
a2.plot(x, Fx, color=BLUE, lw=3)
Fa, Fb = np.interp(xa, x, Fx), np.interp(xb, x, Fx)
a2.plot([xa, xb], [Fa, Fa], color=GREY, ls="--", lw=1.5)
a2.annotate("", xy=(xb, Fb), xytext=(xb, Fa), arrowprops=dict(arrowstyle="<->", color="#b35900", lw=2.5))
a2.text(xb + 0.25, (Fa + Fb) / 2, "rise\n$F(b) - F(a)$", va="center", fontsize=18, color="#b35900")
a2.plot([xa, xb], [Fa, Fb], "o", color="#444", ms=7)
a2.set_ylabel("$F$")
a2.set_yticks([])
a2.set_xticks([xa, xb])
a2.set_xticklabels(["$a$", "$b$"])
for ax_ in (a1, a2):
    ax_.axvline(xa, color=GREY, lw=1, ls=":")
    ax_.axvline(xb, color=GREY, lw=1, ls=":")
a2.set_xlabel("$x$")
save(fg, f"{OUT}/fig-antiderivative.svg")

# --- C/D. a small image and its table ----------------------------------------------
I = np.array([[1, 2, 0, 3], [2, 1, 4, 1], [0, 3, 2, 2], [1, 0, 1, 3]])
T = np.zeros((5, 5), int)
T[1:, 1:] = I.cumsum(0).cumsum(1)


def grid(ax, vals, x0, y0, c=1.0, shade=None, zero_pad=False, hl=None):
    n, m = vals.shape
    for r in range(n):
        for k in range(m):
            fc = "white"
            if zero_pad and (r == 0 or k == 0):
                fc = "#e6e6e6"
            if shade is not None and shade[r, k]:
                fc = "#bfe0bf"
            if hl is not None and (r, k) in hl:
                fc = hl[(r, k)]
            ax.add_patch(Rectangle((x0 + k * c, y0 - (r + 1) * c), c, c, fc=fc, ec="#999", lw=1))
            ax.text(x0 + (k + .5) * c, y0 - (r + .5) * c, str(vals[r, k]), ha="center", va="center", fontsize=19)


# C: the region counted by one entry
fg, ax = fig(600, 300)
ax.set_aspect("equal"); ax.axis("off")
shade = np.zeros((4, 4), bool); shade[:3, :3] = True
grid(ax, I, 0, 4, shade=shade)
ax.text(2, 4.3, "image $I$", ha="center", fontsize=18)
grid(ax, T, 5.2, 4.5, c=0.9, zero_pad=True, hl={(3, 3): "#9fd49f"})
ax.text(5.2 + 2.25, 4.8, "table $S$", ha="center", fontsize=18)
ax.text(2, -0.6, f"green block: sum {I[:3, :3].sum()}", ha="center", fontsize=18)
ax.text(5.2 + 2.25, -0.6, f"entry $S(3, 3)$ = {T[3, 3]}", ha="center", fontsize=18)
ax.set_xlim(-0.1, 9.8); ax.set_ylim(-1.0, 5.1)
save(fg, f"{OUT}/fig-table2d.svg")

# D: one step of the raster fill
fg, ax = fig(560, 330)
ax.set_aspect("equal"); ax.axis("off")
r0, k0 = 3, 3  # filling S(3, 3)
c = 0.9
for r in range(5):
    for k in range(5):
        filled = (r < r0) or (r == r0 and k < k0) or r == 0 or k == 0
        fc = "#e6e6e6" if (r == 0 or k == 0) else ("white" if filled else "#f7f7f7")
        if (r, k) == (r0, k0): fc = ORANGE
        elif (r, k) in [(r0, k0 - 1), (r0 - 1, k0)]: fc = "#9fd49f"
        elif (r, k) == (r0 - 1, k0 - 1): fc = "#f2a3a3"
        ax.add_patch(Rectangle((k * c, 4.5 - (r + 1) * c), c, c, fc=fc, ec="#999", lw=1))
        if filled and (r, k) != (r0, k0):
            ax.text((k + .5) * c, 4.5 - (r + .5) * c, str(T[r, k]), ha="center", va="center", fontsize=19)
        elif (r, k) == (r0, k0):
            ax.text((k + .5) * c, 4.5 - (r + .5) * c, "?", ha="center", va="center", fontsize=21, weight="bold")
ax.text(5.2 * c, 4.5 - 0.5 * c, "row by row,\nleft to right", fontsize=17, va="center", color="#555")
ax.text(5.2 * c, 4.5 - 2.4 * c, f"pixel $I(3,3)$ = {I[2, 2]}", fontsize=18)
ax.text(5.2 * c, 4.5 - 3.2 * c,
        f"{I[2, 2]} + {T[r0, k0 - 1]} + {T[r0 - 1, k0]} − {T[r0 - 1, k0 - 1]} = {T[r0, k0]}", fontsize=19)
ax.text(5.2 * c, 4.5 - 4.0 * c, "+ left, + up, − up-left", fontsize=17, color="#555")
ax.set_xlim(-0.1, 10.2); ax.set_ylim(-0.1, 4.6)
save(fg, f"{OUT}/fig-raster.svg")
assert I[2, 2] + T[r0, k0 - 1] + T[r0 - 1, k0] - T[r0 - 1, k0 - 1] == T[r0, k0]

# --- E. cost per window against window width ----------------------------------------
k = np.arange(1, 102)
fg, ax = fig(400, 330)
ax.semilogy(k, k ** 2 - 1 + (k == 1), color=RED, lw=3, label="direct: $k^2 - 1$")
ax.semilogy(k, np.full_like(k, 3), color=BLUE, lw=3, label="table: 3")
ax.plot([101], [101 ** 2 - 1], "o", color=RED, ms=8)
ax.text(99, 101 ** 2 * 1.5, "10 200", ha="right", fontsize=18, color=RED)
ax.set_xlabel("window width $k$")
ax.set_ylabel("additions per window")
ax.set_ylim(1, 1e5)
ax.set_yticks([1, 10, 100, 1000, 10000])
ax.set_yticklabels(["1", "10", "100", "1000", "10 000"])
ax.set_xlim(0, 105)
ax.legend(loc="center right", frameon=False, bbox_to_anchor=(1.0, 0.42))
save(fg, f"{OUT}/fig-cost.svg")

# --- F. the 8 corners of a box in 3D ------------------------------------------------
fg, ax = fig(420, 330)
ax.set_aspect("equal"); ax.axis("off")
W, H, D = 3.0, 2.0, 1.6
ox, oy = 0.55 * D, 0.35 * D  # oblique offset for depth


def P(u, v, w):  # u across, v up, w depth
    return (u * W + w * ox, v * H + w * oy)


for (p, q) in [((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)), ((0, 1, 0), (1, 1, 0)),
               ((0, 0, 1), (1, 0, 1)), ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)), ((0, 1, 1), (1, 1, 1)),
               ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1))]:
    hidden = p == (0, 0, 1) or q == (0, 0, 1)
    xs, ys = zip(P(*p), P(*q))
    ax.plot(xs, ys, color="#555", lw=2, ls="--" if hidden else "-")
for u in (0, 1):
    for v in (0, 1):
        for w in (0, 1):
            upper = u + v + w  # 1 = upper corner b along that axis
            sign = (-1) ** (3 - upper)
            x_, y_ = P(u, v, w)
            col = GREEN if sign > 0 else RED
            ax.plot(x_, y_, "o", color=col, ms=20, mec="white", mew=2)
            ax.text(x_, y_, "+" if sign > 0 else "−", color="white", ha="center", va="center", fontsize=19, weight="bold")
ax.text(*P(1, 1, 1), "   $\\mathbf{b}$", fontsize=24, va="bottom")
ax.text(*P(0, 0, 0), "$\\mathbf{a}$   ", fontsize=24, ha="right", va="top")
ax.set_xlim(-0.6, W + ox + 0.8); ax.set_ylim(-0.5, H + oy + 0.5)
save(fg, f"{OUT}/fig-cube.svg")
print("done")
