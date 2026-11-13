# Figures for the images unit (images/lec.qmd). Run: python3 -I tools/figs/images/figs.py
# Figure sizes are the display size in px / 96, so 12.5 pt text shows at about 17 px.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
from scipy import ndimage as ndi

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/images/'
plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a"})
INK, BLUE, RED, ORANGE = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e"
COL1, COL2 = "#d62728", "#1f77b4"        # the matrix2x2 demo: column 1 red, column 2 blue
cell = plt.imread(R + '/demos/src/fourier2d/cell256.png').astype(float)
if cell.ndim == 3: cell = cell[..., 0]
cell8 = np.round(cell * 255).astype(int)

def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)

# ------------------------------------------------------------ 1. the image is a matrix
r0, c0, n = 98, 194, 6                  # top-left corner of the zoomed patch
fig = plt.figure(figsize=(560 / 96, 300 / 96))
a1 = fig.add_axes([0.0, 0.02, 0.42, 0.78]); a2 = fig.add_axes([0.56, 0.02, 0.43, 0.78])
a1.imshow(cell, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
a1.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), n, n, fill=False, ec=ORANGE, lw=2.2))
a1.set_xticks([]); a1.set_yticks([])
a1.set_title("256 × 256 pixels", fontsize=14, pad=4)
P = cell8[r0:r0 + n, c0:c0 + n]
a2.imshow(P, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
for i in range(n):
    for j in range(n):
        a2.text(j, i, str(P[i, j]), ha="center", va="center", fontsize=13.5,
                color="white" if P[i, j] < 128 else "black")
for s in a2.spines.values(): s.set_edgecolor(ORANGE); s.set_linewidth(2.2)
a2.set_xticks(range(n)); a2.set_xticklabels([str(c0 + 1 + j) for j in range(n)], fontsize=12)
a2.set_yticks(range(n)); a2.set_yticklabels([str(r0 + 1 + i) for i in range(n)], fontsize=12)
a2.xaxis.tick_top(); a2.tick_params(length=0, pad=2)
a2.set_title("column j →", fontsize=14, pad=22)
a2.set_ylabel("← row i", fontsize=14, labelpad=2)
from matplotlib.patches import ConnectionPatch
for (ya, yb) in [(r0 - 0.5, -0.5), (r0 + n - 0.5, n - 0.5)]:
    fig.add_artist(ConnectionPatch((c0 + n - 0.5, ya), (-0.5, yb), "data", "data", axesA=a1, axesB=a2,
                                   color=ORANGE, lw=1.3, ls="--"))
save(fig, "fig-matrix.png", dpi=192)

# ------------------------------------------------------------ 2. values vs positions
crop = cell[40:216, 40:216]
contrast = np.clip((crop - 0.25) / 0.5, 0, 1)       # stretch grey levels 0.25..0.75 to 0..1
rot = ndi.rotate(crop, 30, reshape=False, order=1, mode="constant", cval=1.0)
fig, ax = plt.subplots(1, 3, figsize=(720 / 96, 245 / 96))
for a, img, t in zip(ax, [crop, contrast, rot],
                     ["input", "values changed:\ncontrast stretched", "positions changed:\nrotated 30°"]):
    a.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="bilinear"); a.set_xticks([]); a.set_yticks([])
    a.set_xlabel(t, fontsize=12.5, linespacing=1.1)
fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.2, wspace=0.08)
save(fig, "fig-twokinds.png", dpi=192)

# ------------------------------------------------------------ 3. read a matrix from its columns
Fpoly = np.array([[0.15, 0.1], [0.3, 0.1], [0.3, 0.45], [0.6, 0.45], [0.6, 0.6], [0.3, 0.6], [0.3, 0.75],
                  [0.7, 0.75], [0.7, 0.9], [0.15, 0.9]])   # an F inside the unit square
A = np.array([[1, 2], [0, 1]])
fig, ax = plt.subplots(1, 2, figsize=(480 / 96, 270 / 96), gridspec_kw={"width_ratios": [1.6, 3.1]})
for a, M, ttl in zip(ax, [np.eye(2), A], ["before", "after  A = [1 2; 0 1]"]):
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) @ M.T
    a.add_patch(Polygon(sq, closed=True, fc="#9467bd", alpha=0.13, ec="#9467bd", lw=1.5))
    a.add_patch(Polygon(Fpoly @ M.T, closed=True, fc="#26303d", ec="none"))
    for v, c in [(M[:, 0], COL1), (M[:, 1], COL2)]:
        a.add_patch(FancyArrowPatch((0, 0), tuple(v), arrowstyle="-|>", mutation_scale=18, lw=2.6, color=c))
    a.set_aspect("equal"); a.axhline(0, color="#bbb", lw=1, zorder=0); a.axvline(0, color="#bbb", lw=1, zorder=0)
    a.set_title(ttl, fontsize=14.5)
    a.tick_params(labelsize=13.5); a.set_yticks([0, 1]); a.grid(color="#eee")
ax[0].set_xlim(-0.3, 1.3); ax[0].set_ylim(-0.3, 1.45); ax[0].set_xticks([0, 1])
ax[1].set_xlim(-0.3, 3.3); ax[1].set_ylim(-0.3, 1.45); ax[1].set_xticks([0, 1, 2, 3])
ax[0].text(1.0, -0.22, "(1, 0)", color=COL1, ha="center", fontsize=14.5)
ax[0].text(-0.05, 1.08, "(0, 1)", color=COL2, ha="left", fontsize=14.5)
ax[1].text(1.0, -0.22, "col 1 = (1, 0)", color=COL1, ha="center", fontsize=14.5)
ax[1].text(1.55, 1.12, "col 2 = (2, 1)", color=COL2, ha="left", fontsize=14.5)
fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.08, wspace=0.18)
save(fig, "fig-columns.svg")

# ------------------------------------------------------------ 4. forward holes vs inverse mapping
small = cell[95:107, 191:203]           # 12 x 12 pixels, enlarged 3 times
k = 3; N = small.shape[0] * k
fwd = np.full((N, N), np.nan)
for i in range(small.shape[0]):
    for j in range(small.shape[1]):
        fwd[i * k + 1, j * k + 1] = small[i, j]        # each input pixel lands on one output pixel
yy, xx = np.mgrid[0:N, 0:N]
src_y, src_x = (yy + 0.5) / k - 0.5, (xx + 0.5) / k - 0.5   # where each output pixel came from
near = ndi.map_coordinates(small, [src_y, src_x], order=0, mode="nearest")
bil = ndi.map_coordinates(small, [src_y, src_x], order=1, mode="nearest")
filled = np.isfinite(fwd).mean()
fig, ax = plt.subplots(1, 3, figsize=(690 / 96, 255 / 96))
cm = matplotlib.colormaps["gray"].copy(); cm.set_bad("#f3c7c7")
for a, img, t in zip(ax, [fwd, near, bil], ["forward: holes (pink)\n%d of %d pixels filled" % (np.isfinite(fwd).sum(), N * N),
                                            "backward, nearest\nneighbour: blocky", "backward, bilinear:\nsmooth"]):
    a.imshow(np.ma.masked_invalid(img), cmap=cm, vmin=0, vmax=1, interpolation="nearest")
    a.set_xticks([]); a.set_yticks([]); a.set_xlabel(t, fontsize=12.5, linespacing=1.1)
fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.21, wspace=0.08)
save(fig, "fig-backward.png", dpi=192)
print("forward filled fraction", filled)

# ------------------------------------------------------------ 5. bilinear weights as areas
al, be = 0.3, 0.6
fig, a = plt.subplots(figsize=(300 / 96, 290 / 96))
cols = {"00": "#cfe3f0", "10": "#f6dcc0", "01": "#d8ecd5", "11": "#ead6ef"}
# the weight of a corner is the area of the rectangle OPPOSITE to it
a.add_patch(Rectangle((al, be), 1 - al, 1 - be, fc=cols["00"], ec="none"))
a.add_patch(Rectangle((0, be), al, 1 - be, fc=cols["10"], ec="none"))
a.add_patch(Rectangle((al, 0), 1 - al, be, fc=cols["01"], ec="none"))
a.add_patch(Rectangle((0, 0), al, be, fc=cols["11"], ec="none"))
a.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=INK, lw=1.5))
a.plot([al, al], [0, 1], color="#888", lw=1, ls="--"); a.plot([0, 1], [be, be], color="#888", lw=1, ls="--")
for (x, y, t) in [(0, 0, "$I_{00}$"), (1, 0, "$I_{10}$"), (0, 1, "$I_{01}$"), (1, 1, "$I_{11}$")]:
    a.plot(x, y, "o", color=INK, ms=7)
    a.text(x + (-0.07 if x == 0 else 0.07), y + (-0.08 if y == 0 else 0.07), t, ha="center", va="center", fontsize=13)
a.plot(al, be, "o", color=RED, ms=9); a.text(al + 0.04, be + 0.05, "$(\\alpha, \\beta)$", color=RED, fontsize=13)
a.text((1 + al) / 2, (1 + be) / 2, "weight\nof $I_{00}$", ha="center", va="center", fontsize=12)
a.text(al / 2, be / 2, "of\n$I_{11}$", ha="center", va="center", fontsize=12)
a.set_xlim(-0.2, 1.2); a.set_ylim(-0.2, 1.2); a.set_aspect("equal"); a.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
save(fig, "fig-bilinear.svg")

# ------------------------------------------------------------ 6. order matters
S = np.diag([2, 1]); Rq = np.array([[0, -1], [1, 0]])
Fc = Fpoly - 0.5                         # centred F
fig, ax = plt.subplots(1, 3, figsize=(520 / 96, 235 / 96))
for a, M, t in zip(ax, [np.eye(2), Rq @ S, S @ Rq],
                   ["input", "stretch x by 2,\nthen quarter turn", "quarter turn,\nthen stretch x by 2"]):
    a.add_patch(Polygon(np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]) @ M.T, closed=True,
                        fc="#9467bd", alpha=0.12, ec="#9467bd", lw=1.3))
    a.add_patch(Polygon(Fc @ M.T, closed=True, fc="#26303d", ec="none"))
    a.set_xlim(-1.15, 1.15); a.set_ylim(-1.15, 1.15); a.set_aspect("equal")
    a.set_xticks([]); a.set_yticks([]); a.set_xlabel(t, fontsize=13, linespacing=1.1)
    for s in a.spines.values(): s.set_edgecolor("#ccc")
fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.24, wspace=0.08)
save(fig, "fig-order.svg")
