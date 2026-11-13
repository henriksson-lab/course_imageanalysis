# Figures for the kernels unit (kernels/lec.qmd). Run: python3 -I tools/figs/kernels/figs.py
# Figure sizes are the display size in px / 96, so 13 pt text shows at about 17 px.
# Filters are correlations with the kernel3x3 demo's kernels, borders replicated as in the demo.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from scipy import ndimage as ndi

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/kernels/'
plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a"})
INK, BLUE, RED, ORANGE, GREEN = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c"
cell = plt.imread(R + '/demos/src/fourier2d/cell256.png').astype(float)
if cell.ndim == 3: cell = cell[..., 0]
cell = np.round(cell * 255)

K = {"gauss": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
     "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
     "laplace": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
     "sobelx": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
     "sobely": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])}
def filt(img, k, mode="nearest"): return ndi.correlate(img, k.astype(float), mode=mode)
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)

# ------------------------------------------------------------ 1. a kernel slides over the image
I = np.array([[1 if i + j <= 2 else (5 if i + j == 3 else 9) for j in range(5)] for i in range(5)])
G = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
out = np.full((5, 5), np.nan)
for i in range(1, 4):
    for j in range(1, 4):
        out[i, j] = (I[i - 1:i + 2, j - 1:j + 2] * G).sum() / 16
assert out[2, 2] == 7.5
fig = plt.figure(figsize=(560 / 96, 230 / 96))
def grid(ax, M, hl=None, fmt="{:g}", title="", cmap_vals=None):
    n, m = M.shape
    for i in range(n):
        for j in range(m):
            v = M[i, j]
            fc = "white" if np.isnan(v) else plt.cm.Greys(0.08 + 0.5 * (v / 9))
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, fc=fc, ec="#999", lw=1))
            if not np.isnan(v):
                ax.text(j + 0.5, n - 0.5 - i, fmt.format(v), ha="center", va="center", fontsize=13.5 if len(fmt.format(v)) < 4 else 12.5,
                        color="white" if v > 6.5 and fc != "white" else INK)
    if hl is not None:
        i, j = hl
        ax.add_patch(Rectangle((j - 1, n - 2 - i), 3, 3, fill=False, ec=ORANGE, lw=3))
    ax.set_xlim(-0.1, m + 0.1); ax.set_ylim(-0.1, n + 0.1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=13.5, pad=4)
a1 = fig.add_axes([0.0, 0.02, 0.36, 0.86]); grid(a1, I.astype(float), hl=(2, 2), title="input")
a2 = fig.add_axes([0.40, 0.2, 0.2, 0.5])
for i in range(3):
    for j in range(3):
        a2.add_patch(Rectangle((j, 2 - i), 1, 1, fc="#fdf1e2", ec=ORANGE, lw=1.2))
        a2.text(j + 0.5, 2.5 - i, str(G[i, j]), ha="center", va="center", fontsize=13.5)
a2.set_xlim(-0.1, 3.1); a2.set_ylim(-0.1, 3.1); a2.set_aspect("equal"); a2.axis("off")
a2.set_title("weights ÷ 16", fontsize=13.5, pad=4)
a3 = fig.add_axes([0.64, 0.02, 0.36, 0.86])
o2 = out.copy(); grid(a3, o2, fmt="{:g}", title="output")
a3.add_patch(Rectangle((2, 2), 1, 1, fill=False, ec=ORANGE, lw=3))
fig.add_artist(FancyArrowPatch((0.36, 0.45), (0.40, 0.45), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=16, color=INK))
fig.add_artist(FancyArrowPatch((0.60, 0.45), (0.64, 0.45), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=16, color=INK))
save(fig, "fig-slide.svg")
print("output interior\n", out[1:4, 1:4])

# ------------------------------------------------------------ 2. a gallery: same input, different weights
crop = cell[60:196, 120:256]
panels = [(crop, "input", (0, 255)), (filt(crop, K["gauss"]), "Gaussian", (0, 255)),
          (filt(crop, K["sharpen"]), "sharpen", (0, 255)),
          (filt(crop, K["sobelx"]), "Sobel x\n(0 = grey)", (-300, 300)),
          ]
fig, ax = plt.subplots(2, 2, figsize=(400 / 96, 440 / 96))
for a, (img, t, (lo, hi)) in zip(ax.ravel(), panels):
    a.imshow(np.clip(img, lo, hi), cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    a.set_xticks([]); a.set_yticks([]); a.set_xlabel(t, fontsize=15.5, linespacing=1.05)
fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.13, wspace=0.06, hspace=0.25)
save(fig, "fig-gallery.png", dpi=192)

# ------------------------------------------------------------ 3. slope and curvature of a step
x = np.arange(5); f = np.array([0, 0, 3, 6, 6.])
slope = np.full(5, np.nan); curv = np.full(5, np.nan)
for i in range(1, 4):
    slope[i] = (f[i + 1] - f[i - 1]) / 2; curv[i] = f[i + 1] - 2 * f[i] + f[i - 1]
assert list(slope[1:4]) == [1.5, 3, 1.5] and list(curv[1:4]) == [3, 0, -3]
fig, ax = plt.subplots(3, 1, figsize=(340 / 96, 360 / 96), sharex=True)
for a, y, c, lab in zip(ax, [f, slope, curv], [INK, BLUE, RED], ["pixel\nvalues", "slope", "curvature"]):
    ok = np.isfinite(y)
    a.axhline(0, color="#bbb", lw=1)
    a.vlines(x[ok], 0, y[ok], color=c, lw=3); a.plot(x[ok], y[ok], "o", color=c, ms=8)
    for xi, yi in zip(x[ok], y[ok]):
        a.text(xi + 0.12, yi + (0.5 if yi >= 0 else -0.6), f"{yi:g}", color=c, fontsize=13, va="center")
    a.set_ylabel(lab, fontsize=13.5, linespacing=1.0)
    a.set_ylim(-4.5, 8.2) if lab != "pixel\nvalues" else a.set_ylim(-0.8, 8.2)
    a.set_yticks([]); a.spines[["top", "right", "left"]].set_visible(False)
    a.axvline(2, color=ORANGE, lw=1.2, ls="--", zorder=0)
ax[2].set_xticks(x); ax[2].set_xlim(-0.4, 4.6); ax[2].set_xlabel("pixel", fontsize=13)
ax[2].text(2.12, 3.2, "zero\ncrossing", color=ORANGE, fontsize=13, va="center", linespacing=1.0)
fig.subplots_adjust(left=0.2, right=0.97, top=0.98, bottom=0.13, hspace=0.12)
save(fig, "fig-derivative.svg")

# ------------------------------------------------------------ 4. noise and edges
rng = np.random.default_rng(3)
crop = cell[40:200, 90:250]
noisy = np.clip(np.round(crop + 30 * rng.standard_normal(crop.shape)), 0, 255)
lap_clean = filt(crop, K["laplace"])
lap_noisy = filt(noisy, K["laplace"])
blurred = filt(filt(noisy, K["gauss"]), K["gauss"])
log_noisy = filt(blurred, K["laplace"])
print("Laplacian SD: clean %.1f  noisy %.1f  blurred-first %.1f" % (lap_clean.std(), lap_noisy.std(), log_noisy.std()))
fig, ax = plt.subplots(1, 3, figsize=(720 / 96, 280 / 96))
for a, img, t, lim in zip(ax, [noisy, lap_noisy, log_noisy],
                          ["input + noise (SD 30)", "Laplacian:\nnoise swamps the edges", "Gaussian twice, then\nLaplacian: edges again"],
                          [None, 120, 40]):
    if lim is None: a.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    else: a.imshow(np.clip(img, -lim, lim), cmap="gray", vmin=-lim, vmax=lim, interpolation="nearest")
    a.set_xticks([]); a.set_yticks([]); a.set_xlabel(t, fontsize=13.5, linespacing=1.05)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.2, wspace=0.07)
save(fig, "fig-noise.png", dpi=192)

# ------------------------------------------------------------ 5. border rules
crop = cell[30:94, 150:214]                       # 64 x 64 near the image's right side
p = 16
modes = [("constant", "zeros"), ("edge", "copy the edge"), ("symmetric", "mirror"), ("wrap", "wrap around")]
fig, ax = plt.subplots(2, 2, figsize=(440 / 96, 470 / 96))
for a, (m, t) in zip(ax.ravel(), modes):
    img = np.pad(crop, p, mode=m) if m != "constant" else np.pad(crop, p, mode="constant", constant_values=0)
    a.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    a.add_patch(Rectangle((p - 0.5, p - 0.5), 64, 64, fill=False, ec=ORANGE, lw=2, ls="--"))
    a.set_xticks([]); a.set_yticks([]); a.set_xlabel(t, fontsize=15.5)
fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.08, wspace=0.06, hspace=0.16)
save(fig, "fig-border.png", dpi=192)
