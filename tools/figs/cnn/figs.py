# Figures for the cnn unit (cnn/lec.qmd). Run: python3 -I tools/figs/cnn/figs.py
# Figure sizes are the display size in px / 96. Filters are correlations, borders replicated, as in
# the kernel3x3/convlayer demos. Image: the course micrograph demos/src/fourier2d/cell256.png
# (Transmission electron micrograph of a mesenchymal stem cell, Robert M. Hunt, CC BY 3.0, via
# Wikimedia Commons) and MNIST test digits from demos/src/vaeimages/digit_examples.ts.
import re, base64
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from scipy import ndimage as ndi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/cnn/'
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a"})
INK, BLUE, RED, ORANGE, GREEN = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c"
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)
def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color("#999")
cell = plt.imread(R + '/demos/src/fourier2d/cell256.png').astype(float)
if cell.ndim == 3: cell = cell[..., 0]
cell = np.round(cell * 255)
crop = cell[40:200, 80:240]   # the nucleus
K = {"Sobel x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
     "Sobel y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
     "Laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
     "Gaussian": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float) / 16}
def filt(img, k): return ndi.correlate(img, k, mode="nearest")

# ------------------------------------------------------------ 1. a hand-made feature stack (day 14)
g = ndi.gaussian_filter(crop, 2)
H = hessian_matrix(crop, sigma=3, order="rc", use_gaussian_derivatives=False)
ev = hessian_matrix_eigvals(H)[0]
feats = [("input", crop, "gray"), ("Gaussian", g, "gray"),
         ("gradient", ndi.gaussian_gradient_magnitude(crop, 2), "gray"),
         ("LoG", ndi.gaussian_laplace(crop, 3), "RdBu_r"), ("Hessian", ev, "RdBu_r")]
fig, axs = plt.subplots(2, 3, figsize=(560 / 96, 420 / 96))
for ax in axs.flat: ax.axis("off")
for ax, (t, im, cm) in zip([axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1]], feats):
    ax.axis("on"); off(ax)
    if cm == "RdBu_r":
        m = np.percentile(np.abs(im), 99); ax.imshow(im, cmap=cm, vmin=-m, vmax=m)
    else:
        ax.imshow(im, cmap=cm, vmin=np.percentile(im, 0.5), vmax=np.percentile(im, 99.5))
    ax.set_title(t, fontsize=15)
axs[1, 2].text(0.05, 0.5, "chosen by us:\nfilters and\nscales", fontsize=16, va="center", color=BLUE)
fig.tight_layout(h_pad=0.6, w_pad=0.4); save(fig, "fig-featurestack.png", dpi=192)

# ------------------------------------------------------------ 2. a dense layer would not notice shuffled pixels
small = crop[::2, ::2]
perm = np.random.default_rng(1).permutation(small.size)
shuf = small.ravel()[perm].reshape(small.shape)
fig, axs = plt.subplots(1, 2, figsize=(470 / 96, 260 / 96))
for ax, im, t in [(axs[0], small, "an image"), (axs[1], shuf, "its pixels shuffled")]:
    ax.imshow(im, cmap="gray"); off(ax); ax.set_title(t, fontsize=14)
fig.tight_layout(w_pad=0.6); save(fig, "fig-shuffle.png", dpi=192)

# ------------------------------------------------------------ 3. a bank of kernels → feature maps (after ReLU)
inp = ndi.zoom(ndi.gaussian_filter(crop, 1.0), 0.5, order=1)   # 80 × 80: a smoothed, smaller input
fig = plt.figure(figsize=(700 / 96, 290 / 96))
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1], height_ratios=[0.62, 1], wspace=0.12, hspace=0.22,
                      left=0.01, right=0.99, top=0.9, bottom=0.02)
ax = fig.add_subplot(gs[1, 0]); ax.imshow(inp, cmap="gray"); off(ax); ax.set_title("input", fontsize=14)
for j, (name, k) in enumerate(K.items()):
    ax = fig.add_subplot(gs[0, j + 1])
    kk = k * 16 if name == "Gaussian" else k
    m = np.abs(kk).max()
    ax.imshow(kk, cmap="RdBu_r", vmin=-m, vmax=m)
    for (r, c), v in np.ndenumerate(kk):
        ax.text(c, r, f"{v:g}", ha="center", va="center", fontsize=10.5, color="white" if abs(v) > 0.6 * m else INK)
    off(ax); ax.set_title(name + (" ÷16" if name == "Gaussian" else ""), fontsize=13.5)
    fm = np.maximum(0, filt(inp, k) + (0 if name != "Gaussian" else -128))
    ax = fig.add_subplot(gs[1, j + 1]); ax.imshow(fm, cmap="gray", vmin=0, vmax=np.percentile(fm, 99.5)); off(ax)
fig.text(0.6, 0.02, "", fontsize=12)
save(fig, "fig-convbank.png", dpi=192)

# ------------------------------------------------------------ 4. shift the input, the feature map shifts
dx, dy = 24, 16
big = cell[20:220, 60:260]
a = big[16:176, 24:184]; b = big[16 - dy:176 - dy, 24 - dx:184 - dx]   # b = the view moved by (dx, dy): content shifted
sm = lambda im: ndi.gaussian_filter(im, 1.5, mode="nearest")   # Sobel x after a little smoothing
fa = np.maximum(0, filt(sm(a), K["Sobel x"])); fb = np.maximum(0, filt(sm(b), K["Sobel x"]))
# away from the border the maps agree exactly after shifting
assert np.allclose(fa[10:-dy - 10, 10:-dx - 10], fb[dy + 10:-10, dx + 10:-10], atol=1e-6)
vm = np.percentile(fa, 99.5)
fig, axs = plt.subplots(2, 2, figsize=(400 / 96, 420 / 96))
for (r, im, cm, vmax, t) in [((0, 0), a, "gray", None, "input"), ((0, 1), b, "gray", None, "shifted input"),
                              ((1, 0), fa, "gray", vm, "feature map"), ((1, 1), fb, "gray", vm, "same map, shifted")]:
    ax = axs[r]; ax.imshow(im, cmap=cm, vmin=0, vmax=vmax); off(ax); ax.set_title(t, fontsize=13)
for (r, c), (x, y) in [((0, 0), (100, 80)), ((0, 1), (100 + dx, 80 + dy)), ((1, 0), (100, 80)), ((1, 1), (100 + dx, 80 + dy))]:
    axs[r, c].scatter([x], [y], s=160, facecolor="none", edgecolor=ORANGE, lw=2.5)
fig.tight_layout(h_pad=0.6, w_pad=0.4); save(fig, "fig-shift.png", dpi=192)

# ------------------------------------------------------------ 5. 2 × 2 max pooling by hand
X = np.array([[1, 3, 2, 0], [4, 2, 1, 1], [0, 1, 5, 2], [2, 2, 3, 6]])
P = X.reshape(2, 2, 2, 2).max(axis=(1, 3))
assert P.tolist() == [[4, 2], [2, 6]]
cols = ["#dbe8f2", "#fbe4c6", "#dcefe2", "#f3d6d6"]
fig, ax = plt.subplots(figsize=(430 / 96, 220 / 96))
for i in range(4):
    for j in range(4):
        blk = (i // 2) * 2 + j // 2
        ax.add_patch(Rectangle((j, 3 - i), 1, 1, fc=cols[blk], ec="#888"))
        win = X[i, j] == P[i // 2, j // 2]
        ax.text(j + 0.5, 3.5 - i, str(X[i, j]), ha="center", va="center", fontsize=17, fontweight="bold" if win else "normal")
for i in range(2):
    for j in range(2):
        ax.add_patch(Rectangle((6 + j, 2 - i), 1, 1, fc=cols[i * 2 + j], ec="#888"))
        ax.text(6.5 + j, 2.5 - i, str(P[i, j]), ha="center", va="center", fontsize=17, fontweight="bold")
for i in range(1, 2): ax.plot([0, 4], [2, 2], color=INK, lw=2.5); ax.plot([2, 2], [0, 4], color=INK, lw=2.5)
ax.add_patch(FancyArrowPatch((4.3, 2), (5.7, 2), arrowstyle="-|>", mutation_scale=22, color=INK, lw=2))
ax.text(5, 2.25, "max", ha="center", fontsize=14)
ax.set_xlim(-0.1, 8.1); ax.set_ylim(-0.1, 4.1); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout(); save(fig, "fig-maxpool.svg")

# ------------------------------------------------------------ 6. receptive field of three 3×3 layers, in 1D
fig, ax = plt.subplots(figsize=(500 / 96, 300 / 96))
n = 9; c = 4
for L in range(4):
    y = L * 1.0
    lo, hi = c - (3 - L), c + (3 - L)   # units of layer L that feed the top unit
    for i in range(n):
        inside = lo <= i <= hi if L < 3 else i == c
        ax.add_patch(plt.Circle((i, y), 0.22, fc=ORANGE if inside else "white", ec=ORANGE if inside else "#aaa", lw=1.8, zorder=3))
    if L > 0:
        for i in range(n):
            if (lo <= i <= hi) if L < 3 else i == c:
                for d in (-1, 0, 1):
                    ax.plot([i, i + d], [y - 0.22, y - 1 + 0.22], color=ORANGE, lw=1.4, zorder=1)
    ax.text(-0.9, y, ["input", "layer 1", "layer 2", "layer 3"][L], ha="right", va="center", fontsize=15)
    if L < 3: ax.text(n - 0.3, y, f"{hi - lo + 1} wide", va="center", fontsize=15, color=ORANGE)
ax.set_xlim(-3.0, n + 1.6); ax.set_ylim(-0.5, 3.5); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout(); save(fig, "fig-receptive.svg")

# ------------------------------------------------------------ 7. augmentation: modified copies of a training digit
s = open(R + '/demos/src/vaeimages/digit_examples.ts').read()
ims = re.findall(r'"label":"([^"]+)","img":"([^"]+)"', s)
lab, b64 = ims[12]; assert lab == "a test 6"
d = np.frombuffer(base64.b64decode(b64), np.uint8).reshape(28, 28).astype(float) / 255
rot = lambda im, deg: np.clip(ndi.rotate(im, deg, reshape=False, order=1), 0, 1)
copies = [("original", d), ("rotated 30°", rot(d, 30)), ("rotated −30°", rot(d, -30)),
          ("shifted", ndi.shift(d, (2, -3), order=0)), ("fainter", 0.5 * d),
          ("rotated 180°:\na 9?", rot(d, 180))]
fig, axs = plt.subplots(1, 6, figsize=(760 / 96, 190 / 96))
for ax, (t, im) in zip(axs, copies):
    ax.imshow(im, cmap="Greys", vmin=0, vmax=1, interpolation="nearest"); off(ax)
    ax.set_xlabel(t, fontsize=13, color=RED if "9" in t else INK)
for sp in axs[-1].spines.values(): sp.set_color(RED); sp.set_linewidth(2)
fig.tight_layout(w_pad=0.5); save(fig, "fig-augment.png", dpi=192)
