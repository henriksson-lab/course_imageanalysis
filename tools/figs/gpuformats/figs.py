# Static figures for gpuformats/lec.qmd. Run from the repo root:
#   python3 -I tools/figs/gpuformats/figs.py
# Uses the course micrograph demos/src/fourier2d/cell256.png (transmission electron micrograph of
# a mesenchymal stem cell, Robert M. Hunt, CC BY 3.0, via Wikimedia Commons) and synthetic data.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import io

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(R, "gpuformats")
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans"})
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#888888"
CH = {"R": "#f4a6a6", "G": "#a8dca8", "B": "#a9c4ec"}
cell = io.imread(os.path.join(R, "demos/src/fourier2d/cell256.png")).astype(float) / 255


def save(fig, name, dpi=None):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", **({"dpi": dpi} if dpi else {}))
    plt.close(fig)


def show(ax, im, label, interp="nearest", cmap="gray"):
    ax.imshow(im, cmap=cmap, vmin=0, vmax=1, interpolation=interp)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(label, fontsize=30)
    for s in ax.spines.values():
        s.set_color("#999")


# ---------------------------------------------------------------- 1. interleaved vs planar
fig, ax = plt.subplots(figsize=(12, 3.2))
ax.set_xlim(-3.2, 24.2); ax.set_ylim(-0.4, 4.2); ax.axis("off")
inter = [(c, x, y) for y in range(2) for x in range(4) for c in "RGB"]
planar = [(c, x, y) for c in "RGB" for y in range(2) for x in range(4)]
for row, (seq, lab) in enumerate([(inter, "interleaved"), (planar, "planar")]):
    y0 = 2.6 - row * 2.2
    ax.text(-0.3, y0 + 0.45, lab, ha="right", va="center", fontsize=16, weight="bold")
    for k, (c, x, yy) in enumerate(seq):
        ax.add_patch(Rectangle((k, y0), 0.95, 0.9, fc=CH[c], ec="#555", lw=1))
        ax.text(k + 0.47, y0 + 0.45, f"{c}{x + 4 * yy}", ha="center", va="center", fontsize=15)
    if row == 0:
        for k in range(0, 24, 3):
            ax.plot([k, k + 2.95], [y0 - 0.12] * 2, color="#555", lw=1.2)
        ax.text(1.5, y0 - 0.22, "pixel 0", ha="center", va="top", fontsize=15, color="#444")
        ax.plot([12, 12], [y0 - 0.1, y0 + 1.0], color=RED, lw=2.5)
        ax.text(12, y0 + 1.05, "row 1 (pixels 4–7) starts", color=RED, ha="center", va="bottom", fontsize=15)
ax.text(12, -0.35, "byte offset →   (R5 = red of pixel 5)", ha="center", fontsize=15, color="#444")
save(fig, "fig-layout.svg")

# ---------------------------------------------------------------- 2. wrong stride: shear + colour cycling
w = 253  # 3*253 = 759 bytes of data, padded to a stride of 760
img = plt.get_cmap("viridis")(cell[:, :w])[:, :, :3]
h = img.shape[0]
data = (img * 255).astype(np.uint8)
stride = 760
buf = np.zeros(h * stride, np.uint8)
for y in range(h):
    buf[y * stride:y * stride + 3 * w] = data[y].reshape(-1)
wrong = np.zeros_like(data)
for y in range(h):
    s0 = y * 759
    wrong[y] = buf[s0:s0 + 3 * w].reshape(w, 3)
fig, axs = plt.subplots(1, 2, figsize=(10, 5.3))
show(axs[0], data / 255, "stride 760")
show(axs[1], wrong / 255, "read as 759")
fig.tight_layout()
save(fig, "fig-stride.png", dpi=90)

# ---------------------------------------------------------------- 3. byte order
v16 = np.round(cell * 4000).astype(np.uint16)
swapped = v16.byteswap()
fig, axs = plt.subplots(1, 2, figsize=(10, 5.0))
show(axs[0], v16 / 4000, "correct")
show(axs[1], swapped / 65535, "bytes swapped")
fig.tight_layout()
save(fig, "fig-endian.png", dpi=90)
print("pixel 1000 swapped ->", np.array([1000], np.uint16).byteswap()[0])

# ---------------------------------------------------------------- 4. 8-8-8 vs 5-6-5: grey ramp
x = np.linspace(0.30, 0.45, 600)
r8 = np.round(x * 255) / 255
r5 = np.round(x * 31) / 31; g6 = np.round(x * 63) / 63
lo, hi = 0.30, 0.45
st = lambda v: np.clip((v - lo) / (hi - lo), 0, 1)
ramp8 = np.repeat(np.stack([st(r8)] * 3, -1)[None], 60, 0)
ramp565 = np.repeat(np.stack([st(r5), st(g6), st(r5)], -1)[None], 60, 0)
fig, axs = plt.subplots(2, 1, figsize=(9, 2.9))
for ax, im, t in [(axs[0], ramp8, "8-8-8: grey steps"), (axs[1], ramp565, "5-6-5: green steps at other values than red and blue: tinted")]:
    ax.imshow(im, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=15, loc="left")
fig.text(0.5, -0.02, "values 0.30 … 0.45, contrast stretched", ha="center", fontsize=13, color="#444")
fig.tight_layout()
save(fig, "fig-565.png", dpi=100)

# ---------------------------------------------------------------- 5. sRGB code spacing
def srgb(L):
    L = np.asarray(L, float)
    return np.where(L <= 0.0031308, 12.92 * L, 1.055 * L ** (1 / 2.4) - 0.055)


def srgb_inv(V):
    V = np.asarray(V, float)
    return np.where(V <= 0.04045, V / 12.92, ((V + 0.055) / 1.055) ** 2.4)


print("0.005 -> linear code", round(0.005 * 255), " sRGB code", int(np.round(srgb(0.005) * 255)))
fig, ax = plt.subplots(figsize=(6.0, 3.4))
xmax = 0.03
lin_codes = np.arange(0, 256) / 255; lin_codes = lin_codes[lin_codes <= xmax]
s_codes = srgb_inv(np.arange(0, 256) / 255); s_codes = s_codes[s_codes <= xmax]
for xv in lin_codes:
    ax.plot([xv, xv], [1.1, 1.9], color=BLUE, lw=2)
for xv in s_codes:
    ax.plot([xv, xv], [0.1, 0.9], color=ORANGE, lw=1.2)
ax.text(xmax * 1.01, 1.5, f"linear 8 bit:\n{len(lin_codes)} codes", color=BLUE, va="center", fontsize=14)
ax.text(xmax * 1.01, 0.5, f"sRGB 8 bit:\n{len(s_codes)} codes", color=ORANGE, va="center", fontsize=14)
ax.set_xlim(0, xmax); ax.set_ylim(0, 2.0); ax.set_yticks([])
ax.set_xlabel("light intensity (linear), the darkest 3 %")
ax.spines[["top", "right", "left"]].set_visible(False)
save(fig, "fig-srgb.svg")
print("codes below 0.03: linear", len(lin_codes), "sRGB", len(s_codes))

# ---------------------------------------------------------------- 6. nearest vs bilinear magnification
crop = cell[120:136, 100:116]
fig, axs = plt.subplots(1, 2, figsize=(9, 4.6))
show(axs[0], crop, "nearest", interp="nearest")
# true bilinear (texel centres), 12x
from scipy.ndimage import map_coordinates
k = 12
yy, xx = np.mgrid[0:16 * k, 0:16 * k]
u = (xx + 0.5) / k - 0.5; v = (yy + 0.5) / k - 0.5
bil = map_coordinates(crop, [np.clip(v, 0, 15), np.clip(u, 0, 15)], order=1)
show(axs[1], bil, "bilinear")
fig.tight_layout()
save(fig, "fig-filter.png", dpi=90)

# ---------------------------------------------------------------- 7. straight vs premultiplied alpha
n = 12
yy, xx = np.mgrid[0:n, 0:n]
inside = ((xx - 5.5) ** 2 + (yy - 5.5) ** 2) < 4.2 ** 2
rgb = np.zeros((n, n, 3)); rgb[inside] = [0.85, 0.1, 0.1]; rgb[~inside] = [0.1, 0.8, 0.1]  # hidden green
alpha = inside.astype(float)
k = 20
Y, X = np.mgrid[0:n * k, 0:n * k]
u = (X + 0.5) / k - 0.5; v = (Y + 0.5) / k - 0.5
def bl(ch):
    return map_coordinates(ch, [np.clip(v, 0, n - 1), np.clip(u, 0, n - 1)], order=1)
A = bl(alpha)
straight = np.stack([bl(rgb[..., c]) for c in range(3)], -1)
pre = np.stack([bl(rgb[..., c] * alpha) for c in range(3)], -1)
bg = np.ones(3)
over_straight = straight * A[..., None] + (1 - A[..., None]) * bg
over_pre = pre + (1 - A[..., None]) * bg
fig, axs = plt.subplots(1, 2, figsize=(9, 4.6))
show(axs[0], np.clip(over_straight, 0, 1), "straight")
show(axs[1], np.clip(over_pre, 0, 1), "premultiplied")
fig.tight_layout()
save(fig, "fig-premult.png", dpi=90)

# ---------------------------------------------------------------- 8. mipmap chain
levels = [cell]
while levels[-1].shape[0] > 1:
    a = levels[-1]
    levels.append(0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]))
canvas = np.ones((256, 256 + 128 + 64 + 32 + 16 + 8 + 6 * 8))
x0 = 0
for L in levels[:6]:
    s = L.shape[0]
    canvas[256 - s:, x0:x0 + s] = L
    x0 += s + 8
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.imshow(canvas[:, :x0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
x0 = 0
for L in levels[:6]:
    s = L.shape[0]
    if s >= 32:
        ax.text(x0 + s / 2, 256 + 8, f"{s}", ha="center", va="top", fontsize=26)
    x0 += s + 8
ax.text(x0 - 30, 256 + 8, "… 1", ha="left", va="top", fontsize=26)
ax.axis("off")
save(fig, "fig-mipmap.png", dpi=100)
