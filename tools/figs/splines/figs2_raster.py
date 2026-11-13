"""Raster figures for the splines day (spline-images/), from the data that figs2.ts computes
with the course's own interpolation and FFD code.

    python3 -I tools/figs/splines/figs2_raster.py <rawdir> --export-cell   # writes <rawdir>/cell256.json
    (run figs2.ts with <rawdir>)
    python3 -I tools/figs/splines/figs2_raster.py <rawdir>                 # writes the PNGs

Run from the repository root. The micrograph is demos/src/fourier2d/cell256.png (Robert M.
Hunt, CC BY 3.0, via Wikimedia Commons).
"""
import json
import sys

import numpy as np

raw = sys.argv[1]
if "--export-cell" in sys.argv:
    from PIL import Image
    im = np.asarray(Image.open("demos/src/fourier2d/cell256.png").convert("RGB"), dtype=float)
    grey = 0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2]   # as fromImageData
    json.dump(grey.ravel().tolist(), open(f"{raw}/cell256.json", "w"))
    sys.exit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.size": 18, "font.family": "DejaVu Sans"})
D = json.load(open(f"{raw}/raster.json"))
N = 128


def show(ax, a, title, n=N):
    ax.imshow(np.clip(np.asarray(a).reshape(n, n), 0, 255), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(title, fontsize=18)
    ax.set_xticks([]); ax.set_yticks([])


# fig-zoom.png: one 16 x 16 patch, zoomed x8 three ways
fig, axs = plt.subplots(1, 3, figsize=(6.0, 2.75), dpi=100)
for ax, m, t in zip(axs, ["nearest", "bilinear", "bspline"], ["nearest", "bilinear", "cubic B-spline"]):
    show(ax, D["zoom"][m], t)
fig.text(0.5, 0.03, "16 × 16 pixels, zoomed ×8", ha="center", fontsize=18)
fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.14, wspace=0.05)
fig.savefig("spline-images/fig-zoom.png", facecolor="white")
plt.close(fig)

# fig-rotate10.png: central 64 x 64 after 10 x 36 degrees
c0, c1 = 32, 96
fig, axs = plt.subplots(2, 2, figsize=(5.0, 5.7), dpi=100)
crop = lambda a: np.asarray(a).reshape(N, N)[c0:c1, c0:c1]
items = [(D["cell"], "original"), (D["rot"]["nearest"]["img"], f"nearest ({D['rot']['nearest']['rms']:.1f})"),
         (D["rot"]["bilinear"]["img"], f"bilinear ({D['rot']['bilinear']['rms']:.1f})"),
         (D["rot"]["bspline"]["img"], f"cubic B-spline ({D['rot']['bspline']['rms']:.1f})")]
for ax, (a, t) in zip(axs.ravel(), items):
    show(ax, crop(a), t, n=c1 - c0)
fig.text(0.5, 0.015, "after 10 × 36° (RMS change)", ha="center", fontsize=18)
fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.07, wspace=0.05, hspace=0.14)
fig.savefig("spline-images/fig-rotate10.png", facecolor="white")
plt.close(fig)

# fig-fold.png: the fold preset: warped checkerboard with the deformed grid, and det J
F = D["fold"]
W = 256
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.9), dpi=100)
axs[0].imshow(np.asarray(F["warped"]).reshape(W, W), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
for ln in F["lines"]:
    ln = np.asarray(ln)
    axs[0].plot(ln[:, 0], ln[:, 1], color="#ffcc00", lw=1.2)
for hx, hy in F["handles"]:
    axs[0].plot(hx, hy, "o", ms=9, mfc="#d62728", mec="white")
axs[0].set_title("moved ±44 px", fontsize=18)
axs[1].imshow(np.asarray(F["jrgb"]).reshape(W, W, 3), interpolation="nearest")
axs[1].set_title(f"det J (min {F['minJ']:.3f})".replace("-", "−"), fontsize=18)
for ax in axs:
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(W - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
fig.text(0.5, 0.025, "blue: compressed · red: stretched · black: fold", ha="center", fontsize=17)
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.12, wspace=0.05)
fig.savefig("spline-images/fig-fold.png", facecolor="white")
plt.close(fig)
print("wrote spline-images/fig-zoom.png fig-rotate10.png fig-fold.png")
