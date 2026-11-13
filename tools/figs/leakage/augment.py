# fig-augment.png: "augment, then split" with the course micrograph (Robert M. Hunt, CC BY 3.0, via Wikimedia Commons).
# Run: python3 -I tools/figs/leakage/augment.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import io

R = "/data/henriksson/github/course/imageanalysis/"
img = io.imread(R + "demos/src/fourier2d/cell256.png")
if img.ndim == 3:
    img = img[..., :3].mean(axis=2)
crop = img[40:168, 70:198]
copies = [("original", crop), ("rotated 90°", np.rot90(crop)), ("flipped", crop[:, ::-1]), ("rotated 180°", np.rot90(crop, 2))]
fig, axs = plt.subplots(2, 2, figsize=(4.2, 5.2), dpi=150)
axs = axs.ravel()
for k, (ax, (name, im)) in enumerate(zip(axs, copies)):
    ax.imshow(im, cmap="gray", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    col = "#d62728" if k == 0 else "#1f77b4"
    for sp in ax.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(4)
    ax.set_title(name, fontsize=14)
    ax.set_xlabel("test" if k == 0 else "training", fontsize=14, color=col, weight="bold")
fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.12, wspace=0.08, hspace=0.5)
fig.savefig(R + "leakage/fig-augment.png", facecolor="white", bbox_inches="tight", pad_inches=0.08)
print("wrote fig-augment.png")
