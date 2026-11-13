# Figures for box-filters/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/box-filters/figs.py
# Needs node and esbuild (node_modules/.bin/esbuild): the local-statistics data are
# computed with the course's demo code (tools/figs/box-filters/data.ts), so the numbers
# agree with the localstats demo.
import os, sys, json, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "integral-image"))
import numpy as np
from scipy import ndimage as ndi
import skimage.io as io
import matplotlib.pyplot as plt
from style import setup, fig, save, BLUE, ORANGE, GREEN, RED, GREY

setup()
OUT = "box-filters"
CREDIT = "Micrograph: Robert M. Hunt, CC BY 3.0"
cell = io.imread("demos/src/fourier2d/cell256.png").astype(float)
if cell.ndim == 3:
    cell = cell[..., 0]
N = 256

# --- data from the demo code ----------------------------------------------------------
tmp = tempfile.mkdtemp()
with open(os.path.join(tmp, "cell.json"), "w") as fh:
    json.dump(cell.ravel().tolist(), fh)
subprocess.run(["node_modules/.bin/esbuild", os.path.join(HERE, "data.ts"), "--bundle", "--platform=node",
                "--log-level=warning", "--outfile=" + os.path.join(tmp, "data.js")], check=True)
subprocess.run(["node", os.path.join(tmp, "data.js"), os.path.join(tmp, "cell.json"), os.path.join(tmp, "out.json")], check=True)
D = json.load(open(os.path.join(tmp, "out.json")))
im = lambda k: np.array(D[k]).reshape(N, N)


def save_png(f, path):
    f.savefig(path, dpi=144, bbox_inches="tight", pad_inches=0.04)
    plt.close(f)


# --- 1. additions per pixel against k -------------------------------------------------
k = np.arange(1, 102, 2)
fg, ax = fig(330, 330)
for lab, y, c in [("direct", k ** 2 - 1, RED), ("separable", 2 * (k - 1), ORANGE),
                  ("running sum", np.full_like(k, 4), GREEN), ("integral image", np.full_like(k, 6), BLUE)]:
    ax.semilogy(k, np.maximum(y, 1), color=c, lw=4, label=lab)
ax.set_xlabel("window width $k$", fontsize=25)
ax.set_ylabel("additions per pixel", fontsize=25)
ax.tick_params(labelsize=23)
ax.set_xticks([0, 50, 100])
ax.set_ylim(1, 3e4)
ax.set_yticks([1, 10, 100, 1000, 10000])
ax.set_yticklabels(["1", "10", "100", "1000", "10⁴"])
for lab, y, c, va in [("direct", 101 ** 2 - 1, RED, "center"), ("separable", 200, ORANGE, "center"),
                      ("integral image", 6, BLUE, "bottom"), ("running sum", 4, GREEN, "top")]:
    ax.text(103, y * (1.15 if va == "bottom" else 0.87 if va == "top" else 1), lab, color=c, va=va, fontsize=25)
ax.set_xlim(0, 101)
ax.axvline(51, color=GREY, lw=1, ls=":")
save(fg, f"{OUT}/fig-opcount.svg")

# --- 2. border rule: zeros vs clipped window ------------------------------------------
kk = 51
zero = ndi.uniform_filter(cell, kk, mode="constant", cval=0)
clip = zero / ndi.uniform_filter(np.ones_like(cell), kk, mode="constant", cval=0)
fg, axs = plt.subplots(1, 2, figsize=(5.4, 2.95))
for a, img, t in [(axs[0], zero, "pad with zeros"), (axs[1], clip, "clip the window")]:
    a.imshow(img, cmap="gray", vmin=0, vmax=255)
    a.set_title(t, fontsize=14)
    a.axis("off")
save_png(fg, f"{OUT}/fig-border.png")

# --- 3. local mean and SD of the micrograph (demo preset "cell", 9 x 9) -----------------
fg, axs = plt.subplots(1, 3, figsize=(7.2, 2.85))
for a, img, t, lim in [(axs[0], cell, "input $I$", (0, 255)), (axs[1], im("cellMean"), "local mean $m$", (0, 255)),
                       (axs[2], im("cellSd"), "local SD $s$", (0, np.percentile(im("cellSd"), 99.5)))]:
    a.imshow(img, cmap="gray", vmin=lim[0], vmax=lim[1])
    a.set_title(t, fontsize=16)
    a.axis("off")
save_png(fg, f"{OUT}/fig-localstats.png")

# --- 4. contrast normalisation of the synthetic image (preset "equalise") -------------
fg, axs = plt.subplots(1, 2, figsize=(5.4, 3.0))
axs[0].imshow(im("synth"), cmap="gray", vmin=0, vmax=255)
axs[0].set_title("input", fontsize=14)
axs[1].imshow(im("synthCn"), cmap="gray", vmin=-3, vmax=3)
axs[1].set_title("$(I - m)/(s + 4)$", fontsize=14)
for a in axs:
    a.axis("off")
save_png(fg, f"{OUT}/fig-normalise.png")

# --- 5. float32 tables on a bright image: SD error maps ---------------------------------
from matplotlib.colors import ListedColormap
neg, off = D["negative"], D["offset"]
vmax = neg["emax"]
cmap = plt.get_cmap("inferno").copy()
cmap.set_under("#ff2828")
fg, axs = plt.subplots(1, 2, figsize=(5.4, 3.25))
for a, d, t in [(axs[0], neg, "float32 tables"), (axs[1], off, f"offset {D['offsetValue']} subtracted")]:
    e = np.array(d["err"]).reshape(N, N)
    a.imshow(np.where(e < 0, -1, e), cmap=cmap, vmin=0, vmax=vmax)
    a.set_title(t, fontsize=14)
    a.set_xlabel(f"largest error {d['emax']:.3g}\nnegative: {d['negatives']}", fontsize=14)
    a.set_xticks([]); a.set_yticks([])
save_png(fg, f"{OUT}/fig-precision.png")
print("negative:", neg["emax"], neg["negatives"], "offset:", off["emax"], off["negatives"], D["offsetValue"])
