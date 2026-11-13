# Figures for label-measure/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/label-measure/figs.py
# Image, masks, labels and the measurement table come from the regionfilter demo's pipeline
# (data.ts, via node): the synthetic "mixed population" (seed 3), threshold 100, fill holes, 8.
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "label-connectivity"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from labstyle import setup, fig, save, save_png, run_ts, PIX, GRID, INK, BLUE, ORANGE, GREEN, RED, GREY, PURPLE, LABELS

setup()
OUT = "label-measure"
D = run_ts(HERE)
N = 256
img = np.array(D["image"], float).reshape(N, N)
truth = np.array(D["truth"]).reshape(N, N)
K = D["keep"]
mask = np.array(K["mask"]).reshape(N, N)
lab = np.array(K["labels"]).reshape(N, N)
rows = K["rows"]
KCOL = {"nucleus": "#0072B2", "rod": "#E69F00", "speck": "#999999", "pair": "#D55E00"}
KMARK = {"nucleus": "o", "rod": "s", "speck": "^", "pair": "D"}
KNAME = {"nucleus": "round nuclei", "rod": "rods", "speck": "specks", "pair": "merged pairs"}
MCM = ListedColormap(["white", PIX])


def label_rgb(L):
    """Label image to RGB with the label palette (0 = white)."""
    out = np.ones(L.shape + (3,))
    pal = np.array([[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] for c in LABELS[:7]])
    for l in np.unique(L):
        if l: out[L == l] = pal[(l - 1) % len(pal)]
    return out


def frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(GRID)


# --- 1. mask vs label image on a touching pair --------------------------------------------------
pair = [r for r in rows if r["kind"] == "pair"][0]
cx, cy, h = int(round(pair["cx"])), int(round(pair["cy"])), 30
sl = (slice(max(0, cy - h), cy + h), slice(max(0, cx - h), cx + h))
f, axs = plt.subplots(3, 1, figsize=(330 / 72, 560 / 72), gridspec_kw=dict(hspace=0.08))
axs[0].imshow(img[sl], cmap="gray", vmin=0, vmax=255)
axs[1].imshow(mask[sl], cmap=MCM, vmin=0, vmax=1, interpolation="nearest")
t = truth[sl].copy(); ids = [v for v in np.unique(t) if v]; tt = np.zeros_like(t)
for k, v in enumerate(ids): tt[t == v] = [1, 5, 3, 2][k % 4]
axs[2].imshow(label_rgb(tt), interpolation="nearest")
for ax, txt in zip(axs, ["image", "mask:\none blob", "labels:\n1 and 2"]):
    frame(ax); ax.text(1.06, 0.5, txt, transform=ax.transAxes, fontsize=21, va="center", ha="left")
save_png(f, f"{OUT}/fig-labelvsmask.png")

# --- 2. scatter: area vs solidity, coloured by the kind of object ---------------------------
f, ax = fig(520, 380)
for kind in ["speck", "rod", "nucleus", "pair"]:
    rs = [r for r in rows if r["kind"] == kind]
    ax.scatter([r["area"] for r in rs], [r["solidity"] for r in rs], s=70, marker=KMARK[kind], c=KCOL[kind],
               edgecolors="white", linewidths=0.8, label=f"{KNAME[kind]} ({len(rs)})", zorder=3)
ax.set_xscale("log"); ax.set_xlim(0.7, 1500); ax.set_ylim(0.7, 1.02)
ax.set_xticks([1, 10, 40, 100, 1000]); ax.set_xticklabels(["1", "10", "40", "100", "1000"])
ax.axvline(40, color=GREY, lw=1.3, ls="--"); ax.axhline(0.91, color=GREY, lw=1.3, ls="--")
ax.text(36, 0.705, "area 40", ha="right", fontsize=16, color="#555", va="bottom")
ax.text(1.0, 0.915, "solidity 0.91", fontsize=16, color="#555", va="bottom")
ax.set_xlabel("area (pixels, log scale)"); ax.set_ylabel("solidity")
ax.legend(loc="lower right", frameon=False, fontsize=16, bbox_to_anchor=(1.0, 0.04), handletextpad=0.2)
save(f, f"{OUT}/fig-scatter.svg")

# --- 3. area opening vs morphological opening -------------------------------------------------
a40 = np.array(D["area40"]).reshape(N, N)
o2 = np.array(D["open2"]["mask"]).reshape(N, N)
# a 64 x 64 window around a rod with specks nearby
best, bw = None, -1
for y0 in range(0, N - 64, 4):
    for x0 in range(0, N - 64, 4):
        inside = [r for r in rows if x0 + 6 < r["cx"] < x0 + 58 and y0 + 6 < r["cy"] < y0 + 58]
        sc = sum(1 for r in inside if r["kind"] == "speck") + 3 * sum(1 for r in inside if r["kind"] == "rod") + 2 * sum(1 for r in inside if r["kind"] == "nucleus")
        if sc > bw: best, bw = (y0, x0), sc
y0, x0 = best
sl = (slice(y0, y0 + 64), slice(x0, x0 + 64))
from scipy import ndimage as ndi
nlab = lambda m: int(ndi.label(m, structure=np.ones((3, 3)))[1])
RM = "#D55E00"
f, axs = plt.subplots(3, 1, figsize=(330 / 72, 560 / 72), gridspec_kw=dict(hspace=0.08))
for ax, m, t in [(axs[0], mask, "thresholded"), (axs[1], a40, "area opening\n≥ 40"), (axs[2], o2, "opening,\nradius 2")]:
    rgb = np.ones(m[sl].shape + (3,))
    rgb[m[sl] > 0] = [int(PIX[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    rgb[(mask[sl] > 0) & (m[sl] == 0)] = [int(RM[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    ax.imshow(rgb, interpolation="nearest")
    frame(ax); ax.text(1.06, 0.5, f"{t}\n{nlab(m)} objects", transform=ax.transAxes, fontsize=20, va="center", ha="left")
save_png(f, f"{OUT}/fig-filters.png")
print("filters window", best, "objects", nlab(mask), nlab(a40), nlab(o2))

# --- 4. the pipeline: counts after each step --------------------------------------------------
Fr = D["frame"]
labF = np.array(Fr["labels"]).reshape(N, N)
big = {r["label"] for r in rows if r["area"] >= 40}
solid = {r["label"] for r in rows if r["area"] >= 40 and r["solidity"] >= 0.91}
cnt = {r["label"] for r in Fr["rows"] if r["counted"] and r["area"] >= 40}
f, axs = plt.subplots(2, 2, figsize=(500 / 72, 520 / 72), gridspec_kw=dict(wspace=0.06, hspace=0.22))
axs = axs.ravel()
axs[0].imshow(img, cmap="gray", vmin=0, vmax=255); axs[0].set_xlabel("image", fontsize=22)
axs[1].imshow(label_rgb(lab), interpolation="nearest"); axs[1].set_xlabel(f"labels: {K['count']}", fontsize=22)
keep = np.where(np.isin(lab, list(big)), lab, 0)
axs[2].imshow(label_rgb(keep), interpolation="nearest"); axs[2].set_xlabel(f"area ≥ 40: {len(big)}", fontsize=22)
fk = np.where(np.isin(labF, list(cnt)), labF, 0)
rgb = label_rgb(fk); grey = np.isin(labF, list({r["label"] for r in Fr["rows"] if r["area"] >= 40} - cnt))
rgb[grey] = [0.78, 0.78, 0.78]
axs[3].imshow(rgb, interpolation="nearest")
m = 20
axs[3].add_patch(Rectangle((m - 0.5, m - 0.5), N - 2 * m, N - 2 * m, fill=False, ec=GREEN, lw=2))
axs[3].plot([N - m - 0.5, N - m - 0.5, -0.5], [-0.5, N - m - 0.5, N - m - 0.5], color=RED, lw=2.5)
axs[3].set_xlabel(f"frame: {len(cnt)}", fontsize=22)
for ax in axs: frame(ax)
save_png(f, f"{OUT}/fig-pipeline.png")
assert K["count"] == 41 and len(big) == 27 and len(solid) == 15 and len(cnt) == 19, (K["count"], len(big), len(solid), len(cnt))
