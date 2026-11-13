# fig-hierarchy.png (split by image vs split by day) and fig-domain.png (one image per day, accuracy per split)
# from the leakage demo data. Run: python3 -I tools/figs/leakage/thumbs.py <tmp>/leakage-set.json
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

d = json.load(open(sys.argv[1]))
S = d["S"]
imgs = [np.array(a).reshape(S, S) for a in d["images"]]
days = d["days"]
lo, hi = np.percentile(np.concatenate([i.ravel() for i in imgs]), [1, 99.5])
ROOT = "/data/henriksson/github/course/imageanalysis/leakage/"
RED, BLUE = "#d62728", "#1f77b4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})

# ---- hierarchy: 4 days x 3 images; row 1 split by image (3 test images from different days), row 2 by day
fig = plt.figure(figsize=(7.2, 3.5), dpi=150)
test_rows = [{1, 5, 10}, {6, 7, 8}]
titles = ["split by image: test images share days with training", "split by day: a whole day is held out"]
w = 0.068; h = w * 7.2 / 3.5
for r in range(2):
    y0 = 0.6 - r * 0.47
    fig.text(0.5, y0 + h + 0.035, titles[r], ha="center", fontsize=14.5, weight="bold")
    for i in range(12):
        dd, k = days[i], i % 3
        ax = fig.add_axes([0.03 + dd * 0.245 + k * 0.075, y0, w, h])
        ax.imshow(imgs[i], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        col = RED if i in test_rows[r] else BLUE
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(3.5)
    for dd in range(4):
        fig.text(0.03 + dd * 0.245 + 0.109, y0 - 0.075, f"day {dd + 1}", ha="center", fontsize=13.5)
fig.savefig(ROOT + "fig-hierarchy.png", facecolor="white", bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

# ---- domain: one image per day, and the accuracy of each split
fig = plt.figure(figsize=(4.8, 5.4), dpi=150)
for dd in range(4):
    i = days.index(dd)
    ax = fig.add_axes([0.02 + dd * 0.245, 0.74, 0.225, 0.225 * 4.8 / 5.4])
    ax.imshow(imgs[i], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"day {dd + 1}", fontsize=14)
ax = fig.add_axes([0.25, 0.1, 0.72, 0.5])
labels = ["pixels", "blocks", "blocks +\nbuffer", "image", "day"]
acc = [100 * a["mean"] for a in d["acc"]]
y = np.arange(5)[::-1]
ax.barh(y, acc, color=["#9ecae1"] * 4 + [BLUE], height=0.62)
ax.axvline(100 * d["maj"], color="#555", ls="--", lw=1.6)
ax.text(100 * d["maj"] + 0.4, 4.95, f"always 'background' {100 * d['maj']:.1f} %", ha="left", va="center", fontsize=13, color="#333")
ax.set_ylim(-0.5, 5.3)
for yy, a in zip(y, acc):
    ax.text(a + 0.3, yy, f"{a:.1f}", va="center", fontsize=14)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=14)
ax.set_xlim(75, 101); ax.set_xlabel("test accuracy (%), split by …", fontsize=14); ax.tick_params(axis="x", labelsize=14)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
fig.savefig(ROOT + "fig-domain.png", facecolor="white", bbox_inches="tight", pad_inches=0.08)
print("acc", [round(a, 1) for a in acc], "majority", round(100 * d["maj"], 1), "strong day effect, by day", round(100 * d["acc2"][4]["mean"], 1))
