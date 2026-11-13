# fig-annotators.png: two annotators' outlines on the synthetic nuclei of the objmatch demo.
# Input: the JSON written by figs.ts. Run: python3 -I tools/figs/agreement/annotators.py <tmp>/agreement-scene.json
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

d = json.load(open(sys.argv[1]))
W = d["W"]
img = np.array(d["image"], float).reshape(W, W)
A = np.array(d["a"]).reshape(W, W)
B = np.array(d["b"]).reshape(W, W)
out = "/data/henriksson/github/course/imageanalysis/agreement/fig-annotators.png"
# the nucleus that only A drew: an A object with no overlap with any B object
only_a = [k for k in range(1, d["nA"] + 1) if not (B[A == k] > 0).any()]
# crop the left 60 % of the image around the disputed nucleus so outlines are large
ys, xs = np.nonzero(A == only_a[0])
cy, cx = int(ys.mean()), int(xs.mean())
h = 100
y0, x0 = max(0, min(W - 2 * h, cy - h)), max(0, min(W - 2 * h, cx - h))
sl = (slice(y0, y0 + 2 * h), slice(x0, x0 + 2 * h))

fig = plt.figure(figsize=(4.6, 5.2), dpi=170)
ax = fig.add_axes([0, 0.14, 1, 0.86])
ax.imshow(img[sl], cmap="gray", vmin=0, vmax=230, interpolation="nearest")
for k in range(1, d["nA"] + 1):
    ax.contour((A[sl] == k).astype(float), levels=[0.5], colors="#56B4E9", linewidths=1.9)
for k in range(1, d["nB"] + 1):
    ax.contour((B[sl] == k).astype(float), levels=[0.5], colors="#E69F00", linewidths=1.9, linestyles="dashed")
ys, xs = np.nonzero(A[sl] == only_a[0])
ax.annotate("only A", (xs.mean(), ys.min() - 2), xytext=(xs.mean(), max(8, ys.min() - 22)), color="white", fontsize=14,
            ha="center", arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
ax.set_axis_off()
fig.legend(handles=[Line2D([0], [0], color="#56B4E9", lw=2.4, label="annotator A"),
                    Line2D([0], [0], color="#E69F00", lw=2.4, ls="--", label="annotator B")],
           loc="lower center", ncol=2, fontsize=14, frameon=False)
fig.savefig(out, facecolor="white")
print("wrote", out, "only A:", only_a, "crop", y0, x0, "F1 B vs A @0.5", round(d["f1"], 3))
