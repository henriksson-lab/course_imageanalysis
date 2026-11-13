# fig-matchscene.png: the objmatch demo scene ("a typical result", tau = 0.5) with the matching drawn on it.
# Input: the JSON written by figs.ts. Run: python3 -I tools/figs/object-eval/scene.py <tmp>/objeval-scene.json
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

d = json.load(open(sys.argv[1]))
W = d["W"]
img = np.array(d["image"], float).reshape(W, W)
truth = np.array(d["truth"]).reshape(W, W)
pred = np.array(d["pred"]).reshape(W, W)
matchT, matchP = d["matchT"], d["matchP"]
out = "/data/henriksson/github/course/imageanalysis/object-eval/fig-matchscene.png"

fig = plt.figure(figsize=(5.0, 5.6), dpi=160)
ax = fig.add_axes([0, 0.15, 1, 0.85])
ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
rgba = np.zeros((W, W, 4))
for k, m in enumerate(matchP):
    col = (0.17, 0.63, 0.17) if m >= 0 else (0.84, 0.15, 0.16)
    rgba[pred == k + 1] = (*col, 0.45)
ax.imshow(rgba, interpolation="nearest")
for k, m in enumerate(matchP):
    if m < 0:  # small FP objects: ring them so they are visible
        ys, xs = np.nonzero(pred == k + 1)
        ax.add_patch(plt.Circle((xs.mean(), ys.mean()), 9, fill=False, ec="#d62728", lw=2.2))
for k, m in enumerate(matchT):
    mask = (truth == k + 1).astype(float)
    if m >= 0:
        ax.contour(mask, levels=[0.5], colors="white", linewidths=1.3)
    else:
        ax.contour(mask, levels=[0.5], colors="#ff7f0e", linewidths=2.6, linestyles="dashed")
ax.set_axis_off()
fig.legend(handles=[Patch(facecolor=(0.17, 0.63, 0.17, 0.6), label=f"TP {d['tp']}"),
                    Patch(facecolor=(0.84, 0.15, 0.16, 0.6), label=f"FP {d['fp']}"),
                    Line2D([0], [0], color="#ff7f0e", lw=2.6, ls="--", label=f"FN {d['fn']}"),
                    Line2D([0], [0], color="0.4", lw=1.3, label="true outline")],
           loc="lower center", ncol=2, fontsize=13, frameon=False, handlelength=1.6, columnspacing=1.2)
fig.savefig(out, facecolor="white")
print("wrote", out, "TP", d["tp"], "FP", d["fp"], "FN", d["fn"], "F1", round(d["f1"], 3))
