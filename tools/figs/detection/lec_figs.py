"""Slide figures of the detection/ lecture (modern deep learning day).

Run from the repository root:
  for p in rod anchor heat noisyheat crowd high; do tools/figs/instances/demo_shot.sh SHOTS nmsboxes $p nb_$p 1100; done
  python3 -I tools/figs/detection/lec_figs.py SHOTS

Demo panels are crops of the nmsboxes demo in its presets (the numbers in the labels are the
demo's readouts in these states). The box diagrams and the IoU-loss plot are computed here.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

OUT = "detection/"
SHOTS = sys.argv[1]
BLUE, ORANGE, RED, GREEN, GREY = "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9a9a9a"
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
CANVAS = (10, 26, 480, 496)          # the demo's image canvas, CSS px at 1220 px window width


def crop(shot, box=CANVAS):
    im = Image.open(f"{SHOTS}/{shot}.png").convert("RGB")
    return np.asarray(im.crop(tuple(2 * v for v in box)))


def panels(cells, out, size, fs=17):
    fig, axs = plt.subplots(1, len(cells), figsize=size, squeeze=False, gridspec_kw={"wspace": 0.05})
    for a, (img, title, sub) in zip(axs.flat, cells):
        a.imshow(img); a.axis("off")
        if title:
            a.set_title(title, fontsize=fs + 1)
        if sub:
            a.text(0.5, -0.02, sub, transform=a.transAxes, ha="center", va="top", fontsize=fs)
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return w * h / (area(a) + area(b) - w * h)


# 1. diagonal rod -------------------------------------------------------------------------------
panels([(crop("nb_rod"), "", "the rod fills 16 % of its box")], OUT + "fig-rod.png", (4.6, 4.9))

# 2. IoU of two boxes by hand ------------------------------------------------------------------
A = (0, 0, 4, 4)
pairs = [((2, 2, 6, 6), "B = [2,6] × [2,6]"), ((1, 0, 5, 4), "shifted right by 1"), ((1, 1, 5, 5), "right and down by 1")]
fig, axs = plt.subplots(1, 3, figsize=(10, 3.9))
for a, (B, lab) in zip(axs, pairs):
    w = max(0, min(A[2], B[2]) - max(A[0], B[0])); h = max(0, min(A[3], B[3]) - max(A[1], B[1]))
    inter = w * h; union = 32 - inter
    a.add_patch(Rectangle((A[0], A[1]), 4, 4, fc=BLUE, alpha=0.18, ec=BLUE, lw=2.5))
    a.add_patch(Rectangle((B[0], B[1]), 4, 4, fc=ORANGE, alpha=0.18, ec=ORANGE, lw=2.5, ls="--"))
    a.add_patch(Rectangle((max(A[0], B[0]), max(A[1], B[1])), w, h, fc="#555", alpha=0.45, lw=0))
    a.set_xlim(-0.3, 6.3); a.set_ylim(6.3, -0.3); a.set_aspect("equal")
    a.set_xticks(range(7)); a.set_yticks(range(7)); a.grid(alpha=0.35)
    a.tick_params(labelsize=13)
    a.set_title(f"{inter}/{union} = {inter / union:.2f}", fontsize=17)
    a.set_xlabel(lab, fontsize=15)
    a.text(0.15, 0.25, "A", color=BLUE, fontsize=17, va="top", fontweight="bold")
    a.text(B[2] - 0.15, B[3] - 0.15, "B", color="#b85a00", fontsize=17, ha="right", va="bottom", fontweight="bold")
fig.savefig(OUT + "fig-iou.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)

# 3. 1 − IoU as a loss: flat when the boxes do not overlap -------------------------------------
dx = np.linspace(0, 8, 321)
loss = [1 - iou(A, (d, 0, d + 4, 4)) for d in dx]
fig, ax = plt.subplots(figsize=(5.0, 3.4))
ax.plot(dx, loss, c=BLUE, lw=2.5)
ax.axvspan(4, 8, color=GREY, alpha=0.15, lw=0)
ax.text(6, 0.55, "no overlap:\nflat, no gradient", ha="center", fontsize=15)
ax.set_xlabel("shift of B (pixels)"); ax.set_ylabel("1 − IoU")
ax.set_xlim(0, 8); ax.set_ylim(0, 1.08)
fig.savefig(OUT + "fig-iouloss.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)

# 4. anchors; 5. heat maps; 7. NMS in a crowd ------------------------------------------------
panels([(crop("nb_anchor"), "", "1 of 5 objects matched")], OUT + "fig-anchors.png", (4.6, 4.9))
panels([(crop("nb_heat"), "clean heat map", "12 peaks for 12 cells"),
        (crop("nb_noisyheat"), "noisy heat map", "14 peaks for 9 cells")], OUT + "fig-heat.png", (8.6, 4.8), fs=22)
panels([(crop("nb_crowd"), "t = 0.2", "4 cells lost"),
        (crop("nb_high"), "t = 0.75", "34 duplicates")], OUT + "fig-crowd.png", (8.6, 4.8), fs=22)

# 6. greedy NMS on the five boxes of the slide ------------------------------------------------
boxes = [((0, 0, 10, 10), 0.9), ((1, 0, 11, 10), 0.8), ((8, 0, 18, 10), 0.7), ((9, 1, 19, 11), 0.6), ((20, 20, 26, 26), 0.3)]


def nms(t):
    order = sorted(range(len(boxes)), key=lambda i: -boxes[i][1])
    keep, rem = [], order[:]
    while rem:
        b = rem.pop(0); keep.append(b)
        rem = [c for c in rem if iou(boxes[b][0], boxes[c][0]) <= t]
    return keep


print("IoU b3,b4 = %.2f, b1,b3 = %.2f, b1,b2 = %.2f" % (iou(boxes[2][0], boxes[3][0]), iou(boxes[0][0], boxes[2][0]), iou(boxes[0][0], boxes[1][0])))
cols = [BLUE, "#56b4e9", ORANGE, "#cc79a7", GREEN]
fig, axs = plt.subplots(1, 3, figsize=(10, 3.9))
for a, t in zip(axs, [0.3, 0.7, 0.1]):
    k = nms(t)
    for i, ((x0, y0, x1, y1), sc) in enumerate(boxes):
        kept = i in k
        a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=cols[i] if kept else GREY,
                              lw=3 if kept else 1.5, ls="-" if kept else "--"))
        lx, ly, ha, va = [(x0, y0 - 0.4, "left", "bottom"), (x0, y1 + 0.4, "left", "top"),
                          (x1, y0 - 0.4, "right", "bottom"), (x1, y1 + 0.4, "right", "top"),
                          (x0, y0 - 0.4, "left", "bottom")][i]
        a.text(lx, ly, f"b{i + 1}", fontsize=19, color=cols[i] if kept else GREY, ha=ha, va=va,
               fontweight="bold" if kept else "normal")
    a.set_xlim(-1, 27); a.set_ylim(27, -4); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_color("#bbb")
    a.set_title(f"t = {t}\nkeep " + ", ".join(f"b{i + 1}" for i in sorted(k)), fontsize=20)
fig.savefig(OUT + "fig-nmsfive.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)
