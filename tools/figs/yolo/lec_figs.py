"""Slide and exercise figures of the yolo/ unit (Networks for cells and objects day).

Run from the repository root:  python3 -I tools/figs/yolo/lec_figs.py
(demo screenshots: tools/figs/yolo/demo_shots.py; measured readouts: measure_yolonet.py).
Every number drawn here is computed here (the by-hand numbers are also in numbers.py).
The nuclei image (grids figure) is the StarDist test image from `make models`
(demos/dist/models/images/stardist-img2d.png, BSD-3-Clause); without it a synthetic image is used.
fig-anchors.png is the detection/ unit's figure (tools/figs/detection/lec_figs.py),
copied here unchanged.
"""
import math, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = os.environ.get("REPO_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.environ.get("YOLO_FIGS") or os.path.join(ROOT, "yolo")
S8, S16, S32 = "#e69f00", "#56b4e9", "#009e73"        # stride colours, as in the demos
TRUTH, PRED, GREY, RED = "#d6339a", "#1f4e9c", "#9a9a9a", "#d62728"
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "yolo"})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white", metadata={"Date": None} if name.endswith(".svg") else None)
    plt.close(fig)


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def giou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    C = (max(a[2], b[2]) - min(a[0], b[0])) * (max(a[3], b[3]) - min(a[1], b[1]))
    return i / u - (C - u) / C


def box(ax, b, **kw):
    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, **kw))


# ---- 1. three grids over a 512 × 512 input --------------------------------------------------
def fig_grids():
    p = os.path.join(ROOT, "demos/dist/models/images/stardist-img2d.png")
    if os.path.exists(p):
        from PIL import Image
        img = np.asarray(Image.open(p).convert("L"), dtype=float)
    else:
        y, x = np.mgrid[0:512, 0:512]; img = 40 + 150 * (np.sin(x / 23) * np.sin(y / 31) > 0.6)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw={"wspace": 0.08})
    for ax, s, col in zip(axs, (8, 16, 32), (S8, S16, S32)):
        ax.imshow(img, cmap="gray", extent=(0, 512, 512, 0))
        for k in range(0, 513, s):
            ax.plot([k, k], [0, 512], color=col, lw=0.5 if s == 8 else 0.9, alpha=0.8)
            ax.plot([0, 512], [k, k], color=col, lw=0.5 if s == 8 else 0.9, alpha=0.8)
        g = 512 // s
        ax.set_title(f"stride {s}: {g} × {g} = {g * g}", fontsize=17, color="black")
        ax.set_xlim(0, 512); ax.set_ylim(512, 0); ax.axis("off")
        for sp in ax.spines.values(): sp.set_color(col)
    save(fig, "fig-grids.png")
    from PIL import Image
    p = os.path.join(OUT, "fig-grids.png")
    Image.open(p).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(p, optimize=True)


# ---- 2. one output cell decoded (exercise B1) --------------------------------------------------
def fig_cell():
    s, i, j = 16, 5, 7
    cx, cy = (j + 0.5) * s, (i + 0.5) * s
    d = (2.5, 1.0, 1.5, 2.0)
    b = (cx - d[0] * s, cy - d[1] * s, cx + d[2] * s, cy + d[3] * s)
    t = (84, 70, 150, 122)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for k in range(0, 225, s):
        ax.plot([k, k], [32, 160], color=S16, lw=0.8, alpha=0.6)
    for k in range(32, 161, s):
        ax.plot([48, 208], [k, k], color=S16, lw=0.8, alpha=0.6)
    ax.add_patch(Rectangle((j * s, i * s), s, s, color=S16, alpha=0.45))
    box(ax, t, ec=TRUTH, lw=2.5, ls="--")
    box(ax, b, ec=PRED, lw=3)
    ends = [(b[0], cy), (cx, b[1]), (b[2], cy), (cx, b[3])]
    names = ["left 2.5", "top 1", "right 1.5", "bottom 2"]
    offs = [(-2, -5, "right", "bottom"), (3, -1, "left", "bottom"), (2, -5, "left", "bottom"), (3, 2, "left", "top")]
    for (x, y), n, (dx, dy, ha, va) in zip(ends, names, offs):
        ax.add_patch(FancyArrowPatch((cx, cy), (x, y), arrowstyle="-|>", mutation_scale=16, color="black", lw=2))
        tx = (cx + x) / 2 + dx if n.startswith(("top", "bottom")) else x + dx
        ty = (cy + y) / 2 + dy if n.startswith(("top", "bottom")) else y + dy
        ax.text(tx, ty, n, ha=ha, va=va, fontsize=14)
    ax.plot(cx, cy, "o", color="white", mec="black", ms=9)
    ax.text(cx + 3, cy + 4, "(120, 88)", fontsize=14, va="top")
    ax.text(b[0], b[3] + 4, "decoded (80, 72)–(144, 120)", color=PRED, fontsize=14, va="top")
    ax.text(t[0] + 2, t[1] - 3, "true (84, 70)–(150, 122)", color=TRUTH, fontsize=14, va="bottom")
    ax.set_title(f"stride 16, row 5, column 7: IoU with the true box {iou(b, t):.2f}", fontsize=16)
    ax.set_xlim(44, 212); ax.set_ylim(160, 32); ax.set_aspect("equal")
    ax.set_xticks([48, 80, 112, 144, 176, 208]); ax.set_yticks([32, 64, 96, 128, 160])
    ax.tick_params(labelsize=12)
    save(fig, "fig-cell.svg")


# ---- 3. distances as distributions ------------------------------------------------------------
def fig_dist():
    cases = [("half on 2, half on 3", {2: .5, 3: .5}), ("sharp: all on 1", {1: 1.0}),
             ("two peaks: 0.6 on 2, 0.4 on 9", {2: .6, 9: .4})]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.3), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (title, m) in zip(axs, cases):
        p = np.zeros(16)
        for k, v in m.items(): p[k] = v
        mean = float((np.arange(16) * p).sum())
        ax.bar(np.arange(16), p, color=S16, width=0.8)
        ax.axvline(mean, color="black", lw=2.5)
        ax.text(mean + 0.3, 0.95, f"mean {mean:g}", fontsize=15, va="top")
        ax.set_title(title, fontsize=15); ax.set_xticks([0, 5, 10, 15]); ax.set_ylim(0, 1.05)
        ax.set_xlabel("distance (strides)", fontsize=14)
    axs[0].set_ylabel("probability", fontsize=14)
    save(fig, "fig-dist.svg")


# ---- 4. the pipeline with the measured numbers of the bus photo ---------------------------------
def fig_pipeline():
    steps = [("image", "512 × 512"), ("network", "one pass"), ("5376 cells", "4 × 16 bins\n+ 80 scores each"),
             ("score > 0.25", "45 candidates"), ("NMS", "IoU > 0.45,\nsame class"), ("list", "5 boxes")]
    fig, ax = plt.subplots(figsize=(13, 2.3))
    for k, (a, b) in enumerate(steps):
        x = k * 2.2
        ax.add_patch(Rectangle((x, 0), 1.75, 1.2, fc="#eef3fa" if k % 2 == 0 else "#fff5e6", ec="#555", lw=1.5))
        ax.text(x + 0.875, 0.88, a, ha="center", va="center", fontsize=16, weight="bold")
        ax.text(x + 0.875, 0.38, b, ha="center", va="center", fontsize=13)
        if k < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.8, 0.6), (x + 2.15, 0.6), arrowstyle="-|>", mutation_scale=18, color="#333"))
    ax.set_xlim(-0.1, 13.1); ax.set_ylim(-0.1, 1.3); ax.axis("off")
    save(fig, "fig-pipeline.svg")


# ---- 5. NMS with classes (exercise B3) ----------------------------------------------------------
B3 = [("b1", (10, 10, 50, 50), .90, "cell"), ("b2", (14, 12, 54, 52), .80, "cell"), ("b3", (40, 10, 80, 50), .75, "cell"),
      ("b4", (12, 10, 52, 50), .60, "debris"), ("b5", (44, 14, 84, 54), .50, "cell"), ("b6", (100, 100, 120, 120), .20, "cell")]


def greedy(items, t, agnostic, conf=0.25):
    alive = [x for x in sorted(items, key=lambda x: -x[2]) if x[2] > conf]
    kept = []
    while alive:
        k = alive.pop(0); kept.append(k[0])
        alive = [x for x in alive if not ((agnostic or x[3] == k[3]) and iou(x[1], k[1]) > t)]
    return kept


def fig_nmsclasses():
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"wspace": 0.06})
    lab = {"b1": (10, 7, "bottom", "left"), "b2": (14, 55, "top", "left"), "b3": (80, 7, "bottom", "right"),
           "b4": (31, 31, "center", "center"), "b5": (84, 57, "top", "right")}
    for ax, agn in zip(axs, (False, True)):
        kept = greedy(B3, 0.5, agn)
        for name, b, sc, cl in B3[:5]:
            on = name in kept
            col = (PRED if cl == "cell" else "#c45f00") if on else GREY
            box(ax, b, ec=col, lw=3.5 if on else 1.6, ls="-" if on else "--")
            x, y, va, ha = lab[name]
            ax.text(x, y, f"{name} {cl} {sc:.2f}", fontsize=14, color=col, va=va, ha=ha, weight="bold" if on else "normal")
        ax.set_title(("class-aware" if agn is False else "class-agnostic") + ": kept " + ", ".join(kept), fontsize=17)
        ax.set_xlim(4, 90); ax.set_ylim(64, 0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.5, 0.02, "score threshold 0.25 removes b6 (0.20, far away); NMS removes a box with IoU > 0.5", ha="center", fontsize=15)
    save(fig, "fig-nmsclasses.svg")


# ---- 6. GIoU (math slide) ----------------------------------------------------------------------
def fig_giou():
    A = (0, 0, 4, 4)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.25})
    B = (6, 0, 10, 4)
    a0.add_patch(Rectangle((0, 0), 10, 4, fc="#f3d9e8", ec=TRUTH, lw=1.5, ls="--", hatch="//"))
    a0.add_patch(Rectangle((0, 0), 4, 4, fc="#dbe7f7", ec=PRED, lw=2.5))
    a0.add_patch(Rectangle((6, 0), 4, 4, fc="#dbe7f7", ec=PRED, lw=2.5))
    a0.text(2, 2, "A", ha="center", va="center", fontsize=20); a0.text(8, 2, "B", ha="center", va="center", fontsize=20)
    a0.text(5, 2, "gap\n8 of 40", ha="center", va="center", fontsize=14, color=TRUTH)
    a0.set_title(f"IoU 0, GIoU = 0 − 8/40 = {giou(A, B):.2f}", fontsize=16)
    a0.set_xlim(-0.3, 10.3); a0.set_ylim(4.5, -0.5); a0.set_aspect("equal"); a0.axis("off")
    x = np.linspace(-2, 12, 561)
    a1.plot(x, [1 - iou(A, (v, 0, v + 4, 4)) for v in x], color=GREY, lw=3, label="1 − IoU")
    a1.plot(x, [1 - giou(A, (v, 0, v + 4, 4)) for v in x], color=PRED, lw=3, label="1 − GIoU")
    a1.plot([6, 5], [1 - giou(A, (6, 0, 10, 4)), 1 - giou(A, (5, 0, 9, 4))], "o", color=TRUTH, ms=8)
    a1.set_xlabel("left edge of B (A at 0)"); a1.set_ylabel("loss"); a1.legend(frameon=False, fontsize=14, loc="lower right")
    a1.set_ylim(-0.05, 1.8)
    save(fig, "fig-giou.svg")


# ---- 7. which cells learn an object (training assignment) ----------------------------------------
def fig_assign():
    t = (84, 70, 150, 122)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.1), gridspec_kw={"wspace": 0.08})
    for ax, s, col in zip(axs, (8, 16, 32), (S8, S16, S32)):
        n = 0
        for i in range(0, 224 // s):
            for j in range(0, 224 // s):
                cx, cy = (j + .5) * s, (i + .5) * s
                l, tt, r, b = cx - t[0], cy - t[1], t[2] - cx, t[3] - cy
                if min(l, tt, r, b) > 0:
                    n += 1
                    c = math.sqrt(min(l, r) / max(l, r) * min(tt, b) / max(tt, b))
                    ax.add_patch(Rectangle((j * s, i * s), s, s, color=col, alpha=0.15 + 0.75 * c, lw=0))
                    ax.plot(cx, cy, ".", color="black", ms=3)
        for k in range(0, 225, s):
            ax.plot([k, k], [0, 224], color=col, lw=0.6, alpha=0.6); ax.plot([0, 224], [k, k], color=col, lw=0.6, alpha=0.6)
        box(ax, t, ec=TRUTH, lw=2.5, ls="--")
        ax.set_title(f"stride {s}: {n} cells inside", fontsize=16)
        ax.set_xlim(40, 200); ax.set_ylim(170, 30); ax.set_aspect("equal"); ax.axis("off")
    fig.text(0.5, 0.0, "cells whose centre lies in the box; darker = more central (centreness)", ha="center", fontsize=15)
    save(fig, "fig-assign.svg")


# ---- 8. distribution-loss target (math slide) --------------------------------------------------
def fig_dfl():
    fig, ax = plt.subplots(figsize=(5.5, 3))
    p = np.zeros(16); p[2], p[3] = 0.7, 0.3
    ax.bar(np.arange(16), p, color=S16, width=0.8)
    ax.axvline(2.3, color="black", lw=2.5); ax.text(2.6, 0.9, "target 2.3 = mean", fontsize=14)
    ax.set_xticks([0, 2, 3, 5, 10, 15]); ax.set_ylim(0, 1); ax.set_xlabel("bin (strides)", fontsize=14)
    ax.set_ylabel("probability", fontsize=14)
    save(fig, "fig-dfl.svg")


def copy_detection():
    for f in ("fig-anchors.png",):
        shutil.copy(os.path.join(ROOT, "detection", f), os.path.join(OUT, f))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_grids(); fig_cell(); fig_dist(); fig_pipeline(); fig_nmsclasses(); fig_giou(); fig_assign(); fig_dfl(); copy_detection()
    print("written to", OUT)
