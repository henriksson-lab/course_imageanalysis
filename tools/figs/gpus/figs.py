# Static figures for the GPU day (gpu-parallel/, gpu-memory/, gpu-patterns/, gpu-fit/).
# Run from the repo root:  python3 tools/figs/gpus/figs.py
# Numbers follow demos/src/lib/gpusim.ts (desktop GPU preset: 30e12 op/s, 1 TB/s).
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans"})
ORANGE, BLUE, RED, GREEN, GREY = "#ff9f1c", "#5b8ff9", "#d62728", "#2ca02c", "#888888"


def save(fig, path):
    fig.savefig(os.path.join(R, path), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 1. chip area: CPU vs GPU
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for a in ax:
    a.set_xlim(0, 10); a.set_ylim(0, 10); a.set_aspect("equal"); a.axis("off")
a = ax[0]
a.set_title("CPU: a few big cores", fontsize=18)
a.add_patch(Rectangle((0, 0), 10, 10, fc="#f3f3f3", ec="k"))
a.add_patch(Rectangle((0.4, 0.4), 9.2, 3.6, fc="#c5dbef", ec="k")); a.text(5, 2.2, "big cache", ha="center", va="center")
for i in range(4):
    for j in range(2):
        x, y = 0.4 + i * 2.35, 4.4 + j * 2.7
        a.add_patch(Rectangle((x, y), 2.1, 2.4, fc="#fde0b8", ec="k"))
        a.add_patch(Rectangle((x + 0.15, y + 0.15), 0.8, 0.8, fc=ORANGE, ec="k", lw=0.6))
        a.text(x + 1.5, y + 1.6, "control", ha="center", va="center", fontsize=10)
a = ax[1]
a.set_title("GPU: thousands of simple lanes", fontsize=18)
a.add_patch(Rectangle((0, 0), 10, 10, fc="#f3f3f3", ec="k"))
a.add_patch(Rectangle((0.4, 0.4), 9.2, 1.2, fc="#c5dbef", ec="k")); a.text(5, 1.0, "small caches", ha="center", va="center")
for i in range(4):
    for j in range(3):
        x, y = 0.4 + i * 2.35, 1.9 + j * 2.7
        a.add_patch(Rectangle((x, y), 2.1, 2.4, fc="#fde0b8", ec="k", lw=0.8))
        a.add_patch(Rectangle((x + 0.1, y + 1.95), 1.9, 0.35, fc="#c5dbef", ec="k", lw=0.4))
        for u in range(8):
            for v in range(4):
                a.add_patch(Rectangle((x + 0.12 + u * 0.235, y + 0.12 + v * 0.45), 0.19, 0.38, fc=ORANGE, ec="none"))
fig.text(0.5, 0.01, "orange = arithmetic, blue = cache / fast memory (schematic, not to scale)", ha="center", fontsize=14)
save(fig, "gpu-parallel/fig-chip.svg")

# ---------------------------------------------------------------- 2. divergence: one warp
fig, ax = plt.subplots(figsize=(12, 3.6))
ax.set_xlim(-6.5, 33); ax.set_ylim(-0.6, 3.4); ax.axis("off")
inside = [8 <= i < 13 for i in range(32)]
for i in range(32):
    ax.add_patch(Rectangle((i, 2.3), 0.9, 0.9, fc="#444" if not inside[i] else "#ddd", ec="k", lw=0.5))
ax.text(-0.5, 2.75, "32 pixels\n(one warp)", ha="right", va="center", fontsize=14)
for row, (lab, on, col, cyc) in enumerate([("path A: 100 cycles", True, ORANGE, 100), ("path B: 10 cycles", False, BLUE, 10)]):
    y = 1.1 - row * 1.2
    for i in range(32):
        ax.add_patch(Rectangle((i, y), 0.9, 0.9, fc=col if inside[i] == on else "#f2f2f2", ec="#999", lw=0.5))
    ax.text(-0.5, y + 0.45, lab, ha="right", va="center", fontsize=14)
ax.text(16, -0.75, "The warp runs both paths, one after the other: 110 cycles. Masked lanes (grey) wait.", ha="center", fontsize=14)
save(fig, "gpu-parallel/fig-divergence.svg")

# ---------------------------------------------------------------- 3. memory hierarchy
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.axis("off")
levels = [
    ("registers", "per lane", "~1 cycle", ""),
    ("shared memory / L1", "~100 kB per multiprocessor", "~30 cycles", "~10 TB/s (all)"),
    ("L2 cache", "a few MB", "~200 cycles", "~5 TB/s"),
    ("global memory", "8–80 GB", "~500 cycles", "~1 TB/s"),
    ("link to the CPU", "", "~10 µs per transfer", "~25 GB/s"),
    ("CPU main memory", "16–512 GB", "", ""),
]
for i, (n, size, lat, bw) in enumerate(levels):
    y = 5.6 - i * 1.0
    w = 3 + i * 0.9
    col = "#c5dbef" if i < 4 else ("#e8d1f0" if i == 4 else "#eeeeee")
    ax.add_patch(Rectangle((6 - w / 2 - 2.4, y - 0.4), w, 0.8, fc=col, ec="k"))
    ax.text(6 - 2.4, y, n, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(8.9, y + 0.12, size, ha="left", va="center", fontsize=13)
    ax.text(8.9, y - 0.22, "  ".join(t for t in (lat, bw) if t), ha="left", va="center", fontsize=13, color="#333")
ax.text(0.2, 6.25, "fast, small", fontsize=14, color=GREY)
ax.text(0.2, 0.05, "slow, large", fontsize=14, color=GREY)
ax.annotate("", xy=(0.5, 0.4), xytext=(0.5, 6.0), arrowprops=dict(arrowstyle="->", color=GREY, lw=2))
save(fig, "gpu-memory/fig-hierarchy.svg")

# ---------------------------------------------------------------- 4. roofline, desktop preset
peak, bw = 30e12, 1e12
I = np.logspace(-2, 3.5, 300)
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(I, np.minimum(peak, bw * I) / 1e12, "k", lw=3)
ax.axvline(peak / bw, color=GREY, ls="--", lw=1.5)
ax.text(peak / bw * 1.1, 0.012, "ridge I* = 30", color=GREY)
pts = [("add two images", 1 / 12), ("4 passes", 1 / 8), ("fused", 0.5), ("3×3 filter (tiled)", 2.25),
       ("11×11 filter", 121 / 4), ("FFT", 3.75), ("matrix product N = 1024", 1024 / 6)]
pos = {"add two images": (0.8, 2.0, "right"), "4 passes": (1.25, 0.45, "left"), "fused": (0.8, 2.0, "right"),
       "3×3 filter (tiled)": (0.8, 2.2, "right"), "11×11 filter": (0.8, 2.5, "right"), "FFT": (1.3, 0.5, "left"),
       "matrix product N = 1024": (1.0, 0.35, "center")}
for lab, x in pts:
    y = min(peak, bw * x) / 1e12
    ax.plot([x], [y], "o", color=RED if x < 30 else BLUE, ms=10)
    fx, fy, ha = pos[lab]
    ax.text(x * fx, y * fy, lab, ha=ha, fontsize=14)
ax.text(0.012, 50, "roof: peak 30·10¹² op/s", fontsize=14)
ax.text(40, 0.03, "slope: bandwidth 1 TB/s", fontsize=14, color=GREY)
ax.set_xlabel("arithmetic intensity I (operations per byte)")
ax.set_ylabel("attainable (10¹² op/s)")
ax.set_ylim(0.01, 3000); ax.set_xlim(0.01, 3000)
ax.text(0.1, 600, "memory-bound", color=RED, fontsize=16)
ax.text(60, 600, "compute-bound", color=BLUE, fontsize=16)
save(fig, "gpu-memory/fig-roofline.svg")

# ---------------------------------------------------------------- 5. reduction tree and Hillis–Steele ladder
def ladder(ax, rows, ops, title, labels):
    n = len(rows[0])
    ax.set_xlim(-1.6, n); ax.set_ylim(-len(rows) * 1.25, 1.0); ax.axis("off"); ax.set_title(title, fontsize=17)
    for k, row in enumerate(rows):
        y = -k * 1.25
        ax.text(-0.7, y + 0.35, labels[k], ha="right", va="center", fontsize=13, color=GREY)
        for i, v in enumerate(row):
            hit = k > 0 and any(t == i for t, _ in ops[k])
            ax.add_patch(Rectangle((i, y), 0.86, 0.7, fc="#ffd8a8" if hit else "#f6f7f9", ec="#999"))
            ax.text(i + 0.43, y + 0.35, str(v), ha="center", va="center", fontsize=15)
        if k > 0:
            for t, s in ops[k]:
                ax.annotate("", xy=(t + 0.43, y + 0.72), xytext=(s + 0.43, y + 1.25), arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

xs = [5, 2, 7, 1, 3, 8, 4, 6]
red_rows = [xs, [7, 2, 8, 1, 11, 8, 10, 6], [15, 2, 8, 1, 21, 8, 10, 6], [36, 2, 8, 1, 21, 8, 10, 6]]
red_ops = [[], [(0, 1), (2, 3), (4, 5), (6, 7)], [(0, 2), (4, 6)], [(0, 4)]]
v = [3, 1, 7, 0, 4, 1, 6, 3]
hs_rows = [v, [3, 4, 8, 7, 4, 5, 7, 9], [3, 4, 11, 11, 12, 12, 11, 14], [3, 4, 11, 11, 15, 16, 22, 25]]
hs_ops = [[], [(i, i - 1) for i in range(1, 8)], [(i, i - 2) for i in range(2, 8)], [(i, i - 4) for i in range(4, 8)]]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
ladder(ax[0], red_rows, red_ops, "reduction: 3 steps, 7 additions", ["in", "1", "2", "3"])
ladder(ax[1], hs_rows, hs_ops, "Hillis–Steele scan: 3 steps, 17 additions", ["in", "1", "2", "3"])
save(fig, "gpu-patterns/fig-reduce-scan.svg")

# ---------------------------------------------------------------- 6. Amdahl
fig, ax = plt.subplots(figsize=(11, 3.4))
stages = [("load (serial)", 2, GREY, 1), ("filter", 6, ORANGE, 50), ("label (serial)", 1, GREY, 1), ("measure", 1, BLUE, 1)]
for row, (title, speed) in enumerate([("before: 10 s", False), ("filter 50× faster: 4.12 s", True)]):
    x = 0
    y = 1 - row
    for name, t, col, f in stages:
        w = t / f if speed else t
        ax.barh(y, w, left=x, color=col, edgecolor="k", height=0.6)
        if w > 0.7:
            ax.text(x + w / 2, y, name, ha="center", va="center", fontsize=13)
        x += w
    ax.text(x + 0.15, y, title, va="center", fontsize=15)
ax.set_yticks([]); ax.set_xlim(0, 13.5); ax.set_xlabel("time (s)")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
save(fig, "gpu-fit/fig-amdahl.svg")
print("figures written")
