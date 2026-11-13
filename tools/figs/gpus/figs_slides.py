# More slide figures for the GPU day (gpu-parallel/, gpu-memory/, gpu-patterns/, gpu-fit/).
# Run from the repo root:  python3 -I tools/figs/gpus/figs_slides.py
# Numbers follow the slides and demos/src/lib/gpusim.ts (desktop GPU: 30e12 op/s, 1 TB/s,
# link 25 GB/s, 10 us per launch); the pixel data of the parscan and parlabel demos are
# regenerated here with the same generators.
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np
from scipy import ndimage as ndi

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
plt.rcParams.update({"font.size": 18, "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
# Okabe-Ito based, colour-blind safe
ORANGE, BLUE, VERM, GREEN, GREY, SKY, PURPLE = "#E69F00", "#0072B2", "#D55E00", "#009E73", "#777777", "#56B4E9", "#CC79A7"
LIGHT = "#f2f2f2"


def save(fig, path):
    fig.savefig(os.path.join(R, path), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def cells(ax, row, y, x0=0.0, w=0.9, h=0.8, fc=None, fs=16, ec="#888"):
    for i, v in enumerate(row):
        c = fc[i] if isinstance(fc, list) else (fc or "#f6f7f9")
        ax.add_patch(Rectangle((x0 + i, y), w, h, fc=c, ec=ec, lw=0.8))
        if v is not None and v != "":
            ax.text(x0 + i + w / 2, y + h / 2, str(v), ha="center", va="center", fontsize=fs)


def arrow(ax, xy, xytext, col=VERM, lw=1.4):
    ax.annotate("", xy=xy, xytext=xytext, arrowprops=dict(arrowstyle="->", color=col, lw=lw, shrinkA=0, shrinkB=0))


def lcg(seed):
    s = [seed]
    def u():
        s[0] = (s[0] * 1103515245 + 12345) & 0xFFFFFFFF
        return (s[0] >> 8) / 16777216
    return u


# =============================================================== gpu-parallel
# P1. latency ladder
fig, ax = plt.subplots(figsize=(4.0, 3.0))
items = [("one instruction", 1, "0.3 ns"), ("cache hit", 4, "~1 ns"), ("main memory read", 300, "100 ns")]
for i, (lab, c, t) in enumerate(items):
    ax.barh(2 - i, c, color=[GREEN, BLUE, VERM][i], height=0.6)
    ax.text(c * 1.25, 2 - i, f"{c} cycle{'s' if c > 1 else ''} ({t})", va="center", fontsize=17)
ax.set_yticks([2, 1, 0], [x[0] for x in items])
ax.set_xscale("log"); ax.set_xlim(0.5, 8000)
ax.set_xlabel("cycles at 3·10⁹ per second (log scale)")
save(fig, "gpu-parallel/fig-latency.svg")

# P2. break-even
n = np.logspace(3, 7, 200)
cpu = 0.01 + n / 2e8 * 1e3
gpu = 0.5 + n / 1e10 * 1e3
fig, ax = plt.subplots(figsize=(4.6, 3.9))
ax.loglog(n, cpu, color=ORANGE, lw=3, label="CPU: t₀ = 0.01 ms, 2·10⁸ px/s")
ax.loglog(n, gpu, color=BLUE, lw=3, label="GPU: t₀ = 0.5 ms, 10¹⁰ px/s")
ax.axvline(1e5, color=GREY, ls="--", lw=1.5)
ax.text(1.15e5, 0.012, "n* = 100 000\n(316 × 316)", fontsize=17, color="#444")
ax.text(1.4e3, 2.2, "CPU faster", fontsize=17, color=ORANGE)
ax.text(4e5, 0.09, "GPU faster", fontsize=17, color=BLUE)
ax.set_xlabel("pixels n"); ax.set_ylabel("time (ms)")
ax.set_ylim(0.008, 100)
ax.legend(fontsize=17, loc="upper left", frameon=False)
save(fig, "gpu-parallel/fig-breakeven.svg")

# P3. fragment stage: one program per covered pixel
fig, ax = plt.subplots(figsize=(4.4, 3.6))
W, H = 14, 10
tri = np.array([[1.2, 1.0], [12.6, 2.6], [5.0, 9.2]])
from matplotlib.path import Path
pth = Path(tri)
for y in range(H):
    for x in range(W):
        inside = pth.contains_point((x + 0.5, y + 0.5))
        ax.add_patch(Rectangle((x, y), 1, 1, fc=ORANGE if inside else "white", ec="#bbb", lw=0.6))
ax.add_patch(Polygon(tri, fill=False, ec="k", lw=2.2))
ax.set_xlim(0, W); ax.set_ylim(-1.6, H); ax.set_aspect("equal"); ax.axis("off")
ncov = sum(pth.contains_point((x + 0.5, y + 0.5)) for y in range(H) for x in range(W))
ax.text(W / 2, -0.8, f"{ncov} pixels: one program, {ncov} times", ha="center", fontsize=17)
save(fig, "gpu-parallel/fig-fragment.svg")

# P4. SIMT hierarchy: image -> blocks -> warps -> threads
fig, ax = plt.subplots(figsize=(5.2, 4.5))
ax.set_xlim(-0.2, 16.4); ax.set_ylim(-0.4, 14.2); ax.set_aspect("equal"); ax.axis("off")
# image of 4 x 8 blocks (each block 32 x 8 pixels), drawn 2 units per 32 pixels
bw_, bh_ = 2.0, 0.5 * 2
for i in range(5):
    ax.plot([i * bw_, i * bw_], [8, 8 + 8 * bh_ * 0.75], color="#999", lw=0.8)
for j in range(9):
    ax.plot([0, 4 * bw_], [8 + j * bh_ * 0.75] * 2, color="#999", lw=0.8)
ax.add_patch(Rectangle((2, 8 + 5 * 0.75), bw_, 0.75, fc=ORANGE, ec="k", lw=1.5))
ax.text(8.6, 11.6, "image, cut into\nblocks of 32 × 8\npixels", fontsize=18, va="center")
# zoomed block: 8 warps (rows) of 32 lanes
x0, y0, cw = 0, 0.6, 0.5
for r in range(8):
    for c in range(32):
        ax.add_patch(Rectangle((x0 + c * cw, y0 + r * 0.8), cw, 0.66, fc="#fbe3b3" if r != 7 else BLUE, ec="white", lw=0.4))
ax.plot([2, 0], [8 + 5 * 0.75, y0 + 8 * 0.8], color=GREY, lw=1, zorder=0); ax.plot([4, 16], [8 + 5 * 0.75, y0 + 8 * 0.8], color=GREY, lw=1, zorder=0)
ax.text(8, -0.15, "one block: 256 threads = 8 warps", ha="center", va="top", fontsize=18)
ax.text(8, y0 + 8 * 0.8 + 0.25, "one warp: 32 lanes, one instruction", ha="center", va="bottom", fontsize=17, color=BLUE, bbox=dict(fc="white", ec="none", pad=1))
save(fig, "gpu-parallel/fig-simt.svg")

# P5. slowest lane
it = [3, 1, 1, 12, 2, 1, 4, 1]
fig, ax = plt.subplots(figsize=(5.2, 3.7))
ax.bar(range(1, 9), [12] * 8, color=LIGHT, edgecolor="#bbb", label="lane idle (masked)")
ax.bar(range(1, 9), it, color=BLUE, label="useful iterations")
ax.axhline(12, color=VERM, ls="--", lw=2)
ax.text(8.6, 12, " warp: 12", color=VERM, ha="left", va="center", fontsize=17)
ax.axhline(np.mean(it), color=GREEN, ls=":", lw=2)
ax.text(8.6, np.mean(it), " mean 3.1", color=GREEN, ha="left", va="center", fontsize=17)
ax.set_xticks(range(1, 9)); ax.set_xlabel("lane"); ax.set_ylabel("loop iterations")
ax.set_ylim(0, 15.5); ax.set_xlim(0.4, 8.6)
ax.legend(fontsize=16, loc="upper left", frameon=False, ncol=1)
ax.set_title("25 of 96 lane-iterations useful: 26 %", fontsize=18)
save(fig, "gpu-parallel/fig-slowest.svg")

# P6. occupancy: busy fraction vs warps
N = np.arange(1, 49)
fig, ax = plt.subplots(figsize=(4.6, 3.6))
for k, col in [(8, BLUE), (4, ORANGE)]:
    U = np.minimum(1, N * k / (k + 200))
    ax.plot(N, 100 * U, "-o", ms=3.5, color=col, lw=2.5)
ax.axvline(26, color=GREY, ls="--", lw=1.4)
ax.text(27, 15, "26 warps\nhide L = 200", fontsize=17, color="#444")
ax.text(36, 102, "k = 8", fontsize=17, color=BLUE, va="bottom")
ax.text(33, 58, "k = 4", fontsize=17, color=ORANGE)
ax.annotate("1 warp, k = 4: 2 %", xy=(1, 2), xytext=(9, 3), fontsize=17, color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE))
ax.set_xlabel("resident warps N"); ax.set_ylabel("lanes busy (%)")
ax.set_title("k work cycles per load, L = 200", fontsize=17)
ax.set_ylim(0, 115); ax.set_xlim(0, 48)
save(fig, "gpu-parallel/fig-occupancy.svg")

# =============================================================== gpu-memory
# M1. row-major layout
fig, ax = plt.subplots(figsize=(12, 3.9))
ax.set_xlim(-0.5, 13); ax.set_ylim(-4.3, 4.2); ax.set_aspect("equal"); ax.axis("off")
W4, H3 = 4, 3
rowcol = [SKY, ORANGE, GREEN]
for y in range(H3):
    for x in range(W4):
        ax.add_patch(Rectangle((x, 3 - 1 - y + 0.2), 0.95, 0.95, fc=rowcol[y], ec="k", lw=0.8, alpha=0.75))
        ax.text(x + 0.47, 2.2 - y + 0.47, f"{x},{y}", ha="center", va="center", fontsize=16)
ax.text(2, 3.55, "4 × 3 image (x, y)", ha="center", fontsize=18)
for i in range(12):
    y, x = divmod(i, W4)
    ax.add_patch(Rectangle((i, -2.3), 0.95, 0.95, fc=rowcol[y], ec="k", lw=0.8, alpha=0.75))
    ax.text(i + 0.47, -1.83, f"{x},{y}", ha="center", va="center", fontsize=16)
    ax.text(i + 0.47, -2.9, str(4 * i), ha="center", va="center", fontsize=14, color="#555")
ax.text(-0.4, -2.9, "byte", ha="right", va="center", fontsize=14, color="#555")
ax.text(6, -3.75, "memory: one long row (float32 = 4 bytes each)", ha="center", fontsize=18)
# neighbours: (1,1)->(2,1) adjacent; (1,1)->(1,2) W apart
ax.annotate("", xy=(6.47, -1.32), xytext=(5.47, -1.32), arrowprops=dict(arrowstyle="->", color=BLUE, lw=2, connectionstyle="arc3,rad=-0.6"))
ax.annotate("", xy=(9.47, -1.32), xytext=(5.47, -1.32), arrowprops=dict(arrowstyle="->", color=VERM, lw=2, connectionstyle="arc3,rad=-0.35"))
ax.text(7.2, 2.6, "row neighbour: next address", color=BLUE, fontsize=17)
ax.text(7.2, 1.8, "column neighbour: 4W bytes = 16 bytes away", color=VERM, fontsize=17)
save(fig, "gpu-memory/fig-rowmajor.svg")

# M2. memory-bound: adding two 4096^2 images
fig, ax = plt.subplots(figsize=(5.2, 2.8))
vals = [("moving 201 MB\nat 1 TB/s", 201, VERM), ("16.8 M additions\nat 3·10¹³ op/s", 0.56, BLUE)]
for i, (lab, v, c) in enumerate(vals):
    ax.barh(1 - i, v, color=c, height=0.55)
    ax.text(v * 1.3, 1 - i, f"{v:g} µs", va="center", fontsize=18)
ax.set_yticks([1, 0], [v[0] for v in vals], fontsize=17)
ax.set_xscale("log"); ax.set_xlim(0.1, 5000); ax.set_xlabel("time (µs, log scale)")
ax.set_title("add two 4096 × 4096 float32 images", fontsize=18)
save(fig, "gpu-memory/fig-membound.svg")

# M3. coalescing: bytes moved for the 128 bytes a warp asks for (coalesce demo presets)
fig, ax = plt.subplots(figsize=(4.4, 3.4))
cases = [("a row", 128, 128), ("RGB, only R", 384, 128), ("a column", 1024, 128)]
for i, (lab, moved, used) in enumerate(cases):
    y = 2 - i
    ax.barh(y, used, color=BLUE, height=0.55)
    ax.barh(y, moved - used, left=used, color=SKY, height=0.55, alpha=0.6)
    ax.text(moved + 15, y, f"{moved} B: {100 * used / moved:.3g} % used", va="center", fontsize=17)
ax.set_yticks([2, 1, 0], [c[0] for c in cases])
ax.set_xlim(0, 1500); ax.set_ylim(-0.5, 3.0); ax.set_xlabel("bytes moved for 128 useful bytes")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="asked for"), Patch(color=SKY, alpha=0.6, label="moved, unused")],
          fontsize=17, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
save(fig, "gpu-memory/fig-coalesce.svg")

# M4. tile + halo
fig, ax = plt.subplots(figsize=(4.6, 4.6))
T = 16
for y in range(T + 2):
    for x in range(T + 2):
        halo = x == 0 or y == 0 or x == T + 1 or y == T + 1
        ax.add_patch(Rectangle((x, y), 1, 1, fc=ORANGE if halo else SKY, ec="white", lw=0.5))
ax.add_patch(Rectangle((1, 1), T, T, fill=False, ec="k", lw=2))
ax.add_patch(Rectangle((0, 7), 3, 3, fill=False, ec=VERM, lw=2.5))
ax.add_patch(Rectangle((1, 8), 1, 1, fc=VERM, ec=VERM))
ax.set_xlim(-0.2, T + 2.2); ax.set_ylim(-3.2, T + 2.2); ax.set_aspect("equal"); ax.axis("off")
ax.text(9, -0.9, "16 × 16 outputs (blue) + halo of 1 (orange)", ha="center", fontsize=17)
ax.text(9, -2.3, "18² = 324 loads for 256 outputs: 1.27 each", ha="center", fontsize=17)
save(fig, "gpu-memory/fig-halo.svg")

# M5. fusing: 4 passes vs 1
fig, ax = plt.subplots(figsize=(12, 2.9))
steps = ["scale", "subtract bg", "clip", "threshold"]
cols = [SKY, ORANGE, GREEN, PURPLE]
for i, (s, c) in enumerate(zip(steps, cols)):
    ax.barh(1, 134, left=134 * i, color=c, edgecolor="k", height=0.6)
    ax.text(134 * i + 67, 1, f"{s}\nread + write", ha="center", va="center", fontsize=16)
ax.barh(0, 134, left=0, color="#bbbbbb", edgecolor="k", height=0.6)
ax.text(67, 0, "all four\nread + write", ha="center", va="center", fontsize=16)
ax.text(540, 1, "4 passes: 537 µs", va="center", fontsize=18)
ax.text(145, 0, "fused: 134 µs", va="center", fontsize=18)
ax.set_yticks([]); ax.set_xlim(0, 720); ax.set_xlabel("time on the desktop GPU, 4096 × 4096 float32 (µs)")
ax.spines["left"].set_visible(False)
save(fig, "gpu-memory/fig-fuse.svg")

# M6. transfers
up, k1, la = 67.1e6 / 25e9 * 1e3, 2 * 67.1e6 / 1e12 * 1e3, 0.01
fig, ax = plt.subplots(figsize=(12, 2.4))
rows = [("one kernel", [("upload", up, PURPLE), ("kernel", k1, ORANGE), ("download", up, PURPLE)])]
x = 0
for name, t, c in rows[0][1]:
    ax.barh(0, t, left=x, color=c, edgecolor="k", height=0.55)
    ax.text(x + t / 2, 0.45, f"{name}\n{t:.2g} ms", ha="center", va="bottom", fontsize=17)
    x += t
ax.set_ylim(-0.5, 1.3)
ax.set_yticks([]); ax.set_xlim(0, 6); ax.set_xlabel("time (ms), 4096 × 4096 float32 = 67 MB, link 25 GB/s")
ax.spines["left"].set_visible(False)
save(fig, "gpu-memory/fig-transfer.svg")

# =============================================================== gpu-patterns
# T1. map vs stencil
fig, axs = plt.subplots(2, 1, figsize=(5.4, 5.6))
for a, title, k in [(axs[0], "map: from the same pixel", 0), (axs[1], "stencil: from a 3 × 3 neighbourhood", 1)]:
    a.set_xlim(-0.5, 13.5); a.set_ylim(-0.8, 6.8); a.set_aspect("equal"); a.axis("off"); a.set_title(title, fontsize=18)
    for (ox, lab) in [(0, "input"), (8, "output")]:
        for y in range(5):
            for x in range(5):
                hit = (ox == 0 and abs(x - 2) <= k and abs(y - 2) <= k) or (ox == 8 and x == 2 and y == 2)
                a.add_patch(Rectangle((ox + x, y + 0.5), 0.92, 0.92, fc=(BLUE if ox == 8 else ORANGE) if hit else LIGHT, ec="#aaa", lw=0.6))
        a.text(ox + 2.5, -0.4, lab, ha="center", fontsize=17)
    arrow(a, (10.4, 3.0), (2.5 + k + 0.6, 3.0), col="k", lw=1.8)
save(fig, "gpu-patterns/fig-map-stencil.svg")

# T2. reduction order in float16 (parscan preset "order")
VF = [2000, 1.5, 3, 0.5, 2.5, 1, 3.5, 0.5, 1.5, 2, 1, 3, 0.5, 2.5, 1.5, 1]
f16 = lambda v: float(np.float16(v))
seq = f16(VF[0])
for v in VF[1:]:
    seq = f16(seq + f16(v))
rows = [[f16(v) for v in VF]]
while len(rows[-1]) > 1:
    r = rows[-1]
    rows.append([f16(r[2 * i] + r[2 * i + 1]) for i in range(len(r) // 2)])
exact = sum(VF)
assert seq == 2025 and rows[-1][0] == 2027 and exact == 2025.5, (seq, rows[-1], exact)
fmt = lambda v: str(int(v)) if v == int(v) else f"{v:g}"
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.set_xlim(-1.6, 19.6); ax.set_ylim(-4.4, 1.25); ax.axis("off")
for k, r in enumerate(rows):
    span = 16 // len(r)
    for i, v in enumerate(r):
        x = i * span + span / 2 - 0.45
        y = -k * 1.05
        ax.add_patch(Rectangle((x - (0.03 if k == 0 else 0), y), 0.96 if k == 0 else min(span - 0.1, 1.6), 0.8, fc="#ffd8a8" if k else "#f6f7f9", ec="#999"))
        ax.text(x + (0.45 if k == 0 else min(span - 0.1, 1.6) / 2), y + 0.4, fmt(v), ha="center", va="center", fontsize=(12.5 if v >= 1000 else 14) if k == 0 else 15)
    ax.text(-0.6, -k * 1.05 + 0.4, "in" if k == 0 else f"step {k}", ha="right", va="center", fontsize=16, color=GREY)
ax.text(16.4, -3 * 1.05 + 0.4, "tree: 2027\nloop: 2025\nexact: 2025.5", ha="left", va="center", fontsize=18)
ax.text(7.5, 1.05, "float16: each partial sum rounded (spacing 1 near 2000)", ha="center", fontsize=17)
save(fig, "gpu-patterns/fig-order.svg")

# T3. Blelloch scan on 8 numbers
x = [3, 1, 7, 0, 4, 1, 6, 3]
brow, bops, blab = [x[:]], [[]], ["in"]
a = x[:]
d = 1
while d < 8:
    ops = []
    for i in range(2 * d - 1, 8, 2 * d):
        a[i] += a[i - d]; ops.append((i, [i - d, i]))
    brow.append(a[:]); bops.append(ops); blab.append(f"up {len(brow) - 1}")
    d *= 2
a[7] = 0; brow.append(a[:]); bops.append([(7, [])]); blab.append("last = 0")
d = 4
while d >= 1:
    ops = []
    for i in range(2 * d - 1, 8, 2 * d):
        t = a[i - d]; a[i - d] = a[i]; a[i] = t + a[i]
        ops.append((i - d, [i])); ops.append((i, [i - d, i]))
    brow.append(a[:]); bops.append(ops); blab.append(f"down {len(blab) - 4}")
    d //= 2
assert brow[-1] == [0, 3, 4, 11, 11, 15, 16, 22]
fig, ax = plt.subplots(figsize=(5.4, 6.6))
ax.set_xlim(-2.6, 8); ax.set_ylim(-len(brow) * 1.2 + 0.2, 1.0); ax.axis("off")
for k, r in enumerate(brow):
    y = -k * 1.2
    hit = {t for t, _ in bops[k]}
    for i, v in enumerate(r):
        ax.add_patch(Rectangle((i, y), 0.86, 0.7, fc=("#ffd8a8" if k <= 3 else "#cfe3f3") if i in hit else "#f6f7f9", ec="#999"))
        ax.text(i + 0.43, y + 0.35, str(v), ha="center", va="center", fontsize=18)
    ax.text(-0.3, y + 0.35, blab[k], ha="right", va="center", fontsize=17, color=GREY)
    if k > 0:
        for t, srcs in bops[k]:
            for s in srcs:
                arrow(ax, (t + 0.43, y + 0.72), (s + 0.43, y + 1.2), col=VERM if k <= 3 else BLUE, lw=1.1)
ax.text(3.9, -len(brow) * 1.2 + 0.25, "exclusive scan: 8 steps (2 log₂ n), 14 additions", ha="center", fontsize=17)
save(fig, "gpu-patterns/fig-blelloch.svg")

# T4. stream compaction (parscan V0, M0)
V0 = [3, 1, 7, 0, 4, 1, 6, 3, 5, 2, 2, 8, 1, 4, 0, 6]
M0 = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1]
ex = list(np.concatenate([[0], np.cumsum(M0)[:-1]]))
out = [V0[i] for i in range(16) if M0[i]]
fig, ax = plt.subplots(figsize=(5.0, 7.4))
ax.set_xlim(-0.2, 7.3); ax.set_ylim(16.3, -1.6); ax.axis("off")
xs_ = [0, 1.6, 3.2, 6.2]
for i in range(16):
    c = ORANGE if M0[i] else "#f6f7f9"
    for x_, v, fc in [(xs_[0], V0[i], c), (xs_[1], M0[i], c), (xs_[2], int(ex[i]), "#ffd8a8" if M0[i] else "#f6f7f9")]:
        ax.add_patch(Rectangle((x_, i), 0.9, 0.84, fc=fc, ec="#888", lw=0.8))
        ax.text(x_ + 0.45, i + 0.43, str(v), ha="center", va="center", fontsize=17)
    if M0[i]:
        arrow(ax, (xs_[3], ex[i] + 0.42), (xs_[2] + 0.92, i + 0.42), col=BLUE, lw=1.3)
for j, v in enumerate(out):
    ax.add_patch(Rectangle((xs_[3], j), 0.9, 0.84, fc=BLUE, ec="#888", lw=0.8))
    ax.text(xs_[3] + 0.45, j + 0.43, str(v), ha="center", va="center", fontsize=17, color="white")
for x_, t in zip(xs_, ["value", "1 keep", "2 scan", "3 out"]):
    ax.text(x_ + 0.45, -0.3, t, ha="center", va="bottom", fontsize=16, rotation=0)
ax.text(xs_[3] + 0.45, len(out) + 0.3, f"{len(out)}\nkept", ha="center", va="top", fontsize=16)
save(fig, "gpu-patterns/fig-compact.svg")

# T5. histogram contention (parscan pixels())
def pixels(kind):
    u = lcg(4242)
    out = []
    for _ in range(64):
        if kind == "uniform":
            out.append(int(u() * 8))
        elif kind == "dark":
            out.append(0 if u() < 0.94 else 1)
        else:
            out.append((1 if u() < 0.7 else 2) if u() < 0.5 else (6 if u() < 0.7 else 5))
    return out
fig, axs = plt.subplots(1, 2, figsize=(5.6, 3.4), sharey=True)
for a, kind in zip(axs, ["dark", "uniform"]):
    h = np.bincount(pixels(kind), minlength=8)
    a.bar(range(8), h, color=[VERM if v == h.max() else BLUE for v in h])
    a.set_title(f"{kind}: {h.max()} cycles", fontsize=18)
    a.set_xticks(range(8)); a.set_xlabel("bin")
axs[0].set_ylabel("atomic updates, serialised")
if True:
    hd = np.bincount(pixels("dark"), minlength=8)
    assert hd[0] == 58, hd
save(fig, "gpu-patterns/fig-contention.svg")

# T6. gather vs scatter
fig, axs = plt.subplots(2, 1, figsize=(5.6, 6.0))
for a, title, gather in [(axs[0], "gather: read many, write one", True),
                         (axs[1], "scatter: write where the data says", False)]:
    a.set_xlim(-2.4, 8.3); a.set_ylim(-0.8, 4.0); a.axis("off"); a.set_title(title, fontsize=18)
    cells(a, [""] * 8, 3, fc=[ORANGE if 2 <= i <= 4 else LIGHT for i in range(8)] if gather else LIGHT)
    cells(a, [""] * 8, 0, fc=LIGHT)
    a.text(-0.3, 3.4, "input", ha="right", va="center", fontsize=17)
    a.text(-0.3, 0.4, "output", ha="right", va="center", fontsize=17)
    if gather:
        for s in (2, 3, 4):
            arrow(a, (3.45, 0.82), (s + 0.45, 3.0), col=BLUE, lw=2)
        a.add_patch(Rectangle((3, 0), 0.9, 0.8, fc=BLUE, ec="#888"))
        a.text(4, -0.5, "no write conflicts", ha="center", fontsize=17, color=BLUE)
    else:
        for s, t in [(0, 1), (2, 5), (3, 5), (6, 5), (7, 2)]:
            arrow(a, (t + 0.45, 0.82), (s + 0.45, 3.0), col=VERM if t == 5 else BLUE, lw=2)
            a.add_patch(Rectangle((s, 3), 0.9, 0.8, fc=ORANGE, ec="#888"))
        a.add_patch(Rectangle((5, 0), 0.9, 0.8, fc=VERM, ec="#888"))
        a.text(4.5, -0.5, "3 writes to one cell: atomics", ha="center", fontsize=17, color=VERM)
save(fig, "gpu-patterns/fig-gather-scatter.svg")

# =============================================================== gpu-fit
# F1. scorecard
methods = ["element-wise steps", "small filters", "FFT", "network layer", "ray marching",
           "watershed queue", "union–find labelling", "DP table", "error diffusion"]
qs = ["1", "2", "3", "4"]
Y, P_, N_ = 2, 1, 0
S = [[Y, Y, Y, Y], [Y, Y, Y, Y], [Y, Y, P_, Y], [Y, Y, Y, Y], [Y, P_, P_, Y],
     [N_, N_, N_, N_], [N_, P_, N_, N_], [P_, Y, Y, N_], [N_, Y, Y, N_]]
sym = {Y: ("✓", BLUE), P_: ("~", ORANGE), N_: ("✗", VERM)}
CW = 1.9
fig, ax = plt.subplots(figsize=(7.6, 5.8))
ax.set_xlim(-4.6, 4 * CW); ax.set_ylim(len(methods) + 0.8, -1.0); ax.axis("off")
for j, q in enumerate(qs):
    ax.text(CW * j + CW / 2, -0.15, q, ha="center", va="bottom", fontsize=20, weight="bold")
for i, (m, row) in enumerate(zip(methods, S)):
    if i == 5:
        ax.plot([-4.5, 4 * CW], [i, i], color="k", lw=1.2)
    ax.text(-0.15, i + 0.5, m, ha="right", va="center", fontsize=17)
    for j, v in enumerate(row):
        s, c = sym[v]
        ax.add_patch(Rectangle((CW * j + 0.05, i + 0.05), CW - 0.1, 0.9, fc="white", ec="#ccc"))
        ax.text(CW * j + CW / 2, i + 0.52, s, ha="center", va="center", fontsize=20, color=c, weight="bold")
ax.text(-4.5, -0.25, "question:", fontsize=17, va="bottom", color="#444")
ax.text(-4.5, len(methods) + 0.05, "✓ yes   ~ partly   ✗ no", fontsize=16, va="top", color="#444")
save(fig, "gpu-fit/fig-scorecard.svg")

# F2. cell micrograph: stencil + map pipeline
from skimage import io
img = io.imread(os.path.join(R, "demos/src/fourier2d/cell256.png")).astype(float)
if img.ndim == 3:
    img = img[..., :3].mean(axis=2)
img /= img.max()
sm = ndi.gaussian_filter(img, 2.0)
gm = np.hypot(ndi.sobel(sm, 0), ndi.sobel(sm, 1))
edge = gm > np.percentile(gm, 85)
fig, axs = plt.subplots(2, 2, figsize=(5.4, 6.0))
for a, im, t in zip(axs.flat, [img, sm, gm, edge], ["1. input", "2. Gaussian σ = 2\n(stencil)",
                                          "3. gradient size\n(stencil + map)", "4. > threshold\n(map)"]):
    a.imshow(im, cmap="gray" if im is not gm else "magma", interpolation="nearest"); a.set_title(t, fontsize=17); a.axis("off")
fig.subplots_adjust(hspace=0.3, wspace=0.05)
save(fig, "gpu-fit/fig-cellpipe.svg")

# F3. roofline with a network layer
peak, bw = 30e12, 1e12
I = np.logspace(-2, 3.2, 300)
fig, ax = plt.subplots(figsize=(4.6, 3.8))
ax.loglog(I, np.minimum(peak, bw * I) / 1e12, "k", lw=3)
ax.axvline(30, color=GREY, ls="--", lw=1.4)
ax.text(33, 0.013, "ridge 30", color=GREY, fontsize=16)
for lab, x, dx, dy, ha in [("add images", 1 / 12, 1.6, 0.6, "left"), ("3×3 filter", 2.25, 1.6, 0.6, "left"),
                           ("network layer\nC = 64: I = 143", 143, 1.0, 2.0, "center")]:
    y = min(peak, bw * x) / 1e12
    ax.plot([x], [y], "o", ms=11, color=VERM if x < 30 else BLUE)
    ax.text(x * dx, y * dy, lab, ha=ha, va="bottom" if dy > 1 else "top", fontsize=17, bbox=dict(fc="white", ec="none", pad=1))
ax.set_xlabel("operations per byte"); ax.set_ylabel("10¹² op/s")
ax.set_xlim(0.01, 1500); ax.set_ylim(0.01, 5000)
save(fig, "gpu-fit/fig-nnroof.svg")

# F4. label propagation: arrival pass per pixel (parlabel shapes)
NN = 24
def shape(s):
    m = np.zeros((NN, NN), np.uint8)
    def setp(x, y):
        if 0 <= x < NN and 0 <= y < NN: m[y, x] = 1
    if s == "blobs":
        yy, xx = np.mgrid[:NN, :NN]
        for cx, cy, r in [(6, 6, 4), (16, 5, 3), (17, 16, 5), (5, 17, 3.5), (11, 11, 1.5)]:
            m[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 1
    else:
        x, y, d, ln, k = 1, 1, 0, 21, 0
        D = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        setp(x, y)
        while ln > 0:
            for _ in range(ln):
                x += D[d][0]; y += D[d][1]; setp(x, y)
            d = (d + 1) % 4; k += 1
            if k >= 3 and k % 2 == 1: ln -= 2
    return m
def propagate(m):
    L = np.where(m, np.arange(NN * NN).reshape(NN, NN) + 1, 0)
    BIG = 10 ** 9
    arrive = np.zeros_like(L); passes = 0
    while True:
        P = np.where(L > 0, L, BIG)
        Pp = np.pad(P, 1, constant_values=BIG)
        nmin = np.minimum.reduce([Pp[1:-1, 1:-1], Pp[:-2, 1:-1], Pp[2:, 1:-1], Pp[1:-1, :-2], Pp[1:-1, 2:]])
        Nn = np.where(m > 0, nmin, 0)
        if (Nn == L).all(): break
        passes += 1
        arrive[(Nn != L)] = passes
        L = Nn
    return passes, arrive
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.9))
for a, s in zip(axs, ["blobs", "spiral"]):
    m = shape(s)
    p, arr = propagate(m)
    assert p == {"blobs": 12, "spiral": 262}[s], (s, p)
    im = np.ma.masked_where(m == 0, arr)
    cm = plt.get_cmap("viridis").copy(); cm.set_bad("white")
    h = a.imshow(im, cmap=cm, interpolation="nearest")
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_visible(True); sp.set_color("#999")
    a.set_title(f"{s}: {p} passes", fontsize=18)
fig.subplots_adjust(bottom=0.12, wspace=0.08)
fig.text(0.5, 0.08, "colour: pass of the last change\n(dark early, yellow last)", ha="center", va="top", fontsize=17)
save(fig, "gpu-fit/fig-propagation.svg")

# F5. DP wavefront (10 x 16 table)
Rr, Cc = 10, 16
fig, axs = plt.subplots(2, 1, figsize=(5.4, 5.4), gridspec_kw=dict(height_ratios=[1.25, 1]))
a = axs[0]
D = np.add.outer(np.arange(Rr), np.arange(Cc))
a.imshow(D, cmap="cividis", interpolation="nearest")
for i in range(Rr):
    j = 9 - i
    if 0 <= j < Cc:
        a.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=VERM, lw=2.2))
a.set_xticks([]); a.set_yticks([])
a.set_title("cell (i, j) waits for i + j; step 9 in red", fontsize=17)
cnt = [int(np.sum(D == d)) for d in range(Rr + Cc - 1)]
b = axs[1]
b.bar(range(len(cnt)), cnt, color=[VERM if d == 9 else BLUE for d in range(len(cnt))])
b.set_xlabel("step (anti-diagonal)"); b.set_ylabel("busy lanes")
b.set_title(f"{len(cnt)} steps for {Rr * Cc} cells: {Rr * Cc / len(cnt):.1f} per step", fontsize=17)
save(fig, "gpu-fit/fig-wavefront.svg")

# F6. float16 accumulation stalls
K = 10000
s16 = np.float16(0); s32 = np.float32(0); t16 = []; t32 = []
v16 = np.float16(0.01)
for i in range(K):
    s16 = np.float16(s16 + v16); s32 = np.float32(s32 + np.float32(v16))
    t16.append(float(s16)); t32.append(float(s32))
assert t16[-1] == 32 and abs(t32[-1] - 100.02) < 0.01, (t16[-1], t32[-1])
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.plot(np.arange(1, K + 1), t32, color=BLUE, lw=3, label="float32 accumulator")
ax.plot(np.arange(1, K + 1), t16, color=VERM, lw=3, label="float16 accumulator")
ax.text(9600, t32[-1] - 8, f"{t32[-1]:.2f}", ha="left", va="top", color=BLUE, fontsize=17)
ax.text(9800, 36, "stuck at 32", ha="right", color=VERM, fontsize=17)
ax.set_xlabel("terms added (each 0.01 in float16)"); ax.set_ylabel("running sum")
ax.legend(fontsize=16, frameon=False, loc="lower right")
save(fig, "gpu-fit/fig-stall.svg")
