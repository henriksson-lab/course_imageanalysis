"""Static figures for the descriptors/ deck.

Run from the repository root (after tools/figs/keypoints/export.ts, see its header):
    python3 -I tools/figs/descriptors/figs.py <data-dir>
Writes descriptors/fig-*.svg / *.png. Descriptors from demos/src/lib/descriptors.ts.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "keypoints"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, Wedge
import kplib
from kplib import load, show, save, arr, apply, kpcircle, BLUE, ORANGE, GREEN, RED, PURPLE, CYAN, YELLOW, GREY, FONT

kplib.DATA = sys.argv[1]
OUT = "descriptors"


def frame_square(ax, k, half, color, lw=2.2):
    t = k["t"]; u = np.array([np.cos(t), -np.sin(t)]); v = np.array([np.sin(t), np.cos(t)])
    c = np.array([k["x"], k["y"]])
    pts = [c + half * (a * u + b * v) for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    ax.add_patch(Polygon(pts, fill=False, ec=color, lw=lw))
    ax.plot([c[0], c[0] + half * u[0]], [c[1], c[1] + half * u[1]], color=color, lw=lw)


def fig_job():
    d = load("job")
    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1.5], hspace=0.6)
    a1 = fig.add_subplot(gs[:, 0]); a2 = fig.add_subplot(gs[:, 1])
    show(a1, d["v1"]); show(a2, d["v2"])
    frame_square(a1, d["k1"], 6 * d["k1"]["s"], YELLOW)
    a1.set_title("view 1", fontsize=FONT - 2)
    rows = [("true partner", d["partner"], GREEN)] + [(f"impostor {i + 1}", im, RED) for i, im in enumerate(d["impostors"])]
    for r, (lab, e, col) in enumerate(rows):
        frame_square(a2, e["k"], 6 * e["k"]["s"], col)
        ax = fig.add_subplot(gs[r, 2])
        ax.bar(np.arange(128), d["d1"], color=GREY, width=1.0, alpha=0.6, label="view 1")
        ax.bar(np.arange(128), e["d"], color=col, width=1.0, alpha=0.6)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 128)
        ax.set_title(f"{lab}: distance {e['dist']:.2f}", fontsize=FONT - 2, color=col)
    a2.set_title("view 2", fontsize=FONT - 2)
    save(fig, OUT, "fig-job.png", dpi=100)


def fig_frame():
    d = load("job")
    k = d["k1"]; det = d["detail"]; sp = det["spacing"]
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.6))
    v1 = arr(d["v1"]); x0, y0, R = int(k["x"]), int(k["y"]), int(9 * sp + 6)
    axs[0].imshow(v1, cmap="gray", vmin=0, vmax=255, extent=(-0.5, 255.5, 255.5, -0.5))
    axs[0].set_xlim(x0 - R, x0 + R); axs[0].set_ylim(y0 + R, y0 - R); axs[0].set_xticks([]); axs[0].set_yticks([])
    t = k["t"]; u = np.array([np.cos(t), -np.sin(t)]); v = np.array([np.sin(t), np.cos(t)]); c = np.array([k["x"], k["y"]])
    for i in range(16):
        for j in range(16):
            p = c + sp * ((i - 7.5) * u + (j - 7.5) * v)
            axs[0].plot(p[0], p[1], ".", color=YELLOW, ms=3)
    frame_square(axs[0], k, 8 * sp, YELLOW)
    axs[0].add_patch(Circle(c, 2 * k["s"], fill=False, ec=CYAN, lw=2))
    axs[0].set_title(f"frame: centre, size ∝ σ, turned by θ = {np.degrees(t):.0f}°", fontsize=FONT - 2)
    show(axs[1], det["patch"], lo=None, hi=None)
    axs[1].set_title("16 × 16 samples in the frame", fontsize=FONT - 2)
    save(fig, OUT, "fig-frame.png", dpi=100)


def fig_ncc_shift():
    d = load("nccshift")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(d["shifts"], d["texture"], "-o", color=RED, lw=2.4, label="textured window")
    ax.plot(d["shifts"], d["edge"], "-o", color=BLUE, lw=2.4, label="window on a straight edge")
    ax.set_xlabel("shift of the window (pixels)"); ax.set_ylabel("NCC with itself")
    ax.set_ylim(-0.2, 1.05); ax.axhline(0, color=GREY, lw=1)
    ax.legend(frameon=False, loc="lower left")
    save(fig, OUT, "fig-ncc-shift.svg")
    print("  NCC texture", [round(v, 2) for v in d["texture"]], "edge", [round(v, 2) for v in d["edge"]])


def roses(ax, desc, cells=4):
    desc = np.array(desc); m = desc.max()
    for cj in range(cells):
        for ci in range(cells):
            cx, cy = ci + 0.5, cj + 0.5
            ax.add_patch(Rectangle((ci + 0.03, cj + 0.03), 0.94, 0.94, fc="#f4f6f8", ec="#ccc"))
            for b in range(8):
                L = 0.46 * desc[(cj * cells + ci) * 8 + b] / m
                a = b * np.pi / 4
                ax.add_patch(Polygon([(cx, cy), (cx + L * np.cos(a - 0.33), cy - L * np.sin(a - 0.33)), (cx + L * np.cos(a + 0.33), cy - L * np.sin(a + 0.33))], fc=BLUE, ec="none", alpha=0.85))
    ax.set_xlim(0, cells); ax.set_ylim(cells, 0); ax.set_aspect("equal"); ax.axis("off")


def fig_steps():
    d = load("job"); det = d["detail"]
    p = arr(det["patch"]); n = p.shape[0]
    gx = np.array(det["gx"]).reshape(n, n); gy = np.array(det["gy"]).reshape(n, n)
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.4]})
    yy, xx = np.mgrid[0:n, 0:n]
    w = np.exp(-((xx - 7.5) ** 2 + (yy - 7.5) ** 2) / (2 * 8 ** 2))
    for k in (0, 1):
        axs[k].imshow(p, cmap="gray", extent=(-0.5, n - 0.5, n - 0.5, -0.5)); axs[k].set_xticks([]); axs[k].set_yticks([])
        axs[k].quiver(xx, yy, gx * (w if k else 1), -gy * (w if k else 1), color=YELLOW, scale=np.hypot(gx, gy).max() * 9, width=0.012)
    axs[0].set_title("1. gradients in the frame", fontsize=FONT - 3)
    axs[1].add_patch(Circle((7.5, 7.5), 8, fill=False, ec=YELLOW, ls="--", lw=1.5))
    for q in range(1, 4):
        axs[1].axvline(4 * q - 0.5, color=CYAN, lw=2); axs[1].axhline(4 * q - 0.5, color=CYAN, lw=2)
    axs[1].set_title("2. Gaussian weight, 4 × 4 cells", fontsize=FONT - 3)
    roses(axs[2], d["d1"]); axs[2].set_title("3. 8 directions per cell", fontsize=FONT - 3)
    axs[3].bar(np.arange(128), d["d1"], color=BLUE, width=1.0); axs[3].axhline(0.2, color=RED, ls="--", lw=1.5)
    axs[3].text(127, 0.205, "clip 0.2", color=RED, ha="right", va="bottom", fontsize=FONT - 4)
    axs[3].set_xlim(0, 128); axs[3].set_xticks([0, 32, 64, 96, 128]); axs[3].set_title("4. 128 numbers, unit length", fontsize=FONT - 3)
    save(fig, OUT, "fig-steps.png", dpi=100)


def fig_cell_rotate():
    def h(gr):
        out = np.zeros(8)
        for m, a in gr:
            out[int(round(a / 45)) % 8] += m
        return out
    g = [(2, 10), (1, 50), (3, 100), (2, 355)]
    gr = [(m, (a + 90) % 360) for m, a in g]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
    labs = ["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"]
    for ax, hh, t, col in [(axs[0], h(g), "one cell", BLUE), (axs[1], h(gr), "image turned 90°", RED), (axs[2], h([(m, (a - 90) % 360) for m, a in gr]), "turned, relative to θ", GREEN)]:
        ax.bar(range(8), hh, color=col); ax.set_xticks(range(8)); ax.set_xticklabels(labs, rotation=60, fontsize=FONT - 5)
        ax.set_title(t + "\n[" + " ".join(str(int(v)) for v in hh) + "]", fontsize=FONT - 3)
    axs[0].set_ylabel("votes")
    save(fig, OUT, "fig-cell-rotate.svg")


def fig_binary():
    d = load("job"); b = d["binary"]
    p = arr(b["patch"]); S = p.shape[0]; c = (S - 1) / 2
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, n in ((axs[0], 30), (axs[1], 0)):
        ax.imshow(p, cmap="gray", extent=(-0.5, S - 0.5, S - 0.5, -0.5)); ax.set_xticks([]); ax.set_yticks([])
    pr = b["pairs"]
    for k in range(30):
        x1, y1, x2, y2 = [pr[4 * k + q] + c for q in range(4)]
        col = GREEN if b["bits"][k] else RED
        axs[0].plot([x1, x2], [y1, y2], color=col, lw=2.2); axs[0].plot([x1], [y1], "o", color=col, ms=5)
    axs[0].set_title("30 of 256 pairs: green = 1 (p darker than q)", fontsize=FONT - 3)
    # intensity centroid
    yy, xx = np.mgrid[0:S, 0:S] - c
    disc = xx ** 2 + yy ** 2 <= c ** 2
    m10 = (xx * p * disc).sum(); m01 = (yy * p * disc).sum(); m00 = (p * disc).sum()
    axs[1].add_patch(Circle((c, c), c, fill=False, ec=YELLOW, ls="--", lw=1.5))
    L = np.hypot(m10, m01) or 1
    axs[1].annotate("", xy=(c + 11 * m10 / L, c + 11 * m01 / L), xytext=(c, c), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=3.5))
    axs[1].plot(c + m10 / m00, c + m01 / m00, "o", color=ORANGE, ms=9)
    axs[1].set_title("intensity centroid: the direction θ", fontsize=FONT - 3)
    save(fig, OUT, "fig-binary.png", dpi=100)


def fig_hamming():
    a = "1011001011100001"; b = "1001011011000011"
    x = "".join("1" if p != q else "0" for p, q in zip(a, b))
    fig, ax = plt.subplots(figsize=(9, 2.6))
    for r, (lab, s) in enumerate([("a", a), ("b", b), ("a XOR b", x)]):
        for i, ch in enumerate(s):
            fc = ("#222" if ch == "1" else "#f2f2f2") if r < 2 else (RED if ch == "1" else "#f2f2f2")
            ax.add_patch(Rectangle((i, -r * 1.2), 0.92, 0.92, fc=fc, ec="#999"))
            ax.text(i + 0.46, -r * 1.2 + 0.46, ch, ha="center", va="center", color="w" if ch == "1" else "#444", fontsize=FONT - 3)
        ax.text(-0.4, -r * 1.2 + 0.46, lab, ha="right", va="center")
    ax.text(16.4, -2.4 + 0.46, f"4 bits differ:\nHamming distance 4", va="center", color=RED, fontsize=FONT - 2)
    ax.set_xlim(-3.5, 21); ax.set_ylim(-2.6, 1.1); ax.axis("off"); ax.set_aspect("equal")
    save(fig, OUT, "fig-hamming.svg")


def fig_failures():
    d = load("failures")
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.3))
    show(axs[0], d["grid"]); axs[0].set_title("repetitive: many identical windows", fontsize=FONT - 3)
    show(axs[1], d["tilt"]); axs[1].set_title("strong tilt: the window changes", fontsize=FONT - 3)
    show(axs[2], d["inv"]); axs[2].set_title("contrast inverted: gradients flip", fontsize=FONT - 3)
    save(fig, OUT, "fig-failures.png", dpi=100)


def fig_concentration():
    d = load("concentration")
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, dim in zip(axs, ["2", "128"]):
        bins = np.linspace(0, 2, 41)
        ax.hist(d[dim]["nearest"], bins=bins, color=GREEN, alpha=0.7, label="nearest")
        ax.hist(d[dim]["farthest"], bins=bins, color=RED, alpha=0.7, label="farthest")
        ax.set_title(f"{dim} dimensions"); ax.set_xlabel("distance between unit vectors")
        nn, ff = np.mean(d[dim]["nearest"]), np.mean(d[dim]["farthest"])
        print(f"  dim {dim}: mean nearest {nn:.3f}, farthest {ff:.3f}, ratio {nn / ff:.2f}")
    axs[0].set_ylabel("queries"); axs[0].legend(frameon=False)
    save(fig, OUT, "fig-concentration.svg")


if __name__ == "__main__":
    for f in [fig_job, fig_frame, fig_ncc_shift, fig_steps, fig_cell_rotate, fig_binary, fig_hamming, fig_failures, fig_concentration]:
        f()
