"""Static figures for the keypoints/ deck.

Run from the repository root (after export.ts, see its header):
    python3 -I tools/figs/keypoints/figs.py <data-dir>
Writes keypoints/fig-*.svg / *.png. Synthetic scenes from demos/src/lib/transforms2d.ts and
demos/src/dogpyramid/scene.ts; keypoints from demos/src/lib/keypoints.ts.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Polygon
import kplib
from kplib import load, show, save, arr, apply, kpcircle, side_by_side, BLUE, ORANGE, GREEN, RED, PURPLE, CYAN, YELLOW, GREY, FONT

kplib.DATA = sys.argv[1]
OUT = "keypoints"


def fig_task():
    d = load("task")
    im, off = side_by_side(d["v1"], d["v2"])
    fig, ax = plt.subplots(figsize=(10, 4.9))
    show(ax, im)
    pairs = sorted(d["pairs"], key=lambda p: -p[0]["r"])[:16]
    for a, b in pairs:
        ax.plot([a["x"], b["x"] + off], [a["y"], b["y"]], color=YELLOW, lw=1.4, alpha=0.9)
        for k, dx in ((a, 0), (b, off)):
            ax.add_patch(Circle((k["x"] + dx, k["y"]), 2 * k["s"], fill=False, ec=YELLOW, lw=1.6))
    ax.text(128, -8, "view 1", ha="center"); ax.text(off + 128, -8, "view 2: turned 30°, zoomed 1.3×", ha="center")
    save(fig, OUT, "fig-task.png", dpi=110)


def fig_harriszoom():
    d = load("harriszoom")
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.2))
    show(axs[0], d["v1"]); show(axs[1], d["v2"])
    T = d["T12"]
    for k, p in zip(d["h1"], d["hp"]):
        if p == -1:
            continue
        axs[0].add_patch(Circle((k["x"], k["y"]), 3.5, fill=False, ec=GREEN if p >= 0 else RED, lw=2.2))
    for k in d["h2"]:
        axs[1].add_patch(Circle((k["x"], k["y"]), 3.5, fill=False, ec=CYAN, lw=2))
    # common region
    c = np.array([apply(np.linalg.inv(np.array(T).reshape(3, 3)).ravel(), x, y) for x, y in [(0, 0), (255, 0), (255, 255), (0, 255)]])
    axs[0].add_patch(Polygon(c, fill=False, ec="white", ls="--", lw=1.5))
    n = sum(1 for p in d["hp"] if p != -1); rep = sum(1 for p in d["hp"] if p >= 0)
    axs[0].set_title(f"view 1: {rep} of {n} found again", fontsize=FONT)
    axs[1].set_title("view 2 = view 1 zoomed 2.2×", fontsize=FONT)
    save(fig, OUT, "fig-harris-zoom.png", dpi=100)
    print("  Harris one sigma at zoom 2.2: repeatability", d["hrep"], "DoG", d["drep"])


def fig_scale_covariance():
    # normalised LoG at the centre of a Gaussian blob of width s: σ² ∇²(G_s * G_σ)(0) ∝ σ²/(s²+σ²)²
    sig = np.geomspace(0.5, 40, 400)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for s, col, lab in [(3, BLUE, "blob in view 1 (width 3 px)"), (6, RED, "the same blob zoomed 2× (6 px)")]:
        r = sig ** 2 * 2 / (s ** 2 + sig ** 2) ** 2
        ax.plot(sig, r / r.max(), color=col, lw=2.5, label=lab)
        ax.axvline(s, color=col, ls="--", lw=1.2)
    ax.set_xscale("log"); ax.set_xticks([1, 3, 6, 10, 30]); ax.set_xticklabels(["1", "3", "6", "10", "30"]); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("σ (pixels, log scale)"); ax.set_ylabel("normalised response")
    ax.annotate("", xy=(6, 1.04), xytext=(3, 1.04), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(4.25, 1.07, "shift by log 2", ha="center")
    ax.set_ylim(0, 1.18); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=1)
    save(fig, OUT, "fig-scale-covariance.svg")


def fig_octaves():
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set_xlim(0, 13.6); ax.set_ylim(-0.3, 6.3); ax.axis("off")
    S = 3
    for o, (y, size) in enumerate([(3.4, 1.0), (1.05, 0.5)]):
        for i in range(S + 3):
            x = 0.2 + i * 1.25
            ax.add_patch(Rectangle((x, y), size, size, fc="#cfd8e3", ec="#333"))
            ax.text(x + size / 2, y + size + 0.12, f"σ$_0$k$^{i}$" if o == 0 else "", ha="center", fontsize=FONT - 4)
        for i in range(S + 2):
            x = 8.0 + i * 1.1
            ax.add_patch(Rectangle((x, y), size, size, fc="#f4d7d1" if 1 <= i <= S else "#eee", ec="#333"))
        ax.text(-0.05, y + size / 2, f"octave {o}", ha="right", va="center", rotation=90, fontsize=FONT - 2)
    ax.text(3.9, 5.75, "blurred images: S + 3 = 6 per octave", ha="center")
    ax.text(10.4, 5.75, "DoGs: S + 2 = 5", ha="center")
    ax.text(10.4, 2.4, "extrema searched on\nthe S = 3 inner levels (red)", ha="center", fontsize=FONT - 3)
    ax.add_patch(FancyArrowPatch((7.05, 3.9), (7.9, 3.9), arrowstyle="->", mutation_scale=18, lw=1.5))
    ax.text(7.45, 3.15, "subtract\nneighbours", ha="center", fontsize=FONT - 5)
    ax.add_patch(FancyArrowPatch((0.2 + 3 * 1.25 + 0.5, 3.35), (0.45, 1.6), arrowstyle="->", mutation_scale=18, lw=1.5, connectionstyle="arc3,rad=0.25", color=RED))
    ax.text(4.6, 2.4, "level S = 3 has σ = 2σ$_0$:\nkeep every 2nd pixel", color=RED, fontsize=FONT - 4, ha="center")
    ax.text(3.9, 0.35, "half the size: 1/4 of the pixels", ha="center", fontsize=FONT - 3)
    save(fig, OUT, "fig-octaves.svg")


def stage_markers(ax, recs, lvl=None, octave=0, show_edge=True):
    col = {"kept": GREEN, "edge": ORANGE, "contrast": BLUE, "unstable": GREY}
    for r in recs:
        if lvl is not None and (r["o"] != octave or r["l"] != lvl):
            continue
        c = col[r["stage"]]
        if r["stage"] in ("kept", "edge"):
            if r["stage"] == "edge" and not show_edge:
                c = GREEN
            ax.add_patch(Circle((r["x"], r["y"]), 2 * r["s"], fill=False, ec=c, lw=2.2))
        else:
            ax.plot(r["x"], r["y"], "x", color=c, ms=9, mew=2.5)


def fig_dog_levels():
    d = load("dogscene")
    fig, axs = plt.subplots(1, 4, figsize=(14, 4.0))
    show(axs[0], d["img"]); axs[0].set_title("image, 192 × 192", fontsize=FONT - 2)
    m = max(np.abs(arr(g)).max() for g in d["dog"][:4])
    for j, l in enumerate([1, 2, 3]):
        a = arr(d["dog"][l])
        axs[j + 1].imshow(a, cmap="RdBu_r", vmin=-m, vmax=m, interpolation="nearest"); axs[j + 1].set_xticks([]); axs[j + 1].set_yticks([])
        s = 1.6 * 2 ** (l / 3)
        axs[j + 1].set_title(f"DoG level {l}, σ {s:.2f}", fontsize=FONT - 2)
        recs = [r for r in d["records"] if r["o"] == 0 and r["l"] == l]
        for r in recs:
            axs[j + 1].add_patch(Circle((r["x"], r["y"]), 2 * r["s"], fill=False, ec="k", lw=1.6))
    save(fig, OUT, "fig-dog-levels.png", dpi=100)


def fig_refine():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = np.array([-1, 0, 1]); ys = np.array([0.6, 1.0, 0.8])
    a = (ys[0] - 2 * ys[1] + ys[2]) / 2; b = (ys[2] - ys[0]) / 2; c = ys[1]
    x = np.linspace(-1.5, 1.5, 200)
    ax.plot(x, a * x ** 2 + b * x + c, color=BLUE, lw=2.5, label="parabola through the 3 samples")
    ax.plot(xs, ys, "o", color="k", ms=10, zorder=3, label="DoG samples")
    h = -b / (2 * a); peak = c - b * b / (4 * a)
    ax.plot([h], [peak], "*", color=RED, ms=18, zorder=4, label=f"peak at +{h:.2f} px, value {peak:.3f}")
    ax.axvline(0, color=GREY, ls=":"); ax.axvline(h, color=RED, ls="--", lw=1.2)
    ax.set_xlabel("x (pixels from the sample)"); ax.set_ylabel("D"); ax.set_ylim(0.3, 1.12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=FONT - 3)
    save(fig, OUT, "fig-refine.svg")


def fig_edge_test():
    d = load("dogscene")
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax, edge, title in [(axs[0], False, "all candidates (contrast OK)"), (axs[1], True, "after the edge test (r = 10)")]:
        show(ax, d["img"])
        for r in d["records"]:
            if r["stage"] in ("contrast", "unstable"):
                continue
            if r["stage"] == "edge" and edge:
                ax.plot(r["x"], r["y"], "x", color=ORANGE, ms=10, mew=2.5)
                continue
            ax.add_patch(Circle((r["x"], r["y"]), 2 * r["s"], fill=False, ec=GREEN if r["stage"] == "kept" or not edge else ORANGE, lw=2.2))
        ax.set_title(title)
    save(fig, OUT, "fig-edge-test.png", dpi=100)


def fig_orientation():
    d = load("orientation")
    p = np.array(d["patch"]); s = d["s"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
    R = (p.shape[0] - 1) // 2
    axs[0].imshow(p, cmap="gray", extent=(-R - 0.5, R + 0.5, R + 0.5, -R - 0.5)); axs[0].set_xticks([]); axs[0].set_yticks([])
    gy, gx = np.gradient(p)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    w = np.exp(-(xx ** 2 + yy ** 2) / (2 * (1.5 * s) ** 2))
    m = np.hypot(gx, gy) * w
    st = max(2, R // 5)
    sel = (slice(None, None, st), slice(None, None, st))
    axs[0].quiver(xx[sel], yy[sel], gx[sel], -gy[sel], m[sel], cmap="autumn", scale=m.max() * 9, width=0.008)
    axs[0].add_patch(Circle((0, 0), 3 * 1.5 * s, fill=False, ec=YELLOW, ls="--", lw=1.5))
    axs[0].set_title("gradients around the keypoint")
    h = np.array(d["hist"]); bins = np.arange(36) * 10 + 5
    axs[1].bar(bins, h / h.max(), width=9, color=BLUE)
    axs[1].axhline(0.8, color=RED, ls="--", lw=1.5); axs[1].text(362, 0.8, "80 %", color=RED, va="center")
    for pk in d["peaks"]:
        axs[1].axvline(np.degrees(pk), color=ORANGE, lw=2)
    axs[1].set_xlim(0, 360); axs[1].set_xticks([0, 90, 180, 270, 360])
    axs[1].set_xlabel("gradient direction (°, counter-clockwise)"); axs[1].set_ylabel("weighted votes")
    axs[1].set_title(f"36 bins: peaks at {', '.join(f'{np.degrees(x):.0f}°' for x in d['peaks'])}")
    save(fig, OUT, "fig-orientation.svg")


def fig_frames():
    d = load("frames")
    im, off = side_by_side(d["v1"], d["v2"])
    fig, ax = plt.subplots(figsize=(10, 4.9))
    show(ax, im)
    T = d["T12"]
    k2 = d["k2"]
    shown = 0
    for k in sorted(d["k1"], key=lambda k: -k["r"]):
        x, y = apply(T, k["x"], k["y"])
        cand = [q for q in k2 if np.hypot(q["x"] - x, q["y"] - y) < 2 and abs(np.angle(np.exp(1j * (q["t"] - k["t"] - np.radians(40))))) < 0.25]
        if not cand or k["s"] < 2.2:
            continue
        q = cand[0]
        col = plt.cm.tab10(shown % 10)
        kpcircle(ax, k, col, lw=2.2, scale=3)
        kpcircle(ax, dict(q, x=q["x"] + off), col, lw=2.2, scale=3)
        shown += 1
        if shown >= 9:
            break
    ax.text(128, -8, "view 1", ha="center"); ax.text(off + 128, -8, "view 2: turned 40°, zoomed 1.4×", ha="center")
    save(fig, OUT, "fig-frames.png", dpi=110)


def fig_repeatability():
    d = load("repcurves")
    names = {"harris": ("Harris, one σ", RED), "harrisms": ("Harris, multi-scale", PURPLE), "dog": ("DoG", BLUE), "fast": ("FAST", ORANGE)}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    for k, (lab, col) in names.items():
        axs[0].plot(d["zooms"], np.array(d["curves"][k]["zoom"]) * 100, "-o", color=col, lw=2.2, ms=5, label=lab)
        axs[1].plot(d["angles"], np.array(d["curves"][k]["angle"]) * 100, "-o", color=col, lw=2.2, ms=5, label=lab)
    axs[0].set_xscale("log"); axs[0].set_xticks([0.5, 1, 2, 3]); axs[0].set_xticklabels(["0.5", "1", "2", "3"]); axs[0].xaxis.set_minor_formatter(plt.NullFormatter())
    axs[0].set_xlabel("zoom of view 2"); axs[1].set_xlabel("rotation of view 2 (°)"); axs[1].set_xticks([0, 45, 90, 135, 180])
    for ax in axs:
        ax.set_ylabel("repeatability (%)"); ax.set_ylim(0, 105)
    axs[0].legend(loc="lower left", frameon=False, fontsize=FONT - 4)
    save(fig, OUT, "fig-repeatability.svg")
    for k in names:
        print(" ", k, "zoom", [round(v * 100) for v in d["curves"][k]["zoom"]], "angle", [round(v * 100) for v in d["curves"][k]["angle"]])


def fig_fast():
    ring = [(0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3), (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3)]
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    bright = set(range(2, 12))   # 10 contiguous brighter
    for y in range(-4, 5):
        for x in range(-4, 5):
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc="#f2f2f2", ec="#ccc"))
    for i, (x, y) in enumerate(ring):
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc="#ffd27f" if i in bright else "#9ab", ec="#333"))
        ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=FONT - 5)
    ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, fc="#555", ec="k")); ax.text(0, 0, "c", color="w", ha="center", va="center")
    ax.set_xlim(-4.6, 4.6); ax.set_ylim(4.6, -4.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("FAST: 9 or more neighbours in a row\nall brighter (or all darker) than c ± t", fontsize=FONT - 2)
    save(fig, OUT, "fig-fast.svg")


if __name__ == "__main__":
    for f in [fig_task, fig_harriszoom, fig_scale_covariance, fig_octaves, fig_dog_levels, fig_refine, fig_edge_test, fig_orientation, fig_frames, fig_repeatability, fig_fast]:
        f()
