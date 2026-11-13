"""Slide figures of the cellpose/ lecture (networks for cells and objects).

Run from the repository root:
  python3 -I tools/figs/cellpose/lec_figs.py            # synthetic figures
  tools/figs/cellpose/shots.sh SHOTS                     # real-model screenshots (make wasm, make models)
  python3 -I tools/figs/cellpose/lec_figs.py SHOTS       # + the figures made from the screenshots

The heat target and the arrows are computed as in demos/src/flowtrack/model.ts (source = cell
pixel nearest the median of the coordinates; "add 1 at the source, then average over the 3 x 3
neighbours inside the cell"; arrows = normalised central differences, outside the cell = 0).
fig-cflow.svg is the instances/ figure (copied: the recall slide). Sample images: the files that
`make models` puts into demos/dist/models/images/ (credits in demos/dist/models/index.json).
"""
import json, os, shutil, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image

OUT = "cellpose/"
SHOTS = sys.argv[1] if len(sys.argv) > 1 else None
BLUE, ORANGE, RED, GREEN, GREY = "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9a9a9a"
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans"})
N = 48


def c_shape():
    y, x = np.mgrid[0:N, 0:N]
    X, Y = x - 23.5, y - 23.5
    r = np.hypot(X, Y)
    a = np.abs(np.degrees(np.arctan2(Y, X)))
    return ((r <= 18) & (r >= 9) & (a > 55)).astype(int)


def source(lab, l=1):
    ys, xs = np.nonzero(lab == l)
    mx, my = np.median(xs), np.median(ys)
    k = np.argmin((xs - mx) ** 2 + (ys - my) ** 2 + 1e-9 * (ys * N + xs))
    return ys[k], xs[k]


def heat(lab, iters, l=1):
    m = lab == l
    sy, sx = source(lab, l)
    h = np.zeros(lab.shape)
    cnt = np.zeros(lab.shape)
    pad = np.pad(m, 1)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cnt += pad[1 + dy:1 + dy + N, 1 + dx:1 + dx + N]
    for _ in range(iters):
        h[sy, sx] += 1
        hp = np.pad(h * m, 1)
        s = np.zeros(lab.shape)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                s += hp[1 + dy:1 + dy + N, 1 + dx:1 + dx + N]
        h = np.where(m, s / np.maximum(cnt, 1), 0)
    return h


def arrows_from_heat(lab, h, l=1):
    m = lab == l
    H = np.pad(np.where(m, h, 0), 1)
    gx = H[1:-1, 2:] - H[1:-1, :-2]
    gy = H[2:, 1:-1] - H[:-2, 1:-1]
    mag = np.hypot(gx, gy)
    vx = np.where(m & (mag > 0), gx / np.where(mag > 0, mag, 1), 0)
    vy = np.where(m & (mag > 0), gy / np.where(mag > 0, mag, 1), 0)
    return vx, vy


def bilinear(a, x, y):
    x = min(max(x, 0), N - 1); y = min(max(y, 0), N - 1)
    x0, y0 = min(int(x), N - 2), min(int(y), N - 2)
    fx, fy = x - x0, y - y0
    return (1 - fy) * ((1 - fx) * a[y0, x0] + fx * a[y0, x0 + 1]) + fy * ((1 - fx) * a[y0 + 1, x0] + fx * a[y0 + 1, x0 + 1])


def path(vx, vy, x, y, steps=60):
    P = [(x, y)]
    for _ in range(steps):
        u, v = bilinear(vx, x, y), bilinear(vy, x, y)
        x = min(max(x + u, 0), N - 1); y = min(max(y + v, 0), N - 1)
        P.append((x, y))
    return np.array(P)


def heat_rgb(lab, h):
    from matplotlib import colormaps
    t = np.zeros(lab.shape)
    mx = h.max()
    pos = (lab > 0) & (h > 0)
    t[pos] = np.clip(1 + np.log10(h[pos] / mx) / 2.5, 0, 1)
    rgb = colormaps["inferno"](0.08 + 0.92 * t)[..., :3]
    rgb[lab == 0] = [0.11, 0.11, 0.125]
    return rgb


def outline(ax, lab, color="w", lw=1.2):
    ax.contour(lab > 0, levels=[0.5], colors=color, linewidths=lw)


def quiver(ax, lab, vx, vy, step=2, color="w"):
    ys, xs = np.mgrid[0:N:step, 0:N:step]
    m = lab[ys, xs] > 0
    ax.quiver(xs[m], ys[m], vx[ys, xs][m], -vy[ys, xs][m], color=color, scale=1 / 1.6, scale_units="xy", angles="xy",
              width=0.006, headwidth=4, headlength=4, pivot="middle")


def fig_heat1d():
    steps = []
    h = [0.0] * 5
    for _ in range(3):
        h[2] += 1
        h = [sum(h[j] for j in (i - 1, i, i + 1) if 0 <= j < 5) / len([j for j in (i - 1, i, i + 1) if 0 <= j < 5]) for i in range(5)]
        steps.append(h[:])
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True, gridspec_kw={"wspace": 0.12})
    for k, (ax, hv) in enumerate(zip(axs, steps), 1):
        ax.bar(range(1, 6), hv, color=ORANGE, edgecolor="k")
        for i, v in enumerate(hv, 1):
            ax.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=14)
        ax.set_title(f"after step {k}", fontsize=16)
        ax.set_xticks(range(1, 6)); ax.set_ylim(0, 1.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("pixel")
    axs[0].set_ylabel("heat")
    # arrows below the last step
    arrs = ["→", "→", "●", "←", "←"]
    for i, a in enumerate(arrs, 1):
        axs[2].text(i, 1.17, a, ha="center", va="center", fontsize=20, color=BLUE if a != "●" else RED)
    axs[2].text(5.6, 1.17, "uphill", ha="left", va="center", fontsize=14, color=BLUE)
    fig.savefig(OUT + "fig-heat1d.svg", bbox_inches="tight")
    plt.close(fig)


def fig_heat_c():
    lab = c_shape()
    fig, axs = plt.subplots(1, 4, figsize=(14, 4.0), gridspec_kw={"wspace": 0.05})
    sy, sx = source(lab)
    for ax, k in zip(axs[:3], (10, 50, 200)):
        h = heat(lab, k)
        ax.imshow(heat_rgb(lab, h), interpolation="nearest")
        outline(ax, lab)
        ax.plot(sx, sy, "*", color=RED, ms=16, mec="k")
        ax.set_title(f"heat after {k} steps", fontsize=15)
        ax.axis("off")
    h = heat(lab, 200)
    vx, vy = arrows_from_heat(lab, h)
    ax = axs[3]
    bg = np.where(lab[..., None] > 0, 0.85, 0.11) * np.ones((N, N, 3))
    ax.imshow(bg, interpolation="nearest")
    quiver(ax, lab, vx, vy, color=BLUE)
    outline(ax, lab, "k")
    for (x, y) in [(30, 12), (30, 35), (8, 23)]:
        P = path(vx, vy, x, y)
        ax.plot(P[:, 0], P[:, 1], color=ORANGE, lw=3)
        ax.plot(x, y, "o", color=ORANGE, mec="k", ms=8)
    ax.plot(sx, sy, "*", color=RED, ms=16, mec="k")
    ax.set_title("arrows uphill, paths", fontsize=15)
    ax.axis("off")
    fig.savefig(OUT + "fig-heat-c.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_straight():
    lab = c_shape()
    sy, sx = source(lab)
    y, x = np.mgrid[0:N, 0:N]
    dx, dy = sx - x, sy - y
    m = np.hypot(dx, dy)
    svx = np.where((lab > 0) & (m > 0), dx / np.where(m > 0, m, 1), 0)
    svy = np.where((lab > 0) & (m > 0), dy / np.where(m > 0, m, 1), 0)
    hvx, hvy = arrows_from_heat(lab, heat(lab, 200))
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.6), gridspec_kw={"wspace": 0.05})
    for ax, (vx, vy), t in zip(axs, [(svx, svy), (hvx, hvy)], ["straight to the source", "uphill in the heat"]):
        bg = np.where(lab[..., None] > 0, 0.85, 0.11) * np.ones((N, N, 3))
        ax.imshow(bg, interpolation="nearest")
        quiver(ax, lab, vx, vy, color=BLUE)
        outline(ax, lab, "k")
        ax.plot(sx, sy, "*", color=RED, ms=16, mec="k")
        ax.set_title(t)
        ax.axis("off")
    # tip -> source: a straight line leaves the cell
    axs[0].plot([30, sx], [12, sy], color=ORANGE, lw=3, ls="--")
    axs[0].plot(30, 12, "o", color=ORANGE, mec="k", ms=8)
    axs[0].text(28, 25, "leaves\nthe cell", color="w", fontsize=15, ha="center", fontweight="bold")
    P = path(hvx, hvy, 30, 12)
    axs[1].plot(P[:, 0], P[:, 1], color=ORANGE, lw=3)
    axs[1].plot(30, 12, "o", color=ORANGE, mec="k", ms=8)
    fig.savefig(OUT + "fig-straight.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_grid():
    F = ["→←↓←←", "→↘↓↙←", "→→·←←", "↗↑↑↘↓", "↗↑↑→·"]
    D = {"→": (0, 1), "←": (0, -1), "↑": (-1, 0), "↓": (1, 0), "↗": (-1, 1), "↖": (-1, -1), "↘": (1, 1), "↙": (1, -1), "·": (0, 0)}
    fig, ax = plt.subplots(figsize=(5.2, 5.4))
    for r in range(5):
        for c in range(5):
            ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, ec="#bbb", lw=1))
            ax.text(c + 0.5, r + 0.5, F[r][c] if F[r][c] != "·" else "•", ha="center", va="center", fontsize=24, color="#333")
    for i in range(5):
        ax.text(i + 0.5, -0.25, str(i), ha="center", va="center", fontsize=14, color=GREY)
        ax.text(-0.25, i + 0.5, str(i), ha="center", va="center", fontsize=14, color=GREY)

    def follow(p):
        P = [p]
        while True:
            dy, dx = D[F[p[0]][p[1]]]
            q = (min(4, max(0, p[0] + dy)), min(4, max(0, p[1] + dx)))
            if q in P: return P + [q]
            P.append(q); p = q
    for (start, col, off) in [((4, 0), ORANGE, 0.0), ((3, 3), GREEN, 0.0), ((0, 0), RED, 0.12)]:
        P = follow(start)
        for a, b in zip(P[:-1], P[1:]):
            ax.add_patch(FancyArrowPatch((a[1] + 0.5, a[0] + 0.5 + off), (b[1] + 0.5, b[0] + 0.5 + off), arrowstyle="-|>",
                                         mutation_scale=18, color=col, lw=3, shrinkA=8, shrinkB=8))
        ax.add_patch(plt.Circle((start[1] + 0.5, start[0] + 0.5), 0.36, fill=False, ec=col, lw=3, zorder=5))
    ax.set_xlim(-0.5, 5.05); ax.set_ylim(5.05, -0.5); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(OUT + "fig-grid.svg", bbox_inches="tight")
    plt.close(fig)


def fig_samples():
    root = "demos/dist/models/"
    idx = json.load(open(root + "index.json"))
    items = [("images/cellpose-img02.png", "cells (2 channels)"), ("images/cellpose-img03.png", "membranes"),
             ("images/bbbc039-A02_s1.png", "nuclei"), ("images/dsb-histology.png", "stained tissue")]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.0), gridspec_kw={"wspace": 0.04})
    credits = []
    for ax, (f, t) in zip(axs, items):
        im = np.asarray(Image.open(root + f)).astype(float)
        if im.ndim == 2:
            lo, hi = np.percentile(im, [1, 99.5]); im = np.clip((im - lo) / (hi - lo), 0, 1)
            ax.imshow(im, cmap="gray")
        else:
            im = im[..., :3]
            if f.endswith("img02.png"):
                g = im[..., 1]; b = im[..., 2]
                n = lambda a: np.clip((a - np.percentile(a, 1)) / (np.percentile(a, 99.5) - np.percentile(a, 1)), 0, 1)
                im = np.stack([np.zeros_like(g), n(g), n(b)], -1)
            else:
                im = im / 255
            ax.imshow(im)
        s = min(im.shape[0], im.shape[1])
        ax.set_xlim(0, s); ax.set_ylim(s, 0)
        ax.set_title(t); ax.axis("off")
        credits.append(idx[f]["credit"])
    fig.savefig(OUT + "fig-samples.png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    print("credits:", *sorted(set(credits)), sep="\n  ")


def crop(shot, box=None):
    im = Image.open(os.path.join(SHOTS, shot)).convert("RGB")
    return np.asarray(im if box is None else im.crop(box))


def row(items, out, size, dpi=100):
    fig, axs = plt.subplots(1, len(items), figsize=size, gridspec_kw={"wspace": 0.04})
    for ax, (im, t) in zip(np.atleast_1d(axs), items):
        ax.imshow(im); ax.set_title(t, fontsize=16); ax.axis("off")
    fig.savefig(OUT + out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fig_shots():
    # stage canvases clipped by shots.sh (cellposenet .cp-stage)
    row([(crop("cells-input.png"), "input (cytoplasm green, nuclei blue)"), (crop("cells-flows.png"), "predicted arrows (colour = direction)")],
        "fig-cp-flows.png", (12, 4.8))
    row([(crop("cells-track.png"), "following the arrows")], "fig-cp-track.png", (6.5, 5.2))
    row([(crop("cells-check.png"), "flow check")], "fig-cp-check.png", (6.5, 5.2))
    row([(crop("nuclei-d8-masks.png"), "diameter 8: F1 0.84"), (crop("nuclei-d29-masks.png"), "diameter 29 (right): F1 0.92"),
         (crop("nuclei-d100-masks.png"), "diameter 100: F1 0.14")], "fig-cp-diameter.png", (15, 4.2))
    row([(crop("cells-masks.png"), "default (200 steps)"), (crop("cells-10-check.png"), "10 steps: kept (green), dropped (red)")],
        "fig-cp-steps.png", (12, 4.8))
    row([(crop("tissue-inv-masks.png"), "inverted (dark nuclei → bright)"), (crop("tissue-noinv-masks.png"), "not inverted")],
        "fig-cp-tissue.png", (10, 4.4))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    shutil.copy("instances/fig-cflow.svg", OUT + "fig-cflow.svg")
    fig_heat1d(); fig_heat_c(); fig_straight(); fig_grid(); fig_samples()
    if SHOTS:
        fig_shots()
