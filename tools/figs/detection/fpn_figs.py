"""Feature-pyramid figures of the detection/ lecture (slides "Objects of every size" and
"A pyramid inside the network").

Run from the repository root:
  python3 -I tools/figs/detection/fpn_figs.py

fig-fpnscale.png: a synthetic fluorescence scene (small spots, single nuclei, one large cluster)
  with the grid of an early layer (stride 8) and of a deep layer (stride 32); one cell of each
  with the region it sees (illustrative sizes, 40 and 140 px).
fig-fpn.svg: the feature pyramid network as a diagram (bottom-up path, lateral 1x1 connections,
  top-down path with upsampling and addition, boxes predicted at every level).
Nothing is measured; the scene is drawn with a fixed seed.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

OUT = "detection/"
BLUE, ORANGE, RED, GREEN, GREY, INK = "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9a9a9a", "#1b2a3a"
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans"})
rng = np.random.default_rng(7)

# 1. the scene: spots, nuclei, one cluster ------------------------------------------------------
N = 256
yy, xx = np.mgrid[0:N, 0:N].astype(float)
img = np.zeros((N, N))


def blob(cx, cy, r, a=1.0, e=1.0, th=0.0):
    c, s = np.cos(th), np.sin(th)
    u = ((xx - cx) * c + (yy - cy) * s) / r
    v = (-(xx - cx) * s + (yy - cy) * c) / (r * e)
    return a / (1 + np.exp(8 * (np.sqrt(u * u + v * v) - 1)))


cluster = (176, 84, 50)                         # centre x, centre y, radius
for _ in range(400):                            # packed nuclei of the cluster
    ang, rad = rng.uniform(0, 2 * np.pi), cluster[2] * np.sqrt(rng.uniform(0, 1))
    cx, cy = cluster[0] + rad * np.cos(ang), cluster[1] + rad * np.sin(ang)
    img = np.maximum(img, blob(cx, cy, 7, rng.uniform(0.55, 0.9), 0.8, rng.uniform(0, np.pi)) *
                     (np.hypot(xx - cluster[0], yy - cluster[1]) < cluster[2] + 6))
singles = [(40, 50), (82, 120), (52, 190), (120, 210), (200, 200), (236, 160), (110, 40)]
for cx, cy in singles:
    img = np.maximum(img, blob(cx, cy, 8, rng.uniform(0.7, 0.95), 0.8, rng.uniform(0, np.pi)))
spots = [(60, 100), (150, 170), (95, 235), (230, 236), (20, 140), (166, 236)]
for cx, cy in spots:
    img += 1.2 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.3 ** 2))
img = np.clip(img + rng.normal(0, 0.03, img.shape), 0, 1)

fig, axs = plt.subplots(1, 2, figsize=(9.2, 5.1), gridspec_kw={"wspace": 0.06})
for a, stride, (cx, cy), seen, title, sub in [
        (axs[0], 8, spots[1], 40, "early layer: stride 8", "sees a spot, not the cluster"),
        (axs[1], 32, cluster[:2], 140, "deep layer: stride 32", "sees the cluster; a spot\nis 1/100 of one cell")]:
    a.imshow(img, cmap="gray", vmin=0, vmax=1, extent=(0, N, N, 0))
    for g in range(0, N + 1, stride):
        a.plot([g, g], [0, N], c="#ffd84d", lw=0.5 if stride == 8 else 1.2, alpha=0.45)
        a.plot([0, N], [g, g], c="#ffd84d", lw=0.5 if stride == 8 else 1.2, alpha=0.45)
    gx, gy = (cx // stride) * stride, (cy // stride) * stride
    a.add_patch(Rectangle((gx, gy), stride, stride, fill=False, ec="#ffd84d", lw=3))
    mx, my = gx + stride / 2, gy + stride / 2
    a.add_patch(Rectangle((mx - seen / 2, my - seen / 2), seen, seen, fill=False, ec=ORANGE, lw=3, ls="--"))
    a.set_xlim(0, N); a.set_ylim(N, 0); a.set_xticks([]); a.set_yticks([])
    a.set_title(title, fontsize=19)
    a.text(0.5, -0.02, sub, transform=a.transAxes, ha="center", va="top", fontsize=17)
fig.text(0.5, 0.93, "yellow: one cell of the map;  orange dashed: the region it sees",
         ha="center", va="bottom", fontsize=15, color="#555")
fig.savefig(OUT + "fig-fpnscale.png", dpi=100, bbox_inches="tight", facecolor="white")
plt.close(fig)

# 2. the feature pyramid network as a diagram ---------------------------------------------------
levels = [  # level, stride, y centre, drawn size, darkness of the backbone map
    (2, 4, 1.4, 1.6, 0.18), (3, 8, 3.6, 1.15, 0.40), (4, 16, 5.45, 0.8, 0.65), (5, 32, 7.0, 0.55, 0.95)]
XL, XR, XO = 1.9, 4.9, 6.15
what = {2: "small spots", 3: "nuclei", 4: "", 5: "large clusters"}
R = 0.13


def fmap(x, y, s, dark):
    for k in (2, 1):                           # two channels behind: a stack of maps
        o = 0.07 * k
        ax.add_patch(Rectangle((x - s / 2 + o, y - s / 2 + o), s, s, fc="white", ec="#777", lw=1))
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, fc=BLUE, alpha=dark, ec="none"))
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, fill=False, ec="#444", lw=1.3))


def arrow(p, q, c=INK, lw=2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, color=c, lw=lw,
                                 shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(7.4, 7.5))
for i, (k, st, y, s, dark) in enumerate(levels):
    fmap(XL, y, s, dark)
    fmap(XR, y, s, 0.9)
    ax.text(XL - s / 2 - 0.15, y, f"stride {st}", ha="right", va="center", fontsize=15)
    # lateral 1x1 connection
    arrow((XL + s / 2 + 0.2, y), (XR - s / 2 - 0.05, y), c=GREEN)
    ax.text((XL + XR) / 2 + 0.1, y + 0.08, "1×1", ha="center", va="bottom", fontsize=15, color=GREEN)
    # prediction head
    arrow((XR + s / 2 + 0.2, y), (XO - 0.05, y), c=ORANGE)
    ax.text(XO + 0.05, y, "boxes" + (f"\n{what[k]}" if what[k] else ""), ha="left", va="center",
            fontsize=15, color="#b85a00")
    if i + 1 < len(levels):
        k2, st2, y2, s2, _ = levels[i + 1]
        # bottom-up: the backbone
        arrow((XL, y + s / 2 + 0.18), (XL, y2 - s2 / 2 - 0.12), c=GREY)
        # top-down: upsample x2, then add
        cy = y + s / 2 + 0.18 + R
        arrow((XR, y2 - s2 / 2 - 0.05), (XR, cy + R), c=BLUE)
        ax.add_patch(Circle((XR, cy), R, fc="white", ec=BLUE, lw=2))
        ax.text(XR, cy, "+", ha="center", va="center", fontsize=16, color=BLUE, fontweight="bold")
        ax.text(XR + 0.12, (y2 - s2 / 2 + cy + R) / 2, "×2 up", ha="left", va="center", fontsize=14, color=BLUE)
arrow((XL, 0.05), (XL, levels[0][2] - levels[0][3] / 2 - 0.12), c=GREY)
ax.text(XL + 0.1, 0.05, "image", ha="left", va="bottom", fontsize=15)
ax.text(XL, 7.75, "bottom-up\n(the CNN)", ha="center", va="bottom", fontsize=16, fontweight="bold")
ax.text(XR, 7.75, "top-down", ha="center", va="bottom", fontsize=16, fontweight="bold", color=BLUE)
ax.text(XL - 1.6, -0.45, "darker = knows more about what is there", ha="left", va="center",
        fontsize=14, color="#555")
ax.set_xlim(-0.2, 7.6); ax.set_ylim(-0.7, 8.45); ax.set_aspect("equal"); ax.axis("off")
fig.savefig(OUT + "fig-fpn.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)
