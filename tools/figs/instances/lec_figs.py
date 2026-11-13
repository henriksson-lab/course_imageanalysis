"""Slide figures of the instances/ lecture (modern deep learning day).

Run from the repository root:
  for p in semantic gap dist elong cshape flows; do tools/figs/instances/demo_shot.sh SHOTS instancemaps $p im_$p; done
  tools/figs/instances/demo_shot.sh SHOTS nmsboxes dup nb_dup 1100
  python3 -I tools/figs/instances/lec_figs.py SHOTS

Demo panels are crops of the instancemaps / nmsboxes demos in their presets (the numbers in
the slide text are the demos' readouts in these states). The flow-field figure and the panoptic
figure are computed here (synthetic shapes).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy import ndimage as ndi

OUT = "instances/"
SHOTS = sys.argv[1]
BLUE, ORANGE, RED, GREY = "#1f77b4", "#ff7f0e", "#d62728", "#9a9a9a"
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans"})

# instancemaps panels, CSS px of the demo at 1220 px window width (screenshots at scale 2)
IM = {"input": (10, 50, 287, 327), "map": (295, 50, 572, 327), "objects": (580, 50, 857, 327)}


def crop(shot, box, pad=0):
    im = Image.open(f"{SHOTS}/{shot}.png").convert("RGB")
    return np.asarray(im.crop((2 * box[0] + pad, 2 * box[1] + pad, 2 * box[2] - pad, 2 * box[3] - pad)))


def panels(cells, out, size, ncol=None, sub_size=16):
    ncol = ncol or len(cells)
    nrow = (len(cells) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=size, squeeze=False, gridspec_kw={"wspace": 0.05, "hspace": 0.25})
    for a in axs.flat:
        a.axis("off")
    for a, (img, title, sub) in zip(axs.flat, cells):
        a.imshow(img)
        a.set_title(title, fontsize=17)
        if sub:
            a.text(0.5, -0.03, sub, transform=a.transAxes, ha="center", va="top", fontsize=sub_size)
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1. semantic mask merges touching cells -----------------------------------------------------------
panels([(crop("im_semantic", IM["input"]), "image", "14 cells"),
        (crop("im_semantic", IM["objects"]), "connected pieces of the mask", "11 objects, 3 merges (M)")],
       OUT + "fig-semantic.png", (7.2, 4.0))

# 2. four maps and their classical steps ---------------------------------------------------------
panels([(crop("im_gap", IM["map"]), "boundary class", "→ label interiors"),
        (crop("im_dist", IM["map"]), "distance map", "→ watershed"),
        (crop("im_cshape", IM["map"]), "objectness + rays", "→ suppression"),
        (crop("im_flows", IM["map"]), "flow field", "→ follow arrows")],
       OUT + "fig-maps.png", (11, 3.6))

# 3.-5. one route each: map and objects ----------------------------------------------------------
panels([(crop("im_gap", IM["map"]), "interior / boundary", "gaps at contacts"),
        (crop("im_gap", IM["objects"]), "objects", "11 objects, 3 merges (M)")],
       OUT + "fig-boundary.png", (7.2, 4.0))
panels([(crop("im_elong", IM["map"]), "distance (noise 0.2)", "peaks on long ridges"),
        (crop("im_elong", IM["objects"]), "objects (h = 0.1)", "20 objects, 2 splits (S)")],
       OUT + "fig-distance.png", (7.2, 4.0))
panels([(crop("im_cshape", IM["map"]), "objectness + 32 rays", "rays leave the C"),
        (crop("im_cshape", IM["objects"]), "objects", "the C is split (S)")],
       OUT + "fig-star.png", (7.2, 4.0))


# 6. flow field in a C-shaped cell (computed like the demo's flow target) ----------------------
n = 64
yy, xx = np.mgrid[0:n, 0:n]
r = np.hypot(xx - 32, yy - 32)
ang = np.degrees(np.arctan2(yy - 32, xx - 32))
mask = (r > 13) & (r < 25) & (np.abs(ang) > 50)
D = ndi.distance_transform_edt(mask)
cy, cx = np.argwhere(mask).mean(0)
score = D - 1e-3 * np.hypot(xx - cx, yy - cy)
score[~mask] = -np.inf
sy, sx = np.unravel_index(np.argmax(score), score.shape)
h = np.zeros((n, n))
for _ in range(2 * 2 * 50 + 20):
    h[sy, sx] += 1
    s = h.copy(); c = np.ones((n, n))
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nb = np.roll(np.roll(h, dy, 0), dx, 1)
        nm = np.roll(np.roll(mask, dy, 0), dx, 1)
        s += np.where(nm, nb, 0); c += nm
    h = np.where(mask, s / c, 0)
lg = np.log(h + 1e-30)


def nbv(a, dy, dx):
    v = np.roll(np.roll(a, -dy, 0), -dx, 1)
    m = np.roll(np.roll(mask, -dy, 0), -dx, 1)
    return np.where(m, v, a)


gx = nbv(lg, 0, 1) - nbv(lg, 0, -1)
gy = nbv(lg, 1, 0) - nbv(lg, -1, 0)
m = np.hypot(gx, gy) + 1e-12
vx, vy = np.where(mask, gx / m, 0), np.where(mask, gy / m, 0)
vx[sy, sx] = vy[sy, sx] = 0

fig, ax = plt.subplots(figsize=(5.0, 5.0))
ax.imshow(np.where(mask, lg - lg[mask].min(), np.nan), cmap="Greys", alpha=0.35, origin="upper")
ax.contour(mask, levels=[0.5], colors="black", linewidths=1.5)
sel = mask & (xx % 3 == 0) & (yy % 3 == 0)
ax.quiver(xx[sel], yy[sel], vx[sel], vy[sel], color=BLUE, scale=28, width=0.006, headwidth=4)
starts = [(23, 52), (12, 30), (22, 9), (40, 50), (45, 14)]
for (px, py) in starts:
    path = [(px, py)]
    x, y = float(px), float(py)
    for _ in range(80):
        i, j = int(round(y)), int(round(x))
        x, y = x + vx[i, j], y + vy[i, j]
        path.append((x, y))
    P = np.array(path)
    ax.plot(P[:, 0], P[:, 1], c=ORANGE, lw=2.5)
    ax.plot(px, py, "o", c=ORANGE, ms=7)
ax.plot(sx, sy, "*", c=RED, ms=20, mec="black")
ax.annotate("sink", (sx, sy), xytext=(4, 8), fontsize=18, ha="left",
            arrowprops=dict(arrowstyle="->", color="black"))
ax.set_xlim(2, 62); ax.set_ylim(62, 2); ax.axis("off")
fig.savefig(OUT + "fig-cflow.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)

# 7. detection: kept boxes in the sparse scene --------------------------------------------------
panels([(crop("nb_dup", (10, 26, 480, 496)), "", "")], OUT + "fig-boxes.png", (4.6, 4.6))

# 8. panoptic: stuff classes and thing identities (synthetic tissue) ---------------------------
rng = np.random.default_rng(5)
n = 160
yy, xx = np.mgrid[0:n, 0:n]
cls = np.zeros((n, n), int)                         # 0 background (slide glass)
stroma = (xx + 0.6 * yy) > 55
cls[stroma] = 1                                      # 1 stroma
ell = ((xx - 98) / 38) ** 2 + ((yy - 92) / 28) ** 2
cls[ell < 1.0] = 2                                   # 2 epithelium (counted as stuff here)
cls[ell < 0.45] = 3                                  # 3 lumen
inst = np.zeros((n, n), int)
k = 0
for t in np.linspace(0, 2 * np.pi, 15, endpoint=False):
    cx_, cy_ = 98 + 31 * np.cos(t), 92 + 22.5 * np.sin(t)
    k += 1
    inst[((xx - cx_) / 5.2) ** 2 + ((yy - cy_) / 4.0) ** 2 < 1] = k
for _ in range(9):                                   # scattered stromal nuclei
    while True:
        cx_, cy_ = rng.uniform(8, 152, 2)
        if stroma[int(cy_), int(cx_)] and ((cx_ - 98) / 44) ** 2 + ((cy_ - 92) / 34) ** 2 > 1:
            break
    k += 1
    a = rng.uniform(0, np.pi)
    u, v = (xx - cx_) * np.cos(a) + (yy - cy_) * np.sin(a), -(xx - cx_) * np.sin(a) + (yy - cy_) * np.cos(a)
    inst[(u / 6.5) ** 2 + (v / 2.6) ** 2 < 1] = k
nuc = inst > 0
grey = np.choose(cls, [0.95, 0.72, 0.6, 0.9]).astype(float)
fib = ndi.gaussian_filter(rng.standard_normal((n, n)), (1, 6))
grey += np.where(cls == 1, 0.08 * fib / fib.std(), 0)
grey[nuc] = 0.25
img = np.clip(ndi.gaussian_filter(grey, 0.8) + 0.02 * rng.standard_normal((n, n)), 0, 1)

stuff_col = {0: (1, 1, 1), 1: (0.94, 0.82, 0.62), 2: (0.80, 0.90, 0.80), 3: (0.75, 0.87, 0.97)}
sem = np.zeros((n, n, 3))
for c_, col in stuff_col.items():
    sem[cls == c_] = col
sem_nuc = sem.copy(); sem_nuc[nuc] = (0.0, 0.30, 0.55)
pan = sem.copy()
tab = plt.get_cmap("tab20").colors
for i in range(1, k + 1):
    pan[inst == i] = tab[(2 * i) % 20]
edges = nuc & ~ndi.binary_erosion(inst > 0) | (ndi.maximum_filter(inst, 3) != ndi.minimum_filter(inst, 3)) & nuc
pan[edges] = (0, 0, 0)

fig, axs = plt.subplots(1, 3, figsize=(11, 4.4), gridspec_kw={"wspace": 0.05})
for a, im_, t_ in [(axs[0], img, "image"), (axs[1], sem_nuc, "semantic"),
                   (axs[2], pan, "panoptic")]:
    a.imshow(im_, cmap="gray", vmin=0, vmax=1); a.set_title(t_, fontsize=17)
    a.set_xticks([]); a.set_yticks([])
from matplotlib.patches import Patch
handles = [Patch(color=stuff_col[1], label="stroma"), Patch(color=stuff_col[2], label="epithelium"),
           Patch(color=stuff_col[3], label="lumen"), Patch(color=(0.0, 0.30, 0.55), label="nucleus (thing)")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=16, bbox_to_anchor=(0.5, -0.04))
fig.savefig(OUT + "fig-panoptic.png", dpi=100, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("things:", k)
