"""Slide figures of the stardist/ lecture (networks for cells and objects day).

Run from the repository root (after `make wasm` and `make models` for the real-model parts):
  S=SHOTS; M=MEASURE
  W="document.querySelector('.qb-readout') && document.querySelector('.qb-readout').textContent.includes('after NMS')"
  for v in image prob ray cand nms labels; do
    tools/figs/stardist/demo_shot.sh $S sd_$v "d=stardistnet&image=nuclei&view=$v" --wait "$W" --size 1220x800 --clip .sd-stage --timeout 90; done
  for p in low high strong weak; do
    tools/figs/stardist/demo_shot.sh $S sr_$p "d=starrays&preset=$p" --size 1220x800 --clip .sr-panels; done
  tools/figs/stardist/demo_shot.sh $S im_cshape "d=instancemaps&preset=cshape" --size 1220x1000
  python3 -I tools/figs/stardist/measure_prep.py $M
  node_modules/.bin/esbuild tools/figs/stardist/measure.ts --bundle --platform=node --log-level=warning | node - $M > tools/figs/stardist/measured.json
  python3 -I tools/figs/stardist/lec_figs.py $S $M

Synthetic figures (rays, rectangle, shapes, touching nuclei vs boxes) are computed here with the
same geometry as the starrays demo (exact grid traversal for the rays, pixel centres for the
polygon). Real-image figures: crops of the stardistnet demo (the real network in the browser) and
label images of the same network written by measure.ts. Images: StarDist test image (BSD-3-Clause);
2018 Data Science Bowl (BBBC038v1, CC0); BBBC039 (CC0).
"""
import json, shutil, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon as MPoly, Rectangle
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

OUT = "stardist/"
SHOTS, MEAS = sys.argv[1], sys.argv[2]
BLUE, ORANGE, RED, GREEN, GREY, GOLD = "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9a9a9a", "#b8860b"
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans"})


# ---- geometry (as demos/src/starrays/geom.ts) ------------------------------------------------
def ray_dir(k, n):
    a = 2 * np.pi * k / n
    return np.cos(a), np.sin(a)


def ray_length(lab, cx, cy, dx, dy, lid=1):
    h, w = lab.shape
    x, y = int(np.floor(cx)), int(np.floor(cy))
    if not (0 <= x < w and 0 <= y < h) or lab[y, x] != lid:
        return 0.0
    sx, sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
    ddx = abs(1 / dx) if abs(dx) > 1e-12 else np.inf
    ddy = abs(1 / dy) if abs(dy) > 1e-12 else np.inf
    tx = np.inf if ddx == np.inf else ((x + 1 - cx) if dx > 0 else (cx - x)) * ddx
    ty = np.inf if ddy == np.inf else ((y + 1 - cy) if dy > 0 else (cy - y)) * ddy
    while True:
        if tx < ty:
            t = tx; x += sx; tx += ddx
        else:
            t = ty; y += sy; ty += ddy
        if not (0 <= x < w and 0 <= y < h) or lab[y, x] != lid:
            return t


def rays_at(lab, cx, cy, n, lid=1):
    return np.array([ray_length(lab, cx, cy, *ray_dir(k, n), lid) for k in range(n)])


def tips(cx, cy, r):
    n = len(r)
    return np.array([[cx + r[k] * ray_dir(k, n)[0], cy + r[k] * ray_dir(k, n)[1]] for k in range(n)])


def poly_mask(cx, cy, r, shape):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.c_[xx.ravel() + 0.5 + 1.37e-6, yy.ravel() + 0.5 + 2.71e-6]
    return Path(tips(cx, cy, r)).contains_points(pts).reshape(h, w)


def poly_area(r):
    n = len(r)
    return 0.5 * np.sin(2 * np.pi / n) * np.sum(r * np.roll(r, -1))


def raster(polys, n=64):
    yy, xx = np.mgrid[0:n, 0:n]
    pts = np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5]
    m = np.zeros(n * n, bool)
    for p in polys:
        m ^= Path(np.asarray(p)).contains_points(pts)
    return m.reshape(n, n).astype(np.uint8)


def objectness(mask):
    d = ndi.distance_transform_edt(mask)
    return d / d.max() if d.max() > 0 else d


def fit(mask, cx, cy, n):
    r = rays_at(mask, cx, cy, n)
    pm = poly_mask(cx, cy, r, mask.shape)
    m = mask.astype(bool)
    inter = (pm & m).sum()
    return r, pm, inter / (pm | m).sum()


def best_centre(mask, n):
    best = (-1, None)
    ob = objectness(mask)
    for y, x in np.argwhere(mask):
        r, pm, iou = fit(mask, x + 0.5, y + 0.5, n)
        key = iou + 1e-6 * ob[y, x]
        if key > best[0]:
            best = (key, (x + 0.5, y + 0.5, iou))
    return best[1]


def ellipse_poly(cx, cy, rx, ry, n=64):
    t = 2 * np.pi * np.arange(n) / n
    return np.c_[cx + rx * np.cos(t), cy + ry * np.sin(t)]


def arc_band(cx, cy, r0, r1, a0, a1, m=40):
    a = np.radians(np.linspace(a0, a1, m + 1))
    return np.r_[np.c_[cx + r1 * np.cos(a), cy + r1 * np.sin(a)], np.c_[cx + r0 * np.cos(a[::-1]), cy + r0 * np.sin(a[::-1])]]


def hatch_pixels(ax, m, color, hatch):
    for y, x in np.argwhere(m):
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor="none", edgecolor=color, hatch=hatch, lw=0))


def draw_fit(ax, mask, cx, cy, n, title, sub, rays_lw=0.8, show_obj=False):
    r, pm, iou = fit(mask, cx, cy, n)
    m = mask.astype(bool)
    if show_obj:
        ax.imshow(np.where(m, objectness(mask), np.nan), cmap="viridis", vmin=0, vmax=1, extent=(0, mask.shape[1], mask.shape[0], 0))
    else:
        ax.imshow(np.where(m, 1.0, np.nan), cmap="Blues", vmin=0, vmax=3, extent=(0, mask.shape[1], mask.shape[0], 0))
    plt.rcParams["hatch.linewidth"] = 1.2
    hatch_pixels(ax, m & ~pm, RED, "////")
    hatch_pixels(ax, pm & ~m, BLUE, "\\\\\\\\")
    T = tips(cx, cy, r)
    for (x, y) in T:
        ax.plot([cx, x], [cy, y], c=GOLD if not show_obj else "white", lw=rays_lw)
    ax.add_patch(MPoly(T, closed=True, fill=False, ec=RED, lw=2.2))
    ax.plot(cx, cy, "o", ms=9, mfc="#ffd400", mec="black")
    ax.set_xlim(0, mask.shape[1]); ax.set_ylim(mask.shape[0], 0); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=17)
    if sub:
        ax.text(0.5, -0.02, sub.format(iou=iou), transform=ax.transAxes, ha="center", va="top", fontsize=15)
    return iou


# 1. recall figure: reused from instances/ -----------------------------------------------------
shutil.copy("instances/fig-star.png", OUT + "fig-star.png")

# 2. what the network predicts: rays from a central and from a border pixel ---------------------
nuc = raster([ellipse_poly(32, 32, 22, 15, 96)])
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, (cx, cy), t in zip(axs, [(32.5, 32.5), (12.5, 36.5)], ["central pixel", "near the border"]):
    ob = objectness(nuc)[int(cy), int(cx)]
    draw_fit(ax, nuc, cx, cy, 32, f"{t}: objectness {ob:.2f}", "polygon IoU {iou:.2f} (exact rays)", show_obj=True)
fig.savefig(OUT + "fig-rays.svg", bbox_inches="tight", facecolor="white"); plt.close(fig)

# 3. worked example: the 8 x 4 rectangle with 8 and 32 rays (exact, continuous) ---------------
fig, axs = plt.subplots(2, 1, figsize=(5.2, 6.4))
for ax, n in zip(axs, [8, 32]):
    r = []
    for k in range(n):
        c, s = abs(np.cos(2 * np.pi * k / n)), abs(np.sin(2 * np.pi * k / n))
        r.append(min(4 / c if c > 1e-12 else np.inf, 2 / s if s > 1e-12 else np.inf))
    r = np.array(r); T = tips(0, 0, r); A = poly_area(r)
    ax.add_patch(Rectangle((-4, -2), 8, 4, fc="#dbe7f3", ec="black", lw=1.5))
    ax.add_patch(MPoly(T, closed=True, fc="none", ec=RED, lw=2.2))
    for (x, y) in T:
        ax.plot([0, x], [0, y], c=GOLD, lw=1)
    if n == 8:
        for (x, y), v in zip(T[:3], r[:3]):
            ax.text(x * 0.55 + (0.15 if y else 0), y * 0.55 - 0.15, f"{v:.2f}".rstrip("0").rstrip("."), fontsize=14, color="black")
    ax.plot(0, 0, "o", ms=8, mfc="#ffd400", mec="black")
    ax.set_xlim(-4.3, 4.3); ax.set_ylim(2.3, -2.3); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{n} rays: area {A:.1f} of 32, IoU {A / 32:.2f}", fontsize=16)
fig.savefig(OUT + "fig-rect.svg", bbox_inches="tight", facecolor="white"); plt.close(fig)

# 4. which shapes fit: ellipse, L from its corner and from an arm's end, C, ring ----------------
L = raster([[(12, 8), (26, 8), (26, 42), (54, 42), (54, 56), (12, 56)]])
C = raster([arc_band(34, 32, 12, 23, 40, 320)])
R = raster([ellipse_poly(32, 32, 23, 23, 64), ellipse_poly(32, 32, 11, 11, 48)])
E = raster([ellipse_poly(32, 32, 24, 13, 64)])
cases = [(E, (32.5, 32.5), "ellipse", "IoU {iou:.2f}: yes"),
         (L, (19.5, 49.5), "L from its corner", "IoU {iou:.2f}: yes"),
         (L, (19.5, 12.5), "L from an arm's end", "IoU {iou:.2f}: no"),
         (C, None, "C, best centre", "IoU {iou:.2f}: no"),
         (R, None, "ring, best centre", "IoU {iou:.2f}: no")]
fig, axs = plt.subplots(1, 5, figsize=(16, 3.9))
for ax, (m, c, t, s) in zip(axs, cases):
    if c is None:
        bx, by, _ = best_centre(m, 32); c = (bx, by)
    draw_fit(ax, m, c[0], c[1], 32, t, s, rays_lw=0.6)
fig.savefig(OUT + "fig-shapes.png", dpi=100, bbox_inches="tight", facecolor="white"); plt.close(fig)

# 5. why touching nuclei separate: polygons vs boxes on two touching diagonal nuclei ------------
n = 64
yy, xx = np.mgrid[0:n, 0:n]
lab = np.zeros((n, n), int)
off = 6 * np.sqrt(2)   # centres 2b apart across the long axes: the two nuclei touch along their sides
cen = [(32 - off / 2, 32 - off / 2), (32 + off / 2, 32 + off / 2)]
for i, (ex, ey) in enumerate(cen):
    u = ((xx + 0.5 - ex) + (yy + 0.5 - ey)) / np.sqrt(2); v = (-(xx + 0.5 - ex) + (yy + 0.5 - ey)) / np.sqrt(2)
    d = (v / 22) ** 2 + (u / 6) ** 2
    lab[(d <= 1) & (lab == 0)] = i + 1
fig, axs = plt.subplots(1, 2, figsize=(9, 4.4))
pms, boxes = [], []
for i, (ex, ey) in enumerate(cen):
    r = rays_at(lab, ex + 0.5, ey + 0.5, 32, i + 1)
    pms.append(poly_mask(ex + 0.5, ey + 0.5, r, lab.shape))
    T = tips(ex + 0.5, ey + 0.5, r)
    boxes.append((T[:, 0].min(), T[:, 1].min(), T[:, 0].max(), T[:, 1].max()))
    axs[0].add_patch(MPoly(T, closed=True, fill=False, ec=[RED, BLUE][i], lw=2.4))
    axs[1].add_patch(Rectangle(boxes[-1][:2], boxes[-1][2] - boxes[-1][0], boxes[-1][3] - boxes[-1][1], fill=False, ec=[RED, BLUE][i], lw=2.4))
piou = (pms[0] & pms[1]).sum() / (pms[0] | pms[1]).sum()
b0, b1 = boxes
iw = max(0, min(b0[2], b1[2]) - max(b0[0], b1[0])); ih = max(0, min(b0[3], b1[3]) - max(b0[1], b1[1]))
area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
biou = iw * ih / (area(b0) + area(b1) - iw * ih)
for ax, t in zip(axs, [f"polygons: IoU {piou:.2f}", f"boxes: IoU {biou:.2f}"]):
    ax.imshow(np.where(lab > 0, lab, np.nan), cmap=matplotlib.colors.ListedColormap(["#f4c7c3", "#c6dbef"]), extent=(0, n, n, 0))
    ax.set_xlim(4, 60); ax.set_ylim(60, 4); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(t, fontsize=17)
fig.savefig(OUT + "fig-touch.svg", bbox_inches="tight", facecolor="white"); plt.close(fig)
print(f"touching: polygon IoU {piou:.3f}, box IoU {biou:.3f}")


# ---- real network: crops of the stardistnet demo ---------------------------------------------
def shot(name, box=None):
    im = Image.open(f"{SHOTS}/{name}.png").convert("RGB")
    if box:   # box in image pixels of the 512 x 512 sample; the stage is square
        k = im.width / 512
        im = im.crop(tuple(int(round(v * k)) for v in box))
    return np.asarray(im)


def panels(cells, out, size, sub_size=15, ncol=None):
    ncol = ncol or len(cells)
    nrow = (len(cells) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=size, squeeze=False, gridspec_kw={"wspace": 0.04, "hspace": 0.22})
    for a in axs.flat:
        a.axis("off")
    for a, (img, title, sub) in zip(axs.flat, cells):
        a.imshow(img)
        a.set_title(title, fontsize=17)
        if sub:
            a.text(0.5, -0.02, sub, transform=a.transAxes, ha="center", va="top", fontsize=sub_size)
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


meas = json.load(open("tools/figs/stardist/measured.json"))
mi = {im["name"]: im for im in meas["images"]}
tr = mi["nuclei@1"]["presets"]["trained"]
CROP = (0, 0, 232, 232)   # top left; the demo marks a selected nucleus near the centre
panels([(shot("sd_image", CROP), "input", ""),
        (shot("sd_prob", CROP), "objectness", "highest in the middle"),
        (shot("sd_ray", CROP), "length of the ray to the right", "longest at the left edge")],
       OUT + "fig-prob.png", (12, 4.4))
panels([(shot("sd_image", CROP), "input", ""),
        (shot("sd_cand", CROP), f"{tr['candidates']} candidates", "all cells above the threshold"),
        (shot("sd_nms", CROP), f"after NMS: {tr['count']} nuclei", "one polygon per nucleus"),
        (shot("sd_labels", CROP), "label image", "polygons painted, best last")],
       OUT + "fig-decode.png", (8.4, 9.0), ncol=2)


# 6. the two thresholds: crops of the starrays decode mode (simulated network) -----------------
def right_half(name):
    im = Image.open(f"{SHOTS}/{name}.png").convert("RGB")
    w, h = im.size
    return np.asarray(im.crop((int(w * 0.505), int(h * 0.055), w, h)))


panels([(right_half("sr_low"), "probability too low (0.3)", "debris specks become nuclei (FP)"),
        (right_half("sr_high"), "probability too high (0.7)", "the dim nucleus is lost (FN)"),
        (right_half("sr_strong"), "NMS overlap too low (0.02)", "touching neighbours deleted (FN)"),
        (right_half("sr_weak"), "NMS overlap too high (0.8)", "duplicates survive (FP)")],
       OUT + "fig-thresholds.png", (9.6, 10.2), ncol=2)


# 7. failures by construction: simulated C (instancemaps) and a real split (U2OS) ---------------
def grey_of(name, w, h):
    return np.fromfile(f"{MEAS}/{name}.f32", "<f4").reshape(h, w)


def labels_of(name, w, h, ext):
    return np.fromfile(f"{MEAS}/{name}.{ext}", "<i4").reshape(h, w)


def outline_rgb(grey, pred, truth=None, lo=None, hi=None):
    lo = np.percentile(grey, 1) if lo is None else lo
    hi = np.percentile(grey, 99.8) if hi is None else hi
    g = np.clip((grey - lo) / (hi - lo), 0, 1)
    rgb = np.dstack([g, g, g])
    if truth is not None:
        rgb[find_boundaries(truth, mode="inner")] = [0.3, 1.0, 0.3]
    rgb[find_boundaries(pred, mode="inner")] = [1.0, 0.55, 0.0]
    return rgb


IMK = {"map": (295, 50, 572, 327), "objects": (580, 50, 857, 327)}
im = Image.open(f"{SHOTS}/im_cshape.png").convert("RGB")
cmap_ = np.asarray(im.crop(tuple(2 * v for v in IMK["map"])))
cobj = np.asarray(im.crop(tuple(2 * v for v in IMK["objects"])))


def fill_rgb(grey, pred, truth):
    """Predicted objects as colour fills, true outlines in green."""
    lo, hi = np.percentile(grey, 1), np.percentile(grey, 99.8)
    g = np.clip((grey - lo) / (hi - lo), 0, 1)
    rgb = np.dstack([g, g, g])
    pal = np.array([matplotlib.colors.to_rgb(c) for c in ["#1f77b4", "#ff7f0e", "#9467bd", "#e377c2", "#17becf", "#d62728", "#bcbd22", "#8c564b"]])
    cols = pal[pred % len(pal)]
    rgb = np.where(pred[..., None] > 0, 0.5 * rgb + 0.5 * cols, rgb)
    rgb[find_boundaries(truth, mode="inner")] = [0.2, 1.0, 0.2]
    return rgb


def real_crop(name, box, pad=10):
    e = mi[name]; w, h = e["w"], e["h"]
    rgb = fill_rgb(grey_of(name, w, h), labels_of(name, w, h, "pred"), labels_of(name, w, h, "lab"))
    x0, y0, x1, y1 = box
    return rgb[max(0, y0 - pad):min(h, y1 + pad + 1), max(0, x0 - pad):min(w, x1 + pad + 1)]


# a peanut-shaped U2OS nucleus split in two, two elongated touching nuclei merged (measured.json)
split_box = mi["u2os@1"]["splitBoxes"][9]
merge_box = mi["dense@1"]["mergeBoxes"][2]
panels([(cmap_, "simulated C", "the rays leave the C"),
        (cobj, "simulated: objects", "the C is split (S)"),
        (real_crop("u2os@1", split_box), "real: one nucleus", "split in two"),
        (real_crop("dense@1", merge_box), "real: two nuclei", "merged into one")],
       OUT + "fig-failures.png", (8.4, 8.8), ncol=2)

# 8. failures of the training data: histology (H&E) and bright field --------------------------
h_ = mi["histology@1"]; b_ = mi["brightfield@1"]
hg = grey_of("histology@1", h_["w"], h_["h"]); hp = labels_of("histology@1", h_["w"], h_["h"], "pred"); ht = labels_of("histology@1", h_["w"], h_["h"], "lab")
bg = grey_of("brightfield@1", b_["w"], b_["h"]); bp = labels_of("brightfield@1", b_["w"], b_["h"], "pred"); bt = labels_of("brightfield@1", b_["w"], b_["h"], "lab")
hrgb = np.asarray(Image.open("demos/dist/models/images/dsb-histology.png").convert("RGB")) / 255.0
hcol = hrgb.copy(); hcol[find_boundaries(hp, mode="inner")] = [1.0, 0.55, 0.0]
ht_ = hrgb.copy(); ht_[find_boundaries(ht, mode="inner")] = [0.1, 0.8, 0.1]
hs, bs = h_["presets"]["trained"], b_["presets"]["trained"]
bt_rgb = outline_rgb(bg, np.zeros_like(bp), bt)[200:712, 200:712]
bp_rgb = outline_rgb(bg, bp)[200:712, 200:712]
panels([(ht_, "H&E: true nuclei (green)", f"{h_['nTrue']} nuclei, dark on bright"),
        (hcol, "H&E: the network (orange)", f"{hs['count']} objects, {hs['tp']} correct (F1 {hs['f1']:.2f})"),
        (bt_rgb, "bright field: true nuclei", f"{b_['nTrue']} small nuclei (green)"),
        (bp_rgb, "bright field: the network", f"{bs['count']} objects, {bs['tp']} correct (F1 {bs['f1']:.2f})")],
       OUT + "fig-domain.png", (8.4, 8.6), ncol=2)
print("done")
