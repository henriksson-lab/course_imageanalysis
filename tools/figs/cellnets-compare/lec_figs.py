"""Slide figures of the cellnets-compare/ lecture (Networks for cells and objects day).

Run from the repository root, after `make wasm`, `make models` and the measurement script
(measure.ts, see its header), which writes tools/figs/cellnets-compare/results.json and the
label images .cache/cellnets-compare/<image>[-inv]-<method>.bin:
  python3 -I tools/figs/cellnets-compare/lec_figs.py

Real images: demos/dist/models/images/ (public annotated sets; credits in the captions).
The worked F1(tau) figure is synthetic (the numbers of exercise B1). The bootstrap figure
resamples the seven per-image F1 values of results.json (seed 1).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi

OUT = os.environ.get("CC_FIGS", "cellnets-compare").rstrip("/") + "/"
IMG = "demos/dist/models/images/"
CACHE = os.environ.get("CC_OUT", ".cache/cellnets-compare").rstrip("/") + "/"
R = json.load(open(os.environ.get("CC_RESULTS", "tools/figs/cellnets-compare/results.json")))
TAUS = np.array(R["taus"])
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans"})

METHODS = [("classical", "classical", "#e69f00"), ("stardist", "StarDist", "#56b4e9"),
           ("cellpose", "Cellpose", "#009e73"), ("sam", "SAM automatic", "#cc79a7")]
SAMPLES = [("sparse", "dsb-fluo-small", "sparse"), ("dim", "dsb-fluo-medium", "dim"),
           ("dense", "dsb-fluo-dense", "dense"), ("u2os", "bbbc039-A02_s1", "U2OS 1"),
           ("u2os2", "bbbc039-A06_s6", "U2OS 2"), ("histology", "dsb-histology", "histology"),
           ("brightfield", "dsb-brightfield", "bright field")]
FILE = {s: f for s, f, _ in SAMPLES}
SHORT = {s: t for s, _, t in SAMPLES}
MARK = {"miss": "#ff3030", "extra": "#ffd700", "merge": "#ff40ff", "split": "#40e0ff"}


def grey(sid, inv=False):
    a = np.asarray(Image.open(IMG + FILE[sid] + ".png").convert("RGB")).astype(float).mean(axis=2)
    if inv:
        a = 255 - a
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.8)
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def labels(key, m):
    run = R["runs"][key]
    return np.fromfile(f"{CACHE}{key}-{m}.bin", dtype=np.int32).reshape(run["h"], run["w"])


def outline(lab):
    e = np.zeros(lab.shape, bool)
    e[:-1] |= lab[:-1] != lab[1:]
    e[1:] |= lab[1:] != lab[:-1]
    e[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    e[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    return e & (lab > 0)


def overlay(g, lab, colour, truth=None):
    rgb = np.repeat((0.85 * g)[..., None], 3, axis=2)
    if truth is not None:
        rgb[outline(truth)] = 1
    c = np.array(matplotlib.colors.to_rgb(colour))
    o = ndi.binary_dilation(outline(lab)) if max(lab.shape) > 500 else outline(lab)
    rgb[o] = c
    return rgb


def match(truth, pred, tau=0.5):
    """IoU table and a greedy one-to-one matching (IoU > 0.5 pairs are unique, so this is exact at 0.5)."""
    nt, np_ = truth.max(), pred.max()
    idx = (truth.ravel() > 0) & (pred.ravel() > 0)
    inter = np.zeros((nt + 1, np_ + 1))
    np.add.at(inter, (truth.ravel()[idx], pred.ravel()[idx]), 1)
    at = np.bincount(truth.ravel(), minlength=nt + 1).astype(float)
    ap = np.bincount(pred.ravel(), minlength=np_ + 1).astype(float)
    iou = inter / (at[:, None] + ap[None, :] - inter + 1e-12)
    iou[0, :] = iou[:, 0] = 0
    return inter, at, ap, iou


def marks(truth, pred):
    inter, at, ap, iou = match(truth, pred)
    mt = (iou >= 0.5).any(axis=1)
    mp = (iou >= 0.5).any(axis=0)
    out = []
    cont_t = (inter >= 0.5 * at[:, None]) & (iou >= 0.1)   # true object mostly inside prediction
    cont_p = (inter >= 0.5 * ap[None, :]) & (iou >= 0.1)
    merged_t, split_p = set(), set()
    ct = ndi.center_of_mass(np.ones_like(truth), truth, range(truth.max() + 1))
    cp = ndi.center_of_mass(np.ones_like(pred), pred, range(pred.max() + 1))
    for j in range(1, pred.max() + 1):
        parts = np.nonzero(cont_t[1:, j])[0] + 1
        if len(parts) >= 2:
            out.append(("merge", cp[j])); merged_t.update(parts)
    for i in range(1, truth.max() + 1):
        parts = np.nonzero(cont_p[i, 1:])[0] + 1
        if len(parts) >= 2:
            out.append(("split", ct[i])); split_p.update(parts)
    for i in range(1, truth.max() + 1):
        if not mt[i] and i not in merged_t and at[i] > 0:
            out.append(("miss", ct[i]))
    for j in range(1, pred.max() + 1):
        if not mp[j] and j not in split_p and ap[j] > 0:
            out.append(("extra", cp[j]))
    return out


def draw_marks(ax, mk, x0=0, y0=0, size=12):
    for kind, (y, x) in mk:
        x, y = x - x0, y - y0
        if kind == "miss":
            ax.plot(x, y, "o", mfc="none", mec=MARK[kind], ms=size, mew=2.2)
        elif kind == "extra":
            ax.plot(x, y, "x", color=MARK[kind], ms=size * 0.8, mew=2.5)
        else:
            ax.text(x, y, "M" if kind == "merge" else "S", color=MARK[kind], fontsize=size + 3, fontweight="bold", ha="center", va="center")


def f1(key, m, k=0):
    return R["runs"][key]["methods"][m]["f1"][k]


# 1. the seven images with their object counts ----------------------------------------------------
fig, axs = plt.subplots(1, 7, figsize=(15, 3.0))
for a, (sid, f, t) in zip(axs, SAMPLES):
    a.imshow(np.asarray(Image.open(IMG + f + ".png").convert("RGB")) if sid == "histology" else grey(sid), cmap="gray")
    a.set_title(t, fontsize=17)
    a.text(0.5, -0.04, f"{R['runs'][sid]['nTrue']} nuclei", transform=a.transAxes, ha="center", va="top", fontsize=16)
    a.axis("off")
fig.savefig(OUT + "fig-images.png", dpi=100, bbox_inches="tight", facecolor="white")
plt.close(fig)

# 2. the contestants on one image (U2OS 1, a crop), same colours everywhere -------------------------
key, (y0, y1, x0, x1) = "u2os", (150, 390, 230, 530)
g = grey("u2os")
fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.4), gridspec_kw={"wspace": 0.04, "hspace": 0.16})
for a, (m, name, col) in zip(axs.ravel(), METHODS):
    p = labels(key, m)
    rgb = np.repeat((0.85 * g)[..., None], 3, axis=2)
    rgb[ndi.binary_dilation(outline(p))] = matplotlib.colors.to_rgb(col)
    a.imshow(rgb[y0:y1, x0:x1])
    a.set_title(f"{name}: F1 {f1(key, m):.2f}", fontsize=17)
    a.axis("off")
fig.savefig(OUT + "fig-contestants.png", dpi=100, bbox_inches="tight", facecolor="white")
plt.close(fig)

# 3. worked F1(tau) curves (exercise B1): 20 nuclei; A: 21 objects, IoUs of its 18 matches;
#    B: 17 objects, 16 matches. IoUs chosen so A has 18 at 0.5 and 12 at 0.7, B 16 and 15.
iouA = np.r_[np.linspace(0.52, 0.68, 6), np.linspace(0.71, 0.86, 12)]
iouB = np.r_[0.62, np.linspace(0.74, 0.93, 15)]
fA = [2 * (iouA >= t).sum() / (21 + 20) for t in TAUS]
fB = [2 * (iouB >= t).sum() / (17 + 20) for t in TAUS]
fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.plot(TAUS, fA, "-o", color="#d55e00", lw=3, label="A: finds all, sloppy outlines")
ax.plot(TAUS, fB, "-s", color="#0072b2", lw=3, label="B: misses some, careful outlines")
for t, a, b in [(0.5, fA[0], fB[0]), (0.7, fA[4], fB[4])]:
    ax.annotate(f"{a:.2f}", (t, a), textcoords="offset points", xytext=(4, 8) if t == 0.5 else (-6, -20), color="#d55e00", fontsize=15)
    ax.annotate(f"{b:.2f}", (t, b), textcoords="offset points", xytext=(4, -22) if t == 0.5 else (2, 8), color="#0072b2", fontsize=15)
ax.set_xlabel("IoU threshold τ")
ax.set_ylabel("F1")
ax.set_ylim(0, 1)
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax.grid(alpha=0.3)
ax.legend(fontsize=13, loc="lower left")
fig.savefig(OUT + "fig-f1worked.svg", bbox_inches="tight")
plt.close(fig)

# 4. measured F1 at 0.5 per image and method (dot plot) -------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ids = [s for s, _, _ in SAMPLES]
for k, (m, name, col) in enumerate(METHODS):
    ys = [f1(s, m) for s in ids]
    ax.plot(np.arange(7) + (k - 1.5) * 0.12, ys, "o", ms=11, color=col, label=name, mec="black", mew=0.6)
ax.set_xticks(range(7))
ax.set_xticklabels([SHORT[s] for s in ids], rotation=25, ha="right")
ax.set_ylabel("F1 at τ = 0.5")
ax.set_ylim(-0.03, 1.03)
ax.axvspan(4.5, 6.5, color="#eeeeee", zorder=-1)
ax.text(5.5, 0.5, "not\nfluorescence", ha="center", va="center", fontsize=13, color="#555")
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=13, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, handletextpad=0.2, columnspacing=0.8)
fig.savefig(OUT + "fig-results.svg", bbox_inches="tight")
plt.close(fig)

# 5. measured F1(tau) on the dense image ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.4))
for m, name, col in METHODS:
    ax.plot(TAUS, R["runs"]["dense"]["methods"][m]["f1"], "-o", color=col, lw=3, label=name)
ax.set_xlabel("IoU threshold τ")
ax.set_ylabel("F1")
ax.set_ylim(0, 1)
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax.grid(alpha=0.3)
ax.legend(fontsize=13, loc="upper right")
fig.savefig(OUT + "fig-f1dense.svg", bbox_inches="tight")
plt.close(fig)

# 6. where the errors are: dense image crop, classical vs StarDist vs Cellpose, with marks ---------
key, (y0, y1, x0, x1) = "dense", (150, 400, 250, 560)
g = grey("dense")
tr = labels(key, "truth")
fig, axs = plt.subplots(1, 3, figsize=(14, 3.9))
for a, (m, name, col) in zip(axs, METHODS[:3]):
    p = labels(key, m)
    a.imshow(overlay(g, p, col)[y0:y1, x0:x1])
    mk = [(k, (y, x)) for k, (y, x) in marks(tr, p) if y0 <= y < y1 and x0 <= x < x1]
    draw_marks(a, mk, x0, y0)
    cnt = {k: sum(1 for q, _ in marks(tr, p) if q == k) for k in MARK}
    a.set_title(f"{name}: F1 {f1(key, m):.2f}", fontsize=17)
    a.text(0.5, -0.03, f"whole image: {cnt['merge']} merges, {cnt['split']} splits,\n{cnt['miss']} missed, {cnt['extra']} extra",
           transform=a.transAxes, ha="center", va="top", fontsize=14)
    a.set_xlim(0, x1 - x0); a.set_ylim(y1 - y0, 0); a.axis("off")
fig.savefig(OUT + "fig-errors.png", dpi=100, bbox_inches="tight", facecolor="white")
plt.close(fig)

# 7. out of domain: histology and bright field, as is and with inverted grey levels -------------
COL = {m: c for m, _, c in METHODS}
NAME = {m: n for m, n, _ in METHODS}
fig, axs = plt.subplots(1, 4, figsize=(15, 4.1))
cells = [("histology", False, "stardist"), ("histology", True, "stardist"),
         ("brightfield", True, "stardist"), ("brightfield", True, "cellpose")]
for a, (sid, inv, m) in zip(axs, cells):
    key = sid + ("-inv" if inv else "")
    g = grey(sid, inv)
    p = labels(key, m)
    t = labels(key, "truth")
    rgb = np.repeat((0.85 * g)[..., None], 3, axis=2)
    rgb[ndi.binary_dilation(outline(t))] = 1
    rgb[outline(p) if sid == "histology" else ndi.binary_dilation(outline(p))] = matplotlib.colors.to_rgb(COL[m])
    sl = (slice(330, 650), slice(200, 600)) if sid == "brightfield" else (slice(None), slice(None))
    a.imshow(rgb[sl])
    a.set_title(f"{SHORT[sid]}{', inverted' if inv else ''}", fontsize=17)
    a.text(0.5, -0.03, f"{NAME[m]}: F1 {f1(key, m):.2f}", transform=a.transAxes, ha="center", va="top", fontsize=16, color="black")
    a.axis("off")
fig.savefig(OUT + "fig-domain.png", dpi=100, bbox_inches="tight", facecolor="white")
plt.close(fig)

# 8. bootstrap of the mean difference in F1 over the seven images (StarDist - classical) -----------
d = np.array([f1(s, "stardist") - f1(s, "classical") for s in ids])
rng = np.random.default_rng(1)
boot = np.array([d[rng.integers(0, 7, 7)].mean() for _ in range(10000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.hist(boot, bins=50, color="#bbbbbb", edgecolor="white")
ax.axvline(d.mean(), color="black", lw=2.5)
ax.axvspan(lo, hi, color="#56b4e9", alpha=0.25)
ax.axvline(0, color="#d62728", lw=2, ls="--")
ax.set_xlabel("mean F1 difference, StarDist − classical")
ax.set_ylabel("bootstrap samples")
ax.set_title(f"mean {d.mean():+.3f}, 95 % interval [{lo:+.3f}, {hi:+.3f}]", fontsize=15)
fig.savefig(OUT + "fig-bootstrap.svg", bbox_inches="tight")
plt.close(fig)

# 9. cost: images per hour on one laptop core (512 x 512; network times from demos/wasm READMEs,
#    classical measured here on the 512 x 512 StarDist sample size scaled from results.json) ----
cl_ms = np.mean([R["runs"][s]["methods"]["classical"]["ms"] / (R["runs"][s]["w"] * R["runs"][s]["h"]) for s in ids]) * 512 * 512
cost = [("classical", cl_ms / 1000, "5 knobs", "#e69f00"), ("StarDist", 2.13, "1.4 M parameters", "#56b4e9"),
        ("Cellpose", 7.7, "6.6 M", "#009e73"), ("SAM auto 8 × 8", 48.3, "39 M", "#cc79a7")]
fig, ax = plt.subplots(figsize=(6.0, 3.4))
y = np.arange(len(cost))[::-1]
ax.barh(y, [3600 / c[1] for c in cost], color=[c[3] for c in cost])
for yy, c in zip(y, cost):
    ax.text(3600 / c[1] * 1.15, yy, f"{3600 / c[1]:,.0f} / h", va="center", fontsize=15)
ax.set_xscale("log")
ax.set_xlim(30, 3e5)
ax.set_yticks(y)
ax.set_yticklabels([c[0] for c in cost])
ax.set_xlabel("images (512 × 512) per hour, one core")
ax.grid(axis="x", alpha=0.3)
fig.savefig(OUT + "fig-cost.svg", bbox_inches="tight")
plt.close(fig)
print("classical, ms per 512 x 512 (scaled):", round(cl_ms))
print("bootstrap", d.round(3), d.mean().round(4), lo.round(4), hi.round(4))

# 10. the five outcomes of matching, sketched (white: truth, blue: prediction) ---------------------
from matplotlib.patches import Ellipse
fig, axs = plt.subplots(2, 3, figsize=(6.6, 3.9))
axs = axs.ravel()
axs[5].axis("off")
P = "#56b4e9"
def ell(ax, x, y, w, h, col, lw=3, ls="-"):
    ax.add_patch(Ellipse((x, y), w, h, fill=False, ec=col, lw=lw, ls=ls))
cases = [
    ("match", [(0, 0, 2, 1.4)], [(0.1, 0.05, 2, 1.4)]),
    ("miss", [(0, 0, 2, 1.4)], []),
    ("extra", [], [(0, 0, 1.6, 1.2)]),
    ("merge", [(-0.7, 0, 1.3, 1.1), (0.7, 0, 1.3, 1.1)], [(0, 0, 2.8, 1.3)]),
    ("split", [(0, 0, 2.4, 1.5)], [(-0.6, 0, 1.1, 1.3), (0.6, 0, 1.1, 1.3)]),
]
for a, (name, ts, ps) in zip(axs, cases):
    a.set_facecolor("#222")
    for e in ts:
        ell(a, *e, "white", 3.5)
    for e in ps:
        ell(a, *e, P, 3, "--")
    a.set_xlim(-1.6, 1.6); a.set_ylim(-1.1, 1.1); a.set_aspect("equal")
    a.set_xticks([]); a.set_yticks([])
    a.set_title(name, fontsize=17)
fig.savefig(OUT + "fig-errors-key.svg", bbox_inches="tight")
plt.close(fig)
