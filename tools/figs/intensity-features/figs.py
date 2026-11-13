# Figures for intensity-features/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/intensity-features/figs.py
# Cells, masks, rings and plates come from the intensityfeat demo's code (data.ts, via node),
# so every number matches the demo and the slides.
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "label-connectivity"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from labstyle import setup, fig, save, save_png, run_ts, INK, GREY, LABELS

setup()
OUT = "intensity-features"
D = run_ts(HERE)
N = D["N"]
OI_BLUE, OI_ORANGE, OI_GREEN, OI_YELLOW, OI_DBLUE, OI_VERM, OI_PINK = "#0072B2", "#E69F00", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
OI_SKY = "#56B4E9"
NAMES = {"uniform": "uniform", "rim": "bright rim", "spot": "off-centre spot", "nuclear": "nuclear", "gradient": "gradient", "punctate": "punctate"}


def cell(p):
    c = D["cells"][p]
    return np.array(c["img"]).reshape(N, N), np.array(c["mask"]).reshape(N, N).astype(bool), c


def show(ax, img, vmax=None, crop=(10, 150)):
    ax.imshow(img, cmap="gray", vmin=0, vmax=vmax or np.percentile(img, 99.7), interpolation="nearest")
    ax.set_xlim(crop[0] - 0.5, crop[1] - 0.5); ax.set_ylim(crop[1] - 0.5, crop[0] - 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)


def outline(ax, mask, col, lw=2.0, ls="-"):
    ax.contour(mask.astype(float), levels=[0.5], colors=[col], linewidths=lw, linestyles=ls)


# --- 1. histograms of the values inside the mask --------------------------------------------
f, axs = plt.subplots(2, 2, figsize=(470 / 72, 400 / 72), sharex=True)
for ax, p in zip(axs.flat, ["uniform", "rim", "spot", "nuclear"]):
    img, m, c = cell(p)
    v = np.array(c["values"]); r = c["row"]
    ax.hist(v, bins=np.arange(0, 500, 10), color=OI_SKY, edgecolor="white", linewidth=0.4)
    ax.axvline(r["mean"], color=OI_VERM, lw=3)
    ax.axvline(r["median"], color=INK, lw=2.5, ls="--")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.text(0.97, 0.92, NAMES[p], transform=ax.transAxes, ha="right", va="top", fontsize=18)
    ax.text(0.97, 0.70, f"std {r['sd']:.1f}", transform=ax.transAxes, ha="right", va="top", fontsize=16, color=GREY)
    # thumbnail
    ins = ax.inset_axes([0.62, 0.05, 0.36, 0.42] if p != "uniform" else [0.30, 0.25, 0.32, 0.42])
    show(ins, img, vmax=np.percentile(img, 99.5))
f.supxlabel("pixel value inside the mask", fontsize=18, y=0.02)
axs[0, 0].plot([], [], color=OI_VERM, lw=3, label="mean (100)")
axs[0, 0].plot([], [], color=INK, lw=2.5, ls="--", label="median")
f.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
f.tight_layout()
save(f, f"{OUT}/fig-hist.svg")

# --- 2. which numbers survive: offset, mask too large, clipping -----------------------------
S = D["survive"]
f, axs = plt.subplots(1, 3, figsize=(1000 / 72, 230 / 72))
panels = [
    ("ideal", "rIdeal", "uniform cell", np.arange(0, 131, 5)),
    ("offset", "rOffset", "offset +10", np.arange(0, 131, 5)),
    ("dilated", "rDilated", "offset +10, mask +2 px", np.arange(0, 131, 5)),
]
for ax, (k, rk, lab, bins) in zip(axs, panels):
    v = np.array(S[k]); r = S[rk]
    ax.hist(v, bins=bins, color=OI_SKY, edgecolor="white", linewidth=0.3)
    ax.axvline(r["mean"], color=OI_VERM, lw=3)
    ax.set_yscale("log"); ax.set_yticks([]); ax.minorticks_off(); ax.spines["left"].set_visible(False)
    ax.set_title(f"{lab}: mean {r['mean']:.1f}, std {r['sd']:.1f}", fontsize=17, loc="left")
    ax.set_xlabel("pixel value")
f.tight_layout(w_pad=1.5)
save(f, f"{OUT}/fig-survive.svg")

# --- 3. local background ring ------------------------------------------------------------------
B = D["bg"]
img = np.array(B["img"]).reshape(N, N); m = np.array(B["mask"]).reshape(N, N).astype(bool)
ring = np.array(B["ring"]).reshape(N, N).astype(bool); nb = np.array(B["nb"]).reshape(N, N).astype(bool)
f, ax = fig(430, 430)
ax.imshow(img, cmap="gray", vmin=0, vmax=60, interpolation="nearest")
rgba = np.zeros((N, N, 4)); rgba[ring] = [*plt.matplotlib.colors.to_rgb(OI_ORANGE), 0.75]
ax.imshow(rgba, interpolation="nearest")
outline(ax, m, OI_SKY, 2.5)
outline(ax, nb, OI_PINK, 2.5, "--")
ax.set_xlim(-0.5, N - 0.5); ax.set_ylim(N - 0.5, -0.5); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax.text(80, 82, "object", color=INK, ha="center", va="center", fontsize=20,
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
ax.text(150, 6, "other\nobject", color=INK, ha="right", va="top", fontsize=17, bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
ax.text(4, 155, f"ring: gap 3 px, band 6 px\nmedian {B['bg']:.1f}", color=INK, ha="left", va="bottom", fontsize=17,
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
save_png(f, f"{OUT}/fig-bgring.png")

# --- 4. edge band and profiles ----------------------------------------------------------------
f, (a0, a1) = plt.subplots(1, 2, figsize=(1000 / 72, 330 / 72), gridspec_kw=dict(width_ratios=[1, 2.1]))
img, m, c = cell("rim")
band = np.array(D["band"]).reshape(N, N).astype(bool)
show(a0, img, vmax=330, crop=(15, 145))
rgba = np.zeros((N, N, 4)); rgba[band] = [*plt.matplotlib.colors.to_rgb(OI_ORANGE), 0.9]
a0.imshow(rgba, interpolation="nearest")
a0.text(80, 148, "edge band (orange)", ha="center", va="bottom", fontsize=16, color=INK, clip_on=False,
        bbox=dict(fc="white", ec="none", pad=1))
# profile along the major axis through the centre (row through the centroid, angle 20 degrees)
t = np.radians(20); s = np.arange(-62, 62.01, 0.25)
xs, ys = 80 + s * np.cos(t), 80 + s * np.sin(t)
a0.plot(xs, ys, color=OI_SKY, lw=1.5, ls=":")
from scipy.ndimage import map_coordinates
for p, col in [("uniform", OI_BLUE), ("rim", OI_VERM)]:
    im, mm, cc = cell(p)
    prof = map_coordinates(im, [ys, xs], order=1)
    a1.plot(s, prof, color=col, lw=2.5, label=f"{NAMES[p]}: edge / mean {cc['row']['edgeRatio']:.2f}")
a1.axvspan(-52, -51, color=OI_ORANGE, alpha=0.5, lw=0); a1.axvspan(51, 52, color=OI_ORANGE, alpha=0.5, lw=0)
a1.axhline(100, color=GREY, lw=1.2, ls="--"); a1.text(-40, 108, "mean 100", color=GREY, fontsize=16)
a1.set_xlabel("position along the long axis (px)"); a1.set_ylabel("value")
a1.set_xlim(-62, 62); a1.set_ylim(0, 340)
a1.legend(loc="upper center", frameon=False)
f.tight_layout()
save(f, f"{OUT}/fig-edge.svg")

# --- 5. mass displacement ---------------------------------------------------------------------
f, axs = plt.subplots(1, 3, figsize=(900 / 72, 330 / 72))
for ax, p in zip(axs, ["uniform", "spot", "gradient"]):
    img, m, c = cell(p); r = c["row"]
    show(ax, img, crop=(20, 140))
    outline(ax, m, "#777777", 1.2)
    ax.plot(r["cx"], r["cy"], "+", color=OI_SKY, ms=22, mew=4)
    ax.plot(r["wx"], r["wy"], "o", color=OI_ORANGE, ms=11, mec="black", mew=1.2)
    if r["massDisplacement"] > 1:
        ax.annotate("", xy=(r["wx"], r["wy"]), xytext=(r["cx"], r["cy"]),
                    arrowprops=dict(arrowstyle="-|>", color=OI_ORANGE, lw=3, mutation_scale=20, shrinkA=0, shrinkB=6))
    ax.set_title(f"{NAMES[p]}: {r['massDisplacement']:.1f} px", fontsize=18)
axs[0].plot([], [], "+", color=OI_SKY, ms=16, mew=3, ls="none", label="geometric centroid")
axs[0].plot([], [], "o", color=OI_ORANGE, ms=10, mec="black", ls="none", label="intensity-weighted centroid")
f.legend(loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False)
f.tight_layout()
save_png(f, f"{OUT}/fig-massdisp.png")

# --- 6. radial rings --------------------------------------------------------------------------
f, axs = plt.subplots(2, 1, figsize=(340 / 72, 560 / 72), gridspec_kw=dict(height_ratios=[1, 1.1]))
img, m, c = cell("nuclear")
rings = np.array(c["ring"]).reshape(N, N)
show(axs[0], img, vmax=340, crop=(20, 140))
for k in range(1, 4):
    outline(axs[0], rings >= k, OI_YELLOW, 2.2)
outline(axs[0], m, OI_YELLOW, 2.2)
# ring numbers along the long axis (towards the lower right, angle 20 degrees)
t = np.radians(20)
for k, rho in enumerate([0.0, 0.375, 0.625, 0.875]):
    if k == 0:
        continue
    x_, y_ = 80 + rho * 52 * np.cos(t), 80 + rho * 52 * np.sin(t)
    axs[0].text(x_, y_, str(k + 1), color=OI_YELLOW, fontsize=17, ha="center", va="center", fontweight="bold")
axs[0].text(80, 80, "1", color=INK, fontsize=17, ha="center", va="center", fontweight="bold")
axs[0].set_title("nuclear cell, 4 rings", fontsize=17)
ax = axs[1]
x = np.arange(4)
for j, (p, col) in enumerate([("nuclear", OI_BLUE), ("rim", OI_VERM), ("uniform", "#999999")]):
    mf = D["cells"][p]["meanFrac"]
    ax.bar(x + (j - 1) * 0.27, mf, width=0.26, color=col, label=NAMES[p])
ax.axhline(1, color=INK, lw=1.2, ls="--")
ax.set_xticks(x, ["1\ncentre", "2", "3", "4\nedge"])
ax.set_ylabel("MeanFrac"); ax.set_xlabel("ring")
ax.legend(frameon=False, loc="upper right", fontsize=16, handlelength=1.2, borderaxespad=0.1)
ax.set_ylim(0, 3.5)
f.tight_layout()
save_png(f, f"{OUT}/fig-rings.png")

# --- 7. two plates: raw and robust z ----------------------------------------------------------
P = D["plate"]
ctrl = np.array(P["control"]).reshape(4, 6).astype(bool); trt = np.array(P["treated"]).reshape(4, 6).astype(bool)
f, axs = plt.subplots(1, 4, figsize=(1150 / 72, 230 / 72))
raw_lo, raw_hi = 85, 220
for ax, key, lab in [(axs[0], "A", "plate A, raw"), (axs[1], "B", "plate B, raw"), (axs[2], "zA", "plate A, robust z"), (axs[3], "zB", "plate B, robust z")]:
    v = np.array(P[key]).reshape(4, 6)
    if key.startswith("z"):
        im = ax.imshow(v, cmap="PuOr_r", norm=TwoSlopeNorm(0, -13, 13))
    else:
        im = ax.imshow(v, cmap="viridis", vmin=raw_lo, vmax=raw_hi)
    for i in range(4):
        for j in range(6):
            val = v[i, j]
            txt = f"{val:.0f}" if not key.startswith("z") else f"{val:.1f}"
            dark = (not key.startswith("z") and val < 170) or (key.startswith("z") and abs(val) > 7)
            ax.text(j, i, txt, ha="center", va="center", fontsize=15, color="white" if dark else INK,
                    fontweight="bold" if trt[i, j] else "normal")
    for j in (0, 5):
        ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, 4, fill=False, ec=INK, lw=2.2, ls="--"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(lab, fontsize=17, loc="left")
f.text(0.5, -0.06, "dashed: control columns; bold: treated wells", ha="center", fontsize=16)
f.tight_layout()
save(f, f"{OUT}/fig-plates.svg")
print("done")
