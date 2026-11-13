"""Figures for coloc-confounds/lec.qmd from the data written by data.ts (run that first; see its
header). fig-leak (r against the leak fraction) is computed here from the formula.
Usage: python3 -I tools/figs/coloc-confounds/figs.py <datadir>   (from the repo root)"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "coloc"))
from colocfig import BLUE, ORANGE, VERM, GREY, GREEN, fig, save, merge, noaxis, num, np  # noqa: E402

D = json.load(open(os.path.join(sys.argv[1], "coloc-confounds.json")))
OUT = "coloc-confounds"
OFFSET = 100
MAG = "#C2369C"


def im(v):
    return np.asarray(v, float).reshape(128, 128)


# 1. the forward model step by step, on a crop of the "everything at once" scene
d = D["forward"]
y0, y1, x0, x1 = 30, 70, 22, 62
crop = (slice(y0, y1), slice(x0, x1))
psfA, psfB = im(d["psfA"]), im(d["psfB"])
ra, rb = (0, 0.8 * psfA[crop].max()), (0, 0.8 * psfB[crop].max())
bg = im(d["bg"])
stages = [
    ("PSF", merge(psfA, psfB, ra, rb)),
    ("chromatic\nshift", merge(psfA, im(d["shiftB"]), ra, rb)),
    ("bleed-\nthrough", merge(psfA, im(d["bleedB"]), ra, rb)),
    ("shared\nbackground", merge(psfA + bg, im(d["bleedB"]) + bg, ra, rb)),
    ("noise", merge(im(d["measA"]) - OFFSET, im(d["measB"]) - OFFSET, ra, rb)),
]
f, axs = fig(1000, 225, 1, 6)
ax = axs[0]
ax.set_facecolor("black")
ax_, ay_, bx_, by_ = (np.asarray(d[k]) for k in ("ax", "ay", "bx", "by"))
partner = np.asarray(d["partner"])
for i in range(len(ax_)):
    if partner[i] >= 0:
        ax.plot([ax_[i], bx_[partner[i]]], [ay_[i], by_[partner[i]]], color="#F0E442", lw=1.5)
ax.scatter(ax_, ay_, s=22, color="#33dd55", zorder=3)
ax.scatter(bx_, by_, s=22, color="#ff66dd", marker="s", zorder=2)
ax.set_xlim(x0 - 0.5, x1 - 0.5); ax.set_ylim(y1 - 0.5, y0 - 0.5); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("true\npositions", fontsize=16)
for ax, (t, rgb) in zip(axs[1:], stages):
    ax.imshow(rgb[crop], interpolation="nearest")
    ax.set_title(t, fontsize=16)
    noaxis(ax)
for k in range(5):
    f.text((k + 1) / 6 + 0.002, 0.42, "→", ha="center", va="center", fontsize=20, color=GREY)
f.subplots_adjust(left=0.005, right=0.995, bottom=0.02, top=0.74, wspace=0.12)
save(f, f"{OUT}/fig-forward.svg")


def pair_panel(path, panels, w=420, h=270):
    f, axs = fig(w, h, 1, len(panels))
    for ax, (t, rgb) in zip(axs, panels):
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(t, fontsize=17)
        noaxis(ax)
    f.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.8, wspace=0.06)
    save(f, path)


# 2. shared background: measured and after the top-hat (whole field)
d = D["background"]
mA, mB = im(d["measA"]) - OFFSET, im(d["measB"]) - OFFSET
cA, cB = im(d["corrA"]), im(d["corrB"])
iA, iB = im(d["idealA"]) - OFFSET, im(d["idealB"]) - OFFSET
ra, rb = (0, 0.6 * iA.max()), (0, 0.6 * iB.max())
pair_panel(f"{OUT}/fig-background.svg", [
    (f"measured\nr = {num(d['measured']['r'])}", merge(mA, mB, ra, rb)),
    (f"after top-hat\nr = {num(d['corrected']['r'])}", merge(cA, cB, ra, rb)),
])

# 3. single-stained control: B against A (offset and background removed), slope through the origin
d = D["control"]
a, b = np.asarray(d["a"]), np.asarray(d["b"])
f, ax = fig(420, 330)
ax.scatter(a, b, s=5, color=BLUE, alpha=0.35, lw=0, rasterized=True)
xs = np.array([0, a.max()])
ax.plot(xs, d["s"] * xs, color=VERM, lw=3)
ax.text(0.97 * a.max(), d["s"] * a.max() + 6, f"slope ŝ = {d['s']:.2f}\n(true 0.30)", color=VERM, ha="right", va="bottom", fontsize=17)
ax.set_xlabel("channel A (dye A only)"); ax.set_ylabel("channel B: leakage")
ax.set_xlim(-10, a.max() * 1.03); ax.set_ylim(-10, max(b.max(), 0.45 * a.max()))
f.subplots_adjust(left=0.17, right=0.97, bottom=0.18, top=0.97)
save(f, f"{OUT}/fig-control.svg")

# 4. chromatic shift 100 nm: crop of the merge, measured and after registration with beads
d = D["chromatic"]
y0, y1, x0, x1 = 34, 66, 24, 56
crop = (slice(y0, y1), slice(x0, x1))
iA, iB = im(d["idealA"]) - OFFSET, im(d["idealB"]) - OFFSET
ra, rb = (0, 0.7 * iA.max()), (0, 0.7 * iB.max())
pair_panel(f"{OUT}/fig-chromatic.svg", [
    (f"shifted 100 nm\nr = {num(d['measured']['r'])}", merge(im(d["measA"]) - OFFSET, im(d["measB"]) - OFFSET, ra, rb)[crop]),
    (f"registered\nr = {num(d['corrected']['r'])}", merge(im(d["corrA"]) - OFFSET, im(d["corrB"]) - OFFSET, ra, rb)[crop]),
])

# 5. r through the corrections of "everything at once"
d = D["steps"]
keys = [("measured", "measured"), ("subtract", "1 back-\nground"), ("compensate", "2 compen-\nsate"), ("register", "3 register")]
f, ax = fig(450, 330)
vals = [d[k]["r"] for k, _ in keys]
ax.bar(range(4), vals, color=[GREY, BLUE, BLUE, BLUE], width=0.6)
for k, v in enumerate(vals):
    ax.text(k, v + 0.015, num(v), ha="center", fontsize=17)
ax.axhline(d["ideal"]["r"], color=ORANGE, lw=3, ls="--")
ax.text(3.45, d["ideal"]["r"] + 0.015, f"ideal {num(d['ideal']['r'])}", color=ORANGE, ha="right", va="bottom", fontsize=17)
ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in keys], fontsize=16)
ax.set_ylabel("Pearson r"); ax.set_ylim(0, 0.72); ax.set_yticks([0, 0.2, 0.4, 0.6])
f.subplots_adjust(left=0.17, right=0.98, bottom=0.22, top=0.97)
save(f, f"{OUT}/fig-order.svg")

# 6. r of A with B + sA for independent equal-variance channels
s = np.linspace(0, 1.5, 200)
f, ax = fig(400, 290)
ax.plot(s, s / np.sqrt(1 + s ** 2), color=BLUE, lw=3)
for sv in (0.3, 0.4):
    rv = sv / np.sqrt(1 + sv ** 2)
    ax.scatter([sv], [rv], color=VERM, s=50, zorder=3)
    ax.text(sv + 0.06, rv - 0.06, f"s = {sv}: {rv:.2f}", color=VERM, fontsize=16)
ax.set_xlabel("leak fraction s"); ax.set_ylabel("r(A, B + sA)")
ax.set_xlim(0, 1.5); ax.set_xticks([0, 0.5, 1, 1.5]); ax.set_ylim(0, 0.85); ax.set_yticks([0, 0.25, 0.5, 0.75])
f.subplots_adjust(left=0.2, right=0.94, bottom=0.2, top=0.96)
save(f, f"{OUT}/fig-leak.svg")
