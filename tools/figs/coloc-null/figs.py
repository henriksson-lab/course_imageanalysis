"""Figures for coloc-null/lec.qmd from the data written by data.ts (run that first; see its header).
Usage: python3 -I tools/figs/coloc-null/figs.py <datadir>   (from the repo root)"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "coloc"))
from colocfig import BLUE, ORANGE, VERM, GREY, PURPLE, fig, save, merge, noaxis, num, np  # noqa: E402

D = json.load(open(os.path.join(sys.argv[1], "coloc-null.json")))
OUT = "coloc-null"


def im(v):
    return np.asarray(v, float).reshape(128, 128)


def three_images(panels, path, w=520, h=230, cmap=None):
    f, axs = fig(w, h, 1, 3)
    for ax, (arr, title) in zip(axs, panels):
        ax.imshow(arr, interpolation="nearest", cmap=cmap)
        ax.set_title(title, fontsize=17)
        noaxis(ax)
    f.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.78, wspace=0.06)
    save(f, path)


# 1. references for "no colocalisation": shift B, or shuffle it in blocks
d = D["refs"]
A, B = im(d["a"]), im(d["b"])
cells = im(d["cells"]) > 0
r = d["r"]
three_images([(merge(A, B), f"A with B\nr = {num(r[0])}"), (merge(A, im(d["shifted"])), f"B shifted 10 px\nr = {num(r[1])}"),
              (merge(A, im(d["shuf"])), f"B in shuffled blocks\nr = {num(r[2])}")], f"{OUT}/fig-references.svg", w=650, h=285)

# 2. block sizes: B, its 1-px shuffle and its 8-px block shuffle (inside the cells)
d = D["blocks"]
three_images([(im(d["b"]), "channel B"), (im(d["b1"]), "1-px shuffle"), (im(d["b8"]), "8-px blocks")], f"{OUT}/fig-blocks.svg", w=670, h=265, cmap="gray")

# 3. bleed-through passes the randomisation test
d = D["bleed"]
f, ax = fig(450, 330)
nul = np.asarray(d["null"])
ax.hist(nul, bins=np.arange(-0.3, 0.4, 0.02), color=BLUE, alpha=0.75, label="200 block shuffles")
ax.axvline(d["observed"], color=VERM, lw=3)
ax.text(d["observed"] - 0.01, 33, f"observed\nr = {num(d['observed'])}", color=VERM, ha="right", va="top", fontsize=16)
ax.set_xlabel("Pearson r, B shuffled in 8-px blocks"); ax.set_ylabel("count")
ax.set_xlim(-0.3, 0.4); ax.set_ylim(0, 35)
ax.set_title(f"40 % bleed-through: p = {d['p']:.3f}", fontsize=17)
f.subplots_adjust(left=0.15, right=0.97, bottom=0.19, top=0.9)
save(f, f"{OUT}/fig-bleednull.svg")

# 4. PDM images: colocalised, independent, segregated (red +, blue -; 99th-percentile scale as in the demo)
from matplotlib.colors import ListedColormap  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
panels = []
for k, title in [("coloc", "colocalised"), ("independent", "independent"), ("segregated", "segregated")]:
    d = D["pdm_" + k]
    p, c = im(d["pdm"]), im(d["cells"]) > 0
    lim = np.quantile(np.abs(p[c]), 0.99)
    rgb = plt.get_cmap("RdBu_r")(0.5 + 0.5 * np.clip(p / lim, -1, 1))[..., :3]
    rgb[~c] = 0.85
    panels.append((rgb, f"{title}\nICQ = {num(d['icq'])}"))
three_images(panels, f"{OUT}/fig-pdm.svg", w=420, h=215)

# 5. Costes traces
f, ax = fig(450, 340)
for k, col, lab in [("coloc", BLUE, "colocalised"), ("weak", ORANGE, "weak correlation"), ("background", PURPLE, "shared background")]:
    d = D["costes_" + k]
    tr = d["trace"]
    ax.plot([t["tA"] for t in tr], [t["r"] for t in tr], color=col, lw=2.5, label=lab)
    ax.scatter([tr[-1]["tA"]], [tr[-1]["r"]], color=col, s=60, marker="D", zorder=3)
ax.axhline(0, color=GREY, lw=1)
ax.set_xlim(240, 0)
ax.set_ylim(-0.1, 1.05)
ax.set_xlabel("$t_A$ (lowered →)"); ax.set_ylabel("r of the pixels below")
ax.legend(loc="center left", frameon=False, bbox_to_anchor=(0.0, 0.62))
f.subplots_adjust(left=0.16, right=0.97, bottom=0.19, top=0.97)
save(f, f"{OUT}/fig-costestrace.svg")
