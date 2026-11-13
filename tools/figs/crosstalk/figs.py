# Static figures for crosstalk/lec.qmd. Run from the repository root:
#   node_modules/.bin/esbuild tools/figs/crosstalk/export.ts --bundle --platform=node --outfile=<tmp>/x.js
#   node <tmp>/x.js <tmp>
#   python3 -I tools/figs/crosstalk/figs.py <tmp>
# The cell images come from the bleedthrough demo's model (same seed, 1000 photons, true
# spillover 0.25 and 0.05); everything else is computed here from the slides' numbers.
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams.update({"font.size": 15, "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})
G1, G2, GREY = "#009E73", "#D55E00", "#8c8c8c"   # dye 1, dye 2, background (as in the demo)
OUT = "crosstalk/"

def save(fig, name, **kw):
    fig.savefig(OUT + name, **kw)
    if os.environ.get("PREVIEW"):   # optional PNG previews of the SVGs
        fig.savefig(os.path.join(os.environ["PREVIEW"], name.rsplit(".", 1)[0] + ".png"), dpi=96)
    plt.close(fig); print("wrote", OUT + name)

D = json.load(open(sys.argv[1] + "/crosstalk.json"))
w, h = D["w"], D["h"]
img = lambda v: np.array(v, float).reshape(h, w)
x, y, g = img(D["x"]), img(D["y"]), np.array(D["g"]).reshape(h, w)

# 1. fig-ghost.png: channel 1 and channel 2 as measured (the ghost of the nucleus)
fig, axs = plt.subplots(1, 2, figsize=(4.6, 2.6))
for ax, v, lab in ((axs[0], x, "channel 1"), (axs[1], y, "channel 2")):
    ax.imshow(v, cmap="gray", vmin=0, vmax=np.percentile(v, 99.5), interpolation="nearest")
    ax.set_title(lab, fontsize=15); ax.axis("off")
axs[1].annotate("ghost of\nthe nucleus", xy=(62, 62), xytext=(4, 124), color="#ffd166", fontsize=14, bbox=dict(fc="black", ec="none", alpha=0.7),
                arrowprops=dict(arrowstyle="->", color="#ffd166", lw=2))
fig.tight_layout(pad=0.3); save(fig, "fig-ghost.png", dpi=150, bbox_inches="tight")

# 2. fig-compensated.png: channel 2 measured and after compensation with the true S
cy = img(D["g0.25"]["cy"])
fig, axs = plt.subplots(1, 2, figsize=(4.6, 2.6))
hi = np.percentile(y, 99.5)
for ax, v, lab in ((axs[0], y, "channel 2, measured"), (axs[1], cy, "dye 2 after $S^{-1}$")):
    ax.imshow(v, cmap="gray", vmin=0, vmax=hi, interpolation="nearest"); ax.set_title(lab, fontsize=15); ax.axis("off")
fig.tight_layout(pad=0.3); save(fig, "fig-compensated.png", dpi=150, bbox_inches="tight")

# 3. fig-bands.svg: the five-bin example: spectra as bars, filter bands shaded
L = np.array([500, 525, 550, 575, 600]); d1 = [0.2, 1.0, 0.6, 0.3, 0.1]; d2 = [0, 0.1, 0.5, 1.0, 0.6]
fig, ax = plt.subplots(figsize=(6.0, 3.4))
ax.axvspan(487.5, 537.5, color=G1, alpha=0.12, lw=0); ax.axvspan(562.5, 612.5, color=G2, alpha=0.12, lw=0)
ax.text(512.5, 1.27, "channel 1", ha="center", color=G1, fontsize=15)
ax.text(587.5, 1.27, "channel 2", ha="center", color=G2, fontsize=15)
b1 = ax.bar(L - 5, d1, width=10, color=G1, label="dye 1"); b2 = ax.bar(L + 5, d2, width=10, color=G2, label="dye 2")
for xs, vs, c in ((L - 5, d1, G1), (L + 5, d2, G2)):
    for xx, vv in zip(xs, vs):
        ax.text(xx, vv + 0.02, f"{vv:g}", ha="center", va="bottom", fontsize=12, color=c)
ax.set_xticks(L); ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("emission"); ax.set_ylim(0, 1.38)
ax.set_xlim(485, 615); ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.86), frameon=False, fontsize=14)
fig.tight_layout(); save(fig, "fig-bands.svg")

# 4. fig-columns.svg: the columns in channel space and a mixture as their sum
m1, m2, a = np.array([1.2, 0.4]), np.array([0.1, 1.6]), (1.0, 0.5)
yv = a[0] * m1 + a[1] * m2
fig, ax = plt.subplots(figsize=(4.6, 4.2))
arr = dict(head_width=0.05, length_includes_head=True, lw=2.5)
ax.arrow(0, 0, *m1, color=G1, **arr); ax.arrow(0, 0, *m2, color=G2, **arr)
ax.arrow(*m1, *(a[1] * m2), color=G2, ls="--", **dict(arr, lw=1.5))
ax.plot(*yv, "o", color="black", ms=8)
ax.text(*(m1 + [0.0, -0.32]), "$\\mathbf{m}_1 = (1.2,\\ 0.4)$", color=G1, fontsize=15, ha="center")
ax.text(*(m2 + [0.05, 0.03]), "$\\mathbf{m}_2 = (0.1,\\ 1.6)$", color=G2, fontsize=15)
ax.text(yv[0] + 0.05, yv[1] + 0.03, "$\\mathbf{y} = \\mathbf{m}_1 + 0.5\\,\\mathbf{m}_2$\n$= (1.25,\\ 1.2)$", fontsize=15, va="bottom", ha="center")
ax.set_xlim(0, 1.75); ax.set_ylim(0, 1.9); ax.set_aspect("equal")
ax.set_xlabel("channel 1"); ax.set_ylabel("channel 2")
fig.tight_layout(); save(fig, "fig-columns.svg")

# 5. fig-control.svg: four control pixels and the slope through the origin
cx_, cy_ = np.array([100, 200, 300, 400.]), np.array([27, 48, 77, 98.])
s = (cx_ * cy_).sum() / (cx_ ** 2).sum()
fig, ax = plt.subplots(figsize=(4.6, 3.6))
xx = np.array([0, 450]); ax.plot(xx, s * xx, color=G1, lw=2)
ax.plot(cx_, cy_, "o", color="black", ms=8)
ax.text(250, 103, f"slope {s:.3f}", color=G1, fontsize=15, ha="right")
ax.set_xlim(0, 450); ax.set_ylim(0, 130); ax.set_xlabel("channel 1 (dye 1 only)"); ax.set_ylabel("channel 2")
fig.tight_layout(); save(fig, "fig-control.svg")

# 6. fig-underover.svg: the dye-1 population after compensation with three guesses
rng = np.random.default_rng(1)
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)
for ax, gs, lab in zip(axs, (0.15, 0.25, 0.35), ("guess 0.15: under", "guess 0.25: right", "guess 0.35: over")):
    C = D["g" + str(gs)]; ccx, ccy = img(C["cx"]), img(C["cy"])
    for grp, col in ((0, GREY), (1, G1)):
        idx = np.flatnonzero(g.ravel() == grp); idx = rng.choice(idx, min(600, idx.size), replace=False)
        ax.plot(ccx.ravel()[idx], ccy.ravel()[idx], ".", ms=3, color=col, alpha=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(C["med1"], color=G1, lw=1.5, ls="--")
    ax.text(1590, C["med1"] + (10 if C["med1"] > 0 else -10), f"{C['med1']:.0f}".replace("-", "\u2212"), color=G1,
            ha="right", va="bottom" if C["med1"] > 0 else "top", fontsize=15)
    ax.set_title(lab, fontsize=15); ax.set_xlabel("dye 1 (compensated)"); ax.set_xlim(0, 1600)
axs[0].set_ylabel("dye 2 (compensated)"); axs[0].set_ylim(-220, 260)
fig.tight_layout(); save(fig, "fig-underover.svg")

# 7. fig-det.svg: the parallelogram of the columns, well separated and similar dyes
fig, axs = plt.subplots(1, 2, figsize=(5.4, 3.0))
for ax, (s21, s12), lab in zip(axs, ((0.25, 0.02), (0.9, 0.9)), ("$S$", "$S_2$")):
    c1, c2 = np.array([1, s21]), np.array([s12, 1]); det = 1 - s12 * s21
    ax.add_patch(Polygon([[0, 0], c1, c1 + c2, c2], closed=True, color="#bbbbbb", alpha=0.5, lw=0))
    ax.arrow(0, 0, *c1, color=G1, **arr); ax.arrow(0, 0, *c2, color=G2, **arr)
    ax.set_title(f"{lab}: det = {det:.3g}", fontsize=15)
    ax.set_xlim(0, 2.0); ax.set_ylim(0, 2.0); ax.set_aspect("equal"); ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xlabel("channel 1"); ax.set_ylabel("channel 2")
fig.tight_layout(); save(fig, "fig-det.svg")
