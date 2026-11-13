"""Figures for the FRET day decks (fret-physics, sensitised-emission, acceptor-bleach,
ratiometric), the ones that are not in figs.ts. Run from the repo root (see data.ts):
   python3 -I tools/figs/fret/figs.py <tmp>/fret-data.json
Image figures use the demos' own scenes and preset parameters (data.ts); plots and diagrams
use the numbers of the slides. All data synthetic.
SVG sizes: 1 pt in the figure = 1 px on the slide at the width given in lec.qmd."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, FancyBboxPatch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
plt.rcParams.update({"font.size": 18, "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "fret"})
DON, ACC, PURPLE, BLUE, RED, GREEN, GREY, FG = "#1b9fc0", "#d39e00", "#7b3294", "#0072b2", "#d55e00", "#009e73", "#777777", "#222222"
D = json.load(open(sys.argv[1]))


def fig(w, h, disp=None):
    """A figure laid out on a w × h grid, drawn at `disp` px wide on the slide (fonts in px there)."""
    disp = disp or w
    return plt.figure(figsize=(disp / 72, h * disp / w / 72))


def save(f, unit, name):
    f.savefig(os.path.join(ROOT, unit, name), facecolor="white", dpi=200)
    if os.environ.get("FIG_PREVIEW"):  # optional PNG previews for checking
        f.savefig(os.path.join(os.environ["FIG_PREVIEW"], name.replace(".svg", ".png")), facecolor="white", dpi=72)
    plt.close(f)
    print("wrote", unit + "/" + name)


def img(ax, a, cmap="gray", lo=None, hi=None, title=None):
    ax.imshow(a, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_color("#999")
    if title: ax.set_title(title, fontsize=18)


def grid(o, key):
    return np.array(o[key], float).reshape(o["h"], o["w"])


def arrow(ax, p, q, col, lw=3, style="-|>", ms=22, ls="-", cs="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=col, lw=lw, linestyle=ls, connectionstyle=cs))


R0 = 5.0
E = lambda r: 1 / (1 + (r / R0) ** 6)
rofE = lambda e: R0 * (1 / e - 1) ** (1 / 6)

# =====================================================================================
# fret-physics
# -------------------------------------------------------------------------------------
# 1. length scales on a log axis
f = fig(1100, 130, 1040); ax = f.add_axes([0.03, 0.3, 0.92, 0.66])
ax.set_xscale("log"); ax.set_xlim(1, 1000); ax.set_ylim(0, 1)
ax.spines["left"].set_visible(False); ax.set_yticks([])
ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "200", "500", "1000"])
ax.text(1.0, -0.42, "nm", transform=ax.transAxes, fontsize=18, ha="left")
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
for (a, b, y, col, lab) in [(2, 10, 0.55, PURPLE, "FRET range 2–10 nm"), (2.4, 4.2, 0.12, DON, "protein 2.4 × 4.2"),
                            (65, 100, 0.55, GREY, "pixel 65–100"), (180, 220, 0.12, RED, "resolution ~200 nm")]:
    ax.add_patch(Rectangle((a, y), b - a, 0.33, color=col, alpha=0.85))
    ax.text(b * 1.12 if lab.startswith(("protein", "resolution")) is False else b * 1.12, y + 0.16, lab, va="center", fontsize=18, color=col if col != GREY else FG)
save(f, "fret-physics", "fig-scale.svg")

# -------------------------------------------------------------------------------------
# 2. the cartoon: energy levels, transfer without a photon
f = fig(520, 440, 450); ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 520); ax.set_ylim(0, 440); ax.axis("off")
for x0, lab, col in [(40, "donor", DON), (300, "acceptor", ACC)]:
    ax.plot([x0, x0 + 180], [70, 70], color=FG, lw=3)
    ax.text(x0 + 90, 30, lab, ha="center", fontsize=20, color=col, fontweight="bold")
ax.plot([40, 220], [330, 330], color=FG, lw=3); ax.text(220, 342, "excited", fontsize=16, color=GREY, ha="right")
ax.plot([300, 480], [300, 300], color=FG, lw=3); ax.text(308, 312, "excited", fontsize=17, color=GREY)
ax.text(220, 80, "ground", fontsize=16, color=GREY, ha="right"); ax.text(308, 80, "ground", fontsize=17, color=GREY)
arrow(ax, (75, 72), (75, 326), BLUE, lw=4); ax.text(66, 200, "blue light\nabsorbed", ha="right", fontsize=16, color=BLUE, rotation=90, va="center")
arrow(ax, (150, 328), (150, 74), DON, lw=3, ls=(0, (6, 4))); ax.text(160, 150, "cyan photon\n(no FRET)", fontsize=16, color=DON)
arrow(ax, (222, 332), (298, 304), PURPLE, lw=4, cs="arc3,rad=-0.35"); ax.text(262, 385, "transfer, no photon\nr ≈ a few nm", ha="center", fontsize=16, color=PURPLE)
arrow(ax, (420, 298), (420, 74), ACC, lw=4); ax.text(430, 185, "yellow\nphoton", fontsize=16, color="#a77b00")
save(f, "fret-physics", "fig-cartoon.svg")

# -------------------------------------------------------------------------------------
# 3. orientation: kappa^2 for three dipole arrangements
f = fig(520, 300, 450); ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 520); ax.set_ylim(0, 300); ax.axis("off")
def dip(x, y, dx, dy, col):
    arrow(ax, (x - dx, y - dy), (x + dx, y + dy), col, lw=5, ms=26)
    ax.add_patch(Circle((x, y), 6, color=col))
cases = [(90, "collinear", "κ² = 4", [(40, 0), (40, 0)]), (260, "side by side", "κ² = 1", [(0, 40), (0, 40)]), (430, "perpendicular", "κ² = 0", [(0, 40), (0, 0)])]
for cx, name, k, ((ax1, ay1), (ax2, ay2)) in cases:
    yD, yA = 210, 110
    if name == "collinear":
        dip(cx - 45, 160, 32, 0, DON); dip(cx + 45, 160, 32, 0, ACC)
    elif name == "side by side":
        dip(cx - 40, 160, 0, 34, DON); dip(cx + 40, 160, 0, 34, ACC)
    else:
        dip(cx - 40, 160, 0, 34, DON)
        ax.add_patch(Circle((cx + 40, 160), 18, fill=False, ec=ACC, lw=4)); ax.add_patch(Circle((cx + 40, 160), 5, color=ACC))
    ax.text(cx, 255, name, ha="center", fontsize=18)
    ax.text(cx, 70, k, ha="center", fontsize=22, fontweight="bold", color=PURPLE)
ax.text(260, 20, "donor cyan, acceptor yellow; ⊙ out of the page", ha="center", fontsize=16, color=GREY)
save(f, "fret-physics", "fig-kappa.svg")

# -------------------------------------------------------------------------------------
# 4. apparent distance: the inverse curve and two error bands
f = fig(520, 420, 420); ax = f.add_axes([0.17, 0.17, 0.78, 0.78])
e = np.linspace(0.005, 0.995, 400); ax.plot(e, rofE(e), color=BLUE, lw=3.5)
for lo, hi, col in [(0.25, 0.35, GREEN), (0.90, 1.0, RED)]:
    rlo, rhi = rofE(hi), (rofE(lo) if lo > 0 else 0)
    if hi >= 1: rlo = 0
    ax.axvspan(lo, hi, color=col, alpha=0.15); ax.axhspan(rlo, rhi, xmin=0, xmax=1, color=col, alpha=0.15)
ax.annotate("E = 0.30 ± 0.05\n→ r 5.54–6.00 nm", (0.35, 5.8), (0.42, 7.2), fontsize=17, color=GREEN, arrowprops=dict(arrowstyle="-", color=GREEN))
ax.annotate("E = 0.95 ± 0.05\n→ r 0–3.47 nm", (0.9, 2.2), (0.42, 1.1), fontsize=17, color=RED, arrowprops=dict(arrowstyle="-", color=RED))
ax.set_xlim(0, 1); ax.set_ylim(0, 9); ax.set_xlabel("efficiency E"); ax.set_ylabel("apparent distance r (nm)")
ax.axhline(R0, color=GREY, ls="--", lw=1.5); ax.text(0.02, R0 + 0.15, "R₀ = 5 nm", color=GREY, fontsize=16)
save(f, "fret-physics", "fig-apparent.svg")
print("apparent check", rofE(0.25), rofE(0.35), rofE(0.9))

# -------------------------------------------------------------------------------------
# 5. mixture: mean of E is not E of the mean distance
f = fig(520, 400, 490); ax = f.add_axes([0.16, 0.17, 0.8, 0.78])
x = np.linspace(0, 2, 400); ax.plot(x, E(x * R0), color=BLUE, lw=3.5)
p1, p2 = (0.5, E(2.5)), (1.0, E(5.0)); mE = (p1[1] + p2[1]) / 2; ra = rofE(mE) / R0
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=GREY, lw=2, ls="--")
for p in (p1, p2): ax.plot(*p, "o", color=RED, ms=11)
ax.plot(0.75, mE, "s", color=RED, ms=11); ax.plot(ra, mE, "D", color=PURPLE, ms=11)
ax.plot([0.75, 0.75], [0, mE], color=RED, lw=1.5, ls=":"); ax.plot([ra, ra], [0, mE], color=PURPLE, lw=1.5, ls=":")
ax.text(0.53, 1.03, "0.985", color=RED, fontsize=16); ax.text(1.03, 0.52, "0.500", color=RED, fontsize=16)
ax.text(0.06, mE - 0.1, f"mean E = {mE:.3f}", color=RED, fontsize=16)
ax.text(0.66, 0.08, "mean r\n0.75 R₀", color=RED, fontsize=16, ha="right")
ax.annotate(f"apparent r\n{ra:.2f} R₀", (ra, mE), (1.15, 0.7), color=PURPLE, fontsize=16, arrowprops=dict(arrowstyle="-", color=PURPLE))
ax.set_xlim(0, 2); ax.set_ylim(0, 1.1); ax.set_xticks([0, 0.5, 1, 1.5, 2]); ax.set_xticklabels(["0", "0.5", "1", "1.5", "2"])
ax.set_xlabel("distance r / R₀"); ax.set_ylabel("efficiency E")
save(f, "fret-physics", "fig-mixture.svg")
print("mixture mean E", mE, "apparent", ra)

# =====================================================================================
# sensitised-emission (threecube scene)
FD = D["field"]; OFF = FD["offset"]; PH = FD["photons"]
lab = np.array(FD["labels"]).reshape(FD["h"], FD["w"])
I = {k: grid(FD, k) - OFF for k in ["IDD", "IDA", "IAA", "sIDD", "sIDA", "sIAA"]}
SHORT = {"donor": "D", "acceptor": "A", "free": "N", "fusion": "F", "binding": "B"}
cent = [np.argwhere(lab == k + 1).mean(0) for k in range(len(FD["cls"]))]
mask = (I["IDD"] + I["IAA"]) >= 0.12 * PH
def letters(ax, col="k", bg=None):
    for k, c in enumerate(FD["cls"]):
        ax.text(cent[k][1], cent[k][0], SHORT[c], ha="center", va="center", fontsize=17, fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.1", fc=bg, ec="none", alpha=0.8) if bg else None)
def fcmap(dd, da, aa, d, a):
    v = da - d * dd - a * aa; return np.where(mask, v, np.nan)
cm_div = matplotlib.colormaps["RdBu_r"].copy(); cm_div.set_bad("#282828")

# 6. the FRET channel lights up, the corrected one does not
f = fig(540, 360, 470)
a1 = f.add_axes([0.01, 0.2, 0.47, 0.7]); a2 = f.add_axes([0.52, 0.2, 0.47, 0.7])
img(a1, I["IDA"] + OFF, lo=0, hi=1.6 * PH + OFF, title=r"FRET channel $I_\mathrm{DA}$"); letters(a1, "k", "white")
img(a2, fcmap(I["IDD"], I["IDA"], I["IAA"], FD["d"], FD["a"]), cm_div, -0.6 * PH, 0.6 * PH, title=r"corrected $F_\mathrm{c}$"); letters(a2, "k", "white")
f.text(0.5, 0.05, "D donor only · A acceptor only · N no FRET\nF fusion · B binding;  red: $F_\\mathrm{c}$ > 0, white: 0", ha="center", fontsize=16)
save(f, "sensitised-emission", "fig-field.svg")

# 7. stacked bar of the worked example
f = fig(560, 210, 450); ax = f.add_axes([0.03, 0.22, 0.94, 0.5]); ax.axis("off"); ax.set_xlim(0, 520); ax.set_ylim(0, 1)
x0 = 10
for w, col, t1, t2 in [(200, DON, "$d\\,I_\\mathrm{DD}$\n= 200", "bleed-through"), (100, ACC, "$a\\,I_\\mathrm{AA}$\n= 100", "cross-exc."), (200, PURPLE, "$F_\\mathrm{c}$\n= 200", "sensitised")]:
    ax.add_patch(Rectangle((x0, 0), w, 1, color=col, alpha=0.85)); ax.text(x0 + w / 2, 0.5, t1, ha="center", va="center", fontsize=17, color="white" if col != ACC else FG, fontweight="bold")
    ax.text(x0 + w / 2, -0.12, t2, ha="center", va="top", fontsize=17)
    x0 += w
ax.text(10, 1.2, r"$I_\mathrm{DA}$ = 500 (FRET channel)", fontsize=18)
save(f, "sensitised-emission", "fig-contamination.svg")

# 8. the controls: slopes through the origin
f = fig(540, 330, 460)
for j, (src, key, k, col, nm) in enumerate([("donor", "IDD", FD["d"], DON, "d"), ("acceptor", "IAA", FD["a"], ACC, "a")]):
    ax = f.add_axes([0.13 + 0.48 * j, 0.21, 0.36, 0.68])
    m = np.isin(lab, [i + 1 for i, c in enumerate(FD["cls"]) if c == src]); idx = np.flatnonzero(m.ravel())[::2]
    xx, yy = I[key].ravel()[idx], I["IDA"].ravel()[idx]
    ax.plot(xx, yy, ".", color=col, ms=4, alpha=0.5); xm = xx.max() * 1.05
    ax.plot([0, xm], [0, k * xm], color=FG, lw=2.5); ax.set_xlim(0, xm); ax.set_ylim(-20, 330)
    ax.set_xlabel({"IDD": r"$I_\mathrm{DD}$", "IAA": r"$I_\mathrm{AA}$"}[key]); ax.set_title(f"{src} only", fontsize=18)
    ax.text(0.05, 0.95, f"slope {nm} = {k:.3f}", transform=ax.transAxes, fontsize=17, va="top")
    if j == 0: ax.set_ylabel(r"$I_\mathrm{DA}$")
    ax.set_xticks([0, 200, 400]) if xm > 400 else None
save(f, "sensitised-emission", "fig-controls.svg")

# 9. wrong d: false Fc follows the donor
f = fig(520, 380, 450); ax = f.add_axes([0.17, 0.19, 0.78, 0.72])
m = np.isin(lab, [i + 1 for i, c in enumerate(FD["cls"]) if c in ("free", "donor")]); idx = np.flatnonzero(m.ravel())[::2]
dd, da, aa = I["IDD"].ravel()[idx], I["IDA"].ravel()[idx], I["IAA"].ravel()[idx]
for d, col, nm in [(0.4, RED, "d = 0.4 (too small)"), (FD["d"], GREY, f"d = {FD['d']:.3f} (measured)")]:
    fc = da - d * dd - FD["a"] * aa; ax.plot(dd, fc, ".", color=col, ms=4, alpha=0.5)
    s = np.sum(dd * fc) / np.sum(dd * dd); xm = dd.max(); ax.plot([0, xm], [0, s * xm], color=col, lw=3, label=nm)
ax.axhline(0, color=FG, lw=1); ax.set_xlabel(r"donor $I_\mathrm{DD}$ (donor-only and no-FRET cells)"); ax.set_ylabel(r"$F_\mathrm{c}$")
ax.legend(frameon=False, fontsize=16, loc="upper left")
save(f, "sensitised-emission", "fig-wrong-d.svg")

# 10. indices against the acceptor amount (the worked example, acceptor scaled)
f = fig(500, 360, 410); ax = f.add_axes([0.15, 0.2, 0.8, 0.72])
s_ = np.geomspace(0.25, 4, 200); IAA = 1000 * s_
ax.plot(s_, 200 / (200 + 2 * 400) + 0 * s_, color=GREEN, lw=3.5, label=r"$E_\mathrm{app}$")
ax.plot(s_, 200 / IAA, color=RED, lw=3, label=r"$F_\mathrm{c}\,/\,I_\mathrm{AA}$")
ax.plot(s_, 200 / np.sqrt(400 * IAA), color=BLUE, lw=3, ls="--", label=r"$F_\mathrm{c}\,/\sqrt{I_\mathrm{DD} I_\mathrm{AA}}$")
ax.plot([1, 1, 1], [0.2, 0.2, 0.316], "o", color=FG, ms=8)
ax.set_xscale("log"); ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["¼", "½", "1", "2", "4"]); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_ylim(0, 0.8); ax.set_xlabel("acceptor amount (× the example)"); ax.set_ylabel("value")
ax.legend(frameon=False, fontsize=16, loc="upper right")
save(f, "sensitised-emission", "fig-indices.svg")

# 11. one pixel shift: edges in F_c
f = fig(520, 260, 410)
a1 = f.add_axes([0.01, 0.03, 0.47, 0.8]); a2 = f.add_axes([0.52, 0.03, 0.47, 0.8])
CR = (slice(0, 64), slice(40, 128))
img(a1, fcmap(I["IDD"], I["IDA"], I["IAA"], FD["d"], FD["a"])[CR], cm_div, -0.6 * PH, 0.6 * PH, title="aligned")
img(a2, np.where(mask, I["sIDA"] - FD["ds"] * I["sIDD"] - FD["as"] * I["sIAA"], np.nan)[CR], cm_div, -0.6 * PH, 0.6 * PH, title=r"$I_\mathrm{DA}$ shifted 1 px")
save(f, "sensitised-emission", "fig-shift.svg")

# =====================================================================================
# acceptor-bleach (apbleach scene)
B = D["bleach"]; bw, bh = B["w"], B["h"]; x0, y0, x1, y1 = B["rect"]
def bgrid(k, which): return np.array(B[which][k], float).reshape(bh, bw)
def emap(which):
    P, Q = bgrid("pre", which), bgrid("post", which); thr = 0.15 * B["photons"]
    with np.errstate(divide="ignore", invalid="ignore"): e = 1 - P / Q
    return np.where((P > thr) | (Q > thr), e, np.nan)
def box(ax): ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec="#ffd400", lw=2.5, ls="--"))

# 12. before, after, E map
f = fig(560, 260, 490)
for j, (a, t, kw) in enumerate([(bgrid("pre", "ideal"), "donor before", dict(lo=0, hi=1.2 * B["photons"])),
                                (bgrid("post", "ideal"), "donor after", dict(lo=0, hi=1.2 * B["photons"])),
                                (emap("ideal"), "E = 1 − pre/post", dict(cmap=cm_div, lo=-0.6, hi=0.6))]):
    ax = f.add_axes([0.005 + j * 0.333, 0.17, 0.323, 0.7]); img(ax, a, title=t, **kw); box(ax)
f.text(0.5, 0.04, f"region (ratio of sums): E = {B['ideal']['Ereg']:.3f}; outside ≈ 0", ha="center", fontsize=17)
save(f, "acceptor-bleach", "fig-bleach-maps.svg")

# 13. incomplete bleaching
f = fig(500, 360, 450); ax = f.add_axes([0.16, 0.2, 0.78, 0.72])
b = np.linspace(0, 1, 200)
for Et, col in [(0.3, BLUE), (0.6, PURPLE)]:
    ax.plot(b, b * Et / (1 - (1 - b) * Et), color=col, lw=3.5); ax.axhline(Et, color=col, ls=":", lw=1.5)
    ax.text(0.02, Et + 0.02, f"true E = {Et}", color=col, fontsize=16)
eo = 0.8 * 0.3 / (1 - 0.2 * 0.3); ax.plot(0.8, eo, "o", color=RED, ms=10); ax.annotate(f"b = 0.8: {eo:.3f}", (0.8, eo), (0.42, 0.1), fontsize=17, color=RED, arrowprops=dict(arrowstyle="-", color=RED))
ax.set_xlim(0, 1); ax.set_ylim(0, 0.7); ax.set_xlabel("fraction of acceptor bleached b"); ax.set_ylabel("observed E")
save(f, "acceptor-bleach", "fig-incomplete.svg")

# 14. the cell moved
f = fig(500, 300, 410)
for j, (k, t) in enumerate([("moved", f"3 px drift: E = {B['moved']['Ereg']:.3f}"), ("aligned", f"aligned: E = {B['aligned']['Ereg']:.3f}")]):
    ax = f.add_axes([0.01 + 0.5 * j, 0.04, 0.47, 0.8]); img(ax, emap(k), cm_div, -0.6, 0.6, title=t); box(ax)
save(f, "acceptor-bleach", "fig-moved.svg")

# 15. pixel E maps at two photon levels (uniform E = 0.25 patch)
rng = np.random.default_rng(11)
f = fig(540, 330, 460)
f.text(0.43, 0.93, "true E = 0.25 in every pixel", ha="center", fontsize=17)
for j, (P, Q) in enumerate([(600, 800), (60, 80)]):
    p, q = rng.poisson(P, (40, 40)), rng.poisson(Q, (40, 40)); e = 1 - p / q
    ax = f.add_axes([0.01 + 0.42 * j, 0.03, 0.4, 0.66]); img(ax, e, "viridis", 0, 0.6, title=f"{P} / {Q} photons\npixel sd {e.std():.2f}")
    if j == 1:
        cax = f.add_axes([0.85, 0.03, 0.025, 0.66]); cb = f.colorbar(ax.images[0], cax=cax); cb.set_ticks([0, 0.25, 0.5])
    print("pixel E sd", P, Q, e.std(), "pred", P / Q * np.sqrt(1 / P + 1 / Q))
save(f, "acceptor-bleach", "fig-pixel-noise.svg")

# 16. f·E: two populations with the same total donor brightness
f = fig(560, 340, 460); ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 560); ax.set_ylim(0, 340); ax.axis("off")
rows = [(205, "50 % paired, E = 0.4", [0.6] * 5 + [1.0] * 5), (40, "100 % paired, E = 0.2", [0.8] * 10)]
for yb, t, hs in rows:
    ax.text(20, yb + 105, t, fontsize=18)
    for i, hgt in enumerate(hs):
        ax.add_patch(Rectangle((20 + i * 30, yb), 22, 80, fill=False, ec="#bbb", lw=1.5))
        ax.add_patch(Rectangle((20 + i * 30, yb), 22, 80 * hgt, color=DON if hgt < 1 else "#8fd3e6"))
    ax.text(330, yb + 40, f"total {sum(hs):.0f} of 10\nE_app = {1 - sum(hs) / 10:.2f}", fontsize=18, va="center")
ax.text(20, 8, "box = unquenched donor; dark: paired, light: unpaired", fontsize=16, color=GREY)
save(f, "acceptor-bleach", "fig-fe.svg")

# =====================================================================================
# ratiometric (ratioimage scene, frame 20)
S = D["sensor"]; sw, sh = S["w"], S["h"]
Ds, As = np.array(S["D"]).reshape(sh, sw), np.array(S["A"]).reshape(sh, sw)
Db, Ab = Ds - S["bgD"], As - S["bgA"]
with np.errstate(divide="ignore", invalid="ignore"): Rr = Ab / Db
smask = (Ab + Db) >= 0.4 * S["photons"]
cm_v = matplotlib.colormaps["viridis"]
def rgb_ratio(Rv, m, imd=False):
    t = np.clip((np.log(np.clip(Rv, 1e-6, None)) - np.log(0.3)) / (np.log(3.3) - np.log(0.3)), 0, 1)
    t = np.where(np.isfinite(Rv) & (Rv > 0), t, 0)
    c = cm_v(t)[..., :3]
    if imd: c = c * np.clip((Ab + Db) / (2.8 * S["photons"]), 0, 1)[..., None]
    if m is not None: c[~m] = 0.16
    return c

# 17. the ratio cancels thickness
f = fig(900, 300, 800)
a1 = f.add_axes([0.005, 0.08, 0.27, 0.78]); a2 = f.add_axes([0.285, 0.08, 0.27, 0.78]); a3 = f.add_axes([0.565, 0.08, 0.27, 0.78])
img(a1, Ds, lo=0, hi=2.6 * S["photons"] + S["offset"], title="donor channel D")
img(a2, As, lo=0, hi=1.3 * (2.6 * S["photons"] + S["offset"]), title="acceptor channel A")
img(a3, rgb_ratio(Rr, smask), title="ratio A/D")
ca = f.add_axes([0.85, 0.12, 0.02, 0.7]); sm = matplotlib.cm.ScalarMappable(matplotlib.colors.LogNorm(0.3, 3.3), cm_v)
cb = f.colorbar(sm, cax=ca); cb.set_ticks([0.3, 0.5, 1, 2, 3.3]); cb.set_ticklabels(["0.3", "0.5", "1", "2", "3.3"]); cb.ax.minorticks_off()
for k, (yy, xx) in enumerate([np.argwhere(np.array(S["labels"]).reshape(sh, sw) == L).mean(0) for L in (1, 2, 3)]):
    a3.text(xx, yy - 30 if k != 0 else yy - 33, str(k + 1), color="white", fontsize=17, ha="center", fontweight="bold")
save(f, "ratiometric", "fig-ratio-cancels.svg")

# 18. an offset pulls the ratio towards 1
f = fig(500, 350, 410); ax = f.add_axes([0.16, 0.2, 0.78, 0.72])
dd = np.geomspace(10, 2000, 300)
ax.plot(dd, (0.25 * dd + 50) / (dd + 50), color=RED, lw=3.5, label="offset b = 50 left in")
ax.axhline(0.25, color=GREEN, lw=3, label="true ratio 0.25")
ax.plot(400, 150 / 450, "o", color=FG, ms=9); ax.annotate("D = 400: 0.33", (400, 150 / 450), (120, 0.62), fontsize=17, arrowprops=dict(arrowstyle="-"))
ax.set_xscale("log"); ax.set_xticks([10, 100, 1000]); ax.set_xticklabels(["10", "100", "1000"]); ax.set_ylim(0, 1); ax.set_xlabel("donor signal D (dim → bright)"); ax.set_ylabel("measured ratio")
ax.legend(frameon=False, fontsize=16, loc="upper right", bbox_to_anchor=(1, 1.02))
save(f, "ratiometric", "fig-offset.svg")

# 19. plain ratio without a mask vs intensity-modulated display
f = fig(520, 300, 460)
a1 = f.add_axes([0.01, 0.04, 0.47, 0.8]); a2 = f.add_axes([0.52, 0.04, 0.47, 0.8])
img(a1, rgb_ratio(Rr, None), title="ratio, no mask"); img(a2, rgb_ratio(Rr, None, imd=True), title="ratio × intensity")
save(f, "ratiometric", "fig-imd.svg")

# 20. acceptor bleaching: the control cell's ratio drifts
f = fig(500, 330, 430); ax = f.add_axes([0.16, 0.21, 0.78, 0.72]); t = np.arange(len(S["course"]))
ax.plot(t, S["course"], color=GREEN, lw=3.5, label="no bleaching")
ax.plot(t, S["courseBleach"], color=RED, lw=3.5, ls="--", label="acceptor −1 % per frame")
ax.set_ylim(0, 1.6); ax.set_xlim(0, t[-1]); ax.set_xlabel("frame"); ax.set_ylabel("ratio, control cell")
ax.text(t[-1], S["courseBleach"][-1] - 0.14, f"{S['courseBleach'][-1]:.2f}", ha="right", color=RED, fontsize=16)
ax.legend(frameon=False, fontsize=16, loc="lower left")
save(f, "ratiometric", "fig-bleach-drift.svg")
print("control course", S["course"][0], S["courseBleach"][-1])

# 21. biosensor cartoon
f = fig(540, 330, 430); ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 540); ax.set_ylim(0, 330); ax.axis("off")
def sensor(cx, closed):
    dx = 28 if closed else 85
    xs = np.linspace(cx - dx, cx + dx, 50)
    ys = 210 + (np.sin(np.linspace(0, np.pi, 50)) * (55 if closed else 10))
    ax.plot(xs, ys, color=GREY, lw=4)
    ax.add_patch(Circle((cx - dx, 210), 22, color=DON)); ax.add_patch(Circle((cx + dx, 210), 22, color=ACC))
    ax.text(cx - dx, 203, "D", ha="center", fontsize=17, color="white", fontweight="bold"); ax.text(cx + dx, 203, "A", ha="center", fontsize=17, fontweight="bold")
    if closed: ax.add_patch(Circle((cx, 270), 9, color=GREEN)); ax.text(cx + 14, 282, "Ca²⁺", fontsize=16, color=GREEN)
    # bars: donor and acceptor channel
    dv, av = (0.6, 1.5) if closed else (1.0, 0.5)
    for k, (v, col, nm) in enumerate([(dv, DON, "D"), (av, ACC, "A")]):
        ax.add_patch(Rectangle((cx - 40 + k * 50, 40), 30, 50 * v, color=col)); ax.text(cx - 25 + k * 50, 14, nm, ha="center", fontsize=17)
    ax.text(cx, 160, f"R = A/D = {av / dv:.1f}", ha="center", fontsize=18, fontweight="bold")
sensor(135, False); sensor(405, True)
ax.text(135, 300, "empty: far apart", ha="center", fontsize=18); ax.text(405, 305, "bound: close", ha="center", fontsize=18)
arrow(ax, (252, 210), (338, 210), FG, lw=2.5)
save(f, "ratiometric", "fig-biosensor.svg")
