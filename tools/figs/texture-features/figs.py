# Figures for texture-features/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/texture-features/figs.py
# Patches and GLCM features come from the glcm demo's code (data.ts, via node).
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "label-connectivity"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from labstyle import setup, fig, save, run_ts, INK, GREY, GRID

setup()
OUT = "texture-features"
D = run_ts(HERE)
P = D["P"]
BLUE, ORANGE, VERM, GREEN, SKY = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#56B4E9"
LIGHT, DARK = "#f2f2f2", "#555555"


def grid_image(ax, a, x0=0, y0=0, numbers=True, fs=20):
    h, w = a.shape
    for y in range(h):
        for x in range(w):
            v = a[y, x]
            ax.add_patch(Rectangle((x0 + x, y0 + y), 1, 1, fc=DARK if v else LIGHT, ec=GRID, lw=1))
            if numbers:
                ax.text(x0 + x + 0.5, y0 + y + 0.5, str(v), ha="center", va="center", fontsize=fs, color="white" if v else INK)


def glcm2(a):
    """Symmetric 2-level co-occurrence counts, offset one to the right."""
    C = np.zeros((2, 2), int)
    for row in a:
        for u, v in zip(row[:-1], row[1:]):
            C[u, v] += 1
    return C + C.T


def matrix(ax, C, x0, y0, fs=20, cmap=plt.cm.Blues):
    vmax = C.max()
    for i in range(2):
        for j in range(2):
            v = C[i, j]
            ax.add_patch(Rectangle((x0 + j, y0 + i), 1, 1, fc=cmap(0.15 + 0.75 * v / vmax) if v else "white", ec=INK, lw=1.2))
            ax.text(x0 + j + 0.5, y0 + i + 0.5, str(v), ha="center", va="center", fontsize=fs, color="white" if v / vmax > 0.6 else INK)
    for k in range(2):
        ax.text(x0 + k + 0.5, y0 - 0.25, str(k), ha="center", va="bottom", fontsize=16, color=GREY)
        ax.text(x0 - 0.2, y0 + k + 0.5, str(k), ha="right", va="center", fontsize=16, color=GREY)


# --- 1. two 4 x 4 images and their co-occurrence matrices ---------------------------------
checker = np.array([[(x + y) % 2 for x in range(4)] for y in range(4)])
halves = np.array([[0, 0, 1, 1]] * 4)
f, ax = fig(560, 330)
for k, (a, lab) in enumerate([(checker, "checkerboard"), (halves, "halves")]):
    y0 = 0
    x0 = k * 7.9
    grid_image(ax, a, x0, y0, fs=18)
    C = glcm2(a)
    matrix(ax, C, x0 + 5.3, y0 + 1.0)
    ax.annotate("", xy=(x0 + 4.75, 2.0), xytext=(x0 + 4.1, 2.0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
    ax.text(x0 + 2, -0.3, lab, ha="center", va="bottom", fontsize=19)
    ax.text(x0 + 6.3, 3.35, "partner →", ha="center", va="top", fontsize=15, color=GREY)
# offset arrow example on the checkerboard
ax.set_xlim(-0.3, 15.6); ax.set_ylim(4.3, -1.2); ax.set_aspect("equal"); ax.axis("off")
save(f, f"{OUT}/fig-glcm-small.svg")

# --- 2. feature weights and the pair's matrices --------------------------------------------
L = 8
i, j = np.mgrid[0:L, 0:L]
pA = np.array(D["pair"]["pA"]).reshape(L, L); pB = np.array(D["pair"]["pB"]).reshape(L, L)
fA, fB = D["pair"]["fA"], D["pair"]["fB"]
f, axs = plt.subplots(2, 2, figsize=(500 / 72, 520 / 72))
panels = [
    (axs[0, 0], (i - j) ** 2, "Oranges", "contrast weight\n(i − j)²"),
    (axs[0, 1], 1 / (1 + (i - j) ** 2), "Greens", "homogeneity weight\n1 / (1 + (i − j)²)"),
    (axs[1, 0], pA, "Blues", f"smoothed noise\ncontrast {fA['contrast']:.2f}"),
    (axs[1, 1], pB, "Blues", f"shuffled\ncontrast {fB['contrast']:.2f}"),
]
vmaxp = max(pA.max(), pB.max())
for ax, M, cm, lab in panels:
    ax.imshow(M, cmap=cm, vmin=0, vmax=vmaxp if cm == "Blues" else M.max(), interpolation="nearest")
    ax.set_xticks([0, 7]); ax.set_yticks([0, 7])
    ax.tick_params(labelsize=15, length=0)
    ax.set_title(lab, fontsize=17)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(GRID)
axs[1, 0].set_xlabel("j", fontsize=17); axs[1, 0].set_ylabel("i", fontsize=17, rotation=0, labelpad=8)
f.tight_layout(h_pad=1.0)
save(f, f"{OUT}/fig-weights.svg")

# --- 3. correlation against d -----------------------------------------------------------------
S = D["scale"]; d = np.array(S["d"])
f, (a0, a1, a2) = plt.subplots(1, 3, figsize=(980 / 72, 280 / 72), gridspec_kw=dict(width_ratios=[1, 1, 2.6]))
for a, img, lab in [(a0, S["smooth"], "smoothed noise\n(σ = 4 px)"), (a1, S["noise"], "white noise")]:
    a.imshow(np.array(img).reshape(P, P), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    a.set_xticks([]); a.set_yticks([]); a.set_title(lab, fontsize=17)
    for sp in a.spines.values(): sp.set_visible(True)
a0.annotate("", xy=(40, 56), xytext=(32, 56), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.5))
a0.text(36, 51, "d = 8", color=INK, ha="center", va="bottom", fontsize=16, bbox=dict(fc="white", ec="none", pad=1))
a2.plot(d, S["corrSmooth"], "o-", color=BLUE, lw=2.5, ms=6, label="smoothed noise")
a2.plot(d, S["corrNoise"], "s-", color=VERM, lw=2.5, ms=6, label="white noise")
a2.axhline(0, color=GREY, lw=1)
for dd in (1, 4, 8):
    v = S["corrSmooth"][dd - 1]
    a2.annotate(f"{v:.2f}", (dd, v), xytext=(8, 6), textcoords="offset points", fontsize=16, color=BLUE)
a2.set_xlabel("distance d (px)"); a2.set_ylabel("GLCM correlation")
a2.set_xticks([1, 4, 8, 12, 16]); a2.set_ylim(-0.3, 1.1)
a2.legend(frameon=False, loc="upper right")
f.tight_layout()
save(f, f"{OUT}/fig-corr-d.svg")

# --- 4. stripes: contrast against d -----------------------------------------------------------
T = D["stripes"]
f, (a0, a1) = plt.subplots(1, 2, figsize=(410 / 72, 230 / 72), gridspec_kw=dict(width_ratios=[1, 2.2]))
a0.imshow(np.array(T["img"]).reshape(P, P)[:32, :32], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
a0.set_xticks([]); a0.set_yticks([]); a0.set_title("period 8", fontsize=17)
a1.plot(d, T["con"], "o-", color=BLUE, lw=2.5, ms=6)
a1.set_xticks([1, 4, 8, 12, 16]); a1.set_xlabel("distance d (px)"); a1.set_ylabel("contrast")
for dd in (8, 16):
    a1.axvline(dd, color=GREEN, ls="--", lw=1.5)
f.tight_layout()
save(f, f"{OUT}/fig-stripes-d.svg")

# --- 5. the four standard offsets on the pixel grid ---------------------------------------------
f, ax = fig(330, 300)
for y in range(-2, 3):
    for x in range(-2, 3):
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc="#d9e4f5" if (x, y) == (0, 0) else "white", ec=GRID, lw=1))
for (dx, dy, lab, col) in [(1, 0, "0°", BLUE), (1, -1, "45°", ORANGE), (0, -1, "90°", GREEN), (-1, -1, "135°", VERM)]:
    for k in (1, 2):
        ax.annotate("", xy=(k * dx, k * dy), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5 if k == 1 else 1.5,
                    ls="-" if k == 1 else "--", mutation_scale=18, shrinkA=0, shrinkB=2))
    ax.text(2.25 * dx + (0.1 if dx >= 0 else -0.1), 2.25 * dy - 0.05, lab, color=col, fontsize=18, ha="left" if dx > 0 else ("center" if dx == 0 else "right"), va="center" if dy == 0 else "bottom")
ax.text(0, 2.75, "solid d = 1, dashed d = 2; y down", ha="center", va="top", fontsize=15, color=GREY)
ax.set_xlim(-3.3, 3.3); ax.set_ylim(3.2, -3.1); ax.set_aspect("equal"); ax.axis("off")
save(f, f"{OUT}/fig-offsets.svg")

# --- 6. Gabor kernels and an LBP code ----------------------------------------------------------
f = plt.figure(figsize=(480 / 72, 330 / 72))
gs = f.add_gridspec(2, 4, height_ratios=[1, 1.1], hspace=0.3)
yy, xx = np.mgrid[-15:16, -15:16]
sig, fr = 4.0, 1 / 8
for k, th in enumerate([0, 45, 90, 135]):
    t = np.radians(th)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sig ** 2)) * np.cos(2 * np.pi * fr * (xx * np.cos(t) - yy * np.sin(t)))
    a = f.add_subplot(gs[0, k])
    a.imshow(g, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="bilinear")
    a.set_xticks([]); a.set_yticks([]); a.set_title(f"{th}°", fontsize=17)
f.text(0.5, 0.97, "Gabor kernels (real part), period 8 px", ha="center", va="bottom", fontsize=17)
a = f.add_subplot(gs[1, :])
Lb = D["lbp"]; v = np.array(Lb["v"]).reshape(3, 3); c = v[1, 1]
for y in range(3):
    for x in range(3):
        a.add_patch(Rectangle((x, y), 1, 1, fc="#d9e4f5" if (x, y) == (1, 1) else "white", ec=INK, lw=1.2))
        a.text(x + 0.5, y + 0.5, str(v[y, x]), ha="center", va="center", fontsize=19)
        if (x, y) != (1, 1):
            b = int(v[y, x] >= c)
            a.add_patch(Rectangle((x + 4.6, y), 1, 1, fc=DARK if b else LIGHT, ec=INK, lw=1.2))
            a.text(x + 5.1, y + 0.5, str(b), ha="center", va="center", fontsize=19, color="white" if b else INK)
a.add_patch(Rectangle((5.6, 1), 1, 1, fc="white", ec=INK, lw=1.2))
a.annotate("", xy=(4.45, 1.5), xytext=(3.15, 1.5), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
a.text(3.8, 1.75, "≥ 64?", ha="center", va="top", fontsize=15, color=GREY)
code = Lb["code"]
a.text(8.0, 0.6, "read clockwise\nfrom top-left:", ha="left", va="center", fontsize=16, color=GREY)
a.text(8.0, 1.65, f"{code:08b} = {code}", ha="left", va="center", fontsize=19)
a.text(8.0, 2.5, f"rotation-invariant: {Lb['rot']}", ha="left", va="center", fontsize=16)
a.set_xlim(-0.1, 13.2); a.set_ylim(3.1, -0.1); a.set_aspect("equal"); a.axis("off")
save(f, f"{OUT}/fig-gabor-lbp.svg")
print("done")
