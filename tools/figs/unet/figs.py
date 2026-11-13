# Figures for the unet unit (unet/lec.qmd). Run (from the repo root) after dump.ts (see its header):
#   python3 -I tools/figs/unet/figs.py
# data.json holds images of the unetseg demo's presets (same seeds) with the demo network's probability
# maps, masks at 0.5, Otsu masks and Dice values. Figure sizes are the display size in px / 96.
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/unet/'
D = json.load(open(R + '/tools/figs/unet/data.json'))
plt.rcParams.update({"font.size": 16, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c"
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)
def A(case, key): return np.array(D[case][key], float).reshape(64, 64)
def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#999")
def grey(ax, im, t=None, lo=0, hi=255):
    ax.imshow(im, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest"); off(ax)
    if t: ax.set_title(t, fontsize=17)
def errors(ax, pred, truth, t=None):
    # the demo's colours: TP white, FP red, FN blue, TN black
    rgb = np.zeros((64, 64, 3)) + 25 / 255
    rgb[(pred == 1) & (truth == 1)] = np.array([235, 235, 235]) / 255
    rgb[(pred == 1) & (truth == 0)] = np.array([214, 39, 40]) / 255
    rgb[(pred == 0) & (truth == 1)] = np.array([31, 119, 220]) / 255
    ax.imshow(rgb, interpolation="nearest"); off(ax)
    if t: ax.set_title(t, fontsize=17)

def legend(fig):
    fig.text(0.5, 0.005, "errors: white correct, red false positive, blue missed", ha="center", va="bottom", fontsize=15, color="#555")

# ------------------------------------------------------------ 1. what vs which pixels
fig, axs = plt.subplots(1, 2, figsize=(480 / 96, 270 / 96))
grey(axs[0], A("typical", "image"), "input")
axs[0].set_xlabel("classification:\n\"contains nuclei\"", fontsize=17)
axs[1].imshow(A("typical", "prob"), cmap="magma", vmin=0, vmax=1, interpolation="nearest"); off(axs[1])
axs[1].set_title("p(nucleus) per pixel", fontsize=17)
axs[1].set_xlabel("segmentation:\n64 × 64 answers", fontsize=17)
fig.tight_layout(w_pad=0.8); save(fig, "fig-whatwhich.png", dpi=192)

# ------------------------------------------------------------ 2. upsampling (1, 2, 3) by 2, by hand
x = np.array([1, 2, 3.])
nn = np.repeat(x, 2)
sm = np.convolve(nn, [0.25, 0.5, 0.25], mode="same")
assert np.allclose(sm, [0.75, 1.25, 1.75, 2.25, 2.75, 2.25])
tc = np.zeros(7)
for i, v in enumerate(x): tc[2 * i:2 * i + 3] += v * np.array([0.5, 1, 0.5])
assert np.allclose(tc, [0.5, 1, 1.5, 2, 2.5, 3, 1.5])
fig, axs = plt.subplots(1, 3, figsize=(620 / 96, 230 / 96), sharey=True, gridspec_kw=dict(width_ratios=[3, 6, 7]))
def stem(ax, v, col, t):
    ax.vlines(range(len(v)), 0, v, color=col, lw=3); ax.scatter(range(len(v)), v, color=col, s=45, zorder=3)
    ax.set_xticks(range(len(v))); ax.set_title(t, fontsize=17); ax.set_xlim(-0.6, len(v) - 0.4); ax.set_ylim(0, 3.4)
stem(axs[0], x, INK, "input")
stem(axs[1], sm, BLUE, "nearest + smooth")
axs[1].step(np.arange(6) + 0.5, nn, where="pre", color="#aaa", lw=1.5, zorder=0)
stem(axs[2], tc, ORANGE, "transposed")
for i, v in enumerate(x):
    k = np.array([0.5, 1, 0.5]) * v
    axs[2].plot(np.arange(2 * i, 2 * i + 3), k, color=ORANGE, lw=1.2, ls="--", alpha=0.7)
fig.tight_layout(w_pad=0.8); save(fig, "fig-upsample.svg")

# ------------------------------------------------------------ 3. skip connections: small nuclei
fig, axs = plt.subplots(1, 3, figsize=(600 / 96, 275 / 96))
grey(axs[0], A("noskip", "image"), "small nuclei")
errors(axs[1], A("noskip", "pred"), A("noskip", "mask"), "no skips")
axs[1].set_xlabel(f"Dice {D['noskip']['dice']:.3f}", fontsize=17)
errors(axs[2], A("noskipU", "pred"), A("noskipU", "mask"), "U-Net")
axs[2].set_xlabel(f"Dice {D['noskipU']['dice']:.3f}", fontsize=17)
fig.tight_layout(w_pad=0.6, rect=(0, 0.08, 1, 1)); legend(fig); save(fig, "fig-skips.png", dpi=192)

# ------------------------------------------------------------ 4. class imbalance
m = A("typical", "mask"); share = m.mean()
fig, axs = plt.subplots(1, 3, figsize=(600 / 96, 275 / 96))
grey(axs[0], A("typical", "image"), "input")
grey(axs[1], m, "true mask", 0, 1); axs[1].set_xlabel(f"{100 * share:.0f} % nucleus", fontsize=17)
grey(axs[2], np.zeros((64, 64)), "\"all background\"", 0, 1)
axs[2].set_xlabel(f"{100 * (1 - share):.0f} % correct\nDice 0", fontsize=17, color=RED)
fig.tight_layout(w_pad=0.6); save(fig, "fig-imbalance.png", dpi=192)
print("nucleus share typical:", round(100 * share, 1))

# ------------------------------------------------------------ 5. look at the errors: U-Net vs Otsu on uneven illumination
fig, axs = plt.subplots(1, 3, figsize=(600 / 96, 275 / 96))
grey(axs[0], A("otsu", "image"), "uneven light")
errors(axs[1], A("otsu", "pred"), A("otsu", "mask"), "U-Net"); axs[1].set_xlabel(f"Dice {D['otsu']['dice']:.3f}", fontsize=17)
errors(axs[2], A("otsu", "otsu"), A("otsu", "mask"), "Otsu"); axs[2].set_xlabel(f"Dice {D['otsu']['otsuDice']:.3f}", fontsize=17)
fig.tight_layout(w_pad=0.6, rect=(0, 0.08, 1, 1)); legend(fig); save(fig, "fig-errors.png", dpi=192)

# ------------------------------------------------------------ 6. domain shift: a dimmer microscope
fig, axs = plt.subplots(1, 4, figsize=(720 / 96, 285 / 96))
grey(axs[0], A("dimTrain", "image"), "as in training")
errors(axs[1], A("dimTrain", "pred"), A("dimTrain", "mask"), "U-Net"); axs[1].set_xlabel(f"Dice {D['dimTrain']['dice']:.3f}", fontsize=17)
grey(axs[2], A("dim", "image"), "half as bright")
errors(axs[3], A("dim", "pred"), A("dim", "mask"), "U-Net"); axs[3].set_xlabel(f"Dice {D['dim']['dice']:.3f}", fontsize=17, color=RED)
fig.tight_layout(w_pad=0.6, rect=(0, 0.08, 1, 1)); legend(fig); save(fig, "fig-domain.png", dpi=192)

# ------------------------------------------------------------ 7. touching nuclei: one mask, fewer pieces
lab = A("touching", "labels").astype(int)
cc, ncc = ndi.label(A("touching", "pred"))
ntrue = len(np.unique(lab)) - 1
pal = np.array([[0.1, 0.1, 0.1]] + [matplotlib.colors.to_rgb(c) for c in
               ["#e69f00", "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00", "#cc79a7", "#999999"]])
def colour(L):
    u = np.unique(L[L > 0]); out = np.zeros(L.shape, int)
    for i, v in enumerate(u): out[L == v] = 1 + i % 8
    return pal[out]
fig, axs = plt.subplots(1, 3, figsize=(600 / 96, 275 / 96))
grey(axs[0], A("touching", "image"), "input")
axs[1].imshow(colour(lab), interpolation="nearest"); off(axs[1]); axs[1].set_title("true nuclei", fontsize=17)
axs[1].set_xlabel(f"{ntrue} nuclei", fontsize=17)
axs[2].imshow(colour(cc), interpolation="nearest"); off(axs[2]); axs[2].set_title("mask, labelled", fontsize=17)
axs[2].set_xlabel(f"{ncc} pieces", fontsize=17, color=RED)
fig.tight_layout(w_pad=0.6); save(fig, "fig-touching.png", dpi=192)
print("touching:", ntrue, "nuclei,", ncc, "pieces")
