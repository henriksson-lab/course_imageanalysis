# Figures for the self-denoise unit (self-denoise/lec.qmd). Run from the repository root after
# dump.ts (see its header):  python3 -I tools/figs/self-denoise/figs.py DIR/sd.json
# The images are the blindspot demo's micrograph crop (128 x 128 of the course micrograph:
# transmission electron micrograph of a mesenchymal stem cell, Robert M. Hunt, CC BY 3.0, via
# Wikimedia Commons), noise sigma 25 grey levels, 5 x 5 linear filters. Sizes: display px / 96.
import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/self-denoise/'
D = json.load(open(sys.argv[1]))
N = D["W"]
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, GREY = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c", "#9aa5b1"
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)
def img(a):
    return np.array(a, dtype=float).reshape(N, N)

# ------------------------------------------------------------ 1. noisy targets average to the clean signal
rng = np.random.default_rng(3)
row = img(D["clean"])[64]
fig, ax = plt.subplots(figsize=(560 / 96, 330 / 96))
x = np.arange(64)
many = row[:64] + 25 * rng.standard_normal((50, 64))
ax.plot(x, many[0], color=ORANGE, lw=1, alpha=0.9, label="noisy target 1")
ax.plot(x, many[1], color=BLUE, lw=1, alpha=0.7, label="noisy target 2")
ax.plot(x, many.mean(0), color=GREEN, lw=2, label="mean of 50 noisy targets")
ax.plot(x, row[:64], color=INK, lw=2, ls="--", label="clean (never seen)")
ax.set_xlabel("position along a row (px)"); ax.set_ylabel("grey level")
ax.set_xlim(0, 63); ax.set_ylim(-10, 380)
ax.legend(fontsize=11.5, ncol=2, loc="upper center", frameon=False)
fig.tight_layout(); save(fig, "fig-n2n.svg")

# ------------------------------------------------------------ 2. the blind spot
fig, axs = plt.subplots(1, 2, figsize=(620 / 96, 300 / 96))
for k, ax in enumerate(axs):
    for i in range(5):
        for j in range(5):
            c = (i == 2 and j == 2)
            fc = ("#ffffff" if k == 1 else "#f2c48d") if c else "#cfe0ec"
            ax.add_patch(Rectangle((j, 4 - i), 0.94, 0.94, fc=fc, ec=INK if c else "#7a8b99", lw=2.5 if c else 1, hatch="xx" if (c and k == 1) else None))
    ax.set_xlim(-0.2, 5.2); ax.set_ylim(-1.5, 5.1); ax.set_aspect("equal"); ax.axis("off")
    if k == 0:
        ax.annotate("", xy=(2.47, 2.47), xytext=(2.47, -0.6), arrowprops=dict(arrowstyle="->", color=RED, lw=2))
        ax.text(2.47, -1.35, "input = target:\ncopy the centre", ha="center", fontsize=13, color=RED)
        ax.set_title("ordinary window", fontsize=15)
    else:
        for (j, i) in [(1, 2), (3, 2), (2, 1), (2, 3)]:
            ax.add_patch(FancyArrowPatch((j + 0.47, 4 - i + 0.47), (2.47, 2.47), arrowstyle="->", mutation_scale=14, color=GREEN, lw=2, shrinkA=6, shrinkB=12))
        ax.text(2.47, -1.35, "predict the centre\nfrom its neighbours", ha="center", fontsize=13, color=GREEN)
        ax.set_title("blind spot", fontsize=15)
fig.tight_layout(); save(fig, "fig-blindspot.svg")

# ------------------------------------------------------------ 3. results on the micrograph crop
W = D["white"]
panels = [("clean (truth)", D["clean"], None), ("noisy", W["noisy"], W["psnr"]["noisy"]),
          ("Noise2Void", W["n2v"], W["psnr"]["out"]), ("Noise2Void + pixel", D["mixed"]["img"], D["mixed"]["psnr"]),
          ("Noise2Noise", D["n2n"]["img"], D["n2n"]["psnr"]), ("Gaussian σ 0.8", W["gauss"], W["psnr"]["cmp"])]
fig, axs = plt.subplots(1, 6, figsize=(1100 / 96, 250 / 96))
for ax, (t, a, p) in zip(axs, panels):
    ax.imshow(img(a), cmap="gray", vmin=0, vmax=255, interpolation="nearest"); ax.axis("off")
    ax.set_title(t + ("" if p is None else f"\n{p:.2f} dB"), fontsize=13)
fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.78, wspace=0.04); save(fig, "fig-results.png", dpi=110)

# ------------------------------------------------------------ 4. learned weights: independent vs correlated noise
def weights(ax, w, title):
    w = np.array(w).reshape(5, 5)
    ax.imshow(w, cmap="RdBu_r", vmin=-0.45, vmax=0.45)
    for i in range(5):
        for j in range(5):
            s = "×" if (i == 2 and j == 2) else ("0" if abs(w[i, j]) < 0.005 else f"{w[i, j]:.2f}")
            ax.text(j, i, s, ha="center", va="center", fontsize=11.5, color="white" if abs(w[i, j]) > 0.25 else INK)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=13.5)
    for sp in ax.spines.values(): sp.set_visible(False)
fig, axs = plt.subplots(1, 2, figsize=(620 / 96, 320 / 96))
weights(axs[0], W["weights"], "independent noise")
weights(axs[1], D["corr"]["weights"], "correlated noise")
fig.tight_layout(); save(fig, "fig-weights.svg")

# ------------------------------------------------------------ 5. correlated noise: the noise, its autocorrelation, the result
C = D["corr"]
fig = plt.figure(figsize=(1050 / 96, 270 / 96))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.5])
for k, (t, a) in enumerate([("independent\nnoise", D["noise_white"]), ("correlated\nnoise", D["noise_corr"]),
                            (f"noisy, correlated\n{C['psnr']['noisy']:.2f} dB", C["noisy"]), (f"Noise2Void\n{C['psnr']['out']:.2f} dB", C["n2v"])]):
    ax = fig.add_subplot(gs[k])
    a = img(a)
    if k < 2: a = a[:48, :48]
    ax.imshow(a, cmap="gray", vmin=0, vmax=255, interpolation="nearest"); ax.axis("off"); ax.set_title(t, fontsize=12.5)
ax = fig.add_subplot(gs[4])
d = np.arange(5)
ax.plot(d, D["ac_white"], "-o", color=BLUE, lw=2, label="independent")
ax.plot(d, D["ac_corr"], "-o", color=RED, lw=2, label="correlated")
ax.axhline(0, color=GREY, lw=1)
ax.set_xlabel("shift (px)"); ax.set_ylabel("correlation"); ax.set_xticks(d)
ax.legend(fontsize=11, frameon=False); ax.set_title("noise vs its shifted copy", fontsize=12.5)
fig.tight_layout(pad=0.4); save(fig, "fig-corr.png", dpi=110)
print("wrote", OUT)

# ------------------------------------------------------------ 6. PSNR of the methods (micrograph crop, independent noise)
P = W["psnr"]
items = [("noisy", P["noisy"], GREY), ("Noise2\nVoid", P["all"]["n2v"], ORANGE), ("Noise2Void\n+ pixel", D["mixed"]["psnr"], ORANGE),
         ("Noise2\nNoise", P["all"]["n2n"], BLUE), ("super-\nvised", P["all"]["sup"], INK), ("Gaussian\nσ 0.8", P["cmp"], GREEN)]
fig, ax = plt.subplots(figsize=(560 / 96, 330 / 96))
xs = np.arange(len(items))
ax.bar(xs, [v - 18 for _, v, _ in items], bottom=18, color=[c for _, _, c in items], width=0.65)
for x, (_, v, _) in zip(xs, items):
    ax.text(x, v + 0.1, f"{v:.2f}", ha="center", fontsize=12.5)
ax.set_xticks(xs); ax.set_xticklabels([t for t, _, _ in items], fontsize=11)
ax.set_ylim(18, 24.8); ax.set_ylabel("PSNR (dB)")
fig.tight_layout(); save(fig, "fig-psnr.svg")

# ------------------------------------------------------------ 7. stripes: noisy and Noise2Void
S_ = D["stripes"]
fig, axs = plt.subplots(1, 2, figsize=(520 / 96, 300 / 96))
for ax, (t, a) in zip(axs, [(f"stripes: {S_['psnr']['noisy']:.2f} dB", S_["noisy"]), (f"Noise2Void: {S_['psnr']['out']:.2f} dB", S_["n2v"])]):
    ax.imshow(img(a), cmap="gray", vmin=0, vmax=255, interpolation="nearest"); ax.axis("off"); ax.set_title(t, fontsize=13)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88, wspace=0.05); save(fig, "fig-stripes.png", dpi=110)
print("wrote the PSNR and stripes figures")
