# Figures for the unit image-translation (image-translation/lec.qmd). Run (from the repo root) after
# dump.ts (see its header):  python3 -I tools/figs/image-translation/figs.py
# data.json: the translate1d demo's pairs and translators. The cell images are synthetic sketches
# made here (seeded): what a mean predictor and a sampler produce when a marker is not visible.
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/image-translation/'
D = json.load(open(R + '/tools/figs/image-translation/data.json'))
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, GREY, PURPLE = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c", "#9aa3ab", "#7b5ea7"
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)
gx = np.array(D["gx"])

def box(ax, x, y, w, h, text, col, fs=14):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=col, ec=INK, lw=1.5))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)
def arrow(ax, a, b, col=INK, ls="-", rad=0.0):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=col, lw=2, ls=ls, mutation_scale=18, connectionstyle=f"arc3,rad={rad}"))

# ------------------------------------------------------------ 1. paired translation (conditional GAN)
fig, ax = plt.subplots(figsize=(720 / 96, 300 / 96))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")
box(ax, 1.1, 2.6, 2.2, 1.0, "input image\n(label-free)", "#eef2f6", 13)
box(ax, 3.75, 2.6, 2.3, 1.0, "generator G\n(+ noise z)", "#d7e6f2", 13)
box(ax, 6.3, 2.6, 2.0, 1.0, "translated\n(stain)", "#eef2f6", 13)
box(ax, 6.3, 0.7, 2.0, 1.0, "measured\nstain", "#eef2f6", 13)
box(ax, 8.9, 3.3, 2.0, 1.0, "D: is this\npair real?", "#f4dede", 13)
arrow(ax, (2.2, 2.6), (2.6, 2.6)); arrow(ax, (4.9, 2.6), (5.3, 2.6)); arrow(ax, (7.3, 2.8), (7.9, 3.1))
arrow(ax, (1.1, 3.1), (7.95, 3.6), rad=-0.12)
ax.annotate("", xy=(6.3, 1.2), xytext=(6.3, 2.1), arrowprops=dict(arrowstyle="<|-|>", color=GREEN, lw=2, mutation_scale=16))
ax.text(6.55, 1.65, "+ λ · |difference|  (L1)", color=GREEN, fontsize=13)
save(fig, "fig-pix2pix.svg")

# ------------------------------------------------------------ 2. cycle consistency (unpaired)
fig, ax = plt.subplots(figsize=(620 / 96, 300 / 96))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
box(ax, 1.7, 2.3, 3.0, 1.6, "domain A\nlabel-free\nimages", "#eef2f6", 14)
box(ax, 8.3, 2.3, 3.0, 1.6, "domain B\nstained\nimages", "#eef2f6", 14)
arrow(ax, (3.2, 2.9), (6.8, 2.9), rad=-0.25); ax.text(5.0, 4.15, "F: A → B", ha="center", fontsize=14)
arrow(ax, (6.8, 1.7), (3.2, 1.7), rad=-0.25); ax.text(5.0, 0.3, "G: B → A", ha="center", fontsize=14)
ax.text(5.0, 2.3, "G(F(a)) ≈ a\nF(G(b)) ≈ b", ha="center", va="center", fontsize=14, color=RED)
save(fig, "fig-cycle.svg")

# ------------------------------------------------------------ 3. mean against samples (hidden marker)
H = D["hidden"]
fig, axs = plt.subplots(1, 2, figsize=(880 / 96, 330 / 96), sharey=True)
for ax in axs:
    T = np.array(H["train"]); ax.scatter(T[:, 0], T[:, 1], s=7, color=GREY, alpha=0.7, lw=0)
    ax.set_xlabel("input x (label-free)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.1)
axs[0].plot(gx, H["mean"], color=ORANGE, lw=3, label="mean (squared error)")
axs[0].plot(gx, H["median"], color=GREEN, lw=3, label="median (absolute error)")
axs[0].legend(fontsize=12, frameon=False, loc="upper left"); axs[0].set_ylabel("target y (stain)")
S_ = np.array(H["samples"]); axs[1].scatter(S_[:, 0], S_[:, 1], s=9, color=BLUE, lw=0, label="samples (conditional GAN)")
axs[1].legend(fontsize=12, frameon=False, loc="upper left")
sc = H["scores"]
axs[0].set_title(f"error {sc['rmseMean']:.3f}, realistic {100 * sc['realMean']:.0f} %", fontsize=13)
axs[1].set_title(f"error {sc['rmseSample']:.3f}, realistic {100 * sc['realSample']:.0f} %", fontsize=13)
fig.tight_layout(); save(fig, "fig-meansample.svg")

# ------------------------------------------------------------ 4. treated test cells
Tr = D["treated"]; sc = Tr["scores"]
fig, axs = plt.subplots(1, 2, figsize=(820 / 96, 300 / 96), gridspec_kw=dict(width_ratios=[1.5, 1]))
ax = axs[0]
Te = np.array(Tr["test"]); ax.scatter(Te[:, 0], Te[:, 1], s=9, color=INK, lw=0, label="treated cells (truth)")
S_ = np.array(Tr["samples"]); ax.scatter(S_[:, 0] + 0.004, S_[:, 1], s=9, color=BLUE, lw=0, alpha=0.7, label="translated")
ax.set_xlabel("input x"); ax.set_ylabel("stain y"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.1); ax.legend(fontsize=12, frameon=False, loc="upper left")
ax = axs[1]
ax.bar([0, 1], [100 * sc["onTruth"], 100 * sc["onSample"]], color=[INK, BLUE], width=0.6)
ax.set_xticks([0, 1]); ax.set_xticklabels(["measured", "translated"]); ax.set_ylabel("marker-on cells (%)"); ax.set_ylim(0, 100)
for i, v in enumerate([sc["onTruth"], sc["onSample"]]): ax.text(i, 100 * v + 3, f"{100 * v:.0f} %", ha="center")
ax.axhline(20, color=GREY, ls="--", lw=1); ax.text(0.5, 13, "training\n20 %", fontsize=10, ha="center", color="#555")
fig.tight_layout(); save(fig, "fig-treated.svg")

# ------------------------------------------------------------ 5. unpaired: matching distributions, not pairs
fig, ax = plt.subplots(figsize=(430 / 96, 330 / 96))
T = np.array(H["train"]); ax.scatter(T[:, 0], T[:, 1], s=7, color=GREY, alpha=0.7, lw=0, label="pairs (not used)")
ax.plot(gx, H["upInc"], color=PURPLE, lw=3, label="increasing map")
ax.plot(gx, H["upDec"], color=PURPLE, lw=3, ls="--", label="decreasing map")
ax.set_xlabel("input x"); ax.set_ylabel("target y"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.1); ax.legend(fontsize=12, frameon=False, loc="lower right")
fig.tight_layout(); save(fig, "fig-unpaired.svg")

# ------------------------------------------------------------ 6. synthetic cells: measured, mean, samples
rng = np.random.default_rng(11)
N = 128
yy, xx = np.mgrid[0:N, 0:N].astype(float)
cells = [(30, 34, 20, 15, 0.4), (88, 28, 18, 14, -0.3), (40, 90, 17, 21, 0.9), (95, 92, 22, 16, 0.2)]
def ell(cx, cy, a, b, t, s=1.0):
    u = ((xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)) / (a * s); v = (-(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)) / (b * s)
    return u * u + v * v
def blur(img, s):
    k = np.exp(-0.5 * (np.arange(-3 * s, 3 * s + 1) / s) ** 2); k /= k.sum()
    img = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, img)
    return np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, img)
def spots(mask, n, r):
    out = np.zeros((N, N)); pts = np.argwhere(mask)
    for i in rng.choice(len(pts), n, replace=False):
        y0, x0 = pts[i]; out += np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * r * r))
    return out
inp = np.full((N, N), 0.5); nuc = np.zeros((N, N)); cyto = []
for (cx, cy, a, b, t) in cells:
    e = ell(cx, cy, a, b, t); n = ell(cx, cy, a, b, t, 0.45)
    inp += 0.25 * np.exp(-((np.sqrt(e) - 1) / 0.08) ** 2) - 0.08 * (e < 1) - 0.07 * (n < 1)
    nuc += (n < 1); cyto.append((e < 1) & (n >= 1))
inp = blur(inp, 1.2) + 0.02 * rng.normal(size=(N, N))
nucS = blur(nuc, 1.5) * 0.8
on = [True, False, True, False]
truth = nucS + sum(spots(c, 6, 1.6) * 0.9 for c, o in zip(cyto, on) if o)
mean = nucS + sum(blur(c.astype(float), 2) * 0.12 for c in cyto)          # every cell: half the time 6 spots, averaged into haze
samp1 = nucS + sum(spots(c, 6, 1.6) * 0.9 for c, o in zip(cyto, [False, True, True, False]) if o)
samp2 = nucS + sum(spots(c, 6, 1.6) * 0.9 for c, o in zip(cyto, [True, True, False, True]) if o)
fig, axs = plt.subplots(1, 5, figsize=(1000 / 96, 250 / 96))
for ax, im, t, cmap in [(axs[0], inp, "input (label-free)", "gray"), (axs[1], truth, "measured stain", "magma"), (axs[2], mean, "mean: a haze", "magma"),
                        (axs[3], samp1, "sample 1: sharp", "magma"), (axs[4], samp2, "sample 2: other cells", "magma")]:
    ax.imshow(im, cmap=cmap, vmin=0 if cmap != "gray" else None, vmax=1 if cmap != "gray" else None); ax.set_title(t, fontsize=13); ax.axis("off")
fig.tight_layout(); save(fig, "fig-stain.png", dpi=110)
print("wrote image-translation figures")
