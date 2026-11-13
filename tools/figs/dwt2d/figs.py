# Extra slide figures for dwt2d/lec.qmd. Run from the repo root: python3 -I tools/figs/dwt2d/figs.py
# Needs node and esbuild (node_modules/.bin/esbuild): the coefficients and reconstructions are
# computed by data.ts with the demo's own code, so the numbers match the dwt2d demo.
# Image: transmission electron micrograph of a mesenchymal stem cell, Robert M. Hunt, CC BY 3.0,
# via Wikimedia Commons (demos/src/fourier2d/cell256.png).
import os, json, subprocess, tempfile
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "..", "wavelets", "figs.py")).read().split("# ---- haar/fig-spike-step.svg ----")[0])
os.chdir(R)
N = 256
cell = np.array(Image.open("demos/src/fourier2d/cell256.png")).astype(np.float64)
tmp = tempfile.mkdtemp()
cell.tofile(os.path.join(tmp, "cell.f64"))
subprocess.run(["node_modules/.bin/esbuild", os.path.join(HERE, "data.ts"), "--bundle", "--platform=node", "--log-level=warning",
                "--loader:.png=dataurl", "--outfile=" + os.path.join(tmp, "data.js")], check=True)
subprocess.run(["node", os.path.join(tmp, "data.js"), os.path.join(tmp, "cell.f64"), tmp], check=True)
ld = lambda name: np.fromfile(os.path.join(tmp, name + ".f64")).reshape(N, N)
info = json.load(open(os.path.join(tmp, "info.json"))); print(info)
cmap = plt.get_cmap("RdBu_r")
CREDIT = "Image: R. M. Hunt, CC BY 3.0"

def band_rgb(c, levels, rects=None):
    """Mallat layout: LL grey (own range), detail bands diverging, each scaled to its own max |w| (as the demo)."""
    rgb = np.zeros((N, N, 3)); m = N >> levels
    LL = c[:m, :m]; rgb[:m, :m] = ((LL - LL.min()) / (LL.max() - LL.min()))[..., None]
    for j in range(1, levels + 1):
        s = N >> j
        for (y0, x0) in [(0, s), (s, 0), (s, s)]:
            b = c[y0:y0 + s, x0:x0 + s]; mx = np.abs(b).max() or 1
            rgb[y0:y0 + s, x0:x0 + s] = cmap(0.5 + 0.5 * b / mx)[..., :3]
    return rgb

def lines(a, levels):
    for j in range(1, levels + 1):
        s = N >> j; a.plot([s - 0.5, s - 0.5], [-0.5, 2 * s - 0.5], color="#e0a000", lw=1.5); a.plot([-0.5, 2 * s - 0.5], [s - 0.5, s - 0.5], color="#e0a000", lw=1.5)

# ---- dwt2d/fig-separable.png: image -> rows -> columns (one level of Haar) ----
s2 = np.sqrt(0.5)
rows = np.hstack([(cell[:, 0::2] + cell[:, 1::2]) * s2, (cell[:, 0::2] - cell[:, 1::2]) * s2])
full = ld("cell_haar1")
chk = np.vstack([(rows[0::2] + rows[1::2]) * s2, (rows[0::2] - rows[1::2]) * s2]); print("separable check", np.abs(chk - full).max())
fig, ax = plt.subplots(1, 2, figsize=(8.4, 4.6))
rr = np.zeros((N, N, 3)); h = N // 2
L_ = rows[:, :h]; rr[:, :h] = ((L_ - L_.min()) / (L_.max() - L_.min()))[..., None]
Hb = rows[:, h:]; rr[:, h:] = cmap(0.5 + 0.5 * Hb / np.abs(Hb).max())[..., :3]
ax[0].imshow(rr, interpolation="nearest"); ax[0].axvline(h - 0.5, color="#e0a000", lw=2)
ax[0].set_title("1. every row", fontsize=18)
ax[0].text(h / 2, N - 16, "low along x", ha="center", fontsize=16, bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
ax[0].text(1.5 * h, N - 16, "high along x", ha="center", fontsize=16, bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
ax[1].imshow(band_rgb(full, 1), interpolation="nearest"); lines(ax[1], 1); ax[1].set_title("2. then every column", fontsize=18)
for (x, y, s_) in [(h / 2, h - 16, "LL"), (1.5 * h, h - 16, "V"), (h / 2, N - 16, "H"), (1.5 * h, N - 16, "D")]:
    ax[1].text(x, y, s_, ha="center", va="center", fontsize=17, bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
for a in ax: a.axis("off")
fig.patches.append(matplotlib.patches.FancyArrowPatch((0.487, 0.5), (0.517, 0.5), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=22, lw=2, color="k"))
fig.text(0.01, 0.02, CREDIT, fontsize=11, color=GREY)
fig.tight_layout(w_pad=2.2); fig.savefig("dwt2d/fig-separable.png", dpi=110, bbox_inches="tight", pad_inches=0.05); plt.close(fig); print("wrote dwt2d/fig-separable.png")

# ---- dwt2d/fig-block2x2.svg: the four 2x2 Haar patterns and a vertical edge ----
pats = [("LL", [[1, 1], [1, 1]]), ("V", [[1, -1], [1, -1]]), ("H", [[1, 1], [-1, -1]]), ("D", [[1, -1], [-1, 1]])]
fig, ax = plt.subplots(1, 4, figsize=(6.4, 2.3))
for a, (name, p) in zip(ax, pats):
    p = np.array(p)
    for i in range(2):
        for j in range(2):
            a.add_patch(Rectangle((j, 1 - i), 1, 1, fc=cmap(0.5 + 0.35 * p[i, j]), ec="k", lw=1.5))
            a.text(j + 0.5, 1.5 - i, "+" if p[i, j] > 0 else "−", ha="center", va="center", fontsize=22)
    a.set_xlim(-0.05, 2.05); a.set_ylim(-0.05, 2.05); a.set_aspect("equal"); a.axis("off"); a.set_title(name, fontsize=18)
fig.text(0.5, 0.0, "each × ½", ha="center", fontsize=15, color=GREY)
fig.tight_layout(); save(fig, "dwt2d/fig-block2x2.svg")

# ---- dwt2d/fig-pyramid.png: the micrograph and its 4-level db2 pyramid (demo preset "pyramid") ----
c = ld("cell_db2_4")
fig, ax = plt.subplots(1, 2, figsize=(9.6, 5.0))
ax[0].imshow(cell, cmap="gray", vmin=0, vmax=255); ax[0].set_title("micrograph", fontsize=17)
ax[1].imshow(band_rgb(c, 4), interpolation="nearest"); lines(ax[1], 4); ax[1].set_title("db2, 4 levels (Mallat layout)", fontsize=17)
for (x, y, s) in [(192, 60, "V₁"), (60, 192, "H₁"), (192, 192, "D₁"), (96, 30, "V₂"), (30, 96, "H₂"), (96, 96, "D₂")]:
    ax[1].text(x, y, s, ha="center", va="center", fontsize=16, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
for a in ax: a.axis("off")
fig.text(0.01, 0.01, "Image: R. M. Hunt, CC BY 3.0, via Wikimedia Commons", fontsize=11, color=GREY)
fig.tight_layout(); fig.savefig("dwt2d/fig-pyramid.png", dpi=110, bbox_inches="tight", pad_inches=0.05); plt.close(fig); print("wrote dwt2d/fig-pyramid.png")

# ---- dwt2d/fig-keep.png: keep the largest 1 %: Haar, db4, Fourier, 8x8 DCT (crop 128 x 128) ----
y0, x0 = 64, 64
fig, ax = plt.subplots(2, 2, figsize=(5.6, 6.2))
for a, (key, lab) in zip(ax.ravel(), [("haar", "Haar"), ("db4", "db4"), ("fourier", "Fourier"), ("dct", "8×8 DCT")]):
    r = ld("keep_" + key)
    a.imshow(np.clip(r[y0:y0 + 128, x0:x0 + 128], 0, 255), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    a.set_title(f"{lab}: {info['psnr_' + key]:.1f} dB", fontsize=16); a.axis("off")
fig.text(0.02, 0.0, CREDIT + " (centre crop)", fontsize=11, color=GREY)
fig.tight_layout(h_pad=0.6); fig.savefig("dwt2d/fig-keep.png", dpi=110, bbox_inches="tight", pad_inches=0.05); plt.close(fig); print("wrote dwt2d/fig-keep.png")

# ---- dwt2d/fig-shift.png: square, Haar, level-1 V band before and after a 1-px shift ----
sq = ld("square"); c0 = ld("square_c0"); c1 = ld("square_c1")
V0 = c0[:128, 128:]; V1 = c1[:128, 128:]; mx = max(np.abs(V0).max(), np.abs(V1).max())
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.9))
ax[0].imshow(sq, cmap="gray", vmin=0, vmax=255); ax[0].set_title("square", fontsize=17)
cy, cx = slice(25, 101), slice(27, 101)   # zoom on the square's part of the band
ax[1].imshow(cmap(0.5 + 0.5 * V0[cy, cx] / mx), interpolation="nearest"); ax[1].set_title("level-1 V band (zoom)", fontsize=17)
ax[2].imshow(cmap(0.5 + 0.5 * V1[cy, cx] / mx), interpolation="nearest"); ax[2].set_title("… image shifted 1 px", fontsize=17)
for a in ax[1:]:
    a.add_patch(Rectangle((-0.5, -0.5), 74, 76, fill=False, ec="#999999", lw=1))
for a in ax: a.axis("off")
fig.tight_layout(); fig.savefig("dwt2d/fig-shift.png", dpi=110, bbox_inches="tight", pad_inches=0.05); plt.close(fig); print("wrote dwt2d/fig-shift.png")
print("V level-1 |max| left edge cols before/after", np.abs(V0[:, 30:45]).max(), np.abs(V1[:, 30:45]).max(), "right edge", np.abs(V0[:, 80:95]).max(), np.abs(V1[:, 80:95]).max())
