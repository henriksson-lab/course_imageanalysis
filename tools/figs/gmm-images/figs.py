"""Slide figures for gmm-images/lec.qmd.

Run from the repo root:
  python3 -I tools/figs/gmm-images/figs.py --cell128 $TMP/cell128.json   # micrograph, 128 x 128
  (bundle and run data.ts, see there) -> $TMP/gmmimg.json
  python3 -I tools/figs/gmm-images/figs.py $TMP/gmmimg.json
The data file holds the demos' own results (histgmm and colourgmm models), so the numbers in
the figures are those the demos show.
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from skimage import io, segmentation, color

OUT = "gmm-images/"
CELL = "demos/src/fourier2d/cell256.png"
C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]
plt.rcParams.update({"font.family": "sans-serif", "font.size": 16, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})

if sys.argv[1] == "--cell128":
    im = io.imread(CELL).astype(float)
    json.dump(im.reshape(128, 2, 128, 2).mean(axis=(1, 3)).ravel().tolist(), open(sys.argv[2], "w"))
    sys.exit()

D = json.load(open(sys.argv[1]))
g = np.arange(256)


def npdf(x, m, v):
    return np.exp(-0.5 * (x - m) ** 2 / v) / np.sqrt(2 * np.pi * v)


# ---- 1. minimum-error threshold on the micrograph (K = 2) ----
d = D["cellK2"]
cnt = np.array(d["counts"]); n = cnt.sum(); m = d["model"]
fig, ax = plt.subplots(figsize=(6.4, 3.7))
bw = 4
h = cnt.reshape(-1, bw).sum(1) / (n * bw)
ax.bar(np.arange(0, 256, bw) + bw / 2, h, width=bw, color="#ccc", edgecolor="#aaa", lw=0.4)
for k in range(2):
    ax.plot(g, m["weights"][k] * npdf(g, m["means"][k], m["vars"][k]), color=C[k], lw=2.5,
            label=f"π = {m['weights'][k]:.2f}, σ = {np.sqrt(m['vars'][k]):.0f}")
b = d["bounds"][0]; o = d["otsu"][0]
ax.axvline(b, color="#222", lw=2)
ax.axvline(o, color="#222", lw=1.5, ls="--")
top = ax.get_ylim()[1]
box = dict(fc="white", ec="none", alpha=0.85, pad=1.5)
ax.text(b + 3, top * 0.97, f"minimum\nerror {b:.0f}", fontsize=15, va="top", bbox=box)
ax.text(o - 3, top * 0.97, f"Otsu\n{o}", fontsize=15, va="top", ha="right", bbox=box)
ax.set_xlim(0, 255); ax.set_yticks([])
ax.set_xlabel("grey value"); ax.set_ylabel("fraction of pixels")
ax.legend(loc="upper left", fontsize=15, frameon=False)
fig.tight_layout(); fig.savefig(OUT + "fig-minerror.svg"); plt.close(fig)

# ---- 2. image, label image, probability map (synthetic, noise 12) ----
s = D["synth12"]
img = np.array(s["img"]).reshape(128, 128)
lab = np.array(s["label"]).reshape(128, 128)
post = np.array(s["post"]).reshape(128, 128, 3)
labcmap = ListedColormap(["#1f3b73", "#ff7f0e", "#f7e04a"])
fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.6), layout="constrained")
axs[0].imshow(img, cmap="gray", vmin=0, vmax=255); axs[0].set_title("image", fontsize=15)
axs[1].imshow(lab, cmap=labcmap, vmin=0, vmax=2); axs[1].set_title(f"labels ({100 * s['correct']:.1f} % correct)", fontsize=15)
im2 = axs[2].imshow(post[:, :, 1], cmap="viridis", vmin=0, vmax=1); axs[2].set_title("P(cytoplasm)", fontsize=15)
for a in axs: a.set_axis_off()
cb = fig.colorbar(im2, ax=axs, fraction=0.02, pad=0.01); cb.set_ticks([0, 0.5, 1])
fig.savefig(OUT + "fig-labelmaps.png", dpi=150); plt.close(fig)

# ---- 3. speckle: noise 12 against noise 22 ----
fig, axs = plt.subplots(2, 2, figsize=(5.4, 5.6))
for r, key in enumerate(["synth12", "synth22"]):
    s = D[key]
    axs[r, 0].imshow(np.array(s["img"]).reshape(128, 128), cmap="gray", vmin=0, vmax=255)
    axs[r, 0].set_title(f"noise σ = {12 if r == 0 else 22}", fontsize=15)
    axs[r, 1].imshow(np.array(s["label"]).reshape(128, 128), cmap=labcmap, vmin=0, vmax=2)
    axs[r, 1].set_title(f"labels: {100 * s['correct']:.1f} % correct", fontsize=15)
    for a in axs[r]: a.set_axis_off()
fig.tight_layout(); fig.savefig(OUT + "fig-speckle.png", dpi=150); plt.close(fig)

# ---- 4. colour scene: k-means splits the shadowed ball, a full GMM keeps it ----
s = D["scene"]
rgb = np.array(s["rgb"]).reshape(128, 128, 3)
truth = np.array(s["truth"])


def class_colours(lab):
    """Paint every cluster with the mean colour of its pixels (so the result reads like the image)."""
    lab = np.array(lab)
    out = np.zeros((128 * 128, 3))
    flat = rgb.reshape(-1, 3)
    for k in np.unique(lab):
        out[lab == k] = flat[lab == k].mean(0)
    return out.reshape(128, 128, 3), lab


fig = plt.figure(figsize=(5.6, 5.4))
ax0 = fig.add_axes([0.02, 0.53, 0.44, 0.41]); ax0.imshow(rgb); ax0.set_title("image", fontsize=15)
km, kml = class_colours(s["km"]); gm, gml = class_colours(s["gm"])
ax1 = fig.add_axes([0.02, 0.02, 0.44, 0.41]); ax1.imshow(km); ax1.set_title("k-means", fontsize=15)
ax2 = fig.add_axes([0.54, 0.02, 0.44, 0.41]); ax2.imshow(gm); ax2.set_title("Gaussian mixture", fontsize=15)
for a in (ax0, ax1, ax2):
    a.set_axis_off()
for a, l in ((ax1, kml), (ax2, gml)):
    bd = segmentation.find_boundaries(l.reshape(128, 128), mode="inner")
    ov = np.zeros((128, 128, 4)); ov[bd] = (1, 1, 1, 1)
    a.imshow(ov)
# R-G projection with the fitted GMM ellipses and the k-means centres
ax3 = fig.add_axes([0.63, 0.61, 0.31, 0.33])
X = 100 * rgb.reshape(-1, 3)
ax3.scatter(X[:, 0], X[:, 1], s=1, c=rgb.reshape(-1, 3), alpha=0.5, rasterized=True)
mu = np.array(s["gmMeans"]).reshape(4, 3); cov = np.array(s["gmCovs"]).reshape(4, 3, 3)
for k in range(4):
    S2 = cov[k][:2, :2]; w, v = np.linalg.eigh(S2)
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    ax3.add_patch(Ellipse(mu[k, :2], 2 * 2 * np.sqrt(w[1]), 2 * 2 * np.sqrt(w[0]), angle=ang, fill=False, lw=2, color="#222"))
cen = np.array(s["kmCentres"]).reshape(4, 3)
ax3.scatter(cen[:, 0], cen[:, 1], marker="X", s=120, c="white", edgecolors="#222", linewidths=1.5, zorder=5)
ax3.set_xlim(0, 100); ax3.set_ylim(0, 100); ax3.set_xlabel("R"); ax3.set_ylabel("G")
ax3.set_title("pixels in R–G", fontsize=15)
fig.savefig(OUT + "fig-colour.png", dpi=150); plt.close(fig)
print("scene: k-means Dice", np.round(s["diceKm"], 2), "GMM Dice", np.round(s["diceGm"], 2))

# ---- 5. superpixels on the micrograph ----
cell = io.imread(CELL)
seg = segmentation.slic(cell, n_segments=250, compactness=0.3, channel_axis=None, start_label=1)
meanimg = color.label2rgb(seg, np.dstack([cell] * 3), kind="avg", bg_label=0)
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.95))
axs[0].imshow(segmentation.mark_boundaries(np.dstack([cell] * 3) / 255, seg, color=(1, 0.5, 0.05)))
axs[0].set_title(f"{seg.max()} superpixels", fontsize=15)
axs[1].imshow(meanimg.astype(np.uint8)); axs[1].set_title("mean of each", fontsize=15)
for a in axs: a.set_axis_off()
fig.tight_layout(); fig.savefig(OUT + "fig-superpixels.png", dpi=150); plt.close(fig)
print("superpixels:", seg.max())

# ---- 6. DNA content: G1 peak, S plateau, G2 peak (synthetic) ----
rs = np.random.default_rng(4)
n = 3000; w = np.array([0.55, 0.20, 0.25]); g1, cv = 100.0, 0.06
z = rs.choice(3, size=n, p=w)
x = np.where(z == 0, rs.normal(g1, cv * g1, n),
             np.where(z == 2, rs.normal(2 * g1, cv * 2 * g1, n),
                      rs.uniform(g1, 2 * g1, n) + rs.normal(0, cv * 1.5 * g1, n)))
xs = np.linspace(40, 280, 600)
fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.hist(x, bins=np.arange(40, 281, 4), density=True, color="#ccc", edgecolor="#aaa", lw=0.4)
p1 = w[0] * npdf(xs, g1, (cv * g1) ** 2)
p3 = w[2] * npdf(xs, 2 * g1, (cv * 2 * g1) ** 2)
# S phase: uniform on [100, 200] blurred by the measurement noise
from scipy.stats import norm
sd = cv * 1.5 * g1
p2 = w[1] * (norm.cdf((xs - g1) / sd) - norm.cdf((xs - 2 * g1) / sd)) / g1
ax.plot(xs, p1, color=C[0], lw=2.5, label="G1 (2N): 55 %")
ax.plot(xs, p2, color=C[1], lw=2.5, label="S: 20 %")
ax.plot(xs, p3, color=C[2], lw=2.5, label="G2 (4N): 25 %")
ax.plot(xs, p1 + p2 + p3, color="#222", lw=1.5, ls="--", label="mixture")
ax.set_xlim(40, 280); ax.set_yticks([])
ax.set_xticks([100, 200]); ax.set_xticklabels(["100\n(2N)", "200\n(4N)"])
ax.set_xlabel("integrated nuclear intensity"); ax.set_ylabel("fraction of cells")
ax.legend(frameon=False, fontsize=15)
fig.tight_layout(); fig.savefig(OUT + "fig-dna.svg"); plt.close(fig)

# ---- 7. anomalies: d_M > 3 to every component ----
rs = np.random.default_rng(11)
mus = [np.array([2.0, 2.0]), np.array([6.0, 4.5])]
covs = [np.array([[1.0, 0.5], [0.5, 0.8]]), np.array([[1.4, -0.6], [-0.6, 0.7]])]
P = np.vstack([rs.multivariate_normal(mus[0], covs[0], 300), rs.multivariate_normal(mus[1], covs[1], 200)])
odd = np.array([[4.6, -0.2], [0.0, 6.0], [9.4, 1.6], [2.4, 6.8]])
A = np.vstack([P, odd])
# fit each component from its own points (labels known for this picture)
fm = [P[:300].mean(0), P[300:].mean(0)]
fc = [np.cov(P[:300].T, bias=True), np.cov(P[300:].T, bias=True)]
dm = np.stack([np.sqrt(np.einsum("ij,jk,ik->i", A - m_, np.linalg.inv(c_), A - m_)) for m_, c_ in zip(fm, fc)], 1)
flag = (dm > 3).all(1)
fig, ax = plt.subplots(figsize=(5.4, 4.4))
ax.scatter(A[~flag, 0], A[~flag, 1], s=12, color="#888")
ax.scatter(A[flag, 0], A[flag, 1], s=90, marker="X", color="#d62728", zorder=5, label=f"$d_M > 3$ to both: {flag[-4:].sum()} odd + {flag[:-4].sum()} normal")
for k, (m_, c_) in enumerate(zip(fm, fc)):
    wv, v = np.linalg.eigh(c_)
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    for r in (1, 3):
        ax.add_patch(Ellipse(m_, 2 * r * np.sqrt(wv[1]), 2 * r * np.sqrt(wv[0]), angle=ang, fill=False,
                             lw=2 if r == 3 else 1.2, ls="-" if r == 3 else ":", color=C[k]))
ax.set_aspect("equal"); ax.set_xlim(-2, 10.5); ax.set_ylim(-1.5, 11.5)
ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
from matplotlib.lines import Line2D
h, l = ax.get_legend_handles_labels()
h += [Line2D([], [], color="#555", lw=1.2, ls=":"), Line2D([], [], color="#555", lw=2)]
l += ["$d_M = 1$", "$d_M = 3$"]
ax.legend(h, l, loc="upper left", frameon=False, fontsize=15, ncol=1)
fig.tight_layout(); fig.savefig(OUT + "fig-anomaly.svg"); plt.close(fig)
print("anomaly flags:", flag.sum(), "of which odd:", flag[-4:].sum())
