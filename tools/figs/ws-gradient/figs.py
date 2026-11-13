# Figures for ws-gradient/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/ws-gradient/figs.py
# Images and segmentations come from the watershed demo's own pipeline (data.ts, via node).
# Micrograph: demos/src/fourier2d/cell256.png (Robert M. Hunt, CC BY 3.0, via Wikimedia Commons).
import os, sys, json, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ws-landscape"))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from wsstyle import *

setup()
OUT = "ws-gradient"
cell = mpimg.imread("demos/src/fourier2d/cell256.png")
if cell.ndim == 3: cell = cell[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
tmp = os.path.join(tempfile.mkdtemp(), "cell.json")
json.dump([float(v) for v in (cell * 255).ravel()], open(tmp, "w"))
D = run_ts(HERE, args=(tmp,))
n = D["n"]
A = lambda k: np.array(k).reshape(n, n)
img = A(D["image"])
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--numbers":
    print(D["raw"]["n"], D["smooth"]["n"], D["hsweep"]); sys.exit()


def overlay(ax, base, labels, colour=(1.0, 0.85, 0.0), lw=1, vmax=255):
    rgb = np.repeat((np.clip(base, 0, vmax) / vmax)[..., None], 3, axis=2)
    b = boundary(labels)
    if lw > 1:
        from scipy.ndimage import binary_dilation
        b = binary_dilation(b, iterations=lw - 1)
    rgb[b] = colour
    ax.imshow(rgb, interpolation="nearest"); ax.axis("off")


# --- 1. the image and its gradient magnitude ---------------------------------------------------
g = A(D["grad15"])
f, axs = plt.subplots(1, 2, figsize=(560 / 72, 300 / 72))
axs[0].imshow(img, cmap="gray", vmin=0, vmax=255); axs[0].set_title("test image $I$", fontsize=18)
axs[1].imshow(g, cmap="gray_r", vmin=0, vmax=np.percentile(g, 99.5)); axs[1].set_title(r"$\|\nabla(G_{1.5} * I)\|$ (dark = high)", fontsize=17)
for a in axs: a.axis("off")
f.subplots_adjust(wspace=0.05)
save_png(f, f"{OUT}/fig-gradient.png")

# --- 2. smoothing alone: sigma 1 vs 4 -----------------------------------------------------------
f, axs = plt.subplots(1, 2, figsize=(560 / 72, 300 / 72))
for ax, key, s in [(axs[0], "raw", 1), (axs[1], "smooth", 4)]:
    overlay(ax, img, A(D[key]["labels"]))
    ax.set_title(f"σ = {s}: {D[key]['n']} regions", fontsize=18)
f.subplots_adjust(wspace=0.05)
save_png(f, f"{OUT}/fig-smooth.png")

# --- 3. number of regions against h --------------------------------------------------------------
hs = np.array(D["hsweep"])
f, ax = plt.subplots(figsize=(480 / 72, 330 / 72))
ax.axvspan(3, 6, color="#e8eef8")
ax.semilogy(hs[:, 0], hs[:, 1], "o-", color=BLUE, ms=5, lw=2)
ax.text(4.5, 300, "plateau:\n8 regions", ha="center", fontsize=17)
ax.set_xlabel("$h$ (grey levels per pixel)"); ax.set_ylabel("regions")
ax.set_yticks([10, 100, 1000]); ax.set_yticklabels(["10", "100", "1000"])
ax.set_xlim(-0.2, 8.2)
save(f, f"{OUT}/fig-hsweep.svg")

# --- 4. region merging: a 1D profile and its merge tree --------------------------------------------
v = np.array([6, 2, 4, 3, 5, 9, 7, 1, 3, 2, 8, 4, 6, 5, 7.5])
# basins of the regional minima, merged across the lowest pass first
mins = [i for i in range(1, len(v) - 1) if v[i] < v[i - 1] and v[i] < v[i + 1]]
passes = []
for a, b in zip(mins[:-1], mins[1:]):
    k = a + int(np.argmax(v[a:b + 1])); passes.append((v[k], k))
f, (ax, at) = plt.subplots(2, 1, figsize=(560 / 72, 400 / 72), sharex=True, gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.08))
ax.bar(np.arange(len(v)), v, width=1.0, color=LAND)
for k, m in enumerate(mins):
    ax.plot(m, v[m] + 0.35, "^", ms=11, mfc="white", mec=INK, mew=1.6)
    ax.text(m, -0.6, "abcdef"[k], ha="center", va="top", fontsize=18, fontweight="bold")
for h, k in passes:
    ax.text(k, h + 0.2, f"{h:g}", ha="center", va="bottom", fontsize=16, color=DAM)
ax.set_ylim(-2.0, 11); ax.axis("off")
# merge tree: leaves at the minima, joins at pass heights (lowest first)
groups = {m: (m, v[m]) for m in mins}           # root -> (x position, height)
order = sorted(range(len(passes)), key=lambda i: passes[i][0])
parent = {m: m for m in mins}
def root(m):
    while parent[m] != m: m = parent[m]
    return m
for i in order:
    h, k = passes[i]
    ra, rb = root(mins[i]), root(mins[i + 1])
    (xa, ya), (xb, yb) = groups[ra], groups[rb]
    at.plot([xa, xa, xb, xb], [ya, h, h, yb], color=INK, lw=2)
    parent[rb] = ra; groups[ra] = ((xa + xb) / 2, h)
for m in mins: at.plot(m, v[m], "^", ms=11, mfc="white", mec=INK, mew=1.6)
cut = 6.5
ncut = len(mins) - sum(1 for h, _ in passes if h < cut)
at.axhline(cut, color=ORANGE, ls="--", lw=2)
at.text(-0.4, cut + 0.2, f"cut at {cut:g}: {ncut} regions", ha="left", va="bottom", color=ORANGE, fontsize=17)
at.set_ylim(10, 0.3); at.set_ylim(0.3, 10); at.set_ylabel("dam height"); at.set_xticks([])
at.spines["bottom"].set_visible(False)
save(f, f"{OUT}/fig-merge.svg")
print("passes", passes, "minima", mins)

# --- 5. leakage on the micrograph ------------------------------------------------------------------
lab = A(D["cell"]["labels"])
f, axs = plt.subplots(1, 2, figsize=(560 / 72, 300 / 72))
axs[0].imshow(cell * 255, cmap="gray", vmin=0, vmax=255); axs[0].axis("off"); axs[0].set_title("micrograph", fontsize=18)
overlay(axs[1], cell * 255, (lab == 2).astype(int), colour=(1.0, 0.85, 0.0), lw=2)
for c in D["cell"]["clicks"]:
    axs[1].plot(c["x"], c["y"], "o", ms=7, mfc=(ORANGE if c["label"] == 2 else "#56B4E9"), mec="black")
axs[1].set_title("1 + 6 markers", fontsize=18)
f.subplots_adjust(wspace=0.05)
save_png(f, f"{OUT}/fig-leak-cell.png")
print("ws-gradient figures written")
