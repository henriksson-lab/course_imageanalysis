# Static figures for stain-unmixing/lec.qmd. Run from the repository root:
#   node_modules/.bin/esbuild tools/figs/stain-unmixing/export.ts --bundle --platform=node --outfile=<tmp>/x.js
#   node <tmp>/x.js <tmp>
#   python3 -I tools/figs/stain-unmixing/figs.py <tmp>
# The slide, its concentration maps and Macenko's vectors come from the stainsep demo's model
# (synthetic H&E, Beer–Lambert, seeded); the optical-density example uses the slide's numbers.
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.size": 17, "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})
H_C, E_C = "#3b2f8f", "#d6337f"                    # as in the stainsep demo
mapH = LinearSegmentedColormap.from_list("H", ["#ffffff", "#a99fd9", "#3b2f8f", "#120a40"])
mapE = LinearSegmentedColormap.from_list("E", ["#ffffff", "#f3a6c8", "#d6337f", "#6b0a37"])
OUT = "stain-unmixing/"

def save(fig, name, **kw):
    fig.savefig(OUT + name, **kw)
    if os.environ.get("PREVIEW"):
        fig.savefig(os.path.join(os.environ["PREVIEW"], name.rsplit(".", 1)[0] + ".png"), dpi=96, bbox_inches=kw.get("bbox_inches"))
    plt.close(fig); print("wrote", OUT + name)

D = json.load(open(sys.argv[1] + "/stain.json")); N = D["N"]
rgb = np.stack([np.array(D[c]).reshape(N, N) for c in "RGB"], -1) / 255
mp = lambda key, k: np.array(D[key][k]).reshape(N, N)

# 1. fig-he.png: the synthetic H&E slide
fig, ax = plt.subplots(figsize=(3.4, 3.4)); ax.imshow(rgb, interpolation="nearest"); ax.axis("off")
fig.tight_layout(pad=0.1); save(fig, "fig-he.png", dpi=150)

# 2. fig-odmix.svg: 0.4 units of hematoxylin, 0.3 of eosin, and both: OD adds, intensity multiplies
h = 0.4 * np.array([0.65, 0.70, 0.29]); e = 0.3 * np.array([0.07, 0.99, 0.11])
fig, ax = plt.subplots(figsize=(4.9, 3.0)); ax.axis("off"); ax.set_xlim(0, 3.15); ax.set_ylim(-1.45, 1.15)
for i, (od, lab) in enumerate(((h, "0.4 H"), (e, "0.3 E"), (h + e, "both"))):
    I = 255 * 10 ** -od
    ax.add_patch(Rectangle((i * 1.05 + 0.1, 0), 0.85, 0.85, color=np.clip(I / 255, 0, 1), ec="#666"))
    ax.text(i * 1.05 + 0.525, 0.95, lab, ha="center", va="bottom", fontsize=18)
    ax.text(i * 1.05 + 0.525, -0.12, f"OD\n({od[0]:.2f}, {od[1]:.2f})", ha="center", va="top", fontsize=15, linespacing=1.1)
    ax.text(i * 1.05 + 0.525, -0.78, f"I ({I[0]:.0f}, {I[1]:.0f})", ha="center", va="top", fontsize=15)
    print(lab, "OD", od.round(3), "I", I.round(1))
ax.text(1.05 * 1.5 + 0.03, -1.15, "(red, green)", ha="center", va="top", fontsize=15, color="#555")
fig.tight_layout(pad=0.1); save(fig, "fig-odmix.svg")

# 3. fig-deconv.png: the slide and its two concentration maps (the slide's own stain vectors)
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.6))
axs[0].imshow(rgb, interpolation="nearest"); axs[0].set_title("RGB", fontsize=17)
axs[1].imshow(mp("tru", 0), cmap=mapH, vmin=0, vmax=1.2, interpolation="nearest"); axs[1].set_title("H", fontsize=17, color=H_C)
axs[2].imshow(mp("tru", 1), cmap=mapE, vmin=0, vmax=1.2, interpolation="nearest"); axs[2].set_title("E", fontsize=17, color=E_C)
for ax in axs: ax.axis("off")
fig.tight_layout(pad=0.2); save(fig, "fig-deconv.png", dpi=150, bbox_inches="tight")

# 4. fig-wrong.png: the H map with Ruifrok's published vectors and with the slide's true vectors
fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.9))
for ax, key, lab in zip(axs, ("ruif", "tru"), ("published vectors", "true vectors")):
    ax.imshow(mp(key, 0), cmap=mapH, vmin=0, vmax=1.2, interpolation="nearest"); ax.set_title(lab, fontsize=17); ax.axis("off")
fig.tight_layout(pad=0.2); save(fig, "fig-wrong.png", dpi=150, bbox_inches="tight")

# 5. fig-macenko.svg: OD pixels in their plane; the extreme angles are the stain vectors
od = np.stack([np.array(D["od"][i]) for i in range(3)], 1)
keep = np.linalg.norm(od, axis=1) > 0.15; X = od[keep]
w, V = np.linalg.eigh(X.T @ X); e1, e2 = V[:, -1], V[:, -2]
if e1.sum() < 0: e1 = -e1
x, y = X @ e1, X @ e2; ang = np.arctan2(y, x)
lo, hi = np.percentile(ang, 1), np.percentile(ang, 99)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
sel = np.random.default_rng(1).choice(len(x), min(2500, len(x)), replace=False)
ax.plot(x[sel], y[sel], ".", color="#888", ms=2.5, alpha=0.4)
R = np.max(np.hypot(x, y)) * 1.05
for t in (lo, hi):
    ax.plot([0, R * np.cos(t)], [0, R * np.sin(t)], color="black", lw=2)
for v, col, lab in zip(D["trueV"][:2], (H_C, E_C), ("H", "E")):
    v = np.array(v); px, py = v @ e1, v @ e2; t = np.arctan2(py, px)
    ax.plot([0, R * np.cos(t)], [0, R * np.sin(t)], color=col, lw=2, ls="--")
    ax.text(R * 1.03 * np.cos(t), R * 1.03 * np.sin(t), lab, color=col, fontsize=18, ha="left", va="center")
a = np.linspace(lo, hi, 50); ax.plot(0.25 * R * np.cos(a), 0.25 * R * np.sin(a), color="#444", lw=1)
ax.set_aspect("equal"); ax.set_xlabel("OD along $\\mathbf{v}_1$"); ax.set_ylabel("OD along $\\mathbf{v}_2$")
ax.plot(0, 0, "k.", ms=6); yl = ax.get_ylim(); ax.set_ylim(yl[0] - 0.06, yl[1] + 0.03)
fig.tight_layout(); save(fig, "fig-macenko.svg")
