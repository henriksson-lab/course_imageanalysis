# Static figures for volrender/lec.qmd. Run from the repo root (after render.ts, see there):
#   python3 -I tools/figs/volrender/figs.py $TMP/data
# The phantom and the renders come from demos/src/lib/volrender.ts (headPhantom, castImage).
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
D = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join(R, "volrender")
plt.rcParams.update({"font.size": 16, "svg.fonttype": "path", "font.family": "DejaVu Sans", "mathtext.fontset": "cm"})  # path: the script T must not depend on browser fonts
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#888888"
BG = np.array([0.08, 0.09, 0.12])  # the demo's background


def save(fig, name, dpi=None):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", **({"dpi": dpi} if dpi else {}))
    plt.close(fig)


def clean(ax, label=None, fs=16):
    ax.set_xticks([]); ax.set_yticks([])
    if label:
        ax.set_xlabel(label, fontsize=fs)
    for s in ax.spines.values():
        s.set_visible(False)


ph = json.load(open(os.path.join(D, "phantom.json")))
vol = np.array(ph["data"]).reshape(ph["nz"], ph["ny"], ph["nx"])


def load(name):
    d = json.load(open(os.path.join(D, name + ".json")))
    return np.array(d["img"], float).reshape(d["size"], d["size"], 4), d


def grey_img(a, gain=1.0):
    v = np.clip(a[..., 0] * gain / 255, 0, 1)
    out = np.repeat(v[..., None], 3, -1)
    out[a[..., 0] < 0] = BG
    return out


def dvr_img(a):
    C, T = a[..., :3], a[..., 3:4]
    out = C + T * BG
    out[a[..., 0] < 0] = BG
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------- 1. a sagittal slice: as stored vs with the spacing
sag = vol[:, :, 32].T  # rows y (down), columns z (voxel index)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [40, 64]})
ax = axs[0]
ax.imshow(sag, cmap="gray", vmin=0, vmax=255, extent=[0, 40, 64, 0], interpolation="nearest")
clean(ax, "as stored", fs=26)
ax = axs[1]
ax.imshow(sag, cmap="gray", vmin=0, vmax=255, extent=[0, 40 * 1.6, 64, 0], interpolation="nearest")
clean(ax, "with 1.6 mm in z", fs=26)
fig.tight_layout()
save(fig, "fig-slices.png", dpi=90)

# ---------------------------------------------------------------- 2. MIP, MinIP, average
fig, axs = plt.subplots(1, 3, figsize=(12, 4.4))
for ax, (nm, lab, gain) in zip(axs, [("vr-mip", "MIP", 1), ("vr-minip", "MinIP", 1), ("vr-avg", "average", 2)]):
    a, _ = load(nm)
    ax.imshow(grey_img(a, gain)[8:-8, 8:-8])
    clean(ax, lab, fs=24)
fig.tight_layout()
save(fig, "fig-projections.png", dpi=90)

# ---------------------------------------------------------------- 3. transfer function over the histogram + render
tfs = json.load(open(os.path.join(D, "tfs.json")))
soft = np.array(tfs["soft"]["points"])
fig = plt.figure(figsize=(9.5, 3.6))
ax = fig.add_axes([0.09, 0.2, 0.55, 0.72])
h, edges = np.histogram(vol.ravel(), bins=128, range=(0, 255))
ax.bar(edges[:-1], np.log10(1 + h) / np.log10(1 + h.max()), width=2, color="#cccccc", align="edge")
vs = np.linspace(0, 255, 512)
ax.plot(vs, np.interp(vs, soft[:, 0], soft[:, 1]), color=RED, lw=3)
ax.plot(soft[:, 0], soft[:, 1], "o", color=RED, ms=6)
for v, lab in [(70, "soft"), (100, "brain"), (170, "bone"), (230, "vessel")]:
    ax.text(v, 1.05, lab, ha="center", fontsize=17, color="#444")
ax.text(5, 0.55, "histogram\n(log)", fontsize=17, color="#888")
ax.text(150, 0.45, "opacity", fontsize=18, color=RED)
ax.set_xlim(0, 255); ax.set_ylim(0, 1.15)
ax.set_xlabel("value v", fontsize=18); ax.set_yticks([0, 0.5, 1])
ax.tick_params(labelsize=16)
ax.spines[["top", "right"]].set_visible(False)
ax2 = fig.add_axes([0.66, 0.0, 0.34, 1.0])
a, _ = load("vr-soft")
ax2.imshow(dvr_img(a)[10:-10, 10:-10]); clean(ax2)
save(fig, "fig-tf.png", dpi=110)

# ---------------------------------------------------------------- 4. light along a ray: fog, blob, wall and T(t)
t = np.linspace(0, 10, 1000)
tau = 0.06 + 0.9 * np.exp(-((t - 3.5) / 0.5) ** 2) + 8.0 * ((t > 6.5) & (t < 7.1))
T = np.exp(-np.cumsum(tau) * (t[1] - t[0]))
fig, axs = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
ax = axs[0]
ax.imshow(tau[None, :], cmap="Greys", aspect="auto", extent=[0, 10, 0, 1], vmin=0, vmax=3)
ax.set_yticks([]); ax.text(-0.15, 0.5, "eye", ha="right", va="center", color=RED, fontsize=16)
ax.text(1.5, 0.5, "thin fog", ha="center", va="center", fontsize=14)
ax.text(3.5, 1.15, "dense blob", ha="center", fontsize=14)
ax.text(6.8, 1.15, "opaque wall", ha="center", fontsize=14)
ax.text(8.6, 0.5, "hidden", ha="center", va="center", fontsize=14, color="#777")
ax.set_ylim(0, 1)
for s in ax.spines.values():
    s.set_visible(False)
ax = axs[1]
ax.plot(t, T, color=BLUE, lw=3)
ax.set_ylabel(r"$\mathcal{T}(t)$", fontsize=20)
ax.set_xlabel("distance t from the eye along the ray")
ax.set_ylim(0, 1.05); ax.set_xlim(0, 10)
ax.spines[["top", "right"]].set_visible(False)
print("T before wall", round(float(T[t < 6.5][-1]), 3), "after", round(float(T[-1]), 5))
save(fig, "fig-transmittance.svg")

# ---------------------------------------------------------------- 5. the 5-sample example
vals = [30, 80, 180, 230, 90]
al = np.array([0, 0.2, 0.5, 0.9, 0.2])
Ts = np.concatenate([[1], np.cumprod(1 - al)])
contrib = Ts[:-1] * al
print("T:", np.round(Ts, 3), "contributions T*alpha:", np.round(contrib, 3), "sum", contrib.sum())
fig, ax = plt.subplots(figsize=(6.4, 4.8))
x = np.arange(1, 6)
ax.bar(x - 0.2, al, width=0.38, color=RED, alpha=0.8, label=r"opacity $\alpha$ of the sample")
ax.bar(x + 0.2, contrib, width=0.38, color=GREEN, label=r"its share $\mathcal{T}\,\alpha$ in the pixel")
ax.step(np.arange(0.5, 6.5, 1), np.concatenate([Ts, [Ts[-1]]])[:6], where="post", color=BLUE, lw=3, label=r"$\mathcal{T}$ in front of the sample")
ax.set_xticks(x); ax.set_xticklabels([f"{i}\nv = {v}" for i, v in zip(x, vals)], fontsize=15); ax.set_xlabel("sample, front to back")
ax.set_ylim(0, 1.05); ax.set_xlim(0.4, 5.6)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=16)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig-composite.svg")

# ---------------------------------------------------------------- 6. step too large: wood grain
fig, axs = plt.subplots(2, 1, figsize=(4.4, 9.0))
for ax, (nm, lab) in zip(axs, [("vr-wood", "step 3: rings"), ("vr-wood-s0.5", "step 0.5")]):
    a, _ = load(nm)
    ax.imshow(dvr_img(a)[28:-22, 28:-28]); clean(ax, lab, fs=26)
fig.tight_layout()
save(fig, "fig-woodgrain.png", dpi=90)

# ---------------------------------------------------------------- 7. isosurface crossing on a ray
s = np.arange(0, 9)
v = np.array([20, 30, 45, 70, 100, 150, 170, 168, 160], float)
c = 135
i = np.argmax(v >= c)
hit = i - 1 + (c - v[i - 1]) / (v[i] - v[i - 1])
print("iso hit at sample", round(hit, 3))
fig, ax = plt.subplots(figsize=(5.8, 4.2))
ax.plot(s, v, "o-", color=BLUE, lw=2, ms=8, label="samples along the ray")
ax.axhline(c, color=ORANGE, lw=2, ls="--")
ax.text(0.1, c + 6, f"isovalue c = {c}", color=ORANGE, fontsize=15)
ax.plot([hit], [c], "*", color=RED, ms=20, zorder=5)
ax.annotate(f"crossing at {hit:.2f}\n(linear interpolation)", xy=(hit, c), xytext=(hit + 0.3, 40), fontsize=16, arrowprops=dict(arrowstyle="->", color="#444"))
ax.set_xlabel("sample number along the ray"); ax.set_ylabel("value")
ax.set_ylim(0, 190)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig-isocross.svg")
