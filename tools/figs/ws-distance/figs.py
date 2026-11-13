# Figures for ws-distance/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/ws-distance/figs.py
# Masks, distance transforms and segmentations come from the distws demo's own code
# (data.ts, via node), with the demo's presets.
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ws-landscape"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from wsstyle import *

setup()
OUT = "ws-distance"
D = run_ts(HERE)
n = D["n"]
A = lambda k: np.array(k, float).reshape(n, n)
AI = lambda k: np.array(k, int).reshape(n, n)

# --- 1. a touching pair: no dip in intensity, but a neck in the outline -------------------------
nuc = D["nuc"]; img = A(nuc["image"]); E = nuc["ellipses"]
a, b = [E[i - 1] for i in nuc["pairs"][0]]
p0, p1 = np.array([a["x"], a["y"]]), np.array([b["x"], b["y"]])
u = (p1 - p0) / np.linalg.norm(p1 - p0)
s0, s1 = p0 - 22 * u, p1 + 22 * u
L = np.linalg.norm(s1 - s0); t = np.linspace(0, L, 200)
pts = s0[None, :] + t[:, None] * u[None, :]
prof = map_coordinates(img, [pts[:, 1], pts[:, 0]], order=1)
cx, cy = (p0 + p1) / 2; R = 40
x0, y0 = int(max(0, cx - R)), int(max(0, cy - R))
f = plt.figure(figsize=(500 / 72, 250 / 72))
ax = f.add_axes([0.0, 0.05, 0.36, 0.9])
crop = img[y0:y0 + 2 * R, x0:x0 + 2 * R]
ax.imshow(crop, cmap="gray", vmin=0, vmax=255, extent=(x0 - 0.5, x0 + 2 * R - 0.5, y0 + 2 * R - 0.5, y0 - 0.5))
ax.contour(np.arange(x0, x0 + 2 * R), np.arange(y0, y0 + 2 * R), (crop > 115).astype(float), levels=[0.5], colors=[ORANGE], linewidths=2)
ax.plot([s0[0], s1[0]], [s0[1], s1[1]], "--", color="#56B4E9", lw=2.5)
ax.axis("off")
ax2 = f.add_axes([0.55, 0.22, 0.44, 0.7])
ax2.plot(t, prof, color="#0072B2", lw=2)
ax2.axhline(115, color=ORANGE, ls="--", lw=2); ax2.text(L, 120, "threshold 115", ha="right", va="bottom", color=ORANGE, fontsize=16)
ax2.set_xlabel("position (pixels)"); ax2.set_ylabel("grey level"); ax2.set_ylim(0, 255)
save_png(f, f"{OUT}/fig-pair.png")

# --- 2. the distance transform of two touching discs ---------------------------------------------
M = D["metrics"]["euclidean"]; dt = A(M["dt"]); mask = AI(D["twoMask"])
f, ax = plt.subplots(figsize=(440 / 72, 300 / 72))
sl = (slice(45, 136), slice(25, 156))
im = ax.imshow(np.where(mask, dt, np.nan)[sl], cmap="viridis", extent=(24.5, 155.5, 135.5, 44.5))
ax.contour(np.arange(25, 156), np.arange(45, 136), dt[sl], levels=np.arange(4, 30, 4), colors="white", linewidths=0.9)
ax.contour(np.arange(25, 156), np.arange(45, 136), mask[sl], levels=[0.5], colors=INK, linewidths=1.5)
for pk in M["peaks"]: ax.plot(pk["x"], pk["y"], "o", ms=8, mfc="white", mec=INK)
ax.plot(90, 90, "o", ms=8, mfc=RED, mec=INK)
ax.text(90, 82, f"saddle {M['neck']:.0f}", ha="center", fontsize=16, color="white")
ax.text(68, 104, f"peak {M['p1']:.0f}", ha="center", fontsize=16, color="white")
ax.text(112, 104, f"peak {M['p2']:.0f}", ha="center", fontsize=16, color="white")
ax.axis("off")
cb = f.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("distance")
save_png(f, f"{OUT}/fig-dt.png")

# --- 3. labelling vs watershed of -DT on the nuclei image -----------------------------------------
lab = AI(nuc["labels"]); msk = AI(nuc["mask"])
from skimage.measure import label as cclabel
cc = cclabel(msk, connectivity=2)
f, axs = plt.subplots(1, 2, figsize=(560 / 72, 300 / 72))
axs[0].imshow(label_rgb(cc, bg=(0.15, 0.15, 0.15))); axs[0].set_title(f"labelling: {nuc['comps']}", fontsize=18)
rgb = label_rgb(lab, seed=5, bg=(0.15, 0.15, 0.15)); rgb[boundary(lab) & (msk > 0)] = (0.15, 0.15, 0.15)
axs[1].imshow(rgb); axs[1].set_title(f"watershed of −DT: {nuc['n']}", fontsize=18)
for pk in nuc["peaks"]: axs[1].plot(pk["x"], pk["y"], "o", ms=5, mfc="white", mec=INK)
for a_ in axs: a_.axis("off")
f.subplots_adjust(wspace=0.05)
save_png(f, f"{OUT}/fig-nuclei-ws.png")

# --- 4. the four metrics on the two discs ----------------------------------------------------------
names = [("euclidean", "Euclidean"), ("cityblock", "city-block"), ("3-4", "chamfer 3-4"), ("chessboard", "chessboard")]
f, axs = plt.subplots(2, 2, figsize=(520 / 72, 360 / 72))
for ax, (k, nm) in zip(axs.ravel(), names):
    m = D["metrics"][k]; d = A(m["dt"])
    ax.imshow(np.where(mask, d, np.nan)[sl], cmap="viridis", vmin=0, vmax=31, extent=(24.5, 155.5, 135.5, 44.5))
    ax.contour(np.arange(25, 156), np.arange(45, 136), d[sl], levels=np.arange(4, 32, 4), colors="white", linewidths=0.8)
    lb = AI(m["labels"]); cut = np.zeros(lb.shape, bool)
    cut[:, :-1] |= (lb[:, :-1] > 0) & (lb[:, 1:] > 0) & (lb[:, :-1] != lb[:, 1:])
    cut[:-1, :] |= (lb[:-1, :] > 0) & (lb[1:, :] > 0) & (lb[:-1, :] != lb[1:, :])
    from scipy.ndimage import binary_dilation
    cut = binary_dilation(cut)
    ov = np.zeros(lb.shape + (4,)); ov[cut] = (0.84, 0.15, 0.16, 1)
    ax.imshow(ov[sl], extent=(24.5, 155.5, 135.5, 44.5), interpolation="nearest")
    ax.set_title(f"{nm}: {m['n']} object{'s' if m['n'] > 1 else ''}", fontsize=17, pad=3)
    ax.axis("off")
f.subplots_adjust(wspace=0.05, hspace=0.18)
save_png(f, f"{OUT}/fig-metrics.png")

# --- 5. failure modes: an ellipse and holes ----------------------------------------------------------
f, axs = plt.subplots(2, 1, figsize=(400 / 72, 420 / 72))
for ax, key, nm in [(axs[0], "ellipse", "ellipse"), (axs[1], "holes", "holes")]:
    R_ = D[key]; lb = AI(R_["labels"]); mk_ = AI(R_["mask"])
    rgb = label_rgb(lb, seed=2); rgb[(mk_ == 0) & (lb == 0)] = (1, 1, 1)
    rgb[boundary(lb) & (mk_ > 0)] = (0.15, 0.15, 0.15)
    rgb[(mk_ == 0)] = (1, 1, 1)
    ax.imshow(rgb[52:128, 25:155])
    ax.contour(mk_[52:128, 25:155], levels=[0.5], colors=INK, linewidths=1)
    for pk in R_["peaks"]: ax.plot(pk["x"] - 25, pk["y"] - 52, "o", ms=5, mfc="white", mec=INK)
    ax.set_title(f"{nm}: {R_['n']} objects", fontsize=18); ax.axis("off")
f.subplots_adjust(hspace=0.2)
save_png(f, f"{OUT}/fig-failures.png")
print("ws-distance figures written")
