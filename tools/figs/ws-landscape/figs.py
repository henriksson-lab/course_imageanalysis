# Figures for ws-landscape/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/ws-landscape/figs.py
# The 1D profiles and the nuclei image come from the demos' own code (data.ts, via node).
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from skimage.segmentation import watershed
from wsstyle import *

setup()
OUT = "ws-landscape"
D = run_ts(HERE)


def tri(ax, x, y, s=0.32, filled=False, colour=INK):
    ax.add_patch(Polygon([[x, y], [x - s, y - 2.2 * s], [x + s, y - 2.2 * s]], closed=True,
                         fc=colour if filled else "white", ec=colour, lw=1.6, zorder=5))


# --- 1. thresholding + labelling merges touching nuclei ----------------------------------
n = D["nuc"]["n"]; img = np.array(D["nuc"]["image"]).reshape(n, n); lab = np.array(D["nuc"]["labels"]).reshape(n, n)
f, axs = plt.subplots(1, 2, figsize=(560 / 72, 300 / 72))
axs[0].imshow(img, cmap="gray", vmin=0, vmax=255)
axs[0].set_title(f"{D['nuc']['truth']} nuclei", fontsize=18)
axs[1].imshow(label_rgb(lab, bg=(0.15, 0.15, 0.15)))
axs[1].set_title(f"threshold, label: {D['nuc']['count']} objects", fontsize=17)
for a in axs: a.axis("off")
f.subplots_adjust(wspace=0.05)
save_png(f, f"{OUT}/fig-touching.png")

# --- 2. an image as a landscape: two dark nuclei --------------------------------------------
g = lambda x, y, cx, cy, s: np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / s ** 2)
Y, X = np.mgrid[0:60, 0:60].astype(float)
F = 8.6 - 6.2 * g(X, Y, 19, 30, 5.5) - 5.2 * g(X, Y, 40, 30, 6)
f = plt.figure(figsize=(620 / 72, 300 / 72))
ax = f.add_axes([0.0, 0.08, 0.38, 0.84])
ax.imshow(F, cmap="gray", vmin=0, vmax=10, origin="upper")
ax.plot([0, 59], [30, 30], "--", color=ORANGE, lw=2)
ax.set_title("grey level $f(x, y)$", fontsize=18); ax.axis("off")
a3 = f.add_axes([0.36, -0.08, 0.66, 1.16], projection="3d")
a3.plot_surface(X, Y, F, cmap="gray", vmin=0, vmax=11, rstride=1, cstride=1, linewidth=0, antialiased=False)
a3.plot(np.arange(60), np.full(60, 30), F[30] + 0.05, color=ORANGE, lw=2.5, zorder=10)
a3.view_init(elev=32, azim=-62); a3.set_box_aspect((1, 1, 0.45))
a3.set_xticks([]); a3.set_yticks([]); a3.set_zticks([0, 5, 10])
a3.set_title("the same as a height map", fontsize=18, y=0.93)
save_png(f, f"{OUT}/fig-landscape.png", dpi=120)

# --- 3. minima, basins, watershed lines, drops of water --------------------------------------
Y, X = np.mgrid[0:160, 0:240].astype(float)
cen = [(60, 70, 34, 6.0), (170, 55, 30, 5.0), (150, 125, 26, 4.2)]
H = 9 - sum(a * g(X, Y, cx, cy, s) for cx, cy, s, a in cen) + 0.6 * ((X - 120) / 120) ** 2
mk = np.zeros(H.shape, int)
for k, (cx, cy, s, a) in enumerate(cen):
    yy, xx = np.unravel_index(np.argmin(np.where(np.hypot(X - cx, Y - cy) < 15, H, 99)), H.shape); mk[yy, xx] = k + 1
B = watershed(H, mk, watershed_line=True)
fills = np.array([[1, 1, 1]] + [[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] for c in BASIN_FILL[:3]])
f, ax = plt.subplots(figsize=(560 / 72, 380 / 72))
ax.imshow(fills[B], origin="upper", alpha=0.9)
ax.contour(H, levels=np.arange(3, 9.5, 0.5), colors="#7a7268", linewidths=0.8)
ax.contour(B == 0, levels=[0.5], colors=DAM, linewidths=3)
gy, gx = np.gradient(H)
for (x0, y0) in [(20, 20), (110, 10), (118, 150), (200, 150), (95, 95), (230, 20), (128, 70)]:
    pts = [(x0, y0)]; x, y = float(x0), float(y0)
    for _ in range(4000):
        i, j = int(round(y)), int(round(x))
        d = np.array([gx[i, j], gy[i, j]]); nn = np.hypot(*d)
        if nn < 1e-4: break
        x -= 0.8 * d[0] / nn; y -= 0.8 * d[1] / nn
        x = min(max(x, 0), 239); y = min(max(y, 0), 159); pts.append((x, y))
        if np.hypot(*(np.array(pts[-1]) - pts[-12])) < 1 if len(pts) > 12 else False: break
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], color=BLUE, lw=2)
    ax.plot(x0, y0, "o", color=BLUE, ms=7)
    ax.annotate("", xy=pts[-1], xytext=pts[-6], arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
for k in range(3):
    yy, xx = np.argwhere(mk == k + 1)[0]
    ax.plot(xx, yy, "v", ms=14, mfc="white", mec=INK, mew=2)
ax.text(232, 152, "watershed line", ha="right", va="bottom", fontsize=17, color=DAM,
        bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
ax.set_xlim(0, 239); ax.set_ylim(159, 0); ax.axis("off")
save(f, f"{OUT}/fig-basins.svg")

# --- 4. flood f or |f'|: the step profile --------------------------------------------------
fs = [1, 1, 1, 1, 5, 5, 5, 2, 2, 2]
d = [0] + [abs(fs[i + 1] - fs[i - 1]) / 2 for i in range(1, 9)] + [0]
ld = flood1d(d, minima1d(d)); lf = flood1d(fs, minima1d(fs))
f, axs = plt.subplots(2, 1, figsize=(560 / 72, 400 / 72), sharex=True, gridspec_kw=dict(hspace=0.35))
for ax, v, name, labs in [(axs[0], fs, "$f$", lf), (axs[1], d, "$|f'|$", ld)]:
    for i, h in enumerate(v):
        ax.add_patch(Rectangle((i - 0.45, 0), 0.9, h, fc=LAND, ec="none"))
        ax.text(i, h + 0.1, f"{h:g}", ha="center", va="bottom", fontsize=16)
    ax.set_ylabel(name); ax.set_ylim(0, 6.4 if v is fs else 2.9); ax.set_xlim(-0.6, 9.6)
    ax.spines["left"].set_visible(False); ax.set_yticks([])
for ax, labs, nm, y in [(axs[0], lf, "flood $f$", -0.9), (axs[1], ld, "flood $|f'|$", -0.45)]:
    pass
# label strips under each panel
for ax, labs, nm in [(axs[0], lf, "flood $f$:"), (axs[1], ld, "flood $|f'|$:")]:
    for i, l in enumerate(labs):
        ax.text(i, -0.2, "ABC"[l - 1], transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontweight="bold", color=BASIN_INK[l - 1], fontsize=18)
        if i and labs[i] != labs[i - 1]:
            ax.axvline(i - 0.5, color=DAM, lw=3, ymax=1.0)
    ax.text(-0.75, -0.2, nm, transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=16)
for ax in axs: ax.set_xticks([]); ax.spines["bottom"].set_visible(True)
save(f, f"{OUT}/fig-deriv.svg")


# --- 5. every minimum makes a region: clean vs noisy profile ---------------------------------
def profile(ax, P, title):
    v = np.array(P["f"]); lab = np.array(P["labels"]); mkk = np.array(P["markers"]); n = len(v)
    ax.bar(np.arange(n), v, width=1.0, color=LAND)
    for i in range(n):
        ax.add_patch(Rectangle((i - 0.5, -0.9), 1, 0.6, fc=BASIN_FILL[(lab[i] - 1) % 6], ec="none"))
    for i in range(1, n):
        if lab[i] != lab[i - 1]: ax.plot([i - 0.5, i - 0.5], [-0.9, max(v[i], v[i - 1])], color=DAM, lw=2.2)
    for i in range(n):
        if mkk[i] and (i == 0 or mkk[i - 1] != mkk[i]): tri(ax, i, v[i] - 0.15, s=0.7)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-1.0, 10); ax.set_yticks([0, 5, 10])
    ax.text(0.99, 0.97, title, transform=ax.transAxes, ha="right", va="top", fontsize=17,
            bbox=dict(fc="white", ec="none", pad=1))
    ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
f, axs = plt.subplots(2, 1, figsize=(560 / 72, 400 / 72), gridspec_kw=dict(hspace=0.15))
profile(axs[0], D["clean"], f"two cells: {D['clean']['nMinima']} minima, {D['clean']['nMinima']} regions")
profile(axs[1], D["noisy"], f"with noise: {D['noisy']['nMinima']} minima, {D['noisy']['nMinima']} regions")
axs[1].set_xlabel("position along the line")
save(f, f"{OUT}/fig-noisy.svg")

# --- 6. markers: the 12 samples flooded from samples 3 and 10 only ----------------------------
v = [7, 1, 4, 3, 4, 6, 5, 1, 6, 4, 2, 3]
mk = [0] * 12; mk[3] = 1; mk[10] = 2
lab = flood1d(v, mk); level = 6
f, ax = plt.subplots(figsize=(560 / 72, 300 / 72))
for i, h in enumerate(v):
    ax.add_patch(Rectangle((i - 0.51, 0), 1.02, h, fc=LAND, ec="none"))
    if h < level: ax.add_patch(Rectangle((i - 0.51, h), 1.02, level - h, fc=BASIN_FILL[lab[i] - 1], ec="none"))
    ax.text(i, -0.5, str(h), ha="center", va="top", family="monospace")
    ax.text(i, -1.6, "AB"[lab[i] - 1], ha="center", va="top", fontweight="bold", color=BASIN_INK[lab[i] - 1])
ax.add_patch(Rectangle((7.5 - 0.09, 0), 0.18, 6, fc=DAM, ec="none"))
ax.text(7.5, 6.2, "6", ha="center", va="bottom")
ax.axhline(level, color="#1f5fa8", ls="--", lw=1.2)
for i in (3, 10): tri(ax, i, v[i] - 0.15, s=0.22, filled=True)
for i in (1, 7):
    tri(ax, i, v[i] - 0.15, s=0.22)
    ax.text(i, v[i] + 0.35, "drowned", ha="center", va="bottom", fontsize=16, color=INK)
ax.text(-0.7, -0.5, "$f$", ha="right", va="top"); ax.text(-0.7, -1.6, "label", ha="right", va="top", fontsize=16)
ax.set_xlim(-0.5, 11.5); ax.set_ylim(-2.6, 8); ax.axis("off")
save(f, f"{OUT}/fig-markers.svg")
print("ws-landscape figures written")
