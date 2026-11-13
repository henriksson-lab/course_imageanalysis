"""Slide figures for kmeans/lec.qmd. Run from the repo root: python3 -I tools/figs/kmeans/figs.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io

OUT = "kmeans/"
C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
plt.rcParams.update({"font.family": "sans-serif", "font.size": 16, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})


def kmeans(X, M, iters=100):
    M = M.copy()
    for _ in range(iters):
        a = ((X[:, None, :] - M[None]) ** 2).sum(2).argmin(1)
        M2 = np.array([X[a == k].mean(0) if (a == k).any() else M[k] for k in range(len(M))])
        if np.allclose(M2, M):
            break
        M = M2
    a = ((X[:, None, :] - M[None]) ** 2).sum(2).argmin(1)
    return M, a, ((X - M[a]) ** 2).sum()


def best_of(X, K, runs, rs):
    best = None
    for _ in range(runs):
        M0 = X[rs.choice(len(X), K, replace=False)]
        r = kmeans(X, M0)
        if best is None or r[2] < best[2]:
            best = r
    return best


# ---- 1. pixels as points: intensity against local mean ----
cell = io.imread("demos/src/fourier2d/cell256.png").astype(float)
loc = ndimage.uniform_filter(cell, 9)
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.3), gridspec_kw={"width_ratios": [1, 1.15]})
axs[0].imshow(cell, cmap="gray", vmin=0, vmax=255); axs[0].set_axis_off()
axs[0].set_title("image", fontsize=16)
rs = np.random.default_rng(1)
idx = rs.choice(cell.size, 6000, replace=False)
axs[1].scatter(cell.ravel()[idx], loc.ravel()[idx], s=2, c="#333", alpha=0.25, rasterized=True)
axs[1].set_xlabel("grey value"); axs[1].set_ylabel("local mean (9 × 9)")
axs[1].set_xlim(0, 255); axs[1].set_ylim(0, 255); axs[1].set_xticks([0, 128, 255]); axs[1].set_yticks([0, 128, 255])
axs[1].set_title("every pixel a point", fontsize=16)
fig.tight_layout(); fig.savefig(OUT + "fig-featurespace.png", dpi=150); plt.close(fig)

# ---- 2. isodata: two means and the threshold halfway ----
h = np.bincount(cell.astype(int).ravel(), minlength=256).astype(float)
g = np.arange(256)
t = 128.0
for _ in range(100):
    lo = (h[g <= t] * g[g <= t]).sum() / h[g <= t].sum()
    hi = (h[g > t] * g[g > t]).sum() / h[g > t].sum()
    t2 = (lo + hi) / 2
    if abs(t2 - t) < 1e-6:
        break
    t = t2
fig, ax = plt.subplots(figsize=(5.6, 3.3))
bw = 4
hb = h.reshape(-1, bw).sum(1)
x = np.arange(0, 256, bw) + bw / 2
ax.bar(x, hb, width=bw, color=np.where(x <= t, C[0], C[1]), alpha=0.55, edgecolor="none")
ax.axvline(t, color="#222", lw=2, ls="--")
for m_, c_, name in ((lo, C[0], "mean below"), (hi, C[1], "mean above")):
    ax.plot([m_], [hb.max() * 1.05], marker="v", ms=14, color=c_, clip_on=False)
    ax.text(m_, hb.max() * 1.12, f"{m_:.0f}", ha="center", color=c_, fontsize=15)
ax.text(t - 3, hb.max() * 0.97, f"t = {t:.0f}\n(halfway)", fontsize=15, ha="right", va="top",
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
ax.set_xlim(0, 255); ax.set_yticks([]); ax.set_xlabel("grey value"); ax.set_ylabel("pixels")
fig.tight_layout(); fig.savefig(OUT + "fig-isodata.svg"); plt.close(fig)
print(f"isodata: t = {t:.1f}, means {lo:.1f}, {hi:.1f}")

# ---- 3. one Lloyd iteration in 2D: Voronoi cells, then centres to the means ----
rs = np.random.default_rng(3)
X = np.vstack([rs.normal([2, 2], 0.6, (40, 2)), rs.normal([6, 2.5], 0.6, (40, 2)), rs.normal([4, 6], 0.6, (40, 2))])
M0 = np.array([[1.0, 4.5], [3.5, 1.0], [7.5, 5.5]])
a = ((X[:, None] - M0[None]) ** 2).sum(2).argmin(1)
M1 = np.array([X[a == k].mean(0) for k in range(3)])
gx, gy = np.meshgrid(np.linspace(-0.5, 8.5, 400), np.linspace(-0.5, 8.5, 400))
G = np.stack([gx.ravel(), gy.ravel()], 1)
from matplotlib.colors import ListedColormap
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.8))
for i, (ax, title) in enumerate(zip(axs, ["1. assign", "2. update"])):
    cellmap = ((G[:, None] - M0[None]) ** 2).sum(2).argmin(1).reshape(gx.shape)
    ax.imshow(cellmap, origin="lower", extent=(-0.5, 8.5, -0.5, 8.5), cmap=ListedColormap(["#d6e6f4", "#ffe2c6", "#d5efd2"]), alpha=1)
    ax.contour(gx, gy, cellmap, levels=[0.5, 1.5], colors="#666", linewidths=1.5, linestyles="--")
    ax.scatter(X[:, 0], X[:, 1], s=14, c=[C[k] for k in a])
    ax.scatter(M0[:, 0], M0[:, 1], marker="X", s=180, c=C[:3], edgecolors="#222", linewidths=1.5, zorder=5)
    if i == 1:
        for k in range(3):
            ax.annotate("", xy=M1[k], xytext=M0[k], arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#222", mutation_scale=18), zorder=6)
        ax.scatter(M1[:, 0], M1[:, 1], marker="o", s=150, c=C[:3], edgecolors="#222", linewidths=1.8, zorder=7)
    ax.set_title(title, fontsize=17); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for s_ in ax.spines.values():
        s_.set_visible(True); s_.set_color("#999")
fig.tight_layout(); fig.savefig(OUT + "fig-voronoi.svg"); plt.close(fig)

# ---- 4. J per iteration on the six points ----
P = np.array([1, 2, 4, 7, 9, 10.0])[:, None]
M = np.array([[1.0], [2.0]])
Js = []
for _ in range(10):
    a = np.abs(P - M.T).argmin(1)
    Js.append(((P[:, 0] - M[a, 0]) ** 2).sum())
    M2 = np.array([[P[a == k, 0].mean()] for k in range(2)])
    if np.allclose(M2, M):
        break
    M = M2
print("J trace:", np.round(Js, 2))
fig, ax = plt.subplots(figsize=(4.4, 3.3))
ax.plot(range(len(Js)), Js, "-o", color=C[0], lw=2.5, ms=9)
for i, j in enumerate(Js):
    ax.text(i + 0.12, j + 5, f"{j:.2f}" if j % 1 else f"{j:.0f}", fontsize=15)
ax.set_xticks(range(len(Js))); ax.set_xlabel("assignment"); ax.set_ylabel("J")
ax.set_xlim(-0.3, len(Js) - 0.2); ax.set_ylim(0, 160)
fig.tight_layout(); fig.savefig(OUT + "fig-jtrace.svg"); plt.close(fig)

# ---- 5. stuck against best on 0, 1, 5, 6, 10, 11 ----
pts = np.array([0, 1, 5, 6, 10, 11.0])
fig, ax = plt.subplots(figsize=(6.6, 2.9))
rows = [("stuck: J = 26", [0, 1, 8], [0, 1, 2, 2, 2, 2], 1.0), ("best: J = 1.5", [0.5, 5.5, 10.5], [0, 0, 1, 1, 2, 2], 0.0)]
for name, cen, lab, y in rows:
    ax.plot([-0.5, 11.5], [y, y], color="#ccc", lw=1, zorder=0)
    for k in range(2):
        b = (cen[k] + cen[k + 1]) / 2
        ax.plot([b, b], [y - 0.22, y + 0.22], color="#777", ls="--", lw=1.5)
    ax.scatter(pts, [y] * 6, s=110, c=[C[l] for l in lab], zorder=3)
    ax.scatter(cen, [y] * 3, marker="X", s=220, c=C[:3], edgecolors="#222", linewidths=1.5, zorder=4)
    ax.text(-0.8, y + 0.3, name, fontsize=16, va="bottom")
ax.set_ylim(-0.45, 1.75); ax.set_xlim(-1, 12); ax.set_yticks([]); ax.set_xticks(range(0, 12))
ax.spines["left"].set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "fig-stuck.svg"); plt.close(fig)

# ---- 6. k-means++: probability of each point as the second centre ----
D2 = pts ** 2
p = D2 / D2.sum()
fig, ax = plt.subplots(figsize=(5.6, 3.3))
bars = ax.bar(pts, p, width=0.8, color=[C[1] if v >= 10 else C[2] if v >= 5 else "#999" for v in pts])
for x_, p_, d_ in zip(pts[1:], p[1:], D2[1:]):
    ax.text(x_, p_ + 0.012, f"{d_:.0f}", ha="center", fontsize=14, color="#333")
ax.scatter([0], [0], marker="X", s=200, c=C[0], edgecolors="#222", zorder=5, clip_on=False)
ax.text(0.3, 0.08, "first centre", ha="left", fontsize=14)
ax.set_xticks(pts); ax.set_xticklabels([f"{v:.0f}" for v in pts], fontsize=14); ax.set_xlabel("point (numbers on bars: $D^2$)"); ax.set_ylabel("P(second centre)")
ax.set_ylim(0, 0.5)
fig.tight_layout(); fig.savefig(OUT + "fig-kmpp.svg"); plt.close(fig)
print("k-means++ probs:", np.round(p, 3), "sum", D2.sum())

# ---- 7. what k-means assumes: three failure cases ----
rs = np.random.default_rng(5)
fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.0))
# elongated: two parallel cigars
A = np.vstack([np.c_[rs.normal(0, 2.0, 80), rs.normal(0, 0.25, 80)], np.c_[rs.normal(0, 2.0, 80), rs.normal(1.6, 0.25, 80)]])
truth_e = np.r_[np.zeros(80), np.ones(80)]
M, a, _ = best_of(A, 2, 20, rs)
axs[0].scatter(A[:, 0], A[:, 1], s=14, c=[C[k] for k in a])
axs[0].scatter(M[:, 0], M[:, 1], marker="X", s=160, c=C[:2], edgecolors="#222", zorder=5)
axs[0].set_title("elongated: cut across", fontsize=16)
print("elongated agreement:", max((a == truth_e).mean(), (a != truth_e).mean()))
axs[0].set_xlim(-6, 6); axs[0].set_ylim(-3.6, 5.2)
# big and small
B = np.vstack([rs.normal([0, 0], 1.3, (120, 2)), rs.normal([3.6, 0], 0.3, (30, 2))])
M, a, _ = best_of(B, 2, 20, rs)
axs[1].scatter(B[:, 0], B[:, 1], s=14, c=[C[k] for k in a])
axs[1].scatter(M[:, 0], M[:, 1], marker="X", s=160, c=C[:2], edgecolors="#222", zorder=5)
mid = M.mean(0); nrm = M[1] - M[0]
axs[1].axline(mid, mid + np.array([-nrm[1], nrm[0]]), color="#666", ls="--", lw=1.5)
axs[1].set_title("big and small: points cross", fontsize=16)
axs[1].set_ylim(-4, 4); axs[1].set_xlim(-4.5, 5.5)
print("big/small sizes:", np.bincount(a))
# units: the same cigars with y measured in units 3 times smaller (y x 3)
Cc = A * np.array([1.0, 3.0])
M, a, _ = best_of(Cc, 2, 20, rs)
axs[2].scatter(Cc[:, 0], Cc[:, 1], s=14, c=[C[k] for k in a])
axs[2].scatter(M[:, 0], M[:, 1], marker="X", s=160, c=C[:2], edgecolors="#222", zorder=5)
axs[2].set_title("same data, y × 3: split right", fontsize=16)
axs[2].set_xlim(-6, 6); axs[2].set_ylim(-3, 7.8)
print("rescaled agreement:", max((a == truth_e).mean(), (a != truth_e).mean()))
for ax in axs:
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", adjustable="box")
    for s_ in ax.spines.values():
        s_.set_visible(True); s_.set_color("#999")
fig.tight_layout(); fig.savefig(OUT + "fig-assume.svg"); plt.close(fig)

# ---- 8. elbow: best J against K (four blobs) ----
rs = np.random.default_rng(8)
X = np.vstack([rs.normal(c, 0.55, (50, 2)) for c in ([0, 0], [4, 0.5], [1.5, 4], [5, 4.5])])
Ks = range(1, 9)
Jb = [best_of(X, K, 20, rs)[2] for K in Ks]
print("elbow J:", np.round(Jb, 1))
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1, 1.4]})
axs[0].scatter(X[:, 0], X[:, 1], s=10, c="#555"); axs[0].set_xticks([]); axs[0].set_yticks([])
axs[0].set_aspect("equal"); axs[0].set_title("data: 4 groups", fontsize=16)
for s_ in axs[0].spines.values():
    s_.set_visible(True); s_.set_color("#999")
axs[1].plot(list(Ks), Jb, "-o", color=C[0], lw=2.5, ms=8)
axs[1].plot([4], [Jb[3]], "o", ms=16, mfc="none", mec=C[1], mew=2.5)
axs[1].annotate("elbow", xy=(4, Jb[3]), xytext=(5.2, Jb[0] * 0.45), fontsize=16,
                arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.5))
axs[1].set_xlabel("K"); axs[1].set_ylabel("best J"); axs[1].set_xticks(list(Ks))
fig.tight_layout(); fig.savefig(OUT + "fig-elbow.svg"); plt.close(fig)
