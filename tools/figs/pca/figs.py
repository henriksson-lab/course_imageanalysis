# Figures for the pca unit (pca/lec.qmd). Run from the repo root:
#   node_modules/.bin/esbuild tools/figs/pca/pcadata.ts --bundle --platform=node --outfile=<tmp>/pcadata.js
#   node <tmp>/pcadata.js > <tmp>/pcadata.json
#   python3 -I tools/figs/pca/figs.py <tmp>/pcadata.json
# The point clouds are the pca2d demo's data sets (pcadata.ts dumps them), so the numbers match the demo.
# Figure sizes are the display size in px / 96, so 13 pt text shows at about 17 px.
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/pca/'
plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, PT = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c", "#555"
D = {k: np.array(v) for k, v in json.load(open(sys.argv[1])).items()}
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)
def pca(P):
    m = P.mean(0); Y = P - m; S = Y.T @ Y / len(P); w, V = np.linalg.eigh(S)
    return m, S, w[::-1], V[:, ::-1]
def axes01(a, lo=0, hi=1):
    a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal")
    a.set_xlabel("pixel 1"); a.set_ylabel("pixel 2", labelpad=2)

# ------------------------------------------------------------ 1. centring (demo "no centring": unrelated pixels)
P = D["unrelated"]; m, S, w, V = pca(P)
M2 = P.T @ P / len(P); w2, V2 = np.linalg.eigh(M2); u0 = V2[:, -1] * np.sign(V2[0, -1])
err_pca = w[1]; err_0 = (P ** 2).sum(1).mean() - w2[-1]
print("centring: PCA error %.4f, line through (0,0) error %.4f, angle %.1f" % (err_pca, err_0, np.degrees(np.arctan2(u0[1], u0[0]))))
fig, a = plt.subplots(figsize=(330 / 96, 330 / 96))
a.plot(P[:, 0], P[:, 1], "o", color=PT, ms=4, alpha=0.8)
t = np.linspace(-2, 2, 2)
a.plot(t * u0[0], t * u0[1], color=RED, lw=2.2, ls="--")
v1 = V[:, 0]
a.plot(m[0] + t * v1[0], m[1] + t * v1[1], color=BLUE, lw=2.6)
a.plot(*m, "P", color=BLUE, ms=11, mec="white")
a.plot(0, 0, "o", color=RED, ms=7, clip_on=False)
a.text(0.04, 0.95, "through (0, 0)\nerror %.4f" % err_0, color=RED, fontsize=12.5, va="top")
a.text(0.36, 0.26, "through the mean\nerror %.4f" % err_pca, color=BLUE, fontsize=12.5, va="top")
axes01(a)
fig.subplots_adjust(left=0.17, right=0.94, top=0.97, bottom=0.15)
save(fig, "fig-centre.svg")

# ------------------------------------------------------------ 2. variance and error vs angle (neighbouring pixels)
P = D["pixels"]; m, S, w, V = pca(P)
ang = np.radians(np.linspace(0, 180, 361))
U = np.stack([np.cos(ang), np.sin(ang)])
var = np.einsum("in,ij,jn->n", U, S, U); err = np.trace(S) - var
th1 = np.degrees(np.arctan2(V[1, 0], V[0, 0])) % 180
print("pixels: var at 0 %.4f err at 0 %.4f, PC1 %.1f deg, l1 %.4f l2 %.4f tr %.4f share %.3f" % (var[0], err[0], th1, w[0], w[1], np.trace(S), w[0] / w.sum()))
fig, ax = plt.subplots(1, 2, figsize=(620 / 96, 290 / 96), gridspec_kw={"width_ratios": [1, 1.35]})
a = ax[0]
a.plot(P[:, 0], P[:, 1], "o", color=PT, ms=3.5, alpha=0.8)
for th, c, lab in [(0, RED, "u at 0°"), (np.radians(th1), BLUE, "PC1")]:
    a.annotate("", m + 0.38 * np.array([np.cos(th), np.sin(th)]), m, arrowprops=dict(arrowstyle="-|>", color=c, lw=2.4, mutation_scale=16))
a.text(m[0] + 0.30, m[1] - 0.09, "0°", color=RED, fontsize=13)
a.text(m[0] + 0.02, m[1] + 0.33, "45.3°", color=BLUE, fontsize=13, ha="right")
axes01(a)
b = ax[1]
deg = np.degrees(ang)
b.plot(deg, var, color=BLUE, lw=2.6, label="variance along u")
b.plot(deg, err, color=RED, lw=2.2, ls="--", label="error")
b.plot([0], [var[0]], "o", color=BLUE, ms=7); b.plot([th1], [w[0]], "o", color=BLUE, ms=8)
b.annotate("%.4f" % var[0], (0, var[0]), (8, var[0] + 0.004), color=BLUE, fontsize=12.5)
b.annotate("PC1: %.4f" % w[0], (th1, w[0]), (th1 + 8, w[0] - 0.002), color=BLUE, fontsize=12.5)
b.set_xticks([0, 45, 90, 135, 180]); b.set_xlabel("angle of u (degrees)")
b.set_yticks([0, 0.04, 0.08]); b.set_ylim(0, 0.088)
b.legend(fontsize=12.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=2, handlelength=1.5, columnspacing=1)
fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.17, wspace=0.35)
save(fig, "fig-spread.svg")

# ------------------------------------------------------------ 3. four images: products of the centred values
P = D["four"]; m, S, w, V = pca(P); Y = P - m
assert np.allclose(S, [[0.05, 0.03], [0.03, 0.05]])
fig, a = plt.subplots(figsize=(300 / 96, 300 / 96))
a.add_patch(Rectangle((0, 0), 0.4, 0.4, fc="#dcecdc", ec="none")); a.add_patch(Rectangle((-0.4, -0.4), 0.4, 0.4, fc="#dcecdc", ec="none"))
a.add_patch(Rectangle((-0.4, 0), 0.4, 0.4, fc="#f6dede", ec="none")); a.add_patch(Rectangle((0, -0.4), 0.4, 0.4, fc="#f6dede", ec="none"))
for (x, y, t, c) in [(0.31, 0.31, "+", GREEN), (-0.31, -0.31, "+", GREEN), (-0.22, 0.22, "−", RED), (0.22, -0.22, "−", RED)]:
    a.text(x, y, t, ha="center", va="center", fontsize=24, color=c, fontweight="bold")
a.axhline(0, color="#999", lw=1); a.axvline(0, color="#999", lw=1)
a.plot(Y[:, 0], Y[:, 1], "o", color=INK, ms=9)
a.set_title("sign of the product", fontsize=13, pad=4)
a.set_xlim(-0.4, 0.4); a.set_ylim(-0.4, 0.4); a.set_aspect("equal")
a.set_xticks([-0.3, 0, 0.3]); a.set_yticks([-0.3, 0, 0.3])
a.set_xlabel("pixel 1 − mean"); a.set_ylabel("pixel 2 − mean", labelpad=0)
fig.subplots_adjust(left=0.23, right=0.96, top=0.91, bottom=0.17)
save(fig, "fig-four.svg")

# ------------------------------------------------------------ 4. projection onto PC1: the four images
fig, a = plt.subplots(figsize=(310 / 96, 310 / 96))
u1 = np.array([1, 1]) / np.sqrt(2)
t = np.linspace(-1, 1, 2)
a.plot(m[0] + t * u1[0], m[1] + t * u1[1], color=BLUE, lw=2.4)
for p in P:
    s = (p - m) @ u1; q = m + s * u1
    a.plot([p[0], q[0]], [p[1], q[1]], color=RED, lw=2.2)
    a.plot(*q, "o", color=BLUE, ms=7)
a.plot(P[:, 0], P[:, 1], "o", color=INK, ms=8)
a.plot(*m, "P", color=BLUE, ms=11, mec="white")
a.annotate("(0.2, 0.4)", P[0], (0.05, 0.55), fontsize=12.5, arrowprops=dict(arrowstyle="-", color="#999"))
a.annotate("rebuilt (0.3, 0.3)", (0.3, 0.3), (0.38, 0.14), fontsize=12.5, color=BLUE, arrowprops=dict(arrowstyle="-", color="#999"))
a.text(0.52, 0.42, "mean", color=BLUE, fontsize=12.5)
axes01(a)
a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1])
fig.subplots_adjust(left=0.19, right=0.94, top=0.97, bottom=0.16)
save(fig, "fig-project.svg")

# ------------------------------------------------------------ 5. patch PCs next to the closest cosine waves
cell = plt.imread(R + '/demos/src/fourier2d/cell256.png').astype(float)
if cell.ndim == 3: cell = cell[..., 0]
p = 8
from numpy.lib.stride_tricks import sliding_window_view
X = sliding_window_view(cell, (p, p)).reshape(-1, p * p)
Xc = X - X.mean(0); Sx = Xc.T @ Xc / len(X)
wx, Vx = np.linalg.eigh(Sx); wx, Vx = wx[::-1], Vx[:, ::-1]
print("patches:", len(X), "shares %", np.round(100 * wx[:5] / wx.sum(), 1))
def dct1(k, n=p):
    c = np.cos(np.pi * (np.arange(n) + 0.5) * k / n); return c / np.linalg.norm(c)
Dct = {(i, j): np.outer(dct1(i), dct1(j)).ravel() for i in range(p) for j in range(p)}
fig, ax = plt.subplots(2, 5, figsize=(560 / 96, 270 / 96))
for k in range(5):
    v = Vx[:, k]
    (bi, bj), best = max(((ij, abs(v @ d)) for ij, d in Dct.items()), key=lambda z: z[1])
    d = Dct[(bi, bj)]; v = v * np.sign(v @ d)        # same sign for both (an eigenvector's sign is arbitrary)
    print("PC%d best DCT (%d,%d) overlap %.3f" % (k + 1, bi, bj, best))
    for r, img in [(0, v), (1, d)]:
        lim = np.abs(img).max()
        ax[r, k].imshow(img.reshape(p, p), cmap="gray", vmin=-lim if k else 0, vmax=lim, interpolation="nearest")
        ax[r, k].set_xticks([]); ax[r, k].set_yticks([])
    ax[0, k].set_title("PC%d" % (k + 1), fontsize=13.5, pad=3)
    ax[1, k].set_xlabel("%.2f" % best, fontsize=13.5)
ax[0, 0].set_ylabel("PCA", fontsize=13.5); ax[1, 0].set_ylabel("cosine", fontsize=13.5)
fig.text(0.005, 0.035, "overlap", ha="left", fontsize=12, color="#555")
fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.12, wspace=0.12, hspace=0.12)
save(fig, "fig-pcdct.png", dpi=192)

# ------------------------------------------------------------ 6. one outlier turns PC1
P = D["outlier"]
fig, a = plt.subplots(figsize=(320 / 96, 320 / 96))
a.plot(P[:-1, 0], P[:-1, 1], "o", color=PT, ms=4.5)
a.plot(*P[-1], "o", color=RED, ms=10)
a.text(P[-1, 0] - 0.02, P[-1, 1] + 0.05, "one hot pixel", color=RED, ha="right", fontsize=12.5)
t = np.linspace(-1, 1, 2)
for Q, c, ls, lab in [(P[:-1], BLUE, "-", "without: %.0f %%"), (P, RED, "--", "with: %.0f %%")]:
    m, S, w, V = pca(Q)
    a.plot(m[0] + t * V[0, 0], m[1] + t * V[1, 0], color=c, lw=2.4, ls=ls, label="PC1 " + lab % (100 * w[0] / w.sum()))
    print(lab % (100 * w[0] / w.sum()), "angle %.1f" % (np.degrees(np.arctan2(V[1, 0], V[0, 0])) % 180))
axes01(a)
a.legend(fontsize=12, frameon=False, loc="upper left", handlelength=1.6)
fig.subplots_adjust(left=0.18, right=0.94, top=0.97, bottom=0.15)
save(fig, "fig-outlier.svg")
