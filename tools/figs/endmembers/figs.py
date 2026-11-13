# Static figures for endmembers/lec.qmd. Run from the repository root:
#   node_modules/.bin/esbuild tools/figs/endmembers/export.ts --bundle --platform=node --outfile=<tmp>/x.js
#   node <tmp>/x.js <tmp>
#   python3 -I tools/figs/endmembers/figs.py <tmp>
# Singular values, PCA axes and the projected clouds come from the endmembers demo's model
# (16 channels, 400 photons, seed 1); the small 2D examples are computed here (fixed seeds).
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams.update({"font.size": 17, "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})
COL = ["#3f72ff", "#1fa83a", "#e03cc8", "#d99a12"]          # nucleus, mitochondria, membrane, autofl. (as in the demos)
NAMES = ["nucleus", "mitochondria", "membrane", "autofluor."]
LS_C, NN_C, GREY = "#d62728", "#ff7f0e", "#888888"            # LS red, NNLS orange (as in the unmixcloud demo)
OUT = "endmembers/"

def save(fig, name, **kw):
    fig.savefig(OUT + name, **kw)
    if os.environ.get("PREVIEW"):
        fig.savefig(os.path.join(os.environ["PREVIEW"], name.rsplit(".", 1)[0] + ".png"), dpi=96)
    plt.close(fig); print("wrote", OUT + name)

D = json.load(open(sys.argv[1] + "/endmembers.json"))
cen = np.array(D["cen"])
TRI = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])

def tri_axes(ax):
    ax.add_patch(Polygon(TRI, closed=True, fill=False, ec="#555", lw=1.5))
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.12, 0.98)

# 1. fig-simplex.svg: normalised mixtures in the triangle, one point with its fractions
rng = np.random.default_rng(2)
f = rng.dirichlet([0.8, 0.8, 0.8], 300); pts = f @ TRI
fig, ax = plt.subplots(figsize=(4.6, 4.0)); tri_axes(ax)
ax.plot(pts[:, 0], pts[:, 1], ".", color=GREY, ms=4, alpha=0.7)
for k in range(3):
    ax.plot(*TRI[k], "o", color=COL[k], ms=13)
ax.text(*(TRI[0] + [-0.02, -0.08]), "dye 1", ha="center", va="top", color=COL[0])
ax.text(*(TRI[1] + [0.02, -0.08]), "dye 2", ha="center", va="top", color=COL[1])
ax.text(*(TRI[2] + [0, 0.05]), "dye 3", ha="center", va="bottom", color=COL[2])
p = np.array([0.5, 0.3, 0.2]) @ TRI
ax.plot(*p, "o", color="black", ms=9)
ax.annotate("(0.5, 0.3, 0.2)", xy=p, xytext=(p[0] + 0.12, p[1] + 0.33), fontsize=17,
            arrowprops=dict(arrowstyle="->", color="black"))
fig.tight_layout(pad=0.2); save(fig, "fig-simplex.svg")

# 2. fig-nnls.svg: the two-channel example: LS goes negative, NNLS lands on the edge
m1, m2, y = np.array([1, 0.2]), np.array([0.3, 1]), np.array([0.1, 0.9])
a_ls = np.linalg.solve(np.column_stack([m1, m2]), y)
a2 = y @ m2 / (m2 @ m2); nn = a2 * m2
print("LS", a_ls.round(3), "NNLS a2", round(a2, 3))
fig, ax = plt.subplots(figsize=(4.8, 4.4))
ax.add_patch(Polygon([[0, 0], 5 * m1, 5 * m2], closed=True, color="#cccccc", alpha=0.55, lw=0))
arr = dict(head_width=0.035, length_includes_head=True, lw=2.5)
ax.arrow(0, 0, *m1, color=COL[0], **arr); ax.arrow(0, 0, *m2, color=COL[1], **arr)
ax.text(1.0, 0.11, "spectrum 1", color=COL[0], ha="right", va="top")
ax.text(0.34, 0.99, "spectrum 2", color=COL[1], ha="left", va="top")
ax.plot(*y, "o", color="black", ms=9)
ax.plot([y[0], nn[0]], [y[1], nn[1]], color=NN_C, lw=2, ls="--"); ax.plot(*nn, "s", color=NN_C, ms=9)
ax.text(nn[0] + 0.06, nn[1] - 0.06, f"NNLS: (0, {a2:.2f})", color=NN_C, va="top")
ax.text(0.02, 1.1, f"pixel; LS gives\n({a_ls[0]:.2f}, {a_ls[1]:.2f})".replace("-", "−"), color=LS_C, va="bottom")
ax.text(0.62, 0.55, "cone", color="#666", fontsize=17)
ax.set_xlim(0, 1.2); ax.set_ylim(0, 1.35); ax.set_aspect("equal")
ax.set_xlabel("channel 1"); ax.set_ylabel("channel 2")
fig.tight_layout(); save(fig, "fig-nnls.svg")

# 3. fig-constraints.svg: (a) clipping a noisy zero biases it up, (b) a wrong endmember hides in the residual
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axs[0]; z = np.random.default_rng(5).normal(0, 0.1, 20000); zc = np.maximum(z, 0)
bins = np.linspace(-0.35, 0.35, 36)
ax.hist(z, bins, color="#bbbbbb", label="LS (mean 0)")
ax.hist(zc[zc > 0], bins, color=NN_C, alpha=0.75, label="clipped at 0")
ax.axvline(zc.mean(), color=NN_C, lw=2, ls="--")
ax.text(zc.mean() + 0.02, ax.get_ylim()[1] * 0.92, f"mean {zc.mean():.2f}", color=NN_C, va="top", fontsize=16)
ax.set_xlabel("estimated abundance (true 0)"); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.legend(frameon=False, fontsize=15, loc="upper left", bbox_to_anchor=(-0.04, 1.12))
ax = axs[1]; tri_axes(ax)
wrong = np.array([0.62, 0.45])   # the third vertex dragged inward
ax.add_patch(Polygon([TRI[0], TRI[1], wrong], closed=True, fill=True, color=NN_C, alpha=0.15, lw=0))
ax.add_patch(Polygon([TRI[0], TRI[1], wrong], closed=True, fill=False, ec=NN_C, lw=2, ls="--"))
r = np.random.default_rng(6); q = TRI[2] + r.normal(0, 0.03, (25, 2)) * [1, 1] - [0, 0.04]
for pnt in q:
    # project onto the nearer of the two upper edges of the wrong triangle (approximately)
    best = None
    for A, B in ((TRI[0], wrong), (TRI[1], wrong)):
        t = np.clip((pnt - A) @ (B - A) / ((B - A) @ (B - A)), 0, 1); c = A + t * (B - A)
        if best is None or np.linalg.norm(pnt - c) < np.linalg.norm(pnt - best): best = c
    ax.plot([pnt[0], best[0]], [pnt[1], best[1]], color="#999", lw=0.8)
ax.plot(q[:, 0], q[:, 1], ".", color=COL[2], ms=7)
ax.plot(*wrong, "o", color=NN_C, ms=10)
ax.text(0.53, 0.27, "wrong\nendmember", color=NN_C, fontsize=16, va="top", ha="center")
ax.text(TRI[2][0] + 0.07, TRI[2][1] - 0.02, "real pixels", color=COL[2], fontsize=16)
fig.tight_layout(); save(fig, "fig-constraints.svg")

# 4. fig-cluster.svg: the same cloud, clustered (one class each) and unmixed (fractions)
rng = np.random.default_rng(8)
f = rng.dirichlet([0.6, 0.6, 0.6], 500); pts = f @ TRI
fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.7))
cols = np.array([[0x3f, 0x72, 0xff], [0x1f, 0xa8, 0x3a], [0xe0, 0x3c, 0xc8]]) / 255
for ax, lab in zip(axs, ("clustering", "unmixing")):
    tri_axes(ax); ax.set_title(lab, fontsize=17)
    c = cols[np.argmax(f, 1)] if lab.startswith("cl") else f @ cols
    ax.scatter(pts[:, 0], pts[:, 1], s=6, c=np.clip(c, 0, 1))
fig.tight_layout(); save(fig, "fig-cluster.svg")

# 5. fig-scree.svg: singular values of the day's cell (4 components), log scale
sv = np.array(D["sv"])[:8]; r = sv / sv[0]
fig, ax = plt.subplots(figsize=(4.8, 3.6))
ax.bar(np.arange(1, 9), r, color=[COL[0] if i < 4 else "#aaaaaa" for i in range(8)], width=0.6)
ax.set_yscale("log"); ax.set_ylim(0.03, 1.4); ax.set_xticks(range(1, 9))
ax.set_xlabel("index i"); ax.set_ylabel("$\\sigma_i / \\sigma_1$")
ax.axhline(0.07, color="#666", ls="--", lw=1.2); ax.text(8.4, 0.078, "noise floor", ha="right", va="bottom", fontsize=16, color="#555")
for i in range(4): ax.text(i + 1, r[i] * 1.08, f"{r[i]:.2f}", ha="center", va="bottom", fontsize=14)
fig.tight_layout(); save(fig, "fig-scree.svg")

# 6. fig-pcaaxes.svg: true spectra (non-negative) and the first three principal axes (signed)
Mt = np.array(D["Mtrue"]); pca = np.array(D["pca"])
fig, axs = plt.subplots(2, 1, figsize=(5.0, 4.6), sharex=True)
for k in range(4): axs[0].plot(cen, Mt[k], color=COL[k], lw=2.5)
axs[0].set_ylabel("spectrum"); axs[0].set_yticks([0, 0.2]); axs[0].set_title("true spectra", fontsize=17, loc="left")
axs[1].axhline(0, color="black", lw=0.8)
for k, ls in zip(range(3), ("-", "--", ":")): axs[1].plot(cen, pca[k], color="#444", lw=2.2, ls=ls, label=f"PC {k + 1}")
axs[1].set_ylabel("PC axis"); axs[1].set_xlabel("wavelength (nm)")
for k, (xx, yy) in enumerate(((cen[2] + 4, pca[0][2] + 0.05), (552, -0.28), (cen[10] + 8, pca[2][10]))):
    axs[1].text(xx, yy, f"PC {k + 1}", fontsize=15, color="#222", va="bottom" if k != 1 else "top")
fig.tight_layout(); save(fig, "fig-pcaaxes.svg")

# 7. fig-vertices.svg: normalised cloud in its two main directions; true (circles) and found (squares)
fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.9))
for ax, key, lab in zip(axs, ("pure", "nopure"), ("pure pixels", "no pure pixels")):
    d = D[key]; c = np.array(d["cloud"]); tv = np.array(d["tv"]); fv = np.array(d["fv"])
    ax.plot(c[:, 0], c[:, 1], ".", color=GREY, ms=2, alpha=0.35)
    ax.plot(*np.vstack([tv, tv[:1]]).T, color="#555", ls="--", lw=1.4)
    for k in range(3): ax.plot(*tv[k], "o", color=COL[k], ms=11, mfc="none", mew=2.2)
    for j in range(3):
        k = d["perm"].index(j) if j in d["perm"] else j
        ax.plot(*fv[j], "s", color=COL[k], ms=7)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(lab, fontsize=17)
    for s in ("left", "bottom"): ax.spines[s].set_visible(False)
allp = np.vstack([np.array(D[k]["tv"]) for k in ("pure", "nopure")]); e = np.abs(allp).max() * 1.15
for ax in axs: ax.set_xlim(-e, e); ax.set_ylim(-e, e)
fig.tight_layout(); save(fig, "fig-vertices.svg")

# 8. fig-nmf-cones.svg: two channels; the data wedge and two different pairs of spectra that both enclose it
fig, ax = plt.subplots(figsize=(4.6, 4.2))
rng = np.random.default_rng(3)
ang = rng.uniform(np.radians(28), np.radians(62), 250); rad = rng.uniform(0.15, 1.0, 250)
ax.plot(rad * np.cos(ang), rad * np.sin(ang), ".", color=GREY, ms=4, alpha=0.7)
for (a1, a2), col, lab in (((20, 70), "#0072b2", "answer A"), ((5, 85), "#d55e00", "answer B")):
    for a in (a1, a2):
        ax.arrow(0, 0, 1.15 * np.cos(np.radians(a)), 1.15 * np.sin(np.radians(a)), color=col, head_width=0.03, lw=2.2, length_includes_head=True)
ax.text(1.12, 0.36, "A", color="#0072b2", fontsize=17); ax.text(0.42, 1.12, "A", color="#0072b2", fontsize=17)
ax.text(1.16, 0.06, "B", color="#d55e00", fontsize=17); ax.text(0.04, 1.17, "B", color="#d55e00", fontsize=17)
ax.set_xlim(0, 1.3); ax.set_ylim(0, 1.3); ax.set_aspect("equal")
ax.set_xlabel("channel 1"); ax.set_ylabel("channel 2")
fig.tight_layout(); save(fig, "fig-nmf-cones.svg")
