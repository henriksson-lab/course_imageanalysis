# Figures for orthogonal-moments/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/orthogonal-moments/figs.py
# Reconstructions come from the momentrecon demo's code (data.ts and cell.ts, via node); the
# rest is computed here. The micrograph is demos/src/fourier2d/cell256.png (Robert M. Hunt,
# CC BY 3.0, via Wikimedia Commons). fig-polybasis.svg was made earlier by the day builder.
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "label-connectivity"))
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as npleg
from labstyle import setup, fig, save, run_ts, BLUE, ORANGE, GREEN, RED, GREY, INK

setup()
OUT = "orthogonal-moments"
D = run_ts(HERE)
W = 64


def flat(ax):
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------- 1. Gram matrices as cosines
N = 10
G = np.array([[(2 / (j + k + 1) if (j + k) % 2 == 0 else 0) for k in range(N + 1)] for j in range(N + 1)])
kM = np.linalg.cond(G)
GL = np.diag([2 / (2 * n + 1) for n in range(N + 1)])
kL = np.linalg.cond(GL)
print(f"gram: kappa monomials {kM:.3g}, Legendre {kL:.3g}")
def cosm(G):
    d = np.sqrt(np.diag(G)); return G / np.outer(d, d)
f, axs = fig(460, 300, ncols=2, gridspec_kw=dict(wspace=0.12))
for ax, (G_, name, k) in zip(axs, [(G, "monomials", kM), (GL, "Legendre", kL)]):
    im = ax.imshow(np.abs(cosm(G_)), cmap="Greys", vmin=0, vmax=1)
    flat(ax)
    ax.set_title(name, fontsize=17, pad=4)
    kt = f"{k:.1e}".replace("e+0", " × 10^").split(" × 10^")
    ktxt = f"κ = {kt[0]} × $10^{{{kt[1]}}}$" if k > 1000 else f"κ = {k:.0f}"
    ax.text(5, 11.6, ktxt, ha="center", va="top", fontsize=18, color=RED)
cb = f.colorbar(im, ax=axs, shrink=0.75, pad=0.03); cb.set_label("|cos| between pairs", fontsize=16); cb.ax.tick_params(labelsize=15)
save(f, f"{OUT}/fig-gram.svg")


# ---------------------------------------------------------------- 2. Gram-Schmidt step and the first Legendre polynomials
x = np.linspace(-1, 1, 400)
f, axs = fig(480, 300, ncols=2, gridspec_kw=dict(wspace=0.3))
ax = axs[0]
ax.plot(x, x**2, color=GREY, lw=2.5, label="$x^2$")
ax.plot(x, np.full_like(x, 1 / 3), color=ORANGE, lw=2.5, ls="--", label="its average 1/3")
ax.plot(x, x**2 - 1 / 3, color=BLUE, lw=3, label="$x^2 - 1/3$")
ax.axhline(0, color="#ccc", lw=1, zorder=0)
ax.set_xlim(-1, 1); ax.set_ylim(-0.5, 1.15); ax.set_xticks([-1, 0, 1]); ax.set_yticks([0, 1])
ax.legend(loc="upper center", frameon=False, fontsize=16, ncol=1, bbox_to_anchor=(0.5, 1.12), handlelength=1.2)
ax.set_xlabel("x")
ax = axs[1]
cols = ["#888888", "#E69F00", "#0072B2", "#009E73", "#D55E00", "#CC79A7"]
for n in range(6):
    c = np.zeros(n + 1); c[n] = 1
    ax.plot(x, npleg.legval(x, c), color=cols[n], lw=2.4, label=f"$P_{n}$")
ax.axhline(0, color="#ccc", lw=1, zorder=0)
ax.set_xlim(-1, 1); ax.set_ylim(-1.1, 1.1); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=17, handlelength=1.0)
ax.set_xlabel("x")
save(f, f"{OUT}/fig-gramschmidt.svg")


# ---------------------------------------------------------------- 3. the 2D Legendre basis P_p(x) P_q(y), p + q <= 4
n = 80
u = np.linspace(-1, 1, n)
P = [npleg.legval(u, np.eye(5)[k]) for k in range(5)]
f, axs = fig(370, 370, nrows=5, ncols=5, gridspec_kw=dict(wspace=0.06, hspace=0.06))
for q in range(5):
    for p in range(5):
        ax = axs[q, p]; flat(ax)
        for s_ in ax.spines.values(): s_.set_visible(False)
        if p + q <= 4:
            img = np.outer(P[q][::-1], P[p])
            ax.imshow(img, cmap="RdBu_r", vmin=-1, vmax=1)
        if q == 0: ax.set_title(f"p = {p}", fontsize=16, pad=3)
        if p == 0: ax.set_ylabel(f"q = {q}", fontsize=16)
save(f, f"{OUT}/fig-legendre2d.svg")


# ---------------------------------------------------------------- 4. reconstruction of the nucleus at N = 2, 8, 20 (demo numbers)
nuc = np.array(D["nucleus"]).reshape(W, W)
lo, hi = nuc.min(), nuc.max(); lo, hi = lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo)   # the demo's display range
f, axs = fig(620, 230, ncols=4, gridspec_kw=dict(wspace=0.08))
axs[0].imshow(nuc, cmap="gray", vmin=lo, vmax=hi); axs[0].set_title("input", fontsize=17, pad=3)
axs[0].text(32, 66, "64 × 64 pixels", ha="center", va="top", fontsize=15)
for ax, img, N_, e in zip(axs[1:], D["nucLeg"], [2, 8, 20], D["nucErr"]):
    ax.imshow(np.array(img).reshape(W, W), cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(f"N = {N_}", fontsize=17, pad=3)
    ax.text(32, 66, f"{(N_ + 1) * (N_ + 2) // 2} numbers\nerror {e:.3f}", ha="center", va="top", fontsize=15)
for ax in axs: flat(ax)
save(f, f"{OUT}/fig-recon.svg")


# ---------------------------------------------------------------- 5. ringing: square at N = 16, Legendre vs Fourier
sq = np.array(D["square"]).reshape(W, W)
lg = np.array(D["sqLeg"]).reshape(W, W); fo = np.array(D["sqFour"]).reshape(W, W)
el, ef = D["sqErr"]
lo, hi = -0.3, 1.3
row = 32
f = plt.figure(figsize=(470 / 72, 400 / 72))
gs = f.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.32, wspace=0.08)
for k, (img, ttl) in enumerate([(sq, "input"), (lg, f"Legendre, error {el:.3f}"), (fo, f"Fourier, error {ef:.3f}")]):
    ax = f.add_subplot(gs[0, k]); ax.imshow(img, cmap="gray", vmin=lo, vmax=hi); flat(ax)
    ax.set_title(ttl if k == 0 else ttl.split(",")[0], fontsize=16, pad=3)
    if k: ax.text(32, 66, ttl.split(", ")[1], ha="center", va="top", fontsize=16)
    ax.axhline(row, color=RED if k == 1 else (BLUE if k == 2 else GREY), lw=1.2, ls=":")
ax = f.add_subplot(gs[1, :])
xs = np.arange(W)
ax.plot(xs, sq[row], color=GREY, lw=2, label="input")
ax.plot(xs, lg[row], color=RED, lw=2.4, label="Legendre")
ax.plot(xs, fo[row], color=BLUE, lw=2.4, ls="--", label="Fourier")
ax.set_xlim(0, W - 1); ax.set_xlabel("x along the middle row (N = 16)"); ax.set_yticks([0, 1]); ax.set_xticks([0, 20, 40, 60])
ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=16, bbox_to_anchor=(0.5, 1.25), handlelength=1.3, columnspacing=1.0)
save(f, f"{OUT}/fig-ringing.svg")


# ---------------------------------------------------------------- 6. pixels: error against N on the micrograph (demo code, cell.ts)
import json, tempfile
from skimage import io
raw = io.imread("demos/src/fourier2d/cell256.png").astype(float)
tmp = os.path.join(tempfile.mkdtemp(), "cell.json"); json.dump(raw.ravel().tolist(), open(tmp, "w"))
C = run_ts(HERE, "cell.ts", (tmp,))
NN = len(C["point"]) - 1
print("micrograph errors at N = 40:", {k: round(C[k][NN], 3) for k in ("point", "integrated", "discrete")})
f, axs = fig(480, 330, ncols=2, gridspec_kw=dict(width_ratios=[1, 2.6], wspace=0.35))
axs[0].imshow(np.array(C["image"]).reshape(W, W), cmap="gray"); flat(axs[0])
axs[0].set_title("input", fontsize=16, pad=3)
ax = axs[1]
nn = np.arange(NN + 1)
for key, col, ls, lab in [("point", RED, "-", "sampled"), ("integrated", ORANGE, "--", "averaged per pixel"), ("discrete", BLUE, ":", "discrete")]:
    ax.plot(nn, C[key], color=col, lw=2.6, ls=ls, label=lab)
    ax.text(NN + 0.8, C[key][NN] + (0.022 if key == "integrated" else (-0.022 if key == "discrete" else 0)), f"{C[key][NN]:.3f}", color=col, fontsize=16, va="center", ha="left")
ax.set_xlim(0, NN); ax.set_ylim(0, 0.6); ax.set_xticks([0, 10, 20, 30, 40]); ax.set_xlabel("order N"); ax.set_ylabel("error")
ax.legend(loc="upper right", frameon=False, fontsize=16, bbox_to_anchor=(1.05, 1.08), handlelength=1.5)
save(f, f"{OUT}/fig-sampled.svg")


# ---------------------------------------------------------------- 7. orthogonal projection (math slide)
f, ax = fig(330, 245)
plane = np.array([[0, 0], [5.2, 0], [6.6, 1.6], [1.4, 1.6]])
ax.add_patch(plt.Polygon(plane, closed=True, fc="#e8eef8", ec="#9ab", lw=1.2))
o = np.array([1.2, 0.35]); fh = np.array([4.6, 1.0]); fv = np.array([4.6, 3.6])
ax.annotate("", xy=fh, xytext=o, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=3, mutation_scale=18))
ax.annotate("", xy=fv, xytext=o, arrowprops=dict(arrowstyle="-|>", color=INK, lw=3, mutation_scale=18))
ax.annotate("", xy=fv, xytext=fh, arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.5, ls="--", mutation_scale=16))
ax.plot([fh[0] - 0.25, fh[0] - 0.25, fh[0]], [fh[1] - 0.07, fh[1] + 0.28, fh[1] + 0.35], color=INK, lw=1.2)
ax.text(*(fv + [0.15, 0.0]), "$f$", fontsize=22, va="bottom")
ax.text(3.0, 0.45, "$\\hat f$", fontsize=22, color=BLUE)
ax.text(4.75, 2.2, "$f - \\hat f$", fontsize=19, color=RED)
ax.text(0.3, 1.25, "$V$: degree $\\leq N$", fontsize=17, color="#456")
ax.set_xlim(0, 7); ax.set_ylim(-0.1, 4.0); ax.set_aspect("equal"); ax.axis("off")
save(f, f"{OUT}/fig-projection.svg")
