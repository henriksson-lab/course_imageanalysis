"""Static figures for the topic page topics/eigenvectors.qmd.

Run from the repository root:  python3 -I tools/figs/topics-eigenvectors/figs.py
Writes topics/fig-eig-ellipses.svg, fig-eig-planes.svg, fig-eig-chain.svg,
fig-eig-pca-dct.png, fig-eig-ties.svg, and prints every number quoted on the page.
Micrograph: demos/src/fourier2d/cell256.png (R. M. Hunt, CC BY 3.0, Wikimedia Commons).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from PIL import Image
from scipy.ndimage import gaussian_filter, rotate

OUT = "topics"
FONT = 15
plt.rcParams.update({
    "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
    "xtick.labelsize": FONT - 3, "ytick.labelsize": FONT - 3, "legend.fontsize": FONT - 3,
    "svg.fonttype": "path", "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white",
})
BLUE, ORANGE, GREEN, RED, PURPLE, CYAN, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#777777")


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.05, **kw)
    if os.environ.get("PREVIEW"):          # optional PNG copies for checking by eye
        fig.savefig(os.path.join(os.environ["PREVIEW"], name.rsplit(".", 1)[0] + ".png"), dpi=70,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def cell():
    return np.asarray(Image.open("demos/src/fourier2d/cell256.png").convert("L")).astype(float)


def eig2(M):
    w, V = np.linalg.eigh(M)
    return w[::-1], V[:, ::-1]          # largest first


def axes_arrows(ax, c, w, V, scale, cols=(ORANGE, PURPLE), lw=3):
    for k in range(2):
        L = 0.8 * scale * np.sqrt(max(w[k], 0))
        ax.annotate("", xy=(c[0] + L * V[0, k], c[1] + L * V[1, k]), xytext=c,
                    arrowprops=dict(arrowstyle="-|>", color=cols[k], lw=lw, mutation_scale=16),
                    annotation_clip=False)


def ellipse(ax, c, w, V, scale, **kw):
    ang = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
    ax.add_patch(Ellipse(c, 2 * scale * np.sqrt(max(w[0], 0)), 2 * scale * np.sqrt(max(w[1], 0)),
                         angle=ang, fill=False, ls="--", lw=1.5, ec="k", **kw))


# 1. One picture, four meanings ----------------------------------------------------------------
def fig_ellipses():
    rng = np.random.default_rng(1)
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.1))
    # (a) PCA: neighbouring pixel pairs of the micrograph
    img = cell() / 255
    ys, xs = rng.integers(0, 256, 300), rng.integers(0, 255, 300)
    P = np.stack([img[ys, xs], img[ys, xs + 1]], 1)
    S = np.cov(P.T, bias=True)
    w, V = eig2(S)
    ax = axs[0]
    ax.scatter(P[:, 0], P[:, 1], s=8, c=BLUE, alpha=0.6, lw=0)
    c = P.mean(0)
    ellipse(ax, c, w, V, 2); axes_arrows(ax, c, w, V, 2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("pixel 1"); ax.set_ylabel("pixel 2 (its neighbour)")
    ax.set_title("PCA: two-pixel images")
    print(f"[ellipses] PCA neighbouring pixels: eigenvalues {w}, PC1 share {w[0] / w.sum():.3f}, "
          f"PC1 angle {np.degrees(np.arctan2(abs(V[1, 0]), abs(V[0, 0]))):.1f} deg")
    # (b) structure tensor: gradients in a window on stripes at 30 degrees
    yy, xx = np.mgrid[0:64, 0:64].astype(float)
    a = np.radians(30)
    st = 100 + 50 * np.cos(2 * np.pi * (xx * np.cos(a) + yy * np.sin(a)) / 10) + 8 * rng.standard_normal((64, 64))
    st = gaussian_filter(st, 1.0)
    gy, gx = np.gradient(st)
    G = np.stack([gx[16:48, 16:48].ravel(), gy[16:48, 16:48].ravel()], 1)
    J = G.T @ G / len(G)
    w, V = eig2(J)
    ax = axs[1]
    ax.scatter(G[:, 0], G[:, 1], s=6, c=BLUE, alpha=0.5, lw=0)
    ellipse(ax, (0, 0), w, V, 2); axes_arrows(ax, (0, 0), w, V, 2)
    m = np.abs(G).max() * 1.05
    ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_aspect("equal")
    ax.set_xlabel("gradient x"); ax.set_ylabel("gradient y")
    ax.set_title("Structure tensor: gradients")
    print(f"[ellipses] structure tensor stripes 30deg: mu {w}, coherence {((w[0] - w[1]) / (w[0] + w[1])) ** 2:.3f}, "
          f"e1 angle {np.degrees(np.arctan2(V[1, 0], V[0, 0])) % 180:.1f} deg")
    # (c) moments: pixel coordinates of an elongated object
    H = 80
    yy, xx = np.mgrid[0:H, 0:H].astype(float)
    u = (xx - 40) * np.cos(np.radians(35)) + (yy - 40) * np.sin(np.radians(35))
    v = -(xx - 40) * np.sin(np.radians(35)) + (yy - 40) * np.cos(np.radians(35))
    mask = (u / 30) ** 2 + (v / 12) ** 2 + 0.25 * np.sin(u / 4) * (v / 12) < 1
    pys, pxs = np.nonzero(mask)
    Q = np.stack([pxs, pys], 1).astype(float)
    C = np.cov(Q.T, bias=True)
    w, V = eig2(C)
    ax = axs[2]
    ax.imshow(mask, cmap="Greys", vmin=0, vmax=2.2, origin="lower", interpolation="nearest")
    c = Q.mean(0)
    ellipse(ax, c, w, V, 2); axes_arrows(ax, c, w, V, 2)
    ax.set_xlim(0, H); ax.set_ylim(0, H); ax.set_aspect("equal")
    ax.set_xlabel("x (pixels)"); ax.set_ylabel("y")
    ax.set_title("Moments: an object's pixels")
    print(f"[ellipses] moments: area {mask.sum()}, eigenvalues {w}, orientation "
          f"{np.degrees(np.arctan2(V[1, 0], V[0, 0])) % 180:.1f} deg, axis ratio {np.sqrt(w[0] / w[1]):.2f}")
    # (d) Hessian: a bright ridge as a landscape
    yy, xx = np.mgrid[-20:21, -20:21].astype(float)
    b = np.radians(-25)
    d = -xx * np.sin(b) + yy * np.cos(b)
    t = xx * np.cos(b) + yy * np.sin(b)
    Z = np.exp(-d ** 2 / (2 * 4.0 ** 2)) * (1 + 0.004 * t)
    Zy = np.gradient(Z, axis=0); Zx = np.gradient(Z, axis=1)
    Hm = np.array([[np.gradient(Zx, axis=1)[20, 20], np.gradient(Zx, axis=0)[20, 20]],
                   [np.gradient(Zy, axis=1)[20, 20], np.gradient(Zy, axis=0)[20, 20]]])
    Hm = (Hm + Hm.T) / 2
    hw, hV = np.linalg.eigh(Hm)                 # most negative first
    ax = axs[3]
    ax.contourf(xx, yy, Z, levels=12, cmap="Greys_r")
    for k, col in [(0, ORANGE), (1, PURPLE)]:
        L = 14 if k == 0 else 14 * max(abs(hw[1]) / abs(hw[0]), 0.08)
        ax.annotate("", xy=(L * hV[0, k], L * hV[1, k]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=3, mutation_scale=16))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Hessian: a bright ridge")
    print(f"[ellipses] Hessian ridge: eigenvalues {hw}")
    fig.tight_layout()
    save(fig, "fig-eig-ellipses.svg")


# 2. Reading two eigenvalues: structure tensor and Hessian --------------------------------------
def patch(kind):
    yy, xx = np.mgrid[-15:16, -15:16].astype(float)
    if kind == "flat":
        return np.zeros_like(xx)
    if kind == "edge":
        return (xx > 0).astype(float)
    if kind == "corner":
        return ((xx > 0) & (yy > 0)).astype(float)
    if kind == "bright blob":
        return np.exp(-(xx ** 2 + yy ** 2) / 50)
    if kind == "dark blob":
        return -np.exp(-(xx ** 2 + yy ** 2) / 50)
    if kind == "bright line":
        return np.exp(-xx ** 2 / 12)
    if kind == "saddle":
        return (xx ** 2 - yy ** 2) / 450
    raise ValueError(kind)


def inset(ax, xy, kind, size=0.2):
    ia = ax.inset_axes([xy[0] - size / 2, xy[1] - size / 2, size, size], transform=ax.transAxes)
    p = patch(kind)
    pad = 0.5 if np.ptp(p) == 0 else 1e-9
    ia.imshow(p, cmap="gray", vmin=p.min() - pad, vmax=p.max() + pad)
    ia.set_xticks([]); ia.set_yticks([])
    for s in ia.spines.values():
        s.set_color(GREY)
    ax.text(xy[0], xy[1] + size / 2 + 0.015, kind, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=FONT - 2, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))


def fig_planes():
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.8))
    ax = axs[0]
    x = np.linspace(0, 1, 50)
    ax.fill_between(x, x, 1, color="#eeeeee", lw=0)                       # impossible: mu2 > mu1
    ax.add_patch(Rectangle((0, 0), 0.2, 0.2, color=CYAN, alpha=0.4, lw=0))
    xe = np.linspace(0.2, 1, 30)
    ax.fill_between(xe, 0, 0.2 * xe, color=ORANGE, alpha=0.4, lw=0)
    xc = np.linspace(0.45, 1, 30)
    ax.fill_between(xc, 0.45, xc, color=GREEN, alpha=0.4, lw=0)
    ax.plot([0, 1], [0, 1], color=GREY, lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("larger eigenvalue μ₁"); ax.set_ylabel("smaller eigenvalue μ₂")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Structure tensor (both ≥ 0)")
    inset(ax, (0.11, 0.1), "flat", size=0.15); inset(ax, (0.84, 0.1), "edge", size=0.15); inset(ax, (0.68, 0.72), "corner")
    ax.text(0.25, 0.8, "not possible:\nμ₂ ≤ μ₁", ha="center", va="center", transform=ax.transAxes,
            color=GREY, fontsize=FONT - 2)
    ax = axs[1]
    ax.fill_between([-1, 0], -1, 0, color=ORANGE, alpha=0.3, lw=0)
    ax.fill_between([0, 1], 0, 1, color=BLUE, alpha=0.2, lw=0)
    ax.fill_between([-1, 0], 0, 1, color=PURPLE, alpha=0.18, lw=0)
    ax.fill_between([0, 1], -1, 0, color=PURPLE, alpha=0.18, lw=0)
    ax.fill_between([-1, 1], -0.08, 0.08, color=GREEN, alpha=0.45, lw=0)
    ax.axhline(0, color=GREY, lw=1); ax.axvline(0, color=GREY, lw=1)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
    ax.set_xlabel("curvature across the structure"); ax.set_ylabel("curvature along it")
    ax.set_xticks([-1, 0, 1]); ax.set_xticklabels(["< 0", "0", "> 0"])
    ax.set_yticks([-1, 0, 1]); ax.set_yticklabels(["< 0", "0", "> 0"])
    ax.set_title("Hessian (signed curvatures)")
    inset(ax, (0.2, 0.16), "bright blob"); inset(ax, (0.8, 0.76), "dark blob")
    inset(ax, (0.2, 0.5), "bright line", size=0.16); inset(ax, (0.8, 0.16), "saddle")
    fig.tight_layout()
    save(fig, "fig-eig-planes.svg")
    for k in ["edge", "corner", "flat"]:
        p = gaussian_filter(patch(k), 1.0)
        gy, gx = np.gradient(p)
        G = np.stack([gx[10:21, 10:21].ravel(), gy[10:21, 10:21].ravel()], 1)
        w, _ = eig2(G.T @ G / len(G))
        print(f"[planes] structure tensor of '{k}' window (11x11 at the centre): mu = {w}")


# 3. Laplacian eigenvectors: waves, and the Fiedler vector ---------------------------------------
def chain_laplacian(wts):
    n = len(wts) + 1
    L = np.zeros((n, n))
    for i, w in enumerate(wts):
        L[i, i] += w; L[i + 1, i + 1] += w
        L[i, i + 1] -= w; L[i + 1, i] -= w
    return L


def fig_chain():
    n = 12
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
    L = chain_laplacian(np.ones(n - 1))
    w, V = np.linalg.eigh(L)
    ax = axs[0]
    x = np.arange(1, n + 1)
    for k, col in [(1, ORANGE), (2, PURPLE), (3, GREEN)]:
        v = V[:, k] * np.sign(V[0, k])
        ax.plot(x, v, "-o", color=col, lw=2, ms=6, label=f"k = {k}:  eigenvalue {w[k]:.2f}")
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xlabel("node of the chain"); ax.set_ylabel("eigenvector entry")
    ax.set_title("Uniform chain: the eigenvectors are waves")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=FONT - 4)
    ax.set_xticks(x)
    print(f"[chain] uniform chain n={n}: eigenvalues {np.round(w, 3)}; 2-2cos(pi k/n) = "
          f"{np.round(2 - 2 * np.cos(np.pi * np.arange(n) / n), 3)}")
    wts = np.ones(n - 1); wts[6] = 0.05
    L2 = chain_laplacian(wts)
    w2, V2 = np.linalg.eigh(L2)
    f = V2[:, 1] * np.sign(V2[-1, 1])
    ax = axs[1]
    ax.bar(x, f, color=[BLUE if v < 0 else ORANGE for v in f], width=0.6)
    ax.axhline(0, color=GREY, lw=1)
    for i in range(n - 1):
        ax.plot([x[i] + 0.3, x[i + 1] - 0.3], [-0.62, -0.62], color="k", lw=3 * wts[i] + 0.4,
                ls="-" if wts[i] > 0.5 else ":")
    ax.text(7.5, -0.56, "weak link (0.05)", ha="center", va="bottom", fontsize=FONT - 3, color=RED)
    ax.set_ylim(-0.7, 0.55)
    ax.set_xticks(x)
    ax.set_xlabel("node (links drawn below; thick = strong)")
    ax.set_title(f"A weak link: the Fiedler vector jumps there")
    fig.tight_layout()
    save(fig, "fig-eig-chain.svg")
    print(f"[chain] weak link between nodes 7 and 8: eigenvalues {np.round(w2[:4], 4)}; "
          f"Fiedler entries {np.round(f, 3)}; sign split {np.sum(f < 0)} | {np.sum(f > 0)}")


# 4. PCA of 8x8 patches of the micrograph vs the DCT ---------------------------------------------
def dct_basis(n=8):
    B = np.zeros((n, n))
    for k in range(n):
        B[k] = np.cos(np.pi * (np.arange(n) + 0.5) * k / n)
        B[k] /= np.linalg.norm(B[k])
    return B


def fig_pca_dct():
    img = cell()
    P = np.array([img[i:i + 8, j:j + 8].ravel() for i in range(0, 249, 2) for j in range(0, 249, 2)])
    Y = P - P.mean(0)
    S = Y.T @ Y / len(Y)
    w, V = np.linalg.eigh(S)
    w, V = w[::-1], V[:, ::-1]
    B = dct_basis()
    # DCT basis images ordered by the variance they carry on this image
    D = np.array([np.outer(B[a], B[b]).ravel() for a in range(8) for b in range(8)])
    var_d = np.array([d @ S @ d for d in D])
    order = np.argsort(-var_d)
    k = 10
    share_pca = w[:k].sum() / w.sum()
    share_dct = var_d[order[:k]].sum() / var_d.sum()
    fig, axs = plt.subplots(2, k, figsize=(14, 3.4))
    for i in range(k):
        v = V[:, i].reshape(8, 8)
        d = D[order[i]].reshape(8, 8)
        # match sign for easier comparison
        if (v * d).sum() < 0:
            v = -v
        for r, (im, lab) in enumerate([(v, f"{100 * w[i] / w.sum():.1f}%"), (d, f"{100 * var_d[order[i]] / var_d.sum():.1f}%")]):
            ax = axs[r, i]
            m = np.abs(im).max()
            ax.imshow(im, cmap="RdBu_r", vmin=-m, vmax=m, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(lab, fontsize=FONT - 4, pad=2)
    axs[0, 0].set_ylabel("PCA", fontsize=FONT - 1)
    axs[1, 0].set_ylabel("DCT", fontsize=FONT - 1)
    fig.tight_layout(h_pad=0.6, w_pad=0.3)
    save(fig, "fig-eig-pca-dct.png", dpi=110)
    print(f"[pca-dct] {len(P)} patches 8x8; first {k} PCs carry {100 * share_pca:.1f}% of the variance; "
          f"best {k} DCT images carry {100 * share_dct:.1f}%")
    print(f"[pca-dct] PC1 {100 * w[0] / w.sum():.1f}%, DCT DC {100 * var_d[0] / var_d.sum():.1f}%")


# 5. Ties: when two eigenvalues are nearly equal the direction is not defined --------------------
def fig_ties():
    rng = np.random.default_rng(7)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    stats = []
    for ax, sd, title in [(axs[0], (2.0, 0.7), "elongated cloud (axis ratio ≈ 3)"),
                          (axs[1], (1.0, 0.92), "nearly round cloud (ratio ≈ 1.1)")]:
        angles = []
        for r in range(30):
            X = rng.standard_normal((40, 2)) * np.array(sd)
            S = np.cov(X.T, bias=True)
            w, V = eig2(S)
            a = np.degrees(np.arctan2(V[1, 0], V[0, 0])) % 180
            angles.append(a)
            if r == 0:
                ax.scatter(X[:, 0], X[:, 1], s=14, c=BLUE, alpha=0.7, lw=0, zorder=3)
            ax.plot([-3 * V[0, 0], 3 * V[0, 0]], [-3 * V[1, 0], 3 * V[1, 0]], color=ORANGE, alpha=0.45, lw=1.5)
        a = np.array(angles)
        dev = (a + 90) % 180 - 90
        stats.append(dev)
        ax.set_xlim(-4.2, 4.2); ax.set_ylim(-3.2, 3.2); ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.02, 0.02, f"PC1 over 30 samples of 40 points:\nspread (sd) {dev.std():.0f}°",
                transform=ax.transAxes, fontsize=FONT - 3, va="bottom")
        print(f"[ties] {title}: PC1 angle sd over 30 resamples = {dev.std():.1f} deg, range {dev.min():.0f}..{dev.max():.0f}")
    fig.tight_layout()
    save(fig, "fig-eig-ties.svg")


if __name__ == "__main__":
    fig_ellipses()
    fig_planes()
    fig_chain()
    fig_pca_dct()
    fig_ties()
