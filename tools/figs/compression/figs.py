"""Slide figures for the compression day (entropy, lossless, dct, jpeg, compress-science).

Run from the repository root:
    S=<scratch dir>
    python3 -I tools/figs/compression/figs.py celljson $S        # writes $S/cell.json
    node_modules/.bin/esbuild tools/figs/compression/dump.ts --bundle --platform=node --outfile=$S/dump.js
    node $S/dump.js $S                                             # demo code → $S/dump.json
    python3 -I tools/figs/compression/figs.py all $S [unit ...]

Numbers in the figures come from the demos' own code (dump.ts) or are computed here.
Micrograph: demos/src/fourier2d/cell256.png (Robert M. Hunt, CC BY 3.0, via Wikimedia Commons).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image
from scipy.fft import dct as sdct

ROOT = os.getcwd()
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

S = None
D = None


def out(unit, name):
    return os.path.join(ROOT, unit, name)


def cell():
    return np.asarray(Image.open(os.path.join(ROOT, "demos/src/fourier2d/cell256.png")), dtype=float)


def img(key):
    return np.asarray(D[key], dtype=float).reshape(256, 256)


def H(values):
    _, c = np.unique(np.asarray(values).ravel(), return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def grey(ax, a, vmin=0, vmax=255, title=None):
    ax.imshow(a, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=15)


# ------------------------------------------------------------------------------------------
# entropy

def entropy_figs():
    u = "entropy"
    # information of an outcome
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    p = np.logspace(np.log10(1 / 400), 0, 400)
    ax.plot(p, -np.log2(p), color=BLUE, lw=2.5)
    ax.plot([1 / 256], [8], "o", color=ORANGE, ms=8, zorder=3)
    ax.annotate("1 of 256 grey levels: 8 bits", (1 / 256, 8), xytext=(10, -2), textcoords="offset points", fontsize=15)
    pts = [(1 / 6, "die shows 6: 2.585 bits", 3.35), (0.5, "coin, heads: 1 bit", 2.45),
           (0.9, "p = 0.9: 0.15 bits", 1.45), (1.0, "sure thing: 0 bits", 0.45)]
    for pp, lab, ty in pts:
        v = -np.log2(pp) + 0.0
        ax.plot([pp], [v], "o", color=ORANGE, ms=8, zorder=3)
        ax.annotate(lab, (pp, v), xytext=(1 / 330, ty), fontsize=15, va="center",
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=1, shrinkB=5))
    ax.set_xscale("log")
    ax.set_xticks([1 / 256, 1 / 16, 1 / 4, 1]); ax.set_xticklabels(["1/256", "1/16", "1/4", "1"])
    ax.minorticks_off()
    ax.set_xlabel("probability p (log scale)"); ax.set_ylabel("information −log₂ p (bits)")
    ax.set_ylim(-0.4, 9.2)
    fig.savefig(out(u, "fig-information.svg")); plt.close(fig)

    # three histograms and their entropies
    hs = [("example", [0.4, 0.2, 0.2, 0.1, 0.1]), ("flat", [0.2] * 5), ("one dominant", [0.9, 0.05, 0.025, 0.025])]   # the huffman demo's preset (counts 36, 2, 1, 1)
    fig, axs = plt.subplots(1, 3, figsize=(5.0, 3.0), sharey=True)
    for ax, (lab, pr) in zip(axs, hs):
        pr = np.array(pr)
        h = -(pr * np.log2(pr)).sum()
        ax.bar(list("ABCDE"[:len(pr)]), pr, color=BLUE, width=0.7)
        ax.set_xlim(-0.6, 4.6)
        ax.set_title(f"{lab}\nH = {h:.2f} bits", fontsize=14)
        ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1])
    axs[0].set_ylabel("probability")
    fig.tight_layout()
    fig.savefig(out(u, "fig-entropyhist.svg")); plt.close(fig)

    # run-length encoding: a binary row and a noisy grey row
    row = [0] * 7 + [1] * 4 + [0] * 5
    rng = np.random.default_rng(4)
    noisy = np.clip(np.round(120 + 12 * rng.standard_normal(16)), 0, 255).astype(int)
    noisy[3] = noisy[2] + 1 if noisy[3] == noisy[2] else noisy[3]
    W, c = 44, 30
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {c + 16 * W + 20} 300" font-family="sans-serif" font-size="20">',
           f'<rect width="{c + 16 * W + 20}" height="300" fill="white"/>']
    y0 = 20
    for i, v in enumerate(row):
        svg.append(f'<rect x="{c + i * W}" y="{y0}" width="{W}" height="{W}" fill="{"#333" if v else "white"}" stroke="#888"/>')
        svg.append(f'<text x="{c + i * W + W / 2}" y="{y0 + W / 2 + 7}" text-anchor="middle" fill="{"white" if v else "#333"}">{v}</text>')
    x = 0
    for v, n in [(0, 7), (1, 4), (0, 5)]:
        x0, x1 = c + x * W + 3, c + (x + n) * W - 3
        yb = y0 + W + 10
        svg.append(f'<path d="M{x0},{yb} v8 H{x1} v-8" fill="none" stroke="{ORANGE}" stroke-width="2.5"/>')
        svg.append(f'<text x="{(x0 + x1) / 2}" y="{yb + 34}" text-anchor="middle" fill="{ORANGE}" font-weight="700">({v}, {n})</text>')
        x += n
    y1 = 175
    for i, v in enumerate(noisy):
        g = int(v)
        svg.append(f'<rect x="{c + i * W}" y="{y1}" width="{W}" height="{W}" fill="rgb({g},{g},{g})" stroke="#888"/>')
        svg.append(f'<text x="{c + i * W + W / 2}" y="{y1 + W / 2 + 6}" text-anchor="middle" font-size="16" fill="white">{v}</text>')
    for i in range(16):
        x0, x1 = c + i * W + 3, c + (i + 1) * W - 3
        svg.append(f'<path d="M{x0},{y1 + W + 8} v6 H{x1} v-6" fill="none" stroke="{RED}" stroke-width="2"/>')
    svg.append(f'<text x="{c}" y="{y1 + W + 45}" fill="{RED}">noisy grey row: 16 runs of length 1, each (value, 1): RLE makes it bigger</text>')
    svg.append(f'<text x="{c}" y="{y1 - 12}" fill="#333">mask row: 3 runs</text>')
    svg.append("</svg>")
    # move the 'mask row' label above the first row
    svg[-2] = f'<text x="{c}" y="{y0 + W + 92}" fill="#333">16 pixels → 3 runs: (0, 7) (1, 4) (0, 5)</text>'
    open(out(u, "fig-rle.svg"), "w").write("\n".join(svg) + "\n")

    # the Huffman tree with ties broken the other way (lengths 1, 2, 3, 4, 4), and decoding
    nodes = {"r": (170, 30), "n1": (250, 105), "n2": (330, 180), "n3": (410, 255)}
    leaves = {"A": (90, 105, "0.4", "0"), "B": (170, 180, "0.2", "10"), "C": (250, 255, "0.2", "110"),
              "D": (330, 330, "0.1", "1110"), "E": (490, 330, "0.1", "1111")}
    edges = [("r", "A", 0), ("r", "n1", 1), ("n1", "B", 0), ("n1", "n2", 1), ("n2", "C", 0), ("n2", "n3", 1), ("n3", "D", 0), ("n3", "E", 1)]
    pos = {k: v[:2] for k, v in nodes.items()}
    pos.update({k: v[:2] for k, v in leaves.items()})
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="40 0 500 400" font-family="sans-serif" font-size="24">',
           '<rect x="40" width="500" height="400" fill="white"/>']
    for a_, b_, bit in edges:
        (x0, y0_), (x1, y1_) = pos[a_], pos[b_]
        svg.append(f'<line x1="{x0}" y1="{y0_}" x2="{x1}" y2="{y1_}" stroke="#444" stroke-width="2"/>')
        col = BLUE if bit == 0 else ORANGE
        svg.append(f'<text x="{(x0 + x1) / 2 + (-16 if bit == 0 else 14)}" y="{(y0_ + y1_) / 2 - 2}" text-anchor="middle" font-weight="700" fill="{col}">{bit}</text>')
    for k, (x0, y0_) in nodes.items():
        svg.append(f'<circle cx="{x0}" cy="{y0_}" r="12" fill="white" stroke="#444" stroke-width="2"/>')
    for k, (x0, y0_, pr, code) in leaves.items():
        svg.append(f'<rect x="{x0 - 26}" y="{y0_ - 21}" width="52" height="42" rx="6" fill="#eaf2fb" stroke="{BLUE}" stroke-width="2"/>')
        svg.append(f'<text x="{x0}" y="{y0_ + 8}" text-anchor="middle" font-weight="700">{k}</text>')
        svg.append(f'<text x="{x0}" y="{y0_ + 50}" text-anchor="middle" fill="#333" font-family="monospace">{code}</text>')
    svg.append("</svg>")
    open(out(u, "fig-huffties.svg"), "w").write("\n".join(svg) + "\n")

    # the three redundancies on real images
    c = cell()
    sm = img("smooth")
    q50 = img("cell_q50")
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3), gridspec_kw={"width_ratios": [1.15, 1, 1.6]})
    ax = axs[0]
    hist = np.bincount(c.astype(int).ravel(), minlength=256) / c.size
    ax.fill_between(np.arange(256), hist, step="mid", color=BLUE, alpha=0.85)
    ax.set_xlabel("grey level"); ax.set_ylabel("share of pixels"); ax.set_yticks([])
    ax.set_title(f"coding: uneven histogram\nH = {H(c):.2f} of 8 bits", fontsize=14)
    ax = axs[1]
    a, b = sm[:, 1:].ravel(), sm[:, :-1].ravel()
    hh, xe, ye = np.histogram2d(b, a, bins=128, range=[[0, 256], [0, 256]])
    ax.imshow(np.log1p(hh.T), origin="lower", extent=[0, 256, 0, 256], cmap="Blues")
    ax.set_xlabel("left neighbour"); ax.set_ylabel("pixel")
    ax.set_title("inter-pixel: neighbours\nare alike", fontsize=14)
    ax.set_xticks([0, 128, 255]); ax.set_yticks([0, 128, 255])
    ax = axs[2]
    crop = (slice(70, 134), slice(120, 184))
    ax.imshow(np.hstack([c[crop], np.full((64, 3), 255), q50[crop]]), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"psychovisual: original | JPEG q 50\n({8 / D['rd'][9]['bpp']:.1f} times fewer bits)", fontsize=14)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out(u, "fig-redundancies.png"), dpi=130); plt.close(fig)
    print("cell H", H(c), "q50 ratio", 65536 / D["rd"][9]["bytes"])


# ------------------------------------------------------------------------------------------
# lossless

def lossless_figs():
    u = "lossless"
    sm = img("smooth")
    res = {k: np.asarray(v) for k, v in D["res"].items()}
    Hs = D["H"]
    # neighbours are alike: scatter density + two histograms
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axs[0]
    a, b = sm[:, 1:].ravel(), sm[:, :-1].ravel()
    hh, _, _ = np.histogram2d(b, a, bins=128, range=[[0, 256], [0, 256]])
    ax.imshow(np.log1p(hh.T), origin="lower", extent=[0, 256, 0, 256], cmap="Blues")
    ax.set_xlabel("left neighbour a"); ax.set_ylabel("pixel x")
    ax.set_xticks([0, 128, 255]); ax.set_yticks([0, 128, 255])
    ax = axs[1]
    hp = np.bincount(sm.astype(int).ravel(), minlength=256) / sm.size
    e = res["smooth_left"]
    he = np.bincount(e + 255, minlength=511) / e.size
    ax.fill_between(np.arange(256), hp, step="mid", color=BLUE, alpha=0.75, label=f"pixels x: H = {Hs['smooth']['pixels']:.2f} bits")
    ax.fill_between(np.arange(-255, 256), he, step="mid", color=ORANGE, alpha=0.75, label=f"differences x − a: H = {Hs['smooth']['left']:.2f} bits")
    ax.set_xlim(-40, 256); ax.set_ylim(0, 0.25); ax.set_yticks([0, 0.05, 0.1, 0.15])
    ax.set_xlabel("grey level / difference"); ax.set_ylabel("share of pixels")
    ax.legend(loc="upper right", frameon=False, fontsize=15)
    fig.tight_layout()
    fig.savefig(out(u, "fig-scatter.png"), dpi=130); plt.close(fig)

    # a row by hand
    rowv = np.array([100, 102, 104, 103, 105, 107, 106, 108])
    r = np.diff(rowv)
    fig, axs = plt.subplots(2, 1, figsize=(5.4, 3.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    axs[0].plot(range(8), rowv, "o-", color=BLUE, ms=8, lw=2)
    axs[0].set_ylabel("pixel"); axs[0].set_ylim(98, 110)
    for i in range(1, 8):
        axs[0].annotate("", (i, rowv[i]), (i - 1, rowv[i - 1]), arrowprops=dict(arrowstyle="-", color=GREY, ls=":"))
    axs[1].bar(range(1, 8), r, color=[ORANGE if v > 0 else PURPLE for v in r], width=0.6)
    axs[1].axhline(0, color="#444", lw=1)
    for i, v in enumerate(r, start=1):
        axs[1].text(i, v + (0.25 if v > 0 else -0.75), f"{v:+d}".replace("-", "−"), ha="center", fontsize=14)
    axs[1].set_ylabel("residual"); axs[1].set_ylim(-2, 3.2)
    axs[1].set_xlabel("position in the row"); axs[1].set_xticks(range(8))
    fig.tight_layout()
    fig.savefig(out(u, "fig-row.svg")); plt.close(fig)

    # edges are what remains: shapes, residual of left and of MED
    sh = img("shapes")
    def resmap(ax, e, title):
        ax.imshow(e.reshape(256, 256), cmap="RdBu_r", vmin=-32, vmax=32, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=19)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.4))
    grey(axs[0], sh); axs[0].set_title(f"shapes: {Hs['shapes']['pixels']:.2f} bits", fontsize=19)
    resmap(axs[1], res["shapes_left"], f"left: {Hs['shapes']['left']:.2f}")
    resmap(axs[2], res["shapes_med"], f"MED: {Hs['shapes']['med']:.2f}")
    fig.tight_layout()
    fig.savefig(out(u, "fig-edges.png"), dpi=120); plt.close(fig)

    # noise sets the floor: residual histograms (average predictor)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    for key, hk, lab, col in [("smooth_avg", "smooth", "smooth", BLUE), ("smoothN10_avg", "smoothN10", "smooth + noise 10", ORANGE), ("cell_avg", "cell", "cell micrograph", GREEN)]:
        e = res[key]
        h = np.bincount(e + 255, minlength=511) / e.size
        ax.plot(np.arange(-255, 256), h, color=col, lw=2.2, drawstyle="steps-mid", label=f"{lab}: {Hs[hk]['average']:.2f} bits")
    ax.set_xlim(-50, 50); ax.set_ylim(0, None)
    ax.set_xlabel("residual (average predictor)"); ax.set_ylabel("share of pixels")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), fontsize=14)
    fig.tight_layout()
    fig.savefig(out(u, "fig-noisefloor.svg")); plt.close(fig)


# ------------------------------------------------------------------------------------------
# dct

def dct2(b):
    return sdct(sdct(b, axis=0, norm="ortho"), axis=1, norm="ortho")


def idct2(c):
    from scipy.fft import idct
    return idct(idct(c, axis=0, norm="ortho"), axis=1, norm="ortho")


LUMA_Q = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                   14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                   49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]).reshape(8, 8)


def blocks_of(a):
    return a.reshape(32, 8, 32, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)


def unblock(bl):
    return bl.reshape(32, 32, 8, 8).transpose(0, 2, 1, 3).reshape(256, 256)


def real_dft_matrix(N=8):
    A = np.zeros((N, N)); n = np.arange(N)
    A[0] = 1 / np.sqrt(N); r = 1
    for k in range(1, N // 2):
        A[r] = np.sqrt(2 / N) * np.cos(2 * np.pi * k * n / N)
        A[r + 1] = np.sqrt(2 / N) * np.sin(2 * np.pi * k * n / N)
        r += 2
    A[r] = (-1.0) ** n / np.sqrt(N)
    return A


def dct_matrix(N=8):
    return sdct(np.eye(N), axis=0, norm="ortho")


def numgrid(ax, M, fmt, cmap, vmin, vmax, title, fs=11):
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    for i in range(8):
        for j in range(8):
            v = M[i, j]
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=fs,
                    color="white" if (cmap == "gray" and v < 110) else "#111")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=14)


def dct_figs():
    u = "dct"
    c = cell()
    # one block through the codec steps (block (14, 12), as the jpeg demo)
    bx, by = 14, 12
    blk = c[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8]
    C = dct2(blk - 128)
    Q = np.round(C / LUMA_Q)
    nz = int((Q != 0).sum())
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.75))
    lim = np.abs(C).max()
    axs[0].imshow(blk, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axs[1].imshow(C, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    axs[2].imshow(Q, cmap="RdBu_r", vmin=-6, vmax=6, interpolation="nearest")
    for ax, t in zip(axs, ["64 pixels", "64 DCT\ncoefficients", f"quantised:\n{nz} not 0"]):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=19)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for i in range(2):
        fig.text(0.355 + 0.322 * i, 0.45, "→", fontsize=26, ha="center", va="center")
    fig.tight_layout(w_pad=2.2)
    fig.savefig(out(u, "fig-steps.png"), dpi=120); plt.close(fig)
    e = (C ** 2).sum()
    srt = np.sort((C ** 2).ravel())[::-1]
    print("block energy share of 6 largest:", srt[:6].sum() / e, "nonzero", nz)

    # KLT of the micrograph's blocks, and compaction curves (as the dctblock demo)
    X = blocks_of(c).reshape(-1, 64) - 128
    Sm = X.T @ X / len(X)
    w, V = np.linalg.eigh(Sm)
    o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
    V = V * np.where(V.sum(0) < -1e-9, -1, 1)
    bases = {"DCT": np.kron(dct_matrix(), dct_matrix()), "DFT": np.kron(real_dft_matrix(), real_dft_matrix()), "KLT": V.T}
    curves = {}
    for k, A in bases.items():
        v = np.sort(np.einsum("ri,ij,rj->r", A, Sm, A))[::-1]
        curves[k] = 100 * np.cumsum(v) / v.sum()
    fig = plt.figure(figsize=(7.0, 3.4))
    gs = fig.add_gridspec(4, 9, wspace=0.12, hspace=0.12)
    for k in range(16):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        b = V[:, k].reshape(8, 8)
        m = np.abs(b).max()
        ax.imshow(b, cmap="gray", vmin=-m, vmax=m, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.text(0.3, 0.93, "first 16 KLT basis images", ha="center", fontsize=16)
    ax = fig.add_subplot(gs[:, 5:])
    ax.set_position([0.62, 0.17, 0.37, 0.72])
    ks = np.arange(1, 65)
    for k, col in [("KLT", GREEN), ("DCT", BLUE), ("DFT", RED)]:
        ax.plot(ks, curves[k], color=col, lw=2.5, label=k)
    ax.set_xlim(1, 24); ax.set_ylim(min(curves["DFT"][0], 80) - 1, 100.5)
    ax.set_xlabel("k coefficients kept"); ax.set_ylabel("energy kept (%)")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out(u, "fig-kltcurves.png"), dpi=120); plt.close(fig)
    print("curves k=1,2,4,8:", {k: [round(v[i], 1) for i in (0, 1, 3, 7)] for k, v in curves.items()})

    # why blocks: keep the largest 3 % of the coefficients
    cm = c - c.mean()
    tot = (cm ** 2).sum()
    def keep(coef, frac=0.03):
        t = np.sort(np.abs(coef).ravel())[::-1][int(round(frac * coef.size)) - 1]
        return np.where(np.abs(coef) >= t, coef, 0)
    Bc = np.array([dct2(b) for b in blocks_of(cm)])
    rb = unblock(np.array([idct2(b) for b in keep(Bc)]))
    Wc = dct2(cm)
    rw = idct2(keep(Wc))
    F = np.fft.fft2(cm)
    t = np.sort(np.abs(F).ravel())[::-1][int(round(0.03 * F.size)) - 1]
    rf = np.real(np.fft.ifft2(np.where(np.abs(F) >= t, F, 0)))
    sh = {k: 100 * (1 - ((cm - r) ** 2).sum() / tot) for k, r in [("block", rb), ("whole", rw), ("dft", rf)]}
    print("3% kept:", sh)
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 3.35))
    crop = (slice(64, 160), slice(96, 192))
    grey(axs[0], np.clip(rb + c.mean(), 0, 255)[crop])
    grey(axs[1], np.clip(rw + c.mean(), 0, 255)[crop])
    axs[0].set_title("8 × 8 blocks", fontsize=20); axs[1].set_title("whole image", fontsize=20)
    fig.tight_layout()
    fig.savefig(out(u, "fig-blocks.png"), dpi=120); plt.close(fig)


# ------------------------------------------------------------------------------------------
# jpeg

def jpeg_figs():
    u = "jpeg"
    # the luminance table
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.imshow(LUMA_Q, cmap="Oranges", vmin=0, vmax=130)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, str(LUMA_Q[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_xlabel("u: horizontal frequency →"); ax.set_ylabel("v: vertical frequency →")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(out(u, "fig-qtable.svg")); plt.close(fig)

    # zigzag with the example block's quantised values
    order = sorted(((i, j) for i in range(8) for j in range(8)), key=lambda p: (p[0] + p[1], p[0] if (p[0] + p[1]) % 2 else p[1]))
    vals = {(0, 0): 12, (0, 1): -3, (1, 0): 2, (2, 0): -1, (1, 1): 0, (0, 2): 1}
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    M = np.zeros((8, 8))
    for (i, j), v in vals.items():
        M[i, j] = v
    ax.imshow(np.where(M != 0, 1, 0), cmap="Blues", vmin=0, vmax=3)
    xs = [p[1] for p in order]; ys = [p[0] for p in order]
    ax.plot(xs, ys, color=ORANGE, lw=2, alpha=0.9)
    for n, (i, j) in enumerate(order):
        v = vals.get((i, j), 0)
        ax.text(j, i, f"{v}".replace("-", "−"), ha="center", va="center", fontsize=14,
                fontweight="bold" if v else "normal", color="#111" if v else "#888",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-0.5, 7.5); ax.set_ylim(7.5, -0.5)
    fig.savefig(out(u, "fig-zigzag.svg")); plt.close(fig)

    # rate-distortion of the micrograph
    rd = D["rd"]
    b = [r["bpp"] for r in rd]; p = [r["psnr"] for r in rd]; s = [r["ssim"] for r in rd]
    fig, axs = plt.subplots(2, 1, figsize=(5.4, 5.0), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    axs[0].plot(b, p, "o-", color=BLUE, lw=2.2, ms=4)
    axs[1].plot(b, s, "o-", color=GREEN, lw=2.2, ms=4)
    for r in rd:
        if r["q"] in (10, 50, 90):
            axs[0].plot([r["bpp"]], [r["psnr"]], "o", color=ORANGE, ms=9, zorder=3)
            axs[0].annotate(f"q {r['q']}", (r["bpp"], r["psnr"]), xytext=(16, -20) if r["q"] == 10 else (-14, 9), textcoords="offset points", fontsize=15, ha="center")
            axs[1].plot([r["bpp"]], [r["ssim"]], "o", color=ORANGE, ms=9, zorder=3)
    axs[0].set_ylabel("PSNR (dB)"); axs[1].set_ylabel("SSIM")
    axs[1].set_xlabel("rate (bits per pixel)")
    axs[0].set_xlim(0, 8.8); axs[0].set_ylim(14, 62)
    fig.tight_layout()
    fig.savefig(out(u, "fig-rd.svg")); plt.close(fig)
    print("rd max bpp", max(b), "q100 psnr", p[-1])

    # same MSE, different structure
    c = cell()
    sm = D["same"]
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.9))
    crop = (slice(40, 168), slice(60, 188))
    grey(axs[0], c[crop], title="original")
    grey(axs[1], np.asarray(sm["shift"]["img"]).reshape(256, 256)[crop], title=f"brightness shift\nMSE {sm['shift']['mse']:.0f}\nSSIM {sm['shift']['ssim']:.3f}")
    grey(axs[2], np.asarray(sm["blur"]["img"]).reshape(256, 256)[crop], title=f"blur\nMSE {sm['blur']['mse']:.0f}\nSSIM {sm['blur']['ssim']:.3f}")
    axs[0].set_title("original\n\n ", fontsize=15)
    fig.tight_layout()
    fig.savefig(out(u, "fig-ssim.png"), dpi=120); plt.close(fig)


# ------------------------------------------------------------------------------------------
# compress-science

def lift53(x):
    """One level of the reversible integer 5/3 lifting along the last axis (even length)."""
    x = x.astype(np.int64)
    ev, od = x[..., 0::2], x[..., 1::2]
    evn = np.concatenate([ev[..., 1:], ev[..., -1:]], axis=-1)        # symmetric extension
    d = od - np.floor_divide(ev + evn, 2)
    dp = np.concatenate([d[..., :1], d[..., :-1]], axis=-1)
    s = ev + np.floor_divide(dp + d + 2, 4)
    return s, d


def unlift53(s, d):
    dp = np.concatenate([d[..., :1], d[..., :-1]], axis=-1)
    ev = s - np.floor_divide(dp + d + 2, 4)
    evn = np.concatenate([ev[..., 1:], ev[..., -1:]], axis=-1)
    od = d + np.floor_divide(ev + evn, 2)
    x = np.empty(s.shape[:-1] + (2 * s.shape[-1],), dtype=np.int64)
    x[..., 0::2], x[..., 1::2] = ev, od
    return x


def science_figs():
    u = "compress-science"
    sc, q90, q30, wav = img("scene"), img("scene_q90"), img("scene_q30"), img("scene_wav")
    cv = {r["q"]: r for r in D["curve"]}
    crop = (slice(56, 152), slice(32, 128))   # the 96 × 96 window with the most dim spots (9 of 20)
    vmax = np.percentile(sc, 99.7)
    vmin = np.percentile(sc, 1)
    # images are data: original, q90, q30, wavelet at the size of q30
    fig, axs = plt.subplots(2, 2, figsize=(6.2, 6.8))
    for ax, a, t in [(axs[0, 0], sc, "original (12 bit)"), (axs[0, 1], q90, f"JPEG q 90: {cv[90]['dim']:.1f}%"),
                     (axs[1, 0], q30, f"JPEG q 30: {cv[30]['dim']:.1f}%"), (axs[1, 1], wav, f"wavelet, same size: {D['wavelet']['dim']:.1f}%")]:
        grey(ax, a[crop], vmin, vmax)
        ax.set_title(t, fontsize=17)
    fig.tight_layout()
    fig.savefig(out(u, "fig-damage4.png"), dpi=120); plt.close(fig)

    # the spectrum of |neighbour differences|
    fig, ax = plt.subplots(figsize=(4.6, 4.7))
    P = np.fft.fftshift(np.asarray(D["spec_q30"]).reshape(256, 256))
    L = np.log10(P + 1)
    ax.imshow(L, cmap="magma", vmin=np.percentile(L, 1), vmax=np.percentile(L, 99.95), extent=[-128, 128, 128, -128], interpolation="nearest")
    for f in (32, 64, 96):
        for fx, fy in [(f, 0), (-f, 0), (0, f), (0, -f)]:
            ax.add_patch(plt.Circle((fx, fy), 7, fill=False, ec="#4fc3f7", lw=2))
    ax.set_xticks([-128, -64, 0, 64, 128]); ax.set_yticks([-128, -64, 0, 64, 128])
    ax.set_xlabel("horizontal frequency (cycles / 256 px)"); ax.set_ylabel("vertical frequency")
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(out(u, "fig-spectrum.png"), dpi=120); plt.close(fig)

    # test your own measurement: dim-spot change and spots found against the size
    cur = D["curve"]
    b = [r["bpp"] for r in cur]
    fig, axs = plt.subplots(2, 1, figsize=(5.4, 5.0), sharex=True)
    axs[0].plot(b, [r["dim"] for r in cur], "o-", color=BLUE, lw=2.2, ms=4, label="JPEG")
    axs[0].plot([D["wavelet"]["bpp"]], [D["wavelet"]["dim"]], "s", color=GREEN, ms=9, label="wavelet")
    axs[0].set_ylabel("dim spots:\n|change| (%)")
    axs[1].plot(b, [r["found"] for r in cur], "o-", color=BLUE, lw=2.2, ms=4)
    axs[1].plot([D["wavelet"]["bpp"]], [D["wavelet"]["found"]], "s", color=GREEN, ms=9)
    axs[1].axhline(60, color=GREY, ls="--", lw=1.2)
    axs[1].text(4.5, 56.5, "60 true spots", color=GREY, fontsize=14, va="top", ha="right")
    axs[1].set_ylabel("spots found")
    axs[1].set_xscale("log"); axs[1].set_xticks([0.2, 0.5, 1, 2, 5]); axs[1].set_xticklabels(["0.2", "0.5", "1", "2", "5"])
    axs[1].set_xlabel("bits per pixel (lossless: 6.8)")
    for q in (10, 30, 90):
        axs[0].annotate(f"q {q}", (cv[q]["bpp"], cv[q]["dim"]), xytext=(6, 4), textcoords="offset points", fontsize=12)
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out(u, "fig-curve.svg")); plt.close(fig)
    print("lossless bpp", 8 * D["lossless"] / 65536)

    # JPEG 2000: one stream, every resolution (integer 5/3, decoded LL at each level)
    c = cell().astype(np.int64)
    ll = c
    levels = [c]
    for _ in range(3):
        s, d = lift53(ll)                 # rows
        s2, _ = lift53(s.swapaxes(0, 1))  # columns of the low band
        ll = s2.swapaxes(0, 1)
        levels.append(ll)
    # check exact reversibility of one level on the full image
    s, d = lift53(c); assert np.array_equal(unlift53(s, d), c)
    widths = [256, 128, 64, 32]
    gap = 24
    tot = sum(widths) + gap * (len(widths) - 1)
    fw, ih, lab = 10.5, 10.5 * 256 / tot, 0.6
    fig = plt.figure(figsize=(fw, ih + lab))
    x = 0.0
    for a, wd in zip(levels, widths):
        ax = fig.add_axes([x / tot, lab / (ih + lab), wd / tot, ih * wd / 256 / (ih + lab)])
        ax.imshow(a, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel(f"{a.shape[0]}²" if wd < 64 else f"{a.shape[0]} × {a.shape[1]}", fontsize=22)
        x += wd + gap
    fig.savefig(out(u, "fig-resolutions.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    mode, S = sys.argv[1], sys.argv[2]
    if mode == "celljson":
        json.dump(cell().astype(int).ravel().tolist(), open(os.path.join(S, "cell.json"), "w"))
        sys.exit()
    D = json.load(open(os.path.join(S, "dump.json")))
    units = sys.argv[3:] or ["entropy", "lossless", "dct", "jpeg", "science"]
    for un in units:
        globals()[un.replace("-", "_") + "_figs"]()
