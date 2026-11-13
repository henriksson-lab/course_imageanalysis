"""Static figures for the scale-space/ deck (slide pictures).

Run from the repository root:  python3 -I tools/figs/scale-space/figs.py
Writes scale-space/fig-scalespace1d.png, fig-box-vs-gauss.svg, fig-zerocross.svg,
fig-size-scale.png, fig-blob-detect.png, fig-dog-log.svg, fig-pyramid.png.
The 1D signals are those of the scale1d demo (demos/src/scale1d/signals.ts); kernels as in
demos/src/lib/edges.ts (numpy port in tools/figs/canny/edgelib.py).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "canny"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from edgelib import (gauss_kernel, corr1d, blur, gderiv, cell, show, geom_scales,
                     BLUE, ORANGE, GREEN, RED, CYAN, GREY, PURPLE, FONT)

OUT = "scale-space"
N = 256
BUMPS = [(40, 2), (100, 6), (180, 18)]


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.05, **kw)
    plt.close(fig)
    print("wrote", name)


def signal(kind):
    x = np.arange(N)
    if kind == "bumps":
        f = np.full(N, 40.0)
        for c, a in BUMPS:
            f[(x >= c - a) & (x < c + a)] += 100
        return f
    if kind == "boxdemo":
        f = np.full(N, 60.0)
        f[(x >= 108) & (x <= 148)] -= 40
        f[(x >= 124) & (x <= 131)] += 90
        return f
    raise ValueError(kind)


def extrema(a, eps=1e-6):
    out, prev, prev_at = [], 0, 0
    for x in range(1, len(a)):
        d = a[x] - a[x - 1]
        sg = 1 if d > eps else -1 if d < -eps else 0
        if not sg:
            continue
        if prev and sg != prev:
            out.append(round((prev_at + x - 1) / 2))
        prev, prev_at = sg, x
    return out


def box_kernel(s):
    half = max(0, round((np.sqrt(12) * s - 1) / 2))
    return np.full(2 * half + 1, 1 / (2 * half + 1))


# 1. 1D scale space --------------------------------------------------------------------
def fig_scalespace1d():
    f = signal("bumps")
    sig = geom_scales(0.5, 32, 64)
    stack = np.array([corr1d(f, gauss_kernel(s)) for s in sig])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.6), gridspec_kw={"height_ratios": [1, 2.2], "hspace": 0.12},
                                 sharex=True)
    a1.plot(f, color=GREY, lw=1.5, label="signal")
    a1.plot(stack[np.argmin(np.abs(np.array(sig) - 6))], color=BLUE, lw=2, label="blurred, σ = 6")
    a1.set_ylabel("value")
    a1.set_yticks([40, 140])
    a1.legend(frameon=False, loc="upper left", fontsize=FONT - 3, ncol=2, bbox_to_anchor=(0, 1.25))
    lo, hi = np.log10(sig[0]), np.log10(sig[-1])
    a2.imshow(stack, aspect="auto", cmap="gray", origin="lower", extent=[-0.5, N - 0.5, lo, hi])
    for j, s in enumerate(sig):
        ex = extrema(stack[j])
        a2.plot(ex, [np.log10(s)] * len(ex), ".", color=ORANGE, ms=3.5)
    a2.set_yticks(np.log10([0.5, 1, 2, 4, 8, 16, 32]))
    a2.set_yticklabels(["0.5", "1", "2", "4", "8", "16", "32"])
    a2.set_ylabel("σ (log scale)")
    a2.set_xlabel("x (pixels)")
    for c, a in BUMPS:
        a1.text(c + a + 3, 128, f"a = {a}", ha="left", va="center", fontsize=FONT - 3, color="k")
    save(fig, "fig-scalespace1d.png", dpi=130)


# 2. Box kernel creates structure ------------------------------------------------------
def fig_box_vs_gauss():
    f = signal("boxdemo")
    s = 8
    g = corr1d(f, gauss_kernel(s))
    kb = box_kernel(s)
    b = corr1d(f, kb)
    eg, eb, ef = extrema(g), extrema(b), extrema(f)
    print("  box demo: extrema signal", len(ef), "gauss", len(eg), eg, "box", len(eb), eb, "box width", len(kb))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(N)
    ax.plot(x, f, color=GREY, lw=1.5, label="signal")
    ax.plot(x, g, color=BLUE, lw=2.4, label=f"Gaussian, σ = {s}: {len(eg)} extrema")
    ax.plot(x, b, color=ORANGE, lw=2.4, ls="--", label=f"box, {len(kb)} px wide: {len(eb)} extrema")
    ax.plot(eb, b[eb], "o", color=ORANGE, ms=7)
    ax.plot(eg, g[eg], "o", color=BLUE, ms=6)
    ax.set_xlim(70, 186)
    ax.set_ylim(0, 150)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("value")
    ax.legend(frameon=False, loc="upper left", fontsize=FONT - 3)
    save(fig, "fig-box-vs-gauss.svg")


# 3. Zero crossings of the second derivative at an edge ---------------------------------
def fig_zerocross():
    x = np.arange(-30, 31)
    f = np.where(x < 0, 40.0, 160.0)
    s = 3
    L = corr1d(f, gauss_kernel(s))
    Lx = corr1d(f, gauss_kernel(s, 1))
    Lxx = corr1d(f, gauss_kernel(s, 2))
    fig, axs = plt.subplots(3, 1, figsize=(6.6, 5.8), sharex=True, gridspec_kw={"hspace": 0.25})
    xs = x - 0.5   # the step lies between pixels -1 and 0
    axs[0].plot(xs, f, color=GREY, lw=1.2)
    axs[0].plot(xs, L, color=BLUE, lw=2.4)
    axs[0].set_ylabel("L")
    axs[1].plot(xs, Lx, color=GREEN, lw=2.4)
    axs[1].set_ylabel("Lₓ")
    axs[2].plot(xs, Lxx, color=RED, lw=2.4)
    axs[2].axhline(0, color="k", lw=0.8)
    axs[2].set_ylabel("Lₓₓ")
    axs[2].set_xlabel("x (pixels), edge at 0")
    for a in axs:
        a.axvline(0, color=GREY, ls=":", lw=1.5)
    axs[2].text(-18, Lxx.max() * 0.3, "dark side: +", color=RED, fontsize=FONT - 2, ha="center")
    axs[2].text(18, Lxx.min() * 0.6, "bright side: −", color=RED, fontsize=FONT - 2, ha="center")
    axs[1].text(4, Lx.max() * 0.8, "steepest", fontsize=FONT - 2)
    axs[2].annotate("zero crossing", xy=(0, 0), xytext=(6, Lxx.max() * 0.6), fontsize=FONT - 2,
                    arrowprops=dict(arrowstyle="->", lw=1))
    axs[0].text(-29, 140, f"σ = {s}", fontsize=FONT - 2)
    save(fig, "fig-zerocross.svg")


# 4. Size from scale --------------------------------------------------------------------
def norm_log_centre(img, cy, cx, sigmas):
    out = []
    for s in sigmas:
        r = max(1, int(np.ceil(4 * s)))
        patch = np.pad(img, r + 2, mode="edge")[cy:cy + 2 * r + 5, cx:cx + 2 * r + 5]
        lxx = gderiv(patch, s, 2, 0)
        lyy = gderiv(patch, s, 0, 2)
        out.append(-(s * s) * (lxx + lyy)[r + 2, r + 2])
    return np.array(out)


def fig_size_scale():
    h, w = 46, 300
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w), 40.0)
    discs = [(30, 3), (95, 6), (205, 12)]
    sigs = np.exp(np.linspace(np.log(1), np.log(16), 400))
    res = []
    for cx, r in discs:
        img[np.hypot(xx - cx, yy - 23) <= r] = 150
    fig, ax = plt.subplots(figsize=(15.5, 2.5))
    show(ax, img, vmin=0, vmax=255)
    cols = [CYAN, ORANGE, "#7ee3b8"]
    for (cx, r), c in zip(discs, cols):
        resp = norm_log_centre(img, 23, cx, sigs)
        k = int(np.argmax(resp))
        sb = sigs[k]
        res.append((r, sb))
        ax.add_patch(Circle((cx, 23), np.sqrt(2) * sb, fill=False, ec=c, lw=3, ls="--"))
        ax.text(cx + (r + 8 if r < 12 else 24), 23, f"r = {r}\nbest σ = {sb:.2f}\n√2·σ = {np.sqrt(2) * sb:.2f}",
                ha="left", va="center", color="w", fontsize=FONT)
    print("  size from scale:", [(r, round(s, 2), round(np.sqrt(2) * s, 2), round(r / np.sqrt(2), 2)) for r, s in res])
    save(fig, "fig-size-scale.png", dpi=110)
    return res


# 5. Blob detection in (x, y, sigma) ------------------------------------------------------
def blob_scene():
    rng = np.random.default_rng(7)
    h = w = 200
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    img = 40 + 0.25 * xx                       # uneven background
    discs = [(30, 30, 3), (75, 28, 5), (135, 32, 8), (40, 95, 12), (110, 100, 16), (170, 100, 4)]
    for cx, cy, r in discs:
        img[np.hypot(xx - cx, yy - cy) <= r] += 90
    # touching pair
    for cx in (40, 56):
        img[np.hypot(xx - cx, yy - 162) <= 8] += 90
    # elongated blob (ellipse 22 x 4)
    img[((xx - 125) / 24) ** 2 + ((yy - 165) / 4.5) ** 2 <= 1] += 90
    img += 6 * rng.standard_normal((h, w))
    return img


def maxima3d(stack, thr):
    S, h, w = stack.shape
    out = []
    from scipy.ndimage import maximum_filter
    mx = maximum_filter(stack, size=3, mode="nearest")
    cand = np.argwhere((stack == mx) & (stack > thr))
    for k, y, x in cand:
        if 2 <= y < h - 2 and 2 <= x < w - 2:
            out.append((k, y, x, stack[k, y, x]))
    return out


def fig_blob_detect():
    img = blob_scene()
    sig = geom_scales(1, 16, 16)
    stack = np.array([-(s * s) * (gderiv(img, s, 2, 0) + gderiv(img, s, 0, 2)) for s in sig])
    thr = 40
    det = maxima3d(stack, thr)
    print("  blob detections", len(det))
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    show(ax, img)
    for k, y, x, v in det:
        ax.add_patch(Circle((x, y), np.sqrt(2) * sig[k], fill=False, ec=ORANGE, lw=2))
    ax.text(48, 185, "touching", ha="center", va="top", color="w", fontsize=FONT - 1)
    ax.text(125, 185, "elongated", ha="center", va="top", color="w", fontsize=FONT - 1)
    ax.set_xlim(-0.5, 199.5); ax.set_ylim(199.5, -0.5)
    save(fig, "fig-blob-detect.png", dpi=110)
    return det, sig


# 6. DoG vs normalised LoG (the slide's worked example) -----------------------------------
def fig_dog_log():
    x = np.linspace(-15, 15, 601)
    s0, A, s, k = 3.0, 100.0, 2.0, 1.2

    def blurred(t):   # Gaussian bump of width s0 blurred by sigma t (continuous)
        w2 = s0 ** 2 + t ** 2
        return A * s0 / np.sqrt(w2) * np.exp(-x ** 2 / (2 * w2))

    L1, L2 = blurred(s), blurred(k * s)
    w2 = s0 ** 2 + s ** 2
    Lxx = A * s0 / np.sqrt(w2) * (x ** 2 / w2 - 1) / w2 * np.exp(-x ** 2 / (2 * w2))
    dog = L2 - L1
    approx = (k - 1) * s ** 2 * Lxx
    c = len(x) // 2
    print("  dog example: L(σ)", round(L1[c], 2), "L(kσ)", round(L2[c], 2), "DoG", round(dog[c], 2),
          "Lxx", round(Lxx[c], 2), "(k-1)σ²Lxx", round(approx[c], 2))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 6.4), sharex=True, gridspec_kw={"hspace": 0.12})
    a1.plot(x, A * np.exp(-x ** 2 / (2 * s0 ** 2)), color=GREY, lw=1.5, label="bump")
    a1.plot(x, L1, color=BLUE, lw=2.2, label=f"σ = 2: {L1[c]:.2f}")
    a1.plot(x, L2, color=GREEN, lw=2.2, label=f"kσ = 2.4: {L2[c]:.2f}")
    a1.set_ylabel("value")
    a1.set_xlim(-15, 15)
    a1.legend(frameon=False, fontsize=FONT - 4, loc="upper left", bbox_to_anchor=(-0.03, 1.08))
    a2.plot(x, dog, color=RED, lw=2.6, label=f"DoG: {dog[c]:.2f}".replace("-", "−"))
    a2.plot(x, approx, color="k", lw=1.6, ls="--", label=f"(k − 1)σ²Lₓₓ: {approx[c]:.2f}".replace("-", "−"))
    a2.axhline(0, color=GREY, lw=0.8)
    a2.set_xlabel("x"); a2.set_ylabel("value")
    a2.legend(frameon=False, fontsize=FONT - 4, loc="lower right", bbox_to_anchor=(1.04, -0.02))
    save(fig, "fig-dog-log.svg")


# 7. Pyramids --------------------------------------------------------------------------------
def canvas(levels, gap=10, signed=False):
    H = levels[0].shape[0]
    W = sum(l.shape[1] for l in levels) + gap * (len(levels) - 1)
    c = np.ones((H, W, 3))
    x = 0
    for i, l in enumerate(levels):
        if signed and i < len(levels) - 1:
            v = np.percentile(np.abs(l), 99)
            n = np.clip(0.5 + 0.5 * l / v, 0, 1)
        else:
            n = np.clip(l / 255, 0, 1)
        c[0:l.shape[0], x:x + l.shape[1]] = n[..., None]
        x += l.shape[1] + gap
    return c


def fig_pyramid():
    img = cell()
    g = [img]
    for _ in range(3):
        b = blur(g[-1], 1.0)
        g.append(b[::2, ::2])
    lap = []
    for i in range(3):
        up = blur(np.kron(g[i + 1], np.ones((2, 2))), 1.0)
        lap.append(g[i] - up)
    lap.append(g[3])
    # check exactness of the reconstruction
    rec = lap[3]
    for i in (2, 1, 0):
        rec = lap[i] + blur(np.kron(rec, np.ones((2, 2))), 1.0)
    print("  laplacian pyramid reconstruction error", np.abs(rec - img).max())
    fig, axs = plt.subplots(1, 2, figsize=(16, 3.6), gridspec_kw={"wspace": 0.06})
    show(axs[0], canvas(g), "Gaussian pyramid: 256², 128², 64², 32²")
    show(axs[1], canvas(lap, signed=True), "Laplacian pyramid (grey = 0)")
    fig.text(0.98, 0.04, "Micrograph: R. M. Hunt, CC BY 3.0, Wikimedia Commons", ha="right",
             fontsize=FONT - 5, color=GREY)
    save(fig, "fig-pyramid.png", dpi=110)


if __name__ == "__main__":
    fig_scalespace1d()
    fig_box_vs_gauss()
    fig_zerocross()
    fig_size_scale()
    fig_blob_detect()
    fig_dog_log()
    fig_pyramid()
