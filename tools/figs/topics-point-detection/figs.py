"""Figures for topics/point-detection.qmd. Run from the repository root:
    python3 -I tools/figs/topics-point-detection/figs.py
Writes topics/fig-pd-*.svg|png and prints every number quoted on the page (seeded, reproducible).
"""
import os

import numpy as np
from scipy import ndimage as ndi
from scipy.special import erf
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import hough_line, hough_line_peaks

FONT = 15
plt.rcParams.update({
    "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
    "xtick.labelsize": FONT - 2, "ytick.labelsize": FONT - 2, "legend.fontsize": FONT - 3,
    "svg.fonttype": "path", "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.facecolor": "white", "svg.hashsalt": "topics-pd",
})
BLUE, ORANGE, GREEN, RED, PURPLE, CYAN, YELLOW, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#777777")
OUT = "topics"


def save(fig, name, **kw):
    os.makedirs(OUT, exist_ok=True)
    if os.environ.get("PREVIEW") and name.endswith(".svg"):      # optional PNG copies for checking
        fig.savefig(os.path.join(os.environ["PREVIEW"], name[:-4] + ".png"), bbox_inches="tight", dpi=70)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.06, **kw)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def pixfrac(x, c, s):
    """Fraction of a unit Gaussian (centre c, sd s) that falls in pixel x (pixel = [x-0.5, x+0.5])."""
    return 0.5 * (erf((x + 0.5 - c) / (np.sqrt(2) * s)) - erf((x - 0.5 - c) / (np.sqrt(2) * s)))


def spot_image(n, spots, s, bg, noise, rng):
    x = np.arange(n)
    img = bg.copy()
    for (cx, cy, a) in spots:
        img += a * 2 * np.pi * s * s * np.outer(pixfrac(x, cy, s), pixfrac(x, cx, s))
    return img + noise * rng.standard_normal(img.shape)


def local_max(img, r):
    m = ndi.maximum_filter(img, size=2 * r + 1, mode="nearest")
    return np.argwhere(img == m)


def match(det, truth, gate=2.0):
    if len(det) == 0:
        return 0, 0, len(truth)
    d = np.linalg.norm(det[:, None, :] - truth[None, :, :], axis=2)
    cost = np.where(d <= gate, d, 1e6)
    r, c = linear_sum_assignment(cost)
    tp = int(np.sum(cost[r, c] <= gate))
    return tp, len(det) - tp, len(truth) - tp


def mad_sigma(img):
    """Noise sd from the median absolute deviation of the finest-scale differences."""
    d = (img[:, 1:] - img[:, :-1]).ravel() / np.sqrt(2)
    return np.median(np.abs(d - np.median(d))) / 0.6745


# ======================================================================= fig-pd-recipe.svg (1D)
rng = np.random.default_rng(3)
N1, S = 200, 1.5
xs = np.arange(N1)
centres = [30.0, 62.4, 70.0, 108.0, 141.3, 172.0]
amps = [40, 35, 35, 14, 30, 45]
bg1 = 20 + 0.15 * xs + 25 * np.exp(-((xs - 150) / 25.0) ** 2)
prof = bg1.copy()
for c, a in zip(centres, amps):
    prof += a * np.sqrt(2 * np.pi) * S * pixfrac(xs, c, S)
prof += 6 * rng.standard_normal(N1)
# zero-mean Gaussian (matched filter minus its mean), radius 4 sigma
R = int(np.ceil(4 * S))
k = np.exp(-0.5 * (np.arange(-R, R + 1) / S) ** 2)
k /= k.sum()
kz = k - k.mean()
resp = np.convolve(np.pad(prof, R, mode="reflect"), kz[::-1], mode="valid")
sig_n = 6.0
noise_resp = sig_n * np.sqrt(np.sum(kz ** 2))
thr = 4 * noise_resp
cand = [i for i in range(2, N1 - 2) if resp[i] == resp[max(0, i - 2):i + 3].max()]
keep = [i for i in cand if resp[i] > thr]
print("recipe: candidates", len(cand), "kept", len(keep), "at", keep, "noise of response %.2f thr %.2f" % (noise_resp, thr))
# sub-pixel: log-parabola on the raw profile minus local background (bg known here for the drawing)
fig, axs = plt.subplots(3, 1, figsize=(9.5, 7.2), sharex=False, gridspec_kw={"height_ratios": [1.1, 1.1, 0.9]})
a0 = axs[0]
a0.plot(xs, prof, color=GREY, lw=1.2, drawstyle="steps-mid")
a0.plot(xs, bg1, color=ORANGE, lw=1.5, ls="--", label="background")
for c in centres:
    a0.axvline(c, color=GREEN, lw=1, alpha=0.6)
a0.set_ylabel("raw")
a0.set_title("1  raw profile: six spots (green lines) on a sloping background, noise", loc="left", fontsize=FONT - 1)
a0.legend(loc="upper left", frameon=False)
a1 = axs[1]
a1.plot(xs, resp, color=BLUE, lw=1.5)
a1.axhline(thr, color=RED, ls="--", lw=1.3)
a1.text(90, 17, "threshold = 4 × noise of the response", color=RED, ha="center", fontsize=FONT - 3)
a1.plot(cand, resp[cand], "o", ms=4, color=GREY, label="local maxima (candidates)")
a1.plot(keep, resp[keep], "o", ms=7, mfc="none", mec=RED, mew=2, label="kept")
a1.set_ylabel("filtered")
a1.set_title("2–3  zero-mean filter, local maxima, threshold from the noise", loc="left", fontsize=FONT - 1)
a1.legend(loc="upper left", frameon=False, ncol=2)
a2 = axs[2]
# sub-pixel: least-squares fit of Gaussian (known width) + local background to the raw pixels, ±4 px
from scipy.optimize import curve_fit
sub = []
for i in keep:
    xw = np.arange(i - 4, i + 5)
    model = lambda x, c, A, B: A * np.sqrt(2 * np.pi) * S * pixfrac(x, c, S) + B
    popt, _ = curve_fit(model, xw, prof[xw], p0=[i, resp[i], np.median(prof[xw])])
    sub.append(popt[0])
lo_, hi_ = 54, 80
xw = np.arange(lo_, hi_ + 1)
a2.bar(xw, prof[xw], width=1.0, color="#dddddd", edgecolor=GREY, lw=0.6)
for c in centres:
    if lo_ <= c <= hi_:
        a2.axvline(c, color=GREEN, lw=2.5)
for c in sub:
    if lo_ <= c <= hi_:
        a2.axvline(c, color=RED, lw=2, ls="--")
a2.set_xlim(lo_ - 0.5, hi_ + 0.5)
a2.set_ylim(0, 75)
a2.set_ylabel("raw")
a2.set_xlabel("pixel")
a2.set_title("4  zoom on the close pair: fitted positions (red dashed) against the truth (green)", loc="left", fontsize=FONT - 1)
fig.tight_layout()
save(fig, "fig-pd-recipe.svg")
print("recipe: sub-pixel", ["%.2f" % v for v in sub], "errors", ["%.2f" % (v - c) for v, c in zip(sub, centres)])

# ======================================================================= fig-pd-maps.png (four response maps)
fig, axs = plt.subplots(1, 4, figsize=(15, 4.4))
# (a) spots: LoG response with maxima
rng = np.random.default_rng(11)
n = 96
spots = [(rng.uniform(8, n - 8), rng.uniform(8, n - 8), rng.uniform(40, 90)) for _ in range(14)]
img = spot_image(n, spots, 1.5, np.full((n, n), 30.0), 8, rng)
log = -ndi.gaussian_laplace(img, 1.5) * 1.5 ** 2
# noise of the response = noise of the image (estimated) × noise gain of the filter (from white noise)
gain = np.std(-ndi.gaussian_laplace(np.random.default_rng(98).standard_normal((256, 256)), 1.5) * 1.5 ** 2)
nl = mad_sigma(img) * gain
pk = [p for p in local_max(log, 2) if log[p[0], p[1]] > 5 * nl]
ax = axs[0]
ax.imshow(log, cmap="magma", interpolation="nearest")
ax.plot([p[1] for p in pk], [p[0] for p in pk], "o", ms=11, mfc="none", mec=CYAN, mew=1.8)
ax.set_title("spots: filtered image\npeak = a spot (x, y)")
ax.set_xticks([]); ax.set_yticks([])
print("maps: spots", len(spots), "peaks", len(pk))
# (b) blobs over scale: scale-normalised LoG at the centre of three discs against sigma
ax = axs[1]
sig = np.linspace(0.8, 12, 120)
for r, col in [(3, BLUE), (6, ORANGE), (10, GREEN)]:
    m = 64
    yy, xx = np.mgrid[:m, :m] - (m - 1) / 2
    disc = (xx ** 2 + yy ** 2 <= r * r).astype(float)
    v = [-(s ** 2) * ndi.gaussian_laplace(disc, s)[m // 2, m // 2] for s in sig]
    sb = sig[int(np.argmax(v))]
    ax.plot(sig, v, color=col, lw=2, label=f"disc r = {r}")
    ax.plot(sb, max(v), "o", color=col, ms=7)
    print("maps: disc r", r, "best sigma %.2f (r/sqrt2 = %.2f)" % (sb, r / np.sqrt(2)))
ax.set_xlabel("filter scale σ (px)")
ax.set_ylabel("normalised LoG")
ax.legend(frameon=False, loc="upper right")
ax.set_title("blobs: response against scale\npeak = a blob and its size")
# (c) phase correlation surface
cell = np.asarray(Image.open("demos/src/fourier2d/cell256.png").convert("L")).astype(float)
a = cell[40:168, 40:168]
b = cell[40 + 9:168 + 9, 40 - 5:168 - 5]
rng = np.random.default_rng(5)
a = a + 4 * rng.standard_normal(a.shape)
b = b + 4 * rng.standard_normal(b.shape)
w = np.outer(np.hanning(128), np.hanning(128))
F = np.fft.fft2((a - a.mean()) * w) * np.conj(np.fft.fft2((b - b.mean()) * w))
pc = np.real(np.fft.ifft2(F / (np.abs(F) + 1e-9)))
pcs = np.fft.fftshift(pc)
iy, ix = np.unravel_index(np.argmax(pcs), pcs.shape)
dy, dx = iy - 64, ix - 64
ax = axs[2]
ax.imshow(np.sqrt(np.clip(pcs[64 - 24:64 + 25, 64 - 24:64 + 25], 0, None)), cmap="magma", interpolation="nearest", extent=[-24.5, 24.5, 24.5, -24.5])
ax.plot(dx, dy, "o", ms=14, mfc="none", mec=CYAN, mew=2)
ax.set_xlabel("shift x"); ax.set_ylabel("shift y")
ax.set_title("two images: phase correlation\npeak = the shift between them")
print("maps: phase correlation peak (dy, dx) =", dy, dx, "height %.3f" % pcs.max(), "next %.3f" % np.sort(pcs.ravel())[-2])
# (d) Hough accumulator
rng = np.random.default_rng(2)
E = np.zeros((96, 96), bool)
for t in np.linspace(0, 1, 70):
    E[int(10 + 75 * t), int(8 + 80 * t)] = True                     # line 1
    E[int(80 - 20 * t), int(5 + 86 * t)] = True                     # line 2
for _ in range(120):
    E[rng.integers(0, 96), rng.integers(0, 96)] = True              # clutter
H, th, d = hough_line(E, theta=np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False))
hp = hough_line_peaks(H, th, d, num_peaks=2)
ax = axs[3]
ax.imshow(np.sqrt(H), cmap="magma", aspect="auto", interpolation="nearest",
          extent=[np.degrees(th[0]), np.degrees(th[-1]), d[-1], d[0]])
ax.plot(np.degrees(hp[1]), hp[2], "o", ms=12, mfc="none", mec=CYAN, mew=2)
ax.set_xlabel("angle θ (°)"); ax.set_ylabel("distance ρ")
ax.set_title("edge points: Hough accumulator\npeak = a line (θ, ρ)")
print("maps: hough peaks votes", list(hp[0]), "max clutter-free votes", H.max())
fig.tight_layout()
save(fig, "fig-pd-maps.png", dpi=100)

# ======================================================================= fig-pd-background.png
rng = np.random.default_rng(7)
n = 160
yy, xx = np.mgrid[:n, :n]
bg = 40 + 50 * xx / n + 90 * np.exp(-((xx - 105) ** 2 + (yy - 60) ** 2) / (2 * 22 ** 2))
spots = [(rng.uniform(6, n - 6), rng.uniform(6, n - 6), rng.uniform(25, 90)) for _ in range(45)]
img = spot_image(n, spots, 1.5, bg, 10, rng)
truth = np.array([[s[1], s[0]] for s in spots])
sn = mad_sigma(img)
print("background: estimated noise %.2f (true 10)" % sn)
filters = {}
g = ndi.gaussian_filter(img, 1.5, mode="nearest")
filters["Gaussian (sums to 1)"] = g
filters["LoG (sums to 0)"] = -ndi.gaussian_laplace(img, 1.5, mode="nearest")
th_ = ndi.white_tophat(img, size=9, mode="nearest")
filters["top-hat, then Gaussian"] = ndi.gaussian_filter(th_, 1.5, mode="nearest")
fig, axs = plt.subplots(1, 4, figsize=(15, 4.3))
axs[0].imshow(img, cmap="gray", interpolation="nearest")
axs[0].set_title("45 spots, uneven background")
for ax, (name, f) in zip(axs[1:], filters.items()):
    # threshold k = 5 in units of the noise of the response: noise from pure-noise filtering (same filter, same sigma)
    nz = 10 * np.random.default_rng(99).standard_normal((n, n))
    if name.startswith("Gaussian"):
        fn = ndi.gaussian_filter(nz, 1.5, mode="nearest")
        base = np.median(f)                                  # one global background level
    elif name.startswith("LoG"):
        fn = -ndi.gaussian_laplace(nz, 1.5, mode="nearest"); base = 0.0
    else:
        fn = ndi.gaussian_filter(ndi.white_tophat(nz, size=9, mode="nearest"), 1.5, mode="nearest")
        base = np.median(fn)
    sr = np.std(fn)
    det = np.array([p for p in local_max(f, 2) if f[p[0], p[1]] - base > 5 * sr], float)
    tp, fp, fn_ = match(det, truth)
    ax.imshow(f, cmap="gray", interpolation="nearest")
    dd = det.astype(int) if len(det) else np.zeros((0, 2), int)
    ok = []
    if len(det):
        d = np.linalg.norm(det[:, None, :] - truth[None, :, :], axis=2)
        ok = d.min(axis=1) <= 2
    for p, o in zip(dd, ok):
        ax.plot(p[1], p[0], "o", ms=7, mfc="none", mec=GREEN if o else RED, mew=1.6)
    ax.set_title(f"{name}\nTP {tp}, FP {fp}, FN {fn_}")
    print("background:", name, "TP", tp, "FP", fp, "FN", fn_, "F1 %.2f" % (2 * tp / (2 * tp + fp + fn_)))
for ax in axs:
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
save(fig, "fig-pd-background.png", dpi=100)

# ======================================================================= fig-pd-falsespots.svg
rng = np.random.default_rng(1)
sf, side, reps = 1.5, 512, 20
ks = np.array([2.5, 3, 3.5, 4, 4.5, 5])
counts = np.zeros(len(ks)); nmax = 0; above_px = np.zeros(len(ks))
for _ in range(reps):
    z = ndi.gaussian_filter(rng.standard_normal((side, side)), sf, mode="wrap")
    z /= z.std()
    m = ndi.maximum_filter(z, size=3, mode="wrap") == z
    nmax += m.sum()
    v = z[m]
    counts += np.array([(v > k).sum() for k in ks])
counts /= reps; nmax /= reps
naive = nmax * 0.5 * (1 - erf(ks / np.sqrt(2)))
ec = side ** 2 / (2 * sf ** 2) * ks * np.exp(-ks ** 2 / 2) / (2 * np.pi) ** 1.5
for k, c, nv, e in zip(ks, counts, naive, ec):
    print("falsespots: k %.1f measured %.2f naive %.2f formula %.2f" % (k, c, nv, e))
print("falsespots: maxima per image %.0f" % nmax)
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.semilogy(ks, counts, "o-", color=BLUE, lw=2, label="measured (20 noise images)")
ax.semilogy(ks, naive, "s--", color=GREY, lw=1.5, label="naive: maxima × P(Z > k)")
ax.axhline(1, color=RED, lw=1, ls=":")
ax.text(2.55, 1.25, "1 false spot per image", color=RED, fontsize=FONT - 3)
ax.set_xlabel("threshold k (noise sd of the filtered image)")
ax.set_ylabel("false spots per 512² image")
ax.legend(frameon=False)
fig.tight_layout()
save(fig, "fig-pd-falsespots.svg")

# ======================================================================= fig-pd-subpixel.svg
s = 1.0
offs = np.linspace(-0.49, 0.49, 99)
res = {"brightest pixel": [], "centroid (3 × 3)": [], "parabola": [], "log-parabola": []}
for o in offs:
    x = np.arange(-4, 5)
    p = pixfrac(x, o, s) * 100
    res["brightest pixel"].append(x[np.argmax(p)] - o)
    w = p[3:6]
    res["centroid (3 × 3)"].append(np.sum(w * x[3:6]) / w.sum() - o)
    l, c, r = p[3], p[4], p[5]
    res["parabola"].append(0.5 * (l - r) / (l - 2 * c + r) - o)
    L = np.log([l, c, r])
    res["log-parabola"].append(0.5 * (L[0] - L[2]) / (L[0] - 2 * L[1] + L[2]) - o)
for kname, v in res.items():
    print("subpixel: %-18s max |error| %.3f px" % (kname, np.max(np.abs(v))))
fig, ax = plt.subplots(figsize=(6.6, 4.4))
for (kname, v), col, ls in zip(res.items(), [GREY, ORANGE, BLUE, GREEN], ["-", "--", "-", "-"]):
    ax.plot(offs, v, color=col, lw=2.2, ls=ls, label=kname)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("true position within the pixel (px)")
ax.set_ylabel("error of the estimate (px)")
ax.set_title("noise-free spot, σ = 1 px", fontsize=FONT - 1)
ax.legend(frameon=False, fontsize=FONT - 4, loc="upper right")
fig.tight_layout()
save(fig, "fig-pd-subpixel.svg")
