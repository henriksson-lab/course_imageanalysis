"""Figures for topics/denoising.qmd. Run from the repository root:
    python3 -I tools/figs/topics-denoising/figs.py
Writes topics/fig-dn-*.svg|png and prints every number quoted on the page (seeded, reproducible).
Set PREVIEW=<dir> to also write PNG copies of the SVG figures for checking.
The figures use standard reference implementations of each method; the course demos implement the
same methods themselves, so their best settings and scores differ slightly.
"""
import os

import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.restoration import denoise_bilateral, denoise_nl_means, denoise_tv_chambolle

FONT = 15
plt.rcParams.update({
    "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
    "xtick.labelsize": FONT - 2, "ytick.labelsize": FONT - 2, "legend.fontsize": FONT - 3,
    "svg.fonttype": "path", "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.facecolor": "white", "svg.hashsalt": "topics-dn",
})
BLUE, ORANGE, GREEN, RED, PURPLE, CYAN, YELLOW, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#777777")
OUT = "topics"


def save(fig, name, **kw):
    os.makedirs(OUT, exist_ok=True)
    if os.environ.get("PREVIEW"):
        fig.savefig(os.path.join(os.environ["PREVIEW"], name.rsplit(".", 1)[0] + ".png"), bbox_inches="tight", dpi=70)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.06, **kw)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def psnr(u, x, peak=255.0):
    return 10 * np.log10(peak ** 2 / np.mean((u - x) ** 2))


def show(ax, img, lo=0, hi=255, title=None):
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=FONT - 2)


# The methods, each with one strength parameter (image values 0–255)
def gauss(y, p): return ndi.gaussian_filter(y, p, mode="reflect")
def median(y, p): return ndi.median_filter(y, size=2 * int(p) + 1, mode="reflect")
def bilateral(y, p): return 255 * denoise_bilateral(y / 255, sigma_color=p / 255, sigma_spatial=2, win_size=7)
def nlm(y, p, sn=None): return 255 * denoise_nl_means(y / 255, h=p / 255, sigma=(SN if sn is None else sn) / 255, patch_size=5, patch_distance=6, fast_mode=True)
def tv(y, p): return 255 * denoise_tv_chambolle(y / 255, weight=p, eps=1e-5, max_num_iter=2000)
def haar2(a):
    """One level of the orthonormal 2D Haar transform (even sizes)."""
    p, q, r, s = a[0::2, 0::2], a[0::2, 1::2], a[1::2, 0::2], a[1::2, 1::2]
    return (p + q + r + s) / 2, ((p - q + r - s) / 2, (p + q - r - s) / 2, (p - q - r + s) / 2)


def ihaar2(ll, d):
    lh, hl, hh = d
    a = np.empty((2 * ll.shape[0], 2 * ll.shape[1]))
    a[0::2, 0::2] = (ll + lh + hl + hh) / 2
    a[0::2, 1::2] = (ll - lh + hl - hh) / 2
    a[1::2, 0::2] = (ll + lh - hl - hh) / 2
    a[1::2, 1::2] = (ll - lh - hl + hh) / 2
    return a


def soft(c, t): return np.sign(c) * np.maximum(np.abs(c) - t, 0)


def wavelet(y, p, levels=3, shifts=4):
    """Soft thresholding of the Haar detail coefficients (threshold p), averaged over shifts (cycle spinning)."""
    out = np.zeros_like(y)
    for dy in range(shifts):
        for dx in range(shifts):
            a = np.roll(y, (dy, dx), (0, 1))
            stack = []
            for _ in range(levels):
                a, d = haar2(a)
                stack.append(tuple(soft(c, p) for c in d))
            for d in reversed(stack):
                a = ihaar2(a, d)
            out += np.roll(a, (-dy, -dx), (0, 1))
    return out / shifts ** 2


def wiener_oracle(y, x, sn):
    """Wiener filter with the power spectrum of the truth (the best linear filter: an upper bound)."""
    m = x.mean()
    P = np.abs(np.fft.fft2(x - m)) ** 2 / x.size
    g = P / (P + sn ** 2)
    return np.real(np.fft.ifft2(np.fft.fft2(y - m) * g)) + m


METHODS = [("Gaussian", gauss, "σ", np.linspace(0.4, 2.0, 17)),
           ("median", median, "radius", [1, 2, 3]),
           ("bilateral", bilateral, "σ_r", np.linspace(5, 80, 16)),
           ("non-local means", nlm, "h", np.linspace(4, 40, 13)),
           ("total variation", tv, "λ", np.linspace(0.01, 0.2, 20)),
           ("wavelet shrinkage", wavelet, "t", np.linspace(10, 70, 13))]

# ======================================================================= the micrograph
x = np.asarray(Image.open("demos/src/fourier2d/cell256.png").convert("L")).astype(float)
SN = 30.0
rng = np.random.default_rng(4)
y = x + SN * rng.standard_normal(x.shape)
print("micrograph: noisy PSNR %.1f dB" % psnr(y, x))
best = {}
for name, f, pn, grid in METHODS:
    scores = [(psnr(f(y, p), x), p) for p in grid]
    sc, p = max(scores)
    best[name] = (p, f(y, p), sc)
    print("micrograph: %-18s best %s = %.3g  PSNR %.1f dB" % (name, pn, p, sc))
w = wiener_oracle(y, x, SN)
best["Wiener (known spectrum)"] = (None, w, psnr(w, x))
print("micrograph: Wiener (oracle spectrum) PSNR %.1f dB" % psnr(w, x))

fig, axs = plt.subplots(2, 4, figsize=(15, 8.9))
show(axs[0, 0], y, title=f"noisy, σ = {SN:.0f} ({psnr(y, x):.1f} dB)")
order = ["Gaussian", "median", "Wiener (known spectrum)", "bilateral", "total variation", "wavelet shrinkage", "non-local means"]
for ax, name in zip(axs.ravel()[1:], order):
    p, u, sc = best[name]
    show(ax, u, title=f"{name}\n{sc:.1f} dB")
fig.tight_layout()
save(fig, "fig-dn-gallery.png", dpi=90)

# zoom on the edge of the nucleus: what each result looks like up close
r0, c0, R = 96, 140, 72
fig, axs = plt.subplots(1, 6, figsize=(16, 3.4))
crops = [("reference", x), (f"noisy ({psnr(y, x):.1f} dB)", y)] + [
    (f"{nm} ({best[nm][2]:.1f} dB)", best[nm][1]) for nm in ["Gaussian", "total variation", "wavelet shrinkage", "non-local means"]]
for ax, (nm, im) in zip(axs, crops):
    show(ax, im[r0:r0 + R, c0:c0 + R], title=nm)
fig.tight_layout()
save(fig, "fig-dn-zoom.png", dpi=90)

# ======================================================================= 1D profile: beliefs
n = 160
t = np.arange(n)
truth = np.full(n, 60.0)
truth[20:50] = 160                                   # a plateau (two steps)
truth[65:105] = 60 + (t[65:105] - 65) * 2.5          # a ramp
truth[105:112] = 160
truth[124] = 160                                      # a one-pixel line
truth[140:143] = 140                                  # a small spot (3 px)
rng = np.random.default_rng(8)
noisy = truth + 20 * rng.standard_normal(n)


def bilat1(u, sr, ss=2.0, R=6):
    out = np.empty_like(u)
    for i in range(len(u)):
        j = np.arange(max(0, i - R), min(len(u), i + R + 1))
        wgt = np.exp(-((j - i) ** 2) / (2 * ss ** 2) - ((u[j] - u[i]) ** 2) / (2 * sr ** 2))
        out[i] = np.sum(wgt * u[j]) / wgt.sum()
    return out


res1 = {
    "Gaussian (σ = 2)": ndi.gaussian_filter1d(noisy, 2, mode="reflect"),
    "median (window 7)": ndi.median_filter(noisy, size=7, mode="reflect"),
    "bilateral (σ_r = 40)": bilat1(noisy, 40),
    "total variation (λ = 0.12)": 255 * denoise_tv_chambolle(noisy / 255, weight=0.12, eps=1e-5, max_num_iter=2000),
}
for k_, v in res1.items():
    flat = np.r_[0:18, 52:63, 145:160]
    print("profile: %-28s rmse %.1f  noise left in flat %.1f  line peak %.0f (true 160)  spot peak %.0f (true 140)  step 10-90 width at 20: %s" % (
        k_, np.sqrt(np.mean((v - truth) ** 2)), np.std(v[flat] - truth[flat]), v[124], v[140:143].max(),
        int(np.sum((v[14:27] > 70) & (v[14:27] < 150)))))
fig, axs = plt.subplots(len(res1), 1, figsize=(10, 9.0), sharex=True)
for ax, (k_, v) in zip(axs, res1.items()):
    ax.plot(t, noisy, color="#bbbbbb", lw=1, drawstyle="steps-mid")
    ax.plot(t, truth, color=GREEN, lw=1.6, drawstyle="steps-mid")
    ax.plot(t, v, color=BLUE, lw=2.2)
    ax.set_ylim(0, 210)
    ax.set_yticks([60, 160])
    ax.set_title(k_, loc="left", fontsize=FONT - 1)
for xx, lab in [(35, "plateau"), (85, "ramp"), (124, "1-px line"), (141, "spot")]:
    axs[0].text(xx, 195, lab, ha="center", fontsize=FONT - 3, color=GREY)
axs[-1].set_xlabel("pixel")
fig.legend(handles=[plt.Line2D([], [], color=GREEN, lw=1.6, label="truth"),
                    plt.Line2D([], [], color="#bbbbbb", lw=1, label="noisy (σ = 20)"),
                    plt.Line2D([], [], color=BLUE, lw=2.2, label="result")],
           loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(1.0, 1.02))
fig.tight_layout()
save(fig, "fig-dn-profile.svg")

# ======================================================================= trade-off curves on a test card
N = 128
card = np.full((N, N), 60.0)
card[16:56, 16:56] = 160                               # square
card[:, 80] = 140                                       # 1-px vertical line
yy, xx = np.mgrid[:N, :N]
for cy, cx in [(90, 20), (90, 44), (110, 32)]:
    card[(yy - cy) ** 2 + (xx - cx) ** 2 <= 4] = 140    # small discs r = 2
rng = np.random.default_rng(12)
yc = card + 20 * rng.standard_normal(card.shape)
flat = np.zeros((N, N), bool); flat[64:124, 92:124] = True
discmask = np.zeros((N, N), bool)
for cy, cx in [(90, 20), (90, 44), (110, 32)]:
    discmask |= (yy - cy) ** 2 + (xx - cx) ** 2 <= 4
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
TRADE = [("Gaussian", gauss, np.linspace(0.5, 3, 11), BLUE),
         ("median", median, [1, 2, 3, 4], ORANGE),
         ("bilateral", bilateral, np.linspace(20, 160, 8), PURPLE),
         ("total variation", tv, np.linspace(0.03, 0.4, 10), RED),
         ("wavelet shrinkage", wavelet, np.linspace(10, 90, 9), CYAN),
         ("non-local means", lambda y_, p: nlm(y_, p, 20.0), np.linspace(4, 40, 10), GREEN)]
for name, f, grid, col in TRADE:
    pts = []
    for p in grid:
        u = f(yc, p)
        left = np.std(u[flat] - card[flat])
        line = (u[20:110, 80].mean() - 0.5 * (u[20:110, 77].mean() + u[20:110, 83].mean())) / 80.0
        disc = (u[discmask].mean() - 60) / 80.0
        pts.append((left, line, disc))
    pts = np.array(pts)
    a1.plot(pts[:, 0], 100 * pts[:, 1], "o-", color=col, lw=2, ms=5, label=name)
    a2.plot(pts[:, 0], 100 * pts[:, 2], "o-", color=col, lw=2, ms=5, label=name)
    print("tradeoff: %-16s noise left %s" % (name, " ".join("%.1f" % v for v in pts[:, 0])))
    print("tradeoff: %-16s line kept %% %s" % (name, " ".join("%.0f" % (100 * v) for v in pts[:, 1])))
    print("tradeoff: %-16s disc kept %% %s" % (name, " ".join("%.0f" % (100 * v) for v in pts[:, 2])))
for ax, what in [(a1, "1-px line: contrast kept (%)"), (a2, "small disc (r = 2): contrast kept (%)")]:
    ax.set_xlabel("noise left in a flat region (sd; input 20)")
    ax.set_ylabel(what)
    ax.set_xlim(0, 21)
    ax.set_ylim(-5, 110)
    ax.axhline(100, color="k", lw=0.8, ls=":")
    ax.invert_xaxis()
a1.legend(frameon=False, loc="lower left")
a1.set_title("more smoothing →", loc="left", fontsize=FONT - 2)
fig.tight_layout()
save(fig, "fig-dn-tradeoff.svg")

# ======================================================================= mean against sample (two-valued prior)
m0, m1, s0, pi1, sn = 50.0, 150.0, 8.0, 0.3, 40.0
ys = np.linspace(-40, 240, 561)
xg = np.linspace(-20, 220, 2401)
prior = (1 - pi1) * np.exp(-(xg - m0) ** 2 / (2 * s0 ** 2)) + pi1 * np.exp(-(xg - m1) ** 2 / (2 * s0 ** 2))
prior /= np.trapezoid(prior, xg)
mu, var = np.trapezoid(xg * prior, xg), np.trapezoid((xg - np.trapezoid(xg * prior, xg)) ** 2 * prior, xg)
cmean, cmap_ = [], []
for yv in ys:
    post = prior * np.exp(-(yv - xg) ** 2 / (2 * sn ** 2))
    post /= np.trapezoid(post, xg)
    cmean.append(np.trapezoid(xg * post, xg)); cmap_.append(xg[np.argmax(post)])
wien = mu + var / (var + sn ** 2) * (ys - mu)
y0 = 105.0
post0 = prior * np.exp(-(y0 - xg) ** 2 / (2 * sn ** 2)); post0 /= np.trapezoid(post0, xg)
pm0 = np.trapezoid(xg * post0, xg)
cdf = np.cumsum(post0); cdf /= cdf[-1]
smp = np.interp(np.random.default_rng(3).uniform(size=8), cdf, xg)
print("meansample: prior mean %.1f sd %.1f; Wiener slope %.2f" % (mu, np.sqrt(var), var / (var + sn ** 2)))
print("meansample: y = %.0f: posterior mean %.1f, MAP %.1f, P(object) %.2f, samples %s" % (
    y0, pm0, xg[np.argmax(post0)], np.trapezoid(post0[xg > 100], xg[xg > 100]), " ".join("%.0f" % v for v in smp)))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.8))
a1.plot(ys, ys, color="#bbbbbb", lw=1, ls=":", label="no denoising")
a1.plot(ys, wien, color=ORANGE, lw=2.2, label="best linear (Wiener)")
a1.plot(ys, cmean, color=BLUE, lw=2.4, label="posterior mean (best PSNR)")
a1.plot(ys, cmap_, color=RED, lw=1.8, ls="--", label="most probable value")
a1.axvline(y0, color=GREY, lw=1)
a1.set_xlabel("noisy value y")
a1.set_ylabel("denoised value")
a1.set_xlim(-40, 240); a1.set_ylim(-20, 230)
a1.legend(frameon=False, loc="upper left", fontsize=FONT - 4)
a1.set_title("pixels are 50 or 150; noise σ = 40", loc="left", fontsize=FONT - 2)
a2.fill_between(xg, post0, color=CYAN, alpha=0.5, lw=0)
a2.axvline(pm0, color=BLUE, lw=2.4, label=f"posterior mean {pm0:.0f}")
for i, v in enumerate(smp):
    a2.plot([v], [0.002 + 0.0012 * (i % 4)], "v", color=PURPLE, ms=10, label="posterior samples" if i == 0 else None)
a2.set_xlabel("clean value x")
a2.set_ylabel("probability given y")
a2.set_yticks([])
a2.legend(frameon=False, loc="upper right", fontsize=FONT - 4)
a2.set_title(f"one pixel measured at y = {y0:.0f}", loc="left", fontsize=FONT - 2)
fig.tight_layout()
save(fig, "fig-dn-meansample.svg")
