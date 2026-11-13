# Slide figures for resolution/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/resolution/figs.py
# Writes fig-*.svg / fig-*.png into resolution/. Synthetic data only; fixed seed.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import j1
from math import pi
plt.rcParams.update({"font.size": 19, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
OUT = "resolution/"
rng = np.random.default_rng(3)
def save(fig, name, **kw):
    fig.savefig(OUT + name, **kw); plt.close(fig); print("wrote", OUT + name)
def airy(v):
    v = np.where(np.abs(v) < 1e-9, 1e-9, v); return (2 * j1(v) / v) ** 2
NA, lam = 1.4, 520.0

def pixelate(points, px, half, sub=None, NA=NA, lam=lam):
    """Integrate a sum of Airy PSFs (each normalised to 1 photon) over square pixels of size px (nm).
    Returns pixel image and pixel centres; field [-half, half]."""
    npx = int(round(2 * half / px)); edges = -npx * px / 2 + np.arange(npx + 1) * px
    sub = sub or max(4, int(np.ceil(px / 5)))
    s = (np.arange(sub) + 0.5) / sub * px
    xs = (edges[:-1, None] + s[None, :]).ravel()
    X, Y = np.meshgrid(xs, xs)
    I = np.zeros_like(X)
    for (x0, y0) in points:
        I += airy(2 * pi * NA * np.hypot(X - x0, Y - y0) / lam)
    # normalise one Airy to integrate to 1: integral of [2J1(v)/v]^2 over the plane = 4 pi / (2 pi NA/lam)^2 ... = lam^2/(pi NA^2)
    norm = lam ** 2 / (pi * NA ** 2)
    I *= (px / sub) ** 2 / norm
    P = I.reshape(npx, sub, npx, sub).sum(axis=(1, 3))
    return P, (edges[:-1] + edges[1:]) / 2

# ---------------------------------------------------------------- 1. Nyquist: finest period sampled well and badly
per = lam / (2 * NA)  # 186 nm
x = np.linspace(0, 1500, 3000)
fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
for a, dx, lab in [(ax[0], 93.0, "pixels of 93 nm (λ/4NA)"), (ax[1], 140.0, "pixels of 140 nm")]:
    a.plot(x, np.cos(2 * pi * x / per), color="#999", lw=2)
    xs = np.arange(0, 1500, dx)
    a.plot(xs, np.cos(2 * pi * xs / per), "o", color=C[0], ms=8)
    a.set_yticks([]); a.set_title(lab, fontsize=19, loc="left")
    if dx > 100:
        fa = 1 / dx - 1 / per; a.plot(x, np.cos(2 * pi * x * fa), color=C[1], lw=2.5, ls="--")
        print("alias period nm", 1 / fa)
        a.text(1500, -1.45, f"looks like a period of {1/fa:.0f} nm", color=C[1], ha="right", fontsize=18)
ax[1].set_xlabel("x (nm)"); ax[1].set_ylim(-1.6, 1.2)
fig.tight_layout(); save(fig, "fig-nyquist.svg")

# ---------------------------------------------------------------- 2. two points at the Rayleigh distance on different pixels
d = 0.61 * lam / NA
fig, ax = plt.subplots(1, 3, figsize=(8.4, 3.5))
for a, px in zip(ax, [65, 108, 325]):
    half = 650 if px != 325 else 650
    P, c = pixelate([(-d / 2, 0), (d / 2, 0)], px, np.ceil(half / px) * px + (px / 2 if int(np.ceil(half / px)) % 1 else 0))
    e = c[0] - px / 2, c[-1] + px / 2
    a.imshow(P, cmap="gray", extent=[e[0], e[1], e[1], e[0]], interpolation="nearest")
    a.set_xlim(-650, 650); a.set_ylim(650, -650); a.set_xticks([-500, 0, 500]); a.set_xticklabels(["−0.5", "0", "0.5"]); a.set_yticks([])
    a.set_title(f"{px} nm", fontsize=19); a.set_xlabel("x (µm)")
fig.tight_layout(); save(fig, "fig-pixelgrids.png", dpi=100)

# ---------------------------------------------------------------- 3. oversampling: 300 nm apart, 500 photons each
d3 = 300.0
fig, ax = plt.subplots(2, 2, figsize=(7.4, 6.6), gridspec_kw=dict(height_ratios=[1.15, 1]))
for j, px in enumerate([65, 20]):
    half = np.ceil(500 / px) * px
    if int(round(2 * half / px)) % 2 == 0: half += px / 2  # odd number of pixels: one centred row
    P, c = pixelate([(-d3 / 2, 0), (d3 / 2, 0)], px, half, sub=max(4, int(px // 4)))
    lamb = 500 * P
    Nn = rng.poisson(lamb)
    print(px, "peak pixel mean photons", lamb.max())
    a = ax[0, j]; e = c[0] - px / 2, c[-1] + px / 2
    a.imshow(Nn, cmap="gray", extent=[e[0], e[1], e[1], e[0]], interpolation="nearest")
    a.set_xlim(-500, 500); a.set_ylim(500, -500); a.set_xticks([-400, 0, 400]); a.set_yticks([])
    a.set_title(f"{px} nm pixels", fontsize=21)
    row = len(c) // 2
    b = ax[1, j]
    b.step(c, Nn[row], where="mid", color=C[1], lw=2, label="pixel counts")
    b.plot(c, lamb[row], color=C[0], lw=2.5, label="expected")
    b.set_xlim(-500, 500); b.set_xlabel("x (nm)"); b.set_xticks([-400, 0, 400])
    if j == 0: b.set_ylabel("photons")
fig.tight_layout(); save(fig, "fig-oversampling.png", dpi=100)

# ---------------------------------------------------------------- 4. beads: raw, one bead, average of 30, profiles
px = 65.0; nb = 30; bg = 10.0; phot = 500
def gaussfwhm(prof, xs):
    m = prof.max(); h = m / 2; i = np.argmax(prof)
    l = i
    while prof[l] > h: l -= 1
    r = i
    while prof[r] > h: r += 1
    xl = xs[l] + (h - prof[l]) / (prof[l + 1] - prof[l]) * (xs[l + 1] - xs[l])
    xr = xs[r - 1] + (h - prof[r - 1]) / (prof[r] - prof[r - 1]) * (xs[r] - xs[r - 1])
    return xr - xl
# field of 30 beads at random sub-pixel positions
F = 64; field = np.full((F, F), bg)
pos = []
gx = np.linspace(8, F - 9, 6)
for i in range(nb):
    cy, cx = gx[i // 6 % 5] + rng.uniform(-1, 1), gx[i % 6] + rng.uniform(-1, 1)
    pos.append((cy, cx))
# render each bead as an Airy integrated over pixels (40 nm beads: point-like)
ii = np.arange(F)
sub = 8; s = ((np.arange(sub) + 0.5) / sub - 0.5)
YY, XX = np.meshgrid((ii[:, None] + s[None, :]).ravel(), (ii[:, None] + s[None, :]).ravel(), indexing="ij")
norm = lam ** 2 / (pi * NA ** 2)
for (cy, cx) in pos:
    I = airy(2 * pi * NA * np.hypot((XX - cx) * px, (YY - cy) * px) / lam) * (px / sub) ** 2 / norm
    field += phot * I.reshape(F, sub, F, sub).sum(axis=(1, 3))
img = rng.poisson(field).astype(float)
# background subtracted crops; sub-pixel centre by a Gaussian fit; then every pixel is put
# into a fine bin (1/4 pixel) by its position relative to the centre, and the bins are averaged
from scipy.optimize import curve_fit
R = 6; up = 4
yy_, xx_ = np.mgrid[-R:R + 1, -R:R + 1].astype(float)
def g2(c, a, y0, x0, s): y, x = c; return a * np.exp(-((y - y0) ** 2 + (x - x0) ** 2) / (2 * s * s))
nbin = 2 * R * up + 1; cen = R * up
def binned(beads):
    acc = np.zeros((nbin, nbin)); cnt = np.zeros((nbin, nbin))
    for cr, oy, ox in beads:
        by = np.round((yy_ - oy) * up).astype(int) + cen; bx = np.round((xx_ - ox) * up).astype(int) + cen
        ok = (by >= 0) & (by < nbin) & (bx >= 0) & (bx < nbin)
        np.add.at(acc, (by[ok], bx[ok]), cr[ok] / cr.sum()); np.add.at(cnt, (by[ok], bx[ok]), 1)
    return acc, cnt
beads = []
for (cy, cx) in pos:
    iy, ix = int(round(cy)), int(round(cx))
    cr = img[iy - R:iy + R + 1, ix - R:ix + R + 1] - bg
    popt, _ = curve_fit(g2, (yy_.ravel(), xx_.ravel()), cr.ravel(), p0=(cr.max(), 0, 0, 1.2))
    beads.append((cr, popt[1], popt[2]))
def profile(beads):
    acc, cnt = binned(beads)
    # profile along x through the centre: all pixels within half a pixel of the line y = 0
    xs_b = (np.arange(nbin) - cen) * px / up
    band = slice(cen - up // 2, cen + up // 2 + 1)
    a_, c_ = acc[band].sum(axis=0), cnt[band].sum(axis=0)
    prof = np.where(c_ > 0, a_ / np.maximum(c_, 1), np.nan)
    ok = ~np.isnan(prof)
    return xs_b[ok], prof[ok]
x1, p1 = profile(beads[:1]); xa, pa = profile(beads)
from scipy.interpolate import interp1d
def fwhm(xs, pr):
    xf = np.linspace(xs[0], xs[-1], 4001); pf = interp1d(xs, pr, kind="linear")(xf)
    return gaussfwhm(pf, xf)
f1 = fwhm(x1, p1); fa = fwhm(xa, pa)
print("bead FWHM one", f1, "avg", fa)
# images: one bead crop (pixels), the 30-bead average on the fine grid
acc, cnt = binned(beads); avg = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0)
crop1 = beads[0][0]; xs_up = xa
plt.rcParams.update({"font.size": 23})
fig = plt.figure(figsize=(7.6, 6.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
a = fig.add_subplot(gs[0, 0]); a.imshow(img, cmap="gray"); a.set_title("beads", fontsize=23); a.axis("off")
a = fig.add_subplot(gs[0, 1]); a.imshow(crop1, cmap="inferno"); a.set_title("one", fontsize=23); a.axis("off")
a = fig.add_subplot(gs[0, 2]); a.imshow(avg, cmap="inferno"); a.set_title(f"mean of {nb}", fontsize=23); a.axis("off")
a = fig.add_subplot(gs[1, :])
a.plot(x1, p1 / p1.max(), "o-", color=C[1], lw=2, ms=6, label=f"one: {f1:.0f} nm")
a.plot(xa, pa / pa.max(), color=C[0], lw=3, label=f"mean: {fa:.0f} nm")
a.set_xlabel("x (nm)"); a.set_yticks([0, 0.5, 1]); a.legend(fontsize=21, frameon=False, title="FWHM", title_fontsize=21, loc="upper left", bbox_to_anchor=(0.55, 1.08))
a.set_xlim(-400, 400); a.set_ylim(-0.12, 1.1); a.set_xticks([-400, 0, 400])
fig.tight_layout(); save(fig, "fig-beads.png", dpi=100)
