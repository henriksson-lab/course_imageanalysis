# Slide figures for illposed/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/illposed/figs.py
# Writes fig-*.svg / fig-*.png into illposed/. Uses demos/src/fourier2d/cell256.png
# (TEM micrograph, Robert M. Hunt, CC BY 3.0, via Wikimedia Commons) and synthetic data.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse, Circle
from math import pi
plt.rcParams.update({"font.size": 19, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
OUT = "illposed/"
rng = np.random.default_rng(11)
def save(fig, name, **kw):
    fig.savefig(OUT + name, **kw); plt.close(fig); print("wrote", OUT + name)
cell = plt.imread("demos/src/fourier2d/cell256.png")
if cell.ndim == 3: cell = cell[..., 0]
cell = cell.astype(float)
n = 128; img = cell.reshape(128, 2, 128, 2).mean(axis=(1, 3)); img /= img.max()
f = np.fft.fftfreq(n); FX, FY = np.meshgrid(f, f); fr = np.hypot(FX, FY); s = 1.5
H = np.exp(-2 * pi ** 2 * s ** 2 * fr ** 2)
noise = 0.001 * rng.standard_normal((n, n))
g = np.real(np.fft.ifft2(np.fft.fft2(img) * H)) + noise

# ---------------------------------------------------------------- 1. forward and backward
gq = np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2 * pi ** 2 * 3.0 ** 2 * fr ** 2))) + 0.02 * rng.standard_normal((n, n))
fig = plt.figure(figsize=(8.0, 4.4))
a1 = fig.add_axes([0.02, 0.08, 0.36, 0.72]); a2 = fig.add_axes([0.62, 0.08, 0.36, 0.72])
a1.imshow(img, cmap="gray", vmin=0, vmax=1); a1.axis("off"); a1.set_title("object", fontsize=20)
a2.imshow(gq, cmap="gray", vmin=0, vmax=1); a2.axis("off"); a2.set_title("image", fontsize=20)
fig.add_artist(FancyArrowPatch((0.40, 0.62), (0.60, 0.62), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=28, lw=3, color=C[0]))
fig.text(0.50, 0.70, "forward\nknown", ha="center", va="bottom", fontsize=18, color=C[0])
fig.add_artist(FancyArrowPatch((0.60, 0.30), (0.40, 0.30), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=28, lw=3, color=C[1], ls="--"))
fig.text(0.50, 0.22, "backward\n?", ha="center", va="top", fontsize=18, color=C[1])
save(fig, "fig-question.png", dpi=100)

# ---------------------------------------------------------------- 2. gain 1/H
k = np.linspace(0, 0.5, 500)
Hk = np.exp(-2 * pi ** 2 * s ** 2 * k ** 2)
fig, ax = plt.subplots(figsize=(6.6, 4.8))
ax.semilogy(k, Hk, lw=3, color=C[0], label="OTF $H$")
ax.semilogy(k, 1 / Hk, lw=3, color=C[1], label="gain $1/H$")
for h in [0.5, 0.1, 0.01]:
    kk = np.sqrt(-np.log(h) / (2 * pi ** 2 * s ** 2))
    ax.plot([kk, kk], [h, 1 / h], color="#999", ls=":", lw=1.5)
    ax.plot(kk, 1 / h, "o", color=C[1], ms=8); ax.plot(kk, h, "o", color=C[0], ms=8)
    ax.text(kk - 0.01, 1 / h * 1.6, f"×{1/h:g}", ha="right", fontsize=17, color=C[1])
print("max gain at 0.5 cycles/px (1D)", 1 / Hk[-1])
ax.set_xlabel("frequency (cycles/pixel)"); ax.set_ylim(1e-5, 1e5)
ax.set_yticks([1e-4, 1e-2, 1, 1e2, 1e4])
ax.legend(fontsize=17, frameon=False, loc="upper left")
fig.tight_layout(); save(fig, "fig-gain.svg")

# ---------------------------------------------------------------- 3. signal and noise spectra
def radial(P):
    b = np.minimum((fr / 0.5 * 40).astype(int), 40)
    m = np.bincount(b.ravel(), P.ravel(), minlength=41) / np.bincount(b.ravel(), minlength=41)
    return (np.arange(41) + 0.5) / 40 * 0.5, m
F = np.fft.fft2(img); N = np.fft.fft2(noise)
kr, sF = radial(np.abs(F)); _, sFH = radial(np.abs(F * H)); _, sN = radial(np.abs(N)); _, sNH = radial(np.abs(N / H))
sel = kr <= 0.5
fig, ax = plt.subplots(1, 2, figsize=(7.0, 4.8), sharey=True)
ax[0].semilogy(kr[sel], sFH[sel], lw=3, color=C[0], label="signal $FH$")
ax[0].semilogy(kr[sel], sN[sel], lw=3, color="#777", label="noise $N$")
ax[0].set_title("in the image", fontsize=19)
ax[1].semilogy(kr[sel], sF[sel], lw=3, color=C[0], label="signal $F$")
ax[1].semilogy(kr[sel], sNH[sel], lw=3, color=C[1], label="noise $N/H$")
ax[1].set_title("after dividing by $H$", fontsize=19)
for a in ax:
    a.set_xlabel("cycles/pixel"); a.legend(fontsize=17, frameon=False, loc="upper right", handlelength=1.2); a.set_xticks([0, 0.25, 0.5]); a.set_xticklabels(["0", "0.25", "0.5"])
ax[0].set_ylabel("amplitude")
ax[0].set_ylim(1e-5, 1e9)
ax[0].set_yticks([1e-4, 1, 1e4, 1e8])
cross = kr[np.argmax(sN > sFH)]; print("noise exceeds signal from", cross, "cycles/px")
fig.tight_layout(); save(fig, "fig-noisespectrum.svg")

# ---------------------------------------------------------------- 4. stability: two-pixel blur, kappa = 10
A = np.array([[0.55, 0.45], [0.45, 0.55]]); Ai = np.linalg.inv(A)
f0 = np.array([0.6, 0.4]); g0 = A @ f0
t = np.linspace(0, 2 * pi, 200); r = 0.01
dg = r * np.stack([np.cos(t), np.sin(t)]); df = Ai @ dg
fig, ax = plt.subplots(1, 2, figsize=(7.0, 4.2))
ax[0].fill(g0[0] + dg[0], g0[1] + dg[1], color=C[1], alpha=0.5); ax[0].plot(*g0, "o", color="k", ms=6)
ax[1].fill(f0[0] + df[0], f0[1] + df[1], color=C[1], alpha=0.5); ax[1].plot(*f0, "o", color="k", ms=6)
for a, c, lab, ttl in [(ax[0], g0, "g", "images ±0.01"), (ax[1], f0, "f", "objects")]:
    a.set_xlim(c[0] - 0.13, c[0] + 0.13); a.set_ylim(c[1] - 0.13, c[1] + 0.13); a.set_aspect("equal")
    a.set_xlabel(f"pixel 1"); a.set_ylabel(f"pixel 2"); a.set_title(ttl, fontsize=18)
    a.set_xticks([round(c[0] - 0.1, 2), round(c[0] + 0.1, 2)]); a.set_yticks([round(c[1] - 0.1, 2), round(c[1] + 0.1, 2)])
fig.text(0.5, 0.52, "$A^{-1}$\n→", ha="center", va="center", fontsize=22)
fig.tight_layout(w_pad=3.5); save(fig, "fig-stability.svg")
print("df extent", np.abs(df).max(axis=1))

# ---------------------------------------------------------------- 5. tomography: object and sinogram
from skimage.transform import radon
ph = np.pad(img, 0); yy, xx = np.mgrid[:n, :n]; ph = ph * (np.hypot(yy - n / 2, xx - n / 2) < n / 2)
theta = np.linspace(0, 180, 180, endpoint=False)
sino = radon(ph, theta=theta, circle=True)
fig, ax = plt.subplots(1, 2, figsize=(8.6, 4.2), gridspec_kw=dict(width_ratios=[1, 1.4]))
ax[0].imshow(ph, cmap="gray"); ax[0].axis("off"); ax[0].set_title("object", fontsize=19)
ax[1].imshow(sino, cmap="gray", aspect="auto", extent=[0, 180, n / 2, -n / 2])
ax[1].set_xlabel("angle (degrees)"); ax[1].set_ylabel("detector position"); ax[1].set_title("measured: line integrals", fontsize=19)
ax[1].set_xticks([0, 90, 180]); ax[1].set_yticks([])
fig.tight_layout(); save(fig, "fig-radon.png", dpi=100)
