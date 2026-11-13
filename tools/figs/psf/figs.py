# Slide figures for psf/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/psf/figs.py
# Writes fig-*.svg / fig-*.png into psf/. Uses demos/src/fourier2d/cell256.png
# (TEM micrograph, Robert M. Hunt, CC BY 3.0, via Wikimedia Commons) and synthetic data.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import j1
from math import pi
plt.rcParams.update({"font.size": 19, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
OUT = "psf/"
def save(fig, name, **kw):
    fig.savefig(OUT + name, **kw); plt.close(fig); print("wrote", OUT + name)
def otf_airy(s):
    s = np.minimum(np.abs(s), 1.0); return 2 / pi * (np.arccos(s) - s * np.sqrt(1 - s * s))
def airy(v):
    v = np.where(np.abs(v) < 1e-9, 1e-9, v); return (2 * j1(v) / v) ** 2

# ---------------------------------------------------------------- 1. sum of shifted PSFs (by hand)
obj = np.zeros(8); obj[[1, 4, 6]] = [1, 2, 1]
k = np.array([1, 2, 1]) / 4
contrib = []
for p in [1, 4, 6]:
    c = np.zeros(8)
    for j, w in zip([-1, 0, 1], k):
        if 0 <= p + j < 8: c[p + j] += obj[p] * w
    contrib.append(c)
img = sum(contrib)
fig, ax = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.5]))
x = np.arange(8)
for i, p in enumerate([1, 4, 6]):
    ax[0].vlines(p, 0, obj[p], color=C[i], lw=5); ax[0].plot(p, obj[p], "o", color=C[i], ms=10)
ax[0].set_ylim(0, 2.5); ax[0].set_yticks([0, 1, 2]); ax[0].set_ylabel("object")
bot = np.zeros(8)
for i, c in enumerate(contrib):
    ax[1].bar(x, c, bottom=bot, color=C[i], width=0.8, edgecolor="white"); bot += c
for xi, v in zip(x, img):
    ax[1].text(xi, v + 0.04, f"{v:g}", ha="center", va="bottom", fontsize=16)
ax[1].set_ylim(0, 1.25); ax[1].set_ylabel("image"); ax[1].set_xlabel("position")
ax[1].set_xticks(x)
fig.tight_layout(); save(fig, "fig-sumofpsfs.svg")

# ---------------------------------------------------------------- 2. gratings through NA 1.4
NA, lam = 1.4, 520.0
kc = 2 * NA / lam * 1000  # cycles/um
pers = [500, 250, 150]
fig = plt.figure(figsize=(7.2, 7.0))
gs0 = fig.add_gridspec(2, 1, height_ratios=[1.3, 1.3], hspace=0.55)
gs = gs0[1].subgridspec(2, 3, hspace=0.08, wspace=0.12)
a0 = fig.add_subplot(gs0[0])
ff = np.linspace(0, 7, 400)
a0.plot(ff, otf_airy(ff / kc), lw=3, color=C[0])
a0.axvline(kc, color="#555", ls=":", lw=2); a0.text(kc + 0.1, 0.72, "cut-off\n5.4 /µm", fontsize=17, color="#555")
xx = np.linspace(0, 1500, 600)
for i, p in enumerate(pers):
    h = otf_airy((1000 / p) / kc)
    a0.plot(1000 / p, h, "o", ms=10, color=C[1]); tx, ty, ha = {500: (2.12, h + 0.06, "left"), 250: (3.8, h - 0.02, "right"), 150: (5.55, 0.1, "left")}[p]
    a0.text(tx, ty, f"{p} nm: {100*h:.0f} %", fontsize=17, ha=ha)
    for r, (val, lab) in enumerate([(1, "object"), (h, "image")]):
        a = fig.add_subplot(gs[r, i])
        g = 0.5 + 0.5 * val * np.cos(2 * pi * xx / p)
        a.imshow(np.tile(g, (60, 1)), cmap="gray", vmin=0, vmax=1, aspect="auto"); a.set_xticks([]); a.set_yticks([])
        if i == 0: a.set_ylabel(lab, fontsize=18)
        if r == 0: a.set_title(f"{p} nm", fontsize=18)
        for s in a.spines.values(): s.set_visible(True)
a0.set_xlabel("spatial frequency (cycles/µm)"); a0.set_ylabel("contrast kept"); a0.set_ylim(0, 1.05); a0.set_xlim(-0.2, 7.6)
save(fig, "fig-gratings.png", dpi=110, bbox_inches="tight")
print("gratings", [round(otf_airy((1000 / p) / kc), 3) for p in pers])

# ---------------------------------------------------------------- 3. colour matters: Airy profiles
fig, ax = plt.subplots(figsize=(6.2, 4.6))
r = np.linspace(0, 450, 451)
for lam_, col in [(450, "#0072b2"), (520, "#009e73"), (650, "#d55e00")]:
    ax.plot(r, airy(2 * pi * 1.4 * np.abs(r) / lam_), lw=3, color=col, label=f"{lam_} nm: {0.61*lam_/1.4:.0f} nm")
    ax.plot([0.61 * lam_ / 1.4], [0], "v", color=col, ms=12, clip_on=False)
ax.set_xlabel("distance from the centre (nm)"); ax.set_ylabel("intensity")
ax.legend(title="λ: first dark ring", fontsize=17, title_fontsize=17, frameon=False, loc="upper right")
ax.set_ylim(0, 1.05)
fig.tight_layout(); save(fig, "fig-colour.svg")

# ---------------------------------------------------------------- 4. other PSFs and their OTFs
N, dx = 512, 40.0
f = np.fft.fftfreq(N, d=dx / 1000)  # cycles/um
FX, FY = np.meshgrid(f, f); rho = np.hypot(FX, FY)
# defocus: NA 0.7 in air, 2 um, 520 nm
NAd, n, z = 0.7, 1.0, 2.0; lu = 0.520
P = (rho <= NAd / lu).astype(float)
phi = 2 * pi * z * (np.sqrt(np.maximum((n / lu) ** 2 - rho ** 2, 0)) - n / lu)
a = np.fft.ifft2(P * np.exp(1j * phi)); hdef = np.abs(a) ** 2; hdef /= hdef.sum()
Hdef = np.real(np.fft.fft2(hdef))
a0_ = np.fft.ifft2(P); hfoc = np.abs(a0_) ** 2; hfoc /= hfoc.sum(); Hfoc = np.real(np.fft.fft2(hfoc))
prof_f = f[:N // 2]; Hd = Hdef[0, :N // 2]; Hf = Hfoc[0, :N // 2]
im = np.argmin(Hd); print("defocus OTF min", Hd[im], "at", prof_f[im])
plt.rcParams.update({"font.size": 22})
fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.2))
def show(a, h, half):
    hh = np.fft.fftshift(h); c = N // 2; m = int(half / dx)
    a.imshow(hh[c - m:c + m, c - m:c + m] / hh.max(), cmap="inferno", extent=[-half / 1000, half / 1000, -half / 1000, half / 1000])
    a.set_xlabel("x (µm)")
show(ax[0, 0], hdef, 1500); ax[0, 0].set_title("defocus: a ring", fontsize=22)
mot = np.zeros((N, N)); L = 1.0
xs = (np.arange(N) - N // 2) * dx / 1000
mot[N // 2 - 1:N // 2 + 2, np.abs(xs) <= L / 2] = 1; mot = np.fft.ifftshift(mot)
show(ax[0, 1], mot, 1500); ax[0, 1].set_title("motion: a line", fontsize=22)
ax[0, 0].set_ylabel("y (µm)")
kk = np.linspace(0, 4, 600)
ax[1, 0].plot(prof_f, Hf, color="#999", lw=2, label="in focus"); ax[1, 0].plot(prof_f, Hd, color=C[0], lw=3, label="defocused")
ax[1, 0].set_xlim(0, 3); ax[1, 0].legend(fontsize=19, frameon=False, handlelength=1.2, loc="upper right", bbox_to_anchor=(1.08, 1.12))
ax[1, 0].fill_between(prof_f, Hd, 0, where=Hd < 0, color=C[1], alpha=0.7)
ax[1, 1].plot(kk, np.sinc(L * kk), color=C[0], lw=3); ax[1, 1].fill_between(kk, np.sinc(L * kk), 0, where=np.sinc(L * kk) < 0, color=C[1], alpha=0.7)
ax[1, 1].plot([1, 2, 3, 4], [0] * 4, "o", color="k", ms=7, clip_on=False); ax[1, 1].set_xlim(0, 4)
for a_ in ax[1]:
    a_.axhline(0, color="k", lw=1); a_.set_xlabel("cycles/µm"); a_.set_ylim(-0.3, 1.05)
ax[1, 0].set_ylabel("OTF along x")
fig.tight_layout(); save(fig, "fig-blurs.png", dpi=100)
plt.rcParams.update({"font.size": 19})

# ---------------------------------------------------------------- 5. shift variance: a bead field with a PSF that changes across the field
M = 400; yy, xx2 = np.mgrid[0:M, 0:M].astype(float)
field = np.zeros((M, M))
cs = np.linspace(40, 360, 6)
for cy in cs:
    for cx in cs:
        ux, uy = (cx - 200) / 200, (cy - 200) / 200; rr = np.hypot(ux, uy)
        s_r = 3.0 + 7.0 * rr ** 2; s_t = 3.0 + 1.5 * rr ** 2  # radial blur grows off-axis
        th = np.arctan2(uy, ux); dxp, dyp = xx2 - cx, yy - cy
        a_r = dxp * np.cos(th) + dyp * np.sin(th); a_t = -dxp * np.sin(th) + dyp * np.cos(th)
        g = np.exp(-a_r ** 2 / (2 * s_r ** 2) - a_t ** 2 / (2 * s_t ** 2)); field += g / g.sum()
field /= field.max()
fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.imshow(field ** 0.6, cmap="gray"); ax.axis("off")
fig.tight_layout(pad=0.1); save(fig, "fig-shiftvariant.png", dpi=100)
