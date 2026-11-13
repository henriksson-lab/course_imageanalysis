# Figures for the fourier unit (fourier/lec.qmd). Run: python3 -I tools/figs/fourier/figs.py
# Figure sizes are the display size in px / 96, so 13 pt text shows at about 17 px.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/fourier/'
plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, GREY = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c", "#888"
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)

# ------------------------------------------------------------ 1. square wave from odd sines
t = np.linspace(0, 1, 1001)
def partial(fmax): return sum(np.sin(2 * np.pi * f * t) / f for f in range(1, fmax + 1, 2))
sq = np.pi / 4 * np.sign(np.sin(2 * np.pi * t))         # the limit of 1·sin + 1/3·sin3 + ...
fig, ax = plt.subplots(figsize=(480 / 96, 300 / 96))
ax.plot(t, sq, color=GREY, lw=1.5, ls="--", label="square (limit)")
for fmax, c, lab in [(1, "#9ecae1", "f = 1"), (5, BLUE, "f = 1, 3, 5 (demo)"), (25, RED, "f = 1, 3, …, 25")]:
    ax.plot(t, partial(fmax), color=c, lw=2.4 if fmax == 5 else 1.8, label=lab)
ax.set_xlabel("position in the window"); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0])
ax.axhline(0, color="#ccc", lw=1, zorder=0)
ax.legend(fontsize=12, frameon=False, loc="upper right", bbox_to_anchor=(1.02, 1.2), ncol=2, columnspacing=1)
ax.set_ylim(-1.15, 1.15)
fig.subplots_adjust(left=0.06, right=0.98, top=0.8, bottom=0.17)
save(fig, "fig-square.svg")

# ------------------------------------------------------------ 2. the DFT matrix as an image
N = 16
k, n = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
F = np.exp(-2j * np.pi * k * n / N)
fig, ax = plt.subplots(1, 2, figsize=(345 / 96, 215 / 96))
for a, M, t_ in zip(ax, [F.real, F.imag], ["real part", "imaginary"]):
    a.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    a.set_xticks([]); a.set_yticks([]); a.set_title(t_, fontsize=14, pad=4)
    a.set_xlabel("n →", fontsize=14, labelpad=3)
    for s in a.spines.values(): s.set_visible(True)
ax[0].set_ylabel("← k", fontsize=14, labelpad=3)
fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.12, wspace=0.06)
save(fig, "fig-dftmatrix.svg")

# ------------------------------------------------------------ 3. plane waves and their spectra
Nn = 256
y, x = np.mgrid[0:Nn, 0:Nn]
waves = [("horizontal stripes,\n8 px apart", np.cos(2 * np.pi * y / 8)),
         ("diagonal stripes", np.cos(2 * np.pi * (16 * x + 16 * y) / Nn)),
         ("vertical stripes,\n32 px apart", np.cos(2 * np.pi * x / 32))]
fig, ax = plt.subplots(2, 3, figsize=(560 / 96, 400 / 96), gridspec_kw={"height_ratios": [1, 1.05]})
for j, (lab, img) in enumerate(waves):
    ax[0, j].imshow(img[:96, :96], cmap="gray", interpolation="bilinear")   # top-left corner, for visibility
    ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
    ax[0, j].set_title(lab, fontsize=14, pad=3, linespacing=1.0)
    X = np.fft.fftshift(np.abs(np.fft.fft2(img)))
    pk = np.argwhere(X > 0.25 * X.max()) - Nn // 2                         # (row, col) offsets from the centre
    a = ax[1, j]
    a.axhline(0, color="#ddd", lw=1, zorder=0); a.axvline(0, color="#ddd", lw=1, zorder=0)
    a.plot(0, 0, "+", color=GREY, ms=10, mew=1.5)
    a.plot(pk[:, 1], -pk[:, 0], "o", color=RED, ms=8)                      # v up on the plot
    for r, c in pk:
        if c > 0 or (c == 0 and -r > 0):
            a.text(c + 4 if c < 10 else c, -r + 5 if c < 10 else -r - 8, f"({c}, {-r})", color=RED, fontsize=13.5,
                   ha="left" if c < 10 else "center", va="bottom" if c < 10 else "top")
    a.set_xlim(-48, 48); a.set_ylim(-48, 48); a.set_aspect("equal")
    a.set_xticks([-32, 0, 32]); a.set_yticks([-32, 0, 32]); a.tick_params(labelsize=12.5)
    for s in a.spines.values(): s.set_visible(True); s.set_edgecolor("#bbb")
    if j: a.set_yticklabels([])
    print(lab.replace("\n", " "), "peaks at", [(int(c), int(-r)) for r, c in pk])
ax[1, 0].set_ylabel("spectrum", fontsize=14)
fig.subplots_adjust(left=0.14, right=0.99, top=0.89, bottom=0.07, wspace=0.12, hspace=0.27)
save(fig, "fig-planewaves.svg")

# ------------------------------------------------------------ 4. frequency responses
f = np.linspace(0, 0.5, 400)
H_blur = np.cos(np.pi * f) ** 2                    # (1, 2, 1)/4  ->  (1 + cos 2πf)/2
H_curv = 4 * np.sin(np.pi * f) ** 2                # |[1, -2, 1]| ->  |2 cos 2πf − 2|
fig, ax = plt.subplots(figsize=(420 / 96, 300 / 96))
ax.plot(f, H_blur, color=BLUE, lw=2.6, label="blur ¼(1, 2, 1)")
ax.plot(f, H_curv / 4, color=RED, lw=2.6, ls="--", label="curvature (1, −2, 1), ÷ 4")
for ff, hv, lab in [(0, 1, "flat: 1"), (0.25, 0.5, "slow: 0.5"), (0.5, 0, "fastest: 0")]:
    ax.plot(ff, hv, "o", color=BLUE, ms=8, zorder=5)
ax.text(0.02, 1.02, "flat", color=BLUE, fontsize=13.5, va="bottom")
ax.text(0.235, 0.51, "slow", color=BLUE, fontsize=13.5, ha="right", va="bottom")
ax.text(0.485, 0.08, "fastest", color=BLUE, fontsize=13.5, ha="right", va="bottom")
ax.set_xticks([0, 0.25, 0.5]); ax.set_xticklabels(["0", "¼", "½"])
ax.set_xlabel("frequency (cycles per pixel)"); ax.set_ylabel("multiplied by")
ax.set_ylim(-0.05, 1.15); ax.set_yticks([0, 0.5, 1]); ax.set_xlim(-0.01, 0.52)
ax.legend(fontsize=13.5, frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.3), ncol=1)
fig.subplots_adjust(left=0.17, right=0.97, top=0.79, bottom=0.17)
save(fig, "fig-response.svg")

# ------------------------------------------------------------ 5. aliasing
nn = np.arange(0, 41)
tt = np.linspace(0, 40, 801)
fig, ax = plt.subplots(figsize=(600 / 96, 220 / 96))
ax.plot(tt, np.cos(2 * np.pi * tt / 5), color="#bbb", lw=1.2)
ax.plot(nn, np.cos(2 * np.pi * nn / 5), "o", color=GREY, ms=4, label="all pixels: period 5")
ks = nn[::4]
ax.plot(tt, np.cos(2 * np.pi * tt / 20), color=RED, lw=2, ls="--", label="every 4th pixel: period 20")
ax.plot(ks, np.cos(2 * np.pi * ks / 5), "o", color=RED, ms=9, zorder=5)
assert np.allclose(np.cos(2 * np.pi * ks / 5), np.cos(2 * np.pi * ks / 20))
ax.set_xlabel("pixel"); ax.set_yticks([-1, 0, 1]); ax.set_xticks(range(0, 41, 4)); ax.set_xlim(-0.5, 40.5)
ax.legend(fontsize=13, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=2)
fig.subplots_adjust(left=0.06, right=0.99, top=0.8, bottom=0.21)
save(fig, "fig-alias.svg")
