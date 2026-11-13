# Static figures for lsunmix/lec.qmd. Run from the repository root:
#   node_modules/.bin/esbuild tools/figs/lsunmix/export.ts --bundle --platform=node --outfile=<tmp>/x.js
#   node <tmp>/x.js <tmp>
#   python3 -I tools/figs/lsunmix/figs.py <tmp>
# The cell, spectra and unmixing results come from the lsunmix demo's model (16 channels,
# peaks 45 nm apart, 400 photons, seed 1); the 3-channel examples are computed here.
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams.update({"font.size": 17, "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})
COL = ["#3f72ff", "#1fa83a", "#e03cc8", "#d99a12"]          # as in the demo
NAMES = ["nucleus", "mitochondria", "membrane", "autofluor."]
OUT = "lsunmix/"
N = 128; P = N * N

def save(fig, name, **kw):
    fig.savefig(OUT + name, **kw)
    if os.environ.get("PREVIEW"):
        fig.savefig(os.path.join(os.environ["PREVIEW"], name.rsplit(".", 1)[0] + ".png"), dpi=96, bbox_inches=kw.get("bbox_inches"))
    plt.close(fig); print("wrote", OUT + name)

D = json.load(open(sys.argv[1] + "/lsunmix.json"))
wl = np.array(D["wl"]); bands = D["bands"]; cen = np.array([(b["lo"] + b["hi"]) / 2 for b in bands])
bw = bands[0]["hi"] - bands[0]["lo"]
Y = np.array(D["Y"]).reshape(16, N, N); M = np.array(D["M"]).reshape(16, 4)

# 1. fig-stack.png: four of the sixteen band images, and one pixel's spectrum
pix = (60, 55)    # the demo's default pixel (a nucleus pixel)
pix2 = (64, 108)  # a membrane pixel
fig = plt.figure(figsize=(6.0, 5.0))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.45, wspace=0.08)
for i, c in enumerate((2, 6, 10, 13)):
    ax = fig.add_subplot(gs[0, i]); ax.imshow(Y[c], cmap="gray", vmin=0, vmax=np.percentile(Y[c], 99.7), interpolation="nearest")
    ax.set_title(f"{cen[c]:.0f} nm", fontsize=17); ax.axis("off")
    ax.plot(pix[1], pix[0], "+", color="#ffd166", ms=10, mew=2); ax.plot(pix2[1], pix2[0], "x", color="#56b4e9", ms=8, mew=2)
ax = fig.add_subplot(gs[1, :])
for (r, c), col, mk, lab in ((pix, "#c79000", "o", "nucleus (+)"), (pix2, "#2b8cbe", "s", "membrane (×)")):
    ax.plot(cen, Y[:, r, c], marker=mk, color=col, lw=1.5, ms=6, label=lab)
ax.set_xlabel("channel (nm)"); ax.set_ylabel("photons"); ax.legend(frameon=False, fontsize=16, loc="upper right")
ax.set_xlim(478, 702)
fig.subplots_adjust(left=0.13, right=0.95, top=0.93, bottom=0.14)
save(fig, "fig-stack.png", dpi=150)

# 2. fig-pinv.svg: the rows of the pseudo-inverse, the weights of each abundance map
Pi = np.array(D["Pi"]).reshape(4, 16)
fig, ax = plt.subplots(figsize=(5.0, 3.9))
ax.axhline(0, color="black", lw=0.8)
for k in range(4): ax.plot(cen, Pi[k], "o-", color=COL[k], lw=2, ms=4, label=NAMES[k])
ax.set_xlabel("channel (nm)"); ax.set_ylabel("weight in $M^+$"); ax.legend(frameon=False, fontsize=15, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.25), columnspacing=0.8, handlelength=1.5)
ax.set_xlim(478, 702)
fig.tight_layout(); save(fig, "fig-pinv.svg")

# 3. fig-residual.png: residual maps (in units of the expected noise) without and with the autofluorescence
fig, axs = plt.subplots(1, 2, figsize=(5.0, 3.0))
for ax, key, lab in zip(axs, ("chi3", "chi4"), ("3 spectra", "4 spectra")):
    im = ax.imshow(np.array(D[key]).reshape(N, N), cmap="inferno", vmin=0, vmax=2.5, interpolation="nearest")
    ax.set_title(lab, fontsize=16); ax.axis("off")
cb = fig.colorbar(im, ax=axs, fraction=0.04, pad=0.03, ticks=[0, 1, 2]); cb.set_label("residual / noise", fontsize=16)
save(fig, "fig-residual.png", dpi=150, bbox_inches="tight")

# 4. fig-ellipse.svg: the cloud of LS estimates for near-parallel and separated spectra
rng = np.random.default_rng(4); sd = 0.03; a = np.array([1.0, 1.0])
fig, axs = plt.subplots(1, 2, figsize=(5.6, 3.4), sharex=True, sharey=True)
for ax, Mx, lab in zip(axs, (np.array([[1, 0.9], [0.9, 1], [0.5, 0.5]]), np.array([[1, 0.1], [0.1, 1], [0.5, 0.5]])),
                       ("near-parallel", "separated")):
    Pinv = np.linalg.pinv(Mx); ys = Mx @ a + sd * rng.standard_normal((400, 3))
    est = ys @ Pinv.T; s = np.linalg.svd(Mx, compute_uv=False)
    ax.plot(est[:, 0], est[:, 1], ".", ms=3, color="#3f72ff", alpha=0.6)
    ax.plot(*a, "+", color="black", ms=12, mew=2)
    ax.set_title(f"{lab}\nκ = {s[0] / s[-1]:.3g}", fontsize=17); ax.set_aspect("equal")
    ax.set_xlabel("$\\hat a_1$")
axs[0].set_ylabel("$\\hat a_2$"); axs[0].set_xlim(0.2, 1.8); axs[0].set_ylim(0.2, 1.8)
fig.tight_layout(); save(fig, "fig-ellipse.svg")

# 5. fig-close.svg: nucleus and mitochondria spectra 6 nm and 45 nm apart, with the 16 bins
fig, axs = plt.subplots(1, 2, figsize=(5.6, 3.0), sharey=True)
for ax, sep in zip(axs, (6, 45)):
    em = np.array(D["em" + str(sep)])
    for b in bands: ax.axvline(b["lo"], color="#dddddd", lw=0.8, zorder=0)
    for k in (0, 1): ax.plot(wl, em[k], color=COL[k], lw=2.5)
    ax.set_title(f"{sep} nm apart", fontsize=17); ax.set_xlabel("wavelength (nm)"); ax.set_xlim(480, 640)
    ax.set_xticks([500, 550, 600])
axs[0].set_ylabel("emission"); axs[0].set_yticks([0, 1])
fig.tight_layout(); save(fig, "fig-close.svg")

# 6. fig-negative.png: the mitochondria map with negative values in red, and the histogram of all abundances
Ah = np.array(D["Ah"]).reshape(4, N, N)
fig, axs = plt.subplots(2, 1, figsize=(4.4, 5.2), gridspec_kw={"height_ratios": [1.25, 1]})
m = Ah[1]; rgb = np.zeros((N, N, 3))
pos = np.clip(m, 0, 1); rgb[..., 0] = pos * 0.25; rgb[..., 1] = pos * 0.95; rgb[..., 2] = pos * 0.3
neg = m < 0; t = np.clip(-m * 2 / 0.1, 0, 1)    # demo shows −0.5 as full red; here −0.05 for visibility
rgb[neg, 0] = 0.35 + 0.65 * t[neg]
axs[0].imshow(rgb, interpolation="nearest"); axs[0].axis("off"); axs[0].set_title("mitochondria map (red: < 0)", fontsize=17)
v = Ah.ravel(); share = (v < 0).mean()
bins = np.linspace(-0.15, 0.4, 56)
h, e = np.histogram(v, bins)
ax = axs[1]; ax.bar(e[:-1], h, width=np.diff(e), align="edge", color=np.where(e[:-1] < 0, "#d62728", "#888888"))
ax.set_yscale("log"); ax.set_xlabel("abundance (LS)"); ax.set_ylabel("pixels")
ax.text(0.03, h.max() * 0.4, f"{share * 100:.0f} % below 0", color="#d62728", fontsize=17)
fig.tight_layout(); save(fig, "fig-negative.png", dpi=150)
print("negative share", share)
