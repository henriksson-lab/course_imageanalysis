"""Static figures for the deconvolution day (deconv/, regularise/, richardson-lucy/, denoise/).

Data come from data.ts (the demos' own code), run first:
  node_modules/.bin/esbuild tools/figs/deconvolution/data.ts --bundle --platform=node --log-level=warning | node - data.json
  python3 -I tools/figs/deconvolution/figs.py data.json .
Outputs go into the unit folders as fig-*.svg / fig-*.png. Numbers used in captions are printed.
"""
import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = json.load(open(sys.argv[1]))
ROOT = sys.argv[2]
N = 128
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#777777"
plt.rcParams.update({
    "font.size": 16, "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14,
    "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "font.family": "DejaVu Sans",
    "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.8, "figure.facecolor": "white",
    "lines.linewidth": 2.4, "savefig.facecolor": "white",
})


def save(fig, unit, name):
    p = os.path.join(ROOT, unit, name)
    fig.savefig(p, dpi=150 if name.endswith(".png") else None, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", p)


def img(a, n=N):
    return np.asarray(a, dtype=float).reshape(-1, n)


def show(ax, a, title, lo=0.0, hi=1.0, cmap="gray", fs=17):
    ax.imshow(a, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title(title, fontsize=fs)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# =============================================================================== deconv/
U = "deconv"
# --- the inverse filter: |H| and 1/|H| for a Gaussian PSF, sigma = 2 px
f = np.linspace(0, 0.5, 400)
H = np.exp(-2 * np.pi ** 2 * 4 * f ** 2)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.semilogy(f, H, color=BLUE, label="|H|: what the optics pass")
ax.semilogy(f, 1 / H, color=RED, label="1/|H|: the inverse filter's gain")
ax.axhline(1, color=GREY, lw=1)
f3 = np.sqrt(np.log(1000) / (2 * np.pi ** 2 * 4))
ax.plot([f3, f3], [1e-3, 1e3], color=GREY, ls=":", lw=1.5)
ax.plot([f3], [1e3], "o", color=RED); ax.plot([f3], [1e-3], "o", color=BLUE)
ax.annotate("|H| = 0.001: noise × 1000", (f3, 1e3), xytext=(0.03, 3e4), fontsize=14, arrowprops=dict(arrowstyle="->", color=GREY))
ax.set_ylim(1e-6, 1e6); ax.set_xlim(0, 0.5)
ax.set_xlabel("frequency (cycles/pixel)"); ax.set_ylabel("gain (log scale)")
ax.legend(loc="lower left", frameon=False)
save(fig, U, "fig-inverse-gain.svg")
print("  f where |H| = 0.001:", f3)

# --- truncated inverse: gain curves (Airy NA 1.0) and the beads image
tr = data["truncated"]
aH = np.array(data["airyH"]); fk = np.arange(len(aH)) / N
with plt.rc_context({"font.size": 21, "axes.labelsize": 21, "xtick.labelsize": 19, "ytick.labelsize": 19, "legend.fontsize": 18}):
    fig = plt.figure(figsize=(6.4, 7.0))
    ax = fig.add_axes([0.15, 0.62, 0.82, 0.37])
    inv = np.where(aH > 1e-10, 1 / np.maximum(aH, 1e-12), np.nan)
    ax.semilogy(fk, aH, color=BLUE, label="|H| (Airy, NA 1.0)")
    ax.semilogy(fk, inv, color=GREY, ls="--", label="inverse 1/|H|")
    trg = np.where(aH > 0.1, 1 / np.maximum(aH, 1e-12), np.nan)
    ax.semilogy(fk, trg, color=RED, lw=3, label="truncated, ε = 0.1")
    cut = fk[np.argmax(aH < 0.1)]
    ax.plot([cut, cut], [1e-3, 10], color=RED, lw=3)
    ax.plot([cut, 0.5], [1e-3 * 1.05, 1e-3 * 1.05], color=RED, lw=3)
    ax.set_ylim(1e-3, 1e5); ax.set_xlim(0, 0.5); ax.set_yticks([1e-3, 1e-1, 1e1, 1e3, 1e5])
    ax.set_xlabel("frequency (cycles/pixel)"); ax.set_ylabel("gain")
    ax.legend(frameon=False, loc="upper right", handlelength=1.4)
    for i, (a, t) in enumerate([(tr["obs"], "observed\n1 % noise"), (tr["x"], f"truncated\nRMSE {tr['rmse']:.4f}")]):
        a2 = fig.add_axes([0.02 + i * 0.5, 0.0, 0.47, 0.42])
        show(a2, img(a)[10:90, 14:110], t, 0, 0.8, fs=22)
save(fig, U, "fig-truncated.png")

# --- Wiener gain against |H|
h = np.logspace(-2.3, 0, 400)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.semilogx(h, 1 / h, color=GREY, ls="--", label="inverse 1/H")
for snr, c in [(10, ORANGE), (100, BLUE), (1000, PURPLE)]:
    ax.semilogx(h, h / (h ** 2 + 1 / snr), color=c, label=f"Wiener, SNR {snr}")
ax.plot([0.5], [0.5 / 0.26], "o", color=BLUE, ms=8); ax.annotate("1.92 (inverse: 2)", (0.5, 1.92), xytext=(0.28, 6.5), fontsize=14, arrowprops=dict(arrowstyle="->", color=GREY))
ax.plot([0.05], [4], "o", color=BLUE, ms=8); ax.annotate("4 (inverse: 20)", (0.05, 4), xytext=(0.012, 10.5), fontsize=14, arrowprops=dict(arrowstyle="->", color=GREY))
ax.plot([0.05], [20], "o", color=GREY, ms=7, mfc="white")
ax.set_ylim(0, 22); ax.set_xlim(0.005, 1)
ax.set_xlabel("|H| at that frequency (log scale)"); ax.set_ylabel("gain")
ax.legend(frameon=False, loc="upper right")
save(fig, U, "fig-wiener-gain.svg")

# --- power spectrum of the cell micrograph: signal, blurred signal, noise
from skimage import io
cell = io.imread(os.path.join(ROOT, "demos/src/fourier2d/cell256.png")).astype(float)
if cell.ndim == 3:
    cell = cell[..., :3].mean(axis=2)
cell /= 255.0
n2 = cell.shape[0]
F = np.fft.fft2(cell - cell.mean())
Pw = np.abs(F) ** 2 / n2 ** 2   # power per pixel: white noise of SD s has s² at every frequency
fy, fx = np.meshgrid(np.fft.fftfreq(n2), np.fft.fftfreq(n2), indexing="ij")
fr = np.hypot(fx, fy)
bins = np.linspace(0.004, 0.5, 60)
idx = np.digitize(fr.ravel(), bins)
fc = np.array([fr.ravel()[idx == i].mean() for i in range(1, len(bins))])
pc = np.array([Pw.ravel()[idx == i].mean() for i in range(1, len(bins))])
Hc = np.exp(-2 * np.pi ** 2 * 4 * fc ** 2)
sn = 0.01
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.loglog(fc, pc, color=BLUE, label="signal power (micrograph)")
ref = pc[3] * (fc[3] / fc) ** 2
ax.loglog(fc, ref, color=GREY, ls="--", lw=1.6, label="∝ 1/f²")
ax.loglog(fc, pc * Hc ** 2, color=ORANGE, label="after blur, σ = 2 px")
ax.axhline(sn ** 2, color=RED, lw=2, label="noise power, σn = 1 %")
cross = fc[np.argmax(pc * Hc ** 2 < sn ** 2)]
ax.axvspan(cross, 0.5, color=RED, alpha=0.07)
ax.text(cross * 1.06, 3e-9, "noise\nwins", color=RED, fontsize=14)
ax.set_ylim(1e-9, pc.max() * 3); ax.set_xlim(fc[0], 0.5)
ax.set_xlabel("frequency (cycles/pixel, log)"); ax.set_ylabel("power per pixel (log)")
ax.legend(frameon=False, fontsize=14, loc="lower left")
save(fig, U, "fig-power-spectrum.svg")
print("  power spectrum: blurred signal falls below noise at f =", cross)

# --- Tikhonov (roughness) for three lambdas
tk = data["tikhonov"]
fig, axs = plt.subplots(2, 2, figsize=(6.4, 7.0))
axs = axs.ravel()
show(axs[0], img(tk["obs"]), f"observed\nRMSE {tk['rmseObs']:.4f}", 0, 0.8, fs=21)
labels = ["λ = 10⁻⁴: noisy", "λ = 0.033", "λ = 1: blurred"]
for a, r, lab in zip(axs[1:], tk["runs"], labels):
    show(a, img(r["x"]), f"{lab}\nRMSE {r['rmse']:.4f}", 0, 0.8, fs=21)
fig.tight_layout()
save(fig, U, "fig-tikhonov-lambda.png")

# --- wrong PSF width: Wiener SNR 85 through the single bead
wp = data["wrongpsf"]; xs = np.arange(4, 44)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(xs, data["wrongpsfTruth"], color=GREY, ls="--", lw=2, label="truth")
for r, c, lab in zip(wp, [BLUE, GREEN, RED], ["PSF 0.8 × true", "right PSF", "PSF 1.3 × true"]):
    ax.plot(xs, r["row"], color=c, label=lab)
ax.axhline(0.05, color=GREY, lw=1, ls=":")
ax.set_xlabel("x through the single bead (px)"); ax.set_ylabel("value")
ax.set_ylim(-0.02, 0.6)
ax.legend(frameon=False, fontsize=13, loc="upper right")
ax.annotate("dark rims", (18, 0.02), xytext=(6, 0.3), color=RED, fontsize=14, arrowprops=dict(arrowstyle="->", color=RED))
save(fig, U, "fig-wrongpsf.svg")
print("  wrong PSF row minima:", [(r["used"], r["min"]) for r in wp])

# --- borders: what the FFT sees, periodic vs padded
bd = data["border"]
obs = img(bd["obs"])
fig = plt.figure(figsize=(6.4, 6.9))
tile = np.block([[obs, obs], [obs, obs]])[64:192, 32:224]
a0 = fig.add_axes([0.1, 0.55, 0.8, 0.40])
show(a0, tile, "what the FFT sees: copies meet", 0, 0.8, fs=21)
a0.axvline(95.5, color=ORANGE, lw=1.5, ls="--"); a0.axhline(63.5, color=ORANGE, lw=1.5, ls="--")
a1 = fig.add_axes([0.0, 0.0, 0.48, 0.40]); a2 = fig.add_axes([0.52, 0.0, 0.48, 0.40])
show(a1, np.clip(img(bd["periodic"]), -0.2, 1.0), f"periodic\nRMSE {bd['rmseP']:.2f}", 0, 0.8, fs=21)
show(a2, img(bd["padded"]), f"padded + tapered\nRMSE {bd['rmseQ']:.4f}", 0, 0.8, fs=21)
save(fig, U, "fig-borders.png")

# =============================================================================== regularise/
U = "regularise"
rp = data["regpath"]; xs = np.arange(256)
with plt.rc_context({"font.size": 19, "axes.labelsize": 19, "xtick.labelsize": 17, "ytick.labelsize": 17}):
    fig, axs = plt.subplots(3, 1, figsize=(6.4, 6.6), sharex=True)
    for ax, key, lab in zip(axs, ["small", "best", "large"], ["λ = 10⁻⁶: fits the noise", "λ = 0.1: balanced", "λ = 100: too smooth"]):
        ax.plot(xs, rp["truth"], color=GREY, ls="--", lw=2)
        ax.plot(xs, rp["y"], color=BLUE, lw=1.2)
        ax.plot(xs, np.clip(rp["est"][key], -0.6, 1.6), color=ORANGE, lw=2.4)
        ax.set_title(f"{lab}: error {rp['errs'][key]:.3f}", fontsize=19)
        ax.set_ylim(-0.4, 1.3); ax.set_ylabel("value")
    axs[2].set_xlabel("position")
    axs[2].legend(["truth", "data", "estimate"], frameon=False, fontsize=16, loc="upper right")
    fig.tight_layout()
save(fig, U, "fig-pattern.svg")

path = rp["path"]
lam = np.array([q["lam"] for q in path]); err = np.array([q["err"] for q in path]); res = np.array([q["res"] for q in path])
ib = int(np.argmin(err))
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.loglog(lam, err, color=RED, label="error against the truth")
ax.loglog(lam, res, color=BLUE, label="RMS residual")
ax.axhline(0.02, color=GREEN, ls="--", lw=2, label="noise σn = 0.02")
ax.axvline(rp["lamDisc"], color=GREEN, ls=":", lw=2)
ax.axvline(lam[ib], color=RED, ls=":", lw=2)
ax.text(rp["lamDisc"] * 1.3, 0.004, f"discrepancy\nλ = {rp['lamDisc']:.2f}", color=GREEN, fontsize=15)
ax.text(lam[ib] / 1.4, 0.004, f"best\nλ = {lam[ib]:.1f}", color=RED, fontsize=15, ha="right")
ax.set_xlabel("λ"); ax.set_ylabel("RMS"); ax.set_ylim(2.5e-3, 4)
ax.legend(frameon=False, fontsize=14, loc="upper right")
save(fig, U, "fig-discrepancy.svg")
print("  regpath: disc", rp["lamDisc"], "corner", rp["lamCorner"], "best", lam[ib], err[ib])

rho = np.array([q["rho"] for q in path]); eta = np.array([q["eta"] for q in path]); ic = rp["corner"]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.loglog(rho, eta, color=PURPLE, marker="o", ms=3.5)
ax.plot([rho[ic]], [eta[ic]], "o", color=RED, ms=11, label=f"corner, λ = {rp['lamCorner']:.3f}")
ax.plot([rp["discPt"]["rho"]], [rp["discPt"]["eta"]], "o", color=GREEN, ms=11, label=f"discrepancy, λ = {rp['lamDisc']:.2f}")
ax.text(rho[8] * 1.08, eta[8], "small λ: the\npenalty explodes", fontsize=15, va="center")
ax.text(rho[-4], eta[-1] / 1.6, "large λ: the fit gets worse", fontsize=15, ha="center", va="top"); ax.set_ylim(eta[-1] / 5, None)
ax.set_xlabel("residual norm ‖Ax − y‖ (log)"); ax.set_ylabel("roughness ‖Dx‖ (log)")
ax.legend(frameon=False, fontsize=14, loc="center right")
save(fig, U, "fig-lcurve.svg")

lw = data["landweber"]
ks = np.array([q["k"] for q in lw]); le = np.array([q["err"] for q in lw]); lr = np.array([q["res"] for q in lw])
with plt.rc_context({"font.size": 20, "axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 17}):
    s = np.logspace(-2, 0, 300)
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 7.6))
    for k, c in [(1, ORANGE), (4, GREEN), (16, BLUE), (100, PURPLE)]:
        axs[0].semilogx(s, 1 - (1 - s ** 2) ** k, color=c, label=f"k = {k}")
    axs[0].set_xlabel("gain of the blur s (log)"); axs[0].set_ylabel("kept fraction φ")
    axs[0].legend(frameon=False, loc="upper left")
    kb = int(np.argmin(le))
    axs[1].loglog(ks, le, color=RED, label="error against the truth")
    axs[1].loglog(ks, lr, color=BLUE, label="RMS residual")
    axs[1].axhline(0.02, color=GREEN, ls="--", lw=2, label="σn")
    axs[1].plot([ks[kb]], [le[kb]], "o", color=RED, ms=9)
    axs[1].annotate(f"best: k = {ks[kb]}", (ks[kb], le[kb]), xytext=(ks[kb] * 0.9, 0.005), fontsize=18, ha="center", arrowprops=dict(arrowstyle="->", color=GREY))
    axs[1].set_xlabel("iterations k"); axs[1].set_ylabel("RMS"); axs[1].set_ylim(2e-3, 3)
    axs[1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
save(fig, U, "fig-landweber.svg")
print("  landweber best", ks[kb], le[kb])

# --- MAP: prior x likelihood -> posterior for one value
t = np.linspace(-3, 5, 600)
g = lambda m, sd: np.exp(-(t - m) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi))
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(t, g(0, 1), color=GREEN, label="prior: what we expect (τ = 1)")
ax.plot(t, g(2, 1), color=BLUE, label="likelihood: what the data say (σn = 1)")
ax.plot(t, g(1, np.sqrt(0.5)), color=ORANGE, lw=3, label="posterior: both (λ = 1)")
ax.axvline(1, color=ORANGE, ls=":", lw=2)
ax.annotate("most probable\nvalue: 1", (1.05, 0.56), xytext=(2.5, 0.52), color=ORANGE, fontsize=15, arrowprops=dict(arrowstyle="->", color=ORANGE))
ax.set_xlabel("value of one pixel difference"); ax.set_ylabel("probability density")
ax.set_ylim(0, 0.98)
ax.legend(frameon=False, fontsize=14, loc="upper left")
save(fig, U, "fig-map.svg")

# =============================================================================== richardson-lucy/
U = "richardson-lucy"
ph = data["photons"]
wi = ph["wiener"][1]
fig, axs = plt.subplots(1, 3, figsize=(12, 4.0))
show(axs[0], img(ph["obs"]), "observed: 100 photons at 1", 0, 0.8)
w = img(wi["x"])
rgb = np.repeat(np.clip(w / 0.8, 0, 1)[..., None], 3, axis=2)
rgb[w < 0] = [0.84, 0.15, 0.16]
axs[1].imshow(rgb, interpolation="nearest"); show(axs[1], rgb, f"Wiener, SNR {wi['snr']:.0f}:\n{100 * wi['neg']:.1f} % negative (red)")
axs[1].imshow(rgb, interpolation="nearest")
show(axs[2], img(ph["rl5"]), f"Richardson–Lucy, 5 iterations:\nnone negative", 0, 0.8)
fig.tight_layout()
save(fig, U, "fig-photons.png")
print("  photons: wiener", [(q["snr"], q["neg"], q["err"]) for q in ph["wiener"]], "RL", ph["rlErr"], ph["rlMin"])

# --- the RL loop, 1D
n1 = 120; x1 = np.arange(n1)
truth1 = 0.2 + np.exp(-(x1 - 52) ** 2 / 4) + 0.6 * np.exp(-(x1 - 64) ** 2 / 4)
kr = np.exp(-np.arange(-15, 16) ** 2 / (2 * 4.0 ** 2)); kr /= kr.sum()
bl = lambda v: np.convolve(np.pad(v, 15, mode="wrap"), kr, mode="valid")
y1 = bl(truth1)
g1 = np.full(n1, y1.mean())
for _ in range(3):
    g1 = g1 * bl(y1 / bl(g1))
bg1 = bl(g1); ratio = y1 / bg1; back = bl(ratio); new = g1 * back
fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.6))
axs = axs.ravel()
axs[0].plot(x1, g1, color=ORANGE, label="guess"); axs[0].plot(x1, bg1, color=PURPLE, label="blurred guess"); axs[0].plot(x1, y1, color=BLUE, ls="--", label="data")
axs[0].set_title("1. blur the guess", fontsize=17); axs[0].legend(frameon=False, fontsize=13, loc="upper left")
axs[1].plot(x1, ratio, color=GREEN); axs[1].axhline(1, color=GREY, lw=1); axs[1].set_title("2. ratio data / blurred", fontsize=17)
axs[2].plot(x1, back, color=GREEN); axs[2].axhline(1, color=GREY, lw=1); axs[2].set_title("3. spread the ratio back", fontsize=17)
axs[3].plot(x1, g1, color=ORANGE, lw=1.5, ls=":", label="old guess"); axs[3].plot(x1, new, color=ORANGE, label="new guess"); axs[3].plot(x1, truth1, color=GREY, ls="--", lw=1.5, label="truth")
axs[3].set_title("4. multiply", fontsize=17); axs[3].legend(frameon=False, fontsize=13, loc="upper left")
for a in axs:
    a.set_xlim(25, 95); a.set_xticks([])
axs[0].set_ylim(0, 2.0); axs[3].set_ylim(0, 2.0)
rr = max(abs(ratio[25:95] - 1).max(), abs(back[25:95] - 1).max()) * 1.3
axs[1].set_ylim(1 - rr, 1 + rr); axs[2].set_ylim(1 - rr, 1 + rr)
fig.tight_layout()
save(fig, U, "fig-rl-loop.svg")

# --- semi-convergence for three photon levels
sm = data["semi"]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for ph_, c in [("1000", BLUE), ("100", ORANGE), ("20", RED)]:
    cv = sm[ph_]["curve"]; kk = np.array([max(1, q["k"]) for q in cv]); ee = np.array([q["err"] for q in cv])
    ax.loglog(kk, ee, color=c, label=f"{ph_} photons: best k = {sm[ph_]['bestK']}")
    ax.plot([sm[ph_]["bestK"]], [sm[ph_]["bestErr"]], "o", color=c, ms=9)
ax.set_xlabel("iterations k"); ax.set_ylabel("error (% of the peak)")
ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1000"])
ax.set_yticks([2, 5, 10, 20]); ax.set_yticklabels(["2", "5", "10", "20"]); ax.minorticks_off()
ax.legend(frameon=False, fontsize=14, loc="upper center")
save(fig, U, "fig-semiconv.svg")

# --- background
bgd = data["background"]
fig, axs = plt.subplots(2, 1, figsize=(3.6, 7.4))
show(axs[0], img(bgd["off100"]), f"background left out\n(k = 100): {bgd['off100Err']:.1f} %", 0, 0.8, fs=19)
show(axs[1], img(bgd["on10"]), f"background in model\n(k = 10): {bgd['on10Err']:.2f} %", 0, 0.8, fs=19)
fig.tight_layout()
save(fig, U, "fig-background.png")

# --- 3D: xz slices and the OTF with the missing cone
td = data["threeD"]; NX, NZ, zs, px = td["NX"], td["NZ"], td["zstep"], td["px"]
ext = [-NX / 2 * px / 1000, NX / 2 * px / 1000, NZ / 2 * zs / 1000, -NZ / 2 * zs / 1000]
xz = lambda a: np.asarray(a).reshape(NZ, NX)
def otfpanel(ax, o, title):
    o = xz(o)
    kxmax = 1 / (2 * px) * 1000; kzmax = 1 / (2 * zs) * 1000   # cycles/µm
    ax.imshow(np.log10(np.maximum(o, 1e-3)), cmap="viridis", vmin=-3, vmax=0, extent=[-kxmax, kxmax, -kzmax, kzmax], aspect="auto", interpolation="nearest", origin="lower")
    al = np.arcsin(td["NA"]); kr = 1.2 * td["NA"] / td["wavelength"] * 1000
    for sgn in (1, -1):
        for sz in (1, -1):
            ax.plot([0, sgn * kr], [0, sz * kr * np.tan(al)], color="white", ls="--", lw=1.5)
    ax.set_title(title, fontsize=14); ax.grid(False)
    ax.set_xlabel("kx (cycles/µm)")


with plt.rc_context({"font.size": 19, "axes.labelsize": 19, "xtick.labelsize": 17, "ytick.labelsize": 17}):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 7.6))
    axs = axs.ravel()
    for a, k, t_, hi in [(axs[0], "truth", "truth", 1.2), (axs[1], "obs", "observed", None), (axs[2], "rl", "RL, k = 100", None)]:
        v = xz(td[k]); hi = hi or np.percentile(v, 99.7)
        a.imshow(v, cmap="gray", vmin=0, vmax=hi, extent=ext, interpolation="nearest")
        a.set_title(t_ + " (xz)", fontsize=19); a.grid(False)
        a.set_xticks([-3, 0, 3]); a.set_yticks([-3, 0, 3])
    axs[0].set_ylabel("z (µm)"); axs[2].set_ylabel("z (µm)"); axs[2].set_xlabel("x (µm)")
    axs[2].annotate("sheet", (0.9, 0.3), xytext=(1.0, 2.6), color=ORANGE, fontsize=18, arrowprops=dict(arrowstyle="->", color=ORANGE))
    otfpanel(axs[3], td["otfW"], "OTF: missing cone")
    axs[3].set_ylabel("kz (cycles/µm)"); axs[3].title.set_fontsize(19)
    fig.tight_layout()
save(fig, U, "fig-3d-xz.png")


fig, axs = plt.subplots(2, 1, figsize=(5.0, 7.0))
otfpanel(axs[0], td["otfW"], "widefield: a cone is missing")
otfpanel(axs[1], td["otfC"], "confocal: the cone is filled")
axs[0].set_ylabel("kz (cycles/µm)"); axs[1].set_ylabel("kz (cycles/µm)")
for a in axs: a.title.set_fontsize(18)
fig.tight_layout()
save(fig, U, "fig-otf-cone.png")

# --- PSF mismatch rows (RL k = 100)
mm = data["psfmismatch"]; xs = np.arange(4, 44)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(xs, data["psfmismatchTruth"], color=GREY, ls="--", lw=2, label="truth")
for r, c, lab in zip(mm, [BLUE, GREEN, RED], ["PSF 0.7 × true", "right PSF", "PSF 1.3 × true"]):
    ax.plot(xs, r["row"], color=c, label=lab)
ax.axhline(0.05, color=GREY, lw=1, ls=":")
rim = [r for r in mm if r["used"] == 1.3][0]["rim"]
ax.annotate(f"rim: {rim:.3f}\n(background 0.05)", (19.5, rim + 0.01), xytext=(32, 0.35), fontsize=13, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
ax.set_xlabel("x through the single bead (px)"); ax.set_ylabel("value"); ax.set_ylim(-0.02, 1.0)
ax.legend(frameon=False, fontsize=14, loc="upper left")
save(fig, U, "fig-psf-mismatch.svg")
print("  psf mismatch:", [(r["used"], r["fwhm"], r["rim"]) for r in mm])

# --- blind: PSF width against iterations
bd = data["blind"]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for m, c, lab in [("none", RED, "no constraint"), ("size", ORANGE, "size limit (r ≤ 6 px)"), ("gauss", BLUE, "Gaussian shape")]:
    a = np.array(bd[m]); ax.plot(a[:, 0], a[:, 1], color=c, label=lab)
ax.axhline(bd["trueFwhm"], color=GREY, ls="--", lw=2)
ax.text(300, bd["trueFwhm"] + 0.12, f"true PSF: {bd['trueFwhm']:.1f} px", ha="right", color=GREY, fontsize=15)
ax.set_xlabel("iterations k"); ax.set_ylabel("PSF estimate FWHM (px)"); ax.set_xlim(0, 300); ax.set_ylim(2.5, 7.5)
ax.legend(frameon=False, fontsize=14, loc="lower left")
save(fig, U, "fig-blind.svg")
print("  blind:", {m: (bd[m][100][1], bd[m][300][1]) for m in ("none", "size", "gauss")})

# =============================================================================== denoise/
U = "denoise"
row = np.array([10, 10, 10, 200, 10, 10, 50, 50, 50], float)
pad = np.concatenate([[row[0]], row, [row[-1]]])
mean = np.array([pad[i:i + 3].mean() for i in range(9)]); med = np.array([np.median(pad[i:i + 3]) for i in range(9)])
xs = np.arange(1, 10)
fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.bar(xs, row, color="#c9d9ea", width=0.75, label="data")
ax.plot(xs, mean, color=ORANGE, marker="o", label="mean of 3")
ax.plot(xs, med, color=BLUE, marker="s", label="median of 3")
ax.set_xticks(xs); ax.set_xlabel("pixel"); ax.set_ylabel("value"); ax.set_ylim(0, 210)
ax.legend(frameon=False, fontsize=14, loc="upper right")
save(fig, U, "fig-median.svg")

# --- bilateral weights at one pixel next to a step
pos = np.arange(-5, 6); val = np.where(pos < 2, 10.0, 50.0)
ws = np.exp(-pos ** 2 / (2 * 2.0 ** 2)); wr = np.exp(-(val - 10) ** 2 / (2 * 10.0 ** 2)); wt = ws * wr
with plt.rc_context({"font.size": 21, "axes.labelsize": 21, "xtick.labelsize": 19, "ytick.labelsize": 19, "legend.fontsize": 17}):
    fig, axs = plt.subplots(2, 1, figsize=(6.2, 6.8))
    axs[0].step(pos, val, where="mid", color=GREY, lw=2.4); axs[0].plot([0], [10], "o", color=RED, ms=10)
    axs[0].set_xlabel("position"); axs[0].set_ylabel("value"); axs[0].set_ylim(0, 60)
    wdt = 0.27
    axs[1].bar(pos - wdt, ws, width=wdt, color=BLUE, label="close (σs = 2)")
    axs[1].bar(pos, wr, width=wdt, color=GREEN, label="similar (σr = 10)")
    axs[1].bar(pos + wdt, wt, width=wdt, color=ORANGE, label="product = weight")
    axs[1].set_xlabel("position"); axs[1].set_ylabel("weight"); axs[1].set_ylim(0, 2.3); axs[1].set_yticks([0, 0.5, 1])
    axs[1].legend(frameon=False, loc="upper left", ncol=1, handlelength=1.2)
    fig.tight_layout()
save(fig, U, "fig-bilateral.svg")

# --- method noise
mnz = data["methodnoise"]
dv = ListedColormap(plt.get_cmap("RdBu_r")(np.linspace(0, 1, 256)))
fig, axs = plt.subplots(2, 2, figsize=(6.6, 7.2))
axs = axs.ravel()
show(axs[0], img(mnz["noisy"]), "noisy, σn = 10\n", 0, 255, fs=19)
for a, k, t_ in zip(axs[1:], ["gauss", "bilateral", "nlm"], ["Gaussian σ = 3", "bilateral (2, 30)", "non-local means"]):
    m_ = img(mnz[k]); show(a, m_, f"method noise:\n{t_}", -40, 40, dv, fs=19)
fig.tight_layout()
save(fig, U, "fig-methodnoise.png")
for k in ["gauss", "bilateral", "nlm"]:
    m_ = img(mnz[k]); print("  method noise", k, "flat RMS", m_[72:120, 72:88].std(), "step min/max", m_[8:56, 20:44].min(), m_[8:56, 20:44].max())

# --- measurement field: raw and RL after 10, 100, 1000 iterations
ms_ = data["measure"]
fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.7))
show(axs[0], img(ms_["obs"])[:84], f"observed\n{ms_['readRaw']['count']} spots found", 0, 0.6)
for a, k in zip(axs[1:], ["10", "100", "1000"]):
    show(a, img(ms_["snaps"][k])[:84], f"RL, k = {k}\n{ms_['reads'][k]['count']} spots found", 0, 0.6)
fig.tight_layout()
save(fig, U, "fig-estimate.png")
print("  measure reads", ms_["readRaw"], ms_["reads"])
