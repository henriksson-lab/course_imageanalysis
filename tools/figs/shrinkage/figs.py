# Extra slide figures for shrinkage/lec.qmd. Run from the repo root: python3 -I tools/figs/shrinkage/figs.py
# Needs node and esbuild (node_modules/.bin/esbuild): images, noise and denoised results come from
# data.ts, which uses the wavedenoise demo's own code and seeded noise.
import os, json, subprocess, tempfile
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "..", "wavelets", "figs.py")).read().split("# ---- haar/fig-spike-step.svg ----")[0])
os.chdir(R)
N = 256
tmp = tempfile.mkdtemp()
subprocess.run(["node_modules/.bin/esbuild", os.path.join(HERE, "data.ts"), "--bundle", "--platform=node", "--log-level=warning",
                "--outfile=" + os.path.join(tmp, "data.js")], check=True)
subprocess.run(["node", os.path.join(tmp, "data.js"), tmp], check=True)
ld = lambda name, shape=(N, N): np.fromfile(os.path.join(tmp, name + ".f64")).reshape(shape)
info = json.load(open(os.path.join(tmp, "info.json"))); print(info)
cmap = plt.get_cmap("RdBu_r")
def savepng(fig, path): fig.savefig(path, dpi=110, bbox_inches="tight", pad_inches=0.05); plt.close(fig); print("wrote", path)

truth = ld("truth"); noisy = ld("noisy")

# ---- shrinkage/fig-noisy.png: clean and noisy shapes image ----
fig, ax = plt.subplots(1, 2, figsize=(7.6, 4.1))
ax[0].imshow(truth, cmap="gray", vmin=0, vmax=255); ax[0].set_title("true image x", fontsize=17)
ax[1].imshow(np.clip(noisy, 0, 255), cmap="gray", vmin=0, vmax=255); ax[1].set_title(f"y = x + n, σₙ = 20: {info['psnr_noisy']:.1f} dB", fontsize=17)
for a in ax: a.axis("off")
fig.tight_layout(); savepng(fig, "shrinkage/fig-noisy.png")

# ---- shrinkage/fig-maxgauss.svg: largest |z| among N standard Gaussians vs sqrt(2 ln N) ----
rng = np.random.default_rng(1)
Ns = np.unique(np.round(np.logspace(1, 6, 21)).astype(int))
med, lo, hi = [], [], []
for n in Ns:
    m = np.array([np.abs(rng.standard_normal(n)).max() for _ in range(200 if n < 200000 else 60)])
    med.append(np.median(m)); lo.append(np.percentile(m, 5)); hi.append(np.percentile(m, 95))
fig, a = plt.subplots(figsize=(5.8, 4.6))
a.fill_between(Ns, lo, hi, color=PURPLE, alpha=0.2, lw=0, label="largest |noise|: 5–95 %")
a.plot(Ns, med, "o", color=PURPLE, ms=5, label="largest |noise|: median")
nn = np.logspace(1, 6, 200); a.plot(nn, np.sqrt(2 * np.log(nn)), color=RED, lw=2.5, label="t = √(2 ln N)")
a.set_xscale("log"); a.set_xlabel("number of coefficients N"); a.set_ylabel("in units of σₙ")
for n, lab in [(1024, "3.72"), (65536, "4.71")]:
    u = np.sqrt(2 * np.log(n)); a.plot([n], [u], "s", color=RED, ms=8); a.annotate(f"N = {n}: {lab}", (n, u), xytext=(10, -26) if n < 5000 else (-12, 12), textcoords="offset points", fontsize=15, ha="left" if n < 5000 else "right")
a.set_ylim(1, 6); a.legend(frameon=False, fontsize=14, loc="lower right")
fig.tight_layout(); save(fig, "shrinkage/fig-maxgauss.svg")
print("median max at 65536-ish:", dict(zip(Ns[-6:], np.round(med[-6:], 2))))

# ---- shrinkage/fig-finestD.png: the finest D band of the noisy image and the histogram of |w| ----
D = ld("finestD", (128, 128)); sh = info["sigma_hat_db4"]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.9), gridspec_kw={"width_ratios": [1, 1.2]})
mx = 4 * sh
ax[0].imshow(cmap(0.5 + 0.5 * np.clip(D / mx, -1, 1)), interpolation="nearest"); ax[0].axis("off")
ax[0].set_title("finest D band", fontsize=16)
aw = np.abs(D).ravel(); medv = np.median(aw)
ax[1].hist(aw, bins=np.linspace(0, 100, 26), color=PURPLE, alpha=0.75)
ax[1].axvline(medv, color="k", lw=2, ls="--")
ax[1].set_title(f"median |w| = {medv:.1f} → σ̂ₙ = {sh:.1f}", fontsize=15)
ax[1].set_yscale("log"); ax[1].set_xlabel("|w|"); ax[1].set_ylabel("count")
fig.tight_layout(); savepng(fig, "shrinkage/fig-finestD.png")
print("max |D|", aw.max(), "sd", info["sd_D"])

# ---- shrinkage/fig-spin.png: Haar, soft, t = 3 sigma-hat: without and with cycle spinning (crop) ----
y0, x0, s = 24, 24, 112
fig, ax = plt.subplots(2, 1, figsize=(3.4, 6.6))
for a, (img, lab) in zip(ax, [(ld("haar_spin1"), f"no shifts: {info['psnr_haar_spin1']:.1f} dB"), (ld("haar_spin16"), f"16 shifts: {info['psnr_haar_spin16']:.1f} dB")]):
    a.imshow(np.clip(img[y0:y0 + s, x0:x0 + s], 0, 255), cmap="gray", vmin=0, vmax=255, interpolation="nearest"); a.set_title(lab, fontsize=16); a.axis("off")
fig.tight_layout(); savepng(fig, "shrinkage/fig-spin.png")
