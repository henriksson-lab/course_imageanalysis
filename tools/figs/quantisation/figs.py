# Static figures for the quantisation unit. Run from the repository root:
#   python3 -I tools/figs/quantisation/figs.py
# Writes fig-*.svg / fig-*.png into quantisation/. The quantiser is the one of the
# quantise demo (demos/src/quantise/quant.ts): codes k = round(x L), L = 2^b - 1,
# Floyd-Steinberg error diffusion; random dither uses numpy's generator (fixed seed).
# Micrograph: demos/src/fourier2d/cell256.png (Robert M. Hunt, CC BY 3.0, Wikimedia Commons).
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import io

plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
C = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
OUT = "quantisation/"

def save(fig, name, **kw):
    fig.savefig(OUT + name, **kw); plt.close(fig); print("wrote", OUT + name)

def quantise(x, b, dither="none", seed=11):
    L = 2 ** b - 1
    if dither == "none":
        return np.clip(np.round(x * L), 0, L) / L
    if dither == "random":
        u = np.random.default_rng(seed).random(x.shape)
        return np.clip(np.round(x * L + u - 0.5), 0, L) / L
    buf = x.astype(float) * L; h, w = buf.shape; out = np.zeros_like(buf)
    for y in range(h):
        for c in range(w):
            k = min(L, max(0, round(buf[y, c]))); e = buf[y, c] - k; out[y, c] = k / L
            if c + 1 < w: buf[y, c + 1] += e * 7 / 16
            if y + 1 < h:
                if c > 0: buf[y + 1, c - 1] += e * 3 / 16
                buf[y + 1, c] += e * 5 / 16
                if c + 1 < w: buf[y + 1, c + 1] += e / 16
    return out

def blobs(N):
    g = [[0.32, 0.35, 0.16, 0.65], [0.7, 0.3, 0.12, 0.5], [0.55, 0.72, 0.2, 0.75], [0.2, 0.8, 0.09, 0.35]]
    x = (np.arange(N) + 0.5) / N; X, Y = np.meshgrid(x, x)
    v = 0.08 + 0.1 * X
    for cx, cy, s, a in g: v = v + a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s))
    return np.minimum(0.95, v)

cell = io.imread("demos/src/fourier2d/cell256.png").astype(float) / 255

# 1. fig-sample-quant.svg: one continuous profile, sampled (where) and quantised (which values)
t = np.linspace(0, 10, 600)
f = lambda t: 0.5 + 0.3 * np.sin(0.9 * t) + 0.12 * np.sin(2.3 * t + 1)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.3), sharey=True)
ax = axs[0]
ts = np.arange(0, 10.01, 0.8)
ax.plot(t, f(t), color="#999", lw=2)
ax.vlines(ts, 0, f(ts), color=C[0], lw=1.5); ax.plot(ts, f(ts), "o", color=C[0], ms=7)
ax.set_xlabel("position x: read only at grid points"); ax.set_ylabel("brightness")
ax = axs[1]
lv = np.arange(8) / 7
for l in lv: ax.axhline(l, color="#ccc", lw=1, zorder=0)
ax.plot(t, f(t), color="#999", lw=2)
ax.plot(t, np.round(f(t) * 7) / 7, color=C[1], lw=2.5)
ax.set_xlabel("values: rounded to 8 levels (3 bits)")
for a in axs: a.set_xticks([]); a.set_ylim(0, 1)
axs[0].set_yticks([0, 1]); axs[0].set_yticklabels(["black", "white"])
fig.tight_layout(); save(fig, "fig-sample-quant.svg", format="svg")

# 2. fig-bits-cell.png: the micrograph at 1, 2, 3, 4 and 8 bits
bits = [1, 2, 3, 4, 8]
fig, axs = plt.subplots(1, 5, figsize=(12.5, 3.0))
for ax, b in zip(axs, bits):
    ax.imshow(quantise(cell, b), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"{b} bit{'s' if b > 1 else ''}: {2 ** b} levels", fontsize=16); ax.axis("off")
fig.tight_layout(pad=0.3); save(fig, "fig-bits-cell.png", dpi=110)

# 3. fig-sawtooth.svg: a slow ramp, its 3-bit version and the error, with the error histogram
b = 3; L = 2 ** b - 1; D = 1 / L
x = np.linspace(0, 1, 2000); q = np.round(x * L) / L; e = (q - x) / D
fig = plt.figure(figsize=(7.2, 5.6))
gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.25], height_ratios=[1.25, 1], wspace=0.08, hspace=0.35)
ax = fig.add_subplot(gs[0, 0])
ax.plot(x, x, color="#999", lw=2, label="true value")
ax.plot(x, q, color=C[1], lw=2.5, label="rounded, 3 bits")
ax.set_ylabel("value"); ax.set_xticks([]); ax.legend(fontsize=14, frameon=False, loc="upper left")
ax = fig.add_subplot(gs[1, 0])
ax.plot(x, e, color=C[0], lw=2)
for yv in (-0.5, 0.5): ax.axhline(yv, color="#888", ls="--", lw=1)
ax.set_yticks([-0.5, 0, 0.5]); ax.set_yticklabels(["−Δ/2", "0", "+Δ/2"])
ax.set_ylabel("error"); ax.set_xlabel("position along a smooth ramp")
axh = fig.add_subplot(gs[1, 1], sharey=ax)
axh.hist(e, bins=24, range=(-0.6, 0.6), orientation="horizontal", color="#a6c8e4", edgecolor="none")
axh.set_xticks([]); plt.setp(axh.get_yticklabels(), visible=False)
axh.set_xlabel("histogram\n(flat)", fontsize=14)
save(fig, "fig-sawtooth.svg", format="svg", bbox_inches="tight")

# 4. fig-dither.png: smooth blobs at 3 bits: plain rounding, random dither, error diffusion
B = blobs(256)
fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.55))
for ax, (img, lab) in zip(axs, [(B, "original"), (quantise(B, 3), "3 bits: banding"),
                                (quantise(B, 3, "random"), "3 bits + random dither"),
                                (quantise(B, 3, "diffuse"), "3 bits + error diffusion")]):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); ax.set_title(lab, fontsize=16); ax.axis("off")
fig.tight_layout(pad=0.3); save(fig, "fig-dither.png", dpi=110)

# check numbers quoted on the slides: RMS error on the ramp at 4 bits (demo ramp, N = 256)
N = 256; j, i = np.meshgrid(np.arange(N), np.arange(N)); rampimg = (j * N + i) / (N * N - 1)
print("ramp 4 bits RMS", np.sqrt(np.mean((quantise(rampimg, 4) - rampimg) ** 2)), "theory", 1 / 15 / math.sqrt(12))
