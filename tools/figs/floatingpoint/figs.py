# Static figures for the floating-point unit. Run from the repository root, after dump.ts:
#   TMP=$(mktemp -d)
#   node_modules/.bin/esbuild --bundle --platform=node tools/figs/floatingpoint/dump.ts --outfile=$TMP/dump.js && node $TMP/dump.js $TMP
#   python3 -I tools/figs/floatingpoint/figs.py $TMP
# The summation figure uses the floatbits demo's own pixel values (mean 1000, sd 50, seed 21);
# float16 arithmetic is numpy's IEEE half (round to nearest even), as in the demo.
# Micrograph: demos/src/fourier2d/cell256.png (Robert M. Hunt, CC BY 3.0, Wikimedia Commons).
import json, math, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from skimage import io

plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
C = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
OUT = "floatingpoint/"
D = sys.argv[1]

def save(fig, name, **kw):
    fig.savefig(OUT + name, **kw); plt.close(fig); print("wrote", OUT + name)

# 1. fig-fields.svg: sign / exponent / fraction fields of float32, float16, bfloat16, to scale
fig, ax = plt.subplots(figsize=(11, 2.5))
ax.axis("off"); ax.set_xlim(-5.6, 32.3); ax.set_ylim(-0.75, 2.75)
fmts = [("float32", 8, 23), ("float16", 5, 10), ("bfloat16", 8, 7)]
cols = {"s": "#cccccc", "e": "#9ecae1", "f": "#fdd0a2"}
for r, (name, e, m) in enumerate(fmts):
    y = 2.3 - r * 1.0
    ax.text(-0.4, y, name, ha="right", va="center", fontsize=20, fontweight="bold")
    x = 0
    for lab, n, key in (("s", 1, "s"), (f"exponent {e}" if e > 5 else f"exp. {e}", e, "e"), (f"fraction {m}", m, "f")):
        ax.add_patch(Rectangle((x, y - 0.36), n, 0.72, fc=cols[key], ec="#555", lw=1.2))
        for k in range(1, n): ax.plot([x + k, x + k], [y - 0.36, y - 0.26], color="#777", lw=0.8)
        ax.text(x + n / 2, y, lab, ha="center", va="center", fontsize=18 if n > 1 else 17)
        x += n
ax.text(0, -0.6, "one tick per bit; m = number of fraction bits", fontsize=16, color="#555")
save(fig, "fig-fields.svg", format="svg", bbox_inches="tight")

# 2. fig-toyfloats.svg: every number of a toy float with 2 fraction bits between 1/4 and 16
vals = sorted({(1 + f / 4) * 2.0 ** E for E in range(-2, 4) for f in range(4)} | {16.0})
fig, ax = plt.subplots(figsize=(9, 2.5))
for s in ("left", "top", "right"): ax.spines[s].set_visible(False)
ax.set_yticks([]); ax.set_ylim(-0.2, 1.3); ax.set_xlim(0, 16.4)
for v in vals:
    pw = abs(math.log2(v) - round(math.log2(v))) < 1e-9
    ax.plot([v, v], [0, 0.55 if pw else 0.35], color=C[1] if pw else C[0], lw=2.5 if pw else 2)
for E in range(1, 4):
    a, b = 2 ** E, 2 ** (E + 1)
    ax.annotate("", xy=(a, 0.8), xytext=(a + (b - a) / 4, 0.8), arrowprops=dict(arrowstyle="<->", color="#444", lw=1.3))
    ax.text(a + (b - a) / 8, 0.92, f"gap {(b - a) / 4:g}", ha="center", fontsize=16 if E > 1 else 15)
ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xlabel("every number of a toy float with 2 fraction bits (red: powers of two)")
fig.tight_layout(); save(fig, "fig-toyfloats.svg", format="svg")

# 3. fig-int16half.svg: integers 0..65535 stored as float16: stored minus true value
v = np.arange(0, 65536)
h = v.astype(np.float16)
fin = np.isfinite(h)
err = h[fin].astype(np.float64) - v[fin]
fig, ax = plt.subplots(figsize=(6.2, 4.4))
for k in range(11, 16):  # in [2^k, 2^(k+1)) the gap is 2^(k-10): errors fill [-gap/2, gap/2]
    a, b, g = 2 ** k, 2 ** (k + 1), 2.0 ** (k - 10)
    ax.fill_between([a, min(b, 65519)], -g / 2, g / 2, color=C[0], alpha=0.25, lw=0)
sub = slice(0, None, 61)
ax.plot(v[fin][sub], err[sub], ".", color=C[0], ms=2.5)
ax.plot([0, 2048], [0, 0], color=C[0], lw=2.5)
ax.axvline(2048, color="#888", ls="--", lw=1.2); ax.text(2900, 12, "exact up\nto 2048", fontsize=16)
ax.axvspan(65520, 66500, color=C[1], alpha=0.5); ax.text(64300, -14.5, "→ ∞", color=C[1], fontsize=20, ha="right",
                                                         bbox=dict(fc="white", ec="none", pad=1))
ax.set_xticks([0, 20000, 40000, 60000]); ax.set_xticklabels(["0", "20 000", "40 000", "60 000"])
ax.set_xlim(0, 66500); ax.set_ylim(-17.5, 17.5); ax.set_yticks([-16, -8, 0, 8, 16])
ax.set_xlabel("16-bit pixel value"); ax.set_ylabel("stored − true")
fig.tight_layout(); save(fig, "fig-int16half.svg", format="svg")
print("float16 checks:", float(np.float16(2049)), float(np.float16(3001)), float(np.float16(40001)), float(np.float16(65535)),
      "max err", np.abs(err).max(), "first inf", v[~fin].min())

# 4. fig-nanblur.png: one NaN pixel and two 3x3 box blurs
cell = io.imread("demos/src/fourier2d/cell256.png").astype(float) / 255
img = cell[100:121, 60:81].copy(); img[10, 10] = np.nan
def box(a):
    p = np.pad(a, 1, mode="edge")
    return sum(p[1 + dy:1 + dy + a.shape[0], 1 + dx:1 + dx + a.shape[1]] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9
steps = [img, box(img), box(box(img)), box(box(box(img)))]
cm = plt.get_cmap("gray").copy(); cm.set_bad(C[1])
fig, axs = plt.subplots(2, 2, figsize=(5.6, 6.0)); axs = axs.ravel()
for ax, a, lab in zip(axs, steps, ("one NaN pixel", "1 blur", "2 blurs", "3 blurs")):
    ax.imshow(np.ma.masked_invalid(a), cmap=cm, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(lab, fontsize=21); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(pad=0.3, h_pad=1.2); save(fig, "fig-nanblur.png", dpi=110, bbox_inches="tight", pad_inches=0.06)
print("NaN pixels after k blurs:", [int(np.isnan(a).sum()) for a in steps])

# 5. fig-stall.svg: running sum of normalised pixel values in float16 against the exact sum
d = json.load(open(f"{D}/runsum.json"))
px = np.array(d["first"][:12000], dtype=np.float64)
s = np.float16(0); run16 = []
for x in px:
    s = np.float16(s + np.float16(x / 65535)); run16.append(float(s))
exact = np.cumsum(px) / 65535
n = np.arange(1, len(px) + 1)
fig, ax = plt.subplots(figsize=(6.0, 4.6))
ax.plot(n, exact, color="#555", lw=2.5, label="exact")
ax.plot(n, run16, color=C[1], lw=2.5, label="float16 running sum")
for lv in (32, 64): ax.axhline(lv, color="#bbb", lw=1, ls=":")
ax.set_yticks([0, 32, 64, 100, 150]); ax.set_xlabel("pixels added"); ax.set_ylabel("sum of pixel / 65 535")
ax.legend(fontsize=15, frameon=False, loc="upper left")
fig.tight_layout(); save(fig, "fig-stall.svg", format="svg")
r = d["rows"][-1]; print("float32 naive rel. error at", r[0], (r[2] - r[1]) / r[1])
