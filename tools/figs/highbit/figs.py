# Static figures for the highbit unit. Run from the repository root, after dump.ts:
#   TMP=$(mktemp -d)
#   node_modules/.bin/esbuild --bundle --platform=node tools/figs/highbit/dump.ts --outfile=$TMP/dump.js && node $TMP/dump.js $TMP
#   python3 -I tools/figs/highbit/figs.py $TMP
# Uses the window16 demo's own synthetic images (fluorescence 12 bit, CT phantom in HU),
# so the figures show exactly what the demo shows. Writes fig-* into highbit/.
import math, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
C = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
OUT = "highbit/"
D = sys.argv[1]
fluo = np.fromfile(f"{D}/fluo_u16.bin", dtype="<u2").reshape(256, 256).astype(np.int64)
ct = np.fromfile(f"{D}/ct_i16.bin", dtype="<i2").reshape(256, 256).astype(np.int64)

def save(fig, name, **kw):
    fig.savefig(OUT + name, **kw); plt.close(fig); print("wrote", OUT + name)

def window(v, lo, hi):  # windowLevel() of demos/src/window16/data.ts
    return np.round(255 * np.clip((v - lo) / (hi - lo), 0, 1))

def show(ax, img8, title, fs=17):
    ax.imshow(img8, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(title, fontsize=fs); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_color("#999")

# 1. fig-black.png: the 12-bit image shown as 0..65535 and as 0..4095
fig, axs = plt.subplots(1, 2, figsize=(8.0, 4.3))
show(axs[0], window(fluo, 0, 65535), "as 0 … 65 535", 24)
show(axs[1], window(fluo, 0, 4095), "as 0 … 4095", 24)
fig.tight_layout(pad=0.3); save(fig, "fig-black.png", dpi=110, bbox_inches="tight", pad_inches=0.06)

# 2. fig-twos.svg: one row of 16-bit patterns, read as uint16 and as int16
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.axis("off"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.35, 3.0)
rows = [("bits", ["0x0000", "0x7FFF", "0x8000", "0xFFFF"], "#666"),
        ("uint16", ["0", "32 767", "32 768", "65 535"], C[0]),
        ("int16", ["0", "32 767", "−32 768", "−1"], C[3])]
for r, (lab, vals, col) in enumerate(rows):
    y = 2.6 - r * 1.05
    ax.add_patch(Rectangle((0.12, y - 0.3), 0.43, 0.6, fc="#eef4fa" if r != 2 else "#fdf0e3", ec=col, lw=2))
    ax.add_patch(Rectangle((0.57, y - 0.3), 0.43, 0.6, fc="#eef4fa" if r == 1 else ("#fbe3e3" if r == 2 else "#f4f4f4"), ec=col, lw=2))
    ax.text(0.10, y, lab, ha="right", va="center", fontsize=22, color=col, fontweight="bold")
    for x, v in zip((0.135, 0.535, 0.585, 0.985), vals):
        ax.text(x, y, v, ha="left" if x in (0.135, 0.585) else "right", va="center", fontsize=21,
                family="monospace" if r == 0 else "sans-serif")
ax.text(0.785, -0.2, "negative in int16", ha="center", fontsize=20, color=C[1])
ax.text(0.335, -0.2, "same in both", ha="center", fontsize=20, color="#444")
save(fig, "fig-twos.svg", format="svg", bbox_inches="tight")

# 3. fig-subtract.png: subtract a background of 160 in uint16 wrap, uint16 saturate, float
bg = 160
wrap = (fluo - bg) % 65536
sat = np.clip(fluo - bg, 0, 65535)
flt = (fluo - bg).astype(float)
py, px = slice(0, 20), slice(110, 150)  # a background patch, away from the cells
lo, hi = -60, 600
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, img, lab in zip(axs, (wrap, sat, flt), ("uint16, wrap-around", "uint16, saturate", "float")):
    show(ax, window(img, lo, hi)[:150], lab, 23)
    m = img[py, px].mean()
    ax.add_patch(Rectangle((109.5, -0.5), 40, 20, fill=False, ec=C[3], lw=2.5))
    ax.set_xlabel(f"patch mean {m:,.1f}".replace(",", " "), fontsize=23)
fig.tight_layout(pad=0.3); save(fig, "fig-subtract.png", dpi=110, bbox_inches="tight", pad_inches=0.06)
print("patch means: raw", fluo[py, px].mean(), "wrap", wrap[py, px].mean(), "sat", sat[py, px].mean(), "float", flt[py, px].mean())

# 4. fig-window.svg: histogram of the 12-bit image with a percentile window and the mapping
lo, hi = np.percentile(fluo, 0.5, method="nearest"), np.percentile(fluo, 99.5, method="nearest")
fig, ax = plt.subplots(figsize=(7.2, 4.8))
cnt, edges = np.histogram(fluo, bins=128, range=(0, 4096))
ax.bar(edges[:-1], cnt, width=edges[1] - edges[0], align="edge", color="#bbb", label="histogram")
ax.set_yscale("log"); ax.set_ylabel("pixels (log)"); ax.set_xlabel("raw value (12 bit)")
ax.axvspan(0, lo, color=C[0], alpha=0.12); ax.axvspan(hi, 4096, color=C[1], alpha=0.12)
ax.set_xlim(0, 4096)
a2 = ax.twinx(); a2.spines["right"].set_visible(True); a2.spines["top"].set_visible(False)
v = np.linspace(0, 4096, 800)
a2.plot(v, window(v, lo, hi), color="#222", lw=2.5)
a2.set_ylim(0, 270); a2.set_yticks([0, 128, 255]); a2.set_ylabel("display level")
a2.text(lo + 30, 262, f"ℓ = {lo:.0f}", fontsize=15, color=C[0])
a2.text(hi - 30, 262, f"h = {hi:.0f}", fontsize=15, color=C[1], ha="right")
a2.text(60, 120, "black", color=C[0], fontsize=15, rotation=90, va="center")
a2.text(hi + 140, 120, "white (clipped)", color=C[1], fontsize=15, rotation=90, va="center")
fig.tight_layout(); save(fig, "fig-window.svg", format="svg")
print("percentile window", lo, hi)

# 5. fig-hu.svg: the HU scale with tissues and the three standard windows
fig, ax = plt.subplots(figsize=(5.6, 6.2))
ax.set_ylim(-1150, 2050); ax.set_xlim(0, 4.5)
grad = np.linspace(-1150, 2050, 400)[:, None]
ax.imshow(window(grad, -160, 240), cmap="gray", vmin=0, vmax=255, extent=(0, 0.5, -1150, 2050), origin="lower", aspect="auto")
for hu, ty, lab in ((-1000, -1000, "air −1000"), (-100, -250, "fat ≈ −100"), (0, 0, "water 0"), (50, 250, "soft tissue 20…80"),
                    (700, 700, "bone"), (1800, 1800, "dense bone 2000+")):
    ax.plot([0.5, 0.6, 0.7], [hu, hu, ty], color="#333", lw=1.2)
    ax.text(0.74, ty, lab, va="center", fontsize=17)
wins = (("soft tissue", 40, 400, C[2]), ("lung", -600, 1500, C[0]), ("bone", 400, 1800, C[3]))
for k, (lab, L, W, col) in enumerate(wins):
    x = 3.0 + k * 0.5
    ax.add_patch(Rectangle((x, L - W / 2), 0.32, W, fc=col, alpha=0.75, ec="none"))
    ax.text(x + 0.16, L + W / 2 + 40, lab, rotation=90, ha="center", va="bottom", fontsize=17, color=col)
ax.set_xticks([]); ax.set_ylabel("Hounsfield units"); ax.spines["bottom"].set_visible(False)
ax.text(3.5, -1130, "windows", ha="center", fontsize=15, color="#444")
fig.tight_layout(); save(fig, "fig-hu.svg", format="svg")

# 6. fig-tone.svg: display value against scene brightness for a linear window and two tone curves
x = np.logspace(-3, 3, 400)
fig, ax = plt.subplots(figsize=(6.4, 5.4))
ax.plot(x, np.clip(x, 0, 1), color=C[0], lw=2.5, label="linear window [0, 1]")
ax.plot(x, np.clip(x / 100, 0, 1), color=C[0], lw=2.5, ls="--", label="linear window [0, 100]")
ax.plot(x, x / (1 + x), color=C[3], lw=2.5, label="x / (1 + x)")
ax.plot(x, (np.log10(x) + 3) / 6, color=C[2], lw=2.5, label="logarithmic")
ax.set_xscale("log"); ax.set_ylim(0, 1.04); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["black", "", "white"])
ax.set_xlabel("scene brightness (relative, log axis)"); ax.set_ylabel("display")
ax.legend(fontsize=16, frameon=False, loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=2, handlelength=1.5, columnspacing=0.8)
fig.tight_layout(); save(fig, "fig-tone.svg", format="svg")
