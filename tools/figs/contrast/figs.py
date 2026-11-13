# Static figures for the contrast unit. Run from the repository root, after data.ts:
#   TMP=$(mktemp -d)
#   node_modules/.bin/esbuild --bundle --platform=node tools/figs/contrast/data.ts --outfile=$TMP/data.js && node $TMP/data.js $TMP
#   python3 -I tools/figs/contrast/figs.py $TMP
# Uses the demos' own synthetic images (lib/contrast.ts: the 12-bit fluorescence image, the
# foggy scene, the faint-cell scene, CLAHE results) and the course micrograph
# demos/src/fourier2d/cell256.png (credit in the captions). Writes fig-* into contrast/.
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
C = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
GREY = "#9a9a9a"
OUT = "contrast/"
D = sys.argv[1]
fluo = np.fromfile(f"{D}/fluo_u16.bin", dtype="<u2").reshape(256, 256).astype(np.int64)
foggy = np.fromfile(f"{D}/foggy_u8.bin", dtype="u1").reshape(256, 256).astype(np.int64)
faint = np.fromfile(f"{D}/faint_u8.bin", dtype="u1").reshape(256, 256).astype(np.int64)
cl_i = np.fromfile(f"{D}/clahe_interp_f32.bin", dtype="<f4").reshape(256, 256)
cl_s = np.fromfile(f"{D}/clahe_seams_f32.bin", dtype="<f4").reshape(256, 256)
cl_luts = np.fromfile(f"{D}/clahe_luts_f64.bin", dtype="<f8").reshape(16, 256)
rng_ = json.load(open(f"{D}/range.json"))
cellimg = mpimg.imread("demos/src/fourier2d/cell256.png")
if cellimg.ndim == 3:
    cellimg = cellimg[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
cell = np.round(cellimg * 255).astype(np.int64) if cellimg.max() <= 1.0 else cellimg.astype(np.int64)


def save(fig, name, **kw):
    fig.savefig(OUT + name, **kw); plt.close(fig); print("wrote", OUT + name)


def window(v, lo, hi):
    return np.round(255 * np.clip((v - lo) / (hi - lo), 0, 1))


def show(ax, img8, title=None, fs=17):
    ax.imshow(img8, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    if title: ax.set_title(title, fontsize=fs)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_color("#999")


def equalise(img, L_in, L_out=256):
    h = np.bincount(img.ravel(), minlength=L_in)
    c = np.cumsum(h) / img.size
    lut = np.round((L_out - 1) * c)
    return lut[img], lut, h, c


# 1. fig-window-curve.svg: histogram of the 12-bit image (log counts) with the window as a curve
h = np.bincount(fluo.ravel(), minlength=4096)
hb = h[:4096].reshape(256, 16).sum(1)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax2 = ax.twinx()
ax2.bar(np.arange(256) * 16 + 8, np.log10(1 + hb), width=16, color=GREY, alpha=0.45)
ax2.set_yticks([]); ax2.set_ylim(0, np.log10(1 + hb.max()) * 1.1); ax2.spines["right"].set_visible(False); ax2.spines["top"].set_visible(False)
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
v = np.arange(4096)
ax.plot(v, 255 * np.clip((v - 130) / (3176 - 130), 0, 1), color=C[0], lw=3.5)
for x in (130, 3176): ax.axvline(x, color=C[0], ls=":", lw=1.5)
ax.annotate("", xy=(0, 128), xytext=(1653, 128), arrowprops=dict(arrowstyle="->", color=C[3], lw=2.2))
ax.plot([1653, 1653], [0, 128], color=C[3], lw=2.2, ls="--")
ax.text(1700, 8, "raw 1653", color=C[3], ha="left", fontsize=15)
ax.text(60, 138, "display 128", color=C[3], fontsize=15)
ax.text(190, 268, "ℓ = 130", ha="left", fontsize=15, color=C[0]); ax.text(3176, 268, "h = 3176", ha="center", fontsize=15, color=C[0])
ax.set_xlim(0, 4095); ax.set_ylim(0, 290); ax.set_yticks([0, 128, 255])
ax.set_xlabel("raw value (12 bit)"); ax.set_ylabel("display level")
ax.text(3900, 40, "histogram\n(log counts)", ha="right", color="#666", fontsize=14)
save(fig, "fig-window-curve.svg", format="svg", bbox_inches="tight")

# 2. fig-pointop.svg: a 3x3 patch -> curve -> 3x3 patch; neighbours not used
patch_in = np.array([[12, 40, 41], [15, 44, 90], [13, 38, 87]])
f_pt = lambda x: np.round(255 * (x / 255) ** 0.5)
patch_out = f_pt(patch_in).astype(int)
fig = plt.figure(figsize=(10.5, 3.6))
axl = fig.add_axes([0.0, 0.08, 0.27, 0.84]); axc = fig.add_axes([0.44, 0.2, 0.22, 0.72]); axr = fig.add_axes([0.73, 0.08, 0.27, 0.84])
for ax_, P, title in ((axl, patch_in, "input"), (axr, patch_out, "output")):
    ax_.set_xlim(0, 3); ax_.set_ylim(3, 0); ax_.axis("off"); ax_.set_title(title, fontsize=18)
    for i in range(3):
        for j in range(3):
            centre = (i, j) == (1, 1)
            g = P[i, j] / 255
            ax_.add_patch(Rectangle((j, i), 1, 1, fc=(g, g, g) if centre else "#f2f2f2", ec="#555" if centre else "#ccc", lw=3 if centre else 1))
            ax_.text(j + 0.5, i + 0.5, str(P[i, j]), ha="center", va="center", fontsize=20,
                     color=("white" if g < 0.5 else "black") if centre else "#aaa", fontweight="bold" if centre else "normal")
xx = np.linspace(0, 255, 256)
axc.plot(xx, 255 * (xx / 255) ** 0.5, color=C[0], lw=3)
axc.plot([44, 44, 0], [0, f_pt(44), f_pt(44)], color=C[3], ls="--", lw=2)
axc.set_xlim(0, 255); axc.set_ylim(0, 255); axc.set_xticks([0, 44, 255]); axc.set_yticks([0, int(f_pt(44)), 255])
axc.set_xlabel("old value v", fontsize=15); axc.set_ylabel("new value T(v)", fontsize=15); axc.tick_params(labelsize=13)
fig.text(0.315, 0.55, "→", fontsize=34, ha="center", va="center"); fig.text(0.70, 0.55, "→", fontsize=34, ha="center", va="center")
fig.text(0.135, -0.02, "grey: neighbours, not used", ha="center", fontsize=14, color="#888")
save(fig, "fig-pointop.svg", format="svg", bbox_inches="tight")

# 3. fig-curves.png: a gallery of curves and their results on the micrograph
curves = [
    ("identity", lambda x: x),
    ("negative", lambda x: 255 - x),
    ("stretch 60–180", lambda x: 255 * np.clip((x - 60) / 120, 0, 1)),
    ("clip above 150", lambda x: np.minimum(x, 150)),
    ("threshold at 110", lambda x: np.where(x >= 110, 255, 0)),
    ("S-curve", lambda x: 255 / (1 + np.exp(-(x - 120) / 18))),
]
fig, axs = plt.subplots(2, 6, figsize=(13.5, 4.9), gridspec_kw={"height_ratios": [1, 1.25]})
for k, (name, f) in enumerate(curves):
    a = axs[0, k]
    a.plot(xx, f(xx), color=C[0], lw=3); a.set_xlim(0, 255); a.set_ylim(-5, 260)
    a.set_xticks([0, 255]); a.set_yticks([0, 255] if k == 0 else []); a.tick_params(labelsize=12)
    a.set_title(name, fontsize=16); a.set_aspect("equal")
    show(axs[1, k], np.clip(np.round(f(cell.astype(float))), 0, 255))
axs[0, 0].set_ylabel("output", fontsize=13)
fig.tight_layout(pad=0.4)
save(fig, "fig-curves.png", dpi=100, bbox_inches="tight", pad_inches=0.05)

# 4. fig-density.svg: a curve reshapes the histogram (equal input bins -> unequal output bins)
fig = plt.figure(figsize=(8.0, 5.6))
axm = fig.add_axes([0.30, 0.32, 0.62, 0.62]); axb = fig.add_axes([0.30, 0.06, 0.62, 0.2]); axl2 = fig.add_axes([0.08, 0.32, 0.17, 0.62])
gam = 0.5
T = lambda x: x ** gam
x = np.linspace(0, 1, 400)
pin = np.exp(-0.5 * ((x - 0.25) / 0.1) ** 2) + 0.35 * np.exp(-0.5 * ((x - 0.7) / 0.08) ** 2)
pin /= np.trapezoid(pin, x)
axm.plot(x, T(x), color=C[0], lw=3)
edges = np.linspace(0.05, 0.85, 9)
for k in range(len(edges) - 1):
    a_, b_ = edges[k], edges[k + 1]
    col = C[k % 2 + 2]
    axm.fill_between([a_, b_], [0, 0], [T(a_), T(b_)], color=col, alpha=0.12, lw=0)
    axm.fill_betweenx([T(a_), T(b_)], [0, 0], [a_, b_], color=col, alpha=0.18, lw=0)
    axb.fill_between(x[(x >= a_) & (x <= b_)], 0, pin[(x >= a_) & (x <= b_)], color=col, alpha=0.6)
s = T(x)
pout = pin / (gam * np.maximum(x, 1e-3) ** (gam - 1))  # p_S(s) = p_X(v) / T'(v)
for k in range(len(edges) - 1):
    a_, b_ = edges[k], edges[k + 1]
    m = (x >= a_) & (x <= b_)
    axl2.fill_betweenx(s[m], 0, pout[m], color=C[k % 2 + 2], alpha=0.6)
axb.plot(x, pin, color="#333", lw=1.5); axl2.plot(pout, s, color="#333", lw=1.5)
axm.set_xlim(0, 1); axm.set_ylim(0, 1); axm.set_xticks([]); axm.set_yticks([])
axm.text(0.55, 0.45, "$s = T(v) = \\sqrt{v}$", fontsize=18, color=C[0])
axm.text(0.03, 0.93, "steep: strips widen,\ndensity thins", fontsize=14, va="top")
axm.text(0.62, 0.15, "flat: strips narrow,\ndensity piles up", fontsize=14)
axb.set_xlim(0, 1); axb.set_yticks([]); axb.set_xlabel("input v (density $p_X$)", fontsize=15); axb.set_xticks([0, 1])
axl2.set_ylim(0, 1); axl2.invert_xaxis(); axl2.set_xticks([]); axl2.set_ylabel("output s (density $p_S$)", fontsize=15); axl2.set_yticks([0, 1])
save(fig, "fig-density.svg", format="svg", bbox_inches="tight")

# 5. fig-gamma.png: gamma curves and the 12-bit image through them
fig = plt.figure(figsize=(13, 4.0))
axc = fig.add_axes([0.035, 0.14, 0.22, 0.78])
for g, col in ((0.4, C[0]), (1, "#555"), (2.5, C[1])):
    axc.plot(xx / 255, (xx / 255) ** g, color=col, lw=3, label=f"γ = {g}")
axc.set_xlim(0, 1); axc.set_ylim(0, 1); axc.set_xticks([0, 1]); axc.set_yticks([0, 1]); axc.set_aspect("equal")
axc.set_xlabel("v (raw, scaled)", fontsize=14); axc.set_ylabel("display", fontsize=14); axc.legend(fontsize=13, frameon=False, loc="upper left")
t = np.clip((fluo - 100) / (4095 - 100), 0, 1)
for k, (g, col) in enumerate(((0.4, C[0]), (1, "#555"), (2.5, C[1]))):
    a = fig.add_axes([0.30 + k * 0.235, 0.02, 0.22, 0.88])
    show(a, np.round(255 * t ** g), f"γ = {g}")
    a.title.set_color(col)
save(fig, "fig-gamma.png", dpi=100, bbox_inches="tight", pad_inches=0.05)

# 6. fig-log.png: log curve 12 -> 8 bit with the decades, and a spectrum shown linearly and as log
fig = plt.figure(figsize=(12.5, 4.4))
axc = fig.add_axes([0.06, 0.16, 0.38, 0.78])
c_ = 255 / np.log(4096)
vv = np.arange(0, 4096)
axc.plot(vv, c_ * np.log1p(vv), color=C[0], lw=3, label="log")
for d_, col in ((10, C[2]), (100, C[3]), (1000, C[1])):
    yv = c_ * np.log1p(d_)
    axc.plot([d_, d_], [0, yv], color=col, lw=1.6, ls=":"); axc.plot(d_, yv, "o", color=col, ms=8)
    axc.text(d_ * 1.15, yv - 18, f"{d_} → {yv:.0f}", color=col, fontsize=15)
axc.set_xscale("symlog", linthresh=1); axc.set_xlim(0, 4095); axc.set_ylim(0, 265)
axc.set_xticks([0, 1, 10, 100, 1000, 4095]); axc.set_xticklabels(["0", "1", "10", "100", "1000", "4095"])
axc.set_yticks([0, 74, 141, 212, 255]); axc.set_xlabel("raw value (log axis)"); axc.set_ylabel("display level")
axc.text(2, 245, "every decade: about 71 levels", fontsize=15, color=C[0])
F = np.fft.fftshift(np.abs(np.fft.fft2(cell - cell.mean())))
a1 = fig.add_axes([0.50, 0.04, 0.23, 0.82]); a2 = fig.add_axes([0.76, 0.04, 0.23, 0.82])
show(a1, np.round(255 * F / F.max()), "|X| linear")
lf = np.log1p(F); show(a2, np.round(255 * lf / lf.max()), "log(1 + |X|)")
save(fig, "fig-log.png", dpi=100, bbox_inches="tight", pad_inches=0.05)

# 7. fig-comb.svg: stretching 8-bit data x4 leaves a comb; the same stretch on 12-bit data does not
fl8 = np.round(fluo / 16).astype(np.int64)
o8 = window(fl8, 6, 70).astype(int)
o12 = window(fluo, 96, 1120).astype(int)
fig, axs = plt.subplots(2, 2, figsize=(11, 5.4))
def hist_plot(a, data, n, rngx, col, title, xl):
    hh = np.bincount(data.ravel(), minlength=n)[: n]
    a.stairs(np.sqrt(hh), np.arange(n + 1) - 0.5, fill=True, color=col, lw=0)
    a.set_xlim(*rngx); a.set_yticks([]); a.set_title(title, fontsize=16); a.set_xlabel(xl, fontsize=14)
hist_plot(axs[0, 0], fl8, 256, (0, 80), GREY, "8-bit data, values 6 … 70", "raw value")
hist_plot(axs[0, 1], o8, 256, (0, 255), C[1], "× 4 stretch: only every 4th level (64 used)", "display level")
hist_plot(axs[1, 0], fluo, 4096, (0, 1280), GREY, "12-bit data, values 96 … 1120", "raw value")
hist_plot(axs[1, 1], o12, 256, (0, 255), C[0], f"same stretch: {len(np.unique(o12[(fluo >= 96) & (fluo <= 1120)]))} levels used", "display level")
for a in axs.ravel(): a.set_ylabel("√count", fontsize=13)
fig.tight_layout(pad=0.6)
save(fig, "fig-comb.svg", format="svg", bbox_inches="tight")

# 8. fig-idea.svg: the histogram and a curve whose slope follows it (foggy scene)
_, lutc, hc, cc = equalise(cell, 256)
_, _, hf, cf = equalise(foggy, 256)
fig, ax = plt.subplots(figsize=(7.0, 4.6))
ax2 = ax.twinx()
ax2.bar(np.arange(256), hf, width=1.0, color=GREY, alpha=0.5, lw=0); ax2.set_yticks([]); ax2.set_ylim(0, hf.max() * 1.25)
ax2.spines["right"].set_visible(False); ax2.spines["top"].set_visible(False)
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
ax.plot(np.arange(256), 255 * cf, color=C[0], lw=3.5)
ax.plot([0, 255], [0, 255], color="#999", lw=1.5, ls="--")
pk = int(np.argmax(hf))
ax.annotate("steep where\nmany pixels", xy=(pk, 255 * cf[pk]), xytext=(pk + 45, 255 * cf[pk] - 60), fontsize=15, arrowprops=dict(arrowstyle="->", lw=1.8))
ax.annotate("flat where\nfew pixels", xy=(215, 255 * cf[215]), xytext=(185, 170), fontsize=15, arrowprops=dict(arrowstyle="->", lw=1.8))
ax.annotate("", xy=(40, 255 * cf[40] + 2), xytext=(30, 60), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.text(8, 64, "flat", fontsize=15)
ax.set_xlim(0, 255); ax.set_ylim(0, 265); ax.set_xticks([0, 64, 128, 192, 255]); ax.set_yticks([0, 128, 255])
ax.set_xlabel("input value (foggy scene)"); ax.set_ylabel("output level")
save(fig, "fig-idea.svg", format="svg", bbox_inches="tight")

# 9. fig-cdf-map.svg: the cumulative histogram is the mapping; two arrows
fig, ax = plt.subplots(figsize=(6.6, 4.8))
ax.step(np.arange(256), cc, where="post", color=C[1], lw=3)
for v0, col in ((80, C[2]), (150, C[3])):
    ax.annotate("", xy=(0, cc[v0]), xytext=(v0, cc[v0]), arrowprops=dict(arrowstyle="->", lw=2.2, color=col))
    ax.plot([v0, v0], [0, cc[v0]], color=col, lw=2.2, ls="--")
    ax.text(4, cc[v0] + 0.025, f"P = {cc[v0]:.2f} → level {255 * cc[v0]:.0f}", color=col, fontsize=15)
    ax.text(v0, -0.09, str(v0), color=col, ha="center", fontsize=14)
ax.set_xlim(0, 255); ax.set_ylim(0, 1.02); ax.set_xticks([0, 255]); ax.set_yticks([0, 0.5, 1])
ax.set_xlabel("input value v"); ax.set_ylabel("P(v): fraction ≤ v")
ax2 = ax.twinx(); ax2.set_ylim(0, 1.02 * 255); ax2.set_yticks([0, 128, 255]); ax2.set_ylabel("output level = 255 · P(v)", fontsize=15)
save(fig, "fig-cdf-map.svg", format="svg", bbox_inches="tight")

# 10. fig-flat.svg: the worked example's output histogram, its cumulative, and the micrograph
hw = np.array([6, 22, 18, 8, 5, 3, 1, 1]); mw = np.round(7 * np.cumsum(hw) / 64).astype(int)
ow = np.bincount(mw, weights=hw, minlength=8).astype(int)
eq_cell, _, _, _ = equalise(cell, 256)
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0))
a = axs[0]; a.bar(np.arange(8), ow, color=C[2], width=0.8); a.axhline(8, color="#333", ls="--", lw=1.8)
for k, c in enumerate(ow): a.text(k, c + 0.5, str(c), ha="center", fontsize=15)
a.text(7.45, 8.6, "flat: 8", ha="right", fontsize=14); a.set_xticks(range(8)); a.set_title("worked example: output", fontsize=16); a.set_xlabel("output level"); a.set_ylim(0, 25)
a = axs[1]; a.step(np.arange(8), np.cumsum(ow) / 64, where="post", color=C[1], lw=3, label="output")
a.plot([-0.5, 7], [0, 1], color="#888", ls="--", lw=1.8, label="ideal line"); a.set_xticks(range(8)); a.set_ylim(0, 1.05)
a.set_title("its cumulative histogram", fontsize=16); a.set_xlabel("output level"); a.legend(fontsize=13, frameon=False, loc="upper left")
a = axs[2]; he = np.bincount(eq_cell.astype(int).ravel(), minlength=256)
a.bar(np.arange(256), he, width=1.0, color=C[0], lw=0); a.axhline(cell.size / 256, color="#333", ls="--", lw=1.5)
a.set_title(f"micrograph equalised: {np.count_nonzero(he)} of 256 levels", fontsize=16); a.set_xlabel("output level"); a.set_yticks([]); a.set_xlim(0, 255)
fig.tight_layout(pad=0.5)
save(fig, "fig-flat.svg", format="svg", bbox_inches="tight")

# 11. fig-global-fail.png: global equalisation of the 12-bit image
eq_f, lut_f, h_f, c_f = equalise(fluo, 4096)
fig = plt.figure(figsize=(13.2, 4.3))
a1 = fig.add_axes([0.0, 0.03, 0.26, 0.85]); a2 = fig.add_axes([0.27, 0.03, 0.26, 0.85]); a3 = fig.add_axes([0.62, 0.17, 0.36, 0.72])
show(a1, window(fluo, 130, 3176), "window 130–3176"); show(a2, eq_f, "equalised")
for a_ in (a1, a2): a_.add_patch(Rectangle((8, 216), 32, 32, fill=False, ec=C[2], lw=2))
a3b = a3.twinx(); hb = h_f.reshape(256, 16).sum(1)
a3b.bar(np.arange(256) * 16 + 8, hb, width=16, color=GREY, alpha=0.5, lw=0); a3b.set_yticks([]); a3b.set_ylim(0, hb.max() * 1.15)
a3b.spines["right"].set_visible(False); a3b.spines["top"].set_visible(False)
a3.set_zorder(a3b.get_zorder() + 1); a3.patch.set_visible(False)
a3.plot(np.arange(4096), lut_f, color=C[0], lw=3)
a3.set_xlim(0, 4095); a3.set_ylim(0, 260); a3.set_xlabel("raw value"); a3.set_ylabel("output level"); a3.set_xticks([0, 1000, 2000, 3000, 4000])
bgsd_raw = window(fluo, 130, 3176)[216:248, 8:40].std(); bgsd_eq = eq_f[216:248, 8:40].std()
a3.text(600, 70, "background peak:\nhalf of the levels", fontsize=14)
a3.set_title(f"noise SD in the green box: {bgsd_raw:.1f} → {bgsd_eq:.1f}", fontsize=15)
save(fig, "fig-global-fail.png", dpi=100, bbox_inches="tight", pad_inches=0.05)
print("global-fail SD", bgsd_raw, bgsd_eq, "levels below 300 ->", lut_f[300])

# 12. fig-tiles.png: tiles, the four nearest centres with weights, seams vs interpolation
lo, hi = rng_["lo"], rng_["hi"]
fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.6))
show(axs[0], window(fluo, lo, hi), "4 × 4 tiles")
for k in range(1, 4):
    axs[0].axvline(k * 64 - 0.5, color=C[0], lw=1.5); axs[0].axhline(k * 64 - 0.5, color=C[0], lw=1.5)
px, py = 105, 140
gx, gy = (px + 0.5) / 64 - 0.5, (py + 0.5) / 64 - 0.5
x0, y0 = int(np.floor(gx)), int(np.floor(gy)); fx, fy = gx - x0, gy - y0
ws = [((x0, y0), (1 - fx) * (1 - fy)), ((x0 + 1, y0), fx * (1 - fy)), ((x0, y0 + 1), (1 - fx) * fy), ((x0 + 1, y0 + 1), fx * fy)]
cols4 = [C[0], C[2], C[4], C[5]]
for ((tx, ty), w), col in zip(ws, cols4):
    cx, cy = (tx + 0.5) * 64 - 0.5, (ty + 0.5) * 64 - 0.5
    axs[0].plot([px, cx], [py, cy], color=col, lw=2)
    axs[0].plot(cx, cy, "o", color=col, ms=10, mec="white")
    axs[0].text(cx + 5, cy - 6, f"{w:.2f}", color="white", fontsize=15, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc=col, ec="none", alpha=0.9))
axs[0].plot(px, py, "o", color=C[3], ms=11, mec="white", mew=2)
show(axs[1], np.round(cl_s), "each tile its own curve: seams")
show(axs[2], np.round(cl_i), "blend of four curves (CLAHE, β = 3)")
fig.tight_layout(pad=0.4)
save(fig, "fig-tiles.png", dpi=100, bbox_inches="tight", pad_inches=0.05)
print("tile weights", [round(w, 3) for _, w in ws])

# 13. fig-clip.svg: a flat tile's histogram, the clip line, the excess spread; the two curves
tile = faint[192:256, 0:64]   # background only: one narrow peak
lo_t, hi_t = faint.min(), faint.max() + 1e-6
B = 64
bins_ = np.clip(((tile - lo_t) / (hi_t - lo_t) * B).astype(int), 0, B - 1)
ht = np.bincount(bins_.ravel(), minlength=B).astype(float)
Nt = ht.sum(); beta = 3; limit = beta * Nt / B
hcl = ht.copy()
for _ in range(200):
    ex = np.maximum(hcl - limit, 0).sum()
    if ex < 1e-9: break
    hcl = np.minimum(hcl, limit) + ex / B
ex1 = np.maximum(ht - limit, 0).sum()
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.3))
a = axs[0]
a.bar(np.arange(B), ht, width=1.0, color=GREY, alpha=0.55, lw=0, label="tile histogram")
a.bar(np.arange(B), hcl, width=1.0, color=C[0], alpha=0.75, lw=0, label="clipped + excess spread")
a.axhline(limit, color=C[1], ls="--", lw=2.2); a.text(B - 1, limit * 1.25, f"clip limit = β · N/B (β = {beta})", color=C[1], ha="right", fontsize=15)
a.set_yscale("log"); a.set_ylim(1, ht.max() * 1.6); a.set_xlabel(f"bin (B = {B})"); a.set_ylabel("count (log)")
a.legend(fontsize=13, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0))
a.set_ylim(1, ht.max() * 12)
a = axs[1]
a.plot(np.arange(B), 255 * np.cumsum(ht) / Nt, color=GREY, lw=3, label="no clip: a near-step")
a.plot(np.arange(B), 255 * np.cumsum(hcl) / Nt, color=C[0], lw=3, label=f"clipped: slope ≤ {beta}×")
a.plot([0, B - 1], [0, 255], color="#999", ls=":", lw=1.8, label="straight line (slope 1×)")
a.set_xlim(0, B - 1); a.set_ylim(0, 262); a.set_xlabel("bin"); a.set_ylabel("output"); a.legend(fontsize=13, frameon=False, loc="lower right")
fig.tight_layout(pad=0.6)
save(fig, "fig-clip.svg", format="svg", bbox_inches="tight")
print("clip: tile N", Nt, "max gain raw", ht.max() * B / Nt, "excess frac", ex1 / Nt)

# 14. fig-match.svg: matching with two cumulative histograms
hs = np.bincount(foggy.ravel(), minlength=256); Fs = np.cumsum(hs) / hs.sum()
ht2 = np.bincount(cell.ravel(), minlength=256); Gt = np.cumsum(ht2) / ht2.sum()
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
axs[0].step(np.arange(256), Fs, where="post", color=C[0], lw=3); axs[0].set_title("source (foggy scene): F", fontsize=16)
axs[1].step(np.arange(256), Gt, where="post", color=C[2], lw=3); axs[1].set_title("target (micrograph): G", fontsize=16)
v0 = 130; u = Fs[v0]; t0 = int(np.argmax(Gt >= u))
axs[0].plot([v0, v0], [0, u], color=C[3], ls="--", lw=2.2); axs[0].plot([v0, 255], [u, u], color=C[3], lw=2.2)
axs[1].annotate("", xy=(t0, u), xytext=(0, u), arrowprops=dict(arrowstyle="->", color=C[3], lw=2.2))
axs[1].plot([t0, t0], [u, 0], color=C[3], ls="--", lw=2.2)
axs[0].text(v0 + 3, 0.04, f"v = {v0}", color=C[3], fontsize=15)
axs[0].text(v0 + 8, u + 0.04, f"F(v) = {u:.2f}", color=C[3], fontsize=15)
axs[1].text(t0 + 4, 0.04, f"T(v) = {t0}", color=C[3], fontsize=15)
for a in axs: a.set_xlim(0, 255); a.set_xticks([0, 128, 255]); a.set_xlabel("value")
axs[0].set_ylabel("fraction ≤ value"); axs[0].set_ylim(0, 1.03)
fig.tight_layout(pad=0.6)
save(fig, "fig-match.svg", format="svg", bbox_inches="tight")
print("match: v", v0, "u", u, "->", t0)
