"""Slide figures of the attention/ lecture (modern deep learning day).

Run from the repository root:
  tools/figs/instances/demo_shot.sh SHOTS selfattn content sa_content
  tools/figs/instances/demo_shot.sh SHOTS embedprobe prompt ep_prompt
  node_modules/.bin/esbuild tools/figs/attention/calib_data.ts --bundle --platform=node --log-level=warning | node > /tmp/calib.json
  python3 -I tools/figs/attention/lec_figs.py SHOTS /tmp/calib.json

Demo panels are crops of the selfattn / embedprobe demos in their presets; the reliability
diagrams use the calibration demo's model code (calib_data.ts), so the numbers match its readouts. The diagrams are drawn here. The
micrograph is demos/src/fourier2d/cell256.png (Robert M. Hunt, CC BY 3.0, Wikimedia Commons).
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from PIL import Image
from scipy import ndimage as ndi

OUT = "attention/"
SHOTS = sys.argv[1]
BLUE, ORANGE, RED, GREEN, GREY = "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9a9a9a"
LBLUE, LORANGE, LGREY = "#d6e6f4", "#ffe3c4", "#ececec"
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})


def crop(shot, box):
    im = Image.open(f"{SHOTS}/{shot}.png").convert("RGB")
    return np.asarray(im.crop(tuple(2 * v for v in box)))


cell = np.asarray(Image.open("demos/src/fourier2d/cell256.png").convert("L"), float) / 255

# 1. receptive field of stacked 3 × 3 convolutions ---------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 4.6))
n = 9
for i in range(n + 1):
    ax.plot([0, n], [i, i], c="#ccc", lw=1); ax.plot([i, i], [0, n], c="#ccc", lw=1)
for L, col in [(3, "#cc79a7"), (2, ORANGE), (1, BLUE)]:
    r = L
    ax.add_patch(Rectangle((4 - r, 4 - r), 2 * r + 1, 2 * r + 1, fc=col, alpha=0.22, ec=col, lw=3))
    ax.text(4 - r + 0.12, 4 + r + 0.88, f"{L} layer{'s' if L > 1 else ''}: {2 * L + 1}×{2 * L + 1}",
            fontsize=15, color=col, va="top", fontweight="bold")
ax.add_patch(Rectangle((4, 4), 1, 1, fc="black"))
ax.set_xlim(-0.1, n + 0.1); ax.set_ylim(-0.1, n + 0.1); ax.set_aspect("equal"); ax.axis("off")
fig.savefig(OUT + "fig-receptive.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)

# 2. patches as tokens -------------------------------------------------------------------------
P = 32
fig = plt.figure(figsize=(8.0, 4.3))
ax = fig.add_axes([0.0, 0.0, 0.5, 0.92])
ax.imshow(cell, cmap="gray", vmin=0, vmax=1)
for k in range(0, 257, P):
    ax.axhline(k - 0.5, c="white", lw=1.2); ax.axvline(k - 0.5, c="white", lw=1.2)
picks = [(0, 0), (0, 1), (3, 4), (7, 7)]
for (r_, c_) in picks:
    ax.add_patch(Rectangle((c_ * P - 0.5, r_ * P - 0.5), P, P, fill=False, ec=ORANGE, lw=3))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("8 × 8 = 64 patches", fontsize=18)
for j, (r_, c_) in enumerate(picks):
    y0 = 0.72 - j * 0.235
    a = fig.add_axes([0.54, y0, 0.11, 0.2])
    a.imshow(cell[r_ * P:(r_ + 1) * P, c_ * P:(c_ + 1) * P], cmap="gray", vmin=0, vmax=1)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(ORANGE); sp.set_linewidth(2)
    v = fig.add_axes([0.69, y0 + 0.06, 0.2, 0.08])
    emb = np.tanh(np.random.default_rng(10 + j).standard_normal(16) + 2 * (cell[r_ * P:(r_ + 1) * P, c_ * P:(c_ + 1) * P].mean() - 0.5))
    v.imshow(emb[None, :], cmap="PuOr", vmin=-1, vmax=1, aspect="auto")
    v.set_xticks([]); v.set_yticks([])
    fig.text(0.665, y0 + 0.1, "→", fontsize=18, ha="center", va="center")
    fig.text(0.90, y0 + 0.1, f"+({r_},{c_})", fontsize=17, va="center")
fig.text(0.79, 0.99, "tokens + position", fontsize=18, ha="center", va="top")
fig.savefig(OUT + "fig-patches.png", dpi=100, bbox_inches="tight", facecolor="white")
plt.close(fig)

# 3. attention by hand: three tokens ------------------------------------------------------------
q = np.array([1, 0]); K = np.array([[1, 0], [0, 1], [1, 1]]); V = np.array([10, 0, 4])
s = K @ q
fig, axs = plt.subplots(1, 2, figsize=(6.0, 3.4), sharey=True)
for a, T, col in [(axs[0], 1.0, BLUE), (axs[1], 0.5, ORANGE)]:
    w = np.exp(s / T); w /= w.sum()
    out = w @ V
    bars = a.bar([0, 1, 2], w, color=col, width=0.6)
    for x_, w_ in enumerate(w):
        a.text(x_, w_ + 0.02, f"{w_:.2f}", ha="center", fontsize=15)
    a.set_xticks([0, 1, 2]); a.set_xticklabels([f"({k[0]},{k[1]})\n{v}" for k, v in zip(K, V)], fontsize=15)
    a.set_xlabel("key, value", fontsize=15)
    a.set_ylim(0, 0.62)
    a.set_title(f"T = {T:g}: output {out:.1f}", fontsize=16)
    print("T", T, "weights", np.round(w, 2), "out", round(out, 2))
axs[0].set_ylabel("weight")
fig.savefig(OUT + "fig-byhand.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)

# 4. the selfattn demo, content preset ---------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.6), gridspec_kw={"wspace": 0.05})
for a, box, t_ in [(axs[0], (10, 50, 350, 390), "weights of the query"),
                   (axs[1], (364, 50, 704, 390), "after one layer")]:
    a.imshow(crop("sa_content", box)); a.axis("off"); a.set_title(t_, fontsize=17)
fig.savefig(OUT + "fig-attnstill.png", dpi=100, bbox_inches="tight", facecolor="white")
plt.close(fig)


# block diagrams --------------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc, ec="#555", fs=16, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.8, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q, col="#333", ls="-", cs="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18, lw=1.8, color=col, ls=ls, connectionstyle=cs))


# 5. transformer block --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.8, 5.6))
ax.set_xlim(0, 5.4); ax.set_ylim(0, 9.8); ax.axis("off")
box(ax, 1.0, 0.2, 3.2, 0.9, "N tokens", LGREY)
box(ax, 1.0, 1.9, 3.2, 0.7, "normalise", "white")
box(ax, 1.0, 3.0, 3.2, 1.0, "attention\n(several heads)", LBLUE, ec=BLUE)
ax.text(2.6, 4.6, "+", ha="center", va="center", fontsize=24)
box(ax, 1.0, 5.3, 3.2, 0.7, "normalise", "white")
box(ax, 1.0, 6.4, 3.2, 1.0, "small network\nper token", LORANGE, ec=ORANGE)
ax.text(2.6, 8.0, "+", ha="center", va="center", fontsize=24)
box(ax, 1.0, 8.7, 3.2, 0.9, "N tokens", LGREY)
for y0, y1 in [(1.1, 1.9), (2.6, 3.0), (4.0, 4.35), (4.85, 5.3), (6.0, 6.4), (7.4, 7.75), (8.25, 8.7)]:
    arrow(ax, (2.6, y0), (2.6, y1))
for yb, yp in [(1.5, 4.6), (4.95, 8.0)]:
    ax.plot([2.6, 4.75, 4.75], [yb, yb, yp], c=GREEN, lw=1.8)
    arrow(ax, (4.75, yp), (2.85, yp), col=GREEN)
ax.text(4.95, 3.1, "skip", color=GREEN, fontsize=15, rotation=90, va="center")
ax.text(4.95, 6.5, "skip", color=GREEN, fontsize=15, rotation=90, va="center")
ax.text(0.45, 5.0, "× 12, 24, …", fontsize=16, rotation=90, va="center", ha="center")
fig.savefig(OUT + "fig-block.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)

# 6. transfer learning: frozen features + probe, or fine-tuning ---------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.set_xlim(0, 13.2); ax.set_ylim(0, 8.4); ax.axis("off")
crop_ = cell[60:160, 70:170]
emb_v = np.random.default_rng(3).standard_normal(8)
for y, lab, net_fc, net_ec, net_ls, head, note, note_col in [
        (4.6, "frozen features", LGREY, "#777", "--", "linear\nprobe", "frozen", "#555"),
        (0.7, "fine-tuning", LORANGE, ORANGE, "-", "new\nhead", "trained", "#b85a00")]:
    ax.text(0.0, y + 2.75, lab, fontsize=18, fontweight="bold", va="bottom")
    ax.imshow(crop_, cmap="gray", extent=(0, 1.8, y + 0.1, y + 1.9))
    arrow(ax, (1.9, y + 1.0), (2.45, y + 1.0))
    for k, hgt in enumerate([1.9, 1.5, 1.15, 0.85]):
        ax.add_patch(Rectangle((2.55 + 0.8 * k, y + 1.0 - hgt / 2), 0.55, hgt, fc=net_fc, ec=net_ec, lw=1.8, ls=net_ls))
    ax.text(3.95, y + 2.0, "pretrained network", ha="center", va="bottom", fontsize=15)
    ax.text(2.5, y - 0.05, note, ha="left", va="top", fontsize=15, color=note_col)
    arrow(ax, (5.55, y + 1.0), (6.15, y + 1.0))
    for k, val in enumerate(emb_v):
        ax.add_patch(Rectangle((6.25, y + 0.2 + 0.2 * k), 0.4, 0.2, fc=plt.get_cmap("PuOr")(0.5 + 0.35 * np.tanh(val)), ec="#555", lw=0.6))
    ax.text(6.45, y - 0.05, "embedding", ha="center", va="top", fontsize=15)
    arrow(ax, (6.75, y + 1.0), (7.35, y + 1.0))
    box(ax, 7.45, y + 0.4, 1.6, 1.2, head, LBLUE, ec=BLUE, fs=15)
    arrow(ax, (9.15, y + 1.0), (9.7, y + 1.0))
    ax.text(9.8, y + 1.0, "nucleus /\nnot nucleus", va="center", fontsize=15)
fig.savefig(OUT + "fig-transfer.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)

# 7. self-supervised pretraining: two views; masked patches ---------------------------------------
rng = np.random.default_rng(7)
c0 = cell[40:168, 60:188]
v1 = ndi.gaussian_filter(np.rot90(c0), 1.6)
v2 = np.clip(0.75 * c0 + 0.3 + 0.08 * rng.standard_normal(c0.shape), 0, 1)
other = cell[128:256, 0:128]
v3 = ndi.gaussian_filter(np.fliplr(other), 1.0)
masked = c0.copy()
G = 8
hide = rng.permutation(G * G)[: int(0.75 * G * G)]
for h in hide:
    r_, c_ = divmod(h, G)
    masked[r_ * 16:(r_ + 1) * 16, c_ * 16:(c_ + 1) * 16] = np.nan
fig, axs = plt.subplots(1, 4, figsize=(11, 3.4), gridspec_kw={"wspace": 0.08})
cm = plt.get_cmap("gray").copy(); cm.set_bad("#f2b38a")
for a, im_, t_ in [(axs[0], v1, "view 1"), (axs[1], v2, "view 2"),
                   (axs[2], v3, "other crop"), (axs[3], masked, "75 % hidden")]:
    a.imshow(im_, cmap=cm, vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([]); a.set_title(t_, fontsize=18)
for sp in axs[0].spines.values():
    sp.set_color(GREEN); sp.set_linewidth(3)
for sp in axs[1].spines.values():
    sp.set_color(GREEN); sp.set_linewidth(3)
for sp in axs[2].spines.values():
    sp.set_color(RED); sp.set_linewidth(3)
fig.text(0.31, -0.02, "same crop: embeddings close", color=GREEN, ha="center", fontsize=16)
fig.text(0.59, -0.02, "different: far", color=RED, ha="center", fontsize=16)
fig.text(0.85, -0.02, "predict the hidden patches", ha="center", fontsize=16)
fig.savefig(OUT + "fig-selfsup.png", dpi=100, bbox_inches="tight", facecolor="white")
plt.close(fig)

# 8. promptable segmentation: click and mask (embedprobe, prompt preset) -----------------------
img = crop("ep_prompt", (10, 50, 340, 380)).astype(float) / 255
msk = crop("ep_prompt", (353, 50, 683, 380)).astype(float) / 255
yellow = (msk[..., 0] > 0.7) & (msk[..., 1] > 0.7) & (msk[..., 2] < 0.5)
edge = yellow & ~ndi.binary_erosion(yellow, iterations=3)
over = img.copy(); over[edge] = (1.0, 0.85, 0.0)
fig, axs = plt.subplots(1, 2, figsize=(7.4, 4.0), gridspec_kw={"wspace": 0.05})
axs[0].imshow(img); axs[0].set_title("a click (red)", fontsize=17)
axs[1].imshow(over); axs[1].set_title("→ a mask (yellow)", fontsize=17)
for a in axs:
    a.axis("off")
fig.savefig(OUT + "fig-prompt.png", dpi=100, bbox_inches="tight", facecolor="white")
plt.close(fig)

# 9. reliability diagrams (calibration demo's model, calib_data.ts) ----------------------------
cal = json.load(open(sys.argv[2]))
fig, axs = plt.subplots(1, 3, figsize=(6.6, 3.3), sharey=True, gridspec_kw={"wspace": 0.1})
for a, key, t_ in [(axs[0], "over", "overconfident"), (axs[1], "fixed", f"T = {cal['fixed']['T']:.2f}"),
                   (axs[2], "shift", "new domain")]:
    R = cal[key]
    e = np.array(R["edges"]); mid = (e[:-1] + e[1:]) / 2
    a.bar(mid, R["acc"], width=0.09, color=BLUE, label="accuracy")
    a.plot(R["conf"], R["conf"], "o", c=RED, ms=6, label="confidence")
    a.plot([0.5, 1], [0.5, 1], ls="--", c="#666", lw=1.5)
    a.set_xlim(0.5, 1); a.set_ylim(0.5, 1); a.set_xticks([0.6, 0.8, 1]); a.set_yticks([0.5, 0.75, 1])
    a.tick_params(labelsize=14)
    a.set_title(f"{t_}\nECE {100 * R['ece']:.1f} %", fontsize=15)
axs[1].set_xlabel("confidence", fontsize=15)
axs[0].set_ylabel("accuracy", fontsize=15)
axs[0].legend(loc="upper left", frameon=False, fontsize=13, handlelength=1.0, handletextpad=0.3, borderaxespad=0.1)
fig.savefig(OUT + "fig-reliability.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)
