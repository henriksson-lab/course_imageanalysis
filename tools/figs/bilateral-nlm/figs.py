"""Figures for bilateral-nlm/lec.qmd. Run from the repository root:
    python3 -I tools/figs/bilateral-nlm/figs.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "heat-equation"))
from pslib import *  # noqa: E402,F401,F403
from matplotlib.patches import Rectangle  # noqa: E402

OUT = "bilateral-nlm"
D = run_node("bilateral-nlm")
y = arr(D["noisy"]); truth = arr(D["truth"])

# ---- fig-weightmaps.png ----
fig, axs = plt.subplots(2, 4, figsize=(11, 6.0))
for ax, (k, lab) in zip(axs[0], (("flat", "flat"), ("edge", "edge"), ("corner", "corner"), ("tj", "1-px line"))):
    e = D["wb"][k]; w = arr(e["w"], e["size"])
    show(ax, w, 0, w.max(), title=f"bilateral, {lab}\nN_eff = {e['neff']:.0f}", cmap="magma")
for ax, (k, lab) in zip(axs[1], (("flat", "flat"), ("lattice", "lattice"), ("unique", "unique spot"), ("stripes", "stripes"))):
    e = D["wn"][k]; w = arr(e["w"], e["size"])
    show(ax, w, 0, w.max(), title=f"NLM, {lab}\nN_eff = {e['neff']:.0f}", cmap="magma")
fig.tight_layout()
save(fig, f"{OUT}/fig-weightmaps.png", dpi=100)
print("neff", {k: round(v["neff"], 1) for k, v in D["wb"].items()}, {k: (round(v["neff"], 1), round(v["self"], 2)) for k, v in D["wn"].items()})

# ---- fig-neff.png ----
fig, axs = plt.subplots(1, 3, figsize=(11, 4.0))
show(axs[0], y, title="noisy card (σn = 10)")
nb, nn = arr(D["neffB"]), arr(D["neffN"])
im1 = axs[1].imshow(nb, cmap="viridis", vmin=0, vmax=50); axs[1].set_xticks([]); axs[1].set_yticks([]); axs[1].set_title("bilateral N_eff (σs 2, σr 25)", fontsize=FONT - 2)
im2 = axs[2].imshow(nn, cmap="viridis", vmin=0, vmax=100); axs[2].set_xticks([]); axs[2].set_yticks([]); axs[2].set_title("NLM N_eff (5×5, 15×15, h 15)", fontsize=FONT - 2)
fig.colorbar(im1, ax=axs[1], fraction=0.046); fig.colorbar(im2, ax=axs[2], fraction=0.046)
fig.tight_layout()
save(fig, f"{OUT}/fig-neff.png", dpi=100)
print("neff ranges bil", nb.min(), nb.max(), "nlm", nn.min(), nn.max(), "nlm at unique", nn[76, 46], "lattice", nn[20, 82])

# ---- fig-sigmar.png ----
fig, axs = plt.subplots(1, 3, figsize=(11, 4.0))
for ax, e in zip(axs, D["sr"]):
    show(ax, arr(e["u"]), title=f"σr = {e['s']} ({e['s'] / 10:g} σn): {e['psnr']:.1f} dB")
fig.tight_layout()
save(fig, f"{OUT}/fig-sigmar.png", dpi=100)

# ---- fig-nlm-byhand.svg ----
x = np.array([11, 9, 30, 10, 12, 29, 9, 11.])
i = 3
pos = list(range(1, 7))
d2 = [((x[i - 1:i + 2] - x[j - 1:j + 2]) ** 2).mean() for j in pos]
w = np.exp(-np.array(d2) / 100)
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.0, 5.2), gridspec_kw={"height_ratios": [1.5, 1]}, sharex=True)
ax.bar(np.arange(8), x, color=GREY, width=0.7)
for k, v in enumerate(x):
    ax.text(k, v + 0.6, f"{v:g}", ha="center", fontsize=FONT - 2)
ax.add_patch(Rectangle((i - 1.45, -1), 2.9, 34, fill=False, ec=RED, lw=2.5))
ax.add_patch(Rectangle((6 - 1.45, -1), 2.9, 34, fill=False, ec=BLUE, lw=2.5, ls="--"))
ax.text(i, 35, "target patch", color=RED, ha="center", fontsize=FONT - 2)
ax.text(6, 35, "best match, d² = 1", color=BLUE, ha="center", fontsize=FONT - 2)
ax.set_ylim(0, 40); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax2.bar(pos, w, color=[BLUE if p in (3, 6) else ORANGE for p in pos], width=0.6)
for p, ww, dd in zip(pos, w, d2):
    ax2.text(p, ww + 0.04, f"{ww:.2f}", ha="center", fontsize=FONT - 3)
ax2.set_ylim(0, 1.25); ax2.set_ylabel("weight"); ax2.set_xticks(range(8)); ax2.set_xticklabels([str(k) for k in range(8)])
ax2.set_xlabel("pixel index (patch centres 1 … 6)")
fig.tight_layout()
save(fig, f"{OUT}/fig-nlm-byhand.svg")
print("nlm byhand d2", np.round(d2, 2), "w", np.round(w, 3), "est", (w * x[1:7]).sum() / w.sum())

# ---- fig-selfweight.png ----
S = D["self"]
fig, axs = plt.subplots(1, 3, figsize=(11, 4.0))
show(axs[0], arr(S["h5"]), title=f"h = 5: {S['psnr'][0]:.1f} dB")
show(axs[1], arr(S["h5f"]), title=f"h = 5, own weight capped: {S['psnr'][1]:.1f} dB")
show(axs[2], arr(S["h15"]), title=f"h = 15: {S['psnr'][2]:.1f} dB")
fig.tight_layout()
save(fig, f"{OUT}/fig-selfweight.png", dpi=100)
print("self psnr", S["psnr"])

# ---- fig-bilateral-iter.png ----
It = D["iter"]
fig = plt.figure(figsize=(11, 6.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.6])
for k, key in enumerate(("k1", "k3", "k10")):
    ax = fig.add_subplot(gs[0, k]); show(ax, arr(It[key]), title=f"bilateral × {key[1:]}")
ax = fig.add_subplot(gs[1, :])
xs = np.arange(128)
ax.plot(xs, truth[112], color="k", lw=1.4, ls="--", label="truth")
for key, col in (("k1", ORANGE), ("k10", BLUE)):
    ax.plot(xs, arr(It[key])[112], color=col, lw=2.2, label=f"× {key[1:]}")
ax.set_xlabel("row 112: step, ramp, line, soft step"); ax.set_ylabel("value"); ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=FONT - 4)
fig.tight_layout()
save(fig, f"{OUT}/fig-bilateral-iter.png", dpi=100)

# ---- fig-patch-graph.svg: pixels of the lattice crop as nodes, NLM edges to look-alikes ----
crop = y[0:44, 64:128]
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.imshow(crop, cmap="gray", vmin=0, vmax=255, extent=(64, 128, 44, 0))
centres = [(70 + 6 * i, 8 + 6 * j) for i in range(10) for j in range(6)]
src = (82, 20)
for c in centres:
    if c != src and abs(c[0] - src[0]) <= 18 and abs(c[1] - src[1]) <= 12:
        ax.plot([src[0], c[0]], [src[1], c[1]], color=YELLOW, lw=1.6, alpha=0.9)
ax.plot(*src, "o", color=RED, ms=11)
ax.plot(85, 23, "s", color=CYAN, ms=10)
ax.plot([85, 85, 79], [23, 29, 23], color=CYAN, lw=1.6)
ax.set_xticks([]); ax.set_yticks([])
save(fig, f"{OUT}/fig-patch-graph.svg")

# ---- fig-joint.png ----
fig = plt.figure(figsize=(11, 6.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.62])
for k, (img, lab) in enumerate(((y, "noisy channel"), (arr(D["guide"]), "guide (clean, with an extra step)"), (arr(D["joint"]), "joint bilateral"))):
    ax = fig.add_subplot(gs[0, k]); show(ax, img, title=lab)
    if k == 2:
        ax.add_patch(Rectangle((34, 98), 32, 28, fill=False, ec=RED, lw=2))
ax = fig.add_subplot(gs[1, :])
xs = np.arange(128); prof = lambda a: arr(a)[100:124].mean(axis=0)
ax.plot(xs, truth[100:124].mean(axis=0), color="k", lw=1.4, ls="--", label="truth")
ax.plot(xs, prof(D["bil4"]), color=ORANGE, lw=2.2, label="bilateral (own range)")
ax.plot(xs, prof(D["joint"]), color=BLUE, lw=2.5, label="joint (guide's range)")
ax.set_xlim(20, 80); ax.set_ylim(40, 200)
ax.set_xlabel("x (mean of rows 100–123): the ramp gets a step at x = 50"); ax.set_ylabel("value")
ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=FONT - 4)
fig.tight_layout()
save(fig, f"{OUT}/fig-joint.png", dpi=100)
jp = prof(D["joint"]); print("joint jump at 49->50:", jp[49], jp[50], "bil", prof(D["bil4"])[49], prof(D["bil4"])[50])
