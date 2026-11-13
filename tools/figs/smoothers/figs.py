"""Figures for smoothers/lec.qmd. Run from the repository root:
    python3 -I tools/figs/smoothers/figs.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "heat-equation"))
from pslib import *  # noqa: E402,F401,F403
from matplotlib.colors import ListedColormap  # noqa: E402

OUT = "smoothers"
D = run_node("smoothers")
truth = arr(D["truth"]); y = arr(D["noisy"])

# ---- fig-testcard.png ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.6))
show(a1, truth, title="truth (synthetic test card)")
OFF = {"square": (8, 26), "step": (4, -12), "ramp": (-6, -14), "line": (-4, -14), "soft step": (6, -14), "disc r = 2": (6, 2),
       "disc r = 3": (6, 2), "disc r = 5": (8, 4), "disc r = 8": (12, 10), "lattice": (-8, 26), "unique spot": (8, 12), "stripes": (-20, 26)}
for f in D["features"]:
    if f["name"] not in OFF:
        continue
    dx, dy = OFF[f["name"]]
    a1.annotate(f["name"], (f["x"], f["y"]), (f["x"] + dx, f["y"] + dy), color=YELLOW, fontsize=FONT - 4, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=YELLOW, lw=1))
show(a2, y, title=f"with Gaussian noise σn = 15 ({D['noisyPsnr']:.1f} dB)")
fig.tight_layout()
save(fig, f"{OUT}/fig-testcard.png", dpi=100)

R = D["res"]
# ---- fig-eight.png ----
fig, axs = plt.subplots(2, 4, figsize=(11, 6.0))
for ax, e in zip(axs.ravel(), R):
    show(ax, arr(e["u"]), title=f"{e['label']}\n{e['param']} {e['p']:g}: {e['sc']['psnr']:.1f} dB")
fig.tight_layout()
save(fig, f"{OUT}/fig-eight.png", dpi=100)

# ---- fig-methodnoise8.png ----
fig, axs = plt.subplots(2, 4, figsize=(11, 6.0))
for ax, e in zip(axs.ravel(), R):
    show(ax, y - arr(e["u"]), -50, 50, title=e["label"])
fig.tight_layout()
save(fig, f"{OUT}/fig-methodnoise8.png", dpi=100)

# ---- fig-profiles8.svg ----
fig, axs = plt.subplots(2, 4, figsize=(11, 5.0), sharex=True, sharey=True)
xs = np.arange(128)
for ax, e in zip(axs.ravel(), R):
    ax.plot(xs, truth[112], color="k", lw=1.2, ls="--")
    ax.plot(xs, arr(e["u"])[112], color=BLUE, lw=1.8)
    ax.set_title(e["label"], fontsize=FONT - 3)
    ax.set_ylim(30, 215); ax.set_xticks([0, 64, 127]); ax.tick_params(labelsize=FONT - 5)
fig.supxlabel("row 112: step, ramp, 1-px line, soft step (dashed: truth)", fontsize=FONT - 3)
fig.tight_layout()
save(fig, f"{OUT}/fig-profiles8.svg")

# ---- fig-scores.svg: a heat-map table, each column ranked ----
cols = [("PSNR", "psnr", "max", "{:.1f}"), ("SSIM", "ssim", "max", "{:.2f}"), ("edge width", "edge", "min", "{:.2f}"),
        ("line kept", "line", "one", "{:.2f}"), ("disc kept", "disc", "one", "{:.2f}"), ("staircase", "stairs", "min", "{:.2f}")]
V = np.array([[e["sc"][k] for _, k, _, _ in cols] for e in R])
score = np.zeros_like(V)
for j, (_, _, kind, _) in enumerate(cols):
    v = V[:, j]
    s = v if kind == "max" else -v if kind == "min" else -np.abs(v - 1)
    score[:, j] = (s - s.min()) / (s.max() - s.min() + 1e-12)
fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.imshow(score, cmap=ListedColormap(plt.cm.Greens(np.linspace(0.05, 0.75, 256))), aspect="auto")
for i in range(len(R)):
    for j, (_, _, _, f) in enumerate(cols):
        ax.text(j, i, f.format(V[i, j]), ha="center", va="center", fontsize=FONT - 3, fontweight="bold" if score[i, j] > 0.999 else "normal")
ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[0] for c in cols], fontsize=FONT - 3)
ax.set_yticks(range(len(R))); ax.set_yticklabels([e["label"] for e in R], fontsize=FONT - 3)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("darker = better in that column (bold: best)", fontsize=FONT - 3)
save(fig, f"{OUT}/fig-scores.svg")
for e in R:
    print(e["id"], e["p"], {k: round(v, 3) for k, v in e["sc"].items()}, e["ms"], "ms")

# ---- fig-seg-after.png: threshold 120 after smoothing, noise 40; wrong pixels on the large objects ----
S = D["seg"]
fig, axs = plt.subplots(1, 4, figsize=(11, 4.6))
y40 = arr(D["noisy40"])
items = [(y40, f"noisy (σn = 40)\n{D['segNoisy']['wrong']} wrong")] + [(arr(e["u"]), f"{e['label']}, {e['param']} {e['p']:g}\n{e['wrong']} wrong, flat SD {e['sd']:.1f}") for e in S]
for ax, (img, lab) in zip(axs, items):
    img = img[0:96, 0:64]
    rgb = np.dstack([np.clip(img, 0, 255) / 255 * 0.8] * 3)
    rgb[img > 120] = [0.91, 0.25, 0.16]
    ax.imshow(rgb); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(lab, fontsize=FONT - 3)
fig.tight_layout()
save(fig, f"{OUT}/fig-seg-after.png", dpi=100)
for e in S:
    print("seg", e["id"], e["p"], "wrong", e["wrong"], "flat sd", round(e["sd"], 1))
print("seg noisy", D["segNoisy"])
