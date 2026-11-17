# Figures for the unit domain-shift/ (domain-shift/lec.qmd). Run from the repo root after
# tools/figs/few-labels/dump.ts (see its header):  python3 -I tools/figs/domain-shift/figs.py
# Data: tools/figs/few-labels/data.json (demos/src/domainshift/model.ts, demos/src/transfercurves/toy.ts).
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/domain-shift/'
D = json.load(open(R + '/tools/figs/few-labels/data.json'))
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, PURPLE = "#1b2a3a", "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"
C0, C1 = "#2a6f97", "#e08a1e"
REM = {"none": ("as is", RED), "normalise": ("normalise per microscope", BLUE), "augment": ("trained with augmentation", GREEN), "finetune": ("fine-tuned on 10 B cells", PURPLE)}
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)

# ------------------------------------------------------------ 1. the same three cells through six microscopes
looks = D["looks"]
fig, axs = plt.subplots(3, len(looks), figsize=(900 / 96, 420 / 96))
for j, lk in enumerate(looks):
    for i in range(3):
        ax = axs[i, j]
        ax.imshow(np.array(lk["X"][i]).reshape(10, 10), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#999")
    axs[0, j].set_title(lk["name"].replace(" ", "\n", 1) if len(lk["name"]) > 12 else lk["name"], fontsize=12)
for i, n in enumerate(["interphase", "mitotic", "apoptotic"]):
    axs[i, 0].set_ylabel(n, fontsize=12)
fig.tight_layout(w_pad=0.5, h_pad=0.4); save(fig, "fig-looks.png", dpi=144)

# ------------------------------------------------------------ 2. feature space A vs B with the A line
A = np.array(D["cellsA"]); B = np.array(D["cellsB"]); w = D["wNone"]
fig, axs = plt.subplots(1, 2, figsize=(880 / 96, 360 / 96), sharey=True)
x = np.array([0, 1])
for ax, P, title in [(axs[0], A, "microscope A (training)"), (axs[1], B, "microscope B (shift 0.8)")]:
    ax.scatter(P[:, 0], P[:, 1], s=14, c=[C1 if c else C0 for c in P[:, 2]], alpha=0.75, lw=0)
    ax.plot(x, -(w[0] + w[1] * x) / w[2], color=INK, lw=2.5, ls="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("mean intensity"); ax.set_title(title, fontsize=14)
axs[0].set_ylabel("size")
axs[0].text(0.03, 0.04, "● interphase", color=C0, fontsize=12); axs[0].text(0.45, 0.04, "● mitotic", color=C1, fontsize=12)
s = D["shift"]
axs[1].text(0.97, 0.04, f"accuracy {100 * s['f50']['none'][16]:.0f} % (A: {100 * D['accA']['none']:.0f} %)", ha="right", fontsize=12)
fig.tight_layout(w_pad=1.0); save(fig, "fig-features.svg")

# ------------------------------------------------------------ 3. accuracy against shift, equal and unequal class balance
fig, axs = plt.subplots(1, 2, figsize=(900 / 96, 340 / 96), sharey=True)
for ax, key, title in [(axs[0], "f50", "half of B's cells mitotic (as in A)"), (axs[1], "f20", "a fifth of B's cells mitotic")]:
    for r, (lab, col) in REM.items():
        ax.plot(s["s"], s[key][r], color=col, lw=2.5, label=lab)
    ax.set_xlabel("shift of microscope B"); ax.set_title(title, fontsize=13); ax.set_ylim(0.55, 1); ax.grid(alpha=0.3)
axs[0].set_ylabel("accuracy on B"); axs[0].legend(loc="lower left", fontsize=11, frameon=True, framealpha=0.9)
fig.tight_layout(w_pad=1.0); save(fig, "fig-remedies.svg")

# ------------------------------------------------------------ 4. intensity histograms A vs B: detecting the shift
fig, ax = plt.subplots(figsize=(460 / 96, 300 / 96))
bins = np.linspace(0, 1, 31)
ax.hist(A[:, 0], bins=bins, color=BLUE, alpha=0.55, label="A")
ax.hist(B[:, 0], bins=bins, color=RED, alpha=0.55, label="B (shift 0.8)")
ax.set_xlabel("mean intensity of a cell"); ax.set_ylabel("cells"); ax.legend(frameon=False, fontsize=12)
ax.set_title(f"shift score {s['score50'][16]:.2f} SD (no labels needed)", fontsize=13)
fig.tight_layout(); save(fig, "fig-detect.svg")
print("done")
