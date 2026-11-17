# Figures for the unit finetune-strategies/ (finetune-strategies/lec.qmd). Run from the repo root after
# tools/figs/transfer/dump.ts (see its header):  python3 -I tools/figs/finetune-strategies/figs.py
# Data: tools/figs/transfer/data.json (the toy network of demos/src/transfercurves/toy.ts and the
# real low-rank change of demos/src/lowrank/model.ts).
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/finetune-strategies/'
D = json.load(open(R + '/tools/figs/transfer/data.json'))
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, PURPLE = "#1b2a3a", "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)

# ------------------------------------------------------------ 1. forgetting: new and old task per strategy
B = D["byLabels"]["near"]; L = D["byLabels"]["labels"]
fig, axs = plt.subplots(1, 2, figsize=(880 / 96, 320 / 96), sharey=True)
names = [("frozen", "frozen\n+ head"), ("partial", "last\nlayers"), ("full", "all\nlayers")]
for ax, n in zip(axs, [9, 300]):
    i = L.index(n); x = np.arange(3)
    new = [B[s]["last"][i] for s, _ in names]; old = [B[s]["old"][i] for s, _ in names]
    ax.bar(x - 0.2, new, 0.38, color=BLUE, label="new task")
    ax.bar(x + 0.2, old, 0.38, color=ORANGE, label="old task (old head put back)")
    for k in range(3):
        ax.text(x[k] - 0.2, new[k] + 0.02, f"{int(100 * new[k] + 0.5)}", ha="center", fontsize=12)
        ax.text(x[k] + 0.2, old[k] + 0.02, f"{int(100 * old[k] + 0.5)}", ha="center", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels([b for _, b in names], fontsize=12)
    ax.set_title(f"{n} labels", fontsize=15, loc="left"); ax.set_ylim(0, 1.12)
axs[0].set_ylabel("accuracy (%)"); axs[0].set_yticks([0, 0.25, 0.5, 0.75, 1]); axs[0].set_yticklabels(["0", "25", "50", "75", "100"])
axs[1].legend(loc="upper right", bbox_to_anchor=(1.0, 1.32), fontsize=11, frameon=False, ncol=1)
fig.tight_layout(w_pad=1.0); save(fig, "fig-forgetting.svg")

# ------------------------------------------------------------ 2. learning rates for full fine-tuning on 9 labels
Rt = D["rates"]
fig, ax = plt.subplots(figsize=(520 / 96, 330 / 96))
for k, col, lab in [("0.003", ORANGE, "one rate 0.003"), ("0.0005", PURPLE, "one rate 0.0005"), ("layerwise", GREEN, "0.0003 / 0.001 / 0.003")]:
    ax.plot(Rt[k]["step"], Rt[k]["acc"], color=col, lw=2.5, label=lab)
ax.set_ylim(0.2, 1); ax.set_xlabel("training step"); ax.set_ylabel("validation accuracy")
ax.legend(loc="lower right", fontsize=12, frameon=False, title="all layers tuned, 9 labels", title_fontsize=12)
fig.tight_layout(); save(fig, "fig-rates.svg")

# ------------------------------------------------------------ 3. singular values of the real change, accuracy against rank
lr = D["lowrank"]
fig, axs = plt.subplots(1, 2, figsize=(880 / 96, 320 / 96))
sv = np.array(lr["sv1"][:32]); k = np.arange(1, len(sv) + 1)
axs[0].bar(k[:3], sv[:3], color=RED); axs[0].bar(k[3:], sv[3:], color="#999")
axs[0].set_xlabel("k"); axs[0].set_ylabel("singular value"); axs[0].set_title("ΔW of layer 1 (32 × 100)", fontsize=15)
r = np.arange(len(lr["accByRank"]))
axs[1].plot(r, lr["accByRank"], "-o", color=BLUE, lw=2.5, ms=4, label="new task")
axs[1].plot(r, lr["oldByRank"], "--o", color=ORANGE, lw=2, ms=4, label="old task")
axs[1].axvline(3, color=INK, ls=":", lw=1.5)
axs[1].set_ylim(0, 1.05); axs[1].set_xlabel("rank r of both layers' changes"); axs[1].set_ylabel("accuracy")
axs[1].legend(loc="center right", fontsize=12, frameon=False)
axs[1].set_title("network with W₀ + (ΔW cut to rank r)", fontsize=15)
fig.tight_layout(w_pad=1.5); save(fig, "fig-spectrum.svg")

# ------------------------------------------------------------ 4. LoRA and an adapter as diagrams
fig, axs = plt.subplots(1, 2, figsize=(880 / 96, 300 / 96))
def box(ax, x, y, w, h, text, fc, ec, fs=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
def arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))
FROZ, TR = ("#dde3ea", "#7a8796"), ("#fde3c4", ORANGE)
ax = axs[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off"); ax.set_title("low-rank update (LoRA)", fontsize=15)
ax.text(0.3, 3, "x", fontsize=18, va="center")
box(ax, 1.8, 3.6, 3.6, 1.6, "W₀ (m × n)\nfrozen", *FROZ)
box(ax, 1.6, 0.6, 1.6, 1.6, "A\n(r × n)", *TR, fs=12); box(ax, 4, 0.6, 1.6, 1.6, "B\n(m × r)", *TR, fs=12)
arrow(ax, (0.8, 3.2), (2, 4.4)); arrow(ax, (0.8, 2.8), (1.6, 1.4)); arrow(ax, (3.2, 1.4), (4, 1.4))
ax.text(6.8, 3, "+", fontsize=24, ha="center", va="center"); arrow(ax, (5.2, 4.4), (6.6, 3.3)); arrow(ax, (5.6, 1.4), (6.6, 2.7))
ax.text(8.9, 3, "W₀x + BAx", fontsize=14, ha="center", va="center"); arrow(ax, (7.1, 3), (7.75, 3))
ax = axs[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off"); ax.set_title("adapter between frozen layers", fontsize=15)
box(ax, 0.2, 2.2, 2.2, 1.6, "layer l\n(frozen)", *FROZ)
box(ax, 3.3, 3.6, 3.0, 1.6, "small network\nd → 8 → d", *TR, fs=12)
ax.text(4.8, 1.7, "+ skip", fontsize=13, ha="center")
box(ax, 7.4, 2.2, 2.4, 1.6, "layer l + 1\n(frozen)", *FROZ)
arrow(ax, (2.4, 3.3), (3.3, 4.2)); arrow(ax, (6.3, 4.2), (7.4, 3.3)); arrow(ax, (2.4, 2.7), (7.4, 2.7))
fig.tight_layout(); save(fig, "fig-lora.svg")

# ------------------------------------------------------------ 5. how far each layer moved
rc = D["relChange"]
fig, ax = plt.subplots(figsize=(420 / 96, 280 / 96))
ax.bar([0, 1], rc[:2], color=[BLUE, GREEN], width=0.6)
for i in range(2): ax.text(i, rc[i] + 0.01, f"{100 * rc[i]:.0f} %", ha="center", fontsize=13)
ax.set_xticks([0, 1]); ax.set_xticklabels(["layer 1", "layer 2"]); ax.set_ylabel("‖ΔW‖ / ‖W₀‖")
ax.set_ylim(0, 0.62); ax.set_title("all layers tuned, 300 labels", fontsize=13)
fig.tight_layout(); save(fig, "fig-moved.svg")
print("done", rc)

# ------------------------------------------------------------ 6. warm-up and decay of the learning rate
t = np.arange(0, 1001)
warm = 100
lr = np.where(t < warm, 1e-3 * t / warm, 1e-3 * 0.5 * (1 + np.cos(np.pi * (t - warm) / (1000 - warm))))
fig, ax = plt.subplots(figsize=(420 / 96, 260 / 96))
ax.plot(t, lr * 1e3, color=BLUE, lw=2.5)
ax.axvspan(0, warm, color="#fde3c4", alpha=0.7, lw=0); ax.text(warm / 2, 0.2, "warm-up", rotation=90, ha="center", fontsize=12)
ax.set_xlabel("training step"); ax.set_ylabel("learning rate (× 10⁻³)"); ax.set_ylim(0, 1.1)
fig.tight_layout(); save(fig, "fig-schedule.svg")
