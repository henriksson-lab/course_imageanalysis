# Figures for the unit transfer/ (transfer/lec.qmd, exercise.qmd). Run from the repo root after dump.ts
# (see its header):  python3 -I tools/figs/transfer/figs.py
# data.json: the toy network of demos/src/transfercurves/toy.ts (pretraining, example patches, fine-tuning runs).
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/transfer/'
D = json.load(open(R + '/tools/figs/transfer/data.json'))
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, PURPLE = "#1b2a3a", "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"
COL = {"frozen": BLUE, "partial": GREEN, "full": ORANGE, "scratch": RED}
LAB = {"frozen": "frozen + new head", "partial": "last layers tuned", "full": "all layers tuned", "scratch": "from scratch"}
SRC = ["disc", "ring", "bar", "two dots"]; TGT = ["interphase", "mitotic", "apoptotic"]
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)
def patch(ax, v, lo=0, hi=1, cmap="gray"):
    ax.imshow(np.array(v).reshape(10, 10), cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_color("#999")

# ------------------------------------------------------------ 1. what the pretrained network learnt
fig = plt.figure(figsize=(1000 / 96, 400 / 96))
gs = fig.add_gridspec(4, 15, wspace=0.12, hspace=0.12, left=0.01, right=0.965, top=0.86, bottom=0.02)
F = np.array(D["filters"])
for k in range(32):
    ax = fig.add_subplot(gs[k // 8, k % 8])
    m = np.abs(F[k]).max()
    patch(ax, F[k], -m, m, "RdBu_r")
fig.text(0.27, 0.92, "layer 1: the 32 filters (red +, blue −)", ha="center", fontsize=14)
# layer 2: top patches of four units with different preferences
tops = D["top"]; chosen = []; seen = set()
for t in tops:
    c = t["cls"][0]
    if c not in seen: chosen.append(t); seen.add(c)
for t in tops:
    if len(chosen) >= 4: break
    if t not in chosen: chosen.append(t)
for r, t in enumerate(chosen[:4]):
    for j in range(5):
        ax = fig.add_subplot(gs[r, 9 + j])
        patch(ax, t["patches"][j])
    fig.text(0.985, 0.86 - (r + 0.5) * 0.84 / 4, f"unit {t['unit'] + 1}", ha="left", va="center", fontsize=12, rotation=90)
fig.text(0.76, 0.92, "layer 2: what excites a unit most", ha="center", fontsize=14)
save(fig, "fig-layers.png", dpi=144)

# ------------------------------------------------------------ 2. old task and new task
E = D["examples"]
fig = plt.figure(figsize=(900 / 96, 270 / 96))
gs = fig.add_gridspec(2, 12, wspace=0.12, hspace=0.12, left=0.01, right=0.99, top=0.74, bottom=0.02, width_ratios=[1, 1, 1, 1, 0.4, 1, 1, 1, 0.4, 1, 1, 1])
for i in range(8):
    ax = fig.add_subplot(gs[i // 4, i % 4]); patch(ax, E["source"]["X"][i])
    if i < 4: ax.set_title(SRC[E["source"]["y"][i]], fontsize=11)
for key, col0, title in [("near", 5, "new task, same microscope"), ("far", 9, "new task, other modality")]:
    for i in range(6):
        ax = fig.add_subplot(gs[i // 3, col0 + i % 3]); patch(ax, E[key]["X"][i])
        if i < 3: ax.set_title(["interph.", "mitotic", "apopt."][E[key]["y"][i]], fontsize=11)
fig.text(0.18, 0.92, "old task (2400 labels)", ha="center", fontsize=14)
fig.text(0.55, 0.92, "new task, same microscope", ha="center", fontsize=14)
fig.text(0.86, 0.92, "other modality", ha="center", fontsize=14)
save(fig, "fig-tasks.png", dpi=144)

# ------------------------------------------------------------ 3. curves with 9 labels
C = D["curves"]
fig, axs = plt.subplots(1, 2, figsize=(900 / 96, 330 / 96))
for s in ["frozen", "partial", "full", "scratch"]:
    c = C[f"near9_{s}"]
    axs[0].plot(c["step"], c["acc"], color=COL[s], lw=2.5, label=LAB[s])
    axs[1].plot(c["step"], c["train"], color=COL[s], lw=2.5, ls="--")
    axs[1].plot(c["step"], c["loss"], color=COL[s], lw=2.5)
axs[0].axhline(1 / 3, color="#999", ls=":", lw=1.5); axs[0].text(5, 0.35, "guessing", ha="left", color="#777", fontsize=12)
axs[0].set_ylim(0.2, 1.0); axs[0].set_xlabel("training step"); axs[0].set_ylabel("validation accuracy")
axs[0].legend(loc="center right", fontsize=11, frameon=False)
axs[1].set_yscale("log"); axs[1].set_ylim(1e-4, 3); axs[1].set_xlabel("training step"); axs[1].set_ylabel("loss")
axs[1].text(5, 3e-4, "solid: validation, dashed: training", fontsize=12)
fig.suptitle("9 labelled images (3 per class)", fontsize=15)
fig.tight_layout(w_pad=1.5); save(fig, "fig-curves9.svg")

# ------------------------------------------------------------ 4. accuracy against the number of labels
B = D["byLabels"]; L = B["labels"]
fig, axs = plt.subplots(1, 2, figsize=(900 / 96, 340 / 96), sharey=True)
for ax, dom, title in [(axs[0], "near", "same microscope"), (axs[1], "far", "other modality")]:
    for s in ["frozen", "partial", "full", "scratch"]:
        ax.plot(L, B[dom][s]["last"], "-o", color=COL[s], lw=2.5, ms=5, label=LAB[s])
    ax.set_xscale("log"); ax.set_xticks(L); ax.set_xticklabels([str(v) for v in L], fontsize=11)
    ax.minorticks_off(); ax.set_title(title, fontsize=15); ax.set_xlabel("labelled images of the new task")
    ax.set_ylim(0.55, 1.0); ax.grid(alpha=0.3)
axs[0].set_ylabel("validation accuracy\nafter 300 steps")
axs[1].legend(loc="lower right", fontsize=11, frameon=True)
fig.tight_layout(w_pad=1.0); save(fig, "fig-labels.svg")

# ------------------------------------------------------------ 5. the four options as a diagram
fig, axs = plt.subplots(4, 1, figsize=(620 / 96, 400 / 96))
rows = [("from scratch", [1, 1, 1], "new"), ("frozen + new head", [0, 0, 1], "pre"), ("last layers tuned", [0, 1, 1], "pre"), ("all layers tuned", [1, 1, 1], "pre")]
for ax, (name, train, init) in zip(axs, rows):
    ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0, 0.5, name, va="center", fontsize=14)
    for k, (lab, x) in enumerate([("layer 1", 3.6), ("layer 2", 5.75), ("head", 7.9)]):
        tr = train[k]
        fc = "#fde3c4" if tr else "#dde3ea"
        ec = ORANGE if tr else "#7a8796"
        if init == "new" or k == 2: hatch = None
        ax.add_patch(FancyBboxPatch((x, 0.12), 1.85, 0.76, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=2))
        txt = lab + ("\npretrained\ntuned" if tr else "\npretrained\nfrozen")
        if k < 2 and init == "new": txt = lab + "\nrandom\ntrained"
        if k == 2: txt = ("head" if init == "new" else "new head") + "\nrandom\ntrained"
        ax.text(x + 0.92, 0.5, txt, ha="center", va="center", fontsize=10)
        if k < 2: ax.annotate("", xy=(x + 2.15, 0.5), xytext=(x + 1.88, 0.5), arrowprops=dict(arrowstyle="-|>", color="#555"))
fig.tight_layout(h_pad=0.2); save(fig, "fig-options.svg")
print("done")

# ------------------------------------------------------------ 6. other modality, 300 labels: curves
fig, axs = plt.subplots(1, 2, figsize=(620 / 96, 300 / 96), sharex=True)
for s in ["frozen", "partial", "full", "scratch"]:
    c = C[f"far300_{s}"]
    axs[0].plot(c["step"], c["acc"], color=COL[s], lw=2.5, label=LAB[s])
    axs[1].plot(c["step"], c["train"], color=COL[s], lw=2.5)
axs[0].set_ylim(0.2, 1); axs[0].set_ylabel("validation accuracy"); axs[0].set_xlabel("training step")
axs[1].set_yscale("log"); axs[1].set_ylabel("training loss"); axs[1].set_xlabel("training step")
axs[0].legend(loc="lower right", fontsize=10, frameon=False)
fig.suptitle("other modality, 300 labels", fontsize=14)
fig.tight_layout(w_pad=1.0); save(fig, "fig-far.svg")
