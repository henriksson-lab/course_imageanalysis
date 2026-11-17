# Figures for the unit few-labels/ (few-labels/lec.qmd). Run from the repo root after dump.ts (see its header):
#   python3 -I tools/figs/few-labels/figs.py
# Data: tools/figs/few-labels/data.json (demos/src/activelearn/model.ts) and tools/figs/transfer/data.json.
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/few-labels/'
D = json.load(open(R + '/tools/figs/few-labels/data.json'))
T = json.load(open(R + '/tools/figs/transfer/data.json'))
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, PURPLE = "#1b2a3a", "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"
C0, C1 = "#2a6f97", "#e08a1e"
RULE = {"random": ("random", "#888"), "uncertain": ("most uncertain", RED), "diverse": ("most different", GREEN),
        "both": ("uncertain + different", PURPLE), "mistakes": ("the model's mistakes", BLUE)}
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)

# ------------------------------------------------------------ 1. a learning curve: error against labels, log-log
B = T["byLabels"]; L = np.array(B["labels"])
fig, axs = plt.subplots(1, 2, figsize=(880 / 96, 320 / 96))
for s, col, lab in [("scratch", RED, "from scratch"), ("full", ORANGE, "all layers tuned"), ("frozen", BLUE, "frozen + new head")]:
    acc = np.array(B["near"][s]["last"])
    axs[0].plot(L, acc, "-o", color=col, lw=2.5, ms=5, label=lab)
    axs[1].plot(L, 1 - acc, "-o", color=col, lw=2.5, ms=5, label=lab)
for ax in axs:
    ax.set_xscale("log"); ax.set_xticks(L); ax.set_xticklabels([str(v) for v in L], fontsize=11); ax.minorticks_off()
    ax.set_xlabel("labelled images"); ax.grid(alpha=0.3)
axs[0].set_ylabel("validation accuracy"); axs[0].legend(loc="lower right", fontsize=11, frameon=False)
axs[1].set_yscale("log"); axs[1].set_ylabel("error rate (log)")
# straight-line fit for "from scratch"
e = 1 - np.array(B["near"]["scratch"]["last"]); b, a = np.polyfit(np.log(L), np.log(e), 1)
axs[1].plot(L, np.exp(a) * L ** b, ":", color=INK, lw=1.5)
axs[1].text(300, 0.33, f"dotted: {np.exp(a):.2f} · N^({b:.2f})", fontsize=12, ha="right")
axs[1].set_yticks([0.02, 0.05, 0.1, 0.2, 0.4]); axs[1].set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.4"]); axs[1].set_ylim(0.01, 0.5)
print("power law", np.exp(a), b)
fig.tight_layout(w_pad=1.5); save(fig, "fig-learningcurve.svg")

# ------------------------------------------------------------ 2. pool snapshots: uncertain 40 vs both 40
P = np.array(D["pool"])
def pool_ax(ax, key, title):
    S = D["snap"][key]; g = np.array(S["grid"]).reshape(60, 60)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("ab", ["#a9c4d8", "#ffffff", "#f3cf9f"])
    ax.imshow(g, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
    ax.contour(np.linspace(0, 1, 60), np.linspace(0, 1, 60), g, levels=[0.5], colors=INK, linewidths=2)
    ax.scatter(P[:, 0], P[:, 1], s=7, c=[C1 if c else C0 for c in P[:, 2]], alpha=0.55, lw=0)
    lab = np.array(S["labelled"])
    ax.scatter(P[lab, 0], P[lab, 1], s=55, c=[C1 if P[i, 2] else C0 for i in lab], edgecolors=INK, lw=1.3, zorder=3)
    ax.add_patch(Circle((0.12, 0.88), 0.11, fill=False, ls="--", ec=INK, lw=1.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#999")
    ax.set_title(title, fontsize=14)
    sc = S["score"]
    ax.set_xlabel(f"test accuracy {100 * sc['acc']:.0f} %, rare cells {100 * sc['rare']:.0f} %", fontsize=13)
fig, axs = plt.subplots(1, 3, figsize=(980 / 96, 380 / 96))
pool_ax(axs[0], "random40", "40 labels at random")
pool_ax(axs[1], "uncertain40", "40 labels, most uncertain")
pool_ax(axs[2], "both40", "40 labels, uncertain + different")
fig.tight_layout(w_pad=1.0); save(fig, "fig-pools.png", dpi=144)

# ------------------------------------------------------------ 3. learning curves of the rules
fig, axs = plt.subplots(1, 2, figsize=(880 / 96, 330 / 96))
for rule, (lab, col) in RULE.items():
    c = D["curves"][rule]
    axs[0].plot(c["n"], c["acc"], color=col, lw=2.5, label=lab)
    axs[1].plot(c["n"], c["rare"], color=col, lw=2.5)
axs[0].axhline(D["allLabels"]["acc"], color=INK, ls=":", lw=1.5); axs[0].text(40, D["allLabels"]["acc"] + 0.008, "all 500 labelled", ha="right", fontsize=11)
axs[0].set_ylim(0.65, 1); axs[0].set_xlabel("labelled cells"); axs[0].set_ylabel("test accuracy")
axs[1].set_ylim(0, 1); axs[1].set_xlabel("labelled cells"); axs[1].set_ylabel("rare subtype correct")
axs[0].legend(loc="lower right", fontsize=11, frameon=False)
fig.tight_layout(w_pad=1.5); save(fig, "fig-rules.svg")

# ------------------------------------------------------------ 4. the human-in-the-loop cycle
fig, ax = plt.subplots(figsize=(480 / 96, 330 / 96))
ax.set_xlim(-0.6, 10.6); ax.set_ylim(0, 7); ax.axis("off"); ax.set_aspect("equal")
steps = [(5, 6.0, "train on the\nlabels so far"), (8.4, 3.5, "predict on\nnew images"), (5, 1.0, "a person corrects\nthe mistakes"), (1.6, 3.5, "add them to\nthe labels")]
for x, y, t in steps:
    ax.add_patch(FancyBboxPatch((x - 1.75, y - 0.6), 3.5, 1.2, boxstyle="round,pad=0.05", fc="#eef3f8", ec=BLUE, lw=2))
    ax.text(x, y, t, ha="center", va="center", fontsize=12)
for (x0, y0), (x1, y1) in [((6.6, 6.0), (8.4, 4.15)), ((8.4, 2.85), (6.6, 1.0)), ((3.4, 1.0), (1.6, 2.85)), ((1.6, 4.15), (3.4, 6.0))]:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.8, connectionstyle="arc3,rad=-0.3"))
save(fig, "fig-loop.svg")
print("done")

# ------------------------------------------------------------ 5. full against sparse annotation (synthetic nuclei)
from matplotlib.patches import Ellipse
rng = np.random.default_rng(4)
H = 200
yy, xx = np.mgrid[0:H, 0:H]
img = np.zeros((H, H)); nuc = []
while len(nuc) < 22:
    cx, cy = rng.uniform(15, H - 15, 2); a, b = rng.uniform(8, 13), rng.uniform(6, 10); th = rng.uniform(0, np.pi)
    if all((cx - p[0]) ** 2 + (cy - p[1]) ** 2 > (a + p[2] + 2) ** 2 for p in nuc):
        nuc.append((cx, cy, a, b, th))
for cx, cy, a, b, th in nuc:
    u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th); v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
    img += 0.8 / (1 + np.exp(-(1 - np.sqrt((u / a) ** 2 + (v / b) ** 2)) / 0.08))
img += 0.08 * rng.standard_normal((H, H)) + 0.1
strokes = []
while len(strokes) < 3:
    x0, y0 = rng.uniform(5, H - 5, 2); ang = rng.uniform(0, np.pi); x1, y1 = x0 + 45 * np.cos(ang), y0 + 45 * np.sin(ang)
    pts = [(x0 + t * (x1 - x0), y0 + t * (y1 - y0)) for t in np.linspace(0, 1, 20)]
    if 5 < x1 < H - 5 and 5 < y1 < H - 5 and all(np.hypot(px - n[0], py - n[1]) > n[2] + 6 for px, py in pts for n in nuc):
        strokes.append((x0, y0, x1, y1))
fig, axs = plt.subplots(1, 2, figsize=(820 / 96, 420 / 96))
for ax, sparse in zip(axs, [False, True]):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
    for k, (cx, cy, a, b, th) in enumerate(nuc):
        if sparse and k % 5 != 0: continue
        ax.add_patch(Ellipse((cx, cy), 2 * a, 2 * b, angle=np.degrees(th), fill=False, ec=ORANGE, lw=2))
    if sparse:
        for (x0, y0, x1, y1) in strokes:
            ax.plot([x0, x1], [y0, y1], color="#5ec8f0", lw=4, solid_capstyle="round")
    ax.set_title("every nucleus outlined (22)" if not sparse else "5 nuclei + 3 background strokes", fontsize=14)
fig.text(0.75, 0.015, "unmarked pixels: ignored by the loss", ha="center", fontsize=12)
fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, "fig-sparse.png", dpi=144)
