# Figures for the backprop unit (backprop/lec.qmd). Run (from the repo root) after dump.ts (see its header):
#   python3 -I tools/figs/backprop/figs.py
# data.json holds the gdescent landscapes/paths and the nnplayground "overfit", "decay" and "dropout"
# presets after 2000 steps, computed with the demos' own code. Figure sizes are the display size in px / 96.
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/backprop/'
D = json.load(open(R + '/tools/figs/backprop/data.json'))
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c"
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)

# ------------------------------------------------------------ 1. cross-entropy -ln p
fig, ax = plt.subplots(figsize=(470 / 96, 330 / 96))
p = np.linspace(0.005, 1, 400)
ax.plot(p, -np.log(p), color=INK, lw=2.5)
for q in [0.99, 0.9, 0.5, 0.2, 0.01]:
    ax.scatter([q], [-np.log(q)], s=55, color=RED if q == 0.01 else BLUE, zorder=3)
    dx, dy, ha = {0.99: (0.0, 0.35, "center"), 0.9: (-0.03, 0.35, "center")}.get(q, (0.03, 0.12, "left"))
    if q != 0.99: ax.text(q + dx, -np.log(q) + dy, f"{-np.log(q):.2f}", fontsize=14, color=RED if q == 0.01 else INK, ha=ha)
ax.text(0.2, 4.2, "sure and wrong:\nexpensive", color=RED, fontsize=14)
ax.set_xlabel("probability given to the true class"); ax.set_ylabel("loss  −ln p")
ax.set_xlim(0, 1.05); ax.set_ylim(0, 5.3)
fig.tight_layout(); save(fig, "fig-crossentropy.svg")

# ------------------------------------------------------------ helpers for contour plots
def landscape(ax, L):
    a0, a1, b0, b1 = L["range"]; n = L["n"]
    Z = np.array(L["z"]).reshape(n, n)
    A, B = np.meshgrid(np.linspace(a0, a1, n), np.linspace(b0, b1, n))
    m = L["minima"][0]
    lev = m[2] + np.array([0.02, 0.08, 0.2, 0.4, 0.8, 1.5, 2.5, 4, 6, 9])
    ax.contour(A, B, Z, levels=lev, colors="#b9c3cc", linewidths=1)
    ax.scatter([m[0]], [m[1]], marker="*", s=220, c=INK, zorder=5)
    ax.set_xlim(a0, a1); ax.set_ylim(b0, b1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("weight a"); ax.set_ylabel("weight b")
def path(ax, P, col, n=None, lw=2, ms=4.5):
    q = np.array(P["pts"][: (n or len(P["pts"]))])
    ax.plot(q[:, 0], q[:, 1], "-o", color=col, lw=lw, ms=ms, zorder=4)

# ------------------------------------------------------------ 2. step size: just right vs zig-zag in a narrow valley
fig, axs = plt.subplots(1, 2, figsize=(520 / 96, 290 / 96))
landscape(axs[0], D["bowl"]); path(axs[0], D["paths"]["right"], GREEN, 25)
axs[0].set_title("bowl, η = 0.3", fontsize=14)
landscape(axs[1], D["valley"]); path(axs[1], D["paths"]["zigzag"], RED, 25)
axs[1].set_title("narrow valley, η = 0.6", fontsize=14)
fig.tight_layout(w_pad=1.0); save(fig, "fig-gdpaths.svg")

# ------------------------------------------------------------ 3. full batch vs mini-batches of 4
fig, ax = plt.subplots(figsize=(420 / 96, 380 / 96))
landscape(ax, D["bowl"])
path(ax, D["paths"]["full01"], BLUE, 61, lw=2.2, ms=3.5)
path(ax, D["paths"]["sgd4"], ORANGE, 61, lw=1.8, ms=3.5)
ax.plot([], [], "-o", color=BLUE, label="all 40 examples"); ax.plot([], [], "-o", color=ORANGE, label="batches of 4")
ax.legend(loc="lower left", fontsize=13, frameon=True, framealpha=0.9)
ax.text(-1.5, 2.15, "start", fontsize=13, va="bottom")
fig.tight_layout(); save(fig, "fig-sgd.svg")

# ------------------------------------------------------------ 4. local derivatives of ReLU and sigmoid
s = np.linspace(-10, 10, 801)
sig = 1 / (1 + np.exp(-s))
fig, axs = plt.subplots(1, 2, figsize=(560 / 96, 300 / 96), sharey=True)
ax = axs[0]
ax.plot(s, np.maximum(0, s) / 10, color=INK, lw=2.5, label="ReLU / 10")
ax.plot(s, (s > 0).astype(float), color=RED, lw=2.5, label="slope")
ax.set_title("ReLU: passes or blocks", fontsize=14)
ax = axs[1]
ax.plot(s, sig, color=INK, lw=2.5, label="value")
ax.plot(s, sig * (1 - sig), color=RED, lw=2.5, label="slope")
d8 = 1 / (1 + np.exp(-8)) * (1 - 1 / (1 + np.exp(-8)))
assert abs(d8 - 0.000335) < 1e-5
ax.annotate(f"slope at 8:\n{d8:.4f}", xy=(8, d8), xytext=(1.5, 0.45), fontsize=13, color=RED,
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5))
ax.set_title("sigmoid: saturates", fontsize=14)
for ax in axs:
    ax.set_xlabel("input z"); ax.set_xticks([-8, 0, 8]); ax.set_ylim(-0.05, 1.12)
    ax.axhline(0, color="#ccc", lw=0.8, zorder=0)
axs[0].text(-9.5, 0.85, "slope", color=RED, fontsize=13); axs[0].text(1.5, 0.12, "ReLU ÷ 10", color=INK, fontsize=13)
axs[1].text(-9.5, 0.85, "value", color=INK, fontsize=13); axs[1].text(-9.5, 0.12, "slope", color=RED, fontsize=13)
fig.tight_layout(w_pad=1.0); save(fig, "fig-activations.svg")

# ------------------------------------------------------------ 5. overfitting, weight decay, dropout: decision regions after 2000 steps
P = np.array(D["points"])
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("bo", ["#bcd3e3", "#ffffff", "#f6d3a6"])
fig, axs = plt.subplots(1, 3, figsize=(1020 / 96, 380 / 96))
for ax, (k, name) in zip(axs, [("overfit", "no regularisation"), ("decay", "weight decay λ = 0.003"), ("dropout", "dropout 0.4")]):
    g = np.array(D[k]["grid"]).reshape(100, 100)
    ax.imshow(g, origin="lower", extent=(-0.1, 1.1, -0.1, 1.1), cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
    cc = -0.1 + 1.2 * (np.arange(100) + 0.5) / 100
    ax.contour(cc, cc, g, levels=[0.5], colors=INK, linewidths=2)
    for x, y, c, v in P:
        col = ORANGE if c == 1 else BLUE
        ax.scatter([x], [y], s=46, marker="s" if c == 1 else "o", c="white" if v else col, edgecolor=col, lw=2, zorder=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#999")
    ax.set_title(name, fontsize=15)
    ax.set_xlabel(f"training loss {D[k]['train']:.3f}\nvalidation loss {D[k]['val']:.3f}", fontsize=14)
fig.tight_layout(w_pad=1.2); save(fig, "fig-regularise.png", dpi=192, bbox_inches="tight", pad_inches=0.05)
print({k: (round(D[k]["train"], 3), round(D[k]["val"], 3)) for k in ["overfit", "decay", "dropout"]})
