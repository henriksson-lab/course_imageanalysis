# Figures for the neuralnet unit (neuralnet/lec.qmd). Run: python3 -I tools/figs/neuralnet/figs.py
# Figure sizes are the display size in px / 96, so 13 pt text shows at about 17 px.
# All numbers are the ones on the slides: XOR network h1 = ReLU(x1 + x2), h2 = ReLU(x1 + x2 - 1),
# out = h1 - 2 h2; one-weight loss L(w) = (3 - 2w)^2; softmax of (2, 1, 0).
import re, base64
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/neuralnet/'
plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c"
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)

# ------------------------------------------------------------ 1. objects in a 2D feature space, one line
rng = np.random.default_rng(3)
nA = rng.normal([110, 160], [22, 18], (40, 2))   # nuclei: larger, brighter
nB = rng.normal([55, 95], [20, 22], (40, 2))     # debris: smaller, dimmer
w1, w2, b = 1.0, 1.2, -250.0                      # the line x1 + 1.2 x2 = 250
fig, ax = plt.subplots(figsize=(470 / 96, 400 / 96))
ax.scatter(*nB.T, s=42, c=BLUE, marker="o", edgecolor="white", lw=0.8, label="debris")
ax.scatter(*nA.T, s=46, c=ORANGE, marker="s", edgecolor="white", lw=0.8, label="nucleus")
xx = np.array([0, 200])
ax.plot(xx, (-b - w1 * xx) / w2, color=INK, lw=2.5)
ax.annotate("", xy=(150, 133), xytext=(125, 103), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
ax.text(152, 120, "w", fontsize=16, fontweight="bold")
ax.text(18, 205, "$w_1x_1 + w_2x_2 + b = 0$", fontsize=14, bbox=dict(fc="white", ec="none", alpha=0.85))
ax.set_xlim(0, 180); ax.set_ylim(30, 220)
ax.set_xlabel("$x_1$: area (px²)"); ax.set_ylabel("$x_2$: mean intensity")
ax.legend(loc="lower right", frameon=False, fontsize=13)
fig.tight_layout(); save(fig, "fig-scatter.svg")

# ------------------------------------------------------------ 2. AND, OR, XOR on the four corners
C = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
T = {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "XOR": [0, 1, 1, 0]}
fig, axs = plt.subplots(1, 3, figsize=(620 / 96, 240 / 96))
for ax, (name, t) in zip(axs, T.items()):
    for (x, y), v in zip(C, t):
        ax.scatter(x, y, s=260, c=ORANGE if v else "white", edgecolor=ORANGE if v else BLUE, lw=2.5,
                   marker="s" if v else "o", zorder=3)
    xx = np.array([-0.4, 1.4])
    if name == "AND": ax.plot(xx, 1.5 - xx, color=INK, lw=2.5)
    if name == "OR": ax.plot(xx, 0.5 - xx, color=INK, lw=2.5)
    if name == "XOR": ax.text(0.5, 0.5, "no line\nworks", ha="center", va="center", fontsize=14, color=RED)
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.4, 1.4); ax.set_aspect("equal")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xlabel("$x_1$", labelpad=0)
    ax.set_title(name, fontsize=15)
axs[0].set_ylabel("$x_2$")
fig.tight_layout(w_pad=1.5); save(fig, "fig-andor.svg")

# ------------------------------------------------------------ 3. the hidden layer straightens XOR
fig, axs = plt.subplots(1, 2, figsize=(600 / 96, 330 / 96), gridspec_kw=dict(width_ratios=[1, 1.35]))
ax = axs[0]
xor = [0, 1, 1, 0]
for (x, y), v in zip(C, xor):
    ax.scatter(x, y, s=200, c=ORANGE if v else "white", edgecolor=ORANGE if v else BLUE, lw=2.5, marker="s" if v else "o", zorder=3)
xx = np.array([-0.4, 1.5])
ax.plot(xx, -xx, color=GREEN, lw=2, ls="--"); ax.plot(xx, 1 - xx, color=GREEN, lw=2)
ax.text(1.0, 0.25, "$h_2 = 0$\nbelow", fontsize=13, color=GREEN)
ax.set_xlim(-0.4, 1.5); ax.set_ylim(-0.4, 1.5); ax.set_aspect("equal")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$", labelpad=0)
ax.set_title("input", fontsize=14)
ax = axs[1]
H = np.array([[0, 0], [1, 0], [2, 1]])
ax.scatter(0, 0, s=200, c="white", edgecolor=BLUE, lw=2.5, zorder=3)
ax.scatter(2, 1, s=200, c="white", edgecolor=BLUE, lw=2.5, zorder=3)
ax.scatter(1, 0, s=200, c=ORANGE, edgecolor=ORANGE, lw=2.5, marker="s", zorder=3)
ax.text(1.0, -0.5, "(1,0), (0,1)", ha="center", fontsize=13)
ax.text(0.12, 0.12, "(0,0)", fontsize=13); ax.text(1.5, 1.25, "(1,1)", fontsize=13)
hh = np.array([-0.3, 2.5])
ax.plot(hh, (hh - 0.5) / 2, color=INK, lw=2.5)
ax.set_xlim(-0.4, 2.5); ax.set_ylim(-0.7, 1.6); ax.set_aspect("equal")
ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1]); ax.set_xlabel("$h_1$"); ax.set_ylabel("$h_2$", labelpad=0)
ax.set_title("hidden units", fontsize=14)
fig.tight_layout(w_pad=2); save(fig, "fig-xorhidden.svg")

# ------------------------------------------------------------ 4. walking downhill on L(w) = (3 - 2w)^2
L = lambda w: (3 - 2 * w) ** 2
fig, ax = plt.subplots(figsize=(500 / 96, 360 / 96))
w = np.linspace(-0.4, 3.4, 300)
ax.plot(w, L(w), color=INK, lw=2.5)
for eta, w1_, col, dy in [(0.1, 1.2, GREEN, 0), (0.25, 3.0, RED, 0)]:
    assert abs(0 - eta * (-12) - w1_) < 1e-12
    ax.add_patch(FancyArrowPatch((0, L(0)), (w1_, L(w1_)), connectionstyle="arc3,rad=-0.35", arrowstyle="-|>",
                                 mutation_scale=18, color=col, lw=2.2))
    ax.scatter([w1_], [L(w1_)], s=60, c=col, zorder=4)
ax.scatter([0], [9], s=60, c=INK, zorder=4)
ax.text(-0.36, 4.6, "start\nL = 9", fontsize=13)
ax.text(1.3, 4.0, "η = 0.1:\nL = 0.36", color=GREEN, fontsize=13)
ax.text(1.5, 13.0, "η = 0.25: L = 9 again", color=RED, fontsize=13, ha="center")
ax.set_xlabel("weight w"); ax.set_ylabel("loss L(w)")
ax.set_ylim(-0.5, 14.5); ax.set_xlim(-0.4, 3.4)
fig.tight_layout(); save(fig, "fig-descent.svg")

# ------------------------------------------------------------ 5. softmax: scores to probabilities
z = np.array([2.0, 1.0, 0.0]); p = np.exp(z) / np.exp(z).sum()
assert np.allclose(np.round(p, 2), [0.67, 0.24, 0.09])
fig, axs = plt.subplots(1, 2, figsize=(520 / 96, 270 / 96))
cols = [ORANGE, BLUE, GREEN]
for ax, v, lab, fmt in [(axs[0], z, "score $z_k$", "{:g}"), (axs[1], p, "probability $p_k$", "{:.2f}")]:
    ax.bar(range(3), v, color=cols, width=0.65)
    for i, t in enumerate(v): ax.text(i, t + 0.03 * max(v.max(), 1), fmt.format(t), ha="center", va="bottom", fontsize=14)
    ax.set_xticks(range(3), ["1", "2", "3"]); ax.set_xlabel("class")
    ax.set_ylabel(lab); ax.set_ylim(0, v.max() * 1.25 if v.max() > 1 else 1.0)
axs[1].set_yticks([0, 0.5, 1])
fig.tight_layout(w_pad=3)
fig.text(0.5, 0.55, "→", fontsize=26, ha="center", va="center")
save(fig, "fig-softmax.svg")

# ------------------------------------------------------------ 6. a digit as a vector of 784 numbers, and shifted by one pixel
s = open(R + '/demos/src/vaeimages/digit_examples.ts').read()
ims = re.findall(r'"label":"([^"]+)","img":"([^"]+)"', s)
lab, b64 = ims[4]; assert lab == "a test 2"
a = np.frombuffer(base64.b64decode(b64), np.uint8).reshape(28, 28).astype(float)
sh = np.roll(a, 1, axis=1)   # one pixel to the right (the left column is empty, so nothing wraps)
assert a[:, -1].sum() == 0
changed = (a != sh)
nch, nink = int(changed.sum()), int(((a > 0) | (sh > 0)).sum())
print("digit 2: inputs changed by a 1-pixel shift:", nch, "of 784; inputs with ink in either:", nink)
fig = plt.figure(figsize=(1000 / 96, 270 / 96))
gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 9.2], wspace=0.15, hspace=0.5, left=0.12, right=0.99, top=0.9, bottom=0.13)
def frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#999")
for r, (img, name) in enumerate([(a, "digit"), (sh, "shifted\n1 px")]):
    ax = fig.add_subplot(gs[r, 0]); ax.imshow(img, cmap="Greys", vmin=0, vmax=255, interpolation="nearest"); frame(ax)
    ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=13, labelpad=8)
    ax = fig.add_subplot(gs[r, 1]); ax.imshow(img.reshape(1, 784), cmap="Greys", vmin=0, vmax=255, aspect="auto", interpolation="nearest"); frame(ax)
    if r == 0: ax.set_title("the same image as one vector of 784 numbers (row after row)", fontsize=13, loc="left")
ax = fig.add_subplot(gs[2, 1])
ax.imshow(changed.reshape(1, 784), cmap=matplotlib.colors.ListedColormap(["white", RED]), aspect="auto", interpolation="nearest"); frame(ax)
ax.set_ylabel("changed", rotation=0, ha="right", va="center", fontsize=13, color=RED, labelpad=8)
ax.set_xlabel(f"{nch} of the 784 inputs change ({nink} carry ink in either image)", fontsize=13)
save(fig, "fig-vector.png", dpi=192)
