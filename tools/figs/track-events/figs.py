# Slide figures for track-events/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/track-events/figs.py
# fig-censor runs the gapclose demo's own simulation (preset "censor") through esbuild + node
# (cells.ts), so its numbers are the demo's. Sizes are display px / 96.
# (fig-events and fig-lineage come from tools/figs/tracking/figs.ts.)
import os, json, subprocess, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(R, "track-events") + "/"
plt.rcParams.update({"font.size": 17, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "track-events"})
INK, BLUE, ORANGE, GREEN, RED, PURPLE, GREY, GOLD = "#1b2a3a", "#1f77b4", "#e69f00", "#2a9d4b", "#d62728", "#cc79a7", "#888", "#e69f00"
SEG = ["#1f77b4", "#2a9d4b", "#8c564b", "#555555", "#56b4e9"]
def save(fig, name):
    fig.savefig(OUT + name, facecolor="white", metadata={"Date": None}); plt.close(fig); print("wrote", OUT + name)

# ------------------------------------------------------------ 1. two stages: segments, then joins (time down)
# segments as (x positions over frames t0..t1)
segs = [(0, [1.0, 1.1, 0.9, 1.0, 1.1]),             # ends at 4 (missed frame 5)
        (6, [1.2, 1.1, 1.2, 1.0, 1.1]),             # restarts at 6: gap
        (0, [3.6, 3.7, 3.6, 3.7, 3.6, 3.4, 3.2, 3.1, 3.0, 3.0]),   # mother -> daughter 1
        (6, [4.3, 4.6, 4.8, 4.9]),                  # daughter 2: split from mother at frame 5
        (0, [6.0, 6.1, 6.3, 6.5, 6.7, 6.9]),        # object that touches: merges at frame 6
        (0, [8.0, 7.9, 7.8, 7.7, 7.6, 7.5, 7.4, 7.4, 7.5, 7.6])]
cols = [SEG[0], SEG[0], SEG[1], SEG[2], SEG[3], SEG[4]]
fig, axs = plt.subplots(1, 2, figsize=(580 / 96, 340 / 96))
for k, ax in enumerate(axs):
    for (t0, xs), c in zip(segs, cols if k else [GREY] * 6):
        t = np.arange(t0, t0 + len(xs))
        ax.plot(xs, t, "-", color=c if k else "#999", lw=3); ax.plot(xs, t, "o", color=c if k else "#999", ms=5)
    if k:
        ax.plot([1.1, 1.2], [4, 6], "--", color=BLUE, lw=3); ax.text(0.2, 5.3, "gap", color=BLUE, ha="center", fontsize=17)
        ax.plot([3.4, 4.3], [5, 6], "--", color=ORANGE, lw=3); ax.text(4.6, 4.6, "split", color=ORANGE, ha="left", fontsize=17)
        ax.plot([6.9, 7.4], [5, 6], "--", color=PURPLE, lw=3); ax.text(6.4, 7.6, "merge", color=PURPLE, ha="center", fontsize=17)
    ax.set_ylim(10, -0.8); ax.set_xlim(-0.3, 8.8); ax.set_xticks([]); ax.set_yticks([0, 5, 9] if k == 0 else [])
    ax.set_title(["1: strict segments", "2: join ends and starts"][k], fontsize=17)
    ax.set_xlabel("x")
axs[0].set_ylabel("frame (time ↓)")
axs[1].spines["left"].set_visible(False)
fig.subplots_adjust(left=0.1, right=0.99, top=0.9, bottom=0.1, wspace=0.08)
save(fig, "fig-twostage.svg")

# ------------------------------------------------------------ 2. the gap gate grows as sqrt(g)
fig, ax = plt.subplots(figsize=(600 / 96, 340 / 96))
end = np.array([0.0, 0.0])
for g, c in zip([1, 2, 3], ["#9ecae1", "#4292c6", "#08519c"]):
    r = np.sqrt(g)
    ax.add_patch(Circle(end, r, fc="none", ec=c, lw=2.5, ls="--"))
    ax.plot([], [], color=c, lw=2.5, ls="--", label=f"g = {g}: radius √{g}" if g > 1 else "g = 1: radius 1")
ax.plot([-1.9, -1.2, -0.6, 0.0], [1.1, 0.7, 0.45, 0.0], "-", color=BLUE, lw=3)
ax.plot(*end, "o", ms=12, color=BLUE); ax.text(-0.15, -0.38, "track ends", ha="left", fontsize=17)
for p, c, lab in [((1.25, -0.9), GREEN, "start after 3 frames:\ninside the g = 3 gate"), ((2.0, -0.75), RED, "too far")]:
    ax.plot(*p, "o", ms=12, mfc="white", mec=c, mew=3)
    ax.text(p[0] + 0.15, p[1] - 1.0 if c == GREEN else p[1] + 0.1, lab, color=c, fontsize=17, va="top" if c == GREEN else "bottom", ha="center" if c == GREEN else "left")
ax.set_aspect("equal"); ax.set_xlim(-2.2, 6.8); ax.set_ylim(-3.1, 2.0); ax.axis("off")
ax.legend(frameon=False, fontsize=17, loc="upper right", bbox_to_anchor=(1.02, 1.02), title="gate after g frames", title_fontsize=17)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
save(fig, "fig-gapgate.svg")

# ------------------------------------------------------------ 3. a division: mother and both daughters
fig, ax = plt.subplots(figsize=(560 / 96, 260 / 96))
ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.5, 11.5); ax.set_ylim(-2.6, 2.6)
ax.add_patch(Ellipse((1, 0), 2.2, 1.6, fc="#c6dbef", ec=BLUE, lw=2)); ax.text(1, 0, "100", ha="center", va="center", fontsize=17)
ax.add_patch(Circle((5, 0), 1.0, fc="#6baed6", ec=BLUE, lw=2.5)); ax.text(5, 0, "100", ha="center", va="center", fontsize=17)
ax.text(5, -1.55, "round, bright", ha="center", fontsize=17)
for y, a, ls, c in [(0.95, "52", "-", INK), (-0.95, "49", "--", ORANGE)]:
    ax.add_patch(Circle((9.2, y), 0.72, fc="#c6dbef", ec=BLUE, lw=2)); ax.text(9.2, y, a, ha="center", va="center", fontsize=17)
    ax.annotate("", (8.4, y * 0.95), (6.05, y * 0.2), arrowprops=dict(arrowstyle="->", color=c, lw=2.5, ls=ls))
ax.annotate("", (3.95, 0), (2.15, 0), arrowprops=dict(arrowstyle="->", color=INK, lw=2.5))
for x, t in [(1, "t − 1"), (5, "t"), (9.2, "t + 1")]: ax.text(x, 2.25, t, ha="center", fontsize=17, weight="bold")
ax.text(9.2, -2.35, "split link", color=ORANGE, fontsize=17, ha="center")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
save(fig, "fig-division.svg")

# ------------------------------------------------------------ 4. division or bubble (space-time, time down)
fig, axs = plt.subplots(1, 2, figsize=(540 / 96, 360 / 96))
t = np.arange(0, 11)
for ax, kind in zip(axs, ["div", "bub"]):
    if kind == "div":
        ax.plot(np.full(5, 2.0), t[:5], "-o", color=BLUE, lw=3, ms=5)
        ax.plot(2.0 - 0.25 * (t[5:] - 4), t[5:], "-o", color=BLUE, lw=3, ms=5)
        ax.plot(2.0 + 0.25 * (t[5:] - 4), t[5:], "-o", color=ORANGE, lw=3, ms=5)
        ax.set_title("division", fontsize=17); ax.text(2, 11.6, "apart ≥ 3 frames:\nkept", ha="center", fontsize=17, color=GREEN, va="top")
    else:
        ax.plot(np.full(11, 2.0), t, "-", color=BLUE, lw=3, alpha=0.0)
        ax.plot(np.full(5, 2.0), t[:5], "-o", color=BLUE, lw=3, ms=5); ax.plot(np.full(5, 2.0), t[6:], "-o", color=BLUE, lw=3, ms=5)
        ax.plot([2.0, 1.6, 2.0], [4, 5, 6], "-o", color=BLUE, lw=3, ms=5); ax.plot([2.0, 2.4, 2.0], [4, 5, 6], "-o", color=ORANGE, lw=3, ms=5)
        ax.set_title("segmentation split", fontsize=17); ax.text(2, 11.6, "rejoins after\n1 frame: a “bubble”,\nremoved", ha="center", fontsize=17, color=RED, va="top")
    ax.set_ylim(15.5, -0.6); ax.set_xlim(0.4, 3.6); ax.set_xticks([]); ax.set_yticks([0, 5, 10] if kind == "div" else [])
    ax.set_xlabel("")
axs[0].set_ylabel("frame (time ↓)"); axs[1].spines["left"].set_visible(False)
for ax in axs: ax.spines["bottom"].set_visible(False)
fig.subplots_adjust(left=0.13, right=0.99, top=0.9, bottom=0.02, wspace=0.1)
save(fig, "fig-bubble.svg")

# ------------------------------------------------------------ 5. censoring (gapclose demo, preset "censor")
bundle = subprocess.run([os.path.join(R, "node_modules/.bin/esbuild"), "tools/figs/track-events/cells.ts", "--bundle",
                         "--platform=node", "--log-level=warning"], cwd=R, check=True, capture_output=True, text=True).stdout
D = json.loads(subprocess.run(["node"], input=bundle, check=True, capture_output=True, text=True).stdout)
cells = sorted([c for c in D["cells"] if c["parent"] >= 0], key=lambda c: (c["t0"], c["id"]))
comp = [c["t1"] - c["t0"] + 1 for c in cells if c["divides"]]
fig, ax = plt.subplots(figsize=(540 / 96, 360 / 96))
for i, c in enumerate(cells):
    if c["divides"]: ax.plot([c["t0"], c["t1"] + 1], [i, i], color=BLUE, lw=4, solid_capstyle="butt")
    else: ax.plot([c["t0"], D["T"]], [i, i], color=ORANGE, lw=4, solid_capstyle="butt"); ax.plot([D["T"], D["T"] + 3], [i, i], color=ORANGE, lw=2, ls=":")
ax.axvline(D["T"], color=INK, lw=2); ax.text(D["T"] - 0.5, -2.5, "movie ends", ha="right", fontsize=17)
ax.plot([], [], color=BLUE, lw=4, label=f"complete: {len(comp)}, mean {np.mean(comp):.1f}")
ax.plot([], [], color=ORANGE, lw=4, label=f"censored: {len(cells) - len(comp)}")
ax.legend(frameon=False, loc="lower left", fontsize=17, bbox_to_anchor=(-0.02, -0.02))
ax.set_xlabel("frame"); ax.set_ylabel("cells, by birth time"); ax.set_yticks([]); ax.set_ylim(len(cells) + 0.5, -4)
ax.set_xlim(0, 64); ax.set_xticks([0, 20, 40, 60])
fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.17)
save(fig, "fig-censor.svg")
print("complete", len(comp), np.mean(comp), "censored", len(cells) - len(comp))
