# Slide figures for track-linking/lec.qmd. Run from anywhere: python3 -I tools/figs/track-linking/figs.py
# Figure sizes are the display size in px / 96; 16 pt text shows at about 18 px in a 45 % column.
# (The older diagrams fig-frames, fig-nn-fail, fig-abpq come from tools/figs/tracking/figs.ts.)
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(R, "track-linking") + "/"
plt.rcParams.update({"font.size": 17, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "track-linking"})
INK, BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#1b2a3a", "#1f77b4", "#e69f00", "#2a9d4b", "#d62728", "#cc79a7", "#888"
def save(fig, name):
    fig.savefig(OUT + name, facecolor="white", metadata={"Date": None}); plt.close(fig); print("wrote", OUT + name)

# ------------------------------------------------------------ 1. difficulty ratio -> NN error
rho = np.linspace(0, 1.1, 300)
a = np.pi / 4 * rho ** 2
fig, ax = plt.subplots(figsize=(520 / 96, 330 / 96))
ax.axvspan(0, 0.2, color=GREEN, alpha=0.12, lw=0)
ax.axvspan(0.8, 1.1, color=RED, alpha=0.10, lw=0)
ax.text(0.1, 2.5, "easy", ha="center", color=GREEN)
ax.text(0.95, 8, "nobody\ncan", ha="center", color=RED)
ax.plot(rho, 100 * a / (1 + a), color=BLUE, lw=3, label="formula a/(1+a)")
ax.plot([0.4], [11.2], "o", color=INK, ms=9)
ax.annotate("worked example:\nratio 0.4 → 11 %", (0.4, 11.2), (0.02, 24), arrowprops=dict(arrowstyle="->", color=INK))
ax.plot([0.5], [15.5], "s", color=BLUE, ms=9, mfc="white", mew=2.5, label="demo: nearest")
ax.plot([0.5, 1.0], [9.3, 34], "D", color=ORANGE, ms=9, label="demo: global")
ax.set_xlabel("difficulty = displacement / spacing"); ax.set_ylabel("wrong links (%)")
ax.set_xlim(0, 1.1); ax.set_ylim(0, 62); ax.set_yticks([0, 10, 20, 30, 40, 50])
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.14, 1.03), fontsize=17)
fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.2)
save(fig, "fig-difficulty.svg")

# ------------------------------------------------------------ 2. squared distance = -log probability
r = np.linspace(-2.6, 2.6, 400)   # step in units of sqrt(4 D dt)
fig, ax = plt.subplots(1, 2, figsize=(540 / 96, 300 / 96))
ax[0].plot(r, np.exp(-r ** 2), color=BLUE, lw=3); ax[0].fill_between(r, np.exp(-r ** 2), color=BLUE, alpha=0.12)
ax[0].set_title("probability of a step", fontsize=17); ax[0].set_yticks([0, 1]); ax[0].set_ylim(0, 1.1)
ax[1].plot(r, r ** 2, color=ORANGE, lw=3)
ax[1].set_title("cost = −log p", fontsize=17); ax[1].set_ylim(0, 6.8); ax[1].set_yticks([0, 2, 4, 6])
for k, (x, c) in enumerate([(0.5, GREEN), (2.0, RED)]):
    ax[0].plot([x], [np.exp(-x * x)], "o", color=c, ms=9); ax[1].plot([x], [x * x], "o", color=c, ms=9)
ax[1].text(1.6, 4.0, "4", ha="right", va="center", color=RED); ax[1].text(0.5, 0.25 + 0.7, "0.25", ha="center", color=GREEN)
for a_ in ax:
    a_.set_xlabel("step r (rms units)"); a_.set_xticks([-2, 0, 2])
fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.21, wspace=0.3)
save(fig, "fig-cost.svg")

# ------------------------------------------------------------ 3. a gate: candidates and the sparse cost matrix
T0 = {"A": (1.0, 2.0), "B": (3.6, 2.6), "C": (6.0, 1.6)}
T1 = {"P": (1.7, 2.6), "Q": (3.9, 1.6), "S": (8.6, 2.2)}
G = 1.5
fig = plt.figure(figsize=(560 / 96, 380 / 96))
ax = fig.add_axes([0.0, 0.42, 1.0, 0.58]); ax.set_aspect("equal"); ax.axis("off")
for n, (x, y) in T0.items():
    ax.add_patch(plt.Circle((x, y), G, fc=BLUE, alpha=0.08, ec=BLUE, ls="--", lw=1.5, clip_on=False))
for n, (x, y) in T0.items():
    for m, (u, v) in T1.items():
        if np.hypot(u - x, v - y) <= G: ax.plot([x, u], [y, v], color=INK, lw=2, zorder=1)
ax.annotate("", T1["S"], T0["C"], arrowprops=dict(arrowstyle="->", color=RED, lw=2, ls=":"))
ax.text(8.2, 1.25, "outside the gate", color=RED, ha="center", fontsize=17)
for n, (x, y) in T0.items():
    ax.plot(x, y, "o", ms=13, color=BLUE); ax.text(x - 0.35, y - 0.05, n, ha="right", va="center", fontsize=17)
for m, (u, v) in T1.items():
    ax.plot(u, v, "o", ms=13, mfc="white", mec=ORANGE, mew=3); ax.text(u + 0.32, v + 0.05, m, ha="left", va="center", fontsize=17)
ax.set_xlim(-0.6, 10.0); ax.set_ylim(-0.05, 4.25)
# matrix
ax2 = fig.add_axes([0.0, 0.0, 0.52, 0.42]); ax2.axis("off"); ax2.set_xlim(0, 4); ax2.set_ylim(0, 4)
names0, names1 = list(T0), list(T1)
for j, m in enumerate(names1): ax2.text(1.5 + j, 3.5, m, ha="center", va="center", fontsize=17, color=ORANGE, weight="bold")
for i, n in enumerate(names0):
    ax2.text(0.5, 2.5 - i, n, ha="center", va="center", fontsize=17, color=BLUE, weight="bold")
    for j, m in enumerate(names1):
        (x, y), (u, v) = T0[n], T1[m]
        d2 = (u - x) ** 2 + (v - y) ** 2
        inside = np.sqrt(d2) <= G
        ax2.add_patch(plt.Rectangle((1 + j, 2 - i), 1, 1, fc="#f2f2f2" if not inside else "white", ec="#bbb"))
        ax2.text(1.5 + j, 2.5 - i, f"{d2:.2f}" if inside else "–", ha="center", va="center", fontsize=17, color=INK if inside else GREY)
ax2.text(4.15, 1.5, "cost = d²\n– = not a candidate\nC: no candidate,\nits track ends", va="center", fontsize=17, color=GREY)
save(fig, "fig-gate.svg")
for n in T0:
    print(n, [round((T1[m][0]-T0[n][0])**2+(T1[m][1]-T0[n][1])**2, 2) for m in T1])

# ------------------------------------------------------------ 4. best total cost against b (A, B, P, Q)
b = np.linspace(0, 5, 501)
fig, ax = plt.subplots(figsize=(520 / 96, 340 / 96))
opts = [(4 * b, "no links: 4b", PURPLE), (1 + 2 * b, "B–P only: 1 + 2b", ORANGE), (np.full_like(b, 8.0), "A–P + B–Q: 8", BLUE)]
for y, lab, c in opts: ax.plot(b, y, color=c, lw=1.5, ls="--")
best = np.minimum.reduce([o[0] for o in opts])
ax.plot(b, best, color=INK, lw=4, label="cheapest")
ax.axvline(0.5, color=GREY, lw=1, ls=":"); ax.axvline(3.5, color=GREY, lw=1, ls=":")
ax.text(4.95, 6.3, "A–P + B–Q: 8", color=BLUE, ha="right")
ax.text(2.6, 2.6, "B–P only:\n1 + 2b", color=ORANGE, ha="center")
ax.text(0.1, 6.3, "no links:\n4b", color=PURPLE, ha="left")
ax.set_xlabel("cost b of ending or starting a track"); ax.set_ylabel("best total cost")
ax.set_xticks([0, 0.5, 1, 2, 3, 3.5, 4, 5]); ax.set_xticklabels(["0", "0.5", "1", "2", "3", "3.5", "4", "5"])
ax.set_ylim(0, 10); ax.set_xlim(0, 5)
fig.subplots_adjust(left=0.13, right=0.97, top=0.96, bottom=0.18)
save(fig, "fig-endcost.svg")

# ------------------------------------------------------------ 5. errors compound
T = np.arange(0, 61)
fig, ax = plt.subplots(figsize=(560 / 96, 340 / 96))
for e, c, lab in [(0.005, GREEN, "0.5 %\nper link"), (0.02, ORANGE, "2 %"), (0.10, RED, "10 %")]:
    ax.plot(T, 100 * (1 - e) ** T, color=c, lw=3); ax.text(61, 100 * (1 - e) ** 60, lab, color=c, va="center")
    for t in (20, 50): ax.plot([t], [100 * (1 - e) ** t], "o", color=c, ms=7)
ax.axvline(20, color=GREY, lw=1, ls=":"); ax.axvline(50, color=GREY, lw=1, ls=":")
ax.set_xlabel("track length (links)"); ax.set_ylabel("tracks perfect (%)")
ax.set_ylim(0, 102); ax.set_xlim(0, 60); ax.set_xticks([0, 20, 40, 50, 60])
fig.subplots_adjust(left=0.15, right=0.72, top=0.96, bottom=0.18)
save(fig, "fig-compound.svg")
