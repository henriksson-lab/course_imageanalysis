# Figures for the optimisers unit (optimisers/lec.qmd). Run (from the repo root) after dump.ts (see its header):
#   python3 -I tools/figs/optimisers/figs.py
# data.json holds the optrace surfaces, paths and step counts, computed with the demo's own code.
# Figure sizes are the display size in px / 96.
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/optimisers/'
D = json.load(open(R + '/tools/figs/optimisers/data.json'))
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, PURPLE = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c", "#7b5ea7"
COL = {"gd": RED, "momentum": ORANGE, "adam": PURPLE}
NAME = {"gd": "gradient descent", "momentum": "momentum", "adam": "Adam"}
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)

def landscape(ax, L, lim=2):
    n = L["n"]; Z = np.array(L["z"]).reshape(n, n)
    A, B = np.meshgrid(np.linspace(-2, 2, n), np.linspace(-2, 2, n))
    lev = L["min"] + np.array([0.01, 0.03, 0.1, 0.3, 0.6, 1, 2, 4, 8, 15])
    ax.contour(A, B, Z, levels=lev, colors="#b9c3cc", linewidths=1)
    for m in L["minima"]:
        ax.scatter([m[0]], [m[1]], marker="*", s=200, c=INK, zorder=5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
def path(ax, pts, col, n=None, lw=2, ms=3.5):
    q = np.array(pts[: (n or len(pts))]); q = np.clip(q, -2.3, 2.3)
    ax.plot(q[:, 0], q[:, 1], "-o", color=col, lw=lw, ms=ms, zorder=4)

# 1. one learning rate, three outcomes on the long valley
fig, axs = plt.subplots(1, 3, figsize=(620 / 96, 270 / 96))
for ax, run, title in zip(axs, D["lr3"]["runs"], ["too small", "just below\nthe limit", "over the limit"]):
    landscape(ax, D["lr3"]); path(ax, run["path"], [BLUE, GREEN, RED][axs.tolist().index(ax)], 40, lw=1.6, ms=2.5)
    ax.set_title(f"η = {run['eta']}\n{title}", fontsize=12)
fig.tight_layout(w_pad=0.5); save(fig, "fig-lr3.svg")

# 2. the race on the long valley, each at its best learning rate
fig, ax = plt.subplots(figsize=(430 / 96, 300 / 96))
landscape(ax, D["valley"])
for r in D["valley"]["runs"]:
    path(ax, r["path"], COL[r["opt"]], 40, lw=1.8, ms=2.5)
    ax.plot([], [], "-o", color=COL[r["opt"]], label=f"{NAME[r['opt']]}, η = {r['eta']}: {r['stepsTo']} steps")
ax.set_ylim(-1.2, 1.2)
ax.legend(loc="lower center", fontsize=11, frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(); save(fig, "fig-race-valley.svg")

# 3. steps to converge against the learning rate
fig, ax = plt.subplots(figsize=(470 / 96, 320 / 96))
e = np.array(D["sweep"]["etas"])
for o in ["gd", "momentum", "adam"]:
    s = np.array(D["sweep"]["steps"][o], dtype=float); s[s < 0] = np.nan
    ax.plot(e, s, "-o", color=COL[o], lw=2.2, ms=4, label=NAME[o])
ax.axvline(0.2, color=RED, ls="--", lw=1.2); ax.text(0.23, 400, "η = 2/10:\nplain\ndescent\ndiverges", color=RED, fontsize=11, va="top")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(40, 2000)
ax.set_xlabel("learning rate η"); ax.set_ylabel("steps to the minimum")
ax.legend(loc="upper right", fontsize=11, bbox_to_anchor=(1.0, 1.02))
fig.tight_layout(); save(fig, "fig-sweep.svg")

# 4. the saddle: same η = 0.03 for all
fig, axs = plt.subplots(1, 2, figsize=(620 / 96, 290 / 96), gridspec_kw={"width_ratios": [1, 1.25]})
landscape(axs[0], D["saddle"])
for r in D["saddle"]["runs"]:
    path(axs[0], r["path"], COL[r["opt"]], 120, lw=1.6, ms=2)
axs[0].scatter([0], [0], marker="x", s=90, c=INK, zorder=6); axs[0].set_title("saddle at the cross", fontsize=13)
ax = axs[1]
for r in D["saddle"]["runs"]:
    ax.plot(np.array(r["losses"]) + 0.25, color=COL[r["opt"]], lw=2.2, label=NAME[r["opt"]])
ax.set_yscale("log"); ax.set_ylim(1e-4, 3); ax.set_xlabel("step"); ax.set_ylabel("loss − min")
ax.legend(fontsize=11, loc="lower left"); ax.set_title("all with η = 0.03", fontsize=13)
fig.tight_layout(); save(fig, "fig-saddle.svg")

# 5. noisy gradients: constant learning rate against warm-up + cosine (plain descent)
fig, axs = plt.subplots(1, 2, figsize=(560 / 96, 280 / 96))
for ax, sc, title in zip(axs, ["constant", "warmup"], ["constant η = 0.1", "warm-up + cosine"]):
    landscape(ax, D["noisyGrid"])
    r = D["noisy"][sc][0]
    path(ax, r["path"], BLUE if sc == "constant" else GREEN, None, lw=1.2, ms=2)
    q = np.array(r["path"][-1]); ax.scatter([q[0]], [q[1]], s=60, c=RED, zorder=6)
    ax.set_title(f"{title}\nlast 50 steps, 20 runs: loss {r['meanTail']:.3f}", fontsize=12)
fig.tight_layout(); save(fig, "fig-noise.svg")

# 6. the schedules
fig, ax = plt.subplots(figsize=(450 / 96, 280 / 96))
names = {"constant": "constant", "step": "step decay", "cosine": "cosine", "warmup": "warm-up + cosine"}
for s, c in zip(["constant", "step", "cosine", "warmup"], [INK, ORANGE, BLUE, GREEN]):
    ax.plot(D["schedules"][s], color=c, lw=2.5, label=names[s])
ax.set_xlabel("step (of 200)"); ax.set_ylabel("learning rate / η")
ax.set_ylim(0, 1.08); ax.legend(fontsize=11, loc="lower left")
fig.tight_layout(); save(fig, "fig-schedules.svg")
print("written")
