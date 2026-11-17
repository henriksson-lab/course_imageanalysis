# Figures for drifting/lec.qmd (and the exercise). Run from the repository root after data.ts
# (see its header):
#   python3 -I tools/figs/drifting/figs.py
# data.json holds the drifting demo's presets run with its own model (free points: 300 steps of
# η = 0.5; generator: 1500 Adam steps). fig-field.svg is computed here from the formulas on a small
# hand-made example. Figure sizes are the display size in px / 96.
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/drifting/'
D = json.load(open(R + '/tools/figs/drifting/data.json'))
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, GREY = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c", "#b8b8b8"
REAL = np.array(D["real"]).reshape(-1, 2)


def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw)
    plt.close(fig)


def pts(a):
    return np.array(a).reshape(-1, 2)


def panel(ax, P, title, V=None, views=None, lim=2.4):
    ax.scatter(REAL[:, 0], REAL[:, 1], s=7, c=GREY, lw=0)
    if views is not None:
        ax.scatter(views[:, 0], views[:, 1], s=5, c=BLUE, alpha=0.3, lw=0)
    if V is not None:
        for (x, y), (u, v) in zip(P, V):
            ax.plot([x, x + 0.5 * u], [y, y + 0.5 * v], color=ORANGE, lw=1.2, clip_on=True)
    ax.scatter(P[:, 0], P[:, 1], s=14, c=BLUE, lw=0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#ccc")
    ax.set_title(title, fontsize=14)


def kmean(x, S, tau, skip=None):
    d = np.sqrt(((S - x) ** 2).sum(1))
    w = np.exp(-(d - d.min()) / tau)
    if skip is not None:
        w[skip] = 0
    return (w[:, None] * S).sum(0) / w.sum()


# ------------------------------------------------------------ 1. the field at one point
rng = np.random.default_rng(3)
real = rng.normal([1.3, 1.0], 0.3, (16, 2))
gen = np.r_[rng.normal([-0.9, 0.7], 0.3, (10, 2)), [[0.0, 0.0]]]
i, tau = len(gen) - 1, 1.0
x = gen[i]
mp, mm = kmean(x, real, tau), kmean(x, gen, tau, skip=i)
vp, vm = mp - x, mm - x
fig, ax = plt.subplots(figsize=(470 / 96, 400 / 96))
ax.set_xlim(-1.8, 2.6); ax.set_ylim(-1.3, 2.0)
ax.scatter(real[:, 0], real[:, 1], s=40, c=GREY, label="real points", zorder=2)
ax.scatter(gen[:-1, 0], gen[:-1, 1], s=40, c=BLUE, alpha=0.55, label="other generated points", zorder=2)
ax.scatter([x[0]], [x[1]], s=110, c=BLUE, edgecolors=INK, zorder=4)
kw = dict(arrowstyle="-|>", mutation_scale=18, lw=2.6)
ax.annotate("", xy=x + vp, xytext=x, arrowprops=dict(color=GREEN, **kw), zorder=5)
ax.annotate("", xy=x - vm, xytext=x, arrowprops=dict(color=RED, **kw), zorder=5)
ax.annotate("", xy=x + vp - vm, xytext=x, arrowprops=dict(color=ORANGE, **kw), zorder=6)
ax.text(*(x + vp + [0.05, 0.08]), "pull V⁺", color=GREEN, fontsize=14)
ax.text(*(x - vm + [-0.15, -0.28]), "push −V⁻", color=RED, fontsize=14)
ax.text(*(x + vp - vm + [0.08, -0.05]), "V = V⁺ − V⁻", color=ORANGE, fontsize=14)
ax.scatter([mp[0]], [mp[1]], marker="x", s=70, c=GREEN, zorder=5)
ax.scatter([mm[0]], [mm[1]], marker="x", s=70, c=RED, zorder=5)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=12, frameon=False)
fig.tight_layout(); save(fig, "fig-field.svg")
print("fig-field: V+ =", np.round(vp, 3), "V- =", np.round(vm, 3))

# ------------------------------------------------------------ 2. noise drifts to the moons
fig, axs = plt.subplots(1, 4, figsize=(1000 / 96, 280 / 96))
d = D["drift"]
for ax, s in zip(axs, ["0", "10", "50", "300"]):
    panel(ax, pts(d["snaps"][s]), f"step {s}: coverage {d['cov'][int(s)]:.2f}", V=pts(d["vs"][s]) if s == "0" else None)
fig.tight_layout(w_pad=0.6); save(fig, "fig-snapshots.svg")

# ------------------------------------------------------------ 3. |V| and coverage against the step
fig, axs = plt.subplots(1, 2, figsize=(560 / 96, 260 / 96))
st = np.arange(len(D["drift"]["norm"]))
for key, col, lab in [("drift", BLUE, "drifting"), ("attract", RED, "attraction only"), ("large", ORANGE, "τ = 3")]:
    axs[0].semilogy(st, np.maximum(D[key]["norm"], 1e-4), color=col, lw=2, label=lab)
    axs[1].plot(st, D[key]["cov"], color=col, lw=2, label=lab)
axs[1].axhline(0.897, color="#888", ls="--", lw=1.5)
axs[1].text(300, 0.84, "new draw of\n200 real points", ha="right", va="top", fontsize=11, color="#666")
axs[0].set_title("mean |V|", fontsize=14); axs[1].set_title("coverage", fontsize=14)
for ax in axs:
    ax.set_xlabel("step")
axs[1].set_ylim(0, 1.02); axs[0].legend(fontsize=11, frameon=False)
fig.tight_layout(); save(fig, "fig-curves.svg")

# ------------------------------------------------------------ 4. the temperature
fig, axs = plt.subplots(1, 3, figsize=(780 / 96, 290 / 96))
for ax, (key, t) in zip(axs, [("small", "τ = 0.03"), ("drift", "τ = 0.2"), ("large", "τ = 3")]):
    panel(ax, pts(D[key]["snaps"]["300"]), f"{t}: coverage {D[key]['cov'][300]:.2f}")
fig.tight_layout(w_pad=0.6); save(fig, "fig-tau.svg")

# ------------------------------------------------------------ 5. attraction only = mean shift
fig, axs = plt.subplots(1, 2, figsize=(540 / 96, 290 / 96))
panel(axs[0], pts(D["attract"]["snaps"]["300"]), f"pull only: {D['attract']['cov'][300]:.2f}")
panel(axs[1], pts(D["drift"]["snaps"]["300"]), f"pull − push: {D['drift']['cov'][300]:.2f}")
fig.tight_layout(w_pad=0.6); save(fig, "fig-attract.svg")

# ------------------------------------------------------------ 6. twenty points, with and without views
fig, axs = plt.subplots(1, 2, figsize=(540 / 96, 290 / 96))
panel(axs[0], pts(D["few"]["snaps"]["300"]), f"20 points: {D['few']['cov'][300]:.2f}")
panel(axs[1], pts(D["views"]["snaps"]["300"]), f"20 points × 8 views: {D['views']['cov'][300]:.2f}", views=pts(D["views"]["views"]))
fig.tight_layout(w_pad=0.6); save(fig, "fig-views.svg")

# ------------------------------------------------------------ 7. the generator
fig, axs = plt.subplots(1, 4, figsize=(1000 / 96, 280 / 96))
g, ga = D["gen"], D["genattract"]
for ax, (P, t) in zip(axs, [(g["snaps"]["0"], "untrained"), (g["snaps"]["100"], "step 100"), (g["snaps"]["1500"], f"step 1500: {g['cov'][-1]:.2f}"), (ga["snaps"]["1500"], f"pull only, 1500: {ga['cov'][-1]:.2f}")]):
    panel(ax, pts(P), t)
fig.tight_layout(w_pad=0.6); save(fig, "fig-generator.svg")

# ------------------------------------------------------------ 8. targets of the stop-gradient loss (step 10 of the drift preset)
fig, ax = plt.subplots(figsize=(420 / 96, 400 / 96))
P, V = pts(D["drift"]["snaps"]["0"]), pts(D["drift"]["vs"]["0"])
ax.scatter(REAL[:, 0], REAL[:, 1], s=7, c=GREY, lw=0)
sel = [k for k in range(0, len(P), 3) if np.abs(P[k]).max() < 2.2 and np.abs(P[k] + V[k]).max() < 2.2]
for k in sel:
    ax.annotate("", xy=P[k] + V[k], xytext=P[k], arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3, mutation_scale=10))
ax.scatter(P[sel, 0], P[sel, 1], s=18, c=BLUE, lw=0, label="samples f(ε)", zorder=3)
ax.scatter((P + V)[sel, 0], (P + V)[sel, 1], s=22, facecolors="none", edgecolors=ORANGE, lw=1.3, label="targets f(ε) + V (frozen)", zorder=3)
ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.4, 2.4); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.legend(loc="lower left", fontsize=11, frameon=True, framealpha=0.85)
fig.tight_layout(); save(fig, "fig-targets.svg")
print("done")
