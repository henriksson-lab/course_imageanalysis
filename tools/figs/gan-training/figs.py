# Figures for the unit gan-training (gan-training/lec.qmd). Run (from the repo root) after
# tools/figs/gan/dump.ts (see its header):  python3 -I tools/figs/gan-training/figs.py
# Uses tools/figs/gan/data.json (runs of the gan1d demo's line GAN); the JS/Wasserstein curves and
# the precision/recall sketch are computed here.
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/gan-training/'
D = json.load(open(R + '/tools/figs/gan/data.json'))
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, GREY, PURPLE = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c", "#9aa3ab", "#7b5ea7"
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)
xs = np.array(D["xs"])

# ------------------------------------------------------------ 1. collapse: discriminator too slow
run = D["slow"]
fig = plt.figure(figsize=(980 / 96, 300 / 96))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.3])
for j, s in enumerate(run["snaps"]):
    ax = fig.add_subplot(gs[0, j])
    ax.fill_between(xs, 0, D["one"], color=GREY, alpha=0.6, lw=0)
    ax.plot(xs, np.minimum(s["gen"], 2.5), color=BLUE, lw=2.5)
    ax.set_ylim(0, 2.5); ax.set_xlim(-1.5, 3.5); ax.set_title(f"step {s['step']}: a = {s['a']:.2f}", fontsize=13); ax.set_xlabel("x")
    if j: ax.set_yticklabels([])
    else: ax.set_ylabel("density")
ax = fig.add_subplot(gs[0, 3])
st = np.arange(10, 2001, 10)
ax.plot(st, run["recall"], color=PURPLE, lw=2.5, label="recall (coverage)")
ax.plot(st, run["precision"], color=GREEN, lw=2.5, label="precision (realism)")
ax.set_ylim(0, 1.12); ax.set_xlabel("step")
ax.text(450, 0.14, "recall\n(coverage)", color=PURPLE, fontsize=12); ax.text(200, 0.86, "precision (realism)", color=GREEN, fontsize=12)
fig.tight_layout(); save(fig, "fig-collapse.svg")

# ------------------------------------------------------------ 2. JS against Wasserstein
fig, axs = plt.subplots(1, 2, figsize=(860 / 96, 300 / 96), gridspec_kw=dict(width_ratios=[1.2, 1]))
g = np.linspace(-12, 12, 6001); dx = g[1] - g[0]
npdf = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
p = npdf(g, 0, 0.5)
dist = np.linspace(0, 6, 61); js = []
for d in dist:
    q = npdf(g, d, 0.5); m = p + q
    with np.errstate(divide="ignore", invalid="ignore"):
        t = 0.5 * np.where(p > 0, p * np.log(2 * p / m), 0) + 0.5 * np.where(q > 0, q * np.log(2 * q / m), 0)
    js.append(np.nansum(t) * dx)
ax = axs[0]
ax.fill_between(g, 0, p, color=GREY, alpha=0.6, lw=0); ax.text(-0.7, 0.85, "data", color="#666")
for d, a in [(1.5, 1.0), (4, 0.6)]:
    ax.plot(g, npdf(g, d, 0.5), color=BLUE, lw=2, alpha=a)
ax.annotate("", xy=(4, 0.45), xytext=(1.5, 0.45), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
ax.text(2.75, 0.5, "generator moves away", ha="center", fontsize=12)
ax.set_xlim(-2, 6); ax.set_yticks([]); ax.set_xlabel("x")
ax = axs[1]
ax.plot(dist, js, color=BLUE, lw=2.5, label="Jensen–Shannon")
ax.axhline(np.log(2), color=GREY, ls="--", lw=1); ax.text(3.0, 0.6, "ln 2: flat, no slope", fontsize=12, color="#555")
ax.set_xlabel("distance between the bumps"); ax.set_ylim(0, 1.2); ax.set_ylabel("Jensen–Shannon", color=BLUE)
a2 = ax.twinx(); a2.spines["right"].set_visible(True); a2.plot(dist, dist, color=ORANGE, lw=2.5); a2.set_ylim(0, 7.2)
a2.set_ylabel("Wasserstein-1", color=ORANGE)
fig.tight_layout(); save(fig, "fig-js-w.svg")

# ------------------------------------------------------------ 3. two bumps: dropped or in between
fig, axs = plt.subplots(1, 2, figsize=(860 / 96, 280 / 96), sharey=True)
for ax, key, title in [(axs[0], "drop", "best discriminator: one bump dropped"), (axs[1], "between", "learned quadratic D: mass in between")]:
    s = D[key]["snaps"][-1]
    ax.fill_between(xs, 0, D["two"], color=GREY, alpha=0.6, lw=0)
    ax.plot(xs, np.minimum(s["gen"], 1.6), color=BLUE, lw=2.5)
    pr, rc = D[key]["precision"][-1], D[key]["recall"][-1]
    ax.set_title(title, fontsize=13); ax.set_xlabel("x"); ax.set_ylim(0, 1.45)
    ax.text(2.6, 1.0, f"precision {pr:.2f}\nrecall {rc:.2f}", fontsize=13)
axs[0].set_ylabel("density")
fig.tight_layout(); save(fig, "fig-dropping.svg")

# ------------------------------------------------------------ 4. precision and recall with k-NN balls (sketch)
rng = np.random.default_rng(3)
real = np.concatenate([rng.normal([-1.2, 0], 0.35, (25, 2)), rng.normal([1.2, 0], 0.35, (25, 2))])
fake = np.concatenate([rng.normal([1.25, 0.05], 0.3, (22, 2)), rng.normal([0.0, 1.3], 0.2, (6, 2))])
def radii(X, k=3):
    d = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)); d.sort(1); return d[:, k]
rr, rf = radii(real), radii(fake)
inside = lambda P, Q, r: np.array([np.any(np.sqrt(((Q - p) ** 2).sum(1)) <= r) for p in P])
prec = inside(fake, real, rr).mean(); rec = inside(real, fake, rf).mean()
fig, axs = plt.subplots(1, 2, figsize=(820 / 96, 330 / 96))
for ax, centres, rad, col, title in [(axs[0], real, rr, GREY, f"precision {prec:.2f}: generated points\ninside the balls of the real ones"),
                                      (axs[1], fake, rf, BLUE, f"recall {rec:.2f}: real points\ninside the balls of the generated ones")]:
    for c, r in zip(centres, rad):
        ax.add_patch(plt.Circle(c, r, color=col, alpha=0.12, lw=0))
    ax.scatter(real[:, 0], real[:, 1], s=22, color="#555", label="real")
    ax.scatter(fake[:, 0], fake[:, 1], s=22, color=BLUE, marker="s", label="generated")
    ax.set_aspect("equal"); ax.set_xlim(-2.4, 2.4); ax.set_ylim(-1.3, 1.9); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=13)
axs[0].legend(fontsize=12, loc="upper left", frameon=False)
fig.tight_layout(); save(fig, "fig-pr.svg")
print("wrote gan-training figures; sketch precision", prec, "recall", rec)

# ------------------------------------------------------------ 5. a game rotates: min over x, max over y of x·y
fig, ax = plt.subplots(figsize=(400 / 96, 360 / 96))
eta = 0.2
for mode, colr, lab in [("sim", RED, "simultaneous steps"), ("alt", BLUE, "alternating steps")]:
    x, y = 1.0, 0.0; P = [(x, y)]
    for _ in range(60):
        if mode == "sim": x, y = x - eta * y, y + eta * x
        else: x = x - eta * y; y = y + eta * x
        P.append((x, y))
    P = np.array(P); ax.plot(P[:, 0], P[:, 1], "-o", ms=2.5, lw=1.5, color=colr, label=lab)
ax.scatter([0], [0], marker="*", s=200, color=INK, zorder=4); ax.text(0.15, -0.5, "equilibrium", fontsize=11)
ax.set_xlabel("x (minimiser)"); ax.set_ylabel("y (maximiser)"); ax.set_aspect("equal")
ax.legend(fontsize=11, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.15)); ax.set_xlim(-3.4, 3.4); ax.set_ylim(-3.4, 3.4)
fig.tight_layout(); save(fig, "fig-spiral.svg")
print("radius after 60 simultaneous steps", (1 + eta ** 2) ** 30)
