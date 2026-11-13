"""Slide figures of the diffusion/ lecture (modern deep learning day).

Run from the repository root:
  node_modules/.bin/esbuild tools/figs/diffusion/lec_data.ts --bundle --platform=node --log-level=warning | node > /tmp/lec_data.json
  python3 -I tools/figs/diffusion/lec_figs.py /tmp/lec_data.json [shots_dir]

The point clouds come from the diffusion2d demo's own model code (lec_data.ts, same seeds as
the demo). The genrestore panels are crops of demo screenshots taken with
tools/figs/instances/demo_shot.sh (scale factor 2) into shots_dir; skipped if not given.
The micrograph is demos/src/fourier2d/cell256.png (Robert M. Hunt, CC BY 3.0, Wikimedia Commons).
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = "diffusion/"
BLUE, ORANGE, RED, GREY = "#1f77b4", "#ff7f0e", "#d62728", "#9a9a9a"
plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "font.family": "DejaVu Sans"})

d = json.load(open(sys.argv[1]))
shots = sys.argv[2] if len(sys.argv) > 2 else None


def pts(X):
    X = np.asarray(X)
    return X[0::2], X[1::2]


def cloud(ax, run, lim=2.6, title=None, show_train=True):
    if show_train:
        tx, ty = pts(run["train"])
        ax.scatter(tx, ty, s=7, c=GREY, alpha=0.6, lw=0, label="training data")
    x, y = pts(run["X"])
    ax.scatter(x, y, s=9, c=BLUE, alpha=0.8, lw=0, label="samples")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
    if title:
        ax.set_title(title, fontsize=16)


# 1. forward process: the micrograph and the two moons at increasing τ ------------------------
cell = np.asarray(Image.open("demos/src/fourier2d/cell256.png").convert("L"), float) / 255
cell = 2 * cell - 1                                   # data scaled to [-1, 1]
rng = np.random.default_rng(3)
eps = rng.standard_normal(cell.shape)
taus = ["0", "0.2", "0.4", "0.6", "1"]
fig, axs = plt.subplots(2, 5, figsize=(8.5, 4.6), gridspec_kw={"hspace": 0.06, "wspace": 0.08})
for k, t in enumerate(taus):
    ab = d["forward"][t]["ab"]
    s, n = np.sqrt(ab), np.sqrt(1 - ab)
    img = s * cell + n * eps
    axs[0, k].imshow(img, cmap="gray", vmin=-2, vmax=2)
    axs[0, k].set_xticks([]); axs[0, k].set_yticks([])
    for sp in axs[0, k].spines.values():
        sp.set_visible(False)
    axs[0, k].set_title(f"τ = {t}\nsignal {s:.2f}\nnoise {n:.2f}", fontsize=16)
    x, y = pts(d["forward"][t]["X"])
    a = axs[1, k]
    a.scatter(x, y, s=4, c=BLUE, alpha=0.6, lw=0)
    a.set_xlim(-3, 3); a.set_ylim(-3, 3); a.set_aspect("equal")
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_color("#888")
fig.savefig(OUT + "fig-forward.png", dpi=110, bbox_inches="tight", facecolor="white")
plt.close(fig)

# 2. schedules ----------------------------------------------------------------------------------
t = np.array(d["sched"]["t"])
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(t, np.sqrt(d["sched"]["cosine"]), c=BLUE, lw=2.5, label="cosine")
ax.plot(t, np.sqrt(d["sched"]["linear"]), c=ORANGE, lw=2.5, ls="--", label="linear")
ax.set_xlabel("time τ"); ax.set_ylabel(r"signal $\sqrt{\bar\alpha}$")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(frameon=False)
ax.grid(alpha=0.3)
fig.savefig(OUT + "fig-schedule.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)

# 3. best denoiser for ±1 data --------------------------------------------------------------------
x = np.linspace(-3, 3, 400)
fig, ax = plt.subplots(figsize=(5.6, 4.2))
for s, c, ls in [(0.95, BLUE, "-"), (0.6, ORANGE, "--"), (0.3, RED, ":")]:
    v = 1 - s * s
    ax.plot(x, np.tanh(s * x / v), c=c, lw=2.5, ls=ls, label=f"signal {s}, noise {np.sqrt(v):.2f}")
xt = 0.5
y = np.tanh(0.6 * xt / 0.64)
ax.plot([xt], [y], "o", c="black", ms=8, zorder=5)
ax.annotate(f"$x_\\tau$ = 0.5 → {y:.2f}", (xt, y), xytext=(-2.8, 0.75), fontsize=15,
            arrowprops=dict(arrowstyle="->", color="black"))
ax.axhline(1, c=GREY, lw=1, ls="-"); ax.axhline(-1, c=GREY, lw=1, ls="-")
ax.set_xlabel(r"noisy value $x_\tau$"); ax.set_ylabel(r"best guess of $x_0$")
ax.set_ylim(-1.15, 1.15); ax.set_xlim(-3, 3)
ax.legend(frameon=False, fontsize=14, loc="lower right")
fig.savefig(OUT + "fig-denoiser.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)

# 4. few steps against many --------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.6), sharey=True)
for a, key, lab in [(axs[0], "few", "2 steps"), (axs[1], "many", "100 steps")]:
    r = d[key]
    cloud(a, r, title=f"{lab}: {100 * r['summary']['off']:.0f} % off data")
axs[0].legend(loc="upper right", frameon=False, fontsize=14, markerscale=2.5, handletextpad=0.1, borderaxespad=0)
fig.savefig(OUT + "fig-steps.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)

# 5. memorising: perfect score of 20 points; learned score drops the rare blob ---------------------
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.6))
r = d["copy"]
tx, ty = pts(r["mu"])
x0, y0 = pts(r["X"])
axs[0].scatter(x0, y0, s=40, c=BLUE, alpha=0.25, lw=0, label="samples")
axs[0].scatter(tx, ty, s=60, facecolors="none", edgecolors="black", lw=1.4, label="20 training points")
axs[0].set_xlim(-2.6, 2.6); axs[0].set_ylim(-2.6, 2.6); axs[0].set_aspect("equal")
axs[0].set_xticks([-2, 0, 2]); axs[0].set_yticks([-2, 0, 2])
axs[0].set_title(f"perfect: {100 * r['summary']['nearTrain']:.0f} % copies", fontsize=16)
axs[0].legend(loc="lower left", frameon=False, fontsize=14, handletextpad=0.1, borderaxespad=0)
r = d["rare"]
cloud(axs[1], r, title=f"learned: rare blob {100 * r['summary']['perGroup'][1]:.0f} %", lim=2.6)
axs[1].annotate("5 % of\nthe data", (1.6, 1.0), xytext=(-2.4, 1.6), fontsize=15,
                arrowprops=dict(arrowstyle="->", color="black"))
fig.savefig(OUT + "fig-memorise.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)

# 6. guidance on the ring of 8 -------------------------------------------------------------------
def counts(r):
    X = np.asarray(r["X"]).reshape(-1, 2)
    mu = np.asarray(r["mu"]).reshape(-1, 2)
    b = np.argmin(((X[:, None, :] - mu[None]) ** 2).sum(-1), axis=1)
    return np.bincount(b, minlength=8)


fig, axs = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
for a, key, lab in [(axs[0], "guide0", "upper half, w = 0"), (axs[1], "guide", "upper half, w = 4")]:
    r = d[key]
    a.axhspan(0, 2.6, color=ORANGE, alpha=0.08, lw=0)
    cloud(a, r, title=lab)
    mu = np.asarray(r["mu"]).reshape(-1, 2)
    for j, c in enumerate(counts(r)):
        ang = np.arctan2(mu[j, 1], mu[j, 0])
        a.text(2.25 * np.cos(ang), 2.25 * np.sin(ang), str(c), ha="center", va="center", fontsize=16)
    print(key, counts(r))
fig.savefig(OUT + "fig-guidance.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)

# 7., 8. genrestore panels: cropped from demo screenshots (device scale 2), relabelled -------------
# Panel boxes in CSS pixels of the demo at 1220 px window width.
BOX = {"truth": (10, 46, 212, 248), "measurement": (219, 46, 420, 247), "classical": (428, 46, 628, 247),
       "mean": (636, 46, 837, 247), "s1": (10, 304, 109, 403), "s2": (111, 304, 211, 403),
       "s3": (10, 405, 109, 505), "s4": (111, 405, 211, 505), "sd": (219, 304, 420, 505)}
if shots:
    def panel(name, key):
        im = Image.open(f"{shots}/{name}.png").convert("L")
        b = BOX[key]
        return np.asarray(im.crop((2 * b[0] + 4, 2 * b[1] + 4, 2 * b[2] - 4, 2 * b[3] - 4)))

    def grid(name, cells, ncol, out, size):
        nrow = (len(cells) + ncol - 1) // ncol
        fig, axs = plt.subplots(nrow, ncol, figsize=size, squeeze=False,
                                gridspec_kw={"hspace": 0.42, "wspace": 0.06})
        for a in axs.flat:
            a.axis("off")
        for a, (key, title, sub) in zip(axs.flat, cells):
            a.imshow(panel(name, key), cmap="gray", vmin=0, vmax=255)
            a.set_title(title, fontsize=17)
            a.text(0.5, -0.04, sub, transform=a.transAxes, ha="center", va="top", fontsize=16)
        fig.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # numbers: the demo's readouts in these states
    grid("gr_heavy", [("truth", "truth", "2 nucleoli"), ("measurement", "measured", "σ 3.5"),
                      ("classical", "classical", "25.1 dB"), ("mean", "post. mean", "0 ncl., 30.5 dB"),
                      ("s1", "sample 1", "1 nucleolus"), ("s2", "sample 2", "1 nucleolus"),
                      ("s3", "sample 3", "2 nucleoli"), ("sd", "disagreement", "")],
         4, OUT + "fig-heavyblur.png", (8.4, 5.2))
    grid("gr_ring", [("truth", "truth: a ring", ""), ("measurement", "measured", "σ 2"),
                     ("classical", "classical", "ring kept"), ("mean", "posterior mean", "filled"),
                     ("s1", "sample 1", "filled"), ("s2", "sample 2", "filled")],
         3, OUT + "fig-ring.png", (6.6, 5.0))
