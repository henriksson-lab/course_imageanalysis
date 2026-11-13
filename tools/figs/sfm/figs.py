"""Slide figures for sfm/lec.qmd (fig-sparsity.svg comes from tools/figs/multiview/figs.py).

Run from the repository root (the flatland data come from the demos' own code):
    node_modules/.bin/esbuild tools/figs/sfm/compute.ts --bundle --platform=node --outfile=$TMP/f.js && node $TMP/f.js $TMP
    python3 -I tools/figs/sfm/figs.py $TMP
Writes fig-*.svg into sfm/. All data synthetic.
"""
import math
import os
import sys
import numpy as np
from matplotlib.patches import Polygon, Ellipse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../multiview"))
from common import plt, px, save, load, BLUE, ORANGE, GREEN, RED, PURPLE, GREY, CAMS  # noqa: E402

TMP = sys.argv[1] if len(sys.argv) > 1 else "."
U = "sfm"
D = load(TMP, "sfm.json")


def cam(ax, c, theta, col, size=0.42):
    d = np.array([math.cos(theta), math.sin(theta)]); n = np.array([-d[1], d[0]])
    c = np.array(c)
    a, b = c + size * (d + 0.6 * n), c + size * (d - 0.6 * n)
    ax.add_patch(Polygon([c, a, b], closed=True, fc=col, ec=col, alpha=0.3, lw=0, zorder=4))
    ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=3.5, zorder=5, solid_capstyle="butt")
    ax.plot([c[0], a[0]], [c[1], a[1]], color=col, lw=1.2, zorder=5)
    ax.plot([c[0], b[0]], [c[1], b[1]], color=col, lw=1.2, zorder=5)


def scene(ax, st, outline=None, ray_alpha=0.35, pts_col="#111"):
    if outline is not None:
        ax.add_patch(Polygon(outline, closed=True, fc="#eee", ec="#aaa", lw=1.2, zorder=0))
    for x0, y0, x1, y1, j in st["rays"]:
        ax.plot([x0, x1], [y0, y1], color=CAMS[j % len(CAMS)], lw=0.9, alpha=ray_alpha, zorder=1)
    for j, c in enumerate(st["cams"]):
        cam(ax, (c["x"], c["y"]), c["theta"], CAMS[j % len(CAMS)])
    P = np.array(st["pts"])
    ax.plot(P[:, 0], P[:, 1], "o", color=pts_col, ms=4, zorder=6)
    ax.set_aspect("equal"); ax.axis("off")


outline = np.array(D["truth"]["outline"])

# 1. the problem: 10 cameras around an object, every observation a ray -----------------------------
fig, ax = plt.subplots(figsize=px(440, 440))
scene(ax, D["truth"], outline)
ax.set_xlim(-3.7, 3.7); ax.set_ylim(-3.7, 3.7)
save(fig, U, "fig-problem.svg")

# 2. bundle adjustment: the perturbed start, rays miss the points -----------------------------------
fig, axs = plt.subplots(1, 2, figsize=px(560, 330))
scene(axs[0], D["start"], outline)
axs[0].set_title(f"start:\nRMS {D['largeHist']['LM'][0]:.0f} px", fontsize=20)
scene(axs[1], D["lmFinal"], outline)
axs[1].set_title(f"after LM:\nRMS {D['largeHist']['LM'][-1]:.2f} px", fontsize=20)
for ax in axs:
    ax.set_xlim(-3.9, 3.9); ax.set_ylim(-3.9, 3.9)
fig.tight_layout(w_pad=0.5)
save(fig, U, "fig-ba-start.svg")

# 3. Gauss–Newton by hand, and GN vs LM on the large perturbation -----------------------------------
fig, axs = plt.subplots(1, 2, figsize=px(1060, 290))
ax = axs[0]
Z = np.linspace(2.4, 5.2, 200)
ax.plot(Z, 0.25 - 1 / Z, color=BLUE, lw=2.5, label="r(Z) = 0.25 − 1/Z")
ax.axhline(0, color=GREY, lw=1)
z = 3.0
for k in range(2):
    r, sl = 0.25 - 1 / z, 1 / z ** 2
    zn = z - r / sl
    ax.plot([z, zn], [r, 0], color=ORANGE, lw=2, ls="--")
    ax.plot([z], [r], "o", color=ORANGE, ms=9)
    ax.plot([zn, zn], [0, 0.25 - 1 / zn], color=GREY, lw=1, ls=":")
    z = zn
ax.plot([4], [0], "*", color=RED, ms=16)
ax.text(3.0, -0.098, "Z₀ = 3", ha="center", va="top", fontsize=19, color=ORANGE)
ax.text(3.75, 0.012, "3.75", ha="center", va="bottom", fontsize=19, color=ORANGE)
ax.text(4.12, -0.025, "3.98 → 4", ha="left", va="top", fontsize=19, color=RED)
ax.set_xlabel("depth Z"); ax.set_ylabel("residual r")
ax.set_ylim(-0.13, 0.07)
ax = axs[1]
H = D["largeHist"]
ax.plot(range(len(H["GN"])), H["GN"], "-o", color=RED, lw=2.5, label="Gauss–Newton")
ax.plot(range(len(H["LM"])), H["LM"], "-s", color=BLUE, lw=2.5, label="Levenberg–Marquardt")
ax.axhline(0.2, color=GREY, ls="--", lw=1.5)
ax.text(14, 0.23, "noise σ = 0.2 px", color=GREY, fontsize=18, ha="right", va="bottom")
ax.set_xticks([0, 5, 10]); ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("RMS error (px)")
ax.set_ylim(0.1, 3000)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(w_pad=2)
save(fig, U, "fig-gn-lm.svg")

# 4. gauge freedom: the reconstruction moved, turned and scaled gives the same images ----------------
P = np.array(D["truth"]["pts"]); cams = D["truth"]["cams"]
s_, a_, t_ = 0.6, math.radians(35), np.array([1.4, -0.9])
Rm = np.array([[math.cos(a_), -math.sin(a_)], [math.sin(a_), math.cos(a_)]])
tf = lambda p: s_ * (Rm @ np.asarray(p).T).T + t_  # noqa: E731
fig, ax = plt.subplots(figsize=px(440, 520))
for (pts, cs, col, alpha) in ((P, cams, BLUE, 1.0), (tf(P), None, ORANGE, 1.0)):
    ax.plot(pts[:, 0], pts[:, 1], "o", color=col, ms=4)
for c in cams:
    cam(ax, (c["x"], c["y"]), c["theta"], BLUE, 0.42)
    q = tf(np.array([[c["x"], c["y"]]]))[0]
    cam(ax, q, c["theta"] + a_, ORANGE, 0.42 * s_)
ax.text(-3.6, 5.6, "a reconstruction", color=BLUE, fontsize=20, va="top")
ax.text(-3.6, 5.0, "moved, turned and scaled:\nevery image stays the same", color=ORANGE, fontsize=20, va="top")
ax.set_xlim(-3.7, 3.7); ax.set_ylim(-3.7, 5.7); ax.set_aspect("equal"); ax.axis("off")
save(fig, U, "fig-gauge.svg")

# 5. squared vs Huber loss (δ = 1 px, as in the demo) -------------------------------------------------
r = np.linspace(-6, 6, 400)
fig, ax = plt.subplots(figsize=px(440, 360))
ax.plot(r, r ** 2, color=RED, lw=2.5, label="squared: r²")
hub = np.where(np.abs(r) <= 1, r ** 2, 2 * np.abs(r) - 1)
ax.plot(r, hub, color=BLUE, lw=2.5, label="Huber (δ = 1 px)")
ax.set_xlabel("residual r (px)"); ax.set_ylabel("cost ρ")
ax.set_ylim(0, 20)
ax.set_title("a 20 px residual: 400 squared, 39 Huber", fontsize=19)
ax.legend(frameon=False, loc="upper center")
save(fig, U, "fig-robust.svg")


# 6. uncertainty ellipses: cameras all round vs on one side (2σ, × 10) ---------------------------------
def ellipses(ax, st, cov, mag, col=PURPLE):
    for p, c in zip(st["pts"], cov):
        if c is None:
            continue
        C = np.array([[c[0], c[1]], [c[1], c[2]]]) * mag ** 2
        w, V = np.linalg.eigh(C)
        ang = math.degrees(math.atan2(V[1, 1], V[0, 1]))
        ax.add_patch(Ellipse(p, 2 * 2 * math.sqrt(w[1]), 2 * 2 * math.sqrt(w[0]), angle=ang, fc="none", ec=col, lw=1.3, zorder=7))


cr = D["covRing"]; far = D["far"]
big = lambda cov: max(math.sqrt(max(np.linalg.eigvalsh(np.array([[c[0], c[1]], [c[1], c[2]]])))) for c in cov if c)  # noqa: E731
print("largest σ: ring", big(cr), "far", big(far["cov"]))
MAG = 2
fig, axs = plt.subplots(1, 2, figsize=px(560, 340))
st = dict(D["truth"]); st["rays"] = []
scene(axs[0], st, outline)
ellipses(axs[0], D["truth"], cr, MAG)
axs[0].set_title("all round", fontsize=20)
axs[0].set_xlim(-3.6, 3.6); axs[0].set_ylim(-3.6, 3.6)
stf = dict(far); stf["rays"] = []
scene(axs[1], stf, outline)
ellipses(axs[1], far, far["cov"], MAG)
axs[1].set_title("on one side", fontsize=20)
axs[1].set_xlim(-4.6, 4.6); axs[1].set_ylim(-4.8, 3.2)
fig.text(0.5, 0.0, f"2σ ellipses × {MAG}", ha="center", va="top", color=PURPLE, fontsize=19)
fig.tight_layout(w_pad=0.5)
save(fig, U, "fig-ellipses.svg")

# 7. plane sweep: photo-consistency against depth for one pixel ----------------------------------------
fig, ax = plt.subplots(figsize=px(440, 400))
for key, col, lab in (("sweep_textured", BLUE, "textured (pixel 60)"), ("sweep_textureless", ORANGE, "textureless (pixel 66)")):
    sw = D[key]
    c = np.array([np.nan if v is None else v for v in sw["cost"]])
    ax.plot(sw["depths"], c, color=col, lw=2.5, label=lab)
    ax.axvline(sw["truthU"], color=col, ls="--", lw=1.5)
    k = int(np.nanargmin(c))
    ax.plot([sw["depths"][k]], [c[k]], "o", color=col, ms=9)
ax.set_xlabel("depth hypothesis Z"); ax.set_ylabel("variance across views")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax.text(0.98, 0.02, "dashed: true depth", transform=ax.transAxes, ha="right", va="bottom", fontsize=18, color="#444")
save(fig, U, "fig-sweep.svg")
