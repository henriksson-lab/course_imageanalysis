# Figures for shape-features/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/shape-features/figs.py
# Masks and features come from the moments demo's "features" code (data.ts, via node).
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "label-connectivity"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.colors import ListedColormap
from labstyle import setup, fig, save, run_ts, PIX, PIXLIGHT, GRID, INK, BLUE, ORANGE, GREEN, RED, GREY, PURPLE

setup()
OUT = "shape-features"
D = run_ts(HERE)
S = 64
CM = ListedColormap(["white", PIX])


def show(ax, k, lim=None):
    m = np.array(D[k]["mask"]).reshape(S, S)
    ax.imshow(m, cmap=CM, extent=(0, S, S, 0), interpolation="nearest")
    if lim: ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[3], lim[2])
    else: ax.set_xlim(0, S); ax.set_ylim(S, 0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(GRID)


def hull(ax, k, **kw):
    ax.add_patch(Polygon(np.array(D[k]["hull"]), closed=True, fill=False, ec=BLUE, lw=2.5, **kw))


# --- 1. the five shapes of the table --------------------------------------------------------
names = [("disc16", "disc"), ("star", "star"), ("pair", "two touching\ndiscs"), ("rod", "rod"), ("crescent", "crescent")]
f, axs = plt.subplots(1, 5, figsize=(900 / 72, 230 / 72), gridspec_kw=dict(wspace=0.08))
for ax, (k, n) in zip(axs, names):
    show(ax, k, lim=(6, 58, 6, 58))
    ax.set_xlabel(n, fontsize=22)
save(f, f"{OUT}/fig-shapes.svg")

# --- 2. solidity: star and touching pair with their hulls ------------------------------------
f, axs = plt.subplots(1, 2, figsize=(520 / 72, 300 / 72), gridspec_kw=dict(wspace=0.08))
for ax, k, n in [(axs[0], "star", "star"), (axs[1], "pair", "touching pair")]:
    show(ax, k, lim=(4, 60, 4, 60))
    hull(ax, k)
    ax.set_xlabel(f"{n}\nsolidity {D[k]['solidity']:.2f}", fontsize=20)
save(f, f"{OUT}/fig-solidity.svg")

# --- 3. form factor against the disc radius ---------------------------------------------------
ff = D["ff"]; r = np.array(ff["radii"])
f, ax = fig(520, 360)
ax.plot(r, ff["crofton"], color=GREEN, lw=2, label="Crofton")
ax.plot(r, ff["chain"], color=ORANGE, lw=2, label="chain code")
ax.plot(r, ff["edges"], color=PURPLE, lw=2, label="edge count")
ax.axhline(1, color=INK, lw=1.2)
ax.axhline(np.pi ** 2 / 16, color=GREY, lw=1.2, ls="--")
ax.text(40, np.pi ** 2 / 16 + 0.03, r"$\pi^2/16 \approx 0.62$", ha="right", va="bottom", fontsize=17, color="#555")
sm = D["small"]["ffChain"]
ax.plot(3.2, sm, "o", color=ORANGE, ms=9)
ax.annotate(f"demo, r = 3.2: {sm:.2f}", (3.2, sm), xytext=(8, 1.3), fontsize=17, va="center",
            arrowprops=dict(arrowstyle="-", color="#555", lw=1))
ax.set_xlim(2, 40); ax.set_ylim(0.3, 1.4)
ax.set_xlabel("disc radius (pixels)"); ax.set_ylabel(r"form factor $4\pi A/P^2$")
ax.legend(loc="lower right", frameon=False, ncol=3, fontsize=16, bbox_to_anchor=(1.0, -0.02), handlelength=1.2, columnspacing=0.8)
save(f, f"{OUT}/fig-formfactor.svg")

# --- 4. Feret: the rod with calipers, and w(theta) --------------------------------------------
R = D["rod"]
f, axs = plt.subplots(1, 2, figsize=(560 / 72, 270 / 72), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.35))
ax = axs[0]
show(ax, "rod", lim=(8, 56, 8, 56))
hull(ax, "rod")
p = np.array(R["maxPair"]); ax.plot(p[:, 0], p[:, 1], color=RED, lw=3)
a = R["minAngle"]; u = np.array([np.cos(a), np.sin(a)]); nrm = np.array([-u[1], u[0]])
H = np.array(R["hull"]); t = H @ u
for tt in [t.min(), t.max()]:
    c = tt * u; ax.plot([c[0] - 40 * nrm[0], c[0] + 40 * nrm[0]], [c[1] - 40 * nrm[1], c[1] + 40 * nrm[1]], color=ORANGE, lw=2.5, ls="--")
ax = axs[1]
th = np.degrees(R["wTheta"]); w = np.array(R["w"])
ax.plot(th, w, color=INK, lw=2.5)
ax.axhline(R["maxFeret"], color=RED, lw=1.5, ls=":")
ax.axhline(R["minFeret"], color=ORANGE, lw=1.5, ls=":")
ax.text(3, R["maxFeret"] + 1, f"max {R['maxFeret']:.1f}", color=RED, ha="left", va="bottom", fontsize=17)
ax.text(180, R["minFeret"] + 1.5, f"min {R['minFeret']:.1f}", color=ORANGE, ha="right", va="bottom", fontsize=17)
ax.set_xlim(0, 180); ax.set_xticks([0, 90, 180]); ax.set_ylim(0, 54)
ax.set_xlabel(r"direction $\theta$ (°)"); ax.set_ylabel(r"width $w(\theta)$")
save(f, f"{OUT}/fig-feret.svg")

# --- 5. distance transform of the crescent and the inscribed circle ---------------------------
C = D["crescent"]
dt = np.array(C["dt"]).reshape(S, S)
f, ax = fig(400, 360)
im = ax.imshow(np.where(dt > 0, dt, np.nan), cmap="viridis", extent=(0, S, S, 0), interpolation="nearest")
ax.add_patch(Circle(C["inscribedAt"], C["inscribedR"], fill=False, ec=RED, lw=3))
ax.plot(*C["inscribedAt"], "+", color=RED, ms=14, mew=2.5)
ax.set_xlim(6, 54); ax.set_ylim(56, 8); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(GRID)
cb = f.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label("distance to background (px)", fontsize=17)
ax.set_xlabel(f"inscribed radius {C['inscribedR']:.1f}", fontsize=19)
save(f, f"{OUT}/fig-radii.svg")

# --- 6. Fourier descriptors: the star with K coefficients ------------------------------------
F = D["fstar"]
f, axs = plt.subplots(1, 4, figsize=(620 / 72, 200 / 72), gridspec_kw=dict(wspace=0.05))
o = np.array(F["outline"])
for ax, K in zip(axs, ["1", "4", "6", "24"]):
    sm = np.array(F["fourier"][K])
    ax.add_patch(Polygon(o, closed=True, fc=PIX, ec="none"))
    ax.add_patch(Polygon(sm, closed=True, fill=False, ec=RED, lw=2.5))
    ax.set_xlim(2, 62); ax.set_ylim(66, 2); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xlabel(f"$|k| \\leq {K}$", fontsize=20)
save(f, f"{OUT}/fig-fourier.svg")
print({k: (D[k]["solidity"], D[k]["minFeret"], D[k]["maxFeret"], D[k]["inscribedR"]) for k in ["star", "pair", "rod", "crescent"]})
