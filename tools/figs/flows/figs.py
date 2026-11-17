# Figures for the unit flows (flows/lec.qmd). Run (from the repo root) after train.ts and dump.ts
# (see their headers):  python3 -I tools/figs/flows/figs.py
# data.json: the flow2d demo's trained flow (6 coupling layers, two moons, 5 000 steps).
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/flows/'
D = json.load(open(R + '/tools/figs/flows/data.json'))
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, GREY = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c", "#9aa3ab"
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)

# ------------------------------------------------------------ 1. change of variables in 1D
fig, axs = plt.subplots(1, 3, figsize=(980 / 96, 290 / 96), gridspec_kw=dict(width_ratios=[1, 1.1, 1]))
z = np.linspace(-3.5, 3.5, 701); pz = np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
f = lambda z: z + 0.6 * np.tanh(2 * z)         # an increasing map noise -> data
fp = lambda z: 1 + 1.2 / np.cosh(2 * z) ** 2
x = f(z); px = pz / fp(z)
axs[0].fill_between(z, 0, pz, color=GREY, alpha=0.7, lw=0); axs[0].set_title("noise z ~ N(0, 1)", fontsize=13); axs[0].set_xlabel("z"); axs[0].set_ylim(0, 0.45)
axs[1].plot(z, x, color=INK, lw=2.5); axs[1].set_xlabel("z"); axs[1].set_ylabel("x = g(z)"); axs[1].set_title("g stretches near 0 (slope 2.2)", fontsize=13)
for a in [-1.5, -0.5, 0.0, 0.5, 1.5]:
    axs[1].plot([a, a], [-4, f(a)], color=GREY, lw=0.8); axs[1].plot([-3.5, a], [f(a), f(a)], color=GREY, lw=0.8)
axs[1].set_xlim(-3.5, 3.5); axs[1].set_ylim(-4, 4)
axs[2].fill_between(x, 0, px, color=BLUE, alpha=0.5, lw=0); axs[2].set_title("p(x) = p(z) / slope", fontsize=13); axs[2].set_xlabel("x"); axs[2].set_ylim(0, 0.45)
axs[2].text(0, 0.24, "thinned\nwhere g stretches", ha="center", fontsize=11)
fig.tight_layout(); save(fig, "fig-1d.svg")

# ------------------------------------------------------------ 2. an affine coupling layer
def box(ax, x, y, w, h, text, col, fs=14):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=col, ec=INK, lw=1.5))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)
def arrow(ax, a, b, col=INK):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=col, lw=2, mutation_scale=18))
fig, ax = plt.subplots(figsize=(680 / 96, 300 / 96))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")
box(ax, 1.0, 3.3, 1.3, 0.8, "x₁", "#eef2f6"); box(ax, 1.0, 1.0, 1.3, 0.8, "x₂", "#eef2f6")
box(ax, 4.2, 2.2, 2.4, 1.0, "small network\n→ s(x₁), t(x₁)", "#d7e6f2", 13)
box(ax, 7.2, 1.0, 2.6, 0.9, "x₂ · e^s + t", "#f4dede")
box(ax, 9.3, 3.3, 1.1, 0.8, "y₁", "#eef2f6"); box(ax, 9.3, 1.0, 1.1, 0.8, "y₂", "#eef2f6")
arrow(ax, (1.65, 3.3), (8.75, 3.3)); arrow(ax, (1.65, 1.0), (5.9, 1.0)); arrow(ax, (8.5, 1.0), (8.75, 1.0))
arrow(ax, (2.4, 3.3), (3.4, 2.7)); arrow(ax, (5.4, 1.8), (6.4, 1.4))
ax.text(5.0, 3.55, "copied unchanged", fontsize=12, ha="center")
ax.text(5.0, 0.05, "inverse: x₂ = (y₂ − t(y₁)) · e^(−s(y₁));   log |det| = s(x₁)", fontsize=12, ha="center", color=RED)
save(fig, "fig-coupling.svg")

# ------------------------------------------------------------ 3. layer by layer
L = D["layers"]; data = np.array(D["data"])
col = plt.cm.viridis(np.clip((data[:, 0] + 2.2) / 4.4, 0, 1))
fig, axs = plt.subplots(1, 4, figsize=(1000 / 96, 270 / 96))
for ax, k in zip(axs, [0, 2, 4, 6]):
    P = np.array(L[k]); ax.scatter(P[:, 0], P[:, 1], s=6, c=col, lw=0)
    if k == 6:
        t = np.linspace(0, 2 * np.pi, 100)
        for r in [1, 2]: ax.plot(r * np.cos(t), r * np.sin(t), color=GREY, ls="--", lw=1)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("data" if k == 0 else f"after {k} layers", fontsize=13)
fig.tight_layout(); save(fig, "fig-layers.png", dpi=110)

# ------------------------------------------------------------ 4. density and samples
G = 120
dens = np.array(D["density"]).reshape(G, G)
fig, axs = plt.subplots(1, 2, figsize=(760 / 96, 360 / 96))
axs[0].imshow(dens, extent=[-3, 3, -3, 3], cmap="viridis", vmin=0, vmax=0.35)
for p, (dx, dy, ha) in zip(D["probes"][:3], [(0.15, -0.35, "left"), (0.15, 0.15, "left"), (-0.1, 0.35, "center")]):
    axs[0].scatter([p["x"][0]], [p["x"][1]], color=RED, s=40, edgecolor="white", zorder=3)
    axs[0].text(p["x"][0] + dx, p["x"][1] + dy, f"log p = {p['logp']:.2f}", color="white", fontsize=11, ha=ha)
axs[0].set_title("exact density p(x)", fontsize=13)
S = np.array(D["samples"]); axs[1].scatter(data[:, 0], data[:, 1], s=6, color=GREY, lw=0, label="data")
axs[1].scatter(S[:, 0], S[:, 1], s=6, color=BLUE, lw=0, label="samples")
axs[1].legend(fontsize=12, frameon=False, loc="upper left", markerscale=2); axs[1].set_title("samples: noise run backwards", fontsize=13)
for ax in axs: ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
fig.tight_layout(); save(fig, "fig-density.png", dpi=110)
print("wrote flows figures")
