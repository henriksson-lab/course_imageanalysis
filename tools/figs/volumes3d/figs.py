"""Figures for volumes3d/lec.qmd. Run from the repo root (data.ts first, see there):
   python3 -I tools/figs/volumes3d/figs.py <tmp>/vol-data.json
Figure size in inches = display size / 100 px, fonts >= 13 pt (about 18 px on the slide)."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Circle, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "volumes3d")
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
ORANGE, BLUE, GREEN, RED, PURPLE, GREY = "#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, facecolor="white")
    if os.environ.get("FIGPREVIEW"):   # optional PNG previews for checking
        fig.savefig(os.path.join(os.environ["FIGPREVIEW"], name.replace(".svg", ".png")), facecolor="white", dpi=100)
    plt.close(fig)
    print("wrote", p)


def arrow(ax, a, b, color, lw=2.2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=18, color=color, lw=lw, shrinkA=0, shrinkB=0))


# 1. voxel -> world: slice k = 1 seen from above, voxel centres at 10 + 0.5 i, 20 + 0.5 j
def fig_voxel2world():
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for i in range(6):
        for j in range(5):
            x, y = 10 + 0.5 * i, 20 + 0.5 * j
            ax.add_patch(Rectangle((x - 0.25, y - 0.25), 0.5, 0.5, fc="#f4f4f4", ec="#bbb", lw=1))
            ax.plot(x, y, "o", color=GREY, ms=4)
    ax.plot(10, 20, "o", color="black", ms=9)
    ax.text(9.8, 19.6, "origin (10, 20) µm", fontsize=15, va="bottom")
    arrow(ax, (10, 20), (11, 20), BLUE)
    ax.text(10.5, 19.88, "2 × 0.5", color=BLUE, ha="center", va="bottom", fontsize=15)
    arrow(ax, (11, 20), (11, 21.5), GREEN)
    ax.text(11.06, 20.75, "3 × 0.5", color=GREEN, ha="left", va="center", fontsize=15)
    ax.plot(11, 21.5, "s", color=ORANGE, ms=13, mec="black")
    ax.annotate("voxel (2, 3, 1) → (11, 21.5, 7) µm", (11, 21.62), (9.6, 22.55), fontsize=15, va="center",
                arrowprops=dict(arrowstyle="-", color="#333", lw=1))
    ax.plot(12.125, 21.75, "x", color=RED, ms=13, mew=3)
    ax.annotate("(12.125, 21.75, 8.5) µm → voxel (4.25, 3.5, 1.75)", (12.125, 21.85), (13.2, 23.05), color=RED,
                ha="right", fontsize=15, va="center", arrowprops=dict(arrowstyle="-", color=RED, lw=1))
    ax.set_xlim(9.5, 13.2); ax.set_ylim(23.3, 19.45)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)       (slice k = 1: z = 5 + 1 · 2 = 7 µm)"); ax.set_ylabel("y (µm)")
    ax.set_xticks([10, 11, 12, 13]); ax.set_yticks([20, 21, 22])
    fig.tight_layout()
    save(fig, "fig-voxel2world.svg")


# 2. trilinear interpolation in a cube: x, then y, then z (values from the slide)
def fig_trilinear():
    W, H, bx, by = 3.0, 2.4, 1.6, 1.1    # screen: x right, y down, z back (up and right)
    P = lambda x, y, z: (x * W + z * bx, y * H - z * by)
    vals = {(0, 0, 0): 10, (1, 0, 0): 20, (0, 1, 0): 30, (1, 1, 0): 60,
            (0, 0, 1): 50, (1, 0, 1): 50, (0, 1, 1): 90, (1, 1, 1): 70}
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    for a in vals:
        for k in range(3):
            if a[k] == 0:
                b = list(a); b[k] = 1; b = tuple(b)
                pa, pb = P(*a), P(*b)
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#555", lw=1.5, zorder=1)
    for (x, y, z), v in vals.items():
        px, py = P(x, y, z)
        ax.plot(px, py, "o", color="#333", ms=7)
        ax.text(px + (-0.15 if x == 0 else 0.15), py + (-0.2 if y == 0 else 0.2), str(v), fontsize=15,
                ha="right" if x == 0 else "left", va="bottom" if y == 0 else "top", fontweight="bold")
    fx, fy, fz = 0.25, 0.5, 0.75
    xs = {}
    for y in (0, 1):
        for z in (0, 1):
            v = vals[(0, y, z)] + fx * (vals[(1, y, z)] - vals[(0, y, z)])
            xs[(y, z)] = v
            px, py = P(fx, y, z)
            ax.plot(px, py, "o", color=BLUE, ms=9, zorder=3)
            ax.text(px + 0.12, py + (-0.12 if y == 0 else 0.12), f"{v:g}", color=BLUE, ha="left",
                    va="bottom" if y == 0 else "top", fontsize=16)
    ys = {}
    for z in (0, 1):
        a, b = P(fx, 0, z), P(fx, 1, z)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=1.5, ls="--", zorder=2)
        v = xs[(0, z)] + fy * (xs[(1, z)] - xs[(0, z)])
        ys[z] = v
        px, py = P(fx, fy, z)
        ax.plot(px, py, "D", color=GREEN, ms=9, zorder=3)
        if z == 0:
            ax.text(px - 0.15, py, f"{v:g}", color=GREEN, ha="right", va="center", fontsize=16)
        else:
            ax.text(px + 0.1, py + 0.12, f"{v:g}", color=GREEN, ha="left", va="top", fontsize=16)
    a, b = P(fx, fy, 0), P(fx, fy, 1)
    ax.plot([a[0], b[0]], [a[1], b[1]], color=GREEN, lw=1.5, ls="--", zorder=2)
    v = ys[0] + fz * (ys[1] - ys[0])
    assert abs(v - 56.875) < 1e-9
    px, py = P(fx, fy, fz)
    ax.plot(px, py, "*", color=RED, ms=22, mec="black", zorder=4)
    ax.annotate(f"{v:g}", (px, py), (px + 1.5, py + 0.75), color=RED, fontsize=16, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="-", color=RED, lw=1))
    ax.text(-0.3, 3.45, "1. along x: $f_x$ = 0.25 (blue)", color=BLUE, fontsize=16, va="top")
    ax.text(-0.3, 4.05, "2. along y: $f_y$ = 0.5 (green)", color=GREEN, fontsize=16, va="top")
    ax.text(-0.3, 4.65, "3. along z: $f_z$ = 0.75 (red)", color=RED, fontsize=16, va="top")
    ax.text(W + bx + 0.1, -by - 0.6, "z (back)", fontsize=15, color="#555", ha="left")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.0, 5.4); ax.set_ylim(5.2, -1.8)
    fig.tight_layout()
    save(fig, "fig-trilinear.svg")


# 3. xz slice through the tilted cell C, and the oblique slice along it (volview "oblique" preset)
def fig_oblique(d):
    xz = np.array(d["xz"]["d"], float).reshape(d["xz"]["h"], d["xz"]["w"])
    ob = np.array([np.nan if v is None else v for v in d["ob"]["d"]], float).reshape(d["ob"]["h"], d["ob"]["w"])
    sx, sy, sz = d["spacing"]
    plt.rcParams.update({"font.size": 16})
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 3.3), gridspec_kw={"width_ratios": [12.8, 14.08]})
    a = axs[0]
    a.imshow(xz, cmap="gray", vmin=0, vmax=210, extent=(-sx / 2, xz.shape[1] * sx - sx / 2, xz.shape[0] * sz - sz / 2, -sz / 2),
             interpolation="nearest")
    (x0, z0), (x1, z1) = d["ends"]
    a.plot([x0, x1], [z0, z1], color=ORANGE, lw=2.5, ls="--")
    a.text(12.4, 0.4, "plane", color=ORANGE, fontsize=15, ha="right", va="top")
    a.set_xlabel("x (µm)"); a.set_ylabel("z (µm)")
    a.set_title("xz slice", fontsize=15)
    b = axs[1]
    L = ob.shape[1] * d["ob"]["step"]
    cm = plt.get_cmap("gray").copy(); cm.set_bad("#e8e8e8")
    b.imshow(np.ma.masked_invalid(ob), cmap=cm, vmin=0, vmax=210, extent=(-L / 2, L / 2, L / 2, -L / 2))
    b.text(0, 4.7, "outside the stack", color="#555", ha="center", va="center", fontsize=15)
    b.annotate("", (-3.6, 1.6), (3.6, 1.6), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=2.5))
    b.text(0, 2.1, "7.2 µm", color=ORANGE, ha="center", va="top", fontsize=15)
    b.set_xlabel("in the plane (µm)"); b.set_ylabel("y (µm)")
    b.set_title("oblique, −35°", fontsize=15)
    for s in ("top", "right"):
        a.spines[s].set_visible(True); b.spines[s].set_visible(True)
    fig.tight_layout()
    save(fig, "fig-oblique.svg")
    plt.rcParams.update({"font.size": 16})


# 4. one Gaussian, two meanings: sigma in µm vs sigma in voxels, drawn in the xz plane (µm)
def fig_aniso_gauss():
    dx, dz = 0.2, 0.5
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for i in range(-7, 11):
        for k in range(-3, 3):
            ax.add_patch(Rectangle((i * dx - dx / 2, k * dz - dz / 2), dx, dz, fc="none", ec="#ccc", lw=0.8))
            ax.plot(i * dx, k * dz, ".", color="#aaa", ms=3)
    ax.add_patch(Ellipse((0, 0), 2 * 0.4, 2 * 0.4, fc=BLUE, alpha=0.18, ec=BLUE, lw=2.5))
    ax.add_patch(Ellipse((0, 0), 2 * 2 * dx, 2 * 2 * dz, fc="none", ec=RED, lw=2.5, ls="--"))
    ax.text(0.5, -0.05, "σ = 0.4 µm:\n2 vox in x,\n0.8 vox in z", color=BLUE, fontsize=15, va="center",
            bbox=dict(fc="white", ec="none", pad=1))
    ax.text(0.5, -1.05, "σ = 2 vox per axis:\n1 µm along z", color=RED, fontsize=15, va="center",
            bbox=dict(fc="white", ec="none", pad=1))
    ax.set_xlim(-1.3, 2.1); ax.set_ylim(1.4, -1.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm), Δx = 0.2"); ax.set_ylabel("z (µm), Δz = 0.5")
    fig.tight_layout()
    save(fig, "fig-aniso-gauss.svg")


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    fig_voxel2world(); fig_trilinear(); fig_oblique(data); fig_aniso_gauss()
