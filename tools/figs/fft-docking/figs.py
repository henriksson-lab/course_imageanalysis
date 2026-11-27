"""Static figures for fft-docking/ (docking by correlation).

Run from the repository root:  python3 -I tools/figs/fft-docking/figs.py
Writes fft-docking/fig-*.svg|png. The 2D scenes, score maps and searches come from the dock2d
demo's own model (tools/figs/fft-docking/data.ts, run with node), so scores and poses match the
demo. The shapes are synthetic stand-ins for two molecules. The 3D figures are schematic.
"""
import json, os, subprocess, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch, Circle

plt.rcParams.update({"font.size": 16, "axes.titlesize": 16, "svg.fonttype": "none",
                     "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, RED, GREEN, GREY = "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#888888"
CORE, SURF, EMPTY = "#4a6ca8", "#f5aa46", "#fafafa"
LAYERS = ListedColormap([CORE, EMPTY, SURF])     # −1, 0, +1
LIG = "#1a7a1a"
OUT = "fft-docking/"


def demo_data():
    with tempfile.TemporaryDirectory() as d:
        js = os.path.join(d, "dock.json")
        subprocess.run("node_modules/.bin/esbuild tools/figs/fft-docking/data.ts --bundle --platform=node "
                       "--log-level=warning | node - %s" % js, shell=True, check=True)
        return json.load(open(js))


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def layers(ax, d, extent=None):
    n = d["n"]; box = 64.0
    img = np.array(d["layer"]).reshape(n, n)
    ax.imshow(img, cmap=LAYERS, vmin=-1, vmax=1, extent=[0, box, box, 0], interpolation="nearest")
    if extent:
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[3], extent[2])
    off(ax)


def ligand(ax, p, h=1.0, cells=True, outline=LIG, lw=2.5, ls="-"):
    if cells:
        for cx, cy, kind in p["cells"]:
            if kind:
                ax.add_patch(Rectangle((cx * h, cy * h), h, h, color=GREEN if kind > 0 else RED, alpha=0.9, lw=0))
    ax.add_patch(Polygon(p["poly"], closed=True, fill=False, ec=outline, lw=lw, ls=ls))


def save(fig, name):
    fig.savefig(OUT + name, bbox_inches="tight", facecolor="white", dpi=110 if name.endswith(".png") else None)
    plt.close(fig)


# ------------------------------------------------------------ the problem
def problem(D):
    k = D["key"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for a in ax:
        layers(a, k, [4, 60, 4, 60])
    ligand(ax[0], k["apart"], cells=False)
    ax[0].add_patch(Polygon(k["apart"]["poly"], closed=True, color=LIG, alpha=0.25))
    ax[0].add_patch(FancyArrowPatch((k["apart"]["x"] + 4, k["apart"]["y"] + 3), (D["truePose"]["x"] - 2, D["truePose"]["y"] - 4),
                                    arrowstyle="-|>", mutation_scale=22, lw=2.5, color="k", connectionstyle="arc3,rad=-0.2"))
    ax[0].text(44, 14, "shift (x, y)\nand angle θ?", fontsize=16, ha="center")
    ax[0].set_title("receptor and ligand")
    ligand(ax[1], k["truePose"], cells=False)
    ax[1].add_patch(Polygon(k["truePose"]["poly"], closed=True, color=LIG, alpha=0.25))
    ax[1].set_title("docked: large contact, no overlap")
    save(fig, "fig-problem.png")


# ------------------------------------------------------------ the grid score
def grids(D):
    k = D["key"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 5.4))
    ext = [18, 48, 14, 44]
    for a, p, t in [(ax[0], k["truePose"], "true pose"), (ax[1], k["deeper"], "2 cells deeper")]:
        layers(a, k, ext)
        for x in range(ext[0], ext[1] + 1):
            a.axvline(x, color="#00000014", lw=0.6)
        for y in range(ext[2], ext[3] + 1):
            a.axhline(y, color="#00000014", lw=0.6)
        ligand(a, p, lw=2.5)
        a.set_title("%s\n%d contacts, %d clashes\nscore %s" % (t, p["contact"], p["clash"], fmt_score(p["score"])), fontsize=20)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig-grids.png")


def fmt_score(v):
    return ("%d" % round(v)).replace("-", "−")


# ------------------------------------------------------------ 1D by hand
def by_hand():
    rho = -9
    a = [rho, rho, rho, 1, 0, 0, 0, 0]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    def row(y, vals, colors, labels=True):
        for i, v in enumerate(vals):
            ax.add_patch(Rectangle((i, y), 1, 0.8, fc=colors[i], ec="k", lw=1.2))
            if labels:
                ax.text(i + 0.5, y + 0.4, ("%g" % v).replace("-", "−") if v != 0 else "0", ha="center", va="center", fontsize=16,
                        color="white" if colors[i] == CORE else "k")
    cols = [CORE] * 3 + [SURF] + [EMPTY] * 4
    row(4.2, a, cols)
    ax.text(-0.3, 4.6, "receptor a", ha="right", va="center", fontsize=16)
    S = {1: 2 * rho, 2: rho + 1, 3: 1, 4: 0}
    for j, d in enumerate([1, 2, 3, 4]):
        y = 3.0 - j * 0.95
        for i in (d, d + 1):
            kind = a[i]
            c = RED if kind < 0 else GREEN if kind > 0 else "#9fd89f"
            ax.add_patch(Rectangle((i + 0.1, y), 0.8, 0.6, fc=c, ec="k", lw=1))
        ax.text(-0.3, y + 0.3, "ligand at d = %d" % d, ha="right", va="center", fontsize=15)
        ax.text(8.3, y + 0.3, "S(%d) = %s" % (d, ("%d" % S[d]).replace("-", "−")), ha="left", va="center", fontsize=16,
                fontweight="bold" if d == 3 else "normal")
    for i in range(8):
        ax.text(i + 0.5, 5.25, str(i), ha="center", fontsize=13, color=GREY)
    ax.set_xlim(-3.6, 10.6); ax.set_ylim(-0.1, 5.6); off(ax)
    save(fig, "fig-1d.svg")


# ------------------------------------------------------------ score maps
def maps(D):
    k = D["key"]; n = k["n"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.2))
    for a, ang in zip(ax, ["20", "110"]):
        m = np.array(k["maps"][ang]).reshape(n, n)
        hi = max(np.array(k["maps"]["20"]).max(), 1)
        a.imshow(np.clip(m, -hi, hi), cmap="RdBu_r", norm=TwoSlopeNorm(0, -hi, hi), extent=[0, 64, 64, 0])
        iy, ix = np.unravel_index(np.argmax(m), m.shape)
        a.add_patch(Circle((ix + 0.5, iy + 0.5), 2.2, fill=False, ec="#00a8c8", lw=3))
        a.set_title("ligand turned %s°: best %d" % (ang, round(m.max())))
        off(a)
    save(fig, "fig-maps.png")


# ------------------------------------------------------------ the angle loop
def angles(D):
    k = D["key"]; c = D["coarse"]
    A = np.array(k["perAngle"]); C = np.array(c["perAngle"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(A[:, 0], A[:, 1], "-", color=BLUE, lw=2.5, label="5° steps (72 angles)")
    ax.plot(C[:, 0], C[:, 1], "o", color=ORANGE, ms=9, label="30° steps (12 angles)")
    ax.axvline(D["truePose"]["angle"], color=GREEN, lw=2)
    ax.text(D["truePose"]["angle"] + 12, A[:, 1].max() * 0.72, "true fit", color=GREEN, fontsize=15, va="top")
    ax.set_xlabel("ligand angle (°)"); ax.set_ylabel("best score at that angle")
    ax.set_xticks([0, 90, 180, 270, 360]); ax.set_xlim(0, 360); ax.set_ylim(0, A[:, 1].max() * 1.1)
    ax.legend(loc="upper right", fontsize=14, frameon=False)
    save(fig, "fig-angles.svg")


# ------------------------------------------------------------ rho too mild, flexible bump
def mild_and_bump(D):
    m = D["mild"]; b = D["bump"]
    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    layers(ax, m, [14, 50, 12, 48]); ligand(ax, m["best"], lw=2.5)
    ax.set_title("ρ = −0.1: the best pose sinks in\n%d contacts, %d clashes\nscore %.1f" % (m["best"]["contact"], m["best"]["clash"], m["best"]["score"]), fontsize=20)
    save(fig, "fig-mild.png")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.6))
    layers(ax[0], b, [14, 50, 12, 48]); ligand(ax[0], b["truePose"], lw=2.5)
    ax[0].set_title("bump, ρ = −9: true pose\n%d contacts, %d clashes\nscore %s" % (b["truePose"]["contact"], b["truePose"]["clash"], fmt_score(b["truePose"]["score"])), fontsize=20)
    layers(ax[1], b, [4, 60, 16, 72]); ligand(ax[1], b["best"], lw=2.5)
    ax[1].set_title("…so the best pose is here\n%d contacts, %d clashes\nscore %s" % (b["best"]["contact"], b["best"]["clash"], fmt_score(b["best"]["score"])), fontsize=20)
    save(fig, "fig-bump.png")


# ------------------------------------------------------------ two scores in one complex grid
def complex_layers():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    def blob(cx, cy, r, label):
        ax.add_patch(Circle((cx, cy), r, color=SURF, alpha=0.9))
        ax.add_patch(Circle((cx, cy), r - 1.0, color=CORE))
        ax.text(cx, cy, label, ha="center", va="center", color="white", fontsize=15)
    blob(0, 0, 3.2, "receptor\ns + i·c")
    blob(6.6, 0.4, 2.2, "ligand\ns − i·c")
    ax.annotate("surfaces overlap:\n+1 per cell (contact)", (3.6, 0.3), (2.4, 4.2), fontsize=15, ha="center",
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="k"))
    ax.text(3.4, -4.4, "cores overlap: −1 per cell (clash)\nrest: imaginary part, ignored", fontsize=15, ha="center")
    ax.set_xlim(-4, 9.5); ax.set_ylim(-5.6, 5.4); ax.set_aspect("equal"); off(ax)
    save(fig, "fig-complex.svg")


# ------------------------------------------------------------ 3D: same recipe, bigger loop
def blob3d(n=60, seed=3):
    """A lumpy closed surface (receptor) with a pocket, as a parametric mesh."""
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 2 * n), np.linspace(0, np.pi, n))
    r = 1 + 0.12 * np.sin(3 * u) * np.sin(2 * v) + 0.08 * np.cos(5 * v)
    # a pocket near the top
    d = np.arccos(np.clip(np.sin(v) * np.cos(u - 0.6) * np.sin(0.7) + np.cos(v) * np.cos(0.7), -1, 1))
    r -= 0.35 * np.exp(-(d / 0.35) ** 2)
    x, y, z = r * np.sin(v) * np.cos(u), r * np.sin(v) * np.sin(u), r * np.cos(v)
    return x, y, z


def ellipsoid(c, axes, R, n=20):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n // 2 + 1))
    p = np.stack([axes[0] * np.sin(v) * np.cos(u), axes[1] * np.sin(v) * np.sin(u), axes[2] * np.cos(v)], -1) @ R.T
    return p[..., 0] + c[0], p[..., 1] + c[1], p[..., 2] + c[2]


def rot(axis, ang):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def poses3d():
    fig = plt.figure(figsize=(6.4, 5.6))
    ax = fig.add_subplot(projection="3d")
    x, y, z = blob3d()
    ax.plot_surface(x, y, z, color=CORE, alpha=0.55, linewidth=0.2, edgecolor="#2b3f66", shade=True)
    rng = np.random.default_rng(4)
    pocket = np.array([np.sin(0.7) * np.cos(0.6), np.sin(0.7) * np.sin(0.6), np.cos(0.7)])
    for i in range(5):
        c = pocket * (1.25 + 0.5 * i) + rng.normal(0, 0.25, 3) * (i > 0)
        R = rot(rng.normal(size=3), rng.uniform(0, np.pi))
        col = GREEN if i == 0 else "#9aa0a6"
        ax.plot_surface(*ellipsoid(c, (0.45, 0.22, 0.18), R), color=col, alpha=0.9 if i == 0 else 0.55, linewidth=0)
    ax.set_box_aspect((1, 1, 1), zoom=1.45); ax.set_axis_off()
    ax.view_init(elev=22, azim=20)
    lim = 1.9
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-1.3, 2.5)
    save(fig, "fig-3d-poses.png")


def sphere_samples():
    step = np.radians(15)
    n_dirs = int(round(4 * np.pi / step ** 2))           # directions about step apart
    i = np.arange(n_dirs) + 0.5
    phi = np.arccos(1 - 2 * i / n_dirs); th = np.pi * (1 + 5 ** 0.5) * i
    P = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], -1)
    fig = plt.figure(figsize=(5.6, 5.4))
    ax = fig.add_subplot(projection="3d")
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20))
    ax.plot_surface(np.sin(v) * np.cos(u), np.sin(v) * np.sin(u), np.cos(v), color="#dfe6f2", alpha=0.35, linewidth=0)
    front = P @ np.array([np.cos(np.radians(30)), np.sin(np.radians(30)), 0.3]) > -0.1
    ax.scatter(*P[front].T, s=16, color=BLUE, depthshade=False)
    # one direction with its turns
    d = np.array([0.35, 0.55, 0.76]); d /= np.linalg.norm(d)
    ax.quiver(0, 0, 0, *(1.25 * d), color=RED, lw=2.5, arrow_length_ratio=0.12)
    e1 = np.cross(d, [0, 0, 1]); e1 /= np.linalg.norm(e1); e2 = np.cross(d, e1)
    t = np.linspace(0, 2 * np.pi, 100)
    circ = 1.25 * d + 0.25 * (np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2))
    ax.plot(*circ.T, color=RED, lw=2)
    ax.set_box_aspect((1, 1, 1), zoom=1.5); ax.set_axis_off(); ax.view_init(elev=18, azim=30)
    ax.set_title("%d directions × %d turns ≈ %d" % (n_dirs, round(360 / 15), n_dirs * round(360 / 15)), fontsize=15, pad=-6)
    save(fig, "fig-sphere.png")
    return n_dirs


def harmonics():
    fig = plt.figure(figsize=(11, 5.6))
    t = np.linspace(0, 2 * np.pi, 361)
    for m in range(4):
        ax = fig.add_subplot(2, 4, m + 1)
        rr, tt = np.meshgrid(np.linspace(0, 1, 60), t)
        ax.pcolormesh(rr * np.cos(tt), rr * np.sin(tt), np.cos(m * tt), cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
        ax.set_aspect("equal"); off(ax)
        ax.set_title("2D: cos %sθ" % ("" if m == 1 else m) if m else "2D: constant", fontsize=21)
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 90), np.linspace(0, np.pi, 45))
    x, y, z = np.sin(v) * np.cos(u), np.sin(v) * np.sin(u), np.cos(v)
    Ys = [(np.ones_like(z), "3D: degree 0 (1)"), (z, "degree 1 (3)"), (x * x - y * y, "degree 2 (5)"), (x ** 3 - 3 * x * y * y, "degree 3 (7)")]
    cmap = plt.get_cmap("RdBu_r")
    for i, (Y, lab) in enumerate(Ys):
        ax = fig.add_subplot(2, 4, 5 + i, projection="3d")
        mx = np.abs(Y).max()
        ax.plot_surface(x, y, z, facecolors=cmap(0.5 + 0.5 * Y / mx), linewidth=0, shade=False)
        ax.set_box_aspect((1, 1, 1), zoom=1.55); ax.set_axis_off(); ax.view_init(elev=20, azim=35)
        ax.set_title(lab, fontsize=20)
    fig.subplots_adjust(wspace=0.05, hspace=0.15)
    save(fig, "fig-harmonics.png")


if __name__ == "__main__":
    D = demo_data()
    problem(D); grids(D); by_hand(); maps(D); angles(D); mild_and_bump(D); complex_layers()
    poses3d(); nd = sphere_samples(); harmonics()
    k = D["key"]
    print("true pose", k["truePose"]["score"], k["truePose"]["contact"], k["truePose"]["clash"],
          "deeper", k["deeper"]["score"], k["deeper"]["contact"], k["deeper"]["clash"],
          "shifted", k["shifted"]["score"], k["shifted"]["contact"], k["shifted"]["clash"])
    print("top", [(p["angle"], p["x"], p["y"], p["score"], p["isTrue"]) for p in k["top"]], "ffts", k["ffts"])
    print("map maxima", {a: max(v) for a, v in k["maps"].items()})
    print("mild", D["mild"]["best"]["angle"], D["mild"]["best"]["score"], D["mild"]["best"]["contact"], D["mild"]["best"]["clash"], "rank", D["mild"]["rank"])
    print("coarse", D["coarse"]["best"]["angle"], D["coarse"]["best"]["x"], D["coarse"]["best"]["y"], D["coarse"]["best"]["score"])
    b = D["bump"]
    print("bump true", b["truePose"]["score"], b["truePose"]["contact"], b["truePose"]["clash"], "best", b["best"]["angle"], b["best"]["x"], b["best"]["y"], b["best"]["score"], "rank", b["rank"],
          "soft", b["soft"]["best"]["angle"], b["soft"]["best"]["score"], b["soft"]["isTrue"])
    print("sphere directions", nd)
