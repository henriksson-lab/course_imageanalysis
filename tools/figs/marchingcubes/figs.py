"""Figures for marchingcubes/lec.qmd. Run from the repo root (data.ts first, see there):
   python3 -I tools/figs/marchingcubes/figs.py <tmp>/mc-data.json
Corner numbering k = x + 2y + 4z as in the course (lib/marching.ts). Colours: orange = above c,
blue = surface / contour, as in fig-cube.svg and fig-classes.svg. Fonts >= 15 pt on the slide."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "marchingcubes")
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
ORANGE, BLUE, GREEN, RED, PURPLE, GREY = "#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"
LIGHTBLUE = "#9ecae1"


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, facecolor="white")
    if os.environ.get("FIGPREVIEW"):
        fig.savefig(os.path.join(os.environ["FIGPREVIEW"], name.replace(".svg", ".png")), facecolor="white", dpi=100)
    plt.close(fig)
    print("wrote", p)


CORNERS = [(k & 1, (k >> 1) & 1, (k >> 2) & 1) for k in range(8)]
CUBE_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def proj(p):
    """Oblique projection used for the cube diagrams: x right, y up the page, z towards the viewer (down-left), as in fig-cube.svg."""
    x, y, z = p
    return np.array([x - 0.42 * z, y - 0.32 * z])


def draw_cube(ax, above=(), labels=True, lw=1.6, ms=11, origin=(0, 0), scale=1.0):
    o = np.array(origin)
    P = lambda p: o + scale * proj(p)
    for a, b in CUBE_EDGES:
        pa, pb = P(CORNERS[a]), P(CORNERS[b])
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#555", lw=lw, zorder=1)
    for k, c in enumerate(CORNERS):
        q = P(c)
        ax.plot(*q, "o", ms=ms, color=ORANGE if k in above else "white", mec="#333", zorder=5)
        if labels:
            dx = -0.1 if c[0] == 0 else 0.1
            ax.text(q[0] + dx * scale, q[1] + (0.07 if c[1] else -0.07) * scale, str(k), fontsize=15,
                    ha="right" if c[0] == 0 else "left", va="bottom" if c[1] else "top", color="#333")
    return P


# 1. a marching-cubes mesh of a sphere: shared vertices, triangles
def shade(ax, P, T, color=(0.62, 0.79, 0.88), edge="#1f4e79", lw=0.6):
    tri = P[T]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    L = np.array([-0.4, -0.5, 0.75]); L /= np.linalg.norm(L)
    lam = 0.35 + 0.65 * np.abs(n @ L)
    cols = np.clip(np.array(color)[None, :] * lam[:, None] + 0.12, 0, 1)
    pc = Poly3DCollection(tri, facecolors=cols, edgecolors=edge, linewidths=lw)
    ax.add_collection3d(pc)


def fig_mesh(d):
    P = np.array(d["mc"]["p"]).reshape(-1, 3); T = np.array(d["mc"]["t"]).reshape(-1, 3)
    fig = plt.figure(figsize=(4.0, 4.0))
    ax = fig.add_subplot(111, projection="3d")
    shade(ax, P, T)
    c = P.mean(0)
    for s, f in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        s(c[f] - 4.6, c[f] + 4.6)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=-58); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, "fig-mesh.svg")
    return len(P), len(T)


# 2. generating the table: case 3 (corners 0 and 1 above)
def fig_table():
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 3.3))
    pts = {"02": (0, .5, 0), "13": (1, .5, 0), "15": (1, 0, .5), "04": (0, 0, .5)}
    segs = [("02", "13", "−z"), ("13", "15", "+x"), ("15", "04", "−y"), ("04", "02", "−x")]
    for ax in axs:
        P = draw_cube(ax, above=(0, 1), labels=ax is axs[0])
        ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.65, 1.25); ax.set_ylim(-0.55, 1.2)
    P = lambda p: proj(p)
    a = axs[0]
    for u, v, f in segs:
        pu, pv = P(pts[u]), P(pts[v])
        a.plot([pu[0], pv[0]], [pu[1], pv[1]], color=BLUE, lw=3.5, zorder=3)
    for k in pts:
        a.plot(*P(pts[k]), "o", color=BLUE, ms=8, zorder=4)
    a.set_title("1. marching squares\non 4 cut faces", fontsize=16)
    b = axs[1]
    loop = [P(pts[k]) for k in ("02", "13", "15", "04")]
    b.add_patch(Polygon([loop[0], loop[1], loop[2]], closed=True, fc=LIGHTBLUE, ec=BLUE, lw=2, zorder=2))
    b.add_patch(Polygon([loop[0], loop[2], loop[3]], closed=True, fc="#c6dbef", ec=BLUE, lw=2, zorder=2))
    for i in range(4):
        p, q = loop[i], loop[(i + 1) % 4]
        m = (p + q) / 2
        b.add_patch(FancyArrowPatch(tuple(p + 0.35 * (q - p)), tuple(p + 0.6 * (q - p)), arrowstyle="-|>",
                                    mutation_scale=20, color=BLUE, lw=0, zorder=4))
    b.plot([loop[0][0], loop[2][0]], [loop[0][1], loop[2][1]], color=BLUE, lw=1.5, ls="--", zorder=3)
    b.set_title("2. chain: a loop\n3. fan: 2 triangles", fontsize=16)
    fig.tight_layout(w_pad=1)
    save(fig, "fig-table.svg")


# 3. marching tetrahedra: 6 tetrahedra around the diagonal 0-7, and the 3 classes
def fig_tets(d):
    fig = plt.figure(figsize=(5.8, 3.2))
    ax = fig.add_axes([0.0, 0.0, 0.42, 0.9])
    draw_cube(ax, labels=True)
    P = proj
    diag = set()
    for T in d["tets6"]:
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = sorted((T[i], T[j]))
                if bin(a ^ b).count("1") > 1:
                    diag.add((a, b))
    for a, b in diag:
        pa, pb = P(CORNERS[a]), P(CORNERS[b])
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=PURPLE if (a, b) == (0, 7) else GREY,
                lw=2.5 if (a, b) == (0, 7) else 1.2, ls="-" if (a, b) == (0, 7) else "--", zorder=2)
    # shade one of the six tetrahedra
    T0 = d["tets6"][0]
    for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        ax.add_patch(Polygon([P(CORNERS[T0[i]]) for i in f], closed=True, fc=PURPLE, alpha=0.12, ec="none"))
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.65, 1.25); ax.set_ylim(-0.6, 1.2)
    ax.set_title("6 tetrahedra around\nthe diagonal 0–7", fontsize=16)
    # the three classes on a tetrahedron
    tet = np.array([[0, 0.15], [1, 0.15], [0.45, 0.95], [0.6, -0.15]])     # edge 0-1 at the back (dashed)
    edges = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
    cases = [((), "nothing"), ((2,), "triangle"), ((2, 3), "quad")]
    for i, (above, lab) in enumerate(cases):
        a2 = fig.add_axes([0.43 + i * 0.19, 0.12, 0.18, 0.62])
        for u, v in edges:
            a2.plot(*zip(tet[u], tet[v]), color="#555", lw=1.5, ls="--" if (u, v) == (0, 1) else "-")
        cut = [tet[u] * 0.5 + tet[v] * 0.5 for u, v in edges if (u in above) != (v in above)]
        if len(cut) == 3:
            a2.add_patch(Polygon(cut, closed=True, fc=LIGHTBLUE, ec=BLUE, lw=2))
        elif len(cut) == 4:
            c0 = np.mean(cut, axis=0)
            cut = sorted(cut, key=lambda p: np.arctan2(p[1] - c0[1], p[0] - c0[0]))
            a2.add_patch(Polygon(cut, closed=True, fc=LIGHTBLUE, ec=BLUE, lw=2))
            a2.plot(*zip(cut[0], cut[2]), color=BLUE, lw=1.2, ls="--")
        for k, p in enumerate(tet):
            a2.plot(*p, "o", ms=11, color=ORANGE if k in above else "white", mec="#333")
        a2.set_title(lab, fontsize=15, y=-0.42 if "\n" in lab else -0.3)
        a2.set_aspect("equal"); a2.axis("off"); a2.set_xlim(-0.12, 1.12); a2.set_ylim(-0.27, 1.05)
    fig.text(0.715, 0.86, "3 classes per tetrahedron", ha="center", fontsize=16)
    save(fig, "fig-tets.svg")


# 4. normals from the image: level lines of a bright blob, gradient inwards, normal outwards
def fig_gradnormal():
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    x = np.linspace(-3, 3, 241)
    X, Y = np.meshgrid(x, x)
    I = 40 + 160 * np.exp(-((X / 1.5) ** 2 + (Y / 1.1) ** 2) / 2)
    ax.imshow(I, cmap="gray", vmin=0, vmax=255, extent=(-3, 3, 3, -3))
    ax.contour(X, Y, I, levels=[70, 100, 160, 190], colors=["#888"], linewidths=1)
    cs = ax.contour(X, Y, I, levels=[120], colors=[BLUE], linewidths=3)
    # points on the level 120 ellipse: exp(-q/2) = 0.5 -> q = 2 ln 2
    r = np.sqrt(2 * np.log(2))
    for th in np.deg2rad([20, 100, 200, 300]):
        px, py = 1.5 * r * np.cos(th), 1.1 * r * np.sin(th)
        gx = -px / 1.5 ** 2; gy = -py / 1.1 ** 2           # direction of grad I (up to a positive factor)
        g = np.array([gx, gy]) / np.hypot(gx, gy)
        ax.add_patch(FancyArrowPatch((px, py), (px + 0.8 * g[0], py + 0.8 * g[1]), arrowstyle="-|>", mutation_scale=18,
                                     color=ORANGE, lw=2.5))
        ax.add_patch(FancyArrowPatch((px, py), (px - 0.8 * g[0], py - 0.8 * g[1]), arrowstyle="-|>", mutation_scale=18,
                                     color=RED, lw=2.5))
    ax.text(-2.85, 2.85, "∇I: inwards", color=ORANGE, fontsize=17, va="bottom", fontweight="bold")
    ax.text(2.85, 2.85, "n = −∇I/‖∇I‖", color=RED, fontsize=17, va="bottom", ha="right", fontweight="bold",
            bbox=dict(fc="white", ec="none", pad=1))
    ax.text(-2.85, -2.85, "level c = 120", color=BLUE, fontsize=17, va="top", bbox=dict(fc="white", ec="none", pad=1))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-3, 3); ax.set_ylim(3, -3)
    fig.tight_layout()
    save(fig, "fig-gradnormal.svg")


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    print(fig_mesh(data)); fig_table(); fig_tets(data); fig_gradnormal()
