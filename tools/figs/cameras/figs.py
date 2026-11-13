"""Slide figures for cameras/lec.qmd (fig-pinhole.svg and fig-distortion.svg come from
tools/figs/multiview/figs.py).

Run from the repository root (the calibration data come from the dltcalib demo's own model):
    node_modules/.bin/esbuild tools/figs/cameras/compute.ts --bundle --platform=node --outfile=$TMP/c.js && node $TMP/c.js $TMP
    python3 -I tools/figs/cameras/figs.py $TMP
Writes fig-*.svg into cameras/. All data synthetic.
"""
import os
import sys
import numpy as np
from matplotlib.patches import Polygon, Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../multiview"))
from common import plt, px, save, load, BLUE, ORANGE, GREEN, RED, PURPLE, GREY, SKY  # noqa: E402

TMP = sys.argv[1] if len(sys.argv) > 1 else "."
U = "cameras"
D = load(TMP, "calib.json")
IMG_W, IMG_H = 640, 480


def image_axes(ax, title=None, w=IMG_W, h=IMG_H, ticks=True):
    ax.add_patch(Rectangle((0, 0), w, h, fill=True, fc="#fafafa", ec="#666", lw=1.5, zorder=0))
    ax.set_xlim(-10, w + 10); ax.set_ylim(h + 10, -10); ax.set_aspect("equal")
    if ticks:
        ax.set_xticks([0, w // 2, w]); ax.set_yticks([0, h // 2, h])
    else:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    if title:
        ax.set_title(title)


def camera_glyph(ax, c, direction, size=0.35, color="#222", label=None):
    d = np.array(direction, float); d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    tri = [c, c + size * (d + 0.6 * n), c + size * (d - 0.6 * n)]
    ax.add_patch(Polygon(tri, closed=True, fc=color, ec=color, alpha=0.85, zorder=4))
    if label:
        ax.text(*(c - 0.32 * d), label, ha="center", va="top", color=color)


# 1. a pixel is a ray: one camera cannot tell depth, two rays meet ------------------------------
fig, ax = plt.subplots(figsize=px(460, 420))
C1, C2 = np.array([0.0, 0.0]), np.array([3.4, 0.0])
X = np.array([1.8, 3.6])
d1 = (X - C1) / np.linalg.norm(X - C1)
ax.plot(*np.c_[C1, C1 + 5.0 * d1], color=BLUE, lw=2.5)
for s, a in ((2.0, 0.35), (3.71, 1.0), (4.8, 0.35)):
    p = C1 + s * d1
    ax.plot(*p, "o", color=BLUE if a < 1 else RED, ms=11, alpha=1.0, mfc="white" if a < 1 else RED, mew=2.5)
ax.plot(*np.c_[C2, X + 0.6 * (X - C2) / np.linalg.norm(X - C2)], color=ORANGE, lw=2.5, ls="--")
camera_glyph(ax, C1, d1, color=BLUE, label="camera 1")
camera_glyph(ax, C2, X - C2, color=ORANGE, label="camera 2")
ax.text(-1.25, 3.0, "one camera:\nall points on\nthe ray give\nthe same pixel", color=BLUE, va="center", fontsize=19)
ax.text(X[0] + 0.75, X[1] - 0.6, "two cameras:\nthe rays meet\nat the point", color=RED, va="center", fontsize=19)
ax.set_xlim(-1.3, 5.0); ax.set_ylim(-0.75, 5.0); ax.set_aspect("equal"); ax.axis("off")
save(fig, U, "fig-two-rays.svg")

# 2. worked example: where the point (0.5, −0.25, Z) lands for growing Z -----------------------
f, cx, cy = 800, 320, 240
fig, ax = plt.subplots(figsize=px(540, 330))
image_axes(ax)
ax.plot(cx, cy, "+", color=GREEN, ms=22, mew=3)
ax.text(cx - 4, cy + 6, "principal point (320, 240)", color=GREEN, va="top", fontsize=19)
Zs = (5, 10, 20)
uv = [(f * 0.5 / Z + cx, f * -0.25 / Z + cy) for Z in Zs]
ax.plot([cx, uv[0][0] + 40], [cy, uv[0][1] - 20], color=GREY, lw=1.5, ls=":")
for (u, v), Z, col in zip(uv, Zs, (RED, ORANGE, PURPLE)):
    ax.plot(u, v, "o", color=col, ms=12)
for (u, v), Z, col in zip(uv, Zs, (RED, ORANGE, PURPLE)):
    ax.text(u - 8, v - 7, f"Z = {Z} m: ({u:.0f}, {v:.0f})" if Z < 20 else "20 m", color=col, fontsize=19, ha="right", va="bottom")
ax.set_xlim(250, 430); ax.set_ylim(268, 178)
ax.set_xticks([280, 320, 360, 400]); ax.set_yticks([200, 220, 240, 260])
ax.set_xlabel("u (pixels)"); ax.set_ylabel("v (pixels)")
ax.set_title("part of the 640 × 480 image, f = 800 px")
save(fig, U, "fig-project-example.svg")

# 3. zoom versus walking closer --------------------------------------------------------------------
def box(x0, x1, y0, y1, z0, z1):
    P = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    E = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
    return P, E


scene = [box(-0.5, 0.5, 0.0, 1.0, 4.0, 5.0)]
for x in (-2.2, 2.2):
    scene.append(box(x - 0.08, x + 0.08, -1.6, 1.0, 10.0, 10.16))
ground = [(np.array([[-3, 1, 3.0], [-3, 1, 14.0]]), RED), (np.array([[3, 1, 3.0], [3, 1, 14.0]]), RED)]


def draw_view(ax, fpx, camZ, title):
    image_axes(ax, title, ticks=False)
    for k, (P, E) in enumerate(scene):
        Q = P.copy(); Q[:, 2] -= camZ
        u = fpx * Q[:, 0] / Q[:, 2] + cx; v = fpx * Q[:, 1] / Q[:, 2] + cy
        for i, j in E:
            ax.plot([u[i], u[j]], [v[i], v[j]], color=BLUE if k == 0 else ORANGE, lw=2.5, solid_capstyle="round")
    ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0)
    for a in ax.lines:
        a.set_clip_on(True)


fig, axs = plt.subplots(1, 3, figsize=px(1100, 300))
draw_view(axs[0], 600, 0.0, "start: f = 600 px, 4 m away")
draw_view(axs[1], 1200, 0.0, "zoom: f = 1200 px")
draw_view(axs[2], 600, 2.0, "walk closer: 2 m away")
fig.tight_layout(w_pad=1.2)
save(fig, U, "fig-zoom-walk.svg")

# 4. from the pixel (400, 200) back to its ray (top view, X against Z) ---------------------------
fig, ax = plt.subplots(figsize=px(480, 440))
ax.plot([0, 11], [0, 1.1], color=BLUE, lw=2.5)
ax.plot([1, 1], [-0.25, 0.4], color=GREEN, lw=4)
ax.text(1.3, -0.3, "image plane Z = 1", color=GREEN, fontsize=20, va="center")
ax.plot([1], [0.1], "o", color=GREEN, ms=11)
ax.annotate("u = 400: x = (400 − 320) / 800 = 0.1", (1, 0.1), (-3.2, 1.18), color=GREEN, fontsize=19,
            arrowprops=dict(arrowstyle="->", color=GREEN))
for Z in (5, 10):
    ax.plot([Z], [0.1 * Z], "o", color=RED, ms=11)
    ax.text(Z + 0.3, 0.1 * Z - 0.08, f"X = {0.1 * Z:.1f}", color=RED, va="top", fontsize=20)
ax.plot([0], [0], "s", color="#222", ms=11)
ax.text(-0.4, 0.0, "camera\ncentre", va="center", ha="right", fontsize=20)
ax.set_xlabel("depth Z (m)"); ax.set_ylabel("X (m)")
ax.set_xlim(-3.3, 12.5); ax.set_ylim(-0.45, 1.3); ax.set_xticks([0, 5, 10]); ax.set_yticks([0, 0.5, 1])
save(fig, U, "fig-pixel-ray.svg")


# 5. the calibration target in the image, with the 72 corners -----------------------------------
def checker_cells(ax, T, n_i, n_j, off, flat_idx):
    for i in range(n_i - 1):
        for j in range(n_j - 1):
            q = [T[flat_idx(i, j) + off], T[flat_idx(i + 1, j) + off], T[flat_idx(i + 1, j + 1) + off], T[flat_idx(i, j + 1) + off]]
            ax.add_patch(Polygon(q, closed=True, fc="#333" if (i + j) % 2 == 0 else "#e8e8e8", ec="none", zorder=1))


nz = D["noise"]
T = np.array(nz["truth"]); O = np.array(nz["obs"])
fig, ax = plt.subplots(figsize=px(480, 400))
image_axes(ax)
for off in (0, 36):
    checker_cells(ax, T, 6, 6, off, lambda i, j: 6 * i + j)
ax.plot(O[:, 0], O[:, 1], "o", ms=8, mfc="none", mec=RED, mew=2.2, zorder=3)
ax.set_xlabel("u (pixels)"); ax.set_ylabel("v (pixels)")
ax.set_title("72 known corners on two walls")
save(fig, U, "fig-target.svg")

# 6. singular values of the DLT matrix: 72 points on two walls vs 36 on one plane ---------------
fig, axs = plt.subplots(2, 1, figsize=px(400, 480), sharex=True)
for ax, S, title in ((axs[0], D["noise"]["S"], "two walls (72)"), (axs[1], D["coplanarS"], "one plane (36)")):
    S = np.maximum(np.array(S), 1e-6)
    cols = [RED if v < 1e-3 * S[0] else BLUE for v in S]
    ax.bar(np.arange(1, 13), S, color=cols)
    ax.set_yscale("log"); ax.set_ylim(1e-6, 50)
    ax.set_xticks([1, 6, 12]); ax.set_title(title, fontsize=19)
    ax.set_yticks([1e-6, 1e-3, 1])
    ax.set_ylabel("singular value")
axs[1].set_xlabel("index (largest to smallest)")
axs[1].text(12.4, 3e-6, "≈ 0", color=RED, ha="right", va="bottom", fontsize=19)
fig.tight_layout()
save(fig, U, "fig-dlt-sv.svg")

# 7. flat board: three tilted poses and one nearly parallel pose ---------------------------------
fig, axs = plt.subplots(2, 2, figsize=px(460, 400))
views = [(D["planar"][0], "pose 1"), (D["planar"][1], "pose 2"), (D["planar"][2], "pose 3"), (D["parallel"][0], "parallel to sensor")]
for ax, (v, title) in zip(axs.ravel(), views):
    image_axes(ax, title, ticks=False)
    Tv = np.array(v["truth"])
    checker_cells(ax, Tv, 8, 6, 0, lambda i, j: 8 * j + i)
    ax.title.set_color(RED if "parallel" in title else "black")
fig.tight_layout(h_pad=0.4, w_pad=0.6)
save(fig, U, "fig-planar-views.svg")

# 8. residual arrows (× 30): distortion ignored (linear DLT) vs refined with k1, k2 --------------
dd = D["distortion"]
O = np.array(dd["obs"])
fig, axs = plt.subplots(1, 2, figsize=px(1000, 250))
for ax, key, title in ((axs[0], "lin", f"linear, distortion ignored: RMS {dd['linRms']:.2f} px"),
                       (axs[1], "ref", f"refined with k₁, k₂: RMS {dd['refRms']:.2f} px")):
    image_axes(ax, title, ticks=False)
    E = np.array(dd[key])
    ax.plot(O[:, 0], O[:, 1], "o", ms=5, color=BLUE, zorder=3)
    for o, e in zip(O, E):
        ax.annotate("", xy=o + 30 * (e - o), xytext=o, arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8, mutation_scale=12))
    ax.set_xlim(O[:, 0].min() - 35, O[:, 0].max() + 35); ax.set_ylim(O[:, 1].max() + 35, O[:, 1].min() - 35)
fig.text(1.0, 0.5, "arrows:\nresidual\n× 30", ha="left", va="center", color=RED, fontsize=19)
fig.tight_layout(w_pad=1.5)
save(fig, U, "fig-residuals.svg")
