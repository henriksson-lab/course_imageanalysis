"""Slide figures for stereo/lec.qmd.

Run from the repository root (the disparity maps come from the course's own stereo code):
    node_modules/.bin/esbuild tools/figs/stereo/compute.ts --bundle --platform=node --outfile=$TMP/s.js && node $TMP/s.js $TMP
    python3 -I tools/figs/stereo/figs.py $TMP
Writes fig-*.svg into stereo/. The rectification figure uses the course micrograph
(demos/src/fourier2d/cell256.png, Robert M. Hunt, CC BY 3.0) as a flat textured scene.
"""
import os
import sys
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../multiview"))
from common import plt, px, save, load, cell256, BLUE, ORANGE, GREEN, RED, PURPLE, GREY, YELLOW  # noqa: E402

TMP = sys.argv[1] if len(sys.argv) > 1 else "."
U = "stereo"
D = load(TMP, "stereo.json")


def img(name, key):
    s = D[name]
    return np.array(s[key], float).reshape(s["h"], s["w"])


def disp_img(name, key):
    s = D[key] if key in D else None
    w, h = D[name]["w"], D[name]["h"]
    return np.array(s["d"], float).reshape(h, w)


def show_disp(ax, d, maxD, title):
    m = np.ma.masked_less(d, 0)
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad("#9a9a9a")
    ax.imshow(m, cmap=cm, vmin=0, vmax=maxD)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def cbar(fig, axs, maxD, im=None):
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, maxD))
    fig.tight_layout(w_pad=0.8)
    fig.subplots_adjust(right=0.93)
    pos = axs[-1].get_position()
    cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.012, pos.height])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("disparity (px)", fontsize=18)
    cb.set_ticks([0, maxD // 2, maxD])


def show_grey(ax, a, title):
    ax.imshow(a, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")


# 1. window size on random dots: truth, 3 × 3, 7 × 7, 15 × 15 (preset "window")
maxD = D["dots"]["maxD"]
fig, axs = plt.subplots(1, 4, figsize=px(1120, 235))
show_disp(axs[0], img("dots", "truth"), maxD, "truth")
for ax, r in zip(axs[1:], (1, 3, 7)):
    k = f"dots_wta_r{r}"
    show_disp(ax, disp_img("dots", k), maxD, f"{2 * r + 1} × {2 * r + 1}: {100 * D[k]['rate']:.1f} % bad")
cbar(fig, axs, maxD)
save(fig, U, "fig-window.svg")

# 2. gain change: left, right (darker, offset), WTA 7 × 7 with SAD / NCC / census (preset "gain")
maxD = D["gain"]["maxD"]
fig, axs = plt.subplots(1, 5, figsize=px(1150, 200))
show_grey(axs[0], img("gain", "left"), "left")
show_grey(axs[1], img("gain", "right"), "right: ×0.6, +60")
for ax, k, lab in zip(axs[2:], ("sad", "ncc", "census"), ("SAD", "NCC", "census")):
    key = f"gain_{k}"
    show_disp(ax, disp_img("gain", key), maxD, f"{lab}: {100 * D[key]['rate']:.1f} % bad")
cbar(fig, axs, maxD)
save(fig, U, "fig-gain.svg")

# 3. failures and the left–right check on "wall, fence, post" (SAD 3 × 3)
maxD = D["mixed"]["maxD"]
fig, axs = plt.subplots(1, 3, figsize=px(1120, 300))
show_grey(axs[0], img("mixed", "left"), "left image")
for x, y, t in ((58, 47, "wall"), (138, 42, "fence"), (89, 120, "post"), (150, 110, "ground")):
    axs[0].text(x, y, t, color="white", fontsize=19, ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.55))
show_disp(axs[1], disp_img("mixed", "mixed_wta"), maxD, f"winner-take-all: {100 * D['mixed_wta']['rate']:.1f} % bad")
show_disp(axs[2], disp_img("mixed", "mixed_sgm8_lr"), maxD, "SGM 8 + left–right check")
cbar(fig, axs, maxD)
save(fig, U, "fig-failures.svg")

# 4. aggregation: WTA, scanline DP, SGM 8 (P1 = 0.1, P2 = 1)
fig, axs = plt.subplots(1, 4, figsize=px(1120, 235))
show_disp(axs[0], img("mixed", "truth"), maxD, "truth")
for ax, k, lab in zip(axs[1:], ("mixed_wta", "mixed_dp", "mixed_sgm8"), ("WTA", "scanline DP", "SGM 8")):
    show_disp(ax, disp_img("mixed", k), maxD, f"{lab}: {100 * D[k]['rate']:.1f} %")
cbar(fig, axs, maxD)
save(fig, U, "fig-aggregation.svg")

# 5. the exercise's 4 × 5 cost slice with the WTA profile and the DP path
C = np.array([[5, 4, 6, 1, 5], [2, 3, 1, 2, 2], [6, 2, 4, 6, 6], [4, 5, 5, 4, 7]])
wta = C.argmin(0)
fig, ax = plt.subplots(figsize=px(520, 380))
ax.imshow(C, cmap="Greys_r", vmin=0, vmax=9, origin="lower", aspect="auto")
xs = np.arange(5)
for d in range(4):
    for x in range(5):
        ax.text(x, d, str(C[d, x]), ha="center", va="center", fontsize=20, zorder=5,
                color="white" if C[d, x] < 4 else "black")
for x in xs:
    ax.add_patch(Rectangle((x - 0.46, wta[x] - 0.46), 0.92, 0.92, fill=False, ec=RED, lw=3.5, zorder=4))
    ax.add_patch(Rectangle((x - 0.36, 1 - 0.36), 0.72, 0.72, fill=False, ec=YELLOW, lw=3.5, ls="--", zorder=4))
ax.plot([], [], "-", color=RED, lw=3.5, label="WTA: 8 + 4 steps = 12")
ax.plot([], [], "--", color=YELLOW, lw=3.5, label="DP: d = 1, energy 10")
ax.set_xticks(xs); ax.set_yticks(range(4))
ax.set_xlabel("pixel x"); ax.set_ylabel("disparity d")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, frameon=False)
ax.set_title("cost slice of one row, P₁ = 1")
save(fig, U, "fig-dp-slice.svg")

# 6. sub-pixel parabola through C(−1) = 4, C(0) = 1, C(+1) = 2
fig, ax = plt.subplots(figsize=px(470, 340))
cm_, c0, cp = 4, 1, 2
a, b = 0.5 * (cm_ - 2 * c0 + cp), 0.5 * (cp - cm_)
t = np.linspace(-1.4, 1.4, 200)
ax.plot(t, a * t ** 2 + b * t + c0, color=BLUE, lw=2.5, label="parabola")
ax.plot([-1, 0, 1], [cm_, c0, cp], "o", color="black", ms=10, label="costs")
ds = -b / (2 * a)
ax.plot([ds], [a * ds ** 2 + b * ds + c0], "D", color=RED, ms=10)
ax.axvline(ds, color=RED, ls="--", lw=1.5)
ax.annotate(f"δ* = +{ds:.2f}", (ds, 0.55), (0.42, 3.5), color=RED, fontsize=20, arrowprops=dict(arrowstyle="->", color=RED))
ax.set_xticks([-1, 0, 1]); ax.set_xticklabels(["d − 1", "d", "d + 1"])
ax.set_ylabel("cost"); ax.set_ylim(0, 4.8)
save(fig, U, "fig-subpixel.svg")

# 7. rectification of a toed-in pair looking at a flat textured plane
tex = cell256() * 255
f, W, H = 260.0, 320, 240
K = np.array([[f, 0, W / 2 - 0.5], [0, f, H / 2 - 0.5], [0, 0, 1]])


def rot(ax_, ang):
    c, s = np.cos(ang), np.sin(ang)
    if ax_ == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if ax_ == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


Z0, sc, B = 4.0, 6.5 / 256, 1.0
cams_before = [(rot("y", np.radians(10)) @ rot("x", np.radians(3)), np.array([-B / 2, 0, 0])),
               (rot("y", np.radians(-9)) @ rot("z", np.radians(4)), np.array([B / 2, 0, 0]))]
cams_after = [(np.eye(3), np.array([-B / 2, 0, 0])), (np.eye(3), np.array([B / 2, 0, 0]))]


def plane_H(R, C):
    """Texture pixel (s, t, 1) → image pixel, for the plane Z = Z0 (texture centred on the axis)."""
    M = np.array([[sc, 0, -128 * sc - C[0]], [0, sc, -128 * sc - C[1]], [0, 0, Z0 - C[2]]])
    return K @ R @ M


def render(R, C):
    from scipy.ndimage import map_coordinates
    Hi = np.linalg.inv(plane_H(R, C))
    v, u = np.mgrid[0:H, 0:W]
    p = Hi @ np.vstack([u.ravel(), v.ravel(), np.ones(u.size)])
    s, t_ = p[0] / p[2], p[1] / p[2]
    out = map_coordinates(tex, [t_, s], order=1, cval=255)
    return out.reshape(H, W)


def proj(R, C, X):
    q = K @ R @ (X - C)
    return q[:2] / q[2]


pts3 = [np.array([x, y, Z0]) for x, y in ((-1.0, -0.9), (0.6, -0.2), (-0.4, 0.5), (0.9, 0.95))]
cols = [BLUE, ORANGE, GREEN, PURPLE]
fig, axs = plt.subplots(2, 2, figsize=px(520, 470))
for row, (cams, title) in enumerate(((cams_before, "original"), (cams_after, "rectified"))):
    for k, (R, C) in enumerate(cams):
        ax = axs[row, k]
        ax.imshow(render(R, C), cmap="gray", vmin=0, vmax=255)
        other = cams[1 - k]
        for X, col in zip(pts3, cols):
            # epipolar line in this image of X seen by the other camera: through proj(X) and the epipole
            a = proj(R, C, X)
            bpt = proj(R, C, X + 3.0 * (X - other[1]))      # a point further along the other camera's ray
            d = (bpt - a) / np.linalg.norm(bpt - a)
            ax.plot([a[0] - 600 * d[0], a[0] + 600 * d[0]], [a[1] - 600 * d[1], a[1] + 600 * d[1]], color=col, lw=2.2)
            ax.plot(*a, "o", color=col, ms=8, mec="white", mew=1.2)
        ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
        ax.axis("off")
        ax.set_title(f"{title}: {'left' if k == 0 else 'right'}", fontsize=20)
fig.tight_layout(h_pad=0.6, w_pad=0.6)
save(fig, U, "fig-rectify.svg")
