"""Figures for hough-lines/lec.qmd.

Run from the repository root:
    tools/figs/hough-lines/capture.sh /tmp/houghstills      # demo screenshots (needs docs/demos/dist)
    python3 -I tools/figs/hough-lines/figs.py /tmp/houghstills
Diagrams are computed here; the demo stills are crops of the captured screenshots, with the
numbers in their titles read from the demo's readout for the same preset (see the slide notes).
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stills import plt, panel, grid_of_panels, BLUE, ORANGE, GREEN, RED, PURPLE, GREY  # noqa: E402
from matplotlib.patches import Arc, FancyArrowPatch  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage import feature, io  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = os.path.join(REPO, "hough-lines")
SH = sys.argv[1] if len(sys.argv) > 1 else None


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200 if name.endswith(".png") else None,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ---------------------------------------------------------------- edge points with directions
def fig_edgepoints():
    img = io.imread(os.path.join(REPO, "demos/src/fourier2d/cell256.png"), as_gray=True).astype(float)
    if img.max() > 1:
        img /= 255
    sigma = 3.0
    ed = feature.canny(img, sigma=sigma, low_threshold=0.80, high_threshold=0.93, use_quantiles=True)
    gy = ndi.gaussian_filter(img, sigma, order=(1, 0))
    gx = ndi.gaussian_filter(img, sigma, order=(0, 1))
    # zoom window: pick the 24x24 window with most edge pixels in a band around the middle
    # zoom window: centred on the longest connected edge (a membrane), at its middle point
    lab, nl = ndi.label(ed, structure=np.ones((3, 3)))
    sizes = ndi.sum(ed, lab, range(1, nl + 1))
    ys, xs = np.nonzero(lab == 1 + int(np.argmax(sizes)))
    mid = len(ys) // 2
    by, bx = int(np.clip(ys[mid] - 12, 0, 232)), int(np.clip(xs[mid] - 12, 0, 232))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.4), gridspec_kw={"width_ratios": [1, 1]})
    rgb = np.dstack([img] * 3) * 0.75
    rgb[ed] = [1.0, 0.5, 0.05]
    a1.imshow(rgb, interpolation="nearest")
    a1.add_patch(plt.Rectangle((bx - 0.5, by - 0.5), 24, 24, fill=False, ec=BLUE, lw=2.5))
    a1.set_title("edge points (Canny)", fontsize=14)
    a1.axis("off")
    sub = img[by:by + 24, bx:bx + 24]
    a2.imshow(sub, cmap="gray", vmin=0, vmax=1, interpolation="nearest", alpha=0.6)
    ys, xs = np.nonzero(ed[by:by + 24, bx:bx + 24])
    a2.plot(xs, ys, "o", color=ORANGE, ms=4)
    for k in range(0, len(xs), 3):
        x, y = xs[k], ys[k]
        u, v = gx[by + y, bx + x], gy[by + y, bx + x]
        n = np.hypot(u, v) + 1e-12
        a2.annotate("", xy=(x + 3 * u / n, y + 3 * v / n), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6, mutation_scale=10))
    a2.set_xlim(-0.5, 23.5)
    a2.set_ylim(23.5, -0.5)
    a2.set_title("zoom: position + gradient", fontsize=14)
    for s in a2.spines.values():
        s.set_visible(True)
        s.set_color(BLUE)
        s.set_linewidth(2.5)
    a2.set_xticks([])
    a2.set_yticks([])
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig-edgepoints.png")


# ---------------------------------------------------------------- slope-intercept vs normal form
def fig_slope():
    pts = [(0.6, -3.0), (1.0, 0.0), (1.4, 3.0)]   # demo houghdual, preset "towards vertical"
    cols = [BLUE, GREEN, RED]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.3))
    m = np.linspace(-4, 4, 200)
    for (x, y), c in zip(pts, cols):
        a1.plot(m, y - m * x, color=c, lw=2.5)
        a1.text(-3.85, y + 3.85 * x + 0.5, f"({x:g}, {y:g})".replace("-", "−"), color=c, fontsize=13, va="bottom")
    a1.set_xlim(-4, 4)
    a1.set_ylim(-9, 10.5)
    a1.set_xlabel("slope m")
    a1.set_ylabel("intercept c")
    a1.axhline(0, color="#ccc", lw=0.8, zorder=0)
    a1.axvline(0, color="#ccc", lw=0.8, zorder=0)
    a1.annotate("meet at m = 7.5", xy=(4.0, -8.0), xytext=(-3.8, -8.0),
                fontsize=13, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
    a1.set_title("slope–intercept", fontsize=14)
    a1.set_xticks([-4, -2, 0, 2, 4])
    th = np.linspace(0, np.pi, 361)
    for (x, y), c in zip(pts, cols):
        a2.plot(np.degrees(th), x * np.cos(th) + y * np.sin(th), color=c, lw=2.5)
    # common line: through (0.6,-3) and (1.4,3); normal perpendicular to (0.8, 6)
    t0 = np.arctan2(-0.8, 6.0) % np.pi
    r0 = 1.0 * np.cos(t0)
    a2.plot(np.degrees(t0), r0, "o", ms=13, mfc="none", mec="k", mew=2)
    a2.annotate(f"θ = {np.degrees(t0):.1f}°\nρ = {r0:.2f}".replace("-", "−"), xy=(np.degrees(t0), r0),
                xytext=(118, -2.6), fontsize=13, ha="center",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
    a2.set_xlim(0, 180)
    a2.set_xticks([0, 45, 90, 135, 180])
    a2.set_xlabel("θ (degrees)")
    a2.set_ylabel("ρ")
    a2.axhline(0, color="#ccc", lw=0.8, zorder=0)
    a2.set_title("normal form", fontsize=14)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig-slope.svg")
    return np.degrees(t0), r0


# ---------------------------------------------------------------- normal form geometry
def fig_normalform():
    fig, ax = plt.subplots(figsize=(3.9, 3.9))
    ax.set_aspect("equal")
    ax.axhline(0, color="#999", lw=1)
    ax.axvline(0, color="#999", lw=1)
    xs = np.linspace(-1.0, 3.0, 2)
    ax.plot(xs, 2 - xs, color=GREEN, lw=3)
    for (x, y), c in zip([(2, 0), (1, 1), (0, 2)], [BLUE, GREEN, RED]):
        ax.plot(x, y, "o", color=c, ms=10, zorder=5)
    ax.annotate("", xy=(1, 1), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=2.2, mutation_scale=16))
    ax.text(0.28, 0.66, "ρ = √2", fontsize=15, rotation=45, ha="center", va="center")
    ax.add_patch(Arc((0, 0), 1.1, 1.1, theta1=0, theta2=45, color=PURPLE, lw=2))
    ax.text(0.68, 0.22, "θ = 45°", fontsize=15, color=PURPLE)
    # right-angle mark at the foot of the normal
    d = 0.16
    u, v = np.array([1, 1]) / np.sqrt(2), np.array([-1, 1]) / np.sqrt(2)
    p = np.array([1, 1])
    sq = np.array([p - u * d, p - u * d + v * d, p + v * d])
    ax.plot(sq[:, 0], sq[:, 1], color="k", lw=1.2)
    ax.text(2.75, 0.08, "x", fontsize=14)
    ax.text(0.08, 2.72, "y", fontsize=14)
    ax.text(2.0, 1.15, "x + y = 2", fontsize=14, color=GREEN)
    ax.set_xlim(-0.6, 2.9)
    ax.set_ylim(-0.6, 2.9)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    save(fig, "fig-normalform.svg")


# ---------------------------------------------------------------- voting: a small example
def fig_voting():
    rng = np.random.default_rng(9)
    th0, r0 = np.radians(60), 10.0
    n, t = np.array([np.cos(th0), np.sin(th0)]), np.array([-np.sin(th0), np.cos(th0)])
    s = np.r_[np.arange(-24, -6, 3), np.arange(6, 25, 3)]          # a line with a gap
    line = r0 * n + s[:, None] * t
    clutter = rng.uniform(-28, 28, size=(10, 2))
    P = np.vstack([line, clutter])
    dth, drho, R = 3, 2.0, 40.0
    thetas = np.radians(np.arange(0, 180, dth))
    nr = int(2 * R / drho)
    H = np.zeros((nr, len(thetas)), int)
    for x, y in P:
        rho = x * np.cos(thetas) + y * np.sin(thetas)
        k = np.floor((rho + R) / drho).astype(int)
        H[k, np.arange(len(thetas))] += 1
    k, j = np.unravel_index(H.argmax(), H.shape)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.2), gridspec_kw={"width_ratios": [1, 1.35]})
    a1.set_aspect("equal")
    a1.plot(line[:, 0], line[:, 1], "o", color=ORANGE, ms=7, label=f"{len(line)} on a line")
    a1.plot(clutter[:, 0], clutter[:, 1], "o", color=GREY, ms=7, label=f"{len(clutter)} clutter")
    a1.set_xlim(-30, 30)
    a1.set_ylim(-52, 30)
    a1.set_yticks([-20, 0, 20])
    a1.set_xlabel("x")
    a1.set_ylabel("y")
    a1.legend(fontsize=13, loc="lower left", frameon=False, handletextpad=0.1, borderaxespad=0.1)
    a1.set_title("edge points", fontsize=14)
    im = a2.imshow(H, origin="lower", aspect="auto", cmap="magma_r",
                   extent=[0, 180, -R, R], interpolation="nearest")
    a2.plot(j * dth + dth / 2, -R + (k + 0.5) * drho, "o", ms=16, mfc="none", mec=BLUE, mew=2.5)
    a2.annotate(f"{H[k, j]} votes", xy=(j * dth + dth / 2, -R + (k + 0.5) * drho), xytext=(105, 28),
                fontsize=13, color=BLUE, arrowprops=dict(arrowstyle="-|>", color=BLUE))
    a2.set_xlabel("θ (degrees)")
    a2.set_ylabel("ρ")
    a2.set_xticks([0, 45, 90, 135, 180])
    a2.set_title(f"accumulator: {dth}° × {drho:g} px bins", fontsize=14)
    cb = fig.colorbar(im, ax=a2, fraction=0.05, pad=0.02)
    cb.set_label("votes")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig-voting.svg")
    others = np.sort(H.ravel())[::-1]
    return H[k, j], j * dth, -R + k * drho, len(line), others[1]


# ---------------------------------------------------------------- demo stills
def fig_stills():
    grid_of_panels([panel(SH, "hl_nonms", 1), panel(SH, "hl_nonms", 2),
                    panel(SH, "hl_clean", 1), panel(SH, "hl_clean", 2)],
                   ["no NMS: accumulator", "22 detections",
                    "NMS ±4 bins", "8 detections"],
                   os.path.join(OUT, "fig-nms.png"), ncols=4, w=8.8)
    grid_of_panels([panel(SH, "hl_noise18", 1), panel(SH, "hl_fine", 1), panel(SH, "hl_coarse", 1),
                    panel(SH, "hl_noise18", 2), panel(SH, "hl_fine", 2), panel(SH, "hl_coarse", 2)],
                   ["1°, 1 px", "0.25°, 0.5 px", "5°, 8 px (grid)",
                    "9 lines", "3 lines", "21: rows mixed"],
                   os.path.join(OUT, "fig-binsize.png"), ncols=3, w=5.0)
    grid_of_panels([panel(SH, "hl_clutter", 1), panel(SH, "hl_clutter", 2),
                    panel(SH, "hl_grad", 1), panel(SH, "hl_grad", 2)],
                   ["all θ: 381,240 votes", "32 detections",
                    "gradient ±3°: 14,826", "5 detections"],
                   os.path.join(OUT, "fig-gradvote.png"), ncols=2, w=4.4)
    grid_of_panels([panel(SH, "hl_segments", 0), panel(SH, "hl_segments", 2)],
                   ["edge points", "7 segments (gap 6 px)"],
                   os.path.join(OUT, "fig-segments.png"), ncols=2, w=5.0)
    grid_of_panels([panel(SH, "hl_deskew", 0), panel(SH, "hl_deskew", 1), panel(SH, "hl_deskew", 2)],
                   ["tilted page: edges", "text rows: θ = 97°", "detected lines"],
                   os.path.join(OUT, "fig-deskew.png"), ncols=3, w=7.0)


if __name__ == "__main__":
    fig_edgepoints()
    print("slope fig: common line", fig_slope())
    fig_normalform()
    print("voting fig: peak votes, theta, rho, n line, second", fig_voting())
    if SH:
        fig_stills()
