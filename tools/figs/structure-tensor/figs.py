"""Static figures for the structure-tensor/ deck (slide pictures).

Run from the repository root:  python3 -I tools/figs/structure-tensor/figs.py
Writes structure-tensor/fig-windows.png, fig-st-entries.png, fig-orient-map.png,
fig-corners.png, fig-line-profile.svg, fig-orient-cell.png.
Structure tensor and Harris as in demos/src/lib/edges.ts (numpy port in
tools/figs/canny/edgelib.py). Hue = orientation of the structure (counter-clockwise on the
screen, 0-180 degrees), saturation = coherence, brightness = energy, as in the demo.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "canny"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import maximum_filter, rotate
from edgelib import (structure_tensor, hessian, blur, cell, show, colour_wheel, hsv_to_rgb,
                     BLUE, ORANGE, GREEN, RED, CYAN, GREY, PURPLE, YELLOW, FONT)

OUT = "structure-tensor"


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.05, **kw)
    plt.close(fig)
    print("wrote", name)


def orient_rgb(J, energy_pct=99):
    ori = np.mod(-J["angle"] + np.pi / 2, np.pi) / np.pi          # structure orientation on screen
    en = J["mu1"] + J["mu2"]
    v = np.clip(en / np.percentile(en, energy_pct), 0, 1)
    return hsv_to_rgb(ori, np.clip(J["coh"], 0, 1), v)


def wheel(fig, rect):
    ax = fig.add_axes(rect)
    colour_wheel(ax, axial=True)
    for t, (dx, dy) in [("0°", (1.6, 0)), ("90°", (0, 1.4)), ("45°", (1.25, 1.2)), ("135°", (-1.35, 1.2))]:
        ax.text(dx, dy, t, ha="center", va="center", fontsize=FONT - 2)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-1.3, 1.8)
    return ax


# 1. Three windows: oriented, isotropic, corner ---------------------------------------------
def fig_windows():
    rng = np.random.default_rng(2)
    h, w = 120, 300
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    img = np.full((h, w), 110.0)
    a = np.radians(30)
    stripes = 110 + 60 * np.cos(2 * np.pi * (xx * np.cos(a) - yy * np.sin(a)) / 9)
    img[:, :100] = stripes[:, :100]
    blobs = blur(rng.standard_normal((h, w)), 2.0)
    img[:, 100:200] = 110 + 60 * blobs[:, 100:200] / blobs.std()
    sq = np.full((h, w), 70.0)
    sq[40:, 240:] = 190
    img[:, 200:] = sq[:, 200:]
    img = np.clip(img + 5 * rng.standard_normal((h, w)), 0, 255)
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    show(ax, img, vmin=0, vmax=255)
    for x0, lab in [(25, "one clear\norientation"), (125, "no preferred\norientation"), (215, "two\norientations")]:
        ax.add_patch(Rectangle((x0, 20), 50, 50, fill=False, ec=ORANGE, lw=3))
        ax.text(x0 + 25, 76, lab, ha="center", va="top", color="k", fontsize=FONT - 1,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
    save(fig, "fig-windows.png", dpi=130)


# 2. The entries of J on a square: before and after the window -----------------------------
def square_img():
    rng = np.random.default_rng(4)
    img = np.full((96, 96), 60.0)
    img[28:68, 28:68] = 180
    return img + 3 * rng.standard_normal(img.shape)


def fig_st_entries():
    img = square_img()
    J0 = structure_tensor(img, 1, 0)
    J = structure_tensor(img, 1, 3)
    fig, axs = plt.subplots(2, 3, figsize=(5.4, 4.0), gridspec_kw={"wspace": 0.06, "hspace": 0.12})
    for r, (JJ, lab) in enumerate([(J0, "per pixel"), (J, "ρ = 3")]):
        m = max(np.abs(JJ["jxx"]).max(), np.abs(JJ["jyy"]).max())
        show(axs[r, 0], JJ["jxx"], "Lₓ²" if r == 0 else "", cmap="magma", vmin=0, vmax=m)
        show(axs[r, 1], JJ["jxy"], "LₓL_y" if r == 0 else "", cmap="RdBu_r", vmin=-m / 2, vmax=m / 2)
        show(axs[r, 2], JJ["jyy"], "L_y²" if r == 0 else "", cmap="magma", vmin=0, vmax=m)
        axs[r, 0].set_ylabel(lab, fontsize=FONT)
        axs[r, 0].yaxis.set_visible(True); axs[r, 0].set_yticks([])
    axs[0, 1].set_title("LₓLᵧ"); axs[0, 2].set_title("Lᵧ²")
    print("  st entries: corner (28,28) per pixel mu2", round(J0["mu2"][28, 28], 2), "windowed mu1, mu2",
          round(J["mu1"][29, 29], 1), round(J["mu2"][29, 29], 1))
    save(fig, "fig-st-entries.png", dpi=120)


# 3. Orientation map on concentric rings --------------------------------------------------------
def fig_orient_map():
    rng = np.random.default_rng(5)
    n = 160
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    r = np.hypot(xx - 80, yy - 80)
    img = 110 + 60 * np.cos(2 * np.pi * r / 10)
    img[:, 120:] = 110 + 60 * blur(rng.standard_normal((n, n)), 1.5)[:, 120:] / 0.18
    img = np.clip(img + 15 * rng.standard_normal((n, n)), 0, 255)
    J = structure_tensor(img, 1, 3)
    rgb = orient_rgb(J)
    print("  orient map: coherence rings (80,40)", round(J["coh"][40, 80], 2), "noise texture (80,140)",
          round(J["coh"][80, 140], 2))
    ori = np.mod(-J["angle"] + np.pi / 2, np.pi) / np.pi
    en = J["mu1"] + J["mu2"]
    hue = hsv_to_rgb(ori, np.ones_like(ori), np.clip(en / np.percentile(en, 99), 0, 1))
    fig, axs = plt.subplots(2, 2, figsize=(6.8, 6.9), gridspec_kw={"wspace": 0.06, "hspace": 0.14})
    show(axs[0, 0], img, "rings | isotropic", vmin=0, vmax=255)
    show(axs[0, 1], hue, "orientation (hue)")
    show(axs[1, 0], J["coh"], "coherence (white = 1)", vmin=0, vmax=1)
    colour_wheel(axs[1, 1], axial=True)
    for t, (dx, dy) in [("0°", (1.3, 0)), ("90°", (0, 1.2)), ("45°", (1.05, 1.0)), ("135°", (-1.1, 1.0)),
                        ("180°", (-1.4, 0))]:
        axs[1, 1].text(dx, dy, t, ha="center", va="center", fontsize=FONT - 1)
    axs[1, 1].set_xlim(-1.8, 1.8); axs[1, 1].set_ylim(-1.5, 1.5)
    save(fig, "fig-orient-map.png", dpi=130)


# 4. Corners: Harris on shapes, and repeatability on the micrograph -----------------------------
def harris_r(J, k=0.05):
    return J["jxx"] * J["jyy"] - J["jxy"] ** 2 - k * (J["jxx"] + J["jyy"]) ** 2


def corners(R, n, border=8, r=4):
    mx = maximum_filter(R, size=2 * r + 1)
    c = np.argwhere((R == mx) & (R > 0))
    c = [(y, x) for y, x in c if border <= y < R.shape[0] - border and border <= x < R.shape[1] - border]
    c.sort(key=lambda p: -R[p])
    return c[:n]


def fig_corners():
    rng = np.random.default_rng(6)
    # shapes
    h = w = 128
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    img = np.full((h, w), 60.0)
    img[20:60, 16:64] = 180
    tri = (yy > 74) & (yy < 112) & (xx - 70 > (112 - yy) * 0.0) & (xx < 70 + (yy - 74) * 1.2)
    img[tri] = 160
    img[np.hypot(xx - 98, yy - 38) < 18] = 200
    img += 3 * rng.standard_normal((h, w))
    J = structure_tensor(img, 1, 2)
    R = harris_r(J)
    c = corners(R, 30, r=5)
    c = [p for p in c if R[p] > 0.1 * R.max()]
    print("  shapes corners", len(c))
    # micrograph and a rotated copy
    cimg = cell()
    crop = cimg[64:192, 64:192]
    rot = rotate(cimg, 25, reshape=False, order=1, mode="nearest")[64:192, 64:192]
    out = []
    for im in (crop, rot):
        Jc = structure_tensor(im, 1.5, 3)
        Rc = harris_r(Jc)
        out.append(corners(Rc, 25, border=10, r=6))
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"wspace": 0.06})
    show(axs[0], img, "Harris corners on shapes")
    for y, x in c:
        axs[0].plot(x, y, "o", ms=11, mfc="none", mec=ORANGE, mew=2.5)
    for ax, im, pts, t in [(axs[1], crop, out[0], f"micrograph: {len(out[0])} corners"),
                           (axs[2], rot, out[1], "the image rotated by 25°")]:
        show(ax, im, t)
        for y, x in pts:
            ax.plot(x, y, "o", ms=10, mfc="none", mec=ORANGE, mew=2.5)
    # how many of the corners are found again (within 2 px) after rotation
    th = np.radians(25)
    found = 0
    for y, x in out[0]:
        X, Y = x + 64 - 127.5, y + 64 - 127.5
        # scipy rotate: positive angle rotates the image counter-clockwise on screen
        xr = np.cos(th) * X + np.sin(th) * Y + 127.5 - 64
        yr = -np.sin(th) * X + np.cos(th) * Y + 127.5 - 64
        if any(np.hypot(xr - x2, yr - y2) <= 2 for y2, x2 in out[1]):
            found += 1
    print("  repeatability after rotation:", found, "of", len(out[0]))
    fig.text(0.98, 0.01, "Micrograph: R. M. Hunt, CC BY 3.0, Wikimedia Commons", ha="right", fontsize=FONT - 4,
             color=GREY)
    save(fig, "fig-corners.png", dpi=110)
    return found, len(out[0])


# 5. Profile across a bright line: Hessian vs structure tensor ---------------------------------
def fig_line_profile():
    x = np.arange(-20, 21)
    img = np.tile(40 + 120 * np.exp(-x ** 2 / (2 * 2.0 ** 2)), (41, 1))
    H = hessian(img, 2)
    J = structure_tensor(img, 1, 1)
    row = 20
    fig, ax = plt.subplots(figsize=(12.5, 3.0))
    nrm = lambda v: v / np.abs(v).max()
    ax.plot(x, nrm(img[row] - 40), color=GREY, lw=2, label="intensity across the line")
    ax.plot(x, nrm(-H["l2"][row]), color=BLUE, lw=2.6, label="Hessian: −λ₂ (σ = 2)")
    ax.plot(x, nrm(J["mu1"][row]), color=ORANGE, lw=2.6, ls="--", label="structure tensor: μ₁ (σ = 1, ρ = 1)")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlabel("x (pixels) across the line")
    ax.set_ylabel("scaled to max 1")
    ax.legend(frameon=False, loc="upper right", fontsize=FONT - 2)
    ax.set_xlim(-20, 34)
    print("  line profile: mu1 at centre / max", round(J["mu1"][row, 20] / J["mu1"][row].max(), 3))
    save(fig, "fig-line-profile.svg")


# 6. Orientation map of the micrograph -------------------------------------------------------------
def fig_orient_cell():
    img = cell()[40:216, 40:216]
    J = structure_tensor(img, 1.5, 4)
    rgb = orient_rgb(J)
    fig = plt.figure(figsize=(5.6, 3.1))
    a1 = fig.add_axes([0.0, 0.0, 0.42, 0.88]); a2 = fig.add_axes([0.44, 0.0, 0.42, 0.88])
    show(a1, img, "micrograph")
    show(a2, rgb, "orientation")
    wheel(fig, [0.88, 0.18, 0.34, 0.58])
    fig.text(0.0, -0.08, "Micrograph: R. M. Hunt, CC BY 3.0", fontsize=FONT - 4, color=GREY)
    save(fig, "fig-orient-cell.png", dpi=130)


if __name__ == "__main__":
    fig_windows()
    fig_st_entries()
    fig_orient_map()
    fig_corners()
    fig_line_profile()
    fig_orient_cell()
