"""Slide figures for two-view/lec.qmd (fig-epipolar.svg and fig-depth-error.svg come from
tools/figs/multiview/figs.py).

Run from the repository root:
    python3 -I tools/figs/two-view/figs.py
Writes fig-*.svg into two-view/. The matching figures use the course micrograph
(demos/src/fourier2d/cell256.png, Robert M. Hunt, CC BY 3.0) and a synthetic second view of it
(rotated, shifted, brightness changed, noise; fixed seed); everything else is synthetic.
"""
import math
import os
import sys
import numpy as np
from matplotlib.patches import Polygon, ConnectionPatch
from scipy import ndimage as ndi
from skimage.feature import corner_harris, corner_peaks

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../multiview"))
from common import plt, px, save, cell256, BLUE, ORANGE, GREEN, RED, PURPLE, GREY, SKY  # noqa: E402

U = "two-view"
rng = np.random.default_rng(7)

# ---------------------------------------------------------------------------------------------
# A second view of the micrograph: rotate by 8° about the centre, shift by (14, −9) px,
# brightness 0.8 I + 0.1, Gaussian noise σ = 0.02. Ground truth: the same similarity.
I1 = cell256()
ang, shift = math.radians(8), np.array([14.0, -9.0])
c = np.array([127.5, 127.5])
Rm = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])


def to2(p):            # (row, col) in image 1 → (row, col) in image 2
    return (Rm @ (p - c).T).T + c + shift


Rinv = Rm.T
rr, cc = np.mgrid[0:256, 0:256]
src = (Rinv @ (np.c_[rr.ravel(), cc.ravel()] - c - shift).T).T + c
I2 = ndi.map_coordinates(I1, [src[:, 0], src[:, 1]], order=1, cval=0.5).reshape(256, 256)
I2 = np.clip(0.8 * I2 + 0.1 + 0.02 * rng.standard_normal(I2.shape), 0, 1)

R = 7     # descriptor window 15 × 15


def corners(img, n=70):
    H = corner_harris(ndi.gaussian_filter(img, 1.0), sigma=2)
    return corner_peaks(H, min_distance=7, num_peaks=n, exclude_border=R + 2)


def describe(img, pts):
    D = []
    for r, q in pts:
        p = img[r - R:r + R + 1, q - R:q + R + 1].ravel().astype(float)
        p = p - p.mean()
        D.append(p / (np.linalg.norm(p) + 1e-9))
    return np.array(D)


k1, k2 = corners(I1), corners(I2)
d1, d2 = describe(I1, k1), describe(I2, k2)
dist = np.sqrt(np.maximum(0, 2 - 2 * d1 @ d2.T))      # unit vectors: ‖a − b‖² = 2 − 2 a·b
nn = dist.argmin(1)
srt = np.sort(dist, 1)
ratio = srt[:, 0] / srt[:, 1]
back = dist.argmin(0)
correct = np.linalg.norm(to2(k1.astype(float)) - k2[nn], axis=1) < 3
keep = (ratio < 0.8) & (back[nn] == np.arange(len(k1)))
print(f"corners {len(k1)} / {len(k2)}; nearest neighbour: {correct.sum()} correct of {len(k1)}; "
      f"ratio + cross-check: {keep.sum()} kept, {(keep & ~correct).sum()} wrong")


# 1. interest points and one descriptor ---------------------------------------------------------
fig = plt.figure(figsize=px(440, 500))
ax0 = fig.add_axes([0.0, 0.42, 1.0, 0.58])
ax0.imshow(I1, cmap="gray", vmin=0, vmax=1)
ax0.plot(k1[:, 1], k1[:, 0], "o", ms=6, mfc="none", mec=ORANGE, mew=1.8)
sel = int(np.argmin(np.linalg.norm(k1 - np.array([150, 100]), axis=1)))
r0, q0 = k1[sel]
ax0.add_patch(Polygon([(q0 - R, r0 - R), (q0 + R, r0 - R), (q0 + R, r0 + R), (q0 - R, r0 + R)], closed=True, fill=False, ec=SKY, lw=2.5))
ax0.set_title(f"{len(k1)} corners (structure tensor)", fontsize=20)
ax0.axis("off")
ax1 = fig.add_axes([0.02, 0.02, 0.36, 0.3])
patch = I1[r0 - R:r0 + R + 1, q0 - R:q0 + R + 1]
ax1.imshow(patch, cmap="gray", vmin=0, vmax=1)
ax1.set_title("window", fontsize=20)
ax1.set_xticks([]); ax1.set_yticks([])
for s_ in ax1.spines.values():
    s_.set_color(SKY); s_.set_linewidth(2.5); s_.set_visible(True)
ax2 = fig.add_axes([0.52, 0.08, 0.46, 0.24])
gy, gx = np.gradient(ndi.gaussian_filter(patch, 0.8))
th = (np.degrees(np.arctan2(gy, gx)) + 360) % 360
h, _ = np.histogram(th, bins=8, range=(0, 360), weights=np.hypot(gx, gy))
ax2.bar(np.arange(8) * 45 + 22.5, h / h.max(), width=40, color=BLUE)
ax2.set_xticks([0, 180, 360]); ax2.set_yticks([])
ax2.set_xlabel("gradient direction (°)", fontsize=20)
ax2.set_title("descriptor", fontsize=20)
save(fig, U, "fig-interest.svg")


# 2. matches: nearest neighbour alone vs ratio test + cross-check ---------------------------------
def draw_pair(ax_l, ax_r, idx, title):
    for ax, img in ((ax_l, I1), (ax_r, I2)):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    for i in idx:
        ok = correct[i]
        con = ConnectionPatch(xyA=(k1[i, 1], k1[i, 0]), coordsA=ax_l.transData, xyB=(k2[nn[i], 1], k2[nn[i], 0]), coordsB=ax_r.transData,
                              color=GREEN if ok else RED, lw=1.6 if ok else 2.2, ls="-" if ok else (0, (4, 2)))
        ax_r.add_artist(con)
    nw = int((~correct[idx]).sum())
    ax_l.text(256 + 4, -12, title + f": {len(idx)} matches, {nw} wrong", ha="center", va="bottom", fontsize=20)


fig, axs = plt.subplots(1, 4, figsize=px(1120, 300), gridspec_kw={"wspace": 0.04})
draw_pair(axs[0], axs[1], np.arange(len(k1)), "nearest neighbour")
draw_pair(axs[2], axs[3], np.where(keep)[0], "ratio < 0.8 + cross-check")
fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.01)
save(fig, U, "fig-matches.svg")

# 3. epipolar lines for three motions (image 2, f = 700 px, 640 × 480) ----------------------------
f, W, H = 700.0, 640, 480
K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])


def roty(a):
    c_, s_ = math.cos(a), math.sin(a)
    return np.array([[c_, 0, s_], [0, 1, 0], [-s_, 0, c_]])


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def lines_in_image2(Rr, t, pts1):
    E = skew(t) @ Rr
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    return [F @ np.array([u, v, 1.0]) for u, v in pts1], F


pts1 = [(u, v) for u in (110, 320, 530) for v in (80, 240, 400)]
cases = [("sideways", np.eye(3), np.array([1.0, 0, 0])),
         ("forward", np.eye(3), np.array([0, 0, 1.0])),
         ("general", roty(math.radians(-10)), roty(math.radians(-10)) @ np.array([math.cos(math.radians(20)), 0.15, math.sin(math.radians(20))]))]
fig, axs = plt.subplots(1, 3, figsize=px(1120, 300))
for ax, (name, Rr, t) in zip(axs, cases):
    ls, F = lines_in_image2(Rr, t, pts1)
    ax.add_patch(Polygon([(0, 0), (W, 0), (W, H), (0, H)], closed=True, fc="#fafafa", ec="#666", lw=1.5))
    xs = np.array([-2000, 3000.0])
    for (a, b, c_) in ls:
        if abs(b) > 1e-9:
            ax.plot(xs, -(a * xs + c_) / b, color=GREEN, lw=2)
        else:
            ax.plot([-c_ / a] * 2, [-2000, 3000], color=GREEN, lw=2)
    e = np.linalg.svd(F.T)[2][-1]                # F^T e' = 0
    title = name
    if abs(e[2]) > 1e-9 * np.abs(e).max():
        e = e[:2] / e[2]
        if 0 <= e[0] <= W and 0 <= e[1] <= H:
            ax.plot(*e, "s", color=PURPLE, ms=13)
            ax.text(e[0] + 40, e[1] - 50, "epipole", color=PURPLE, fontsize=19, bbox=dict(fc="white", ec="none", pad=1))
        else:
            ax.annotate("", xy=(W - 5, np.clip(e[1], 20, H - 20)), xytext=(W - 110, np.clip(e[1], 20, H - 20)),
                        arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=3, mutation_scale=22))
            ax.text(W - 120, np.clip(e[1], 20, H - 20) - 70, f"epipole far right,\nx ≈ {e[0]:.0f}", color=PURPLE, fontsize=19, ha="right", va="bottom", bbox=dict(fc="white", ec="none", pad=1))
    else:
        ax.text(W / 2, H - 20, "epipole at infinity", color=PURPLE, fontsize=19, ha="center", va="bottom", bbox=dict(fc="white", ec="none", pad=1))
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title)
fig.tight_layout(w_pad=1.0)
save(fig, U, "fig-three-motions.svg")

# 4. RANSAC iterations against the inlier fraction --------------------------------------------------
w = np.linspace(0.3, 0.97, 300)
fig, ax = plt.subplots(figsize=px(470, 420))
for s, col, lab in ((8, RED, "eight-point (s = 8)"), (7, ORANGE, "seven-point (s = 7)"), (5, BLUE, "five-point (s = 5)")):
    N = np.log(0.01) / np.log(1 - w ** s)
    ax.plot(w, N, color=col, lw=2.5, label=lab)
    for ww in (0.5, 0.8):
        n = math.ceil(math.log(0.01) / math.log(1 - ww ** s))
        ax.plot([ww], [n], "o", color=col, ms=8)
        dy = {8: 1.35, 7: 1.0, 5: 0.62}[s] if ww == 0.8 else 1.12
        ax.text(ww - 0.02, n * dy, str(n), color=col, fontsize=20, va="center", ha="right")
ax.set_yscale("log"); ax.set_ylim(3, 3e4)
ax.set_xlabel("fraction of correct matches w"); ax.set_ylabel("iterations N (p = 0.99)")
ax.legend(frameon=False, loc="upper right")
save(fig, U, "fig-ransac-iters.svg")


# 5. no epipolar geometry: pure rotation and a tiny baseline (top views) ---------------------------
def cam(ax, c_, d, col, size=0.42):
    """Camera glyph: centre dot and a frustum opening in the viewing direction, image plane drawn thick."""
    d = np.array(d, float); d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    a, b = c_ + size * (d + 0.75 * n), c_ + size * (d - 0.75 * n)
    ax.add_patch(Polygon([c_, a, b], closed=True, fc=col, ec=col, alpha=0.25, lw=0, zorder=4))
    ax.plot([c_[0], a[0]], [c_[1], a[1]], color=col, lw=1.5, zorder=5)
    ax.plot([c_[0], b[0]], [c_[1], b[1]], color=col, lw=1.5, zorder=5)
    ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=4, zorder=5, solid_capstyle="butt")
    ax.plot(*c_, "o", color=col, ms=6, zorder=6)


fig, axs = plt.subplots(1, 2, figsize=px(450, 400))
ax = axs[0]
C0 = np.array([0.0, 0.0])
cam(ax, C0, (-0.35, 1), BLUE); cam(ax, C0, (0.45, 1), ORANGE, 0.32)
for X_ in (np.array([0.5, 1.5]), np.array([1.0, 3.0])):
    ax.plot(*X_, "o", color=RED, ms=10)
ax.plot([0, 1.25], [0, 3.75], color=GREY, lw=2)
ax.text(0.1, -0.3, "one centre,\ntwo directions", fontsize=20, va="top", ha="center")
ax.set_title("pure rotation", fontsize=20)
ax.set_xlim(-0.9, 1.4); ax.set_ylim(-1.5, 4.0); ax.set_aspect("equal"); ax.axis("off")
ax = axs[1]
Ca, Cb, X_ = np.array([-0.08, 0.0]), np.array([0.08, 0.0]), np.array([0.2, 3.2])
for cc_, col in ((Ca, BLUE), (Cb, ORANGE)):
    d = X_ - cc_
    for sgn in (-1, 1):
        dd = d + sgn * np.array([d[1], -d[0]]) * 0.025
        ax.plot([cc_[0], cc_[0] + 1.25 * dd[0]], [cc_[1], cc_[1] + 1.25 * dd[1]], color=col, lw=1.4)
    cam(ax, cc_, d, col, 0.32)
ax.plot(*X_, "o", color=RED, ms=10)
ax.annotate("", xy=(0.27, 4.1), xytext=(0.13, 2.4), arrowprops=dict(arrowstyle="<->", color=RED, lw=2.5))
ax.text(0.15, -0.3, "depth\nuncertain", color=RED, fontsize=20, va="top", ha="center")
ax.set_title("tiny baseline", fontsize=20)
ax.set_xlim(-0.9, 1.2); ax.set_ylim(-1.5, 4.4); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout(w_pad=0.5)
save(fig, U, "fig-degenerate.svg")


# 6. the four candidate poses from E (top view; HZ fig. 9.12 in flatland) ---------------------------
def meet(o1, d1, o2, d2):
    A = np.array([d1, -d2]).T
    s, t = np.linalg.solve(A, o2 - o1)
    return s, t


A0, dA = np.array([0.0, 0.0]), np.array([0.5, 1.0])
B0 = np.array([1.2, 0.0])
X0 = A0 + 2.0 * dA
dB = X0 - B0
cands = [("true pose", B0, (0, 1), dB),
         ("baseline reversed", -B0, (0, 1), dB),
         ("camera 2 turned", B0, (0, -1), np.array([dB[0], -dB[1]])),
         ("both", -B0, (0, -1), np.array([dB[0], -dB[1]]))]
fig, axs = plt.subplots(1, 4, figsize=px(1120, 300))
for ax, (title, Bc, look, d) in zip(axs, cands):
    s, t = meet(A0, dA, Bc, d)
    Xc = A0 + s * dA
    ok = s > 0 and t > 0
    for o, dd, par, col in ((A0, dA, s, BLUE), (Bc, d, t, ORANGE)):
        dd = dd / np.linalg.norm(dd)
        L = abs(par) * np.linalg.norm(d if col == ORANGE else dA) + 1.0
        ax.plot([o[0], o[0] + L * dd[0] * np.sign(par)], [o[1], o[1] + L * dd[1] * np.sign(par)], color=col, lw=2.2,
                ls="-" if par > 0 else ":")
    cam(ax, A0, (0, 1), BLUE, 0.6); cam(ax, Bc, look, ORANGE, 0.6)
    ax.plot(*Xc, "o", color=GREEN if ok else RED, ms=12)
    ax.text(0.5, -0.02, "in front of both" if ok else "behind a camera", transform=ax.transAxes, ha="center", va="top",
            color=GREEN if ok else RED, fontsize=19)
    ax.set_title(title, fontsize=20)
    ax.set_xlim(-2.4, 2.6); ax.set_ylim(-3.7, 3.7); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout(w_pad=0.5)
save(fig, U, "fig-four-poses.svg")

# 7. triangulation: two noisy rays miss; midpoint of the shortest segment -----------------------------
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401

c1, c2 = np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])
Xt = np.array([1.1, 0.0, 3.0])
d1 = (Xt - c1) + np.array([0, 0.12, 0]); d1 /= np.linalg.norm(d1)
d2 = (Xt - c2) + np.array([0, -0.12, 0]); d2 /= np.linalg.norm(d2)
A = np.array([[d1 @ d1, -d1 @ d2], [d1 @ d2, -d2 @ d2]])
bvec = np.array([(c2 - c1) @ d1, (c2 - c1) @ d2])
s, t = np.linalg.solve(A, bvec)
p1, p2 = c1 + s * d1, c2 + t * d2
mid = (p1 + p2) / 2
fig = plt.figure(figsize=px(480, 440))
ax = fig.add_subplot(projection="3d")
for cc_, dd, col, lab in ((c1, d1, BLUE, "ray 1"), (c2, d2, ORANGE, "ray 2")):
    e = cc_ + 4.2 * dd
    ax.plot(*np.c_[cc_, e], color=col, lw=2.5)
    ax.scatter(*cc_, color=col, s=60)
ax.plot(*np.c_[p1, p2], color=GREY, lw=2, ls="--")
ax.scatter(*mid, color=RED, s=90, depthshade=False)
ax.text(*(mid + np.array([0.15, 0.0, 0.25])), "midpoint", color=RED, fontsize=19)
ax.text(*(c1 + np.array([-0.2, 0, -0.45])), "C", fontsize=20)
ax.text(*(c2 + np.array([0.1, 0, -0.45])), "C′", fontsize=20)
ax.text2D(0.02, 0.95, "dashed: shortest segment\nbetween the rays", transform=ax.transAxes, color="#444", fontsize=18, va="top")
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.view_init(elev=18, azim=-62)
ax.set_box_aspect((2.2, 1.2, 3.6))
ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
save(fig, U, "fig-triangulation.svg")
