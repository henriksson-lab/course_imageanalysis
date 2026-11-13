# Figures for zernike/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/zernike/figs.py
# Test images, Zernike moments and radial-polynomial errors come from the zernikebasis demo's
# code and lib/moments.ts (data.ts, via node); the aberration figure is computed here.
# fig-zernike.svg was made earlier by the day builder.
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "label-connectivity"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from labstyle import setup, fig, save, save_png, run_ts, BLUE, ORANGE, GREEN, RED, GREY, INK

setup()
OUT = "zernike"
FILL = "#a6c4f4"
D = run_ts(HERE)
S = D["S"]


def flat(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def zset(z):
    return [tuple(t) for t in z["nm"]], np.array(z["re"]) + 1j * np.array(z["im"])


def show(ax, img, title=None):
    ax.imshow(np.array(img).reshape(S, S), cmap="gray_r", vmin=0, vmax=1)
    ax.add_patch(Circle(((S - 1) / 2, (S - 1) / 2), S / 2, fill=False, ec=GREEN, lw=1.5, ls="--"))
    ax.set_xlim(-1, S); ax.set_ylim(S, -1); flat(ax)
    if title: ax.set_title(title, fontsize=18, pad=3)


def rot(P, a):
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]); return P @ R.T


# ---------------------------------------------------------------- 1. why a disc
f, axs = fig(400, 210, ncols=2, gridspec_kw=dict(wspace=0.15))
sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
a = np.radians(30)
ax = axs[0]
ax.add_patch(Polygon(sq, closed=True, fc=FILL, ec=BLUE, lw=2))
ax.add_patch(Polygon(rot(sq, a), closed=True, fill=False, ec=ORANGE, lw=2.5, ls="--"))
ax.set_title("square", fontsize=18, pad=3)
ax = axs[1]
ax.add_patch(Circle((0, 0), 1.15, fc=FILL, ec=BLUE, lw=2))
ax.add_patch(Circle((0, 0), 1.15, fill=False, ec=ORANGE, lw=2.5, ls="--"))
ax.set_title("disc", fontsize=18, pad=3)
for ax in axs:
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal"); flat(ax)
f.text(0.5, -0.04, "dashed: rotated by 30°", ha="center", fontsize=16, color="#555")
save(f, f"{OUT}/fig-squaredisc.svg")


# ---------------------------------------------------------------- 2. rotation: magnitudes stay, phases turn
nm, z0 = zset(D["z0"]); _, z6 = zset(D["z60"])
m0, m6 = np.abs(z0), np.abs(z6)
big = m0 > 0.01 * m0.max()
rel = np.abs(m0 - m6) / np.maximum(m0, m6)
k22 = nm.index((2, 2))
p0, p6 = np.degrees(np.angle(z0[k22])), np.degrees(np.angle(z6[k22]))
dphi = (p6 - p0 + 180) % 360 - 180
print(f"rotation: max rel. diff of |Z| (all) {rel.max():.4f}, (|Z| > 1% of max) {rel[big].max():.4f}; Z22 phase {p0:.1f} -> {p6:.1f} (change {dphi:.1f})")
f = plt.figure(figsize=(430 / 72, 470 / 72))
gs = f.add_gridspec(2, 2, height_ratios=[1, 0.95], hspace=0.3, wspace=0.08)
show(f.add_subplot(gs[0, 0]), D["blob0"], "blob")
show(f.add_subplot(gs[0, 1]), D["blob60"], "rotated by 60°")
ax = f.add_subplot(gs[1, :])
x = np.arange(len(nm))
ax.bar(x - 0.2, m0, width=0.4, color=BLUE, label="blob")
ax.bar(x + 0.2, m6, width=0.4, color=ORANGE, label="rotated")
ticks = [i for i, (n, m) in enumerate(nm) if m == 0]
ax.set_xticks(ticks); ax.set_xticklabels([f"{nm[i][0]}" for i in ticks], fontsize=17)
ax.set_xlabel("order n", fontsize=17, labelpad=2)
ax.set_ylabel("$|Z_{nm}|$"); ax.set_xlim(-0.8, len(nm) - 0.2)
ax.legend(loc="upper right", frameon=False, fontsize=17)
ax.text(0.98, 0.6, f"$Z_{{22}}$: phase {p0:.0f}° → {p6:.0f}°\n(turned by {dphi:.0f}° = −2 · 60°)".replace("-", "−"), transform=ax.transAxes,
        ha="right", va="top", fontsize=17)
save(f, f"{OUT}/fig-zrot.svg")


# ---------------------------------------------------------------- 3. six and nine
nm, zs = zset(D["zsix"]); _, zn = zset(D["znine"])
ms, mn = np.abs(zs), np.abs(zn)
print("six/nine: max |Z| difference", np.abs(ms - mn).max(), "relative to max", np.abs(ms - mn).max() / ms.max())
f = plt.figure(figsize=(390 / 72, 400 / 72))
gs = f.add_gridspec(2, 2, height_ratios=[1, 0.9], hspace=0.3, wspace=0.08)
show(f.add_subplot(gs[0, 0]), D["six"], "6")
show(f.add_subplot(gs[0, 1]), D["nine"], "9 = 6 turned 180°")
ax = f.add_subplot(gs[1, :])
x = np.arange(len(nm))
ax.bar(x - 0.2, ms, width=0.4, color=BLUE, label="6")
ax.bar(x + 0.2, mn, width=0.4, color=ORANGE, label="9")
ticks = [i for i, (n, m) in enumerate(nm) if m == 0]
ax.set_xticks(ticks); ax.set_xticklabels([f"{nm[i][0]}" for i in ticks], fontsize=17)
ax.set_xlabel("order n", fontsize=17, labelpad=2); ax.set_ylabel("$|Z_{nm}|$")
ax.set_xlim(-0.8, len(nm) - 0.2)
ax.legend(loc="upper right", frameon=False, fontsize=17, ncol=2)
save(f, f"{OUT}/fig-sixnine.svg")


# ---------------------------------------------------------------- 4. square pixels on a round domain; cancellation
f = plt.figure(figsize=(380 / 72, 500 / 72))
gs = f.add_gridspec(2, 1, height_ratios=[1.1, 1], hspace=0.28)
ax = f.add_subplot(gs[0])
n = 16
c = (n - 1) / 2
for y in range(n):
    for x in range(n):
        inside = np.hypot(x - c, y - c) <= n / 2
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc=FILL if inside else "white", ec="#bbb", lw=0.8))
ax.add_patch(Circle((c, c), n / 2, fill=False, ec=GREEN, lw=2.5))
ax.set_xlim(-0.7, n - 0.3); ax.set_ylim(n - 0.3, -0.7); ax.set_aspect("equal"); flat(ax)
cnt = D["discCount"]
ax.set_title(f"16 × 16 pixels in the disc\n(64 × 64: {cnt} of 4096, {100 * cnt / 4096:.0f} %)", fontsize=17, pad=4)
ax = f.add_subplot(gs[1])
r = np.array(D["rhos"])
for nn_, col in [("30", BLUE), ("40", ORANGE), ("50", RED)]:
    e = np.maximum(np.array(D["radial"][nn_]), 1e-18)
    ax.semilogy(r, e, color=col, lw=2.2, label=f"$R_{{{nn_},0}}$, max {e.max():.2g}".replace("e-07", " × 10⁻⁷"))
ax.axhline(1, color=GREY, lw=1, ls=":")
ax.set_xlim(0, 1); ax.set_ylim(1e-12, 1e6); ax.set_yticks([1e-12, 1e-8, 1e-4, 1]); ax.set_xticks([0, 0.5, 1])
ax.set_xlabel("ρ"); ax.set_ylabel("error, factorial\nformula", fontsize=17)
ax.legend(loc="upper left", frameon=False, fontsize=16, handlelength=1.0, borderaxespad=0.1)
print("radial maxima:", {k: max(v) for k, v in D["radial"].items()})
save(f, f"{OUT}/fig-discpixels.svg")


# ---------------------------------------------------------------- 5. aberrations: wavefront over the pupil and the resulting point spread function
M, R = 256, 40
yy, xx = np.mgrid[-M // 2:M // 2, -M // 2:M // 2]
rho = np.hypot(xx, yy) / R; th = np.arctan2(-yy, xx)
pupil = rho <= 1
ab = [("none", np.zeros_like(rho)),
      ("defocus $V_{20}$", 2 * rho**2 - 1),
      ("astigmatism $V_{22}$", rho**2 * np.cos(2 * th)),
      ("coma $V_{31}$", (3 * rho**3 - 2 * rho) * np.cos(th)),
      ("spherical $V_{40}$", 6 * rho**4 - 6 * rho**2 + 1)]
amp = 0.6       # waves (peak of the polynomial, |V| <= 1 on the disc)
f, axs = fig(540, 290, nrows=2, ncols=5, gridspec_kw=dict(wspace=0.06, hspace=0.08))
crop = 28
for j, (name, V) in enumerate(ab):
    Wf = np.where(pupil, amp * V, np.nan)
    a0 = axs[0, j]
    a0.imshow(Wf[M // 2 - R - 2:M // 2 + R + 2, M // 2 - R - 2:M // 2 + R + 2], cmap="RdBu_r", vmin=-amp, vmax=amp)
    a0.set_title(name.replace(" $", "\n$"), fontsize=16, pad=3)
    E = np.where(pupil, np.exp(2j * np.pi * amp * V), 0)
    I = np.abs(np.fft.fftshift(np.fft.fft2(E)))**2
    I /= I.max() if j else I.max()
    if j == 0: I0 = I.max()
    a1 = axs[1, j]
    a1.imshow(np.sqrt(I[M // 2 - crop:M // 2 + crop, M // 2 - crop:M // 2 + crop]), cmap="magma", vmin=0, vmax=1)
    for a_ in (a0, a1): flat(a_)
axs[0, 0].set_ylabel("wavefront", fontsize=17); axs[1, 0].set_ylabel("image of\na point", fontsize=17)
for a_ in (axs[0, 0], axs[1, 0]):
    a_.yaxis.label.set_visible(True)
save(f, f"{OUT}/fig-aberrations.svg")


# ---------------------------------------------------------------- 6. the pyramid of Zernike polynomials, n <= 4
from math import factorial
def Rnm(n, m, r):
    m = abs(m)
    return sum((-1)**k * factorial(n - k) / (factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)) * r**(n - 2 * k)
               for k in range((n - m) // 2 + 1))
g = np.linspace(-1, 1, 101)
GX, GY = np.meshgrid(g, -g)
rr, tt = np.hypot(GX, GY), np.arctan2(GY, GX)
FW, FH, T = 460, 500, 76
f = plt.figure(figsize=(FW / 72, FH / 72))
for n in range(5):
    for m in range(-n, n + 1, 2):
        cx, top = FW / 2 + m * 0.6 * T, 4 + n * 98
        ax = f.add_axes([(cx - T / 2) / FW, 1 - (top + T) / FH, T / FW, T / FH]); ax.axis("off")
        V = Rnm(n, m, rr) * (np.cos(m * tt) if m >= 0 else np.sin(-m * tt))
        ax.imshow(np.where(rr <= 1, V, np.nan), cmap="RdBu_r", vmin=-1, vmax=1)
        f.text(cx / FW, 1 - (top + T + 1) / FH, f"{n}, {m}".replace("-", "−"), ha="center", va="top", fontsize=19)
f.text(0.0, 1 - 20 / FH, "label: n, m\n(m < 0: sin |m|θ)", ha="left", va="top", fontsize=18, color="#555")
save(f, f"{OUT}/fig-zernike.svg")
