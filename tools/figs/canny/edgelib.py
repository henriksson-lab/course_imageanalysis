"""Shared helpers for the edges-day figure scripts (canny/, scale-space/, hessian/,
structure-tensor/). A small numpy port of demos/src/lib/edges.ts and imageops.ts with the
same kernel conventions (Gaussian cut at 4σ, derivative kernels normalised so that a ramp of
slope 1 gives 1 and a parabola x² gives 2; replicate border).

Imported by the scripts in tools/figs/<unit>/figs.py (they put this folder on sys.path).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

FONT = 17
plt.rcParams.update({
    "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
    "xtick.labelsize": FONT - 2, "ytick.labelsize": FONT - 2, "legend.fontsize": FONT - 2,
    "svg.fonttype": "path", "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.facecolor": "white",
})
# colour-blind-safe (Okabe-Ito / tab-like)
BLUE, ORANGE, GREEN, RED, PURPLE, CYAN, YELLOW, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#777777")

CELL_PATH = "demos/src/fourier2d/cell256.png"
CREDIT = "Micrograph: R. M. Hunt, CC BY 3.0, Wikimedia Commons"


def cell():
    a = np.asarray(Image.open(CELL_PATH).convert("L")).astype(float)
    return a


def gauss_kernel(s, order=0, radius=None):
    r = radius if radius is not None else max(1, int(np.ceil(4 * s)))
    x = np.arange(-r, r + 1, dtype=float)
    g = np.exp(-x * x / (2 * s * s))
    if order == 0:
        return g / g.sum()
    if order == 1:
        k = x * g
        return k / (k * x).sum()
    k = (x * x / (s * s) - 1) * g
    k = k - k.mean()
    return k * 2 / (k * x * x).sum()


def corr1d(f, k, axis=-1):
    """Correlate along an axis with replicate border (as correlate1d in edges.ts)."""
    f = np.asarray(f, float)
    r = len(k) // 2
    f = np.moveaxis(f, axis, -1)
    pad = [(0, 0)] * (f.ndim - 1) + [(r, r)]
    a = np.pad(f, pad, mode="edge")
    out = np.zeros_like(f)
    n = f.shape[-1]
    for i, w in enumerate(k):
        out += w * a[..., i:i + n]
    return np.moveaxis(out, -1, axis)


def sep(img, kx, ky):
    return corr1d(corr1d(img, kx, axis=1), ky, axis=0)


def gderiv(img, s, ox, oy):
    return sep(img, gauss_kernel(s, ox), gauss_kernel(s, oy))


def blur(img, s):
    return gderiv(img, s, 0, 0)


def gradient(img, s):
    gx, gy = gderiv(img, s, 1, 0), gderiv(img, s, 0, 1)
    return gx, gy, np.hypot(gx, gy)


def screen_angle(gx, gy):
    return np.arctan2(-gy, gx)


NMS_OFFSETS = [(1, 0), (1, -1), (0, 1), (1, 1)]


def quantise(gx, gy):
    a = np.degrees(screen_angle(gx, gy))
    a = np.where(a < 0, a + 180, a)
    return (np.round(a / 45).astype(int)) % 4


def nms(mag, gx, gy):
    h, w = mag.shape
    out = np.zeros_like(mag)
    q = quantise(gx, gy)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            m = mag[y, x]
            if m <= 0:
                continue
            dx, dy = NMS_OFFSETS[q[y, x]]
            a, b = mag[y + dy, x + dx], mag[y - dy, x - dx]
            if m > a and m >= b:
                out[y, x] = m
    return out


def hysteresis(v, lo, hi):
    """Returns (edges bool, strong mask, weak mask)."""
    from collections import deque
    h, w = v.shape
    strong = v >= hi
    weak = (v >= lo) & ~strong
    edges = strong.copy()
    dq = deque(zip(*np.nonzero(strong)))
    while dq:
        y, x = dq.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and not edges[yy, xx] and v[yy, xx] >= lo:
                    edges[yy, xx] = True
                    dq.append((yy, xx))
    return edges, strong, weak


def canny(img, s, lo, hi):
    gx, gy, mag = gradient(img, s)
    n = nms(mag, gx, gy)
    e, st, wk = hysteresis(n, lo, hi)
    return dict(gx=gx, gy=gy, mag=mag, nms=n, edges=e, strong=st, weak=wk)


def eig_sym2(a, b, c):
    """Eigenvalues l1 >= l2 by value, and the angle (image coords) of l1's eigenvector."""
    m = (a + c) / 2
    d = np.sqrt(((a - c) / 2) ** 2 + b * b)
    ang = 0.5 * np.arctan2(2 * b, a - c)
    return m + d, m - d, ang


def hessian(img, s, normalised=False):
    k = s * s if normalised else 1.0
    xx, xy, yy = k * gderiv(img, s, 2, 0), k * gderiv(img, s, 1, 1), k * gderiv(img, s, 0, 2)
    la, lb, _ = eig_sym2(xx, xy, yy)
    swap = np.abs(la) > np.abs(lb)
    l1 = np.where(swap, lb, la)
    l2 = np.where(swap, la, lb)
    return dict(xx=xx, xy=xy, yy=yy, l1=l1, l2=l2)


def frangi_value(l1, l2, beta=0.5, c=15, bright=True):
    l1 = np.asarray(l1, float); l2 = np.asarray(l2, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = np.where(l2 != 0, l1 / np.where(l2 == 0, 1, l2), 0)
    v = np.exp(-rb ** 2 / (2 * beta ** 2)) * (1 - np.exp(-(l1 ** 2 + l2 ** 2) / (2 * c * c)))
    bad = (l2 >= 0) if bright else (l2 <= 0)
    return np.where(bad, 0.0, v)


def frangi(img, sigmas, beta=0.5, c=15, bright=True):
    best = np.zeros_like(img, dtype=float)
    arg = np.full(img.shape, -1)
    for k, s in enumerate(sigmas):
        H = hessian(img, s, True)
        v = frangi_value(H["l1"], H["l2"], beta, c, bright)
        arg = np.where(v > best, k, arg)
        best = np.maximum(best, v)
    return best, arg


def structure_tensor(img, s, rho):
    gx, gy, _ = gradient(img, s)
    jxx, jxy, jyy = gx * gx, gx * gy, gy * gy
    if rho > 0:
        jxx, jxy, jyy = blur(jxx, rho), blur(jxy, rho), blur(jyy, rho)
    m1, m2, ang = eig_sym2(jxx, jxy, jyy)
    m1, m2 = np.maximum(m1, 0), np.maximum(m2, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.where(m1 + m2 > 0, ((m1 - m2) / np.where(m1 + m2 > 0, m1 + m2, 1)) ** 2, 0)
    return dict(gx=gx, gy=gy, jxx=jxx, jxy=jxy, jyy=jyy, mu1=m1, mu2=m2, angle=ang, coh=coh)


def geom_scales(s0, s1, n):
    return [s0 * (s1 / s0) ** (k / (n - 1)) for k in range(n)]


def hsv_to_rgb(h, s, v):
    import matplotlib.colors as mc
    return mc.hsv_to_rgb(np.stack([h, s, v], -1))


def colour_wheel(ax, n=200, axial=False):
    """Draw a hue wheel (counter-clockwise on screen from +x). axial: hue doubles (orientation)."""
    yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    a = np.arctan2(-yy, xx)
    if axial:
        a = np.mod(a, np.pi) / np.pi
    else:
        a = np.mod(a, 2 * np.pi) / (2 * np.pi)
    r = np.hypot(xx, yy)
    rgb = hsv_to_rgb(a, np.ones_like(a), np.ones_like(a))
    rgba = np.concatenate([rgb, (r <= 1)[..., None].astype(float)], -1)
    ax.imshow(rgba, extent=[-1, 1, -1, 1])
    ax.set_axis_off()


def show(ax, img, title=None, cmap="gray", **kw):
    ax.imshow(img, cmap=cmap, interpolation="nearest", **kw)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=FONT)
