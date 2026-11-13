"""Static figures for the optical flow day (print fallback of the demos).

Run from the repository root:  python3 -I tools/figs/opticalflow/figs.py
Writes PNGs into flow-constraint/, lucas-kanade/, horn-schunck/, nonrigid-reg/.
The algorithms are small numpy versions of demos/src/lib/flow.ts and deform.ts
(same definitions; numbers in the exercises come from the TypeScript code and the
scratch scripts, not from these figures).
"""
import colorsys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "figure.dpi": 110})
CELL = np.asarray(Image.open("demos/src/fourier2d/cell256.png")).astype(float)
CROP = CELL[64:192, 64:192]
CREDIT = "Micrograph: R. M. Hunt, CC BY 3.0, Wikimedia Commons"


def gauss_kernel(s, order=0):
    r = max(1, int(np.ceil(4 * s)))
    x = np.arange(-r, r + 1)
    g = np.exp(-x ** 2 / (2 * s * s))
    if order == 0:
        return g / g.sum()
    k = x * g
    return k / (k * x).sum()


def sep(img, kx, ky):
    from numpy.lib.stride_tricks import sliding_window_view as sw
    rx, ry = len(kx) // 2, len(ky) // 2
    a = np.pad(img, ((0, 0), (rx, rx)), mode="edge")
    a = (sw(a, len(kx), axis=1) * kx).sum(-1)
    a = np.pad(a, ((ry, ry), (0, 0)), mode="edge")
    return (sw(a, len(ky), axis=0) * ky).sum(-1)


def blur(img, s):
    k = gauss_kernel(s)
    return sep(img, k, k)


def bilinear(img, x, y):
    h, w = img.shape
    x = np.clip(x, 0, w - 1); y = np.clip(y, 0, h - 1)
    x0 = np.minimum(np.floor(x).astype(int), w - 2); y0 = np.minimum(np.floor(y).astype(int), h - 2)
    fx, fy = x - x0, y - y0
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x0 + 1]
            + (1 - fx) * fy * img[y0 + 1, x0] + fx * fy * img[y0 + 1, x0 + 1])


def flow_rgb(u, v, vmax):
    h, w = u.shape
    out = np.zeros((h, w, 3))
    ang = (np.degrees(np.arctan2(-v, u)) + 360) % 360
    sat = np.clip(np.hypot(u, v) / vmax, 0, 1)
    for i in range(h):
        for j in range(w):
            out[i, j] = colorsys.hsv_to_rgb(ang[i, j] / 360, sat[i, j], 1)
    return out


def derivs(I1, I2, s=1.0):
    k0, k1 = gauss_kernel(s), gauss_kernel(s, 1)
    m = 0.5 * (I1 + I2)
    return sep(m, k1, k0), sep(m, k0, k1), sep(I2 - I1, k0, k0)


def wheel_inset(ax, vmax):
    n = 61
    yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    rgb = flow_rgb(xx * vmax, yy * vmax, vmax)
    rgb[np.hypot(xx, yy) > 1] = 1
    ax.imshow(rgb); ax.axis("off")


# ---------------------------------------------------------------- unit 1
def fig_flowfield():
    n = 33
    y, x = np.mgrid[0:n, 0:n].astype(float)
    c = (n - 1) / 2
    fields = {"rotation": (0.1 * (y - c), -0.1 * (x - c)), "expansion": (0.1 * (x - c), 0.1 * (y - c)),
              "two cells moving apart": (2 * (np.exp(-((x - c - 7) ** 2 + (y - c) ** 2) / 50) - np.exp(-((x - c + 7) ** 2 + (y - c) ** 2) / 50)), 0 * x)}
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2), gridspec_kw={"width_ratios": [1, 1, 1, 0.6]})
    for col, (name, (u, v)) in enumerate(fields.items()):
        vmax = np.hypot(u, v).max()
        a = axes[0, col]
        s = 4
        a.quiver(x[::s, ::s], y[::s, ::s], u[::s, ::s], v[::s, ::s], angles="xy", scale_units="xy", scale=vmax / 3.5, color="#1f4e9a", width=0.012)
        a.set_xlim(-1, n); a.set_ylim(n, -1); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.set_title(name)
        axes[1, col].imshow(flow_rgb(u, v, vmax)); axes[1, col].axis("off")
    axes[0, 3].axis("off")
    axes[0, 3].text(0, 0.5, "arrows:\nlength = speed\n\ncolour:\nhue = direction\nsaturation = speed", fontsize=13, va="center")
    wheel_inset(axes[1, 3], 1)
    axes[1, 3].set_title("legend", fontsize=13)
    fig.tight_layout()
    fig.savefig("flow-constraint/fig-flowfield.png"); plt.close(fig)


def fig_bump():
    f1 = np.array([0, 0, 1, 4, 1, 0, 0.0]); f2 = np.array([0, 0, 0, 1, 4, 1, 0.0])
    x = np.arange(7)
    ix = np.gradient(f1); it = f2 - f1
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(np.abs(ix) > 1e-9, -it / ix, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(x, f1, "o-", lw=3, label="frame 1"); axes[0].plot(x, f2, "s--", lw=3, label="frame 2")
    axes[0].set_xlabel("x"); axes[0].set_title("a bump moves 1 px"); axes[0].legend()
    axes[1].bar(x - 0.18, ix, 0.36, label="$I_x$"); axes[1].bar(x + 0.18, it, 0.36, label="$I_t$")
    axes[1].axhline(0, color="k", lw=0.8); axes[1].set_xlabel("x"); axes[1].legend(); axes[1].set_title("slope and change")
    axes[2].plot(x[1:6], u[1:6], "o", ms=10, label="$-I_t/I_x$ per pixel")
    axes[2].axhline(1, color="g", ls="--", lw=2, label="window: 8.5 / 8.5 = 1")
    axes[2].text(3, 0.2, "undefined", ha="center", fontsize=12)
    axes[2].set_ylim(-0.3, 2.4); axes[2].set_xlabel("x"); axes[2].legend(fontsize=11, loc="upper left"); axes[2].set_title("estimates of u")
    fig.tight_layout(); fig.savefig("flow-constraint/fig-bump.png"); plt.close(fig)


def fig_kymograph():
    L, T = 128, 40
    x = np.arange(L)
    kymo = np.zeros((T, L))
    for t in range(T):
        p = 30.0 + np.zeros(L)
        for c, k in [(10, 1.5), (40, 0.5), (70, 2.5), (110, -1)]:
            p += 120 * np.exp(-((x - k * t - c) ** 2) / (2 * 2.2 ** 2))
        kymo[t] = p
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.imshow(kymo, cmap="gray", aspect="auto")
    ax.plot([40, 40 + 0.5 * 30], [0, 30], color="#ffd400", lw=2.5)
    ax.text(48, 34, "slope 0.5 px/frame", color="#ffd400", fontsize=13)
    ax.set_xlabel("position x (pixels)"); ax.set_ylabel("time (frames) ↓")
    ax.set_title("kymograph: each moving spot draws a line")
    fig.tight_layout(); fig.savefig("flow-constraint/fig-kymograph.png"); plt.close(fig)


def constraint_panel(ax, cons, truth, est=None, lim=3, title=""):
    for a, b, c in cons:
        if abs(b) > abs(a):
            us = np.array([-lim * 2, lim * 2]); vs = -(a * us + c) / b
        else:
            vs = np.array([-lim * 2, lim * 2]); us = -(b * vs + c) / a
        ax.plot(us, -vs, color="#1f77b4", lw=1.3, alpha=0.6)
    ax.plot(truth[0], -truth[1], "o", color="#2ca02c", ms=12, label="true")
    if est is not None:
        ax.plot(est[0], -est[1], "o", color="#d62728", ms=8, label="least squares")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("u"); ax.set_ylabel("v (up = −v)"); ax.set_title(title); ax.grid(alpha=0.3)


def pattern_constraints(P, u, v, r=14, step=2):
    cons = []
    for yy in range(-r, r + 1, step):
        for xx in range(-r, r + 1, step):
            if xx * xx + yy * yy > r * r:
                continue
            Ix = (P(xx + 1, yy) - P(xx - 1, yy)) / 2
            Iy = (P(xx, yy + 1) - P(xx, yy - 1)) / 2
            It = P(xx - u / 2, yy - v / 2) - P(xx + u / 2, yy + v / 2)
            if Ix * Ix + Iy * Iy > 4:
                cons.append((Ix, Iy, It))
    return cons


def lsq(cons):
    A = np.array([[a, b] for a, b, _ in cons]); bb = -np.array([c for *_, c in cons])
    return np.linalg.lstsq(A, bb, rcond=0.03)[0]   # minimum-norm point when all lines are parallel


def fig_aperture():
    sm = lambda t: 1 / (1 + np.exp(-t))
    bar = lambda x, y: 50 + 150 * sm(((x * np.sin(np.pi / 3) + y * np.cos(np.pi / 3)) + 6) / 1.2) * sm((6 - (x * np.sin(np.pi / 3) + y * np.cos(np.pi / 3))) / 1.2)
    corner = lambda x, y: 50 + 150 * sm((x + 4) / 1.2) * sm((4 - y) / 1.2)
    u, v = 1.5, -1.0
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for k, (P, name, uv) in enumerate([(bar, "an edge", (u, v)), (corner, "a corner", (1.0, 1.0))]):
        g = np.linspace(-24, 24, 97)
        X, Y = np.meshgrid(g, g)
        img = P(X, Y); img[X ** 2 + Y ** 2 > 14 ** 2] = 75
        ax = axes[2 * k]
        ax.imshow(img, cmap="gray", extent=(-24, 24, 24, -24), vmin=0, vmax=255)
        ax.add_patch(plt.Circle((0, 0), 14, fill=False, ec="#ffd400", ls="--", lw=1.5))
        cons = pattern_constraints(P, *uv)
        for a, b, c in cons[::6]:
            pass
        for yy in range(-12, 13, 6):
            for xx in range(-12, 13, 6):
                if xx * xx + yy * yy > 13 ** 2:
                    continue
                Ix = (P(xx + 1, yy) - P(xx - 1, yy)) / 2; Iy = (P(xx, yy + 1) - P(xx, yy - 1)) / 2
                It = P(xx - uv[0] / 2, yy - uv[1] / 2) - P(xx + uv[0] / 2, yy + uv[1] / 2)
                g2 = Ix * Ix + Iy * Iy
                if g2 > 4:
                    ax.arrow(xx, yy, -3 * It * Ix / g2, -3 * It * Iy / g2, color="#ffd400", width=0.35, head_width=1.6, length_includes_head=True)
        ax.set_title(f"{name} through a hole"); ax.axis("off")
        constraint_panel(axes[2 * k + 1], cons, uv, lsq(cons), title="velocity space")
        if k == 0:
            axes[1].legend(loc="lower right", fontsize=10)
    fig.tight_layout(); fig.savefig("flow-constraint/fig-aperture.png"); plt.close(fig)


# ---------------------------------------------------------------- unit 2
def fig_lk_windows():
    # a synthetic scene: a bright rectangle on a flat background (edges and corners), slightly blurred
    n = 96
    Y, X = np.mgrid[0:n, 0:n].astype(float)
    sm = lambda t: 1 / (1 + np.exp(-t))
    P = lambda x, y: 60 + 140 * sm((x - 30) / 1.5) * sm((70 - x) / 1.5) * sm((y - 30) / 1.5) * sm((70 - y) / 1.5)
    true = (1.0, -0.5)
    I1, I2 = P(X - 0 * true[0], Y), P(X - true[0], Y - true[1])
    ix, iy, it = derivs(I1, I2, 1.0)
    spots = [((14, 82), "flat"), ((30, 50), "edge"), ((30, 30), "corner")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(I1, cmap="gray", vmin=0, vmax=255)
    for (x, y), name in spots:
        axes[0].add_patch(plt.Circle((x, y), 6, fill=False, ec="#ffd400", lw=2))
        axes[0].text(x - 6, y - 9, name, color="#ffd400", fontsize=13, fontweight="bold")
    axes[0].axis("off"); axes[0].set_title("three windows (radius 6)")
    for k, ((x, y), name) in enumerate(spots):
        cons = [(ix[y + dy, x + dx], iy[y + dy, x + dx], it[y + dy, x + dx]) for dy in range(-6, 7) for dx in range(-6, 7) if dx * dx + dy * dy <= 36]
        A = np.array(cons)
        J = np.array([[np.mean(A[:, 0] ** 2), np.mean(A[:, 0] * A[:, 1])], [np.mean(A[:, 0] * A[:, 1]), np.mean(A[:, 1] ** 2)]])
        mu = np.linalg.eigvalsh(J)[::-1]
        strong = [c for c in cons if c[0] ** 2 + c[1] ** 2 > 1]
        est = lsq(strong) if strong else None
        constraint_panel(axes[k + 1], strong[::2], true, est, title=f"{name}: μ₁ = {mu[0]:.1f}, μ₂ = {mu[1]:.1f}")
        if not strong:
            axes[k + 1].text(0, 0.5, "no lines:\nno gradient", ha="center", fontsize=14)
    axes[3].legend(loc="lower right", fontsize=10)
    fig.tight_layout(); fig.savefig("lucas-kanade/fig-lk-windows.png"); plt.close(fig)


def fig_pyramid():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    img = CROP.copy()
    for l in range(4):
        axes[l].imshow(img, cmap="gray")
        h = img.shape[0]
        c = h / 2
        d = 8 / 2 ** l * h / 128 * (128 / h) if False else 8 / 2 ** l
        axes[l].arrow(c - d / 2, c, d, 0, color="#ff7f0e", width=h / 60, head_width=h / 18, length_includes_head=True)
        axes[l].set_title(f"level {l}: {h}×{h}, motion {8 / 2 ** l:g} px")
        axes[l].axis("off")
        img = blur(img, 1.0)[::2, ::2]
    fig.text(0.01, 0.01, CREDIT, fontsize=9, color="#666")
    fig.tight_layout(); fig.savefig("lucas-kanade/fig-pyramid.png"); plt.close(fig)


# ---------------------------------------------------------------- unit 3
def hs(I1, I2, lam, iters, init=None):
    ix, iy, it = derivs(I1, I2, 1.0)
    h, w = I1.shape
    u = np.zeros((h, w)); v = np.zeros((h, w))
    nb = np.zeros((h, w))
    nb[:, 1:] += 1; nb[:, :-1] += 1; nb[1:, :] += 1; nb[:-1, :] += 1
    mask = [((np.add.outer(np.arange(h), np.arange(w)) % 2) == c) for c in (0, 1)]
    for _ in range(iters):
        for m in mask:
            su = np.zeros((h, w)); sv = np.zeros((h, w))
            su[:, 1:] += u[:, :-1]; su[:, :-1] += u[:, 1:]; su[1:, :] += u[:-1, :]; su[:-1, :] += u[1:, :]
            sv[:, 1:] += v[:, :-1]; sv[:, :-1] += v[:, 1:]; sv[1:, :] += v[:-1, :]; sv[:-1, :] += v[1:, :]
            ub, vb = su / nb, sv / nb
            k = (ix * ub + iy * vb + it) / (lam * nb + ix ** 2 + iy ** 2)
            u = np.where(m, ub - ix * k, u); v = np.where(m, vb - iy * k, v)
    return u, v


def ring_scene(n=128, seed=3):
    rng = np.random.default_rng(seed)
    t = blur(rng.standard_normal((n, n)), 1.5); t = 128 + 45 * (t - t.mean()) / t.std()
    y, x = np.mgrid[0:n, 0:n]; c = (n - 1) / 2; d = np.hypot(x - c, y - c)
    a, b = 0.24 * n, 0.38 * n
    s = lambda lo, hi, z: np.clip((z - lo) / (hi - lo), 0, 1) ** 2 * (3 - 2 * np.clip((z - lo) / (hi - lo), 0, 1))
    inr = s(a - 1, a + 1, d) * (1 - s(b - 1, b + 1, d))
    flat = np.where(d < a, 150.0, 90.0)
    return flat * (1 - inr) + t * inr


def fig_fillin():
    I1 = ring_scene()
    dx, dy = 1.5 * np.cos(np.pi / 6), -1.5 * np.sin(np.pi / 6)
    X, Y = np.meshgrid(np.arange(128.0), np.arange(128.0))
    I2 = bilinear(I1, X - dx, Y - dy)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.5]})
    axes[0].imshow(I1, cmap="gray"); axes[0].set_title("frame 1"); axes[0].axis("off")
    done = 0
    u = v = None
    for k, n in enumerate([10, 100, 1000]):
        u, v = hs(I1, I2, 100, n)
        axes[k + 1].imshow(flow_rgb(u, v, 1.5)); axes[k + 1].set_title(f"after {n} sweeps"); axes[k + 1].axis("off")
    wheel_inset(axes[4], 1); axes[4].set_title("1.5 px", fontsize=12)
    fig.tight_layout(); fig.savefig("horn-schunck/fig-fillin.png"); plt.close(fig)


def fig_lambda():
    rng = np.random.default_rng(5)
    def tex(seed, s, sd):
        t = blur(np.random.default_rng(seed).standard_normal((128, 128)), s); return 128 + sd * (t - t.mean()) / t.std()
    bg, fg = tex(5, 2.5, 14) - 40, tex(6, 1.5, 35) + 40
    X, Y = np.meshgrid(np.arange(128.0), np.arange(128.0)); c = 63.5; r = 0.26 * 128
    dx, dy = 1.5 * np.cos(np.pi / 6), -1.5 * np.sin(np.pi / 6)
    def comp(ox, oy):
        d = np.hypot(X - ox - c, Y - oy - c); a = 1 / (1 + np.exp((d - r) / 0.35))
        return a * bilinear(fg, X - ox, Y - oy) + (1 - a) * bg
    I1, I2 = comp(0, 0) + 2 * rng.standard_normal((128, 128)), comp(dx, dy) + 2 * rng.standard_normal((128, 128))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 0.45]})
    axes[0].imshow(I1, cmap="gray"); axes[0].set_title("a cell moves, substrate still"); axes[0].axis("off")
    for k, lam in enumerate([1, 10000]):
        u, v = hs(I1, I2, lam, 1500)
        axes[k + 1].imshow(flow_rgb(u, v, 1.5)); axes[k + 1].set_title(f"λ = {lam}: " + ("noisy" if lam < 10 else "boundary smeared")); axes[k + 1].axis("off")
    wheel_inset(axes[3], 1); axes[3].set_title("1.5 px", fontsize=12)
    fig.tight_layout(); fig.savefig("horn-schunck/fig-lambda.png"); plt.close(fig)


# ---------------------------------------------------------------- unit 4
def b3(t):
    a = np.abs(t)
    return np.where(a < 1, 2 / 3 - a ** 2 + 0.5 * a ** 3, np.where(a < 2, (2 - a) ** 3 / 6, 0))


def fig_bspline():
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9))
    t = np.linspace(-3, 7, 600)
    for k in range(-1, 6):
        axes[0].plot(t, b3(t - k), lw=2)
    axes[0].plot(t, sum(b3(t - k) for k in range(-3, 9)), "k--", lw=1.5, label="sum = 1")
    axes[0].plot([0, 1, -1], [4 / 6, 1 / 6, 1 / 6], "ko")
    axes[0].set_xlim(-1.5, 5.5); axes[0].set_title("one smooth bump per control point"); axes[0].set_xlabel("x / Δ"); axes[0].legend(loc="upper right")
    x = np.linspace(-5, 45, 600)
    c = {0: 0, 10: 0, 20: 6, 30: 0, 40: 0}
    d = sum(cv * b3((x - k) / 10) for k, cv in c.items())
    axes[1].plot(x, d, lw=3, color="#d62728", label="displacement d(x)")
    axes[1].stem(list(c.keys()), list(c.values()), basefmt=" ", label="coefficients")
    for xx in (10, 15, 20):
        axes[1].annotate(f"{sum(cv * b3((xx - k) / 10) for k, cv in c.items()):.3g}", (xx, sum(cv * b3((xx - k) / 10) for k, cv in c.items())), textcoords="offset points", xytext=(6, 6), fontsize=12)
    axes[1].set_xlabel("x (px), control points every 10 px"); axes[1].set_title("one coefficient of 6: a local bump"); axes[1].legend(loc="upper right", fontsize=11)
    fig.tight_layout(); fig.savefig("nonrigid-reg/fig-bspline.png"); plt.close(fig)


def fig_jointhist():
    F = CELL[80:176, 80:176] / 255
    bins = 24
    def jh(A, B):
        H, _, _ = np.histogram2d(A.ravel(), B.ravel(), bins=bins, range=[[0, 1], [0, 1]])
        return np.log1p(H.T)
    shifted = CELL[80:176, 86:182] / 255
    cases = [("aligned", F), ("shifted by 6 px", shifted), ("aligned, inverted", 1 - F), ("aligned, non-monotone", 1 - np.abs(2 * F - 1) ** 1.2)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (name, B) in zip(axes, cases):
        H = jh(F, B)
        ax.imshow(H, origin="lower", extent=(0, 1, 0, 1), cmap="inferno")
        p = np.histogram2d(F.ravel(), B.ravel(), bins=bins, range=[[0, 1], [0, 1]])[0]; p /= p.sum()
        pa, pb = p.sum(1), p.sum(0); nz = p > 0
        mi = (p[nz] * np.log2(p[nz] / np.outer(pa, pb)[nz])).sum()
        ncc = np.corrcoef(F.ravel(), B.ravel())[0, 1]
        ax.set_title(f"{name}\nMI {mi:.2f} bits, NCC {ncc:.2f}", fontsize=13)
        ax.set_xlabel("fixed value"); ax.set_ylabel("moving value")
    fig.text(0.01, 0.01, CREDIT, fontsize=9, color="#666")
    fig.tight_layout(); fig.savefig("nonrigid-reg/fig-jointhist.png"); plt.close(fig)


if __name__ == "__main__":
    fig_flowfield(); fig_bump(); fig_kymograph(); fig_aperture()
    fig_lk_windows(); fig_pyramid()
    fig_fillin(); fig_lambda()
    fig_bspline(); fig_jointhist()
    print("done")
