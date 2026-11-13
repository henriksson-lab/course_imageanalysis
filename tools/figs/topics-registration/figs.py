"""Figures for topics/registration.qmd (the registration and alignment topic page).

Everything is synthetic (fixed seeds); every number quoted on the page is printed here and
written to the JSON file given as the first argument. Run from anywhere:
   python3 -I tools/figs/topics-registration/figs.py [numbers.json]
PNG: figure size in inches = display size / 100 px, dpi 200, fonts >= 12 pt.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import ndimage as ndi
from skimage import transform, feature, measure

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "topics")
plt.rcParams.update({"font.size": 13, "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, ORANGE, SKY, PURPLE, YELLOW, GREY = "#0072b2", "#d55e00", "#009e73", "#e69f00", "#56b4e9", "#cc79a7", "#ffd400", "#777777"
NUM = {}


def save(fig, name, **kw):
    if name.endswith(".png"):
        kw.setdefault("dpi", 200)
    fig.savefig(os.path.join(OUT, name), facecolor="white", bbox_inches="tight", pad_inches=0.04, **kw)
    plt.close(fig); print("wrote", name)


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def show(ax, img, vmin=0, vmax=1, cmap="gray"):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    clean(ax)


def texture(n, sigma, seed, blobs=True):
    rng = np.random.default_rng(seed)
    t = ndi.gaussian_filter(rng.normal(0, 1, (n, n)), sigma)
    t = (t - t.mean()) / t.std()
    if blobs:
        yy, xx = np.mgrid[0:n, 0:n]
        for _ in range(14):
            cy, cx, r = rng.uniform(15, n - 15), rng.uniform(15, n - 15), rng.uniform(4, 9)
            t += 2.2 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
    t = (t - t.min()) / (t.max() - t.min())
    return t


def rigid(theta_deg, t, c):
    """Forward map p -> R(p - c) + c + t (x, y order, as skimage uses)."""
    return (transform.EuclideanTransform(translation=-np.asarray(c)) +
            transform.EuclideanTransform(rotation=np.deg2rad(theta_deg)) +
            transform.EuclideanTransform(translation=np.asarray(c) + np.asarray(t)))


def phasecorr(a, b):
    """Shift s (dy, dx) with b(x) ~ a(x - s), and the peak height."""
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    A = np.fft.fft2((a - a.mean()) * w); B = np.fft.fft2((b - b.mean()) * w)
    R = B * np.conj(A); R /= np.abs(R) + 1e-12
    r = np.real(np.fft.ifft2(R))
    k = np.unravel_index(np.argmax(r), r.shape)
    s = [(k[i] + r.shape[i] // 2) % r.shape[i] - r.shape[i] // 2 for i in range(2)]
    return np.array(s), r.max(), r


N = 200
C = ((N - 1) / 2, (N - 1) / 2)
FIELD = texture(N, 3.0, 5)
THETA, SHIFT = 8.0, (10.0, -6.0)        # degrees; (dx, dy) in pixels


def make_moving():
    T = rigid(THETA, SHIFT, C)
    mov = transform.warp(FIELD, T.inverse, order=3, cval=np.nan)
    rng = np.random.default_rng(9)
    return mov + rng.normal(0, 0.03, mov.shape), T


def overlay(a, b):
    a = np.nan_to_num(a, nan=0); b = np.nan_to_num(b, nan=0)
    return np.dstack([b, a, b]).clip(0, 1)      # fixed green, moving magenta: grey where they agree


# ------------------------------------------------------------------ figure 1: the question + the search
def fig_question():
    mov, T = make_moving()
    filled = np.where(np.isnan(mov), np.nanmean(mov), mov)
    best = None
    angles = np.arange(-15, 15.001, 0.25)
    peaks = []
    for th in angles:
        R = rigid(th, (0, 0), C)
        unrot = transform.warp(filled, R, order=1)    # unrot(x) = moving(R x)
        s, h, _ = phasecorr(FIELD, unrot)
        peaks.append(h)
        if best is None or h > best[2]:
            best = (th, s, h)
    th, s, h = best
    # unrot(x) = field(x - s): the full estimate is moving(R(x + s)) ... compose and apply
    est = rigid(th, (0, 0), C)
    shift_back = transform.EuclideanTransform(translation=(s[1], s[0]))
    Tinv = shift_back + est            # x -> R(x + s)
    aligned = transform.warp(mov, Tinv, order=3, cval=np.nan)
    m0 = ~np.isnan(mov); m1 = ~np.isnan(aligned)
    rms0 = np.sqrt(np.mean((FIELD[m0] - mov[m0]) ** 2))
    rms1 = np.sqrt(np.mean((FIELD[m1] - aligned[m1]) ** 2))
    NUM["question"] = {"true_angle": THETA, "true_shift_xy": SHIFT, "found_angle": float(th),
                       "phasecorr_shift_rc": [int(s[0]), int(s[1])], "peak": round(float(h), 3),
                       "angles_tried": int(len(angles)), "rms_before": round(float(rms0), 3),
                       "rms_after": round(float(rms1), 3), "noise_sd": 0.03,
                       "peak_at_0deg": round(float(peaks[int(np.argmin(np.abs(angles)))]), 3)}
    print("question", NUM["question"])

    fig, axs = plt.subplots(1, 5, figsize=(13.6, 3.25), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]})
    show(axs[0], FIELD); axs[0].set_title("fixed")
    show(axs[1], np.nan_to_num(mov, nan=0)); axs[1].set_title("moving")
    axs[2].imshow(overlay(FIELD, mov)); clean(axs[2]); axs[2].set_title(f"before: RMS {rms0:.2f}")
    axs[3].imshow(overlay(FIELD, aligned)); clean(axs[3]); axs[3].set_title(f"after: RMS {rms1:.2f}")
    ax = axs[4]
    ax.plot(angles, peaks, color=BLUE, lw=2)
    ax.plot([th], [h], "o", color=RED, ms=8)
    ax.annotate(f"{th:g}°", (th, h), (th - 13, h * 0.92), fontsize=13, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel("angle tried (°)"); ax.set_ylabel("peak height")
    ax.set_title("search: try angles,\nphase-correlate each", fontsize=12.5)
    fig.tight_layout(pad=0.3, w_pad=0.5); save(fig, "fig-reg-question.png")


# ------------------------------------------------------------------ figure 2: transform zoo
def fig_zoo():
    g = np.linspace(0, 1, 6)
    lines = [np.stack([np.full(60, v), np.linspace(0, 1, 60)], 1) for v in g] + \
            [np.stack([np.linspace(0, 1, 60), np.full(60, v)], 1) for v in g]

    def aff(Mx, t):
        return lambda p: (p - 0.5) @ np.asarray(Mx).T + 0.5 + np.asarray(t)

    def homog(H):
        def f(p):
            q = np.c_[p, np.ones(len(p))] @ np.asarray(H).T
            return q[:, :2] / q[:, 2:]
        return f

    th = np.deg2rad(20)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    def ffd(p):
        x, y = p[:, 0], p[:, 1]
        return np.c_[x + 0.08 * np.sin(2 * np.pi * y) * np.sin(np.pi * x), y + 0.08 * np.sin(np.pi * x) * np.sin(np.pi * y)]

    rng = np.random.default_rng(2)
    U = ndi.gaussian_filter(rng.normal(0, 1, (2, 40, 40)), (0, 2.5, 2.5))
    U = 0.12 * U / np.abs(U).max()

    def flow(p):
        i = np.clip((p * 39).round().astype(int), 0, 39)
        return p + np.c_[U[0][i[:, 1], i[:, 0]], U[1][i[:, 1], i[:, 0]]]

    models = [("translation", "2 numbers", aff(np.eye(2), (0.2, 0.1))),
              ("rigid", "3", aff(Rm, (0.1, 0.05))),
              ("similarity", "4", aff(0.75 * Rm, (0.1, 0.05))),
              ("affine", "6", aff([[1, 0.45], [0.1, 0.8]], (0, 0))),
              ("homography", "8", homog([[1, 0.1, 0], [0.05, 1, 0], [0.55, 0.35, 1]])),
              ("free-form (spline)", "2 per control point", ffd),
              ("dense flow", "2 per pixel", flow)]
    fig, axs = plt.subplots(1, 7, figsize=(13.8, 2.75))
    for ax, (name, k, f) in zip(axs, models):
        for L in lines:
            ax.plot(L[:, 0], L[:, 1], color="#bbbbbb", lw=1)
            q = f(L)
            ax.plot(q[:, 0], q[:, 1], color=BLUE, lw=1.6)
        ax.set_aspect("equal"); ax.set_xlim(-0.25, 1.4); ax.set_ylim(-0.3, 1.35)
        ax.axis("off")
        ax.set_title(f"{name}\n{k}", fontsize=12.5)
    fig.tight_layout(pad=0.2, w_pad=0.2); save(fig, "fig-reg-zoo.svg")


# ------------------------------------------------------------------ figure 3: score landscapes
def ncc_map(a, b, R):
    """NCC of a with b shifted by (dy, dx) for |d| <= R, over the overlap."""
    out = np.zeros((2 * R + 1, 2 * R + 1))
    n = a.shape[0]
    for i, dy in enumerate(range(-R, R + 1)):
        for j, dx in enumerate(range(-R, R + 1)):
            ya, yb = (slice(dy, n), slice(0, n - dy)) if dy >= 0 else (slice(0, n + dy), slice(-dy, n))
            xa, xb = (slice(dx, n), slice(0, n - dx)) if dx >= 0 else (slice(0, n + dx), slice(-dx, n))
            u = a[ya, xa].ravel(); v = b[yb, xb].ravel()
            u = u - u.mean(); v = v - v.mean()
            out[i, j] = u @ v / np.sqrt((u @ u) * (v @ v))
    return out


def capture(profile, centre):
    """Half-width of the basin around the peak: steps until the score first rises again."""
    k = 0
    while centre + k + 1 < len(profile) and profile[centre + k + 1] < profile[centre + k]:
        k += 1
    l = 0
    while centre - l - 1 >= 0 and profile[centre - l - 1] < profile[centre - l]:
        l += 1
    return min(k, l)


def fig_landscape():
    n, R = 128, 30
    sh = (7, -12)    # (dy, dx): b(x) = a(x - s)
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.7), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    a = texture(n + 40, 3.0, 21)
    A = a[20:20 + n, 20:20 + n]
    B = a[20 - sh[0]:20 - sh[0] + n, 20 - sh[1]:20 - sh[1] + n]
    B = B + np.random.default_rng(4).normal(0, 0.03, B.shape)
    M = ncc_map(B, A, R)        # NCC(d): B shifted against A
    k = np.unravel_index(np.argmax(M), M.shape)
    s, h, r = phasecorr(A, B)
    rs = np.fft.fftshift(r)
    c = n // 2
    NUM["landscape"] = {"true_shift_rc": list(sh), "ncc_peak_rc": [int(k[0] - R), int(k[1] - R)],
                        "ncc_max": round(float(M.max()), 3), "phasecorr_rc": [int(s[0]), int(s[1])],
                        "phasecorr_peak": round(float(h), 3)}
    ext = [-R - 0.5, R + 0.5, R + 0.5, -R - 0.5]
    axs[0].imshow(M, cmap="viridis", extent=ext); axs[0].set_title("NCC over all shifts:\na broad hill", fontsize=12.5)
    axs[0].set_xlabel("dx"); axs[0].set_ylabel("dy")
    axs[1].imshow(rs[c - R:c + R + 1, c - R:c + R + 1], cmap="viridis", extent=ext)
    axs[1].set_title("phase correlation:\none sharp peak", fontsize=12.5); axs[1].set_xlabel("dx")
    for ax in axs[:2]:
        ax.plot(sh[1], sh[0], "+", color=RED, ms=16, mew=2)
    # 1D: capture range for coarse and fine texture
    L = 60
    prof = {}
    for sig, col in ((6.0, BLUE), (1.2, ORANGE)):
        t = texture(n + 2 * L + 4, sig, 33, blobs=False)
        A1 = t[L:L + n, L:L + n]
        vals = []
        for dx in range(-L, L + 1):
            B1 = t[L:L + n, L + dx:L + dx + n]
            u = A1.ravel() - A1.mean(); v = B1.ravel() - B1.mean()
            vals.append(u @ v / np.sqrt((u @ u) * (v @ v)))
        vals = np.array(vals)
        cr = capture(vals, L)
        prof[sig] = cr
        axs[2].plot(np.arange(-L, L + 1), vals, color=col, lw=2,
                    label=f"texture scale {sig:g} px: basin ±{cr} px")
        axs[2].axvspan(-cr, cr, color=col, alpha=0.12)
    NUM["landscape"]["capture_px"] = {str(k): int(v) for k, v in prof.items()}
    axs[2].set_xlabel("shift tried (px), true shift 0"); axs[2].set_ylabel("NCC")
    axs[2].set_title("downhill search works only\ninside the basin", fontsize=12.5)
    axs[2].legend(fontsize=11, frameon=False, loc="upper right")
    axs[2].set_ylim(-0.4, 1.25)
    print("landscape", NUM["landscape"])
    fig.tight_layout(pad=0.3); save(fig, "fig-reg-landscape.png")


# ------------------------------------------------------------------ figure 4: joint histograms
def mi(a, b, bins=32):
    h, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins, range=[[0, 1], [0, 1]])
    p = h / h.sum(); px = p.sum(1); py = p.sum(0)
    nz = p > 0
    return float(np.sum(p[nz] * np.log2(p[nz] / np.outer(px, py)[nz]))), h


def ncc(a, b):
    u = a.ravel() - a.mean(); v = b.ravel() - b.mean()
    return float(u @ v / np.sqrt((u @ u) * (v @ v)))


def fig_jointhist():
    n = 160
    t = texture(n + 20, 3.0, 8)
    A = t[10:10 + n, 10:10 + n]
    rng = np.random.default_rng(12)
    cases = [("aligned", t[10:10 + n, 10:10 + n]),
             ("4 px off", t[10:10 + n, 14:14 + n]),
             ("aligned, other contrast", 1 - t[10:10 + n, 10:10 + n] ** 2)]
    fig, axs = plt.subplots(2, 3, figsize=(9.6, 6.6), gridspec_kw={"height_ratios": [1, 1]})
    NUM["jointhist"] = {}
    for j, (name, B) in enumerate(cases):
        B = np.clip(B + rng.normal(0, 0.02, B.shape), 0, 1)
        m, h = mi(A, B)
        c = ncc(A, B)
        ssd = float(np.mean((A - B) ** 2))
        NUM["jointhist"][name] = {"mi_bits": round(m, 2), "ncc": round(c, 2), "msd": round(ssd, 4)}
        show(axs[0, j], B); axs[0, j].set_title(name, fontsize=12.5)
        axs[1, j].imshow(np.log1p(h.T), origin="lower", cmap="magma", extent=[0, 1, 0, 1])
        axs[1, j].set_xlabel("fixed grey level")
        if j == 0:
            axs[1, j].set_ylabel("moving grey level")
        axs[1, j].set_title(f"MI {m:.2f} bits · NCC {c:+.2f}\nmean sq. diff. {ssd:.4f}", fontsize=12)
        axs[1, j].set_xticks([0, 1]); axs[1, j].set_yticks([0, 1])
    print("jointhist", NUM["jointhist"])
    fig.tight_layout(pad=0.3, h_pad=0.8); save(fig, "fig-reg-jointhist.png")


# ------------------------------------------------------------------ figure 5: features + RANSAC
def fig_features():
    mov, T = make_moving()
    m2 = np.where(np.isnan(mov), 0, mov)
    orb = feature.ORB(n_keypoints=150, fast_threshold=0.05)
    orb.detect_and_extract(FIELD); k1, d1 = orb.keypoints, orb.descriptors
    orb.detect_and_extract(m2); k2, d2 = orb.keypoints, orb.descriptors
    mt = feature.match_descriptors(d1, d2, cross_check=True)
    src = k1[mt[:, 0]][:, ::-1]; dst = k2[mt[:, 1]][:, ::-1]      # (x, y)
    model, inl = measure.ransac((src, dst), transform.EuclideanTransform, min_samples=2,
                                residual_threshold=2, max_trials=2000, rng=0)
    # truly correct matches: within 2 px of the true transform
    true_ok = np.linalg.norm(T(src) - dst, axis=1) < 2
    ls = transform.EuclideanTransform()
    ls.estimate(src, dst)
    err_ls = float(np.sqrt(np.mean(np.sum((ls(src[true_ok]) - T(src[true_ok])) ** 2, 1))))
    err_r = float(np.sqrt(np.mean(np.sum((model(src[true_ok]) - T(src[true_ok])) ** 2, 1))))
    NUM["features"] = {"keypoints": [int(len(k1)), int(len(k2))], "matches": int(len(mt)),
                       "truly_correct": int(true_ok.sum()), "ransac_inliers": int(inl.sum()),
                       "ransac_angle": round(float(np.rad2deg(model.rotation)), 2),
                       "lsq_all_angle": round(float(np.rad2deg(ls.rotation)), 2),
                       "err_ransac_px": round(err_r, 2), "err_lsq_all_px": round(err_ls, 2)}
    print("features", NUM["features"])
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    gap = 12
    canvas = np.ones((N, 2 * N + gap))
    canvas[:, :N] = FIELD; canvas[:, N + gap:] = m2
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    for (p, q, ok) in zip(src, dst, inl):
        ax.plot([p[0], q[0] + N + gap], [p[1], q[1]], color=GREEN if ok else RED,
                lw=1.1 if ok else 0.9, alpha=0.9 if ok else 0.7)
    ax.plot(src[:, 0], src[:, 1], ".", color=YELLOW, ms=4)
    ax.plot(dst[:, 0] + N + gap, dst[:, 1], ".", color=YELLOW, ms=4)
    clean(ax)
    f = NUM["features"]
    ax.set_title(f"{f['matches']} matches: {f['ransac_inliers']} kept by RANSAC (green), "
                 f"{f['matches'] - f['ransac_inliers']} rejected (red)", fontsize=13)
    fig.tight_layout(pad=0.2); save(fig, "fig-reg-features.png")


if __name__ == "__main__":
    fig_question()
    fig_zoo()
    fig_landscape()
    fig_jointhist()
    fig_features()
    if len(sys.argv) > 1:
        json.dump(NUM, open(sys.argv[1], "w"), indent=1)
