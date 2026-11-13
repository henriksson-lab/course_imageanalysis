# Static figures for the colour day (colour-vision, rgb-linear, colour-perceptual,
# colour-display). Run from the repository root, after data.ts (see its header):
#   python3 -I tools/figs/colour/figs.py <tmp>/colour-data.json
# The JSON holds data computed with the demos' own code (spectra, scenes, segmentation,
# white balance), so the numbers in the figures match the demos. Everything is synthetic
# except demos/src/fourier2d/cell256.png (Robert M. Hunt, CC BY 3.0, Wikimedia Commons).
import sys, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon, Rectangle, Circle, Wedge, FancyArrowPatch
from PIL import Image

D = json.load(open(sys.argv[1]))
ONLY = set(sys.argv[2:])  # optional: names of figures to (re)make
INK = "#1b2a3a"
OK = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "black": "#000000"}
plt.rcParams.update({
    "font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def want(name):
    return not ONLY or name in ONLY


def save(fig, path, dpi=None):
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("wrote", path)


# ---- colour helpers (same formulas as demos/src/lib/colour.ts) ---------------------------
M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
M_XYZ2RGB = np.linalg.inv(M_RGB2XYZ)
WHITE = np.array([0.95047, 1.0, 1.08883])


def dec(v):
    v = np.asarray(v, float)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def enc(v):
    v = np.clip(np.asarray(v, float), 0, 1)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)


def f_lab(t):
    d = 6 / 29
    return np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4 / 29)


def xyz2lab(xyz):
    f = f_lab(np.asarray(xyz) / WHITE)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab2xyz(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    d = 6 / 29
    finv = lambda f: np.where(f > d, f ** 3, 3 * d * d * (f - 4 / 29))
    return np.stack([finv(fx), finv(fy), finv(fz)], -1) * WHITE


def srgb8_to_lab(c):
    return xyz2lab(dec(np.asarray(c, float) / 255) @ M_RGB2XYZ.T)


def scene_img(s, key=("r", "g", "b"), linear=True):
    w, h = s["w"], s["h"]
    im = np.stack([np.array(s[k], float).reshape(h, w) for k in key], -1)
    return enc(im) if linear else im


def chroma_bg(ax, alpha=0.55, n=300):
    """Fill the spectral locus with approximate colours (clipped into sRGB)."""
    loc = D["locus"]
    lx, ly = np.array([p["x"] for p in loc]), np.array([p["y"] for p in loc])
    path = Path(np.c_[np.r_[lx, lx[0]], np.r_[ly, ly[0]]])
    xs, ys = np.meshgrid(np.linspace(0, 0.8, n), np.linspace(0, 0.9, n))
    inside = path.contains_points(np.c_[xs.ravel(), ys.ravel()]).reshape(xs.shape)
    X = xs / np.maximum(ys, 1e-6); Z = (1 - xs - ys) / np.maximum(ys, 1e-6)
    rgb = np.stack([X, np.ones_like(X), Z], -1) @ M_XYZ2RGB.T
    rgb = rgb - np.minimum(rgb.min(-1, keepdims=True), 0)
    rgb = rgb / rgb.max(-1, keepdims=True)
    img = np.ones(xs.shape + (4,))
    img[..., :3] = enc(rgb)
    img[..., 3] = np.where(inside, alpha, 0)
    ax.imshow(img, origin="lower", extent=(0, 0.8, 0, 0.9), interpolation="bilinear", zorder=0)
    ax.plot(np.r_[lx, lx[0]], np.r_[ly, ly[0]], color=INK, lw=1.6, zorder=2)
    return lx, ly, loc


def locus_labels(ax, loc, wls=(460, 480, 500, 520, 540, 560, 580, 600, 620), fs=14):
    cx, cy = 0.3127, 0.329
    for w in wls:
        p = min(loc, key=lambda q: abs(q["l"] - w))
        dx, dy = p["x"] - cx, p["y"] - cy; n = math.hypot(dx, dy)
        ax.plot(p["x"], p["y"], "o", color=INK, ms=3.5, zorder=3)
        ax.text(p["x"] + 0.045 * dx / n, p["y"] + 0.045 * dy / n, f"{w}", ha="center", va="center", fontsize=fs, zorder=3)


def wl_colour(l):
    xyz = np.array([D["cmf"][int(round(li)) - 380] for li in np.atleast_1d(l)])
    rgb = xyz @ M_XYZ2RGB.T
    rgb = rgb - np.minimum(rgb.min(-1, keepdims=True), 0)
    rgb = rgb / rgb.max(-1, keepdims=True).clip(1e-9)
    return enc(rgb)


def wl_bar(ax, y0, h):
    l = np.arange(380, 731)
    ax.imshow(wl_colour(l)[None], extent=(380, 730, y0, y0 + h), aspect="auto", zorder=0)


def panel(ax, img, title=None, below=None, fs=15):
    ax.imshow(np.clip(img, 0, 1), interpolation="nearest", cmap="gray" if np.ndim(img) == 2 else None, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_color("#888")
    if title: ax.set_title(title, fontsize=fs, pad=6)
    if below: ax.set_xlabel(below, fontsize=fs - 1)


WL = np.array(D["wl"]); WLF = np.array(D["wlFine"])

# =========================================================================================
# colour-vision
# =========================================================================================
if want("spectra"):
    fig, ax = plt.subplots(figsize=(6.07, 4.6))
    ax.plot(WL, D["d65"], color=OK["blue"], lw=2.6, label="daylight (D65)")
    ax.plot(WL, D["tungsten2800"], color=OK["verm"], lw=2.6, ls="--", label="light bulb (2800 K)")
    ax.plot(WL, D["whiteLED"], color=OK["green"], lw=2.6, ls="-.", label="white LED")
    ax.plot([532, 532], [0, 245], color=INK, lw=3, label="laser (532 nm)")
    wl_bar(ax, -14, 10)
    ax.set_xlim(380, 730); ax.set_ylim(-14, 250)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_xlabel("wavelength λ (nm)"); ax.set_ylabel("power (100 at 560 nm)")
    ax.legend(frameon=False, fontsize=15, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2)
    fig.tight_layout(); save(fig, "colour-vision/fig-spectra.svg")

if want("metamers"):
    fig, ax = plt.subplots(figsize=(6.07, 3.77))
    led = np.array(D["metamericLED"])
    ax.plot(WL, D["d65"], color=OK["blue"], lw=2.6, label="daylight (D65)")
    ax.plot(WL, led, color=OK["orange"], lw=2.6, ls="--", label="white LED (tuned)")
    wl_bar(ax, -16, 11)
    ax.set_xlim(380, 730); ax.set_ylim(-16, 265)
    ax.set_xlabel("wavelength λ (nm)"); ax.set_ylabel("power")
    ax.legend(frameon=False, fontsize=14, loc="upper right")
    r = D["resp"]
    ax.set_title("eye: L, M, S = %.3f, %.3f, %.3f for both" % tuple(r["led_lms"]), fontsize=16, loc="left")
    fig.tight_layout(); save(fig, "colour-vision/fig-metamers.svg")

if want("camera"):
    r = D["resp"]
    fig, axs = plt.subplots(1, 2, figsize=(6.07, 3.28), sharey=True)
    for ax, (a, b, labs, ttl) in zip(axs, [(r["d65_lms"], r["led_lms"], "LMS", "eye (cones)"), (r["d65_cam"], r["led_cam"], "RGB", "camera")]):
        x = np.arange(3)
        ax.bar(x - 0.2, a, 0.38, color=OK["blue"], label="daylight")
        ax.bar(x + 0.2, b, 0.38, color=OK["orange"], hatch="//", edgecolor="white", label="white LED")
        for i in range(3):
            ax.text(i + 0.2, b[i] + 0.02, f"{b[i]:.3f}", ha="center", va="bottom", fontsize=14, rotation=90)
        ax.set_xticks(x, list(labs)); ax.set_title(ttl, fontsize=16)
        ax.set_ylim(0, 1.5); ax.set_yticks([0, 0.5, 1])
    axs[0].set_ylabel("response (daylight = 1)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=15, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.09, 1, 1)); save(fig, "colour-vision/fig-camera.svg")

if want("matching"):
    xyz = np.array(D["cmf"])
    P = np.array([D["cmf"][700 - 380], 0.9 * np.array(D["cmf"][546 - 380]) + 0.1 * np.array(D["cmf"][547 - 380]), 0.2 * np.array(D["cmf"][435 - 380]) + 0.8 * np.array(D["cmf"][436 - 380])]).T
    rgb = xyz @ np.linalg.inv(P).T
    rgb = rgb / rgb.sum(0)  # equal-energy white needs equal amounts
    rgb = rgb / rgb.max()
    fig = plt.figure(figsize=(4.92, 5.25))
    ax0 = fig.add_axes([0.0, 0.56, 1.0, 0.44]); ax0.set_axis_off(); ax0.set_xlim(0, 2.2); ax0.set_ylim(0, 1.0); ax0.set_aspect("equal")
    ax0.add_patch(Wedge((0.62, 0.5), 0.42, 90, 270, fc=wl_colour(500)[0], ec=INK, lw=1.5))
    ax0.add_patch(Wedge((0.62, 0.5), 0.42, -90, 90, fc=enc(np.clip(dec(wl_colour(500)[0]) * 0.85 + 0.03, 0, 1)), ec=INK, lw=1.5))
    ax0.text(0.42, 0.5, "test\n500 nm", ha="center", va="center", fontsize=15)
    ax0.text(0.83, 0.5, "R+G+B", ha="center", va="center", fontsize=15)
    for i, (c, n) in enumerate([("#d62728", "R 700 nm"), ("#2ca02c", "G 546 nm"), ("#1f4fd0", "B 436 nm")]):
        y = 0.82 - 0.3 * i
        ax0.add_patch(Circle((1.3, y), 0.08, fc=c, ec=INK))
        ax0.text(1.43, y, n, ha="left", va="center", fontsize=15)
    ax0.annotate("", xy=(1.06, 0.5), xytext=(1.2, 0.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax = fig.add_axes([0.17, 0.14, 0.8, 0.42])
    cols = [OK["verm"], OK["green"], OK["blue"]]; ls = ["-", "--", "-."]
    for k, n in enumerate(["r̄", "ḡ", "b̄"]):
        ax.plot(WLF, rgb[:, k], color=cols[k], lw=2.4, ls=ls[k], label=n)
    neg = rgb[:, 0] < 0
    ax.fill_between(WLF, rgb[:, 0], 0, where=neg, color=OK["verm"], alpha=0.3, lw=0)
    ax.axhline(0, color=INK, lw=0.8)
    imin = np.argmin(rgb[:, 0])
    ax.annotate("negative: red\nadded to the test", xy=(WLF[imin], rgb[imin, 0]), xytext=(585, -0.3), fontsize=14,
                arrowprops=dict(arrowstyle="->", color=INK), va="center")
    ax.set_xlim(380, 730); ax.set_ylim(-0.55, 1.08); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("test wavelength (nm)"); ax.set_ylabel("amount")
    ax.legend(frameon=False, fontsize=15, loc="upper right")
    save(fig, "colour-vision/fig-matching.svg")

if want("xyz"):
    xyz = np.array(D["cmf"])
    fig, ax = plt.subplots(figsize=(5.90, 3.61))
    ax.plot(WLF, xyz[:, 0], color=OK["verm"], lw=2.4, label="x̄")
    ax.plot(WLF, xyz[:, 1], color=OK["green"], lw=4, ls="--", label="ȳ = brightness (Y)")
    ax.plot(WLF, xyz[:, 2], color=OK["blue"], lw=2.4, ls="-.", label="z̄")
    wl_bar(ax, -0.13, 0.08)
    ax.set_xlim(380, 730); ax.set_ylim(-0.13, 1.95)
    ax.set_xlabel("wavelength λ (nm)"); ax.set_ylabel("sensitivity")
    ax.legend(frameon=False, fontsize=15)
    fig.tight_layout(); save(fig, "colour-vision/fig-xyz.svg")

if want("gamuts"):
    fig, ax = plt.subplots(figsize=(4.92, 4.59))
    lx, ly, loc = chroma_bg(ax, alpha=0.45)
    locus_labels(ax, loc)
    sty = [(OK["blue"], "-", "sRGB"), (OK["verm"], "--", "Display P3"), (INK, ":", "Rec. 2020")]
    for d, (c, l, n) in zip(D["displays"], sty):
        t = np.array([d["r"], d["g"], d["b"], d["r"]])
        ax.plot(t[:, 0], t[:, 1], color=c, ls=l, lw=2.6, label=n, zorder=4)
    ax.plot(0.3127, 0.329, "o", ms=7, mfc="white", mec=INK, zorder=5); ax.text(0.33, 0.30, "D65", fontsize=14)
    ax.set_xlim(-0.06, 0.8); ax.set_ylim(0, 0.9); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(frameon=False, fontsize=14, loc="upper right")
    fig.tight_layout(); save(fig, "colour-vision/fig-gamuts.svg")

# =========================================================================================
# rgb-linear
# =========================================================================================
if want("device"):
    fig, ax = plt.subplots(figsize=(4.92, 4.43))
    lx, ly, loc = chroma_bg(ax, alpha=0.45)
    locus_labels(ax, loc, wls=(480, 520))
    for d, (c, l) in zip(D["displays"][:2], [(OK["blue"], "-"), (OK["verm"], "--")]):
        t = np.array([d["r"], d["g"], d["b"], d["r"]])
        ax.plot(t[:, 0], t[:, 1], color=c, ls=l, lw=2.6, label=d["name"], zorder=4)
        ax.plot(*d["r"], "o", color=c, ms=8, zorder=5)
    ax.annotate("(255, 0, 0)\nin sRGB", xy=D["displays"][0]["r"], xytext=(0.36, 0.06), fontsize=15, color=OK["blue"],
                arrowprops=dict(arrowstyle="->", color=OK["blue"]))
    ax.annotate("(255, 0, 0)\nin P3", xy=D["displays"][1]["r"], xytext=(0.5, 0.5), fontsize=15, color=OK["verm"],
                arrowprops=dict(arrowstyle="->", color=OK["verm"]))
    ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.9); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(frameon=False, fontsize=14, loc="upper right")
    fig.tight_layout(); save(fig, "rgb-linear/fig-device.svg")

if want("seam"):
    s = D["seam"]; n = s["n"]
    get = lambda k: np.stack([np.array(c, float).reshape(n, n) for c in s[k]], -1)[:64, :64]
    I, Nv, Lv = get("input"), get("naive"), get("linear")
    fig, axs = plt.subplots(2, 2, figsize=(5.4, 5.6))
    row = 16
    for ax, im, t in zip(axs.ravel(), (I, Nv, Lv), ("input", "blur on codes", "blur in linear light")):
        panel(ax, im, t)
        c = np.round(255 * im[row, 32]).astype(int)
        ax.set_xlabel(f"border: {c[0]}, {c[1]}, {c[2]}", fontsize=15)
        ax.axhline(row, color="white", lw=1, ls="--")
    ax = axs[1, 1]
    x = np.arange(16, 48)
    lum = lambda im: 255 * enc(dec(im[row, 16:48]) @ M_RGB2XYZ[1])
    ax.plot(x, lum(Nv), color=OK["verm"], lw=2.4, label="codes")
    ax.plot(x, lum(Lv), color=OK["blue"], lw=2.4, ls="--", label="linear")
    ax.set_ylim(80, 240); ax.set_xticks([16, 32, 48]); ax.set_xlabel("x along the dashed row", fontsize=14)
    ax.set_title("luminance (as code)", fontsize=15, pad=6)
    ax.text(33, 92, "codes", color=OK["verm"], fontsize=14); ax.text(17, 160, "linear", color=OK["blue"], fontsize=14)
    fig.tight_layout(); save(fig, "rgb-linear/fig-seam.png", dpi=150)

if want("lumweights"):
    w = np.array(D["srgbMatrix"])[1]
    fig, ax = plt.subplots(figsize=(4.43, 2.95))
    cols = ["#ff0000", "#00c000", "#0000ff"]
    ax.bar([0, 1, 2], 100 * w, color=cols, width=0.6, edgecolor=INK)
    for i, v in enumerate(w):
        ax.text(i, 100 * v + 2, f"{100 * v:.1f} %", ha="center", va="bottom", fontsize=16)
    ax.set_xticks([0, 1, 2], ["red\nprimary", "green\nprimary", "blue\nprimary"])
    ax.set_ylim(0, 85); ax.set_ylabel("share of white's\nluminance (%)")
    fig.tight_layout(); save(fig, "rgb-linear/fig-lumweights.svg")

if want("grey"):
    s = D["isolum"]; h, w = s["h"], s["w"]
    lin = np.stack([np.array(s[k]).reshape(h, w) for k in "rgb"], -1)
    E = enc(lin)
    Y = enc(lin @ M_RGB2XYZ[1])
    mean = E.mean(-1)
    G = E[..., 1]
    t = np.array(s["truth"]).reshape(h, w).astype(bool)
    fig, axs = plt.subplots(2, 2, figsize=(5.08, 5.41))
    items = [(E, "colour", None), (Y, "luminance Y (linear)", Y), (mean, "mean (R′+G′+B′)/3", mean), (G, "G′ channel", G)]
    for ax, (im, ttl, v) in zip(axs.ravel(), items):
        panel(ax, im, ttl)
        if v is not None:
            ax.set_xlabel(f"letter {255 * v[t].mean():.1f}, back {255 * v[~t].mean():.1f}", fontsize=14)
    fig.tight_layout(); save(fig, "rgb-linear/fig-grey.png", dpi=150)

if want("ycbcr"):
    s = D["objects160"]; img = scene_img(s)
    Rp, Gp, Bp = [255 * img[..., k] for k in range(3)]
    Yp = 0.299 * Rp + 0.587 * Gp + 0.114 * Bp
    Cb = -0.1687 * Rp - 0.3313 * Gp + 0.5 * Bp + 128
    Cr = 0.5 * Rp - 0.4187 * Gp - 0.0813 * Bp + 128
    fig, axs = plt.subplots(2, 2, figsize=(4.92, 5.25))
    for ax, im, ttl in zip(axs.ravel(), (img, Yp / 255, Cb / 255, Cr / 255), ("R′G′B′", "Y′ (luma)", "Cb (128 = grey)", "Cr (128 = grey)")):
        panel(ax, im, ttl)
    fig.tight_layout(); save(fig, "rgb-linear/fig-ycbcr.png", dpi=150)


def chart(ax, cols, ncol=6, title=None):
    cols = np.array(cols)
    for i, c in enumerate(cols):
        x, y = i % ncol, i // ncol
        ax.add_patch(Rectangle((x + 0.06, -y - 0.94), 0.88, 0.88, fc=enc(c), ec="none"))
    ax.set_xlim(0, ncol); ax.set_ylim(-len(cols) / ncol, 0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.add_patch(Rectangle((0, -len(cols) / ncol), ncol, len(cols) / ncol, fc="none", ec="#999", lw=1))
    if title: ax.set_ylabel(title, fontsize=15, rotation=0, ha="right", va="center", labelpad=8)


if want("wbchart"):
    wb = D["wb"]
    fig, axs = plt.subplots(3, 1, figsize=(4.59, 4.43))
    chart(axs[0], wb["captured"], title=f"under a\nlight bulb\nΔE {wb['dENone']:.1f}")
    chart(axs[1], wb["corrected"], title=f"diagonal fix\n(grey card)\nΔE {wb['dECard']:.2f}")
    chart(axs[2], wb["reference"], title="truth:\nin daylight")
    fig.tight_layout(h_pad=0.6); save(fig, "rgb-linear/fig-wbchart.png", dpi=150)

if want("greyworld"):
    f = D["wbForest"]
    fig, axs = plt.subplots(2, 2, figsize=(4.8, 3.9))
    for ax, k, t in zip(axs.ravel(), ("captured", "greyworld", "whitepatch", "reference"),
                        ("lamp 4000 K, no fix", f"grey world: ΔE {f['dEgw']:.1f}", f"white patch: ΔE {f['dEwp']:.2f}", "truth (daylight)")):
        chart(ax, f[k]); ax.set_title(t, fontsize=15)
    fig.tight_layout(); save(fig, "rgb-linear/fig-greyworld.png", dpi=150)

# =========================================================================================
# colour-perceptual
# =========================================================================================
S = D["seg"]; SW, SH = S["w"], S["h"]
SIMG = np.stack([np.array(S[k]).reshape(SH, SW) for k in "rgb"], -1)  # encoded


def masked(mask):
    m = np.array(mask).reshape(SH, SW).astype(bool)
    out = SIMG * 0.25 + 25 / 255
    out[m] = SIMG[m]
    return out


if want("rthreshold"):
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 3.1))
    panel(axs[0], SIMG, "image (synthetic)")
    panel(axs[1], masked(S["maskR"]), "R′ ≥ 120, G′, B′ ≤ 90", below=f"Dice {S['diceR']:.3f}")
    fig.tight_layout(); save(fig, "colour-perceptual/fig-rthreshold.png", dpi=150)

if want("cloud"):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    lin = np.stack([np.array(c).reshape(SH, SW) for c in S["lin"]], -1)
    pts = lin.reshape(-1, 3); cols = SIMG.reshape(-1, 3)
    rng = np.random.default_rng(1); idx = rng.choice(len(pts), 5000, replace=False)
    fig = plt.figure(figsize=(5.25, 4.59))
    ax = fig.add_axes([0.0, 0.0, 0.82, 1.0], projection="3d")
    ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2], c=np.clip(cols[idx], 0, 1), s=5, depthshade=False)
    E = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    for a in E:
        for b in E:
            if sum(abs(np.subtract(a, b))) == 1 and a < b:
                ax.plot(*zip(a, b), color="#999", lw=0.8)
    ax.plot([0, 1], [0, 1], [0, 1], color=INK, lw=2.2)
    ax.text(1.02, 1.02, 1.08, "white", fontsize=14); ax.text(0, 0, -0.12, "black", fontsize=14)
    apple = np.array(S["truth"]).reshape(-1) == 1
    lit = pts[apple][np.argmax(pts[apple][:, 0])]
    ax.plot([0, 1.05 * lit[0]], [0, 1.05 * lit[1]], [0, 1.05 * lit[2]], color=OK["verm"], lw=2, ls="--")
    ax.text(lit[0] * 0.75, lit[1] * 0.75, lit[2] * 0.75 - 0.25, "apple: a ray\nfrom black", color=OK["verm"], fontsize=15)
    ax.set_xlabel("R", labelpad=-8); ax.set_ylabel("G", labelpad=-8); ax.set_zlabel("B", labelpad=-8)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_zticks([0, 1])
    ax.tick_params(labelsize=12, pad=-2)
    ax.view_init(elev=18, azim=-58)
    ax.set_box_aspect((1, 1, 1))
    ins = fig.add_axes([0.70, 0.62, 0.3, 0.34]); panel(ins, SIMG)
    save(fig, "colour-perceptual/fig-cloud.png", dpi=150)


def hsv2rgb(h, s, v):
    h = np.asarray(h) % 360 / 60; c = v * s; x = c * (1 - np.abs(h % 2 - 1)); m = v - c
    k = np.floor(h).astype(int) % 6
    tab = np.stack([np.stack([c, x, 0 * c], -1), np.stack([x, c, 0 * c], -1), np.stack([0 * c, c, x], -1),
                    np.stack([0 * c, x, c], -1), np.stack([x, 0 * c, c], -1), np.stack([c, 0 * c, x], -1)], 0)
    out = np.take_along_axis(tab, k[None, ..., None].repeat(3, -1), 0)[0]
    return out + np.asarray(m)[..., None]


if want("hexagon"):
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    n = 400; xs, ys = np.meshgrid(np.linspace(-1.1, 1.1, n), np.linspace(-1.1, 1.1, n))
    ang = np.degrees(np.arctan2(ys, xs)) % 360
    # radius of the hexagon at angle: vertices at 0, 60, ...
    a = np.radians((ang % 60) - 30)
    rmax = math.cos(math.radians(30)) / np.cos(a)
    r = np.hypot(xs, ys) / rmax
    img = np.ones((n, n, 4)); img[..., :3] = hsv2rgb(ang, np.clip(r, 0, 1), np.ones_like(r)); img[..., 3] = (r <= 1)
    ax.imshow(img, origin="lower", extent=(-1.1, 1.1, -1.1, 1.1), interpolation="bilinear")
    V = [(math.cos(math.radians(k * 60)), math.sin(math.radians(k * 60))) for k in range(6)]
    ax.add_patch(Polygon(V, closed=True, fc="none", ec=INK, lw=1.5))
    names = ["red 0°", "yellow 60°", "green 120°", "cyan 180°", "blue 240°", "magenta 300°"]
    for (vx, vy), nm in zip(V, names):
        side = "left" if vx > 0 else "right"
        dx = 0.08 if vx > 0 else -0.08
        ax.text(vx + dx, vy * 1.12, nm, ha=side, va="center", fontsize=16)
    ax.plot(0, 0, "o", color=INK, ms=5); ax.text(-0.3, -0.1, "grey\n(S = 0)", fontsize=15, ha="center", va="top")
    # (200, 100, 50): H = 20°, S = 0.75
    h, s_ = 20, 0.75
    rr = s_ * math.cos(math.radians(30)) / math.cos(math.radians((h % 60) - 30))
    px, py = rr * math.cos(math.radians(h)), rr * math.sin(math.radians(h))
    ax.plot([0, 0.85], [0, 0], color=INK, lw=1, ls=":")
    ax.plot([0, px], [0, py], color=INK, lw=1.5)
    ax.plot(px, py, "o", ms=12, mfc=np.array([200, 100, 50]) / 255, mec=INK, mew=2)
    ax.annotate("(200, 100, 50)\nH = 20°, S = 0.75", xy=(px, py), xytext=(1.05, 0.45), fontsize=16, va="center",
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.set_xlim(-2.0, 2.3); ax.set_ylim(-1.3, 1.15); ax.set_aspect("equal"); ax.set_axis_off()
    fig.tight_layout(); save(fig, "colour-perceptual/fig-hexagon.png", dpi=150)

if want("huewheel"):
    fig, ax = plt.subplots(figsize=(4.59, 4.43))
    for k in range(360):
        ax.add_patch(Wedge((0, 0), 1.0, k, k + 1.2, width=0.22, fc=hsv2rgb(np.array(k + 0.5), np.array(1.0), np.array(1.0)), ec="none"))
    ax.add_patch(Wedge((0, 0), 1.08, -20, 20, width=0.38, fc="none", ec=INK, lw=2.2))
    ax.text(1.12, 0.0, "red:\n340°–20°", fontsize=15, va="center")
    for hdeg in (350, 10):
        hx, hy = math.cos(math.radians(hdeg)), math.sin(math.radians(hdeg))
        ax.add_patch(FancyArrowPatch((0, 0), (0.74 * hx, 0.74 * hy), arrowstyle="-|>", mutation_scale=16, color=INK, lw=1.6))
    ax.text(0.5, 0.2, "10°", fontsize=15, ha="center"); ax.text(0.5, -0.3, "350°", fontsize=15, ha="center")
    ax.add_patch(FancyArrowPatch((0, 0), (0.72, 0), arrowstyle="-|>", mutation_scale=18, color=OK["verm"], lw=3))
    ax.add_patch(FancyArrowPatch((0, 0), (-0.74, 0), arrowstyle="-|>", mutation_scale=16, color="#777", lw=1.6, ls="--"))
    ax.text(-1.1, -1.25, "→ mean of the unit vectors: 0°, red", color=OK["verm"], fontsize=15, ha="left")
    ax.text(-1.1, -1.5, "⇠ (350 + 10)/2 = 180°: cyan, wrong", color="#555", fontsize=15, ha="left")
    ax.set_xlim(-1.15, 1.6); ax.set_ylim(-1.65, 1.1); ax.set_aspect("equal"); ax.set_axis_off()
    fig.tight_layout(); save(fig, "colour-perceptual/fig-huewheel.svg")

if want("abplane"):
    fig, axs = plt.subplots(1, 2, figsize=(5.9, 3.5), sharey=True)
    n = 300; a, b = np.meshgrid(np.linspace(-110, 110, n), np.linspace(-110, 110, n))
    for ax, L in zip(axs, (40, 75)):
        lab = np.stack([np.full_like(a, L), a, b], -1)
        rgb = lab2xyz(lab) @ M_XYZ2RGB.T
        ok = np.all((rgb >= -1e-4) & (rgb <= 1 + 1e-4), -1)
        img = np.ones((n, n, 4)); img[..., :3] = enc(rgb); img[..., 3] = ok
        ax.imshow(img, origin="lower", extent=(-110, 110, -110, 110))
        ax.axhline(0, color=INK, lw=0.7); ax.axvline(0, color=INK, lw=0.7)
        ax.set_title(f"L* = {L}", fontsize=16); ax.set_xlabel("a* (green − / red +)", fontsize=14)
        ax.set_xticks([-100, 0, 100]); ax.set_yticks([-100, 0, 100])
        ax.set_aspect("equal")
    axs[0].set_ylabel("b* (blue − / yellow +)", fontsize=14)
    axs[0].annotate("greys:\na* = b* = 0", xy=(0, 0), xytext=(-105, 65), fontsize=14, arrowprops=dict(arrowstyle="->", color=INK))
    fig.tight_layout(); save(fig, "colour-perceptual/fig-abplane.png", dpi=150)

if want("deltae"):
    pairs = [((255, 255, 0), (255, 255, 40)), ((0, 80, 0), (0, 120, 0))]
    fig, ax = plt.subplots(figsize=(5.25, 2.79))
    for i, (p, q) in enumerate(pairs):
        y = 1 - i * 1.25
        de = float(np.linalg.norm(srgb8_to_lab(p) - srgb8_to_lab(q)))
        rd = float(np.linalg.norm(np.subtract(p, q)))
        ax.add_patch(Rectangle((0, y), 1, 0.9, fc=np.array(p) / 255, ec="#999"))
        ax.add_patch(Rectangle((1, y), 1, 0.9, fc=np.array(q) / 255, ec="#999"))
        ax.text(2.2, y + 0.45, f"RGB distance {rd:.0f}\nΔE*ab {de:.1f}", va="center", fontsize=16)
    ax.set_xlim(0, 4.6); ax.set_ylim(-0.35, 2.0); ax.set_aspect("equal"); ax.set_axis_off()
    fig.tight_layout(); save(fig, "colour-perceptual/fig-deltae.svg")

if want("huehist"):
    hsv = np.array(S["hsv"]).reshape(-1, 3)
    m = np.array(S["maskH"]).astype(bool)
    ok = hsv[:, 0] >= 0
    fig = plt.figure(figsize=(4.43, 5.25))
    ax = fig.add_axes([0.17, 0.62, 0.76, 0.35])
    B = 90
    Hall, xe, ye = np.histogram2d(hsv[ok, 0], hsv[ok, 1], bins=[B, 50], range=[[0, 360], [0, 1]])
    Hsel, _, _ = np.histogram2d(hsv[ok & m, 0], hsv[ok & m, 1], bins=[B, 50], range=[[0, 360], [0, 1]])
    t = np.log1p(Hall) / np.log1p(Hall.max())
    img = np.ones(Hall.shape + (3,))
    g = np.where(Hall > 0, (175 - 150 * t) / 255, 1)
    img[...] = g[..., None]
    sel = Hsel > 0
    img[sel] = np.stack([np.ones_like(t), (140 - 60 * t) / 255, np.zeros_like(t)], -1)[sel]
    ax.imshow(img.transpose(1, 0, 2), origin="lower", extent=(0, 360, 0, 1), aspect="auto", interpolation="nearest")
    for x0, x1 in ((0, 20), (340, 360)):
        ax.add_patch(Rectangle((x0, 0.4), x1 - x0, 0.6, fc="none", ec=OK["blue"], lw=2.2))
    ax.set_xticks([0, 90, 180, 270, 360]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("hue (°)"); ax.set_ylabel("saturation")
    ax2 = fig.add_axes([0.22, 0.0, 0.56, 0.42])
    panel(ax2, masked(S["maskH"]), below=None)
    ax2.set_title(f"selected: Dice {S['diceH']:.3f}", fontsize=15)
    save(fig, "colour-perceptual/fig-huehist.png", dpi=150)

if want("kmeans"):
    K = D["kmeans"]
    fig, axs = plt.subplots(1, 2, figsize=(4.9, 3.0))
    for ax, key, sp in zip(axs, ("rgb", "lab"), ("RGB", "Lab")):
        lab = np.array(K[key]).reshape(SH, SW)
        out = np.zeros((SH, SW, 3))
        for j in range(4):
            out[lab == j] = SIMG[lab == j].mean(0)
        best = K[key + "Best"]
        edge = (lab == best["cls"])
        bd = edge & ~(np.roll(edge, 1, 0) & np.roll(edge, -1, 0) & np.roll(edge, 1, 1) & np.roll(edge, -1, 1))
        out[bd] = [1, 1, 1]
        panel(ax, out, f"k-means in {sp}", below=f"apple: Dice {best['dice']:.3f}")
    fig.tight_layout(); save(fig, "colour-perceptual/fig-kmeans.png", dpi=150)

# =========================================================================================
# colour-display
# =========================================================================================
if want("range"):
    cell = np.asarray(Image.open("demos/src/fourier2d/cell256.png").convert("L"), float)
    lo, hi = 100, 200
    fig, axs = plt.subplots(1, 2, figsize=(5.8, 3.4))
    panel(axs[0], cell / 255, "display range 0–255")
    st = np.clip((cell - lo) / (hi - lo), 0, 1)
    rgb = np.repeat(st[..., None], 3, -1)
    rgb[cell > hi] = np.array([230, 159, 0]) / 255
    rgb[cell < lo] = np.array([0, 114, 178]) / 255
    panel(axs[1], rgb, f"display range {lo}–{hi}")
    fa, fb = (cell > hi).mean(), (cell < lo).mean()
    axs[1].set_xlabel(f"clipped: {100 * fa:.0f} % white (orange)\n{100 * fb:.0f} % black (blue)", fontsize=14)
    axs[0].set_xlabel("micrograph: R. M. Hunt,\nCC BY 3.0", fontsize=14, color="#555")
    fig.tight_layout(); save(fig, "colour-display/fig-range.png", dpi=150)


def two_channels(n=128, seed=4):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n]
    a = np.zeros((n, n)); b = np.zeros((n, n))
    for _ in range(9):
        cx, cy, r = rng.uniform(12, n - 12), rng.uniform(12, n - 12), rng.uniform(5, 9)
        a += np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r * r)) * rng.uniform(0.6, 1)
    for _ in range(7):
        cx, cy, ang = rng.uniform(10, n - 10), rng.uniform(10, n - 10), rng.uniform(0, np.pi)
        d = np.abs((x - cx) * np.sin(ang) - (y - cy) * np.cos(ang)); along = np.abs((x - cx) * np.cos(ang) + (y - cy) * np.sin(ang))
        b += np.exp(-d ** 2 / 8) * (along < 35) * rng.uniform(0.6, 1)
    a = np.clip(a / a.max() + rng.normal(0, 0.02, a.shape), 0, 1)
    b = np.clip(b / b.max() + rng.normal(0, 0.02, b.shape), 0, 1)
    return a, b


def merge(a, b, ca, cb):
    return np.clip(a[..., None] * np.array(ca) + b[..., None] * np.array(cb), 0, 1)


def deutan(img):
    M = np.array(D["dichromat"]["deutan"])
    return enc(np.clip(dec(img) @ M.T, 0, 1))


if want("merge"):
    a, b = two_channels()
    fig, axs = plt.subplots(1, 3, figsize=(7.87, 3.03))
    panel(axs[0], a, "channel 1 (grey)")
    panel(axs[1], b, "channel 2 (grey)")
    panel(axs[2], merge(a, b, (1, 0, 1), (0, 1, 0)), "merge: magenta + green")
    axs[2].set_xlabel("overlap: white", fontsize=14)
    fig.tight_layout(); save(fig, "colour-display/fig-merge.png", dpi=150)

if want("deutan"):
    a, b = two_channels()
    rg, mg = merge(a, b, (1, 0, 0), (0, 1, 0)), merge(a, b, (1, 0, 1), (0, 1, 0))
    fig, axs = plt.subplots(2, 2, figsize=(4.59, 4.92))
    panel(axs[0, 0], rg, "red + green"); panel(axs[0, 1], mg, "magenta + green")
    panel(axs[1, 0], deutan(rg)); panel(axs[1, 1], deutan(mg))
    axs[0, 0].set_ylabel("normal", fontsize=15); axs[1, 0].set_ylabel("deuteranope\n(simulated)", fontsize=15)
    fig.tight_layout(); save(fig, "colour-display/fig-deutan.png", dpi=150)

if want("mapkinds"):
    n = 128; y, x = np.mgrid[0:n, 0:n] / n
    seq = 0.5 + 0.5 * np.sin(5 * x) * np.cos(4 * y) * (0.6 + 0.4 * x)
    sig = 3 * np.sin(6 * x + 1) * np.sin(5 * y) + 2 * (x - 0.3)
    lab = (np.floor(x * 4) + 4 * np.floor(y * 3)).astype(int)
    rng = np.random.default_rng(2)
    jitter = (np.floor((x * 4 + 0.3 * np.sin(9 * y)) % 4) + 4 * np.floor((y * 3 + 0.2 * np.sin(7 * x)) % 3)).astype(int)
    vir = np.array(D["maps"]["viridis"]); div = np.array(D["maps"]["diverging"])
    q = np.array([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for h in D["qualitative"]])
    fig, axs = plt.subplots(1, 3, figsize=(8.36, 2.95))
    panel(axs[0], vir[np.clip((seq * 255).astype(int), 0, 255)], "sequential: viridis")
    m = np.abs(sig).max()
    panel(axs[1], div[np.clip(((sig / m * 0.5 + 0.5) * 255).astype(int), 0, 255)], "diverging: 0 = white")
    panel(axs[2], q[jitter % len(q)], "qualitative: labels")
    fig.tight_layout(); save(fig, "colour-display/fig-mapkinds.png", dpi=150)

if want("decorr"):
    s = D["he"]; img = scene_img(s)
    X = img.reshape(-1, 3); mu = X.mean(0); C = np.cov(X.T, bias=True)
    lam, U = np.linalg.eigh(C); lam, U = lam[::-1], U[:, ::-1]
    sbar = math.sqrt(lam.mean())
    T = U @ np.diag(sbar / np.sqrt(lam)) @ U.T
    Y = (X - mu) @ T.T + mu
    clipped = np.any((Y < 0) | (Y > 1), 1).mean()
    print("decorr sigma", np.sqrt(lam), "PC1 share", lam[0] / lam.sum(), "clipped", clipped)
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 3.0))
    panel(axs[0], img, "H&E-like (synthetic)")
    panel(axs[1], np.clip(Y.reshape(img.shape), 0, 1), "decorrelation stretch")
    axs[1].set_xlabel(f"{100 * clipped:.0f} % of pixels clipped", fontsize=14)
    fig.tight_layout(); save(fig, "colour-display/fig-decorr.png", dpi=150)
