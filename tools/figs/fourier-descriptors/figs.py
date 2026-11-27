"""Static figures for fourier-descriptors/ (outlines as periodic signals).

Run from the repository root:  python3 -I tools/figs/fourier-descriptors/figs.py
Writes fourier-descriptors/fig-*.svg. The outlines, descriptors and match scores come from
lib/contour.ts and the fdmatch demo's model (tools/figs/fourier-descriptors/data.ts, run with
node), so the numbers match the demo.
"""
import json, os, subprocess, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

plt.rcParams.update({"font.size": 16, "axes.titlesize": 16, "svg.fonttype": "none",
                     "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, GREY, PURPLE, ORANGE = "#1f77b4", "#d62728", "#2a9d4b", "#888888", "#6a3d9a", "#ff7f0e"
OUT = "fourier-descriptors/"


def data():
    with tempfile.TemporaryDirectory() as d:
        js = os.path.join(d, "fd.json")
        subprocess.run("node_modules/.bin/esbuild tools/figs/fourier-descriptors/data.ts --bundle --platform=node "
                       "--log-level=warning | node - %s" % js, shell=True, check=True)
        return json.load(open(js))


def arr(p):
    return np.array(p, dtype=float)


def closed(p):
    p = arr(p)
    return np.vstack([p, p[:1]])


def outline_ax(ax, pad=0.15):
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.margins(pad)


def start_marker(ax, z, col, size=9):
    z = arr(z)
    ax.plot(*z[0], "o", ms=size, mfc=col, mec="k", mew=1.8, zorder=5)
    d = z[1] - z[0]
    ax.annotate("", xy=z[0] + 0.9 * d, xytext=z[0], arrowprops=dict(arrowstyle="-|>", color="k", lw=1.8), zorder=6)


def save(fig, name):
    fig.savefig(OUT + name, bbox_inches="tight")
    if os.environ.get("PREVIEW_DIR"):      # optional PNG copies for a quick look
        fig.savefig(os.path.join(os.environ["PREVIEW_DIR"], name.replace(".svg", ".png")), bbox_inches="tight", dpi=60)
    plt.close(fig)
    print("wrote", OUT + name)


def C(c, k):
    v = c[str(k)]
    return complex(v[0], v[1])


# ------------------------------------------------------------ an outline is a periodic signal
def signal(D):
    z, dense = arr(D["signal"]["z"]), arr(D["signal"]["dense"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [0.8, 1.4], "wspace": 0.3})
    a = ax[0]
    a.plot(*closed(dense).T, color=GREY, lw=1.5)
    a.plot(*z.T, "o", color=BLUE, ms=6)
    for n in [0, 4, 8, 12]:
        off = (z[n] - z.mean(0)) * 0.22
        a.text(*(z[n] + off), "$z_{%d}$" % n, ha="center", va="center", fontsize=15)
    start_marker(a, z, BLUE)
    outline_ax(a, 0.2)
    a.set_title("16 points on a leaf")
    b = ax[1]
    n = np.arange(32)
    zz = np.concatenate([z, z])
    b.plot(n, zz[:, 0], "o-", color=BLUE, label="$x_n$ (real part)")
    b.plot(n, zz[:, 1], "s-", color=ORANGE, label="$y_n$ (imaginary part)")
    b.axvline(16, color=GREY, ls="--", lw=1.5)
    b.text(16.4, 0.95, "the same again", color=GREY, fontsize=14, va="top")
    b.set_xlabel("sample $n$"); b.set_xticks([0, 8, 16, 24, 32])
    b.legend(frameon=False, fontsize=14, loc="lower left")
    b.set_title("$z_n = x_n + i\\,y_n$: periodic, period $N = 16$")
    save(fig, "fig-signal.svg")


# ------------------------------------------------------------ sampling a traced outline
def sampling(D):
    s = D["sampling"]; w = s["w"]; m = np.array(s["mask"]).reshape(w, w)
    tr = arr(s["trace"])
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.6), gridspec_kw={"wspace": 0.08})
    for a, key, col, title in [(ax[0], "idx", ORANGE, "equal steps along the pixel chain"), (ax[1], "arc", BLUE, "equal steps in arc length")]:
        a.imshow(np.where(m, 0.82, 1.0), cmap="gray", vmin=0, vmax=1, extent=[-0.5, w - 0.5, w - 0.5, -0.5])
        a.plot(*closed(tr).T, color=GREY, lw=1.2)
        p = arr(s[key])
        a.plot(*closed(p).T, color=col, lw=1.2, alpha=0.6)
        a.plot(*p.T, "o", color=col, ms=7)
        gaps = np.hypot(*(np.roll(p, -1, 0) - p).T)
        a.set_title(title, fontsize=15, pad=16)
        ys, xs = np.nonzero(m)
        a.text((xs.min() + xs.max()) / 2, ys.max() + 1.2, "gaps %.2f to %.2f px" % (gaps.min(), gaps.max()), ha="center", va="top", fontsize=14)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        a.set_xlim(xs.min() - 1.5, xs.max() + 1.5); a.set_ylim(ys.max() + 3.2, ys.min() - 1.5)
    save(fig, "fig-sampling.svg")


# ------------------------------------------------------------ epicycles
def epicycles_fig(D):
    e = D["epi"]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    full = arr(e["curves"][3])
    ax.plot(*closed(full).T, color=GREY, lw=1.5, label="the outline")
    for K, col, ls in [(1, ORANGE, "--"), (3, BLUE, "-")]:
        cur = arr(e["curves"][[1, 2, 3, 32].index(K)])
        ax.plot(*closed(cur).T, color=col, lw=2.5, ls=ls, label="$|k| \\leq %d$" % K)
    for link in e["chain"]:
        if link["k"] == 0:
            continue
        c = link["centre"]; t = link["tip"]
        ax.add_patch(Circle(c, link["r"], fill=False, ec="#555", lw=1, alpha=0.7))
        ax.plot([c[0], t[0]], [c[1], t[1]], color="k", lw=1.6)
    tip = e["chain"][-1]["tip"]
    ax.plot(*tip, "o", color=BLUE, ms=8, mec="w")
    outline_ax(ax, 0.08)
    ax.legend(frameon=False, fontsize=14, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig-epicycles.svg")


# ------------------------------------------------------------ smoothing by truncation (recall)
def truncate(D):
    e = D["epi"]
    full = arr(e["curves"][3])
    fig, ax = plt.subplots(1, 4, figsize=(9.5, 3.6), gridspec_kw={"wspace": 0.05})
    for a, K, cur in zip(ax, [1, 2, 3, 32], e["curves"]):
        a.plot(*closed(full).T, color=GREY, lw=1.2)
        a.plot(*closed(cur).T, color=BLUE, lw=2.5)
        outline_ax(a, 0.05)
        a.set_title("$|k| \\leq %d$" % K if K < 32 else "all 64 samples")
    save(fig, "fig-truncate.svg")


# ------------------------------------------------------------ the rules
def rules(D):
    R = D["rules"]
    order = [("original", "outline A"), ("moved", "moved: only $c_0$"), ("scaled", "scaled: all $\\times s$"),
             ("turned", "turned: all $\\times e^{i\\alpha}$"), ("restarted", "new start: $\\times e^{2\\pi ikm/N}$"),
             ("reversed", "other way round:\n$c_k \\leftrightarrow c_{-k}$"), ("mirrored", "mirror: conjugate\nand swap")]
    fig, ax = plt.subplots(2, 7, figsize=(15, 4.6), gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.15, "wspace": 0.25})
    ref = arr(R["original"]["z"])
    for j, (key, title) in enumerate(order):
        a = ax[0, j]
        z = arr(R[key]["z"])
        a.plot(*closed(ref).T, color=GREY, lw=1, alpha=0.5)
        a.plot(*closed(z).T, color=BLUE if key == "original" else RED, lw=2)
        start_marker(a, z, BLUE if key == "original" else RED, 7)
        a.set_xlim(-1.1, 1.7); a.set_ylim(1.4, -1.4); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        a.set_title(title, fontsize=13)
        # coefficient arrows k = -2..2 (except 0), c_k as arrows from the origin
        b = ax[1, j]
        c = R[key]["c"]
        for k, col in [(1, PURPLE), (-1, GREEN)]:
            v = C(c, k)
            b.annotate("", xy=(v.real, v.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
        b.plot(0, 0, "k.", ms=4)
        b.set_xlim(-0.8, 0.8); b.set_ylim(0.8, -0.8); b.set_aspect("equal")
        b.set_xticks([]); b.set_yticks([])
        for s in b.spines.values():
            s.set_visible(False)
        b.add_patch(Circle((0, 0), 0.75, fill=False, ec="#ddd", lw=1))
        if j == 0:
            for k, col, lab in [(1, PURPLE, "$c_1$"), (-1, GREEN, "$c_{-1}$")]:
                v = C(c, k)
                b.text(v.real * 1.25, v.imag * 1.25, lab, color=col, fontsize=14, ha="center", va="center")
    save(fig, "fig-rules.svg")


# ------------------------------------------------------------ by hand: rotation or new start?
def rotstart(D):
    R = D["rotstart"]
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 5.6), gridspec_kw={"height_ratios": [1, 1.05], "hspace": 0.25})
    t = np.linspace(0, 2 * np.pi, 200)
    for j, (key, title) in enumerate([("original", "$z = (2, i, -2, -i)$"), ("turned", "turned 90°: $\\times i$"), ("restarted", "started one point later")]):
        a = ax[0, j]
        z = arr(R[key]["z"])
        el = np.c_[2 * np.cos(t), np.sin(t)]
        if key == "turned":
            el = np.c_[-np.sin(t), 2 * np.cos(t)]
        a.plot(*el.T, color=GREY, lw=1.5)
        a.plot(*z.T, "o", color=BLUE if key == "original" else RED, ms=8)
        for n, p in enumerate(z):
            a.text(p[0] * 1.28 + (0.25 if abs(p[0]) < 1e-9 else 0), p[1] * 1.45, "$z_%d$" % n, ha="center", va="center", fontsize=14)
        start_marker(a, z, BLUE if key == "original" else RED, 9)
        a.set_xlim(-2.8, 2.8); a.set_ylim(2.6, -2.6); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        a.set_title(title, fontsize=15, pad=16)
        b = ax[1, j]
        c = R[key]["c"]
        for k, col, lab in [(1, PURPLE, "$c_1$"), (-1, GREEN, "$c_{-1}$")]:
            v = C(c, k)
            b.annotate("", xy=(v.real, v.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
            txt = "%s = %s" % (lab, fmtc(v))
            dy = (-0.35 if k == 1 else 0.35) if abs(v.imag) < 1e-9 else 0.0
            b.text(v.real * 1.05 + (0.12 if abs(v.real) < 1e-9 else -0.2), v.imag * 1.1 + dy, txt, color=col, fontsize=14,
                   ha="left", va="center")
        b.axhline(0, color="#ccc", lw=1); b.axvline(0, color="#ccc", lw=1)
        b.set_xlim(-1.8, 2.6); b.set_ylim(1.8, -1.8); b.set_aspect("equal")
        b.set_xticks([]); b.set_yticks([])
        for s in b.spines.values():
            s.set_visible(False)
        if j == 0:
            b.text(-1.75, -1.6, "complex plane\n(imaginary axis down)", fontsize=12, color=GREY, va="top")
    save(fig, "fig-rotstart.svg")


def fmtc(v):
    re, im = v.real, v.imag
    re = 0 if abs(re) < 1e-9 else re
    im = 0 if abs(im) < 1e-9 else im
    f = lambda x: {0.5: "1/2", 1.5: "3/2"}.get(round(x, 6), "%g" % x)
    if im == 0:
        return f(re)
    if re == 0:
        return ("−" if im < 0 else "") + f(abs(im)) + "i"
    return "%s %s %si" % (f(re), "−" if im < 0 else "+", f(abs(im)))


# ------------------------------------------------------------ a square sampled at its corners
def square(D):
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.7), gridspec_kw={"wspace": 0.1})
    sq = closed([[1, 0], [0, 1], [-1, 0], [0, -1]])
    for a, s in zip(ax, D["square"]):
        a.plot(*sq.T, color=GREY, lw=1.5, ls="--")
        a.plot(*closed(s["curve"]).T, color=BLUE, lw=2.5)
        a.plot(*arr(s["z"]).T, "o", color=RED, ms=7)
        a.set_xlim(-1.3, 1.3); a.set_ylim(1.3, -1.3); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        c1, cm3 = C(s["c"], 1), C(s["c"], -3) if "-3" in s["c"] else 0
        a.set_title("N = %d\n$c_1$ = %.3f" % (s["n"], c1.real) + ("" if s["n"] == 4 else ", $c_{-3}$ = %.3f" % cm3.real), fontsize=14)
    save(fig, "fig-square.svg")


# ------------------------------------------------------------ same magnitudes, different shape
def samemag(D):
    s = D["samemag"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.5], "wspace": 0.3})
    for a, key, col, title in [(ax[0], "a", BLUE, "shape S"), (ax[1], "b", RED, "shape S′")]:
        a.plot(*closed(s[key]).T, color=col, lw=2.5)
        outline_ax(a, 0.1)
        a.set_title(title)
    b = ax[2]
    ks = [-2, -1, 1, 3]
    for k in ks:
        va, vb = C(s["ca"], k), C(s["cb"], k)
        b.bar(k - 0.17, abs(va), width=0.3, color=BLUE)
        b.bar(k + 0.17, abs(vb), width=0.3, color=RED)
        if np.sign(va.real) != np.sign(vb.real):
            b.text(k - 0.3, abs(va) + 0.12, "$c_3$ = +0.12\nvs −0.12", ha="center", fontsize=13)
    b.set_xticks(range(-3, 4)); b.set_xlabel("k"); b.set_ylabel("$|c_k|$")
    b.set_ylim(0, 1.15)
    b.set_title("identical sizes $|c_k|$")
    save(fig, "fig-samemag.svg")


# ------------------------------------------------------------ matching two leaves
def match(D):
    s = D["match"]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.9), gridspec_kw={"width_ratios": [0.8, 0.8, 1.5, 0.9], "wspace": 0.3})
    a, b = arr(s["a"]), arr(s["b"])
    ax[0].plot(*closed(a).T, "o-", color=BLUE, ms=3, lw=1.5); start_marker(ax[0], a, BLUE); outline_ax(ax[0], 0.12); ax[0].set_title("outline A")
    ax[1].plot(*closed(b).T, "o-", color=RED, ms=3, lw=1.5); start_marker(ax[1], b, RED); outline_ax(ax[1], 0.12); ax[1].set_title("outline B")
    c = ax[2]
    sc = np.array(s["scores"]); m = np.arange(len(sc))
    c.plot(m, sc, "o-", color=PURPLE, ms=3)
    c.plot(s["m"], sc[s["m"]], "o", ms=11, mfc=PURPLE, mec="k")
    c.annotate("m = %d: %.1f°, × %.3f" % (s["m"], s["angle"], s["scale"]), (s["m"], sc[s["m"]]), (s["m"] + 6, sc[s["m"]] - 0.02), fontsize=14,
               arrowprops=dict(arrowstyle="-", color="k"))
    sm, sv = s["second"]
    c.annotate("%.3f at m = %d: half a turn" % (sv, sm), (sm, sv), (sm - 16, 0.64), fontsize=14, ha="center", arrowprops=dict(arrowstyle="-", color="k"))
    c.set_ylim(0.6, 1.02)
    c.set_xlabel("start m"); c.set_ylabel("score"); c.set_xlim(0, 63)
    c.set_title("score for every start: one FFT")
    d = ax[3]
    al = arr(s["aligned"])
    d.plot(*closed(b).T, color=RED, lw=2.5)
    d.plot(*closed(al).T, color=BLUE, lw=2, ls="--")
    outline_ax(d, 0.12)
    d.set_title("A moved onto B")
    save(fig, "fig-match.svg")


# ------------------------------------------------------------ mirror image
def mirror_fig(D):
    s = D["mirror"]
    fig, ax = plt.subplots(1, 4, figsize=(14.5, 3.8), gridspec_kw={"width_ratios": [0.85, 0.85, 1.4, 0.85], "wspace": 0.28})
    a, b = arr(s["a"]), arr(s["b"])
    ax[0].plot(*closed(a).T, color=BLUE, lw=2); start_marker(ax[0], a, BLUE, 7); outline_ax(ax[0], 0.1); ax[0].set_title("A: a hand")
    ax[1].plot(*closed(b).T, color=RED, lw=2); start_marker(ax[1], b, RED, 7); outline_ax(ax[1], 0.1); ax[1].set_title("B: mirrored, turned")
    c = ax[2]
    m = np.arange(len(s["scores"]))
    c.plot(m, s["scores"], "-", color=PURPLE, lw=2, label="A")
    c.plot(m, s["scoresMirror"], "--", color=GREY, lw=2, label="mirror image of A")
    c.set_xlabel("start m"); c.set_ylabel("score"); c.set_xlim(0, len(m) - 1)
    c.legend(frameon=False, fontsize=13, loc="lower center")
    c.set_title("best: %.3f (A), 1.000 (mirror)" % s["no"]["score"], fontsize=15)
    d = ax[3]
    d.plot(*closed(b).T, color=RED, lw=2.5)
    d.plot(*closed(s["alignedNo"]).T, color=BLUE, lw=2, ls="--")
    outline_ax(d, 0.1)
    d.set_title("best fit of A itself", fontsize=15)
    save(fig, "fig-mirror.svg")


# ------------------------------------------------------------ z(t) vs r(theta)
def star(D):
    s = D["star"]; z = arr(s["z"]); c = s["centre"]
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.plot(*closed(z).T, color=BLUE, lw=2.5)
    ax.plot(*c, "k+", ms=12, mew=2)
    for th in np.deg2rad([-110, -80, -62, -100]):
        d = np.array([np.cos(th), np.sin(th)])
        # crossings of the ray with the outline
        P = z - c
        ang = np.arctan2(P[:, 1], P[:, 0])
        ax.plot([c[0], c[0] + 1.25 * d[0]], [c[1], c[1] + 1.25 * d[1]], color=GREY, lw=1.2)
        for i in range(len(P)):
            j = (i + 1) % len(P)
            a1, a2 = ang[i] - th, ang[j] - th
            a1 = (a1 + np.pi) % (2 * np.pi) - np.pi; a2 = (a2 + np.pi) % (2 * np.pi) - np.pi
            if a1 * a2 < 0 and abs(a1 - a2) < 1:
                f = a1 / (a1 - a2); p = P[i] + f * (P[j] - P[i])
                if p @ d > 0:
                    ax.plot(*(c + p), "o", color=RED, ms=6)
    outline_ax(ax, 0.0)
    ax.set_xlim(z[:, 0].min() - 0.05, z[:, 0].max() + 0.05); ax.set_ylim(z[:, 1].max() + 0.05, z[:, 1].min() - 0.05)
    save(fig, "fig-star.svg")


# ------------------------------------------------------------ elliptic Fourier descriptors
def efd(D):
    s = D["efd"]; z = arr(s["z"]); el = arr(s["ellipse"]); e = s["e"]; c = s["centre"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(*closed(z).T, color=BLUE, lw=2.5, label="outline")
    ax.plot(*closed(el).T, color=ORANGE, lw=2, ls="--", label="first ellipse ($c_{\\pm1}$)")
    d = np.array([np.cos(e["psi"]), np.sin(e["psi"])])
    ax.plot([c[0] - e["a"] * d[0], c[0] + e["a"] * d[0]], [c[1] - e["a"] * d[1], c[1] + e["a"] * d[1]], color=ORANGE, lw=1.5)
    ax.plot(*s["startPt"], "o", ms=10, mfc=ORANGE, mec="k", mew=1.5)
    ax.text(s["startPt"][0], s["startPt"][1] - 0.15, "new start", ha="center", fontsize=13)
    outline_ax(ax, 0.1)
    ax.legend(frameon=False, fontsize=13, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    save(fig, "fig-efd.svg")


# ------------------------------------------------------------ staircase: spectra
def staircase(D):
    s = D["staircase"]
    k = np.arange(-32, 33)
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for key, col, lab, mk in [("exact", "k", "the polygon itself", "o"), ("arc", BLUE, "traced, arc length", "s"), ("index", ORANGE, "traced, equal steps in index", "^")]:
        v = np.maximum(np.array(s[key]), 1e-5)
        ax.semilogy(k, v, mk + "-", color=col, ms=4, lw=1.2, label=lab)
    ax.set_ylim(1e-4, 1.5); ax.set_xlim(-32, 32)
    ax.set_xlabel("k"); ax.set_ylabel("$|c_k| / |c_1|$")
    ax.legend(frameon=False, fontsize=13, loc="upper left")
    save(fig, "fig-staircase.svg")


if __name__ == "__main__":
    D = data()
    signal(D); sampling(D); epicycles_fig(D); truncate(D); rules(D); rotstart(D); square(D); samemag(D)
    match(D); mirror_fig(D); star(D); efd(D); staircase(D)
