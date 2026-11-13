"""Static figures for matching/ (finding a shift by sliding and scoring).

Run from the repository root:  python3 -I tools/figs/matching/figs.py
Writes matching/fig-*.svg|png. Pure numpy; the scores are the same definitions as
demos/src/lib/registration.ts (overlapScores1d) and the template matcher of the phasecorr demo.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image

plt.rcParams.update({"font.size": 16, "axes.titlesize": 16, "svg.fonttype": "none",
                     "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, RED, GREEN, GREY, PURPLE = "#1f77b4", "#ff7f0e", "#d62728", "#2a9d4b", "#888888", "#cc79a7"
CELL = np.asarray(Image.open("demos/src/fourier2d/cell256.png")).astype(float)
OUT = "matching/"


def scores(a, b, t):
    """SSD, CC, NCC of B (length nb) placed at offset t on A (pairs A[x+t], B[x])."""
    xs = np.arange(len(b)); i = xs + t; ok = (i >= 0) & (i < len(a))
    u, v = a[i[ok]], b[ok]
    n = ok.sum()
    if n < 2:
        return np.nan, np.nan, np.nan, n
    du, dv = u - u.mean(), v - v.mean()
    den = np.sqrt((du ** 2).sum() * (dv ** 2).sum())
    return ((u - v) ** 2).sum(), (u * v).sum(), (du * dv).sum() / den if den > 0 else np.nan, n


# ------------------------------------------------------------ three cases
def three_cases():
    fig, ax = plt.subplots(1, 3, figsize=(8.6, 3.0), gridspec_kw={"wspace": 0.12})
    # (a) two tiles on the micrograph
    ax[0].imshow(CELL, cmap="gray", vmin=0, vmax=255)
    ax[0].add_patch(Rectangle((20, 50), 130, 130, fill=False, ec=BLUE, lw=3))
    ax[0].add_patch(Rectangle((112, 62), 130, 130, fill=False, ec=ORANGE, lw=3, ls="--"))
    ax[0].text(28, 44, "A", color=BLUE, fontsize=20, weight="bold")
    ax[0].text(220, 222, "B", color=ORANGE, fontsize=20, weight="bold")
    ax[0].set_title("mosaic tiles")
    # (b) drift: frame 1 and frame 50 of a crop, same marker position
    d = (13, 7)
    f1 = CELL[70:170, 70:170]
    f50 = CELL[70 - d[1]:170 - d[1], 70 - d[0]:170 - d[0]]
    both = np.concatenate([f1, np.full((100, 6), 255.0), f50], axis=1)
    ax[1].imshow(both, cmap="gray", vmin=0, vmax=255)
    for x0 in (0, 106):
        ax[1].plot([x0 + 50], [50], marker="+", ms=26, mew=3, color=ORANGE)
    ax[1].text(50, 112, "frame 1", ha="center", va="top", fontsize=16)
    ax[1].text(156, 112, "frame 50", ha="center", va="top", fontsize=16)
    ax[1].set_ylim(126, -2)
    ax[1].set_title("drift in a time-lapse")
    # (c) two colour channels, offset: magenta / green overlay
    c = CELL[60:180, 60:180] / 255.0
    g = CELL[60 + 4:180 + 4, 60 - 6:180 - 6] / 255.0
    rgb = np.dstack([c, g, c])
    ax[2].imshow(rgb)
    ax[2].set_title("two colour channels")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    fig.savefig(OUT + "fig-three-cases.png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------ slide and score
def slide_score():
    row = CELL[128, 40:168].copy()          # A: 128 samples of one micrograph row
    a = row
    t_true, L = 52, 32
    b = a[t_true:t_true + L]
    ts = np.arange(-(L - 8), len(a) - 8 + 1)
    ncc = np.array([scores(a, b, t)[2] if scores(a, b, t)[3] >= 8 else np.nan for t in ts])
    t_cur = 30
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 5.6), gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.55})
    x = np.arange(len(a))
    ax[0].plot(x, a, color=BLUE, lw=2.2, label="A")
    ax[0].plot(np.arange(L) + t_cur, b, color=ORANGE, lw=2.6, ls="--", label="B at t = %d" % t_cur)
    ax[0].axvspan(t_cur - 0.5, t_cur + L - 0.5, color=ORANGE, alpha=0.15)
    ax[0].set_xlabel("position x"); ax[0].set_ylabel("grey value")
    ax[0].legend(loc="upper left", frameon=False, fontsize=15, ncol=2, bbox_to_anchor=(0, 1.32))
    ax[0].set_xlim(-2, 130)
    ax[1].plot(ts, ncc, color=GREY, lw=2)
    ax[1].plot([t_cur], [ncc[ts == t_cur][0]], "o", color=ORANGE, ms=10)
    k = np.nanargmax(ncc)
    ax[1].plot([ts[k]], [ncc[k]], "o", color=GREEN, ms=11)
    ax[1].annotate("best: t = %d" % ts[k], (ts[k], ncc[k]), xytext=(ts[k] + 12, 0.85), fontsize=15, color=GREEN,
                   arrowprops=dict(arrowstyle="->", color=GREEN))
    ax[1].set_xlabel("candidate shift t"); ax[1].set_ylabel("score (NCC)")
    ax[1].set_ylim(-1, 1.05); ax[1].set_xlim(-26, 122)
    fig.savefig(OUT + "fig-slide-score.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("slide-score: best", ts[k], ncc[k])


# ------------------------------------------------------------ by hand
def by_hand():
    A = np.array([0, 1, 3, 1, 0, 2, 6, 2], float); B = np.array([1, 3, 1], float)
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ax.bar(np.arange(8), A, color=BLUE, alpha=0.75, width=0.7, label="A")
    for t, c, lab in ((1, ORANGE, "B at t = 1"), (5, PURPLE, "B at t = 5")):
        ax.plot(np.arange(3) + t, B, "o--", color=c, lw=2.4, ms=9, label=lab)
    ax.set_xticks(range(8)); ax.set_xlabel("position"); ax.set_yticks([0, 2, 4, 6])
    ax.legend(frameon=False, fontsize=14, loc="upper left")
    ax.text(2, 3.6, "exact copy", ha="center", fontsize=14, color=ORANGE)
    ax.text(6, 6.4, "brighter copy", ha="center", fontsize=14, color=PURPLE)
    ax.set_ylim(0, 7.4)
    fig.savefig(OUT + "fig-hand-template.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for t in range(6):
        print("hand", t, scores(A, B, t))


# ------------------------------------------------------------ cost
def cost():
    N = 1024
    w = np.unique(np.round(np.logspace(0, np.log10(N - 1), 300)).astype(int))
    ops = ((2 * w + 1) * N - w * (w + 1)) ** 2.0
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.loglog(w, ops, color=BLUE, lw=2.6)
    for ww, lab, dx in ((32, "±32 px: 4.3·10⁹\nabout 4 s", 1.6), (1023, "every shift: 1.1·10¹²\nabout 18 min", 0.03)):
        o = ((2 * ww + 1) * N - ww * (ww + 1)) ** 2.0
        ax.plot([ww], [o], "o", color=ORANGE, ms=10)
        ax.annotate(lab, (ww, o), xytext=(ww * dx, o * (0.08 if ww == 32 else 0.25)), fontsize=15,
                    ha="left" if ww == 32 else "right",
                    arrowprops=dict(arrowstyle="->", color=GREY))
        print("cost", ww, o, o / 1e9, "s")
    ax.set_xlabel("search window ± w (pixels)")
    ax.set_ylabel("multiply-adds")
    ax.set_xlim(1, 1500); ax.set_ylim(5e6, 5e12)
    sec = ax.secondary_yaxis("right", functions=(lambda y: y / 1e9, lambda s: s * 1e9))
    sec.set_ylabel("seconds at 10⁹/s")
    ax.grid(True, which="major", alpha=0.3)
    fig.savefig(OUT + "fig-cost.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------ template matching
def template():
    img, n, tx, ty = CELL, 20, 99, 59
    T = img[ty:ty + n, tx:tx + n]
    from numpy.lib.stride_tricks import sliding_window_view as sw
    W = sw(img, (n, n))                       # (237, 237, n, n), index [y, x]
    Tm = T - T.mean()
    Wm = W - W.mean(axis=(2, 3), keepdims=True)
    num = (Wm * Tm).sum(axis=(2, 3))
    den = np.sqrt((Wm ** 2).sum(axis=(2, 3)) * (Tm ** 2).sum())
    m = num / den
    full = np.full(img.shape, np.nan); full[:m.shape[0], :m.shape[1]] = m
    # local maxima >= 0.5 (radius n/2)
    r = n // 2; peaks = []
    for y in range(m.shape[0]):
        for x in range(m.shape[1]):
            v = m[y, x]
            if v < 0.5:
                continue
            y0, y1, x0, x1 = max(0, y - r), y + r + 1, max(0, x - r), x + r + 1
            if v >= np.nanmax(m[y0:y1, x0:x1]):
                peaks.append((v, x, y))
    peaks.sort(reverse=True)
    print("template peaks", [(x, y, round(v, 2)) for v, x, y in peaks[:6]])
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.8), gridspec_kw={"wspace": 0.08})
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    ax[0].add_patch(Rectangle((tx - 0.5, ty - 0.5), n, n, fill=False, ec="#ffd200", lw=3))
    for v, x, y in peaks[1:4]:
        ax[0].add_patch(Rectangle((x - 0.5, y - 0.5), n, n, fill=False, ec="#00e5ff", lw=2.5))
    ax[0].set_title("image A, template (yellow)")
    cm = plt.get_cmap("RdBu_r").copy(); cm.set_bad("#dddddd")
    im = ax[1].imshow(np.ma.masked_invalid(full), cmap=cm, vmin=-1, vmax=1)
    for v, x, y in peaks[:4]:
        ax[1].plot([x], [y], "o", mfc="none", mec="k", ms=12, mew=2)
        ax[1].text(x + 7 if x < 180 else x - 7, y - 6 if y > 30 else y + 22, "%.2f" % v, fontsize=15, color="k", ha="left" if x < 180 else "right")
    ax[1].set_title("NCC map")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03, ticks=[-1, 0, 1])
    fig.savefig(OUT + "fig-template-map.png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    three_cases(); slide_score(); by_hand(); cost(); template()
