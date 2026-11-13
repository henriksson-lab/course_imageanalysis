"""Shared plotting helpers for the colocalisation day's figures (coloc, coloc-null,
coloc-objects, coloc-confounds). Figures are sized in display pixels: a figure made with
fig(w, h) is meant to be shown about w px wide on a 1280 x 720 slide, so an 18 pt font
appears at about 18 px."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.size": 18, "font.family": "DejaVu Sans", "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 18,
    "axes.labelsize": 18, "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 16,
    "svg.hashsalt": "coloc", "figure.facecolor": "white", "axes.facecolor": "white",
})

# Okabe-Ito (colour-blind safe)
BLUE, ORANGE, VERM, SKY, GREEN, PURPLE, YELLOW, GREY = "#0072B2", "#E69F00", "#D55E00", "#56B4E9", "#009E73", "#CC79A7", "#F0E442", "#777777"
S = 128


def fig(wpx, hpx, *a, **kw):
    return plt.subplots(*a, figsize=(wpx / 72, hpx / 72), **kw)


def save(f, path):
    f.savefig(path, facecolor="white", dpi=144, metadata={"Date": None})
    plt.close(f)
    print("wrote", path)


def img(v):
    return np.asarray(v, dtype=float).reshape(S, S) if np.size(v) == S * S else np.asarray(v, float)


def merge(a, b, ra=None, rb=None):
    """Green/magenta merge as in the demos: A green, B red + blue, each scaled to 0.8 x max."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = ra or (0, 0.8 * a.max())
    rb = rb or (0, 0.8 * b.max())
    g = np.clip((a - ra[0]) / (ra[1] - ra[0]), 0, 1)
    m = np.clip((b - rb[0]) / (rb[1] - rb[0]), 0, 1)
    return np.dstack([m, g, m])


def noaxis(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def density(ax, x, y, lim=(0, 200), bins=80, cmap="viridis"):
    """2D histogram (log counts, empty bins white) of the pixel pairs."""
    from matplotlib.colors import LogNorm
    ax.hist2d(x, y, bins=bins, range=[lim, lim], cmap=cmap, norm=LogNorm(), cmin=1, rasterized=True)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def num(v, nd=2):
    """Number with a typographic minus sign."""
    return f"{v:.{nd}f}".replace("-", "\u2212")
