"""Shared helpers for the keypoints-day figure scripts (keypoints/, descriptors/,
feature-matching/, feature-alignment/). Data come from export.ts (see its header)."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT = 17
plt.rcParams.update({
    "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
    "xtick.labelsize": FONT - 2, "ytick.labelsize": FONT - 2, "legend.fontsize": FONT - 2,
    "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN, RED, PURPLE, CYAN, YELLOW, GREY = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf", "#ffd400", "#7f7f7f")

DATA = None


def load(name):
    with open(os.path.join(DATA, name + ".json")) as f:
        return json.load(f)


def arr(im):
    return np.array(im["data"], dtype=float).reshape(im["h"], im["w"])


def show(ax, im, lo=0, hi=255, cmap="gray"):
    """lo = hi = None: auto range."""
    a = arr(im) if isinstance(im, dict) else im
    ax.imshow(a, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def save(fig, out, name, **kw):
    fig.savefig(os.path.join(out, name), bbox_inches="tight", pad_inches=0.04, **kw)
    plt.close(fig)
    print("wrote", os.path.join(out, name))


def apply(T, x, y):
    T = np.asarray(T)
    w = T[6] * x + T[7] * y + T[8]
    return (T[0] * x + T[1] * y + T[2]) / w, (T[3] * x + T[4] * y + T[5]) / w


def kpcircle(ax, k, color, lw=1.5, tick=True, scale=2.0, fill=None):
    from matplotlib.patches import Circle
    r = max(2.0, scale * k["s"])
    ax.add_patch(Circle((k["x"], k["y"]), r, fill=fill is not None, fc=fill or "none", ec=color, lw=lw))
    if tick:
        ax.plot([k["x"], k["x"] + r * np.cos(k["t"])], [k["y"], k["y"] - r * np.sin(k["t"])], color=color, lw=lw)


def side_by_side(v1, v2, gap=12):
    a, b = arr(v1), arr(v2)
    h = a.shape[0]
    return np.hstack([a, np.full((h, gap), 255.0), b]), a.shape[1] + gap
