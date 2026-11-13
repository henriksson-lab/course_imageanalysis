# Shared style for the geometry-from-images day's slide figures (tools/figs/<unit>/figs.py).
# Figures are drawn at 1 inch = 72 display pixels: a figure meant to be shown W px wide is
# W / 72 inches wide, so a font size in points is its size in pixels on the slide.
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20, "xtick.labelsize": 19, "ytick.labelsize": 19,
    "legend.fontsize": 19, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.sans-serif": ["Lato", "DejaVu Sans"], "figure.facecolor": "white",
    "savefig.facecolor": "white", "image.interpolation": "nearest",
})
# Okabe–Ito (colour-blind safe)
BLUE, ORANGE, GREEN, RED, PURPLE, SKY, YELLOW, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#777777"
CAMS = [BLUE, ORANGE, GREEN, PURPLE, RED, SKY, "#8c564b", "#555555", "#bcbd22", "#17becf"]


def px(w, h):
    """Figure size for a figure displayed w × h pixels."""
    return (w / 72, h / 72)


def save(fig, unit, name, tight=True):
    path = os.path.join(REPO, unit, name)
    if tight:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=150 if name.endswith(".png") else None)
    else:
        fig.savefig(path, dpi=150 if name.endswith(".png") else None)
    plt.close(fig)
    print("wrote", path)


def load(tmp, name):
    with open(os.path.join(tmp, name)) as f:
        return json.load(f)


def cell256():
    from skimage import io
    a = io.imread(os.path.join(REPO, "demos/src/fourier2d/cell256.png"), as_gray=True).astype(float)
    return a / 255 if a.max() > 1 else a      # values in [0, 1]
