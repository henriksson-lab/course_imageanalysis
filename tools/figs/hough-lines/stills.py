"""Helpers shared by the Hough day's figure scripts (hough-lines, hough-circles, ght, ransac).

Demo stills come from screenshots made by capture.sh (same folder): each demo draws three
square panels side by side; panel(name, i) returns panel i (0, 1, 2) of screenshot `name`.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

STYLE = {"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
         "axes.titlesize": 14, "axes.spines.top": False, "axes.spines.right": False,
         "savefig.facecolor": "white", "figure.facecolor": "white"}
plt.rcParams.update(STYLE)

# tab10, as in the existing figures of the day
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c8c8c"

PANEL_X = [(10, 319), (329, 638), (648, 957)]
PANEL_Y = (50, 359)


def shot(dirname, name):
    return np.asarray(Image.open(os.path.join(dirname, name + ".png")).convert("RGB"))


def panel(dirname, name, i):
    a = shot(dirname, name)
    x0, x1 = PANEL_X[i]
    return a[PANEL_Y[0] + 1:PANEL_Y[1] - 1, x0 + 1:x1 - 1]


def grid_of_panels(cells, titles, out, ncols, w=6.0, h=None, fontsize=14):
    """cells: list of RGB arrays; titles under/over each; save to `out` (PNG)."""
    n = len(cells)
    nrows = (n + ncols - 1) // ncols
    cw = w / ncols
    h = h or nrows * (cw + 0.45)
    fig, axs = plt.subplots(nrows, ncols, figsize=(w, h), squeeze=False)
    for k, ax in enumerate(axs.flat):
        ax.axis("off")
        if k < n:
            ax.imshow(cells[k], interpolation="nearest")
            ax.set_title(titles[k], fontsize=fontsize, pad=4)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.05, hspace=0.25)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
