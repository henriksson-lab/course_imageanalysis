"""Shared helpers for the figure scripts of the edge-preserving smoothing day (heat-equation/,
perona-malik/, tv-flow/, bilateral-nlm/, inpainting/, smoothers/).

Image results are computed with the course TypeScript libraries (demos/src/lib/pde.ts,
inpaint.ts, smooth.ts, denoise.ts) by a small node script per unit (tools/figs/<unit>/data.ts),
so the figures show exactly what the demos compute; 1D examples are computed here in numpy.
Run every script from the repository root, e.g.  python3 -I tools/figs/heat-equation/figs.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

FONT = 17
plt.rcParams.update({
    "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
    "xtick.labelsize": FONT - 2, "ytick.labelsize": FONT - 2, "legend.fontsize": FONT - 3,
    "svg.fonttype": "path", "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.facecolor": "white", "svg.hashsalt": "pdesmooth",
})
BLUE, ORANGE, GREEN, RED, PURPLE, CYAN, YELLOW, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#777777")
CREDIT = "Micrograph: R. M. Hunt, CC BY 3.0, Wikimedia Commons"
ROOT = os.getcwd()


def cell128():
    a = np.asarray(Image.open("demos/src/fourier2d/cell256.png").convert("L")).astype(float)
    return a.reshape(128, 2, 128, 2).mean(axis=(1, 3))


def run_node(unit):
    """Run tools/figs/<unit>/data.ts with the course libraries; returns its JSON output.
    The micrograph (128 × 128, area-averaged) is passed in a temporary JSON file."""
    tmp = os.path.join("tools", "figs", unit, ".cell128.json")
    with open(tmp, "w") as f:
        json.dump(cell128().ravel().tolist(), f)
    try:
        js = subprocess.run(["node_modules/.bin/esbuild", f"tools/figs/{unit}/data.ts", "--bundle",
                             "--platform=node", "--log-level=error"], check=True, capture_output=True, text=True).stdout
        out = subprocess.run(["node", "-", tmp], input=js, check=True, capture_output=True, text=True).stdout
    finally:
        os.remove(tmp)
    return json.loads(out)


def arr(a, n=128):
    a = np.asarray(a, float)
    return a.reshape(-1, n) if a.ndim == 1 else a


def show(ax, img, lo=0, hi=255, title=None, cmap="gray"):
    ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=FONT - 1)


def save(fig, path, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.06, **kw)
    plt.close(fig)
    print("wrote", path)


# ---- 1D helpers (same conventions as demos/src/lib/pde.ts) ----

def heat1(u, dt, border="mirror"):
    u = np.asarray(u, float)
    p = np.pad(u, 1, mode="edge" if border == "mirror" else "constant")
    return u + dt * (p[:-2] + p[2:] - 2 * u)


def g_of(kind, s, K):
    if kind == "linear":
        return np.ones_like(np.asarray(s, float))
    q = (np.asarray(s, float) / K) ** 2
    return np.exp(-q) if kind == "g1" else 1 / (1 + q)


def pm1(u, dt, K, kind="g2"):
    u = np.asarray(u, float)
    d = np.diff(u)
    f = g_of(kind, np.abs(d), K) * d
    out = u.copy()
    out[:-1] += dt * f
    out[1:] -= dt * f
    return out


def gauss1(u, sigma):
    """Gaussian blur with the half-sample symmetric border (as gaussianHeat)."""
    u = np.asarray(u, float)
    if sigma <= 0:
        return u.copy()
    r = max(1, int(np.ceil(4 * sigma)))
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma * sigma)); k /= k.sum()
    p = np.pad(u, r, mode="symmetric")
    return np.convolve(p, k, mode="valid")


if __name__ == "__main__":
    sys.exit("helper module")
