"""Shared helpers for the Radon day's slide figures (radon/, backprojection/, fbp/,
algebraic-recon/). Style as the other days' figure scripts (Okabe-Ito colours, font 17).

Data computed with the demos' own library: compute.ts in this folder, bundled with esbuild
and run by ensure_data() below (needs node_modules/.bin/esbuild and node).
"""
import os
import json
import subprocess
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT = 17
plt.rcParams.update({
    "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
    "xtick.labelsize": FONT - 2, "ytick.labelsize": FONT - 2, "legend.fontsize": FONT - 2,
    "svg.fonttype": "path", "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.facecolor": "white", "font.family": "DejaVu Sans",
})
BLUE, ORANGE, GREEN, RED, PURPLE, CYAN, YELLOW, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#777777")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.environ.get("RADON_FIG_DATA", os.path.join(tempfile.gettempdir(), "radon-fig-data"))
N = 128


def ensure_data():
    if os.path.exists(os.path.join(DATA, "data.json")):
        return
    os.makedirs(DATA, exist_ok=True)
    js = os.path.join(DATA, "compute.js")
    subprocess.run([os.path.join(REPO, "node_modules/.bin/esbuild"), os.path.join(HERE, "compute.ts"), "--bundle",
                    "--platform=node", "--outfile=" + js, "--log-level=warning"], check=True, cwd=REPO)
    subprocess.run(["node", js, DATA], check=True, cwd=REPO)


def load(name, n=N):
    ensure_data()
    return np.fromfile(os.path.join(DATA, name + ".f64"), dtype="<f8").reshape(n, n)


def data():
    ensure_data()
    with open(os.path.join(DATA, "data.json")) as f:
        return json.load(f)


def save(fig, unit, name, **kw):
    path = os.path.join(REPO, unit, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.06, **kw)
    plt.close(fig)
    print("wrote", os.path.relpath(path, REPO))


def show(ax, img, lo=None, hi=None, cmap="gray", title=None):
    ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=FONT)
