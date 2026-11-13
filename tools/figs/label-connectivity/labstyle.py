# Shared matplotlib style for the labelling-day figures (label-connectivity/, regionprops/,
# shape-features/, label-measure/). Figures are sized in points = display pixels
# (figsize = px / 72), so font size 18 shows as 18 px at the SVG's nominal width.
# Colours follow the hand-drawn labelling figures: pixel grid #bbb, object pixels #a6c4f4,
# accents from the tab10 palette.
import os, json, subprocess, tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#888888", "#9467bd"
PIX, PIXLIGHT, GRID, INK = "#a6c4f4", "#d9e4f5", "#bbbbbb", "#222222"
# label palette (Okabe-Ito, colour-blind safe), neighbouring labels very different
LABELS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"]


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 18, "axes.labelsize": 18,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
        "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def fig(wpx, hpx, **kw):
    return plt.subplots(figsize=(wpx / 72, hpx / 72), **kw)


def save(f, path):
    f.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(f)


def save_png(f, path, dpi=144):
    f.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(f)


def pixels(ax, mask, colour=PIX, grid=True, colours=None, lw=0.8):
    """Draw a binary mask (or a label image with `colours`) as pixel squares, y down."""
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            v = mask[y, x]
            fc = (colours[(int(v) - 1) % len(colours)] if colours is not None else colour) if v else "white"
            ax.add_patch(Rectangle((x, y), 1, 1, fc=fc, ec=GRID if grid else fc, lw=lw if grid else 0))
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_aspect("equal"); ax.axis("off")


def run_ts(here, name="data.ts", args=()):
    """Bundle a TypeScript data script with esbuild (node platform), run it, return its JSON."""
    tmp = tempfile.mkdtemp()
    js, out = os.path.join(tmp, "data.js"), os.path.join(tmp, "out.json")
    subprocess.run(["node_modules/.bin/esbuild", os.path.join(here, name), "--bundle", "--platform=node",
                    "--log-level=warning", "--outfile=" + js], check=True)
    subprocess.run(["node", js, out, *args], check=True)
    return json.load(open(out))
