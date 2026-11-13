# Shared matplotlib style for the integral-image day figures.
# Figures are sized in points = display pixels (figsize = px / 72), so a font size of
# 18 shows as 18 px when the SVG is displayed at its nominal width.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#888888"

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
