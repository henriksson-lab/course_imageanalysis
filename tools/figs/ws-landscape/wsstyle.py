# Shared matplotlib style for the watershed-day figures (ws-landscape/, ws-flooding/,
# ws-gradient/, ws-distance/). Figures are sized in points = display pixels
# (figsize = px / 72), so font size 18 shows as 18 px at the SVG's nominal width.
# Colours: the landscape grey and the basin colours of the existing fig-flood.svg
# (A red, B green, C blue, D orange), Okabe-Ito labels for many regions.
import os, json, subprocess, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAND, WATER, DAM, INK, GRID = "#a39a8c", "#9fc3e6", "#3b2f2f", "#222222", "#bbbbbb"
# basin fills (light) and label inks (dark), as in fig-flood.svg / the flood1d demo
BASIN_FILL = ["#f4a6a6", "#9fd8b4", "#a8b4ee", "#f2c58f", "#d7b4e6", "#b8e0e6"]
BASIN_INK = ["#c0392b", "#1e8449", "#3949ab", "#b9770e", "#7d3c98", "#117a8b"]
LABELS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"]
BLUE, ORANGE, RED = "#1f77b4", "#ff7f0e", "#d62728"


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 18, "axes.labelsize": 18,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
        "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def save(f, path):
    f.savefig(path, bbox_inches="tight", pad_inches=0.05)
    if os.environ.get("WS_PREVIEW"):  # optional PNG copies for checking by eye
        f.savefig(os.path.join(os.environ["WS_PREVIEW"], os.path.basename(path)[:-4] + ".png"),
                  dpi=72, bbox_inches="tight", pad_inches=0.05)
    plt.close(f)


def save_png(f, path, dpi=110):
    f.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(f)


def label_rgb(labels, seed=3, bg=(1, 1, 1)):
    """Colour a label image (0 = background/line) with a shuffled palette of distinct hues."""
    labels = np.asarray(labels)
    n = int(labels.max())
    rng = np.random.default_rng(seed)
    base = [tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)) for c in LABELS[:7]]
    cols = np.array([base[k % 7] for k in range(n + 1)], float)
    if n > 7:
        cols = np.vstack([cols[:8], plt.get_cmap("tab20")(rng.permutation(n + 1 - 8) % 20)[:, :3]])
    cols[0] = bg
    return cols[labels]


def boundary(labels):
    L = np.asarray(labels)
    b = np.zeros(L.shape, bool)
    b[:-1, :] |= L[:-1, :] != L[1:, :]
    b[:, :-1] |= L[:, :-1] != L[:, 1:]
    return b


def run_ts(here, name="data.ts", args=()):
    """Bundle a TypeScript data script with esbuild (node platform), run it, return its JSON."""
    tmp = tempfile.mkdtemp()
    js, out = os.path.join(tmp, "data.js"), os.path.join(tmp, "out.json")
    subprocess.run(["node_modules/.bin/esbuild", os.path.join(here, name), "--bundle", "--platform=node",
                    "--log-level=warning", "--loader:.png=empty", "--outfile=" + js], check=True)
    subprocess.run(["node", js, out, *args], check=True)
    return json.load(open(out))


def flood1d(f, markers, lines=False):
    """Meyer flooding of a 1D signal from labelled markers (FIFO among equal heights).
    Returns labels (0 = line sample when lines=True)."""
    import heapq
    f = list(f); n = len(f)
    lab = [m for m in markers]; queued = [m > 0 for m in markers]; q = []; k = 0
    for p in range(n):
        if lab[p] > 0:
            for r in (p - 1, p + 1):
                if 0 <= r < n and not queued[r]:
                    queued[r] = True; heapq.heappush(q, (f[r], k, r, lab[p])); k += 1
    while q:
        v, _, p, fr = heapq.heappop(q)
        ls = {lab[r] for r in (p - 1, p + 1) if 0 <= r < n and lab[r] > 0}
        L = (-1 if len(ls) > 1 else (ls.pop() if ls else fr)) if lines else fr
        lab[p] = L
        if L != -1:
            for r in (p - 1, p + 1):
                if 0 <= r < n and not queued[r]:
                    queued[r] = True; heapq.heappush(q, (f[r], k, r, L)); k += 1
    return [max(0, l) for l in lab]


def minima1d(f):
    """Label regional minima (plateaus with only higher neighbours) of a 1D signal, 1..k."""
    n = len(f); m = [0] * n; k = 0; i = 0
    while i < n:
        j = i
        while j + 1 < n and f[j + 1] == f[i]: j += 1
        lo = (i == 0 or f[i - 1] > f[i]) and (j == n - 1 or f[j + 1] > f[i])
        if lo:
            k += 1
            for t in range(i, j + 1): m[t] = k
        i = j + 1
    return m
