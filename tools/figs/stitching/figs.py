"""Static figures for stitching/ (from a peak to a mosaic).

Run from the repository root:  python3 -I tools/figs/stitching/figs.py
Writes stitching/fig-*.svg|png. All data come from the stitch, subpixel and drift demos' own
models (tools/figs/stitching/data.ts, run with node), so the numbers match the demos.
"""
import json, os, subprocess, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image

plt.rcParams.update({"font.size": 16, "axes.titlesize": 16, "svg.fonttype": "none",
                     "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, RED, GREEN, GREY, PURPLE = "#1f77b4", "#ff7f0e", "#d62728", "#2a9d4b", "#888888", "#cc79a7"
CELL = np.asarray(Image.open("demos/src/fourier2d/cell256.png")).astype(float)
OUT = "stitching/"


def demo_data():
    with tempfile.TemporaryDirectory() as d:
        cell, js = os.path.join(d, "cell.f64"), os.path.join(d, "st.json")
        CELL.astype("<f8").tofile(cell)
        subprocess.run("node_modules/.bin/esbuild tools/figs/stitching/data.ts --bundle --platform=node "
                       "--log-level=warning | node - %s %s" % (cell, js), shell=True, check=True)
        return json.load(open(js))


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ------------------------------------------------------------ four candidates, scored by NCC
def candidates(D):
    s = D["stitch"]; T = s["T"]; e = s["edge1"]
    A = np.array(s["tiles"][e["i"]]).reshape(T, T); B = np.array(s["tiles"][e["j"]]).reshape(T, T)
    order = sorted(e["candidates"], key=lambda c: (c["ty"] < 0, c["tx"] < 0))
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 6.0), gridspec_kw={"hspace": 0.42, "wspace": 0.1})
    for ax, c in zip(axs.ravel(), order):
        tx, ty = c["tx"], c["ty"]
        x0, x1 = min(0, tx), max(T, tx + T); y0, y1 = min(0, ty), max(T, ty + T)
        ax.imshow(A, cmap="gray", vmin=0, vmax=255, extent=[-0.5, T - 0.5, T - 0.5, -0.5])
        ax.imshow(B, cmap="gray", vmin=0, vmax=255, alpha=0.55, extent=[tx - 0.5, tx + T - 0.5, ty + T - 0.5, ty - 0.5])
        ax.add_patch(Rectangle((-0.5, -0.5), T, T, fill=False, ec=BLUE, lw=2.5))
        ax.add_patch(Rectangle((tx - 0.5, ty - 0.5), T, T, fill=False, ec=ORANGE, lw=2.5, ls="--"))
        ax.set_xlim(x0 - 4, x1 + 4); ax.set_ylim(y1 + 4, y0 - 4); ax.set_aspect("equal")
        best = c is e["candidates"][0]
        ncc = "too small" if c["ncc"] is None else "NCC %.2f" % c["ncc"]
        ax.set_title("t = (%d, %d)\n%d px, %s" % (tx, ty, c["overlap"], ncc), fontsize=15,
                     color=GREEN if best else "k", weight="bold" if best else "normal")
        off(ax)
    fig.savefig(OUT + "fig-candidates.png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------ sub-pixel: parabola vs the true peak
def parabola(D):
    p = D["subpixel"]; c = np.array(p["c"]); N = len(c); t0 = p["t0"]
    ts = np.arange(t0 - 3, t0 + 4); cs = c[ts % N]
    cm, c0, cp = c[(t0 - 1) % N], c[t0 % N], c[(t0 + 1) % N]
    a = 0.5 * (cm + cp) - c0; b = 0.5 * (cp - cm)
    xx = np.linspace(t0 - 1.6, t0 + 1.6, 200)
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    ax.plot(p["curve"]["xs"], p["curve"]["cs"], color=BLUE, lw=2.4, label="true peak shape")
    ax.plot(xx, a * (xx - t0) ** 2 + b * (xx - t0) + c0, color=RED, lw=2.2, ls="--", label="parabola, 3 samples")
    ax.plot(ts, cs, "o", color="k", ms=9, label="samples", zorder=5)
    ax.axvline(p["shift"], color=GREEN, lw=2.2)
    ax.axvline(p["parabola"], color=RED, lw=1.8, ls=":")
    ax.text(p["shift"] + 0.06, 1.02, "truth\n%.2f" % p["shift"], color=GREEN, fontsize=15, va="top")
    ax.text(p["parabola"] - 0.06, 1.02, "parabola\n%.3f" % p["parabola"], color=RED, fontsize=15, va="top", ha="right")
    ax.set_xlabel("shift t (pixels)"); ax.set_ylabel("phase correlation")
    ax.set_ylim(-0.25, 1.08); ax.set_xlim(t0 - 2.5, t0 + 2.5)
    ax.legend(frameon=False, fontsize=14, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, columnspacing=1.0)
    fig.savefig(OUT + "fig-parabola.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------ mosaic graph: chain vs global
def graph(D):
    g = D["noise12"]; E = g["edges"]; snake = set(g["snake"])
    pos = {k: (k % 3, k // 3) for k in range(9)}
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.7), gridspec_kw={"wspace": 0.12})
    vmax = max(max(g["errC"]), max(g["errG"]))
    cmap = plt.get_cmap("Oranges")
    for ax, err, title, use in ((axs[0], g["errC"], "chained along a path", snake),
                                (axs[1], g["errG"], "global least squares", set(range(len(E))))):
        for k, (i, j) in enumerate(E):
            (x1, y1), (x2, y2) = pos[i], pos[j]
            on = k in use
            ax.plot([x1, x2], [y1, y2], color=GREEN if on else "#cccccc", lw=5 if on else 2.5, zorder=1,
                    ls="-" if on else ":")
        for k in range(9):
            x, y = pos[k]
            ax.add_patch(Circle((x, y), 0.3, fc=cmap(0.15 + 0.75 * err[k] / vmax), ec="#333", lw=1.8, zorder=2))
            ax.text(x, y, "%.1f" % err[k], ha="center", va="center", fontsize=16, zorder=3)
        ax.set_xlim(-0.45, 2.45); ax.set_ylim(2.45, -0.45); ax.set_aspect("equal"); off(ax)
        ax.set_title(title + "\nmax error %.2f px" % max(err), fontsize=16)
    fig.savefig(OUT + "fig-graph.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------ seams vs feathering
def blend(D):
    s = D["stitch"]; T, BIG = s["T"], s["BIG"]
    tiles = [np.array(t).reshape(T, T) for t in s["tiles"]]
    P = s["globalPos"]
    u = np.arange(T); f = np.minimum(1, (np.minimum(u, T - 1 - u) + 1) / 24)
    W = np.outer(f, f)

    def mosaic(feather):
        acc = np.zeros((BIG, BIG)); ws = np.zeros((BIG, BIG))
        for k, t in enumerate(tiles):
            ox, oy = int(round(P[k][0])), int(round(P[k][1]))
            y0, x0 = max(0, oy), max(0, ox); y1, x1 = min(BIG, oy + T), min(BIG, ox + T)
            sub = t[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
            if feather:
                w = W[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
                acc[y0:y1, x0:x1] += w * sub; ws[y0:y1, x0:x1] += w
            else:
                acc[y0:y1, x0:x1] = sub; ws[y0:y1, x0:x1] = 1
        m = np.where(ws > 0, acc / np.maximum(ws, 1e-12), np.nan)
        return m
    cm = plt.get_cmap("gray").copy(); cm.set_bad("#ebebeb")
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 4.5), gridspec_kw={"wspace": 0.06})
    for a, feather, title in ((ax[0], False, "pasted: seams"), (ax[1], True, "feathered")):
        a.imshow(np.ma.masked_invalid(mosaic(feather)), cmap=cm, vmin=0, vmax=255)
        a.set_title(title); off(a)
        a.set_xlim(70, 290); a.set_ylim(290, 70)
    fig.savefig(OUT + "fig-blend.png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------ drift: to first vs to previous
def drift(D):
    d = D["drift"]; k = np.arange(len(d["first"]))
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(k, d["previous"], "o-", color=ORANGE, lw=2.2, ms=5, label="to the previous frame")
    ax.plot(k, d["first"], "s-", color=BLUE, lw=2.2, ms=5, label="to the first frame")
    ax.set_xlabel("frame"); ax.set_ylabel("error of the drift (px)")
    ax.legend(frameon=False, fontsize=14, loc="upper left")
    ax.set_xlim(0, k[-1] + 0.5); ax.set_ylim(0, 5.6)
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT + "fig-drift-error.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    D = demo_data()
    candidates(D); parabola(D); graph(D); blend(D); drift(D)
