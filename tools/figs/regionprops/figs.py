# Figures for regionprops/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/regionprops/figs.py
# Masks and properties come from the moments demo's code (data.ts, via node).
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "label-connectivity"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.colors import ListedColormap
from labstyle import setup, fig, save, pixels, run_ts, PIX, PIXLIGHT, GRID, INK, BLUE, ORANGE, GREEN, RED, GREY, LABELS

setup()
OUT = "regionprops"
D = run_ts(HERE)
S = 64


def ellipse_of(ax, p, col=ORANGE, axes=True, lw=2.5):
    """Equivalent ellipse (semi-axes 2 sqrt(lambda)) and the eigenvectors, y down."""
    a, b = 2 * np.sqrt(p["l1"]), 2 * np.sqrt(p["l2"])
    ang = np.degrees(p["theta"])  # from +x towards +y (down): with ylim inverted this is the drawn angle
    ax.add_patch(Ellipse((p["cx"], p["cy"]), 2 * a, 2 * b, angle=ang, fill=False, ec=col, lw=lw))
    ax.plot(p["cx"], p["cy"], "+", color=INK, ms=14, mew=2.5)
    if axes:
        v1 = np.array(p["v1"]); v2 = np.array([-v1[1], v1[0]])
        for v, l, c in [(v1, a, RED), (v2, b, GREEN)]:
            ax.annotate("", xy=(p["cx"] + l * v[0], p["cy"] + l * v[1]), xytext=(p["cx"], p["cy"]),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=2.5, mutation_scale=18))


def show_mask(ax, m, colour=PIX):
    m = np.array(m).reshape(S, S)
    ax.imshow(m, cmap=ListedColormap(["white", colour]), extent=(-0.5, S - 0.5, S - 0.5, -0.5), interpolation="nearest")
    ax.set_xlim(-0.5, S - 0.5); ax.set_ylim(S - 0.5, -0.5); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(GRID)


# --- 1. one scan: a label image, centroids, boxes ---------------------------------------
lab = np.array(D["two"]["labels"]).reshape(S, S)
f, ax = fig(440, 440)
cm = ListedColormap(["white", LABELS[4], LABELS[0]])
ax.imshow(lab, cmap=cm, vmin=0, vmax=2, extent=(-0.5, S - 0.5, S - 0.5, -0.5), interpolation="nearest")
for r in D["two"]["rows"]:
    ax.add_patch(Rectangle((r["x0"] - 0.5, r["y0"] - 0.5), r["x1"] - r["x0"] + 1, r["y1"] - r["y0"] + 1, fill=False, ec=INK, lw=1.8, ls="--"))
    ax.plot(r["cx"], r["cy"], "+", color=INK if r["label"] == 2 else INK, ms=16, mew=3)
    ax.text(r["x0"] - 0.5, r["y0"] - 1.5, f"label {r['label']}", fontsize=19, va="bottom")
ax.set_xlim(-0.5, S - 0.5); ax.set_ylim(S - 0.5, -0.5); ax.set_aspect("equal")
ax.set_xlabel("x (column)"); ax.set_ylabel("y (row)")
ax.set_xticks([0, 20, 40, 60]); ax.set_yticks([0, 20, 40, 60])
ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
save(f, f"{OUT}/fig-onescan.svg")

# --- 2. an object as a cloud of points ---------------------------------------------------
E = D["ellipse"]; p = E["props"]
m = np.array(E["mask"]).reshape(S, S)
ys, xs = np.nonzero(m)
f, ax = fig(470, 400)
ax.plot(xs, ys, "o", ms=3.2, color=BLUE, alpha=0.75)
ellipse_of(ax, p, axes=True)
v1 = np.array(p["v1"])
ax.text(p["cx"] + 2 * np.sqrt(p["l1"]) * v1[0] + 1, p["cy"] + 2 * np.sqrt(p["l1"]) * v1[1] + 1, r"$\mathbf{v}_1$", color=RED, fontsize=21, va="top")
v2 = np.array([-v1[1], v1[0]])
ax.text(p["cx"] + 2 * np.sqrt(p["l2"]) * v2[0] - 6.5, p["cy"] + 2 * np.sqrt(p["l2"]) * v2[1] - 1.5, r"$\mathbf{v}_2$", color=GREEN, fontsize=21)
ax.set_xlim(6, 64); ax.set_ylim(51, 13); ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("y (down)")
ax.text(41, 50, f"$\\lambda_1$ = {p['l1']:.2f}\n$\\lambda_2$ = {p['l2']:.2f}\norientation {p['orientation']:.1f}°", fontsize=17, va="bottom")
save(f, f"{OUT}/fig-cloud.svg")

# --- 3. |cos| + |sin| and its mean 4/pi -----------------------------------------------------
phi = np.linspace(0, 2 * np.pi, 721)
f, ax = fig(560, 220)
ax.plot(np.degrees(phi), np.abs(np.cos(phi)) + np.abs(np.sin(phi)), color=BLUE, lw=2.5, label=r"$|\cos\varphi| + |\sin\varphi|$")
ax.axhline(4 / np.pi, color=ORANGE, lw=2.5, ls="--", label=r"mean $4/\pi \approx 1.273$")
ax.axhline(1, color=GREY, lw=1.2)
ax.set_xlim(0, 360); ax.set_ylim(0.95, 1.5); ax.set_xticks([0, 45, 90, 180, 270, 360])
ax.set_xlabel(r"direction of the boundary $\varphi$ (degrees)"); ax.set_ylabel("edges / length")
ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=16, bbox_to_anchor=(1.0, 1.08))
save(f, f"{OUT}/fig-staircase.svg")

# --- 4. what the ellipse does and does not tell -----------------------------------------
f, axs = plt.subplots(1, 3, figsize=(900 / 72, 340 / 72), gridspec_kw=dict(wspace=0.25))
for ax, k, txt in [(axs[0], "L", "an L: the ellipse\nis not the shape"),
                   (axs[1], "square", "a square: $\\lambda_1 = \\lambda_2$,\nno orientation"),
                   (axs[2], "ring", "a ring: $\\lambda_1 = \\lambda_2$,\nEuler number 0")]:
    q = D[k]["props"]
    show_mask(ax, D[k]["mask"])
    ellipse_of(ax, q, axes=(k == "L"))
    ax.set_xlabel(txt, fontsize=25, labelpad=8)
save(f, f"{OUT}/fig-summary-ellipse.svg")

# --- 5. Euler number: bit-quads and examples ---------------------------------------------
f = plt.figure(figsize=(500 / 72, 520 / 72))
pats = [("$Q_1$: +1", [[1, 0], [0, 0]]), ("$Q_3$: −1", [[1, 1], [1, 0]]), ("$Q_D$: −2", [[1, 0], [0, 1]])]
for k, (t, pm) in enumerate(pats):
    ax = f.add_axes([0.06 + k * 0.33, 0.79, 0.2, 0.17])
    pixels(ax, np.array(pm), colour=INK, lw=1.5)
    ax.add_patch(Rectangle((0, 0), 2, 2, fill=False, ec=GREY, lw=2))
    ax.set_title(t, fontsize=22, pad=5)
f.text(0.5, 0.71, "$E = (Q_1 - Q_3 - 2Q_D)/4$", ha="center", fontsize=22)
for k, (key, name) in enumerate([("disc", "disc"), ("ring", "ring"), ("eight", "digit 8"), ("two", "two objects")]):
    e = D["euler"][key]
    m = np.array(e["mask"]).reshape(e["h"], e["w"])
    pad = (7 - e["w"]) // 2
    m = np.pad(m, ((0, 7 - e["h"]), (pad, 7 - e["w"] - pad)))
    ax = f.add_axes([0.01 + (k % 2) * 0.5, 0.36 - (k // 2) * 0.36, 0.2, 0.30])
    pixels(ax, m, colour=PIX, lw=0.8)
    q = e["quads"]
    ax.text(7.6, 0.0, f"{name}\n$Q_1$ = {q['q1']}\n$Q_3$ = {q['q3']}\n$Q_D$ = {q['qd']}\n$E$ = {e['e8']:.0f}".replace("-", "−"), fontsize=20, va="top", ha="left", linespacing=1.25)
save(f, f"{OUT}/fig-bitquads.svg")
# check the examples against labelling: objects - holes
for e in D["euler"].values():
    assert e["e8"] == e["objects"] - e["holes"], e
print({r["label"]: (r["area"], round(r["cx"], 2), round(r["cy"], 2), r["x0"], r["x1"], r["y0"], r["y1"], round(r["extent"], 2)) for r in D["two"]["rows"]})
