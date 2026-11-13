# Figures for relations/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/relations/figs.py
# Populations, neighbour lists, border rules and parents come from the neighbours demo's
# code (data.ts, via node), so every number matches the demo and the slides.
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "label-connectivity"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle, Circle, Wedge, Ellipse, Arc
from labstyle import setup, fig, save, save_png, run_ts, INK, GREY

setup()
OUT = "relations"
D = run_ts(HERE)
W, H, GUARD = D["W"], D["H"], D["GUARD"]
BLUE, ORANGE, VERM, GREEN, SKY, PINK, YELLOW = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#56B4E9", "#CC79A7", "#F0E442"
TINTS = ["#cfe3f3", "#fde0b3", "#cdeadf", "#f3d3e6", "#e6e1a8", "#d9d2ec"]
NUC = "#3a4a6b"


def img(a):
    return np.array(a).reshape(H, W)


def colouring(nb, count):
    """Greedy graph colouring so that neighbouring regions get different tints."""
    col = [0] * (count + 1)
    for l in range(1, count + 1):
        used = {col[m] for m in nb[l] if m < l}
        col[l] = next(c for c in range(len(TINTS)) if c not in used)
    return col


def paint(lab, col, nuclei=None, base="white"):
    rgb = np.ones((H, W, 3)) * np.array(to_rgb(base))
    for l in range(1, len(col)):
        rgb[lab == l] = to_rgb(TINTS[col[l]])
    if nuclei is not None:
        rgb[nuclei > 0] = to_rgb(NUC)
    return rgb


def boundaries(lab):
    b = np.zeros(lab.shape, bool)
    b[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    b[1:, :] |= lab[1:, :] != lab[:-1, :]
    return b


def frame(ax, crop=None):
    x0, x1, y0, y1 = crop or (0, W, 0, H)
    ax.set_xlim(x0 - 0.5, x1 - 0.5); ax.set_ylim(y1 - 0.5, y0 - 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#999999")


def edges(ax, p, nb, col=INK, lw=1.8):
    xs, ys = p["xs"], p["ys"]
    for l, ns in enumerate(nb):
        for m in ns:
            if m > l:
                ax.plot([xs[l - 1], xs[m - 1]], [ys[l - 1], ys[m - 1]], color=col, lw=lw, solid_capstyle="round")


P = D["pops"]["poisson"]
nuc = img(P["nuclei"]); cells = img(P["cells"]); vor = img(P["voronoi"])
count = P["count"]
colV = colouring(P["expanded"]["nb"], count)

# --- 1. three definitions of neighbour ------------------------------------------------------
f, axs = plt.subplots(1, 3, figsize=(900 / 72, 330 / 72))
cellcol = colouring(P["adjacent"]["nb"], count)
panels = [
    (axs[0], paint(cells, cellcol, nuc), P["adjacent"], "adjacent"),
    (axs[1], None, P["within20"], "within r = 20 px"),
    (axs[2], paint(vor, colV, nuc), P["expanded"], "expanded"),
]
for ax, rgb, a, lab in panels:
    if rgb is None:
        rgb = np.ones((H, W, 3)); rgb[nuc > 0] = to_rgb(NUC)
        for l in range(1, count + 1):
            ax.add_patch(Circle((P["xs"][l - 1], P["ys"][l - 1]), 10, fc="none", ec=SKY, lw=1, ls="--"))
    else:
        rgb[boundaries(cells if a is P["adjacent"] else vor) & (nuc == 0)] = to_rgb("#888888")
    ax.imshow(rgb, interpolation="nearest")
    edges(ax, P, a["nb"], col=VERM, lw=2.2)
    frame(ax)
    ax.set_title(f"{lab}: {a['meanCount']:.2f}", fontsize=21)
f.tight_layout(w_pad=1.0)
save_png(f, f"{OUT}/fig-defs.png")

# --- 2. expansion = nearest object, distance map + Voronoi boundaries -----------------------
dist = img(P["dist"])
f, axs = plt.subplots(1, 2, figsize=(500 / 72, 290 / 72))
ax = axs[0]
im = ax.imshow(dist, cmap="viridis", interpolation="nearest")
b = boundaries(vor)
ov = np.zeros((H, W, 4)); ov[b] = (1, 1, 1, 1)
ax.imshow(ov, interpolation="nearest")
frame(ax); ax.set_title("distance to\nnearest nucleus", fontsize=17)
ax = axs[1]
rgb = paint(vor, colV, nuc); rgb[b] = to_rgb("#777777")
ax.imshow(rgb, interpolation="nearest")
edges(ax, P, P["expanded"]["nb"], col=VERM, lw=2)
frame(ax); ax.set_title("regions and\nneighbour graph", fontsize=17)
f.tight_layout(w_pad=1.5)
save_png(f, f"{OUT}/fig-expand.png")

# --- 3. per-object columns: closest, second closest, angle, percent touching ----------------
E = P["expanded"]
# choose an interior object with a clear angle
cands = [l for l in range(1, count + 1) if 60 < P["xs"][l - 1] < 196 and 60 < P["ys"][l - 1] < 196]
k = max(cands, key=lambda l: (P["cellPercent"][l] > 5 and P["cellPercent"][l] < 80, -abs(E["angle"][l - 1] - 90)))
x, y = P["xs"][k - 1], P["ys"][k - 1]
i1, i2 = E["first"][k - 1], E["second"][k - 1]
d1, d2, ang = E["firstDist"][k - 1], E["secondDist"][k - 1], E["angle"][k - 1]
f, ax = fig(470, 470)
rgb = paint(cells, cellcol, nuc)
bc = boundaries(cells)
rgb[bc & (nuc == 0) & (cells > 0)] = to_rgb("#888888")
# boundary pixels of object k that touch another cell
touch = np.zeros((H, W), bool)
ck = cells == k
for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
        sh = np.roll(np.roll(cells, dy, 0), dx, 1)
        touch |= ck & (sh > 0) & (sh != k)
rgb[touch] = to_rgb(PINK)
ax.imshow(rgb, interpolation="nearest")
for j, d, col, lab in [(i1, d1, BLUE, "1st"), (i2, d2, GREEN, "2nd")]:
    xj, yj = P["xs"][j], P["ys"][j]
    ax.plot([x, xj], [y, yj], color=col, lw=3.5)
    ax.plot(xj, yj, "o", color=col, ms=8, mec="white", mew=1.5)
    ux, uy = (xj - x) / d, (yj - y) / d
    ax.text(xj + 9 * ux, yj + 9 * uy, f"{lab}\n{d:.1f} px", color=col, fontsize=18, ha="center", va="center", fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5))
ax.plot(x, y, "o", color=INK, ms=8, mec="white", mew=1.5)
a1 = np.degrees(np.arctan2(P["ys"][i1] - y, P["xs"][i1] - x)); a2 = np.degrees(np.arctan2(P["ys"][i2] - y, P["xs"][i2] - x))
lo, hi = sorted([a1, a2])
if hi - lo > 180: lo, hi = hi, lo + 360
ax.add_patch(Arc((x, y), 14, 14, theta1=lo, theta2=hi, color=INK, lw=2.5))
R = 42
crop = (int(x - R), int(x + R), int(y - R), int(y + R))
frame(ax, crop)
ax.text(crop[0] + 2, crop[3] - 2, f"angle {ang:.0f}°, touching {P['cellPercent'][k]:.0f} % (pink)", fontsize=17, ha="left", va="bottom",
        bbox=dict(fc="white", ec="none", alpha=0.9, pad=2))
save_png(f, f"{OUT}/fig-perobject.png")
print("per-object", k, d1, d2, ang, P["cellPercent"][k])

# --- 4. clustered, random, regular -----------------------------------------------------------
f, axs = plt.subplots(1, 3, figsize=(780 / 72, 300 / 72))
for ax, key, lab in zip(axs, ["clustered", "poisson", "regular"], ["clustered", "random placement", "regular"]):
    Q = D["pops"][key]
    rgb = np.ones((H, W, 3)); rgb[img(Q["nuclei"]) > 0] = to_rgb(NUC)
    ax.imshow(rgb, interpolation="nearest")
    a = Q["expanded"]
    for l in range(1, Q["count"] + 1):
        j = a["first"][l - 1]
        ax.annotate("", xy=(Q["xs"][j], Q["ys"][j]), xytext=(Q["xs"][l - 1], Q["ys"][l - 1]),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6, mutation_scale=10, shrinkA=4, shrinkB=4))
    frame(ax)
    ax.set_title(f"{lab}: R = {a['R']:.2f}", fontsize=19)
f.tight_layout(w_pad=1.0)
save_png(f, f"{OUT}/fig-patterns.png")

# --- 5. border objects and the guard zone ----------------------------------------------------
B = D["pops"]["border"]
bv = img(B["voronoi"]); bn = img(B["nuclei"])
colB = colouring(B["expanded"]["nb"], B["count"])
f, ax = fig(440, 440)
rgb = np.ones((H, W, 3))
tb = B["expanded"]["touchesBorder"]; meas = B["guard"]["measured"]
for l in range(1, B["count"] + 1):
    rgb[bv == l] = to_rgb("#fde0b3" if tb[l] else "#cfe3f3")
rgb[boundaries(bv)] = to_rgb("#888888")
rgb[bn > 0] = to_rgb(NUC)
ax.imshow(rgb, interpolation="nearest")
ax.add_patch(Rectangle((GUARD - 0.5, GUARD - 0.5), W - 2 * GUARD, H - 2 * GUARD, fill=False, ec=VERM, lw=3, ls="--"))
for l in range(1, B["count"] + 1):
    if meas[l]:
        ax.plot(B["xs"][l - 1], B["ys"][l - 1], "o", ms=5, color=YELLOW, mec=INK, mew=0.8)
frame(ax)
ax.text(W / 2, H + 4, f"orange: at the border ({B['expanded']['atBorder']:.2f})\nblue: interior ({B['expanded']['interior']:.2f})\ndashed: guard frame, {GUARD} px\ndots: {B['guard']['nMeasured']} objects measured",
        ha="center", va="top", fontsize=18)
save_png(f, f"{OUT}/fig-border.png")

# --- 6. spots and their parents ----------------------------------------------------------------
sp = img(P["spots"]); pn = P["parN"]; pc = P["parC"]
f, ax = fig(440, 440)
rgb = paint(cells, cellcol, None)
rgb[boundaries(cells) & (cells > 0)] = to_rgb("#999999")
rgb[nuc > 0] = to_rgb("#9aa5bd")
nS = P["nSpots"]
for s in range(1, nS + 1):
    m = sp == s
    if pc["nParents"][s] > 1: c = PINK
    elif pn["parent"][s]: c = BLUE
    else: c = VERM
    rgb[m] = to_rgb(c)
ax.imshow(rgb, interpolation="nearest")
# circle the spots without a nucleus parent
for s in range(1, nS + 1):
    if not pn["parent"][s]:
        yy, xx = np.nonzero(sp == s)
        ax.add_patch(Circle((xx.mean(), yy.mean()), 4.5, fc="none", ec=VERM if pc["nParents"][s] <= 1 else PINK, lw=2.2))
frame(ax, (88, 208, 82, 202))
ax.text(90, 203, "blue: in a nucleus; orange: no nucleus;\npink: across two cells (circled: no nucleus)", fontsize=16, ha="left", va="top")
save_png(f, f"{OUT}/fig-parents.png")
print("spots without nucleus", sum(1 for s in range(1, nS + 1) if not pn["parent"][s]))

# --- 7. overlap vs centroid: four children --------------------------------------------------------
f, axs = plt.subplots(1, 4, figsize=(1040 / 72, 250 / 72))
cases = ["inside", "mostly in 1", "crescent", "background"]
titles = ["inside one parent", "across two parents", "crescent in a cytoplasm", "in the background"]
res = [("1", "1"), ("1 (flag)", "1"), ("1", "none"), ("none", "none")]
for ax, case, ttl, (ro, rc) in zip(axs, cases, titles, res):
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-2.5, 2.3); ax.set_aspect("equal"); ax.axis("off")
    if case == "mostly in 1":
        ax.add_patch(Circle((-1.0, 0), 1.6, fc="#cfe3f3", ec=BLUE, lw=2))
        ax.add_patch(Circle((2.25, 0), 1.6, fc="#fde0b3", ec=ORANGE, lw=2))
        ax.text(-1.6, 0.0, "1", fontsize=17, color=BLUE, va="center"); ax.text(2.45, 0.95, "2", fontsize=17, color=ORANGE)
        ch = Ellipse((0.2, -0.1), 1.4, 0.7, fc=VERM, ec="none", alpha=0.85); ax.add_patch(ch); cxy = (0.2, -0.1)
    else:
        ax.add_patch(Circle((0, 0), 1.6, fc="#cfe3f3", ec=BLUE, lw=2)); ax.text(-1.25, 0.0, "1", fontsize=17, color=BLUE, va="center")
        if case == "inside":
            ax.add_patch(Circle((0.2, -0.2), 0.45, fc=VERM, ec="none")); cxy = (0.2, -0.2)
        elif case == "crescent":
            # parent 1 is a cytoplasm (cell minus nucleus); the crescent hugs the nucleus
            ax.add_patch(Circle((0, 0), 0.9, fc="white", ec=GREY, lw=1.5, ls="--"))
            ax.add_patch(Wedge((0, 0), 1.4, -120, 120, width=0.4, fc=VERM, ec="none"))
            r1, r2, al = 1.0, 1.4, np.radians(120)
            cxy = (2 / 3 * (r2 ** 3 - r1 ** 3) / (r2 ** 2 - r1 ** 2) * np.sin(al) / al, 0)
        else:
            ax.add_patch(Circle((2.4, 1.4), 0.45, fc=VERM, ec="none")); cxy = (2.4, 1.4)
    ax.plot(*cxy, "x", color=INK, ms=11, mew=2.5)
    ax.set_title(ttl, fontsize=16)
    ax.text(0, -2.0, f"overlap: {ro}\ncentroid: {rc}", ha="center", va="top", fontsize=16)
f.tight_layout(w_pad=0.5)
save(f, f"{OUT}/fig-overlap.svg")
print("done")
