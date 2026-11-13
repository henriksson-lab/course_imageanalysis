# Figures for rect-features/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/rect-features/figs.py
# Box-feature maps and pool sizes come from the course's demo code (data.ts, via node).
import os, sys, json, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "integral-image"))
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from style import setup, fig, save, BLUE, ORANGE, GREEN, RED, GREY

setup()
OUT = "rect-features"
N = 256
cell = io.imread("demos/src/fourier2d/cell256.png").astype(float)
if cell.ndim == 3:
    cell = cell[..., 0]

tmp = tempfile.mkdtemp()
json.dump(cell.ravel().tolist(), open(os.path.join(tmp, "cell.json"), "w"))
subprocess.run(["node_modules/.bin/esbuild", os.path.join(HERE, "data.ts"), "--bundle", "--platform=node",
                "--log-level=warning", "--outfile=" + os.path.join(tmp, "data.js")], check=True)
subprocess.run(["node", os.path.join(tmp, "data.js"), os.path.join(tmp, "cell.json"), os.path.join(tmp, "out.json")], check=True)
D = json.load(open(os.path.join(tmp, "out.json")))


def save_png(f, path):
    f.savefig(path, dpi=144, bbox_inches="tight", pad_inches=0.04)
    plt.close(f)


# --- 1. two rectangles, six lookups -------------------------------------------------
fg, ax = fig(470, 300)
ax.set_aspect("equal"); ax.axis("off")
w, h = 3.0, 3.6
ax.add_patch(Rectangle((0, 0), w, h, fc="white", ec="#333", lw=2.5))
ax.add_patch(Rectangle((w, 0), w, h, fc="#333", ec="#333", lw=2.5))
ax.text(w / 2, h / 2, "white", ha="center", va="center", fontsize=20)
ax.text(1.5 * w, h / 2, "black", ha="center", va="center", fontsize=20, color="white")
coef = {(0, h): +1, (w, h): -2, (2 * w, h): +1, (0, 0): -1, (w, 0): +2, (2 * w, 0): -1}
for (x, y), c in coef.items():
    col = GREEN if c > 0 else RED
    ax.plot(x, y, "o", ms=30, color=col, mec="white", mew=2.5)
    ax.text(x, y, f"{c:+d}".replace("-", "−"), ha="center", va="center", color="white", fontsize=17, weight="bold")
ax.set_xlim(-0.8, 2 * w + 0.8); ax.set_ylim(-0.8, h + 0.8)
save(fg, f"{OUT}/fig-sixlookups.svg")
# check: coefficient pattern = (white rect corners) - (black rect corners), the middle ones shared
white = {(0, h): 1, (w, h): -1, (0, 0): -1, (w, 0): 1}
black = {(w, h): 1, (2 * w, h): -1, (w, 0): -1, (2 * w, 0): 1}
for k in coef:
    assert coef[k] == white.get(k, 0) - black.get(k, 0)

# --- 2. normalising for lighting -------------------------------------------------------
fw, fh = 8, 16  # each rectangle 8 wide, 16 tall; template 16 x 16
S = np.zeros((N + 1, N + 1)); S[1:, 1:] = cell.cumsum(0).cumsum(1)
rs = lambda S, x0, y0, x1, y1: S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0]
best, bx, by = 0, 0, 0
for y in range(24, N - 40):
    for x in range(24, N - 40):
        r = rs(S, x, y, x + fw, y + fh) - rs(S, x + fw, y, x + 2 * fw, y + fh)
        if abs(r) > abs(best):
            best, bx, by = r, x, y
cx0, cy0 = bx + fw - 24, by + fh // 2 - 24
fg, axs = plt.subplots(1, 2, figsize=(5.4, 3.35))
for a, (gain, off, name) in zip(axs, [(1.0, 0.0, "bright"), (0.3, 60.0, "dim")]):
    img = gain * cell + off
    win = img[by:by + fh, bx:bx + 2 * fw]
    raw = win[:, :fw].sum() - win[:, fw:].sum()
    norm = raw / (fw * fh) / win.std()
    a.imshow(img[cy0:cy0 + 48, cx0:cx0 + 48], cmap="gray", vmin=0, vmax=255)
    a.add_patch(Rectangle((bx - cx0 - .5, by - cy0 - .5), fw, fh, fc="none", ec=ORANGE, lw=2.5))
    a.add_patch(Rectangle((bx - cx0 + fw - .5, by - cy0 - .5), fw, fh, fc="none", ec=BLUE, lw=2.5))
    a.set_title(name, fontsize=14)
    a.set_xticks([]); a.set_yticks([])
    a.set_xlabel(f"raw {raw:,.0f}".replace(",", " ").replace("-", "−") + f"\nnormalised {norm:.2f}".replace("-", "−"), fontsize=14)
    print(name, raw, norm)
save_png(fg, f"{OUT}/fig-lighting.png")

# --- 3. the pool size ---------------------------------------------------------------
pool = D["pool"]
names = list(pool.keys())
vals = [pool[k] for k in names]
assert sum(vals) == 170800 and sum(vals[:5]) == 162336 and pool["edge across"] == 43200
fg, ax = fig(340, 360)
y = np.arange(len(names) + 1)
ax.barh(y[1:], vals, color=BLUE, height=0.7)
ax.barh([0], [576], color=ORANGE, height=0.7)
for yy, v in zip(y[1:], vals):
    ax.text(v + 800, yy, f"{v:,}".replace(",", " "), va="center", fontsize=21)
ax.text(576 + 800, 0, "576", va="center", fontsize=21, color="#b35900")
ax.tick_params(labelsize=21)
ax.set_yticks(y)
ax.set_yticklabels(["pixels"] + names)
ax.invert_yaxis()
ax.set_xlim(0, 60000)
ax.set_xticks([0, 40000])
ax.set_xticklabels(["0", "40 000"])
ax.set_xlabel("features in 24 × 24", fontsize=21)
save(fg, f"{OUT}/fig-pool.svg")

# --- 4. Haar wavelets (dyadic tiling) vs Haar-like features (any size, overlapping) ----
fg, axs = fig(620, 300, ncols=2)


def tmpl(ax, x, y, w, h, kind, alpha=1.0, lw=1.5):
    if kind == "H":  # left white, right black
        ax.add_patch(Rectangle((x, y), w / 2, h, fc="white", ec="#333", lw=lw, alpha=alpha))
        ax.add_patch(Rectangle((x + w / 2, y), w / 2, h, fc="#333", ec="#333", lw=lw, alpha=alpha))
    elif kind == "V":
        ax.add_patch(Rectangle((x, y), w, h / 2, fc="white", ec="#333", lw=lw, alpha=alpha))
        ax.add_patch(Rectangle((x, y + h / 2), w, h / 2, fc="#333", ec="#333", lw=lw, alpha=alpha))
    elif kind == "D":
        for (dx, dy, c) in [(0, 0, "white"), (1, 0, "#333"), (0, 1, "#333"), (1, 1, "white")]:
            ax.add_patch(Rectangle((x + dx * w / 2, y + dy * h / 2), w / 2, h / 2, fc=c, ec="#333", lw=lw, alpha=alpha))
    elif kind == "L":  # line: white, black, white
        for i, c in enumerate(["white", "#333", "white"]):
            ax.add_patch(Rectangle((x + i * w / 3, y), w / 3, h, fc=c, ec="#333", lw=lw, alpha=alpha))


a = axs[0]
kinds = ["H", "V", "D", "H", "V", "D", "H", "V", "D", "V", "H", "D", "D", "H", "V", "H"]
for i in range(4):
    for j in range(4):
        tmpl(a, j * 2 + 0.18, i * 2 + 0.18, 1.64, 1.64, kinds[i * 4 + j], lw=1.2)
a.add_patch(Rectangle((0, 0), 8, 8, fc="#d9d9d9", ec=BLUE, lw=2.5, zorder=0))
a.set_title("Haar wavelets", fontsize=27)
b = axs[1]
b.add_patch(Rectangle((0, 0), 8, 8, fc="#f2f2f2", ec=BLUE, lw=2.5))
for (x, y_, w_, h_, k) in [(0.4, 0.5, 5.0, 2.2, "H"), (2.2, 1.6, 2.4, 5.4, "V"), (4.6, 3.2, 3.0, 3.0, "D"),
                           (0.6, 4.6, 4.5, 1.8, "L"), (5.6, 0.4, 2.0, 1.4, "H")]:
    tmpl(b, x, y_, w_, h_, k, alpha=0.75, lw=1.5)
b.set_title("Haar-like", fontsize=27)
for a_ in axs:
    a_.set_aspect("equal"); a_.axis("off"); a_.set_xlim(-0.2, 8.2); a_.set_ylim(8.2, -0.2)
save(fg, f"{OUT}/fig-haar.svg")

# --- 5. box feature stack thumbnails ----------------------------------------------------
fg, axs = plt.subplots(3, 3, figsize=(4.2, 4.6))
rows = [("mean", "box mean"), ("sd", "box SD"), ("dob", "centre −\nsurround")]
for i, (f, lab) in enumerate(rows):
    for j, s in enumerate([1, 4, 16]):
        v = np.array(D[f"{f}{s}"]).reshape(N, N)
        a = axs[i, j]
        if f == "dob":
            m = np.percentile(np.abs(v), 99)
            a.imshow(v, cmap="gray", vmin=-m, vmax=m)
        else:
            a.imshow(v, cmap="gray", vmin=np.percentile(v, 0.5), vmax=np.percentile(v, 99.5))
        a.set_xticks([]); a.set_yticks([])
        if i == 0:
            a.set_title(f"{s} px", fontsize=16)
        if j == 0:
            a.set_ylabel(lab, fontsize=15)
fg.subplots_adjust(wspace=0.04, hspace=0.04)
save_png(fg, f"{OUT}/fig-stack.png")
print("pool", pool)
