# Figures for viola-jones/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/viola-jones/figs.py
# Scenes, training patches, the first stump and the detections come from the boostdetect
# demo code (data.ts, via node), so the numbers agree with the demo.
import os, sys, json, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "integral-image"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from style import setup, fig, save, BLUE, ORANGE, GREEN, RED, GREY

setup()
OUT = "viola-jones"
tmp = tempfile.mkdtemp()
subprocess.run(["node_modules/.bin/esbuild", os.path.join(HERE, "data.ts"), "--bundle", "--platform=node",
                "--log-level=warning", "--outfile=" + os.path.join(tmp, "data.js")], check=True)
subprocess.run(["node", os.path.join(tmp, "data.js"), os.path.join(tmp, "out.json")], check=True)
D = json.load(open(os.path.join(tmp, "out.json")))
W = 24


def save_png(f, path):
    f.savefig(path, dpi=144, bbox_inches="tight", pad_inches=0.04)
    plt.close(f)


# --- 1. sliding windows over the test scene ----------------------------------------------
P = D["plain"]
img = np.array(P["img"]).reshape(256, 256)
fg, ax = plt.subplots(figsize=(3.6, 3.6))
ax.imshow(img, cmap="gray", vmin=0, vmax=255)
for k in range(6):
    ax.add_patch(Rectangle((8 + 6 * k - .5, 200 - .5), 24, 24, fc="none", ec=ORANGE, lw=1.2, alpha=0.9))
ax.add_patch(Rectangle((8 - .5, 200 - .5), 24, 24, fc="none", ec=ORANGE, lw=2.5))
ax.annotate("", xy=(70, 196), xytext=(14, 196), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2))
ax.add_patch(Rectangle((176 - .5, 96 - .5), 48, 48, fc="none", ec=BLUE, lw=2.5))
ax.text(200, 92, "48 px", color=BLUE, ha="center", va="bottom", fontsize=12, weight="bold",
        bbox=dict(fc="white", ec="none", pad=1, alpha=0.8))
ax.text(20, 232, "24 px", color=ORANGE, ha="center", va="top", fontsize=12, weight="bold",
        bbox=dict(fc="white", ec="none", pad=1, alpha=0.8))
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel(f"{P['windows']:,} windows at 4 sizes".replace(",", " "), fontsize=13)
save_png(fg, f"{OUT}/fig-windows.png")

# --- 2. training examples ------------------------------------------------------------------
fg, axs = plt.subplots(2, 6, figsize=(5.4, 2.2))
for j in range(6):
    for i, key in enumerate(["posPatches", "negPatches"]):
        a = axs[i, j]
        a.imshow(np.array(D[key][j]).reshape(W, W), cmap="gray", vmin=0, vmax=255)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color(ORANGE if i == 0 else BLUE); sp.set_linewidth(2.5)
axs[0, 0].set_ylabel("face", fontsize=13)
axs[1, 0].set_ylabel("backgr.", fontsize=13)
fg.subplots_adjust(wspace=0.08, hspace=0.08)
save_png(fg, f"{OUT}/fig-examples.png")

# --- 3. the first stump: feature on a face, histogram, threshold -----------------------
s = D["stump1"]
vals = np.array(s["values"]); y = np.array(D["y"])
spec = s["spec"]; assert spec["t"] == 5  # centre - surround
thr = s["threshold"]
err = np.mean(np.where(vals > thr, 1, -1) != y)
assert abs(err - s["error"]) < 1e-9
mean_face = np.mean(np.array(D["posPatches"]), axis=0).reshape(W, W)
fg, (a0, a1) = fig(470, 290, ncols=2, gridspec_kw=dict(width_ratios=[1, 2.2], wspace=0.45))
face = np.array(D["posPatches"][0]).reshape(W, W)
a0.imshow(face, cmap="gray", vmin=0, vmax=255)
x0, y0, sx, sy = spec["x"], spec["y"], spec["sx"], spec["sy"]
a0.add_patch(Rectangle((x0 - .5, y0 - .5), 3 * sx, 3 * sy, fc=BLUE, alpha=0.35, ec=BLUE, lw=2))
a0.add_patch(Rectangle((x0 + sx - .5, y0 + sy - .5), sx, sy, fc=ORANGE, alpha=0.8, ec=ORANGE, lw=2))
a0.set_xticks([]); a0.set_yticks([])
a0.set_title("the feature", fontsize=17)
bins = np.linspace(min(vals.min(), -1.5), max(vals.max(), 2.5), 36)
a1.hist(vals[y < 0], bins=bins, color=BLUE, alpha=0.75, label="background")
a1.hist(vals[y > 0], bins=bins, color=ORANGE, alpha=0.75, label="face")
a1.set_xlim(-2.6, 2.8)
a1.axvline(thr, color="#333", lw=2.5, ls="--")
a1.text(1.55, a1.get_ylim()[1] * 0.75, f"threshold\n{thr:.2f}", va="top", fontsize=15)
a1.set_xlabel("feature value", fontsize=16)
a1.set_ylabel("examples", fontsize=16)
a1.legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.04, 1.0), fontsize=14, handlelength=1)
save(fg, f"{OUT}/fig-stump.svg")
print("stump1 error", err, "threshold", thr)

# --- 4. AdaBoost reweighting on a toy 2D set (own implementation, synthetic) ----------
rng = np.random.default_rng(3)
n = 40
X = np.r_[rng.normal([0.35, 0.6], 0.15, (n // 2, 2)), rng.normal([0.65, 0.4], 0.15, (n // 2, 2))]
Y = np.r_[np.ones(n // 2), -np.ones(n // 2)]
w = np.full(n, 1 / n)


def best_stump(X, Y, w):
    best = None
    for d in (0, 1):
        for t in np.unique(X[:, d]):
            for pol in (1, -1):
                h = np.where(pol * (X[:, d] - t) > 0, 1, -1)
                e = w[h != Y].sum()
                if best is None or e < best[0]:
                    best = (e, d, t, pol)
    return best


fg, axs = fig(450, 220, ncols=3)
for r in range(3):
    e, d, t, pol = best_stump(X, Y, w)
    h = np.where(pol * (X[:, d] - t) > 0, 1, -1)
    a = axs[r]
    sz = 1600 * w
    a.scatter(X[Y > 0, 0], X[Y > 0, 1], s=sz[Y > 0], c=ORANGE, edgecolors="#333", lw=0.6)
    a.scatter(X[Y < 0, 0], X[Y < 0, 1], s=sz[Y < 0], c=BLUE, edgecolors="#333", lw=0.6, marker="s")
    if d == 0:
        a.axvline(t, color="#333", lw=2.5, ls="--")
    else:
        a.axhline(t, color="#333", lw=2.5, ls="--")
    a.set_title(f"round {r + 1}\nerror {e:.2f}", fontsize=17)
    a.set_xticks([]); a.set_yticks([]); a.set_xlim(X[:, 0].min() - 0.08, X[:, 0].max() + 0.08); a.set_ylim(X[:, 1].min() - 0.08, X[:, 1].max() + 0.08)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(True)
    alpha = 0.5 * np.log((1 - e) / e)
    w = w * np.exp(-alpha * Y * h); w /= w.sum()
save(fg, f"{OUT}/fig-reweight.svg")

# --- 5. detection on the scene with a rotated face -------------------------------------
R = D["rotated"]
img = np.array(R["img"]).reshape(256, 256)
fg, ax = plt.subplots(figsize=(3.6, 3.6))
ax.imshow(img, cmap="gray", vmin=0, vmax=255)
for t in R["targets"]:
    ax.add_patch(Rectangle((t["x"], t["y"]), t["size"], t["size"], fc="none", ec="white", lw=1.5, ls="--"))
for m in R["merged"]:
    ax.add_patch(Rectangle((m["x"], m["y"]), m["size"], m["size"], fc="none", ec=GREEN if m["ok"] else RED, lw=2.5))
rt = [t for t in R["targets"] if t["kind"] == "rotated"][0]
assert not R["found"][R["targets"].index(rt)]
ax.text(rt["x"] + rt["size"] / 2, rt["y"] + rt["size"] + 4, "rotated: missed", color=RED, ha="center", va="top",
        fontsize=12, weight="bold", bbox=dict(fc="white", ec="none", pad=1, alpha=0.85))
ax.set_xticks([]); ax.set_yticks([])
nf = sum(not m["ok"] for m in R["merged"])
ax.set_xlabel(f"{sum(R['found'])} of {len(R['found'])} faces found\n{nf} false detections", fontsize=12)
save_png(fg, f"{OUT}/fig-rotated.png")
print("rotated: found", sum(R["found"]), "false", nf)
