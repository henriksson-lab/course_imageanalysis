"""Figures for inpainting/lec.qmd. Run from the repository root:
    python3 -I tools/figs/inpainting/figs.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "heat-equation"))
from pslib import *  # noqa: E402,F401,F403
from matplotlib.patches import Rectangle, FancyArrowPatch  # noqa: E402

OUT = "inpainting"
D = run_node("inpainting")


def outline(ax, mask, n=128, color=CYAN):
    m = arr(mask, n)
    ax.contour(m, levels=[0.5], colors=[color], linewidths=1.4)


# ---- fig-defects.png: synthetic defects on the micrograph ----
c = cell128()
dead = c.copy(); dead[:, 70] = 0
sat = c.copy(); yy, xx = np.mgrid[0:128, 0:128]; sat[(xx - 40) ** 2 + (yy - 50) ** 2 < 64] = 255
scr = c.copy()
for t in np.linspace(0, 1, 600):
    x = int(10 + 108 * t); y = int(20 + 90 * t + 8 * np.sin(6 * t)); scr[y:y + 2, x:x + 2] = 235
seam = c.copy(); seam[:, 64:] = np.clip(seam[:, 64:] * 1.12 + 6, 0, 255); seam[:, 63:65] = 30
fig, axs = plt.subplots(1, 4, figsize=(11, 3.3))
for ax, img, lab in zip(axs, (dead, sat, scr, seam), ("dead column", "saturated spot", "scratch", "tile seam")):
    show(ax, img, title=lab)
fig.tight_layout()
save(fig, f"{OUT}/fig-defects.png", dpi=100)

# ---- fig-harmonic-1d.svg: 2 _ _ _ 10, solution and two Jacobi sweeps from 0 ----
fig, ax = plt.subplots(figsize=(6.0, 3.9))
x = np.arange(5)
ax.plot([0, 4], [2, 10], "s", color="k", ms=11, label="known")
ax.plot(x, [2, 4, 6, 8, 10], "o-", color=BLUE, lw=3, ms=9, label="harmonic fill: 4, 6, 8")
ax.plot(x, [2, 1, 0, 5, 10], "^:", color=ORANGE, lw=2, ms=8, label="Jacobi sweep 1: 1, 0, 5")
ax.plot(x, [2, 1, 3, 5, 10], "v--", color=RED, lw=2, ms=8, label="Jacobi sweep 2: 1, 3, 5")
ax.axvspan(0.5, 3.5, color=GREY, alpha=0.12); ax.text(2, 10.4, "hole", ha="center", color="#555", fontsize=FONT - 2)
ax.set_xticks(x); ax.set_xlabel("pixel"); ax.set_ylabel("value"); ax.set_ylim(-0.5, 11.5)
ax.legend(loc="upper left", frameon=False, fontsize=FONT - 4)
save(fig, f"{OUT}/fig-harmonic-1d.svg")

# ---- fig-two-pixel.svg ----
fig, ax = plt.subplots(figsize=(5.4, 3.6))
cells = {(0, 1): "10", (1, 1): "30", (-1, 0): "10", (0, 0): "a", (1, 0): "b", (2, 0): "30", (0, -1): "10", (1, -1): "30"}
for (i, j), t in cells.items():
    hole = t in ("a", "b")
    ax.add_patch(Rectangle((i - 0.45, j - 0.45), 0.9, 0.9, fc="#f6d9b8" if hole else "#dfe9f5", ec="#555", lw=1.5))
    ax.text(i, j, t, ha="center", va="center", fontsize=FONT + 6, style="italic" if hole else "normal")
ax.text(0.5, -1.9, "4a = 10 + 10 + 10 + b,   4b = 30 + 30 + 30 + a\n→ a = 14, b = 26", ha="center", va="center", fontsize=FONT - 1)
ax.set_xlim(-1.6, 2.6); ax.set_ylim(-2.4, 1.6); ax.set_aspect("equal"); ax.axis("off")
save(fig, f"{OUT}/fig-two-pixel.svg")

# ---- fig-jacobi.png ----
J = D["jacobi"]
fig, axs = plt.subplots(1, 4, figsize=(11, 3.3))
for ax, key, lab in zip(axs, ("k10", "k100", "k1000", None), ("10 sweeps", "100 sweeps", "1000 sweeps", f"direct solve")):
    img = arr(J["snaps"][key] if key else J["solve"])
    show(ax, img[0:48, 14:62], title=lab)
    ax.add_patch(Rectangle((30 - 14 - 0.5, 14 - 0.5), 16, 16, fill=False, ec=CYAN, lw=1.5))
fig.tight_layout()
save(fig, f"{OUT}/fig-jacobi.png", dpi=100)
print("jacobi max diff to solve", J["maxdiff"], "cg iters", J["iters"])

# ---- fig-harmonic-vs-tv.png ----
H = D["htv"]
fig, axs = plt.subplots(1, 3, figsize=(11, 3.9))
for ax, key, lab in zip(axs, ("truth", "harm", "tv"), ("truth", f"harmonic: error {H['err'][0]:.1f}", f"TV: error {H['err'][1]:.1f}")):
    show(ax, arr(H[key])[8:72, 8:72], title=lab)
    m = arr(H["mask"])[8:72, 8:72]; ax.contour(m, levels=[0.5], colors=[CYAN], linewidths=1.4)
fig.tight_layout()
save(fig, f"{OUT}/fig-harmonic-vs-tv.png", dpi=100)
print("htv err", H["err"])

# ---- fig-connectivity.png ----
C = D["conn"]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
for ax, g in zip(axs, (6, 16)):
    e = C[f"g{g}"]; img = arr(e["u"], 128)
    show(ax, img, title=f"bar width 10, gap {g}: {'reconnected' if e['centre'] < 125 else 'left broken'}")
    ax.contour(arr(e["mask"], 128), levels=[0.5], colors=[CYAN], linewidths=1.4)
fig.tight_layout()
save(fig, f"{OUT}/fig-connectivity.png", dpi=100)
print("conn centres", C["g6"]["centre"], C["g16"]["centre"])

# ---- fig-fill-order.png ----
T = D["tex"]
order = arr(T["order"]).astype(float); order[order < 0] = np.nan
fig, axs = plt.subplots(1, 3, figsize=(11, 3.9))
crop = (slice(28, 96), slice(28, 96))
show(axs[0], arr(T["harm"])[crop], title=f"harmonic: error {T['err'][1]:.1f}")
show(axs[1], arr(T["ex"])[crop], title=f"exemplar: error {T['err'][0]:.1f}")
axs[2].imshow(arr(T["truth"])[crop], cmap="gray", vmin=0, vmax=255)
axs[2].imshow(order[crop], cmap="viridis", interpolation="nearest"); axs[2].set_xticks([]); axs[2].set_yticks([])
axs[2].set_title(f"fill order ({T['steps']} patches)", fontsize=FONT - 1)
for ax in axs[:2]:
    ax.contour(arr(T["mask"])[crop], levels=[0.5], colors=[CYAN], linewidths=1.4)
fig.tight_layout()
save(fig, f"{OUT}/fig-fill-order.png", dpi=100)
print("tex err", T["err"], "steps", T["steps"])

# ---- fig-fabrication.png ----
Nn = D["nuc"]
fig, axs = plt.subplots(1, 3, figsize=(11, 3.9))
for ax, key, lab in zip(axs, ("truth", "harm", "ex"), ("truth (hole: empty background)", "harmonic", "exemplar")):
    show(ax, arr(Nn[key]), title=lab)
    ax.contour(arr(Nn["mask"]), levels=[0.5], colors=[CYAN], linewidths=1.4)
fig.tight_layout()
save(fig, f"{OUT}/fig-fabrication.png", dpi=100)
print("nuc bright pixels in hole", Nn["bright"], "truth max in hole", Nn["holeMax"])
