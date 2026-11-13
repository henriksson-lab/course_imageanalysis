"""Figures for coloc-objects/lec.qmd. fig-detect, fig-relocate and fig-below use the data written
by data.ts (run that first; see its header); fig-overlapdefs, fig-nn and fig-rayleigh are
diagrams and a seeded simulation computed here.
Usage: python3 -I tools/figs/coloc-objects/figs.py <datadir>   (from the repo root)"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "coloc"))
from colocfig import BLUE, ORANGE, VERM, GREY, GREEN, PURPLE, fig, save, merge, noaxis, hexrgb, np  # noqa: E402

D = json.load(open(os.path.join(sys.argv[1], "coloc-objects.json")))
OUT = "coloc-objects"
PX = 65  # nm per pixel
MAG = "#C2369C"  # magenta for channel B marks (as in the demo)


def im(v):
    return np.asarray(v, float).reshape(128, 128)


# 1. detected spots on the merge (preset "pairs at 0 nm"), zoomed on the lower cell
d = D["pairs0"]
A, B, mask = im(d["a"]), im(d["b"]), im(d["mask"]) > 0
lo = 100  # camera offset
f, ax = fig(450, 430)
ax.imshow(merge(A - lo, B - lo, (0, 0.6 * (A - lo).max()), (0, 0.6 * (B - lo).max())), interpolation="nearest")
ax.contour(mask, [0.5], colors=["#bbbbbb"], linewidths=1)
da, db = np.asarray(d["detA"]), np.asarray(d["detB"])
dist, partner = np.asarray(d["dist"]), np.asarray(d["partner"])
for i, (x, y) in enumerate(da):
    if dist[i] * PX <= 130:
        bx, by = db[partner[i]]
        ax.plot([x, bx], [y, by], color="#F0E442", lw=2)
ax.scatter(da[:, 0], da[:, 1], s=110, facecolors="none", edgecolors="#33dd55", linewidths=1.8)
ax.scatter(db[:, 0], db[:, 1], s=200, facecolors="none", edgecolors="#ff66dd", linewidths=1.8)
ax.set_xlim(-0.5, 127.5); ax.set_ylim(127.5, -0.5)
noaxis(ax)
ax.set_title(f"{len(da)} A spots (green), {len(db)} B spots (magenta)", fontsize=17)
f.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93)
save(f, f"{OUT}/fig-detect.svg")

# 2. three definitions of "A has a partner in B" on two pixel examples (geometry of the demo
#    presets "big objects" and "small A inside big B": radii 6 and 4 px, centres 5 px apart)


def disc(n, cx, cy, r):
    y, x = np.mgrid[0:n, 0:n]
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


f, axs = fig(500, 290, 1, 2)
n = 19
for ax, (rA, rB), title in zip(axs, [(6, 4), (4, 6)], ["large A, small B", "small A, large B"]):
    a = disc(n, 7, 9, rA); b = disc(n, 12, 9, rB)
    rgb = np.ones((n, n, 3))
    rgb[a] = hexrgb(GREEN); rgb[b] = (0.85, 0.85, 0.85); rgb[a & b] = hexrgb(BLUE)
    ax.imshow(rgb, interpolation="nearest")
    ax.contour(b.astype(float), [0.5], colors=[MAG], linewidths=2)
    ax.plot(7, 9, marker="+", color="black", ms=16, mew=3)
    share = (a & b).sum() / a.sum()
    cin = b[9, 7]
    ax.set_title(title, fontsize=17)
    ax.text(n / 2 - 0.5, n - 2.0, f"any: yes\n> 50 %: {'yes' if share > 0.5 else 'no'} ({100 * share:.0f} %)\ncentroid in B: {'yes' if cin else 'no'}",
            ha="center", va="top", fontsize=16)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n + 4.5, 1.5)
    noaxis(ax)
f.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.9, wspace=0.08)
save(f, f"{OUT}/fig-overlapdefs.svg")

# 3. nearest neighbour vs one-to-one (a made-up example, d_max = 2 px)
Apts = np.array([(2, 2), (5.8, 6.4), (8.8, 6.2), (13, 3)], float)
Bpts = np.array([(3.3, 3.0), (7.3, 7.2), (14, 8), (11, 11)], float)
dmax = 2.0
f, axs = fig(500, 330, 1, 2, sharey=True)
D2 = np.hypot(*(Apts[:, None, :] - Bpts[None, :, :]).transpose(2, 0, 1))
nn = D2.argmin(1)
# one-to-one: most pairs within d_max, then the smallest total distance (brute force, 4 x 4)
import itertools  # noqa: E402
best = min(itertools.permutations(range(4)), key=lambda pm: (-sum(D2[i, j] <= dmax for i, j in enumerate(pm)), sum(min(D2[i, j], dmax) for i, j in enumerate(pm))))
one = {i: j for i, j in enumerate(best) if D2[i, j] <= dmax}
for ax, title, links in [(axs[0], "nearest neighbour", {i: nn[i] for i in range(4)}), (axs[1], "one-to-one", one)]:
    for c in Bpts:
        ax.add_patch(__import__("matplotlib").patches.Circle(c, dmax, color=MAG, alpha=0.08, lw=0))
    for i, j in links.items():
        ok = D2[i, j] <= dmax
        ax.annotate("", xy=Bpts[j], xytext=Apts[i], arrowprops=dict(arrowstyle="->", color=BLUE if ok else GREY, lw=2, ls="-" if ok else "--", shrinkA=6, shrinkB=6))
    ax.scatter(Apts[:, 0], Apts[:, 1], s=90, color=GREEN, zorder=3, label="A")
    ax.scatter(Bpts[:, 0], Bpts[:, 1], s=90, color=MAG, marker="s", zorder=3, label="B")
    within = sum(D2[i, j] <= dmax for i, j in links.items())
    ax.set_title(f"{title}\n{within} of 4 within {dmax:.0f} px", fontsize=16)
    ax.set_aspect("equal"); ax.set_xlim(0, 16); ax.set_ylim(13, 0)
    ax.set_xticks([0, 5, 10, 15]); ax.set_xlabel("x (px)")
axs[0].set_ylabel("y (px)"); axs[0].set_yticks([0, 5, 10])
axs[0].legend(loc="lower left", frameon=False, handletextpad=0.2, borderaxespad=0)
f.subplots_adjust(left=0.1, right=0.99, bottom=0.18, top=0.84, wspace=0.08)
save(f, f"{OUT}/fig-nn.svg")

# 4. random relocation of the B spots: inside the cells vs over the whole image
f, axs = fig(500, 300, 1, 3)
for ax, (pts, title) in zip(axs, [(db, "detected"), (np.asarray(d["inMask"]), "random,\nin the cells"), (np.asarray(d["whole"]), "random,\nanywhere")]):
    ax.imshow(np.where(mask, 0.93, 1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.contour(mask, [0.5], colors=[GREY], linewidths=1)
    ax.scatter(da[:, 0], da[:, 1], s=14, color=GREEN)
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color=MAG, marker="s")
    ax.set_title(f"B {title}", fontsize=16)
    ax.set_xlim(-0.5, 127.5); ax.set_ylim(127.5, -0.5)
    noaxis(ax)
f.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.84, wspace=0.05)
save(f, f"{OUT}/fig-relocate.svg")

# 5. below the resolution: two spots 60 nm apart, and the measured distances of the demo
e = D["d60"]
f, axs = fig(500, 340, 1, 2)
x = np.linspace(-400, 460, 600)
sA, sB = 0.21 * 520 / 1.4, 0.21 * 670 / 1.4
axs[0].plot(x, np.exp(-x ** 2 / (2 * sA ** 2)), color=GREEN, lw=2.5, label="A")
axs[0].plot(x, np.exp(-(x - 60) ** 2 / (2 * sB ** 2)), color=MAG, lw=2.5, ls="--", label="B")
axs[0].axvline(0, color=GREEN, lw=1.5, ymax=0.25); axs[0].axvline(60, color=MAG, lw=1.5, ymax=0.25)
axs[0].set_xlim(-350, 410); axs[0].set_ylim(0, 1.25)
axs[0].set_xticks([-300, 0, 300]); axs[0].set_yticks([])
axs[0].set_xlabel("position (nm)")
axs[0].set_title("60 nm apart", fontsize=17)
axs[0].legend(frameon=False, loc="upper right", fontsize=16, handlelength=1.4)
m = np.asarray(e["measured"])
axs[1].hist(m, bins=np.arange(0, 141, 10), color=BLUE, alpha=0.75)
axs[1].axvline(60, color=GREEN, lw=3, zorder=3)
axs[1].axvline(m.mean(), color=VERM, lw=3, ls="--", zorder=3)
axs[1].text(57, 11.5, "true", color=GREEN, ha="right", fontsize=16)
axs[1].text(68, 11.5, "mean", color=VERM, ha="left", fontsize=16)
axs[1].set_ylim(0, 12.5)
axs[1].set_xlabel("measured (nm)"); axs[1].set_ylabel("pairs")
axs[1].set_title(f"mean {m.mean():.0f} nm", fontsize=17)
f.subplots_adjust(left=0.03, right=0.98, bottom=0.2, top=0.9, wspace=0.35)
save(f, f"{OUT}/fig-below.svg")

# 6. the distance bias at d = 0: Rayleigh distribution, sigma 20 nm per channel and axis
rng = np.random.default_rng(7)
s = np.sqrt(20 ** 2 + 20 ** 2)
eps = rng.normal(0, s, (20000, 2))
dm = np.hypot(eps[:, 0], eps[:, 1])
f, ax = fig(450, 300)
ax.hist(dm, bins=np.arange(0, 121, 4), density=True, color=BLUE, alpha=0.6, label="simulated, 20 000 pairs")
r = np.linspace(0, 120, 300)
ax.plot(r, r / s ** 2 * np.exp(-r ** 2 / (2 * s ** 2)), color=VERM, lw=2.5, label="Rayleigh, s = 28.3 nm")
ax.axvline(s * np.sqrt(np.pi / 2), color="black", lw=2, ls="--")
ax.text(s * np.sqrt(np.pi / 2) + 3, 0.024, f"mean {s * np.sqrt(np.pi / 2):.1f} nm", fontsize=16)
ax.set_xlabel("measured distance (nm), true distance 0"); ax.set_ylabel("density")
ax.set_yticks([0, 0.01, 0.02]); ax.set_ylim(0, 0.036)
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.02, 1.04))
f.subplots_adjust(left=0.16, right=0.97, bottom=0.2, top=0.96)
save(f, f"{OUT}/fig-rayleigh.svg")
