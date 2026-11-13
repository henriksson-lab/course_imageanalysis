"""Static figures for the graph-cut day (slides of seg-energy/, maxflow/, graphcut-seg/, ncut/).

Run from the repository root:  python3 -I tools/figs/graphcut/figs.py
It first bundles tools/figs/graphcut/compute.ts with esbuild and runs it with node: that
script computes every label map, flow and eigenvector with the demos' own code, so the
numbers in the figures are the numbers of the demos and the slides. Then it draws.
Micrograph: demos/src/fourier2d/cell256.png (R. M. Hunt, CC BY 3.0, Wikimedia Commons).
"""
import itertools, json, math, os, subprocess, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})
BLUE, ORANGE, RED, GREEN, PURPLE, GREY, FG = "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9467bd", "#888888", "#1b2a3a"
LBLUE, LORANGE = "#d6e6f4", "#ffe3c4"
DIV = LinearSegmentedColormap.from_list("bo", [BLUE, "#f4f4f4", ORANGE])

# ---------------------------------------------------------------- data from the demo code
tmp = tempfile.mkdtemp()
cell = np.asarray(Image.open("demos/src/fourier2d/cell256.png").convert("L"))
open(os.path.join(tmp, "cell.raw"), "wb").write(cell.tobytes())
subprocess.run(["node_modules/.bin/esbuild", "tools/figs/graphcut/compute.ts", "--bundle", "--platform=node",
                "--log-level=warning", "--outfile=" + os.path.join(tmp, "c.js")], check=True)
subprocess.run(["node", os.path.join(tmp, "c.js"), os.path.join(tmp, "cell.raw"), os.path.join(tmp, "d.json")], check=True)
D = json.load(open(os.path.join(tmp, "d.json")))
N = 128


def img(a, n=N):
    return np.asarray(a, float).reshape(n, n)


def save(fig, path, png=False):
    if png:
        fig.savefig(path, dpi=144, facecolor="white", bbox_inches="tight", pad_inches=0.06)
    else:
        fig.savefig(path, format="svg", facecolor="white", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", path)


def noaxes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def outline(ax, mask, color, lw=2.0):
    ax.contour(np.asarray(mask, float), levels=[0.5], colors=[color], linewidths=lw)


def overlay(ax, mask, color=(1.0, 0.83, 0.0), alpha=0.45):
    m = np.asarray(mask, float)
    rgba = np.zeros(m.shape + (4,)); rgba[..., :3] = color; rgba[..., 3] = alpha * m
    ax.imshow(rgba, interpolation="nearest")


# ---------------------------------------------------------------- graph drawing helpers
def node(ax, xy, name, fc="white", ec=FG, r=0.32, fs=18, lw=2):
    ax.add_patch(Circle(xy, r, fc=fc, ec=ec, lw=lw, zorder=5))
    ax.text(*xy, name, ha="center", va="center", fontsize=fs, zorder=6, fontweight="bold" if name in "st" else "normal")


def arrow(ax, a, b, color=FG, lw=2, r=0.32, ls="-", label=None, lpos=0.5, loff=0.22, fs=16, rad=0.0, lcolor=None, z=2, head=True):
    p = FancyArrowPatch(a, b, arrowstyle="-|>,head_length=7,head_width=4" if head else "-", mutation_scale=1.6,
                        shrinkA=r * 72 / 1.0 * 0 + 0, shrinkB=0, color=color, lw=lw, ls=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=z)
    # shorten ends by the node radius
    a, b = np.array(a, float), np.array(b, float)
    u = (b - a) / np.linalg.norm(b - a)
    p.set_positions(tuple(a + u * r), tuple(b - u * (r + 0.02)))
    ax.add_patch(p)
    if label is not None:
        m = a + (b - a) * lpos
        nrm = np.array([-u[1], u[0]])
        ax.text(*(m + nrm * loff), label, ha="center", va="center", fontsize=fs, color=lcolor or color, zorder=7,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))


# ================================================================ seg-energy
E = D["energy"]
noisy = img(E["I"])
truth = np.zeros((N, N))
yy, xx = np.mgrid[0:N, 0:N]
truth[((xx - 52) ** 2 + (yy - 64) ** 2 <= 900) | ((xx >= 80) & (xx <= 116) & (yy >= 63) & (yy <= 65))] = 1

# fig-specks.png: noisy image and its thresholding
r0 = E["res"][0]
fig, axs = plt.subplots(1, 2, figsize=(6.2, 3.5))
axs[0].imshow(noisy, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
axs[0].set_xlabel("noisy image")
axs[1].imshow(img(r0["L"]), cmap="gray", interpolation="nearest")
axs[1].set_xlabel(f"thresholded: {r0['wrong']} wrong")
for a in axs: noaxes(a)
fig.tight_layout()
save(fig, "seg-energy/fig-specks.png", png=True)

# fig-dataterm.svg: the two data costs against intensity
s, m0, m1 = 50, 80, 170
I = np.linspace(0, 255, 300)
fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.plot(I, (I - m0) ** 2 / (2 * s * s), color=BLUE, lw=3, label=r"$D_p(0)$: background")
ax.plot(I, (I - m1) ** 2 / (2 * s * s), color=ORANGE, lw=3, label=r"$D_p(1)$: object")
ax.axvline(125, color=GREY, ls="--", lw=1.5)
ax.axvspan(0, 125, color=LBLUE, alpha=0.5, lw=0); ax.axvspan(125, 255, color=LORANGE, alpha=0.5, lw=0)
ax.text(62, 6.5, "label 0\ncheaper", ha="center", fontsize=15, color=BLUE)
ax.text(195, 6.5, "label 1\ncheaper", ha="center", fontsize=15, color="#b85a00")
for mu, c in ((m0, BLUE), (m1, ORANGE)):
    ax.plot([mu], [0], "o", color=c, ms=8, clip_on=False)
ax.set_xticks([0, 80, 125, 170, 255]); ax.set_xticklabels(["0", r"$\mu_0$=80", "125", r"$\mu_1$=170", "255"])
ax.set_xlim(0, 255); ax.set_ylim(0, 8); ax.set_yticks([0, 2, 4, 6])
ax.set_xlabel("pixel intensity $I_p$"); ax.set_ylabel("data cost")
ax.legend(loc="upper center", frameon=False, fontsize=15, bbox_to_anchor=(0.5, 1.16), ncol=2)
fig.tight_layout()
save(fig, "seg-energy/fig-dataterm.svg")

# fig-lambda.png: the same noisy image segmented for increasing lambda
fig, axs = plt.subplots(2, 2, figsize=(4.8, 5.7))
for a, r in zip(axs.flat, E["res"]):
    a.imshow(img(r["L"]), cmap="gray", interpolation="nearest")
    outline(a, truth, ORANGE, 1.0)
    a.set_xlabel(f"λ = {r['lambda']:g}\n{r['wrong']} wrong", fontsize=16)
    noaxes(a)
fig.tight_layout(w_pad=0.4, h_pad=1.2)
save(fig, "seg-energy/fig-lambda.png", png=True)

# fig-byhand.svg: energy of all 32 labellings of the 1 x 5 example against lambda
Ipx = np.array([3, 3, 1, 3, 0]); D0, D1 = Ipx ** 2, (Ipx - 3) ** 2
lam = np.linspace(0, 12, 400)
fig, ax = plt.subplots(figsize=(5.6, 4.4))
allE = []
for L in itertools.product([0, 1], repeat=5):
    L = np.array(L); data = np.where(L == 1, D1, D0).sum(); nb = int(np.sum(L[1:] != L[:-1]))
    allE.append(data + nb * lam)
    ax.plot(lam, data + nb * lam, color="#d0d0d0", lw=1, zorder=1)
env = np.min(allE, axis=0)
ax.plot(lam, env, color=FG, lw=5, alpha=0.18, zorder=2)
for lab, data, nb, c, xy in (("11010", 1, 3, BLUE, (1.6, 1.2)), ("11110", 4, 1, ORANGE, (6.5, 8.2)), ("11111", 13, 0, GREEN, (10.0, 13.6))):
    ax.plot(lam, data + nb * lam, color=c, lw=2.8, zorder=3)
    ax.text(*xy, lab, color=c, fontsize=17, fontweight="bold", family="DejaVu Sans Mono")
for x, y in ((1.5, 5.5), (9, 13)):
    ax.plot([x], [y], "o", color=FG, ms=7, zorder=4)
    ax.annotate(f"λ = {x:g}", (x, y), xytext=(x + 0.4, y - 2.4), fontsize=15, arrowprops=dict(arrowstyle="-", color="#555"))
ax.set_xlim(0, 12); ax.set_ylim(0, 20)
ax.set_xlabel("λ"); ax.set_ylabel("energy E(L)")
ax.text(11.8, 0.6, "grey: the other 29 labellings", ha="right", fontsize=13, color="#777")
fig.tight_layout()
save(fig, "seg-energy/fig-byhand.svg")

# fig-chaingrid.svg: a chain has no loops, a grid does
fig, axs = plt.subplots(2, 1, figsize=(4.4, 5.6), gridspec_kw=dict(height_ratios=[0.7, 1.3]))
ax = axs[0]
for k in range(6):
    if k: ax.plot([k - 1, k], [0, 0], color=FG, lw=2, zorder=1)
    ax.add_patch(Circle((k, 0), 0.22, fc="#cfcfcf" if k in (1, 2, 3) else "white", ec=FG, lw=2, zorder=3))
ax.text(2.5, -0.45, "1D chain: no loops\n→ dynamic programming", ha="center", va="top", fontsize=16)
ax.set_xlim(-0.4, 5.4); ax.set_ylim(-1.7, 0.4); ax.set_aspect("equal"); ax.axis("off")
ax = axs[1]
for i in range(4):
    for j in range(4):
        if j < 3: ax.plot([j, j + 1], [i, i], color=FG, lw=2, zorder=1)
        if i < 3: ax.plot([j, j], [i, i + 1], color=FG, lw=2, zorder=1)
for (a, b), (c, d) in (((1, 1), (2, 1)), ((2, 1), (2, 2)), ((2, 2), (1, 2)), ((1, 2), (1, 1))):
    ax.plot([a, c], [b, d], color=ORANGE, lw=5, zorder=2)
for i in range(4):
    for j in range(4):
        ax.add_patch(Circle((j, i), 0.2, fc="white", ec=FG, lw=2, zorder=3))
ax.text(1.5, -0.55, "2D grid: loops\n→ graph cuts", ha="center", va="top", fontsize=16)
ax.set_xlim(-0.6, 3.6); ax.set_ylim(-2.0, 3.5); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
save(fig, "seg-energy/fig-chaingrid.svg")

# fig-trellis.svg: the DP table of the 1 x 5 example, lambda = 2
lam2 = 2
C = np.zeros((2, 5)); B = np.zeros((2, 5), int)
C[:, 0] = [D0[0], D1[0]]
for k in range(1, 5):
    for l, Dk in ((0, D0[k]), (1, D1[k])):
        stay, switch = C[l, k - 1], C[1 - l, k - 1] + lam2
        C[l, k] = Dk + min(stay, switch); B[l, k] = l if stay <= switch else 1 - l
assert list(C[0]) == [9, 11, 3, 12, 6] and list(C[1]) == [0, 0, 4, 4, 13]
path = [int(np.argmin(C[:, 4]))]
for k in range(4, 0, -1): path.insert(0, B[path[0], k])
fig, ax = plt.subplots(figsize=(7.6, 3.4))
ypos = {1: 1.0, 0: 0.0}
X = lambda k: 1.5 * k
for k in range(5):
    for l in (0, 1):
        on = path[k] == l
        if k:
            pl = B[l, k]
            hot = on and path[k - 1] == pl
            ax.annotate("", (X(k) - 0.33, ypos[l]), (X(k - 1) + 0.33, ypos[pl]),
                        arrowprops=dict(arrowstyle="-|>", color=ORANGE if hot else "#aaa", lw=3.5 if hot else 1.4,
                                        shrinkA=0, shrinkB=0))
        ax.add_patch(plt.Rectangle((X(k) - 0.33, ypos[l] - 0.2), 0.66, 0.4, fc=LORANGE if on else "white", ec=ORANGE if on else FG, lw=2.5 if on else 1.5, zorder=3))
        ax.text(X(k), ypos[l], f"{C[l, k]:g}", ha="center", va="center", fontsize=18, zorder=4)
    ax.text(X(k), 1.35, f"pixel {k + 1}\n$I$ = {Ipx[k]}", ha="center", va="bottom", fontsize=15)
ax.text(-0.5, 1.0, "$C_k(1)$", ha="right", va="center", fontsize=17)
ax.text(-0.5, 0.0, "$C_k(0)$", ha="right", va="center", fontsize=17)
ax.text(X(2), -0.4, "orange: read back from the minimum 6 → 11110", ha="center", va="top", fontsize=15, color="#555")
ax.set_xlim(-1.4, 6.5); ax.set_ylim(-0.75, 2.05); ax.axis("off")
save(fig, "seg-energy/fig-trellis.svg")

# ================================================================ maxflow
MF = D["maxflow"]
names = [n["name"] for n in MF["nodes"]]
P = {n["name"]: (n["x"] / 100, (460 - n["y"]) / 100) for n in MF["nodes"]}
EDG = [(names[e["u"]], names[e["v"]], e["c"]) for e in MF["edges"]]
LOFF = {("a", "b"): -0.28, ("d", "c"): 0.28}


def mfgraph(ax, labels=None, colors=None, widths=None, S=None, cut=False, dashed=None, r=0.3, fs=16):
    if S is not None:
        for n in names:
            pass
    for i, (u, v, c) in enumerate(EDG):
        col = (colors or {}).get((u, v), FG)
        lw = (widths or {}).get((u, v), 2)
        ls = "--" if dashed and (u, v) in dashed else "-"
        lab = labels[i] if labels is not None else str(c)
        if col == "#cccccc": lw = 1.4
        arrow(ax, P[u], P[v], color=col, lw=lw, r=r, ls=ls, label=lab, loff=LOFF.get((u, v), 0.24), fs=fs,
              lcolor=col if col != "#bbbbbb" else "#999")
    for n in names:
        fc = "white"
        if S is not None:
            fc = LBLUE if S[n] else LORANGE
        node(ax, P[n], n, fc=fc, r=r)
    ax.set_xlim(0.2, 7.8); ax.set_ylim(0.45, 4.15); ax.set_aspect("equal"); ax.axis("off")


# fig-graph.svg
fig, ax = plt.subplots(figsize=(6.8, 3.4))
mfgraph(ax)
fig.tight_layout(pad=0.2)
save(fig, "maxflow/fig-graph.svg")


def Sset(members):
    return {n: (n in members) for n in names}


def cutinfo(members):
    ce = [(u, v) for u, v, c in EDG if u in members and v not in members]
    return ce, sum(c for u, v, c in EDG if (u, v) in ce)


# fig-cuts.svg: two cuts
fig, axs = plt.subplots(1, 2, figsize=(11.0, 3.4))
for ax, mem in zip(axs, (["s"], ["s", "b"])):
    ce, cost = cutinfo(mem)
    back = [(u, v) for u, v, c in EDG if v in mem and u not in mem]
    cols = {e: RED for e in ce}
    cols.update({e: "#bbbbbb" for e in back})
    mfgraph(ax, colors=cols, widths={e: 3.2 for e in ce}, S=Sset(mem), dashed=ce)
    ax.set_title(f"S = {{{', '.join(mem)}}}: cost " + " + ".join(str(c) for u, v, c in EDG if (u, v) in ce) + f" = {cost}", fontsize=17)
fig.tight_layout(pad=0.3)
save(fig, "maxflow/fig-cuts.svg")

# fig-paths.svg: the four augmenting paths of Edmonds-Karp
fig, axs = plt.subplots(2, 2, figsize=(6.4, 3.9))
prev = [0] * len(EDG)
for ax, st in zip(axs.flat, MF["steps"]):
    pn = [names[k] for k in st["nodes"]]
    pe = list(zip(pn[:-1], pn[1:]))
    labs = [f"{f}/{c}" if (u, v) in pe else None for f, (u, v, c) in zip(st["flow"], EDG)]
    cols = {(u, v): ("#cccccc" if (u, v) not in pe else ORANGE) for u, v, c in EDG}
    mfgraph(ax, labels=labs, colors=cols, widths={e: 3.5 for e in pe}, fs=18, r=0.4)
    ax.set_title(f"{' '.join(pn)}: +{st['b']:g}, flow {st['total']:g}", fontsize=18)
fig.tight_layout(pad=0.3, h_pad=0.6)
save(fig, "maxflow/fig-paths.svg")
assert [s["b"] for s in MF["steps"]] == [4, 6, 4, 5] and MF["total"] == 19

# fig-final.svg: final flow and the minimum cut from the reachable set
Sfin = {n: MF["S"][i] for i, n in enumerate(names)}
mem = [n for n in names if Sfin[n]]
assert sorted(mem) == ["b", "s"]
ce, cost = cutinfo(mem)
fig, ax = plt.subplots(figsize=(5.2, 3.0))
labs = [f"{f}/{c}" for f, (u, v, c) in zip(MF["flow"], EDG)]
back = [(u, v) for u, v, c in EDG if v in mem and u not in mem]
cols = {e: RED for e in ce}; cols.update({e: "#bbbbbb" for e in back})
mfgraph(ax, labels=labs, colors=cols, widths={e: 3.2 for e in ce}, S=Sfin, dashed=ce, fs=17, r=0.34)
ax.set_title(f"flow 19 = cut 10 + 9 = {cost}", fontsize=17)
fig.tight_layout(pad=0.2)
save(fig, "maxflow/fig-final.svg")

# fig-backedge.svg
BP = {"s": (0, 1), "a": (1.6, 2), "b": (1.6, 0), "t": (3.2, 1)}
fig, axs = plt.subplots(2, 1, figsize=(4.4, 5.0))
ax = axs[0]
for (u, v, f) in (("s", "a", 1), ("s", "b", 0), ("a", "b", 1), ("a", "t", 0), ("b", "t", 1)):
    on = f == 1
    arrow(ax, BP[u], BP[v], color=ORANGE if on else FG, lw=4 if on else 2, r=0.28, label=f"{f}/1", loff=0.25 if (u, v) != ("a", "b") else -0.3)
ax.set_title("after path s a b t: flow 1", fontsize=15)
ax2 = axs[1]
# residual graph after the first path
res = [("s", "b", "1", FG, False), ("a", "t", "1", FG, False), ("a", "s", "1", RED, False), ("b", "a", "1", RED, True), ("t", "b", "1", RED, False)]
arrow(ax2, BP["s"], BP["b"], color=ORANGE, lw=4, r=0.28, label="1", loff=-0.25)
arrow(ax2, BP["a"], BP["t"], color=ORANGE, lw=4, r=0.28, label="1", loff=0.25)
arrow(ax2, BP["b"], BP["a"], color=ORANGE, lw=4, r=0.28, label="1", loff=0.3, rad=0.0)
arrow(ax2, BP["a"], BP["s"], color=RED, lw=1.6, r=0.28, label="1", loff=0.25, ls="--")
arrow(ax2, BP["t"], BP["b"], color=RED, lw=1.6, r=0.28, label="1", loff=0.25, ls="--")
ax2.text(1.72, 0.85, "b→a", color=RED, fontsize=17, va="center", ha="left")
ax2.set_title("residual: path s b a t, +1", fontsize=15)
for a in axs:
    for n, xy in BP.items(): node(a, xy, n, r=0.28)
    a.set_xlim(-0.4, 3.6); a.set_ylim(-0.4, 2.4); a.set_aspect("equal"); a.axis("off")
fig.tight_layout(pad=0.3)
save(fig, "maxflow/fig-backedge.svg")
assert D["backedge"] == 2

# ================================================================ graphcut-seg
# fig-graph.svg: a 3 x 3 image as a graph
vals = np.array([[60, 200, 210], [50, 190, 220], [40, 70, 200]])
gpos = {(i, j): (j * 1.5 + (2 - i) * 0.55, (2 - i) * 0.75) for i in range(3) for j in range(3)}
sP, tP = (2.05, 3.4), (2.05, -1.9)
fig, ax = plt.subplots(figsize=(4.9, 4.4))
for (i, j), xy in gpos.items():
    v = vals[i, j]
    d0 = (v - 30) ** 2 / 2 / 60 ** 2; d1 = (v - 220) ** 2 / 2 / 60 ** 2
    ax.plot([sP[0], xy[0]], [sP[1], xy[1]], color=BLUE, lw=0.6 + 1.6 * d0, alpha=0.75, zorder=1)
    ax.plot([tP[0], xy[0]], [tP[1], xy[1]], color=ORANGE, lw=0.6 + 1.6 * d1, alpha=0.75, zorder=1)
for (i, j), xy in gpos.items():
    for di, dj in ((0, 1), (1, 0)):
        if (i + di, j + dj) in gpos:
            w = math.exp(-(vals[i, j] - vals[i + di, j + dj]) ** 2 / (2 * 40 ** 2))
            q = gpos[(i + di, j + dj)]
            ax.plot([xy[0], q[0]], [xy[1], q[1]], color=FG, lw=0.6 + 5 * w, zorder=2, solid_capstyle="round")
for (i, j), xy in gpos.items():
    g = vals[i, j] / 255
    ax.add_patch(Circle(xy, 0.26, fc=(g, g, g), ec=FG, lw=1.8, zorder=4))
node(ax, sP, "s", fc=LBLUE, ec=BLUE, r=0.34)
node(ax, tP, "t", fc=LORANGE, ec=ORANGE, r=0.34)
ax.text(sP[0] + 0.5, sP[1], "source = object", va="center", fontsize=16, color=BLUE)
ax.text(tP[0] + 0.5, tP[1], "sink = background", va="center", fontsize=16, color="#b85a00")
ax.text(4.75, 0.75, "n-links:\nthick =\nsimilar", va="center", fontsize=15, color=FG)
ax.text(-0.95, 2.35, "t-links", fontsize=15, color=BLUE)
ax.set_xlim(-1.1, 6.0); ax.set_ylim(-2.4, 3.9); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout(pad=0.1)
save(fig, "graphcut-seg/fig-graph.svg")

# fig-1x3.svg: the by-hand example with its minimum cut (lambda = 1)
px = {1: (0, 0), 2: (2, 0), 3: (4, 0)}
sP, tP = (2, 2.2), (2, -2.2)
D0x, D1x = {1: 9, 2: 1, 3: 9}, {1: 0, 2: 4, 3: 0}
S = {1, 3}
fig, ax = plt.subplots(figsize=(6.2, 5.0))
total = 0
for p, xy in px.items():
    cut_s = p not in S; cut_t = p in S
    c = RED if cut_s else BLUE
    arrow(ax, sP, xy, color=c, lw=3.2 if cut_s else 2, ls="--" if cut_s else "-", r=0.34, label=str(D0x[p]), lpos=0.55, loff=0.0, lcolor=c)
    if D1x[p] > 0:
        c = RED if cut_t else ORANGE
        arrow(ax, xy, tP, color=c, lw=3.2 if cut_t else 2, ls="--" if cut_t else "-", r=0.34, label=str(D1x[p]), lpos=0.45, loff=0.0, lcolor=c)
    else:
        arrow(ax, xy, tP, color="#c8c8c8", lw=1.2, ls=":", r=0.34, label="0", lpos=0.45, loff=0.0, lcolor="#999")
    total += D0x[p] if cut_s else (D1x[p] if cut_t else 0)
for p, q in ((1, 2), (3, 2)):
    arrow(ax, px[p], px[q], color=RED, lw=3.2, ls="--", r=0.34, label="1", loff=0.3, lcolor=RED)
    arrow(ax, px[q], px[p], color="#bbbbbb", lw=1.2, r=0.34, head=True, rad=0.35)
    total += 1
assert total == 3
for p, xy in px.items():
    node(ax, xy, str(p), fc=LBLUE if p in S else LORANGE)
node(ax, sP, "s", fc=LBLUE, ec=BLUE, r=0.36)
node(ax, tP, "t", fc=LORANGE, ec=ORANGE, r=0.36)
ax.text(4.9, 0, "", fontsize=15)
ax.text(2, -3.0, "cut = 1 + 1 + 1 = 3, labelling 101", ha="center", fontsize=17, color=RED)
ax.set_xlim(-0.8, 4.8); ax.set_ylim(-3.3, 2.8); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout(pad=0.1)
save(fig, "graphcut-seg/fig-1x3.svg")

# fig-contrast.png: the micrograph and its contrast weights
CT = D["contrast"]
fig, axs = plt.subplots(1, 2, figsize=(5.8, 3.1), gridspec_kw=dict(width_ratios=[1, 1.17]))
axs[0].imshow(img(CT["I"]), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
axs[0].set_xlabel("micrograph")
im = axs[1].imshow(img(CT["w"]["25"]), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
axs[1].set_xlabel("weight $w_{pq}$, σ = 25")
for a in axs: noaxes(a)
cb = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.03, ticks=[0, 0.5, 1])
cb.ax.tick_params(labelsize=14)
fig.tight_layout()
save(fig, "graphcut-seg/fig-contrast.png", png=True)


def seedrgb(I, seeds):
    g = np.clip(img(I) / 255, 0, 1)
    rgb = np.dstack([g, g, g])
    sd = img(seeds)
    rgb[sd == 1] = [0.84, 0.15, 0.16]
    rgb[sd == -1] = [0.12, 0.47, 0.71]
    return rgb


# fig-seeds.png: nucleus preset: seeds, graph cut, thresholding with the same model
G = D["gc"]
nu = G["nucleus"]
fig, axs = plt.subplots(1, 3, figsize=(8.0, 3.2))
axs[0].imshow(seedrgb(nu["I"], nu["seeds"]), interpolation="nearest")
axs[0].set_xlabel("seeds")
axs[1].imshow(img(nu["I"]), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
overlay(axs[1], img(nu["L"])); outline(axs[1], img(nu["L"]), "#ffd400", 1.6)
axs[1].set_xlabel("graph cut, λ = 10")
axs[2].imshow(img(nu["I"]), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
overlay(axs[2], img(nu["thr"]))
axs[2].set_xlabel("λ = 0")
for a in axs: noaxes(a)
fig.tight_layout()
save(fig, "graphcut-seg/fig-seeds.png", png=True)

# fig-failures.png: shrinking, metrication, overlapping intensities
fig, axs = plt.subplots(2, 2, figsize=(5.4, 5.9))
items = [("shrink", "λ = 6: arm cut off"), ("blocky4", "4 neighbours"),
         ("blocky8", "8 neighbours"), ("badmodel", "overlapping")]
for a, (k, lab) in zip(axs.flat, items):
    g = G[k]
    a.imshow(img(g["I"]), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    overlay(a, img(g["L"]), alpha=0.35); outline(a, img(g["L"]), "#ffd400", 1.8)
    a.set_xlabel(lab, fontsize=16)
    noaxes(a)
fig.tight_layout(h_pad=0.6, w_pad=0.6)
save(fig, "graphcut-seg/fig-failures.png", png=True)

# ================================================================ ncut
# fig-triangles.svg: two triangles and a pendant node; the minimum cut and the cut we want
TP = {1: (0, 0), 2: (0, 2), 3: (1.6, 1), 4: (3.4, 1), 5: (5.0, 2), 6: (5.0, 0), 7: (6.6, -0.6)}
TE = [(1, 2, 1), (1, 3, 1), (2, 3, 1), (4, 5, 1), (4, 6, 1), (5, 6, 1), (3, 4, 0.1), (6, 7, 0.05)]
fig, ax = plt.subplots(figsize=(5.8, 3.4))
for u, v, w in TE:
    a, b = TP[u], TP[v]
    ax.plot([a[0], b[0]], [a[1], b[1]], color=FG, lw=1 + 3 * w, zorder=1)
    if w < 1:
        m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        ax.text(m[0] + (0.35 if w > 0.06 else -0.55), m[1] + (0.32 if w > 0.06 else -0.6), f"{w:g}", ha="center", fontsize=16)
xt = D["tri"]["x"]
for k, xy in TP.items():
    ax.add_patch(Circle(xy, 0.3, fc=LBLUE if xt[k - 1] < 0 else LORANGE, ec=FG, lw=2, zorder=3))
    ax.text(*xy, str(k), ha="center", va="center", fontsize=16, zorder=4)
ax.plot([2.5, 2.5], [-0.7, 2.6], color=BLUE, lw=3, ls="--")
ax.text(2.5, 2.75, "cut 0.1: the two groups", ha="center", fontsize=16, color=BLUE)
ax.plot([5.45, 6.4], [-1.1, 0.3], color=RED, lw=3, ls="--")
ax.text(6.0, -1.45, "minimum cut 0.05", ha="center", fontsize=16, color=RED)
ax.text(0.8, -1.45, "other edges: weight 1", ha="center", fontsize=14, color="#666")
ax.set_xlim(-0.6, 7.4); ax.set_ylim(-1.8, 3.2); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout(pad=0.1)
save(fig, "ncut/fig-triangles.svg")
assert abs(D["tri"]["minCut"] - 0.05) < 1e-9

# fig-fiedler.svg: two blobs coloured by the Fiedler vector, and the vector sorted
T2 = D["two"]
pts = np.array(T2["pts"]); x = np.array(T2["x"])
fig, axs = plt.subplots(2, 1, figsize=(4.6, 5.0), gridspec_kw=dict(height_ratios=[1, 0.9]))
ax = axs[0]
for p, q, w in T2["edges"]:
    if w > 0.3:
        ax.plot(*pts[[p, q]].T, color="#cccccc", lw=0.6, zorder=1)
sc = ax.scatter(pts[:, 0], pts[:, 1], c=x, cmap=DIV, vmin=-1, vmax=1, s=70, edgecolors=FG, linewidths=0.8, zorder=3)
ax.set_aspect("equal"); ax.set_xlim(0.05, 0.95); ax.set_ylim(0.15, 0.85); noaxes(ax)
ax = axs[1]
xs = np.sort(x)
ax.scatter(np.arange(len(xs)), xs, c=xs, cmap=DIV, vmin=-1, vmax=1, s=40, edgecolors=FG, linewidths=0.6, zorder=3)
ax.axhline(0, color=GREY, ls="--", lw=1.5)
nneg = int(np.sum(x < 0))
ax.text(1, 0.12, f"{nneg} negative | {len(x) - nneg} positive", fontsize=15)
ax.set_xlabel("nodes, sorted"); ax.set_ylabel("Fiedler vector value")
ax.set_yticks([-1, 0, 1])
fig.tight_layout()
save(fig, "ncut/fig-fiedler.svg")

# fig-path4.svg: the Fiedler vector of a path of 4
v = np.array([1, math.sqrt(2) - 1, 1 - math.sqrt(2), -1])
Lp = np.array([[1, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]], float)
assert np.allclose(Lp @ v, (2 - math.sqrt(2)) * v)
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.bar(range(1, 5), v, color=[ORANGE if t > 0 else BLUE for t in v], width=0.55)
ax.plot(range(1, 5), [0] * 4, "-", color=FG, lw=2)
for k in range(1, 5):
    ax.add_patch(Circle((k, 0), 0.13, fc="white", ec=FG, lw=1.8, zorder=4, transform=ax.transData))
ax.set_xticks(range(1, 5)); ax.set_xlabel("node"); ax.set_ylabel("$v_2$")
ax.set_yticks([-1, -0.41, 0, 0.41, 1]); ax.set_yticklabels(["−1", "−0.41", "0", "0.41", "1"])
ax.set_ylim(-1.2, 1.2); ax.set_aspect(0.9)
fig.tight_layout()
save(fig, "ncut/fig-path4.svg")

# fig-outlier.svg: min cut vs normalized cut on blobs + outlier (sigma 0.08)
O = D["outlier0.08"]
pts = np.array(O["pts"])
fig, axs = plt.subplots(1, 2, figsize=(5.6, 3.3))
for ax, side, st, ttl in ((axs[0], O["minCut"], O["minStats"], "min cut"), (axs[1], O["ncutSide"], O["ncutStats"], "Ncut")):
    side = np.array(side)
    # colour the smaller side orange
    if side.sum() > len(side) / 2: side = 1 - side
    ax.scatter(pts[:, 0], pts[:, 1], c=[ORANGE if s_ else BLUE for s_ in side], s=70, edgecolors=FG, linewidths=0.8)
    nA = int(side.sum())
    ax.set_title(f"{ttl}: {nA} | {len(side) - nA}", fontsize=16)
    ax.set_xlabel(f"Ncut = {st['ncut']:.3f}", fontsize=16)
    ax.set_aspect("equal"); ax.set_xlim(0.02, 1.0); ax.set_ylim(0.0, 0.95)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#cccccc")
fig.tight_layout()
save(fig, "ncut/fig-outlier.svg")
print("outlier", O["minStats"], O["ncutStats"])

# fig-imgseg.png: image, Fiedler vector as an image, its sign
for name in ("disk",):
    R = D["img_" + name]
    I32 = img(R["I"], 32); X32 = img(R["x"], 32)
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.9))
    axs[0].imshow(I32, cmap="gray", vmin=0, vmax=255, interpolation="nearest"); axs[0].set_xlabel("32 × 32 image")
    axs[1].imshow(X32, cmap=DIV, vmin=-1, vmax=1, interpolation="nearest"); axs[1].set_xlabel("Fiedler vector")
    axs[2].imshow(X32 > 0, cmap="gray", interpolation="nearest"); axs[2].set_xlabel("threshold at 0")
    for a in axs: noaxes(a)
    fig.tight_layout()
    save(fig, f"ncut/fig-imgseg.png", png=True)
    print(name, "mu", R["mu"], "positive", int((X32 > 0).sum()))
