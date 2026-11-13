"""Figures for chanvese/lec.qmd. The image figures use the runs of data.ts (the chanvese demo's
model and settings); the small diagrams are computed here. Run from the repo root (see data.ts):
   python3 -I tools/figs/chanvese/figs.py <tmp>/cv-data.json
PNG: figure size in inches = display size / 100 px, dpi 200, fonts >= 12 pt.
SVG: 1 pt in the figure = 1 px on the slide at the width given in lec.qmd."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_gradient_magnitude

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "chanvese")
plt.rcParams.update({"font.size": 13, "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, ORANGE, SKY, PURPLE, YELLOW, GREY = "#0072b2", "#d55e00", "#009e73", "#e69f00", "#56b4e9", "#cc79a7", "#ffd400", "#777777"


def save(fig, name, **kw):
    if name.endswith(".png"):
        kw.setdefault("dpi", 200)
    fig.savefig(os.path.join(OUT, name), facecolor="white", bbox_inches="tight", pad_inches=0.04, **kw)
    plt.close(fig); print("wrote", name)


def show(ax, img, vmin=0, vmax=1, cmap="gray"):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])


def front(ax, phi, color=YELLOW, lw=2.0, ls="-"):
    cs = ax.contour(phi, levels=[0], colors=[color], linewidths=lw, linestyles=ls)
    cs.set_path_effects([pe.Stroke(linewidth=lw + 2.0, foreground="black"), pe.Normal()])


# ------------------------------------------------------------------ small diagrams (no data)
I8 = np.array([1, 1, 2, 6, 7, 7, 6, 1], float)


def fig_eight():
    ins = np.zeros(8, bool); ins[3:7] = True
    c1, c2 = I8[ins].mean(), I8[~ins].mean()
    assert abs(c1 - 6.5) < 1e-9 and abs(c2 - 1.25) < 1e-9
    fig, ax = plt.subplots(figsize=(7.4, 2.35))
    x = np.arange(1, 9)
    ax.bar(x, I8, color=[SKY if i else "#cccccc" for i in ins], edgecolor="black", width=0.8)
    ax.plot([3.5, 7.5], [c1, c1], color=BLUE, lw=3)
    ax.plot([0.5, 3.5], [c2, c2], color=RED, lw=3); ax.plot([7.5, 8.5], [c2, c2], color=RED, lw=3)
    ax.text(5.5, 7.4, "c₁ = 6.5", color=BLUE, ha="center", fontsize=16)
    ax.text(1.5, c2 + 0.4, "c₂ = 1.25", color=RED, ha="center", fontsize=16)
    for xb in (3.5, 7.5):
        ax.axvline(xb, color="black", ls="--", lw=1.5)
    ax.set_xticks(x); ax.set_xlim(0.4, 8.6); ax.set_ylim(0, 9.0)
    ax.set_xlabel("pixel"); ax.set_ylabel("I", rotation=0, labelpad=10)
    ax.tick_params(labelsize=14)
    fig.tight_layout(pad=0.2); save(fig, "fig-eight.svg")


def fig_means():
    c1, c2, v = 6.5, 1.25, 4.0
    d1, d2 = (v - c1) ** 2, (v - c2) ** 2
    print("pixel 4:", d1, d2, d1 - d2)
    fig, ax = plt.subplots(figsize=(7.6, 2.1))
    ax.axhline(0, color="black", lw=1.5)
    for t in range(0, 9):
        ax.plot([t, t], [-0.08, 0.08], color="black", lw=1)
        ax.text(t, -0.28, str(t), ha="center", va="top", fontsize=14)
    ax.plot([c2], [0], "o", ms=13, color=RED); ax.text(c2, 0.3, "c₂ = 1.25\n(outside)", color=RED, ha="center", va="bottom", fontsize=15)
    ax.plot([c1], [0], "o", ms=13, color=BLUE); ax.text(c1, 0.3, "c₁ = 6.5\n(inside)", color=BLUE, ha="center", va="bottom", fontsize=15)
    ax.plot([(c1 + c2) / 2], [0], "|", ms=26, mew=2.5, color=GREY)
    ax.text((c1 + c2) / 2, 0.3, "midpoint\n3.875", color=GREY, ha="center", va="bottom", fontsize=14)
    ax.plot([v], [0], "D", ms=11, color="black", mfc=YELLOW, zorder=5)
    ax.text(4.0, -0.8, f"pixel of grey level 4:  (4 − 6.5)² = {d1:.2f}  <  (4 − 1.25)² = {d2:.2f}: joins the inside", ha="center", va="top", fontsize=14)
    ax.set_xlim(-0.4, 8.6); ax.set_ylim(-1.35, 1.35)
    ax.axis("off")
    fig.tight_layout(pad=0.1); save(fig, "fig-means.svg")


def fig_metric():
    """A disk of radius 6 on the pixel grid: its 4-neighbour boundary (pixel edges) and its smooth outline."""
    r, n = 6.0, 16
    yy, xx = np.mgrid[0:n, 0:n] + 0.5
    cx = cy = n / 2
    m = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
    edges = 0
    fig, axs = plt.subplots(1, 2, figsize=(4.6, 2.75))
    for ax in axs:
        ax.set_xlim(0, n); ax.set_ylim(n, 0); ax.set_aspect("equal"); ax.axis("off")
        for k in range(n + 1):
            ax.plot([k, k], [0, n], color="#e6e6e6", lw=0.8); ax.plot([0, n], [k, k], color="#e6e6e6", lw=0.8)
    ax = axs[0]
    for y in range(n):
        for x in range(n):
            if m[y, x]:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color=SKY, alpha=0.6, lw=0))
                for dx, dy, seg in ((1, 0, ((x + 1, y), (x + 1, y + 1))), (-1, 0, ((x, y), (x, y + 1))),
                                    (0, 1, ((x, y + 1), (x + 1, y + 1))), (0, -1, ((x, y), (x + 1, y)))):
                    if not m[y + dy, x + dx]:
                        edges += 1
                        ax.plot(*zip(*seg), color="black", lw=2.2)
    ax.set_title(f"graph cut\n(4 neighbours): {edges}", fontsize=16)
    ax = axs[1]
    t = np.linspace(0, 2 * np.pi, 200)
    ax.fill(cx + r * np.cos(t), cy + r * np.sin(t), color=SKY, alpha=0.6, lw=0)
    ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), color="black", lw=2.2)
    ax.set_title(f"level set curve\n2πr = {2 * np.pi * r:.1f}", fontsize=16)
    fig.tight_layout(pad=0.2, w_pad=0.5); save(fig, "fig-metric.svg")
    print("4-neighbour boundary", edges, "true", 2 * np.pi * r)


# ------------------------------------------------------------------ image figures
def figs_data(d):
    N = d["N"]
    A = lambda a: np.array(a).reshape(N, N)

    # when edges are not enough: noise 0.3
    R = d["noise"]; I = A(R["I"]); mask = A(R["mask"]); s = R["snaps"][-1]
    th = (s["c1"] + s["c2"]) / 2
    thr = I > th
    assert int((thr != mask).sum()) == s["thr"]
    gm = gaussian_gradient_magnitude(I, 1.0)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.85))
    show(axs[0], I); axs[0].set_title("image, noise 0.3", fontsize=12.5)
    show(axs[1], gm, vmin=0, vmax=np.percentile(gm, 99.5), cmap="magma"); axs[1].set_title("edge strength |∇I|", fontsize=12.5)
    show(axs[2], thr.astype(float)); axs[2].set_title(f"threshold: {s['thr']} wrong", fontsize=12.5)
    fig.tight_layout(pad=0.2, w_pad=0.5); save(fig, "fig-noedge.png")

    # the energy: two-tone fit and squared error (same run, iteration 300)
    phi = A(s["phi"]); inside = phi < 0
    two = np.where(inside, s["c1"], s["c2"])
    err = (I - two) ** 2
    fig, axs = plt.subplots(1, 2, figsize=(4.2, 2.45))
    show(axs[0], two); front(axs[0], phi, lw=1.4); axs[0].set_title(f"{s['c1']:.2f} in, {s['c2']:.2f} out", fontsize=15)
    show(axs[1], err, vmin=0, vmax=0.3, cmap="magma"); axs[1].set_title("(I − c)²", fontsize=15)
    fig.tight_layout(pad=0.2, w_pad=0.5); save(fig, "fig-twotone.png")
    print("noise run: c1 %.3f c2 %.3f wrong %d thr %d" % (s["c1"], s["c2"], s["wrong"], s["thr"]))

    # λ
    fig, axs = plt.subplots(2, 2, figsize=(5.0, 5.35))
    show(axs[1, 1], A(d["lambdaD"]["mask"]).astype(float)); axs[1, 1].set_title("true objects", fontsize=13)
    for ax, key, lab in zip(axs.flat, ("lambda0", "lambdaD", "lambda1"), ("λ = 0", "λ = 0.1", "λ = 1")):
        R = d[key]; s = R["snaps"][-1]
        show(ax, A(R["I"])); front(ax, A(s["phi"]), lw=1.6)
        ax.set_title(f"{lab}: {s['wrong']} wrong", fontsize=13)
        print(key, "wrong", s["wrong"], "thr", s["thr"])
    fig.tight_layout(pad=0.2, w_pad=0.5, h_pad=1.2); save(fig, "fig-lambda.png")

    # starts
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.95))
    for ax, key, lab in zip(axs, ("ring", "grid", "bad"), ("around the ring", "grid of blobs", "small circle in the hole")):
        R = d[key]; s0, s = R["snaps"][0], R["snaps"][-1]
        show(ax, A(R["I"]))
        if key == "grid":
            ax.contour(A(s0["phi"]), levels=[0], colors=["white"], linewidths=0.6, alpha=0.7)
        else:
            front(ax, A(s0["phi"]), color="white", lw=1.4, ls="--")
        if s["pieces"]:
            front(ax, A(s["phi"]), lw=1.6)
        res = f"{s['wrong']} wrong" if s["pieces"] else "vanished"
        ax.set_title(f"{lab}\nit. {s['it']}: {res}, E = {s['E']:.0f}", fontsize=12)
        print(key, "it", s["it"], "E", round(s["E"], 1), "wrong", s["wrong"], "pieces", s["pieces"])
    fig.tight_layout(pad=0.2, w_pad=0.5); save(fig, "fig-starts.png")


if __name__ == "__main__":
    fig_eight(); fig_means(); fig_metric()
    if len(sys.argv) > 1:
        figs_data(json.load(open(sys.argv[1])))
