"""Figures for polar-boundary/lec.qmd. Run from the repo root after data.ts (see there):
   python3 -I tools/figs/polar-boundary/figs.py <tmp>/polar-data.json
Figure size in inches = display size / 100 px, fonts >= 12 pt."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle
from scipy import ndimage as ndi
from skimage import measure

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "polar-boundary")
plt.rcParams.update({"font.size": 13, "svg.fonttype": "none", "font.family": "DejaVu Sans"})
ORANGE, BLUE, GREY, CYAN = "#ff7f0e", "#1f77b4", "#7f7f7f", "#17becf"
d = json.load(open(sys.argv[1]))
N = d["size"]
raw = np.array(d["raw"]).reshape(N, N)
smooth = np.array(d["smooth"]).reshape(N, N)
CX, CY = 100, 128


def show(ax, img, crop=(20, 230, 40, 220)):
    x0, x1, y0, y1 = crop
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")


def fig_threshold():
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.9))
    show(axs[0], raw); axs[0].set_title("image", fontsize=13)
    for ax, T in ((axs[1], 50), (axs[2], 125)):
        show(ax, raw)
        lab, _ = ndi.label(smooth > T)
        m = lab == lab[CY, CX]
        for c in measure.find_contours(m.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color=ORANGE, lw=2)
        ax.set_title(f"threshold {T}", fontsize=13)
    fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig-threshold.png"), dpi=200); plt.close(fig)


def fig_starshaped():
    fig, axs = plt.subplots(1, 2, figsize=(5.8, 2.9))
    # left: an ellipse, rays from its centre, one crossing each
    ax = axs[0]
    ax.add_patch(Ellipse((0, 0), 3.2, 2.2, angle=20, fc="#dddddd", ec="black", lw=2))
    th = np.linspace(0, 2 * np.pi, 13)[:-1]
    for t in th:
        ax.plot([0, 2.0 * np.cos(t)], [0, 2.0 * np.sin(t)], color=CYAN, lw=1.3)
    ax.add_patch(Circle((0, 0), 0.09, fc="white", ec="#d62728", lw=2, zorder=5))
    ax.set_title("star-shaped: one crossing per ray", fontsize=12)
    # right: a bean from a point in one lobe; one ray crosses three times
    ax = axs[1]
    t = np.linspace(0, 2 * np.pi, 400)
    x = 1.6 * np.cos(t); y = 1.0 * np.sin(t)
    bite = ((x / 0.85) ** 2 + ((y - 1.2) / 0.95) ** 2) < 1
    # draw bean as filled region via grid
    gx, gy = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-1.6, 1.6, 320))
    inside = ((gx / 1.6) ** 2 + (gy / 1.0) ** 2 < 1) & ~(((gx / 0.85) ** 2 + ((gy - 1.2) / 0.95) ** 2) < 1)
    ax.contourf(gx, gy, inside.astype(float), levels=[0.5, 1.5], colors=["#dddddd"])
    ax.contour(gx, gy, inside.astype(float), levels=[0.5], colors="black", linewidths=2)
    c = (-1.0, -0.1)
    for a in np.linspace(0, 2 * np.pi, 13)[:-1]:
        ax.plot([c[0], c[0] + 3.2 * np.cos(a)], [c[1], c[1] + 3.2 * np.sin(a)], color=CYAN, lw=1.0, alpha=0.6)
    a = np.deg2rad(22)
    ax.plot([c[0], c[0] + 3.2 * np.cos(a)], [c[1], c[1] + 3.2 * np.sin(a)], color=BLUE, lw=2.5)
    # crossings of that ray with the boundary
    s = np.linspace(0, 3.2, 3000)
    px, py = c[0] + s * np.cos(a), c[1] + s * np.sin(a)
    ins = ((px / 1.6) ** 2 + (py / 1.0) ** 2 < 1) & ~(((px / 0.85) ** 2 + ((py - 1.2) / 0.95) ** 2) < 1)
    idx = np.nonzero(np.diff(ins.astype(int)))[0]
    for i in idx:
        ax.plot(px[i], py[i], "o", ms=8, mfc=ORANGE, mec="black", zorder=6)
    ax.add_patch(Circle(c, 0.09, fc="white", ec="#d62728", lw=2, zorder=5))
    ax.set_title(f"not star-shaped: {len(idx)} crossings", fontsize=12)
    for ax in axs:
        ax.set_aspect("equal"); ax.set_xlim(-2.0, 2.0); ax.set_ylim(-1.5, 1.5); ax.axis("off")
    fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig-starshaped.svg")); plt.close(fig)
    return len(idx)


def fig_rayprofile():
    r = d["runs"]["done"]
    K, R = r["polarW"], r["polarH"]
    P = np.array(r["polar"]).reshape(R, K)
    Cst = np.array(r["cost"]).reshape(K, R)
    k = K // 4  # 90°: straight down on the screen
    prof = P[:, k]; cost = Cst[k]
    rb = r["path"][k]
    rr = np.arange(R)
    fig, axs = plt.subplots(2, 1, figsize=(4.8, 4.0), sharex=True)
    axs[0].plot(rr, prof, color="black", lw=2)
    axs[0].set_ylabel("intensity P")
    axs[1].plot(rr, cost, color=BLUE, lw=2)
    axs[1].axhline(0, color=GREY, lw=0.8)
    axs[1].set_ylabel("cost = ∂P/∂r")
    axs[1].set_xlabel("radius r along the ray (pixels)")
    for ax in axs:
        ax.axvline(rb, color=ORANGE, lw=2, ls="--")
        ax.spines[["top", "right"]].set_visible(False)
    axs[1].annotate(f"boundary, r = {rb}", xy=(rb, cost[rb]), xytext=(rb + 12, cost[rb] * 0.8), color=ORANGE,
                    fontsize=12, arrowprops=dict(arrowstyle="->", color=ORANGE))
    axs[0].set_yticks([0, 100, 200])
    axs[1].set_ylim(-1.1, 0.4)
    fig.align_ylabels(axs)
    fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig-rayprofile.svg")); plt.close(fig)
    return rb, cost[rb]


def fig_smoothness():
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.9))
    for ax, key, lab in ((axs[0], "done", "δ = 1, α = 0.1"), (axs[1], "nosmooth", "δ = 8, α = 0")):
        r = d["runs"][key]
        show(ax, raw, crop=(30, 215, 50, 210))
        c = np.array(r["contour"] + [r["contour"][0]])
        ax.plot(c[:, 0], c[:, 1], color=ORANGE, lw=2.2)
        ax.add_patch(Circle((CX, CY), 3, fc="white", ec="#d62728", lw=2))
        ax.set_title(f"{lab}\ncost {r['closedCost']:.2f}", fontsize=14)
    fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig-smoothness.png"), dpi=200); plt.close(fig)


def fig_closure():
    ro, re = d["runs"]["open"], d["runs"]["exact"]
    K, R = ro["polarW"], ro["polarH"]
    P = np.array(ro["polar"]).reshape(R, K)
    ang = np.arange(K) * 360 / K
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.imshow(P, cmap="gray", vmin=0, vmax=255, extent=(-1.5, 358.5, R - 0.5, -0.5), aspect="auto")
    ax.plot(ang, ro["path"], color=ORANGE, lw=2.5, label=f"open: gap {ro['gap']} px")
    ax.plot(ang, re["path"], color=CYAN, lw=1.8, ls=(0, (3, 2)), label=f"exact: gap {re['gap']} px")
    ax.annotate("", xy=(357, ro["path"][-1]), xytext=(357, ro["path"][0]),
                arrowprops=dict(arrowstyle="<->", color="#d62728", lw=2))
    ax.annotate("", xy=(1, ro["path"][-1]), xytext=(1, ro["path"][0]),
                arrowprops=dict(arrowstyle="<->", color="#d62728", lw=2))
    ax.set_xticks([0, 90, 180, 270, 360]); ax.set_xlim(-1.5, 358.5); ax.set_ylim(90, 15)
    ax.set_xlabel("angle θ (degrees)"); ax.set_ylabel("radius r (px)")
    ax.legend(loc="lower center", fontsize=12, framealpha=0.9)
    fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig-closure.png"), dpi=200); plt.close(fig)
    return ro["path"][0], ro["path"][-1]


def fig_amini():
    """Snake points, each with a 3x3 window of candidate positions; the chain links neighbours."""
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    n = 8
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.c_[3 * np.cos(t), 2.2 * np.sin(t)]
    tt = np.linspace(0, 2 * np.pi, 200)
    ax.plot(3.0 * np.cos(tt) + 0.25 * np.cos(3 * tt), 2.2 * np.sin(tt), color="#bbbbbb", lw=6, zorder=0)
    pc = np.r_[pts, pts[:1]]
    ax.plot(pc[:, 0], pc[:, 1], color="black", lw=1.5, zorder=2)
    h = 0.32
    for (x, y) in pts:
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                ax.plot(x + a * h, y + b * h, "o", ms=4, color=BLUE, zorder=3)
        ax.add_patch(Rectangle((x - 1.5 * h, y - 1.5 * h), 3 * h, 3 * h, fc="none", ec=BLUE, lw=1, zorder=3))
        ax.plot(x, y, "o", ms=8, mfc=ORANGE, mec="black", zorder=4)
    ax.text(0, 0.25, "each point: m = 9\ncandidate positions", ha="center", va="center", fontsize=12, color=BLUE)
    ax.text(0, -0.75, "neighbours linked:\na closed chain", ha="center", va="center", fontsize=12)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-3.7, 3.7); ax.set_ylim(-2.8, 2.8)
    fig.tight_layout(pad=0.1); fig.savefig(os.path.join(OUT, "fig-amini.svg")); plt.close(fig)


if __name__ == "__main__":
    fig_threshold()
    print("crossings", fig_starshaped())
    print("ray", fig_rayprofile())
    fig_smoothness()
    print("open ends", fig_closure())
    fig_amini()
