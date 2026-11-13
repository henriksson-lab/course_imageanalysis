"""Figures for levelset-eq/lec.qmd. Run from the repo root (see data.ts for the JSON):
   python3 -I tools/figs/levelset-eq/figs.py <tmp>/lse-data.json
The 1D figures use the upwind rule of demos/src/lib/levelset.ts (upwindGrad) on the grid
x = -3..3 of the slides; the 2D ones are snapshots from the curveflow demo's code (data.ts).
SVG sizes: 1 pt in the figure = 1 px on the slide at the width given in lec.qmd."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "levelset-eq")
plt.rcParams.update({"font.size": 17, "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, ORANGE, SKY, PURPLE, GREY = "#0072b2", "#d55e00", "#009e73", "#e69f00", "#56b4e9", "#cc79a7", "#777777"
X = np.arange(-3, 4)
V = np.abs(X) - 1.0


def upwind(phi, F):
    """1D |phi_x| as in lib/levelset.ts: one-sided differences, 0 across the border."""
    n = len(phi)
    dm = np.r_[0, phi[1:] - phi[:-1]]
    dp = np.r_[phi[1:] - phi[:-1], 0]
    if F > 0:
        return np.sqrt(np.maximum(dm, 0) ** 2 + np.minimum(dp, 0) ** 2)
    return np.sqrt(np.minimum(dm, 0) ** 2 + np.maximum(dp, 0) ** 2)


def central(phi):
    p = np.r_[phi[0], phi, phi[-1]]
    return np.abs(p[2:] - p[:-2]) / 2


def step(phi, F, dt, scheme="upwind"):
    g = upwind(phi, F) if scheme == "upwind" else central(phi)
    return phi - dt * F * g


def axes1d(ax, ylim=(-2.3, 2.4)):
    ax.axhline(0, color="#999", lw=1)
    ax.set_xticks(X); ax.set_xlabel("x"); ax.set_ylabel("φ")
    ax.set_ylim(*ylim); ax.set_xlim(-3.4, 3.4)
    ax.axhspan(ylim[0], 0, color=SKY, alpha=0.10, lw=0)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), transparent=False, facecolor="white"); plt.close(fig)
    print("wrote", name)


def crossings(phi):
    """Zero crossings by linear interpolation between grid points."""
    out = []
    for i in range(len(phi) - 1):
        a, b = phi[i], phi[i + 1]
        if (a < 0) != (b < 0):
            out.append(X[i] + a / (a - b))
    return out


# ------------------------------------------------------------------ the idea: drop by F × slope
def fig_slope():
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    x = np.linspace(0, 2.6, 100)
    drop = 1.0  # F Δt
    xa = 2.6
    for s, col, lab, dx in ((2.0, BLUE, "steep", 0.0), (0.5, RED, "shallow", 0.0)):
        y = s * (x - 1)
        ax.plot(x, y, color=col, lw=2.5, label=f"{lab}: slope {s:g}")
        ax.plot(x, y - s * drop, color=col, lw=2.5, ls="--")
        ax.plot([1], [0], "o", color=col, ms=8)
        ax.plot([1 + drop], [0], "o", color=col, ms=9, mfc="white", mew=2, zorder=5)
        y0 = s * (xa - 1)
        ax.annotate("", xy=(xa + dx, y0 - s * drop), xytext=(xa + dx, y0),
                    arrowprops=dict(arrowstyle="->", lw=2, color=col, shrinkA=0, shrinkB=0))
        ax.text(xa + 0.1, y0 - s * drop / 2, f"drop {s * drop:g}", color=col, fontsize=16, va="center")
    ax.annotate("", xy=(2.0, -0.3), xytext=(1.0, -0.3), arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.text(1.5, -0.55, "both crossings\nmove F·Δt = 1", ha="center", va="top", fontsize=16)
    ax.axhline(0, color="#999", lw=1)
    ax.set_xlim(0, 3.4); ax.set_ylim(-2.2, 3.6)
    ax.set_xlabel("x"); ax.set_ylabel("φ")
    ax.legend(loc="upper left", frameon=False, fontsize=16, handlelength=1.5)
    fig.tight_layout(pad=0.3); save(fig, "fig-slope.svg")


# ------------------------------------------------------------------ 1D: one step, Δt = 0.5
def fig_v1d():
    after = step(V, 1, 0.5)
    assert np.allclose(after, [1.5, 0.5, -0.5, -1, -0.5, 0.5, 1.5]), after
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    axes1d(ax)
    ax.plot(X, V, "-o", color=GREY, lw=2, ms=7, label="before")
    ax.plot(X, after, "-o", color=BLUE, lw=2.5, ms=7, label="after one step")
    for c in crossings(V):
        ax.plot([c], [0], "o", color=GREY, ms=10, mfc="white", mew=2)
    for c in crossings(after):
        ax.plot([c], [0], "o", color=BLUE, ms=10, mfc="white", mew=2)
    ax.annotate("bottom stays", xy=(0, -1.05), xytext=(0, -2.05), ha="center", fontsize=16,
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(1.65, -0.3, "±1.5", color=BLUE, fontsize=16, ha="left", va="top")
    ax.legend(loc="upper center", frameon=False, fontsize=16, ncol=2, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(pad=0.3); save(fig, "fig-v1d.svg")


# ------------------------------------------------------------------ upwind: which neighbour
def fig_upwind():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    axes1d(ax, ylim=(-2.3, 2.6))
    ax.plot(X, V, "-o", color=GREY, lw=2, ms=8, zorder=3)
    off = np.array([-0.18, 0.32])  # arrows drawn above-left / above-right of the line
    for i, x in enumerate(X):
        if x == 0:
            continue
        j = i + 1 if x < 0 else i - 1  # the lower neighbour: where a growing front comes from
        o = off * (1 if x < 0 else np.array([-1, 1]))
        ax.annotate("", xy=(X[i] + o[0], V[i] + o[1]), xytext=(X[j] + o[0], V[j] + o[1]),
                    arrowprops=dict(arrowstyle="->", lw=2.2, color=GREEN, shrinkA=6, shrinkB=6))
    ax.plot([0], [-1], "o", ms=13, mfc="white", mec=RED, mew=2.5, zorder=4)
    ax.text(0, -1.35, "no lower neighbour:\nslope 0, value stays", color=RED, ha="center", va="top", fontsize=16)
    ax.text(0, 2.45, "F > 0: each value is\nupdated from its\nlower neighbour", color=GREEN, ha="center", va="top", fontsize=16)
    fig.tight_layout(pad=0.3); save(fig, "fig-upwind.svg")


# ------------------------------------------------------------------ central differences stall
def fig_central():
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, scheme, title in zip(axs, ("upwind", "central"), ("upwind", "central differences")):
        axes1d(ax, ylim=(-1.4, 3.3))
        phi = V.copy()
        cols = [GREY, SKY, BLUE]
        ax.plot(X, phi, "-o", color=cols[0], lw=2, ms=6, label="start")
        for k in (1, 2):
            phi = step(phi, -1, 0.5, scheme)
            ax.plot(X, phi, "-o", color=cols[k], lw=2.5, ms=6, label=f"after {k} step{'s' if k > 1 else ''}")
        print(scheme, "after 2 steps:", np.round(phi, 3))
        ax.set_title(title, fontsize=17)
        ax.annotate(f"φ(0) = {phi[3]:.2f}".replace("-", "−"), xy=(0, phi[3]), xytext=(0.7, -1.0),
                    fontsize=16, color=BLUE, va="center", arrowprops=dict(arrowstyle="->", lw=1.5, color=BLUE))
        if scheme == "central":
            ax.set_ylabel("")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", frameon=False, fontsize=16, ncol=3)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9)); save(fig, "fig-central.svg")


# ------------------------------------------------------------------ CFL in 1D
def fig_cfl():
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    axes1d(ax, ylim=(-3.4, 4.4))
    ax.set_yticks([-2, 0, 2])
    big = step(V, 1, 2.0)
    print("dt=2:", big)
    ax.plot(X, V, "-o", color=GREY, lw=2, ms=6, label="start")
    ax.plot(X, np.abs(X) - 3, ":", color=BLUE, lw=2, label="exact at t = 2")
    ax.plot(X, big, "-o", color=RED, lw=2.5, ms=7, label="one step, Δt = 2")
    ax.legend(loc="upper center", frameon=False, fontsize=15, borderaxespad=0.1)
    fig.tight_layout(pad=0.3); save(fig, "fig-cfl1d.svg")


# ------------------------------------------------------------------ 2D snapshots
def contour_panel(ax, runs, N, cmap="viridis", fill_first=True, label_fmt="t = {:.0f}", lw=2.4, legend=True, ylim=None):
    cols = plt.get_cmap(cmap)(np.linspace(0.0, 0.85, len(runs)))
    ax.set_xlim(0, N - 1); ax.set_ylim(*(ylim or (N - 1, 0))); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#bbb")
    handles = []
    for r, c in zip(runs, cols):
        phi = np.array(r["phi"]).reshape(N, N)
        if fill_first and r is runs[0]:
            ax.contourf(phi, levels=[-1e9, 0], colors=[SKY], alpha=0.35)
        cs = ax.contour(phi, levels=[0], colors=[c], linewidths=lw)
        handles.append(plt.Line2D([], [], color=c, lw=3, label=label_fmt.format(r["t"])))
    if legend:
        ax.legend(handles=handles, loc="upper center", frameon=False, fontsize=13, ncol=2,
                  bbox_to_anchor=(0.5, -0.01), handlelength=1.0, columnspacing=0.8, handletextpad=0.4)


def fig_topology(d):
    N = d["N"]
    plt.rcParams["font.size"] = 13
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2))
    m = [d["merge"][i] for i in (0, 1, 3, 4)]
    Y = (100, 28)
    contour_panel(axs[0], m, N, label_fmt="t = {:.1f}", ylim=Y)
    axs[0].set_title("F = 1: two blobs merge", fontsize=14)
    contour_panel(axs[1], d["split"], N, label_fmt="t = {:.1f}", ylim=Y)
    axs[1].set_title("F = −1: the dumbbell splits", fontsize=14)
    contour_panel(axs[2], d["nosplit"], N, ylim=Y)
    axs[2].set_title("curvature flow: no split", fontsize=14)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.22, wspace=0.06); fig.savefig(os.path.join(OUT, "fig-topology.png"), dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.04); plt.close(fig)
    print("wrote fig-topology.png; merge pieces", [r["pieces"] for r in d["merge"]], "split", [r["pieces"] for r in d["split"]])
    plt.rcParams["font.size"] = 17


def fig_speeds(d):
    N = d["N"]
    plt.rcParams["font.size"] = 13
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 3.3))
    contour_panel(axs[0], d["square"], N, label_fmt="t = {:.0f}")
    axs[0].set_title("constant F = 1", fontsize=14)
    contour_panel(axs[1], d["star"], N)
    axs[1].set_title("curvature flow, ε = 1", fontsize=14)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.15, wspace=0.06); fig.savefig(os.path.join(OUT, "fig-speeds.png"), dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.04); plt.close(fig)
    print("wrote fig-speeds.png")
    plt.rcParams["font.size"] = 17


def fig_reinit(d):
    N = d["N"]
    plt.rcParams["font.size"] = 13
    fig, axs = plt.subplots(2, 1, figsize=(3.0, 5.6))
    for ax, r, title in zip(axs, (d["star"][2], d["starReinit"][0]), ("never reinitialised", "reinitialised every 10 steps")):
        phi = np.array(r["phi"]).reshape(N, N)
        m = 40
        ax.imshow(np.clip(phi, -m, m), cmap="RdBu_r", vmin=-m, vmax=m, interpolation="nearest")
        ax.contour(phi, levels=np.arange(-40, 41, 4), colors="#555", linewidths=0.8)
        ax.contour(phi, levels=[0], colors="black", linewidths=2.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}\nmean |∇φ| at the front: {r['grad']:.2f}", fontsize=11.5)
    fig.tight_layout(pad=0.3, h_pad=0.6); fig.savefig(os.path.join(OUT, "fig-reinit.png"), dpi=200, facecolor="white"); plt.close(fig)
    print("wrote fig-reinit.png", d["star"][2]["grad"], d["starReinit"][0]["grad"])
    plt.rcParams["font.size"] = 17


if __name__ == "__main__":
    fig_slope(); fig_v1d(); fig_upwind(); fig_central(); fig_cfl()
    if len(sys.argv) > 1:
        d = json.load(open(sys.argv[1]))
        fig_topology(d); fig_speeds(d); fig_reinit(d)
