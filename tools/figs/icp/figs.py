"""Figures for icp/lec.qmd. The point sets and ICP runs come from the demo's own code
(tools/figs/icp/data.ts, see its header for how to make the JSON); the by-hand example of the
closed form (exercise A2) is computed here.
Run from the repository root:
    python3 -I tools/figs/icp/figs.py <tmp>/icp-data.json
Writes icp/fig-*.svg. Colours as in the icp demo: fixed set blue, moving set orange.
Coordinates as in images: x to the right, y down."""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "icp")
FONT = 17
plt.rcParams.update({
    "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
    "xtick.labelsize": FONT - 2, "ytick.labelsize": FONT - 2, "legend.fontsize": FONT - 2,
    "svg.fonttype": "none", "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8a9099"

with open(sys.argv[1]) as f:
    D = json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def frame(ax, lim=(-10, 460, -10, 410)):
    ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[3], lim[2])   # y down
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor("#f6f7f9")


def fixed(ax, F, s=40, outline=False):
    F = np.asarray(F)
    if outline:
        ax.plot(np.r_[F[:, 0], F[0, 0]], np.r_[F[:, 1], F[0, 1]], color=BLUE, lw=2, alpha=0.35, zorder=1)
        s = 14
    ax.scatter(F[:, 0], F[:, 1], s=s, color=BLUE, zorder=3)


def moving(ax, M, s=46, ghost=False):
    M = np.asarray(M)
    if ghost:
        ax.scatter(M[:, 0], M[:, 1], s=s * 1.4, facecolor="none", edgecolor=ORANGE, lw=1.2, alpha=0.45, zorder=2)
    else:
        ax.scatter(M[:, 0], M[:, 1], s=s, facecolor="white", edgecolor=ORANGE, lw=2.2, zorder=4)


def matches(ax, step, F):
    F = np.asarray(F)
    for p, j, k in zip(step["moved"], step["match"], step["kept"]):
        q = F[j]
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#444" if k else RED, lw=1.6 if k else 1.4,
                ls="-" if k else (0, (4, 3)), zorder=2)


def fig_problem():
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
    c, o = D["close"], D["outline"]
    frame(axs[0], (15, 385, 15, 385)); fixed(axs[0], c["F"]); moving(axs[0], c["steps"][0]["moved"])
    axs[0].set_title("spots: 40 + 40, no labels")
    frame(axs[1], (15, 385, 15, 385)); fixed(axs[1], o["F"], outline=True); moving(axs[1], o["steps"][0]["moved"], s=36)
    axs[1].set_title("a surface: 90 + 30 samples")
    save(fig, "fig-problem.svg")


def fig_pairs():
    P = np.array([[2, 0], [-2, 0], [0, 1], [0, -1]], float)
    Q = np.array([[11.6, 6.2], [8.4, 3.8], [9.4, 5.8], [10.6, 4.2]])
    pb, qb = P.mean(0), Q.mean(0)
    Pc, Qc = P - pb, Q - qb
    sdot = (Pc * Qc).sum(); scross = (Pc[:, 0] * Qc[:, 1] - Pc[:, 1] * Qc[:, 0]).sum()
    th = np.arctan2(scross, sdot)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    panels = [("known pairs", P, Q), ("1. centre both", Pc, Qc), (f"2. turn by atan2({scross:.0f}, {sdot:.0f}) = {np.degrees(th):.1f}°", Pc @ R.T, Qc)]
    for ax, (t, A, B) in zip(axs, panels):
        for a, b in zip(A, B):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=GREY, lw=1.3, ls=(0, (3, 3)), zorder=1)
        ax.plot(*np.r_[A[:2], [[np.nan, np.nan]], A[2:]].T, color=ORANGE, lw=2, alpha=0.6)
        ax.plot(*np.r_[B[:2], [[np.nan, np.nan]], B[2:]].T, color=BLUE, lw=2, alpha=0.6)
        ax.scatter(B[:, 0], B[:, 1], s=60, color=BLUE, zorder=3)
        ax.scatter(A[:, 0], A[:, 1], s=70, facecolor="white", edgecolor=ORANGE, lw=2.2, zorder=4)
        for k, (a, b) in enumerate(zip(A, B)):
            ax.annotate(str(k + 1), a, xytext=(-14, -12), textcoords="offset points", color=ORANGE, fontsize=FONT - 3)
            ax.annotate(str(k + 1), b, xytext=(6, 4), textcoords="offset points", color=BLUE, fontsize=FONT - 3)
        ax.set_title(t, fontsize=FONT - 1)
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.grid(color="#e3e3e3"); ax.set_axisbelow(True)
    axs[0].scatter(*pb, marker="+", s=160, color=ORANGE, lw=2.5); axs[0].scatter(*qb, marker="+", s=160, color=BLUE, lw=2.5)
    axs[0].annotate("p̄ = (0, 0)", pb, xytext=(8, -44), textcoords="offset points", color=ORANGE, fontsize=FONT - 3)
    axs[0].annotate("q̄ = (10, 5)", qb, xytext=(-110, -34), textcoords="offset points", color=BLUE, fontsize=FONT - 3)
    save(fig, "fig-pairs.svg")


def fig_steps():
    c = D["close"]
    st = c["steps"]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.5))
    for k, ax in enumerate(axs):
        s = st[k]
        frame(ax, (25, 375, 25, 375)); matches(ax, s, c["F"]); fixed(ax, c["F"], s=30); moving(ax, s["moved"], s=34)
        last = k == len(st) - 1
        ax.set_title(("start" if k == 0 else f"after move {k}") + f"\nRMS {s['rms']:.2f} px" + (" (no change)" if last else ""), fontsize=FONT - 1)
    save(fig, "fig-steps.svg")


def fig_basin():
    f, sw = D["far"], D["sweep"]
    fig = plt.figure(figsize=(13, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.45])
    ax = fig.add_subplot(gs[0])
    s = f["steps"][-1]
    frame(ax, (0, 400, 0, 400)); matches(ax, s, f["F"]); fixed(ax, f["F"], s=30); moving(ax, f["truth"], s=34, ghost=True); moving(ax, s["moved"], s=34)
    ax.set_title(f"turned 40°: stuck at {s['angle']:.1f}°\nRMS {s['rms']:.1f} px, true error {s['err']:.0f} px", fontsize=FONT - 1)
    ax2 = fig.add_subplot(gs[1])
    a = np.array(sw["angles"])
    ax2.axhline(5, color=GREY, ls=(0, (4, 4)), lw=1.3)
    ax2.semilogy(a, np.clip(sw["spots"], 0.1, 400), "o-", color=BLUE, ms=4, lw=1.2, label="spots")
    ax2.semilogy(a, np.clip(sw["outline"], 0.1, 400), "s-", color=GREEN, ms=4, lw=1.2, label="outline")
    ax2.set_xticks([-180, -90, 0, 90, 180]); ax2.set_ylim(0.1, 400)
    ax2.set_xlabel("rotation between the sets (°)"); ax2.set_ylabel("final true error (px)")
    ax2.legend(loc="lower right", frameon=True)
    ax2.text(-175, 6.5, "5 px", color=GREY, fontsize=FONT - 3)
    save(fig, "fig-basin.svg")


def fig_overlap():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.9))
    for ax, key, t in [(axs[0], "overlap", "all pairs"), (axs[1], "trimmed", "trimmed: closest 70 %")]:
        d = D[key]; s = d["steps"][-1]
        frame(ax, (40, 470, 20, 380)); matches(ax, s, d["F"]); fixed(ax, d["F"], s=30); moving(ax, s["moved"], s=34)
        ax.set_title(f"{t}: true error {s['err']:.2f} px", fontsize=FONT - 1)
    save(fig, "fig-overlap.svg")


def fig_plane():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axs[0]
    # a gently curved surface sampled at x = 0, 2, 4, ...; a moving point between two samples
    xs = np.linspace(-1, 9, 200); f = lambda x: 0.06 * (x - 4) ** 2
    sx = np.arange(0, 9, 2.0)
    ax.plot(xs, f(xs), color=BLUE, lw=2, alpha=0.4)
    ax.scatter(sx, f(sx), s=60, color=BLUE, zorder=3)
    p = np.array([7.0, -0.9])
    j = np.argmin(np.hypot(sx - p[0], f(sx) - p[1])); q = np.array([sx[j], f(sx[j])])
    # tangent at the nearest sample, and the foot of the perpendicular
    t = np.array([1, 0.12 * (q[0] - 4)]); t /= np.hypot(*t); n = np.array([-t[1], t[0]])
    foot = p - np.dot(p - q, n) * n
    ax.plot([q[0] - 2.4 * t[0], q[0] + 2.4 * t[0]], [q[1] - 2.4 * t[1], q[1] + 2.4 * t[1]], color=BLUE, ls=(0, (4, 3)), lw=1.4)
    ax.annotate("", q, p, arrowprops=dict(arrowstyle="->", color=RED, lw=2.2))
    ax.annotate("", foot, p, arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.2))
    ax.scatter(*p, s=80, facecolor="white", edgecolor=ORANGE, lw=2.4, zorder=4)
    ax.text(p[0] - 4.6, p[1] + 0.2, "point to point", color=RED, fontsize=FONT - 2)
    ax.text(p[0] + 0.5, p[1] - 0.5, "point to plane", color=GREEN, fontsize=FONT - 2)
    ax.text(2.6, 1.05, "tangent", color=BLUE, fontsize=FONT - 3)
    ax.set_xlim(-1, 9.5); ax.set_ylim(2.2, -2.4); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("two distances to a sampled surface", fontsize=FONT - 1)
    # the outline run: the moving samples snap onto the fixed samples
    o = D["outline"]; s = o["steps"][-1]
    ax = axs[1]
    frame(ax, (170, 365, 50, 215)); fixed(ax, o["F"], outline=True)
    moving(ax, o["truth"], s=60, ghost=True); moving(ax, s["moved"], s=60)
    ax.set_title(f"outline, point to point: {s['angle']:.1f}° instead of 30°,\ntrue error {s['err']:.1f} px (rings: true positions)", fontsize=FONT - 2)
    save(fig, "fig-plane.svg")


fig_problem()
fig_pairs()
fig_steps()
fig_basin()
fig_overlap()
fig_plane()
