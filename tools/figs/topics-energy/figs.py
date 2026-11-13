"""Static figures for the topic page topics/energy.qmd.

Run from the repository root:  python3 -I tools/figs/topics-energy/figs.py
Writes topics/fig-en-profile.svg, fig-en-lambda.svg, fig-en-landscape.svg, fig-en-icm-cut.png,
fig-en-cut-lambda.png, and prints every number quoted on the page.
All data are synthetic with fixed seeds. Exact minima: dynamic programming (1D labels), a
minimum s-t cut (2D labels, networkx), linear solves (quadratic) and ADMM run to convergence
(total variation, checked against the optimality gap).
Optional: PREVIEW=<dir> also writes small PNG copies there for checking by eye.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import solve_banded

OUT = "topics"
FONT = 15
plt.rcParams.update({
    "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
    "xtick.labelsize": FONT - 3, "ytick.labelsize": FONT - 3, "legend.fontsize": FONT - 3,
    "svg.fonttype": "path", "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white",
})
BLUE, ORANGE, GREEN, RED, PURPLE, CYAN, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#777777")


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.05, **kw)
    if os.environ.get("PREVIEW"):
        fig.savefig(os.path.join(os.environ["PREVIEW"], name.rsplit(".", 1)[0] + ".png"), dpi=70,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# 1. A 1D labelling: threshold vs the exact minimum (dynamic programming) -----------------------
MU0, MU1 = 0.2, 0.6


def dp_labels(y, lam):
    """Exact minimum of sum (y - mu_l)^2 + lam * #(label changes), by dynamic programming."""
    n = len(y)
    D = np.stack([(y - MU0) ** 2, (y - MU1) ** 2], 1)
    C = np.zeros((n, 2)); B = np.zeros((n, 2), int)
    C[0] = D[0]
    for i in range(1, n):
        for l in range(2):
            stay, switch = C[i - 1, l], C[i - 1, 1 - l] + lam
            B[i, l] = l if stay <= switch else 1 - l
            C[i, l] = D[i, l] + min(stay, switch)
    lab = np.zeros(n, int); lab[-1] = int(np.argmin(C[-1]))
    for i in range(n - 1, 0, -1):
        lab[i - 1] = B[i, lab[i]]
    return lab, C[-1].min()


def energy1d(y, lab, lam):
    mu = np.where(lab == 1, MU1, MU0)
    return ((y - mu) ** 2).sum(), lam * np.sum(lab[1:] != lab[:-1])


def fig_profile():
    rng = np.random.default_rng(3)
    n = 80
    truth = np.zeros(n, int); truth[13:51] = 1; truth[65:67] = 1
    y = np.where(truth == 1, MU1, MU0) + 0.12 * rng.standard_normal(n)
    lams = [0.0, 0.1, 0.4]
    fig = plt.figure(figsize=(12, 5.6))
    ax = fig.add_axes([0.08, 0.45, 0.9, 0.5])
    ax.plot(np.arange(n), y, "o", color=GREY, ms=4)
    ax.plot(np.arange(n), y, "-", color=GREY, lw=0.8)
    ax.axhline(MU0, color=BLUE, ls="--", lw=1.2); ax.axhline(MU1, color=ORANGE, ls="--", lw=1.2)
    ax.axhline((MU0 + MU1) / 2, color="k", ls=":", lw=1)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylabel("intensity")
    ax.set_xticks([])
    box = dict(fc="white", ec="none", alpha=0.85, pad=1)
    ax.text(n - 1, MU1 + 0.03, "object level", color=ORANGE, ha="right", fontsize=FONT - 3, bbox=box)
    ax.text(n - 1, MU0 - 0.06, "background level", color=BLUE, ha="right", va="top", fontsize=FONT - 3, bbox=box)
    ax.text(68, 0.78, "thin structure", ha="left", fontsize=FONT - 3, bbox=box)
    rows = [("truth", truth)]
    for lam in lams:
        lab, e = dp_labels(y, lam)
        d, s = energy1d(y, lab, lam)
        nb = int(np.sum(lab[1:] != lab[:-1]))
        wrong = int(np.sum(lab != truth))
        name = "threshold (λ = 0)" if lam == 0 else f"minimum, λ = {lam}"
        rows.append((name, lab))
        print(f"[profile] lam={lam}: wrong {wrong}, boundaries {nb}, thin kept {bool(lab[65:67].all())}, "
              f"energy data {d:.3f} + smooth {s:.3f} = {e:.3f}")
    # where is the thin structure lost? scan lambda
    lost = None
    clean = None
    for lam in np.arange(0, 1.0, 0.005):
        lab, _ = dp_labels(y, lam)
        if clean is None and np.sum(lab[:60] != truth[:60]) == 0:
            clean = lam
        if lost is None and not lab[65:67].any():
            lost = lam
    print(f"[profile] specks in samples 0-59 gone from lam={clean:.3f}; thin structure lost from lam={lost:.3f}")
    for r, (name, lab) in enumerate(rows):
        a = fig.add_axes([0.08, 0.33 - r * 0.08, 0.9, 0.055])
        a.imshow(lab[None, :], cmap=matplotlib.colors.ListedColormap([BLUE, ORANGE]), vmin=0, vmax=1,
                 aspect="auto", extent=(-0.5, n - 0.5, 0, 1))
        bad = np.nonzero(lab != truth)[0]
        a.plot(bad, np.full(len(bad), 0.5), "k.", ms=5)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        a.text(-1.5, 0.5, name, ha="right", va="center", fontsize=FONT - 3)
    fig.axes[0].set_position([0.22, 0.45, 0.76, 0.5])
    for a in fig.axes[1:]:
        p = a.get_position(); a.set_position([0.22, p.y0, 0.76, p.height])
    save(fig, "fig-en-profile.svg")


# 2. lambda in 1D denoising: quadratic vs TV ------------------------------------------------------
def diffmat(n):
    D = np.zeros((n - 1, n))
    D[np.arange(n - 1), np.arange(n - 1)] = -1
    D[np.arange(n - 1), np.arange(1, n)] = 1
    return D


def quad_smooth(y, lam):
    n = len(y)
    ab = np.zeros((3, n))
    main = np.full(n, 2.0); main[0] = main[-1] = 1.0
    ab[0, 1:] = -lam; ab[1] = 1 + lam * main; ab[2, :-1] = -lam
    return solve_banded((1, 1), ab, y)


def tv_denoise(y, lam, rho=1.0, iters=20000):
    n = len(y)
    D = diffmat(n)
    A = np.eye(n) + rho * D.T @ D
    Ainv = np.linalg.inv(A)
    z = np.zeros(n - 1); u = np.zeros(n - 1)
    for _ in range(iters):
        x = Ainv @ (y + rho * D.T @ (z - u))
        Dx = D @ x
        v = Dx + u
        z = np.sign(v) * np.maximum(np.abs(v) - lam / rho, 0)
        u += Dx - z
    return x


def tv_energy(x, y, lam):
    return 0.5 * np.sum((x - y) ** 2) + lam * np.sum(np.abs(np.diff(x)))


def fig_lambda():
    rng = np.random.default_rng(5)
    n = 200
    t = np.zeros(n)
    t[40:] = 2.0
    t[80:140] = np.linspace(2, 0, 60)
    t[140:] = 0
    t[166:170] = 2.0
    y = t + 0.3 * rng.standard_normal(n)
    fig, axs = plt.subplots(2, 3, figsize=(14, 6.2), sharex=True, sharey=True)
    print(f"[lambda] noisy data RMSE {np.sqrt(np.mean((y - t) ** 2)):.3f}")
    for c, lam in enumerate([2, 20, 200]):
        x = quad_smooth(y, lam)
        rm = np.sqrt(np.mean((x - t) ** 2))
        ax = axs[0, c]
        ax.plot(y, ".", color=GREY, ms=3); ax.plot(t, "--", color="k", lw=1); ax.plot(x, color=BLUE, lw=2)
        ax.set_title(f"quadratic, λ = {lam}   (RMSE {rm:.2f})", fontsize=FONT - 2)
        print(f"[lambda] quadratic lam={lam}: RMSE {rm:.3f}, peak max {x[160:176].max():.2f}")
    best_q = min((np.sqrt(np.mean((quad_smooth(y, l) - t) ** 2)), l) for l in np.geomspace(0.1, 1000, 400))
    for c, lam in enumerate([0.3, 1.5, 8]):
        x = tv_denoise(y, lam)
        rm = np.sqrt(np.mean((x - t) ** 2))
        # optimality check: compare with small perturbations
        e0 = tv_energy(x, y, lam)
        worst = min(tv_energy(x + 1e-3 * rng.standard_normal(n), y, lam) - e0 for _ in range(50))
        ax = axs[1, c]
        ax.plot(y, ".", color=GREY, ms=3); ax.plot(t, "--", color="k", lw=1); ax.plot(x, color=ORANGE, lw=2)
        ax.set_title(f"total variation, λ = {lam}   (RMSE {rm:.2f})", fontsize=FONT - 2)
        print(f"[lambda] TV lam={lam}: RMSE {rm:.3f}, peak max {x[160:176].max():.2f}, "
              f"step 38->42 {x[38]:.2f}->{x[42]:.2f}, min energy increase on perturbation {worst:.2e}")
    best_t = min((np.sqrt(np.mean((tv_denoise(y, l, iters=4000) - t) ** 2)), l) for l in np.geomspace(0.05, 10, 40))
    print(f"[lambda] best quadratic RMSE {best_q[0]:.3f} at lam {best_q[1]:.1f}; best TV RMSE {best_t[0]:.3f} at lam {best_t[1]:.2f}")
    axs[0, 0].set_ylabel("quadratic prior"); axs[1, 0].set_ylabel("TV prior")
    for ax in axs[1]:
        ax.set_xlabel("sample")
    axs[0, 0].text(0.02, 0.95, "λ small: noise stays", transform=axs[0, 0].transAxes, va="top", fontsize=FONT - 3)
    axs[0, 2].text(0.98, 0.95, "λ large: detail goes", transform=axs[0, 2].transAxes, va="top", ha="right", fontsize=FONT - 3)
    fig.tight_layout()
    save(fig, "fig-en-lambda.svg")


# 3. Landscapes: a convex bowl and a two-valley energy ---------------------------------------------
def fig_landscape():
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    # convex: two samples y = (1, -1), quadratic smoothing with lam = 2
    y = np.array([1.0, -1.0]); lam = 2.0
    E = lambda a, b: (a - y[0]) ** 2 + (b - y[1]) ** 2 + lam * (a - b) ** 2
    g = np.linspace(-2, 2, 200)
    A, Bv = np.meshgrid(g, g)
    ax = axs[0]
    ax.contour(A, Bv, E(A, Bv), levels=np.geomspace(0.6, 30, 12), colors=GREY, linewidths=1)
    xstar = np.linalg.solve(np.array([[1 + lam, -lam], [-lam, 1 + lam]]), y)
    for start, col in [((-1.8, 1.6), BLUE), ((1.8, -1.9), ORANGE), ((1.7, 1.7), GREEN)]:
        p = np.array(start, float); path = [p.copy()]
        for _ in range(60):
            gr = np.array([2 * (p[0] - y[0]) + 2 * lam * (p[0] - p[1]), 2 * (p[1] - y[1]) - 2 * lam * (p[0] - p[1])])
            p = p - 0.05 * gr; path.append(p.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "-", color=col, lw=2)
        ax.plot(*start, "o", color=col, ms=7)
    ax.plot(*xstar, "k*", ms=16)
    ax.set_aspect("equal"); ax.set_xlabel("value of sample 1"); ax.set_ylabel("value of sample 2")
    ax.set_title("Convex: every start reaches the same minimum")
    print(f"[landscape] convex: minimum at {xstar}, energy {E(*xstar):.3f}")
    # non-convex: a circle of radius r on a two-ring image; energy = -edge strength + small shrink
    r = np.linspace(0, 40, 801)
    prof = np.where(r < 10, 1.0, np.where(r < 25, 0.45, 0.0))
    prof = gaussian_filter1d(prof, 2.0 / (r[1] - r[0]), mode="nearest")
    edge = np.abs(np.gradient(prof, r))
    alpha = 0.002
    En = -edge + alpha * r
    ax = axs[1]
    ax.plot(r, En, color="k", lw=2)
    dE = np.gradient(En, r)
    ends = []
    for r0, col in [(6.0, BLUE), (14.0, ORANGE), (21.0, GREEN), (33.0, PURPLE)]:
        rr = r0; traj = [rr]
        for _ in range(4000):
            rr = rr - 40.0 * np.interp(rr, r, dE); traj.append(rr)
        traj = np.array(traj)
        ax.plot(r0, np.interp(r0, r, En), "o", color=col, ms=8)
        ax.annotate("", xy=(traj[-1], np.interp(traj[-1], r, En) - 0.005), xytext=(r0, np.interp(r0, r, En) - 0.005),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2, connectionstyle="arc3,rad=0.25"))
        ends.append((r0, traj[-1]))
    ax.set_xlabel("radius of a circular contour (pixels)"); ax.set_ylabel("energy")
    ax.set_yticks([])
    ax.text(10, En.min() - 0.02, "nucleus\nedge", ha="center", va="top", fontsize=FONT - 3)
    ax.text(25, np.interp(25, r, En) - 0.02, "cell\nedge", ha="center", va="top", fontsize=FONT - 3)
    ax.set_ylim(En.min() - 0.08, En.max() + 0.03)
    ax.set_title("Not convex: the start decides")
    fig.tight_layout()
    save(fig, "fig-en-landscape.svg")
    print(f"[landscape] two-ring descent (start -> end radius): {[(a, round(float(b), 1)) for a, b in ends]}")
    print(f"[landscape] energy at r=10: {np.interp(10, r, En):.4f}, at r=25: {np.interp(25, r, En):.4f}")


# 4. Binary denoising: ICM vs the exact minimum cut -------------------------------------------------
def test_image(h=64, w=64):
    yy, xx = np.mgrid[0:h, 0:w]
    x = np.zeros((h, w), int)
    x[(yy - 22) ** 2 + (xx - 20) ** 2 < 12 ** 2] = 1
    x[38:56, 34:58] = 1
    x[6:58, 46:48] = 1          # a thin bar, 2 pixels wide
    x[44:50, 8:14] = 1          # a small square
    return x


def energy2d(lab, obs, beta, lam):
    data = beta * np.sum(lab != obs)
    sm = lam * (np.sum(lab[1:, :] != lab[:-1, :]) + np.sum(lab[:, 1:] != lab[:, :-1]))
    return data + sm


def graphcut(obs, beta, lam):
    h, w = obs.shape
    G = nx.DiGraph()
    for i in range(h):
        for j in range(w):
            p = (i, j)
            # cost of label 1 = beta*[obs==0]; cost of label 0 = beta*[obs==1]
            # source side = label 1: edge p->t carries the cost of label 1, s->p the cost of label 0
            G.add_edge("s", p, capacity=beta * (obs[i, j] == 1))
            G.add_edge(p, "t", capacity=beta * (obs[i, j] == 0))
            for di, dj in ((1, 0), (0, 1)):
                q = (i + di, j + dj)
                if q[0] < h and q[1] < w:
                    G.add_edge(p, q, capacity=lam); G.add_edge(q, p, capacity=lam)
    cut, (S, T) = nx.minimum_cut(G, "s", "t")
    lab = np.zeros((h, w), int)
    for v in S:
        if v != "s":
            lab[v] = 1
    return lab, cut


def icm(obs, beta, lam, sweeps=20):
    lab = obs.copy()
    h, w = obs.shape
    for s in range(sweeps):
        changed = 0
        for i in range(h):
            for j in range(w):
                best = None
                for l in (0, 1):
                    e = beta * (l != obs[i, j])
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        a, b = i + di, j + dj
                        if 0 <= a < h and 0 <= b < w:
                            e += lam * (l != lab[a, b])
                    if best is None or e < best[0]:
                        best = (e, l)
                if best[1] != lab[i, j]:
                    lab[i, j] = best[1]; changed += 1
        if changed == 0:
            break
    return lab, s + 1


def fig_icm_cut():
    rng = np.random.default_rng(11)
    truth = test_image()
    flip = rng.random(truth.shape) < 0.2
    obs = np.where(flip, 1 - truth, truth)
    beta = np.log(0.8 / 0.2)
    lam = 1.0
    li, sweeps = icm(obs, beta, lam)
    lg, cut = graphcut(obs, beta, lam)
    res = [("truth", truth, None), ("noisy (20 % flipped)", obs, None),
           ("ICM: pixel by pixel", li, energy2d(li, obs, beta, lam)),
           ("minimum cut: exact", lg, energy2d(lg, obs, beta, lam))]
    fig, axs = plt.subplots(1, 4, figsize=(14, 4.0))
    for ax, (name, im, e) in zip(axs, res):
        ax.imshow(im, cmap="gray", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        err = 100 * np.mean(im != truth)
        sub = f"\nenergy {e:.0f}, wrong {err:.1f} %" if e is not None else (f"\nwrong {err:.1f} %" if name != "truth" else "\n ")
        ax.set_title(name + sub, fontsize=FONT - 2)
    fig.tight_layout()
    save(fig, "fig-en-icm-cut.png", dpi=110)
    print(f"[icm-cut] beta {beta:.3f}, lam {lam}; noisy wrong {100 * np.mean(obs != truth):.1f}%, energy {energy2d(obs, obs, beta, lam):.1f}; "
          f"ICM sweeps {sweeps}, energy {energy2d(li, obs, beta, lam):.1f}, wrong {100 * np.mean(li != truth):.1f}%; "
          f"cut {cut:.1f}, energy {energy2d(lg, obs, beta, lam):.1f}, wrong {100 * np.mean(lg != truth):.1f}%; "
          f"truth energy {energy2d(truth, obs, beta, lam):.1f}")
    # lambda series with the exact minimum
    lams = [0.3, 1.0, 2.0, 5.0]
    fig, axs = plt.subplots(1, 4, figsize=(14, 4.0))
    for ax, l in zip(axs, lams):
        lab, _ = graphcut(obs, beta, l)
        err = 100 * np.mean(lab != truth)
        bar = 100 * np.mean(lab[10:54, 46:48])
        sq = 100 * np.mean(lab[44:50, 8:14])
        ax.imshow(lab, cmap="gray", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"λ = {l}\nwrong {err:.1f} %", fontsize=FONT - 2)
        print(f"[cut-lambda] lam={l}: wrong {err:.2f}%, thin bar kept {bar:.0f}%, small square kept {sq:.0f}%")
    fig.tight_layout()
    save(fig, "fig-en-cut-lambda.png", dpi=110)


if __name__ == "__main__":
    fig_profile()
    fig_lambda()
    fig_landscape()
    fig_icm_cut()
