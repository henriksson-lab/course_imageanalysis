"""Figures for ransac/lec.qmd.

Run from the repository root:
    tools/figs/hough-lines/capture.sh /tmp/houghstills      # demo screenshots (needs docs/demos/dist)
    python3 -I tools/figs/ransac/figs.py /tmp/houghstills
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../hough-lines"))
from stills import plt, shot, grid_of_panels, BLUE, ORANGE, GREEN, RED, PURPLE, GREY  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = os.path.join(REPO, "ransac")
SH = sys.argv[1] if len(sys.argv) > 1 else None
EPS = 0.4


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200 if name.endswith(".png") else None,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def data(seed=4):
    rng = np.random.default_rng(seed)
    n_in, n_out = 24, 16
    x = rng.uniform(0, 10, n_in)
    yin = 0.5 * x + 1 + rng.normal(0, 0.15, n_in)
    out = rng.uniform([0, 0], [10, 10], size=(n_out, 2))
    P = np.vstack([np.c_[x, yin], out])
    truth = np.r_[np.ones(n_in, bool), np.zeros(n_out, bool)]
    return P, truth


def line_through(p, q):
    d = q - p
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    return n, n @ p


def dist(P, n, rho):
    return np.abs(P @ n - rho)


def tls(P):
    c = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    n = V[:, 0]
    return n, n @ c


def angle(n):
    """direction angle of the line (degrees) from its normal"""
    return np.degrees(np.arctan2(n[0], -n[1])) % 180


def draw_line(ax, n, rho, **kw):
    xs = np.array([-1, 11.0])
    ax.plot(xs, (rho - n[0] * xs) / n[1], **kw)


def draw_band(ax, n, rho, eps, **kw):
    xs = np.array([-1, 11.0])
    h = eps / abs(n[1])
    yc = (rho - n[0] * xs) / n[1]
    ax.fill_between(xs, yc - h, yc + h, **kw)


def setup(ax):
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-0.3, 10.3)
    ax.set_aspect("equal")
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0, 5, 10])


# ---------------------------------------------------------------- a good and a bad sample
def fig_samples():
    P, truth = data()
    ii, oo = np.nonzero(truth)[0], np.nonzero(~truth)[0]
    # a good sample: the two inliers 3rd from each end in x; a bad one: an inlier and an outlier
    xs = np.argsort(P[ii, 0])
    good = [ii[xs[3]], ii[xs[-4]]]
    bad = [ii[xs[8]], oo[2]]
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.9))
    counts = []
    for ax, smp, title in zip(axs, [good, bad], ["both inliers", "one outlier"]):
        n, rho = line_through(P[smp[0]], P[smp[1]])
        inl = dist(P, n, rho) < EPS
        counts.append(int(inl.sum()))
        draw_band(ax, n, rho, EPS, color=BLUE, alpha=0.15, lw=0)
        draw_line(ax, n, rho, color=BLUE, lw=2)
        ax.plot(*P[~inl].T, "o", color=GREY, ms=5)
        ax.plot(*P[inl].T, "o", color=GREEN, ms=6)
        ax.plot(*P[smp].T, "o", ms=13, mfc="none", mec=RED, mew=2.5)
        setup(ax)
        ax.set_title(f"{title}: {inl.sum()} inliers", fontsize=13)
    axs[0].set_ylabel("y")
    for ax in axs:
        ax.set_xlabel("x")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig-samples.svg")
    return counts, int(truth.sum()), len(P)


# ---------------------------------------------------------------- refit on the inliers
def ransac(P, iters=17, seed=1):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(P), 2, replace=False)
        n, rho = line_through(P[i], P[j])
        inl = dist(P, n, rho) < EPS
        if best is None or inl.sum() > best[2].sum():
            best = (n, rho, inl, (i, j))
    return best


def fig_refit(runs=1000):
    """Angle error of the best sample's line and of the refit, over many seeded data sets and runs."""
    nt, rt = line_through(np.array([0, 1.0]), np.array([10, 6.0]))
    e_s, e_r = [], []
    for k in range(runs):
        P, truth = data(seed=100 + k)
        n, rho, inl, _ = ransac(P, iters=17, seed=k)
        nr, rr = tls(P[inl])
        e_s.append(abs((angle(n) - angle(nt) + 90) % 180 - 90))
        e_r.append(abs((angle(nr) - angle(nt) + 90) % 180 - 90))
    e_s, e_r = np.array(e_s), np.array(e_r)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    bins = np.linspace(0, 3, 31)
    ax.hist(e_s, bins, color=RED, histtype="step", lw=2.5, label=f"best sample: median {np.median(e_s):.2f}°")
    ax.hist(e_r, bins, color=BLUE, histtype="step", lw=2.5, label=f"refit: median {np.median(e_r):.2f}°")
    ax.set_xlabel("angle error (degrees)")
    ax.set_ylabel(f"runs (of {runs})")
    ax.legend(fontsize=12, frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "fig-refit.svg")
    return np.median(e_s), np.median(e_r), np.mean(e_s > 3), np.mean(e_r > 3)


# ---------------------------------------------------------------- probability of success
def fig_iterations():
    cases = [(0.5, 2, BLUE), (0.5, 3, ORANGE), (0.2, 2, GREEN), (0.3, 3, PURPLE)]
    N = np.arange(0, 220)
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    need = []
    for w, s, c in cases:
        p = 1 - (1 - w ** s) ** N
        n99 = int(np.ceil(np.log(0.01) / np.log(1 - w ** s)))
        need.append(n99)
        ax.plot(N, p, color=c, lw=2.5, label=f"w = {w}, s = {s}: N = {n99}")
        ax.plot(n99, 1 - (1 - w ** s) ** n99, "o", color=c, ms=7)
    ax.axhline(0.99, color=GREY, lw=1, ls="--")
    ax.text(218, 0.955, "99 %", color=GREY, fontsize=13, ha="right", va="top")
    ax.set_xlim(0, 220)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("samples N")
    ax.set_ylabel("P(an all-inlier sample)")
    ax.legend(fontsize=12, loc="lower right", frameon=False, bbox_to_anchor=(1.0, 0.04))
    fig.tight_layout()
    save(fig, "fig-iterations.svg")
    return need


# ---------------------------------------------------------------- randomised Hough
def fig_randhough():
    P, truth = data()
    rng = np.random.default_rng(2)
    K = 300
    TH, RH, both = [], [], []
    for _ in range(K):
        i, j = rng.choice(len(P), 2, replace=False)
        n, rho = line_through(P[i], P[j])
        th = np.degrees(np.arctan2(n[1], n[0]))
        if th < 0:
            th += 180
            rho = -rho
        if th >= 180:
            th -= 180
            rho = -rho
        TH.append(th)
        RH.append(rho)
        both.append(truth[i] and truth[j])
    TH, RH, both = np.array(TH), np.array(RH), np.array(both)
    nt, rt = line_through(np.array([0, 1.0]), np.array([10, 6.0]))
    tt = np.degrees(np.arctan2(nt[1], nt[0]))
    if tt < 0:
        tt, rt = tt + 180, -rt
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    ax.plot(TH[~both], RH[~both], "o", color=GREY, ms=4, alpha=0.7, label="pair with an outlier")
    ax.plot(TH[both], RH[both], "o", color=GREEN, ms=5, label="pair of inliers")
    ax.plot(tt, rt, "o", ms=20, mfc="none", mec=BLUE, mew=2.5)
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("θ (degrees)")
    ax.set_ylabel("ρ")
    ax.legend(fontsize=12, loc="lower left", frameon=False, handletextpad=0.1)
    fig.tight_layout()
    save(fig, "fig-randhough.svg")
    near = (np.abs(TH - tt) < 3) & (np.abs(RH - rt) < 0.3)
    return K, int(both.sum()), int(near.sum()), round(tt, 1), round(rt, 2)


def fig_stills():
    crops = [shot(SH, name)[49:480, 0:450] for name in ("rs_epsbig", "rs_twolines")]
    grid_of_panels(crops, ["ε = 6: one model for both", "ε = 2, sequential: two"],
                   os.path.join(OUT, "fig-eps.png"), ncols=2, w=5.4)


if __name__ == "__main__":
    print("samples: inliers good/bad, true inliers, points:", fig_samples())
    print("refit: median error sample, refit; fraction > 3 deg sample, refit:", fig_refit())
    print("iterations needed:", fig_iterations())
    print("randomised Hough: pairs, inlier pairs, near true, true (theta, rho):", fig_randhough())
    if SH:
        fig_stills()
