"""Figures for sparsity/lec.qmd (sparse representations day, unit 1).
Run from the repository root:  python3 -I tools/figs/sparsity/figs.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *   # noqa: F401,F403
from matplotlib.patches import Polygon, Circle, Ellipse

U = "sparsity"
rng = np.random.default_rng(7)


def k_for(c, frac):
    e = np.sort(np.asarray(c) ** 2)[::-1]
    cs = np.cumsum(e) / e.sum()
    return int(np.searchsorted(cs, frac) + 1)


# ---- 1. sparse vs compressible: sorted magnitudes ----------------------------------------
def fig_sorted():
    n = 256
    spikes = np.zeros(n); spikes[rng.choice(n, 8, replace=False)] = rng.choice([-1, 1], 8) * (1 + rng.random(8))
    steps = np.zeros(n); steps[40:] += 1; steps[100:] -= 1.6; steps[176:] += 0.9   # breaks at 40, 100, 176
    img = cell()
    cases = [("8 spikes, pixel basis", np.abs(spikes), RED),
             ("3 steps, Haar basis", np.abs(haar1(steps, 8)), BLUE),
             ("micrograph, 2D Haar (4 levels)", np.abs(haar2(img - img.mean(), 4)).ravel(), GREEN)]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for lab, c, col in cases:
        s = np.sort(c)[::-1]; s = s / s[0]
        r = np.arange(1, len(s) + 1) / len(s)
        nz = s > 1e-12
        ax.loglog(r[nz], s[nz], color=col, lw=2.6, label=lab)
        print(f"  {lab}: nonzero {nz.sum()} of {len(s)}, 99% energy in {k_for(c, 0.99)} ({100 * k_for(c, 0.99) / len(s):.1f} %)")
    ax.set_xlabel("rank / number of coefficients (fraction)")
    ax.set_ylabel("|coefficient| / largest")
    ax.set_ylim(1e-4, 1.5)
    ax.legend(frameon=False, loc="lower left")
    ax.annotate("exactly sparse:\nthe curve stops", xy=(8 / 256, 0.45), xytext=(2e-3, 2e-2), color=RED, fontsize=FONT - 2,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.annotate("compressible: falls\nlike a power law", xy=(0.2, 0.012), xytext=(0.06, 0.25), color=GREEN, fontsize=FONT - 2,
                arrowprops=dict(arrowstyle="->", color=GREEN))
    save(fig, U, "fig-sorted.svg")


# ---- 2. the right basis depends on the signal ---------------------------------------------
def fig_bases():
    n = 64
    t = np.arange(n)
    sp = np.zeros(n); sp[[9, 30, 47]] = [1, -0.8, 0.6]
    sn = np.cos(2 * np.pi * 3 * (t + 0.5) / (2 * n) * 2) * 0.8 + 0.4 * np.cos(np.pi * 11 * (2 * t + 1) / (2 * n))
    st = np.where(t < 20, 0.2, np.where(t < 44, 1.0, -0.4))
    sigs = [("spikes", sp), ("two cosines", sn), ("steps", st)]
    bases = [("pixel basis", lambda x: x), ("DCT", dct1), ("Haar", lambda x: haar1(x, 6))]
    fig, axs = plt.subplots(3, 4, figsize=(11.5, 5.6), gridspec_kw=dict(width_ratios=[1.1, 1, 1, 1]))
    for i, (sname, s) in enumerate(sigs):
        a = axs[i, 0]
        a.plot(t, s, color=BLACK, lw=2); a.set_xticks([]); a.set_yticks([])
        a.set_ylabel(sname, fontsize=FONT)
        for j, (bname, f) in enumerate(bases):
            c = f(s)
            a = axs[i, j + 1]
            ml, sl, bl = a.stem(np.arange(n), c, basefmt=" ")
            plt.setp(sl, color=BLUE, lw=1.4); plt.setp(ml, color=BLUE, ms=2.5)
            k = k_for(c, 0.99)
            good = k <= 10
            a.text(0.97, 0.92, f"{k} for 99 %", transform=a.transAxes, ha="right", va="top", fontsize=FONT - 2,
                   color=GREEN if good else RED, fontweight="bold")
            a.set_xticks([]); a.set_yticks([]); a.axhline(0, color=GREY, lw=0.6)
            m = np.abs(c).max(); a.set_ylim(-1.15 * m, 1.3 * m)
            if i == 0: a.set_title(bname)
            print(f"  {sname} in {bname}: {k} coefficients carry 99 % of the energy")
    axs[0, 0].set_title("signal (64 samples)")
    fig.tight_layout()
    save(fig, U, "fig-bases.svg")


# ---- 3. transfer curves --------------------------------------------------------------------
def fig_transfer():
    w = np.linspace(-4, 4, 801)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(w, w, color=GREY, ls="--", lw=1.2, label="no penalty: c = w")
    ax.plot(w, w / 2, color=BLUE, lw=2.6, label="ℓ2: c = w / (1 + λ)")
    ax.plot(w, soft(w, 1), color=RED, lw=3, label="ℓ1: soft threshold at λ")
    wh = hard(w, 1); wh[np.abs(np.abs(w) - 1) < 0.006] = np.nan
    ax.plot(w, wh, color=ORANGE, lw=2.2, label="ℓ0: hard threshold at λ")
    ex = np.array([4, -0.5, 1.2, -2.5, 0.3, 3])
    ax.plot(ex, soft(ex, 1), "o", color=RED, ms=8, zorder=5)
    ax.axvspan(-1, 1, color=RED, alpha=0.08)
    ax.text(0, -3.3, "set to 0", ha="center", color=RED, fontsize=FONT - 1)
    ax.set_xlabel("noisy coefficient w"); ax.set_ylabel("estimate c")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left", fontsize=FONT - 3)
    ax.set_title("λ = 1")
    save(fig, U, "fig-transfer.svg")


# ---- 4. the geometry ------------------------------------------------------------------------
def ellipse_pts(b, Q, level, n=200):
    vals, vecs = np.linalg.eigh(Q)
    t = np.linspace(0, 2 * np.pi, n)
    u = np.stack([np.cos(t) * np.sqrt(2 * level / vals[0]), np.sin(t) * np.sqrt(2 * level / vals[1])])
    return (vecs @ u).T + b


def lasso2(Q, b, lam):
    best, bf = None, np.inf
    Qb = Q @ b
    cands = [np.zeros(2)]
    for s in (-1, 1):
        x1 = (Qb[0] - lam * s) / Q[0, 0]
        if np.sign(x1) == s: cands.append(np.array([x1, 0]))
        x2 = (Qb[1] - lam * s) / Q[1, 1]
        if np.sign(x2) == s: cands.append(np.array([0, x2]))
        for s2 in (-1, 1):
            x = np.linalg.solve(Q, Qb - lam * np.array([s, s2]))
            if np.sign(x[0]) == s and np.sign(x[1]) == s2: cands.append(x)
    for c in cands:
        f = 0.5 * (c - b) @ Q @ (c - b) + lam * np.abs(c).sum()
        if f < bf: bf, best = f, c
    return best


def fig_geometry():
    Q = np.array([[1, 0.3], [0.3, 1]]); b = np.array([3.0, 1.0]); lam = 2.0
    x1 = lasso2(Q, b, lam)
    x2 = np.linalg.solve(Q + lam * np.eye(2), Q @ b)
    print(f"  geometry: rho 0.3, b (3,1), lambda 2: l1 {x1.round(3)}, l2 {x2.round(3)}")
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 5.0))
    for ax, x, kind in [(axs[0], x2, "ℓ2"), (axs[1], x1, "ℓ1")]:
        for lev in [0.25, 1, 2.25, 4]:
            p = ellipse_pts(b, Q, lev); ax.plot(p[:, 0], p[:, 1], color=GREY, lw=1)
        lev = 0.5 * (x - b) @ Q @ (x - b)
        p = ellipse_pts(b, Q, lev); ax.plot(p[:, 0], p[:, 1], color=BLACK, lw=2.2)
        if kind == "ℓ2":
            r = np.hypot(*x); ax.add_patch(Circle((0, 0), r, fc=BLUE, alpha=0.15, ec=BLUE, lw=2.5))
            col = BLUE
        else:
            r = np.abs(x).sum(); ax.add_patch(Polygon([[r, 0], [0, r], [-r, 0], [0, -r]], fc=RED, alpha=0.15, ec=RED, lw=2.5))
            col = RED
        ax.plot(*b, "o", ms=12, mfc="white", mec=BLACK, mew=2.5)
        ax.text(b[0] + 0.15, b[1] + 0.25, "least squares", fontsize=FONT - 2)
        ax.plot(*x, "o", ms=11, color=col, mec="white", mew=1.5, zorder=5)
        ax.text(x[0] + 0.1, x[1] + (0.25 if kind == "ℓ2" else -0.45), f"({x[0]:.2f}, {x[1]:.2f})", color=col, fontsize=FONT - 1, fontweight="bold")
        ax.axhline(0, color=BLACK, lw=0.8); ax.axvline(0, color=BLACK, lw=0.8)
        ax.set_xlim(-2.2, 4.6); ax.set_ylim(-2.2, 3.4); ax.set_aspect("equal")
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        ax.set_title(f"{kind}: {'round ball, touches anywhere' if kind == 'ℓ2' else 'diamond, touches at a corner'}", fontsize=FONT - 1)
    save(fig, U, "fig-geometry.svg")


def fig_budget():
    b = np.array([3.0, 1.0])
    p2 = 2 * b / np.hypot(*b); p1 = np.array([2.0, 0.0])
    print(f"  budget 2: l2 {p2.round(3)}, l1 {p1}")
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    for r in [0.5, 1, 1.5, 2, 2.5]:
        ax.add_patch(Circle(b, r, fill=False, ec=GREY, lw=1))
    ax.add_patch(Circle((0, 0), 2, fc=BLUE, alpha=0.12, ec=BLUE, lw=2.5))
    ax.add_patch(Polygon([[2, 0], [0, 2], [-2, 0], [0, -2]], fc=RED, alpha=0.12, ec=RED, lw=2.5))
    ax.plot(*b, "o", ms=12, mfc="white", mec=BLACK, mew=2.5); ax.text(3.15, 1.2, "(3, 1)", fontsize=FONT - 1)
    ax.plot(*p2, "o", ms=10, color=BLUE); ax.text(p2[0] - 0.3, p2[1] + 0.45, "ℓ2: (1.90, 0.63)", color=BLUE, fontsize=FONT - 1, ha="right")
    ax.plot(*p1, "o", ms=10, color=RED); ax.text(2.1, -0.55, "ℓ1: (2, 0)", color=RED, fontsize=FONT - 1)
    ax.plot([b[0], p2[0]], [b[1], p2[1]], color=BLUE, ls="--"); ax.plot([b[0], p1[0]], [b[1], p1[1]], color=RED, ls="--")
    ax.axhline(0, color=BLACK, lw=0.8); ax.axvline(0, color=BLACK, lw=0.8)
    ax.set_xlim(-2.4, 4.4); ax.set_ylim(-2.4, 3.0); ax.set_aspect("equal")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.set_title("round contours, budget ‖x‖ ≤ 2", fontsize=FONT - 1)
    save(fig, U, "fig-budget.svg")


def fig_lpballs():
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw=dict(wspace=0.75))
    ax = axs[0]
    t = np.linspace(0, 2 * np.pi, 800)
    for p, col, lab in [(2, BLUE, "ℓ2"), (1, RED, "ℓ1"), (0.5, PURPLE, "ℓ½")]:
        c, s = np.cos(t), np.sin(t)
        r = (np.abs(c) ** p + np.abs(s) ** p) ** (-1 / p)
        ax.plot(r * c, r * s, color=col, lw=2.5, label=lab)
    ax.plot([-1.3, 1.3], [0, 0], color=ORANGE, lw=5, alpha=0.4, label="ℓ0 ≤ 1 (the axes)")
    ax.plot([0, 0], [-1.3, 1.3], color=ORANGE, lw=5, alpha=0.4)
    ax.set_aspect("equal"); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_title("unit balls: pointier as p falls", fontsize=FONT - 1); ax.legend(frameon=False, fontsize=FONT - 4, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax = axs[1]
    x = np.linspace(-2, 2, 801)
    ax.plot(x, x ** 2, color=BLUE, lw=2.5, label="ℓ2: x²")
    ax.plot(x, np.abs(x), color=RED, lw=2.5, label="ℓ1: |x|")
    ax.plot(x, np.sqrt(np.abs(x)), color=PURPLE, lw=2.5, label="ℓ½: √|x|")
    y0 = np.where(x == 0, 0, 1.0); ax.plot(x[x < 0], y0[x < 0], color=ORANGE, lw=2.5, label="ℓ0: count")
    ax.plot(x[x > 0], y0[x > 0], color=ORANGE, lw=2.5); ax.plot([0], [0], "o", color=ORANGE)
    ax.set_ylim(-0.1, 2.2); ax.set_title("penalty of one coefficient", fontsize=FONT - 1)
    ax.legend(frameon=False, fontsize=FONT - 4, loc="upper center")
    ax.text(1.0, 0.12, "convex: ℓ1, ℓ2", fontsize=FONT - 3, color=GREY)
    save(fig, U, "fig-lpballs.svg")


# ---- 5. lasso path -------------------------------------------------------------------------
def lasso_cd(A, y, lam, x0=None, sweeps=2000):
    x = np.zeros(A.shape[1]) if x0 is None else x0.copy()
    cn = (A ** 2).sum(0)
    r = y - A @ x
    for _ in range(sweeps):
        mx = 0
        for j in range(A.shape[1]):
            rho = A[:, j] @ r + cn[j] * x[j]
            nx = soft(rho, lam) / cn[j]
            d = nx - x[j]
            if d: r -= A[:, j] * d; x[j] = nx; mx = max(mx, abs(d))
        if mx < 1e-12: break
    return x


def fig_path():
    r = np.random.default_rng(2026)
    m, n = 24, 6
    A = r.normal(size=(m, n)) / np.sqrt(m)
    A[:, 1] = 0.6 * A[:, 0] + 0.8 * A[:, 1]; A[:, 4] = 0.5 * A[:, 3] + 0.87 * A[:, 4]
    xt = np.array([2.5, 0, -1.5, 1, 0, 0.5]); y = A @ xt + 0.15 * r.normal(size=m)
    lmax = np.abs(A.T @ y).max()
    lams = lmax * np.logspace(0.08, -2.5, 120)
    P1, x = [], None
    for l in lams:
        x = lasso_cd(A, y, l, x); P1.append(x.copy())
    P1 = np.array(P1)
    P2 = np.array([np.linalg.solve(A.T @ A + l * np.eye(n), A.T @ y) for l in lams])
    enter = [lams[np.argmax(np.abs(P1[:, j]) > 0)] / lmax if np.any(P1[:, j]) else 0 for j in range(n)]
    print(f"  path: lambda_max {lmax:.3f}; entry points (λ/λmax): {np.round(enter, 3)}")
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.2), sharey=True)
    cols = [BLUE, ORANGE, GREEN, RED, PURPLE, "#8c564b"]
    for ax, P, title in [(axs[0], P1, "ℓ1 (lasso): enter one at a time"), (axs[1], P2, "ℓ2 (ridge): all at once, never 0")]:
        for j in range(n):
            ax.semilogx(lams / lmax, P[:, j], color=cols[j], lw=2.4)
        ax.axvline(1, color=GREY, ls="--", lw=1.2); ax.axhline(0, color=BLACK, lw=0.6)
        ax.set_xlabel("λ / λmax (log scale)"); ax.set_title(title, fontsize=FONT - 1)
        ax.invert_xaxis()
    axs[0].text(0.95, 2.2, "λmax", color=GREY, fontsize=FONT - 2, ha="right")
    axs[0].set_ylabel("coefficients")
    save(fig, U, "fig-path.svg")


# ---- 6. shrinkage bias ---------------------------------------------------------------------
def fig_bias():
    n = 128
    x = np.zeros(n); x[[20, 52, 90, 110]] = [3, -2.5, 2, 1.2]
    r = np.random.default_rng(3)
    y = x + 0.3 * r.normal(size=n)
    lam = 1.0
    xs = soft(y, lam)
    S = np.flatnonzero(xs); xd = np.zeros(n); xd[S] = y[S]
    print(f"  bias: peaks true {x[[20, 52, 90, 110]]}, soft {xs[[20, 52, 90, 110]].round(2)}, debiased {xd[[20, 52, 90, 110]].round(2)}, false atoms {len(set(S) - {20, 52, 90, 110})}")
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.plot(np.arange(n), y, color=GREY, lw=1, label="noisy (σ = 0.3)")
    ml, sl, bl = ax.stem(np.arange(n), xd, basefmt=" ", label="debiased (refit)")
    plt.setp(sl, color=GREEN, lw=4, alpha=0.5); plt.setp(ml, color=GREEN, ms=9, alpha=0.6)
    ml, sl, bl = ax.stem(np.arange(n), xs, basefmt=" ", label="ℓ1 (soft, λ = 1)")
    plt.setp(sl, color=RED, lw=2.2); plt.setp(ml, color=RED, ms=6)
    ax.plot(np.flatnonzero(x), x[x != 0], "_", color=BLACK, ms=26, mew=2.5, label="true height")
    ax.axhline(0, color=BLACK, lw=0.6)
    ax.legend(frameon=False, fontsize=FONT - 4, ncol=2, loc="lower left")
    ax.set_xlabel("position")
    save(fig, U, "fig-bias.svg")


# ---- 7. Laplace prior ----------------------------------------------------------------------
def fig_laplace():
    img = cell()
    c = haar2(img, 1)
    d = np.concatenate([c[:128, 128:].ravel(), c[128:, :128].ravel(), c[128:, 128:].ravel()])
    sd, b = d.std(), np.abs(d - np.median(d)).mean()
    print(f"  laplace: level-1 Haar details of the micrograph: sd {sd:.2f}, mean |w| {b:.2f}, frac |w|>4sd {np.mean(np.abs(d) > 4 * sd) * 100:.2f} %")
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    edges = np.linspace(-80, 80, 81)
    h, e = np.histogram(d, bins=edges, density=True)
    ax.bar(0.5 * (e[1:] + e[:-1]), h, width=e[1] - e[0], color="#bbbbbb", label="detail coefficients")
    w = np.linspace(-80, 80, 400)
    ax.plot(w, np.exp(-w ** 2 / (2 * sd ** 2)) / (np.sqrt(2 * np.pi) * sd), color=BLUE, lw=2.5, label="Gaussian (ℓ2 prior)")
    ax.plot(w, np.exp(-np.abs(w) / b) / (2 * b), color=RED, lw=2.5, label="Laplace (ℓ1 prior)")
    ax.set_yscale("log"); ax.set_ylim(1e-6, 0.2)
    ax.set_xlabel("coefficient value (grey levels)"); ax.set_ylabel("density (log)")
    ax.legend(frameon=False, fontsize=FONT - 3, loc="upper right")
    ax.text(-78, 2e-6, CREDIT, fontsize=10, color=GREY)
    save(fig, U, "fig-laplace.svg")


# ---- 8. analysis vs synthesis --------------------------------------------------------------
def fig_analysis():
    n = 100; t = np.arange(n)
    x = np.where(t < 30, 1.0, 3.0); x = x + np.clip((t - 60) / 25, 0, 1) * 1.5
    g = np.diff(x)
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 4.2), sharex=True)
    axs[0].plot(t, x, color=BLACK, lw=2.5); axs[0].set_ylabel("image row")
    axs[0].text(28, 2.4, "step", ha="right", fontsize=FONT - 2); axs[0].text(72, 3.6, "ramp", fontsize=FONT - 2)
    ml, sl, bl = axs[1].stem(t[:-1] + 0.5, g, basefmt=" ")
    plt.setp(sl, color=RED, lw=1.8); plt.setp(ml, color=RED, ms=4)
    axs[1].set_ylabel("difference"); axs[1].set_xlabel("position")
    axs[1].text(31, 1.7, "one nonzero", color=RED, fontsize=FONT - 2)
    axs[1].text(62, 0.4, "25 small nonzeros", color=RED, fontsize=FONT - 2)
    axs[1].set_ylim(-0.2, 2.3)
    save(fig, U, "fig-analysis.svg")


if __name__ == "__main__":
    os.chdir(ROOT)
    fig_sorted(); fig_bases(); fig_transfer(); fig_geometry(); fig_budget(); fig_lpballs(); fig_path(); fig_bias(); fig_laplace(); fig_analysis()
