"""Figures for ista/lec.qmd (sparse representations day, unit 2).
Run from the repository root:  python3 -I tools/figs/ista/figs.py
1D figures with numpy (tools/figs/sparsity/common.py); the 1D deblurring comparison and the 2D
inpainting with the demos' code (data.ts, via node + esbuild).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sparsity"))
from common import *   # noqa: F401,F403

U = "ista"


def stems(ax, x, col, ms=4, lw=1.6, label=None):
    idx = np.flatnonzero(np.abs(x) > 1e-9)
    if len(idx) == 0: return
    ml, sl, bl = ax.stem(idx, x[idx], basefmt=" ", label=label)
    plt.setp(sl, color=col, lw=lw); plt.setp(ml, color=col, ms=ms)


# ---- the four forward operators ------------------------------------------------------------
def fig_operators():
    n = 128; r = np.random.default_rng(5)
    x = np.zeros(n); x[[14, 40, 47, 80, 105]] = [1, -0.7, 0.9, 1.2, -0.8]
    B = gauss_blur_matrix(n, 2.5)
    mask = r.random(n) > 0.5
    F = np.fft.fft(x); keep = np.zeros(n, bool); keep[r.choice(n, 32, replace=False)] = True
    zf = np.real(np.fft.ifft(np.where(keep, F, 0)))
    fig, axs = plt.subplots(1, 4, figsize=(13, 2.9), sharey=True)
    t = np.arange(n)
    for a in axs: a.plot(t, x, color=GREY, lw=1, alpha=0.6); a.set_xticks([]); a.axhline(0, color=BLACK, lw=0.5)
    axs[0].plot(t, x + 0.05 * r.normal(size=n), color=BLUE, lw=1.6); axs[0].set_title("identity (+ noise)")
    axs[1].plot(t, B @ x, color=BLUE, lw=2); axs[1].set_title("blur")
    axs[2].plot(t[mask], x[mask], "o", color=BLUE, ms=3.5); axs[2].set_title("mask: half missing")
    axs[3].plot(t, zf * n / 32, color=BLUE, lw=1.4); axs[3].set_title("¼ of Fourier, zero-filled")
    axs[0].set_ylim(-1.3, 1.6)
    fig.tight_layout()
    save(fig, U, "fig-operators.svg")


# ---- ISTA steps on a blurred spike signal --------------------------------------------------
def spike_problem():
    n = 96; x = np.zeros(n); x[[20, 30, 52, 70, 76]] = [1.2, -0.8, 1.0, 0.9, 0.7]
    B = gauss_blur_matrix(n, 2.0); r = np.random.default_rng(1)
    y = B @ x + 0.01 * r.normal(size=n)
    return x, B, y


def fig_ista_steps():
    x, B, y = spike_problem()
    lam = 0.05; L = np.linalg.norm(B, 2) ** 2; tau = 1 / L
    c = np.zeros(len(x)); frames = {}
    for k in range(1, 31):
        z = c + tau * B.T @ (y - B @ c)
        cn = soft(z, tau * lam)
        if k in (1, 3, 30): frames[k] = (z, cn)
        c = cn
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 4.6), sharex=True, sharey=True)
    t = np.arange(len(x))
    for col, (k, (z, cn)) in enumerate(frames.items()):
        a = axs[0, col]; a.plot(t, z, color=ORANGE, lw=2); a.axhspan(-tau * lam, tau * lam, color=ORANGE, alpha=0.25)
        a.plot(np.flatnonzero(x), x[x != 0], "o", mfc="none", mec=BLACK, ms=6)
        a.set_title(f"iteration {k}: gradient step", fontsize=FONT - 2)
        a = axs[1, col]; stems(a, cn, BLUE); a.plot(np.flatnonzero(x), x[x != 0], "o", mfc="none", mec=BLACK, ms=6)
        a.set_title(f"then threshold: {np.count_nonzero(cn)} nonzero", fontsize=FONT - 2)
        for a in axs[:, col]: a.axhline(0, color=BLACK, lw=0.5)
    axs[0, 0].set_ylim(-1.0, 1.4)
    fig.tight_layout()
    save(fig, U, "fig-ista-steps.svg")
    print(f"  ista steps: L {L:.3f}, lambda {lam}, nonzeros at iterations 1, 3, 30: {[int(np.count_nonzero(f[1])) for f in frames.values()]}")


# ---- the by-hand example: iterates in the plane --------------------------------------------
def fig_ista_path():
    A = np.array([[1, 0.5], [0.5, 1]]); y = np.array([2.0, 1.0]); lam, tau = 0.5, 0.4
    c = np.zeros(2); pts = [c.copy()]; zs = []
    for k in range(25):
        z = c + tau * A.T @ (y - A @ c); zs.append(z.copy())
        c = soft(z, tau * lam); pts.append(c.copy())
    print("  ista by hand:", [p.round(4).tolist() for p in pts[:4]], "->", pts[-1].round(4))
    Q = A.T @ A; b = np.linalg.solve(A, y)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    g1, g2 = np.meshgrid(np.linspace(-0.4, 2.6, 300), np.linspace(-0.6, 1.4, 300))
    F = 0.5 * ((A[0, 0] * g1 + A[0, 1] * g2 - y[0]) ** 2 + (A[1, 0] * g1 + A[1, 1] * g2 - y[1]) ** 2) + lam * (np.abs(g1) + np.abs(g2))
    ax.contour(g1, g2, F, levels=np.linspace(F.min(), F.min() + 2, 12), colors=GREY, linewidths=0.8)
    for k in range(6):
        p, z, q = pts[k], zs[k], pts[k + 1]
        ax.annotate("", xy=z, xytext=p, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8))
        ax.annotate("", xy=q, xytext=z, arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.8))
    ax.plot(*np.array(pts).T, "o", color=BLUE, ms=4)
    ax.plot(*pts[-1], "*", color=RED, ms=16, zorder=5); ax.text(pts[-1][0] - 0.1, pts[-1][1] - 0.22, "(1.6, 0)", color=RED, fontsize=FONT - 1)
    ax.plot(*b, "s", color=BLACK, ms=7); ax.text(b[0] + 0.05, b[1] + 0.08, "truth (2, 0)", fontsize=FONT - 2)
    ax.axhline(0, color=BLACK, lw=0.8); ax.axvline(0, color=BLACK, lw=0.8)
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.set_aspect("equal")
    ax.plot([], [], color=ORANGE, lw=2, label="gradient step"); ax.plot([], [], color=BLUE, lw=2, label="soft threshold")
    ax.legend(frameon=False, loc="upper right", fontsize=FONT - 3)
    save(fig, U, "fig-ista-path.svg")


# ---- FISTA vs ISTA ------------------------------------------------------------------------
def fig_fista():
    x, B, y = spike_problem()
    lam = 0.02
    _, hI, _ = ista(B, y, lam, 3000)
    _, hF, _ = ista(B, y, lam, 3000, fista=True)
    fstar = min(hI.min(), hF.min())
    _, hR, _ = ista(B, y, lam, 20000, fista=True); fstar = min(fstar, hR.min())
    k = np.arange(1, 3001)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.loglog(k, hI - fstar + 1e-12, color=ORANGE, lw=2.5, label="ISTA")
    ax.loglog(k, hF - fstar + 1e-12, color=BLUE, lw=2.5, label="FISTA")
    ax.set_ylim(1e-9, 1); ax.set_xlabel("iteration"); ax.set_ylabel("objective − minimum")
    ax.legend(frameon=False)
    tol = 1e-6
    iI = int(np.argmax(hI - fstar < tol)) + 1; iF = int(np.argmax(hF - fstar < tol)) + 1
    print(f"  fista: iterations to gap < 1e-6: ISTA {iI if (hI - fstar < tol).any() else '>3000'}, FISTA {iF}")
    save(fig, U, "fig-fista.svg")


# ---- OMP on a blurred spike signal ---------------------------------------------------------
def fig_omp():
    x, B, y = spike_problem()
    xo, steps = omp(B, y, 5)
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.2), sharey=True)
    t = np.arange(len(x))
    for k, a in enumerate(axs):
        xk, rk, j = steps[k]
        a.plot(t, rk, color=GREY, lw=1.5, label="residual")
        stems(a, xk, BLUE, label="chosen atoms (refit)")
        a.plot(np.flatnonzero(x), x[x != 0], "o", mfc="none", mec=BLACK, ms=6, label="truth")
        a.set_title(f"step {k + 1}: picks position {j}, ‖r‖ = {np.linalg.norm(rk):.2f}", fontsize=FONT - 3)
        a.axhline(0, color=BLACK, lw=0.5)
    axs[0].legend(frameon=False, fontsize=FONT - 5, loc="lower left")
    fig.tight_layout()
    save(fig, U, "fig-omp.svg")
    print("  omp picks:", [s[2] for s in steps], "residuals", [round(float(np.linalg.norm(s[1])), 3) for s in steps], "final err", np.abs(xo - x).max().round(3))


# ---- union of two dictionaries -------------------------------------------------------------
def fig_union():
    n = 128; r = np.random.default_rng(9)
    sp = np.zeros(n); sp[[20, 61, 97]] = [1.5, -1.2, 1.0]
    dc = np.zeros(n); dc[[3, 7]] = [4.0, -3.0]
    D = idct(np.eye(n), norm="ortho", axis=0)   # columns: DCT atoms
    x = sp + D @ dc + 0.03 * r.normal(size=n)
    res = {}
    for name, A in [("DCT only", D), ("spikes only", np.eye(n)), ("spikes ∪ DCT", np.hstack([np.eye(n), D]))]:
        c, _, _ = ista(A, x, 0.08, 3000, fista=True)
        res[name] = (A @ c, np.count_nonzero(np.abs(c) > 1e-6))
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.0), sharey=True)
    t = np.arange(n)
    for a, (name, (xh, nz)) in zip(axs, res.items()):
        a.plot(t, x, color=GREY, lw=1.2); a.plot(t, xh, color=BLUE, lw=2)
        a.set_title(f"{name}: {nz} coefficients", fontsize=FONT - 1); a.set_xticks([])
    fig.tight_layout()
    save(fig, U, "fig-union.svg")
    print("  union nonzeros:", {k: v[1] for k, v in res.items()})


# ---- node data: 1D deblurring, 2D inpainting -----------------------------------------------
def fig_node():
    tmpc = tempfile.mkdtemp(); cp = os.path.join(tmpc, "cell.f64"); cell().tofile(cp)
    tmp = run_node(os.path.join(HERE, "data.ts"), [cp])
    d = load_json(tmp, "deblur1d.json")
    print("  deblur1d:", d["lam"], {k: round(v, 4) for k, v in d["rmse"].items()}, {k: round(v, 3) for k, v in d["bump"].items()})
    t = np.arange(len(d["x"]))
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.4), sharey=True)
    for a, key, name, col in [(axs[0], "tik", "Tikhonov (smoothness)", ORANGE), (axs[1], "tv", "TV", GREEN), (axs[2], "wav", "wavelet ℓ1 (db4)", BLUE)]:
        a.plot(t, d["y"], color=GREY, lw=1.2, label="blurred + noise")
        a.plot(t, d["x"], color=BLACK, lw=1, ls="--", label="truth")
        a.plot(t, d[key], color=col, lw=2.2)
        a.set_title(f"{name}: RMSE {d['rmse'][key]:.3f}", fontsize=FONT - 1); a.set_xticks([])
    axs[0].legend(frameon=False, fontsize=FONT - 5, loc="upper left")
    fig.tight_layout()
    save(fig, U, "fig-deblur.svg")
    p = load_json(tmp, "inpaint.json")
    n = 128; truth = np.array(p["truth"]).reshape(n, n)
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 7.6))
    for row, key, title in [(0, "random", "50 % of the pixels missing"), (1, "scratch", "scratches and a 16 × 16 hole")]:
        m = np.array(p[key]["mask"]).reshape(n, n).astype(bool)
        obs = np.where(m, truth, np.nan)
        cmap = plt.get_cmap("gray").copy(); cmap.set_bad("#ffcc33")
        axs[row, 0].imshow(obs, cmap=cmap, vmin=0, vmax=255); axs[row, 0].set_title(title, fontsize=FONT - 3)
        axs[row, 1].imshow(np.array(p[key]["rec"]).reshape(n, n), cmap="gray", vmin=0, vmax=255)
        axs[row, 1].set_title(f"wavelet ℓ1: {p[key]['psnrRec']:.1f} dB", fontsize=FONT - 3)
        print(f"  inpaint {key}: observed (mean-filled) {p[key]['psnrObs']:.2f} dB, restored {p[key]['psnrRec']:.2f} dB")
    for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.text(0.5, 0.005, CREDIT, ha="center", fontsize=10, color=GREY)
    fig.tight_layout()
    save(fig, U, "fig-inpaint.png", dpi=110)


if __name__ == "__main__":
    os.chdir(ROOT)
    fig_operators(); fig_ista_steps(); fig_ista_path(); fig_fista(); fig_omp(); fig_union(); fig_node()
