"""Figures for compressed-sensing/lec.qmd (sparse representations day, unit 3).
Run from the repository root:  python3 -I tools/figs/compressed-sensing/figs.py
The real orthonormal Fourier basis below has the same layout as lib/wavelet.ts realFourier
(DC, cos/sin pairs, Nyquist); the phase diagram is computed with the demo's code (data.ts).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sparsity"))
from common import *   # noqa: F401,F403
from matplotlib.patches import Polygon, Circle

U = "compressed-sensing"
N = 256


def real_fourier_matrix(n):
    F = np.zeros((n, n)); t = np.arange(n)
    F[0] = 1 / np.sqrt(n); F[n - 1] = np.cos(np.pi * t) / np.sqrt(n)
    for k in range(1, n // 2):
        F[2 * k - 1] = np.sqrt(2 / n) * np.cos(2 * np.pi * k * t / n)
        F[2 * k] = -np.sqrt(2 / n) * np.sin(2 * np.pi * k * t / n)
    return F


def freq_of_row(j, n):
    return n // 2 if j == n - 1 else (j + 1) // 2


F = real_fourier_matrix(N)
assert np.allclose(F @ F.T, np.eye(N), atol=1e-10)
rng = np.random.default_rng(31)
x = np.zeros(N); sup = np.sort(rng.choice(N, 8, replace=False)); x[sup] = rng.choice([-1, 1], 8) * (1 + rng.random(8))


def fig_surprise():
    c = F @ x
    mag = np.zeros(N // 2 + 1)
    for j in range(N): f = freq_of_row(j, N); mag[f] = np.hypot(mag[f], c[j])
    rows = np.sort(rng.choice(N, 40, replace=False)); fs = sorted(set(freq_of_row(j, N) for j in rows))
    fig, axs = plt.subplots(2, 1, figsize=(7.6, 4.8))
    ml, sl, bl = axs[0].stem(np.flatnonzero(x), x[x != 0], basefmt=" "); plt.setp(sl, color=BLACK, lw=2); plt.setp(ml, color=BLACK, ms=5)
    axs[0].axhline(0, color=GREY, lw=0.8); axs[0].set_xlim(0, N); axs[0].set_title("8 spikes in 256 samples", fontsize=FONT - 1)
    axs[1].bar(np.arange(N // 2 + 1), mag, color="#cccccc", width=0.8)
    axs[1].plot(fs, mag[fs], "o", color=GREEN, ms=5)
    axs[1].set_xlim(0, N // 2); axs[1].set_title("its Fourier coefficients: 40 random measurements (green)", fontsize=FONT - 1)
    axs[1].set_xlabel("frequency")
    fig.tight_layout()
    save(fig, U, "fig-surprise.svg")


def fig_line():
    a = np.array([1.0, 2.0]); y = 2.0      # one equation, two unknowns: x1 + 2 x2 = 2
    xmn = a * y / (a @ a)                  # minimum norm (0.4, 0.8)
    xl1 = np.array([0.0, 1.0])             # sparsest (the vertex)
    print("  line: min-norm", xmn, "l1", xl1, "l1 norms", np.abs(xmn).sum(), np.abs(xl1).sum())
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    t = np.linspace(-1.5, 3, 10)
    ax.plot(t, (y - t) / 2, color=BLACK, lw=2.4, label="all solutions of x₁ + 2x₂ = 2")
    r2 = np.linalg.norm(xmn); ax.add_patch(Circle((0, 0), r2, fc=BLUE, alpha=0.15, ec=BLUE, lw=2.2))
    ax.add_patch(Polygon([[1, 0], [0, 1], [-1, 0], [0, -1]], fc=RED, alpha=0.15, ec=RED, lw=2.2))
    ax.plot(*xmn, "o", color=BLUE, ms=10); ax.text(xmn[0] + 0.1, xmn[1] + 0.1, "least energy (0.4, 0.8)", color=BLUE, fontsize=FONT - 2)
    ax.plot(*xl1, "o", color=RED, ms=10); ax.text(-1.3, 0.45, "sparsest,\nsmallest ℓ1:\n(0, 1)", color=RED, fontsize=FONT - 2)
    ax.axhline(0, color=BLACK, lw=0.7); ax.axvline(0, color=BLACK, lw=0.7)
    ax.set_xlim(-1.4, 2.6); ax.set_ylim(-1.2, 1.7); ax.set_aspect("equal"); ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.legend(frameon=False, loc="lower left", fontsize=FONT - 4)
    save(fig, U, "fig-line.svg")


def zero_filled(rows):
    P = F[rows]
    return P.T @ (P @ x) * N / len(rows)


def fig_aliasing():
    reg = np.array([j for j in range(N) if freq_of_row(j, N) % 4 == 0])
    rnd = np.sort(np.random.default_rng(57).choice(N, len(reg), replace=False))
    zr, zn = zero_filled(reg), zero_filled(rnd)
    print(f"  aliasing: {len(reg)} rows each; regular: copies of height {np.abs(zr).max():.2f}; random: junk sd {np.std(zn[x == 0]):.3f}, spikes ≈ {zn[sup].round(2)}")
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 3.4), sharey=True)
    for a, z, title in [(axs[0], zr, "every 4th frequency: 4 copies of each spike"), (axs[1], zn, "the same number at random: noise-like junk")]:
        a.plot(np.arange(N), z, color=RED, lw=1.5)
        a.plot(sup, x[sup], "o", mfc="none", mec=BLACK, ms=7)
        a.axhline(0, color=GREY, lw=0.6); a.set_title(title, fontsize=FONT - 1); a.set_xlim(0, N)
    axs[0].set_ylabel("zero-filled × N/M")
    fig.tight_layout()
    save(fig, U, "fig-aliasing.svg")


def fig_iterations():
    rows = np.sort(np.random.default_rng(57).choice(N, 64, replace=False))
    A = F[rows]; y = A @ x
    lam = 0.05
    shots = {}
    c = np.zeros(N)
    for k in range(1, 201):
        z = c + A.T @ (y - A @ c)
        c = soft(z, lam)
        if k in (1, 5, 20, 200): shots[k] = c.copy()
    fig, axs = plt.subplots(1, 4, figsize=(13.5, 2.9), sharey=True)
    for a, (k, ck) in zip(axs, shots.items()):
        a.plot(np.arange(N), ck, color=BLUE, lw=1.4)
        a.plot(sup, x[sup], "o", mfc="none", mec=BLACK, ms=6)
        a.set_title(f"iteration {k}: {np.count_nonzero(ck)} nonzero", fontsize=FONT - 2); a.set_xticks([])
        a.axhline(0, color=GREY, lw=0.5)
    fig.tight_layout()
    save(fig, U, "fig-iterations.svg")
    e = np.linalg.norm(shots[200] - x) / np.linalg.norm(x)
    print(f"  iterations (64 random rows, λ = {lam}): nonzeros {[int(np.count_nonzero(v)) for v in shots.values()]}, rel. error at 200: {e:.3f}")


def fig_incoherence():
    t = np.arange(N)
    pix = np.sort(np.random.default_rng(3).choice(N, 16, replace=False))
    fig, axs = plt.subplots(2, 1, figsize=(7.4, 4.2), sharex=True)
    for a in axs:
        ml, sl, bl = a.stem(sup, x[sup], basefmt=" "); plt.setp(sl, color=BLACK, lw=2); plt.setp(ml, color=BLACK, ms=4)
        a.axhline(0, color=GREY, lw=0.6); a.set_yticks([])
    hits = np.intersect1d(pix, sup)
    axs[0].plot(pix, np.zeros_like(pix), "v", color=RED, ms=9)
    axs[0].set_title(f"16 random pixels: {len(hits)} of them see a spike", fontsize=FONT - 1)
    row = np.sqrt(2 / N) * np.cos(2 * np.pi * 9 * t / N) * 12
    axs[1].plot(t, row, color=GREEN, lw=2)
    axs[1].set_title("one Fourier measurement: a weighted sum that sees every spike", fontsize=FONT - 1)
    axs[1].set_xlim(0, N)
    fig.tight_layout()
    save(fig, U, "fig-incoherence.svg")
    print(f"  incoherence: random pixels hitting a spike: {len(hits)} of 16")


def fig_psf():
    reg = [j for j in range(N) if freq_of_row(j, N) % 4 == 0]
    rnd = np.sort(np.random.default_rng(57).choice(N, len(reg), replace=False))
    e = np.zeros(N); e[N // 2] = 1
    fig, axs = plt.subplots(1, 2, figsize=(11, 2.9), sharey=True)
    for a, rows, title in [(axs[0], reg, "regular mask: a few equal peaks"), (axs[1], rnd, "random mask: one peak + small sidelobes")]:
        P = F[rows]; psf = P.T @ (P @ e) * N / len(rows)
        a.plot(np.arange(N) - N // 2, psf, color=RED, lw=1.5); a.set_title(title, fontsize=FONT - 1)
        a.axhline(0, color=GREY, lw=0.5)
        if a is axs[1]: print(f"  psf random: largest sidelobe {np.sort(np.abs(psf))[-2]:.3f} (peak 1)")
    axs[0].set_ylabel("PSF (peak 1)")
    fig.tight_layout()
    save(fig, U, "fig-psf.svg")


def fig_phase():
    tmp = run_node(os.path.join(HERE, "data.ts"))
    d = load_json(tmp, "phase.json")
    G = d["G"]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
    for a, key, title in [(axs[0], "l1", "ℓ1 (basis pursuit)"), (axs[1], "omp", "OMP")]:
        v = np.array(d[key]).reshape(G, G)
        im = a.imshow(v, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", vmin=0, vmax=1, aspect="auto")
        a.plot(d["dt"]["delta"], d["dt"]["rho"], color="white", lw=5); a.plot(d["dt"]["delta"], d["dt"]["rho"], color=RED, lw=2.5, ls="--", label="theory: ℓ1, N → ∞")
        a.set_xlabel("δ = M/N"); a.set_title(f"{title}, N = {d['N']}, {d['T']} trials per cell", fontsize=FONT - 2)
        frac = v.mean()
        print(f"  phase {key}: mean success {frac:.3f}")
    axs[0].set_ylabel("ρ = K/M"); axs[0].legend(loc="upper left", fontsize=FONT - 4)
    fig.colorbar(im, ax=axs, fraction=0.025, label="fraction recovered")
    save(fig, U, "fig-phase.png", dpi=110)
    l1 = np.array(d["l1"]).reshape(G, G); om = np.array(d["omp"]).reshape(G, G)
    for dl in (0.25, 0.5, 0.75):
        i = int(dl * G)
        col1 = l1[:, i]; col2 = om[:, i]
        b1 = (np.argmax(col1 < 0.5) + 0.0) / G; b2 = (np.argmax(col2 < 0.5) + 0.0) / G
        print(f"  δ ≈ {(i + 0.5) / G:.3f}: first ρ cell with < 50 % success: ℓ1 {b1:.3f}, OMP {b2:.3f}")


if __name__ == "__main__":
    os.chdir(ROOT)
    fig_surprise(); fig_line(); fig_aliasing(); fig_iterations(); fig_incoherence(); fig_psf(); fig_phase()
