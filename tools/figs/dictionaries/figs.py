"""Figures for dictionaries/lec.qmd (sparse representations day, unit 5).
Run from the repository root:  python3 -I tools/figs/dictionaries/figs.py
Dictionaries, codes and restorations come from the demos' code (data.ts, via node + esbuild);
the 2D k-means / K-SVD picture is computed here with numpy.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sparsity"))
from common import *   # noqa: F401,F403

U = "dictionaries"
P = 8


def atom_grid(D, k, cols, order=None, signed=False):
    D = np.array(D).reshape(k, P, P)
    order = range(k) if order is None else order
    rows = int(np.ceil(k / cols)); s = P + 1
    G = np.ones((rows * s + 1, cols * s + 1))
    for slot, j in enumerate(order):
        a = D[j]; lo, hi = a.min(), a.max()
        v = (a - lo) / (hi - lo) if hi > lo else np.full_like(a, 0.5)
        r, c = divmod(slot, cols)
        G[1 + r * s:1 + r * s + P, 1 + c * s:1 + c * s + P] = v
    return G


def main():
    tmpc = tempfile.mkdtemp(); cp = os.path.join(tmpc, "cell.f64"); cell().tofile(cp)
    tmp = run_node(os.path.join(HERE, "data.ts"), [cp])
    d = load_json(tmp, "dict.json")
    n = 128

    # 1. a patch written as a few atoms
    c = d["code"]
    fig, axs = plt.subplots(1, 6, figsize=(12, 2.6), gridspec_kw=dict(width_ratios=[1, 0.25, 1, 1, 1, 1]))
    def show(a, v, title, signed=False):
        v = np.array(v).reshape(P, P)
        if signed: m = np.abs(v).max() or 1; a.imshow(v, cmap="RdBu_r", vmin=-m, vmax=m)
        else: a.imshow(v, cmap="gray")
        a.set_title(title, fontsize=FONT - 2); a.set_xticks([]); a.set_yticks([])
    show(axs[0], c["patch"], "8 × 8 patch")
    axs[1].axis("off"); axs[1].text(0.5, 0.5, "≈", fontsize=34, ha="center", va="center")
    for a, j, v, at in zip(axs[2:5], c["idx"], c["val"], c["atoms"]):
        show(a, at, f"{v:+.0f} × atom {j}", signed=True)
    rec = sum(v * np.array(at) for v, at in zip(c["val"], c["atoms"]))
    show(axs[5], rec, "= sum of the three")
    fig.text(0.5, -0.02, f"atom 0 is the constant patch (the mean); residual {np.sqrt(c['err'] / 64):.1f} grey levels RMS per pixel", ha="center", fontsize=12, color=GREY)
    save(fig, U, "fig-patchcode.svg")
    print("  code:", c["idx"], np.round(c["val"], 1), "rms residual", round(float(np.sqrt(c["err"] / 64)), 2))

    # 2. overcomplete DCT
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.imshow(atom_grid(d["odct"], 256, 16), cmap="gray", interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("256 atoms for 8 × 8 patches (64 pixels)", fontsize=FONT - 2)
    save(fig, U, "fig-odct.png", dpi=100)

    # 3. learning: atoms after 0, 5, 20 iterations
    L = d["learning"]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.8), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.2]))
    for a, it in zip(axs[:3], ["0", "5", "20"]):
        a.imshow(atom_grid(L["snaps"][it], 64, 8), cmap="gray", interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"after {it} iterations" if it != "0" else "start: random patches", fontsize=FONT - 2)
    axs[3].plot(L["errs"], "o-", color=BLUE, ms=4); axs[3].set_xlabel("iteration"); axs[3].set_ylabel("error per pixel (grey²)")
    axs[3].set_title("representation error, K = 4", fontsize=FONT - 2)
    fig.tight_layout(); save(fig, U, "fig-learning.png", dpi=90)
    print("  learning errors:", np.round(L["errs"][:1] + L["errs"][5:6] + L["errs"][-1:], 1))

    # 4. k-means vs K-SVD in 2D
    r = np.random.default_rng(5)
    dirs = np.array([[1, 0.2], [0.3, 1], [-0.8, 0.9]]); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    pts = []
    for dv in dirs:
        a = r.normal(0, 2.0, 60); pts.append(a[:, None] * dv + 0.12 * r.normal(size=(60, 2)))
    X = np.vstack(pts)
    # k-means, 3 centres
    C = X[r.choice(len(X), 3, replace=False)].copy()
    for _ in range(50):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), 1)
        C = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(3)])
    # K-SVD with K = 1: atom = direction, code = best single atom with any coefficient
    D = X[r.choice(len(X), 3, replace=False)].T.copy(); D /= np.linalg.norm(D, axis=0)
    for _ in range(50):
        corr = D.T @ X.T; lab = np.argmax(np.abs(corr), 0)
        for j in range(3):
            Xj = X[lab == j]
            if len(Xj): u, s, vt = np.linalg.svd(Xj.T, full_matrices=False); D[:, j] = u[:, 0]
    errk = np.mean(((X - C[lab]) ** 2).sum(1)) if False else np.mean(np.min(((X[:, None, :] - C[None]) ** 2).sum(-1), 1))
    errs = np.mean(np.sum(X ** 2, 1) - np.max((D.T @ X.T) ** 2, 0))
    print(f"  2D: k-means error {errk:.3f}, K-SVD (K = 1) error {errs:.3f}")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    for a in axs: a.plot(X[:, 0], X[:, 1], ".", color=GREY, ms=5); a.set_aspect("equal"); a.axhline(0, color=BLACK, lw=0.5); a.axvline(0, color=BLACK, lw=0.5)
    axs[0].plot(C[:, 0], C[:, 1], "o", color=RED, ms=13, mec="white", mew=2)
    axs[0].set_title(f"k-means: 3 centres (mean sq. error {errk:.2f})", fontsize=FONT - 2)
    for j in range(3): axs[1].plot([-6 * D[0, j], 6 * D[0, j]], [-6 * D[1, j], 6 * D[1, j]], color=BLUE, lw=2.5)
    axs[1].set_title(f"K-SVD, K = 1: 3 directions ({errs:.2f})", fontsize=FONT - 2)
    axs[0].set_xlim(-6, 6); axs[0].set_ylim(-6, 6)
    save(fig, U, "fig-kmeans.svg")

    # 5. PCA vs learned
    pc = d["pca"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    axs[0].imshow(atom_grid(pc["atoms"], 16, 4), cmap="gray", interpolation="nearest"); axs[0].set_title("PCA: first 16 components", fontsize=FONT - 2)
    order = np.argsort(pc["usage"])[::-1][:16]
    axs[1].imshow(atom_grid(pc["learned"], 128, 4, order=order), cmap="gray", interpolation="nearest"); axs[1].set_title("K-SVD: 16 most used of 128", fontsize=FONT - 2)
    for a in axs[:2]: a.set_xticks([]); a.set_yticks([])
    k = np.arange(1, 9)
    axs[2].plot(k, pc["curves"]["pca"], "o-", color=ORANGE, lw=2.4, label="PCA, first K (linear)")
    axs[2].plot(k, pc["curves"]["dct"], "o-", color=PURPLE, lw=2.4, label="DCT 256, best K")
    axs[2].plot(k, pc["curves"]["learned"], "o-", color=BLUE, lw=2.4, label="learned 128, best K")
    axs[2].set_xlabel("K coefficients per patch"); axs[2].set_ylabel("error per pixel (grey²)"); axs[2].legend(frameon=False, fontsize=FONT - 4)
    fig.tight_layout(); save(fig, U, "fig-pca.png", dpi=90)
    print("  curves at K = 1, 4, 8:", {kk: [round(v[i], 1) for i in (0, 3, 7)] for kk, v in pc["curves"].items()})

    # 6. atoms of a texture and of the micrograph
    tx = d["texture"]; ca = d["cellAtoms"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    axs[0].imshow(np.array(tx["image"]).reshape(n, n), cmap="gray"); axs[0].set_title("texture image (synthetic)", fontsize=FONT - 2)
    axs[1].imshow(atom_grid(tx["atoms"], 64, 8, order=np.argsort(tx["usage"])[::-1]), cmap="gray", interpolation="nearest"); axs[1].set_title("its learned atoms", fontsize=FONT - 2)
    axs[2].imshow(atom_grid(ca["atoms"], 64, 8, order=np.argsort(ca["usage"])[::-1]), cmap="gray", interpolation="nearest"); axs[2].set_title("atoms of the micrograph", fontsize=FONT - 2)
    for a in axs: a.set_xticks([]); a.set_yticks([])
    fig.text(0.5, 0.01, CREDIT + " (right)", ha="center", fontsize=10, color=GREY)
    fig.tight_layout(); save(fig, U, "fig-atoms.png", dpi=90)

    # 7. denoising
    dn = d["denoise"]
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 8.2))
    items = [("truth", "truth"), ("noisy", "noisy, σ = 20"), ("wavelet", "wavelet shrinkage"), ("nlm", "NL-means"), ("dct", "8 × 8 DCT dictionary"), ("learned", "learned dictionary")]
    for a, (key, title) in zip(axs.ravel(), items):
        a.imshow(np.array(dn[key]).reshape(n, n), cmap="gray", vmin=0, vmax=255); a.set_xticks([]); a.set_yticks([])
        a.set_title(title + (f": {dn['psnr'][key]:.1f} dB" if key in dn["psnr"] else ""), fontsize=FONT - 2)
    fig.text(0.5, 0.005, CREDIT, ha="center", fontsize=10, color=GREY)
    fig.tight_layout(); save(fig, U, "fig-denoise.png", dpi=85)
    print("  denoise:", {k: round(v, 2) for k, v in dn["psnr"].items()}, dn["params"])

    # 8. inpainting
    ip = d["inpaint"]; truth = np.array(dn["truth"]).reshape(n, n)
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 8.0))
    cmap = plt.get_cmap("gray").copy(); cmap.set_bad("#ffcc33")
    for row, key, title in [(0, "random", "50 % missing"), (1, "scratch", "scratches + 16 × 16 hole")]:
        m = np.array(ip[key]["mask"]).reshape(n, n).astype(bool)
        axs[row, 0].imshow(np.where(m, truth, np.nan), cmap=cmap, vmin=0, vmax=255); axs[row, 0].set_title(title, fontsize=FONT - 2)
        axs[row, 1].imshow(np.array(ip[key]["wavelet"]).reshape(n, n), cmap="gray", vmin=0, vmax=255); axs[row, 1].set_title(f"wavelet ℓ1: {ip[key]['psnrWavelet']:.1f} dB", fontsize=FONT - 2)
        axs[row, 2].imshow(np.array(ip[key]["patch"]).reshape(n, n), cmap="gray", vmin=0, vmax=255); axs[row, 2].set_title(f"learned patches: {ip[key]['psnrPatch']:.1f} dB", fontsize=FONT - 2)
        print(f"  inpaint {key}: wavelet {ip[key]['psnrWavelet']:.2f} dB, patches {ip[key]['psnrPatch']:.2f} dB")
    for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.text(0.5, 0.005, CREDIT, ha="center", fontsize=10, color=GREY)
    fig.tight_layout(); save(fig, U, "fig-inpaint.png", dpi=85)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
