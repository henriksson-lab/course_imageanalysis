"""Figures for cs-mri/lec.qmd (sparse representations day, unit 4).
Run from the repository root:  python3 -I tools/figs/cs-mri/figs.py
The phantom (Shepp–Logan, synthetic), masks and reconstructions come from the csmri demo's code (data.ts).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sparsity"))
from common import *   # noqa: F401,F403

U = "cs-mri"


def shift(a, n):
    return np.fft.fftshift(np.array(a).reshape(n, n))


def main():
    tmp = run_node(os.path.join(HERE, "data.ts"))
    d = load_json(tmp, "mri.json")
    n = d["n"]
    shepp = np.array(d["shepp"]).reshape(n, n)
    for k in ["centre", "regular", "vardens", "random", "les4", "les8"]:
        r = d[k]; print(f"  {k}: sampled {r['frac']:.3f}, PSNR zero-filled {r['psnrZf']:.2f} dB, CS {r['psnrCs']:.2f} dB, SSIM {r['ssimZf']:.3f} / {r['ssimCs']:.3f}")

    # 1. object and k-space
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.8))
    axs[0].imshow(shepp, cmap="gray", vmin=0, vmax=1); axs[0].set_title("object (synthetic head phantom)", fontsize=FONT - 2)
    K = shift(d["klog"], n)
    axs[1].imshow(K, cmap="viridis"); axs[1].set_title("its k-space (log |Fourier|)", fontsize=FONT - 2)
    row = n // 2 + 30
    axs[1].axhline(row, color=RED, lw=2.5); axs[1].text(8, row - 6, "one line = one readout", color="white", fontsize=FONT - 3)
    axs[1].set_xlabel("k_x (frequency encoding)"); axs[1].set_ylabel("k_y (phase encoding)")
    for a in axs: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); save(fig, U, "fig-kspace.png", dpi=100)

    # 2. scan time vs acceleration
    acc = np.array([1, 2, 4, 6, 8])
    tmin = 256 * 128 * 0.010 / 60 / acc
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar([str(a) + "×" for a in acc], tmin, color=BLUE)
    for i, t in enumerate(tmin): ax.text(i, t + 0.1, f"{t * 60:.0f} s" if t < 1 else f"{t:.1f} min", ha="center", fontsize=FONT - 3)
    ax.set_ylabel("scan time (min)"); ax.set_xlabel("acceleration"); ax.set_ylim(0, 6.3)
    ax.set_title("256 × 128 lines, 10 ms per line", fontsize=FONT - 2)
    save(fig, U, "fig-time.svg")
    print("  times (s):", (tmin * 60).round(1))

    # 3. three ways to skip lines
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 7.2))
    for j, (k, title) in enumerate([("centre", "centre only (4×)"), ("regular", "every 2nd line (2×)"), ("vardens", "random, denser at the centre (4×)")]):
        axs[0, j].imshow(shift(d[k]["mask"], n), cmap="gray", vmin=0, vmax=1); axs[0, j].set_title(title, fontsize=FONT - 2)
        axs[1, j].imshow(np.array(d[k]["zf"]).reshape(n, n), cmap="gray", vmin=0, vmax=1)
        axs[1, j].set_title(f"zero-filled: {['blurred', 'folded', 'noise-like'][j]}", fontsize=FONT - 2)
    for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); save(fig, U, "fig-masks.png", dpi=90)

    # 4. reconstruction
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.3))
    v = d["vardens"]
    for a, img, title in [(axs[0], shepp, "fully sampled"), (axs[1], np.array(v["zf"]).reshape(n, n), f"zero-filled, {v['frac'] * 100:.0f} % of k-space: {v['psnrZf']:.1f} dB"),
                          (axs[2], np.array(v["cs"]).reshape(n, n), f"CS (Haar ℓ1): {v['psnrCs']:.1f} dB")]:
        a.imshow(img, cmap="gray", vmin=0, vmax=1); a.set_title(title, fontsize=FONT - 3); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); save(fig, U, "fig-recon.png", dpi=90)

    # 5. variable density
    kk = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for q, col in [(1, GREEN), (3, BLUE), (6, PURPLE)]:
        ax.plot(kk, (1 - kk) ** q, color=col, lw=2.5, label=f"q = {q}")
    ax.axvspan(0, 16 / 128, color=ORANGE, alpha=0.25); ax.text(0.01, 1.03, "fully\nsampled\ncentre", fontsize=FONT - 4, color=ORANGE, va="bottom")
    ax.set_xlabel("|k| / k_max"); ax.set_ylabel("relative probability of a line"); ax.set_ylim(0, 1.35)
    ax.legend(frameon=False)
    save(fig, U, "fig-density.svg")

    # 6. lesions
    les = np.array(d["lesions"]).reshape(n, n)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.3))
    sl = (slice(30, 140), slice(40, 216))
    for a, img, title in [(axs[0], les, "truth: 5 faint lesions (contrast 2–5 %)"), (axs[1], np.array(d["les4"]["cs"]).reshape(n, n), f"CS at 4×: {d['les4']['psnrCs']:.1f} dB"),
                          (axs[2], np.array(d["les8"]["cs"]).reshape(n, n), f"CS at 8×: {d['les8']['psnrCs']:.1f} dB")]:
        a.imshow(img, cmap="gray", vmin=0.15, vmax=0.4); a.set_title(title, fontsize=FONT - 3); a.set_xticks([]); a.set_yticks([])
    fig.text(0.5, 0.01, "display window 0.15–0.40 (the brain region) to show the low-contrast lesions", ha="center", fontsize=11, color=GREY)
    fig.tight_layout(); save(fig, U, "fig-lesions.png", dpi=90)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
