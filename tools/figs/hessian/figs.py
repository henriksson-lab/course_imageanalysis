"""Static figures for the hessian/ deck (slide pictures).

Run from the repository root:  python3 -I tools/figs/hessian/figs.py
Writes hessian/fig-curvatures.svg, fig-trace-det.png, fig-frangi-plane.svg, fig-ridges-cell.png.
Hessian and Frangi as in demos/src/lib/edges.ts (numpy port in tools/figs/canny/edgelib.py).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "canny"))
import numpy as np
import matplotlib.pyplot as plt
from edgelib import (hessian, frangi, frangi_value, geom_scales, cell, show,
                     BLUE, ORANGE, GREEN, RED, CYAN, GREY, PURPLE, FONT)

OUT = "hessian"


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.05, **kw)
    plt.close(fig)
    print("wrote", name)


# 1. Curvature as a function of direction --------------------------------------------------
def fig_curvatures():
    th = np.linspace(0, 180, 361)
    t = np.radians(th)
    # u = (cos t, sin t); curvature u^T H u for H = diag(a, c): a cos^2 + c sin^2
    cases = [("hilltop −(x² + y²)", -2, -2, GREEN), ("ridge −x²", -2, 0, RED), ("saddle −x² + y²", -2, 2, BLUE)]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for lab, a, c, col in cases:
        k = a * np.cos(t) ** 2 + c * np.sin(t) ** 2
        ax.plot(th, k, color=col, lw=2.6, label=lab)
        for ang, lam in [(0, a), (90, c)]:
            ax.plot([ang], [lam], "o", color=col, ms=9, zorder=3)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xticklabels(["0°\n(along x)", "45°", "90°\n(along y)", "135°", "180°"])
    ax.set_xlabel("direction u")
    ax.set_ylabel("curvature uᵀHu")
    ax.set_ylim(-2.6, 3.4)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=FONT - 2)
    ax.text(178, 2.9, "dots: extremes = eigenvalues", ha="right", fontsize=FONT - 3, color=GREY)
    save(fig, "fig-curvatures.svg")


# 2. Trace vs determinant on a blob and a line --------------------------------------------
def fig_trace_det():
    h, w = 80, 160
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    img = 40 + 120 * np.exp(-((xx - 40) ** 2 + (yy - 40) ** 2) / (2 * 3.0 ** 2))
    img += 120 * np.exp(-((xx - 115) ** 2) / (2 * 1.5 ** 2))
    s = 3
    H = hessian(img, s, True)
    tr = H["l1"] + H["l2"]
    det = H["l1"] * H["l2"]
    vb = (tr[40, 40], det[40, 40]); vl = (tr[40, 115], det[40, 115])
    print("  trace/det (sigma^2-normalised) blob", np.round(vb, 1), "line", np.round(vl, 1),
          "lambdas blob", round(H["l1"][40, 40], 1), round(H["l2"][40, 40], 1),
          "line", round(H["l1"][40, 115], 1), round(H["l2"][40, 115], 1))
    fig, axs = plt.subplots(1, 3, figsize=(16, 3.4), gridspec_kw={"wspace": 0.06})
    show(axs[0], img, "image: a spot and a line")
    m = np.abs(tr).max()
    show(axs[1], -tr, "−trace (LoG): both light up", cmap="magma", vmin=0, vmax=m)
    show(axs[2], np.maximum(det, 0), "determinant: only the spot", cmap="magma", vmin=0, vmax=det.max())
    for ax, vals in [(axs[1], (vb[0], vl[0])), (axs[2], (vb[1], vl[1]))]:
        for (x, y), v in zip([(40, 40), (115, 40)], vals):
            ax.text(x + (10 if x < 100 else 6), y + 25, f"{v:.0f}".replace("-", "−"), color="w", fontsize=FONT,
                    ha="left")
    fig.text(0.5, 0.0, "σ = 3, values × σ²", ha="center", fontsize=FONT - 2, color=GREY)
    save(fig, "fig-trace-det.png", dpi=110)


# 3. Frangi vesselness over the eigenvalue plane --------------------------------------------
def fig_frangi_plane():
    beta, c = 0.5, 15
    l2 = np.linspace(-40, 0, 401)
    l1 = np.linspace(-40, 40, 801)
    L2, L1 = np.meshgrid(l2, l1)
    V = frangi_value(L1, L2, beta, c)
    V = np.where(np.abs(L1) <= np.abs(L2), V, np.nan)
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(V, extent=[l2[0], l2[-1], l1[0], l1[-1]], origin="lower", cmap="viridis", vmin=0, vmax=1,
                   aspect="auto")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("vesselness")
    pts = [((-20, 0), "line", (-26, 12)), ((-10, -10), "blob", (-12, -26)), ((-20.23, -2.77), "example", (-38, -16))]
    for (a, b), lab, (tx, ty) in pts:
        v = float(frangi_value(b, a, beta, c))
        ax.plot([a], [b], "o", color="w", mec="k", ms=9)
        ax.annotate(f"{lab}: {v:.2f}", xy=(a, b), xytext=(tx, ty), color="w" if lab != "blob" else "k",
                    fontsize=FONT, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="w" if lab != "blob" else "k", lw=1.2))
        print("  frangi", lab, a, b, round(v, 3))
    ax.set_xlabel("λ₂ (strong curvature)")
    ax.set_ylabel("λ₁ (weak curvature)")
    ax.set_title("β = 0.5, c = 15, bright lines", fontsize=FONT)
    save(fig, "fig-frangi-plane.svg")


# 4. Membranes in the micrograph ---------------------------------------------------------------
def fig_ridges_cell():
    img = cell()[40:216, 40:216]
    sig = geom_scales(1, 4, 5)
    V, _ = frangi(img, sig, 0.5, 20, bright=False)
    print("  ridges cell: max vesselness", round(V.max(), 2))
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 4.2), gridspec_kw={"wspace": 0.05})
    show(axs[0], img, "micrograph")
    show(axs[1], V, "vesselness, dark lines", cmap="magma", vmin=0, vmax=1)
    fig.text(0.5, 0.08, "Micrograph: R. M. Hunt, CC BY 3.0, Wikimedia Commons",
             ha="center", va="top", fontsize=FONT - 4, color=GREY)
    save(fig, "fig-ridges-cell.png", dpi=110)


if __name__ == "__main__":
    fig_curvatures()
    fig_trace_det()
    fig_frangi_plane()
    fig_ridges_cell()
