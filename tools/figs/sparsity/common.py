"""Shared helpers for the figure scripts of the sparse representations day
(tools/figs/sparsity, ista, compressed-sensing, cs-mri, dictionaries).

numpy versions of the small algorithms (soft threshold, orthonormal Haar, DCT, ISTA/FISTA, OMP),
plot style, and a runner for the node data scripts (data.ts) that compute the 2D results with the
demos' own TypeScript code (demos/src/lib/sparse.ts, patches.ts).
Image: transmission electron micrograph of a mesenchymal stem cell, Robert M. Hunt, CC BY 3.0,
via Wikimedia Commons (demos/src/fourier2d/cell256.png).
"""
import os, json, subprocess, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import dct, idct

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
FONT = 17
plt.rcParams.update({
    "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
    "xtick.labelsize": FONT - 2, "ytick.labelsize": FONT - 2, "legend.fontsize": FONT - 2,
    "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, RED, GREEN, ORANGE, PURPLE, GREY, BLACK = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#777777", "#222222"
CREDIT = "Micrograph: R. M. Hunt, CC BY 3.0"


def cell():
    from PIL import Image
    return np.array(Image.open(os.path.join(ROOT, "demos/src/fourier2d/cell256.png"))).astype(np.float64)


def save(fig, unit, name, **kw):
    path = os.path.join(ROOT, unit, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05, **kw)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def soft(w, t):
    return np.sign(w) * np.maximum(np.abs(w) - t, 0)


def hard(w, t):
    return np.where(np.abs(w) > t, w, 0.0)


# ---- orthonormal Haar (periodic, Mallat layout), 1D and 2D -------------------------------
def haar1(x, levels):
    c = np.array(x, dtype=float)
    n = len(c)
    for _ in range(levels):
        a = (c[0:n:2] + c[1:n:2]) / np.sqrt(2)
        d = (c[0:n:2] - c[1:n:2]) / np.sqrt(2)
        c[:n // 2], c[n // 2:n] = a, d
        n //= 2
    return c


def ihaar1(c, levels):
    x = np.array(c, dtype=float)
    n = len(x) >> levels
    for _ in range(levels):
        a, d = x[:n].copy(), x[n:2 * n].copy()
        x[0:2 * n:2] = (a + d) / np.sqrt(2)
        x[1:2 * n:2] = (a - d) / np.sqrt(2)
        n *= 2
    return x


def haar2(img, levels):
    c = np.array(img, dtype=float)
    h, w = c.shape
    for _ in range(levels):
        blk = c[:h, :w]
        blk = np.apply_along_axis(lambda r: haar1(r, 1), 1, blk)
        blk = np.apply_along_axis(lambda r: haar1(r, 1), 0, blk)
        c[:h, :w] = blk
        h //= 2; w //= 2
    return c


def dct1(x):
    return dct(x, norm="ortho")


def idct1(c):
    return idct(c, norm="ortho")


# ---- solvers -------------------------------------------------------------------------------
def ista(A, y, lam, iters, tau=None, x0=None, fista=False):
    L = np.linalg.norm(A, 2) ** 2
    tau = tau or 1 / L
    x = np.zeros(A.shape[1]) if x0 is None else x0.copy()
    z, t, hist, xs = x.copy(), 1.0, [], []
    for _ in range(iters):
        g = z + tau * A.T @ (y - A @ z)
        xn = soft(g, tau * lam)
        if fista:
            tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
            z = xn + (t - 1) / tn * (xn - x)
            t = tn
        else:
            z = xn
        x = xn
        xs.append(x.copy())
        hist.append(0.5 * np.sum((A @ x - y) ** 2) + lam * np.sum(np.abs(x)))
    return x, np.array(hist), xs


def omp(A, y, K, tol=0.0):
    r, S, steps = y.copy(), [], []
    x = np.zeros(A.shape[1])
    cn = np.linalg.norm(A, axis=0)
    while len(S) < K and np.linalg.norm(r) > tol:
        j = int(np.argmax(np.abs(A.T @ r) / cn))
        if j in S:
            break
        S.append(j)
        xs, *_ = np.linalg.lstsq(A[:, S], y, rcond=None)
        x = np.zeros(A.shape[1]); x[S] = xs
        r = y - A @ x
        steps.append((x.copy(), r.copy(), j))
    return x, steps


def gauss_blur_matrix(n, sigma):
    r = int(np.ceil(4 * sigma))
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma ** 2)); k /= k.sum()
    B = np.zeros((n, n))
    for i in range(n):
        for j, kv in zip(range(-r, r + 1), k):
            B[i, (i + j) % n] += kv
    return B


def run_node(data_ts, args=(), loader_png=False):
    """Bundle data.ts with esbuild and run it with node; it writes JSON to the returned directory."""
    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, "data.js")
    cmd = [os.path.join(ROOT, "node_modules/.bin/esbuild"), data_ts, "--bundle", "--platform=node", "--log-level=warning", "--outfile=" + out]
    subprocess.run(cmd, check=True)
    subprocess.run(["node", out, tmp, *args], check=True, cwd=ROOT)
    return tmp


def load_json(tmp, name):
    with open(os.path.join(tmp, name)) as f:
        return json.load(f)
