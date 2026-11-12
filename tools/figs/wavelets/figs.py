# Static figures for the wavelets day (SVG, text kept as text). Run: python3 -I figs.py
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
R = "/data/henriksson/github/course/imageanalysis/"
plt.rcParams.update({"font.size": 15, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "axes.titlesize": 16, "axes.titleweight": "bold"})
BLUE, RED, GREEN, GREY, ORANGE, PURPLE = "#1f77b4", "#d62728", "#2ca02c", "#7f7f7f", "#ff7f0e", "#9467bd"
S2 = np.sqrt(0.5)
H = {"haar": [S2, S2],
     "db2": [0.48296291314453414, 0.83651630373780791, 0.22414386804201338, -0.12940952255126038],
     "db4": [0.2303778133088965, 0.71484657055291565, 0.63088076792985891, -0.027983769416859854, -0.18703481171909308, 0.030841381835560764, 0.0328830116668852, -0.010597401785069032],
     "db8": [0.05441584224310401, 0.31287159091429997, 0.67563073629728981, 0.58535468365420671, -0.015829105256349306, -0.28401554296154693, 0.00047248457391328277, 0.12874742662047846, -0.017369301001807546, -0.044088253930794752, 0.013981027917398282, 0.0087460940474057767, -0.0048703529934515743, -0.00039174037337694705, 0.00067544940645056937, -0.00011747678412476953]}
def qmf(h): h = np.asarray(h); L = len(h); return np.array([(-1) ** n * h[L - 1 - n] for n in range(L)])
def dwt1(x, h, levels):
    h = np.asarray(h); g = qmf(h); c = np.array(x, float); n = len(c); L = len(h)
    for _ in range(levels):
        a = np.zeros(n // 2); d = np.zeros(n // 2)
        for k in range(n // 2):
            idx = [(2 * k + m) % n for m in range(L)]
            a[k] = h @ c[idx]; d[k] = g @ c[idx]
        c[:n // 2] = a; c[n // 2:n] = d; n //= 2
    return c
def idwt1(c, h, levels):
    h = np.asarray(h); g = qmf(h); x = np.array(c, float); L = len(h); n = len(x) >> levels
    for _ in range(levels):
        n *= 2; y = np.zeros(n)
        for k in range(n // 2):
            for m in range(L): y[(2 * k + m) % n] += h[m] * x[k] + g[m] * x[n // 2 + k]
        x[:n] = y
    return x
def realfourier(x):
    n = len(x); X = np.fft.fft(x) / np.sqrt(n); c = np.zeros(n); c[0] = X[0].real; c[n - 1] = X[n // 2].real
    for k in range(1, n // 2): c[2 * k - 1] = np.sqrt(2) * X[k].real; c[2 * k] = np.sqrt(2) * X[k].imag
    return c
def irealfourier(c):
    n = len(c); X = np.zeros(n, complex); X[0] = c[0]; X[n // 2] = c[n - 1]
    for k in range(1, n // 2): X[k] = (c[2 * k - 1] + 1j * c[2 * k]) / np.sqrt(2); X[n - k] = np.conj(X[k])
    return np.real(np.fft.ifft(X * np.sqrt(n)))
def keep(c, K): o = np.argsort(-np.abs(c), kind="stable"); r = np.zeros_like(c); r[o[:K]] = c[o[:K]]; return r
def save(fig, path): fig.savefig(R + path, bbox_inches="tight", pad_inches=0.05); plt.close(fig); print("wrote", path)

# ---- haar/fig-spike-step.svg ----
n = 64; t = np.arange(n)
spike = np.zeros(n); spike[23] = 1
step = (t >= 23).astype(float)
fig, ax = plt.subplots(2, 3, figsize=(13, 6.2))
for r, (x, name) in enumerate([(spike, "spike"), (step, "step")]):
    ax[r, 0].stem(t, x, linefmt="k-", markerfmt="ko", basefmt=" "); ax[r, 0].set_title(f"a {name}"); ax[r, 0].set_ylim(-0.3, 1.3)
    X = np.abs(np.fft.fft(x))[: n // 2 + 1]
    ax[r, 1].bar(np.arange(n // 2 + 1), X, color=RED); ax[r, 1].set_title("its Fourier magnitudes |X_k|")
    if r == 1:
        ax[r, 1].plot(np.arange(1, n // 2 + 1), n / (np.pi * np.arange(1, n // 2 + 1)), "k--", lw=1.5, label="64/(πk)"); ax[r, 1].set_ylim(0, 25); ax[r, 1].legend(frameon=False)
    K = 6
    rf = irealfourier(keep(realfourier(x), K)); rh = idwt1(keep(dwt1(x, H["haar"], 6), K), H["haar"], 6)
    ax[r, 2].plot(t, x, color="k", lw=1.5, label="signal")
    ax[r, 2].plot(t, rf, color=RED, lw=2.5, label="6 Fourier coeff.")
    ax[r, 2].plot(t, rh, color=BLUE, lw=2.5, ls="--", label="6 Haar coeff.")
    ax[r, 2].set_title("keep the 6 largest (10 %)"); ax[r, 2].set_ylim(-0.4, 1.4)
    if r == 0: ax[r, 2].legend(frameon=False, fontsize=13, loc="upper right")
for a in ax[1]: a.set_xlabel("position" if a is not ax[1, 1] else "frequency k")
ax[0, 1].set_xlabel("frequency k")
fig.tight_layout(); save(fig, "haar/fig-spike-step.svg")

# ---- haar/fig-tiling.svg ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for a in ax: a.set_xlim(0, 8); a.set_ylim(0, 8); a.set_xlabel("position"); a.set_ylabel("frequency"); a.set_xticks([]); a.set_yticks([])
for i in range(4):
    for j in range(4): ax[0].add_patch(Rectangle((2 * i, 2 * j), 2, 2, fill=True, fc="#dbe9f6", ec=BLUE, lw=2))
ax[0].set_title("short-time Fourier: one window size")
cols = ["#f2f2f2", "#dbe9f6", "#d9f0d3", "#efe2f5"]
bands = [(0, 1, 1), (1, 2, 1), (2, 4, 2), (4, 8, 4)]  # (f0, f1, number of tiles across = width 8/n)
widths = [8, 8, 4, 2]
for (f0, f1, _), w, c in zip(bands, widths, cols):
    for i in range(int(8 / w)): ax[1].add_patch(Rectangle((i * w, f0), w, f1 - f0, fc=c, ec="k", lw=1.8))
ax[1].text(4, 0.5, "approximation", ha="center", va="center", fontsize=13)
ax[1].set_title("wavelets: short windows for high frequencies")
fig.tight_layout(); save(fig, "haar/fig-tiling.svg")

# ---- haar/fig-haar-basis.svg ----
N = 8
M = np.array([dwt1(np.eye(N)[i], H["haar"], 3) for i in range(N)]).T  # rows = basis vectors
labels = ["s (average)", "d, level 3", "d, level 2", "d, level 2", "d, level 1", "d, level 1", "d, level 1", "d, level 1"]
colors = [GREY, PURPLE, GREEN, GREEN, BLUE, BLUE, BLUE, BLUE]
fig, ax = plt.subplots(N, 1, figsize=(7.5, 8.2), sharex=True)
for r in range(N):
    ax[r].bar(np.arange(N), M[r], color=colors[r], width=0.8); ax[r].axhline(0, color="k", lw=1)
    ax[r].set_ylim(-0.8, 0.8); ax[r].set_yticks([]); ax[r].spines["left"].set_visible(False)
    ax[r].text(-0.9, 0, labels[r], ha="right", va="center", fontsize=14, color=colors[r])
ax[-1].set_xticks(range(N)); ax[-1].set_xticklabels([str(i + 1) for i in range(N)]); ax[-1].set_xlabel("sample")
fig.subplots_adjust(left=0.27, hspace=0.25); save(fig, "haar/fig-haar-basis.svg")

# ---- filterbank/fig-mra.svg ----
n = 64; t = (np.arange(n) + 0.5) / n
f = np.sin(2 * np.pi * t) + 0.6 * (t > 0.6) + 0.5 * t
fig, ax = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
for a, j in zip(ax, [5, 3, 2, 1]):
    w = n // 2 ** j
    proj = np.repeat(f.reshape(2 ** j, w).mean(axis=1), w)
    a.plot(t, f, color=GREY, lw=1.5); a.step(np.r_[t - 0.5 / n, 1], np.r_[proj, proj[-1]], where="post", color=BLUE, lw=2.5)
    a.set_title(f"in $V_{j}$: {2 ** j} pieces"); a.set_xticks([0, 0.5, 1]); a.set_xlabel("t")
fig.tight_layout(); save(fig, "filterbank/fig-mra.svg")

# ---- filterbank/fig-filterbank.svg ----
fig, a = plt.subplots(figsize=(12, 4.2)); a.set_xlim(0, 12); a.set_ylim(0, 4.2); a.axis("off")
def box(x, y, txt, c):
    a.add_patch(FancyBboxPatch((x - 0.55, y - 0.35), 1.1, 0.7, boxstyle="round,pad=0.05", fc="white", ec=c, lw=2.5)); a.text(x, y, txt, ha="center", va="center", fontsize=17, color=c)
def arrow(x0, y0, x1, y1): a.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=2, color="k"))
a.text(0.4, 2.1, "x", fontsize=22, ha="center", va="center")
arrow(0.6, 2.1, 1.2, 2.1); a.plot([1.2, 1.2], [1.0, 3.2], "k", lw=2)
for y, c, flt in [(3.2, BLUE, "h"), (1.0, RED, "g")]:
    arrow(1.2, y, 1.85, y); box(2.4, y, f"filter {flt}", c); arrow(2.95, y, 3.55, y); box(4.1, y, "↓2", c); arrow(4.65, y, 5.35, y)
    a.text(5.95, y, "a" if flt == "h" else "d", fontsize=22, ha="center", va="center", color=c)
    arrow(6.45, y, 7.05, y); box(7.6, y, "↑2", c); arrow(8.15, y, 8.75, y); box(9.3, y, f"filter {flt}", c); arrow(9.85, y, 10.5, y)
a.plot([10.5, 10.5], [1.0, 3.2], "k", lw=2); a.add_patch(plt.Circle((10.5, 2.1), 0.25, fc="white", ec="k", lw=2)); a.text(10.5, 2.1, "+", ha="center", va="center", fontsize=20)
arrow(10.75, 2.1, 11.35, 2.1); a.text(11.65, 2.1, "x", fontsize=22, ha="center", va="center")
a.text(3.2, 4.0, "analysis: N numbers → N/2 + N/2", ha="center", fontsize=15); a.text(8.5, 4.0, "synthesis: back to N", ha="center", fontsize=15)
a.text(5.95, 2.1, "keep or\nchange", ha="center", va="center", fontsize=13, color=GREY)
save(fig, "filterbank/fig-filterbank.svg")

# ---- filterbank/fig-daubechies.svg ----
def cascade(h, it):
    h = np.asarray(h); g = qmf(h); L = len(h); v = np.array([1.0]); prev = v
    for i in range(it):
        prev = v; up = np.zeros(2 * (len(v) - 1) + 1); up[::2] = v; v = np.sqrt(2) * np.convolve(up, h)
    s = 2 ** (it - 1); psi = np.zeros(len(prev) + (L - 1) * s)
    for nn in range(L): psi[nn * s: nn * s + len(prev)] += np.sqrt(2) * g[nn] * prev
    tt = np.arange(len(v)) / 2 ** it
    return tt, v, psi[: len(v)]
fig, ax = plt.subplots(2, 4, figsize=(14, 5.4))
for c, name in enumerate(["haar", "db2", "db4", "db8"]):
    tt, phi, psi = cascade(H[name], 9)
    ax[0, c].plot(tt, phi, color=BLUE, lw=2); ax[1, c].plot(tt, psi, color=RED, lw=2)
    ax[0, c].set_title(("Haar = db1" if name == "haar" else name) + f": {len(H[name])} taps")
    for r in range(2): ax[r, c].axhline(0, color="k", lw=0.8); ax[r, c].set_xlim(0, len(H[name]) - 1)
ax[0, 0].set_ylabel("φ (scaling)"); ax[1, 0].set_ylabel("ψ (wavelet)")
for c in range(4): ax[1, c].set_xlabel("t")
fig.tight_layout(); save(fig, "filterbank/fig-daubechies.svg")

# ---- dwt2d/fig-subbands.svg ----
n = 128
img = np.full((n, n), 40.0)
yy, xx = np.mgrid[0:n, 0:n]
img[(xx >= 21) & (xx < 31) & (yy >= 13) & (yy < 115)] = 210
img[(yy >= 21) & (yy < 31) & (xx >= 45) & (xx < 115)] = 210
img[(np.abs(xx - yy - 15) < 6) & (xx > 50) & (yy > 40) & (xx < 117) & (yy < 117)] = 210
def haar2(a):
    lo = (a[:, 0::2] + a[:, 1::2]) / np.sqrt(2); hi = (a[:, 0::2] - a[:, 1::2]) / np.sqrt(2)
    r = np.hstack([lo, hi]); lo = (r[0::2] + r[1::2]) / np.sqrt(2); hi = (r[0::2] - r[1::2]) / np.sqrt(2)
    return np.vstack([lo, hi])
c = haar2(img); m = n // 2
fig, ax = plt.subplots(1, 2, figsize=(12, 5.8))
ax[0].imshow(img, cmap="gray", vmin=0, vmax=255); ax[0].set_title("image"); ax[0].axis("off")
rgb = np.zeros((n, n, 3))
LL = c[:m, :m]; rgb[:m, :m] = ((LL - LL.min()) / (LL.max() - LL.min()))[..., None]
cmap = plt.get_cmap("RdBu_r")
for (y0, x0) in [(0, m), (m, 0), (m, m)]:
    b = c[y0:y0 + m, x0:x0 + m]; mx = np.abs(b).max() or 1; rgb[y0:y0 + m, x0:x0 + m] = cmap(0.5 + 0.5 * b / mx)[..., :3]
ax[1].imshow(rgb, interpolation="nearest"); ax[1].axis("off"); ax[1].set_title("one level of 2D Haar")
ax[1].axhline(m - 0.5, color="#e0a000", lw=2); ax[1].axvline(m - 0.5, color="#e0a000", lw=2)
for (x, y, s) in [(m / 2, m / 2, "LL"), (1.5 * m, m / 2, "V: vertical edges"), (m / 2, 1.5 * m, "H: horizontal edges"), (1.5 * m, 1.5 * m, "D: diagonal")]:
    ax[1].text(x, y + m * 0.42, s, ha="center", va="center", fontsize=15, color="k", bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
fig.tight_layout(); save(fig, "dwt2d/fig-subbands.svg")

# ---- dwt2d/fig-histogram.svg (cell, db4, 4 levels, level-1 details; coefficients from the demo code) ----
N = 256
cd = np.fromfile("cell_db4_L4.f64").reshape(N, N)
d1 = np.concatenate([cd[:128, 128:].ravel(), cd[128:, :128].ravel(), cd[128:, 128:].ravel()])
fig, ax = plt.subplots(figsize=(8, 4.2))
bins = np.linspace(-150, 150, 121)
cnt, e = np.histogram(d1, bins)
ax.bar((e[:-1] + e[1:]) / 2, cnt, width=e[1] - e[0], color=PURPLE, alpha=0.75, label="level-1 details of the micrograph")
sd = d1.std(); xs = np.linspace(-150, 150, 400)
ax.plot(xs, len(d1) * (e[1] - e[0]) * np.exp(-xs ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)), "k--", lw=2, label=f"Gaussian, same SD ({sd:.1f})")
ax.set_yscale("log"); ax.set_ylim(0.8, cnt.max() * 2); ax.set_xlabel("coefficient value"); ax.set_ylabel("count (log)")
ax.legend(frameon=False, fontsize=13, loc="upper right")
fig.tight_layout(); save(fig, "dwt2d/fig-histogram.svg")
print("level-1 sd", sd, "fraction |w|<10", np.mean(np.abs(d1) < 10), "max", np.abs(d1).max(), "count beyond 4 sd", np.mean(np.abs(d1) > 4 * sd), "gauss would be", 6.3e-5)

# ---- shrinkage/fig-thresholds.svg ----
fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
w = np.linspace(-5, 5, 401); tt = 2
ax[0].plot(w, np.where(np.abs(w) > tt, w, 0), color=RED, lw=3); ax[0].set_title("hard: keep or kill")
ax[1].plot(w, np.sign(w) * np.maximum(np.abs(w) - tt, 0), color=BLUE, lw=3); ax[1].set_title("soft: kill and shrink")
for a in ax:
    a.plot(w, w, color=GREY, ls=":", lw=1.5); a.axhline(0, color="k", lw=0.8); a.axvline(0, color="k", lw=0.8)
    a.axvline(-tt, color=GREY, ls="--"); a.axvline(tt, color=GREY, ls="--"); a.text(tt + 0.1, -4.6, "t", fontsize=16); a.text(-tt - 0.5, -4.6, "−t", fontsize=16)
    a.set_xlabel("coefficient w"); a.set_ylabel("after thresholding"); a.set_xlim(-5, 5); a.set_ylim(-5, 5)
fig.tight_layout(); save(fig, "shrinkage/fig-thresholds.svg")
