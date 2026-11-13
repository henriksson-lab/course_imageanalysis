# Extra slide figures for filterbank/lec.qmd. Run: python3 -I tools/figs/filterbank/figs.py
# Shares the helpers (filters, dwt1, colours, rcParams) of tools/figs/wavelets/figs.py; the
# test signals and boundary rules follow demos/src/lib/wavelet.ts (testSignal, analysisStep).
import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "..", "wavelets", "figs.py")).read().split("# ---- haar/fig-spike-step.svg ----")[0])

def test_signal(name, n):
    t = np.arange(n) / n
    if name == "chirp": return np.sin(2 * np.pi * (2 * t + 30 * t * t))
    if name == "ramp": return t.copy()
    if name == "heavisine": return 4 * np.sin(4 * np.pi * t) - np.sign(t - 0.3) - np.sign(0.72 - t)
    raise ValueError(name)

def ext(i, n, mode):
    if mode == "periodic": return i % n
    if mode == "zero": return i if 0 <= i < n else -1
    j = i % (2 * n); return 2 * n - 1 - j if j >= n else j

def analysis(x, h, mode="periodic"):
    h = np.asarray(h); g = qmf(h); n = len(x); L = len(h)
    if mode == "periodic": k0, cnt = 0, n // 2
    else: k0 = int(np.ceil(-(L - 1) / 2)); cnt = (n - 1) // 2 - k0 + 1
    a = np.zeros(cnt); d = np.zeros(cnt)
    for j in range(cnt):
        s = 2 * (k0 + j)
        v = np.array([0.0 if ext(s + m, n, mode) < 0 else x[ext(s + m, n, mode)] for m in range(L)])
        a[j] = h @ v; d[j] = g @ v
    return a, d, k0

def synthesis(a, d, h, n):  # periodic
    h = np.asarray(h); g = qmf(h); L = len(h); y = np.zeros(n)
    for k in range(len(a)):
        for m in range(L): y[(2 * k + m) % n] += h[m] * a[k] + g[m] * d[k]
    return y

# ---- filterbank/fig-psijk.svg: phi, psi and the squeezed/shifted psi_{j,k} ----
t = np.linspace(0, 1, 1601)[:-1]
psi = lambda u: np.where((u >= 0) & (u < 0.5), 1.0, np.where((u >= 0.5) & (u < 1), -1.0, 0.0))
rows = [("φ", [np.ones_like(t)], [GREY]), ("ψ = ψ₀,₀", [psi(t)], [PURPLE]),
        ("ψ₁,₀  ψ₁,₁", [np.sqrt(2) * psi(2 * t), np.sqrt(2) * psi(2 * t - 1)], [GREEN, "#1b7837"]),
        ("ψ₂,₀ … ψ₂,₃", [2 * psi(4 * t - k) for k in range(4)], [BLUE, "#0b4f8a", BLUE, "#0b4f8a"])]
fig, ax = plt.subplots(4, 1, figsize=(6.0, 5.6), sharex=True)
for a, (lab, ys, cs) in zip(ax, rows):
    for y, c in zip(ys, cs):
        m = y != 0; yy = np.where(m, y, np.nan); a.plot(t, yy, color=c, lw=3)
    a.axhline(0, color="k", lw=0.8); a.set_ylim(-2.3, 2.3); a.set_yticks([]); a.spines["left"].set_visible(False)
    a.text(-0.04, 0, lab, ha="right", va="center", fontsize=17)
ax[-1].set_xticks([0, 0.25, 0.5, 0.75, 1]); ax[-1].set_xticklabels(["0", "¼", "½", "¾", "1"]); ax[-1].set_xlabel("t"); ax[-1].set_xlim(0, 1)
fig.subplots_adjust(left=0.3, right=0.98, hspace=0.3); save(fig, "filterbank/fig-psijk.svg")

# ---- filterbank/fig-aliasing.svg: chirp (64 samples), db2: each branch alone, and their sum ----
n = 64; x = test_signal("chirp", n); h = H["db2"]
a, d, _ = analysis(x, h)
lo = synthesis(a, 0 * d, h, n); hi = synthesis(0 * a, d, h, n); out = lo + hi
print("aliasing: max|x - low only| =", np.abs(x - lo).max(), " max|x - (low+high)| =", np.abs(x - out).max())
fig, ax = plt.subplots(3, 1, figsize=(6.6, 5.6), sharex=True)
k = np.arange(n)
for aa, y, c, lab in [(ax[0], lo, BLUE, "low branch only"), (ax[1], hi, RED, "high branch only"), (ax[2], out, ORANGE, "low + high")]:
    aa.plot(k, x, color="k", lw=1.5, label="input"); aa.plot(k, y, color=c, lw=2.5, ls="--" if aa is ax[2] else "-", label=lab)
    aa.set_ylim(-1.6, 2.4); aa.set_yticks([-1, 0, 1]); aa.legend(frameon=False, loc="upper left", ncol=2, fontsize=15, handlelength=1.5)
ax[2].set_xlabel("sample")
fig.tight_layout(); save(fig, "filterbank/fig-aliasing.svg")

# ---- filterbank/fig-cascade.svg: the coefficient vector after 0..3 levels (64 samples) ----
fig, a = plt.subplots(figsize=(7.2, 3.9)); a.axis("off"); a.set_xlim(-1.5, 64.5); a.set_ylim(-0.9, 4.6)
cols = {"x": "#d9d9d9", "a": "#dbe9f6", "d1": "#f4cccc", "d2": "#f9e0c7", "d3": "#e4d7ef"}
layouts = [[("x", 64, "x: 64")], [("a", 32, "a₁: 32"), ("d1", 32, "d₁: 32")],
           [("a", 16, "a₂: 16"), ("d2", 16, "d₂: 16"), ("d1", 32, "d₁: 32")],
           [("a", 8, "a₃"), ("d3", 8, "d₃"), ("d2", 16, "d₂: 16"), ("d1", 32, "d₁: 32")]]
work = [0, 64, 32, 16]
for r, lay in enumerate(layouts):
    y = 3.6 - 1.2 * r; x0 = 0
    for key, w, lab in lay:
        a.add_patch(Rectangle((x0, y), w, 0.75, fc=cols[key], ec="k", lw=1.5)); a.text(x0 + w / 2, y + 0.37, lab, ha="center", va="center", fontsize=16); x0 += w
    if r: a.text(-1.0, y + 0.37, f"level {r}", ha="right", va="center", fontsize=16, color=GREY)
    if r: a.annotate("", xy=(layouts[r][0][1] / 2, y + 0.78), xytext=(layouts[r - 1][0][1] / 2, y + 1.18), arrowprops=dict(arrowstyle="->", lw=1.8, color=BLUE))
a.text(32, -0.75, "filtered: 64 + 32 + 16 = 112 < 2·64;  always 64 numbers", ha="center", fontsize=16)
save(fig, "filterbank/fig-cascade.svg")

# ---- filterbank/fig-moments.svg: HeaviSine (256), level-1 details for Haar, db2, db4 (periodic) ----
n = 256; x = test_signal("heavisine", n)
fig, ax = plt.subplots(4, 1, figsize=(6.4, 6.0), sharex=False)
ax[0].plot(np.arange(n), x, color="k", lw=2); ax[0].set_xlim(0, n); ax[0].set_yticks([]); ax[0].set_xticks([])
ax[0].text(0.99, 0.95, "signal", transform=ax[0].transAxes, ha="right", va="top", fontsize=16)
for aa, (name, c, p) in zip(ax[1:], [("haar", GREY, 1), ("db2", BLUE, 2), ("db4", PURPLE, 4)]):
    _, d, _ = analysis(x, H[name])
    kk = np.arange(len(d)) * 2
    aa.vlines(kk, 0, d, color=c, lw=2); aa.axhline(0, color="k", lw=0.8)
    aa.set_ylim(-1.6, 1.6); aa.set_xlim(0, n); aa.set_yticks([-1, 0, 1])
    sm = np.abs(d)[np.r_[5:30, 45:70, 110:125]].max()
    aa.text(0.99, 0.95, f"{'Haar' if name == 'haar' else name}: {p} vanishing moment{'s' if p > 1 else ''}", transform=aa.transAxes, ha="right", va="top", fontsize=16, color=c)
    print(name, "max |d| on smooth stretches", sm, "max overall", np.abs(d).max())
    if aa is not ax[-1]: aa.set_xticks([])
ax[-1].set_xlabel("position (level-1 details)")
fig.tight_layout(h_pad=0.4); save(fig, "filterbank/fig-moments.svg")

# ---- filterbank/fig-borders.svg: ramp (64), db4, level-1 details, periodic vs symmetric ----
n = 64; x = test_signal("ramp", n)
fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True)
for aa, mode, c in [(ax[0], "periodic", RED), (ax[1], "symmetric", BLUE)]:
    a_, d, k0 = analysis(x, H["db4"], mode)
    kk = np.arange(len(d)) + k0
    aa.vlines(kk, 0, d, color=c, lw=3); aa.plot(kk, d, "o", color=c, ms=4); aa.axhline(0, color="k", lw=0.8)
    aa.set_ylim(-0.6, 0.6); aa.set_yticks([-0.5, 0, 0.5])
    aa.text(0.02, 0.92, f"{mode}: {len(d)} details", transform=aa.transAxes, va="top", fontsize=16, color=c)
    inner = np.abs(d[(kk >= 0) & (2 * kk + 7 < n)]).max()
    print(mode, "count", len(d), "max |d| interior", inner, "max |d|", np.abs(d).max())
ax[1].set_xlabel("detail index k (uses samples 2k … 2k+7)")
fig.tight_layout(); save(fig, "filterbank/fig-borders.svg")
