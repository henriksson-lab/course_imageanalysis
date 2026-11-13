"""Slide figures for em/lec.qmd. Run from the repo root: python3 -I tools/figs/em/figs.py
(fig-chicken.svg is hand-written.)"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

OUT = "em/"
C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
CRGB = np.array([[0x1f, 0x77, 0xb4], [0xff, 0x7f, 0x0e], [0x2c, 0xa0, 0x2c]]) / 255
plt.rcParams.update({"font.family": "sans-serif", "font.size": 18, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})


def npdf(x, m, v):
    return np.exp(-0.5 * (x - m) ** 2 / v) / np.sqrt(2 * np.pi * v)


def logmvn(X, mu, S):
    d = X - mu
    q = np.einsum("ni,ij,nj->n", d, np.linalg.inv(S), d)
    return -0.5 * q - np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(S))


def estep(X, pi, mu, S):
    lj = np.stack([np.log(pi[k]) + logmvn(X, mu[k], S[k]) for k in range(len(pi))], 1)
    mx = lj.max(1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(lj - mx).sum(1))
    return np.exp(lj - lse[:, None]), lse.sum()


def mstep(X, r, floor=1e-3):
    N = r.sum(0)
    mu = (r.T @ X) / N[:, None]
    S = []
    for k in range(r.shape[1]):
        d = X - mu[k]
        S.append((r[:, k, None] * d).T @ d / N[k] + floor * np.eye(X.shape[1]))
    return N / len(X), mu, np.array(S)


def ell(ax, mu, S, r=2.0, **kw):
    w, v = np.linalg.eigh(S)
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    ax.add_patch(Ellipse(mu, 2 * r * np.sqrt(w[1]), 2 * r * np.sqrt(w[0]), angle=ang, fill=False, **kw))


def box(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for s_ in ax.spines.values():
        s_.set_visible(True); s_.set_color("#999")


# ---- 1. EM in 2D: a few snapshots ----
rs = np.random.default_rng(12)
X = np.vstack([rs.multivariate_normal([0, 0], [[1.0, 0.6], [0.6, 0.8]], 90),
               rs.multivariate_normal([3.6, 0.6], [[0.5, -0.3], [-0.3, 0.9]], 70),
               rs.multivariate_normal([1.4, 3.4], [[1.2, 0.0], [0.0, 0.3]], 60)])
pi = np.ones(3) / 3
mu = np.array([[-1.0, 3.5], [1.5, 1.0], [4.5, 3.5]])
S = np.array([np.eye(2) * 1.5] * 3)
snaps, L = {}, []
for it in range(200):
    r, ll = estep(X, pi, mu, S)
    L.append(ll)
    if it in (0, 1, 4):
        snaps[it] = (r, mu.copy(), S.copy())
    if it > 0 and abs(L[-1] - L[-2]) < 1e-6:
        break
    pi, mu, S = mstep(X, r)
snaps["end"] = (r, mu.copy(), S.copy())
print("2D EM: iterations", it, "log p", np.round(L[0], 1), "->", np.round(L[-1], 1))
fig, axs = plt.subplots(2, 2, figsize=(5.4, 5.6))
for ax, key, title in zip(axs.ravel(), [0, 1, 4, "end"], ["start", "after 1 iteration", "after 4", f"converged ({it})"]):
    r, m_, S_ = snaps[key]
    ax.scatter(X[:, 0], X[:, 1], s=9, c=np.clip(r @ CRGB, 0, 1))
    for k in range(3):
        ell(ax, m_[k], S_[k], 2, color=C[k], lw=2.2)
        ax.plot(*m_[k], "X", ms=10, color=C[k], mec="#222")
    ax.set_xlim(-3.5, 6.5); ax.set_ylim(-3.2, 6.2); ax.set_title(title, fontsize=17); box(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-emsteps.svg"); plt.close(fig)

# ---- 2. soft to hard: r1(x) as the variance shrinks ----
x = np.linspace(0, 11, 500)
pts = np.array([1, 2, 4, 7, 9, 10.0])
fig, ax = plt.subplots(figsize=(5.4, 3.6))
for v, c, ls in ((16, "#9ecae1", "-"), (4, C[0], "-"), (1, "#08519c", "-"), (0.01, "#222", "--")):
    a = ((x - 8) ** 2 - (x - 2) ** 2) / (2 * v)
    ax.plot(x, 1 / (1 + np.exp(-a)), color=c, lw=2.5, ls=ls, label=f"σ² = {v:g}" if v > 0.05 else "σ² → 0: k-means")
a4 = ((pts - 8) ** 2 - (pts - 2) ** 2) / 8
ax.plot(pts, 1 / (1 + np.exp(-a4)), "o", color=C[0], ms=8, zorder=5)
ax.set_xlabel("x"); ax.set_ylabel("$r_1(x)$"); ax.set_xticks([0, 2, 4, 5, 6, 8, 10])
ax.legend(frameon=False, fontsize=14, loc="upper center", ncol=2)
ax.set_ylim(-0.03, 1.5); ax.set_yticks([0, 0.5, 1])
fig.tight_layout(); fig.savefig(OUT + "fig-hardsoft.svg"); plt.close(fig)

# ---- 3. local maxima: log-likelihood traces from 20 random starts ----
rs = np.random.default_rng(3)
X2 = np.vstack([rs.normal([0, 0], 0.5, (60, 2)), rs.normal([4, 0], 0.5, (60, 2)), rs.normal([2, 3.2], 0.5, (60, 2))])
traces = []
for run in range(20):
    mu = X2[rs.choice(len(X2), 3, replace=False)]
    pi = np.ones(3) / 3; S = np.array([np.eye(2)] * 3)
    L = []
    for it in range(150):
        r, ll = estep(X2, pi, mu, S)
        L.append(ll)
        if it > 0 and abs(L[-1] - L[-2]) < 1e-6:
            break
        pi, mu, S = mstep(X2, r)
    traces.append(L)
finals = np.array([t[-1] for t in traces])
best = finals.max()
nbest = int((finals > best - 0.5).sum())
print("local maxima: finals", np.round(np.sort(finals), 1), "best", round(best, 1), "count", nbest)
fig, ax = plt.subplots(figsize=(5.6, 3.8))
for t in traces:
    good = t[-1] > best - 0.5
    ax.plot(range(len(t)), t, color=C[0] if good else C[1], lw=1.8, alpha=0.75)
ax.set_xlabel("iteration"); ax.set_ylabel("log p(D)")
ax.set_xlim(0, 40)
lo = np.percentile([t[0] for t in traces], 10)
ax.set_ylim(lo, best + 0.08 * (best - lo))
ax.text(39, best + 0.02 * (best - lo), f"{nbest} of 20 runs: {best:.1f}", color=C[0], ha="right", va="bottom", fontsize=16)
worse = sorted(set(np.round(finals[finals <= best - 0.5], 1)))
if worse:
    ax.text(39, max(worse) - 0.03 * (best - lo), f"{20 - nbest} stop lower", color=C[1], ha="right", va="top", fontsize=16)
fig.tight_layout(); fig.savefig(OUT + "fig-localmax.svg"); plt.close(fig)

# ---- 4. collapse: one component shrinks onto a single point ----
pts = np.array([1, 2, 4, 7, 9, 10.0])
m1, v1 = pts.mean(), pts.var()
sig = 10.0 ** np.arange(0.5, -6.01, -0.25)
logp = [np.log(0.5 * npdf(pts, m1, v1) + 0.5 * npdf(pts, 10, s ** 2)).sum() for s in sig]
print("collapse:", dict(zip(sig.tolist(), np.round(logp, 2))))
fig, axs = plt.subplots(2, 1, figsize=(5.2, 5.6))
x = np.linspace(-2, 13, 3000)
axs[0].plot(x, 0.5 * npdf(x, m1, v1), color=C[0], lw=2.5)
for s, a in ((1, 0.35), (0.3, 0.6), (0.1, 1.0)):
    axs[0].plot(x, 0.5 * npdf(x, 10, s ** 2), color=C[1], lw=2, alpha=a)
    axs[0].text(10.4, 0.5 * npdf(10, 10, s ** 2) * (0.97 if s < 0.2 else 1), f"σ = {s}", color=C[1], fontsize=15, va="center")
axs[0].plot(pts, np.zeros(6), "o", color="#222", ms=8, clip_on=False, zorder=5)
axs[0].set_ylim(0, 2.2); axs[0].set_xlim(-1, 14); axs[0].set_xlabel("x"); axs[0].set_ylabel("weighted density")
axs[0].set_yticks([0, 1, 2])
axs[1].semilogx(sig, logp, "-", color=C[1], lw=2.5)
axs[1].set_xticks([1, 1e-2, 1e-4, 1e-6])
axs[1].invert_xaxis()
axs[1].set_xlabel("σ of the spike"); axs[1].set_ylabel("log p(D)")
fig.tight_layout(); fig.savefig(OUT + "fig-collapse.svg"); plt.close(fig)

# ---- 5. BIC and AIC by hand (numbers from the slide's table) ----
K = np.array([1, 2, 3, 4]); logL = np.array([-4210, -4020, -4012, -4010.0]); n = 1000
p = 3 * K - 1
bic = -2 * logL + p * np.log(n); aic = -2 * logL + 2 * p
print("BIC", np.round(bic, 1), "AIC", np.round(aic, 1))
fig, ax = plt.subplots(figsize=(5.0, 3.8))
ax.plot(K, bic, "-o", color=C[0], lw=2.5, ms=8, label="BIC")
ax.plot(K, aic, "-s", color=C[1], lw=2.5, ms=8, label="AIC")
ax.plot(K[bic.argmin()], bic.min(), "o", ms=20, mfc="none", mec=C[0], mew=2.5)
ax.plot(K[aic.argmin()], aic.min(), "s", ms=20, mfc="none", mec=C[1], mew=2.5)
ax.set_xticks(K); ax.set_xlabel("K"); ax.set_ylabel("smaller is better")
ax.set_ylim(8020, 8110)
ax.legend(frameon=False, loc="center right")
fig.tight_layout(); fig.savefig(OUT + "fig-bic.svg"); plt.close(fig)
