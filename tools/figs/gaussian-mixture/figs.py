"""Slide figures for gaussian-mixture/lec.qmd. Run from the repo root:
python3 -I tools/figs/gaussian-mixture/figs.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap

OUT = "gaussian-mixture/"
C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
LIGHT = ["#d6e6f4", "#ffe2c6", "#d5efd2", "#e6dcf0"]
plt.rcParams.update({"font.family": "sans-serif", "font.size": 18, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})


def npdf(x, m, v):
    return np.exp(-0.5 * (x - m) ** 2 / v) / np.sqrt(2 * np.pi * v)


def ell(ax, mu, S, r=1.0, **kw):
    w, v = np.linalg.eigh(S)
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    ax.add_patch(Ellipse(mu, 2 * r * np.sqrt(w[1]), 2 * r * np.sqrt(w[0]), angle=ang, fill=False, **kw))


def mvn(X, mu, S):
    d = X - mu
    Si = np.linalg.inv(S)
    q = np.einsum("...i,ij,...j->...", d, Si, d)
    return np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(np.linalg.det(S)))


def box(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for s_ in ax.spines.values():
        s_.set_visible(True); s_.set_color("#999")


gx, gy = np.meshgrid(np.linspace(-4, 4, 300), np.linspace(-4, 4, 300))
G = np.stack([gx, gy], -1)

# ---- 1. from a bell to hills ----
fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.9))
x = np.linspace(-4, 4, 400)
axs[0].plot(x, npdf(x, 0, 1), color=C[0], lw=2.5)
axs[0].fill_between(x, npdf(x, 0, 1), where=np.abs(x) <= 1, color=C[0], alpha=0.2)
axs[0].set_yticks([]); axs[0].set_xticks([-2, 0, 2]); axs[0].set_xticklabels(["μ−2σ", "μ", "μ+2σ"], fontsize=16)
axs[0].set_title("1D: a bell", fontsize=19)
rs = np.random.default_rng(2)
for ax, S, t in zip(axs[1:], [np.eye(2), np.diag([3.0, 0.5]), np.array([[2.0, 1.4], [1.4, 1.5]])],
                    ["round", "stretched", "tilted ($c > 0$)"]):
    ax.contourf(gx, gy, mvn(G, np.zeros(2), S), levels=8, cmap="Blues")
    for r in (1, 2):
        ell(ax, (0, 0), S, r, color="#222", lw=1.6, ls="-" if r == 1 else "--")
    ax.set_title(t, fontsize=19); box(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-hills.svg"); plt.close(fig)

# ---- 2. Mahalanobis: P and Q at the same Euclidean distance ----
S = np.array([[5.0, 2.0], [2.0, 2.0]])
Si = np.linalg.inv(S)
P, Q = np.array([2.0, 1.0]), np.array([-1.0, 2.0])
dP, dQ = np.sqrt(P @ Si @ P), np.sqrt(Q @ Si @ Q)
print(f"d_M(P) = {dP:.2f}, d_M(Q) = {dQ:.2f}")
fig, ax = plt.subplots(figsize=(6.0, 3.6))
u = np.array([2.0, 1.0]) / np.sqrt(5)
for r in (1, 2):
    ell(ax, (0, 0), S, r, color=C[0], lw=2.2 if r == 1 else 1.5)
    q = -r * np.sqrt(6) * u
    ax.text(q[0] - 0.15, q[1] - 0.15, f"$d_M = {r}$", color=C[0], fontsize=18, ha="right", va="top")
ax.add_patch(plt.Circle((0, 0), np.sqrt(5), fill=False, color="#777", ls="--", lw=1.5))
ax.text(1.75, -2.0, "Euclidean $\\sqrt{5}$", color="#555", fontsize=18, ha="left", va="top")
for pt, d, name, c, off, ha in ((P, dP, "P", "#222", (0.3, -0.75), "left"), (Q, dQ, "Q", "#d62728", (-0.3, 0.35), "right")):
    ax.plot(*pt, "o", ms=11, color=c, zorder=5)
    ax.text(pt[0] + off[0], pt[1] + off[1], f"{name}: $d_M$ = {d:.2f}", fontsize=19, color=c, ha=ha,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
ax.plot(0, 0, "+", ms=12, color="#222")
ax.set_xlim(-8.6, 8.6); ax.set_ylim(-4.4, 4.4); box(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-mahal.svg"); plt.close(fig)

# ---- 3. covariance types ----
mus = [np.array([-1.8, -1.2]), np.array([1.6, -1.4]), np.array([0.2, 1.8])]
full = [np.array([[1.0, 0.6], [0.6, 0.6]]), np.array([[0.3, -0.1], [-0.1, 0.9]]), np.array([[1.2, -0.5], [-0.5, 0.5]])]
types = {
    "spherical": [np.eye(2) * np.trace(S_) / 2 for S_ in full],
    "diagonal": [np.diag(np.diag(S_)) for S_ in full],
    "full": full,
    "tied": [sum(full) / 3] * 3,
}
fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.8))
for ax, (name, covs) in zip(axs, types.items()):
    for k in range(3):
        for r in (1, 2):
            ell(ax, mus[k], covs[k], r, color=C[k], lw=2 if r == 1 else 1.2, ls="-" if r == 1 else "--")
        ax.plot(*mus[k], "o", color=C[k], ms=5)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-3.8, 4.4); ax.set_title(name, fontsize=19); box(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-covtypes.svg"); plt.close(fig)

# ---- 4. a 1D mixture as a recipe ----
w = [0.6, 0.3, 0.1]; m = [60, 130, 200]; s = [15, 20, 15]
names = ["background", "cytoplasm", "nucleus"]
g = np.linspace(0, 255, 600)
fig, ax = plt.subplots(figsize=(6.0, 3.4))
tot = sum(wk * npdf(g, mk, sk ** 2) for wk, mk, sk in zip(w, m, s))
ax.fill_between(g, tot, color="#ddd")
ax.plot(g, tot, color="#222", lw=2, label="$p(x)$")
for k in range(3):
    ax.plot(g, w[k] * npdf(g, m[k], s[k] ** 2), color=C[k], lw=2.5, ls="--", label=f"{names[k]}: π = {w[k]}")
ax.set_xlim(0, 255); ax.set_yticks([]); ax.set_xlabel("grey value"); ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=17)
fig.tight_layout(); fig.savefig(OUT + "fig-mixture1d.svg"); plt.close(fig)

# ---- 5. the labels are hidden ----
rs = np.random.default_rng(4)
n = 300
z = rs.random(n) < 0.6
A = rs.multivariate_normal([-0.6, -0.4], [[1.0, 0.5], [0.5, 0.9]], n)
B = rs.multivariate_normal([1.3, 1.0], [[0.6, -0.2], [-0.2, 0.7]], n)
X = np.where(z[:, None], A, B)
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.4))
axs[0].scatter(X[:, 0], X[:, 1], s=10, c=np.where(z, C[0], C[1]))
axs[0].set_title("what the recipe made", fontsize=19)
axs[1].scatter(X[:, 0], X[:, 1], s=10, c="#777")
axs[1].set_title("what we observe", fontsize=19)
for ax in axs:
    ax.set_xlim(-3.8, 3.8); ax.set_ylim(-3.4, 3.4); box(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-hidden.svg"); plt.close(fig)

# ---- 6. responsibilities in 1D ----
pi = (0.7, 0.3); mu = (100, 160); var = 400
x = np.linspace(40, 220, 600)
j1 = pi[0] * npdf(x, mu[0], var); j2 = pi[1] * npdf(x, mu[1], var)
r1 = j1 / (j1 + j2)
xb = (mu[0] + mu[1]) / 2 + var / (mu[1] - mu[0]) * np.log(pi[0] / pi[1])
fig, axs = plt.subplots(2, 1, figsize=(5.6, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
axs[0].plot(x, j1, color=C[0], lw=2.5, label="$0.7\\,\\mathcal{N}_1$")
axs[0].plot(x, j2, color=C[1], lw=2.5, label="$0.3\\,\\mathcal{N}_2$")
axs[0].set_yticks([]); axs[0].legend(frameon=False, fontsize=17, loc="upper right")
axs[1].plot(x, r1, color=C[0], lw=2.5)
for xv in (120, 130, 140):
    rv = 1 / (1 + pi[1] * npdf(xv, mu[1], var) / (pi[0] * npdf(xv, mu[0], var)))
    axs[1].plot(xv, rv, "o", color="#222", ms=8, zorder=5)
    axs[1].text(xv - 3 if xv != 140 else xv + 3, rv + (0.05 if xv != 140 else -0.12), f"{rv:.2f}", fontsize=18,
                ha="right" if xv != 140 else "left")
    print(f"r1({xv}) = {rv:.3f}")
for ax in axs:
    ax.axvline(xb, color="#555", ls="--", lw=1.5)
axs[1].text(xb + 3, 0.9, f"boundary\n{xb:.1f}", fontsize=18)
axs[1].set_ylabel("$r_1(x)$"); axs[1].set_xlabel("x"); axs[1].set_ylim(-0.05, 1.08)
fig.tight_layout(); fig.savefig(OUT + "fig-resp.svg"); plt.close(fig)
print(f"boundary {xb:.2f}")

# ---- 7. Bayes boundaries: tied (straight) and different (curved) ----
gx2, gy2 = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
G2 = np.stack([gx2, gy2], -1)
cases = [("same covariance", [np.array([[1.2, 0.5], [0.5, 0.8]])] * 2),
         ("different covariances", [np.array([[0.4, 0.0], [0.0, 0.4]]), np.array([[3.5, 0.0], [0.0, 2.5]])])]
m1, m2 = np.array([-1.0, -0.6]), np.array([1.4, 0.8])
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.6))
for ax, (title, (S1, S2)) in zip(axs, cases):
    lab = (0.5 * mvn(G2, m2, S2) > 0.5 * mvn(G2, m1, S1)).astype(int)
    ax.imshow(lab, origin="lower", extent=(-5, 5, -5, 5), cmap=ListedColormap(LIGHT[:2]))
    ax.contour(gx2, gy2, lab, levels=[0.5], colors="#222", linewidths=2)
    for k, (mk, Sk) in enumerate(((m1, S1), (m2, S2))):
        for r in (1, 2):
            ell(ax, mk, Sk, r, color=C[k], lw=2 if r == 1 else 1.2, ls="-" if r == 1 else "--")
    ax.set_title(title, fontsize=19); box(ax); ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
fig.tight_layout(); fig.savefig(OUT + "fig-boundaries.svg"); plt.close(fig)

# ---- 8. generative and discriminative on the same labelled cloud ----
rs = np.random.default_rng(6)
XA = rs.multivariate_normal([-1.2, -0.5], [[1.0, 0.4], [0.4, 0.7]], 120)
XB = rs.multivariate_normal([1.4, 0.9], [[0.7, -0.3], [-0.3, 1.0]], 120)
gx3, gy3 = np.meshgrid(np.linspace(-4.5, 4.5, 300), np.linspace(-4, 4, 300))
G3 = np.stack([gx3, gy3], -1)
fits = [(XA.mean(0), np.cov(XA.T)), (XB.mean(0), np.cov(XB.T))]
dens = [mvn(G3, mm, SS) for mm, SS in fits]
lab = (dens[1] > dens[0]).astype(int)
# a linear boundary (logistic regression by gradient ascent) for the discriminative side
Xall = np.vstack([XA, XB]); y = np.r_[np.zeros(120), np.ones(120)]
Xa = np.c_[Xall, np.ones(len(Xall))]
wv = np.zeros(3)
for _ in range(5000):
    p_ = 1 / (1 + np.exp(-Xa @ wv))
    wv += 0.05 * Xa.T @ (y - p_) / len(y)
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.5))
for ax in axs:
    ax.scatter(XA[:, 0], XA[:, 1], s=9, color=C[0]); ax.scatter(XB[:, 0], XB[:, 1], s=9, color=C[1])
xs = np.linspace(-4.5, 4.5, 2)
axs[0].plot(xs, -(wv[0] * xs + wv[2]) / wv[1], color="#222", lw=2.5)
axs[0].set_title("discriminative:\na boundary", fontsize=19)
for k in range(2):
    axs[1].contour(gx3, gy3, dens[k], levels=4, colors=C[k], linewidths=1.4)
axs[1].contour(gx3, gy3, lab, levels=[0.5], colors="#222", linewidths=2.5)
axs[1].set_title("generative:\na density per class", fontsize=19)
for ax in axs:
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4, 4); box(ax)
fig.tight_layout(); fig.savefig(OUT + "fig-gendisc.svg"); plt.close(fig)
