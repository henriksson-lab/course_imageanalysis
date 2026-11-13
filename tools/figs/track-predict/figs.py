# Slide figures for track-predict/lec.qmd. Run from the repository root:
#   python3 -I tools/figs/track-predict/figs.py
# The two filter plots run the kalman1d demo's own model (demos/src/kalman1d/model.ts) through
# esbuild + node (kfdata.ts), so their numbers are the demo's. Sizes are display px / 96.
# (fig-crossing and fig-kalman-1d come from tools/figs/tracking/figs.ts.)
import os, json, subprocess, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(R, "track-predict") + "/"
plt.rcParams.update({"font.size": 17, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "track-predict"})
INK, BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#1b2a3a", "#1f77b4", "#e69f00", "#2a9d4b", "#d62728", "#cc79a7", "#888"
def save(fig, name):
    fig.savefig(OUT + name, facecolor="white", metadata={"Date": None}); plt.close(fig); print("wrote", OUT + name)

bundle = subprocess.run([os.path.join(R, "node_modules/.bin/esbuild"), "tools/figs/track-predict/kfdata.ts", "--bundle",
                         "--platform=node", "--log-level=warning"], cwd=R, check=True, capture_output=True, text=True).stdout
D = json.loads(subprocess.run(["node"], input=bundle, check=True, capture_output=True, text=True).stdout)

# ------------------------------------------------------------ 1. two motion models
fig, axs = plt.subplots(1, 2, figsize=(560 / 96, 290 / 96))
past = np.array([[0, 0.25], [1, 0.0], [2, 0.3], [3, 0.1]])
for ax, title, mode in zip(axs, ["random walk", "constant velocity"], ["rw", "cv"]):
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.6, 5.4); ax.set_ylim(-1.6, 1.8)
    ax.set_title(title, fontsize=17)
    ax.plot(past[:, 0], past[:, 1], "-", color=GREY, lw=2)
    ax.plot(past[:, 0], past[:, 1], "o", color=BLUE, ms=10)
    last = past[-1]
    if mode == "rw":
        pred = last; ax.add_patch(Ellipse(pred, 2.4, 2.4, fc=ORANGE, alpha=0.15, ec=ORANGE, lw=2, ls="--"))
        ax.text(last[0], -1.45, "guess = last position", ha="center", fontsize=17)
    else:
        v = (past[-1] - past[0]) / 3; pred = last + v
        ax.annotate("", pred, last, arrowprops=dict(arrowstyle="->", color=INK, lw=2))
        ax.add_patch(Ellipse(pred, 1.5, 1.0, fc=ORANGE, alpha=0.15, ec=ORANGE, lw=2, ls="--"))
        ax.text(2.3, -1.45, "guess = last + v·Δt", ha="center", fontsize=17)
    ax.plot(*pred, "x", color=ORANGE, ms=14, mew=3.5)
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.05)
save(fig, "fig-models.svg")

# ------------------------------------------------------------ 2. uncertainty: 1D bell, 2D ellipses
fig, axs = plt.subplots(1, 2, figsize=(560 / 96, 280 / 96), gridspec_kw={"width_ratios": [1.15, 1]})
x = np.linspace(4, 16, 400); m, sd = 10, 2
axs[0].plot(x, np.exp(-(x - m) ** 2 / (2 * sd * sd)), color=BLUE, lw=3)
axs[0].fill_between(x, np.exp(-(x - m) ** 2 / (2 * sd * sd)), where=abs(x - m) <= sd, color=BLUE, alpha=0.18)
axs[0].annotate("", (m + sd, 0.3), (m, 0.3), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.8))
axs[0].text(m + sd / 2, 0.36, "σ", ha="center"); axs[0].axvline(m, color=INK, lw=1, ls=":")
axs[0].set_yticks([]); axs[0].set_xticks([6, 10, 14]); axs[0].set_xlabel("position (1D)"); axs[0].set_title("10 ± 2", fontsize=17)
axs[0].spines["left"].set_visible(False)
rng = np.random.default_rng(3)
C = np.array([[2.0, 1.2], [1.2, 1.0]]); w, V = np.linalg.eigh(C); ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
pts = rng.multivariate_normal([0, 0], C, 150)
axs[1].plot(pts[:, 0], pts[:, 1], ".", color=GREY, ms=4)
for k, c in zip([1, 2, 3], [ORANGE, PURPLE, GREEN]):
    axs[1].add_patch(Ellipse((0, 0), 2 * k * np.sqrt(w[1]), 2 * k * np.sqrt(w[0]), angle=ang, fc="none", ec=c, lw=2.5))
    axs[1].text(-4.8 + 1.75 * (k - 1), 3.0, f"{k}σ", color=c, fontsize=17, weight="bold")
axs[1].set_aspect("equal"); axs[1].set_xlim(-5, 5.5); axs[1].set_ylim(-4, 4); axs[1].set_xticks([]); axs[1].set_yticks([])
axs[1].set_xlabel("x"); axs[1].set_ylabel("y"); axs[1].set_title("in 2D: an ellipse", fontsize=17)
fig.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.2, wspace=0.12)
save(fig, "fig-uncertainty.svg")

# ------------------------------------------------------------ 3. the velocity locks on (demo preset "cv")
rec = D["cv"]["rec"]; t = np.arange(len(rec))
fig, ax = plt.subplots(figsize=(520 / 96, 300 / 96))
ax.axhline(1, color=GREY, lw=2, ls="--"); ax.text(38.5, 0.35, "dashed: true velocity 1", ha="right", color=GREY)
ax.plot(t, [r["v"] for r in rec], "-o", color=BLUE, lw=2.5, ms=5)
ax.set_xlabel("frame"); ax.set_ylabel("estimated velocity"); ax.set_ylim(-0.1, 1.6); ax.set_xlim(0, 39)
ax.text(2, 1.38, "never measured: only positions are", fontsize=17, color=INK)
fig.subplots_adjust(left=0.15, right=0.97, top=0.96, bottom=0.19)
save(fig, "fig-velocity.svg")

# ------------------------------------------------------------ 4. missed frames (demo preset "missed")
rec = D["missed"]["rec"][:21]; tru = D["missed"]["truth"][:21]; t = np.arange(len(rec))
x_ = np.array([r["x"] for r in rec]); sd_ = 2 * np.sqrt([r["p"] for r in rec])
fig, ax = plt.subplots(figsize=(540 / 96, 380 / 96))
tru = np.array(tru)
ax.fill_between(t, x_ - sd_ - tru, x_ + sd_ - tru, color=BLUE, alpha=0.2, lw=0, label="estimate ± 2σ")
ax.plot(t, x_ - tru, color=BLUE, lw=2.5)
ax.axhline(0, color=GREY, lw=2, ls="--", label="truth")
mt = [k for k in t if not rec[k]["missed"]]
ax.plot(mt, [rec[k]["z"] - tru[k] for k in mt], "o", color=ORANGE, ms=6, label="measurement")
for k in (12, 13, 14): ax.text(k, -6.0, "×", ha="center", va="center", fontsize=20, color=RED)
ax.text(13, -7.4, "missed", ha="center", color=RED, fontsize=17)
ax.annotate("±1.9", (11, x_[11] - tru[11] + sd_[11]), (6.5, 5.2), arrowprops=dict(arrowstyle="->", color=INK), fontsize=17)
ax.annotate("±4.1", (14, x_[14] - tru[14] + sd_[14]), (15.5, 5.8), arrowprops=dict(arrowstyle="->", color=INK), fontsize=17)
ax.set_xlabel("frame"); ax.set_ylabel("position − truth"); ax.set_ylim(-8.2, 7.5); ax.set_xlim(0, 20)
ax.legend(frameon=False, fontsize=17, loc="lower left", ncol=2, bbox_to_anchor=(-0.15, 1.0), columnspacing=1.0, handlelength=1.5)
fig.subplots_adjust(left=0.13, right=0.97, top=0.78, bottom=0.16)
save(fig, "fig-missed.svg")
print("missed: x-t at 11, 14:", round(x_[11] - 11, 2), round(x_[14] - 14, 2), "2sd", round(sd_[11], 2), round(sd_[14], 2))

# ------------------------------------------------------------ 5. Mahalanobis gate
fig, ax = plt.subplots(figsize=(460 / 96, 330 / 96))
S = np.array([[4.0, 2.4], [2.4, 2.0]]); w, V = np.linalg.eigh(S); ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
ax.add_patch(Ellipse((0, 0), 2 * 3 * np.sqrt(w[1]), 2 * 3 * np.sqrt(w[0]), angle=ang, fc=ORANGE, alpha=0.12, ec=ORANGE, lw=2.5, ls="--"))
ax.plot(0, 0, "x", color=ORANGE, ms=14, mew=3.5); ax.text(0.3, -0.9, "prediction", fontsize=17)
Si = np.linalg.inv(S)
for p, c, lab, off in [((4.5, 3.5), GREEN, "inside", (0.0, 0.6)), ((-1.6, 2.4), RED, "outside", (0.0, 0.6))]:
    d = np.hypot(*p); dm = np.sqrt(np.array(p) @ Si @ np.array(p))
    ax.plot(*p, "o", ms=13, mfc="white", mec=c, mew=3); ax.plot([0, p[0]], [0, p[1]], color=c, lw=1.5, ls=":")
    ax.text(p[0] + off[0], p[1] + off[1], f"{lab}: {d:.1f} px,\n{dm:.1f} σ", color=c, fontsize=17, ha="center" if off[0] == 0 else "right")
    print(lab, round(d, 2), round(dm, 2))
ax.set_aspect("equal"); ax.set_xlim(-7.5, 8.5); ax.set_ylim(-5.5, 6.2); ax.axis("off")
ax.text(-7.3, -5.2, "dashed: 3σ gate", color=ORANGE, fontsize=17)
fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
save(fig, "fig-mahalanobis.svg")

# ------------------------------------------------------------ 6. steady-state gain against Q/R (random walk)
q = np.logspace(-3, 2, 300)
P = 0.5 * (q + np.sqrt(q * q + 4 * q)); K = P / (P + 1)
fig, ax = plt.subplots(figsize=(500 / 96, 320 / 96))
ax.semilogx(q, K, color=BLUE, lw=3)
for qq, lab, off in [(1.0, "Q = R:\nK = 0.618", (4, 0.3)), (0.025, "Q = 0.1, R = 4:\nK = 0.146", (0.0013, 0.38))]:
    pp = 0.5 * (qq + np.sqrt(qq * qq + 4 * qq)); kk = pp / (pp + 1)
    ax.plot(qq, kk, "o", color=ORANGE, ms=9); ax.annotate(lab, (qq, kk), off, arrowprops=dict(arrowstyle="->", color=INK), fontsize=17)
ax.text(0.0013, 0.97, "high K: trust the\nmeasurements", fontsize=17, color=GREY, ha="left", va="top")
ax.text(95, 0.04, "low K: trust the model", fontsize=17, color=GREY, ha="right")
ax.set_xlabel("Q / R"); ax.set_ylabel("steady-state gain K"); ax.set_ylim(0, 1); ax.set_xlim(1e-3, 1e2)
fig.subplots_adjust(left=0.15, right=0.94, top=0.96, bottom=0.19)
save(fig, "fig-gain.svg")
