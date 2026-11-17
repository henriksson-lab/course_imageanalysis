# Figures for the unit "Reading training curves" (training-curves/lec.qmd). Run (from the repo root)
# after dump.ts (see its header) and the course program:
#   (cd code && cargo run -p training --features solution -- training-curves)
#   cp code/out/training-curves/batch-losses.csv tools/figs/training-curves/
#   python3 -I tools/figs/training-curves/figs.py
# demos/src/curvedoctor/curves.json: the demo's recorded runs; batch-losses.csv: the loss of every
# mini-batch of the course code's healthy run, raw and smoothed.
import json, csv
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = '/data/henriksson/github/course/imageanalysis'
OUT = R + '/training-curves/'
C = json.load(open(R + '/demos/src/curvedoctor/curves.json'))
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#1b2a3a",
                     "xtick.color": "#555", "ytick.color": "#555", "text.color": "#1b2a3a",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, RED, ORANGE, GREEN, GREY = "#1b2a3a", "#2a6f97", "#bc4749", "#e08a1e", "#3a8a5c", "#8a96a3"
LN3 = np.log(3)
def save(fig, name, **kw):
    fig.savefig(OUT + name, facecolor="white", **kw); plt.close(fig)
def case(d): return next(c for c in C if c["diagnosis"] == d)
def panel(ax, d, title, top=1.6, legend=False):
    r = case(d)["run"]
    ax.plot(np.minimum(r["train"], top), color=BLUE, lw=2.3, label="training")
    ax.plot(np.minimum(r["val"], top), color=RED, lw=2.3, label="validation")
    ax.axhline(LN3, color=GREY, ls="--", lw=1.2)
    ax.set_ylim(0, top); ax.set_title(title, fontsize=13); ax.set_xlabel("epoch")
    if legend: ax.legend(fontsize=11, loc="upper right")

# 1. healthy
fig, ax = plt.subplots(figsize=(460 / 96, 300 / 96))
panel(ax, "healthy", "healthy"); ax.set_ylabel("loss"); ax.legend(fontsize=11, loc="center right", bbox_to_anchor=(1, 0.62))
ax.text(195, LN3 + 0.04, "ln 3: guessing", color=GREY, fontsize=11, ha="right")
r = case("healthy")["run"]; b = int(np.argmin(r["val"]))
ax.annotate(f"gap: {r['val'][-1] - r['train'][-1]:.2f}", xy=(200, (r["val"][-1] + r["train"][-1]) / 2), xytext=(120, 0.75), fontsize=11, arrowprops=dict(arrowstyle="->", color=INK))
fig.tight_layout(); save(fig, "fig-healthy.svg")

# 2. under- and overfitting
fig, axs = plt.subplots(1, 2, figsize=(620 / 96, 290 / 96), sharey=True)
panel(axs[0], "underfit", "too simple: underfitting", legend=True); axs[0].set_ylabel("loss")
panel(axs[1], "overfit", "30 points: overfitting")
r = case("overfit")["run"]; b = int(np.argmin(r["val"]))
axs[1].scatter([b], [r["val"][b]], s=50, c=INK, zorder=5); axs[1].annotate(f"lowest: epoch {b}", xy=(b, r["val"][b]), xytext=(80, 0.25), fontsize=11, arrowprops=dict(arrowstyle="->", color=INK))
fig.tight_layout(); save(fig, "fig-fit.svg")

# 3. learning rate too high and too low
fig, axs = plt.subplots(1, 2, figsize=(620 / 96, 290 / 96), sharey=True)
panel(axs[0], "lrhigh", "learning rate 0.3: too high", legend=True); axs[0].set_ylabel("loss")
panel(axs[1], "lrlow", "learning rate 0.0001: too low")
fig.tight_layout(); save(fig, "fig-lr.svg")

# 4. the two bugs
fig, axs = plt.subplots(1, 2, figsize=(620 / 96, 290 / 96), sharey=True)
panel(axs[0], "deadinput", "all inputs 0", legend=True); axs[0].set_ylabel("loss")
axs[0].text(195, LN3 + 0.05, "ln 3", color=GREY, fontsize=12, ha="right")
panel(axs[1], "shuffled", "labels shuffled")
fig.tight_layout(); save(fig, "fig-bugs.svg")

# 5. imbalance: recall per class, plain and weighted loss
c = case("imbalance")
fig, ax = plt.subplots(figsize=(460 / 96, 290 / 96))
x = np.arange(3)
ax.bar(x - 0.2, 100 * np.array(c["run"]["recalls"]), 0.38, color=GREY, label=f"plain loss ({100 * c['run']['valAcc']:.1f} % correct)")
ax.bar(x + 0.2, 100 * np.array(c["fix"]["recalls"]), 0.38, color=GREEN, label=f"class weights ({100 * c['fix']['valAcc']:.1f} % correct)")
for k in range(3):
    ax.text(k - 0.2, 100 * c["run"]["recalls"][k] + 2, f"{100 * c['run']['recalls'][k]:.0f}", ha="center", fontsize=11)
    ax.text(k + 0.2, 100 * c["fix"]["recalls"][k] + 2, f"{100 * c['fix']['recalls'][k]:.0f}", ha="center", fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(["class 1\n(240 points)", "class 2\n(50)", "class 3\n(10)"])
ax.set_ylim(0, 135); ax.set_yticks([0, 50, 100]); ax.set_ylabel("recall (%)"); ax.legend(fontsize=10.5, loc="upper right")
fig.tight_layout(); save(fig, "fig-imbalance.svg")

# 6. mini-batch losses, raw and smoothed (course code)
rows = list(csv.DictReader(open(R + '/tools/figs/training-curves/batch-losses.csv')))
fig, ax = plt.subplots(figsize=(520 / 96, 280 / 96))
ax.plot([float(r["batch_loss"]) for r in rows], color="#9fb8cc", lw=0.8, label="loss of each mini-batch")
ax.plot([float(r["smoothed"]) for r in rows], color=ORANGE, lw=2.5, label="smoothed (β = 0.95)")
ax.set_ylim(0, 1.4); ax.set_xlabel("step (10 per epoch)"); ax.set_ylabel("loss"); ax.legend(fontsize=11)
fig.tight_layout(); save(fig, "fig-batches.svg")
print("written")
