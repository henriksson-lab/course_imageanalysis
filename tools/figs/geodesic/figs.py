"""Figures for geodesic/lec.qmd from the runs in data.ts (the gac demo's code and settings).
Run from the repo root (see data.ts):  python3 -I tools/figs/geodesic/figs.py <tmp>/gac-data.json
PNG figures: figure size in inches = display size / 100 px, dpi 200, fonts >= 12 pt."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "geodesic")
plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, ORANGE, SKY, PURPLE, YELLOW = "#0072b2", "#d55e00", "#009e73", "#e69f00", "#56b4e9", "#cc79a7", "#ffd400"
d = json.load(open(sys.argv[1]))
N = d["N"]
A = lambda a: np.array(a).reshape(N, N)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.04); plt.close(fig); print("wrote", name)


def show(ax, img, crop=None, vmin=0, vmax=255, cmap="gray"):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if crop:
        (x0, x1), (y0, y1) = crop
        ax.set_xlim(x0 - 0.5, x1 - 0.5); ax.set_ylim(y1 - 0.5, y0 - 0.5)


def front(ax, phi, color=YELLOW, lw=2.2, ls="-"):
    cs = ax.contour(A(phi), levels=[0], colors=[color], linewidths=lw, linestyles=ls)
    cs.set_path_effects([pe.Stroke(linewidth=lw + 2.2, foreground="black"), pe.Normal()])


# ------------------------------------------------------------------ stage 1: the braking balloon
def fig_brake():
    run = d["brake"]
    S = run["snaps"]
    fig, axs = plt.subplots(1, 2, figsize=(5.3, 2.75), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axs[0]
    show(ax, A(run["I"]), crop=((0, 64), (0, 64)))
    pick = [2, 4, 8, 11, 13, 14]
    cols = plt.get_cmap("viridis")(np.linspace(0.15, 1.0, len(pick)))
    for i, c in zip(pick, cols):
        cs = ax.contour(A(S[i]["phi"]), levels=[0], colors=[c], linewidths=2)
    ax.set_title(f"t = {S[pick[0]]['t']:.0f} (dark) to {S[pick[-1]]['t']:.0f}", fontsize=12.5)
    ax = axs[1]
    t = [s["t"] for s in S]; n = [s["inside"] for s in S]
    disk = int(sum(1 for y in range(N) for x in range(N) if (x - 28) ** 2 + (y - 28) ** 2 < 15 ** 2 and d["mask"][y * N + x]))
    ax.plot(t, n, "-o", color=BLUE, ms=4, lw=2)
    ax.axhline(disk, color=RED, ls="--", lw=1.5)
    ax.text(400, disk * 0.62, f"disk: {disk} px", color=RED, fontsize=13, ha="right", va="top")
    ax.set_yscale("log"); ax.set_ylim(30, 12000)
    ax.set_xlabel("time t"); ax.set_ylabel("pixels inside")
    ax.annotate("creeps\nthrough", xy=(212, 1000), xytext=(20, 2200), fontsize=13, arrowprops=dict(arrowstyle="->"))
    fig.tight_layout(pad=0.3, w_pad=1.0); save(fig, "fig-brake.png")
    print("disk A area", disk)


# ------------------------------------------------------------------ stage 1 vs stage 2 (shrink-wrap)
def fig_stages():
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.95))
    for ax, run, lab in zip(axs, (d["wrap0"], d["wrap"]), ("braking balloon (a = 0)", "geodesic (a = 5)")):
        s = run["snaps"][-1]
        show(ax, A(run["I"]))
        front(ax, s["phi"])
        ax.set_title(f"{lab}\nt = {s['t']:.0f}: {s['wrong']} wrong pixels", fontsize=12.5)
    fig.tight_layout(pad=0.3, w_pad=0.8); save(fig, "fig-stages.png")


# ------------------------------------------------------------------ the balloon
def fig_balloon():
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 3.3))
    C = ((68, 128), (32, 96))
    for ax, run, lab in zip(axs[:2], (d["cNo"], d["cYes"]), ("c = 0: bridges the slot", "c = −0.3: enters the slot")):
        s = run["snaps"][-1]
        show(ax, A(run["I"]), crop=C)
        front(ax, run["snaps"][0]["phi"], color="white", lw=1.2, ls="--")
        front(ax, s["phi"])
        ax.set_title(f"{lab}\n{s['inside']} px inside (C: 1020)", fontsize=12.5)
    ax = axs[2]
    L, H = d["leak"], d["hold"]
    show(ax, A(L["I"]), crop=((0, 64), (64, 128)))
    front(ax, H["snaps"][-1]["phi"], color=GREEN)
    front(ax, L["snaps"][1]["phi"], color=YELLOW)
    ax.set_title(f"weak edge: c = 1 leaks (yellow)\nc = 0.3, a = 3 holds (green)", fontsize=12.5)
    fig.tight_layout(pad=0.3, w_pad=0.8); save(fig, "fig-balloon.png")
    print("leak step", L["snaps"][1]["step"], L["snaps"][1]["inside"], "hold", H["snaps"][-1]["inside"])


# ------------------------------------------------------------------ σ
def fig_sigma():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.95))
    for ax, s in zip(axs, d["sig"]):
        show(ax, A(s["g"]), vmin=0, vmax=1, cmap="magma")
        ax.set_title(f"σ = {s['sigma']:g}: median g = {s['median']:.2f}", fontsize=12.5)
    fig.tight_layout(pad=0.3, w_pad=0.6); save(fig, "fig-sigma.png")


# ------------------------------------------------------------------ outside-in and inside-out
def fig_wrap():
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 3.25))
    for ax, run, lab in zip(axs, (d["wrap"], d["cell"]), ("outside-in: one box → 4 pieces", "inside-out: seed in the nucleus")):
        show(ax, A(run["I"]))
        cols = plt.get_cmap("viridis")(np.linspace(0.1, 1.0, len(run["snaps"])))
        for s, c in zip(run["snaps"], cols):
            cs = ax.contour(A(s["phi"]), levels=[0], colors=[c], linewidths=2)
            cs.set_path_effects([pe.Stroke(linewidth=3.6, foreground="black"), pe.Normal()])
        ts = ", ".join(f"{s['t']:.0f}" for s in run["snaps"])
        ax.set_title(f"{lab}\nt = {ts}", fontsize=12.5)
    fig.tight_layout(pad=0.3, w_pad=0.8); save(fig, "fig-wrap.png")
    print("wrap pieces", [s["pieces"] for s in d["wrap"]["snaps"]], "cell inside", [s["inside"] for s in d["cell"]["snaps"]])


if __name__ == "__main__":
    fig_brake(); fig_stages(); fig_balloon(); fig_sigma(); fig_wrap()
