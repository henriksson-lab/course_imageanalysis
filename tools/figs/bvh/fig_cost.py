"""Tests per ray against the number of objects (bvh/fig-cost.svg).
Measured points: tools/figs/bvh/cost.json, written by figs.ts (the bvhstep demo's uniform scene,
1000 random rays per N, SAH BVH, box + object tests). Run from the repo root."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 17, "svg.fonttype": "none", "font.family": "sans-serif"})
d = json.load(open("tools/figs/bvh/cost.json"))
n = np.array([r["n"] for r in d])
t = np.array([r["bvh"] for r in d])
N = np.logspace(np.log10(8), 6, 200)
fig, ax = plt.subplots(figsize=(6.0, 4.4))
ax.loglog(N, N, color="#D55E00", lw=2.8, label="brute force: N")
ax.loglog(N, 2 * np.log2(N), color="#0072B2", lw=2, ls="--", label="2 log₂ N")
ax.loglog(n, t, "o", color="#0072B2", ms=7, label="BVH, measured (demo)")
ax.annotate("N = 10⁶:\n10⁶ tests", (1e6, 1e6), xytext=(1.2e4, 4e2), fontsize=16, color="#D55E00", arrowprops=dict(arrowstyle="->", color="#D55E00"))
ax.annotate(f"≈ {2*np.log2(1e6):.0f} tests", (1e6, 2 * np.log2(1e6)), xytext=(4e4, 4), fontsize=16, color="#0072B2", arrowprops=dict(arrowstyle="->", color="#0072B2"))
ax.set_xlabel("number of objects N")
ax.set_ylabel("tests per ray")
ax.set_ylim(1, 3e6)
ax.legend(loc="upper left", frameon=False, fontsize=15)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("bvh/fig-cost.svg", facecolor="white")
print("wrote bvh/fig-cost.svg; BVH at 1024:", t[n == 1024])
