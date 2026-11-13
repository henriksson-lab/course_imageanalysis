"""Fresnel reflectance against the angle of incidence (whitted/fig-fresnel.svg).
Run from the repo root: python3 -I tools/figs/whitted/fig_fresnel.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 17, "svg.fonttype": "none", "font.family": "sans-serif"})


def fresnel(th1, n1, n2):
    c1 = np.cos(th1)
    s2 = n1 / n2 * np.sin(th1)
    tir = s2 >= 1
    c2 = np.sqrt(np.clip(1 - s2**2, 0, 1))
    rs = ((n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)) ** 2
    rp = ((n1 * c2 - n2 * c1) / (n1 * c2 + n2 * c1)) ** 2
    rs[tir] = 1
    rp[tir] = 1
    return rs, rp


th = np.linspace(0, np.pi / 2, 2001)
deg = np.degrees(th)
rs, rp = fresnel(th, 1.0, 1.5)
R = (rs + rp) / 2
R0 = ((1 - 1.5) / 2.5) ** 2
schlick = R0 + (1 - R0) * (1 - np.cos(th)) ** 5
gs, gp = fresnel(th, 1.5, 1.0)
G = (gs + gp) / 2
thc = np.degrees(np.arcsin(1 / 1.5))
thb = np.degrees(np.arctan(1.5))
for a in (0, 60, 85):
    i = np.argmin(abs(deg - a))
    print(f"air->glass {a} deg: R = {R[i]:.4f}, Schlick = {schlick[i]:.4f}")
print(f"critical {thc:.1f}, Brewster {thb:.1f}")

fig, ax = plt.subplots(figsize=(6.0, 4.6))
ax.plot(deg, R, color="#0072B2", lw=2.8, label="air→glass R")
ax.plot(deg, rp, color="#0072B2", lw=1.4, ls=":", label="air→glass $R_p$")
ax.plot(deg, schlick, color="#E69F00", lw=2, ls="--", label="Schlick")
ax.plot(deg, G, color="#D55E00", lw=2.8, label="glass→air R")
ax.axvline(thc, color="#D55E00", lw=1, ls="-.")
ax.annotate(f"critical {thc:.1f}°:\ntotal reflection", (thc, 1.0), xytext=(47, 0.62), color="#D55E00", fontsize=16,
            arrowprops=dict(arrowstyle="->", color="#D55E00"))
ax.annotate(f"Brewster {thb:.1f}°:\n$R_p$ = 0", (thb, 0.0), xytext=(44, 0.27), fontsize=16, color="#0072B2",
            arrowprops=dict(arrowstyle="->", color="#0072B2"))
ax.annotate("4 % head-on", (0, R0), xytext=(4, 0.12), fontsize=16, arrowprops=dict(arrowstyle="->"))
ax.set_xlim(0, 90)
ax.set_ylim(0, 1.03)
ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
ax.set_xlabel("angle of incidence θ₁ (degrees)")
ax.set_ylabel("reflected fraction R")
ax.legend(loc="upper left", fontsize=15, frameon=False, bbox_to_anchor=(0.0, 0.93), handlelength=1.6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("whitted/fig-fresnel.svg", facecolor="white")
print("wrote whitted/fig-fresnel.svg")
