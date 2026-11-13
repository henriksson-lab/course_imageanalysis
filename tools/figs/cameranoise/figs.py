# Static figures for the camera-noise unit. Run from the repository root:
#   python3 -I tools/figs/cameranoise/figs.py
# Writes fig-*.svg / fig-*.png into cameranoise/. Noise model as in the camerabits demo
# (demos/src/camerabits/sensor.ts): shot sqrt(S), read sigma_r, quantisation g/sqrt(12),
# g = full well / (2^b - 1). Colours as in the demo: read blue, quantisation red dashed,
# shot green. Micrograph: demos/src/fourier2d/cell256.png (Robert M. Hunt, CC BY 3.0,
# Wikimedia Commons).
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
from skimage import io

plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
C = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
OUT = "cameranoise/"

def save(fig, name, **kw):
    fig.savefig(OUT + name, **kw); plt.close(fig); print("wrote", OUT + name)

cell = io.imread("demos/src/fourier2d/cell256.png").astype(float) / 255

# 1. fig-pipeline.svg: photons -> well -> read noise -> ADC -> ADU (hand-built SVG)
dots = []
rng = np.random.default_rng(5)
for k in range(46):
    dots.append(f'<circle cx="{150 + rng.uniform(-62, 62):.1f}" cy="{rng.uniform(150, 228):.1f}" r="5" fill="{C[0]}"/>')
photons = "".join(
    f'<path d="M{x},{12} q6,8 0,16 q-6,8 0,16 q6,8 0,16 q-6,8 0,16" fill="none" stroke="{C[3]}" stroke-width="3"/>'
    f'<path d="M{x - 7},{78} l7,12 l7,-12" fill="none" stroke="{C[3]}" stroke-width="3"/>'
    for x in (100, 135, 170, 205))
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 560" width="520" height="560" font-family="sans-serif" font-size="20">
<rect width="520" height="560" fill="#fff"/>
{photons}
<text x="240" y="50" fill="{C[3]}">photons, at random</text>
<path d="M78,108 L78,240 L222,240 L222,108" fill="#eef4fa" stroke="#333" stroke-width="3"/>
<line x1="70" x2="232" y1="122" y2="122" stroke="#666" stroke-width="2" stroke-dasharray="7 5"/>
{''.join(dots)}
<text x="240" y="128" fill="#444">full well N<tspan font-size="14" dy="5">max</tspan></text>
<text x="240" y="190">pixel: counts</text>
<text x="240" y="215">electrons e⁻</text>
<line x1="150" x2="150" y1="246" y2="292" stroke="#333" stroke-width="3"/>
<path d="M141,284 l9,14 l9,-14 z" fill="#333"/>
<text x="240" y="282" fill="{C[0]}">+ read noise (a few e⁻)</text>
<rect x="60" y="304" width="180" height="78" rx="10" fill="#f4f4f4" stroke="#333" stroke-width="3"/>
<text x="150" y="338" text-anchor="middle" font-weight="bold">ADC</text>
<text x="150" y="366" text-anchor="middle">÷ gain g, round</text>
<text x="256" y="338">g e⁻ per ADU,</text>
<text x="256" y="364">b bits</text>
<line x1="150" x2="150" y1="388" y2="434" stroke="#333" stroke-width="3"/>
<path d="M141,426 l9,14 l9,-14 z" fill="#333"/>
<rect x="60" y="446" width="180" height="60" rx="6" fill="#fff" stroke="#333" stroke-width="3"/>
<text x="150" y="484" text-anchor="middle" font-family="monospace" font-size="24">1500</text>
<text x="256" y="474">raw value in ADU</text>
<text x="256" y="500" fill="#555">e.g. 3000 e⁻ ÷ 2 e⁻/ADU</text>
</svg>'''
open(OUT + "fig-pipeline.svg", "w").write(svg); print("wrote", OUT + "fig-pipeline.svg")

# 2. fig-shot.png: the micrograph as mean photon counts, average 10, 100, 10 000 per pixel
rng = np.random.default_rng(7)
means = [10, 100, 10000]
fig, axs = plt.subplots(1, len(means), figsize=(10.5, 3.9))
for ax, m in zip(axs, means):
    lam = cell / cell.mean() * m
    k = rng.poisson(lam)
    ax.imshow(k[40:232, 40:232], cmap="gray", vmin=0, vmax=np.percentile(k, 99.5), interpolation="nearest")
    ax.set_title(f"{m:,} e⁻: SNR ≈ {math.sqrt(m):.0f}".replace(",", " "), fontsize=23); ax.axis("off")
fig.tight_layout(pad=0.4); save(fig, "fig-shot.png", dpi=110, bbox_inches="tight", pad_inches=0.08)

# 3. fig-poisson.svg: Poisson counts against the Gaussian with the same mean and variance
fig, axs = plt.subplots(2, 1, figsize=(5.2, 5.6))
for ax, m in zip(axs, (3, 100)):
    k = np.arange(0, int(m + 5 * math.sqrt(m)) + 2)
    if m < 10:
        ax.bar(k, poisson.pmf(k, m), color=C[0], alpha=0.55, width=0.8, label="Poisson")
    else:
        ks = k[(k > m - 4.5 * math.sqrt(m))]
        ax.bar(ks, poisson.pmf(ks, m), color=C[0], alpha=0.55, width=1.0, label="Poisson")
        k = ks
    x = np.linspace(k.min() - 2, k.max(), 400)
    ax.plot(x, norm.pdf(x, m, math.sqrt(m)), color=C[1], lw=2.5, label="Gaussian,\nsame mean\nand variance")
    ax.set_yticks([]); ax.set_xlabel(f"electrons (mean {m})", fontsize=16)
    ax.spines["left"].set_visible(False)
axs[0].set_xlim(-2.5, 12)
axs[0].legend(fontsize=16, frameon=False, loc="upper right", bbox_to_anchor=(1.06, 1.05))
fig.tight_layout()
save(fig, "fig-poisson.svg", format="svg")

# 4. fig-ptc.svg: noise against signal (electrons, log-log) for FWC 30 000 e-, read 1.5 e-
fwc, rd = 30000, 1.5
S = np.logspace(-1, math.log10(fwc), 300)
g8, g16 = fwc / 255, fwc / 65535
fig, ax = plt.subplots(figsize=(7.4, 6.2))
ax.plot(S, np.sqrt(S), color=C[2], lw=2.5, label="shot noise √S")
ax.plot(S, np.full_like(S, rd), color=C[0], lw=2.5, label=f"read noise {rd} e⁻")
ax.plot(S, np.full_like(S, g8 / math.sqrt(12)), color=C[1], lw=2.5, ls="--", label=f"rounding, 8-bit ADC ({g8 / math.sqrt(12):.0f} e⁻)")
ax.plot(S, np.full_like(S, g16 / math.sqrt(12)), color=C[1], lw=1.8, ls=":", label=f"rounding, 16-bit ADC ({g16 / math.sqrt(12):.2f} e⁻)")
ax.axvline(fwc, color="#888", ls="--", lw=1.2); ax.text(fwc / 1.15, 0.06, "full well", ha="right", color="#555", fontsize=15)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.05, 400); ax.set_xlim(0.1, 6e4)
ax.annotate("", xy=(rd, 160), xytext=(fwc, 160), arrowprops=dict(arrowstyle="<->", color="#333", lw=1.8))
dr = fwc / rd
ax.text(math.sqrt(rd * fwc), 200, f"dynamic range {dr:,.0f} : 1 = {20 * math.log10(dr):.0f} dB = {math.log2(dr):.1f} bits".replace(",", " "),
        ha="center", fontsize=15)
ax.set_xlabel("mean signal S (electrons)"); ax.set_ylabel("noise SD (electrons)")
ax.legend(fontsize=15.5, frameon=False, loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=2, handlelength=1.5, columnspacing=0.8)
fig.tight_layout(); save(fig, "fig-ptc.svg", format="svg")
print("g8", g8, "q8", g8 / math.sqrt(12), "dr", dr, 20 * math.log10(dr), math.log2(dr))
