"""Slide figures of the sam/ lecture (Networks for cells and objects day).

Run from the repository root (after `make wasm` and `make models`):
  python3 -I tools/figs/sam/to_raw.py RAW
  node_modules/.bin/esbuild tools/figs/sam/dump_samnet.ts --bundle --platform=node --log-level=warning > d.cjs && node d.cjs RAW DUMP
  python3 -I tools/figs/sam/lec_figs.py DUMP

Synthetic diagrams (block diagram with the measured costs, cost curves, stability, memory) are
drawn here; the real-network figures use the outputs of SAM 2.1 Hiera tiny dumped by
dump_samnet.ts (the same WebAssembly module and post-processing as the samnet demo).
Timings: demos/wasm/sam/README.md. Images: StarDist test image (BSD-3-Clause) and the 2018 Data
Science Bowl (BBBC038, CC0), credited in the captions.
"""
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "sam"
DUMP = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
BLUE, ORANGE, PURPLE, GREEN, RED, GREY = "#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#d62728", "#9a9a9a"
CAND = [BLUE, ORANGE, PURPLE]
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none", "font.family": "DejaVu Sans"})
ENC, DEC = 26.6, 0.42


def box(ax, x, y, w, h, text, fc, ec="#333", fs=16, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#333"))


# 1. Three parts with their costs ----------------------------------------------------------------
def fig_parts():
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set_xlim(0, 11.4); ax.set_ylim(0, 5.4); ax.axis("off")
    box(ax, 0.1, 3.3, 1.9, 1.2, "image\n(→ 1024²)", "#eeeeee", fs=15)
    box(ax, 2.5, 3.1, 2.8, 1.6, "image encoder\nheavy, ONCE\n26.6 s", "#cfe3f5", fs=15)
    box(ax, 5.8, 3.3, 2.3, 1.2, "embedding\n64×64×256", "#eeeeee", fs=15)
    box(ax, 0.1, 0.6, 1.9, 1.3, "prompts:\nclicks, box,\nmask", "#eeeeee", fs=14)
    box(ax, 2.5, 0.6, 2.8, 1.3, "prompt encoder\n→ a few tokens", "#fde3c8", fs=15)
    box(ax, 5.8, 0.5, 2.3, 1.5, "mask decoder\nper prompt\n0.42 s", "#fde3c8", fs=15)
    box(ax, 8.6, 0.4, 2.7, 1.7, "3 masks,\npredicted IoU\nfor each", "#e3f1e3", fs=15)
    arrow(ax, 2.0, 3.9, 2.5, 3.9); arrow(ax, 5.3, 3.9, 5.8, 3.9)
    arrow(ax, 2.0, 1.25, 2.5, 1.25); arrow(ax, 5.3, 1.25, 5.8, 1.25)
    arrow(ax, 6.95, 3.3, 6.95, 2.0); arrow(ax, 8.1, 1.25, 8.6, 1.25)
    ax.text(8.3, 3.9, "kept\n(cached)", fontsize=15, style="italic", color="#555", va="center")
    ax.text(5.7, 5.15, "times measured in the browser (one CPU core, 512² image)", fontsize=13, color="#555", ha="center")
    fig.savefig(OUT / "fig-parts.svg", bbox_inches="tight"); plt.close(fig)


# 2. Cost of m clicks ----------------------------------------------------------------------------
def fig_cost():
    m = np.arange(0, 41)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(m, m * (ENC + DEC), color=RED, lw=2.5, label="re-encode for every click")
    ax.plot(m, np.where(m > 0, ENC + m * DEC, 0), color=BLUE, lw=2.5, label="encode once, decode per click")
    ax.set_xlabel("number of clicks m"); ax.set_ylabel("total time (s)")
    ax.annotate(f"{ENC + 40 * DEC:.0f} s", (40, ENC + 40 * DEC), (31, 230), color=BLUE, fontsize=16, arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.annotate(f"{40 * (ENC + DEC):.0f} s", (40, 40 * (ENC + DEC)), (31, 650), color=RED, fontsize=16, arrowprops=dict(arrowstyle="->", color=RED))
    ax.legend(frameon=False, fontsize=14, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "fig-cost.svg", bbox_inches="tight"); plt.close(fig)


# 3. Stability: steep vs shallow logits at the border ---------------------------------------------
def fig_stability():
    # Disc-shaped masks; logit linear in the radius. Radii from the areas (π r² = area).
    cases = [("shallow border", 380, 430, ORANGE), ("steep border", 400, 412, GREEN)]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.2))
    r = np.linspace(8, 14, 400)
    for ax, (title, a_in, a_out, col) in zip(axs, cases):
        r_in, r_out = np.sqrt(a_in / np.pi), np.sqrt(a_out / np.pi)
        slope = 2 / (r_out - r_in); r0 = (r_in + r_out) / 2
        ax.plot(r, -slope * (r - r0), color=col, lw=2.5)
        for t, ls in ((1, "--"), (0, "-"), (-1, "--")):
            ax.axhline(t, color="#555", ls=ls, lw=1)
        ax.axvspan(r_in, r_out, color=col, alpha=0.2)
        ax.set_ylim(-4, 4); ax.set_xlim(8, 14)
        ax.set_xlabel("distance from the centre (px)")
        ax.set_title(f"{title}\n{a_in} / {a_out} = {a_in / a_out:.2f}", fontsize=16)
        ax.text(8.2, 1.15, "ℓ = +1", fontsize=13, color="#555"); ax.text(8.2, -1.6, "ℓ = −1", fontsize=13, color="#555")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("mask logit ℓ")
    fig.tight_layout()
    fig.savefig(OUT / "fig-stability.svg", bbox_inches="tight"); plt.close(fig)


# 4. Memory for video / z-stacks -----------------------------------------------------------------
def fig_memory():
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    xs = [0.3, 2.6, 4.9, 7.2]
    for i, x in enumerate(xs):
        ax.add_patch(Rectangle((x, 1.9), 1.9, 1.9, fc="#111", ec="#333"))
        cx, cy = x + 0.7 + 0.18 * i, 2.85 + 0.08 * np.sin(i)
        ax.add_patch(Circle((cx, cy), 0.42, fc="#cfcfcf", ec=BLUE if i else GREEN, lw=3))
        ax.text(x + 0.95, 3.9, f"frame {i + 1}" + (": one click" if i == 0 else ""), ha="center", va="bottom", fontsize=14,
                color=GREEN if i == 0 else "black")
        if i == 0:
            ax.add_patch(Circle((cx, cy), 0.08, fc=GREEN, ec="white"))
    box(ax, 0.3, 0.1, 8.8, 0.9, "memory: embeddings + masks of earlier frames", "#fde3c8", fs=15)
    for x in xs[:-1]:
        arrow(ax, x + 0.6, 1.9, x + 0.6, 1.0)
    for x in xs[1:]:
        arrow(ax, x + 1.3, 1.0, x + 1.3, 1.9)
    fig.savefig(OUT / "fig-memory.svg", bbox_inches="tight"); plt.close(fig)


# ---- real-network figures ---------------------------------------------------------------------
def load_dump():
    meta = json.loads((DUMP / "meta.json").read_text())
    return meta


def img_of(name):
    stem = {"nuclei": "stardist-img2d", "brightfield": "dsb-brightfield"}[name]
    return np.asarray(Image.open(ROOT / f"demos/dist/models/images/{stem}.png").convert("RGB"))


def overlay(rgb, mask, col, alpha=0.45):
    out = rgb.astype(float) / 255
    c = np.array(matplotlib.colors.to_rgb(col))
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * c
    edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    out[edge] = c
    return out


def masks_of(meta, img, name):
    w, h = meta[img]["w"], meta[img]["h"]
    return np.fromfile(DUMP / f"{img}-{name}.masks", dtype=np.uint8).reshape(3, h, w)


def prompt_marks(ax, pr, x0, y0):
    for (x, y, l) in pr["points"]:
        ax.plot(x - x0, y - y0, "o", ms=11, mfc=GREEN if l else RED, mec="white", mew=2)
    if pr["box"]:
        b = pr["box"]
        ax.add_patch(Rectangle((b[0] - x0, b[1] - y0), b[2] - b[0], b[3] - b[1], fill=False, ec=ORANGE, lw=2.5))


def fig_candidates(meta, rows, out, crops, labels=None):
    rgb = img_of("nuclei")
    fig, axs = plt.subplots(len(rows), 3, figsize=(9.6, 3.5 * len(rows)), squeeze=False, gridspec_kw={"wspace": 0.04, "hspace": 0.3})
    for r, name in enumerate(rows):
        x0, y0, s = crops[r]
        pr = meta["nuclei"]["prompts"][name]
        M = masks_of(meta, "nuclei", name)
        best = int(np.argmax(pr["iou"]))
        for k in range(3):
            a = axs[r, k]
            a.imshow(overlay(rgb, M[k], CAND[k])[y0:y0 + s, x0:x0 + s]); a.axis("off")
            prompt_marks(a, pr, x0, y0)
            a.set_title(f"mask {k + 1}: score {pr['iou'][k]:.2f}" + ("  ◀" if k == best else ""), fontsize=16)
            area = int(M[k].sum())
            a.text(0.5, -0.02, f"{area} px" if area < 0.5 * M[k].size else f"{area} px (whole image)", transform=a.transAxes, ha="center", va="top", fontsize=14)
        if labels:
            axs[r, 0].text(-0.06, 0.5, labels[r], transform=axs[r, 0].transAxes, rotation=90, ha="right", va="center", fontsize=16)
    fig.savefig(OUT / out, dpi=90, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_embedding(meta):
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.8), gridspec_kw={"wspace": 0.05})
    rgb = img_of("nuclei")
    emb = np.fromfile(DUMP / "nuclei-emb.rgba", dtype=np.uint8).reshape(64, 64, 4)[..., :3]
    axs[0].imshow(rgb); axs[0].set_title("image, 512 × 512", fontsize=16)
    axs[1].imshow(emb, interpolation="nearest"); axs[1].set_title("embedding, 64 × 64: 3 main directions", fontsize=16)
    # One embedding cell = 8 px of this image: draw a few cells of the grid on the image.
    for k in range(0, 9):
        axs[0].plot([300 + 8 * k] * 2, [300, 364], color="yellow", lw=0.8)
        axs[0].plot([300, 364], [300 + 8 * k] * 2, color="yellow", lw=0.8)
    axs[0].text(332, 296, "8 × 8 cells", color="yellow", ha="center", va="bottom", fontsize=13)
    for a in axs: a.axis("off")
    fig.savefig(OUT / "fig-embedding.png", dpi=90, bbox_inches="tight", facecolor="white"); plt.close(fig)


FATE_COL = {"kept": "white", "nms": "#b07cff", "stability": ORANGE, "iou": GREY, "empty": GREY}


def fig_auto(meta, img, out, crop=None, title=""):
    rgb = img_of(img)
    m = meta[img]; w, h = m["w"], m["h"]
    lab = np.fromfile(DUMP / f"{img}-auto.lab", dtype=np.int32).reshape(h, w)
    o = rgb.astype(float) / 255
    cols = plt.get_cmap("tab10")
    for i in range(1, lab.max() + 1):
        mk = lab == i
        a = 0.15 if mk.sum() > 0.5 * w * h else 0.5
        o = overlay((o * 255).astype(np.uint8), mk, cols((i - 1) % 10), a)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(o)
    for (x, y), fa in zip(m["auto"]["grid"], m["auto"]["fates"]):
        ax.plot(x, y, "o", ms=8, mfc=FATE_COL[fa] if fa == "kept" else "none", mec="black" if fa == "kept" else FATE_COL[fa], mew=2.2)
    c = m["auto"]["counts"]
    ax.set_title(title or f"8 × 8 points → {c['kept']} masks", fontsize=16)
    ax.axis("off")
    xs = [0.0, 0.18, 0.48, 0.775]
    for x, lab, fa in zip(xs, ["kept", "duplicate", "unstable", "low score"], ["kept", "nms", "stability", "iou"]):
        ax.plot([x], [-0.05], "o", ms=9, transform=ax.transAxes, clip_on=False, mfc="white" if fa == "kept" else "none",
                mec="black" if fa == "kept" else FATE_COL[fa], mew=2.2)
        ax.text(x + 0.035, -0.05, lab, transform=ax.transAxes, va="center", fontsize=14)
    fig.savefig(OUT / out, dpi=90, bbox_inches="tight", facecolor="white"); plt.close(fig)


fig_parts(); fig_cost(); fig_stability(); fig_memory()
if DUMP and (DUMP / "meta.json").exists():
    meta = load_dump()
    fig_candidates(meta, ["cluster"], "fig-cluster.png", [(60, 340, 172)])
    fig_candidates(meta, ["negnb", "box"], "fig-negative.png", [(60, 340, 172), (150, 180, 172)], labels=["+ negative click", "box"])
    fig_embedding(meta)
    fig_auto(meta, "nuclei", "fig-auto.png")
    fig_auto(meta, "brightfield", "fig-auto-bf.png", title="bright field: whole cells, not nuclei")
    print(json.dumps({k: {n: p["iou"] for n, p in v["prompts"].items()} for k, v in meta.items()}))
    print(json.dumps({k: v["auto"]["counts"] for k, v in meta.items()}))
print("figures written to", OUT)
