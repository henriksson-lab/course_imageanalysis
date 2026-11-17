// lowrank: a weight change ΔW and its best rank-r approximation B·A (as in LoRA).
// Top: the change ΔW, its rank-r approximation, and what is left out (same colour scale:
// blue negative, red positive). Bottom left: the singular values, the kept ones coloured.
// Panel: the rank slider, the numbers to store (full: m·n, low rank: r·(m + n)), the relative
// error, and a plot of the error against the rank. The presets "real fine-tuning" use the
// change that full fine-tuning made to the toy network of the transfercurves demo, and also
// show the network's accuracy when its change is cut to rank r.
//
// <div data-demo="lowrank" data-preset="rank2"></div>
//   data-preset  rank2 | noisy | fullrank | real1 | real2
import { autoMount, htmlEl, presets as presetRow, slider, type Slider } from "../lib/ui";
import { Plot, seriesColor, type Mark } from "../lib/plot";
import { drawScalarImage, diverging } from "../lib/colour";
import type { SVD } from "../lib/linalg";
import { syntheticChange, decompose, truncate, relError, fullParams, lowRankParams, realChange, REAL, type Change } from "./model";

const NAME = "lowrank";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .lr-grid { flex: 1 1 480px; max-width: 500px; min-width: 0; display: grid; grid-template-columns: 1fr 1fr; gap: .4em .8em; align-items: start; }
[data-demo="${NAME}"] .lr-grid figure { margin: 0; }
[data-demo="${NAME}"] .lr-grid figcaption { font-size: 15px; text-align: center; margin-bottom: .15em; }
[data-demo="${NAME}"] .lr-grid canvas { width: 100%; object-fit: fill; image-rendering: pixelated; border: 1px solid var(--qb-grid); display: block; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 320px; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .lr-grid { flex-basis: 480px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface Preset { id: string; label: string; make: () => { c: Change; S: SVD; real?: 0 | 1 }; r: number }
const PRESETS: Preset[] = [
  { id: "rank2", label: "a rank-2 change", r: 1, make: () => { const c = syntheticChange(24, 24, [3, 1.5], 0, 3); return { c, S: decompose(c) }; } },
  { id: "noisy", label: "rank 2 + noise", r: 2, make: () => { const c = syntheticChange(24, 24, [3, 1.5], 0.25, 3); return { c, S: decompose(c) }; } },
  { id: "fullrank", label: "a random (full-rank) change", r: 4, make: () => { const c = syntheticChange(24, 24, [], 3, 5); return { c, S: decompose(c) }; } },
  { id: "real1", label: "real fine-tuning: layer 1", r: 3, make: () => { const R = realChange(); return { c: R.layers[0], S: R.svds[0], real: 0 }; } },
  { id: "real2", label: "real fine-tuning: layer 2", r: 3, make: () => { const R = realChange(); return { c: R.layers[1], S: R.svds[1], real: 1 }; } },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const start = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];

  const grid = htmlEl("div", "lr-grid", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);
  const fig = (cap: string) => { const f = htmlEl("figure", "", "", grid); const c = htmlEl("figcaption", "", cap, f); return { c, cv: htmlEl("canvas", "", "", f) }; };
  const fD = fig("ΔW: the change"), fR = fig("rank-r approximation B·A"), fE = fig("left out: ΔW − B·A");
  const svDiv = htmlEl("figure", "", "", grid);
  htmlEl("figcaption", "", "singular values", svDiv);
  const svPlot = new Plot(svDiv, { width: 300, height: 210, x: [0, 25], y: [0, 1], xlabel: "k", margin: { l: 50, b: 50, t: 8, r: 8 }, font: 17 });
  const keptBars: Mark<any> = svPlot.bars([], [], { color: seriesColor(1), barWidth: 0.8 });
  const restBars: Mark<any> = svPlot.bars([], [], { color: "var(--qb-muted)", barWidth: 0.8 });

  presetRow(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  let sR: Slider = slider(panel, "rank r", 0, 24, 1, start.r, v => { r = v; draw(); });
  const readout = htmlEl("div", "qb-readout", "", panel);
  const errDiv = htmlEl("div", "", "", panel);
  const errPlot = new Plot(errDiv, { width: 320, height: 200, x: [0, 24], y: [0, 1], xlabel: "rank r", margin: { l: 50, b: 50, t: 10, r: 10 }, font: 17 });
  const errLine: Mark<any> = errPlot.line([], [], { color: seriesColor(1), width: 3, label: "relative error" });
  const accLine: Mark<any> = errPlot.line([], [], { color: seriesColor(0), width: 3, label: "accuracy" });
  const oldLine: Mark<any> = errPlot.line([], [], { color: seriesColor(3), width: 2.5, dash: "6 4", label: "old task" });
  const marker: Mark<any> = errPlot.vline(start.r, { color: "var(--qb-fg)", width: 2 });
  const legend = { g: null as SVGGElement | null };
  const note = htmlEl("div", "qb-note", "", panel);

  let cur: { c: Change; S: SVD; real?: 0 | 1 } | null = null;
  let r = start.r;

  function load(p: Preset): void {
    if (p.id.startsWith("real")) note.textContent = "fine-tuning the network (once)…";
    setTimeout(() => {
      cur = p.make();
      const maxR = Math.min(cur.c.m, cur.c.n, cur.real !== undefined ? REAL.maxRank : 24);
      const old = sR.input.parentElement!;
      sR = slider(panel, "rank r", 0, maxR, 1, Math.min(p.r, maxR), v => { r = v; draw(); });
      old.replaceWith(sR.input.parentElement!);
      r = Math.min(p.r, maxR);
      const k = cur.S.S.length;
      svPlot.setDomain([0, Math.min(k, 33) + 0.5], [0, cur.S.S[0] * 1.05]);
      errPlot.setDomain([0, maxR], [0, 1]);
      const rs = Array.from({ length: maxR + 1 }, (_, i) => i);
      errLine.set(rs, rs.map(q => relError(cur!.S.S, q)));
      if (cur.real !== undefined) {
        const R = realChange();
        accLine.set(rs, rs.map(q => R.accByRank[q])); oldLine.set(rs, rs.map(q => R.oldByRank[q]));
      } else { accLine.set([], []); oldLine.set([], []); }
      legend.g?.remove();
      legend.g = errPlot.legend(cur.real !== undefined ? undefined : [{ label: "relative error", color: seriesColor(1) }], "tr");
      note.textContent = cur.real !== undefined
        ? `Full fine-tuning's change of layer ${cur.real + 1} (${REAL.n} labels). Accuracy: both layers' changes cut to rank r.`
        : "A made-up change of a 24 × 24 weight matrix.";
      draw();
    }, 20);
  }

  function draw(): void {
    if (!cur) return;
    const { c, S } = cur;
    const Dr = truncate(S, c.m, c.n, r);
    const E = c.D.map((v, i) => v - Dr[i]);
    const absSorted = Array.from(c.D, Math.abs).sort((a, b) => a - b);
    const mx = absSorted[Math.floor(0.99 * (absSorted.length - 1))] || 1;   // colour scale: 99th percentile of |ΔW|
    const opts = { map: diverging, lo: -mx, hi: mx };
    for (const [f, V] of [[fD, c.D], [fR, Dr], [fE, E]] as const) {
      drawScalarImage(f.cv, V, c.n, c.m, opts);
      f.cv.style.aspectRatio = String(Math.min(c.n / c.m, 2));   // wide matrices: cells drawn taller
    }
    fR.c.textContent = `rank-${r} approximation B·A`;
    const ks = Array.from(S.S, (_, i) => i + 1).slice(0, 33);
    const sv = Array.from(S.S).slice(0, 33);
    keptBars.set(ks.slice(0, r), sv.slice(0, r));
    restBars.set(ks.slice(r), sv.slice(r));
    marker.set(r);
    const full = fullParams(c.m, c.n), low = lowRankParams(c.m, c.n, r);
    const err = relError(S.S, r);
    let txt = `ΔW: ${c.m} × ${c.n} = ${full} numbers\nB·A: ${r}·(${c.m} + ${c.n}) = ${low} numbers (${(100 * low / full).toFixed(0)} %)\nrelative error ${(100 * err).toFixed(1)} %`;
    if (cur.real !== undefined) {
      const R = realChange();
      txt += `\naccuracy ${(100 * R.accByRank[r]).toFixed(1)} % (full ΔW: ${(100 * R.accFull).toFixed(1)} %)\nold task ${(100 * R.oldByRank[r]).toFixed(0)} % (before: ${(100 * R.oldByRank[0]).toFixed(0)} %)`;
    }
    readout.textContent = txt;
  }

  load(start);
}

autoMount(NAME, mount);
