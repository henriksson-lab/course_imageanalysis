// Sub-pixel peaks (FFT registration day). B is A shifted by 3 + a fraction of a pixel (by the
// Fourier phase ramp, so the truth is exact). Left: the correlation samples near the peak,
// the parabola through three of them, and the upsampled-DFT curve; vertical lines mark the
// truth and the three estimates. Right: the error of each method for true fractions 0 … 1
// (the parabola's S-curve is "pixel locking").
// Options: data-preset="parabola|upsampled|noise|cc".
import { htmlEl, slider, presets, autoMount, button } from "../lib/ui";
import { Plot, seriesColor, plotFigure } from "../lib/plot";
import { loadCell } from "../lib/cell";
import { estimate, sweep, signalOf, SubSettings, SubSignal } from "./model";

const NAME = "subpixel";
const STYLE = `
[data-demo="${NAME}"] .sp-plots { flex: 1 1 560px; min-width: 0; display: grid; grid-template-columns: 1fr 1fr; gap: .3em .8em; max-width: 860px; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 320px; gap: .35em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .sp-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .12em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 13px; }
@media (max-width: 700px) { [data-demo="${NAME}"] .sp-plots { grid-template-columns: 1fr; } }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

interface Preset { id: string; label: string; s: Partial<SubSettings> }
const PRESETS: Preset[] = [
  { id: "parabola", label: "parabola: pixel locking", s: { method: "phase", noise: 0, frac: 0.3 } },
  { id: "cc", label: "a broad CC peak", s: { method: "cc", noise: 0, frac: 0.3 } },
  { id: "upsampled", label: "upsampled DFT: flat error", s: { method: "phase", noise: 0, U: 20, frac: 0.7 } },
  { id: "coarse", label: "upsampling too coarse", s: { method: "phase", noise: 0, U: 4, frac: 0.6 } },
  { id: "noise", label: "noise sets the floor", s: { method: "cc", noise: 10, frac: 0.3 } },
];
const DEF: SubSettings = { signal: "bumps", noise: 0, U: 20, method: "phase", frac: 0.3 };
const COL = { truth: "var(--qb-c3)", par: seriesColor(1), cen: seriesColor(3), up: seriesColor(0) };

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: SubSettings = { ...DEF };
  let row: Float64Array | null = null;

  const plots = htmlEl("div", "sp-plots", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);
  const f1 = plotFigure(plots, "correlation near the peak");
  const P1 = new Plot(f1, { width: 430, height: 300, x: [0, 6], y: [0, 1], xlabel: "shift t", ylabel: "c(t)" });
  const f2 = plotFigure(plots, "error for every true fraction");
  const P2 = new Plot(f2, { width: 430, height: 300, x: [0, 1], y: [-0.2, 0.2], xlabel: "true fraction of a pixel", ylabel: "estimate − truth (px)" });

  htmlEl("div", "sp-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "sp-lab", "correlation · signal", panel);
  const r1 = htmlEl("div", "qb-presets", "", panel);
  const mBtns = [button(r1, "CC", () => { st.method = "cc"; update(); }), button(r1, "phase", () => { st.method = "phase"; update(); })];
  const sigs: [SubSignal, string][] = [["bumps", "bumps"], ["profile", "micrograph"]];
  const sBtns = sigs.map(([k, l]) => button(r1, l, () => { st.signal = k; update(); }));
  const fS = slider(panel, "true fraction", 0, 1, 0.01, st.frac, v => { st.frac = v; update(false); });
  const nS = slider(panel, "noise", 0, 30, 1, 0, v => { st.noise = v; update(); });
  const uS = slider(panel, "upsampling U", 2, 50, 1, 20, v => { st.U = v; update(); });
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Green: the truth (3 + fraction). Red: parabola through the highest sample and its two neighbours. Orange: centroid of the same three samples. Blue: the upsampled DFT (curve and its maximum, on a grid of 1/U px).", panel);

  function load(p: Preset): void {
    Object.assign(st, DEF, p.s);
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    update();
  }

  let sw: ReturnType<typeof sweep> | null = null;
  const marks: { remove(): void }[] = [];
  function update(resweep = true): void {
    if (st.signal === "profile" && !row) { loadCell().then(g => { row = Float64Array.from(g.data.subarray(128 * 256, 129 * 256)); update(); }); return; }
    mBtns.forEach((b, i) => b.classList.toggle("active", ["cc", "phase"][i] === st.method));
    sBtns.forEach((b, i) => b.classList.toggle("active", sigs[i][0] === st.signal));
    fS.set(st.frac); nS.set(st.noise); uS.set(st.U);
    const a = signalOf(st.signal, row);
    const e = estimate(a, st);
    if (resweep || !sw) sw = sweep({ ...st, steps: 51 }, row);
    marks.forEach(m => m.remove()); marks.length = 0;
    // left plot
    const t0 = e.t0, n = e.c.length;
    const xs = [-2, -1, 0, 1, 2].map(d => t0 + d), ys = xs.map(x => e.c[((x % n) + n) % n]);
    const lo = Math.min(...ys, ...e.curve.cs), hi = Math.max(...ys, ...e.curve.cs), pd = 0.08 * (hi - lo || 1);
    P1.setDomain([t0 - 2.6, t0 + 2.6], [lo - pd, hi + 3 * pd]);
    marks.push(P1.line(e.curve.xs, e.curve.cs, { color: COL.up, width: 2.5, label: "upsampled DFT" }));
    // parabola through the three middle samples
    const [cm, c0, cp] = [ys[1], ys[2], ys[3]];
    const A = (cm + cp) / 2 - c0, B = (cp - cm) / 2;
    marks.push(P1.fn(x => { const u = x - t0; return Math.abs(u) <= 1.6 ? A * u * u + B * u + c0 : NaN; }, { color: COL.par, width: 2.5, dash: "7 5", label: "parabola" }));
    marks.push(P1.scatter(xs, ys, { color: "var(--qb-fg)", r: 7 }));
    marks.push(P1.vline(e.shift, { color: COL.truth, width: 3, dash: "" }));
    marks.push(P1.vline(e.parabola, { color: COL.par, width: 2.5 }));
    marks.push(P1.vline(e.centroid, { color: COL.cen, width: 2.5 }));
    marks.push(P1.vline(e.up, { color: COL.up, width: 2.5 }));
    P1.legend(undefined, "tl");
    // right plot
    const all = [...sw.parabola, ...sw.centroid, ...sw.up].map(Math.abs).filter(v => v < 0.6);
    const ym = Math.max(0.05, Math.min(0.6, 1.15 * Math.max(...all, 0)));
    P2.setDomain([0, 1], [-ym, ym]);
    marks.push(P2.hline(0, { color: "var(--qb-axis)", dash: "", width: 1 }));
    marks.push(P2.line(sw.fracs, sw.parabola, { color: COL.par, width: 3, label: "parabola" }));
    marks.push(P2.line(sw.fracs, sw.centroid, { color: COL.cen, width: 3, label: "centroid" }));
    marks.push(P2.line(sw.fracs, sw.up, { color: COL.up, width: 3, label: `upsampled (U = ${st.U})` }));
    marks.push(P2.vline(st.frac, { width: 2 }));
    P2.legend(undefined, "tr");
    const f = (v: number) => v.toFixed(3), d = (v: number) => (v - e.shift >= 0 ? "+" : "") + (v - e.shift).toFixed(3);
    const mx = (a: number[]) => Math.max(...a.map(Math.abs)).toFixed(3);
    ro.textContent = [
      `truth       ${f(e.shift)}`,
      `parabola    ${f(e.parabola)}  (${d(e.parabola)})`,
      `centroid    ${f(e.centroid)}  (${d(e.centroid)})`,
      `upsampled   ${f(e.up)}  (${d(e.up)})`,
      `worst error over all fractions:`,
      `  parabola ${mx(sw.parabola)}  centroid ${mx(sw.centroid)}  upsampled ${mx(sw.up)}`,
    ].join("\n");
  }

  load(p0);
}

autoMount(NAME, mount);
