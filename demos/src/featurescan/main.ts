// Many features, one plate: m features per well, some with a real treatment effect, plate
// effects (gradients, edge wells) and a layout of treated and control wells. Left: the plate
// as a heat map of one feature (treated wells outlined). Middle: the histogram of all
// p-values (flat when nothing is real). Right: the sorted p-values against their rank (log
// axes) with the lines of an uncorrected test (α), Bonferroni (α/m) and Benjamini–Hochberg
// (k α/m); dots green = real feature, red = null feature (known from the simulation).
// Auto-mounts on <div data-demo="featurescan">. Options: data-preset="null|real|confounded|randomised|edge",
// data-panels="plate,hist,rank,controls" (any subset).
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl } from "../lib/ui";
import { Plot, seriesColor, plotFigure } from "../lib/plot";
import { diverging } from "../lib/colour";
import { PlateLayout } from "../lib/stats";
import * as M from "./model";

const NAME = "featurescan";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] .fs-plate { flex: 1 1 340px; max-width: 420px; min-width: 240px; }
[data-demo="${NAME}"] .fs-plots { flex: 0 1 300px; min-width: 220px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .fs-plots .qb-plot { flex: none; max-width: none; }
[data-demo="${NAME}"] .fs-plots svg { max-height: calc(50vh - 90px); margin: 0 auto; }
[data-demo="${NAME}"] .qb-figure figcaption { font-size: 15px; }
[data-demo="${NAME}"] .qb-readout { font-size: 13px; white-space: pre; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 260px; gap: .25em; font-size: 15px; }
[data-demo="${NAME}"] .fs-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .15em; flex-basis: 250px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 13px; }
.reveal [data-demo="${NAME}"] .fs-plate { flex-basis: 330px; }
.reveal [data-demo="${NAME}"] .fs-plots { flex-basis: 360px; }
.reveal [data-demo="${NAME}"] .fs-plots svg { max-height: 480px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

// lib/plot.ts numbers its clip paths per bundle ("qb-clip-1", …), so two demos on one page
// share ids and clip each other's plots. Give every plot of this demo a unique clip id.
let clipCount = 0;
function uniqueClip<T extends { svg: SVGSVGElement; layer: SVGGElement }>(p: T): T {
  const cp = p.svg.querySelector("clipPath");
  if (cp) { cp.id = `${NAME}-clip-${++clipCount}`; p.layer.setAttribute("clip-path", `url(#${cp.id})`); }
  return p;
}

const LAYOUTS: { id: PlateLayout; label: string }[] = [{ id: "left", label: "treated left" }, { id: "random", label: "random" }, { id: "blocked", label: "random per row" }];

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = M.PRESETS.find(p => p.id === host.dataset.preset) ?? M.PRESETS[0];
  let st: M.Settings = { ...M.DEFAULTS, ...p0.settings }, seed = M.SEED, feat = 0;
  const slide = !!host.closest(".reveal"), fs = slide ? 26 : 18;
  const panels = new Set((host.dataset.panels ?? "plate,hist,rank,controls").split(/[\s,]+/));

  const left = htmlEl("div", "fs-plate", "", host);
  const plateF = plotFigure(left, "plate: one feature");
  const featRow = htmlEl("div", "qb-presets", "", left);
  featRow.style.justifyContent = "center";
  button(featRow, "smallest p", () => { feat = ana.p.indexOf(Math.min(...ana.p)); drawPlate(); });
  button(featRow, "a real feature", () => { const k = plate.real.indexOf(true); if (k >= 0) { feat = k; drawPlate(); } });
  button(featRow, "a null feature", () => { feat = plate.real.indexOf(false); drawPlate(); });
  const plots = htmlEl("div", "fs-plots", "", host);
  const histF = plotFigure(plots, "all p-values");
  const rankF = plotFigure(plots, "sorted p-values, three thresholds");
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "fs-lab", "presets", panel);
  const pbtn = presets(panel, M.PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const mS = slider(panel, "features m", 10, 500, 10, st.features, v => set({ features: v }));
  const rS = slider(panel, "share real", 0, 0.5, 0.05, st.fracReal, v => set({ fracReal: v }));
  const eS = slider(panel, "effect (SD)", 0, 1, 0.05, st.effect, v => set({ effect: v }));
  const cS = slider(panel, "left–right gradient", 0, 3, 0.1, st.colGradient, v => set({ colGradient: v }));
  const gS = slider(panel, "top–bottom gradient", 0, 3, 0.1, st.rowGradient, v => set({ rowGradient: v }));
  const edS = slider(panel, "edge wells", 0, 2, 0.1, st.edge, v => set({ edge: v }));
  htmlEl("div", "fs-lab", "layout", panel);
  const lrow = htmlEl("div", "qb-presets", "", panel);
  const lbtn = LAYOUTS.map(l => button(lrow, l.label, () => set({ layout: l.id })));
  const bigC = checkbox(panel, "384 wells (else 96)", st.big, v => set({ big: v }));
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "new plate", () => { seed++; recompute(); });
  button(row, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  if (!panels.has("plate")) left.style.display = "none";
  if (!panels.has("hist")) histF.style.display = "none";
  if (!panels.has("rank")) rankF.style.display = "none";
  if (!panels.has("hist") && !panels.has("rank")) plots.style.display = "none";
  if (!panels.has("controls")) panel.style.display = "none";

  let plate = M.makePlate(st, seed), ana = M.analyse(plate, st.alpha);

  function set(p: Partial<M.Settings>): void { st = { ...st, ...p }; pbtn.forEach(b => b.classList.remove("active")); recompute(); }
  function load(p: M.Preset): void {
    st = { ...M.DEFAULTS, ...p.settings }; seed = M.SEED;
    mS.set(st.features); rS.set(st.fracReal); eS.set(st.effect); cS.set(st.colGradient); gS.set(st.rowGradient); edS.set(st.edge); bigC.checked = st.big;
    pbtn.forEach((b, i) => b.classList.toggle("active", M.PRESETS[i] === p));
    recompute();
  }
  function recompute(): void {
    lbtn.forEach((b, i) => b.classList.toggle("active", LAYOUTS[i].id === st.layout));
    plate = M.makePlate(st, seed); ana = M.analyse(plate, st.alpha);
    feat = ana.p.indexOf(Math.min(...ana.p));
    drawPlate(); drawHist(); drawRank(); readout();
  }

  function drawPlate(): void {
    plateF.querySelector("svg")?.remove();
    const R = plate.rows, C = plate.cols, v = plate.values[feat];
    const pl = new Plot(plateF, { x: [0, C], y: [0, R], width: 480, height: 40 + 330 * R / C + 40, xticks: [], yticks: [], grid: false, margin: { l: 10, r: 10, t: slide ? 44 : 34, b: 10 }, font: slide ? 24 : 18, title: `feature ${feat + 1} (${plate.real[feat] ? "real" : "null"}), p = ${ana.p[feat] < 0.001 ? ana.p[feat].toExponential(1) : ana.p[feat].toFixed(3)}` });
    uniqueClip(pl);
    const lim = Math.max(...Array.from(v, Math.abs));
    pl.heatmap(v, C, R, { map: diverging, lo: -lim, hi: lim });
    pl.mark(g => {
      for (let y = 0; y < R; y++) for (let x = 0; x < C; x++) if (plate.cond[y * C + x]) {
        const x0 = pl.x(x), x1 = pl.x(x + 1), y0 = pl.y(R - y), y1 = pl.y(R - y - 1);
        const r = svgEl("rect", { x: Math.min(x0, x1) + 1.5, y: Math.min(y0, y1) + 1.5, width: Math.abs(x1 - x0) - 3, height: Math.abs(y1 - y0) - 3 }, g);
        r.style.fill = "none"; r.style.stroke = "var(--qb-fg)"; r.style.strokeWidth = "1.6";
      }
    }, []);
  }

  function drawHist(): void {
    histF.querySelector("svg")?.remove();
    const nb = 20, c = new Array(nb).fill(0), cn = new Array(nb).fill(0);
    ana.p.forEach((p, i) => { const k = Math.min(nb - 1, Math.floor(p * nb)); c[k]++; if (!plate.real[i]) cn[k]++; });
    const edges = Array.from({ length: nb + 1 }, (_, i) => i / nb);
    const pl = new Plot(histF, { x: [0, 1], y: [0, Math.max(...c, ana.p.length / nb) * 1.15], xlabel: "p-value", ylabel: "features", width: 420, height: 330, font: fs });
    uniqueClip(pl);
    pl.histogram(edges, c, { color: seriesColor(0), label: "all features" });
    pl.histogram(edges, cn, { color: "#999", label: "null features", fillOpacity: 0.35 });
    pl.hline(ana.p.length / nb, { color: seriesColor(1), width: 2, text: "flat: m/20" });
    pl.legend(undefined, "tr");
  }

  function drawRank(): void {
    rankF.querySelector("svg")?.remove();
    const m = ana.p.length, order = Array.from({ length: m }, (_, i) => i).sort((a, b) => ana.p[a] - ana.p[b]);
    const pmin = Math.max(1e-12, Math.min(...ana.p) / 3);
    const lo = Math.max(slide ? 1e-6 : 1e-8, Math.min(1e-4, 10 ** Math.floor(Math.log10(pmin))));
    const pl = new Plot(rankF, { x: [1, m], xlog: true, y: [lo, 1], ylog: true, xlabel: "rank k", ylabel: "p-value (sorted)", width: 420, height: 330, font: fs, margin: { l: 2 * fs + 62 } });
    uniqueClip(pl);
    const a = st.alpha;
    pl.line([1, m], [a, a], { color: seriesColor(3), width: 2.5, label: "α (uncorrected)" });
    pl.line([1, m], [a / m, a / m], { color: seriesColor(4), width: 2.5, label: "α/m (Bonferroni)" });
    pl.line([1, m], [a / m, a], { color: seriesColor(0), width: 2.5, dash: "6 4", label: "kα/m (BH)" });
    const ks = order.map((_, k) => k + 1), ps = order.map(i => Math.max(lo, ana.p[i]));
    pl.scatter(ks, ps, { colors: order.map(i => (plate.real[i] ? seriesColor(2) : seriesColor(1))), r: 4 });
    pl.legend(undefined, "tl");
  }

  function readout(): void {
    ro.textContent =
      `m = ${ana.p.length} features, ${ana.nReal} real\n` +
      `            hits  false  true\n` +
      `uncorrected ${String(ana.raw).padStart(4)}  ${String(ana.falseRaw).padStart(5)}  ${String(ana.trueRaw).padStart(4)}\n` +
      `Bonferroni  ${String(ana.bonf).padStart(4)}  ${String(ana.falseBonf).padStart(5)}  ${String(ana.trueBonf).padStart(4)}\n` +
      `BH (q 0.05) ${String(ana.bh).padStart(4)}  ${String(ana.falseBH).padStart(5)}  ${String(ana.trueBH).padStart(4)}`;
  }

  load(p0);
}

autoMount(NAME, mount);
