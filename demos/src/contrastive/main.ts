// contrastive: a toy of contrastive learning (unit contrastive/). 64 synthetic cells (three
// shapes, any brightness) are embedded on the unit circle by a tiny encoder trained with the
// InfoNCE loss on pairs of augmented views. The plot shows every cell's embedding (big dots,
// coloured by a label the training never sees) and the two views of the last step (small
// dots joined by a line). Sliders: temperature τ, augmentation strength; switches for the
// augmentations and for the negatives. Readout: 5-NN accuracy of shape and of dim/bright on
// the embeddings, and the length of the mean embedding (1 = collapsed to one point). See model.ts.
//
// <div data-demo="contrastive" data-preset="all"></div>
import { autoMount, button, checkbox, htmlEl, presets as presetRow, slider } from "../lib/ui";
import { Plot, type Mark } from "../lib/plot";
import { viridis, drawScalarImage, grey } from "../lib/colour";
import { baseStyle, injectStyle, logSlider, select } from "../deconvlab/widgets";
import { augment, makeCells, scores, Trainer, S, SHAPES, type Settings } from "./model";
import { rng } from "../lib/ui";

const NAME = "contrastive";
const STEPS = 600;
const STYLE = baseStyle(NAME) + `
[data-demo="${NAME}"] .ct-left { flex: 1 1 400px; max-width: 470px; }
[data-demo="${NAME}"] .ct-mid { flex: 1 1 280px; max-width: 330px; display: flex; flex-direction: column; gap: .35em; }
[data-demo="${NAME}"] .ct-thumbs { display: grid; grid-template-columns: repeat(6, 1fr); gap: 3px; }
[data-demo="${NAME}"] .ct-thumbs canvas { width: 100%; aspect-ratio: 1; image-rendering: pixelated; border-radius: 2px; }
[data-demo="${NAME}"] .ct-cap { font-size: 14px; color: var(--qb-muted); }
[data-demo="${NAME}"] .ct-left .axis { display: none; }
.reveal [data-demo="${NAME}"] .ct-left { flex: 0 1 400px; }
.reveal [data-demo="${NAME}"] .ct-mid { flex: 0 1 280px; }
`;
const SHAPE_COL = ["#1f77b4", "#d62728", "#2ca02c"];
type ColourBy = "shape" | "bright";
const BASE: Settings = { tau: 0.2, aug: { strength: 1, geom: true, bright: true, noise: true }, negatives: true };
interface Preset { id: string; label: string; s: Settings; by: ColourBy }
const PRESETS: Preset[] = [
  { id: "all", label: "all augmentations: shape survives", s: BASE, by: "shape" },
  { id: "nobright", label: "no brightness augmentation", s: { ...BASE, aug: { ...BASE.aug, bright: false } }, by: "bright" },
  { id: "none", label: "no augmentation: trivial", s: { ...BASE, aug: { ...BASE.aug, strength: 0 } }, by: "shape" },
  { id: "noneg", label: "no negatives: collapse", s: { ...BASE, negatives: false }, by: "shape" },
  { id: "tau1", label: "τ = 1: a weak push", s: { ...BASE, tau: 1 }, by: "shape" },
  { id: "tau005", label: "τ = 0.05", s: { ...BASE, tau: 0.05 }, by: "shape" },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const cells = makeCells(1);
  let s: Settings = { ...p0.s, aug: { ...p0.s.aug } }, by: ColourBy = p0.by;
  let tr = new Trainer(cells), timer = 0, target = 0;

  const left = htmlEl("div", "ct-left", "", host);
  const plot = new Plot(left, { width: 470, height: 470, x: [-1.35, 1.35], y: [-1.35, 1.35], grid: false, xticks: [], yticks: [], equal: true, margin: { l: 10, r: 10, t: 10, b: 10 }, xlabel: "", ylabel: "" });
  plot.fn(() => NaN);
  const circ = plot.line(Array.from({ length: 121 }, (_, i) => Math.cos(i * Math.PI / 60)), Array.from({ length: 121 }, (_, i) => Math.sin(i * Math.PI / 60)), { color: "#bbb", width: 1.5 });
  void circ;
  let pairMarks: Mark<any>[] = [];
  const viewDots = plot.scatter([], [], { r: 3, opacity: 0.55 });
  const dots = plot.scatter([], [], { r: 7, stroke: "#fff" });
  const legend = htmlEl("div", "dc-legend", "", left);

  const mid = htmlEl("div", "ct-mid", "", host);
  const lplot = new Plot(mid, { width: 330, height: 190, x: [0, STEPS], y: [0, 5], xlabel: "step", ylabel: "loss", margin: { l: 55, b: 45, t: 8, r: 8 } });
  const lline = lplot.line([], [], { color: "#1f77b4", width: 2 });
  const ro = htmlEl("div", "qb-readout", "", mid);
  htmlEl("div", "ct-cap", "two augmented views of three cells:", mid);
  const thumbs = htmlEl("div", "ct-thumbs", "", mid);
  const tcanv = Array.from({ length: 6 }, () => htmlEl("canvas", "", "", thumbs));

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "dc-lab", "presets", panel);
  const pbtns = presetRow(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const tauS = logSlider(panel, "temperature τ", 0.02, 3, s.tau, v => { s.tau = v; });
  const stS = slider(panel, "augmentation strength", 0, 1, 0.05, s.aug.strength, v => { s.aug.strength = v; drawThumbs(); });
  const cG = checkbox(panel, "rotate and shift", s.aug.geom, v => { s.aug.geom = v; drawThumbs(); });
  const cB = checkbox(panel, "brightness gain", s.aug.bright, v => { s.aug.bright = v; drawThumbs(); });
  const cN = checkbox(panel, "noise", s.aug.noise, v => { s.aug.noise = v; drawThumbs(); });
  const cNeg = checkbox(panel, "negatives (other cells pushed away)", s.negatives, v => { s.negatives = v; });
  const byS = select<ColourBy>(panel, "colour by", [["shape", "shape (never used)"], ["bright", "brightness (never used)"]], by, v => { by = v; draw(); });
  const row = htmlEl("div", "qb-presets", "", panel);
  const runB = button(row, `train ${STEPS} steps`, () => (timer ? stop() : run(tr.step + STEPS)));
  button(row, "1 step", () => { stop(); tr.train(s); draw(); });
  button(row, "new network", () => { stop(); tr = new Trainer(cells); draw(); });
  htmlEl("div", "qb-note", "Each step makes two new random views of all 64 cells. The labels only colour the dots and score the result; training never sees them.", panel);

  function drawThumbs(): void {
    const r = rng(77);
    [0, 1, 2].forEach((c, k) => {
      for (let v = 0; v < 2; v++) drawScalarImage(tcanv[2 * k + v], augment(cells[c * 7 + 1].img, s.aug, r), S, S, { map: grey, lo: 0, hi: 1.2 });
    });
  }
  function colour(i: number): string {
    if (by === "shape") return SHAPE_COL[cells[i].shape];
    const t = (cells[i].bright - 0.35) / 0.65, c = viridis(t);
    return `rgb(${c.map(v => Math.round(255 * v)).join(",")})`;
  }
  function draw(): void {
    const sc = scores(tr.net, cells);
    const cols = cells.map((_, i) => colour(i));
    dots.set(sc.z.map(q => q[0]), sc.z.map(q => q[1]), cols);
    pairMarks.forEach(m => m.remove()); pairMarks = [];
    const { a, b } = tr.views;
    if (a.length) {
      for (let i = 0; i < a.length; i += 2) pairMarks.push(plot.line([a[i][0], b[i][0]], [a[i][1], b[i][1]], { color: cols[i], width: 1, opacity: 0.5 }));
      viewDots.set([...a.map(q => q[0]), ...b.map(q => q[0])], [...a.map(q => q[1]), ...b.map(q => q[1])], [...cols, ...cols]);
    } else viewDots.set([], [], []);
    const L = tr.losses;
    lplot.setDomain([0, Math.max(STEPS, L.length)], [Math.min(0, ...L.slice(-1000)) - 0.1, Math.max(1, ...L.slice(-1000)) + 0.2]);
    lline.set(L.map((_, i) => i + 1), L);
    ro.textContent = `step ${tr.step}   loss ${L.length ? L[L.length - 1].toFixed(3) : "–"}\n` +
      `5-NN accuracy, shape: ${(100 * sc.shape).toFixed(0)} % (chance 33 %)\n` +
      `5-NN accuracy, dim/bright: ${(100 * sc.bright).toFixed(0)} % (chance 50 %)\n` +
      `length of the mean embedding: ${sc.collapse.toFixed(2)}`;
    legend.replaceChildren();
    const items = by === "shape" ? SHAPES.map((n, k) => [n, SHAPE_COL[k]]) : [["dim", "rgb(68,1,84)"], ["bright", "rgb(253,231,37)"]];
    for (const [n, c] of items) { const sp = htmlEl("span", "", "", legend); const i = htmlEl("i", "", "", sp); i.style.borderTop = `8px solid ${c}`; sp.append(n); }
  }
  function run(until: number): void {
    target = until; runB.textContent = "pause";
    const tick = () => {
      if (!host.isConnected) return;
      const t0 = performance.now();
      while (tr.step < target && performance.now() - t0 < 25) tr.train(s);
      draw();
      if (tr.step < target) timer = window.setTimeout(tick, 0); else stop();
    };
    tick();
  }
  function stop(): void { clearTimeout(timer); timer = 0; runB.textContent = `train ${STEPS} steps`; }
  function load(p: Preset): void {
    stop();
    s = { ...p.s, aug: { ...p.s.aug } }; by = p.by;
    tauS.setValue(s.tau); stS.set(s.aug.strength); cG.checked = s.aug.geom; cB.checked = s.aug.bright; cN.checked = s.aug.noise; cNeg.checked = s.negatives; byS.set(by);
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    tr = new Trainer(cells); drawThumbs(); draw();
  }
  load(p0);
  if (host.dataset.autorun) run(STEPS);
}

autoMount(NAME, mount);
