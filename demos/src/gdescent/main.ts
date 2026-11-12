// gdescent: gradient descent on a loss landscape over two parameters (a, b). The colour bands
// are contour bands of the loss (log scale); click to start a path there (up to three paths),
// then step. Learning rate η, mini-batch size (40 = the full batch: a smooth path; small batches:
// a wandering path) and momentum are sliders. A small plot shows the loss of each path.
//
// <div data-demo="gdescent" data-preset="right"></div>
import { autoMount, button, checkbox, htmlEl, presets as presetRow, slider } from "../lib/ui";
import { Plot, type Mark } from "../lib/plot";
import { viridis } from "../lib/colour";
import { LANDSCAPES, NEX, lossGrad, newPath, stepPath, findMinima, type Landscape, type PathState } from "./model";

const NAME = "gdescent";
const STYLE = `
[data-demo="${NAME}"] { gap: 1em; align-items: flex-start; }
[data-demo="${NAME}"] .gd-left { flex: 1 1 440px; min-width: 0; max-width: 540px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .gd-map { cursor: crosshair; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 320px; max-width: 520px; gap: .45em; }
[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .gd-left { max-width: 460px; }
.reveal [data-demo="${NAME}"] .gd-map { max-width: 400px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const COLORS = ["#d62728", "#ff7f0e", "#e377c2"];
interface Preset { id: string; label: string; land: string; eta: number; batch: number; mom: boolean; starts: [number, number][]; run?: number }
const PRESETS: Preset[] = [
  { id: "small", label: "too small", land: "bowl", eta: 0.02, batch: NEX, mom: false, starts: [[-1.5, 2]] },
  { id: "right", label: "just right", land: "bowl", eta: 0.3, batch: NEX, mom: false, starts: [[-1.5, 2]] },
  { id: "zigzag", label: "zig-zag across a narrow valley", land: "valley", eta: 0.6, batch: NEX, mom: false, starts: [[-1.5, 2]] },
  { id: "momentum", label: "momentum in the valley", land: "valley", eta: 0.03, batch: NEX, mom: true, starts: [[-1.5, 2]] },
  { id: "diverge", label: "diverges", land: "bowl", eta: 0.8, batch: NEX, mom: false, starts: [[-0.5, 1]] },
  { id: "sgd", label: "SGD noise (batch 4)", land: "bowl", eta: 0.1, batch: 4, mom: false, starts: [[-1.5, 2]] },
  { id: "wells", label: "two starts, two minima", land: "wells", eta: 0.05, batch: NEX, mom: false, starts: [[-0.3, 1.6], [0.3, 1.6]] },
  { id: "neuron", label: "one neuron's real loss", land: "neuron", eta: 2, batch: NEX, mom: false, starts: [[-1.5, 3]] },
];
const NG = 120;

function mount(host: HTMLElement): void {
  injectStyle();
  const start = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[1];
  let land: Landscape = LANDSCAPES[0], eta = start.eta, batch = start.batch, mom = start.mom;
  let paths: PathState[] = [], seedCount = 0, timer = 0;

  const leftCol = htmlEl("div", "gd-left", "", host);
  const mapDiv = htmlEl("div", "gd-map", "", leftCol);
  const plot = new Plot(mapDiv, { width: 540, height: 520, x: [0, 1], y: [0, 1], xlabel: "parameter a", ylabel: "parameter b", grid: false, margin: { l: 70, r: 10, t: 10, b: 60 } });
  let heat: Mark<any> | null = null;
  const minMark = plot.scatter([], [], { color: "#fff", r: 7, stroke: "#000" });
  let pathMarks: Mark<any>[] = [], dotMarks: Mark<any>[] = [];

  const panel = htmlEl("div", "qb-panel", "", host);
  const landRow = htmlEl("div", "qb-presets", "", panel);
  const landBtns = LANDSCAPES.map(L => button(landRow, L.label, () => { setLand(L); clearPaths(); draw(); }));
  const sEta = slider(panel, "learning rate η", -2.5, 0.5, 0.05, Math.log10(eta), v => { eta = 10 ** v; }, v => (10 ** v).toPrecision(2));
  const sBatch = slider(panel, "batch size", 1, NEX, 1, batch, v => { batch = v; }, v => (v >= NEX ? `${NEX} (all)` : String(v)));
  const cMom = checkbox(panel, "momentum (β = 0.9)", mom, v => { mom = v; });
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "1 step", () => { stop(); stepAll(1); draw(); });
  button(row, "run 50 steps", () => run(50));
  button(row, "clear paths", () => { stop(); clearPaths(); draw(); });
  htmlEl("div", "qb-note", "Click the map to start a path there (up to three).", panel);
  const lossDiv = htmlEl("div", "", "", panel);
  const lplot = new Plot(lossDiv, { width: 500, height: 170, x: [0, 50], y: [1e-3, 10], ylog: true, xlabel: "step", ylabel: "loss", margin: { l: 70, b: 50, t: 6 } });
  let lossMarks: Mark<any>[] = [];
  const readout = htmlEl("div", "qb-readout", "", panel);
  const presetBtns = presetRow(leftCol, PRESETS.map(p => ({ label: p.label, apply: () => applyPreset(p) })));

  let minima: [number, number, number][] = [], lmin = 0;
  function setLand(L: Landscape): void {
    land = L;
    const [a0, a1, b0, b1] = L.range;
    plot.setDomain([a0, a1], [b0, b1]);
    minima = findMinima(L);
    lmin = Math.min(...minima.map(m => m[2]));
    const V = new Float64Array(NG * NG);
    for (let j = 0; j < NG; j++) for (let i = 0; i < NG; i++) {
      const a = a0 + (a1 - a0) * (i + 0.5) / NG, b = b1 - (b1 - b0) * (j + 0.5) / NG;
      const t = Math.log10(lossGrad(L, a, b)[0] - lmin + 0.01);
      V[j * NG + i] = Math.floor(t * 4) / 4;   // bands: contour lines appear as colour steps
    }
    heat?.remove();
    heat = plot.heatmap(V, NG, NG, { map: (t: number) => viridis(0.15 + 0.85 * t), maxCells: 100 });
    plot.layer.insertBefore(heat.g, plot.layer.firstChild);
    minMark.set(minima.map(m => m[0]), minima.map(m => m[1]));
    landBtns.forEach((b, i) => b.classList.toggle("active", LANDSCAPES[i] === L));
  }

  function clearPaths(): void {
    paths = [];
    [...pathMarks, ...dotMarks, ...lossMarks].forEach(m => m.remove());
    pathMarks = []; dotMarks = []; lossMarks = [];
  }
  function addPath(a: number, b: number): void {
    if (paths.length >= 3) { clearPaths(); }
    const k = paths.length;
    paths.push(newPath(land, a, b, 100 + seedCount++));
    pathMarks.push(plot.line([], [], { color: COLORS[k], width: 2.5 }));
    dotMarks.push(plot.scatter([], [], { color: COLORS[k], r: 3.5 }));
    lossMarks.push(lplot.line([], [], { color: COLORS[k], width: 2.5 }));
  }
  function stepAll(n: number): void {
    for (const p of paths) for (let i = 0; i < n; i++) if (Number.isFinite(p.a)) stepPath(land, p, eta, batch, mom ? 0.9 : 0);
  }

  plot.svg.addEventListener("pointerdown", ev => {
    const [a, b] = plot.toData(ev), [a0, a1, b0, b1] = land.range;
    if (a < a0 || a > a1 || b < b0 || b > b1) return;
    stop(); addPath(a, b); draw();
  });

  function draw(): void {
    let maxStep = 50;
    paths.forEach((p, k) => {
      const ok = p.pts.filter(q => Number.isFinite(q[0]));
      pathMarks[k].set(ok.map(q => q[0]), ok.map(q => q[1]));
      dotMarks[k].set(ok.map(q => q[0]), ok.map(q => q[1]));
      maxStep = Math.max(maxStep, p.losses.length - 1);
    });
    lplot.setDomain([0, maxStep], [1e-3, 10]);
    paths.forEach((p, k) => lossMarks[k].set(p.losses.map((_, i) => i), p.losses.map(l => Math.min(10, Math.max(1e-3, l - lmin + 1e-3)))));
    readout.textContent = (paths.length ? paths.map((p, k) => {
      const l = p.losses[p.losses.length - 1];
      return `path ${k + 1}: step ${p.pts.length - 1}, (a, b) = ` + (Number.isFinite(p.a) ? `(${p.a.toFixed(3)}, ${p.b.toFixed(3)}), loss ${l.toFixed(4)}` : "diverged");
    }).join("\n") : "click the map to start") + `\nminimum: ` + minima.map(m => `(${m[0].toFixed(2)}, ${m[1].toFixed(2)}) loss ${m[2].toFixed(3)}`).join("; ") +
      `\nloss plot: loss − minimum`;
    cMom.checked = mom;
  }

  function run(n: number): void {
    stop();
    let left = n;
    const tick = () => {
      if (!host.isConnected) return;
      stepAll(1); left--; draw();
      if (left > 0) timer = window.setTimeout(tick, 40);
    };
    tick();
  }
  function stop(): void { clearTimeout(timer); }

  function applyPreset(p: Preset): void {
    stop();
    eta = p.eta; batch = p.batch; mom = p.mom; seedCount = 0;
    sEta.set(Math.log10(eta)); sBatch.set(batch);
    setLand(LANDSCAPES.find(L => L.id === p.land)!);
    clearPaths();
    p.starts.forEach(([a, b]) => addPath(a, b));
    draw();
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
  }

  applyPreset(start);
}

autoMount(NAME, mount);
