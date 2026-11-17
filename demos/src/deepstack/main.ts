// deepstack: a batch of 64 random inputs goes up a stack of up to 50 dense layers (64 units
// each); a random loss gradient comes back down. Top plot: the size (root mean square) of the
// activations (blue) and of the gradients (red) after every layer, log scale. Below: histograms
// of the activations at five depths (values beyond ±3 pile up in the red end bins). Controls:
// the bend, the initial weight scale (gain × 1/sqrt(fan-in)), batch or layer normalisation,
// residual connections, the depth.
//
// <div data-demo="deepstack" data-preset="he"></div>
import { autoMount, button, checkbox, htmlEl, presets as presetRow, slider } from "../lib/ui";
import { Plot, type Mark } from "../lib/plot";
import { runStack, histogram, DEFAULT_STACK, type Act, type Norm, type StackConfig } from "./model";

const NAME = "deepstack";
const STYLE = `
[data-demo="${NAME}"] { gap: 1em; align-items: flex-start; }
[data-demo="${NAME}"] .ds-left { flex: 1 1 520px; min-width: 0; max-width: 620px; display: flex; flex-direction: column; gap: .2em; }
[data-demo="${NAME}"] .ds-hists { display: flex; gap: 4px; }
[data-demo="${NAME}"] .ds-hists > div { flex: 1 1 0; min-width: 0; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 420px; gap: .4em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .4em; }
.reveal [data-demo="${NAME}"] .ds-left { max-width: 600px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 13px; padding: .2em .4em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const SQ2 = Math.SQRT2;
interface Preset { id: string; label: string; cfg: Partial<StackConfig> }
const PRESETS: Preset[] = [
  { id: "small", label: "ReLU, gain 1: fades", cfg: { act: "relu", gain: 1 } },
  { id: "he", label: "ReLU, gain √2: steady", cfg: { act: "relu", gain: SQ2 } },
  { id: "large", label: "ReLU, gain 2: explodes", cfg: { act: "relu", gain: 2 } },
  { id: "tanhsmall", label: "tanh, gain 0.5", cfg: { act: "tanh", gain: 0.5 } },
  { id: "tanhbig", label: "tanh, gain 3: saturates", cfg: { act: "tanh", gain: 3 } },
  { id: "sigmoid", label: "sigmoid, gain 1", cfg: { act: "sigmoid", gain: 1 } },
  { id: "bn", label: "gain 2 + batch norm", cfg: { act: "relu", gain: 2, norm: "batch" } },
  { id: "ln", label: "gain 1 + layer norm", cfg: { act: "relu", gain: 1, norm: "layer" } },
  { id: "plain", label: "plain, gain √2, 50 layers", cfg: { act: "relu", gain: SQ2, depth: 50 } },
  { id: "res", label: "residual, gain √2, 50 layers", cfg: { act: "relu", gain: SQ2, residual: true, depth: 50 } },
  { id: "ressig", label: "sigmoid: plain vs residual", cfg: { act: "sigmoid", gain: 1, residual: true } },
];
const ACTS: Act[] = ["linear", "relu", "tanh", "sigmoid"];
const NORMS: [Norm, string][] = [["none", "no normalisation"], ["batch", "batch norm"], ["layer", "layer norm"]];
const BINS = 24;
const fmt = (v: number) => (v === 0 ? "0" : v < 1e-3 || v >= 1e4 ? v.toExponential(1) : v.toPrecision(3));

function mount(host: HTMLElement): void {
  injectStyle();
  const start = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[1];
  let cfg: StackConfig = { ...DEFAULT_STACK };

  const left = htmlEl("div", "ds-left", "", host);
  const plot = new Plot(left, { width: 600, height: 300, x: [0, 30], y: [1e-8, 1e8], ylog: true, yticks: [1e-8, 1e-4, 1, 1e4, 1e8], xlabel: "layer", ylabel: "size (rms)", margin: { l: 72, r: 10, t: 8, b: 50 } });
  plot.hline(1, { color: "#999", dash: "4 4" });
  const actLine = plot.line([], [], { color: "#1f77b4", width: 3, label: "activations" });
  const gradLine = plot.line([], [], { color: "#d62728", width: 3, label: "gradients" });
  plot.legend(undefined, "tl");
  const histRow = htmlEl("div", "ds-hists", "", left);
  const hplots: { p: Plot; bars: Mark<any>; edge: Mark<any> }[] = [];
  for (let k = 0; k < 5; k++) {
    const d = htmlEl("div", "", "", histRow);
    const p = new Plot(d, { width: 150, height: 130, x: [-3, 3], y: [0, 1], xticks: [-3, 0, 3], yticks: [], title: "", margin: { l: 8, r: 8, t: 26, b: 34 }, font: 15 });
    const bars = p.bars([], [], { color: "#1f77b4", barWidth: 6 / BINS * 0.9 });
    const edge = p.bars([], [], { color: "#d62728", barWidth: 6 / BINS * 0.9 });
    hplots.push({ p, bars, edge });
  }
  const presetBtns = presetRow(left, PRESETS.map(p => ({ label: p.label, apply: () => applyPreset(p) })));

  const panel = htmlEl("div", "qb-panel", "", host);
  const actRow = htmlEl("div", "qb-presets", "", panel);
  const actBtns = ACTS.map(a => button(actRow, a === "linear" ? "no bend" : a, () => { cfg.act = a; update(); }));
  const sGain = slider(panel, "gain", -0.6, 0.6, 0.01, Math.log10(cfg.gain), v => { cfg.gain = 10 ** v; update(); }, v => (10 ** v).toFixed(2));
  htmlEl("div", "qb-note", "weights: sd = gain / √fan-in; gain 1 keeps a linear signal, √2 = 1.41 a ReLU signal", panel);
  const normRow = htmlEl("div", "qb-presets", "", panel);
  const normBtns = NORMS.map(([n, l]) => button(normRow, l, () => { cfg.norm = n; update(); }));
  const cRes = checkbox(panel, "residual connections: a + f(a), blocks start small", false, v => { cfg.residual = v; update(); });
  const sDepth = slider(panel, "layers", 2, 50, 1, cfg.depth, v => { cfg.depth = v; update(); });
  const seedRow = htmlEl("div", "qb-presets", "", panel);
  button(seedRow, "new random weights", () => { cfg.seed++; update(); });
  const readout = htmlEl("div", "qb-readout", "", panel);

  function update(): void {
    const res = runStack(cfg);
    const L = res.layers, xs = L.map((_, i) => i + 1);
    plot.setDomain([0, cfg.depth], [1e-8, 1e8]);
    const clip = (v: number) => Math.max(1e-8, Math.min(1e8, v));
    actLine.set([0, ...xs], [res.inputRms, ...L.map(l => clip(l.rms))]);
    gradLine.set(xs, L.map(l => clip(l.gradRms)));
    const picks = [...[1, 2, 5, 10, 20].filter(d => d < cfg.depth), cfg.depth].slice(-5);
    while (picks.length < 5) picks.unshift(picks[0]);
    hplots.forEach((h, k) => {
      const l = picks[k], v = L[l - 1].values;
      const counts = histogram(v, -3, 3, BINS), total = v.length;
      const centres = counts.map((_, i) => -3 + 6 * (i + 0.5) / BINS);
      const mid = counts.map((c, i) => (i === 0 || i === BINS - 1 ? 0 : c / total));
      const ends = counts.map((c, i) => (i === 0 || i === BINS - 1 ? c / total : 0));
      const top = Math.max(0.05, ...mid, ...ends);
      h.p.setDomain([-3, 3], [0, top * 1.05]);
      h.bars.set(centres, mid); h.edge.set(centres, ends);
      h.p.setLabels({ title: `layer ${l}` });
    });
    const last = L[L.length - 1], first = L[0];
    readout.textContent =
      `input size ${fmt(res.inputRms)}\n` +
      `activations: layer 1 ${fmt(first.rms)}, layer ${cfg.depth} ${fmt(last.rms)}\n` +
      `gradients:   layer 1 ${fmt(first.gradRms)}, layer ${cfg.depth} ${fmt(last.gradRms)}\n` +
      `gradient ratio first/last layer ${fmt(first.gradRms / last.gradRms)}\n` +
      (cfg.act === "relu" ? `dead units in layer ${cfg.depth}: ${(100 * last.dead).toFixed(0)} %` :
        cfg.act === "linear" ? "no bend: the stack is one matrix" :
          `saturated (slope < 0.01) in layer ${cfg.depth}: ${(100 * last.saturated).toFixed(0)} %`);
    actBtns.forEach((b, i) => b.classList.toggle("active", ACTS[i] === cfg.act));
    normBtns.forEach((b, i) => b.classList.toggle("active", NORMS[i][0] === cfg.norm));
    cRes.checked = cfg.residual;
  }

  function applyPreset(p: Preset): void {
    cfg = { ...DEFAULT_STACK, ...p.cfg };
    sGain.set(Math.log10(cfg.gain)); sDepth.set(cfg.depth);
    update();
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
  }
  applyPreset(start);
}

autoMount(NAME, mount);
