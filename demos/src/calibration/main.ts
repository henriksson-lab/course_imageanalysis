// Calibration of a classifier's probabilities (modern deep learning day, unit attention/).
// A simulated network (calibration/model.ts): the overconfidence slider sharpens its logits;
// temperature T divides them (button: fit T on validation data); domain shift moves the test
// data after T was fitted. Panels: histogram of the confidences, and the reliability diagram
// (accuracy per confidence bin against the diagonal) with ECE, accuracy and NLL.
// Auto-mounts on <div data-demo="calibration">. Options: data-preset="calibrated|over|fixed|shift|same".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import * as M from "./model";

const NAME = "calibration";
const NTEST = 4000, NVAL = 2000, BINS = 5;

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .cb-plot { flex: 1 1 330px; max-width: 400px; min-width: 240px; }
[data-demo="${NAME}"] .cb-cap { font-size: 16px; text-align: center; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 360px; gap: .4em; }
[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre; line-height: 1.3; }
[data-demo="${NAME}"] .cb-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface State { s: number; T: number; shift: number }
const PRESETS: { id: string; label: string; st: State; fit?: boolean }[] = [
  { id: "calibrated", label: "calibrated", st: { s: 1, T: 1, shift: 0 } },
  { id: "over", label: "overconfident network", st: { s: 3, T: 1, shift: 0 } },
  { id: "fixed", label: "temperature fixes it in-domain", st: { s: 3, T: 1, shift: 0 }, fit: true },
  { id: "shift", label: "shift: overconfident again", st: { s: 3, T: 1, shift: 0.5 }, fit: true },
  { id: "same", label: "accuracy unchanged, calibration changed", st: { s: 0.4, T: 1, shift: 0 } },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...p0.st };

  const mkPlot = (cap: string, opts: ConstructorParameters<typeof Plot>[1]) => {
    const d = htmlEl("div", "cb-plot", "", host);
    htmlEl("div", "cb-cap", cap, d);
    return new Plot(d, { width: 420, height: 360, ...opts });
  };
  const hist = mkPlot("confidences on the test data", { x: [0.5, 1], y: [0, 1], xlabel: "confidence", ylabel: "fraction" });
  const rel = mkPlot("reliability diagram", { x: [0.5, 1], y: [0.5, 1], xlabel: "confidence", ylabel: "accuracy", equal: false });
  rel.line([0.5, 1], [0.5, 1], { color: "var(--qb-muted)", dash: "6 4", width: 2 });
  const histBars = hist.bars([], [], { color: seriesColor(0), barWidth: 0.09 });
  const relBars = rel.bars([], [], { color: seriesColor(0), barWidth: 0.09, base: 0.5 });
  const gapMark = rel.scatter([], [], { color: seriesColor(1), r: 6 });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "cb-lab", "presets", panel);
  const pb = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => apply(p) })));
  const sS = slider(panel, "overconfidence", 0.25, 4, 0.05, st.s, v => { st.s = v; draw(); });
  const tS = slider(panel, "temperature T", 0.25, 5, 0.05, st.T, v => { st.T = v; draw(); });
  const shS = slider(panel, "domain shift", 0, 0.8, 0.05, st.shift, v => { st.shift = v; draw(); });
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "fit T on validation", () => fit());
  button(row, "T = 1", () => { st.T = 1; tS.set(1); draw(); });
  button(row, "reset", () => apply(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Bars: accuracy in each confidence bin; red dots: the mean confidence in the bin. On the diagonal = calibrated. T is fitted on validation data from the training domain (no shift).", panel);

  function fit(): void {
    const val = M.simulate(NVAL, st.s, 0, 2);
    st.T = Math.round(M.fitTemperature(val) * 100) / 100;
    tS.set(Math.min(5, st.T)); draw();
  }
  function apply(p: typeof PRESETS[number]): void {
    Object.assign(st, p.st); sS.set(st.s); tS.set(st.T); shS.set(st.shift);
    if (p.fit) fit(); else draw();
    pb.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
  }
  function draw(): void {
    const test = M.simulate(NTEST, st.s, st.shift, 1);
    const pr = M.predictions(test, st.T), R = M.reliability(pr, BINS);
    const mids = R.edges.slice(0, -1).map((e, b) => (e + R.edges[b + 1]) / 2);
    histBars.set(mids, R.count.map(c => c / pr.length));
    relBars.set(mids, R.acc.map(a => (Number.isFinite(a) ? a : 0.5)));
    gapMark.set(mids.filter((_, b) => R.count[b] > 0), R.conf.filter((_, b) => R.count[b] > 0));
    const meanConf = pr.reduce((s, p) => s + p.conf, 0) / pr.length;
    ro.textContent = `accuracy ${(100 * R.accuracy).toFixed(1)} %   mean confidence ${(100 * meanConf).toFixed(1)} %\n` +
      `ECE ${(100 * R.ece).toFixed(1)} %   NLL ${M.nll(test, st.T).toFixed(3)}\n` +
      `bins: ${R.count.map((c, b) => `${c ? (100 * R.acc[b]).toFixed(0) : "–"}`).join(" / ")} % correct`;
  }
  apply(p0);
}

autoMount(NAME, mount);
