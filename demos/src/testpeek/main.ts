// Tuning on the test set: k candidate methods with the same true accuracy (or spread around
// it) are scored on a test set of n objects. "Run once" shows every test score, the best one
// (red) and the same method re-tested on fresh data (green). "Run 1000" repeats this and shows
// the histograms of the reported best score and of its fresh re-test, with the mean optimism.
// "Validation + test" chooses on a separate validation set and reports the test score.
// Auto-mounts on <div data-demo="testpeek">. Options: data-preset="one|fifty|bigger|valid|spread".
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { Plot, seriesColor, plotFigure } from "../lib/plot";
import { rng } from "../lib/random";
import * as M from "./model";

const NAME = "testpeek";
const STYLE = `
[data-demo="${NAME}"] { gap: .9em; }
[data-demo="${NAME}"] .qb-plot { flex: 1 1 340px; max-width: 450px; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 280px; gap: .4em; }
[data-demo="${NAME}"] .tp-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const pc = (v: number) => (100 * v).toFixed(1) + " %";

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = M.PRESETS.find(p => p.id === host.dataset.preset) ?? M.PRESETS[1];
  let st: M.Settings = { ...M.DEFAULTS, ...p0.settings };
  let seed = 1;

  const f1 = plotFigure(host, "one run: test score of each candidate");
  const f2 = plotFigure(host, "1000 runs: reported vs re-tested");
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "tp-lab", "presets", panel);
  const pbtn = presets(panel, M.PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const kS = slider(panel, "candidates k", 1, 100, 1, st.k, v => set({ k: v }));
  const aS = slider(panel, "true accuracy", 0.5, 0.95, 0.01, st.a, v => set({ a: v }));
  const sS = slider(panel, "spread of true acc.", 0, 0.08, 0.005, st.spread, v => set({ spread: v }));
  const nOf = (u: number) => Math.round(10 ** u), uOf = (n: number) => Math.log10(n);
  const nS = slider(panel, "test objects n", 1.3, 3.5, 0.05, uOf(st.n), u => set({ n: nOf(u) }), u => String(nOf(u)));
  const vC = checkbox(panel, "validation + test", st.validation, v => set({ validation: v }));
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "run once again", () => { seed++; drawOnce(); });
  button(row, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);

  let p1: Plot, p2: Plot;

  function set(p: Partial<M.Settings>): void { st = { ...st, ...p }; pbtn.forEach(b => b.classList.remove("active")); draw(); }
  function load(p: M.Preset): void {
    st = { ...M.DEFAULTS, ...p.settings }; seed = 1;
    kS.set(st.k); aS.set(st.a); sS.set(st.spread); nS.set(uOf(st.n)); vC.checked = st.validation;
    pbtn.forEach((b, i) => b.classList.toggle("active", M.PRESETS[i] === p));
    draw();
  }

  let once: M.Run, many: M.Many;
  function drawOnce(): void {
    once = M.runOnce(st, rng(seed));
    f1.querySelector("svg")?.remove();
    const xs = Array.from({ length: st.k }, (_, j) => j + 1);
    const lo = Math.max(0, st.a - 0.25), hi = Math.min(1, st.a + 0.2);
    p1 = new Plot(f1, { x: [0, st.k + 1], y: [lo, hi], xlabel: "candidate", ylabel: "accuracy", height: 380 });
    p1.scatter(xs, once.trueAcc, { color: "var(--qb-muted)", r: 3 });
    p1.hline(st.a, { color: "var(--qb-muted)", width: 1.5, text: "true " + st.a.toFixed(2) });
    if (once.val) p1.scatter(xs, once.val, { color: seriesColor(4), r: 4, fillOpacity: 0.6, label: "validation" });
    p1.scatter(xs, once.test, { color: seriesColor(0), r: 5, label: "test" });
    const b = once.best + 1;
    p1.scatter([b], [once.reported], { color: seriesColor(1), r: 9, label: "reported" });
    p1.scatter([b], [once.fresh], { color: seriesColor(2), r: 9, label: "fresh re-test" });
    p1.arrow(b, once.reported, b, once.fresh, { color: "var(--qb-fg)", width: 2 });
    p1.legend(undefined, "br");
    readout();
  }

  function drawMany(): void {
    many = M.runMany(st, 1000, 7);
    f2.querySelector("svg")?.remove();
    const lo = Math.max(0, st.a - 0.2), hi = Math.min(1, st.a + 0.2), nb = 40, w = (hi - lo) / nb;
    const hist = (x: Float64Array) => { const c = new Array(nb).fill(0); x.forEach(v => { const k = Math.floor((v - lo) / w); if (k >= 0 && k < nb) c[k]++; }); return c; };
    const hr = hist(many.reported), hf = hist(many.fresh), edges = Array.from({ length: nb + 1 }, (_, i) => lo + i * w);
    p2 = new Plot(f2, { x: [lo, hi], y: [0, Math.max(...hr, ...hf) * 1.15], xlabel: "accuracy", ylabel: "runs", height: 380 });
    p2.histogram(edges, hf, { color: seriesColor(2), label: "fresh re-test" });
    p2.histogram(edges, hr, { color: seriesColor(1), label: "reported (best)", fillOpacity: 0.4 });
    p2.vline(many.meanFresh, { color: seriesColor(2) });
    p2.vline(many.meanReported, { color: seriesColor(1) });
    p2.legend(undefined, "tl");
    readout();
  }

  function readout(): void {
    if (!once || !many) return;
    ro.textContent =
      `this run: reported ${pc(once.reported)}, fresh ${pc(once.fresh)}\n` +
      `1000 runs: mean reported ${pc(many.meanReported)}\n` +
      `           mean fresh    ${pc(many.meanFresh)}\n` +
      `optimism ${(100 * many.optimism).toFixed(1)} percentage points\n` +
      `SE of one score √(a(1−a)/n) = ${(100 * Math.sqrt(st.a * (1 - st.a) / st.n)).toFixed(1)} points`;
  }

  function draw(): void { drawOnce(); drawMany(); }
  load(p0);
}

autoMount(NAME, mount);
