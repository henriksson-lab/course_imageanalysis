// The phase transition of sparse recovery (Donoho–Tanner diagram). Axes: δ = M/N (how many
// measurements, as a fraction of the unknowns) and ρ = K/M (nonzeros per measurement). Each cell
// is the fraction of random trials (K-sparse ±1 signals of length N, Gaussian matrix or random
// rows of the real Fourier basis) recovered to 1 % by ℓ1 (FISTA with continuation) or OMP.
// Cells fill in progressively (animation-frame chunks; "stop" halts). The dashed curve is the
// theoretical ℓ1 boundary for Gaussian matrices (large N). Click a cell: one of its trials is
// shown on the right (truth and recovery).
// Auto-mounts on <div data-demo="phasetransition">. Optional: data-n="64|128", data-grid="16|20",
// data-trials="2", data-preset="l1|omp|fourier|noise".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { viridis } from "../lib/colour";
import { recoveryTrial, cellSeed, DONOHO_TANNER_L1, Sensing, Recovery } from "../lib/sparse";

const NAME = "phasetransition";
interface State { N: number; grid: number; trials: number; sensing: Sensing; solver: Recovery; noise: number }
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "l1", label: "ℓ1, Gaussian", s: { sensing: "gaussian", solver: "l1", noise: 0 } },
  { id: "omp", label: "OMP: lower boundary", s: { sensing: "gaussian", solver: "omp", noise: 0 } },
  { id: "fourier", label: "partial Fourier", s: { sensing: "fourier", solver: "l1", noise: 0 } },
  { id: "noise", label: "with noise: the edge blurs", s: { sensing: "gaussian", solver: "l1", noise: 0.05 } },
];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; }
.qb-demo[data-demo="${NAME}"] .pt-diag { flex: 0 1 520px; min-width: 280px; }
.qb-demo[data-demo="${NAME}"] .pt-diag svg { cursor: crosshair; }
.qb-demo[data-demo="${NAME}"] .pt-side { flex: 1 1 280px; min-width: 0; max-width: 400px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 240px; max-width: 300px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .pt-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; gap: .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .pt-diag { flex-basis: 500px; max-width: 520px; }
.reveal .qb-demo[data-demo="${NAME}"] .pt-side { flex-basis: 300px; max-width: 320px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex-basis: 260px; max-width: 290px; gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .pt-lab { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const init: State = {
    N: Number(host.dataset.n) === 128 ? 128 : 64, grid: Number(host.dataset.grid) === 20 ? 20 : Number(host.dataset.grid) === 10 ? 10 : 16,
    trials: Math.max(1, Math.min(10, Number(host.dataset.trials ?? 2))), sensing: "gaussian", solver: "l1", noise: 0, ...(p0?.s ?? {}),
  };
  const st: State = { ...init };

  const diag = htmlEl("div", "pt-diag", "", host);
  const P = new Plot(diag, { width: 520, height: 500, x: [0, 1], y: [0, 1], xlabel: "δ = M/N (measurements per unknown)", ylabel: "ρ = K/M (nonzeros per measurement)", font: 21, xticks: [0, 0.2, 0.4, 0.6, 0.8, 1], yticks: [0, 0.2, 0.4, 0.6, 0.8, 1] });
  const side = htmlEl("div", "pt-side", "", host);
  const pTrial = new Plot(htmlEl("div", "", "", side), { width: 400, height: 260, x: [0, 64], y: [-1.5, 1.5], xlabel: "index", font: 20, margin: { t: 8 } });
  const leg = htmlEl("div", "qb-note", "", side);
  const ro = htmlEl("div", "qb-readout", "", side);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "pt-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, { noise: 0 }, p.s); sync(); restart(); } })));
  htmlEl("div", "pt-lab", "measurements", panel);
  const sRow = htmlEl("div", "qb-presets", "", panel);
  const sBtns = (["gaussian", "fourier"] as Sensing[]).map(v => button(sRow, v === "gaussian" ? "Gaussian" : "partial Fourier", () => { st.sensing = v; changed(); sync(); restart(); }));
  htmlEl("div", "pt-lab", "solver", panel);
  const oRow = htmlEl("div", "qb-presets", "", panel);
  const oBtns = (["l1", "omp"] as Recovery[]).map(v => button(oRow, v === "l1" ? "ℓ1" : "OMP", () => { st.solver = v; changed(); sync(); restart(); }));
  htmlEl("div", "pt-lab", "signal length N", panel);
  const nRow = htmlEl("div", "qb-presets", "", panel);
  const nBtns = [64, 128].map(v => button(nRow, "N = " + v, () => { st.N = v; changed(); sync(); restart(); }));
  const gRow = htmlEl("div", "qb-presets", "", panel);
  const gBtns = [10, 16, 20].map(v => button(gRow, v + "×" + v, () => { st.grid = v; changed(); sync(); restart(); }));
  const tS = slider(panel, "trials", 1, 10, 1, st.trials, v => { st.trials = v; changed(); restart(); });
  const cRow = htmlEl("div", "qb-presets", "", panel);
  const stopBtn = button(cRow, "stop", () => { running = false; status(); });
  button(cRow, "run", () => { if (!running) { running = true; loop(); } });
  button(cRow, "reset", () => { Object.assign(st, init); sync(); restart(); presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0)); });
  htmlEl("div", "qb-note", "Each cell: the fraction of trials recovered to 1 % relative error (10 % with noise). Fixed seeds: the same diagram every time. Click a cell to see one trial.", panel);
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");

  function changed(): void { presetBtns.forEach(b => b.classList.remove("active")); }
  function sync(): void {
    sBtns.forEach((b, i) => b.classList.toggle("active", ["gaussian", "fourier"][i] === st.sensing));
    oBtns.forEach((b, i) => b.classList.toggle("active", ["l1", "omp"][i] === st.solver));
    nBtns.forEach((b, i) => b.classList.toggle("active", [64, 128][i] === st.N));
    gBtns.forEach((b, i) => b.classList.toggle("active", [10, 16, 20][i] === st.grid));
    tS.set(st.trials);
  }

  // ---- progressive computation ----
  let values = new Float64Array(0), order: number[] = [], next = 0, running = false, heat: ReturnType<Plot["heatmap"]> | null = null;
  const cellD = (i: number) => (i + 0.5) / st.grid, cellR = (j: number) => (j + 0.5) / st.grid;
  function restart(): void {
    const G = st.grid;
    values = new Float64Array(G * G).fill(NaN);
    // fill column by column (δ small → large)
    order = []; for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) order.push(j * G + i);
    order.sort((a, b) => (a % G) - (b % G) || Math.floor(a / G) - Math.floor(b / G));
    next = 0;
    P.clear();
    heat = P.heatmap(flipRows(values, G), G, G, { map: viridis, lo: 0, hi: 1, x0: 0, x1: 1, y0: 0, y1: 1, maxCells: 100000 });
    P.line(DONOHO_TANNER_L1.delta, DONOHO_TANNER_L1.rho, { color: "#fff", width: 5 });
    P.line(DONOHO_TANNER_L1.delta, DONOHO_TANNER_L1.rho, { color: seriesColor(1), width: 3, dash: "8 5", label: "theory: ℓ1, Gaussian, N → ∞" });
    P.legend(undefined, "tl");
    if (!running) { running = true; loop(); }
    status();
  }
  // heatmap rows run top to bottom; our row j = ρ index from the bottom
  function flipRows(v: Float64Array, G: number): Float64Array {
    const o = new Float64Array(G * G);
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) o[(G - 1 - j) * G + i] = v[j * G + i];
    return o;
  }
  function loop(): void {
    if (!running) return;
    const t0 = performance.now(), G = st.grid;
    while (next < order.length && performance.now() - t0 < 30) {
      const k = order[next++], i = k % G, j = Math.floor(k / G);
      const d = cellD(i), r = cellR(j), M = Math.max(1, Math.round(d * st.N)), K = Math.max(1, Math.round(r * M));
      let ok = 0;
      for (let t = 0; t < st.trials; t++) if (recoveryTrial(st.N, M, K, st.sensing, st.solver, cellSeed(1, d, r, t), st.noise > 0 ? 0.1 : 1e-2, st.noise).success) ok++;
      values[k] = ok / st.trials;
    }
    heat?.set(flipRows(values, G));
    status();
    if (next < order.length) setTimeout(loop, 0); else { running = false; status(); }
  }
  function status(): void {
    const done = next, tot = order.length;
    stopBtn.classList.toggle("active", running);
    const meas = st.sensing === "gaussian" ? "Gaussian matrix" : "random Fourier rows";
    ro.textContent = `${st.solver === "l1" ? "ℓ1" : "OMP"}, ${meas}, N = ${st.N}\n` +
      `${st.grid}×${st.grid} cells × ${st.trials} trials` + (st.noise > 0 ? `, noise ${Math.round(100 * st.noise)} %` : "") + `\n` +
      (done < tot ? `computing… ${done} of ${tot} cells` + (running ? "" : " (stopped)") : "done") + (sel ? `\n${sel}` : "");
  }

  // ---- click a cell ----
  let sel = "";
  P.svg.addEventListener("pointerdown", ev => { const [dx, dy] = P.toData(ev); showTrial(dx, dy); });
  function showTrial(dx: number, dy: number): void {
    if (!(dx > 0 && dx < 1 && dy > 0 && dy < 1)) return;
    const G = st.grid, i = Math.min(G - 1, Math.floor(dx * G)), j = Math.min(G - 1, Math.floor(dy * G));
    const d = cellD(i), r = cellR(j), M = Math.max(1, Math.round(d * st.N)), K = Math.max(1, Math.round(r * M));
    const tr = recoveryTrial(st.N, M, K, st.sensing, st.solver, cellSeed(1, d, r, 0), st.noise > 0 ? 0.1 : 1e-2, st.noise);
    pTrial.setDomain([0, st.N], [-1.6, 1.6]);
    pTrial.clear();
    const idx = Array.from({ length: st.N }, (_, q) => q);
    pTrial.scatter(idx.filter(q => tr.x[q] !== 0), idx.filter(q => tr.x[q] !== 0).map(q => tr.x[q]), { color: "var(--qb-fg)", r: 6 });
    pTrial.line(idx, Array.from(tr.xhat, v => Math.max(-1.6, Math.min(1.6, v))), { color: tr.success ? seriesColor(2) : seriesColor(1), width: 2 });
    leg.innerHTML = `● truth (K = ${K})  <span style="color:${tr.success ? seriesColor(2) : seriesColor(1)}">━ recovery from M = ${M}: ${tr.success ? "success" : "failure"}</span>`;
    sel = `cell δ = ${d.toFixed(2)}, ρ = ${r.toFixed(2)}: M = ${M}, K = ${K}\ntrial 1 rel. error ${tr.relErr.toExponential(1)}; cell rate ${Number.isFinite(values[j * G + i]) ? Math.round(100 * values[j * G + i]) + " %" : "–"}`;
    status();
  }

  sync();
  restart();
  showTrial(0.5, 0.3);
}

autoMount(NAME, mount);
