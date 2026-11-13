// Fitting a decay through the forward model. Left: one simulated TCSPC histogram (log axis)
// with the fitted model (decay convolved with the IRF, plus background), and below it the
// weighted residuals (y − μ)/√μ. Middle: the cost landscape over (τ, amplitude) for one
// lifetime, or over (τ₁, τ₂) for two (amplitudes profiled out), with the optimiser's path.
// Costs: least squares, χ² with 1/y weights (empty bins weight 1), Poisson likelihood
// (deviance). "Reconvolution" off: the model pretends the IRF is a spike at its centre.
//
// <div data-demo="decayfit" data-preset="agree|neyman|noirf|valley|resolved|wavy|background"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { Plot } from "../lib/plot";
import { inferno } from "../lib/colour";
import { rng } from "../lib/random";
import * as D from "../lib/decay";
import { LMFitter, Cost, costValue, linearAmplitudes, FitResult } from "../lib/nlfit";
import { injectStyle } from "../canny/imgview";
import { logMap, fmtCount, FLIM_CSS } from "../phasorplot/plane";

const NAME = "decayfit";
const SEL = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${SEL} { gap: .6em; flex-wrap: wrap; align-items: flex-start; }
${SEL} .df-a { flex: 1 1 500px; min-width: 0; max-width: 600px; }
${SEL} .df-b { flex: 1 1 300px; min-width: 0; max-width: 400px; }
${SEL} .qb-panel { flex: 1 1 260px; gap: .25em; }
.reveal ${SEL} { flex-wrap: nowrap; }
.reveal ${SEL} .df-a { flex: 0 0 500px; }
.reveal ${SEL} .df-b { flex: 0 0 330px; }
.reveal ${SEL} .qb-panel { flex: 1 1 280px; font-size: 17px; gap: .1em; }
.reveal ${SEL} .qb-readout { font-size: 16px; }
`;

type Model = "mono" | "bi" | "bifix";
interface State { tau1: number; tau2: number; frac: number; bg: number; N: number; irf: number; model: Model; cost: Cost; reconv: boolean; fitBg: boolean; seed: number }
const DEFAULT: State = { tau1: 2.5, tau2: 0.8, frac: 0, bg: 1, N: 10000, irf: 0.15, model: "mono", cost: "poisson", reconv: true, fitBg: true, seed: 1 };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "agree", label: "10 000 photons: costs agree", s: {} },
  { id: "neyman", label: "200 photons: 1/y goes wrong", s: { N: 200, cost: "neyman" } },
  { id: "noirf", label: "ignore the IRF: τ too long", s: { tau1: 0.8, irf: 0.35, N: 100000, reconv: false } },
  { id: "valley", label: "two τ, 1 000 photons: a valley", s: { tau1: 0.8, tau2: 3, frac: 0.6, N: 1000, model: "bi" } },
  { id: "resolved", label: "two τ, 100 000 photons", s: { tau1: 0.8, tau2: 3, frac: 0.6, N: 100000, model: "bi" } },
  { id: "wavy", label: "one τ fitted to two: wavy residual", s: { tau1: 0.8, tau2: 3, frac: 0.6, N: 100000, model: "mono" } },
  { id: "background", label: "background left out", s: { bg: 15, fitBg: false, N: 10000 } },
];
const COSTS: { id: Cost; label: string }[] = [{ id: "ls", label: "least squares" }, { id: "neyman", label: "χ² with 1/y" }, { id: "poisson", label: "Poisson" }];
const MODELS: { id: Model; label: string }[] = [{ id: "mono", label: "one τ" }, { id: "bi", label: "two τ" }, { id: "bifix", label: "two, τ₁ fixed" }];
const AX: D.TimeAxis = { T: 12.5, nBins: 128 }, DELAY = 1.0;
const tauM = logMap(0.2, 8), nM = logMap(30, 1e6);
const G = 40;

function mount(host: HTMLElement): void {
  injectStyle("flim-common", FLIM_CSS);
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(p0?.s ?? {}) };

  const colA = htmlEl("div", "df-a fl-col", "", host);
  const hist = new Plot(colA, { width: 520, height: 300, x: [0, AX.T], y: [0.5, 1e4], ylog: true, ylabel: "photons per bin", margin: { l: 78, r: 10, t: 10, b: 34 } });
  const dataM = hist.scatter([], [], { color: "var(--qb-c1)", r: 2.6 });
  const irfM = hist.line([], [], { color: "var(--qb-c4)", width: 2, dash: "6 4", label: "IRF" });
  const fitM = hist.line([], [], { color: "var(--qb-c2)", width: 3, label: "model" });
  const resP = new Plot(colA, { width: 520, height: 150, x: [0, AX.T], y: [-4, 4], yticks: [-3, 0, 3], xlabel: "time (ns)", ylabel: "residual", margin: { l: 78, r: 10, t: 6, b: 58 } });
  resP.hline(0, { color: "var(--qb-axis)", width: 1, dash: "" });
  const resM = resP.step([], [], { color: "var(--qb-c1)", width: 1.5 }, "mid");

  const colB = htmlEl("div", "df-b fl-col", "", host);
  const land = new Plot(colB, { width: 340, height: 320, x: [0.2, 8], y: [0.5, 1.5], xlog: true, xlabel: "τ (ns)", ylabel: "amplitude / photons", xticks: [0.2, 0.5, 1, 2, 4, 8], xfmt: v => String(v), margin: { l: 74, r: 10, t: 30, b: 58 }, title: "cost landscape" });
  let heat = land.heatmap(new Float64Array(G * G), G, G, { map: inferno, flipY: false, maxCells: 1 });
  const truthM = land.scatter([], [], { color: "var(--qb-c3)", r: 7 });
  const pathL = land.line([], [], { color: "#fff", width: 2 });
  const pathD = land.scatter([], [], { color: "var(--qb-c1)", r: 4, stroke: "#fff" });
  const ro = htmlEl("div", "qb-readout fl-ro", "", colB);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "fl-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); newData(); } })));
  const s1 = slider(panel, "true τ₁ (ns)", 0, 100, 0.5, tauM.u(st.tau1), u => { st.tau1 = +tauM.v(u).toFixed(2); newData(); }, u => tauM.v(u).toFixed(2));
  const s2 = slider(panel, "true τ₂ (ns)", 0, 100, 0.5, tauM.u(st.tau2), u => { st.tau2 = +tauM.v(u).toFixed(2); newData(); }, u => tauM.v(u).toFixed(2));
  const sF = slider(panel, "photon fraction of τ₂", 0, 1, 0.05, st.frac, v => { st.frac = v; newData(); });
  const sB = slider(panel, "background (%)", 0, 30, 1, st.bg, v => { st.bg = v; newData(); });
  const sN = slider(panel, "photons N", 0, 100, 0.5, nM.u(st.N), u => { st.N = Math.round(nM.v(u)); newData(); }, u => fmtCount(nM.v(u)));
  const sI = slider(panel, "IRF width σ (ns)", 0.05, 0.5, 0.01, st.irf, v => { st.irf = v; newData(); });
  htmlEl("div", "fl-lab", "model", panel);
  const mBtns = presets(panel, MODELS.map(m => ({ label: m.label, apply: () => { st.model = m.id; refit(); } })));
  htmlEl("div", "fl-lab", "cost", panel);
  const cBtns = presets(panel, COSTS.map(c => ({ label: c.label, apply: () => { st.cost = c.id; refit(); } })));
  const rowC = htmlEl("div", "qb-presets", "", panel);
  const cRe = checkbox(rowC, "reconvolution", st.reconv, v => { st.reconv = v; refit(); });
  const cBg = checkbox(rowC, "fit background", st.fitBg, v => { st.fitBg = v; refit(); });
  const rowB = htmlEl("div", "qb-presets", "", panel);
  button(rowB, "new noise", () => { st.seed++; newData(); });
  button(rowB, "start over", () => { startFit(); drawFit(); });
  button(rowB, "step", () => { fitter.step(); drawFit(); });
  button(rowB, "fit", () => { fitter.run(200); drawFit(); });
  button(rowB, "reset", () => { Object.assign(st, DEFAULT, p0?.s ?? {}); presetBtns.forEach(b => b.classList.remove("active")); sync(); newData(); });

  let y: Float64Array = new Float64Array(AX.nBins);
  let fitter!: LMFitter;
  const irfSpec = () => (st.reconv ? { sigma: st.irf, delay: DELAY } : { sigma: 0, delay: DELAY });
  const trueDecay = (): D.Decay => ({ tau: [st.tau1, st.tau2], f: [1 - st.frac, st.frac] });
  const nPar = () => (st.model === "mono" ? 3 : 5);
  /** Parameters: mono [N1, τ1, b]; bi [N1, τ1, N2, τ2, b] (photon counts per component). */
  function model(p: Float64Array): Float64Array {
    const irf = irfSpec();
    if (p.length === 3) return D.expectedCounts({ tau: [p[1]], f: [1] }, irf, AX, p[0], p[2]);
    const tot = p[0] + p[2];
    return D.expectedCounts({ tau: [p[1], p[3]], f: [p[0] / tot, p[2] / tot] }, irf, AX, tot, p[4]);
  }

  function newData(): void {
    const r = rng(1000 + st.seed);
    const mu = D.expectedCounts(trueDecay(), { sigma: st.irf, delay: DELAY }, AX, st.N * (1 - st.bg / 100), st.N * st.bg / 100 / AX.nBins);
    y = D.poissonHistogram(r, mu);
    refit();
  }

  function startFit(): void {
    let tot = 0;
    for (const v of y) tot += v;
    const irfMean = D.meanArrival(D.gaussianIRF(st.irf, DELAY, AX), AX);
    const t0 = Math.max(0.3, Math.min(6, D.meanArrival(y, AX) - irfMean));
    const b0 = st.fitBg ? Math.max(0.05, Math.min(...Array.from(y.slice(0, 8)))) : 0;
    const mono = st.model === "mono";
    const p0 = mono ? [tot, t0, b0] : [tot / 2, st.model === "bifix" ? st.tau1 : t0 / 2.5, tot / 2, t0 * 1.5, b0];
    const fixed = mono ? [false, false, !st.fitBg] : [false, st.model === "bifix", false, false, !st.fitBg];
    fitter = new LMFitter(model, y, p0, { cost: st.cost, fixed, positive: mono ? [true, true, false] : [true, true, true, true, false], lower: mono ? [0, 0.02, 0] : [0, 0.02, 0, 0.02, 0], upper: mono ? [Infinity, 50, Infinity] : [Infinity, 50, Infinity, 50, Infinity] });
  }

  let landVals: Float64Array = new Float64Array(G * G), landX: [number, number] = [0.2, 8], landY: [number, number] = [0.5, 1.5];
  function landscape(): void {
    const mono = st.model === "mono";
    const res = fitter.result();
    if (mono) {
      // (τ, N1) with b at its fitted value
      const b = res.p[2];
      let tot = 0;
      for (const v of y) tot += v;
      landX = [0.2, 8]; landY = [0.3 * Math.max(tot, 10), 1.7 * Math.max(tot, 10)];
      for (let i = 0; i < G; i++) {
        const tau = landX[0] * Math.pow(landX[1] / landX[0], (i + 0.5) / G);
        const m = D.decayBins({ tau: [tau], f: [1] }, irfSpec(), AX);
        for (let j = 0; j < G; j++) {
          const A = landY[0] + (landY[1] - landY[0]) * (j + 0.5) / G;
          landVals[j * G + i] = costValue(y, m.map(v => A * v + b), st.cost);
        }
      }
    } else {
      landX = [0.2, 8]; landY = [0.2, 8];
      const b = res.p[4], basis: Float64Array[] = [];
      for (let i = 0; i < G; i++) basis.push(D.decayBins({ tau: [landX[0] * Math.pow(landX[1] / landX[0], (i + 0.5) / G)], f: [1] }, irfSpec(), AX));
      const n = AX.nBins, B = new Float64Array(n * 2);
      for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
        for (let k = 0; k < n; k++) { B[2 * k] = basis[i][k]; B[2 * k + 1] = basis[j][k]; }
        const { mu } = linearAmplitudes(B, n, 2, y, st.cost, b, 4);
        landVals[j * G + i] = costValue(y, mu, st.cost);
      }
    }
    let mn = Infinity;
    for (const v of landVals) if (Number.isFinite(v)) mn = Math.min(mn, v);
    landVals = landVals.map(v => (Number.isFinite(v) ? Math.log10(1 + v - mn) : NaN));
    land.opts.ylog = !mono;
    land.opts.yticks = mono ? undefined : [0.2, 0.5, 1, 2, 4, 8];
    land.opts.yfmt = mono ? undefined : (v: number) => String(v);
    land.setLabels({ xlabel: mono ? "τ (ns)" : "τ₁ (ns)", ylabel: mono ? "photons in the decay" : "τ₂ (ns)" });
    land.setDomain(landX, landY);
    let hi = 0;
    for (const v of landVals) if (Number.isFinite(v)) hi = Math.max(hi, v);
    heat.remove();
    heat = land.heatmap(landVals, G, G, { map: (t: number) => inferno(1 - t), flipY: false, lo: 0, hi, maxCells: 1 });
    // keep the marks above the heat map
    for (const m of [truthM, pathL, pathD]) land.layer.appendChild(m.g);
    if (mono) truthM.set([st.frac < 0.5 ? st.tau1 : st.tau2], [st.N * (1 - st.bg / 100)]);
    else truthM.set([st.tau1], [st.tau2]);
  }

  function refit(): void {
    mBtns.forEach((b, i) => b.classList.toggle("active", MODELS[i].id === st.model));
    cBtns.forEach((b, i) => b.classList.toggle("active", COSTS[i].id === st.cost));
    startFit();
    fitter.run(200);
    landscape();
    drawFit();
  }

  function drawFit(): void {
    const res: FitResult = fitter.result(), p = res.p, mono = p.length === 3;
    const xs = D.binCentres(AX), mu = model(p);
    let ymax = 1;
    for (const v of y) ymax = Math.max(ymax, v);
    hist.setDomain([0, AX.T], [0.5, ymax * 2]);
    dataM.set(xs, y.map(v => (v > 0 ? v : NaN)));
    const irf = D.gaussianIRF(st.irf, DELAY, AX), im = Math.max(...irf);
    irfM.set(xs, irf.map(v => v / im * ymax * 0.8));
    fitM.set(xs, mu);
    hist.legend(undefined, "tr");
    resM.set(xs, Float64Array.from(mu, (m, j) => (m > 0 ? (y[j] - m) / Math.sqrt(m) : 0)));
    // path on the landscape
    const tr = res.trace;
    if (mono) { pathL.set(tr.map(s => s.p[1]), tr.map(s => s.p[0])); pathD.set(tr.map(s => s.p[1]), tr.map(s => s.p[0])); }
    else { pathL.set(tr.map(s => s.p[1]), tr.map(s => s.p[3])); pathD.set(tr.map(s => s.p[1]), tr.map(s => s.p[3])); }
    // readout
    let pear = 0, tot = 0;
    for (let j = 0; j < y.length; j++) { pear += (y[j] - mu[j]) ** 2 / Math.max(mu[j], 1e-9); tot += y[j]; }
    const free = (mono ? 2 : st.model === "bifix" ? 3 : 4) + (st.fitBg ? 1 : 0);
    const dof = AX.nBins - free;
    const fmt = (v: number, e: number) => `${v.toFixed(2)} ± ${Number.isFinite(e) ? e.toFixed(2) : "?"}`;
    const lines = [`counts ${fmtCount(tot)}; ${res.iterations} steps${res.converged ? " (converged)" : ""}`];
    if (mono) lines.push(`τ = ${fmt(p[1], res.se[1])} ns`);
    else {
      const a = D.amplitudeFractions([p[0], p[2]], [p[1], p[3]]), f2 = p[2] / (p[0] + p[2]);
      lines.push(`τ₁ = ${fmt(p[1], res.se[1])}, τ₂ = ${fmt(p[3], res.se[3])} ns`);
      lines.push(`photon fraction of τ₂ = ${f2.toFixed(2)}`);
      lines.push(`τ̄_a = ${D.tauAmplitude(a, [p[1], p[3]]).toFixed(2)}, τ̄_f = ${D.tauIntensity(a, [p[1], p[3]]).toFixed(2)} ns`);
    }
    lines.push(`background ${p[p.length - 1].toFixed(2)} per bin` + (st.fitBg ? "" : " (fixed)"));
    lines.push(`reduced χ² = ${(pear / dof).toFixed(2)}, deviance/dof = ${(costValue(y, mu, "poisson") / dof).toFixed(2)}`);
    const td = trueDecay(), ta = D.amplitudeFractions(td.f, td.tau);
    lines.push(st.frac > 0 && st.frac < 1 ? `truth: τ̄_a = ${D.tauAmplitude(ta, td.tau).toFixed(2)}, τ̄_f = ${D.tauIntensity(ta, td.tau).toFixed(2)} ns` : `truth: τ = ${(st.frac >= 1 ? st.tau2 : st.tau1).toFixed(2)} ns`);
    ro.textContent = lines.join("\n");
  }

  function sync(): void {
    s1.set(tauM.u(st.tau1)); s2.set(tauM.u(st.tau2)); sF.set(st.frac); sB.set(st.bg); sN.set(nM.u(st.N)); sI.set(st.irf);
    cRe.checked = st.reconv; cBg.checked = st.fitBg;
  }

  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  newData();
}

autoMount(NAME, mount);
