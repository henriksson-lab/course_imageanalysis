// Regularisation path in 1D: a profile (step and ramp, close peaks, a line through the
// cell phantom) is blurred (Gaussian, circular) and noise is added. Restore it with
// data term + λ·penalty (energy, roughness, total variation, roughness + non-negativity),
// or by stopping an iteration (Landweber, optionally non-negative) after k steps.
// Panels: truth/data/estimate; the residual with the ±σn band (or the filter factors);
// the L-curve with the current λ and the corner; error and residual against λ (or k),
// with the discrepancy-principle λ, the L-curve corner and the best λ marked.
// Auto-mounts on <div data-demo="regpath">. Options: data-preset="<id>", data-presets="id,..."
import { htmlEl, presets as presetRow, button, checkbox, autoMount, slider } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import * as D from "../lib/deconv";
import { rng } from "../lib/ui";
import { truthOf, ProfileId } from "./profiles";
import { injectStyle, baseStyle, select, logSlider, Jobs, fmtSci, legendRow } from "../deconvlab/widgets";

const NAME = "regpath";
const n = 256;
type Pen = "energy" | "roughness" | "tv" | "nonneg";
type Tab = "lambda" | "iterate";
interface State { profile: ProfileId; sigma: number; noise: number; corr: boolean; seed: number; tab: Tab; pen: Pen; lambda: number; kIdx: number; lwNonneg: boolean; show: "residual" | "factors" }
const DEFAULT: State = { profile: "stepramp", sigma: 3, noise: 0.02, corr: false, seed: 1, tab: "lambda", pen: "roughness", lambda: 1e-3, kIdx: 20, lwNonneg: false, show: "residual" };
interface Preset { id: string; label: string; s: Partial<State>; pick?: "disc" | "corner" }
const PRESETS: Preset[] = [
  { id: "small", label: "λ too small: fits the noise", s: { pen: "roughness", lambda: 1e-6 } },
  { id: "large", label: "λ too large: blurred", s: { pen: "roughness", lambda: 100 } },
  { id: "disc", label: "discrepancy principle", s: { pen: "roughness" }, pick: "disc" },
  { id: "corner", label: "L-curve corner", s: { pen: "roughness" }, pick: "corner" },
  { id: "disagree", label: "the two rules disagree", s: { pen: "roughness", corr: true }, pick: "corner" },
  { id: "mean", label: "roughness keeps the mean, energy shrinks it", s: { pen: "energy", lambda: 0.3 } },
  { id: "tvstep", label: "TV keeps the step, staircases the ramp", s: { pen: "tv", lambda: 0.01 } },
  { id: "tvpeak", label: "TV shrinks the narrow peak", s: { profile: "peaks", pen: "tv", lambda: 0.02 } },
  { id: "nonneg", label: "non-negativity removes the dips", s: { profile: "peaks", pen: "nonneg", lambda: 1e-4 } },
  { id: "iter", label: "iterations as λ: semi-convergence", s: { tab: "iterate", kIdx: 20, show: "factors" } },
];
const LAMS = Array.from({ length: 37 }, (_, i) => 10 ** (-7 + i * 0.25));            // 1e-7 .. 1e2
const KS = Array.from(new Set(Array.from({ length: 41 }, (_, i) => Math.round(10 ** (i * 0.1))))); // 1 .. 10000

const STYLE = baseStyle(NAME) + `
[data-demo="${NAME}"] .rp-plots { flex: 1 1 640px; display: grid; grid-template-columns: 1fr 1fr; gap: .2em .8em; min-width: 0; }
[data-demo="${NAME}"] .rp-plots > div { min-width: 0; }
.reveal [data-demo="${NAME}"] .rp-plots { flex: 0 1 820px; }
`;

/** Log-axis ticks at 1, 2, 5 × 10^k (only 1 × 10^k if the range spans more than two decades). */
function ticks125(lo: number, hi: number): number[] {
  const out: number[] = [], wide = hi / lo > 300;
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of wide ? [1] : [1, 2, 5]) { const v = m * 10 ** e; if (v >= lo && v <= hi) out.push(v); }
  return out;
}

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const only = host.dataset.presets?.split(",").map(s => s.trim());
  const shown = only ? PRESETS.filter(p => only.includes(p.id)) : PRESETS;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[2];
  const st: State = { ...DEFAULT };

  // ---- layout ----
  const grid = htmlEl("div", "rp-plots", "", host);
  const box = () => htmlEl("div", "", "", grid);
  const b1 = box(), b2 = box(), b3 = box(), b4 = box();
  const W = 420, H = 250, F = 19;
  const prof = new Plot(b1, { width: W, height: H, x: [0, n], y: [-0.2, 1.2], xlabel: "position (samples)", ylabel: "value", font: F });
  const lTruth = prof.line([], [], { color: "var(--qb-muted, #777)", width: 3, dash: "7 5" });
  const lData = prof.line([], [], { color: seriesColor(0), width: 1.5 });
  const lEst = prof.line([], [], { color: seriesColor(1), width: 3 });
  legendRow(b1, [{ label: "truth", color: "var(--qb-muted, #777)", dash: true }, { label: "data", color: seriesColor(0) }, { label: "estimate", color: seriesColor(1) }]);
  const resP = new Plot(b2, { width: W, height: H, x: [0, n], y: [-0.1, 0.1], xlabel: "position (samples)", ylabel: "residual", font: F });
  const resBand = resP.band([0, n], [-0.02, -0.02], [0.02, 0.02], { color: seriesColor(2), fillOpacity: 0.2 });
  const lRes = resP.line([], [], { color: seriesColor(0), width: 1.5 });
  const facP = new Plot(b2, { width: W, height: H, x: [0, 0.5], y: [0, 1.05], xlabel: "frequency (cycles/sample)", ylabel: "kept fraction", font: F });
  const lH = facP.line([], [], { color: seriesColor(0), dash: "6 4" });
  const lPhi = facP.line([], [], { color: seriesColor(1) });
  const facLeg = legendRow(b2, [{ label: "|H| blur", color: seriesColor(0), dash: true }, { label: "filter factor φ", color: seriesColor(1) }]);
  const resLeg = legendRow(b2, [{ label: "data − blurred estimate", color: seriesColor(0) }, { label: "±σn", color: seriesColor(2) }]);
  const showSel = select<"residual" | "factors">(b2, "show", [["residual", "residual"], ["factors", "filter factors"]], st.show, v => { st.show = v; layoutB2(); });
  const lc = new Plot(b3, { width: W, height: H, x: [1e-2, 1], y: [1e-2, 10], xlog: true, ylog: true, xfmt: v => fmtSci(v), yfmt: v => fmtSci(v), xlabel: "residual norm ‖Ax − y‖", ylabel: "penalty", font: F });
  const lcLine = lc.line([], [], { color: seriesColor(4) });
  const lcCorner = lc.scatter([], [], { color: seriesColor(3), r: 8 });
  const lcDisc = lc.scatter([], [], { color: seriesColor(2), r: 8 });
  const lcNow = lc.scatter([], [], { color: seriesColor(1), r: 7 });
  legendRow(b3, [{ label: "current", color: seriesColor(1) }, { label: "corner", color: seriesColor(3) }, { label: "discrepancy", color: seriesColor(2) }]);
  const ev = new Plot(b4, { width: W, height: H, x: [1e-7, 1e2], y: [1e-3, 1], xlog: true, ylog: true, xlabel: "λ", ylabel: "RMS", font: F, xticks: [1e-6, 1e-4, 1e-2, 1] });
  const evErr = ev.line([], [], { color: seriesColor(3) });
  const evRes = ev.line([], [], { color: seriesColor(0) });
  const evSig = ev.hline(0.02, { color: seriesColor(2), dash: "6 4", width: 2 });
  const evNow = ev.vline(1e-3, { color: seriesColor(1), width: 2.5 });
  const evDisc = ev.vline(1e-3, { color: seriesColor(2), width: 2, dash: "3 4" });
  const evCorner = ev.vline(1e-3, { color: seriesColor(3), width: 2, dash: "3 4" });
  legendRow(b4, [{ label: "error vs truth", color: seriesColor(3) }, { label: "residual", color: seriesColor(0) }, { label: "σn", color: seriesColor(2), dash: true }]);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "dc-lab", "presets", panel);
  const pbtns = presetRow(panel, shown.map(p => ({ label: p.label, apply: () => load(p) })));
  const profS = select<ProfileId>(panel, "profile", [["stepramp", "step and ramp"], ["peaks", "close peaks, narrow peak"], ["cell", "line across the cell phantom"]], st.profile, v => { st.profile = v; remake(); });
  const sigS = slider(panel, "blur σ", 0.5, 6, 0.1, st.sigma, v => { st.sigma = v; remake(); });
  const noiseS = logSlider(panel, "noise σn", 1e-3, 0.1, st.noise, v => { st.noise = v; remake(); });
  const corrC = checkbox(panel, "correlated noise", st.corr, v => { st.corr = v; remake(); });
  const tabS = select<Tab>(panel, "restore by", [["lambda", "data + λ · penalty"], ["iterate", "stopping after k iterations"]], st.tab, v => { st.tab = v; syncTab(); recompute(); });
  const penS = select<Pen>(panel, "penalty", [["energy", "energy ‖x‖²"], ["roughness", "roughness ‖D²x‖²"], ["tv", "total variation Σ|Dx|"], ["nonneg", "roughness + x ≥ 0"]], st.pen, v => { st.pen = v; recompute(); });
  const lamS = logSlider(panel, "λ", 1e-7, 1e2, st.lambda, v => { st.lambda = v; current(); });
  const kS = slider(panel, "k", 0, KS.length - 1, 1, st.kIdx, v => { st.kIdx = v; current(); }, v => String(KS[v]));
  const lwC = checkbox(panel, "non-negative steps", st.lwNonneg, v => { st.lwNonneg = v; recompute(); });
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  const note = htmlEl("div", "qb-note", "", panel);

  const rowOf = (e: HTMLElement) => e.closest("label") as HTMLElement;
  function layoutB2(): void {
    const r = st.show === "residual";
    resP.svg.style.display = r ? "" : "none"; resLeg.style.display = r ? "" : "none";
    facP.svg.style.display = r ? "none" : ""; facLeg.style.display = r ? "none" : "";
    showSel.set(st.show);
  }
  function syncTab(): void {
    const it = st.tab === "iterate";
    penS.row.style.display = it ? "none" : ""; rowOf(lamS.input).style.display = it ? "none" : "";
    rowOf(kS.input).style.display = it ? "" : "none"; rowOf(lwC).style.display = it ? "" : "none";
    (ev.opts as { xticks?: number[] }).xticks = it ? [1, 10, 100, 1000, 1e4] : [1e-6, 1e-4, 1e-2, 1];
    ev.setLabels({ xlabel: it ? "iterations k" : "λ" });
    ev.setDomain(it ? [1, 1e4] : [1e-7, 1e2]);
  }
  function syncAll(): void {
    profS.set(st.profile); sigS.set(st.sigma); noiseS.setValue(st.noise); corrC.checked = st.corr; tabS.set(st.tab);
    penS.set(st.pen); lamS.setValue(st.lambda); kS.set(st.kIdx); lwC.checked = st.lwNonneg;
    syncTab(); layoutB2();
  }
  let pendingPick: Preset["pick"];
  function load(p: Preset): void {
    Object.assign(st, DEFAULT, p.s);
    pendingPick = p.pick;
    pbtns.forEach((b, i) => b.classList.toggle("active", shown[i] === p));
    syncAll(); remake();
  }

  // ---- model ----
  let truth: Float64Array = new Float64Array(n), y: Float64Array = new Float64Array(n), B = D.blur1dOf(D.gaussian1d(n, st.sigma));
  let R2 = D.penaltySymbol1d(n, "roughness"), E2 = D.penaltySymbol1d(n, "energy");
  function remake(): void {
    truth = truthOf(st.profile);
    B = D.blur1dOf(D.gaussian1d(n, st.sigma));
    const r = rng(st.seed);
    let e = Float64Array.from({ length: n }, () => r.norm());
    if (st.corr) {   // smooth with (1, 2, 3, 2, 1)/9 and rescale to unit SD: neighbouring samples correlated
      const s = new Float64Array(n), k = [1, 2, 3, 2, 1];
      for (let i = 0; i < n; i++) { let v = 0; for (let j = -2; j <= 2; j++) v += k[j + 2] * e[(i + j + n) % n]; s[i] = v / Math.sqrt(19); }
      e = s;
    }
    const b = D.blur1d(truth, B);
    y = b.map((v, i) => v + st.noise * e[i]);
    recompute();
  }
  const penOp = () => (st.pen === "energy" ? "energy" : "roughness") as D.Penalty1D;
  /** Estimate for a λ (penalty tab), warm-started where it helps. */
  function solveLam(lam: number, warm?: Float64Array): Float64Array {
    switch (st.pen) {
      case "energy": return D.tikhonov1d(y, B, lam, E2);
      case "roughness": return D.tikhonov1d(y, B, lam, R2);
      case "nonneg": return D.tikhonovNonneg1d(y, B, lam, R2, 400, warm);
      case "tv": return D.tv1d(y, B, lam, { beta: 1e-3, outer: 15, cg: 40, x0: warm });
    }
  }
  const penaltyOf = (x: Float64Array) => (st.pen === "tv" ? D.totalVariation1d(x) : D.norm2(D.penaltyOp1d(x, penOp())));
  // the path: per parameter value, estimate, residual RMS, error RMS, ρ, η
  interface Pt { p: number; x: Float64Array; res: number; err: number; rho: number; eta: number }
  let path: Pt[] = [];
  const jobs = new Jobs();
  const ptOf = (p: number, x: Float64Array): Pt => {
    const r = D.residual1d(x, y, B), e = x.map((v, i) => v - truth[i]);
    return { p, x, res: D.rms(r), err: D.rms(e), rho: D.norm2(r), eta: Math.max(1e-12, penaltyOf(x)) };
  };
  function recompute(): void {
    path = [];
    const it = st.tab === "iterate";
    const gen = function* () {
      if (!it) {
        let warm: Float64Array | undefined;
        for (const lam of LAMS) {
          // warm start from the next smaller λ for the iterative penalties
          const x = solveLam(lam, warm);
          warm = x;
          path.push(ptOf(lam, x)); yield;
        }
      } else if (!st.lwNonneg) {
        for (const k of KS) { path.push(ptOf(k, D.applyFactors(y, B, D.landweberFactors(B.H2, k)))); yield; }
      } else {
        let x: Float64Array = new Float64Array(n), done = 0;
        for (const k of KS) {
          x = D.landweber1d(y, B, k - done, { nonneg: true, x0: x });
          done = k;
          path.push(ptOf(k, x.slice())); yield;
        }
      }
    };
    jobs.run(gen(), finished => { drawPath(finished); if (finished) { applyPick(); current(); } }, 14, 200);
    current();
  }
  let lamDisc = NaN, lamCorner = NaN, iBest = 0;
  function drawPath(finished: boolean): void {
    if (!path.length) return;
    const it = st.tab === "iterate";
    evErr.set(path.map(q => q.p), path.map(q => q.err));
    evRes.set(path.map(q => q.p), path.map(q => q.res));
    evSig.set(st.noise);
    const all = path.flatMap(q => [q.err, q.res]).concat([st.noise]);
    ev.setDomain(undefined, [Math.max(1e-4, Math.min(...all) / 1.5), Math.min(10, Math.max(...all) * 1.5)]);
    lcLine.set(path.map(q => q.rho), path.map(q => q.eta));
    const rhos = path.map(q => q.rho), etas = path.map(q => q.eta);
    const xd: [number, number] = [Math.min(...rhos) / 1.3, Math.max(...rhos) * 1.3], yd: [number, number] = [Math.min(...etas) / 1.5, Math.max(...etas) * 1.5];
    Object.assign(lc.opts, { xticks: ticks125(...xd), yticks: ticks125(...yd) });
    lc.setDomain(xd, yd);
    if (!finished) return;
    // ordered by increasing regularisation: λ up, or k down
    const ord = it ? [...path].reverse() : path;
    const ic = D.lcurveCorner(ord.map(q => q.rho), ord.map(q => q.eta));
    lamCorner = ord[ic].p;
    lamDisc = D.crossing(ord.map(q => q.p), ord.map(q => q.res), st.noise);
    if (it) { const k = path.find(q => q.res <= st.noise); lamDisc = k ? k.p : NaN; }
    iBest = path.reduce((b, q, i) => (q.err < path[b].err ? i : b), 0);
    lcCorner.set([ord[ic].rho], [ord[ic].eta]);
    evCorner.set(lamCorner);
    const dPt = Number.isFinite(lamDisc) ? nearest(lamDisc) : null;
    lcDisc.set(dPt ? [dPt.rho] : [], dPt ? [dPt.eta] : []);
    evDisc.set(Number.isFinite(lamDisc) ? lamDisc : -1);
  }
  const nearest = (p: number) => path.reduce((b, q) => (Math.abs(Math.log(q.p / p)) < Math.abs(Math.log(b.p / p)) ? q : b), path[0]);
  function applyPick(): void {
    if (!pendingPick) return;
    const v = pendingPick === "disc" ? lamDisc : lamCorner;
    pendingPick = undefined;
    if (!Number.isFinite(v)) return;
    if (st.tab === "lambda") { st.lambda = v; lamS.setValue(v); }
    else { st.kIdx = KS.indexOf(v); kS.set(st.kIdx); }
  }

  let raf = 0;
  function current(): void {
    if (raf) return;
    raf = requestAnimationFrame(() => { raf = 0; currentNow(); });
    setTimeout(() => { if (raf) { cancelAnimationFrame(raf); raf = 0; currentNow(); } }, 60);
  }
  function currentNow(): void {
    const it = st.tab === "iterate";
    let x: Float64Array, phi: Float64Array | null = null;
    const p = it ? KS[st.kIdx] : st.lambda;
    if (!it) {
      x = solveLam(p);
      if (st.pen === "energy" || st.pen === "roughness") phi = D.tikhonovFactors(B.H2, p, st.pen === "energy" ? E2 : R2);
    } else {
      const q = path.find(q2 => q2.p === p);
      x = q ? q.x : st.lwNonneg ? D.landweber1d(y, B, p, { nonneg: true }) : D.applyFactors(y, B, D.landweberFactors(B.H2, p));
      if (!st.lwNonneg) phi = D.landweberFactors(B.H2, p);
    }
    const pt = ptOf(p, x);
    const xs = Array.from({ length: n }, (_, i) => i);
    lTruth.set(xs, truth); lData.set(xs, y); lEst.set(xs, x);
    const r = D.residual1d(x, y, B);
    lRes.set(xs, r);
    resBand.set([0, n], [-st.noise, -st.noise], [st.noise, st.noise]);
    const ra = Math.max(3 * st.noise, ...Array.from(r, Math.abs)) * 1.1;
    resP.setDomain(undefined, [-ra, ra]);
    const fs = Array.from({ length: n / 2 + 1 }, (_, k) => k / n);
    lH.set(fs, fs.map((_, k) => Math.sqrt(B.H2[k])));
    lPhi.set(phi ? fs : [], phi ? fs.map((_, k) => phi![k]) : []);
    lcNow.set([pt.rho], [pt.eta]);
    evNow.set(p);
    const mean = (a: ArrayLike<number>) => Array.from(a).reduce((s, v) => s + v, 0) / a.length;
    const lab = it ? "k" : "λ", f = (v: number) => (it ? String(v) : fmtSci(v));
    const lines = [
      `${lab} = ${f(p)}: error ${pt.err.toFixed(4)}, residual ${pt.res.toFixed(4)} (σn ${st.noise.toFixed(4)})`,
      `mean: truth ${mean(truth).toFixed(3)}, data ${mean(y).toFixed(3)}, estimate ${mean(x).toFixed(3)}`,
    ];
    if (path.length === (it ? KS.length : LAMS.length)) {
      lines.push(`discrepancy ${lab} ≈ ${Number.isFinite(lamDisc) ? f(lamDisc) : "–"}, corner ${lab} ≈ ${f(lamCorner)}`);
      lines.push(`best on the path: ${lab} ≈ ${f(path[iBest].p)} (error ${path[iBest].err.toFixed(4)})`);
    }
    ro.textContent = lines.join("\n");
    note.textContent = it
      ? "Landweber: x ← x + Aᵀ(y − Ax) from x = 0. The filter factors 1 − (1 − |H|²)^k fill in the strong frequencies first; k plays the role of 1/λ."
      : "Discrepancy principle: choose λ where the residual reaches the noise level σn. L-curve: the corner between 'penalty explodes' and 'residual grows'.";
  }

  syncAll();
  load(p0);
}

autoMount(NAME, mount);
