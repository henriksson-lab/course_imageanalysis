// FLIM-FRET on the phasor plot. Left: the phasor plane with the donor-only point, the FRET
// trajectory (E from 0 to 1, a dot every 0.1) for the current interacting fraction, the line
// from the donor point to the FRET state, and a cloud of simulated pixels (calibrated). Right:
// E(r) with the Förster distance R0 = 5 nm and the current distance (drag on the curve), and
// the donor decay (log) with its components. Extras: background, autofluorescence (a third
// species, 0.4 ns) and acceptor emission leaking into the donor channel (a rising component:
// its phasor is P_DA · P_A, outside the semicircle). Estimates of E and α by (a) a
// bi-exponential Poisson fit with τ_D fixed (pooled pixels) and (b) the phasor line.
//
// <div data-demo="fretphasor" data-preset="none|all|thirty|same|auto|leak|few"></div>
import { autoMount, htmlEl, slider, presets, button } from "../lib/ui";
import { Plot } from "../lib/plot";
import { rng } from "../lib/random";
import * as D from "../lib/decay";
import * as Ph from "../lib/phasor";
import { lmFit } from "../lib/nlfit";
import { injectStyle } from "../canny/imgview";
import { phasorPlane, logMap, FLIM_CSS } from "../phasorplot/plane";

const NAME = "fretphasor";
const SEL = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${SEL} { gap: .6em; flex-wrap: wrap; align-items: flex-start; }
${SEL} .fp-a { flex: 1 1 460px; min-width: 0; max-width: 540px; }
${SEL} .fp-b { flex: 1 1 320px; min-width: 0; max-width: 420px; }
${SEL} .qb-panel { flex: 1 1 260px; gap: .25em; }
.reveal ${SEL} { flex-wrap: nowrap; }
.reveal ${SEL} .fp-a { flex: 0 0 470px; }
.reveal ${SEL} .fp-b { flex: 0 0 330px; }
.reveal ${SEL} .qb-panel { flex: 1 1 280px; font-size: 17px; gap: .1em; }
.reveal ${SEL} .qb-readout { font-size: 15px; }
`;

interface State { tauD: number; E: number; alpha: number; N: number; bg: number; auto: number; leak: number; compare: { E: number; alpha: number } | null }
const DEFAULT: State = { tauD: 2.5, E: 0.5, alpha: 0.5, N: 500, bg: 0, auto: 0, leak: 0, compare: null };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "none", label: "no FRET: on the donor point", s: { alpha: 0 } },
  { id: "all", label: "all donors interact, E = 0.5", s: { alpha: 1, E: 0.5 } },
  { id: "thirty", label: "30 % interact: closer to the donor than you think", s: { alpha: 0.3, E: 0.6 } },
  { id: "same", label: "E vs fraction: same mean lifetime", s: { alpha: 1, E: 0.3, compare: { E: 0.6, alpha: 5 / 7 } } },
  { id: "auto", label: "autofluorescence: a third corner", s: { alpha: 0.5, E: 0.5, auto: 20 } },
  { id: "leak", label: "acceptor leak: pulled outside the semicircle", s: { alpha: 1, E: 0.5, leak: 70 } },
  { id: "few", label: "too few photons", s: { alpha: 0.3, E: 0.6, N: 60 } },
];
const R0 = 5, OMEGA = D.omegaOf(80), AX: D.TimeAxis = { T: 12.5, nBins: 64 }, IRF = { sigma: 0.12, delay: 0.5 };
const TAU_AUTO = 0.4, TAU_A = 3.0, NPIX = 150;
const nM = logMap(20, 1e4);

/** Bin probabilities of acceptor emission excited through FRET: ∝ e^{-t/τA} − e^{-t/τDA}
 *  (sum 1), seen through the IRF. */
function leakBins(tauDA: number): Float64Array {
  const a = D.decayBins({ tau: [TAU_A], f: [1] }, IRF, AX), b = D.decayBins({ tau: [tauDA], f: [1] }, IRF, AX);
  // photon-normalised components: (τA·pA − τDA·pDA)/(τA − τDA) is the normalised difference
  const out = Float64Array.from(a, (v, j) => (TAU_A * v - tauDA * b[j]) / (TAU_A - tauDA));
  let s = 0;
  for (const v of out) s += v;
  return out.map(v => Math.max(0, v / s));
}

function mount(host: HTMLElement): void {
  injectStyle("flim-common", FLIM_CSS);
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = structuredClone({ ...DEFAULT, ...(p0?.s ?? {}) });

  const colA = htmlEl("div", "fp-a fl-col", "", host);
  const pp = phasorPlane(colA, OMEGA, { width: 500, height: 360 });
  const pl = pp.plot;
  const traj = pl.line([], [], { color: "var(--qb-c5)", width: 2.5 });
  const trajDots = pl.scatter([], [], { color: "var(--qb-c5)", r: 3.5 });
  const lineM = pl.line([], [], { color: "var(--qb-c2)", width: 2, dash: "6 4" });
  const cloud2 = pl.scatter([], [], { color: "var(--qb-c4)", r: 2.6, opacity: 0.6 });
  const cloud = pl.scatter([], [], { color: "var(--qb-c1)", r: 2.6, opacity: 0.6 });
  const pts = pl.scatter([], [], { r: 8 });
  const leakPt = pl.scatter([], [], { color: "var(--qb-c4)", r: 8 });
  const leakLab = pl.text(0, 0.6, "", { size: 17, color: "var(--qb-c4)" });
  const ptLab = [pl.text(0, 0.6, "", { size: 17 }), pl.text(0, 0.6, "", { size: 17 })];
  const ro = htmlEl("div", "qb-readout fl-ro", "", colA);

  const colB = htmlEl("div", "fp-b fl-col", "", host);
  const er = new Plot(colB, { width: 340, height: 200, x: [2, 10], y: [0, 1], xlabel: "distance r (nm)", ylabel: "E", margin: { l: 60, r: 10, t: 10, b: 56 } });
  er.fn(r => D.fretEfficiency(r, R0), { color: "var(--qb-c5)", width: 3 });
  er.vline(R0, { color: "var(--qb-muted)", width: 1.5, text: "R₀" });
  const erPt = er.scatter([], [], { color: "var(--qb-c2)", r: 9 });
  er.svg.style.cursor = "ew-resize"; er.svg.style.touchAction = "none";
  let drag = false;
  const setR = (ev: PointerEvent) => { const [x] = er.toData(ev); st.E = Math.max(0.01, Math.min(0.99, D.fretEfficiency(Math.max(2, Math.min(10, x)), R0))); sE.set(st.E); presetBtns.forEach(b => b.classList.remove("active")); draw(); };
  er.svg.addEventListener("pointerdown", ev => { drag = true; er.svg.setPointerCapture(ev.pointerId); setR(ev); });
  er.svg.addEventListener("pointermove", ev => { if (drag) setR(ev); });
  er.svg.addEventListener("pointerup", () => { drag = false; });
  const dec = new Plot(colB, { width: 340, height: 250, x: [0, 12.5], y: [1e-4, 0.2], ylog: true, xlabel: "time (ns)", ylabel: "photon fraction per bin", margin: { l: 80, r: 10, t: 10, b: 56 } });
  const dD = dec.line([], [], { color: "var(--qb-c3)", width: 2, dash: "6 4", label: "donor alone" });
  const dDA = dec.line([], [], { color: "var(--qb-c2)", width: 2, dash: "6 4", label: "with acceptor" });
  const dX = dec.line([], [], { color: "var(--qb-c4)", width: 2, dash: "3 3", label: "other" });
  const dSum = dec.line([], [], { color: "var(--qb-c1)", width: 3, label: "total" });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "fl-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, structuredClone(DEFAULT), structuredClone(p.s)); sync(); draw(); } })));
  const sTD = slider(panel, "τ_D donor alone (ns)", 1, 4, 0.1, st.tauD, v => { st.tauD = v; draw(); });
  const sE = slider(panel, "efficiency E", 0.01, 0.99, 0.01, st.E, v => { st.E = v; st.compare = null; draw(); });
  const sA = slider(panel, "interacting fraction α", 0, 1, 0.01, st.alpha, v => { st.alpha = v; st.compare = null; draw(); });
  const sN = slider(panel, "photons per pixel", 0, 100, 0.5, nM.u(st.N), u => { st.N = Math.round(nM.v(u)); draw(); }, u => String(Math.round(nM.v(u))));
  const sB = slider(panel, "background (%)", 0, 40, 1, st.bg, v => { st.bg = v; draw(); });
  const sAu = slider(panel, "autofluorescence (%)", 0, 50, 1, st.auto, v => { st.auto = v; draw(); });
  const sL = slider(panel, "acceptor leak (%)", 0, 90, 1, st.leak, v => { st.leak = v; draw(); });
  button(panel, "reset", () => { Object.assign(st, structuredClone(DEFAULT), structuredClone(p0?.s ?? {})); presetBtns.forEach(b => b.classList.remove("active")); sync(); draw(); });

  const K = Ph.calibrationFactor(Ph.phasor(D.decayBins({ tau: [4], f: [1] }, IRF, AX)), 4, OMEGA);

  /** Photon fractions of the parts and their bin probabilities. */
  function parts(E: number, alpha: number): { names: string[]; f: number[]; bins: Float64Array[]; tauDA: number } {
    const tauDA = D.lifetimeWithAcceptor(st.tauD, E);
    const fd = D.photonFractions([1 - alpha, alpha], [st.tauD, tauDA]);
    const other = (st.bg + st.auto + st.leak) / 100, w = Math.max(0, 1 - other);
    const flat = new Float64Array(AX.nBins).fill(1 / AX.nBins);
    return {
      names: ["donor alone", "with acceptor", "background", "autofluorescence", "acceptor leak"],
      f: [w * fd[0], w * fd[1], st.bg / 100, st.auto / 100, st.leak / 100],
      bins: [D.decayBins({ tau: [st.tauD], f: [1] }, IRF, AX), D.decayBins({ tau: [tauDA], f: [1] }, IRF, AX), flat, D.decayBins({ tau: [TAU_AUTO], f: [1] }, IRF, AX), leakBins(tauDA)],
      tauDA,
    };
  }
  function mixture(E: number, alpha: number): Float64Array {
    const q = parts(E, alpha), y = new Float64Array(AX.nBins);
    q.f.forEach((f, k) => { if (f > 0) for (let j = 0; j < AX.nBins; j++) y[j] += f * q.bins[k][j]; });
    return y;
  }
  function cloudOf(E: number, alpha: number, seed: number): { g: number[]; s: number[]; pooled: Float64Array } {
    const y = mixture(E, alpha), r = rng(seed), g: number[] = [], s: number[] = [], pooled = new Float64Array(AX.nBins);
    for (let p = 0; p < NPIX; p++) {
      const h = D.poissonHistogram(r, y.map(v => v * st.N));
      for (let j = 0; j < AX.nBins; j++) pooled[j] += h[j];
      const q = Ph.calibrate(Ph.phasor(h), K);
      g.push(q.g); s.push(q.s);
    }
    return { g, s, pooled };
  }

  function draw(): void {
    const PD = Ph.semicircle(OMEGA * st.tauD);
    // trajectory for the current α and extras: E from 0 to 1
    const other = (st.bg + st.auto + st.leak) / 100;
    const trajAt = (E: number) => {
      const q = parts(E, st.alpha);
      let g = 0, s = 0;
      q.f.forEach((f, k) => { if (f > 0) { const p = Ph.calibrate(Ph.phasor(q.bins[k]), K); g += f * p.g; s += f * p.s; } });
      return { g, s };
    };
    const tg: number[] = [], ts: number[] = [], dg: number[] = [], ds: number[] = [];
    for (let i = 0; i <= 50; i++) { const p = trajAt(Math.min(0.995, i / 50)); tg.push(p.g); ts.push(p.s); }
    for (let i = 0; i <= 10; i++) { const p = trajAt(Math.min(0.995, i / 10)); dg.push(p.g); ds.push(p.s); }
    traj.set(tg, ts); trajDots.set(dg, ds);
    const q = parts(st.E, st.alpha), PDA = Ph.semicircle(OMEGA * q.tauDA);
    lineM.set([PD.g, PDA.g], [PD.s, PDA.s]);
    pts.set([PD.g, PDA.g], [PD.s, PDA.s], ["var(--qb-c3)", "var(--qb-c2)"]);
    ptLab[0].set(PD.g - 0.02, PD.s + 0.04, "donor"); ptLab[1].set(PDA.g + 0.02, PDA.s + 0.03, "FRET state");
    if (st.leak > 0) { const L = Ph.calibrate(Ph.phasor(q.bins[4]), K); leakPt.set([L.g], [L.s]); leakLab.set(L.g + 0.02, L.s + 0.035, "acceptor emission"); }
    else { leakPt.set([], []); leakLab.set(0, 0.6, ""); }
    const c = cloudOf(st.E, st.alpha, 3);
    cloud.set(c.g, c.s);
    let cmp: { tauPhi: number; tauM: number } | null = null;
    if (st.compare) {
      const c2 = cloudOf(st.compare.E, st.compare.alpha, 4); cloud2.set(c2.g, c2.s);
      cmp = Ph.lifetimeFromPhasor({ g: c2.g.reduce((a, b) => a + b, 0) / NPIX, s: c2.s.reduce((a, b) => a + b, 0) / NPIX }, OMEGA);
    } else cloud2.set([], []);
    // E(r) and decays
    erPt.set([D.fretDistance(st.E, R0)], [st.E]);
    const xs = D.binCentres(AX), y = mixture(st.E, st.alpha);
    const comp = (k: number) => Float64Array.from(q.bins[k], v => v * q.f[k]);
    dD.set(q.f[0] > 0 ? xs : [], q.f[0] > 0 ? comp(0) : []);
    dDA.set(q.f[1] > 0 ? xs : [], q.f[1] > 0 ? comp(1) : []);
    const ox = Float64Array.from(xs, (_, j) => q.f[2] * q.bins[2][j] + q.f[3] * q.bins[3][j] + q.f[4] * q.bins[4][j]);
    dX.set(other > 0 ? xs : [], other > 0 ? ox : []);
    dSum.set(xs, y);
    let ymax = 0;
    for (const v of y) ymax = Math.max(ymax, v);
    dec.setDomain(undefined, [ymax * 3e-4, ymax * 2]);
    const items = [{ label: "total", color: "var(--qb-c1)", dash: "" }];
    if (q.f[0] > 0) items.push({ label: "donor alone", color: "var(--qb-c3)", dash: "6 4" });
    if (q.f[1] > 0) items.push({ label: "with acceptor", color: "var(--qb-c2)", dash: "6 4" });
    if (other > 0) items.push({ label: "other", color: "var(--qb-c4)", dash: "3 3" });
    dec.legend(items, "tr");
    // estimates: (a) Poisson fit, τ_D fixed, background free; (b) the phasor line from the donor point
    const tot = c.pooled.reduce((a, b) => a + b, 0);
    const model = (p: Float64Array) => {
      const a = D.decayBins({ tau: [st.tauD], f: [1] }, IRF, AX), b = D.decayBins({ tau: [p[2]], f: [1] }, IRF, AX);
      return Float64Array.from(a, (v, j) => p[0] * v + p[1] * b[j] + p[3]);
    };
    const r = lmFit(model, c.pooled, [tot / 2, tot / 2, st.tauD / 2, 1], { cost: "poisson", positive: [true, true, true, false], lower: [0, 0, 0.05, 0], upper: [Infinity, Infinity, st.tauD * 0.99, Infinity], maxIter: 200 });
    const tFit = r.p[2], aFit = D.amplitudeFractions([r.p[0], r.p[1]], [st.tauD, tFit])[1];
    const mg = c.g.reduce((a, b) => a + b, 0) / NPIX, ms = c.s.reduce((a, b) => a + b, 0) / NPIX;
    const dir = { g: mg - PD.g, s: ms - PD.s };
    const hits = Ph.lineSemicircle(PD, dir, OMEGA).filter(t => t < st.tauD * 0.98);
    const tLine = hits.length ? hits[hits.length - 1] : NaN;
    const fLine = Ph.fractionOnSegment({ g: mg, s: ms }, PD, Ph.semicircle(OMEGA * tLine));
    const aLine = D.amplitudeFractions([1 - fLine, fLine], [st.tauD, tLine])[1];
    const app = Ph.lifetimeFromPhasor({ g: mg, s: ms }, OMEGA);
    const outside = Math.hypot(mg - 0.5, ms) > 0.505;
    const f3 = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : "–"), fa = (v: number) => (v >= -0.005 && v <= 1.005 ? v.toFixed(2) : "–");
    ro.textContent = [
      `truth: τ_DA = ${q.tauDA.toFixed(2)} ns, E = ${st.E.toFixed(2)}, α = ${st.alpha.toFixed(2)}`,
      `photon fraction of the FRET state f = ${(q.f[1] / Math.max(1e-9, q.f[0] + q.f[1])).toFixed(2)}`,
      `cloud mean (${mg.toFixed(3)}, ${ms.toFixed(3)})${outside ? " — outside!" : ""}: τ_φ ${f3(app.tauPhi)}, τ_m ${f3(app.tauM)} ns`,
      `(a) fit, τ_D fixed: τ_DA ${f3(tFit)}, E ${f3(1 - tFit / st.tauD)}, α ${fa(aFit)}`,
      `(b) phasor line: ` + (outside ? "not possible outside the semicircle" : `τ_DA ${f3(tLine)}, E ${f3(1 - tLine / st.tauD)}, α ${fa(aLine)}`),
      st.compare ? `orange: E = ${st.compare.E}, α = ${st.compare.alpha.toFixed(2)}: τ_φ ${f3(cmp!.tauPhi)}, τ_m ${f3(cmp!.tauM)} ns` : `r = ${D.fretDistance(st.E, R0).toFixed(2)} nm (R₀ = ${R0} nm)`,
    ].join("\n");
  }

  function sync(): void { sTD.set(st.tauD); sE.set(st.E); sA.set(st.alpha); sN.set(nM.u(st.N)); sB.set(st.bg); sAu.set(st.auto); sL.set(st.leak); }

  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  draw();
}

autoMount(NAME, mount);
