// The photon budget: a Monte Carlo of lifetime estimators. For the current settings, 400
// seeded histograms (64 bins over 12.5 ns, Gaussian IRF σ = 0.15 ns at 1 ns) are analysed by
// five methods: mean arrival time minus the IRF's mean, two gates, and fits by least squares,
// χ² with 1/y weights and Poisson likelihood. Left: every estimate as a dot (one row per
// method) with mean ± SD and the true τ. Right (button): the relative RMS error against N on
// log–log axes with the 1/√N line. Two lifetimes: the Poisson fit of a bi-exponential; the left
// plot shows the estimates (τ₁, τ₂) as a scatter.
//
// <div data-demo="photonbudget" data-preset="sqrt|few|background|two"></div>
import { autoMount, htmlEl, slider, presets, button } from "../lib/ui";
import { Plot } from "../lib/plot";
import { rng } from "../lib/random";
import * as D from "../lib/decay";
import { lmFit, Cost } from "../lib/nlfit";
import { injectStyle } from "../canny/imgview";
import { logMap, fmtShort, FLIM_CSS } from "../phasorplot/plane";

const NAME = "photonbudget";
const SEL = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${SEL} { gap: .6em; flex-wrap: wrap; align-items: flex-start; }
${SEL} .pb-a { flex: 1 1 440px; min-width: 0; max-width: 520px; }
${SEL} .pb-b { flex: 1 1 360px; min-width: 0; max-width: 460px; }
${SEL} .qb-panel { flex: 1 1 250px; gap: .3em; }
${SEL} .pb-vs { font-size: 14px; line-height: 1.25; }
.reveal ${SEL} { flex-wrap: nowrap; }
.reveal ${SEL} .pb-a { flex: 0 0 450px; }
.reveal ${SEL} .pb-b { flex: 0 0 400px; }
.reveal ${SEL} .qb-panel { flex: 1 1 260px; font-size: 17px; gap: .15em; }
`;

interface State { mode: "mono" | "bi"; tau1: number; tau2: number; frac: number; N: number; bg: number }
const DEFAULT: State = { mode: "mono", tau1: 2.5, tau2: 0.8, frac: 0.5, N: 1000, bg: 0 };
const PRESETS: { id: string; label: string; s: Partial<State>; vsN?: boolean }[] = [
  { id: "sqrt", label: "the 1/√N line", s: {}, vsN: true },
  { id: "few", label: "few photons: who is biased?", s: { N: 100 } },
  { id: "background", label: "background hurts the mean arrival time", s: { N: 10000, bg: 10 } },
  { id: "two", label: "two lifetimes need far more photons", s: { mode: "bi", tau1: 0.8, tau2: 3, frac: 0.6, N: 10000 }, vsN: true },
];
const METHODS = ["mean arrival", "two gates", "least squares", "χ² with 1/y", "Poisson"];
const MCOL = ["var(--qb-c4)", "var(--qb-c5)", "var(--qb-c3)", "var(--qb-c2)", "var(--qb-c1)"];
const AX: D.TimeAxis = { T: 12.5, nBins: 64 }, IRF = { sigma: 0.15, delay: 1 };
const M = 400, M_N = 150, N_LIST = [30, 100, 300, 1000, 3000, 10000, 30000, 100000];
const nM = logMap(30, 1e6), tauM = logMap(0.2, 8);
const IRF_MEAN = D.meanArrival(D.gaussianIRF(IRF.sigma, IRF.delay, AX), AX);

interface Est { m: number[][] }    // per method, the estimates (mono); bi: [τ1s, τ2s]

/** Analyse one histogram with every mono method; returns τ estimates (NaN if failed). */
function monoMethods(y: Float64Array, fitBg: boolean): number[] {
  let tot = 0;
  for (const v of y) tot += v;
  const ma = D.meanArrival(y, AX) - IRF_MEAN;
  const tg = D.twoGateTau(y, AX, 1.5, 2.5).tau;
  const t0 = Math.max(0.3, Math.min(6, Number.isFinite(ma) ? ma : 2));
  const model = (p: Float64Array) => D.expectedCounts({ tau: [p[1]], f: [1] }, IRF, AX, p[0], p[2]);
  const fit = (cost: Cost) => {
    const r = lmFit(model, y, [Math.max(tot, 1), t0, fitBg ? 0.1 : 0], { cost, positive: [true, true, false], lower: [0, 0.02, 0], upper: [Infinity, 50, Infinity], fixed: [false, false, !fitBg], maxIter: 60, tol: 1e-8 });
    return r.p[1];
  };
  return [ma, tg, fit("ls"), fit("neyman"), fit("poisson")];
}

function biFit(y: Float64Array, st: State): [number, number] {
  let tot = 0;
  for (const v of y) tot += v;
  const model = (p: Float64Array) => { const s = p[0] + p[2]; return D.expectedCounts({ tau: [p[1], p[3]], f: [p[0] / s, p[2] / s] }, IRF, AX, s, 0); };
  const r = lmFit(model, y, [tot * (1 - st.frac), st.tau1, tot * st.frac, st.tau2], { cost: "poisson", positive: [true, true, true, true], lower: [0, 0.02, 0, 0.02], upper: [Infinity, 50, Infinity, 50], maxIter: 100, tol: 1e-8 });
  const a = r.p[1], b = r.p[3];
  return a <= b ? [a, b] : [b, a];   // label switching: report the shorter as τ1
}

function simulate(st: State, N: number, k: number, seed: number): Float64Array {
  const r = rng(seed * 7919 + k);
  const dec: D.Decay = st.mode === "mono" ? { tau: [st.tau1], f: [1] } : { tau: [Math.min(st.tau1, st.tau2), Math.max(st.tau1, st.tau2)], f: st.tau1 <= st.tau2 ? [1 - st.frac, st.frac] : [st.frac, 1 - st.frac] };
  return D.poissonHistogram(r, D.expectedCounts(dec, IRF, AX, N * (1 - st.bg / 100), N * st.bg / 100 / AX.nBins));
}

const meanOf = (a: number[]) => { const v = a.filter(Number.isFinite); return v.reduce((s, x) => s + x, 0) / v.length; };
const sdOf = (a: number[]) => { const v = a.filter(Number.isFinite), m = meanOf(v); return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / Math.max(1, v.length - 1)); };
const rmsRel = (a: number[], t: number) => { const v = a.filter(Number.isFinite); return Math.sqrt(v.reduce((s, x) => s + (x - t) ** 2, 0) / v.length) / t; };

/** Yield to the browser and continue (a message, not a timer: timers are throttled in
 *  background tabs). */
const later = (() => {
  const ch = new MessageChannel(), queue: (() => void)[] = [];
  ch.port1.onmessage = () => { const f = queue.shift(); if (f) f(); };
  return (f: () => void) => { queue.push(f); ch.port2.postMessage(0); };
})();

function mount(host: HTMLElement): void {
  injectStyle("flim-common", FLIM_CSS);
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(p0?.s ?? {}) };

  const colA = htmlEl("div", "pb-a fl-col", "", host);
  const strip = new Plot(colA, { width: 470, height: 340, x: [0, 5], y: [-0.6, 4.6], yticks: [0, 1, 2, 3, 4], yfmt: v => METHODS[4 - Math.round(v)] ?? "", xlabel: "estimated τ (ns)", margin: { l: 150, r: 12, t: 10, b: 60 } });
  const dots = METHODS.map((_, i) => strip.scatter([], [], { color: MCOL[i], r: 2.2, opacity: 0.55 }));
  const bars = strip.mark<[number[], number[]]>((g, mu, sd) => {
    mu.forEach((m, i) => {
      if (!Number.isFinite(m)) return;
      const y = strip.y(4 - i), x0 = strip.x(m - sd[i]), x1 = strip.x(m + sd[i]);
      const l = document.createElementNS("http://www.w3.org/2000/svg", "path");
      l.setAttribute("d", `M${x0},${y - 14}V${y + 14}M${x0},${y}H${x1}M${x1},${y - 14}V${y + 14}M${strip.x(m)},${y - 20}V${y + 20}`);
      l.style.cssText = "stroke:var(--qb-fg);stroke-width:3;fill:none";
      g.appendChild(l);
    });
  }, [[], []]);
  const truthL = strip.vline(2.5, { color: "var(--qb-fg)", width: 2, text: "true τ" });
  const scat = new Plot(colA, { width: 470, height: 340, x: [0.2, 8], y: [0.2, 8], xlog: true, ylog: true, xlabel: "estimated τ₁ (ns)", ylabel: "estimated τ₂ (ns)", xticks: [0.2, 0.5, 1, 2, 4, 8], yticks: [0.2, 0.5, 1, 2, 4, 8], xfmt: v => String(v), yfmt: v => String(v), margin: { l: 78, r: 12, t: 10, b: 60 } });
  const scatDots = scat.scatter([], [], { color: "var(--qb-c1)", r: 2.5, opacity: 0.6 });
  const scatTrue = scat.scatter([], [], { color: "var(--qb-c2)", r: 8 });

  const colB = htmlEl("div", "pb-b fl-col", "", host);
  const vs = new Plot(colB, { width: 420, height: 340, x: [30, 1e5], y: [0.003, 3], xlog: true, ylog: true, xlabel: "photons N", ylabel: "relative RMS error", margin: { l: 80, r: 12, t: 10, b: 60 } });
  const sqrtL = vs.fn(n => 1 / Math.sqrt(n), { color: "var(--qb-muted)", width: 2, dash: "6 4", label: "1/√N" });
  const vsLines = METHODS.map((m, i) => vs.line([], [], { color: MCOL[i], width: 2.5, label: m }));
  const vsBi = [vs.line([], [], { color: "var(--qb-c3)", width: 2.5, dash: "3 3", label: "two τ: τ₁" }), vs.line([], [], { color: "var(--qb-c2)", width: 2.5, dash: "3 3", label: "two τ: τ₂" })];
  const vsNow = vs.vline(1000, { color: "var(--qb-fg)", width: 1.5 });
  void sqrtL;
  const vsro = htmlEl("div", "fl-lab pb-vs", "", colB);
  const ro = htmlEl("div", "qb-readout fl-ro", "", colA);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "fl-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); run(); if (p.vsN) runVsN(); else clearVsN(); } })));
  htmlEl("div", "fl-lab", "decay", panel);
  const modeBtns = presets(panel, [{ label: "one lifetime", apply: () => { st.mode = "mono"; clearVsN(); run(); } }, { label: "two lifetimes", apply: () => { st.mode = "bi"; clearVsN(); run(); } }]);
  const s1 = slider(panel, "τ₁ (ns)", 0, 100, 0.5, tauM.u(st.tau1), u => { st.tau1 = +tauM.v(u).toFixed(2); run(); }, u => tauM.v(u).toFixed(2));
  const s2 = slider(panel, "τ₂ (ns, two)", 0, 100, 0.5, tauM.u(st.tau2), u => { st.tau2 = +tauM.v(u).toFixed(2); run(); }, u => tauM.v(u).toFixed(2));
  const sF = slider(panel, "photon fraction τ₂", 0.05, 0.95, 0.05, st.frac, v => { st.frac = v; run(); });
  const sN = slider(panel, "photons N", 0, 100, 0.5, nM.u(st.N), u => { st.N = Math.round(nM.v(u)); run(); }, u => fmtShort(nM.v(u)));
  const sB = slider(panel, "background (%)", 0, 30, 1, st.bg, v => { st.bg = v; run(); });
  const rowB = htmlEl("div", "qb-presets", "", panel);
  button(rowB, "error vs N", () => runVsN());
  button(rowB, "new seeds", () => { seed++; run(); });
  button(rowB, "reset", () => { Object.assign(st, DEFAULT, p0?.s ?? {}); seed = 1; presetBtns.forEach(b => b.classList.remove("active")); sync(); clearVsN(); run(); });
  const status = htmlEl("div", "fl-lab", "", panel);

  let seed = 1;
  const jobs: Record<string, number> = { run: 0, vs: 0 }, prog: Record<string, string> = { run: "", vs: "" };
  /** Run `count` units of work in slices (keeps the page responsive), then `done`. A new job
   *  of the same kind cancels the old one. */
  function sliced(kind: "run" | "vs", count: number, work: (i: number) => void, done: () => void, label: string): void {
    const my = ++jobs[kind];
    let i = 0;
    const tick = () => {
      if (my !== jobs[kind] || !host.isConnected) return;
      const t0 = performance.now();
      while (i < count && performance.now() - t0 < 20) work(i++);
      prog[kind] = i < count ? `${label}: ${Math.round(100 * i / count)} %` : "";
      status.textContent = [prog.run, prog.vs].filter(x => x).join(" · ");
      if (i < count) later(tick); else done();
    };
    tick();
  }

  function run(): void {
    modeBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.mode === "mono")));
    strip.svg.style.display = st.mode === "mono" ? "" : "none";
    scat.svg.style.display = st.mode === "bi" ? "" : "none";
    vsNow.set(st.N);
    if (st.mode === "mono") {
      const est: Est = { m: METHODS.map(() => []) };
      sliced("run", M, k => { const e = monoMethods(simulate(st, st.N, k, seed), st.bg > 0); e.forEach((v, i) => est.m[i].push(v)); }, () => drawMono(est), "simulating");
    } else {
      const t1: number[] = [], t2: number[] = [];
      sliced("run", M, k => { const [a, b] = biFit(simulate(st, st.N, k, seed), st); t1.push(a); t2.push(b); }, () => drawBi(t1, t2), "fitting");
    }
  }

  function drawMono(est: Est): void {
    const t = st.tau1;
    const mu = est.m.map(meanOf), sd = est.m.map(sdOf);
    const spread = Math.max(0.15 * t, 4.5 * t / Math.sqrt(st.N));
    let lo = t - spread, hi = t + spread;
    for (let i = 0; i < 5; i++) if (Number.isFinite(mu[i])) { lo = Math.min(lo, mu[i] - 1.3 * sd[i]); hi = Math.max(hi, mu[i] + 1.3 * sd[i]); }
    strip.setDomain([Math.max(0, lo), hi]);
    const r = rng(5);
    est.m.forEach((v, i) => dots[i].set(v, v.map(() => 4 - i + 0.55 * (r.unif() - 0.5))));
    bars.set(mu, sd);
    truthL.set(t);
    const lines = [`${M} × ${st.N} photons, τ = ${t.toFixed(2)} ns`, "method        mean ± SD   bias"];
    METHODS.forEach((m, i) => {
      const fails = est.m[i].filter(v => !Number.isFinite(v)).length;
      lines.push(`${m.padEnd(13)} ${mu[i].toFixed(2)} ± ${sd[i].toFixed(2)}  ${(100 * (mu[i] - t) / t).toFixed(1).padStart(5)} %` + (fails ? ` (${fails} failed)` : ""));
    });
    lines.push(`ideal (τ/√N): ± ${(t / Math.sqrt(st.N)).toFixed(3)}`);
    ro.textContent = lines.join("\n");
  }

  function drawBi(t1: number[], t2: number[]): void {
    scatDots.set(t1, t2);
    const a = Math.min(st.tau1, st.tau2), b = Math.max(st.tau1, st.tau2);
    scatTrue.set([a], [b]);
    const m1 = meanOf(t1), m2 = meanOf(t2), s1v = sdOf(t1), s2v = sdOf(t2);
    let c = 0;
    for (let i = 0; i < t1.length; i++) c += (t1[i] - m1) * (t2[i] - m2);
    c /= (t1.length - 1) * s1v * s2v;
    ro.textContent = [`${M} fits of two τ, ${st.N} photons`,
      `τ₁ = ${m1.toFixed(2)} ± ${s1v.toFixed(2)} (true ${a.toFixed(2)})`,
      `τ₂ = ${m2.toFixed(2)} ± ${s2v.toFixed(2)} (true ${b.toFixed(2)})`,
      `correlation of the estimates ${c.toFixed(2)}`,
      `one lifetime, same N: ± ${(100 / Math.sqrt(st.N)).toFixed(1)} % at best`].join("\n");
  }

  function clearVsN(): void { jobs.vs++; prog.vs = ""; vsro.textContent = ""; vsLines.forEach(l => l.set([], [])); vsBi.forEach(l => l.set([], [])); vs.legend([]); }

  function runVsN(): void {
    clearVsN();
    const res: number[][] = METHODS.map(() => []), bi: number[][] = [[], []];
    const bucket: number[][] = METHODS.map(() => []), b1: number[] = [], b2: number[] = [];
    const total = N_LIST.length * M_N;
    const stMono: State = { ...st, mode: "mono", bg: st.mode === "mono" ? st.bg : 0, tau1: st.mode === "mono" ? st.tau1 : Math.max(st.tau1, st.tau2) };
    const keepRun = st.mode;
    sliced("vs", total, i => {
      const ni = Math.floor(i / M_N), k = i % M_N, N = N_LIST[ni];
      if (keepRun === "mono") { const e = monoMethods(simulate(stMono, N, k, seed + 100), stMono.bg > 0); e.forEach((v, j) => bucket[j].push(v)); }
      else { const [a, b] = biFit(simulate(st, N, k, seed + 100), st); b1.push(a); b2.push(b); }
      if (k === M_N - 1) {
        if (keepRun === "mono") { bucket.forEach((v, j) => { res[j].push(rmsRel(v, stMono.tau1)); v.length = 0; }); METHODS.forEach((_, j) => vsLines[j].set(N_LIST.slice(0, ni + 1), res[j])); }
        else {
          bi[0].push(rmsRel(b1, Math.min(st.tau1, st.tau2))); bi[1].push(rmsRel(b2, Math.max(st.tau1, st.tau2))); b1.length = 0; b2.length = 0;
          vsBi[0].set(N_LIST.slice(0, ni + 1), bi[0]); vsBi[1].set(N_LIST.slice(0, ni + 1), bi[1]);
        }
      }
    }, () => {
      const items = [{ label: "1/√N", color: "var(--qb-muted)", dash: "6 4" }];
      if (keepRun === "mono") METHODS.forEach((m, j) => items.push({ label: m, color: MCOL[j], dash: "" }));
      else items.push({ label: "two τ: τ₁", color: "var(--qb-c3)", dash: "3 3" }, { label: "two τ: τ₂", color: "var(--qb-c2)", dash: "3 3" });
      vs.legend(items, "bl");
      const pct = (v: number) => (100 * v).toPrecision(2) + "%";
      vsro.textContent = keepRun === "mono"
        ? "Poisson fit, relative RMS error: " + N_LIST.map((N, j) => `${fmtShort(N)} ${pct(res[4][j])}`).join(", ")
        : "two τ, relative RMS error of τ₁ / τ₂: " + N_LIST.map((N, j) => `${fmtShort(N)} ${pct(bi[0][j])}/${pct(bi[1][j])}`).join(", ");
    }, "error vs N");
  }

  function sync(): void { s1.set(tauM.u(st.tau1)); s2.set(tauM.u(st.tau2)); sF.set(st.frac); sN.set(nM.u(st.N)); sB.set(st.bg); }

  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  run();
  if (p0?.vsN) runVsN();
}

autoMount(NAME, mount);
