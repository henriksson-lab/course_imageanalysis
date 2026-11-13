// Solving with ℓ1 in 1D: ISTA, FISTA and orthogonal matching pursuit on a signal of length N
// (256) that is sparse in a dictionary (spikes, Haar, db4, DCT, or spikes ∪ DCT), seen through
// a forward operator (identity, Gaussian blur, random missing samples) plus seeded noise.
// Unknowns are the dictionary coefficients c; the problem is ½‖A W c − y‖² + λ‖c‖₁.
// Panels: the signal (truth, data, estimate); the coefficients as stems (green: on the true
// support, red: false atoms, open circles: missed true atoms) with the ±τλ band; the
// objective gap f(c_k) − f* against the iteration for ISTA and FISTA (log–log); the error
// against λ. The "step" button shows one ISTA iteration as two frames: the gradient
// (Landweber) step, then the soft threshold. "debias" refits the found support by least squares.
// Auto-mounts on <div data-demo="ista1d">. Optional: data-preset="…", data-n="256".
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { WAVELETS } from "../lib/wavelet";
import {
  LinOp, basisOp1d, blurOp1d, maskOp, composeOp, unionOp, identityOp, powerIteration, istaStep, ista, fista, omp, debias,
  denseMatrix, lambdaMax, softVec, lassoObjective, spikesPlusSines, sparseSignal, norm2, Vec,
} from "../lib/sparse";
import { rng } from "../lib/random";

const NAME = "ista1d";
type Dict = "spikes" | "haar" | "db4" | "dct" | "union";
type OpKind = "identity" | "blur" | "mask";
type Sig = "sparse" | "mix" | "pair";
type Solver = "ista" | "fista" | "omp";
interface State { dict: Dict; op: OpKind; sig: Sig; K: number; blur: number; missing: number; noise: number; solver: Solver; lam: number; iters: number; stepMult: number; debias: boolean }
const DEF: State = { dict: "spikes", op: "blur", sig: "sparse", K: 10, blur: 2, missing: 0.5, noise: 0.05, solver: "fista", lam: -1, iters: 100, stepMult: 1, debias: false };
const DICTS: { id: Dict; label: string }[] = [
  { id: "spikes", label: "spikes" }, { id: "haar", label: "Haar" }, { id: "db4", label: "db4" }, { id: "dct", label: "DCT" }, { id: "union", label: "spikes ∪ DCT" },
];
const OPS: { id: OpKind; label: string }[] = [{ id: "identity", label: "identity" }, { id: "blur", label: "blur" }, { id: "mask", label: "missing samples" }];
const SIGS: { id: Sig; label: string }[] = [{ id: "sparse", label: "K-sparse in dictionary" }, { id: "mix", label: "spikes + sines" }, { id: "pair", label: "two close spikes" }];
const SOLVERS: { id: Solver; label: string }[] = [{ id: "ista", label: "ISTA" }, { id: "fista", label: "FISTA" }, { id: "omp", label: "OMP" }];
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "identity", label: "identity: ISTA = one soft threshold", s: { dict: "spikes", op: "identity", sig: "sparse", K: 10, noise: 0.1, solver: "ista", lam: -0.6, iters: 1 } },
  { id: "blur", label: "blur: ISTA needs many steps", s: { dict: "spikes", op: "blur", blur: 2, sig: "sparse", K: 10, noise: 0.02, solver: "ista", lam: -1, iters: 20 } },
  { id: "fista", label: "FISTA vs ISTA", s: { dict: "spikes", op: "blur", blur: 2, sig: "sparse", K: 10, noise: 0.02, solver: "fista", lam: -1, iters: 100 } },
  { id: "ompexact", label: "OMP is exact here", s: { dict: "dct", op: "mask", missing: 0.6, sig: "sparse", K: 6, noise: 0, solver: "omp", lam: -2, iters: 20 } },
  { id: "ompwrong", label: "OMP commits to the wrong atom", s: { dict: "spikes", op: "blur", blur: 3, sig: "pair", noise: 0.005, solver: "omp", lam: -2, iters: 5 } },
  { id: "diverge", label: "step size too large: diverges", s: { dict: "spikes", op: "blur", blur: 2, sig: "sparse", K: 10, noise: 0.02, solver: "fista", lam: -1, iters: 50, stepMult: 2.2 } },
  { id: "inpaint", label: "inpaint a sparse-in-DCT signal", s: { dict: "dct", op: "mask", missing: 0.6, sig: "sparse", K: 8, noise: 0.01, solver: "fista", lam: -2, iters: 300 } },
  { id: "union", label: "spikes + sines need both bases", s: { dict: "union", op: "identity", sig: "mix", noise: 0.05, solver: "fista", lam: -1.2, iters: 300 } },
  { id: "bias", label: "shrinkage bias and debiasing", s: { dict: "spikes", op: "identity", sig: "sparse", K: 10, noise: 0.1, solver: "fista", lam: -0.5, iters: 50, debias: false } },
];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .6em; }
.qb-demo[data-demo="${NAME}"] .i1-left { flex: 1 1 600px; min-width: 0; max-width: 820px; display: flex; flex-direction: column; gap: .1em; }
.qb-demo[data-demo="${NAME}"] .i1-row { display: flex; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .i1-row > div { flex: 1 1 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] .i1-legend { font-size: 16px; display: flex; flex-wrap: wrap; gap: .2em 1em; padding-left: 50px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 330px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .i1-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 16px; gap: .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .i1-left { flex-basis: 700px; max-width: 720px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex-basis: 380px; max-width: 400px; gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 13px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-presets { gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .i1-lab { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; padding: .2em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-slider { gap: .3em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}
const ITERS = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000];
const COL = { truth: "var(--qb-fg)", data: "var(--qb-muted)", est: seriesColor(0), hit: seriesColor(2), fa: seriesColor(1), ista: seriesColor(3), fista: seriesColor(0) };

function mount(host: HTMLElement): void {
  injectStyle();
  const N = Number(host.dataset.n ?? 256) === 128 ? 128 : 256;
  const W = { haar: WAVELETS.haar, db4: WAVELETS.db4 };
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const init: State = { ...DEF, ...(p0?.s ?? {}) };
  const st: State = { ...init };

  // ---- layout ----
  const left = htmlEl("div", "i1-left", "", host);
  const legend = htmlEl("div", "i1-legend", "", left);
  const pSig = new Plot(htmlEl("div", "", "", left), { width: 800, height: 200, x: [0, N], y: [-2, 2], ylabel: "signal", xticks: [0, 64, 128, 192, 256].filter(v => v <= N), margin: { b: 28, t: 8 }, font: 20 });
  const pCoef = new Plot(htmlEl("div", "", "", left), { width: 800, height: 180, x: [0, N], y: [-2, 2], ylabel: "coefficients", xticks: [0, 64, 128, 192, 256].filter(v => v <= N), margin: { b: 28, t: 8 }, font: 20 });
  const row = htmlEl("div", "i1-row", "", left);
  const pObj = new Plot(htmlEl("div", "", "", row), { width: 400, height: 210, x: [1, 1000], y: [1e-8, 10], xlog: true, ylog: true, xlabel: "iteration", ylabel: "f − f*", font: 20, margin: { t: 8 }, yticks: [1e-8, 1e-4, 1, 1e4] });
  const pErr = new Plot(htmlEl("div", "", "", row), { width: 400, height: 210, x: [1e-3, 1], y: [0, 1.2], xlog: true, xlabel: "λ / λmax", ylabel: "rel. error", font: 20, margin: { t: 8 } });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "i1-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEF, p.s); sync(); regenerate(); } })));
  const mkRow = <T extends string>(label: string, items: { id: T; label: string }[], key: keyof State, after: () => void) => {
    htmlEl("div", "i1-lab", label, panel);
    const r = htmlEl("div", "qb-presets", "", panel);
    const btns = items.map(it => button(r, it.label, () => { (st as unknown as Record<string, unknown>)[key] = it.id; changed(); sync(); after(); }));
    return () => btns.forEach((b, i) => b.classList.toggle("active", (st as unknown as Record<string, unknown>)[key] === items[i].id));
  };
  const syncSig = mkRow("signal", SIGS, "sig", () => regenerate());
  const syncDict = mkRow("dictionary", DICTS, "dict", () => regenerate());
  const syncOp = mkRow("forward operator", OPS, "op", () => regenerate());
  const syncSol = mkRow("solver", SOLVERS, "solver", () => solve());
  const kS = slider(panel, "K nonzeros", 1, 40, 1, st.K, v => { st.K = v; changed(); regenerate(); });
  const bS = slider(panel, "blur σ", 0.5, 6, 0.5, st.blur, v => { st.blur = v; changed(); regenerate(); });
  const mS = slider(panel, "missing", 0.1, 0.9, 0.05, st.missing, v => { st.missing = v; changed(); regenerate(); }, v => Math.round(100 * v) + " %");
  const nS = slider(panel, "noise σ", 0, 0.3, 0.005, st.noise, v => { st.noise = v; changed(); regenerate(); });
  const lS = slider(panel, "λ / λmax", -3, 0, 0.05, st.lam, v => { st.lam = v; changed(); solve(); }, v => Math.pow(10, v).toPrecision(2));
  const itS = slider(panel, "iterations", 0, ITERS.length - 1, 1, ITERS.indexOf(st.iters), v => { st.iters = ITERS[v]; changed(); solve(); }, v => String(ITERS[v]));
  const tS = slider(panel, "step τ·L", 0.25, 2.5, 0.05, st.stepMult, v => { st.stepMult = v; changed(); solve(); });
  const cRow = htmlEl("div", "qb-presets", "", panel);
  const dbC = checkbox(cRow, "debias", st.debias, v => { st.debias = v; changed(); solve(); });
  const stepBtn = button(cRow, "step (½ ISTA)", () => halfStep());
  button(cRow, "reset", () => { Object.assign(st, init); sync(); regenerate(); presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0)); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Step τ·L = 1 is the largest safe step 1/L; above 2 ISTA diverges. The objective plot always shows both ISTA and FISTA; OMP has no λ (it stops at the noise level).", panel);
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");

  function changed(): void { presetBtns.forEach(b => b.classList.remove("active")); }
  function sync(): void {
    syncSig(); syncDict(); syncOp(); syncSol();
    kS.set(st.K); bS.set(st.blur); mS.set(st.missing); nS.set(st.noise); lS.set(st.lam); itS.set(Math.max(0, ITERS.indexOf(st.iters))); tS.set(st.stepMult);
    dbC.checked = st.debias;
  }

  // ---- problem ----
  let A: LinOp = identityOp(N), S: LinOp = identityOp(N), y: Vec = new Float64Array(N), xTrue: Vec = new Float64Array(N), cTrue: Vec = new Float64Array(N);
  let known: Uint8Array = new Uint8Array(N).fill(1), L = 1, lmax = 1, Adense: Vec | null = null, fStar = 0, errCurve: { l: number[]; e: number[] } = { l: [], e: [] };
  let stepState: { c: Vec; z: Vec | null; k: number } | null = null;

  function synthOp(): LinOp {
    if (st.dict === "union") return unionOp(identityOp(N), basisOp1d(N, "dct"));
    return basisOp1d(N, st.dict, W, 5);
  }
  function regenerate(): void {
    const r = rng(1234);
    S = synthOp();
    // truth
    if (st.sig === "sparse") {
      const n = S.n, cc = new Float64Array(n);
      if (st.dict === "union") { const half = Math.max(1, Math.round(st.K / 2)); cc.set(sparseSignal(N, half, r), 0); const d = sparseSignal(N / 4, st.K - half || 1, r); for (let i = 0; i < N / 4; i++) cc[N + i] = 3 * d[i]; }
      else if (st.dict === "spikes") cc.set(sparseSignal(N, st.K, r));
      else { const d = sparseSignal(N / 4, st.K, r); for (let i = 0; i < N / 4; i++) cc[i] = 4 * d[i]; }   // low-index (coarse) atoms
      cTrue = cc; xTrue = S.forward(cc);
    } else {
      let x: Vec, cu: Vec | null = null;
      if (st.sig === "mix") { const m = spikesPlusSines(N, r, 4, 3); x = m.x; cu = new Float64Array(2 * N); cu.set(m.spikes, 0); cu.set(m.dct, N); }
      else { x = new Float64Array(N); x[N / 2 - 2] = 1.5; x[N / 2 + 2] = 1.5; x[40] = -1.2; x[200 % N] = 1; x[90] = 0.8; cu = new Float64Array(2 * N); cu.set(x, 0); }
      xTrue = x;
      cTrue = st.dict === "union" ? cu! : S.adjoint(x);
    }
    // operator and data
    known = new Uint8Array(N).fill(1);
    let F: LinOp = identityOp(N);
    if (st.op === "blur") F = blurOp1d(N, st.blur);
    else if (st.op === "mask") { const rm = rng(77); for (let i = 0; i < N; i++) known[i] = rm.unif() < st.missing ? 0 : 1; F = maskOp(known); }
    A = composeOp(F, S);
    const rn = rng(99), clean = F.forward(xTrue);
    y = Float64Array.from(clean, (v, i) => (known[i] ? v + st.noise * rn.norm() : 0));
    L = powerIteration(A, 80);
    lmax = lambdaMax(A, y);
    Adense = null;
    // reference optimum and error curve (independent of λ-slider for the curve; f* depends on λ)
    errCurve = { l: [], e: [] };
    for (let k = 0; k <= 15; k++) {
      const rel = Math.pow(10, -3 + 3 * k / 15);
      const c = fista(A, y, { lambda: rel * lmax, iters: 250, step: 1 / L, history: false }).x;
      errCurve.l.push(rel); errCurve.e.push(relErr(S.forward(c)));
    }
    stepState = null;
    solve();
  }
  const relErr = (x: ArrayLike<number>) => { let s = 0; for (let i = 0; i < N; i++) s += (x[i] - xTrue[i]) ** 2; return Math.sqrt(s) / (norm2(xTrue) || 1); };
  const lam = () => Math.pow(10, st.lam) * lmax;
  const tau = () => st.stepMult / L;

  let cur: { c: Vec; z: Vec | null; label: string; hist: { ista: number[]; fista: number[] } } = { c: new Float64Array(N), z: null, label: "", hist: { ista: [], fista: [] } };
  function solve(): void {
    stepState = null;
    const l = lam(), t = tau();
    fStar = Infinity;
    const ref = fista(A, y, { lambda: l, iters: 1500, step: 1 / L });
    fStar = Math.min(...ref.history);
    const runI = ista(A, y, { lambda: l, iters: Math.max(st.iters, 1), step: t });
    const runF = fista(A, y, { lambda: l, iters: Math.max(st.iters, 1), step: t });
    let c: Vec, label: string;
    if (st.solver === "omp") {
      Adense ??= denseMatrix(A);
      const tol = st.noise > 0 ? 1.05 * st.noise * Math.sqrt(known.reduce((a, b) => a + b, 0)) : 1e-9 * norm2(y);
      const o = omp(Adense, A.m, A.n, y, { K: st.iters, tol });
      c = o.x; label = `OMP, ${o.support.length} atoms`;
    } else {
      c = st.solver === "ista" ? runI.x : runF.x;
      label = `${st.solver.toUpperCase()}, ${st.iters} iterations`;
      if (st.debias) {
        Adense ??= denseMatrix(A);
        const sup: number[] = []; for (let i = 0; i < c.length; i++) if (c[i] !== 0) sup.push(i);
        if (sup.length < A.m) { c = debias(Adense, A.m, A.n, y, sup); label += ", debiased"; }
      }
    }
    cur = { c, z: null, label, hist: { ista: runI.history, fista: runF.history } };
    draw();
  }

  function halfStep(): void {
    changed();
    if (st.solver === "omp") { st.solver = "ista"; sync(); }
    const l = lam(), t = tau();
    if (!stepState) stepState = { c: new Float64Array(A.n), z: null, k: 0 };
    if (!stepState.z) {
      stepState.z = istaStep(A, y, stepState.c, t, v => v).z;
      cur = { ...cur, c: stepState.c, z: stepState.z, label: `ISTA iteration ${stepState.k + 1}: gradient step` };
    } else {
      stepState.c = softVec(stepState.z, t * l); stepState.z = null; stepState.k++;
      cur = { ...cur, c: stepState.c, z: null, label: `ISTA iteration ${stepState.k}: threshold at τλ` };
    }
    draw();
  }

  // ---- drawing ----
  function draw(): void {
    const c = cur.c, xh = S.forward(cur.z ?? c);
    const idx = Float64Array.from({ length: N }, (_, i) => i);
    const all = [xTrue, xh, y.filter((_, i) => known[i])];
    let lo = Infinity, hi = -Infinity;
    for (const a of all) for (const v of a) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (!Number.isFinite(lo) || hi - lo > 1e6) { lo = -3; hi = 3; }
    const pad = 0.1 * (hi - lo || 1);
    pSig.setDomain([0, N], [lo - pad, hi + pad]);
    pSig.clear();
    if (st.op === "mask") {
      const kx: number[] = [], ky: number[] = [];
      for (let i = 0; i < N; i++) if (known[i]) { kx.push(i); ky.push(y[i]); }
      pSig.scatter(kx, ky, { color: COL.data, r: 3 });
    } else pSig.line(idx, y, { color: COL.data, width: 2 });
    pSig.line(idx, xTrue, { color: COL.truth, width: 1.5 });
    pSig.line(idx, xh, { color: COL.est, width: 2.5 });
    legend.innerHTML = `<span>━ <b style="color:${COL.truth}">truth</b></span><span style="color:${COL.data}">━ data</span><span style="color:${COL.est}">━ ${cur.z ? "after the gradient step" : "estimate"}</span><span>${cur.label}</span>`;

    // coefficients
    const n = c.length, show = cur.z ?? c;
    let cmax = 0; for (let i = 0; i < n; i++) cmax = Math.max(cmax, Math.abs(cTrue[i]), Math.abs(show[i]));
    if (!Number.isFinite(cmax) || cmax > 1e6) cmax = 1e6;
    pCoef.setDomain([0, n], [-1.15 * (cmax || 1), 1.15 * (cmax || 1)]);
    pCoef.clear();
    const t = tau() * lam();
    if (st.solver !== "omp" || cur.z) pCoef.band([0, n], [-t, -t], [t, t], { color: seriesColor(3), fillOpacity: 0.25 });
    const tmax = Math.max(...Array.from(cTrue, Math.abs));
    const onT = (i: number) => Math.abs(cTrue[i]) > 1e-3 * tmax;
    pCoef.mark(g => {
      for (let i = 0; i < n; i++) {
        const v = show[i];
        if (v !== 0) {
          const col = cur.z ? COL.est : onT(i) ? COL.hit : COL.fa;
          const ln = document.createElementNS("http://www.w3.org/2000/svg", "line");
          ln.setAttribute("x1", String(pCoef.x(i))); ln.setAttribute("x2", String(pCoef.x(i)));
          ln.setAttribute("y1", String(pCoef.y(0))); ln.setAttribute("y2", String(pCoef.y(Math.max(-1e6, Math.min(1e6, v)))));
          ln.setAttribute("style", `stroke:${col};stroke-width:2.5`);
          g.appendChild(ln);
        }
        if (onT(i) && !cur.z) {
          const ci = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          ci.setAttribute("cx", String(pCoef.x(i))); ci.setAttribute("cy", String(pCoef.y(cTrue[i]))); ci.setAttribute("r", "4");
          ci.setAttribute("style", `fill:none;stroke:${v === 0 ? COL.fa : "var(--qb-fg)"};stroke-width:1.5`);
          g.appendChild(ci);
        }
      }
    }, []);
    pCoef.hline(0, { color: "var(--qb-axis)", width: 1 });

    // objective gap
    pObj.clear();
    const gap = (h: number[]) => h.map(v => Math.max(v - fStar, 1e-8));
    const ks = (h: number[]) => h.map((_, i) => i + 1);
    const hI = cur.hist.ista, hF = cur.hist.fista;
    const top = Math.max(10, ...gap(hI).filter(Number.isFinite), ...gap(hF).filter(Number.isFinite));
    pObj.setDomain([1, Math.max(10, st.iters)], [1e-8, Math.min(1e12, top * 2)]);
    if (hI.length > 1) { pObj.line(ks(hI), gap(hI), { color: COL.ista, label: "ISTA" }); pObj.line(ks(hF), gap(hF), { color: COL.fista, label: "FISTA" }); pObj.legend(undefined, "bl"); }
    else if (hI.length === 1) { pObj.scatter([1], gap(hI), { color: COL.ista, r: 5, label: "ISTA" }); pObj.scatter([1], gap(hF), { color: COL.fista, r: 4, label: "FISTA" }); }

    // error vs λ
    pErr.clear();
    const emax = Math.max(1.05, ...errCurve.e.map(v => Math.min(v, 3)));
    pErr.setDomain([1e-3, 1], [0, emax]);
    pErr.line(errCurve.l, errCurve.e.map(v => Math.min(v, 3)), { color: COL.fista });
    pErr.vline(Math.pow(10, st.lam), { color: "var(--qb-fg)", width: 2 });

    // readout
    let nz = 0, hits = 0, fa = 0, miss = 0;
    for (let i = 0; i < n; i++) { const a = c[i] !== 0; if (a) nz++; if (a && onT(i)) hits++; if (a && !onT(i)) fa++; if (!a && onT(i)) miss++; }
    const f = lassoObjective(A, y, c, lam());
    ro.textContent = [
      `λ = ${lam().toPrecision(3)}   λmax = ${lmax.toPrecision(3)}`,
      `L = ${L.toPrecision(3)}   τ = ${tau().toPrecision(3)}   τλ = ${(tau() * lam()).toPrecision(3)}`,
      `nonzeros ${nz}: on support ${hits}, false ${fa}, missed ${miss}`,
      `relative error ${Number.isFinite(relErr(xh)) && relErr(xh) < 1e6 ? relErr(xh).toFixed(3) : "∞ (diverged)"}`,
      `objective ${Number.isFinite(f) && f < 1e9 ? f.toPrecision(4) : "∞"}`,
    ].join("\n");
    stepBtn.classList.toggle("active", !!stepState);
  }

  sync();
  regenerate();
}

autoMount(NAME, mount);
