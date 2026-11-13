// Compressed sensing in 1D: a signal of length N = 256 that is K-sparse (or compressible) in a
// basis (spikes, DCT, Haar) is measured with M < N linear measurements: random rows of the real
// Fourier basis, every r-th frequency (regular), a random Gaussian matrix, or random pixels.
// Panels: the truth with the ℓ1 recovery (FISTA with continuation, then a least-squares polish
// on the support when there is no noise) and the OMP recovery; what is measured (the spectrum
// with the sampled rows, the Gaussian measurements, or the sampled pixels); the zero-filled
// (minimum-norm for orthonormal rows) reconstruction Aᵀy with the true signal overlaid; and the
// sampling point spread function (the zero-filled image of one spike). "animate" replays the
// thresholding iterations.
// Auto-mounts on <div data-demo="csrecover">. Optional: data-preset="…".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { WAVELETS, realFourier } from "../lib/wavelet";
import {
  LinOp, basisOp1d, composeOp, realFourierOp1d, pixelOp1d, matrixOp, gaussianMatrix, denseMatrix, fourierRows1d, rowsPsf1d,
  sparseSignal, compressibleSignal, l1Recover, omp, norm2, randomSupport, Vec, realFourierFreq, meanColumnNorm,
} from "../lib/sparse";
import { rng } from "../lib/random";

const NAME = "csrecover";
const N = 256;
type Basis = "spikes" | "dct" | "haar";
type Sensing = "fourier" | "regular" | "gaussian" | "pixels";
interface State { basis: Basis; sig: "sparse" | "compressible"; K: number; sensing: Sensing; M: number; noise: number }
const DEF: State = { basis: "spikes", sig: "sparse", K: 8, sensing: "fourier", M: 64, noise: 0 };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "regular", label: "regular: copies", s: { sensing: "regular", M: 64, K: 8 } },
  { id: "random", label: "random: noise-like junk", s: { sensing: "fourier", M: 64, K: 8 } },
  { id: "recover", label: "ℓ1 recovers from 25 %", s: { sensing: "fourier", M: 64, K: 10 } },
  { id: "toofew", label: "too few: abrupt failure", s: { sensing: "fourier", M: 32, K: 10 } },
  { id: "pixels", label: "random pixels see nothing", s: { sensing: "pixels", M: 64, K: 8 } },
  { id: "gaussian", label: "Gaussian matrix works for any basis", s: { basis: "dct", sensing: "gaussian", M: 64, K: 8 } },
  { id: "compressible", label: "compressible signal: best-K error", s: { basis: "dct", sig: "compressible", sensing: "gaussian", M: 96 } },
  { id: "noise", label: "noise: stable, not exact", s: { sensing: "fourier", M: 64, K: 8, noise: 0.05 } },
];
const BASES: { id: Basis; label: string }[] = [{ id: "spikes", label: "spikes" }, { id: "dct", label: "DCT" }, { id: "haar", label: "Haar" }];
const SENS: { id: Sensing; label: string }[] = [
  { id: "fourier", label: "random Fourier" }, { id: "regular", label: "every r-th frequency" }, { id: "gaussian", label: "Gaussian matrix" }, { id: "pixels", label: "random pixels" },
];
const COL = { truth: "var(--qb-fg)", l1: seriesColor(0), omp: seriesColor(3), zf: seriesColor(1), meas: seriesColor(2) };

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .6em; }
.qb-demo[data-demo="${NAME}"] .cs-left { flex: 1 1 600px; min-width: 0; max-width: 820px; display: flex; flex-direction: column; gap: 0; }
.qb-demo[data-demo="${NAME}"] .cs-legend { font-size: 16px; display: flex; flex-wrap: wrap; gap: .1em 1em; padding-left: 50px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 340px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .cs-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 16px; gap: .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .cs-left { flex-basis: 720px; max-width: 730px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex-basis: 370px; max-width: 390px; gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 13px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-presets { gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .cs-lab { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; padding: .2em .4em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const W = { haar: WAVELETS.haar, db4: WAVELETS.db4 };
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const init: State = { ...DEF, ...(p0?.s ?? {}) };
  const st: State = { ...init };
  const idx = Float64Array.from({ length: N }, (_, i) => i);
  const ticks = [0, 64, 128, 192, 256];

  const left = htmlEl("div", "cs-left", "", host);
  const leg1 = htmlEl("div", "cs-legend", "", left);
  const pRec = new Plot(htmlEl("div", "", "", left), { width: 800, height: 190, x: [0, N], y: [-2, 2], ylabel: "recovered", xticks: ticks, margin: { b: 26, t: 6 }, font: 20 });
  const leg2 = htmlEl("div", "cs-legend", "", left);
  const pMeas = new Plot(htmlEl("div", "", "", left), { width: 800, height: 150, x: [0, N], y: [0, 1], ylabel: "measured", xticks: ticks, margin: { b: 26, t: 6 }, font: 20 });
  const leg3 = htmlEl("div", "cs-legend", "", left);
  const pZf = new Plot(htmlEl("div", "", "", left), { width: 800, height: 160, x: [0, N], y: [-2, 2], ylabel: "zero-filled", xticks: ticks, margin: { b: 26, t: 6 }, font: 20 });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "cs-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEF, p.s); sync(); schedule(); } })));
  htmlEl("div", "cs-lab", "sparse in", panel);
  const bRow = htmlEl("div", "qb-presets", "", panel);
  const bBtns = BASES.map(b => button(bRow, b.label, () => { st.basis = b.id; changed(); sync(); schedule(); }));
  const sigBtns = [
    button(bRow, "K-sparse", () => { st.sig = "sparse"; changed(); sync(); schedule(); }),
    button(bRow, "compressible", () => { st.sig = "compressible"; changed(); sync(); schedule(); }),
  ];
  htmlEl("div", "cs-lab", "measurements", panel);
  const sRow = htmlEl("div", "qb-presets", "", panel);
  const sBtns = SENS.map(b => button(sRow, b.label, () => { st.sensing = b.id; changed(); sync(); schedule(); }));
  const kS = slider(panel, "K nonzeros", 1, 40, 1, st.K, v => { st.K = v; changed(); schedule(); });
  const mS = slider(panel, "M measurements", 8, 256, 4, st.M, v => { st.M = v; changed(); schedule(); });
  const nS = slider(panel, "noise σ", 0, 0.2, 0.01, st.noise, v => { st.noise = v; changed(); schedule(); });
  const aRow = htmlEl("div", "qb-presets", "", panel);
  const animBtn = button(aRow, "animate ℓ1 iterations", () => animate());
  button(aRow, "reset", () => { Object.assign(st, init); sync(); schedule(); presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0)); });
  const pPsf = new Plot(htmlEl("div", "", "", panel), { width: 380, height: 150, x: [-N / 2, N / 2], y: [-0.2, 1], xlabel: "PSF: zero-filled image of one spike", xticks: [-128, -64, 0, 64, 128], font: 20, margin: { t: 6 } });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "ℓ1: FISTA with λ decreasing to the noise level, then least squares on the support found. OMP stops at the noise level.", panel);
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");

  function changed(): void { presetBtns.forEach(b => b.classList.remove("active")); }
  function sync(): void {
    bBtns.forEach((b, i) => b.classList.toggle("active", BASES[i].id === st.basis));
    sigBtns.forEach((b, i) => b.classList.toggle("active", ["sparse", "compressible"][i] === st.sig));
    sBtns.forEach((b, i) => b.classList.toggle("active", SENS[i].id === st.sensing));
    kS.set(st.K); mS.set(st.M); nS.set(st.noise);
  }
  let pending = 0;
  function schedule(): void { if (!pending) pending = requestAnimationFrame(() => { pending = 0; compute(); }); }

  // ---- computation ----
  let res: {
    x: Vec; c: Vec; xl1: Vec; xomp: Vec; zf: Vec; y: Vec; rows: Int32Array | null; Phi: LinOp; Psi: LinOp; A: Vec; M: number; snaps: Vec[]; r: number;
  } | null = null;
  function compute(): void {
    const Psi = basisOp1d(N, st.basis, W, 5);
    const r0 = rng(31);
    let c: Vec;
    if (st.sig === "compressible") c = compressibleSignal(N, r0, 1.3);
    else if (st.basis === "spikes") c = sparseSignal(N, st.K, r0);
    else { c = new Float64Array(N); const d = sparseSignal(N / 4, st.K, r0); for (let i = 0; i < N / 4; i++) c[i] = 4 * d[i]; }
    const x = Psi.forward(c);
    // measurement operator (seeded per kind)
    const rs = rng(57);
    let Phi: LinOp, rows: Int32Array | null = null, r = 1;
    if (st.sensing === "fourier") { rows = fourierRows1d(N, "random", st.M, rs); Phi = realFourierOp1d(N, rows); }
    else if (st.sensing === "regular") {
      r = Math.max(1, Math.round(N / st.M)); rows = fourierRows1d(N, "regular", 0, rs, r); Phi = realFourierOp1d(N, rows);
    } else if (st.sensing === "pixels") { rows = randomSupport(N, st.M, rs); Phi = pixelOp1d(N, rows); }
    else Phi = matrixOp(gaussianMatrix(st.M, N, rs), st.M, N);
    const M = Phi.m;
    const A = denseMatrix(composeOp(Phi, Psi));
    const rn = rng(83);
    const y = Float64Array.from(Phi.forward(x), v => v + st.noise * rn.norm());
    const snaps: Vec[] = [];
    const lamF = st.noise > 0 ? st.noise * meanColumnNorm(A, M, N) * Math.sqrt(2 * Math.log(N)) : undefined;
    const cl1 = l1Recover(A, M, N, y, { lambdaFinal: lamF, iters: 120, snapshotEvery: 12, snapshots: snaps });
    const tol = st.noise > 0 ? 1.05 * st.noise * Math.sqrt(M) : 1e-8 * norm2(y);
    const comp = omp(A, M, N, y, { K: Math.min(M, st.sig === "compressible" ? Math.floor(M / 2) : Math.max(2 * st.K, 4)), tol }).x;
    res = { x, c, xl1: Psi.forward(cl1), xomp: Psi.forward(comp), zf: Float64Array.from(Phi.adjoint(y), v => v * N / Phi.m), y, rows, Phi, Psi, A, M, snaps: snaps.map(s => Psi.forward(s)), r };
    draw();
  }

  let anim = 0;
  function animate(): void {
    if (!res) return;
    cancelAnimationFrame(anim);
    let k = 0;
    const step = () => {
      if (!res) return;
      if (k < res.snaps.length) { draw(res.snaps[k], `iteration ${12 * (k + 1)}`); k++; anim = requestAnimationFrame(() => setTimeout(step, 60)); }
      else { draw(); animBtn.classList.remove("active"); }
    };
    animBtn.classList.add("active");
    step();
  }

  const rel = (a: ArrayLike<number>, b: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return Math.sqrt(s) / (norm2(b) || 1); };

  function draw(frame?: Vec, frameLabel = ""): void {
    if (!res) return;
    const { x, xl1, xomp, zf, y } = res;
    let lo = Math.min(...x), hi = Math.max(...x);
    const pad = 0.15 * (hi - lo || 1); lo -= pad; hi += pad;
    pRec.setDomain([0, N], [lo, hi]);
    pRec.clear();
    pRec.line(idx, x, { color: COL.truth, width: 1.5 });
    if (frame) pRec.line(idx, frame, { color: COL.l1, width: 2.5 });
    else { pRec.line(idx, xomp, { color: COL.omp, width: 2, dash: "5 3" }); pRec.line(idx, xl1, { color: COL.l1, width: 2.5 }); }
    leg1.innerHTML = frame ? `<span>━ truth</span><span style="color:${COL.l1}">━ ℓ1, ${frameLabel}</span>`
      : `<span>━ truth</span><span style="color:${COL.l1}">━ ℓ1 recovery</span><span style="color:${COL.omp}">╍ OMP</span>`;

    // measured
    pMeas.clear();
    if (st.sensing === "fourier" || st.sensing === "regular") {
      const sp = realFourier(x), mag = Float64Array.from({ length: N / 2 + 1 }, (_, f) => 0);
      for (let j = 0; j < N; j++) { const f = realFourierFreq(j, N); mag[f] = Math.hypot(mag[f], sp[j]); }
      const mx = Math.max(...mag) || 1;
      pMeas.setDomain([0, N / 2], [0, 1.1 * mx]);
      const fs = Float64Array.from(mag, (_, f) => f);
      pMeas.bars(fs, mag, { color: "var(--qb-muted)", barWidth: 0.6, opacity: 0.6 });
      const sel = new Set<number>(); for (const j of res.rows!) sel.add(realFourierFreq(j, N));
      const sx: number[] = [], sy: number[] = [];
      for (const f of sel) { sx.push(f); sy.push(mag[f]); }
      pMeas.scatter(sx, sy, { color: COL.meas, r: 4 });
      leg2.innerHTML = `<span style="color:var(--qb-muted)">▮ |Fourier coefficient| at each frequency (0 … ${N / 2})</span><span style="color:${COL.meas}">● measured (${res.M} of ${N} numbers)</span>`;
    } else if (st.sensing === "pixels") {
      pMeas.setDomain([0, N], [lo, hi]);
      pMeas.line(idx, x, { color: "var(--qb-muted)", width: 1.2 });
      const px = Array.from(res.rows!), py = px.map(i => y[res!.rows!.indexOf(i)]);
      pMeas.scatter(px, py, { color: COL.meas, r: 4 });
      leg2.innerHTML = `<span style="color:${COL.meas}">● measured pixels (${res.M} of ${N})</span>`;
    } else {
      const m = res.M, mx = Math.max(...Array.from(y, Math.abs)) || 1;
      pMeas.setDomain([0, m], [-1.1 * mx, 1.1 * mx]);
      pMeas.bars(Float64Array.from({ length: m }, (_, i) => i + 0.5), y, { color: COL.meas, barWidth: 0.7 });
      leg2.innerHTML = `<span style="color:${COL.meas}">▮ ${m} measurements, each a random weighted sum of all ${N} samples</span>`;
    }

    // zero-filled
    let zl = Math.min(lo, ...zf), zh = Math.max(hi, ...zf);
    pZf.setDomain([0, N], [zl, zh]);
    pZf.clear();
    pZf.line(idx, x, { color: COL.truth, width: 1.2, opacity: 0.6 });
    pZf.line(idx, zf, { color: COL.zf, width: 2 });
    leg3.innerHTML = `<span style="color:${COL.zf}">━ zero-filled Aᵀy × N/M (unmeasured = 0)</span><span>━ truth</span>`;

    // PSF
    pPsf.clear();
    let psf: Vec;
    if (st.sensing === "fourier" || st.sensing === "regular") psf = rowsPsf1d(N, res.rows!);
    else if (st.sensing === "pixels") { psf = new Float64Array(N); if (Array.from(res.rows!).includes(N / 2)) psf[N / 2] = 1; }
    else { const e = new Float64Array(N); e[N / 2] = 1; psf = res.Phi.adjoint(res.Phi.forward(e)); }
    psf = Float64Array.from(psf, v => v * N / res!.M);
    const pk = Math.max(...Array.from(psf, Math.abs)) || 1;
    pPsf.setDomain([-N / 2, N / 2], [-0.35 * Math.max(pk, 0.5), 1.1 * Math.max(pk, 0.5)]);
    pPsf.line(Float64Array.from(idx, i => i - N / 2), psf, { color: COL.zf, width: 1.8 });

    // readout
    const eL1 = rel(xl1, x), eOmp = rel(xomp, x), eZf = rel(zf, x);
    let extra = "";
    if (st.sig === "compressible") {
      const Kb = Math.floor(res.M / 4), sorted = Array.from(res.c, v => v * v).sort((a, b) => b - a);
      const tail = sorted.slice(Kb).reduce((a, b) => a + b, 0), tot = sorted.reduce((a, b) => a + b, 0);
      extra = `\nbest ${Kb}-term error ${Math.sqrt(tail / tot).toFixed(3)}`;
    }
    const kNow = st.sig === "compressible" ? NaN : st.K;
    ro.textContent = [
      `M = ${res.M}  M/N = ${(res.M / N).toFixed(2)}` + (Number.isFinite(kNow) ? `  K/M = ${(kNow / res.M).toFixed(2)}` : "") + (st.sensing === "regular" ? `  r = ${res.r}` : ""),
      `relative error: ℓ1 ${eL1.toFixed(3)}  OMP ${eOmp.toFixed(3)}`,
      `zero-filled ${eZf.toFixed(3)}  ${eL1 < 1e-2 ? "ℓ1: recovered" : eL1 < 0.2 ? "ℓ1: close" : "ℓ1: failed"}` + extra,
    ].join("\n");
  }

  sync();
  compute();
}

autoMount(NAME, mount);
