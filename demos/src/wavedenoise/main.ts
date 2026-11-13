// Denoising by wavelet shrinkage. A 256 × 256 image plus seeded Gaussian noise of standard
// deviation σ_n; DWT, threshold the detail coefficients (LL untouched) with a hard or soft
// threshold t, inverse DWT. t is set in units of σ̂_n, the noise estimate median(|w|)/0.6745
// from the finest diagonal band D; the universal threshold √(2 ln N) σ̂_n is marked.
// Cycle spinning averages the result over 4 or 16 shifts. Panels: noisy, denoised, removed
// part (noisy − denoised) and error (denoised − truth). Plots: PSNR against t for hard and
// soft (computed in the background, a few thresholds per frame), and the histogram of the
// finest D coefficients with ±t. data-mode="1d": the same on a 1D test signal (length 1024).
// Auto-mounts on <div data-demo="wavedenoise">. Optional: data-mode="2d|1d", data-image="shapes|cell",
// data-signal="blocks|bumps|heavisine|doppler", data-preset="…".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import {
  WAVELETS, WaveletName, Wavelet, dwt2, subband, madSigma, universalThreshold, denoise2, denoise1, dwt, testSignal, SignalName, SIGNAL_LABELS,
} from "../lib/wavelet";
import { psnr } from "../lib/metrics";
import { rng } from "../lib/random";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { N, synthImage, drawImage, drawSigned } from "../dwt2d/images";

const NAME = "wavedenoise";
const WNAMES: WaveletName[] = ["haar", "db2", "db4", "db8"];
const SPINS = [1, 4, 16];
const TS = Array.from({ length: 25 }, (_, i) => i * 0.25);   // sweep 0 … 6 (units of σ̂)
const N1 = 1024;
const SIG1: SignalName[] = ["blocks", "bumps", "heavisine", "doppler"];
const COL = { hard: seriesColor(1), soft: seriesColor(0) };

type Kind = "hard" | "soft";
interface Settings { image: "shapes" | "cell"; signal: SignalName; sigma: number; wavelet: WaveletName; levels: number; kind: Kind; t: number; spin: number }
const DEF2: Settings = { image: "shapes", signal: "blocks", sigma: 20, wavelet: "db4", levels: 4, kind: "soft", t: 3, spin: 1 };
const DEF1: Settings = { ...DEF2, sigma: 1, levels: 5 };
interface Preset { id: string; label: string; s: Partial<Settings> }
const UNIV2 = Math.sqrt(2 * Math.log(N * N)), UNIV1 = Math.sqrt(2 * Math.log(N1));
const r2 = (v: number) => Math.round(v * 4) / 4;
const PRESETS2: Preset[] = [
  { id: "universal", label: "universal threshold, soft", s: { image: "shapes", wavelet: "db4", kind: "soft", t: r2(UNIV2) } },
  { id: "hardsoft", label: "hard vs soft at the same t", s: { image: "shapes", wavelet: "db4", kind: "hard", t: 3 } },
  { id: "low", label: "threshold too low: blips", s: { image: "shapes", wavelet: "db4", kind: "hard", t: 1.5 } },
  { id: "high", label: "too high: blur", s: { image: "shapes", wavelet: "db4", kind: "soft", t: 6 } },
  { id: "haar", label: "Haar blocks", s: { image: "shapes", wavelet: "haar", kind: "soft", t: 3, spin: 1 } },
  { id: "spin", label: "… with cycle spinning", s: { image: "shapes", wavelet: "haar", kind: "soft", t: 3, spin: 16 } },
  { id: "cell", label: "micrograph: texture vs noise", s: { image: "cell", sigma: 40, wavelet: "db4", kind: "soft", t: 1.25 } },
  { id: "sigma", label: "σ̂_n fooled by texture", s: { image: "cell", sigma: 10, wavelet: "db4", kind: "soft", t: 3 } },
];
const PRESETS1: Preset[] = [
  { id: "blocks", label: "Blocks, Haar", s: { signal: "blocks", wavelet: "haar", kind: "soft", t: r2(UNIV1) } },
  { id: "heavisine", label: "HeaviSine, db4", s: { signal: "heavisine", wavelet: "db4", kind: "soft", t: r2(UNIV1) } },
  { id: "doppler", label: "Doppler, hard vs soft", s: { signal: "doppler", wavelet: "db8", kind: "hard", t: 3 } },
  { id: "bumps", label: "Bumps: too high", s: { signal: "bumps", wavelet: "db4", kind: "soft", t: 6 } },
];

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] .wd-grid { flex: 0 1 520px; min-width: 260px; display: grid; grid-template-columns: 1fr 1fr; gap: .3em .5em; }
[data-demo="${NAME}"] .wd-wide { flex: 1 1 560px; min-width: 0; max-width: 760px; }
[data-demo="${NAME}"] figure { margin: 0; min-width: 0; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; line-height: 1.15; margin: 0 0 .15em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 2px; }
[data-demo="${NAME}"] .wd-side { flex: 1 1 250px; max-width: 330px; display: flex; flex-direction: column; gap: .2em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 240px; max-width: 320px; gap: .38em; }
[data-demo="${NAME}"] .wd-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .wd-credit { font-size: 12px; color: var(--qb-muted); }
/* Pages: figures, plots and controls as three columns that fit the wide column (~1075 px). */
.qb-demo[data-demo="${NAME}"] .wd-grid { flex: 0 1 400px; max-width: calc(100vh - 250px); }
.qb-demo[data-demo="${NAME}"] .wd-wide { flex: 1 1 400px; }
.qb-demo[data-demo="${NAME}"] .wd-side { flex: 0 1 290px; }
.qb-demo[data-demo="${NAME}"] .wd-side .qb-readout { font-size: 13px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 290px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-btn { font-size: 14px; padding: .15em .5em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-presets { gap: .25em; }
.reveal .qb-demo[data-demo="${NAME}"] .wd-grid { flex: 0 1 450px; max-width: none; }
.reveal .qb-demo[data-demo="${NAME}"] .wd-side { flex: 1 1 260px; }
.reveal .qb-demo[data-demo="${NAME}"] .wd-side .qb-readout { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 290px; gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel .qb-btn { font-size: 16px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] { font-size: 16px; gap: .5em; }
.reveal [data-demo="${NAME}"] .wd-grid { flex-basis: 450px; gap: .1em .4em; }
.reveal [data-demo="${NAME}"] figcaption { font-size: 15px; }
.reveal [data-demo="${NAME}"] .wd-side { flex-basis: 260px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex-basis: 290px; gap: .2em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.reveal [data-demo="${NAME}"] .wd-lab { display: none; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 13px; padding: .1em .4em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const one = host.dataset.mode === "1d";
  const PRESETS = one ? PRESETS1 : PRESETS2;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const init: Settings = { ...(one ? DEF1 : DEF2) };
  if (host.dataset.image === "cell") init.image = "cell";
  if (SIG1.includes(host.dataset.signal as SignalName)) init.signal = host.dataset.signal as SignalName;
  if (p0) Object.assign(init, p0.s);
  const st: Settings = { ...init };
  const UNIV = one ? UNIV1 : UNIV2;

  // ---- layout ----
  const F: { c: HTMLElement; cv: HTMLCanvasElement }[] = [];
  let pSig: Plot | null = null;
  if (one) {
    const wide = htmlEl("div", "wd-wide", "", host);
    pSig = new Plot(wide, { width: 760, height: 420, x: [0, N1], y: [-15, 15], xlabel: "position", ylabel: "signal" });
  } else {
    const grid = htmlEl("div", "wd-grid", "", host);
    for (const cap of ["noisy", "denoised", "removed (noisy − denoised)", "error (denoised − truth)"]) {
      const f = htmlEl("figure", "", "", grid), c = htmlEl("figcaption", "", cap, f), cv = htmlEl("canvas", "", "", f);
      cv.width = N; cv.height = N; F.push({ c, cv });
    }
  }
  const side = htmlEl("div", "wd-side", "", host);
  const pT = new Plot(htmlEl("div", "", "", side), { width: 360, height: 270, x: [0, 6], y: [0, 1], xlabel: "t / σ̂", ylabel: one ? "RMSE" : "PSNR (dB)", font: 22 });
  const pH = new Plot(htmlEl("div", "", "", side), { width: 360, height: 210, x: [-1, 1], y: [0, 1], xlabel: "finest D coefficients", yticks: [], margin: { l: 14 }, font: 22 });
  const ro = htmlEl("div", "qb-readout", "", side);
  const credit = htmlEl("div", "wd-credit", CELL_CREDIT, side);
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "wd-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, one ? DEF1 : DEF2, p.s); sync(); newNoisy(); } })));
  htmlEl("div", "wd-lab", one ? "signal" : "image", panel);
  const iRow = htmlEl("div", "qb-presets", "", panel);
  const iBtns = one
    ? SIG1.map(s => button(iRow, SIGNAL_LABELS[s], () => { st.signal = s; changed(); newNoisy(); }))
    : (["shapes", "cell"] as const).map(im => button(iRow, im === "cell" ? "micrograph" : "shapes", () => { st.image = im; changed(); newNoisy(); }));
  const sigS = one
    ? slider(panel, "noise σ_n", 0.25, 3, 0.25, st.sigma, v => { st.sigma = v; changed(); newNoisy(); })
    : slider(panel, "noise σ_n", 5, 50, 5, st.sigma, v => { st.sigma = v; changed(); newNoisy(); });
  htmlEl("div", "wd-lab", "wavelet", panel);
  const wRow = htmlEl("div", "qb-presets", "", panel);
  const wBtns = WNAMES.map(w => button(wRow, WAVELETS[w].label, () => { st.wavelet = w; changed(); newTransform(); }));
  const levS = slider(panel, "levels", 1, one ? 7 : 5, 1, st.levels, v => { st.levels = v; changed(); newTransform(); });
  const kRow = htmlEl("div", "qb-presets", "", panel);
  const kBtns = (["hard", "soft"] as Kind[]).map(k => button(kRow, k === "hard" ? "hard (keep or kill)" : "soft (kill and shrink)", () => { st.kind = k; changed(); run(); }));
  const tS = slider(panel, "t / σ̂_n", 0, 6, 0.25, st.t, v => { st.t = v; changed(); run(); });
  button(panel, `t = universal (${UNIV.toFixed(2)})`, () => { st.t = r2(UNIV); changed(); sync(); run(); });
  htmlEl("div", "wd-lab", "cycle spinning (shifts)", panel);
  const cRow = htmlEl("div", "qb-presets", "", panel);
  const cBtns = SPINS.map(s => button(cRow, String(s), () => { st.spin = s; changed(); newTransform(); }));
  button(panel, "reset", () => { Object.assign(st, init); sync(); newNoisy(); presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0)); });
  htmlEl("div", "qb-note", "Only the detail coefficients are thresholded; the coarsest approximation is kept. The removed part should look like pure noise.", panel);

  function changed(): void { presetBtns.forEach(b => b.classList.remove("active")); }
  function sync(): void {
    iBtns.forEach((b, i) => b.classList.toggle("active", one ? SIG1[i] === st.signal : ["shapes", "cell"][i] === st.image));
    wBtns.forEach((b, i) => b.classList.toggle("active", WNAMES[i] === st.wavelet));
    kBtns.forEach((b, i) => b.classList.toggle("active", ["hard", "soft"][i] === st.kind));
    cBtns.forEach((b, i) => b.classList.toggle("active", SPINS[i] === st.spin));
    sigS.set(st.sigma); levS.set(st.levels); tS.set(st.t);
    credit.style.display = !one && st.image === "cell" ? "" : "none";
  }

  // ---- data ----
  type F64 = Float64Array<ArrayBufferLike>;
  let cell: F64 | null = null;
  const noise2 = (() => { const r = rng(7); return Float64Array.from({ length: N * N }, () => r.norm()); })();
  const noise1 = (() => { const r = rng(11); return Float64Array.from({ length: N1 }, () => r.norm()); })();
  let truth: F64 = new Float64Array(0), noisy: F64 = new Float64Array(0), sigHat = 0, finestD: F64 = new Float64Array(0);
  let w: Wavelet = WAVELETS.db4;

  function newNoisy(): void {
    sync();
    if (one) {
      const s = testSignal(st.signal, N1);
      let m = 0, v = 0;
      for (const x of s) m += x / N1;
      for (const x of s) v += (x - m) ** 2 / N1;
      truth = Float64Array.from(s, x => 7 * (x - m) / Math.sqrt(v));   // standard deviation 7 (Donoho–Johnstone)
      noisy = Float64Array.from(truth, (x, i) => x + st.sigma * noise1[i]);
    } else {
      if (st.image === "cell" && !cell) { loadCell().then(g => { cell = Float64Array.from(g.data); newNoisy(); }); return; }
      truth = st.image === "cell" ? cell! : synthImage("shapes");
      noisy = Float64Array.from(truth, (x, i) => x + st.sigma * noise2[i]);
    }
    newTransform();
  }

  // the PSNR/RMSE-vs-t sweep runs in slices between frames
  let sweepGen = 0;
  const sweep: Record<Kind, Float64Array> = { hard: new Float64Array(TS.length).fill(NaN), soft: new Float64Array(TS.length).fill(NaN) };
  function newTransform(): void {
    sync();
    w = WAVELETS[st.wavelet];
    if (one) { const c = dwt(noisy, w, st.levels); finestD = c.slice(N1 / 2); }
    else { const c = dwt2(noisy, N, N, w, st.levels); finestD = subband(c, N, N, 1, "D"); }
    sigHat = madSigma(finestD);
    sweep.hard.fill(NaN); sweep.soft.fill(NaN);
    const gen = ++sweepGen;
    const jobs: [Kind, number][] = [];
    for (let i = 0; i < TS.length; i++) for (const k of ["soft", "hard"] as Kind[]) jobs.push([k, i]);
    const step = () => {
      if (gen !== sweepGen) return;
      const t0 = performance.now();
      while (jobs.length && performance.now() - t0 < 20) {
        const [k, i] = jobs.shift()!;
        sweep[k][i] = quality(denoise(TS[i] * sigHat, k));
      }
      drawSweep();
      if (jobs.length) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
    run();
  }

  function denoise(t: number, kind: Kind): F64 {
    return one ? denoise1(noisy, w, st.levels, t, kind, st.spin) : denoise2(noisy, N, N, w, st.levels, t, kind, st.spin).out;
  }
  function quality(est: ArrayLike<number>): number {
    if (one) { let s = 0; for (let i = 0; i < N1; i++) s += (est[i] - truth[i]) ** 2; return Math.sqrt(s / N1); }
    return psnr(truth, est, 255);
  }

  let pending = false;
  function run(): void {
    sync();
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; render(); });
  }

  let tMark: ReturnType<Plot["vline"]> | null = null;
  function drawSweep(): void {
    pT.clear();
    const vals = [...sweep.hard, ...sweep.soft, quality(noisy)].filter(Number.isFinite);
    const lo = Math.min(...vals), hi = Math.max(...vals), pad = Math.max(0.08 * (hi - lo), one ? 0.02 : 0.5);
    pT.setDomain([0, 6], [lo - pad, hi + pad]);
    const xs = Float64Array.from(TS);
    pT.line(xs, sweep.soft, { color: COL.soft, label: "soft" });
    pT.line(xs, sweep.hard, { color: COL.hard, label: "hard" });
    pT.vline(UNIV, { color: "var(--qb-muted)", width: 2, text: "univ." });
    pT.legend(undefined, one ? "tr" : "br");
    tMark = pT.vline(st.t, { color: "var(--qb-fg)", width: 2.5, dash: "" });
  }

  function render(): void {
    const t = st.t * sigHat;
    const est = denoise(t, st.kind);
    const qIn = quality(noisy), qOut = quality(est);
    // kept fraction of the details (unshifted)
    let kept = 0, nd = 0;
    if (one) {
      const c = dwt(noisy, w, st.levels);
      for (let i = N1 >> st.levels; i < N1; i++) { nd++; if (Math.abs(c[i]) > t) kept++; }
      pSig!.clear();
      const lo = Math.min(...noisy), hi = Math.max(...noisy);
      pSig!.setDomain([0, N1], [lo - 1, hi + 1]);
      const xs = Float64Array.from({ length: N1 }, (_, i) => i);
      pSig!.scatter(xs, noisy, { color: "var(--qb-muted)", r: 1.6, stroke: "none", opacity: 0.6 });
      pSig!.line(xs, truth, { color: "var(--qb-fg)", width: 1.5, label: "truth" });
      pSig!.line(xs, est, { color: COL.soft, width: 2.5, label: "denoised" });
      pSig!.legend(undefined, "tr");
    } else {
      const c = dwt2(noisy, N, N, w, st.levels), a = N >> st.levels;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (x >= a || y >= a) { nd++; if (Math.abs(c[y * N + x]) > t) kept++; }
      drawImage(F[0].cv, noisy, N); drawImage(F[1].cv, est, N);
      drawSigned(F[2].cv, Float64Array.from(est, (v, i) => noisy[i] - v), N, 3 * st.sigma);
      drawSigned(F[3].cv, Float64Array.from(est, (v, i) => v - truth[i]), N, 3 * st.sigma);
      F[0].c.textContent = `noisy: ${qIn.toFixed(1)} dB`;
      F[1].c.textContent = `denoised: ${qOut.toFixed(1)} dB`;
      F[2].c.textContent = `removed (±${3 * st.sigma})`;
      F[3].c.textContent = `error vs truth (±${3 * st.sigma})`;
    }
    tMark?.remove();
    tMark = pT.vline(st.t, { color: "var(--qb-fg)", width: 2.5, dash: "" });
    // histogram of the finest D band with ±t
    const R = Math.max(5 * sigHat, 1.3 * t, 1e-9), nb = 61, edges = Float64Array.from({ length: nb + 1 }, (_, i) => -R + 2 * R * i / nb);
    const counts = new Float64Array(nb);
    for (const v of finestD) { const b = Math.floor((v + R) / (2 * R) * nb); if (b >= 0 && b < nb) counts[b]++; }
    pH.clear();
    pH.setDomain([-R, R], [0, Math.max(...counts) * 1.1]);
    pH.histogram(edges, counts, { color: seriesColor(4) });
    pH.vline(-t, { color: "var(--qb-fg)", width: 2 }); pH.vline(t, { color: "var(--qb-fg)", width: 2, text: "±t" });
    const unit = one ? "" : " dB";
    ro.textContent = [
      `σ_n true ${st.sigma.toFixed(2)}   σ̂_n ${sigHat.toFixed(2)}`,
      `t = ${st.t.toFixed(2)} σ̂ = ${t.toFixed(2)}`,
      `details kept ${(100 * kept / nd).toFixed(1)} %`,
      one ? `RMSE noisy ${qIn.toFixed(3)} → ${qOut.toFixed(3)}` : `PSNR noisy ${qIn.toFixed(1)}${unit} → ${qOut.toFixed(1)}${unit}`,
      `${w.label}, ${st.levels} levels, ${st.kind}, ${st.spin} shift${st.spin > 1 ? "s" : ""}`,
    ].join("\n");
  }

  sync();
  newNoisy();
  if (p0) presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0));
}

autoMount(NAME, mount);
