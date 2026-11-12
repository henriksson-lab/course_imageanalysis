// Linear deconvolution lab: blur a test object with a PSF, add noise, and restore it with
// the inverse filter, a truncated inverse, a Wiener filter or Tikhonov regularisation.
// Panels: truth, observed, restored, error vs truth; a radial plot of |H|, the filter gain
// |W|, the effective transfer |WH| and the noise floor; the error for ~30 values of the
// regularisation parameter (computed in the background), with the current value marked.
// Options: the PSF used for restoration can be wider/narrower than the true one; periodic
// or padded-and-tapered borders; a "realistic" forward model (scene continues past the
// border, blur on a finer grid) instead of the inverse crime.
// Auto-mounts on <div data-demo="deconvlab">. Options:
//   data-mode="full" (default) | "inverse" (image formation day: naive inverse only, with
//     1/|H| in the plot and an invisible-grating toggle);
//   data-preset="<id>" (see PRESETS / INV_PRESETS); data-presets="id,id,..." limits the buttons.
import { htmlEl, presets as presetRow, button, checkbox, autoMount, slider } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { diverging } from "../lib/colour";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { psnr, rmse } from "../lib/metrics";
import { radialAverage } from "../lib/psf";
import * as P from "../lib/psf";
import { LinearMethod, GainContext, gainContext, gainFactor, restoreSpectrum, spectrum2d, padTaper, Spectrum } from "../lib/deconv";
import { SceneId, SceneFn, phantom, star, beads, micrograph, PsfSpec, PsfKind, NoiseSpec, makePSF, simulate, addNoise, noiseSD, blur, render } from "./scenes";
import { injectStyle, baseStyle, select, logSlider, imagePanel, Jobs, fmtSci, legendRow } from "./widgets";

const NAME = "deconvlab";
const N = 128;

type MethodKind = LinearMethod["kind"];
interface State {
  scene: SceneId; psf: PsfSpec; used: number; noise: NoiseSpec;
  method: MethodKind; eps: number; snr: number; model: "flat" | "1/f2"; lambda: number; penalty: "energy" | "roughness";
  border: "periodic" | "padded"; realistic: boolean; grating: boolean;
}
const DEFAULT: State = {
  scene: "phantom", psf: { kind: "gauss", width: 2 }, used: 1, noise: { kind: "gauss", level: 0.01 },
  method: "wiener", eps: 0.05, snr: 1000, model: "flat", lambda: 1e-3, penalty: "energy",
  border: "periodic", realistic: false, grating: false,
};
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "explode", label: "0.1% noise: inverse explodes", s: { scene: "phantom", psf: { kind: "gauss", width: 2 }, noise: { kind: "gauss", level: 0.001 }, method: "inverse" } },
  { id: "truncated", label: "truncated inverse: ringing", s: { scene: "beads", psf: { kind: "airy", width: 1.0 }, method: "truncated", eps: 0.1 } },
  { id: "wiener", label: "Wiener at the right SNR", s: { scene: "phantom", method: "wiener", model: "flat", snr: 45 } },
  { id: "wrongsnr", label: "Wiener with the wrong SNR", s: { scene: "phantom", method: "wiener", model: "flat", snr: 2000 } },
  { id: "f2", label: "1/f² beats flat", s: { scene: "star", psf: { kind: "motion", width: 9, angle: 0 }, method: "wiener", model: "1/f2", snr: 300 } },
  { id: "toomuch", label: "too much regularisation: blur", s: { scene: "phantom", method: "tikhonov", penalty: "energy", lambda: 0.1 } },
  { id: "tikwiener", label: "Tikhonov = Wiener with a flat spectrum", s: { scene: "phantom", method: "tikhonov", penalty: "energy", lambda: 1 / 45 } },
  { id: "rough", label: "roughness penalty", s: { scene: "phantom", method: "tikhonov", penalty: "roughness", lambda: 0.033 } },
  { id: "wrongpsf", label: "wrong PSF width", s: { scene: "beads", used: 1.3, method: "wiener", model: "flat", snr: 85 } },
  { id: "border", label: "periodic borders ring", s: { scene: "beads", noise: { kind: "gauss", level: 0.001 }, realistic: true, method: "wiener", snr: 1e5, border: "periodic" } },
  { id: "motion", label: "motion blur: Wiener bridges the zeros", s: { scene: "star", psf: { kind: "motion", width: 9, angle: 0 }, method: "wiener", model: "flat", snr: 160 } },
];
const INV_PRESETS: Preset[] = [
  { id: "perfect", label: "no noise: inverse is perfect", s: { scene: "phantom", psf: { kind: "gauss", width: 1.5 }, noise: { kind: "gauss", level: 0 } } },
  { id: "explode", label: "0.1% noise: inverse explodes", s: { scene: "phantom", psf: { kind: "gauss", width: 1.5 }, noise: { kind: "gauss", level: 0.001 } } },
  { id: "motion", label: "motion blur: zeros", s: { scene: "star", psf: { kind: "motion", width: 9, angle: 0 }, noise: { kind: "gauss", level: 0.01 } } },
  { id: "grating", label: "an invisible grating", s: { scene: "phantom", psf: { kind: "airy", width: 1.0 }, noise: { kind: "gauss", level: 0 }, grating: true } },
];

const PSF_RANGE: Record<PsfKind, [number, number, number, string]> = {
  gauss: [0.5, 4, 0.1, "σ (px)"], airy: [0.4, 1.4, 0.05, "NA"], motion: [2, 20, 1, "length (px)"],
};
const SWEEP: Record<Exclude<MethodKind, "inverse">, [number, number]> = { truncated: [1e-4, 1], wiener: [1, 1e8], tikhonov: [1e-8, 10] };

const STYLE = baseStyle(NAME) + `
[data-demo="${NAME}"] .dl-imgs { flex: 1 1 360px; max-width: 420px; display: grid; grid-template-columns: 1fr 1fr; gap: .4em .5em; }
[data-demo="${NAME}"] .dl-plots { flex: 1 1 330px; max-width: 420px; display: flex; flex-direction: column; gap: .3em; }
.reveal [data-demo="${NAME}"] .dl-imgs { flex: 0 1 330px; }
.reveal [data-demo="${NAME}"] .dl-plots { flex: 0 1 400px; max-width: 400px; }
`;

/** Noise slider position u (log10 of σn in units of 0.01 %) → σn; the left end is 0. */
const lev = (u: number) => (u < -0.25 ? 0 : 10 ** u / 1e4);

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const inverseOnly = host.dataset.mode === "inverse";
  const allPresets = inverseOnly ? INV_PRESETS : PRESETS;
  const only = host.dataset.presets?.split(",").map(s => s.trim());
  const shown = only ? allPresets.filter(p => only.includes(p.id)) : allPresets;
  const p0 = allPresets.find(p => p.id === host.dataset.preset) ?? (inverseOnly ? INV_PRESETS[1] : PRESETS[2]);
  const st: State = { ...DEFAULT, psf: { ...DEFAULT.psf }, noise: { ...DEFAULT.noise } };
  if (inverseOnly) st.method = "inverse";

  // ---- layout ----
  const imgs = htmlEl("div", "dl-imgs", "", host);
  const pTruth = imagePanel(imgs, "truth"), pObs = imagePanel(imgs, "observed (blur + noise)");
  const pRest = imagePanel(imgs, "restored"), pErr = imagePanel(imgs, "error: restored − truth");
  const credit = htmlEl("div", "dc-credit", CELL_CREDIT, imgs);
  credit.style.gridColumn = "1 / 3";
  const plots = htmlEl("div", "dl-plots", "", host);
  const gainPlot = new Plot(plots, { width: 420, height: 270, x: [0, 0.5], y: [1e-3, 1e4], ylog: true, xlabel: "frequency (cycles/pixel)", ylabel: "gain", font: 19, margin: { l: 84 } });
  const lH = gainPlot.line([], [], { color: seriesColor(0), label: "|H| optics" });
  const lW = gainPlot.line([], [], { color: seriesColor(1), label: inverseOnly ? "1/|H| inverse" : "|W| filter" });
  const lWH = gainPlot.line([], [], { color: seriesColor(2), label: "|WH| optics + filter" });
  const lFloor = gainPlot.line([], [], { color: "var(--qb-muted, #777)", dash: "6 5", label: "noise floor" });
  if (inverseOnly) lWH.remove();
  legendRow(plots, [
    { label: "|H| optics", color: seriesColor(0) },
    { label: inverseOnly ? "1/|H| inverse" : "|W| filter", color: seriesColor(1) },
    ...(inverseOnly ? [] : [{ label: "|WH| both", color: seriesColor(2) }]),
    { label: "noise floor", color: "var(--qb-muted, #777)", dash: true },
  ]);
  let sweepPlot: Plot | null = null;
  if (!inverseOnly) sweepPlot = new Plot(plots, { width: 420, height: 230, x: [1, 10], y: [0, 1], xlog: true, xlabel: "λ", ylabel: "RMSE vs truth", font: 19, margin: { l: 84 } });
  const sweepLine = sweepPlot?.line([], [], { color: seriesColor(3) });
  const sweepNow = sweepPlot?.vline(1, { color: seriesColor(1), width: 2.5 });
  const sweepBest = sweepPlot?.scatter([], [], { color: seriesColor(2), r: 7 });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "dc-lab", "presets", panel);
  const pbtns = presetRow(panel, shown.map(p => ({ label: p.label, apply: () => load(p) })));
  const sceneS = select<SceneId>(panel, "object", [["phantom", "cell phantom"], ["star", "Siemens star"], ["beads", "beads + edge"], ["cell", "micrograph"]], st.scene, v => { st.scene = v; resim(); });
  const psfS = select<PsfKind>(panel, "PSF", [["gauss", "Gaussian"], ["airy", "Airy (λ 520 nm, 65 nm px)"], ["motion", "motion (horizontal)"]], st.psf.kind, v => {
    st.psf = { kind: v, width: v === "gauss" ? 2 : v === "airy" ? 1.0 : 9, angle: 0 }; syncPsfSlider(); resim();
  });
  const psfW = slider(panel, "σ (px)", 0.5, 4, 0.1, st.psf.width, v => { st.psf.width = v; resim(); });
  const usedS = slider(panel, "PSF used ×", 0.6, 1.6, 0.05, st.used, v => { st.used = v; recontext(); }, v => v.toFixed(2));
  if (inverseOnly) usedS.input.closest("label")!.style.display = "none";
  const noiseK = select<"gauss" | "photons">(panel, "noise", [["gauss", "Gaussian σn"], ["photons", "photon counts"]], st.noise.kind, v => {
    st.noise = v === "gauss" ? { kind: v, level: 0.01 } : { kind: v, level: 1000 }; syncNoiseSlider(); renoise();
  });
  const noiseG = slider(panel, "σn (%)", -0.5, 3, 0.01, 2, u => { st.noise.level = lev(u); renoise(); }, u => fmtSci(100 * lev(u)));
  const noiseP = logSlider(panel, "photons at 1", 10, 1e5, 1000, v => { st.noise.level = Math.round(v); renoise(); }, v => fmtSci(Math.round(v)));
  let methodS: ReturnType<typeof select<MethodKind>> | null = null;
  let modelS: ReturnType<typeof select<"flat" | "1/f2">> | null = null;
  let penS: ReturnType<typeof select<"energy" | "roughness">> | null = null;
  let epsS: ReturnType<typeof logSlider> | null = null, snrS: ReturnType<typeof logSlider> | null = null, lamS: ReturnType<typeof logSlider> | null = null;
  let borderS: ReturnType<typeof select<"periodic" | "padded">> | null = null, realC: HTMLInputElement | null = null, gratC: HTMLInputElement | null = null;
  if (!inverseOnly) {
    methodS = select<MethodKind>(panel, "method", [["inverse", "inverse 1/H"], ["truncated", "truncated inverse"], ["wiener", "Wiener"], ["tikhonov", "Tikhonov"]], st.method, v => { st.method = v; syncMethod(); restore(); sweep(); });
    epsS = logSlider(panel, "ε", 1e-4, 1, st.eps, v => { st.eps = v; restore(); });
    snrS = logSlider(panel, "SNR", 1, 1e8, st.snr, v => { st.snr = v; restore(); });
    modelS = select<"flat" | "1/f2">(panel, "signal model", [["flat", "flat spectrum"], ["1/f2", "1/f² spectrum"]], st.model, v => { st.model = v; restore(); sweep(); });
    lamS = logSlider(panel, "λ", 1e-8, 10, st.lambda, v => { st.lambda = v; restore(); });
    penS = select<"energy" | "roughness">(panel, "penalty", [["energy", "energy ‖x‖²"], ["roughness", "roughness ‖Lx‖²"]], st.penalty, v => { st.penalty = v; restore(); sweep(); });
    borderS = select<"periodic" | "padded">(panel, "borders", [["periodic", "periodic (wrap)"], ["padded", "padded + tapered"]], st.border, v => { st.border = v; recontext(); });
    realC = checkbox(panel, "realistic blur (scene beyond the border, finer grid)", st.realistic, v => { st.realistic = v; resim(); });
  } else {
    gratC = checkbox(panel, "add a grating beyond the cut-off", st.grating, v => { st.grating = v; renoise(); });
  }
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  const note = htmlEl("div", "qb-note", "", panel);

  function syncPsfSlider(): void {
    const [lo, hi, step, lab] = PSF_RANGE[st.psf.kind];
    Object.assign(psfW.input, { min: String(lo), max: String(hi), step: String(step) });
    (psfW.input.parentElement!.querySelector(".qb-slider-name") as HTMLElement).textContent = lab;
    psfW.set(st.psf.width);
  }
  function syncNoiseSlider(): void {
    const g = st.noise.kind === "gauss";
    noiseG.input.closest("label")!.style.display = g ? "" : "none";
    noiseP.input.closest("label")!.style.display = g ? "none" : "";
    if (g) noiseG.set(st.noise.level <= 0 ? -0.5 : Math.log10(st.noise.level * 1e4)); else noiseP.setValue(st.noise.level);
  }
  function syncMethod(): void {
    if (!methodS) return;
    const m = st.method, show = (e: { input?: HTMLInputElement; row?: HTMLElement } | null, on: boolean) => {
      if (!e) return;
      const el = e.row ?? e.input!.closest("label")!;
      el.style.display = on ? "" : "none";
    };
    show(epsS, m === "truncated"); show(snrS, m === "wiener"); show(modelS, m === "wiener"); show(lamS, m === "tikhonov"); show(penS, m === "tikhonov");
    sweepPlot?.setLabels({ xlabel: m === "truncated" ? "ε" : m === "wiener" ? "SNR used by the filter" : "λ" });
    if (sweepPlot) sweepPlot.svg.style.visibility = m === "inverse" ? "hidden" : "visible";
  }
  function syncAll(): void {
    sceneS.set(st.scene); psfS.set(st.psf.kind); syncPsfSlider(); usedS.set(st.used); noiseK.set(st.noise.kind); syncNoiseSlider();
    methodS?.set(st.method); epsS?.setValue(st.eps); snrS?.setValue(st.snr); modelS?.set(st.model); lamS?.setValue(st.lambda); penS?.set(st.penalty);
    borderS?.set(st.border); if (realC) realC.checked = st.realistic; if (gratC) gratC.checked = st.grating;
    syncMethod();
  }
  function load(p: Preset): void {
    Object.assign(st, { ...DEFAULT, psf: { ...DEFAULT.psf }, noise: { ...DEFAULT.noise } }, JSON.parse(JSON.stringify(p.s)));
    if (inverseOnly) st.method = "inverse";
    pbtns.forEach((b, i) => b.classList.toggle("active", shown[i] === p));
    syncAll(); resim();
  }

  // ---- pipeline ----
  let cellImg: Float64Array | null = null;
  let truth: Float64Array = new Float64Array(N * N), clean: Float64Array = new Float64Array(N * N), obs: Float64Array = new Float64Array(N * N), shownTruth: Float64Array = truth;
  let hi = 1, sd = 0;
  let ctxTrue: GainContext, ctxUsed: GainContext, ctxRest: GainContext, G: Spectrum, pad = { ox: 0, oy: 0, M: N };
  let floorF: Float64Array = new Float64Array(0), floorV: Float64Array = new Float64Array(0);
  const jobs = new Jobs();

  function sceneFn(): SceneFn | null {
    if (st.scene === "phantom") return phantom(N);
    if (st.scene === "star") return star(N);
    if (st.scene === "beads") return beads(N);
    if (!cellImg) { loadCell().then(g => { cellImg = Float64Array.from(g.data, v => v / 255); resim(); }); return null; }
    return micrograph(cellImg, 256, N);
  }
  let sceneTruthPower: Float64Array = new Float64Array(0);
  function resim(): void {
    const f = sceneFn();
    credit.style.visibility = st.scene === "cell" ? "visible" : "hidden";
    if (!f) return;
    ({ truth, clean } = simulate(f, N, st.psf, st.realistic));
    hi = Math.max(...truth);
    const T = spectrum2d(truth, N, N), pw = new Float64Array(N * N);
    for (let i = 0; i < N * N; i++) pw[i] = T.re[i] ** 2 + T.im[i] ** 2;
    sceneTruthPower = pw;
    ctxTrue = gainContext(spectrum2d(P.toOrigin(makePSF(N, st.psf), N, N), N, N), N, N);
    renoise(false);
    recontext();
  }
  function gratingImage(): Float64Array {
    // a full-field cosine with an integer number of periods, beyond the cut-off (or on a zero of the motion OTF)
    const fx = st.psf.kind === "airy" ? Math.min(0.45, 1.25 * P.cutoff(st.psf.width, 520, 65)) : st.psf.kind === "motion" ? 1 / st.psf.width : 0.4;
    const k = Math.round(fx * N), g = new Float64Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) g[y * N + x] = 0.05 * Math.cos(2 * Math.PI * k * x / N);
    return g;
  }
  let gratingInfo = "";
  function renoise(chain = true): void {
    shownTruth = truth;
    let c = clean;
    gratingInfo = "";
    if (inverseOnly && st.grating) {
      const g = gratingImage(), gb = blur(g, makePSF(N, st.psf), N);
      shownTruth = truth.map((v, i) => v + g[i]);
      c = clean.map((v, i) => v + gb[i]);
      gratingInfo = `grating: max |change of the observed image| = ${fmtSci(Math.max(...gb.map(Math.abs)))}`;
    }
    sd = st.noise.kind === "gauss" ? st.noise.level : noiseSD(st.noise, c);
    obs = st.noise.kind === "gauss" && st.noise.level === 0 ? c.slice() : addNoise(c, st.noise);
    // noise floor: sqrt(noise power / signal power) per frequency ring
    const rp = radialAverage(sceneTruthPower, N, N, 64);
    floorF = rp.f; floorV = rp.v.map(v => Math.sqrt(N * N * sd * sd / v));
    if (chain) recontext();
  }
  function recontext(): void {
    const usedPsf = makePSF(N, st.psf, 1, st.used);
    ctxUsed = gainContext(spectrum2d(P.toOrigin(usedPsf, N, N), N, N), N, N);
    if (st.border === "padded" && !inverseOnly) {
      const M = 2 * N, pt = padTaper(obs, N, N, M, M);
      pad = { ox: pt.ox, oy: pt.oy, M };
      ctxRest = gainContext(spectrum2d(P.toOrigin(makePSF(M, st.psf, 1, st.used), M, M), M, M), M, M);
      G = spectrum2d(pt.data, M, M);
    } else {
      pad = { ox: 0, oy: 0, M: N };
      ctxRest = ctxUsed;
      G = spectrum2d(obs, N, N);
    }
    restore(); sweep();
  }
  function methodOf(param?: number): LinearMethod {
    switch (st.method) {
      case "inverse": return { kind: "inverse" };
      case "truncated": return { kind: "truncated", eps: param ?? st.eps };
      case "wiener": return { kind: "wiener", snr: param ?? st.snr, model: st.model };
      case "tikhonov": return { kind: "tikhonov", lambda: param ?? st.lambda, penalty: st.penalty };
    }
  }
  function restoredWith(m: LinearMethod): Float64Array {
    const x = restoreSpectrum(G, m, ctxRest);
    if (pad.M === N) return x;
    const out = new Float64Array(N * N);
    for (let y = 0; y < N; y++) for (let xx = 0; xx < N; xx++) out[y * N + xx] = x[(y + pad.oy) * pad.M + xx + pad.ox];
    return out;
  }
  const paramNow = () => (st.method === "truncated" ? st.eps : st.method === "wiener" ? st.snr : st.lambda);

  let rest: Float64Array = new Float64Array(N * N);
  function restore(): void {
    const m = methodOf();
    rest = restoredWith(m);
    // radial curves on the N grid
    const nb = 64, hR = new Float64Array(N * N), wR = new Float64Array(N * N), whR = new Float64Array(N * N);
    for (let i = 0; i < N * N; i++) {
      const g = gainFactor(m, ctxUsed, i), Wr = ctxUsed.H.re[i] * g, Wi = -ctxUsed.H.im[i] * g;
      hR[i] = Math.sqrt(ctxTrue.H2[i]);
      wR[i] = inverseOnly ? (hR[i] > 1e-10 ? 1 / hR[i] : Infinity) : Math.hypot(Wr, Wi);
      whR[i] = Math.hypot(Wr * ctxTrue.H.re[i] - Wi * ctxTrue.H.im[i], Wr * ctxTrue.H.im[i] + Wi * ctxTrue.H.re[i]);
    }
    const clampLog = (v: Float64Array) => v.map(x => Math.min(1e4, Math.max(1e-4, x)));
    // isotropic PSFs: average over rings; motion blur: the cut along x (fy = 0), where the zeros are
    const prof = (v: Float64Array) => st.psf.kind === "motion"
      ? { f: Float64Array.from({ length: N / 2 + 1 }, (_, u) => u / N), v: Float64Array.from({ length: N / 2 + 1 }, (_, u) => v[u]) }
      : radialAverage(v, N, N, nb);
    gainPlot.setLabels({ xlabel: st.psf.kind === "motion" ? "frequency along x (cycles/pixel)" : "frequency (cycles/pixel)" });
    const a = prof(hR), b = prof(wR.map(v => Math.min(v, 1e12))), c = prof(whR);
    lH.set(a.f, clampLog(a.v)); lW.set(b.f, clampLog(b.v)); if (!inverseOnly) lWH.set(c.f, clampLog(c.v));
    lFloor.set(floorF, clampLog(floorV));
    sweepNow?.set(paramNow());
    draw();
  }

  let sweepX: number[] = [], sweepY: number[] = [];
  function sweep(): void {
    if (!sweepPlot || st.method === "inverse") { jobs.cancel(); draw(); return; }
    const [lo, hi2] = SWEEP[st.method as Exclude<MethodKind, "inverse">], K = 30;
    sweepX = Array.from({ length: K }, (_, i) => lo * (hi2 / lo) ** (i / (K - 1)));
    sweepY = [];
    const e0 = Math.round(Math.log10(lo)), e1 = Math.round(Math.log10(hi2)), stp = e1 - e0 > 5 ? 2 : 1;
    (sweepPlot.opts as { xticks?: number[] }).xticks = Array.from({ length: Math.floor((e1 - e0) / stp) + 1 }, (_, i) => 10 ** (e0 + i * stp));
    sweepPlot.setDomain([lo, hi2]);
    sweepLine!.set([], []); sweepBest!.set([], []);
    const gen = function* (this: void) {
      for (const p of sweepX) { sweepY.push(rmse(restoredWith(methodOf(p)), truth)); yield; }
    };
    jobs.run(gen(), done => {
      const top = sweepY.length ? Math.min(Math.max(...sweepY.filter(Number.isFinite)) * 1.1, Math.min(...sweepY) * 4) : 1;
      sweepLine!.set(sweepX.slice(0, sweepY.length), sweepY.map(v => Math.min(v, 2 * top)));
      if (sweepY.length) sweepPlot!.setDomain(undefined, [0, top]);
      if (!done) return;
      const ib = sweepY.indexOf(Math.min(...sweepY));
      sweepBest!.set([sweepX[ib]], [sweepY[ib]]);
      draw();
    }, 12, 250);
  }

  function draw(): void {
    pTruth.draw(shownTruth, N, N, { lo: 0, hi });
    pObs.draw(obs, N, N, { lo: 0, hi });
    pRest.draw(rest, N, N, { lo: 0, hi });
    const err = rest.map((v, i) => v - shownTruth[i]);
    pErr.draw(err, N, N, { lo: -0.2 * hi, hi: 0.2 * hi, map: diverging });
    const e = rmse(rest, shownTruth), neg = rest.reduce((a, v) => a + (v < 0 ? 1 : 0), 0) / rest.length;
    const lines = [`RMSE ${e < 1e3 ? e.toFixed(4) : fmtSci(e)} (observed ${rmse(obs, shownTruth).toFixed(4)})   PSNR ${psnr(rest, shownTruth, 1).toFixed(1)} dB (peak 1)`];
    lines.push(`noise σn ${sd.toFixed(4)}   negative pixels ${(100 * neg).toFixed(1)} %`);
    const mean = (a: Float64Array) => a.reduce((x, v) => x + v, 0) / a.length;
    if (!inverseOnly) lines.push(`true SNR ${sd > 0 ? fmtSci(mean(shownTruth.map(v => v * v)) / (sd * sd)) : "∞"} (mean truth² / σn²)   mean: truth ${mean(shownTruth).toFixed(4)}, restored ${mean(rest).toFixed(4)}`);
    if (inverseOnly) {
      let mx = 0;
      for (let i = 0; i < N * N; i++) { const h = Math.sqrt(ctxTrue.H2[i]); if (h > 1e-10) mx = Math.max(mx, 1 / h); else mx = Infinity; }
      lines.push(`largest gain max 1/|H| = ${Number.isFinite(mx) ? fmtSci(mx) : "∞ (H = 0)"}`);
      if (gratingInfo) lines.push(gratingInfo);
    } else if (sweepY.length === sweepX.length && sweepY.length && st.method !== "inverse") {
      const ib = sweepY.indexOf(Math.min(...sweepY));
      lines.push(`best on the sweep: ${st.method === "truncated" ? "ε" : st.method === "wiener" ? "SNR" : "λ"} = ${fmtSci(sweepX[ib])}, RMSE ${sweepY[ib].toFixed(4)}`);
    }
    ro.textContent = lines.join("\n");
    note.textContent = inverseOnly
      ? "Blue: what the optics pass (|H|); red: what the inverse filter multiplies by (1/|H|); dashed: where |H| falls below it, the data at that frequency are mostly noise."
      : "Plot: blue |H| (optics), red |W| (restoration filter), green |WH| (both together), dashed noise floor. Lower plot: error against the truth for 30 values of the parameter; red line = current, green dot = best.";
  }

  syncAll();
  load(p0);
}

autoMount(NAME, mount);
