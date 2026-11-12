// Least-squares unmixing of a lambda stack (main demo of the unmixing day). A synthetic
// cell (seeded, 128 × 128) with four components: nucleus, mitochondria, membrane and a
// diffuse autofluorescence with a broad spectrum, rendered into an N-channel stack
// (contiguous bins 480–700 nm) with Poisson noise. The unmixing: one channel per dye (each
// dye read in its brightest channel), least squares (one pseudo-inverse for the whole
// image), weighted least squares (weights 1/expected count), or NNLS (with data-methods).
// Panels: true and unmixed pseudocolour merges, the residual (in units of the expected
// Poisson noise), the abundance maps (unmixed or true), the spectra with the channel bins,
// and one pixel: measured spectrum, fitted mixture and the stacked contributions (click a
// pixel). Readouts: κ(M), RMSE per component against the truth, share of negative values.
// Auto-mounts on <div data-demo="lsunmix">. Options: data-preset (one|ls|close|more|forgot|
// add|starved), data-methods (e.g. "one ls wls nnls"; default "one ls wls"), data-channels,
// data-seed.
import { htmlEl, slider, presets, checkbox, button, autoMount } from "../lib/ui";
import { Plot } from "../lib/plot";
import { RGB, inferno } from "../lib/colour";
import * as U from "../lib/unmix";
import { injectStyle, uniqueClip, canvasFigure, drawMerge, drawMap, canvasPixel, plane, fmt, COMMON_CSS } from "./common";
import { L0, L1, N_PIX, P, Method, dyesFor, truthMaps, simulate as simStack, unmix as unmixStack, residualChi, rmse as rmseOf, negativeShare } from "./model";

const NAME = "lsunmix";
const CSS = COMMON_CSS + `
[data-demo="${NAME}"] { gap: .6em; align-items: flex-start; }
[data-demo="${NAME}"] .ls-imgs { flex: 1 1 470px; min-width: 300px; max-width: 560px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .ls-row { display: flex; gap: .4em; }
[data-demo="${NAME}"] .ls-row .um-fig { flex: 1 1 0; min-width: 0; }
[data-demo="${NAME}"] .ls-row .um-fig canvas { cursor: crosshair; }
[data-demo="${NAME}"] .ls-plots { flex: 1 1 320px; min-width: 260px; max-width: 400px; display: flex; flex-direction: column; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 340px; gap: .3em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; flex-wrap: nowrap; }
.reveal [data-demo="${NAME}"] .ls-imgs { flex: 0 0 450px; }
.reveal [data-demo="${NAME}"] .ls-plots { flex: 0 0 330px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 0 0 330px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .08em .35em; }
.reveal [data-demo="${NAME}"] .um-fig figcaption { font-size: 16px; }
.reveal [data-demo="${NAME}"] .um-readout { font-size: 15px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .15em; }
.reveal [data-demo="${NAME}"] .qb-slider { font-size: 17px; }
`;

const COMP = ["nucleus", "mitochondria", "membrane", "autofluor."];
const COL: RGB[] = [[0.25, 0.45, 1], [0.15, 0.95, 0.25], [1, 0.2, 0.8], [0.95, 0.65, 0.1]];
const CSSCOL = ["#3f72ff", "#1fa83a", "#e03cc8", "#d99a12"];
const METHOD_LABEL: Record<Method, string> = { one: "one channel per dye", ls: "least squares", wls: "weighted LS", nnls: "NNLS (≥ 0)" };
interface St { n: number; sep: number; logPhot: number; auto: boolean; method: Method; truth: boolean }
const PRESETS: Record<string, { label: string; s: Partial<St> }> = {
  one: { label: "one channel per dye: bleed-through", s: { n: 16, sep: 45, logPhot: Math.log10(400), auto: true, method: "one" } },
  ls: { label: "least squares removes it", s: { n: 16, sep: 45, logPhot: Math.log10(400), auto: true, method: "ls" } },
  close: { label: "two dyes 6 nm apart: noise explodes", s: { n: 16, sep: 6, logPhot: Math.log10(400), auto: true, method: "ls" } },
  more: { label: "32 channels: no rescue", s: { n: 32, sep: 6, logPhot: Math.log10(400), auto: true, method: "ls" } },
  forgot: { label: "forgot the autofluorescence: the residual lights up", s: { n: 16, sep: 45, logPhot: Math.log10(400), auto: false, method: "ls" } },
  add: { label: "add it as a component", s: { n: 16, sep: 45, logPhot: Math.log10(400), auto: true, method: "ls" } },
  starved: { label: "photon-starved: WLS vs LS", s: { n: 16, sep: 45, logPhot: Math.log10(20), auto: true, method: "wls" } },
};

function mount(host: HTMLElement): void {
  injectStyle(NAME, CSS);
  const methods = (host.dataset.methods ?? "one ls wls").split(/[\s,]+/).filter(m => m in METHOD_LABEL) as Method[];
  const seed = Number(host.dataset.seed ?? 1) || 1;
  const st: St = { n: Number(host.dataset.channels ?? 16) || 16, sep: 45, logPhot: Math.log10(400), auto: true, method: "ls", truth: false };
  let pix = 60 * N_PIX + 55;  // a nucleus pixel
  const A = truthMaps();

  // ---- layout ----
  const imgs = htmlEl("div", "ls-imgs", "", host);
  const row1 = htmlEl("div", "ls-row", "", imgs), row2 = htmlEl("div", "ls-row", "", imgs);
  const cTrue = canvasFigure(row1, "true", N_PIX, N_PIX), cUnm = canvasFigure(row1, "unmixed", N_PIX, N_PIX), cRes = canvasFigure(row1, "residual", N_PIX, N_PIX);
  const cAb = COMP.map((c, k) => { const f = canvasFigure(row2, c, N_PIX, N_PIX); f.cap.style.color = CSSCOL[k]; return f; });
  const plots = htmlEl("div", "ls-plots", "", host);
  const F = host.closest(".reveal") ? 21 : 17;
  const pSpec = new Plot(plots, { width: 400, height: 230, x: [L0 - 10, L1 + 10], y: [0, 1.05], xlabel: "wavelength (nm)", ylabel: "spectrum", xticks: [500, 600, 700], yticks: [0, 1], font: F, margin: { l: 74 } });
  const pPix = new Plot(plots, { width: 400, height: 230, x: [L0 - 10, L1 + 10], y: [0, 1], xlabel: "wavelength (nm)", ylabel: "pixel", xticks: [500, 600, 700], yticks: 3, font: F, margin: { l: 74 } });
  uniqueClip(pSpec.svg, NAME); uniqueClip(pPix.svg, NAME);
  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtn = presets(panel, [
    ...Object.values(PRESETS).map(p => ({ label: p.label, apply: () => apply(p.s) })),
    { label: "reset", apply: () => apply(PRESETS.ls.s) },
  ]);
  const mRow = htmlEl("div", "um-row", "", panel);
  const mBtns = methods.map(m => { const b = button(mRow, METHOD_LABEL[m], () => { st.method = m; manual(true); }); b.dataset.m = m; return b; });
  const nS = slider(panel, "channels", 3, 32, 1, st.n, v => { st.n = v; manual(false); });
  const sepS = slider(panel, "peak 1 → 2 (nm)", 4, 60, 1, st.sep, v => { st.sep = v; manual(false); });
  const phS = slider(panel, "photons", 1, 3.7, 0.05, st.logPhot, v => { st.logPhot = v; manual(false); }, v => String(Math.round(10 ** v)));
  const autoC = checkbox(panel, "autofluorescence in M", st.auto, v => { st.auto = v; manual(true); });
  const truthC = checkbox(panel, "show true abundances", st.truth, v => { st.truth = v; drawAll(); });
  const read = htmlEl("div", "qb-readout um-readout", "", plots);
  const note = htmlEl("div", "qb-note", "Click a pixel to see its spectrum and the fit. Photons = expected count of a pixel with abundance 1, over all channels.", panel);
  void note;

  // ---- computation (cached) ----
  let key = "", Y: Float32Array = new Float32Array(0), M: Float64Array = new Float64Array(0), bands: U.Band[] = [];
  let Ah: Float32Array = new Float32Array(0), R: Float32Array = new Float32Array(0), Mm: Float64Array = new Float64Array(0), K = 4;
  function simulate(): void {
    const k2 = `${st.n}|${st.sep}|${st.logPhot}|${seed}`;
    if (k2 === key) return;
    key = k2;
    ({ bands, M, Y } = simStack(A, st.n, st.sep, 10 ** st.logPhot, seed));
  }
  function unmix(): void {
    K = st.auto ? 4 : 3;
    ({ Ah, Mm, R } = unmixStack(Y, M, st.n, K, st.method, 10 ** st.logPhot));
  }

  // ---- drawing ----
  function drawAb(c: HTMLCanvasElement, v: ArrayLike<number>, col: RGB): void {
    const ctx = c.getContext("2d")!, id = ctx.createImageData(N_PIX, N_PIX), d = id.data;
    for (let p = 0; p < P; p++) {
      const x = v[p];
      if (!Number.isFinite(x)) { d[4 * p + 3] = 255; continue; }
      if (x >= 0) { const t = Math.min(1, x); d[4 * p] = 255 * t * (0.35 + 0.65 * col[0]); d[4 * p + 1] = 255 * t * (0.35 + 0.65 * col[1]); d[4 * p + 2] = 255 * t * (0.35 + 0.65 * col[2]); }
      else { const t = Math.min(1, -x * 2); d[4 * p] = 90 + 165 * t; d[4 * p + 1] = 0; d[4 * p + 2] = 0; }
      d[4 * p + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
  }
  const specMarks: ReturnType<Plot["line"]>[] = [];
  function drawAll(): void {
    const n = st.n, ph = 10 ** st.logPhot;
    const tm = [0, 1, 2, 3].map(k => plane(A, k, P)), um = [0, 1, 2, 3].map(k => plane(Ah, k, P));
    drawMerge(cTrue.canvas, tm, COL, [0.9, 1, 0.8, 0.9], N_PIX, N_PIX);
    drawMerge(cUnm.canvas, um.slice(0, K), COL, [0.9, 1, 0.8, 0.9], N_PIX, N_PIX);
    // residual in units of the expected Poisson noise: sqrt(Σ_c r_c² / max(y_c, 1)) / sqrt(n)
    const chi = residualChi(R, Y, n);
    drawMap(cRes.canvas, chi, N_PIX, N_PIX, inferno, 0, 2.5);
    for (let k = 0; k < 4; k++) {
      if (st.truth) drawAb(cAb[k].canvas, tm[k], COL[k]);
      else if (k < K) drawAb(cAb[k].canvas, um[k], COL[k]);
      else drawAb(cAb[k].canvas, new Float32Array(P), COL[k]);
      cAb[k].cap.textContent = !st.truth && k >= K ? "not in M" : COMP[k];
    }
    // spectra plot: bins and the four normalised columns
    pSpec.clear(); specMarks.length = 0;
    const dy = dyesFor(st.sep);
    pSpec.mark((g: SVGGElement) => {
      bands.forEach((b, i) => {
        const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        r.setAttribute("x", String(pSpec.x(b.lo))); r.setAttribute("width", String(pSpec.x(b.hi) - pSpec.x(b.lo)));
        r.setAttribute("y", String(pSpec.box.t)); r.setAttribute("height", String(pSpec.box.b - pSpec.box.t));
        r.setAttribute("fill", i % 2 ? "#999" : "#ccc"); r.setAttribute("fill-opacity", "0.18"); g.appendChild(r);
      });
    }, []);
    dy.forEach((d, k) => {
      const xs: number[] = [], ys: number[] = [];
      for (let l = L0 - 10; l <= L1 + 10; l += 2) { xs.push(l); ys.push(U.emission(d, l)); }
      pSpec.line(xs, ys, { color: CSSCOL[k], width: 3, dash: k === 3 && !st.auto ? "6 4" : undefined });
    });
    drawPixel();
    // readouts
    const lines: string[] = [];
    if (n < K) lines.push(`${n} channels < ${K} components:\nno unique solution`);
    else {
      lines.push(`κ(M) = ${fmt(U.condition(Mm, n, K), 1)}   (${n} channels, ${K} components)`);
      const g = U.noiseGain(Mm, n, K), amp = Array.from(g, (v, k) => { let s2 = 0; for (let c = 0; c < n; c++) s2 += Mm[c * K + k] ** 2; return v * Math.sqrt(s2); });
      lines.push("noise × (vs the dye alone): " + amp.map(v => fmt(v, 1)).join(", "));
      lines.push("RMSE vs truth (abundance 0..1):");
      const rm = COMP.slice(0, K).map((c, k) => `${c.slice(0, 4)} ${fmt(rmse(k), 3)}`);
      lines.push("  " + rm.slice(0, 2).join("   "));
      lines.push("  " + rm.slice(2).join("   "));
      lines.push(`negative abundances: ${fmt(100 * negativeShare(Ah, K), 1)} %`);
      let chiS = 0;
      for (let p = 0; p < P; p++) chiS += chi[p];
      lines.push(`mean residual: ${fmt(chiS / P, 2)} × noise`);
    }
    read.textContent = lines.join("\n");
    mBtns.forEach(b => b.classList.toggle("active", b.dataset.m === st.method));
    void ph;
  }
  const rmse = (k: number) => rmseOf(Ah, A, k);
  function drawPixel(): void {
    const n = st.n, ph = 10 ** st.logPhot;
    pPix.clear();
    const cs = bands.map(b => (b.lo + b.hi) / 2), y = Array.from({ length: n }, (_, c) => Y[c * P + pix] / ph);
    const fit = new Float64Array(n), stack = new Float64Array(n);
    let ymax = Math.max(...y, 0.01);
    if (n >= K && Number.isFinite(Ah[pix])) {
      for (let k = 0; k < K; k++) {
        const lo = Float64Array.from(stack);
        for (let c = 0; c < n; c++) stack[c] += Mm[c * K + k] * Ah[k * P + pix];
        pPix.band(cs, Array.from(lo), Array.from(stack), { color: CSSCOL[k], fill: CSSCOL[k], fillOpacity: 0.35 });
      }
      fit.set(stack);
      ymax = Math.max(ymax, ...fit);
      pPix.line(cs, Array.from(fit), { color: "#222", width: 2.5 });
    }
    pPix.scatter(cs, y, { color: "#000", r: 4 });
    pPix.setDomain(undefined, [Math.min(0, ...stack) * 1.1, ymax * 1.15]);
    pPix.setLabels({ ylabel: `pixel (${pix % N_PIX}, ${Math.floor(pix / N_PIX)})` });
    for (const c of [cTrue, cUnm, cRes, ...cAb]) {
      const ctx = c.canvas.getContext("2d")!, x = pix % N_PIX, yy = Math.floor(pix / N_PIX);
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x - 6, yy + 0.5); ctx.lineTo(x - 2, yy + 0.5); ctx.moveTo(x + 3, yy + 0.5); ctx.lineTo(x + 7, yy + 0.5);
      ctx.moveTo(x + 0.5, yy - 6); ctx.lineTo(x + 0.5, yy - 2); ctx.moveTo(x + 0.5, yy + 3); ctx.lineTo(x + 0.5, yy + 7); ctx.stroke();
    }
  }
  for (const c of [cTrue, cUnm, cRes, ...cAb]) c.canvas.addEventListener("pointerdown", ev => { const p = canvasPixel(c.canvas, ev, N_PIX, N_PIX); if (p >= 0) { pix = p; drawAll(); } });

  function manual(onlyUnmix: boolean): void {
    pbtn.forEach(b => b.classList.remove("active"));
    if (!onlyUnmix) simulate();
    unmix(); drawAll();
  }
  function apply(s: Partial<St>): void {
    Object.assign(st, s);
    nS.set(st.n); sepS.set(st.sep); phS.set(st.logPhot); autoC.checked = st.auto; truthC.checked = st.truth;
    simulate(); unmix(); drawAll();
  }
  const pre = Object.keys(PRESETS).indexOf(host.dataset.preset ?? "ls");
  apply(PRESETS[Object.keys(PRESETS)[Math.max(0, pre)]].s);
  pbtn[Math.max(0, pre)].classList.add("active");
}

autoMount(NAME, mount);
