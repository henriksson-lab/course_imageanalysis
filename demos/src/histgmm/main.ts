// A Gaussian mixture fitted by EM to the grey-level histogram of an image: the minimum-error
// threshold of the thresholding day, generalised to K classes. Left: the image with the label
// of each pixel (most probable component), the posterior of one class as a probability map,
// or the pixels with low likelihood p(I) < τ. Right: the histogram with the fitted weighted
// components, their sum, the Bayes boundaries (where the most probable class changes) and,
// optionally, the multi-level Otsu thresholds; below it BIC for K = 1..6. EM runs on the 256
// histogram bins weighted by their counts, so it takes a few ms whatever the image size.
// Images: the course micrograph (128 × 128) and a synthetic three-class image with known truth
// (noise and blur sliders).
//
// <div data-demo="histgmm" data-preset="k2|k3cell|toomany|equal|noise|small"></div>
import { autoMount, button, checkbox, htmlEl, presets, slider, type Slider } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { QUALITATIVE, hexToRgb, viridis, colormapLUT } from "../lib/colour";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import type { Grey } from "../lib/imageops";
import { rng } from "../lib/random";
import { type Mixture, nParams, bic } from "../lib/gmm";
import {
  N, synthTruth, synthImage, greyCounts, initHist, histStep, histEM, histLogLik, multiOtsu, bayesBoundaries, sortByMean,
  greyTables, densities, type HistInit,
} from "./model";

const NAME = "histgmm";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: 1em; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin-bottom: .2em; }
.qb-demo[data-demo="${NAME}"] .hg-left { flex: 0 1 330px; min-width: 200px; display: flex; flex-direction: column; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .hg-left canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
.qb-demo[data-demo="${NAME}"] .hg-right { flex: 1 1 480px; min-width: 260px; max-width: 640px; display: flex; flex-direction: column; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .hg-plots { display: flex; gap: .4em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .hg-plots > div:first-child { flex: 3 1 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] .hg-plots > div:last-child { flex: 1.3 1 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] .hg-two { display: grid; grid-template-columns: 1fr 1fr; gap: .2em 1em; }
.qb-demo[data-demo="${NAME}"] .hg-lab { font-size: 14px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .hg-credit { font-size: 12px; color: var(--qb-muted); }
.reveal .qb-demo[data-demo="${NAME}"] { flex-wrap: nowrap; font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .hg-left { flex: 0 0 330px; }
.reveal .qb-demo[data-demo="${NAME}"] .hg-right { flex: 1 1 780px; max-width: 790px; gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .hg-plots svg { max-height: 200px; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; padding: .2em .45em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

type Img = "cell" | "synth" | "small";
type View = "labels" | "posterior" | "lowlik" | "image";
interface Preset { id: string; label: string; img: Img; k: number; init: HistInit; equal: boolean; noise: number; blur: number; view: View; otsu: boolean; bicNow?: boolean }
const PRESETS: Preset[] = [
  { id: "k2", label: "K = 2: the minimum-error threshold again", img: "cell", k: 2, init: "quantiles", equal: false, noise: 12, blur: 1, view: "labels", otsu: true },
  { id: "k3cell", label: "K = 3 on the micrograph", img: "cell", k: 3, init: "quantiles", equal: false, noise: 12, blur: 1, view: "labels", otsu: false },
  { id: "toomany", label: "K too large: components split a peak", img: "synth", k: 6, init: "quantiles", equal: false, noise: 12, blur: 1, view: "labels", otsu: false, bicNow: true },
  { id: "equal", label: "equal variances: Otsu-like", img: "synth", k: 3, init: "quantiles", equal: true, noise: 12, blur: 1, view: "labels", otsu: true },
  { id: "noise", label: "noise: speckle in the label image", img: "synth", k: 3, init: "quantiles", equal: false, noise: 22, blur: 1, view: "labels", otsu: false },
  { id: "small", label: "small bright class: weights matter", img: "small", k: 3, init: "quantiles", equal: false, noise: 12, blur: 1, view: "posterior", otsu: false },
];
const COLS = QUALITATIVE.slice(0, 6);
const RGB = COLS.map(h => hexToRgb(h).map(v => 255 * v));
const VIR = colormapLUT(viridis, 256);
const f1 = (v: number) => v.toFixed(1), f2 = (v: number) => v.toFixed(2);

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[1];
  const st = {
    img: p0.img as Img, k: p0.k, init: p0.init as HistInit, equal: p0.equal, noise: p0.noise, blur: p0.blur, view: p0.view as View,
    otsu: p0.otsu, cls: p0.k - 1, tau: -6, seed: 1,
    m: null as Mixture | null, trace: [] as number[], bic: null as number[] | null,
  };
  let cell: Grey | null = null, img: Grey | null = null, counts: Float64Array = new Float64Array(256), truth: Int32Array | null = null;
  let timer = 0;

  // ---- layout ----
  const left = htmlEl("div", "hg-left", "", host);
  const fig = htmlEl("figure", "", "", left);
  const cap = htmlEl("figcaption", "", "", fig);
  const cv = htmlEl("canvas", "", "", fig);
  cv.width = N; cv.height = N;
  const credit = htmlEl("div", "hg-credit", CELL_CREDIT, left);
  const vRow = htmlEl("div", "qb-presets", "", left);
  const viewBtns = (["image", "labels", "posterior", "lowlik"] as View[]).map(v => button(vRow, v === "lowlik" ? "low p(I)" : v === "posterior" ? "posterior" : v, () => { st.view = v; draw(); }));
  const clsRow = htmlEl("div", "qb-presets", "", left);
  const tauS: Slider = slider(left, "τ = 10^", -9, -2, 0.5, st.tau, v => { st.tau = v; draw(); });

  const right = htmlEl("div", "hg-right", "", host);
  const plots = htmlEl("div", "hg-plots", "", right);
  const hDiv = htmlEl("div", "", "", plots), bDiv = htmlEl("div", "", "", plots);
  const hp = new Plot(hDiv, { width: 520, height: 250, x: [0, 255], y: [0, 1], xlabel: "grey value", ylabel: "pixels per grey level", font: 17 });
  const hist = hp.histogram([], [], { color: "#9a9a9a", fillOpacity: 0.45 });
  const comp = [0, 1, 2, 3, 4, 5].map(j => hp.fn(() => NaN, { color: COLS[j], width: 2.5, dash: "7 4" }, 300));
  const sum = hp.fn(() => NaN, { color: "var(--qb-fg, #222)", width: 3 }, 300);
  const gB: ReturnType<Plot["vline"]>[] = [], gO: ReturnType<Plot["vline"]>[] = [];
  for (let i = 0; i < 5; i++) { gB.push(hp.vline(NaN, { color: "var(--qb-fg, #222)", width: 2.5 })); gO.push(hp.vline(NaN, { color: seriesColor(4), width: 2.5, dash: "2 4" })); }
  const bp = new Plot(bDiv, { width: 220, height: 250, x: [0.4, 6.6], y: [0, 100000], xlabel: "K", ylabel: "BIC (thousands)", xticks: [1, 2, 3, 4, 5, 6], font: 17, margin: { l: 72 }, yfmt: v => (v / 1000).toFixed(1) });
  const bStyle = { color: seriesColor(0), barWidth: 0.6, base: 0 };
  const bBars = bp.bars([], [], bStyle);
  const ro = htmlEl("div", "qb-readout", "", right);

  const r1 = htmlEl("div", "qb-presets", "", right);
  button(r1, "step EM", () => { stop(); stepEM(); draw(); });
  const runBtn = button(r1, "run", () => { if (timer) stop(); else run(); });
  button(r1, "restart", () => { stop(); start(); });
  button(r1, "BIC for K = 1..6", () => { computeBic(); draw(); });
  htmlEl("span", "hg-lab", " start:", r1);
  const initBtns = (["quantiles", "kmeans", "random"] as HistInit[]).map(i => button(r1, i === "kmeans" ? "k-means" : i, () => { st.init = i; start(); }));
  const r2 = htmlEl("div", "qb-presets", "", right);
  htmlEl("span", "hg-lab", "image:", r2);
  const imgBtns = (["cell", "synth", "small"] as Img[]).map(i => button(r2, i === "cell" ? "micrograph" : i === "synth" ? "synthetic" : "synthetic, small bright class", () => { st.img = i; makeImage(); }));
  const kS: Slider = slider(right, "K", 1, 6, 1, st.k, v => { st.k = v; st.cls = Math.min(st.cls, v - 1); fit(); });
  const two = htmlEl("div", "hg-two", "", right);
  const nS: Slider = slider(two, "noise", 0, 40, 1, st.noise, v => { st.noise = v; makeImage(); });
  const blS: Slider = slider(two, "blur σ", 0, 3, 0.25, st.blur, v => { st.blur = v; makeImage(); });
  const r3 = htmlEl("div", "qb-presets", "", right);
  const eqCb = checkbox(r3, "equal variances", st.equal, v => { st.equal = v; fit(); });
  const otCb = checkbox(r3, "multi-level Otsu thresholds", st.otsu, v => { st.otsu = v; draw(); });
  htmlEl("div", "qb-note", "Black lines: Bayes boundaries of the fitted mixture. Dotted purple: multi-level Otsu. EM runs on the histogram bins, weighted by their counts.", right);
  const presetBtns = presets(right, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  button(right, "Reset", () => load(p0)).style.alignSelf = "flex-start";

  function load(p: Preset): void {
    stop();
    Object.assign(st, { img: p.img, k: p.k, init: p.init, equal: p.equal, noise: p.noise, blur: p.blur, view: p.view, otsu: p.otsu, cls: p.k - 1, bic: null });
    kS.set(p.k); nS.set(p.noise); blS.set(p.blur); eqCb.checked = p.equal; otCb.checked = p.otsu;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    makeImage(p.bicNow);
  }

  function makeImage(bicNow = false): void {
    st.bic = null;
    if (st.img === "cell") {
      if (!cell) { loadCell(N).then(g => { cell = g; makeImage(bicNow); }); return; }
      img = { width: N, height: N, data: Float32Array.from(cell.data, v => Math.round(v)) };
      truth = null;
    } else {
      truth = synthTruth(st.img === "small");
      img = synthImage(truth, st.noise, st.blur);
    }
    counts = greyCounts(img);
    credit.style.visibility = st.img === "cell" ? "visible" : "hidden";
    fit();
    if (bicNow) { computeBic(); draw(); }
  }

  function start(): void { st.m = initHist(counts, st.k, st.init, rng(st.seed), st.equal); st.trace = [histLogLik(counts, st.m)]; draw(); }
  function fit(): void {
    stop();
    const r = histEM(counts, initHist(counts, st.k, st.init, rng(st.seed), st.equal), st.equal);
    st.m = sortByMean(r.model); st.trace = r.trace;
    draw();
  }
  function stepEM(): void {
    if (!st.m) return;
    const s = histStep(counts, st.m, st.equal);
    st.m = s.model; st.trace.push(histLogLik(counts, st.m));
  }
  function run(): void {
    runBtn.textContent = "pause";
    timer = window.setInterval(() => {
      const L0 = st.trace[st.trace.length - 1];
      stepEM(); draw();
      if (Math.abs(st.trace[st.trace.length - 1] - L0) < 1e-4 || st.trace.length > 400) { stop(); st.m = st.m ? sortByMean(st.m) : null; draw(); }
    }, 60);
  }
  function stop(): void { if (timer) { clearInterval(timer); timer = 0; } runBtn.textContent = "run"; }
  function computeBic(): void {
    const tot = counts.reduce((a, b) => a + b, 0);
    st.bic = [1, 2, 3, 4, 5, 6].map(K => {
      let best = -Infinity;
      for (let t = 0; t < 4; t++) best = Math.max(best, histEM(counts, initHist(counts, K, t === 0 ? "quantiles" : t === 1 ? "kmeans" : "random", rng(20 + t), st.equal), st.equal).trace.slice(-1)[0]);
      return bic(best, nParams(K, 1, st.equal ? "tied" : "full"), tot);
    });
  }

  function draw(): void {
    const m = st.m;
    if (!img || !m) return;
    const tab = greyTables(m), K = m.k;
    // image
    const ctx = cv.getContext("2d")!, id = ctx.createImageData(N, N), lt = st.tau * Math.LN10;
    let low = 0;
    for (let p = 0; p < N * N; p++) {
      const v = Math.max(0, Math.min(255, Math.round(img.data[p]))), q = 4 * p;
      let r = v, g = v, b = v;
      if (st.view === "labels") { const c = RGB[tab.label[v]]; r = 0.45 * v + 0.55 * c[0]; g = 0.45 * v + 0.55 * c[1]; b = 0.45 * v + 0.55 * c[2]; }
      else if (st.view === "posterior") { const t = Math.round(255 * tab.post[v * K + Math.min(st.cls, K - 1)]); r = VIR[4 * t]; g = VIR[4 * t + 1]; b = VIR[4 * t + 2]; }
      else if (st.view === "lowlik" && tab.logp[v] < lt) { r = 255; g = 20; b = 160; low++; }
      id.data[q] = r; id.data[q + 1] = g; id.data[q + 2] = b; id.data[q + 3] = 255;
    }
    if (st.view !== "lowlik") for (let p = 0; p < N * N; p++) { const v = Math.round(img.data[p]); if (tab.logp[Math.max(0, Math.min(255, v))] < lt) low++; }
    ctx.putImageData(id, 0, 0);
    cap.textContent = st.view === "labels" ? "most probable class per pixel" : st.view === "posterior" ? `posterior of class ${Math.min(st.cls, K - 1) + 1} (0 dark → 1 yellow)` : st.view === "lowlik" ? `p(I) < 10^${st.tau}: ${low} pixels` : (st.img === "cell" ? "micrograph" : "synthetic image");
    clsRow.replaceChildren();
    if (st.view === "posterior") {
      htmlEl("span", "hg-lab", "class:", clsRow);
      for (let j = 0; j < K; j++) { const b = button(clsRow, String(j + 1), () => { st.cls = j; draw(); }); b.style.borderColor = COLS[j]; b.classList.toggle("active", j === Math.min(st.cls, K - 1)); }
    }
    (tauS.input.parentElement as HTMLElement).style.display = st.view === "lowlik" ? "" : "none";
    // histogram: 64 bins of 4 grey levels, curves in pixels per grey level
    const e = Array.from({ length: 65 }, (_, i) => -0.5 + 4 * i), c = new Array(64).fill(0);
    counts.forEach((v, i) => { c[Math.floor(i / 4)] += v / 4; });
    const tot = counts.reduce((a, b) => a + b, 0);
    let top = Math.max(...c);
    for (let v = 0; v < 256; v += 1) top = Math.max(top, tot * densities(m, v).total);
    hp.setDomain([0, 255], [0, top * 1.08]);
    hist.set(e, c);
    comp.forEach((f, j) => f.set(j < K ? (x: number) => tot * densities(m, x).parts[j] : () => NaN));
    sum.set((x: number) => tot * densities(m, x).total);
    const bb = bayesBoundaries(m), ot = st.otsu ? multiOtsu(counts, K).map(t => t - 0.5) : [];
    gB.forEach((l, i) => l.set(i < bb.length ? bb[i] : NaN));
    gO.forEach((l, i) => l.set(i < ot.length ? ot[i] : NaN));
    // BIC
    if (st.bic) {
      const lo = Math.min(...st.bic), hi = Math.max(...st.bic), pad = (hi - lo) * 0.15 || 1;
      bp.setDomain(undefined, [lo - pad, hi + pad]);
      bStyle.base = lo - pad;
      bBars.set([1, 2, 3, 4, 5, 6], st.bic);
    } else bBars.set([], []);
    // readout
    let s = `K = ${K}, ${st.equal ? "equal variances" : "free variances"}   log p(D) = ${f1(st.trace[st.trace.length - 1])}   EM iterations: ${st.trace.length - 1}\n`;
    s += Array.from({ length: K }, (_, j) => `${j + 1}: π ${f2(m.weights[j])}, μ ${f1(m.means[j])}, σ ${f1(Math.sqrt(m.covs[j]))}`).join("\n") + "\n";
    s += `Bayes boundaries: ${bb.map(v => v.toFixed(0)).join(", ") || "none"}` + (st.otsu ? `   Otsu: ${ot.map(v => (v + 0.5).toFixed(0)).join(", ")}` : "");
    if (truth && K === 3) {
      let ok = 0;
      for (let p = 0; p < N * N; p++) if (tab.label[Math.max(0, Math.min(255, Math.round(img.data[p])))] === truth[p]) ok++;
      s += `\npixels labelled as in the truth: ${(100 * ok / (N * N)).toFixed(1)} %`;
    }
    if (st.bic) s += `\nBIC is smallest at K = ${st.bic.indexOf(Math.min(...st.bic)) + 1}`;
    ro.textContent = s;
    viewBtns.forEach((b, i) => b.classList.toggle("active", ["image", "labels", "posterior", "lowlik"][i] === st.view));
    initBtns.forEach((b, i) => b.classList.toggle("active", ["quantiles", "kmeans", "random"][i] === st.init));
    imgBtns.forEach((b, i) => b.classList.toggle("active", ["cell", "synth", "small"][i] === st.img));
    (nS.input.parentElement as HTMLElement).style.opacity = st.img === "cell" ? "0.4" : "1";
    (blS.input.parentElement as HTMLElement).style.opacity = st.img === "cell" ? "0.4" : "1";
  }

  load(p0);
}

autoMount(NAME, mount);
