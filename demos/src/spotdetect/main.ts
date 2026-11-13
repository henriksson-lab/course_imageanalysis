// Spot detection: filter, local maxima, threshold. A seeded synthetic 256 × 256 fluorescence
// image (pixel-integrated Gaussian spots of σ = 1.5 px, amplitudes 20–100, optional uneven
// background, Gaussian noise; truth known). Filters: none, Gaussian, zero-mean Gaussian
// (matched filter with the background removed), LoG, DoG, top-hat + Gaussian, à trous
// product (with the level strip w1…w4, a4 and the product). Maxima in a (2r+1)² window,
// threshold k × the noise of the filtered image (noise estimated by MAD of à trous level 1).
// Detections: green = matched to a true spot (within 2 px, one-to-one), red = false,
// yellow rings = missed. Click: the 9 × 9 response around the pixel with the window.
//
// <div data-demo="spotdetect" data-preset="flat|uneven|unevenlog|atrous|w1|small|large|window|dense|log"
//      data-presets="flat,uneven,unevenlog" data-layout="full|compact"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { Grey, cropGrey } from "../lib/imageops";
import { localMaxima2D } from "../lib/edges";
import { SpotFilter, SPOT_FILTER_LABELS, spotFilter, spotImage, estimateNoise, noiseGain, atrousProduct, matchPoints, SpotTruth, Pt } from "../lib/spotdetect";
import { ImgView, injectStyle, label, shownIds, TP_COL, FP_COL, FN_COL } from "./common";

const NAME = "spotdetect";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] .sd-main { flex: 0 1 440px; min-width: 300px; }
.qb-demo[data-demo="${NAME}"] .sd-mid { flex: 0 1 290px; min-width: 200px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 380px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] .sd-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 3px; }
.qb-demo[data-demo="${NAME}"] .sd-strip figure { margin: 0; }
.qb-demo[data-demo="${NAME}"] .sd-strip figcaption { font-size: 15px; text-align: center; }
.qb-demo[data-demo="${NAME}"] .sd-strip canvas { width: 100%; image-rendering: pixelated; display: block; }
.qb-demo[data-demo="${NAME}"] .sd-inset { display: flex; gap: .5em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .sd-inset .ev-fig { flex: 0 0 120px; }
.qb-demo[data-demo="${NAME}"] .sd-inset .qb-readout { flex: 1 1 auto; font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .sd-busy { font-size: 14px; color: var(--qb-muted); min-height: 1.1em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .sd-main { flex: 0 0 450px; }
.reveal .qb-demo[data-demo="${NAME}"] .sd-mid { flex: 0 0 280px; gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 360px; gap: .12em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 18px; padding: .2em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .sd-inset { display: none; }
`;

const N = 256, PSF = 1.5, GATE = 2;
interface State { filter: SpotFilter; sigma: number; r: number; k: number; noise: number; background: boolean; count: number; j0: number; j1: number; px: number; py: number }
interface Preset { id: string; label: string; s: Partial<State> }
const DEFAULT: State = { filter: "zmgauss", sigma: 1.5, r: 2, k: 4, noise: 10, background: true, count: 80, j0: 2, j1: 3, px: -1, py: -1 };
const PRESETS: Preset[] = [
  { id: "flat", label: "Gaussian, flat background", s: { filter: "gauss", background: false } },
  { id: "uneven", label: "uneven background: Gaussian fails", s: { filter: "gauss", background: true } },
  { id: "unevenlog", label: "… LoG does not", s: { filter: "log", background: true } },
  { id: "atrous", label: "à trous product", s: { filter: "atrous", j0: 2, j1: 3, k: 3 } },
  { id: "w1", label: "à trous: only w1 = noise", s: { filter: "atrous", j0: 1, j1: 1, k: 3 } },
  { id: "small", label: "filter σ = 0.5: noise maxima", s: { filter: "log", sigma: 0.5 } },
  { id: "large", label: "filter σ = 4: neighbours merge", s: { filter: "log", sigma: 4, count: 200 } },
  { id: "window", label: "no filter: window too small", s: { filter: "none", r: 1, k: 4 } },
  { id: "dense", label: "dense spots", s: { filter: "log", count: 300 } },
  { id: "log", label: "LoG at one σ is enough", s: { filter: "log", sigma: 1.5 } },
];
const FILTERS: SpotFilter[] = ["none", "gauss", "zmgauss", "log", "dog", "tophat", "atrous"];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(p0?.s ?? {}) });
  const st = start();

  const main = htmlEl("div", "sd-main", "", host);
  const view = new ImgView(main, "");
  const mid = htmlEl("div", "sd-mid", "", host);
  const resp = new ImgView(mid, "");
  const strip = htmlEl("div", "sd-strip", "", mid);
  const stripC = ["w1", "w2", "w3", "w4", "a4 (background)", "product"].map(t => {
    const f = htmlEl("figure", "", "", strip); htmlEl("figcaption", "", t, f); return htmlEl("canvas", "", "", f);
  });
  const busy = htmlEl("div", "sd-busy", "", mid);
  const insetRow = htmlEl("div", "sd-inset", "", mid);
  const inset = new ImgView(insetRow, "9 × 9 response");
  const insetRo = htmlEl("div", "qb-readout", "click a spot", insetRow);

  const panel = htmlEl("div", "qb-panel", "", host);
  label(panel, "presets");
  const only = shownIds(host);
  const shown = PRESETS.filter(p => !only || only.includes(p.id));
  const presetBtns = presets(panel, shown.map(p => ({ label: p.label, apply: () => load(p) })));
  label(panel, "filter");
  const filtBtns = presets(panel, FILTERS.map(f => ({ label: SPOT_FILTER_LABELS[f], apply: () => { st.filter = f; sync(); computeResponse(); } })));
  const sigDiv = htmlEl("div", "", "", panel);
  const sigS = slider(sigDiv, "filter σ", 0.5, 4, 0.1, st.sigma, v => { st.sigma = v; computeResponse(); });
  const jDiv = htmlEl("div", "", "", panel);
  const j0S = slider(jDiv, "first level", 1, 4, 1, st.j0, v => { st.j0 = v; if (st.j1 < v) { st.j1 = v; j1S.set(v); } computeResponse(); });
  const j1S = slider(jDiv, "last level", 1, 4, 1, st.j1, v => { st.j1 = v; if (st.j0 > v) { st.j0 = v; j0S.set(v); } computeResponse(); });
  const rS = slider(panel, "window r", 1, 6, 1, st.r, v => { st.r = v; computeMaxima(); });
  const kS = slider(panel, "threshold k (noise sd)", 1, 10, 0.25, st.k, v => { st.k = v; st.filter === "atrous" ? computeResponse() : draw(); });
  const nS = slider(panel, "noise sd", 0, 30, 1, st.noise, v => { st.noise = v; makeImage(); });
  const cS = slider(panel, "spots", 10, 400, 10, st.count, v => { st.count = v; makeImage(); });
  const row = htmlEl("div", "", "", panel);
  row.style.display = "flex"; row.style.gap = "0 1em"; row.style.flexWrap = "wrap";
  const bgC = checkbox(row, "uneven background", st.background, v => { st.background = v; makeImage(); });
  button(row, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); makeImage(); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  function load(p: Preset): void { Object.assign(st, DEFAULT, p.s); sync(); makeImage(); }
  function sync(): void {
    sigS.set(st.sigma); j0S.set(st.j0); j1S.set(st.j1); rS.set(st.r); kS.set(st.k); nS.set(st.noise); cS.set(st.count); bgC.checked = st.background;
    FILTERS.forEach((f, i) => filtBtns[i].classList.toggle("active", f === st.filter));
    sigDiv.style.display = st.filter === "atrous" || st.filter === "none" ? "none" : "";
    jDiv.style.display = st.filter === "atrous" ? "" : "none";
    strip.style.display = st.filter === "atrous" ? "" : "none";
  }

  view.onPick((x, y) => { st.px = x; st.py = y; draw(); }, false);
  resp.onPick((x, y) => { st.px = x; st.py = y; draw(); }, false);

  let img: Grey | null = null, truth: SpotTruth[] = [], response: Grey | null = null, noiseEst = 0, respNoise = 1;
  let cands: { x: number; y: number; value: number }[] = [];
  function makeImage(): void {
    const s = spotImage({ size: N, count: st.count, sigma: PSF, amplitude: [20, 100], background: 40, slope: st.background ? 60 : 0, bump: st.background ? 90 : 0, noise: st.noise, seed: 7 });
    img = s.image; truth = s.truth;
    noiseEst = estimateNoise(img);
    computeResponse();
  }
  function computeResponse(): void {
    if (!img) return;
    sync();
    const t0 = performance.now();
    if (st.filter === "atrous") {
      const sc: number[] = []; for (let j = st.j0; j <= st.j1; j++) sc.push(j);
      const A = atrousProduct(img, sc, st.k, dec && decImg === img ? dec : undefined);
      dec = A.dec; decImg = img; response = A.product; respNoise = NaN;
      drawStrip();
    } else {
      response = spotFilter(img, st.filter, st.sigma);
      respNoise = noiseEst * noiseGain(st.filter, st.sigma);
    }
    busy.textContent = `filter: ${Math.round(performance.now() - t0)} ms`;
    computeMaxima();
  }
  let dec: { w: Grey[]; a: Grey[] } | null = null, decImg: Grey | null = null;
  function computeMaxima(): void {
    if (!response) return;
    cands = localMaxima2D(response, st.filter === "atrous" ? 0 : -Infinity, st.r, 2);
    draw();
  }
  function drawStrip(): void {
    if (!dec || !response) return;
    const imgs = [...dec.w.slice(0, 4), dec.a[3] ?? dec.a[dec.a.length - 1], response];
    while (imgs.length < 6) imgs.unshift(response);
    imgs.forEach((g, i) => {
      const c = stripC[i]; c.width = N; c.height = N;
      const ctx = c.getContext("2d")!, id = ctx.createImageData(N, N);
      let lo = Infinity, hi = -Infinity, m = 0;
      for (const v of g.data) { lo = Math.min(lo, v); hi = Math.max(hi, v); m = Math.max(m, Math.abs(v)); }
      const signed = i < 4;
      const pw = i === 5 ? 0.25 : 1;
      for (let q = 0; q < N * N; q++) {
        const v = signed ? 128 + 127 * g.data[q] / (m || 1) : 255 * Math.pow((g.data[q] - lo) / (hi - lo || 1), pw);
        id.data[4 * q] = id.data[4 * q + 1] = id.data[4 * q + 2] = v; id.data[4 * q + 3] = 255;
      }
      ctx.putImageData(id, 0, 0);
    });
  }

  function draw(): void {
    if (!img || !response) return;
    const thr = st.filter === "atrous" ? 0 : st.k * respNoise;
    const dets = cands.filter(c => c.value > thr);
    const T: Pt[] = truth.map(t => [t.x, t.y]), D: Pt[] = dets.map(d => [d.x, d.y]);
    const m = matchPoints(T, D, GATE, "optimal");
    const matchedD = new Set(m.pairs.map(p => p.d)), matchedT = new Set(m.pairs.map(p => p.t));
    view.grey(img, { mode: "auto" });
    view.clear();
    truth.forEach((t, i) => { if (!matchedT.has(i)) view.add("circle", { cx: t.x + 0.5, cy: t.y + 0.5, r: 4.5, fill: "none", stroke: FN_COL, "stroke-width": 2 }); });
    dets.forEach((d, j) => view.add("circle", { cx: d.x + 0.5, cy: d.y + 0.5, r: 3, fill: "none", stroke: matchedD.has(j) ? TP_COL : FP_COL, "stroke-width": 2.2 }));
    view.setCaption(`synthetic image: ${truth.length} spots, noise ${st.noise}`);
    resp.grey(response, st.filter === "atrous" ? { mode: "auto" } : { mode: "mid" });
    resp.clear();
    resp.setCaption(st.filter === "atrous" ? `à trous product, levels ${st.j0}–${st.j1}` : `${SPOT_FILTER_LABELS[st.filter]} response (0 = grey)`);
    dets.forEach((d, j) => resp.add("circle", { cx: d.x + 0.5, cy: d.y + 0.5, r: 3, fill: "none", stroke: matchedD.has(j) ? TP_COL : FP_COL, "stroke-width": 1.5 }));
    // inset
    if (st.px >= 0) {
      let best = -1, bd = 6;
      dets.forEach((d, j) => { const dd = Math.hypot(d.x - st.px, d.y - st.py); if (dd < bd) { bd = dd; best = j; } });
      const cx = best >= 0 ? dets[best].x : st.px, cy = best >= 0 ? dets[best].y : st.py;
      const crop = cropGrey(response, cx - 4, cy - 4, 9, 9, "zero");
      inset.grey(crop, { mode: "auto" });
      inset.clear();
      inset.box(4, 4, st.r, "#ffd400", 2.5); inset.box(4, 4, 0, "#ff4fd8", 2.5);
      view.box(cx, cy, 4, "#ff4fd8", 2); resp.box(cx, cy, 4, "#ff4fd8", 2);
      let wmax = -Infinity;
      for (let dy = -st.r; dy <= st.r; dy++) for (let dx = -st.r; dx <= st.r; dx++) {
        const x = cx + dx, y = cy + dy; if (x >= 0 && y >= 0 && x < N && y < N) wmax = Math.max(wmax, response.data[y * N + x]);
      }
      const v = response.data[cy * N + cx];
      insetRo.textContent = `pixel (${cx}, ${cy})\nvalue ${v.toFixed(1)}\nwindow max ${wmax.toFixed(1)}\n` +
        (v >= wmax ? (v > thr ? "maximum, above threshold: kept" : "maximum, below threshold") : "not the window maximum");
    }
    const sc = `TP ${m.tp}   FP ${m.fp}   FN ${m.fn}   F1 ${m.f1.toFixed(2)}`;
    ro.textContent =
      `candidates (maxima): ${cands.length}   kept: ${dets.length}\n${sc}\n` +
      (st.filter === "atrous"
        ? `per level: keep w_j > ${st.k} σ_j (MAD)`
        : `noise σ̂ (MAD) ${noiseEst.toFixed(1)} (true ${st.noise})\nfiltered noise ${respNoise.toFixed(2)}, threshold ${thr.toFixed(1)}`);
  }

  if (p0 && shown.includes(p0)) presetBtns[shown.indexOf(p0)].classList.add("active");
  makeImage();
}

autoMount(NAME, mount);
