// How many false spots? A seeded image of noise only, or spots + noise; Gaussian noise (flat
// background 100, sd 10) or Poisson noise (20 photons per pixel, a broad bright patch of up to
// 200 photons). Filter: Gaussian of width σ minus a wide blur (σ = 8) or LoG; the response is
// divided by its noise sd (MAD estimate × √Σw², or a local estimate √background, or after the
// Anscombe transform), local maxima in 3 × 3 windows above k are the detections. Panels: the
// normalised response with the detections (red = false, green = true), the histogram of the
// local-maximum values against the standard normal curve, false spots per image against k
// (measured on 16 seeded noise images, the naive tail estimate and the Euler-characteristic
// formula), and in spots mode the recall against k.
//
// <div data-demo="falsespots" data-preset="k3|maxima|formula|bigger|poisson|local|anscombe|dim" data-presets="..."></div>
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { Grey, createGrey, gaussianBlur, mapGrey } from "../lib/imageops";
import { localMaxima2D } from "../lib/edges";
import { spotImage, spotFilter, estimateNoise, localNoiseMap, anscombe, eulerExpectedMaxima, upperTail, noiseMaximaValues, filterResponseNoise, matchPoints, SpotTruth, Pt } from "../lib/spotdetect";
import { ImgView, injectStyle, label, shownIds, TP_COL, FP_COL, FN_COL } from "../spotdetect/common";

const NAME = "falsespots";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] .fs-main { flex: 0 1 min(400px, calc(100vh - 290px)); min-width: 260px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .fs-mid { flex: 0 1 300px; min-width: 220px; display: flex; flex-direction: column; gap: .2em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 330px; gap: .25em; font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-presets { gap: .2em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-btn { padding: .12em .5em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-slider { gap: .35em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; line-height: 1.3; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] .fs-busy { font-size: 14px; color: var(--qb-muted); min-height: 1.1em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .fs-main { flex: 0 0 400px; display: block; }
.reveal .qb-demo[data-demo="${NAME}"] .fs-mid { flex: 0 0 340px; gap: 0; }
.reveal .qb-demo[data-demo="${NAME}"] .fs-mid { max-width: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 360px; max-width: 360px; gap: .12em; font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 18px; padding: .2em .4em; }
`;

type Kind = "gauss" | "log";
interface State { spots: boolean; poisson: boolean; filter: Kind; sigma: number; k: number; size: number; local: boolean; ansc: boolean }
interface Preset { id: string; label: string; s: Partial<State> }
const DEFAULT: State = { spots: false, poisson: false, filter: "gauss", sigma: 1.5, k: 3, size: 256, local: false, ansc: false };
const PRESETS: Preset[] = [
  { id: "k3", label: "k = 3 is not safe", s: { k: 3 } },
  { id: "maxima", label: "maxima are higher than pixels", s: { k: 2 } },
  { id: "formula", label: "the formula vs the count", s: { k: 4 } },
  { id: "bigger", label: "bigger image, more false spots", s: { k: 3, size: 512 } },
  { id: "poisson", label: "Poisson: false spots on the bright patch", s: { poisson: true, k: 4, filter: "log" } },
  { id: "local", label: "local noise estimate fixes it", s: { poisson: true, k: 4, local: true, filter: "log" } },
  { id: "anscombe", label: "Anscombe fixes it", s: { poisson: true, k: 4, ansc: true, filter: "log" } },
  { id: "dim", label: "dim spots: the detection limit", s: { spots: true, k: 5 } },
];
const KS = Array.from({ length: 33 }, (_, i) => 2 + i * 0.125);
const BG_SIGMA = 8;

/** √Σw² of "Gaussian minus a wide Gaussian" (impulse response). */
const gainCache = new Map<string, number>();
function gain(kind: Kind, s: number): number {
  const key = kind + s;
  if (!gainCache.has(key)) {
    const n = 97, d = createGrey(n, n); d.data[(n >> 1) * n + (n >> 1)] = 1;
    let w: Grey;
    if (kind === "gauss") { const a = gaussianBlur(d, s, "zero"), b = gaussianBlur(d, BG_SIGMA, "zero"); w = mapGrey(a, (v, i) => v - b.data[i]); }
    else w = spotFilter(d, "log", s, "zero");
    gainCache.set(key, filterResponseNoise(w.data));
  }
  return gainCache.get(key)!;
}
function respond(img: Grey, kind: Kind, s: number): Grey {
  if (kind === "log") return spotFilter(img, "log", s);
  const a = gaussianBlur(img, s, "mirror"), b = gaussianBlur(img, BG_SIGMA, "mirror");
  return mapGrey(a, (v, i) => v - b.data[i]);
}

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(p0?.s ?? {}) });
  const st = start();

  const main = htmlEl("div", "fs-main", "", host);
  const view = new ImgView(main, "");
  const busy = htmlEl("div", "fs-busy", "", main);
  const mid = htmlEl("div", "fs-mid", "", host);
  const hp = new Plot(mid, { width: 340, height: 185, x: [-1, 7], y: [0, 1], font: 17, margin: { l: 14, r: 8, t: 24, b: 44 }, xlabel: "maximum value (noise sd)", yticks: [] });
  const hBars = hp.histogram([], [], { color: seriesColor(0), fillOpacity: 0.5 });
  const hNorm = hp.fn(() => 0, { color: "#444", width: 2.5, dash: "6 4" });
  const hK = hp.vline(3, { color: FP_COL, width: 2.5 });
  const hT = hp.text(-0.9, 1, "maxima vs pixels (dashed)", { size: 16 });
  const fp = new Plot(mid, { width: 340, height: 215, x: [2, 6], y: [0.01, 10000], ylog: true, font: 17, margin: { l: 60, r: 8, t: 26, b: 44 }, xlabel: "threshold k (noise sd)" });
  const fMeas = fp.line([], [], { color: seriesColor(0), width: 3 });
  const fNaive = fp.line([], [], { color: "#888", width: 2.5, dash: "4 4" });
  const fEuler = fp.line([], [], { color: seriesColor(1), width: 2.5, dash: "8 4" });
  const fK = fp.vline(3, { color: FP_COL, width: 2 });
  const fDot = fp.scatter([], [], { color: FP_COL, r: 6 });
  fp.text(2.05, 6000, "false spots per image", { size: 16 });
  fp.legend([{ label: "measured", color: seriesColor(0) }, { label: "naive tail", color: "#888", dash: "4 4" }, { label: "Euler formula", color: seriesColor(1), dash: "8 4" }], "tr");
  const rp = new Plot(mid, { width: 340, height: 160, x: [2, 6], y: [0, 1], font: 17, margin: { l: 60, r: 8, t: 22, b: 40 }, xlabel: "k", yticks: [0, 0.5, 1] });
  const rLine = rp.line([], [], { color: TP_COL, width: 3 });
  const rK = rp.vline(3, { color: FP_COL, width: 2 });
  rp.text(2.05, 0.12, "recall", { size: 16 });

  const panel = htmlEl("div", "qb-panel", "", host);
  label(panel, "presets");
  const only = shownIds(host);
  const shown = PRESETS.filter(p => !only || only.includes(p.id));
  const presetBtns = presets(panel, shown.map(p => ({ label: p.label, apply: () => load(p) })));
  label(panel, "image");
  const modeBtns = presets(panel, [
    { label: "noise only", apply: () => { st.spots = false; makeImage(); } },
    { label: "spots + noise", apply: () => { st.spots = true; makeImage(); } },
    { label: "Gaussian noise", apply: () => { st.poisson = false; makeImage(); } },
    { label: "Poisson noise", apply: () => { st.poisson = true; makeImage(); } },
  ]);
  label(panel, "filter");
  const filtBtns = presets(panel, [
    { label: "Gaussian − wide blur", apply: () => { st.filter = "gauss"; computeStats(); makeResponse(); } },
    { label: "LoG", apply: () => { st.filter = "log"; computeStats(); makeResponse(); } },
  ]);
  const sigS = slider(panel, "filter σ", 1, 3, 0.25, st.sigma, v => { st.sigma = v; computeStats(); makeResponse(); });
  const kS = slider(panel, "threshold k", 2, 6, 0.125, st.k, v => { st.k = v; draw(); });
  const zS = slider(panel, "image size", 0, 2, 1, [128, 256, 512].indexOf(st.size), v => { st.size = [128, 256, 512][v]; computeStats(); makeImage(); }, v => String([128, 256, 512][v]) + "²");
  const row = htmlEl("div", "", "", panel);
  row.style.display = "flex"; row.style.flexWrap = "wrap"; row.style.gap = "0 1em";
  const locC = checkbox(row, "local noise", st.local, v => { st.local = v; if (v) { st.ansc = false; ansC.checked = false; } makeResponse(); });
  const ansC = checkbox(row, "Anscombe", st.ansc, v => { st.ansc = v; if (v) { st.local = false; locC.checked = false; } makeResponse(); });
  button(row, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); computeStats(); makeImage(); });
  // pages: the readout under the image (keeps the control column short); slides: in the panel
  const ro = htmlEl("div", "qb-readout", "", host.closest(".reveal") ? panel : main);

  function load(p: Preset): void { Object.assign(st, DEFAULT, p.s); sync(); computeStats(); makeImage(); }
  function sync(): void {
    sigS.set(st.sigma); kS.set(st.k); zS.set([128, 256, 512].indexOf(st.size)); locC.checked = st.local; ansC.checked = st.ansc;
    modeBtns.forEach((b, i) => b.classList.toggle("active", i === (st.spots ? 1 : 0) || i === (st.poisson ? 3 : 2)));
    filtBtns.forEach((b, i) => b.classList.toggle("active", i === (st.filter === "gauss" ? 0 : 1)));
    locC.disabled = ansC.disabled = !st.poisson;
    rp.svg.style.display = st.spots ? "" : "none";
  }

  // ---- measured false-spot curve on seeded white noise (per filter, σ, size) ----
  let stats: { counts: number[]; maxima: number; images: number } | null = null, statsKey = "";
  function computeStats(): void {
    const key = `${st.filter}:${st.sigma}:${st.size}`;
    if (key === statsKey) return;
    statsKey = key; stats = null;
    busy.textContent = "simulating noise images …";
    setTimeout(() => {
      if (statsKey !== key) return;
      const t0 = performance.now(), nImg = st.size >= 512 ? 4 : 16, counts = KS.map(() => 0);
      let maxima = 0;
      for (let s = 1; s <= nImg; s++) {
        const { values } = noiseMaximaValues(st.size, st.sigma, 500 + s, 1, st.filter);
        maxima += values.length;
        for (const v of values) for (let i = 0; i < KS.length; i++) if (v > KS[i]) counts[i]++;
      }
      stats = { counts: counts.map(c => c / nImg), maxima: maxima / nImg, images: nImg };
      busy.textContent = `noise statistics: ${nImg} seeded images of ${st.size}², ${Math.round(performance.now() - t0)} ms`;
      draw();
    }, 20);
  }

  // ---- image and normalised response ----
  let bgImg: Grey | null = null, img: Grey | null = null, truth: SpotTruth[] = [], z: Grey | null = null, maxima: { x: number; y: number; value: number }[] = [], sigHat = 0, sigTrue = 10;
  function makeImage(): void {
    sync();
    const o = st.poisson
      ? { background: 20, bump: 180, poisson: true, amplitude: [12, 40] as [number, number] }
      : { background: 100, noise: 10, amplitude: [12, 40] as [number, number] };
    const s = spotImage({ size: st.size, count: st.spots ? Math.round(40 * (st.size / 256) ** 2) : 0, sigma: 1.5, seed: 11, minDistance: 6, ...o });
    img = s.image; truth = s.truth; bgImg = s.background; sigTrue = st.poisson ? NaN : 10;
    makeResponse();
  }
  function makeResponse(): void {
    if (!img) return;
    sync();
    const src = st.ansc && st.poisson ? mapGrey(img, anscombe) : img;
    sigHat = estimateNoise(src);
    const r = respond(src, st.filter, st.sigma), g = gain(st.filter, st.sigma);
    if (st.local && st.poisson) {
      const nm = localNoiseMap(img, 1, 0, BG_SIGMA);
      z = mapGrey(r, (v, i) => v / (nm.data[i] * g));
    } else z = mapGrey(r, v => v / (sigHat * g));
    maxima = localMaxima2D(z, -Infinity, 1, 4);
    draw();
  }

  function draw(): void {
    if (!img || !z) return;
    const n = st.size, dets = maxima.filter(m => m.value > st.k);
    const T: Pt[] = truth.map(t => [t.x, t.y]), D: Pt[] = dets.map(d => [d.x, d.y]);
    const m = matchPoints(T, D, 2, "optimal");
    const md = new Set(m.pairs.map(p => p.d)), mt = new Set(m.pairs.map(p => p.t));
    view.grey(z, { mode: "clip", lo: -6, hi: 6 });
    view.clear();
    const rad = 3 * n / 256;
    dets.forEach((d, j) => view.add("circle", { cx: d.x + 0.5, cy: d.y + 0.5, r: rad, fill: "none", stroke: md.has(j) ? TP_COL : FP_COL, "stroke-width": 2.2 }));
    truth.forEach((t, i) => { if (!mt.has(i)) view.add("circle", { cx: t.x + 0.5, cy: t.y + 0.5, r: rad * 1.5, fill: "none", stroke: FN_COL, "stroke-width": 2 }); });
    view.setCaption(`response / noise sd (${st.poisson ? "Poisson" : "Gaussian"} noise, ${n}²)`);
    // histogram of maxima vs normal pdf
    const lo = -1, hi = 7, bw = 0.25, nb = (hi - lo) / bw, counts = new Array(nb).fill(0);
    for (const q of maxima) { const b = Math.floor((q.value - lo) / bw); if (b >= 0 && b < nb) counts[b]++; }
    const edges = Array.from({ length: nb + 1 }, (_, i) => lo + i * bw);
    const cmax = Math.max(1, ...counts), tot = maxima.length;
    const pdfScale = tot * bw / Math.sqrt(2 * Math.PI);
    hp.setDomain(undefined, [0, Math.max(cmax, pdfScale) * 1.15]);
    hBars.set(edges, counts);
    hNorm.set(x => pdfScale * Math.exp(-x * x / 2));
    hK.set(st.k);
    hT.set(-0.9, Math.max(cmax, pdfScale) * 1.08, `${tot} maxima; dashed: pixels`);
    // false spots vs k
    const area = (n - 8) ** 2;
    if (stats) {
      fMeas.set(KS, stats.counts.map(c => Math.max(c, 0.005)));
      fNaive.set(KS, KS.map(k => stats!.maxima * upperTail(k)));
    } else { fMeas.set([], []); fNaive.set([], []); }
    fEuler.set(st.filter === "gauss" ? KS : [], st.filter === "gauss" ? KS.map(k => eulerExpectedMaxima(area, st.sigma, k)) : []);
    fK.set(st.k);
    const falseNow = dets.length - m.tp;
    fDot.set([st.k], [Math.max(0.012, falseNow)]);
    // recall vs k
    if (st.spots && truth.length) {
      rLine.set(KS, KS.map(k => matchPoints(T, maxima.filter(q => q.value > k).map(q => [q.x, q.y] as Pt), 2, "optimal").tp / truth.length));
      rK.set(st.k);
    }
    // patch split for Poisson
    let fpPatch = 0;
    if (st.poisson && bgImg) dets.forEach((d, j) => { if (!md.has(j) && bgImg!.data[d.y * n + d.x] > 110) fpPatch++; });
    const expected = stats ? interp(KS, stats.counts, st.k) : NaN;
    ro.textContent =
      `noise σ̂ (MAD) ${sigHat.toFixed(2)}` + (Number.isFinite(sigTrue) ? ` (true ${sigTrue})` : st.ansc ? " (Anscombe units)" : " (Poisson: varies)") +
      `\nfiltered noise ${(sigHat * gain(st.filter, st.sigma)).toFixed(2)}   maxima: ${maxima.length}` +
      `\nfalse spots: ${falseNow}` + (st.poisson ? ` (${fpPatch} on the bright patch)` : "") +
      (stats ? `\nexpected from noise images: ${expected.toFixed(expected < 10 ? 2 : 0)}` : "") +
      (st.spots ? `\nmissed spots: ${m.fn} of ${truth.length} (recall ${(m.tp / truth.length).toFixed(2)})` : "");
  }

  if (p0 && shown.includes(p0)) presetBtns[shown.indexOf(p0)].classList.add("active");
  computeStats();
  makeImage();
}

function interp(xs: number[], ys: number[], x: number): number {
  for (let i = 1; i < xs.length; i++) if (x <= xs[i]) { const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]); return ys[i - 1] + t * (ys[i] - ys[i - 1]); }
  return ys[ys.length - 1];
}

autoMount(NAME, mount);
