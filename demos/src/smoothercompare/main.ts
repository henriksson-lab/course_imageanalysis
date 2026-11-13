// All smoothers on one image. The synthetic test card (or the micrograph) with seeded Gaussian,
// Poisson or salt-and-pepper noise, smoothed by eight methods, each started at the setting with
// the best PSNR (pre-computed for the preset noise levels). Tabs: the results (click one to
// select it, its parameter slider is in the panel; "zoom" shows the same 40 × 40 crop in all
// eight), the method noise (noisy − result), profiles along row 112, and a table of scores
// (PSNR, SSIM, edge width, line and disc contrast kept, staircase index, time), best per
// column highlighted. "threshold" marks the pixels above 120 in each result.
//
// <div data-demo="smoothercompare" data-tab="grid|noise|profiles|scores" data-preset="g15|sp|poisson|g40|texture|flat|look|threshold"></div>
import { autoMount, htmlEl, button, slider, checkbox, presets as presetRow } from "../lib/ui";
import { Plot } from "../lib/plot";
import { CELL_CREDIT } from "../lib/cell";
import { ImgView, injectStyle } from "../canny/imgview";
import { tabRow, choice, cell128, PS_CSS } from "../heatflow/common";
import { METHODS, MethodId, NoiseKind, noisyOf, noiseSd, scores, cardTruth, Scores } from "./methods";
import { psnr255 } from "../lib/pde";

const NAME = "smoothercompare";
const STYLE = PS_CSS + `
.qb-demo[data-demo="${NAME}"] { gap: .8em; flex-wrap: wrap; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .sc-main { flex: 1 1 520px; min-width: 0; display: flex; flex-direction: column; gap: .25em; }
.qb-demo[data-demo="${NAME}"] .sc-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .25em .5em; }
.qb-demo[data-demo="${NAME}"] .sc-grid .ev-fig figcaption { font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.qb-demo[data-demo="${NAME}"] .sc-grid .ev-fig.sel .ev-wrap { outline: 3px solid #e8402a; outline-offset: 1px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 460px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] table.sc-tab { border-collapse: collapse; font-size: 16px; width: 100%; }
.qb-demo[data-demo="${NAME}"] table.sc-tab td, .qb-demo[data-demo="${NAME}"] table.sc-tab th { padding: .15em .4em; border-bottom: 1px solid var(--qb-grid, #ddd); text-align: right; }
.qb-demo[data-demo="${NAME}"] table.sc-tab td:first-child, .qb-demo[data-demo="${NAME}"] table.sc-tab th:first-child { text-align: left; }
.qb-demo[data-demo="${NAME}"] table.sc-tab td.best { background: rgba(44, 160, 44, .25); font-weight: 700; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .06em .38em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 330px; gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .sc-main { flex: 0 0 780px; }
.reveal .qb-demo[data-demo="${NAME}"] .sc-grid { max-width: 700px; }
.reveal .qb-demo[data-demo="${NAME}"] .sc-grid .ev-fig figcaption { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] table.sc-tab { font-size: 17px; }
@media (max-width: 860px) { .qb-demo[data-demo="${NAME}"] { flex-wrap: wrap; } .qb-demo[data-demo="${NAME}"] .sc-grid { grid-template-columns: repeat(2, 1fr); } }
`;

type Tab = "grid" | "noise" | "profiles" | "scores";
interface State { tab: Tab; image: "card" | "cell"; noise: NoiseKind; level: number; zoom: boolean; zx: number; zy: number; thresh: boolean; sel: MethodId; look: boolean }
const DEFAULT: State = { tab: "grid", image: "card", noise: "gauss", level: 15, zoom: false, zx: 82, zy: 20, thresh: false, sel: "tv", look: false };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "g15", label: "Gaussian noise, moderate", s: {} },
  { id: "sp", label: "salt and pepper", s: { noise: "saltpepper", level: 5 } },
  { id: "poisson", label: "Poisson, few photons", s: { noise: "poisson", level: 30 } },
  { id: "g40", label: "strong noise: all fail differently", s: { level: 40 } },
  { id: "texture", label: "texture: NLM", s: { zoom: true, zx: 82, zy: 22, sel: "nlm" } },
  { id: "flat", label: "piecewise constant: TV", s: { zoom: true, zx: 26, zy: 26, sel: "tv" } },
  { id: "look", label: "best PSNR vs best look", s: { image: "cell", look: true } },
  { id: "threshold", label: "smoothing before thresholding", s: { thresh: true, level: 40 } },
];
// best-PSNR settings (grid search with these methods and seeds; see tools/figs/smoothers/)
const BEST: Record<string, Record<MethodId, number>> = {
  "card-gauss-15": { gauss: 0.541, median: 1, pm: 0.268, tv: 9.34, bilateral: 28.6, nlm: 19.7, wavelet: 16.5, wiener: 3.66 },
  "card-gauss-40": { gauss: 0.903, median: 1, pm: 0.47, tv: 29.9, bilateral: 88.5, nlm: 38.3, wavelet: 56.7, wiener: 0.738 },
  "card-saltpepper-5": { gauss: 0.783, median: 1, pm: 0.349, tv: 39.5, bilateral: 107, nlm: 35.8, wavelet: 51.9, wiener: 1.1 },
  "card-poisson-30": { gauss: 0.742, median: 1, pm: 0.338, tv: 20.2, bilateral: 57, nlm: 30.9, wavelet: 35.8, wiener: 1.35 },
  "cell-gauss-15": { gauss: 0.501, median: 1, pm: 0.118, tv: 5.49, bilateral: 23.9, nlm: 16, wavelet: 10, wiener: 5.5 },
};
// stronger settings chosen "by eye" (cleaner flat regions) for the micrograph
const BY_EYE: Record<MethodId, number> = { gauss: 1.2, median: 2, pm: 3, tv: 15, bilateral: 45, nlm: 25, wavelet: 30, wiener: 0.3 };
const N = 128, Z = 40;

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(host.dataset.tab ? { tab: host.dataset.tab as Tab } : {}), ...(p0?.s ?? {}) });
  const st: State = start();

  const main = htmlEl("div", "sc-main", "", host);
  const setTab = tabRow(main, [["grid", "results"], ["noise", "method noise"], ["profiles", "profiles"], ["scores", "scores"]], id => { st.tab = id as Tab; draw(); });
  const grid = htmlEl("div", "sc-grid", "", main);
  const views = METHODS.map(m => { const v = new ImgView(grid, m.label); v.onPick((x, y) => pick(m.id, x, y), false); return v; });
  const profDiv = htmlEl("div", "sc-grid", "", main);
  const profs = METHODS.map(m => {
    const p = new Plot(profDiv, { width: 200, height: 150, x: [0, 127], y: [30, 210], font: 15, title: m.label, margin: { l: 40, b: 26, t: 26 }, xticks: [0, 64, 127], yticks: [50, 150] });
    return { p, t: p.line([], [], { color: "var(--qb-muted, #999)", width: 1.5, dash: "3 3" }), u: p.line([], [], { color: "var(--qb-c1)", width: 2.2 }) };
  });
  const tabDiv = htmlEl("div", "", "", main);
  const ro = htmlEl("div", "qb-readout", "", main);
  const credit = htmlEl("div", "ps-credit", CELL_CREDIT, main);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ps-lab", "presets", panel);
  const presetBtns = presetRow(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const setImg = choice(panel, "image", [["card", "test card"], ["cell", "micrograph"]], id => { st.image = id as State["image"]; st.look = false; scenario(); });
  const setNoise = choice(panel, "noise", [["gauss", "Gaussian"], ["poisson", "Poisson"], ["saltpepper", "salt & pepper"]], id => { st.noise = id as NoiseKind; st.level = id === "gauss" ? 15 : id === "poisson" ? 30 : 5; levelS.set(st.level); scenario(); });
  const levelS = slider(panel, "level", 1, 60, 1, st.level, v => { st.level = v; });
  levelS.input.addEventListener("change", () => scenario());
  const zoomC = checkbox(panel, "zoom: 40 × 40 crop (click a result)", st.zoom, v => { st.zoom = v; draw(); });
  const thrC = checkbox(panel, "threshold at 120 (red)", st.thresh, v => { st.thresh = v; draw(); });
  htmlEl("div", "ps-lab", "parameter of the selected method", panel);
  const selLab = htmlEl("div", "", "", panel);
  const fmtU = (u: number): string => { const m = METHODS.find(q => q.id === st.sel)!, f = u / 1000; const v = m.log ? m.lo * Math.pow(m.hi / m.lo, f) : m.lo + (m.hi - m.lo) * f; return m.id === "median" ? String(Math.round(v)) : String(Number(v.toPrecision(3))); };
  const paramS = slider(panel, "value", 0, 1000, 1, 500, () => { showParam(); }, fmtU);
  paramS.input.addEventListener("change", () => { params[st.sel] = paramFromSlider(); computeOne(st.sel); });
  button(panel, "back to best PSNR", () => { const b = bestFor(); for (const m of METHODS) params[m.id] = b[m.id]; computeAll(); });

  const card = cardTruth();
  let cellImg: Float64Array | null = null, truth: Float64Array = card, noisy: Float64Array = card, sd = 0;
  const params: Record<MethodId, number> = { ...BEST["card-gauss-15"] };
  const res: Partial<Record<MethodId, { u: Float64Array; ms: number; sc: Scores; p: number }>> = {};
  let token = 0;

  const key = () => `${st.image}-${st.noise}-${st.level}`;
  function bestFor(): Record<MethodId, number> {
    if (st.look) return BY_EYE;
    return BEST[key()] ?? BEST[`${st.image}-gauss-15`] ?? BEST["card-gauss-15"];
  }
  const sel = () => METHODS.find(m => m.id === st.sel)!;
  function paramFromSlider(): number { const m = sel(), u = paramS.get() / 1000; const v = m.log ? m.lo * Math.pow(m.hi / m.lo, u) : m.lo + (m.hi - m.lo) * u; return m.id === "median" ? Math.round(v) : Number(v.toPrecision(3)); }
  function sliderFromParam(v: number): number { const m = sel(); return 1000 * (m.log ? Math.log(v / m.lo) / Math.log(m.hi / m.lo) : (v - m.lo) / (m.hi - m.lo)); }
  function showParam(): void { const m = sel(); selLab.textContent = `${m.label}: ${m.param} = ${paramFromSlider()}`; }
  function syncParam(): void { paramS.set(sliderFromParam(params[st.sel])); showParam(); }
  void fmtU;

  function pick(id: MethodId, x: number, y: number): void {
    st.sel = id; syncParam();
    if (st.zoom) { /* clicks inside a zoomed crop move within the crop */ st.zx = Math.round(st.zx - Z / 2 + x); st.zy = Math.round(st.zy - Z / 2 + y); }
    else { st.zx = x; st.zy = y; }
    draw();
  }
  function load(p: Preset): void { Object.assign(st, DEFAULT, p.s); sync(); scenario(); }
  function sync(): void { setTab(st.tab); setImg(st.image); setNoise(st.noise); levelS.set(st.level); zoomC.checked = st.zoom; thrC.checked = st.thresh; syncParam(); }
  function scenario(): void {
    if (st.image === "cell" && !cellImg) { cell128().then(g => { cellImg = Float64Array.from(g.data); scenario(); }); return; }
    truth = st.image === "cell" ? cellImg! : card;
    noisy = noisyOf(truth, { image: st.image, noise: st.noise, level: st.level });
    sd = noiseSd(truth, noisy);
    const b = bestFor();
    for (const m of METHODS) params[m.id] = b[m.id];
    syncParam();
    computeAll();
  }
  function computeAll(): void {
    const my = ++token;
    for (const m of METHODS) delete res[m.id];
    draw();
    let k = 0;
    const next = () => { if (my !== token || k >= METHODS.length) return; computeOne(METHODS[k++].id, false); setTimeout(next, 0); };
    setTimeout(next, 0);
  }
  function computeOne(id: MethodId, redraw = true): void {
    const m = METHODS.find(q => q.id === id)!;
    const t0 = performance.now(), u = m.run(noisy, params[id], sd), ms = performance.now() - t0;
    res[id] = { u, ms, sc: scores(u, truth), p: params[id] };
    void redraw;
    draw();
  }

  function crop(a: Float64Array): { d: Float64Array; w: number } {
    if (!st.zoom) return { d: a, w: N };
    const x0 = Math.max(0, Math.min(N - Z, st.zx - Z / 2)), y0 = Math.max(0, Math.min(N - Z, st.zy - Z / 2)), d = new Float64Array(Z * Z);
    for (let y = 0; y < Z; y++) for (let x = 0; x < Z; x++) d[y * Z + x] = a[(y0 + y) * N + x0 + x];
    return { d, w: Z };
  }
  function draw(): void {
    setTab(st.tab);
    grid.style.display = st.tab === "grid" || st.tab === "noise" ? "" : "none";
    profDiv.style.display = st.tab === "profiles" ? "" : "none";
    tabDiv.style.display = st.tab === "scores" ? "" : "none";
    credit.style.display = st.image === "cell" ? "" : "none";
    const xs = Array.from({ length: N }, (_, i) => i), row = (st.image === "cell" ? 64 : 112) * N;
    METHODS.forEach((m, i) => {
      const r = res[m.id], v = views[i];
      v.fig.classList.toggle("sel", m.id === st.sel);
      if (!r) { v.setCaption(`${m.label}: computing…`); return; }
      if (st.tab === "noise") {
        const mn = new Float64Array(N * N); for (let q = 0; q < mn.length; q++) mn[q] = noisy[q] - r.u[q];
        const c = crop(mn); v.grey({ width: c.w, height: c.w, data: Float32Array.from(c.d) }, { mode: "clip", lo: -50, hi: 50 });
        v.setCaption(`${m.label}`);
      } else {
        const c = crop(r.u);
        if (st.thresh) {
          const px = new Uint8ClampedArray(4 * c.w * c.w);
          for (let q = 0; q < c.w * c.w; q++) { const g = Math.max(0, Math.min(255, c.d[q])); const on = c.d[q] > 120; px[4 * q] = on ? 232 : g * 0.8; px[4 * q + 1] = on ? 64 : g * 0.8; px[4 * q + 2] = on ? 42 : g * 0.8; px[4 * q + 3] = 255; }
          v.rgba(px, c.w, c.w);
        } else v.grey({ width: c.w, height: c.w, data: Float32Array.from(c.d) }, { mode: "clip", lo: 0, hi: 255 });
        v.setCaption(`${m.label} · ${r.sc.psnr.toFixed(1)} dB`);
      }
      profs[i].t.set(xs, truth.subarray(row, row + N)); profs[i].u.set(xs, r.u.subarray(row, row + N));
    });
    // scores table
    tabDiv.replaceChildren();
    const done = METHODS.filter(m => res[m.id]);
    if (st.tab === "scores") {
      const t = htmlEl("table", "sc-tab", "", tabDiv);
      const card = st.image === "card";
      const cols: [string, (s: Scores, ms: number) => number, "max" | "min" | "one"][] = [["PSNR (dB)", s => s.psnr, "max"], ["SSIM", s => s.ssim, "max"]];
      if (card) cols.push(["edge width", s => s.edge, "min"], ["line kept", s => s.line, "one"], ["disc kept", s => s.disc, "one"], ["staircase", s => s.stairs, "min"]);
      cols.push(["ms", (_, ms) => ms, "min"]);
      const hr = htmlEl("tr", "", "", t); htmlEl("th", "", "method", hr); for (const c of cols) htmlEl("th", "", c[0], hr);
      const bestIdx = cols.map(([, f, kind]) => {
        let bi = -1, bv = Infinity;
        done.forEach((m, k) => { const r = res[m.id]!, v = f(r.sc, r.ms), score = kind === "max" ? -v : kind === "min" ? v : Math.abs(v - 1); if (score < bv) { bv = score; bi = k; } });
        return bi;
      });
      done.forEach((m, k) => {
        const r = res[m.id]!, tr = htmlEl("tr", "", "", t);
        htmlEl("td", "", `${m.label} (${m.param} ${r.p})`, tr);
        cols.forEach(([, f], c) => { const v = f(r.sc, r.ms); htmlEl("td", bestIdx[c] === k ? "best" : "", c === cols.length - 1 ? v.toFixed(0) : v.toFixed(2), tr); });
      });
    }
    const np = psnr255(noisy, truth);
    ro.textContent = `noisy input: ${np.toFixed(1)} dB (noise SD ${sd.toFixed(1)}) · ` + (st.look ? "settings chosen by eye (stronger than best PSNR)" : "each method at its best-PSNR setting for the preset noise levels") +
      (st.tab === "scores" && st.image === "card" ? "\nedge: 10–90 % width of the step (px) · line / disc: contrast kept (1 = all) · staircase: 1 = smooth ramp" : st.tab === "noise" ? "\nmethod noise = noisy − result (grey = 0, ±50): structure here was removed by the method" : "");
  }

  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  scenario();
}

autoMount(NAME, mount);
