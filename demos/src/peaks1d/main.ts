// Spots in 1D: filter, then local maxima. A 256-sample profile with Gaussian spots of σ = 1.5
// (a pair 7 px apart, an isolated spot, a dim spot next to a bright one, a dim isolated spot,
// two spots on a ramp) and seeded noise. Rows: the raw profile (truth as ticks); the filtered
// profile with the candidates (local maxima in a window of radius r, or h-maxima: peaks whose
// height above their saddle is at least h, drawn as bars); the filter kernel to scale.
// Click a row to select the nearest spot: its single-pixel and filtered SNR are read out.
//
// <div data-demo="peaks1d" data-preset="raw|matched|wide|small|large|ramp|plateau|hmax"
//      data-presets="raw,matched,wide"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, rng } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { SpotFilter, SPOT_FILTER_LABELS, filter1d, kernel1d, localMaxima1d, prominences, atrousProduct1d, pixelFraction, filterResponseNoise } from "../lib/spotdetect";
import { injectStyle, label, shownIds, TP_COL, FP_COL } from "../spotdetect/common";

const NAME = "peaks1d";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] .pk-plots { flex: 1 1 760px; min-width: 0; display: flex; flex-direction: column; gap: .1em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 340px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre-wrap; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .pk-plots { flex: 0 0 770px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 350px; gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .06em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 18px; padding: .2em .4em; }
`;

const L = 256, S_PSF = 1.5, BASE = 20;
/** The spots: position, amplitude (peak of the point-sampled Gaussian). */
const SPOTS: { x: number; a: number; name: string }[] = [
  { x: 27, a: 40, name: "pair, left" }, { x: 34, a: 40, name: "pair, right" },
  { x: 70.5, a: 40, name: "isolated (half-pixel)" },
  { x: 105, a: 80, name: "bright" }, { x: 112, a: 28, name: "dim neighbour" },
  { x: 140, a: 16, name: "dim, isolated" },
  { x: 190, a: 30, name: "on the ramp" }, { x: 230, a: 30, name: "on the ramp, high" },
];
const ramp = (x: number) => (x > 160 ? 0.6 * (x - 160) : 0);

type Mode = "window" | "hmax";
interface State { filter: SpotFilter; sigma: number; r: number; noise: number; h: number; mode: Mode; quant: boolean; sel: number; seed: number }
interface Preset { id: string; label: string; s: Partial<State> }
const DEFAULT: State = { filter: "gauss", sigma: 1.5, r: 3, noise: 5, h: 10, mode: "window", quant: false, sel: 2, seed: 1 };
const PRESETS: Preset[] = [
  { id: "raw", label: "raw maxima: one per noise wiggle", s: { filter: "none", r: 1 } },
  { id: "matched", label: "matched filter", s: { filter: "gauss", sigma: 1.5, r: 3 } },
  { id: "wide", label: "filter too wide: the pair merges", s: { filter: "gauss", sigma: 4, r: 3, sel: 0 } },
  { id: "small", label: "window too small: double peaks", s: { filter: "gauss", sigma: 0.6, r: 1, noise: 8 } },
  { id: "large", label: "window too large: dim neighbour lost", s: { filter: "gauss", sigma: 1.5, r: 8, sel: 4 } },
  { id: "ramp", label: "ramp: zero-mean filter", s: { filter: "zmgauss", sigma: 1.5, r: 3, sel: 6 } },
  { id: "plateau", label: "plateau", s: { filter: "none", noise: 0, quant: true, r: 1, sel: 2 } },
  { id: "hmax", label: "h-maxima instead of a window", s: { filter: "gauss", sigma: 1.5, mode: "hmax", h: 3, sel: 4 } },
];
const FILTERS: SpotFilter[] = ["none", "gauss", "zmgauss", "log", "dog", "atrous"];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(p0?.s ?? {}) });
  const st = start();

  const plots = htmlEl("div", "pk-plots", "", host);
  const W = 770;
  const pRaw = new Plot(plots, { width: W, height: 165, x: [0, L - 1], y: [0, 150], margin: { l: 60, r: 10, t: 28, b: 30 }, font: 20, xticks: [] });
  const pF = new Plot(plots, { width: W, height: 200, x: [0, L - 1], y: [-10, 40], margin: { l: 60, r: 10, t: 28, b: 36 }, font: 20 });
  const pK = new Plot(plots, { width: W, height: 120, x: [-12, 12], y: [-0.2, 1], margin: { l: 60, r: 10, t: 24, b: 46 }, font: 20, xlabel: "filter kernel: offset (px)", yticks: [] });
  const tRaw = pRaw.text(4, 145, "raw profile (truth: ticks)", { size: 18 });
  const tF = pF.text(4, 38, "", { size: 18 });
  void tRaw;
  const raw = pRaw.line([], [], { color: "#555", width: 2 });
  const truthMk = pRaw.mark<[]>(g => {
    for (const s of SPOTS) { const x = pRaw.x(s.x); const l = document.createElementNS("http://www.w3.org/2000/svg", "line"); l.setAttribute("x1", String(x)); l.setAttribute("x2", String(x)); l.setAttribute("y1", String(pRaw.box.b)); l.setAttribute("y2", String(pRaw.box.b - 12)); l.style.stroke = seriesColor(2); l.style.strokeWidth = "3"; g.appendChild(l); }
  }, []);
  void truthMk;
  const selRaw = pRaw.vline(0, { color: "var(--qb-c4)", width: 2 });
  const zero = pF.hline(0, { color: "#999", width: 1.5 });
  void zero;
  const filt = pF.line([], [], { color: seriesColor(0), width: 2.5 });
  const bars = pF.mark<[{ i: number; v: number; saddle: number; keep: boolean }[]]>((g, list) => {
    for (const b of list) {
      const x = pF.x(b.i), y0 = pF.y(b.saddle), y1 = pF.y(b.v);
      const l = document.createElementNS("http://www.w3.org/2000/svg", "line");
      l.setAttribute("x1", String(x + 7)); l.setAttribute("x2", String(x + 7)); l.setAttribute("y1", String(y0)); l.setAttribute("y2", String(y1));
      l.style.stroke = b.keep ? TP_COL : "#aaa"; l.style.strokeWidth = "4"; g.appendChild(l);
    }
  }, [[]]);
  const cands = pF.scatter([], [], { color: FP_COL, r: 6 });
  const bracket = pF.mark<[number, number, number]>((g, c, r, y) => {
    if (c < 0) return;
    const x0 = pF.x(c - r - 0.5), x1 = pF.x(c + r + 0.5), py = pF.y(y) - 16;
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", `M${x0},${py + 8}V${py}H${x1}V${py + 8}`);
    p.style.fill = "none"; p.style.stroke = "var(--qb-c4)"; p.style.strokeWidth = "3"; g.appendChild(p);
  }, [-1, 0, 0]);
  const selF = pF.vline(0, { color: "var(--qb-c4)", width: 2 });
  const kLine = pK.line([], [], { color: seriesColor(0), width: 3 });
  const kStem = pK.scatter([], [], { color: seriesColor(0), r: 4 });
  const kZero = pK.hline(0, { color: "#999", width: 1.5 });
  void kZero;

  const panel = htmlEl("div", "qb-panel", "", host);
  label(panel, "presets");
  const only = shownIds(host);
  const shown = PRESETS.filter(p => !only || only.includes(p.id));
  const presetBtns = presets(panel, shown.map(p => ({ label: p.label, apply: () => load(p) })));
  label(panel, "filter");
  const filtBtns = presets(panel, FILTERS.map(f => ({ label: SPOT_FILTER_LABELS[f], apply: () => { st.filter = f; run(); } })));
  const sigS = slider(panel, "filter σ", 0.3, 6, 0.1, st.sigma, v => { st.sigma = v; run(); });
  const rS = slider(panel, "window r", 1, 12, 1, st.r, v => { st.r = v; st.mode = "window"; modeC.checked = false; run(); });
  const hS = slider(panel, "height h", 0, 30, 0.5, st.h, v => { st.h = v; st.mode = "hmax"; modeC.checked = true; run(); });
  const nS = slider(panel, "noise sd", 0, 15, 0.5, st.noise, v => { st.noise = v; run(); });
  const row = htmlEl("div", "", "", panel);
  row.style.display = "flex"; row.style.flexWrap = "wrap"; row.style.gap = "0 1em";
  const modeC = checkbox(row, "h-maxima", st.mode === "hmax", v => { st.mode = v ? "hmax" : "window"; run(); });
  const qC = checkbox(row, "integers", st.quant, v => { st.quant = v; run(); });
  const brow = htmlEl("div", "qb-presets", "", panel);
  button(brow, "new noise", () => { st.seed++; run(); });
  button(brow, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); run(); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  function load(p: Preset): void { Object.assign(st, DEFAULT, p.s); sync(); run(); }
  function sync(): void { sigS.set(st.sigma); rS.set(st.r); hS.set(st.h); nS.set(st.noise); modeC.checked = st.mode === "hmax"; qC.checked = st.quant; }

  const pick = (ev: PointerEvent, p: Plot) => {
    const [x] = p.toData(ev);
    let b = 0; SPOTS.forEach((s, i) => { if (Math.abs(s.x - x) < Math.abs(SPOTS[b].x - x)) b = i; });
    st.sel = b; run();
  };
  pRaw.svg.addEventListener("pointerdown", ev => pick(ev, pRaw));
  pF.svg.addEventListener("pointerdown", ev => pick(ev, pF));

  function clean(): Float64Array {
    const f = new Float64Array(L);
    for (let i = 0; i < L; i++) {
      let v = BASE + ramp(i);
      for (const s of SPOTS) v += s.a * Math.sqrt(2 * Math.PI) * S_PSF * pixelFraction(i, s.x, S_PSF);
      f[i] = v;
    }
    return f;
  }

  function run(): void {
    const c = clean(), r = rng(1000 + st.seed), f = new Float64Array(L);
    for (let i = 0; i < L; i++) { f[i] = c[i] + st.noise * r.norm(); if (st.quant) f[i] = Math.round(f[i]); }
    const resp = st.filter === "atrous" ? atrousProduct1d(f, [2, 3], 1) : filter1d(f, st.filter, st.sigma);
    raw.set(Array.from({ length: L }, (_, i) => i), f);
    let hi = 0; for (const v of f) hi = Math.max(hi, v);
    pRaw.setDomain(undefined, [0, Math.max(60, hi * 1.12)]);
    tRaw.set(4, Math.max(60, hi * 1.12) * 0.93, "raw profile (truth: green ticks)");
    let lo = 0, rhi = 0; for (const v of resp) { lo = Math.min(lo, v); rhi = Math.max(rhi, v); }
    const pad = 0.1 * (rhi - lo || 1);
    pF.setDomain(undefined, [lo - pad, rhi + 3.5 * pad]);
    filt.set(Array.from({ length: L }, (_, i) => i), resp);
    // candidates
    let idx: number[];
    if (st.mode === "hmax") {
      const pr = prominences(resp);
      bars.set(pr.map(p => ({ i: p.i, v: resp[p.i], saddle: p.saddle, keep: p.prominence >= st.h && resp[p.i] > 0 })));
      idx = pr.filter(p => p.prominence >= st.h && resp[p.i] > 0).map(p => p.i);
    } else {
      bars.set([]);
      idx = localMaxima1d(resp, st.r, 0);
    }
    cands.set(idx, idx.map(i => resp[i]));
    // selected spot
    const s = SPOTS[st.sel];
    selRaw.set(s.x); selF.set(s.x);
    let near = -1; for (const i of idx) if (near < 0 || Math.abs(i - s.x) < Math.abs(near - s.x)) near = i;
    bracket.set(st.mode === "window" && near >= 0 && Math.abs(near - s.x) <= 3 ? near : -1, st.r, near >= 0 ? resp[near] : 0);
    tF.set(4, rhi + 3.0 * pad, `${st.filter === "atrous" ? "à trous product (levels 2·3, k = 1)" : SPOT_FILTER_LABELS[st.filter] + (st.filter === "none" ? "" : `, σ = ${st.sigma.toFixed(1)}`)}: ${idx.length} candidates`);
    // kernel
    const k = kernel1d(st.filter === "atrous" ? "none" : st.filter, st.sigma), R = k.length >> 1;
    const kx = Array.from(k, (_, i) => i - R);
    let km = 0; for (const v of k) km = Math.max(km, Math.abs(v));
    const ks = Array.from(k, v => v / (km || 1));
    pK.setDomain([-Math.max(12, R), Math.max(12, R)], [Math.min(-0.25, ...ks) - 0.05, 1.1]);
    kLine.set(kx, ks); kStem.set(kx, ks);
    // SNR of the selected spot (noise-free filtered peak / filtered noise sd)
    const respClean = st.filter === "atrous" ? null : filter1d(c, st.filter, st.sigma);
    const kk = st.filter === "atrous" ? null : kernel1d(st.filter, st.sigma);
    const gain = kk ? filterResponseNoise(kk) : NaN;
    const xi = Math.round(s.x);
    let peak = -Infinity; if (respClean) for (let i = xi - 2; i <= xi + 2; i++) peak = Math.max(peak, respClean[i] - (st.filter === "gauss" || st.filter === "none" ? localBase(respClean, xi) : 0));
    const snr1 = st.noise > 0 ? s.a / st.noise : Infinity;
    const snrF = respClean && st.noise > 0 && st.filter !== "none" ? peak / (st.noise * gain) : NaN;
    const plateau = idx.filter(i => i + 1 < L && resp[i + 1] === resp[i]).length;
    ro.textContent =
      `candidates: ${idx.length}  (true spots: ${SPOTS.length})\n` +
      `selected: ${s.name}, x = ${s.x}, amplitude ${s.a}\n` +
      `single-pixel SNR ${Number.isFinite(snr1) ? snr1.toFixed(1) : "∞"}` + (Number.isFinite(snrF) ? `, filtered SNR ${snrF.toFixed(1)}` : "") +
      (plateau ? `\nflat-topped maxima: ${plateau} (one pixel kept per plateau)` : "");
    FILTERS.forEach((ff, i) => filtBtns[i].classList.toggle("active", ff === st.filter));
  }
  /** Local base level for filters that keep the background (Gaussian, none): the ramp/base under the spot. */
  function localBase(resp: Float64Array, xi: number): number {
    const a = resp[Math.max(0, xi - 9)], b = resp[Math.min(L - 1, xi + 9)];
    return Math.min(a, b);
  }

  if (p0 && shown.includes(p0)) presetBtns[shown.indexOf(p0)].classList.add("active");
  run();
}

autoMount(NAME, mount);
