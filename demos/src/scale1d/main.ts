// 1D scale space. A signal of 256 samples (three box bumps of half-widths 2, 6 and 18; a
// step and a ramp; a bright line in a dark band; a line through the micrograph). Top: the
// signal and its blur at the current σ. Middle: the scale-space image (x across, σ up on a
// log axis, 64 levels) of L, σ·Lx or −σ²·Lxx (raw without "normalise"), with the extrema
// of L, the edges (extrema of Lx) or the zero crossings of Lxx traced as dots, and the
// current σ as a line. Bottom: the response at the chosen x against σ, normalised and raw,
// with its maximum and, on a bump, the predicted σ = half-width. Kernel: Gaussian or a box
// of the same standard deviation (the box can create new extrema).
//
// <div data-demo="scale1d" data-preset="bumps|raw|norm|curves|box|noise"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { gaussianKernel } from "../lib/imageops";
import { correlate1d, boxKernel1d, geomScales } from "../lib/edges";
import { diverging, grey } from "../lib/colour";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { makeSignal, BUMPS, N } from "./signals";
import { injectStyle } from "../canny/imgview";

const NAME = "scale1d";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; flex-wrap: nowrap; }
[data-demo="${NAME}"] .s1-plots { flex: 1 1 640px; min-width: 0; max-width: 760px; display: flex; flex-direction: column; gap: 0; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 330px; gap: .4em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .s1-plots { flex: 0 0 700px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 1 1 400px; gap: .22em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .08em .4em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .25em .5em; }
@media (max-width: 760px) { [data-demo="${NAME}"] { flex-wrap: wrap; } }
`;

type Sig = "bumps" | "stepramp" | "boxdemo" | "cell";
type Qty = 0 | 1 | 2;
interface State { signal: Sig; noise: number; sigma: number; qty: Qty; norm: boolean; box: boolean; x: number }
interface Preset { id: string; label: string; s: Partial<State> }
const DEFAULT: State = { signal: "bumps", noise: 0, sigma: 6, qty: 2, norm: true, box: false, x: 99 };
const PRESETS: Preset[] = [
  { id: "bumps", label: "three bumps, three scales", s: { qty: 0, sigma: 4, x: 99 } },
  { id: "raw", label: "raw LoG prefers small σ", s: { norm: false, x: 99, sigma: 1 } },
  { id: "norm", label: "normalised: each bump picks its σ", s: { norm: true, x: 179, sigma: 18 } },
  { id: "curves", label: "edges and bumps as curves", s: { signal: "stepramp", qty: 1, sigma: 4, x: 70 } },
  { id: "box", label: "box kernel creates structure", s: { signal: "boxdemo", qty: 0, box: true, sigma: 8, x: 127 } },
  { id: "noise", label: "noise lives at small σ", s: { noise: 10, qty: 0, sigma: 1, x: 39 } },
];
const LEVELS = 64, S0 = 0.5, S1 = 32;
const SIGMAS = geomScales(S0, S1, LEVELS), Q = Math.pow(S1 / S0, 1 / (LEVELS - 1));
const QTY_NAMES = ["L (blurred signal)", "slope σ·Lx", "curvature −σ²·Lxx"];
const QTY_RAW = ["L (blurred signal)", "slope Lx", "curvature −Lxx"];
/** Positions of the local extrema of a (a flat top or bottom counts once, at its middle). */
function extremaAt(a: ArrayLike<number>, eps: number): number[] {
  const out: number[] = [];
  let prevSign = 0, prevAt = 0;
  for (let x = 1; x < a.length; x++) {
    const d = a[x] - a[x - 1], sg = d > eps ? 1 : d < -eps ? -1 : 0;
    if (!sg) continue;
    if (prevSign && sg !== prevSign) out.push(Math.round((prevAt + x - 1) / 2));
    prevSign = sg; prevAt = x;
  }
  return out;
}
const nearestLevel = (s: number) => Math.max(0, Math.min(LEVELS - 1, Math.round(Math.log(s / S0) / Math.log(Q))));

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(p0?.s ?? {}) };
  let cellRow: Float64Array | null = null;

  const plots = htmlEl("div", "s1-plots", "", host);
  const pA = new Plot(plots, { width: 720, height: 150, x: [-0.5, N - 0.5], y: [0, 1], xticks: [], margin: { l: 62, r: 10, t: 6, b: 6 } });
  const pS = new Plot(plots, { width: 720, height: 210, x: [-0.5, N - 0.5], y: [S0 / Math.sqrt(Q), S1 * Math.sqrt(Q)], ylog: true, xticks: 8, grid: false, margin: { l: 62, r: 10, t: 4, b: 30 }, ylabel: "σ" });
  const pR = new Plot(plots, { width: 720, height: 170, x: [S0, S1], xlog: true, y: [0, 1], margin: { l: 62, r: 10, t: 6, b: 48 }, xlabel: "scale σ (pixels)" });
  const raw = pA.line([], [], { color: "#999", width: 1.5 });
  const blur = pA.line([], [], { color: seriesColor(0), width: 3 });
  const xA = pA.vline(0, { color: "var(--qb-c4)", width: 2, dash: "5 4" });
  const tA = pA.text(4, 0, "", { anchor: "start", size: 17 });
  let heat: { remove(): void; g: SVGGElement } | null = null;
  const dots = pS.scatter([], [], { color: "#111", r: 1.6 });
  const sLine = pS.hline(1, { color: "var(--qb-c4)", width: 2.5, dash: "6 4" });
  const xS = pS.vline(0, { color: "var(--qb-c4)", width: 2, dash: "5 4" });
  const curveRaw = pR.line([], [], { color: "#999", width: 2.5, label: "raw" });
  const curveN = pR.line([], [], { color: seriesColor(1), width: 3, label: "normalised" });
  const peak = pR.vline(1, { color: seriesColor(1), width: 2, dash: "4 3" });
  const pred = pR.vline(1, { color: seriesColor(2), width: 2.5, dash: "2 3", text: "a" });
  const sR = pR.vline(1, { color: "var(--qb-c4)", width: 2, dash: "6 4" });
  const tR = pR.text(0, 0, "", { anchor: "start", size: 17 });

  pA.svg.addEventListener("pointerdown", ev => { st.x = Math.max(0, Math.min(N - 1, Math.round(pA.toData(ev)[0]))); draw(); });
  pS.svg.addEventListener("pointerdown", ev => {
    const [x, s] = pS.toData(ev);
    st.x = Math.max(0, Math.min(N - 1, Math.round(x)));
    if (s > S0 * 0.9 && s < S1 * 1.1) { st.sigma = SIGMAS[nearestLevel(s)]; sigS.set(uOf(st.sigma)); }
    draw();
  });
  pR.svg.addEventListener("pointerdown", ev => { const s = pR.toData(ev)[0]; st.sigma = SIGMAS[nearestLevel(s)]; sigS.set(uOf(st.sigma)); draw(); });
  for (const p of [pA, pS, pR]) p.svg.style.cursor = "crosshair";

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ev-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); st.sigma = SIGMAS[nearestLevel(st.sigma)]; sync(); compute(); } })));
  htmlEl("div", "ev-lab", "signal", panel);
  const sigIds: Sig[] = ["bumps", "stepramp", "boxdemo", "cell"];
  const sigBtns = presets(panel, ["three bumps", "step + ramp", "line in dark band", "micrograph"].map((l, i) => ({ label: l, apply: () => { st.signal = sigIds[i]; compute(); } })));
  const uOf = (s: number) => nearestLevel(s);
  const sigS = slider(panel, "σ", 0, LEVELS - 1, 1, uOf(st.sigma), u => { st.sigma = SIGMAS[u]; draw(); }, u => SIGMAS[u].toFixed(2));
  const noiseS = slider(panel, "noise", 0, 30, 1, st.noise, v => { st.noise = v; compute(); });
  htmlEl("div", "ev-lab", "scale-space image of", panel);
  const qBtns = presets(panel, ["L", "slope", "curvature"].map((l, i) => ({ label: l, apply: () => { st.qty = i as Qty; draw(); } })));
  const normC = checkbox(panel, "normalise (× σ per derivative)", st.norm, v => { st.norm = v; draw(); });
  const boxC = checkbox(panel, "box kernel instead of Gaussian", st.box, v => { st.box = v; compute(); });
  button(panel, "reset", () => { Object.assign(st, DEFAULT, p0?.s ?? {}); st.sigma = SIGMAS[nearestLevel(st.sigma)]; presetBtns.forEach(b => b.classList.remove("active")); sync(); compute(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  const credit = htmlEl("div", "ev-credit", CELL_CREDIT, panel);

  function sync(): void { sigS.set(uOf(st.sigma)); noiseS.set(st.noise); normC.checked = st.norm; boxC.checked = st.box; }

  // per level: L, Lx, Lxx (raw)
  let f: Float64Array = new Float64Array(N);
  const L: Float64Array[] = [], Lx: Float64Array[] = [], Lxx: Float64Array[] = [];
  function compute(): void {
    if (st.signal === "cell" && !cellRow) { loadCell(256).then(img => { cellRow = Float64Array.from(img.data.subarray(150 * 256, 151 * 256)); compute(); }); return; }
    if (st.signal === "cell") { f = cellRow!.slice(); if (st.noise) { const a = makeSignal("bumps", st.noise, 1), b = makeSignal("bumps", 0); for (let i = 0; i < N; i++) f[i] += a[i] - b[i]; } }
    else f = makeSignal(st.signal, st.noise, 1);
    for (let k = 0; k < LEVELS; k++) {
      const s = SIGMAS[k];
      if (st.box) {
        const l = correlate1d(f, boxKernel1d(s));
        L[k] = l; Lx[k] = correlate1d(l, [-0.5, 0, 0.5]); Lxx[k] = correlate1d(l, [1, -2, 1]);
      } else {
        L[k] = correlate1d(f, gaussianKernel(s, 0)); Lx[k] = correlate1d(f, gaussianKernel(s, 1)); Lxx[k] = correlate1d(f, gaussianKernel(s, 2));
      }
    }
    draw();
  }

  /** The displayed quantity at level k, position x. */
  const val = (k: number, x: number, norm = st.norm): number => {
    const s = SIGMAS[k];
    if (st.qty === 0) return L[k][x];
    if (st.qty === 1) return (norm ? s : 1) * Lx[k][x];
    return -(norm ? s * s : 1) * Lxx[k][x];
  };

  const xs = Float64Array.from({ length: N }, (_, i) => i);
  function draw(): void {
    sigBtns.forEach((b, i) => b.classList.toggle("active", sigIds[i] === st.signal));
    qBtns.forEach((b, i) => b.classList.toggle("active", i === st.qty));
    credit.style.display = st.signal === "cell" ? "" : "none";
    const k = nearestLevel(st.sigma);
    // top
    let lo = Infinity, hi = -Infinity;
    for (const v of f) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const pad = 0.08 * (hi - lo || 1);
    pA.setDomain(undefined, [lo - pad, hi + 3.2 * pad]);
    raw.set(xs, f); blur.set(xs, L[k]); xA.set(st.x);
    tA.set(4, hi + 2.2 * pad, `signal (grey) and its ${st.box ? "box" : "Gaussian"} blur at σ = ${st.sigma.toFixed(2)} (blue)`);
    // scale-space image
    const V = new Float64Array(N * LEVELS);
    let m = 0, vlo = Infinity, vhi = -Infinity;
    for (let j = 0; j < LEVELS; j++) for (let x = 0; x < N; x++) {
      const v = val(j, x); V[j * N + x] = v;
      m = Math.max(m, Math.abs(v)); vlo = Math.min(vlo, v); vhi = Math.max(vhi, v);
    }
    heat?.remove();
    heat = pS.heatmap(V, N, LEVELS, { x0: -0.5, x1: N - 0.5, y0: S0 / Math.sqrt(Q), y1: S1 * Math.sqrt(Q), flipY: false,
      ...(st.qty === 0 ? { map: grey, lo: vlo, hi: vhi } : { map: diverging, lo: -m, hi: m }) });
    pS.layer.insertBefore(heat.g, pS.layer.firstChild);
    // traced features: extrema of L (qty 0), extrema of Lx = edges (qty 1), zero crossings of Lxx (qty 2)
    const dx: number[] = [], dy: number[] = [];
    for (let j = 0; j < LEVELS; j++) {
      const a = st.qty === 0 ? L[j] : st.qty === 1 ? Lx[j] : Lxx[j];
      const amp = st.qty === 2 ? Math.max(...Array.from(a, Math.abs)) : 0;
      if (st.qty === 2) {
        for (let x = 1; x < N - 1; x++)
          if (((a[x] < 0 && a[x + 1] >= 0) || (a[x] >= 0 && a[x + 1] < 0)) && Math.abs(a[x] - a[x + 1]) > 0.02 * amp) { dx.push(x + 0.5); dy.push(SIGMAS[j]); }
      } else for (const x of extremaAt(a, 1e-6 * (hi - lo || 1))) { dx.push(x); dy.push(SIGMAS[j]); }
    }
    dots.style.color = st.qty === 0 ? "#ffd400" : "#111";
    dots.set(dx, dy);
    sLine.set(st.sigma); xS.set(st.x);
    // response against σ at x
    const rN = SIGMAS.map((_, j) => val(j, st.x, true)), rR = SIGMAS.map((_, j) => val(j, st.x, false));
    let rlo = 0, rhi = 0;
    for (const v of [...rN, ...(st.qty ? rR : [])]) { rlo = Math.min(rlo, v); rhi = Math.max(rhi, v); }
    const rp = 0.1 * (rhi - rlo || 1);
    pR.setDomain(undefined, [rlo - rp, rhi + 2.5 * rp]);
    curveN.set(SIGMAS, rN); curveRaw.set(SIGMAS, st.qty ? rR : []);
    let jb = 0; for (let j = 1; j < LEVELS; j++) if (rN[j] > rN[jb]) jb = j;
    peak.set(st.qty ? SIGMAS[jb] : 1e6);
    const bump = st.signal === "bumps" ? BUMPS.find(b => st.x >= b.c - b.a && st.x < b.c + b.a) : undefined;
    pred.set(bump && st.qty === 2 ? bump.a : 1e6);
    sR.set(st.sigma);
    tR.set(S0 * 1.05, rhi + 1.6 * rp, st.qty ? `${QTY_NAMES[st.qty]} at x = ${st.x} (red), raw (grey)` : `L at x = ${st.x} against σ`);
    const cur = val(k, st.x);
    ro.textContent =
      `kernel: ${st.box ? `box, ${boxKernel1d(st.sigma).length} px wide` : "Gaussian"}, σ = ${st.sigma.toFixed(2)}\n` +
      `at x = ${st.x}: ${(st.norm ? QTY_NAMES : QTY_RAW)[st.qty].split(" (")[0]} = ${cur.toFixed(2)}\n` +
      (st.qty ? `normalised response peaks at σ = ${SIGMAS[jb].toFixed(2)}` : "") +
      (bump && st.qty === 2 ? `\nbump half-width a = ${bump.a} (predicted σ = a)` : "");
  }

  st.sigma = SIGMAS[nearestLevel(st.sigma)];
  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  compute();
}

autoMount(NAME, mount);
