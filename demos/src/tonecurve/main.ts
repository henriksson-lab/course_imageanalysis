// Tone curves: point operations as curves (raw value -> display value). Left: the image
// through the current curve (click a pixel to follow it through the curve). Middle: the
// curve over the input histogram (log counts, shaded), and the output histogram below
// (linear counts: gaps show as a comb). Modes: window, gamma, log, S-curve, free curve
// (5 draggable points, monotone or not), negative, threshold. Sources: the course
// micrograph (8 bit), the synthetic 12-bit fluorescence image, and the same image reduced
// to 8 bits (the comb appears only there).
// Auto-mounts on <div data-demo="tonecurve">. Options: data-preset (window|gamma05|gamma2|
// log|comb8|comb12|scurve|threshold|falling), data-source (cell|fluo12|fluo8).
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, draggable } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import * as C from "../lib/contrast";

const NAME = "tonecurve";
const P = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${P} { gap: .8em; }
${P} figure { margin: 0; }
${P} figcaption { font-size: 16px; text-align: center; margin: 0 0 .2em; }
${P} .tc-img { flex: 0 1 300px; min-width: 220px; }
${P} .tc-img canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; cursor: crosshair; touch-action: none; }
${P} .tc-mid { flex: 1 1 380px; min-width: 280px; max-width: 460px; display: flex; flex-direction: column; gap: .1em; }
${P} .qb-panel { flex: 0 1 300px; gap: .3em; }
${P} .qb-readout { font-size: 15px; }
${P} .tc-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
${P} .tc-credit { font-size: 12px; color: var(--qb-muted); }
${P} .tc-handle { fill: var(--qb-bg, #fff); stroke: var(--qb-c2, #d62728); stroke-width: 3; cursor: grab; }
.reveal ${P} { font-size: 18px; }
.reveal ${P} figcaption, .reveal ${P} .qb-figure figcaption { font-size: 17px; }
.reveal ${P} .qb-btn { font-size: 16px; padding: .05em .4em; }
.reveal ${P} .tc-lab { font-size: 15px; }
.reveal ${P} .qb-panel { gap: .15em; }
.reveal ${P} .qb-readout { font-size: 16px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

type Source = "cell" | "fluo12" | "fluo8";
type Mode = "window" | "gamma" | "log" | "sigmoid" | "free" | "negative" | "threshold";
interface State { source: Source; mode: Mode; lo: number; hi: number; gamma: number; centre: number; width: number; t: number; pts: number[]; monotone: boolean; px: number; py: number }

const MODES: { id: Mode; label: string }[] = [
  { id: "window", label: "window" }, { id: "gamma", label: "gamma" }, { id: "log", label: "log" },
  { id: "sigmoid", label: "S-curve" }, { id: "free", label: "free curve" },
  { id: "negative", label: "negative" }, { id: "threshold", label: "threshold" },
];
const SOURCES: { id: Source; label: string; L: number }[] = [
  { id: "cell", label: "micrograph, 8 bit", L: 256 },
  { id: "fluo12", label: "fluorescence, 12 bit", L: 4096 },
  { id: "fluo8", label: "the same as 8 bit", L: 256 },
];
const BASE: State = { source: "fluo12", mode: "window", lo: 130, hi: 3176, gamma: 1, centre: 128, width: 25, t: 110, pts: [0, 64, 128, 191, 255], monotone: true, px: 160, py: 95 };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "window", label: "the window is a straight line", s: {} },
  { id: "gamma05", label: "γ = 0.5 lifts the darks", s: { mode: "gamma", lo: 100, hi: 4095, gamma: 0.5, px: 200, py: 30 } },
  { id: "gamma2", label: "γ = 2 darkens", s: { mode: "gamma", lo: 100, hi: 4095, gamma: 2, px: 200, py: 30 } },
  { id: "log", label: "log: every decade gets 70 levels", s: { mode: "log", lo: 0, hi: 4095, px: 200, py: 30 } },
  { id: "comb8", label: "stretch 8 bits ×4: the comb", s: { source: "fluo8", lo: 6, hi: 70, px: 200, py: 30 } },
  { id: "comb12", label: "the same stretch on 12 bits", s: { source: "fluo12", lo: 96, hi: 1120, px: 200, py: 30 } },
  { id: "scurve", label: "S-curve", s: { source: "cell", mode: "sigmoid", lo: 0, hi: 255, centre: 120, width: 18, px: 128, py: 128 } },
  { id: "threshold", label: "threshold: the steepest curve", s: { source: "cell", mode: "threshold", lo: 0, hi: 255, t: 110, px: 128, py: 128 } },
  { id: "falling", label: "a falling part reverses the order", s: { source: "cell", mode: "free", lo: 0, hi: 255, pts: [0, 170, 255, 110, 20], monotone: false, px: 128, py: 128 } },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...BASE, pts: [...BASE.pts] };
  const N = C.SCENE_N;

  // ---- layout
  const fig = htmlEl("figure", "tc-img", "", host);
  const cap = htmlEl("figcaption", "", "", fig);
  const cv = htmlEl("canvas", "", "", fig);
  cv.width = N; cv.height = N;
  const credit = htmlEl("div", "tc-credit", CELL_CREDIT, fig);
  const ro = htmlEl("div", "qb-readout", "", fig);

  const mid = htmlEl("div", "tc-mid", "", host);
  const curve = new Plot(mid, { width: 460, height: 300, x: [0, 255], y: [0, 255], xlabel: "raw value", ylabel: "display level", yticks: [0, 64, 128, 192, 255] });
  const outH = new Plot(mid, { width: 460, height: 165, x: [0, 255], y: [0, 1], xlabel: "display level (output histogram, √count)", grid: false, yticks: [], margin: { l: 66 } });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "tc-lab", "presets", panel);
  const pBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "tc-lab", "curve", panel);
  const mRow = htmlEl("div", "qb-presets", "", panel);
  const mBtns = MODES.map(m => button(mRow, m.label, () => { st.mode = m.id; sync(); update(); }));
  const loS = slider(panel, "low", 0, 255, 1, st.lo, v => { st.lo = Math.min(v, st.hi - 1); update(); });
  const hiS = slider(panel, "high", 0, 255, 1, st.hi, v => { st.hi = Math.max(v, st.lo + 1); update(); });
  const gS = slider(panel, "γ", -0.7, 0.7, 0.01, 0, v => { st.gamma = 10 ** v; update(); }, v => (10 ** v).toFixed(2));
  const cS = slider(panel, "centre", 0, 255, 1, st.centre, v => { st.centre = v; update(); });
  const wS = slider(panel, "width", 2, 80, 1, st.width, v => { st.width = v; update(); });
  const tS = slider(panel, "threshold", 0, 255, 1, st.t, v => { st.t = v; update(); });
  const monoCb = checkbox(panel, "keep the curve rising", st.monotone, v => { st.monotone = v; if (v) for (let i = 1; i < 5; i++) st.pts[i] = Math.max(st.pts[i], st.pts[i - 1]); update(); });
  const monoLab = monoCb.parentElement as HTMLElement;
  htmlEl("div", "tc-lab", "image", panel);
  const sRow = htmlEl("div", "qb-presets", "", panel);
  const sBtns = SOURCES.map(s => button(sRow, s.label, () => { setSource(s.id, true); }));
  button(panel, "reset", () => load(p0));

  // ---- images
  let cell: Float32Array | null = null;
  const fluo12 = Float32Array.from(C.fluorescence12());
  const fluo8 = Float32Array.from(fluo12, v => Math.round(v / 16));
  const L = () => SOURCES.find(s => s.id === st.source)!.L;
  const data = (): Float32Array => st.source === "cell" ? (cell ?? new Float32Array(N * N)) : st.source === "fluo12" ? fluo12 : fluo8;

  // ---- curve handles for the free mode (persistent elements, so dragging survives redraws)
  const handles: SVGCircleElement[] = [];
  const hx = (k: number) => k * (L() - 1) / 4;
  for (let k = 0; k < 5; k++) {
    const c = svgEl("circle", { r: 11, class: "tc-handle" }, curve.over);
    draggable(c, ev => curve.toData(ev), (_x, y) => {
      st.pts[k] = Math.max(0, Math.min(255, y));
      if (st.monotone) {
        for (let i = k + 1; i < 5; i++) st.pts[i] = Math.max(st.pts[i], st.pts[k]);
        for (let i = k - 1; i >= 0; i--) st.pts[i] = Math.min(st.pts[i], st.pts[k]);
      }
      update();
    });
    handles.push(c);
  }

  function curveFn(): C.Curve {
    const l = L();
    switch (st.mode) {
      case "window": return C.windowCurve(st.lo, st.hi);
      case "gamma": return C.gammaCurve(st.gamma, st.lo, st.hi);
      case "log": return C.logCurve("fit", st.lo, st.hi);
      case "sigmoid": return C.sigmoidCurve(st.centre, st.width, st.lo, st.hi);
      case "free": return C.piecewiseLinear(st.pts.map((y, k) => [hx(k), y] as [number, number]), st.monotone);
      case "negative": return C.negativeCurve(l);
      case "threshold": return C.thresholdCurve(st.t);
    }
  }

  function setSource(s: Source, keepCurve: boolean): void {
    const oldL = L();
    st.source = s;
    const l = L();
    if (keepCurve && oldL !== l) {   // rescale the parameters to the new range
      const f = (l - 1) / (oldL - 1);
      st.lo = Math.round(st.lo * f); st.hi = Math.round(st.hi * f); st.centre = Math.round(st.centre * f);
      st.width = Math.max(2, Math.round(st.width * f)); st.t = Math.round(st.t * f);
    }
    if (s === "cell" && !cell) { loadCell().then(g => { cell = Float32Array.from(g.data, v => Math.round(v)); sync(); update(); }); }
    sync(); update();
  }

  function sync(): void {
    const l = L();
    for (const s of [loS, hiS, cS, tS]) s.input.max = String(l - 1);
    wS.input.max = String(Math.round(80 * (l - 1) / 255));
    loS.set(st.lo); hiS.set(st.hi); gS.set(Math.log10(st.gamma)); cS.set(st.centre); wS.set(st.width); tS.set(st.t);
    monoCb.checked = st.monotone;
    const show = (e: HTMLElement, on: boolean) => { e.style.display = on ? "" : "none"; };
    const ranged = st.mode === "window" || st.mode === "gamma" || st.mode === "log" || st.mode === "sigmoid";
    show(loS.input.parentElement!, ranged); show(hiS.input.parentElement!, ranged);
    show(gS.input.parentElement!, st.mode === "gamma");
    show(cS.input.parentElement!, st.mode === "sigmoid"); show(wS.input.parentElement!, st.mode === "sigmoid");
    show(tS.input.parentElement!, st.mode === "threshold");
    show(monoLab, st.mode === "free");
    mBtns.forEach((b, i) => b.classList.toggle("active", MODES[i].id === st.mode));
    sBtns.forEach((b, i) => b.classList.toggle("active", SOURCES[i].id === st.source));
    credit.style.display = st.source === "cell" ? "" : "none";
    handles.forEach(h => (h.style.display = st.mode === "free" ? "" : "none"));
  }

  function load(p: typeof PRESETS[number]): void {
    Object.assign(st, BASE, { pts: [...BASE.pts] }, p.s);
    if (p.s.pts) st.pts = [...p.s.pts];
    pBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    setSource(st.source, false);
  }

  // ---- click: follow a pixel through the curve
  const pick = (ev: PointerEvent) => {
    const r = cv.getBoundingClientRect();
    st.px = Math.max(0, Math.min(N - 1, Math.floor((ev.clientX - r.left) / r.width * N)));
    st.py = Math.max(0, Math.min(N - 1, Math.floor((ev.clientY - r.top) / r.height * N)));
    update();
  };
  let down = false;
  cv.addEventListener("pointerdown", ev => { ev.preventDefault(); down = true; cv.setPointerCapture(ev.pointerId); pick(ev); });
  cv.addEventListener("pointermove", ev => { if (down) pick(ev); });
  cv.addEventListener("pointerup", () => { down = false; });
  cv.addEventListener("pointercancel", () => { down = false; });

  // ---- compute and draw
  const ctx = cv.getContext("2d")!;
  const imgData = ctx.createImageData(N, N);
  function update(): void {
    const l = L(), d = data();
    const lut = C.lutFromFunction(curveFn(), l);
    const lut8 = lut.map(v => Math.max(0, Math.min(255, Math.round(v))));
    const out = C.applyLut(d, lut8);
    // image with the picked pixel marked
    for (let i = 0; i < N * N; i++) {
      const g = out[i];
      imgData.data[4 * i] = imgData.data[4 * i + 1] = imgData.data[4 * i + 2] = g; imgData.data[4 * i + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
    ctx.strokeStyle = "#ff7f0e"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(st.px + 0.5, st.py + 0.5, 6, 0, 2 * Math.PI); ctx.stroke();
    cap.textContent = st.mode === "free" ? "drag the red points on the curve" : "click a pixel to follow it";

    // curve plot: input histogram (log counts) as a shaded background
    curve.opts.xticks = l > 256 ? [0, 1000, 2000, 3000, 4000] : [0, 64, 128, 192, 255];
    curve.setDomain([0, l - 1], [0, 255]);
    curve.clear();
    const nb = 256, bw = l / nb, hin = new Float64Array(nb);
    for (let i = 0; i < d.length; i++) hin[Math.min(nb - 1, Math.floor(d[i] / bw))]++;
    const hmax = Math.log1p(Math.max(...hin));
    const edges = Array.from({ length: nb + 1 }, (_, k) => k * bw - 0.5);
    curve.histogram(edges, Array.from(hin, c => 230 * Math.log1p(c) / hmax), { color: "var(--qb-muted, #999)", fillOpacity: 0.3, fill: "var(--qb-muted, #999)", width: 0 });
    const xs = Array.from({ length: l }, (_, v) => v);
    curve.line(xs, Array.from(lut), { color: seriesColor(0), width: 3.5 });
    const v = Math.round(d[st.py * N + st.px]), s = lut8[v];
    curve.line([v, v, 0], [0, s, s], { color: seriesColor(3), width: 2.5, dash: "6 5" });
    curve.scatter([v], [s], { color: seriesColor(3), r: 7 });
    handles.forEach((h, k) => { h.setAttribute("cx", String(curve.x(hx(k)))); h.setAttribute("cy", String(curve.y(st.pts[k]))); });

    // output histogram (linear counts, 256 bins)
    const hout = C.levelHistogram(out, 256);
    const omax = Math.sqrt(Math.max(1, ...hout));
    outH.setDomain([-0.5, 255.5], [0, omax * 1.05]);
    outH.clear();
    outH.histogram(Array.from({ length: 257 }, (_, k) => k - 0.5), Array.from(hout, Math.sqrt), { color: seriesColor(0), fill: seriesColor(0), fillOpacity: 0.85, width: 0 });

    // readouts
    let at0 = 0, at255 = 0;
    for (let i = 0; i < out.length; i++) { if (out[i] === 0) at0++; else if (out[i] === 255) at255++; }
    const slope = C.slopeOf(lut, v), gain = slope * (l - 1) / 255;
    const usedIn = C.distinctCount(d), usedOut = C.distinctCount(out);
    ro.textContent = [
      `pixel (${st.px}, ${st.py}): raw ${v} → display ${s}`,
      `slope there: ×${gain.toFixed(2)} contrast`,
      `levels used: ${usedIn} in → ${usedOut} out`,
      `at black ${(100 * at0 / out.length).toFixed(1)} %, at white ${(100 * at255 / out.length).toFixed(1)} %`,
    ].join("\n");
  }

  load(p0);
  if (host.dataset.source && SOURCES.some(s => s.id === host.dataset.source)) setSource(host.dataset.source as Source, true);
}

autoMount(NAME, mount);
