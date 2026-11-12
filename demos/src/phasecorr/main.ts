// Phase correlation on a micrograph (FFT registration day). Two crops of the image: drag
// frame B (red) relative to frame A (blue), degrade B, and see the correlation surface by
// cross-correlation (CC), partial normalisation (exponent p), phase correlation or overlap
// NCC, with the peak, its height and the peak ratio. A brute-force button times the direct
// route. Template mode (data-template="1"): a draggable template box; the NCC map of the
// template over the whole image and its best matches.
// Options: data-preset="<id>", data-method="cc|phase|ncc", data-template="1".
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, svgPoint } from "../lib/ui";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { drawScalarImage, inferno, diverging, hsvToRgb } from "../lib/colour";
import { fftshift } from "../lib/fft";
import { nuclei } from "../lib/synth";
import { fftReal, TemplateMatcher, bruteForceNcc2d, at } from "../lib/registration";
import { DEFAULTS, PCSettings, IMG, makeCrops, register, texture, Method, Result, Crops } from "./model";

const NAME = "phasecorr";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; line-height: 1.2; margin-bottom: .15em; }
[data-demo="${NAME}"] .pc-stack { position: relative; width: 100%; }
[data-demo="${NAME}"] .pc-stack canvas { width: 100%; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .pc-stack svg { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; }
[data-demo="${NAME}"] .pc-img { flex: 0 1 270px; min-width: 200px; }
[data-demo="${NAME}"] .pc-mid { flex: 0 1 230px; min-width: 170px; display: grid; grid-template-columns: 1fr 1fr; gap: .4em; }
[data-demo="${NAME}"] .pc-surf { flex: 0 1 290px; min-width: 210px; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 320px; gap: .3em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .pc-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .pc-credit { font-size: 12px; color: var(--qb-muted); }
[data-demo="${NAME}"] .pc-wide { grid-column: 1 / 3; }
.reveal [data-demo="${NAME}"] { font-size: 16px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .4em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 13px; }
.reveal [data-demo="${NAME}"] .qb-slider { gap: .3em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

interface Preset { id: string; label: string; s: Partial<PCSettings> }
const PRESETS: Preset[] = [
  { id: "delta", label: "same image, shifted: delta", s: { circular: true, taper: false, method: "phase" } },
  { id: "cc", label: "CC: broad peak", s: { method: "cc" } },
  { id: "phase", label: "phase: sharp peak", s: { method: "phase" } },
  { id: "ramp", label: "illumination ramp on B", s: { method: "cc", ramp: 500 } },
  { id: "noise", label: "noise: phase peak drowns", s: { method: "phase", noise: 120 } },
  { id: "lowpass", label: "noise + low-pass weight", s: { method: "phase", noise: 120, lowpass: 0.08 } },
  { id: "notaper", label: "no taper: false (0, 0) peak", s: { method: "cc", grad: 600, taper: false } },
  { id: "periodic", label: "periodic texture", s: { source: "texture", method: "phase" } },
  { id: "ncc", label: "overlap NCC", s: { method: "ncc" } },
];

interface TPreset { id: string; label: string; src: "cell" | "nuclei"; x: number; y: number; size: number; thr: number }
const TPRESETS: TPreset[] = [
  { id: "granules", label: "dark granules", src: "cell", x: 99, y: 59, size: 20, thr: 0.5 },
  { id: "nuclei", label: "nuclei (synthetic)", src: "nuclei", x: -1, y: -1, size: 32, thr: 0.5 },
  { id: "texture", label: "a piece of texture", src: "cell", x: 200, y: 30, size: 20, thr: 0.5 },
];

const f2 = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : "–");

/** A canvas with an SVG overlay whose viewBox is given. */
function stack(parent: Element, caption: string, vb: [number, number, number, number]): { fig: HTMLElement; cv: HTMLCanvasElement; svg: SVGSVGElement; cap: HTMLElement } {
  const fig = htmlEl("figure", "", "", parent);
  const cap = htmlEl("figcaption", "", caption, fig);
  const st = htmlEl("div", "pc-stack", "", fig);
  const cv = htmlEl("canvas", "", "", st);
  const svg = svgEl("svg", { viewBox: vb.join(" "), preserveAspectRatio: "none" }, st);
  return { fig, cv, svg, cap };
}

function rect(svg: SVGSVGElement, col: string, w = 2.5, dash = ""): SVGRectElement {
  const r = svgEl("rect", { fill: "none" }, svg);
  r.style.stroke = col; r.style.strokeWidth = String(w);
  r.setAttribute("vector-effect", "non-scaling-stroke");
  if (dash) r.style.strokeDasharray = dash;
  return r;
}

function mount(host: HTMLElement): void {
  injectStyle();
  if (host.dataset.template === "1") { mountTemplate(host); return; }
  const ids = host.dataset.presets?.split(",").map(x => x.trim());
  const LIST = ids ? PRESETS.filter(p => ids.includes(p.id)) : PRESETS;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? LIST[0];
  const st: PCSettings = { ...DEFAULTS, ...(host.dataset.method ? { method: host.dataset.method as Method } : {}) };
  let view: "spec" | "phase" = "spec";
  let cell: Float64Array | null = null;
  const tex = texture();

  // ---- layout ----
  const left = htmlEl("div", "pc-img", "", host);
  const big = stack(left, "drag frame B (red)", [0, 0, IMG, IMG]);
  big.cv.width = IMG; big.cv.height = IMG;
  const fa = rect(big.svg, "#1f77b4", 3), fb = rect(big.svg, "#d62728", 3);
  fb.style.cursor = "move"; fb.style.pointerEvents = "all"; fb.setAttribute("fill", "rgba(214,39,40,0.08)");
  const credit = htmlEl("div", "pc-credit", CELL_CREDIT, left);
  const mid = htmlEl("div", "pc-mid", "", host);
  const ca = stack(mid, "crop A", [0, 0, 1, 1]), cb = stack(mid, "crop B", [0, 0, 1, 1]);
  const sp = stack(mid, "", [0, 0, 1, 1]);
  sp.fig.classList.add("pc-wide");
  sp.fig.style.maxWidth = "170px"; sp.fig.style.justifySelf = "center"; sp.fig.style.width = "100%";
  const right = htmlEl("div", "pc-surf", "", host);
  const sf = stack(right, "correlation surface", [-64, -64, 128, 128]);
  const panel = htmlEl("div", "qb-panel", "", host);

  // ---- controls ----
  htmlEl("div", "pc-lab", "presets", panel);
  const presetBtns = presets(panel, LIST.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "pc-lab", "method", panel);
  const methods: [Method, string][] = [["cc", "CC"], ["ncc", "NCC"], ["phase", "phase (exponent p)"]];
  const methodBtns = presets(panel, methods.map(([m, l]) => ({ label: l, apply: () => { st.method = m; if (m === "phase" && st.p === 0) st.p = 1; run(); } })));
  const pS = slider(panel, "p", 0, 1, 0.05, 1, v => { st.p = v; st.method = "phase"; run(); });
  const lpS = slider(panel, "low-pass", 0, 0.25, 0.01, 0, v => { st.lowpass = v; run(); }, v => (v ? v.toFixed(2) : "off"));
  const txS = slider(panel, "shift x", -60, 60, 1, st.tx, v => { st.tx = v; run(); });
  const tyS = slider(panel, "shift y", -60, 60, 1, st.ty, v => { st.ty = v; run(); });
  const nS = slider(panel, "noise σ", 0, 160, 5, 0, v => { st.noise = v; run(); });
  const gS = slider(panel, "gain on B", 0.3, 2, 0.1, 1, v => { st.gain = v; run(); });
  const oS = slider(panel, "offset on B", -100, 100, 10, 0, v => { st.offset = v; run(); });
  const rS = slider(panel, "ramp on B", -600, 600, 50, 0, v => { st.ramp = v; run(); });
  const grS = slider(panel, "uneven light", 0, 800, 50, 0, v => { st.grad = v; run(); });
  const row = htmlEl("div", "qb-presets", "", panel);
  const tapC = checkbox(row, "Hann taper", true, v => { st.taper = v; run(); });
  const padC = checkbox(row, "zero-pad", false, v => { st.pad = v; run(); });
  const circC = checkbox(row, "circular copy", false, v => { st.circular = v; run(); });
  htmlEl("div", "pc-lab", "small panel · crop size", panel);
  const vrow = htmlEl("div", "qb-presets", "", panel);
  const viewBtns = [button(vrow, "spectrum of A", () => { view = "spec"; draw(); }), button(vrow, "cross-power phase", () => { view = "phase"; draw(); })];
  const sizeBtns = [64, 128].map(n => button(vrow, String(n), () => { st.size = n; clampFrames(); run(); }));
  const brow = htmlEl("div", "qb-presets", "", panel);
  button(brow, "reset", () => load(p0));
  const bfBtn = button(brow, "brute force (NCC, all shifts)", () => bruteForce());
  const ro = htmlEl("div", "qb-readout", "", panel);
  hideControls(panel, host.dataset.controls);
  const bfOut = htmlEl("div", "qb-readout", "", panel);
  bfOut.style.display = "none";
  htmlEl("div", "qb-note", "Surface: shift 0 in the centre, x to the right, y down. Green ring: the true shift; cyan cross: the highest peak; white circles: the next peaks.", panel);

  // surface markers
  const otherM = svgEl("g", {}, sf.svg);
  const trueM = svgEl("circle", { r: 4, fill: "none" }, sf.svg);
  trueM.style.stroke = "#2ca02c"; trueM.style.strokeWidth = "3"; trueM.setAttribute("vector-effect", "non-scaling-stroke");
  const peakM = svgEl("path", { fill: "none" }, sf.svg);
  peakM.style.stroke = "#00e5ff"; peakM.style.strokeWidth = "3"; peakM.setAttribute("vector-effect", "non-scaling-stroke");

  // drag frame B
  let drag: { x: number; y: number; tx: number; ty: number } | null = null;
  fb.addEventListener("pointerdown", ev => {
    ev.preventDefault(); fb.setPointerCapture(ev.pointerId);
    const [x, y] = svgPoint(big.svg, ev);
    drag = { x, y, tx: st.tx, ty: st.ty };
  });
  fb.addEventListener("pointermove", ev => {
    if (!drag) return;
    const [x, y] = svgPoint(big.svg, ev);
    st.tx = Math.round(drag.tx + x - drag.x); st.ty = Math.round(drag.ty + y - drag.y);
    clampFrames(); presetBtns.forEach(b => b.classList.remove("active")); run();
  });
  const up = () => { drag = null; };
  fb.addEventListener("pointerup", up); fb.addEventListener("pointercancel", up);

  function clampFrames(): void {
    const n = st.size;
    st.ax = Math.max(0, Math.min(IMG - n, st.ax)); st.ay = Math.max(0, Math.min(IMG - n, st.ay));
    st.tx = Math.max(-st.ax, Math.min(IMG - n - st.ax, st.tx, 60)); st.tx = Math.max(st.tx, -60);
    st.ty = Math.max(-st.ay, Math.min(IMG - n - st.ay, st.ty, 60)); st.ty = Math.max(st.ty, -60);
  }

  function load(p: Preset): void {
    Object.assign(st, DEFAULTS, p.s);
    presetBtns.forEach((b, i) => b.classList.toggle("active", LIST[i] === p));
    run();
  }

  let res: Result | null = null, crops: Crops = { a: new Float64Array(0), b: new Float64Array(0) };
  let pending = false;
  function run(): void {
    if (st.source === "cell" && !cell) { loadCell().then(g => { cell = Float64Array.from(g.data); run(); }); return; }
    sync();
    bfOut.style.display = "none";
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      crops = makeCrops(st.source === "cell" ? cell! : tex, st);
      res = register(crops, st);
      draw();
    });
  }

  function sync(): void {
    methodBtns.forEach((b, i) => b.classList.toggle("active", methods[i][0] === st.method));
    pS.set(st.method === "cc" ? 0 : st.p); lpS.set(st.lowpass); txS.set(st.tx); tyS.set(st.ty);
    nS.set(st.noise); gS.set(st.gain); oS.set(st.offset); rS.set(st.ramp); grS.set(st.grad);
    tapC.checked = st.taper; padC.checked = st.pad; circC.checked = st.circular;
    viewBtns.forEach((b, i) => b.classList.toggle("active", ["spec", "phase"][i] === view));
    sizeBtns.forEach((b, i) => b.classList.toggle("active", [64, 128][i] === st.size));
    const spectral = st.method !== "ncc";
    [pS, lpS].forEach(s => { s.input.disabled = !spectral; });
    tapC.disabled = !spectral; padC.disabled = !spectral;
    credit.style.visibility = st.source === "cell" ? "visible" : "hidden";
  }

  function draw(): void {
    if (!res) return;
    const n = st.size, img = st.source === "cell" ? cell! : tex;
    drawScalarImage(big.cv, img.map((v, i) => v + st.grad * ((i % IMG) + Math.floor(i / IMG)) / 510), IMG, IMG, { lo: 0, hi: 255 + st.grad });
    setRect(fa, st.ax, st.ay, n); setRect(fb, st.ax + st.tx, st.ay + st.ty, n);
    big.cap.textContent = st.circular ? "B = A rolled around (circular copy)" : "drag frame B (red)";
    drawScalarImage(ca.cv, crops.a, n, n, { lo: 0, hi: 255 + st.grad });
    drawScalarImage(cb.cv, crops.b, n, n, { lo: 0, hi: 255 + st.grad });
    // small panel
    if (st.method === "ncc" || !res.R || !res.taperedA) {
      sp.cap.textContent = "(no spectrum for NCC)";
      drawScalarImage(sp.cv, new Float64Array(4), 2, 2, { lo: 0, hi: 1 });
    } else if (view === "spec") {
      sp.cap.textContent = st.taper ? "log |spectrum| of A (tapered)" : "log |spectrum| of A";
      const A = fftReal(res.taperedA, n, n), m = new Float64Array(n * n);
      for (let k = 0; k < n * n; k++) m[k] = Math.log1p(Math.hypot(A.re[k], A.im[k]));
      drawScalarImage(sp.cv, fftshift(m, n, n), n, n, { map: inferno });
    } else {
      sp.cap.textContent = "phase of A·conj(B) as hue";
      const R = res.R, W = res.surface.w, H = res.surface.h, id = new ImageData(W, H);
      const sh = new Float64Array(W * H);
      for (let k = 0; k < W * H; k++) sh[k] = Math.atan2(R.im[k], R.re[k]);
      const s2 = fftshift(sh, W, H);
      for (let k = 0; k < W * H; k++) {
        const [r, g, b] = hsvToRgb([(s2[k] * 180 / Math.PI + 360) % 360, 0.9, 0.95]);
        id.data[4 * k] = 255 * r; id.data[4 * k + 1] = 255 * g; id.data[4 * k + 2] = 255 * b; id.data[4 * k + 3] = 255;
      }
      sp.cv.width = W; sp.cv.height = H;
      sp.cv.getContext("2d")!.putImageData(id, 0, 0);
    }
    // surface, centred
    const S = res.surface, W = S.w, H = S.h;
    const shown = fftshift(S.data, W, H);
    const finite = Array.from(shown).filter(Number.isFinite);
    const lo = st.method === "ncc" ? -1 : Math.min(...finite), hi = st.method === "ncc" ? 1 : Math.max(...finite);
    drawScalarImage(sf.cv, shown, W, H, { map: st.method === "ncc" ? diverging : inferno, lo, hi, nan: [128, 128, 128, 255] });
    sf.svg.setAttribute("viewBox", `${-W / 2 - 0.5} ${-H / 2 - 0.5} ${W} ${H}`);
    const label = st.method === "ncc" ? "overlap NCC" : st.method === "cc" ? "cross-correlation" : st.p === 1 ? "phase correlation" : `normalised with p = ${st.p}`;
    sf.cap.textContent = `${label} (shifts ±${W / 2})`;
    trueM.setAttribute("cx", String(st.tx)); trueM.setAttribute("cy", String(st.ty)); trueM.setAttribute("r", String(Math.max(2.5, W / 40)));
    const px = res.tx, py = res.ty, c = Math.max(4, W / 18);
    peakM.setAttribute("d", `M${px - c},${py}H${px + c}M${px},${py - c}V${py + c}`);
    otherM.replaceChildren();
    for (const p of res.peaks.slice(1, 4)) {
      const o = svgEl("circle", { cx: p.tx, cy: p.ty, r: Math.max(2, W / 50), fill: "none" }, otherM);
      o.style.stroke = "#fff"; o.style.strokeWidth = "1.5"; o.setAttribute("vector-effect", "non-scaling-stroke");
    }
    const err = Math.hypot(res.tx - st.tx, res.ty - st.ty);
    const lines = [
      `true shift      (${st.tx}, ${st.ty})`,
      `detected        (${res.tx}, ${res.ty})${err === 0 ? "  ✓" : `  error ${err.toFixed(1)} px`}`,
      st.method === "ncc" ? "" : `sub-pixel       (${f2(res.sx)}, ${f2(res.sy)})`,
      `peak height     ${st.method === "cc" ? res.height.toExponential(2) : f2(res.height)}`,
      `value at truth  ${st.method === "cc" ? at(S, st.tx, st.ty).toExponential(2) : f2(at(S, st.tx, st.ty))}`,
      `peak ratio      ${Number.isFinite(res.ratio) ? res.ratio.toFixed(2) : "∞"}`,
      `FFT route       ${res.ms.toFixed(0)} ms (${W}×${H})`,
    ];
    ro.textContent = lines.filter(x => x).join("\n");
  }

  function setRect(r: SVGRectElement, x: number, y: number, n: number): void {
    r.setAttribute("x", String(x - 0.5)); r.setAttribute("y", String(y - 0.5));
    r.setAttribute("width", String(n)); r.setAttribute("height", String(n));
  }

  function bruteForce(): void {
    bfBtn.textContent = "running…";
    setTimeout(() => {
      const n = st.size, t0 = performance.now();
      const b = bruteForceNcc2d(crops.a, crops.b, n, n, n - 1, Math.round(0.25 * n * n));
      const ms = performance.now() - t0;
      bfBtn.textContent = "brute force (NCC, all shifts)";
      bfOut.style.display = "";
      bfOut.textContent = `brute force: (${b.tx}, ${b.ty}), NCC ${b.ncc.toFixed(3)}\n${(b.ops / 1e6).toFixed(0)} million multiply-adds\n${(ms / 1000).toFixed(2)} s (FFT route ${res ? res.ms.toFixed(0) : "–"} ms)`;
    }, 30);
  }

  load(p0);
}

// ---------------------------------------------------------------------------------------
// Template mode

function mountTemplate(host: HTMLElement): void {
  const p0 = TPRESETS.find(p => p.id === host.dataset.preset) ?? TPRESETS[0];
  const st = { src: p0.src, x: p0.x, y: p0.y, size: p0.size, thr: p0.thr };
  let cell: Float64Array | null = null;
  const nu = nuclei({ seed: 4, count: 18, noise: 8 }), nuc = Float64Array.from(nu.image.data);
  const nucStart: [number, number] = [Math.round(nu.ellipses[0].x - 16), Math.round(nu.ellipses[0].y - 16)];
  const left = htmlEl("div", "pc-img", "", host);
  left.style.flexBasis = "330px";
  const big = stack(left, "drag the template (yellow)", [0, 0, IMG, IMG]);
  big.cv.width = IMG; big.cv.height = IMG;
  const matchG = svgEl("g", {}, big.svg);
  const tb = rect(big.svg, "#ffd400", 3);
  tb.style.cursor = "move"; tb.style.pointerEvents = "all"; tb.setAttribute("fill", "rgba(255,212,0,0.12)");
  const credit = htmlEl("div", "pc-credit", CELL_CREDIT, left);
  const right = htmlEl("div", "pc-surf", "", host);
  right.style.flexBasis = "330px";
  const mp = stack(right, "NCC of the template at each position", [0, 0, IMG, IMG]);
  const mpG = svgEl("g", {}, mp.svg);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "pc-lab", "presets", panel);
  const presetBtns = presets(panel, TPRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const sizeS = slider(panel, "template size", 12, 48, 2, st.size, v => { st.size = v; clamp(); run(true); });
  const thrS = slider(panel, "NCC threshold", 0.2, 0.95, 0.05, st.thr, v => { st.thr = v; draw(); });
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Boxes: positions whose NCC is a local maximum above the threshold (at most 12). The map is blank where the template does not fit. NCC is +1 for a perfect match up to brightness and contrast, −1 for an inverted one.", panel);

  let drag: { x: number; y: number; px: number; py: number } | null = null;
  tb.addEventListener("pointerdown", ev => { ev.preventDefault(); tb.setPointerCapture(ev.pointerId); const [x, y] = svgPoint(big.svg, ev); drag = { x, y, px: st.x, py: st.y }; });
  tb.addEventListener("pointermove", ev => {
    if (!drag) return;
    const [x, y] = svgPoint(big.svg, ev);
    st.x = Math.round(drag.px + x - drag.x); st.y = Math.round(drag.py + y - drag.y); clamp();
    presetBtns.forEach(b => b.classList.remove("active")); run(false);
  });
  const up = () => { drag = null; };
  tb.addEventListener("pointerup", up); tb.addEventListener("pointercancel", up);

  function clamp(): void { st.x = Math.max(0, Math.min(IMG - st.size, st.x)); st.y = Math.max(0, Math.min(IMG - st.size, st.y)); }
  function load(p: TPreset): void {
    Object.assign(st, { src: p.src, x: p.x < 0 ? nucStart[0] : p.x, y: p.y < 0 ? nucStart[1] : p.y, size: p.size, thr: p.thr });
    clamp();
    presetBtns.forEach((b, i) => b.classList.toggle("active", TPRESETS[i] === p));
    sizeS.set(st.size); thrS.set(st.thr);
    run(true);
  }
  let matcher: TemplateMatcher | null = null, map: Float64Array = new Float64Array(IMG * IMG);
  let pending = false;
  function image(): Float64Array { return st.src === "cell" ? cell! : nuc; }
  function run(rebuild: boolean): void {
    if (st.src === "cell" && !cell) { loadCell().then(g => { cell = Float64Array.from(g.data); run(true); }); return; }
    if (rebuild) matcher = null;
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      const img = image();
      if (!matcher || matcher.tw !== st.size || matcher.img !== img) matcher = new TemplateMatcher(img, IMG, IMG, st.size, st.size);
      const T = new Float64Array(st.size * st.size);
      for (let y = 0; y < st.size; y++) for (let x = 0; x < st.size; x++) T[y * st.size + x] = img[(st.y + y) * IMG + st.x + x];
      map = matcher.match(T);
      draw();
    });
  }
  function draw(): void {
    const img = image(), n = st.size;
    drawScalarImage(big.cv, img, IMG, IMG, { lo: 0, hi: 255 });
    tb.setAttribute("x", String(st.x - 0.5)); tb.setAttribute("y", String(st.y - 0.5)); tb.setAttribute("width", String(n)); tb.setAttribute("height", String(n));
    drawScalarImage(mp.cv, map, IMG, IMG, { map: diverging, lo: -1, hi: 1, nan: [200, 200, 200, 255] });
    credit.style.visibility = st.src === "cell" ? "visible" : "hidden";
    // local maxima above threshold
    const peaks: { x: number; y: number; v: number }[] = [];
    const r = Math.max(3, Math.floor(n / 2));
    for (let y = 0; y + n <= IMG; y++) for (let x = 0; x + n <= IMG; x++) {
      const v = map[y * IMG + x];
      if (!(v >= st.thr)) continue;
      let isMax = true;
      for (let dy = -r; dy <= r && isMax; dy++) for (let dx = -r; dx <= r; dx++) {
        const X = x + dx, Y = y + dy;
        if (X < 0 || Y < 0 || X >= IMG || Y >= IMG || (dx === 0 && dy === 0)) continue;
        const u = map[Y * IMG + X];
        if (u > v || (u === v && (dy < 0 || (dy === 0 && dx < 0)))) { isMax = false; break; }
      }
      if (isMax) peaks.push({ x, y, v });
    }
    peaks.sort((a, b) => b.v - a.v);
    const top = peaks.slice(0, 12);
    matchG.replaceChildren(); mpG.replaceChildren();
    for (const p of top) {
      const b = rect(big.svg, "#00e5ff", 2); matchG.appendChild(b);
      b.setAttribute("x", String(p.x - 0.5)); b.setAttribute("y", String(p.y - 0.5)); b.setAttribute("width", String(n)); b.setAttribute("height", String(n));
      const c = svgEl("circle", { cx: p.x, cy: p.y, r: 4, fill: "none" }, mpG);
      c.style.stroke = "#000"; c.style.strokeWidth = "2"; c.setAttribute("vector-effect", "non-scaling-stroke");
    }
    ro.textContent = [`template ${n}×${n} at (${st.x}, ${st.y})`, `${peaks.length} matches with NCC ≥ ${st.thr.toFixed(2)}`,
      ...top.slice(0, 6).map(p => `  (${p.x}, ${p.y})  NCC ${p.v.toFixed(2)}`)].join("\n");
  }
  load(p0);
}

/** data-controls="method,noise,…": show only the sliders/checkboxes whose names start with these words. */
function hideControls(panel: HTMLElement, spec: string | undefined): void {
  if (!spec) return;
  const keep = spec.split(",").map(x => x.trim().toLowerCase());
  panel.querySelectorAll<HTMLElement>(".qb-slider, .qb-check").forEach(el => {
    const name = (el.querySelector(".qb-slider-name")?.textContent ?? el.textContent ?? "").trim().toLowerCase();
    if (!keep.some(k => name.startsWith(k))) el.style.display = "none";
  });
}

autoMount(NAME, mount);
