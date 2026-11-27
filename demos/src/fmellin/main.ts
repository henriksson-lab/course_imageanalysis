// Fourier–Mellin registration (shape-matching day, unit fourier-mellin/). Two tabs:
//  - "unknown centre": view B of the micrograph is view A rotated, scaled and shifted (sliders)
//    with noise. Panels: A, B, their magnitude spectra |F|, the log-polar resamplings of |F|,
//    the phase-correlation surface in (θ, log r) with the peak → angle and scale, and the
//    overlay of A with B after undoing angle and scale and finding the shift.
//  - "known centre": a disc around a known centre and its rotated copy, both unwrapped on a
//    polar grid; one rotational correlation (one FFT along θ per ring, summed) over the angle.
// Options: data-mode="fm|known", data-preset="<id>", data-presets="id,id,…",
// data-controls="angle,scale,…" (show only these sliders/checkboxes).
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl } from "../lib/ui";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { drawScalarImage, inferno } from "../lib/colour";
import { fftshift } from "../lib/fft";
import { Grey } from "../lib/imageops";
import { Plot } from "../lib/plot";
import { logStep, warpSimilarity, findPeaks } from "../lib/registration";
import { DEFAULTS, FMSettings, KC_DEFAULTS, KCSettings, N, makePair, register, wrapDeg, knownCentre } from "./model";

const NAME = "fmellin";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${S} { gap: .8em; align-items: flex-start; }
${S} figure { margin: 0; min-width: 0; }
${S} figcaption { font-size: 15px; text-align: center; line-height: 1.15; margin-bottom: .15em; }
${S} .fm-stack { position: relative; width: 100%; }
${S} .fm-stack canvas { width: 100%; display: block; image-rendering: pixelated; border-radius: 3px; }
${S} .fm-stack svg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
${S} .fm-figs { flex: 0 1 600px; min-width: 300px; display: grid; grid-template-columns: repeat(4, 1fr); gap: .35em .5em; align-items: end; }
${S} .fm-w2 { grid-column: span 2; }
${S} .fm-plot { grid-column: span 2; }
${S} .qb-panel { flex: 1 1 260px; max-width: 330px; gap: .3em; }
${S} .qb-readout { font-size: 13.5px; white-space: pre; }
${S} .fm-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
${S} .fm-credit { font-size: 12px; color: var(--qb-muted); grid-column: 1 / -1; }
${S} .fm-tabs .qb-btn { font-weight: 600; }
.reveal ${S} { font-size: 16px; }
.reveal ${S} .qb-btn { font-size: 14px; padding: .1em .4em; }
.reveal ${S} .qb-note { display: none; }
.reveal ${S} .qb-readout { font-size: 13px; }
.reveal ${S} .qb-slider { gap: .3em; }
.reveal .slides ${S} figcaption { font-size: 16px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

interface Preset { id: string; label: string; mode: "fm" | "known"; s?: Partial<FMSettings>; k?: Partial<KCSettings> }
const PRESETS: Preset[] = [
  { id: "rot30", label: "pure rotation 30°", mode: "fm", s: { angle: 30 } },
  { id: "scale", label: "scale 1.5 + shift", mode: "fm", s: { angle: 0, scale: 1.5, tx: 12, ty: -8 } },
  { id: "amb", label: "180° ambiguity", mode: "fm", s: { angle: 200, tx: 10, ty: 5, both: false } },
  { id: "notaper", label: "no taper: the cross dominates", mode: "fm", s: { shade: 400, taper: false } },
  { id: "scale3", label: "scale 3: out of range", mode: "fm", s: { angle: 20, scale: 3 } },
  { id: "noise", label: "noise: the high-pass hurts", mode: "fm", s: { noise: 25 } },
  { id: "known", label: "known centre: one 1D correlation", mode: "known", k: {} },
];

/** A figure with a canvas and an SVG overlay (viewBox in data units). */
function stack(parent: Element, caption: string, cls = ""): { fig: HTMLElement; cv: HTMLCanvasElement; svg: SVGSVGElement; cap: HTMLElement } {
  const fig = htmlEl("figure", cls, "", parent);
  const cap = htmlEl("figcaption", "", caption, fig);
  const st = htmlEl("div", "fm-stack", "", fig);
  const cv = htmlEl("canvas", "", "", st);
  const svg = svgEl("svg", { viewBox: "0 0 1 1", preserveAspectRatio: "none" }, st);
  return { fig, cv, svg, cap };
}

function marker(svg: SVGSVGElement, kind: "ring" | "cross", col: string): SVGElement {
  const el = kind === "ring" ? svgEl("circle", { fill: "none" }, svg) : svgEl("path", { fill: "none" }, svg);
  el.style.stroke = col; el.style.strokeWidth = "2.5"; el.setAttribute("vector-effect", "non-scaling-stroke");
  return el;
}

/** Overlay of two grey images: A in magenta, B in green (grey where they agree). */
function drawOverlay(cv: HTMLCanvasElement, a: ArrayLike<number>, b: ArrayLike<number>, w: number, h: number): void {
  cv.width = w; cv.height = h;
  const id = new ImageData(w, h);
  const norm = (v: ArrayLike<number>) => { let lo = Infinity, hi = -Infinity; for (let i = 0; i < v.length; i++) { lo = Math.min(lo, v[i]); hi = Math.max(hi, v[i]); } return (x: number) => Math.max(0, Math.min(255, 255 * (x - lo) / (hi - lo || 1))); };
  const na = norm(a), nb = norm(b);
  for (let i = 0; i < w * h; i++) { const p = na(a[i]), q = nb(b[i]); id.data[4 * i] = p; id.data[4 * i + 1] = q; id.data[4 * i + 2] = p; id.data[4 * i + 3] = 255; }
  cv.getContext("2d")!.putImageData(id, 0, 0);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const ids = host.dataset.presets?.split(",").map(x => x.trim());
  const LIST = ids ? PRESETS.filter(p => ids.includes(p.id)) : PRESETS;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? (host.dataset.mode === "known" ? PRESETS[PRESETS.length - 1] : LIST[0]);
  const st: FMSettings = { ...DEFAULTS };
  const kc: KCSettings = { ...KC_DEFAULTS };
  let mode: "fm" | "known" = p0.mode;
  let cell: Grey | null = null;

  // ---- layout: two figure blocks (one per tab), one panel ----
  const figsFM = htmlEl("div", "fm-figs", "", host);
  const va = stack(figsFM, "view A"), vb = stack(figsFM, "view B");
  const sa = stack(figsFM, "|F| of A"), sb = stack(figsFM, "|F| of B");
  const la = stack(figsFM, "log-polar |F| of A (θ →, log r ↓)", "fm-w2"), lb = stack(figsFM, "log-polar |F| of B", "fm-w2");
  const sf = stack(figsFM, "phase correlation in (θ, log r)", "fm-w2");
  const ov = stack(figsFM, "A and B, aligned");
  const ts = stack(figsFM, "shift step");
  htmlEl("div", "fm-credit", CELL_CREDIT, figsFM);
  const sfTrue = marker(sf.svg, "ring", "#2ca02c"), sfPeak = marker(sf.svg, "cross", "#00e5ff");
  const tsPeak = marker(ts.svg, "cross", "#00e5ff");

  const figsKC = htmlEl("div", "fm-figs", "", host);
  const ka = stack(figsKC, "disc A, known centre"), kb = stack(figsKC, "disc B (turned)");
  const kov = stack(figsKC, "A and B turned back");
  htmlEl("div", "", "", figsKC);
  const kpa = stack(figsKC, "A unwrapped (θ →, r ↓)", "fm-w2"), kpb = stack(figsKC, "B unwrapped", "fm-w2");
  const plotFig = htmlEl("figure", "fm-plot", "", figsKC);
  plotFig.style.gridColumn = "1 / -1";
  htmlEl("figcaption", "", "rotational correlation: every angle at once", plotFig);
  const plot = new Plot(plotFig, { width: 600, height: 190, x: [-180, 180], y: [-1, 1], xlabel: "angle of B relative to A (°)", xticks: [-180, -90, 0, 90, 180], yticks: [0], margin: { l: 30 }, font: 17 });
  const kLine = plot.line([], [], { color: "#1f77b4", width: 2.5 });
  const kTrue = plot.vline(0, { color: "#2ca02c", width: 2, dash: "6 4" });
  const kFound = plot.vline(0, { color: "#00a5bf", width: 2 });
  htmlEl("div", "fm-credit", CELL_CREDIT, figsKC);
  const rings = [ka, kb, kov].map(c => { const ring = svgEl("circle", { fill: "none" }, c.svg); ring.style.stroke = "#ffd400"; ring.style.strokeWidth = "2"; ring.setAttribute("vector-effect", "non-scaling-stroke"); return ring; });

  const panel = htmlEl("div", "qb-panel", "", host);
  const tabRow = htmlEl("div", "qb-presets fm-tabs", "", panel);
  const tabs = [button(tabRow, "unknown centre", () => { mode = "fm"; run(); }), button(tabRow, "known centre", () => { mode = "known"; run(); })];
  htmlEl("div", "fm-lab", "presets", panel);
  const presetBtns = presets(panel, LIST.map(p => ({ label: p.label, apply: () => load(p) })));
  const fmBox = htmlEl("div", "", "", panel);
  fmBox.style.display = "contents";
  const aS = slider(fmBox, "angle", -180, 180, 1, st.angle, v => { st.angle = v; changed(); }, v => `${v}°`);
  const scS = slider(fmBox, "scale", 0.4, 3, 0.05, st.scale, v => { st.scale = v; changed(); });
  const txS = slider(fmBox, "shift x", -30, 30, 1, st.tx, v => { st.tx = v; changed(); });
  const tyS = slider(fmBox, "shift y", -30, 30, 1, st.ty, v => { st.ty = v; changed(); });
  const nS = slider(fmBox, "noise σ", 0, 40, 1, st.noise, v => { st.noise = v; changed(); });
  const shS = slider(fmBox, "uneven light", 0, 400, 25, st.shade, v => { st.shade = v; changed(); });
  const row = htmlEl("div", "qb-presets", "", fmBox);
  const tapC = checkbox(row, "Hann taper", st.taper, v => { st.taper = v; changed(); });
  const hpC = checkbox(row, "high-pass on |F|", st.highpass, v => { st.highpass = v; changed(); });
  const bothC = checkbox(row, "test θ and θ + 180°", st.both, v => { st.both = v; changed(); });
  const kcBox = htmlEl("div", "", "", panel);
  kcBox.style.display = "contents";
  const kaS = slider(kcBox, "turn B", -180, 180, 1, kc.angle, v => { kc.angle = v; changed(); }, v => `${v}°`);
  const knS = slider(kcBox, "noise σ ", 0, 80, 5, kc.noise, v => { kc.noise = v; changed(); });
  const brow = htmlEl("div", "qb-presets", "", panel);
  button(brow, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  const note = htmlEl("div", "qb-note", "", panel);
  hideControls(panel, host.dataset.controls);

  function changed(): void { presetBtns.forEach(b => b.classList.remove("active")); run(); }

  function load(p: Preset): void {
    mode = p.mode;
    Object.assign(st, DEFAULTS, p.s ?? {});
    Object.assign(kc, KC_DEFAULTS, p.k ?? {});
    presetBtns.forEach((b, i) => b.classList.toggle("active", LIST[i] === p));
    run();
  }

  let pending = false;
  function run(): void {
    if (!cell) { loadCell().then(g => { cell = g; run(); }); return; }
    sync();
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; if (mode === "fm") drawFM(); else drawKC(); });
  }

  function sync(): void {
    tabs.forEach((b, i) => b.classList.toggle("active", ["fm", "known"][i] === mode));
    figsFM.style.display = mode === "fm" ? "" : "none";
    figsKC.style.display = mode === "known" ? "" : "none";
    for (const el of fmBox.children) (el as HTMLElement).style.visibility = "";
    fmBox.style.display = mode === "fm" ? "contents" : "none";
    kcBox.style.display = mode === "known" ? "contents" : "none";
    aS.set(st.angle); scS.set(st.scale); txS.set(st.tx); tyS.set(st.ty); nS.set(st.noise); shS.set(st.shade);
    tapC.checked = st.taper; hpC.checked = st.highpass; bothC.checked = st.both;
    kaS.set(kc.angle); knS.set(kc.noise);
    note.textContent = mode === "fm"
      ? "Angles clockwise. Surface: θ shift across (−90° … 90°), log r shift down; green ring: where the true angle and scale should put the peak, cyan cross: the highest peak. Overlay: A magenta, B green; grey where they agree."
      : "Angles clockwise. Each row of the unwrapped discs is one ring; turning B shifts every row by the same number of columns. Green dashed: the true angle; cyan: the highest score.";
  }

  function drawFM(): void {
    const t0 = performance.now();
    const pair = makePair(cell!, st), r = register(pair, st);
    const ms = performance.now() - t0;
    const lo = Math.min(...pair.a), hi = Math.max(...pair.a);
    drawScalarImage(va.cv, pair.a, N, N, { lo, hi });
    drawScalarImage(vb.cv, pair.b, N, N, { lo, hi });
    const lg = (m: Float64Array) => m.map(v => Math.log1p(v));
    const mA = lg(r.magA), mB = lg(r.magB), mHi = Math.max(...mA, ...mB);
    drawScalarImage(sa.cv, mA, N, N, { map: inferno, lo: 0, hi: mHi });
    drawScalarImage(sb.cv, mB, N, N, { map: inferno, lo: 0, hi: mHi });
    const g = r.grid, lA = lg(r.lpA), lB = lg(r.lpB), lHi = Math.max(...lA, ...lB);
    drawScalarImage(la.cv, lA, g.nTheta, g.nR, { map: inferno, lo: 0, hi: lHi });
    drawScalarImage(lb.cv, lB, g.nTheta, g.nR, { map: inferno, lo: 0, hi: lHi });
    // surface, shift 0 in the middle; x = θ bins, y = log-r bins
    const W = r.surface.w, H = r.surface.h, shown = fftshift(r.surface.data, W, H);
    drawScalarImage(sf.cv, shown, W, H, { map: inferno });
    sf.svg.setAttribute("viewBox", `${-W / 2 - 0.5} ${-H / 2 - 0.5} ${W} ${H}`);
    const trueTheta = ((wrapDeg(st.angle) % 180) + 270) % 180 - 90;   // in [−90, 90)
    const tTx = trueTheta * g.nTheta / 180, tTy = -Math.log(st.scale) / logStep(g);
    sfTrue.setAttribute("cx", String(tTx)); sfTrue.setAttribute("cy", String(tTy)); sfTrue.setAttribute("r", "6");
    cross(sfPeak, r.thetaBins, r.rhoBins, 9);
    // overlay: A and B with the found angle, scale and shift undone
    const b = r.best;
    const pk = cand(b);   // B with angle and scale undone is A shifted by this: shift it back
    const sh = warpSimilarity({ width: N, height: N, data: Float32Array.from(b.unwarped) }, N, N, { angle: 0, scale: 1, tx: pk.px, ty: pk.py });
    drawOverlay(ov.cv, pair.a, sh, N, N);
    const T = b.surface, tw = T.w, th = T.h;
    drawScalarImage(ts.cv, fftshift(T.data, tw, th), tw, th, { map: inferno });
    ts.svg.setAttribute("viewBox", `${-tw / 2 - 0.5} ${-th / 2 - 0.5} ${tw} ${th}`);
    cross(tsPeak, pk.px, pk.py, 8);
    // readout
    const found = wrapDeg(b.angle * 180 / Math.PI);
    const ok = Math.abs(wrapDeg(found - st.angle)) < 1 && Math.abs(r.scale / st.scale - 1) < 0.02 && Math.hypot(b.tx - st.tx, b.ty - st.ty) < 1;
    const f1 = (v: number) => (Math.abs(v) < 0.05 ? 0 : v).toFixed(1), f2 = (v: number) => v.toFixed(2);
    ro.textContent = [
      `true      ${f1(wrapDeg(st.angle))}° × ${f2(st.scale)}, shift (${st.tx}, ${st.ty})`,
      `log-polar peak at (${f1(r.thetaBins)}, ${f1(r.rhoBins)}) bins`,
      `  height ${f2(r.peakHeight)}, ratio ${f2(r.peakRatio)}`,
      `  → θ = ${f1(r.angle * 180 / Math.PI)}° (mod 180°), scale ${r.scale.toFixed(3)}`,
      ...r.candidates.map(c => `  try ${f1(wrapDeg(c.angle * 180 / Math.PI)).padStart(6)}°: shift peak ${f2(c.height)}`),
      `found     ${f1(found)}° × ${r.scale.toFixed(3)}, shift (${f1(b.tx)}, ${f1(b.ty)})${ok ? "  ✓" : "  ✗"}`,
      `time      ${ms.toFixed(0)} ms`,
    ].join("\n");
  }

  /** Peak of a candidate's translation surface (the shift between A and B with angle and scale undone). */
  function cand(c: { surface: { data: Float64Array; w: number; h: number } }): { px: number; py: number } {
    const p = findPeaks(c.surface, 1)[0];
    return { px: p.tx, py: p.ty };
  }

  function cross(el: SVGElement, x: number, y: number, c: number): void {
    const gp = c / 2.5;
    el.setAttribute("d", `M${x - c},${y}H${x - gp}M${x + gp},${y}H${x + c}M${x},${y - c}V${y - gp}M${x},${y + gp}V${y + c}`);
  }

  function drawKC(): void {
    const k = knownCentre(cell!, kc), n = k.size;
    const lo = Math.min(...k.a), hi = Math.max(...k.a);
    drawScalarImage(ka.cv, k.a, n, n, { lo, hi });
    drawScalarImage(kb.cv, k.b, n, n, { lo, hi });
    const back = warpSimilarity({ width: n, height: n, data: Float32Array.from(k.b) }, n, n, { angle: -k.angle * Math.PI / 180, scale: 1, tx: 0, ty: 0 });
    drawOverlay(kov.cv, k.a, back, n, n);
    [ka, kb, kov].forEach((c, i) => {
      c.svg.setAttribute("viewBox", `-0.5 -0.5 ${n} ${n}`);
      const ring = rings[i];
      ring.setAttribute("cx", String(n / 2)); ring.setAttribute("cy", String(n / 2)); ring.setAttribute("r", String(kc.radius));
    });
    drawScalarImage(kpa.cv, k.pa, kc.nTheta, k.nR, { lo, hi });
    drawScalarImage(kpb.cv, k.pb, kc.nTheta, k.nR, { lo, hi });
    // stretch the unwraps (256 × 48) to 2.5 : 1
    for (const c of [kpa, kpb]) { c.cv.style.aspectRatio = "2.5 / 1"; c.cv.style.objectFit = "fill"; }
    const xs: number[] = [], ys: number[] = [];
    let m = 0;
    for (let t = 0; t < kc.nTheta; t++) m = Math.max(m, Math.abs(k.score[t]));
    for (let t = 0; t < kc.nTheta; t++) { xs.push(wrapDeg(t * 360 / kc.nTheta)); ys.push(k.score[t] / (m || 1)); }
    // sort by angle so the line runs from −180 to 180
    const idx = xs.map((_, i) => i).sort((i, j) => xs[i] - xs[j]);
    kLine.set(idx.map(i => xs[i]), idx.map(i => ys[i]));
    kTrue.set(wrapDeg(kc.angle)); kFound.set(k.angle);
    const ok = Math.abs(wrapDeg(k.angle - kc.angle)) < 360 / kc.nTheta;
    ro.textContent = [
      `true turn   ${wrapDeg(kc.angle)}°`,
      `found       ${k.angle.toFixed(1)}°${ok ? "  ✓" : "  ✗"}`,
      `grid        ${kc.nTheta} angles (${(360 / kc.nTheta).toFixed(2)}° each), ${k.nR} rings`,
      `cost        ${k.nR} FFTs of length ${kc.nTheta} per image, one inverse`,
    ].join("\n");
  }

  load(p0);
}

/** data-controls="angle,scale,…": show only the sliders/checkboxes whose names start with these words. */
function hideControls(panel: HTMLElement, spec: string | undefined): void {
  if (!spec) return;
  const keep = spec.split(",").map(x => x.trim().toLowerCase());
  panel.querySelectorAll<HTMLElement>(".qb-slider, .qb-check").forEach(el => {
    const name = (el.querySelector(".qb-slider-name")?.textContent ?? el.textContent ?? "").trim().toLowerCase();
    if (!keep.some(k => name.startsWith(k))) el.style.display = "none";
  });
}

autoMount(NAME, mount);
