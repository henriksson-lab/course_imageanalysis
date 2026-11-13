// Histogram equalisation and matching. Panels: input and output image; the mapping curve
// (input value -> output level), which for equalisation is the scaled cumulative histogram;
// the input histogram with its cumulative histogram; the output histogram with its
// cumulative histogram and the flat level N/L. Modes: equalise (all pixels of one value move
// together), exact (rank order with random tie-breaking: exactly flat), match (to a flat or
// Gaussian target, or to another image). A gamma applied first shows that equalisation only
// uses the order of the values. Click a histogram bar to highlight its pixels.
// Auto-mounts on <div data-demo="equalise">. Options: data-preset (hand|curve|notflat|exact|
// gamma|background|matchimg|matchgauss).
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { rng } from "../lib/ui";
import * as C from "../lib/contrast";

const NAME = "equalise";
const P = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${P} { gap: .7em; }
${P} figure { margin: 0; }
${P} figcaption { font-size: 16px; text-align: center; margin: 0 0 .1em; }
${P} .eq-left { flex: 1 1 400px; min-width: 300px; max-width: 440px; display: flex; flex-direction: column; gap: .2em; }
${P} .eq-imgs { display: flex; gap: .5em; }
${P} .eq-imgs figure { flex: 1 1 0; min-width: 0; }
${P} .eq-imgs canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
${P} .eq-mid { flex: 1 1 360px; min-width: 280px; max-width: 420px; display: flex; flex-direction: column; gap: .1em; }
${P} .eq-mid svg { cursor: pointer; }
${P} .qb-panel { flex: 0 1 270px; gap: .25em; }
${P} .qb-readout { font-size: 14px; }
${P} .eq-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
${P} .eq-credit { font-size: 12px; color: var(--qb-muted); }
.reveal ${P} { font-size: 18px; }
.reveal ${P} figcaption, .reveal ${P} .qb-figure figcaption { font-size: 17px; }
.reveal ${P} .qb-btn { font-size: 16px; padding: .05em .4em; }
.reveal ${P} .eq-lab { font-size: 15px; }
.reveal ${P} .qb-panel { gap: .15em; }
.reveal ${P} .qb-readout { font-size: 15px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

type Source = "hand" | "cell" | "fluo12" | "foggy";
type Mode = "equalise" | "exact" | "match";
type Target = "flat" | "gauss" | "image";
interface State { source: Source; Lout: number; mode: Mode; target: Target; gamma: number; noise: number; sel: { kind: "in" | "out"; bin: number } | null }

const SOURCES: { id: Source; label: string }[] = [
  { id: "hand", label: "8 × 8, 8 levels" }, { id: "cell", label: "micrograph" },
  { id: "fluo12", label: "fluorescence" }, { id: "foggy", label: "foggy" },
];
const MODES: { id: Mode; label: string }[] = [{ id: "equalise", label: "equalise" }, { id: "exact", label: "exact (ranks)" }, { id: "match", label: "match to …" }];
const TARGETS: { id: Target; label: string }[] = [{ id: "flat", label: "flat" }, { id: "gauss", label: "Gaussian" }, { id: "image", label: "another image" }];
const LOUTS = [8, 16, 64, 256];
const BASE: State = { source: "cell", Lout: 256, mode: "equalise", target: "flat", gamma: 1, noise: 0, sel: null };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "hand", label: "8 levels by hand", s: { source: "hand", Lout: 8 } },
  { id: "curve", label: "cumulative = curve", s: {} },
  { id: "notflat", label: "not flat: bins move", s: { source: "foggy" } },
  { id: "exact", label: "exact: invented order", s: { source: "foggy", mode: "exact" } },
  { id: "gamma", label: "gamma first: same", s: { gamma: 0.4 } },
  { id: "background", label: "background gets the contrast", s: { source: "fluo12" } },
  { id: "matchimg", label: "match another image", s: { source: "foggy", mode: "match", target: "image" } },
  { id: "matchgauss", label: "match a Gaussian", s: { mode: "match", target: "gauss" } },
];

/** The 8 × 8 worked example: counts 6, 22, 18, 8, 5, 3, 1, 1; brighter towards the middle. */
function handImage(): Float32Array {
  const counts = [6, 22, 18, 8, 5, 3, 1, 1];
  const pos = Array.from({ length: 64 }, (_, i) => i);
  const d = (i: number) => Math.hypot((i % 8) - 4.2, Math.floor(i / 8) - 3.6) + 0.01 * ((i * 7) % 5);
  pos.sort((a, b) => d(b) - d(a));
  const out = new Float32Array(64);
  let k = 0;
  counts.forEach((c, v) => { for (let j = 0; j < c; j++) out[pos[k++]] = v; });
  return out;
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...BASE };

  // ---- layout
  const left = htmlEl("div", "eq-left", "", host);
  const imgs = htmlEl("div", "eq-imgs", "", left);
  const f1 = htmlEl("figure", "", "", imgs), f2 = htmlEl("figure", "", "", imgs);
  const cap1 = htmlEl("figcaption", "", "input", f1), cap2 = htmlEl("figcaption", "", "output", f2);
  const cv1 = htmlEl("canvas", "", "", f1), cv2 = htmlEl("canvas", "", "", f2);
  cv1.width = cv1.height = cv2.width = cv2.height = 256;
  const credit = htmlEl("div", "eq-credit", CELL_CREDIT, left);
  const map = new Plot(left, { width: 440, height: 260, x: [0, 255], y: [0, 255], xlabel: "input value", ylabel: "output level", title: "mapping (dashed: (L − 1) × cumulative)" });

  const mid = htmlEl("div", "eq-mid", "", host);
  const hIn = new Plot(mid, { width: 420, height: 215, x: [0, 255], y: [0, 1], xlabel: "input value", yticks: [], margin: { l: 20, r: 20 }, title: "input histogram, cumulative (red)" });
  const hOut = new Plot(mid, { width: 420, height: 215, x: [0, 255], y: [0, 1], xlabel: "output level", yticks: [], margin: { l: 20, r: 20 }, title: "output; dashed: flat N/L" });
  const ro = htmlEl("div", "qb-readout", "", mid);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "eq-lab", "presets", panel);
  const pBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "eq-lab", "image", panel);
  const sRow = htmlEl("div", "qb-presets", "", panel);
  const sBtns = SOURCES.map(s => button(sRow, s.label, () => { st.source = s.id; st.sel = null; prepare(); }));
  htmlEl("div", "eq-lab", "method", panel);
  const mRow = htmlEl("div", "qb-presets", "", panel);
  const mBtns = MODES.map(m => button(mRow, m.label, () => { st.mode = m.id; st.sel = null; sync(); compute(); }));
  const tRow = htmlEl("div", "qb-presets", "", panel);
  const tBtns = TARGETS.map(t => button(tRow, t.label, () => { st.target = t.id; sync(); compute(); }));
  htmlEl("div", "eq-lab", "output levels L", panel);
  const lRow = htmlEl("div", "qb-presets", "", panel);
  const lBtns = LOUTS.map(l => button(lRow, String(l), () => { st.Lout = l; st.sel = null; sync(); compute(); }));
  const gS = slider(panel, "γ first", -0.7, 0.7, 0.01, 0, v => { st.gamma = 10 ** v; prepare(); }, v => (10 ** v).toFixed(2));
  const nS = slider(panel, "noise", 0, 20, 0.5, 0, v => { st.noise = v; prepare(); }, v => v.toFixed(1));
  button(panel, "reset", () => load(p0));

  // ---- sources
  let cell: Float32Array | null = null;
  const fluo = Float32Array.from(C.fluorescence12());
  const foggy = Float32Array.from(C.foggyScene());
  const hand = handImage();
  const info = (s: Source) => s === "hand" ? { L: 8, w: 8 } : s === "fluo12" ? { L: 4096, w: 256 } : { L: 256, w: 256 };
  const raw = (s: Source): Float32Array => s === "hand" ? hand : s === "cell" ? (cell ?? new Float32Array(65536)) : s === "fluo12" ? fluo : foggy;

  loadCell().then(g => {
    cell = Float32Array.from(g.data, v => Math.round(v));
    if (st.source === "cell" || st.mode === "match") prepare();
  });

  let d: Float32Array = new Float32Array(0), dg: Float32Array = new Float32Array(0), L = 256, W = 256;
  function prepare(): void {
    ({ L, w: W } = info(st.source));
    const r = rng(17), src = raw(st.source);
    d = Float32Array.from(src, v => st.noise > 0 ? Math.max(0, Math.min(L - 1, Math.round(v + st.noise * (L / 256) * r.norm()))) : v);
    dg = st.gamma === 1 ? d : Float32Array.from(d, v => (L - 1) * Math.pow(v / (L - 1), st.gamma));
    sync(); compute();
  }

  function sync(): void {
    sBtns.forEach((b, i) => b.classList.toggle("active", SOURCES[i].id === st.source));
    mBtns.forEach((b, i) => b.classList.toggle("active", MODES[i].id === st.mode));
    tBtns.forEach((b, i) => b.classList.toggle("active", TARGETS[i].id === st.target));
    lBtns.forEach((b, i) => b.classList.toggle("active", LOUTS[i] === st.Lout));
    tRow.style.display = st.mode === "match" ? "" : "none";
    gS.set(Math.log10(st.gamma)); nS.set(st.noise);
    credit.style.display = st.source === "cell" || (st.mode === "match" && st.target === "image") ? "" : "none";
  }

  function load(p: typeof PRESETS[number]): void {
    Object.assign(st, BASE, p.s);
    pBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    prepare();
  }

  // ---- compute
  let out: Float32Array = new Float32Array(0), lut: C.Lut = new Float64Array(0), curveUnrounded: C.Lut = new Float64Array(0);
  function targetHist(): Float64Array {
    const Lo = st.Lout, N = d.length;
    if (st.target === "flat") return C.flatTarget(Lo, N);
    if (st.target === "gauss") return C.gaussianTarget(Lo, (Lo - 1) / 2, Lo / 7, N);
    // another image: the micrograph (or, for the micrograph, the foggy scene), quantised to Lo levels
    const other = st.source === "cell" ? foggy : (cell ?? foggy);
    return C.levelHistogram(Float32Array.from(other, v => Math.floor(v * Lo / 256)), Lo);
  }
  function compute(): void {
    const hist = C.levelHistogram(d, L);
    curveUnrounded = C.equaliseLut(hist, st.Lout, "plain", false);
    if (st.mode === "equalise") { out = C.equaliseValues(dg, st.Lout); lut = C.equaliseLut(hist, st.Lout); }
    else if (st.mode === "exact") { out = C.equaliseExact(dg, st.Lout, "random", 5, W, W); lut = new Float64Array(0); }
    else { lut = C.matchLut(hist, targetHist()); out = C.applyLut(d, lut); }
    draw();
  }

  // ---- drawing
  const ORANGE = [255, 127, 14];
  function drawImage(cv: HTMLCanvasElement, v: Float32Array, levels: number, selMask: Uint8Array | null, numbers: boolean): void {
    const ctx = cv.getContext("2d")!, n = W, s = 256 / n;
    if (n === 256) {
      const id = ctx.createImageData(256, 256);
      for (let i = 0; i < v.length; i++) {
        const g = Math.round(255 * v[i] / (levels - 1)), on = selMask && selMask[i];
        id.data[4 * i] = on ? ORANGE[0] : g; id.data[4 * i + 1] = on ? ORANGE[1] : g; id.data[4 * i + 2] = on ? ORANGE[2] : g; id.data[4 * i + 3] = 255;
      }
      ctx.putImageData(id, 0, 0);
      return;
    }
    for (let i = 0; i < v.length; i++) {
      const g = Math.round(255 * v[i] / (levels - 1)), x = (i % n) * s, y = Math.floor(i / n) * s;
      ctx.fillStyle = `rgb(${g},${g},${g})`; ctx.fillRect(x, y, s, s);
      if (selMask && selMask[i]) { ctx.strokeStyle = "rgb(255,127,14)"; ctx.lineWidth = 4; ctx.strokeRect(x + 2, y + 2, s - 4, s - 4); }
      if (numbers) {
        ctx.fillStyle = g > 140 ? "#000" : "#fff"; ctx.font = "bold 15px system-ui, sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(String(Math.round(v[i])), x + s / 2, y + s / 2);
      }
    }
  }

  function draw(): void {
    const N = d.length, Lo = st.Lout;
    // input histogram (bins over 0..L-1 of the gamma-transformed values)
    const nbIn = Math.min(L, 256), bwIn = L / nbIn;
    const binIn = (v: number) => Math.min(nbIn - 1, Math.max(0, Math.floor((v + 0.5) / bwIn)));
    const hin = new Float64Array(nbIn);
    for (let i = 0; i < N; i++) hin[binIn(dg[i])]++;
    const hout = C.levelHistogram(out, Lo);
    // selection
    let sel: Uint8Array | null = null;
    if (st.sel) {
      sel = new Uint8Array(N);
      for (let i = 0; i < N; i++) sel[i] = st.sel.kind === "in" ? +(binIn(dg[i]) === st.sel.bin) : +(Math.round(out[i]) === st.sel.bin);
    }
    drawImage(cv1, dg, L, sel, W <= 16);
    drawImage(cv2, out, Lo, sel, W <= 16);
    cap1.textContent = st.gamma !== 1 ? `input after γ = ${st.gamma.toFixed(2)}` : `input (${L} levels)`;
    cap2.textContent = `output (${Lo} levels)`;

    // mapping
    map.setDomain([-0.5, L - 0.5], [-0.5, Lo - 0.5]);
    map.clear();
    const xs = Array.from({ length: L }, (_, v) => v);
    if (st.mode === "exact") {
      // a level can spread over a range of outputs: draw min..max per used input level
      const mn = new Float64Array(L).fill(Infinity), mx = new Float64Array(L).fill(-Infinity);
      for (let i = 0; i < N; i++) { const k = Math.round(d[i]); mn[k] = Math.min(mn[k], out[i]); mx[k] = Math.max(mx[k], out[i]); }
      const lx: number[] = [], ly: number[] = [];
      for (let k = 0; k < L; k++) if (mn[k] <= mx[k]) { lx.push(k, k, NaN); ly.push(mn[k], mx[k], NaN); }
      map.line(xs, Array.from(curveUnrounded), { color: "var(--qb-muted, #999)", width: 2, dash: "5 4" });
      map.line(lx, ly, { color: seriesColor(1), width: L <= 16 ? 8 : 2.5 });
    } else {
      map.line(xs, Array.from(curveUnrounded), { color: "var(--qb-muted, #999)", width: 2, dash: "5 4" });
      map.step(xs.map(v => v - 0.5), Array.from(lut), { color: seriesColor(0), width: 3 });
    }
    map.setLabels({ title: st.mode === "match" ? "mapping T = G⁻¹ ∘ F (dashed: equalisation)" : "mapping (dashed: (L − 1) × cumulative)" });

    // input histogram + cumulative
    const hmax = Math.max(...hin);
    hIn.setDomain([-0.5, L - 0.5], [0, 1.16]);
    hIn.clear();
    const eIn = Array.from({ length: nbIn + 1 }, (_, k) => k * bwIn - 0.5);
    hIn.histogram(eIn, Array.from(hin, c => c / hmax), { color: seriesColor(0), fillOpacity: 0.6, width: 0.5 });
    let cum = 0;
    const cIn = Array.from(hin, c => (cum += c) / N);
    hIn.step(eIn.slice(0, nbIn), cIn, { color: seriesColor(1), width: 2.5 });
    if (st.sel?.kind === "in") hIn.histogram([eIn[st.sel.bin], eIn[st.sel.bin + 1]], [hin[st.sel.bin] / hmax], { color: seriesColor(3), fillOpacity: 0.9 });
    if (nbIn <= 16) hin.forEach((c, k) => hIn.text(k, c / hmax + 0.02, String(c), { size: 16 }));

    // output histogram + cumulative + flat level
    const flat = N / Lo, omax = Math.max(flat, ...hout);
    hOut.setDomain([-0.5, Lo - 0.5], [0, 1.16]);
    hOut.clear();
    const eOut = Array.from({ length: Lo + 1 }, (_, k) => k - 0.5);
    hOut.histogram(eOut, Array.from(hout, c => c / omax), { color: seriesColor(2), fillOpacity: 0.6, width: 0.5 });
    hOut.hline(flat / omax, { color: "var(--qb-fg, #333)", dash: "6 4", width: 2 });
    cum = 0;
    hOut.step(eOut.slice(0, Lo), Array.from(hout, c => (cum += c) / N), { color: seriesColor(1), width: 2.5 });
    hOut.line([-0.5, Lo - 0.5], [0, 1], { color: seriesColor(1), width: 1.5, dash: "3 4", opacity: 0.7 });
    if (st.sel?.kind === "out") hOut.histogram([eOut[st.sel.bin], eOut[st.sel.bin + 1]], [hout[st.sel.bin] / omax], { color: seriesColor(3), fillOpacity: 0.9 });
    if (Lo <= 16) hout.forEach((c, k) => hOut.text(k, c / omax + 0.02, String(c), { size: 16 }));

    // readouts
    let used = 0, dev = 0;
    cum = 0;
    for (let k = 0; k < Lo; k++) { if (hout[k] > 0) used++; cum += hout[k]; dev = Math.max(dev, Math.abs(cum / N - (k + 1) / Lo)); }
    const inUsed = C.distinctCount(d);
    const lines = [
      `levels used: ${inUsed} in → ${used} of ${Lo} out`,
      `largest output bin ${Math.max(...hout)} (flat: ${flat.toFixed(flat < 10 ? 1 : 0)})`,
      `cumulative off the line by ≤ ${(100 * dev).toFixed(1)} %`,
    ];
    if (st.sel) {
      const n = sel ? sel.reduce((a, b) => a + b, 0) : 0;
      lines.push(st.sel.kind === "in" ? `selected input bin: ${n} pixels` : `selected output level ${st.sel.bin}: ${n} pixels`);
    } else lines.push("click a bar to highlight its pixels");
    ro.textContent = lines.join("\n");
  }

  // ---- clicks on the histograms
  hIn.svg.addEventListener("pointerdown", ev => {
    const [x] = hIn.toData(ev), nbIn = Math.min(L, 256), b = Math.floor((x + 0.5) / (L / nbIn));
    st.sel = b >= 0 && b < nbIn && !(st.sel?.kind === "in" && st.sel.bin === b) ? { kind: "in", bin: b } : null;
    draw();
  });
  hOut.svg.addEventListener("pointerdown", ev => {
    const [x] = hOut.toData(ev), b = Math.round(x);
    st.sel = b >= 0 && b < st.Lout && !(st.sel?.kind === "out" && st.sel.bin === b) ? { kind: "out", bin: b } : null;
    draw();
  });

  load(p0);
}

autoMount(NAME, mount);
