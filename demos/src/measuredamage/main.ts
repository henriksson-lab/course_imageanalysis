// What compression does to measurements. A synthetic 12-bit fluorescence image with known
// spots is stored lossless, as JPEG at quality q (after a 12 → 8-bit rescale), or with a simple
// wavelet codec at the same size as the JPEG; the same analysis runs on the original and the
// decoded image: spot detection (found, missed, false), integrated intensity of every spot
// (change against the original, scatter), background mean and noise, the share of
// high-frequency power, and the spectrum of the absolute neighbour differences, where JPEG's
// 8-pixel blocks show up as peaks at multiples of 1/8 of the sampling frequency (circled; the
// "blocking index" is their average height above the surrounding spectrum). A plot follows one measurement over all
// JPEG qualities against the size.
// Auto-mounts on <div data-demo="measuredamage">. Options: data-preset="q95|q30|q10|spectrum|
// lossless|wavelet", data-codec="lossless|jpeg|wavelet", data-quality="90",
// data-curve="dim|found|false|bgsd|hf", data-view="error|spectrum".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot } from "../lib/plot";
import { drawScalarImage, diverging } from "../lib/colour";
import * as M from "./scene";

const NAME = "measuredamage";
const N = M.N;
type Codec = "lossless" | "jpeg" | "wavelet";
type Curve = "dim" | "found" | "false" | "bgsd" | "hf";
interface State { codec: Codec; q: number; curve: Curve; view: "error" | "spectrum" }
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "q95", label: "quality 95 looks perfect", s: { codec: "jpeg", q: 95, curve: "dim", view: "error" } },
  { id: "q30", label: "quality 30: dim spots off", s: { codec: "jpeg", q: 30, curve: "dim", view: "error" } },
  { id: "q10", label: "quality 10: spots lost", s: { codec: "jpeg", q: 10, curve: "found", view: "error" } },
  { id: "spectrum", label: "blocking in the spectrum", s: { codec: "jpeg", q: 30, curve: "hf", view: "spectrum" } },
  { id: "lossless", label: "lossless: identical", s: { codec: "lossless", q: 90, curve: "dim", view: "error" } },
  { id: "wavelet", label: "wavelet, same size as q 30", s: { codec: "wavelet", q: 30, curve: "dim", view: "spectrum" } },
];
const CURVE_LABEL: Record<Curve, string> = { dim: "dim spots: mean |change| %", found: "spots found (of 60)", false: "false detections", bgsd: "background noise sd", hf: "high-frequency power %" };

const STYLE = `
.qb-demo[data-demo="${NAME}"] .md-main { display: flex; flex-direction: column; gap: .4em; flex: 1 1 700px; min-width: 0; max-width: 860px; }
.qb-demo[data-demo="${NAME}"] .md-row { display: flex; gap: 10px; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .md-row > figure { margin: 0; flex: 1 1 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; line-height: 1.2; margin-bottom: .15em; }
.qb-demo[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; }
.qb-demo.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 230px; max-width: none; gap: .4em; }
.reveal .qb-demo.qb-demo[data-demo="${NAME}"] .qb-panel { max-width: 270px; }
.qb-demo[data-demo="${NAME}"] .md-btns { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
.qb-demo[data-demo="${NAME}"] .qb-btn.on { background: var(--qb-accent); color: #fff; border-color: var(--qb-accent); }
.qb-demo[data-demo="${NAME}"] .md-credit { font-size: 13px; color: var(--qb-muted); }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; padding: .3em .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { font-size: 13px; }
.reveal .qb-demo[data-demo="${NAME}"] .md-main { max-width: 760px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 17px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function circles(c: HTMLCanvasElement, pts: { x: number; y: number }[], colour: string, r = 5): void {
  const ctx = c.getContext("2d")!;
  ctx.strokeStyle = colour; ctx.lineWidth = 1.2;
  for (const p of pts) { ctx.beginPath(); ctx.arc(p.x + 0.5, p.y + 0.5, r, 0, 2 * Math.PI); ctx.stroke(); }
}

function mount(host: HTMLElement): void {
  injectStyle();
  const d = host.dataset;
  const pre = PRESETS.find(p => p.id === d.preset);
  const init: State = { codec: "jpeg", q: 90, curve: "dim", view: "error", ...(pre ? pre.s : {}) };
  if (d.codec === "lossless" || d.codec === "jpeg" || d.codec === "wavelet") init.codec = d.codec;
  if (d.quality) init.q = Number(d.quality);
  if (d.curve && d.curve in CURVE_LABEL) init.curve = d.curve as Curve;
  if (d.view === "spectrum" || d.view === "error") init.view = d.view;
  const st: State = { ...init };

  const main = htmlEl("div", "md-main", "", host);
  const row1 = htmlEl("div", "md-row", "", main);
  const fig = (p: Element, t: string) => { const f = htmlEl("figure", "", "", p); const cap = htmlEl("figcaption", "", t, f); return { f, cap }; };
  const fo = fig(row1, "original (12 bit)"), fd = fig(row1, "decoded"), fe = fig(row1, "error");
  const cOrig = htmlEl("canvas", "", "", fo.f), cDec = htmlEl("canvas", "", "", fd.f), cErr = htmlEl("canvas", "", "", fe.f);
  const row2 = htmlEl("div", "md-row", "", main);
  const sFig = fig(row2, "integrated intensity of each spot");
  const scatter = new Plot(sFig.f, { width: 420, height: 250, x: [500, 20000], y: [-40, 40], xlog: true, xlabel: "on the original", ylabel: "change %", font: 16, margin: { l: 62, b: 48, r: 10, t: 8 } });
  const cFig = fig(row2, "");
  const curve = new Plot(cFig.f, { width: 420, height: 250, x: [0.1, 6], y: [0, 50], xlog: true, xlabel: "bits per pixel (JPEG)", font: 16, margin: { l: 62, b: 48, r: 10, t: 8 } });
  htmlEl("div", "md-credit", "Synthetic 12-bit fluorescence image: 60 Gaussian spots (peak 60–1500 above the cell, σ 1–2.2 px) in a cell with a darker nucleus, Poisson noise, fixed seed. Detected spots yellow, missed true spots red.", main);

  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, p.s); syncUI(); update(); } })));
  pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === pre));
  const cRow = htmlEl("div", "md-btns", "", panel);
  const codecBtns = (["lossless", "jpeg", "wavelet"] as Codec[]).map(c => button(cRow, c === "wavelet" ? "wavelet (same size)" : c === "jpeg" ? "JPEG" : "lossless", () => { st.codec = c; syncUI(); update(); }));
  const qS = slider(panel, "JPEG quality", 5, 100, 5, st.q, v => { st.q = v; later(); });
  const vRow = htmlEl("div", "md-btns", "", panel);
  const viewBtns = [button(vRow, "error image", () => { st.view = "error"; syncUI(); update(); }), button(vRow, "blocking spectrum", () => { st.view = "spectrum"; syncUI(); update(); })];
  htmlEl("div", "", "plot against size:", panel);
  const kRow = htmlEl("div", "md-btns", "", panel);
  const curveBtns = (Object.keys(CURVE_LABEL) as Curve[]).map(c => button(kRow, { dim: "dim spots", found: "found", false: "false", bgsd: "noise sd", hf: "HF power" }[c], () => { st.curve = c; syncUI(); drawCurve(); }));
  button(panel, "reset", () => { Object.assign(st, init); syncUI(); update(); pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === pre)); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "JPEG works on the image rescaled to 8 bit. Sizes: JPEG = Huffman bits + header; lossless and wavelet = entropy estimates. Raw 16-bit storage: 131 072 B.", panel);

  function syncUI(): void {
    codecBtns.forEach((b, i) => b.classList.toggle("on", ["lossless", "jpeg", "wavelet"][i] === st.codec));
    viewBtns.forEach((b, i) => b.classList.toggle("on", (i === 0) === (st.view === "error")));
    curveBtns.forEach((b, i) => b.classList.toggle("on", Object.keys(CURVE_LABEL)[i] === st.curve));
    qS.set(st.q);
  }

  // ---- data (computed once) ----
  const sc = M.makeScene();
  const T = M.detectionThreshold(sc.image);
  const bgMask = M.backgroundMask(sc.spots, sc.clean);
  const m0 = M.measure(sc.image, T, sc, bgMask);
  const th = M.thirds(sc.spots);
  const bi0 = M.blockingIndex(sc.image);
  const disp = { lo: 100, hi: 1100 };
  drawScalarImage(cOrig, sc.image, N, N, disp);
  circles(cOrig, m0.det.centres, "#ffd400");
  const jpegCache = new Map<number, M.Decoded>();
  const jpegAt = (q: number) => { let r = jpegCache.get(q); if (!r) { r = M.jpeg12(sc.image, q); jpegCache.set(q, r); } return r; };
  const waveCache = new Map<number, M.Decoded>();

  let timer = 0;
  function later(): void { window.clearTimeout(timer); timer = window.setTimeout(update, 100); }

  function update(): void {
    const dec = st.codec === "lossless" ? M.lossless(sc.image) : st.codec === "jpeg" ? jpegAt(st.q)
      : (waveCache.get(st.q) ?? (() => { const w = M.waveletAtSize(sc.image, jpegAt(st.q).bytes); waveCache.set(st.q, w); return w; })());
    const m = M.measure(dec.image, T, sc, bgMask);
    drawScalarImage(cDec, dec.image, N, N, disp);
    circles(cDec, m.det.centres, "#ffd400");
    // missed true spots: no detection within 3 px
    circles(cDec, sc.spots.filter(s => !m.det.centres.some(c => Math.hypot(c.x - s.x, c.y - s.y) <= 3)), "#ff3030", 7);
    fd.cap.textContent = dec.label;
    const err = Float64Array.from(dec.image, (v, i) => v - sc.image[i]);
    if (st.view === "error") {
      drawScalarImage(cErr, err, N, N, { map: diverging, lo: -150, hi: 150 });
      fe.cap.textContent = "error (±150)";
    } else {
      // blocking: the size of the jumps between neighbouring pixels repeats every 8 pixels
      const P = M.powerSpectrum(M.absDiff(dec.image)), L = new Float64Array(N * N);
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) L[((y + N / 2) % N) * N + (x + N / 2) % N] = Math.log10(1 + P[y * N + x]);
      let mx = 0;
      for (const v of L) mx = Math.max(mx, v);
      drawScalarImage(cErr, L, N, N, { lo: Math.max(0, mx - 5), hi: mx });
      const ctx = cErr.getContext("2d")!;   // rings at multiples of N/8 on the axes (blocking frequencies)
      ctx.strokeStyle = "#40d0ff"; ctx.lineWidth = 1.5;
      for (const [fx, fy] of M.BLOCK_FREQS) for (const sg of [1, -1]) {
        if (sg < 0 && (fx === 128 || fy === 128)) continue;
        ctx.beginPath(); ctx.arc(N / 2 + sg * fx + 0.5, N / 2 + sg * fy + 0.5, 6, 0, 2 * Math.PI); ctx.stroke();
      }
      fe.cap.textContent = "spectrum of |pixel differences|";
    }
    // scatter of the per-spot change
    scatter.clear();
    const xs = Array.from(m0.integ, v => Math.max(500, v)), ys = Array.from(m.integ, (v, i) => Math.max(-40, Math.min(40, 100 * (v - m0.integ[i]) / Math.max(1, m0.integ[i]))));
    scatter.hline(0, { color: "var(--qb-axis)", width: 1 });
    scatter.scatter(xs, ys, { r: 5, colors: sc.spots.map((_, i) => (th.dim.includes(i) ? "#d62728" : th.bright.includes(i) ? "#1f77b4" : "#888")) });
    scatter.legend([{ label: "dimmest third", color: "#d62728", kind: "dot" }, { label: "brightest third", color: "#1f77b4", kind: "dot" }], "br");
    const raw = M.RAW_BYTES;
    ro.textContent =
      `${dec.label}\n` +
      `size ${dec.bytes.toLocaleString("en-US").replace(/,/g, " ")} B, ${dec.bpp.toFixed(2)} bits/px (${(raw / dec.bytes).toFixed(1)}× smaller)\n` +
      `spots found ${m.det.found}/60, false ${m.det.falsePos}\n` +
      `dim spots: mean |change| ${M.meanAbsChange(m0.integ, m.integ, th.dim).toFixed(1)}%\n` +
      `bright spots: ${M.meanAbsChange(m0.integ, m.integ, th.bright).toFixed(1)}%\n` +
      `background ${m.bg.mean.toFixed(1)} (orig. ${m0.bg.mean.toFixed(1)})\n` +
      `noise sd ${m.bg.sd.toFixed(1)} (orig. ${m0.bg.sd.toFixed(1)})\n` +
      `HF power ${(100 * m.hf).toFixed(2)}% (orig. ${(100 * m0.hf).toFixed(2)}%)\n` +
      `blocking index ${M.blockingIndex(dec.image).toFixed(1)} (orig. ${bi0.toFixed(1)})`;
    cur = { bpp: dec.bpp, v: curveValue(m, dec) };
    drawCurve();
  }

  // ---- measurement against size over all JPEG qualities (computed in slices) ----
  const QS = Array.from({ length: 20 }, (_, i) => 5 * (i + 1));
  const curveData: { q: number; bpp: number; m: M.Measures }[] = [];
  let cur: { bpp: number; v: number } | null = null;
  function curveValue(m: M.Measures, _dec?: M.Decoded): number {
    switch (st.curve) {
      case "dim": return M.meanAbsChange(m0.integ, m.integ, th.dim);
      case "found": return m.det.found;
      case "false": return m.det.falsePos;
      case "bgsd": return m.bg.sd;
      default: return 100 * m.hf;
    }
  }
  function drawCurve(): void {
    (cFig.cap as HTMLElement).textContent = CURVE_LABEL[st.curve] + (curveData.length < QS.length ? ` (computing ${curveData.length}/${QS.length})` : "");
    curve.clear();
    const pts = [...curveData].sort((a, b) => a.bpp - b.bpp);
    const xs = pts.map(p => p.bpp), ys = pts.map(p => curveValue(p.m));
    const orig = curveValue(m0);
    const all = [...ys, orig, cur ? cur.v : orig];
    const ymax = Math.max(...all) * 1.1 || 1, ymin = Math.min(0, Math.min(...all));
    curve.setDomain([0.1, 6], [ymin, ymax]);
    curve.hline(orig, { color: "var(--qb-axis)", width: 1.5, dash: "5 4", text: "original" });
    if (xs.length) { curve.line(xs, ys, { color: "#1f77b4", width: 2.5 }); curve.scatter(xs, ys, { r: 3.5, color: "#1f77b4" }); }
    if (cur) {
      const m = st.codec === "jpeg" ? cur : null;
      if (m) curve.scatter([m.bpp], [m.v], { r: 8, color: "#ff7f0e" });
      else curve.scatter([cur.bpp], [cur.v], { r: 8, color: st.codec === "wavelet" ? "#2ca02c" : "#9467bd" });
    }
  }
  function sliceCurve(i: number): void {
    if (i >= QS.length) { drawCurve(); return; }
    const dec = jpegAt(QS[i]);
    curveData.push({ q: QS[i], bpp: dec.bpp, m: M.measure(dec.image, T, sc, bgMask) });
    drawCurve();
    window.setTimeout(() => sliceCurve(i + 1), 0);
  }

  syncUI();
  update();
  window.setTimeout(() => sliceCurve(0), 50);
}

autoMount(NAME, mount);
