// Display vs data: what contrast transforms do to measurements. A synthetic control/treated
// pair (12 bit, the same nuclei, treated nuclei 1.5× brighter above the offset of 100,
// Poisson noise). The same transform is applied to both images; the nuclei are segmented
// on the displayed images (Otsu); a table measures both and gives treated/control, green
// when within ±2 % of the truth (1.50 for intensities, 1.00 for areas). Toggle: measure the
// displayed values, or the raw values inside the masks found on the display.
// Auto-mounts on <div data-demo="displayvsdata">. Options: data-preset (raw|same|auto|gamma|
// equalise|match|clahe|segraw).
import { htmlEl, presets, button, checkbox, autoMount } from "../lib/ui";
import { histogram } from "../lib/histogram";
import { otsu } from "../lib/threshold";
import * as C from "../lib/contrast";

const NAME = "displayvsdata";
const P = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${P} { gap: .8em; }
${P} figure { margin: 0; }
${P} figcaption { font-size: 16px; text-align: center; margin: 0 0 .1em; }
${P} .dv-left { flex: 1 1 560px; min-width: 300px; max-width: 640px; display: flex; flex-direction: column; gap: .3em; }
${P} .dv-imgs { display: flex; gap: .6em; }
${P} .dv-imgs figure { flex: 1 1 0; min-width: 0; }
${P} .dv-imgs canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
${P} table.dv-table { border-collapse: collapse; font-size: 15px; width: 100%; }
${P} table.dv-table th, ${P} table.dv-table td { padding: .1em .45em; border-bottom: 1px solid var(--qb-grid, #ddd); text-align: right; }
${P} table.dv-table th:first-child, ${P} table.dv-table td:first-child { text-align: left; }
${P} table.dv-table td.ok { color: #1a7f37; font-weight: 700; }
${P} table.dv-table td.bad { color: #c62828; font-weight: 700; }
${P} .qb-panel { flex: 0 1 330px; gap: .3em; }
${P} .qb-readout { font-size: 14px; }
${P} .dv-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal ${P} { font-size: 18px; }
.reveal ${P} figcaption, .reveal ${P} .qb-figure figcaption { font-size: 17px; }
.reveal ${P} .qb-btn { font-size: 16px; padding: .05em .4em; }
.reveal ${P} .dv-lab { font-size: 15px; }
.reveal ${P} .qb-panel { gap: .15em; }
.reveal ${P} table.dv-table { font-size: 16px; }
.reveal ${P} .qb-readout { font-size: 15px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

type Tf = "none" | "same" | "auto" | "gamma" | "log" | "equalise" | "clahe" | "match";
interface State { tf: Tf; raw: boolean }
const TFS: { id: Tf; label: string }[] = [
  { id: "none", label: "none (raw)" }, { id: "same", label: "same window" }, { id: "auto", label: "auto-window each" },
  { id: "gamma", label: "gamma 0.5" }, { id: "log", label: "log" }, { id: "equalise", label: "equalise each" },
  { id: "clahe", label: "CLAHE each (6 × 6, β = 6)" }, { id: "match", label: "match to control" },
];
const PRESETS: { id: string; label: string; s: State }[] = [
  { id: "raw", label: "raw: 1.50", s: { tf: "none", raw: false } },
  { id: "same", label: "same window: still 1.50", s: { tf: "same", raw: false } },
  { id: "auto", label: "auto-windows: the figure lies", s: { tf: "auto", raw: false } },
  { id: "gamma", label: "gamma: the ratio changes", s: { tf: "gamma", raw: false } },
  { id: "equalise", label: "equalise each: treatment gone", s: { tf: "equalise", raw: false } },
  { id: "match", label: "match: treatment erased", s: { tf: "match", raw: false } },
  { id: "clahe", label: "CLAHE: cells not comparable", s: { tf: "clahe", raw: false } },
  { id: "segraw", label: "segment on CLAHE, measure raw", s: { tf: "clahe", raw: true } },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...p0.s };
  const S = C.controlTreated();
  const W = S.w, H = S.h, n = W * H;
  const ctl = Float32Array.from(S.control), trt = Float32Array.from(S.treated);

  const left = htmlEl("div", "dv-left", "", host);
  const imgs = htmlEl("div", "dv-imgs", "", left);
  const f1 = htmlEl("figure", "", "", imgs), f2 = htmlEl("figure", "", "", imgs);
  htmlEl("figcaption", "", "control", f1); htmlEl("figcaption", "", "treated (nuclei 1.5× brighter)", f2);
  const cv1 = htmlEl("canvas", "", "", f1), cv2 = htmlEl("canvas", "", "", f2);
  cv1.width = cv2.width = W; cv1.height = cv2.height = H;
  const table = htmlEl("table", "dv-table", "", left);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "dv-lab", "presets", panel);
  const pBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, p.s); pBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p)); sync(); update(); } })));
  htmlEl("div", "dv-lab", "transform (applied to both)", panel);
  const tRow = htmlEl("div", "qb-presets", "", panel);
  const tBtns = TFS.map(t => button(tRow, t.label, () => { st.tf = t.id; sync(); update(); }));
  const rawCb = checkbox(panel, "measure raw values (masks from the display)", st.raw, v => { st.raw = v; update(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  button(panel, "reset", () => { Object.assign(st, p0.s); sync(); update(); });

  function sync(): void {
    tBtns.forEach((b, i) => b.classList.toggle("active", TFS[i].id === st.tf));
    rawCb.checked = st.raw;
  }

  const maxOf = (a: Float32Array) => a.reduce((m, v) => Math.max(m, v), -Infinity);
  const minOf = (a: Float32Array) => a.reduce((m, v) => Math.min(m, v), Infinity);
  const r8 = (a: Float32Array) => Float32Array.from(a, v => Math.max(0, Math.min(255, Math.round(v))));

  /** Displayed values (0..255, except "none": raw) for control and treated. */
  function transform(): [Float32Array, Float32Array] {
    const top = Math.max(maxOf(ctl), maxOf(trt));
    switch (st.tf) {
      case "none": return [ctl, trt];
      case "same": return [r8(ctl.map(v => 255 * v / top)), r8(trt.map(v => 255 * v / top))];
      case "auto": {
        const w = (a: Float32Array) => { const lo = minOf(a), hi = maxOf(a); return r8(a.map(v => 255 * (v - lo) / (hi - lo))); };
        return [w(ctl), w(trt)];
      }
      case "gamma": return [r8(ctl.map(v => 255 * Math.sqrt(v / 4095))), r8(trt.map(v => 255 * Math.sqrt(v / 4095)))];
      case "log": { const k = 255 / Math.log(4096); return [r8(ctl.map(v => k * Math.log1p(v))), r8(trt.map(v => k * Math.log1p(v)))]; }
      case "equalise": return [C.equaliseValues(ctl, 256), C.equaliseValues(trt, 256)];
      case "clahe": {
        const f = (a: Float32Array) => r8(C.clahe({ width: W, height: H, data: a }, { tilesX: 6, tilesY: 6, clip: 6, bins: 256 }).out.data);
        return [f(ctl), f(trt)];
      }
      case "match": {
        const lut = C.matchLut(C.levelHistogram(trt, 4096), C.levelHistogram(ctl, 4096));
        return [r8(ctl.map(v => 255 * v / top)), r8(C.applyLut(trt, lut).map(v => 255 * v / top))];
      }
    }
  }

  /** Otsu mask of a displayed image. */
  function segment(D: Float32Array): { mask: Uint8Array; thr: number } {
    const h = histogram(D, { bins: 256 });
    const t = otsu(h.counts).t;
    const thr = h.lo + (t + 1) * h.binWidth;
    return { mask: Uint8Array.from(D, v => (v >= thr ? 1 : 0)), thr };
  }

  function draw(cv: HTMLCanvasElement, D: Float32Array, mask: Uint8Array, scale: number): void {
    const ctx = cv.getContext("2d")!, id = ctx.createImageData(W, H);
    for (let i = 0; i < n; i++) {
      const g = Math.max(0, Math.min(255, Math.round(D[i] * scale)));
      // mask outline: a pixel in the mask with a neighbour outside
      const x = i % W, y = (i - x) / W;
      const edge = mask[i] && ((x > 0 && !mask[i - 1]) || (x < W - 1 && !mask[i + 1]) || (y > 0 && !mask[i - W]) || (y < H - 1 && !mask[i + W]));
      id.data[4 * i] = edge ? 255 : g; id.data[4 * i + 1] = edge ? 127 : g; id.data[4 * i + 2] = edge ? 14 : g; id.data[4 * i + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
  }

  function update(): void {
    const [Dc, Dt] = transform();
    const sc = segment(Dc), stt = segment(Dt);
    const scale = st.tf === "none" ? 255 / Math.max(maxOf(ctl), maxOf(trt)) : 1;
    draw(cv1, Dc, sc.mask, scale); draw(cv2, Dt, stt.mask, scale);
    // values measured: displayed, or raw inside the masks found on the display
    const Vc = st.raw ? ctl : Dc, Vt = st.raw ? trt : Dt;
    const meas = (V: Float32Array, D: Float32Array, mask: Uint8Array, fixedThr: number) => {
      const bg = C.maskedMean(V, S.background);
      const mean = C.maskedMean(V, mask) - bg;
      let tot = 0, area = 0, fixed = 0;
      for (let i = 0; i < n; i++) { if (mask[i]) { tot += V[i] - bg; area++; } if (D[i] >= fixedThr) fixed++; }
      return { mean, tot, med: C.maskedMedian(V, mask) - bg, area, fixed };
    };
    const mc = meas(Vc, Dc, sc.mask, sc.thr), mt = meas(Vt, Dt, stt.mask, sc.thr);
    const rows: [string, number, number, number, number][] = [
      ["mean in nuclei − background", mc.mean, mt.mean, 1.5, 1],
      ["total in nuclei − background", mc.tot, mt.tot, 1.5, 0],
      ["median in nuclei − background", mc.med, mt.med, 1.5, 1],
      ["nucleus area (Otsu on each display)", mc.area, mt.area, 1, 0],
      ["area above the control's threshold", mc.fixed, mt.fixed, 1, 0],
    ];
    const fmt = (v: number, d: number) => Math.abs(v) >= 1e5 ? v.toExponential(2) : v.toFixed(d);
    table.innerHTML = "";
    const hr = htmlEl("tr", "", "", table);
    for (const h of ["measured " + (st.raw ? "(raw values)" : "(displayed values)"), "control", "treated", "ratio", "true"]) htmlEl("th", "", h, hr);
    for (const [label, a, b, truth, d] of rows) {
      const tr = htmlEl("tr", "", "", table), q = b / a;
      htmlEl("td", "", label, tr); htmlEl("td", "", fmt(a, d), tr); htmlEl("td", "", fmt(b, d), tr);
      htmlEl("td", Number.isFinite(q) && Math.abs(q / truth - 1) <= 0.02 ? "ok" : "bad", Number.isFinite(q) ? q.toFixed(2) : "–", tr);
      htmlEl("td", "", truth.toFixed(2), tr);
    }
    // within one image: brightest vs dimmest nucleus of the control
    const cellMean = (V: Float32Array, k: number, mask: Uint8Array) => {
      let s = 0, m = 0;
      for (let i = 0; i < n; i++) if (S.cellOf[i] === k && mask[i]) { s += V[i]; m++; }
      return m ? s / m - C.maskedMean(V, S.background) : NaN;
    };
    const kb = S.signal.indexOf(Math.max(...S.signal)) + 1, kd = S.signal.indexOf(Math.min(...S.signal)) + 1;
    const trueR = S.signal[kb - 1] / S.signal[kd - 1], measR = cellMean(Vc, kb, sc.mask) / cellMean(Vc, kd, sc.mask);
    ro.textContent = [
      "control: brightest / dimmest nucleus",
      `true ${trueR.toFixed(2)}, measured ${Number.isFinite(measR) ? measR.toFixed(2) : "– (missed)"}`,
      `Otsu threshold: ${sc.thr.toFixed(0)} / ${stt.thr.toFixed(0)}`,
    ].join("\n");
  }

  sync();
  pBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0));
  update();
}

autoMount(NAME, mount);
