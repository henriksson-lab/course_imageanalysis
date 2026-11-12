// Histogram and a global threshold. Left: the image with the mask [I > t] tinted orange
// (or [I ≤ t] for dark objects). Right: the histogram with a draggable threshold line,
// and below it the object fraction (mask area) 1 − P(t) as a function of t, with the
// cumulative histogram P(t) and Otsu's criterion η(t) = σ_B²/σ_T² as optional curves.
// Preprocessing: noise, median 3 × 3, Gaussian blur σ. Images: synthetic blobs (25 %
// objects, ground truth known: Dice shown) or the course micrograph.
// Auto-mounts on <div data-demo="histthresh">. Options: data-preset="clean|noise|blur|bins|cell|percentile|otsu|levels3",
// data-criterion="otsu" (show Otsu's criterion and t*), data-levels="3" (multi-level Otsu, two thresholds).
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { Grey } from "../lib/imageops";
import { intHistogram } from "../lib/histogram";
import { confusion, dice } from "../lib/metrics";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { drawOverlay, injectStyle, dragX } from "./scenes";
import { N, blobTruth, makeImage, analyse, percentileT } from "./model";

const NAME = "histthresh";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] .ht-img { flex: 0 1 300px; min-width: 200px; }
[data-demo="${NAME}"] .ht-img canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .ht-plots { flex: 1 1 380px; min-width: 260px; max-width: 470px; display: flex; flex-direction: column; gap: .1em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 320px; gap: .35em; }
[data-demo="${NAME}"] .ht-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .ht-credit { font-size: 12px; color: var(--qb-muted); }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; margin-top: .4em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .45em; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .15em; }
.reveal [data-demo="${NAME}"] .ht-lab { display: none; }
.reveal [data-demo="${NAME}"] .ht-img { flex: 0 0 290px; }
.reveal [data-demo="${NAME}"] .ht-plots { flex: 0 1 420px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 0 1 330px; max-width: 330px; }
`;

const ORANGE: [number, number, number] = [255, 127, 14];
const PURPLE: [number, number, number] = [148, 103, 189];

interface State {
  source: "blobs" | "cell";
  contrast: number; noise: number; blur: number; median: boolean;
  bins: number; t: number; t2: number; log: boolean; cumulative: boolean; criterion: boolean; levels3: boolean; dark: boolean;
}

const BASE: State = { source: "blobs", contrast: 100, noise: 8, blur: 0, median: false, bins: 256, t: 120, t2: 200,
  log: false, cumulative: false, criterion: false, levels3: false, dark: false };

const PRESETS: { id: string; label: string; s: Partial<State>; action?: "otsu" | "percentile" | "multi" }[] = [
  { id: "clean", label: "clean bimodal", s: { t: 120 } },
  { id: "noise", label: "noise fills the valley", s: { noise: 35, t: 110 } },
  { id: "blur", label: "blur restores it", s: { noise: 35, blur: 1.5, t: 110 } },
  { id: "bins", label: "too few bins", s: { bins: 8, contrast: 60, noise: 14, t: 95 } },
  { id: "cell", label: "the micrograph", s: { source: "cell", dark: true, noise: 0, t: 150 } },
  { id: "percentile", label: "percentile: brightest 25 %", s: { noise: 35 }, action: "percentile" },
  { id: "otsu", label: "Otsu's t*", s: { noise: 20, criterion: true }, action: "otsu" },
  { id: "levels3", label: "three classes", s: { source: "cell", dark: true, noise: 0, criterion: true, levels3: true }, action: "multi" },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const crit0 = host.dataset.criterion === "otsu", lev0 = host.dataset.levels === "3";
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? (lev0 ? PRESETS[7] : crit0 ? PRESETS[6] : PRESETS[0]);
  const st: State = { ...BASE };

  // ---- layout ----
  const fig = htmlEl("figure", "ht-img", "", host);
  const cap = htmlEl("figcaption", "", "", fig);
  const cv = htmlEl("canvas", "", "", fig);
  const ro = htmlEl("div", "qb-readout", "", fig);
  const credit = htmlEl("div", "ht-credit", CELL_CREDIT, fig);
  const plots = htmlEl("div", "ht-plots", "", host);
  const hp = new Plot(plots, { width: 470, height: 270, x: [0, 255], y: [0, 1], xlabel: "grey level", ylabel: "pixels per bin", xticks: [0, 50, 100, 150, 200, 250] });
  const ap = new Plot(plots, { width: 470, height: 200, x: [0, 255], y: [0, 1], xlabel: "threshold t", ylabel: "fraction", xticks: [0, 50, 100, 150, 200, 250], yticks: [0, 0.5, 1] });
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "ht-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const srcRow = htmlEl("div", "qb-presets", "", panel);
  const srcBtns = [button(srcRow, "synthetic blobs", () => { st.source = "blobs"; st.dark = false; sync(); remake(); }),
    button(srcRow, "micrograph", () => { st.source = "cell"; st.dark = true; sync(); remake(); })];
  const tS = slider(panel, "t", 0, 254, 1, st.t, v => { st.t = v; update(); });
  const t2S = slider(panel, "t₂", 0, 254, 1, st.t2, v => { st.t2 = v; update(); });
  const binS = slider(panel, "bins", 3, 8, 1, 8, v => { st.bins = 2 ** v; analyseAndDraw(); }, v => String(2 ** v));
  const conS = slider(panel, "contrast", 10, 150, 5, st.contrast, v => { st.contrast = v; remake(); });
  const noiS = slider(panel, "noise σ", 0, 50, 1, st.noise, v => { st.noise = v; remake(); });
  const bluS = slider(panel, "blur σ", 0, 3, 0.25, st.blur, v => { st.blur = v; remake(); });
  const row1 = htmlEl("div", "qb-presets", "", panel);
  const medC = checkbox(row1, "median 3×3", st.median, v => { st.median = v; remake(); });
  const logC = checkbox(row1, "log scale", st.log, v => { st.log = v; analyseAndDraw(); });
  const row2 = htmlEl("div", "qb-presets", "", panel);
  const cumC = checkbox(row2, "P(t)", st.cumulative, v => { st.cumulative = v; analyseAndDraw(); });
  const criC = checkbox(row2, "Otsu", st.criterion, v => { st.criterion = v; analyseAndDraw(); });
  const levC = checkbox(row2, "3 classes", st.levels3, v => { st.levels3 = v; analyseAndDraw(); });
  const row3 = htmlEl("div", "qb-presets", "", panel);
  const darkC = checkbox(row3, "dark objects", st.dark, v => { st.dark = v; update(); });
  button(row3, "t = Otsu", () => { st.t = otsuT; if (st.levels3) { st.t = multi.t1v; st.t2 = multi.t2v; } sync(); update(); });
  button(row3, "reset", () => load(p0));

  function sync(): void {
    tS.set(st.t); t2S.set(st.t2); binS.set(Math.round(Math.log2(st.bins))); conS.set(st.contrast); noiS.set(st.noise); bluS.set(st.blur);
    medC.checked = st.median; logC.checked = st.log; cumC.checked = st.cumulative; criC.checked = st.criterion; levC.checked = st.levels3; darkC.checked = st.dark;
    srcBtns.forEach((b, i) => b.classList.toggle("active", i === (st.source === "blobs" ? 0 : 1)));
    (t2S.input.parentElement as HTMLElement).style.display = st.levels3 ? "" : "none";
    (conS.input.parentElement as HTMLElement).style.display = st.source === "blobs" ? "" : "none";
    credit.style.display = st.source === "cell" ? "" : "none";
  }

  function load(p: typeof PRESETS[number]): void {
    Object.assign(st, BASE, { criterion: crit0, levels3: lev0 }, p.s);
    pendingAction = p.action ?? null;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    sync();
    remake();
  }

  // ---- images ----
  const truth = blobTruth();
  let cell: Grey | null = null;
  let img: Grey = makeImage({ ...st, source: "blobs" }, truth, null);
  let pendingAction: "otsu" | "percentile" | "multi" | null = null;

  function remake(): void {
    if (st.source === "cell" && !cell) { loadCell().then(c => { cell = c; remake(); }); return; }
    img = makeImage(st, truth, cell);
    analyseAndDraw();
  }

  // ---- analysis ----
  let h256 = intHistogram([0], 0, 255), otsuT = 128, multi = { t1v: 85, t2v: 170 };
  function analyseAndDraw(): void {
    const A = analyse(img, st.bins), hb = A.hb, w = A.w, o = A.otsu;
    h256 = A.h256; otsuT = A.otsuT; multi = A.multi;
    if (pendingAction === "otsu") st.t = otsuT;
    if (pendingAction === "multi") { st.t = multi.t1v; st.t2 = multi.t2v; }
    if (pendingAction === "percentile") st.t = percentileT(h256, 0.25);
    pendingAction = null;
    sync();

    // histogram plot
    hp.clear();
    const edges = Float64Array.from({ length: st.bins + 1 }, (_, i) => i * w - 0.5);
    let ymax = 0;
    for (const c of hb.counts) ymax = Math.max(ymax, c);
    if (st.log) { hp.opts.ylog = true; hp.setDomain([0, 255], [0.8, Math.max(10, ymax * 1.5)]); }
    else { hp.opts.ylog = false; hp.setDomain([0, 255], [0, Math.max(1, ymax * 1.08)]); }
    hp.histogram(edges, Float64Array.from(hb.counts, c => (st.log ? Math.max(c, 0.8) : c)), { color: seriesColor(0) });
    if (st.cumulative) {
      const xs: number[] = [], ys: number[] = [];
      let s = 0;
      for (let v = 0; v < 256; v++) { s += h256.counts[v]; xs.push(v); ys.push(s / h256.total); }
      const top = st.log ? Math.max(10, ymax * 1.5) : ymax * 1.08, bot = st.log ? 0.8 : 0;
      hp.line(xs, ys.map(y => (st.log ? bot * (top / bot) ** y : y * top)), { color: seriesColor(2), width: 2.5, label: "P(t) (scaled)" });
    }
    if (st.criterion && !st.levels3) hp.vline(otsuT, { color: seriesColor(4), text: "Otsu", dash: "3 3", width: 2 });
    tLine = hp.vline(st.t, { color: seriesColor(3), text: "t", width: 3.5, dash: "" });
    t2Line = st.levels3 ? hp.vline(st.t2, { color: "rgb(148,103,189)", text: "t₂", width: 3.5, dash: "" }) : null;

    // fraction plot
    ap.clear();
    const xs: number[] = [], area: number[] = [], P: number[] = [];
    let s = 0;
    for (let v = 0; v < 255; v++) { s += h256.counts[v]; xs.push(v); P.push(s / h256.total); area.push(1 - s / h256.total); }
    ap.line(xs, st.dark ? P : area, { color: seriesColor(3), label: "mask area" });
    if (st.cumulative) ap.line(xs, P, { color: seriesColor(2), width: 2, dash: "6 4", label: "P(t)" });
    if (st.criterion) {
      const cx: number[] = [], cy: number[] = [];
      for (let k = 0; k < o.curve.length; k++) if (Number.isFinite(o.curve[k])) { cx.push((k + 1) * w - 1); cy.push(o.curve[k] / (o.varT || 1)); }
      ap.line(cx, cy, { color: seriesColor(4), width: 2.5, label: "Otsu η(t)" });
    }
    ap.legend(undefined, st.dark ? "tl" : "tr");
    aLine = ap.vline(st.t, { color: seriesColor(3), width: 3, dash: "" });
    update();
  }

  let tLine: ReturnType<Plot["vline"]> | null = null, t2Line: ReturnType<Plot["vline"]> | null = null, aLine: ReturnType<Plot["vline"]> | null = null;

  // ---- drawing ----
  function update(): void {
    tLine?.set(st.t); t2Line?.set(st.t2); aLine?.set(st.t);
    tS.set(st.t); t2S.set(st.t2);
    const n = N * N, mask = new Uint8Array(n), mask2 = new Uint8Array(n);
    const lo = Math.min(st.t, st.t2), hi = Math.max(st.t, st.t2);
    for (let i = 0; i < n; i++) {
      const v = img.data[i];
      if (st.levels3) { mask[i] = v > lo && v <= hi ? 1 : 0; mask2[i] = v > hi ? 1 : 0; if (st.dark) { mask2[i] = v <= lo ? 1 : 0; } }
      else mask[i] = st.dark ? (v <= st.t ? 1 : 0) : (v > st.t ? 1 : 0);
    }
    drawOverlay(cv, img, st.levels3 ? [{ mask, rgb: ORANGE, alpha: 0.45 }, { mask: mask2, rgb: PURPLE, alpha: 0.55 }] : [{ mask, rgb: ORANGE, alpha: 0.5 }]);
    cap.textContent = st.levels3 ? "three classes: orange = middle, purple = " + (st.dark ? "darkest" : "brightest")
      : st.dark ? "mask: I ≤ t (dark objects)" : "mask: I > t";
    let frac = 0;
    for (let i = 0; i < n; i++) frac += mask[i];
    frac /= n;
    let s = 0;
    for (let v = 0; v <= st.t; v++) s += h256.counts[v];
    const lines = [`t = ${st.t}   P(t) = ${(s / h256.total).toFixed(3)}`, `${st.levels3 ? "orange" : "mask"} area = ${(100 * frac).toFixed(1)} %`];
    if (st.criterion) lines.push(st.levels3 ? `3-class Otsu: t₁ = ${multi.t1v}, t₂ = ${multi.t2v}` : `Otsu t* = ${otsuT}`);
    if (st.source === "blobs" && !st.levels3 && !st.dark) lines.push(`Dice vs truth = ${dice(confusion(mask, truth)).toFixed(3)}`);
    ro.textContent = lines.join("\n");
  }

  dragX(hp.svg, ev => hp.toData(ev)[0], x => {
    const v = Math.max(0, Math.min(254, Math.round(x)));
    if (st.levels3 && Math.abs(v - st.t2) < Math.abs(v - st.t)) st.t2 = v; else st.t = v;
    presetBtns.forEach(b => b.classList.remove("active"));
    update();
  });
  dragX(ap.svg, ev => ap.toData(ev)[0], x => { st.t = Math.max(0, Math.min(254, Math.round(x))); update(); });

  load(p0);
}

autoMount(NAME, mount);
