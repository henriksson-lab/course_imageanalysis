// Cellpose 3 (cyto3 / nuclei U-net) running in the browser: Rust compiled to WebAssembly
// (demos/wasm/cellpose), weights and sample images from the course site (lib/wasm.ts:
// progress bar, Cache API, file:// message). Shows what the network predicts and how the
// masks are decoded: input, flows (colour wheel + arrows), cell probability with its
// threshold, tracking (paths of pixels, where they gather = sinks, seeds), masks, and the
// flow-error check. Built-in samples or the student's own image (read locally, never
// uploaded). Tiles run in parallel Web Workers (this bundle is also the worker script).
// Auto-mounts on <div data-demo="cellposenet">. Options: data-sample (key of SAMPLES),
// data-view (input|flows|prob|track|masks|check), data-diameter (overrides the sample's),
// data-workers (default: cores - 1, max 4), data-autorun="1" (load the network and run at
// once, else on the Run button).
import { htmlEl, slider, presets, button, checkbox, autoMount, type Slider } from "../lib/ui";
import { scalarToRGBA, viridis, labelColor } from "../lib/colour";
import { iouTable, match, objectScores } from "../lib/eval";
import { loadModelFile, loadStatus, modelIndex, isFileProtocol, fileProtocolMessage, formatBytes, distBase } from "../lib/wasm";
import * as P from "./pipeline";
import { decodePng, inflateBrowser } from "./png";
import { workerMain, Pool, type MasksOut } from "./worker";

const NAME = "cellposenet";

// The script URL, captured while this classic script runs (the workers load the same file).
const SCRIPT_URL: string = (() => {
  try {
    const s = typeof document !== "undefined" ? (document.currentScript as HTMLScriptElement | null) : null;
    return s?.src ?? "";
  } catch { return ""; }
})();

interface Sample {
  key: string; label: string; file: string; labels?: string;
  model: P.ModelKey; chan: [number, number]; diameter: number; invert?: boolean;
}
const SAMPLES: Sample[] = [
  { key: "cells", label: "Cells: cytoplasm (green) + nuclei (blue)", file: "images/cellpose-img02.png", model: "cyto3", chan: [2, 3], diameter: 30 },
  { key: "crowded", label: "Crowded cells, two channels", file: "images/cellpose-img00.png", model: "cyto3", chan: [2, 3], diameter: 35 },
  { key: "membranes", label: "Cell membranes, one channel", file: "images/cellpose-img03.png", model: "cyto3", chan: [0, 0], diameter: 34 },
  { key: "nuclei", label: "Nuclei (with ground truth)", file: "images/bbbc039-A02_s1.png", labels: "images/bbbc039-A02_s1-labels.png", model: "nuclei", chan: [0, 0], diameter: 29 },
  { key: "dense", label: "Dense nuclei (with ground truth)", file: "images/dsb-fluo-dense.png", labels: "images/dsb-fluo-dense-labels.png", model: "nuclei", chan: [0, 0], diameter: 22 },
  { key: "small", label: "Small nuclei (with ground truth)", file: "images/dsb-fluo-small.png", labels: "images/dsb-fluo-small-labels.png", model: "nuclei", chan: [0, 0], diameter: 13 },
  { key: "tissue", label: "Stained tissue section (with ground truth)", file: "images/dsb-histology.png", labels: "images/dsb-histology-labels.png", model: "nuclei", chan: [0, 0], diameter: 24, invert: true },
];

type View = "input" | "flows" | "prob" | "track" | "masks" | "check";
const VIEWS: { v: View; label: string; caption: string }[] = [
  { v: "input", label: "Input", caption: "The image the network sees (after choosing channels; normalised so the 1st and 99th percentiles are 0 and 1)." },
  { v: "flows", label: "Flows", caption: "Predicted flow: colour = direction in which each pixel should move to reach the centre of its cell, brightness = strength. Arrows on cell pixels." },
  { v: "prob", label: "Cell probability", caption: "Predicted cell probability (as a logit: 0 means 50 %). White line: the threshold; only pixels inside are tracked." },
  { v: "track", label: "Tracking", caption: "Every cell pixel follows the flow for the chosen number of steps. Lines: paths of some pixels; red: where pixels end up (the sinks); +: seeds. Click a pixel to follow it." },
  { v: "masks", label: "Masks", caption: "Final masks: all pixels that ended in the same sink form one object (then holes filled, small masks removed)." },
  { v: "check", label: "Flow check", caption: "Flows recomputed from each mask are compared with the network's flows. Green: kept; red: error above the flow threshold (removed); orange: removed as too small. Numbers: flow error." },
];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .cp-left { flex: 1 1 520px; min-width: 280px; max-width: 640px; display: flex; flex-direction: column; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .cp-tabs { display: flex; flex-wrap: wrap; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .cp-stage { position: relative; width: 100%; background: #000; border-radius: 4px; overflow: hidden; touch-action: none; }
.qb-demo[data-demo="${NAME}"] .cp-stage canvas { display: block; width: 100%; height: auto; cursor: crosshair; }
.qb-demo[data-demo="${NAME}"] .cp-busy { position: absolute; left: 0; right: 0; bottom: 0; padding: .3em .6em; background: rgba(0,0,0,.65); color: #fff; font-size: 16px; }
.qb-demo[data-demo="${NAME}"] .cp-caption { font-size: 15px; color: var(--qb-muted); min-height: 2.6em; }
.qb-demo[data-demo="${NAME}"] .cp-credit { font-size: 13px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 340px; max-width: 460px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .cp-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
.qb-demo[data-demo="${NAME}"] select { font: inherit; font-size: 15px; max-width: 100%; padding: .1em .2em; }
.qb-demo[data-demo="${NAME}"] .cp-lab { font-size: 14px; color: var(--qb-muted); margin-top: .2em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; line-height: 1.35; }
.qb-demo[data-demo="${NAME}"] .cp-run { font-weight: 600; }
.qb-demo[data-demo="${NAME}"] .cp-err { color: var(--qb-c2); font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .cp-hide { display: none !important; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .cp-left { flex: 0 1 560px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .08em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .cp-caption { font-size: 15px; min-height: 0; }
.reveal .qb-demo[data-demo="${NAME}"] .cp-lab, .reveal .qb-demo[data-demo="${NAME}"] .cp-credit { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { gap: .15em; font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; padding: .2em .4em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

/** The loaded image: planes (0..255 or 0..65535), and an RGBA picture for display. */
interface Loaded { img: P.Img; rgba: Uint8ClampedArray; credit: string; labels?: Uint32Array; isRGB: boolean; note?: string }

async function loadSample(s: Sample): Promise<Loaded> {
  const bytes = await loadModelFile(s.file);
  const png = await decodePng(bytes, inflateBrowser);
  const { w, h, channels } = png;
  const nc = channels >= 3 ? 3 : 1;
  const planes = Array.from({ length: nc }, () => new Float32Array(w * h));
  const rgba = new Uint8ClampedArray(4 * w * h);
  const sc = png.depth === 16 ? 1 / 257 : 1;
  for (let i = 0; i < w * h; i++) {
    for (let c = 0; c < nc; c++) planes[c][i] = png.data[i * channels + c];
    for (let c = 0; c < 3; c++) rgba[4 * i + c] = png.data[i * channels + (nc === 3 ? c : 0)] * sc;
    rgba[4 * i + 3] = 255;
  }
  let credit = "";
  try { credit = String((await modelIndex())[s.file]?.credit ?? ""); } catch { /* no index */ }
  let labels: Uint32Array | undefined;
  if (s.labels) {
    try {
      const lp = await decodePng(await loadModelFile(s.labels), inflateBrowser);
      labels = Uint32Array.from(lp.data.filter((_, i) => i % lp.channels === 0));
    } catch { labels = undefined; }
  }
  return { img: { w, h, planes }, rgba, credit, labels, isRGB: nc === 3 };
}

const MAX_OWN = 768;

async function loadOwn(file: File): Promise<Loaded> {
  const bmp = await createImageBitmap(file);
  let w = bmp.width, h = bmp.height, note = "";
  const s = Math.min(1, MAX_OWN / Math.max(w, h));
  if (s < 1) { note = `scaled from ${w} × ${h} to fit ${MAX_OWN} pixels`; w = Math.round(w * s); h = Math.round(h * s); }
  const cv = document.createElement("canvas");
  cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d")!;
  ctx.drawImage(bmp, 0, 0, w, h);
  const rgba = ctx.getImageData(0, 0, w, h).data;
  let grey = true;
  for (let i = 0; i < w * h && grey; i++) if (rgba[4 * i] !== rgba[4 * i + 1] || rgba[4 * i] !== rgba[4 * i + 2]) grey = false;
  const nc = grey ? 1 : 3;
  const planes = Array.from({ length: nc }, () => new Float32Array(w * h));
  for (let i = 0; i < w * h; i++) for (let c = 0; c < nc; c++) planes[c][i] = rgba[4 * i + c];
  return { img: { w, h, planes }, rgba: new Uint8ClampedArray(rgba), credit: `Your image (${file.name}), read in this browser only`, isRGB: !grey, note };
}

/** Label colour in 0..255 (lib/colour's labelColor is 0..1), cached. */
const palette = new Map<number, [number, number, number]>();
function labRGB(L: number): [number, number, number] {
  let c = palette.get(L);
  if (!c) { const v = labelColor(L); c = [v[0] * 255, v[1] * 255, v[2] * 255]; palette.set(L, c); }
  return c;
}

interface Params { diameter: number; cellprob: number; flow: number; minSize: number; niterAuto: boolean; niter: number }
const DEFAULTS: Params = { diameter: 30, cellprob: 0, flow: 0.4, minSize: 15, niterAuto: true, niter: 200 };

function mount(host: HTMLElement): void {
  injectStyle();
  const startSample = SAMPLES.find((s) => s.key === host.dataset.sample) ?? SAMPLES[0];
  let view: View = (VIEWS.find((v) => v.v === host.dataset.view)?.v) ?? "masks";
  const nWorkers = Math.max(1, Math.min(8, Number(host.dataset.workers) ||
    Math.min(4, Math.max(1, (navigator.hardwareConcurrency || 2) - 1))));

  // ---- layout
  const left = htmlEl("div", "cp-left", "", host);
  const tabs = htmlEl("div", "cp-tabs", "", left);
  const stage = htmlEl("div", "cp-stage", "", left);
  const canvas = htmlEl("canvas", "", "", stage);
  const busy = htmlEl("div", "cp-busy cp-hide", "", stage);
  const caption = htmlEl("div", "cp-caption", "", left);
  const credit = htmlEl("div", "cp-credit", "", left);
  const panel = htmlEl("div", "qb-panel", "", host);

  const tabBtns = VIEWS.map((v) => button(tabs, v.label, () => setView(v.v)));
  function setView(v: View): void {
    view = v;
    tabBtns.forEach((b, i) => b.classList.toggle("active", VIEWS[i].v === v));
    caption.textContent = VIEWS.find((x) => x.v === v)!.caption;
    if (v === "track") ensurePaths(); else draw();
  }

  // image choice
  htmlEl("div", "cp-lab", "Image", panel);
  const rowImg = htmlEl("div", "cp-row", "", panel);
  const sel = htmlEl("select", "", "", rowImg);
  for (const s of SAMPLES) { const o = htmlEl("option", "", s.label, sel); o.value = s.key; }
  const oOwn = htmlEl("option", "", "Your own image…", sel); oOwn.value = "own";
  sel.value = startSample.key;
  const fileIn = htmlEl("input", "cp-hide", "", rowImg);
  fileIn.type = "file"; fileIn.accept = "image/*";
  button(rowImg, "Open your image…", () => fileIn.click());

  // model and channels
  htmlEl("div", "cp-lab", "Network and input channels", panel);
  const rowModel = htmlEl("div", "cp-row", "", panel);
  const modelBtns = (Object.keys(P.MODELS) as P.ModelKey[]).map((k) => {
    const b = button(rowModel, P.MODELS[k].label, () => { setModel(k); scheduleNet(); });
    b.dataset.key = k;
    return b;
  });
  const rowChan = htmlEl("div", "cp-row", "", panel);
  htmlEl("span", "", "segment:", rowChan);
  const selC0 = htmlEl("select", "", "", rowChan);
  [["0", "grey"], ["1", "red"], ["2", "green"], ["3", "blue"]].forEach(([v, t]) => { const o = htmlEl("option", "", t, selC0); o.value = v; });
  htmlEl("span", "", "nuclei:", rowChan);
  const selC1 = htmlEl("select", "", "", rowChan);
  [["0", "none"], ["1", "red"], ["2", "green"], ["3", "blue"]].forEach(([v, t]) => { const o = htmlEl("option", "", t, selC1); o.value = v; });
  for (const s of [selC0, selC1]) s.addEventListener("change", () => { s.blur(); scheduleNet(); });
  const invertBox = checkbox(panel, "invert (dark objects on a bright background)", false, () => scheduleNet());

  // parameters
  htmlEl("div", "cp-lab", "Parameters", panel);
  const sDiam: Slider = slider(panel, "cell diameter (px)", 5, 100, 1, DEFAULTS.diameter, () => scheduleNet(400));
  const sProb: Slider = slider(panel, "cell prob. threshold", -6, 6, 0.5, DEFAULTS.cellprob, () => scheduleDecode());
  const sFlow: Slider = slider(panel, "flow threshold", 0, 3, 0.1, DEFAULTS.flow, () => scheduleFinish(), (v) => (v === 0 ? "off" : v.toFixed(1)));
  const sMin: Slider = slider(panel, "min. size (px)", 0, 200, 5, DEFAULTS.minSize, () => scheduleFinish());
  const sIter: Slider = slider(panel, "Euler steps", 0, 400, 5, DEFAULTS.niter, () => { autoIter.checked = false; scheduleDecode(); });
  const autoIter = checkbox(panel, "steps = 200 × diameter / training diameter (Cellpose default)", true, () => { syncIter(); scheduleDecode(); });
  presets(panel, [
    { label: "Defaults", apply: () => setParams({ cellprob: 0, flow: 0.4, minSize: 15, niterAuto: true }) },
    { label: "More cells", apply: () => setParams({ cellprob: -2, flow: 0.8, minSize: 15, niterAuto: true }) },
    { label: "Stricter", apply: () => setParams({ cellprob: 1, flow: 0.2, minSize: 30, niterAuto: true }) },
    { label: "No flow check", apply: () => setParams({ flow: 0 }) },
    { label: "Only 10 steps", apply: () => setParams({ niterAuto: false, niter: 10 }) },
  ]);

  const rowRun = htmlEl("div", "cp-row", "", panel);
  const runBtn = button(rowRun, "Run Cellpose", () => { started = true; scheduleNet(0); }, "cp-run");
  const estBtn = button(rowRun, "Estimate diameter", () => estimateDiameter());
  const status = htmlEl("div", "", "", panel);
  const err = htmlEl("div", "cp-err", "", panel);
  const readout = htmlEl("div", "qb-readout", "", panel);

  // ---- state
  let loaded: Loaded | null = null;
  let sample: Sample | null = startSample;
  let model: P.ModelKey = startSample.model;
  let pool: Pool | null = null;
  let poolModel: P.ModelKey | null = null;
  let net: P.NetOutput | null = null;
  let netKey = "";
  let dec: MasksOut | null = null;
  let finalMasks: Uint32Array | null = null;
  let paths: { starts: Uint32Array; xy: Float32Array; per: number } | null = null;
  const clicked: number[] = [];
  let clickedPaths: Float32Array | null = null;
  let started = host.dataset.autorun === "1";
  let sizeModel: P.SizeModel | null = null;
  let netMs = 0, tileMs: number[] = [], loadInfo = "";

  function setModel(k: P.ModelKey): void {
    model = k;
    modelBtns.forEach((b) => b.classList.toggle("active", b.dataset.key === k));
    estBtn.disabled = !P.MODELS[k].sizeFile;
    syncIter();
  }
  function rescale(): number { return P.MODELS[model].diamMean / sDiam.get(); }
  function syncIter(): void { if (autoIter.checked) sIter.set(Math.min(400, P.defaultNiter(rescale()))); }
  function setParams(p: Partial<Params>): void {
    if (p.cellprob !== undefined) sProb.set(p.cellprob);
    if (p.flow !== undefined) sFlow.set(p.flow);
    if (p.minSize !== undefined) sMin.set(p.minSize);
    if (p.niterAuto !== undefined) autoIter.checked = p.niterAuto;
    if (p.niter !== undefined) sIter.set(p.niter);
    syncIter();
    scheduleDecode();
  }
  function applySample(s: Sample): void {
    selC0.value = String(s.chan[0]); selC1.value = String(s.chan[1]);
    invertBox.checked = !!s.invert;
    sDiam.set(s === startSample && Number(host.dataset.diameter) > 0 ? Number(host.dataset.diameter) : s.diameter);
    setModel(s.model);
  }

  // ---- scheduling (latest request wins; one run at a time)
  let netTimer = 0, decTimer = 0, finTimer = 0;
  function scheduleNet(delay = 150): void {
    syncIter();
    if (!started) { draw(); return; }
    clearTimeout(netTimer); netTimer = window.setTimeout(() => { void runNet(); }, delay);
  }
  function scheduleDecode(delay = 120): void {
    if (!net) { draw(); return; }
    clearTimeout(decTimer); decTimer = window.setTimeout(() => void runDecode(), delay);
  }
  function scheduleFinish(delay = 60): void {
    if (!dec) { draw(); return; }
    clearTimeout(finTimer); finTimer = window.setTimeout(() => void runFinish(), delay);
  }
  function showBusy(msg: string | null): void { busy.textContent = msg ?? ""; busy.classList.toggle("cp-hide", !msg); }
  function fail(e: unknown): void { showBusy(null); err.textContent = String((e as Error)?.message ?? e); }

  async function ensurePool(): Promise<Pool> {
    if (!pool) {
      if (isFileProtocol()) throw new Error(fileProtocolMessage("Cellpose"));
      const url = SCRIPT_URL || distBase() + NAME + ".js";
      pool = new Pool(url, nWorkers);
    }
    if (poolModel !== model) {
      const info = P.MODELS[model];
      const st = loadStatus(status, `Loading the ${info.label} network`);
      try {
        const t0 = performance.now();
        const bytes = await loadModelFile(info.file, { onProgress: st.progress });
        const t1 = performance.now();
        st.text(`Starting ${pool.size} worker${pool.size > 1 ? "s" : ""} (${formatBytes(bytes.length)} of weights each)…`);
        await pool.broadcast({ type: "init", weights: bytes });
        loadInfo = `Network ${info.label}: ${formatBytes(bytes.length)} loaded in ${((t1 - t0) / 1000).toFixed(1)} s, ` +
          `${pool.size} worker${pool.size > 1 ? "s" : ""} ready after ${((performance.now() - t1) / 1000).toFixed(1)} s`;
        if (info.sizeFile && !sizeModel) sizeModel = JSON.parse(new TextDecoder().decode(await loadModelFile(info.sizeFile)));
        st.done();
      } catch (e) { st.error(String((e as Error).message ?? e)); throw e; }
      poolModel = model;
      runBtn.textContent = "Run again";
    }
    return pool;
  }

  function inputPlanes(): [Float32Array, Float32Array] | null {
    if (!loaded) return null;
    const ch = P.toChannels(loaded.img, [Number(selC0.value), Number(selC1.value)]);
    return [P.normalize99(ch[0], invertBox.checked), P.normalize99(ch[1], invertBox.checked)];
  }

  // One network run at a time; a request during a run re-runs once it ends (latest settings).
  let netRunning = false, netPending = false, decGen = 0, finGen = 0;
  async function runNet(): Promise<void> {
    if (netRunning) { netPending = true; return; }
    netRunning = true;
    err.textContent = "";
    try {
      do {
        netPending = false;
        if (!loaded) break;
        const p = await ensurePool();
        const inp = inputPlanes()!;
        const { w, h } = loaded.img;
        const diam = sDiam.get();
        const t0 = performance.now();
        const ms: number[] = [];
        const out = await P.runNetwork(inp, w, h, P.MODELS[model].diamMean / diam, async (x, th, tw) => {
          const r = await p.call<{ y: Float32Array; style: Float32Array; ms: number }>({ type: "tile", input: x, h: th, w: tw }, [x.buffer]);
          ms.push(r.ms);
          return r;
        }, (done, total) => showBusy(`Network: tile ${done} of ${total} (${p.size} worker${p.size > 1 ? "s" : ""})…`));
        if (netPending || loaded.img.w !== w || loaded.img.h !== h) continue;
        netMs = performance.now() - t0;
        tileMs = ms;
        net = out;
        netKey = `${model}|${diam}`;
        await runDecode();
      } while (netPending);
    } catch (e) { fail(e); }
    netRunning = false;
    if (afterNet) { const f = afterNet; afterNet = null; f(); }
  }
  let afterNet: (() => void) | null = null;

  async function runDecode(): Promise<void> {
    if (!net || !pool) return;
    const my = ++decGen;
    try {
      showBusy("Following the flows…");
      const r = await pool.call<MasksOut>({
        type: "masks", dy: net.dy, dx: net.dx, cellprob: net.cellprob, h: net.h, w: net.w,
        niter: Math.round(sIter.get()), cellprobThreshold: sProb.get(), flowThreshold: sFlow.get(), minSize: Math.round(sMin.get()),
      });
      if (my !== decGen) return;
      dec = r; finalMasks = r.masks; paths = null; clickedPaths = null;
      showBusy(null);
      if (view === "track") await ensurePaths(); else draw();
    } catch (e) { if (my === decGen) fail(e); }
  }

  async function runFinish(): Promise<void> {
    if (!dec || !pool || !net) return;
    const my = ++finGen;
    try {
      const r = await pool.call<{ masks: Uint32Array; ms: number }>({
        type: "finish", masksRaw: dec.masksRaw, errors: dec.errors, h: net.h, w: net.w, flowThreshold: sFlow.get(), minSize: Math.round(sMin.get()),
      });
      if (my !== finGen) return;
      finalMasks = r.masks;
      draw();
    } catch (e) { if (my === finGen) fail(e); }
  }

  async function ensurePaths(): Promise<void> {
    draw();
    if (!net || !dec || !pool || paths) return;
    const { w, h } = net;
    const step = Math.max(3, Math.round(sDiam.get() / 4));
    const starts: number[] = [];
    const thr = sProb.get();
    for (let y = Math.floor(step / 2); y < h; y += step) for (let x = Math.floor(step / 2); x < w; x += step) {
      const i = y * w + x;
      if (net.cellprob[i] > thr) starts.push(i);
    }
    const keep = Math.ceil(starts.length / 3000);
    if (keep > 1) starts.splice(0, starts.length, ...starts.filter((_, k) => k % keep === 0));
    const niter = Math.round(sIter.get());
    const stride = Math.max(1, Math.ceil(niter / 40));
    const per = Math.floor(niter / stride) + 1;
    const s = Uint32Array.from(starts);
    const r = await pool.call<{ paths: Float32Array }>({ type: "paths", dy: net.dy, dx: net.dx, cellprob: net.cellprob, h, w, cellprobThreshold: thr, starts: s, niter, stride });
    paths = { starts: s, xy: r.paths, per };
    await tracePicked();
    draw();
  }

  async function tracePicked(): Promise<void> {
    if (!net || !pool || !clicked.length) { clickedPaths = null; return; }
    const r = await pool.call<{ paths: Float32Array }>({ type: "paths", dy: net.dy, dx: net.dx, cellprob: net.cellprob, h: net.h, w: net.w,
      cellprobThreshold: sProb.get(), starts: Uint32Array.from(clicked), niter: Math.round(sIter.get()), stride: 1 });
    clickedPaths = r.paths;
  }

  async function estimateDiameter(): Promise<void> {
    if (!loaded || !P.MODELS[model].sizeFile) return;
    started = true;
    const dm = P.MODELS[model].diamMean;
    const finish = () => {
      if (!net || netKey !== `${model}|${dm}` || !sizeModel) return;
      const d = P.diameterFromStyle(net.style, sizeModel);
      const nd = Math.round(Math.min(100, Math.max(5, d)));
      status.textContent = `Size model (from the style vector at diameter ${dm}): about ${d.toFixed(1)} px.`;
      if (nd === sDiam.get()) return;
      sDiam.set(nd);
      syncIter();
      scheduleNet(0);
    };
    // the size model reads the style vector at the training scale (diameter = diam_mean)
    if (net && netKey === `${model}|${dm}`) { finish(); return; }
    sDiam.set(dm);
    syncIter();
    afterNet = finish;
    void runNet();
  }

  // ---- drawing
  const base = document.createElement("canvas");
  function draw(): void {
    if (!loaded) return;
    const { w, h } = loaded.img;
    const k = Math.max(1, Math.min(4, Math.floor(560 / Math.max(w, h))));
    if (canvas.width !== w * k || canvas.height !== h * k) { canvas.width = w * k; canvas.height = h * k; }
    base.width = w; base.height = h;
    const bctx = base.getContext("2d")!;
    const id = bctx.createImageData(w, h);
    const px = id.data;
    const n = w * h;
    const showNet = net && net.w === w && net.h === h;
    const dim = (f: number) => { for (let i = 0; i < n; i++) { px[4 * i] = loaded!.rgba[4 * i] * f; px[4 * i + 1] = loaded!.rgba[4 * i + 1] * f; px[4 * i + 2] = loaded!.rgba[4 * i + 2] * f; px[4 * i + 3] = 255; } };
    const blend = (i: number, c: [number, number, number], a: number) => {
      px[4 * i] = px[4 * i] * (1 - a) + c[0] * a; px[4 * i + 1] = px[4 * i + 1] * (1 - a) + c[1] * a; px[4 * i + 2] = px[4 * i + 2] * (1 - a) + c[2] * a;
    };
    if (view === "input" || !showNet) {
      const inp = inputPlanes()!;
      const twoCh = Number(selC1.value) > 0;
      for (let i = 0; i < n; i++) {
        const a = Math.max(0, Math.min(255, inp[0][i] * 255)), b = Math.max(0, Math.min(255, inp[1][i] * 255));
        if (twoCh) { px[4 * i] = 0; px[4 * i + 1] = a; px[4 * i + 2] = b; } else { px[4 * i] = px[4 * i + 1] = px[4 * i + 2] = a; }
        px[4 * i + 3] = 255;
      }
    } else if (view === "flows") {
      P.flowToRGBA(net!.dy, net!.dx, px);
    } else if (view === "prob") {
      scalarToRGBA(net!.cellprob, { map: viridis, lo: -8, hi: 8 }, px);
      const thr = sProb.get(), cp = net!.cellprob;
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = y * w + x, a = cp[i] > thr;
        if (a && ((x > 0 && !(cp[i - 1] > thr)) || (x < w - 1 && !(cp[i + 1] > thr)) || (y > 0 && !(cp[i - w] > thr)) || (y < h - 1 && !(cp[i + w] > thr)))) {
          px[4 * i] = px[4 * i + 1] = px[4 * i + 2] = 255;
        }
      }
    } else if (view === "track") {
      dim(0.35);
    } else if (view === "masks" || view === "check") {
      dim(view === "masks" ? 0.85 : 0.5);
      const lab = view === "masks" ? finalMasks : dec?.masksRaw;
      if (lab) {
        const ol = P.outlines(lab, w, h);
        const kept = new Uint8Array((dec?.errors.length ?? 0) + 1);
        if (view === "check" && finalMasks) for (let i = 0; i < n; i++) if (finalMasks[i] && lab[i]) kept[lab[i]] = 1;
        for (let i = 0; i < n; i++) {
          const L = lab[i];
          if (!L) continue;
          if (view === "masks") { blend(i, labRGB(L), ol[i] ? 1 : 0.45); continue; }
          const removedFlow = sFlow.get() > 0 && dec!.errors[L - 1] > sFlow.get();
          const c: [number, number, number] = kept[L] ? [60, 200, 90] : removedFlow ? [235, 50, 50] : [255, 150, 30];
          blend(i, c, ol[i] ? 1 : kept[L] ? 0.15 : 0.55);
        }
        if (view === "masks" && loaded.labels && sample && showGT.checked) {
          const gol = P.outlines(loaded.labels, w, h);
          for (let i = 0; i < n; i++) if (gol[i]) blend(i, [255, 255, 0], 1);
        }
      }
    }
    bctx.putImageData(id, 0, 0);
    const ctx = canvas.getContext("2d")!;
    ctx.imageSmoothingEnabled = k === 1;
    ctx.drawImage(base, 0, 0, w * k, h * k);
    const lw = Math.max(1, Math.max(w, h) * k / 500);
    if (showNet && view === "flows") {
      const s = Math.max(5, Math.round(sDiam.get() / 3)), thr = sProb.get();
      ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.lineWidth = lw;
      ctx.beginPath();
      for (let y = s / 2; y < h; y += s) for (let x = s / 2; x < w; x += s) {
        const i = Math.floor(y) * w + Math.floor(x);
        if (!(net!.cellprob[i] > thr)) continue;
        const vy = net!.dy[i], vx = net!.dx[i], m = Math.hypot(vy, vx) || 1, L = 0.45 * s * Math.min(1, m / 5);
        const x0 = (x + 0.5) * k, y0 = (y + 0.5) * k, x1 = x0 + (vx / m) * L * k, y1 = y0 + (vy / m) * L * k;
        ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
        const a = Math.atan2(y1 - y0, x1 - x0), hl = Math.min(4 * lw, L * k * 0.4);
        ctx.moveTo(x1, y1); ctx.lineTo(x1 - hl * Math.cos(a - 0.5), y1 - hl * Math.sin(a - 0.5));
        ctx.moveTo(x1, y1); ctx.lineTo(x1 - hl * Math.cos(a + 0.5), y1 - hl * Math.sin(a + 0.5));
      }
      ctx.stroke();
    }
    if (showNet && view === "track" && dec) {
      if (paths) {
        ctx.lineWidth = lw;
        for (let j = 0; j < paths.starts.length; j++) {
          const L = dec.masksRaw[paths.starts[j]];
          const c = L ? labRGB(L) : [200, 200, 200];
          ctx.strokeStyle = `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},0.8)`;
          ctx.beginPath();
          for (let t = 0; t < paths.per; t++) {
            const o = (j * paths.per + t) * 2, X = (paths.xy[o + 1] + 0.5) * k, Y = (paths.xy[o] + 0.5) * k;
            if (t === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
          }
          ctx.stroke();
          const o = j * paths.per * 2;
          ctx.fillStyle = ctx.strokeStyle;
          ctx.fillRect((paths.xy[o + 1] + 0.5) * k - lw, (paths.xy[o] + 0.5) * k - lw, 2 * lw, 2 * lw);
        }
      }
      // sinks: where the end points pile up (area ~ number of pixels)
      ctx.fillStyle = "rgba(255,50,50,0.85)";
      ctx.beginPath();
      for (let i = 0; i < n; i++) if (dec.hist[i] > 2) {
        const r = lw * (0.6 + Math.sqrt(dec.hist[i]) / 6);
        ctx.moveTo((i % w + 0.5) * k + r, (Math.floor(i / w) + 0.5) * k);
        ctx.arc((i % w + 0.5) * k, (Math.floor(i / w) + 0.5) * k, r, 0, 2 * Math.PI);
      }
      ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = lw;
      const cs = Math.max(3, 3 * lw);
      ctx.beginPath();
      for (const s of dec.seeds) {
        const X = (s % w + 0.5) * k, Y = (Math.floor(s / w) + 0.5) * k;
        ctx.moveTo(X - cs, Y); ctx.lineTo(X + cs, Y); ctx.moveTo(X, Y - cs); ctx.lineTo(X, Y + cs);
      }
      ctx.stroke();
      if (clickedPaths) {
        const per = Math.round(sIter.get()) + 1;
        for (let j = 0; j < clicked.length; j++) {
          ctx.beginPath();
          for (let t = 0; t < per; t++) {
            const o = (j * per + t) * 2, X = (clickedPaths[o + 1] + 0.5) * k, Y = (clickedPaths[o] + 0.5) * k;
            if (t === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
          }
          ctx.strokeStyle = "#000"; ctx.lineWidth = 4 * lw; ctx.stroke();
          ctx.strokeStyle = "#ffeb3b"; ctx.lineWidth = 2 * lw; ctx.stroke();
          const o = j * per * 2;
          ctx.fillStyle = "#ffeb3b";
          ctx.beginPath(); ctx.arc((clickedPaths[o + 1] + 0.5) * k, (clickedPaths[o] + 0.5) * k, 3 * lw, 0, 2 * Math.PI); ctx.fill();
        }
      }
    }
    if (showNet && view === "check" && dec && dec.errors.length && dec.errors.length <= 150) {
      const cy = new Float64Array(dec.errors.length + 1), cx = new Float64Array(dec.errors.length + 1), cn = new Float64Array(dec.errors.length + 1);
      for (let i = 0; i < n; i++) { const L = dec.masksRaw[i]; if (L) { cy[L] += Math.floor(i / w); cx[L] += i % w; cn[L]++; } }
      const fs = Math.max(9, Math.round(Math.max(w, h) * k / 65));
      ctx.font = `600 ${fs}px system-ui, sans-serif`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.lineWidth = 3; ctx.strokeStyle = "#000"; ctx.fillStyle = "#fff";
      for (let L = 1; L <= dec.errors.length; L++) if (cn[L]) {
        const t = dec.errors[L - 1].toFixed(2), X = (cx[L] / cn[L] + 0.5) * k, Y = (cy[L] / cn[L] + 0.5) * k;
        ctx.strokeText(t, X, Y); ctx.fillText(t, X, Y);
      }
    }
    updateReadout();
  }

  function updateReadout(): void {
    if (!loaded) { readout.textContent = ""; return; }
    const { w, h } = loaded.img;
    const lines: string[] = [`Image ${w} × ${h}${loaded.note ? " (" + loaded.note + ")" : ""}`];
    if (!started) {
      lines.push(isFileProtocol() ? fileProtocolMessage("Cellpose") : `Press "Run Cellpose": the ${P.MODELS[model].label} network (about 13 MB) is downloaded once, then runs in this browser.`);
    }
    if (loadInfo) lines.push(loadInfo);
    if (net) {
      const mean = tileMs.length ? tileMs.reduce((a, b) => a + b, 0) / tileMs.length : 0;
      lines.push(`Network on ${net.workW} × ${net.workH} (rescaled ×${net.rescale.toFixed(2)}): ${net.nTiles} tile${net.nTiles > 1 ? "s" : ""} of ${net.tileH} × ${net.tileW}`);
      lines.push(`  ${(netMs / 1000).toFixed(1)} s on ${pool?.size ?? 1} worker${(pool?.size ?? 1) > 1 ? "s" : ""} (${(mean / 1000).toFixed(2)} s per tile)`);
    }
    if (dec && finalMasks) {
      let count = 0;
      for (const v of finalMasks) if (v > count) count = v;
      const raw = dec.errors.length;
      const removedFlow = sFlow.get() > 0 ? dec.errors.filter((e) => e > sFlow.get()).length : 0;
      lines.push(`Decoding: ${dec.ms.toFixed(0)} ms; ${dec.nFg} pixels tracked for ${Math.round(sIter.get())} steps`);
      lines.push(`Sinks with seeds: ${dec.seeds.length}; masks: ${raw} → flow check removes ${removedFlow} → ${count} cells`);
      lines.push(`Median diameter of the masks: ${P.medianDiameter(finalMasks).toFixed(1)} px`);
      if (loaded.labels) {
        const T = iouTable(loaded.labels, finalMasks);
        const sc = objectScores(match(T, 0.5));
        lines.push(`vs ground truth (${T.nt} objects), IoU ≥ 0.5: F1 ${sc.f1.toFixed(3)} (TP ${sc.tp}, FP ${sc.fp}, FN ${sc.fn})`);
      }
    }
    readout.textContent = lines.join("\n");
  }

  const showGT = checkbox(panel, "show ground-truth outlines (yellow) on Masks", false, () => draw());
  panel.insertBefore(showGT.parentElement!, readout);
  showGT.parentElement!.classList.add("cp-hide");

  canvas.addEventListener("pointerdown", (ev) => {
    if (view !== "track" || !net || !loaded) return;
    const r = canvas.getBoundingClientRect();
    const x = Math.floor(((ev.clientX - r.left) / r.width) * loaded.img.w), y = Math.floor(((ev.clientY - r.top) / r.height) * loaded.img.h);
    if (x < 0 || y < 0 || x >= loaded.img.w || y >= loaded.img.h) return;
    clicked.push(y * loaded.img.w + x);
    if (clicked.length > 8) clicked.shift();
    void tracePicked().then(draw);
  });

  // ---- image loading
  async function setLoaded(l: Loaded): Promise<void> {
    loaded = l;
    net = null; dec = null; finalMasks = null; paths = null; clicked.length = 0; clickedPaths = null;
    credit.textContent = l.credit ? "Image: " + l.credit : "";
    selC0.disabled = selC1.disabled = !l.isRGB;
    if (!l.isRGB) { selC0.value = "0"; selC1.value = "0"; }
    showGT.parentElement!.classList.toggle("cp-hide", !l.labels);
    draw();
    scheduleNet(0);
  }
  async function chooseSample(s: Sample): Promise<void> {
    sample = s;
    applySample(s);
    const st = loadStatus(status, "Loading image");
    try { const l = await loadSample(s); st.done(); await setLoaded(l); } catch (e) { st.error(String((e as Error).message ?? e)); }
  }
  sel.addEventListener("change", () => {
    sel.blur();
    if (sel.value === "own") { fileIn.click(); return; }
    void chooseSample(SAMPLES.find((s) => s.key === sel.value)!);
  });
  fileIn.addEventListener("change", async () => {
    const f = fileIn.files?.[0];
    if (!f) return;
    sel.value = "own"; sample = null;
    try { await setLoaded(await loadOwn(f)); } catch (e) { fail(e); }
  });

  setView(view);
  void chooseSample(startSample);
}

if (typeof document === "undefined") workerMain();
else autoMount(NAME, mount);
