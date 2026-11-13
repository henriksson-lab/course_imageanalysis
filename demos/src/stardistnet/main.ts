// StarDist 2D with the real network (2D_versatile_fluo) running in the browser: Rust ->
// WebAssembly (demos/wasm/stardist/, in a Web Worker). Networks for cells and objects day,
// unit stardist/.
// What it shows, step by step (the "Show" buttons): the input; the object probability per grid
// cell (one cell = 2 x 2 pixels); the predicted length of one of the 32 rays per cell (pick the
// direction); every cell above the probability threshold drawn as its star-convex polygon (the
// NMS candidates); the polygons that survive polygon non-maximum suppression; the label image.
// Click a pixel to see the 32 rays the network predicts there. Thresholds and the NMS overlap
// re-run only the decoding (fast); the input percentiles and the image scale re-run the network.
// Built-in sample micrographs (demos/dist/models/images/, credit shown) or the viewer's own image
// (read locally, never uploaded). Module and weights are fetched same-origin with a progress bar
// and cached (lib/wasm.ts); opened as a local file the demo explains why it cannot run.
// Auto-mounts on <div data-demo="stardistnet">. Options: data-image="nuclei|sparse|dim|dense|
// u2os|histology|brightfield", data-view="image|prob|ray|cand|nms|labels",
// data-preset="trained|low|high|weak|strong", data-engine="fast|candle" (candle = the
// stardist-rs network as published, about 3x slower).
import { htmlEl, button, slider, presets, autoMount, type Slider } from "../lib/ui";
import { distBase, isFileProtocol, fileProtocolMessage, loadStatus, loadModelFile, modelIndex, formatBytes, type LoadStatus } from "../lib/wasm";
import { scalarToImageData, inferno, viridis, labelColor, toCss, type RGB } from "../lib/colour";
import type { Req, Res } from "./protocol";
import { runWorker } from "./worker";

const NAME = "stardistnet";
const MAX_SIDE = 1024; // network input at most this many pixels per side (time grows with area)

const SAMPLES: { id: string; key: string; label: string }[] = [
  { id: "nuclei", key: "images/stardist-img2d.png", label: "Nuclei, fluorescence (512²)" },
  { id: "sparse", key: "images/dsb-fluo-small.png", label: "Nuclei, few (256²)" },
  { id: "dim", key: "images/dsb-fluo-medium.png", label: "Nuclei, dim (360²)" },
  { id: "dense", key: "images/dsb-fluo-dense.png", label: "Nuclei, dense (696 × 520)" },
  { id: "u2os", key: "images/bbbc039-A02_s1.png", label: "U2OS nuclei, Hoechst (696 × 520)" },
  { id: "histology", key: "images/dsb-histology.png", label: "Histology, H&E (not fluorescence)" },
  { id: "brightfield", key: "images/dsb-brightfield.png", label: "Bright field (not fluorescence)" },
];

type View = "image" | "prob" | "ray" | "cand" | "nms" | "labels";
const VIEWS: { id: View; label: string }[] = [
  { id: "image", label: "Input" },
  { id: "prob", label: "Probability" },
  { id: "ray", label: "Ray length" },
  { id: "cand", label: "Candidates" },
  { id: "nms", label: "After NMS" },
  { id: "labels", label: "Labels" },
];

const PRESETS: { id: string; label: string; prob?: number; nms?: number }[] = [
  { id: "trained", label: "Trained thresholds" },
  { id: "low", label: "Low probability threshold", prob: 0.15 },
  { id: "high", label: "High probability threshold", prob: 0.85 },
  { id: "weak", label: "Weak suppression", nms: 0.9 },
  { id: "strong", label: "Strong suppression", nms: 0.02 },
];

const C_RAY = "#ffd400", C_POLY = "#ffffff";

const STYLE = `
.qb-demo[data-demo="${NAME}"] { display: flex; flex-wrap: wrap; gap: .8em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .sd-left { flex: 1 1 440px; min-width: 260px; max-width: 540px; display: flex; flex-direction: column; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .sd-stage { position: relative; width: 100%; aspect-ratio: 1; background: #000; border-radius: 4px; overflow: hidden; }
.qb-demo[data-demo="${NAME}"] .sd-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.qb-demo[data-demo="${NAME}"] .sd-stage canvas.sd-px { image-rendering: pixelated; }
.qb-demo[data-demo="${NAME}"] .sd-stage canvas.sd-over { cursor: crosshair; touch-action: none; }
.qb-demo[data-demo="${NAME}"] .sd-busy { position: absolute; left: 0; right: 0; bottom: 0; padding: .3em .6em; font-size: 16px; color: #fff; background: rgba(0,0,0,.65); pointer-events: none; }
.qb-demo[data-demo="${NAME}"] .sd-credit { font-size: 13px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .sd-legend { display: flex; align-items: center; gap: .4em; font-size: 13px; color: var(--qb-muted); min-height: 1.3em; }
.qb-demo[data-demo="${NAME}"] .sd-bar { flex: 0 0 140px; height: 10px; border-radius: 2px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 360px; max-width: 540px; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .sd-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
.qb-demo[data-demo="${NAME}"] .sd-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] select { font: inherit; font-size: .9em; padding: .1em .2em; max-width: 16em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre-wrap; line-height: 1.35; }
.qb-demo[data-demo="${NAME}"] .qb-btn:disabled { opacity: .45; cursor: default; }
@media (max-width: 640px) { .qb-demo[data-demo="${NAME}"] .sd-left { flex-basis: auto; width: 100%; max-width: none; } }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; gap: .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .sd-left { max-width: 470px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

/** Linear-interpolated percentile of a copy (as NumPy's default). */
function percentile(sorted: Float32Array, p: number): number {
  const pos = Math.min(1, Math.max(0, p / 100)) * (sorted.length - 1), i = Math.floor(pos), j = Math.min(sorted.length - 1, i + 1);
  return sorted[i] + (sorted[j] - sorted[i]) * (pos - i);
}

function gradientCss(map: (t: number) => RGB): string {
  const stops: string[] = [];
  for (let i = 0; i <= 8; i++) stops.push(toCss(map(i / 8)));
  return `linear-gradient(to right, ${stops.join(",")})`;
}

interface Img { id: string; w: number; h: number; grey: Float32Array; sorted: Float32Array; note: string; credit: string }
interface Pred { id: number; ow: number; oh: number; prob: Float32Array; dist: Float32Array; ms: number }
interface Dec { ms: number; candidates: number; count: number; points: Float32Array; probs: Float32Array; coords: Float32Array; labels: Uint32Array }

function mount(host: HTMLElement): void {
  injectStyle();
  const engine = host.dataset.engine === "candle" ? 0 : 1;

  // ---- Layout -----------------------------------------------------------------------------------
  const left = htmlEl("div", "sd-left", "", host);
  const stage = htmlEl("div", "sd-stage", "", left);
  const base = htmlEl("canvas", "", "", stage);
  const over = htmlEl("canvas", "sd-over", "", stage);
  const busy = htmlEl("div", "sd-busy", "", stage);
  const legend = htmlEl("div", "sd-legend", "", left);
  const credit = htmlEl("div", "sd-credit", "", left);
  const statusHost = htmlEl("div", "", "", left);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "sd-lab", "Image", panel);
  const imgRow = htmlEl("div", "sd-row", "", panel);
  const sel = htmlEl("select", "", "", imgRow);
  for (const s of SAMPLES) { const o = htmlEl("option", "", s.label, sel); o.value = s.id; }
  const oOwn = htmlEl("option", "", "Your image", sel); oOwn.value = "own"; oOwn.disabled = true;
  const fileIn = htmlEl("input", "", "", imgRow);
  fileIn.type = "file"; fileIn.accept = "image/*"; fileIn.style.display = "none";
  button(imgRow, "Use your own image…", () => fileIn.click());
  const scaleRow = htmlEl("div", "sd-row", "", panel);
  htmlEl("span", "sd-lab", "Scale image by", scaleRow).style.marginBottom = "0";
  const scaleSel = htmlEl("select", "", "", scaleRow);
  for (const f of [0.5, 0.75, 1, 1.5, 2]) { const o = htmlEl("option", "", `${f}×`, scaleSel); o.value = String(f); }
  scaleSel.value = "1";

  htmlEl("div", "sd-lab", "Show", panel);
  const viewRow = htmlEl("div", "sd-row", "", panel);
  const viewBtns = VIEWS.map((v) => button(viewRow, v.label, () => setView(v.id)));

  htmlEl("div", "sd-lab", "Decoding (re-runs only the post-processing)", panel);
  const sProb: Slider = slider(panel, "Probability threshold", 0.05, 0.95, 0.01, 0.48, (v) => { S.probThr = v; clearPreset(); thresholdsChanged(); });
  const sNms: Slider = slider(panel, "NMS overlap threshold", 0.01, 0.95, 0.01, 0.3, () => { clearPreset(); thresholdsChanged(); });
  const presetBtns = presets(panel, PRESETS.map((p) => ({ label: p.label, apply: () => applyPreset(p.id) })));
  const sRay: Slider = slider(panel, "Ray direction", 0, 31, 1, 0, () => { if (S.view === "ray") draw(); else setView("ray"); },
    (v) => `${Math.round((v * 360) / 32)}°`);
  htmlEl("div", "sd-lab", "Input normalisation (re-runs the network)", panel);
  const sLo: Slider = slider(panel, "Dark percentile", 0, 20, 0.5, 1, () => schedulePredict());
  const sHi: Slider = slider(panel, "Bright percentile", 90, 100, 0.1, 99.8, () => schedulePredict());
  const readout = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Click the image: the 32 rays the network predicts at that pixel.", panel);

  // ---- State ------------------------------------------------------------------------------------
  const S = {
    view: (VIEWS.some((v) => v.id === host.dataset.view) ? host.dataset.view : "nms") as View,
    img: null as Img | null,
    scaled: null as Img | null,
    ready: false,
    nRays: 32, grid: 2,
    probDefault: 0.48, nmsDefault: 0.3,
    probThr: 0.48, // exact value (the slider snaps to 0.01; the trained threshold is 0.479…)
    pred: null as Pred | null,
    dec: null as Dec | null,
    predSeq: 0, decSeq: 0,
    decBusy: false, decWanted: false, decFor: new Map<number, number>(),
    sel: null as { gy: number; gx: number } | null,
    selAuto: true, // pick a cell after the first decoding, until the viewer clicks
    loadInfo: "",
    pending: host.dataset.preset ?? "",
  };
  let worker: Worker | null = null;
  let stMod: LoadStatus | null = null, stW: LoadStatus | null = null;
  let predictTimer = 0;
  const post = (m: Req, transfer: Transferable[] = []) => worker?.postMessage(m, transfer);

  function setBusy(msg: string): void { busy.textContent = msg; busy.style.display = msg ? "" : "none"; }
  function clearPreset(): void { presetBtns.forEach((b) => b.classList.remove("active")); }

  function setView(v: View): void {
    S.view = v;
    viewBtns.forEach((b, i) => b.classList.toggle("active", VIEWS[i].id === v));
    draw();
  }

  function applyPreset(id: string): void {
    const p = PRESETS.find((x) => x.id === id);
    if (!p) return;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i].id === id));
    S.probThr = p.prob ?? S.probDefault; sProb.set(S.probThr);
    sNms.set(p.nms ?? S.nmsDefault);
    if (p.prob !== undefined) setView("cand");
    else if (p.nms !== undefined) setView("nms");
    thresholdsChanged();
  }

  // ---- Images -----------------------------------------------------------------------------------
  function greyFromBitmap(bmp: ImageBitmap, id: string, credit: string): Img {
    let w = bmp.width, h = bmp.height, note = "";
    const f = Math.min(1, MAX_SIDE / Math.max(w, h));
    if (f < 1) { note = ` (scaled from ${w} × ${h})`; w = Math.round(w * f); h = Math.round(h * f); }
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const g = c.getContext("2d", { willReadFrequently: true })!;
    g.drawImage(bmp, 0, 0, w, h);
    const d = g.getImageData(0, 0, w, h).data;
    const grey = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) grey[i] = (d[4 * i] + d[4 * i + 1] + d[4 * i + 2]) / 3;
    return { id, w, h, grey, sorted: grey.slice().sort(), note, credit };
  }

  /** The network input: the image scaled by the chosen factor (bilinear via canvas), capped. */
  function scaledImage(img: Img): Img {
    let f = Number(scaleSel.value) || 1;
    f = Math.min(f, MAX_SIDE / Math.max(img.w, img.h));
    if (Math.abs(f - 1) < 1e-6) return img;
    const w = Math.max(16, Math.round(img.w * f)), h = Math.max(16, Math.round(img.h * f));
    const src = document.createElement("canvas");
    src.width = img.w; src.height = img.h;
    const id = new ImageData(img.w, img.h);
    for (let i = 0; i < img.w * img.h; i++) { const v = Math.max(0, Math.min(255, img.grey[i])); id.data[4 * i] = id.data[4 * i + 1] = id.data[4 * i + 2] = v; id.data[4 * i + 3] = 255; }
    src.getContext("2d")!.putImageData(id, 0, 0);
    const dst = document.createElement("canvas");
    dst.width = w; dst.height = h;
    const g = dst.getContext("2d", { willReadFrequently: true })!;
    g.imageSmoothingQuality = "high";
    g.drawImage(src, 0, 0, w, h);
    const d = g.getImageData(0, 0, w, h).data;
    const grey = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) grey[i] = d[4 * i];
    return { ...img, w, h, grey, sorted: grey.slice().sort(), note: img.note + `, scaled ${f}× to ${w} × ${h}` };
  }

  async function setImage(img: Img): Promise<void> {
    S.img = img;
    S.pred = null; S.dec = null; S.sel = null;
    credit.textContent = img.credit;
    predictNow();
  }

  async function loadSample(id: string): Promise<void> {
    const s = SAMPLES.find((x) => x.id === id) ?? SAMPLES[0];
    sel.value = s.id;
    try {
      const [bytes, index] = await Promise.all([loadModelFile(s.key), modelIndex()]);
      const bmp = await createImageBitmap(new Blob([bytes as BlobPart], { type: "image/png" }));
      await setImage(greyFromBitmap(bmp, s.id, `Sample: ${String(index[s.key]?.credit ?? s.label)}`));
    } catch (e) {
      loadStatus(statusHost, "").error(`Could not load the sample image: ${(e as Error).message}`);
    }
  }

  sel.addEventListener("change", () => { if (sel.value !== "own") void loadSample(sel.value); sel.blur(); });
  scaleSel.addEventListener("change", () => { scaleSel.blur(); predictNow(); });
  fileIn.addEventListener("change", async () => {
    const f = fileIn.files?.[0];
    if (!f) return;
    try {
      const bmp = await createImageBitmap(f);
      oOwn.disabled = false; sel.value = "own";
      await setImage(greyFromBitmap(bmp, "own", `Your image: ${f.name} (read locally, not uploaded; colour averaged to grey)`));
    } catch {
      loadStatus(statusHost, "").error("This file could not be read as an image.");
    }
    fileIn.value = "";
  });

  // ---- Running ------------------------------------------------------------------------------------
  function schedulePredict(): void {
    clearTimeout(predictTimer);
    predictTimer = window.setTimeout(predictNow, 350);
  }

  function predictNow(): void {
    if (!S.img) return;
    S.scaled = scaledImage(S.img);
    resizeStage();
    if (!S.ready) { draw(); setBusy("Loading the network …"); return; }
    const im = S.scaled;
    const grey = im.grey.slice();
    const id = ++S.predSeq;
    setBusy(`Running the network on ${im.w} × ${im.h} pixels …`);
    post({ type: "predict", id, grey: grey.buffer, w: im.w, h: im.h, pmin: sLo.get(), pmax: sHi.get() }, [grey.buffer]);
    updateReadout();
  }

  function thresholdsChanged(): void {
    draw();
    requestDecode();
  }

  function requestDecode(): void {
    if (!S.pred) return;
    if (S.decBusy) { S.decWanted = true; return; }
    S.decBusy = true; S.decWanted = false;
    const id = ++S.decSeq;
    S.decFor.set(id, S.pred.id);
    post({ type: "decode", id, probThresh: S.probThr, nmsThresh: sNms.get() });
  }

  function onMessage(m: Res): void {
    switch (m.type) {
      case "progress": {
        const st = m.what === "module" ? stMod : stW;
        st?.progress(m.loaded, m.total);
        return;
      }
      case "ready": {
        stMod?.done(); stW?.done(); stMod = stW = null;
        S.ready = true;
        S.nRays = m.nRays; S.grid = m.grid;
        S.probDefault = m.probThresh; S.nmsDefault = m.nmsThresh;
        S.loadInfo = `Loaded: module ${formatBytes(m.wasmBytes)}, weights ${formatBytes(m.weightBytes)} in ${((m.fetchMs + m.buildMs) / 1000).toFixed(1)} s`;
        S.probThr = m.probThresh; sProb.set(m.probThresh); sNms.set(m.nmsThresh);
        if (S.pending) { const p = S.pending; S.pending = ""; applyPresetSilently(p); }
        predictNow();
        return;
      }
      case "predicted": {
        if (m.id !== S.predSeq) return; // a newer run is on its way
        S.pred = { id: m.id, ow: m.ow, oh: m.oh, prob: new Float32Array(m.prob), dist: new Float32Array(m.dist), ms: m.ms };
        S.dec = null;
        setBusy("");
        if (S.selAuto) S.sel = null;
        S.decBusy = false;
        requestDecode();
        draw();
        return;
      }
      case "decoded": {
        S.decBusy = false;
        const forPred = S.decFor.get(m.id);
        S.decFor.delete(m.id);
        if (S.pred && forPred === S.pred.id) {
          S.dec = { ms: m.ms, candidates: m.candidates, count: m.count, points: new Float32Array(m.points), probs: new Float32Array(m.probs),
            coords: new Float32Array(m.coords), labels: new Uint32Array(m.labels) };
          if (S.selAuto && !S.sel) S.sel = centralObject();
          draw();
        }
        if (S.decWanted) requestDecode();
        return;
      }
      case "error": {
        setBusy("");
        S.decBusy = false;
        loadStatus(statusHost, "").error("The network failed: " + m.message);
        return;
      }
    }
  }

  function applyPresetSilently(id: string): void {
    const p = PRESETS.find((x) => x.id === id);
    if (!p) return;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i].id === id));
    S.probThr = p.prob ?? S.probDefault; sProb.set(S.probThr);
    sNms.set(p.nms ?? S.nmsDefault);
  }

  /** The grid cell of the kept object nearest the image centre (a sensible first ray display). */
  function centralObject(): { gy: number; gx: number } | null {
    const p = S.pred, d = S.dec, im = S.scaled;
    if (!p || !d || !im || !d.count) return null;
    let best = Infinity, bo = 0;
    for (let o = 0; o < d.count; o++) {
      const e = Math.hypot(d.points[2 * o] - im.h / 2, d.points[2 * o + 1] - im.w / 2);
      if (e < best) { best = e; bo = o; }
    }
    return { gy: Math.round(d.points[2 * bo] / S.grid), gx: Math.round(d.points[2 * bo + 1] / S.grid) };
  }

  // ---- Drawing --------------------------------------------------------------------------------------
  function resizeStage(): void {
    const im = S.scaled;
    if (im) stage.style.aspectRatio = `${im.w} / ${im.h}`;
  }

  function greyRGBA(im: Img, dim: number): ImageData {
    const lo = percentile(im.sorted, sLo.get()), hi = percentile(im.sorted, sHi.get());
    const k = hi > lo ? 255 / (hi - lo) : 0, id = new ImageData(im.w, im.h);
    for (let i = 0; i < im.w * im.h; i++) {
      const v = Math.max(0, Math.min(255, (im.grey[i] - lo) * k)) * dim;
      id.data[4 * i] = id.data[4 * i + 1] = id.data[4 * i + 2] = v; id.data[4 * i + 3] = 255;
    }
    return id;
  }

  function putBase(id: ImageData, pixelated: boolean): void {
    base.width = id.width; base.height = id.height;
    base.getContext("2d")!.putImageData(id, 0, 0);
    base.classList.toggle("sd-px", pixelated);
  }

  function setLegend(html: string, bar?: string): void {
    legend.innerHTML = "";
    if (bar) {
      htmlEl("span", "", html.split("|")[0], legend);
      const b = htmlEl("span", "sd-bar", "", legend); b.style.background = bar;
      htmlEl("span", "", html.split("|")[1] ?? "", legend);
      if (html.split("|")[2]) htmlEl("span", "", html.split("|")[2], legend);
    } else legend.textContent = html;
  }

  function draw(): void {
    const im = S.scaled, p = S.pred, d = S.dec;
    // Overlay canvas at the displayed resolution (crisp lines).
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cw = Math.max(1, Math.round(stage.clientWidth * dpr)), ch = Math.max(1, Math.round(stage.clientHeight * dpr));
    if (over.width !== cw) over.width = cw;
    if (over.height !== ch) over.height = ch;
    const g = over.getContext("2d")!;
    g.clearRect(0, 0, cw, ch);
    if (!im) return;
    const sx = cw / im.w, sy = ch / im.h, lw = dpr;
    const v = S.view;
    const thr = S.probThr;

    if (v === "prob" && p) {
      putBase(scalarToImageData(p.prob, p.ow, p.oh, { map: inferno, lo: 0, hi: 1 }), true);
      setLegend(`probability 0|1|  (threshold ${thr.toFixed(2)}: cells inside the green outline)`, gradientCss(inferno));
      // Outline of the cells above the threshold.
      g.strokeStyle = "#2ca02c"; g.lineWidth = 1.5 * lw;
      g.beginPath();
      const cx = cw / p.ow, cy = ch / p.oh;
      for (let y = 0; y < p.oh; y++) for (let x = 0; x < p.ow; x++) {
        const a = p.prob[y * p.ow + x] > thr;
        if (x + 1 < p.ow && a !== (p.prob[y * p.ow + x + 1] > thr)) { g.moveTo((x + 1) * cx, y * cy); g.lineTo((x + 1) * cx, (y + 1) * cy); }
        if (y + 1 < p.oh && a !== (p.prob[(y + 1) * p.ow + x] > thr)) { g.moveTo(x * cx, (y + 1) * cy); g.lineTo((x + 1) * cx, (y + 1) * cy); }
      }
      g.stroke();
    } else if (v === "ray" && p) {
      const k = sRay.get(), n = S.nRays, vals = new Float32Array(p.ow * p.oh);
      let hi = 1;
      for (let i = 0; i < vals.length; i++) { vals[i] = p.dist[i * n + k]; if (vals[i] > hi) hi = vals[i]; }
      for (let i = 0; i < vals.length; i++) if (p.prob[i] < 0.05) vals[i] = NaN; // background: no object to measure
      putBase(scalarToImageData(vals, p.ow, p.oh, { map: viridis, lo: 0, hi, nan: [0, 0, 0, 255] }), true);
      setLegend(`ray length 0|${hi.toFixed(0)} px|  (black: probability < 0.05)`, gradientCss(viridis));
      // Direction arrow in the corner.
      const phi = (2 * Math.PI * k) / n, r = 22 * dpr, ox = 34 * dpr, oy = 34 * dpr;
      g.fillStyle = "rgba(0,0,0,.6)"; g.beginPath(); g.arc(ox, oy, r + 8 * dpr, 0, 2 * Math.PI); g.fill();
      arrow(g, ox, oy, ox + r * Math.cos(phi), oy + r * Math.sin(phi), C_RAY, 3 * lw);
    } else if (v === "labels" && d) {
      const id = greyRGBA(im, 0.45);
      const lut = new Map<number, RGB>();
      for (let i = 0; i < d.labels.length; i++) {
        const l = d.labels[i];
        if (!l) continue;
        let c = lut.get(l);
        if (!c) { c = labelColor(l); lut.set(l, c); }
        id.data[4 * i] = 255 * c[0]; id.data[4 * i + 1] = 255 * c[1]; id.data[4 * i + 2] = 255 * c[2];
      }
      putBase(id, false);
      setLegend(`${d.count} labelled objects (each polygon rasterised; higher probability wins where polygons overlap)`);
    } else {
      putBase(greyRGBA(im, v === "image" ? 1 : 0.7), false);
      setLegend(v === "image" ? `input, displayed between the ${sLo.get()} and ${sHi.get()} percentiles (what the network gets)` : "");
    }

    // Candidates: every cell above the threshold as its polygon.
    if (v === "cand" && p) {
      const n = S.nRays, phis = rayAngles(n);
      const idx: number[] = [];
      for (let i = 0; i < p.prob.length; i++) if (p.prob[i] > thr) idx.push(i);
      const MAXC = 6000;
      if (idx.length > MAXC) { idx.sort((a, b) => p.prob[b] - p.prob[a]); idx.length = MAXC; }
      g.lineWidth = lw;
      for (const i of idx) {
        const gy = Math.floor(i / p.ow) * S.grid, gx = (i % p.ow) * S.grid;
        const pr = p.prob[i];
        const c = inferno(0.35 + 0.65 * pr);
        g.strokeStyle = toCss(c, 0.35);
        g.beginPath();
        for (let r = 0; r < n; r++) {
          const dd = p.dist[i * n + r], x = (gx + dd * Math.cos(phis[r])) * sx, y = (gy + dd * Math.sin(phis[r])) * sy;
          if (r === 0) g.moveTo(x, y); else g.lineTo(x, y);
        }
        g.closePath(); g.stroke();
      }
      setLegend(`${idx.length}${idx.length === MAXC ? "+" : ""} candidate polygons (cells with probability > ${thr.toFixed(2)}), colour = probability`);
    }
    if (v === "nms" && d) {
      const n = S.nRays;
      for (let o = 0; o < d.count; o++) {
        const c = labelColor(o + 1);
        g.strokeStyle = toCss(c); g.lineWidth = 2 * lw;
        g.beginPath();
        for (let r = 0; r < n; r++) {
          const y = d.coords[o * 2 * n + r] * sy, x = d.coords[o * 2 * n + n + r] * sx;
          if (r === 0) g.moveTo(x, y); else g.lineTo(x, y);
        }
        g.closePath(); g.stroke();
        g.fillStyle = toCss(c);
        g.beginPath(); g.arc(d.points[2 * o + 1] * sx, d.points[2 * o] * sy, 2.5 * lw, 0, 2 * Math.PI); g.fill();
      }
      setLegend(`${d.count} polygons kept by NMS (dot: the grid cell that predicted it)`);
    }

    // Selected cell: its 32 rays and polygon.
    if (S.sel && p) {
      const { gy, gx } = S.sel, i = gy * p.ow + gx, n = S.nRays, phis = rayAngles(n);
      const y0 = gy * S.grid, x0 = gx * S.grid;
      const pts: [number, number][] = [];
      for (let r = 0; r < n; r++) { const dd = p.dist[i * n + r]; pts.push([(x0 + dd * Math.cos(phis[r])) * sx, (y0 + dd * Math.sin(phis[r])) * sy]); }
      g.lineWidth = 1.5 * lw;
      g.strokeStyle = C_RAY;
      g.beginPath();
      for (const [x, y] of pts) { g.moveTo(x0 * sx, y0 * sy); g.lineTo(x, y); }
      g.stroke();
      if (v === "ray") {
        const [x, y] = pts[sRay.get()];
        arrow(g, x0 * sx, y0 * sy, x, y, "#ff3b30", 3 * lw);
      }
      g.strokeStyle = C_POLY; g.lineWidth = 2 * lw;
      g.beginPath(); pts.forEach(([x, y], r) => (r ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath(); g.stroke();
      g.fillStyle = C_RAY; g.beginPath(); g.arc(x0 * sx, y0 * sy, 3.5 * lw, 0, 2 * Math.PI); g.fill();
    }
    updateReadout();
  }

  function rayAngles(n: number): number[] {
    return Array.from({ length: n }, (_, k) => (2 * Math.PI * k) / n);
  }

  function arrow(g: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, col: string, w: number): void {
    const a = Math.atan2(y1 - y0, x1 - x0), hl = 4 * w;
    g.strokeStyle = col; g.fillStyle = col; g.lineWidth = w;
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1 - 0.8 * hl * Math.cos(a), y1 - 0.8 * hl * Math.sin(a)); g.stroke();
    g.beginPath(); g.moveTo(x1, y1);
    g.lineTo(x1 - hl * Math.cos(a - 0.45), y1 - hl * Math.sin(a - 0.45));
    g.lineTo(x1 - hl * Math.cos(a + 0.45), y1 - hl * Math.sin(a + 0.45)); g.closePath(); g.fill();
  }

  function updateReadout(): void {
    const im = S.scaled, p = S.pred, d = S.dec;
    const lines: string[] = [];
    if (im) lines.push(`Image ${im.w} × ${im.h}${im.note}` + (p ? ` → ${p.ow} × ${p.oh} grid cells (${S.grid} × ${S.grid} px each)` : ""));
    if (p) lines.push(`Network: ${Math.round(p.ms)} ms` + (d ? `   decoding (NMS + labels): ${Math.round(d.ms)} ms` : "   decoding …"));
    if (d) lines.push(`Cells above threshold: ${d.candidates} → after NMS: ${d.count} objects`);
    if (S.sel && p) {
      const i = S.sel.gy * p.ow + S.sel.gx, n = S.nRays;
      let lo = Infinity, hi = 0;
      for (let r = 0; r < n; r++) { const v = p.dist[i * n + r]; lo = Math.min(lo, v); hi = Math.max(hi, v); }
      lines.push(`Clicked cell (x ${S.sel.gx * S.grid}, y ${S.sel.gy * S.grid}): probability ${p.prob[i].toFixed(3)}, rays ${lo.toFixed(1)}–${hi.toFixed(1)} px` +
        (S.view === "ray" ? `, ray ${sRay.get()}: ${p.dist[i * n + sRay.get()].toFixed(1)} px` : ""));
    }
    if (S.loadInfo) lines.push(S.loadInfo);
    readout.textContent = lines.join("\n");
  }

  over.addEventListener("pointerdown", (ev) => {
    const im = S.scaled, p = S.pred;
    if (!im || !p) return;
    const r = over.getBoundingClientRect();
    const x = ((ev.clientX - r.left) / r.width) * im.w, y = ((ev.clientY - r.top) / r.height) * im.h;
    S.selAuto = false;
    S.sel = { gy: Math.max(0, Math.min(p.oh - 1, Math.round(y / S.grid))), gx: Math.max(0, Math.min(p.ow - 1, Math.round(x / S.grid))) };
    draw();
  });
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => draw()).observe(stage);

  // ---- Start ------------------------------------------------------------------------------------
  viewBtns.forEach((b, i) => b.classList.toggle("active", VIEWS[i].id === S.view));
  setBusy("");
  const want = host.dataset.image && SAMPLES.some((s) => s.id === host.dataset.image) ? host.dataset.image! : "nuclei";
  sel.value = want;
  if (isFileProtocol()) {
    loadStatus(statusHost, "Loading network");
    credit.textContent = fileProtocolMessage("This demo");
    return;
  }
  stMod = loadStatus(statusHost, "Loading the network (module)");
  stW = loadStatus(statusHost, "Loading the network (weights)");
  try {
    worker = new Worker(distBase() + NAME + ".js");
  } catch (e) {
    stMod.error("Could not start a Web Worker: " + (e as Error).message);
    return;
  }
  worker.onmessage = (e: MessageEvent<Res>) => onMessage(e.data);
  worker.onerror = (e) => { (stMod ?? loadStatus(statusHost, "")).error("Worker error: " + (e.message || "could not load")); setBusy(""); };
  post({ type: "init", base: distBase(), engine });
  void loadSample(want);
}

// The same bundle is the worker: no document there.
if (typeof document === "undefined") runWorker();
else autoMount(NAME, mount);
