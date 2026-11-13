// Object detection with a real network: YOLO11 nano (COCO-trained, 80 classes) running in the
// browser (Rust -> WebAssembly, demos/wasm/yolo/, the YOLOv11-rs model on Burn's CPU backend,
// in a Web Worker). Networks for cells and objects day, unit yolo/.
// What it shows: the image is letterboxed into a square input; the network returns, for every
// cell of three grids (stride 8, 16, 32), four distance distributions (16 bins each: how far
// the box edges are from the cell centre) and 80 class probabilities. Decoding turns each cell
// into a box; every cell/class whose probability passes the threshold is a raw candidate; greedy
// non-maximum suppression (NMS) removes candidates that overlap a better one of the same class.
// Views: the raw candidates per scale (before NMS) or the kept boxes (after NMS); the three
// score maps (best class probability per cell); for a selected box, its grid cell and the four
// distance distributions. Thresholds and NMS run live in the page; the input size re-runs the
// network. Sample photos (what the network was trained for) and micrographs (where it fails:
// no cell class exists), or the viewer's own image (read locally, never uploaded). Module and
// weights are fetched same-origin with a progress bar and cached (lib/wasm.ts); opened as a
// local file the demo explains why it cannot run.
// Auto-mounts on <div data-demo="yolonet">. Options: data-image="bus|people|nuclei|u2os|cells|
// histology|brightfield", data-preset="default|raw|nonms|strict|small|nuclei|nucleilow",
// data-size (input side, multiple of 32).
import { htmlEl, button, slider, checkbox, presets, autoMount, svgEl } from "../lib/ui";
import { distBase, isFileProtocol, fileProtocolMessage, loadStatus, loadModelFile, modelIndex, formatBytes, type LoadStatus } from "../lib/wasm";
import { scalarToRGBA, inferno } from "../lib/colour";
import * as P from "./post";
import type { Req, Res } from "./protocol";
import { runWorker } from "./worker";

const NAME = "yolonet";
const MAX_SIDE = 1280; // larger images are scaled down for display (the network sees `size` anyway)
const SCALE_COL = ["#e69f00", "#56b4e9", "#009e73"]; // stride 8, 16, 32 (Okabe-Ito)

const SAMPLES: { id: string; key: string; label: string }[] = [
  { id: "bus", key: "images/yolo-bus.jpg", label: "Photo: street, bus" },
  { id: "people", key: "images/yolo-zidane.jpg", label: "Photo: two people" },
  { id: "nuclei", key: "images/stardist-img2d.png", label: "Nuclei, fluorescence" },
  { id: "u2os", key: "images/bbbc039-A02_s1.png", label: "U2OS nuclei (Hoechst)" },
  { id: "cells", key: "images/cellpose-img02.png", label: "Cells: cytoplasm + nuclei" },
  { id: "histology", key: "images/dsb-histology.png", label: "Histology (H&E)" },
  { id: "brightfield", key: "images/dsb-brightfield.png", label: "Bright field" },
];

interface Params { conf: number; iou: number; size: number; agnostic: boolean; view: "raw" | "nms"; scales: boolean[] }
const DEFAULTS: Params = { conf: 0.25, iou: 0.45, size: 512, agnostic: false, view: "nms", scales: [true, true, true] };

const PRESETS: { id: string; label: string; image: string; p: Partial<Params> }[] = [
  { id: "default", label: "photo: defaults", image: "bus", p: {} },
  { id: "raw", label: "photo: raw candidates", image: "bus", p: { view: "raw", conf: 0.1 } },
  { id: "nonms", label: "photo: no NMS (IoU 1)", image: "bus", p: { iou: 1 } },
  { id: "strict", label: "photo: strict NMS (0.1)", image: "people", p: { iou: 0.1, agnostic: true } },
  { id: "small", label: "photo: input 256", image: "people", p: { size: 256 } },
  { id: "nuclei", label: "nuclei: defaults", image: "nuclei", p: {} },
  { id: "nucleilow", label: "nuclei: threshold 0.01", image: "nuclei", p: { conf: 0.01 } },
];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { display: flex; flex-wrap: wrap; gap: .8em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .yn-left { flex: 1 1 540px; min-width: 260px; max-width: 680px; display: flex; flex-direction: column; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .yn-top { display: flex; gap: .5em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .yn-top .yn-stage { flex: 1 1 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] .yn-top .yn-maps { flex: 0 0 120px; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .yn-top .yn-maps figure { flex: 0 0 auto; }
.qb-demo[data-demo="${NAME}"] .yn-stage { position: relative; width: 100%; background: #000; border-radius: 4px; overflow: hidden; }
.qb-demo[data-demo="${NAME}"] .yn-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.qb-demo[data-demo="${NAME}"] .yn-stage canvas.yn-ov { cursor: crosshair; touch-action: none; }
.qb-demo[data-demo="${NAME}"] .yn-busy { position: absolute; left: 0; right: 0; bottom: 0; padding: .3em .6em; font-size: 16px; color: #fff; background: rgba(0,0,0,.65); pointer-events: none; }
.qb-demo[data-demo="${NAME}"] .yn-credit { font-size: 13px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .yn-maps { display: flex; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .yn-maps figure { margin: 0; flex: 1 1 0; min-width: 0; cursor: crosshair; }
.qb-demo[data-demo="${NAME}"] .yn-maps canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; border: 3px solid transparent; background: #000; }
.qb-demo[data-demo="${NAME}"] .yn-maps figure.yn-off canvas { opacity: .35; }
.qb-demo[data-demo="${NAME}"] .yn-maps figcaption { font-size: 13px; text-align: center; line-height: 1.2; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 380px; max-width: 560px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-btn { font-size: 14px; padding: .15em .5em; }
.qb-demo[data-demo="${NAME}"] .yn-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
.qb-demo[data-demo="${NAME}"] .yn-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] select { font: inherit; font-size: .9em; padding: .1em .2em; max-width: 15em; }
.qb-demo[data-demo="${NAME}"] .yn-dfl { width: 100%; height: auto; display: block; }
.qb-demo[data-demo="${NAME}"] .yn-dfl text { font-size: 13px; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; white-space: pre-wrap; line-height: 1.35; }
.qb-demo[data-demo="${NAME}"] .qb-btn:disabled { opacity: .45; cursor: default; }
.qb-demo[data-demo="${NAME}"] .yn-hide { display: none !important; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; gap: .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .yn-left { flex: 0 1 470px; max-width: none; }
.reveal .qb-demo[data-demo="${NAME}"] .yn-top { display: contents; }
.reveal .qb-demo[data-demo="${NAME}"] .yn-top .yn-stage { flex: none; width: 100%; }
.reveal .qb-demo[data-demo="${NAME}"] .yn-top .yn-maps { flex: none; flex-direction: row; gap: .4em; order: 2; }
.reveal .qb-demo[data-demo="${NAME}"] .yn-top .yn-maps figure { flex: 1 1 0; }
.reveal .qb-demo[data-demo="${NAME}"] .yn-left > .yn-credit { order: 1; }
.reveal .qb-demo[data-demo="${NAME}"] .yn-left > .yn-credit:last-child { order: 3; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 520px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel .qb-btn { font-size: 15px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 520px; max-width: 640px; gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; padding: .25em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note, .reveal .qb-demo[data-demo="${NAME}"] .yn-lab { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface Result { size: number; lb: P.Letterbox; raw: Float32Array; dec: Float32Array; best: { cls: Uint8Array; p: Float32Array }; netMs: number; decodeMs: number }

function mount(host: HTMLElement): void {
  injectStyle();
  // ---- DOM ------------------------------------------------------------------------------------
  const left = htmlEl("div", "yn-left", "", host);
  // pages: the score maps as a column beside the image; slides: a row under it (CSS)
  const top = htmlEl("div", "yn-top", "", left);
  const stage = htmlEl("div", "yn-stage", "", top);
  const cImg = htmlEl("canvas", "", "", stage);
  const cOv = htmlEl("canvas", "yn-ov", "", stage);
  const busy = htmlEl("div", "yn-busy yn-hide", "", stage);
  const maps = htmlEl("div", "yn-maps", "", top);
  const credit = htmlEl("div", "yn-credit", "", left);
  const mapFigs = P.STRIDES.map((s, k) => {
    const f = htmlEl("figure", "", "", maps);
    const c = htmlEl("canvas", "", "", f);
    c.width = c.height = 256;
    c.style.borderColor = SCALE_COL[k];
    const fc = htmlEl("figcaption", "", `stride ${s}`, f);
    return { f, c, fc };
  });
  const mapNote = htmlEl("div", "yn-credit", "", left);

  const panel = htmlEl("div", "qb-panel", "", host);
  const statusHost = htmlEl("div", "", "", panel);
  htmlEl("div", "yn-lab", "Image", panel);
  const imgRow = htmlEl("div", "yn-row", "", panel);
  const sel = htmlEl("select", "", "", imgRow);
  for (const s of SAMPLES) { const o = htmlEl("option", "", s.label, sel); o.value = s.id; }
  const oOwn = htmlEl("option", "", "your image", sel); oOwn.value = "own"; oOwn.disabled = true;
  const fileIn = htmlEl("input", "yn-hide", "", imgRow);
  fileIn.type = "file"; fileIn.accept = "image/*";
  button(imgRow, "use your own image …", () => fileIn.click());
  htmlEl("div", "qb-note", "Your image is read in this browser and never uploaded.", panel);

  htmlEl("div", "yn-lab", "Show", panel);
  const viewRow = htmlEl("div", "yn-row", "", panel);
  const bRaw = button(viewRow, "raw candidates (before NMS)", () => { S.p.view = "raw"; refresh(); });
  const bNms = button(viewRow, "after NMS", () => { S.p.view = "nms"; refresh(); });
  const scaleRow = htmlEl("div", "yn-row", "", panel);
  const cbScale = P.STRIDES.map((s, k) => checkbox(scaleRow, `stride ${s}`, true, (v) => { S.p.scales[k] = v; refresh(); }));
  cbScale.forEach((cb, k) => { (cb.parentElement as HTMLElement).style.color = SCALE_COL[k]; (cb.parentElement as HTMLElement).style.fontWeight = "600"; });
  const sConf = slider(panel, "score threshold", 0.01, 0.95, 0.01, DEFAULTS.conf, (v) => { S.p.conf = v; refresh(); });
  const sIou = slider(panel, "NMS: remove if IoU >", 0.05, 1, 0.05, DEFAULTS.iou, (v) => { S.p.iou = v; refresh(); });
  const sSize = slider(panel, "input size (re-runs the network)", 128, 800, 32, DEFAULTS.size, (v) => { S.p.size = v; scheduleRun(); }, (v) => `${v}×${v}`);
  const cbAgn = checkbox(panel, "NMS across classes (class-agnostic)", false, (v) => { S.p.agnostic = v; refresh(); });
  htmlEl("div", "yn-lab", "Presets", panel);
  const presetBtns = presets(panel, PRESETS.map((p) => ({ label: p.label, apply: () => applyPreset(p.id) })));
  htmlEl("div", "yn-lab", "Selected box (click a box on the image, or a cell of a score map)", panel);
  const dfl = svgEl("svg", { class: "yn-dfl", viewBox: "0 0 520 120" }, panel);
  const readout = htmlEl("div", "qb-readout", "", panel);

  // ---- State ----------------------------------------------------------------------------------
  const S = {
    ready: false, running: false, pending: false,
    img: null as null | { w: number; h: number; bmp: ImageBitmap; id: string; credit: string },
    imgSeq: 0, reqSeq: 0, reqImg: 0,
    res: null as Result | null,
    p: { ...DEFAULTS, scales: [...DEFAULTS.scales] } as Params,
    selAnchor: -1, selCls: -1,
    nms: null as P.NmsResult | null, nmsMs: 0,
    t: { load: "", report: "" },
  };
  let worker: Worker | null = null;
  let stMod: LoadStatus | null = null, stW: LoadStatus | null = null;
  let runTimer = 0, busyTimer = 0;
  const post = (m: Req, transfer: Transferable[] = []) => worker?.postMessage(m, transfer);

  function setBusy(msg: string): void { busy.textContent = msg; busy.classList.toggle("yn-hide", !msg); }

  function syncControls(): void {
    sConf.set(S.p.conf); sIou.set(S.p.iou); sSize.set(S.p.size); cbAgn.checked = S.p.agnostic;
    cbScale.forEach((cb, k) => { cb.checked = S.p.scales[k]; });
    bRaw.classList.toggle("active", S.p.view === "raw"); bNms.classList.toggle("active", S.p.view === "nms");
    mapFigs.forEach((m, k) => m.f.classList.toggle("yn-off", !S.p.scales[k]));
  }

  // ---- Network runs -----------------------------------------------------------------------------
  function scheduleRun(): void {
    clearTimeout(runTimer);
    runTimer = window.setTimeout(requestRun, 250);
  }

  function requestRun(): void {
    if (!S.ready || !S.img) return;
    if (S.running) { S.pending = true; return; }
    const size = S.p.size, lb = P.letterbox(S.img.w, S.img.h, size);
    const c = document.createElement("canvas"); c.width = c.height = size;
    const g = c.getContext("2d")!;
    g.fillStyle = "#000"; g.fillRect(0, 0, size, size);
    g.imageSmoothingQuality = "high";
    g.drawImage(S.img.bmp, lb.x0, lb.y0, lb.nw, lb.nh);
    const input = P.toInput(g.getImageData(0, 0, size, size).data, size);
    S.running = true; S.reqSeq++; S.reqImg = S.imgSeq;
    const t0 = performance.now();
    const tick = () => setBusy(`Network running on ${size}×${size} … ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    tick(); clearInterval(busyTimer); busyTimer = window.setInterval(tick, 200);
    post({ type: "run", id: S.reqSeq, size, input: input.buffer as ArrayBuffer }, [input.buffer as ArrayBuffer]);
    pendingLb = lb;
  }
  let pendingLb: P.Letterbox | null = null;

  function onMessage(m: Res): void {
    switch (m.type) {
      case "progress": (m.what === "module" ? stMod : stW)?.progress(m.loaded, m.total); return;
      case "ready":
        stMod?.done(); stW?.done(); stMod = stW = null;
        S.ready = true;
        S.t.load = `Network loaded: ${formatBytes(m.wasmBytes)} module + ${formatBytes(m.weightBytes)} weights in ` +
          `${((m.fetchMs + m.buildMs) / 1000).toFixed(1)} s (download/cache ${(m.fetchMs / 1000).toFixed(1)} s, build ${(m.buildMs / 1000).toFixed(1)} s)`;
        requestRun();
        return;
      case "result": {
        clearInterval(busyTimer); setBusy("");
        S.running = false;
        // A newer request is queued (size or image changed meanwhile): drop this result.
        if (S.pending) { S.pending = false; requestRun(); return; }
        if (m.id !== S.reqSeq || S.reqImg !== S.imgSeq || !pendingLb) return;
        const raw = new Float32Array(m.raw), dec = new Float32Array(m.decoded);
        S.res = { size: m.size, lb: pendingLb, raw, dec, best: P.bestClass(dec), netMs: m.netMs, decodeMs: m.decodeMs };
        S.selAnchor = -1; S.selCls = -1;
        if (pendingPreset) pendingPreset = null;
        selectBest();
        refresh();
        return;
      }
      case "error":
        clearInterval(busyTimer); setBusy("");
        S.running = false;
        (stMod ?? stW ?? loadStatus(statusHost, "")).error("Error: " + m.message);
        return;
    }
  }

  // ---- Drawing --------------------------------------------------------------------------------
  function sizeStage(w: number, h: number): void {
    stage.style.aspectRatio = `${w} / ${h}`;
    // at most 300 px high on a slide; on a page at most 560 px and the window height minus room
    stage.style.maxWidth = host.closest(".reveal") ? `${Math.round((300 * w) / h)}px` : `min(${Math.round((560 * w) / h)}px, calc((100vh - 170px) * ${(w / h).toFixed(4)}))`;
    cImg.width = cOv.width = w; cImg.height = cOv.height = h;
  }

  function scaleOf(a: number): number {
    const sc = P.scales(S.res!.size);
    return a < sc[1].start ? 0 : a < sc[2].start ? 1 : 2;
  }

  function refresh(): void {
    syncControls();
    const r = S.res;
    if (r) {
      const t0 = performance.now();
      S.nms = P.nms(r.dec, S.p.conf, S.p.iou, 300, S.p.agnostic);
      S.nmsMs = performance.now() - t0;
    } else S.nms = null;
    drawOverlay(); drawMaps(); drawDfl(); showReadout();
  }

  function drawOverlay(): void {
    const g = cOv.getContext("2d")!;
    g.clearRect(0, 0, cOv.width, cOv.height);
    const r = S.res, n = S.nms;
    if (!r || !n || !S.img) return;
    const lw = Math.max(1.5, cOv.width / 400), fs = Math.max(12, Math.round(cOv.width / 38));
    const box = (a: number) => P.toImage(r.dec, r.lb, a * P.ND);
    const stroke = (b: P.Box, col: string, w: number, alpha: number, dash: number[] = []) => {
      g.globalAlpha = alpha; g.strokeStyle = col; g.lineWidth = w; g.setLineDash(dash);
      g.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
      g.globalAlpha = 1; g.setLineDash([]);
    };
    const label = (b: P.Box, text: string, col: string) => {
      g.font = `600 ${fs}px system-ui, sans-serif`;
      const tw = g.measureText(text).width + fs * 0.4;
      const y = Math.max(0, b[1] > fs * 1.2 ? b[1] - fs * 1.2 : b[1]);
      const x = Math.max(0, Math.min(b[0], cOv.width - tw));
      g.fillStyle = col; g.fillRect(x, y, tw, fs * 1.2);
      g.fillStyle = "#000"; g.fillText(text, x + fs * 0.2, y + fs * 0.95);
    };
    if (S.p.view === "raw") {
      // One box per (anchor, class) candidate, faint to strong by score; scales by colour.
      for (let k = n.cands.length - 1; k >= 0; k--) {
        const c = n.cands[k], s = scaleOf(c.anchor);
        if (!S.p.scales[s]) continue;
        stroke(box(c.anchor), SCALE_COL[s], lw, 0.25 + 0.75 * c.score);
      }
    } else {
      for (let k = n.kept.length - 1; k >= 0; k--) {
        const c = n.cands[n.kept[k]], s = scaleOf(c.anchor);
        if (!S.p.scales[s]) continue;
        const b = box(c.anchor);
        stroke(b, SCALE_COL[s], lw * 2, 1);
        label(b, `${P.COCO[c.cls]} ${c.score.toFixed(2)}`, SCALE_COL[s]);
      }
    }
    // Selection: the box, its grid cell and anchor point, and (after NMS) what it removed.
    if (S.selAnchor >= 0) {
      const a = S.selAnchor, cell = P.anchorCell(a, r.size), s = cell.stride;
      if (S.p.view === "nms") {
        const ki = n.cands.findIndex((c) => c.anchor === a && c.cls === S.selCls);
        if (ki >= 0) for (let j = 0; j < n.cands.length; j++) if (n.by[j] === ki) stroke(box(n.cands[j].anchor), "#fff", lw, 0.8, [lw * 3, lw * 2]);
      }
      const b = box(a);
      stroke(b, "#fff", lw * 3, 1);
      stroke(b, SCALE_COL[cell.scale], lw * 1.5, 1);
      const cb = P.toImage([cell.j * s, cell.i * s, (cell.j + 1) * s, (cell.i + 1) * s], r.lb);
      stroke(cb, "#fff", lw * 1.5, 1);
      const ax = ((cell.j + 0.5) * s - r.lb.x0) / r.lb.r, ay = ((cell.i + 0.5) * s - r.lb.y0) / r.lb.r;
      g.fillStyle = "#fff"; g.beginPath(); g.arc(ax, ay, lw * 3, 0, 2 * Math.PI); g.fill();
      g.strokeStyle = "#000"; g.lineWidth = lw; g.stroke();
    }
  }

  function drawMaps(): void {
    const r = S.res;
    let hi = 0;
    if (r) for (let a = 0; a < r.best.p.length; a++) hi = Math.max(hi, r.best.p[a]);
    hi = Math.max(hi, 1e-3);
    mapNote.textContent = r ? `Score maps: best class probability per cell, colour from 0 (black) to ${hi.toFixed(hi < 0.1 ? 3 : 2)} (the highest in this image, yellow); outlined: cells above the threshold. Click a cell to decode it.` : "";
    P.STRIDES.forEach((stride, k) => {
      const { c, fc } = mapFigs[k];
      const g = c.getContext("2d")!;
      g.fillStyle = "#000"; g.fillRect(0, 0, c.width, c.height);
      if (!r) { fc.textContent = `stride ${stride}`; return; }
      const sc = P.scales(r.size)[k];
      const vals = r.best.p.subarray(sc.start, sc.start + sc.count);
      const tmp = document.createElement("canvas"); tmp.width = tmp.height = sc.g;
      tmp.getContext("2d")!.putImageData(new ImageData(scalarToRGBA(vals, { map: inferno, lo: 0, hi }) as Uint8ClampedArray<ArrayBuffer>, sc.g, sc.g), 0, 0);
      g.imageSmoothingEnabled = false;
      g.drawImage(tmp, 0, 0, c.width, c.height);
      const cs = c.width / sc.g;
      // Cells above the threshold: outlined.
      g.strokeStyle = "#fff"; g.lineWidth = Math.max(1, cs / 6);
      let above = 0;
      for (let a = 0; a < sc.count; a++) if (vals[a] > S.p.conf) {
        above++;
        if (cs >= 4) g.strokeRect((a % sc.g) * cs + 0.5, Math.floor(a / sc.g) * cs + 0.5, cs - 1, cs - 1);
      }
      if (S.selAnchor >= sc.start && S.selAnchor < sc.start + sc.count) {
        const a = S.selAnchor - sc.start;
        g.strokeStyle = "#0f0"; g.lineWidth = Math.max(2, cs / 3);
        g.strokeRect((a % sc.g) * cs - 1, Math.floor(a / sc.g) * cs - 1, cs + 2, cs + 2);
      }
      fc.textContent = `stride ${stride}: ${sc.g}×${sc.g} cells, ${above} > ${S.p.conf.toFixed(2)}`;
    });
  }

  function drawDfl(): void {
    dfl.replaceChildren();
    const r = S.res;
    if (!r || S.selAnchor < 0) {
      svgEl("text", { x: 4, y: 20 }, dfl).textContent = "No box selected.";
      return;
    }
    const a = S.selAnchor, cell = P.anchorCell(a, r.size);
    const names = ["left", "top", "right", "bottom"];
    const W = 125, H = 70, top = 28;
    for (let side = 0; side < 4; side++) {
      const p = P.softmax(r.raw, a * P.NO + side * P.REG_MAX);
      const x0 = side * (W + 5) + 2, bw = W / P.REG_MAX;
      const pmax = Math.max(...p);
      svgEl("rect", { x: x0, y: top, width: W, height: H, fill: "var(--qb-panel)" }, dfl);
      p.forEach((v, k) => svgEl("rect", { x: x0 + k * bw + 0.5, y: top + H - (H * v) / Math.max(pmax, 1e-6), width: bw - 1,
        height: (H * v) / Math.max(pmax, 1e-6), fill: SCALE_COL[cell.scale] }, dfl));
      const d = r.dec[a * P.ND + 4 + side];
      svgEl("line", { x1: x0 + (d + 0.5) * bw, x2: x0 + (d + 0.5) * bw, y1: top - 2, y2: top + H, stroke: "var(--qb-fg)", "stroke-width": 2 }, dfl);
      svgEl("text", { x: x0, y: top - 8 }, dfl).textContent = `${names[side]}: ${d.toFixed(2)} cells`;
      svgEl("text", { x: x0, y: top + H + 16, fill: "var(--qb-muted)" }, dfl).textContent = "0";
      svgEl("text", { x: x0 + W, y: top + H + 16, "text-anchor": "end", fill: "var(--qb-muted)" }, dfl).textContent = "15";
    }
  }

  function showReadout(): void {
    const lines: string[] = [];
    const r = S.res, n = S.nms;
    if (r && n) {
      const sc = P.scales(r.size);
      lines.push(`Input ${r.size}×${r.size} (image scaled by ${r.lb.r.toFixed(2)}${r.lb.nw < r.size || r.lb.nh < r.size ? ", black bars" : ""}). Grid cells: ` +
        sc.map((s) => `${s.g}²`).join(" + ") + ` = ${P.anchorCount(r.size)}, each predicting 4 × 16 distance bins + 80 class scores.`);
      const per = [0, 0, 0], kept = [0, 0, 0];
      n.cands.forEach((c) => per[scaleOf(c.anchor)]++);
      n.kept.forEach((k) => kept[scaleOf(n.cands[k].anchor)]++);
      lines.push(`Score > ${S.p.conf.toFixed(2)}: ${n.cands.length} candidates (stride 8/16/32: ${per.join(" / ")}) → ` +
        `after NMS (IoU > ${S.p.iou.toFixed(2)}${S.p.agnostic ? ", any class" : ", same class"}): ${n.kept.length} (${kept.join(" / ")}).`);
      lines.push(`Network ${(r.netMs / 1000).toFixed(2)} s, decoding ${r.decodeMs.toFixed(0)} ms (WebAssembly, worker); ` +
        `threshold + NMS ${S.nmsMs.toFixed(1)} ms (in the page).`);
      if (S.selAnchor >= 0) {
        const a = S.selAnchor, cell = P.anchorCell(a, r.size), o = a * P.ND;
        const top3 = Array.from({ length: P.NC }, (_, c) => c).sort((x, y) => r.dec[o + 8 + y] - r.dec[o + 8 + x]).slice(0, 3);
        const b = P.toImage(r.dec, r.lb, o);
        lines.push(`Selected: stride ${cell.stride}, cell (row ${cell.i}, col ${cell.j}), centre (${((cell.j + 0.5) * cell.stride).toFixed(0)}, ` +
          `${((cell.i + 0.5) * cell.stride).toFixed(0)}) in the input; box ${(b[2] - b[0]).toFixed(0)}×${(b[3] - b[1]).toFixed(0)} image px. ` +
          `Classes: ${top3.map((c) => `${P.COCO[c]} ${r.dec[o + 8 + c].toFixed(2)}`).join(", ")}.`);
        if (S.p.view === "nms") {
          const ki = n.cands.findIndex((c) => c.anchor === a && c.cls === S.selCls);
          if (ki >= 0) {
            let removed = 0;
            for (let j = 0; j < n.cands.length; j++) if (n.by[j] === ki) removed++;
            lines.push(n.by[ki] === -1 ? `Kept; it removed ${removed} overlapping candidates (dashed).` : "Removed by NMS.");
          }
        }
      }
      if (S.p.view === "nms" && n.kept.length) {
        const list = n.kept.slice(0, 8).map((k) => `${P.COCO[n.cands[k].cls]} ${n.cands[k].score.toFixed(2)}`);
        lines.push(`Kept: ${list.join(", ")}${n.kept.length > 8 ? ", …" : ""}`);
      } else if (S.p.view === "nms") lines.push("Kept: nothing above the threshold.");
    }
    if (S.t.load) lines.push(S.t.load + ".");
    readout.textContent = lines.join("\n");
  }

  // ---- Picking ----------------------------------------------------------------------------------
  cOv.addEventListener("pointerdown", (ev) => {
    const r = S.res, n = S.nms;
    if (!r || !n || !S.img) return;
    ev.preventDefault();
    const rect = cOv.getBoundingClientRect();
    const x = (ev.clientX - rect.left) / rect.width * cOv.width, y = (ev.clientY - rect.top) / rect.height * cOv.height;
    const list = S.p.view === "raw" ? n.cands : n.kept.map((k) => n.cands[k]);
    let best: P.Candidate | null = null, bestArea = Infinity;
    for (const c of list) {
      if (!S.p.scales[scaleOf(c.anchor)]) continue;
      const b = P.toImage(r.dec, r.lb, c.anchor * P.ND);
      const area = (b[2] - b[0]) * (b[3] - b[1]);
      if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3] && area < bestArea) { best = c; bestArea = area; }
    }
    S.selAnchor = best ? best.anchor : -1; S.selCls = best ? best.cls : -1;
    drawOverlay(); drawMaps(); drawDfl(); showReadout();
  });
  mapFigs.forEach(({ c }, k) => c.addEventListener("pointerdown", (ev) => {
    const r = S.res;
    if (!r) return;
    ev.preventDefault();
    const sc = P.scales(r.size)[k], rect = c.getBoundingClientRect();
    const j = Math.min(sc.g - 1, Math.max(0, Math.floor((ev.clientX - rect.left) / rect.width * sc.g)));
    const i = Math.min(sc.g - 1, Math.max(0, Math.floor((ev.clientY - rect.top) / rect.height * sc.g)));
    S.selAnchor = sc.start + i * sc.g + j; S.selCls = r.best.cls[S.selAnchor];
    drawOverlay(); drawMaps(); drawDfl(); showReadout();
  }));

  // ---- Images ---------------------------------------------------------------------------------
  async function setImage(src: ImageBitmap, id: string, creditText: string): Promise<void> {
    const sc = Math.min(1, MAX_SIDE / Math.max(src.width, src.height));
    const w = Math.max(1, Math.round(src.width * sc)), h = Math.max(1, Math.round(src.height * sc));
    const bmp = sc < 1 ? await createImageBitmap(src, { resizeWidth: w, resizeHeight: h, resizeQuality: "high" }) : src;
    S.img = { w, h, bmp, id, credit: creditText };
    S.imgSeq++; S.res = null; S.nms = null; S.selAnchor = -1;
    sizeStage(w, h);
    cImg.getContext("2d")!.drawImage(bmp, 0, 0, w, h);
    credit.textContent = creditText + (sc < 1 ? ` (shown at ${w}×${h})` : ` (${w}×${h})`);
    refresh();
    requestRun();
  }

  async function loadSample(id: string): Promise<void> {
    const s = SAMPLES.find((x) => x.id === id) ?? SAMPLES[0];
    sel.value = s.id;
    if (isFileProtocol()) return;
    try {
      const [bytes, index] = await Promise.all([loadModelFile(s.key), modelIndex()]);
      const type = s.key.endsWith(".jpg") ? "image/jpeg" : "image/png";
      const bmp = await createImageBitmap(new Blob([bytes as BlobPart], { type }));
      await setImage(bmp, s.id, `Sample: ${String(index[s.key]?.credit ?? s.label)}`);
    } catch (e) {
      loadStatus(statusHost, "").error(`Could not load the sample image: ${(e as Error).message}`);
    }
  }

  sel.addEventListener("change", () => { if (sel.value !== "own") void loadSample(sel.value); sel.blur(); });
  fileIn.addEventListener("change", async () => {
    const f = fileIn.files?.[0];
    if (!f) return;
    try {
      const bmp = await createImageBitmap(f);
      oOwn.disabled = false; sel.value = "own";
      presetBtns.forEach((b) => b.classList.remove("active"));
      await setImage(bmp, "own", `Your image: ${f.name} (read locally, not uploaded)`);
    } catch {
      loadStatus(statusHost, "").error("This file could not be read as an image.");
    }
    fileIn.value = "";
  });

  // ---- Presets ----------------------------------------------------------------------------------
  let pendingPreset: string | null = null;
  function applyPreset(id: string): void {
    const p = PRESETS.find((x) => x.id === id);
    if (!p) return;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i].id === id));
    S.p = { ...DEFAULTS, scales: [...DEFAULTS.scales], ...p.p };
    syncControls();
    if (!S.img || S.img.id !== p.image) { pendingPreset = id; void loadSample(p.image); return; }
    if (!S.res || S.res.size !== S.p.size) { pendingPreset = id; requestRun(); return; }
    selectBest();
    refresh();
  }
  /** Select the best kept box (else the highest-scoring cell), so the distance distributions show something at once. */
  function selectBest(): void {
    const r = S.res;
    if (!r) return;
    const n = P.nms(r.dec, S.p.conf, S.p.iou, 300, S.p.agnostic);
    if (n.kept.length) { const c = n.cands[n.kept[0]]; S.selAnchor = c.anchor; S.selCls = c.cls; return; }
    let a = 0;
    for (let k = 1; k < r.best.p.length; k++) if (r.best.p[k] > r.best.p[a]) a = k;
    S.selAnchor = a; S.selCls = r.best.cls[a];
  }

  // ---- Start ----------------------------------------------------------------------------------
  const want = host.dataset.image && SAMPLES.some((s) => s.id === host.dataset.image) ? host.dataset.image! : "bus";
  const dsSize = Number(host.dataset.size);
  if (dsSize >= 128 && dsSize <= 800 && dsSize % 32 === 0) DEFAULTS.size = S.p.size = dsSize;
  sel.value = want;
  syncControls(); drawMaps(); drawDfl();
  if (isFileProtocol()) {
    loadStatus(statusHost, "Loading network");
    stage.style.aspectRatio = "4 / 3"; credit.textContent = fileProtocolMessage("This demo");
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
  worker.onerror = (e) => { (stMod ?? loadStatus(statusHost, "")).error("Worker error: " + (e.message || "could not load")); };
  post({ type: "init", base: distBase() });
  const preset = host.dataset.preset && PRESETS.find((p) => p.id === host.dataset.preset);
  if (preset) applyPreset(preset.id);
  else void loadSample(want);
}

// The same bundle is the worker: no document there.
if (typeof document === "undefined") runWorker();
else autoMount(NAME, mount);
