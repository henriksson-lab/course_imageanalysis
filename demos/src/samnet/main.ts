// Promptable segmentation with a real network: SAM 2.1 (Hiera tiny) running in the browser
// (Rust -> WebAssembly, demos/wasm/sam/, in a Web Worker). Networks for cells and objects day,
// unit sam/.
// What it shows: the heavy image encoder runs ONCE per image (its 64 x 64 x 256 embedding is
// drawn as RGB via its first three principal components); every click or box only runs the
// light prompt encoder + mask decoder, which returns THREE candidate masks with predicted IoU
// scores (the ambiguity of one click) as 256 x 256 logits, restored to the image by bilinear
// resizing and a threshold at 0. Automatic mask generation: a grid of single-point prompts,
// then the filters (predicted IoU, stability score, box NMS) live on sliders, with the fate of
// every grid point shown.
// Built-in sample micrographs (demos/dist/models/images/, credits shown) or the viewer's own
// image (read locally, never uploaded). Weights and module are fetched same-origin with a
// progress bar and cached (lib/wasm.ts); opened as a local file the demo explains why it
// cannot run.
// Auto-mounts on <div data-demo="samnet">. Options: data-image="nuclei|u2os|cells|histology|
// brightfield", data-preset="click|cluster|box|negative|auto|strict|loose".
import { htmlEl, button, slider, checkbox, presets, autoMount, svgEl, type Slider } from "../lib/ui";
import { distBase, isFileProtocol, fileProtocolMessage, loadStatus, loadModelFile, modelIndex, formatBytes, type LoadStatus } from "../lib/wasm";
import { scalarToRGBA, diverging, labelColor, type RGB } from "../lib/colour";
import * as P from "./post";
import type { Req, Res } from "./protocol";
import { runWorker } from "./worker";

const NAME = "samnet";
const MAX_SIDE = 1024; // larger images are scaled down first (the network sees 1024 x 1024 anyway)

const SAMPLES: { id: string; key: string; label: string }[] = [
  { id: "nuclei", key: "images/stardist-img2d.png", label: "Nuclei, fluorescence" },
  { id: "u2os", key: "images/bbbc039-A02_s1.png", label: "U2OS nuclei (Hoechst)" },
  { id: "cells", key: "images/cellpose-img02.png", label: "Cells: cytoplasm + nuclei" },
  { id: "histology", key: "images/dsb-histology.png", label: "Histology (H&E)" },
  { id: "brightfield", key: "images/dsb-brightfield.png", label: "Bright field" },
];

const C_POS = "#2ca02c", C_NEG = "#d62728", C_BOX = "#ff7f0e";
const CAND_RGB: RGB[] = [[31, 119, 180], [255, 127, 14], [148, 103, 189]];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { display: flex; flex-wrap: wrap; gap: .8em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .sn-left { flex: 1 1 460px; min-width: 260px; max-width: 560px; display: flex; flex-direction: column; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .sn-stage { position: relative; width: 100%; background: #000; border-radius: 4px; overflow: hidden; }
.qb-demo[data-demo="${NAME}"] .sn-stage canvas, .qb-demo[data-demo="${NAME}"] .sn-stage svg { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.qb-demo[data-demo="${NAME}"] .sn-stage svg { touch-action: none; cursor: crosshair; }
.qb-demo[data-demo="${NAME}"] .sn-busy { position: absolute; left: 0; right: 0; bottom: 0; padding: .3em .6em; font-size: 16px; color: #fff; background: rgba(0,0,0,.65); pointer-events: none; }
.qb-demo[data-demo="${NAME}"] .sn-credit { font-size: 13px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 380px; max-width: 560px; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .sn-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
.qb-demo[data-demo="${NAME}"] .sn-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] select { font: inherit; font-size: .9em; padding: .1em .2em; max-width: 15em; }
.qb-demo[data-demo="${NAME}"] .sn-figs { display: flex; gap: .4em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .sn-figs figure { margin: 0; flex: 1 1 0; min-width: 0; cursor: default; }
.qb-demo[data-demo="${NAME}"] .sn-figs figure.sn-pick { cursor: pointer; }
.qb-demo[data-demo="${NAME}"] .sn-figs canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; border: 3px solid transparent; background: #000; }
.qb-demo[data-demo="${NAME}"] .sn-figs canvas.sn-px { image-rendering: pixelated; }
.qb-demo[data-demo="${NAME}"] .sn-figs figure.sn-sel canvas { border-color: var(--qb-accent); }
.qb-demo[data-demo="${NAME}"] .sn-figs figcaption { font-size: 13px; text-align: center; line-height: 1.2; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre-wrap; line-height: 1.35; }
.qb-demo[data-demo="${NAME}"] .qb-btn:disabled { opacity: .45; cursor: default; }
.qb-demo[data-demo="${NAME}"] .sn-hide { display: none !important; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; gap: .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .sn-left { flex: 0 1 470px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 520px; max-width: 640px; gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; padding: .25em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note, .reveal .qb-demo[data-demo="${NAME}"] .sn-lab { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .sn-figs figcaption { font-size: 13px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

type Mode = "pos" | "neg" | "box";
interface Pt { x: number; y: number; l: 1 | 0 }
interface Decoded { logits: Float32Array; iou: number[]; masks: number; ms: number; tokens: number; obj: number; full: Uint8Array[] }

const PRESETS: { id: string; label: string; points?: Pt[]; box?: P.Box; auto?: Partial<P.AmgParams> }[] = [
  { id: "click", label: "one click on a nucleus", points: [{ x: 340, y: 317, l: 1 }] },
  { id: "cluster", label: "click in a cluster", points: [{ x: 149, y: 428, l: 1 }] },
  { id: "box", label: "box around a nucleus", box: [204, 243, 267, 290] },
  { id: "negative", label: "+ a background click", points: [{ x: 149, y: 428, l: 1 }, { x: 120, y: 410, l: 0 }] },
  { id: "auto", label: "automatic, 8 × 8 grid", auto: { pointsPerSide: 8, predIou: 0.8, stability: 0.95, nms: 0.7 } },
  { id: "strict", label: "automatic, strict filters", auto: { predIou: 0.9, stability: 0.97, nms: 0.7 } },
  { id: "loose", label: "automatic, loose filters", auto: { predIou: 0.6, stability: 0.7, nms: 0.95 } },
];

function mount(host: HTMLElement): void {
  injectStyle();
  // ---- DOM ------------------------------------------------------------------------------------
  const left = htmlEl("div", "sn-left", "", host);
  const stage = htmlEl("div", "sn-stage", "", left);
  const cImg = htmlEl("canvas", "", "", stage);
  const cOv = htmlEl("canvas", "", "", stage);
  const svg = svgEl("svg", {}, stage);
  const busy = htmlEl("div", "sn-busy sn-hide", "", stage);
  const credit = htmlEl("div", "sn-credit", "", left);

  const panel = htmlEl("div", "qb-panel", "", host);
  const statusHost = htmlEl("div", "", "", panel);
  htmlEl("div", "sn-lab", "Image", panel);
  const imgRow = htmlEl("div", "sn-row", "", panel);
  const sel = htmlEl("select", "", "", imgRow);
  for (const s of SAMPLES) { const o = htmlEl("option", "", s.label, sel); o.value = s.id; }
  const oOwn = htmlEl("option", "", "your image", sel); oOwn.value = "own"; oOwn.disabled = true;
  const fileIn = htmlEl("input", "sn-hide", "", imgRow);
  fileIn.type = "file"; fileIn.accept = "image/*";
  button(imgRow, "use your own image …", () => fileIn.click());
  htmlEl("div", "qb-note", "Your image is read in this browser and never uploaded.", panel);

  htmlEl("div", "sn-lab", "Prompts (click or drag on the image)", panel);
  const modeRow = htmlEl("div", "sn-row", "", panel);
  const modeBtns: Record<Mode, HTMLButtonElement> = {
    pos: button(modeRow, "+ object click", () => setMode("pos")),
    neg: button(modeRow, "− background click", () => setMode("neg")),
    box: button(modeRow, "box", () => setMode("box")),
  };
  button(modeRow, "undo", () => { if (S.box && !S.points.length) S.box = null; else S.points.pop(); promptsChanged(); });
  button(modeRow, "clear", () => { S.points = []; S.box = null; S.prevLow = null; promptsChanged(); });
  const optRow = htmlEl("div", "sn-row", "", panel);
  const cbMulti = checkbox(optRow, "three candidates", true, (v) => { S.multimask = v; promptsChanged(); });
  const cbRefine = checkbox(optRow, "feed back the chosen mask", false, (v) => { S.refine = v; if (!v) S.prevLow = null; promptsChanged(); });

  const figs = htmlEl("div", "sn-figs", "", panel);
  const mkFig = (cap: string, px = false) => {
    const f = htmlEl("figure", "", "", figs);
    const c = htmlEl("canvas", px ? "sn-px" : "", "", f);
    const fc = htmlEl("figcaption", "", cap, f);
    return { f, c, fc };
  };
  const fEmb = mkFig("image embedding (encoder, once)", true);
  const fLow = mkFig("decoder output", true);
  const fCand = [0, 1, 2].map((k) => mkFig(`mask ${k + 1}`));
  fCand.forEach((fc, k) => { fc.f.classList.add("sn-pick"); fc.f.addEventListener("click", () => { if (S.dec && k < S.dec.masks) { S.sel = k; drawPrompt(); } }); });

  htmlEl("div", "sn-lab", "Automatic masks (a grid of single clicks)", panel);
  const autoRow = htmlEl("div", "sn-row", "", panel);
  const runBtn = button(autoRow, "run on the grid", () => runAuto());
  const stopBtn = button(autoRow, "stop", () => { post({ type: "cancel" }); });
  const viewBtn = button(autoRow, "show click result", () => { S.view = S.view === "auto" ? "prompt" : "auto"; redraw(); });
  const sGrid = slider(panel, "grid points per side", 2, 16, 1, P.AMG_DEFAULT.pointsPerSide, (v) => { S.amgP.pointsPerSide = v; updateButtons(); }, (v) => `${v}×${v}`);
  const sIou = slider(panel, "keep if predicted IoU >", 0.5, 1, 0.01, P.AMG_DEFAULT.predIou, (v) => { S.amgP.predIou = v; refilter(); });
  const sStab = slider(panel, "keep if stability ≥", 0.5, 1, 0.01, P.AMG_DEFAULT.stability, (v) => { S.amgP.stability = v; refilter(); });
  const sNms = slider(panel, "remove if box IoU >", 0.3, 1, 0.01, P.AMG_DEFAULT.nms, (v) => { S.amgP.nms = v; refilter(); });
  htmlEl("div", "sn-lab", "Presets (on the nuclei image)", panel);
  const presetBtns = presets(panel, PRESETS.map((p) => ({ label: p.label, apply: () => applyPreset(p.id) })));
  const readout = htmlEl("div", "qb-readout", "", panel);

  // ---- State ----------------------------------------------------------------------------------
  const S = {
    ready: false, encoding: false, decoding: false, pendingDecode: false, amgRunning: false,
    img: null as null | { w: number; h: number; rgb: Uint8Array; id: string; credit: string },
    encodedId: 0, imgSeq: 0, reqSeq: 0, amgId: 0,
    mode: "pos" as Mode, points: [] as Pt[], box: null as P.Box | null,
    multimask: true, refine: false, prevLow: null as Float32Array | null,
    dec: null as Decoded | null, sel: 0,
    amgP: { ...P.AMG_DEFAULT } as P.AmgParams,
    cands: [] as P.Candidate[], amgDone: 0, amgTotal: 0, amgFilter: null as ReturnType<typeof P.filterCandidates> | null,
    view: "prompt" as "prompt" | "auto",
    t: { load: "", encode: "", decode: "", auto: "" },
    pendingPreset: null as string | null,
  };
  let worker: Worker | null = null;
  let stMod: LoadStatus | null = null, stW: LoadStatus | null = null;
  let encodeTimer = 0;

  const post = (m: Req, transfer: Transferable[] = []) => worker?.postMessage(m, transfer);

  function setBusy(msg: string): void { busy.textContent = msg; busy.classList.toggle("sn-hide", !msg); }
  function setMode(m: Mode): void {
    S.mode = m;
    (Object.keys(modeBtns) as Mode[]).forEach((k) => modeBtns[k].classList.toggle("active", k === m));
  }
  setMode("pos");
  function updateButtons(): void {
    const canRun = S.ready && !!S.img && S.encodedId === S.imgSeq && !S.encoding;
    runBtn.disabled = !canRun || S.amgRunning;
    stopBtn.disabled = !S.amgRunning;
    viewBtn.disabled = !S.cands.length;
    viewBtn.textContent = S.view === "auto" ? "show click result" : "show automatic masks";
  }
  function showReadout(): void {
    const lines: string[] = [];
    if (S.t.load) lines.push(S.t.load);
    if (S.t.encode) lines.push(S.t.encode);
    if (S.t.decode) lines.push(S.t.decode);
    if (S.view === "prompt" && S.dec) {
      lines.push(`Predicted IoU: ${S.dec.iou.map((v, k) => `${k === S.sel ? "▶" : ""}${v.toFixed(3)}`).join("  ")}` +
        `   object score ${S.dec.obj >= 0 ? "+" : ""}${S.dec.obj.toFixed(1)}`);
      const a = S.dec.full[S.sel];
      if (a && S.img) { let n = 0; for (let i = 0; i < a.length; i++) n += a[i]; lines.push(`Chosen mask: ${n} px (${(100 * n / a.length).toFixed(1)} % of the image)`); }
    }
    if (S.t.auto) lines.push(S.t.auto);
    if (S.amgFilter && S.view === "auto") {
      const c = S.amgFilter.counts, n = S.cands.length;
      lines.push(`${S.amgTotal} points → ${n} candidates → IoU filter drops ${c.iou} → stability drops ${c.stability + c.empty}` +
        ` → NMS drops ${c.nms} → ${c.kept} masks`);
    }
    readout.textContent = lines.join("\n");
  }

  // ---- Drawing --------------------------------------------------------------------------------
  function sizeStage(w: number, h: number): void {
    stage.style.aspectRatio = `${w} / ${h}`;
    cImg.width = cOv.width = w; cImg.height = cOv.height = h;
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    svg.setAttribute("preserveAspectRatio", "none");
  }
  function drawImage(): void {
    if (!S.img) return;
    const { w, h, rgb } = S.img;
    const id = new ImageData(w, h);
    for (let p = 0; p < w * h; p++) { id.data[4 * p] = rgb[3 * p]; id.data[4 * p + 1] = rgb[3 * p + 1]; id.data[4 * p + 2] = rgb[3 * p + 2]; id.data[4 * p + 3] = 255; }
    cImg.getContext("2d")!.putImageData(id, 0, 0);
  }
  /** Mask overlay: translucent fill, opaque edge. */
  function paintMask(data: Uint8ClampedArray, mask: ArrayLike<number>, w: number, h: number, c: RGB, fill = 0.38): void {
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const p = y * w + x;
      if (!mask[p]) continue;
      const edge = x === 0 || y === 0 || x === w - 1 || y === h - 1 || !mask[p - 1] || !mask[p + 1] || !mask[p - w] || !mask[p + w];
      const a = edge ? 1 : fill, o = 4 * p;
      data[o] = c[0]; data[o + 1] = c[1]; data[o + 2] = c[2]; data[o + 3] = Math.max(data[o + 3], Math.round(255 * a));
    }
  }
  function clearOverlay(): ImageData { return new ImageData(S.img!.w, S.img!.h); }

  function svgClear(): void { while (svg.firstChild) svg.removeChild(svg.firstChild); }
  function drawPromptsSvg(): void {
    if (!S.img) return;
    const r = Math.max(4, Math.max(S.img.w, S.img.h) / 70);
    if (S.box) {
      const [x0, y0, x1, y1] = S.box;
      svgEl("rect", { x: x0, y: y0, width: x1 - x0, height: y1 - y0, fill: "none", stroke: C_BOX, "stroke-width": r / 2.5 }, svg);
    }
    for (const p of S.points) {
      svgEl("circle", { cx: p.x, cy: p.y, r, fill: p.l ? C_POS : C_NEG, stroke: "#fff", "stroke-width": r / 3 }, svg);
      svgEl("path", { d: p.l ? `M${p.x - r * .55},${p.y}h${r * 1.1}M${p.x},${p.y - r * .55}v${r * 1.1}` : `M${p.x - r * .55},${p.y}h${r * 1.1}`, stroke: "#fff", "stroke-width": r / 3.5 }, svg);
    }
  }

  /** The square region the small panels zoom into: around the prompts and the masks that are
   *  not most of the image, padded; the whole image when there are no prompts. */
  function zoomRegion(): [number, number, number] {
    const { w, h } = S.img!;
    const xs: number[] = [], ys: number[] = [];
    for (const p of S.points) { xs.push(p.x); ys.push(p.y); }
    if (S.box) { xs.push(S.box[0], S.box[2]); ys.push(S.box[1], S.box[3]); }
    if (!xs.length) return [0, 0, Math.max(w, h)];
    for (const m of S.dec?.full ?? []) {
      const b = P.maskBox(m, w, h);
      if (b && b.area < 0.25 * w * h) { xs.push(b.box[0], b.box[2] + 1); ys.push(b.box[1], b.box[3] + 1); }
    }
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const side = Math.min(Math.max(w, h), Math.max(48, 1.5 * Math.max(x1 - x0, y1 - y0)));
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    return [Math.min(Math.max(0, cx - side / 2), Math.max(0, w - side)), Math.min(Math.max(0, cy - side / 2), Math.max(0, h - side)), side];
  }

  function drawThumb(c: HTMLCanvasElement, mask: ArrayLike<number> | null, col: RGB, z: [number, number, number]): void {
    if (!S.img) return;
    const { w, h } = S.img, side = 160, [zx, zy, zs] = z;
    c.width = side; c.height = side;
    const g = c.getContext("2d")!;
    g.fillStyle = "#000"; g.fillRect(0, 0, side, side);
    g.imageSmoothingEnabled = zs > side;
    g.drawImage(cImg, zx, zy, zs, zs, 0, 0, side, side);
    if (mask) {
      const ov = document.createElement("canvas"); ov.width = w; ov.height = h;
      const id = new ImageData(w, h);
      paintMask(id.data, mask, w, h, col, 0.5);
      ov.getContext("2d")!.putImageData(id, 0, 0);
      g.drawImage(ov, zx, zy, zs, zs, 0, 0, side, side);
    }
  }

  function drawPrompt(): void {
    if (!S.img) return;
    const { w, h } = S.img;
    const ov = clearOverlay();
    const d = S.dec;
    if (d && d.full[S.sel]) paintMask(ov.data, d.full[S.sel], w, h, CAND_RGB[S.sel]);
    cOv.getContext("2d")!.putImageData(ov, 0, 0);
    svgClear(); drawPromptsSvg();
    // Candidate thumbnails (zoomed to the prompts), chosen one framed.
    const z = zoomRegion();
    const zoomed = z[2] < Math.max(w, h) - 1;
    fCand.forEach((fc, k) => {
      const has = !!d && k < d.masks;
      fc.f.classList.toggle("sn-sel", has && k === S.sel);
      fc.f.style.visibility = has || !d ? "visible" : "hidden";
      drawThumb(fc.c, has ? d!.full[k] : null, CAND_RGB[k], z);
      fc.fc.textContent = has ? `${d!.masks > 1 ? `mask ${k + 1}` : "single mask"}: IoU score ${d!.iou[k].toFixed(2)}` : `mask ${k + 1}`;
    });
    // Low-resolution logits of the chosen candidate, same region (its coarse grid shows).
    if (d) {
      const lo = P.plane(d.logits, S.sel);
      const off = document.createElement("canvas"); off.width = off.height = P.LOW;
      const rgba = new Uint8ClampedArray(scalarToRGBA(lo, { map: diverging, lo: -12, hi: 12 }));
      off.getContext("2d")!.putImageData(new ImageData(rgba, P.LOW, P.LOW), 0, 0);
      fLow.c.width = fLow.c.height = 160;
      const g = fLow.c.getContext("2d")!;
      g.imageSmoothingEnabled = false;
      g.drawImage(off, z[0] * P.LOW / w, z[1] * P.LOW / h, z[2] * P.LOW / w, z[2] * P.LOW / h, 0, 0, 160, 160);
      fLow.fc.textContent = `decoder output, 256² logits${zoomed ? " (zoom)" : ""}: red > 0`;
    } else {
      const g = fLow.c.getContext("2d")!; fLow.c.width = fLow.c.height = 8; g.clearRect(0, 0, 8, 8);
    }
    showReadout(); updateButtons();
  }

  function drawAuto(): void {
    if (!S.img) return;
    const { w, h } = S.img;
    const ov = clearOverlay();
    const f = S.amgFilter;
    if (f) {
      // Largest first, so small objects stay visible on top.
      const order = f.kept.map((_, i) => i).sort((a, b) => f.kept[b].area - f.kept[a].area);
      for (const i of order) {
        const big = f.kept[i].area > 0.5 * w * h;
        paintMask(ov.data, f.kept[i].mask!, w, h, labelColor(i + 1).map((v) => 255 * v) as RGB, big ? 0.15 : 0.5);
      }
    }
    cOv.getContext("2d")!.putImageData(ov, 0, 0);
    svgClear();
    // Grid points: filled = one of its candidates was kept; hollow = all dropped; small = not yet run.
    const r = Math.max(3, Math.max(w, h) / 110);
    const keptBy = new Set<number>();
    if (f) S.cands.forEach((c, i) => { if (f.fate[i] === "kept") keptBy.add(c.point); });
    const pts = P.pointGrid(S.amgTotal ? Math.round(Math.sqrt(S.amgTotal)) : S.amgP.pointsPerSide);
    pts.forEach(([x, y], i) => {
      const done = i < S.amgDone, kept = keptBy.has(i);
      svgEl("circle", { cx: x * w, cy: y * h, r: done ? r : r * 0.6, fill: kept ? "#fff" : "none",
        stroke: kept ? "#000" : done ? "#ddd" : "#888", "stroke-width": r / 3, opacity: done ? 1 : 0.7 }, svg);
    });
    showReadout(); updateButtons();
  }

  function redraw(): void { if (S.view === "auto") drawAuto(); else drawPrompt(); }

  // ---- Worker messages ------------------------------------------------------------------------
  function onMessage(m: Res): void {
    switch (m.type) {
      case "progress": {
        const st = m.what === "module" ? stMod : stW;
        st?.progress(m.loaded, m.total);
        return;
      }
      case "ready":
        stMod?.done(); stW?.done(); stMod = stW = null;
        S.ready = true;
        S.t.load = `Network loaded: ${formatBytes(m.wasmBytes)} module + ${formatBytes(m.weightBytes)} weights, ` +
          `${((m.fetchMs + m.buildMs) / 1000).toFixed(1)} s (download/cache ${(m.fetchMs / 1000).toFixed(1)} s, build ${(m.buildMs / 1000).toFixed(1)} s)`;
        encodeIfNeeded();
        return;
      case "encoded":
        clearInterval(encodeTimer);
        S.encoding = false;
        if (m.id !== S.imgSeq) { encodeIfNeeded(); return; }
        S.encodedId = m.id;
        S.t.encode = `Image encoder: ${(m.ms / 1000).toFixed(1)} s (once per image)`;
        setBusy("");
        {
          fEmb.c.width = fEmb.c.height = 64;
          fEmb.c.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(m.pca), 64, 64), 0, 0);
          fEmb.fc.textContent = "image embedding 64²×256, 3 main directions (encoder, once)";
        }
        if (S.pendingPreset) { const p = S.pendingPreset; S.pendingPreset = null; applyPreset(p); }
        else if (S.points.length || S.box) requestDecode();
        updateButtons(); showReadout();
        return;
      case "decoded": {
        S.decoding = false;
        if (S.pendingDecode) { S.pendingDecode = false; requestDecode(); }
        if (m.id !== S.reqSeq || !S.img) return;
        const logits = new Float32Array(m.logits);
        const full: Uint8Array[] = [];
        for (let k = 0; k < m.masks; k++) full.push(P.threshold(P.restore(P.plane(logits, k), S.img.w, S.img.h), 0));
        let best = 0;
        m.iou.forEach((v, k) => { if (v > m.iou[best]) best = k; });
        S.dec = { logits, iou: m.iou, masks: m.masks, ms: m.ms, tokens: m.tokens, obj: m.objectScore, full };
        S.sel = best;
        S.t.decode = `Prompt encoder + mask decoder: ${m.ms.toFixed(0)} ms (per click; ${m.tokens} prompt tokens)`;
        setBusy("");
        S.view = "prompt";
        drawPrompt();
        return;
      }
      case "amgBatch": {
        if (m.id !== S.amgId || !S.img) return;
        for (const c of m.candidates) S.cands.push({ ...c, mask: c.mask ? new Uint8Array(c.mask) : null });
        S.amgDone = m.done; S.amgTotal = m.total;
        const left = m.done ? (m.elapsedMs / m.done) * (m.total - m.done) / 1000 : 0;
        setBusy(`Decoding the grid: ${m.done} / ${m.total} points, about ${Math.ceil(left)} s left`);
        refilter();
        return;
      }
      case "amgDone":
        if (m.id !== S.amgId) return;
        S.amgRunning = false; setBusy("");
        S.t.auto = `Automatic: ${m.points} points in ${(m.ms / 1000).toFixed(1)} s ` +
          `(decoder ${(m.decodeMs / 1000).toFixed(1)} s, restoring + scoring masks ${(m.postMs / 1000).toFixed(1)} s)`;
        refilter();
        return;
      case "amgCancelled":
        S.amgRunning = false; setBusy("");
        S.t.auto = `Automatic: stopped after ${S.amgDone} of ${S.amgTotal} points`;
        refilter();
        return;
      case "error":
        clearInterval(encodeTimer);
        S.encoding = S.decoding = S.amgRunning = false;
        setBusy("");
        (stMod ?? stW ?? loadStatus(statusHost, "")).error("Error: " + m.message);
        updateButtons();
        return;
    }
  }

  function encodeIfNeeded(): void {
    if (!S.ready || !S.img || S.encoding || S.encodedId === S.imgSeq) return;
    S.encoding = true;
    S.dec = null; S.cands = []; S.amgFilter = null; S.amgDone = S.amgTotal = 0; S.t.auto = ""; S.t.decode = "";
    const id = S.imgSeq, t0 = performance.now();
    const tick = () => setBusy(`Image encoder running … ${((performance.now() - t0) / 1000).toFixed(0)} s (once per image)`);
    tick();
    encodeTimer = window.setInterval(tick, 500);
    const rgb = S.img.rgb.slice();
    post({ type: "encode", id, rgb: rgb.buffer, w: S.img.w, h: S.img.h }, [rgb.buffer]);
    redraw();
  }

  function requestDecode(): void {
    if (!S.img || S.encodedId !== S.imgSeq) return;
    if (!S.points.length && !S.box) { S.dec = null; S.t.decode = ""; drawPrompt(); return; }
    if (S.decoding) { S.pendingDecode = true; return; }
    S.decoding = true;
    S.reqSeq++;
    const mask = S.refine && S.prevLow ? S.prevLow.slice() : null;
    post({ type: "decode", id: S.reqSeq, points: S.points.flatMap((p) => [p.x, p.y]), labels: S.points.map((p) => p.l),
      box: S.box ? [...S.box] : null, maskInput: mask ? mask.buffer : null, multimask: S.multimask }, mask ? [mask.buffer] : []);
  }

  function promptsChanged(): void {
    if (S.refine && S.dec) S.prevLow = P.plane(S.dec.logits, S.sel).slice();
    S.view = "prompt";
    drawPrompt();
    requestDecode();
  }

  function refilter(): void {
    S.amgFilter = S.cands.length ? P.filterCandidates(S.cands, S.amgP) : null;
    if (S.cands.length) S.view = "auto";
    redraw();
  }

  function runAuto(): void {
    if (!S.img || S.encodedId !== S.imgSeq || S.amgRunning) return;
    S.amgId++; S.amgRunning = true;
    S.cands = []; S.amgFilter = null; S.amgDone = 0; S.amgTotal = S.amgP.pointsPerSide ** 2;
    S.view = "auto"; S.t.auto = "";
    setBusy(`Decoding the grid: 0 / ${S.amgTotal} points`);
    post({ type: "amg", id: S.amgId, pointsPerSide: S.amgP.pointsPerSide, batch: 4, stabilityOffset: S.amgP.stabilityOffset });
    redraw();
  }

  // ---- Images ---------------------------------------------------------------------------------
  async function setImageFromBitmap(bmp: ImageBitmap, id: string, creditText: string): Promise<void> {
    const sc = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
    const w = Math.max(1, Math.round(bmp.width * sc)), h = Math.max(1, Math.round(bmp.height * sc));
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    const g = c.getContext("2d")!;
    g.drawImage(bmp, 0, 0, w, h);
    const d = g.getImageData(0, 0, w, h).data, rgb = new Uint8Array(w * h * 3);
    for (let p = 0; p < w * h; p++) { rgb[3 * p] = d[4 * p]; rgb[3 * p + 1] = d[4 * p + 1]; rgb[3 * p + 2] = d[4 * p + 2]; }
    if (S.amgRunning) post({ type: "cancel" });
    S.img = { w, h, rgb, id, credit: creditText };
    S.imgSeq++;
    S.points = []; S.box = null; S.prevLow = null; S.dec = null; S.cands = []; S.amgFilter = null;
    S.amgDone = S.amgTotal = 0; S.t.encode = S.t.decode = S.t.auto = ""; S.view = "prompt";
    sizeStage(w, h); drawImage();
    credit.textContent = creditText + (sc < 1 ? ` (scaled to ${w}×${h})` : ` (${w}×${h})`);
    redraw();
    encodeIfNeeded();
  }

  async function loadSample(id: string): Promise<void> {
    const s = SAMPLES.find((x) => x.id === id) ?? SAMPLES[0];
    sel.value = s.id;
    if (isFileProtocol()) return;
    try {
      const [bytes, index] = await Promise.all([loadModelFile(s.key), modelIndex()]);
      const bmp = await createImageBitmap(new Blob([bytes as BlobPart], { type: "image/png" }));
      await setImageFromBitmap(bmp, s.id, `Sample: ${String(index[s.key]?.credit ?? s.label)}`);
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
      await setImageFromBitmap(bmp, "own", `Your image: ${f.name} (read locally, not uploaded)`);
    } catch {
      loadStatus(statusHost, "").error("This file could not be read as an image.");
    }
    fileIn.value = "";
  });

  function applyPreset(id: string): void {
    const p = PRESETS.find((x) => x.id === id);
    if (!p) return;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i].id === id));
    if (!S.img || S.img.id !== "nuclei") { S.pendingPreset = id; void loadSample("nuclei"); return; }
    if (S.encodedId !== S.imgSeq) { S.pendingPreset = id; return; }
    if (p.auto) {
      Object.assign(S.amgP, p.auto);
      sGrid.set(S.amgP.pointsPerSide); sIou.set(S.amgP.predIou); sStab.set(S.amgP.stability); sNms.set(S.amgP.nms);
      if (!S.cands.length || (p.auto.pointsPerSide && p.auto.pointsPerSide ** 2 !== S.amgTotal)) runAuto();
      else refilter();
      return;
    }
    S.points = (p.points ?? []).map((q) => ({ ...q }));
    S.box = p.box ? [...p.box] as P.Box : null;
    S.prevLow = null;
    S.multimask = true; cbMulti.checked = true;
    S.refine = false; cbRefine.checked = false;
    promptsChanged();
  }

  // ---- Pointer input on the image -------------------------------------------------------------
  let drag: { x0: number; y0: number; rect: SVGRectElement | null } | null = null;
  const toImg = (ev: PointerEvent): [number, number] => {
    const r = svg.getBoundingClientRect();
    const w = S.img?.w ?? 1, h = S.img?.h ?? 1;
    return [Math.min(w - 1, Math.max(0, (ev.clientX - r.left) / r.width * w)), Math.min(h - 1, Math.max(0, (ev.clientY - r.top) / r.height * h))];
  };
  svg.addEventListener("pointerdown", (ev) => {
    if (!S.img) return;
    ev.preventDefault();
    const [x, y] = toImg(ev);
    drag = { x0: x, y0: y, rect: null };
    svg.setPointerCapture(ev.pointerId);
  });
  svg.addEventListener("pointermove", (ev) => {
    if (!drag || !S.img) return;
    const [x, y] = toImg(ev);
    if (S.mode !== "box" && !ev.shiftKey) return;
    if (!drag.rect) drag.rect = svgEl("rect", { fill: "none", stroke: C_BOX, "stroke-width": Math.max(2, S.img.w / 250), "stroke-dasharray": "6 4" }, svg);
    const [a, b] = [Math.min(drag.x0, x), Math.min(drag.y0, y)];
    Object.entries({ x: a, y: b, width: Math.abs(x - drag.x0), height: Math.abs(y - drag.y0) }).forEach(([k, v]) => drag!.rect!.setAttribute(k, String(v)));
  });
  svg.addEventListener("pointerup", (ev) => {
    if (!drag || !S.img) return;
    const [x, y] = toImg(ev);
    const d = drag; drag = null;
    const big = Math.hypot(x - d.x0, y - d.y0) > Math.max(4, S.img.w / 100);
    if ((S.mode === "box" || ev.shiftKey) && big) {
      S.box = [Math.min(d.x0, x), Math.min(d.y0, y), Math.max(d.x0, x), Math.max(d.y0, y)];
    } else if (S.mode !== "box") {
      S.points.push({ x, y, l: S.mode === "pos" ? 1 : 0 });
    } else { d.rect?.remove(); return; }
    presetBtns.forEach((b) => b.classList.remove("active"));
    promptsChanged();
  });

  // ---- Start ----------------------------------------------------------------------------------
  updateButtons();
  const want = host.dataset.image && SAMPLES.some((s) => s.id === host.dataset.image) ? host.dataset.image! : "nuclei";
  sel.value = want;
  if (isFileProtocol()) {
    loadStatus(statusHost, "Loading network");
    setBusy("");
    stage.style.aspectRatio = "1"; credit.textContent = fileProtocolMessage("This demo");
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
  if (host.dataset.preset) S.pendingPreset = host.dataset.preset;
  void loadSample(want);
}

// The same bundle is the worker: no document there.
if (typeof document === "undefined") runWorker();
else autoMount(NAME, mount);
