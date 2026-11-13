// Interactive graph-cut segmentation (Boykov & Jolly 2001) of a square image (128, 256 or
// 512 pixels). Paint object and background seeds; the intensity model (a Gaussian or a
// histogram per label) is estimated from the seed pixels and gives the data term
// -log P(I | label); n-links carry lambda * w_pq with w_pq = exp(-(I_p - I_q)^2 / 2 sigma^2)
// (sigma = infinity: w = 1), over a 4- or 8-neighbourhood. One max-flow gives the exact minimum:
// in Rust/WebAssembly (wasmcut.ts, Boykov-Kolmogorov), or lib/maxflow.ts if Wasm is unavailable.
// Beside it: the thresholded result (lambda = 0) and the energy of both. Presets show the
// typical failure modes: shrinking bias, blocky boundaries, a bad intensity model.
// Images: a synthetic test image, an electron micrograph (inlined), and public microscopy
// images from demos/dist/models/images/ when the site has them (`make models`, CI).
// Auto-mounts on <div data-demo="graphcut">. Optional: data-preset="noise|shrink|blocky|nucleus|badmodel".
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { modelIndex, loadModelFile, isFileProtocol, IndexEntry } from "../lib/wasm";
import {
  N, SIZES, PRESETS, Preset, Settings, Segmenter, synthImage, synthTruth, downsample, resampleImage,
  resampleSeeds, addNoise, rasterize, paintLine, Segmentation,
} from "./model";
import { wasmSolver } from "./wasmcut";
// @ts-ignore -- esbuild inlines the image as a data URL
import cellUrl from "../fourier2d/cell256.png";

const NAME = "graphcut";
const CELL_CREDIT = "Image: transmission electron micrograph of a mesenchymal stem cell, Robert M. Hunt, CC BY 3.0, via Wikimedia Commons";
const SIGMAS = [3, 6, 12, 25, 50, 100, Infinity];
const BRUSH = 2;               // brush radius at 128 x 128 (scaled with the image)
const LIVE_MS = 60;            // update while painting if an update takes less than this
const SLOW_MS = 150;           // show "computing" first if the last update took longer

/** Sample images from demos/dist/models/images/ (tools/models/manifest.json), shown when present. */
const SAMPLES = [
  { key: "images/bbbc039-A02_s1.png", label: "nuclei" },
  { key: "images/stardist-img2d.png", label: "nuclei, dense" },
  { key: "images/dsb-brightfield.png", label: "brightfield" },
  { key: "images/dsb-histology.png", label: "histology" },
];

const D = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${D} { gap: .9em; }
${D} figure { margin: 0; }
${D} figcaption { font-size: 17px; text-align: center; line-height: 1.2; margin: 0 0 .2em; }
${D} .gc-main { flex: 0 1 410px; min-width: 240px; }
${D} .gc-main canvas { cursor: crosshair; }
${D} canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; touch-action: none; image-rendering: pixelated; }
${D} .gc-mid { flex: 0 1 290px; min-width: 200px; display: flex; flex-direction: column; gap: .4em; }
${D} .gc-mid figure { max-width: 210px; align-self: center; width: 100%; }
${D} .qb-readout { font-size: 14px; white-space: pre; }
${D} .qb-panel { flex: 1 1 250px; max-width: 330px; gap: .4em; }
${D} .gc-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.25em; }
${D} .gc-credit { font-size: 12px; color: var(--qb-muted); min-height: 1.2em; }
${D} .gc-status { font-size: 13px; color: var(--qb-muted); min-height: 1.2em; }
${D} .qb-btn.obj.active { background: #ff7f0e; border-color: #ff7f0e; }
${D} .qb-btn.bg.active { background: #1f77b4; border-color: #1f77b4; }
.reveal ${D} { font-size: 17px; }
.reveal ${D} .gc-main { flex-basis: 400px; }
.reveal ${D} .qb-readout { font-size: 14px; }
.reveal ${D} .qb-note { display: none; }
.reveal ${D} .qb-panel { gap: .25em; }
.reveal ${D} .qb-btn { font-size: 15px; padding: .15em .5em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

/** Grey levels (0..255) of a centred square crop of an image, resampled to side n by the canvas. */
function greyOf(img: CanvasImageSource, w: number, h: number, n: number): Float64Array {
  const c = document.createElement("canvas");
  c.width = n; c.height = n;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  const s = Math.min(w, h);
  ctx.drawImage(img, (w - s) / 2, (h - s) / 2, s, s, 0, 0, n, n);
  const d = ctx.getImageData(0, 0, n, n).data;
  const out = new Float64Array(n * n);
  for (let k = 0; k < out.length; k++) out[k] = 0.299 * d[4 * k] + 0.587 * d[4 * k + 1] + 0.114 * d[4 * k + 2];
  return out;
}

/** The micrograph at 256 x 256 (its own size), red channel as before. */
function loadCell(): Promise<Float64Array> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = 256; c.height = 256;
      const ctx = c.getContext("2d", { willReadFrequently: true })!;
      ctx.drawImage(img, 0, 0, 256, 256);
      const d = ctx.getImageData(0, 0, 256, 256).data;
      const big = new Float64Array(256 * 256);
      for (let k = 0; k < big.length; k++) big[k] = d[4 * k];
      resolve(big);
    };
    img.onerror = reject;
    img.src = cellUrl;
  });
}

/** A sample image from the site's models folder, as a decoded bitmap. */
async function loadSample(key: string): Promise<ImageBitmap> {
  const bytes = await loadModelFile(key);
  return await createImageBitmap(new Blob([bytes as BlobPart], { type: "image/png" }));
}

/** Draw the image with the labelling (tint + boundary) and the seeds on top. */
function paint(c: HTMLCanvasElement, I: ArrayLike<number>, L: Uint8Array | null, seeds: Int8Array | null): void {
  const n = c.width;
  const ctx = c.getContext("2d")!;
  const id = ctx.createImageData(n, n);
  for (let p = 0; p < n * n; p++) {
    const g = I[p];
    let r = g, gg = g, b = g;
    if (L) {
      const x = p % n, y = (p - x) / n;
      const edge = L[p] === 1 && ((x > 0 && !L[p - 1]) || (x < n - 1 && !L[p + 1]) || (y > 0 && !L[p - n]) || (y < n - 1 && !L[p + n]));
      if (edge) { r = 255; gg = 230; b = 0; }
      else if (L[p]) { r = 0.55 * g + 0.45 * 255; gg = 0.55 * g + 0.45 * 127; b = 0.55 * g + 0.45 * 14; }
    }
    if (seeds && seeds[p] === 1) { r = 230; gg = 40; b = 40; }
    if (seeds && seeds[p] === -1) { r = 30; gg = 90; b = 230; }
    id.data[4 * p] = r; id.data[4 * p + 1] = gg; id.data[4 * p + 2] = b; id.data[4 * p + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

const f1 = (v: number) => v.toFixed(1);

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = {
    image: p0.image as string, size: N, noise: p0.noise, settings: { ...p0.settings } as Settings,
    seeds: rasterize(p0.strokes), brush: 1 as 1 | -1 | 0, show: true,
  };
  const seg = new Segmenter();       // TypeScript max flow, replaced by the Wasm one when it loads
  let wasmReady = false, solverSettled = false;

  // ---- layout ----
  const main = htmlEl("figure", "gc-main", "", host);
  const mainCap = htmlEl("figcaption", "", "graph cut: paint seeds here", main);
  const cv = htmlEl("canvas", "", "", main);
  const credit = htmlEl("div", "gc-credit", "", main);
  const mid = htmlEl("div", "gc-mid", "", host);
  const thrFig = htmlEl("figure", "", "", mid);
  htmlEl("figcaption", "", "threshold (λ = 0)", thrFig);
  const cvT = htmlEl("canvas", "", "", thrFig);
  const ro = htmlEl("div", "qb-readout", "", mid);
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "gc-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "gc-lab", "image", panel);
  const imgRow = htmlEl("div", "qb-presets", "", panel);
  const imgBtns = new Map<string, HTMLButtonElement>();
  // A new image starts without seeds: seeds painted on another image mean nothing here.
  const addImage = (id: string, label: string) => imgBtns.set(id, button(imgRow, label, () => {
    if (id === st.image) return;
    st.image = id; st.seeds = new Int8Array(st.size * st.size); unPreset(); makeImage();
  }));
  addImage("synth", "test image");
  addImage("cell", "micrograph");
  const sizeRow = htmlEl("div", "qb-presets", "", panel);
  const sizeBtns = SIZES.map(n => button(sizeRow, `${n}²`, () => setSize(n)));
  const status = htmlEl("div", "gc-status", "", panel);
  const noiseS = slider(panel, "noise", 0, 100, 5, st.noise, v => { st.noise = v; makeImage(); });
  htmlEl("div", "gc-lab", "paint", panel);
  const brushRow = htmlEl("div", "qb-presets", "", panel);
  const brushBtns = [
    button(brushRow, "object", () => setBrush(1), "obj"),
    button(brushRow, "background", () => setBrush(-1), "bg"),
    button(brushRow, "erase", () => setBrush(0)),
    button(brushRow, "clear", () => { st.seeds = new Int8Array(st.size * st.size); unPreset(); run(); }),
  ];
  // lambda slider is quadratic: lambda = 20 (u / 100)^2
  const lamOf = (u: number) => Math.round(20 * (u / 100) ** 2 * 20) / 20;
  const uOf = (l: number) => 100 * Math.sqrt(l / 20);
  const lamS = slider(panel, "λ", 0, 100, 0.5, uOf(st.settings.lambda), u => { st.settings.lambda = lamOf(u); run(); }, u => lamOf(u).toFixed(2));
  const sigS = slider(panel, "σ", 0, SIGMAS.length - 1, 1, SIGMAS.indexOf(st.settings.sigma),
    v => { st.settings.sigma = SIGMAS[v]; run(); }, v => (Number.isFinite(SIGMAS[v]) ? String(SIGMAS[v]) : "∞"));
  htmlEl("div", "gc-lab", "neighbours · intensity model", panel);
  const optRow = htmlEl("div", "qb-presets", "", panel);
  const nbBtns = [4, 8].map(k => button(optRow, String(k), () => { st.settings.nbhd = k as 4 | 8; run(); }));
  const modelBtns = (["gauss", "hist"] as const).map(m => button(optRow, m === "gauss" ? "Gaussian" : "histogram", () => { st.settings.model = m; run(); }));
  checkbox(panel, "show the result", st.show, v => { st.show = v; draw(); });
  button(panel, "reset", () => load(p0));
  const note = htmlEl("div", "qb-note", "", panel);

  function setBrush(b: 1 | -1 | 0): void { st.brush = b; syncButtons(); }
  function unPreset(): void { presetBtns.forEach(b => b.classList.remove("active")); }

  function load(p: Preset): void {
    st.image = p.image; st.size = N; st.noise = p.noise; st.settings = { ...p.settings }; st.seeds = rasterize(p.strokes);
    noiseS.set(st.noise); lamS.set(uOf(st.settings.lambda)); sigS.set(SIGMAS.indexOf(st.settings.sigma));
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    makeImage();
  }

  function setSize(n: number): void {
    if (n === st.size) return;
    st.seeds = resampleSeeds(st.seeds, n);
    st.size = n;
    unPreset();
    makeImage();
  }

  function syncButtons(): void {
    imgBtns.forEach((b, id) => b.classList.toggle("active", id === st.image));
    sizeBtns.forEach((b, i) => b.classList.toggle("active", SIZES[i] === st.size));
    brushBtns.forEach((b, i) => b.classList.toggle("active", i < 3 && [1, -1, 0][i] === st.brush));
    nbBtns.forEach((b, i) => b.classList.toggle("active", [4, 8][i] === st.settings.nbhd));
    modelBtns.forEach((b, i) => b.classList.toggle("active", ["gauss", "hist"][i] === st.settings.model));
  }

  // ---- images ----
  // Clean images per (image, size), so a slider move does not resample again.
  const clean = new Map<string, Float64Array>();
  const sources = new Map<string, Promise<unknown>>();
  const credits = new Map<string, string>([["cell", CELL_CREDIT]]);
  let cellBig: Float64Array | null = null;
  const bitmaps = new Map<string, ImageBitmap>();
  let I: Float64Array = new Float64Array(N * N);
  let shown = "";   // image + size + noise of I

  function cleanImage(id: string, n: number): Float64Array | null {
    const k = `${id}/${n}`;
    let c = clean.get(k);
    if (c) return c;
    if (id === "synth") c = synthImage(n);
    else if (id === "cell") {
      if (!cellBig) return null;
      c = n === 128 ? downsample(cellBig) : resampleImage(cellBig, n);
    } else {
      const bm = bitmaps.get(id);
      if (!bm) return null;
      c = greyOf(bm, bm.width, bm.height, n);
    }
    clean.set(k, c);
    return c;
  }

  function fetchSource(id: string): void {
    if (sources.has(id)) return;
    status.textContent = "loading the image …";
    const p = id === "cell"
      ? loadCell().then(v => { cellBig = v; })
      : loadSample(id).then(bm => { bitmaps.set(id, bm); });
    sources.set(id, p);
    p.then(() => { status.textContent = ""; makeImage(); },
      e => { status.textContent = `Could not load the image: ${(e as Error).message}`; });
  }

  function makeImage(): void {
    syncButtons();
    const c = cleanImage(st.image, st.size);
    if (!c) { fetchSource(st.image); return; }
    const key = `${st.image}/${st.size}/${st.noise}`;
    if (key !== shown) { I = addNoise(c, st.noise); shown = key; }
    if (cv.width !== st.size) { cv.width = cv.height = cvT.width = cvT.height = st.size; }
    if (st.seeds.length !== st.size * st.size) st.seeds = resampleSeeds(st.seeds, st.size);
    credit.textContent = credits.get(st.image) ?? "";
    run();
  }

  // Sample microscopy images, when the site has them (not from file://, not before `make models`).
  if (!isFileProtocol()) {
    modelIndex().then((index: Record<string, IndexEntry>) => {
      const have = SAMPLES.filter(s => index[s.key]);
      for (const s of have) {
        const e = index[s.key];
        credits.set(s.key, typeof e.credit === "string" ? `Image: ${e.credit}` : "");
        addImage(s.key, s.label);
      }
      syncButtons();
    }, () => { /* no sample images on this copy of the site: only the built-in ones */ });
  }

  // ---- painting ----
  let last: [number, number] | null = null;
  const brushR = () => BRUSH * st.size / N;
  const toPix = (ev: PointerEvent): [number, number] => {
    const r = cv.getBoundingClientRect();
    return [(ev.clientX - r.left) / r.width * st.size, (ev.clientY - r.top) / r.height * st.size];
  };
  cv.addEventListener("pointerdown", ev => {
    ev.preventDefault();
    cv.setPointerCapture(ev.pointerId);
    last = toPix(ev);
    paintLine(st.seeds, ...last, ...last, brushR(), st.brush);
    unPreset();
    live();
  });
  cv.addEventListener("pointermove", ev => {
    if (!last) return;
    const q = toPix(ev);
    paintLine(st.seeds, ...last, ...q, brushR(), st.brush);
    last = q;
    live();
  });
  const up = () => { if (last) { last = null; run(); } };
  cv.addEventListener("pointerup", up);
  cv.addEventListener("pointercancel", up);
  /** While painting: recompute if updates are fast, else only show the seeds. */
  function live(): void {
    if (res && res.ms < LIVE_MS) run();
    else draw();
  }

  // ---- segmentation ----
  let res: Segmentation | null = null;
  let pending = false;
  function run(): void {
    syncButtons();
    if (pending || !solverSettled) return;
    pending = true;
    const slow = res !== null && res.ms > SLOW_MS;
    if (slow) status.textContent = "computing the minimum cut …";
    const go = () => {
      pending = false;
      if (slow) status.textContent = "";
      if (st.seeds.length !== I.length) return;   // a new image size is still loading
      res = seg.segment(I, st.seeds, st.settings);
      draw();
    };
    // a slow update waits one more frame, so the message is shown before the page is busy
    requestAnimationFrame(() => (slow ? setTimeout(go, 0) : go()));
  }

  function draw(): void {
    const ok = res !== null && res.ok && res.cut.L.length === I.length;
    paint(cv, I, st.show && ok ? res!.cut.L : null, st.seeds);
    paint(cvT, I, ok ? res!.thr : null, null);
    mainCap.textContent = `graph cut (λ = ${st.settings.lambda}): paint seeds here`;
    if (!res || !ok) { ro.textContent = res && !res.ok ? res.info : ""; note.textContent = "Paint at least one object and one background seed."; return; }
    const c = res.cut.energy, t = res.thrEnergy;
    const r = (nm: string, a: string, b: string) => nm.padEnd(10) + a.padStart(10) + b.padStart(10);
    const cnt = (L: Uint8Array) => L.reduce((a, b) => a + b, 0);
    const lines = [
      r("", "graph cut", "threshold"),
      r("data", f1(c.data), f1(t.data)),
      r("Σ w (cut)", f1(c.boundary), f1(t.boundary)),
      r("λ·Σ w", f1(c.smooth), f1(t.smooth)),
      r("E", f1(c.total), f1(t.total)),
      r("object px", String(cnt(res.cut.L)), String(cnt(res.thr))),
    ];
    if (st.image === "synth") {
      const truth = truthOf(st.size);
      const wrong = (L: Uint8Array) => L.reduce((a, l, p) => a + (l !== truth[p] ? 1 : 0), 0);
      lines.push(r("wrong px", String(wrong(res.cut.L)), String(wrong(res.thr))));
    }
    lines.push(`max flow = ${f1(res.cut.flow)}`,
      `${st.size}²: ${res.ms.toFixed(0)} ms (max flow ${res.solveMs.toFixed(0)} ms${wasmReady ? "" : ", no Wasm"})`);
    ro.textContent = lines.join("\n");
    note.textContent = `Intensity model from the seeds: ${res.info}. Red: object seeds, blue: background seeds; yellow line: the boundary of the graph-cut object.`;
  }
  const truths = new Map<number, Uint8Array>();
  function truthOf(n: number): Uint8Array {
    let t = truths.get(n);
    if (!t) { t = synthTruth(n); truths.set(n, t); }
    return t;
  }

  syncButtons();
  load(p0);
  wasmSolver().then(solver => { seg.solver = solver; wasmReady = true; },
    e => {
      // TypeScript fallback: fine at 128 x 128, too slow for the larger sizes
      console.warn("graphcut: WebAssembly max flow unavailable, using TypeScript:", e);
      sizeBtns.forEach((b, i) => { if (SIZES[i] > N) b.style.display = "none"; });
    }).then(() => { solverSettled = true; run(); });
}

autoMount(NAME, mount);
