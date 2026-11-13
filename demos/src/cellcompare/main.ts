// Networks and classical pipelines compared on the same annotated images (Networks for cells and
// objects day, unit cellnets-compare/). Seven public images of nuclei with ground truth (from
// `make models`, credits shown). The classical pipeline (smooth, threshold, fill holes, distance
// transform, h-maxima, watershed, minimum size) runs live in TypeScript; three buttons run the
// real networks with the same Wasm modules and weights as the stardistnet, cellposenet and samnet
// demos (their worker code is imported unchanged; this bundle is also the worker script, started
// with ?net=stardist|cellpose|sam). Every result is scored against the annotation with the
// evaluation day's matching: outlines and error marks (miss, extra, merge, split at tau = 0.5),
// F1(tau) curves of all methods on one plot, and a table with F1 at 0.5 and 0.7, TP/FP/FN,
// splits, merges, time and model size. Results are cached per image.
// Auto-mounts on <div data-demo="cellcompare">. Options: data-image (sparse|dim|dense|u2os|u2os2|
// histology|brightfield), data-preset (sparse|dense|histology|dim|brightfield).
import { htmlEl, slider, button, checkbox, presets, autoMount, type Slider } from "../lib/ui";
import { Plot } from "../lib/plot";
import { loadModelFile, loadStatus, modelIndex, isFileProtocol, fileProtocolMessage, formatBytes, distBase, type LoadStatus } from "../lib/wasm";
import * as C from "./compare";
import * as CP from "../cellposenet/pipeline";
import { decodePng, inflateBrowser } from "../cellposenet/png";
import { workerMain as cellposeWorkerMain, Pool, type MasksOut } from "../cellposenet/worker";
import { runWorker as stardistWorker } from "../stardistnet/worker";
import type { Req as SdReq, Res as SdRes } from "../stardistnet/protocol";
import { runWorker as samWorker } from "../samnet/worker";
import type { Req as SamReq, Res as SamRes } from "../samnet/protocol";
import * as SP from "../samnet/post";

const NAME = "cellcompare";

/** Parameters (millions), from the weight files (plan/cellnets-day.md); weights are fetched once. */
const NETS: Record<Exclude<C.MethodId, "classical">, { params: string; file: string }> = {
  stardist: { params: "1.4 M", file: "stardist/2D_versatile_fluo/weights_best.h5" },
  cellpose: { params: "6.6 M", file: "cellpose/nuclei.safetensors" },
  sam: { params: "39 M", file: "sam2/sam2.1_hiera_tiny.safetensors" },
};

const PRESETS: { id: string; label: string; image: string; invert: boolean; p?: Partial<C.ClassicalParams> }[] = [
  { id: "sparse", label: "sparse: classical holds up", image: "sparse", invert: false },
  { id: "dense", label: "dense: networks win?", image: "dense", invert: false },
  { id: "dim", label: "dim nuclei", image: "dim", invert: false },
  { id: "histology", label: "histology: everyone struggles", image: "histology", invert: false },
  { id: "brightfield", label: "bright field, inverted", image: "brightfield", invert: true },
];

type View = "truth" | C.MethodId | "all";

const STYLE = `
.qb-demo[data-demo="${NAME}"] { display: flex; flex-wrap: wrap; gap: .8em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .cc-left { flex: 1 1 480px; min-width: 260px; max-width: 600px; display: flex; flex-direction: column; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .cc-tabs { display: flex; flex-wrap: wrap; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .cc-stage { position: relative; width: 100%; background: #000; border-radius: 4px; overflow: hidden; }
.qb-demo[data-demo="${NAME}"] .cc-stage canvas { display: block; width: 100%; height: auto; }
.qb-demo[data-demo="${NAME}"] .cc-busy { position: absolute; left: 0; right: 0; bottom: 0; padding: .3em .6em; background: rgba(0,0,0,.65); color: #fff; font-size: 16px; }
.qb-demo[data-demo="${NAME}"] .cc-credit, .qb-demo[data-demo="${NAME}"] .cc-key { font-size: 13px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 380px; max-width: 560px; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .cc-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .cc-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
.qb-demo[data-demo="${NAME}"] select { font: inherit; font-size: .9em; padding: .1em .2em; max-width: 18em; }
.qb-demo[data-demo="${NAME}"] .qb-btn:disabled { opacity: .45; cursor: default; }
.qb-demo[data-demo="${NAME}"] .cc-tabwrap { width: 100%; overflow-x: auto; }
.qb-demo[data-demo="${NAME}"] table.cc-tab { border-collapse: collapse; font-size: 13px; width: 100%; }
.qb-demo[data-demo="${NAME}"] table.cc-tab th, .qb-demo[data-demo="${NAME}"] table.cc-tab td { padding: .1em .3em; text-align: right; border-bottom: 1px solid var(--qb-grid, #ddd); white-space: nowrap; }
.qb-demo[data-demo="${NAME}"] table.cc-tab th:first-child, .qb-demo[data-demo="${NAME}"] table.cc-tab td:first-child { text-align: left; }
.qb-demo[data-demo="${NAME}"] .cc-sw { display: inline-block; width: .8em; height: .8em; border-radius: 2px; margin-right: .3em; vertical-align: -.05em; }
.qb-demo[data-demo="${NAME}"] .cc-plot { width: 100%; max-width: 420px; }
@media (max-width: 640px) { .qb-demo[data-demo="${NAME}"] .cc-left { flex-basis: auto; width: 100%; max-width: none; } }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 16px; gap: .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .cc-left { max-width: 430px; gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .cc-tabs .qb-btn { padding: .1em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { gap: .2em; max-width: 600px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-slider { margin: 0; }
.reveal .qb-demo[data-demo="${NAME}"] .cc-plot { max-width: 400px; }
.reveal .qb-demo[data-demo="${NAME}"] table.cc-tab { font-size: 12px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface Loaded { s: C.Sample; w: number; h: number; grey: Float32Array; truth: Int32Array; credit: string }
interface Result { labels: Int32Array; score: C.Score; marks: C.Mark[]; ms: number; note: string }

const MARK_COL: Record<C.MarkKind, string> = { miss: "#ff3030", extra: "#ffd700", merge: "#ff40ff", split: "#40e0ff" };

function mount(host: HTMLElement): void {
  injectStyle();
  const left = htmlEl("div", "cc-left", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);

  if (isFileProtocol()) {
    htmlEl("div", "qb-readout", fileProtocolMessage("This comparison"), left);
    return;
  }

  // ---- left: view tabs, stage, key, credit
  const tabs = htmlEl("div", "cc-tabs", "", left);
  const VIEWS: { v: View; label: string }[] = [
    { v: "truth", label: "Truth" }, ...C.METHODS.map((m) => ({ v: m.id as View, label: m.label })), { v: "all", label: "All four" },
  ];
  const tabBtns = VIEWS.map((x) => { const b = button(tabs, x.label, () => { view = x.v; draw(); }); return { v: x.v, b }; });
  const stage = htmlEl("div", "cc-stage", "", left);
  const canvas = htmlEl("canvas", "", "", stage);
  const busy = htmlEl("div", "cc-busy", "", stage);
  busy.style.display = "none";
  const rowShow = htmlEl("div", "cc-row", "", left);
  const showTruth = checkbox(rowShow, "truth outlines (white)", false, () => draw());
  const showMarks = checkbox(rowShow, "error marks", true, () => draw());
  const key = htmlEl("div", "cc-key", "", left);
  key.innerHTML = `Marks at τ = 0.5: <b style="color:${MARK_COL.miss}">○</b> missed &nbsp; <b style="color:#b8a000">×</b> extra &nbsp; ` +
    `<b style="color:${MARK_COL.merge}">M</b> merge &nbsp; <b style="color:#00a0c0">S</b> split`;
  const credit = htmlEl("div", "cc-credit", "", left);

  // ---- panel: image, presets, classical knobs, networks, plot, table
  htmlEl("div", "cc-lab", "Image (annotated)", panel);
  const sel = htmlEl("select", "", "", panel);
  for (const s of C.SAMPLES) { const o = htmlEl("option", "", s.label, sel); o.value = s.id; }
  const invBox = checkbox(panel, "invert grey levels (dark nuclei)", false, () => { void reload(); });
  htmlEl("div", "cc-lab", "Presets", panel);
  const presetBtns = presets(panel, PRESETS.map((p) => ({ label: p.label, apply: () => applyPreset(p.id) })));

  htmlEl("div", "cc-lab", "Classical pipeline (runs live)", panel);
  const sSigma = slider(panel, "smoothing σ (px)", 0, 4, 0.5, C.CLASSICAL_DEFAULT.sigma, () => scheduleClassical());
  const rowThr = htmlEl("div", "cc-row", "", panel);
  const otsuBox = checkbox(rowThr, "Otsu threshold", true, () => scheduleClassical());
  const sThr = slider(panel, "manual threshold", 0.02, 0.9, 0.01, C.CLASSICAL_DEFAULT.thr, () => { otsuBox.checked = false; scheduleClassical(); });
  const sH = slider(panel, "h of the h-maxima (px)", 0, 8, 0.5, C.CLASSICAL_DEFAULT.h, () => scheduleClassical());
  const sMin = slider(panel, "minimum size (px)", 0, 200, 5, C.CLASSICAL_DEFAULT.minSize, () => scheduleClassical());

  htmlEl("div", "cc-lab", "Networks (same modules and weights as the network demos)", panel);
  const rowNet = htmlEl("div", "cc-row", "", panel);
  const btnSd = button(rowNet, "Run StarDist", () => { void runStardist(); });
  const btnCp = button(rowNet, "Run Cellpose (nuclei)", () => { void runCellpose(); });
  const btnSam = button(rowNet, "Run SAM automatic (~1 min)", () => { void runSam(); });
  const sDiam = slider(panel, "Cellpose diameter (px)", 6, 60, 1, 13, () => { /* used on the next run */ });
  const statusHost = htmlEl("div", "", "", panel);

  // On a slide (1280 x 720) the plot goes under the image, so the panel fits the height.
  const onSlide = !!host.closest(".reveal");
  const plotHost = htmlEl("div", "cc-plot", "", onSlide ? left : panel);
  const plot = new Plot(plotHost, { width: 440, height: onSlide ? 190 : 230, x: [0.5, 0.95], y: [0, 1], xlabel: "IoU threshold τ", ylabel: "F1", xticks: [0.5, 0.6, 0.7, 0.8, 0.9], yticks: [0, 0.25, 0.5, 0.75, 1], margin: { l: 70, b: 50, t: 10, r: 12 }, ariaLabel: "F1 against the IoU threshold for every method run on this image" });
  const tableHost = htmlEl("div", "cc-tabwrap", "", panel);
  if (onSlide) panel.appendChild(credit);

  // ---- state
  let view: View = "all";
  let cur: Loaded | null = null;
  let loadId = 0;
  const cache = new Map<string, Result>(); // key: image|inv|method(+diam)
  const resKey = (m: C.MethodId) => `${cur!.s.id}|${invBox.checked ? 1 : 0}|${m}${m === "cellpose" ? "|" + sDiam.get() : ""}`;
  const results = (): Partial<Record<C.MethodId, Result>> => {
    const r: Partial<Record<C.MethodId, Result>> = {};
    if (!cur) return r;
    for (const m of C.METHODS) { const x = cache.get(resKey(m.id)); if (x) r[m.id] = x; }
    return r;
  };
  let sizes: Record<string, number> = {};
  void modelIndex().then((ix) => { for (const k of Object.values(NETS)) sizes[k.file] = Number(ix[k.file]?.bytes ?? 0); renderTable(); }).catch(() => { sizes = {}; });

  function setBusy(msg: string): void { busy.textContent = msg; busy.style.display = msg ? "" : "none"; }

  function grey(): Float32Array { return invBox.checked ? C.invertGrey(cur!.grey) : cur!.grey; }
  let disp: { key: string; n: Float32Array } | null = null;
  function display(): Float32Array {
    const k = cur!.s.id + "|" + invBox.checked;
    if (!disp || disp.key !== k) disp = { key: k, n: C.normalise(grey()) };
    return disp.n;
  }

  function record(m: C.MethodId, lab: ArrayLike<number>, ms: number, note = ""): void {
    if (!cur) return;
    const labels = C.compact(lab);
    cache.set(resKey(m), { labels, score: C.score(cur.truth, labels), marks: C.errorMarks(cur.truth, labels, cur.w), ms, note });
  }

  // ---- image loading
  async function loadSample(id: string): Promise<Loaded> {
    const s = C.SAMPLES.find((x) => x.id === id) ?? C.SAMPLES[0];
    const im = await decodePng(await loadModelFile(s.file), inflateBrowser);
    const lp = await decodePng(await loadModelFile(s.labels), inflateBrowser);
    let cr = "";
    try { cr = String((await modelIndex())[s.file]?.credit ?? ""); } catch { /* no index */ }
    return { s, w: im.w, h: im.h, grey: C.greyFromPixels(im.data, im.w, im.h, im.channels, im.depth),
      truth: Int32Array.from(lp.data.filter((_, i) => i % lp.channels === 0)), credit: cr };
  }

  async function choose(id: string): Promise<void> {
    const my = ++loadId;
    setBusy("Loading image …");
    try {
      const L = await loadSample(id);
      if (my !== loadId) return;
      cur = L;
      sel.value = L.s.id;
      sDiam.set(L.s.diameter);
      credit.textContent = `${L.s.label}: ${C.compact(L.truth).reduce((a, v) => Math.max(a, v), 0)} annotated nuclei. ${L.credit}`;
      canvas.width = L.w; canvas.height = L.h;
      await reload();
    } catch (e) {
      setBusy("");
      loadStatus(statusHost, "").error("Could not load the image: " + ((e as Error).message ?? e) + " (the images come with the course web site).");
    }
  }

  async function reload(): Promise<void> {
    if (!cur) return;
    runClassical();
    setBusy("");
    draw();
  }

  // ---- classical
  let clTimer = 0;
  function scheduleClassical(): void { clearTimeout(clTimer); clTimer = window.setTimeout(() => { runClassical(); draw(); }, 120); }
  function runClassical(): void {
    if (!cur) return;
    const p: C.ClassicalParams = { sigma: sSigma.get(), method: otsuBox.checked ? "otsu" : "manual", thr: sThr.get(), h: sH.get(), minSize: sMin.get() };
    const r = C.classical(grey(), cur.w, cur.h, p);
    if (p.method === "otsu") sThr.set(Math.round(r.thr * 100) / 100);
    record("classical", r.labels, r.ms, `threshold ${r.thr.toFixed(2)}`);
  }

  // ---- networks
  const base = distBase();
  const workerUrl = (net: string) => `${base}${NAME}.js?net=${net}`;
  let sdWorker: Worker | null = null, sdReady: Promise<void> | null = null;
  let sdThr = { prob: 0.5, nms: 0.4 };
  let samW: Worker | null = null, samReady: Promise<void> | null = null;
  let pool: Pool | null = null, cpReady: Promise<void> | null = null;
  const pendingSd = new Map<number, (m: SdRes) => void>();
  const pendingSam = new Map<number, (m: SamRes) => void>();
  let reqId = 1;
  let running = false;

  function lock(on: boolean): void { running = on; [btnSd, btnCp, btnSam].forEach((b) => { b.disabled = on; }); }

  function startSd(st: LoadStatus): Promise<void> {
    if (sdReady) return sdReady;
    sdReady = new Promise<void>((resolve, reject) => {
      sdWorker = new Worker(workerUrl("stardist"));
      sdWorker.onerror = (e) => reject(new Error(e.message || "worker failed"));
      sdWorker.onmessage = (e: MessageEvent<SdRes>) => {
        const m = e.data;
        if (m.type === "progress") st.progress(m.loaded, m.total);
        else if (m.type === "ready") { sdThr = { prob: m.probThresh, nms: m.nmsThresh }; resolve(); }
        else if (m.type === "error") { const f = pendingSd.get(m.id); if (f) f(m); else reject(new Error(m.message)); }
        else { const f = pendingSd.get(m.id); if (f) { pendingSd.delete(m.id); f(m); } }
      };
      sdWorker.postMessage({ type: "init", base, engine: 1 } as SdReq);
    });
    sdReady.catch(() => { sdReady = null; });
    return sdReady;
  }
  const sdCall = (req: SdReq & { id: number }, transfer: Transferable[] = []) => new Promise<SdRes>((resolve, reject) => {
    pendingSd.set(req.id, (m) => (m.type === "error" ? reject(new Error(m.message)) : resolve(m)));
    sdWorker!.postMessage(req, transfer);
  });

  async function runStardist(): Promise<void> {
    if (!cur || running) return;
    const L = cur; const inv = invBox.checked;
    lock(true);
    const st = loadStatus(statusHost, "Loading StarDist");
    try {
      await startSd(st);
      st.done();
      setBusy("StarDist: network …");
      const t0 = performance.now();
      const g = (inv ? C.invertGrey(L.grey) : L.grey).slice();
      const id = reqId++;
      const p = await sdCall({ type: "predict", id, grey: g.buffer, w: L.w, h: L.h, pmin: 1, pmax: 99.8 }, [g.buffer]);
      if (p.type !== "predicted") throw new Error("unexpected reply");
      setBusy("StarDist: decoding (polygon NMS) …");
      const d = await sdCall({ type: "decode", id: reqId++, probThresh: sdThr.prob, nmsThresh: sdThr.nms });
      if (d.type !== "decoded") throw new Error("unexpected reply");
      if (cur === L && invBox.checked === inv) { record("stardist", new Uint32Array(d.labels), performance.now() - t0, `${d.candidates} candidates`); view = view === "truth" ? "all" : view; }
    } catch (e) { st.error("StarDist: " + ((e as Error).message ?? e)); }
    setBusy(""); lock(false); draw();
  }

  function startCp(st: LoadStatus): Promise<void> {
    if (cpReady) return cpReady;
    cpReady = (async () => {
      const weights = await loadModelFile(NETS.cellpose.file, { onProgress: st.progress });
      const n = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1));
      pool = new Pool(workerUrl("cellpose"), n);
      st.text("Starting Cellpose workers …");
      await pool.broadcast({ type: "init", weights });
    })();
    cpReady.catch(() => { cpReady = null; pool?.terminate(); pool = null; });
    return cpReady;
  }

  async function runCellpose(): Promise<void> {
    if (!cur || running) return;
    const L = cur; const inv = invBox.checked; const diam = sDiam.get();
    lock(true);
    const st = loadStatus(statusHost, "Loading Cellpose (nuclei)");
    try {
      await startCp(st);
      st.done();
      const t0 = performance.now();
      const g = inv ? C.invertGrey(L.grey) : L.grey;
      const ch: [Float32Array, Float32Array] = [CP.normalize99(g), new Float32Array(L.w * L.h)];
      const rescale = CP.MODELS.nuclei.diamMean / diam;
      const net = await CP.runNetwork(ch, L.w, L.h, rescale, (x, th, tw) => pool!.call<{ y: Float32Array; style: Float32Array }>({ type: "tile", input: x, h: th, w: tw }, [x.buffer]),
        (done, total) => setBusy(`Cellpose: tile ${done} / ${total} (${pool!.size} workers)`));
      setBusy("Cellpose: following the flows …");
      const out = await pool!.call<MasksOut>({ type: "masks", dy: net.dy, dx: net.dx, cellprob: net.cellprob, h: L.h, w: L.w,
        niter: CP.defaultNiter(rescale), cellprobThreshold: 0, flowThreshold: 0.4, minSize: 15 });
      if (cur === L && invBox.checked === inv && sDiam.get() === diam) record("cellpose", out.masks, performance.now() - t0, `diameter ${diam} px`);
    } catch (e) { st.error("Cellpose: " + ((e as Error).message ?? e)); }
    setBusy(""); lock(false); draw();
  }

  function startSam(st: LoadStatus): Promise<void> {
    if (samReady) return samReady;
    samReady = new Promise<void>((resolve, reject) => {
      samW = new Worker(workerUrl("sam"));
      samW.onerror = (e) => reject(new Error(e.message || "worker failed"));
      samW.onmessage = (e: MessageEvent<SamRes>) => {
        const m = e.data;
        if (m.type === "progress") st.progress(m.loaded, m.total);
        else if (m.type === "ready") resolve();
        else if (m.type === "error" && !pendingSam.has(m.id)) reject(new Error(m.message));
        else if ("id" in m) pendingSam.get(m.id)?.(m);
      };
      samW.postMessage({ type: "init", base } as SamReq);
    });
    samReady.catch(() => { samReady = null; });
    return samReady;
  }

  async function runSam(): Promise<void> {
    if (!cur || running) return;
    const L = cur; const inv = invBox.checked;
    lock(true);
    const st = loadStatus(statusHost, "Loading SAM (78 MB, once)");
    try {
      await startSam(st);
      st.done();
      const t0 = performance.now();
      setBusy("SAM: image encoder (tens of seconds) …");
      const rgb = C.rgbFromGrey(inv ? C.invertGrey(L.grey) : L.grey);
      const encId = reqId++, amgId = reqId++;
      const cands: SP.Candidate[] = [];
      await new Promise<void>((resolve, reject) => {
        pendingSam.set(encId, (m) => {
          if (m.type === "error") { reject(new Error(m.message)); return; }
          pendingSam.delete(encId);
          setBusy("SAM: 8 × 8 grid of point prompts …");
          samW!.postMessage({ type: "amg", id: amgId, pointsPerSide: 8, batch: 4, stabilityOffset: 1 } as SamReq);
        });
        pendingSam.set(amgId, (m) => {
          if (m.type === "error") { reject(new Error(m.message)); return; }
          if (m.type === "amgBatch") {
            for (const c of m.candidates) cands.push({ point: c.point, output: c.output, px: c.px, py: c.py, iou: c.iou, stability: c.stability,
              box: c.box, area: c.area, mask: c.mask ? new Uint8Array(c.mask) : null });
            setBusy(`SAM: prompt ${m.done} / ${m.total}`);
          } else if (m.type === "amgDone" || m.type === "amgCancelled") { pendingSam.delete(amgId); resolve(); }
        });
        samW!.postMessage({ type: "encode", id: encId, rgb: rgb.buffer, w: L.w, h: L.h } as SamReq, [rgb.buffer]);
      });
      const { kept } = SP.filterCandidates(cands, SP.AMG_DEFAULT);
      if (cur === L && invBox.checked === inv) record("sam", SP.labelImage(kept, L.w, L.h), performance.now() - t0, `${kept.length} masks kept of ${cands.length}`);
    } catch (e) { st.error("SAM: " + ((e as Error).message ?? e)); }
    setBusy(""); lock(false); draw();
  }

  // ---- drawing
  function drawPanel(ctx: CanvasRenderingContext2D, ox: number, oy: number, scale: number, lab: Int32Array | null, colour: string, marks: C.Mark[] | null, title: string): void {
    const L = cur!, w = L.w, h = L.h;
    const n = display();
    const img = new ImageData(w, h);
    const tr = showTruth.checked || title.startsWith("Truth") ? C.outlinePixels(L.truth, w, h) : null;
    const ol = lab ? C.outlinePixels(lab, w, h) : null;
    const [cr, cg, cb] = [1, 3, 5].map((k) => parseInt(colour.slice(k, k + 2), 16));
    for (let i = 0; i < w * h; i++) {
      const v = Math.max(0, Math.min(255, 255 * n[i]));
      let r = v * 0.8, gg = v * 0.8, b = v * 0.8;
      if (tr && tr[i]) { r = gg = b = 255; }
      if (ol && ol[i]) { r = cr; gg = cg; b = cb; }
      img.data[4 * i] = r; img.data[4 * i + 1] = gg; img.data[4 * i + 2] = b; img.data[4 * i + 3] = 255;
    }
    const tmp = document.createElement("canvas");
    tmp.width = w; tmp.height = h;
    tmp.getContext("2d")!.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = scale < 1;
    ctx.lineJoin = "round";
    ctx.drawImage(tmp, ox, oy, w * scale, h * scale);
    const fs = Math.max(12, Math.round(ctx.canvas.width / 42));
    if (marks && showMarks.checked) {
      ctx.lineWidth = Math.max(2, fs / 7);
      for (const m of marks) {
        const x = ox + m.x * scale, y = oy + m.y * scale, r = fs * 0.45;
        ctx.strokeStyle = MARK_COL[m.kind]; ctx.fillStyle = MARK_COL[m.kind];
        if (m.kind === "miss") { ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.stroke(); }
        else if (m.kind === "extra") { ctx.beginPath(); ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r); ctx.stroke(); }
        else { ctx.font = `bold ${fs}px sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(m.kind === "merge" ? "M" : "S", x, y); }
      }
    }
    ctx.font = `bold ${fs}px sans-serif`; ctx.textAlign = "left"; ctx.textBaseline = "top";
    const tw = ctx.measureText(title).width;
    ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(ox, oy, tw + fs * 0.6, fs * 1.4);
    ctx.fillStyle = colour; ctx.fillText(title, ox + fs * 0.3, oy + fs * 0.2);
  }

  function draw(): void {
    tabBtns.forEach((t) => t.b.classList.toggle("active", t.v === view));
    if (!cur) return;
    const L = cur, R = results();
    const ctx = canvas.getContext("2d")!;
    // draw at least ~900 px wide, so marks and labels stay readable on small images
    const k = Math.max(1, Math.ceil(900 / L.w));
    canvas.width = L.w * k; canvas.height = L.h * k;
    const W = canvas.width, H = canvas.height;
    if (view === "all") {
      const sc = k / 2;
      ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
      C.METHODS.forEach((m, j) => {
        const r = R[m.id];
        const ox = (j % 2) * L.w * sc, oy = Math.floor(j / 2) * L.h * sc;
        drawPanel(ctx, ox, oy, sc, r?.labels ?? null, m.colour, r?.marks ?? null, r ? `${m.label}  F1 ${r.score.f1[0].toFixed(2)}` : `${m.label}: not run`);
      });
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
    } else {
      if (view === "truth") drawPanel(ctx, 0, 0, k, null, "#ffffff", null, `Truth: ${C.compact(L.truth).reduce((a, v) => Math.max(a, v), 0)} nuclei`);
      else {
        const m = C.METHODS.find((x) => x.id === view)!, r = R[m.id];
        drawPanel(ctx, 0, 0, k, r?.labels ?? null, m.colour, r?.marks ?? null, r ? `${m.label}  F1 ${r.score.f1[0].toFixed(2)}` : `${m.label}: not run yet`);
      }
    }
    drawPlot(R);
    renderTable();
  }

  function drawPlot(R: Partial<Record<C.MethodId, Result>>): void {
    plot.clear();
    for (const m of C.METHODS) {
      const r = R[m.id];
      if (!r) continue;
      plot.line(C.TAUS, r.score.f1, { color: m.colour, width: 3 });
      plot.scatter(C.TAUS, r.score.f1, { color: m.colour, r: 4 });
    }
  }

  function renderTable(): void {
    if (!cur) return;
    const R = results();
    const rows = C.METHODS.map((m) => {
      const r = R[m.id];
      const sw = `<span class="cc-sw" style="background:${m.colour}"></span>`;
      if (!r) return `<tr><td>${sw}${m.label}</td><td colspan="6" style="text-align:left;color:var(--qb-muted)">not run</td></tr>`;
      const s = r.score;
      const t = r.ms < 1000 ? `${Math.round(r.ms)} ms` : `${(r.ms / 1000).toFixed(1)} s`;
      return `<tr title="${r.note}"><td>${sw}${m.label}</td><td>${s.nPred}</td><td><b>${s.f1[0].toFixed(2)}</b></td><td>${s.f1[4].toFixed(2)}</td>` +
        `<td>${s.tp5}/${s.fp5}/${s.fn5}</td><td>${s.splits}/${s.merges}</td><td>${t}</td></tr>`;
    }).join("");
    const nT = C.compact(cur.truth).reduce((a, v) => Math.max(a, v), 0);
    tableHost.innerHTML = `<table class="cc-tab"><tr><th>method (truth: ${nT})</th><th>objects</th><th>F1 .5</th><th>F1 .7</th><th>TP/FP/FN</th><th>split/ merge</th><th>time</th></tr>${rows}</table>` +
      `<div class="cc-key">Model files and parameters: ${C.METHODS.filter((m) => m.id !== "classical").map((m) => { const n = NETS[m.id as Exclude<C.MethodId, "classical">]; return `${m.label} ${sizes[n.file] ? formatBytes(sizes[n.file]) + ", " : ""}${n.params}`; }).join(" · ")} · classical: 5 knobs, no training. Times: this browser.</div>`;
  }

  function applyPreset(id: string): void {
    const p = PRESETS.find((x) => x.id === id);
    if (!p) return;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i].id === id));
    sSigma.set(C.CLASSICAL_DEFAULT.sigma); sH.set(C.CLASSICAL_DEFAULT.h); sMin.set(C.CLASSICAL_DEFAULT.minSize); otsuBox.checked = true;
    invBox.checked = p.invert;
    view = "all";
    void choose(p.image);
  }

  sel.addEventListener("change", () => { sel.blur(); invBox.checked = false; void choose(sel.value); });

  const want = host.dataset.preset && PRESETS.some((p) => p.id === host.dataset.preset) ? host.dataset.preset : "";
  if (want) applyPreset(want);
  else void choose(host.dataset.image && C.SAMPLES.some((s) => s.id === host.dataset.image) ? host.dataset.image : "sparse");
}

// The same bundle is the worker script of each network: no document there.
if (typeof document === "undefined") {
  const net = new URLSearchParams((self as unknown as { location: { search: string } }).location.search).get("net");
  if (net === "stardist") stardistWorker();
  else if (net === "cellpose") cellposeWorkerMain();
  else if (net === "sam") samWorker();
} else {
  autoMount(NAME, mount);
}
