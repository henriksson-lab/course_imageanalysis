// The JPEG pipeline, live. Original | decoded | difference (or SSIM map) of a grey or colour
// image at a quality factor 1..100 (IJG scaling of the standard tables), with 4:4:4 or 4:2:0
// chroma, the standard or a flat quantisation table (the flat step chosen to give the same
// size), and repeated recompression (generations, optionally with the 8 × 8 grid moved by 4
// pixels each time). Below, in tabs: one block through the pipeline (DCT coefficients,
// quantised values, zigzag sequence, (run, size) symbols and the Huffman bit string), the
// quantisation table, and the rate–distortion curves (PSNR and SSIM against bits per pixel
// over all qualities). Size = the exact number of Huffman-coded bits with the standard tables
// + a header estimate; the browser's own JPEG encoder (canvas.toBlob) is shown as a cross-check.
// Auto-mounts on <div data-demo="jpeg">. Options: data-preset="q90|q10|ringing|bleeding|flat|
// recompress", data-image="cell|card|shapes|colour", data-quality="50", data-tab="block|table|rd",
// data-zoom="1".
import { htmlEl, svgEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { jpegGrey, jpegColour, flatTable, tableForQuality, LUMA_Q, CHROMA_Q, blockSymbols, blockBitString, toZigzag, ChannelResult } from "../lib/jpeg";
import { ssim, psnr, mse } from "../lib/metrics";
import { Plot } from "../lib/plot";
import { diverging, divergingT, toCss, textColorOn } from "../lib/colour";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { card, shapes, colourCard, luma, SIZE } from "./testimages";

const NAME = "jpeg";
const N = SIZE;
type ImgId = "cell" | "card" | "shapes" | "colour";
type Tab = "block" | "table" | "rd";
const IMGS: { id: ImgId; label: string; credit: string }[] = [
  { id: "cell", label: "cell", credit: CELL_CREDIT },
  { id: "card", label: "test card", credit: "Synthetic test card: gradient, flat areas, an edge, stripes, thin lines." },
  { id: "shapes", label: "shapes", credit: "Synthetic: flat shapes with straight and diagonal edges." },
  { id: "colour", label: "colour card", credit: "Synthetic colour card: a red | blue edge, a green disc, a gradient, dark strokes." },
];

interface State {
  img: ImgId; q: number; sub: boolean; flat: boolean; gens: number; shift: boolean;
  view: "diff" | "ssim"; zoom: boolean; bx: number; by: number; tab: Tab;
}
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "q90", label: "quality 90: invisible", s: { img: "cell", q: 90, zoom: false, bx: 14, by: 12 } },
  { id: "q10", label: "quality 10: blocking", s: { img: "cell", q: 10, zoom: true, bx: 27, by: 4 } },
  { id: "ringing", label: "ringing at an edge", s: { img: "card", q: 25, zoom: true, bx: 16, by: 11 } },
  { id: "bleeding", label: "colour bleeding (4:2:0)", s: { img: "colour", q: 75, sub: true, zoom: true, bx: 16, by: 22 } },
  { id: "flat", label: "flat table, same size", s: { img: "cell", q: 50, flat: true, zoom: true, bx: 14, by: 12 } },
  { id: "recompress", label: "recompress 10×", s: { img: "cell", q: 75, gens: 10, shift: true, zoom: true, bx: 14, by: 12 } },
];
const DEFAULTS: State = { img: "cell", q: 50, sub: false, flat: false, gens: 1, shift: false, view: "diff", zoom: false, bx: 14, by: 12, tab: "block" };

const STYLE = `
.qb-demo[data-demo="${NAME}"] .jp-main { display: flex; flex-direction: column; gap: .4em; flex: 1 1 700px; min-width: 0; max-width: 860px; }
.qb-demo[data-demo="${NAME}"] .jp-imgs { display: flex; gap: 10px; }
.qb-demo[data-demo="${NAME}"] .jp-imgs figure { margin: 0; flex: 1 1 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; line-height: 1.2; margin-bottom: .15em; }
.qb-demo[data-demo="${NAME}"] canvas.img { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; cursor: crosshair; touch-action: none; }
.qb-demo[data-demo="${NAME}"] .jp-tabs { display: flex; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .jp-tabbody { min-height: 220px; }
.qb-demo[data-demo="${NAME}"] .jp-block { display: flex; gap: 10px; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .jp-block figure { margin: 0; flex: 0 0 190px; }
.qb-demo[data-demo="${NAME}"] .jp-sym { flex: 1 1 0; min-width: 0; font-family: ui-monospace, monospace; font-size: 14px; line-height: 1.4; word-break: break-all; }
.qb-demo[data-demo="${NAME}"] .jp-sym { max-height: min(250px, calc(100vh - 500px)); min-height: 150px; overflow-y: auto; scrollbar-width: thin; }
.qb-demo[data-demo="${NAME}"] .jp-imgs figure { max-width: min(280px, calc(100vh - 440px)); }
.qb-demo[data-demo="${NAME}"] .jp-block figure { flex-basis: min(190px, calc(100vh - 530px)); min-width: 120px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-btn { font-size: 14px; padding: .15em .5em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-readout { font-size: 13px; padding: .3em .5em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-note { font-size: 13px; }
.qb-demo[data-demo="${NAME}"] .jp-sym b { font-family: system-ui, sans-serif; }
.qb-demo[data-demo="${NAME}"] .jp-rd { display: flex; gap: 10px; }
.qb-demo[data-demo="${NAME}"] .jp-rd figure { margin: 0; flex: 1 1 0; }
.qb-demo.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 230px; max-width: none; gap: .4em; }
.reveal .qb-demo.qb-demo[data-demo="${NAME}"] .qb-panel { max-width: 270px; }
.qb-demo[data-demo="${NAME}"] .jp-btns { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
.qb-demo[data-demo="${NAME}"] .qb-btn.on { background: var(--qb-accent); color: #fff; border-color: var(--qb-accent); }
.qb-demo[data-demo="${NAME}"] .qb-check.off, .qb-demo[data-demo="${NAME}"] .qb-btn.off { opacity: .35; pointer-events: none; }
.qb-demo[data-demo="${NAME}"] .jp-credit { font-size: 13px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] svg.grid text { font-family: ui-monospace, monospace; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; padding: .3em .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { font-size: 13px; }
.reveal .qb-demo[data-demo="${NAME}"] .jp-main { max-width: 820px; }
.reveal .qb-demo[data-demo="${NAME}"] .jp-imgs figure { max-width: 250px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .jp-sym { max-height: 230px; }
.reveal .qb-demo[data-demo="${NAME}"] .jp-block figure { flex-basis: 170px; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 17px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

/** Shift an image by (dx, dy) with edge replication. */
function shiftImg(a: ArrayLike<number>, dx: number, dy: number): Float64Array {
  const out = new Float64Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
    out[y * N + x] = a[Math.min(N - 1, Math.max(0, y - dy)) * N + Math.min(N - 1, Math.max(0, x - dx))];
  return out;
}

interface Coded { planes: Float64Array[]; y: ChannelResult; bytes: number; bpp: number; nonzero: number; Ql: Int32Array; Qc: Int32Array | null; flatStep: number }

/** Grey or colour JPEG with the state's options (one generation). */
function codeOnce(planes: Float64Array[], st: State, table?: Int32Array): Coded {
  if (planes.length === 1) {
    const r = jpegGrey(planes[0], N, N, table ? { table } : { quality: st.q });
    return { planes: [r.decoded], y: r, bytes: r.bytes, bpp: r.bpp, nonzero: r.nonzero, Ql: r.Q, Qc: null, flatStep: table ? table[0] : 0 };
  }
  const r = jpegColour(planes[0], planes[1], planes[2], N, N, { quality: st.q, subsample: st.sub });
  return { planes: r.rgb, y: r.y, bytes: r.bytes, bpp: r.bpp, nonzero: r.nonzero, Ql: tableForQuality(LUMA_Q, st.q), Qc: tableForQuality(CHROMA_Q, st.q), flatStep: 0 };
}

/** The flat table whose size is closest to the standard table's at this quality (grey only). */
function flatForSameSize(img: Float64Array, q: number): Int32Array {
  const target = jpegGrey(img, N, N, { quality: q }).bytes;
  let lo = 1, hi = 255;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (jpegGrey(img, N, N, { table: flatTable(mid) }).bytes > target) lo = mid; else hi = mid;
  }
  const bl = jpegGrey(img, N, N, { table: flatTable(lo) }).bytes, bh = jpegGrey(img, N, N, { table: flatTable(hi) }).bytes;
  return flatTable(Math.abs(bl - target) <= Math.abs(bh - target) ? lo : hi);
}

/** The full pipeline with generations. */
function pipeline(planes: Float64Array[], st: State): Coded {
  const table = planes.length === 1 && st.flat ? flatForSameSize(planes[0], st.q) : undefined;
  let cur = planes, out = codeOnce(cur, st, table);
  for (let g = 1; g < st.gens; g++) {
    const s = st.shift ? (g % 2 ? 4 : 0) : 0;
    cur = out.planes.map(p => (s ? shiftImg(p, s, s) : p));
    out = codeOnce(cur, st, table);
    if (s) out = { ...out, planes: out.planes.map(p => shiftImg(p, -s, -s)) };
  }
  return out;
}

const lumaOf = (p: Float64Array[]) => (p.length === 1 ? p[0] : luma(p[0], p[1], p[2]));
const psnrOf = (a: Float64Array[], b: Float64Array[]) => {
  let m = 0;
  for (let k = 0; k < a.length; k++) m += mse(a[k], b[k]);
  m /= a.length;
  return m === 0 ? Infinity : 10 * Math.log10(255 * 255 / m);
};

function putPlanes(c: HTMLCanvasElement, planes: Float64Array[], zoom: { x0: number; y0: number; s: number } | null): void {
  const s = zoom ? zoom.s : N;
  c.width = s; c.height = s;
  const ctx = c.getContext("2d")!, id = ctx.createImageData(s, s);
  for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) {
    const i = (zoom ? (zoom.y0 + y) * N + zoom.x0 + x : y * N + x), p = 4 * (y * s + x);
    if (planes.length === 1) id.data[p] = id.data[p + 1] = id.data[p + 2] = planes[0][i];
    else { id.data[p] = planes[0][i]; id.data[p + 1] = planes[1][i]; id.data[p + 2] = planes[2][i]; }
    id.data[p + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

function drawGrid(svg: SVGSVGElement, vals: ArrayLike<number>, maxAbs: number, fmt: (v: number) => string, faintZero: boolean): void {
  svg.replaceChildren();
  const c = 24;
  for (let i = 0; i < 64; i++) {
    const x = (i % 8) * c, y = Math.floor(i / 8) * c, v = vals[i];
    const col = faintZero && v === 0 ? null : diverging(divergingT(Math.max(-maxAbs, Math.min(maxAbs, v)), maxAbs));
    svgEl("rect", { x, y, width: c, height: c, fill: col ? toCss(col) : "#eee", stroke: "var(--qb-bg)" }, svg);
    const s = fmt(v);
    const t = svgEl("text", { x: x + c / 2, y: y + c / 2 + 4, "text-anchor": "middle", "font-size": s.length > 3 ? 9.5 : 11.5 }, svg);
    t.textContent = s; t.style.fill = col ? textColorOn(col) : "#aaa";
  }
}

function mount(host: HTMLElement): void {
  injectStyle();
  const d = host.dataset;
  const pre = PRESETS.find(p => p.id === d.preset);
  const init: State = { ...DEFAULTS, ...(pre ? pre.s : {}) };
  if (d.image && IMGS.some(i => i.id === d.image)) init.img = d.image as ImgId;
  if (d.quality) init.q = Number(d.quality);
  if (d.tab === "table" || d.tab === "rd" || d.tab === "block") init.tab = d.tab;
  if (d.zoom) init.zoom = d.zoom === "1";
  const st: State = { ...init };

  const main = htmlEl("div", "jp-main", "", host);
  const row = htmlEl("div", "jp-imgs", "", main);
  const fig = (p: Element, t: string) => { const f = htmlEl("figure", "", "", p); const cap = htmlEl("figcaption", "", t, f); return { f, cap }; };
  const fo = fig(row, "original"), fd = fig(row, "decoded"), fe = fig(row, "difference");
  const cOrig = htmlEl("canvas", "img", "", fo.f), cDec = htmlEl("canvas", "img", "", fd.f), cErr = htmlEl("canvas", "img", "", fe.f);
  const tabRow = htmlEl("div", "jp-tabs", "", main);
  const tabBody = htmlEl("div", "jp-tabbody", "", main);
  const credit = htmlEl("div", "jp-credit", "", main);
  // tab: block
  const blockDiv = htmlEl("div", "jp-block", "", tabBody);
  const gfig = (t: string) => { const f = htmlEl("figure", "", "", blockDiv); htmlEl("figcaption", "", t, f); return svgEl("svg", { viewBox: "0 0 192 192", width: "100%", class: "grid" }, f); };
  const gCoef = gfig("DCT coefficients c");
  const gQuant = gfig("quantised q = round(c/Q)");
  const symDiv = htmlEl("div", "jp-sym", "", blockDiv);
  // tab: table
  const tableDiv = htmlEl("div", "jp-block", "", tabBody);
  const tfig = (t: string) => { const f = htmlEl("figure", "", "", tableDiv); htmlEl("figcaption", "", t, f); return svgEl("svg", { viewBox: "0 0 192 192", width: "100%", class: "grid" }, f); };
  const gQl = tfig("table Q (luma)");
  const gQc = tfig("table Q (chroma)");
  const tableNote = htmlEl("div", "jp-sym", "", tableDiv);
  // tab: rate-distortion
  const rdDiv = htmlEl("div", "jp-rd", "", tabBody);
  const rdFig = (t: string) => { const f = htmlEl("figure", "", "", rdDiv); htmlEl("figcaption", "", t, f); return f; };
  const pPsnr = new Plot(rdFig("PSNR against size"), { width: 400, height: 220, x: [0, 4], y: [20, 50], xlabel: "bits per pixel", ylabel: "PSNR (dB)", font: 16, margin: { l: 62, b: 48, r: 10, t: 8 } });
  const pSsim = new Plot(rdFig("SSIM against size"), { width: 400, height: 220, x: [0, 4], y: [0.5, 1], xlabel: "bits per pixel", ylabel: "SSIM", font: 16, margin: { l: 62, b: 48, r: 10, t: 8 } });
  const tabs: [Tab, string, HTMLElement][] = [["block", "one block", blockDiv], ["table", "quantisation table", tableDiv], ["rd", "rate–distortion", rdDiv]];
  const tabBtns = tabs.map(([id, label]) => button(tabRow, label, () => { st.tab = id; syncUI(); draw(); }));

  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULTS, { tab: st.tab }, p.s); syncUI(); load(); } })));
  pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === pre));
  const imgRow = htmlEl("div", "jp-btns", "", panel);
  const imgBtns = IMGS.map(im => button(imgRow, im.label, () => { st.img = im.id; if (im.id === "colour") st.flat = false; syncUI(); load(); }));
  const qS = slider(panel, "quality", 1, 100, 1, st.q, v => { st.q = v; schedule(); });
  const subBox = checkbox(panel, "chroma 4:2:0", st.sub, on => { st.sub = on; rdCache.clear(); schedule(); });
  const flatBox = checkbox(panel, "flat table (same size)", st.flat, on => { st.flat = on; schedule(); });
  const gS = slider(panel, "generations", 1, 10, 1, st.gens, v => { st.gens = v; schedule(); });
  const shiftBox = checkbox(panel, "move grid 4 px each time", st.shift, on => { st.shift = on; schedule(); });
  const vRow = htmlEl("div", "jp-btns", "", panel);
  const viewBtns = [button(vRow, "difference", () => { st.view = "diff"; syncUI(); draw(); }), button(vRow, "SSIM map", () => { st.view = "ssim"; syncUI(); draw(); })];
  const zoomBox = checkbox(vRow, "zoom ×4", st.zoom, on => { st.zoom = on; draw(); });
  button(panel, "reset", () => { Object.assign(st, init); syncUI(); load(); pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === pre)); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Click an image to choose the block (yellow). Difference: ×4, mid grey = 0. SSIM map: white = 1. Size = Huffman-coded bits + header.", panel);

  function syncUI(): void {
    imgBtns.forEach((b, i) => b.classList.toggle("on", IMGS[i].id === st.img));
    tabBtns.forEach((b, i) => b.classList.toggle("on", tabs[i][0] === st.tab));
    tabs.forEach(([id, , el]) => { el.style.display = id === st.tab ? "flex" : "none"; });
    viewBtns.forEach((b, i) => b.classList.toggle("on", (i === 0) === (st.view === "diff")));
    qS.set(st.q); gS.set(st.gens);
    subBox.checked = st.sub; flatBox.checked = st.flat; shiftBox.checked = st.shift; zoomBox.checked = st.zoom;
    subBox.parentElement!.classList.toggle("off", st.img !== "colour");
    flatBox.parentElement!.classList.toggle("off", st.img === "colour");
    shiftBox.parentElement!.classList.toggle("off", st.gens < 2);
  }

  // ---- data ----
  let planes: Float64Array[] = [new Float64Array(N * N)];
  let cellImg: Float64Array | null = null;
  const rdCache = new Map<string, { bpp: number[]; ps: number[]; ss: number[] }>();
  function load(): void {
    credit.textContent = IMGS.find(i => i.id === st.img)!.credit;
    if (st.img === "cell") {
      if (!cellImg) { loadCell().then(g => { cellImg = Float64Array.from(g.data, v => Math.round(v)); load(); }); return; }
      planes = [cellImg];
    } else if (st.img === "card") planes = [card()];
    else if (st.img === "shapes") planes = [shapes()];
    else planes = colourCard();
    schedule();
  }

  let pending = false;
  function schedule(): void {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; draw(); });
  }

  let blobVersion = 0;
  function draw(): void {
    const res = pipeline(planes, st);
    const zoom = st.zoom ? { x0: Math.max(0, Math.min(N - 64, st.bx * 8 + 4 - 32)), y0: Math.max(0, Math.min(N - 64, st.by * 8 + 4 - 32)), s: 64 } : null;
    putPlanes(cOrig, planes, zoom);
    putPlanes(cDec, res.planes, zoom);
    const yo = lumaOf(planes), yd = lumaOf(res.planes);
    const S = ssim(yo, yd, N, N, { peak: 255 });
    // third panel
    const e = new Float64Array(N * N);
    if (st.view === "diff") for (let i = 0; i < N * N; i++) e[i] = Math.max(0, Math.min(255, 128 + 4 * (yd[i] - yo[i])));
    else for (let i = 0; i < N * N; i++) e[i] = Math.max(0, Math.min(255, 255 * S.map.data[i]));
    putPlanes(cErr, [e], zoom);
    fe.cap.textContent = st.view === "diff" ? "difference ×4" : "SSIM map";
    // block outline
    for (const c of [cOrig, cDec, cErr]) {
      const ctx = c.getContext("2d")!, k = zoom ? 1 : 1;
      ctx.strokeStyle = "#ffd400"; ctx.lineWidth = zoom ? 1 : 2;
      ctx.strokeRect((st.bx * 8 - (zoom ? zoom.x0 : 0)) * k - (zoom ? 0.5 : 1), (st.by * 8 - (zoom ? zoom.y0 : 0)) * k - (zoom ? 0.5 : 1), 8 + (zoom ? 1 : 2), 8 + (zoom ? 1 : 2));
    }
    // readout
    const colour = planes.length === 3, raw = N * N * (colour ? 3 : 1);
    const P = psnrOf(planes, res.planes);
    const v = ++blobVersion;
    const text = (blob: number) => {
      ro.textContent =
        `quality ${st.q}${st.flat ? `, flat step ${res.flatStep}` : ""}${colour ? (st.sub ? ", 4:2:0" : ", 4:4:4") : ""}\n` +
        `non-zero coefficients ${res.nonzero}\n` +
        `size ${res.bytes.toLocaleString("en-US").replace(/,/g, " ")} B, ${res.bpp.toFixed(2)} bits/px\n` +
        `ratio ${(raw / res.bytes).toFixed(1)} : 1\n` +
        `PSNR ${Number.isFinite(P) ? P.toFixed(2) + " dB" : "∞"}, SSIM ${S.mean.toFixed(3)}\n` +
        `browser encoder: ${Number.isFinite(blob) ? blob.toLocaleString("en-US").replace(/,/g, " ") + " B" : "…"}`;
    };
    text(NaN);
    if (st.gens === 1 && !st.flat) {
      const c = document.createElement("canvas");
      putPlanes(c, planes, null);
      c.toBlob(b => { if (v === blobVersion && b) text(b.size); }, "image/jpeg", st.q / 100);
    } else ro.textContent += "\n(only for 1 generation, standard table)";
    // tabs
    if (st.tab === "block") drawBlock(res);
    else if (st.tab === "table") drawTable(res);
    else drawRd(res, P, S.mean);
  }

  function drawBlock(res: Coded): void {
    const r = res.y, pw = r.pw, c = new Float64Array(64), q = new Int32Array(64);
    for (let v = 0; v < 8; v++) for (let u = 0; u < 8; u++) {
      const i = (st.by * 8 + v) * pw + st.bx * 8 + u;
      c[v * 8 + u] = r.coef[i]; q[v * 8 + u] = r.quant[i];
    }
    const nbx = pw / 8, bi = st.by * nbx + st.bx;
    const prev = bi === 0 ? 0 : r.quant[Math.floor((bi - 1) / nbx) * 8 * pw + ((bi - 1) % nbx) * 8];
    let m = 1;
    for (let i = 1; i < 64; i++) m = Math.max(m, Math.abs(c[i]));
    drawGrid(gCoef, c, m, v => (Math.abs(v) >= 9.5 ? v.toFixed(0) : v.toFixed(1)), false);
    let mq = 1;
    for (let i = 1; i < 64; i++) mq = Math.max(mq, Math.abs(q[i]));
    drawGrid(gQuant, q, mq, v => String(v), true);
    const s = blockSymbols(q, prev), bits = blockBitString(s);
    const z = toZigzag(q);
    let last = 0;
    for (let i = 63; i >= 0; i--) if (z[i] !== 0) { last = i; break; }
    const zz = Array.from(z.slice(0, last + 1)).join(" ") + (last < 63 ? " EOB" : "");
    const syms = s.ac.map(a => (a.symbol === 0 ? "EOB" : a.symbol === 0xf0 ? "ZRL" : `(${a.run},${a.size}) ${a.value}`)).join("  ");
    symDiv.innerHTML = "";
    const add = (h: string, t: string) => { const p = htmlEl("div", "", "", symDiv); htmlEl("b", "", h + " ", p); p.append(t); };
    add(`block (${st.bx}, ${st.by})${planes.length === 3 ? ", Y′" : ""}:`, "");
    add("zigzag:", zz);
    add("DC:", `${z[0]} − ${prev} (previous block) = ${s.dcDiff}, size ${s.dcSize}`);
    add("AC (run, size) value:", syms || "–");
    add(`bits (${bits.length}):`, bits);
  }

  function drawTable(res: Coded): void {
    drawGrid(gQl, res.Ql, 255, v => String(v), false);
    (gQc.parentElement as HTMLElement).style.display = res.Qc ? "" : "none";
    if (res.Qc) drawGrid(gQc, res.Qc, 255, v => String(v), false);
    tableNote.textContent = st.flat ? `Flat table: every step ${res.flatStep}, chosen so that the file has about the same size as with the standard table at quality ${st.q}.`
      : `Standard table scaled for quality ${st.q}: scale ${st.q < 50 ? (5000 / st.q).toFixed(0) : (200 - 2 * st.q)}%, Q' = ⌊(Q·scale + 50)/100⌋, at least 1. Small steps at low frequencies (top left), large steps at high frequencies.`;
  }

  function drawRd(res: Coded, P: number, Sm: number): void {
    const key = st.img + (st.sub ? "s" : "");
    let rd = rdCache.get(key);
    if (!rd) {
      rd = { bpp: [], ps: [], ss: [] };
      const s2: State = { ...st, flat: false, gens: 1 };
      for (let q = 5; q <= 100; q += 5) {
        const r = pipeline(planes, { ...s2, q });
        rd.bpp.push(r.bpp); rd.ps.push(Math.min(60, psnrOf(planes, r.planes)));
        rd.ss.push(ssim(lumaOf(planes), lumaOf(r.planes), N, N, { peak: 255 }).mean);
      }
      rdCache.set(key, rd);
    }
    const xm = Math.ceil(Math.max(...rd.bpp, res.bpp) + 0.2);
    pPsnr.clear(); pSsim.clear();
    pPsnr.setDomain([0, xm], [Math.floor(Math.min(...rd.ps, P) / 5) * 5, Math.ceil(Math.max(...rd.ps.filter(Number.isFinite)) / 5) * 5 + 2]);
    pSsim.setDomain([0, xm], [Math.floor(Math.min(...rd.ss, Sm) * 10) / 10, 1]);
    pPsnr.line(rd.bpp, rd.ps, { color: "#1f77b4", width: 2.5 });
    pSsim.line(rd.bpp, rd.ss, { color: "#2ca02c", width: 2.5 });
    pPsnr.scatter([res.bpp], [Math.min(60, P)], { r: 7, color: "#ff7f0e" });
    pSsim.scatter([res.bpp], [Sm], { r: 7, color: "#ff7f0e" });
  }

  for (const c of [cOrig, cDec, cErr]) {
    c.addEventListener("pointerdown", ev => {
      const r = c.getBoundingClientRect();
      const fx = (ev.clientX - r.left) / r.width, fy = (ev.clientY - r.top) / r.height;
      let x = fx * N, y = fy * N;
      if (st.zoom) {
        const x0 = Math.max(0, Math.min(N - 64, st.bx * 8 + 4 - 32)), y0 = Math.max(0, Math.min(N - 64, st.by * 8 + 4 - 32));
        x = x0 + fx * 64; y = y0 + fy * 64;
      }
      st.bx = Math.max(0, Math.min(N / 8 - 1, Math.floor(x / 8)));
      st.by = Math.max(0, Math.min(N / 8 - 1, Math.floor(y / 8)));
      draw();
    });
  }
  syncUI();
  load();
}

autoMount(NAME, mount);
