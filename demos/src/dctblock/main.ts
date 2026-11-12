// Transform coding of one 8 × 8 block. Left: the image with the selected block (click to move
// it) and the energy-compaction curves of three bases over all blocks of the image. Middle:
// the block's pixel values and its reconstruction from the kept coefficients. Right: the 64
// coefficients (kept ones in colour, dropped ones grey) and the 64 basis images (click one to
// keep or drop it). Bases: DCT, DFT (as a real cos/sin basis) and the KLT learnt from all
// blocks of the current image (eigenvectors of their second-moment matrix, by eigenvalue).
// Keep: the k largest coefficients of this block, the first k in frequency order (zigzag;
// eigenvalue order for the KLT), or a hand-picked set.
// data-mode="1d": one row of 8 samples (drag the points), its periodic extension (what the
// DFT assumes) and its mirrored extension (what the DCT assumes), and the DFT and DCT
// coefficients with the error of keeping the first k of each.
// Auto-mounts on <div data-demo="dctblock">. Options: data-mode="1d", data-preset (2D: flat,
// edge, ramp, texture, klt; 1D: ramp, step, smooth, random), data-basis="dct|dft|klt",
// data-k="6", data-keep="largest|first".
import { htmlEl, svgEl, slider, presets, button, autoMount, svgPoint } from "../lib/ui";
import { dctMatrix, realDftMatrix, separableBasis, zigzag, applyMatrix, applyTranspose, energy } from "../lib/dct";
import { jacobiEigen } from "../lib/linalg";
import { Plot } from "../lib/plot";
import { drawScalarImage, diverging, divergingT, toCss, textColorOn } from "../lib/colour";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { smooth, shapes, card, SIZE } from "../jpeg/testimages";

const NAME = "dctblock";
const N = SIZE, B = 8, NB = N / B;
type BasisId = "dct" | "dft" | "klt";
type ImgId = "cell" | "smooth" | "shapes" | "card";
const COL: Record<BasisId, string> = { dct: "#1f77b4", dft: "#d62728", klt: "#2ca02c" };
const BASIS_LABEL: Record<BasisId, string> = { dct: "DCT", dft: "DFT (cos/sin)", klt: "KLT (this image)" };
const IMG_LABEL: Record<ImgId, string> = { cell: "cell", smooth: "smooth", shapes: "shapes", card: "test card" };
const CREDIT: Record<ImgId, string> = {
  cell: CELL_CREDIT, smooth: "Synthetic: soft blobs on a gradient.", shapes: "Synthetic: flat shapes with straight and diagonal edges.",
  card: "Synthetic test card: gradient, flat areas, an edge, stripes.",
};

const STYLE = `
[data-demo="${NAME}"] .db-cols { display: flex; gap: 12px; flex: 1 1 760px; min-width: 0; max-width: 900px; }
[data-demo="${NAME}"] .db-col { display: flex; flex-direction: column; gap: 6px; flex: 1 1 0; min-width: 0; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; line-height: 1.2; margin-bottom: .15em; }
[data-demo="${NAME}"] canvas.img { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; cursor: crosshair; touch-action: none; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 220px; max-width: 270px; }
[data-demo="${NAME}"] .db-btns { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
[data-demo="${NAME}"] .qb-btn.on { background: var(--qb-accent); color: #fff; border-color: var(--qb-accent); }
[data-demo="${NAME}"] .db-credit { font-size: 13px; color: var(--qb-muted); }
[data-demo="${NAME}"] svg.grid text { font-family: ui-monospace, monospace; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { font-size: 14px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const grey = (v: number) => { const g = Math.max(0, Math.min(255, Math.round(v))); return `rgb(${g},${g},${g})`; };
const fmt1 = (v: number) => (Math.abs(v) < 0.05 ? "0.0" : v.toFixed(1));
const fmtC = (v: number) => (Math.abs(v) >= 99.5 ? v.toFixed(0) : Math.abs(v) >= 9.95 ? v.toFixed(0) : v.toFixed(1));

/** An 8 × 8 grid of cells with numbers in an SVG (viewBox 240 × 240). */
function drawGrid(svg: SVGSVGElement, fill: (i: number) => string, text: (i: number) => string, textCol: (i: number) => string,
  outline?: (i: number) => string | null): void {
  svg.replaceChildren();
  const c = 30;
  for (let i = 0; i < 64; i++) {
    const x = (i % 8) * c, y = Math.floor(i / 8) * c;
    svgEl("rect", { x, y, width: c, height: c, fill: fill(i), stroke: "var(--qb-bg)", "stroke-width": 1 }, svg);
    const s = text(i);
    if (s) {
      const t = svgEl("text", { x: x + c / 2, y: y + c / 2 + 4.5, "text-anchor": "middle", "font-size": s.length > 3 ? 11 : 13 }, svg);
      t.textContent = s; t.style.fill = textCol(i);
    }
    const o = outline?.(i);
    if (o) svgEl("rect", { x: x + 1.5, y: y + 1.5, width: c - 3, height: c - 3, fill: "none", stroke: o, "stroke-width": 3 }, svg);
  }
}

// =======================================================================================
// 2D: one block of an image

interface Bases { A: Record<BasisId, Float64Array>; order: Record<BasisId, Int32Array>; curves: Record<BasisId, Float64Array>; mean: number }

/** Bases (64 × 64, rows = basis images), their frequency order, and compaction curves over all blocks. */
function computeBases(img: Float64Array): Bases {
  const mean = 128;   // level shift as in JPEG; the KLT uses the second moments of x − 128
  const X = new Float64Array(NB * NB * 64);
  for (let by = 0; by < NB; by++) for (let bx = 0; bx < NB; bx++)
    for (let y = 0; y < B; y++) for (let x = 0; x < B; x++) X[(by * NB + bx) * 64 + y * B + x] = img[(by * B + y) * N + bx * B + x] - mean;
  const S = new Float64Array(64 * 64), nb = NB * NB;
  for (let b = 0; b < nb; b++) {
    const o = b * 64;
    for (let i = 0; i < 64; i++) { const xi = X[o + i]; if (xi === 0) continue; for (let j = i; j < 64; j++) S[i * 64 + j] += xi * X[o + j]; }
  }
  for (let i = 0; i < 64; i++) for (let j = i; j < 64; j++) { S[i * 64 + j] /= nb; S[j * 64 + i] = S[i * 64 + j]; }
  const eig = jacobiEigen(S, 64, true);
  const klt = new Float64Array(64 * 64);
  for (let k = 0; k < 64; k++) {
    let s = 0;
    for (let i = 0; i < 64; i++) s += eig.vectors[i * 64 + k];
    const sg = s < -1e-9 ? -1 : 1;
    for (let i = 0; i < 64; i++) klt[k * 64 + i] = sg * eig.vectors[i * 64 + k];
  }
  const A: Record<BasisId, Float64Array> = { dct: separableBasis(dctMatrix(B), B), dft: separableBasis(realDftMatrix(B), B), klt };
  const zz = zigzag(B);
  const order: Record<BasisId, Int32Array> = { dct: zz, dft: zz, klt: Int32Array.from({ length: 64 }, (_, i) => i) };
  // compaction: share of the energy in the k coefficients with the largest average energy
  const curves = {} as Record<BasisId, Float64Array>;
  for (const id of ["dct", "dft", "klt"] as BasisId[]) {
    const v = new Float64Array(64);
    for (let r = 0; r < 64; r++) {
      let s = 0;
      for (let i = 0; i < 64; i++) for (let j = 0; j < 64; j++) s += A[id][r * 64 + i] * S[i * 64 + j] * A[id][r * 64 + j];
      v[r] = s;
    }
    const sorted = Array.from(v).sort((a, b) => b - a);
    const tot = sorted.reduce((a, b) => a + b, 0), c = new Float64Array(64);
    let acc = 0;
    sorted.forEach((x, i) => { acc += x; c[i] = 100 * acc / tot; });
    curves[id] = c;
  }
  return { A, order, curves, mean };
}

interface Preset2 { id: string; label: string; img: ImgId; bx: number; by: number; basis: BasisId; keep: "largest" | "first"; k: number }
const PRESETS2: Preset2[] = [
  { id: "flat", label: "flat block: only DC", img: "card", bx: 4, by: 10, basis: "dct", keep: "largest", k: 1 },
  { id: "edge", label: "vertical edge", img: "card", bx: 16, by: 11, basis: "dct", keep: "largest", k: 8 },
  { id: "ramp", label: "ramp: DFT leaks, DCT does not", img: "card", bx: 12, by: 2, basis: "dft", keep: "first", k: 3 },
  { id: "texture", label: "texture block", img: "card", bx: 6, by: 18, basis: "dct", keep: "largest", k: 6 },
  { id: "cell", label: "a block of the cell", img: "cell", bx: 14, by: 12, basis: "dct", keep: "largest", k: 6 },
  { id: "klt", label: "KLT looks like the DCT", img: "smooth", bx: 8, by: 8, basis: "klt", keep: "first", k: 6 },
];

function mount2d(host: HTMLElement): void {
  const d = host.dataset;
  const p0 = PRESETS2.find(p => p.id === d.preset) ?? PRESETS2[4];
  const init = {
    img: p0.img, bx: p0.bx, by: p0.by,
    basis: (["dct", "dft", "klt"].includes(d.basis ?? "") ? d.basis : p0.basis) as BasisId,
    keep: (d.keep === "first" || d.keep === "largest" ? d.keep : p0.keep) as "largest" | "first" | "manual",
    k: Number(d.k ?? p0.k),
  };
  const st: typeof init & { manual: Uint8Array } = { ...init, manual: new Uint8Array(64) };

  const cols = htmlEl("div", "db-cols", "", host);
  const c1 = htmlEl("div", "db-col", "", cols), c2 = htmlEl("div", "db-col", "", cols), c3 = htmlEl("div", "db-col", "", cols);
  const fig = (p: Element, t: string) => { const f = htmlEl("figure", "", "", p); htmlEl("figcaption", "", t, f); return f; };
  const cImg = htmlEl("canvas", "img", "", fig(c1, "image: click a block"));
  const curveFig = fig(c1, "energy in the best k coefficients");
  const curve = new Plot(curveFig, { width: 300, height: 250, x: [1, 64], y: [50, 100], xlabel: "k", ylabel: "% energy", font: 17, margin: { l: 66, b: 50, r: 10, t: 8 }, xticks: [1, 16, 32, 48, 64] });
  const gridSvg = (p: Element, t: string) => svgEl("svg", { viewBox: "0 0 240 240", width: "100%", class: "grid" }, fig(p, t));
  const gBlock = gridSvg(c2, "block (pixel values)");
  const recCap = fig(c2, "");
  const gRec = svgEl("svg", { viewBox: "0 0 240 240", width: "100%", class: "grid" }, recCap);
  const gCoef = gridSvg(c3, "coefficients");
  const basisFig = fig(c3, "basis images: click to keep");
  const cBasis = htmlEl("canvas", "img", "", basisFig);
  cBasis.width = 8 * 9 + 1; cBasis.height = 8 * 9 + 1;
  const credit = htmlEl("div", "db-credit", "", c1);

  const panel = htmlEl("div", "qb-panel", "", host);
  const pbtn = presets(panel, PRESETS2.map(p => ({ label: p.label, apply: () => { Object.assign(st, { img: p.img, bx: p.bx, by: p.by, basis: p.basis, keep: p.keep, k: p.k }); syncUI(); load(); } })));
  pbtn.forEach((b, i) => b.classList.toggle("active", PRESETS2[i] === p0));
  htmlEl("div", "", "image:", panel);
  const imgRow = htmlEl("div", "db-btns", "", panel);
  const imgBtns = (Object.keys(IMG_LABEL) as ImgId[]).map(id => button(imgRow, IMG_LABEL[id], () => { st.img = id; syncUI(); load(); }));
  htmlEl("div", "", "basis:", panel);
  const bRow = htmlEl("div", "db-btns", "", panel);
  const basisBtns = (["dct", "dft", "klt"] as BasisId[]).map(id => button(bRow, BASIS_LABEL[id], () => { st.basis = id; if (st.keep === "manual") st.manual.fill(0); syncUI(); draw(); }));
  htmlEl("div", "", "keep:", panel);
  const kRow = htmlEl("div", "db-btns", "", panel);
  const keepBtns = [
    button(kRow, "k largest", () => { st.keep = "largest"; syncUI(); draw(); }),
    button(kRow, "first k", () => { st.keep = "first"; syncUI(); draw(); }),
    button(kRow, "by hand", () => { st.manual = keptMask(); st.keep = "manual"; syncUI(); draw(); }),
  ];
  const kS = slider(panel, "k", 1, 64, 1, st.k, v => { st.k = v; if (st.keep === "manual") st.keep = "largest"; syncUI(); draw(); });
  button(panel, "reset", () => { Object.assign(st, init); st.manual = new Uint8Array(64); syncUI(); load(); pbtn.forEach((b, i) => b.classList.toggle("active", PRESETS2[i] === p0)); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Pixels are shifted by −128 (as in JPEG) before the transform. Coefficients: red = positive, blue = negative, grey = dropped. Basis images: kept ones framed orange. First k: zigzag order (DCT, DFT), eigenvalue order (KLT).", panel);

  function syncUI(): void {
    imgBtns.forEach((b, i) => b.classList.toggle("on", Object.keys(IMG_LABEL)[i] === st.img));
    basisBtns.forEach((b, i) => b.classList.toggle("on", ["dct", "dft", "klt"][i] === st.basis));
    keepBtns.forEach((b, i) => b.classList.toggle("on", ["largest", "first", "manual"][i] === st.keep));
    kS.set(st.k);
  }

  let img: Float64Array = new Float64Array(N * N), bases: Bases | null = null;
  const cache = new Map<ImgId, { img: Float64Array; bases: Bases }>();
  function load(): void {
    const id = st.img;
    credit.textContent = CREDIT[id];
    const hit = cache.get(id);
    if (hit) { img = hit.img; bases = hit.bases; draw(); return; }
    const done = (v: Float64Array) => { const b = computeBases(v); cache.set(id, { img: v, bases: b }); if (st.img === id) { img = v; bases = b; draw(); } };
    if (id === "cell") loadCell().then(g => done(Float64Array.from(g.data, v => Math.round(v))));
    else done(id === "smooth" ? smooth() : id === "shapes" ? shapes() : card());
  }

  function block(): Float64Array {
    const b = new Float64Array(64);
    for (let y = 0; y < B; y++) for (let x = 0; x < B; x++) b[y * B + x] = img[(st.by * B + y) * N + st.bx * B + x];
    return b;
  }
  /** Kept coefficients of the current block (by coefficient index). */
  function keptMask(): Uint8Array {
    if (st.keep === "manual") return Uint8Array.from(st.manual);
    const m = new Uint8Array(64);
    if (!bases) return m;
    const c = applyMatrix(bases.A[st.basis], centred(), 64);
    const ord = st.keep === "first" ? Array.from(bases.order[st.basis]) : Array.from({ length: 64 }, (_, i) => i).sort((a, b) => Math.abs(c[b]) - Math.abs(c[a]) || a - b);
    for (let j = 0; j < Math.min(64, st.k); j++) m[ord[j]] = 1;
    return m;
  }
  const centred = () => block().map(v => v - (bases ? bases.mean : 0));

  function draw(): void {
    if (!bases) return;
    // image with the block
    drawScalarImage(cImg, img, N, N, { lo: 0, hi: 255 });
    const ctx = cImg.getContext("2d")!;
    ctx.strokeStyle = "#ffd400"; ctx.lineWidth = 2;
    ctx.strokeRect(st.bx * B - 1, st.by * B - 1, B + 2, B + 2);
    // compaction curves
    curve.clear();
    for (const id of ["dct", "dft", "klt"] as BasisId[]) curve.line(Float64Array.from({ length: 64 }, (_, i) => i + 1), bases.curves[id], { color: COL[id], width: id === st.basis ? 4 : 2, label: id.toUpperCase() });
    curve.legend(undefined, "br");
    // block, coefficients, reconstruction
    const A = bases.A[st.basis], x = centred(), c = applyMatrix(A, x, 64), keep = keptMask();
    const ck = c.map((v, i) => (keep[i] ? v : 0));
    const rec = applyTranspose(A, ck, 64).map(v => v + bases!.mean);
    const blk = block();
    const tc = (v: number) => (v > 140 ? "#000" : "#fff");
    drawGrid(gBlock, i => grey(blk[i]), i => blk[i].toFixed(0), i => tc(blk[i]));
    drawGrid(gRec, i => grey(rec[i]), i => rec[i].toFixed(0), i => tc(rec[i]));
    let cmax = 0;
    for (let i = 1; i < 64; i++) cmax = Math.max(cmax, Math.abs(c[i]));
    cmax = Math.max(cmax, 1);
    const cc = (i: number) => diverging(divergingT(Math.max(-cmax, Math.min(cmax, c[i])), cmax));
    drawGrid(gCoef, i => (keep[i] ? toCss(cc(i)) : "#d9d9d9"), i => fmtC(c[i]), i => (keep[i] ? textColorOn(cc(i)) : "#888"));
    // basis images
    const bctx = cBasis.getContext("2d")!, id = bctx.createImageData(cBasis.width, cBasis.height);
    for (let p = 0; p < id.data.length; p += 4) { id.data[p] = id.data[p + 1] = id.data[p + 2] = 255; id.data[p + 3] = 255; }
    for (let r = 0; r < 64; r++) {
      let m = 0;
      for (let i = 0; i < 64; i++) m = Math.max(m, Math.abs(A[r * 64 + i]));
      const ox = (r % 8) * 9 + 1, oy = Math.floor(r / 8) * 9 + 1;
      for (let i = 0; i < 64; i++) {
        const g = Math.round(127.5 + 127.5 * A[r * 64 + i] / (m || 1)), p = 4 * ((oy + Math.floor(i / 8)) * cBasis.width + ox + (i % 8));
        id.data[p] = id.data[p + 1] = id.data[p + 2] = g;
      }
      if (keep[r]) for (let t = -1; t <= 8; t++) for (const [xx, yy] of [[ox + t, oy - 1], [ox + t, oy + 8], [ox - 1, oy + t], [ox + 8, oy + t]]) {
        if (xx < 0 || yy < 0 || xx >= cBasis.width || yy >= cBasis.height) continue;
        const p = 4 * (yy * cBasis.width + xx);
        id.data[p] = 255; id.data[p + 1] = 127; id.data[p + 2] = 14;
      }
    }
    bctx.putImageData(id, 0, 0);
    const nk = keep.reduce((a, b) => a + b, 0);
    let err = 0;
    for (let i = 0; i < 64; i++) err += (rec[i] - blk[i]) ** 2;
    const eTot = energy(c), eKept = energy(ck);
    (recCap.firstChild as HTMLElement).textContent = `rebuilt from ${nk} of 64`;
    ro.textContent =
      `block (${st.bx}, ${st.by}) of ${NB}×${NB}\n` +
      `${BASIS_LABEL[st.basis]}\n` +
      `kept ${nk} coefficients\n` +
      `energy kept ${(100 * eKept / (eTot || 1)).toFixed(1)}%\n` +
      `RMS error ${Math.sqrt(err / 64).toFixed(2)} grey levels`;
  }

  cImg.addEventListener("pointerdown", ev => {
    const r = cImg.getBoundingClientRect();
    st.bx = Math.max(0, Math.min(NB - 1, Math.floor((ev.clientX - r.left) / r.width * NB)));
    st.by = Math.max(0, Math.min(NB - 1, Math.floor((ev.clientY - r.top) / r.height * NB)));
    if (st.keep === "manual") st.keep = "largest";
    syncUI(); draw();
  });
  cBasis.addEventListener("pointerdown", ev => {
    const r = cBasis.getBoundingClientRect();
    const u = Math.floor((ev.clientX - r.left) / r.width * 8), v = Math.floor((ev.clientY - r.top) / r.height * 8);
    if (u < 0 || v < 0 || u > 7 || v > 7) return;
    if (st.keep !== "manual") { st.manual = keptMask(); st.keep = "manual"; }
    st.manual[v * 8 + u] ^= 1;
    syncUI(); draw();
  });
  syncUI();
  load();
}

// =======================================================================================
// 1D: one row, periodic vs mirrored extension

const PRESETS1: { id: string; label: string; x: number[] }[] = [
  { id: "ramp", label: "ramp", x: [10, 20, 30, 40, 50, 60, 70, 80] },
  { id: "step", label: "step", x: [20, 20, 20, 20, 80, 80, 80, 80] },
  { id: "smooth", label: "smooth bump", x: [30, 42, 60, 74, 78, 70, 54, 40] },
  { id: "random", label: "random", x: [62, 18, 75, 40, 33, 90, 12, 55] },
];

function mount1d(host: HTMLElement): void {
  const d = host.dataset;
  const p0 = PRESETS1.find(p => p.id === d.preset) ?? PRESETS1[0];
  const st = { x: [...p0.x], k: Number(d.k ?? 3) };
  const cols = htmlEl("div", "db-cols", "", host);
  const c1 = htmlEl("div", "db-col", "", cols), c2 = htmlEl("div", "db-col", "", cols);
  const fig = (p: Element, t: string) => { const f = htmlEl("figure", "", "", p); htmlEl("figcaption", "", t, f); return f; };
  const ext = (t: string, col: string) => new Plot(fig(c1, t), { width: 520, height: 200, x: [-8.5, 15.5], y: [0, 100], font: 17, margin: { l: 50, b: 34, r: 8, t: 6 }, xticks: [-8, 0, 7, 15], yticks: [0, 50, 100], ariaLabel: col });
  const pPer = ext("periodic extension (the DFT's view): drag the black points", "periodic");
  const pMir = ext("mirrored extension (the DCT's view)", "mirrored");
  const pCoef = new Plot(fig(c2, "|coefficients|, frequency order"), { width: 380, height: 220, x: [-0.6, 7.6], y: [0, 100], xlabel: "index", font: 17, margin: { l: 56, b: 50, r: 8, t: 8 }, xticks: [0, 1, 2, 3, 4, 5, 6, 7] });
  const pRec = new Plot(fig(c2, "rebuilt from the first k"), { width: 380, height: 200, x: [-0.5, 7.5], y: [0, 100], font: 17, margin: { l: 56, b: 34, r: 8, t: 8 }, xticks: [0, 1, 2, 3, 4, 5, 6, 7] });
  const panel = htmlEl("div", "qb-panel", "", host);
  presets(panel, PRESETS1.map(p => ({ label: p.label, apply: () => { st.x = [...p.x]; draw(); } }))).forEach((b, i) => b.classList.toggle("active", PRESETS1[i] === p0));
  slider(panel, "keep first k", 1, 8, 1, st.k, v => { st.k = v; draw(); });
  button(panel, "reset", () => { st.x = [...p0.x]; draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Both bases are orthonormal, so the squared error equals the energy of the dropped coefficients. DFT coefficients as a real cos/sin basis: 0, cos 1, sin 1, cos 2, sin 2, cos 3, sin 3, cos 4.", panel);
  const D = dctMatrix(8), F = realDftMatrix(8);
  const idx = Array.from({ length: 8 }, (_, i) => i);

  function draw(): void {
    const x = st.x;
    for (const [p, mirror] of [[pPer, false], [pMir, true]] as [Plot, boolean][]) {
      p.clear();
      const xs: number[] = [], ys: number[] = [];
      for (let n = -8; n < 16; n++) {
        const m = ((n % 8) + 8) % 8, period = Math.floor(n / 8);
        xs.push(n); ys.push(mirror && period % 2 !== 0 ? x[7 - m] : x[m]);
      }
      p.line(xs, ys, { color: mirror ? "#1f77b4" : "#d62728", width: 2.5 });
      p.scatter(xs.filter(n => n < 0 || n > 7), ys.filter((_, i) => xs[i] < 0 || xs[i] > 7), { r: 4, color: mirror ? "#1f77b4" : "#d62728", opacity: 0.6 });
      p.vline(-0.5, { color: "var(--qb-grid)", dash: "4 3", width: 1.5 }); p.vline(7.5, { color: "var(--qb-grid)", dash: "4 3", width: 1.5 });
      const dots = p.scatter(idx, x, { r: 8, color: "var(--qb-fg)" });
      dots.g.querySelectorAll("circle").forEach((cEl, i) => {
        (cEl as SVGElement).style.cursor = "ns-resize"; (cEl as SVGElement).style.touchAction = "none";
        cEl.addEventListener("pointerdown", ev => {
          ev.preventDefault();
          const move = (e: PointerEvent) => { const [, yy] = svgPoint(p.svg, e); st.x[i] = Math.max(0, Math.min(100, Math.round(p.y.invert(yy)))); draw(); };
          const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
          window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
        });
      });
    }
    const cD = applyMatrix(D, x, 8), cF = applyMatrix(F, x, 8);
    pCoef.clear();
    const amax = Math.max(...Array.from(cD, Math.abs).slice(1), ...Array.from(cF, Math.abs).slice(1), 1);
    pCoef.setDomain(undefined, [0, Math.ceil(amax / 10) * 10]);
    pCoef.bars(idx.map(i => i - 0.18), Array.from(cF, Math.abs).map((v, i) => (i === 0 ? Math.min(v, amax) : v)), { color: "#d62728", barWidth: 0.34, label: "DFT" });
    pCoef.bars(idx.map(i => i + 0.18), Array.from(cD, Math.abs).map((v, i) => (i === 0 ? Math.min(v, amax) : v)), { color: "#1f77b4", barWidth: 0.34, label: "DCT" });
    pCoef.legend(undefined, "tr");
    if (Math.abs(cD[0]) > amax) pCoef.text(0, amax * 0.93, "DC " + Math.abs(cD[0]).toFixed(0), { anchor: "start", size: 15 });
    const keep = (c: Float64Array) => c.map((v, i) => (i < st.k ? v : 0));
    const rD = applyTranspose(D, keep(cD), 8), rF = applyTranspose(F, keep(cF), 8);
    pRec.clear();
    pRec.line(idx, x, { color: "var(--qb-fg)", width: 2, dash: "5 4" });
    pRec.line(idx, rF, { color: "#d62728", width: 3 });
    pRec.line(idx, rD, { color: "#1f77b4", width: 3 });
    const rms = (r: Float64Array) => Math.sqrt(r.reduce((s, v, i) => s + (v - x[i]) ** 2, 0) / 8);
    ro.textContent =
      `x = ${x.join(" ")}\n` +
      `DCT: ${Array.from(cD, fmt1).join(" ")}\n` +
      `DFT: ${Array.from(cF, fmt1).join(" ")}\n` +
      `first ${st.k}: RMS error DCT ${rms(rD).toFixed(2)}, DFT ${rms(rF).toFixed(2)}` +
      "";
  }
  draw();
}

function mount(host: HTMLElement): void {
  injectStyle();
  if (host.dataset.mode === "1d") mount1d(host); else mount2d(host);
}

autoMount(NAME, mount);
