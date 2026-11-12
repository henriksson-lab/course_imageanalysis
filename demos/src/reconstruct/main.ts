// reconstruct: morphological reconstruction on 256 × 256 images (morphology day, unit
// morph-reconstruct). Two tabs.
//  binary: a mask of blobs (some on the border, some with holes, thin bridges, specks).
//    fill holes, clear border, opening vs opening by reconstruction (disc radius r), or
//    click a blob: its reconstruction from that one pixel is exactly the blob. The replay
//    slider shows the reconstruction growing one geodesic dilation at a time; orange = the
//    marker. Dark: kept · red: removed · green: added.
//  grey: synthetic nuclei or the micrograph. h slider: regional maxima (or minima) of the
//    h-maxima transform as dots, with their count; views: input, HMAX_h = R_f(f − h), the
//    domes f − HMAX_h, and minima imposition (click to place markers).
// Options: data-preset (fill | clear | openrec | label | hmax | noise | impose), data-tab.
import { autoMount, htmlEl, slider, presets, button, svgEl } from "../lib/ui";
import { Grey, createGrey, toRGBA } from "../lib/imageops";
import { edt } from "../lib/distance";
import { label2d, centroids } from "../lib/label";
import { regionalMaxima, regionalMinima, hMaxTransform, hMinTransform, imposeMinima } from "../lib/morph";
import { nuclei } from "../lib/synth";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { geodesicSteps, frameOf } from "./binrec";
import { N, blobs } from "./images";

const NAME = "reconstruct";
type BinOp = "fill" | "clear" | "open" | "openrec" | "click";
type View = "input" | "hmax" | "domes" | "impose";
const BINOPS: { id: BinOp; label: string }[] = [
  { id: "fill", label: "fill holes" }, { id: "clear", label: "clear border" },
  { id: "open", label: "opening" }, { id: "openrec", label: "opening by reconstruction" }, { id: "click", label: "click a blob" },
];
const VIEWS: { id: View; label: string }[] = [
  { id: "input", label: "input" }, { id: "hmax", label: "reconstruction" }, { id: "domes", label: "domes / basins" }, { id: "impose", label: "impose minima" },
];
interface Preset { id: string; label: string; tab: "binary" | "grey"; op?: BinOp; r?: number; img?: "nuclei" | "cell"; ext?: "max" | "min"; h?: number; view?: View; click?: number }
export const NUCLEI_H = 26;
const PRESETS: Preset[] = [
  { id: "fill", label: "fill holes", tab: "binary", op: "fill" },
  { id: "clear", label: "clear border", tab: "binary", op: "clear" },
  { id: "openrec", label: "opening breaks shapes, reconstruction does not", tab: "binary", op: "openrec", r: 4 },
  { id: "label", label: "one pixel → its blob", tab: "binary", op: "click", click: 0 },
  { id: "hmax", label: `one dot per nucleus with h = ${NUCLEI_H}`, tab: "grey", img: "nuclei", ext: "max", h: NUCLEI_H, view: "input" },
  { id: "noise", label: "noise: hundreds of maxima", tab: "grey", img: "nuclei", ext: "max", h: 0, view: "input" },
  { id: "impose", label: "impose two minima", tab: "grey", img: "cell", ext: "min", h: 0, view: "impose" },
];

const STYLE = `
[data-demo="${NAME}"] { gap: 1em; }
[data-demo="${NAME}"] .rc-img { flex: 0 1 470px; min-width: 0; position: relative; }
[data-demo="${NAME}"] .rc-img canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; cursor: crosshair; touch-action: none; }
[data-demo="${NAME}"] .rc-img svg { position: absolute; left: 0; top: 0; width: 100%; aspect-ratio: 1; pointer-events: none; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 380px; gap: .4em; }
[data-demo="${NAME}"] .rc-lab { font-size: 15px; color: var(--qb-muted); }
[data-demo="${NAME}"] .rc-credit { font-size: 12px; color: var(--qb-muted); margin-top: .2em; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .4em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.reveal [data-demo="${NAME}"] .rc-img { flex-basis: 440px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const COL = { bg: [255, 255, 255], kept: [40, 50, 75], removed: [232, 105, 100], added: [110, 200, 105], marker: [255, 150, 30], reach: [170, 205, 240], other: [200, 200, 205] };

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS.find(p => p.tab === host.dataset.tab) ?? PRESETS[0];
  const st = {
    tab: "binary" as "binary" | "grey", op: "fill" as BinOp, r: 4, click: -1, k: 0,
    img: "nuclei" as "nuclei" | "cell", ext: "max" as "max" | "min", h: 0, view: "input" as View, markers: [] as number[],
  };
  const mask = blobs();
  let steps: Int32Array = new Int32Array(0), result: Uint8Array = new Uint8Array(0), markerImg: Uint8Array = new Uint8Array(0), maxStep = 0;
  let nuc: Grey | null = null, cell: Grey | null = null;

  const box = htmlEl("div", "rc-img", "", host);
  const canvas = htmlEl("canvas", "", "", box);
  canvas.width = N; canvas.height = N;
  const ctx = canvas.getContext("2d")!;
  const svg = svgEl("svg", { viewBox: `0 0 ${N} ${N}` }, box);
  const credit = htmlEl("div", "rc-credit", "", box);

  const panel = htmlEl("div", "qb-panel", "", host);
  const tabBtns = presets(panel, [
    { label: "binary", apply: () => { st.tab = "binary"; recompute(); } },
    { label: "grey", apply: () => { st.tab = "grey"; recompute(); } },
  ]);
  const binBox = htmlEl("div", "qb-panel", "", panel);
  binBox.style.maxWidth = "none";
  const opBtns = presets(binBox, BINOPS.map(o => ({ label: o.label, apply: () => { st.op = o.id; recompute(); } })));
  const rS = slider(binBox, "SE radius r", 1, 12, 1, st.r, v => { st.r = v; if (st.op === "open" || st.op === "openrec") recompute(); });
  const kS = slider(binBox, "replay: iteration", 0, 100, 1, 100, v => { st.k = v; draw(); });
  const greyBox = htmlEl("div", "qb-panel", "", panel);
  greyBox.style.maxWidth = "none";
  const imgBtns = presets(greyBox, [
    { label: "nuclei (synthetic)", apply: () => { st.img = "nuclei"; st.markers = []; recompute(); } },
    { label: "cell (micrograph)", apply: () => { st.img = "cell"; st.markers = []; recompute(); } },
  ]);
  const extBtns = presets(greyBox, [
    { label: "maxima", apply: () => { st.ext = "max"; recompute(); } },
    { label: "minima", apply: () => { st.ext = "min"; recompute(); } },
  ]);
  const hS = slider(greyBox, "h", 0, 80, 1, st.h, v => { st.h = v; recompute(); });
  const viewBtns = presets(greyBox, VIEWS.map(v => ({ label: v.label, apply: () => { st.view = v.id; recompute(); } })));
  const ro = htmlEl("div", "qb-readout", "", panel);
  ro.style.whiteSpace = "pre-line";
  const note = htmlEl("div", "qb-note", "", panel);
  htmlEl("div", "rc-lab", "examples", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  button(panel, "reset", () => load(p0));

  // a pixel inside the largest blob that does not touch the border (for the "one pixel" preset)
  const defaultClick = (() => {
    const lab = label2d(mask, N, N, 8), cs = centroids(lab), fr = frameOf(N, N), touches = new Set<number>();
    for (let i = 0; i < N * N; i++) if (fr[i] && lab.data[i]) touches.add(lab.data[i]);
    let best = -1, bestN = 0;
    cs.forEach((c, k) => { if (!touches.has(k + 1) && c.n > bestN) { bestN = c.n; best = k + 1; } });
    for (let i = 0; i < N * N; i++) if (lab.data[i] === best) return i;
    return 0;
  })();

  async function load(p: Preset): Promise<void> {
    st.tab = p.tab;
    if (p.op) st.op = p.op;
    if (p.r) { st.r = p.r; rS.set(p.r); }
    st.click = p.click !== undefined ? defaultClick : -1;
    if (p.img) st.img = p.img;
    if (p.ext) st.ext = p.ext;
    if (p.h !== undefined) { st.h = p.h; hS.set(p.h); }
    if (p.view) st.view = p.view;
    st.markers = p.id === "impose" ? [110 * N + 128, 60 * N + 200] : [];
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    await recompute();
  }

  async function greyImage(): Promise<Grey> {
    if (st.img === "nuclei") { nuc ??= nuclei({ seed: 3, count: 22 }).image; return nuc; }
    if (!cell) { try { cell = await loadCell(); } catch { cell = createGrey(N, N, 128); } }
    return cell;
  }

  let grey: { show: Float32Array; dots: { x: number; y: number }[]; count0: number; count1: number; lo: number; hi: number } | null = null;

  async function recompute(): Promise<void> {
    tabBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.tab === "binary")));
    binBox.style.display = st.tab === "binary" ? "" : "none";
    greyBox.style.display = st.tab === "grey" ? "" : "none";
    if (st.tab === "binary") {
      opBtns.forEach((b, i) => b.classList.toggle("active", BINOPS[i].id === st.op));
      const n = N * N;
      markerImg = new Uint8Array(n);
      if (st.op === "fill") {
        const bg = Uint8Array.from(mask, v => (v ? 0 : 1)), fr = frameOf(N, N);
        for (let i = 0; i < n; i++) markerImg[i] = fr[i] & bg[i];
        steps = geodesicSteps(markerImg, bg, N, N, 4);
        result = Uint8Array.from(mask, (v, i) => (v || steps[i] < 0 ? 1 : 0));
      } else if (st.op === "clear") {
        const fr = frameOf(N, N);
        for (let i = 0; i < n; i++) markerImg[i] = fr[i] & mask[i];
        steps = geodesicSteps(markerImg, mask, N, N, 8);
        result = Uint8Array.from(mask, (v, i) => (v && steps[i] < 0 ? 1 : 0));
      } else if (st.op === "open" || st.op === "openrec") {
        const d = edt(mask, N, N, { squared: true });
        for (let i = 0; i < n; i++) markerImg[i] = d[i] > st.r * st.r ? 1 : 0;   // erosion by the disc
        if (st.op === "open") {
          const dc = edt(Uint8Array.from(markerImg, v => (v ? 0 : 1)), N, N, { squared: true });
          result = Uint8Array.from(dc, v => (v <= st.r * st.r ? 1 : 0));         // dilation by the disc
          steps = new Int32Array(n).fill(-1);
        } else {
          steps = geodesicSteps(markerImg, mask, N, N, 8);
          result = Uint8Array.from(steps, v => (v >= 0 ? 1 : 0));
        }
      } else {
        if (st.click >= 0 && mask[st.click]) markerImg[st.click] = 1;
        steps = geodesicSteps(markerImg, mask, N, N, 8);
        result = Uint8Array.from(steps, v => (v >= 0 ? 1 : 0));
      }
      maxStep = 0;
      for (const v of steps) if (v > maxStep) maxStep = v;
      kS.input.max = String(Math.max(1, maxStep));
      st.k = maxStep; kS.set(maxStep);
      (kS.input.closest("label") as HTMLElement).style.opacity = st.op === "open" ? "0.4" : "1";
      (rS.input.closest("label") as HTMLElement).style.opacity = st.op === "open" || st.op === "openrec" ? "1" : "0.4";
      credit.textContent = "Synthetic mask.";
      grey = null;
    } else {
      imgBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.img === "nuclei")));
      extBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.ext === "max")));
      viewBtns.forEach((b, i) => b.classList.toggle("active", VIEWS[i].id === st.view));
      const f = await greyImage(), n = N * N, mx = st.ext === "max";
      const ext0 = mx ? regionalMaxima(f.data, N, N, 8) : regionalMinima(f.data, N, N, 8);
      let show: Float32Array, extMask: Uint8Array;
      if (st.view === "impose") {
        const mk = new Uint8Array(n);
        for (const p of st.markers) { const x = p % N, y = (p - x) / N; for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) { const xx = x + dx, yy = y + dy; if (xx >= 0 && yy >= 0 && xx < N && yy < N && dx * dx + dy * dy <= 5) mk[yy * N + xx] = 1; } }
        show = imposeMinima(f.data, mk, N, N, 8);
        extMask = regionalMinima(show, N, N, 8);
      } else {
        const t = mx ? hMaxTransform(f.data, N, N, st.h, 8) : hMinTransform(f.data, N, N, st.h, 8);
        extMask = mx ? regionalMaxima(t, N, N, 8) : regionalMinima(t, N, N, 8);
        show = st.view === "input" ? Float32Array.from(f.data) : st.view === "hmax" ? t : Float32Array.from(t, (v, i) => (mx ? f.data[i] - v : v - f.data[i]));
      }
      const c0 = label2d(ext0, N, N, 8).count, lab = label2d(extMask, N, N, 8);
      let lo = Infinity, hi = -Infinity;
      for (const v of show) { if (v < lo) lo = v; if (v > hi) hi = v; }
      if (st.view !== "domes") { lo = Math.min(lo, 0); hi = Math.max(hi, 255); }
      grey = { show, dots: centroids(lab).map(c => ({ x: c.x, y: c.y })), count0: c0, count1: lab.count, lo, hi };
      credit.textContent = st.img === "cell" ? CELL_CREDIT : "Synthetic nuclei (22 placed; seed 3).";
    }
    draw();
  }

  function draw(): void {
    svg.replaceChildren();
    if (st.tab === "binary") {
      const id = ctx.createImageData(N, N), px = id.data;
      let kept = 0, removed = 0, added = 0, reached = 0;
      for (let i = 0; i < N * N; i++) {
        const a = mask[i], s = steps[i], reachedNow = s >= 0 && s <= st.k, done = st.k >= maxStep;
        if (reachedNow) reached++;
        let c = COL.bg;
        if (st.op === "fill") {
          if (a) c = COL.kept;
          else if (reachedNow) c = s === 0 ? COL.marker : COL.reach;
          else if (done) { c = COL.added; added++; }
        } else if (st.op === "clear") {
          if (a) c = reachedNow ? (s === 0 ? COL.marker : COL.removed) : COL.kept;
          if (a && done && s >= 0) removed++;
        } else if (st.op === "open") {
          if (a) c = result[i] ? COL.kept : COL.removed;
          if (a && !result[i]) removed++;
        } else if (st.op === "openrec") {
          if (a) c = reachedNow ? (s === 0 ? COL.marker : COL.kept) : COL.removed;
          if (a && !result[i]) removed++;
        } else {
          if (a) c = reachedNow ? (s === 0 ? COL.marker : COL.kept) : COL.other;
        }
        if (a && result[i]) kept++;
        px[4 * i] = c[0]; px[4 * i + 1] = c[1]; px[4 * i + 2] = c[2]; px[4 * i + 3] = 255;
      }
      ctx.putImageData(id, 0, 0);
      const it = st.op === "open" ? "" : `\nreplay: iteration ${st.k} of ${maxStep}`;
      const lines: Record<BinOp, string> = {
        fill: `holes filled: ${added} pixels (green)`,
        clear: `removed: ${removed} pixels of objects touching the border`,
        open: `opening, disc r = ${st.r}: ${removed} pixels removed`,
        openrec: `opening by reconstruction, r = ${st.r}: ${removed} pixels removed`,
        click: st.click >= 0 ? `reconstruction from one pixel: ${result.reduce((s, v) => s + v, 0)} pixels = its blob` : "click a blob",
      };
      ro.textContent = lines[st.op] + it;
      const notes: Record<BinOp, string> = {
        fill: "Marker (orange): the background on the image border. Its reconstruction in the background (blue) is everything reachable from outside; the rest of the background is a hole.",
        clear: "Marker (orange): object pixels on the border. Their reconstruction inside the mask = the objects touching the border, which are removed (red).",
        open: "Erosion then dilation by the disc: small objects go, but the survivors lose corners, bridges and protrusions.",
        openrec: "Marker (orange): the erosion by the disc. Reconstruction inside the mask restores every object that contains part of it, unchanged.",
        click: "Click a blob. Its reconstruction from that single pixel is exactly its connected component (8-connected): the link to labelling.",
      };
      note.textContent = notes[st.op];
      return;
    }
    if (!grey) return;
    const g: Grey = { width: N, height: N, data: grey.show };
    ctx.putImageData(new ImageData(toRGBA(g, { mode: "clip", lo: grey.lo, hi: grey.hi }), N, N), 0, 0);
    const col = st.ext === "max" ? "#ff3b30" : "#2f8cff";
    for (const d of grey.dots) svgEl("circle", { cx: d.x + 0.5, cy: d.y + 0.5, r: grey.dots.length > 80 ? 1.6 : 3.2, fill: col, stroke: "#fff", "stroke-width": grey.dots.length > 80 ? 0.4 : 1 }, svg);
    const what = st.ext === "max" ? "maxima" : "minima";
    ro.textContent = st.view === "impose"
      ? `markers: ${st.markers.length}\nregional minima after imposition: ${grey.count1}`
      : `regional ${what} of the image: ${grey.count0}\nh-${what}, h = ${st.h}: ${grey.count1}`;
    const vn: Record<View, string> = {
      input: `The image with its h-${what} as dots (regional ${what} of the reconstruction).`,
      hmax: st.ext === "max" ? `HMAX_h = reconstruction of f − ${st.h} under f: every peak lowered by ${st.h}; lower ones flattened.` : `HMIN_h = reconstruction by erosion of f + ${st.h} above f: every basin filled by ${st.h}.`,
      domes: st.ext === "max" ? "Domes f − HMAX_h: the top h of every peak." : "Basins HMIN_h − f: the water poured into every basin.",
      impose: "Click to add markers. The imposed landscape keeps the shape of the image but has minima only at the markers (blue dots).",
    };
    note.textContent = vn[st.view];
  }

  canvas.addEventListener("pointerdown", ev => {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((ev.clientX - rect.left) / rect.width * N), y = Math.floor((ev.clientY - rect.top) / rect.height * N);
    if (x < 0 || y < 0 || x >= N || y >= N) return;
    if (st.tab === "binary") { st.op = "click"; st.click = y * N + x; void recompute(); }
    else if (st.view === "impose") { st.markers.push(y * N + x); void recompute(); }
  });

  void load(p0);
}

autoMount(NAME, mount);
