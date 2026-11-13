// Felzenszwalb–Huttenlocher and single-linkage segmentation of a 256 × 256 image
// (graph-based segmentation day, unit 2). Edges: 4/8 neighbours with dissimilarity |ΔI| (grey)
// or the CIELAB distance ΔE (colour), after Gaussian pre-smoothing σ. Edges are sorted once per
// image and σ; the merge pass is re-run live on every k / t / minimum size. Click a region: its
// internal difference Int(C), size |C| and threshold Int(C) + k/|C|.
// <div data-demo="fhseg" data-preset="chain|texture|fh|scale|minsize|leak|cell|colour"></div>
import { autoMount, htmlEl, slider, button, presets, Slider } from "../lib/ui";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { gaussianBlur, fromArray } from "../lib/imageops";
import { srgbToLab, labelColor } from "../lib/colour";
import { DGraph, gridDissim, sortEdges, felzenszwalb, singleLinkage, FHResult } from "../lib/mst";
import { fhScene, colourScene } from "./scenes";
import { rng } from "../lib/random";

const NAME = "fhseg";
const N = 256;
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .7em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; flex: 0 1 380px; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin: 0 0 .15em; }
.qb-demo[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; cursor: crosshair; image-rendering: pixelated; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 360px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] .fh-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .fh-credit { font-size: 12px; color: var(--qb-muted); }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] figure { flex: 0 0 370px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 360px; gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .03em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .15em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .fh-lab, .reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

type Img = "synth" | "cell" | "colour";
type Method = "fh" | "single";
type View = "random" | "mean";
interface Settings { img: Img; method: Method; k: number; t: number; sigma: number; minSize: number; conn: 4 | 8; noise: number; view: View }
interface Preset { id: string; label: string; s: Partial<Settings>; click?: [number, number] }
const DEF: Settings = { img: "synth", method: "fh", k: 300, t: 6, sigma: 0.8, minSize: 0, conn: 8, noise: 4, view: "random" };
const PRESETS: Preset[] = [
  { id: "chain", label: "single linkage chains through the ramp", s: { method: "single", t: 9 }, click: [128, 218] },
  { id: "texture", label: "single linkage: texture or faint step, not both", s: { method: "single", t: 5 } },
  { id: "fh", label: "FH: both", s: { method: "fh", k: 3000 }, click: [70, 70] },
  { id: "scale", label: "k sets scale, not size", s: { method: "fh", k: 10000 }, click: [110, 140] },
  { id: "minsize", label: "minimum size cleanup", s: { method: "fh", k: 300, noise: 10, minSize: 60 } },
  { id: "leak", label: "leak through a narrow gap", s: { method: "fh", k: 3000 }, click: [170, 157] },
  { id: "cell", label: "micrograph", s: { img: "cell", method: "fh", k: 300, minSize: 30, noise: 0 } },
  { id: "colour", label: "colour scene", s: { img: "colour", method: "fh", k: 600, minSize: 30, noise: 8, view: "mean" } },
];
const KS = (u: number) => Math.round(10 * Math.pow(10, u)); // u in [0, 3]: k 10..10000

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: Settings = { ...DEF, ...p0.s };
  let click: [number, number] | null = p0.click ?? null;

  const f1 = htmlEl("figure", "", "", host);
  const cap1 = htmlEl("figcaption", "", "regions", f1);
  const cvR = htmlEl("canvas", "", "", f1); cvR.width = N; cvR.height = N;
  const f2 = htmlEl("figure", "", "", host);
  htmlEl("figcaption", "", "boundaries on the image", f2);
  const cvB = htmlEl("canvas", "", "", f2); cvB.width = N; cvB.height = N;
  const credit = htmlEl("div", "fh-credit", CELL_CREDIT, f2);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "fh-lab", "presets", panel);
  const pb = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "fh-lab", "image · method · view", panel);
  const r1 = htmlEl("div", "qb-presets", "", panel);
  const ib = (["synth", "cell", "colour"] as Img[]).map(i => button(r1, i === "synth" ? "test scene" : i === "cell" ? "micrograph" : "colour", () => { st.img = i; makeImage(); }));
  const r2 = htmlEl("div", "qb-presets", "", panel);
  const mb = (["fh", "single"] as Method[]).map(m => button(r2, m === "fh" ? "FH" : "single linkage", () => { st.method = m; segment(); }));
  const vb = (["random", "mean"] as View[]).map(v => button(r2, v === "random" ? "random colours" : "mean colours", () => { st.view = v; draw(); }));
  const kS: Slider = slider(panel, "k (FH)", 0, 3, 0.01, Math.log10(st.k / 10), u => { st.k = KS(u); segment(); }, u => String(KS(u)));
  const tS: Slider = slider(panel, "t (single linkage)", 1, 40, 1, st.t, v => { st.t = v; segment(); });
  const mS: Slider = slider(panel, "minimum size", 0, 200, 5, st.minSize, v => { st.minSize = v; segment(); });
  const SIG = [0, 0.5, 0.8, 1.5, 2.5];
  const sS: Slider = slider(panel, "pre-smoothing σ", 0, SIG.length - 1, 1, SIG.indexOf(st.sigma), v => { st.sigma = SIG[v]; makeEdges(); }, v => String(SIG[v]));
  const nS: Slider = slider(panel, "noise", 0, 30, 1, st.noise, v => { st.noise = v; makeImage(); });
  const r3 = htmlEl("div", "qb-presets", "", panel);
  const cb = [4, 8].map(c => button(r3, c + " neighbours", () => { st.conn = c as 4 | 8; makeEdges(); }));
  button(r3, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Click a region for its internal difference Int(C) (heaviest edge of its spanning tree), size and threshold Int(C) + k/|C|. Two neighbouring regions merge across their cheapest edge d only if d is below both thresholds.", panel);

  function sync(): void {
    ib.forEach((b, i) => b.classList.toggle("active", ["synth", "cell", "colour"][i] === st.img));
    mb.forEach((b, i) => b.classList.toggle("active", ["fh", "single"][i] === st.method));
    vb.forEach((b, i) => b.classList.toggle("active", ["random", "mean"][i] === st.view));
    cb.forEach((b, i) => b.classList.toggle("active", [4, 8][i] === st.conn));
  }
  function load(p: Preset): void {
    Object.assign(st, DEF, p.s);
    click = p.click ?? null;
    kS.set(Math.log10(st.k / 10)); tS.set(st.t); mS.set(st.minSize); sS.set(SIG.indexOf(st.sigma)); nS.set(st.noise);
    pb.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    makeImage();
  }

  // ---- data ----
  let cell: Float64Array | null = null;
  let rgb: [Float64Array, Float64Array, Float64Array] | null = null, I: Float64Array = new Float64Array(N * N);
  let G: DGraph | null = null, order: Uint32Array | null = null, sortMs = 0;
  let res: FHResult | { labels: Int32Array; count: number } | null = null, segMs = 0;
  function makeImage(): void {
    sync();
    if (st.img === "cell") {
      if (!cell) { loadCell(N).then(g => { cell = Float64Array.from(g.data); makeImage(); }); return; }
      I = cell.slice(); rgb = null;
      if (st.noise > 0) { const r = rng(21); for (let p = 0; p < I.length; p++) I[p] = Math.min(255, Math.max(0, I[p] + st.noise * r.norm())); }
    } else if (st.img === "synth") { I = fhScene(st.noise).I; rgb = null; }
    else { const s = colourScene(st.noise); I = s.I; rgb = s.rgb!; }
    credit.style.visibility = st.img === "cell" ? "visible" : "hidden";
    makeEdges();
  }
  function makeEdges(): void {
    sync();
    const t0 = performance.now();
    const blur = (a: Float64Array) => (st.sigma > 0 ? Float64Array.from(gaussianBlur(fromArray(N, N, a), st.sigma).data) : a);
    if (rgb) {
      const R = blur(rgb[0]), Gc = blur(rgb[1]), B = blur(rgb[2]);
      const L = new Float64Array(N * N), A = new Float64Array(N * N), Bb = new Float64Array(N * N);
      for (let p = 0; p < N * N; p++) { const lab = srgbToLab([R[p] / 255, Gc[p] / 255, B[p] / 255]); L[p] = lab[0]; A[p] = lab[1]; Bb[p] = lab[2]; }
      G = gridDissim([L, A, Bb], N, N, st.conn);
    } else G = gridDissim([blur(I)], N, N, st.conn);
    order = sortEdges(G);
    sortMs = performance.now() - t0;
    segment();
  }
  function segment(): void {
    sync();
    if (!G || !order) return;
    const t0 = performance.now();
    if (st.method === "fh") res = felzenszwalb(G, st.k, st.minSize, order);
    else {
      const sl = singleLinkage(G, st.t);
      res = sl;
      if (st.minSize > 1) {   // same clean-up pass as FH
        const fh = felzenszwalb({ ...G, d: G.d.map(d => (d < st.t ? -1 : d)) }, 0, st.minSize, order);
        res = fh;
      }
    }
    segMs = performance.now() - t0;
    draw();
  }

  function draw(): void {
    if (!res) return;
    const lab = res.labels, n = res.count;
    const ctxR = cvR.getContext("2d")!, idR = ctxR.createImageData(N, N);
    let mean: Float64Array[] | null = null;
    if (st.view === "mean") {
      const ch = rgb ? [rgb[0], rgb[1], rgb[2]] : [I, I, I];
      mean = ch.map(() => new Float64Array(n + 1));
      const cnt = new Float64Array(n + 1);
      for (let p = 0; p < N * N; p++) { cnt[lab[p]]++; for (let j = 0; j < 3; j++) mean[j][lab[p]] += ch[j][p]; }
      for (let l = 1; l <= n; l++) for (let j = 0; j < 3; j++) mean[j][l] /= cnt[l] || 1;
    }
    const sel = click ? lab[click[1] * N + click[0]] : -1;
    for (let p = 0; p < N * N; p++) {
      const l = lab[p];
      let c: number[];
      if (mean) c = [mean[0][l], mean[1][l], mean[2][l]];
      else { const q = labelColor(((l * 7919) % 997) + 1); c = [q[0] * 255, q[1] * 255, q[2] * 255]; }
      if (l === sel && (p % 4 === 0) && (Math.floor(p / N) % 4 === 0)) c = [0, 0, 0];
      idR.data[4 * p] = c[0]; idR.data[4 * p + 1] = c[1]; idR.data[4 * p + 2] = c[2]; idR.data[4 * p + 3] = 255;
    }
    ctxR.putImageData(idR, 0, 0);
    const ctxB = cvB.getContext("2d")!, idB = ctxB.createImageData(N, N);
    for (let p = 0; p < N * N; p++) {
      const x = p % N, y = (p - x) / N, l = lab[p];
      const edge = (x + 1 < N && lab[p + 1] !== l) || (y + 1 < N && lab[p + N] !== l);
      let c = rgb ? [rgb[0][p], rgb[1][p], rgb[2][p]] : [I[p], I[p], I[p]];
      if (edge) c = [255, 230, 0];
      idB.data[4 * p] = c[0]; idB.data[4 * p + 1] = c[1]; idB.data[4 * p + 2] = c[2]; idB.data[4 * p + 3] = 255;
    }
    ctxB.putImageData(idB, 0, 0);
    cap1.textContent = st.method === "fh" ? `FH, k = ${st.k}: ${n} regions` : `single linkage, t = ${st.t}: ${n} regions`;
    const lines = [`${n} regions · sort ${sortMs.toFixed(0)} ms (once), merge pass ${segMs.toFixed(0)} ms`];
    if (click && "int" in res) {
      const fh = res as FHResult, l = sel;
      lines.push(`clicked region (${click[0]}, ${click[1]}): |C| = ${fh.size[l]} px, Int(C) = ${fh.int[l].toFixed(1)}`, `threshold Int + k/|C| = ${(fh.int[l] + st.k / fh.size[l]).toFixed(2)}`);
    } else if (click) {
      let s = 0; for (let p = 0; p < N * N; p++) if (lab[p] === sel) s++;
      lines.push(`clicked region (${click[0]}, ${click[1]}): ${s} px`);
    }
    ro.textContent = lines.join("\n");
  }
  const pick = (cv: HTMLCanvasElement) => (ev: PointerEvent) => {
    const r = cv.getBoundingClientRect();
    click = [Math.min(N - 1, Math.max(0, Math.floor((ev.clientX - r.left) / r.width * N))), Math.min(N - 1, Math.max(0, Math.floor((ev.clientY - r.top) / r.height * N)))];
    draw();
  };
  cvR.addEventListener("pointerdown", pick(cvR));
  cvB.addEventListener("pointerdown", pick(cvB));

  load(p0);
}

autoMount(NAME, mount);
