// SLIC superpixels (graph-based segmentation day, unit 5): k-means on (colour, x, y) with
// centres on a grid of spacing S = sqrt(N/K), distance D² = d_c² + (m d_s / S)², each centre
// searching a 2S × 2S window. Step through the iterations (centres move), switch the
// connectivity clean-up off to see orphan fragments, compare with FH and marker-watershed
// superpixels at about the same K, and measure them against the ground truth of the test
// scenes (boundary recall, under-segmentation error, achievable accuracy). "use them": the
// same seeds segmented by a graph cut on the pixel graph and on the superpixel graph.
// <div data-demo="slic" data-preset="grid|msmall|mlarge|orphans|methods|use|thin|cell"></div>
import { autoMount, htmlEl, slider, button, presets, checkbox, Slider } from "../lib/ui";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { gradient, fromArray } from "../lib/imageops";
import { srgbToLab } from "../lib/colour";
import { slic, SlicResult, gridMarkers, boundaryRecall, undersegmentationError, achievableAccuracy, spGraph, relabel } from "../lib/superpixels";
import { gridDissim, felzenszwalb, sortEdges, DGraph } from "../lib/mst";
import { watershed } from "../lib/watershed";
import { gridEdges, contrastWeights, solveSeg } from "../lib/graph";
import { Plot } from "../lib/plot";
import { colourScene, fhScene, Scene } from "../fhseg/scenes";

const NAME = "slic";
const N = 256;
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .6em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; flex: 0 1 360px; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin: 0 0 .15em; }
.qb-demo[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; image-rendering: pixelated; }
.qb-demo[data-demo="${NAME}"] .sl-main canvas { cursor: crosshair; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 380px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] .sl-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .sl-credit { font-size: 12px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .sl-plot { flex: 1 1 100%; max-width: 760px; order: 5; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] figure { flex: 0 0 360px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 380px; gap: .12em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .02em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .1em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .sl-lab, .reveal .qb-demo[data-demo="${NAME}"] .qb-note, .reveal .qb-demo[data-demo="${NAME}"] .sl-plot { display: none; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

type Img = "colour" | "scene" | "cell";
type Method = "slic" | "fh" | "ws";
type View = "mean" | "use";
interface Settings { img: Img; method: Method; K: number; m: number; iters: number; conn: boolean; view: View }
interface Preset { id: string; label: string; s: Partial<Settings>; select?: [number, number] }
const DEF: Settings = { img: "colour", method: "slic", K: 200, m: 20, iters: 10, conn: true, view: "mean" };
const PRESETS: Preset[] = [
  { id: "grid", label: "grid start", s: { iters: 0, K: 100 }, select: [128, 128] },
  { id: "msmall", label: "m small: follows edges", s: { m: 6 } },
  { id: "mlarge", label: "m large: squares", s: { m: 80 } },
  { id: "orphans", label: "orphans without connectivity", s: { m: 6, conn: false } },
  { id: "methods", label: "same K, three methods", s: { method: "fh" } },
  { id: "use", label: "graph cut on superpixels", s: { view: "use", K: 400 } },
  { id: "thin", label: "thin line lost", s: { img: "scene", K: 60, m: 20 } },
  { id: "cell", label: "micrograph", s: { img: "cell", K: 300, m: 20 } },
];
const Klog = (u: number) => Math.round(Math.pow(10, u));

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: Settings = { ...DEF, ...p0.s };
  let select: [number, number] | null = p0.select ?? null;

  const f1 = htmlEl("figure", "sl-main", "", host);
  const cap1 = htmlEl("figcaption", "", "", f1);
  const cvA = htmlEl("canvas", "", "", f1); cvA.width = N * 2; cvA.height = N * 2;
  const f2 = htmlEl("figure", "", "", host);
  const cap2 = htmlEl("figcaption", "", "", f2);
  const cvB = htmlEl("canvas", "", "", f2); cvB.width = N; cvB.height = N;
  const credit = htmlEl("div", "sl-credit", CELL_CREDIT, f2);
  const panel = htmlEl("div", "qb-panel", "", host);
  const plotBox = htmlEl("div", "sl-plot", "", host);
  htmlEl("div", "sl-lab", "presets", panel);
  const pb = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const r1 = htmlEl("div", "qb-presets", "", panel);
  const ib = (["colour", "scene", "cell"] as Img[]).map(i => button(r1, i === "colour" ? "colour scene" : i === "scene" ? "grey scene" : "micrograph", () => { st.img = i; makeImage(); }));
  const r2 = htmlEl("div", "qb-presets", "", panel);
  const mb = (["slic", "fh", "ws"] as Method[]).map(m => button(r2, m === "slic" ? "SLIC" : m === "fh" ? "FH" : "watershed", () => { st.method = m; compute(); }));
  const vb = (["mean", "use"] as View[]).map(v => button(r2, v === "mean" ? "mean colours" : "use them", () => { st.view = v; draw(); }));
  const kS: Slider = slider(panel, "K", Math.log10(25), 3, 0.01, Math.log10(st.K), u => { st.K = Klog(u); compute(); }, u => String(Klog(u)));
  const mS: Slider = slider(panel, "compactness m", 1, 80, 1, st.m, v => { st.m = v; compute(); });
  const itS: Slider = slider(panel, "iterations", 0, 10, 1, st.iters, v => { st.iters = v; draw(); });
  const connC = checkbox(panel, "connectivity clean-up", st.conn, v => { st.conn = v; compute(); });
  const r3 = htmlEl("div", "qb-presets", "", panel);
  button(r3, "step", () => { st.iters = Math.min(10, st.iters + 1); itS.set(st.iters); draw(); });
  button(r3, "curves vs K", () => curves());
  button(r3, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Click a superpixel to see its centre's 2S × 2S search window. Metrics (test scenes only): boundary recall = true boundary pixels within 2 px of a superpixel boundary; under-segmentation error = leakage of superpixels across true regions; achievable accuracy = the best any superpixel-constant labelling can do.", panel);

  function sync(): void {
    ib.forEach((b, i) => b.classList.toggle("active", ["colour", "scene", "cell"][i] === st.img));
    mb.forEach((b, i) => b.classList.toggle("active", ["slic", "fh", "ws"][i] === st.method));
    vb.forEach((b, i) => b.classList.toggle("active", ["mean", "use"][i] === st.view));
  }
  function load(p: Preset): void {
    Object.assign(st, DEF, p.s);
    select = p.select ?? null;
    kS.set(Math.log10(st.K)); mS.set(st.m); itS.set(st.iters); connC.checked = st.conn;
    pb.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    makeImage();
  }

  // ---- data ----
  let cell: Float64Array | null = null, scene: Scene | null = null, ch: Float64Array[] = [], I: Float64Array = new Float64Array(N * N);
  let rgb: Float64Array[] = [], grad: Float64Array = new Float64Array(N * N), G: DGraph | null = null, order: Uint32Array | null = null;
  function makeImage(): void {
    sync();
    if (st.img === "cell") {
      if (!cell) { loadCell(N).then(g => { cell = Float64Array.from(g.data); makeImage(); }); return; }
      scene = null; I = cell; rgb = [I, I, I]; ch = [Float64Array.from(I, v => v * 100 / 255)];
    } else {
      scene = st.img === "colour" ? colourScene(6) : fhScene(4);
      I = scene.I;
      if (scene.rgb) {
        rgb = scene.rgb;
        const L = new Float64Array(N * N), A = new Float64Array(N * N), B = new Float64Array(N * N);
        for (let p = 0; p < N * N; p++) { const lab = srgbToLab([rgb[0][p] / 255, rgb[1][p] / 255, rgb[2][p] / 255]); L[p] = lab[0]; A[p] = lab[1]; B[p] = lab[2]; }
        ch = [L, A, B];
      } else { rgb = [I, I, I]; ch = [Float64Array.from(I, v => v * 100 / 255)]; }
    }
    credit.style.visibility = st.img === "cell" ? "visible" : "hidden";
    grad = Float64Array.from(gradient(fromArray(N, N, I), 1).mag.data);
    G = gridDissim(ch, N, N, 8); order = sortEdges(G);
    compute();
  }

  let res: SlicResult | null = null, labels: Int32Array<ArrayBufferLike> = new Int32Array(N * N), count = 0, ms = 0, info = "";
  function fhForK(K: number): { labels: Int32Array<ArrayBufferLike>; count: number; k: number } {
    let lo = 1, hi = 100000, best: { labels: Int32Array<ArrayBufferLike>; count: number; k: number } = { labels: new Int32Array(N * N), count: 0, k: 1 };
    const min = Math.max(5, Math.floor(N * N / K / 10));
    for (let it = 0; it < 18; it++) {
      const k = Math.sqrt(lo * hi), r = felzenszwalb(G!, k, min, order!);
      best = { labels: r.labels, count: r.count, k };
      if (Math.abs(r.count - K) <= Math.max(2, 0.05 * K)) break;
      if (r.count > K) lo = k; else hi = k;
    }
    return best;
  }
  function compute(): void {
    sync();
    const t0 = performance.now();
    res = null;
    if (st.method === "slic") {
      res = slic(ch, N, N, { K: st.K, m: st.m, iters: 10, connectivity: st.conn, trace: true });
      labels = res.labels; count = res.count;
      info = `SLIC: K = ${st.K}, S = ${res.S.toFixed(1)} px, m = ${st.m}${st.conn ? `; ${res.orphans} orphan px relabelled` : ""}`;
    } else if (st.method === "fh") {
      const r = fhForK(st.K);
      labels = r.labels; count = r.count;
      info = `FH with k = ${r.k.toFixed(0)} (chosen for about K regions)`;
    } else {
      labels = watershed(grad, N, N, gridMarkers(N, N, st.K), { conn: 8 });
      const rr = relabel(labels); labels = rr.labels; count = rr.count;
      info = "watershed of the gradient from grid markers";
    }
    ms = performance.now() - t0;
    gc = null;
    draw();
  }

  // ---- use them: graph cut on pixels vs superpixels ----
  let gc: { pix: Uint8Array; sp: Uint8Array; tPix: number; tSp: number; nodes: number; edges: number } | null = null;
  function seedMask(): Int8Array {
    const s = new Int8Array(N * N);
    const disc = (cx: number, cy: number, r: number, v: number) => { for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) if (x * x + y * y <= r * r) { const xx = cx + x, yy = cy + y; if (xx >= 0 && yy >= 0 && xx < N && yy < N) s[yy * N + xx] = v; } };
    if (st.img === "colour") { disc(75, 80, 14, 1); disc(10, 10, 6, -1); disc(128, 128, 6, -1); disc(250, 128, 6, -1); disc(75, 140, 5, -1); disc(240, 245, 6, -1); }
    else if (st.img === "scene") { disc(70, 70, 14, 1); disc(10, 10, 6, -1); disc(128, 128, 6, -1); disc(245, 245, 6, -1); disc(160, 70, 6, -1); }
    else { disc(122, 132, 12, 1); disc(10, 10, 6, -1); disc(245, 10, 6, -1); disc(10, 245, 6, -1); disc(245, 245, 6, -1); }
    return s;
  }
  function useThem(): void {
    const seeds = seedMask(), sigma = 15;
    let t = performance.now();
    const E = gridEdges(N, N, 4);
    const pix = solveSeg({ n: N * N, D0: new Float64Array(N * N), D1: new Float64Array(N * N), E, w: contrastWeights(I, E, sigma), lambda: 1, seeds }).L;
    const tPix = performance.now() - t;
    t = performance.now();
    const g = spGraph(labels, N, N, I);
    const sps = new Int8Array(g.n);
    for (let p = 0; p < N * N; p++) if (seeds[p]) sps[labels[p] - 1] = seeds[p];
    const w = new Float64Array(g.a.length);
    for (let e = 0; e < w.length; e++) w[e] = g.len[e] * Math.exp(-((g.mean[g.a[e]] - g.mean[g.b[e]]) ** 2) / (2 * sigma * sigma));
    const Esp = { p: g.a, q: g.b, geo: new Float64Array(g.a.length).fill(1) };
    const L = solveSeg({ n: g.n, D0: new Float64Array(g.n), D1: new Float64Array(g.n), E: Esp, w, lambda: 1, seeds: sps }).L;
    const sp = Uint8Array.from(labels, l => L[l - 1]);
    const tSp = performance.now() - t;
    gc = { pix, sp, tPix, tSp, nodes: g.n, edges: g.a.length };
  }

  // ---- drawing ----
  function draw(): void {
    sync();
    const it = st.method === "slic" ? st.iters : 10;
    const showLab = st.method === "slic" && res && it < 10 && res.traceLabels ? res.traceLabels[it] : labels;
    const ctx = cvA.getContext("2d")!, id = ctx.createImageData(N, N);
    for (let p = 0; p < N * N; p++) {
      const x = p % N, y = (p - x) / N, l = showLab[p];
      const edge = it > 0 && ((x + 1 < N && showLab[p + 1] !== l) || (y + 1 < N && showLab[p + N] !== l));
      let c = [rgb[0][p], rgb[1][p], rgb[2][p]];
      if (edge) c = [255, 235, 0];
      id.data[4 * p] = c[0]; id.data[4 * p + 1] = c[1]; id.data[4 * p + 2] = c[2]; id.data[4 * p + 3] = 255;
    }
    const tmp = document.createElement("canvas"); tmp.width = N; tmp.height = N; tmp.getContext("2d")!.putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.drawImage(tmp, 0, 0, 2 * N, 2 * N);
    const sc = 2;
    if (st.method === "slic" && res && res.traceCentres) {
      const cen = res.traceCentres[it], C = ch.length;
      ctx.fillStyle = "#d62728";
      for (const c of cen) { ctx.beginPath(); ctx.arc(c[C] * sc + 1, c[C + 1] * sc + 1, 3.5, 0, 2 * Math.PI); ctx.fill(); }
      if (select) {
        const k0 = it === 0 ? nearestCentre(cen, C, select) : showLab[select[1] * N + select[0]] - 1;
        const c = cen[Math.max(0, k0)];
        if (c) {
          const S = res.S;
          ctx.strokeStyle = "#d62728"; ctx.lineWidth = 3;
          ctx.strokeRect((c[C] - S) * sc, (c[C + 1] - S) * sc, 2 * S * sc, 2 * S * sc);
          ctx.fillStyle = "#000"; ctx.beginPath(); ctx.arc(c[C] * sc + 1, c[C + 1] * sc + 1, 6, 0, 2 * Math.PI); ctx.fill();
        }
      }
    }
    cap1.textContent = st.method === "slic" ? (it === 0 ? "iteration 0: centres on the grid" : `SLIC after ${it} iteration${it > 1 ? "s" : ""}: ${it < 10 ? "before clean-up" : count + " superpixels"}`) : `${st.method === "fh" ? "FH" : "watershed"}: ${count} superpixels`;
    // second panel
    const cb = cvB.getContext("2d")!, ib2 = cb.createImageData(N, N);
    if (st.view === "mean") {
      const n = count, S = [new Float64Array(n + 1), new Float64Array(n + 1), new Float64Array(n + 1)], cnt = new Float64Array(n + 1);
      for (let p = 0; p < N * N; p++) { const l = labels[p]; cnt[l]++; for (let j = 0; j < 3; j++) S[j][l] += rgb[j][p]; }
      for (let p = 0; p < N * N; p++) { const l = labels[p]; for (let j = 0; j < 3; j++) ib2.data[4 * p + j] = S[j][l] / cnt[l]; ib2.data[4 * p + 3] = 255; }
      cap2.textContent = "each superpixel in its mean colour";
    } else {
      if (!gc) useThem();
      const g = gc!;
      for (let p = 0; p < N * N; p++) {
        const a = g.pix[p], b = g.sp[p];
        const base = rgb[0][p] * 0.5;
        const c = a && b ? [255, 160, 40] : a ? [255, 60, 60] : b ? [60, 120, 255] : [base, base, base];
        for (let j = 0; j < 3; j++) ib2.data[4 * p + j] = c[j]; ib2.data[4 * p + 3] = 255;
      }
      cap2.textContent = "graph cut: orange both, red pixels only, blue superpixels only";
    }
    cb.putImageData(ib2, 0, 0);
    const lines = [info, `${count} superpixels in ${ms.toFixed(0)} ms`];
    if (scene && st.view === "mean") {
      lines.push(`boundary recall (2 px): ${boundaryRecall(labels, scene.truth, N, N, 2).toFixed(3)}`,
        `under-segmentation error: ${undersegmentationError(labels, scene.truth).toFixed(3)}`,
        `achievable accuracy: ${achievableAccuracy(labels, scene.truth).toFixed(4)}`);
    }
    if (st.view === "use" && gc) lines.push(`pixel graph: ${N * N} nodes, ${(gridEdges(N, N, 4).p.length)} edges, ${gc.tPix.toFixed(0)} ms`, `superpixel graph: ${gc.nodes} nodes, ${gc.edges} edges, ${gc.tSp.toFixed(1)} ms (+ ${ms.toFixed(0)} ms superpixels)`);
    ro.textContent = lines.join("\n");
  }
  function nearestCentre(cen: Float64Array[], C: number, s: [number, number]): number {
    let b = 0, bd = Infinity;
    cen.forEach((c, k) => { const d = (c[C] - s[0]) ** 2 + (c[C + 1] - s[1]) ** 2; if (d < bd) { bd = d; b = k; } });
    return b;
  }
  cvA.addEventListener("pointerdown", ev => {
    const r = cvA.getBoundingClientRect();
    select = [Math.min(N - 1, Math.max(0, Math.floor((ev.clientX - r.left) / r.width * N))), Math.min(N - 1, Math.max(0, Math.floor((ev.clientY - r.top) / r.height * N)))];
    draw();
  });

  // ---- curves against K (computed on request) ----
  function curves(): void {
    if (!scene) { plotBox.textContent = "Curves need a test scene with ground truth."; return; }
    plotBox.textContent = "computing…";
    const Ks = [50, 100, 200, 400, 800], out: Record<Method, number[][]> = { slic: [], fh: [], ws: [] };
    const saved = { method: st.method, K: st.K };
    let i = 0;
    const step = () => {
      const K = Ks[Math.floor(i / 3)], m = (["slic", "fh", "ws"] as Method[])[i % 3];
      st.method = m; st.K = K;
      let lab: Int32Array;
      if (m === "slic") lab = slic(ch, N, N, { K, m: st.m, iters: 10 }).labels;
      else if (m === "fh") lab = fhForK(K).labels;
      else lab = relabel(watershed(grad, N, N, gridMarkers(N, N, K), { conn: 8 })).labels;
      out[m].push([K, boundaryRecall(lab, scene!.truth, N, N, 2), undersegmentationError(lab, scene!.truth)]);
      i++;
      if (i < Ks.length * 3) { setTimeout(step, 0); return; }
      st.method = saved.method; st.K = saved.K;
      plotBox.innerHTML = "";
      const pl = new Plot(plotBox, { width: 760, height: 260, x: [40, 1000], xlog: true, y: [0, 1], xlabel: "number of superpixels K", ylabel: "boundary recall / UE", font: 16 });
      const cols: Record<Method, string> = { slic: "#d62728", fh: "#2ca02c", ws: "#1f77b4" };
      (["slic", "fh", "ws"] as Method[]).forEach(mm => {
        pl.line(out[mm].map(r => r[0]), out[mm].map(r => r[1]), { color: cols[mm], label: (mm === "slic" ? "SLIC" : mm === "fh" ? "FH" : "watershed") + " recall" });
        pl.line(out[mm].map(r => r[0]), out[mm].map(r => r[2]), { color: cols[mm], dash: "6 4", label: "UE" });
      });
      pl.legend(undefined, "br");
    };
    setTimeout(step, 0);
  }

  load(p0);
}

autoMount(NAME, mount);
