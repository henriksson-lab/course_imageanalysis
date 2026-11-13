// The random walker on a 128 × 128 image (graph-based segmentation day, units 1 and 3).
// Paint seeds (object, background, optionally a third label); every unseeded pixel gets the
// probability that a random walk started there reaches each seed label first, from one
// sparse linear system per label (conjugate gradients, solved a little per frame, warm-started).
// Panels: the image with seeds and the label boundary; the probability of "object"
// (diverging map, 0.5 = white), the probabilities as colours (3 labels), or the uncertainty
// 1 − max p. Click the right panel to probe a pixel: its probability and 30 seeded walks.
// Options: σ of the contrast weights, 4/8 neighbours, priors (an intensity model from the
// seeds, weight λ). data-mode="compare": the same seeds and weights segmented three ways:
// random walker (squares), graph cut with seeds only (sum, lib/graph.ts) and seeded watershed
// = minimum spanning forest (max, lib/mst.ts).
// <div data-demo="randwalk" data-preset="..." data-mode="compare"></div>
import { autoMount, htmlEl, slider, button, presets, checkbox } from "../lib/ui";
import { LabelPainter, paintSegment } from "../lib/paint";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { gridGraph, WalkerSolver, seedPriors, csr, simulateWalks, WalkerOptions, WGraph } from "../lib/randomwalker";
import { gridEdges, contrastWeights, solveSeg } from "../lib/graph";
import { gridDissim, primForest } from "../lib/mst";
import { diverging, inferno } from "../lib/colour";
import { rng } from "../lib/random";
import { walkerScene } from "../fhseg/scenes";

const NAME = "randwalk";
const N = 128;
const SIGMAS = [3, 5, 8, 12, 20, 30, 50, 80, Infinity];
const COLS: [number, number, number][] = [[255, 127, 14], [31, 119, 180], [44, 160, 44]];
const LABELS = [
  { value: 1, name: "object", color: "#ff7f0e" },
  { value: 2, name: "background", color: "#1f77b4" },
  { value: 3, name: "label 3", color: "#2ca02c" },
];
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .7em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .qb-paint { flex: 0 1 380px; }
.qb-demo[data-demo="${NAME}"] .rw-right { flex: 0 1 380px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .rw-right canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; cursor: crosshair; }
.qb-demo[data-demo="${NAME}"] .rw-cmp { flex: 0 1 230px; display: flex; flex-direction: column; gap: .2em; }
.qb-demo[data-demo="${NAME}"] .rw-cmp canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin: 0 0 .15em; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 360px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] .rw-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .rw-credit { font-size: 12px; color: var(--qb-muted); }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-paint { flex: 0 0 360px; }
.reveal .qb-demo[data-demo="${NAME}"] .rw-right { flex: 0 0 360px; }
.reveal .qb-demo[data-demo="${NAME}"] .rw-cmp { flex: 0 0 225px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 340px; gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .03em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .15em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .rw-lab, .reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

type SceneId = "objects" | "gap" | "ramp" | "cell";
type View = "prob" | "rgb" | "unc";
interface Stroke { l: number; r: number; pts: [number, number][] }
interface Preset { id: string; label: string; scene: SceneId; noise: number; sigma: number; strokes: Stroke[]; priors?: boolean; lambda?: number; view?: View; probe?: [number, number]; conn?: 4 | 8 }
const ring = (cx: number, cy: number, rr: number, l = 2): Stroke => ({ l, r: 1, pts: Array.from({ length: 41 }, (_, i) => [cx + rr * Math.cos(i * Math.PI / 20), cy + rr * Math.sin(i * Math.PI / 20)] as [number, number]) });
const dot = (x: number, y: number, l: number, r = 2): Stroke => ({ l, r, pts: [[x, y]] });
const PRESETS: Preset[] = [
  { id: "clicks", label: "two clicks", scene: "objects", noise: 10, sigma: 20, strokes: [dot(50, 46, 1), dot(10, 10, 2)], probe: [100, 100] },
  { id: "three", label: "three labels", scene: "objects", noise: 10, sigma: 20, strokes: [dot(50, 46, 1), dot(52, 92, 3), dot(10, 10, 2), dot(118, 118, 2)], view: "rgb" },
  { id: "gap", label: "weak boundary: the walker bridges the gap", scene: "gap", noise: 8, sigma: 20, strokes: [{ l: 1, r: 2, pts: [[30, 20], [30, 108]] }, { l: 2, r: 2, pts: [[98, 20], [98, 108]] }], probe: [64, 63] },
  { id: "thin", label: "thin protrusion", scene: "objects", noise: 10, sigma: 20, strokes: [{ l: 1, r: 2, pts: [[40, 40], [60, 50]] }, { l: 1, r: 2, pts: [[45, 85], [60, 92]] }, ring(64, 64, 60), dot(112, 52, 2), dot(112, 28, 2)] },
  { id: "far", label: "seeds far away: fading to 0.5", scene: "ramp", noise: 4, sigma: Infinity, strokes: [dot(4, 4, 1, 1), dot(123, 123, 2, 1)], probe: [64, 64] },
  { id: "ramp", label: "no edge between two brightnesses: fails", scene: "ramp", noise: 6, sigma: 20, strokes: [{ l: 1, r: 2, pts: [[6, 30], [6, 98]] }, { l: 2, r: 2, pts: [[86, 30], [86, 98]] }] },
  { id: "priors", label: "… priors fix it", scene: "ramp", noise: 6, sigma: 20, strokes: [{ l: 1, r: 2, pts: [[6, 30], [6, 98]] }, { l: 2, r: 2, pts: [[86, 30], [86, 98]] }], priors: true, lambda: 0.05 },
  { id: "lowsigma", label: "σ too small: noise edges trap the walker", scene: "objects", noise: 18, sigma: 5, strokes: [dot(50, 46, 1, 3), dot(10, 10, 2, 3), dot(118, 118, 2, 3)] },
  { id: "cell", label: "micrograph", scene: "cell", noise: 0, sigma: 12, strokes: [{ l: 1, r: 2, pts: [[52, 60], [70, 66]] }, { l: 2, r: 2, pts: [[8, 8], [8, 120]] }, { l: 2, r: 2, pts: [[120, 8], [120, 120]] }, { l: 2, r: 2, pts: [[28, 112], [100, 120]] }] },
];
const CMP_PRESETS: Preset[] = [
  { id: "agree", label: "all three agree on a clean edge", scene: "objects", noise: 6, sigma: 20, strokes: [dot(50, 46, 1, 3), dot(10, 10, 2, 3), dot(118, 118, 2, 3), dot(10, 118, 2, 3)] },
  { id: "gap", label: "a gap in a wall, strong noise", scene: "gap", noise: 16, sigma: 20, strokes: [dot(20, 64, 1, 3), dot(120, 120, 2, 2)] },
  { id: "smallseed", label: "no edge, a small seed", scene: "ramp", noise: 4, sigma: 20, strokes: [dot(40, 64, 1, 3), dot(124, 64, 2, 3)] },
  { id: "weak", label: "two discs, a weak boundary", scene: "objects", noise: 8, sigma: 20, strokes: [dot(50, 46, 1, 3), dot(52, 92, 2, 3), dot(10, 10, 2, 3), dot(118, 118, 2, 3)] },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const compare = host.dataset.mode === "compare";
  const list = compare ? CMP_PRESETS : PRESETS;
  const p0 = list.find(p => p.id === host.dataset.preset) ?? list[0];
  const st = { scene: p0.scene, noise: p0.noise, sigma: p0.sigma, conn: (p0.conn ?? 4) as 4 | 8, priors: !!p0.priors, lambda: p0.lambda ?? 0.05, view: (p0.view ?? "prob") as View };
  let probe: [number, number] | null = p0.probe ?? null;

  const painter = new LabelPainter(host, { width: N, height: N, labels: LABELS, brush: 2, controls: false, caption: compare ? "random walker: paint seeds here" : "image and seeds: paint here", onChange: (_m, fin) => { if (fin) { clearPresetHighlight(); run(); } } });
  const credit = htmlEl("div", "rw-credit", CELL_CREDIT, painter.root);
  let right: HTMLElement, cv: HTMLCanvasElement, cap: HTMLElement;
  let cvGC: HTMLCanvasElement | null = null, cvWS: HTMLCanvasElement | null = null;
  if (!compare) {
    right = htmlEl("figure", "rw-right", "", host);
    cap = htmlEl("figcaption", "", "", right);
    cv = htmlEl("canvas", "", "", right); cv.width = N * 2; cv.height = N * 2;
  } else {
    right = htmlEl("div", "rw-cmp", "", host);
    const f1 = htmlEl("figure", "", "", right); htmlEl("figcaption", "", "graph cut (sum)", f1);
    cvGC = htmlEl("canvas", "", "", f1); cvGC.width = N; cvGC.height = N;
    const f2 = htmlEl("figure", "", "", right); htmlEl("figcaption", "", "watershed / MSF (max)", f2);
    cvWS = htmlEl("canvas", "", "", f2); cvWS.width = N; cvWS.height = N;
    cap = htmlEl("div", "", "", right); cv = htmlEl("canvas", "", ""); // unused
  }
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "rw-lab", "presets", panel);
  const pb = presets(panel, list.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "rw-lab", "paint", panel);
  const prow = htmlEl("div", "qb-presets", "", panel);
  const lb = LABELS.map(l => { const b = button(prow, l.name, () => { painter.setActive(l.value); syncBtns(); }, "qb-paint-label"); b.style.setProperty("--qb-label", l.color); return b; });
  const eb = button(prow, "erase", () => { painter.setActive(0); syncBtns(); });
  button(prow, "undo", () => painter.undo());
  button(prow, "clear", () => painter.clear());
  htmlEl("div", "rw-lab", "image", panel);
  const irow = htmlEl("div", "qb-presets", "", panel);
  const ib = (["objects", "gap", "ramp", "cell"] as SceneId[]).map(s => button(irow, s === "cell" ? "micrograph" : s, () => { st.scene = s; makeImage(); }));
  const noiseS = slider(panel, "noise", 0, 30, 1, st.noise, v => { st.noise = v; makeImage(); });
  const sigS = slider(panel, "σ", 0, SIGMAS.length - 1, 1, SIGMAS.indexOf(st.sigma), v => { st.sigma = SIGMAS[v]; run(); }, v => (Number.isFinite(SIGMAS[v]) ? String(SIGMAS[v]) : "∞"));
  const orow = htmlEl("div", "qb-presets", "", panel);
  const nb = [4, 8].map(k => button(orow, k + " neighbours", () => { st.conn = k as 4 | 8; run(); }));
  let vb: HTMLButtonElement[] = [];
  if (!compare) vb = (["prob", "rgb", "unc"] as View[]).map(v => button(orow, v === "prob" ? "p(object)" : v === "rgb" ? "all labels" : "uncertainty", () => { st.view = v; draw(); }));
  const priC = compare ? null : checkbox(panel, "priors from the seeds' histograms", st.priors, v => { st.priors = v; run(); });
  const lamS = compare ? null : slider(panel, "λ (priors)", -4, 0, 0.1, Math.log10(st.lambda), v => { st.lambda = 10 ** v; if (st.priors) run(); }, v => (10 ** v).toPrecision(2));
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", compare
    ? "Same seeds, same contrast weights w = exp(−ΔI²/2σ²): the random walker minimises Σ w (x_p − x_q)², the graph cut Σ w over cut edges, the watershed takes the path with the lowest highest edge (dissimilarity |ΔI|)."
    : "Paint seeds; the system is re-solved (warm start) after every stroke. Click the right panel to probe a pixel: its probability and 30 random walks from it (white; dots: where they end).", panel);

  function syncBtns(): void {
    lb.forEach((b, i) => b.classList.toggle("active", painter.active === LABELS[i].value));
    eb.classList.toggle("active", painter.active === 0);
    ib.forEach((b, i) => b.classList.toggle("active", ["objects", "gap", "ramp", "cell"][i] === st.scene));
    nb.forEach((b, i) => b.classList.toggle("active", [4, 8][i] === st.conn));
    vb.forEach((b, i) => b.classList.toggle("active", ["prob", "rgb", "unc"][i] === st.view));
  }
  function clearPresetHighlight(): void { pb.forEach(b => b.classList.remove("active")); }

  function load(p: Preset): void {
    Object.assign(st, { scene: p.scene, noise: p.noise, sigma: p.sigma, conn: p.conn ?? 4, priors: !!p.priors, lambda: p.lambda ?? 0.05, view: p.view ?? "prob" });
    probe = p.probe ?? null;
    noiseS.set(st.noise); sigS.set(SIGMAS.indexOf(st.sigma));
    if (priC) priC.checked = st.priors;
    lamS?.set(Math.log10(st.lambda));
    const map = new Uint8Array(N * N);
    for (const s of p.strokes) for (let i = 0; i < s.pts.length; i++) { const a = s.pts[i], b = s.pts[Math.min(i + 1, s.pts.length - 1)]; paintSegment(map, N, N, a[0], a[1], b[0], b[1], s.r, s.l); }
    painter.setLabels(map);
    painter.setActive(1);
    pb.forEach((b, i) => b.classList.toggle("active", list[i] === p));
    prev = null;
    makeImage();
  }

  // ---- image ----
  let cellImg: Float64Array | null = null, I: Float64Array = new Float64Array(N * N);
  function makeImage(): void {
    if (st.scene === "cell") {
      if (!cellImg) { loadCell(N).then(g => { cellImg = Float64Array.from(g.data); makeImage(); }); return; }
      const r = rng(9);
      I = Float64Array.from(cellImg, v => Math.min(255, Math.max(0, v + st.noise * r.norm())));
    } else I = walkerScene(st.scene, st.noise).I;
    credit.style.visibility = st.scene === "cell" ? "visible" : "hidden";
    painter.setImage(Uint8ClampedArray.from(I));
    prev = null;
    run();
  }

  // ---- solving ----
  let solver: WalkerSolver | null = null, G: WGraph | null = null, K = 2, seeds = new Int32Array(N * N);
  let prev: Float64Array[] | null = null, t0 = 0, raf = 0, info = "";
  let gcL: Uint8Array | null = null, wsL: Int32Array | null = null, gcMs = 0, wsMs = 0;
  function run(): void {
    syncBtns();
    const m = painter.labels;
    seeds = Int32Array.from(m);
    const has = [0, 0, 0, 0];
    for (let p = 0; p < N * N; p++) has[seeds[p]] = 1;
    K = has[3] ? 3 : 2;
    cancelAnimationFrame(raf);
    if (!has[1] || !has[2]) { solver = null; info = "Paint at least one object and one background seed."; draw(); return; }
    t0 = performance.now();
    G = gridGraph(I, N, N, st.sigma, st.conn, 1e-4);
    const o: WalkerOptions = { tol: 1e-5 };
    if (st.priors && !compare) { o.lambda = st.lambda; o.priors = seedPriors(I, seeds, K); }
    if (prev && prev.length >= K - 1) o.x0 = prev;
    solver = new WalkerSolver(G, seeds, K, o);
    if (compare) runOthers();
    const tick = () => {
      solver!.step(14);
      info = `${solver!.done ? "solved" : "solving"}: ${solver!.iterations} CG iterations, residual ${solver!.residual.toExponential(1)}, ${(performance.now() - t0).toFixed(0)} ms`;
      draw();
      if (!solver!.done) raf = requestAnimationFrame(tick); else prev = solver!.prob();
    };
    raf = requestAnimationFrame(tick);
  }
  function runOthers(): void {
    const seeds8 = new Int8Array(N * N);
    for (let p = 0; p < N * N; p++) seeds8[p] = seeds[p] === 1 ? 1 : seeds[p] ? -1 : 0;
    let t = performance.now();
    const E = gridEdges(N, N, st.conn);
    gcL = solveSeg({ n: N * N, D0: new Float64Array(N * N), D1: new Float64Array(N * N), E, w: contrastWeights(I, E, st.sigma), lambda: 1, seeds: seeds8 }).L;
    gcMs = performance.now() - t;
    t = performance.now();
    const mk = Int32Array.from(seeds, s => (s === 1 ? 1 : s ? 2 : 0));
    wsL = primForest(gridDissim([I], N, N, st.conn), mk).labels;
    wsMs = performance.now() - t;
  }

  // ---- drawing ----
  function boundaryOverlay(lab: ArrayLike<number>, tint: boolean): ImageData {
    const o = new ImageData(N, N);
    for (let p = 0; p < N * N; p++) {
      const x = p % N, y = (p - x) / N, l = lab[p];
      const edge = (x + 1 < N && lab[p + 1] !== l) || (y + 1 < N && lab[p + N] !== l) || (x > 0 && lab[p - 1] !== l) || (y > 0 && lab[p - N] !== l);
      if (edge && l === 1) { o.data[4 * p] = 255; o.data[4 * p + 1] = 235; o.data[4 * p + 2] = 0; o.data[4 * p + 3] = 255; }
      else if (tint && l >= 1) { const c = COLS[l - 1]; o.data[4 * p] = c[0]; o.data[4 * p + 1] = c[1]; o.data[4 * p + 2] = c[2]; o.data[4 * p + 3] = 70; }
    }
    return o;
  }
  function drawSmall(c: HTMLCanvasElement, lab: ArrayLike<number>): void {
    const ov = boundaryOverlay(lab, true), id = c.getContext("2d")!.createImageData(N, N);
    for (let p = 0; p < N * N; p++) {
      const g = I[p], a = ov.data[4 * p + 3] / 255;
      for (let j = 0; j < 3; j++) id.data[4 * p + j] = g + a * (ov.data[4 * p + j] - g);
      if (seeds[p]) { const s = COLS[seeds[p] - 1]; for (let j = 0; j < 3; j++) id.data[4 * p + j] = s[j]; }
      id.data[4 * p + 3] = 255;
    }
    c.getContext("2d")!.putImageData(id, 0, 0);
  }
  function draw(): void {
    syncBtns();
    if (!solver) { painter.setOverlay(null); ro.textContent = info; if (!compare) cv.getContext("2d")!.clearRect(0, 0, cv.width, cv.height); return; }
    const P = solver.prob();
    const lab = new Int32Array(N * N);
    for (let p = 0; p < N * N; p++) { let b = 1, bp = -1; for (let k = 0; k < K; k++) if (P[k][p] > bp) { bp = P[k][p]; b = k + 1; } lab[p] = b; }
    painter.setOverlay(boundaryOverlay(lab, compare));
    const lines = [info];
    let obj = 0; for (let p = 0; p < N * N; p++) if (lab[p] === 1) obj++;
    lines.push(`random walker: ${obj} object px`);
    if (compare && gcL && wsL) {
      drawSmall(cvGC!, Int32Array.from(gcL, v => (v ? 1 : 2)));
      drawSmall(cvWS!, wsL);
      let og = 0, ow = 0; for (let p = 0; p < N * N; p++) { if (gcL[p]) og++; if (wsL[p] === 1) ow++; }
      lines.push(`graph cut: ${og} object px (${gcMs.toFixed(0)} ms)`, `watershed: ${ow} object px (${wsMs.toFixed(0)} ms)`);
    }
    if (!compare) {
      const ctx = cv.getContext("2d")!, id = ctx.createImageData(N, N);
      for (let p = 0; p < N * N; p++) {
        let c: [number, number, number];
        if (st.view === "prob") { const v = diverging(P[0][p]); c = [v[0] * 255, v[1] * 255, v[2] * 255]; }
        else if (st.view === "rgb") { c = [0, 0, 0]; for (let k = 0; k < K; k++) for (let j = 0; j < 3; j++) c[j] += P[k][p] * COLS[k][j]; }
        else { let m = 0; for (let k = 0; k < K; k++) m = Math.max(m, P[k][p]); const u = (1 - m) / (1 - 1 / K); const v = inferno(u); c = [v[0] * 255, v[1] * 255, v[2] * 255]; }
        id.data[4 * p] = c[0]; id.data[4 * p + 1] = c[1]; id.data[4 * p + 2] = c[2]; id.data[4 * p + 3] = 255;
      }
      const tmp = document.createElement("canvas"); tmp.width = N; tmp.height = N;
      tmp.getContext("2d")!.putImageData(id, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(tmp, 0, 0, cv.width, cv.height);
      cap.textContent = st.view === "prob" ? "p(object): red 1, white 0.5, blue 0" : st.view === "rgb" ? "probabilities as colours" : "uncertainty 1 − max p (bright = unsure)";
      if (probe && solver.done) drawProbe(ctx, P, lines);
    }
    ro.textContent = lines.join("\n");
  }
  function drawProbe(ctx: CanvasRenderingContext2D, P: Float64Array[], lines: string[]): void {
    const [px, py] = probe!, p = py * N + px, s = cv.width / N;
    if (!G) return;
    const wk = simulateWalks(csr(G), seeds, p, 30, rng(77 + p), 200000, 30);
    ctx.lineWidth = 1; ctx.strokeStyle = "rgba(255,255,255,0.8)";
    for (const path of wk.paths) {
      ctx.beginPath();
      const L = Math.min(path.length, 4000);
      for (let i = 0; i < L; i++) { const v = path[i], x = (v % N + 0.5) * s, y = (Math.floor(v / N) + 0.5) * s; if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
      ctx.stroke();
      const e = path[path.length - 1], c = COLS[(seeds[e] || 1) - 1];
      ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      ctx.beginPath(); ctx.arc((e % N + 0.5) * s, (Math.floor(e / N) + 0.5) * s, 4, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.strokeStyle = "#000"; ctx.lineWidth = 2; ctx.strokeRect(px * s - 4, py * s - 4, s + 8, s + 8);
    lines.push(`probe (${px}, ${py}): p(object) = ${P[0][p].toFixed(3)}; 30 walks: ${wk.counts[1] ?? 0} end on object seeds`);
  }
  if (!compare) cv.addEventListener("pointerdown", ev => {
    const r = cv.getBoundingClientRect();
    probe = [Math.min(N - 1, Math.max(0, Math.floor((ev.clientX - r.left) / r.width * N))), Math.min(N - 1, Math.max(0, Math.floor((ev.clientY - r.top) / r.height * N)))];
    draw();
  });

  load(p0);
}

autoMount(NAME, mount);
