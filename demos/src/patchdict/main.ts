// Learning a patch dictionary. 4000 overlapping p × p patches (seeded subset) of an image are
// coded with K atoms each (OMP), and the atoms are updated: K-SVD (each atom = the best rank-1
// fit to the residuals of the patches that use it), MOD (all atoms at once by least squares) or
// k-means (one atom per patch with coefficient 1: the atoms are cluster means). Start from random
// patches or from the overcomplete DCT. Panels: the atoms sorted by usage (with the usage count of
// the selected one), the image with a movable patch selector (click), the selected patch written
// as coefficient × atom + …, and its reconstruction; a plot of the representation error against
// the iteration, or (tab) of the error against K for PCA (the first K principal components), the
// DCT and the learned dictionary. The learned dictionary is shared with sparserestore on the
// same page (same image and patch size).
// Auto-mounts on <div data-demo="patchdict">. Optional: data-preset="…".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { drawScalarImage, diverging } from "../lib/colour";
import {
  extractPatches, Patches, dctDictionary, randomPatchDictionary, learnStep, batchOmp, atomUsage, gram, patchPCA, linearApproxError,
  representationError, putDictionary, DictMethod, Codes,
} from "../lib/patches";
import { rng, sampleIndices } from "../lib/random";
import { injectStyle } from "../csmri/view";
import { shapesImage, textureImage } from "../sparserestore/images";

const NAME = "patchdict";
type Img = "cell" | "texture" | "shapes";
interface State { img: Img; p: 6 | 8; atoms: 64 | 128 | 256; K: number; init: "random" | "dct"; method: DictMethod; tab: "iter" | "k" }
const DEF: State = { img: "cell", p: 8, atoms: 128, K: 4, init: "random", method: "ksvd", tab: "iter" };
interface Preset { id: string; label: string; s: Partial<State>; run: number }
const PRESETS: Preset[] = [
  { id: "dct", label: "DCT start", s: { init: "dct", atoms: 64, method: "ksvd" }, run: 0 },
  { id: "learned", label: "learned: edges appear", s: { init: "random", method: "ksvd", K: 4 }, run: 20 },
  { id: "kmeans", label: "k-means atoms: blurry averages", s: { init: "random", method: "kmeans", K: 1 }, run: 20 },
  { id: "k1", label: "K = 1 vs K = 4", s: { init: "random", method: "ksvd", K: 1 }, run: 20 },
  { id: "pca", label: "PCA vs learned at K = 4", s: { init: "random", method: "ksvd", K: 4, tab: "k" }, run: 20 },
  { id: "texture", label: "texture image: stripes in the atoms", s: { img: "texture", init: "random", method: "ksvd", K: 4 }, run: 20 },
];
const N = 128, NTRAIN = 4000;

const CSS = `
.qb-demo[data-demo="${NAME}"] { gap: .6em; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; line-height: 1.15; margin: 0 0 .15em; }
.qb-demo[data-demo="${NAME}"] canvas { display: block; image-rendering: pixelated; border-radius: 2px; }
.qb-demo[data-demo="${NAME}"] .pd-atoms { flex: 0 1 340px; min-width: 220px; }
.qb-demo[data-demo="${NAME}"] .pd-atoms canvas { width: 100%; }
.qb-demo[data-demo="${NAME}"] .pd-mid { flex: 0 1 360px; min-width: 240px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .pd-img canvas { width: 62%; margin: 0 auto; aspect-ratio: 1; cursor: crosshair; touch-action: none; }
.qb-demo[data-demo="${NAME}"] .pd-code { display: flex; flex-wrap: wrap; align-items: center; gap: .25em; font-size: 17px; }
.qb-demo[data-demo="${NAME}"] .pd-code canvas { width: 46px; height: 46px; border: 1px solid var(--qb-grid); }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 320px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .pd-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
.qb-demo[data-demo="${NAME}"] .pd-credit { font-size: 12px; color: var(--qb-muted); }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 16px; gap: .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .pd-atoms { flex-basis: 330px; max-width: 330px; }
.reveal .qb-demo[data-demo="${NAME}"] .pd-mid { flex-basis: 330px; max-width: 340px; }
.reveal .qb-demo[data-demo="${NAME}"] .pd-code canvas { width: 40px; height: 40px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex-basis: 330px; max-width: 340px; gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 13px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-presets { gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .pd-lab { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; padding: .2em .4em; }
`;

function drawPatch(cv: HTMLCanvasElement, v: ArrayLike<number>, p: number, signed: boolean): void {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < v.length; i++) { lo = Math.min(lo, v[i]); hi = Math.max(hi, v[i]); }
  if (signed) { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; drawScalarImage(cv, v, p, p, { lo: -m, hi: m, map: diverging }); }
  else drawScalarImage(cv, v, p, p, { lo, hi: hi > lo ? hi : lo + 1 });
}

function mount(host: HTMLElement): void {
  injectStyle(NAME, CSS);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const init: State = { ...DEF, ...(p0?.s ?? {}) };
  const st: State = { ...init };

  const colA = htmlEl("figure", "pd-atoms", "", host);
  const capA = htmlEl("figcaption", "", "atoms", colA);
  const cvA = htmlEl("canvas", "", "", colA);
  const mid = htmlEl("div", "pd-mid", "", host);
  const figI = htmlEl("figure", "pd-img", "", mid);
  htmlEl("figcaption", "", "image: click to pick a patch", figI);
  const cvI = htmlEl("canvas", "", "", figI);
  const code = htmlEl("div", "pd-code", "", mid);
  const plotBox = htmlEl("div", "", "", mid);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "pd-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEF, p.s); sync(); restart(p.run); } })));
  htmlEl("div", "pd-lab", "image", panel);
  const iRow = htmlEl("div", "qb-presets", "", panel);
  const iBtns = (["cell", "texture", "shapes"] as Img[]).map(v => button(iRow, { cell: "micrograph", texture: "texture", shapes: "shapes" }[v], () => { st.img = v; changed(); sync(); restart(0); }));
  htmlEl("div", "pd-lab", "method", panel);
  const mRow = htmlEl("div", "qb-presets", "", panel);
  const mBtns = (["ksvd", "mod", "kmeans"] as DictMethod[]).map(v => button(mRow, { ksvd: "K-SVD", mod: "MOD", kmeans: "k-means" }[v], () => { st.method = v; changed(); sync(); restart(0); }));
  const sRow = htmlEl("div", "qb-presets", "", panel);
  const initBtns = (["random", "dct"] as const).map(v => button(sRow, v === "random" ? "start: random patches" : "start: DCT", () => { st.init = v; changed(); sync(); restart(0); }));
  const aRow = htmlEl("div", "qb-presets", "", panel);
  const atomBtns = ([64, 128, 256] as const).map(v => button(aRow, v + " atoms", () => { st.atoms = v; changed(); sync(); restart(0); }));
  const pszBtns = ([6, 8] as const).map(v => button(aRow, `${v}×${v}`, () => { st.p = v; changed(); sync(); restart(0); }));
  const kS = slider(panel, "sparsity K", 1, 8, 1, st.K, v => { st.K = v; changed(); restart(0); });
  const bRow = htmlEl("div", "qb-presets", "", panel);
  button(bRow, "one iteration", () => { queue += 1; pump(); });
  button(bRow, "run 20", () => { queue += 20; pump(); });
  button(bRow, "reset", () => { Object.assign(st, init); sync(); restart(p0?.run ?? 0); presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0)); });
  const tRow = htmlEl("div", "qb-presets", "", panel);
  const tabBtns = [button(tRow, "error vs iteration", () => { st.tab = "iter"; sync(); drawPlot(); }), button(tRow, "error vs K: PCA, DCT, learned", () => { st.tab = "k"; sync(); drawPlot(); })];
  const ro = htmlEl("div", "qb-readout", "", panel);
  const credit = htmlEl("div", "pd-credit", "", panel);
  htmlEl("div", "qb-note", "Error = mean squared difference per pixel between the training patches and their codes (grey levels²). Atoms are shown with their own contrast (blue negative, red positive).", panel);
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");

  function changed(): void { presetBtns.forEach(b => b.classList.remove("active")); }
  function sync(): void {
    iBtns.forEach((b, i) => b.classList.toggle("active", ["cell", "texture", "shapes"][i] === st.img));
    mBtns.forEach((b, i) => b.classList.toggle("active", ["ksvd", "mod", "kmeans"][i] === st.method));
    initBtns.forEach((b, i) => b.classList.toggle("active", ["random", "dct"][i] === st.init));
    atomBtns.forEach((b, i) => b.classList.toggle("active", [64, 128, 256][i] === st.atoms));
    pszBtns.forEach((b, i) => b.classList.toggle("active", [6, 8][i] === st.p));
    tabBtns.forEach((b, i) => b.classList.toggle("active", ["iter", "k"][i] === st.tab));
    kS.set(st.K);
  }

  // ---- state ----
  let img: Float64Array = new Float64Array(N * N), cell: Float64Array | null = null;
  let P: Patches | null = null, D: Float64Array = new Float64Array(0), codes: Codes | null = null, errs: number[] = [], it = 0, queue = 0, busy = false;
  let sel = { x: 60, y: 40 }, kcurve: { pca: number[]; dct: number[]; learned: number[] } | null = null;
  const d = () => st.p * st.p;

  async function restart(run: number): Promise<void> {
    queue = 0;
    if (st.img === "cell") {
      if (!cell) { const g = await loadCell(256); cell = new Float64Array(N * N); for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) cell[y * N + x] = g.data[(y + 64) * 256 + x + 64]; }
      img = cell;
    } else img = st.img === "texture" ? textureImage(N) : shapesImage(N);
    credit.textContent = st.img === "cell" ? CELL_CREDIT : "Synthetic test image.";
    const all = extractPatches(img, N, N, st.p, 1), r = rng(7);
    const pick = sampleIndices(r, all.n, Math.min(NTRAIN, all.n));
    const Y = new Float64Array(pick.length * d()), pos = new Int32Array(2 * pick.length);
    pick.forEach((q, i) => { Y.set(all.Y.subarray(q * d(), (q + 1) * d()), i * d()); pos[2 * i] = all.pos[2 * q]; pos[2 * i + 1] = all.pos[2 * q + 1]; });
    P = { Y, pos, n: pick.length, d: d(), p: st.p };
    if (st.init === "dct") {
      const m = Math.max(st.p, Math.round(Math.sqrt(st.atoms)));
      const full = dctDictionary(st.p, m * m);
      D = full.slice(0, st.atoms * d());
      if (D.length < st.atoms * d()) { const ex = randomPatchDictionary(P, st.atoms, rng(3)); const o = new Float64Array(st.atoms * d()); o.set(D); o.set(ex.subarray(D.length), D.length); D = o; }
    } else D = randomPatchDictionary(P, st.atoms, rng(3));
    if (st.method === "kmeans") {   // k-means atoms are means of raw patches: start from patches themselves
      const idx = sampleIndices(rng(4), P.n, st.atoms);
      D = new Float64Array(st.atoms * d()); idx.forEach((q, j) => D.set(P!.Y.subarray(q * d(), (q + 1) * d()), j * d()));
    }
    codes = st.method === "kmeans" ? null : batchOmp(D, d(), P.Y, P.n, { K: st.K });
    errs = [codes ? representationError(D, d(), P.Y, codes) : NaN];
    it = 0; kcurve = null;
    queue = run;
    drawAll();
    pump();
  }

  function pump(): void {
    if (busy || !P) return;
    if (queue <= 0) { finish(); return; }
    busy = true;
    setTimeout(() => {
      const res = learnStep(D, d(), P!, st.K, st.method);
      codes = res.codes; it++; queue--;
      errs.push(res.error);
      busy = false;
      drawAll();
      pump();
    }, 0);
  }
  function finish(): void {
    if (!P) return;
    kcurve = null;
    if (st.tab === "k") computeK();
    putDictionary(`${st.img}:${N}:${st.p}`, { D: Float64Array.from(D), d: d(), p: st.p, k: st.atoms, label: `${st.method}, ${it} iterations`, source: "patchdict" });
    drawAll();
  }
  function computeK(): void {
    if (!P) return;
    const sub: Patches = { ...P, n: 800, Y: P.Y.subarray(0, 800 * d()) };
    const pca = patchPCA(sub, Math.min(d(), 9)), dctD = dctDictionary(st.p, Math.max(st.p * st.p, 64));
    const G1 = gram(D, d()), G2 = gram(dctD, d());
    const out = { pca: [] as number[], dct: [] as number[], learned: [] as number[] };
    for (let K = 1; K <= 8; K++) {
      out.pca.push(linearApproxError(pca, d(), sub, K));
      out.dct.push(representationError(dctD, d(), sub.Y, batchOmp(dctD, d(), sub.Y, sub.n, { K, G: G2 })));
      out.learned.push(st.method === "kmeans" ? NaN : representationError(D, d(), sub.Y, batchOmp(D, d(), sub.Y, sub.n, { K, G: G1 })));
    }
    kcurve = out;
  }

  // ---- drawing ----
  function drawAll(): void { drawAtoms(); drawImage(); drawCode(); drawPlot(); readout(); }
  function usage(): Int32Array { return codes ? atomUsage(codes, st.atoms) : new Int32Array(st.atoms); }
  function drawAtoms(): void {
    const k = st.atoms, p = st.p, cols = k === 128 ? 16 : Math.round(Math.sqrt(k)), rows = Math.ceil(k / cols), s = p + 1;
    const W = cols * s + 1, H = rows * s + 1, buf = new Float64Array(W * H).fill(NaN);
    const u = usage(), order = Array.from({ length: k }, (_, j) => j).sort((a, b) => u[b] - u[a]);
    order.forEach((j, slot) => {
      const ox = (slot % cols) * s + 1, oy = Math.floor(slot / cols) * s + 1;
      let lo = Infinity, hi = -Infinity;
      for (let t = 0; t < p * p; t++) { lo = Math.min(lo, D[j * p * p + t]); hi = Math.max(hi, D[j * p * p + t]); }
      for (let t = 0; t < p * p; t++) buf[(oy + Math.floor(t / p)) * W + ox + (t % p)] = hi > lo ? (D[j * p * p + t] - lo) / (hi - lo) : 0.5;
    });
    drawScalarImage(cvA, buf, W, H, { lo: 0, hi: 1, nan: [255, 255, 255, 255] });
    cvA.style.aspectRatio = `${W} / ${H}`;
    capA.textContent = `${k} atoms (${st.p}×${st.p}), sorted by use · ${{ ksvd: "K-SVD", mod: "MOD", kmeans: "k-means" }[st.method]}, ${it} iterations`;
  }
  function drawImage(): void {
    const c = cvI; c.width = N; c.height = N;
    const ctx = c.getContext("2d")!;
    const id = ctx.createImageData(N, N);
    for (let i = 0; i < N * N; i++) { const v = Math.max(0, Math.min(255, img[i])); id.data[4 * i] = v; id.data[4 * i + 1] = v; id.data[4 * i + 2] = v; id.data[4 * i + 3] = 255; }
    ctx.putImageData(id, 0, 0);
    ctx.strokeStyle = "#ff3030"; ctx.lineWidth = 1.5;
    ctx.strokeRect(sel.x - 0.5, sel.y - 0.5, st.p + 1, st.p + 1);
  }
  cvI.addEventListener("pointerdown", ev => {
    const r = cvI.getBoundingClientRect();
    sel = { x: Math.max(0, Math.min(N - st.p, Math.floor((ev.clientX - r.left) / r.width * N - st.p / 2))), y: Math.max(0, Math.min(N - st.p, Math.floor((ev.clientY - r.top) / r.height * N - st.p / 2))) };
    drawImage(); drawCode();
  });
  function drawCode(): void {
    code.replaceChildren();
    const p = st.p, y = new Float64Array(p * p);
    for (let v = 0; v < p; v++) for (let u = 0; u < p; u++) y[v * p + u] = img[(sel.y + v) * N + sel.x + u];
    const mkCv = (vals: ArrayLike<number>, signed: boolean) => { const cv = htmlEl("canvas", "", "", code); drawPatch(cv, vals, p, signed); return cv; };
    mkCv(y, false);
    htmlEl("span", "", "≈", code);
    let terms: { j: number; c: number }[] = [];
    if (st.method === "kmeans") {
      let best = 0, bd = Infinity;
      for (let j = 0; j < st.atoms; j++) { let s = 0; for (let t = 0; t < p * p; t++) s += (y[t] - D[j * p * p + t]) ** 2; if (s < bd) { bd = s; best = j; } }
      terms = [{ j: best, c: 1 }];
    } else {
      const c = batchOmp(D, p * p, y, 1, { K: st.K });
      for (let s = 0; s < c.maxK; s++) if (c.idx[s] >= 0) terms.push({ j: c.idx[s], c: c.val[s] });
    }
    const rec = new Float64Array(p * p);
    terms.forEach((t, i) => {
      htmlEl("span", "", `${i ? (t.c < 0 ? "−" : "+") : t.c < 0 ? "−" : ""} ${Math.abs(t.c).toFixed(0)}×`, code);
      const a = D.subarray(t.j * p * p, (t.j + 1) * p * p);
      mkCv(a, true);
      for (let q = 0; q < p * p; q++) rec[q] += t.c * a[q];
    });
    htmlEl("span", "", "=", code);
    mkCv(rec, false);
  }
  let plot: Plot | null = null;
  function drawPlot(): void {
    plotBox.replaceChildren();
    if (st.tab === "iter") {
      const fin = errs.filter(Number.isFinite);
      const hi = Math.max(...fin, 1), lo = Math.min(...fin, hi * 0.5);
      plot = new Plot(plotBox, { width: 420, height: 200, x: [0, Math.max(10, it)], y: [Math.max(0, lo * 0.9), hi * 1.05], xlabel: "iteration", ylabel: "error", font: 20, margin: { t: 8 } });
      plot.line(errs.map((_, i) => i), errs, { color: seriesColor(0) });
      plot.scatter(errs.map((_, i) => i), errs, { color: seriesColor(0), r: 3 });
    } else {
      if (!kcurve && queue <= 0) computeK();
      if (!kcurve) return;
      const all = [...kcurve.pca, ...kcurve.dct, ...kcurve.learned].filter(Number.isFinite);
      plot = new Plot(plotBox, { width: 420, height: 200, x: [1, 8], y: [0, Math.max(...all) * 1.05], xlabel: "K atoms per patch", ylabel: "error", font: 20, margin: { t: 8 } });
      const ks = [1, 2, 3, 4, 5, 6, 7, 8];
      plot.line(ks, kcurve.pca, { color: seriesColor(3), label: "PCA (first K)" });
      plot.line(ks, kcurve.dct, { color: seriesColor(4), label: "DCT (best K)" });
      if (kcurve.learned.some(Number.isFinite)) plot.line(ks, kcurve.learned, { color: seriesColor(0), label: "learned (best K)" });
      plot.legend(undefined, "tr");
    }
  }
  function readout(): void {
    const u = usage(), used = Array.from(u).filter(v => v > 0).length;
    ro.textContent = `training patches ${P?.n ?? 0}, K = ${st.method === "kmeans" ? "1 (coef. 1)" : st.K}\n` +
      `error ${errs.length ? errs[errs.length - 1].toFixed(1) : "–"} (start ${errs.length ? errs[0].toFixed(1) : "–"})\n` +
      `atoms used ${used} of ${st.atoms}` + (queue > 0 ? `\nlearning… ${queue} iterations to go` : "");
  }

  sync();
  restart(p0?.run ?? 0);
}

autoMount(NAME, mount);
