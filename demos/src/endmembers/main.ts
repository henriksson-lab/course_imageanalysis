// Finding the spectra when they are unknown. The day's cell with 3 or 4 components in a
// 16-channel lambda stack (Poisson noise), optionally without pure pixels (every pixel
// gets a share of the other components). Panels: the singular values of the pixel matrix
// (scree plot, log scale: how many components?); the pixels normalised to sum one and
// projected onto their two main principal directions (the simplex seen from above), with
// the true spectra (circles) and the found ones (squares); true (dashed) and found (solid)
// spectra; the abundance maps of the found spectra. Methods: PCA axes (orthogonal, with
// negative entries: the wrong answer), vertex finding (most extreme pixel, project out,
// repeat; step button), NMF (multiplicative updates from a seeded start; step / run, seed,
// error curve; optionally with the first three spectra fixed).
// Auto-mounts on <div data-demo="endmembers">. Options: data-preset (count|pure|nopure|
// seeds|known).
import { htmlEl, slider, presets, checkbox, button, svgEl, autoMount } from "../lib/ui";
import { Plot } from "../lib/plot";
import { RGB } from "../lib/colour";
import * as U from "../lib/unmix";
import { injectStyle, uniqueClip, canvasFigure, fmt, COMMON_CSS } from "../lsunmix/common";
import { L0, L1, N_PIX, P } from "../lsunmix/model";
import { NCH, buildData, vertexSpectra, knownSpectra, EmData } from "./model";

const NAME = "endmembers";
const CSS = COMMON_CSS + `
[data-demo="${NAME}"] { gap: .6em; align-items: flex-start; }
[data-demo="${NAME}"] .em-col { flex: 1 1 300px; min-width: 240px; max-width: 380px; display: flex; flex-direction: column; gap: .2em; }
[data-demo="${NAME}"] .em-maps { display: flex; gap: .3em; }
[data-demo="${NAME}"] .em-maps .um-fig { flex: 1 1 0; min-width: 0; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 330px; gap: .3em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; flex-wrap: nowrap; }
.reveal [data-demo="${NAME}"] .em-col { flex: 0 0 380px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 0 0 340px; max-width: 340px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .08em .35em; }
.reveal [data-demo="${NAME}"] .um-readout { font-size: 14px; }
.reveal [data-demo="${NAME}"] .qb-check, .reveal [data-demo="${NAME}"] .qb-slider { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .15em; }
`;

const COMP = ["nucleus", "mitochondria", "membrane", "autofluor."];
const CSSCOL = ["#3f72ff", "#1fa83a", "#e03cc8", "#d99a12"];
const FOUND = ["#000000", "#7f7f7f", "#8c564b", "#17becf"];
type Method = "pca" | "vertex" | "nmf";
interface St { K: number; pure: boolean; logPhot: number; method: Method; seed: number; known: boolean }

const hex = (h: string): RGB => [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];

function mount(host: HTMLElement): void {
  injectStyle(NAME, CSS);
  const st: St = { K: 3, pure: true, logPhot: Math.log10(400), method: "vertex", seed: 1, known: false };
  const bands = U.contiguousBands(L0, L1, NCH), cs = bands.map(b => (b.lo + b.hi) / 2);
  let data: EmData | null = null;

  // ---- layout ----
  const c1 = htmlEl("div", "em-col", "", host), c2 = htmlEl("div", "em-col", "", host);
  const pScree = new Plot(c1, { width: 380, height: 200, x: [0.4, 8.6], y: [-3, 0.2], xlabel: "index i", ylabel: "log₁₀ σᵢ/σ₁", xticks: [1, 2, 3, 4, 5, 6, 7, 8], yticks: [-3, -2, -1, 0], font: 17, title: "singular values of the pixels" });
  const pCloud = new Plot(c1, { width: 380, height: 300, x: [-1, 1], y: [-1, 1], equal: true, grid: false, xticks: [], yticks: [], font: 17, title: "normalised pixels (2 PCs)", margin: { l: 10, b: 10 } });
  const pSpec = new Plot(c2, { width: 380, height: 230, x: [L0, L1], y: [-0.05, 0.3], xlabel: "wavelength (nm)", ylabel: "spectrum", xticks: [500, 550, 600, 650, 700], yticks: 3, font: 17, title: "true (dashed) and found" });
  const errBox = htmlEl("div", "", "", c2);
  const pErr = new Plot(errBox, { width: 380, height: 150, x: [0, 300], y: [0.9, 2], xlabel: "NMF iteration", ylabel: "÷ true", xticks: 4, yticks: [1, 1.5, 2], font: 16, margin: { b: 52 } });
  uniqueClip(pErr.svg, NAME);
  const mapsRow = htmlEl("div", "em-maps", "", c2);
  const maps = [0, 1, 2, 3].map(k => canvasFigure(mapsRow, "found " + (k + 1), N_PIX, N_PIX));
  for (const p of [pScree, pCloud, pSpec]) uniqueClip(p.svg, NAME);
  const panel = htmlEl("div", "qb-panel", "", host);
  const presetDefs: { id: string; label: string; s: Partial<St>; run?: number }[] = [
    { id: "count", label: "count the components", s: { K: 4, pure: true, method: "pca", known: false } },
    { id: "pure", label: "pure pixels: the vertices are the spectra", s: { K: 3, pure: true, method: "vertex", known: false }, run: 3 },
    { id: "nopure", label: "no pure pixels: vertex finding undershoots", s: { K: 3, pure: false, method: "vertex", known: false }, run: 3 },
    { id: "seeds", label: "NMF from two seeds: two answers, same error", s: { K: 3, pure: false, method: "nmf", seed: 1, known: false }, run: 400 },
    { id: "known", label: "three spectra known: the 4th still has dips", s: { K: 4, pure: true, method: "nmf", seed: 1, known: true }, run: 200 },
  ];
  const pbtn = presets(panel, [...presetDefs.map(p => ({ label: p.label, apply: () => applyPreset(p) })), { label: "reset", apply: () => applyPreset({ s: { K: 3, pure: true, method: "vertex", seed: 1, known: false, logPhot: Math.log10(400) } }) }]);
  const mRow = htmlEl("div", "um-row", "", panel);
  const mBtns = (["pca", "vertex", "nmf"] as Method[]).map(m => { const b = button(mRow, m === "pca" ? "PCA axes" : m === "vertex" ? "vertex finding" : "NMF", () => { st.method = m; manual(); restart(); }); b.dataset.m = m; return b; });
  const sRow = htmlEl("div", "um-row", "", panel);
  const stepB = button(sRow, "step", () => { step(st.method === "nmf" ? 10 : 1); });
  const runB = button(sRow, "run", () => run(st.method === "nmf" ? 300 : st.K));
  button(sRow, "restart", () => restart());
  const kS = slider(panel, "components K", 3, 4, 1, st.K, v => { st.K = v; manual(); rebuild(); });
  const pureC = checkbox(panel, "pure pixels", st.pure, v => { st.pure = v; manual(); rebuild(); });
  const phS = slider(panel, "photons", 1.5, 3.7, 0.05, st.logPhot, v => { st.logPhot = v; manual(); rebuild(); }, v => String(Math.round(10 ** v)));
  const seedS = slider(panel, "NMF seed", 1, 5, 1, st.seed, v => { st.seed = v; manual(); restart(); });
  const knownC = checkbox(panel, "NMF: first 3 spectra known", st.known, v => { st.known = v; manual(); restart(); });
  const read = htmlEl("div", "qb-readout um-readout", "", c2);
  void stepB; void runB;
  function manual(): void { pbtn.forEach(b => b.classList.remove("active")); }

  // ---- data ----
  let K = 3, Mtrue: Float64Array = new Float64Array(0), cellIdx: number[] = [];
  let Yc: Float32Array = new Float32Array(0);          // cell pixels only (NCH × n)
  let pcaMean: Float64Array = new Float64Array(0), pc1: Float64Array = new Float64Array(0), pc2: Float64Array = new Float64Array(0), sv: Float64Array = new Float64Array(0);
  let found: Float64Array | null = null, verts: number[] = [], nmf: U.NMF | null = null, errs: number[] = [];
  let timer = 0;

  function rebuild(): void {
    K = st.K;
    const d = buildData(K, st.pure, 10 ** st.logPhot);
    data = d; Mtrue = d.Mtrue; cellIdx = d.cellIdx; Yc = d.Yc; sv = d.sv;
    const n = cellIdx.length;
    // projection plane for the normalised cloud: PCA of the normalised bright pixels
    const Yn2 = U.normaliseSum(Yc, NCH, n), w = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = 0; for (let c = 0; c < NCH; c++) s += Yc[c * n + i]; w[i] = s > 0.15 && Number.isFinite(Yn2[i]) ? 1 : 0; if (!w[i]) for (let c = 0; c < NCH; c++) Yn2[c * n + i] = 0; }
    const pca = U.stackPCA(Yn2, NCH, n, w);
    pcaMean = pca.mean;
    pc1 = Float64Array.from({ length: NCH }, (_, c) => pca.vectors[c * NCH]);
    pc2 = Float64Array.from({ length: NCH }, (_, c) => pca.vectors[c * NCH + 1]);
    cloudN = Yn2; cloudW = w; Yk = null;
    {   // fit error with the true spectra (NNLS), the reference for "same error"
      const G = U.gram(Mtrue, NCH, K), y = new Float64Array(NCH);
      let e = 0, t = 0;
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < NCH; c++) y[c] = Yc[c * n + i];
        const a = U.nnlsGram(G, U.mty(Mtrue, NCH, K, y), K);
        for (let c = 0; c < NCH; c++) { let v = 0; for (let j = 0; j < K; j++) v += Mtrue[c * K + j] * a[j]; e += (y[c] - v) ** 2; t += y[c] ** 2; }
      }
      trueErr = Math.sqrt(e / t); trueErr2 = e;
    }
    restart();
  }
  let Yk: Float32Array | null = null, trueErr = 0, trueErr2 = 1;
  let cloudN: Float32Array = new Float32Array(0), cloudW: Float64Array = new Float64Array(0);

  function restart(): void {
    cancelAnimationFrame(timer);
    found = null; verts = []; nmf = null; errs = [];
    if (st.method === "pca") {
      // the K leading principal axes of the (centred) pixels: orthogonal, signed
      const pca = U.stackPCA(Yc, NCH, cellIdx.length);
      found = new Float64Array(NCH * K);
      for (let k = 0; k < K; k++) for (let c = 0; c < NCH; c++) found[c * K + k] = pca.vectors[c * NCH + k];
    } else if (st.method === "nmf") {
      const known = st.known ? knownSpectra(data!) : [];
      nmf = U.nmfInit(Yc, NCH, cellIdx.length, K, st.seed, st.known ? [...known, ...(K > 3 ? [null] : [])] : []);
      errs = [U.nmfError(Yc, nmf)];
      found = Float64Array.from(nmf.W);
    }
    draw();
  }
  function step(times: number): void {
    if (st.method === "vertex") {
      if (verts.length >= K) return;
      // on the rank-K projection of the pixels (noise outside the K-dimensional signal space
      // removed); the found spectrum = the (denoised) pixel at the vertex, scaled to sum 1
      const r = vertexSpectra(data!, verts.length + 1, Yk ??= U.projectRank(Yc, NCH, cellIdx.length, K));
      verts = r.verts; found = r.F;
    } else if (st.method === "nmf" && nmf) {
      for (let t = 0; t < times; t++) { U.nmfStep(Yc, nmf); }
      const e = U.nmfError(Yc, nmf);
      errs.push(e);
      const c = { ...nmf, W: Float64Array.from(nmf.W), H: Float64Array.from(nmf.H) };
      U.nmfNormalise(c);
      found = c.W;
    }
    draw();
  }
  function run(total: number): void {
    cancelAnimationFrame(timer);
    let done = 0;
    const tick = () => {
      const n = st.method === "nmf" ? 10 : 1;
      step(n); done += n;
      if (done < total && (st.method === "nmf" || verts.length < K)) timer = requestAnimationFrame(tick);
    };
    timer = requestAnimationFrame(tick);
  }

  // ---- drawing ----
  const proj = (v: ArrayLike<number>): [number, number] => {
    let s = 0;
    for (let c = 0; c < NCH; c++) s += v[c];
    let x = 0, y = 0;
    for (let c = 0; c < NCH; c++) { const d = v[c] / (s || 1) - pcaMean[c]; x += d * pc1[c]; y += d * pc2[c]; }
    return [x, y];
  };
  function draw(): void {
    const n = cellIdx.length;
    // scree
    pScree.clear();
    const xs = [1, 2, 3, 4, 5, 6, 7, 8], ys = xs.map(i => Math.log10(Math.max(1e-6, sv[i - 1] / sv[0])));
    pScree.bars(xs, ys, { base: -3, barWidth: 0.6, colors: xs.map(i => (i <= K ? "#3f72ff" : "#aaa")) } as Parameters<Plot["bars"]>[2]);
    errBox.style.display = st.method === "nmf" ? "" : "none";
    if (st.method === "nmf" && errs.length) {
      pErr.clear();
      const it = errs.map((_, i) => i * 10);
      pErr.setDomain([0, Math.max(100, it[it.length - 1])], [0.9, 2]);
      pErr.hline(1, { color: "#999", width: 1.5, dash: "5 4" });
      pErr.line(it, errs.map(e => Math.min(2.5, e / trueErr2)), { color: "#333", width: 2.5 });
      pErr.setLabels({ ylabel: "÷ true" });
    }
    // cloud
    pCloud.clear();
    const px: number[] = [], py: number[] = [];
    for (let i = 0; i < n; i += 2) if (cloudW[i]) { let x = 0, y = 0; for (let c = 0; c < NCH; c++) { const d = cloudN[c * n + i] - pcaMean[c]; x += d * pc1[c]; y += d * pc2[c]; } px.push(x); py.push(y); }
    const tv = Array.from({ length: K }, (_, k) => proj(Array.from({ length: NCH }, (_, c) => Mtrue[c * K + k])));
    const ext = Math.max(...tv.flatMap(v => v.map(Math.abs)), ...px.map(Math.abs), ...py.map(Math.abs)) * 1.15 || 1;
    pCloud.setDomain([-ext, ext], [-ext, ext]);
    pCloud.scatter(px, py, { color: "#777", r: 1.8, opacity: 0.35, stroke: "none" });
    const order = K === 3 ? [0, 1, 2, 0] : [0, 1, 2, 3, 0];
    pCloud.line(order.map(k => tv[k][0]), order.map(k => tv[k][1]), { color: "#555", width: 1.5, dash: "5 4" });
    tv.forEach((v, k) => pCloud.scatter([v[0]], [v[1]], { color: CSSCOL[k], r: 10 }));
    const match = found && st.method !== "pca" ? U.matchColumns(U.normaliseSum(Mtrue, NCH, K), found, NCH, K) : null;
    const foundCol = (j: number) => { if (!match) return FOUND[j]; const i = match.perm.indexOf(j); return i >= 0 ? CSSCOL[i] : FOUND[j]; };
    if (found && st.method !== "pca") {
      const fk = verts.length && st.method === "vertex" ? verts.length : K;
      for (let j = 0; j < fk; j++) {
        const v = proj(Array.from({ length: NCH }, (_, c) => found![c * K + j]));
        pCloud.mark((g: SVGGElement) => { svgEl("rect", { x: pCloud.x(v[0]) - 8, y: pCloud.y(v[1]) - 8, width: 16, height: 16, fill: "#fff", stroke: foundCol(j), "stroke-width": 4 }, g); }, []);
      }
    }
    if (st.method === "pca") {
      for (let k = 0; k < Math.min(2, K); k++) {
        const len = 0.6 * ext;
        pCloud.arrow(0, 0, k === 0 ? len : 0, k === 1 ? len : 0, { color: "#000", width: 3 });
      }
    }
    // spectra
    pSpec.clear();
    let ymax = 0;
    for (let k = 0; k < K; k++) {
      let s = 0; for (let c = 0; c < NCH; c++) s += Mtrue[c * K + k];
      const v = cs.map((_, c) => Mtrue[c * K + k] / s);
      ymax = Math.max(ymax, ...v);
      pSpec.line(cs, v, { color: CSSCOL[k], width: 2.5, dash: "6 4" });
    }
    let ymin = -0.02;
    if (found) {
      const fk = st.method === "vertex" ? verts.length : K;
      for (let j = 0; j < fk; j++) {
        let v = cs.map((_, c) => found![c * K + j]);
        if (st.method === "pca") { const m = Math.max(...v.map(Math.abs)); v = v.map(x => x / m * ymax); }
        ymin = Math.min(ymin, ...v);
        pSpec.line(cs, v, { color: foundCol(j), width: 3.5 });
      }
    }
    pSpec.setDomain(undefined, [Math.min(-0.02, ymin * 1.1), ymax * 1.15]);
    // abundance maps of the found spectra
    const lines: string[] = [];
    lines.push("σᵢ/σ₁: " + Array.from(sv.slice(0, 6)).map(v => fmt(v / sv[0], 3)).join(" "));
    let ab: Float32Array | null = null;
    const fk = found ? (st.method === "vertex" ? verts.length : K) : 0;
    if (found && fk > 0) {
      const F = new Float64Array(NCH * fk);
      for (let c = 0; c < NCH; c++) for (let j = 0; j < fk; j++) F[c * fk + j] = found[c * K + j];
      ab = new Float32Array(fk * P);
      if (st.method === "nmf" && nmf) {
        const c2 = { ...nmf, W: Float64Array.from(nmf.W), H: Float64Array.from(nmf.H) }; U.nmfNormalise(c2);
        cellIdx.forEach((p, i) => { for (let j = 0; j < fk; j++) ab![j * P + p] = c2.H[j * n + i]; });
      } else if (st.method === "pca") {
        cellIdx.forEach((p, i) => { for (let j = 0; j < fk; j++) { let s = 0; for (let c = 0; c < NCH; c++) s += F[c * fk + j] * Yc[c * n + i]; ab![j * P + p] = s; } });
      } else {
        const G = U.gram(F, NCH, fk), y = new Float64Array(NCH);
        cellIdx.forEach((p, i) => { for (let c = 0; c < NCH; c++) y[c] = Yc[c * n + i]; const a = U.nnlsGram(G, U.mty(F, NCH, fk, y), fk); for (let j = 0; j < fk; j++) ab![j * P + p] = a[j]; });
      }
      // fit error with these spectra
      let e = 0, t = 0;
      cellIdx.forEach((p, i) => { for (let c = 0; c < NCH; c++) { let v = 0; for (let j = 0; j < fk; j++) v += F[c * fk + j] * ab![j * P + p]; e += (Yc[c * n + i] - v) ** 2; t += Yc[c * n + i] ** 2; } });
      lines.push(`relative fit error ‖Y − WH‖/‖Y‖: ${fmt(Math.sqrt(e / t), 4)}`);
      lines.push(`  (with the true spectra: ${fmt(trueErr, 4)})`);
      if (match && st.method !== "pca" && fk === K) lines.push("angle to the true spectra: " + match.angleDeg.map(a => fmt(a, 1) + "°").join(", "));
      if (st.method === "pca") lines.push("PCA axes: orthogonal, with negative\nentries: not spectra");
    }
    if (st.method === "nmf" && errs.length) lines.push(`NMF iterations: ${(errs.length - 1) * 10}, ‖Y − WH‖² = ${fmt(errs[errs.length - 1], 4)}\n  (true spectra + NNLS: ${fmt(trueErr2, 4)})`);
    if (st.method === "vertex") lines.push(`vertices found: ${verts.length} of ${K}`);
    read.textContent = lines.join("\n");
    for (let j = 0; j < 4; j++) {
      const f = maps[j];
      f.fig.style.display = j < K ? "" : "none";
      const ctx = f.canvas.getContext("2d")!, id = ctx.createImageData(N_PIX, N_PIX), d = id.data;
      let mx = 1e-9;
      if (ab && j < fk) for (const p of cellIdx) mx = Math.max(mx, Math.abs(ab[j * P + p]));
      const col = hex(foundCol(j) === "#000000" ? "#ffffff" : foundCol(j));
      for (let p = 0; p < P; p++) {
        const v = ab && j < fk ? ab[j * P + p] / mx : 0;
        if (v >= 0) { d[4 * p] = 255 * v * col[0]; d[4 * p + 1] = 255 * v * col[1]; d[4 * p + 2] = 255 * v * col[2]; }
        else { d[4 * p] = 255 * Math.min(1, -v); }
        d[4 * p + 3] = 255;
      }
      ctx.putImageData(id, 0, 0);
      f.cap.textContent = j < fk ? "found " + (j + 1) : "—";
    }
    mBtns.forEach(b => b.classList.toggle("active", b.dataset.m === st.method));
  }

  function applyPreset(p: { s: Partial<St>; run?: number }): void {
    Object.assign(st, p.s);
    kS.set(st.K); pureC.checked = st.pure; phS.set(st.logPhot); seedS.set(st.seed); knownC.checked = st.known;
    rebuild();
    if (p.run) {
      if (st.method === "vertex") { for (let i = 0; i < p.run; i++) step(1); }
      else for (let i = 0; i < p.run / 10; i++) step(10);
    }
  }
  const pre = presetDefs.findIndex(d => d.id === host.dataset.preset);
  if (pre >= 0) { applyPreset(presetDefs[pre]); pbtn[pre].classList.add("active"); }
  else applyPreset({ s: {} });
}

autoMount(NAME, mount);
