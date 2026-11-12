// The support vector machine in 2D. Click to add points of the chosen class (blue = −1,
// red = +1), drag to move them, "remove points" to delete. The SVM is retrained (SMO,
// lib/svm.ts) on every edit and shows: the decision line or curve f = 0 (solid), the margins
// f = ±1 (dashed), the street |f| < 1 (grey), the sides tinted by the sign of f, support
// vectors ringed (solid: on the edge of the street, 0 < α < C; dashed: inside it or on the
// wrong side, α = C), and for the linear kernel function each violation's slack as a short
// segment to its margin. Sliders: C (log), γ (RBF, log), degree (polynomial). Readout: street
// width 2/‖w‖ (linear), support vectors, training errors.
// Options for the kernel-function unit: RBF bumps (circles of radius 1/√γ around the support
// vectors) and the kernel matrix as a heat map (points sorted by class). Cross-validation mode
// (data-cv="1"): a test set (hollow points) from the same source, training and test error
// against C or γ, and a 5-fold cross-validation map over (C, γ).
//
// Auto-mounts on <div data-demo="svm2d">. Options:
//   data-preset   id of presets.ts (default the first offered)
//   data-presets  comma-separated preset ids to offer (default: all of the chosen kernels)
//   data-kernel   comma-separated kernel functions offered: linear, poly, rbf (default linear)
//   data-cv       "1": cross-validation mode
//   data-kmatrix  "1": offer the kernel-matrix panel
import { htmlEl, svgEl, slider, presets, checkbox, button, autoMount, Slider } from "../lib/ui";
import { pointPlayground } from "../lib/playground";
import { Plot, heatmapGrid, seriesColor } from "../lib/plot";
import { QUALITATIVE, viridis } from "../lib/colour";
import { trainSVM, decision, SVMModel, marginWidth, makeFolds, cvError, logGrid } from "../lib/svm";
import { KernelSpec, KernelType, kernelFn, kernelMatrix } from "../lib/svmkernels";
import type { LPoint } from "../lib/datasets2d";
import { worldOf, sampleGrid, contourPath, pathEl, signColour } from "./field";
import { SVM_PRESETS, SVMPreset, presetTest } from "./presets";

const NAME = "svm2d";
const RES = 90;
const KNAMES: Record<KernelType, string> = { linear: "linear", poly: "polynomial", rbf: "RBF" };

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .sv-left { flex: 0 1 470px; min-width: 260px; }
[data-demo="${NAME}"] .sv-left .qb-plot { max-width: none; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 430px; gap: .35em; }
[data-demo="${NAME}"] .sv-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; overflow-x: auto; }
[data-demo="${NAME}"] .sv-extra { flex: 1 1 100%; display: flex; flex-wrap: wrap; gap: .8em; }
[data-demo="${NAME}"] .sv-extra > div { flex: 1 1 300px; min-width: 0; max-width: 480px; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .sv-left { flex-basis: 450px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; }
.reveal [data-demo="${NAME}"] .qb-panel { max-height: 560px; overflow-y: auto; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const p10 = (v: number) => Math.pow(10, v);
const fmtNum = (v: number) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toPrecision(2));

function mount(host: HTMLElement): void {
  injectStyle();
  const kernels = (host.dataset.kernel ?? "linear").split(",").map(s => s.trim()).filter(s => s in KNAMES) as KernelType[];
  if (!kernels.length) kernels.push("linear");
  const cvMode = host.dataset.cv === "1", kOffer = host.dataset.kmatrix === "1";
  const offered = (host.dataset.presets ?? "").split(",").map(s => s.trim()).filter(Boolean);
  const group = cvMode ? "cv" : kernels.some(k => k !== "linear") ? "kernel" : "linear";
  const list = SVM_PRESETS.filter(p => (offered.length ? offered.includes(p.id) : p.group === group));
  const p0 = list.find(p => p.id === host.dataset.preset) ?? list[0] ?? SVM_PRESETS[0];
  const colors = [QUALITATIVE[0], QUALITATIVE[1]];
  const st = { kernel: (kernels.includes(p0.kernel) ? p0.kernel : kernels[0]) as KernelType, logC: p0.logC, logG: p0.logGamma, degree: p0.degree, bumps: false, kmat: false, sweep: "C" as "C" | "gamma" };
  let preset: SVMPreset = p0, model: SVMModel | null = null, test: LPoint[] = [];

  const left = htmlEl("div", "sv-left", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "sv-lab", "presets", panel);
  const presetBtns = presets(panel, list.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "sv-lab", "add points of class", panel);
  const classBox = htmlEl("div", "", "", panel);
  const pg = pointPlayground(left, { k: 2, colors, names: ["class −1", "class +1"], controlsParent: classBox, resolution: RES, r: 8, onChange: (_, fin) => update(fin) });
  const W = worldOf(pg, p0.range[0], p0.range[1]);

  let kBtns: HTMLButtonElement[] = [];
  if (kernels.length > 1) {
    htmlEl("div", "sv-lab", "kernel function", panel);
    kBtns = presets(panel, kernels.map(k => ({ label: KNAMES[k], apply: () => { st.kernel = k; syncControls(); update(true); } })));
  }
  const cS: Slider = slider(panel, "C", -2, 3, 0.1, st.logC, v => { st.logC = v; update(true); }, v => fmtNum(p10(v)));
  const gBox = htmlEl("div", "", "", panel);
  const gS: Slider = slider(gBox, "γ", -2, 2, 0.1, st.logG, v => { st.logG = v; update(true); }, v => fmtNum(p10(v)));
  const dBox = htmlEl("div", "", "", panel);
  const dS: Slider = slider(dBox, "degree p", 2, 5, 1, st.degree, v => { st.degree = v; update(true); });
  const bumpCb = kernels.includes("rbf") ? checkbox(panel, "show the RBF bumps (radius 1/√γ)", st.bumps, v => { st.bumps = v; update(true); }) : null;
  const kmCb = kOffer ? checkbox(panel, "show the kernel matrix", st.kmat, v => { st.kmat = v; update(true); }) : null;
  const ro = htmlEl("div", "qb-readout", "", panel);
  const note = htmlEl("div", "qb-note", "", panel);
  htmlEl("div", "qb-note", "Click to add a point, drag to move it. Ringed points are support vectors.", panel);

  // extra panels below: kernel matrix, CV curves and map
  const extra = htmlEl("div", "sv-extra", "", host);
  const kmBox = htmlEl("div", "", "", extra);
  let curve: Plot | null = null, cvBox: HTMLElement | null = null, cvMap: HTMLElement | null = null;
  let sweepBtns: HTMLButtonElement[] = [];
  if (cvMode) {
    const cb = htmlEl("div", "", "", extra);
    sweepBtns = presets(cb, [{ label: "error against C", apply: () => { st.sweep = "C"; drawCurve(); } }, { label: "error against γ", apply: () => { st.sweep = "gamma"; drawCurve(); } }]);
    curve = new Plot(cb, { width: 480, height: 260, x: [0.01, 1000], y: [0, 0.5], xlog: true, xlabel: "C", ylabel: "error", yfmt: v => (100 * v).toFixed(0) + "%", margin: { l: 70 }, ariaLabel: "training and test error" });
    cvBox = htmlEl("div", "", "", extra);
    const row = htmlEl("div", "qb-presets", "", cvBox);
    button(row, "5-fold CV over (C, γ)", () => runCV());
    cvMap = htmlEl("div", "", "", cvBox);
    htmlEl("div", "qb-note", "Cross-validated error of the training points (rows γ, columns C); the outlined cell is the best.", cvBox);
  }

  const gBg = svgEl("g", {}, pg.overlay);
  const gTest = svgEl("g", {}, pg.overlay);
  const gTop = svgEl("g", {}, pg.top);

  function spec(): KernelSpec {
    if (st.kernel === "linear") return { type: "linear" };
    if (st.kernel === "poly") { const s = Math.max(Math.abs(W.lo), Math.abs(W.hi)); return { type: "poly", degree: st.degree, coef0: 1, gamma: 1 / (s * s) }; }
    return { type: "rbf", gamma: p10(st.logG) };
  }

  function train(pts: LPoint[], C: number, sp: KernelSpec): SVMModel {
    const X = new Float64Array(2 * pts.length);
    pts.forEach((p, i) => { X[2 * i] = p.x; X[2 * i + 1] = p.y; });
    return trainSVM(X, pts.length, 2, pts.map(p => (p.c === 1 ? 1 : -1)), { kernel: sp, C, tol: 1e-4 });
  }
  const fAt = (m: SVMModel, x: number, y: number, kf: ReturnType<typeof kernelFn>) => decision(m, [x, y], 0, kf);
  const errRate = (m: SVMModel, pts: LPoint[]) => {
    const kf = kernelFn(m.kernel);
    return pts.length ? pts.filter(p => (p.c === 1 ? 1 : -1) * fAt(m, p.x, p.y, kf) <= 0).length / pts.length : NaN;
  };

  function syncControls(): void {
    kBtns.forEach((b, i) => b.classList.toggle("active", kernels[i] === st.kernel));
    gBox.style.display = st.kernel === "rbf" ? "" : "none";
    dBox.style.display = st.kernel === "poly" ? "" : "none";
    if (bumpCb) bumpCb.parentElement!.style.display = st.kernel === "rbf" ? "" : "none";
    cS.set(st.logC); gS.set(st.logG); dS.set(st.degree);
  }

  let raf = 0;
  function update(final: boolean): void {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => draw(final));
  }

  function draw(final: boolean): void {
    const pts = W.points(), sp = spec(), C = p10(st.logC);
    gBg.replaceChildren(); gTop.replaceChildren();
    const nPos = pts.filter(p => p.c === 1).length;
    if (nPos === 0 || nPos === pts.length) {
      model = null; pg.setBackgroundColour(null);
      ro.textContent = "add points of both classes";
      return;
    }
    model = train(pts, C, sp);
    const m = model, kf = kernelFn(sp);
    const grid = sampleGrid(W, RES, (x, y) => fAt(m, x, y, kf));
    pg.setBackgroundColour(signColour(W, colors, (x, y) => {
      const i = Math.round(W.toU(x) * (RES - 1)), j = Math.round((1 - W.toU(y)) * (RES - 1));
      return grid[Math.max(0, Math.min(RES - 1, j)) * RES + Math.max(0, Math.min(RES - 1, i))];
    }, { street: true }));
    // contours
    pathEl(gBg, contourPath(W, pg, grid, RES, 0), "var(--qb-fg, #222)", 3.2);
    pathEl(gBg, contourPath(W, pg, grid, RES, 1), colors[1], 2.4, "8 6");
    pathEl(gBg, contourPath(W, pg, grid, RES, -1), colors[0], 2.4, "8 6");
    // RBF bumps
    if (st.kernel === "rbf" && st.bumps) {
      const r = 1 / Math.sqrt(p10(st.logG)), rpx = Math.abs(W.X(r) - W.X(0));
      for (let q = 0; q < m.sv.length; q++) {
        const p = pts[m.sv[q]];
        const c = svgEl("circle", { cx: W.X(p.x), cy: W.Y(p.y), r: rpx, fill: colors[p.c] }, gBg);
        c.style.setProperty("fill-opacity", "0.07"); c.style.setProperty("stroke", colors[p.c]); c.style.setProperty("stroke-opacity", "0.5");
      }
    }
    // slack segments (linear)
    let nIn = 0, nEdge = 0;
    const ny = (p: LPoint) => (p.c === 1 ? 1 : -1);
    if (m.w) {
      const n2 = m.w[0] ** 2 + m.w[1] ** 2;
      for (const p of pts) {
        const yf = ny(p) * fAt(m, p.x, p.y, kf);
        if (yf < 1 - 1e-3 && n2 > 0) {
          const t = (1 - yf) * ny(p) / n2, tx = p.x + t * m.w[0], ty = p.y + t * m.w[1];
          const l = svgEl("line", { x1: W.X(p.x), y1: W.Y(p.y), x2: W.X(tx), y2: W.Y(ty) }, gTop);
          l.style.setProperty("stroke", "#e6a700"); l.style.setProperty("stroke-width", "4"); l.style.setProperty("stroke-linecap", "round");
        }
      }
    }
    // support vector rings
    for (let q = 0; q < m.sv.length; q++) {
      const i = m.sv[q], p = pts[i], bound = m.alpha[i] >= m.Ci[i] * (1 - 1e-6);
      if (bound) nIn++; else nEdge++;
      const r = svgEl("circle", { cx: W.X(p.x), cy: W.Y(p.y), r: 13, fill: "none" }, gTop);
      r.style.setProperty("stroke", "var(--qb-fg, #222)"); r.style.setProperty("stroke-width", "2.5");
      if (bound) r.style.setProperty("stroke-dasharray", "4 3");
    }
    // test points
    gTest.replaceChildren();
    if (cvMode) for (const p of test) {
      const c = svgEl("circle", { cx: W.X(p.x), cy: W.Y(p.y), r: 6, fill: "none" }, gTest);
      c.style.setProperty("stroke", colors[p.c]); c.style.setProperty("stroke-width", "2.5");
    }
    // readout
    const trainErr = errRate(m, pts);
    let s = `kernel function: ${KNAMES[st.kernel]}` + (st.kernel === "rbf" ? `\nγ = ${fmtNum(p10(st.logG))}, length 1/√γ = ${fmtNum(1 / Math.sqrt(p10(st.logG)))}` : st.kernel === "poly" ? `, degree ${st.degree}` : "");
    s += `\nC = ${fmtNum(C)}`;
    if (m.w) s += `\nf(x) = ${m.w[0].toFixed(2)} x₁ ${m.w[1] >= 0 ? "+" : "−"} ${Math.abs(m.w[1]).toFixed(2)} x₂ ${m.b >= 0 ? "+" : "−"} ${Math.abs(m.b).toFixed(2)}\nstreet width 2/‖w‖ = ${Math.hypot(m.w[0], m.w[1]) < 1e-3 ? "∞ (w ≈ 0: no line helps)" : marginWidth(m).toFixed(2)}`;
    s += `\nsupport vectors: ${m.sv.length}\n  ${nEdge} on the edge, ${nIn} inside or wrong`;
    s += `\ntraining errors: ${Math.round(trainErr * pts.length)} of ${pts.length}`;
    if (cvMode) s += `\ntest errors: ${Math.round(errRate(m, test) * test.length)} of ${test.length} (${(100 * errRate(m, test)).toFixed(1)}%)`;
    if (!m.converged) s += "\n(solver stopped early)";
    ro.textContent = s;
    note.textContent = preset.note;
    if (final) { drawKernelMatrix(pts, sp); drawCurve(); }
  }

  function drawKernelMatrix(pts: LPoint[], sp: KernelSpec): void {
    kmBox.replaceChildren();
    if (!st.kmat) { kmBox.style.display = "none"; return; }
    kmBox.style.display = "";
    const P = [...pts].sort((a, b) => a.c - b.c).slice(0, 80), n = P.length;
    const X = new Float64Array(2 * n);
    P.forEach((p, i) => { X[2 * i] = p.x; X[2 * i + 1] = p.y; });
    const K = kernelMatrix(sp, X, n, 2);
    const nNeg = P.filter(p => p.c === 0).length;
    heatmapGrid(kmBox, K, n, n, { map: viridis, numbers: false, gridLines: false, cell: Math.max(4, Math.floor(360 / n)), title: `kernel matrix K (${nNeg} blue, then ${n - nNeg} red)`, font: 17 });
  }

  function drawCurve(): void {
    if (!curve) return;
    sweepBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.sweep === "C")));
    const pts = W.points();
    curve.clear();
    const xs = st.sweep === "C" ? logGrid(-2, 3, 4) : logGrid(-2, 2, 4);
    curve.setDomain([xs[0], xs[xs.length - 1]]);
    curve.setLabels({ xlabel: st.sweep === "C" ? `C   (γ = ${fmtNum(p10(st.logG))})` : `γ   (C = ${fmtNum(p10(st.logC))})` });
    if (pts.filter(p => p.c === 1).length === 0 || pts.filter(p => p.c === 0).length === 0) return;
    const tr: number[] = [], te: number[] = [];
    for (const v of xs) {
      const sp: KernelSpec = st.kernel === "rbf" ? { type: "rbf", gamma: st.sweep === "gamma" ? v : p10(st.logG) } : spec();
      const m = train(pts, st.sweep === "C" ? v : p10(st.logC), sp);
      tr.push(errRate(m, pts)); te.push(errRate(m, test));
    }
    curve.line(xs, tr, { color: seriesColor(0), label: "training error" });
    curve.line(xs, te, { color: seriesColor(1), dash: "7 4", label: "test error" });
    curve.vline(st.sweep === "C" ? p10(st.logC) : p10(st.logG), { color: "var(--qb-muted)", dash: "3 3", width: 2 });
    curve.legend(undefined, "tr");
  }

  function runCV(): void {
    if (!cvMap) return;
    const pts = W.points(), n = pts.length;
    const cls = pts.map(p => p.c), X = new Float64Array(2 * n);
    pts.forEach((p, i) => { X[2 * i] = p.x; X[2 * i + 1] = p.y; });
    const Cs = logGrid(-1, 3), gs = logGrid(-2, 2), folds = makeFolds(n, 5, cls);
    const E = new Float64Array(Cs.length * gs.length);
    gs.forEach((g, gi) => Cs.forEach((C, ci) => {
      E[gi * Cs.length + ci] = cvError(n, cls, folds, 5, trIdx => {
        const sub = trIdx.map(i => pts[i]);
        const m = train(sub, C, { type: "rbf", gamma: g }), kf = kernelFn(m.kernel);
        return i => (fAt(m, pts[i].x, pts[i].y, kf) > 0 ? 1 : 0);
      });
    }));
    let best = 0;
    E.forEach((v, i) => { if (v < E[best] - 1e-12) best = i; });
    cvMap.replaceChildren();
    const hm = heatmapGrid(cvMap, E, gs.length, Cs.length, {
      map: viridis, lo: 0, hi: 0.5, fmt: v => (100 * v).toFixed(0) + "%", cell: 56, font: 17,
      rowLabels: gs.map(g => "γ " + fmtNum(g)), colLabels: Cs.map(c => fmtNum(c)), title: "CV error (columns: C)",
    });
    const cell = 56, f = 17, lw = Math.max(...gs.map(g => ("γ " + fmtNum(g)).length)) * f * 0.6 + 10, th = f * 1.5 + f * 1.4;
    const r = Math.floor(best / Cs.length), c = best % Cs.length;
    const box = svgEl("rect", { x: lw + c * cell + 1.5, y: th + r * cell + 1.5, width: cell - 3, height: cell - 3, fill: "none" }, hm.svg);
    box.style.setProperty("stroke", "#e6a700"); box.style.setProperty("stroke-width", "4");
    const use = button(cvMap, `use the best: C = ${fmtNum(Cs[c])}, γ = ${fmtNum(gs[r])}`, () => {
      st.kernel = "rbf"; st.logC = Math.log10(Cs[c]); st.logG = Math.log10(gs[r]); syncControls(); update(true);
    });
    use.style.marginTop = ".3em";
  }

  function load(p: SVMPreset): void {
    preset = p;
    st.kernel = kernels.includes(p.kernel) ? p.kernel : kernels[0];
    st.logC = p.logC; st.logG = p.logGamma; st.degree = p.degree;
    presetBtns.forEach((b, i) => b.classList.toggle("active", list[i] === p));
    W.setRange(p.range[0], p.range[1]);
    test = cvMode ? presetTest(p) : [];
    if (cvMap) cvMap.replaceChildren();
    syncControls();
    W.set(p.points());
  }
  if (bumpCb) bumpCb.checked = st.bumps;
  if (kmCb) kmCb.checked = st.kmat;
  load(p0);
}

autoMount(NAME, mount);
