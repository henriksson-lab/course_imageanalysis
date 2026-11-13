// Bundle adjustment in flatland (lib/flatland.ts, lib/bundle.ts): 8 cameras around an object
// outline with 48 points, noisy 1D observations (σ = 0.2 px). The estimate starts from the
// truth plus a seeded perturbation; Gauss–Newton or Levenberg–Marquardt steps pull the bundle
// of rays (one per observation, drawn from its estimated camera) onto the estimated points.
// Side: RMS reprojection error per step (log scale) and the sparsity pattern of J or JᵀJ.
// Gauge: camera 1 and the x coordinate of camera 2 fixed, or nothing fixed (JᵀJ then has 4
// zero eigenvalues). Outliers with squared or Huber loss; 2σ covariance ellipses; and an
// incremental mode (initial pair → triangulate → resect the next camera → triangulate → BA).
//
// <div data-demo="bundleadjust" data-preset="small|large|gauge|outlier|huber|far|incremental"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { P, project1d, rayDir } from "../lib/flatland";
import { FlatBA, gaugeMask, lmStep, robustCost, pointCovariances, normalEigenvalues, Loss } from "../lib/bundle";
import { Plot } from "../lib/plot";
import { viridis } from "../lib/colour";
import { FlatView, injectStyle, CAM_COLORS } from "../triangulate/flatview";
import { makeWorld, perturbed, BAWorld, alignSimilarity, startIncremental, stepIncremental, Incremental, gaugeFor } from "./model";

const NAME = "bundleadjust";
const SEL = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${SEL} { flex-wrap: nowrap; gap: .8em; }
${SEL} .ba-main { flex: 0 1 470px; min-width: 260px; }
${SEL} .ba-side { flex: 0 1 300px; min-width: 220px; display: flex; flex-direction: column; gap: .3em; }
${SEL} .qb-panel { flex: 1 1 280px; max-width: 340px; gap: .35em; }
${SEL} .qb-readout { font-size: 15px; }
${SEL} .ba-tabs { display: flex; gap: .3em; }
.reveal ${SEL} { font-size: 18px; }
.reveal ${SEL} .ba-main { flex: 0 0 450px; }
.reveal ${SEL} .ba-side { flex: 0 0 290px; }
.reveal ${SEL} .qb-panel { gap: .2em; }
.reveal ${SEL} .qb-btn { font-size: 16px; padding: .08em .4em; }
.reveal ${SEL} .qb-readout { font-size: 16px; padding: .25em .45em; }
@media (max-width: 860px) { ${SEL} { flex-wrap: wrap; } }
`;

const SIGMA = 0.2, HUBER = 1;
interface State {
  perturb: number; seed: number; gauge: boolean; outliers: number; loss: Loss; far: boolean;
  ellipses: boolean; emag: number; rays: boolean; sparsity: "J" | "JtJ"; incremental: boolean;
}
const DEFAULT: State = { perturb: 0.05, seed: 5, gauge: true, outliers: 0, loss: "squared", far: false, ellipses: false, emag: 10, rays: true, sparsity: "J", incremental: false };
interface Preset { id: string; label: string; s: Partial<State>; run?: boolean }
const PRESETS: Preset[] = [
  { id: "small", label: "small perturbation: a few steps", s: {} },
  { id: "large", label: "large perturbation: GN diverges, LM copes", s: { perturb: 0.6, seed: 10 } },
  { id: "gauge", label: "no gauge fixed", s: { gauge: false } },
  { id: "outlier", label: "3 outliers, squared loss", s: { outliers: 3, perturb: 0.05 }, run: true },
  { id: "huber", label: "3 outliers, Huber loss", s: { outliers: 3, perturb: 0.05, loss: "huber" }, run: true },
  { id: "far", label: "cameras on one side: long ellipses", s: { far: true, perturb: 0, ellipses: true, emag: 1 } },
  { id: "incremental", label: "incremental reconstruction", s: { incremental: true, perturb: 0 } },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(p0?.s ?? {}) };

  const main = htmlEl("div", "ba-main", "", host);
  const view = new FlatView(main, 470, 470, -4, 4, -4.3, 3.7);
  const side = htmlEl("div", "ba-side", "", host);
  const rmsPlot = new Plot(side, { width: 300, height: 220, x: [0, 10], y: [0.1, 100], ylog: true, xlabel: "step", ylabel: "RMS error (px)", font: 17, margin: { l: 64, r: 10, t: 8, b: 50 } });
  const rmsLine = rmsPlot.line([], [], { color: "var(--qb-c1)", width: 2.5 });
  const rmsDots = rmsPlot.scatter([], [], { color: "var(--qb-c1)", r: 4 });
  const noiseLine = rmsPlot.hline(SIGMA, { color: "var(--qb-muted)", dash: "5 4", width: 1.5, text: "noise σ" });
  void noiseLine;
  const tabs = htmlEl("div", "ba-tabs", "", side);
  const tJ = button(tabs, "sparsity of J", () => { st.sparsity = "J"; drawSparsity(); });
  const tA = button(tabs, "of JᵀJ", () => { st.sparsity = "JtJ"; drawSparsity(); });
  const spDiv = htmlEl("div", "", "", side);
  const ro = htmlEl("div", "qb-readout", "", side);

  const panel = htmlEl("div", "qb-panel", "", host);
  const pBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const stepRow = htmlEl("div", "qb-presets", "", panel);
  const gnB = button(stepRow, "GN step", () => { stop(); doStep(false); });
  const lmB = button(stepRow, "LM step", () => { stop(); doStep(true); });
  const runB = button(stepRow, "run LM", () => { if (timer) stop(); else run(); });
  const incB = button(stepRow, "next camera", () => { stop(); if (inc && stepIncremental(world, inc)) { theta = Float64Array.from(inc.theta); hist.push(rms()); } draw(); });
  const pertS = slider(panel, "perturbation", 0, 0.8, 0.05, st.perturb, v => { st.perturb = v; restart(); });
  const gaugeC = checkbox(panel, "fix the gauge (camera 1 + scale)", st.gauge, v => { st.gauge = v; restart(); });
  const lossRow = htmlEl("div", "qb-presets", "", panel);
  const lossBtns = [button(lossRow, "squared loss", () => { st.loss = "squared"; draw(); }), button(lossRow, "Huber loss", () => { st.loss = "huber"; draw(); })];
  const outC = checkbox(panel, "3 outlier observations", st.outliers > 0, v => { st.outliers = v ? 3 : 0; rebuild(); });
  const ellC = checkbox(panel, "covariance ellipses (2σ)", st.ellipses, v => { st.ellipses = v; draw(); });
  const eS = slider(panel, "ellipses ×", 1, 20, 1, st.emag, v => { st.emag = v; draw(); });
  const raysC = checkbox(panel, "rays of the observations", st.rays, v => { st.rays = v; draw(); });
  button(panel, "reset", () => { stop(); Object.assign(st, DEFAULT, p0?.s ?? {}); pBtns.forEach(b => b.classList.remove("active")); sync(); rebuild(); });

  let world: BAWorld, prob: FlatBA, fixed: boolean[], theta: Float64Array, mu = 1e-2, hist: number[] = [], last = "", inc: Incremental | null = null;
  let timer = 0;
  function stop(): void { if (timer) { clearInterval(timer); timer = 0; } runB.textContent = "run LM"; }
  function run(): void {
    runB.textContent = "pause";
    timer = window.setInterval(() => {
      const before = hist[hist.length - 1];
      doStep(true);
      const after = hist[hist.length - 1];
      if (hist.length > 40 || (Math.abs(before - after) < 1e-5 * before && last.startsWith("LM: accepted"))) stop();
    }, 280);
  }
  function load(p: Preset): void {
    stop();
    Object.assign(st, DEFAULT, p.s);
    sync(); rebuild();
    if (p.run) run();
  }
  function sync(): void { pertS.set(st.perturb); gaugeC.checked = st.gauge; outC.checked = st.outliers > 0; ellC.checked = st.ellipses; eS.set(st.emag); raysC.checked = st.rays; }

  function rebuild(): void {
    world = st.far ? makeWorld({ nCams: 6, arc: 2.0, radius: 4, outliers: st.outliers }) : makeWorld({ outliers: st.outliers });
    prob = new FlatBA(world.cams, world.pts.length, world.obs);
    restart();
  }
  function restart(): void {
    stop();
    fixed = gaugeFor(world, st.gauge);
    mu = 1e-2; last = "";
    if (st.incremental) { inc = startIncremental(world); theta = Float64Array.from(inc.theta); }
    else { inc = null; theta = perturbed(world, st.perturb, gaugeFor(world), st.seed); }
    hist = [rms()];
    draw();
  }
  const activeObs = () => world.obs.map((o, i) => i).filter(i => !inc || (inc.camOn[world.obs[i].cam] && inc.ptOn[world.obs[i].pt]));
  function rms(): number {
    const r = prob.evaluate(theta, false).r, idx = activeObs();
    return Math.sqrt(idx.reduce((a, i) => a + r[i] * r[i], 0) / Math.max(1, idx.length));
  }
  function doStep(lm: boolean): void {
    if (inc && inc.camOn.some(v => !v)) { last = "incremental: add all cameras first"; draw(); return; }
    const s = lmStep(prob, theta, { mu, lm, fixed, loss: st.loss, delta: HUBER });
    theta = s.theta; mu = lm ? s.mu : mu;
    last = lm ? `LM: ${s.accepted ? "accepted" : "rejected"}, μ → ${s.mu.toExponential(1)}` : "GN: full step";
    hist.push(rms());
    draw();
  }

  function draw(): void {
    view.clear();
    lossBtns.forEach((b, i) => b.classList.toggle("active", (i === 1) === (st.loss === "huber")));
    gnB.disabled = lmB.disabled = runB.disabled = false;
    incB.style.display = st.incremental ? "" : "none";
    const cams = prob.cams(theta), pts = prob.pts(theta);
    // the truth, faint
    view.poly(world.scene.polys[0], "#d9d9d9", "#bbb", 0.5, 1.2);
    world.cams.forEach(c => view.dot([c.x, c.y], 4, "#bbb"));
    const camOn = (j: number) => !inc || inc.camOn[j], ptOn = (i: number) => !inc || inc.ptOn[i];
    // rays
    const outSet = new Set(world.outliers);
    if (st.rays) world.obs.forEach((o, i) => {
      if (!camOn(o.cam) || !ptOn(o.pt)) return;
      const c = cams[o.cam], d = rayDir(c, o.u), p = pts[o.pt];
      const L = Math.hypot(p[0] - c.x, p[1] - c.y) + 0.25;
      const l = view.line([c.x, c.y], [c.x + L * d[0], c.y + L * d[1]], outSet.has(i) ? "#e8402a" : CAM_COLORS[o.cam], outSet.has(i) ? 2.2 : 1);
      l.style.opacity = outSet.has(i) ? "0.9" : "0.35";
    });
    // ellipses
    if (st.ellipses && (!inc || inc.camOn.every(v => v))) {
      const cov = pointCovariances(prob, theta, gaugeFor(world), SIGMA);
      cov.forEach((c, i) => { if (c) view.ellipse(pts[i], [c[0] * st.emag ** 2, c[1] * st.emag ** 2, c[2] * st.emag ** 2], 2, "#6a3d9a"); });
    }
    cams.forEach((c, j) => { if (camOn(j)) view.camera(c, CAM_COLORS[j % CAM_COLORS.length], { len: 0.45, label: String(j + 1) }); else view.camera(world.cams[j], "#999", { len: 0.45, label: String(j + 1), faded: true }); });
    pts.forEach((p, i) => { if (ptOn(i)) view.dot(p, 3.5, "#111"); });

    // plot
    const xs = hist.map((_, k) => k);
    const ymax = Math.max(2, ...hist.filter(Number.isFinite)) * 1.5, ymin = Math.min(0.1, ...hist) * 0.8;
    rmsPlot.setDomain([0, Math.max(5, hist.length - 1)], [Math.max(1e-2, ymin), Math.min(1e5, ymax)]);
    rmsLine.set(xs, hist.map(v => Math.min(v, 1e5)));
    rmsDots.set(xs, hist.map(v => Math.min(v, 1e5)));

    // readout
    const r = prob.evaluate(theta, false).r;
    const cost = robustCost(r, st.loss, HUBER);
    let errTxt = "";
    if (!inc || inc.ptOn.every(v => v)) {
      const est = st.gauge ? pts : alignSimilarity(pts, world.pts).mapped;
      const e = Math.sqrt(est.reduce((a, p, i) => a + (p[0] - world.pts[i][0]) ** 2 + (p[1] - world.pts[i][1]) ** 2, 0) / est.length);
      errTxt = `point error vs truth: ${e.toFixed(4)}${st.gauge ? "" : " (after aligning)"}\n`;
    }
    let eigTxt = "";
    if (!st.gauge && !inc) {
      const ev = normalEigenvalues(prob, theta, fixed);
      eigTxt = `smallest eigenvalues of JᵀJ:\n${Array.from(ev.slice(0, 5)).map(v => v.toExponential(1)).join("  ")}\n`;
    }
    let outTxt = "";
    if (world.outliers.length) {
      const inl = Array.from(r).filter((_, i) => !world.outliers.includes(i));
      outTxt = `outlier residuals: ${world.outliers.map(i => r[i].toFixed(1)).join(", ")} px; inliers RMS ${Math.sqrt(inl.reduce((a, b) => a + b * b, 0) / inl.length).toFixed(2)} px\n`;
    }
    ro.textContent = `step ${hist.length - 1}: RMS ${hist[hist.length - 1].toFixed(3)} px, cost ${cost.toFixed(1)}\n` + outTxt +
      (last ? last + "\n" : "") + errTxt + eigTxt +
      `${world.cams.length} cameras × 3 + ${world.pts.length} points × 2 = ${prob.nParams} unknowns, ${world.obs.length} observations` +
      (inc ? `\n${inc.log}` : "");
    drawSparsity();
  }

  function drawSparsity(): void {
    tJ.classList.toggle("active", st.sparsity === "J"); tA.classList.toggle("active", st.sparsity === "JtJ");
    spDiv.replaceChildren();
    const { J } = prob.evaluate(theta, true), m = world.obs.length, n = prob.nParams;
    let vals: Float32Array, nx = n, ny = m;
    if (st.sparsity === "J") vals = Float32Array.from(J!, v => (v !== 0 ? 1 : 0));
    else {
      vals = new Float32Array(n * n); ny = n;
      for (let i = 0; i < m; i++) for (let a = 0; a < n; a++) if (J![i * n + a] !== 0) for (let b = 0; b < n; b++) if (J![i * n + b] !== 0) vals[a * n + b] = 1;
    }
    const h = Math.round(Math.min(host.closest(".reveal") ? 130 : 260, 280 * ny / nx));
    const pl = new Plot(spDiv, { width: 300, height: h + 10, x: [0, nx], y: [0, ny], grid: false, xticks: [], yticks: [], margin: { l: 6, r: 6, t: 4, b: 6 }, font: 15 });
    pl.heatmap(vals, nx, ny, { map: t => (t > 0.5 ? [0.12, 0.47, 0.71] : [0.96, 0.96, 0.96]), lo: 0, hi: 1, flipY: true });
    void viridis;
    htmlEl("div", "qb-note", st.sparsity === "J" ? `J: ${m} rows (observations) × ${n} columns (cameras | points)` : `JᵀJ: ${n} × ${n} (cameras | points)`, spDiv);
  }

  sync();
  if (p0) pBtns[PRESETS.indexOf(p0)].classList.add("active");
  rebuild();
  if (p0?.run) run();
}

autoMount(NAME, mount);
