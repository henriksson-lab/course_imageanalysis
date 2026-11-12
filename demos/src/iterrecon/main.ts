// Iterative reconstruction. A 128 x 128 phantom is measured at a chosen number of angles,
// angular range and dose; it is reconstructed by FBP (Hann) for comparison, and by ART
// (one sweep over all rays per iteration) or SIRT (one simultaneous update per iteration),
// optionally with non-negativity and a TV denoising step after each data step (weight lambda).
// Panels: reconstruction, error vs truth, and the RMSE vs truth and the data residual per
// iteration (semi-convergence: the residual keeps falling while the error turns up).
// Auto-mounts on <div data-demo="iterrecon">. Options: data-preset="sirt1|sirt|noisy|fewviews|wedge|cartoon".
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { diverging } from "../lib/colour";
import * as R from "../lib/radon";
import { ObjectName, OBJECT_LABELS, objectImage, cellObject, imagePanel, injectStyle, baseCss, layout, CELL_CREDIT } from "../radonproj/view";

const NAME = "iterrecon";
const N = 128;
const DOSES = [Infinity, 1e5, 3e4, 1e4, 3e3, 1e3];
const LAMBDAS = [0, 0.001, 0.003, 0.01, 0.02, 0.05, 0.1];

type Method = "fbp" | "art" | "sirt";
interface Settings { obj: ObjectName; nA: number; range: number; dose: number; method: Method; omega: number; nonneg: boolean; tv: number; target: number }
interface Preset { id: string; label: string; s: Settings }
const base: Settings = { obj: "shepp", nA: 60, range: 180, dose: Infinity, method: "sirt", omega: 1, nonneg: false, tv: 0, target: 50 };
const PRESETS: Preset[] = [
  { id: "sirt1", label: "SIRT step 1 = back-projection", s: { ...base, target: 1 } },
  { id: "sirt", label: "SIRT converging", s: { ...base, target: 100 } },
  { id: "noisy", label: "noisy: stop early", s: { ...base, method: "art", omega: 0.5, dose: 3e3, target: 30 } },
  { id: "fewviews", label: "30 views: FBP streaks vs TV clean", s: { ...base, method: "art", omega: 0.5, nonneg: true, tv: 0.003, nA: 30, target: 30 } },
  { id: "wedge", label: "±60° wedge: iterate with priors", s: { ...base, method: "art", omega: 0.5, nonneg: true, tv: 0.003, nA: 61, range: 120, target: 30 } },
  { id: "noisytv", label: "noisy + TV", s: { ...base, method: "art", omega: 0.5, nonneg: true, tv: 0.01, dose: 3e3, target: 30 } },
  { id: "cartoon", label: "TV too strong: cartoon", s: { ...base, method: "art", omega: 0.5, nonneg: true, tv: 0.05, dose: 3e3, target: 30 } },
];

const STYLE = baseCss(NAME);

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const pAttr = PRESETS.find(p => p.id === host.dataset.preset);
  const start: Settings = pAttr ? { ...pAttr.s } : { ...base };
  const st: Settings = { ...start };

  const L = layout(host, 3);
  const recP = imagePanel(L.figs, "reconstruction", N, 0, "rd-fig");
  const credit = htmlEl("div", "rd-credit", CELL_CREDIT, recP.fig);
  const errP = imagePanel(L.figs, "error", N, 0, "rd-fig");
  const plotFig = plotFigure(L.figs, "error and residual per iteration", "rd-fig wide");
  const plot = new Plot(plotFig, { width: 400, height: 300, x: [0, 10], y: [0.003, 1], ylog: true, xlabel: "iteration", ylabel: "" });
  const [colA, colB, colC] = L.cols;

  htmlEl("div", "rd-lab", "presets", colA);
  const presetBtns = presets(colA, PRESETS.map(p => ({ label: p.label, apply: () => load(p.s, p) })));

  htmlEl("div", "rd-lab", "measurement", colB);
  const objs: ObjectName[] = ["shepp", "disc", "cell"];
  const objRow = htmlEl("div", "qb-presets", "", colB);
  const objBtns = objs.map(o => button(objRow, OBJECT_LABELS[o], () => { st.obj = o; measure(); }));
  const aS = slider(colB, "angles", 4, 180, 1, st.nA, v => { st.nA = v; measure(); });
  const rS = slider(colB, "angular range", 90, 180, 2, st.range, v => { st.range = v; measure(); }, v => (v >= 180 ? "180°" : `±${v / 2}°`));
  const dS = slider(colB, "photons per ray", 0, DOSES.length - 1, 1, DOSES.indexOf(st.dose), v => { st.dose = DOSES[v]; measure(); },
    v => (Number.isFinite(DOSES[v]) ? DOSES[v].toExponential(0).replace("e+", "e") : "∞"));

  htmlEl("div", "rd-lab", "method", colC);
  const mRow = htmlEl("div", "qb-presets", "", colC);
  const methods: Method[] = ["fbp", "art", "sirt"];
  const mBtns = methods.map(m => button(mRow, m === "fbp" ? "FBP (Hann)" : m.toUpperCase(), () => { st.method = m; restart(); }));
  const oS = slider(colC, "relaxation ω", 0.1, 1.9, 0.05, st.omega, v => { st.omega = v; restart(); }, v => v.toFixed(2));
  const opt = htmlEl("div", "rd-row", "", colC);
  const nnCb = checkbox(opt, "non-negative", st.nonneg, v => { st.nonneg = v; restart(); });
  const tS = slider(colC, "TV weight λ", 0, LAMBDAS.length - 1, 1, LAMBDAS.indexOf(st.tv), v => { st.tv = LAMBDAS[v]; restart(); }, v => (LAMBDAS[v] === 0 ? "off" : String(LAMBDAS[v])));
  const ctl = htmlEl("div", "rd-row", "", colC);
  const playBtn = button(ctl, "▶ run", () => (running ? stop() : run(st.method === "art" ? solver!.iter + 30 : solver!.iter + 100)));
  button(ctl, "+1 iteration", () => { stop(); run(solver!.iter + 1); });
  button(ctl, "restart", () => restart());
  button(ctl, "reset", () => load(start));
  const ro = htmlEl("div", "qb-readout", "", colC);

  let truth = new Float64Array(N * N) as Float64Array, tmax = 1, g = R.geometry(N, R.angleSet(60)), b = new Float64Array(0) as Float64Array;
  let fbpRec = new Float64Array(N * N) as Float64Array, fbpErr = 0, starved = 0;
  let solver: R.IterativeRecon | null = null;
  let errHist: number[] = [], resHist: number[] = [];
  const mask = R.fovMask(N);
  let running = false, timer = 0;

  const errLine = plot.line([0], [1], { color: "var(--qb-c2)", width: 3, label: "RMSE vs truth" });
  const resLine = plot.line([0], [1], { color: "var(--qb-c1)", width: 3, label: "data residual (relative)" });
  const fbpLine = plot.hline(1, { color: "var(--qb-c2)", width: 2, dash: "6 4", text: "FBP RMSE" });
  plot.legend(undefined, "tr");

  function measure(): void {
    if (st.obj === "cell") { cellObject().then(() => measure2()); return; }
    measure2();
  }
  function measure2(): void {
    truth = objectImage(st.obj, N); tmax = Math.max(...truth);
    const ang = st.range >= 180 ? R.angleSet(st.nA) : R.angleRange(-st.range / 2, st.range / 2, st.nA);
    g = R.geometry(N, ang);
    const ns = R.noisySinogram(R.project(truth, g), st.dose, 7);
    b = ns.sino; starved = ns.starved;
    fbpRec = R.fbp(b, g, { kind: "hann" });
    fbpErr = R.rmseOf(fbpRec, truth, mask);
    restart();
  }

  function restart(): void {
    stop();
    solver = new R.IterativeRecon(b, g, { method: st.method === "sirt" ? "sirt" : "art", omega: st.omega, nonneg: st.nonneg, tv: st.tv, tvIters: 20 });
    errHist = [R.rmseOf(solver.x, truth, mask)]; resHist = [1];
    sync(); draw();
    if (st.method !== "fbp" && st.target > 0) run(st.target);
  }

  function run(to: number): void {
    if (st.method === "fbp" || !solver) return;
    running = true; playBtn.textContent = "❚❚ pause";
    const tick = () => {
      if (!running || !solver) return;
      solver.step();
      errHist.push(R.rmseOf(solver.x, truth, mask));
      resHist.push(solver.residual());
      draw();
      if (solver.iter >= to) { stop(); return; }
      timer = window.setTimeout(tick, 0);
    };
    timer = window.setTimeout(tick, 0);
  }
  function stop(): void { running = false; playBtn.textContent = "▶ run"; clearTimeout(timer); }

  function sync(): void {
    objBtns.forEach((bt, i) => bt.classList.toggle("active", objs[i] === st.obj));
    mBtns.forEach((bt, i) => bt.classList.toggle("active", methods[i] === st.method));
    aS.set(st.nA); rS.set(st.range); dS.set(DOSES.indexOf(st.dose)); oS.set(st.omega); nnCb.checked = st.nonneg; tS.set(LAMBDAS.indexOf(st.tv));
    credit.style.display = st.obj === "cell" ? "" : "none";
  }

  function draw(): void {
    const fbp = st.method === "fbp", x = fbp ? fbpRec : solver!.x;
    const k = fbp ? 0 : solver!.iter;
    recP.show(x, 0, tmax);
    recP.cap.textContent = fbp ? "FBP (Hann)" : `${st.method.toUpperCase()}${st.nonneg ? " + non-neg." : ""}${st.tv ? " + TV" : ""}, iteration ${k}`;
    const err = x.map((v, i) => (v - truth[i]) * mask[i]);
    errP.show(err, -0.5 * tmax, 0.5 * tmax, diverging);
    errP.cap.textContent = "error (reconstruction − truth)";
    const its = errHist.map((_, i) => i);
    plot.setDomain([0, Math.max(10, errHist.length - 1)], undefined);
    errLine.set(its, errHist); resLine.set(its.slice(1), resHist.slice(1));
    fbpLine.set(fbpErr);
    let best = 0;
    errHist.forEach((e, i) => { if (e < errHist[best]) best = i; });
    const cur = errHist[errHist.length - 1];
    ro.textContent = fbp ? `FBP (Hann): RMSE ${fbpErr.toFixed(4)}` :
      `iteration ${k}: RMSE ${cur.toFixed(4)}, residual ${(resHist[resHist.length - 1]).toFixed(4)}\nlowest RMSE ${errHist[best].toFixed(4)} at iteration ${best}\nFBP (Hann) RMSE ${fbpErr.toFixed(4)}` +
      (starved ? `\nphoton starvation: ${starved} rays` : "");
  }

  function load(s: Settings, p?: Preset): void {
    stop();
    Object.assign(st, s);
    presetBtns.forEach((bt, i) => bt.classList.toggle("active", PRESETS[i] === p));
    measure();
  }
  load(start, pAttr);
}

autoMount(NAME, mount);
