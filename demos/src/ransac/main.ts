// RANSAC step by step. A point set (inliers on a line or a circle, plus uniform outliers;
// seeded) in a 100 x 100 box: click to add a point, click a point to remove it, or drag it.
// "One sample" draws a minimal sample (2 points for a line, 3 for a circle), fits it
// exactly, draws the band of width +-eps and counts the inliers; the best model so far is
// kept. "Run all" finishes the iterations; then the best model is refitted by least
// squares on its inliers. Beside it: inlier count per iteration, and the probability
// 1 - (1 - w^s)^N that at least one of N samples is all inliers. Options: least squares on
// all points (the baseline), sequential RANSAC for two models, and the line Hough
// accumulator of the same points.
//
// <div data-demo="ransac" data-preset="ls|seventeen|eighty|epsbig|twolines|circle"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl, Slider } from "../lib/ui";
import { Plot } from "../lib/plot";
import { inferno } from "../lib/colour";
import { ransac, sequentialRansac, LINE_MODEL, CIRCLE_MODEL, Model, Line, Circle, P2, lineTLS, circleAlgebraic, makePointSet, successProbability, iterationsNeeded, inliersOf, RansacResult } from "../lib/ransac";
import { lineGeometry, houghLines, linePeaks, thetaOf, rhoOf } from "../lib/hough";
import { injectStyle } from "../houghlines/view";

const NAME = "ransac";
const STYLE = `
[data-demo="${NAME}"] { gap: .7em; flex-wrap: nowrap; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin: 0 0 .1em; }
[data-demo="${NAME}"] .rs-main { flex: 0 1 435px; min-width: 240px; }
[data-demo="${NAME}"] .rs-mid { flex: 0 1 330px; min-width: 220px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 320px; gap: .3em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre-wrap; }
[data-demo="${NAME}"] .rs-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .rs-hough canvas { width: 100%; height: 150px; display: block; image-rendering: pixelated; border-radius: 3px; }
.qb-demo[data-demo="${NAME}"] .rs-main { flex: 0 1 420px; }
.qb-demo[data-demo="${NAME}"] .rs-mid { flex: 0 1 290px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 330px; gap: .25em; font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-btn { font-size: 13px; padding: .12em .5em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-presets { gap: .25em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] { flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { font-size: inherit; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel .qb-btn { font-size: 15px; padding: .06em .4em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .rs-main { flex: 0 0 400px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .15em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .06em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; padding: .2em .4em; line-height: 1.25; }
.reveal [data-demo="${NAME}"] .rs-hough canvas { height: 100px; }
.reveal .qb-demo[data-demo="${NAME}"] .rs-mid { gap: .1em; flex: 0 0 270px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { max-width: 450px; flex: 1 1 420px; }
.reveal [data-demo="${NAME}"] .qb-panel .qb-readout { max-height: 8.4em; overflow-y: auto; }
@media (max-width: 860px) { [data-demo="${NAME}"] { flex-wrap: wrap; } }
`;

type Kind = "line" | "circle";
interface State { kind: Kind; scene: "one" | "two" | "close"; npts: number; outliers: number; eps: number; iters: number; seed: number; ls: boolean; seq: boolean; hough: boolean }
interface Preset { id: string; label: string; s: Partial<State>; run?: boolean }
const DEFAULT: State = { kind: "line", scene: "one", npts: 60, outliers: 0.5, eps: 2, iters: 17, seed: 1, ls: true, seq: false, hough: false };
const PRESETS: Preset[] = [
  { id: "ls", label: "least squares fails", s: { outliers: 0.3, ls: true, iters: 17 } },
  { id: "seventeen", label: "RANSAC: 17 iterations", s: { outliers: 0.5, iters: 17 }, run: true },
  { id: "eighty", label: "80 % outliers", s: { npts: 100, outliers: 0.8, iters: 17, hough: true }, run: true },
  { id: "epsbig", label: "ε too large: two lines merge", s: { scene: "close", outliers: 0.2, eps: 6, iters: 40 }, run: true },
  { id: "twolines", label: "two lines: sequential", s: { scene: "two", outliers: 0.3, iters: 40, seq: true }, run: true },
  { id: "circle", label: "circle", s: { kind: "circle", outliers: 0.5, iters: 35 }, run: true },
];
const SIZE = 100, NPTS = 100;
const C_IN = "#2ca02c", C_OUT = "#9a9a9a", C_SAMPLE = "#d62728", C_BEST = "#1f77b4", C_LS = "#ff7f0e", C_TWO = "#9467bd";

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(p0?.s ?? {}) });
  const st: State = start();

  // ---- layout ----
  const mainFig = htmlEl("figure", "rs-main", "", host);
  const cap = htmlEl("figcaption", "", "", mainFig);
  const P = new Plot(mainFig, { width: 470, height: 470, x: [0, SIZE], y: [0, SIZE], equal: true, margin: { l: 40, b: 36, r: 8, t: 8 }, xticks: [0, 25, 50, 75, 100], yticks: [0, 25, 50, 75, 100] });
  const mid = htmlEl("div", "rs-mid", "", host);
  const itFig = htmlEl("figure", "", "", mid);
  htmlEl("figcaption", "", "inliers per sample", itFig);
  const itPlot = new Plot(itFig, { width: 330, height: 170, x: [0.5, 17.5], y: [0, NPTS], xlabel: "iteration", font: 16, margin: { l: 44, b: 44, r: 8, t: 8 } });
  const prFig = htmlEl("figure", "", "", mid);
  const prCap = htmlEl("figcaption", "", "", prFig);
  const prPlot = new Plot(prFig, { width: 330, height: 170, x: [0, 120], y: [0, 1], xlabel: "samples N", font: 16, margin: { l: 44, b: 44, r: 8, t: 8 }, yticks: [0, 0.5, 1] });
  const hFig = htmlEl("figure", "rs-hough", "", mid);
  htmlEl("figcaption", "", "Hough accumulator (θ →, ρ ↑)", hFig);
  const hCanvas = htmlEl("canvas", "", "", hFig);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "rs-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "rs-lab", "model", panel);
  const kindBtns = presets(panel, (["line", "circle"] as Kind[]).map(k => ({ label: k, apply: () => { st.kind = k; newPoints(); } })));
  const runRow = htmlEl("div", "qb-presets", "", panel);
  button(runRow, "one sample", () => { if (shown < steps()) shown++; draw(); });
  button(runRow, "run all", () => { shown = steps(); draw(); });
  button(runRow, "restart", () => { shown = 0; draw(); });
  const sl: Record<string, Slider> = {};
  sl.eps = slider(panel, "ε", 0.25, 10, 0.25, st.eps, v => { st.eps = v; rerun(); });
  sl.npts = slider(panel, "points", 20, 150, 10, st.npts, v => { st.npts = v; newPoints(); });
  sl.outliers = slider(panel, "outliers", 0, 0.9, 0.05, st.outliers, v => { st.outliers = v; newPoints(); }, v => `${Math.round(v * 100)} %`);
  sl.iters = slider(panel, "iterations", 1, 300, 1, st.iters, v => { st.iters = v; rerun(); });
  sl.seed = slider(panel, "seed", 1, 20, 1, st.seed, v => { st.seed = v; rerun(); });
  const lsC = checkbox(panel, "least squares (all points)", st.ls, v => { st.ls = v; draw(); });
  const seqC = checkbox(panel, "sequential: 2nd model", st.seq, v => { st.seq = v; rerun(); });
  const hC = checkbox(panel, "Hough accumulator", st.hough, v => { st.hough = v; draw(); });
  button(panel, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); newPoints(p0?.run ?? false); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  function load(p: Preset): void { Object.assign(st, DEFAULT, p.s); sync(); newPoints(p.run ?? false); }
  function sync(): void {
    sl.eps.set(st.eps); sl.npts.set(st.npts); sl.outliers.set(st.outliers); sl.iters.set(st.iters); sl.seed.set(st.seed);
    lsC.checked = st.ls; seqC.checked = st.seq; hC.checked = st.hough;
  }

  // ---- data and runs ----
  let pts: P2[] = [], isIn: boolean[] = [], wTrue = 0.5;
  let res: RansacResult<Line | Circle> | null = null, seqRes: RansacResult<Line | Circle>[] = [], shown = 0;
  const model = (): Model<Line | Circle> => (st.kind === "line" ? LINE_MODEL : CIRCLE_MODEL) as Model<Line | Circle>;
  const steps = () => (res ? res.steps.length : 0);
  function newPoints(runAll = false): void {
    kindBtns.forEach((b, i) => b.classList.toggle("active", ["line", "circle"][i] === st.kind));
    const models = st.kind === "circle" ? [{ kind: "circle" as const, a: 52, b: 48, r: 28 }]
      : st.scene === "two" ? [{ kind: "line" as const, theta: 1.25, rho: 62 }, { kind: "line" as const, theta: 2.3, rho: -10 }]
      : st.scene === "close" ? [{ kind: "line" as const, theta: 1.35, rho: 50 }, { kind: "line" as const, theta: 1.35, rho: 58 }]
      : [{ kind: "line" as const, theta: 1.1, rho: 58 }];
    const ps = makePointSet(st.npts, st.outliers, models, 0.8, SIZE, 11);
    pts = ps.pts; isIn = ps.inlier;
    rerun(runAll);
  }
  function rerun(runAll = false): void {
    wTrue = isIn.filter(Boolean).length / Math.max(1, pts.length);
    res = ransac(model(), pts, { eps: st.eps, iterations: st.iters, seed: st.seed });
    seqRes = st.seq ? sequentialRansac(model(), pts, { eps: st.eps, iterations: st.iters, seed: st.seed, maxModels: 2, minInliers: 6 }) : [];
    shown = runAll ? steps() : Math.min(shown, steps());
    draw();
  }

  // ---- interaction: click adds, click on a point removes, drag moves ----
  P.svg.style.touchAction = "none"; P.svg.style.cursor = "crosshair";
  let drag: { i: number; moved: boolean } | null = null;
  P.svg.addEventListener("pointerdown", ev => {
    const [x, y] = P.toData(ev);
    let bi = -1, bd = 4;
    pts.forEach((p, i) => { const d = Math.hypot(p.x - x, p.y - y); if (d < bd) { bd = d; bi = i; } });
    if (bi >= 0) { drag = { i: bi, moved: false }; P.svg.setPointerCapture(ev.pointerId); return; }
    if (x < 0 || y < 0 || x > SIZE || y > SIZE) return;
    pts.push({ x, y }); isIn.push(false);
    rerun(shown === steps());
  });
  P.svg.addEventListener("pointermove", ev => {
    if (!drag) return;
    const [x, y] = P.toData(ev);
    pts[drag.i] = { x: Math.max(0, Math.min(SIZE, x)), y: Math.max(0, Math.min(SIZE, y)) };
    drag.moved = true;
    rerun(shown === steps());
  });
  const up = () => {
    if (!drag) return;
    if (!drag.moved) { pts.splice(drag.i, 1); isIn.splice(drag.i, 1); rerun(shown === steps()); }
    drag = null;
  };
  P.svg.addEventListener("pointerup", up);
  P.svg.addEventListener("pointercancel", up);

  // ---- drawing helpers ----
  function drawModel(m: Line | Circle, color: string, width: number, dash?: string, band = 0): void {
    if ("theta" in m) {
      const c = Math.cos(m.theta), s = Math.sin(m.theta), D = 200;
      for (const off of band ? [-band, band] : [0]) {
        const fx = (m.rho + off) * c, fy = (m.rho + off) * s;
        P.line([fx + D * s, fx - D * s], [fy - D * c, fy + D * c], { color, width: band ? 1.5 : width, dash: band ? "5 4" : dash });
      }
      if (band) {
        const fx = m.rho * c, fy = m.rho * s;
        P.line([fx + D * s, fx - D * s], [fy - D * c, fy + D * c], { color, width, dash });
      }
    } else {
      const ring = (r: number, w: number, d?: string) => {
        const t = Array.from({ length: 121 }, (_, i) => 2 * Math.PI * i / 120);
        P.line(t.map(a => m.a + r * Math.cos(a)), t.map(a => m.b + r * Math.sin(a)), { color, width: w, dash: d });
      };
      if (band) { ring(Math.max(0, m.r - band), 1.5, "5 4"); ring(m.r + band, 1.5, "5 4"); }
      ring(m.r, width, dash);
    }
  }

  function draw(): void {
    P.clear(); itPlot.clear(); prPlot.clear();
    const M = model(), s = M.s, n = pts.length;
    const done = res !== null && shown === steps() && shown > 0;
    const cur = res && shown > 0 ? res.steps[shown - 1] : null;
    // best so far among the shown steps
    let best: Line | Circle | null = null, bestN = -1, bestIt = 0;
    if (res) for (let i = 0; i < shown; i++) { const t = res.steps[i]; if (t.model && t.inliers > bestN) { best = t.model; bestN = t.inliers; bestIt = t.iter; } }
    const bestIn = new Set(best ? inliersOf(M, best, pts, st.eps) : []);
    // least squares on everything
    let ls: Line | Circle | null = null;
    if (st.ls && n >= s) ls = st.kind === "line" ? lineTLS(pts) : circleAlgebraic(pts);
    if (ls) drawModel(ls, C_LS, 3, "9 5");
    // hypothesis of the current step with its band
    if (cur?.model && !done) drawModel(cur.model, C_SAMPLE, 2.5, undefined, st.eps);
    // best / final
    if (done && res?.refit) drawModel(res.refit, C_BEST, 4, undefined, st.eps);
    else if (best) drawModel(best, C_BEST, 3.5);
    if (done && seqRes.length > 1) drawModel(seqRes[1].refit ?? seqRes[1].best!, C_TWO, 4);
    // points
    const finalIn = new Set(done && res ? res.refitInliers : [...bestIn]);
    const second = new Set(done && seqRes.length > 1 ? seqRes[1].refitInliers : []);
    P.scatter(pts.map(p => p.x), pts.map(p => p.y), { colors: pts.map((_, i) => (second.has(i) ? C_TWO : finalIn.has(i) ? C_IN : C_OUT)), r: 5 });
    if (cur && !done) P.scatter(cur.sample.map(i => pts[i].x), cur.sample.map(i => pts[i].y), { color: C_SAMPLE, r: 9 });
    cap.textContent = done ? "after all iterations: best model refitted on its inliers" : cur ? `iteration ${cur.iter}: red = sample and its ±ε band` : "the points (click to add, click a point to remove)";

    // inliers per iteration
    const N = Math.max(1, st.iters);
    itPlot.setDomain([0.5, N + 0.5], [0, Math.max(10, n)]);
    if (res && shown) {
      const xs = res.steps.slice(0, shown).map(t => t.iter);
      itPlot.bars(xs, res.steps.slice(0, shown).map(t => t.inliers), { color: C_SAMPLE, barWidth: Math.min(0.8, 0.8 * 17 / N + 0.1) });
      itPlot.step([0.5, ...xs.map(x => x + 0.5)], [0, ...res.steps.slice(0, shown).map(t => t.best)], { color: C_BEST, width: 3 });
    }
    itPlot.hline(Math.round(wTrue * n), { color: C_IN, width: 2, dash: "5 4", text: "true inliers" });

    // probability curve
    const Nmax = Math.max(40, Math.min(1000, 1.5 * Math.max(st.iters, iterationsNeeded(wTrue, s) || 1)));
    prPlot.setDomain([0, Nmax], [0, 1]);
    prPlot.fn(x => successProbability(wTrue, s, x), { color: C_BEST, width: 3 });
    const need = iterationsNeeded(wTrue, s);
    if (Number.isFinite(need) && need <= Nmax) prPlot.vline(need, { color: C_IN, width: 2, dash: "5 4", text: `N = ${need}` });
    prPlot.scatter([shown], [successProbability(wTrue, s, shown)], { color: C_SAMPLE, r: 7 });
    prCap.textContent = `P(success after N), w = ${wTrue.toFixed(2)}, s = ${s}`;

    // Hough of the same points (lines only)
    hFig.style.display = st.hough && st.kind === "line" ? "" : "none";
    let houghLine: Line | null = null;
    if (st.hough && st.kind === "line") {
      const g = lineGeometry(SIZE + 1, SIZE + 1, 60, 3);
      const A = houghLines(pts.map(p => ({ x: p.x, y: SIZE - p.y })), g);
      hCanvas.width = g.nTheta; hCanvas.height = g.nRho;
      const ctx = hCanvas.getContext("2d")!, id = ctx.createImageData(g.nTheta, g.nRho);
      let mx = 1; for (const v of A.acc) mx = Math.max(mx, v);
      for (let j = 0; j < g.nTheta; j++) for (let k = 0; k < g.nRho; k++) {
        const [r, gg, b] = inferno(Math.sqrt(A.acc[j * g.nRho + k] / mx)), q = 4 * ((g.nRho - 1 - k) * g.nTheta + j);
        id.data[q] = 255 * r; id.data[q + 1] = 255 * gg; id.data[q + 2] = 255 * b; id.data[q + 3] = 255;
      }
      ctx.putImageData(id, 0, 0);
      const pk = linePeaks(A, 3, 3, 1)[0];
      if (pk) {
        // back to the plot frame: x = col, y = SIZE - row; the world origin of g is (50, 50)
        const th = thetaOf(g, pk.j), rh = rhoOf(g, pk.k);
        // world (u, v) = (x - 50, y - 50) since v = cy - row = 50 - (SIZE - y) = y - 50
        houghLine = { theta: th, rho: rh + 50 * Math.cos(th) + 50 * Math.sin(th) };
        drawModel(houghLine, C_TWO, 2.5, "2 4");
      }
    }

    // readout
    const lines: string[] = [];
    lines.push(`n = ${n}, w = ${wTrue.toFixed(2)}, s = ${s}: 99 % needs N = ${Number.isFinite(need) ? need : "∞"}`);
    lines.push(`iteration ${shown}/${steps()}: P(success) = ${(100 * successProbability(wTrue, s, shown)).toFixed(0)} %`);
    if (cur) lines.push(`sample: ${cur.inliers} inliers${cur.sample.every(i => isIn[i]) ? " (all-inlier sample)" : ""}`);
    if (best) lines.push(`best: ${bestN} inliers (iteration ${bestIt})`);
    if (done && res?.refit) lines.push(`refit (blue): ${res.refitInliers.length} inliers, ${res.refitInliers.filter(i => isIn[i]).length} true; ${describe(res.refit)}`);
    if (ls) lines.push(`least squares (orange): ${describe(ls)}`);
    if (houghLine) lines.push(`Hough peak (purple): ${describe(houghLine)}`);
    if (done && seqRes.length > 1) lines.push(`second model (purple): ${seqRes[1].refitInliers.length} inliers`);
    ro.textContent = lines.join("\n");
  }

  const describe = (m: Line | Circle): string => ("theta" in m ? `θ = ${(m.theta * 180 / Math.PI).toFixed(1)}°, ρ = ${m.rho.toFixed(1)}` : `centre (${m.a.toFixed(1)}, ${m.b.toFixed(1)}), r = ${m.r.toFixed(1)}`);

  void svgEl;
  sync();
  newPoints(p0?.run ?? false);
}

autoMount(NAME, mount);
