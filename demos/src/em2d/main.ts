// EM for a Gaussian mixture on 2D points: the main demo of the Gaussian-mixture day. Points
// come from a dataset menu (seeded) and can be added, dragged or removed. K = 1..6 components,
// covariance type spherical / diagonal / full / tied, start from random points, k-means,
// k-means++ seeds or by hand (drag the centres). Buttons: E-step, M-step, step (both), run,
// reset (back to the start), new start. Points are coloured by mixing the component colours
// with their responsibilities; the ellipses are 1σ (solid) and 2σ (dashed). Background: the
// mixture density or the most probable component (Bayes regions). Right: log p(D) per
// iteration (a red ring if it ever drops), the variance floor ε, "20 restarts" (histogram of
// the final log p), and BIC / AIC for K = 1..6. "k-means" switches to hard assignments by the
// nearest centre (Lloyd's algorithm), for comparison.
//
// <div data-demo="em2d" data-preset="separated|elongated|unequal|badstart|collapse|toomany|diagok|diagno|overlap"></div>
import { autoMount, button, htmlEl, presets, slider, svgEl, draggable, type Slider } from "../lib/ui";
import { pointPlayground, type Playground } from "../lib/playground";
import { DATASETS, anisotropic, blobs, withOutliers, type LPoint } from "../lib/datasets2d";
import { Plot, seriesColor, ellipsePoints } from "../lib/plot";
import { QUALITATIVE, hexToRgb } from "../lib/colour";
import { rng } from "../lib/random";
import {
  type Mixture, type CovType, type GmmInit, eStep, mStep, initMixture, fromCentres, fitGMM, nParams, bic, aic,
  cloneMixture, prepare, logJoint, logSumExp, cov2,
} from "../lib/gmm";
import { assignStep, updateStep, kmeans, initCentres } from "../lib/kmeans";

const NAME = "em2d";
const STYLE = `
[data-demo="${NAME}"] { gap: 1em; flex-wrap: nowrap; }
[data-demo="${NAME}"] .e2-left { flex: 0 1 470px; min-width: 260px; }
[data-demo="${NAME}"] .e2-right { flex: 1 1 440px; min-width: 260px; max-width: 580px; display: flex; flex-direction: column; gap: .35em; }
[data-demo="${NAME}"] .e2-plots { display: flex; gap: .4em; }
[data-demo="${NAME}"] .e2-plots > div { flex: 1 1 33%; min-width: 0; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre-wrap; }
[data-demo="${NAME}"] .e2-lab { font-size: 14px; color: var(--qb-muted); }
.reveal [data-demo="${NAME}"] { font-size: 16px; }
.reveal [data-demo="${NAME}"] .e2-left { flex: 0 0 490px; }
.reveal [data-demo="${NAME}"] .e2-right { flex: 1 1 620px; max-width: 650px; gap: .2em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 13px; padding: .2em .45em; }
.reveal [data-demo="${NAME}"] .qb-slider { font-size: 15px; }
@media (max-width: 760px) { [data-demo="${NAME}"] { flex-wrap: wrap; } }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const COLS = QUALITATIVE.slice(0, 6);
const RGB = COLS.map(h => hexToRgb(h).map(v => 255 * v));
type Init = GmmInit | "hand";
type Bg = "density" | "regions" | "none";
type Algo = "em" | "kmeans";

interface Preset {
  id: string; label: string; points: () => LPoint[]; k: number; cov: CovType; init: Init; seed?: number;
  centres?: number[]; floor?: number; bg?: Bg; algo?: Algo; bicNow?: boolean;
}
const ds = (id: string, seed = 1) => () => DATASETS.find(d => d.id === id)!.make({ seed }).points;
const PRESETS: Preset[] = [
  { id: "separated", label: "well separated", points: ds("blobs3"), k: 3, cov: "full", init: "kmeans" },
  { id: "elongated", label: "elongated: k-means fails, full GMM works", points: ds("aniso", 2), k: 2, cov: "full", init: "kmeans++", algo: "kmeans" },
  { id: "unequal", label: "one big, one small: weights matter", points: ds("unequal"), k: 2, cov: "full", init: "kmeans" },
  { id: "badstart", label: "bad start: local maximum", points: ds("blobs3"), k: 3, cov: "full", init: "hand", centres: [0.2, 0.25, 0.3, 0.35, 0.62, 0.52] },
  { id: "collapse", label: "collapse (floor off)", points: () => withOutliers(blobs([[0.3, 0.4], [0.68, 0.6]], { sd: 0.07, seed: 3 }), 1, { at: [[0.88, 0.15]] }).points, k: 3, cov: "full", init: "hand", centres: [0.3, 0.4, 0.68, 0.6, 0.86, 0.17], floor: 0 },
  { id: "toomany", label: "K too large: BIC says no", points: ds("blobs"), k: 5, cov: "full", init: "kmeans", bicNow: true },
  { id: "diagok", label: "diagonal is enough", points: () => anisotropic([{ centre: [0.3, 0.5], sd: [0.05, 0.18], angle: 0 }, { centre: [0.7, 0.5], sd: [0.16, 0.05], angle: 0 }], { seed: 3, n: 200 }).points, k: 2, cov: "diagonal", init: "kmeans", bicNow: true },
  { id: "diagno", label: "diagonal is not enough", points: ds("aniso", 2), k: 2, cov: "diagonal", init: "kmeans", bicNow: true },
  { id: "overlap", label: "heavy overlap: soft labels near 0.5", points: ds("overlap"), k: 2, cov: "full", init: "kmeans" },
];

const FLOORS = [0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2];
const f2 = (v: number) => v.toFixed(2), f3 = (v: number) => v.toFixed(3), f1 = (v: number) => v.toFixed(1);

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = {
    k: p0.k, cov: p0.cov as CovType, init: p0.init as Init, seed: p0.seed ?? 1, floor: p0.floor ?? 1e-4, bg: (p0.bg ?? "regions") as Bg, algo: (p0.algo ?? "em") as Algo,
    m: null as Mixture | null, m0: null as Mixture | null, resp: null as Float64Array | null, next: "E" as "E" | "M",
    trace: [] as number[], iter: 0, done: false, restarts: null as number[] | null, ic: null as { bic: number[]; aic: number[] } | null,
  };
  let timer = 0;

  // ---- layout ----
  const left = htmlEl("div", "e2-left", "", host);
  const right = htmlEl("div", "e2-right", "", host);
  const pg: Playground = pointPlayground(left, {
    data: { points: p0.points().map(p => ({ ...p, c: 0 })), k: 1 }, k: 1, colors: ["#888"], controls: false, size: 480, r: 7,
    ariaLabel: "2D points and a fitted Gaussian mixture", onChange: () => pointsChanged(),
  });
  const ptsG = pg.overlay.nextElementSibling as SVGGElement;
  const gEll = svgEl("g", {}, pg.overlay);
  const gHandles = svgEl("g", {}, pg.top);

  const plots = htmlEl("div", "e2-plots", "", right);
  const lpDiv = htmlEl("div", "", "", plots), rsDiv = htmlEl("div", "", "", plots), icDiv = htmlEl("div", "", "", plots);
  const lp = new Plot(lpDiv, { width: 230, height: 180, x: [0, 5], y: [0, 1], xlabel: "iteration", ylabel: "log p(D)", font: 16, margin: { l: 66 } });
  const lpLine = lp.line([], [], { color: seriesColor(0), width: 2.5 });
  const lpDrop = lp.scatter([], [], { color: "none", r: 8, stroke: "#d62728" });
  const rs = new Plot(rsDiv, { width: 230, height: 180, x: [0, 1], y: [0, 10], xlabel: "final log p (20 starts)", font: 16, margin: { l: 40 } });
  const rsBars = rs.histogram([], [], { color: seriesColor(3) });
  const ic = new Plot(icDiv, { width: 230, height: 180, x: [0.4, 6.6], y: [0, 1], xlabel: "K", xticks: [1, 2, 3, 4, 5, 6], font: 16, margin: { l: 56 } });
  const icStyleB = { color: seriesColor(0), barWidth: 0.38, base: 0 }, icStyleA = { color: seriesColor(3), barWidth: 0.38, base: 0 };
  const icB = ic.bars([], [], icStyleB);
  const icA = ic.bars([], [], icStyleA);
  ic.legend([{ label: "BIC", color: seriesColor(0), kind: "box" }, { label: "AIC", color: seriesColor(3), kind: "box" }], "tr");

  const ro = htmlEl("div", "qb-readout", "", right);
  const r1 = htmlEl("div", "qb-presets", "", right);
  const eBtn = button(r1, "E-step", () => { stop(); doE(); draw(); });
  const mBtn = button(r1, "M-step", () => { stop(); doM(); draw(); });
  button(r1, "step", () => { stop(); step(); draw(); });
  const runBtn = button(r1, "run", () => { if (timer) stop(); else run(); });
  button(r1, "reset", () => { stop(); restart(false); });
  button(r1, "new start", () => { stop(); st.seed++; restart(true); });
  button(r1, "20 restarts", () => { stop(); restarts(); draw(); });
  button(r1, "BIC / AIC", () => { stop(); criteria(); draw(); });
  const r2 = htmlEl("div", "qb-presets", "", right);
  htmlEl("span", "e2-lab", "Σ:", r2);
  const covBtns = (["spherical", "diagonal", "full", "tied"] as CovType[]).map(c => button(r2, c, () => { st.cov = c; st.ic = null; restart(true); }));
  htmlEl("span", "e2-lab", " start:", r2);
  const initBtns = (["random", "kmeans", "kmeans++", "hand"] as Init[]).map(m => button(r2, m === "hand" ? "by hand" : m === "kmeans" ? "k-means" : m === "kmeans++" ? "k-means++" : "random", () => { st.init = m; restart(true); }));
  const r3 = htmlEl("div", "qb-presets", "", right);
  htmlEl("span", "e2-lab", "fit:", r3);
  const algoBtns = (["em", "kmeans"] as Algo[]).map(a => button(r3, a === "em" ? "EM (soft)" : "k-means (hard)", () => { st.algo = a; restart(false); }));
  htmlEl("span", "e2-lab", " background:", r3);
  const bgBtns = (["regions", "density", "none"] as Bg[]).map(b => button(r3, b === "regions" ? "regions" : b, () => { st.bg = b; draw(); }));
  const remBtn = button(r3, "remove points", () => { pg.tool = pg.tool === "remove" ? "add" : "remove"; sync(); });
  const kS: Slider = slider(right, "K", 1, 6, 1, st.k, v => { st.k = v; restart(true); });
  const fS: Slider = slider(right, "floor ε", 0, FLOORS.length - 1, 1, FLOORS.indexOf(st.floor), v => { st.floor = FLOORS[v]; restart(false); }, v => (FLOORS[v] ? FLOORS[v].toExponential(0) : "0"));
  htmlEl("div", "qb-note", "Click to add points, drag to move them. Drag a centre to start by hand. Colours: responsibilities.", right);
  const presetBtns = presets(right, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  button(right, "Reset demo", () => load(p0)).style.alignSelf = "flex-start";

  // ---- data ----
  function data(): { X: Float64Array; n: number } {
    const P = pg.points, X = new Float64Array(2 * P.length);
    P.forEach((p, i) => { X[2 * i] = p.x; X[2 * i + 1] = p.y; });
    return { X, n: P.length };
  }
  let handCentres: number[] = [];

  function restart(newStart: boolean): void {
    stop();
    const { X, n } = data();
    st.resp = null; st.next = "E"; st.trace = []; st.iter = 0; st.done = false; st.restarts = newStart ? null : st.restarts;
    if (n < 2) { st.m = null; draw(); return; }
    const o = { covType: st.cov, floor: st.floor };
    if (newStart || !st.m0 || st.m0.k !== st.k) {
      if (st.init === "hand") {
        if (handCentres.length !== 2 * st.k) handCentres = Array.from({ length: 2 * st.k }, (_, i) => (i % 2 ? 0.25 + 0.5 * (Math.floor(i / 2) + 0.5) / st.k : 0.2 + 0.6 * (Math.floor(i / 2) + 0.5) / st.k));
        st.m0 = fromCentres(X, n, 2, st.k, handCentres, o);
      } else st.m0 = initMixture(X, n, 2, st.k, st.init, rng(st.seed), o);
    }
    st.m = cloneMixture(st.m0);
    buildHandles();
    draw();
  }

  function load(p: Preset): void {
    stop();
    st.k = p.k; st.cov = p.cov; st.init = p.init; st.seed = p.seed ?? 1; st.floor = p.floor ?? 1e-4; st.bg = p.bg ?? "regions"; st.algo = p.algo ?? "em";
    st.restarts = null; st.ic = null; st.m0 = null;
    handCentres = p.centres ? p.centres.slice() : [];
    kS.set(st.k); fS.set(FLOORS.indexOf(st.floor));
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    loading = true;
    pg.setPoints({ points: p.points().map(q => ({ ...q, c: 0 })), k: 1 });
    loading = false;
    restart(true);
    if (p.bicNow) criteria();
    draw();
  }
  let loading = false;
  function pointsChanged(): void {
    if (loading) return;
    st.restarts = null; st.ic = null;
    if (st.init !== "hand" && st.m) { st.resp = null; st.next = "E"; }
    draw();
  }

  // ---- algorithm ----
  const mopts = () => ({ covType: st.cov, floor: st.floor });
  function kmResp(m: Mixture, X: Float64Array, n: number): { resp: Float64Array; J: number } {
    const a = assignStep(X, n, 2, m.means, m.k), resp = new Float64Array(n * m.k);
    for (let i = 0; i < n; i++) resp[i * m.k + a.assign[i]] = 1;
    return { resp, J: a.J };
  }
  function score(m: Mixture, X: Float64Array, n: number): number {
    return st.algo === "kmeans" ? kmResp(m, X, n).J : eStep(m, X, n).logLik;
  }
  function doE(): void {
    if (!st.m) return;
    const { X, n } = data();
    if (st.algo === "kmeans") { const r = kmResp(st.m, X, n); st.done = !!st.resp && r.resp.every((v, i) => v === st.resp![i]) && st.next === "E"; st.resp = r.resp; }
    else st.resp = eStep(st.m, X, n).resp;
    if (!st.trace.length) st.trace.push(score(st.m, X, n));
    st.next = "M";
  }
  function doM(): void {
    if (!st.m) return;
    if (!st.resp || st.next === "E") doE();
    const { X, n } = data();
    if (st.algo === "kmeans") {
      const a = Int32Array.from({ length: n }, (_, i) => { for (let j = 0; j < st.k; j++) if (st.resp![i * st.k + j] === 1) return j; return 0; });
      const u = updateStep(X, n, 2, a, st.k, undefined, st.m.means);
      st.m = { ...st.m, means: u.centres, weights: new Float64Array(st.k).fill(1 / st.k) };
    } else st.m = mStep(X, n, 2, st.resp!, st.k, mopts(), undefined, st.m);
    st.iter++;
    const L = score(st.m, X, n);
    if (st.algo === "em" && st.trace.length && Math.abs(L - st.trace[st.trace.length - 1]) < 1e-7) st.done = true;
    st.trace.push(L);
    st.next = "E";
  }
  function step(): void { doE(); doM(); doE(); }
  function run(): void {
    runBtn.textContent = "pause";
    timer = window.setInterval(() => { step(); draw(); if (st.done || st.iter >= 300) stop(); }, 110);
  }
  function stop(): void { if (timer) { clearInterval(timer); timer = 0; } runBtn.textContent = "run"; }

  function restarts(): void {
    const { X, n } = data();
    if (n < 2) return;
    const how: GmmInit = st.init === "hand" ? "random" : st.init, r = rng(500 + st.seed), out: number[] = [];
    for (let t = 0; t < 20; t++) {
      if (st.algo === "kmeans") out.push(kmeans(X, n, 2, st.k, { init: initCentres(X, n, 2, st.k, how === "kmeans" ? "kmeans++" : how === "kmeans++" ? "kmeans++" : "random", r) }).J);
      else out.push(fitGMM(X, n, 2, st.k, { ...mopts(), init: how, seed: r, maxIter: 300 }).logLik);
    }
    st.restarts = out;
  }
  function criteria(): void {
    const { X, n } = data();
    if (n < 2) return;
    const B: number[] = [], A: number[] = [];
    for (let K = 1; K <= 6; K++) {
      let best = -Infinity;
      for (let t = 0; t < 5; t++) best = Math.max(best, fitGMM(X, n, 2, K, { ...mopts(), init: "kmeans", seed: rng(900 + 7 * K + t), maxIter: 300 }).logLik);
      const p = nParams(K, 2, st.cov);
      B.push(bic(best, p, n)); A.push(aic(best, p));
    }
    st.ic = { bic: B, aic: A };
  }

  // ---- drawing ----
  function buildHandles(): void {
    gHandles.replaceChildren();
    if (!st.m) return;
    for (let j = 0; j < st.k; j++) {
      const g = svgEl("g", {}, gHandles);
      const c = svgEl("circle", { r: 11, "stroke-width": 4, stroke: COLS[j] }, g);
      c.style.setProperty("fill", "var(--qb-bg, #fff)");
      svgEl("path", { d: "M-6,0H6M0,-6V6", "stroke-width": 3, stroke: COLS[j] }, g);
      g.addEventListener("pointerdown", ev => ev.stopPropagation());
      draggable(g as unknown as SVGElement, ev => pg.toData(ev), (x, y) => {
        stop();
        const { X, n } = data();
        if (!st.m || n < 2) return;
        handCentres = Array.from(st.m.means);
        handCentres[2 * j] = Math.min(1, Math.max(0, x)); handCentres[2 * j + 1] = Math.min(1, Math.max(0, y));
        st.init = "hand";
        st.m0 = fromCentres(X, n, 2, st.k, handCentres, mopts());
        st.m = cloneMixture(st.m0);
        st.resp = null; st.next = "E"; st.trace = []; st.iter = 0; st.done = false;
        draw();
      });
    }
  }

  function draw(): void {
    const m = st.m, { X, n } = data();
    pg.redraw();
    // background
    if (m && st.bg !== "none") {
      const prep = prepare(m);
      let mx = 0;
      if (st.bg === "density") for (let i = 0; i < 32; i++) for (let j = 0; j < 32; j++) mx = Math.max(mx, Math.exp(logSumExp(logJoint(m, [(i + 0.5) / 32, (j + 0.5) / 32], 1, prep))));
      pg.setBackgroundColour((x, y) => {
        if (st.algo === "kmeans" || st.bg === "regions") {
          let b = 0;
          if (st.algo === "kmeans") { let bd = Infinity; for (let j = 0; j < m.k; j++) { const d = (x - m.means[2 * j]) ** 2 + (y - m.means[2 * j + 1]) ** 2; if (d < bd) { bd = d; b = j; } } }
          else { const lj = logJoint(m, [x, y], 1, prep); for (let j = 1; j < m.k; j++) if (lj[j] > lj[b]) b = j; }
          return [RGB[b][0], RGB[b][1], RGB[b][2], 0.16];
        }
        const v = Math.pow(Math.exp(logSumExp(logJoint(m, [x, y], 1, prep))) / (mx || 1), 0.6);
        return [30, 110, 220, Math.min(0.6, 0.6 * v)];
      });
    } else pg.setBackgroundColour(null);
    // point colours from responsibilities (current model if no E-step yet)
    const resp = m ? (st.resp && st.resp.length === n * m.k ? st.resp : st.algo === "kmeans" ? kmResp(m, X, n).resp : eStep(m, X, n).resp) : null;
    const circles = ptsG.children;
    for (let i = 0; i < circles.length; i++) {
      let col = "#888";
      if (resp && m) {
        let r = 0, g = 0, b = 0;
        for (let j = 0; j < m.k; j++) { const w = resp[i * m.k + j]; r += w * RGB[j][0]; g += w * RGB[j][1]; b += w * RGB[j][2]; }
        col = `rgb(${r.toFixed(0)},${g.toFixed(0)},${b.toFixed(0)})`;
      }
      (circles[i] as SVGElement).style.setProperty("fill", col);
    }
    // ellipses
    gEll.replaceChildren();
    if (m && st.algo === "em") {
      for (let j = 0; j < m.k; j++) {
        if (!(m.weights[j] > 0)) continue;
        for (const nsd of [1, 2]) {
          const pts = ellipsePoints([m.means[2 * j], m.means[2 * j + 1]], cov2(m, j), nsd, 72);
          svgEl("path", { d: "M" + pts.map(([x, y]) => `${pg.sx(x).toFixed(1)},${pg.sy(y).toFixed(1)}`).join("L") + "Z", fill: "none", stroke: COLS[j], "stroke-width": nsd === 1 ? 3 : 1.8, "stroke-dasharray": nsd === 1 ? "" : "6 5" }, gEll);
        }
      }
    }
    if (m && gHandles.children.length !== m.k) buildHandles();
    Array.from(gHandles.children).forEach((g, j) => { if (m) g.setAttribute("transform", `translate(${pg.sx(m.means[2 * j]).toFixed(1)},${pg.sy(m.means[2 * j + 1]).toFixed(1)})`); });
    // log p plot (or J for k-means)
    const T = st.trace, isKm = st.algo === "kmeans";
    lp.setLabels({ ylabel: isKm ? "J" : "log p(D)" });
    if (T.length) {
      const lo = Math.min(...T), hi = Math.max(...T), pad = (hi - lo) * 0.1 || 1;
      lp.setDomain([0, Math.max(5, T.length - 1)], [lo - pad, hi + pad]);
    }
    lpLine.set(T.map((_, i) => i), T);
    const dropsX: number[] = [], dropsY: number[] = [];
    if (!isKm) T.forEach((v, i) => { if (i > 0 && v < T[i - 1] - 1e-9 * Math.abs(v)) { dropsX.push(i); dropsY.push(v); } });
    lpDrop.set(dropsX, dropsY);
    // restarts histogram
    if (st.restarts) {
      const R = st.restarts, lo = Math.min(...R), hi = Math.max(...R), span = hi - lo || Math.max(1, Math.abs(hi) * 0.02);
      const nb = 12, e = Array.from({ length: nb + 1 }, (_, i) => lo - span * 0.05 + i * span * 1.1 / nb), c = new Array(nb).fill(0);
      R.forEach(v => { c[Math.min(nb - 1, Math.floor((v - e[0]) / (e[nb] - e[0]) * nb))]++; });
      rs.setDomain([e[0], e[nb]], [0, Math.max(...c) * 1.15]);
      rs.setLabels({ xlabel: isKm ? "final J (20 starts)" : "final log p (20 starts)" });
      rsBars.set(e, c);
    } else rsBars.set([], []);
    // BIC / AIC
    if (st.ic) {
      const all = [...st.ic.bic, ...st.ic.aic], lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.15 || 1;
      ic.setDomain(undefined, [lo - pad, hi + pad]);
      icStyleB.base = lo - pad; icStyleA.base = lo - pad;
      icB.set([1, 2, 3, 4, 5, 6].map(x => x - 0.2), st.ic.bic); icA.set([1, 2, 3, 4, 5, 6].map(x => x + 0.2), st.ic.aic);
    } else { icB.set([], []); icA.set([], []); }
    // readout
    let s = "";
    if (!m) s = "Add at least two points.";
    else {
      const L = eStep(m, X, n).logLik, p = nParams(m.k, 2, st.cov);
      const minVar = Math.min(...Array.from({ length: m.k }, (_, j) => { const c = cov2(m, j); const t = (c[0][0] + c[1][1]) / 2, d = Math.hypot((c[0][0] - c[1][1]) / 2, c[0][1]); return t - d; }));
      s += isKm ? `k-means: J = ${f3(kmResp(m, X, n).J)}   iteration ${st.iter}` + (st.done ? "   converged" : "")
        : `log p(D) = ${f1(L)}   iteration ${st.iter}` + (st.done ? "   converged" : "") + `   next: ${st.next}-step`;
      if (!isKm) s += `\nparameters p = ${p}   BIC = ${f1(bic(L, p, n))}   AIC = ${f1(aic(L, p))}   n = ${n}`;
      if (!isKm && minVar < 1e-6) s += "\na component has collapsed onto a point: its variance is ≈ 0 and log p shoots up";
      s += "\n" + Array.from({ length: m.k }, (_, j) => `${j + 1}: π ${f2(m.weights[j])} μ (${f2(m.means[2 * j])}, ${f2(m.means[2 * j + 1])})`).join("  ");
      if (st.restarts) { const R = st.restarts; s += isKm ? `\n20 starts: best J ${f3(Math.min(...R))}, worst ${f3(Math.max(...R))}` : `\n20 starts: best log p ${f1(Math.max(...R))}, worst ${f1(Math.min(...R))}`; }
      if (st.ic) { const b = st.ic.bic, a = st.ic.aic; s += `\nBIC is smallest at K = ${b.indexOf(Math.min(...b)) + 1}, AIC at K = ${a.indexOf(Math.min(...a)) + 1} (${st.cov})`; }
    }
    ro.textContent = s;
    sync();
  }

  function sync(): void {
    covBtns.forEach((b, i) => b.classList.toggle("active", ["spherical", "diagonal", "full", "tied"][i] === st.cov));
    initBtns.forEach((b, i) => b.classList.toggle("active", ["random", "kmeans", "kmeans++", "hand"][i] === st.init));
    algoBtns.forEach((b, i) => b.classList.toggle("active", ["em", "kmeans"][i] === st.algo));
    bgBtns.forEach((b, i) => b.classList.toggle("active", ["regions", "density", "none"][i] === st.bg));
    remBtn.classList.toggle("active", pg.tool === "remove");
    eBtn.classList.toggle("active", st.next === "E" && !!st.m);
    mBtn.classList.toggle("active", st.next === "M");
  }

  load(p0);
  const steps = Number(host.dataset.steps) || 0;
  for (let i = 0; i < steps; i++) step();
  if (steps) draw();
}

autoMount(NAME, mount);
