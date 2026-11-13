// Interpolating, approximating and smoothing splines (splines day, unit 3). Noisy samples
// (fixed seed) of a known curve, or a profile across the micrograph. Modes: interpolate
// (natural cubic spline through every point), approximate (least squares with m clamped cubic
// B-spline coefficients; knots shown), smooth (cubic smoothing spline: Σ(y − f)² + λ∫f''²).
// Below: the leave-one-out error and the effective number of parameters (tr H) against λ (or
// m), with markers at the current value and at the minimum. Data points can be dragged up
// and down; "outlier" moves one point far off.
// <div data-demo="splinefit" data-preset="interp|line|cvmin|few|many|outlier|step" data-fn="bump|kink|step|cell"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { rng } from "../lib/random";
import { naturalCubic, smoothingSpline, regressionSpline, LinearFit } from "../lib/spline";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { gaussianBlur, Grey } from "../lib/imageops";

const NAME = "splinefit";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; }
.qb-demo[data-demo="${NAME}"] .sf-left { flex: 1 1 560px; max-width: 660px; display: flex; flex-direction: column; gap: .1em; }
.qb-demo[data-demo="${NAME}"] .sf-left .qb-plot { flex: none; max-width: none; }
.qb-demo[data-demo="${NAME}"] .sf-small { display: flex; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .sf-small .qb-plot { flex: 1 1 0; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 380px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 18px; white-space: pre; }
.qb-demo[data-demo="${NAME}"] .sf-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .sf-credit { font-size: 12px; color: var(--qb-muted); }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .sf-left { flex: 0 0 640px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 450px; gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .04em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { padding: .2em .45em; font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .sf-lab, .reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

type Fn = "bump" | "kink" | "step" | "cell";
type Mode = "interp" | "approx" | "smooth";
interface State { fn: Fn; mode: Mode; noise: number; m: number; logLam: number; outlier: boolean; truth: boolean; resid: boolean; edits: Map<number, number> }
const N = 30, X0 = 0, X1 = 10, OUT_I = 20;
const LAM_MIN = -4, LAM_MAX = 4;
const DEFAULT: Omit<State, "edits"> = { fn: "kink", mode: "smooth", noise: 0.25, m: 10, logLam: -1, outlier: false, truth: true, resid: true };
const PRESETS: { id: string; label: string; s: Partial<State>; cvmin?: boolean }[] = [
  { id: "interp", label: "λ = 0 interpolates the noise", s: { mode: "interp" } },
  { id: "line", label: "λ huge: a straight line", s: { mode: "smooth", logLam: 4 } },
  { id: "cvmin", label: "the CV minimum", s: { mode: "smooth" }, cvmin: true },
  { id: "few", label: "too few control points", s: { mode: "approx", m: 5 } },
  { id: "many", label: "too many control points", s: { mode: "approx", m: 26 } },
  { id: "outlier", label: "an outlier", s: { mode: "smooth", outlier: true }, cvmin: true },
  { id: "step", label: "a step: every smoother rounds it", s: { fn: "step", mode: "smooth" }, cvmin: true },
];

const truths: Record<Exclude<Fn, "cell">, (x: number) => number> = {
  bump: x => 2 * Math.exp(-((x - 5) ** 2) / 2),
  kink: x => Math.sin(0.8 * x) + 0.4 * Math.abs(x - 6),
  step: x => (x < 5 ? 0 : 2),
};

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[2];
  const fn0 = (host.dataset.fn as Fn) || undefined;
  const st: State = { ...DEFAULT, edits: new Map() };
  let cellTruth: ((x: number) => number) | null = null;

  const left = htmlEl("div", "sf-left", "", host);
  const f1 = plotFigure(left, "");
  const plot = new Plot(f1, { width: 640, height: 330, x: [-0.2, 10.2], y: [-1.2, 3.4], ylabel: "y", margin: { b: 30 } });
  const small = htmlEl("div", "sf-small", "", left);
  const fcv = plotFigure(small, "");
  const cvp = new Plot(fcv, { width: 320, height: 190, x: [1e-4, 1e4], y: [0, 1], xlog: true, xlabel: "λ", ylabel: "LOO error", font: 18, margin: { l: 70 } });
  const fdf = plotFigure(small, "");
  const dfp = new Plot(fdf, { width: 320, height: 190, x: [1e-4, 1e4], y: [0, 31], xlog: true, xlabel: "λ", ylabel: "df = tr H", font: 18, margin: { l: 70 } });
  const credit = htmlEl("div", "sf-credit", CELL_CREDIT, left);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "sf-lab", "presets", panel);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "sf-lab", "data · mode", panel);
  const fnBtns = presets(panel, (["bump", "kink", "step", "cell"] as Fn[]).map(f => ({ label: f === "kink" ? "sine with a kink" : f === "cell" ? "cell profile" : f, apply: () => { st.fn = f; st.edits.clear(); draw(); } })));
  const modeRow = htmlEl("div", "qb-presets", "", panel);
  const modeBtns = (["interp", "approx", "smooth"] as Mode[]).map(m => button(modeRow, m === "interp" ? "interpolate" : m === "approx" ? "approximate (m)" : "smooth (λ)", () => { st.mode = m; draw(); }));
  const noiseS = slider(panel, "noise", 0, 0.6, 0.05, st.noise, v => { st.noise = v; draw(); });
  const mS = slider(panel, "control points m", 4, N, 1, st.m, v => { st.m = v; if (st.mode !== "approx") st.mode = "approx"; draw(); });
  const lamS = slider(panel, "log₁₀ λ", LAM_MIN, LAM_MAX, 0.05, st.logLam, v => { st.logLam = v; if (st.mode !== "smooth") st.mode = "smooth"; draw(); }, v => v.toFixed(2));
  const row = htmlEl("div", "qb-presets", "", panel);
  const outBtn = button(row, "outlier", () => { st.outlier = !st.outlier; draw(); });
  button(row, "go to the CV minimum", () => { gotoMin(); draw(); });
  button(row, "reset", () => load(p0));
  const rowc = htmlEl("div", "qb-presets", "", panel);
  const cks = [checkbox(rowc, "true curve", st.truth, v => { st.truth = v; draw(); }), checkbox(rowc, "residuals", st.resid, v => { st.resid = v; draw(); })];
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Drag a data point up or down.", panel);

  const noiseVals = (() => { const r = rng(3); return Array.from({ length: N }, () => r.norm()); })();
  const xs = Array.from({ length: N }, (_, i) => X0 + (X1 - X0) * i / (N - 1));

  const truth = (x: number) => (st.fn === "cell" ? (cellTruth ? cellTruth(x) : 0) : truths[st.fn](x));
  function data(): number[] {
    const y = xs.map((x, i) => truth(x) + st.noise * noiseVals[i]);
    if (st.outlier) y[OUT_I] += 2.2;
    for (const [i, v] of st.edits) y[i] = v;
    return y;
  }
  const lamGrid = Array.from({ length: 49 }, (_, k) => Math.pow(10, LAM_MIN + (LAM_MAX - LAM_MIN) * k / 48));
  const mGrid = Array.from({ length: N - 3 }, (_, k) => k + 4);

  function fit(y: number[]): LinearFit & { knots?: number[] } {
    if (st.mode === "approx") return regressionSpline(xs, y, st.m);
    if (st.mode === "smooth") return smoothingSpline(xs, y, Math.pow(10, st.logLam));
    const s = naturalCubic(xs, y), f = Float64Array.from(y), h = new Float64Array(N).fill(1);
    return { fitted: f, hdiag: h, df: N, loo: NaN, gcv: NaN, rss: 0, at: s.at };
  }
  function gotoMin(): void {
    const y = data();
    if (st.mode === "approx") {
      let best = 4, bv = Infinity;
      for (const m of mGrid) { const v = regressionSpline(xs, y, m).loo; if (v < bv) { bv = v; best = m; } }
      st.m = best;
    } else {
      st.mode = "smooth";
      let best = 0, bv = Infinity;
      for (let k = 0; k <= 160; k++) { const l = LAM_MIN + (LAM_MAX - LAM_MIN) * k / 160, v = smoothingSpline(xs, y, Math.pow(10, l)).loo; if (v < bv) { bv = v; best = l; } }
      st.logLam = Math.round(best * 20) / 20;
    }
  }

  function load(p: typeof PRESETS[number]): void {
    Object.assign(st, DEFAULT, p.s);
    if (fn0 && !p.s.fn) st.fn = fn0;
    st.edits = new Map();
    cks[0].checked = st.truth; cks[1].checked = st.resid;
    if (p.cvmin) gotoMin();
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    draw();
  }

  const mk = {
    truth: plot.fn(truth, { color: "var(--qb-muted, #888)", width: 2.5, dash: "7 5" }, 400),
    resid: plot.mark((g, X: number[], Y: number[], F: number[]) => {
      for (let i = 0; i < X.length; i++) {
        const l = document.createElementNS("http://www.w3.org/2000/svg", "line");
        l.setAttribute("x1", String(plot.x(X[i]))); l.setAttribute("x2", String(plot.x(X[i])));
        l.setAttribute("y1", String(plot.y(Y[i]))); l.setAttribute("y2", String(plot.y(F[i])));
        l.style.cssText = "stroke:var(--qb-c2,#d62728);stroke-width:2;opacity:.7";
        g.appendChild(l);
      }
    }, [[], [], []] as [number[], number[], number[]]),
    fit: plot.line([], [], { color: "var(--qb-c1, #1f77b4)", width: 4 }),
    knots: plot.scatter([], [], { color: "var(--qb-c4, #ff7f0e)", r: 6 }),
    dots: plot.scatter([], [], { color: "var(--qb-fg, #222)", r: 7 }),
  };

  function draw(): void {
    if (st.fn === "cell" && !cellTruth) {
      loadCell().then((img: Grey) => {
        const b = gaussianBlur(img, 2), row = 120, w = img.width;
        cellTruth = (x: number) => { const u = 30 + x / 10 * 200, i = Math.min(w - 2, Math.floor(u)), f = u - i; return (b.data[row * w + i] * (1 - f) + b.data[row * w + i + 1] * f) / 80 - 0.5; };
        draw();
      });
    }
    credit.style.visibility = st.fn === "cell" ? "visible" : "hidden";
    fnBtns.forEach((b, i) => b.classList.toggle("active", ["bump", "kink", "step", "cell"][i] === st.fn));
    modeBtns.forEach((b, i) => b.classList.toggle("active", ["interp", "approx", "smooth"][i] === st.mode));
    outBtn.classList.toggle("active", st.outlier);
    noiseS.set(st.noise); mS.set(st.m); lamS.set(st.logLam);
    const y = data(), F = fit(y);
    const gx: number[] = [], gy: number[] = [];
    for (let k = 0; k <= 400; k++) { const x = X0 + (X1 - X0) * k / 400; gx.push(x); gy.push(F.at(x)); }
    mk.truth.set(st.truth ? truth : () => NaN);
    mk.fit.set(gx, gy);
    mk.resid.set(st.resid ? xs : [], y, Array.from(F.fitted));
    if (st.mode === "approx" && F.knots) { const ks = [...new Set(F.knots)]; mk.knots.set(ks, ks.map(() => -1.05)); } else mk.knots.set([], []);
    mk.dots.set(xs, y);
    attachDrag();
    // CV and df curves
    cvp.clear(); dfp.clear();
    let curX: number[], loo: number[], df: number[], cur: number;
    if (st.mode === "approx") {
      curX = mGrid; loo = mGrid.map(m => regressionSpline(xs, y, m).loo); df = mGrid.slice(); cur = st.m;
      cvp.setDomain([4, N], undefined); dfp.setDomain([4, N], undefined);
      for (const q of [cvp, dfp]) { q.opts.xlog = false; q.opts.xticks = [5, 10, 15, 20, 25, 30]; q.opts.xfmt = undefined; }
      cvp.setLabels({ xlabel: "control points m" }); dfp.setLabels({ xlabel: "control points m" });
    } else {
      curX = lamGrid;
      const fits = lamGrid.map(l => smoothingSpline(xs, y, l));
      loo = fits.map(f => f.loo); df = fits.map(f => f.df); cur = st.mode === "interp" ? 1e-4 : Math.pow(10, st.logLam);
      for (const q of [cvp, dfp]) { q.opts.xlog = true; q.opts.xticks = [1e-4, 1e-2, 1, 1e2, 1e4]; q.opts.xfmt = LOGFMT; }
      cvp.setDomain([1e-4, 1e4], undefined); dfp.setDomain([1e-4, 1e4], undefined);
      cvp.setLabels({ xlabel: "λ" }); dfp.setLabels({ xlabel: "λ" });
    }
    const ymax = Math.min(Math.max(...loo) * 1.05, 4 * Math.min(...loo) + 0.5);
    cvp.setDomain(undefined, [0, Math.max(0.05, ymax)]);
    cvp.line(curX, loo.map(v => Math.min(v, 10 * ymax + 1)), { color: "var(--qb-c2, #d62728)", width: 3 });
    const kmin = loo.indexOf(Math.min(...loo));
    cvp.vline(curX[kmin], { color: "var(--qb-c3, #2ca02c)", dash: "3 3" });
    cvp.vline(cur, { color: "var(--qb-c1, #1f77b4)" });
    dfp.line(curX, df, { color: "var(--qb-c5, #9467bd)", width: 3 });
    dfp.vline(cur, { color: "var(--qb-c1, #1f77b4)" });
    // true error at the data (synthetic data only)
    let terr = 0;
    for (let i = 0; i < N; i++) terr += (F.fitted[i] - truth(xs[i])) ** 2;
    const what = st.mode === "interp" ? "interpolate: df = 30" : st.mode === "approx" ? `approximate: m = ${st.m}` : `smooth: λ = ${fmtL(Math.pow(10, st.logLam))}`;
    ro.textContent = `${what}\ndf (tr H)     ${F.df.toFixed(2)}\nLOO error     ${Number.isFinite(F.loo) ? F.loo.toFixed(4) : "—"}\nCV minimum at ${st.mode === "approx" ? "m = " + curX[kmin] : "λ ≈ " + fmtL(curX[kmin])}\nRMS to truth  ${Math.sqrt(terr / N).toFixed(3)}`;
  }

  function attachDrag(): void {
    mk.dots.g.querySelectorAll("circle").forEach((c, i) => {
      const el = c as SVGCircleElement;
      el.style.cursor = "ns-resize"; el.style.touchAction = "none";
      el.addEventListener("pointerdown", ev => {
        ev.preventDefault();
        const move = (e: PointerEvent) => {
          const [, yv] = plot.toData(e);
          st.edits.set(i, Math.max(-1.1, Math.min(3.3, yv)));
          pbtns.forEach(b => b.classList.remove("active"));
          const y = data(), F = fit(y), gx: number[] = [], gy: number[] = [];
          for (let k = 0; k <= 200; k++) { const x = X0 + (X1 - X0) * k / 200; gx.push(x); gy.push(F.at(x)); }
          mk.fit.set(gx, gy);
          mk.resid.set(st.resid ? xs : [], y, Array.from(F.fitted));
          mk.dots.g.querySelectorAll("circle").forEach((cc, k) => cc.setAttribute("cy", plot.y(y[k]).toFixed(2)));
        };
        const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); draw(); };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
      });
    });
  }

  load(p0);
}

const LOGFMT = (v: number) => { const e = Math.round(Math.log10(v)); return e === 0 ? "1" : "10" + String(e).replace("-", "⁻").replace(/\d/g, d => "⁰¹²³⁴⁵⁶⁷⁸⁹"[Number(d)]); };
const fmtL = (l: number) => (l >= 100 || l < 0.01 ? l.toExponential(1) : l.toPrecision(2));

autoMount(NAME, mount);
