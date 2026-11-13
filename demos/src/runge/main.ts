// Runge's phenomenon: one polynomial through n samples against piecewise-linear and the
// natural cubic spline (unit 3 of the splines day). Functions: the bump 1/(1 + 25x²), a step,
// and a profile across the course micrograph. Nodes equispaced or Chebyshev; samples
// draggable vertically ("one point moved": the polynomial changes everywhere). Readout: the
// largest error against the true function for each interpolant.
// <div data-demo="runge" data-preset="five|runge|cheb|step|moved|cell"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { polyInterp, equispacedNodes, chebyshevNodes, linearInterp, naturalCubic } from "../lib/spline";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { gaussianBlur, Grey } from "../lib/imageops";

const NAME = "runge";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .9em; }
.qb-demo[data-demo="${NAME}"] .qb-plot { flex: 1 1 520px; max-width: 680px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 330px; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 18px; white-space: pre; }
.qb-demo[data-demo="${NAME}"] .rg-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .rg-credit { font-size: 12px; color: var(--qb-muted); }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-plot { flex: 0 0 660px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 400px; gap: .25em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .08em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { padding: .25em .45em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

type Fn = "bump" | "step" | "cell";
interface State { fn: Fn; n: number; nodes: "equi" | "cheb"; poly: boolean; lin: boolean; spline: boolean; truth: boolean; moved: [number, number] | null }
const DEFAULT: State = { fn: "bump", n: 11, nodes: "equi", poly: true, lin: true, spline: true, truth: true, moved: null };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "five", label: "5 points are fine", s: { fn: "bump", n: 5, nodes: "equi" } },
  { id: "runge", label: "11 points: Runge", s: { fn: "bump", n: 11, nodes: "equi" } },
  { id: "cheb", label: "Chebyshev fixes the ends", s: { fn: "bump", n: 11, nodes: "cheb" } },
  { id: "step", label: "the step: everything rings", s: { fn: "step", n: 12, nodes: "equi" } },
  { id: "moved", label: "one point moved", s: { fn: "bump", n: 9, nodes: "equi", moved: [6, 0.3] } },
  { id: "cell", label: "a profile across the cell", s: { fn: "cell", n: 15, nodes: "equi" } },
];

const bump = (x: number) => 1 / (1 + 25 * x * x);
const step = (x: number) => (x < 0.1 ? 0 : 1);

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[1];
  const st: State = { ...DEFAULT, ...p0.s };
  let cellProfile: ((x: number) => number) | null = null;

  const fig = plotFigure(host, "");
  const plot = new Plot(fig, { width: 660, height: 440, x: [-1.05, 1.05], y: [-0.6, 1.6], xlabel: "x", ylabel: "f(x)", xticks: [-1, -0.5, 0, 0.5, 1] });
  const credit = htmlEl("div", "rg-credit", CREDIT_TEXT(), fig);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "rg-lab", "presets", panel);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p.s) })));
  htmlEl("div", "rg-lab", "function · nodes", panel);
  const fnBtns = presets(panel, [
    { label: "bump", apply: () => { st.fn = "bump"; st.moved = null; draw(); } },
    { label: "step", apply: () => { st.fn = "step"; st.moved = null; draw(); } },
    { label: "cell profile", apply: () => { st.fn = "cell"; st.moved = null; draw(); } },
    { label: "equispaced", apply: () => { st.nodes = "equi"; st.moved = null; draw(); } },
    { label: "Chebyshev", apply: () => { st.nodes = "cheb"; st.moved = null; draw(); } },
  ]);
  const nS = slider(panel, "points n", 3, 21, 1, st.n, v => { st.n = v; st.moved = null; draw(); });
  const cks = [
    checkbox(panel, "one polynomial (degree n − 1)", st.poly, v => { st.poly = v; draw(); }),
    checkbox(panel, "piecewise linear", st.lin, v => { st.lin = v; draw(); }),
    checkbox(panel, "natural cubic spline (unit 3)", st.spline, v => { st.spline = v; draw(); }),
    checkbox(panel, "true function", st.truth, v => { st.truth = v; draw(); }),
  ];
  button(panel, "reset", () => load(p0.s));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Drag a sample up or down.", panel);

  function CREDIT_TEXT() { return CELL_CREDIT; }

  function load(s: Partial<State>): void {
    Object.assign(st, DEFAULT, s);
    nS.set(st.n);
    [st.poly, st.lin, st.spline, st.truth].forEach((v, i) => (cks[i].checked = v));
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i].s === s));
    draw();
  }

  const truth = (x: number) => (st.fn === "bump" ? bump(x) : st.fn === "step" ? step(x) : cellProfile ? cellProfile(x) : 0);

  const marks = {
    truth: plot.fn(truth, { color: "var(--qb-muted, #777)", dash: "6 5", width: 2.5, label: "true" }, 400),
    lin: plot.line([], [], { color: "var(--qb-c3, #2ca02c)", width: 3, label: "piecewise linear" }),
    spline: plot.line([], [], { color: "var(--qb-c1, #1f77b4)", width: 3.5, label: "cubic spline" }),
    poly: plot.line([], [], { color: "var(--qb-c2, #d62728)", width: 3.5, label: "polynomial" }),
  };
  const dots = plot.scatter([], [], { color: "var(--qb-fg, #222)", r: 8 });
  let ys: number[] = [], xs: number[] = [];

  function samples(): void {
    xs = st.nodes === "equi" ? equispacedNodes(st.n) : chebyshevNodes(st.n);
    ys = xs.map(truth);
    if (st.moved && st.moved[0] < ys.length) ys[st.moved[0]] += st.moved[1];
  }

  function draw(): void {
    if (st.fn === "cell" && !cellProfile) {
      loadCell().then((img: Grey) => {
        const b = gaussianBlur(img, 1.5), row = 120, w = img.width;
        const vals = Array.from({ length: w }, (_, x) => b.data[row * w + x] / 255);
        cellProfile = (x: number) => { const u = (x + 1) / 2 * (w - 1), i = Math.max(0, Math.min(w - 2, Math.floor(u))), f = u - i; return vals[i] * (1 - f) + vals[i + 1] * f; };
        draw();
      });
    }
    credit.style.visibility = st.fn === "cell" ? "visible" : "hidden";
    fnBtns.forEach((b, i) => b.classList.toggle("active", i < 3 ? ["bump", "step", "cell"][i] === st.fn : ["equi", "cheb"][i - 3] === st.nodes));
    samples();
    const P = polyInterp(xs, ys), S = naturalCubic(xs, ys);
    const gx: number[] = [], gp: number[] = [], gl: number[] = [], gs: number[] = [];
    for (let k = 0; k <= 600; k++) {
      const x = -1 + 2 * k / 600;
      gx.push(x); gp.push(P(x)); gl.push(linearInterp(xs, ys, x)); gs.push(S.at(x));
    }
    marks.poly.set(st.poly ? gx : [], gp);
    marks.lin.set(st.lin ? gx : [], gl);
    marks.spline.set(st.spline ? gx : [], gs);
    marks.truth.set(st.truth ? truth : () => NaN);
    dots.set(xs, ys);
    plot.legend(undefined, "tr");
    attachDrag();
    const err = (vals: number[]) => {
      let m = 0;
      for (let k = 0; k < gx.length; k++) {
        if (st.fn === "step" && Math.abs(gx[k] - 0.1) < 0.02) continue;   // skip the jump itself
        m = Math.max(m, Math.abs(vals[k] - truth(gx[k])));
      }
      return m;
    };
    const fmt = (v: number) => (v >= 100 ? v.toExponential(1) : v.toFixed(3));
    ro.textContent = `max |error| (n = ${st.n})\n` +
      `  polynomial  ${fmt(err(gp)).padStart(8)}\n  linear      ${fmt(err(gl)).padStart(8)}\n  spline      ${fmt(err(gs)).padStart(8)}` +
      (st.moved ? "\n(against the unmoved function)" : "");
  }

  // dragging the sample dots vertically
  function attachDrag(): void {
    const circles = dots.g.querySelectorAll("circle");
    circles.forEach((c, i) => {
      (c as SVGCircleElement).style.cursor = "ns-resize";
      (c as SVGCircleElement).style.touchAction = "none";
      c.addEventListener("pointerdown", ev => {
        ev.preventDefault();
        const target = ev.target as SVGCircleElement;
        const base = truth(xs[i]);
        const move = (e: PointerEvent) => {
          const [, y] = plot.toData(e);
          st.moved = [i, Math.max(-0.5, Math.min(1.5, y)) - base];
          pbtns.forEach(b => b.classList.remove("active"));
          redrawKeepingDot(i);
        };
        const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); draw(); };
        void target;
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
      });
    });
  }
  // during a drag: recompute the curves without rebuilding the listeners
  function redrawKeepingDot(_i: number): void {
    samples();
    const P = polyInterp(xs, ys), S = naturalCubic(xs, ys);
    const gx: number[] = [], gp: number[] = [], gl: number[] = [], gs: number[] = [];
    for (let k = 0; k <= 600; k++) { const x = -1 + 2 * k / 600; gx.push(x); gp.push(P(x)); gl.push(linearInterp(xs, ys, x)); gs.push(S.at(x)); }
    marks.poly.set(st.poly ? gx : [], gp); marks.lin.set(st.lin ? gx : [], gl); marks.spline.set(st.spline ? gx : [], gs);
    const circles = dots.g.querySelectorAll("circle");
    circles.forEach((c, k) => c.setAttribute("cy", plot.y(ys[k]).toFixed(2)));
  }

  load(p0.s);
}

autoMount(NAME, mount);
