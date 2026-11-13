// B-spline and NURBS curves (splines day, unit 2). One control polygon (drag, add, delete
// points) drawn as a B-spline of degree 1–5 (open, clamped or closed) and, in a small panel,
// as one Bézier curve through the same points. While a point is dragged, the part of each
// curve that changes is highlighted: a few spans of the B-spline, all of the Bézier. A
// selected point gets a weight slider (NURBS: the curve is evaluated in homogeneous
// coordinates (w x, w y, w) and divided by w). Toggles: curvature comb, knot markers, convex
// hulls of the spans. Circle presets show the largest radius error.
// <div data-demo="bsplinecurve" data-preset="local|square|closed|triple|quarter|quarter1|circle|ellipse|parabola|hyperbola"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { CtrlPoly } from "../lib/ctrlpoly";
import { convexHull } from "../lib/convexhull";
import { Pt, SplineCurve, KnotKind, curveFrom, curveDomain, curveDerivs, curvePoint, curvature, influence, bezierPoint, knotDomain } from "../lib/spline";

const NAME = "bsplinecurve";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; }
.qb-demo[data-demo="${NAME}"] .bc-main { flex: 1 1 480px; max-width: 600px; }
.qb-demo[data-demo="${NAME}"] .bc-side { flex: 1 1 360px; max-width: 520px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .bc-top { display: flex; gap: .5em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .bc-top .qb-plot { flex: 0 0 210px; max-width: 210px; }
.qb-demo[data-demo="${NAME}"] .bc-top figcaption { font-size: 16px; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 17px; white-space: pre-wrap; flex: 1 1 auto; min-width: 0; }
.qb-demo[data-demo="${NAME}"] .bc-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .bc-row { display: flex; flex-wrap: wrap; gap: .2em .9em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .bc-main { flex: 0 0 560px; }
.reveal .qb-demo[data-demo="${NAME}"] .bc-side { flex: 0 0 540px; gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .04em .38em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { padding: .2em .4em; font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .bc-lab, .reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

interface State {
  P: Pt[]; w: number[] | null; p: number; kind: KnotKind; knots: number[] | null; sel: number;
  comb: boolean; showKnots: boolean; hulls: boolean; circle: [number, number] | null;
}
const R2 = Math.SQRT1_2;
const WAVE: Pt[] = [[-4, -1], [-3, 2], [-1.5, -1.5], [0, 2], [1.5, -1.5], [3, 2], [4.5, -1], [5.5, 1.5]];
const BLOB: Pt[] = [[0, -3], [3, -2.5], [4, 0.5], [2, 1.5], [3, 4], [-1, 3.5], [-3.5, 2], [-2.5, -1.5]];
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "local", label: "local vs global", s: { P: WAVE, p: 3, kind: "clamped", sel: 4 } },
  { id: "square", label: "corner cutting on a square", s: { P: [[0, 0], [4, 0], [4, 4], [0, 4]], p: 3, kind: "closed", sel: 1, showKnots: true } },
  { id: "closed", label: "closed outline", s: { P: BLOB, p: 3, kind: "closed", sel: 0, comb: true } },
  { id: "triple", label: "triple knot: corner", s: { P: [[-4, -2], [-3, 2], [-1, 3], [0.5, 0], [2, 3], [4, 2], [5, -2]], p: 3, kind: "clamped", knots: [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2], sel: 3, showKnots: true } },
  { id: "quarter", label: "quarter circle: weight 1/√2", s: { P: [[4, 0], [4, 4], [0, 4]], w: [1, R2, 1], p: 2, kind: "clamped", sel: 1, circle: [0, 0] } },
  { id: "quarter1", label: "weight 1: not a circle", s: { P: [[4, 0], [4, 4], [0, 4]], w: [1, 1, 1], p: 2, kind: "clamped", sel: 1, circle: [0, 0] } },
  { id: "circle", label: "full circle from 9 points", s: { P: [[3, 0], [3, 3], [0, 3], [-3, 3], [-3, 0], [-3, -3], [0, -3], [3, -3], [3, 0]], w: [1, R2, 1, R2, 1, R2, 1, R2, 1], p: 2, kind: "clamped", knots: [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4], sel: 1, circle: [0, 0], showKnots: true } },
  { id: "ellipse", label: "w = 0.4: ellipse arc", s: { P: [[-3, -2], [0, 3], [3, -2]], w: [1, 0.4, 1], p: 2, kind: "clamped", sel: 1 } },
  { id: "parabola", label: "w = 1: parabola", s: { P: [[-3, -2], [0, 3], [3, -2]], w: [1, 1, 1], p: 2, kind: "clamped", sel: 1 } },
  { id: "hyperbola", label: "w = 2.5: hyperbola", s: { P: [[-3, -2], [0, 3], [3, -2]], w: [1, 2.5, 1], p: 2, kind: "clamped", sel: 1 } },
];
const DEFAULT: State = { P: WAVE, w: null, p: 3, kind: "clamped", knots: null, sel: -1, comb: false, showKnots: false, hulls: false, circle: null };
const W_MIN = -1, W_MAX = 1;   // log10 range of the weight slider

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...DEFAULT };

  const main = plotFigure(host, "B-spline (or NURBS)", "bc-main");
  const plot = new Plot(main, { width: 600, height: 520, x: [-5.2, 6.2], y: [-4.5, 5.2], equal: true, xticks: [-4, -2, 0, 2, 4, 6], yticks: [-4, -2, 0, 2, 4] });
  const side = htmlEl("div", "bc-side", "", host);
  const top = htmlEl("div", "bc-top", "", side);
  const bfig = plotFigure(top, "one Bézier, same points");
  const bplot = new Plot(bfig, { width: 300, height: 270, x: [-5.2, 6.2], y: [-4.5, 5.2], equal: true, xticks: [], yticks: [], margin: { l: 6, r: 6, t: 6, b: 6 } });
  const ro = htmlEl("div", "qb-readout", "", top);
  htmlEl("div", "bc-lab", "presets", side);
  const pbtns = presets(side, PRESETS.map(p => ({ label: p.label, apply: () => load(p.s) })));
  htmlEl("div", "bc-lab", "degree · knots", side);
  const row = htmlEl("div", "qb-presets", "", side);
  const degBtns = [1, 2, 3, 4, 5].map(d => button(row, "p = " + d, () => { st.p = d; st.knots = null; draw(); }));
  const kindBtns = (["open", "clamped", "closed"] as KnotKind[]).map(k => button(row, k, () => { st.kind = k; st.knots = null; cp.setClosed(k === "closed"); bp.setClosed(false); draw(); }));
  const row2 = htmlEl("div", "qb-presets", "", side);
  button(row2, "add point", () => { syncW(); cp.setWeights(st.w!); cp.add(); st.P = cp.pts; st.w = cp.weights!.slice(); st.sel = cp.selected; st.knots = null; bp.set(st.P); syncSlider(); draw(); });
  button(row2, "delete point", () => { syncW(); cp.setWeights(st.w!); cp.remove(); st.P = cp.pts; st.w = cp.weights!.slice(); st.sel = cp.selected; st.knots = null; bp.set(st.P); syncSlider(); draw(); });
  button(row2, "reset", () => load(p0.s));
  const wS = slider(side, "weight of the selected point", W_MIN, W_MAX, 0.005, 0, u => {
    if (st.sel < 0) return;
    syncW();
    st.w![st.sel] = Math.pow(10, u);
    cp.setWeights(st.w!);
    pbtns.forEach(b => b.classList.remove("active"));
    draw();
  }, u => Math.pow(10, u).toFixed(3));
  const row3 = htmlEl("div", "bc-row", "", side);
  const cks = [
    checkbox(row3, "curvature comb", st.comb, v => { st.comb = v; draw(); }),
    checkbox(row3, "knots", st.showKnots, v => { st.showKnots = v; draw(); }),
    checkbox(row3, "convex hulls of spans", st.hulls, v => { st.hulls = v; draw(); }),
  ];
  htmlEl("div", "qb-note", "Drag a point: the orange part of each curve is what changes. Select a point to change its weight.", side);

  const under = svgEl("g", {}, plot.layer), bunder = svgEl("g", {}, bplot.layer);
  let before: SplineCurve | null = null, beforeBez: Pt[] | null = null, dragging = -1;
  const cp = new CtrlPoly(plot, {
    points: [], label: "P",
    onSelect: i => { st.sel = i; syncSlider(); draw(); },
    onChange: (pts, i) => {
      if (dragging !== i) { dragging = i; before = makeCurve(); beforeBez = st.P.map(p => [...p] as Pt); }
      st.P = pts; bp.pts = pts.map(p => [...p] as Pt); bp.update();
      pbtns.forEach(b => b.classList.remove("active"));
      draw();
    },
    onRelease: () => { dragging = -1; draw(); },
  });
  const bp = new CtrlPoly(bplot, { points: [], r: 7, fixed: () => true });

  function syncW(): void { if (!st.w) st.w = st.P.map(() => 1); }
  function syncSlider(): void {
    const w = st.sel >= 0 && st.w ? st.w[st.sel] : 1;
    wS.set(Math.log10(w));
    wS.input.disabled = st.sel < 0;
  }

  function load(s: Partial<State>): void {
    Object.assign(st, DEFAULT, s);
    st.P = (s.P ?? DEFAULT.P).map(p => [...p] as Pt);
    st.w = s.w ? s.w.slice() : null;
    st.knots = s.knots ? s.knots.slice() : null;
    cks[0].checked = st.comb; cks[1].checked = st.showKnots; cks[2].checked = st.hulls;
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i].s === s));
    cp.setOptions({ closed: st.kind === "closed", weights: st.w ?? undefined });
    cp.set(st.P); cp.selected = st.sel; cp.draw();
    bp.set(st.P);
    before = null; beforeBez = null;
    syncSlider();
    draw();
  }

  function makeCurve(): SplineCurve {
    const n = st.P.length;
    if (st.knots && st.knots.length === n + st.p + 1) return { P: st.P.slice(), w: st.w?.slice(), p: st.p, knots: st.knots, nUser: n, kind: "clamped" };
    st.knots = null;
    return curveFrom(st.P, Math.min(st.p, n - 1), st.kind, st.w ?? undefined);
  }

  function draw(): void {
    const c = makeCurve(), X = plot.x, Y = plot.y;
    degBtns.forEach((b, i) => b.classList.toggle("active", i + 1 === c.p));
    kindBtns.forEach((b, i) => b.classList.toggle("active", ["open", "clamped", "closed"][i] === st.kind && !st.knots));
    under.replaceChildren(); bunder.replaceChildren();
    const path = (g: SVGGElement, P: Plot, pts: Pt[], style: string, close = false) => {
      const e = svgEl("path", { d: pts.map((p, i) => (i ? "L" : "M") + P.x(p[0]).toFixed(1) + "," + P.y(p[1]).toFixed(1)).join("") + (close ? "Z" : ""), fill: "none" }, g);
      e.style.cssText = style;
    };
    const [a, b] = curveDomain(c);
    const sampleRange = (lo: number, hi: number, cc: SplineCurve, n = 160): Pt[] => {
      const out: Pt[] = [];
      for (let k = 0; k <= n; k++) out.push(curvePoint(cc, Math.min(lo + (hi - lo) * k / n, hi - 1e-12 * (k === n ? 1 : 0))));
      return out;
    };
    // span hulls
    const distinct = c.knots.filter((v, i) => v >= a && v <= b && (i === 0 || v !== c.knots[i - 1]));
    if (st.hulls) {
      for (let s = 0; s + 1 < distinct.length; s++) {
        const mid = (distinct[s] + distinct[s + 1]) / 2;
        let span = c.p;
        while (span + 1 < c.knots.length && c.knots[span + 1] <= mid) span++;
        const pts = c.P.slice(span - c.p, span + 1);
        path(under, plot, convexHull(pts), `fill:${s % 2 ? "var(--qb-c3,#2ca02c)" : "var(--qb-c1,#1f77b4)"};fill-opacity:.09;stroke:none`, true);
      }
    }
    if (before) path(under, plot, sampleRange(...curveDomain(before), before, 300), "stroke:var(--qb-muted,#999);stroke-width:2.5;stroke-dasharray:4 4");
    // comb
    if (st.comb) {
      const samples: { p: Pt; d1: Pt; k: number }[] = [];
      for (let k = 0; k <= 240; k++) {
        const t = a + (b - a) * Math.min(k / 240, 1 - 1e-9), [p, d1, d2] = curveDerivs(c, t, 2);
        samples.push({ p, d1, k: curvature(d1, d2) });
      }
      const km = Math.max(1e-9, ...samples.map(s => Math.min(3, Math.abs(s.k)))), sc = 1.5 / km, tips: Pt[] = [];
      for (const s of samples) {
        const L = Math.hypot(...s.d1) || 1, tip: Pt = [s.p[0] + sc * s.k * s.d1[1] / L, s.p[1] - sc * s.k * s.d1[0] / L];
        tips.push(tip);
        path(under, plot, [s.p, tip], "stroke:#e377c2;stroke-width:1.3;opacity:.8");
      }
      path(under, plot, tips, "stroke:#e377c2;stroke-width:2");
    }
    path(under, plot, sampleRange(a, b, c, 400), "stroke:var(--qb-fg,#222);stroke-width:4");
    // highlight what changes while dragging
    if (dragging >= 0) for (const [lo, hi] of influence(c, dragging)) path(under, plot, sampleRange(lo, hi, c, 120), "stroke:var(--qb-c4,#ff7f0e);stroke-width:7;opacity:.85");
    if (st.showKnots) for (const k of distinct) {
      const q = curvePoint(c, Math.min(k, b - 1e-12));
      const e = svgEl("rect", { x: X(q[0]) - 6, y: Y(q[1]) - 6, width: 12, height: 12 }, under);
      e.style.cssText = "fill:var(--qb-c2,#d62728)";
    }
    // the single Bézier
    const bz: Pt[] = [];
    for (let k = 0; k <= 300; k++) bz.push(bezierPoint(st.P, k / 300));
    if (beforeBez && dragging >= 0) {
      const old: Pt[] = [];
      for (let k = 0; k <= 300; k++) old.push(bezierPoint(beforeBez, k / 300));
      path(bunder, bplot, old, "stroke:var(--qb-muted,#999);stroke-width:2;stroke-dasharray:4 4");
    }
    path(bunder, bplot, bz, `stroke:${dragging >= 0 ? "var(--qb-c4,#ff7f0e)" : "var(--qb-fg,#222)"};stroke-width:${dragging >= 0 ? 5 : 3}`);
    // readout
    const n = st.P.length;
    let txt = `${n} points, degree ${c.p}, ${st.knots ? "custom knots" : st.kind}\nknots ${fmtKnots(c.knots)}`;
    if (st.sel >= 0 && st.sel < n) {
      const inf = influence(c, st.sel).reduce((s, [lo, hi]) => s + countSpans(distinct, lo, hi), 0);
      txt += `\nP${st.sel} moves ${inf} span${inf === 1 ? "" : "s"}; w = ${(st.w?.[st.sel] ?? 1).toFixed(3)}`;
      // curve point at the centre of P_sel's bump
      const idx = st.sel, lo = c.knots[idx], hi = c.knots[idx + c.p + 1];
      let tc = (lo + hi) / 2;
      if (c.kind === "closed" && tc < a) tc += n;
      if (tc >= a && tc <= b) { const q = curvePoint(c, Math.min(tc, b - 1e-12)); txt += `\nC at its bump centre (${q[0].toFixed(3)}, ${q[1].toFixed(3)})`; }
    }
    if (st.circle) {
      let rmin = Infinity, rmax = 0;
      for (let k = 0; k <= 400; k++) { const q = curvePoint(c, a + (b - a) * Math.min(k / 400, 1 - 1e-12)); const r = Math.hypot(q[0] - st.circle[0], q[1] - st.circle[1]); rmin = Math.min(rmin, r); rmax = Math.max(rmax, r); }
      const r0 = Math.hypot(st.P[0][0] - st.circle[0], st.P[0][1] - st.circle[1]);
      txt += `\nradius ${rmin.toFixed(4)} … ${rmax.toFixed(4)}\nmax error ${(100 * Math.max(rmax - r0, r0 - rmin) / r0).toFixed(2)} %`;
    }
    ro.textContent = txt;
  }

  load(p0.s);
}

function countSpans(distinct: number[], lo: number, hi: number): number {
  let k = 0;
  for (let s = 0; s + 1 < distinct.length; s++) if (distinct[s] >= lo - 1e-12 && distinct[s + 1] <= hi + 1e-12) k++;
  return k;
}
function fmtKnots(k: number[]): string {
  const s = k.map(v => String(Math.round(v * 100) / 100));
  return s.length > 12 ? s.slice(0, 5).join(" ") + " … " + s.slice(-4).join(" ") : s.join(" ");
}
void knotDomain;

autoMount(NAME, mount);
