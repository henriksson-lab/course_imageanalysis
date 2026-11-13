// De Casteljau and Bézier curves (splines day, unit 1). Drag the control points of a Bézier
// curve (degree 2–6); the slider t shows the construction (each level of linear
// interpolations in its own colour) and the Bernstein weights of every control point at t
// (the plot below, with the numbers). Toggles: convex hull, curvature comb, tangent at t.
// A second cubic piece can be attached with a join rule (free, C⁰, G¹, C¹); the comb shows
// what each join does to the curvature. "Transform" rotates and scales the control points:
// the transformed old curve (dots) lies exactly on the new curve (affine invariance).
// <div data-demo="casteljau" data-preset="worked|s|loop|cusp|c0|g1|c1"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { CtrlPoly } from "../lib/ctrlpoly";
import { convexHull as hull } from "../lib/convexhull";
import { Pt, deCasteljau, bernstein, bezierPoint, bezierDerivs, curvature } from "../lib/spline";

const NAME = "casteljau";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; }
.qb-demo[data-demo="${NAME}"] .cj-main { flex: 1 1 500px; max-width: 620px; }
.qb-demo[data-demo="${NAME}"] .cj-side { flex: 1 1 330px; max-width: 470px; display: flex; flex-direction: column; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 18px; white-space: pre; }
.qb-demo[data-demo="${NAME}"] .cj-side .qb-plot { flex: none; max-width: none; }
.qb-demo[data-demo="${NAME}"] .cj-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .cj-row { display: flex; flex-wrap: wrap; gap: .3em 1em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .cj-main { flex: 0 0 600px; }
.reveal .qb-demo[data-demo="${NAME}"] .cj-side { flex: 0 0 470px; gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .05em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { padding: .15em .45em; font-size: 17px; line-height: 1.25; }
.reveal .qb-demo[data-demo="${NAME}"] .cj-lab { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

type Join = "none" | "free" | "C0" | "G1" | "C1";
interface State { P: Pt[]; Q: Pt[]; join: Join; t: number; hull: boolean; comb: boolean; tangent: boolean; construct: boolean }
const LEVEL_COL = ["var(--qb-c1, #1f77b4)", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#d62728"];
const Q0: Pt[] = [[4, 0], [5, -2], [7, -1.5], [8, 1]];
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "worked", label: "the worked example", s: { P: [[0, 0], [1, 2], [3, 2], [4, 0]], join: "none", t: 0.5 } },
  { id: "s", label: "S-curve", s: { P: [[0, 0], [3, 3], [1, -2], [4, 1]], join: "none", t: 0.5, comb: true } },
  { id: "loop", label: "loop", s: { P: [[0, 0], [5, 3], [-1, 3], [4, 0]], join: "none", t: 0.5 } },
  { id: "cusp", label: "cusp (inner legs crossed)", s: { P: [[0, 0], [4, 2.5], [0, 2.5], [4, 0]], join: "none", t: 0.5, comb: true } },
  { id: "c0", label: "C⁰ join: a kink", s: { P: [[0, 0], [1, 2], [3, 2], [4, 0]], Q: [[4, 0], [4.5, 2], [6.5, 2.5], [8, 1]], join: "C0", comb: true, construct: false } },
  { id: "g1", label: "G¹ but not C¹: the comb jumps", s: { P: [[0, 0], [1, 2], [3, 2], [4, 0]], Q: [[4, 0], [4.4, -0.8], [7, -1.5], [8, 1]], join: "G1", comb: true, construct: false } },
  { id: "c1", label: "C¹ join", s: { P: [[0, 0], [1, 2], [3, 2], [4, 0]], Q: [[4, 0], [5, -2], [7, -1.5], [8, 1]], join: "C1", comb: true, construct: false } },
];
const DEFAULT: State = { P: PRESETS[0].s.P!, Q: Q0, join: "none", t: 0.5, hull: false, comb: false, tangent: true, construct: true };

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...DEFAULT, ...p0.s, P: [], Q: [] };

  const main = plotFigure(host, "", "cj-main");
  const plot = new Plot(main, { width: 620, height: 470, x: [-1.5, 9], y: [-3.5, 4.5], equal: true, xticks: [0, 2, 4, 6, 8], yticks: [-2, 0, 2, 4] });
  const side = htmlEl("div", "cj-side", "", host);
  const bfig = plotFigure(side, "");
  const bplot = new Plot(bfig, { width: 470, height: 165, x: [0, 1], y: [0, 1.02], ylabel: "weight", xticks: [0, 0.25, 0.5, 0.75, 1], yticks: [0, 0.5, 1], xfmt: v => String(v), margin: { b: 34 } });
  const tS = slider(side, "t", 0, 1, 0.01, st.t, v => { st.t = v; draw(); });
  const ro = htmlEl("div", "qb-readout", "", side);
  htmlEl("div", "cj-lab", "presets", side);
  const pbtns = presets(side, PRESETS.map(p => ({ label: p.label, apply: () => load(p.s) })));
  htmlEl("div", "cj-lab", "degree · second piece", side);
  const row1 = htmlEl("div", "qb-presets", "", side);
  const degBtns = [2, 3, 4, 5, 6].map(d => button(row1, String(d), () => setDegree(d)));
  const joinBtns = (["none", "free", "C0", "G1", "C1"] as Join[]).map(j => button(row1, j === "none" ? "no 2nd" : j === "C0" ? "C⁰" : j === "G1" ? "G¹" : j === "C1" ? "C¹" : "free", () => { st.join = j; applyJoin(); draw(); }));
  const row2 = htmlEl("div", "cj-row", "", side);
  const ck = [
    checkbox(row2, "construction", st.construct, v => { st.construct = v; draw(); }),
    checkbox(row2, "convex hull", st.hull, v => { st.hull = v; draw(); }),
    checkbox(row2, "curvature comb", st.comb, v => { st.comb = v; draw(); }),
    checkbox(row2, "tangent", st.tangent, v => { st.tangent = v; draw(); }),
  ];
  const row3 = htmlEl("div", "qb-presets", "", side);
  button(row3, "transform (rotate 30°, scale 0.8)", () => transform());
  button(row3, "reset", () => load(p0.s));
  htmlEl("div", "qb-note", "Drag the control points; dashed handles follow the join rule.", side);

  // layers below the control polygons
  const under = svgEl("g", {}, plot.layer);
  const cpP = new CtrlPoly(plot, { points: [], label: "P", onChange: () => { applyJoin(); cpQ.update(); draw(); }, onRelease: () => draw() });
  const cpQ = new CtrlPoly(plot, {
    points: [], label: "Q", color: "var(--qb-c5, #9467bd)",
    constrained: i => (i === 0 && st.join !== "free") || (i === 1 && st.join === "C1"),
    onChange: (_p, i) => { if (i === 1 && st.join === "G1") projectG1(); applyJoin(); cpQ.update(); draw(); },
  });
  let ghost: Pt[] = [];

  function load(s: Partial<State>): void {
    Object.assign(st, DEFAULT, s);
    st.P = (s.P ?? DEFAULT.P).map(p => [...p] as Pt);
    st.Q = (s.Q ?? Q0).map(p => [...p] as Pt);
    ghost = [];
    tS.set(st.t);
    [st.construct, st.hull, st.comb, st.tangent].forEach((v, i) => (ck[i].checked = v));
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i].s === s));
    cpP.set(st.P); cpQ.set(st.Q);
    applyJoin(); cpQ.set(st.Q);
    draw();
  }

  function setDegree(d: number): void {
    const P = st.P, n = P.length - 1;
    if (d === n) return;
    // keep the end points, spread new inner points along the old curve
    const pts: Pt[] = [];
    for (let i = 0; i <= d; i++) pts.push(i === 0 ? P[0] : i === d ? P[n] : bezierPoint(P, i / d));
    // lift them off the curve a little so they are control points, not curve points
    st.P = pts.map((p, i) => (i === 0 || i === d ? p : [p[0], p[1] + 0.6 * Math.sin(Math.PI * i / d)] as Pt));
    ghost = [];
    cpP.set(st.P); applyJoin(); cpQ.set(st.Q); draw();
  }

  /** Enforce the join rule on the second piece. */
  function applyJoin(): void {
    st.P = cpP.pts; st.Q = cpQ.pts.length ? cpQ.pts : st.Q;
    if (st.join === "none" || st.join === "free") return;
    const n = st.P.length - 1, end = st.P[n], prev = st.P[n - 1];
    st.Q[0] = [end[0], end[1]];
    if (st.join === "C1") {
      // Q1 − Q0 = (n / 3)(P_n − P_{n−1}) so that the velocities agree (Q is a cubic)
      const k = n / 3;
      st.Q[1] = [end[0] + k * (end[0] - prev[0]), end[1] + k * (end[1] - prev[1])];
    }
    if (st.join === "G1") projectG1();
    cpQ.pts = st.Q;
  }
  /** G¹: Q1 on the ray from P_n in the direction P_n − P_{n−1}, at its current distance. */
  function projectG1(): void {
    const n = st.P.length - 1, end = st.P[n], prev = st.P[n - 1];
    const dx = end[0] - prev[0], dy = end[1] - prev[1], L = Math.hypot(dx, dy) || 1;
    const q = st.Q[1], s = Math.max(0.2, ((q[0] - end[0]) * dx + (q[1] - end[1]) * dy) / L);
    st.Q[1] = [end[0] + s * dx / L, end[1] + s * dy / L];
  }

  function transform(): void {
    const all = st.join === "none" ? st.P : [...st.P, ...st.Q];
    const cx = all.reduce((a, p) => a + p[0], 0) / all.length, cy = all.reduce((a, p) => a + p[1], 0) / all.length;
    const a = Math.PI / 6, c = 0.8 * Math.cos(a), s = 0.8 * Math.sin(a);
    const T = (p: Pt): Pt => [cx + c * (p[0] - cx) - s * (p[1] - cy), cy + s * (p[0] - cx) + c * (p[1] - cy)];
    ghost = [];
    for (let k = 0; k <= 40; k++) ghost.push(T(bezierPoint(st.P, k / 40)));
    st.P = st.P.map(T); st.Q = st.Q.map(T);
    cpP.set(st.P); cpQ.set(st.Q); draw();
  }

  const fmt = (v: number) => v.toFixed(3);
  function draw(): void {
    const X = plot.x, Y = plot.y, P = st.P, n = P.length - 1, t = st.t;
    degBtns.forEach((b, i) => b.classList.toggle("active", i + 2 === n));
    joinBtns.forEach((b, i) => b.classList.toggle("active", ["none", "free", "C0", "G1", "C1"][i] === st.join));
    cpQ.g.style.display = st.join === "none" ? "none" : "";
    under.replaceChildren();
    const path = (pts: Pt[], style: string, close = false) => {
      const e = svgEl("path", { d: pts.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join("") + (close ? "Z" : ""), fill: "none" }, under);
      e.style.cssText = style;
      return e;
    };
    const pieces: Pt[][] = st.join === "none" ? [P] : [P, st.Q];
    // convex hulls
    if (st.hull) for (const C of pieces) path(hull(C), "fill:var(--qb-c1,#1f77b4);fill-opacity:.08;stroke:var(--qb-c1,#1f77b4);stroke-width:1.5;stroke-opacity:.5", true);
    // curves and combs
    let kmax = 1e-9;
    if (st.comb) for (const C of pieces) for (let j = 0; j <= 60; j++) { const [a, b] = bezierDerivs(C, j / 60); kmax = Math.max(kmax, Math.min(5, Math.abs(curvature(a, b)))); }
    pieces.forEach((C, k) => {
      const pts: Pt[] = [];
      for (let j = 0; j <= 200; j++) pts.push(bezierPoint(C, j / 200));
      if (st.comb) {
        const tips: Pt[] = [], sc = 1.6 / kmax;
        for (let j = 0; j <= 60; j++) {
          const u = j / 60, p = bezierPoint(C, u), [d1, d2] = bezierDerivs(C, u), kap = curvature(d1, d2), L = Math.hypot(d1[0], d1[1]) || 1;
          // comb along the normal, pointing away from the centre of curvature
          const nx = d1[1] / L, ny = -d1[0] / L, tip: Pt = [p[0] + sc * kap * nx, p[1] + sc * kap * ny];
          tips.push(tip);
          path([p, tip], "stroke:#e377c2;stroke-width:1.5;opacity:.8");
        }
        path(tips, "stroke:#e377c2;stroke-width:2");
      }
      path(pts, `stroke:${k ? "var(--qb-c5,#9467bd)" : "var(--qb-fg,#222)"};stroke-width:4`);
    });
    if (ghost.length) for (const g of ghost) { const c = svgEl("circle", { cx: X(g[0]), cy: Y(g[1]), r: 4.5 }, under); c.style.cssText = "fill:var(--qb-c4,#ff7f0e)"; }
    // construction
    const L = deCasteljau(P, t);
    if (st.construct) L.forEach((lev, k) => {
      if (k === 0) return;
      if (lev.length > 1) path(lev, `stroke:${LEVEL_COL[k % LEVEL_COL.length]};stroke-width:2.5`);
      for (const p of lev) { const c = svgEl("circle", { cx: X(p[0]), cy: Y(p[1]), r: k === L.length - 1 ? 9 : 6 }, under); c.style.cssText = `fill:${k === L.length - 1 ? "var(--qb-c2,#d62728)" : LEVEL_COL[k % LEVEL_COL.length]}`; }
    });
    const pt = L[L.length - 1][0];
    if (!st.construct) { const c = svgEl("circle", { cx: X(pt[0]), cy: Y(pt[1]), r: 9 }, under); c.style.cssText = "fill:var(--qb-c2,#d62728)"; }
    const [d1, d2] = bezierDerivs(P, t);
    if (st.tangent) {
      const s = 0.25;
      path([[pt[0] - s * d1[0], pt[1] - s * d1[1]], [pt[0] + s * d1[0], pt[1] + s * d1[1]]], "stroke:var(--qb-c2,#d62728);stroke-width:2;stroke-dasharray:5 4");
    }
    // Bernstein basis
    bplot.clear();
    for (let i = 0; i <= n; i++) bplot.fn(u => bernstein(n, i, u), { color: LEVEL_COL[i % LEVEL_COL.length], width: 3 }, 120);
    bplot.vline(t, { color: "var(--qb-c2, #d62728)" });
    for (let i = 0; i <= n; i++) bplot.scatter([t], [bernstein(n, i, t)], { color: LEVEL_COL[i % LEVEL_COL.length], r: 6 });
    const w = Array.from({ length: n + 1 }, (_, i) => bernstein(n, i, t));
    const kap = curvature(d1, d2);
    let txt = `t = ${t.toFixed(2)}   point (${fmt(pt[0])}, ${fmt(pt[1])})\nweights ${w.map(v => v.toFixed(3)).join("  ")}\n`;
    txt += `sum ${w.reduce((a, b) => a + b, 0).toFixed(3)}   κ(t) = ${kap.toFixed(3)}`;
    if (st.join !== "none") {
      const [e1, e2] = bezierDerivs(P, 1), [f1, f2] = bezierDerivs(st.Q, 0);
      const k1 = curvature(e1, e2), k2 = curvature(f1, f2);
      const gap = Math.hypot(P[n][0] - st.Q[0][0], P[n][1] - st.Q[0][1]);
      const ang = Math.abs(Math.atan2(e1[0] * f1[1] - e1[1] * f1[0], e1[0] * f1[0] + e1[1] * f1[1])) * 180 / Math.PI;
      txt += `\njoin: gap ${gap.toFixed(2)}, angle ${ang.toFixed(1)}°, |v| ${Math.hypot(...e1).toFixed(2)} → ${Math.hypot(...f1).toFixed(2)}\n      κ ${k1.toFixed(3)} → ${k2.toFixed(3)}`;
    }
    ro.textContent = txt;
  }

  load(p0.s);
}

autoMount(NAME, mount);
