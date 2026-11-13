// Second batch of static figures for the splines day (pictures on the slides).
// Computed with the demos' own libraries. Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/splines/figs2.ts --bundle --platform=node --log-level=warning | node - <rawdir>
// SVGs go straight into the unit folders. Raster data (micrograph zooms, rotations, FFD fold)
// goes as JSON into <rawdir>; tools/figs/splines/figs2_raster.py turns it into PNGs:
//   python3 -I tools/figs/splines/figs2_raster.py <rawdir>
// <rawdir>/cell256.json must exist first (the course micrograph as grey values):
//   python3 -I tools/figs/splines/figs2_raster.py <rawdir> --export-cell
import * as fs from "fs";
import * as S from "../../../demos/src/lib/spline";
import * as SI from "../../../demos/src/lib/splineimage";
import { rng } from "../../../demos/src/lib/random";
import { rasterise } from "../../../demos/src/lib/shapecanvas";
import { traceBoundary } from "../../../demos/src/splineoutline/boundary";
import { fromArray, createGrey, resize, Grey } from "../../../demos/src/lib/imageops";
import { checkerboard } from "../../../demos/src/lib/synth";
import { diverging } from "../../../demos/src/lib/colour";
import { fft1d } from "../../../demos/src/lib/fft";

const RAW = process.argv[2] ?? ".";
const C = { fg: "#222", grey: "#888", light: "#ddd", blue: "#1f77b4", red: "#d62728", green: "#2ca02c", orange: "#ff7f0e", purple: "#9467bd", pink: "#e377c2" };
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
type P2 = [number, number];
function svg(w: number, h: number, body: string): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" font-family="Helvetica, Arial, sans-serif" font-size="20">\n<rect width="${w}" height="${h}" fill="#fff"/>\n${body}\n</svg>\n`;
}
let FS = 1;   // text scale for the current figure (column figures are shown smaller than their viewBox)
const text = (x: number, y: number, s: string, o: { size?: number; anchor?: string; fill?: string; weight?: string } = {}) =>
  `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" font-size="${((o.size ?? 20) * FS).toFixed(1)}" text-anchor="${o.anchor ?? "middle"}" fill="${o.fill ?? C.fg}"${o.weight ? ` font-weight="${o.weight}"` : ""}>${esc(s)}</text>`;
const path = (pts: P2[], o: { stroke?: string; w?: number; fill?: string; op?: number; close?: boolean; dash?: string } = {}) =>
  `<path d="M${pts.map(p => p[0].toFixed(1) + "," + p[1].toFixed(1)).join("L")}${o.close ? "Z" : ""}" fill="${o.fill ?? "none"}" fill-opacity="${o.op ?? 1}" stroke="${o.stroke ?? C.fg}" stroke-width="${o.w ?? 2.5}" stroke-linejoin="round"${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}/>`;
const dot = (x: number, y: number, r: number, fill: string, stroke = "#fff") => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>`;
const arrow = (a: P2, b: P2, col: string, w = 2.5) => {
  const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L, s = 12;
  return path([a, b], { stroke: col, w }) + `<path d="M${b[0]},${b[1]}L${(b[0] - s * ux + 0.5 * s * uy).toFixed(1)},${(b[1] - s * uy - 0.5 * s * ux).toFixed(1)}L${(b[0] - s * ux - 0.5 * s * uy).toFixed(1)},${(b[1] - s * uy + 0.5 * s * ux).toFixed(1)}Z" fill="${col}"/>`;
};

function panel(x0: number, y0: number, w: number, h: number, xd: [number, number], yd: [number, number], o: { axes?: boolean; xt?: number[]; yt?: number[]; xl?: string; yl?: string; fmt?: (v: number) => string; title?: string; tsize?: number } = {}) {
  const X = (v: number) => x0 + (v - xd[0]) / (xd[1] - xd[0]) * w, Y = (v: number) => y0 + h - (v - yd[0]) / (yd[1] - yd[0]) * h;
  const M = (p: P2): P2 => [X(p[0]), Y(p[1])];
  let s = "";
  const fmt = o.fmt ?? ((v: number) => String(v));
  if (o.axes !== false) {
    for (const t of o.xt ?? []) s += `<line x1="${X(t)}" x2="${X(t)}" y1="${y0}" y2="${y0 + h}" stroke="#eee"/>` + text(X(t), y0 + h + 6 + 18 * FS, fmt(t), { size: 18 });
    for (const t of o.yt ?? []) s += `<line x1="${x0}" x2="${x0 + w}" y1="${Y(t)}" y2="${Y(t)}" stroke="#eee"/>` + text(x0 - 8, Y(t) + 6, fmt(t), { size: 18, anchor: "end" });
    s += path([[x0, y0], [x0, y0 + h], [x0 + w, y0 + h]], { w: 1.5 });
    if (o.xl) s += text(x0 + w / 2, y0 + h + 14 + 40 * FS, o.xl);
    if (o.yl) s += `<text x="${x0 - 52}" y="${y0 + h / 2}" font-size="${(20 * FS).toFixed(1)}" text-anchor="middle" transform="rotate(-90 ${x0 - 52} ${y0 + h / 2})">${esc(o.yl)}</text>`;
  }
  if (o.title) s += text(x0 + w / 2, y0 - 10, o.title, { weight: "bold", size: o.tsize ?? 20 });
  return { X, Y, M, s };
}
const sample = (f: (x: number) => number, a: number, b: number, n = 300): P2[] => Array.from({ length: n + 1 }, (_, k) => { const x = a + (b - a) * k / n; return [x, f(x)] as P2; });
const clip = (pts: P2[], lo: number, hi: number): P2[] => pts.map(([x, y]) => [x, Math.max(lo, Math.min(hi, y))] as P2);
const out = (file: string, s: string) => { fs.writeFileSync(file, s); console.log("wrote", file); };
const curvePts = (c: S.SplineCurve, n = 300): P2[] => { const [a, b] = S.curveDomain(c); return Array.from({ length: n + 1 }, (_, j) => S.curvePoint(c, a + (b - a) * Math.min(j / n, 1 - 1e-12)) as P2); };
const bez = (P: S.Pt[], n = 200): P2[] => Array.from({ length: n + 1 }, (_, j) => S.bezierPoint(P, j / n) as P2);

// ======================================================================= bezier/
{ // fig-poly5.svg: the degree-4 polynomial through five samples of a bump
  FS = 1.25;
  const p = (x: number) => (x * x - 1) * (x * x - 4) / 4;
  const Pn = panel(80, 20, 440, 300, [-2.3, 2.3], [-0.8, 1.25], { xt: [-2, -1, 0, 1, 2], yt: [-0.5, 0, 0.5, 1], xl: "x", yl: "p(x)" });
  let body = Pn.s + path([Pn.M([-2.3, 0]), Pn.M([2.3, 0])], { stroke: "#bbb", w: 1.5 });
  body += path(sample(p, -2.3, 2.3).map(Pn.M), { stroke: C.red, w: 3.5 });
  for (const [x, y] of [[-2, 0], [-1, 0], [0, 1], [1, 0], [2, 0]]) body += dot(Pn.X(x), Pn.Y(y), 7, C.fg);
  body += dot(Pn.X(1.5), Pn.Y(-0.546875), 7, C.red) + text(Pn.X(0), Pn.Y(-0.55) + 8, "p(±1.5) = −0.55", { fill: C.red });
  body += dot(Pn.X(-1.5), Pn.Y(-0.546875), 7, C.red);
  out("bezier/fig-poly5.svg", svg(560, 405, body));
}
{ // fig-kinks.svg: piecewise linear vs cubic through the same samples, and their slopes
  FS = 1.25;
  const xs = [0, 1, 2, 3, 4, 5], ys = [0.2, 1.3, 0.9, 2.0, 1.1, 1.6];
  const sp = S.naturalCubic(xs, ys), lin = (x: number) => S.linearInterp(xs, ys, x);
  const A = panel(80, 30, 440, 170, [-0.1, 5.1], [0, 2.3], { xt: [], yt: [0, 1, 2], yl: "value", title: "the curves" });
  const B = panel(80, 260, 440, 170, [-0.1, 5.1], [-2.2, 2.2], { xt: [0, 1, 2, 3, 4, 5], yt: [-2, 0, 2], xl: "x", yl: "slope", title: "their slopes" });
  let body = A.s + B.s;
  body += path(sample(lin, 0, 5).map(A.M), { stroke: C.green, w: 3 }) + path(sample(x => sp.at(x), 0, 5).map(A.M), { stroke: C.blue, w: 3 });
  xs.forEach((x, i) => (body += dot(A.X(x), A.Y(ys[i]), 6, C.fg)));
  const h = 1e-5;
  for (let i = 0; i < 5; i++) body += path([B.M([i, ys[i + 1] - ys[i]]), B.M([i + 1, ys[i + 1] - ys[i]])], { stroke: C.green, w: 3 });
  for (let i = 1; i < 5; i++) body += path([B.M([i, ys[i] - ys[i - 1]]), B.M([i, ys[i + 1] - ys[i]])], { stroke: C.green, w: 1.5, dash: "3 3" });
  body += path(sample(x => (sp.at(Math.min(5, x + h)) - sp.at(Math.max(0, x - h))) / (Math.min(5, x + h) - Math.max(0, x - h)), 0, 5).map(B.M), { stroke: C.blue, w: 3 });
  body += text(530, 80, "piecewise", { fill: C.green, anchor: "start", size: 18 }) + text(530, 102, "linear", { fill: C.green, anchor: "start", size: 18 });
  body += text(530, 140, "cubic", { fill: C.blue, anchor: "start", size: 18 }) + text(530, 162, "pieces", { fill: C.blue, anchor: "start", size: 18 });
  body += text(530, 330, "jumps", { fill: C.green, anchor: "start", size: 18 }) + text(530, 352, "= kinks", { fill: C.green, anchor: "start", size: 18 });
  out("bezier/fig-kinks.svg", svg(640, 510, body));
}
{ // fig-parametric.svg: a closed outline and its two coordinate functions
  FS = 1.1;
  const P: S.Pt[] = [[0, 0], [2, -0.6], [4, 0.2], [4.6, 2], [3.2, 2.4], [2.2, 1.4], [1, 2.6], [-0.6, 1.8]];
  const c = S.curveFrom(P, 3, "closed"), [a, b] = S.curveDomain(c), n = P.length;
  const pts = curvePts(c, 400);
  const A = panel(40, 20, 300, 220, [-1.2, 5.2], [-1.2, 3.4], { axes: false });
  let body = path(pts.map(A.M), { w: 3.5, close: true });
  for (let k = 0; k < n; k++) { const q = S.curvePoint(c, a + k) as P2; body += dot(...A.M(q), 6, C.blue); const ang = Math.atan2(q[1] - 1.1, q[0] - 2.1); body += text(A.X(q[0]) + 30 * Math.cos(ang), A.Y(q[1]) - 26 * Math.sin(ang) + 7, `t=${k}`, { size: 18, fill: C.blue }); }
  const d1 = S.curveDerivs(c, a + 0.5, 1)[1], p05 = S.curvePoint(c, a + 0.5) as P2, L = Math.hypot(...d1);
  body += arrow(A.M(p05), A.M([p05[0] + 0.9 * d1[0] / L, p05[1] + 0.9 * d1[1] / L]), C.red);
  const X = panel(400, 20, 230, 90, [0, n], [-1, 5], { xt: [], yt: [0, 4], yl: "x(t)" });
  const Y = panel(400, 150, 230, 90, [0, n], [-1, 3], { xt: [0, 4, 8], yt: [0, 2], yl: "y(t)", xl: "t" });
  body += X.s + Y.s;
  const ts = Array.from({ length: 401 }, (_, j) => n * j / 400);
  body += path(ts.map((t, j) => X.M([t, pts[j][0]])), { stroke: C.purple, w: 3 }) + path(ts.map((t, j) => Y.M([t, pts[j][1]])), { stroke: C.orange, w: 3 });
  out("bezier/fig-parametric.svg", svg(660, 320, body));
}
{ // fig-bezier-anatomy.svg: one cubic Bézier, annotated
  FS = 1.25;
  const P: S.Pt[] = [[0, 0], [1, 2], [3, 2], [4, 0]];
  const Pn = panel(20, 40, 520, 300, [-1.9, 5.9], [-0.5, 2.5], { axes: false });
  let body = path(P.map(Pn.M), { stroke: C.blue, w: 2, dash: "7 5" }) + path(bez(P).map(Pn.M), { w: 4.5 });
  body += arrow(Pn.M([0, 0]), Pn.M([0.55, 1.1]), C.red, 3.5) + arrow(Pn.M([4, 0]), Pn.M([3.45, 1.1]), C.red, 3.5);
  P.forEach((p, i) => (body += dot(...Pn.M(p as P2), 8, C.blue)));
  body += text(Pn.X(0), Pn.Y(0) + 30, "P0: start", { fill: C.blue }) + text(Pn.X(4), Pn.Y(0) + 30, "P3: end", { fill: C.blue });
  body += text(Pn.X(1), Pn.Y(2) - 14, "P1 pulls", { fill: C.blue }) + text(Pn.X(3), Pn.Y(2) - 14, "P2 pulls", { fill: C.blue });
  body += text(Pn.X(0.2) - 8, Pn.Y(0.9), "leaves", { fill: C.red, anchor: "end", size: 18 }) + text(Pn.X(0.2) - 8, Pn.Y(0.9) + 24, "along P0P1", { fill: C.red, anchor: "end", size: 18 });
  body += text(Pn.X(3.8) + 8, Pn.Y(0.9), "arrives", { fill: C.red, anchor: "start", size: 18 }) + text(Pn.X(3.8) + 8, Pn.Y(0.9) + 24, "along P2P3", { fill: C.red, anchor: "start", size: 18 });
  out("bezier/fig-bezier-anatomy.svg", svg(560, 385, body));
}
{ // fig-subdivide.svg: convex hull and the split at t = 1/2
  FS = 1.2;
  const P: S.Pt[] = [[0, 0], [1, 2], [3, 2], [4, 0]], [L, R] = S.bezierSplit(P, 0.5);
  const Pn = panel(60, 20, 440, 260, [-0.4, 4.4], [-0.3, 2.3], { xt: [0, 1, 2, 3, 4], yt: [0, 1, 2] });
  let body = Pn.s + path(P.map(Pn.M), { stroke: "none", fill: C.grey, op: 0.15, close: true });
  body += path(bez(P).map(Pn.M), { w: 4 });
  body += path(L.map(Pn.M), { stroke: C.orange, w: 2.5, dash: "7 4" }) + path(R.map(Pn.M), { stroke: C.purple, w: 2.5, dash: "7 4" });
  L.forEach(p => (body += dot(...Pn.M(p as P2), 7, C.orange))); R.forEach(p => (body += dot(...Pn.M(p as P2), 6, C.purple)));
  body += dot(...Pn.M([2, 1.5]), 8, C.fg) + text(Pn.X(2), Pn.Y(1.5) - 16, "(2, 1.5)", { size: 18 });
  body += text(Pn.X(2), Pn.Y(0.55), "convex hull", { fill: "#555", size: 18 });
  body += text(10, 350, "left half: (0,0) (0.5,1) (1.25,1.5) (2,1.5)", { fill: C.orange, anchor: "start", size: 17 });
  body += text(10, 380, "right: (2,1.5) (2.75,1.5) (3.5,1) (4,0)", { fill: C.purple, anchor: "start", size: 17 });
  out("bezier/fig-subdivide.svg", svg(520, 392, body));
}
{ // fig-global.svg: move one control point: degree-7 Bézier vs clamped cubic B-spline
  FS = 1.1;
  const P0: S.Pt[] = [[0, 0], [1, 2], [2, -0.5], [3, 2.2], [4, 0.2], [5, 2], [6, 0], [7, 1.5]];
  const P1 = P0.map(p => [...p] as S.Pt); P1[4] = [4, 1.6];
  let body = "";
  const curves: [string, (P: S.Pt[]) => P2[]][] = [["one Bézier of degree 7", P => bez(P, 300)], ["cubic B-spline (next unit)", P => curvePts(S.curveFrom(P, 3, "clamped"), 300)]];
  curves.forEach(([t, f], k) => {
    const Pn = panel(20 + k * 490, 40, 450, 220, [-0.4, 7.4], [-0.8, 2.6], { axes: false, title: t });
    const a = f(P0), bq = f(P1);
    body += Pn.s + path(P1.map(Pn.M), { stroke: C.blue, w: 1.3, dash: "5 4" });
    body += path(a.map(Pn.M), { stroke: C.grey, w: 2.5, dash: "7 5" });
    // changed part in orange, unchanged black
    for (let j = 0; j < bq.length - 1; j++) {
      const ch = Math.hypot(a[j][0] - bq[j][0], a[j][1] - bq[j][1]) > 1e-3;
      body += path([Pn.M(bq[j]), Pn.M(bq[j + 1])], { stroke: ch ? C.orange : C.fg, w: 4 });
    }
    P1.forEach((p, i) => (body += dot(...Pn.M(p as P2), i === 4 ? 8 : 5.5, i === 4 ? C.red : C.blue)));
    body += arrow(Pn.M([4, 0.35]), Pn.M([4, 1.4]), C.red, 2);
  });
  body += text(490, 300, "P4 moved up (red). Orange: the part of the curve that changed; dashed grey: before.", { size: 18 });
  out("bezier/fig-global.svg", svg(980, 315, body));
}

// ======================================================================= bsplines/
{ // fig-bumps-sum.svg: a 1D spline as a sum of scaled bumps
  FS = 1.25;
  const c = [1, 2.4, 1.4, 2.8, 2.0, 2.6, 1.2];
  const Pn = panel(70, 20, 460, 300, [-1.5, 7.5], [0, 3.1], { xt: [0, 1, 2, 3, 4, 5, 6], yt: [0, 1, 2, 3], xl: "t", yl: "value" });
  const cols = [C.blue, C.red, C.green, C.orange, C.purple, "#8c564b", C.pink];
  let body = Pn.s;
  const f = (x: number) => c.reduce((s, ci, i) => s + ci * S.bspline3(x - i), 0);
  c.forEach((ci, i) => { body += path(sample(x => ci * S.bspline3(x - i), i - 2, i + 2, 120).map(Pn.M), { stroke: cols[i], w: 2.2, dash: "6 4" }); });
  body += path(sample(f, 1, 5, 300).map(Pn.M), { stroke: C.fg, w: 4.5 });
  c.forEach((ci, i) => (body += dot(Pn.X(i), Pn.Y(ci), 7, cols[i])));
  body += text(Pn.X(3), Pn.Y(2.8) - 14, "P3", { size: 18, fill: C.orange });
  out("bsplines/fig-bumps-sum.svg", svg(560, 410, body));
}
{ // fig-knots.svg: one control polygon, four knot vectors
  FS = 1.25;
  const P: S.Pt[] = [[0, 0], [1, 2], [2.5, 2.3], [3.2, 0.8], [4.5, 1.2], [5.5, 2.4], [6.5, 0.6]];
  const n = P.length, p = 3;
  const triple = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]; // clamped, triple interior knot at 1: a corner at P3
  const items: [string, S.SplineCurve][] = [
    ["open (uniform) knots", S.curveFrom(P, p, "open")],
    ["clamped", S.curveFrom(P, p, "clamped")],
    ["triple knot: a corner", { P: P.slice(), p, knots: triple, nUser: n, kind: "clamped" }],
    ["closed", S.curveFrom(P, p, "closed")],
  ];
  let body = "";
  items.forEach(([t, c], k) => {
    const Pn = panel(10 + (k % 2) * 280, 45 + Math.floor(k / 2) * 215, 260, 150, [-0.4, 6.9], [-0.4, 2.7], { axes: false, title: t, tsize: 18 });
    body += Pn.s + path(P.map(Pn.M), { stroke: C.blue, w: 1.5, dash: "5 4", close: c.kind === "closed" });
    body += path(curvePts(c, 400).map(Pn.M), { w: 3.5, close: c.kind === "closed" });
    P.forEach(q => (body += dot(...Pn.M(q as P2), 5, C.blue)));
  });
  out("bsplines/fig-knots.svg", svg(560, 425, body));
}
{ // fig-coxdeboor.svg: the tower on uniform knots 0..4
  FS = 1.2;
  const K = [0, 1, 2, 3, 4, 5, 6];
  let body = "";
  const titles = ["degree 0: boxes", "degree 1: hats", "degree 2", "degree 3: the cubic"];
  for (let p = 0; p <= 3; p++) {
    const Pn = panel(50 + (p % 2) * 275, 40 + Math.floor(p / 2) * 215, 235, 140, [-0.2, 4.2], [0, 1.15], { xt: [0, 1, 2, 3, 4], yt: p % 2 === 0 ? [0, 1] : [], title: titles[p], tsize: 19 });
    body += Pn.s;
    if (p > 0) {
      // the two lower-degree functions (light) and their ramps (dashed)
      for (const i of [0, 1]) body += path(sample(x => S.basisN(i, p - 1, K, x), -0.2, 4.2, 840).map(Pn.M), { stroke: i ? C.orange : C.green, w: 2 });
      body += path([Pn.M([0, 0]), Pn.M([p, 1])], { stroke: C.green, w: 1.5, dash: "5 4" });
      body += path([Pn.M([1, 1]), Pn.M([p + 1, 0])], { stroke: C.orange, w: 1.5, dash: "5 4" });
    }
    body += path(sample(x => S.basisN(0, p, K, x), -0.2, 4.2, 840).map(Pn.M), { stroke: C.blue, w: 3.5 });
  }
  body += text(290, 478, "blue = green × rising ramp", { size: 18 }) + text(290, 504, "+ orange × falling ramp", { size: 18 });
  out("bsplines/fig-coxdeboor.svg", svg(580, 515, body));
}
{ // fig-lift.svg: NURBS as lift, B-spline, divide (quarter circle)
  FS = 1.2;
  // oblique view: w up, x right, y back (up-right)
  const V = (x: number, y: number, w: number): P2 => [70 + 300 * x + 150 * y, 480 - 300 * w - 75 * y];
  let body = "";
  body += path([V(0, 0, 1), V(1.2, 0, 1), V(1.2, 1.2, 1), V(0, 1.2, 1)], { stroke: "#bbb", w: 1, fill: C.blue, op: 0.07, close: true });
  body += arrow(V(0, 0, 0), V(1.35, 0, 0), C.fg, 1.5) + arrow(V(0, 0, 0), V(0, 1.35, 0), C.fg, 1.5) + arrow(V(0, 0, 0), V(0, 0, 1.3), C.fg, 1.5);
  body += text(V(1.35, 0, 0)[0] + 8, V(1.35, 0, 0)[1] + 7, "x", { anchor: "start" }) + text(V(0, 1.35, 0)[0] + 10, V(0, 1.35, 0)[1] + 4, "y", { anchor: "start" }) + text(V(0, 0, 1.3)[0] - 10, V(0, 0, 1.3)[1] + 10, "w", { anchor: "end" });
  body += text(V(0, 1.2, 1)[0] - 8, V(0, 1.2, 1)[1] + 6, "plane w = 1", { anchor: "end", size: 17, fill: "#555" });
  const s = Math.SQRT1_2;
  const lifted: [number, number, number][] = [[1, 0, 1], [s, s, s], [0, 1, 1]], ctrl: [number, number][] = [[1, 0], [1, 1], [0, 1]];
  const lc = Array.from({ length: 101 }, (_, j) => { const t = j / 100, b = [(1 - t) ** 2, 2 * t * (1 - t), t * t]; return [0, 1, 2].map(d => b.reduce((acc, bi, i) => acc + bi * lifted[i][d], 0)); });
  const m = lc[50];
  body += path([V(0, 0, 0), V(1, 1, 1)], { stroke: C.grey, w: 1.5, dash: "4 4" });
  body += path([V(0, 0, 0), V(m[0] / m[2], m[1] / m[2], 1)], { stroke: C.grey, w: 1.5, dash: "4 4" });
  body += path(lifted.map(q => V(...q)), { stroke: C.orange, w: 1.5, dash: "5 4" });
  body += path(lc.map(q => V(q[0], q[1], q[2])), { stroke: C.orange, w: 3.5 });
  body += path(lc.map(q => V(q[0] / q[2], q[1] / q[2], 1)), { stroke: C.blue, w: 4 });
  lifted.forEach(q => (body += dot(...V(...q), 7, C.orange)));
  ctrl.forEach(q => (body += dot(...V(q[0], q[1], 1), 6, C.blue)));
  body += dot(...V(m[0], m[1], m[2]), 6, C.orange) + dot(...V(m[0] / m[2], m[1] / m[2], 1), 7, C.red);
  const a = V(s, s, s), b = V(m[0], m[1], m[2]), c = V(m[0] / m[2], m[1] / m[2], 1);
  body += text(a[0] + 12, a[1] + 26, "(0.71, 0.71, 0.71)", { size: 17, fill: C.orange, anchor: "start" });
  body += text(b[0] - 14, b[1] + 22, "(0.60, 0.60, 0.85)", { size: 17, fill: C.orange, anchor: "end" });
  body += text(c[0] + 14, c[1] + 2, "(0.707, 0.707)", { size: 17, fill: C.red, anchor: "start" });
  body += text(V(1, 1, 1)[0], V(1, 1, 1)[1] - 14, "(1, 1)", { size: 17, fill: C.blue });
  body += text(220, 528, "orange: lifted points and their B-spline", { size: 17, fill: C.orange, anchor: "start" });
  body += text(220, 556, "blue: divided by w, an exact circle arc", { size: 17, fill: C.blue, anchor: "start" });
  out("bsplines/fig-lift.svg", svg(640, 570, body));
}

// ======================================================================= spline-fit/
const r = rng(3), nv = Array.from({ length: 30 }, () => r.norm()), XS = Array.from({ length: 30 }, (_, i) => 10 * i / 29);
const kink = (x: number) => Math.sin(0.8 * x) + 0.4 * Math.abs(x - 6), YS = XS.map((x, i) => kink(x) + 0.25 * nv[i]);
const rmsTruth = (f: (x: number) => number) => Math.sqrt(XS.reduce((s, x) => s + (f(x) - kink(x)) ** 2, 0) / XS.length);
{ // fig-natural.svg: the natural spline through three points, and the tridiagonal system
  FS = 1.25;
  const s = S.naturalCubic([0, 1, 2], [0, 1, 0]);
  const Pn = panel(70, 20, 300, 240, [-0.1, 2.1], [-0.1, 1.15], { xt: [0, 1, 2], yt: [0, 0.5, 1], xl: "x", yl: "s(x)" });
  let body = Pn.s + path(sample(x => s.at(x), 0, 2).map(Pn.M), { stroke: C.blue, w: 3.5 });
  for (const [x, y] of [[0, 0], [1, 1], [2, 0]]) body += dot(Pn.X(x), Pn.Y(y), 7, C.fg);
  body += dot(Pn.X(0.5), Pn.Y(s.at(0.5)), 7, C.red) + text(Pn.X(1), Pn.Y(0.3), `s(0.5) = ${s.at(0.5).toFixed(4)}`, { size: 18, fill: C.red });
  // matrix pattern
  const n = 9, x0 = 405, y0 = 40, cell = 20;
  body += text(x0 + n * cell / 2, y0 - 12, "the system", { weight: "bold" });
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const v = i === j ? 4 : Math.abs(i - j) === 1 ? 1 : 0;
    body += `<rect x="${x0 + j * cell}" y="${y0 + i * cell}" width="${cell}" height="${cell}" fill="${v === 4 ? C.blue : v ? "#9ecae1" : "#fff"}" stroke="#ddd"/>`;
  }
  body += text(x0 + n * cell / 2, y0 + n * cell + 28, "rows: 1 4 1", { size: 18 }) + text(x0 + n * cell / 2, y0 + n * cell + 52, "(the rest is 0)", { size: 18 });
  out("spline-fit/fig-natural.svg", svg(600, 330, body));
}
{ // fig-bending.svg: natural spline vs parabola, and s''² (bending 6 vs 8)
  FS = 1.25;
  const s = S.naturalCubic([0, 1, 2], [0, 1, 0]), par = (x: number) => 2 * x - x * x;
  const A = panel(80, 30, 400, 170, [-0.1, 2.1], [-0.1, 1.15], { xt: [], yt: [0, 1], yl: "curve", title: "through (0,0), (1,1), (2,0)" });
  const B = panel(80, 260, 400, 170, [-0.1, 2.1], [0, 10], { xt: [0, 1, 2], yt: [0, 4, 9], xl: "x", yl: "s''(x)²" });
  let body = A.s + B.s;
  body += path(sample(par, 0, 2).map(A.M), { stroke: C.orange, w: 3 }) + path(sample(x => s.at(x), 0, 2).map(A.M), { stroke: C.blue, w: 3.5 });
  for (const [x, y] of [[0, 0], [1, 1], [2, 0]]) body += dot(A.X(x), A.Y(y), 6.5, C.fg);
  const ss = (x: number) => 9 * (x <= 1 ? x : 2 - x) ** 2;
  body += path([B.M([0, 0]), B.M([0, 4]), B.M([2, 4]), B.M([2, 0])], { stroke: C.orange, w: 3, fill: C.orange, op: 0.15 });
  body += path([...sample(ss, 0, 2, 200), [2, 0] as P2, [0, 0] as P2].map(B.M), { stroke: C.blue, w: 3.5, fill: C.blue, op: 0.2 });
  body += text(500, 90, "parabola", { fill: C.orange, anchor: "start", size: 18 }) + text(500, 130, "spline", { fill: C.blue, anchor: "start", size: 18 });
  body += text(500, 318, "area 8", { fill: C.orange, anchor: "start", size: 18 }) + text(500, 360, "area 6", { fill: C.blue, anchor: "start", size: 18 });
  out("spline-fit/fig-bending.svg", svg(590, 495, body));
}
{ // fig-approx-m.svg: least squares with 5, 7, 26 control points
  FS = 1.25;
  let body = "";
  [5, 7, 26].forEach((m, k) => {
    const f = S.regressionSpline(XS, YS, m);
    const lab = m === 5 ? "too few" : m === 7 ? "about right" : "too many";
    const looTxt = f.loo > 1e6 ? "> 10⁶" : f.loo.toFixed(3);
    const Pn = panel(70, 40 + k * 165, 440, 115, [-0.2, 10.2], [-1.3, 3.4], { xt: k === 2 ? [0, 5, 10] : [], yt: [0, 2], title: `m = ${m}: ${lab} (LOO ${looTxt})`, tsize: 19 });
    body += Pn.s + path(sample(kink, 0, 10).map(Pn.M), { stroke: C.grey, w: 2, dash: "6 5" });
    body += path(clip(sample(f.at, 0, 10, 600), -1.3, 3.4).map(Pn.M), { stroke: C.blue, w: 3 });
    XS.forEach((x, i) => (body += dot(Pn.X(x), Pn.Y(YS[i]), 4, C.fg)));
    console.log("approx m", m, "loo", f.loo, "rms truth", rmsTruth(f.at));
  });
  body += text(290, 545, "x", {});
  out("spline-fit/fig-approx-m.svg", svg(540, 555, body));
}
{ // fig-lambda.svg: smoothing splines for three λ
  FS = 1.25;
  const Pn = panel(70, 20, 440, 280, [-0.2, 10.2], [-1.3, 3.4], { xt: [0, 5, 10], yt: [-1, 0, 1, 2, 3], xl: "x", yl: "y" });
  let body = Pn.s + path(sample(kink, 0, 10).map(Pn.M), { stroke: C.grey, w: 2, dash: "6 5" });
  const L: [number, string, string][] = [[1e-4, C.orange, "λ = 0.0001: follows the noise"], [0.35, C.blue, "λ = 0.35 (CV minimum)"], [1e4, C.purple, "λ = 10000: a straight line"]];
  L.forEach(([l, col, lab], k) => {
    const f = S.smoothingSpline(XS, YS, l);
    body += path(sample(f.at, 0, 10, 500).map(Pn.M), { stroke: col, w: l === 0.35 ? 4 : 2.5 });
    body += text(20, 400 + 28 * k, lab, { fill: col, anchor: "start", size: 19 });
    console.log("lambda", l, "df", f.df, "loo", f.loo);
  });
  XS.forEach((x, i) => (body += dot(Pn.X(x), Pn.Y(YS[i]), 4.5, C.fg)));
  out("spline-fit/fig-lambda.svg", svg(540, 480, body));
}
{ // fig-filter.svg: frequency response of the discrete smoother
  FS = 1.25;
  const Pn = panel(80, 20, 420, 280, [0, Math.PI], [0, 1.08], { xt: [0, Math.PI / 4, Math.PI / 2, 3 * Math.PI / 4, Math.PI], yt: [0, 0.5, 1], xl: "frequency ω", yl: "gain", fmt: v => (v === 0 ? "0" : v === 1 ? "1" : v === 0.5 ? "0.5" : (["π/4", "π/2", "3π/4", "π"][Math.round(v / (Math.PI / 4)) - 1])) });
  let body = Pn.s + path([Pn.M([0, 0.5]), Pn.M([Math.PI, 0.5])], { stroke: "#aaa", w: 1.5, dash: "3 4" });
  const L: [number, string][] = [[1, C.blue], [16, C.orange], [256, C.purple]];
  L.forEach(([l, col], k) => {
    body += path(sample(w => S.discreteSmoothResponse(w, l), 0, Math.PI).map(Pn.M), { stroke: col, w: 3.5 });
    const wh = Math.acos(1 - 0.5 / Math.sqrt(l));   // λ(2 − 2cos ω)² = 1
    body += dot(Pn.X(wh), Pn.Y(0.5), 6.5, col);
    body += text(Pn.X(Math.PI) - 6, 60 + 26 * k, `λ = ${l}: half at ω = ${wh.toFixed(2)}`, { fill: col, anchor: "end", size: 18 });
  });
  body += path(sample(w => 1 / (1 + (2 - 2 * Math.cos(w))), 0, Math.PI).map(Pn.M), { stroke: C.fg, w: 2, dash: "7 5" });
  body += text(Pn.X(Math.PI) - 6, 60 + 26 * 3, "dashed: first differences, λ = 1", { anchor: "end", size: 18 });
  out("spline-fit/fig-filter.svg", svg(540, 380, body));
}
{ // fig-fourier.svg: square of side 36: spline vs truncated Fourier descriptors (K = 8), one corner
  FS = 1.25;
  const G = 64, polys = [[[14, 14], [50, 14], [50, 50], [14, 50]]] as any;
  const mask = rasterise(polys, G, G, "nonzero"), B = traceBoundary(mask, G, G);
  const pts: S.Pt[] = B.points.map(p => [p[0] + 0.5, p[1] + 0.5] as S.Pt);
  const fit = S.fitClosedSpline(pts, Math.min(80, pts.length), Math.pow(10, -1));
  const len = S.arcLength(fit.curve, 4);
  const F = fourierOutline(pts, 8);
  let fLen = 0; for (let i = 0; i < F.length; i++) { const a = F[i], b = F[(i + 1) % F.length]; fLen += Math.hypot(b[0] - a[0], b[1] - a[1]); }
  console.log("square spline", len, "fourier", fLen);
  // left: the whole outline; right: distance inside the true top side, along that side
  const A = panel(20, 30, 250, 250, [10, 54], [54, 10], { axes: false });
  let body = path([[14, 14], [50, 14], [50, 50], [14, 50]].map(p => A.M(p as P2)), { stroke: C.green, w: 2, dash: "5 4", close: true });
  body += path(F.map(p => A.M(p as P2)), { stroke: C.orange, w: 3, close: true });
  const sp = curvePts(fit.curve, 1600);
  body += path(sp.map(A.M), { stroke: C.blue, w: 3.5, close: true });
  body += `<rect x="${A.X(10.5)}" y="${A.Y(10.5)}" width="${A.X(53.5) - A.X(10.5)}" height="${A.Y(17.5) - A.Y(10.5)}" fill="none" stroke="${C.fg}" stroke-width="1.2" stroke-dasharray="3 3"/>`;
  const Bp = panel(360, 30, 240, 250, [14, 50], [-0.6, 3.2], { xt: [14, 32, 50], yt: [0, 1, 2, 3], xl: "along the top side", yl: "inside by (px)" });
  body += Bp.s + path([Bp.M([14, 0]), Bp.M([50, 0])], { stroke: C.green, w: 2, dash: "5 4" });
  const top = (P: P2[]) => P.filter(p => p[1] < 18).sort((a, b) => a[0] - b[0]).map(p => [p[0], p[1] - 14] as P2);
  body += `<defs><clipPath id="cpf"><rect x="360" y="30" width="240" height="250"/></clipPath></defs><g clip-path="url(#cpf)">`;
  body += path(top(F as P2[]).map(Bp.M), { stroke: C.orange, w: 3 }) + path(top(sp).map(Bp.M), { stroke: C.blue, w: 3.5 }) + `</g>`;
  body += text(20, 362, "true: 144", { fill: C.green, anchor: "start" });
  body += text(20, 390, `spline: ${len.toFixed(1)} (corners rounded, sides straight)`, { fill: C.blue, anchor: "start", size: 18 });
  body += text(20, 418, `Fourier, K = 8: ${fLen.toFixed(1)} (ripples along every side)`, { fill: C.orange, anchor: "start", size: 18 });
  out("spline-fit/fig-fourier.svg", svg(620, 430, body));
}
function fourierOutline(P: S.Pt[], K: number): S.Pt[] {   // as in demos/src/splineoutline/main.ts
  const N = 256, n = P.length, L = [0];
  for (let i = 0; i < n; i++) { const a = P[i], b = P[(i + 1) % n]; L.push(L[i] + Math.hypot(b[0] - a[0], b[1] - a[1])); }
  const re = new Float64Array(N), im = new Float64Array(N);
  let j = 0;
  for (let k = 0; k < N; k++) {
    const s = L[n] * k / N;
    while (j < n - 1 && L[j + 1] < s) j++;
    const a = P[j], b = P[(j + 1) % n], f = (s - L[j]) / (L[j + 1] - L[j] || 1);
    re[k] = a[0] + f * (b[0] - a[0]); im[k] = a[1] + f * (b[1] - a[1]);
  }
  fft1d(re, im);
  for (let k = K + 1; k < N - K; k++) { re[k] = 0; im[k] = 0; }
  fft1d(re, im, true);
  return Array.from({ length: N }, (_, k) => [re[k] / N, im[k] / N] as S.Pt);
}

// ======================================================================= spline-images/
{ // fig-prefilter.svg: without and with the prefilter
  FS = 1.25;
  const f = [0, 0, 1, 3, 2, 2.5, 0.5, 0, 0];
  const c = SI.bsplineCoeffs1d(f);
  console.log("prefilter coeffs", Array.from(c).map(v => v.toFixed(2)).join(" "));
  let body = "";
  const rows: [string, ArrayLike<number>][] = [["no prefilter: weights = samples", f], ["prefilter: weights c (sharper)", c]];
  rows.forEach(([t, w], k) => {
    const Pn = panel(60, 40 + k * 230, 460, 160, [-0.5, 8.5], [-1.2, 4.2], { xt: k ? [0, 2, 4, 6, 8] : [], yt: [0, 2, 4], title: t, tsize: 19 });
    body += Pn.s;
    for (let i = 0; i < f.length; i++) {
      if (Math.abs(w[i]) > 1e-9) body += path(sample(x => w[i] * S.bspline3(x - i), Math.max(-0.5, i - 2), Math.min(8.5, i + 2), 80).map(Pn.M), { stroke: C.purple, w: 1.5, dash: "4 3" });
      body += path([Pn.M([i, 0]), Pn.M([i, w[i]])], { stroke: C.orange, w: 5 });
    }
    body += path(sample(x => SI.bsplineValue1d(w, x), 0, 8, 400).map(Pn.M), { stroke: C.blue, w: 3.5 });
    f.forEach((v, i) => (body += dot(Pn.X(i), Pn.Y(v), 6, C.fg)));
  });
  body += text(290, 500, "x (pixels)");
  body += text(280, 532, "black: samples · orange: weights", { size: 18 }) + text(280, 556, "blue: the spline", { size: 18 });
  out("spline-images/fig-prefilter.svg", svg(560, 568, body));
}
{ // fig-overshoot.svg: a step 0 → 100 through four interpolators
  FS = 1.25;
  const v = new Float64Array(21); for (let i = 10; i < 21; i++) v[i] = 100;
  const g = fromArray(21, 1, v);
  const Pn = panel(80, 20, 440, 300, [6.5, 13.5], [-20, 120], { xt: [7, 8, 9, 10, 11, 12, 13], yt: [0, 50, 100], xl: "x (pixels)", yl: "value" });
  let body = Pn.s;
  for (const yv of [0, 100]) body += path([Pn.M([6.5, yv]), Pn.M([13.5, yv])], { stroke: "#aaa", w: 1.5, dash: "3 4" });
  const ms: [SI.SplineInterp, string, string][] = [["bilinear", "bilinear", C.green], ["keys", "cubic convolution", C.orange], ["bspline", "cubic B-spline", C.blue]];
  ms.forEach(([m, lab, col], k) => {
    const s = SI.sampler(g, m), pts = sample(x => s(x, 0), 6.5, 13.5, 700);
    const lo = Math.min(...pts.map(p => p[1])), hi = Math.max(...pts.map(p => p[1]));
    body += path(pts.map(Pn.M), { stroke: col, w: m === "bspline" ? 4 : 2.8 });
    body += text(20, 405 + 28 * k, `${lab}: ${lo.toFixed(0).replace("-", "−")} … ${hi.toFixed(0)}`, { fill: col, anchor: "start", size: 19 });
    console.log("overshoot", m, lo, hi);
  });
  for (let i = 7; i <= 13; i++) body += dot(Pn.X(i), Pn.Y(v[i]), 6, C.fg);
  out("spline-images/fig-overshoot.svg", svg(540, 485, body));
}
// ---- raster data for figs2_raster.py
{
  const cellJson = JSON.parse(fs.readFileSync(RAW + "/cell256.json", "utf8")) as number[];
  const cell256 = fromArray(256, 256, Float64Array.from(cellJson));
  const cell = resize(cell256, 128, 128, "area");
  const N = 128, PATCH = 16, ZOOM = 8, px = 56, py = 56;
  const zoom: Record<string, number[]> = {};
  for (const m of ["nearest", "bilinear", "bspline"] as SI.SplineInterp[]) {
    const s = SI.sampler(cell, m), o: number[] = [];
    for (let v = 0; v < N; v++) for (let u = 0; u < N; u++) o.push(s(px + (u + 0.5) / ZOOM - 0.5, py + (v + 0.5) / ZOOM - 0.5));
    zoom[m] = o;
  }
  const rot: Record<string, { img: number[]; rms: number }> = {};
  for (const m of ["nearest", "bilinear", "keys", "bspline"] as SI.SplineInterp[]) {
    let cur: Grey = cell;
    for (let k = 0; k < 10; k++) cur = SI.rotateWith(cur, 36 * Math.PI / 180, m);
    let ss = 0, cnt = 0;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const i = y * N + x, d = cur.data[i] - cell.data[i]; if ((x - 63.5) ** 2 + (y - 63.5) ** 2 < (0.42 * N) ** 2) { ss += d * d; cnt++; } }
    rot[m] = { img: Array.from(cur.data), rms: Math.sqrt(ss / cnt) };
    console.log("rotate", m, rot[m].rms.toFixed(2));
  }
  // FFD fold preset: checkerboard 256, h = 32, (4,4) +44 px and (5,4) −44 px in x
  const W = 256, g = SI.ffdGrid(W, W, 32, 3);
  g.dx[SI.ffdIndex(g, 4, 4)] = 44; g.dx[SI.ffdIndex(g, 5, 4)] = -44;
  const warped = SI.ffdWarp(checkerboard(W, W, 16, 40, 215), g, "bilinear");
  const jrgb: number[] = []; let minJ = Infinity, folds = 0;
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
    const d = SI.ffdJacobianDet(g, x, y); minJ = Math.min(minJ, d);
    if (d <= 0) { folds++; jrgb.push(0, 0, 0); } else { const c = diverging(0.5 + 0.5 * Math.max(-1, Math.min(1, 3 * Math.log2(d)))); jrgb.push(c[0], c[1], c[2]); }
  }
  console.log("fold minJ", minJ.toFixed(3), "folds", folds);
  // deformed grid lines every 32 px (forward map)
  const lines: number[][][] = [];
  for (let c0 = 0; c0 <= W - 1; c0 += 32) {
    const a: number[][] = [], b: number[][] = [];
    for (let s = 0; s <= W - 1; s += 1) { const [u1, v1] = SI.ffdAt(g, c0, s), [u2, v2] = SI.ffdAt(g, s, c0); a.push([c0 + u1, s + v1]); b.push([s + u2, c0 + v2]); }
    lines.push(a, b);
  }
  const handles = [[4 * 32 + 44, 4 * 32], [5 * 32 - 44, 4 * 32]];
  fs.writeFileSync(RAW + "/raster.json", JSON.stringify({ cell: Array.from(cell.data), px, py, PATCH, zoom, rot, fold: { warped: Array.from(warped.data), jrgb, minJ, folds, lines, handles } }));
  console.log("wrote", RAW + "/raster.json");
}
