// Static figures (SVG) for the splines day: bezier/, bsplines/, spline-fit/, spline-images/.
// Computed with the demos' own libraries (demos/src/lib/spline.ts, splineimage.ts). Run from
// the repository root:
//   node_modules/.bin/esbuild tools/figs/splines/figs.ts --bundle --platform=node --log-level=warning | node
import * as fs from "fs";
import * as S from "../../../demos/src/lib/spline";
import * as SI from "../../../demos/src/lib/splineimage";
import { rng } from "../../../demos/src/lib/random";
import { rasterise, ellipsePoly } from "../../../demos/src/lib/shapecanvas";
import { traceBoundary } from "../../../demos/src/splineoutline/boundary";
import { fromArray } from "../../../demos/src/lib/imageops";

const C = { fg: "#222", grey: "#888", light: "#ddd", blue: "#1f77b4", red: "#d62728", green: "#2ca02c", orange: "#ff7f0e", purple: "#9467bd", pink: "#e377c2" };
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
type P2 = [number, number];
function svg(w: number, h: number, body: string): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" font-family="Helvetica, Arial, sans-serif" font-size="20">\n<rect width="${w}" height="${h}" fill="#fff"/>\n${body}\n</svg>\n`;
}
const text = (x: number, y: number, s: string, o: { size?: number; anchor?: string; fill?: string; weight?: string } = {}) =>
  `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" font-size="${o.size ?? 20}" text-anchor="${o.anchor ?? "middle"}" fill="${o.fill ?? C.fg}"${o.weight ? ` font-weight="${o.weight}"` : ""}>${esc(s)}</text>`;
const path = (pts: P2[], o: { stroke?: string; w?: number; fill?: string; op?: number; close?: boolean; dash?: string } = {}) =>
  `<path d="M${pts.map(p => p[0].toFixed(1) + "," + p[1].toFixed(1)).join("L")}${o.close ? "Z" : ""}" fill="${o.fill ?? "none"}" fill-opacity="${o.op ?? 1}" stroke="${o.stroke ?? C.fg}" stroke-width="${o.w ?? 2.5}" stroke-linejoin="round"${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}/>`;
const dot = (x: number, y: number, r: number, fill: string, stroke = "#fff") => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>`;

/** A panel with a linear data → pixel map (y up) and optional axes. */
function panel(x0: number, y0: number, w: number, h: number, xd: [number, number], yd: [number, number], o: { axes?: boolean; xt?: number[]; yt?: number[]; xl?: string; yl?: string; fmt?: (v: number) => string; title?: string } = {}) {
  const X = (v: number) => x0 + (v - xd[0]) / (xd[1] - xd[0]) * w, Y = (v: number) => y0 + h - (v - yd[0]) / (yd[1] - yd[0]) * h;
  const M = (p: P2): P2 => [X(p[0]), Y(p[1])];
  let s = "";
  const fmt = o.fmt ?? ((v: number) => String(v));
  if (o.axes !== false) {
    for (const t of o.xt ?? []) s += `<line x1="${X(t)}" x2="${X(t)}" y1="${y0}" y2="${y0 + h}" stroke="#eee"/>` + text(X(t), y0 + h + 24, fmt(t), { size: 18 });
    for (const t of o.yt ?? []) s += `<line x1="${x0}" x2="${x0 + w}" y1="${Y(t)}" y2="${Y(t)}" stroke="#eee"/>` + text(x0 - 8, Y(t) + 6, fmt(t), { size: 18, anchor: "end" });
    s += path([[x0, y0], [x0, y0 + h], [x0 + w, y0 + h]], { w: 1.5 });
    if (o.xl) s += text(x0 + w / 2, y0 + h + 50, o.xl);
    if (o.yl) s += `<text x="${x0 - 48}" y="${y0 + h / 2}" font-size="20" text-anchor="middle" transform="rotate(-90 ${x0 - 48} ${y0 + h / 2})">${esc(o.yl)}</text>`;
  }
  if (o.title) s += text(x0 + w / 2, y0 - 10, o.title, { weight: "bold" });
  return { X, Y, M, s };
}
const sample = (f: (x: number) => number, a: number, b: number, n = 300): P2[] => Array.from({ length: n + 1 }, (_, k) => { const x = a + (b - a) * k / n; return [x, f(x)] as P2; });
const clip = (pts: P2[], lo: number, hi: number): P2[] => pts.map(([x, y]) => [x, Math.max(lo, Math.min(hi, y))] as P2);
const out = (file: string, s: string) => { fs.writeFileSync(file, s); console.log("wrote", file); };

// ======================================================================= bezier/
{ // fig-runge.svg: 11 equispaced samples of the bump: polynomial, spline
  const f = (x: number) => 1 / (1 + 25 * x * x);
  let body = "";
  [5, 11].forEach((n, k) => {
    const P = panel(70 + k * 470, 40, 390, 260, [-1.05, 1.05], [-0.6, 1.6], { xt: [-1, 0, 1], yt: [0, 1], xl: "x", title: `${n} equispaced samples` });
    body += P.s;
    const xs = S.equispacedNodes(n), ys = xs.map(f), poly = S.polyInterp(xs, ys), sp = S.naturalCubic(xs, ys);
    body += path(sample(f, -1, 1).map(P.M), { stroke: C.grey, dash: "6 5", w: 2 });
    body += path(clip(sample(poly, -1, 1, 600), -0.6, 1.6).map(P.M), { stroke: C.red, w: 3 });
    body += path(sample(x => sp.at(x), -1, 1).map(P.M), { stroke: C.blue, w: 3 });
    xs.forEach((x, i) => (body += dot(P.X(x), P.Y(ys[i]), 6, C.fg)));
  });
  body += text(330, 375, "one polynomial", { fill: C.red, anchor: "end" }) + text(360, 375, "cubic spline", { fill: C.blue, anchor: "start" }) + text(560, 375, "true function", { fill: C.grey, anchor: "start" });
  out("bezier/fig-runge.svg", svg(940, 390, body));
}
{ // fig-three-tasks.svg: through, near, controlled by the same points
  const pts: P2[] = [[0, 0.2], [1, 1.4], [2, 0.9], [3, 2.0], [4, 1.1], [5, 1.8], [6, 0.6]];
  let body = "";
  const titles = ["through the points", "near the points", "controlled by the points"];
  titles.forEach((t, k) => {
    const P = panel(20 + k * 310, 50, 280, 200, [-0.3, 6.3], [-0.3, 2.5], { axes: false, title: t });
    let curve: P2[];
    if (k === 0) { const s = S.naturalCubic(pts.map(p => p[0]), pts.map(p => p[1])); curve = sample(x => s.at(x), 0, 6); }
    else if (k === 1) { const r = S.regressionSpline(pts.map(p => p[0]), pts.map(p => p[1]), 4); curve = sample(r.at, 0, 6); }
    else { const c = S.curveFrom(pts, 3, "clamped"); const [a, b] = S.curveDomain(c); curve = Array.from({ length: 201 }, (_, j) => S.curvePoint(c, a + (b - a) * Math.min(j / 200, 1 - 1e-12))); body += path(pts.map(P.M), { stroke: C.blue, w: 1.5, dash: "6 4" }); }
    body += path(curve.map(P.M), { stroke: k === 2 ? C.fg : C.blue, w: 3.5 });
    pts.forEach(p => (body += dot(...P.M(p), 6.5, k === 2 ? C.blue : C.fg)));
  });
  out("bezier/fig-three-tasks.svg", svg(950, 270, body));
}
{ // fig-casteljau.svg: the worked construction at t = 1/2
  const P: S.Pt[] = [[0, 0], [1, 2], [3, 2], [4, 0]], L = S.deCasteljau(P, 0.5);
  const Pn = panel(60, 30, 520, 280, [-0.4, 4.4], [-0.3, 2.3], { xt: [0, 1, 2, 3, 4], yt: [0, 1, 2] });
  let body = Pn.s;
  body += path(Array.from({ length: 101 }, (_, k) => S.bezierPoint(P, k / 100)).map(Pn.M), { w: 4 });
  const cols = [C.blue, C.green, C.orange, C.red];
  L.forEach((lev, k) => {
    if (lev.length > 1) body += path(lev.map(Pn.M), { stroke: cols[k], w: k ? 2.5 : 2, dash: k ? "" : "7 5" });
    lev.forEach(p => (body += dot(...Pn.M(p), k === 3 ? 9 : 7, cols[k])));
  });
  const lab = (p: P2, s: string, dx: number, dy: number, col: string) => text(Pn.X(p[0]) + dx, Pn.Y(p[1]) + dy, s, { size: 18, fill: col });
  body += lab([0, 0], "P0 (0,0)", 0, 28, C.blue) + lab([1, 2], "P1 (1,2)", 0, -14, C.blue) + lab([3, 2], "P2 (3,2)", 0, -14, C.blue) + lab([4, 0], "P3 (4,0)", 0, 28, C.blue);
  body += lab([0.5, 1], "(0.5, 1)", -48, 0, C.green) + lab([2, 2], "(2, 2)", 0, 26, C.green) + lab([3.5, 1], "(3.5, 1)", 50, 0, C.green);
  body += lab([1.25, 1.5], "(1.25, 1.5)", -36, 26, C.orange) + lab([2.75, 1.5], "(2.75, 1.5)", 40, 26, C.orange) + lab([2, 1.5], "B(½) = (2, 1.5)", 0, -18, C.red);
  out("bezier/fig-casteljau.svg", svg(620, 360, body));
}
{ // fig-joins.svg: C0, G1, C1 joins with curvature combs
  const P: S.Pt[] = [[0, 0], [1, 2], [3, 2], [4, 0]];
  const Qs: [string, S.Pt[]][] = [["C⁰: shared end point, a kink", [[4, 0], [4.5, 2], [6.5, 2.5], [8, 1]]], ["G¹: same direction, other speed", [[4, 0], [4.4, -0.8], [7, -1.5], [8, 1]]], ["C¹: mirrored handle (5, −2)", [[4, 0], [5, -2], [7, -1.5], [8, 1]]]];
  let body = "";
  Qs.forEach(([t, Q], k) => {
    const Pn = panel(10 + k * 330, 40, 310, 230, [-0.5, 8.5], [-2.6, 3.4], { axes: false, title: t });
    let kmax = 0;
    for (const B of [P, Q]) for (let j = 0; j <= 60; j++) { const [a, b] = S.bezierDerivs(B, j / 60); kmax = Math.max(kmax, Math.min(4, Math.abs(S.curvature(a, b)))); }
    const sc = 1.3 / kmax;
    for (const B of [P, Q]) {
      const tips: P2[] = [];
      for (let j = 0; j <= 50; j++) {
        const u = j / 50, p = S.bezierPoint(B, u), [d1, d2] = S.bezierDerivs(B, u), kap = S.curvature(d1, d2), L = Math.hypot(...d1);
        const tip: P2 = [p[0] + sc * kap * d1[1] / L, p[1] - sc * kap * d1[0] / L];
        tips.push(tip); body += path([Pn.M(p), Pn.M(tip)], { stroke: C.pink, w: 1 });
      }
      body += path(tips.map(Pn.M), { stroke: C.pink, w: 1.8 });
      body += path(B.map(Pn.M), { stroke: B === P ? C.blue : C.purple, w: 1.3, dash: "5 4" });
      body += path(Array.from({ length: 101 }, (_, j) => S.bezierPoint(B, j / 100)).map(Pn.M), { stroke: B === P ? C.fg : C.purple, w: 3.5 });
      B.forEach(p => (body += dot(...Pn.M(p), 5, B === P ? C.blue : C.purple)));
    }
  });
  body += text(500, 300, "pink: curvature comb (length ∝ curvature, drawn along the normal)", { size: 18, fill: C.pink });
  out("bezier/fig-joins.svg", svg(1000, 315, body));
}

// ======================================================================= bsplines/
{ // fig-boxes.svg: β0 … β3 as repeated convolution of boxes
  let body = "";
  [0, 1, 2, 3].forEach(p => {
    const Pn = panel(60 + p * 235, 40, 200, 170, [-2.2, 2.2], [0, 1.1], { xt: [-2, -1, 0, 1, 2], yt: p === 0 ? [0, 0.5, 1] : [], title: p === 0 ? "degree 0: box" : p === 1 ? "1: box * box" : p === 2 ? "2: three boxes" : "3: four boxes" });
    body += Pn.s;
    const pts = sample(x => S.bsplineKernel(p, x), -2.2, 2.2, 880);
    if (p === 0) { body += path([[-2.2, 0], [-0.5, 0], [-0.5, 1], [0.5, 1], [0.5, 0], [2.2, 0]].map(q => Pn.M(q as P2)), { stroke: C.blue, w: 3.5 }); }
    else body += path(pts.map(Pn.M), { stroke: C.blue, w: 3.5 });
    if (p === 3) for (const x of [-1, 0, 1]) { body += dot(Pn.X(x), Pn.Y(S.bspline3(x)), 6, C.red); body += text(Pn.X(x), Pn.Y(S.bspline3(x)) - 12, x === 0 ? "4/6" : "1/6", { size: 18, fill: C.red }); }
  });
  out("bsplines/fig-boxes.svg", svg(1000, 260, body));
}
{ // fig-basis.svg: uniform and clamped cubic bases with their sum
  let body = "";
  const kinds: [string, number[]][] = [["uniform knots 0, 1, …, 10", Array.from({ length: 11 }, (_, i) => i)], ["clamped: end knots repeated 4 times", [0, 0, 0, 0, 2, 4, 6, 8, 10, 10, 10, 10]]];
  const cols = [C.blue, C.red, C.green, C.orange, C.purple, "#8c564b", C.pink, "#17becf", "#bcbd22"];
  kinds.forEach(([t, K], k) => {
    const Pn = panel(70 + k * 480, 40, 400, 220, [-0.2, 10.2], [0, 1.1], { xt: [0, 2, 4, 6, 8, 10], yt: [0, 0.5, 1], xl: "t", title: t });
    body += Pn.s;
    const nb = K.length - 4, [a, b] = S.knotDomain(K, 3);
    body += `<rect x="${Pn.X(a)}" y="40" width="${Pn.X(b) - Pn.X(a)}" height="220" fill="${C.green}" fill-opacity=".07"/>`;
    const xs = Array.from({ length: 501 }, (_, j) => -0.2 + 10.4 * j / 500);
    const sum = xs.map(() => 0);
    for (let i = 0; i < nb; i++) { const ys = xs.map(x => S.basisN(i, 3, K, x)); ys.forEach((v, j) => (sum[j] += v)); body += path(xs.map((x, j) => Pn.M([x, ys[j]])), { stroke: cols[i % cols.length], w: 2.5 }); }
    body += path(xs.map((x, j) => Pn.M([x, sum[j]])), { stroke: C.fg, w: 2.5, dash: "7 5" });
    for (const v of K) body += `<path d="M${Pn.X(v)},${Pn.Y(0) + 4}l-6,10h12z" fill="${C.fg}"/>`;
  });
  body += text(500, 340, "dashed: the sum of all basis functions, 1 on the valid range (green)", { size: 18 });
  out("bsplines/fig-basis.svg", svg(960, 352, body));
}
{ // fig-corner-cut.svg: closed cubic B-spline on a square
  const sq: S.Pt[] = [[0, 0], [4, 0], [4, 4], [0, 4]], c = S.curveFrom(sq, 3, "closed");
  const Pn = panel(60, 20, 300, 300, [-0.5, 4.5], [-0.5, 4.5], { xt: [0, 2, 4], yt: [0, 2, 4] });
  let body = Pn.s + path(sq.map(Pn.M), { stroke: C.blue, w: 2, dash: "7 5", close: true });
  body += path(Array.from({ length: 401 }, (_, j) => S.curvePoint(c, 3 + 4 * Math.min(j / 400, 1 - 1e-12))).map(Pn.M), { w: 4, close: true });
  sq.forEach(p => (body += dot(...Pn.M(p), 7, C.blue)));
  body += dot(...Pn.M([10 / 3, 2 / 3]), 7, C.red) + text(Pn.X(10 / 3) - 10, Pn.Y(2 / 3) - 14, "(10/3, 2/3)", { size: 18, fill: C.red, anchor: "end" });
  out("bsplines/fig-corner-cut.svg", svg(400, 370, body));
}
{ // fig-quarter.svg: quadratic with weight 1 vs 1/√2 against the circle
  const Pn = panel(60, 20, 320, 320, [-0.1, 1.15], [-0.1, 1.15], { xt: [0, 0.5, 1], yt: [0, 0.5, 1] });
  let body = Pn.s;
  body += path(sample(x => Math.sqrt(Math.max(0, 1 - x * x)), 0, 1, 200).map(Pn.M), { stroke: C.grey, w: 7, op: 0.4 });
  for (const [w, col] of [[1, C.orange], [Math.SQRT1_2, C.blue]] as [number, string][]) {
    const c = S.quarterCircle(w);
    body += path(Array.from({ length: 201 }, (_, j) => S.curvePoint(c, Math.min(j / 200, 1 - 1e-12))).map(Pn.M), { stroke: col, w: 3 });
    const m = S.curvePoint(c, 0.5);
    body += dot(...Pn.M(m), 6, col);
  }
  body += path([[1, 0], [1, 1], [0, 1]].map(p => Pn.M(p as P2)), { stroke: C.fg, w: 1.5, dash: "6 4" });
  for (const p of [[1, 0], [1, 1], [0, 1]] as P2[]) body += dot(...Pn.M(p), 6, C.fg);
  body += text(Pn.X(0.75) + 12, Pn.Y(0.75) - 6, "w = 1: (0.75, 0.75)", { size: 17, fill: C.orange, anchor: "start" });
  body += text(Pn.X(0.7071) - 10, Pn.Y(0.7071) + 22, "w = 1/√2: (0.707, 0.707)", { size: 17, fill: C.blue, anchor: "end" });
  out("bsplines/fig-quarter.svg", svg(420, 390, body));
}

// ======================================================================= spline-fit/
const r = rng(3), nv = Array.from({ length: 30 }, () => r.norm()), XS = Array.from({ length: 30 }, (_, i) => 10 * i / 29);
const kink = (x: number) => Math.sin(0.8 * x) + 0.4 * Math.abs(x - 6), YS = XS.map((x, i) => kink(x) + 0.25 * nv[i]);
{ // fig-three-fits.svg
  let body = "";
  const fits: [string, (x: number) => number][] = [
    ["interpolate (natural spline)", S.naturalCubic(XS, YS).at],
    ["approximate (m = 7)", S.regressionSpline(XS, YS, 7).at],
    ["smooth (λ = 0.35)", S.smoothingSpline(XS, YS, 0.35).at],
  ];
  fits.forEach(([t, f], k) => {
    const Pn = panel(60 + k * 320, 40, 280, 200, [-0.2, 10.2], [-1.3, 3.4], { xt: [0, 5, 10], yt: k === 0 ? [-1, 0, 1, 2, 3] : [], title: t });
    body += Pn.s + path(sample(kink, 0, 10).map(Pn.M), { stroke: C.grey, w: 2, dash: "6 5" });
    body += path(sample(f, 0, 10, 400).map(Pn.M), { stroke: C.blue, w: 3 });
    XS.forEach((x, i) => (body += dot(Pn.X(x), Pn.Y(YS[i]), 4.5, C.fg)));
  });
  out("spline-fit/fig-three-fits.svg", svg(1000, 290, body));
}
{ // fig-cv.svg: LOO and df against λ
  const lams = Array.from({ length: 81 }, (_, k) => Math.pow(10, -4 + 8 * k / 80));
  const fits = lams.map(l => S.smoothingSpline(XS, YS, l));
  const lx = lams.map(Math.log10);
  const A = panel(80, 30, 360, 220, [-4, 4], [0, 0.5], { xt: [-4, -2, 0, 2, 4], yt: [0, 0.25, 0.5], xl: "log₁₀ λ", yl: "LOO error" });
  const B = panel(570, 30, 360, 220, [-4, 4], [0, 31], { xt: [-4, -2, 0, 2, 4], yt: [0, 10, 20, 30], xl: "log₁₀ λ", yl: "df = tr H" });
  let body = A.s + B.s;
  body += path(lx.map((x, i) => A.M([x, Math.min(0.5, fits[i].loo)])), { stroke: C.red, w: 3 });
  body += path(lx.map((x, i) => B.M([x, fits[i].df])), { stroke: C.purple, w: 3 });
  const im = fits.map(f => f.loo).indexOf(Math.min(...fits.map(f => f.loo)));
  body += `<line x1="${A.X(lx[im])}" x2="${A.X(lx[im])}" y1="30" y2="250" stroke="${C.green}" stroke-width="2" stroke-dasharray="5 4"/>` + text(A.X(lx[im]) + 6, 54, `minimum, λ ≈ ${lams[im].toPrecision(2)}`, { size: 17, anchor: "start", fill: C.green });
  body += text(B.X(-3.6), B.Y(29.5) + 22, "λ → 0: n = 30", { size: 17, anchor: "start" }) + text(B.X(3.8), B.Y(2) - 10, "λ → ∞: 2 (a line)", { size: 17, anchor: "end" });
  out("spline-fit/fig-cv.svg", svg(960, 320, body));
}
{ // fig-outline.svg: staircase, spline, true circle (disc r = 10)
  const G = 32, mask = rasterise([ellipsePoly(16, 16, 10, 10, 400)], G, G), B = traceBoundary(mask, G, G);
  const fit = S.fitClosedSpline(B.points, 40, Math.pow(10, 0.5));
  const Pn = panel(20, 20, 330, 330, [4.5, 26.5], [26.5, 4.5], { axes: false });
  let body = "";
  for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) if (mask[y * G + x]) { const [px, py] = Pn.M([x - 0.5, y - 0.5]); body += `<rect x="${px.toFixed(1)}" y="${(py).toFixed(1)}" width="${(Pn.X(1) - Pn.X(0)).toFixed(1)}" height="${(Pn.Y(0) - Pn.Y(1)).toFixed(1)}" fill="#c6d9f1" stroke="#fff" stroke-width=".6"/>`; }
  body += path(B.corners.map(Pn.M), { stroke: C.grey, w: 2, close: true });
  body += path(Array.from({ length: 401 }, (_, j) => [15.5 + 10 * Math.cos(2 * Math.PI * j / 400), 15.5 + 10 * Math.sin(2 * Math.PI * j / 400)] as P2).map(Pn.M), { stroke: C.green, w: 2, dash: "5 4" });
  const [a, b] = S.curveDomain(fit.curve);
  body += path(Array.from({ length: 401 }, (_, j) => S.curvePoint(fit.curve, a + (b - a) * Math.min(j / 400, 1 - 1e-12))).map(Pn.M), { stroke: C.blue, w: 3.5, close: true });
  const L = S.arcLength(fit.curve, 4);
  body += text(400, 120, `edge count: ${B.edges}`, { anchor: "start", fill: C.grey }) + text(400, 150, `chain code: ${B.chain.toFixed(1)}`, { anchor: "start" }) + text(400, 180, `smoothing spline: ${L.toFixed(1)}`, { anchor: "start", fill: C.blue }) + text(400, 210, `true (2π·10): ${(20 * Math.PI).toFixed(1)}`, { anchor: "start", fill: C.green });
  out("spline-fit/fig-outline.svg", svg(660, 370, body));
}

// ======================================================================= spline-images/
{ // fig-impulse.svg: an impulse 0 0 1 0 0 through each interpolator
  const f = new Float64Array(21); f[10] = 1;
  const g = fromArray(21, 1, f);
  const ms: [SI.SplineInterp, string, string][] = [["nearest", "nearest", C.grey], ["bilinear", "bilinear", C.green], ["keys", "cubic convolution", C.orange], ["bspline-nopre", "B-spline, no prefilter", C.purple], ["bspline", "cubic B-spline", C.blue]];
  const Pn = panel(80, 30, 560, 280, [-3.2, 3.2], [-0.25, 1.1], { xt: [-3, -2, -1, 0, 1, 2, 3], yt: [0, 0.5, 1], xl: "x (pixels)" });
  let body = Pn.s + `<line x1="${Pn.X(-3.2)}" x2="${Pn.X(3.2)}" y1="${Pn.Y(0)}" y2="${Pn.Y(0)}" stroke="#bbb"/>`;
  ms.forEach(([m, lab, col], k) => {
    const s = SI.sampler(g, m);
    body += path(sample(x => s(10 + x, 0), -3.2, 3.2, 640).map(Pn.M), { stroke: col, w: m === "bspline" ? 4 : 2.5 });
    body += text(660, 70 + 30 * k, lab, { size: 18, fill: col, anchor: "start" });
  });
  for (let x = -3; x <= 3; x++) body += dot(Pn.X(x), Pn.Y(x === 0 ? 1 : 0), 6, C.fg);
  out("spline-images/fig-impulse.svg", svg(900, 370, body));
}
{ // fig-freq.svg: frequency responses
  const ms: [SI.SplineInterp, string, string][] = [["nearest", "nearest", C.grey], ["bilinear", "bilinear", C.green], ["keys", "cubic convolution", C.orange], ["bspline", "cubic B-spline", C.blue]];
  const Pn = panel(80, 30, 560, 260, [0, 2 * Math.PI], [-0.1, 1.15], { xt: [0, Math.PI, 2 * Math.PI], yt: [0, 0.5, 1], xl: "frequency ω (π = Nyquist)", fmt: v => (v === 0 ? "0" : v < 4 ? "π" : v < 7 ? "2π" : String(v)) });
  let body = Pn.s + path([[0, 1], [Math.PI, 1], [Math.PI, 0], [2 * Math.PI, 0]].map(p => Pn.M(p as P2)), { stroke: C.fg, w: 1.5, dash: "5 4" });
  ms.forEach(([m, lab, col], k) => {
    body += path(sample(w => SI.interpResponse(m, w), 0, 2 * Math.PI, 300).map(Pn.M), { stroke: col, w: m === "bspline" ? 4 : 2.5 });
    body += text(660, 70 + 30 * k, lab, { size: 18, fill: col, anchor: "start" });
  });
  body += text(660, 70 + 30 * 4, "ideal (dashed)", { size: 18, anchor: "start" });
  out("spline-images/fig-freq.svg", svg(880, 350, body));
}
{ // fig-ffd.svg: one control point dragged by 9 px (h = 32): the deformed grid
  const g = SI.ffdGrid(161, 161, 32); g.dx[SI.ffdIndex(g, 2, 2)] = 9 * 2.5; // exaggerated ×2.5 for visibility
  const Pn = panel(30, 20, 360, 360, [-8, 168], [168, -8], { axes: false });
  let body = "";
  for (let c = 0; c <= 160; c += 16) {
    const v: P2[] = [], hh: P2[] = [];
    for (let s = 0; s <= 160; s += 2) { const [u1, w1] = SI.ffdAt(g, c, s), [u2, w2] = SI.ffdAt(g, s, c); v.push([c + u1, s + w1]); hh.push([s + u2, c + w2]); }
    body += path(v.map(Pn.M), { stroke: c % 32 ? "#bbb" : "#666", w: c % 32 ? 1 : 1.8 }) + path(hh.map(Pn.M), { stroke: c % 32 ? "#bbb" : "#666", w: c % 32 ? 1 : 1.8 });
  }
  for (let l = 0; l <= 5; l++) for (let k = 0; k <= 5; k++) { const i = SI.ffdIndex(g, k, l); body += dot(...Pn.M([k * 32 + g.dx[i], l * 32 + g.dy[i]]), k === 2 && l === 2 ? 8 : 5, k === 2 && l === 2 ? C.red : "#fff", C.red); }
  body += path([Pn.M([64, 64]), Pn.M([64 + 22.5, 64])], { stroke: C.red, w: 2, dash: "4 3" });
  out("spline-images/fig-ffd.svg", svg(420, 400, body));
}
