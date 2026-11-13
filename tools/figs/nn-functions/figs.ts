// Figures for nn-functions/ (SVG). Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/nn-functions/figs.ts --bundle --platform=node --log-level=warning | node
import { svg, text, line, rect, circle, path, out, C, patternPanel, curvePanel, greyImage } from "../point-patterns/kit";
import * as PP from "../../../demos/src/lib/pointpattern";
import { rng } from "../../../demos/src/lib/random";
import { makePattern, DEFAULT_SPEC, makeWindow, Spec } from "../../../demos/src/csrplay/source";

const D = "nn-functions/";
const sq = makeWindow("square");
const pat = (s: Partial<Spec>) => makePattern({ ...DEFAULT_SPEC, ...s });
const R = PP.linspace(0, 30, 61);
const COL = [C.fp, C.grey, C.blue];

const cl = pat({ model: "thomas", k: 15, mu: 10, sigma: 5, seed: 2 }), n = cl.x.length;
const arche: [PP.Points, string][] = [[cl, "clustered"], [PP.binomialProcess(rng(21), n, sq), "random"], [PP.ssi(rng(22), n, 11, sq), "regular"]];
const rho = n / sq.area;

function nnArrows(p: PP.Points) {
  const { idx } = PP.nnDistances(p);
  return (X: (v: number) => number, Y: (v: number) => number) => {
    let s = "";
    for (let i = 0; i < p.x.length; i++) if (idx[i] >= 0) s += line(X(p.x[i]), Y(p.y[i]), X(p.x[idx[i]]), Y(p.y[idx[i]]), C.orange, 1.6);
    return s;
  };
}

// ---- G of the archetypes
{
  let b = "";
  arche.forEach(([p, t], k) => { b += patternPanel(30 + k * 230, 40, 200, sq, p, { r: 2.6, title: t, under: nnArrows(p) }); });
  const G = arche.map(([p]) => PP.Gest(p, sq, R, "border"));
  b += curvePanel(790, 30, 330, 210, [0, 30], [0, 1], [0, 10, 20, 30], [0, 0.5, 1], "r (µm)", "G(r)",
    [{ xs: R, ys: Array.from(R, r => PP.csrG(r, rho)), color: "#000", dash: "7 5", w: 2.5, label: "CSR" }, ...G.map((g, k) => ({ xs: R, ys: g, color: COL[k], label: arche[k][1] }))], { legendAt: [1120, 135] });
  out(D + "fig-g-archetypes.svg", svg(1140, 310, b));
  console.log("G at 6 µm:", G.map(g => g[12].toFixed(3)), "CSR", PP.csrG(6, rho).toFixed(3), "n", n);
}

// ---- F: distance map with test locations, and F curves
{
  const p = cl, N = 100;
  const loc = PP.makePoints(Array.from({ length: N * N }, (_, k) => ((k % N) + 0.5) * 2), Array.from({ length: N * N }, (_, k) => (Math.floor(k / N) + 0.5) * 2));
  const e = PP.emptySpaceDistances(loc, p), mx = Math.max(...e);
  const u = new Uint8Array(N * N);
  for (let i = 0; i < N * N; i++) u[i] = Math.round(255 * e[i] / mx);
  let b = greyImage(u, N, 0, 0, N, N, 30, 40, 3, 0, 255) + text(180, 30, "distance to the nearest point");
  const g = PP.gridLocations(sq, 25), ge = PP.emptySpaceDistances(g, p);
  for (let k = 0; k < g.x.length; k++) {
    let best = Infinity, bi = 0;
    for (let i = 0; i < p.x.length; i++) { const q = (p.x[i] - g.x[k]) ** 2 + (p.y[i] - g.y[k]) ** 2; if (q < best) { best = q; bi = i; } }
    b += line(30 + g.x[k] * 1.5, 40 + g.y[k] * 1.5, 30 + p.x[bi] * 1.5, 40 + p.y[bi] * 1.5, "#ffd400", 1.6) + rect(30 + g.x[k] * 1.5 - 3, 40 + g.y[k] * 1.5 - 3, 6, 6, { fill: "#ffd400", stroke: "#000", w: 0.8 });
  }
  for (let i = 0; i < p.x.length; i++) b += circle(30 + p.x[i] * 1.5, 40 + p.y[i] * 1.5, 2.4, { fill: C.sky });
  const F = arche.map(([q]) => PP.Fest(q, sq, R, "border", 4));
  b += curvePanel(430, 30, 380, 250, [0, 30], [0, 1], [0, 10, 20, 30], [0, 0.5, 1], "r (µm)", "F(r)",
    [{ xs: R, ys: Array.from(R, r => PP.csrG(r, rho)), color: "#000", dash: "7 5", w: 2.5, label: "CSR" }, ...F.map((f, k) => ({ xs: R, ys: f, color: COL[k], label: arche[k][1] }))], { legendAt: [810, 175] });
  out(D + "fig-f-dt.svg", svg(840, 360, b));
  console.log("F at 6 µm:", F.map(f => f[12].toFixed(3)), "F at 10:", F.map(f => f[20].toFixed(3)), "test locations", g.x.length, "max empty", mx.toFixed(1), "mean ge", (ge.reduce((a, v) => a + v, 0) / ge.length).toFixed(1));
}

// ---- J curves
{
  const J = arche.map(([p]) => PP.Jfun(PP.Gest(p, sq, R, "border"), PP.Fest(p, sq, R, "border", 4)));
  const b = curvePanel(80, 20, 520, 300, [0, 12], [0, 3], [0, 4, 8, 12], [0, 1, 2, 3], "r (µm)", "J(r) = (1 − G)/(1 − F)",
    [{ xs: R, ys: R.map(() => 1), color: "#000", dash: "7 5", w: 2.5, label: "CSR: J = 1" }, ...J.map((j, k) => ({ xs: R, ys: Array.from(j, v => Math.min(v, 3)), color: COL[k], label: arche[k][1] }))], { legendAt: [595, 25] });
  out(D + "fig-j.svg", svg(640, 400, b));
  console.log("J at 4,8:", J.map(j => `${j[8].toFixed(2)} ${j[16].toFixed(2)}`));
}

// ---- tight pairs: G sees only the pair distance
{
  const p = pat({ model: "pairs", pairD: 3 }), rp = p.x.length / sq.area;
  let b = patternPanel(30, 40, 260, sq, p, { r: 3, title: `tight pairs, n = ${p.x.length}`, under: nnArrows(p) });
  const G = PP.Gest(p, sq, R, "border");
  b += curvePanel(390, 30, 380, 240, [0, 30], [0, 1], [0, 10, 20, 30], [0, 0.5, 1], "r (µm)", "G(r)",
    [{ xs: R, ys: Array.from(R, r => PP.csrG(r, rp)), color: "#000", dash: "7 5", w: 2.5, label: "CSR" }, { xs: R, ys: G, color: C.fp, label: "pairs" }], { legendAt: [770, 160] });
  out(D + "fig-pairs.svg", svg(800, 340, b));
  console.log("pairs G(3)", G[6].toFixed(3), "G(2.5)", G[5].toFixed(3));
}

// ---- the edge effect: a point whose true nearest neighbour is outside the window
{
  const big = PP.binomialProcess(rng(41), 70, PP.rectWindow(140, 140)), W0 = 30, W1 = 110;   // window [30, 110]²
  const k = 3.2, X = (v: number) => 20 + v * k, Y = (v: number) => 20 + v * k;
  let b = rect(X(0), Y(0), 140 * k, 140 * k, { fill: "#fafafa", stroke: "#ccc", w: 1 }) + rect(X(W0), Y(W0), (W1 - W0) * k, (W1 - W0) * k, { fill: "#eef1f6", stroke: "#333", w: 2.5 });
  const insidePt = (i: number) => big.x[i] >= W0 && big.x[i] < W1 && big.y[i] >= W0 && big.y[i] < W1;
  // the inside point whose nearest neighbour inside W is farthest beyond its true nearest neighbour
  let sel = -1, tn = -1, tb = Infinity, wn = -1, wb = Infinity, gain = -1;
  for (let i = 0; i < big.x.length; i++) {
    if (!insidePt(i)) continue;
    let t1 = -1, d1 = Infinity, w1 = -1, e1 = Infinity;
    for (let j = 0; j < big.x.length; j++) {
      if (j === i) continue;
      const d = Math.hypot(big.x[j] - big.x[i], big.y[j] - big.y[i]);
      if (d < d1) { d1 = d; t1 = j; }
      if (insidePt(j) && d < e1) { e1 = d; w1 = j; }
    }
    if (e1 - d1 > gain) { gain = e1 - d1; sel = i; tn = t1; tb = d1; wn = w1; wb = e1; }
  }
  for (let i = 0; i < big.x.length; i++) b += circle(X(big.x[i]), Y(big.y[i]), 5, insidePt(i) ? { fill: C.blue, stroke: "#fff", w: 1 } : { fill: "#bbb" });
  b += line(X(big.x[sel]), Y(big.y[sel]), X(big.x[tn]), Y(big.y[tn]), C.fp, 3) + line(X(big.x[sel]), Y(big.y[sel]), X(big.x[wn]), Y(big.y[wn]), C.orange, 3, "7 4");
  b += circle(X(big.x[sel]), Y(big.y[sel]), 8, { stroke: "#000", w: 2.5 });
  b += text(X(W0) + 8, Y(W0) + 24, "window W", { anchor: "start", weight: "bold" });
  b += text(X(70), Y(134), "grey: points outside W, never seen", { fill: "#777", size: 19 });
  b += text(490, 100, "true nearest neighbour:", { anchor: "start", fill: C.fp, size: 20 }) + text(490, 126, `${tb.toFixed(1)} µm, outside W`, { anchor: "start", fill: C.fp, size: 20 });
  b += text(490, 160, "nearest seen in W:", { anchor: "start", fill: C.orange, size: 20 }) + text(490, 186, `${wb.toFixed(1)} µm`, { anchor: "start", fill: C.orange, size: 20 });
  b += text(490, 240, "near the border, distances", { anchor: "start", size: 20 }) + text(490, 266, "come out too long:", { anchor: "start", size: 20 }) + text(490, 292, "G too low, looks regular", { anchor: "start", size: 20, weight: "bold" });
  out(D + "fig-edge.svg", svg(860, 490, b));
  console.log("edge: true NN", tb.toFixed(1), "seen", wb.toFixed(1));
}

// ---- guard zone and the bias of an uncorrected G in a small window
{
  const w = PP.rectWindow(60, 60), p = PP.binomialProcess(rng(43), 14, w), r0 = 10;
  const b0 = Array.from(p.x, (x, i) => PP.borderDistance(w, x, p.y[i]));
  const k = 4.5;
  let b = patternPanel(30, 40, 60 * k, w, p, { r: 6, colors: i => (b0[i] >= r0 ? C.blue : "#bbb"), title: "border rule at r = 10 µm", under: (X, Y) => path([[X(0), Y(0)], [X(60), Y(0)], [X(60), Y(60)], [X(0), Y(60)]], { fill: C.fp, opacity: 0.12, stroke: "none", w: 0 }) + rect(X(r0), Y(r0), (60 - 2 * r0) * k, (60 - 2 * r0) * k, { fill: "#f4f5f7", stroke: C.fp, w: 2, dash: "6 4" }) });
  // mean curves over 400 CSR patterns with the same n
  const rr = rng(99), T = 400, none = new Float64Array(R.length), bord = new Float64Array(R.length), cnt = new Float64Array(R.length);
  for (let t = 0; t < T; t++) {
    const q = PP.binomialProcess(rr, 14, w), a = PP.Gest(q, w, R, "none"), c = PP.Gest(q, w, R, "border");
    for (let j = 0; j < R.length; j++) { none[j] += a[j] / T; if (Number.isFinite(c[j])) { bord[j] += c[j]; cnt[j]++; } }
  }
  for (let j = 0; j < R.length; j++) bord[j] = cnt[j] > T / 2 ? bord[j] / cnt[j] : NaN;
  const rh = 14 / w.area;
  b += curvePanel(390, 30, 420, 250, [0, 20], [0, 1], [0, 5, 10, 15, 20], [0, 0.5, 1], "r (µm)", "mean Ĝ(r), 400 CSR patterns",
    [{ xs: R, ys: Array.from(R, r => PP.csrG(r, rh)), color: "#000", dash: "7 5", w: 2.5, label: "theory" }, { xs: R, ys: none, color: C.fp, label: "no correction" }, { xs: R, ys: bord, color: C.tp, label: "border rule" }], { legendAt: [810, 170] });
  out(D + "fig-guard.svg", svg(840, 360, b));
  console.log("guard: kept", b0.filter(v => v >= r0).length, "of 14; mean G(10) none", none[20].toFixed(3), "border", bord[20].toFixed(3), "theory", PP.csrG(10, rh).toFixed(3));
}
