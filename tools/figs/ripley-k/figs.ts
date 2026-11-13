// Figures for ripley-k/ (SVG). Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/ripley-k/figs.ts --bundle --platform=node --log-level=warning | node
import { svg, text, line, rect, circle, path, out, C, patternPanel, curvePanel } from "../point-patterns/kit";
import * as PP from "../../../demos/src/lib/pointpattern";
import { rng } from "../../../demos/src/lib/random";
import { makePattern, DEFAULT_SPEC, makeWindow, Spec } from "../../../demos/src/csrplay/source";

const D = "ripley-k/";
const sq = makeWindow("square");
const pat = (s: Partial<Spec>) => makePattern({ ...DEFAULT_SPEC, ...s });
const R = PP.linspace(0, 40, 81);
const COL = [C.fp, C.grey, C.blue];
const cl = pat({ model: "thomas", k: 15, mu: 10, sigma: 5, seed: 2 }), n = cl.x.length;
const arche: [PP.Points, string][] = [[cl, "clustered"], [PP.binomialProcess(rng(21), n, sq), "random"], [PP.ssi(rng(22), n, 11, sq), "regular"]];

// ---- circles of radius r around three points, with counts
{
  const p = pat({ model: "csr", n: 150, seed: 6 }), r0 = 20, rho = p.x.length / sq.area;
  const picks = [0, 1, 2].map(k => { let best = Infinity, bi = 0; const t = [[60, 60], [140, 90], [90, 150]][k]; for (let i = 0; i < p.x.length; i++) { const q = (p.x[i] - t[0]) ** 2 + (p.y[i] - t[1]) ** 2; if (q < best) { best = q; bi = i; } } return bi; });
  const counts = picks.map(i => { let c = 0; for (let j = 0; j < p.x.length; j++) if (j !== i && Math.hypot(p.x[j] - p.x[i], p.y[j] - p.y[i]) <= r0) c++; return c; });
  const b = patternPanel(30, 30, 400, sq, p, { r: 4, extra: (X, Y, k) => picks.map((i, m) => circle(X(p.x[i]), Y(p.y[i]), r0 * k, { stroke: C.orange, w: 3, fill: C.orange }).replace('fill="' + C.orange + '"', `fill="${C.orange}" fill-opacity="0.12"`) + circle(X(p.x[i]), Y(p.y[i]), 6, { fill: C.fp }) + `<text x="${X(p.x[i]) + 10}" y="${Y(p.y[i]) - 8}" font-size="26" font-weight="bold" fill="${C.fp}" stroke="#fff" stroke-width="4" paint-order="stroke">${counts[m]}</text>`).join("") })
    + text(460, 80, `r = ${r0} µm`, { anchor: "start", size: 22 }) + text(460, 120, `neighbours: ${counts.join(", ")}`, { anchor: "start", size: 22 })
    + text(460, 160, `CSR expects ρπr² = ${(rho * Math.PI * r0 * r0).toFixed(1)}`, { anchor: "start", size: 22 })
    + text(460, 230, "K(r) = mean count / ρ", { anchor: "start", size: 22, weight: "bold" }) + text(460, 265, "CSR: K(r) = πr²", { anchor: "start", size: 22 });
  out(D + "fig-circles.svg", svg(800, 460, b));
  console.log("circles counts", counts, "csr", (rho * Math.PI * r0 * r0).toFixed(2));
}

// ---- K and L − r for the archetypes
{
  const Ks = arche.map(([p]) => PP.Kest(p, sq, R, "translation"));
  let b = curvePanel(90, 20, 420, 300, [0, 40], [0, 15000], [0, 10, 20, 30, 40], [0, 5000, 10000, 15000], "r (µm)", "K(r)  (µm²)",
    [{ xs: R, ys: Array.from(R, r => Math.PI * r * r), color: "#000", dash: "7 5", w: 2.5, label: "CSR: πr²" }, ...Ks.map((K, k) => ({ xs: R, ys: K, color: COL[k], label: arche[k][1] }))], { legendAt: [330, 25] });
  const Ls = Ks.map(K => PP.LminusR(K, R));
  b += curvePanel(660, 20, 420, 300, [0, 40], [-12, 20], [0, 10, 20, 30, 40], [-10, 0, 10, 20], "r (µm)", "L(r) − r  (µm)",
    [{ xs: R, ys: R.map(() => 0), color: "#000", dash: "7 5", w: 2.5 }, ...Ls.map((L, k) => ({ xs: R, ys: L, color: COL[k] }))]);
  out(D + "fig-k-l.svg", svg(1110, 400, b));
  console.log("L-r at 10, 20:", Ls.map(L => `${L[20].toFixed(2)} ${L[40].toFixed(2)}`));
}

// ---- disc vs ring
{
  const p = pat({ model: "csr", n: 150, seed: 6 }), cx = 100, cy = 100, r0 = 30, h = 5;
  let i0 = 0, best = Infinity;
  for (let i = 0; i < p.x.length; i++) { const q = (p.x[i] - cx) ** 2 + (p.y[i] - cy) ** 2; if (q < best) { best = q; i0 = i; } }
  const k = 2.2, X = (v: number) => 20 + (v - 40) * k, Y = (v: number) => 20 + (v - 40) * k, inView = (i: number) => p.x[i] > 40 && p.x[i] < 160 && p.y[i] > 40 && p.y[i] < 160;
  let disc = 0, ring = 0;
  const mk = (ox: number, mode: "disc" | "ring") => {
    let s = rect(ox + 20, 20, 120 * k, 120 * k, { fill: "#f4f5f7", stroke: "#999", w: 1 });
    if (mode === "disc") s += circle(ox + X(p.x[i0]), Y(p.y[i0]), r0 * k, { stroke: C.orange, w: 3 }).replace('fill="none"', `fill="${C.orange}" fill-opacity="0.15"`);
    else s += `<path d="M${ox + X(p.x[i0]) + (r0 + h) * k},${Y(p.y[i0])}a${(r0 + h) * k},${(r0 + h) * k} 0 1,0 ${-2 * (r0 + h) * k},0a${(r0 + h) * k},${(r0 + h) * k} 0 1,0 ${2 * (r0 + h) * k},0zM${ox + X(p.x[i0]) + (r0 - h) * k},${Y(p.y[i0])}a${(r0 - h) * k},${(r0 - h) * k} 0 1,1 ${-2 * (r0 - h) * k},0a${(r0 - h) * k},${(r0 - h) * k} 0 1,1 ${2 * (r0 - h) * k},0z" fill="${C.purple}" fill-opacity="0.25" stroke="${C.purple}" stroke-width="2"/>`;
    for (let i = 0; i < p.x.length; i++) {
      if (!inView(i)) continue;
      const d = Math.hypot(p.x[i] - p.x[i0], p.y[i] - p.y[i0]), hit = i !== i0 && (mode === "disc" ? d <= r0 : Math.abs(d - r0) < h);
      if (hit && mode === "disc") disc++;
      if (hit && mode === "ring") ring++;
      s += circle(ox + X(p.x[i]), Y(p.y[i]), i === i0 ? 7 : 5, { fill: i === i0 ? C.fp : hit ? (mode === "disc" ? C.orange : C.purple) : C.blue, stroke: "#fff", w: 1 });
    }
    return s;
  };
  let b = mk(0, "disc") + mk(320, "ring");
  const rho = p.x.length / sq.area;
  b += text(152, 300, `disc r = ${r0}: ${disc} points (CSR ${(rho * Math.PI * r0 * r0).toFixed(1)})`, { size: 19 });
  b += text(472, 300, `ring r ± ${h}: ${ring} points (CSR ${(rho * Math.PI * ((r0 + h) ** 2 - (r0 - h) ** 2)).toFixed(1)})`, { size: 19 });
  b += text(152, 330, "K: everything closer than r", { size: 19, weight: "bold" }) + text(472, 330, "g: only at distance r", { size: 19, weight: "bold" });
  out(D + "fig-ring.svg", svg(640, 350, b));
  console.log("disc", disc, "ring", ring);
}

// ---- K remembers: small clusters, L − r and g
{
  const p = pat({ model: "thomas", k: 25, mu: 6, sigma: 1.5 }), pd = PP.pairData(p, sq, 46, "translation");
  const L = PP.LminusR(PP.KfromPairs(pd, R), R), g = PP.pcfFromPairs(pd, R, 2);
  let b = patternPanel(30, 40, 260, sq, p, { r: 2.6, title: `small clusters (σ = 1.5 µm), n = ${p.x.length}` });
  b += curvePanel(390, 20, 330, 260, [0, 40], [-2, 16], [0, 10, 20, 30, 40], [0, 5, 10, 15], "r (µm)", "L(r) − r  (µm)", [{ xs: R, ys: R.map(() => 0), color: "#000", dash: "7 5", w: 2 }, { xs: R, ys: L, color: C.blue }]);
  b += curvePanel(810, 20, 330, 260, [0, 40], [0, 10], [0, 10, 20, 30, 40], [0, 1, 5, 10], "r (µm)", "g(r)", [{ xs: R, ys: R.map(() => 1), color: "#000", dash: "7 5", w: 2 }, { xs: R, ys: Array.from(g, v => Math.min(v, 10)), color: C.purple }]);
  b += text(555, 370, "L − r stays high far out: K remembers", { size: 19 }) + text(975, 370, "g is back at 1 beyond 6 µm", { size: 19 });
  out(D + "fig-k-vs-g.svg", svg(1160, 385, b));
  console.log("remember L-r at 5,20,30:", [10, 40, 60].map(k => L[k].toFixed(2)), "g at 2,4,6,10:", [4, 8, 12, 20].map(k => g[k].toFixed(2)));
}

// ---- reading scales: Thomas g (with formula) and hard-core g
{
  const kappa = 20 / sq.area, sigma = 4;
  const acc = new Float64Array(R.length), T = 40, r0 = rng(5);
  for (let t = 0; t < T; t++) { const p = PP.thomas(r0, kappa, 8, sigma, PP.rectWindow(400, 400)); const g = PP.pcf(p, PP.rectWindow(400, 400), R, 1.5); for (let k = 0; k < R.length; k++) acc[k] += g[k] / T; }
  let b = curvePanel(90, 20, 440, 300, [0, 40], [0, 12], [0, 10, 20, 30, 40], [0, 1, 4, 8, 12], "r (µm)", "g(r)",
    [{ xs: R, ys: R.map(() => 1), color: "#000", dash: "7 5", w: 2 }, { xs: R, ys: Array.from(R, r => PP.thomasPcf(r, kappa, sigma)), color: C.tp, dash: "4 3", label: "Thomas formula" }, { xs: R, ys: acc, color: C.purple, label: "mean of 40 patterns" }],
    { vlines: [{ x: 4 * sigma, color: C.fp, label: "4σ = 16 µm" }], legendAt: [530, 110] });
  b += text(310, 410, "clusters (κ = 20 per 200², μ = 8, σ = 4 µm)", { size: 19 });
  const hc = pat({ model: "hardcore", n: 200, h: 8 }), gh = PP.pcf(hc, sq, R, 1.5);
  b += curvePanel(680, 20, 440, 300, [0, 40], [0, 2], [0, 10, 20, 30, 40], [0, 1, 2], "r (µm)", "g(r)",
    [{ xs: R, ys: R.map(() => 1), color: "#000", dash: "7 5", w: 2 }, { xs: R, ys: gh, color: C.blue }], { vlines: [{ x: 8, color: C.fp, label: "h = 8 µm" }] });
  b += text(900, 410, "hard core: g = 0 below h, a bump just above", { size: 19 });
  out(D + "fig-scales.svg", svg(1140, 425, b));
  console.log("thomas mean g at 0.5,4,8,16,20:", [1, 8, 16, 32, 40].map(k => acc[k].toFixed(2)), "formula", [0.5, 4, 8, 16, 20].map(r => PP.thomasPcf(r, kappa, sigma).toFixed(2)), "hardcore g at 9,10,12:", [18, 20, 24].map(k => gh[k].toFixed(2)));
}

// ---- two scales: colonies with a hard core: G, L − r, g
{
  const p = pat({ model: "colonies" }), rho = p.x.length / sq.area, pd = PP.pairData(p, sq, 46, "translation");
  const G = PP.Gest(p, sq, R, "border"), L = PP.LminusR(PP.KfromPairs(pd, R), R), g = PP.pcfFromPairs(pd, R, 1.5);
  let b = patternPanel(30, 40, 250, sq, p, { r: 2.6, title: `colonies + hard core 6 µm, n = ${p.x.length}` });
  b += curvePanel(370, 20, 230, 230, [0, 30], [0, 1], [0, 10, 20, 30], [0, 0.5, 1], "r (µm)", "G(r)", [{ xs: R, ys: Array.from(R, r => PP.csrG(r, rho)), color: "#000", dash: "7 5", w: 2 }, { xs: R, ys: G, color: C.fp }]);
  b += curvePanel(680, 20, 230, 230, [0, 40], [-6, 12], [0, 20, 40], [-5, 0, 5, 10], "r (µm)", "L − r", [{ xs: R, ys: R.map(() => 0), color: "#000", dash: "7 5", w: 2 }, { xs: R, ys: L, color: C.blue }]);
  b += curvePanel(990, 20, 230, 230, [0, 40], [0, 6], [0, 20, 40], [0, 1, 3, 6], "r (µm)", "g(r)", [{ xs: R, ys: R.map(() => 1), color: "#000", dash: "7 5", w: 2 }, { xs: R, ys: Array.from(g, v => Math.min(v, 6)), color: C.purple }], { vlines: [{ x: 6, color: C.fp }, { x: 20, color: C.tp }] });
  b += text(485, 330, "G: 0 below 6 µm, then steep", { size: 18 }) + text(795, 330, "L − r: < 0, then > 0", { size: 18 }) + text(1105, 330, "g: 0 below 6, > 1 to ≈ 20", { size: 18 });
  out(D + "fig-two-scales.svg", svg(1240, 345, b));
  console.log("two scales n", p.x.length, "g at 4,7,10,16,24:", [8, 14, 20, 32, 48].map(k => g[k].toFixed(2)), "L-r at 5,10,20:", [10, 20, 40].map(k => L[k].toFixed(2)));
}

// ---- edge weights: translation and isotropic
{
  const W = 300, H = 200, x0 = 40, y0 = 40, xi = [90, 60], xj = [170, 105];
  const dx = xj[0] - xi[0], dy = xj[1] - xi[1];
  let b = rect(x0, y0, W, H, { fill: "#eef1f6", stroke: "#333", w: 2.5 }) + rect(x0 + dx, y0 + dy, W, H, { fill: "none", stroke: C.orange, w: 2.5, dash: "8 5" });
  b += rect(x0 + dx, y0 + dy, W - dx, H - dy, { fill: C.orange, opacity: 0.2 });
  b += line(x0 + xi[0], y0 + xi[1], x0 + xj[0], y0 + xj[1], C.fp, 3) + circle(x0 + xi[0], y0 + xi[1], 7, { fill: C.fp }) + circle(x0 + xj[0], y0 + xj[1], 7, { fill: C.fp });
  b += text(x0 + xi[0] - 10, y0 + xi[1] + 28, "xᵢ", { size: 22 }) + text(x0 + xj[0] + 18, y0 + xj[1] - 8, "xⱼ", { size: 22 });
  b += text(x0 + W / 2 + 40, y0 + H + dy + 40, "translation: W and W shifted by xⱼ − xᵢ", { size: 19 });
  b += text(x0 + W / 2 + 40, y0 + H + dy + 66, `weight = |W| / overlap = ${((W * H) / ((W - Math.abs(dx)) * (H - Math.abs(dy)))).toFixed(2)}`, { size: 19, weight: "bold" });
  // isotropic: circle through xj centred at xi, part outside
  const ox = 520, d = Math.hypot(dx, dy) * 1.2, ci = [60, 110];
  b += rect(ox, y0, W, H, { fill: "#eef1f6", stroke: "#333", w: 2.5 });
  const frac = PP.circleFraction(PP.rectWindow(W, H), ci[0], ci[1], d);
  // arc inside (solid) and outside (dashed)
  let inside = "", outside = "";
  for (let a = 0; a <= 360; a += 2) {
    const px = ci[0] + d * Math.cos(a * Math.PI / 180), py = ci[1] + d * Math.sin(a * Math.PI / 180);
    const pt = `${(ox + px).toFixed(1)},${(y0 + py).toFixed(1)}`;
    if (px >= 0 && px <= W && py >= 0 && py <= H) { inside += (inside ? " L" : "M") + pt; outside += " M" + pt; } else { outside += " L" + pt; inside += " M" + pt; }
  }
  b += `<path d="${inside.replace(/^ M/, "M")}" fill="none" stroke="${C.tp}" stroke-width="4"/><path d="${outside.replace(/^ L/, "M")}" fill="none" stroke="${C.fp}" stroke-width="3" stroke-dasharray="6 4"/>`;
  b += circle(ox + ci[0], y0 + ci[1], 7, { fill: C.fp }) + text(ox + ci[0] + 4, y0 + ci[1] + 30, "xᵢ", { size: 22 });
  const jx = ci[0] + d * Math.cos(-0.5), jy = ci[1] + d * Math.sin(-0.5);
  b += line(ox + ci[0], y0 + ci[1], ox + jx, y0 + jy, C.fp, 3) + circle(ox + jx, y0 + jy, 7, { fill: C.fp }) + text(ox + jx + 18, y0 + jy - 6, "xⱼ", { size: 22 });
  b += text(ox + W / 2, y0 + H + dy + 40, `isotropic: ${(100 * frac).toFixed(0)} % of the circle inside`, { size: 19 });
  b += text(ox + W / 2, y0 + H + dy + 66, `weight = 1 / ${frac.toFixed(2)} = ${(1 / frac).toFixed(2)}`, { size: 19, weight: "bold" });
  out(D + "fig-weights.svg", svg(870, 360, b));
  console.log("weights: translation", ((W * H) / ((W - Math.abs(dx)) * (H - Math.abs(dy)))).toFixed(3), "isotropic frac", frac.toFixed(3));
}

// ---- bias of the four corrections (200 CSR patterns of 100 points in 200 × 200 µm)
{
  const T = 200, r0 = rng(77), radii = PP.linspace(5, 50, 46), corrs: PP.KCorr[] = ["none", "border", "translation", "isotropic"];
  const acc = corrs.map(() => new Float64Array(radii.length));
  for (let t = 0; t < T; t++) {
    const p = PP.binomialProcess(r0, 100, sq);
    corrs.forEach((c, k) => { const K = PP.Kest(p, sq, radii, c); for (let j = 0; j < radii.length; j++) acc[k][j] += (K[j] / (Math.PI * radii[j] ** 2) - 1) * 100 / T; });
  }
  const b = curvePanel(100, 20, 560, 320, [0, 50], [-25, 5], [0, 10, 20, 30, 40, 50], [-25, -20, -15, -10, -5, 0, 5], "r (µm)", "mean bias of K̂ (%)",
    corrs.map((c, k) => ({ xs: radii, ys: acc[k], color: [C.fp, C.orange, C.blue, C.tp][k], label: c })), { hlines: [{ y: 0 }], legendAt: [300, 230] });
  out(D + "fig-bias.svg", svg(700, 420, b));
  console.log("bias at 20, 50 µm:", corrs.map((c, k) => `${c} ${acc[k][15].toFixed(1)} ${acc[k][45].toFixed(1)}`).join(" | "));
}
