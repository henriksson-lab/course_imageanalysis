// Figures for point-patterns/ (SVG). Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/point-patterns/figs.ts --bundle --platform=node --log-level=warning | node
// All data synthetic and seeded; the numbers in captions are printed.
import { svg, text, line, rect, circle, path, greyImage, out, C, patternPanel, curvePanel } from "./kit";
import * as PP from "../../../demos/src/lib/pointpattern";
import { rng } from "../../../demos/src/lib/random";
import { nuclei } from "../../../demos/src/lib/synth";
import { makePattern, DEFAULT_SPEC, makeWindow, Spec } from "../../../demos/src/csrplay/source";

const D = "point-patterns/";
const sq = makeWindow("square");
const pat = (s: Partial<Spec>) => makePattern({ ...DEFAULT_SPEC, ...s });

// ---- objects → points
{
  const nu = nuclei({ width: 200, height: 200, count: 26, radius: [7, 11], touching: 0.15, seed: 4 });
  const n = nu.count, sx = new Float64Array(n + 1), sy = new Float64Array(n + 1), sc = new Float64Array(n + 1);
  for (let y = 0; y < 200; y++) for (let x = 0; x < 200; x++) { const l = nu.labels[y * 200 + x]; if (l) { sx[l] += x + 0.5; sy[l] += y + 0.5; sc[l]++; } }
  const xs: number[] = [], ys: number[] = [];
  for (let l = 1; l <= n; l++) if (sc[l]) { xs.push(sx[l] / sc[l]); ys.push(sy[l] / sc[l]); }
  const p = PP.makePoints(xs, ys), S = 300, k = S / 200;
  let b = greyImage(nu.image.data, 200, 0, 0, 200, 200, 30, 50, k) + text(30 + S / 2, 38, "image (synthetic nuclei)");
  // labels as outlines: boundary pixels of each label
  let lab = "";
  for (let y = 1; y < 199; y++) for (let x = 1; x < 199; x++) {
    const l = nu.labels[y * 200 + x];
    if (l && (nu.labels[y * 200 + x + 1] !== l || nu.labels[y * 200 + x - 1] !== l || nu.labels[(y + 1) * 200 + x] !== l || nu.labels[(y - 1) * 200 + x] !== l))
      lab += `<rect x="${(380 + x * k).toFixed(1)}" y="${(50 + y * k).toFixed(1)}" width="${k.toFixed(1)}" height="${k.toFixed(1)}" fill="${C.orange}"/>`;
  }
  b += rect(380, 50, S, S, { fill: "#222" }) + lab;
  for (let i = 0; i < xs.length; i++) b += circle(380 + xs[i] * k, 50 + ys[i] * k, 4, { fill: "#fff" });
  b += text(380 + S / 2, 38, "labels and centroids");
  b += patternPanel(730, 50, S, sq, p, { r: 5, title: `point pattern: ${xs.length} points + window` });
  b += path([[340, 200], [368, 200]], { w: 3 }) + path([[360, 193], [368, 200], [360, 207]], { w: 3 });
  b += path([[690, 200], [718, 200]], { w: 3 }) + path([[710, 193], [718, 200], [710, 207]], { w: 3 });
  out(D + "fig-objects-to-points.svg", svg(1060, 370, b));
  console.log("objects → points:", xs.length, "nuclei");
}

// ---- detection artefacts: the same true positions detected three ways
{
  const truth = pat({ model: "csr", n: 120, seed: 3 });
  const merged = PP.mergeClose(truth, 8), split = PP.splitSome(rng(5), truth, 0.3, 3, sq);
  const S = 300, list: [PP.Points, string][] = [[truth, "correct: 120 points"], [merged, `close spots merged: ${merged.x.length}`], [split, `30 % split: ${split.x.length}`]];
  let b = "";
  list.forEach(([p, t], k) => {
    const R = PP.Kest(p, sq, [3], "translation");
    b += patternPanel(30 + k * 350, 50, S, sq, p, { r: 4, title: t });
    void R;
  });
  const vmr = list.map(([p]) => PP.quadratCounts(p, sq, 10, 10).vmr);
  list.forEach((_, k) => { b += text(30 + k * 350 + S / 2, 50 + S + 30, `VMR (20 µm quadrats) = ${vmr[k].toFixed(2)}`, { size: 19 }); });
  out(D + "fig-artefacts.svg", svg(1060, 420, b));
  console.log("artefacts: n", list.map(([p]) => p.x.length), "VMR", vmr.map(v => v.toFixed(2)));
}

// ---- three archetypes, same n
const arche: [PP.Points, string][] = (() => {
  const cl = pat({ model: "thomas", k: 15, mu: 10, sigma: 5, seed: 2 });
  const n = cl.x.length;
  return [[cl, "clustered"], [PP.binomialProcess(rng(21), n, sq), "random (CSR)"], [PP.ssi(rng(22), n, 11, sq), "regular"]];
})();
{
  let b = "";
  arche.forEach(([p, t], k) => { b += patternPanel(30 + k * 350, 50, 300, sq, p, { r: 4, title: `${t}, n = ${p.x.length}` }); });
  out(D + "fig-archetypes.svg", svg(1060, 380, b));
}

// ---- Poisson counts in regions
{
  const p = pat({ model: "csr", n: 150, seed: 6 }), rho = 150 / sq.area;
  const A: [number, number, number, number] = [20, 30, 50, 50], B: [number, number, number, number] = [120, 110, 50, 50];
  const inA = Array.from(p.x).filter((x, i) => x >= A[0] && x < A[0] + A[2] && p.y[i] >= A[1] && p.y[i] < A[1] + A[3]).length;
  const inB = Array.from(p.x).filter((x, i) => x >= B[0] && x < B[0] + B[2] && p.y[i] >= B[1] && p.y[i] < B[1] + B[3]).length;
  let b = patternPanel(30, 40, 340, sq, p, { r: 4, extra: (X, Y) =>
    rect(X(A[0]), Y(A[1]), A[2] * 1.7, A[3] * 1.7, { stroke: C.orange, w: 3 }) + text(X(A[0] + A[2] / 2), Y(A[1]) - 6, `A: ${inA}`, { fill: C.orange, weight: "bold" }) +
    rect(X(B[0]), Y(B[1]), B[2] * 1.7, B[3] * 1.7, { stroke: C.purple, w: 3 }) + text(X(B[0] + B[2] / 2), Y(B[1]) - 6, `B: ${inB}`, { fill: C.purple, weight: "bold" }) });
  // pmf of Poisson(ρ|A|) for |A| = 2500 µm²
  const m = rho * 2500, ks = Array.from({ length: 21 }, (_, k) => k);
  const pmf = ks.map(k => Math.exp(k * Math.log(m) - m - lf(k)));
  const ax = { x0: 470, y0: 50, w: 520, h: 270 };
  let bars = "";
  const Xk = (k: number) => ax.x0 + (k + 0.5) / 21 * ax.w, Yp = (v: number) => ax.y0 + ax.h - v / 0.2 * ax.h;
  bars += curvePanel(ax.x0, ax.y0, ax.w, ax.h, [-0.5, 20.5], [0, 0.2], [0, 5, 10, 15, 20], [0, 0.05, 0.1, 0.15, 0.2], "points in a region of 2500 µm²", "probability", []);
  for (const k of ks) bars += rect(Xk(k) - 9, Yp(pmf[k]), 18, ax.y0 + ax.h - Yp(pmf[k]), { fill: C.blue, opacity: 0.8 });
  bars += line(Xk(inA), ax.y0, Xk(inA), ax.y0 + ax.h, C.orange, 3, "6 4") + text(Xk(inA) + 4, ax.y0 + 18, "A", { fill: C.orange, anchor: "start", weight: "bold" });
  bars += line(Xk(inB), ax.y0, Xk(inB), ax.y0 + ax.h, C.purple, 3, "6 4") + text(Xk(inB) + 4, ax.y0 + 18, "B", { fill: C.purple, anchor: "start", weight: "bold" });
  bars += text(ax.x0 + ax.w - 10, ax.y0 + 50, `Poisson, mean ρ·2500 = ${m.toFixed(2)}`, { anchor: "end", size: 19 });
  b += bars;
  out(D + "fig-poisson-counts.svg", svg(1010, 420, b));
  console.log("poisson counts: A", inA, "B", inB, "mean", m, "rho", rho);
}

// ---- quadrat grids with counts
{
  let b = "";
  const res: string[] = [];
  arche.forEach(([p, t], k) => {
    const q = PP.quadratCounts(p, sq, 5, 5), S = 300, x0 = 30 + k * 350, qs = S / 5;
    b += patternPanel(x0, 50, S, sq, p, { r: 3.4, title: t, extra: () => {
      let e = "";
      for (let j = 1; j < 5; j++) e += line(x0 + j * qs, 50, x0 + j * qs, 50 + S, C.orange, 1.5) + line(x0, 50 + j * qs, x0 + S, 50 + j * qs, C.orange, 1.5);
      for (let gy = 0; gy < 5; gy++) for (let gx = 0; gx < 5; gx++) e += `<text x="${x0 + (gx + 0.5) * qs}" y="${50 + (gy + 0.5) * qs + 9}" font-size="26" font-weight="bold" text-anchor="middle" fill="${C.fp}" stroke="#fff" stroke-width="4" paint-order="stroke">${q.counts[gy * 5 + gx]}</text>`;
      return e;
    } });
    b += text(x0 + S / 2, 50 + S + 30, `mean ${q.mean.toFixed(1)}, variance ${q.variance.toFixed(1)}, VMR ${q.vmr.toFixed(2)}`, { size: 19 });
    res.push(`${t}: mean ${q.mean} var ${q.variance.toFixed(2)} VMR ${q.vmr.toFixed(2)} X2 ${q.X2.toFixed(1)} pC ${q.pClustered.toFixed(4)} pR ${q.pRegular.toFixed(4)}`);
  });
  out(D + "fig-quadrats.svg", svg(1060, 420, b));
  console.log("quadrats:\n  " + res.join("\n  "));
}

// ---- VMR against quadrat size
{
  const sides = [2, 4, 5, 8, 10, 20, 25, 40], specs: [Partial<Spec>, string, string][] = [
    [{ model: "colonies" }, "colonies + hard core", C.orange], [{ model: "thomas", k: 20, mu: 8, sigma: 4 }, "clusters (Thomas)", C.fp],
    [{ model: "csr" }, "random", C.grey], [{ model: "hardcore", h: 12 }, "hard core 12 µm", C.blue], [{ model: "inhom", grad: 0.8 }, "gradient", C.tp]];
  const series = specs.map(([s, l, c]) => { const p = pat(s); return { xs: sides.map(k => Math.log10(200 / k)), ys: sides.map(k => PP.quadratCounts(p, sq, k, k).vmr), color: c, label: l }; });
  const b = curvePanel(90, 20, 620, 330, [Math.log10(4), Math.log10(120)], [0, 14], [Math.log10(5), 1, Math.log10(20), Math.log10(50), 2], [0, 1, 2, 4, 6, 8, 10, 12, 14], "quadrat side (µm, log scale)", "VMR",
    series, { fmt: v => String(Math.round(10 ** v)), yfmt: v => String(v), hlines: [{ y: 1 }], legendAt: [330, 25] });
  out(D + "fig-vmr-scale.svg", svg(740, 420, b));
  console.log("VMR vs side:", series.map(s => s.label + " " + s.ys.map(v => v.toFixed(2)).join(",")).join(" | "));
}

// ---- kernel intensity at three bandwidths
{
  const p = pat({ model: "inhom", grad: 0.8 }), N = 50;
  let b = "";
  [4, 15, 60].forEach((bw, k) => {
    const map = PP.kernelIntensity(p, sq, bw, N, N, true), x0 = 30 + k * 350;
    const fin = Array.from(map), hi = Math.max(...fin);
    const u = new Uint8Array(N * N);
    for (let i = 0; i < N * N; i++) u[i] = Math.round(255 * map[i] / hi);
    b += greyImage(u, N, 0, 0, N, N, x0, 50, 6, 0, 255);
    for (let i = 0; i < p.x.length; i++) b += circle(x0 + p.x[i] * 1.5, 50 + p.y[i] * 1.5, 2.6, { fill: C.orange });
    b += text(x0 + 150, 38, `bandwidth σ = ${bw} µm`);
    b += text(x0 + 150, 385, ["a bump per point", "follows the gradient", "almost a constant"][k], { size: 19 });
  });
  out(D + "fig-intensity.svg", svg(1060, 400, b));
}

// ---- four models
{
  const list: [PP.Points, string][] = [
    [pat({ model: "thomas", k: 20, mu: 8, sigma: 4 }), "cluster (Thomas)"], [pat({ model: "hardcore", n: 150, h: 10 }), "hard core (h = 10 µm)"],
    [pat({ model: "lattice", spacing: 16, jitter: 2 }), "jittered lattice"], [pat({ model: "inhom", grad: 0.8 }), "inhomogeneous Poisson"]];
  let b = "";
  list.forEach(([p, t], k) => { b += patternPanel(30 + (k % 2) * 300, 40 + Math.floor(k / 2) * 300, 250, sq, p, { r: 3, title: t }); });
  out(D + "fig-models.svg", svg(590, 610, b));
}

function lf(k: number): number { let s = 0; for (let i = 2; i <= k; i++) s += Math.log(i); return s; }
