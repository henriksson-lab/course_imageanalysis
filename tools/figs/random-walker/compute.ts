// Data for the figures of random-walker/, computed with the course's own code
// (demos/src/lib/randomwalker.ts, lib/graph.ts). Run from the repository root:
//   python3 -I tools/figs/random-walker/figs.py <rawdir> --export-cell
//   node_modules/.bin/esbuild tools/figs/random-walker/compute.ts --bundle --platform=node --log-level=warning > <rawdir>/rw.js && node <rawdir>/rw.js <rawdir>
//   python3 -I tools/figs/random-walker/figs.py <rawdir>
import * as RW from "../../../demos/src/lib/randomwalker";
import { gridEdges, contrastWeights, solveSeg } from "../../../demos/src/lib/graph";
import { rng } from "../../../demos/src/lib/random";
import { walkerScene } from "../../../demos/src/fhseg/scenes";

declare const process: { argv: string[] };
declare function require(m: string): { readFileSync(p: string, e: string): string; writeFileSync(p: string, d: string): void };
const fs = require("fs");
const raw = process.argv[2];
const out: Record<string, unknown> = {};
const arr = (a: ArrayLike<number>, d = 4) => Array.from(a, v => Math.round(v * 10 ** d) / 10 ** d);

// cell at 128 × 128 (2 × 2 means)
const c256: number[] = JSON.parse(fs.readFileSync(`${raw}/cell256.json`, "utf8"));
const cell = new Float64Array(128 * 128);
for (let y = 0; y < 128; y++) for (let x = 0; x < 128; x++) cell[y * 128 + x] = (c256[2 * y * 256 + 2 * x] + c256[2 * y * 256 + 2 * x + 1] + c256[(2 * y + 1) * 256 + 2 * x] + c256[(2 * y + 1) * 256 + 2 * x + 1]) / 4;

const disc = (s: Int32Array, w: number, cx: number, cy: number, r: number, v: number) => { for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) if (x * x + y * y <= r * r) s[(cy + y) * w + cx + x] = v; };
const line = (s: Int32Array, w: number, x0: number, y0: number, x1: number, y1: number, r: number, v: number) => { const n = Math.ceil(Math.hypot(x1 - x0, y1 - y0)); for (let i = 0; i <= n; i++) disc(s, w, Math.round(x0 + (x1 - x0) * i / n), Math.round(y0 + (y1 - y0) * i / n), r, v); };
const solve = (I: Float64Array, w: number, h: number, seeds: Int32Array, K: number, sigma: number, o: RW.WalkerOptions = {}) => RW.randomWalker(I, w, h, seeds, K, { sigma, tol: 1e-7, maxIter: 30000, ...o });

// 1. walks on a 24 × 24 two-region image
{
  const w = 24, h = 24, I = new Float64Array(w * h);
  // a bright disc; σ = 60 makes its rim a weak (not impassable) barrier
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const r2 = (x - 9) ** 2 + (y - 11) ** 2; I[y * w + x] = r2 < 49 ? 200 : 60; }
  const seeds = new Int32Array(w * h); for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) seeds[(11 + dy) * w + 9 + dx] = 1; seeds[21 * w + 21] = 2; seeds[2 * w + 21] = 2; seeds[22 * w + 2] = 2;
  const G = RW.gridGraph(I, w, h, 60, 4, 1e-4), A = RW.csr(G);
  const res = RW.walkerGraph(G, seeds, 2, { tol: 1e-10 });
  let start = 0, bd = 9;
  for (let p = 0; p < w * h; p++) if (!seeds[p] && Math.abs(res.prob[0][p] - 0.8) < bd) { bd = Math.abs(res.prob[0][p] - 0.8); start = p; }
  const wk = RW.simulateWalks(A, seeds, start, 20, rng(4), 200000, 20);
  out.walks = { w, h, I: arr(I, 0), seeds: Array.from(seeds), start, paths: wk.paths.map(p => p.slice(0, 3000)), counts: wk.counts, p: res.prob[0][start] };
}
// 2. micrograph from two strokes
{
  const s = new Int32Array(128 * 128);
  line(s, 128, 52, 60, 70, 66, 2, 1);
  line(s, 128, 8, 8, 8, 120, 2, 2); line(s, 128, 120, 8, 120, 120, 2, 2); line(s, 128, 28, 116, 100, 120, 2, 2);
  const r = solve(cell, 128, 128, s, 2, 12);
  out.cell = { I: arr(cell, 1), seeds: Array.from(s), p: arr(r.prob[0], 3), it: r.iterations };
}
// 3. three labels on the objects scene
{
  const sc = walkerScene("objects", 10);
  const s = new Int32Array(128 * 128);
  disc(s, 128, 50, 46, 2, 1); disc(s, 128, 52, 92, 2, 3); disc(s, 128, 10, 10, 2, 2); disc(s, 128, 118, 118, 2, 2);
  const r = solve(sc.I, 128, 128, s, 3, 20);
  out.three = { I: arr(sc.I, 1), seeds: Array.from(s), p: r.prob.map(p => arr(p, 3)) };
}
// 4. fading: ramp scene, σ = ∞ (no edges at all), two corner seeds
{
  const sc = walkerScene("ramp", 4);
  const s = new Int32Array(128 * 128); disc(s, 128, 4, 4, 1, 1); disc(s, 128, 123, 123, 1, 2);
  const r = solve(sc.I, 128, 128, s, 2, Infinity);
  out.fade = { I: arr(sc.I, 1), seeds: Array.from(s), p: arr(r.prob[0], 3) };
}
// 5. no edge: ramp scene, walker vs priors
{
  const sc = walkerScene("ramp", 6);
  const s = new Int32Array(128 * 128); line(s, 128, 6, 30, 6, 98, 2, 1); line(s, 128, 86, 30, 86, 98, 2, 2);
  const r0 = solve(sc.I, 128, 128, s, 2, 20);
  const pri = RW.seedPriors(sc.I, s, 2);
  const r1 = solve(sc.I, 128, 128, s, 2, 20, { lambda: 0.05, priors: pri });
  const cross = (p: Float64Array) => { let x = 0; for (; x < 127; x++) if (p[64 * 128 + x] >= 0.5 && p[64 * 128 + x + 1] < 0.5) break; return x + 0.5; };
  out.priors = { I: arr(sc.I, 1), seeds: Array.from(s), p0: arr(r0.prob[0], 3), p1: arr(r1.prob[0], 3), cross0: cross(r0.prob[0]), cross1: cross(r1.prob[0]) };
}
// 6. graph cut (seeds only) vs walker: thin protrusion and a gap
{
  const res: Record<string, unknown> = {};
  for (const kind of ["objects", "gap"] as const) {
    const sc = walkerScene(kind, 8);
    const s = new Int32Array(128 * 128);
    if (kind === "objects") { disc(s, 128, 50, 46, 3, 1); disc(s, 128, 52, 92, 3, 2); disc(s, 128, 10, 10, 3, 2); disc(s, 128, 124, 40, 2, 2); }
    else { disc(s, 128, 20, 20, 3, 1); disc(s, 128, 110, 100, 3, 2); }
    const rw = solve(sc.I, 128, 128, s, 2, kind === "objects" ? 30 : 20);
    const E = gridEdges(128, 128, 4), s8 = Int8Array.from(s, v => (v === 1 ? 1 : v ? -1 : 0));
    const gc = solveSeg({ n: 128 * 128, D0: new Float64Array(128 * 128), D1: new Float64Array(128 * 128), E, w: contrastWeights(sc.I, E, kind === "objects" ? 30 : 20), lambda: 1, seeds: s8 }).L;
    res[kind] = { I: arr(sc.I, 1), seeds: Array.from(s), rw: Array.from(rw.labels), gc: Array.from(gc) };
  }
  out.gcrw = res;
}
// 7. CG residual against iteration (objects scene, three σ)
{
  const sc = walkerScene("objects", 10);
  const s = new Int32Array(128 * 128); disc(s, 128, 50, 46, 2, 1); disc(s, 128, 10, 10, 2, 2);
  const curves: Record<string, number[]> = {};
  for (const sigma of [5, 20, 80]) {
    const G = RW.gridGraph(sc.I, 128, 128, sigma, 4, 1e-4);
    const ws = new RW.WalkerSolver(G, s, 2, { tol: 1e-8 });
    const hist: number[] = [];
    let guard = 0;
    while (!ws.done && guard++ < 6000) { (ws as unknown as { runs: { step(n: number): void }[] }).runs[0].step(1); hist.push(ws.residual); }
    curves[String(sigma)] = hist.filter((_, i) => i % 5 === 0);
  }
  out.cg = curves;
}
fs.writeFileSync(`${raw}/rw.json`, JSON.stringify(out));
console.log("walks", (out.walks as { counts: number[]; p: number }).counts, (out.walks as { p: number }).p.toFixed(3), "priors crossings", (out.priors as { cross0: number }).cross0, (out.priors as { cross1: number }).cross1);
