// Data for the figures of ws-msf/: the three seeded methods on the same seeds and weights
// (lib/randomwalker.ts, lib/graph.ts, lib/mst.ts), and timings. Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/ws-msf/compute.ts --bundle --platform=node --log-level=warning > <rawdir>/ws.js && node <rawdir>/ws.js <rawdir>
//   python3 -I tools/figs/ws-msf/figs.py <rawdir>
import * as RW from "../../../demos/src/lib/randomwalker";
import * as M from "../../../demos/src/lib/mst";
import { gridEdges, contrastWeights, solveSeg } from "../../../demos/src/lib/graph";
import { walkerScene } from "../../../demos/src/fhseg/scenes";

declare const process: { argv: string[] };
declare function require(m: string): { writeFileSync(p: string, d: string): void };
const fs = require("fs");
const raw = process.argv[2];
const out: Record<string, unknown> = {};
const N = 128;
const disc = (s: Int32Array, cx: number, cy: number, r: number, v: number) => { for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) if (x * x + y * y <= r * r) s[(cy + y) * N + cx + x] = v; };

function three(I: Float64Array, seeds: Int32Array, sigma: number, n = N) {
  const t0 = performance.now();
  const G = RW.gridGraph(I, n, n, sigma, 4, 1e-4);
  const ws = new RW.WalkerSolver(G, seeds, 2, { tol: 1e-5 });
  while (!ws.done) ws.step(1e6);
  const rw = ws.prob()[0];
  const t1 = performance.now();
  const E = gridEdges(n, n, 4), s8 = Int8Array.from(seeds, v => (v === 1 ? 1 : v ? -1 : 0));
  const gc = solveSeg({ n: n * n, D0: new Float64Array(n * n), D1: new Float64Array(n * n), E, w: contrastWeights(I, E, sigma), lambda: 1, seeds: s8 }).L;
  const t2 = performance.now();
  const msf = M.primForest(M.gridDissim([I], n, n, 4), seeds).labels;
  const t3 = performance.now();
  return { rw: Array.from(rw, v => (v > 0.5 ? 1 : 0)), gc: Array.from(gc), ws: Array.from(msf, v => (v === 1 ? 1 : 0)), ms: [t1 - t0, t2 - t1, t3 - t2] };
}
const cases: Record<string, unknown> = {};
{
  const s = new Int32Array(N * N); disc(s, 20, 64, 3, 1); disc(s, 120, 120, 2, 2);
  const I = walkerScene("gap", 16).I; cases.gap = { I: Array.from(I, Math.round), seeds: Array.from(s), ...three(I, s, 20) };
}
{
  const s = new Int32Array(N * N); disc(s, 40, 64, 3, 1); disc(s, 124, 64, 3, 2);
  const I = walkerScene("ramp", 4).I; cases.ramp = { I: Array.from(I, Math.round), seeds: Array.from(s), ...three(I, s, 20) };
}
{
  const s = new Int32Array(N * N); disc(s, 50, 46, 3, 1); disc(s, 52, 92, 3, 2); disc(s, 10, 10, 3, 2); disc(s, 118, 118, 3, 2);
  const I = walkerScene("objects", 8).I; cases.objects = { I: Array.from(I, Math.round), seeds: Array.from(s), ...three(I, s, 20) };
}
out.cases = cases;
for (const [k, v] of Object.entries(cases)) { const c = v as { rw: number[]; gc: number[]; ws: number[] }; console.log(k, [c.rw, c.gc, c.ws].map(a => a.reduce((x, y) => x + y, 0))); }
// timings at 128² and 256² (a scene with two seeds), median of 3
const tim: Record<string, number[]> = {};
for (const n of [128, 256]) {
  const I = new Float64Array(n * n);
  for (let p = 0; p < n * n; p++) { const x = p % n, y = (p - x) / n; I[p] = ((x - n / 2) ** 2 + (y - n / 2) ** 2 < (n / 4) ** 2 ? 180 : 60) + 10 * Math.sin(p * 12.9898) ; }
  const s = new Int32Array(n * n); s[(n / 2) * n + n / 2] = 1; s[n + 1] = 2; s[n * n - n - 2] = 2;
  const runs: number[][] = [];
  for (let r = 0; r < 3; r++) {
    let t = performance.now(); M.primForest(M.gridDissim([I], n, n, 4), s); const a = performance.now() - t;
    t = performance.now(); RW.randomWalker(I, n, n, s, 2, { sigma: 20, tol: 1e-5, maxIter: 20000 }); const b = performance.now() - t;
    const E = gridEdges(n, n, 4), s8 = Int8Array.from(s, v => (v === 1 ? 1 : v ? -1 : 0));
    t = performance.now(); solveSeg({ n: n * n, D0: new Float64Array(n * n), D1: new Float64Array(n * n), E, w: contrastWeights(I, E, 20), lambda: 1, seeds: s8 }); const c = performance.now() - t;
    runs.push([a, b, c]);
  }
  tim[n] = [0, 1, 2].map(j => runs.map(r => r[j]).sort((x, y) => x - y)[1]);
}
out.timing = tim;
console.log("timing", JSON.stringify(tim));
fs.writeFileSync(`${raw}/ws.json`, JSON.stringify(out));
