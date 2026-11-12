// Correctness tests for the demo algorithms, run by `make test`:
// esbuild bundles this file for node and pipes it into node. No DOM, no dependencies.
//  - max-flow (Dinic and the step-through Edmonds-Karp) against brute-force enumeration
//    of all s-t cuts on random graphs with up to 12 nodes;
//  - graph-cut segmentation against brute force over all labellings of small images;
//  - the dynamic programme of segchain against brute force on short chains;
//  - the spectral code of ncut against a dense eigen-solver and brute force.
import { FlowGraph, edmondsKarpSteps, cutCost, Edge, EPS } from "../src/lib/maxflow";
import { gridEdges, contrastWeights, solveSeg, segEnergy, SegProblem, Nbhd } from "../src/lib/graph";
import { rng } from "../src/lib/ui";
import { chainDP, chainEnergy } from "../src/segchain/dp";
import { tests as ncutTests } from "./ncut-test";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 20) console.log("FAIL: " + msg); }
}
const close = (a: number, b: number, tol = 1e-7) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

// ---- max-flow against all cuts ----
function bruteMinCut(n: number, edges: Edge[], s: number, t: number): number {
  const others = [...Array(n).keys()].filter(v => v !== s && v !== t);
  let best = Infinity;
  for (let m = 0; m < 1 << others.length; m++) {
    const S = new Array<boolean>(n).fill(false);
    S[s] = true;
    others.forEach((v, k) => { if (m >> k & 1) S[v] = true; });
    best = Math.min(best, cutCost(edges, S));
  }
  return best;
}

function testMaxflow(): void {
  const r = rng(12345);
  for (let trial = 0; trial < 600; trial++) {
    const n = 2 + Math.floor(r.unif() * 11);           // 2..12 nodes
    const dens = 0.15 + 0.7 * r.unif();
    const ints = trial % 2 === 0;
    const edges: Edge[] = [];
    for (let u = 0; u < n; u++) for (let v = 0; v < n; v++) {
      if (u === v || r.unif() > dens) continue;
      const c = ints ? Math.floor(r.unif() * 10) : r.unif() * 5;
      edges.push({ u, v, c });
    }
    const s = 0, t = n - 1;
    const best = bruteMinCut(n, edges, s, t);
    // Dinic
    const g = new FlowGraph(n);
    const ids = edges.map(e => g.addEdge(e.u, e.v, e.c));
    const f = g.maxflow(s, t);
    check(close(f, best), `dinic trial ${trial}: flow ${f} vs brute min cut ${best}`);
    const S = Array.from(g.sourceSide(s), x => x === 1);
    check(S[s] && !S[t], `dinic trial ${trial}: s and t separated`);
    check(close(cutCost(edges, S), best), `dinic trial ${trial}: cut from residual graph is minimal`);
    // feasibility: capacity and conservation
    const net = new Array<number>(n).fill(0);
    ids.forEach((e, k) => {
      const fl = g.flow(e);
      check(fl >= -EPS && fl <= edges[k].c + 1e-7, `dinic trial ${trial}: capacity`);
      net[edges[k].u] -= fl; net[edges[k].v] += fl;
    });
    for (let v = 0; v < n; v++) if (v !== s && v !== t) check(Math.abs(net[v]) < 1e-7, `dinic trial ${trial}: conservation`);
    // Edmonds-Karp (step-through demo)
    const ek = edmondsKarpSteps(n, edges, s, t);
    check(close(ek.total, best), `EK trial ${trial}: flow ${ek.total} vs ${best}`);
    check(close(cutCost(edges, ek.S), best), `EK trial ${trial}: residual cut is minimal`);
    for (const st of ek.steps) check(st.bottleneck > 0, `EK trial ${trial}: positive bottleneck`);
  }
  // undirected (two-way) edges, as the n-links use
  for (let trial = 0; trial < 300; trial++) {
    const n = 3 + Math.floor(r.unif() * 10);
    const g = new FlowGraph(n);
    const edges: Edge[] = [];
    for (let u = 0; u < n; u++) for (let v = u + 1; v < n; v++) {
      if (r.unif() > 0.4) continue;
      const c = r.unif() * 3, c2 = r.unif() < 0.5 ? c : r.unif() * 3;
      g.addEdge(u, v, c, c2);
      edges.push({ u, v, c }, { u: v, v: u, c: c2 });
    }
    const best = bruteMinCut(n, edges, 0, n - 1);
    check(close(g.maxflow(0, n - 1), best), `two-way trial ${trial}`);
    check(close(cutCost(edges, Array.from(g.sourceSide(0), x => x === 1)), best), `two-way cut trial ${trial}`);
  }
}

// ---- graph-cut segmentation against all labellings ----
function testSeg(): void {
  const r = rng(777);
  for (let trial = 0; trial < 400; trial++) {
    const rows = 1 + Math.floor(r.unif() * 3), cols = 1 + Math.floor(r.unif() * 4);
    const n = rows * cols;
    const nb: Nbhd = trial % 2 ? 4 : 8;
    const I = Array.from({ length: n }, () => Math.floor(r.unif() * 4));
    const E = gridEdges(rows, cols, nb);
    const w = contrastWeights(I, E, 0.5 + 2 * r.unif());
    const D0 = I.map(v => v * v), D1 = I.map(v => (3 - v) ** 2);
    const seeds = new Int8Array(n);
    if (trial % 3 === 0) seeds[Math.floor(r.unif() * n)] = 1;
    if (trial % 3 === 0) seeds[Math.floor(r.unif() * n)] = -1;
    const P: SegProblem = { n, D0, D1, E, w, lambda: 4 * r.unif(), seeds };
    let best = Infinity;
    for (let m = 0; m < 1 << n; m++) {
      const L = Array.from({ length: n }, (_, p) => m >> p & 1);
      if (L.some((l, p) => (seeds[p] === 1 && l !== 1) || (seeds[p] === -1 && l !== 0))) continue;
      best = Math.min(best, segEnergy(P, L).total);
    }
    const res = solveSeg(P);
    if (!Number.isFinite(best)) continue;   // contradictory seeds on the same pixel
    check(close(res.energy.total, best), `seg trial ${trial}: energy ${res.energy.total} vs brute ${best}`);
    check(close(res.flow, best), `seg trial ${trial}: flow ${res.flow} vs brute ${best}`);
    check(res.L.every((l, p) => !(seeds[p] === 1 && l !== 1) && !(seeds[p] === -1 && l !== 0)), `seg trial ${trial}: seeds kept`);
  }
}

// ---- segchain DP against all labellings ----
function testChain(): void {
  const r = rng(99);
  for (let trial = 0; trial < 500; trial++) {
    const n = 1 + Math.floor(r.unif() * 12);
    const D0 = Array.from({ length: n }, () => r.unif() * 5);
    const D1 = Array.from({ length: n }, () => r.unif() * 5);
    const lam = 4 * r.unif();
    let best = Infinity;
    for (let m = 0; m < 1 << n; m++) {
      const L = Array.from({ length: n }, (_, p) => m >> p & 1);
      best = Math.min(best, chainEnergy(D0, D1, lam, L).total);
    }
    const L = chainDP(D0, D1, lam);
    check(close(chainEnergy(D0, D1, lam, L).total, best), `chain trial ${trial}`);
  }
}

function timeIt(): void {
  // a 128 x 128 problem like the graphcut demo, for information
  const N = 128, r = rng(5);
  const I = new Float64Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
    I[y * N + x] = ((x - 60) ** 2 + (y - 64) ** 2 < 900 ? 170 : 80) + 40 * r.norm();
  for (const nb of [4, 8] as Nbhd[]) {
    const E = gridEdges(N, N, nb);
    const w = contrastWeights(I, E, 30);
    const D0 = I.map(v => (v - 80) ** 2 / 3200), D1 = I.map(v => (v - 170) ** 2 / 3200);
    const t0 = Date.now();
    const reps = 5;
    for (let k = 0; k < reps; k++) solveSeg({ n: N * N, D0, D1, E, w, lambda: 3 });
    console.log(`  (128 x 128, ${nb}-neighbourhood: ${((Date.now() - t0) / reps).toFixed(1)} ms per max-flow)`);
  }
}

testMaxflow();
testSeg();
testChain();
for (const [name, fn] of ncutTests) { fn(check, close); void name; }
timeIt();
console.log(`${checks - failures} of ${checks} checks passed`);
if (failures) { console.log(`${failures} FAILED`); process.exit(1); }
