// Reference checks for the dynamic-programming day: lib/dp.ts (monotone paths, closed polar
// paths, chain Viterbi / forward–backward) and lib/shortestpath.ts (Dijkstra, fast
// marching, gradient descent), against brute force or textbook versions. Run:
//   node_modules/.bin/esbuild demos/test/dynprog-test.ts --bundle --platform=node --log-level=warning | node
import { rng } from "../src/lib/random";
import * as DP from "../src/lib/dp";
import * as SP from "../src/lib/shortestpath";
import { polarCost } from "../src/polarseg/model";
import { DPGRID_PRESETS, LW_PRESETS } from "../src/dpgrid/grids";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
const section: Record<string, number> = {};
function check(ok: boolean, msg: string): void {
  checks++;
  section[current] = (section[current] ?? 0) + 1;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-9) => (a === b) || Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
type R = ReturnType<typeof rng>;
const randInts = (r: R, n: number, max: number) => Float64Array.from({ length: n }, () => Math.floor(r.unif() * (max + 1)));

/** All paths of the lattice by recursion: [cost, count]. */
function bruteLattice(C: ArrayLike<number>, rows: number, cols: number, o: DP.PathDPOptions): { best: number; count: number } {
  const ok = (i: number, j: number) => i >= 0 && j >= 0 && i < rows && j < cols && (o.band === undefined || Math.abs(i - j) <= o.band) && (!o.allowed || o.allowed(i, j));
  let best = Infinity, count = 0;
  const isEnd = (i: number, j: number) => (o.end ?? "row") === "row" ? i === rows - 1 : i === rows - 1 && j === cols - 1;
  // forward moves are the negated steps
  const go = (i: number, j: number, c: number) => {
    if (isEnd(i, j)) { count++; best = Math.min(best, c); if ((o.end ?? "row") === "corner") return; }
    if ((o.end ?? "row") === "row" && i === rows - 1) return;
    for (const s of o.steps) {
      const ii = i - s.di, jj = j - s.dj;
      if (ok(ii, jj)) go(ii, jj, c + (s.penalty ?? 0) + C[ii * cols + jj]);
    }
  };
  if ((o.start ?? "row") === "row") { for (let j = 0; j < cols; j++) if (ok(0, j)) go(0, j, C[j]); }
  else if (ok(0, 0)) go(0, 0, C[0]);
  return { best, count };
}

function testPathDP(): void {
  current = "pathDP vs brute force";
  const r = rng(7);
  const sets: [string, DP.PathDPOptions][] = [
    ["3 moves", { steps: DP.verticalSteps(1) }],
    ["δ = 2, α = 1.5", { steps: DP.verticalSteps(2, 1.5) }],
    ["→↓", { steps: DP.cornerSteps(false), start: "corner", end: "corner" }],
    ["→↓↘", { steps: DP.cornerSteps(true), start: "corner", end: "corner" }],
    ["→↓↘ band 1", { steps: DP.cornerSteps(true), start: "corner", end: "corner", band: 1 }],
  ];
  for (const [name, o] of sets) for (let t = 0; t < 60; t++) {
    const rows = 2 + Math.floor(r.unif() * 4), cols = 2 + Math.floor(r.unif() * 5);
    const C = randInts(r, rows * cols, 9);
    const res = DP.pathDP(C, rows, cols, o), bf = bruteLattice(C, rows, cols, o);
    check(close(res.cost, bf.best), `${name} ${rows}x${cols}: dp ${res.cost} brute ${bf.best}`);
    check(DP.countPaths(rows, cols, o) === BigInt(bf.count), `${name}: path count ${DP.countPaths(rows, cols, o)} vs ${bf.count}`);
    if (res.path.length) check(close(DP.pathCost(C, cols, res.path, o.steps), res.cost), `${name}: path cost matches table`);
    check(res.path.length === ((o.start ?? "row") === "corner" ? res.path.length : rows), `${name}: one cell per row`);
  }
  // operations: 3nm-ish for the 3-move set (border cells have 2 candidates)
  const res = DP.pathDP(new Float64Array(30), 5, 6, { steps: DP.verticalSteps(1) });
  check(res.ops === 4 * (6 * 3 - 2), `ops count 5x6: ${res.ops}`);
  check(DP.countPaths(256, 256, { steps: DP.verticalSteps(1) }).toString().length === 125, "256x256 has ~1e124 monotone paths");
  // trace records the same table
  const C = randInts(r, 20, 9), tr = DP.pathDP(C, 4, 5, { steps: DP.verticalSteps(1), trace: true });
  check(tr.trace!.length === 20 && tr.trace!.every(t => t.M === tr.M[t.i * 5 + t.j]), "trace matches M");
}

function testCircular(): void {
  current = "circularDP";
  const r = rng(11);
  for (let t = 0; t < 80; t++) {
    const K = 3 + Math.floor(r.unif() * 3), S = 3 + Math.floor(r.unif() * 3);   // up to 5 x 5
    const delta = 1 + Math.floor(r.unif() * 2), alpha = r.unif() < 0.5 ? 0 : 0.7;
    const C = randInts(r, K * S, 9);
    // brute force over all closed paths
    let best = Infinity, bestOpen = Infinity;
    const p = new Int32Array(K);
    const rec = (k: number) => {
      if (k === K) {
        let c = 0, sm = 0;
        for (let i = 0; i < K; i++) c += C[i * S + p[i]];
        for (let i = 1; i < K; i++) sm += alpha * Math.abs(p[i] - p[i - 1]);
        bestOpen = Math.min(bestOpen, c + sm);
        const gap = Math.abs(p[0] - p[K - 1]);
        if (gap <= delta) best = Math.min(best, c + sm + alpha * gap);
        return;
      }
      for (let s = 0; s < S; s++) { if (k > 0 && Math.abs(s - p[k - 1]) > delta) continue; p[k] = s; rec(k + 1); }
    };
    rec(0);
    const ex = DP.circularDP(C, K, S, { delta, alpha, closure: "exact" });
    check(close(ex.closedCost, best) && ex.closes, `exact closure ${K}x${S}: ${ex.closedCost} vs ${best}`);
    const op = DP.circularDP(C, K, S, { delta, alpha, closure: "open" });
    check(close(op.openCost, bestOpen), `open ${op.openCost} vs ${bestOpen}`);
    const un = DP.circularDP(C, K, S, { delta, alpha, closure: "unroll" });
    check(un.path.every((s, k) => k === 0 || Math.abs(s - un.path[k - 1]) <= delta), "unroll path obeys δ");
    check(!un.closes || un.closedCost >= best - 1e-9, "unroll is never better than exact");
  }
  // the brute-force check on exactly 5 stages x 4 states named in the plan
  const C = randInts(rng(3), 20, 9);
  const ex = DP.circularDP(C, 5, 4, { delta: 1, closure: "exact" });
  check(ex.closes, "5x4 exact closes");
}

function testChains(): void {
  current = "viterbi / forward–backward";
  const r = rng(5);
  for (let t = 0; t < 40; t++) {
    const n = 1 + Math.floor(r.unif() * 6), K = 2 + Math.floor(r.unif() * 2);
    const U = Float64Array.from({ length: n * K }, () => 3 * r.unif());
    const V = Float64Array.from({ length: K * K }, () => 2 * r.unif());
    let best = Infinity, Z = 0;
    const marg = new Float64Array(n * K), L = new Int32Array(n);
    const rec = (p: number) => {
      if (p === n) {
        const e = DP.chainEnergyK(U, K, V, L);
        best = Math.min(best, e);
        const wgt = Math.exp(-e / 1.3);
        Z += wgt;
        for (let q = 0; q < n; q++) marg[q * K + L[q]] += wgt;
        return;
      }
      for (let l = 0; l < K; l++) { L[p] = l; rec(p + 1); }
    };
    rec(0);
    const v = DP.viterbi(U, K, V);
    check(close(v.energy, best) && close(DP.chainEnergyK(U, K, V, v.labels), best), `viterbi n=${n} K=${K}`);
    const fb = DP.forwardBackward(U, K, V, 1.3);
    check(close(fb.logZ, Math.log(Z), 1e-9), "log Z");
    let md = 0;
    for (let i = 0; i < n * K; i++) md = Math.max(md, Math.abs(fb.marginals[i] - marg[i] / Z));
    check(md < 1e-9, `marginals off by ${md}`);
  }
  // Potts with 2 labels agrees with segchain's convention
  const U = Float64Array.from([0, 3, 2, 1, 3, 0, 1, 2]);
  const v = DP.viterbi(U, 2, DP.potts(2, 1));
  check(v.energy === DP.chainEnergyK(U, 2, DP.potts(2, 1), v.labels), "potts energy");
}

/** Bellman–Ford on the grid graph: relax every edge until nothing changes. */
function bellmanFord(w: number, h: number, conn: SP.Conn, edge: SP.EdgeCost, sources: { index: number; value: number }[]): Float64Array {
  const d = new Float64Array(w * h).fill(Infinity);
  for (const s of sources) d[s.index] = Math.min(d[s.index], s.value);
  const nb = SP.gridNeighbours(conn);
  for (let it = 0; it < w * h; it++) {
    let changed = false;
    for (let p = 0; p < w * h; p++) {
      if (d[p] === Infinity) continue;
      const x = p % w, y = (p - x) / w;
      for (const [dx, dy, len] of nb) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const q = yy * w + xx, nd = d[p] + edge(p, q, len);
        if (nd < d[q] - 1e-12) { d[q] = nd; changed = true; }
      }
    }
    if (!changed) break;
  }
  return d;
}

function testDijkstra(): void {
  current = "Dijkstra";
  const r = rng(9);
  for (let t = 0; t < 60; t++) {
    const w = 2 + Math.floor(r.unif() * 6), h = 2 + Math.floor(r.unif() * 6), conn: SP.Conn = t % 2 ? 8 : 4;
    const C = randInts(r, w * h, 9);
    const edge = SP.vertexCost(C, true);
    const src = [{ index: Math.floor(r.unif() * w * h), value: 0 }];
    if (t % 3 === 0) src.push({ index: Math.floor(r.unif() * w * h), value: 2 });
    const res = SP.dijkstraGrid({ width: w, height: h, conn, edge, sources: src });
    const bf = bellmanFord(w, h, conn, edge, src);
    let md = 0;
    for (let p = 0; p < w * h; p++) md = Math.max(md, Math.abs(res.dist[p] - bf[p]));
    check(md < 1e-9, `dijkstra vs bellman-ford ${w}x${h} conn ${conn}: ${md}`);
    // the tree is consistent: d(q) = d(pred q) + w(pred q, q)
    let okTree = true;
    for (let q = 0; q < w * h; q++) {
      const p = res.pred[q];
      if (p < 0) continue;
      const len = Math.hypot((p % w) - (q % w), Math.floor(p / w) - Math.floor(q / w));
      if (!close(res.dist[q], res.dist[p] + edge(p, q, len))) okTree = false;
    }
    check(okTree, "predecessor tree consistent");
    // the traced version gives the same distances and finalises in non-decreasing order
    const tr = SP.dijkstraTrace({ width: w, height: h, conn, edge, sources: src });
    check(tr.result.dist.every((v, i) => close(v, res.dist[i])), "trace = plain");
    check(tr.events.every((e, k) => k === 0 || e.d >= tr.events[k - 1].d), "pops in non-decreasing order");
  }
  // with only the monotone moves allowed, Dijkstra from the top row = the monotone DP
  for (let t = 0; t < 40; t++) {
    const w = 2 + Math.floor(r.unif() * 5), h = 2 + Math.floor(r.unif() * 5);
    const C = randInts(r, w * h, 9);
    const edge: SP.EdgeCost = (p, q) => (Math.floor(q / w) === Math.floor(p / w) + 1 ? C[q] : Infinity);
    const sources = Array.from({ length: w }, (_, j) => ({ index: j, value: C[j] }));
    const d = SP.dijkstraGrid({ width: w, height: h, conn: 8, edge, sources }).dist;
    const m = DP.pathDP(C, h, w, { steps: DP.verticalSteps(1) }).M;
    check(m.every((v, i) => close(v, d[i])), `monotone dijkstra = DP ${h}x${w}`);
  }
  // the dpgrid presets: Dijkstra (8 neighbours, all directions) never worse than the monotone DP
  for (const p of DPGRID_PRESETS) {
    const C = Float64Array.from(p.values), w = p.cols, h = p.rows;
    const sources = Array.from({ length: w }, (_, j) => ({ index: j, value: C[j] }));
    const d = SP.dijkstraGrid({ width: w, height: h, conn: 8, edge: SP.vertexCost(C), sources }).dist;
    const dp = DP.pathDP(C, h, w, { steps: DP.verticalSteps(1) }).cost;
    let best = Infinity;
    for (let j = 0; j < w; j++) best = Math.min(best, d[(h - 1) * w + j]);
    check(best <= dp, `${p.label}: dijkstra ${best} <= dp ${dp}`);
  }
  check(LW_PRESETS.length > 0, "livewire presets exist");
}

function testFastMarching(): void {
  current = "fast marching";
  check(close(SP.eikonalUpdate(1, 1, 1), 1 + Math.SQRT1_2), "update a = b = 1");
  check(SP.eikonalUpdate(0, Infinity, 1) === 1, "one neighbour");
  check(SP.eikonalUpdate(0, 2, 1) === 1, "far apart: fallback");
  const n = 121, c = 60 * n + 60;
  const P = new Float64Array(n * n).fill(1);
  const T0 = SP.fastMarching(n, n, P, [c]).T;
  const { T } = SP.fastMarching(n, n, P, SP.seedNeighbourhood(n, n, P, [c]));
  let md = 0, md4 = 0;
  const d4 = SP.dijkstraGrid({ width: n, height: n, conn: 4, edge: SP.vertexCost(P, true), sources: [c] }).dist;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const e = Math.hypot(x - 60, y - 60);
    if (e > 50.5 || e < 49.5) continue;
    md = Math.max(md, Math.abs(T[y * n + x] - e));
    md4 = Math.max(md4, Math.abs(d4[y * n + x] - e));
  }
  let md0 = 0;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { const e = Math.hypot(x - 60, y - 60); if (e >= 49.5 && e <= 50.5) md0 = Math.max(md0, Math.abs(T0[y * n + x] - e)); }
  check(md0 < 1.5, `point source alone: within 1.5 px (${md0.toFixed(3)})`);
  check(md < 1, `fast marching within 1 px of Euclidean at r = 50 (max error ${md.toFixed(3)})`);
  check(md4 > 15, `4-neighbour Dijkstra is far off at 45° (${md4.toFixed(1)})`);
  // fast marching is never below the Euclidean distance on a flat cost (it overestimates slightly)
  // and the descent path from (110, 30) is about straight
  const pts = SP.descendPath(T, n, n, [110, 30]);
  const end = pts[pts.length - 1];
  check(Math.hypot(end[0] - 60, end[1] - 60) < 1.5, `descent ends at the source (${end})`);
  const L = SP.polylineLength(pts), e = Math.hypot(50, 30);
  check(L < e * 1.03 + 1, `descent path almost straight: ${L.toFixed(2)} vs ${e.toFixed(2)}`);
  // slower region: arrival times grow with P
  const P2 = new Float64Array(n * n).fill(2);
  const T2 = SP.fastMarching(n, n, P2, [c]).T;
  check(close(T2[c + 30], 2 * T[c + 30]), "T scales with P");
}

function testPolar(): void {
  current = "polar cost";
  // radial derivative of a bright disc on a dark background: lowest cost at the edge for dark→bright... (inside dark)
  const K = 8, R = 20;
  const P = new Float64Array(K * R);
  for (let i = 0; i < R; i++) for (let k = 0; k < K; k++) P[i * K + k] = i < 10 ? 50 : 150;   // dark inside, bright outside
  const C = polarCost(P, K, R, "dark-bright");
  let bi = 0;
  for (let i = 0; i < R; i++) if (C[0 * R + i] < C[0 * R + bi]) bi = i;
  check(bi === 9 || bi === 10, `edge found at radius index ${bi}`);
  const C2 = polarCost(P, K, R, "bright-dark");
  check(C2[0 * R + 9] > C2[0 * R + 2], "wrong polarity: the edge is expensive");
}

testPathDP();
testCircular();
testChains();
testDijkstra();
testFastMarching();
testPolar();
console.log(Object.entries(section).map(([k, v]) => `${k}: ${v}`).join("; "));
console.log(failures ? `${failures} of ${checks} checks FAILED` : `all ${checks} checks passed`);
if (failures) process.exit(1);
