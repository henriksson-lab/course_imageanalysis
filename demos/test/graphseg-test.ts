// Tests for the graph-based segmentation day: lib/randomwalker.ts, lib/mst.ts,
// lib/hierarchy.ts, lib/superpixels.ts, and every by-hand number of the exercises. Run:
//   node_modules/.bin/esbuild demos/test/graphseg-test.ts --bundle --platform=node --log-level=warning | node
import * as RW from "../src/lib/randomwalker";
import * as M from "../src/lib/mst";
import * as H from "../src/lib/hierarchy";
import * as SP from "../src/lib/superpixels";
import { solveSeg, gridEdges } from "../src/lib/graph";
import { floodWatershed, hMinimaMarkers, watershed } from "../src/lib/watershed";
import { label2d } from "../src/lib/label";
import { rng } from "../src/lib/random";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log("FAIL: " + msg); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const closeArr = (a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-9) => a.length === b.length && Array.from(a).every((v, i) => close(v, b[i], tol));
const r = rng(11);
const NaNs = (n: number) => new Array(n).fill(NaN);

// ---- random walker on chains (unit 1 by hand) ----------------------------------------------
function chainFixed(n: number, obj: number, bg: number): number[] { const f = NaNs(n); f[obj] = 1; f[bg] = 0; return f; }
{
  // (a) 5 nodes, weights 1, 1, 0.25, 1
  const w = [1, 1, 0.25, 1];
  const v = RW.chainVoltages(w, chainFixed(5, 0, 4));
  check(closeArr(v, [1, 6 / 7, 5 / 7, 1 / 7, 0]), `(a) voltages ${Array.from(v)}`);
  check(closeArr(RW.harmonicDense(RW.chainGraph(w), chainFixed(5, 0, 4)), v), "(a) dense = resistor formula");
  // graph cut on the same chain: cuts the 0.25 edge
  const E = gridEdges(1, 5, 4);
  const seeds = new Int8Array(5); seeds[0] = 1; seeds[4] = -1;
  const gc = solveSeg({ n: 5, D0: new Float64Array(5), D1: new Float64Array(5), E, w, lambda: 1, seeds });
  check(Array.from(gc.L).join("") === "11100", `(a) graph cut ${Array.from(gc.L).join("")}`);
  // (b) one free node
  const star = (wo: number, b1: number, b2: number) => RW.harmonicDense({ n: 4, p: Int32Array.from([0, 0, 0]), q: Int32Array.from([1, 2, 3]), w: Float64Array.from([wo, b1, b2]) }, [NaN, 1, 0, 0])[0];
  check(close(star(3, 1, 1), 3 / 5), "(b) 3/5");
  check(close(star(3, 2, 2), 3 / 7), "(b) 3/7");
  // (c) 7 nodes 0.2, 1, 1, 1, 1, 0.25
  const wc = [0.2, 1, 1, 1, 1, 0.25];
  const vc = RW.chainVoltages(wc, chainFixed(7, 0, 6));
  check(closeArr(vc, [1, 8 / 13, 7 / 13, 6 / 13, 5 / 13, 4 / 13, 0]), `(c) ${Array.from(vc)}`);
  const Ec = gridEdges(1, 7, 4), sc = new Int8Array(7); sc[0] = 1; sc[6] = -1;
  const gcc = solveSeg({ n: 7, D0: new Float64Array(7), D1: new Float64Array(7), E: Ec, w: wc, lambda: 1, seeds: sc });
  check(Array.from(gcc.L).join("") === "1000000", `(c) graph cut keeps node 1 only: ${Array.from(gcc.L).join("")}`);
  // (d) seed moved to node 3 (index 2)
  const vd = RW.harmonicDense(RW.chainGraph(wc), chainFixed(7, 2, 6));
  check(closeArr(vd, [1, 1, 1, 6 / 7, 5 / 7, 4 / 7, 0]), `(d) ${Array.from(vd)}`);
  const sd = new Int8Array(7); sd[2] = 1; sd[6] = -1;
  const gcd = solveSeg({ n: 7, D0: new Float64Array(7), D1: new Float64Array(7), E: Ec, w: wc, lambda: 1, seeds: sd });
  check(Array.from(gcd.L).join("") === "1111110", `(d) graph cut ${Array.from(gcd.L).join("")}`);
  // (e) 3×3 uniform, seeds at the corners: anti-diagonal = 0.5
  const G = RW.gridGraph(new Float64Array(9), 3, 3, Infinity, 4);
  const f = NaNs(9); f[0] = 1; f[8] = 0;
  const ve = RW.harmonicDense(G, f);
  check(close(ve[2], 0.5) && close(ve[4], 0.5) && close(ve[6], 0.5), "(e) anti-diagonal 0.5");
  check(close(ve[1], ve[3]) && ve[1] > 0.5, `(e) symmetric ${ve[1]}`);
  console.log(`  unit 1 (e): 3×3 values ${Array.from(ve).map(v => v.toFixed(4)).join(" ")}`);
  // ws-msf (b): resistances 2, 5, 1, 3, 6, 2  (w = 1/d)
  const wb = [2, 5, 1, 3, 6, 2].map(d => 1 / d);
  const vb = RW.chainVoltages(wb, chainFixed(7, 0, 6));
  check(closeArr(vb, [1, 17 / 19, 12 / 19, 11 / 19, 8 / 19, 2 / 19, 0]), `ws-msf (b) walker ${Array.from(vb)}`);
  const gb = solveSeg({ n: 7, D0: new Float64Array(7), D1: new Float64Array(7), E: Ec, w: wb, lambda: 1, seeds: sc });
  check(Array.from(gb.L).join("") === "1111100", `ws-msf (b) graph cut cuts the 6-edge ${Array.from(gb.L).join("")}`);
}

// ---- random walker: CG vs dense, 3 labels sum to 1, max principle, walks -------------------
{
  for (let trial = 0; trial < 20; trial++) {
    const w = 3 + Math.floor(r.unif() * 6), h = 3 + Math.floor(r.unif() * 6), n = w * h;
    const I = Float64Array.from({ length: n }, () => Math.floor(r.unif() * 100));
    const seeds = new Int32Array(n);
    seeds[0] = 1; seeds[n - 1] = 2; seeds[Math.floor(n / 2)] = 3;
    const res = RW.randomWalker(I, w, h, seeds, 3, { sigma: 30, tol: 1e-12 });
    for (let k = 1; k <= 2; k++) {
      const f = Array.from(seeds, s => (s ? (s === k ? 1 : 0) : NaN));
      const ref = RW.harmonicDense(res.graph, f);
      check(closeArr(res.prob[k - 1], ref, 1e-7), `CG = dense (trial ${trial}, label ${k})`);
    }
    let okSum = true, okRange = true;
    for (let v = 0; v < n; v++) {
      const s = res.prob[0][v] + res.prob[1][v] + res.prob[2][v];
      if (!close(s, 1, 1e-9)) okSum = false;
      for (let k = 0; k < 3; k++) if (res.prob[k][v] < -1e-12 || res.prob[k][v] > 1 + 1e-12) okRange = false;
    }
    check(okSum, `3 labels sum to 1 (trial ${trial})`);
    check(okRange, `probabilities in [0, 1] (trial ${trial})`);
    // the 3rd label solved directly also matches (sum to 1 by linearity)
    const f3 = Array.from(seeds, s => (s ? (s === 3 ? 1 : 0) : NaN));
    check(closeArr(res.prob[2], RW.harmonicDense(res.graph, f3), 1e-7), `third label = 1 − others (trial ${trial})`);
    // priors: probabilities still sum to 1
    const pri = RW.seedPriors(I, seeds, 3, 0, 100);
    const rp = RW.walkerGraph(res.graph, seeds, 3, { lambda: 0.5, priors: pri, tol: 1e-12 });
    let s2 = true;
    for (let v = 0; v < n; v++) if (!close(rp.prob[0][v] + rp.prob[1][v] + rp.prob[2][v], 1, 1e-8)) s2 = false;
    check(s2, `with priors the probabilities sum to 1 (trial ${trial})`);
  }
  // no seeds at all, priors only: x = (I + L/λ)^-1 prior, in [0, 1]
  // simulated walks: within 3 standard errors
  const w = 6, h = 6, n = 36;
  const I = Float64Array.from({ length: n }, (_, i) => (i % 6 < 3 ? 10 : 60) + (i % 5));
  const seeds = new Int32Array(n); seeds[0] = 1; seeds[35] = 2;
  const res = RW.randomWalker(I, w, h, seeds, 2, { sigma: 30, tol: 1e-12 });
  const A = RW.csr(res.graph);
  for (const start of [7, 14, 21, 28]) {
    const N = 4000, wk = RW.simulateWalks(A, seeds, start, N, rng(5 + start));
    const p = res.prob[0][start], est = wk.counts[1] / N, se = Math.sqrt(p * (1 - p) / N);
    check(Math.abs(est - p) < 3 * se + 1e-9, `walks from ${start}: ${est.toFixed(3)} vs ${p.toFixed(3)}`);
  }
  // Gauss–Seidel converges to the same solution
  const x = new Float64Array(n), fx = Array.from(seeds, s => (s ? (s === 1 ? 1 : 0) : NaN));
  for (let it = 0; it < 3000; it++) RW.gaussSeidelSweep(A, x, fx);
  check(closeArr(x, res.prob[0], 1e-6), "Gauss–Seidel converges to the walker");
}

// WalkerSolver (resumable) = walkerGraph
{
  const w = 12, h = 10, n = w * h;
  const I = Float64Array.from({ length: n }, (_, i) => (i % w < 6 ? 30 : 90) + ((i * 7) % 11));
  const seeds = new Int32Array(n); seeds[0] = 1; seeds[n - 1] = 2; seeds[50] = 3;
  const G = RW.gridGraph(I, w, h, 15, 8);
  const ref = RW.walkerGraph(G, seeds, 3, { tol: 1e-12 });
  const ws = new RW.WalkerSolver(G, seeds, 3, { tol: 1e-12 });
  while (!ws.done) ws.step(50);
  const P = ws.prob();
  check([0, 1, 2].every(k => closeArr(P[k], ref.prob[k], 1e-8)), "WalkerSolver = walkerGraph");
}

// ---- spanning trees ------------------------------------------------------------------------
function randGraph(n: number, dens: number, ints = false): M.DGraph {
  const P: number[] = [], Q: number[] = [], D: number[] = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (j === i + 1 || r.unif() < dens) { P.push(i); Q.push(j); D.push(ints ? Math.floor(r.unif() * 6) : r.unif()); }
  return { n, p: Int32Array.from(P), q: Int32Array.from(Q), d: Float64Array.from(D) };
}
function bruteMST(G: M.DGraph): number {
  const m = G.p.length;
  let best = Infinity;
  const pick: number[] = [];
  const rec = (start: number) => {
    if (pick.length === G.n - 1) {
      const uf = new (class { p = Array.from({ length: G.n }, (_, i) => i); f(x: number): number { while (this.p[x] !== x) x = this.p[x]; return x; } })();
      let tot = 0;
      for (const e of pick) { const a = uf.f(G.p[e]), b = uf.f(G.q[e]); if (a === b) return; uf.p[a] = b; tot += G.d[e]; }
      best = Math.min(best, tot); return;
    }
    for (let e = start; e < m; e++) { pick.push(e); rec(e + 1); pick.pop(); }
  };
  rec(0);
  return best;
}
{
  for (let t = 0; t < 40; t++) {
    const n = 3 + Math.floor(r.unif() * 5), G = randGraph(n, 0.6);
    const K = M.kruskal(G);
    if (G.p.length <= 14) check(close(K.total, bruteMST(G)), `Kruskal = brute force (n ${n})`);
    // Prim from one marker gives a spanning tree of the same weight
    const mk = new Int32Array(n); mk[0] = 1;
    const P = M.primForest(G, mk);
    let tot = 0; for (let e = 0; e < G.p.length; e++) if (P.inForest[e]) tot += G.d[e];
    check(close(tot, K.total), `Prim = Kruskal total (${tot} vs ${K.total})`);
    // minimax: the tree path's heaviest edge equals the minimax distance
    const mm = M.minimaxFrom(G, 0);
    for (let v = 1; v < n; v++) { const pth = M.treePath(G, K.inTree, 0, v)!; check(close(G.d[pth.maxEdge], mm[v]), "MST path is a minimax path"); }
  }
  // single linkage = components of {d < t} = label2d on a 1D chain thresholded
  for (let t = 0; t < 20; t++) {
    const n = 12, vals = Array.from({ length: n }, () => Math.floor(r.unif() * 30));
    const G = M.gridDissim([vals], n, 1);
    const thr = 1 + Math.floor(r.unif() * 15);
    const sl = M.singleLinkage(G, thr);
    // MST minus edges ≥ t
    const K = M.kruskal(G);
    const keep = { ...G, d: G.d.map((d, e) => (K.inTree[e] && d < thr ? 0 : 1e9)) };
    const sl2 = M.singleLinkage(keep, 1);
    check(sl.labels.join() === sl2.labels.join(), "single linkage = MST minus heavy edges");
  }
}

// ---- FH by hand (unit 2) -------------------------------------------------------------------
{
  const b = [0, 6, 0, 6, 20, 20, 20, 20, 24, 24, 24, 24];
  const G = M.gridDissim([b], 12, 1);
  check(Array.from(G.d).join() === "6,6,6,14,0,0,0,4,0,0,0", `(b) edges ${Array.from(G.d)}`);
  const fh = M.felzenszwalb(G, 8);
  check(fh.count === 3 && fh.labels.join() === "1,1,1,1,2,2,2,2,3,3,3,3", `(b) FH k=8 ${fh.labels.join()}`);
  for (let t = 0; t <= 30; t += 0.5) check(M.singleLinkage(G, t).count !== 3, `(b) single linkage never 3 regions (t ${t})`);
  const tr = M.fhTrace(G, 8);
  console.log("  unit 2 (b) FH trace: " + tr.map(s => `d${s.d}:${s.merge ? "M" : "k"}(${s.thrA.toFixed(2)},${s.thrB.toFixed(2)})`).join(" "));
  const c = [10, 11, 13, 12, 22, 23, 21, 24];
  const Gc = M.gridDissim([c], 8, 1);
  check(M.felzenszwalb(Gc, 4).labels.join() === "1,1,1,1,2,2,2,2", "(c) k = 4: two regions");
  check(M.felzenszwalb(Gc, 32).count === 1, "(c) k = 32: one region");
  check(M.felzenszwalb(Gc, 31.9).count === 2, "(c) k = 31.9: two regions");
  console.log("  unit 2 (c) trace k=4: " + M.fhTrace(Gc, 4).map(s => `d${s.d}:${s.merge ? "M" : "k"}(${s.thrA.toFixed(2)},${s.thrB.toFixed(2)})`).join(" "));
  const c2 = c.map(v => 2 * v);
  const d2 = M.felzenszwalb(M.gridDissim([c2], 8, 1), 4);
  console.log(`  unit 2 (d) doubled, k=4: ${d2.labels.join()}`);
  // the same doubled = original with k/2
  check(d2.labels.join() === M.felzenszwalb(Gc, 2).labels.join(), "(d) doubling d = halving k");
  // a 6-node Kruskal example: edges listed in the exercise
  const K6: M.DGraph = { n: 6, p: Int32Array.from([0, 0, 1, 1, 2, 2, 3, 4]), q: Int32Array.from([1, 2, 2, 3, 3, 4, 5, 5]), d: Float64Array.from([4, 1, 3, 2, 5, 7, 6, 3.5]) };
  const k6 = M.kruskal(K6, M.sortEdges(K6), true);
  console.log("  unit 2 (a) Kruskal: " + k6.steps!.map(s => `${"ABCDEF"[K6.p[s.edge]]}${"ABCDEF"[K6.q[s.edge]]}${K6.d[s.edge]}:${s.join ? "join" : "skip"}`).join(" ") + ` total ${k6.total}`);
  check(close(k6.total, bruteMST(K6)), "(a) Kruskal = brute force");
}

// ---- seeded MSF = watershed (unit 3) -------------------------------------------------------
{
  // (a) chain a–g, weights 2, 5, 1, 3, 6, 2
  const G: M.DGraph = { n: 7, p: Int32Array.from([0, 1, 2, 3, 4, 5]), q: Int32Array.from([1, 2, 3, 4, 5, 6]), d: Float64Array.from([2, 5, 1, 3, 6, 2]) };
  const mk = new Int32Array(7); mk[0] = 1; mk[6] = 2;
  const P = M.primForest(G, mk, true);
  console.log("  unit 3 (a) Prim: " + P.steps!.map(s => `${"abcdefg"[G.p[s.edge]]}${"abcdefg"[G.q[s.edge]]}${s.d}:${s.kind}`).join(" "));
  check(P.labels.join() === "1,1,1,1,1,2,2", `(a) labels ${P.labels.join()}`);
  check(P.cut[4] === 1, "(a) the 6-edge is the cut");
  const mmA = M.minimaxFrom(G, 0), mmB = M.minimaxFrom(G, 6);
  check(mmA[3] === 5 && mmB[3] === 6, `(a) minimax d: ${mmA[3]}, ${mmB[3]}`);
  check(M.kruskalSeeded(G, mk).labels.join() === P.labels.join(), "(a) Kruskal with a virtual root = Prim");
  // random images: Prim = Kruskal-seeded; labels are minimax-nearest; compare with pixel flooding
  let agree = 0, tot = 0;
  for (let t = 0; t < 200; t++) {
    const w = 4 + Math.floor(r.unif() * 6), h = 4 + Math.floor(r.unif() * 6), n = w * h;
    const perm = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(r.unif() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
    const f = Float64Array.from(perm);      // distinct heights
    const mk2 = new Int32Array(n);
    mk2[Math.floor(r.unif() * n)] = 1;
    let b2 = Math.floor(r.unif() * n); while (mk2[b2]) b2 = Math.floor(r.unif() * n);
    mk2[b2] = 2;
    f[mk2.indexOf(1)] = -2; f[b2] = -1;   // markers sit at the lowest points (as minima do)
    const Gd = M.gridDissim([f], w, h, 4, "altitude");
    // edges with distinct weights: perturb ties by edge index? altitude ties remain: check minimax property only
    const Pm = M.primForest(Gd, mk2);
    const Ks = M.kruskalSeeded(Gd, mk2);
    const dA = M.minimaxFrom(Gd, mk2.indexOf(1)), dB = M.minimaxFrom(Gd, mk2.indexOf(2));
    let okMM = true;
    for (let v = 0; v < n; v++) {
      if (dA[v] < dB[v] && Pm.labels[v] !== 1) okMM = false;
      if (dB[v] < dA[v] && Pm.labels[v] !== 2) okMM = false;
      if (dA[v] < dB[v] && Ks.labels[v] !== 1) okMM = false;
      if (dB[v] < dA[v] && Ks.labels[v] !== 2) okMM = false;
    }
    check(okMM, `every pixel goes to a minimax-nearest marker (trial ${t})`);
    const fl = floodWatershed(f, w, h, mk2, { conn: 4 }).labels;
    tot++; if (fl.join() === Pm.labels.join()) agree++;
    let okFl = true;
    for (let v = 0; v < n; v++) { if (dA[v] < dB[v] && fl[v] !== 1) okFl = false; if (dB[v] < dA[v] && fl[v] !== 2) okFl = false; }
    check(okFl, `pixel flooding also gives minimax-nearest markers (trial ${t})`);
  }
  console.log(`  unit 3: MSF on altitude edges = pixel flooding in ${agree}/${tot} random images (ties at equal minimax distance may differ)`);
  // (c) the 4×4 grid of the exercise
  const f4 = [1, 3, 9, 4, 2, 5, 8, 6, 3, 7, 2, 7, 4, 9, 6, 0];
  const m4 = new Int32Array(16); m4[0] = 1; m4[15] = 2;
  const G4 = M.gridDissim([f4], 4, 4, 4, "altitude");
  const P4 = M.primForest(G4, m4, true);
  console.log("  unit 3 (c) labels: " + P4.labels.join(""));
  console.log("  unit 3 (c) Prim: " + P4.steps!.map(s => `${s.from}->${s.to} ${s.d} ${s.kind}`).join(", "));
  const fl4 = floodWatershed(f4, 4, 4, m4, { conn: 4 }).labels;
  check(fl4.join() === P4.labels.join(), `(c) MSF = flooding on the 4×4 grid (${fl4.join("")})`);
  // (g) an increasing transform leaves the MSF unchanged
  const G4s = { ...G4, d: G4.d.map(v => v * v + 10) };
  check(M.primForest(G4s, m4).labels.join() === P4.labels.join(), "(g) squaring + 10 leaves the watershed unchanged");
}

// ---- hierarchies (unit 4) -------------------------------------------------------------------
{
  // basins a..e as a 1D RAG
  const R: H.Rag = {
    n: 5, size: Float64Array.from([10, 2, 8, 3, 6]), sum: Float64Array.from([10, 2, 8, 3, 6]), min: Float64Array.from([1, 3, 0, 4, 2]),
    a: Int32Array.from([0, 1, 2, 3]), b: Int32Array.from([1, 2, 3, 4]), len: Float64Array.from([1, 1, 1, 1]), pass: Float64Array.from([5, 4, 7, 6]), gsum: Float64Array.from([5, 4, 7, 6]),
  };
  const nm = "abcde";
  const show = (T: H.MergeTree) => Array.from({ length: T.merges }, (_, i) => `${T.level[T.n + i]}`).join(",");
  const alt = H.attributeTree(R, "altitude").tree, dyn = H.attributeTree(R, "dynamics"), area = H.attributeTree(R, "area"), vol = H.attributeTree(R, "volume");
  console.log(`  unit 4 (a) levels: altitude ${show(alt)}; dynamics ${show(dyn.tree)} (edge values ${Array.from(dyn.value)}); area ${show(area.tree)} (${Array.from(area.value)}); volume ${show(vol.tree)} (${Array.from(vol.value)})`);
  const part = (T: H.MergeTree, k: number) => { const c = H.cutNodes(T, T.level[T.n + T.merges - k + 1] - 1e-9); const l = H.labelsFromNodes(T, c), map = new Map<number, number>(); return Array.from(l, v => { if (!map.has(v)) map.set(v, map.size + 1); return map.get(v)!; }).join(""); };
  check(part(alt, 3) === "11123", `(a) altitude 3 regions ${part(alt, 3)}`);
  check(part(dyn.tree, 3) === "12233", `(a) dynamics 3 regions ${part(dyn.tree, 3)}`);
  check(part(area.tree, 3) === "12233", `(a) area 3 regions ${part(area.tree, 3)}`);
  check(Array.from(dyn.value).join() === "4,1,5,2", "(a) dynamics a|b 4, b|c 1, c|d 5, d|e 2");
  check(Array.from(area.value).join() === "10,2,9,3", "(a) area a|b 10, b|c 2, c|d 9, d|e 3");
  // (b) UCM of the dynamics tree thresholded at 3
  const u = H.ucmEdges(dyn.tree, R);
  check(Array.from(u).join() === "4,1,5,2", `(b) UCM ${Array.from(u)}`);
  console.log(`  unit 4: ${nm} ok`);
  // ultrametric inequality on random trees
  for (let t = 0; t < 30; t++) {
    const n = 4 + Math.floor(r.unif() * 8);
    const merges = Array.from({ length: n - 1 }, (_, i) => ({ a: Math.floor(r.unif() * n), b: Math.floor(r.unif() * n), level: i + r.unif() }));
    const T = H.mergeTree(n, merges);
    let ok = true;
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) for (let c = 0; c < n; c++) {
      const ab = H.ultrametric(T, a, b), bc = H.ultrametric(T, b, c), ac = H.ultrametric(T, a, c);
      if (ac > Math.max(ab, bc) + 1e-12) ok = false;
    }
    check(ok, "ultrametric inequality");
  }
  // dynamics hierarchy cut at h = h-minima + flooding (1D landscape)
  for (let t = 0; t < 30; t++) {
    const n = 40;
    const f = Float64Array.from({ length: n }, (_, i) => Math.round(10 * Math.sin(i * 0.7 + t) + 6 * Math.sin(i * 0.23 + 2 * t)) + i * 0.001);
    const base = watershed(f, n, 1, hMinimaMarkers(f, n, 1, 0.0001, 4).data, { conn: 4 });
    const R1 = H.rag(base, n, 1, f, f);
    const dT = H.attributeTree(R1, "dynamics").tree;
    for (const hh of [2.5, 5.5, 9.5]) {
      const c = H.cutAt(dT, hh - 1e-6).labels;
      const merged = Int32Array.from(base, v => c[v - 1]);
      const ref = watershed(f, n, 1, hMinimaMarkers(f, n, 1, hh, 4).data, { conn: 4 });
      // same number of regions (boundary positions on the flattened plateaus of the h-minima transform may shift)
      const same = Math.max(...merged) === Math.max(...ref);
      check(same, `dynamics cut at h = ${hh} equals h-minima flooding (trial ${t})`);
    }
  }
  // single-linkage ultrametric = minimax distance
  for (let t = 0; t < 10; t++) {
    const G = randGraph(7, 0.5);
    const R2: H.Rag = { n: 7, size: new Float64Array(7).fill(1), sum: new Float64Array(7), min: new Float64Array(7), a: G.p, b: G.q, len: new Float64Array(G.p.length).fill(1), pass: G.d, gsum: G.d };
    const T = H.altitudeTree(R2).tree;
    const mm = M.minimaxFrom(G, 0);
    let ok = true;
    for (let v = 1; v < 7; v++) if (!close(H.ultrametric(T, 0, v), mm[v])) ok = false;
    check(ok, "single-linkage ultrametric = minimax distance");
  }
  // (c) 3×3 RAG greedy mean-difference merging
  const means = [10, 12, 30, 11, 25, 31, 40, 26, 50];
  const lab = Int32Array.from({ length: 9 }, (_, i) => i + 1);
  const R3 = H.rag(lab, 3, 3, means, means);
  const T3 = H.meanDiffTree(R3);
  console.log("  unit 4 (c) mean-diff merges: " + Array.from({ length: T3.merges }, (_, i) => `${T3.left[i]}+${T3.right[i]}@${T3.level[T3.n + i]}`).join(" "));
}

// ---- superpixels (unit 5) -------------------------------------------------------------------
{
  // (a) the distance by hand
  const D2 = (I: number, x: number, y: number, Ic: number, cx: number, cy: number, m: number, S: number) => (I - Ic) ** 2 + (m / S) ** 2 * ((x - cx) ** 2 + (y - cy) ** 2);
  check(close(D2(50, 4, 4, 52, 0, 0, 10, 8), 54) && close(D2(50, 4, 4, 70, 6, 4, 10, 8), 406.25), "(a) m = 10");
  check(close(D2(50, 4, 4, 52, 0, 0, 40, 8), 804) && close(D2(50, 4, 4, 70, 6, 4, 40, 8), 500), "(a) m = 40");
  const mSwitch = Math.sqrt(396 * 64 / 28);
  console.log(`  unit 5 (a) switch at m = ${mSwitch.toFixed(3)}`);
  check(close(D2(50, 4, 4, 52, 0, 0, mSwitch, 8), D2(50, 4, 4, 70, 6, 4, mSwitch, 8)), "(a) tie at the switch");
  // SLIC on a synthetic image: connectivity, about K superpixels
  const w = 64, h = 64;
  const I = Float64Array.from({ length: w * h }, (_, p) => { const x = p % w, y = (p - x) / w; return ((x - 30) ** 2 + (y - 34) ** 2 < 300 ? 200 : 50) + 20 * Math.sin(x * 1.3 + y * 0.7); });
  for (const m of [2, 10, 40]) {
    const s = SP.slic([I], w, h, { K: 64, m, iters: 10 });
    check(SP.isConnected(s.labels, w, h), `SLIC m=${m}: every superpixel connected`);
    check(s.count > 30 && s.count <= 80, `SLIC m=${m}: count ${s.count}`);
    const nc = SP.slic([I], w, h, { K: 64, m, iters: 10, connectivity: false });
    console.log(`  SLIC m=${m}: ${s.count} superpixels, ${s.orphans} orphan px; objective ${nc.objective.map(v => v.toFixed(0)).join(" ")}`);
  }
  // achievable accuracy ≥ any superpixel-constant labelling; metrics on the 8×8 example
  for (let t = 0; t < 30; t++) {
    const n = 64, sp = Int32Array.from({ length: n }, () => 1 + Math.floor(r.unif() * 6)), truth = Int32Array.from({ length: n }, () => Math.floor(r.unif() * 3));
    const asa = SP.achievableAccuracy(sp, truth);
    let ok = true;
    for (let k = 0; k < 20; k++) {
      const assign = Array.from({ length: 7 }, () => Math.floor(r.unif() * 3));
      let c = 0; for (let p = 0; p < n; p++) if (assign[sp[p]] === truth[p]) c++;
      if (c / n > asa + 1e-12) ok = false;
    }
    check(ok, "achievable accuracy is an upper bound");
  }
  const T8 = Int32Array.from({ length: 64 }, (_, p) => { const x = p % 8, y = (p - x) / 8; return (x - 4.5) ** 2 + (y - 3.5) ** 2 <= 6.5 ? 1 : 0; });
  const S8 = Int32Array.from({ length: 64 }, (_, p) => { const x = p % 8, y = (p - x) / 8; return 1 + (x >= 4 ? 1 : 0) + (y >= 4 ? 2 : 0); });
  console.log("  unit 5 (c) truth rows: " + Array.from({ length: 8 }, (_, y) => Array.from(T8.slice(8 * y, 8 * y + 8)).join("")).join(" "));
  console.log(`  unit 5 (c) ASA ${SP.achievableAccuracy(S8, T8)} (${SP.achievableAccuracy(S8, T8) * 64}/64), UE ${SP.undersegmentationError(S8, T8) * 64}/64, BR(tol 0) ${SP.boundaryRecall(S8, T8, 8, 8, 0)}, BR(tol 1) ${SP.boundaryRecall(S8, T8, 8, 8, 1)}`);
  console.log(`  unit 5 (b) ${65536 * 256} vs ${4 * 65536}`);
}

// ---- label2d sanity for single linkage on a 2D image -----------------------------------------
{
  const w = 7, h = 6, I = Float64Array.from({ length: w * h }, () => Math.floor(r.unif() * 3) * 10);
  const G = M.gridDissim([I], w, h, 4);
  const sl = M.singleLinkage(G, 1);
  // with t = 1 (join only equal neighbours) components = connected components of each grey level
  const L = new Int32Array(w * h); let next = 0;
  for (const g of [0, 10, 20]) { const lb = label2d(Array.from(I, v => (v === g ? 1 : 0)), w, h, 4); for (let p = 0; p < w * h; p++) if (lb.data[p]) L[p] = next + lb.data[p]; next += lb.count; }
  check(sl.count === next, `single linkage at t = 1 = labelling per grey level (${sl.count} vs ${next})`);
}

console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
