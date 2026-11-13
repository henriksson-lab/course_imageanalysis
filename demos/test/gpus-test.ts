// Reference checks for the GPU day (gpu-parallel/, gpu-memory/, gpu-patterns/, gpu-fit/):
// lib/gpusim.ts (divergence, scheduler, transactions, tiles, roofline, end to end, Amdahl)
// and lib/parallel.ts (reduction, scans, compaction, atomic histogram, label propagation,
// wavefront, flooding) against brute force and closed forms. Run:
//   node_modules/.bin/esbuild demos/test/gpus-test.ts --bundle --platform=node --log-level=warning | node
import { rng } from "../src/lib/random";
import { label2d } from "../src/lib/label";
import { intHistogram } from "../src/lib/histogram";
import * as G from "../src/lib/gpusim";
import * as P from "../src/lib/parallel";
import { roundF16 } from "../src/floatbits/fp";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
function section(s: string): void { current = s; }

function samePartition(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  const f = new Map<number, number>(), g = new Map<number, number>();
  for (let i = 0; i < a.length; i++) {
    if ((a[i] === 0) !== (b[i] === 0)) return false;
    if (a[i] === 0) continue;
    if (f.has(a[i]) && f.get(a[i]) !== b[i]) return false;
    if (g.has(b[i]) && g.get(b[i]) !== a[i]) return false;
    f.set(a[i], b[i]); g.set(b[i], a[i]);
  }
  return true;
}

// ---- divergence ----
section("divergence");
{
  const all = new Array(32).fill(true), none = new Array(32).fill(false);
  const a = G.warpBranch(all, 100, 10), b = G.warpBranch(none, 100, 10);
  check(a.cycles === 100 && a.state === "A" && close(a.useful / a.issued, 1), "uniform A");
  check(b.cycles === 10 && b.state === "B" && close(b.useful / b.issued, 1), "uniform B");
  const five = all.map((_, i) => i < 5);
  const m = G.warpBranch(five, 100, 10);
  check(m.cycles === 110 && m.state === "mixed", "mixed: both paths");
  check(close(m.useful / m.issued, (5 * 100 + 27 * 10) / (32 * 110)), "mixed efficiency closed form");
  // loop: slowest lane
  const L = G.warpLoop([1, 2, 3, 8, 1, 1, 1, 1], 10);
  check(L.cycles === 80 && L.useful === 180, "loop waits for the slowest lane");
  // pMixed brute force over all 2^n patterns for n = 8
  for (const p of [0.01, 0.1, 0.5, 0.9]) {
    let pm = 0, et = 0;
    for (let s = 0; s < 256; s++) {
      let k = 0; for (let i = 0; i < 8; i++) if (s >> i & 1) k++;
      const pr = p ** k * (1 - p) ** (8 - k);
      const r = G.warpBranch(Array.from({ length: 8 }, (_, i) => !!(s >> i & 1)), 100, 10);
      if (r.state === "mixed") pm += pr;
      et += pr * r.cycles;
    }
    check(close(pm, G.pMixed(p, 8)), `pMixed p=${p}`);
    check(close(et, G.expectedWarpTime(p, 100, 10, 8)), `expected time p=${p}`);
  }
  // warps cover the image exactly once
  for (const [tw, th] of [[32, 1], [8, 4]]) {
    const ws = G.warpsOf(64, 32, tw, th), seen = new Uint8Array(64 * 32);
    for (const w of ws) { check(w.length === 32, "full warps"); for (const p of w) seen[p]++; }
    check(seen.every(v => v === 1), `warps ${tw}x${th} partition the image`);
  }
}

// ---- scheduler ----
section("scheduler");
{
  const one = G.schedule(1, 4, 20, 5);
  check(one.busy === 20, "busy = iterations x k");
  check(one.total === 5 * 4 + 4 * 20, "one warp: no hiding");
  for (const N of [1, 2, 4, 6, 8, 16]) {
    const r = G.schedule(N, 4, 20, 30);
    check(r.busy === N * 4 * 30, `busy N=${N}`);
    check(r.utilisation <= 1 + 1e-12 && r.utilisation <= r.predicted + 0.05, `utilisation <= Little N=${N}`);
    check(r.utilisation >= 0.85 * r.predicted, `utilisation near Little N=${N}: ${r.utilisation} vs ${r.predicted}`);
    let cnt = 0; for (const v of r.issue) if (v >= 0) cnt++;
    check(cnt === r.busy, "issue trace consistent");
  }
  check(G.residentWarps(64) === 32 && G.residentWarps(255) === 8 && G.residentWarps(128) === 16, "register-limited occupancy");
}

// ---- transactions ----
section("transactions");
{
  const row = G.transactions(G.accessAddresses("row"));
  check(row.lines === 1 && row.sectors === 4 && row.moved === 128 && row.efficiency === 1, "row: 1 line, 100%");
  const col = G.transactions(G.accessAddresses("column", { W: 1024 }));
  check(col.lines === 32 && col.sectors === 32 && close(col.efficiency, 0.125), "column: 32 lines, 12.5%");
  const s2 = G.transactions(G.accessAddresses("stride", { s: 2 }));
  check(s2.lines === 2 && s2.sectors === 8 && close(s2.efficiency, 0.5), "stride 2: 50%");
  const s8 = G.transactions(G.accessAddresses("stride", { s: 8 }));
  check(s8.sectors === 32 && close(s8.efficiency, 0.125), "stride 8: 12.5%");
  const rgb = G.transactions(G.accessAddresses("rgb"));
  check(rgb.lines === 3 && rgb.sectors === 12 && close(rgb.efficiency, 1 / 3), "rgb R only: a third");
  // brute force: sectors by enumerating bytes
  const r = rng(3);
  for (let t = 0; t < 50; t++) {
    const addrs = Array.from({ length: 32 }, () => 4 * Math.floor(r.unif() * 4096));
    const S = new Set<number>(); for (const a of addrs) for (let b = a; b < a + 4; b++) S.add(Math.floor(b / 32));
    check(G.transactions(addrs).sectors === S.size, "sectors brute force");
  }
}

// ---- tiles ----
section("tiles");
{
  for (const k of [1, 3, 5, 9]) for (const T of [4, 8, 16, 32]) {
    const rc = G.readCounts(32, k, T);
    check(close(rc.total / (32 * 32), G.readsTiled(k, T)), `tile reads k=${k} T=${T}`);
  }
  for (const k of [3, 5]) check(G.readCounts(32, k, null).total === 32 * 32 * k * k, "direct reads");
  check(close(G.readsTiled(3, 16), (18 / 16) ** 2), "(18/16)^2");
}

// ---- roofline, end to end ----
section("roofline");
{
  const d = G.device("gpu");
  check(close(G.ridge(d.peak, d.bw), 30), "desktop ridge = 30");
  for (const I of [0.01, 0.1, 1, 10, 30, 100, 1000]) check(close(G.attainable(I, d.peak, d.bw), Math.min(d.peak, d.bw * I)), "attainable");
  for (const [W, Q] of [[1e9, 1e6], [1e6, 1e9], [3e10, 1e9]]) check(close(G.kernelTime(W, Q, d.peak, d.bw), Math.max(W / d.peak, Q / d.bw)), "kernel time");
  check(close(G.intensity(G.opCost("add")), 1 / 12), "add I = 1/12");
  check(close(G.intensity(G.opCost("conv", { k: 11 })), 121 / 4), "k x k I = k^2/4");
  check(close(G.intensity(G.opCost("matmul", { N: 600 })), 100), "matmul I = N/6");
  check(close(G.intensity(G.opCost("pipe4")) * 4, G.intensity(G.opCost("fused4"))), "fusing: 4x intensity");
  check(close(G.intensity(G.opCost("conv", { k: 5, bytes: 2 })), 2 * G.intensity(G.opCost("conv", { k: 5 }))), "float16 doubles I");
  const e = G.endToEnd(d, 64e6, 3, 1e9, 1e8, true);
  check(close(e.upload, 64e6 / 25e9) && close(e.launch, 3e-5) && close(e.compute, 3 * 1e-4) && close(e.total, e.upload + e.download + e.launch + e.compute), "end to end");
  const e2 = G.endToEnd(d, 64e6, 3, 1e9, 1e8, false);
  check(close(e2.upload, 3 * e.upload), "no staying: transfer per kernel");
  check(close(G.breakEven(0, 1, 1, 2), 2), "break-even");
  check(close(G.inFlight(1e12, 500e-9), 5e5), "Little");
}

// ---- Amdahl ----
section("amdahl");
{
  check(close(G.amdahl(0.1, 1e12), 10, 1e-6), "limit 1/s");
  check(close(G.amdahl(0.3, 50), 1 / (0.3 + 0.7 / 50)), "formula");
  check(close(G.pipelineSpeedup([2, 6, 1, 1], [1, 50, 1, 1]), 10 / (2 + 6 / 50 + 2)), "pipeline");
}

// ---- reduction ----
section("reduction");
{
  const r = rng(11);
  for (const n of [1, 2, 3, 8, 13, 16, 64]) {
    const xs = Array.from({ length: n }, () => Math.floor(r.unif() * 100));
    const red = P.reduceTree(xs);
    check(red.sum === xs.reduce((a, b) => a + b, 0), `tree sum n=${n}`);
    check(red.work === n - 1, `work n-1 n=${n}`);
    check(red.depth === Math.ceil(Math.log2(Math.max(1, n))), `depth n=${n}`);
  }
  // float16: tree vs sequential differ for many small values
  const xs = new Array(4096).fill(0.01);
  const seq = P.sumSequential(xs, roundF16), tree = P.reduceTree(xs, roundF16).sum;
  check(seq < tree && Math.abs(tree - 40.96) < 0.5, `float16 sequential stalls (${seq}) tree ok (${tree})`);
}

// ---- scans ----
section("scan");
{
  const r = rng(5);
  for (const n of [2, 4, 8, 16, 32, 64]) {
    const xs = Array.from({ length: n }, () => Math.floor(r.unif() * 20) - 5);
    const ref = P.prefixSums(xs);
    const hs = P.scanHillisSteele(xs), bl = P.scanBlelloch(xs);
    const lg = Math.log2(n);
    check(hs.out.every((v, i) => v === ref.inclusive[i]), `HS n=${n}`);
    check(bl.out.every((v, i) => v === ref.exclusive[i]), `Blelloch n=${n}`);
    check(hs.work === n * lg - (n - 1), `HS work n=${n}`);
    check(hs.depth === lg, `HS depth n=${n}`);
    check(bl.work === 2 * (n - 1), `Blelloch work n=${n}`);
    check(bl.depth === 2 * lg, `Blelloch depth n=${n}`);
  }
}

// ---- compaction ----
section("compact");
{
  const r = rng(9);
  for (let t = 0; t < 20; t++) {
    const n = 16 + Math.floor(r.unif() * 40);
    const v = Array.from({ length: n }, () => Math.floor(r.unif() * 100)), m = v.map(() => (r.unif() < 0.4 ? 1 : 0));
    const c = P.compact(v, m), ref = v.filter((_, i) => m[i]);
    check(c.count === ref.length && c.out.every((x, i) => x === ref[i]), "compaction = filter");
  }
}

// ---- atomic histogram ----
section("histogram");
{
  const r = rng(13);
  const vals = Array.from({ length: 64 }, () => Math.floor(r.unif() * 8));
  const h = P.atomicHistogram(vals, 8), ref = intHistogram(vals, 0, 7);
  check(h.counts.every((c, i) => c === ref.counts[i]), "counts = lib/histogram");
  check(h.cycles === Math.max(...h.counts), "global: busiest bin");
  const dark = new Array(64).fill(0);
  check(P.atomicHistogram(dark, 8).cycles === 64, "dark: 64 cycles");
  const pr = P.atomicHistogram(dark, 8, 8);
  check(pr.cycles === 8 + 8 && pr.counts[0] === 64, "privatised dark: 8 + 8");
}

// ---- label propagation ----
section("labels");
{
  const r = rng(17);
  for (let t = 0; t < 30; t++) {
    const w = 8 + Math.floor(r.unif() * 16), h = 6 + Math.floor(r.unif() * 12);
    const m = Array.from({ length: w * h }, () => (r.unif() < 0.55 ? 1 : 0));
    for (const conn of [4, 8] as const) {
      const ref = label2d(m, w, h, conn).data;
      const a = P.labelPropagation(m, w, h, conn, false), b = P.labelPropagation(m, w, h, conn, true);
      check(samePartition(a.final, ref), `propagation partition ${w}x${h} c${conn}`);
      check(samePartition(b.final, ref), `pointer jumping partition ${w}x${h} c${conn}`);
      check(b.passes <= a.passes, "jumping never slower");
    }
  }
  for (const l of [2, 5, 10, 24]) {
    const bar = new Array(l).fill(1);
    const a = P.labelPropagation(bar, l, 1, 4, false);
    check(a.passes === l - 1, `bar ${l}: passes = l - 1 (${a.passes})`);
    const b = P.labelPropagation(bar, l, 1, 4, true);
    check(b.passes <= Math.ceil(Math.log2(l)) + 1, `bar ${l}: jumping ~log2 l (${b.passes})`);
  }
}

// ---- wavefront ----
section("wavefront");
{
  for (const [R, C] of [[5, 7], [1, 4], [6, 6], [10, 3]]) {
    const s = P.wavefront(R, C), when = new Int32Array(R * C).fill(-1);
    s.forEach((cells, d) => cells.forEach(c => { check(when[c] < 0, "visited once"); when[c] = d; }));
    check(when.every(v => v >= 0), "all cells");
    check(s.length === R + C - 1, "depth R + C - 1");
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
      if (i > 0) check(when[(i - 1) * C + j] < when[i * C + j], "dep up");
      if (j > 0) check(when[i * C + j - 1] < when[i * C + j], "dep left");
      if (i > 0 && j > 0) check(when[(i - 1) * C + j - 1] < when[i * C + j], "dep diag");
    }
    check(Math.max(...s.map(c => c.length)) === Math.min(R, C), "max parallel = min(R, C)");
  }
}

// ---- flooding ----
section("flood");
{
  const r = rng(23);
  for (let t = 0; t < 10; t++) {
    const w = 12, h = 10, f = Array.from({ length: w * h }, () => Math.floor(r.unif() * 6));
    const mk = new Int32Array(w * h); mk[0] = 1; mk[w * h - 1] = 2; mk[5 * w + 6] = 3;
    const a = P.levelFlood(f, w, h, mk), b = P.queueFlood(f, w, h, mk);
    check(a.labels.every(v => v > 0) && b.labels.every(v => v > 0), "everything labelled");
    check(b.steps.length === w * h - 3, "queue: one pixel per step");
    check(a.steps.length < b.steps.length, "level-synchronous: fewer steps");
    let tot = 0; for (const s of a.steps) tot += s.length;
    check(tot === w * h - 3, "level: each pixel once");
  }
}

console.log(`gpus-test: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
