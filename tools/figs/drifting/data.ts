// Numbers and point sets for drifting/ (slides, exercise, quiz), computed with the drifting demo's own
// model and presets. Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/drifting/data.ts --bundle --platform=node --log-level=warning --outfile=<tmp>/dr.js
//   node <tmp>/dr.js > tools/figs/drifting/data.json
//   python3 -I tools/figs/drifting/figs.py
// Progress and the summary table go to stderr.
import * as M from "../../../demos/src/drifting/model";
import { rng } from "../../../demos/src/lib/ui";

const A = (v: ArrayLike<number>) => Array.from(v, x => Math.round(x * 1e4) / 1e4);
const log = (s: string) => process.stderr.write(s + "\n");
const out: Record<string, unknown> = {};
const f3 = (v: number) => v.toFixed(3);

const SNAP = [0, 10, 50, M.POINT_STEPS];
for (const p of M.PRESETS) {
  const c = M.presetConfig(p.id);
  if (c.mode === "points") {
    const d = M.newDrift(c), snaps: Record<number, number[]> = { 0: A(d.x) }, vs: Record<number, number[]> = { 0: A(d.lastV) };
    for (let s = 1; s <= M.POINT_STEPS; s++) { d.advance(); if (SNAP.includes(s)) { snaps[s] = A(d.x); vs[s] = A(d.lastV); } }
    const h = d.normHist, cv = d.covHist;
    out[p.id] = { snaps, vs, norm: A(h), cov: A(cv), views: p.id === "views" ? A(d.lastViews) : [] };
    log(`${p.id.padEnd(10)} |V| start ${f3(h[0])} @10 ${f3(h[10])} @50 ${f3(h[50])} @300 ${f3(h[300])}   coverage start ${f3(cv[0])} @10 ${f3(cv[10])} @50 ${f3(cv[50])} @300 ${f3(cv[300])}`);
  } else {
    const g = M.newGenerator(c), cov: number[] = [g.coverageNow()], snaps: Record<number, number[]> = { 0: A(g.sample()) };
    for (let s = 1; s <= M.GEN_STEPS; s++) { g.train(); if (s % 50 === 0) cov.push(g.coverageNow()); if ([100, 300, M.GEN_STEPS].includes(s)) snaps[s] = A(g.sample()); }
    out[p.id] = { snaps, loss: A(g.lossHist), cov: A(cov) };
    log(`${p.id.padEnd(10)} loss @1 ${f3(g.lossHist[0])} @100 ${g.lossHist[99].toFixed(4)} @1500 ${g.lossHist[M.GEN_STEPS - 1].toFixed(4)}   coverage start ${f3(cov[0])} @100 ${f3(cov[2])} @300 ${f3(cov[6])} @1500 ${f3(cov[cov.length - 1])}`);
  }
}

// references: how well do real points cover the real points?
const real = M.makeData("moons");
out.real = A(real);
for (const n of [20, 200, 400]) {
  let s = 0;
  for (let k = 0; k < 10; k++) s += M.coverage(real, M.makeData("moons", n, 50 + k));
  log(`reference: ${n} fresh real points cover ${f3(s / 10)} (mean of 10 draws)`);
  out[`ref${n}`] = s / 10;
}
// the dashed line of the demo: one new draw (seed 77) of n points
for (const n of [20, 200, 400]) log(`demo reference: a new draw (seed 77) of ${n} real points covers ${f3(M.coverage(real, M.makeData("moons", n, 77)))}`);
for (const n of [200]) log(`demo reference, blobs: ${f3(M.coverage(M.makeData("blobs"), M.makeData("blobs", n, 77)))}`);
// 20 points chosen at random from the real set itself (the classical baseline of distillation)
{
  let s = 0; const r = rng(9);
  for (let k = 0; k < 10; k++) { const P = new Float64Array(40); for (let i = 0; i < 20; i++) { const j = Math.floor(r.unif() * 400); P[2 * i] = real[2 * j]; P[2 * i + 1] = real[2 * j + 1]; } s += M.coverage(real, P); }
  log(`reference: 20 real points picked at random cover ${f3(s / 10)}`);
}

// blobs 3 : 1: share of the free points / samples in the small blob (x > 0.05)
for (const repel of [1, 0]) {
  const c = { ...M.BASE, data: "blobs" as M.DataId, repel };
  const d = M.newDrift(c);
  for (let s = 0; s < M.POINT_STEPS; s++) d.advance();
  let k = 0; for (let i = 0; i < c.n; i++) if (d.x[2 * i] > 0.05) k++;
  log(`blobs 3:1, repel ${repel}: ${k} of ${c.n} points in the small blob, coverage ${f3(d.covHist[M.POINT_STEPS])}`);
}
// augmentation: 20 points, views 1..16 and jitter 0..0.3 (coverage after 300 steps)
for (const [views, jitter] of [[1, 0], [4, 0.1], [8, 0.05], [8, 0.1], [8, 0.2], [8, 0.3], [16, 0.1]]) {
  const d = M.newDrift({ ...M.BASE, n: 20, views, jitter });
  for (let s = 0; s < M.POINT_STEPS; s++) d.advance();
  log(`20 points, ${views} views, jitter ${jitter}: coverage ${f3(d.covHist[M.POINT_STEPS])}, |V| ${f3(d.normHist[M.POINT_STEPS])}`);
}
console.log(JSON.stringify(out));
