// Checks for the tracking day: lib/linking.ts (assignment with alternatives against brute
// force, segment linking, evaluation), lib/kalman.ts, lib/trackstats.ts, lib/tracksim.ts and
// lib/lineage.ts, and the by-hand numbers of the units track-linking, track-predict,
// track-events and track-analysis.
// Run:
//   node_modules/.bin/esbuild demos/test/tracking-test.ts --bundle --platform=node --log-level=warning | node
import { rng, uniform, poisson } from "../src/lib/random";
import * as LK from "../src/lib/linking";
import * as KF from "../src/lib/kalman";
import * as TS from "../src/lib/trackstats";
import * as SIM from "../src/lib/tracksim";
import * as LIN from "../src/lib/lineage";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

// ---------------------------------------------------------------------------------------
current = "LAP with alternatives vs brute force";
function brute(C: ArrayLike<number>, m: number, n: number, d: number, b: number): number {
  // every row picks a column or "end"; columns used at most once
  let best = Infinity;
  const used = new Uint8Array(n);
  const rec = (i: number, acc: number) => {
    if (acc >= best) return;
    if (i === m) { let s = acc; for (let j = 0; j < n; j++) if (!used[j]) s += b; best = Math.min(best, s); return; }
    rec(i + 1, acc + d);
    for (let j = 0; j < n; j++) if (!used[j] && Number.isFinite(C[i * n + j])) { used[j] = 1; rec(i + 1, acc + C[i * n + j]); used[j] = 0; }
  };
  rec(0, 0);
  return best;
}
{
  const r = rng(7);
  for (let trial = 0; trial < 400; trial++) {
    const m = 1 + Math.floor(r.unif() * 5), n = 1 + Math.floor(r.unif() * 5);
    const C = new Float64Array(m * n);
    for (let k = 0; k < m * n; k++) C[k] = r.unif() < 0.25 ? Infinity : Math.round(uniform(r, 0, 20));
    const b = Math.round(uniform(r, 1, 12));
    const want = brute(C, m, n, b, b);
    const g = LK.assignGlobal(C, m, n, b, b), dn = LK.assignDense(C, m, n, b, b);
    check(close(g.total, want), `global ${g.total} vs brute ${want} (m=${m}, n=${n})`);
    check(close(dn.total, want), `dense ${dn.total} vs brute ${want}`);
    const sp = LK.assignSparse(C, m, n, b, b);
    check(close(sp.total, want), `sparse ${sp.total} vs brute ${want}`);
    check(close(LK.totalCost(sp.links, m, n, b, b), sp.total), "sparse totalCost agrees");
    check(close(LK.totalCost(g.links, m, n, b, b), g.total), "totalCost agrees");
    check(g.links.length + g.ends.length === m && g.links.length + g.starts.length === n, "every row/column accounted for");
  }
}

current = "track-linking by-hand numbers";
{
  const A = [{ x: 0, y: 0 }, { x: 3, y: 0 }], B = [{ x: 2, y: 0 }, { x: 5, y: 0 }];
  const C = LK.costMatrix(A, B, "sq");
  check([...C].join() === "4,25,1,4", "costs 4 25 1 4");
  const nn = LK.nearestNeighbour(C, 2, 2);
  check(nn[0] === 0 && nn[1] === 0, "NN: both choose P (conflict)");
  const gr = LK.greedy(C, 2, 2);
  check(gr.reduce((s, l) => s + l.c, 0) === 26 && gr[0].i === 1 && gr[0].j === 0, "greedy 26, B–P first");
  const gl = LK.assignGlobal(C, 2, 2, 1e6, 1e6);
  check(gl.total === 8 && gl.links.length === 2, "global 8");
  const Cd = LK.costMatrix(A, B, "dist");
  check(LK.greedy(Cd, 2, 2).reduce((s, l) => s + l.c, 0) === 6, "greedy with distances 6");
  check(LK.assignGlobal(Cd, 2, 2, 1e6, 1e6).total === 4 && LK.assignGlobal(Cd, 2, 2, 1e6, 1e6).links.length === 2, "global with distances 4");
  // thresholds of b
  const nl = (b: number) => LK.assignGlobal(C, 2, 2, b, b).links.length;
  check(nl(3.6) === 2 && nl(3.4) === 1 && nl(0.6) === 1 && nl(0.4) === 0, "b thresholds 3.5 and 0.5");
  check(LK.assignGlobal(C, 2, 2, 3, 3).total === 7, "b = 3: B–P + end A + start Q = 7");
  // augmented matrix for b = 3 by enumeration of all 4! permutations equals 7
  const M = LK.augmentedMatrix(C, 2, 2, 3, 3);
  let best = Infinity;
  const perm = [0, 1, 2, 3];
  const permute = (k: number) => {
    if (k === 4) { let s = 0; for (let i = 0; i < 4; i++) s += M[i * 4 + perm[i]]; best = Math.min(best, s); return; }
    for (let i = k; i < 4; i++) { [perm[k], perm[i]] = [perm[i], perm[k]]; permute(k + 1); [perm[k], perm[i]] = [perm[i], perm[k]]; }
  };
  permute(0);
  check(best === 7, `augmented enumeration ${best}`);
  // difficulty numbers
  check(close(SIM.rmsStep(0.1, 0.1), 0.2) && close(SIM.meanSpacing(1), 0.5), "rms 0.2, spacing 0.5");
  check(Math.abs(SIM.nnErrorEstimate(1, 0.1, 0.1) - 0.1116) < 1e-4, "a/(1+a) = 0.1116");
  check(Math.abs(0.9 ** 20 - 0.1216) < 1e-4 && Math.abs(0.98 ** 50 - 0.3642) < 1e-4, "compounding");
  // matchPoints: most pairs first
  const mp = LK.matchPoints([{ x: 0, y: 0 }, { x: 2, y: 0 }], [{ x: 1, y: 0 }, { x: 3.5, y: 0 }], 1.6);
  check(mp.length === 2, "matchPoints keeps both pairs");
}

current = "NN error rate vs a/(1+a)";
{
  // Poisson positions at density n in a large box (periodic distances), random-walk steps.
  const r = rng(11), L = 60, n = 1, D = 0.1, dt = 0.1, s = Math.sqrt(2 * D * dt);
  let errs = 0, tot = 0;
  for (let rep = 0; rep < 6; rep++) {
    const N = poisson(r, n * L * L);
    const x0 = Float64Array.from({ length: N }, () => uniform(r, 0, L)), y0 = Float64Array.from({ length: N }, () => uniform(r, 0, L));
    const x1 = x0.map(v => v + s * r.norm()), y1 = y0.map(v => v + s * r.norm());
    // grid for speed
    const G = 2, nc = Math.ceil(L / G), cells: number[][] = Array.from({ length: nc * nc }, () => []);
    const w = (v: number) => ((v % L) + L) % L;
    for (let j = 0; j < N; j++) cells[Math.floor(w(x1[j]) / G) % nc * nc + Math.floor(w(y1[j]) / G) % nc].push(j);
    for (let i = 0; i < N; i++) {
      const cx = Math.floor(w(x0[i]) / G), cy = Math.floor(w(y0[i]) / G);
      let best = Infinity, bj = -1;
      for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (const j of cells[((cx + a + nc) % nc) * nc + (cy + b + nc) % nc]) {
        let dx = Math.abs(w(x1[j]) - w(x0[i])), dy = Math.abs(w(y1[j]) - w(y0[i]));
        dx = Math.min(dx, L - dx); dy = Math.min(dy, L - dy);
        const d = dx * dx + dy * dy;
        if (d < best) { best = d; bj = j; }
      }
      tot++; if (bj !== i) errs++;
    }
  }
  const est = SIM.nnErrorEstimate(n, D, dt);
  check(Math.abs(errs / tot - est) < 0.03, `NN error ${(errs / tot).toFixed(4)} vs ${est.toFixed(4)}`);
}

current = "simParticles + linkFrames";
{
  const mv = SIM.simParticles({ n: 40, T: 30, W: 256, H: 256, motion: "rw", D: 1, dt: 1, speed: 0, sigma: 0, seed: 3 });
  const frames = LK.byFrame(mv.dets, mv.T);
  const e = LK.linkFrames(mv.dets, frames, { method: "global", gate: 20, b: 200 });
  const s = LK.evalLinks(e, mv.truth);
  check(s.recall > 0.98 && s.precision > 0.98, `sparse slow (errors only at close encounters): P ${s.precision.toFixed(3)} R ${s.recall.toFixed(3)}`);
  // global never worse than greedy in total cost per frame pair
  const m2 = SIM.simParticles({ n: 150, T: 6, W: 256, H: 256, motion: "rw", D: 30, dt: 1, speed: 0, sigma: 0, seed: 5 });
  const f2 = LK.byFrame(m2.dets, m2.T);
  let ok = true;
  for (let t = 0; t + 1 < f2.length; t++) {
    const C = LK.costMatrix(f2[t].map(k => m2.dets[k]), f2[t + 1].map(k => m2.dets[k]), "sq", 30);
    const b = 300, m = f2[t].length, n = f2[t + 1].length;
    const g = LK.assignGlobal(C, m, n, b, b), gr = LK.totalCost(LK.greedy(C, m, n, 2 * b), m, n, b, b);
    if (g.total > gr + 1e-6) ok = false;
    check(close(LK.assignDense(C, m, n, b, b).total, g.total, 1e-9), "components = dense");
    check(close(LK.assignSparse(C, m, n, b, b).total, g.total, 1e-9), "sparse = dense");
  }
  check(ok, "global ≤ greedy");
}

current = "gap closing by hand (track-events a)";
{
  // ends E1, E2 × starts S1, S2: 5, 30 / 8, gated
  const C = new Float64Array([5, 30, 8, Infinity]);
  const r12 = LK.assignGlobal(C, 2, 2, 12, 12), r20 = LK.assignGlobal(C, 2, 2, 20, 20);
  check(r12.total === 29 && r12.links.length === 1 && r12.links[0].i === 0 && r12.links[0].j === 0, `b=12: ${r12.total}`);
  check(r20.total === 38 && r20.links.length === 2, `b=20: ${r20.total}`);
}

current = "segment linking recovers gaps";
{
  const mv = SIM.simParticles({ n: 10, T: 40, W: 256, H: 256, motion: "rw", D: 2, dt: 1, speed: 0, sigma: 0, miss: 0.08, seed: 9 });
  const frames = LK.byFrame(mv.dets, mv.T);
  const fe = LK.linkFrames(mv.dets, frames, { method: "global", gate: 15, b: 100 });
  const G = 3;
  const res = LK.linkSegments(mv.dets, fe, { maxGap: G, gate: 15, b: 100, splitting: false, merging: false });
  const R = new Set(res.edges.map(e => e.a + ":" + e.b));
  const gaps = mv.truth.filter(e => e.kind === "gap" && mv.dets[e.b].t - mv.dets[e.a].t - 1 <= G);
  check(gaps.length > 5, `enough gaps (${gaps.length})`);
  check(gaps.every(e => R.has(e.a + ":" + e.b)), `all ${gaps.length} gaps closed`);
  const s = LK.evalLinks(res.edges, mv.truth);
  check(s.precision > 0.97, `few wrong joins (P ${s.precision})`);
}

current = "division triplets";
{
  const truth: LK.Division[] = [{ mother: 1, daughters: [2, 3] }];
  const s1 = LK.evalDivisions([{ mother: 1, daughters: [2, 4] }], truth);
  check(s1.tp === 0 && s1.fp === 1 && s1.fn === 1, "one wrong daughter = 1 FP + 1 FN");
  const s2 = LK.evalDivisions([{ mother: 1, daughters: [2, 3] }], truth);
  check(s2.tp === 1 && s2.fp === 0 && s2.fn === 0, "exact triplet");
  // track-events (d): 12 reference links, 11 result links of which 10 correct
  const sc = LK.scores(10, 1, 2);
  check(close(sc.precision, 10 / 11) && close(sc.recall, 10 / 12) && close(sc.f1, 20 / 23), "link P/R/F1 10/11, 10/12, 20/23");
  // divisions(): from edges
  const dv = LK.divisions(4, [{ a: 0, b: 1, kind: "link" }, { a: 1, b: 2, kind: "link" }, { a: 1, b: 3, kind: "split" }]);
  check(dv.length === 1 && dv[0].mother === 1 && dv[0].daughters.join() === "2,3", "divisions from edges");
}

current = "cells: divisions found";
{
  const mv = SIM.simCells({ n0: 5, T: 60, W: 300, H: 80, D: 0.5, cycleMean: 20, cycleMin: 10, cycleShape: 4, radius: 6, miss: 0, falseRate: 0, splitProb: 0, enterRate: 0, seed: 4 });
  const frames = LK.byFrame(mv.dets, mv.T);
  const fe = LK.linkFrames(mv.dets, frames, { method: "global", gate: 10, b: 40 });
  const res = LK.linkSegments(mv.dets, fe, { maxGap: 2, gate: 12, b: 40, splitting: true, merging: false });
  const tdiv = LK.divisions(mv.dets.length, mv.truth), rdiv = LK.divisions(mv.dets.length, res.edges);
  const ds = LK.evalDivisions(rdiv, tdiv);
  check(tdiv.length >= 5, `true divisions ${tdiv.length}`);
  check(ds.recall >= 0.8 && ds.precision >= 0.8, `division P ${ds.precision.toFixed(2)} R ${ds.recall.toFixed(2)}`);
  const ls = LK.evalLinks(res.edges, mv.truth);
  check(ls.f1 > 0.95, `cell links F1 ${ls.f1.toFixed(3)}`);
}

// ---------------------------------------------------------------------------------------
current = "Kalman by-hand numbers";
{
  const u = KF.update1(10, 4, 13, 1);
  check(close(u.K, 0.8) && close(u.x, 12.4) && close(u.P, 0.8), "10±2, 13±1 → 12.4, 0.8");
  const u16 = KF.update1(10, 4, 13, 16);
  check(close(u16.K, 0.2) && close(u16.x, 10.6), "R = 16 → 10.6");
  const u4 = KF.update1(10, 4, 13, 4);
  check(close(u4.x, 11.5) && close(u4.P, 2), "R = 4 → 11.5, P 2");
  // (b): predict P 0.8 + 1 = 1.8, miss → 2.8, then K = 2.8/3.8
  const k = KF.update1(12.4, 2.8, 15, 1);
  check(close(k.K, 2.8 / 3.8) && Math.abs(k.x - 14.316) < 1e-3 && Math.abs(k.P - 0.737) < 1e-3, `after a miss: K ${k.K}`);
  // matrix form = 1D form
  const m1 = KF.randomWalk(1, 0, 1), up = KF.update({ x: [10], P: [[4]] }, [13], m1);
  check(close(up.state.x[0], 12.4) && close(up.state.P[0][0], 0.8), "matrix form reduces to 1D");
  // (c) constant velocity
  const cv = KF.constantVelocity(1, 1, 0.1, 1);
  const pr = KF.predict({ x: [0, 0], P: KF.eye(2) }, cv);
  check(close(pr.P[0][0], 2) && close(pr.P[0][1], 1) && close(pr.P[1][1], 1.1), "P⁻ = [[2,1],[1,1.1]]");
  const uc = KF.update(pr, [3], cv);
  check(close(uc.K[0][0], 2 / 3) && close(uc.K[1][0], 1 / 3), "K = (2/3, 1/3)");
  check(close(uc.state.x[0], 2) && close(uc.state.x[1], 1), "innovation 3 → +2 position, +1 velocity");
  check(close(uc.state.P[0][0], 2 / 3) && close(uc.state.P[0][1], 1 / 3) && close(uc.state.P[1][1], 23 / 30), "P = [[2/3,1/3],[1/3,23/30]]");
  // steady state
  const ss = KF.steadyRW(1, 1);
  check(close(ss.K, (Math.sqrt(5) - 1) / 2), `K∞ = ${ss.K}`);
  let P = 10;
  for (let i = 0; i < 200; i++) { const Pm = P + 1; P = KF.update1(0, Pm, 0, 1).P; }
  check(close(P + 1, ss.Pminus, 1e-9), "iteration converges to P⁻∞");
  // gate (d): innovation 2.5, S = 4 → 1.25σ; S = 0.5 → 3.54σ
  check(Math.abs(2.5 / 2 - 1.25) < 1e-12 && Math.abs(2.5 / Math.sqrt(0.5) - 3.536) < 1e-3, "gate numbers");
  check(Math.abs(KF.chi2Quantile2(0.99) - 9.21) < 0.01, "χ²₂ 99 % = 9.21");
  // E[d²] = 2 in 2D when the model is right
  const r = rng(2), m2 = KF.randomWalk(2, 1, 1);
  let st: KF.KState = { x: [0, 0], P: KF.eye(2) }, tx = 0, ty = 0, sd2 = 0;
  const N = 20000;
  for (let i = 0; i < N; i++) {
    tx += r.norm(); ty += r.norm();
    st = KF.predict(st, m2);
    const u2 = KF.update(st, [tx + r.norm(), ty + r.norm()], m2);
    if (i > 50) sd2 += u2.d2;
    st = u2.state;
  }
  check(Math.abs(sd2 / (N - 51) - 2) < 0.06, `E[d²] = ${(sd2 / (N - 51)).toFixed(3)}`);
}

// ---------------------------------------------------------------------------------------
current = "MSD by hand";
{
  const m = TS.msdTrack({ t: [0, 1, 2, 3, 4], x: [0, 1, 3, 2, 4], y: [0, 0, 0, 0, 0] }, 4, 1);
  check(close(m.msd[0], 2.5) && close(m.msd[1], 11 / 3) && close(m.msd[2], 6.5) && close(m.msd[3], 16), `MSD ${m.msd.join(", ")}`);
  check(m.n.join() === "4,3,2,1", "pair counts 4 3 2 1");
  // gaps: lags from the time stamps
  const g = TS.msdTrack({ t: [0, 1, 3], x: [0, 1, 3], y: [0, 0, 0] }, 3, 1);
  check(g.lag.join() === "1,2,3" && close(g.msd[1], 4) && close(g.msd[2], 9), "lags from time stamps");
  const tn = TS.trackNumbers({ t: [0, 1, 2], x: [0, 3, 3], y: [0, 0, 4] });
  check(close(tn.path, 7) && close(tn.net, 5) && close(tn.straightness, 5 / 7), "straightness 5/7");
}

current = "simulated MSD";
{
  const D = 0.1, dt = 0.1;
  const free = SIM.simTracks({ type: "free", n: 200, N: 200, D, v: 0, R: 1, alpha: 1, dt, sigma: 0, seed: 1 });
  const em = TS.msdEnsemble(free, 20);
  const f = TS.lineFit(em.lag.map(L => L * dt), em.msd);
  check(Math.abs(f.slope / (4 * D) - 1) < 0.05, `free slope ${f.slope.toFixed(4)} vs 4D ${4 * D}`);
  const R = 1;
  const conf = SIM.simTracks({ type: "confined", n: 200, N: 400, D: 0.5, v: 0, R, alpha: 1, dt, sigma: 0, seed: 2 });
  const cm = TS.msdEnsemble(conf, 200);
  const plateau = cm.msd.slice(100, 200).reduce((s, v) => s + v, 0) / 100;
  check(Math.abs(plateau / (R * R) - 1) < 0.05, `confined plateau ${plateau.toFixed(4)}`);
  const sig = 0.03;
  const noisy = SIM.simTracks({ type: "free", n: 200, N: 200, D: 0.01, v: 0, R: 1, alpha: 1, dt: 0.05, sigma: sig, seed: 3 });
  const nm = TS.msdEnsemble(noisy, 10), nf = TS.lineFit(nm.lag.map(L => L * 0.05), nm.msd);
  check(Math.abs(nf.offset / (4 * sig * sig) - 1) < 0.1, `offset ${nf.offset.toFixed(5)} vs 4σ² ${4 * sig * sig}`);
  for (const alpha of [0.5, 1.5]) {
    const an = SIM.simTracks({ type: "anomalous", n: 100, N: 256, D: 0.1, v: 0, R: 1, alpha, dt: 1, sigma: 0, seed: 4 });
    const am = TS.msdEnsemble(an, 30), pf = TS.powerFit(am.lag, am.msd);
    check(Math.abs(pf.alpha - alpha) < 0.06, `fBm α ${pf.alpha.toFixed(3)} vs ${alpha}`);
    check(Math.abs(pf.K / (4 * 0.1) - 1) < 0.1, `fBm K ${pf.K.toFixed(3)}`);
  }
  const dir = SIM.simTracks({ type: "directed", n: 100, N: 100, D: 0.1, v: 0.5, R: 1, alpha: 1, dt: 0.1, sigma: 0, seed: 5 });
  const dm = TS.msdEnsemble(dir, 25), q = TS.quadFit(dm.lag.map(L => L * 0.1), dm.msd);
  check(Math.abs(q.b / 0.25 - 1) < 0.05 && Math.abs(q.a / 0.4 - 1) < 0.15, `directed v² ${q.b.toFixed(3)}, 4D ${q.a.toFixed(3)}`);
  // persistent walk against its formula
  const P = 2, s = 1;
  const pw = SIM.simTracks({ type: "persistent", n: 200, N: 200, D: 0, v: s, R: 1, alpha: 1, dt: 0.2, sigma: 0, P, seed: 6 });
  const pm = TS.msdEnsemble(pw, 50);
  let maxRel = 0;
  pm.lag.forEach((L, i) => { maxRel = Math.max(maxRel, Math.abs(pm.msd[i] / TS.persistentMSD(L * 0.2, s, P) - 1)); });
  check(maxRel < 0.08, `persistent walk within ${maxRel.toFixed(3)}`);
  // speed trap: mean step / Δt = √(π D / Δt)
  for (const h of [0.1, 0.01]) {
    const tr = SIM.simTracks({ type: "free", n: 20, N: 2000, D: 0.1, v: 0, R: 1, alpha: 1, dt: h, sigma: 0, seed: 7 });
    const sp = tr.reduce((acc, x) => acc + TS.trackNumbers(x).meanStep, 0) / tr.length / h;
    check(Math.abs(sp / Math.sqrt(Math.PI * 0.1 / h) - 1) < 0.02, `apparent speed ${sp.toFixed(3)} at Δt ${h}`);
  }
}

current = "lineage";
{
  const br: LIN.Branch[] = [
    { id: 0, parent: -1, t0: 0, t1: 9, children: [1, 2] },
    { id: 1, parent: 0, t0: 10, t1: 29, children: [3, 4] },
    { id: 2, parent: 0, t0: 10, t1: 40, children: [] },
    { id: 3, parent: 1, t0: 30, t1: 40, children: [] },
    { id: 4, parent: 1, t0: 30, t1: 40, children: [] },
  ];
  const x = LIN.layoutX(br, 0);
  check(x.get(3) === 0 && x.get(4) === 1 && x.get(2) === 2 && x.get(1) === 0.5 && x.get(0) === 1.25, "tidy layout");
  const ct = LIN.cycleTimes(br, b => b.children.length > 0);
  check(ct[1].complete && ct[1].cycle === 20 && !ct[0].complete && !ct[2].complete, "cycle times and censoring");
  check(ct[3].generation === 2, "generation");
}

console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
