// Monte Carlo data for decayfit/ figures, with exactly the estimators, seeds and settings of the
// photonbudget demo (demos/src/photonbudget/main.ts). Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/decayfit/mc.ts --bundle --platform=node --log-level=warning --outfile=<tmp>/mc.js && node <tmp>/mc.js <tmp>/mc.json
// then  python3 -I tools/figs/decayfit/figs.py <tmp>/mc.json
import * as fs from "fs";
import { rng } from "../../../demos/src/lib/random";
import * as D from "../../../demos/src/lib/decay";
import { lmFit, Cost } from "../../../demos/src/lib/nlfit";

const AX: D.TimeAxis = { T: 12.5, nBins: 64 }, IRF = { sigma: 0.15, delay: 1 };
const M = 400, M_N = 150, N_LIST = [30, 100, 300, 1000, 3000, 10000, 30000, 100000];
const IRF_MEAN = D.meanArrival(D.gaussianIRF(IRF.sigma, IRF.delay, AX), AX);
interface State { mode: "mono" | "bi"; tau1: number; tau2: number; frac: number; N: number; bg: number }

function monoMethods(y: Float64Array, fitBg: boolean): number[] {
  let tot = 0;
  for (const v of y) tot += v;
  const ma = D.meanArrival(y, AX) - IRF_MEAN;
  const tg = D.twoGateTau(y, AX, 1.5, 2.5).tau;
  const t0 = Math.max(0.3, Math.min(6, Number.isFinite(ma) ? ma : 2));
  const model = (p: Float64Array) => D.expectedCounts({ tau: [p[1]], f: [1] }, IRF, AX, p[0], p[2]);
  const fit = (cost: Cost) => lmFit(model, y, [Math.max(tot, 1), t0, fitBg ? 0.1 : 0], { cost, positive: [true, true, false], lower: [0, 0.02, 0], upper: [Infinity, 50, Infinity], fixed: [false, false, !fitBg], maxIter: 60, tol: 1e-8 }).p[1];
  return [ma, tg, fit("ls"), fit("neyman"), fit("poisson")];
}
function biFit(y: Float64Array, st: State): [number, number] {
  let tot = 0;
  for (const v of y) tot += v;
  const model = (p: Float64Array) => { const s = p[0] + p[2]; return D.expectedCounts({ tau: [p[1], p[3]], f: [p[0] / s, p[2] / s] }, IRF, AX, s, 0); };
  const r = lmFit(model, y, [tot * (1 - st.frac), st.tau1, tot * st.frac, st.tau2], { cost: "poisson", positive: [true, true, true, true], lower: [0, 0.02, 0, 0.02], upper: [Infinity, 50, Infinity, 50], maxIter: 100, tol: 1e-8 });
  const a = r.p[1], b = r.p[3];
  return a <= b ? [a, b] : [b, a];
}
function simulate(st: State, N: number, k: number, seed: number): Float64Array {
  const r = rng(seed * 7919 + k);
  const dec: D.Decay = st.mode === "mono" ? { tau: [st.tau1], f: [1] } : { tau: [Math.min(st.tau1, st.tau2), Math.max(st.tau1, st.tau2)], f: st.tau1 <= st.tau2 ? [1 - st.frac, st.frac] : [st.frac, 1 - st.frac] };
  return D.poissonHistogram(r, D.expectedCounts(dec, IRF, AX, N * (1 - st.bg / 100), N * st.bg / 100 / AX.nBins));
}
const rmsRel = (a: number[], t: number) => { const v = a.filter(Number.isFinite); return Math.sqrt(v.reduce((s, x) => s + (x - t) ** 2, 0) / v.length) / t; };

const mono: State = { mode: "mono", tau1: 2.5, tau2: 0.8, frac: 0.5, N: 1000, bg: 0 };
const bi: State = { mode: "bi", tau1: 0.8, tau2: 3, frac: 0.6, N: 10000, bg: 0 };
const res: any = { N_LIST, ideal: N_LIST.map(N => 1 / Math.sqrt(N)) };
// "few photons" preset (N = 100) and the default (N = 1000), seed 1: every estimate per method
for (const N of [100, 1000]) {
  const est: number[][] = [[], [], [], [], []];
  for (let k = 0; k < M; k++) monoMethods(simulate(mono, N, k, 1), false).forEach((v, i) => est[i].push(v));
  res["few" + N] = est;
}
// relative RMS error against N (seed 1 + 100), five methods, and the bi-exponential
res.vsMono = [[], [], [], [], []];
res.vsBi = [[], []];
for (const N of N_LIST) {
  const b: number[][] = [[], [], [], [], []], t1: number[] = [], t2: number[] = [];
  for (let k = 0; k < M_N; k++) {
    monoMethods(simulate(mono, N, k, 101), false).forEach((v, j) => b[j].push(v));
    if (N >= 300) { const [x, y] = biFit(simulate(bi, N, k, 101), bi); t1.push(x); t2.push(y); }
  }
  b.forEach((v, j) => res.vsMono[j].push(rmsRel(v, 2.5)));
  res.vsBi[0].push(N >= 300 ? rmsRel(t1, 0.8) : null); res.vsBi[1].push(N >= 300 ? rmsRel(t2, 3) : null);
  console.error("N", N, "poisson", res.vsMono[4].at(-1), "bi", res.vsBi[0].at(-1), res.vsBi[1].at(-1));
}
fs.writeFileSync(process.argv[2], JSON.stringify(res));
