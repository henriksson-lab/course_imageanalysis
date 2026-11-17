// Tests for the day "GANs and normalising flows" (gan/, gan-training/, image-translation/, flows/):
// the shared networks of gan1d/net.ts (backward pass against finite differences), the line GAN of
// gan1d (by-hand gradients, the best discriminator, the numbers quoted on the pages), the
// translators of translate1d (by-hand values and the quoted scores), and the flow of flow2d
// (inverse, log-determinant against a numerical Jacobian, density integrating to 1, the stored
// weights reproducing the quoted likelihoods, a training step lowering the loss). Run alone:
//   node_modules/.bin/esbuild demos/test/gans-test.ts --bundle --platform=node --log-level=warning | node
import { net, Batch, zeroGrad } from "../src/gan1d/net";
import { Gan1d, lossD, optimalD, precisionRecall1d, type Options } from "../src/gan1d/model";
import * as T from "../src/translate1d/model";
import { Flow, importWeights, sampleData, K_LAYERS, HIDDEN } from "../src/flow2d/model";
import WEIGHTS from "../src/flow2d/weights.json";
import { rng } from "../src/lib/ui";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; console.log("FAIL: " + msg); }
}
const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

// ---- net.ts: backward pass against finite differences ------------------------------------------
{
  const n = net([2, 5, 4, 3], ["tanh", "lrelu", "id"], 3), b = new Batch(n, 4), r = rng(2);
  const X = Float64Array.from({ length: 8 }, () => r.norm()), w = Float64Array.from({ length: 12 }, () => r.norm());
  const L = () => { const o = new Batch(n, 4).forward(X, 4); let s = 0; for (let i = 0; i < 12; i++) s += w[i] * o[i]; return s; };
  b.forward(X, 4);
  const g = zeroGrad(n), dX = b.backward(w, g);
  let worst = 0;
  for (let l = 0; l < n.W.length; l++) for (const [P, G] of [[n.W[l], g.gW[l]], [n.b[l], g.gb[l]]]) for (let k = 0; k < P.length; k++) {
    const o = P[k]; P[k] = o + 1e-6; const lp = L(); P[k] = o - 1e-6; const lm = L(); P[k] = o;
    worst = Math.max(worst, Math.abs((lp - lm) / 2e-6 - G[k]));
  }
  for (let k = 0; k < 8; k++) {
    const o = X[k]; X[k] = o + 1e-6; const lp = L(); X[k] = o - 1e-6; const lm = L(); X[k] = o;
    worst = Math.max(worst, Math.abs((lp - lm) / 2e-6 - dX[k]));
  }
  check(worst < 1e-6, `net.ts backward agrees with finite differences (worst ${worst})`);
}

// ---- gan1d: by-hand numbers (gan/exercise.qmd B1–B4) -----------------------------------------
{
  check(close(optimalD(0.4, 0.1), 0.8, 1e-12) && optimalD(0.2, 0.2) === 0.5 && optimalD(0, 0.3) === 0, "gan B1: best discriminator");
  check(close(lossD([0], [0]), 2 * Math.LN2, 1e-12), "gan B2/B3: loss 2 ln 2 at D = 1/2");
  // B3: one D step by hand, via the model's own update (batch 1 is not possible, so recompute the formula)
  const sig = (s: number) => 1 / (1 + Math.exp(-s));
  const g = [sig(0) - 1 + sig(0), (sig(0) - 1) * 1 + sig(0) * -1, (sig(0) - 1) * 1 + sig(0) * 1];
  check(close(g[0], 0, 1e-12) && close(g[1], -1, 1e-12) && close(g[2], 0, 1e-12), "gan B3: gradient (0, -1, 0)");
  check(close(lossD([0.5], [-0.5]), 0.948, 1e-3), "gan B3: loss after the step 0.948");
  const G = new Gan1d("one", { lrG: 0.03, lrD: 0.5, kD: 5, batch: 128, loss: "ns" }, 1);
  G.c = [0, 0.5, 0];
  check(close(G.disc(1), 0.6225, 1e-4) && close(G.slope(-1), 0.5, 1e-12), "gan B3/B4: D(1) = 0.622, slope 0.5");
  // B4: −(1 − D(−1)) · s'(−1) = −0.311
  check(close(-(1 - G.disc(-1)) * G.slope(-1), -0.3112, 1e-4) && close(-G.disc(-1) * G.slope(-1), -0.1888, 1e-4), "gan B4: generator gradients -0.311 and -0.189");
  // optimal discriminator logit is log p − log q: the slope vanishes when the generator matches
  const M = new Gan1d("one", { lrG: 0.03, lrD: 0.5, kD: 5, batch: 128, loss: "ns" }, 1, [0.5, 1]);
  M.optimal = true;
  check(close(M.disc(0.3), 0.5, 1e-9) && close(M.slope(2), 0, 1e-6), "best discriminator is 1/2 with a perfect generator");
}

// ---- gan1d: the numbers quoted on the pages ------------------------------------------------------
{
  const BASE: Options = { lrG: 0.03, lrD: 0.5, kD: 5, batch: 128, loss: "ns" };
  const runTo = (data: "one" | "two", o: Partial<Options>, optimal: boolean, steps: number) => {
    const g = new Gan1d(data, { ...BASE, ...o }, 1); g.optimal = optimal;
    const at: Record<number, { a: number; b: number; p: number; r: number }> = {};
    for (let s = 1; s <= steps; s++) { g.train(); if ([25, 50, 100, 200, 500, 2000].includes(s)) { const pr = precisionRecall1d(g); at[s] = { a: Math.abs(g.a), b: g.b, p: pr.precision, r: pr.recall }; } }
    return { g, at };
  };
  const c = runTo("one", {}, false, 2000);
  check(close(c.at[25].a, 0.41, 0.006) && close(c.at[25].b, 0.55, 0.006) && close(c.at[100].b, 1.06, 0.006) && close(c.at[50].a, 0.22, 0.006), "gan E1: converge path (0.41, 0.55), a 0.22 at 50, b 1.06 at 100");
  check(close(c.at[2000].a, 0.5, 0.015) && close(c.at[2000].b, 1.0, 0.015) && close(c.g.history[1999].lossD, 1.39, 0.02), "gan E1: ends at (0.5, 1), D loss 2 ln 2");
  const mm = runTo("one", { loss: "mm" }, false, 200);
  check(close(mm.at[100].b, 0.02, 0.006) && close(mm.at[50].a, 1.38, 0.006), "gan E4: minimax b 0.02 at step 100, a 1.38 at 50");
  const sl = runTo("one", { lrD: 0.05, kD: 1 }, false, 2000);
  check(close(sl.at[500].a, 0.02, 0.006) && close(sl.at[500].r, 0.06, 0.006) && close(sl.at[500].b, 1.12, 0.006) && close(sl.at[2000].a, 0.41, 0.006), "gan-training E1: collapse a 0.02, recall 0.06 at step 500");
  const fa = runTo("one", { lrG: 0.3, kD: 1 }, false, 200);
  check(close(fa.at[50].a, 0.02, 0.006) && close(fa.at[200].a, 0.50, 0.006), "gan E5 / gan-training E3: generator too fast");
  const dr = runTo("two", {}, true, 2000);
  check(close(dr.at[2000].a, 0.30, 0.005) && close(dr.at[2000].b, 0, 0.005) && close(dr.at[2000].p, 0.99, 0.006) && close(dr.at[2000].r, 0.49, 0.006), "gan-training E4: one bump dropped, P 0.99 R 0.49");
  const bt = runTo("two", {}, false, 2000);
  check(close(bt.at[2000].a, 0.99, 0.01) && close(bt.at[2000].b, 0.80, 0.01) && close(bt.at[2000].p, 0.71, 0.006) && close(bt.at[2000].r, 1.0, 0.006), "gan-training E5: in between, P 0.71 R 1.00");
}

// ---- translate1d: by-hand values and quoted scores -------------------------------------------
{
  const b1: T.Pair[] = [0.6, 0.6, 0.9, 0.9].map(y => ({ x: 0.7, y, m: y > 0.7 ? 1 : 0 }));
  check(close(T.condMean(b1, 0.7), 0.75, 1e-12) && T.condMedian(b1, 0.7) === 0.6, "image-translation B1: mean 0.75, median 0.6");
  const r = rng(3);
  let on = 0, other = 0;
  for (let i = 0; i < 2000; i++) { const y = T.condSample(b1, 0.7, r); if (y === 0.9) on++; else if (y !== 0.6) other++; }
  check(other === 0, "every sample is a training value");
  check(on > 900 && on < 1100, `image-translation B1: samples split about half and half (${on})`);
  check(T.quantileMap([1, 2, 3, 4], [10, 20, 30, 40], 2.5) === 30 && T.quantileMap([1, 2, 3, 4], [10, 20, 30, 40], 1, true) === 40, "image-translation B4: quantile maps");
  check(T.isReal(0.7, 0.48, 0.3) && T.isReal(0.7, 0.78, 0.3) && !T.isReal(0.7, 0.63, 0.3), "image-translation B2(c): realistic values");
  const sc = (gap: number, qa: number, qb: number) => T.scores(T.pairs(T.N_TRAIN, gap, qa, 1), T.pairs(T.N_TEST, gap, qb, 2), gap);
  const h = sc(0.3, 0.5, 0.5);
  check(close(h.rmseMean, 0.113, 6e-4) && close(h.rmseMedian, 0.131, 6e-4) && close(h.rmseSample, 0.160, 6e-4) && close(h.rmseUnpaired, 0.131, 6e-4), "image-translation E2: errors 0.113 0.131 0.160 0.131");
  check(close(h.realMean, 0.50, 0.006) && close(h.realSample, 0.98, 0.006) && close(h.spreadLow, 0.045, 0.005) && close(h.spreadHigh, 0.141, 0.005), "image-translation E2/E3: realism 50 % / 98 %, spread");
  const t = sc(0.3, 0.2, 0.8);
  check(close(t.onTruth, 0.78, 0.006) && close(t.onSample, 0.28, 0.006) && close(t.rmseMean, 0.145, 6e-4) && close(t.rmseSample, 0.177, 6e-4), "image-translation E4: treated 78 % vs 28 %");
  const v = sc(0, 0.5, 0.5);
  check(close(v.rmseMean, 0.030, 6e-4) && close(v.rmseSample, 0.045, 6e-4), "image-translation E1: visible target");
}

// ---- flow2d: invertibility, log-determinant, density, stored weights ----------------------------
{
  const W = WEIGHTS as unknown as { step: number; nll: number; w: number[][][] }[];
  const f = new Flow(K_LAYERS, HIDDEN, 1);
  importWeights(f, W[3].w);
  const r = rng(4);
  let worstInv = 0, worstDet = 0;
  for (let i = 0; i < 20; i++) {
    const x = 2 * r.norm(), y = r.norm(), q = f.forward(x, y), back = f.inverse(q.z[0], q.z[1]);
    worstInv = Math.max(worstInv, Math.abs(back[0] - x), Math.abs(back[1] - y));
    const h = 1e-5, fx = (a: number, b: number) => f.forward(a, b).z;
    const dxa = fx(x + h, y), dxb = fx(x - h, y), dya = fx(x, y + h), dyb = fx(x, y - h);
    const J = [(dxa[0] - dxb[0]) / (2 * h), (dya[0] - dyb[0]) / (2 * h), (dxa[1] - dxb[1]) / (2 * h), (dya[1] - dyb[1]) / (2 * h)];
    worstDet = Math.max(worstDet, Math.abs(Math.log(Math.abs(J[0] * J[3] - J[1] * J[2])) - q.logdet));
  }
  check(worstInv < 1e-9, `flow2d: inverse(forward(x)) = x (worst ${worstInv})`);
  check(worstDet < 1e-5, `flow2d: Σ s = log |det J| (worst ${worstDet})`);
  let I = 0;
  const N = 300, hh = 10 / N;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) I += Math.exp(f.logDensity(-5 + (i + 0.5) * hh, -5 + (j + 0.5) * hh)) * hh * hh;
  check(close(I, 1, 0.01), `flow2d: density integrates to 1 (${I.toFixed(4)})`);
  const test = sampleData("moons", 2000, rng(5));
  const nll = (g: Flow) => { let s = 0; for (let i = 0; i < 2000; i++) s -= g.logDensity(test[2 * i], test[2 * i + 1]); return s / 2000; };
  for (const [k, e] of [[0, 3.09], [1, 1.82], [2, 1.73], [3, 1.58]]) {
    const g = new Flow(K_LAYERS, HIDDEN, 1); importWeights(g, W[k].w);
    check(close(nll(g), e, 0.006) && close(W[k].nll, e, 0.006), `flow2d: held-out -log p after ${W[k].step} steps = ${e}`);
  }
  const lp = f.logDensity(0.5, 1.2), q = f.forward(0.5, 1.2);
  check(close(lp, -4.11, 0.006) && close(q.logdet, 2.62, 0.006) && close(f.logDensity(0.5, 0.3), -1.37, 0.006), "flows F5: probes log p -4.11 and -1.37");
  // training lowers the loss (fresh flow, 30 steps)
  const t = new Flow(K_LAYERS, HIDDEN, 1);
  const first = t.train("moons");
  for (let s = 0; s < 30; s++) t.train("moons");
  check(nll(t) < first, "flow2d: 30 training steps lower the held-out loss");
}

console.log(`gans: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
