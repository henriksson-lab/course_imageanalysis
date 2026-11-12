// Tests for the modern deep learning day (instances/, detection/, attention/, diffusion/):
// lib/boxes.ts, lib/mixture.ts and the models of instancemaps, nmsboxes, selfattn,
// calibration, embedprobe, diffusion2d and genrestore. Brute-force or reference checks.
// Run: node_modules/.bin/esbuild demos/test/moderndl-test.ts --bundle --platform=node --log-level=warning | node
import { boxIou, nms, type Box, localMaxima, encodeBox, decodeBox, boxIouTable } from "../src/lib/boxes";
import { makeDiag, logDensityDiag, scoreDiag, posteriorDiag, posteriorMean, posteriorSample, noisedMixture, alphaBarAt, sampleDiag } from "../src/lib/mixture";
import { match } from "../src/lib/eval";
import { rng } from "../src/lib/ui";
import * as IM from "../src/instancemaps/model";
import * as NB from "../src/nmsboxes/model";
import * as SA from "../src/selfattn/model";
import * as CB from "../src/calibration/model";
import * as EP from "../src/embedprobe/model";
import { probeScene } from "../src/embedprobe/scene";
import { SSL_REF } from "../src/embedprobe/ssl_weights";
import * as DF from "../src/diffusion2d/model";
import { SCORE_NETS } from "../src/diffusion2d/score_weights";
import * as GR from "../src/genrestore/model";
import { dct2 } from "../src/lib/dct";

let fails = 0, passes = 0;
function check(name: string, ok: boolean, info = ""): void {
  if (ok) passes++; else { fails++; console.log("FAIL", name, info); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

// ---- boxes ------------------------------------------------------------------------------
{
  const r = rng(1);
  let ok = true;
  for (let t = 0; t < 300; t++) {
    const b = (): Box => { const x0 = Math.floor(r.unif() * 20), y0 = Math.floor(r.unif() * 20); return { x0, y0, x1: x0 + 1 + Math.floor(r.unif() * 12), y1: y0 + 1 + Math.floor(r.unif() * 12) }; };
    const A = b(), B = b();
    let i = 0, u = 0;
    for (let y = 0; y < 40; y++) for (let x = 0; x < 40; x++) {
      const ina = x >= A.x0 && x < A.x1 && y >= A.y0 && y < A.y1, inb = x >= B.x0 && x < B.x1 && y >= B.y0 && y < B.y1;
      if (ina && inb) i++; if (ina || inb) u++;
    }
    if (!close(boxIou(A, B), i / u, 1e-12)) ok = false;
  }
  check("box IoU = rasterised IoU", ok);
  check("IoU of [0,4]² and [2,6]² = 4/28", close(boxIou({ x0: 0, y0: 0, x1: 4, y1: 4 }, { x0: 2, y0: 2, x1: 6, y1: 6 }), 4 / 28));
  check("anchor 10×10 vs cell 4×4, same centre: IoU 0.16", close(boxIou({ x0: -5, y0: -5, x1: 5, y1: 5 }, { x0: -2, y0: -2, x1: 2, y1: 2 }), 0.16));
  // NMS against a brute-force reference (repeated scans)
  for (let t = 0; t < 50; t++) {
    const n = 12, boxes: Box[] = [], sc: number[] = [];
    for (let k = 0; k < n; k++) { const x = r.unif() * 30, y = r.unif() * 30; boxes.push({ x0: x, y0: y, x1: x + 5 + 10 * r.unif(), y1: y + 5 + 10 * r.unif() }); sc.push(r.unif()); }
    const thr = 0.1 + 0.6 * r.unif();
    const keep: number[] = [], alive = new Set(boxes.map((_, i) => i));
    while (alive.size) {
      let best = -1; for (const i of alive) if (best < 0 || sc[i] > sc[best]) best = i;
      keep.push(best); alive.delete(best);
      for (const j of [...alive]) if (boxIou(boxes[best], boxes[j]) > thr) alive.delete(j);
    }
    const res = nms(boxes, sc, thr);
    if (res.keep.join() !== keep.join()) { check("NMS = brute force", false, `${res.keep} vs ${keep}`); break; }
    if (t === 49) check("NMS = brute force", true);
  }
  const H = [1, 2, 1, 0, 0, 2, 5, 2, 0, 3, 1, 2, 1, 0, 0, 0, 0, 0, 4, 1, 3, 0, 2, 0, 2];
  check("local maxima of a 5×5 heat map (3×3 window)", localMaxima(H, 5, 5, 1).join() === "6,9,18,20", localMaxima(H, 5, 5, 1).join());
  const a = { x0: 0, y0: 0, x1: 10, y1: 20 }, bx = { x0: 3, y0: -2, x1: 9, y1: 30 }, d = decodeBox(encodeBox(bx, a), a);
  check("box encode/decode round trip", close(d.x0, bx.x0, 1e-9) && close(d.y1, bx.y1, 1e-9));
  // evaluation through lib/eval.ts
  const T = boxIouTable([{ x0: 0, y0: 0, x1: 4, y1: 4 }], [{ x0: 0, y0: 0, x1: 4, y1: 4 }, { x0: 1, y0: 0, x1: 5, y1: 4 }]);
  const m = match(T, 0.5, "optimal");
  check("duplicate box = 1 TP + 1 FP", m.tp === 1 && m.fp === 1 && m.fn === 0);
}

// ---- nmsboxes scenes: crowded neighbours overlap more than 0.2 ----------------------------
{
  const s = NB.makeScene("crowd");
  let mx = 0; for (let i = 0; i < s.truth.length; i++) for (let j = i + 1; j < s.truth.length; j++) mx = Math.max(mx, boxIou(s.truth[i], s.truth[j]));
  check("crowd: neighbouring true boxes overlap (IoU > 0.3)", mx > 0.3 && mx < 0.5, String(mx));
  const c = NB.simulateDetector(s.truth, 7);
  const after = nms(c.boxes, c.scores, 0.2, 0.2).keep.length, after5 = nms(c.boxes, c.scores, 0.5, 0.2).keep.length;
  check("crowd: t = 0.2 keeps fewer boxes than cells, t = 0.5 does not", after < s.truth.length && after5 >= s.truth.length, `${after} ${after5}`);
}

// ---- instancemaps -----------------------------------------------------------------------
{
  // star-convex rays on a disc: from the centre all rays ≈ r
  const lab = new Int32Array(IM.N * IM.N);
  for (let y = 0; y < IM.N; y++) for (let x = 0; x < IM.N; x++) if (Math.hypot(x - 64, y - 64) <= 15.5) lab[y * IM.N + x] = 1;
  const R = IM.rayTarget(lab), c = 64 * IM.N + 64;
  let lo = Infinity, hi = 0; for (let k = 0; k < IM.MAX_RAYS; k++) { lo = Math.min(lo, R[c * IM.MAX_RAYS + k]); hi = Math.max(hi, R[c * IM.MAX_RAYS + k]); }
  check("disc: all rays from the centre within [15, 17]", lo >= 15 && hi <= 17, `${lo} ${hi}`);
  // flows on a single disc end in one sink
  const D = IM.distanceTarget(lab, 1), fl = IM.flowTarget(lab, 1, D);
  const pr: IM.Prediction = { route: "flows", fg: Float32Array.from(lab), vx: fl.vx, vy: fl.vy };
  const res = IM.postprocess(pr, { thr: 0.5, h: 0.1, nRays: 32, nmsT: 0.3, steps: 40 });
  check("disc: flows give one object", res.count === 1, String(res.count));
  // the polygon of a disc's centre covers the disc
  const poly = IM.polygonPixels(c, Array.from({ length: 32 }, (_, k) => R[c * IM.MAX_RAYS + 2 * k]));
  let inside = 0; for (const p of poly) if (lab[p]) inside++;
  const area = lab.reduce((a, b) => a + b, 0);
  check("disc: centre polygon covers ≥ 95 % of the disc", inside / area >= 0.95, String(inside / area));
  // the mixed scene: exact targets give every object on the boundary / distance / flow routes
  const sc = IM.makeScene("mixed"), T = IM.computeTargets(sc);
  const E0 = { noise: 0, blur: 0, contact: 0, missed: 0 };
  for (const route of ["boundary", "distance", "flows"] as IM.Route[]) {
    const r = IM.postprocess(IM.predict(sc, T, route, E0), { thr: route === "distance" ? 0.1 : 0.5, h: 0.15, nRays: 32, nmsT: 0.3, steps: 40 });
    check(`mixed scene, exact ${route} target: ${sc.n} objects`, r.count === sc.n, String(r.count));
  }
  const sem = IM.postprocess(IM.predict(sc, T, "semantic", E0), { thr: 0.5, h: 0.15, nRays: 32, nmsT: 0.3, steps: 40 });
  check("mixed scene, semantic mask: touching pairs merge (11 objects)", sem.count === 11, String(sem.count));
}

// ---- attention --------------------------------------------------------------------------
{
  const t = SA.tinyAttention([1, 0], [[1, 0], [0, 1], [1, 1]], [10, 0, 4]);
  check("tiny attention: weights 0.42 0.16 0.42, output 5.9", close(t.weights[0], 0.4223, 1e-3) && close(t.out, 5.91, 2e-3), JSON.stringify(t));
  const img = SA.synthImage(), F = SA.handFeatures(img);
  const A = SA.attentionWeights(F, { wc: 1, wp: 0.5, T: 0.7, features: "hand" });
  let ok = true; for (let i = 0; i < SA.NT; i++) { let s = 0; for (let j = 0; j < SA.NT; j++) s += A[i * SA.NT + j]; if (!close(s, 1, 1e-9)) ok = false; }
  check("attention rows sum to 1", ok);
  const perm = SA.permutation(5), opt = { wc: 1, wp: 0, T: 0.5, features: "hand" as const };
  const sh = SA.shufflePatches(img, perm);
  const a1 = SA.attend(sh, SA.attentionWeights(SA.features(sh, "hand"), opt));
  const a2 = SA.shufflePatches(SA.attend(img, SA.attentionWeights(SA.features(img, "hand"), opt)), perm);
  let dmax = 0; for (let i = 0; i < a1.length; i++) dmax = Math.max(dmax, Math.abs(a1[i] - a2[i]));
  check("no position code: attention is permutation-equivariant", dmax < 1e-3, String(dmax));
  const optP = { wc: 1, wp: 1, T: 0.5, features: "hand" as const };
  const b1 = SA.attend(sh, SA.attentionWeights(SA.features(sh, "hand"), optP)), b2 = SA.shufflePatches(SA.attend(img, SA.attentionWeights(SA.features(img, "hand"), optP)), perm);
  let dp = 0; for (let i = 0; i < b1.length; i++) dp = Math.max(dp, Math.abs(b1[i] - b2[i]));
  check("with a position code it is not", dp > 1, String(dp));
}

// ---- calibration ------------------------------------------------------------------------
{
  // the 10-row table of the exercise: confidences and correctness, two bins [0.5, 0.75), [0.75, 1]
  const conf = [0.55, 0.6, 0.65, 0.7, 0.72, 0.8, 0.85, 0.9, 0.95, 0.99], cor = [1, 0, 1, 0, 1, 1, 0, 1, 1, 1];
  const R = CB.reliability(conf.map((c, i) => ({ conf: c, correct: cor[i] })), 2);
  check("ECE of the 10-row table = 0.071", close(R.ece, 0.071, 1e-9), String(R.ece));
  const d = CB.simulate(4000, 1, 0, 2), T = CB.fitTemperature(d);
  check("calibrated network: fitted T ≈ 1", Math.abs(T - 1) < 0.1, String(T));
  const d3 = CB.simulate(4000, 3, 0, 2), T3 = CB.fitTemperature(d3);
  check("overconfident (s = 3): fitted T ≈ 3", Math.abs(T3 - 3) < 0.3, String(T3));
  const p1 = CB.predictions(d3, 1), p2 = CB.predictions(d3, T3);
  check("temperature does not change the predicted class", p1.every((p, i) => p.correct === p2[i].correct));
}

// ---- embedprobe: TS inference = the training script's reference --------------------------
{
  const sc = probeScene(900001), img = Float32Array.from(sc.image, v => Math.round(v));
  const E = EP.learnedEmbedding(img);
  let dm = 0;
  SSL_REF.pos.forEach(([y, x]: number[], k: number) => { for (let c = 0; c < E.d; c++) dm = Math.max(dm, Math.abs(E.z[(y * EP.G + x) * E.d + c] - SSL_REF.z[k][c])); });
  check("embedprobe encoder = train_ssl.py reference (int8 weights)", dm < 2e-3, String(dm));
  const P = EP.trainProbe({ d: 2, z: Float32Array.from([1, 0, 0.9, 0.1, 0, 1, 0.1, 0.9]) }, [0, 1, 2, 3], [1, 1, 0, 0]);
  const out = EP.applyProbe({ d: 2, z: Float32Array.from([1, 0, 0, 1]) }, P);
  check("linear probe separates a toy set", out[0] > 0.9 && out[1] < 0.1);
}

// ---- mixtures ---------------------------------------------------------------------------
{
  const m = makeDiag([0.3, 0.7], [[-1, 0.5], [1.2, -0.3]], [[0.2, 0.5], [0.3, 0.1]]);
  const x = [0.1, 0.2], s = scoreDiag(m, x), h = 1e-5;
  const fd = [0, 1].map(i => { const a = [...x], b = [...x]; a[i] += h; b[i] -= h; return (logDensityDiag(m, a) - logDensityDiag(m, b)) / (2 * h); });
  check("mixture score = finite differences of log p", close(s[0], fd[0], 1e-5) && close(s[1], fd[1], 1e-5), `${s} ${fd}`);
  // noised mixture = distribution of √ā x0 + √(1−ā) ε (moments)
  const ab = 0.3, nm = noisedMixture(m, ab), smp = sampleDiag(m, 200000, rng(4)), r = rng(5);
  let mean = 0, v = 0;
  for (let i = 0; i < 200000; i++) { const z = Math.sqrt(ab) * smp.X[2 * i] + Math.sqrt(1 - ab) * r.norm(); mean += z; v += z * z; }
  mean /= 200000; v = v / 200000 - mean * mean;
  const em = 0.3 * nm.mu[0] + 0.7 * nm.mu[2], ev = 0.3 * (nm.v[0] + nm.mu[0] ** 2) + 0.7 * (nm.v[2] + nm.mu[2] ** 2) - em * em;
  check("noised mixture: mean and variance", Math.abs(mean - em) < 0.01 && Math.abs(v - ev) < 0.01, `${mean} ${em} ${v} ${ev}`);
  // exact posterior vs importance sampling (1D, y = 0.5 x + n)
  const m1 = makeDiag([0.5, 0.5], [[-1], [1.5]], [0.3, 0.2]), a = 0.5, sn = 0.4, y = 0.35;
  const post = posteriorDiag(m1, [y], [a], sn * sn), pm = posteriorMean(post)[0];
  const ss = sampleDiag(m1, 400000, rng(8));
  let sw = 0, swx = 0;
  for (let i = 0; i < 400000; i++) { const xx = ss.X[i], w = Math.exp(-((y - a * xx) ** 2) / (2 * sn * sn)); sw += w; swx += w * xx; }
  check("exact posterior mean = importance sampling", Math.abs(pm - swx / sw) < 0.01, `${pm} ${swx / sw}`);
  // a sample has twice the expected squared error of the posterior mean (Gaussian prior)
  const g = makeDiag([1], [[0]], [1]), rr = rng(9);
  let em2 = 0, es2 = 0;
  for (let t = 0; t < 40000; t++) {
    const xt = rr.norm(), yy = xt + 0.8 * rr.norm(), p = posteriorDiag(g, [yy], [1], 0.64);
    em2 += (posteriorMean(p)[0] - xt) ** 2; es2 += (posteriorSample(p, rr).x[0] - xt) ** 2;
  }
  check("posterior sample: twice the squared error of the mean", Math.abs(es2 / em2 - 2) < 0.06, String(es2 / em2));
  // the by-hand numbers
  check("tanh example: data ±1, x_t = 0.5, 0.6/0.8 → 0.44", close(Math.tanh(0.6 * 0.5 / 0.64), 0.4372, 1e-3));
  // the same with the exact mixture machinery: data ±1 (tiny width), ā = 0.36, E[x_0 | x_t = 0.5]
  const two = makeDiag([0.5, 0.5], [[-1], [1]], [1e-9, 1e-9]), pt = posteriorDiag(two, [0.5], [0.6], 0.64);
  check("tanh example by the posterior of the mixture", close(posteriorMean(pt)[0], Math.tanh(0.6 * 0.5 / 0.64), 1e-6));
  check("cosine schedule: ā(0) = 1, ā(1) ≈ 0", close(alphaBarAt(0), 1) && alphaBarAt(1, "cosine") < 1e-3 && alphaBarAt(1, "linear") < 1e-3);
}

// ---- diffusion2d -----------------------------------------------------------------------
{
  const ds = DF.dataSet("two");
  const run = DF.sample(DF.exactScore(ds, null, 0), 2000, 200, "ddpm", "cosine", 3);
  const s = DF.summarise(ds, run.frames[run.N]);
  check("DDPM with the exact score: half the samples per blob, few off the data", Math.abs(s.perGroup[0] - 0.5) < 0.05 && s.off < 0.03, JSON.stringify(s));
  const dd = DF.sample(DF.exactScore(ds, null, 0), 2000, 200, "ddim", "cosine", 3);
  const s2 = DF.summarise(ds, dd.frames[dd.N]);
  check("DDIM with the exact score: same", Math.abs(s2.perGroup[0] - 0.5) < 0.05 && s2.off < 0.03, JSON.stringify(s2));
  const c = DF.sample(DF.exactScore(ds, 1, 0), 1000, 100, "ddpm", "cosine", 3);
  check("conditioned on the right blob", DF.summarise(ds, c.frames[c.N]).perGroup[1] > 0.98);
  const tw = DF.dataSet("twenty"), cp = DF.sample(DF.exactScore(tw, null, 0), 500, 200, "ddpm", "cosine", 3);
  check("perfect score of 20 points: samples copy training points", DF.summarise(tw, cp.frames[cp.N]).nearTrain > 0.95);
  const rare = DF.dataSet("rare"), lr = DF.sample(DF.learnedScore(SCORE_NETS.rare.trained), 1000, 100, "ddpm", "cosine", 3);
  const sr = DF.summarise(rare, lr.frames[lr.N]);
  check("learned score (trained) puts most samples on the data", sr.off < 0.15, JSON.stringify(sr));
}

// ---- genrestore ------------------------------------------------------------------------
{
  const P = GR.makePrior("matched");
  // a training image measured with almost no noise: the posterior picks it
  const x = P.images[17], m = GR.measure(x, "blur", 0.5, 0.005, new Uint8Array(GR.D));
  const R = GR.restore(P, m, 4);
  check("genrestore: a training cell measured sharply is recognised", R.post.w[17] > 0.99, String(R.post.w[17]));
  // the pixel-basis and DCT-basis posteriors agree for pure noise (A = I)
  const y = GR.measure(GR.cell({ kind: "plain", nucleoli: 1, seed: 3 }), "inpaint", 0, 0.05, new Uint8Array(GR.D));
  const Rp = GR.restore(P, y, 1), Pd = posteriorDiag(P.dct, Array.from(dct2(y.y, GR.N, GR.N)), new Array(GR.D).fill(1), 0.0025);
  let dw = 0; for (let k = 0; k < Pd.k; k++) dw = Math.max(dw, Math.abs(Pd.w[k] - Rp.post.w[k]));
  check("pixel and DCT bases give the same posterior weights (orthonormal DCT)", dw < 1e-6, String(dw));
  const mc = GR.measureCell(GR.cell({ kind: "three", nucleoli: 3, seed: 11 }));
  check("the measuring of a cell: 3 nucleoli, 1 nucleus", mc.nucleoli === 3 && mc.nuclei === 1, JSON.stringify(mc));
}

console.log(`moderndl: ${passes} passed, ${fails} failed`);
if (fails) (globalThis as unknown as { process: { exit(c: number): void } }).process.exit(1);
