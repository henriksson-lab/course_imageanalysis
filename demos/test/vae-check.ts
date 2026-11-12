// Numbers quoted in the VAE units (neuralnet, autoencoder, vae, vae-gen), computed with exactly
// the code the demos run. Not part of `make test` (it needs the MNIST and synthetic test sets).
//
//   python3 demos/src/vaeimages/make_digit_examples.py MNIST.npz demos/src/vaeimages/digit_examples.ts DIR
//   python3 demos/src/vaeimages/train_shapes.py demos/src/vaeimages/shape_weights.ts DIR
//   node_modules/.bin/esbuild demos/test/vae-check.ts --bundle --platform=node --outfile=DIR/vae-check.cjs
//   node DIR/vae-check.cjs DIR
// Also checks the KL formula of elbo1d against numerical integration and its closed-form best
// cloud against a grid search.
declare const require: (m: string) => any;
declare const process: { argv: string[] };
const fs = require("fs");
import { digitVae, shapeVae, mse, fromB64, type ImageVae, type V2 } from "../src/vaeimages/imagevae";
import { MNIST_POINTS } from "../src/vaeimages/mnist_weights";
import { SHAPE_POINTS, SHAPE_KINDS, SHAPE_EXAMPLES } from "../src/vaeimages/shape_weights";
import { DIGIT_EXAMPLES } from "../src/vaeimages/digit_examples";
import { DEFAULT, EPS, best, kl, loss, recon, reconMC } from "../src/elbo1d/model";
import { profileData, initHinges } from "../src/nnfit/data";
import { Adam, mlp, predict, trainStep } from "../src/lib/nn";
import { makeData, pca, linError, linEncode, linDecode, makeAE, aeTrain, aeError } from "../src/linae/data";
import { rng } from "../src/lib/ui";

const DIR = process.argv[2];
const f = (v: number, d = 4) => v.toFixed(d);
const pct = (a: number[], q: number) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };

// ---------------------------------------------------------------- elbo1d
console.log("== elbo1d (x = %s, w = %s, sx = %s)", DEFAULT.x, DEFAULT.w, DEFAULT.sx);
{
  // KL vs numerical integration
  for (const [m, s] of [[0, 1], [1, 1], [0, 0.5], [1.2, Math.sqrt(0.2)], [2, 0.1]]) {
    let num = 0; const h = 1e-3;
    for (let z = -12; z < 12; z += h) { const q = Math.exp(-0.5 * ((z - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI)); const p = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI); if (q > 1e-300) num += q * Math.log(q / p) * h; }
    console.log(`KL(mu=${f(m, 3)}, s=${f(s, 3)}) formula ${f(kl(m, s))} numeric ${f(num)}`);
  }
  for (const beta of [1, 5, 0.2]) {
    const p = { ...DEFAULT, beta }, b = best(p);
    let gb = { v: Infinity, m: 0, s: 0 };
    for (let m = -3; m <= 3; m += 0.005) for (let s = 0.01; s <= 1.5; s += 0.0025) { const v = loss(p, m, s); if (v < gb.v) gb = { v, m, s }; }
    console.log(`beta ${beta}: best mu ${f(b.mu)} s ${f(b.s)} s² ${f(b.s ** 2)} R ${f(recon(p, b.mu, b.s))} KL ${f(kl(b.mu, b.s))} total ${f(loss(p, b.mu, b.s))} | grid ${f(gb.m, 3)} ${f(gb.s, 3)} ${f(gb.v)}`);
  }
  const p = DEFAULT;
  for (const [nm, m, s] of [["prior", 0, 1], ["too confident", p.x, 0.05], ["mu=1,s=0.5", 1, 0.5]] as [string, number, number][])
    console.log(`${nm}: R ${f(recon(p, m, s))} (MC ${f(reconMC(p, m, s))}) KL ${f(kl(m, s))} total ${f(loss(p, m, s))}`);
  console.log("EPS[0..2]", EPS.slice(0, 3).map(v => f(v)).join(" "));
}

// ---------------------------------------------------------------- nnfit
console.log("== nnfit");
{
  const { X, Y } = profileData();
  const L = (n: any) => { let s = 0; X.forEach((x, i) => { s += (predict(n, x)[0] - Y[i][0]) ** 2; }); return s / X.length; };
  let mY = 0; Y.forEach(y => mY += y[0]); mY /= Y.length;
  console.log(`mean ${f(mY)} variance (loss of the flat line at the mean) ${f(Y.reduce((s, y) => s + (y[0] - mY) ** 2, 0) / Y.length, 5)}; noise variance 0.03² = 0.0009`);
  for (const [h, lr, bend, steps] of [[1, 0.02, true, 3000], [3, 0.02, true, 3000], [12, 0.02, true, 3000], [12, 0.02, false, 3000], [12, 0.5, true, 500]] as [number, number, boolean, number][]) {
    const net = mlp([1, h, 1], [bend ? "relu" : "id", "id"], 11); initHinges(net, Y, 11);
    const opt = new Adam(net, lr), l0 = L(net);
    for (let s = 0; s < steps; s++) trainStep(net, X, Y, opt);
    const alive = Array.from({ length: h }, (_, k) => X.some(x => net.W[0][k] * x[0] + net.b[0][k] > 0)).filter(Boolean).length;
    console.log(`H ${h} lr ${lr} bend ${bend} steps ${steps}: loss ${f(l0, 5)} → ${f(L(net), 5)}; weights+shifts ${3 * h + 1}; units active somewhere ${alive}`);
  }
}

// ---------------------------------------------------------------- linae
console.log("== linae");
for (const kind of ["pixels", "spot"] as const) {
  const P = makeData(kind), pc = pca(P);
  console.log(`${kind}: mean (${f(pc.m[0], 3)}, ${f(pc.m[1], 3)}) cov a ${f(pc.a)} b ${f(pc.b)} c ${f(pc.c)}; PCA angle ${f(pc.theta * 180 / Math.PI, 1)}°; l1 ${f(pc.l1)} l2 ${f(pc.l2)}; total ${f(pc.l1 + pc.l2)}`);
  console.log(`  error at 0° ${f(linError(P, 0))}, 45° ${f(linError(P, Math.PI / 4))}, -45° ${f(linError(P, -Math.PI / 4))}, 90° ${f(linError(P, Math.PI / 2))}, PCA ${f(linError(P, pc.theta))}`);
  const sel = kind === "spot" ? 20 : 0, x = P[sel];
  for (const th of [0, pc.theta]) { const s = linEncode(x, pc.m, th), q = linDecode(s, pc.m, th); console.log(`  selected #${sel} x = (${f(x[0], 3)}, ${f(x[1], 3)}), angle ${f(th * 180 / Math.PI, 1)}: s ${f(s, 3)} x̂ (${f(q[0], 3)}, ${f(q[1], 3)})`); }
  const { net, opt } = makeAE();
  aeTrain(net, opt, P, 1500); const e1 = aeError(net, P);
  aeTrain(net, opt, P, 1500); const e2 = aeError(net, P);
  console.log(`  nonlinear AE error after 1500 steps ${f(e1)}, after 3000 ${f(e2)}`);
}

// ---------------------------------------------------------------- image VAEs
function load(name: string): Float32Array[] {
  const b: Uint8Array = fs.readFileSync(`${DIR}/${name}`), n = b.length / 784, out: Float32Array[] = [];
  for (let i = 0; i < n; i++) out.push(Float32Array.from(b.subarray(784 * i, 784 * (i + 1)), v => v / 255));
  return out;
}
function centres(points: [number, number, number][], n: number): V2[] {
  return Array.from({ length: n }, (_, d) => {
    const p = points.filter(q => q[2] === d);
    const med = (k: 0 | 1) => p.map(q => q[k]).sort((a, b) => a - b)[Math.floor(p.length / 2)];
    return [med(0), med(1)] as V2;
  });
}
function study(nm: string, vae: ImageVae, X: Float32Array[], y: Uint8Array, classes: string[], points: [number, number, number][], examples: { label: string; img: string }[]): void {
  console.log(`== ${nm}: ${X.length} test images`);
  const enc = X.map(x => vae.encode(x));
  const err = X.map((x, i) => mse(x, vae.decode(enc[i].mu)));
  console.log(`recon MSE per pixel: mean ${f(err.reduce((a, b) => a + b, 0) / err.length)} median ${f(pct(err, 0.5))} 95% ${f(pct(err, 0.95))} 99% ${f(pct(err, 0.99))} max ${f(Math.max(...err))}`);
  const sds = enc.map(e => (e.sd[0] + e.sd[1]) / 2);
  console.log(`encoder sd: mean ${f(sds.reduce((a, b) => a + b, 0) / sds.length, 3)} median ${f(pct(sds, 0.5), 3)} 95% ${f(pct(sds, 0.95), 3)}`);
  const r2 = enc.map(e => e.mu[0] ** 2 + e.mu[1] ** 2);
  console.log(`fraction of encoder means inside radius 2: ${f(r2.filter(v => v < 4).length / r2.length, 3)}; inside [-4.2,4.2]²: ${f(enc.filter(e => Math.abs(e.mu[0]) < 4.2 && Math.abs(e.mu[1]) < 4.2).length / enc.length, 3)}`);
  const c = centres(points, classes.length);
  c.forEach((p, d) => {
    const im = vae.decode(p);
    const ink = im.reduce((a, b) => a + b, 0);
    const errs = err.filter((_, i) => y[i] === d);
    console.log(`  class ${classes[d]}: centre (${f(p[0], 2)}, ${f(p[1], 2)}) ink of decoded centre ${f(ink, 1)}; mean recon error ${f(errs.reduce((a, b) => a + b, 0) / errs.length)}`);
  });
  // 15-NN leave-one-out accuracy in the latent space (on at most 4000 test images)
  const n = Math.min(4000, X.length); let ok = 0;
  for (let i = 0; i < n; i++) {
    const d: [number, number][] = [];
    for (let j = 0; j < n; j++) if (j !== i) d.push([(enc[i].mu[0] - enc[j].mu[0]) ** 2 + (enc[i].mu[1] - enc[j].mu[1]) ** 2, y[j]]);
    d.sort((a, b) => a[0] - b[0]);
    const votes = new Array(classes.length).fill(0); for (let k = 0; k < 15; k++) votes[d[k][1]]++;
    if (votes.indexOf(Math.max(...votes)) === y[i]) ok++;
  }
  console.log(`15-NN accuracy (leave one out, first ${n} test images): ${f(ok / n, 3)}`);
  examples.forEach((e, k) => {
    const img = fromB64(e.img), q = vae.encode(img), rec = vae.decode(q.mu);
    let line = `  ex ${k + 1} ${e.label}: mu (${f(q.mu[0], 2)}, ${f(q.mu[1], 2)}) sd (${f(q.sd[0], 3)}, ${f(q.sd[1], 3)}) error ${f(mse(img, rec))}`;
    for (const nz of [0.3]) {
      const r = rng(1001); const inp = img.map(v => Math.max(0, Math.min(1, v + nz * r.norm())));
      const qn = vae.encode(inp);
      line += ` | noise ${nz}: input vs clean ${f(mse(img, inp))}, output vs clean ${f(mse(img, vae.decode(qn.mu)))}`;
    }
    console.log(line);
  });
  // midpoint vs pixel average between two class centres
  const [a, b] = nm === "digits" ? [c[1], c[7]] : [c[0], c[2]];
  const A = vae.decode(a), B = vae.decode(b), Mid = vae.decode([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]);
  const avg = A.map((v, i) => (v + B[i]) / 2);
  const half = (im: Float32Array) => im.filter(v => v > 0.25 && v < 0.75).length;
  console.log(`  midpoint vs average: pixels with ink in (0.25, 0.75): A ${half(A)} B ${half(B)} midpoint ${half(Mid)} average ${half(avg)}`);
  // prior samples as in the demo (seed 2 for the first press)
  const r = rng(2); const zs = Array.from({ length: 7 }, () => [r.norm(), r.norm()]);
  console.log(`  first "sample from the prior" z: ${zs.map(z => `(${f(z[0], 2)}, ${f(z[1], 2)})`).join(" ")}`);
}
study("digits", digitVae, load("mnist_test_x.u8"), fs.readFileSync(`${DIR}/mnist_test_y.u8`), Array.from({ length: 10 }, (_, d) => String(d)), MNIST_POINTS, DIGIT_EXAMPLES);
study("shapes", shapeVae, load("shapes_test_x.u8"), fs.readFileSync(`${DIR}/shapes_test_y.u8`), SHAPE_KINDS, SHAPE_POINTS, SHAPE_EXAMPLES);
// far outside
for (const [nm, v, z] of [["digits", digitVae, [3.3, -3.3]], ["shapes", shapeVae, [3.3, 3.3]]] as [string, ImageVae, V2][]) {
  const im = v.decode(z); console.log(`${nm} far outside ${z}: ink ${f(im.reduce((a, b) => a + b, 0), 1)}, density of z under prior relative to 0: ${f(Math.exp(-0.5 * (z[0] ** 2 + z[1] ** 2)), 5)}`);
}
