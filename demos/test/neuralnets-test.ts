// Tests for the neural-networks day (neuralnet/, backprop/, cnn/, unet/): lib/nn.ts (old
// behaviour bit-identical, new gradients against finite differences and lib/autodiff.ts),
// lib/autodiff.ts, lib/cnn.ts against naive loops and kernel3x3/conv.ts, lib/q8.ts, the int8
// networks of cnnviewer and unetseg against the reference outputs of their training scripts,
// and the by-hand numbers quoted in the exercises. Run alone:
//   node_modules/.bin/esbuild demos/test/neuralnets-test.ts --bundle --platform=node --log-level=warning | node
import { Adam, mlp, trainStep, predict, gradients, meanLoss, softmax, Batcher, Sgd, applyGradients, type Act, type Loss } from "../src/lib/nn";
import { Graph, finiteDiff } from "../src/lib/autodiff";
import { conv2d, relu, maxPool, upsample, convTranspose2d, concat, dense, tensor, receptiveField, outSize, type ConvLayer } from "../src/lib/cnn";
import { deq, bytesFromB64 } from "../src/lib/q8";
import { filter3 } from "../src/kernel3x3/conv";
import { profileData, initHinges } from "../src/nnfit/data";
import { makeAE, makeData } from "../src/linae/data";
import { build, lossOf, update } from "../src/backprop/model";
import { LANDSCAPES, lossGrad } from "../src/gdescent/model";
import { Trainer, DEFAULT } from "../src/nnplayground/model";
import { dataset } from "../src/lib/datasets2d";
import { runCnn } from "../src/cnnviewer/net";
import { CNN_REF } from "../src/cnnviewer/cnn_weights";
import { runUnet, countWeights } from "../src/unetseg/net";
import { sample } from "../src/unetseg/data";
import { UNET_REF } from "../src/unetseg/unet_weights";
import { losses } from "../src/unetseg/model";
import { rng } from "../src/lib/ui";

declare const process: { exit(code: number): never };
declare const require: (m: string) => any;
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 25) console.log("FAIL: " + msg); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

// ---- 1. lib/nn.ts: the original nnfit and linae training is unchanged (bit for bit) ----
{
  const { X, Y } = profileData();
  const out: number[] = [];
  for (const [h, lr] of [[3, 0.02], [12, 0.02], [12, 0.5]]) {
    const net = mlp([1, h, 1], ["relu", "id"], 11); initHinges(net, Y, 11);
    const opt = new Adam(net, lr);
    for (let i = 0; i < 700; i++) out.push(trainStep(net, X, Y, opt));
    out.push(predict(net, [0.3])[0]);
  }
  const { net, opt } = makeAE(); const P = makeData("spot");
  for (let i = 0; i < 300; i++) out.push(trainStep(net, P, P, opt));
  const md5 = require("crypto").createHash("md5").update(JSON.stringify(out)).digest("hex");
  // fingerprint recorded with lib/nn.ts before the neural-networks day extended it
  check(md5 === "a5d02877d3cf8b7ea3b28980c35fc519", `nnfit/linae training unchanged (md5 ${md5})`);
}

// ---- 2. lib/nn.ts gradients against finite differences ----
{
  const r = rng(5);
  const cases: [number[], Act[], Loss][] = [
    [[2, 3, 1], ["tanh", "id"], "mse"], [[2, 4, 1], ["sigmoid", "id"], "bce"], [[3, 4, 3, 3], ["relu", "tanh", "id"], "ce"], [[2, 5, 2], ["relu", "id"], "mse"],
  ];
  for (const [sizes, acts, loss] of cases) {
    const net = mlp(sizes, acts, 3);
    const X = Array.from({ length: 7 }, () => Array.from({ length: sizes[0] }, () => r.norm()));
    const nOut = sizes[sizes.length - 1];
    const Y = X.map(() => {
      if (loss === "mse") return Array.from({ length: nOut }, () => r.norm());
      if (loss === "bce") return [r.unif() < 0.5 ? 0 : 1];
      const c = Math.floor(r.unif() * nOut); return Array.from({ length: nOut }, (_, j) => (j === c ? 1 : 0));
    });
    for (const decay of [0, 0.01]) {
      const g = gradients(net, X, Y, { loss, decay });
      check(close(g.loss, meanLoss(net, X, Y, loss, decay), 1e-12), `${sizes} ${loss}: loss matches meanLoss`);
      let worst = 0;
      net.W.forEach((W, l) => {
        for (let k = 0; k < W.length; k++) {
          const f = (v: number[]) => { const old = W[k]; W[k] = v[0]; const L = meanLoss(net, X, Y, loss, decay); W[k] = old; return L; };
          worst = Math.max(worst, Math.abs(finiteDiff(f, [W[k]], 0, 1e-6) - g.gW[l][k]));
        }
        for (let k = 0; k < net.b[l].length; k++) {
          const B = net.b[l];
          const f = (v: number[]) => { const old = B[k]; B[k] = v[0]; const L = meanLoss(net, X, Y, loss, decay); B[k] = old; return L; };
          worst = Math.max(worst, Math.abs(finiteDiff(f, [B[k]], 0, 1e-6) - g.gb[l][k]));
        }
      });
      check(worst < 1e-6, `${sizes} ${acts} ${loss} decay ${decay}: gradients vs finite differences (worst ${worst})`);
    }
  }
  // the old trainStep (Adam on mse) and gradients + applyGradients give the same step
  const a = mlp([2, 3, 1], ["relu", "id"], 9), b = mlp([2, 3, 1], ["relu", "id"], 9);
  const X = [[0.3, -0.2], [1, 0.5], [-0.7, 0.1]], Y = [[0.2], [-0.4], [1]];
  const oa = new Adam(a, 0.05), ob = new Adam(b, 0.05);
  for (let i = 0; i < 20; i++) { trainStep(a, X, Y, oa); applyGradients(b, gradients(b, X, Y, { loss: "mse" }), ob); }
  check(a.W.every((W, l) => W.every((v, k) => close(v, b.W[l][k], 1e-12))), "trainStep = gradients + applyGradients (Adam)");
  // dropout: with p = 0 nothing changes; the inverted mask keeps the mean
  const g0 = gradients(a, X, Y, { loss: "mse" }), g1 = gradients(a, X, Y, { loss: "mse", dropout: 0, rand: () => 0.5 });
  check(g0.gW.every((W, l) => W.every((v, k) => v === g1.gW[l][k])), "dropout 0 = no dropout");
  // SGD with momentum: v = βv + g, w -= ηv
  const c = mlp([1, 1], ["id"], 1); c.W[0][0] = 1; c.b[0][0] = 0;
  const sgd = new Sgd(c, 0.1, 0.9);
  const gr = { gW: [Float64Array.of(2)], gb: [Float64Array.of(0)], loss: 0 };
  applyGradients(c, gr, sgd); applyGradients(c, gr, sgd);
  check(close(c.W[0][0], 1 - 0.1 * 2 - 0.1 * (0.9 * 2 + 2)), "SGD momentum update");
  // batcher: every index exactly once per epoch
  const bt = new Batcher(10, 3, 4), seen: number[] = [];
  for (let i = 0; i < 3; i++) seen.push(...bt.next());
  check(new Set(seen).size === 9 && seen.every(v => v >= 0 && v < 10), "batcher: no repeats within an epoch");
  // softmax: invariance to a constant, the by-hand numbers
  const s1 = softmax([2, 1, 0]), s2 = softmax([102, 101, 100]);
  check(close(s1[0], s2[0], 1e-12) && close(s1[0], 0.6652409557748219, 1e-12) && close(s1[2], 0.09003057317038046, 1e-12), "softmax (2, 1, 0) = (0.665, 0.245, 0.090), shift invariant");
}

// ---- 3. lib/autodiff.ts against finite differences, and against lib/nn.ts ----
{
  const r = rng(11);
  for (let t = 0; t < 20; t++) {
    const v = Array.from({ length: 6 }, () => r.norm());
    const f = (q: number[]) => {
      const g = new Graph();
      const [a, b, c, d, e, h] = q.map((x, i) => g.param("p" + i, x));
      const z1 = g.affine([a, b], [c, d], e), z2 = g.tanh(g.mul(z1, h)), z3 = g.sigmoid(g.add2(z1, g.scale(h, 0.5)));
      const L = g.add2(g.sqerr(z2, g.const(0.3)), g.softmaxCE([z1, z2, z3], 1));
      return { g, L, ps: [a, b, c, d, e, h] };
    };
    const { g, L, ps } = f(v);
    g.backward(L);
    let worst = 0;
    ps.forEach((p, i) => worst = Math.max(worst, Math.abs(p.grad - finiteDiff(q => f(q).L.value, v, i, 1e-6))));
    check(worst < 1e-6, `autodiff vs finite differences (worst ${worst})`);
  }
  // a 2-2-1 ReLU network: autodiff (backprop demo graph) = lib/nn.ts gradients
  const vals = { x1: 1, x2: 2, y: 2, w11: 1, w12: 0.5, b1: 0.5, w21: 0.5, w22: 0.5, b2: -1, v1: 1, v2: 2, c: 0 };
  const bg = build("221", vals, "relu"); bg.g.backward(bg.L);
  const net = mlp([2, 2, 1], ["relu", "id"], 1);
  net.W[0].set([1, 0.5, 0.5, 0.5]); net.b[0].set([0.5, -1]); net.W[1].set([1, 2]); net.b[1].set([0]);
  const gn = gradients(net, [[1, 2]], [[2]], { loss: "mse" });
  const want = [gn.gW[0][0], gn.gW[0][1], gn.gb[0][0], gn.gW[0][2], gn.gW[0][3], gn.gb[0][1], gn.gW[1][0], gn.gW[1][1], gn.gb[1][0]];
  check(bg.params.every((p, i) => close(p.grad, want[i], 1e-12)), "backprop demo graph = lib/nn.ts gradients (2-2-1)");
}

// ---- 4. by-hand numbers (backprop/exercise.qmd B3, lec) ----
{
  const v = { x: 2, y: 1, w1: 0.5, b1: 0, w2: 1.5, b2: 0 };
  const b = build("111", v, "relu"); b.g.backward(b.L);
  const grad = Object.fromEntries(b.params.map(p => [p.name, p.grad]));
  check(b.L.value === 0.25 && grad.w1 === 3 && grad.b1 === 1.5 && grad.w2 === 1 && grad.b2 === 1, "B3 forward/backward: L 0.25, grads 3, 1.5, 1, 1");
  check(close(lossOf("111", update("111", v, "relu", 0.1).next, "relu"), 0.5625, 1e-12), "B3 update η = 0.1: L = 0.5625");
  check(Math.abs(lossOf("111", update("111", v, "relu", 0.05).next, "relu") - 0.0207) < 1e-4, "B3 update η = 0.05: L ≈ 0.021");
  const d = build("111", { ...v, x: -1 }, "relu"); d.g.backward(d.L);
  check(d.params.map(p => p.grad).join(",") === "0,0,0,-2", "B3(d) x = −1: only b2 gets a gradient");
  check(close(lossOf("111", update("111", { ...v, x: -1 }, "relu", 0.1).next, "relu"), 0.64, 1e-12), "B3(d) update: L = 0.64");
  // gdescent: the bowl's stability limit 2/λmax ≈ 0.75 (exercise G2)
  const L = LANDSCAPES[0], h = 1e-4;
  const H = [[0, 0], [0, 0]];
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
    const e = [0, 0]; e[j] = h;
    H[i][j] = (lossGrad(L, 1 + e[0], 0.4 + e[1])[1 + i] - lossGrad(L, 1 - e[0], 0.4 - e[1])[1 + i]) / (2 * h);
  }
  const tr = H[0][0] + H[1][1], dt = Math.sqrt(((H[0][0] - H[1][1]) / 2) ** 2 + H[0][1] * H[0][1]);
  check(Math.abs(2 / (tr / 2 + dt) - 0.75) < 0.01, `gdescent bowl: 2/λmax ≈ 0.75 (${2 / (tr / 2 + dt)})`);
  // nnplayground: the overfitting preset's learning curve (exercise B4/E1)
  const t = new Trainer(dataset("moons", { n: 60, noise: 0.1, seed: 1 }).points, 2, { ...DEFAULT, hidden: [8, 8], lr: 0.1, valFrac: 0.5 });
  t.train(2000);
  let best = 0; t.history.forEach((q, i) => { if (q.val < t.history[best].val) best = i; });
  check(t.history[best].step === 126 && Math.abs(t.history[best].val - 0.169) < 5e-4, "overfitting preset: lowest validation loss 0.169 at step 126");
}

// ---- 5. lib/cnn.ts against naive loops ----
{
  const r = rng(21);
  const naive = (x: { c: number; h: number; w: number; data: Float32Array }, L: ConvLayer, p: number, s: number, d: number) => {
    const H = outSize(x.h, L.k, p, s, d), W = outSize(x.w, L.k, p, s, d), out = new Float64Array(L.cout * H * W);
    for (let o = 0; o < L.cout; o++) for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
      let v = L.b[o];
      for (let c = 0; c < x.c; c++) for (let u = 0; u < L.k; u++) for (let q = 0; q < L.k; q++) {
        const yy = i * s + u * d - p, xx = j * s + q * d - p;
        if (yy >= 0 && yy < x.h && xx >= 0 && xx < x.w) v += L.W[((o * L.cin + c) * L.k + u) * L.k + q] * x.data[(c * x.h + yy) * x.w + xx];
      }
      out[(o * H + i) * W + j] = v;
    }
    return out;
  };
  for (const [cin, cout, k, h, w, p, s, d] of [[1, 2, 3, 6, 7, 1, 1, 1], [3, 4, 3, 9, 8, 1, 1, 1], [2, 3, 3, 9, 9, 0, 2, 1], [2, 2, 5, 11, 10, 2, 1, 1], [1, 2, 3, 12, 12, 2, 1, 2], [2, 3, 3, 8, 8, 1, 2, 1]]) {
    const x = tensor(cin, h, w, Float32Array.from({ length: cin * h * w }, () => r.norm()));
    const L: ConvLayer = { W: Float32Array.from({ length: cout * cin * k * k }, () => r.norm()), b: Float32Array.from({ length: cout }, () => r.norm()), cin, cout, k };
    const y = conv2d(x, L, { pad: p, stride: s, dilation: d }), ref = naive(x, L, p, s, d);
    check(y.data.length === ref.length && y.data.every((v, i) => Math.abs(v - ref[i]) < 1e-4), `conv2d ${cin}->${cout} k${k} p${p} s${s} d${d} vs naive`);
  }
  // against kernel3x3/conv.ts (one channel, zero border)
  const img = Array.from({ length: 30 }, () => r.unif() * 255), kern = [1, 2, -1, 0, 3, 1, -2, 0, 1];
  const f3 = filter3(img, 6, 5, kern, 1, "zero"), c3 = conv2d(tensor(1, 5, 6, Float32Array.from(img)), { W: Float32Array.from(kern), b: Float32Array.of(0), cin: 1, cout: 1, k: 3 });
  check(Array.from(f3).every((v, i) => Math.abs(v - c3.data[i]) < 1e-3), "conv2d = kernel3x3 filter3 (zero border)");
  // pooling, upsampling, concat, dense
  const t = tensor(1, 4, 4, Float32Array.from([1, 2, 0, 0, 3, 4, 0, 5, 0, 0, 1, 1, 6, 0, 1, 2]));
  check(Array.from(maxPool(t).data).join() === "4,5,6,2", "maxPool 2×2");
  check(Array.from(upsample(tensor(1, 1, 2, Float32Array.of(1, 2))).data).join() === "1,1,2,2,1,1,2,2", "nearest upsampling");
  check(concat(t, t).c === 2 && concat(t, t).data[16] === 1, "concat");
  check(Array.from(dense([1, 2], Float32Array.of(1, 1, 2, -1), Float32Array.of(0, 1), 2)).join() === "3,1", "dense");
  // transposed convolution is the adjoint of the strided convolution
  for (const [n, k, s] of [[9, 3, 2], [8, 2, 2], [10, 4, 3]]) {
    const K = Float32Array.from({ length: k }, () => r.norm()), x = Float32Array.from({ length: n }, () => r.norm());
    // 1D: a k × k kernel whose first row is K; on a 1-row input only that row matters
    const K2 = new Float32Array(k * k); for (let q = 0; q < k; q++) K2[q] = K[q];
    const m = Math.floor((n - k) / s) + 1, y = Float32Array.from({ length: m }, () => r.norm());
    let lhs = 0; for (let i = 0; i < m; i++) { let v = 0; for (let q = 0; q < k; q++) v += K[q] * x[i * s + q]; lhs += v * y[i]; }
    const Ty = convTranspose2d(tensor(1, 1, m, y), K2, Float32Array.of(0), 1, k, s, 0);
    // row 0 of the transposed output carries the 1D result
    let rhs = 0; for (let t2 = 0; t2 < n; t2++) rhs += x[t2] * (t2 < Ty.w ? Ty.data[t2] : 0);
    check(Math.abs(lhs - rhs) < 1e-4, `transposed convolution is the adjoint (n ${n}, k ${k}, s ${s})`);
  }
  // receptive fields quoted in cnn/ and unet/
  check(receptiveField([{ k: 3 }, { k: 3 }, { k: 3 }]).r[3] === 7, "three 3×3 layers: 7");
  check(receptiveField([{ k: 3 }, { k: 3 }, { k: 2, s: 2 }, { k: 3 }, { k: 3 }, { k: 2, s: 2 }, { k: 3 }, { k: 3 }]).r.join() === "1,3,5,6,10,14,16,24,32", "U-Net bottleneck: 32");
  check(receptiveField([{ k: 3, d: 1 }, { k: 3, d: 2 }, { k: 3, d: 4 }]).r[3] === 15, "dilated 1, 2, 4: 15");
  // cnn/exercise.qmd B1
  const I = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1];
  const B1 = relu(conv2d(tensor(1, 6, 6, Float32Array.from(I)), { W: Float32Array.from([-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1]), b: new Float32Array(2), cin: 1, cout: 2, k: 3 }, { pad: "valid" }));
  check(Array.from(B1.data).join() === "0,3,3,0,1,3,2,0,2,3,1,0,3,3,0,0,0,0,0,0,1,1,1,0,1,1,1,0,0,0,0,0", "cnn B1 feature maps");
  check(Array.from(maxPool(B1).data).join() === "3,3,3,1,1,1,1,1", "cnn B1 pooled");
  // unet/exercise.qmd B2(c)
  const up = convTranspose2d(tensor(1, 1, 3, Float32Array.of(1, 2, 3)), Float32Array.from([0.5, 1, 0.5, 0, 0, 0, 0, 0, 0]), Float32Array.of(0), 1, 3, 2, 0);
  check(Array.from(up.data.slice(0, 7)).join() === "0.5,1,1.5,2,2.5,3,1.5", "unet B2 transposed convolution");
}

// ---- 6. q8 and the trained networks against their training scripts ----
{
  const l = deq({ q: btoa(String.fromCharCode(1, 255, 127, 129)), s: [0.5, 2], b: [0, 1], nIn: 2, nOut: 2 });
  check(Array.from(l.W).join() === "0.5,-0.5,254,-254", "q8 dequantisation (signed bytes, scale per row)");
  CNN_REF.digitsTestImages.forEach((b64: string, i: number) => {
    const lo = runCnn(Float32Array.from(bytesFromB64(b64), v => v / 255), "digits").logits;
    check(CNN_REF.digitsLogits[i].every((v: number, k: number) => Math.abs(v - lo[k]) < 2e-3), `cnnviewer digits logits = training script (image ${i})`);
  });
  const s = sample(200001);
  for (const set of ["unet", "noskip", "dice"] as const) {
    const p = runUnet(s.image, set).prob, m = p.reduce((a, b) => a + b, 0) / p.length;
    check(Math.abs(m - UNET_REF[set].mean) < 2e-4, `unetseg ${set}: mean probability ${m} = training script ${UNET_REF[set].mean}`);
  }
  check(countWeights("unet") === 29617 && countWeights("noskip") === 26737, "U-Net weight counts (unet B3)");
  // unet/exercise.qmd B1
  const pr = [0.9, 0.8, 0.1, 0.7, 0.6, 0.2, 0.1, 0.2, 0.1], g = [1, 1, 0, 1, 1, 0, 0, 0, 0], lo = losses(pr, g);
  check(Math.abs(lo.ce - 0.2176) < 1e-4 && Math.abs(lo.softDice - 6 / 7.7) < 1e-9, "unet B1: CE 0.218, soft Dice 0.779");
}

console.log(`neuralnets: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
