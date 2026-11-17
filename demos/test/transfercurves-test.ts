// Tests of the transfer unit's toy (demos/src/transfercurves/toy.ts): the allocation-free
// backpropagation of LayerAdam against lib/nn.ts gradients, frozen layers that do not move,
// cached features, the data, and the numbers quoted on the unit's pages (9 labels). Run:
//   node_modules/.bin/esbuild demos/test/transfercurves-test.ts --bundle --platform=node --log-level=warning | node
import * as T from "../src/transfercurves/toy";
import { gradients, cloneMlp, mlp } from "../src/lib/nn";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

current = "data";
{
  const a = T.makeSet("target", 6, 77), b = T.makeSet("target", 6, 77);
  check(a.X.every((x, i) => x.every((v, j) => v === b.X[i][j])), "same seed, same patches");
  check(a.y.join() === "0,1,2,0,1,2" && a.k === 3, "balanced classes in turn");
  // the same seed through another look: same objects. With no noise, gain 2 doubles the object part.
  const l1 = { gain: 1, offset: 0, blur: 0, noise: 0 }, l2 = { gain: 2, offset: 0.1, blur: 0, noise: 0 };
  const p = T.makeSet("source", 4, 5, l1), q = T.makeSet("source", 4, 5, l2);
  let ok = true;
  for (let i = 0; i < 4; i++) for (let j = 0; j < T.NPIX; j++) ok &&= close(q.X[i][j], 0.08 + 0.1 + 2 * (p.X[i][j] - 0.08), 1e-12);
  check(ok, "a look changes gain and offset of the same objects");
  const bl = T.blurPatch(Float64Array.from({ length: T.NPIX }, () => 0.7), 1.2);
  check(bl.every(v => close(v, 0.7, 1e-12)), "blurring a constant patch keeps it");
}

current = "backprop";
{
  // LayerAdam's loss equals lib/nn's; lib/nn's gradient agrees with finite differences; and the
  // first Adam step of LayerAdam moves each trained weight by the rate in the gradient's direction.
  const net = mlp([T.NPIX, 6, 5, 3], ["relu", "relu", "id"], 3);
  const d = T.makeSet("target", 9, 8);
  const Y = d.y.map(c => [0, 1, 2].map(j => (j === c ? 1 : 0)));
  const g = gradients(net, d.X, Y, { loss: "ce" });
  // LayerAdam with rates 0 everywhere but one weight's layer: loss must equal lib's loss
  const opt = new T.LayerAdam(cloneMlp(net), [1e-9, 1e-9, 1e-9]);
  const loss = opt.step(d.X, d.y);
  check(close(loss, g.loss, 1e-12), `loss ${loss} vs lib/nn ${g.loss}`);
  // finite-difference check of lib's gradient for a few weights through T.evaluate
  const lossOf = (n: typeof net) => T.evaluate(n, d).loss;
  for (const [l, k] of [[0, 17], [1, 3], [2, 7]]) {
    const n1 = cloneMlp(net), n2 = cloneMlp(net), h = 1e-6;
    n1.W[l][k] += h; n2.W[l][k] -= h;
    const fd = (lossOf(n1) - lossOf(n2)) / (2 * h);
    check(close(fd, g.gW[l][k], 1e-6), `finite difference ${fd} vs gradient ${g.gW[l][k]} (layer ${l})`);
  }
  // Adam's first step moves every trained weight by about lr (m/√v = ±1), frozen ones not at all
  const n0 = cloneMlp(net), o2 = new T.LayerAdam(n0, [0, 0.01, 0.01]);
  o2.step(T.featuresAt(n0, o2.from, d.X), d.y);   // inputs of the first trained layer
  check(n0.W[0].every((v, i) => v === net.W[0][i]), "a frozen first layer does not move");
  let moved = 0, total = 0;
  for (const l of [1, 2]) n0.W[l].forEach((v, i) => {
    total++;
    if (close(v - net.W[l][i], -0.01 * Math.sign(g.gW[l][i]), 1e-4) || Math.abs(g.gW[l][i]) < 1e-6) moved++;
  });
  check(moved === total, `first Adam step: every trained weight moves by the rate against its gradient (${moved}/${total})`);
  // the cached features give the same scores as a full forward pass
  const A = T.featuresAt(net, 2, d.X);
  check(d.X.every((x, i) => T.scoresFrom(net, 0, x).every((s, j) => close(s, T.scoresFrom(net, 2, A[i])[j], 1e-12))), "features at layer 2 + the rest = the whole network");
}

current = "strategies";
{
  const s = T.startNet("frozen", 3, 0.01);
  check(s.rates.join() === "0,0,0.01" && s.net.sizes[3] === 3, "frozen: only the new 3-way head trains");
  check(T.startNet("partial", 3, 0.01).rates.join() === "0,0.01,0.01", "partial: layer 2 and head");
  const pre = T.pretrained();
  check(s.net.W[0].every((v, i) => v === pre.W[0][i]), "the backbone starts from the pretrained weights");
  check(T.oldTaskAccuracy(s.net, T.makeSet("source", 400, 202)) === T.evaluate(pre, T.makeSet("source", 400, 202)).acc, "old head put back = the pretrained network");
}

current = "quoted numbers (9 labels, same microscope)";
{
  const pre = T.pretrained(), src = T.makeSet("source", 400, 202);
  check(close(T.evaluate(pre, src).acc, 0.98, 1e-9), `pretrained old-task accuracy ${T.evaluate(pre, src).acc}, quoted 98 %`);
  const pool = T.makeSet("target", 300, 303), val = T.makeSet("target", 300, 404);
  const want: Record<T.Strategy, number> = { frozen: 0.967, partial: 0.95, full: 0.807, scratch: 0.627 };
  for (const st of T.STRATEGIES) {
    const r = new T.Run(st, T.subset(pool, 9), val, 0.003);
    r.run(300, 5);
    check(close(r.last().valAcc, want[st], 0.002), `${st}: ${r.last().valAcc}, quoted ${want[st]}`);
    if (st === "full") check(close(T.oldTaskAccuracy(r.net, src), 0.34, 0.006), `full: old task ${T.oldTaskAccuracy(r.net, src)}, quoted 34 %`);
  }
}

console.log(`transfercurves: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
