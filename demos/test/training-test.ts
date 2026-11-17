// Tests for the day "Training networks in practice" (optimisers/, normalisation/, augmentation/,
// training-curves/): the optimiser race (optrace), the deep stack with normalisation and residual
// connections (deepstack, gradients against finite differences), the augmentations and the crop
// experiment (augmentlab), the training cases (curvedoctor), and the numbers quoted on the pages.
// Run alone:
//   node_modules/.bin/esbuild demos/test/training-test.ts --bundle --platform=node --log-level=warning | node
import { Walker, surface, stepsToConverge, lrAt, DEFAULT_RACE } from "../src/optrace/model";
import { runStack, normalise, normaliseBack, bend, slope, DEFAULT_STACK, type Norm } from "../src/deepstack/model";
import { flipH, flipV, rotateCw, elastic, intensity, cropSet, augmentCrop, NO_AUG } from "../src/augmentlab/model";
import { CaseRun, spiral3, LN3, CASES } from "../src/curvedoctor/model";
import { createGrey, fromArray } from "../src/lib/imageops";
import { rng } from "../src/lib/ui";
import augCurves from "../src/augmentlab/curves.json";
import docCurves from "../src/curvedoctor/curves.json";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 25) console.log("FAIL: " + msg); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

// ---- 1. optrace: update rules and the numbers of the optimisers exercise ----
{
  // Adam's first step is η in every weight, whatever the size of the gradient (with bias correction)
  for (const scale of [1e-3, 1, 1e3]) {
    const w = new Walker(surface("bowl"), "adam", { ...DEFAULT_RACE, eta: 0.1, scale }, [1, -1]).run(1);
    check(close(w.a, 0.9, 1e-6) && close(w.b, -0.9, 1e-6), `Adam first step at scale ${scale}: (${w.a}, ${w.b})`);
  }
  // without bias correction the first step is 0.1/sqrt(0.001) = 3.16 η
  const nb = new Walker(surface("bowl"), "adam", { ...DEFAULT_RACE, eta: 0.1, biasCorrection: false }, [1, -1]).run(1);
  check(close(1 - nb.a, 0.31623, 1e-4), `Adam without bias correction: step ${1 - nb.a}`);
  // momentum with a constant gradient: on the "bowl" the gradient is not constant, so use the rule directly
  let v = 0, x = 0; const vs: number[] = [];
  for (let i = 0; i < 3; i++) { v = 0.9 * v + 1; x -= 0.1 * v; vs.push(v); }
  check(close(vs[2], 2.71, 1e-12) && close(x, -0.561, 1e-12), "momentum velocities 1, 1.9, 2.71");
  // plain descent on the valley: stable below η = 2/10 = 0.2, diverges above
  check(stepsToConverge("valley", "gd", 0.19) === 75, `GD 0.19 on the valley: ${stepsToConverge("valley", "gd", 0.19)} steps (Rust: 75)`);
  check(stepsToConverge("valley", "gd", 0.21) === -1, "GD 0.21 on the valley diverges");
  check(stepsToConverge("valley", "momentum", 0.03) === 78, `momentum 0.03: ${stepsToConverge("valley", "momentum", 0.03)} (Rust: 78)`);
  check(stepsToConverge("valley", "adam", 0.1) === 69, `Adam 0.1: ${stepsToConverge("valley", "adam", 0.1)} (Rust: 69)`);
  // the loss × 100: GD diverges, Adam's path is the same
  const a1 = new Walker(surface("valley"), "adam", { ...DEFAULT_RACE, eta: 0.1 }).run(150);
  const a100 = new Walker(surface("valley"), "adam", { ...DEFAULT_RACE, eta: 0.1, scale: 100 }).run(150);
  check(Math.abs(a1.a - a100.a) < 1e-6 && Math.abs(a1.b - a100.b) < 1e-6, `Adam scale-free: (${a1.a}, ${a1.b}) vs (${a100.a}, ${a100.b})`);
  check(new Walker(surface("valley"), "gd", { ...DEFAULT_RACE, eta: 0.19, scale: 100 }).run(200).diverged, "GD with the loss × 100 diverges");
  // schedules
  check(close(lrAt("warmup", 0.1, 0, 200), 0.005) && close(lrAt("warmup", 0.1, 19, 200), 0.1) && close(lrAt("warmup", 0.1, 110, 200), 0.05), "warm-up + cosine");
  check(close(lrAt("step", 1, 99, 200), 1) && close(lrAt("step", 1, 100, 200), 0.1) && close(lrAt("step", 1, 150, 200), 0.01), "step decay");
  check(close(lrAt("cosine", 1, 100, 200), 0.5), "cosine half-way");
  // gradients of the surfaces against finite differences
  for (const id of ["valley", "saddle", "curved", "bowl"]) {
    const S = surface(id), [a, b] = [0.37, -0.81], h = 1e-6, g = S.grad(a, b);
    check(close(g[0], (S.f(a + h, b) - S.f(a - h, b)) / (2 * h), 1e-5) && close(g[1], (S.f(a, b + h) - S.f(a, b - h)) / (2 * h), 1e-5), `gradient of ${id}`);
  }
}

// ---- 2. deepstack: normalisation, its backward pass, the residual stack ----
{
  // batch norm and layer norm of the by-hand example
  const z = Float64Array.from([1, 10, 3, 10, 5, 40]);   // 3 examples × 2 units
  normalise(z, 3, 2, "batch", 0);
  const want = [-1.2247449, -0.7071068, 0, -0.7071068, 1.2247449, 1.4142136];
  check(want.every((w, i) => close(z[i], w, 1e-6)), `batch norm by hand: ${Array.from(z)}`);
  const l = Float64Array.from([1, 3, 5]);
  normalise(l, 1, 3, "layer", 0);
  check(close(l[0], -1.2247449, 1e-6) && close(l[1], 0) && close(l[2], 1.2247449, 1e-6), "layer norm by hand");
  // backward through the normalisation against finite differences of L = Σ c · normalise(z)
  const r = rng(3);
  for (const norm of ["batch", "layer"] as Norm[]) {
    const B = 4, W = 3, z0 = Float64Array.from({ length: B * W }, () => r.norm()), c = Float64Array.from({ length: B * W }, () => r.norm());
    const L = (zz: Float64Array) => { const y = zz.slice(); normalise(y, B, W, norm); let s = 0; for (let i = 0; i < y.length; i++) s += c[i] * y[i]; return s; };
    const y = z0.slice(), inv = normalise(y, B, W, norm), d = c.slice();
    normaliseBack(d, y, inv, B, W, norm);
    let ok = true;
    for (let i = 0; i < B * W; i++) {
      const zp = z0.slice(), zm = z0.slice(); zp[i] += 1e-6; zm[i] -= 1e-6;
      if (!close(d[i], (L(zp) - L(zm)) / 2e-6, 1e-4)) ok = false;
    }
    check(ok, `${norm} norm backward against finite differences`);
  }
  // the slope used on the way back
  check(close(slope("tanh", 0.3), (bend("tanh", 0.3 + 1e-6) - bend("tanh", 0.3 - 1e-6)) / 2e-6, 1e-6), "tanh slope");
  // He initialisation keeps a ReLU signal; gain 1 halves the mean square per layer
  const he = runStack({ ...DEFAULT_STACK, depth: 20 }), one = runStack({ ...DEFAULT_STACK, depth: 20, gain: 1 });
  check(he.layers[19].rms > 0.4 && he.layers[19].rms < 2.5, `He: size after 20 layers ${he.layers[19].rms}`);
  check(one.layers[19].rms < 0.01, `gain 1: size after 20 layers ${one.layers[19].rms}`);
  const bn = runStack({ ...DEFAULT_STACK, depth: 20, gain: 2, norm: "batch" });
  check(bn.layers.every(q => Math.abs(q.rms - Math.SQRT1_2) < 0.03), "batch norm: every ReLU layer at sqrt(1/2)");
  const res = runStack({ ...DEFAULT_STACK, depth: 30, gain: 1, residual: true }), plain = runStack({ ...DEFAULT_STACK, depth: 30, gain: 1 });
  check(res.layers[0].gradRms / res.layers[29].gradRms > 0.5 && plain.layers[0].gradRms / plain.layers[29].gradRms < 1e-3,
    `residual keeps the gradient: ratio first/last ${res.layers[0].gradRms / res.layers[29].gradRms} vs plain ${plain.layers[0].gradRms / plain.layers[29].gradRms}`);
}

// ---- 3. augmentlab: augmentations and the crop experiment ----
{
  const img = fromArray(3, 3, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
  const f = flipH(img);
  check(close(f.data[0], 0.3, 1e-7) && Array.from(flipH(f).data).every((v, i) => close(v, img.data[i], 1e-7)), "flipH");
  check(close(flipV(img).data[0], 0.7, 1e-7), "flipV");
  const r90 = rotateCw(img, 90);
  check(close(r90.data[2], 0.1, 1e-5) && close(r90.data[0], 0.7, 1e-5), `rotate 90° clockwise: ${Array.from(r90.data)}`);
  const flat = createGrey(20, 20, 0.4);
  check(elastic(flat, 3, 4, 1).data.every(v => close(v, 0.4, 1e-5)), "elastic keeps a flat image");
  check(close(intensity(createGrey(1, 1, 0.25), 2, 0.1, 0.5).data[0], 1.1, 1e-6), "intensity 2 · 0.25^0.5 + 0.1");
  const tr = cropSet("shape", 20, true, 11);
  check(tr.y.filter(c => c === 1).length === 10, "crop set balanced");
  const same = augmentCrop(tr.imgs[0], NO_AUG, rng(1));
  check(same === tr.imgs[0], "no augmentation returns the crop");
  // the recorded curves: augmentation helps on the shape task, rotations hurt on the orientation task
  const A = augCurves as any[], get = (id: string) => A.find(a => a.id === id);
  const last = (id: string) => get(id).valAcc[300];
  check(last("none") < 0.75 && last("geom") > 0.85, `shape task: no augmentation ${last("none")}, flips + rotations ${last("geom")}`);
  check(last("o-none") > 0.98 && last("o-rot") < 0.7, `orientation task: none ${last("o-none")}, rotations ${last("o-rot")}`);
}

// ---- 4. curvedoctor: the cases, and the recording matches the model ----
{
  const s = spiral3([10, 20, 30], 0.1, 4);
  check(s.c.filter(k => k === 0).length === 10 && s.c.filter(k => k === 2).length === 30, "spiral counts");
  const dead = new CaseRun({ zeroInputs: true, epochs: 20 });
  while (!dead.done()) dead.trainEpoch();
  check(dead.history.every(p => Math.abs(p.train - LN3) < 2e-3), `zero inputs: loss stays at ln 3 (${dead.history[20].train})`);
  // replay the first epochs of the healthy case and compare with curves.json
  const healthy = CASES.find(c => c.diagnosis === "healthy")!;
  const run = new CaseRun(healthy.cfg);
  for (let e = 0; e < 5; e++) run.trainEpoch();
  const rec = (docCurves as any[]).find(c => c.id === healthy.id).run;
  check(run.history.every((p, e) => Math.abs(p.train - rec.train[e]) < 1e-4 && Math.abs(p.val - rec.val[e]) < 1e-4), "curves.json matches the model (healthy, 5 epochs)");
  const imb = (docCurves as any[]).find(c => c.diagnosis === "imbalance");
  check(imb.fix.recalls[2] > imb.run.recalls[2], `class weights raise the rare class's recall: ${imb.run.recalls[2]} -> ${imb.fix.recalls[2]}`);
}

console.log(`${checks - failures} of ${checks} checks passed`);
if (failures) { console.log(`${failures} FAILED`); process.exit(1); }
