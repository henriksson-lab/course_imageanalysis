// Tests for the contrastive demo (unit contrastive/): the InfoNCE loss on the by-hand numbers
// of the exercise, its gradient against finite differences, the collapse without negatives,
// and the preset results quoted on the pages (600 steps).
// Run alone (from the repository root):
//   node_modules/.bin/esbuild demos/test/contrastive-test.ts --bundle --platform=node --log-level=warning | node
import { infoNce, alignOnly, makeCells, scores, Trainer, augment, type Settings } from "../src/contrastive/model";
import { rng } from "../src/lib/ui";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 25) console.log("FAIL: " + msg); }
}
const near = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;
const unit = (t: number): [number, number] => [Math.cos(t), Math.sin(t)];

// ---- B2 by hand: one direction of InfoNCE = cross-entropy of a softmax over the batch ----
{
  // cosines of view a1 with b1, b2, b3: 0.9, 0.1, −0.5 (angles chosen to give them)
  const one = (tau: number) => { const s = [0.9, 0.1, -0.5].map(c => c / tau), m = Math.max(...s); return -(s[0] - m - Math.log(s.reduce((q, v) => q + Math.exp(v - m), 0))); };
  check(near(one(0.5), 0.233, 0.001), `B2a: loss ${one(0.5).toFixed(3)} at τ = 0.5`);
  check(near(one(0.1), 0.00034, 0.00001), `B2b: loss ${one(0.1).toFixed(5)} at τ = 0.1`);
  // collapsed batch: all embeddings equal, both directions: ln N
  const n = 5, same = Array.from({ length: n }, () => unit(0.3));
  check(near(infoNce(same, same, 0.2).loss, Math.log(n), 1e-12), "collapse costs ln N with negatives");
  check(near(alignOnly(same, same).loss, -1, 1e-12), "collapse is optimal without negatives");
}

// ---- gradients against finite differences ----
{
  const r = rng(3), n = 4;
  const a = Array.from({ length: n }, () => unit(6 * r.unif())), b = Array.from({ length: n }, () => unit(6 * r.unif()));
  for (const tau of [0.1, 0.5]) {
    const g = infoNce(a, b, tau);
    let worst = 0;
    for (const [arr, grad] of [[a, g.ga], [b, g.gb]] as const) for (let i = 0; i < n; i++) for (let k = 0; k < 2; k++) {
      const h = 1e-6, keep = arr[i][k];
      arr[i][k] = keep + h; const lp = infoNce(a, b, tau).loss;
      arr[i][k] = keep - h; const lm = infoNce(a, b, tau).loss;
      arr[i][k] = keep;
      worst = Math.max(worst, Math.abs((lp - lm) / (2 * h) - grad[i][k]));
    }
    check(worst < 1e-6, `InfoNCE gradient vs finite differences at τ = ${tau}: ${worst.toExponential(2)}`);
  }
}

// ---- augmentation: strength 0 is the identity ----
{
  const c = makeCells(1)[5];
  const v = augment(c.img, { strength: 0, geom: true, bright: true, noise: true }, rng(1));
  check(v.every((x, i) => near(x, c.img[i], 1e-12)), "strength 0 leaves the image unchanged");
}

// ---- the presets after 600 steps (numbers on the pages) ----
{
  const cells = makeCells(1);
  const BASE: Settings = { tau: 0.2, aug: { strength: 1, geom: true, bright: true, noise: true }, negatives: true };
  const run = (s: Settings) => { const t = new Trainer(cells); for (let i = 0; i < 600; i++) t.train(s); return { ...scores(t.net, cells), loss: t.losses[599], first: t.losses[0] }; };
  const all = run(BASE);
  check(all.shape === 1 && near(all.bright, 0.64, 0.005) && near(all.collapse, 0.52, 0.005), `all: shape ${all.shape}, bright ${all.bright.toFixed(2)}, mean ${all.collapse.toFixed(2)}`);
  const nb = run({ ...BASE, aug: { ...BASE.aug, bright: false } });
  check(near(nb.bright, 0.95, 0.005) && near(nb.shape, 0.83, 0.005), `no brightness aug: shape ${nb.shape.toFixed(2)}, bright ${nb.bright.toFixed(2)}`);
  const ng = run({ ...BASE, negatives: false });
  check(ng.collapse > 0.999 && near(ng.loss, -1, 1e-3), `no negatives: collapsed (mean ${ng.collapse.toFixed(4)}, loss ${ng.loss.toFixed(4)})`);
  const none = run({ ...BASE, aug: { ...BASE.aug, strength: 0 } });
  check(none.collapse < 0.02 && near(none.loss, 2.464, 0.001), `no augmentation: spread out (mean ${none.collapse.toFixed(3)}), loss ${none.loss.toFixed(3)}`);
}

console.log(`contrastive: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
