// Data for the backprop figures, computed with the demos' own code (nnplayground, gdescent).
// Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/backprop/dump.ts --bundle --platform=node --outfile=$TMP/bp.js && node $TMP/bp.js > tools/figs/backprop/data.json
//   python3 -I tools/figs/backprop/figs.py
import { dataset } from "../../../demos/src/lib/datasets2d";
import { Trainer, DEFAULT, isVal, type Config } from "../../../demos/src/nnplayground/model";
import { LANDSCAPES, lossGrad, newPath, stepPath, findMinima, NEX } from "../../../demos/src/gdescent/model";

const out: any = {};
// ---- playground presets "overfit", "decay", "dropout": moons, 60 points, noise 0.1, 8 + 8 units, 2000 steps
const pts = dataset("moons", { n: 60, noise: 0.1, seed: 1 }).points;
out.points = pts.map((p, i) => [p.x, p.y, p.c, isVal(i, 0.5) ? 1 : 0]);
const G = 100;
for (const [id, extra] of [["overfit", {}], ["decay", { decay: 0.003 }], ["dropout", { dropout: 0.4 }]] as [string, Partial<Config>][]) {
  const cfg: Config = { ...DEFAULT, hidden: [8, 8], lr: 0.1, valFrac: 0.5, ...extra };
  const t = new Trainer(pts, 2, cfg);
  t.train(2000);
  const h = t.history;
  let best = h[0]; for (const q of h) if (q.val < best.val) best = q;
  const grid: number[] = [];
  for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) grid.push(t.probs(-0.1 + 1.2 * (i + 0.5) / G, -0.1 + 1.2 * (j + 0.5) / G)[1]);
  const last = h[h.length - 1];
  out[id] = { grid, train: last.train, val: last.val, bestStep: best.step, bestVal: best.val, trainAcc: last.trainAcc, valAcc: last.valAcc };
}
// ---- gdescent: bowl and valley landscapes with paths
function land(id: string) { return LANDSCAPES.find(l => l.id === id)!; }
function contour(id: string) {
  const L = land(id), [a0, a1, b0, b1] = L.range, n = 120, z: number[] = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) z.push(lossGrad(L, a0 + (a1 - a0) * i / (n - 1), b0 + (b1 - b0) * j / (n - 1))[0]);
  return { range: L.range, n, z, minima: findMinima(L) };
}
function path(id: string, start: [number, number], eta: number, batch: number, beta: number, steps: number, seed = 100) {
  const L = land(id), p = newPath(L, start[0], start[1], seed);
  const m = findMinima(L)[0];
  let reached = -1;
  for (let s = 0; s < steps; s++) {
    stepPath(L, p, eta, batch, beta);
    if (reached < 0 && Math.hypot(p.a - m[0], p.b - m[1]) < 1e-3) reached = s + 1;
  }
  return { pts: p.pts, losses: p.losses, reached };
}
out.bowl = contour("bowl"); out.valley = contour("valley");
out.paths = {
  small: path("bowl", [-1.5, 2], 0.02, NEX, 0, 300),
  right: path("bowl", [-1.5, 2], 0.3, NEX, 0, 60),
  zigzag: path("valley", [-1.5, 2], 0.6, NEX, 0, 40),
  full01: path("bowl", [-1.5, 2], 0.1, NEX, 0, 60),
  sgd4: path("bowl", [-1.5, 2], 0.1, 4, 0, 60, 100),
};
console.log(JSON.stringify(out));
