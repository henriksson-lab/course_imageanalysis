// Data for the optimisers figures and the numbers on the unit's pages, computed with the demo's
// own code (demos/src/optrace/model.ts). Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/optimisers/dump.ts --bundle --platform=node --outfile=$TMP/op.js && node $TMP/op.js > tools/figs/optimisers/data.json
//   python3 -I tools/figs/optimisers/figs.py
import { Walker, surface, stepsToConverge, lrAt, DEFAULT_RACE, OPTS, type Schedule } from "../../../demos/src/optrace/model";

const out: any = {};
const grid = (id: string, n = 120) => {
  const S = surface(id), z: number[] = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) z.push(S.f(-2 + 4 * i / (n - 1), -2 + 4 * j / (n - 1)));
  return { n, z, min: S.min, minima: S.minima };
};
const paths = (id: string, etas: number[], o: any = {}) => OPTS.map((opt, k) => {
  const w = new Walker(surface(id), opt, { ...DEFAULT_RACE, eta: etas[k], ...o }).run(o.steps ?? 200);
  return { opt, eta: etas[k], path: w.path, losses: w.losses, diverged: w.diverged, stepsTo: w.stepsTo() };
});
out.valley = { ...grid("valley"), runs: paths("valley", [0.19, 0.03, 0.1]) };
out.lr3 = { ...grid("valley"), runs: [0.02, 0.19, 0.25].map(e => { const w = new Walker(surface("valley"), "gd", { ...DEFAULT_RACE, eta: e }).run(60); return { eta: e, path: w.path, diverged: w.diverged }; }) };
out.saddle = { ...grid("saddle"), runs: paths("saddle", [0.03, 0.03, 0.03]) };
out.curved = { ...grid("curved"), runs: paths("curved", [0.03, 0.01, 0.3]) };
const etas = Array.from({ length: 31 }, (_, i) => 10 ** (-3 + 3 * i / 30));
out.sweep = { etas, steps: Object.fromEntries(OPTS.map(o => [o, etas.map(e => stepsToConverge("valley", o, e))])) };
out.table = Object.fromEntries(["valley", "saddle", "curved"].map(s => [s, Object.fromEntries(OPTS.map(o => [o, [0.003, 0.01, 0.03, 0.1, 0.19, 0.3].map(e => stepsToConverge(s as any, o, e))]))]));
out.noisy = Object.fromEntries((["constant", "warmup"] as Schedule[]).map(sc => [sc, OPTS.map((o, k) => {
  const runs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20].map(seed => new Walker(surface("bowl"), o, { ...DEFAULT_RACE, eta: [0.1, 0.01, 0.1][k], noise: 1, schedule: sc, seed }).run(200));
  const tail = runs.map(w => w.losses.slice(-50).reduce((a, b) => a + b, 0) / 50);
  return { opt: o, path: runs[0].path, losses: runs[0].losses, meanTail: tail.reduce((a, b) => a + b, 0) / tail.length };
})]));
out.noisyGrid = grid("bowl");
out.schedules = Object.fromEntries((["constant", "step", "cosine", "warmup"] as Schedule[]).map(s => [s, Array.from({ length: 200 }, (_, t) => lrAt(s, 1, t, 200))]));
const a1 = new Walker(surface("valley"), "adam", { ...DEFAULT_RACE, eta: 0.1 }).run(200), a100 = new Walker(surface("valley"), "adam", { ...DEFAULT_RACE, eta: 0.1, scale: 100 }).run(200);
out.scale = { adam1: a1.stepsTo(), adam100: a100.stepsTo(), gd100: new Walker(surface("valley"), "gd", { ...DEFAULT_RACE, eta: 0.19, scale: 100 }).run(200).diverged, mom100: new Walker(surface("valley"), "momentum", { ...DEFAULT_RACE, eta: 0.03, scale: 100 }).run(200).diverged };
out.nobias = new Walker(surface("valley"), "adam", { ...DEFAULT_RACE, eta: 0.1, biasCorrection: false }).run(200).stepsTo();
out.nobiasFirst = new Walker(surface("valley"), "adam", { ...DEFAULT_RACE, eta: 0.1, biasCorrection: false }).run(3).path;
console.log(JSON.stringify(out));
