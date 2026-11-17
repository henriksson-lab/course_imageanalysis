// Data for the figures of the units few-labels/ and domain-shift/, computed with the demos' own code
// (demos/src/activelearn/model.ts, demos/src/domainshift/model.ts, demos/src/transfercurves/toy.ts).
// Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/few-labels/dump.ts --bundle --platform=node --outfile=$TMP/fl.js && node $TMP/fl.js > tools/figs/few-labels/data.json
//   python3 -I tools/figs/few-labels/figs.py
//   python3 -I tools/figs/domain-shift/figs.py
import { makePool, learningCurve, Labeller, startSet, score, prob, RULES, type Rule } from "../../../demos/src/activelearn/model";
import { trainA, evaluateB, cellsA, toB, REMEDIES } from "../../../demos/src/domainshift/model";
import { makeSet, LOOK_A, LOOK_FAR, type Look } from "../../../demos/src/transfercurves/toy";

const out: any = {};
const r3 = (v: number) => +v.toFixed(4);
// ---- active learning (the demo's settings: pool 500 seed 1, test 600 seed 2, 6 starts, up to 40 labels)
const pool = makePool(500, 1), test = makePool(600, 2);
out.pool = pool.map(p => [r3(p.x), r3(p.y), p.c, p.rare ? 1 : 0]);
out.curves = {} as any;
for (const rule of RULES) out.curves[rule] = learningCurve(pool, test, rule, 40, [1, 2, 3, 4, 5, 6]);
out.snap = {} as any;
const G = 60;
for (const [rule, n] of [["uncertain", 40], ["both", 40], ["random", 40], ["mistakes", 20], ["uncertain", 4]] as [Rule, number][]) {
  const L = new Labeller(pool, startSet(pool, 1), 51);
  while (L.labelled.length < n) L.step(rule);
  const grid: number[] = [];
  for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) grid.push(r3(prob(L.w, (i + 0.5) / G, (j + 0.5) / G)));
  const next = L.propose(rule);
  out.snap[`${rule}${n}`] = { labelled: L.labelled.slice(), grid, next, score: score(L.w, test), nRare: L.labelled.filter(j => pool[j].rare).length };
}
const all = new Labeller(pool, pool.map((_, i) => i));
out.allLabels = score(all.w, test);
// ---- domain shift
const T = trainA();
out.shift = { s: [] as number[], f50: {} as any, f20: {} as any };
for (const r of REMEDIES) { out.shift.f50[r] = []; out.shift.f20[r] = []; }
out.shift.score50 = []; out.shift.score20 = [];
for (let k = 0; k <= 20; k++) {
  const s = k / 20; out.shift.s.push(s);
  for (const [frac, key] of [[0.5, "50"], [0.2, "20"]] as [number, string][]) {
    const e = evaluateB(T, s, frac);
    for (const r of REMEDIES) out.shift["f" + key][r].push(r3(e.acc[r]));
    out.shift["score" + key].push(r3(e.score));
  }
}
out.accA = { none: score2(T.wNone, T.A), augOnA: score2(T.wAug, T.A) };
function score2(w: [number, number, number], cells: { f: [number, number]; c: number }[]) {
  let ok = 0; for (const c of cells) if (((1 / (1 + Math.exp(-(w[0] + w[1] * c.f[0] + w[2] * c.f[1])))) > 0.5 ? 1 : 0) === c.c) ok++;
  return ok / cells.length;
}
out.cellsA = T.A.map(c => [r3(c.f[0]), r3(c.f[1]), c.c]);
out.cellsB = toB(cellsA(400, 2, 0.5), 0.8).map(c => [r3(c.f[0]), r3(c.f[1]), c.c]);
out.wNone = T.wNone; out.wAug = T.wAug;
// ---- one object through several microscopes (the toy patches)
const looks: [string, Look][] = [
  ["training microscope", LOOK_A],
  ["dimmer", { ...LOOK_A, gain: 0.5 }],
  ["brighter background", { ...LOOK_A, offset: 0.25 }],
  ["blurred", { ...LOOK_A, blur: 1.0 }],
  ["noisier", { ...LOOK_A, noise: 0.15 }],
  ["other modality", LOOK_FAR],
];
out.looks = looks.map(([name, look]) => ({ name, X: makeSet("target", 3, 77, look).X.map(x => Array.from(x, r3)) }));
console.log(JSON.stringify(out));
