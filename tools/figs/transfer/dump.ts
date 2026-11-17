// Data for the figures of the units transfer/ and finetune-strategies/, computed with the demos'
// own code (demos/src/transfercurves/toy.ts, demos/src/lowrank/model.ts). Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/transfer/dump.ts --bundle --platform=node --outfile=$TMP/tr.js && node $TMP/tr.js > tools/figs/transfer/data.json
//   python3 -I tools/figs/transfer/figs.py
//   python3 -I tools/figs/finetune-strategies/figs.py
import {
  pretrained, makeSet, Run, subset, STRATEGIES, LOOK_A, LOOK_FAR, oldTaskAccuracy, evaluate, featuresAt, layerOut, LABELS,
  type Look, type Strategy,
} from "../../../demos/src/transfercurves/toy";

import { realChange } from "../../../demos/src/lowrank/model";

const out: any = {};
const t0 = Date.now();
const pre = pretrained();
const srcTest = makeSet("source", 400, 202);
out.sourceAcc = evaluate(pre, srcTest).acc;
const arr = (a: ArrayLike<number>, d = 4) => Array.from(a, v => +v.toFixed(d));

// ---- what the pretrained network learnt: first-layer weights and top patches of layer-2 units
out.filters = Array.from({ length: 32 }, (_, o) => arr(pre.W[0].slice(o * 100, o * 100 + 100)));
const pool = makeSet("source", 1200, 909);
const h2 = featuresAt(pre, 2, pool.X);
out.top = [] as any[];
for (let u = 0; u < 16; u++) {
  const order = pool.X.map((_, i) => i).sort((a, b) => h2[b][u] - h2[a][u]);
  if (h2[order[0]][u] <= 0) continue;
  out.top.push({ unit: u, patches: order.slice(0, 5).map(i => arr(pool.X[i], 3)), cls: order.slice(0, 5).map(i => pool.y[i]) });
}
void layerOut;

// ---- example patches
const ex = (task: "source" | "target", look: Look, n: number) => { const d = makeSet(task, n, 77, look); return { X: d.X.map(x => arr(x, 3)), y: d.y }; };
out.examples = { source: ex("source", LOOK_A, 8), near: ex("target", LOOK_A, 6), far: ex("target", LOOK_FAR, 6) };

// ---- fine-tuning runs
const data: Record<string, { pool: ReturnType<typeof makeSet>; val: ReturnType<typeof makeSet> }> = {
  near: { pool: makeSet("target", 300, 303, LOOK_A), val: makeSet("target", 300, 404, LOOK_A) },
  far: { pool: makeSet("target", 300, 303, LOOK_FAR), val: makeSet("target", 300, 404, LOOK_FAR) },
};
function runOne(dom: string, s: Strategy, n: number, lr: number, rates?: number[]) {
  const r = new Run(s, subset(data[dom].pool, n), data[dom].val, lr, 0, 11, rates);
  r.run(300, 5);
  return r;
}
out.curves = {} as any;
out.byLabels = { labels: LABELS, near: {} as any, far: {} as any };
for (const dom of ["near", "far"]) {
  for (const s of STRATEGIES) out.byLabels[dom][s] = { last: [] as number[], best: [] as number[], old: [] as number[] };
  for (const n of LABELS) for (const s of STRATEGIES) {
    const r = runOne(dom, s, n, 0.003);
    const B = out.byLabels[dom][s];
    B.last.push(r.last().valAcc); B.best.push(r.best().valAcc); B.old.push(s === "scratch" ? NaN : oldTaskAccuracy(r.net, srcTest));
    if (n === 9 || n === 300) out.curves[`${dom}${n}_${s}`] = { step: r.curve.map(c => c.step), acc: arr(r.curve.map(c => c.valAcc)), loss: arr(r.curve.map(c => c.valLoss)), train: arr(r.curve.map(c => c.trainLoss)) };
    if (dom === "near" && n === 300 && s === "full") {
      out.relChange = [0, 1, 2].map(l => {
        let a = 0, b = 0;
        for (let k = 0; k < r.net.W[l].length; k++) { a += (r.net.W[l][k] - (l < 2 ? pre.W[l][k] : 0)) ** 2; b += (l < 2 ? pre.W[l][k] : r.net.W[l][k]) ** 2; }
        return Math.sqrt(a / b);
      });
    }
  }
}
// ---- learning rates for full fine-tuning with 9 labels: one rate, a smaller one, layer-wise
out.rates = {} as any;
for (const [id, lr, rates] of [["0.003", 0.003, undefined], ["0.0005", 0.0005, undefined], ["layerwise", 0.003, [0.0003, 0.001, 0.003]]] as [string, number, number[] | undefined][]) {
  const r = runOne("near", "full", 9, lr, rates);
  out.rates[id] = { step: r.curve.map(c => c.step), acc: arr(r.curve.map(c => c.valAcc)), last: r.last().valAcc, best: r.best().valAcc, bestStep: r.best().step, old: oldTaskAccuracy(r.net, srcTest) };
}
// ---- the real low-rank change
const R = realChange();
out.lowrank = { sv1: arr(R.svds[0].S), sv2: arr(R.svds[1].S), accByRank: R.accByRank, oldByRank: R.oldByRank, accFull: R.accFull };
out.ms = Date.now() - t0;
console.log(JSON.stringify(out));
