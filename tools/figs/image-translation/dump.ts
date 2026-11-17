// Data for the figures of the unit image-translation, computed with the translate1d demo's own code.
// Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/image-translation/dump.ts --bundle --platform=node --outfile=$TMP/t.js && node $TMP/t.js > tools/figs/image-translation/data.json
//   python3 -I tools/figs/image-translation/figs.py
import * as T from "../../../demos/src/translate1d/model";
import { rng } from "../../../demos/src/lib/ui";

const gx = Array.from({ length: 101 }, (_, i) => i / 100);
const out: any = {};
for (const [id, gap, qtr, qte] of [["hidden", 0.3, 0.5, 0.5], ["treated", 0.3, 0.2, 0.8]] as [string, number, number, number][]) {
  const train = T.pairs(T.N_TRAIN, gap, qtr, 1), test = T.pairs(T.N_TEST, gap, qte, 2), r = rng(9);
  const xs = train.map(p => p.x), ys = train.map(p => p.y);
  out[id] = {
    train: train.map(p => [p.x, p.y]), test: test.map(p => [p.x, p.y]),
    mean: gx.map(x => T.condMean(train, x)), median: gx.map(x => T.condMedian(train, x)),
    samples: test.map(p => [p.x, T.condSample(train, p.x, r)]),
    upInc: gx.map(x => T.quantileMap(xs, ys, x)), upDec: gx.map(x => T.quantileMap(xs, ys, x, true)),
    scores: T.scores(train, test, gap),
  };
}
out.gx = gx;
console.log(JSON.stringify(out, (_k, v) => typeof v === "number" ? +v.toPrecision(4) : v));
