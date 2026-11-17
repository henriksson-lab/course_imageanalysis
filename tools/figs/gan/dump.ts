// Data for the figures of the units gan and gan-training, computed with the gan1d demo's own code.
// Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/gan/dump.ts --bundle --platform=node --outfile=$TMP/g.js && node $TMP/g.js > tools/figs/gan/data.json
//   python3 -I tools/figs/gan/figs.py && python3 -I tools/figs/gan-training/figs.py
import { Gan1d, precisionRecall1d, dataPdf, type Options, type DataId } from "../../../demos/src/gan1d/model";

const BASE: Options = { lrG: 0.03, lrD: 0.5, kD: 5, batch: 128, loss: "ns" };
const xs = Array.from({ length: 281 }, (_, i) => -3 + 7 * i / 280);
function runPreset(data: DataId, o: Partial<Options>, optimal: boolean, snaps: number[]) {
  const g = new Gan1d(data, { ...BASE, ...o }, 1);
  g.optimal = optimal;
  const out: any = { snaps: [], path: [[1, -1]], recall: [], precision: [] };
  const snap = () => out.snaps.push({ step: g.step, gen: xs.map(x => g.genPdf(x)), disc: xs.map(x => g.disc(x)), a: Math.abs(g.a), b: g.b });
  if (snaps.includes(0)) snap();
  for (let s = 1; s <= 2000; s++) {
    g.train();
    out.path.push([Math.abs(g.a), g.b]);
    if (s % 10 === 0) { const pr = precisionRecall1d(g); out.precision.push(pr.precision); out.recall.push(pr.recall); }
    if (snaps.includes(s)) snap();
  }
  return out;
}
const out: any = { xs, one: xs.map(x => dataPdf("one", x)), two: xs.map(x => dataPdf("two", x)) };
out.converge = runPreset("one", {}, false, [0, 25, 100, 1000]);
out.minimax = runPreset("one", { loss: "mm" }, false, [0, 50, 100]);
out.slow = runPreset("one", { lrD: 0.05, kD: 1 }, false, [100, 500, 2000]);
out.fast = runPreset("one", { lrG: 0.3, kD: 1 }, false, [50]);
out.drop = runPreset("two", {}, true, [2000]);
out.between = runPreset("two", {}, false, [2000]);
console.log(JSON.stringify(out, (_k, v) => typeof v === "number" ? +v.toPrecision(4) : v));
