// Data for the unet figures, computed with the unetseg demo's own code (same images, same network).
// Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/unet/dump.ts --bundle --platform=node --outfile=$TMP/un.js && node $TMP/un.js > tools/figs/unet/data.json
//   python3 -I tools/figs/unet/figs.py
import { sample, NO_SHIFT, type Shift } from "../../../demos/src/unetseg/data";
import { runUnet, type WeightSet } from "../../../demos/src/unetseg/net";
import { maskAt, otsuMask, scores } from "../../../demos/src/unetseg/model";

function one(k: number, set: WeightSet, shift: Partial<Shift>) {
  const s = sample(200001 + k, { ...NO_SHIFT, ...shift });
  const run = runUnet(s.image, set);
  const m = maskAt(run.prob, 0.5), sc = scores(m, s.mask), o = otsuMask(s.image), so = scores(o.mask, s.mask);
  return { image: Array.from(s.image), mask: Array.from(s.mask), labels: Array.from(s.labels), prob: Array.from(run.prob),
    pred: Array.from(m), otsu: Array.from(o.mask), otsuT: o.t, dice: sc.dice, iou: sc.iou, otsuDice: so.dice, count: s.params.count };
}
const out = {
  typical: one(0, "unet", {}),
  otsu: one(0, "unet", { gradient: 100 }),
  noskip: one(2, "noskip", { size: 0.5 }),
  noskipU: one(2, "unet", { size: 0.5 }),
  touching: one(4, "unet", {}),
  dimTrain: one(5, "unet", {}),
  dim: one(5, "unet", { brightness: 0.5 }),
};
console.log(JSON.stringify(out));
