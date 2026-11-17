// The recorded runs of the augmentlab demo's "train" view (and the data of the figures of the
// unit "Regularisation and data augmentation"), computed with the demo's own model
// (demos/src/augmentlab/model.ts): the crop experiment with several augmentation settings,
// training and validation loss and validation accuracy after every epoch. Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/augmentation/dump.ts --bundle --platform=node --outfile=$TMP/au.js && node $TMP/au.js > demos/src/augmentlab/curves.json
//   node $TMP/au.js crops > tools/figs/augmentation/crops.json
//   (cd code && cargo run -p training --features solution -- augmentation)   # the micrograph pictures
//   python3 -I tools/figs/augmentation/figs.py
import { experiment, CropRun, NO_AUG, type Aug, type Task } from "../../../demos/src/augmentlab/model";

export const SETTINGS: { id: string; label: string; task: Task; aug: Aug; decay: number }[] = [
  { id: "none", label: "no augmentation", task: "shape", aug: NO_AUG, decay: 0 },
  { id: "decay", label: "weight decay 0.01 only", task: "shape", aug: NO_AUG, decay: 0.01 },
  { id: "geom", label: "flips + rotations", task: "shape", aug: { ...NO_AUG, flips: true, maxAngle: 180 }, decay: 0 },
  { id: "all", label: "flips + rotations + gain + noise", task: "shape", aug: { flips: true, maxAngle: 180, gainRange: 0.4, noise: 0.03, elastic: 0 }, decay: 0 },
  { id: "elastic", label: "flips + rotations + elastic", task: "shape", aug: { flips: true, maxAngle: 180, gainRange: 0, noise: 0, elastic: 1 }, decay: 0 },
  { id: "intensity", label: "gain + noise only", task: "shape", aug: { ...NO_AUG, gainRange: 0.4, noise: 0.03 }, decay: 0 },
  { id: "o-none", label: "lying vs standing: no augmentation", task: "orientation", aug: NO_AUG, decay: 0 },
  { id: "o-flips", label: "lying vs standing: flips", task: "orientation", aug: { ...NO_AUG, flips: true }, decay: 0 },
  { id: "o-rot", label: "lying vs standing: rotations", task: "orientation", aug: { ...NO_AUG, maxAngle: 180 }, decay: 0 },
];
declare const process: { argv: string[] };
const r4 = (v: number) => Math.round(v * 1e4) / 1e4;
if (process.argv[2] === "crops") {
  const a = experiment("shape"), b = experiment("orientation"), px = (g: { data: Float32Array }) => Array.from(g.data, r4);
  console.log(JSON.stringify({ train: a.tr.imgs.slice(0, 10).map(px), val: a.va.imgs.slice(0, 10).map(px), orient: b.tr.imgs.slice(0, 10).map(px) }));
} else {
const out = SETTINGS.map(s => {
  const { tr, va } = experiment(s.task);
  const run = new CropRun(tr, va, s.aug, s.decay);
  for (let e = 0; e < 300; e++) run.trainEpoch();
  const h = run.history;
  return { ...s, train: h.map(p => r4(p.train)), val: h.map(p => r4(p.val)), trainAcc: h.map(p => r4(p.trainAcc)), valAcc: h.map(p => r4(p.valAcc)) };
});
console.log(JSON.stringify(out));
}
