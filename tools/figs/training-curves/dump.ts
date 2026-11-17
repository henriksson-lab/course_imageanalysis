// The recorded runs of the curve-doctor demo (and the data of the figures of the unit
// "Reading training curves"), computed with the demo's own model (demos/src/curvedoctor/model.ts):
// every case and its fix, the training and validation loss after every epoch, the final
// accuracies and the recall of each class. Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/training-curves/dump.ts --bundle --platform=node --outfile=$TMP/tc.js && node $TMP/tc.js > demos/src/curvedoctor/curves.json
//   python3 -I tools/figs/training-curves/figs.py
import { CASES, CaseRun, type CaseConfig } from "../../../demos/src/curvedoctor/model";

const r4 = (v: number) => Math.round(v * 1e4) / 1e4;
function record(cfg: Partial<CaseConfig>) {
  const run = new CaseRun(cfg);
  while (!run.done()) run.trainEpoch();
  const h = run.history, last = h[h.length - 1];
  return {
    train: h.map(p => r4(p.train)), val: h.map(p => r4(p.val)),
    trainAcc: r4(last.trainAcc), valAcc: r4(last.valAcc), recalls: run.recalls().map(r4),
    nTrain: run.X.length,
  };
}
const out = CASES.map(c => ({
  id: c.id, diagnosis: c.diagnosis, fixLabel: c.fixLabel, explain: c.explain,
  run: record(c.cfg), fix: c.diagnosis === "healthy" ? null : record(c.fix),
}));
console.log(JSON.stringify(out));
