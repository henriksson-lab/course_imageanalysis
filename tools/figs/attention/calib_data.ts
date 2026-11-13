// Reliability-diagram data for the attention/ lecture's calibration slide, computed with the
// calibration demo's own model code and seeds (so the numbers match the demo's readouts).
// Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/attention/calib_data.ts --bundle --platform=node --log-level=warning | node > /tmp/calib.json
import * as M from "../../../demos/src/calibration/model";
const NTEST = 4000, NVAL = 2000, BINS = 5;
const fitT = (s: number) => Math.round(M.fitTemperature(M.simulate(NVAL, s, 0, 2)) * 100) / 100;
const one = (s: number, T: number, shift: number) => {
  const test = M.simulate(NTEST, s, shift, 1);
  const R = M.reliability(M.predictions(test, T), BINS);
  const meanConf = M.predictions(test, T).reduce((a, p) => a + p.conf, 0) / NTEST;
  return { s, T, shift, ...R, meanConf };
};
const T = fitT(3);
console.log(JSON.stringify({ over: one(3, 1, 0), fixed: one(3, T, 0), shift: one(3, T, 0.5) }));
