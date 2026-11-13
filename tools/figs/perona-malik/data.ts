// Image data for tools/figs/perona-malik/figs.py (run by it; prints JSON). Uses lib/pde.ts.
import { nonlinearStep, heatStep, conductivityMap, testCard, addNoise, psnr255 } from "../../../demos/src/lib/pde";
const r = (a: ArrayLike<number>) => Array.from(a, v => Math.round(v * 100) / 100);
const out: Record<string, unknown> = {};
const truth = testCard().truth, y = addNoise(truth, 12, 4);
out.noisy = r(y); out.truth = r(truth);
out.g = r(conductivityMap(y, 128, 128, "g2", 10, 1));
out.gRaw = r(conductivityMap(y, 128, 128, "g2", 10, 0));
// heat and PM (g2, K = 10, no pre-smoothing) at the same times; PSNR curves
let h: Float64Array = Float64Array.from(y), p: Float64Array = Float64Array.from(y);
const dt = 0.225, ts: number[] = [0], ph: number[] = [psnr255(y, truth)], pp: number[] = [psnr255(y, truth)];
const snaps: Record<string, number[]> = {};
for (let k = 1; k <= 400; k++) {
  h = heatStep(h, 128, 128, dt); p = nonlinearStep(p, 128, 128, dt, "g2", 10);
  if (k <= 20 || k % 4 === 0) { ts.push(k * dt); ph.push(psnr255(h, truth)); pp.push(psnr255(p, truth)); }
  if (k === 8 || k === 60 || k === 400) { snaps["h" + k] = r(h); snaps["p" + k] = r(p); }
}
out.snaps = snaps; out.ts = ts; out.ph = ph; out.pp = pp;
console.log(JSON.stringify(out));
