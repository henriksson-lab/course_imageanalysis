// Numbers and images for the Radon day's slide figures, computed with the demos' own
// library (demos/src/lib/radon.ts) so that every value matches the demos.
// Build and run from the repository root (see figs.py, which calls this):
//   node_modules/.bin/esbuild tools/figs/radon/compute.ts --bundle --platform=node --outfile=SCRATCH/compute.js
//   node SCRATCH/compute.js OUTDIR
// Writes OUTDIR/<name>.f64 (raw little-endian float64, n x n row-major) and OUTDIR/data.json.
import * as R from "../../../demos/src/lib/radon";
import * as fs from "fs";
import * as path from "path";

const OUT = process.argv[2];
fs.mkdirSync(OUT, { recursive: true });
const N = 128;
const data: Record<string, unknown> = {};
const save = (name: string, a: Float64Array) => fs.writeFileSync(path.join(OUT, name + ".f64"), Buffer.from(a.buffer));
const mask = R.fovMask(N);

// Objects as in demos/src/radonproj/view.ts
const shepp = R.ellipseImage(N, R.sheppLoganEllipses(N), 3);
const disc = R.ellipseImage(N, [R.disc(10, 6, 30, 1)], 3);
save("shepp", shepp); save("disc", disc);

// tomosampling (bins = N, so k = 1: filter + back-project): FBP (Hann) at 18 / 200 angles, wedge
function tomo(truth: Float64Array, ang: Float64Array): Float64Array {
  const g = R.geometry(N, ang);
  const s = R.project(truth, g);
  const filt = R.filterSinogram(s, ang.length, N, { kind: "hann" }, 1);
  return R.backproject(filt, g, R.angleStep(ang));
}
const r18 = tomo(shepp, R.angleSet(18)), r200 = tomo(shepp, R.angleSet(200));
save("fbp18", r18); save("fbp200", r200);
data.rmse18 = R.rmseOf(r18, shepp, mask); data.rmse200 = R.rmseOf(r200, shepp, mask);
const wedgeAng = R.angleRange(-60, 60, 61);
const lemon = tomo(disc, wedgeAng), wedgeShepp = tomo(shepp, wedgeAng);
save("lemon", lemon); save("wedgeshepp", wedgeShepp);
data.rmseLemon = R.rmseOf(lemon, disc, mask);
data.rmseWedge = R.rmseOf(wedgeShepp, shepp, mask);
data.wedgeAngles = Array.from(wedgeAng);

// a point at the centre: plain back-projection and ramp FBP with 16 angles (backproj demo)
const pt = new Float64Array(N * N); R.addPoint(pt, N, 0, 0, 1);
for (const na of [16, 180]) {
  const g = R.geometry(N, R.angleSet(na)), s = R.project(pt, g);
  save(`pt_plain_${na}`, R.fbp(s, g, { kind: "none" }));
  save(`pt_ramp_${na}`, R.fbp(s, g, { kind: "ramp" }));
}
// filtered projection of the point (ramp, padded) for the 1D plot
{
  const g = R.geometry(N, R.angleSet(1)), s = R.project(pt, g);
  data.ptProj = Array.from(s);
  data.ptProjRamp = Array.from(R.filterSinogram(s, 1, N, { kind: "ramp" }));
}

// iterrecon "noisy: stop early": ART omega 0.5, 60 angles, 3e3 photons, seed 7, 30 sweeps
{
  const g = R.geometry(N, R.angleSet(60));
  const ns = R.noisySinogram(R.project(shepp, g), 3e3, 7);
  const fb = R.fbp(ns.sino, g, { kind: "hann" });
  data.semiFbp = R.rmseOf(fb, shepp, mask);
  const sol = new R.IterativeRecon(ns.sino, g, { method: "art", omega: 0.5, nonneg: false, tv: 0, tvIters: 20 });
  const err = [R.rmseOf(sol.x, shepp, mask)], res = [1];
  let best = sol.x.slice(), bestE = 1e9, bestK = 0;
  for (let k = 1; k <= 30; k++) {
    sol.step(); const e = R.rmseOf(sol.x, shepp, mask);
    err.push(e); res.push(sol.residual());
    if (e < bestE) { bestE = e; bestK = k; best = sol.x.slice(); }
  }
  save("semi_best", best); save("semi_30", sol.x.slice());
  data.semiErr = err; data.semiRes = res; data.semiBestK = bestK;
}
fs.writeFileSync(path.join(OUT, "data.json"), JSON.stringify(data));
console.log(JSON.stringify({ rmse18: data.rmse18, rmse200: data.rmse200, rmseLemon: data.rmseLemon, rmseWedge: data.rmseWedge, semiFbp: data.semiFbp, semiBestK: data.semiBestK, semiErr: (data.semiErr as number[]).map(v => v.toFixed(3)).join(" ") }, null, 1));
