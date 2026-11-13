// Calibration data for the cameras/ slide figures, computed with the dltcalib demo's own model
// (demos/src/dltcalib/model.ts: two checkerboard walls, f = 820 px, principal point (335, 245);
// a flat 8 × 6 board in 3 poses). Writes calib.json for figs.py. Build and run from the repo root:
//   node_modules/.bin/esbuild tools/figs/cameras/compute.ts --bundle --platform=node --outfile=<tmp>/c.js && node <tmp>/c.js <tmp>
import * as fs from "fs";
import { makeTarget3D, makePlanar, dltEstimate, pack, camOf, CalibProblem, fixedMask, rmsOf, lmStep } from "../../../demos/src/dltcalib/model";
import { project, Camera, V3 } from "../../../demos/src/lib/camera";

const out = process.argv[2] ?? ".";
const res: Record<string, unknown> = {};
const r2 = (v: number) => +v.toFixed(3);

function gridLines(cam: Camera, X: V3[], nx: number, ny: number, off: number): number[][][] {
  const lines: number[][][] = [];
  const seg = (a: V3, b: V3) => {
    const pts: number[][] = [];
    for (let k = 0; k <= 8; k++) { const t = k / 8, q = project(cam, [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]); pts.push([r2(q.u), r2(q.v)]); }
    lines.push(pts);
  };
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const k = off + j * nx + i;
    if (i + 1 < nx) seg(X[k], X[k + 1]);
    if (j + 1 < ny) seg(X[k], X[k + nx]);
  }
  return lines;
}

// "distortion" preset (k1 = −0.3, σ = 0.3 px) and "noise" preset (k1 = 0, σ = 0.5 px):
// residuals after the linear DLT and after LM refinement (20 steps; distortion modelled for the first)
for (const [name, sigma, k1, modelDist] of [["distortion", 0.3, -0.3, true], ["noise", 0.5, 0, false]] as [string, number, number, boolean][]) {
  const t = makeTarget3D(sigma, k1);
  const use = [t.X.map(() => true)];
  const e = dltEstimate(t, use[0], true)!;
  let th = pack(e.K, 0, 0, [{ R: e.R, t: e.t }]); th[4] = e.K[1];
  const prob = new CalibProblem(t.X, [t.obs], use);
  const resid = (th: Float64Array) => { const cam = camOf(th, 0); return t.X.map(p => { const q = project(cam, p); return [r2(q.u), r2(q.v)]; }); };
  const lin = resid(th), linRms = rmsOf(prob.residuals(th)), linF = th[0];
  th[4] = 0;
  let mu = 1e-3;
  for (let k = 0; k < 20; k++) { const s = lmStep(prob, th, { mu, lm: true, fixed: fixedMask(1, modelDist) }); th = s.theta; mu = s.mu; }
  const ref = resid(th);
  res[name] = { obs: t.obs.map(o => o.map(r2)), truth: t.X.map(p => { const q = project(t.cam, p); return [r2(q.u), r2(q.v)]; }), lin, ref, linRms, refRms: rmsOf(prob.residuals(th)), linF, refF: th[0], refK1: th[5],
    grid: [...gridLines(t.cam, t.X, 6, 6, 0), ...gridLines(t.cam, t.X, 6, 6, 36)], S: Array.from(e.S) };
  console.log(name, "linear fx", linF.toFixed(1), "rms", linRms.toFixed(2), "→ refined fx", th[0].toFixed(1), "k1", th[5].toFixed(3), "rms", rmsOf(prob.residuals(th)).toFixed(2));
}
// coplanar preset: singular values of A (σ = 0.5)
{
  const t = makeTarget3D(0.5, 0);
  const use = t.X.map((_, i) => i >= 36);
  const e = dltEstimate(t, use, true)!;
  res["coplanarS"] = Array.from(e.S);
  console.log("coplanar S", Array.from(e.S).map(v => v.toExponential(1)).join(" "));
}
// flat board: 3 tilted views and 3 nearly parallel views (σ = 0.5)
for (const [name, tilt] of [["planar", 1], ["parallel", 0.04]] as [string, number][]) {
  const pv = makePlanar(3, 0.5, 0, tilt);
  res[name] = pv.cams.map((c, v) => ({ obs: pv.obs[v].map(o => o.map(r2)), truth: pv.X.map(p => { const q = project(c, p); return [r2(q.u), r2(q.v)]; }) }));
}
fs.writeFileSync(`${out}/calib.json`, JSON.stringify(res));
