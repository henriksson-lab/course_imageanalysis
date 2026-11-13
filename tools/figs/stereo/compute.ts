// Disparity maps for the stereo/ slide figures, computed with the course's own stereo code
// (demos/src/lib/stereo.ts) and the same settings as the stereomatch demo presets
// (192 × 144, noise 2, seed 7; costs scaled as in the demo). Writes stereo.json for figs.py.
// Build and run from the repo root:
//   node_modules/.bin/esbuild tools/figs/stereo/compute.ts --bundle --platform=node --outfile=<tmp>/s.js && node <tmp>/s.js <tmp>
import * as fs from "fs";
import { makeScene, costVolume, costVolumeRight, wta, scanlineDP, sgm, leftRightCheck, badPixelRate, CostKind, StereoScene } from "../../../demos/src/lib/stereo";

const out = process.argv[2] ?? ".";
const SCALE: Record<CostKind, number> = { sad: 1 / 64, ssd: 1 / 4096, ncc: 1, census: 1 / 8 };
const res: Record<string, unknown> = {};
const arr = (a: ArrayLike<number>) => Array.from(a, v => +(+v).toFixed(2));

function vol(sc: StereoScene, kind: CostKind, r: number, right = false): Float32Array {
  const C = (right ? costVolumeRight : costVolume)(sc.left, sc.right, sc.w, sc.h, sc.maxD, kind, r);
  for (let i = 0; i < C.length; i++) C[i] *= SCALE[kind];
  return C;
}
function put(name: string, sc: StereoScene, d: ArrayLike<number>) {
  const r = badPixelRate(d, sc);
  res[name] = { d: arr(d), rate: r.rate };
  console.log(name, (100 * r.rate).toFixed(1) + " % bad");
}
function scene(name: string, sc: StereoScene) {
  res[name] = { w: sc.w, h: sc.h, maxD: sc.maxD, left: arr(sc.left), right: arr(sc.right), truth: arr(sc.disp), occ: Array.from(sc.occluded) };
}

// random dots, SAD, WTA, three window sizes (preset "window")
const dots = makeScene("dots", 192, 144, 2, 7);
scene("dots", dots);
for (const r of [1, 3, 7]) put(`dots_wta_r${r}`, dots, wta(vol(dots, "sad", r), dots.w * dots.h, dots.maxD));

// wall, fence, post: SAD 3 × 3, WTA / scanline DP / SGM 8 (P1 = 0.1, P2 = 1)
const mixed = makeScene("mixed", 192, 144, 2, 7);
scene("mixed", mixed);
const n = mixed.w * mixed.h, D = mixed.maxD;
const C = vol(mixed, "sad", 1);
put("mixed_wta", mixed, wta(C, n, D));
put("mixed_dp", mixed, scanlineDP(C, mixed.w, mixed.h, D, 0.1, 1).disp);
const dl = wta(sgm(C, mixed.w, mixed.h, D, 0.1, 1, 8), n, D);
put("mixed_sgm8", mixed, dl);
const CR = vol(mixed, "sad", 1, true);
const dr = wta(sgm(CR, mixed.w, mixed.h, D, 0.1, 1, 8), n, D);
put("mixed_sgm8_lr", mixed, leftRightCheck(dl, dr, mixed.w, mixed.h, 1));

// gain change: SAD vs census, 7 × 7, WTA (preset "gain")
const gain = makeScene("gain", 192, 144, 2, 7);
scene("gain", gain);
for (const k of ["sad", "ncc", "census"] as CostKind[]) put(`gain_${k}`, gain, wta(vol(gain, k, 3), gain.w * gain.h, gain.maxD));

fs.writeFileSync(`${out}/stereo.json`, JSON.stringify(res));
