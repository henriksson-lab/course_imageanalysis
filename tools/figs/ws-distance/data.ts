// Data for the ws-distance figures, computed with the distws demo's own code
// (demos/src/distws/geom.ts) and its presets (N = 180).
import { writeFileSync } from "fs";
import { shapesMask, splitMask, withHoles, profile, neckOf, Shape, Metric } from "../../../demos/src/distws/geom";
import { nuclei } from "../../../demos/src/lib/synth";

const N = 180;
const disc = (x: number, y: number, r: number): Shape => ({ x, y, a: r, b: r, angle: 0 });
const arr = (a: ArrayLike<number>) => Array.from(a, v => Math.round(v * 100) / 100);
const out: any = { n: N };
// two discs: split preset, all four metrics
const two = shapesMask(N, N, [disc(70, 90, 30), disc(110, 90, 30)]);
out.metrics = {} as any;
for (const m of ["euclidean", "cityblock", "3-4", "chessboard"] as Metric[]) {
  const r = splitMask(two, N, N, m, 2, "dt");
  const nk = neckOf(profile(r.dt, N, N, 40, 90, 140, 90));
  // the chamfer 3-4 transform is in units of 3 per pixel step in the lib? report as the demo does
  out.metrics[m] = { dt: arr(r.dt), labels: Array.from(r.labels), n: r.nObjects, ...nk, peaks: r.peakCentres };
}
out.twoMask = Array.from(two);
// nuclei image, threshold 115, h = 2
const nuc = nuclei({ width: N, height: N, count: 15, radius: [10, 15], touching: 0.6, seed: 4, noise: 6 });
const mask = Uint8Array.from(nuc.image.data, v => (v > 115 ? 1 : 0));
const rn = splitMask(mask, N, N, "euclidean", 2, "dt");
out.nuc = { image: arr(nuc.image.data), mask: Array.from(mask), labels: Array.from(rn.labels), n: rn.nObjects, comps: rn.nComponents,
  dt: arr(rn.dt), peaks: rn.peakCentres, ellipses: nuc.ellipses, pairs: nuc.touchingPairs };
// failure modes: ellipse (h = 0) and holes (h = 0)
const ell = shapesMask(N, N, [{ x: 90, y: 90, a: 55, b: 22, angle: 0.3 }]);
const re = splitMask(ell, N, N, "euclidean", 0, "dt");
out.ellipse = { mask: Array.from(ell), labels: Array.from(re.labels), n: re.nObjects, peaks: re.peakCentres };
const hol = withHoles(shapesMask(N, N, [disc(65, 90, 30), disc(115, 90, 30)]), N, N, 12);
const rh = splitMask(hol, N, N, "euclidean", 0, "dt");
out.holes = { mask: Array.from(hol), labels: Array.from(rh.labels), n: rh.nObjects, peaks: rh.peakCentres };
writeFileSync(process.argv[2], JSON.stringify(out));
