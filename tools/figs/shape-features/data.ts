// Data for the shape-features figures, from the moments demo's code ("features" mode, 64 x 64
// grid, pixel size 1). Run via figs.py; argument: output JSON path.
import * as fs from "fs";
import { rasterise, Poly } from "../../../demos/src/lib/shapecanvas";
import { widthProfile } from "../../../demos/src/lib/convexhull";
import { edt } from "../../../demos/src/lib/distance";
import { exactMeasures, disc, crescent, twoDiscs, rod, star } from "../../../demos/src/moments/shapes";
import { shapeFeatures, outline, fourierOutline } from "../../../demos/src/moments/features";

const S = 64, discArea = Math.PI * 16 * 16;
const scaledStar = (): Poly => { const p = star(32, 33, 26, 11, 5); const a = Math.abs(exactMeasures([p])!.area); return p.map(([x, y]) => [32 + (x - 32) * Math.sqrt(discArea / a), 33 + (y - 33) * Math.sqrt(discArea / a)] as [number, number]); };
const shapes: Record<string, Poly[]> = {
  disc16: [disc(32, 32, 16)],
  star: [scaledStar()],
  pair: [twoDiscs(32, 32, 13, 22)],
  rod: [rod(32, 32, 34, 5, 0.5)],
  crescent: [crescent(30, 32, 20, 14)],
  fstar: [star(32, 33, 26, 12, 5)],
};
const out: Record<string, unknown> = {};
for (const [k, polys] of Object.entries(shapes)) {
  const m = rasterise(polys, S, S);
  const f = shapeFeatures(m, S, S, "corners")!;
  const rec: Record<string, unknown> = {
    mask: Array.from(m), poly: polys[0],
    area: f.area, solidity: f.solidity, hull: f.hull, hullArea: f.hullArea,
    ffEdges: f.ffEdges, ffChain: f.ffChain, ffCrofton: f.ffCrofton,
    minFeret: f.calipers.minWidth, maxFeret: f.calipers.maxDiameter, maxPair: f.calipers.maxPair, minAngle: f.calipers.minAngle,
    inscribedR: f.inscribedR, inscribedAt: f.inscribedAt,
  };
  const wp = widthProfile(f.hull, 360);
  rec.wTheta = Array.from(wp.theta); rec.w = Array.from(wp.width);
  if (k === "crescent") rec.dt = Array.from(edt(m, S, S));
  if (k === "fstar") {
    const o = outline(m, S, S, 128);
    rec.outline = o;
    rec.fourier = Object.fromEntries([1, 4, 6, 24].map(K => [K, fourierOutline(o, K).smooth]));
  }
  out[k] = rec;
}
// form factor of digital discs against the radius, three perimeter estimators: mean over 16
// sub-pixel positions of the centre (the demo centres its disc on a pixel corner, which gives
// the jagged single values quoted in the text, e.g. 1.29 for radius 3.2 with the chain code)
const radii: number[] = [], fe: number[] = [], fc: number[] = [], fk: number[] = [];
let seed = 7;
const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
for (let r = 2; r <= 40.001; r += 0.25) {
  const n = 2 * Math.ceil(r + 4);
  let a = 0, b = 0, c = 0;
  const T = 16;
  for (let t = 0; t < T; t++) {
    const cx = n / 2 + rnd() - 0.5, cy = n / 2 + rnd() - 0.5;
    const m = rasterise([disc(cx, cy, r, 360)], n, n);
    const f = shapeFeatures(m, n, n, "corners")!;
    a += f.ffEdges; b += f.ffChain; c += f.ffCrofton;
  }
  radii.push(r); fe.push(a / T); fc.push(b / T); fk.push(c / T);
}
out.ff = { radii, edges: fe, chain: fc, crofton: fk };
{ const m = rasterise([disc(32, 32, 3.2)], S, S); const f = shapeFeatures(m, S, S)!; out.small = { ffChain: f.ffChain, ffEdges: f.ffEdges, ffCrofton: f.ffCrofton }; }
{ const m = rasterise([disc(32, 32, 22)], S, S); const f = shapeFeatures(m, S, S)!; out.disc22 = { ffChain: f.ffChain, ffEdges: f.ffEdges, ffCrofton: f.ffCrofton }; }
fs.writeFileSync(process.argv[2], JSON.stringify(out));
