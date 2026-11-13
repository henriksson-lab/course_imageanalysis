// Data for the figures of coloc-objects/lec.qmd, computed with the colocobjects demo code
// (same scenes, seeds and spot detector), so every number matches the deck. Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/coloc-objects/data.ts --bundle --platform=node --log-level=warning | node - <outdir>
//   python3 -I tools/figs/coloc-objects/figs.py <outdir>
import { writeFileSync } from "fs";
import { makeObjScene, DEFAULT_OBJ, distances, pairedDistances, pixelR, makeBlobScene } from "../../../demos/src/colocobjects/model";
import { randomPointsInMask, S } from "../../../demos/src/lib/twochannel";
import { rng } from "../../../demos/src/lib/random";

const out = process.argv[2] ?? ".";
const arr = (v: ArrayLike<number>) => Array.from(v, x => Math.round(x * 1000) / 1000);
const pts = (s: { x: number; y: number }[]) => s.map(p => [Math.round(p.x * 100) / 100, Math.round(p.y * 100) / 100]);
const data: Record<string, unknown> = {};
const log: Record<string, unknown> = {};

// preset "pairs at 0 nm": image, detections, nearest distances at dMax 130 nm
{
  const sc = makeObjScene({ ...DEFAULT_OBJ });
  const d = distances(sc, 130, false);
  // relocations of the detected B spots: inside the cells (as the demo's envelope) and over the whole image
  const r = rng(5);
  const inMask = randomPointsInMask(sc.detB.length, sc.mask, S, S, r, 2);
  const whole = randomPointsInMask(sc.detB.length, undefined, S, S, r, 2);
  data.pairs0 = {
    a: arr(sc.a), b: arr(sc.b), mask: Array.from(sc.mask), detA: pts(sc.detA), detB: pts(sc.detB), partner: Array.from(d.partner), dist: arr(d.dist),
    inMask: Array.from(inMask.x, (x, i) => [x, inMask.y[i]]), whole: Array.from(whole.x, (x, i) => [x, whole.y[i]]),
  };
  log.pairs0 = { nA: sc.detA.length, nB: sc.detB.length, fObs: d.fObs, fChance: d.fChance, r: pixelR(sc) };
}
// preset "pairs at 60 nm, 2000 photons": measured distances of the true pairs
{
  const sc = makeObjScene({ ...DEFAULT_OBJ, pair: 1, dNm: 60, extraB: 0, photons: 2000 });
  const p = pairedDistances(sc);
  data.d60 = { measured: p.measured, sigA: p.sigA, sigB: p.sigB, predicted: p.predicted, r: pixelR(sc) };
  log.d60 = { n: p.measured.length, mean: p.measured.reduce((a, b) => a + b, 0) / p.measured.length, sigA: p.sigA, sigB: p.sigB, predicted: p.predicted, r: pixelR(sc) };
}
// big objects: the three overlap definitions (preset "big" and "small")
{
  const big = makeBlobScene({ rA: 6, rB: 4, dPx: 5, pair: 0.6 }), small = makeBlobScene({ rA: 4, rB: 6, dPx: 5, pair: 0.6 });
  log.blobs = { big: [big.anyFrac, big.halfFrac, big.centroidFrac], small: [small.anyFrac, small.halfFrac, small.centroidFrac] };
}
writeFileSync(`${out}/coloc-objects.json`, JSON.stringify(data));
console.log(JSON.stringify(log, null, 1));
