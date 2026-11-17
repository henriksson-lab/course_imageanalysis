// Data for the figures of the pretext unit, computed with the maskfill demo's own code.
// Run from the repository root (DIR = any scratch folder):
//   node_modules/.bin/esbuild tools/figs/pretext/dump.ts --bundle --platform=node --outfile=DIR/pt.js && node DIR/pt.js > DIR/pt.json
//   python3 -I tools/figs/pretext/figs.py DIR/pt.json
import { fitPca, fillInterp, fillPca, hiddenPsnr, makeMask, scene, N } from "../../../demos/src/maskfill/model";
import { loadCellNode } from "../../../demos/test/selfsup-cell";

const cell = loadCellNode().data;
const q = (a: ArrayLike<number>) => Array.from(a, v => (Number.isFinite(v) ? Math.round(Math.max(0, Math.min(255, v))) : -1));
const out: any = { N };
for (const sc of ["nuclei", "cell"] as const) {
  const s = scene(sc, cell), m = fitPca(s.windows);
  const o: any = { test: q(s.test), comps: m.comps.slice(0, 8).map(c => Array.from(c, v => Math.round(v * 1e4) / 1e4)), values: m.values.slice(0, 12) };
  for (const [kind, ratio] of [["patches", 0.75], ["pixels", 0.75]] as const) {
    const mask = makeMask(kind, ratio, 7);
    const a = fillInterp(s.test, mask), b = fillPca(s.test, mask, m, 32);
    o[kind] = { seen: q(Float64Array.from(s.test, (v, i) => (mask[i] ? NaN : v))), interp: q(a), pca: q(b), pi: hiddenPsnr(a, s.test, mask), pp: hiddenPsnr(b, s.test, mask) };
  }
  o.curve = [0.1, 0.25, 0.5, 0.75, 0.9].map(r => {
    const res: number[] = [r];
    for (const kind of ["patches", "pixels"] as const) {
      const mask = makeMask(kind, r, 7);
      res.push(hiddenPsnr(fillInterp(s.test, mask), s.test, mask), hiddenPsnr(fillPca(s.test, mask, m, 32), s.test, mask));
    }
    return res;
  });
  out[sc] = o;
}
console.log(JSON.stringify(out));
