// Data for the figures of coloc/lec.qmd, computed with the course demo code (the coloc and
// colocgrid demos), so every number matches the deck. Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/coloc/data.ts --bundle --platform=node --log-level=warning | node - <outdir>
//   python3 -I tools/figs/coloc/figs.py <outdir>
import { writeFileSync } from "fs";
import { pixelScene } from "../../../demos/src/lib/twochannel";
import { pearsonMasked, spearmanMasked, orthoRegression, costes, manders } from "../../../demos/src/lib/coloc";
import { startThresholds } from "../../../demos/src/coloc/model";

const out = process.argv[2] ?? ".";
const arr = (v: ArrayLike<number>) => Array.from(v, x => Math.round(x * 1000) / 1000);
const data: Record<string, unknown> = {};

// three presets of fig-merge: perfect, independent, bleed-through (inside the cells)
for (const [k, o] of [["perfect", { f: 1 }], ["independent", { f: 0 }], ["bleed", { f: 0, bleed: 0.4 }]] as const) {
  const ch = pixelScene(o);
  data[k] = { a: arr(ch.a), b: arr(ch.b), cells: Array.from(ch.cells), r: pearsonMasked(ch.a, ch.b, ch.cells) };
}
// region: whole image vs inside the cells (preset "background inflates r")
{
  const ch = pixelScene({ f: 0, diffuse: 30 });
  data.region = { a: arr(ch.a), b: arr(ch.b), cells: Array.from(ch.cells), rAll: pearsonMasked(ch.a, ch.b), rCells: pearsonMasked(ch.a, ch.b, ch.cells) };
}
// colocgrid preset "one bright outlier" (copied from demos/src/colocgrid/main.ts)
{
  const NA = [1, 2, 0, 1, 3, 2, 2, 1, 1, 0, 2, 3, 0, 3, 2, 1, 1, 0, 1, 2, 3, 0, 2, 1, 3, 0, 1, 2, 1, 2, 2, 1, 0, 3, 1, 0];
  const NB = [2, 0, 1, 3, 1, 0, 1, 3, 0, 2, 2, 1, 3, 1, 1, 0, 2, 3, 0, 2, 1, 3, 0, 2, 1, 2, 3, 0, 2, 1, 0, 1, 2, 1, 3, 2];
  const a = NA.slice(), b = NB.slice();
  a[14] = 20; b[14] = 20;
  data.outlier = { a, b, r: pearsonMasked(a, b), rho: spearmanMasked(a, b), rNo: pearsonMasked(a.filter((_, i) => i !== 14), b.filter((_, i) => i !== 14)) };
}
// A inside B: thresholds as the demo chooses them, M1/M2
{
  const ch = pixelScene({ layout: "inside" });
  const t = startThresholds(ch, ch.cells);
  const m = manders(ch.a, ch.b, t.tA, t.tB, ch.cells);
  data.inside = { a: arr(ch.a), b: arr(ch.b), cells: Array.from(ch.cells), tA: t.tA, tB: t.tB, costesOk: t.costes.ok, m1: m.m1, m2: m.m2, r: pearsonMasked(ch.a, ch.b, ch.cells) };
}
// Costes on the perfect scene: regression line, trace and thresholds
{
  const ch = pixelScene({ f: 1 });
  const reg = orthoRegression(ch.a, ch.b, ch.cells);
  const c = costes(ch.a, ch.b, ch.cells, 100);
  const m = manders(ch.a, ch.b, c.tA, c.tB, ch.cells);
  data.costes = { slope: reg.slope, intercept: reg.intercept, tA: c.tA, tB: c.tB, trace: c.trace, m1: m.m1, m2: m.m2 };
}
writeFileSync(`${out}/coloc.json`, JSON.stringify(data));
console.log(JSON.stringify({ perfect: (data.perfect as any).r, indep: (data.independent as any).r, bleed: (data.bleed as any).r, region: [(data.region as any).rAll, (data.region as any).rCells], outlier: [(data.outlier as any).r, (data.outlier as any).rho, (data.outlier as any).rNo], inside: { ...(data.inside as any), a: 0, b: 0, cells: 0 }, costes: { ...(data.costes as any), trace: (data.costes as any).trace.length } }, null, 1));
