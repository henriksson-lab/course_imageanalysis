// Data for the figures of coloc-null/lec.qmd, computed with the colocnull demo code (same
// scenes, seeds and shuffles), so every number matches the deck. Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/coloc-null/data.ts --bundle --platform=node --log-level=warning | node - <outdir>
//   python3 -I tools/figs/coloc-null/figs.py <outdir>
import { writeFileSync } from "fs";
import { pixelScene, S } from "../../../demos/src/lib/twochannel";
import { pearsonMasked, randomisationTest, blockScrambleMasked, pdm, icq, costes, manders } from "../../../demos/src/lib/coloc";
import { rng } from "../../../demos/src/lib/random";
import { sd } from "../../../demos/src/lib/stats";

const out = process.argv[2] ?? ".";
const arr = (v: ArrayLike<number>) => Array.from(v, x => Math.round(x * 1000) / 1000);
const data: Record<string, unknown> = {};
const log: Record<string, unknown> = {};

// references: colocalised scene; B shifted by 10 px; B block-shuffled (8 px, inside the cells)
{
  const ch = pixelScene({ f: 1 });
  const shifted = new Float64Array(S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) shifted[y * S + x] = ch.b[y * S + Math.min(S - 1, Math.max(0, x - 10))];
  const shuf = blockScrambleMasked(ch.b, S, S, 8, rng(77), ch.cells);
  const r = [pearsonMasked(ch.a, ch.b, ch.cells), pearsonMasked(ch.a, shifted, ch.cells), pearsonMasked(ch.a, shuf, ch.cells)];
  data.refs = { a: arr(ch.a), b: arr(ch.b), shifted: arr(shifted), shuf: arr(shuf), cells: Array.from(ch.cells), r };
  log.refs = r;
}
// block sizes: B of the independent blur-2 scene (seed 15), shuffled in 1-px and 8-px blocks
{
  const ch = pixelScene({ f: 0, blur: 2, seed: 15 });
  const b1 = blockScrambleMasked(ch.b, S, S, 1, rng(78), ch.cells), b8 = blockScrambleMasked(ch.b, S, S, 8, rng(77), ch.cells);
  data.blocks = { b: arr(ch.b), b1: arr(b1), b8: arr(b8), cells: Array.from(ch.cells) };
}
// bleed-through passes the test (f 0, bleed 0.4, 8-px blocks, 200 shuffles, seed 77)
{
  const ch = pixelScene({ f: 0, bleed: 0.4 });
  const t = randomisationTest(ch.a, ch.b, S, S, { block: 8, n: 200, seed: 77, mask: ch.cells });
  data.bleed = { null: arr(t.null), observed: t.observed, p: t.p, sd: sd(Array.from(t.null)) };
  log.bleed = { observed: t.observed, p: t.p, max: Math.max(...t.null) };
}
// PDM images and ICQ: colocalised, independent, segregated
for (const [k, o] of [["coloc", { f: 1 }], ["independent", { f: 0 }], ["segregated", { f: 0, layout: "exclude" as const }]] as const) {
  const ch = pixelScene(o);
  const d = pdm(ch.a, ch.b, ch.cells), q = icq(ch.a, ch.b, ch.cells);
  data["pdm_" + k] = { pdm: arr(d), cells: Array.from(ch.cells), icq: q.icq, r: pearsonMasked(ch.a, ch.b, ch.cells) };
  (log as any)["pdm_" + k] = [q.icq, pearsonMasked(ch.a, ch.b, ch.cells)];
}
// Costes traces: colocalised, weak correlation, shared background (whole image)
for (const [k, o, roi] of [["coloc", { f: 1 }, true], ["weak", { f: 0.15, noise: 12 }, true], ["background", { f: 0, uneven: 40 }, false]] as const) {
  const ch = pixelScene(o);
  const m = roi ? ch.cells : undefined;
  const c = costes(ch.a, ch.b, m, 100);
  let n = 0, nAbove = 0;
  for (let i = 0; i < S * S; i++) if (!m || m[i]) { n++; if (ch.a[i] > c.tA && ch.b[i] > c.tB) nAbove++; }
  const md = manders(ch.a, ch.b, c.tA, c.tB, m);
  data["costes_" + k] = { trace: c.trace, tA: c.tA, n, nAbove, m1: md.m1, m2: md.m2 };
  (log as any)["costes_" + k] = { tA: c.tA, steps: c.trace.length, n, nAbove, m1: md.m1, m2: md.m2, r0: c.trace[0]?.r };
}
writeFileSync(`${out}/coloc-null.json`, JSON.stringify(data));
console.log(JSON.stringify(log, null, 1));
