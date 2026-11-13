// Data for the figures of coloc-confounds/lec.qmd, computed with the colocpipeline demo code
// (same scene, seeds, corrections), so every number matches the deck. Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/coloc-confounds/data.ts --bundle --platform=node --log-level=warning | node - <outdir>
//   python3 -I tools/figs/coloc-confounds/figs.py <outdir>
import { writeFileSync } from "fs";
import { runPipeline, DEFAULT_PIPE, PipeOptions, scene, mask, sigmaA, sigmaB, PIXEL_NM, OFFSET, BASE_BG } from "../../../demos/src/colocpipeline/model";
import { S, spotImage, unevenBackground, addNoise, leakSlope } from "../../../demos/src/lib/twochannel";

const out = process.argv[2] ?? ".";
const arr = (v: ArrayLike<number>) => Array.from(v, x => Math.round(x * 100) / 100);
const data: Record<string, unknown> = {};
const log: Record<string, unknown> = {};
const run = (o: Partial<PipeOptions>) => runPipeline({ ...DEFAULT_PIPE, ...o });
const rd = (r: ReturnType<typeof run>["read"]["measured"]) => ({ r: r.r, m1: r.m1, m2: r.m2, f: r.fObs });

// forward model, step by step ("everything at once": shared background 40, 30 % bleed, 100 nm shift)
{
  const o = { ...DEFAULT_PIPE, uneven: 40, bleed: 0.3, shiftXNm: 100 };
  const t = scene(o), m = mask();
  const ea = spotImage(t.ax, t.ay, S, S, sigmaA, o.photons);
  const eb0 = spotImage(t.bx, t.by, S, S, sigmaB, o.photons);
  const eb1 = spotImage(t.bx, t.by, S, S, sigmaB, o.photons, o.shiftXNm / PIXEL_NM, 0);
  const g = unevenBackground(S, S, 9);
  const bg = Float64Array.from({ length: S * S }, (_, i) => BASE_BG * (0.3 + 0.7 * m[i]) + o.uneven * g[i]);
  const eb2 = eb1.map((v, i) => v + o.bleed * ea[i]);
  const res = run(o);
  data.forward = {
    ax: Array.from(t.ax), ay: Array.from(t.ay), bx: Array.from(t.bx), by: Array.from(t.by), partner: Array.from(t.partner),
    psfA: arr(ea), psfB: arr(eb0), shiftB: arr(eb1), bleedB: arr(eb2), bg: arr(bg), measA: arr(res.measured.a), measB: arr(res.measured.b),
  };
  const steps = {
    measured: rd(res.read.measured), ideal: rd(res.read.ideal),
    subtract: rd(run({ ...o, subtract: true }).read.corrected),
    compensate: rd(run({ ...o, subtract: true, compensate: true, sHat: 0.3 }).read.corrected),
    register: rd(run({ ...o, subtract: true, compensate: true, sHat: 0.3, register: true }).read.corrected),
    skip1: rd(run({ ...o, compensate: true, sHat: 0.3 }).read.corrected),
  };
  data.steps = steps; log.steps = steps;
}
// background: shared uneven background 40, then top-hat
{
  const m = run({ uneven: 40 }), c = run({ uneven: 40, subtract: true });
  data.background = { idealA: arr(m.ideal.a), idealB: arr(m.ideal.b), measA: arr(m.measured.a), measB: arr(m.measured.b), corrA: arr(c.corrected.a), corrB: arr(c.corrected.b), ideal: rd(m.read.ideal), measured: rd(m.read.measured), corrected: rd(c.read.corrected) };
  log.background = { ideal: rd(m.read.ideal), measured: rd(m.read.measured), corrected: rd(c.read.corrected) };
}
// single-stained control: only dye A present, 30 % of it leaks into channel B
{
  const t = scene(DEFAULT_PIPE), m = mask();
  const ea = spotImage(t.ax, t.ay, S, S, sigmaA, DEFAULT_PIPE.photons);
  const bg = (i: number) => BASE_BG * (0.3 + 0.7 * m[i]);
  const a = addNoise(ea.map((v, i) => v + bg(i)), 51, OFFSET), b = addNoise(ea.map((v, i) => 0.3 * v + bg(i)), 52, OFFSET);
  // background-subtracted values (offset + mean background), inside the cells
  const ia: number[] = [], ib: number[] = [];
  for (let i = 0; i < S * S; i++) if (m[i]) { ia.push(a[i] - OFFSET - bg(i)); ib.push(b[i] - OFFSET - bg(i)); }
  const s = leakSlope(ia, ib);
  data.control = { a: arr(ia), b: arr(ib), s };
  log.control = s;
}
// chromatic shift 100 nm, then registered with beads
{
  const sh = run({ shiftXNm: 100 }), rg = run({ shiftXNm: 100, register: true });
  data.chromatic = { idealA: arr(sh.ideal.a), idealB: arr(sh.ideal.b), measA: arr(sh.measured.a), measB: arr(sh.measured.b), corrA: arr(rg.corrected.a), corrB: arr(rg.corrected.b), measured: rd(sh.read.measured), corrected: rd(rg.read.corrected), est: rg.estShiftNm };
  log.chromatic = { measured: rd(sh.read.measured), corrected: rd(rg.read.corrected), est: rg.estShiftNm };
}
writeFileSync(`${out}/coloc-confounds.json`, JSON.stringify(data));
console.log(JSON.stringify(log, null, 1));
