// Data for the intensity-features figures, from the intensityfeat demo's own code
// (same cells, same acquisition, same plates). Run via figs.py; argument: output JSON path.
import * as fs from "fs";
import { makeCell, observe, adjustMask, makePlate, normalisePlate, N, PatternId, ellipseMask, CONTROL, TREATED } from "../../../demos/src/intensityfeat/model";
import { intensityRow, radialDistribution, edgeBand, backgroundRing, pixelsOf, valuesAt } from "../../../demos/src/lib/features";
import { median } from "../../../demos/src/lib/stats";

const out: Record<string, unknown> = { N };
const arr = (a: ArrayLike<number>) => Array.from(a, v => Math.round(v * 1000) / 1000);
const cells: Record<string, unknown> = {};
for (const p of ["uniform", "rim", "spot", "punctate", "gradient", "nuclear"] as PatternId[]) {
  const c = makeCell({ pattern: p });
  const row = intensityRow(c.clean, c.mask, N, N);
  const rad = radialDistribution(c.clean, c.mask, N, N, 4, [row.cx, row.cy]);
  cells[p] = {
    img: arr(c.clean), mask: Array.from(c.mask), row,
    values: arr(valuesAt(c.clean, pixelsOf(c.mask))),
    ring: Array.from(rad.ring), fracAtD: Array.from(rad.fracAtD), meanFrac: Array.from(rad.meanFrac), areaFrac: Array.from(rad.areaFrac),
  };
}
out.cells = cells;
out.band = Array.from(edgeBand((cells.uniform as any).mask, N, N));

// local background: the uniform cell with offset 10 and noise 4, a neighbour in the corner
{
  const c = makeCell({ pattern: "uniform" });
  const nb = ellipseMask(N, 134, 20, 36, 24, -30);
  const occ = Uint8Array.from(nb, (v, i) => (v || c.mask[i] ? 1 : 0));
  const clean = Float64Array.from(c.clean, (v, i) => v + (nb[i] ? 80 : 0));
  const img = observe(clean, { offset: 10, gain: 1, noise: 4, sat: Infinity });
  const ring = backgroundRing(c.mask, N, N, 3, 6, occ);
  const ringNoExcl = backgroundRing(c.mask, N, N, 3, 6);
  const bg = median(valuesAt(img, pixelsOf(ring)));
  const bgNo = median(valuesAt(img, pixelsOf(ringNoExcl)));
  const raw = intensityRow(img, c.mask, N, N);
  const corr = intensityRow(img.map(v => v - bg), c.mask, N, N);
  out.bg = { nRingNo: pixelsOf(ringNoExcl).length, meanNo: (() => { const v = valuesAt(img, pixelsOf(ringNoExcl)); return v.reduce((a, b) => a + b, 0) / v.length; })(), meanRing: (() => { const v = valuesAt(img, pixelsOf(ring)); return v.reduce((a, b) => a + b, 0) / v.length; })(), img: arr(img), mask: Array.from(c.mask), nb: Array.from(nb), ring: Array.from(ring), bg, bgNo, rawMean: raw.mean, corrMean: corr.mean, nRing: pixelsOf(ring).length };
}

// which numbers survive: uniform cell ideal, offset 10, offset 10 + mask 2 px too large; spot clipped at 300
{
  const c = makeCell({ pattern: "uniform" });
  const off = observe(c.clean, { offset: 10, gain: 1, noise: 0, sat: Infinity });
  const big = adjustMask(c.mask, N, N, 2);
  const s = makeCell({ pattern: "spot" });
  const clip = observe(s.clean, { offset: 0, gain: 1, noise: 0, sat: 300 });
  out.survive = {
    ideal: arr(valuesAt(c.clean, pixelsOf(c.mask))), offset: arr(valuesAt(off, pixelsOf(c.mask))), dilated: arr(valuesAt(off, pixelsOf(big))),
    rIdeal: intensityRow(c.clean, c.mask, N, N), rOffset: intensityRow(off, c.mask, N, N), rDilated: intensityRow(off, big, N, N),
    spot: arr(valuesAt(s.clean, pixelsOf(s.mask))), clip: arr(valuesAt(clip, pixelsOf(s.mask))), rSpot: intensityRow(s.clean, s.mask, N, N), rClip: intensityRow(clip, s.mask, N, N),
  };
}
// plates (batch preset: no gradient), raw and robust z
{
  const d = makePlate({ gradient: 0 });
  out.plate = { A: Array.from(d.values), B: Array.from(d.valuesB), zA: Array.from(normalisePlate(d.values, "robustz")), zB: Array.from(normalisePlate(d.valuesB, "robustz")), control: Array.from(CONTROL), treated: Array.from(TREATED) };
}
fs.writeFileSync(process.argv[2], JSON.stringify(out));
