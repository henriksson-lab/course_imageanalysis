// Data for the phasecorr/ figures, computed with the phasecorr demo's own model so the
// numbers match the demo and the slides. Called by figs.py:
//   esbuild data.ts --bundle --platform=node | node - <cell.f64> <out.json>
// cell.f64: the 256² micrograph as little-endian float64 (written by figs.py).
import { readFileSync, writeFileSync } from "fs";
import { DEFAULTS, PCSettings, makeCrops, register, texture, IMG } from "../../../demos/src/phasecorr/model";

const [cellPath, outPath] = process.argv.slice(2);
const buf = readFileSync(cellPath);
const cell = new Float64Array(buf.buffer, buf.byteOffset, IMG * IMG);
const tex = texture();

const cases: Record<string, Partial<PCSettings>> = {
  cc: { method: "cc" },
  phase: { method: "phase" },
  ramp_cc: { method: "cc", ramp: 500 },
  ramp_phase: { method: "phase", ramp: 500 },
  periodic: { source: "texture", method: "phase" },
};
const out: Record<string, unknown> = {};
for (const [id, p] of Object.entries(cases)) {
  const s = { ...DEFAULTS, ...p } as PCSettings;
  const c = makeCrops(s.source === "texture" ? tex : cell, s);
  const r = register(c, s);
  out[id] = {
    n: s.size, a: Array.from(c.a), b: Array.from(c.b), surface: Array.from(r.surface.data),
    peaks: r.peaks.map(q => [q.tx, q.ty, q.value]), ratio: r.ratio, tx: r.tx, ty: r.ty, height: r.height,
    trueShift: [s.tx, s.ty], frameA: [s.ax, s.ay],
    // value at the true shift
    atTrue: r.surface.data[((s.ty % s.size) + s.size) % s.size * s.size + ((s.tx % s.size) + s.size) % s.size],
  };
  console.error(id, "peak", r.tx, r.ty, r.height.toFixed(3), "ratio", r.ratio.toFixed(2));
}
writeFileSync(outPath, JSON.stringify(out));
