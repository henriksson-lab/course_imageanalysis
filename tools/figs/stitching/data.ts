// Data for the stitching/ figures, computed with the stitch, subpixel and drift demos' own
// models so the numbers match the demos and the slides. Called by figs.py:
//   esbuild data.ts --bundle --platform=node | node - <cell.f64> <out.json>
// cell.f64: the 256² micrograph as little-endian float64 (written by figs.py).
import { readFileSync, writeFileSync } from "fs";
import { createGrey, resize } from "../../../demos/src/lib/imageops";
import { solveGlobal, chainPositions } from "../../../demos/src/lib/registration";
import * as St from "../../../demos/src/stitch/model";
import * as Sp from "../../../demos/src/subpixel/model";
import * as Dr from "../../../demos/src/drift/model";

const [cellPath, outPath] = process.argv.slice(2);
const buf = readFileSync(cellPath);
const cell = Float64Array.from(new Float64Array(buf.buffer, buf.byteOffset, 256 * 256));
const g = createGrey(256, 256); g.data.set(cell);
const big = Float64Array.from(resize(g, St.BIG, St.BIG, "bilinear").data);
const out: Record<string, unknown> = {};

// --- stitch: default settings ("the four candidates", "global + blending", "stage")
function solve(s: St.StitchSettings) {
  const t = St.makeTiles(s, big), pairs = St.registerPairs(t).edges, E = St.weighted(pairs, 0.7);
  const gl = solveGlobal(St.NT, E)!, ch = chainPositions(St.NT, E, St.SNAKE);
  return { t, pairs, gl, ch, errG: St.positionErrors(t, gl.x, gl.y), errC: St.positionErrors(t, ch.x, ch.y) };
}
const DEF: St.StitchSettings = { ...St.DEFAULTS, vignette: 0.15 };   // as in the stitch demo (main.ts)
const d = solve(DEF);
const anchor = (x: ArrayLike<number>, y: ArrayLike<number>) => {
  const ox = d.t.pos[0][0] - x[0], oy = d.t.pos[0][1] - y[0];
  return Array.from(x, (v, k) => [v + ox, y[k] + oy]);
};
const e1 = d.pairs[1];
out.stitch = {
  T: St.T, BIG: St.BIG, tiles: d.t.tiles.map(a => Array.from(a)), truePos: d.t.pos,
  nominal: Array.from({ length: St.NT }, (_, k) => St.nominal(k)), globalPos: anchor(d.gl.x, d.gl.y),
  edge1: { i: St.EDGES[1][0], j: St.EDGES[1][1], candidates: e1.candidates, chosen: [e1.edge.dx, e1.edge.dy], ncc: e1.ncc },
  errG: d.errG, errC: d.errC,
};
const n12 = solve({ ...DEF, noise: 12 });
out.noise12 = { errG: n12.errG, errC: n12.errC, edges: St.EDGES, snake: St.SNAKE, ncc: n12.pairs.map(p => p.ncc) };
console.error("noise 12: chain max", Math.max(...n12.errC).toFixed(2), "global max", Math.max(...n12.errG).toFixed(2));
console.error("edge 1 candidates", JSON.stringify(e1.candidates));

// --- subpixel: "parabola: pixel locking" preset
const ss: Sp.SubSettings = { signal: "bumps", noise: 0, U: 20, method: "phase", frac: 0.3 };
const a = Sp.signalOf("bumps"), est = Sp.estimate(a, ss), sw = Sp.sweep({ ...ss, steps: 81 });
out.subpixel = { c: Array.from(est.c), t0: est.t0, shift: est.shift, parabola: est.parabola, centroid: est.centroid, up: est.up,
  curve: est.curve, sweep: sw };
console.error("subpixel parabola", est.parabola.toFixed(3), "up", est.up.toFixed(3), "max |parabola err|",
  Math.max(...sw.parabola.map(Math.abs)).toFixed(3));

// --- drift: "to the previous: errors add up" preset (bleach 0.8, noise 30)
const fr = Dr.makeFrames({ ...Dr.DEFAULTS, bleach: 0.8, noise: 30 }, cell);
const err = (m: Dr.DriftMode) => { const r = Dr.registerFrames(fr, m); return r.x.map((x, k) => Math.hypot(x - fr.dx[k], r.y[k] - fr.dy[k])); };
out.drift = { first: err("first"), previous: err("previous"), dx: fr.dx, dy: fr.dy };
const df = out.drift as { first: number[]; previous: number[] };
console.error("drift: first max", Math.max(...df.first).toFixed(2), "previous @30", df.previous[30].toFixed(2), "end", df.previous[39].toFixed(2));
writeFileSync(outPath, JSON.stringify(out));
