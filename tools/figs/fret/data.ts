// Scene data for the image figures of the FRET day (figs.py), computed with the demos' own
// code (demos/src/lib/fret.ts) and the demos' preset parameters:
//   threecube: fretField(seed 5, E 0.3), measureField(photons 300, offset 100, seed 3)
//   apbleach:  bleachScene(seed 4, E 0.3, f 1, 15 pulses, 600 photons), "moved": drift (3, 2)
//   ratioimage: sensorScene(), sensorFrame(frame 20, photons 100, offset 40, seed 1)
// Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/fret/data.ts --bundle --platform=node --outfile=<tmp>/fret-data.js
//   node <tmp>/fret-data.js <tmp>/fret-data.json
// then python3 -I tools/figs/fret/figs.py <tmp>/fret-data.json
import { writeFileSync } from "fs";
import * as F from "../../../demos/src/lib/fret";

const r2 = (v: number) => Math.round(v * 100) / 100;
const arr = (a: ArrayLike<number>) => Array.from(a, r2);
const out: Record<string, unknown> = {};

// ---- sensitised emission (threecube) ----
{
  const field = F.fretField({ seed: 5, E: 0.3 });
  const OFF = 100;
  const M = F.measureField(field, { photons: 300, offset: OFF, seed: 3 });
  const Ms = F.measureField(field, { photons: 300, offset: OFF, seed: 3, shift: 1 });
  const n = field.w * field.h, dm = new Uint8Array(n), am = new Uint8Array(n);
  for (let i = 0; i < n; i++) { const L = field.labels[i]; if (!L) continue; if (field.cls[L - 1] === "donor") dm[i] = 1; if (field.cls[L - 1] === "acceptor") am[i] = 1; }
  const sub = (v: Float32Array) => Float32Array.from(v, x => x - OFF);
  const d = F.slopeThroughOriginMasked(sub(M.IDD), sub(M.IDA), dm), a = F.slopeThroughOriginMasked(sub(M.IAA), sub(M.IDA), am);
  const ds = F.slopeThroughOriginMasked(sub(Ms.IDD), sub(Ms.IDA), dm), as = F.slopeThroughOriginMasked(sub(Ms.IAA), sub(Ms.IDA), am);
  console.log("threecube measured d, a:", d, a, " shifted:", ds, as);
  out.field = {
    w: field.w, h: field.h, labels: Array.from(field.labels), cls: field.cls, offset: OFF, photons: 300,
    IDD: arr(M.IDD), IDA: arr(M.IDA), IAA: arr(M.IAA), sIDD: arr(Ms.IDD), sIDA: arr(Ms.IDA), sIAA: arr(Ms.IAA),
    d, a, ds, as, G: field.G,
  };
}

// ---- acceptor photobleaching (apbleach) ----
{
  const scene = (drift: [number, number]) => F.bleachScene({ E: 0.3, f: 1, pulses: 15, donorLoss: 0, photons: 600, drift, photoconvert: 0, seed: 4 });
  const pack = (sc: F.BleachScene, post: Float32Array, accPost: Float32Array) => {
    let sPre = 0, sPost = 0, sA0 = 0, sA1 = 0;
    for (let i = 0; i < sc.region.length; i++) if (sc.region[i] && sc.labels[i] && sc.labels[i] !== sc.accOnlyCell) { sPre += sc.donorPre[i]; sPost += post[i]; sA0 += sc.accPre[i]; sA1 += accPost[i]; }
    return { pre: arr(sc.donorPre), post: arr(post), Ereg: F.eFromBleach(sPre, sPost), accLeft: sA1 / sA0 };
  };
  const ideal = scene([0, 0]), moved = scene([3, 2]);
  let x0 = 1e9, y0 = 1e9, x1 = -1, y1 = -1;
  for (let i = 0; i < ideal.region.length; i++) if (ideal.region[i] && ideal.labels[i] !== ideal.accOnlyCell) { const x = i % ideal.w, y = (i - x) / ideal.w; x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x + 1); y1 = Math.max(y1, y + 1); }
  const al = F.shiftMap(moved.donorPost, moved.w, moved.h, -3, -2), alA = F.shiftMap(moved.accPost, moved.w, moved.h, -3, -2);
  const B = { w: ideal.w, h: ideal.h, rect: [x0, y0, x1, y1], photons: 600, ideal: pack(ideal, ideal.donorPost, ideal.accPost), moved: pack(moved, moved.donorPost, moved.accPost), aligned: pack(moved, al, alA) };
  console.log("apbleach E: ideal", B.ideal.Ereg, "moved", B.moved.Ereg, "aligned", B.aligned.Ereg, "acc left moved", B.moved.accLeft);
  out.bleach = B;
}

// ---- ratio imaging (ratioimage) ----
{
  const s = F.sensorScene();
  const T = 20, ph = 100, off = 40;
  const f = F.sensorFrame(s, T, { photons: ph, offset: off, seed: 1 });
  // the demo's background estimate: mean of the 8 × 8 top-left corner
  const bg = (v: Float32Array) => { let t = 0; for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) t += v[y * s.w + x]; return t / 64; };
  // time course of the control cell (label 3), ratio of sums of the expected images, with and without acceptor bleaching 1 %/frame
  const course = (bl: number) => {
    const r: number[] = [];
    for (let t = 0; t < s.frames; t++) {
      const g = F.sensorFrame(s, t, { photons: ph, offset: 0, accBleach: bl, noise: false, seed: 1 });
      let A = 0, D = 0;
      for (let i = 0; i < s.w * s.h; i++) if (s.labels[i] === 3) { A += g.Aexp[i]; D += g.Dexp[i]; }
      r.push(A / D);
    }
    return r;
  };
  out.sensor = { w: s.w, h: s.h, labels: Array.from(s.labels), photons: ph, offset: off, frame: T, D: arr(f.D), A: arr(f.A), bgD: bg(f.D), bgA: bg(f.A), course: course(0), courseBleach: course(0.01) };
  console.log("sensor bg", bg(f.D), bg(f.A), "control R t0/t39 bleach", course(0.01)[0], course(0.01)[39]);
}

writeFileSync(process.argv[2], JSON.stringify(out));
console.log("wrote", process.argv[2]);
