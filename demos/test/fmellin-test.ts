// Tests for the Fourier–Mellin part of lib/registration.ts and the fmellin demo (shape-matching
// day, unit fourier-mellin/): the similarity warp, log-polar resampling, the rotational
// correlation (by hand and against brute force), Fourier–Mellin on the micrograph (30° and
// scale 1.5 within 1° / 2 %, the 180° ambiguity, the presets that are meant to fail) and the
// known-centre tab. Run alone (from the repository root):
//   node_modules/.bin/esbuild demos/test/fmellin-test.ts --bundle --platform=node --log-level=warning | node
import {
  warpSimilarity, logPolarSpectrum, rotationalCorrelation, highpassWeight, centredMagnitude, fourierMellin, logStep,
} from "../src/lib/registration";
import { DEFAULTS, FMSettings, makePair, register, wrapDeg, knownCentre, KC_DEFAULTS, N } from "../src/fmellin/model";
import { loadCellNode } from "./selfsup-cell";
import { rng } from "../src/lib/random";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const grey = (w: number, h: number, f: (x: number, y: number) => number) => {
  const data = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data[y * w + x] = f(x, y);
  return { width: w, height: h, data };
};

// ---- the similarity warp: identity, a quarter turn clockwise, a scale ----
{
  const r = rng(3), src = grey(16, 16, () => r.norm());
  const id = warpSimilarity(src, 16, 16, { angle: 0, scale: 1, tx: 0, ty: 0 });
  check(id.every((v, i) => close(v, src.data[i], 1e-6)), "warp: identity");
  const q = warpSimilarity(src, 16, 16, { angle: Math.PI / 2, scale: 1, tx: 0, ty: 0 });
  // clockwise on screen: the content at (8 + 3, 8) (right of the centre) moves to (8, 8 + 3) (below)
  check(close(q[11 * 16 + 8], src.data[8 * 16 + 11], 1e-5), "warp: 90° turns right into down (clockwise)");
  const sh = warpSimilarity(src, 16, 16, { angle: 0, scale: 1, tx: 2, ty: -1 });
  check(close(sh[5 * 16 + 7], src.data[6 * 16 + 5], 1e-6), "warp: shift (2, −1)");
  const lin = grey(32, 32, x => x);
  const sc = warpSimilarity(lin, 32, 32, { angle: 0, scale: 2, tx: 0, ty: 0 });
  check(close(sc[16 * 32 + 20], 16 + 4 / 2, 1e-6), "warp: scale 2 halves distances to the centre in the source");
}

// ---- high-pass weight and centred magnitude ----
{
  const hp = highpassWeight(8, 8);
  check(hp[4 * 8 + 4] === 0 && hp.every(v => v >= 0), "high-pass: 0 at frequency 0, never negative");
  const wave = Array.from({ length: 64 * 64 }, (_, i) => Math.cos(2 * Math.PI * 4 * (i % 64) / 64));
  const m = centredMagnitude(wave, 64, 64);
  check(close(m[32 * 64 + 36], 2048, 1e-9) && close(m[32 * 64 + 28], 2048, 1e-9) && m[32 * 64 + 32] < 1e-9, "centred |F|: a wave of 4 cycles at ±4 from the centre");
  const r = rng(9), img = Array.from({ length: 32 * 32 }, () => r.norm());
  const rolled = img.map((_, i) => img[((Math.floor(i / 32) + 5) % 32) * 32 + ((i % 32) + 11) % 32]);
  const a = centredMagnitude(img, 32, 32), b = centredMagnitude(rolled, 32, 32);
  check(a.every((v, i) => close(v, b[i], 1e-9)), "|F| ignores a circular shift");
}

// ---- log-polar resampling of a centred spectrum ----
{
  const radius = Array.from({ length: 64 * 64 }, (_, i) => Math.hypot((i % 64) - 32, Math.floor(i / 64) - 32));
  const g = { nTheta: 8, nR: 10, rMin: 1, rMax: 30 };
  const lp = logPolarSpectrum(radius, 64, 64, g);
  let worst = 0;
  for (let i = 0; i < g.nR; i++) for (let j = 0; j < g.nTheta; j++) worst = Math.max(worst, Math.abs(lp[i * 8 + j] - Math.pow(30, i / 9)));
  check(worst < 0.25, `log-polar: row i at radius 30^(i/9) (worst ${worst.toFixed(3)})`);
  check(close(logStep(g), Math.log(30) / 9), "log step");
  const xm = Array.from({ length: 64 * 64 }, (_, i) => (i % 64) - 32);
  const lx = logPolarSpectrum(xm, 64, 64, g);
  check(close(lx[9 * 8 + 6], 30 * Math.cos(Math.PI * 6 / 8), 1e-4), "log-polar: column 6 of 8 is 135° (half a turn)");
}

// ---- rotational correlation: by hand, and against brute force ----
{
  const s = rotationalCorrelation([2, 0, -1, -1], [-1, 2, 0, -1], 4, 1, { meanFree: false });
  [-1, 6, -1, -4].forEach((v, t) => check(close(s[t], v, 1e-12), `by hand: rotational score(${t}) = ${v}`));
  const r = rng(21), nT = 16, nR = 5;
  const pa = Array.from({ length: nT * nR }, () => r.norm()), pb = Array.from({ length: nT * nR }, () => r.norm());
  const w = [1, 2, 3, 4, 5];
  const fast = rotationalCorrelation(pa, pb, nT, nR, { weights: w });
  for (let t = 0; t < nT; t++) {
    let bf = 0;
    for (let i = 0; i < nR; i++) {
      let ma = 0, mb = 0;
      for (let j = 0; j < nT; j++) { ma += pa[i * nT + j] / nT; mb += pb[i * nT + j] / nT; }
      for (let j = 0; j < nT; j++) bf += w[i] * (pa[i * nT + j] - ma) * (pb[i * nT + (j + t) % nT] - mb);
    }
    check(close(fast[t], bf, 1e-9), `rotational correlation = brute force, t = ${t}`);
  }
}

// ---- by-hand bin counts of the exercise ----
check(close(30 * 256 / 180, 42.666666666, 1e-6), "30° = 42.7 θ bins (256 over 180°)");
check(close(Math.log(2) / (Math.log(128) / 64), 9.142857, 1e-5), "scale 2 = 9.1 log-r bins (64 over r = 1…128)");

// ---- Fourier–Mellin on the micrograph ----
const cell = loadCellNode();
const img = { width: cell.width, height: cell.height, data: Float32Array.from(cell.data) };
const run = (p: Partial<FMSettings>) => { const s = { ...DEFAULTS, ...p } as FMSettings; return { s, r: register(makePair(img, s), s) }; };
const good = (s: FMSettings, r: ReturnType<typeof register>) =>
  Math.abs(wrapDeg(r.best.angle * 180 / Math.PI - s.angle)) < 1 && Math.abs(r.scale / s.scale - 1) < 0.02 && Math.hypot(r.best.tx - s.tx, r.best.ty - s.ty) < 1;
{
  const { s, r } = run({ angle: 30, scale: 1.5, tx: 12, ty: -8 });
  check(Math.abs(wrapDeg(r.best.angle * 180 / Math.PI - 30)) < 1, `FM: 30° within 1° (${(r.best.angle * 180 / Math.PI).toFixed(2)})`);
  check(Math.abs(r.scale / 1.5 - 1) < 0.02, `FM: scale 1.5 within 2 % (${r.scale.toFixed(4)})`);
  check(good(s, r), `FM: shift (12, −8) within 1 px (${r.best.tx.toFixed(2)}, ${r.best.ty.toFixed(2)})`);
}
// the same images through the library call directly (no demo model), with a zoom out
{
  const a = warpSimilarity(img, N, N, { angle: 0, scale: 1, tx: 0, ty: 0 }, { cx: 120, cy: 130, border: "mirror" });
  const b = warpSimilarity(img, N, N, { angle: -0.9, scale: 0.75, tx: -6, ty: 9 }, { cx: 120, cy: 130, border: "mirror" });
  const r = fourierMellin(a, b, N, N);
  check(Math.abs(r.best.angle + 0.9) < Math.PI / 180 && Math.abs(r.scale / 0.75 - 1) < 0.02 && Math.hypot(r.best.tx + 6, r.best.ty - 9) < 1,
    `FM: −51.6°, scale 0.75, shift (−6, 9) (${(r.best.angle * 180 / Math.PI).toFixed(2)}°, ${r.scale.toFixed(3)}, (${r.best.tx.toFixed(2)}, ${r.best.ty.toFixed(2)}))`);
}
{
  let bad = 0;
  for (let a = -180; a < 180; a += 15) { const { s, r } = run({ angle: a, tx: 7, ty: -4 }); if (!good(s, r)) bad++; }
  check(bad === 0, `FM: every angle from −180° in steps of 15° (${bad} wrong)`);
}
{
  const amb = run({ angle: 200, tx: 10, ty: 5, both: false });
  check(!good(amb.s, amb.r) && Math.abs(wrapDeg(amb.r.best.angle * 180 / Math.PI - 20)) < 1, "preset 180° ambiguity: one candidate gives 20°");
  const both = run({ angle: 200, tx: 10, ty: 5 });
  check(good(both.s, both.r), "both candidates: 200° (= −160°) and the shift");
  check(both.r.candidates[1].height > 5 * both.r.candidates[0].height, "the right candidate has by far the higher shift peak");
}
{
  const nt = run({ shade: 400, taper: false });
  check(!good(nt.s, nt.r) && Math.abs(nt.r.thetaBins) < 1 && Math.abs(nt.r.rhoBins) < 1, "preset no taper: the cross puts the peak at (0, 0)");
  const t = run({ shade: 400 });
  check(good(t.s, t.r), "uneven light with the taper: right");
}
{
  const s3 = run({ angle: 20, scale: 3 });
  check(!good(s3.s, s3.r), "preset scale 3: out of range (wrong)");
  const s18 = run({ angle: 20, scale: 1.8 });
  check(good(s18.s, s18.r), "scale 1.8: still right");
}
{
  const n = run({ noise: 25 });
  check(!good(n.s, n.r), "preset noise 25 with high-pass: wrong");
  const nh = run({ noise: 25, highpass: false });
  check(good(nh.s, nh.r), "noise 25 without high-pass: right");
}
// ---- known centre ----
for (const a of [50, -20, 135, 180]) {
  const k = knownCentre(img, { ...KC_DEFAULTS, angle: a });
  check(Math.abs(wrapDeg(k.angle - a)) < 360 / KC_DEFAULTS.nTheta, `known centre: ${a}° (found ${k.angle.toFixed(2)})`);
}

console.log(`fmellin: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
