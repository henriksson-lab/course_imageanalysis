// Data for the fourier-mellin/ figures and the numbers quoted on the pages, computed with the
// fmellin demo's own model (lib/registration.ts), so figures, pages and demo agree.
// Called by figs.py:   esbuild data.ts --bundle --platform=node | node - <out.json>
// Run from the repository root (the micrograph is read from demos/src/fourier2d/cell256.png).
import { writeFileSync } from "fs";
import { loadCellNode } from "../../../demos/test/selfsup-cell";
import { DEFAULTS, FMSettings, makePair, register, knownCentre, KC_DEFAULTS, N, wrapDeg } from "../../../demos/src/fmellin/model";
import { correlate2d, findPeaks, peakRatio, warpSimilarity, centredMagnitude, rotationalCorrelation, logStep } from "../../../demos/src/lib/registration";
import { toPolar } from "../../../demos/src/lib/polar";
import { rng } from "../../../demos/src/lib/random";

const [outPath] = process.argv.slice(2);
const c = loadCellNode();
const img = { width: c.width, height: c.height, data: Float32Array.from(c.data) };
const A = (x: ArrayLike<number>) => Array.from(x, v => Math.round(v * 1000) / 1000);
const deg = (r: number) => wrapDeg(r * 180 / Math.PI);
const out: Record<string, unknown> = {};

// ---- the numbers of every preset (and the variations the exercise asks for) ----
const cases: [string, Partial<FMSettings>][] = [
  ["rot30", { angle: 30 }],
  ["rot30_nohp", { angle: 30, highpass: false }],
  ["scale", { angle: 0, scale: 1.5, tx: 12, ty: -8 }],
  ["rot30_scale", { angle: 30, scale: 1.5, tx: 12, ty: -8 }],
  ["amb_one", { angle: 200, tx: 10, ty: 5, both: false }],
  ["amb_both", { angle: 200, tx: 10, ty: 5 }],
  ["notaper", { shade: 400, taper: false }],
  ["notaper_taper", { shade: 400 }],
  ["notaper_noshade", { taper: false }],
  ["scale3", { angle: 20, scale: 3 }],
  ["scale18", { angle: 20, scale: 1.8 }],
  ["scale2", { angle: 20, scale: 2 }],
  ["scale07", { angle: 20, scale: 0.7 }],
  ["scale06", { angle: 20, scale: 0.6 }],
  ["noise", { noise: 25 }],
  ["noise_nohp", { noise: 25, highpass: false }],
  ["noise15", { noise: 15 }],
  ["noise40_nohp", { noise: 40, highpass: false }],
  ["shift_only", { angle: 0, tx: 15, ty: 10 }],
];
const summary: Record<string, unknown> = {};
for (const [id, p] of cases) {
  const s = { ...DEFAULTS, ...p } as FMSettings;
  const t0 = Date.now();
  const r = register(makePair(img, s), s);
  const b = r.best;
  const ok = Math.abs(wrapDeg(deg(b.angle) - s.angle)) < 1 && Math.abs(r.scale / s.scale - 1) < 0.02 && Math.hypot(b.tx - s.tx, b.ty - s.ty) < 1;
  summary[id] = {
    thetaBins: r.thetaBins, rhoBins: r.rhoBins, height: r.peakHeight, ratio: r.peakRatio, angleMod180: deg(r.angle), scale: r.scale,
    candidates: r.candidates.map(q => [deg(q.angle), q.height]), found: [deg(b.angle), r.scale, b.tx, b.ty], ok,
    trueBins: [s.angle * 256 / 180, -Math.log(s.scale) / logStep(r.grid)],
  };
  console.error(id.padEnd(16), ok ? "ok   " : "WRONG", `θ ${r.thetaBins.toFixed(2)} ρ ${r.rhoBins.toFixed(2)} h ${r.peakHeight.toFixed(3)} ratio ${r.peakRatio.toFixed(2)}`,
    `| ${deg(b.angle).toFixed(2)}° × ${r.scale.toFixed(3)} (${b.tx.toFixed(2)}, ${b.ty.toFixed(2)})`,
    `| cands ${r.candidates.map(q => `${deg(q.angle).toFixed(1)}:${q.height.toFixed(3)}`).join(" ")} | ${Date.now() - t0} ms`);
}
out.summary = summary;
// scale range sweep (angle 20)
const sweep: [number, boolean][] = [];
for (const sc of [0.5, 0.55, 0.6, 0.65, 0.7, 1.5, 1.7, 1.8, 1.9, 2, 2.2, 2.5, 3]) {
  const s = { ...DEFAULTS, angle: 20, scale: sc };
  const r = register(makePair(img, s), s);
  sweep.push([sc, Math.abs(wrapDeg(deg(r.best.angle) - 20)) < 1 && Math.abs(r.scale / sc - 1) < 0.02 && Math.hypot(r.best.tx, r.best.ty) < 1]);
}
console.error("scale sweep", sweep.map(([s, ok]) => `${s}:${ok ? "ok" : "--"}`).join(" "));
out.sweep = sweep;

// ---- the problem: plain phase correlation of a rotated view ----
{
  const s = { ...DEFAULTS, angle: 30, tx: 8, ty: -5 };
  const p = makePair(img, s);
  const pc = correlate2d(p.a, p.b, N, N, { p: 1, taper: true, eps: 1e-6 });
  const same = { ...DEFAULTS, angle: 0, tx: 8, ty: -5 };
  const q = makePair(img, same);
  const pc0 = correlate2d(q.a, q.b, N, N, { p: 1, taper: true, eps: 1e-6 });
  const pk = findPeaks(pc, 2), pk0 = findPeaks(pc0, 2);
  out.problem = { a: A(p.a), b: A(p.b), surface: A(pc.data), peak: [pk[0].tx, pk[0].ty, pk[0].value], ratio: peakRatio(pc),
    surface0: A(pc0.data), peak0: [pk0[0].tx, pk0[0].ty, pk0[0].value], ratio0: peakRatio(pc0) };
  console.error("problem: rotated 30°: peak", pk[0].tx, pk[0].ty, pk[0].value.toFixed(3), "ratio", peakRatio(pc).toFixed(2),
    "| not rotated: peak", pk0[0].tx, pk0[0].ty, pk0[0].value.toFixed(3), "ratio", peakRatio(pc0).toFixed(2));
}

// ---- known centre ----
{
  const k = knownCentre(img, KC_DEFAULTS);
  let second = -Infinity;
  const best = Math.round(k.angle * KC_DEFAULTS.nTheta / 360);
  for (let t = 0; t < KC_DEFAULTS.nTheta; t++) {
    const d = Math.min(Math.abs(t - best), KC_DEFAULTS.nTheta - Math.abs(t - best));
    if (d > 3) second = Math.max(second, k.score[t]);
  }
  out.known = { a: A(k.a), b: A(k.b), size: k.size, pa: A(k.pa), pb: A(k.pb), nR: k.nR, nTheta: KC_DEFAULTS.nTheta, score: A(k.score.map(v => v / k.height)),
    angle: k.angle, trueAngle: KC_DEFAULTS.angle, second: second / k.height };
  console.error("known centre: found", k.angle.toFixed(2), "true", KC_DEFAULTS.angle, "second", (second / k.height).toFixed(2));
  for (const noise of [40, 80]) {
    const kn = knownCentre(img, { ...KC_DEFAULTS, noise });
    console.error(`known centre, noise ${noise}: found ${kn.angle.toFixed(2)}`);
  }
  for (const ang of [-20, 135, 180]) console.error(`known centre ${ang}: found ${knownCentre(img, { ...KC_DEFAULTS, angle: ang }).angle.toFixed(2)}`);
}

// ---- magnitude spectra of a rotated and shifted view ----
{
  const s = { ...DEFAULTS, angle: 30, tx: 15, ty: 10 };
  const p = makePair(img, s);
  const ma = centredMagnitude(p.a, N, N, { taper: true }), mb = centredMagnitude(p.b, N, N, { taper: true });
  out.magnitude = { a: A(p.a), b: A(p.b), ma: A(ma.map(v => Math.log1p(v))), mb: A(mb.map(v => Math.log1p(v))) };
}

// ---- the whole pipeline (30°, scale 1.5, shift (12, −8)) ----
{
  const s = { ...DEFAULTS, angle: 30, scale: 1.5, tx: 12, ty: -8 };
  const p = makePair(img, s), r = register(p, s);
  const pk = findPeaks(r.best.surface, 1)[0];
  const aligned = warpSimilarity({ width: N, height: N, data: Float32Array.from(r.best.unwarped) }, N, N, { angle: 0, scale: 1, tx: pk.tx, ty: pk.ty });
  out.pipeline = {
    a: A(p.a), b: A(p.b), ma: A(r.magA.map(v => Math.log1p(v))), mb: A(r.magB.map(v => Math.log1p(v))),
    lpA: A(r.lpA.map(v => Math.log1p(v))), lpB: A(r.lpB.map(v => Math.log1p(v))), nTheta: r.grid.nTheta, nR: r.grid.nR,
    surface: A(r.surface.data), peak: [r.thetaBins, r.rhoBins, r.peakHeight], trueBins: [30 * 256 / 180, -Math.log(1.5) / logStep(r.grid)],
    aligned: A(aligned), found: [deg(r.best.angle), r.scale, r.best.tx, r.best.ty],
  };
}

// ---- the 180° ambiguity: the two candidates' shift surfaces ----
{
  const s = { ...DEFAULTS, angle: 200, tx: 10, ty: 5 };
  const p = makePair(img, s), r = register(p, s);
  out.ambiguity = { b: A(p.b), cands: r.candidates.map(q => ({ angle: deg(q.angle), height: q.height, unwarped: A(q.unwarped), surface: A(q.surface.data) })), a: A(p.a) };
}

// ---- the cross: no taper with uneven light ----
{
  const mk = (taper: boolean) => {
    const s = { ...DEFAULTS, shade: 400, taper };
    const p = makePair(img, s), r = register(p, s);
    return { a: A(p.a), ma: A(r.magA.map(v => Math.log1p(v))), lpA: A(r.lpA.map(v => Math.log1p(v))), surface: A(r.surface.data), peak: [r.thetaBins, r.rhoBins, r.peakHeight] };
  };
  out.cross = { off: mk(false), on: mk(true), trueBins: [30 * 256 / 180, 0] };
}

// ---- log-polar sampling points (for the "few samples at small radius" picture) ----
{
  const g = { nTheta: 256, nR: 128, rMin: 2, rMax: 62 };
  out.grid = g;
  out.gridStep = logStep(g);
}

// ---- averaging noisy rotated copies (cryo-EM style) with and without rotational alignment ----
{
  const R = 28, size = 2 * R + 2, nT = 128, n = 16, rr = rng(5);
  const ref = warpSimilarity(img, size, size, { angle: 0, scale: 1, tx: 0, ty: 0 }, { cx: 110, cy: 75, border: "mirror" });
  const copies: Float64Array[] = [], angles: number[] = [];
  for (let k = 0; k < n; k++) {
    const ang = k === 0 ? 0 : 2 * Math.PI * rr.unif();
    const v = warpSimilarity(img, size, size, { angle: ang, scale: 1, tx: 0, ty: 0 }, { cx: 110, cy: 75, border: "mirror" });
    for (let i = 0; i < v.length; i++) v[i] += 70 * rr.norm();
    copies.push(v); angles.push(ang);
  }
  const g = (d: Float64Array) => ({ width: size, height: size, data: Float32Array.from(d) });
  const o = { cx: size / 2, cy: size / 2, rMin: 1, rMax: R, nR: R, nTheta: nT };
  const polar0 = toPolar(g(copies[0]), o).data;
  const weights = Array.from({ length: R }, (_, i) => 1 + (R - 1) * i / (R - 1));
  const plain = new Float64Array(size * size), aligned = new Float64Array(size * size);
  const errs: number[] = [];
  for (let k = 0; k < n; k++) {
    const pk = toPolar(g(copies[k]), o).data;
    const sc = rotationalCorrelation(polar0, pk, nT, R, { weights });
    let best = 0; for (let t = 1; t < nT; t++) if (sc[t] > sc[best]) best = t;
    const est = best * 2 * Math.PI / nT;
    errs.push(Math.abs(wrapDeg((est - angles[k]) * 180 / Math.PI)));
    const back = warpSimilarity(g(copies[k]), size, size, { angle: -est, scale: 1, tx: 0, ty: 0 });
    for (let i = 0; i < plain.length; i++) { plain[i] += copies[k][i] / n; aligned[i] += back[i] / n; }
  }
  out.average = { size, R, ref: A(ref), copies: copies.slice(0, 6).map(A), plain: A(plain), aligned: A(aligned), errs };
  console.error("averaging: angle errors (deg) max", Math.max(...errs).toFixed(2), "mean", (errs.reduce((s, v) => s + v, 0) / n).toFixed(2));
}

writeFileSync(outPath, JSON.stringify(out));
