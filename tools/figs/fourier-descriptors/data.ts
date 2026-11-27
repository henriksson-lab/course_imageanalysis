// Data for the fourier-descriptors/ figures, computed with lib/contour.ts and the fdmatch
// demo's own model, so the numbers on the slides match the demo. Called by figs.py:
//   esbuild data.ts --bundle --platform=node | node - <out.json>
import { writeFileSync } from "fs";
import {
  Pt, Coeffs, descriptors, coef, evalSeries, epicycles, reconstruct, restart, rotateOutline, mirrorTraced,
  translateOutline, scaleOutline, reverse, mirror, firstEllipse, ellipticDescriptors, resampleArcLength,
  resampleByIndex, matchOutlines, sampleMean,
} from "../../../demos/src/lib/contour";
import { compute, DEFAULTS, PRESETS, Settings, SHAPES, sampleSource, rasterTrace, misfit } from "../../../demos/src/fdmatch/model";

const [outPath] = process.argv.slice(2);
const out: Record<string, unknown> = {};
const preset = (id: string, extra: Partial<Settings> = {}) => ({ ...DEFAULTS, ...PRESETS.find(p => p.id === id)!.s, ...extra } as Settings);
const cs = (c: Coeffs, K: number) => {
  const o: Record<string, [number, number]> = {};
  for (let k = -K; k <= K; k++) o[k] = coef(c, k);
  return o;
};
const poly = (id: keyof typeof SHAPES): Pt[] => {
  const s = SHAPES[id].src;
  if (s.kind !== "poly") throw new Error("poly");
  return s.poly;
};

// an outline is a periodic signal: the leaf at 16 samples
{
  const z = resampleArcLength(poly("leaf"), 16), dense = resampleArcLength(poly("leaf"), 400);
  out.signal = { z, dense };
}

// sampling a traced outline: index vs arc length (a small pixel nucleus)
{
  const r = rasterTrace(poly("nucleus"), 9);
  out.sampling = { mask: Array.from(r.mask), w: r.w, trace: r.trace, idx: resampleByIndex(r.trace, 24), arc: resampleArcLength(r.trace, 24) };
}

// epicycles: the leaf with K = 1, 2, 3
{
  const c = descriptors(resampleArcLength(poly("leaf"), 64));
  out.epi = {
    curves: [1, 2, 3, 32].map(K => reconstruct(c, K, 300)),
    chain: epicycles(c, 3, 0.15),
    coeffs: cs(c, 3),
  };
}

// the rules: six versions of the leaf (16 samples) and their descriptors
{
  const z = resampleArcLength(poly("leaf"), 16), c0 = descriptors(z);
  const versions: Record<string, Pt[]> = {
    original: z,
    moved: translateOutline(z, 0.6, 0.3),
    scaled: scaleOutline(z, 0.6),
    turned: rotateOutline(z, Math.PI / 4),
    restarted: restart(z, 3),
    reversed: reverse(z),
    mirrored: mirror(z),
  };
  out.rules = Object.fromEntries(Object.entries(versions).map(([k, v]) => [k, { z: v, c: cs(descriptors(v), 3) }]));
  out.rules_c0 = cs(c0, 3);
}

// by hand: (2, i, −2, −i), turned by 90°, started one later
{
  const e: Pt[] = [[2, 0], [0, 1], [-2, 0], [0, -1]];
  const v = { original: e, turned: rotateOutline(e, Math.PI / 2), restarted: restart(e, 1) };
  out.rotstart = Object.fromEntries(Object.entries(v).map(([k, z]) => [k, { z, c: cs(descriptors(z), 2) }]));
  const r1 = compute(preset("rotstart")), r2 = compute(preset("rotstart", { start: 0, angle: 90 }));
  out.rotstart_demo = { start: cs(r1.cB, 2), turned: cs(r2.cB, 2) };
}

// a square sampled at its corners, at 8 and at 16 points
{
  out.square = [4, 8, 16].map(n => {
    const r = compute(preset("square", { n, K: n / 2 }));
    return { n, z: r.A.z, curve: reconstruct(r.cA, n / 2, 400), c: cs(r.cA, Math.min(3, n / 2)) };
  });
}

// same magnitudes, different shape
{
  const r = compute(preset("samemag"));
  out.samemag = { a: reconstruct(r.cA, 32, 400), b: reconstruct(r.cB, 32, 400), ca: cs(r.cA, 4), cb: cs(r.cB, 4), magDist: r.magDist, score: r.match.score, misfit: misfit(r.match.score), m: r.match.m };
}

// matching two leaves (the demo's first preset)
{
  const r = compute(preset("leaf"));
  out.match = {
    a: r.A.z, b: r.B.z, aligned: r.match.aligned, scores: Array.from(r.match.scores), m: r.match.m,
    angle: r.match.angle * 180 / Math.PI, scale: r.match.scale, score: r.match.score, misfit: misfit(r.match.score),
    second: (() => { const s = Array.from(r.match.scores); let b = -1, bi = 0; s.forEach((v, i) => { if (Math.min(Math.abs(i - r.match.m), 64 - Math.abs(i - r.match.m)) > 6 && v > b) { b = v; bi = i; } }); return [bi, b]; })(),
  };
  const r0 = compute(preset("leaf", { noise: 0 }));
  out.match_clean = { m: r0.match.m, angle: r0.match.angle * 180 / Math.PI, scale: r0.match.scale, score: r0.match.score };
}

// mirror image of the hand
{
  const no = compute(preset("mirror")), yes = compute(preset("mirror", { allowMirror: true }));
  out.mirror = {
    a: no.A.z, b: no.B.z, alignedNo: no.match.aligned, alignedYes: yes.match.aligned,
    scores: Array.from(yes.match.scores), scoresMirror: Array.from(yes.match.scoresMirror!),
    no: { m: no.match.m, angle: no.match.angle * 180 / Math.PI, score: no.match.score, misfit: misfit(no.match.score) },
    yes: { m: yes.match.m, angle: yes.match.angle * 180 / Math.PI, scale: yes.match.scale, score: yes.match.score, misfit: misfit(yes.match.score) },
    magDist: no.magDist,
  };
}

// z(t) vs r(θ): the hand is not star-shaped
{
  const z = resampleArcLength(poly("hand"), 400);
  out.star = { z, centre: sampleMean(z) };
}

// elliptic descriptors of the nucleus
{
  const z = resampleArcLength(poly("nucleus"), 64), c = descriptors(z), e = firstEllipse(c);
  const ell = reconstruct({ re: Float64Array.from(c.re, (v, j) => (j === 1 || j === 63 ? v : 0)), im: Float64Array.from(c.im, (v, j) => (j === 1 || j === 63 ? v : 0)), n: 64 }, 1, 200)
    .map(p => [p[0] + c.re[0], p[1] + c.im[0]] as Pt);
  const efd = ellipticDescriptors(c, 4);
  // the start point on the first ellipse: parameter θ0
  const startPt: Pt = (() => { const [r1, i1] = coef(c, 1), [rm, im] = coef(c, -1), t = e.theta0; return [c.re[0] + r1 * Math.cos(t) - i1 * Math.sin(t) + rm * Math.cos(-t) - im * Math.sin(-t), c.im[0] + r1 * Math.sin(t) + i1 * Math.cos(t) + rm * Math.sin(-t) + im * Math.cos(-t)]; })();
  out.efd = { z: resampleArcLength(poly("nucleus"), 400), ellipse: ell, e, efd, startPt, centre: [c.re[0], c.im[0]] };
}

// staircase: |c_k| of the pixel nucleus by index and by arc length, against the polygon itself
{
  const st0 = compute(preset("staircase")), st1 = compute(preset("staircase", { arclen: true }));
  const exact = compute(preset("staircase", { pixels: false, arclen: true }));
  const mags = (c: Coeffs) => Array.from({ length: 65 }, (_, i) => Math.hypot(...coef(c, i - 32)) / Math.hypot(...coef(c, 1)));
  out.staircase = {
    index: mags(st0.cA), arc: mags(st1.cA), exact: mags(exact.cA),
    scoreIndex: st0.match.score, scoreArc: st1.match.score, misfitIndex: misfit(st0.match.score), misfitArc: misfit(st1.match.score),
    angleIndex: st0.match.angle * 180 / Math.PI, angleArc: st1.match.angle * 180 / Math.PI, scaleIndex: st0.match.scale, scaleArc: st1.match.scale,
    nA: st0.A.path.length, nB: st0.B.path.length,
  };
}

// nuclei: the pair table
{
  const r = compute(preset("nuclei"));
  out.nuclei = { pairs: r.pairs, outlines: r.nuclei!.map(s => s.path) };
  const rm = compute(preset("nuclei", { allowMirror: true }));
  out.nuclei_mirror = rm.pairs;
}

writeFileSync(outPath, JSON.stringify(out));
for (const k of ["match", "match_clean", "mirror", "samemag", "staircase"]) {
  const v = out[k] as Record<string, unknown>;
  console.error(k, JSON.stringify(Object.fromEntries(Object.entries(v).filter(([, x]) => typeof x === "number" || (Array.isArray(x) && x.length < 3) || (typeof x === "object" && x && !Array.isArray(x))))));
}
console.error("square", JSON.stringify((out.square as { n: number; c: unknown }[]).map(s => [s.n, s.c])));
console.error("rotstart_demo", JSON.stringify(out.rotstart_demo));
console.error("efd", JSON.stringify((out.efd as { efd: unknown; e: unknown }).efd), JSON.stringify((out.efd as { e: unknown }).e));
console.error("nuclei", JSON.stringify((out.nuclei as { pairs: unknown }).pairs));
void [sampleSource, mirrorTraced, matchOutlines, evalSeries];
