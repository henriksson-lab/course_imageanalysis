// Tests for lib/contour.ts and the fdmatch demo model (shape-matching day, unit
// fourier-descriptors/): the by-hand values, the five invariance rules on random outlines,
// matching recovers a known rotation, scale and starting point (and a mirror image), boundary
// tracing, arc-length resampling, the elliptic descriptors. Run alone:
//   node_modules/.bin/esbuild demos/test/fdmatch-test.ts --bundle --platform=node --log-level=warning | node
import {
  Pt, Coeffs, descriptors, coef, samplesFromCoeffs, evalSeries, epicycles, reconstruct, translateOutline, scaleOutline,
  rotateOutline, restart, reverse, mirror, mirrorTraced, matchOutlines, matchScores, magnitudeDistance,
  magnitudeSignature, traceBoundary, signedArea, resampleArcLength, resampleByIndex, polygonLength, firstEllipse,
  ellipticDescriptors, similarity, fourierOutline,
} from "../src/lib/contour";
import { compute, DEFAULTS, PRESETS, Settings, SHELL_A, SHELL_B, misfit } from "../src/fdmatch/model";
import { rng } from "../src/lib/random";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}
const near = (a: number, b: number, tol = 1e-10) => Math.abs(a - b) <= tol;
const cnear = (c: Coeffs, k: number, re: number, im: number, tol = 1e-12) => {
  const [r, i] = coef(c, k);
  return near(r, re, tol) && near(i, im, tol);
};
const r = rng(5);
const randomOutline = (n: number): Pt[] => Array.from({ length: n }, (_, i) => {
  const t = 2 * Math.PI * i / n, rad = 1 + 0.3 * r.norm() * 0.3;
  return [rad * Math.cos(t) + 0.05 * r.norm(), rad * Math.sin(t) + 0.05 * r.norm()] as Pt;
});

// ---- by hand ----
{
  // (a) z = (1, i, −1, −i): only c_1 = 1
  const c = descriptors([[1, 0], [0, 1], [-1, 0], [0, -1]]);
  check(cnear(c, 1, 1, 0) && cnear(c, 0, 0, 0) && cnear(c, -1, 0, 0) && cnear(c, 2, 0, 0), "(a) square corners: c_1 = 1, others 0");
  // the series through the four corners is the unit circle between the samples
  const p = evalSeries(c, 2, 1 / 8);
  check(near(p[0], Math.SQRT1_2) && near(p[1], Math.SQRT1_2), "(a) between the corners: on the unit circle");
  // (b) z = (2, i, −2, −i): c_1 = 3/2, c_−1 = 1/2; semi-axes 2 and 1
  const z: Pt[] = [[2, 0], [0, 1], [-2, 0], [0, -1]], cb = descriptors(z);
  check(cnear(cb, 1, 1.5, 0) && cnear(cb, -1, 0.5, 0) && cnear(cb, 0, 0, 0) && cnear(cb, 2, 0, 0), "(b) c_1 = 3/2, c_−1 = 1/2");
  const e = firstEllipse(cb);
  check(near(e.a, 2) && near(e.b, 1), "(b) semi-axes 2 and 1");
  // (c) rotate by 90° (× i) vs start one point later
  const rot = descriptors(rotateOutline(z, Math.PI / 2)), st = descriptors(restart(z, 1));
  check(cnear(rot, 1, 0, 1.5) && cnear(rot, -1, 0, 0.5), "(c) rotation: c_1 → 3/2 i, c_−1 → 1/2 i");
  check(cnear(st, 1, 0, 1.5) && cnear(st, -1, 0, -0.5), "(c) new start: c_1 → 3/2 i, c_−1 → −1/2 i");
  // ... and the same through the demo model (preset "rotation vs new starting point")
  const pr = compute({ ...DEFAULTS, ...PRESETS.find(q => q.id === "rotstart")!.s } as Settings);
  check(cnear(pr.cA, 1, 1.5, 0, 1e-9) && cnear(pr.cA, -1, 0.5, 0, 1e-9), "demo rotstart: A = (2, i, −2, −i)");
  check(cnear(pr.cB, 1, 0, 1.5, 1e-9) && cnear(pr.cB, -1, 0, -0.5, 1e-9), "demo rotstart: B started one later");
  const pr2 = compute({ ...DEFAULTS, ...PRESETS.find(q => q.id === "rotstart")!.s, start: 0, angle: 90 } as Settings);
  check(cnear(pr2.cB, 1, 0, 1.5, 1e-9) && cnear(pr2.cB, -1, 0, 0.5, 1e-9), "demo rotstart: B turned by 90°");
  const sq = compute({ ...DEFAULTS, ...PRESETS.find(q => q.id === "square")!.s } as Settings);
  check(cnear(sq.cA, 1, 1, 0, 1e-9) && cnear(sq.cA, -1, 0, 0, 1e-9) && cnear(sq.cA, 2, 0, 0, 1e-9), "demo square: only c_1 = 1");
  // c_1 c_−1: same for a new start, turns by 2α for a rotation
  const prod = (c: Coeffs): [number, number] => { const [a, b] = coef(c, 1), [p, q] = coef(c, -1); return [a * p - b * q, a * q + b * p]; };
  check(near(prod(st)[0], 0.75) && near(prod(st)[1], 0), "c_1 c_−1 unchanged by a new start");
  check(near(prod(rot)[0], -0.75) && near(prod(rot)[1], 0), "c_1 c_−1 turns by 2·90° under rotation");
}

// ---- the five rules on random outlines (error < 1e-10) ----
for (const n of [16, 12, 64]) {
  const z = randomOutline(n), c = descriptors(z);
  const tr = descriptors(translateOutline(z, 3, -2)), sc = descriptors(scaleOutline(z, 1.7));
  const al = 0.7, ro = descriptors(rotateOutline(z, al)), m = 5, rs = descriptors(restart(z, m));
  const rv = descriptors(reverse(z)), mi = descriptors(mirror(z)), mt = descriptors(mirrorTraced(z));
  let e = 0;
  for (let k = -Math.floor((n - 1) / 2); k <= Math.floor((n - 1) / 2); k++) {
    const [a, b] = coef(c, k);
    const want = (x: Coeffs, re: number, im: number) => { const [p, q] = coef(x, k); e = Math.max(e, Math.abs(p - re), Math.abs(q - im)); };
    want(tr, k === 0 ? a + 3 : a, k === 0 ? b - 2 : b);
    want(sc, 1.7 * a, 1.7 * b);
    want(ro, Math.cos(al) * a - Math.sin(al) * b, Math.sin(al) * a + Math.cos(al) * b);
    const ph = 2 * Math.PI * k * m / n;
    want(rs, Math.cos(ph) * a - Math.sin(ph) * b, Math.sin(ph) * a + Math.cos(ph) * b);
    const [a2, b2] = coef(c, -k);
    want(rv, a2, b2);
    want(mi, a2, -b2);   // mirror: conjugate and swap
    want(mt, a, -b);     // mirror walked clockwise again: conjugate
  }
  check(e < 1e-10, `five rules, n=${n}: max error ${e}`);
  // round trip and reconstruction with all coefficients
  const back = samplesFromCoeffs(c);
  check(back.every((p, i) => near(p[0], z[i][0]) && near(p[1], z[i][1])), `inverse, n=${n}`);
  const rec = reconstruct(c, n, n);
  check(rec.every((p, i) => near(p[0], z[i][0], 1e-9) && near(p[1], z[i][1], 1e-9)), `series through the samples, n=${n}`);
  // epicycles end at the series value
  const ep = epicycles(c, 3, 0.37), ev = evalSeries(c, 3, 0.37), tip = ep[ep.length - 1].tip;
  check(near(tip[0], ev[0]) && near(tip[1], ev[1]), `epicycle tip = series, n=${n}`);
  // magnitudes: invariant to everything but the direction
  const sig = magnitudeSignature(c, 5);
  for (const [name, x] of [["scale", sc], ["rotation", ro], ["start", rs], ["mirror (traced)", mt]] as [string, Coeffs][])
    check(magnitudeDistance(c, x, 5) < 1e-10, `magnitudes unchanged by ${name}, n=${n}`);
  check(sig.length === 10, "signature length 2K");
}

// ---- matching: known rotation, scale and starting point are recovered exactly ----
{
  const z = randomOutline(64);
  for (const [s, deg, m] of [[1.3, 40, 20], [0.6, -120, 3], [2, 179, 63]]) {
    const b = translateOutline(similarity(restart(z, m), s, deg * Math.PI / 180), 5, 7);
    const res = matchOutlines(z, b);
    check(res.m === m && near(res.scale, s, 1e-9) && near(res.angle, deg * Math.PI / 180, 1e-9) && near(res.score, 1, 1e-12),
      `match s=${s} α=${deg} m=${m}: got m=${res.m} s=${res.scale} α=${res.angle * 180 / Math.PI} score=${res.score}`);
    check(res.aligned.every((p, i) => near(p[0], b[i][0], 1e-9) && near(p[1], b[i][1], 1e-9)), "aligned A = B");
    // the FFT score equals the direct least-squares residual for every m
    const cA = descriptors(z), cB = descriptors(b), ms = matchScores(cA, cB);
    let worst = 0;
    for (let mm = 0; mm < 64; mm += 7) {
      // direct: λ* = Σ conj(A_{n+mm}−ā)(B_n−b̄) / Σ|A−ā|², residual / Σ|B − b̄|²
      const A = restart(z, mm), ma = [cA.re[0], cA.im[0]], mb = [cB.re[0], cB.im[0]];
      let nr = 0, ni = 0, ea = 0, eb = 0;
      for (let i = 0; i < 64; i++) {
        const ax = A[i][0] - ma[0], ay = A[i][1] - ma[1], bx = b[i][0] - mb[0], by = b[i][1] - mb[1];
        nr += ax * bx + ay * by; ni += ax * by - ay * bx; ea += ax * ax + ay * ay; eb += bx * bx + by * by;
      }
      const direct = Math.hypot(nr, ni) / Math.sqrt(ea * eb);
      worst = Math.max(worst, Math.abs(direct - ms.score[mm]));
    }
    check(worst < 1e-12, `FFT score = direct sum over the samples (${worst})`);
  }
  // mirror image: poor without the mirror option, perfect with it (an asymmetric outline:
  // a circle with a bump at 0° and a smaller one at 100°)
  const za: Pt[] = Array.from({ length: 64 }, (_, i) => {
    const t = 2 * Math.PI * i / 64, rad = 1 + 0.5 * Math.exp(-(((t + Math.PI) % (2 * Math.PI) - Math.PI) ** 2) / 0.1) + 0.3 * Math.exp(-((t - 1.75) ** 2) / 0.1);
    return [rad * Math.cos(t), rad * Math.sin(t)] as Pt;
  });
  const b = similarity(restart(mirrorTraced(za), 9), 1.1, 0.5);
  const no = matchOutlines(za, b), yes = matchOutlines(za, b, { mirror: true });
  check(!no.mirrored && misfit(no.score) > 0.03, `mirror not allowed: score ${no.score}`);
  check(yes.mirrored && near(yes.score, 1, 1e-12) && near(yes.scale, 1.1, 1e-9), `mirror allowed: score ${yes.score}`);
}

// ---- same magnitudes, different shape ----
{
  const res = compute({ ...DEFAULTS, ...PRESETS.find(q => q.id === "samemag")!.s } as Settings);
  check(res.magDist < 1e-12, `samemag: magnitude distance ${res.magDist}`);
  check(res.match.score < 0.99, `samemag: complex match score ${res.match.score}`);
  check(SHELL_A.length === SHELL_B.length, "shells");
  // exercise B8(d): J = c_3 c_−1 conj(c_1)² / |c_1|⁴ is +0.03 for S, −0.03 for S′, and invariant
  const J = (c: Coeffs) => {
    const [a, b] = coef(c, 3), [p, q] = coef(c, -1), [u, v] = coef(c, 1);
    const xr = a * p - b * q, xi = a * q + b * p, cr = u * u - v * v, ci = -2 * u * v;   // conj(c1)² = (u − iv)²
    const m4 = (u * u + v * v) ** 2;
    return [(xr * cr - xi * ci) / m4, (xr * ci + xi * cr) / m4];
  };
  const jA = J(res.cA), jB = J(res.cB);
  check(near(jA[0], 0.03, 1e-9) && near(jB[0], -0.03, 1e-9) && near(jA[1], 0, 1e-9), `J: ${jA} ${jB}`);
  const moved = descriptors(similarity(restart(res.A.z, 13), 1.7, 0.9));
  const jm = J(moved);
  check(near(jm[0], 0.03, 1e-9) && near(jm[1], 0, 1e-9), `J invariant: ${jm}`);
}

// ---- boundary tracing and resampling ----
{
  // a 3 × 2 rectangle of pixels: boundary clockwise, 6 points... (all pixels are on the boundary)
  const w = 5, h = 4, m = new Uint8Array(w * h);
  for (let y = 1; y <= 2; y++) for (let x = 1; x <= 3; x++) m[y * w + x] = 1;
  const t = traceBoundary(m, w, h);
  check(t.length === 6 && t[0][0] === 1 && t[0][1] === 1 && t[1][0] === 2 && t[1][1] === 1, `rectangle trace ${JSON.stringify(t)}`);
  check(signedArea(t) > 0, "traced clockwise on screen: positive area");
  // a disc: perimeter of the pixel-centre polygon and arc-length samples are evenly spaced
  const W = 41, d = new Uint8Array(W * W);
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) d[y * W + x] = Math.hypot(x - 20, y - 20) <= 15 ? 1 : 0;
  const td = traceBoundary(d, W, W);
  const pts = new Set(td.map(p => p[0] + "," + p[1]));
  check(pts.size === td.length, "disc trace visits each boundary pixel once");
  const rs = resampleArcLength(td, 64), L = polygonLength(td);
  let dev = 0;
  for (let i = 0; i < 64; i++) {
    // the samples lie on the polygon; the gaps along the polygon are L/64 (check via a fine resampling)
    dev = Math.max(dev, Math.abs(Math.hypot(rs[(i + 1) % 64][0] - rs[i][0], rs[(i + 1) % 64][1] - rs[i][1]) - L / 64));
  }
  check(dev < 0.35, `arc-length samples about L/N apart (dev ${dev})`);
  const ri = resampleByIndex(td, td.length);
  check(ri.every((p, i) => near(p[0], td[i][0]) && near(p[1], td[i][1])), "index resampling with N = vertex count returns the vertices");
  check(near(resampleArcLength([[0, 0], [4, 0], [4, 4], [0, 4]], 8, 0.25)[0][0], 4) && near(resampleArcLength([[0, 0], [4, 0], [4, 4], [0, 4]], 8, 0.25)[0][1], 0), "offset 1/4 starts at the second corner");
}

// ---- elliptic descriptors ----
{
  const z = randomOutline(64), c = descriptors(z);
  const raw = ellipticDescriptors(c, 5, true), nrm = ellipticDescriptors(c, 5);
  // raw EFD rebuild x and y
  const t = 0.3, th = 2 * Math.PI * t;
  let x = c.re[0], y = c.im[0];
  for (let k = 0; k < 5; k++) { x += raw.a[k] * Math.cos((k + 1) * th) + raw.b[k] * Math.sin((k + 1) * th); y += raw.c[k] * Math.cos((k + 1) * th) + raw.d[k] * Math.sin((k + 1) * th); }
  const ev = evalSeries(c, 5, t);
  check(near(x, ev[0], 1e-9) && near(y, ev[1], 1e-9), "elliptic descriptors rebuild the same series");
  check(near(nrm.a[0], 1, 1e-9) && near(nrm.b[0], 0, 1e-9) && near(nrm.c[0], 0, 1e-9), "normalised: a1 = 1, b1 = c1 = 0");
  // invariant to rotation, scale and starting point
  const z2 = similarity(restart(z, 11), 0.8, 1.1), n2 = ellipticDescriptors(descriptors(z2), 5);
  let e = 0;
  for (let k = 0; k < 5; k++) e = Math.max(e, Math.abs(n2.a[k] - nrm.a[k]), Math.abs(n2.b[k] - nrm.b[k]), Math.abs(n2.c[k] - nrm.c[k]), Math.abs(n2.d[k] - nrm.d[k]));
  // the normalisation is defined up to 180° (both ends of the major axis): accept either sign pattern
  let e2 = 0;
  for (let k = 0; k < 5; k++) { const sg = k % 2 === 0 ? 1 : -1; e2 = Math.max(e2, Math.abs(sg * n2.a[k] - nrm.a[k]), Math.abs(sg * n2.b[k] - nrm.b[k]), Math.abs(sg * n2.c[k] - nrm.c[k]), Math.abs(sg * n2.d[k] - nrm.d[k])); }
  check(Math.min(e, e2) < 1e-9, `normalised EFD invariant (up to the 180° ambiguity): ${Math.min(e, e2)}`);
}

// ---- fourierOutline (moved from moments/features.ts) still agrees ----
{
  const z = randomOutline(32), fo = fourierOutline(z, 16), rec = reconstruct(descriptors(z), 16, 32);
  check(fo.smooth.every((p, i) => near(p[0], rec[i][0], 1e-9) && near(p[1], rec[i][1], 1e-9)), "fourierOutline = reconstruct with all coefficients");
}

// ---- demo presets: numbers quoted on the pages ----
{
  const run = (id: string, extra: Partial<Settings> = {}) => compute({ ...DEFAULTS, ...PRESETS.find(q => q.id === id)!.s, ...extra } as Settings);
  const leaf = run("leaf");
  check(leaf.match.m === 20 && Math.abs(leaf.match.angle * 180 / Math.PI - 40) < 1 && Math.abs(leaf.match.scale - 1.2) < 0.02, `leaf preset: m=${leaf.match.m} α=${leaf.match.angle * 180 / Math.PI} s=${leaf.match.scale}`);
  const st0 = run("staircase"), st1 = run("staircase", { arclen: true });
  check(st1.match.score > st0.match.score, `staircase: arc length improves the score (${st0.match.score} → ${st1.match.score})`);
  const mir = run("mirror"), mir2 = run("mirror", { allowMirror: true });
  check(mir.match.score < 0.99 && mir2.match.mirrored && mir2.match.score > 0.999, `mirror preset: ${mir.match.score} → ${mir2.match.score}`);
  check(mir.magDist < 1e-9, "mirror image: same magnitudes");
  const nuc = run("nuclei");
  const best = nuc.pairs!.reduce((a, b) => (b.score > a.score ? b : a));
  check(best.i === 2 && best.j === 5, `nuclei: best pair ${best.i}–${best.j}`);
  check(misfit(1) === 0, "misfit");
  if (process && (globalThis as { VERBOSE?: boolean }).VERBOSE) console.log(nuc.pairs);
}

console.log(`fdmatch: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
