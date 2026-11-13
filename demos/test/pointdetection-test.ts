// Tests for the point-detection day: lib/spotdetect.ts, lib/spotfit.ts and the numbers quoted
// in the units (spot-filters, spot-threshold, spot-fit, spot-crowd). Reference / brute-force
// checks. Run alone:
//   node_modules/.bin/esbuild demos/test/pointdetection-test.ts --bundle --platform=node --log-level=warning | node
import { createGrey, gaussianBlur } from "../src/lib/imageops";
import { localMaxima2D } from "../src/lib/edges";
import {
  filterKernel, filterResponseNoise, spotGain, noiseGain, spotFilter, atrous, atrousRecon, atrous1d, b3Dilated, madNoise, estimateNoise,
  spotImage, whiteNoise, noiseMaximaCount, eulerExpectedMaxima, upperTail, localMaxima1d, prominences, matchPoints, Pt, pixelFraction,
  anscombe, detectSpots,
} from "../src/lib/spotdetect";
import {
  pixelGaussian, poissonImage, fitSpots, gauss3Point, parabola3Point, centroid, thompsonVar, mortensenLSQ, mortensenMLE, crlbGaussian,
  twoGaussianMaxima, amplitudeBackgroundLSQ, gaussNewtonStep1d, selectSpotCount, peakPixel,
} from "../src/lib/spotfit";
import { rng } from "../src/lib/ui";
import { poisson } from "../src/lib/random";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}
const near = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;
const rel = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol * Math.abs(b);

// ---- matched filter SNR and noise gain ----
{
  const w = filterKernel("gauss", 1.5), g = filterResponseNoise(w.data);
  check(rel(g, 1 / (2 * Math.sqrt(Math.PI) * 1.5), 0.01), `Σw² of a Gaussian ≈ 1/(4πσ²): √ = ${g}`);
  const snr = 20 * spotGain(w, 1.5) / (10 * g);
  check(rel(snr, 20 * Math.sqrt(Math.PI * 2.25) / 10, 0.05), `matched-filter SNR ${snr} vs A√(πσ²)/σn`);
  check(rel(snr, 5.32, 0.01), `matched filter SNR at A = 20, σn = 10, σ = 1.5 is 5.3 (${snr})`);
  // no other Gaussian width does better
  for (const s of [0.5, 1, 2, 3]) { const ws = filterKernel("gauss", s); check(spotGain(ws, 1.5) / filterResponseNoise(ws.data) < spotGain(w, 1.5) / g, `σ_f = ${s} worse than matched`); }
  // B-question numbers: amplitude 30, noise 12, σ = 1
  const w1 = filterKernel("gauss", 1);
  check(rel(30 * spotGain(w1, 1) / (12 * filterResponseNoise(w1.data)), 2.5 * Math.sqrt(Math.PI), 0.01), "SNR 2.5 → 4.4 at σ = 1");
  // measured sd of filtered white noise
  for (const kind of ["gauss", "zmgauss", "log", "dog"] as const) {
    const n = whiteNoise(256, 256, 3), f = spotFilter(n, kind, 1.5, "wrap");
    let s = 0, m = 0; for (const v of f.data) m += v; m /= f.data.length; for (const v of f.data) s += (v - m) ** 2;
    const sd = Math.sqrt(s / f.data.length);
    check(rel(sd, noiseGain(kind, 1.5), 0.05), `${kind}: measured noise sd ${sd} vs √Σw² ${noiseGain(kind, 1.5)}`);
  }
}

// ---- MAD ----
{
  check(near(madNoise([3, -1, 0, 2, -2, 1, 25, -1, 0]) * 0.6744897501960817, 1, 1e-12), "MAD of the by-hand example is 1");
  const r = rng(9), x = Float64Array.from({ length: 20000 }, () => r.norm() * 7);
  for (let i = 0; i < 400; i++) x[i * 50] += 200;   // 2 % outliers ("spots")
  check(rel(madNoise(x), 7, 0.05), `MAD with 2 % outliers ${madNoise(x)} ≈ 7`);
  const s = spotImage({ size: 256, count: 60, noise: 10, background: 40, slope: 60, bump: 90, seed: 7 });
  check(rel(estimateNoise(s.image), 10, 0.05), `estimateNoise on a spot image: ${estimateNoise(s.image)}`);
}

// ---- à trous ----
{
  const spike = [0, 0, 0, 0, 16, 0, 0, 0, 0];
  const { w, a } = atrous1d(spike, 2);
  check([0, 0, 1, 4, 6, 4, 1, 0, 0].every((v, i) => near(a[0][i], v, 1e-12)), "a1 of the spike");
  check([0, 0, -1, -4, 10, -4, -1, 0, 0].every((v, i) => near(w[0][i], v, 1e-12)), "w1 of the spike");
  check(near(a[1][4], 2.75, 1e-12), `a2 at the centre = 44/16 (${a[1][4]})`);
  check(b3Dilated(2).length === 9 && b3Dilated(3).length === 17, "holes: 9 and 17 taps");
  const r = rng(4), sig = Array.from({ length: 300 }, () => r.norm() * 5 + 50), D = atrous1d(sig, 4);
  let e = 0; for (let i = 0; i < 300; i++) { let s = D.a[3][i]; for (const d of D.w) s += d[i]; e = Math.max(e, Math.abs(s - sig[i])); }
  check(e < 1e-9, `1D à trous reconstruction error ${e}`);
  const img = spotImage({ size: 64, count: 10, seed: 3 }).image, D2 = atrous(img, 4), rec = atrousRecon(D2.w, D2.a[3]);
  let e2 = 0; for (let i = 0; i < rec.data.length; i++) e2 = Math.max(e2, Math.abs(rec.data[i] - img.data[i]));
  check(e2 < 1e-3, `2D à trous reconstruction error ${e2} (float32)`);
}

// ---- false spots per image (numbers quoted in spot-threshold) ----
{
  const seeds = Array.from({ length: 20 }, (_, i) => i + 1);
  const r = noiseMaximaCount(512, 1.5, [3, 4, 5], seeds);
  check(near(r.maxima, 4862, 15), `noise maxima per 512² image ${r.maxima}`);
  check(near(r.counts[0], 99.35, 0.01) && near(r.counts[1], 4.3, 0.01) && near(r.counts[2], 0.1, 0.01), `false spots at k = 3, 4, 5: ${r.counts}`);
  check(rel(eulerExpectedMaxima(512 * 512, 1.5, 3), 123.3, 0.005) && rel(eulerExpectedMaxima(512 * 512, 1.5, 4), 4.96, 0.01), "Euler formula values");
  for (const [i, k] of [[0, 3], [1, 4]]) { const e = eulerExpectedMaxima(510 * 510, 1.5, k); check(e / r.counts[i] < 2 && r.counts[i] / e < 2, `Euler within ×2 at k = ${k}`); }
  check(r.maxima * upperTail(3) < r.counts[0] / 10, "naive tail estimate is far too optimistic");
  check(rel(upperTail(3), 0.0013499, 1e-3), "P(Z > 3)");
}

// ---- detection pipeline sanity ----
{
  const s = spotImage({ size: 256, count: 60, sigma: 1.5, amplitude: [30, 100], noise: 10, background: 40, slope: 60, bump: 90, seed: 7 });
  const d = detectSpots(s.image, { filter: "log", sigma: 1.5, radius: 2, k: 5 });
  const m = matchPoints(s.truth.map(t => [t.x, t.y] as Pt), d.dets.map(p => [p.x, p.y] as Pt), 2);
  check(m.f1 > 0.9, `LoG on an uneven background: F1 ${m.f1}`);
  const g = detectSpots(s.image, { filter: "gauss", sigma: 1.5, radius: 2, k: 5 });
  const mg = matchPoints(s.truth.map(t => [t.x, t.y] as Pt), g.dets.map(p => [p.x, p.y] as Pt), 2);
  check(mg.f1 < m.f1 - 0.2, `plain Gaussian fails on the uneven background: F1 ${mg.f1}`);
  let tot = 0; for (let i = -10; i <= 10; i++) tot += pixelFraction(i, 0.3, 1.5);
  check(near(tot, 1, 1e-6), "pixel fractions sum to 1");
}

// ---- 1D maxima and prominence ----
{
  check(localMaxima1d([0, 1, 3, 3, 1, 0], 1).join() === "2", "plateau: one pixel kept (the left one)");
  check(localMaxima1d([0, 5, 0, 4, 0], 1).join() === "1,3" && localMaxima1d([0, 5, 0, 4, 0], 2).join() === "1", "window radius 2 suppresses the neighbour");
  const p = prominences([0, 5, 2, 4, 0]);
  check(p.length === 2 && near(p[1].prominence, 2, 1e-12) && near(p[0].prominence, 5, 1e-12), "prominences 5 and 2");
  // 5×5 patch of the exercise
  const P = [[2, 3, 1, 0, 1], [4, 9, 5, 2, 0], [3, 6, 7, 6, 1], [1, 2, 4, 8, 3], [0, 1, 3, 2, 2]];
  const img = createGrey(5, 5); P.flat().forEach((v, i) => (img.data[i] = v));
  const m1 = localMaxima2D(img, -Infinity, 1, 0).map(q => `${q.y}${q.x}`).sort().join(), m2 = localMaxima2D(img, -Infinity, 2, 0).map(q => `${q.y}${q.x}`).sort().join();
  check(m1 === "11,33", `5×5 patch r = 1: ${m1}`);
  check(m2 === "11", `5×5 patch r = 2: ${m2}`);
  // the exercise patch (spot-filters B1)
  const Q = [[1, 2, 2, 1, 0], [2, 7, 3, 1, 1], [1, 3, 2, 4, 2], [0, 2, 5, 6, 3], [1, 1, 2, 3, 2]];
  const im2 = createGrey(5, 5); Q.flat().forEach((v, i) => (im2.data[i] = v));
  check(localMaxima2D(im2, -Infinity, 1, 0).map(q => `${q.y}${q.x}`).sort().join() === "11,33", "exercise patch r = 1");
  check(localMaxima2D(im2, -Infinity, 2, 0).map(q => `${q.y}${q.x}`).sort().join() === "11", "exercise patch r = 2");
  check(localMaxima1d([3, 5, 5, 2], 1).join() === "1" && localMaxima1d([3, 5, 5, 6], 1).join() === "3", "plateau vs staircase");
}

// ---- sub-pixel estimators ----
{
  for (const d of [-0.4, -0.1, 0, 0.27, 0.49]) for (const s of [0.8, 1, 2]) {
    const g = (x: number) => 37 * Math.exp(-((x - d) ** 2) / (2 * s * s));
    check(near(gauss3Point(g(-1), g(0), g(1)), d, 1e-9), `log-parabola exact (d = ${d}, s = ${s})`);
  }
  const v = [0.430, 0.956, 0.783];
  check(near((v[2] - v[0]) / (v[0] + v[1] + v[2]), 0.16, 0.005), "centroid 0.16");
  check(near(parabola3Point(v[0], v[1], v[2]), 0.25, 0.005), "parabola 0.25");
  check(near(gauss3Point(v[0], v[1], v[2]), 0.30, 0.005), "log-parabola 0.30");
  const u = v.map(x => x + 0.5);
  check(near((u[2] - u[0]) / (u[0] + u[1] + u[2]), 0.10, 0.005), "centroid with background 0.10");
  check(near(gauss3Point(u[0], u[1], u[2]), 0.28, 0.005), "log-parabola with background 0.28");
  // centroid bias formula: x̂ = Sδ/(S + (2m+1) b) (1D)
  {
    const S = 100, delta = 0.3, b = 5, m = 3, w = 2 * m + 1, img = new Float64Array(w);
    for (let i = 0; i < w; i++) img[i] = b + S * pixelFraction(i - m, delta, 0.8);
    const c = centroid(img, w, 1, m, 0, m, 0)[0] - m;
    const Sin = Array.from(img).reduce((a, x) => a + x, 0) - w * b;
    check(near(c, Sin * delta / (Sin + w * b), 2e-3), `centroid bias formula (${c})`);
  }
  // noise-free fit recovers the truth
  const mu = pixelGaussian(6.27, 5.81, 1.3, 2000, 10, 13);
  const f = fitSpots(mu, 13, 13, 1.3, [{ x: 6, y: 6, N: 1500 }], 5);
  check(near(f.spots[0].x, 6.27, 1e-5) && near(f.spots[0].y, 5.81, 1e-5) && near(f.spots[0].N, 2000, 1e-2) && near(f.B, 10, 1e-4), "LSQ fit of a noise-free spot");
  const fm = fitSpots(mu, 13, 13, 1.3, [{ x: 6, y: 6, N: 1500 }], 5, { mle: true });
  check(near(fm.spots[0].x, 6.27, 1e-5) && near(fm.spots[0].y, 5.81, 1e-5), "MLE fit of a noise-free spot");
  // linear LSQ for A, B and one Gauss–Newton step (by-hand numbers)
  const xs = [-2, -1, 0, 1, 2], gg = xs.map(x => Math.exp(-(x * x) / 2)), I = gg.map(q => 5 + 30 * q);
  const ab = amplitudeBackgroundLSQ(I, gg);
  check(near(ab.A, 30, 1e-9) && near(ab.B, 5, 1e-9), "A, B by linear least squares");
}

// ---- precision against the bounds ----
{
  const W = 13, run = (N: number, b: number, mle: boolean) => {
    const r = rng(5), e: number[] = [];
    for (let k = 0; k < 600; k++) {
      const x0 = 6 + r.unif() - 0.5, y0 = 6 + r.unif() - 0.5, img = poissonImage(pixelGaussian(x0, y0, 1, N, b, W), 1000 + k);
      const [px, py] = peakPixel(img, W);
      const f = fitSpots(img, W, W, 1, [{ x: px, y: py, N }], Math.max(0.5, b), { mle });
      e.push(f.spots[0].x - x0);
    }
    return Math.sqrt(e.reduce((s, v) => s + v * v, 0) / e.length);
  };
  const lsq = run(1000, 10, false), mle = run(1000, 10, true), th = Math.sqrt(thompsonVar(1, 1, 10, 1000));
  check(lsq < 1.5 * th && lsq > th, `LSQ sd ${lsq} within 1.5× Thompson ${th}`);
  check(mle < 1.15 * th && mle > 0.95 * Math.sqrt(crlbGaussian(1, 1000)), `MLE sd ${mle} within 1.15× Thompson ${th}`);
  check(rel(lsq, Math.sqrt(mortensenLSQ(1, 1, 10, 1000)), 0.1), "LSQ matches Mortensen's formula");
  check(rel(mle, Math.sqrt(mortensenMLE(1, 1, 10, 1000)), 0.1), "MLE matches Mortensen's formula");
  check(near(Math.sqrt(crlbGaussian(100, 100)), 10, 1e-12) && near(Math.sqrt(crlbGaussian(100, 400)), 5, 1e-12) && near(Math.sqrt(crlbGaussian(100, 2500)), 2, 1e-12), "s/√N: 10, 5, 2 nm");
  check(rel(Math.sqrt((100 ** 2 + 100 ** 2 / 12) / 100 ** 2), 1.041, 0.001), "100 nm pixels: ≈ 4 % worse");
}

// ---- crowding ----
{
  for (const r of [1.5, 1.9, 1.99]) check(twoGaussianMaxima(r * 1.5, 1.5) === 1, `one maximum at d = ${r}σ`);
  for (const r of [2.05, 2.5, 3]) check(twoGaussianMaxima(r * 1.5, 1.5) === 2, `two maxima at d = ${r}σ`);
  const se = Math.sqrt(2) * 1.5;
  check(twoGaussianMaxima(4, se) === 1 && twoGaussianMaxima(4.4, se) === 2, "after the matched filter: one maximum at 4 px, two at 4.4 px (2√2σ = 4.24)");
  check(selectSpotCount(1000, 900, 3, 2).keepTwo && !selectSpotCount(1000, 950, 3, 2).keepTwo, "penalty 3·c·σn² = 54");
}

// ---- matching ----
{
  const T: Pt[] = [[0, 0], [2.5, 0]], D: Pt[] = [[1, 0], [-2, 0]];
  const g = matchPoints(T, D, 2.5, "greedy"), o = matchPoints(T, D, 2.5, "optimal");
  check(g.tp === 1 && g.fp === 1 && g.fn === 1 && near(g.f1, 0.5, 1e-12), "greedy example: TP 1");
  check(o.tp === 2 && near(o.f1, 1, 1e-12) && near(o.rmse, Math.sqrt((1.5 ** 2 + 2 ** 2) / 2), 1e-12), `optimal example: TP 2, RMSE ${o.rmse}`);
  const T3: Pt[] = [[0, 0], [3, 0], [6, 0]], D3: Pt[] = [[1.4, 0], [4.4, 0], [-1.2, 0]];
  const g3 = matchPoints(T3, D3, 1.8, "greedy"), o3 = matchPoints(T3, D3, 1.8, "optimal");
  check(g3.tp === 2 && near(g3.rmse, 1.304, 1e-3) && o3.tp === 3 && near(o3.rmse, 1.479, 1e-3), "exercise B2 matching");
  // brute force: most matches, then smallest total distance
  const brute = (T: Pt[], D: Pt[], gate: number) => {
    let best = { n: -1, d: Infinity };
    const used = new Uint8Array(D.length);
    const rec = (i: number, n: number, d: number) => {
      if (i === T.length) { if (n > best.n || (n === best.n && d < best.d - 1e-12)) best = { n, d }; return; }
      rec(i + 1, n, d);
      for (let j = 0; j < D.length; j++) if (!used[j]) {
        const dist = Math.hypot(T[i][0] - D[j][0], T[i][1] - D[j][1]);
        if (dist <= gate) { used[j] = 1; rec(i + 1, n + 1, d + dist); used[j] = 0; }
      }
    };
    rec(0, 0, 0);
    return best;
  };
  const r = rng(77);
  for (let t = 0; t < 300; t++) {
    const nt = 1 + Math.floor(r.unif() * 7), nd = 1 + Math.floor(r.unif() * 7);
    const T2: Pt[] = Array.from({ length: nt }, () => [r.unif() * 6, r.unif() * 6]), D2: Pt[] = Array.from({ length: nd }, () => [r.unif() * 6, r.unif() * 6]);
    const gate = 0.5 + r.unif() * 2.5, m = matchPoints(T2, D2, gate, "optimal"), b = brute(T2, D2, gate);
    const tot = m.pairs.reduce((s, p) => s + p.dist, 0);
    check(m.tp === b.n && near(tot, b.d, 1e-9), `optimal = brute force (trial ${t}: ${m.tp}/${b.n}, ${tot}/${b.d})`);
    const gr = matchPoints(T2, D2, gate, "greedy");
    check(gr.tp <= m.tp, "greedy never beats optimal");
    check(new Set(m.pairs.map(p => p.t)).size === m.pairs.length && new Set(m.pairs.map(p => p.d)).size === m.pairs.length, "one-to-one");
  }
}

// ---- Poisson / Anscombe ----
{
  const r = rng(12);
  for (const lam of [5, 20, 100]) {
    let s = 0, s2 = 0; const n = 20000;
    for (let i = 0; i < n; i++) { const v = anscombe(poisson(r, lam)); s += v; s2 += v * v; }
    const sd = Math.sqrt(s2 / n - (s / n) ** 2);
    check(near(sd, 1, 0.05), `Anscombe sd at λ = ${lam}: ${sd}`);
  }
  void gaussianBlur; void gaussNewtonStep1d;
}

console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
