// Tests for lib/registration.ts (FFT registration day): FFT correlation against brute force,
// phase correlation of circular shifts, brightness invariance, masked NCC against direct NCC,
// template matching, the four candidates, sub-pixel estimates, Fourier shifts and the global
// stitching solver (the by-hand three-tile example). Run alone:
//   node_modules/.bin/esbuild demos/test/fftregistration-test.ts --bundle --platform=node --log-level=warning | node
import {
  correlate1d, bruteForce1d, overlapScores1d, correlate2d, bruteCircular2d, at, maskedNcc, overlapScores2d,
  TemplateMatcher, findPeaks, fourCandidates, parabolaOffset, upsampledPeak1d, upsampledPeak2d, fourierShift1d,
  fourierShift2d, solveGlobal, maxSpanningTree, chainPositions, hann, signed, idftAt, peakRatio, type TileEdge,
} from "../src/lib/registration";
import { simulate1d, PROFILE } from "../src/xcorr1d/signals";
import { makeTiles, registerPairs } from "../src/stitch/model";
import { sweep } from "../src/subpixel/model";
import { makeFrames, registerFrames } from "../src/drift/model";
import { rng } from "../src/lib/random";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const r = rng(12);
const randArr = (n: number) => Float64Array.from({ length: n }, () => r.norm());

// ---- 1D: FFT correlation = brute force (circular and, with padding, linear) ----
for (const n of [4, 8, 7, 32]) {
  const a = randArr(n), b = randArr(n);
  const fft = correlate1d(a, b).c, bf = bruteForce1d(a, b, { circular: true });
  for (let t = 0; t < n; t++) check(close(fft[t], bf.cc[t], 1e-10), `1D circular n=${n} t=${t}`);
  const lin = correlate1d(a, b, { pad: true }), bl = bruteForce1d(a, b);
  bl.t.forEach((t, k) => check(close(lin.c[((t % lin.n) + lin.n) % lin.n], bl.cc[k], 1e-10), `1D linear n=${n} t=${t}`));
}
// by hand: A = (0,1,2,0), B = (1,2,0,0): c = (2,5,2,0), A conj(B) = (9, -5i, -1, 5i)
{
  const res = correlate1d([0, 1, 2, 0], [1, 2, 0, 0]);
  [2, 5, 2, 0].forEach((v, t) => check(close(res.c[t], v, 1e-12), `by-hand c(${t})`));
  check(close(res.R.re[0], 9) && close(res.R.im[1], -5) && close(res.R.re[2], -1) && close(res.R.im[3], 5) && Math.abs(res.R.re[1]) < 1e-12, "by-hand cross-power");
  const ph = correlate1d([0, 1, 2, 0], [1, 2, 0, 0], { p: 1, eps: 0 });
  [0, 1, 0, 0].forEach((v, t) => check(close(ph.c[t], v, 1e-12), `by-hand phase c(${t})`));
  const ph2 = correlate1d([0, 1, 2, 0], [3, 5, 1, 1], { p: 1, eps: 0 });   // B' = 2B + 1
  [0, 1, 0, 0].forEach((v, t) => check(close(ph2.c[t], v, 1e-12), `by-hand phase with gain/offset c(${t})`));
}
// template example of unit 1: t = (1,3,1) over s = (0,1,3,1,0,2,6,2)
{
  const s = [0, 1, 3, 1, 0, 2, 6, 2], tp = [1, 3, 1];
  const sc = [0, 1, 2, 3, 4, 5].map(u => overlapScores1d(s, tp, u));
  check(sc[1].cc === 11 && sc[5].cc === 22, "template CC 11 and 22");
  check(sc[1].ssd === 0 && sc[5].ssd === 11, "template SSD 0 and 11");
  check(close(sc[1].ncc, 1) && close(sc[5].ncc, 1), "template NCC 1 and 1");
}

// ---- 2D: FFT = brute force; phase correlation finds circular shifts exactly ----
{
  const w = 8, h = 4, a = randArr(w * h), b = randArr(w * h);
  const s = correlate2d(a, b, w, h, { meanFree: false }), bf = bruteCircular2d(a, b, w, h);
  for (let i = 0; i < w * h; i++) check(close(s.data[i], bf[i], 1e-10), `2D circular i=${i}`);
  const sp = correlate2d(a, b, w, h, { meanFree: false, pad: true });
  for (let ty = -h + 1; ty < h; ty++) for (let tx = -w + 1; tx < w; tx++)
    check(close(at(sp, tx, ty), overlapScores2d(a, b, w, h, tx, ty).cc, 1e-10), `2D linear (${tx},${ty})`);
}
{
  const w = 32, h = 32, a = randArr(w * h);
  for (const [tx, ty] of [[3, -5], [-10, 7], [16, 0], [0, 0]]) {
    const b = new Float64Array(w * h);   // B(x) = A(x + t)
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) b[y * w + x] = a[((y + ty + h) % h) * w + (x + tx + w) % w];
    const s = correlate2d(a, b, w, h, { p: 1, eps: 1e-9, meanFree: false });
    const pk = findPeaks(s, 2)[0];
    check(pk.tx === signed(((tx % w) + w) % w, w) && pk.ty === ty, `phase corr finds (${tx},${ty}): got (${pk.tx},${pk.ty})`);
    check(close(pk.value, 1, 1e-6), `phase corr peak height 1 (${pk.value})`);
    // gain and offset on B do not change the phase correlation surface (mean removed)
    const b2 = b.map(v => 3 * v + 50);
    const s2 = correlate2d(a, b2, w, h, { p: 1, eps: 1e-9 }), s1 = correlate2d(a, b, w, h, { p: 1, eps: 1e-9 });
    let md = 0;
    for (let i = 0; i < w * h; i++) md = Math.max(md, Math.abs(s2.data[i] - s1.data[i]));
    check(md < 1e-6, `gain/offset invariance (${md})`);
  }
}

// ---- masked NCC = direct overlap NCC; template matcher ----
{
  const w = 8, h = 8, a = randArr(w * h), b = randArr(w * h);
  const m = maskedNcc(a, w, h, b, w, h, 4);
  for (let ty = -h + 1; ty < h; ty++) for (let tx = -w + 1; tx < w; tx++) {
    const d = overlapScores2d(a, b, w, h, tx, ty), v = at(m, tx, ty);
    if (d.n >= 4) check(close(v, d.ncc, 1e-7), `masked NCC (${tx},${ty}) ${v} vs ${d.ncc}`);
    else check(Number.isNaN(v), `masked NCC NaN for small overlap`);
  }
  const W = 32, H = 32, img = randArr(W * H), tw = 5, th = 4, u0 = 11, v0 = 20;
  const T = new Float64Array(tw * th);
  for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) T[y * tw + x] = 2 * img[(v0 + y) * W + u0 + x] + 7;
  const map = new TemplateMatcher(img, W, H, tw, th).match(T);
  let best = -Infinity, bi = -1;
  map.forEach((v, i) => { if (v > best) { best = v; bi = i; } });
  check(bi === v0 * W + u0 && close(best, 1, 1e-7), `template matcher finds (${u0},${v0})`);
  // direct check of one value
  const u = 3, v = 5, n = tw * th;
  let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) { const p = img[(v + y) * W + u + x], q = T[y * tw + x]; sa += p; sb += q; saa += p * p; sbb += q * q; sab += p * q; }
  const direct = (sab - sa * sb / n) / Math.sqrt((saa - sa * sa / n) * (sbb - sb * sb / n));
  check(close(map[v * W + u], direct, 1e-7), "template matcher value");
}

// ---- four candidates; peak ratio ----
{
  const w = 32, h = 32, big = 64, img = randArr(big * big);
  const ox = 5, oy = 3, tx = 20, ty = -6;   // B's frame at (ox + tx, oy + ty + 10) relative to A's at (ox, oy + 10)
  const A = new Float64Array(w * h), B = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    A[y * w + x] = img[(oy + 10 + y) * big + ox + x];
    B[y * w + x] = img[(oy + 10 + ty + y) * big + ox + tx + x];
  }
  const s = correlate2d(A, B, w, h, { p: 1 });
  const pk = findPeaks(s, 1)[0];
  const c = fourCandidates(A, B, w, h, pk.ix, pk.iy, 20);
  check(c[0].tx === tx && c[0].ty === ty, `four candidates pick (${tx},${ty}): got (${c[0].tx},${c[0].ty})`);
  check(c.length === 4, "four candidates");
  check(peakRatio(s) > 1.5, `peak ratio of a good pair > 1.5 (${peakRatio(s).toFixed(2)})`);
}

// ---- sub-pixel ----
check(close(parabolaOffset(0.5, 0.9, 0.7), 1 / 6, 1e-12), "parabola by hand = 1/6");
{
  const n = 64, a = Float64Array.from({ length: n }, (_, x) => Math.exp(-((x - 30) ** 2) / 18) + 0.6 * Math.exp(-((x - 40) ** 2) / 8));
  for (const s of [0.3, 2.75, -1.4]) {
    const b = fourierShift1d(a, -s);   // B(x) = A(x + s)
    const res = correlate1d(a, b, { p: 1, eps: 1e-12 });
    let t0 = 0;
    res.c.forEach((v, i) => { if (v > res.c[t0]) t0 = i; });
    const up = upsampledPeak1d(res.R, signed(t0, n), 100);
    check(Math.abs(up.x - s) < 0.02, `upsampled DFT 1D ${s}: ${up.x}`);
    check(close(idftAt(res.R, signed(t0, n)), res.c[t0], 1e-9), "idftAt at an integer = inverse DFT");
  }
  // Fourier shift by an integer = circular shift
  const sh = fourierShift1d(a, 3);
  for (let x = 0; x < n; x++) check(close(sh[x], a[(x - 3 + n) % n], 1e-9), "fourierShift1d integer");
  // 2D
  const w = 32, img = Float64Array.from({ length: w * w }, (_, i) => { const x = i % w, y = (i - x) / w; return Math.exp(-((x - 14) ** 2 + (y - 17) ** 2) / 20) + 0.5 * Math.exp(-((x - 20) ** 2 + (y - 9) ** 2) / 6); });
  const sx = 1.3, sy = -2.6, b2 = fourierShift2d(img, w, w, -sx, -sy);
  const cs = correlate2d(img, b2, w, w, { p: 1, eps: 1e-12, meanFree: false });
  const pk = findPeaks(cs, 1)[0];
  const up2 = upsampledPeak2d(cs.R, w, w, pk.tx, pk.ty, 50);
  check(Math.abs(up2.tx - sx) < 0.05 && Math.abs(up2.ty - sy) < 0.05, `upsampled DFT 2D (${up2.tx}, ${up2.ty})`);
}

// ---- stitching solver: by-hand three tiles ----
{
  const e: TileEdge[] = [{ i: 0, j: 1, dx: 100, dy: 0, w: 1 }, { i: 1, j: 2, dx: 102, dy: 0, w: 1 }, { i: 0, j: 2, dx: 199, dy: 0, w: 1 }];
  const s = solveGlobal(3, e)!;
  check(close(s.x[1], 99, 1e-12) && close(s.x[2], 200, 1e-12), `three tiles: x2 = ${s.x[1]}, x3 = ${s.x[2]}`);
  check(close(s.rx[0], -1) && close(s.rx[1], -1) && close(s.rx[2], 1), "three tiles residuals");
  const tree = maxSpanningTree(3, e.map((x, k) => ({ ...x, w: [3, 2, 1][k] })));
  const ch = chainPositions(3, e, tree);
  check(close(ch.x[2], 202), "chain along the tree");
  check(solveGlobal(3, [e[0]]) === null, "disconnected graph -> null");
}
check(close(hann(5)[2], 1) && hann(5)[0] === 0, "hann");

// ---- the demos' own pipelines ----
{
  const sim = simulate1d({ ...PROFILE_DEFAULT() });
  check(sim.f.length > 0 && sim.g.length > 0, "xcorr1d simulate");
}
{
  const tiles = makeTiles({ seed: 3, noise: 4, empty: -1, badPair: -1 }, null);
  const pr = registerPairs(tiles);
  const g = solveGlobal(9, pr.edges.map(p => p.edge))!;
  let maxErr = 0;
  for (let k = 0; k < 9; k++) maxErr = Math.max(maxErr, Math.hypot(g.x[k] - (tiles.pos[k][0] - tiles.pos[0][0]), g.y[k] - (tiles.pos[k][1] - tiles.pos[0][1])));
  check(maxErr < 1.5, `stitch: global positions within 1.5 px (${maxErr.toFixed(2)})`);
}
{
  const sw = sweep({ signal: "bumps", noise: 0, U: 20, method: "phase", steps: 21 });
  const maxUp = Math.max(...sw.up.map(Math.abs)), maxPar = Math.max(...sw.parabola.map(Math.abs));
  check(maxUp < 0.05, `subpixel sweep: upsampled DFT error < 0.05 (${maxUp.toFixed(3)})`);
  check(maxPar > maxUp, "subpixel sweep: parabola worse than upsampled DFT");
}
{
  const fr = makeFrames({ seed: 5, n: 12, noise: 5, bleach: 0.5, step: 0.6 }, null);
  const reg = registerFrames(fr, "first");
  let maxErr = 0;
  for (let k = 0; k < fr.n; k++) maxErr = Math.max(maxErr, Math.hypot(reg.x[k] - fr.dx[k], reg.y[k] - fr.dy[k]));
  check(maxErr < 0.5, `drift: to-first errors < 0.5 px (${maxErr.toFixed(3)})`);
}

function PROFILE_DEFAULT() { return PROFILE; }

console.log(`fftregistration: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
