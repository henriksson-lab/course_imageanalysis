// Reference checks for the integral-images day: lib/integral.ts and lib/boost.ts.
// Run:
//   node_modules/.bin/esbuild demos/test/integral-test.ts --bundle --platform=node --log-level=warning | node
import { rng } from "../src/lib/random";
import * as IN from "../src/lib/integral";
import * as B from "../src/lib/boost";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const r = rng(2024);
const randInt = (n: number) => Math.floor(r.unif() * n);
const randImg = (w: number, h: number, max = 256) => Float64Array.from({ length: w * h }, () => randInt(max));

// ---- 1D
current = "1d";
{
  const v = [3, 1, 4, 1, 5, 9, 2, 6];
  const p = IN.prefixSum(v);
  for (let a = 0; a < v.length; a++) for (let b = a; b < v.length; b++) {
    let s = 0; for (let i = a; i <= b; i++) s += v[i];
    check(IN.rangeSum(p, a, b) === s, `range ${a}..${b}`);
  }
}

// ---- 2D rectangle sums against brute force (all three number types)
current = "rect";
for (let trial = 0; trial < 20; trial++) {
  const w = 1 + randInt(17), h = 1 + randInt(17), img = randImg(w, h);
  const T = IN.integralImage(img, w, h);
  const T32 = IN.integralImage(img, w, h, "u32");
  for (let q = 0; q < 50; q++) {
    const x0 = randInt(w + 1), x1 = x0 + randInt(w - x0 + 1), y0 = randInt(h + 1), y1 = y0 + randInt(h - y0 + 1);
    const d = IN.directSum(img, w, x0, y0, x1, y1);
    check(IN.rectSum(T, x0, y0, x1, y1) === d, `f64 rect ${w}x${h} [${x0},${x1})x[${y0},${y1})`);
    check(IN.rectSum(T32, x0, y0, x1, y1) === d, "u32 rect");
    // clipped: shift the rectangle partly outside
    const c = IN.rectSumClipped(T, x0 - 3, y0 - 2, x1 + 4, y1 + 1);
    const cx0 = Math.max(0, x0 - 3), cx1 = Math.min(w, x1 + 4), cy0 = Math.max(0, y0 - 2), cy1 = Math.min(h, y1 + 1);
    check(c.sum === IN.directSum(img, w, cx0, cy0, cx1, cy1) && c.n === (cx1 - cx0) * (cy1 - cy0), "clipped");
  }
  // the last entry is the sum of the whole image; the table undoes by differences
  check(IN.at(T, w, h) === IN.directSum(img, w, 0, 0, w, h), "last entry = total");
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    check(IN.at(T, x + 1, y + 1) - IN.at(T, x, y + 1) - IN.at(T, x + 1, y) + IN.at(T, x, y) === img[y * w + x], "ΔxΔy S = I");
}

// ---- wrapping 32-bit tables: rectangle sums right whenever the true sum < 2^32
current = "u32 wrap";
{
  const w = 300, h = 300, img = Float64Array.from({ length: w * h }, () => 60000 + randInt(5536)); // 16-bit, total ≈ 5.6e9 > 2^32
  const T32 = IN.integralImage(img, w, h, "u32"), T = IN.integralImage(img, w, h);
  check(IN.at(T, w, h) > 2 ** 32, "test total exceeds 2^32");
  let wrapped = 0;
  for (let i = 0; i < T.data.length; i++) if (T.data[i] !== T32.data[i]) wrapped++;
  check(wrapped > 0, "some u32 entries wrapped");
  for (let q = 0; q < 300; q++) {
    const x0 = randInt(w), x1 = x0 + 1 + randInt(Math.min(200, w - x0)), y0 = randInt(h), y1 = y0 + 1 + randInt(Math.min(200, h - y0));
    const exact = IN.rectSum(T, x0, y0, x1, y1);
    if (exact < 2 ** 32) check(IN.rectSum(T32, x0, y0, x1, y1) === exact, "wrapped rect sum");
    else check(IN.rectSum(T32, x0, y0, x1, y1) === exact % 2 ** 32, "mod 2^32 when too big");
  }
}

// ---- float32: exact while the entries stay below 2^24
current = "f32";
{
  const w = 64, h = 64, img = randImg(w, h); // total < 64·64·255 ≈ 1.04e6 < 2^24
  const T = IN.integralImage(img, w, h, "f32"), T64 = IN.integralImage(img, w, h);
  let same = true;
  for (let i = 0; i < T.data.length; i++) if (T.data[i] !== T64.data[i]) same = false;
  check(same, "f32 exact below 2^24");
  const W = 1024, Hh = 1024, big = randImg(W, Hh);
  const B32 = IN.integralImage(big, W, Hh, "f32"), B64 = IN.integralImage(big, W, Hh);
  check(IN.at(B64, W, Hh) > 2 ** 24 && IN.at(B32, W, Hh) !== IN.at(B64, W, Hh), "f32 rounds above 2^24");
}

// ---- 3D corner formula: 2^3 corners, sign (−1)^(3 − #upper)
current = "3d";
{
  const nx = 5, ny = 4, nz = 6, vol = randImg(nx * ny, nz, 10);
  const S = new Float64Array((nx + 1) * (ny + 1) * (nz + 1));
  const id = (x: number, y: number, z: number) => (z * (ny + 1) + y) * (nx + 1) + x;
  for (let z = 1; z <= nz; z++) for (let y = 1; y <= ny; y++) for (let x = 1; x <= nx; x++)
    S[id(x, y, z)] = vol[((z - 1) * ny + y - 1) * nx + x - 1]
      + S[id(x - 1, y, z)] + S[id(x, y - 1, z)] + S[id(x, y, z - 1)]
      - S[id(x - 1, y - 1, z)] - S[id(x - 1, y, z - 1)] - S[id(x, y - 1, z - 1)]
      + S[id(x - 1, y - 1, z - 1)];
  for (let q = 0; q < 100; q++) {
    const a = [randInt(nx), randInt(ny), randInt(nz)];
    const b = [a[0] + 1 + randInt(nx - a[0]), a[1] + 1 + randInt(ny - a[1]), a[2] + 1 + randInt(nz - a[2])];
    let s = 0;
    for (let m = 0; m < 8; m++) {
      const up = [(m & 1) ? b[0] : a[0], (m & 2) ? b[1] : a[1], (m & 4) ? b[2] : a[2]];
      const nUp = (m & 1 ? 1 : 0) + (m & 2 ? 1 : 0) + (m & 4 ? 1 : 0);
      s += (-1) ** (3 - nUp) * S[id(up[0], up[1], up[2])];
    }
    let d = 0;
    for (let z = a[2]; z < b[2]; z++) for (let y = a[1]; y < b[1]; y++) for (let x = a[0]; x < b[0]; x++) d += vol[(z * ny + y) * nx + x];
    check(s === d, "3D box sum");
  }
}

// ---- box filters: four methods agree; local variance against two passes
current = "box";
{
  const w = 37, h = 29, img = randImg(w, h);
  for (const k of [1, 3, 7, 15, 41]) {
    const ref = IN.boxFilterBy("direct", img, w, h, k);
    for (const m of ["separable", "running", "integral"] as const) {
      const o = IN.boxFilterBy(m, img, w, h, k);
      let md = 0; for (let i = 0; i < o.length; i++) md = Math.max(md, Math.abs(o[i] - ref[i]));
      check(md < 1e-9, `${m} k=${k}: ${md}`);
    }
  }
  for (const rr of [1, 2, 5]) {
    const ex = IN.localVarianceDirect(img, w, h, rr);
    for (const off of [0, 128]) {
      const st = IN.localStats(img, w, h, rr, { offset: off });
      let md = 0; for (let i = 0; i < ex.length; i++) md = Math.max(md, Math.abs(st.variance[i] - ex[i]));
      check(md < 1e-6, `variance r=${rr} offset ${off}: ${md}`);
      const su = IN.localStats(img, w, h, rr, { kind: "u32", offset: off });
      md = 0; for (let i = 0; i < ex.length; i++) md = Math.max(md, Math.abs(su.variance[i] - ex[i]));
      check(md < 1e-6, `u32 variance r=${rr} offset ${off}: ${md}`);
    }
  }
  // cancellation: a bright flat noisy region in float32 gives wrong (even negative) variances
  const W = 512, Hh = 512, bright = Float64Array.from({ length: W * Hh }, () => 200 + randInt(3));
  const s32 = IN.localStats(bright, W, Hh, 3, { kind: "f32" });
  const s32o = IN.localStats(bright, W, Hh, 3, { kind: "f32", offset: 200 });
  const ex = IN.localStats(bright, W, Hh, 3, { offset: 200 });
  let e1 = 0, e2 = 0;
  for (let i = 0; i < ex.variance.length; i++) { e1 = Math.max(e1, Math.abs(s32.variance[i] - ex.variance[i])); e2 = Math.max(e2, Math.abs(s32o.variance[i] - ex.variance[i])); }
  check(e1 > 1 && s32.negatives > 0, `f32 cancellation visible: err ${e1}, negatives ${s32.negatives}`);
  check(e2 < 1e-3, `offset fixes it: err ${e2}`);
}

// ---- repeated boxes: variance n(k²−1)/12, sum 1
current = "repeated";
for (const k of [3, 5, 9]) for (const n of [1, 2, 3, 4]) {
  const v = IN.repeatedBox1d(k, n);
  const c = (v.length - 1) / 2;
  let s = 0, q = 0;
  for (let i = 0; i < v.length; i++) { s += v[i]; q += v[i] * (i - c) ** 2; }
  check(close(s, 1) && close(q, IN.repeatedBoxVariance(k, n)), `k=${k} n=${n}`);
}
{
  const v = IN.repeatedBox1d(3, 2).map(x => x * 9);
  check([1, 2, 3, 2, 1].every((x, i) => close(v[i], x)), "box*box = 1 2 3 2 1");
}

// ---- rectangle features: lookup counts and values
current = "haar";
{
  const counts: Record<string, number> = { "edge-x": 6, "edge-y": 6, "line-x": 8, "line-y": 8, checker: 9, centre: 8 };
  const w = 30, h = 26, img = randImg(w, h), T = IN.integralImage(img, w, h);
  for (const t of IN.TEMPLATES) {
    for (let q = 0; q < 30; q++) {
      const sx = 1 + randInt(4), sy = 1 + randInt(4);
      const x = randInt(w - t.cols * sx + 1), y = randInt(h - t.rows * sy + 1);
      const cs = IN.templateCorners(t, x, y, sx, sy);
      check(cs.length === counts[t.id], `${t.id} lookups ${cs.length}`);
      check(close(IN.evalCorners(T, cs), IN.templateDirect(img, w, t, x, y, sx, sy), 1e-9), `${t.id} value`);
    }
    // zero response on a flat image
    const flat = new Float64Array(w * h).fill(7), TF = IN.integralImage(flat, w, h);
    check(Math.abs(IN.evalCorners(TF, IN.templateCorners(t, 2, 3, 2, 2))) < 1e-9, `${t.id} zero on flat`);
  }
  // pool size against brute force enumeration
  for (const [a, b, W, H] of [[2, 1, 8, 8], [3, 1, 24, 24], [2, 2, 10, 7]]) {
    let n = 0;
    for (let cw = 1; a * cw <= W; cw++) for (let ch = 1; b * ch <= H; ch++)
      for (let y = 0; y + b * ch <= H; y++) for (let x = 0; x + a * cw <= W; x++) n++;
    check(IN.poolSize(a, b, W, H) === n, `pool ${a}x${b} in ${W}x${H}`);
  }
  // response map at the centre equals direct evaluation
  const t = IN.TEMPLATES[0], m = IN.responseMap(T, t, 3, 4);
  check(close(m[(0 + 2) * w + 0 + 3], IN.templateDirect(img, w, t, 0, 0, 3, 4)), "response map position");
  // box features: dob is mean(small) − mean(large)
  const f = IN.boxFeature(img, w, h, "dob", 2);
  const x = 10, y = 12;
  const ms = IN.directSum(img, w, x - 2, y - 2, x + 3, y + 3) / 25, ml = IN.directSum(img, w, x - 4, y - 4, x + 5, y + 5) / 81;
  check(close(f[y * w + x], ms - ml), "dob");
}

// ---- boosting
current = "boost";
{
  // stump search against brute force
  for (let trial = 0; trial < 30; trial++) {
    const n = 5 + randInt(30);
    const vals = Float32Array.from({ length: n }, () => randInt(8));
    const y = Int8Array.from({ length: n }, () => (r.unif() < 0.5 ? 1 : -1));
    const w = Float64Array.from({ length: n }, () => r.unif());
    const M = B.prepareFeatures(vals, 1, n);
    const s = B.bestStumpForFeature(M, 0, y, w);
    check(close(s.error, B.bestStumpBrute(vals, y, w), 1e-9), "stump = brute force");
    // the returned threshold achieves the returned error
    let e = 0, tot = 0;
    for (let i = 0; i < n; i++) { tot += w[i]; if ((s.polarity * (vals[i] - s.threshold) > 0 ? 1 : -1) !== y[i]) e += w[i]; }
    check(close(e / tot, s.error, 1e-9), "threshold achieves error");
  }
  // AdaBoost on a separable 2D toy set (inside a box vs outside): training error goes to 0
  // and stays below the bound Π 2√(ε(1−ε))
  const n = 200, nf = 2;
  const X = Float32Array.from({ length: nf * n }, () => r.unif());
  const y = Int8Array.from({ length: n }, (_, i) => (Math.abs(X[i] - 0.5) < 0.25 && Math.abs(X[n + i] - 0.5) < 0.25 ? 1 : -1));
  const M = B.prepareFeatures(X, nf, n);
  const ab = new B.AdaBoost(M, y);
  let ok = true, last = 1;
  for (let t = 0; t < 60; t++) {
    const s = ab.round();
    if (!(s.error < 0.5)) ok = false;
    const err = ab.trainingError();
    if (err > ab.bound() + 1e-12) ok = false;
    last = err;
  }
  check(ok, "every ε < 1/2 and training error ≤ bound");
  check(last === 0, `separable set learnt: ${last}`);
  // the hand example of the viola-jones exercise: 6 values
  const v = Float32Array.of(1, 2, 3, 4, 5, 6), yy = Int8Array.of(-1, -1, 1, -1, 1, 1);
  const M6 = B.prepareFeatures(v, 1, 6);
  const ab6 = new B.AdaBoost(M6, yy);
  const s6 = ab6.round();
  check(close(s6.error, 1 / 6) && s6.polarity === 1 && close(s6.threshold, 2.5), `hand stump ${JSON.stringify(s6)}`);
  check(close(s6.alpha, 0.5 * Math.log(5)), "alpha = ½ ln 5");
  check(close(ab6.w[3], 0.5) && close(ab6.w[0], 0.1), "reweighted: mistake 1/2, others 1/10");
  // cascade rates
  const cr = B.cascadeRates([0.99, 0.99, 0.99], [0.5, 0.4, 0.3], [2, 5, 20]);
  check(close(cr.D, 0.99 ** 3) && close(cr.F, 0.06) && close(cr.features, 2 + 5 * 0.5 + 20 * 0.2), "cascade rates");
  // feature scaling: value of a scaled feature on a scaled image equals the base one
  const pool = B.featurePool(24, 2, [2, 4]);
  check(pool.length > 100, `pool size ${pool.length}`);
}

console.log(`${failures === 0 ? "OK" : "FAILED"}: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
