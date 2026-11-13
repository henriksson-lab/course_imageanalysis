// Reference checks for the contrast unit: lib/contrast.ts (LUTs, curves, equalisation,
// matching, CLAHE, scenes) against hand-worked examples and definitions. Run:
//   node_modules/.bin/esbuild demos/test/contrast-test.ts --bundle --platform=node --log-level=warning | node
import * as C from "../src/lib/contrast";
import { fromFunction, Grey } from "../src/lib/imageops";
import { rng } from "../src/lib/ui";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const arr = (a: ArrayLike<number>) => Array.from(a).join(",");

/** The 8-level, 64-pixel worked example: counts 6, 22, 18, 8, 5, 3, 1, 1. */
const WORKED = [6, 22, 18, 8, 5, 3, 1, 1];
function workedImage(): number[] {
  const px: number[] = [];
  WORKED.forEach((c, v) => { for (let k = 0; k < c; k++) px.push(v); });
  return px;
}

function testLuts(): void {
  current = "luts";
  const id = C.lutFromFunction(v => v, 256);
  const data = Float32Array.from({ length: 1000 }, (_, i) => (i * 37) % 256);
  check(arr(C.applyLut(data, id)) === arr(data), "identity LUT is the identity");
  const a = C.lutFromFunction(C.gammaCurve(0.5, 0, 255), 256).map(Math.round);
  const b = C.lutFromFunction(C.negativeCurve(256), 256).map(Math.round);
  const ab = C.composeLut(a, b);
  check(arr(C.applyLut(data, ab)) === arr(C.applyLut(C.applyLut(data, a), b)), "composeLut equals applying twice");
  check(C.isMonotone(a) && !C.isMonotone(b), "gamma rises, negative falls");
  check(close(C.slopeOf(C.lutFromFunction(v => 3 * v, 10), 4), 3), "slope of a line");
  // the window of highbit/ (window16 windowLevel): round(255 clip((v - lo)/(hi - lo)))
  const wl = (v: number, lo: number, hi: number) => Math.round(255 * Math.min(1, Math.max(0, (v - lo) / (hi - lo))));
  const w = C.windowCurve(130, 3176);
  let ok = true;
  for (const v of [0, 130, 131, 500, 1653, 3175, 3176, 4095]) ok = ok && Math.round(w(v)) === wl(v, 130, 3176);
  check(ok, "window curve = highbit window formula at 130 and 3176");
  check(Math.round(w(130)) === 0 && Math.round(w(3176)) === 255, "window ends");
  // gamma numbers of the deck and B2
  const g = (v: number, gam: number) => Math.pow(v, gam);
  check(close(g(0.04, 0.5), 0.2) && close(g(0.81, 0.5), 0.9) && close(g(0.01, 0.5), 0.1), "gamma 0.5 values");
  check(close(0.5 / Math.sqrt(0.04), 2.5) && Math.abs(0.5 / Math.sqrt(0.81) - 0.5556) < 1e-4, "gamma slopes");
  // log: c = 255/ln 4096
  const lc = C.logCurve("fit", 0, 4095);
  check(Math.round(lc(10)) === 74 && Math.round(lc(100)) === 141 && Math.round(lc(1000)) === 212 && Math.round(lc(4095)) === 255, "log curve 10, 100, 1000, 4095");
  const s = C.sigmoidCurve(128, 20, 0, 255);
  check(close(s(0), 0) && close(s(255), 255) && C.isMonotone(C.lutFromFunction(s, 256), true), "sigmoid ends and rises");
  const pw = C.piecewiseLinear([[0, 0], [100, 200], [200, 100], [255, 255]], true);
  check(C.isMonotone(C.lutFromFunction(pw, 256)), "monotone piecewise-linear");
  check(C.thresholdCurve(100)(99) === 0 && C.thresholdCurve(100)(100) === 255, "threshold curve");
  // comb: stretching 8-bit data x4 uses every 4th level
  const lo8 = Float32Array.from({ length: 64 }, (_, i) => i);
  const st = C.applyLut(lo8, C.lutFromFunction(C.windowCurve(0, 63.75), 256).map(Math.round));
  check(C.distinctCount(st) === 64, "x4 stretch of 64 levels keeps 64 distinct values");
  // B4
  const b4 = C.lutFromFunction(C.windowCurve(60, 123), 256).map(Math.round);
  check(C.distinctCount(C.applyLut(Float32Array.from({ length: 64 }, (_, i) => 60 + i), b4)) === 64, "B4 8-bit");
  const b4b = C.lutFromFunction(C.windowCurve(960, 1980), 4096).map(Math.round);
  check(C.distinctCount(C.applyLut(Float32Array.from({ length: 1021 }, (_, i) => 960 + i), b4b)) === 256, "B4 12-bit");
}

function testEqualise(): void {
  current = "equalise";
  const lut = C.equaliseLut(WORKED, 8);
  check(arr(lut) === "1,3,5,6,6,7,7,7", "worked mapping " + arr(lut));
  const out = C.applyLut(workedImage(), lut);
  check(arr(C.levelHistogram(out, 8)) === "0,6,0,22,0,18,13,5", "worked output counts");
  check(arr(C.equaliseLut(WORKED, 8, "cmin")) === "0,3,5,6,6,7,7,7", "cmin variant");
  check(arr(C.equaliseValues(workedImage(), 8)) === arr(out), "equaliseValues = LUT route");
  // exact equalisation of 65 536 pixels to 256 levels
  const r = rng(11);
  const img = Float32Array.from({ length: 65536 }, () => Math.round(40 + 20 * r.norm()));
  const ex = C.equaliseExact(img, 256, "random", 3);
  check(C.levelHistogram(ex, 256).every(c => c === 256), "exact equalisation: 256 per level");
  const exl = C.equaliseExact(img, 256, "localMean", 3, 256, 256);
  check(C.levelHistogram(exl, 256).every(c => c === 256), "exact (local mean ties): 256 per level");
  // order kept by exact equalisation
  let ord = true;
  for (let i = 1; i < 2000; i++) if (img[i] < img[i - 1] && ex[i] > ex[i - 1]) ord = false;
  check(ord, "exact equalisation keeps the order of different values");
  // gamma first: same result
  const g = Float32Array.from(img, v => Math.pow((v + 200) / 455, 0.4));   // strictly rising on the data
  check(arr(C.equaliseValues(g, 256)) === arr(C.equaliseValues(img, 256)), "equalise(gamma(x)) = equalise(x)");
  // output distinct values <= input distinct values
  const eq = C.equaliseValues(img, 256);
  check(C.distinctCount(eq) <= C.distinctCount(img), "distinct out <= distinct in");
  // equalising twice changes nothing (up to rounding)
  const eq2 = C.equaliseValues(eq, 256);
  let maxd = 0;
  for (let i = 0; i < eq.length; i++) maxd = Math.max(maxd, Math.abs(eq2[i] - eq[i]));
  check(maxd <= 1, "equalise twice: changes by at most one level, got " + maxd);
  // constant image: one level
  check(C.distinctCount(C.equaliseValues(new Float32Array(100).fill(7), 256)) === 1, "constant image: one output level");
}

function testMatch(): void {
  current = "match";
  const t = C.matchLut(WORKED, [0, 0, 8, 8, 16, 16, 8, 8]);
  check(arr(t) === "2,4,5,6,7,7,7,7", "B6 matching " + arr(t));
  // matching to a flat target = equalisation (up to the rounding rule) for continuous-ish data
  const r = rng(5);
  const img = Float32Array.from({ length: 20000 }, () => Math.max(0, Math.min(255, Math.round(100 + 30 * r.norm()))));
  const h = C.levelHistogram(img, 256);
  const m = C.matchLut(h, C.flatTarget(256)), e = C.equaliseLut(h, 256);
  let md = 0;
  for (let k = 0; k < 256; k++) if (h[k] > 0) md = Math.max(md, Math.abs(m[k] - e[k]));
  check(md <= 1, "match to flat ~ equalise, max diff " + md);
  // matching an image to its own histogram is the identity on used levels
  const self = C.matchLut(h, h);
  let id = true;
  for (let k = 0; k < 256; k++) if (h[k] > 0 && self[k] !== k) id = false;
  check(id, "match to own histogram = identity");
  check(C.isMonotone(C.matchLut(h, C.gaussianTarget(256, 128, 40))), "matching LUT rises");
}

function testClip(): void {
  current = "clip";
  // B9: tile of 16 pixels, 4 levels, counts 1 10 4 1, limit 1.5 x mean (= 6)
  const one = C.clipHistogram([1, 10, 4, 1], 6, false);
  check(arr(one.clipped) === "2,7,5,2" && one.excess === 4, "B9 one pass " + arr(one.clipped));
  check(close(4 * 10 / 16, 2.5) && close(4 * 7 / 16, 1.75), "B9 gains");
  const it = C.clipHistogram([1, 10, 4, 1], 6, true);
  check(Math.max(...it.clipped) <= 6 + 1e-6 && close(it.clipped.reduce((a, b) => a + b, 0), 16), "iterated clip: no bin above the limit, total kept");
}

function testClahe(): void {
  current = "clahe";
  const r = rng(9);
  const img: Grey = fromFunction(128, 128, (x, y) => Math.round(Math.max(0, Math.min(255, 60 + 0.6 * x + 20 * Math.sin(y / 9) + 8 * r.norm()))));
  // one tile, no clip = global equalisation with 256 bins over [0, 255]
  const g = C.clahe(img, { tilesX: 1, tilesY: 1, clip: Infinity, range: [0, 256], bins: 256 });
  const h = C.levelHistogram(img.data, 256), lut = C.equaliseLut(h, 256, "plain", false);
  let md = 0;
  for (let i = 0; i < img.data.length; i++) md = Math.max(md, Math.abs(g.out.data[i] - lut[Math.round(img.data[i])]));
  check(md < 1e-9, "1 tile, no clip = global equalisation, diff " + md);
  // beta = 1 (iterated): identity up to rounding (bins = levels)
  const b1 = C.clahe(img, { tilesX: 4, tilesY: 4, clip: 1, range: [0, 256], bins: 256 });
  md = 0;
  for (let i = 0; i < img.data.length; i++) md = Math.max(md, Math.abs(b1.out.data[i] - img.data[i]));
  check(md <= 1, "beta = 1: identity up to rounding, diff " + md.toFixed(3));
  // every tile LUT slope <= beta (iterated clipping)
  for (const beta of [2, 3, 6]) {
    const c = C.clahe(img, { tilesX: 4, tilesY: 4, clip: beta, bins: 64 });
    let ok = true;
    c.tileHists.forEach((hc, t) => { if (C.maxGain(hc, c.tileN[t]) > beta + 1e-6) ok = false; });
    check(ok, `tile gains <= beta = ${beta}`);
  }
  // interpolation: no jump across tile borders on a smooth gradient
  const grad: Grey = fromFunction(128, 128, (x, y) => Math.round(x + y) % 256);
  const ci = C.clahe(grad, { tilesX: 4, tilesY: 4, clip: 3, range: [0, 256] });
  const cs = C.clahe(grad, { tilesX: 4, tilesY: 4, clip: 3, range: [0, 256], interpolate: false });
  const jump = (o: Grey) => { let m = 0; for (let y = 0; y < 128; y++) m = Math.max(m, Math.abs(o.data[y * 128 + 32] - o.data[y * 128 + 31])); return m; };
  check(jump(ci.out) < 6 && jump(cs.out) > 2 * jump(ci.out), `interpolated: small step at a tile border (${jump(ci.out).toFixed(2)} vs seams ${jump(cs.out).toFixed(2)})`);
  // weights sum to one
  let wsum = true;
  for (const [x, y] of [[0, 0], [17, 90], [127, 127], [64, 3]]) {
    const s = C.claheWeights(x, y, 128, 128, 4, 4).reduce((a, b) => a + b.w, 0);
    if (!close(s, 1)) wsum = false;
  }
  check(wsum, "bilinear weights sum to 1");
}

function testMeasure(): void {
  current = "measure";
  // B10: background-subtracted ratio, raw and after gamma 0.5 on v/4095
  const ratio = (f: (v: number) => number) => (f(600) - f(100)) / (f(300) - f(100));
  check(close(ratio(v => v), 2.5), "raw ratio 2.5");
  check(Math.abs(ratio(v => Math.sqrt(v / 4095)) - 1.98) < 0.005, "gamma ratio 1.98");
  check(Math.abs(Math.sqrt(500) - 22.36) < 0.01, "Jensen: sqrt(500)");
  // control/treated scene: true ratio of background-subtracted nucleus means is the factor
  const s = C.controlTreated();
  const bc = C.maskedMean(s.control, s.background), bt = C.maskedMean(s.treated, s.background);
  const rc = C.maskedMean(s.control, s.nuclei) - bc, rt = C.maskedMean(s.treated, s.nuclei) - bt;
  check(Math.abs(rt / rc - 1.5) < 0.01, "control/treated raw ratio " + (rt / rc).toFixed(4));
  check(Math.abs(bc - 100) < 1 && Math.abs(bt - 100) < 1, "background 100");
  // fluorescence12 matches the window16 percentiles used in highbit/ (130 and 3176)
  const f = Array.from(C.fluorescence12()).sort((a, b) => a - b);
  const pct = (p: number) => f[Math.min(f.length - 1, Math.max(0, Math.round(p / 100 * (f.length - 1))))];
  check(pct(0.5) === 130 && pct(99.5) === 3176, `fluorescence12 percentiles ${pct(0.5)} ${pct(99.5)}`);
}

testLuts();
testEqualise();
testMatch();
testClip();
testClahe();
testMeasure();
console.log(failures ? `contrast: ${failures} of ${checks} checks FAILED` : `contrast: all ${checks} checks passed`);
if (failures) process.exit(1);
