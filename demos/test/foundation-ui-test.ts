// Tests for the non-DOM parts of the plotting/UI/colour/WebGL libraries:
//  - lib/colour.ts: sRGB transfer function, RGB<->XYZ<->Lab against published values,
//    HSV round trips, ΔE76, monotone lightness of the sequential maps, label palette;
//  - lib/mat4.ts: inverse, lookAt, perspective/ortho corners, normal matrix;
//  - lib/plot.ts tick helpers, lib/shapecanvas.ts rasteriser, lib/datasets2d.ts generators.
// Run: node_modules/.bin/esbuild demos/test/foundation-ui-test.ts --bundle --platform=node | node
import * as C from "../src/lib/colour";
import * as M from "../src/lib/mat4";
import { niceTicks, logTicks, makeScale, ellipsePoints } from "../src/lib/plot";
import { rasterise, polygonArea, polygonCentroid, rotatePoly, mirrorPoly, ellipsePoly, Poly } from "../src/lib/shapecanvas";
import { DATASETS, dataset, classStats, withLabelNoise } from "../src/lib/datasets2d";
import { paintSegment } from "../src/lib/paint";
import { offsetsFromGrid, gridFromText } from "../src/lib/gridui";
import { rng } from "../src/lib/ui";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}
const near = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;
const nearV = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => a.length === b.length && Array.from(a).every((v, i) => near(v, b[i], tol));
const fmt = (v: ArrayLike<number>) => "[" + Array.from(v).map(x => x.toFixed(4)).join(", ") + "]";

// ---------------------------------------------------------------- colour
function testColour(): void {
  // transfer function: round trip and continuity at the knee
  for (let i = 0; i <= 1000; i++) { const v = i / 1000; check(near(C.linearToSrgb(C.srgbToLinear(v)), v, 1e-9), `sRGB round trip ${v}`); }
  check(near(C.srgbToLinear(0.04045), C.srgbToLinear(0.04045 + 1e-9), 1e-8), "sRGB decode continuous at the knee");
  check(near(C.linearToSrgb(0.0031308), C.linearToSrgb(0.0031308 + 1e-10), 1e-7), "sRGB encode continuous at the knee");
  check(near(C.srgbToLinear(0.5), 0.214041, 1e-6), "sRGB 0.5 -> 0.2140");
  check(C.SRGB8_TO_LINEAR[255] === 1 && C.SRGB8_TO_LINEAR[0] === 0, "8-bit decode table ends");
  for (let b = 0; b < 256; b++) check(C.linearToSrgb8(C.SRGB8_TO_LINEAR[b]) === b, `8-bit round trip ${b}`);

  // matrices: white maps to D65, inverse pair
  check(nearV(C.linearRgbToXyz([1, 1, 1]), C.D65, 1e-6), "M (1,1,1) = D65 white " + fmt(C.linearRgbToXyz([1, 1, 1])));
  const r = rng(11);
  for (let k = 0; k < 200; k++) {
    const v: C.RGB = [r.unif(), r.unif(), r.unif()];
    check(nearV(C.xyzToLinearRgb(C.linearRgbToXyz(v)), v, 1e-6), "RGB->XYZ->RGB " + fmt(v));
    check(nearV(C.labToXyz(C.xyzToLab(C.linearRgbToXyz(v))), C.linearRgbToXyz(v), 1e-9), "XYZ->Lab->XYZ " + fmt(v));
    const hsv = C.rgbToHsv(v);
    check(nearV(C.hsvToRgb(hsv), v, 1e-9), "HSV round trip " + fmt(v));
    check(C.deltaE76(C.srgbToLab(v), C.srgbToLab(v)) === 0, "ΔE76 of a colour with itself is 0");
  }
  // Lab reference values (sRGB primaries, D65, as published by B. Lindbloom's calculator)
  const refs: [C.RGB, C.RGB][] = [
    [[1, 1, 1], [100, 0, 0]],
    [[1, 0, 0], [53.2408, 80.0925, 67.2032]],
    [[0, 1, 0], [87.7347, -86.1827, 83.1793]],
    [[0, 0, 1], [32.2970, 79.1875, -107.8602]],
    [[0, 0, 0], [0, 0, 0]],
  ];
  for (const [rgb, lab] of refs) check(nearV(C.srgbToLab(rgb), lab, 0.01), `Lab of ${fmt(rgb)} = ${fmt(C.srgbToLab(rgb))}, expected ${fmt(lab)}`);
  // greys have a* = b* = 0
  for (let g = 0; g <= 10; g++) { const lab = C.srgbToLab([g / 10, g / 10, g / 10]); check(near(lab[1], 0, 1e-3) && near(lab[2], 0, 1e-3), `grey ${g / 10} is neutral`); }
  // ΔE76 red-green from the reference Labs
  check(near(C.deltaE76(refs[1][1], refs[2][1]), Math.hypot(53.2408 - 87.7347, 80.0925 + 86.1827, 67.2032 - 83.1793), 1e-9), "ΔE76 is the Lab distance");
  check(near(C.deltaE76([50, 0, 0], [53, 4, 0]), 5, 1e-12), "ΔE76 3-4-5");
  // HSV of pure colours and greys
  check(nearV(C.rgbToHsv([1, 0, 0]), [0, 1, 1], 1e-12), "HSV red");
  check(nearV(C.rgbToHsv([0, 1, 0]), [120, 1, 1], 1e-12), "HSV green");
  check(nearV(C.rgbToHsv([0, 0, 1]), [240, 1, 1], 1e-12), "HSV blue");
  check(nearV(C.rgbToHsv([1, 0, 1]), [300, 1, 1], 1e-12), "HSV magenta");
  check(Number.isNaN(C.rgbToHsv([0.4, 0.4, 0.4])[0]), "HSV of grey has undefined hue");
  check(nearV(C.hsvToRgb([NaN, 0, 0.4]), [0.4, 0.4, 0.4], 1e-12), "HSV with undefined hue -> grey");
  // hex
  check(C.rgbToHex(C.hexToRgb("#1f77b4")) === "#1f77b4", "hex round trip");

  // sequential maps: monotone lightness; diverging: white middle, symmetric lightness
  for (const [name, map] of [["grey", C.grey], ["viridis", C.viridis], ["inferno", C.inferno]] as const) {
    const L = C.lightnessProfile(map, 128);
    check(L.every((v, i) => i === 0 || v > L[i - 1] - 1e-6), `${name}: L* increases monotonically`);
    check(L[127] - L[0] > 60, `${name}: L* spans > 60`);
  }
  const Ld = C.lightnessProfile(C.diverging, 129);
  check(Ld[64] > 95, "diverging: light middle");
  check(Ld.slice(0, 64).every((v, i) => i === 0 || v >= Ld[i - 1] - 1e-6) && Ld.slice(64).every((v, i) => i === 0 || v <= Ld[64 + i - 1] + 1e-6), "diverging: L* rises to the middle and falls after");
  check(near(C.divergingT(0, 3), 0.5, 0) && near(C.divergingT(-3, 3), 0, 0) && near(C.divergingT(3, 3), 1, 0), "divergingT");
  // LUTs
  const lut = C.colormapLUT(C.grey, 256);
  check(lut[0] === 0 && lut[4 * 255] === 255 && lut[4 * 128] === 128 && lut[3] === 255, "grey LUT");
  const rgba = C.scalarToRGBA([0, 5, 10, NaN], { lo: 0, hi: 10 });
  check(rgba[0] === 0 && rgba[4] === 128 && rgba[8] === 255 && rgba[15] === 0, "scalarToRGBA maps lo/mid/hi and NaN");
  // label palette: background, distinct consecutive labels, deterministic, in gamut
  check(nearV(C.labelColor(0), [0, 0, 0], 0), "label 0 is the background");
  const pal = C.labelPalette(64);
  let minDE = Infinity;
  for (let k = 2; k < 64; k++) minDE = Math.min(minDE, C.deltaE76(C.srgbToLab(pal[k]), C.srgbToLab(pal[k - 1])));
  check(minDE > 20, `consecutive labels differ (min ΔE76 = ${minDE.toFixed(1)})`);
  check(nearV(C.labelColor(17), C.labelColor(17), 0), "label palette deterministic");
  check(pal.every(c => c.every(v => v >= 0 && v <= 1)), "label palette in [0,1]");
  const ll = C.labelLUT(10);
  check(ll[4 * 3] === Math.round(255 * C.labelColor(3)[0]) || Math.abs(ll[4 * 3] - 255 * C.labelColor(3)[0]) <= 1, "label LUT entry k = label k");
}

// ---------------------------------------------------------------- mat4
function testMat4(): void {
  const r = rng(5);
  const I = M.identity();
  for (let k = 0; k < 100; k++) {
    const A = M.mul(M.translation(r.norm(), r.norm(), r.norm()), M.rotation([r.norm(), r.norm(), r.norm()], 3 * r.norm()), M.scaling(0.5 + r.unif(), 0.5 + r.unif(), 0.5 + r.unif()));
    const Ai = M.invert(A)!;
    check(nearV(M.multiply(A, Ai), I, 1e-5) && nearV(M.multiply(Ai, A), I, 1e-5), "A * inv(A) = I");
    // normal matrix keeps normals perpendicular to transformed tangents
    const t: M.Vec3 = [r.norm(), r.norm(), r.norm()], n0: M.Vec3 = [r.norm(), r.norm(), r.norm()];
    const n = M.sub(n0, M.scale3(t, M.dot(n0, t) / M.dot(t, t)));        // n ⟂ t
    const tn = M.transformDir(A, t), nn = M.mat3Vec(M.normalMatrix(A), n);
    check(Math.abs(M.dot(tn, nn)) < 1e-4 * M.length(tn) * M.length(nn), "M^-T n stays perpendicular to M t");
    // transpose of transpose
    check(nearV(M.transpose(M.transpose(A)), A, 0), "transpose twice");
  }
  check(M.invert(M.scaling(1, 0, 1)) === null, "singular matrix has no inverse");
  check(nearV(M.fromRows([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]), [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16], 0), "fromRows is column-major");
  check(near(M.at(M.translation(1, 2, 3), 1, 3), 2, 0), "at(row, col)");
  // rotations are counter-clockwise about the axis
  check(nearV(M.transformPoint(M.rotationZ(Math.PI / 2), [1, 0, 0]), [0, 1, 0], 1e-6), "rotZ(90°) x -> y");
  check(nearV(M.transformPoint(M.rotationX(Math.PI / 2), [0, 1, 0]), [0, 0, 1], 1e-6), "rotX(90°) y -> z");
  check(nearV(M.transformPoint(M.rotationY(Math.PI / 2), [0, 0, 1]), [1, 0, 0], 1e-6), "rotY(90°) z -> x");
  check(nearV(M.rotation([0, 0, 1], 0.7), M.rotationZ(0.7), 1e-6), "Rodrigues = rotZ");
  // lookAt: orthonormal rotation, eye -> origin, target on -z
  for (let k = 0; k < 50; k++) {
    const eye: M.Vec3 = [3 * r.norm(), 3 * r.norm(), 3 * r.norm()], target: M.Vec3 = [r.norm(), r.norm(), r.norm()];
    const V = M.lookAt(eye, target);
    check(nearV(M.transformPoint(V, eye), [0, 0, 0], 1e-4), "V eye = 0");
    const tv = M.transformPoint(V, target);
    check(near(tv[0], 0, 1e-4) && near(tv[1], 0, 1e-4) && tv[2] < 0 && near(-tv[2], M.length(M.sub(target, eye)), 1e-4), "target on -z at its distance");
    const R = [0, 1, 2].map(c => [V[4 * c], V[4 * c + 1], V[4 * c + 2]] as M.Vec3);
    check(R.every((a, i) => R.every((b, j) => near(M.dot(a, b), i === j ? 1 : 0, 1e-5))), "R is orthonormal");
  }
  // perspective: near/far frustum corners -> NDC cube corners
  const fovy = 0.9, asp = 1.6, n = 0.5, f = 20, P = M.perspective(fovy, asp, n, f);
  for (const [d, z] of [[n, -1], [f, 1]] as const) for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
    const h = d * Math.tan(fovy / 2), w = h * asp;
    check(nearV(M.transformPoint(P, [sx * w, sy * h, -d]), [sx, sy, z], 1e-4), `perspective corner (${sx},${sy}) at ${d}`);
  }
  const O = M.ortho(-2, 3, -1, 4, 0.5, 9);
  check(nearV(M.transformPoint(O, [-2, -1, -0.5]), [-1, -1, -1], 1e-6) && nearV(M.transformPoint(O, [3, 4, -9]), [1, 1, 1], 1e-6), "ortho box -> NDC cube");
  check(M.formatMat(M.identity(), 1).split("\n").length === 4, "formatMat prints 4 rows");
}

// ---------------------------------------------------------------- plot helpers, shapes, datasets, paint, grid
function testOthers(): void {
  check(nearV(niceTicks(0, 1, 5), [0, 0.2, 0.4, 0.6, 0.8, 1], 1e-12), "niceTicks 0..1 = " + niceTicks(0, 1, 5));
  check(nearV(niceTicks(-3, 7, 5), [-2, 0, 2, 4, 6], 1e-12), "niceTicks -3..7 = " + niceTicks(-3, 7, 5));
  check(nearV(logTicks(0.01, 100), [0.01, 0.1, 1, 10, 100], 1e-12), "logTicks");
  const s = makeScale([1, 1000], [0, 300], true);
  check(near(s(10), 100, 1e-9) && near(s.invert(200), 100, 1e-9), "log scale");
  const e = ellipsePoints([1, 2], [[4, 0], [0, 1]], 1, 4);
  check(nearV(e[0], [3, 2], 1e-9) && nearV(e[1], [1, 3], 1e-9), "ellipse points on the axes");

  // rasteriser: area of a rectangle, ring with a hole, area of a disc, rotation/mirror invariance
  const sq: Poly = [[10, 10], [30, 10], [30, 25], [10, 25]];
  const m1 = rasterise([sq], 64, 64);
  check(m1.reduce((a, b) => a + b, 0) === 20 * 15, "rectangle 20 x 15 covers 300 pixels");
  check(m1[10 * 64 + 10] === 1 && m1[9 * 64 + 10] === 0 && m1[10 * 64 + 30] === 0, "rectangle edges at pixel centres");
  const ring = [ellipsePoly(32, 32, 20, 20, 400), ellipsePoly(32, 32, 10, 10, 400)];
  const area = ring.map(p => Math.abs(polygonArea(p)));
  const mr = rasterise(ring, 64, 64).reduce((a, b) => a + b, 0);
  check(near(mr, area[0] - area[1], 25), `ring with hole: ${mr} px vs area ${(area[0] - area[1]).toFixed(1)}`);
  check(rasterise(ring, 64, 64)[32 * 64 + 32] === 0, "hole is empty (even-odd)");
  check(rasterise(ring, 64, 64, "nonzero")[32 * 64 + 32] === 1, "nonzero rule fills the hole when both run the same way");
  check(near(polygonArea(sq), 300, 1e-9) && nearV(polygonCentroid(sq), [20, 17.5], 1e-9), "shoelace area and centroid");
  const rot = rotatePoly(sq, 0.7);
  check(near(polygonArea(rot), polygonArea(sq), 1e-9) && nearV(polygonCentroid(rot), polygonCentroid(sq), 1e-9), "rotation keeps area and centroid");
  check(nearV(rotatePoly([[1, 0]], Math.PI / 2, [0, 0])[0], [0, -1], 1e-12), "rotation is counter-clockwise on screen (y down)");
  check(near(polygonArea(mirrorPoly(sq, "x")), polygonArea(sq), 1e-9), "mirror keeps orientation");

  // datasets: deterministic, labels in range, sizes
  for (const d of DATASETS) {
    const a = d.make({ seed: 3 }), b = d.make({ seed: 3 });
    check(a.points.length > 20, `${d.id}: has points`);
    check(JSON.stringify(a) === JSON.stringify(b), `${d.id}: same seed, same data`);
    check(a.points.every(p => p.c >= 0 && p.c < a.k && Number.isFinite(p.x) && Number.isFinite(p.y)), `${d.id}: labels in 0..k-1`);
    check(new Set(a.points.map(p => p.c)).size === a.k, `${d.id}: every class present`);
  }
  const xo = dataset("xor");
  check(xo.points.every(p => p.c === ((p.x > 0.5) !== (p.y > 0.5) ? 1 : 0)), "xor labels");
  const st = classStats(dataset("blobs", { seed: 4, n: 4000 }), 0);
  check(near(st.mean[0], 0.32, 0.01) && near(st.mean[1], 0.4, 0.01) && near(st.cov[0][0], 0.0081, 0.001), "blob statistics");
  const noisy = withLabelNoise(dataset("blobs"), 0.2);
  const flipped = noisy.points.filter((p, i) => p.c !== dataset("blobs").points[i].c).length;
  check(flipped > 5 && flipped < 40, `label noise flips about 20% (${flipped} of 100)`);

  // paint: a disc of radius 2 and a segment
  const map = new Uint8Array(32 * 32);
  paintSegment(map, 32, 32, 16, 16, 16, 16, 2, 3);
  // pixel centres within 2.5 of (16, 16): offsets ±0.5, ±1.5 in x and y -> 16 pixels
  check(map.filter(v => v === 3).length === 16, "brush of radius 2 paints the 16 pixels within 2.5");
  const m2 = new Uint8Array(32 * 32);
  paintSegment(m2, 32, 32, 2.5, 5.5, 20.5, 5.5, 0, 1);
  check(m2.reduce((a, b) => a + b, 0) === 19 && m2[5 * 32 + 2] === 1 && m2[5 * 32 + 20] === 1, "radius-0 horizontal stroke paints one row");

  // grid helpers
  check(JSON.stringify(offsetsFromGrid([0, 1, 0, 1, 1, 1, 0, 1, 0], 3)) === JSON.stringify([[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0]]), "SE offsets from a 3x3 grid");
  const g = gridFromText(["3 1 2", "4 0 1"]);
  check(g.rows === 2 && g.cols === 3 && g.values[3] === 4, "grid from text");
}

testColour();
testMat4();
testOthers();
console.log(`foundation-ui: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
