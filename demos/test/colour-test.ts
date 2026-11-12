// Tests for the colour day (plan/colour-day.md): lib/spectra.ts, colourcloud/colourx.ts
// (YCbCr, CIEDE2000, adaptation, dichromats, maps), the white-balance model, the
// segmentation rules, the gamma demo and the colour cloud's PCA/decorrelation stretch.
// Reference values: published sRGB matrix, Sharma, Wu & Dalal (2005) CIEDE2000 test
// pairs, brute force (Otsu over all cuts, k-means assignment), and algebraic identities.
// Run: node_modules/.bin/esbuild demos/test/colour-test.ts --bundle --platform=node --log-level=warning | node
import * as SP from "../src/lib/spectra";
import * as C from "../src/lib/colour";
import * as X from "../src/colourcloud/colourx";
import { colourPCA, decorrelationStretch, cloudPositions, SPACES } from "../src/colourcloud/cloud";
import { makeScene } from "../src/colourcloud/scenes";
import * as WB from "../src/whitebalance/wb";
import * as SG from "../src/colourseg/seg";
import { makeImage, process as gammaProcess, meanLuminance } from "../src/gammablur/gamma";
import { ycbcrRoundTrip } from "../src/colourmatrix/ops";
import { rng } from "../src/lib/random";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}
const near = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

// ---- sRGB transfer, matrices ----
for (let i = 0; i <= 1000; i++) { const v = i / 1000; check(near(C.linearToSrgb(C.srgbToLinear(v)), v, 1e-12), `sRGB round trip ${v}`); }
check(near(C.srgbToLinear(0.04045), 0.04045 / 12.92, 1e-12) && near(((0.04045 + 0.055) / 1.055) ** 2.4, 0.04045 / 12.92, 1e-6), "sRGB knee continuous");
const w = C.linearRgbToXyz([1, 1, 1]);
check(w.every((v, k) => near(v, C.D65[k], 2e-4)), "M_sRGB (1,1,1) = D65");
const built = SP.rgbToXyzMatrix(SP.SRGB_DISPLAY).M;
for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) check(near(built[r][c], C.M_RGB_TO_XYZ[r][c], 5e-4), `built sRGB matrix [${r}][${c}] ${built[r][c]}`);
const Mi = SP.inv3(built), I3 = SP.mat3mul(Mi, built);
for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) check(near(I3[r][c], r === c ? 1 : 0, 1e-12), "inv3 · M = I");
check(near(SP.rgbToXyzMatrix(SP.SRGB_DISPLAY).M[1].reduce((s, v) => s + v, 0), 1, 1e-12), "luminance row sums to 1");

// ---- Lab, HSV, YCbCr ----
const labW = C.xyzToLab(C.D65);
check(near(labW[0], 100, 1e-9) && near(labW[1], 0, 1e-9) && near(labW[2], 0, 1e-9), "Lab of white = (100, 0, 0)");
for (const g of [0.01, 0.18, 0.5, 0.9]) { const l = C.srgbToLab([g, g, g]); check(near(l[1], 0, 1e-3) && near(l[2], 0, 1e-3), `grey ${g} has a* = b* = 0`); }
check(near(C.xyzToLab([0, 0.18, 0].map((_, k) => 0.18 * C.D65[k]) as C.RGB)[0], 49.496, 1e-3), "L* of 18 % grey = 49.5");
const r0 = rng(3);
for (let t = 0; t < 500; t++) {
  const c: C.RGB = [r0.unif(), r0.unif(), r0.unif()];
  if (Math.max(...c) - Math.min(...c) < 0.02) continue;
  const back = C.hsvToRgb(C.rgbToHsv(c));
  check(back.every((v, k) => near(v, c[k], 1e-9)), "HSV round trip");
  const y = X.rgbToYcbcr(255 * c[0], 255 * c[1], 255 * c[2]), b = X.ycbcrToRgb(y[0], y[1], y[2]);
  check(b.every((v, k) => near(v, 255 * c[k], 0.6)), "YCbCr round trip within rounding of the coefficients");
}
check(Number.isNaN(C.rgbToHsv([0.5, 0.5, 0.5])[0]), "hue of grey undefined");
const yRed = X.rgbToYcbcr(255, 0, 0);
check(near(yRed[0], 76.245, 1e-9) && near(yRed[1], 128 - 0.564 * 76.245, 1e-9) && near(yRed[2], 128 + 0.713 * (255 - 76.245), 1e-9), "Y'CbCr of red");

// ---- CIEDE2000: Sharma, Wu & Dalal (2005) test pairs ----
const pairs: [C.RGB, C.RGB, number][] = [
  [[50, 2.6772, -79.7751], [50, 0, -82.7485], 2.0425],
  [[50, 3.1571, -77.2803], [50, 0, -82.7485], 2.8615],
  [[50, 2.8361, -74.0200], [50, 0, -82.7485], 3.4412],
  [[50, 0, 0], [50, -1, 2], 2.3669],
  [[50, 2.5, 0], [73, 25, -18], 27.1492],
  [[50, 2.5, 0], [61, -5, 29], 22.8977],
  [[50, 2.5, 0], [56, -27, -3], 31.9030],
  [[50, 2.5, 0], [58, 24, 15], 19.4535],
  [[60.2574, -34.0099, 36.2677], [60.4626, -34.1751, 39.4387], 1.2644],
];
for (const [a, b, d] of pairs) {
  check(near(X.deltaE2000(a, b), d, 1e-4), `ΔE2000 ${a} vs ${b}: ${X.deltaE2000(a, b)} (want ${d})`);
  check(near(X.deltaE2000(b, a), d, 1e-4), "ΔE2000 symmetric");
}

// ---- adaptation and dichromats ----
for (const basis of ["rgb", "lms", "bradford"] as const) {
  const src: C.RGB = [1.4, 1.0, 0.45], M = X.adaptationMatrix(src, [1, 1, 1], basis), o = SP.mat3v(M, src);
  check(o.every(v => near(v, 1, 1e-6)), `adaptation (${basis}) maps the source white to white`);
}
for (const k of ["protan", "deutan", "tritan"] as const) {
  const M = X.DICHROMAT[k], det = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]));
  check(near(det, 0, 1e-9), `${k} matrix is singular (det ${det})`);
  const M2 = SP.mat3mul(M, M);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) check(near(M2[r][c], M[r][c], 1e-9), `${k}: a projection (M² = M)`);
  check(SP.mat3v(M, [1, 1, 1]).every(v => near(v, 1, 1e-9)), `${k}: white unchanged`);
  // rank 2: some 2 x 2 minor is non-zero
  const minor = Math.abs(M[0][0] * M[1][1] - M[0][1] * M[1][0]) + Math.abs(M[1][1] * M[2][2] - M[1][2] * M[2][1]) + Math.abs(M[0][0] * M[2][2] - M[0][2] * M[2][0]);
  check(minor > 1e-3, `${k}: rank 2, not lower`);
}
// red and green collapse for a deuteranope more than for the trichromat
{
  const red = X.simulateDichromat([0.8, 0.3, 0.2], "deutan"), green = X.simulateDichromat([0.45, 0.55, 0.2], "deutan");
  const dT = C.deltaE76(C.srgbToLab([0.8, 0.3, 0.2]), C.srgbToLab([0.45, 0.55, 0.2])), dD = C.deltaE76(C.srgbToLab(red), C.srgbToLab(green));
  check(dD < 0.5 * dT, `deutan: red vs green ΔE ${dD.toFixed(1)} < half of ${dT.toFixed(1)}`);
}

// ---- colour maps: lightness ----
const monotone = (m: C.ColorMap) => { const p = C.lightnessProfile(m, 200); return p.every((v, i) => i === 0 || v >= p[i - 1] - 0.05); };
for (const [n, m] of [["grey", C.grey], ["viridis", C.viridis], ["inferno", C.inferno], ["cividis", X.cividis], ["magma", X.magma]] as [string, C.ColorMap][])
  check(monotone(m), `${n} has monotone L*`);
check(!monotone(X.jet) && !monotone(X.rainbow), "jet and the hue circle are not monotone");
{ const p = C.lightnessProfile(C.diverging, 201), iMax = p.indexOf(Math.max(...p)); check(Math.abs(iMax - 100) <= 3, "diverging: lightest at the centre"); }

// ---- spectra ----
const wxy = SP.xyzToXy(SP.WHITE_XYZ);
check(near(wxy[0], 0.3127, 1e-3) && near(wxy[1], 0.3290, 1e-3), `D65 chromaticity from the CMF fit ${wxy}`);
check(near(SP.WHITE_XYZ[1], 1, 1e-12), "D65 has Y = 1");
const table: [number, number, number][] = [[460, 0.1440, 0.0297], [500, 0.0082, 0.5384], [520, 0.0743, 0.8338], [560, 0.3731, 0.6245], [580, 0.5125, 0.4866], [600, 0.6270, 0.3725], [620, 0.6915, 0.3083], [700, 0.7347, 0.2653]];
for (const [l, x, y] of table) { const c = SP.xyzToXy(SP.cmf(l)); check(near(c[0], x, 0.012) && near(c[1], y, 0.016), `locus at ${l} nm: ${c} vs (${x}, ${y})`); }
check(SP.xyzToLinearSrgbCal(SP.WHITE_XYZ).every(v => near(v, 1, 1e-12)), "calibrated D65 -> RGB (1,1,1)");
// cones are a change of basis of the CMFs: same null space
{
  const A = SP.sensitivity("xyz"), L = SP.sensitivity("lms"), P = SP.nullProjector(A), r = rng(9);
  for (let t = 0; t < 20; t++) {
    const v = Float64Array.from({ length: SP.NB }, () => r.norm()), d = P(v);
    check(SP.apply(A, d).every(x => Math.abs(x) < 1e-9), "A · P v = 0");
    check(SP.apply(L, d).every(x => Math.abs(x) < 1e-9), "null space of XYZ = null space of LMS");
    const d2 = P(d); check(d2.every((x, i) => near(x, d[i], 1e-9)), "P² = P");
  }
}
// metamers
for (const obs of ["lms", "camera", "deutan"] as SP.Observer[]) {
  const A = SP.sensitivity(obs);
  for (let seed = 1; seed <= 20; seed++) {
    const r = rng(seed), s1 = Float64Array.from(SP.WL, l => 60 + 40 * Math.sin(l / 50) ** 2);
    const s2 = SP.makeMetamer(s1, A, r.unif);
    const a1 = SP.apply(A, s1), a2 = SP.apply(A, s2);
    check(a1.every((v, k) => near(v, a2[k], 1e-6 * Math.max(1, Math.abs(v)))), `${obs} metamer: same response (seed ${seed})`);
    check(s2.every(v => v >= 0), `${obs} metamer is non-negative`);
    check(s2.some((v, i) => Math.abs(v - s1[i]) > 5), `${obs} metamer differs from the original`);
  }
}
{ // reflectance metamers stay in [0, 1]
  const A = SP.sensitivity("lms"), r = rng(4), s1 = new Float64Array(SP.NB).fill(0.5);
  const s2 = SP.makeMetamer(s1, A, r.unif, 0.92, 1);
  check(s2.every(v => v >= 0 && v <= 1), "reflectance metamer in [0, 1]");
}
{ const led = SP.metamericLED(), A = SP.sensitivity("lms"), a = SP.apply(A, led), b = SP.apply(A, SP.D65_SPD);
  check(led.every(v => v >= 0) && a.every((v, k) => near(v, b[k], 1e-6 * b[k])), "white LED is a non-negative metamer of D65"); }

// ---- white balance ----
{
  const base: Parameters<typeof WB.runWB>[0] = { kind: "chart", illum: "blackbody", T: 2800, obs: "eye", how: "greyworld", space: "rgb", card: 20, highlight: false };
  const res = WB.runWB(base);
  // grey world on linear RGB: the corrected chart image has equal channel means
  const img = res.img, n = img.w * img.h;
  const g = WB.estimateWhite(img, "greyworld", 0);
  const cols: C.RGB[] = []; for (let p = 0; p < n; p++) if (img.patch[p] >= 0) cols.push([img.r[p], img.g[p], img.b[p]]);
  const corr = WB.correct(cols, "greyworld", "rgb", g), m = [0, 1, 2].map(k => corr.reduce((s, c) => s + c[k], 0) / corr.length);
  check(near(m[0], m[1], 1e-9) && near(m[1], m[2], 1e-9), "grey world equalises the channel means");
  check(WB.runWB({ ...base, illum: "d65", how: "none" }).dE.every(d => d < 1e-6), "D65, no correction: ΔE = 0");
  const mean = (o: Partial<typeof base>) => { const r = WB.runWB({ ...base, ...o }); return r.dE.reduce((s, v) => s + v, 0) / r.dE.length; };
  check(mean({ how: "card", space: "bradford" }) < mean({ how: "card", space: "rgb" }) && mean({ how: "card", space: "rgb" }) < mean({ how: "card", space: "encoded" }), "card: Bradford < linear RGB < encoded");
  check(mean({ how: "lstsq" }) < mean({ how: "card", space: "bradford" }), "3x3 fit beats any diagonal");
  check(mean({ kind: "forest", T: 4000, how: "greyworld", space: "lms" }) > 2 * mean({ kind: "forest", T: 4000, how: "whitepatch", space: "lms" }), "grey world fails in the forest");
  check(mean({ how: "whitepatch", space: "lms", highlight: true }) > 2 * mean({ how: "whitepatch", space: "lms" }), "a clipped highlight fools the white patch");
  // least squares: the fit's residual is minimal (perturbing M increases it)
  const refl = WB.reflectances("chart"), src = WB.renderPatches(refl, SP.blackbody(2800), "eye"), dst = WB.renderPatches(refl, SP.D65_SPD, "eye");
  const F = WB.fitMatrix(src, dst), sse = (M: number[][]) => src.reduce((s, c, i) => { const o = SP.mat3v(M, c); return s + (o[0] - dst[i][0]) ** 2 + (o[1] - dst[i][1]) ** 2 + (o[2] - dst[i][2]) ** 2; }, 0);
  const s0 = sse(F);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) for (const e of [-1e-3, 1e-3]) { const G = F.map(row => [...row]); G[r][c] += e; check(sse(G) >= s0, "LS fit is a minimum"); }
}

// ---- segmentation ----
check(SG.hueDist(350, 10) === 20 && SG.hueDist(10, 350) === 20 && SG.hueDist(0, 180) === 180, "circular hue distance");
{
  const r = rng(12);
  for (let t = 0; t < 20; t++) {
    const v = Float32Array.from({ length: 400 }, (_, i) => (i < 200 ? 50 : 150) + 20 * r.norm());
    const th = SG.otsu(v);
    // brute force: maximise between-class variance over all thresholds at the data values
    let best = -1, bestT = 0;
    const sorted = Array.from(v).sort((a, b) => a - b);
    for (let k = 1; k < sorted.length; k++) {
      const tt = sorted[k]; let n0 = 0, s0 = 0, n1 = 0, s1 = 0;
      for (const x of v) { if (x < tt) { n0++; s0 += x; } else { n1++; s1 += x; } }
      const b = n0 * n1 * (s0 / n0 - s1 / n1) ** 2; if (b > best) { best = b; bestT = tt; }
    }
    let n0 = 0, s0 = 0, n1 = 0, s1 = 0;
    for (const x of v) { if (x < th) { n0++; s0 += x; } else { n1++; s1 += x; } }
    const bo = n0 * n1 * (s0 / n0 - s1 / n1) ** 2;
    check(bo >= 0.995 * best, `Otsu close to the brute-force optimum (${th.toFixed(1)} vs ${bestT.toFixed(1)})`);
  }
}
{
  const r = rng(2), n = 300, pts = new Float32Array(3 * n);
  for (let p = 0; p < n; p++) { const c = p % 3; pts[3 * p] = 50 * c + 5 * r.norm(); pts[3 * p + 1] = 30 * (c === 1 ? 1 : 0) + 5 * r.norm(); pts[3 * p + 2] = 5 * r.norm(); }
  const km = SG.kmeans(pts, n, 3, 100, 7);
  for (let p = 0; p < n; p++) {
    let best = 0, bd = Infinity;
    for (let j = 0; j < 3; j++) { const d = (pts[3 * p] - km.centres[j][0]) ** 2 + (pts[3 * p + 1] - km.centres[j][1]) ** 2 + (pts[3 * p + 2] - km.centres[j][2]) ** 2; if (d < bd) { bd = d; best = j; } }
    check(km.labels[p] === best, "k-means: converged labels are nearest centres");
  }
  check(km.iterations < 100, "k-means converged");
}
{
  const sc = makeScene("objects", 128), n = sc.w * sc.h, enc = (a: Float32Array) => Float32Array.from(a, v => C.linearToSrgb(Math.min(1, v)));
  const px = { r: enc(sc.r), g: enc(sc.g), b: enc(sc.b), n }, dv = SG.derive(px);
  const dHue = SG.diceClass(SG.hueRule(dv, n, 0, 20, 0.4, 0.1), sc.truth!, 1), dRgb = SG.diceClass(SG.rgbRule(px, 120, 90, 90), sc.truth!, 1);
  check(dHue > 0.9 && dHue > dRgb + 0.1, `hue rule survives the shadow (Dice ${dHue.toFixed(3)} vs RGB ${dRgb.toFixed(3)})`);
  const m = SG.hueRule(dv, n, 0, 20, 0, 0); let greySel = 0; for (let p = 0; p < n; p++) if (sc.truth![p] === 5 && m[p]) greySel++;
  check(greySel === 0, "noise-free grey card has undefined hue: never selected");
}

// ---- gamma demo ----
{
  const I = makeImage("stripes"), { naive, linear } = gammaProcess(I, "blur", 2);
  const row = 40 * 128; let a = 0, b = 0; for (let x = 16; x < 48; x++) { a += naive[0][row + x]; b += linear[0][row + x]; }
  check(Math.round(255 * a / 32) === 128 && Math.round(255 * b / 32) === 188, `stripes: 128 vs 188 (${Math.round(255 * a / 32)}, ${Math.round(255 * b / 32)})`);
  const ch = makeImage("lines"), res = gammaProcess(ch, "down4", 0);
  check(near(meanLuminance(res.linear), meanLuminance(ch), 1e-3) && meanLuminance(res.naive) < 0.9 * meanLuminance(ch), "downsampling in linear light keeps the mean luminance; encoded darkens");
}

// ---- Y'CbCr round trip without subsampling ----
{
  const sc = makeScene("objects", 32), e = (a: Float32Array) => Float32Array.from(a, v => 255 * C.linearToSrgb(Math.min(1, v)));
  const R = e(sc.r), G = e(sc.g), B = e(sc.b), rt = ycbcrRoundTrip(R, G, B, 32, 32, 1, "chroma");
  let err = 0; for (let p = 0; p < 32 * 32; p++) err = Math.max(err, Math.abs(rt.R[p] - R[p]), Math.abs(rt.G[p] - G[p]), Math.abs(rt.B[p] - B[p]));
  check(err < 0.6, `Y'CbCr round trip (max error ${err.toFixed(3)})`);
}

// ---- colour cloud: PCA and decorrelation stretch ----
{
  const r = rng(5), n = 2000, enc = { r: new Float32Array(n), g: new Float32Array(n), b: new Float32Array(n) };
  for (let p = 0; p < n; p++) { const t = 0.08 * r.norm(), u = 0.02 * r.norm(), v = 0.01 * r.norm(); enc.r[p] = 0.5 + t + u; enc.g[p] = 0.5 + t - u + v; enc.b[p] = 0.5 + t - v; }
  const pca = colourPCA(enc);
  check(pca.values[0] >= pca.values[1] && pca.values[1] >= pca.values[2], "PCA eigenvalues descending");
  const st = decorrelationStretch(enc, 1, pca), p2 = colourPCA(st);
  const avg = (p2.values[0] + p2.values[1] + p2.values[2]) / 3;
  check(p2.values.every(v => near(v, avg, 1e-3 * avg + 1e-9)), `stretch t=1 equalises the variances (${p2.values.map(v => v.toExponential(3))})`);
  const tr0 = pca.values.reduce((s, v) => s + v, 0), tr1 = p2.values.reduce((s, v) => s + v, 0);
  check(near(tr0, tr1, 1e-3 * tr0), "stretch keeps the total variance");
  const id = decorrelationStretch(enc, 0, pca);
  check(id.r.every((v, i) => near(v, enc.r[i], 1e-6)), "stretch t=0 is the identity");
  // every space maps black and white to finite positions; HSV puts all greys on the axis
  const greys = { r: Float32Array.from([0.01, 0.2, 0.5, 1]), g: Float32Array.from([0.01, 0.2, 0.5, 1]), b: Float32Array.from([0.01, 0.2, 0.5, 1]) };
  for (const [name, sp] of Object.entries(SPACES)) {
    const P = cloudPositions(greys, sp);
    check(Array.from(P).every(Number.isFinite), `${name}: finite positions`);
    if (name === "hsv" || name === "lab" || name === "ycbcr") for (let k = 0; k < 4; k++) check(Math.abs(P[3 * k]) < 1e-6 && Math.abs(P[3 * k + 2]) < 1e-3, `${name}: greys on the axis`);
  }
}

console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
