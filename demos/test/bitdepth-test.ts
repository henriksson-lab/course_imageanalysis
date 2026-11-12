// Tests for the bit-depth day demos (quantise, camerabits, window16, texformats,
// floatbits). Run without the rest of the suite:
//   node_modules/.bin/esbuild demos/test/bitdepth-test.ts --bundle --platform=node | node
// Reference bit patterns (REF) were computed with NumPy (float16, float32) and exact
// rational arithmetic in Python (bfloat16, round to nearest even).
import * as fp from "../src/floatbits/fp";
import { pixels, sumTable } from "../src/floatbits/sum";
import { quantise, stats, ramp } from "../src/quantise/quant";
import { subtractU16, windowLevel } from "../src/window16/data";
import { storeTexel, uploadData, textureBytes, srgbEncode, srgbDecode } from "../src/texformats/pack";
import { gain, measureFlat, noiseModel } from "../src/camerabits/sensor";
import { rng } from "../src/lib/ui";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}

// value, float16 bits, float32 bits, bfloat16 bits (null: below the bfloat16 normal range, not tested)
const REF: [number, number, number, number | null][] = [
  [7723.847668789957, 28555, 1173446344, 17905],
  [0.00012588657963379392, 2080, 956563564, 14596],
  [0.18273832813718957, 12761, 1044062146, 15931],
  [-292588561.1094869, 64512, 3448472673, 52620],
  [94760248.230379, 31744, 1286913447, 19637],
  [3586.7464881261517, 27393, 1163930610, 17760],
  [-2.79331747511087, 49558, 3224552887, 49203],
  [7.735130547930937e-05, 1298, 950155178, 14498],
  [4.6420026211072863e-07, 8, 888747827, 13561],
  [42853230.21342634, 31744, 1277393116, 19491],
  [0.054050316810762764, 10987, 1029530589, 15709],
  [1.1605439654972496e-07, 2, 871971220, 13305],
  [-102.89219987845294, 54894, 3268266190, 49870],
  [-10433.393728844469, 61720, 3324183955, 50723],
  [-44.28156202217143, 53641, 3257999442, 49713],
  [10629649.08091873, 31744, 1260532241, 19234],
  [3.9897102680702876e-10, 0, 802903598, 12251],
  [-0.9177154049783234, 47959, 3211456358, 49003],
  [-0.15372618391160436, 45291, 3189598822, 48669],
  [-6.419354369849161e-09, 32768, 2984022333, 45533],
  [0.1, 11878, 1036831949, 15821],
  [0.3333333333333333, 13653, 1051372203, 16043],
  [2049.0, 26624, 1157632000, 17664],
  [65519.0, 31743, 1199566592, 18304],
  [65520.0, 31744, 1199566848, 18304],
  [1e-07, 2, 869711765, 13271],
  [3e-08, 1, 855693657, 13057],
  [-0.0, 32768, 2147483648, 32768],
  [8.940696716308594e-08, 2, 868220928, 13248],
];

// ---- float formats ----
for (const [v, h, f, b] of REF) {
  check(fp.encode(fp.F16, v) === h, `float16 encode ${v}: ${fp.encode(fp.F16, v)} vs ${h}`);
  check(fp.encode(fp.F32, v) === f, `float32 encode ${v}: ${fp.encode(fp.F32, v)} vs ${f}`);
  if (b !== null) check(fp.encode(fp.BF16, v) === b, `bfloat16 encode ${v}: ${fp.encode(fp.BF16, v)} vs ${b}`);
}
// every float16 pattern decodes and re-encodes to itself (NaN excepted)
for (let bits = 0; bits < 65536; bits++) {
  const v = fp.decode(fp.F16, bits);
  if (Number.isNaN(v)) continue;
  check(fp.encode(fp.F16, v) === bits, `float16 round trip ${bits}`);
  check(Object.is(fp.roundF16(v), v), `roundF16 keeps ${v}`);
}
// fast float16 rounding = exact rounding, on random values and on midpoints
{
  const r = rng(3);
  for (let i = 0; i < 200000; i++) {
    const x = (r.unif() - 0.5) * 2 ** (Math.floor(r.unif() * 44) - 28);
    check(Object.is(fp.roundF16(x), fp.round(fp.F16, x)), `roundF16(${x})`);
  }
  for (let bits = 0; bits < 31743; bits++) {
    const mid = (fp.decode(fp.F16, bits) + fp.decode(fp.F16, bits + 1)) / 2;
    check(fp.roundF16(mid) === fp.round(fp.F16, mid), `roundF16 midpoint ${mid}`);
  }
}
// float32 against the hardware (DataView)
{
  const r = rng(5), dv = new DataView(new ArrayBuffer(4));
  for (let i = 0; i < 100000; i++) {
    const x = (r.unif() - 0.5) * 2 ** (Math.floor(r.unif() * 300) - 150);
    dv.setFloat32(0, x);
    check(fp.encode(fp.F32, x) === dv.getUint32(0), `float32 vs DataView ${x}`);
  }
}
check(fp.maxFinite(fp.F16) === 65504, "float16 max");
check(fp.exactIntLimit(fp.F16) === 2048 && fp.exactIntLimit(fp.F32) === 2 ** 24, "exact integer limits");
check(fp.ulp(fp.F16, 1000) === 0.5 && fp.ulp(fp.F16, 3000) === 2, "float16 ulp");
check(fp.exactDecimal(fp.round(fp.F16, 0.1)) === "0.0999755859375", "exact decimal of float16(0.1)");
check(fp.round(fp.F16, 2049) === 2048 && fp.round(fp.F16, 2051) === 2052, "ties to even at 2049, 2051");

// ---- summation ----
{
  const px = pixels(2 ** 16, 1000, 50);
  const t = sumTable(px, 4);
  const last = t.ks.length - 1;
  check(t.exact[last] === Array.from(px).reduce((a, b) => a + b, 0), "exact sum");
  check(Math.abs(t.res[2][last] - t.exact[last]) <= 2 ** -23 * t.exact[last] * 2, "Kahan within 2 ulp");
  check(Math.abs(t.res[1][last] - t.exact[last]) / t.exact[last] < 1e-6, "pairwise float32 accurate");
}

// ---- quantisation ----
{
  const N = 128, x = ramp(N);
  for (let b = 2; b <= 8; b++) {
    const q = quantise(x, N, N, b, "none");
    const L = 2 ** b - 1;
    let ok = true;
    for (let i = 0; i < x.length; i++) if (Math.abs(q[i] - x[i]) > 0.5 / L + 1e-12) ok = false;
    check(ok, `|error| <= Delta/2 at b = ${b}`);
    const s = stats(x, q, b);
    check(Math.abs(s.rms / s.theory - 1) < 0.02, `RMS error = Delta/sqrt(12) at b = ${b}: ${s.rms} vs ${s.theory}`);
    const qd = quantise(x, N, N, b, "random");
    const sd = stats(x, qd, b);
    check(Math.abs(sd.rms / (s.theory * Math.SQRT2) - 1) < 0.03, `random dither RMS = Delta/sqrt(6) at b = ${b}`);
  }
}

// ---- integer arithmetic and windows ----
check(subtractU16(120, 150, "wrap") === 65506, "uint16 wrap");
check(subtractU16(120, 150, "saturate") === 0, "uint16 saturate");
check(subtractU16(120, 150, "float") === -30, "float subtract");
check(windowLevel(4095, 0, 65535) === 16 && windowLevel(40, -160, 240) === 128, "window levels");

// ---- texture packing ----
check(storeTexel("RGB565", 1).stored.startsWith("0xFFFF"), "RGB565 white");
check((uploadData("RGB565", [0.5]) as Uint16Array)[0] === ((16 << 11) | (32 << 5) | 16), "RGB565 grey 0.5");
check((uploadData("RGB10_A2", [1]) as Uint32Array)[0] === 0xffffffff, "RGB10_A2 white");
check((uploadData("R16F", [0.1]) as Uint16Array)[0] === 11878, "R16F 0.1");
check(Math.abs(srgbDecode(srgbEncode(0.2)) - 0.2) < 1e-12, "sRGB round trip");
check(textureBytes("RGBA8", 2048, 2048, false) === 16777216, "RGBA8 2048^2");
check(textureBytes("RGBA8", 4, 4, true) === 4 * (16 + 4 + 1), "mip chain 4x4");

// ---- camera ----
{
  const c = { fwc: 30000, read: 1.5, bits: 16 };
  check(Math.abs(gain(c) - 30000 / 65535) < 1e-12, "gain");
  const m = measureFlat(c, 1000, 20000, 9);
  const th = noiseModel(c, 1000).total;
  check(Math.abs(m.sd / th - 1) < 0.03, `measured noise ${m.sd} vs model ${th}`);
}

console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
