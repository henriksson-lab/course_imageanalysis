// Binary floating-point formats (IEEE 754 style) with a sign bit, `e` exponent bits and
// `m` stored mantissa bits, encoded and decoded exactly in double precision. Used by the
// floatbits demo (bit explorer, summation), texformats (half floats for R16F) and the
// tests. Rounding is round-to-nearest, ties to even, as in IEEE 754.

export interface FloatFormat { name: string; e: number; m: number; }
export const F32: FloatFormat = { name: "float32", e: 8, m: 23 };
export const F16: FloatFormat = { name: "float16", e: 5, m: 10 };
export const BF16: FloatFormat = { name: "bfloat16", e: 8, m: 7 };
export const FORMATS = [F32, F16, BF16];

export const bias = (f: FloatFormat) => 2 ** (f.e - 1) - 1;
export const nbits = (f: FloatFormat) => 1 + f.e + f.m;
const expMax = (f: FloatFormat) => 2 ** f.e - 1;   // all ones: Inf / NaN

/** Round half to even (x is a double with an exactly representable fractional part). */
function roundEven(q: number): number {
  const f = Math.floor(q), d = q - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** Bit pattern (as an unsigned integer < 2^32) of the value of `x` rounded to format f. */
export function encode(f: FloatFormat, x: number): number {
  const sh = f.e + f.m;
  if (Number.isNaN(x)) return (expMax(f) * 2 ** f.m + 2 ** (f.m - 1));   // a quiet NaN
  const s = x < 0 || Object.is(x, -0) ? 1 : 0;
  const sign = s * 2 ** sh;
  const a = Math.abs(x);
  if (a === Infinity) return sign + expMax(f) * 2 ** f.m;
  if (a === 0) return sign;
  const b = bias(f);
  let E = Math.floor(Math.log2(a));
  if (2 ** E > a) E--;               // guard against log2 rounding
  if (2 ** (E + 1) <= a) E++;
  if (E < 1 - b) {
    // subnormal: a = q * 2^(1 - b - m); q may round up to 2^m = the smallest normal
    const q = roundEven(a / 2 ** (1 - b - f.m));
    return sign + q;
  }
  let q = roundEven(a / 2 ** (E - f.m));   // in [2^m, 2^(m+1)]
  if (q === 2 ** (f.m + 1)) { q /= 2; E++; }
  if (E > b) return sign + expMax(f) * 2 ** f.m;   // overflow: Inf
  return sign + (E + b) * 2 ** f.m + (q - 2 ** f.m);
}

export interface Fields { s: number; E: number; F: number; }
export function fields(f: FloatFormat, bits: number): Fields {
  const F = bits % 2 ** f.m;
  const E = Math.floor(bits / 2 ** f.m) % 2 ** f.e;
  const s = Math.floor(bits / 2 ** (f.e + f.m)) % 2;
  return { s, E, F };
}

/** The value of a bit pattern. */
export function decode(f: FloatFormat, bits: number): number {
  const { s, E, F } = fields(f, bits);
  const sg = s ? -1 : 1;
  if (E === expMax(f)) return F ? NaN : sg * Infinity;
  if (E === 0) return sg * F * 2 ** (1 - bias(f) - f.m);
  return sg * (2 ** f.m + F) * 2 ** (E - bias(f) - f.m);
}

/** Round x to the nearest value of format f. */
export const round = (f: FloatFormat, x: number) => decode(f, encode(f, x));

export type FloatClass = "zero" | "subnormal" | "normal" | "infinity" | "NaN";
export function classify(f: FloatFormat, bits: number): FloatClass {
  const { E, F } = fields(f, bits);
  if (E === expMax(f)) return F ? "NaN" : "infinity";
  if (E === 0) return F ? "subnormal" : "zero";
  return "normal";
}

/** Spacing from |x| to the next larger representable value (one ulp), for finite x. */
export function ulp(f: FloatFormat, x: number): number {
  const a = Math.abs(round(f, x));
  const bits = encode(f, a);
  const next = decode(f, bits + 1);
  if (!Number.isFinite(next)) return a - decode(f, bits - 1);   // at the largest finite value
  return next - a;
}

export const maxFinite = (f: FloatFormat) => decode(f, (expMax(f) - 1) * 2 ** f.m + 2 ** f.m - 1);
export const minNormal = (f: FloatFormat) => 2 ** (1 - bias(f));
export const minSubnormal = (f: FloatFormat) => 2 ** (1 - bias(f) - f.m);
/** Machine epsilon: the gap between 1 and the next larger number. */
export const epsilon = (f: FloatFormat) => 2 ** -f.m;
/** Every integer up to 2^(m+1) is exact; 2^(m+1) + 1 is the first that is not. */
export const exactIntLimit = (f: FloatFormat) => 2 ** (f.m + 1);

export function bitString(f: FloatFormat, bits: number): string {
  return bits.toString(2).padStart(nbits(f), "0");
}
export function hexString(f: FloatFormat, bits: number): string {
  return "0x" + bits.toString(16).toUpperCase().padStart(nbits(f) / 4, "0");
}

/** The exact decimal expansion of a finite double; after `maxDigits` significant digits
 *  the rest is cut and marked with "…" (integers are never cut). */
export function exactDecimal(x: number, maxDigits = 40): string {
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return Object.is(x, -0) ? "-0" : "0";
  const sg = x < 0 ? "-" : "";
  let a = Math.abs(x), k = 0;
  while (!Number.isInteger(a)) { a *= 2; k--; }   // a * 2^k, a an integer
  if (k === 0) return sg + BigInt(a).toString();
  const digits = (BigInt(a) * 5n ** BigInt(-k)).toString();   // value = digits / 10^(-k)
  const point = digits.length + k;                             // digits before the point
  let frac = point <= 0 ? "0".repeat(-point) + digits : digits.slice(point);
  const intPart = point <= 0 ? "0" : digits.slice(0, point);
  const lead = point <= 0 ? -point : 0;                        // zeros that are not significant
  const room = Math.max(1, maxDigits - (point > 0 ? point : 0) + lead);
  const cut = frac.length > room;
  if (cut) frac = frac.slice(0, room);
  return sg + intPart + "." + frac + (cut ? "…" : "");
}

// ---- summation ------------------------------------------------------------------

/** Rounding function of a format: Math.fround for float32 (fast), else the generic one. */
export function rounder(f: FloatFormat): (x: number) => number {
  if (f === F32) return Math.fround;
  if (f === F16) return roundF16;
  return (x: number) => round(f, x);
}

/** Left-to-right sum, rounding after every addition. */
export function sumNaive(xs: ArrayLike<number>, n: number, r: (x: number) => number): number {
  let s = 0;
  for (let i = 0; i < n; i++) s = r(s + r(xs[i]));
  return s;
}

/** Pairwise (cascade) sum: split in halves, sum each, add; blocks of 8 summed naively. */
export function sumPairwise(xs: ArrayLike<number>, lo: number, hi: number, r: (x: number) => number): number {
  if (hi - lo <= 8) {
    let s = 0;
    for (let i = lo; i < hi; i++) s = r(s + r(xs[i]));
    return s;
  }
  const mid = lo + Math.floor((hi - lo) / 2);
  return r(sumPairwise(xs, lo, mid, r) + sumPairwise(xs, mid, hi, r));
}

/** Kahan compensated sum: carry the rounding error of each addition to the next. */
export function sumKahan(xs: ArrayLike<number>, n: number, r: (x: number) => number): number {
  let s = 0, c = 0;
  for (let i = 0; i < n; i++) {
    const y = r(r(xs[i]) - c);
    const t = r(s + y);
    c = r(r(t - s) - y);
    s = t;
  }
  return s;
}

/** Variance (divide by n) the textbook one-pass way, E[x^2] - E[x]^2, in format r. */
export function varOnePass(xs: ArrayLike<number>, n: number, r: (x: number) => number): number {
  let s = 0, s2 = 0;
  for (let i = 0; i < n; i++) { const x = r(xs[i]); s = r(s + x); s2 = r(s2 + r(x * x)); }
  const m = r(s / n);
  return r(r(s2 / n) - r(m * m));
}

/** Variance (divide by n) in two passes: first the mean, then the squared deviations.
 *  Both sums naive, or (pairwise = true) both pairwise. */
export function varTwoPass(xs: ArrayLike<number>, n: number, r: (x: number) => number, pairwise = false): number {
  const sum = (v: ArrayLike<number>) => pairwise ? sumPairwise(v, 0, n, r) : sumNaive(v, n, r);
  const m = r(sum(xs) / n);
  const d2 = new Float64Array(n);
  for (let i = 0; i < n; i++) { const d = r(r(xs[i]) - m); d2[i] = r(d * d); }
  return r(sum(d2) / n);
}

// ---- fast float16 rounding (for summing millions of values) ----------------------
const f64 = new Float64Array(1), hiWord = new Uint32Array(f64.buffer);
const HI = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1 ? 1 : 0;   // little-endian: high word is [1]
const P2 = (k: number) => 2 ** k;

/** Round a double to the nearest float16 (ties to even), the same as round(F16, x) but
 *  fast: add and subtract 1.5 * 2^52 * spacing, which rounds in double arithmetic. */
export function roundF16(x: number): number {
  if (x === 0 || !Number.isFinite(x)) return x;
  f64[0] = x;
  let E = ((hiWord[HI] >>> 20) & 0x7ff) - 1023;
  if (E < -14) E = -14;                    // subnormal range: fixed spacing 2^-24
  const C = 1.5 * P2(52 + E - 10);
  const r = (x + C) - C;
  if (r === 0) return x < 0 ? -0 : 0;
  return Math.abs(r) > 65504 ? (r > 0 ? Infinity : -Infinity) : r;
}
