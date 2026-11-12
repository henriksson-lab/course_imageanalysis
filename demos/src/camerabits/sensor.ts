// A simple camera model, used by the camerabits demo and its tests.
//   photons -> electrons N ~ Poisson(lambda), capped at the full-well capacity
//   + read noise (Gaussian, sigma_read electrons)
//   + offset (bias) so that noise below zero is not clipped
//   -> ADC: ADU = round((electrons + offset) / g), clipped to 0 .. 2^b - 1
// The gain g (electrons per ADU) maps the full well onto the ADC range: g = FWC / (2^b - 1).
import { rng } from "../lib/ui";

export interface Camera {
  fwc: number;      // full-well capacity, electrons
  read: number;     // read noise, electrons RMS
  bits: number;     // ADC bits
}

export const gain = (c: Camera) => c.fwc / (2 ** c.bits - 1);
/** Offset in electrons: 5 read-noise SDs, at least 2 ADU. */
export const offsetE = (c: Camera) => Math.max(5 * c.read, 2 * gain(c));
export const offsetADU = (c: Camera) => Math.round(offsetE(c) / gain(c));

/** Noise (electrons RMS) predicted at a mean signal of S electrons. */
export function noiseModel(c: Camera, S: number) {
  const shot = Math.sqrt(S), read = c.read, quant = gain(c) / Math.sqrt(12);
  return { shot, read, quant, total: Math.sqrt(S + read * read + quant * quant) };
}

export function summary(c: Camera) {
  const g = gain(c);
  const dr = c.fwc / c.read;
  return {
    gain: g,
    quantNoise: g / Math.sqrt(12),
    dynRange: dr, dynRangeDb: 20 * Math.log10(dr), dynRangeBits: Math.log2(dr),
    maxSnr: Math.sqrt(c.fwc), maxSnrDb: 20 * Math.log10(Math.sqrt(c.fwc)),
    // smallest ADC bit depth with a step no larger than the read noise (g <= sigma_read)
    bitsNeeded: Math.ceil(Math.log2(c.fwc / c.read + 1)),
  };
}

/** Poisson sample: Knuth's product method for small means, a rounded normal above 30. */
export function poisson(lambda: number, r: { unif: () => number; norm: () => number }): number {
  if (lambda <= 0) return 0;
  if (lambda < 30) {
    const L = Math.exp(-lambda);
    let k = 0, p = 1;
    do { k++; p *= r.unif(); } while (p > L);
    return k - 1;
  }
  return Math.max(0, Math.round(lambda + Math.sqrt(lambda) * r.norm()));
}

/** Simulate one exposure: `lambda[i]` mean electrons per pixel. Returns ADU values. */
export function expose(c: Camera, lambda: ArrayLike<number>, seed: number): Uint16Array | Uint32Array {
  const r = rng(seed);
  const g = gain(c), off = offsetE(c), max = 2 ** c.bits - 1;
  const out = c.bits <= 16 ? new Uint16Array(lambda.length) : new Uint32Array(lambda.length);
  for (let i = 0; i < lambda.length; i++) {
    const e = Math.min(c.fwc, poisson(lambda[i], r)) + c.read * r.norm() + off;
    out[i] = Math.min(max, Math.max(0, Math.round(e / g)));
  }
  return out;
}

/** Measured noise in electrons of a flat patch of n pixels at mean S electrons. */
export function measureFlat(c: Camera, S: number, n: number, seed: number): { mean: number; sd: number } {
  const lam = new Float64Array(n).fill(S);
  const adu = expose(c, lam, seed);
  const g = gain(c);
  let m = 0;
  for (let i = 0; i < n; i++) m += adu[i];
  m /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (adu[i] - m) ** 2;
  v /= n - 1;
  return { mean: (m - offsetADU(c)) * g, sd: Math.sqrt(v) * g };
}

/** The signal levels (electrons) of the photon-transfer points: 1 .. FWC, log-spaced. */
export function ptcLevels(c: Camera, k = 14): number[] {
  const out: number[] = [];
  for (let i = 0; i < k; i++) out.push(Math.round(10 ** (Math.log10(c.fwc) * i / (k - 1)) * 100) / 100);
  return out;
}
