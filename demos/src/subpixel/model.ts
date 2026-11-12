// Model of the subpixel demo (no DOM): a 1D signal A and a copy B shifted by a non-integer
// amount through the Fourier phase ramp (so the truth is exact): B(x) = A(x + s). The
// correlation peak is located to a fraction of a pixel by a parabola through three samples,
// by the centroid of three samples, and by the upsampled DFT (the inverse DFT evaluated on a
// fine grid near the peak).
import { rng } from "../lib/random";
import { correlate1d, fourierShift1d, parabolaOffset, centroidOffset, upsampledPeak1d, idftAt, signed, Complex } from "../lib/registration";

export const N = 64, INT = 3;
export type SubSignal = "bumps" | "profile";
export interface SubSettings { signal: SubSignal; noise: number; U: number; method: "cc" | "phase"; frac: number }

/** The base signal (bumps; or 64 samples of a micrograph row if given). */
export function signalOf(kind: SubSignal, row: ArrayLike<number> | null = null): Float64Array {
  if (kind === "profile" && row) return Float64Array.from({ length: N }, (_, x) => row[96 + x]);
  return Float64Array.from({ length: N }, (_, x) => 60 + 120 * Math.exp(-((x - 24) ** 2) / 18) + 70 * Math.exp(-((x - 36) ** 2) / 8) + 50 * Math.exp(-((x - 47) ** 2) / 30));
}

export interface Estimate {
  shift: number;            // the truth
  c: Float64Array;          // correlation (circular, index = shift mod N)
  R: Complex;
  t0: number;               // integer peak (signed)
  parabola: number; centroid: number; up: number;
  curve: { xs: number[]; cs: number[] };   // the upsampled-DFT curve near the peak
}

export function estimate(a: Float64Array, s: SubSettings, frac = s.frac, seed = 9): Estimate {
  const shift = INT + frac;
  const b = fourierShift1d(a, -shift), r = rng(seed);
  const an = Float64Array.from(a, v => v + s.noise * r.norm()), bn = Float64Array.from(b, v => v + s.noise * r.norm());
  const ma = an.reduce((x, y) => x + y, 0) / N, mb = bn.reduce((x, y) => x + y, 0) / N;
  const res = correlate1d(an.map(v => v - ma), bn.map(v => v - mb), { p: s.method === "phase" ? 1 : 0, eps: 1e-6 });
  const c = res.c;
  let i0 = 0;
  for (let i = 1; i < N; i++) if (c[i] > c[i0]) i0 = i;
  const t0 = signed(i0, N), at = (d: number) => c[(i0 + d + N) % N];
  const up = upsampledPeak1d(res.R, t0, s.U);
  const xs: number[] = [], cs: number[] = [];
  for (let x = t0 - 2.5; x <= t0 + 2.5 + 1e-9; x += 0.02) { xs.push(x); cs.push(idftAt(res.R, x)); }
  return {
    shift, c, R: res.R, t0,
    parabola: t0 + parabolaOffset(at(-1), at(0), at(1)),
    centroid: t0 + centroidOffset(at(-1), at(0), at(1)),
    up: up.x, curve: { xs, cs },
  };
}

/** Errors (estimate − truth) of the three methods over true fractions 0 … 1. */
export function sweep(o: SubSettings & { steps?: number }, row: ArrayLike<number> | null = null): { fracs: number[]; parabola: number[]; centroid: number[]; up: number[] } {
  const a = signalOf(o.signal, row), n = o.steps ?? 41, out = { fracs: [] as number[], parabola: [] as number[], centroid: [] as number[], up: [] as number[] };
  for (let k = 0; k < n; k++) {
    const f = k / (n - 1), e = estimate(a, o, f);
    out.fracs.push(f); out.parabola.push(e.parabola - e.shift); out.centroid.push(e.centroid - e.shift); out.up.push(e.up - e.shift);
  }
  return out;
}
