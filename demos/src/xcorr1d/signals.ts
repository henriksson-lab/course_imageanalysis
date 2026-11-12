// Signals for the xcorr1d demo (no DOM). A long base signal s(x) is defined on an extended
// range; image A is s on 0..N−1, image B is a window of length L that matches A when it is
// placed at the true shift t:  B(x) = gain · s(x + t) + offset + ramp · x/L + noise.
// A may also get a bright blob that B does not contain (the "CC is fooled" case).
import { rng } from "../lib/random";

export type Kind = "profile" | "bumps" | "step" | "periodic" | "hand";

export interface Sim1D {
  kind: Kind;
  N: number;        // length of A
  L: number;        // length of B
  t: number;        // true shift (where B goes on A)
  gain: number;
  offset: number;
  ramp: number;     // brightness ramp added across B (grey levels from start to end)
  noise: number;    // sd of Gaussian noise (on both)
  blob: number;     // height of a bright blob added to A only (0: none)
  seed: number;
  circular?: boolean; // B = A rolled around by t (an exact circular shift; L = N)
}

export const PROFILE: Sim1D = { kind: "bumps", N: 64, L: 64, t: 9, gain: 1, offset: 0, ramp: 0, noise: 0, blob: 0, seed: 3 };

const MARGIN = 128;

/** Base signal on −MARGIN .. N + MARGIN − 1 (index i ↔ x = i − MARGIN). `row` (0..255 values) feeds "profile". */
export function baseSignal(kind: Kind, N: number, seed: number, row: ArrayLike<number> | null = null): (x: number) => number {
  const r = rng(seed);
  const len = N + 2 * MARGIN, s = new Float64Array(len);
  if (kind === "profile" && row) {
    // the micrograph row, mirrored at its ends so any length fits, centred on A
    const n = row.length, off = Math.floor((n - N) / 2);
    for (let i = 0; i < len; i++) {
      let x = i - MARGIN + off;
      const p = 2 * n;
      x = ((x % p) + p) % p;
      if (x >= n) x = p - 1 - x;
      s[i] = row[x];
    }
  } else if (kind === "step") {
    for (let i = 0; i < len; i++) { const x = i - MARGIN; s[i] = x < N * 0.45 ? 60 : 170; }
  } else if (kind === "periodic") {
    for (let i = 0; i < len; i++) { const x = i - MARGIN; s[i] = 110 + 60 * Math.cos(2 * Math.PI * x / 12) + 20 * Math.cos(2 * Math.PI * x / 6); }
  } else if (kind === "hand") {
    for (let i = 0; i < len; i++) s[i] = 0;
  } else {
    // bumps (also the fallback for "profile" without a micrograph): seeded Gaussian bumps
    for (let i = 0; i < len; i++) s[i] = 50;
    for (let k = 0; k < Math.round(len / 9); k++) {
      const c = -MARGIN + r.unif() * len, h = 30 + 110 * r.unif(), w = 1.5 + 3.5 * r.unif();
      for (let i = 0; i < len; i++) s[i] += h * Math.exp(-(((i - MARGIN) - c) ** 2) / (2 * w * w));
    }
  }
  return (x: number) => s[Math.max(0, Math.min(len - 1, Math.round(x) + MARGIN))];
}

export interface Signals { f: Float64Array; g: Float64Array }

/** A and B for the settings (the by-hand example ignores most settings). */
export function simulate1d(o: Sim1D, row: ArrayLike<number> | null = null): Signals {
  if (o.kind === "hand") {
    const f = Float64Array.from([0, 1, 3, 1, 0, 2, 6, 2]), g = Float64Array.from([1, 3, 1]);
    return { f, g };
  }
  const s = baseSignal(o.kind, o.N, o.seed, row), r = rng(o.seed * 7919 + 17);
  const f = new Float64Array(o.N), g = new Float64Array(o.L);
  for (let x = 0; x < o.N; x++) f[x] = s(x);
  if (o.blob) { const c = blobCentre(o); for (let x = 0; x < o.N; x++) f[x] += o.blob * Math.exp(-((x - c) ** 2) / (2 * 3.5 * 3.5)); }
  for (let x = 0; x < o.L; x++) {
    const v = o.circular ? f[(((x + o.t) % o.N) + o.N) % o.N] : s(x + o.t);
    g[x] = o.gain * v + o.offset + o.ramp * (o.L > 1 ? x / (o.L - 1) : 0);
  }
  if (o.noise > 0) { for (let x = 0; x < o.N; x++) f[x] += o.noise * r.norm(); for (let x = 0; x < o.L; x++) g[x] += o.noise * r.norm(); }
  return { f, g };
}

/** Where the bright blob goes: as far as possible from the part of A that B matches. */
export function blobCentre(o: Sim1D): number {
  const mid = o.t + o.L / 2;
  return mid < o.N / 2 ? Math.min(o.N - 6, mid + o.N / 2) : Math.max(5, mid - o.N / 2);
}
