// Model of the mrf1d demo (no DOM): 1D test profiles with seeded Gaussian noise, and the
// presets. The smoothing itself is in lib/smooth.ts.
import { rng } from "../lib/random";
import { Prior } from "../lib/smooth";

export const N1 = 200;
export type Profile = "mixed" | "steps" | "cell";

/** A step (0 -> 2 at 40), a ramp (2 -> 0 over 80..140), a narrow peak (4 samples of height 2 at 166..169). */
export function mixedProfile(): Float64Array {
  return Float64Array.from({ length: N1 }, (_, i) =>
    i < 40 ? 0 : i < 80 ? 2 : i < 140 ? 2 - 2 * (i - 80) / 60 : i >= 166 && i < 170 ? 2 : 0);
}

/** Plateaus 0, 2, 1, 3, 1.5 of 40 samples each. */
export function stepsProfile(): Float64Array {
  const lv = [0, 2, 1, 3, 1.5];
  return Float64Array.from({ length: N1 }, (_, i) => lv[Math.min(4, Math.floor(i / 40))]);
}

/** Noisy observation y = truth + sd * N(0, 1). */
export function noisy(truth: Float64Array, sd: number, seed = 1): Float64Array {
  const r = rng(seed);
  return Float64Array.from(truth, v => v + sd * r.norm());
}

export function rmse(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s / a.length);
}

export interface Preset { id: string; label: string; profile: Profile; sd: number; prior: Prior; lambda: number; delta: number; view?: [number, number] }
export const PRESETS: Preset[] = [
  { id: "quad", label: "quadratic blurs the step", profile: "mixed", sd: 0.3, prior: "quadratic", lambda: 10, delta: 0.1 },
  { id: "tv", label: "TV keeps the step", profile: "mixed", sd: 0.3, prior: "tv", lambda: 1, delta: 0.1 },
  { id: "stair", label: "TV staircases the ramp", profile: "mixed", sd: 0.3, prior: "tv", lambda: 3, delta: 0.1, view: [60, 160] },
  { id: "peak", label: "TV shrinks the narrow peak", profile: "mixed", sd: 0.3, prior: "tv", lambda: 2, delta: 0.1, view: [140, 199] },
  { id: "huber", label: "Huber in between", profile: "mixed", sd: 0.3, prior: "huber", lambda: 5, delta: 0.1 },
  { id: "none", label: "λ = 0", profile: "mixed", sd: 0.3, prior: "none", lambda: 1, delta: 0.1 },
];
