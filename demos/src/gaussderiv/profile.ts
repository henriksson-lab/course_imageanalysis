// 1D test profiles of 256 samples for the gaussderiv demo (no DOM): a step, a ramp, a thin
// line and two close steps, with seeded Gaussian noise. The micrograph line profile is read
// in main.ts (it needs the image).
import { rng } from "../lib/random";

export const N = 256;

export const PROFILES = [
  { id: "mixed", label: "step, ramp, line, two steps" },
  { id: "step", label: "one step" },
  { id: "line", label: "thin line" },
  { id: "close", label: "two close steps" },
] as const;
export type ProfileId = typeof PROFILES[number]["id"];

/** Noise-free profile. "mixed": flat 40 → step to 160 at x = 48 → ramp down to 60 over 96..128 →
 *  thin bright line (2 px, +80) at 160–161 → steps 60 → 110 → 160 at x = 200 and 206. */
export function cleanProfile(id: string): Float64Array {
  const f = new Float64Array(N);
  for (let x = 0; x < N; x++) {
    let v: number;
    switch (id) {
      case "step": v = x < 128 ? 50 : 170; break;
      case "line": v = 60 + (x === 127 || x === 128 ? 100 : 0); break;
      case "close": v = x < 124 ? 50 : x < 132 ? 110 : 170; break;
      default:
        v = x < 48 ? 40 : x < 96 ? 160 : x < 128 ? 160 - 100 * (x - 96) / 32 : 60;
        if (x === 160 || x === 161) v += 80;
        if (x >= 200) v += 50;
        if (x >= 206) v += 50;
    }
    f[x] = v;
  }
  return f;
}

/** Profile plus Gaussian noise of standard deviation `noise` (seeded). */
export function makeProfile(id: string, noise: number, seed = 1): Float64Array {
  const f = cleanProfile(id), r = rng(seed);
  for (let x = 0; x < N; x++) f[x] += noise * r.norm();
  return f;
}
