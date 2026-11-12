// 1D test signals of 256 samples for the scale1d demo (no DOM). The micrograph line
// profile is read in main.ts.
import { rng } from "../lib/random";

export const N = 256;

/** Box bumps of half-width a (2a pixels, edges at c ± a): the normalised LoG at the centre peaks at σ = a. */
export const BUMPS = [{ c: 40, a: 2 }, { c: 100, a: 6 }, { c: 180, a: 18 }];

export const SIGNALS = [
  { id: "bumps", label: "three bumps" },
  { id: "stepramp", label: "step and ramp" },
  { id: "boxdemo", label: "line in a dark band" },
] as const;

export function makeSignal(id: string, noise: number, seed = 1): Float64Array {
  const f = new Float64Array(N);
  for (let x = 0; x < N; x++) {
    let v = 40;
    if (id === "stepramp") v = x < 70 ? 40 : x < 140 ? 160 : x < 200 ? 160 - 100 * (x - 140) / 60 : 60;
    else if (id === "boxdemo") { v = 60; if (x >= 108 && x <= 148) v -= 40; if (x >= 124 && x <= 131) v += 90; }
    else for (const b of BUMPS) if (x >= b.c - b.a && x < b.c + b.a) v += 100;
    f[x] = v;
  }
  if (noise > 0) { const r = rng(seed); for (let x = 0; x < N; x++) f[x] += noise * r.norm(); }
  return f;
}
