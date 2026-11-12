// 1D flooding helpers for the flood1d demo (no DOM; tested in demos/test/watershed-test.ts).
// The basins themselves come from lib/watershed.ts run on a 1 × n image.
import { rng } from "../lib/random";

/** Water at a given level: for every sample the label of the basin whose water covers it
 *  (connected to one of that basin's marker samples through samples ≤ level inside the basin), else 0. */
export function basinsAtLevel(f: ArrayLike<number>, lab: ArrayLike<number>, markers: ArrayLike<number>, level: number): Int32Array {
  const n = f.length, out = new Int32Array(n);
  for (let s = 0; s < n; s++) {
    if (!markers[s] || f[s] > level || out[s]) continue;
    const L = lab[s];
    out[s] = L;
    for (let i = s - 1; i >= 0 && lab[i] === L && f[i] <= level; i--) out[i] = L;
    for (let i = s + 1; i < n && lab[i] === L && f[i] <= level; i++) out[i] = L;
  }
  return out;
}

/** Dams: between samples at and at + 1 where the basin label changes; height = the pass. */
export function dams(f: ArrayLike<number>, lab: ArrayLike<number>): { at: number; height: number }[] {
  const out: { at: number; height: number }[] = [];
  for (let i = 0; i + 1 < f.length; i++) if (lab[i] !== lab[i + 1] && lab[i] > 0 && lab[i + 1] > 0) out.push({ at: i, height: Math.max(f[i], f[i + 1]) });
  return out;
}

/** |f'| by central differences, |f(i+1) − f(i−1)| / 2, with the end samples repeated. */
export function derivative1d(f: ArrayLike<number>): number[] {
  const n = f.length;
  return Array.from({ length: n }, (_, i) => Math.abs(f[Math.min(n - 1, i + 1)] - f[Math.max(0, i - 1)]) / 2);
}

const r1 = (v: number) => Math.round(v * 10) / 10;
const bump = (x: number, c: number, s: number) => Math.exp(-0.5 * ((x - c) / s) ** 2);

/** A profile across two dark nuclei on a bright background (60 samples, 0..10). */
export function twoCells(noise = 0, seed = 5): number[] {
  const r = rng(seed);
  return Array.from({ length: 60 }, (_, x) => {
    const v = 8.6 - 6.2 * bump(x, 19, 5.5) - 5.2 * bump(x, 40, 6) + noise * r.norm();
    return r1(Math.max(0, Math.min(10, v)));
  });
}

/** Three flat regions with two edges, slightly blurred (for flooding |f'|). */
export function steps3(noise = 0.12, seed = 3): number[] {
  const r = rng(seed);
  return Array.from({ length: 60 }, (_, x) => {
    const s = (t: number) => 1 / (1 + Math.exp(-(x - t) / 0.9));
    const v = 2 + 5.5 * s(21) - 3 * s(41) + noise * r.norm();
    return r1(Math.max(0, Math.min(10, v)));
  });
}

/** Two V-shaped basins joined by a flat ridge of 9 samples at height 6. */
export function plateau(): number[] {
  const left = [8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6];
  const ridge = Array(9).fill(6);
  const right = [5, 4, 3, 2, 2, 1, 2, 3, 4, 5, 6, 7, 8];
  return [...left, ...ridge, ...right];
}

/** The 12 integers of the by-hand exercise (also on the morphology day). */
export const TWELVE = [7, 1, 4, 3, 4, 6, 5, 1, 6, 4, 2, 3];
