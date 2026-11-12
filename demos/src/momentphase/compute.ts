// Pure computations of the momentphase demo (no DOM), shared with the scratch scripts that
// compute the numbers quoted in the moment-invariants exercise.
import { Poly, Pt } from "../lib/shapecanvas";
import { rawMoments, centralFromRaw, normalisedMoments, complexMoments, flipV, idx, polygonCoverageEvenOdd } from "../lib/moments";

export const SIZE = 64;
const N = 4;

/** An irregular blob with no symmetry (all low-order complex moments non-zero). */
export function blob(cx: number, cy: number, s: number, n = 120): Poly {
  return Array.from({ length: n }, (_, i) => {
    const t = 2 * Math.PI * i / n;
    const r = s * (1 + 0.28 * Math.cos(t - 2.6) + 0.16 * Math.cos(2 * t + 0.9) + 0.12 * Math.cos(3 * t - 0.4));
    return [cx + r * Math.cos(t), cy + r * Math.sin(t)] as Pt;
  });
}

/** A shape with 3-fold rotational symmetry (a rounded triangle with dents). */
export function trefoil(cx: number, cy: number, s: number, n = 120): Poly {
  return Array.from({ length: n }, (_, i) => {
    const t = 2 * Math.PI * i / n, r = s * (1 + 0.3 * Math.cos(3 * t + 0.5));
    return [cx + r * Math.cos(t), cy + r * Math.sin(t)] as Pt;
  });
}

export const PAIRS: { p: number; q: number; label: string; speed: string }[] = [
  { p: 2, q: 0, label: "c₂₀", speed: "turns by 2α" },
  { p: 1, q: 1, label: "c₁₁", speed: "stays (p = q)" },
  { p: 3, q: 0, label: "c₃₀", speed: "turns by 3α" },
  { p: 2, q: 1, label: "c₂₁", speed: "turns by α" },
  { p: 1, q: 2, label: "c₁₂", speed: "turns by −α" },
];

/** Complex moments (of η, y up) of the anti-aliased mask of the polygons; index 5 = c30·c12³. */
export function phaseMoments(polys: Poly[]): { re: number; im: number }[] {
  const f = polygonCoverageEvenOdd(polys, SIZE, SIZE, 4);
  const m = rawMoments(f, SIZE, SIZE, N);
  if (!(m[0] > 0)) return [];
  const eta = flipV(normalisedMoments(centralFromRaw(m, N), N), N);
  const c = complexMoments(eta, N);
  const out = PAIRS.map(({ p, q }) => ({ re: c.re[idx(N, p, q)], im: c.im[idx(N, p, q)] }));
  const c30 = out[2], c12 = out[4];
  const mul = (a: { re: number; im: number }, b: { re: number; im: number }) => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re });
  out.push(mul(c30, mul(c12, mul(c12, c12))));
  return out;
}

