// Model of the fmellin demo (no DOM; also used by the test and the slide figures).
// Fourier–Mellin tab: image A is a 128 × 128 view of the micrograph; image B is the same
// field rotated by `angle` (degrees, clockwise on screen), scaled by `scale` about the centre
// of A and shifted by (tx, ty), with independent noise. Registration by lib/registration.ts.
// Known-centre tab: a disc around a known centre, B = A rotated about that centre; both are
// unwrapped on a polar grid and one rotational correlation (one FFT along θ per ring, summed)
// gives the angle.
import { rng } from "../lib/random";
import { Grey } from "../lib/imageops";
import { toPolar, PolarOptions } from "../lib/polar";
import { warpSimilarity, fourierMellin, FMResult, rotationalCorrelation } from "../lib/registration";

export const IMG = 256, N = 128;

export interface FMSettings {
  angle: number;            // degrees, clockwise positive
  scale: number;
  tx: number; ty: number;   // pixels
  noise: number;            // sd in grey levels (0..255), independent on A and B
  shade: number;            // uneven illumination fixed to the camera: + shade · (x + y) / (2N) on both views
  taper: boolean;
  highpass: boolean;
  both: boolean;            // test both 180° candidates
  cx: number; cy: number;   // centre of view A in the micrograph
}

export const DEFAULTS: FMSettings = {
  angle: 30, scale: 1, tx: 0, ty: 0, noise: 5, shade: 0, taper: true, highpass: true, both: true, cx: 128, cy: 128,
};

export interface Pair { a: Float64Array; b: Float64Array }

/** View A and view B = A rotated, scaled, shifted (+ noise), both N × N. */
export function makePair(img: Grey, s: FMSettings): Pair {
  const a = warpSimilarity(img, N, N, { angle: 0, scale: 1, tx: 0, ty: 0 }, { cx: s.cx, cy: s.cy, border: "mirror" });
  const b = warpSimilarity(img, N, N, { angle: s.angle * Math.PI / 180, scale: s.scale, tx: s.tx, ty: s.ty }, { cx: s.cx, cy: s.cy, border: "mirror" });
  if (s.noise > 0) {
    const ra = rng(11), rb = rng(22);
    for (let i = 0; i < N * N; i++) { a[i] += s.noise * ra.norm(); b[i] += s.noise * rb.norm(); }
  }
  if (s.shade) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const v = s.shade * (x + y) / (2 * N);
    a[y * N + x] += v; b[y * N + x] += v;
  }
  return { a, b };
}

/** Fourier–Mellin registration of the pair with the settings' options. */
export function register(p: Pair, s: FMSettings): FMResult {
  return fourierMellin(p.a, p.b, N, N, { taper: s.taper, highpass: s.highpass, both: s.both });
}

/** Angle in degrees wrapped to (−180, 180]. */
export const wrapDeg = (d: number): number => { const r = ((d % 360) + 360) % 360; return r > 180 ? r - 360 : r; };

// ---------------------------------------------------------------------------------------
// Known centre: polar unwrap and rotational correlation

export interface KCSettings {
  angle: number;            // degrees, clockwise
  noise: number;
  cx: number; cy: number;   // the known centre in the micrograph
  radius: number;
  nTheta: number;           // columns over a full turn
}

export const KC_DEFAULTS: KCSettings = { angle: 50, noise: 10, cx: 112, cy: 80, radius: 48, nTheta: 256 };

export interface KCResult {
  a: Float64Array; b: Float64Array; size: number;   // the two discs (2·radius + 1 square)
  pa: Float64Array; pb: Float64Array; nR: number;   // polar unwraps, nTheta × nR (row = radius)
  score: Float64Array;                              // rotational correlation over the angle (columns)
  angle: number; height: number;                    // detected angle (deg, sub-column) and the peak value
}

/** Disc views A and B (B rotated about the known centre), their polar unwraps and the correlation. */
export function knownCentre(img: Grey, s: KCSettings): KCResult {
  const size = 2 * s.radius + 2;    // even, centre at (size/2, size/2)
  const a = warpSimilarity(img, size, size, { angle: 0, scale: 1, tx: 0, ty: 0 }, { cx: s.cx, cy: s.cy, border: "mirror" });
  const b = warpSimilarity(img, size, size, { angle: s.angle * Math.PI / 180, scale: 1, tx: 0, ty: 0 }, { cx: s.cx, cy: s.cy, border: "mirror" });
  if (s.noise > 0) {
    const ra = rng(33), rb = rng(44);
    for (let i = 0; i < size * size; i++) { a[i] += s.noise * ra.norm(); b[i] += s.noise * rb.norm(); }
  }
  const nR = s.radius;
  const o: PolarOptions = { cx: size / 2, cy: size / 2, rMin: 1, rMax: s.radius, nR, nTheta: s.nTheta };
  const g = (d: Float64Array): Grey => ({ width: size, height: size, data: Float32Array.from(d) });
  const pa = Float64Array.from(toPolar(g(a), o).data), pb = Float64Array.from(toPolar(g(b), o).data);
  // weight each ring by its radius: every pixel of the disc counts about equally
  const weights = Array.from({ length: nR }, (_, i) => 1 + (s.radius - 1) * i / (nR - 1));
  const score = rotationalCorrelation(pa, pb, s.nTheta, nR, { weights });
  let best = 0;
  for (let t = 1; t < s.nTheta; t++) if (score[t] > score[best]) best = t;
  const cm = score[(best + s.nTheta - 1) % s.nTheta], c0 = score[best], cp = score[(best + 1) % s.nTheta];
  const den = cm - 2 * c0 + cp, off = den < 0 ? (cm - cp) / (2 * den) : 0;
  const angle = wrapDeg((best + off) * 360 / s.nTheta);
  return { a, b, size, pa, pb, nR, score, angle, height: c0 };
}
