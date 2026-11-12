// Polar boundary finding (no DOM; also used by demos/test/dynprog-test.ts): unwrap the
// image around a centre (lib/polar.ts: one column per angle, one row per radius), turn the
// radial derivative into a cost, and find the closed path r(θ) by lib/dp.ts circularDP.
import { Grey, createGrey, gaussianBlur } from "../lib/imageops";
import { addGaussianNoise } from "../lib/synth";
import { toPolar, PolarOptions } from "../lib/polar";
import { circularDP, Closure, CircularResult } from "../lib/dp";

/** Edge polarity, going outward from the centre. "line": a thin dark line (the nuclear envelope). */
export type Polarity = "bright-dark" | "dark-bright" | "either" | "line";

/**
 * Cost table (stage-major: K angles × R radii, C[k * R + i]) from a polar image P
 * (R rows = radii, K columns = angles, as lib/polar.ts returns it). The radial derivative
 * g = ∂P/∂r (central differences) gives cost −g (dark inside → bright outside), +g
 * (bright inside → dark outside) or −|g|; "line" uses −∂²P/∂r² (a thin dark line across the
 * ray is a valley: cheap). Scaled to [−1, 1] (edges or lines of the wrong polarity cost extra).
 */
export function polarCost(P: ArrayLike<number>, K: number, R: number, polarity: Polarity): Float64Array {
  const C = new Float64Array(K * R);
  let mx = 0;
  for (let k = 0; k < K; k++) for (let i = 0; i < R; i++) {
    let v: number;
    if (polarity === "line") {
      // a thin dark line is a valley along the ray: large second derivative ∂²P/∂r²
      const a = P[Math.max(0, i - 2) * K + k], c = P[i * K + k], b = P[Math.min(R - 1, i + 2) * K + k];
      v = -(a - 2 * c + b) / 4;
    } else {
      const a = P[Math.max(0, i - 1) * K + k], b = P[Math.min(R - 1, i + 1) * K + k];
      const g = (b - a) / (i === 0 || i === R - 1 ? 1 : 2);
      v = polarity === "dark-bright" ? -g : polarity === "bright-dark" ? g : -Math.abs(g);
    }
    C[k * R + i] = v;
    mx = Math.max(mx, Math.abs(v));
  }
  if (mx > 0) for (let q = 0; q < C.length; q++) C[q] /= mx;
  return C;
}

export interface PolarSettings {
  cx: number;
  cy: number;
  /** Angles. */
  K: number;
  /** Radii 0..R−1 px, one row per pixel. */
  R: number;
  rMin: number;
  rMax: number;
  delta: number;
  alpha: number;
  polarity: Polarity;
  closure: Closure;
  /** Gaussian smoothing before the derivative. */
  sigma: number;
}

export interface PolarResult {
  polar: Grey;
  cost: Float64Array;
  dp: CircularResult;
  /** Contour points (x, y), one per angle. */
  contour: [number, number][];
  ms: number;
}

export function polarOptions(s: PolarSettings): PolarOptions {
  return { cx: s.cx, cy: s.cy, rMin: 0, rMax: s.R - 1, nR: s.R, nTheta: s.K, border: "replicate" };
}

/** The whole pipeline for a (pre-smoothed) image. */
export function segmentPolar(smoothed: Grey, s: PolarSettings): PolarResult {
  const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
  const o = polarOptions(s);
  const polar = toPolar(smoothed, o);
  const cost = polarCost(polar.data, s.K, s.R, s.polarity);
  const dp = circularDP(cost, s.K, s.R, { delta: s.delta, alpha: s.alpha, closure: s.closure, lo: s.rMin, hi: s.rMax });
  const contour: [number, number][] = [];
  for (let k = 0; k < s.K; k++) {
    const a = 2 * Math.PI * k / s.K, r = dp.path[k];
    contour.push([s.cx + r * Math.cos(a), s.cy + r * Math.sin(a)]);
  }
  const t1 = typeof performance !== "undefined" ? performance.now() : Date.now();
  return { polar, cost, dp, contour, ms: t1 - t0 };
}

/** Centroid of the area enclosed by a closed polygon. */
export function polygonCentroid(pts: [number, number][]): [number, number] {
  let A = 0, cx = 0, cy = 0;
  for (let k = 0; k < pts.length; k++) {
    const [x0, y0] = pts[k], [x1, y1] = pts[(k + 1) % pts.length], c = x0 * y1 - x1 * y0;
    A += c; cx += (x0 + x1) * c; cy += (y0 + y1) * c;
  }
  if (Math.abs(A) < 1e-9) return pts[0] ?? [0, 0];
  return [cx / (3 * A), cy / (3 * A)];
}

// ---------------------------------------------------------------------------------------
// Synthetic test images (256 × 256, seeded)

export const SIZE = 256;

/** Fraction of the pixel (x, y) inside the shape, by 4 × 4 supersampling. */
function coverage(inside: (x: number, y: number) => boolean, x: number, y: number): number {
  let c = 0;
  for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) if (inside(x - 0.375 + a * 0.25, y - 0.375 + b * 0.25)) c++;
  return c / 16;
}

const inEllipse = (cx: number, cy: number, a: number, b: number, ang: number) => (x: number, y: number) => {
  const dx = x - cx, dy = y - cy, c = Math.cos(ang), s = Math.sin(ang);
  const u = c * dx + s * dy, v = -s * dx + c * dy;
  return (u / a) ** 2 + (v / b) ** 2 <= 1;
};

/** Approximate signed depth (px) of (x, y) inside an ellipse: positive inside. */
function insideDepth(e: { cx: number; cy: number; a: number; b: number; angle: number }, x: number, y: number): number {
  const dx = x - e.cx, dy = y - e.cy, c = Math.cos(e.angle), s = Math.sin(e.angle);
  const u = c * dx + s * dy, v = -s * dx + c * dy, q = Math.hypot(u / e.a, v / e.b);
  return (1 - q) * Math.hypot(u, v) / Math.max(q, 1e-9);
}

/** Main nucleus of the "nucleus" image: centre and size (the preset centre). */
export const NUCLEUS = { cx: 100, cy: 128, a: 52, b: 40, angle: 0.35 };

/**
 * A bright nucleus (an ellipse, with a weak arc where its edge is a wide faint ramp)
 * touching a brighter, smaller neighbour on its right (at angle 0), plus a third nucleus
 * at the top right, out of reach;
 * blurred, with Gaussian noise. Background 40, nucleus 165, neighbour 205.
 */
export function nucleusImage(seed = 4): Grey {
  const img = createGrey(SIZE, SIZE, 40);
  const N = NUCLEUS;
  const main = inEllipse(N.cx, N.cy, N.a, N.b, N.angle);
  const nb = inEllipse(168, 128, 26, 22, 0);
  const top = inEllipse(205, 40, 26, 20, 0.6);
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    const i = y * SIZE + x;
    // the weak arc: towards the lower left (angles around 145°) the edge of the nucleus is a
    // wide, faint ramp instead of a step: its gradient is weak and spread out
    const ang = Math.atan2(y - N.cy, x - N.cx);
    const da = Math.abs(((ang - 2.5 + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI);
    const soft = da < 0.7 ? 1 - (da / 0.7) ** 2 : 0;
    const width = 1 + 26 * soft, depth = insideDepth(N, x, y);
    const cm = soft > 0 ? Math.min(1, Math.max(0, 0.5 + depth / width)) : coverage(main, x, y);
    const cn = coverage(nb, x, y), ct = coverage(top, x, y);
    let v = 40;
    if (cm > 0) v = v * (1 - cm) + cm * 165;
    if (cn > 0) v = Math.max(v, 40 * (1 - cn) + cn * 205);
    if (ct > 0) v = Math.max(v, 40 * (1 - ct) + ct * 185);
    img.data[i] = v;
  }
  return addGaussianNoise(gaussianBlur(img, 2), 14, seed);
}

/** A bean-shaped (kidney) nucleus: an ellipse with a bite out of its top. Not star-shaped from every centre. */
export const BEAN = { cx: 128, cy: 140 };
export function beanImage(seed = 6): Grey {
  const img = createGrey(SIZE, SIZE, 40);
  const ell = inEllipse(128, 138, 78, 50, 0), bite = inEllipse(128, 78, 42, 46, 0);
  const inside = (x: number, y: number) => ell(x, y) && !bite(x, y);
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    const c = coverage(inside, x, y);
    img.data[y * SIZE + x] = 40 * (1 - c) + 170 * c;
  }
  return addGaussianNoise(gaussianBlur(img, 2), 14, seed);
}
