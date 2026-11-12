// Grey-level morphology of a 1D signal with a flat or non-flat structuring element (no DOM;
// tested in demos/test/morphology-test.ts). The SE has half-width k (width 2k + 1) and
// heights s(b), b = −k..k (s ≡ 0: flat). Only samples inside the signal count (no padding).
//   erosion   e(x) = min_b f(x + b) − s(b)
//   dilation  d(x) = max_b f(x − b) + s(b)
//   opening   = dilation of the erosion, closing = erosion of the dilation (same s)

export type Profile = Float64Array<ArrayBufferLike>;

/** Heights of the SE: flat (all 0) or a parabola, 0 at the centre and −depth at the ends. */
export function seHeights(k: number, shape: "flat" | "parabola", depth = 60): Float64Array {
  return Float64Array.from({ length: 2 * k + 1 }, (_, j) => (shape === "flat" || k === 0 ? 0 : -depth * ((j - k) / k) ** 2));
}

export function erode1(f: ArrayLike<number>, s: ArrayLike<number>): Profile {
  const n = f.length, k = (s.length - 1) / 2, out = new Float64Array(n);
  for (let x = 0; x < n; x++) {
    let v = Infinity;
    for (let b = -k; b <= k; b++) { const y = x + b; if (y >= 0 && y < n) v = Math.min(v, f[y] - s[b + k]); }
    out[x] = v;
  }
  return out;
}

export function dilate1(f: ArrayLike<number>, s: ArrayLike<number>): Profile {
  const n = f.length, k = (s.length - 1) / 2, out = new Float64Array(n);
  for (let x = 0; x < n; x++) {
    let v = -Infinity;
    for (let b = -k; b <= k; b++) { const y = x - b; if (y >= 0 && y < n) v = Math.max(v, f[y] + s[b + k]); }
    out[x] = v;
  }
  return out;
}

export const open1 = (f: ArrayLike<number>, s: ArrayLike<number>) => dilate1(erode1(f, s), s);
export const close1 = (f: ArrayLike<number>, s: ArrayLike<number>) => erode1(dilate1(f, s), s);

/** Median over the window x − k .. x + k (inside samples only; even count near the ends: the upper middle value, as lib/morph.ts). */
export function median1(f: ArrayLike<number>, k: number): Profile {
  const n = f.length, out = new Float64Array(n);
  for (let x = 0; x < n; x++) {
    const w: number[] = [];
    for (let y = Math.max(0, x - k); y <= Math.min(n - 1, x + k); y++) w.push(f[y]);
    w.sort((a, b) => a - b);
    out[x] = w[Math.round((w.length - 1) / 2)];
  }
  return out;
}

/** |f(x + 1) − f(x − 1)| / 2 (one-sided at the ends): the size of the derivative. */
export function absDerivative(f: ArrayLike<number>): Profile {
  const n = f.length, out = new Float64Array(n);
  for (let x = 0; x < n; x++) {
    const a = f[Math.max(0, x - 1)], b = f[Math.min(n - 1, x + 1)];
    out[x] = Math.abs(b - a) / (x === 0 || x === n - 1 ? 1 : 2);
  }
  return out;
}

/** Piecewise-linear profile through control points (sorted by x), sampled at 0..n−1. */
export function fromPoints(pts: [number, number][], n: number): Profile {
  const out = new Float64Array(n);
  for (let x = 0; x < n; x++) {
    let j = 0;
    while (j < pts.length - 2 && pts[j + 1][0] < x) j++;
    const [x0, y0] = pts[j], [x1, y1] = pts[j + 1];
    const t = x1 === x0 ? 0 : Math.max(0, Math.min(1, (x - x0) / (x1 - x0)));
    out[x] = y0 + t * (y1 - y0);
  }
  return out;
}
