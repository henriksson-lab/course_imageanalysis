// 1D geodesic operators on integer signals (no DOM; tested against lib/morph.ts in
// demos/test/morphology-test.ts). Neighbours are x − 1 and x + 1 (inside the signal only).
//   geodesic dilation  δ_f(m) = min(max(m(x−1), m(x), m(x+1)), f(x))     (marker below f)
//   geodesic erosion   ε_f(m) = max(min(m(x−1), m(x), m(x+1)), f(x))     (marker above f)
// Iterated until nothing changes: reconstruction by dilation / by erosion.

export type Sig = Int32Array;

/** max (dilate) or min (erode) over x − 1, x, x + 1. */
export function flat3(m: ArrayLike<number>, dilate: boolean): Sig {
  const n = m.length, out = new Int32Array(n);
  for (let x = 0; x < n; x++) {
    let v = m[x];
    if (x > 0) v = dilate ? Math.max(v, m[x - 1]) : Math.min(v, m[x - 1]);
    if (x < n - 1) v = dilate ? Math.max(v, m[x + 1]) : Math.min(v, m[x + 1]);
    out[x] = v;
  }
  return out;
}

export function geoDilate(m: ArrayLike<number>, f: ArrayLike<number>): Sig {
  const d = flat3(m, true);
  for (let x = 0; x < d.length; x++) d[x] = Math.min(d[x], f[x]);
  return d;
}

export function geoErode(m: ArrayLike<number>, f: ArrayLike<number>): Sig {
  const e = flat3(m, false);
  for (let x = 0; x < e.length; x++) e[x] = Math.max(e[x], f[x]);
  return e;
}

const same = (a: ArrayLike<number>, b: ArrayLike<number>) => { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; };

/** All iterates m_0 = clip(marker), m_1, ..., m_K with m_K = m_{K−1} removed (the last one is stable). */
export function iterates(marker: ArrayLike<number>, f: ArrayLike<number>, byDilation: boolean, maxIter = 1000): Sig[] {
  let m: Sig = Int32Array.from(marker, (v, i) => (byDilation ? Math.min(v, f[i]) : Math.max(v, f[i])));
  const out: Sig[] = [m];
  for (let k = 0; k < maxIter; k++) {
    const nx = byDilation ? geoDilate(m, f) : geoErode(m, f);
    if (same(nx, m)) break;
    out.push(nx); m = nx;
  }
  return out;
}

export const reconstruct = (marker: ArrayLike<number>, f: ArrayLike<number>, byDilation: boolean): Sig => {
  const it = iterates(marker, f, byDilation);
  return it[it.length - 1];
};

/** Regional maxima: runs of equal values whose neighbours (if any) are lower. */
export function regionalMax(f: ArrayLike<number>, minima = false): Uint8Array {
  const n = f.length, out = new Uint8Array(n), s = minima ? -1 : 1;
  for (let a = 0; a < n;) {
    let b = a;
    while (b + 1 < n && f[b + 1] === f[a]) b++;
    const left = a === 0 || s * f[a - 1] < s * f[a], right = b === n - 1 || s * f[b + 1] < s * f[a];
    if (left && right) for (let x = a; x <= b; x++) out[x] = 1;
    a = b + 1;
  }
  return out;
}

/** Number of runs of 1s. */
export function runs(m: ArrayLike<number>): number {
  let k = 0;
  for (let i = 0; i < m.length; i++) if (m[i] && (i === 0 || !m[i - 1])) k++;
  return k;
}

/** h-maxima transform R_f(f − h) and h-minima transform R^ε_f(f + h). */
export const hmax = (f: ArrayLike<number>, h: number) => reconstruct(Int32Array.from(f, v => v - h), f, true);
export const hmin = (f: ArrayLike<number>, h: number) => reconstruct(Int32Array.from(f, v => v + h), f, false);

/** Minima imposition: f_m = 0 on markers, max f + 1 elsewhere; mask g = min(f + 1, f_m);
 *  result R^ε_g(f_m), whose regional minima are exactly the marker runs. */
export function imposeParts(f: ArrayLike<number>, markers: ArrayLike<number>): { fm: Sig; g: Sig } {
  let hi = -Infinity;
  for (let i = 0; i < f.length; i++) hi = Math.max(hi, f[i]);
  const fm = Int32Array.from(f, (_, i) => (markers[i] ? 0 : hi + 1));
  const g = Int32Array.from(f, (v, i) => Math.min(v + 1, fm[i]));
  return { fm, g };
}

/** The 12-sample signal of the by-hand exercises (morph-reconstruct, and the watershed day). */
export const TWELVE = [7, 1, 4, 3, 4, 6, 5, 1, 6, 4, 2, 3];
