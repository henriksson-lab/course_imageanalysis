// Distance transforms (no DOM). Input: a mask (flat array, non-zero = object) with its
// size. Output: for every object pixel the distance to the nearest BACKGROUND pixel
// (0 on the background itself), Infinity if the image has no background pixel at all.
//
//  - edt: exact Euclidean distance (Felzenszwalb–Huttenlocher lower envelope of
//    parabolas, separable, O(n)), optionally squared, with anisotropic pixel spacing,
//    and optionally the index of the nearest background pixel (feature transform).
//  - edt3d: the same for volumes.
//  - chamfer: two-pass local-mask approximations (city-block, chessboard, 3-4, 5-7-11).
//  - signedDistance: negative inside, positive outside (level sets).

/** 1D squared distance transform of f (Infinity = no seed) with spacing² s2; writes d and argmin. */
function dt1d(f: Float64Array, n: number, s2: number, d: Float64Array, arg: Int32Array,
  v: Int32Array, z: Float64Array): void {
  let k = -1;
  for (let q = 0; q < n; q++) {
    if (f[q] === Infinity) continue;
    if (k < 0) { k = 0; v[0] = q; z[0] = -Infinity; z[1] = Infinity; continue; }
    let s: number;
    for (;;) {
      const p = v[k];
      s = ((f[q] + s2 * q * q) - (f[p] + s2 * p * p)) / (2 * s2 * (q - p));
      if (s <= z[k]) { k--; if (k < 0) break; } else break;
    }
    k++;
    v[k] = q; z[k] = k === 0 ? -Infinity : s; z[k + 1] = Infinity;
  }
  if (k < 0) { for (let q = 0; q < n; q++) { d[q] = Infinity; arg[q] = -1; } return; }
  let j = 0;
  for (let q = 0; q < n; q++) {
    while (z[j + 1] < q) j++;
    const p = v[j];
    d[q] = s2 * (q - p) * (q - p) + f[p];
    arg[q] = p;
  }
}

export interface EdtOptions {
  /** Return squared distances (exact integers for unit spacing). */
  squared?: boolean;
  /** Pixel size [sx, sy] (default [1, 1]). */
  spacing?: [number, number];
}

/**
 * Exact Euclidean distance transform with the nearest background pixel of each pixel
 * (`nearest[i]` = its index y*w + x, or -1 if there is no background).
 */
export function edtWithNearest(mask: ArrayLike<number>, w: number, h: number, opts: EdtOptions = {}):
  { dist: Float32Array; nearest: Int32Array } {
  const [sx, sy] = opts.spacing ?? [1, 1];
  const m = Math.max(w, h);
  const f = new Float64Array(m), d = new Float64Array(m), arg = new Int32Array(m), v = new Int32Array(m), z = new Float64Array(m + 1);
  const g = new Float64Array(w * h), argRow = new Int32Array(w * h);
  for (let x = 0; x < w; x++) {          // columns: nearest background row in the same column
    for (let y = 0; y < h; y++) f[y] = mask[y * w + x] ? Infinity : 0;
    dt1d(f, h, sy * sy, d, arg, v, z);
    for (let y = 0; y < h; y++) { g[y * w + x] = d[y]; argRow[y * w + x] = arg[y]; }
  }
  const dist = new Float32Array(w * h), nearest = new Int32Array(w * h);
  for (let y = 0; y < h; y++) {          // rows: combine the columns
    for (let x = 0; x < w; x++) f[x] = g[y * w + x];
    dt1d(f, w, sx * sx, d, arg, v, z);
    for (let x = 0; x < w; x++) {
      const i = y * w + x, j = arg[x];
      dist[i] = opts.squared ? d[x] : Math.sqrt(d[x]);
      nearest[i] = j < 0 ? -1 : argRow[y * w + j] * w + j;
    }
  }
  return { dist, nearest };
}

/** Exact Euclidean distance of every object pixel to the nearest background pixel. */
export function edt(mask: ArrayLike<number>, w: number, h: number, opts: EdtOptions = {}): Float32Array {
  return edtWithNearest(mask, w, h, opts).dist;
}

/** Exact Euclidean distance transform of a volume nx × ny × nz (index z*nx*ny + y*nx + x). */
export function edt3d(mask: ArrayLike<number>, nx: number, ny: number, nz: number,
  opts: { squared?: boolean; spacing?: [number, number, number] } = {}): Float32Array {
  const sp = opts.spacing ?? [1, 1, 1], dims = [nx, ny, nz], strides = [1, nx, nx * ny];
  const n = nx * ny * nz, m = Math.max(nx, ny, nz);
  const cur = new Float64Array(n);
  for (let i = 0; i < n; i++) cur[i] = mask[i] ? Infinity : 0;
  const f = new Float64Array(m), d = new Float64Array(m), arg = new Int32Array(m), v = new Int32Array(m), z = new Float64Array(m + 1);
  for (let ax = 0; ax < 3; ax++) {
    const len = dims[ax], st = strides[ax], s2 = sp[ax] * sp[ax];
    for (let i = 0; i < n; i++) {
      // start of a line along `ax`: coordinate along ax is 0
      if (Math.floor(i / st) % len !== 0) continue;
      for (let q = 0; q < len; q++) f[q] = cur[i + q * st];
      dt1d(f, len, s2, d, arg, v, z);
      for (let q = 0; q < len; q++) cur[i + q * st] = d[q];
    }
  }
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = opts.squared ? cur[i] : Math.sqrt(cur[i]);
  return out;
}

/**
 * Chamfer distance by two raster passes with a local mask:
 *  cityblock (4-neighbour steps, |dx|+|dy|), chessboard (8-neighbour steps, max(|dx|,|dy|)),
 *  "3-4" (Borgefors, steps 3 and 4, divided by 3), "5-7-11" (adds knight moves, divided by 5).
 */
export type ChamferMetric = "cityblock" | "chessboard" | "3-4" | "5-7-11";

export function chamfer(mask: ArrayLike<number>, w: number, h: number, metric: ChamferMetric = "3-4"): Float32Array {
  // forward half-mask: [dx, dy, weight]; the backward pass uses (−dx, −dy)
  const fw: [number, number, number][] =
    metric === "cityblock" ? [[-1, 0, 1], [0, -1, 1]]
    : metric === "chessboard" ? [[-1, 0, 1], [-1, -1, 1], [0, -1, 1], [1, -1, 1]]
    : metric === "3-4" ? [[-1, 0, 3], [-1, -1, 4], [0, -1, 3], [1, -1, 4]]
    : [[-1, 0, 5], [-1, -1, 7], [0, -1, 5], [1, -1, 7], [-2, -1, 11], [-1, -2, 11], [1, -2, 11], [2, -1, 11]];
  const scale = metric === "3-4" ? 3 : metric === "5-7-11" ? 5 : 1;
  const d = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) d[i] = mask[i] ? Infinity : 0;
  const pass = (sign: number) => {
    for (let k = 0; k < w * h; k++) {
      const i = sign > 0 ? k : w * h - 1 - k, x = i % w, y = (i - x) / w;
      let best = d[i];
      if (best === 0) continue;
      for (const [dx, dy, c] of fw) {
        const xx = x + sign * dx, yy = y + sign * dy;
        if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
        const t = d[yy * w + xx] + c;
        if (t < best) best = t;
      }
      d[i] = best;
    }
  };
  pass(1); pass(-1);
  const out = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = d[i] / scale;
  return out;
}

/**
 * Signed Euclidean distance: for object pixels −(distance to the nearest background
 * pixel), for background pixels +(distance to the nearest object pixel). The zero level
 * lies between the two pixel layers at the boundary (each side is ≥ 1 away in magnitude).
 */
export function signedDistance(mask: ArrayLike<number>, w: number, h: number, opts: EdtOptions = {}): Float32Array {
  const inv = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) inv[i] = mask[i] ? 0 : 1;
  const din = edt(mask, w, h, opts), dout = edt(inv, w, h, opts), out = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = mask[i] ? -din[i] : dout[i];
  return out;
}
