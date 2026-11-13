// Rectified stereo (no DOM). Owner: the geometry-from-images day (stereo/). Users: the
// stereomatch demo, tests. Images are row-major Float32Arrays w × h, grey 0..255. Disparity
// d ≥ 0: a left pixel (x, y) matches the right pixel (x − d, y); depth Z = f B / d.
//
// Contents: scene generator with ground truth (layers of textured planes, rendered into both
//   views by casting each pixel onto the frontmost layer) · costs (SAD, SSD, NCC, census) as
//   a cost volume C[(y·w + x)·D + d], window sums from lib/integral.ts · winner-take-all,
//   left–right check, sub-pixel parabola · scanline dynamic programming (per row, exact) and
//   semi-global matching (4 or 8 directions) with penalties P1 (|Δd| = 1) and P2 (larger) ·
//   bad-pixel rate against the truth.

import { rng } from "./random";
import { boxMean } from "./integral";

/**
 * Window mean over the clipped (2r + 1)² window, by running sums along rows then columns
 * (same result as lib/integral.ts boxMean, which it is tested against; about 10× faster here
 * because the cost volume needs D of them per update).
 */
export function boxMeanFast(img: ArrayLike<number>, w: number, h: number, r: number, out = new Float32Array(w * h)): Float32Array {
  if (r <= 0) { for (let i = 0; i < w * h; i++) out[i] = img[i]; return out; }
  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    let s = 0;
    for (let x = 0; x <= Math.min(r, w - 1); x++) s += img[o + x];
    for (let x = 0; x < w; x++) {
      const x0 = x - r, x1 = x + r;
      tmp[o + x] = s / (Math.min(x1, w - 1) - Math.max(x0, 0) + 1);
      if (x1 + 1 < w) s += img[o + x1 + 1];
      if (x0 >= 0) s -= img[o + x0];
    }
  }
  const col = new Float64Array(w);
  for (let y = 0; y <= Math.min(r, h - 1); y++) for (let x = 0; x < w; x++) col[x] += tmp[y * w + x];
  for (let y = 0; y < h; y++) {
    const n = Math.min(y + r, h - 1) - Math.max(y - r, 0) + 1;
    for (let x = 0; x < w; x++) out[y * w + x] = col[x] / n;
    if (y + r + 1 < h) for (let x = 0; x < w; x++) col[x] += tmp[(y + r + 1) * w + x];
    if (y - r >= 0) for (let x = 0; x < w; x++) col[x] -= tmp[(y - r) * w + x];
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Scenes

/** A planar layer with disparity d(x, y) = a x + b y + c in LEFT image coordinates. */
export interface Layer {
  a: number; b: number; c: number;
  /** Support in left coordinates (x0 ≤ x < x1, y0 ≤ y < y1); omitted = everywhere. */
  rect?: [number, number, number, number];
  /** Texture at a surface point, given in left coordinates (x, y). */
  tex: (x: number, y: number) => number;
}

export interface StereoScene {
  w: number; h: number;
  left: Float32Array;
  right: Float32Array;
  /** True disparity of every left pixel (frontmost layer). */
  disp: Float32Array;
  /** 1 where the left pixel is hidden in the right view (occluded) or falls outside it. */
  occluded: Uint8Array;
  /** Which layer each left pixel sees. */
  layer: Int16Array;
  maxD: number;
  name: string;
}

/** Smooth value noise in [0, 1]: bilinear interpolation of a seeded random grid of spacing `cell`. */
export function valueNoise(seed: number, cell: number, octaves = 3): (x: number, y: number) => number {
  const N = 97, grids: Float32Array[] = [];
  const r = rng(seed);
  for (let o = 0; o < octaves; o++) grids.push(Float32Array.from({ length: N * N }, () => r.unif()));
  return (x, y) => {
    let s = 0, wsum = 0;
    for (let o = 0; o < octaves; o++) {
      const c = cell / (1 << o), g = grids[o], wt = 1 / (1 << o);
      const fx = x / c, fy = y / c, ix = Math.floor(fx), iy = Math.floor(fy), tx = fx - ix, ty = fy - iy;
      const at = (i: number, j: number) => g[(((j % N) + N) % N) * N + (((i % N) + N) % N)];
      s += wt * ((1 - ty) * ((1 - tx) * at(ix, iy) + tx * at(ix + 1, iy)) + ty * ((1 - tx) * at(ix, iy + 1) + tx * at(ix + 1, iy + 1)));
      wsum += wt;
    }
    return s / wsum;
  };
}

/** Random dots: square cells of `cell` pixels, each black or white (seeded). */
export function randomDots(seed: number, cell = 2): (x: number, y: number) => number {
  const N = 211, r = rng(seed), g = Uint8Array.from({ length: N * N }, () => (r.unif() < 0.5 ? 1 : 0));
  return (x, y) => {
    const i = Math.floor(x / cell), j = Math.floor(y / cell);
    return g[(((j % N) + N) % N) * N + (((i % N) + N) % N)];
  };
}

const dispAt = (L: Layer, x: number, y: number) => L.a * x + L.b * y + L.c;
const inRect = (L: Layer, x: number, y: number) => !L.rect || (x >= L.rect[0] && x < L.rect[2] && y >= L.rect[1] && y < L.rect[3]);

/** Frontmost layer seen at left coordinate (x, y): largest disparity among the covering layers. */
function frontLeft(layers: Layer[], x: number, y: number): number {
  let best = -1, bd = -Infinity;
  layers.forEach((L, k) => { if (inRect(L, x, y)) { const d = dispAt(L, x, y); if (d > bd) { bd = d; best = k; } } });
  return best;
}

/** Frontmost layer seen at RIGHT coordinate (xr, y): the left point x with x − d(x) = xr. */
function frontRight(layers: Layer[], xr: number, y: number): { k: number; x: number } {
  let best = -1, bd = -Infinity, bx = 0;
  layers.forEach((L, k) => {
    const x = (xr + L.b * y + L.c) / (1 - L.a);
    if (inRect(L, x, y)) { const d = dispAt(L, x, y); if (d > bd) { bd = d; best = k; bx = x; } }
  });
  return { k: best, x: bx };
}

/** Render a rectified pair from layers (3 sub-samples per pixel horizontally), add noise and an optional gain/offset to the right view. */
export function renderScene(name: string, w: number, h: number, layers: Layer[], o: { noise?: number; seed?: number; gain?: number; offset?: number; maxD?: number } = {}): StereoScene {
  const left = new Float32Array(w * h), right = new Float32Array(w * h), disp = new Float32Array(w * h);
  const occluded = new Uint8Array(w * h), layer = new Int16Array(w * h);
  const SUB = [-1 / 3, 0, 1 / 3];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    let sl = 0, sr = 0;
    for (const s of SUB) {
      const kL = frontLeft(layers, x + s, y);
      sl += kL >= 0 ? layers[kL].tex(x + s, y) : 0;
      const R = frontRight(layers, x + s, y);
      sr += R.k >= 0 ? layers[R.k].tex(R.x, y) : 0;
    }
    left[i] = sl / 3; right[i] = sr / 3;
    const k = frontLeft(layers, x, y);
    layer[i] = k;
    const d = k >= 0 ? dispAt(layers[k], x, y) : 0;
    disp[i] = d;
    const xr = x - d;
    occluded[i] = xr < 0 || frontRight(layers, xr, y).k !== k ? 1 : 0;
  }
  const r = rng(o.seed ?? 7), g = o.gain ?? 1, off = o.offset ?? 0, sd = o.noise ?? 0;
  for (let i = 0; i < w * h; i++) {
    left[i] = clamp255(left[i] + sd * r.norm());
    right[i] = clamp255(g * right[i] + off + sd * r.norm());
  }
  let mx = 0;
  for (const d of disp) mx = Math.max(mx, d);
  return { w, h, left, right, disp, occluded, layer, maxD: o.maxD ?? Math.ceil(mx + 4), name };
}
const clamp255 = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : v);

export type SceneId = "dots" | "mixed" | "gain";

/**
 * The demo scenes (w × h, default 192 × 144):
 *  - dots: random-dot stereogram, background d = 8, a raised square d = 20;
 *  - mixed: a slanted textured ground (d from 6 at the top to 18 at the bottom), a
 *    textureless wall with a textured frame (d = 26), a striped fence with period 8 px and textured rails (d = 24) and a textured post
 *    in front (d = 32);
 *  - gain: smooth texture, background d = 8 and a square d = 20, with the right camera at
 *    gain 0.6 and offset +60.
 */
export function makeScene(id: SceneId, w = 192, h = 144, noise = 2, seed = 7): StereoScene {
  const sx = w / 192, sy = h / 144;
  if (id === "gain") {
    const bg = valueNoise(seed, 5), fg = valueNoise(seed + 1, 5);
    const layers: Layer[] = [
      { a: 0, b: 0, c: 8, tex: (x, y) => 50 + 160 * bg(x - 8, y) },
      { a: 0, b: 0, c: 20, rect: [56 * sx, 36 * sy, 136 * sx, 108 * sy], tex: (x, y) => 50 + 160 * fg(x - 20, y) },
    ];
    return renderScene(id, w, h, layers, { noise, seed, gain: 0.6, offset: 60, maxD: 32 });
  }
  if (id === "dots") {
    const bg = randomDots(seed, 2), fg = randomDots(seed + 1, 2);
    const layers: Layer[] = [
      { a: 0, b: 0, c: 8, tex: (x, y) => 40 + 170 * bg(x - 8, y) },
      { a: 0, b: 0, c: 20, rect: [56 * sx, 36 * sy, 136 * sx, 108 * sy], tex: (x, y) => 40 + 170 * fg(x - 20, y) },
    ];
    return renderScene(id, w, h, layers, { noise, seed, maxD: 32 });
  }
  const ground = valueNoise(seed, 6), post = valueNoise(seed + 2, 3);
  const dTop = 6, dBot = 18;
  const layers: Layer[] = [
    // slanted ground: disparity grows linearly downwards
    { a: 0, b: (dBot - dTop) / (h - 1), c: dTop, tex: (x, y) => 30 + 200 * ground(x - (dTop + (dBot - dTop) * y / (h - 1)), y) },
    // textureless wall with a textured frame 3 px wide
    { a: 0, b: 0, c: 26, rect: [40 * sx, 14 * sy, 76 * sx, 80 * sy], tex: (x, y) => (x < 43 * sx || x >= 73 * sx || y < 17 * sy || y >= 77 * sy ? 40 + 190 * post(x + 50, y) : 150) },
    // striped fence (repetitive)
    { a: 0, b: 0, c: 24, rect: [100 * sx, 20 * sy, 176 * sx, 64 * sy], tex: (x, y) => (y < 24 * sy || y >= 60 * sy ? 40 + 190 * post(x + 120, y) : Math.floor((x - 24) / 4) % 2 === 0 ? 60 : 200) },
    // textured post in front
    { a: 0, b: 0, c: 32, rect: [82 * sx, 30 * sy, 96 * sx, 140 * sy], tex: (x, y) => 40 + 190 * post(x - 32, y) },
  ];
  return renderScene(id, w, h, layers, { noise, seed, maxD: 40 });
}

// ---------------------------------------------------------------------------------------
// Costs

export type CostKind = "sad" | "ssd" | "ncc" | "census";

/** Census transform with a (2c + 1)² window (c ≤ 2: at most 24 bits): bit k = neighbour k < centre. */
export function census(img: ArrayLike<number>, w: number, h: number, c = 2): Uint32Array {
  const out = new Uint32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = img[y * w + x];
    let bits = 0, k = 0;
    for (let dy = -c; dy <= c; dy++) for (let dx = -c; dx <= c; dx++) {
      if (!dx && !dy) continue;
      const xx = Math.min(w - 1, Math.max(0, x + dx)), yy = Math.min(h - 1, Math.max(0, y + dy));
      if (img[yy * w + xx] < v) bits |= 1 << k;
      k++;
    }
    out[y * w + x] = bits >>> 0;
  }
  return out;
}

export function popcount(v: number): number {
  v = v - ((v >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

/**
 * Cost volume C[(y·w + x)·D + d] for d = 0 … D − 1, comparing the (2r + 1)² window around
 * (x, y) in L with the window around (x − d, y) in R (columns x − d < 0 use column 0).
 * SAD/SSD are window means of |L − R| or (L − R)²; NCC gives 1 − ncc (0 = perfect, 2 = worst);
 * census sums the Hamming distances of 5 × 5 census codes over the window (mean, 0..24).
 * Window sums by running sums (boxMeanFast; lib/integral.ts boxMean is the reference).
 */
export function costVolume(L: ArrayLike<number>, R: ArrayLike<number>, w: number, h: number, D: number, kind: CostKind, r: number): Float32Array {
  const n = w * h, C = new Float32Array(n * D), tmp = new Float32Array(n), bm = new Float32Array(n);
  const shifted = (img: ArrayLike<number>, d: number) => {
    const s = new Float32Array(n);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) s[y * w + x] = img[y * w + Math.max(0, x - d)];
    return s;
  };
  if (kind === "census") {
    const cl = census(L, w, h), cr = census(R, w, h);
    for (let d = 0; d < D; d++) {
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) tmp[y * w + x] = popcount((cl[y * w + x] ^ cr[y * w + Math.max(0, x - d)]) >>> 0);
      const m = boxMeanFast(tmp, w, h, r, bm);
      for (let i = 0; i < n; i++) C[i * D + d] = m[i];
    }
    return C;
  }
  if (kind === "ncc") {
    const mL = boxMeanFast(L, w, h, r), mL2 = boxMeanFast(Float32Array.from(L, v => v * v), w, h, r);
    for (let d = 0; d < D; d++) {
      const Rs = shifted(R, d);
      const mR = boxMeanFast(Rs, w, h, r), mR2 = boxMeanFast(Rs.map(v => v * v), w, h, r);
      for (let i = 0; i < n; i++) tmp[i] = L[i] * Rs[i];
      const mLR = boxMeanFast(tmp, w, h, r);
      for (let i = 0; i < n; i++) {
        const vl = mL2[i] - mL[i] * mL[i], vr = mR2[i] - mR[i] * mR[i], cv = mLR[i] - mL[i] * mR[i];
        const den = Math.sqrt(Math.max(vl, 0) * Math.max(vr, 0));
        C[i * D + d] = den > 1e-6 ? 1 - cv / den : 1;
      }
    }
    return C;
  }
  for (let d = 0; d < D; d++) {
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const e = L[y * w + x] - R[y * w + Math.max(0, x - d)];
      tmp[y * w + x] = kind === "sad" ? Math.abs(e) : e * e;
    }
    const m = boxMeanFast(tmp, w, h, r, bm);
    for (let i = 0; i < n; i++) C[i * D + d] = m[i];
  }
  return C;
}

/** Same comparison from the right image's point of view: right pixel (x, y) vs left (x + d, y). */
export function costVolumeRight(L: ArrayLike<number>, R: ArrayLike<number>, w: number, h: number, D: number, kind: CostKind, r: number): Float32Array {
  // mirror both images horizontally: then the right image plays the left role
  const flip = (img: ArrayLike<number>) => { const o = new Float32Array(w * h); for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) o[y * w + x] = img[y * w + (w - 1 - x)]; return o; };
  const Cm = costVolume(flip(R), flip(L), w, h, D, kind, r), C = new Float32Array(w * h * D);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) C.set(Cm.subarray((y * w + (w - 1 - x)) * D, (y * w + (w - 1 - x) + 1) * D), (y * w + x) * D);
  return C;
}

/** Winner-take-all: the disparity of minimal cost per pixel (ties: smallest d). */
export function wta(C: ArrayLike<number>, n: number, D: number): Int32Array {
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    let b = 0, bv = C[i * D];
    for (let d = 1; d < D; d++) if (C[i * D + d] < bv) { bv = C[i * D + d]; b = d; }
    out[i] = b;
  }
  return out;
}

/** Left–right check: keep a left disparity d at x only if the right map at x − d agrees within `tol`. Invalid = −1. */
export function leftRightCheck(dl: ArrayLike<number>, dr: ArrayLike<number>, w: number, h: number, tol = 1): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const d = dl[y * w + x], xr = Math.round(x - d);
    out[y * w + x] = xr >= 0 && Math.abs(dr[y * w + xr] - d) <= tol ? d : -1;
  }
  return out;
}

/** Parabola through (−1, cm), (0, c0), (1, cp): offset of the minimum, clamped to ±0.5. */
export function parabolaVertex(cm: number, c0: number, cp: number): number {
  const den = cm - 2 * c0 + cp;
  if (!(den > 1e-12)) return 0;
  return Math.max(-0.5, Math.min(0.5, (cm - cp) / (2 * den)));
}

/** Sub-pixel refinement of integer disparities from the cost volume (aggregated costs work too). */
export function subpixel(C: ArrayLike<number>, disp: ArrayLike<number>, n: number, D: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const d = disp[i];
    if (d < 0) { out[i] = d; continue; }
    out[i] = d > 0 && d < D - 1 ? d + parabolaVertex(C[i * D + d - 1], C[i * D + d], C[i * D + d + 1]) : d;
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Aggregation: scanline DP and semi-global matching

/** Pairwise cost of a disparity change: 0, P1 for |Δ| = 1, P2 for larger jumps. */
export const jumpCost = (a: number, b: number, P1: number, P2: number): number => (a === b ? 0 : Math.abs(a - b) === 1 ? P1 : P2);

/**
 * Scanline dynamic programming: for every row independently, the exact minimum of
 * Σ_x C(x, d_x) + Σ_x V(d_x, d_{x+1}) with V = jumpCost (Viterbi, O(D) per pixel by the
 * P1/P2 structure). Returns the disparity map and the total energy.
 */
export function scanlineDP(C: ArrayLike<number>, w: number, h: number, D: number, P1: number, P2: number): { disp: Int32Array; energy: number } {
  const disp = new Int32Array(w * h), back = new Int32Array(w * D);
  let M = new Float64Array(D), Mn = new Float64Array(D), energy = 0;
  for (let y = 0; y < h; y++) {
    const base = y * w;
    for (let d = 0; d < D; d++) M[d] = C[base * D + d];
    for (let x = 1; x < w; x++) {
      let mi = 0;
      for (let d = 1; d < D; d++) if (M[d] < M[mi]) mi = d;
      const mv = M[mi];
      for (let d = 0; d < D; d++) {
        let best = M[d], arg = d;
        if (d > 0 && M[d - 1] + P1 < best) { best = M[d - 1] + P1; arg = d - 1; }
        if (d < D - 1 && M[d + 1] + P1 < best) { best = M[d + 1] + P1; arg = d + 1; }
        if (mv + P2 < best) { best = mv + P2; arg = mi; }
        Mn[d] = C[(base + x) * D + d] + best;
        back[x * D + d] = arg;
      }
      const t = M; M = Mn; Mn = t;
    }
    let bd = 0;
    for (let d = 1; d < D; d++) if (M[d] < M[bd]) bd = d;
    energy += M[bd];
    disp[base + w - 1] = bd;
    for (let x = w - 1; x > 0; x--) disp[base + x - 1] = back[x * D + disp[base + x]];
  }
  return { disp, energy };
}

const DIRS8: [number, number][] = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]];

/**
 * Semi-global matching: for each direction r, L_r(p, d) = C(p, d) + min(L_r(p − r, d),
 * L_r(p − r, d ± 1) + P1, min_k L_r(p − r, k) + P2) − min_k L_r(p − r, k); S = Σ_r L_r.
 * `dirs` = 1 (left to right only), 2 (both horizontal), 4 (+ vertical) or 8 (+ diagonals).
 */
export function sgm(C: ArrayLike<number>, w: number, h: number, D: number, P1: number, P2: number, dirs: 1 | 2 | 4 | 8): Float32Array {
  const S = new Float32Array(w * h * D);
  // two row buffers of L_r (previous and current row), enough for every direction
  let prev = new Float32Array(w * D), cur = new Float32Array(w * D);
  for (let k = 0; k < dirs; k++) {
    const [dx, dy] = DIRS8[k];
    const ys = dy >= 0 ? 0 : h - 1, ye = dy >= 0 ? h : -1, ystep = dy >= 0 ? 1 : -1;
    // horizontal directions traverse x against dx; others any order within a row (use dx's)
    const xs = dx >= 0 ? 0 : w - 1, xe = dx >= 0 ? w : -1, xstep = dx >= 0 ? 1 : -1;
    for (let y = ys; y !== ye; y += ystep) {
      const py = y - dy, rowOK = py >= 0 && py < h;
      for (let x = xs; x !== xe; x += xstep) {
        const p = (y * w + x) * D, c = x * D, px = x - dx;
        if (px < 0 || px >= w || !rowOK) {
          for (let d = 0; d < D; d++) { const v = C[p + d]; cur[c + d] = v; S[p + d] += v; }
          continue;
        }
        const src = dy === 0 ? cur : prev, q = px * D;
        let mv = src[q];
        for (let d = 1; d < D; d++) if (src[q + d] < mv) mv = src[q + d];
        const jump = mv + P2;
        for (let d = 0; d < D; d++) {
          let best = src[q + d];
          if (d > 0) { const a = src[q + d - 1] + P1; if (a < best) best = a; }
          if (d < D - 1) { const a = src[q + d + 1] + P1; if (a < best) best = a; }
          if (jump < best) best = jump;
          const v = C[p + d] + best - mv;
          cur[c + d] = v; S[p + d] += v;
        }
      }
      const t = prev; prev = cur; cur = t;
    }
  }
  return S;
}

/** Fraction of evaluated pixels whose disparity differs from the truth by more than `tol`
 *  (invalid = −1 counts as bad). Evaluated: not occluded and x ≥ maxD (fully inside both views). */
export function badPixelRate(disp: ArrayLike<number>, truth: StereoScene, tol = 1, margin = -1): { bad: number; n: number; rate: number } {
  const { w, h } = truth, m = margin < 0 ? truth.maxD : margin;
  let bad = 0, n = 0;
  for (let y = 0; y < h; y++) for (let x = m; x < w; x++) {
    const i = y * w + x;
    if (truth.occluded[i]) continue;
    n++;
    if (disp[i] < 0 || Math.abs(disp[i] - truth.disp[i]) > tol) bad++;
  }
  return { bad, n, rate: n ? bad / n : 0 };
}

/** Depth from disparity: Z = f B / d (Infinity for d ≤ 0). */
export const depthFromDisparity = (d: number, f: number, B: number): number => (d > 0 ? f * B / d : Infinity);
