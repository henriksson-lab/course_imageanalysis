// Synthetic test images of the compression day (no DOM), 256 × 256, integer grey levels
// 0..255 (row-major Float64Array). Fixed seeds: the same image on every machine.
// Used by the predict, dctblock, jpeg, samemse and measuredamage demos (and the node scripts
// that compute the numbers quoted in the exercises).
//  - smooth: soft blobs on a gradient with weak noise (prediction works well);
//  - shapes: flat regions with straight and diagonal edges (edges are what prediction misses);
//  - card: a test card with a smooth gradient, sharp edges, fine stripes and a flat background
//    (blocking in the flat parts, ringing at the edges);
//  - colourCard: an RGB card with saturated red/blue edges (chroma subsampling);
//  - flat: a constant grey.
import { rng } from "../lib/random";

export const SIZE = 256;

const clamp255 = (v: number) => Math.min(255, Math.max(0, Math.round(v)));

/** Add seeded Gaussian noise of sd `sd`, then round and clamp to 0..255 (sd = 0: just round). */
export function withNoise(img: ArrayLike<number>, sd: number, seed = 7, max = 255): Float64Array {
  const r = rng(seed), out = new Float64Array(img.length);
  for (let i = 0; i < img.length; i++) out[i] = Math.min(max, Math.max(0, Math.round(img[i] + (sd > 0 ? sd * r.norm() : 0))));
  return out;
}

/** Soft blobs on a left-to-right gradient, noise sd 1.5. */
export function smooth(seed = 3): Float64Array {
  const N = SIZE, r = rng(seed), out = new Float64Array(N * N);
  const blobs = Array.from({ length: 12 }, () => ({
    x: 20 + 216 * r.unif(), y: 20 + 216 * r.unif(), sx: 10 + 16 * r.unif(), sy: 10 + 16 * r.unif(),
    a: (r.unif() < 0.5 ? -1 : 1) * (30 + 60 * r.unif()),
  }));
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let v = 100 + 60 * x / (N - 1);
    for (const b of blobs) v += b.a * Math.exp(-((x - b.x) ** 2) / (2 * b.sx * b.sx) - ((y - b.y) ** 2) / (2 * b.sy * b.sy));
    out[y * N + x] = v;
  }
  return withNoise(out, 1.5, seed + 1);
}

/** Coverage (0..1) of a pixel by a region, 4 × 4 supersampled. */
function cover(inside: (x: number, y: number) => boolean, x: number, y: number): number {
  let c = 0;
  for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) if (inside(x - 0.375 + 0.25 * i, y - 0.375 + 0.25 * j)) c++;
  return c / 16;
}

/** Flat shapes with straight and diagonal edges on a flat background, noise sd 1.5. */
export function shapes(): Float64Array {
  const N = SIZE, out = new Float64Array(N * N);
  const regions: { inside: (x: number, y: number) => boolean; v: number }[] = [
    { inside: (x, y) => Math.abs(x - 80) + Math.abs(y - 80) < 55, v: 190 },             // diamond (diagonal edges)
    { inside: (x, y) => (x - 185) ** 2 + (y - 75) ** 2 < 45 * 45, v: 120 },              // disc
    { inside: (x, y) => x > 30 && x < 120 && y > 160 && y < 230, v: 40 },               // rectangle
    { inside: (x, y) => y > 150 && y < 240 && x > 140 && x < 245 && x - 140 > (240 - y) * 0.9, v: 220 },   // triangle
  ];
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let v = 100;
    for (const g of regions) { const c = cover(g.inside, x, y); v = v * (1 - c) + g.v * c; }
    out[y * N + x] = v;
  }
  return withNoise(out, 1.5, 11);
}

/** A grey test card: gradient, flat areas, sharp edges, stripes, a disc and thin lines. */
export function card(): Float64Array {
  const N = SIZE, out = new Float64Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let v: number;
    if (y < 48) v = 30 + 200 * x / (N - 1);                                             // smooth gradient
    else if (y < 128) v = x < 132 ? 200 : 60;                                            // flat halves, a vertical edge inside a block
    else if (y < 192) v = x < 128 ? ((x >> 1) % 2 ? 230 : 40) : (((x + y) >> 2) % 2 ? 200 : 80);   // fine and diagonal stripes
    else v = 150;                                                                       // flat
    out[y * N + x] = v;
  }
  // a dark disc on the flat bottom area and thin dark lines (like text strokes)
  for (let y = 192; y < N; y++) for (let x = 0; x < N; x++) {
    const c = cover((u, w) => (u - 64) ** 2 + (w - 224) ** 2 < 24 * 24, x, y);
    out[y * N + x] = out[y * N + x] * (1 - c) + 40 * c;
    if (x >= 140 && x < 240 && (y === 205 || y === 215 || y === 225 || y === 235) ) out[y * N + x] = 30;
    if ((x === 150 || x === 175 || x === 200 || x === 225) && y >= 205 && y <= 235) out[y * N + x] = 30;
  }
  return out.map(clamp255);
}

/** An RGB card: red | blue halves with a sharp edge, a green disc, a colour gradient, white bars. */
export function colourCard(): [Float64Array, Float64Array, Float64Array] {
  const N = SIZE, R = new Float64Array(N * N), G = new Float64Array(N * N), B = new Float64Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x;
    let r: number, g: number, b: number;
    if (y < 64) { const t = x / (N - 1); r = 255 * (1 - t); g = 120; b = 255 * t; }          // gradient
    else if (y < 192) { if (x < 131) { r = 210; g = 30; b = 40; } else { r = 30; g = 50; b = 200; } }  // red | blue (edge inside a 2 × 2 pair)
    else { r = 235; g = 225; b = 200; }                                                       // pale background
    const c = cover((u, w) => (u - 128) ** 2 + (w - 128) ** 2 < 36 * 36, x, y);              // green disc on the edge
    r = r * (1 - c) + 40 * c; g = g * (1 - c) + 180 * c; b = b * (1 - c) + 60 * c;
    if (y >= 208 && y < 240 && x >= 24 && x < 232 && ((x - 24) >> 3) % 2 === 0 && x % 8 < 3) { r = 20; g = 20; b = 20; }   // thin dark strokes
    R[i] = clamp255(r); G[i] = clamp255(g); B[i] = clamp255(b);
  }
  return [R, G, B];
}

/** A constant grey. */
export const flat = (v = 128): Float64Array => new Float64Array(SIZE * SIZE).fill(v);

/** Luma of an RGB image (Rec. 601, as JPEG's Y′), rounded. */
export function luma(R: ArrayLike<number>, G: ArrayLike<number>, B: ArrayLike<number>): Float64Array {
  return Float64Array.from({ length: R.length }, (_, i) => clamp255(0.299 * R[i] + 0.587 * G[i] + 0.114 * B[i]));
}
