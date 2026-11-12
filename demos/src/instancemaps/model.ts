// Model of the instancemaps demo (no DOM; tested in demos/test/moderndl-test.ts).
//
// A seeded synthetic 128 × 128 fluorescence image of cells with ground-truth labels: round
// and oval cells, touching pairs, an elongated rod, a C-shaped cell, a bent cell, a dim cell
// (or a crowded tissue). From the truth we compute the exact training TARGETS of four
// instance routes, then degrade them with error sliders to SIMULATE a network's output:
//   semantic   P(cell) per pixel;
//   boundary   three classes background / interior / boundary;
//   distance   distance to the cell's border, normalised to 1 per cell;
//   star       objectness + distances to the border along n rays (star-convex polygons);
//   flows      unit vectors along the gradient of a heat distribution with its source at the
//              cell centre (paths stay inside the cell and bend with it).
// Errors: noise (smooth seeded random fields), blur, contact bias (where two cells touch, the
// map looks as if the contact were not there: no boundary, the distance filled up to a plateau,
// rays and flows of the merged mask; the typical real failure), missed dim cells.
// Post-processing turns each map into labels: connected components, boundary class + growing,
// h-maxima + watershed, polygon NMS, following the flows to their sinks.
import { label2d } from "../lib/label";
import { edt } from "../lib/distance";
import { hMaxima } from "../lib/morph";
import { floodWatershed } from "../lib/watershed";
import { createGrey, gaussianBlur, type Grey } from "../lib/imageops";
import { addGaussianNoise } from "../lib/synth";
import { rng } from "../lib/ui";
import { nmsGeneric } from "../lib/boxes";

export const N = 128;
export const MAX_RAYS = 64;

export type Shape =
  | { kind: "ell"; x: number; y: number; a: number; b: number; angle: number; dim?: boolean }
  | { kind: "arc"; x: number; y: number; R: number; w: number; a0: number; a1: number; dim?: boolean };

/** Depth of a pixel centre inside a shape (in pixels, > 0 inside). */
export function depth(s: Shape, px: number, py: number): number {
  if (s.kind === "ell") {
    const c = Math.cos(s.angle), sn = Math.sin(s.angle), dx = px - s.x, dy = py - s.y;
    const u = c * dx + sn * dy, v = -sn * dx + c * dy;
    return (1 - Math.hypot(u / s.a, v / s.b)) * s.b;
  }
  // arc: distance to the centre curve (circle of radius R from angle a0 to a1), round caps
  const dx = px - s.x, dy = py - s.y;
  let th = Math.atan2(dy, dx);
  while (th < s.a0) th += 2 * Math.PI;
  while (th > s.a0 + 2 * Math.PI) th -= 2 * Math.PI;
  let d: number;
  if (th <= s.a1) d = Math.abs(Math.hypot(dx, dy) - s.R);
  else {
    const e0 = [s.x + s.R * Math.cos(s.a0), s.y + s.R * Math.sin(s.a0)], e1 = [s.x + s.R * Math.cos(s.a1), s.y + s.R * Math.sin(s.a1)];
    d = Math.min(Math.hypot(px - e0[0], py - e0[1]), Math.hypot(px - e1[0], py - e1[1]));
  }
  return s.w - d;
}

export type SceneId = "mixed" | "crowded";

export interface Scene {
  shapes: Shape[];
  labels: Int32Array;      // truth, 1..n
  n: number;
  image: Grey;
  dim: Uint8Array;         // per label (index label−1): 1 if dim
}

function mixedShapes(): Shape[] {
  return [
    { kind: "ell", x: 18, y: 18, a: 10, b: 8, angle: 0.3 },
    { kind: "ell", x: 52, y: 15, a: 9, b: 9, angle: 0 },
    // touching pair with a deep neck (two round cells)
    { kind: "ell", x: 86, y: 18, a: 10, b: 9.5, angle: 0 },
    { kind: "ell", x: 104, y: 22, a: 9.5, b: 9, angle: 0.4 },
    // touching pair with a shallow neck (strongly overlapping ovals)
    { kind: "ell", x: 22, y: 52, a: 11, b: 8, angle: 1.4 },
    { kind: "ell", x: 34, y: 56, a: 11, b: 8, angle: 1.2 },
    // elongated rod
    { kind: "ell", x: 70, y: 52, a: 25, b: 5, angle: -0.35 },
    // C-shaped cell
    { kind: "arc", x: 104, y: 64, R: 12, w: 4.5, a0: 0.9, a1: 0.9 + 4.9 },
    // bent (banana) cell
    { kind: "arc", x: 22, y: 128, R: 40, w: 5, a0: -1.45, a1: -0.55 },
    { kind: "ell", x: 68, y: 86, a: 8, b: 7, angle: 0 },
    { kind: "ell", x: 79, y: 97, a: 8.5, b: 7.5, angle: 0.6 },   // touches the previous one
    { kind: "ell", x: 108, y: 106, a: 10, b: 8, angle: -0.5, dim: true },
    { kind: "ell", x: 70, y: 117, a: 8, b: 7, angle: 0.2, dim: true },
    { kind: "ell", x: 14, y: 84, a: 7, b: 6, angle: 0.8 },
  ];
}

function crowdedShapes(): Shape[] {
  const r = rng(77), out: Shape[] = [];
  for (let row = 0; row < 9; row++) for (let col = 0; col < 9; col++) {
    const x = 8 + col * 14.5 + (row % 2) * 7 + 1.2 * r.norm(), y = 8 + row * 13.5 + 1.2 * r.norm();
    if (x > 124 || y > 124) continue;
    out.push({ kind: "ell", x, y, a: 8.2 + r.unif() * 1.4, b: 7 + r.unif(), angle: Math.PI * r.unif(), dim: r.unif() < 0.08 });
  }
  return out;
}

/** Rasterise shapes: each pixel goes to the shape it is deepest in (so touching cells split along the middle). */
export function rasterise(shapes: Shape[]): Int32Array {
  const lab = new Int32Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let best = 0, bd = 0;
    shapes.forEach((s, k) => { const d = depth(s, x, y); if (d > bd) { bd = d; best = k + 1; } });
    lab[y * N + x] = best;
  }
  // drop fragments: keep the largest 4-connected piece of each label
  const n = shapes.length;
  for (let k = 1; k <= n; k++) {
    const m = new Uint8Array(N * N); for (let i = 0; i < m.length; i++) m[i] = lab[i] === k ? 1 : 0;
    const L = label2d(m, N, N, 4);
    if (L.count <= 1) continue;
    const size = new Int32Array(L.count + 1); for (let i = 0; i < m.length; i++) size[L.data[i]]++;
    let big = 1; for (let c = 2; c <= L.count; c++) if (size[c] > size[big]) big = c;
    for (let i = 0; i < m.length; i++) if (m[i] && L.data[i] !== big) lab[i] = 0;
  }
  return lab;
}

export function makeScene(id: SceneId): Scene {
  const shapes = id === "crowded" ? crowdedShapes() : mixedShapes();
  const labels = rasterise(shapes), n = shapes.length;
  const r = rng(id === "crowded" ? 5 : 3);
  const level = shapes.map(s => (s.dim ? 62 : 150 + 50 * r.unif()));
  const img = createGrey(N, N, 30);
  for (let i = 0; i < N * N; i++) if (labels[i]) img.data[i] = level[labels[i] - 1];
  // a little texture and the touching contacts slightly darker (as in real nuclei)
  const tex = createGrey(N, N); for (let i = 0; i < tex.data.length; i++) tex.data[i] = r.norm();
  const tb = gaussianBlur(tex, 1.5);
  for (let i = 0; i < N * N; i++) if (labels[i]) img.data[i] += 25 * tb.data[i] / 0.19;
  const image = addGaussianNoise(gaussianBlur(img, 1.2), 6, 11);
  return { shapes, labels, n, image, dim: Uint8Array.from(shapes, s => (s.dim ? 1 : 0)) };
}

// ---------------------------------------------------------------------------------------
// Targets

/** Contact pixels: in a cell, with a 4-neighbour in a different cell. */
export function contacts(lab: Int32Array): Uint8Array {
  const c = new Uint8Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const p = y * N + x, l = lab[p];
    if (!l) continue;
    if ((x > 0 && lab[p - 1] && lab[p - 1] !== l) || (x < N - 1 && lab[p + 1] && lab[p + 1] !== l) ||
        (y > 0 && lab[p - N] && lab[p - N] !== l) || (y < N - 1 && lab[p + N] && lab[p + N] !== l)) c[p] = 1;
  }
  return c;
}

/** Boundary pixels: in a cell, with a 4-neighbour of another label (background included) or on the image edge. */
export function boundaryPixels(lab: Int32Array): Uint8Array {
  const b = new Uint8Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const p = y * N + x, l = lab[p];
    if (!l) continue;
    if (x === 0 || y === 0 || x === N - 1 || y === N - 1 || lab[p - 1] !== l || lab[p + 1] !== l || lab[p - N] !== l || lab[p + N] !== l) b[p] = 1;
  }
  return b;
}

/** Distance to the border of the pixel's own cell, divided by the cell's maximum (0 outside). */
export function distanceTarget(lab: Int32Array, n: number): Float32Array {
  const m = new Uint8Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const p = y * N + x, l = lab[p];
    m[p] = l && !(x > 0 && lab[p - 1] !== l && lab[p - 1]) && !(y > 0 && lab[p - N] !== l && lab[p - N]) ? 1 : 0;
  }
  // a pixel next to another cell counts as border: remove one side of every contact
  const d = edt(m, N, N);
  const mx = new Float32Array(n + 1);
  for (let i = 0; i < N * N; i++) if (lab[i] && d[i] > mx[lab[i]]) mx[lab[i]] = d[i];
  const out = new Float32Array(N * N);
  for (let i = 0; i < N * N; i++) if (lab[i]) out[i] = d[i] / (mx[lab[i]] || 1);
  return out;
}

/** Unit vector of ray k of n: angle 2πk/n. */
export const rayDir = (k: number, n: number): [number, number] => [Math.cos(2 * Math.PI * k / n), Math.sin(2 * Math.PI * k / n)];

/**
 * Star-convex distances: for every pixel of a cell, the distance along each of MAX_RAYS rays
 * to the first point outside the cell (marching in steps of 0.25 px). Layout: [p * MAX_RAYS + k].
 */
export function rayTarget(lab: Int32Array): Float32Array {
  const R = new Float32Array(N * N * MAX_RAYS), step = 0.25;
  for (let p = 0; p < N * N; p++) {
    const l = lab[p];
    if (!l) continue;
    const x0 = p % N, y0 = (p - x0) / N;
    for (let k = 0; k < MAX_RAYS; k++) {
      const [dx, dy] = rayDir(k, MAX_RAYS);
      let t = 0;
      for (;;) {
        const tt = t + step, xi = Math.round(x0 + tt * dx), yi = Math.round(y0 + tt * dy);
        if (xi < 0 || yi < 0 || xi >= N || yi >= N || lab[yi * N + xi] !== l) break;
        t = tt;
      }
      R[p * MAX_RAYS + k] = t + 0.5;
    }
  }
  return R;
}

/**
 * Flow target: for each cell, heat diffusion with a source at the cell's centre (the pixel
 * deepest inside, ties broken towards the centroid), confined to the cell; the flow is the
 * normalised gradient of log h. Returns vx, vy (0 outside cells) and the centres.
 */
export function flowTarget(lab: Int32Array, n: number, D: Float32Array): { vx: Float32Array; vy: Float32Array; centre: Int32Array } {
  const vx = new Float32Array(N * N), vy = new Float32Array(N * N), centre = new Int32Array(n + 1).fill(-1);
  const pix: number[][] = Array.from({ length: n + 1 }, () => []);
  for (let p = 0; p < N * N; p++) if (lab[p]) pix[lab[p]].push(p);
  for (let l = 1; l <= n; l++) {
    const P = pix[l];
    if (!P.length) continue;
    let cx = 0, cy = 0; for (const p of P) { cx += p % N; cy += Math.floor(p / N); } cx /= P.length; cy /= P.length;
    let best = P[0], bv = -Infinity;
    for (const p of P) { const v = D[p] - 1e-3 * Math.hypot((p % N) - cx, Math.floor(p / N) - cy); if (v > bv) { bv = v; best = p; } }
    centre[l] = best;
    let x0 = N, x1 = 0, y0 = N, y1 = 0;
    for (const p of P) { const x = p % N, y = (p - x) / N; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    const iters = 2 * ((x1 - x0) + (y1 - y0)) + 20;
    let h = new Float64Array(N * N), h2 = new Float64Array(N * N);
    for (let it = 0; it < iters; it++) {
      h[best] += 1;
      for (const p of P) {
        const x = p % N, y = (p - x) / N;
        let s = h[p], c = 1;
        if (x > 0 && lab[p - 1] === l) { s += h[p - 1]; c++; }
        if (x < N - 1 && lab[p + 1] === l) { s += h[p + 1]; c++; }
        if (y > 0 && lab[p - N] === l) { s += h[p - N]; c++; }
        if (y < N - 1 && lab[p + N] === l) { s += h[p + N]; c++; }
        h2[p] = s / c;
      }
      [h, h2] = [h2, h];
    }
    const lg = (q: number) => Math.log(h[q] + 1e-30);
    for (const p of P) {
      const x = p % N, y = (p - x) / N, c = lg(p);
      const gx = ((x < N - 1 && lab[p + 1] === l ? lg(p + 1) : c) - (x > 0 && lab[p - 1] === l ? lg(p - 1) : c));
      const gy = ((y < N - 1 && lab[p + N] === l ? lg(p + N) : c) - (y > 0 && lab[p - N] === l ? lg(p - N) : c));
      const m = Math.hypot(gx, gy);
      if (p !== best && m > 0) { vx[p] = gx / m; vy[p] = gy / m; }
    }
  }
  return { vx, vy, centre };
}

export interface Targets {
  fg: Float32Array; bnd: Uint8Array; D: Float32Array; rays: Float32Array;
  vx: Float32Array; vy: Float32Array; centre: Int32Array;
  /** The same targets for the merged mask (every group of touching cells as one): what a
   *  network shows under contact bias. Computed on first use (mergedTargets). */
  Du?: Float32Array; raysU?: Float32Array; vxU?: Float32Array; vyU?: Float32Array;
  /** Weight of the contact zone (1 at a contact, fading over a few pixels). */
  zone: Float32Array;
  merged: Int32Array; nMerged: number;
}

export function computeTargets(sc: Scene): Targets {
  const lab = sc.labels, n = sc.n;
  const fg = Float32Array.from(lab, l => (l ? 1 : 0));
  const bnd = boundaryPixels(lab);
  const D = distanceTarget(lab, n);
  const rays = rayTarget(lab);
  const fl = flowTarget(lab, n, D);
  const fgm = Uint8Array.from(lab, l => (l ? 1 : 0));
  const L = label2d(fgm, N, N, 4), merged = L.data, nMerged = L.count;
  const c = contacts(lab);
  const zg = createGrey(N, N); for (let i = 0; i < N * N; i++) zg.data[i] = c[i];
  const zb = gaussianBlur(zg, 4);
  let mx = 0; for (const v of zb.data) mx = Math.max(mx, v);
  const zone = Float32Array.from(zb.data, (v, i) => (lab[i] ? Math.min(1, 4 * v / (mx || 1)) : 0));
  return { fg, bnd, D, rays, vx: fl.vx, vy: fl.vy, centre: fl.centre, zone, merged, nMerged };
}

/** Fill in the merged-mask targets (rays and flows of every touching group as one object). */
export function mergedTargets(T: Targets): void {
  if (T.raysU) return;
  T.Du = distanceTarget(T.merged, T.nMerged);
  T.raysU = rayTarget(T.merged);
  const f = flowTarget(T.merged, T.nMerged, T.Du);
  T.vxU = f.vx; T.vyU = f.vy;
}

// ---------------------------------------------------------------------------------------
// Simulated network output

export interface Errors { noise: number; blur: number; contact: number; missed: number }

/** A smooth seeded random field with unit standard deviation. */
export function smoothField(seed: number, sigma = 1.5): Float32Array {
  const g = createGrey(N, N), r = rng(seed);
  for (let i = 0; i < g.data.length; i++) g.data[i] = r.norm();
  const b = gaussianBlur(g, sigma);
  let s = 0; for (const v of b.data) s += v * v;
  const k = 1 / Math.sqrt(s / b.data.length);
  return Float32Array.from(b.data, v => v * k);
}

const blurArr = (a: Float32Array, s: number): Float32Array => (s > 0 ? gaussianBlur({ width: N, height: N, data: a }, s).data : a);

export type Route = "semantic" | "boundary" | "distance" | "star" | "flows";

export interface Prediction {
  route: Route;
  fg: Float32Array;                  // P(cell)
  pInt?: Float32Array; pBnd?: Float32Array;
  D?: Float32Array;                  // distance / objectness
  rays?: Float32Array;               // [p * MAX_RAYS + k]
  vx?: Float32Array; vy?: Float32Array;
}

/** Fading factor for missed dim cells: 1 − missed on dim cells, 1 elsewhere. */
function dimFactor(sc: Scene, missed: number): Float32Array {
  return Float32Array.from(sc.labels, l => (l && sc.dim[l - 1] ? 1 - missed : 1));
}

export function predict(sc: Scene, T: Targets, route: Route, e: Errors): Prediction {
  const f1 = smoothField(101), f2 = smoothField(202), f3 = smoothField(303);
  const dimF = dimFactor(sc, e.missed);
  const fgRaw = Float32Array.from(T.fg, (v, i) => v * dimF[i]);
  const fg = blurArr(fgRaw, e.blur).map((v, i) => Math.min(1, Math.max(0, v + 0.2 * e.noise * f3[i])));
  const z = T.zone, c = e.contact;
  if (c > 0 && (route === "star" || route === "flows")) mergedTargets(T);
  if (route === "semantic") return { route, fg };
  if (route === "boundary") {
    // contact bias: the boundary class fades where cells touch
    const b0 = Float32Array.from(T.bnd, (v, i) => v * (1 - c * z[i]) * dimF[i]);
    const i0 = Float32Array.from(T.fg, (v, i) => (v - T.bnd[i] + c * z[i] * T.bnd[i]) * dimF[i]);
    const bb = blurArr(b0, e.blur), ib = blurArr(i0, e.blur);
    const pBnd = new Float32Array(N * N), pInt = new Float32Array(N * N);
    for (let i = 0; i < N * N; i++) {
      const b = Math.max(0, bb[i] + 0.35 * e.noise * f1[i]), it = Math.max(0, ib[i] + 0.35 * e.noise * f2[i]), bg = Math.max(0, 1 - bb[i] - ib[i]);
      const s = b + it + bg || 1;
      pBnd[i] = b / s; pInt[i] = it / s;
    }
    return { route, fg, pInt, pBnd };
  }
  // distance (also the objectness of the star route): contact bias = the network does not see
  // the contact and fills the neck up to the height of the cell centres
  const Dm = Float32Array.from(T.D, (v, i) => ((1 - c * z[i]) * v + c * z[i] * Math.max(v, 1)) * dimF[i]);
  const D = blurArr(Dm, e.blur).map((v, i) => Math.max(0, v + 0.25 * e.noise * f1[i] * (T.fg[i] ? 1 : 0.4)));
  if (route === "distance") return { route, fg, D };
  if (route === "star") {
    const rays = new Float32Array(N * N * MAX_RAYS), r = rng(55);
    for (let p = 0; p < N * N; p++) {
      if (!T.fg[p]) continue;
      const w = c * z[p];
      for (let k = 0; k < MAX_RAYS; k++) {
        const v = w > 0 ? (1 - w) * T.rays[p * MAX_RAYS + k] + w * T.raysU![p * MAX_RAYS + k] : T.rays[p * MAX_RAYS + k];
        rays[p * MAX_RAYS + k] = Math.max(0.5, v * (1 + 0.35 * e.noise * (f2[p] + r.norm())));
      }
    }
    return { route, fg, D, rays };
  }
  // flows
  let vx: Float32Array = c > 0 ? Float32Array.from(T.vx, (v, i) => (1 - c * z[i]) * v + c * z[i] * T.vxU![i]) : T.vx.slice();
  let vy: Float32Array = c > 0 ? Float32Array.from(T.vy, (v, i) => (1 - c * z[i]) * v + c * z[i] * T.vyU![i]) : T.vy.slice();
  vx = blurArr(vx, e.blur); vy = blurArr(vy, e.blur);
  for (let i = 0; i < N * N; i++) {
    const a = vx[i] + 0.9 * e.noise * f1[i], b = vy[i] + 0.9 * e.noise * f2[i], m = Math.hypot(a, b);
    vx[i] = m > 0 ? a / m : 0; vy[i] = m > 0 ? b / m : 0;
  }
  return { route, fg, vx, vy };
}

// ---------------------------------------------------------------------------------------
// Post-processing

export interface Post { thr: number; h: number; nRays: number; nmsT: number; steps: number }

export interface Result {
  labels: Int32Array; count: number;
  /** distance route: flooding level of every pixel (for the animation). */
  level?: Float64Array;
  /** star route: the kept polygons (centre pixel, n rays) in the order they were kept. */
  polygons?: { p: number; r: number[] }[];
  /** flow route: end point of every pixel after `steps` steps (NaN outside). */
  endX?: Float32Array; endY?: Float32Array;
}

function relabelSmall(lab: Int32Array, minSize: number): { labels: Int32Array; count: number } {
  let mx = 0; for (const v of lab) mx = Math.max(mx, v);
  const size = new Int32Array(mx + 1); for (const v of lab) size[v]++;
  const map = new Int32Array(mx + 1); let c = 0;
  for (let k = 1; k <= mx; k++) if (size[k] >= minSize) map[k] = ++c;
  return { labels: Int32Array.from(lab, v => map[v]), count: c };
}

/** Grow labelled seeds into the mask (breadth-first, 4-neighbours): each mask pixel gets the nearest seed's label. */
export function growInto(seeds: Int32Array, mask: Uint8Array): Int32Array {
  const lab = Int32Array.from(seeds), q: number[] = [];
  for (let p = 0; p < N * N; p++) if (lab[p]) q.push(p);
  for (let h = 0; h < q.length; h++) {
    const p = q[h], x = p % N, y = (p - x) / N;
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= N || yy >= N) continue;
      const r = yy * N + xx;
      if (mask[r] && !lab[r]) { lab[r] = lab[p]; q.push(r); }
    }
  }
  return lab;
}

/** Pixels of the star-convex polygon with centre p and rays r (straight edges between the ray tips). */
export function polygonPixels(p: number, r: ArrayLike<number>): number[] {
  const n = r.length, x0 = p % N, y0 = (p - x0) / N;
  let R = 0; for (let k = 0; k < n; k++) R = Math.max(R, r[k]);
  const tips = Array.from({ length: n }, (_, k) => { const [dx, dy] = rayDir(k, n); return [x0 + r[k] * dx, y0 + r[k] * dy]; });
  const out: number[] = [];
  for (let y = Math.max(0, Math.floor(y0 - R)); y <= Math.min(N - 1, Math.ceil(y0 + R)); y++)
    for (let x = Math.max(0, Math.floor(x0 - R)); x <= Math.min(N - 1, Math.ceil(x0 + R)); x++) {
      const dx = x - x0, dy = y - y0;
      if (!dx && !dy) { out.push(y * N + x); continue; }
      let th = Math.atan2(dy, dx); if (th < 0) th += 2 * Math.PI;
      const k = Math.floor(th / (2 * Math.PI / n)) % n, k2 = (k + 1) % n;
      const [ax, ay] = tips[k], [bx, by] = tips[k2];
      // inside the triangle (centre, tip k, tip k+1): same side of the edge as the centre
      const cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax), crossC = (bx - ax) * (y0 - ay) - (by - ay) * (x0 - ax);
      if (cross * crossC >= 0) out.push(y * N + x);
    }
  return out;
}

export function postprocess(pr: Prediction, o: Post): Result {
  if (pr.route === "semantic") {
    const m = Uint8Array.from(pr.fg, v => (v > o.thr ? 1 : 0));
    const L = label2d(m, N, N, 4);
    return relabelSmall(L.data, 6);
  }
  if (pr.route === "boundary") {
    const inter = new Uint8Array(N * N), fgm = new Uint8Array(N * N);
    for (let i = 0; i < N * N; i++) {
      const pi = pr.pInt![i], pb = pr.pBnd![i], pbg = 1 - pi - pb;
      if (pi >= pb && pi >= pbg) inter[i] = 1;
      if (pi + pb > pbg) fgm[i] = 1;
    }
    const L = label2d(inter, N, N, 4);
    const seeds = relabelSmall(L.data, 4);
    const grown = growInto(seeds.labels, fgm);
    return { labels: grown, count: seeds.count };
  }
  if (pr.route === "distance") {
    const D = pr.D!, mask = Uint8Array.from(D, v => (v > o.thr ? 1 : 0));
    const Dm = Float32Array.from(D, (v, i) => (mask[i] ? v : 0));
    const mk = hMaxima(Dm, N, N, o.h, 8);
    for (let i = 0; i < N * N; i++) if (!mask[i]) mk[i] = 0;
    const M = label2d(mk, N, N, 8);
    const neg = Float32Array.from(Dm, v => -v);
    const fr = floodWatershed(neg, N, N, M.data, { mask, conn: 8 });
    const s = relabelSmall(fr.labels, 6);
    return { labels: s.labels, count: s.count, level: fr.level };
  }
  if (pr.route === "star") {
    const n = o.nRays, D = pr.D!, step = MAX_RAYS / n;
    const cands: number[] = [];
    for (let y = 1; y < N; y += 2) for (let x = 1; x < N; x += 2) { const p = y * N + x; if (D[p] > o.thr) cands.push(p); }
    const polys = cands.map(p => Array.from({ length: n }, (_, k) => pr.rays![p * MAX_RAYS + k * step]));
    const pix = cands.map((p, i) => polygonPixels(p, polys[i]));
    const scores = cands.map(p => D[p]);
    const stamp = new Int32Array(N * N).fill(-1);
    let stamped = -1;
    const overlap = (i: number, j: number): number => {
      if (stamped !== i) { for (const q of pix[i]) stamp[q] = i; stamped = i; }
      let inter = 0; for (const q of pix[j]) if (stamp[q] === i) inter++;
      return inter / (pix[i].length + pix[j].length - inter);
    };
    const res = nmsGeneric(cands.length, scores, overlap, o.nmsT);
    const lab = new Int32Array(N * N);
    // paint from the lowest to the highest score, so the best polygon wins where they overlap
    const kept = res.keep;
    for (let k = kept.length - 1; k >= 0; k--) for (const q of pix[kept[k]]) lab[q] = k + 1;
    const s = relabelSmall(lab, 6);
    return { labels: s.labels, count: s.count, polygons: kept.map(i => ({ p: cands[i], r: polys[i] })) };
  }
  // flows: Euler steps along the predicted field, then group pixels by their end points
  const fgm = Uint8Array.from(pr.fg, v => (v > o.thr ? 1 : 0));
  const ex = new Float32Array(N * N).fill(NaN), ey = new Float32Array(N * N).fill(NaN);
  const vx = pr.vx!, vy = pr.vy!;
  const samp = (a: Float32Array, x: number, y: number) => {
    const xi = Math.max(0, Math.min(N - 2, Math.floor(x))), yi = Math.max(0, Math.min(N - 2, Math.floor(y)));
    const fx = Math.max(0, Math.min(1, x - xi)), fy = Math.max(0, Math.min(1, y - yi)), p = yi * N + xi;
    return (1 - fy) * ((1 - fx) * a[p] + fx * a[p + 1]) + fy * ((1 - fx) * a[p + N] + fx * a[p + N + 1]);
  };
  const end = flowEnds(fgm, vx, vy, o.steps, samp);
  ex.set(end.x); ey.set(end.y);
  const cnt = new Uint8Array(N * N);
  for (let p = 0; p < N * N; p++) if (fgm[p]) {
    const xi = Math.round(ex[p]), yi = Math.round(ey[p]);
    if (xi >= 0 && yi >= 0 && xi < N && yi < N) cnt[yi * N + xi] = 1;
  }
  // sinks: end points that land within one pixel of each other are one sink
  const dil = new Uint8Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (cnt[y * N + x])
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const xx = x + dx, yy = y + dy; if (xx >= 0 && yy >= 0 && xx < N && yy < N) dil[yy * N + xx] = 1; }
  const S = label2d(dil, N, N, 8);
  const lab = new Int32Array(N * N);
  for (let p = 0; p < N * N; p++) if (fgm[p]) {
    const xi = Math.round(ex[p]), yi = Math.round(ey[p]);
    if (xi >= 0 && yi >= 0 && xi < N && yi < N) lab[p] = S.data[yi * N + xi];
  }
  const s = relabelSmall(lab, 12);
  return { labels: s.labels, count: s.count, endX: ex, endY: ey };
}

/** End points of x_{j+1} = x_j + v(x_j) (step length 1) from every mask pixel after `steps` steps. */
export function flowEnds(mask: Uint8Array, vx: Float32Array, vy: Float32Array, steps: number,
  samp: (a: Float32Array, x: number, y: number) => number): { x: Float32Array; y: Float32Array } {
  const X = new Float32Array(N * N).fill(NaN), Y = new Float32Array(N * N).fill(NaN);
  for (let p = 0; p < N * N; p++) {
    if (!mask[p]) continue;
    let x = p % N, y = (p - x) / N;
    for (let j = 0; j < steps; j++) {
      const u = samp(vx, x, y), v = samp(vy, x, y);
      x = Math.max(0, Math.min(N - 1, x + u)); y = Math.max(0, Math.min(N - 1, y + v));
    }
    X[p] = x; Y[p] = y;
  }
  return { x: X, y: Y };
}
