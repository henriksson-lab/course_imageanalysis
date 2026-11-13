// 2D transforms between images, fitted to point pairs, robustly with RANSAC; synthetic
// scenes, views and mosaics with ground truth; warping and blending (no DOM).
// Owner: the keypoints day (feature-alignment/).
//
// A Transform2D is a 3 × 3 row-major Float64Array acting on homogeneous (x, y, 1), mapping
// image 1 → image 2. Kinds: translation (2 numbers, 1 pair), rigid (3, 2), similarity (4, 2),
// affine (6, 3), homography (8, 4). Fits: translation = mean difference; rigid and
// similarity = Procrustes (lib/shapes.ts); affine = linear least squares; homography = the
// normalised DLT (lib/camera.ts). RANSAC over matched pairs reuses drawSample and
// iterationsNeeded from lib/ransac.ts (its Model interface takes points, not pairs, so the
// loop is written here for pairs).
//
// Rotation angles are counter-clockwise on the screen (y down): R(α) = [[cos α, sin α], [−sin α, cos α]].

import { Grey, createGrey, sampleBilinear, gaussianBlur, laplacianPyramid, gaussianPyramid, collapsePyramid, combine } from "./imageops";
import { procrustes } from "./shapes";
import { homographyDLT, mul3 } from "./camera";
import { inv3, solve, lstsq } from "./linalg";
import { rng, Rng } from "./random";
import { drawSample, iterationsNeeded } from "./ransac";

export type Transform2D = Float64Array;
export type Kind = "translation" | "rigid" | "similarity" | "affine" | "homography";
export const KINDS: Kind[] = ["translation", "rigid", "similarity", "affine", "homography"];
export interface P2 { x: number; y: number }

export const dof = (k: Kind): number => ({ translation: 2, rigid: 3, similarity: 4, affine: 6, homography: 8 })[k];
export const minimalSize = (k: Kind): number => ({ translation: 1, rigid: 2, similarity: 2, affine: 3, homography: 4 })[k];

export const identity = (): Transform2D => Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);
export const translation = (tx: number, ty: number): Transform2D => Float64Array.from([1, 0, tx, 0, 1, ty, 0, 0, 1]);
/** Similarity about (cx, cy): zoom s, rotation α counter-clockwise on the screen, then shift (tx, ty). */
export function similarity(s: number, alpha: number, cx = 0, cy = 0, tx = 0, ty = 0): Transform2D {
  const c = s * Math.cos(alpha), d = s * Math.sin(alpha);
  // x' = c (x − cx) + d (y − cy) + cx + tx ; y' = −d (x − cx) + c (y − cy) + cy + ty
  return Float64Array.from([c, d, cx + tx - c * cx - d * cy, -d, c, cy + ty + d * cx - c * cy, 0, 0, 1]);
}
export const compose = (A: Transform2D, B: Transform2D): Transform2D => mul3(A, B);   // A ∘ B (B first)
export function invert(T: Transform2D): Transform2D {
  const I = inv3(T);
  if (!I) throw new Error("singular transform");
  return Float64Array.from(I);
}
export function apply(T: ArrayLike<number>, x: number, y: number): [number, number] {
  const w = T[6] * x + T[7] * y + T[8];
  return [(T[0] * x + T[1] * y + T[2]) / w, (T[3] * x + T[4] * y + T[5]) / w];
}
export const mapOf = (T: ArrayLike<number>) => (x: number, y: number): [number, number] => apply(T, x, y);
export function transferError(T: ArrayLike<number>, p: P2, q: P2): number {
  const [x, y] = apply(T, p.x, p.y); return Math.hypot(x - q.x, y - q.y);
}
export function symmetricTransferError(T: Transform2D, p: P2, q: P2): number {
  return transferError(T, p, q) + transferError(invert(T), q, p);
}
/** Camera turning about its centre: K R K⁻¹ for a rotation by `tilt` about the horizontal axis and `pan` about the vertical axis (focal f, centre (cx, cy)). */
export function rotatingCamera(f: number, cx: number, cy: number, tilt: number, pan = 0): Transform2D {
  const K = Float64Array.from([f, 0, cx, 0, f, cy, 0, 0, 1]);
  const ct = Math.cos(tilt), st = Math.sin(tilt), cp = Math.cos(pan), sp = Math.sin(pan);
  const Rx = Float64Array.from([1, 0, 0, 0, ct, -st, 0, st, ct]);
  const Ry = Float64Array.from([cp, 0, sp, 0, 1, 0, -sp, 0, cp]);
  const H = mul3(mul3(K, mul3(Ry, Rx)), inv3(K)!);
  return Float64Array.from(H, v => v / H[8]);
}

/** Recover zoom, rotation (counter-clockwise on the screen) and shift from a similarity-like matrix. */
export function decompose(T: ArrayLike<number>): { s: number; alpha: number; tx: number; ty: number } {
  const a = T[0] / T[8], b = T[1] / T[8];
  return { s: Math.hypot(a, b), alpha: Math.atan2(b, a), tx: T[2] / T[8], ty: T[5] / T[8] };
}

// ---------------------------------------------------------------------------------------
// Fits

const area2 = (a: P2, b: P2, c: P2): number => Math.abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));

/** Degenerate minimal samples: coincident points (2-point fits), collinear triples (affine, homography). */
export function degenerate(kind: Kind, src: P2[], dst: P2[], tol = 1): boolean {
  const n = src.length;
  for (const P of [src, dst]) {
    if (kind === "rigid" || kind === "similarity") for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (Math.hypot(P[i].x - P[j].x, P[i].y - P[j].y) < tol) return true;
    if (kind === "affine" || kind === "homography")
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) for (let k = j + 1; k < n; k++) if (area2(P[i], P[j], P[k]) < tol * 2) return true;
  }
  return false;
}

/** Least-squares fit of a transform of the given kind (null when there are too few pairs or they are degenerate). */
export function fitTransform(kind: Kind, src: P2[], dst: P2[]): Transform2D | null {
  const n = src.length;
  if (n < minimalSize(kind)) return null;
  switch (kind) {
    case "translation": {
      let tx = 0, ty = 0;
      for (let i = 0; i < n; i++) { tx += (dst[i].x - src[i].x) / n; ty += (dst[i].y - src[i].y) / n; }
      return translation(tx, ty);
    }
    case "rigid": case "similarity": {
      if (n === 2 && Math.hypot(src[0].x - src[1].x, src[0].y - src[1].y) < 1e-9) return null;
      const p = new Float64Array(2 * n), q = new Float64Array(2 * n);
      src.forEach((s, i) => { p[2 * i] = s.x; p[2 * i + 1] = s.y; });
      dst.forEach((s, i) => { q[2 * i] = s.x; q[2 * i + 1] = s.y; });
      const f = procrustes(p, q, { scale: kind === "similarity" });
      // aligned = [[a, −b], [b, a]] p + t (y down)
      return Float64Array.from([f.a, -f.b, f.tx, f.b, f.a, f.ty, 0, 0, 1]);
    }
    case "affine": {
      // x' = a11 x + a12 y + tx and y' = a21 x + a22 y + ty: two n × 3 problems with the same matrix
      const M = new Float64Array(9), bx = new Float64Array(3), by = new Float64Array(3);
      for (let i = 0; i < n; i++) {
        const r = [src[i].x, src[i].y, 1];
        for (let a = 0; a < 3; a++) { for (let b = 0; b < 3; b++) M[3 * a + b] += r[a] * r[b]; bx[a] += r[a] * dst[i].x; by[a] += r[a] * dst[i].y; }
      }
      const ax = solve(M, bx, 3), ay = solve(M, by, 3);
      if (!ax || !ay) return null;
      const T = Float64Array.from([ax[0], ax[1], ax[2], ay[0], ay[1], ay[2], 0, 0, 1]);
      return T.every(Number.isFinite) && Math.abs(T[0] * T[4] - T[1] * T[3]) > 1e-9 ? T : null;
    }
    case "homography": {
      const { H, S } = homographyDLT(src.map(p => [p.x, p.y]), dst.map(p => [p.x, p.y]));
      if (!H.every(Number.isFinite) || S[7] < 1e-9 * S[0]) return null;
      return Float64Array.from(H);
    }
  }
}

/** Gauss–Newton on the transfer error for a homography (h33 = 1), a few steps from T0. */
export function refineHomographyGN(T0: Transform2D, src: P2[], dst: P2[], steps = 5): Transform2D {
  let h = Float64Array.from(T0, v => v / T0[8]);
  const n = src.length;
  for (let it = 0; it < steps; it++) {
    const J = new Float64Array(2 * n * 8), r = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
      const { x, y } = src[i], w = h[6] * x + h[7] * y + 1, u = (h[0] * x + h[1] * y + h[2]) / w, v = (h[3] * x + h[4] * y + h[5]) / w;
      r[2 * i] = dst[i].x - u; r[2 * i + 1] = dst[i].y - v;
      const ru = 2 * i * 8, rv = (2 * i + 1) * 8;
      J[ru] = x / w; J[ru + 1] = y / w; J[ru + 2] = 1 / w; J[ru + 6] = -u * x / w; J[ru + 7] = -u * y / w;
      J[rv + 3] = x / w; J[rv + 4] = y / w; J[rv + 5] = 1 / w; J[rv + 6] = -v * x / w; J[rv + 7] = -v * y / w;
    }
    const d = lstsq(J, r, 2 * n, 8);
    for (let k = 0; k < 8; k++) h[k] += d[k];
  }
  return h;
}

export function rmsError(T: Transform2D, src: P2[], dst: P2[], idx?: number[]): number {
  const I = idx ?? src.map((_, i) => i);
  if (!I.length) return NaN;
  let s = 0; for (const i of I) s += transferError(T, src[i], dst[i]) ** 2;
  return Math.sqrt(s / I.length);
}

// ---------------------------------------------------------------------------------------
// RANSAC on matched pairs

export interface PairStep { iter: number; sample: number[]; T: Transform2D | null; inliers: number; best: number; improved: boolean }
export interface PairRansac {
  best: Transform2D | null; bestSample: number[]; inliers: number[];
  refit: Transform2D | null; refitInliers: number[];
  steps: PairStep[]; foundAt: number;
}

export function inliersOfPairs(T: Transform2D, src: P2[], dst: P2[], eps: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < src.length; i++) if (transferError(T, src[i], dst[i]) <= eps) out.push(i);
  return out;
}

/** RANSAC: minimal sample → exact fit → count pairs within eps → keep the best → refit on its inliers (twice). */
export function ransacTransform(kind: Kind, src: P2[], dst: P2[], o: { eps: number; iterations: number; seed?: number; r?: Rng }): PairRansac {
  const r = o.r ?? rng(o.seed ?? 1), s = minimalSize(kind), idx = src.map((_, i) => i);
  const steps: PairStep[] = [];
  let best: Transform2D | null = null, bestN = -1, bestSample: number[] = [], foundAt = 0;
  for (let it = 1; it <= o.iterations && src.length >= s; it++) {
    const sample = drawSample(r, idx, s);
    const ss = sample.map(i => src[i]), dd = sample.map(i => dst[i]);
    const T = degenerate(kind, ss, dd) ? null : fitTransform(kind, ss, dd);
    const n = T ? inliersOfPairs(T, src, dst, o.eps).length : 0;
    const improved = T !== null && n > bestN;
    if (improved) { best = T; bestN = n; bestSample = sample; foundAt = it; }
    steps.push({ iter: it, sample, T, inliers: n, best: Math.max(0, bestN), improved });
  }
  const inl = best ? inliersOfPairs(best, src, dst, o.eps) : [];
  let ref: Transform2D | null = null, refIn: number[] = [];
  if (inl.length >= s) {
    ref = fitTransform(kind, inl.map(i => src[i]), inl.map(i => dst[i]));
    if (ref) {
      refIn = inliersOfPairs(ref, src, dst, o.eps);
      if (refIn.length >= s) { const r2 = fitTransform(kind, refIn.map(i => src[i]), refIn.map(i => dst[i])); if (r2) { ref = r2; refIn = inliersOfPairs(ref, src, dst, o.eps); } }
      if (kind === "homography" && ref && refIn.length >= 5) ref = refineHomographyGN(ref, refIn.map(i => src[i]), refIn.map(i => dst[i]));
    }
  }
  return { best, bestSample, inliers: inl, refit: ref, refitInliers: refIn, steps, foundAt };
}

export { iterationsNeeded };

// ---------------------------------------------------------------------------------------
// A photo-like synthetic scene

export interface SceneOptions { size?: number; seed?: number; grid?: boolean; texture?: number }

type Shape = { kind: "poly"; pts: [number, number][]; v: number } | { kind: "disc"; x: number; y: number; r: number; v: number } | { kind: "ring"; x: number; y: number; r: number; w: number; v: number };

function inPoly(pts: [number, number][], x: number, y: number): boolean {
  let c = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
  }
  return c;
}

/**
 * Seeded synthetic scene (size² grey 0..255, default 512): gentle shading, smooth texture,
 * polygons with corners (rectangles, triangles, L-shapes, stars), discs and rings of several
 * radii, and (optionally) a repetitive grid of identical small squares. Rendered with 2 × 2
 * supersampling and a slight blur, plus fine seeded texture.
 */
export function featureScene(o: SceneOptions = {}): Grey {
  const N = o.size ?? 512, r = rng(o.seed ?? 11), U = (a: number, b: number) => a + (b - a) * r.unif();
  const shapes: Shape[] = [];
  const sc = N / 512;
  // polygons
  for (let k = 0; k < 34; k++) {
    const cx = U(20, N - 20), cy = U(20, N - 20), s = U(10, 42) * sc, v = U(15, 240), t = r.unif(), a0 = U(0, Math.PI);
    let pts: [number, number][];
    if (t < 0.35) { const w = s, h = s * U(0.4, 1.2); pts = [[-w, -h], [w, -h], [w, h], [-w, h]]; }
    else if (t < 0.6) pts = [0, 1, 2].map(i => [s * Math.cos(a0 + i * 2.1 + U(-0.3, 0.3)), s * Math.sin(a0 + i * 2.1 + U(-0.3, 0.3))] as [number, number]);
    else if (t < 0.8) { const w = s, h = s * 0.45; pts = [[-w, -w], [-w + h, -w], [-w + h, w - h], [w, w - h], [w, w], [-w, w]]; }
    else { const m = 5; pts = Array.from({ length: 2 * m }, (_, i) => { const rr = i % 2 ? s * 0.45 : s; const a = a0 + i * Math.PI / m; return [rr * Math.cos(a), rr * Math.sin(a)] as [number, number]; }); }
    const ca = Math.cos(a0), sa = Math.sin(a0);
    shapes.push({ kind: "poly", pts: pts.map(([x, y]) => [cx + ca * x - sa * y, cy + sa * x + ca * y]), v });
  }
  // discs and rings at several radii
  for (let k = 0; k < 40; k++) {
    const rad = [3, 4, 6, 9, 14, 20][k % 6] * sc * U(0.85, 1.2);
    if (k % 7 === 3) shapes.push({ kind: "ring", x: U(20, N - 20), y: U(20, N - 20), r: rad + 4, w: 3, v: U(20, 235) });
    else shapes.push({ kind: "disc", x: U(15, N - 15), y: U(15, N - 15), r: rad, v: U(10, 245) });
  }
  // the repetitive grid
  if (o.grid !== false) {
    const gx = 0.62 * N, gy = 0.6 * N, step = 14 * sc;
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) {
      const x = gx + i * step, y = gy + j * step, h = 4 * sc;
      shapes.push({ kind: "poly", pts: [[x - h, y - h], [x + h, y - h], [x + h, y + h], [x - h, y + h]], v: 225 });
    }
  }
  // background: shading + smooth texture
  const img = createGrey(N, N);
  const tex = createGrey(N, N);
  for (let i = 0; i < N * N; i++) tex.data[i] = r.norm();
  const smooth = gaussianBlur(tex, 6 * sc), fine = gaussianBlur(tex, 1.2);
  let sd = 0; for (const v of smooth.data) sd += v * v; sd = Math.sqrt(sd / (N * N)) || 1;
  let sf = 0; for (const v of fine.data) sf += v * v; sf = Math.sqrt(sf / (N * N)) || 1;
  const texAmp = o.texture ?? 1;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x;
    img.data[i] = 95 + 35 * (x / N) - 20 * (y / N) + texAmp * 22 * smooth.data[i] / sd;
  }
  // shapes with 2 × 2 supersampling (bounding boxes)
  for (const s of shapes) {
    let x0: number, x1: number, y0: number, y1: number;
    if (s.kind === "poly") { x0 = Math.min(...s.pts.map(p => p[0])); x1 = Math.max(...s.pts.map(p => p[0])); y0 = Math.min(...s.pts.map(p => p[1])); y1 = Math.max(...s.pts.map(p => p[1])); }
    else { const rr = s.r + (s.kind === "ring" ? s.w : 0); x0 = s.x - rr; x1 = s.x + rr; y0 = s.y - rr; y1 = s.y + rr; }
    for (let y = Math.max(0, Math.floor(y0)); y <= Math.min(N - 1, Math.ceil(y1)); y++)
      for (let x = Math.max(0, Math.floor(x0)); x <= Math.min(N - 1, Math.ceil(x1)); x++) {
        let cov = 0;
        for (const [ox, oy] of [[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]]) {
          const px = x + ox, py = y + oy;
          let inside: boolean;
          if (s.kind === "poly") inside = inPoly(s.pts, px, py);
          else if (s.kind === "disc") inside = (px - s.x) ** 2 + (py - s.y) ** 2 <= s.r * s.r;
          else { const d = Math.hypot(px - s.x, py - s.y); inside = d <= s.r + s.w / 2 && d >= s.r - s.w / 2; }
          if (inside) cov += 0.25;
        }
        if (cov > 0) { const i = y * N + x; img.data[i] = (1 - cov) * img.data[i] + cov * s.v; }
      }
  }
  const out = gaussianBlur(img, 0.6);
  for (let i = 0; i < N * N; i++) out.data[i] = Math.max(0, Math.min(255, out.data[i] + texAmp * 3 * fine.data[i] / sf));
  return out;
}

// ---------------------------------------------------------------------------------------
// Views with a known transform

export interface ViewOptions {
  size?: number;
  gain?: number; offset?: number; gamma?: number;
  blur?: number; noise?: number; seed?: number;
}

/**
 * View of a scene: view pixel p shows the scene at V⁻¹ p (V maps scene → view; a homography
 * in general). Before sampling, the scene is blurred when V shrinks it (anti-aliasing). Then
 * the lighting change 255·gain·(v/255)^γ + offset, a blur and seeded Gaussian noise.
 */
export function viewOf(scene: Grey, V: Transform2D, o: ViewOptions = {}): Grey {
  const N = o.size ?? 256, Vi = invert(V);
  // local zoom at the view centre: |det| of the Jacobian
  const c = (N - 1) / 2, [sx, sy] = apply(Vi, c, c), [ax, ay] = apply(V, sx, sy), [bx, by] = apply(V, sx + 1, sy), [dx, dy] = apply(V, sx, sy + 1);
  const zoom = Math.sqrt(Math.abs((bx - ax) * (dy - ay) - (by - ay) * (dx - ax)));
  const src = zoom < 0.95 ? gaussianBlur(scene, 0.5 * Math.sqrt(1 / (zoom * zoom) - 1)) : scene;
  let out = createGrey(N, N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const [u, v] = apply(Vi, x, y);
    out.data[y * N + x] = sampleBilinear(src, u, v, "replicate");
  }
  const g = o.gain ?? 1, b = o.offset ?? 0, gm = o.gamma ?? 1;
  if (g !== 1 || b !== 0 || gm !== 1) for (let i = 0; i < N * N; i++) out.data[i] = 255 * g * Math.pow(Math.max(0, out.data[i]) / 255, gm) + b;
  if (o.blur && o.blur > 0) out = gaussianBlur(out, o.blur);
  if (o.noise && o.noise > 0) { const r = rng(o.seed ?? 5); for (let i = 0; i < N * N; i++) out.data[i] += o.noise * r.norm(); }
  return out;
}

export interface ViewPairSpec {
  /** Zoom of view 2 relative to view 1, rotation (degrees, counter-clockwise on the screen), shift (view-2 pixels). */
  zoom?: number; angle?: number; tx?: number; ty?: number;
  /** Perspective tilt (degrees): a camera turning about the horizontal axis. */
  tilt?: number;
  /** Turn the transform into a homography with this pan (degrees) as well. */
  pan?: number;
  /** Arbitrary extra affine part [a11, a12, a21, a22] (shear), applied before the similarity. */
  affine?: [number, number, number, number];
}

/** V1 (scene → view 1: the centre crop) and V2 (view 1 → view 2 by the spec, then V1); T12 maps view 1 → view 2. */
export function viewPair(sceneSize: number, viewSize: number, s: ViewPairSpec): { V1: Transform2D; V2: Transform2D; T12: Transform2D } {
  const sc = (sceneSize - 1) / 2, vc = (viewSize - 1) / 2;
  const V1 = translation(vc - sc, vc - sc);
  let T = similarity(s.zoom ?? 1, (s.angle ?? 0) * Math.PI / 180, vc, vc, s.tx ?? 0, s.ty ?? 0);
  if (s.affine) {
    const [a, b, c, d] = s.affine;
    const A = Float64Array.from([a, b, vc - a * vc - b * vc, c, d, vc - c * vc - d * vc, 0, 0, 1]);
    T = compose(T, A);
  }
  if (s.tilt || s.pan) {
    // a camera turning about its centre (K R K⁻¹), re-centred so that the view centre stays put and
    // its local zoom is 1: the foreshortening of a tilted plane seen around the centre
    const H = rotatingCamera(viewSize, vc, vc, (s.tilt ?? 0) * Math.PI / 180, (s.pan ?? 0) * Math.PI / 180);
    const [u, v] = apply(H, vc, vc), [a, b] = apply(H, vc + 1, vc), [c, d] = apply(H, vc, vc + 1);
    const z = Math.sqrt(Math.abs((a - u) * (d - v) - (b - v) * (c - u))) || 1;
    T = compose(compose(similarity(1 / z, 0, vc, vc), translation(vc - u, vc - v)), compose(H, T));
  }
  return { V1, V2: compose(T, V1), T12: T };
}

// ---------------------------------------------------------------------------------------
// Warping, mosaics and blending

/** Image 2 resampled into the frame of image 1 through T (image 1 → image 2); NaN outside image 2. */
export function warpToFrame(img2: Grey, T: Transform2D, w: number, h: number): Grey {
  const out = createGrey(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const [u, v] = apply(T, x, y);
    out.data[y * w + x] = u >= -0.5 && v >= -0.5 && u <= img2.width - 0.5 && v <= img2.height - 0.5 ? sampleBilinear(img2, u, v, "replicate") : NaN;
  }
  return out;
}

/** Feather weight of a tile pixel: falls linearly to 0 at the tile edges. */
export const featherWeight = (x: number, y: number, w: number, h: number): number =>
  Math.max(0, Math.min(x + 0.5, w - 0.5 - x)) * Math.max(0, Math.min(y + 0.5, h - 0.5 - y));

export type BlendMode = "none" | "feather" | "multiband";

/**
 * Compose tiles into a canvas (cw × ch, origin at canvas pixel (ox, oy) = world (0, 0)).
 * S[k] maps tile-k pixels to world coordinates. none: the last tile wins; feather: weighted
 * average; multiband: Laplacian pyramids blended with blurred winner masks (Burt–Adelson).
 * Returns the canvas (NaN where no tile) and the per-pixel tile count.
 */
export function blendTiles(tiles: Grey[], S: Transform2D[], cw: number, ch: number, ox: number, oy: number, mode: BlendMode, levels = 5): { img: Grey; count: Uint8Array } {
  const n = tiles.length, Si = S.map(invert);
  const vals = tiles.map(() => new Float32Array(cw * ch).fill(NaN)), wts = tiles.map(() => new Float32Array(cw * ch));
  const count = new Uint8Array(cw * ch);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
    const i = y * cw + x;
    for (let k = 0; k < n; k++) {
      const [u, v] = apply(Si[k], x - ox, y - oy), t = tiles[k];
      if (u < -0.5 || v < -0.5 || u > t.width - 0.5 || v > t.height - 0.5) continue;
      vals[k][i] = sampleBilinear(t, u, v, "replicate"); wts[k][i] = featherWeight(u, v, t.width, t.height) + 1e-6; count[i]++;
    }
  }
  const out = createGrey(cw, ch).data.fill(NaN);
  const feath = new Float32Array(cw * ch).fill(NaN);
  for (let i = 0; i < cw * ch; i++) {
    let s = 0, ws = 0, last = NaN;
    for (let k = 0; k < n; k++) if (wts[k][i] > 0) { s += wts[k][i] * vals[k][i]; ws += wts[k][i]; last = vals[k][i]; }
    if (ws > 0) { feath[i] = s / ws; out[i] = mode === "none" ? last : s / ws; }
  }
  if (mode === "multiband") {
    // winner masks (largest feather weight), images filled outside their support with the feathered mosaic
    const lev = Math.max(1, levels);
    const fill = (a: Float32Array): Grey => ({ width: cw, height: ch, data: Float32Array.from(a, (v, i) => Number.isFinite(v) ? v : (Number.isFinite(feath[i]) ? feath[i] : 0)) });
    let acc: Grey[] | null = null, macc: Grey[] | null = null;
    for (let k = 0; k < n; k++) {
      const mask = createGrey(cw, ch);
      for (let i = 0; i < cw * ch; i++) {
        if (!(wts[k][i] > 0)) continue;
        let win = true; for (let q = 0; q < n; q++) if (q !== k && wts[q][i] > wts[k][i]) { win = false; break; }
        mask.data[i] = win ? 1 : 0;
      }
      const L = laplacianPyramid(fill(vals[k]), lev), M = gaussianPyramid(mask, lev);
      const prod = L.map((l, j) => combine(l, M[j], (a, b) => a * b));
      acc = acc ? acc.map((a, j) => combine(a, prod[j], (u, v) => u + v)) : prod;
      macc = macc ? macc.map((a, j) => combine(a, M[j], (u, v) => u + v)) : M;
    }
    const blended = acc!.map((a, j) => combine(a, macc![j], (u, v) => v > 1e-6 ? u / v : 0));
    const res = collapsePyramid(blended);
    for (let i = 0; i < cw * ch; i++) out[i] = Number.isFinite(feath[i]) ? res.data[i] : NaN;
  }
  return { img: { width: cw, height: ch, data: out }, count };
}

// ---------------------------------------------------------------------------------------
// Global alignment of similarity tiles (linear least squares)

export interface TileMatch { i: number; j: number; p: P2[]; q: P2[] }

/**
 * Similarities S_k (tile k → world = tile 0) minimising Σ |S_i p − S_j q|² over all matched
 * pairs (p in tile i, q in tile j) of all tile pairs, with S_0 = identity. S = [[a, −b, tx],
 * [b, a, ty]]: linear in (a, b, tx, ty), so one least-squares solve.
 */
export function globalSimilarities(nTiles: number, links: TileMatch[]): Transform2D[] {
  const nu = 4 * (nTiles - 1), rows: number[][] = [], rhs: number[] = [];
  const col = (k: number) => 4 * (k - 1);
  for (const L of links) for (let m = 0; m < L.p.length; m++) {
    // S_i p − S_j q = 0, two rows (x and y)
    for (const comp of [0, 1]) {
      const row = new Array(nu).fill(0); let b = 0;
      const add = (k: number, pt: P2, sign: number) => {
        // x: a x − b y + tx ; y: b x + a y + ty
        const vals = comp === 0 ? [pt.x, -pt.y, 1, 0] : [pt.y, pt.x, 0, 1];
        if (k === 0) b -= sign * (comp === 0 ? pt.x : pt.y);
        else for (let t = 0; t < 4; t++) row[col(k) + t] += sign * vals[t];
      };
      add(L.i, L.p[m], 1); add(L.j, L.q[m], -1);
      rows.push(row); rhs.push(b);
    }
  }
  const A = new Float64Array(rows.length * nu);
  rows.forEach((r, i) => r.forEach((v, j) => { A[i * nu + j] = v; }));
  const x = lstsq(A, Float64Array.from(rhs), rows.length, nu);
  const out: Transform2D[] = [identity()];
  for (let k = 1; k < nTiles; k++) { const [a, b, tx, ty] = [x[col(k)], x[col(k) + 1], x[col(k) + 2], x[col(k) + 3]]; out.push(Float64Array.from([a, -b, tx, b, a, ty, 0, 0, 1])); }
  return out;
}

/** Chain pairwise transforms T_{ij} (tile i → tile j) along a list of edges from tile 0: S_j = S_i ∘ T_ij⁻¹. */
export function chainTransforms(nTiles: number, edges: { i: number; j: number; T: Transform2D }[]): (Transform2D | null)[] {
  const S: (Transform2D | null)[] = new Array(nTiles).fill(null);
  S[0] = identity();
  let changed = true;
  while (changed) {
    changed = false;
    for (const e of edges) {
      if (S[e.i] && !S[e.j]) { S[e.j] = compose(S[e.i]!, invert(e.T)); changed = true; }
      else if (S[e.j] && !S[e.i]) { S[e.i] = compose(S[e.j]!, e.T); changed = true; }
    }
  }
  return S;
}

/**
 * A repetitive scene (size², default 512): a regular array of identical rounded wells (like a
 * multi-well plate or a lattice of cells) over gentle shading and fine texture, with a few
 * unique shapes near the border. Locally distinctive, globally ambiguous.
 */
export function gridScene(o: { size?: number; seed?: number; spacing?: number } = {}): Grey {
  const N = o.size ?? 512, sp = (o.spacing ?? 26) * N / 512, r = rng(o.seed ?? 3);
  const base = featureScene({ size: N, seed: (o.seed ?? 3) + 100, grid: false, texture: 0.6 });
  const out = createGrey(N, N);
  const c0 = 0.18 * N, c1 = 0.82 * N, rad = 0.3 * sp;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x;
    let v = base.data[i];
    if (x > c0 - sp / 2 && x < c1 + sp / 2 && y > c0 - sp / 2 && y < c1 + sp / 2) {
      v = 120 + 10 * (x / N) + 2 * r.norm();
      const u = (x - c0) / sp, w = (y - c0) / sp, du = (u - Math.round(u)) * sp, dw = (w - Math.round(w)) * sp;
      const d = Math.hypot(du, dw);
      const cov = Math.max(0, Math.min(1, rad + 0.5 - d));
      v = (1 - cov) * v + cov * 220;
      // a small dark mark in every well, identical in all of them
      const m = Math.max(0, Math.min(1, 1.6 - Math.hypot(du - 2 * N / 512, dw - 1.5 * N / 512)));
      v = (1 - m) * v + m * 60;
    }
    out.data[i] = v;
  }
  return gaussianBlur(out, 0.6);
}
