// Geometry and masks for the distws demo (no DOM; tested in demos/test/watershed-test.ts).

/** Neck depth of two touching discs of radius r with centres d apart (0 ≤ d ≤ 2r):
 *  the peaks of the distance transform have height r, the neck (midpoint) √(r² − d²/4). */
export function neckDepth(r: number, d: number): number {
  return r - Math.sqrt(Math.max(0, r * r - d * d / 4));
}

/** Smallest centre distance d at which two discs of radius r split with h-maxima markers (depth > h). */
export function splitDistance(r: number, h: number): number {
  return h >= r ? 2 * r : 2 * Math.sqrt(r * r - (r - h) ** 2);
}

export interface Shape { x: number; y: number; a: number; b: number; angle: number }

/** Union of ellipses (discs when a = b) rasterised on a w × h grid (pixel centres at integers). */
export function shapesMask(w: number, h: number, shapes: Shape[]): Uint8Array {
  const m = new Uint8Array(w * h);
  for (const s of shapes) {
    const c = Math.cos(s.angle), sn = Math.sin(s.angle), R = Math.max(s.a, s.b) + 1;
    for (let y = Math.max(0, Math.floor(s.y - R)); y <= Math.min(h - 1, Math.ceil(s.y + R)); y++)
      for (let x = Math.max(0, Math.floor(s.x - R)); x <= Math.min(w - 1, Math.ceil(s.x + R)); x++) {
        const dx = x - s.x, dy = y - s.y, u = dx * c + dy * sn, v = -dx * sn + dy * c;
        if ((u / s.a) ** 2 + (v / s.b) ** 2 <= 1) m[y * w + x] = 1;
      }
  }
  return m;
}

/** Two discs (x1, y1, r1) and (x2, y2, r2). */
export function twoDiscMask(w: number, h: number, x1: number, y1: number, x2: number, y2: number, r1: number, r2: number): Uint8Array {
  return shapesMask(w, h, [{ x: x1, y: y1, a: r1, b: r1, angle: 0 }, { x: x2, y: y2, a: r2, b: r2, angle: 0 }]);
}

// ---------------------------------------------------------------------------------------
// The demo's computation: mask → distance transform → peaks (h-maxima) → watershed of −DT.
import { edt, chamfer } from "../lib/distance";
import { regionalMaxima, hMaxima, fillHoles } from "../lib/morph";
import { label2d, centroids } from "../lib/label";
import { gradient, fromArray, gaussianBlur, Grey } from "../lib/imageops";
import { watershed, countLabels } from "../lib/watershed";
import { rng } from "../lib/random";

export type Metric = "cityblock" | "chessboard" | "3-4" | "euclidean";
export type FloodMode = "dt" | "gradient";

export function distanceOf(mask: ArrayLike<number>, w: number, h: number, metric: Metric): Float32Array {
  return metric === "euclidean" ? edt(mask, w, h) : chamfer(mask, w, h, metric);
}

export interface DistRun {
  dt: Float32Array;
  peaks: Int32Array;          // h-maxima of the DT, labelled
  peakCentres: { x: number; y: number }[];
  labels: Int32Array;
  nObjects: number;
  nComponents: number;        // connected components of the mask (what labelling alone gives)
  ms: number;
}

/**
 * Split a mask: markers = h-maxima of the distance transform (h ≤ 0: all regional maxima),
 * flooded either on −DT or on the gradient magnitude of `image` (σ = 1), inside the mask.
 */
export function splitMask(mask: Uint8Array, w: number, h: number, metric: Metric, hh: number, mode: FloodMode, image?: Grey): DistRun {
  const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
  const dt = distanceOf(mask, w, h, metric);
  const pk = hh > 0 ? hMaxima(dt, w, h, hh, 8) : regionalMaxima(dt, w, h, 8);
  for (let i = 0; i < pk.length; i++) if (!mask[i]) pk[i] = 0;
  const lab = label2d(pk, w, h, 8);
  const land = mode === "dt" ? Float32Array.from(dt, v => -v)
    : Float32Array.from(gradient(image ?? fromArray(w, h, Float32Array.from(mask, v => 60 + 130 * v)), 1).mag.data);
  const labels = watershed(land, w, h, lab.data, { conn: 8, mask });
  const ms = (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0;
  return {
    dt, peaks: lab.data, peakCentres: centroids(lab).slice(1).map(c => ({ x: c.x, y: c.y })),
    labels, nObjects: countLabels(labels), nComponents: label2d(mask, w, h, 8).count, ms,
  };
}

/** Make holes (pepper) in a mask and roughen nothing else; seeded. */
export function withHoles(mask: Uint8Array, w: number, h: number, count: number, seed = 4): Uint8Array {
  const r = rng(seed), out = mask.slice();
  let made = 0;
  for (let t = 0; t < 5000 && made < count; t++) {
    const x = 2 + Math.floor(r.unif() * (w - 4)), y = 2 + Math.floor(r.unif() * (h - 4)), i = y * w + x;
    if (!mask[i] || !mask[i - 2] || !mask[i + 2] || !mask[i - 2 * w] || !mask[i + 2 * w]) continue;
    out[i] = 0; if (r.unif() < 0.5) out[i + 1] = 0; if (r.unif() < 0.4) out[i + w] = 0;
    made++;
  }
  return out;
}

export { fillHoles, gaussianBlur };

/** DT values along the segment from (x0, y0) to (x1, y1), nearest-pixel sampling, step 0.5. */
export function profile(dt: Float32Array, w: number, h: number, x0: number, y0: number, x1: number, y1: number): { t: number; v: number }[] {
  const L = Math.hypot(x1 - x0, y1 - y0), out: { t: number; v: number }[] = [];
  for (let t = 0; t <= L; t += 0.5) {
    const x = Math.round(x0 + (x1 - x0) * t / L), y = Math.round(y0 + (y1 - y0) * t / L);
    out.push({ t, v: x >= 0 && y >= 0 && x < w && y < h ? dt[y * w + x] : 0 });
  }
  return out;
}

/** Peaks and neck of a two-peak profile: the highest value on each side of the midpoint and the lowest in between. */
export function neckOf(p: { t: number; v: number }[]): { p1: number; p2: number; neck: number; t1: number; t2: number; tn: number } {
  const mid = Math.floor(p.length / 2);
  let i1 = 0, i2 = mid;
  for (let i = 0; i < mid; i++) if (p[i].v > p[i1].v) i1 = i;
  for (let i = mid; i < p.length; i++) if (p[i].v > p[i2].v) i2 = i;
  let n = i1;
  for (let i = i1; i <= i2; i++) if (p[i].v < p[n].v) n = i;
  return { p1: p[i1].v, p2: p[i2].v, neck: p[n].v, t1: p[i1].t, t2: p[i2].t, tn: p[n].t };
}
