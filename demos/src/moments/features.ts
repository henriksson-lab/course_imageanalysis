// Shape features beyond moments (labelling day, unit shape-features/; the measurements day's
// lib/features.ts should take these over). No DOM.
//
// Units: pixels of the mask; pixel (x, y) covers [x, x+1] × [y, y+1] for the hull (corner
// convention) and has its centre at (x + 0.5, y + 0.5).
import { Pt, maskHull, shoelaceArea, polygonPerimeter, rotatingCalipers, Calipers, minEnclosingCircle, Circle } from "../lib/convexhull";
import { edt } from "../lib/distance";
import { dft1d } from "../lib/fft";
import { maskProps, Region, traceContour } from "./regionprops";

export interface ShapeFeatures {
  region: Region;
  area: number;
  /** Equivalent diameter √(4A/π). */
  eqDiameter: number;
  hull: Pt[];
  hullArea: number;
  hullPerimeter: number;
  solidity: number;
  extent: number;
  /** Form factor 4πA/P² for each perimeter estimator. */
  ffEdges: number; ffChain: number; ffCrofton: number;
  /** Second-moment compactness 2π·mean(r²)/A (1 for a disc, larger otherwise). */
  compactMoments: number;
  calipers: Calipers;
  /** Max inscribed radius: largest distance to the nearest background pixel centre, minus ½
   *  (so a pixel-perfect disc of radius r gives about r), its pixel, and the mean over the object. */
  inscribedR: number; inscribedAt: Pt; meanR: number;
  enclosing: Circle;
}

export function shapeFeatures(full: ArrayLike<number>, W: number, H: number, hullMode: "corners" | "centres" = "corners"): ShapeFeatures | null {
  // work on the bounding box with a 1-pixel background frame, then shift back
  let bx0 = W, by0 = H, bx1 = -1, by1 = -1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (full[y * W + x]) { if (x < bx0) bx0 = x; if (x > bx1) bx1 = x; if (y < by0) by0 = y; if (y > by1) by1 = y; }
  if (bx1 < 0) return null;
  const ox = bx0 - 1, oy = by0 - 1, w = bx1 - bx0 + 3, h = by1 - by0 + 3;
  const mask = new Uint8Array(w * h);
  for (let y = by0; y <= by1; y++) for (let x = bx0; x <= bx1; x++) if (full[y * W + x]) mask[(y - oy) * w + x - ox] = 1;
  const f = shapeFeaturesLocal(mask, w, h, hullMode);
  if (!f) return null;
  const sh = (p: Pt): Pt => [p[0] + ox, p[1] + oy];
  const r = f.region;
  Object.assign(r, { cx: r.cx + ox, cy: r.cy + oy, x0: r.x0 + ox, x1: r.x1 + ox, y0: r.y0 + oy, y1: r.y1 + oy, touchesBorder: bx0 === 0 || by0 === 0 || bx1 === W - 1 || by1 === H - 1 });
  f.hull = f.hull.map(sh);
  f.calipers.maxPair = [sh(f.calipers.maxPair[0]), sh(f.calipers.maxPair[1])];
  f.inscribedAt = sh(f.inscribedAt);
  f.enclosing = { x: f.enclosing.x + ox, y: f.enclosing.y + oy, r: f.enclosing.r };
  return f;
}

function shapeFeaturesLocal(mask: ArrayLike<number>, w: number, h: number, hullMode: "corners" | "centres"): ShapeFeatures | null {
  const region = maskProps(mask, w, h);
  if (!region) return null;
  const A = region.area;
  const hull = maskHull(mask, w, h, hullMode);
  const hullArea = Math.abs(shoelaceArea(hull));
  const cal = rotatingCalipers(hull);
  const d = edt(mask, w, h);
  let best = -1, bi = 0, sum = 0;
  for (let i = 0; i < w * h; i++) if (mask[i]) { sum += d[i]; if (d[i] > best) { best = d[i]; bi = i; } }
  const ff = (P: number) => (P > 0 ? 4 * Math.PI * A / (P * P) : 0);
  const cornerHull = hullMode === "corners" ? hull : maskHull(mask, w, h, "corners");
  return {
    region, area: A, eqDiameter: Math.sqrt(4 * A / Math.PI),
    hull, hullArea, hullPerimeter: polygonPerimeter(hull),
    solidity: hullArea > 0 ? A / hullArea : 1,
    extent: region.extent,
    ffEdges: ff(region.perimEdges), ffChain: ff(region.perimChain), ffCrofton: ff(region.perimCrofton),
    compactMoments: 2 * Math.PI * (region.s11 + region.s22) / A,
    calipers: cal,
    inscribedR: Math.max(0, best - 0.5), inscribedAt: [bi % w + 0.5, Math.floor(bi / w) + 0.5], meanR: sum / A - 0.5,
    enclosing: minEnclosingCircle(cornerHull, 7),
  };
}

/** Outer contour of the largest-first component as points (pixel centres + ½), resampled
 *  by arc length to n points. */
export function outline(mask: ArrayLike<number>, w: number, h: number, n = 128): Pt[] {
  const s = Array.prototype.findIndex.call(mask, (v: number) => !!v) as number;
  if (s < 0) return [];
  const c = traceContour(mask, w, h, s).points.map(([x, y]) => [x + 0.5, y + 0.5] as Pt);
  if (c.length < 3) return c;
  const cum = [0];
  for (let i = 1; i <= c.length; i++) { const a = c[i - 1], b = c[i % c.length]; cum.push(cum[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1])); }
  const L = cum[c.length], out: Pt[] = [];
  let k = 0;
  for (let i = 0; i < n; i++) {
    const t = L * i / n;
    while (cum[k + 1] < t) k++;
    const a = c[k], b = c[(k + 1) % c.length], f = (t - cum[k]) / Math.max(1e-12, cum[k + 1] - cum[k]);
    out.push([a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])]);
  }
  return out;
}

/** Fourier descriptors of a closed outline z(t) = x + i y: keep the coefficients with
 *  frequency |f| ≤ k (and the mean), reconstruct. Returns the smoothed outline and the
 *  magnitudes |c_f| / |c_1| for f = 1..kmax (translation- and scale-invariant). */
export function fourierOutline(pts: Pt[], k: number): { smooth: Pt[]; mags: number[] } {
  const n = pts.length;
  if (n < 3) return { smooth: pts, mags: [] };
  const re = Float64Array.from(pts, p => p[0]), im = Float64Array.from(pts, p => p[1]);
  const F = dft1d(re, im, false);
  const fr = F.re, fi = F.im;
  const kr = new Float64Array(n), ki = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const f = j <= n / 2 ? j : j - n;
    if (Math.abs(f) <= k) { kr[j] = fr[j]; ki[j] = fi[j]; }
  }
  const B = dft1d(kr, ki, true);
  const smooth: Pt[] = Array.from({ length: n }, (_, j) => [B.re[j] / n, B.im[j] / n] as Pt);
  // the boundary is traced clockwise on screen: the dominant coefficient is f = 1 or f = −1
  const m1 = Math.max(Math.hypot(fr[1], fi[1]), Math.hypot(fr[n - 1], fi[n - 1])) || 1;
  const mags: number[] = [];
  for (let f = 1; f <= Math.min(12, Math.floor(n / 2) - 1); f++) mags.push((Math.hypot(fr[f], fi[f]) + Math.hypot(fr[n - f], fi[n - f])) / m1);
  return { smooth, mags };
}
