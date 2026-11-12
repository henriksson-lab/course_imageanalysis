// Seeded synthetic test images with ground truth (no DOM). Same seed → same image, on
// every machine. Every generator returns the image (a Grey, values 0..255 unless stated)
// and what a demo or test needs to score itself: label images, centres, parameters.
//
//  discs · nuclei (touching, elongated) · spots on an uneven background · filaments ·
//  Shepp–Logan phantom (with its ellipse list) · checkerboard, Siemens star, zone plate,
//  ramp · noise (Gaussian, Poisson).
//
// Captions should say "synthetic". Add new generators here rather than in a demo folder.

import { Grey, createGrey, gaussianBlur } from "./imageops";
import { rng, Rng, poisson, uniform } from "./random";

/** An ellipse with semi-axes a ≥ b, rotated by `angle` radians (x right, y down). */
export interface Ellipse { x: number; y: number; a: number; b: number; angle: number }

/** A synthetic image with a ground-truth label image (0 = background, 1..count objects). */
export interface Labelled {
  image: Grey;
  labels: Int32Array;
  count: number;
}

// ---------------------------------------------------------------------------------------
// Noise

/** Add Gaussian noise of standard deviation sd (new image). */
export function addGaussianNoise(img: Grey, sd: number, seed = 1): Grey {
  const r = rng(seed), out = createGrey(img.width, img.height);
  for (let i = 0; i < out.data.length; i++) out.data[i] = img.data[i] + sd * r.norm();
  return out;
}

/**
 * Photon (shot) noise: each pixel becomes Poisson(value · scale) / scale. `scale` is
 * photons per grey level: small scale = few photons = noisy.
 */
export function addPoissonNoise(img: Grey, scale = 1, seed = 1): Grey {
  const r = rng(seed), out = createGrey(img.width, img.height);
  for (let i = 0; i < out.data.length; i++) out.data[i] = poisson(r, Math.max(0, img.data[i]) * scale) / scale;
  return out;
}

// ---------------------------------------------------------------------------------------
// Discs and nuclei

/** Normalised elliptic distance (≤ 1 inside) of (px, py) from ellipse e. */
export function ellipseDistance(e: Ellipse, px: number, py: number): number {
  const c = Math.cos(e.angle), s = Math.sin(e.angle), dx = px - e.x, dy = py - e.y;
  const u = c * dx + s * dy, v = -s * dx + c * dy;
  return Math.sqrt((u / e.a) ** 2 + (v / e.b) ** 2);
}

/**
 * Paint ellipses into a label image: pixel centres inside ellipse k get k+1; where
 * ellipses overlap the pixel goes to the one it is (relatively) deepest inside, so touching
 * objects meet along a sensible line.
 */
export function rasteriseEllipses(w: number, h: number, ellipses: Ellipse[]): Int32Array {
  const lab = new Int32Array(w * h), best = new Float64Array(w * h).fill(1);
  ellipses.forEach((e, k) => {
    const R = Math.ceil(e.a) + 1;
    for (let y = Math.max(0, Math.floor(e.y - R)); y <= Math.min(h - 1, Math.ceil(e.y + R)); y++)
      for (let x = Math.max(0, Math.floor(e.x - R)); x <= Math.min(w - 1, Math.ceil(e.x + R)); x++) {
        const d = ellipseDistance(e, x, y), i = y * w + x;
        if (d <= 1 && (lab[i] === 0 || d < best[i])) { lab[i] = k + 1; best[i] = d; }
      }
  });
  return lab;
}

/** Fraction of pixel (x, y) covered by the ellipse, by s × s supersampling (anti-aliased edges). */
function coverage(e: Ellipse, x: number, y: number, s = 4): number {
  let c = 0;
  for (let j = 0; j < s; j++) for (let i = 0; i < s; i++)
    if (ellipseDistance(e, x - 0.5 + (i + 0.5) / s, y - 0.5 + (j + 0.5) / s) <= 1) c++;
  return c / (s * s);
}

/**
 * Discs (or ellipses) with given grey values on a flat background, anti-aliased, plus the
 * label image. `values[k]` is the grey level of object k (default 200).
 */
export function discs(w: number, h: number, shapes: (Ellipse | { x: number; y: number; r: number })[],
  opts: { background?: number; values?: number[] } = {}): Labelled & { ellipses: Ellipse[] } {
  const ellipses = shapes.map(s => ("r" in s ? { x: s.x, y: s.y, a: s.r, b: s.r, angle: 0 } : s));
  const bg = opts.background ?? 40, img = createGrey(w, h, bg);
  ellipses.forEach((e, k) => {
    const v = opts.values?.[k] ?? 200, R = Math.ceil(e.a) + 1;
    for (let y = Math.max(0, Math.floor(e.y - R)); y <= Math.min(h - 1, Math.ceil(e.y + R)); y++)
      for (let x = Math.max(0, Math.floor(e.x - R)); x <= Math.min(w - 1, Math.ceil(e.x + R)); x++) {
        const c = coverage(e, x, y);
        if (c > 0) img.data[y * w + x] += c * (v - bg);
      }
  });
  const labels = rasteriseEllipses(w, h, ellipses);
  return { image: img, labels, count: ellipses.length, ellipses };
}

export interface NucleiOptions {
  width?: number; height?: number;     // default 256 × 256
  count?: number;                      // number of nuclei (default 20; fewer if they do not fit)
  radius?: [number, number];           // range of the mean radius (default [9, 15])
  elongation?: number;                 // max a/b − 1 (default 0.4)
  touching?: number;                   // fraction placed touching a previous nucleus (default 0.3)
  foreground?: number;                 // default 180
  background?: number;                 // default 50
  gradient?: number;                   // background increase from left to right edge (default 0)
  texture?: number;                    // sd of smooth intensity variation inside nuclei (default 10)
  blur?: number;                       // optical blur σ in pixels (default 1)
  noise?: number;                      // Gaussian noise sd (default 6)
  seed?: number;
}

/**
 * Fluorescent-nucleus-like image: elongated blobs, some touching in pairs or chains, on a
 * (possibly sloped) background, blurred and noisy. Ground truth: labels and ellipses.
 */
export function nuclei(o: NucleiOptions = {}): Labelled & { ellipses: Ellipse[]; touchingPairs: [number, number][] } {
  const w = o.width ?? 256, h = o.height ?? 256, r = rng(o.seed ?? 1);
  const [r0, r1] = o.radius ?? [9, 15], el = o.elongation ?? 0.4, pTouch = o.touching ?? 0.3;
  const ellipses: Ellipse[] = [], touchingPairs: [number, number][] = [];
  const reach = (e: Ellipse) => e.a;
  for (let k = 0, tries = 0; k < (o.count ?? 20) && tries < 4000; tries++) {
    const m = uniform(r, r0, r1), q = 1 + el * r.unif();
    const e: Ellipse = { x: 0, y: 0, a: m * Math.sqrt(q), b: m / Math.sqrt(q), angle: Math.PI * r.unif() };
    let partner = -1;
    if (ellipses.length && r.unif() < pTouch) {
      partner = Math.floor(r.unif() * ellipses.length);
      const p = ellipses[partner], ang = 2 * Math.PI * r.unif(), d = 0.8 * (reach(p) + e.b + (e.a - e.b) * 0.5);
      e.x = p.x + d * Math.cos(ang); e.y = p.y + d * Math.sin(ang);
    } else {
      e.x = uniform(r, e.a + 2, w - e.a - 2); e.y = uniform(r, e.a + 2, h - e.a - 2);
    }
    if (e.x < e.a + 2 || e.x > w - e.a - 2 || e.y < e.a + 2 || e.y > h - e.a - 2) continue;
    let ok = true;
    for (let j = 0; j < ellipses.length && ok; j++) {
      const p = ellipses[j], d = Math.hypot(p.x - e.x, p.y - e.y);
      const minD = j === partner ? 0.65 * (p.b + e.b) : p.a + e.a + 3;
      if (d < minD) ok = false;
    }
    if (!ok) continue;
    if (partner >= 0) touchingPairs.push([partner + 1, ellipses.length + 1]);
    ellipses.push(e);
    k++;
  }
  const labels = rasteriseEllipses(w, h, ellipses);
  const fg = o.foreground ?? 180, bg = o.background ?? 50, grad = o.gradient ?? 0;
  // smooth texture: blurred noise
  const tex = createGrey(w, h);
  const tsd = o.texture ?? 10;
  if (tsd > 0) {
    for (let i = 0; i < tex.data.length; i++) tex.data[i] = r.norm();
    const tb = gaussianBlur(tex, 3);
    const k = tsd / 0.094;   // sd of unit noise blurred with σ = 3 is ≈ 1/(2√π·3) ≈ 0.094
    for (let i = 0; i < tb.data.length; i++) tex.data[i] = k * tb.data[i];
  }
  const img = createGrey(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    img.data[i] = bg + grad * x / Math.max(1, w - 1);
  }
  ellipses.forEach(e => {
    const R = Math.ceil(e.a) + 1;
    for (let y = Math.max(0, Math.floor(e.y - R)); y <= Math.min(h - 1, Math.ceil(e.y + R)); y++)
      for (let x = Math.max(0, Math.floor(e.x - R)); x <= Math.min(w - 1, Math.ceil(e.x + R)); x++) {
        const i = y * w + x, c = coverage(e, x, y);
        if (c > 0) img.data[i] = Math.max(img.data[i], img.data[i] * (1 - c) + c * (fg + tex.data[i]));
      }
  });
  let out = gaussianBlur(img, o.blur ?? 1);
  if ((o.noise ?? 6) > 0) out = addGaussianNoise(out, o.noise ?? 6, (o.seed ?? 1) + 7919);
  return { image: out, labels, count: ellipses.length, ellipses, touchingPairs };
}

// ---------------------------------------------------------------------------------------
// Spots on an uneven background

export interface SpotsOptions {
  width?: number; height?: number;     // default 256 × 256
  count?: number;                      // default 40
  sigma?: number;                      // spot size (Gaussian σ, default 1.5)
  amplitude?: [number, number];        // peak height range above background (default [40, 120])
  background?: number;                 // mean background level (default 40)
  slope?: number;                      // background increase left → right (default 30)
  bump?: number;                       // height of a broad bright patch (uneven illumination, default 40)
  noise?: number;                      // Gaussian noise sd (default 5)
  photons?: number;                    // if set, Poisson noise with this many photons per grey level instead
  minDistance?: number;                // minimum spot spacing (default 4σ)
  seed?: number;
}

/** Diffraction-limited spots (Gaussian profiles) on a smooth uneven background. */
export function spots(o: SpotsOptions = {}): {
  image: Grey; background: Grey; spots: { x: number; y: number; amplitude: number }[]; centres: Uint8Array;
} {
  const w = o.width ?? 256, h = o.height ?? 256, r = rng(o.seed ?? 1), sg = o.sigma ?? 1.5;
  const [a0, a1] = o.amplitude ?? [40, 120], minD = o.minDistance ?? 4 * sg;
  const list: { x: number; y: number; amplitude: number }[] = [];
  for (let tries = 0; list.length < (o.count ?? 40) && tries < 5000; tries++) {
    const x = uniform(r, 3 * sg, w - 3 * sg), y = uniform(r, 3 * sg, h - 3 * sg);
    if (list.some(s => Math.hypot(s.x - x, s.y - y) < minD)) continue;
    list.push({ x, y, amplitude: uniform(r, a0, a1) });
  }
  const bx = uniform(r, 0.25, 0.75) * w, by = uniform(r, 0.25, 0.75) * h, bs = 0.3 * Math.min(w, h);
  const bgLevel = o.background ?? 40, slope = o.slope ?? 30, bump = o.bump ?? 40;
  const background = createGrey(w, h), img = createGrey(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    background.data[y * w + x] = bgLevel + slope * x / Math.max(1, w - 1) + bump * Math.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * bs * bs));
  img.data.set(background.data);
  const R = Math.ceil(4 * sg), centres = new Uint8Array(w * h);
  for (const s of list) {
    centres[Math.round(s.y) * w + Math.round(s.x)] = 1;
    for (let y = Math.max(0, Math.floor(s.y) - R); y <= Math.min(h - 1, Math.floor(s.y) + R); y++)
      for (let x = Math.max(0, Math.floor(s.x) - R); x <= Math.min(w - 1, Math.floor(s.x) + R); x++)
        img.data[y * w + x] += s.amplitude * Math.exp(-((x - s.x) ** 2 + (y - s.y) ** 2) / (2 * sg * sg));
  }
  const seed2 = (o.seed ?? 1) + 104729;
  const out = o.photons ? addPoissonNoise(img, o.photons, seed2) : (o.noise ?? 5) > 0 ? addGaussianNoise(img, o.noise ?? 5, seed2) : img;
  return { image: out, background, spots: list, centres };
}

// ---------------------------------------------------------------------------------------
// Filaments

/** A quadratic Bézier curve from p0 to p2 with control point p1. */
export interface Curve { x0: number; y0: number; x1: number; y1: number; x2: number; y2: number }

export interface FilamentOptions {
  width?: number; height?: number;     // default 256 × 256
  count?: number;                      // default 12
  length?: [number, number];           // default [60, 200]
  curvature?: number;                  // 0 = straight lines; control-point offset as a fraction of the length (default 0.25)
  thickness?: number;                  // Gaussian σ of the cross profile (default 1.2)
  amplitude?: [number, number];        // default [80, 160]
  background?: number;                 // default 30
  noise?: number;                      // default 8
  seed?: number;
}

/** Points along a curve, about one per pixel. */
export function curvePoints(c: Curve, step = 0.5): [number, number][] {
  const L = Math.hypot(c.x1 - c.x0, c.y1 - c.y0) + Math.hypot(c.x2 - c.x1, c.y2 - c.y1);
  const n = Math.max(2, Math.ceil(L / step)), out: [number, number][] = [];
  for (let k = 0; k <= n; k++) {
    const t = k / n, u = 1 - t;
    out.push([u * u * c.x0 + 2 * u * t * c.x1 + t * t * c.x2, u * u * c.y0 + 2 * u * t * c.y1 + t * t * c.y2]);
  }
  return out;
}

/**
 * Thin bright filaments (microtubule- or vessel-like), straight when curvature = 0.
 * Ground truth: the curves, a one-pixel centreline mask and a mask of pixels within 1.5σ.
 */
export function filaments(o: FilamentOptions = {}): { image: Grey; curves: Curve[]; centreline: Uint8Array; mask: Uint8Array } {
  const w = o.width ?? 256, h = o.height ?? 256, r = rng(o.seed ?? 1);
  const [l0, l1] = o.length ?? [60, 200], curv = o.curvature ?? 0.25, sg = o.thickness ?? 1.2;
  const [a0, a1] = o.amplitude ?? [80, 160];
  const curves: Curve[] = [], amps: number[] = [];
  for (let k = 0, tries = 0; k < (o.count ?? 12) && tries < 1000; tries++) {
    const L = uniform(r, l0, l1), ang = Math.PI * r.unif();
    const cx = uniform(r, 0.1, 0.9) * w, cy = uniform(r, 0.1, 0.9) * h;
    const dx = Math.cos(ang) * L / 2, dy = Math.sin(ang) * L / 2, off = curv * L * (2 * r.unif() - 1);
    const c: Curve = { x0: cx - dx, y0: cy - dy, x1: cx - Math.sin(ang) * off, y1: cy + Math.cos(ang) * off, x2: cx + dx, y2: cy + dy };
    if ([c.x0, c.x2].some(v => v < 2 || v > w - 3) || [c.y0, c.y2].some(v => v < 2 || v > h - 3)) continue;
    curves.push(c); amps.push(uniform(r, a0, a1)); k++;
  }
  const bg = o.background ?? 30, img = createGrey(w, h, bg);
  const best = new Float64Array(w * h).fill(Infinity), centreline = new Uint8Array(w * h);
  const val = new Float32Array(w * h);
  const R = Math.ceil(4 * sg) + 1;
  curves.forEach((c, k) => {
    const pts = curvePoints(c), d2 = new Float64Array(w * h).fill(Infinity);
    for (const [px, py] of pts) {
      const ix = Math.round(px), iy = Math.round(py);
      if (ix >= 0 && ix < w && iy >= 0 && iy < h) centreline[iy * w + ix] = 1;
      for (let y = Math.max(0, iy - R); y <= Math.min(h - 1, iy + R); y++)
        for (let x = Math.max(0, ix - R); x <= Math.min(w - 1, ix + R); x++) {
          const d = (x - px) ** 2 + (y - py) ** 2, i = y * w + x;
          if (d < d2[i]) d2[i] = d;
        }
    }
    for (let i = 0; i < w * h; i++) if (d2[i] < Infinity) {
      val[i] = Math.max(val[i], amps[k] * Math.exp(-d2[i] / (2 * sg * sg)));
      if (d2[i] < best[i]) best[i] = d2[i];
    }
  });
  const mask = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) { img.data[i] += val[i]; mask[i] = best[i] <= (1.5 * sg) ** 2 ? 1 : 0; }
  const out = (o.noise ?? 8) > 0 ? addGaussianNoise(img, o.noise ?? 8, (o.seed ?? 1) + 31337) : img;
  return { image: out, curves, centreline, mask };
}

// ---------------------------------------------------------------------------------------
// Shepp–Logan phantom

/** One phantom ellipse in normalised coordinates: x right, y UP, both in [−1, 1]; angle in degrees, counter-clockwise. */
export interface PhantomEllipse { value: number; a: number; b: number; x: number; y: number; angle: number }

/** The modified Shepp–Logan phantom (Toft's higher-contrast values), as used for tomography demos. */
export const SHEPP_LOGAN: PhantomEllipse[] = [
  { value: 1, a: 0.69, b: 0.92, x: 0, y: 0, angle: 0 },
  { value: -0.8, a: 0.6624, b: 0.874, x: 0, y: -0.0184, angle: 0 },
  { value: -0.2, a: 0.11, b: 0.31, x: 0.22, y: 0, angle: -18 },
  { value: -0.2, a: 0.16, b: 0.41, x: -0.22, y: 0, angle: 18 },
  { value: 0.1, a: 0.21, b: 0.25, x: 0, y: 0.35, angle: 0 },
  { value: 0.1, a: 0.046, b: 0.046, x: 0, y: 0.1, angle: 0 },
  { value: 0.1, a: 0.046, b: 0.046, x: 0, y: -0.1, angle: 0 },
  { value: 0.1, a: 0.046, b: 0.023, x: -0.08, y: -0.605, angle: 0 },
  { value: 0.1, a: 0.023, b: 0.023, x: 0, y: -0.606, angle: 0 },
  { value: 0.1, a: 0.023, b: 0.046, x: 0.06, y: -0.605, angle: 0 },
];

/** Value of a phantom at normalised (x, y) (y up): sum of the ellipses containing the point. */
export function phantomValue(el: PhantomEllipse[], x: number, y: number): number {
  let v = 0;
  for (const e of el) {
    const t = e.angle * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
    const u = c * (x - e.x) + s * (y - e.y), w = -s * (x - e.x) + c * (y - e.y);
    if ((u / e.a) ** 2 + (w / e.b) ** 2 <= 1) v += e.value;
  }
  return v;
}

/**
 * n × n Shepp–Logan phantom with values 0..1 (modified version), pixel centres mapped to
 * [−1, 1]² with y up. `supersample` (default 2) anti-aliases the edges.
 */
export function sheppLogan(n = 256, el: PhantomEllipse[] = SHEPP_LOGAN, supersample = 2): Grey {
  const img = createGrey(n, n), s = supersample;
  for (let row = 0; row < n; row++) for (let col = 0; col < n; col++) {
    let v = 0;
    for (let j = 0; j < s; j++) for (let i = 0; i < s; i++) {
      const x = -1 + 2 * (col + (i + 0.5) / s) / n, y = 1 - 2 * (row + (j + 0.5) / s) / n;
      v += phantomValue(el, x, y);
    }
    img.data[row * n + col] = v / (s * s);
  }
  return img;
}

// ---------------------------------------------------------------------------------------
// Test patterns

/** Checkerboard of `square`-pixel squares, top-left square `lo`. */
export function checkerboard(w: number, h: number, square = 16, lo = 0, hi = 255): Grey {
  const img = createGrey(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    img.data[y * w + x] = (Math.floor(x / square) + Math.floor(y / square)) % 2 ? hi : lo;
  return img;
}

/** Siemens star with `spokes` black/white pairs (resolution and aliasing test), anti-aliased. */
export function siemensStar(n = 256, spokes = 18, lo = 0, hi = 255): Grey {
  const img = createGrey(n, n), c = (n - 1) / 2, R = n / 2 - 1, s = 3;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    let v = 0;
    for (let j = 0; j < s; j++) for (let i = 0; i < s; i++) {
      const dx = x - 0.5 + (i + 0.5) / s - c, dy = y - 0.5 + (j + 0.5) / s - c;
      const r = Math.hypot(dx, dy), on = Math.sin(spokes * Math.atan2(dy, dx)) >= 0;
      v += r > R ? 0.5 : on ? 1 : 0;
    }
    img.data[y * n + x] = lo + (hi - lo) * v / (s * s);
  }
  return img;
}

/** Zone plate cos(k r²): frequency increasing linearly with radius, reaching Nyquist at the edge. */
export function zonePlate(n = 256, lo = 0, hi = 255): Grey {
  const img = createGrey(n, n), c = n / 2, k = Math.PI / n;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const r2 = (x - c) ** 2 + (y - c) ** 2;
    img.data[y * n + x] = lo + (hi - lo) * 0.5 * (1 + Math.cos(k * r2));
  }
  return img;
}

/** Horizontal ramp from lo (left) to hi (right). */
export function ramp(w: number, h: number, lo = 0, hi = 255): Grey {
  const img = createGrey(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) img.data[y * w + x] = lo + (hi - lo) * x / Math.max(1, w - 1);
  return img;
}

/** A straight step edge through the centre at `angle` radians (normal direction), optionally blurred. */
export function stepEdge(w: number, h: number, angle = 0, lo = 50, hi = 200, blur = 0): Grey {
  const img = createGrey(w, h), c = Math.cos(angle), s = Math.sin(angle), cx = (w - 1) / 2, cy = (h - 1) / 2;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) img.data[y * w + x] = c * (x - cx) + s * (y - cy) >= 0 ? hi : lo;
  return blur > 0 ? gaussianBlur(img, blur) : img;
}

/** Re-exported for generators written elsewhere. */
export type { Rng };
