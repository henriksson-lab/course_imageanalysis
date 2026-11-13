// Flow-field decoding of the Cellpose family, small and exact (no DOM, used by the flowtrack
// demo and demos/test/cellpose-test.ts):
//
//  - the training target: heat put at a source inside each cell and averaged over the 3 x 3
//    neighbours that belong to the same cell ("diffusion inside the cell only"); the arrows are
//    the normalised central differences of the heat (neighbours outside the cell count as 0);
//  - following the arrows: Euler steps x <- x + eta v(x) with bilinear interpolation, from
//    every foreground pixel;
//  - sinks: a histogram of the rounded end points, its local maxima (5 x 5) with more than
//    `minSeed` particles are seeds, each grown 5 times by 3 x 3 inside histogram > 2; a pixel
//    gets the label of the seed region its end point lies in (as Cellpose's get_masks);
//  - the flow check: the arrows recomputed from each mask, mean squared difference to the
//    predicted arrows; masks above the threshold are dropped.
// Plus the integer 5 x 5 field of the exercise (B2) and its path following.
// Shared code for lib/instances.ts (plan/cellnets-day.md, "Shared engineering"): the heat
// target and flow following here are candidates to move there together with instancemaps'.

export type Lab = Int32Array;

/** Source pixel of label l: the mask pixel nearest to the median of its x and y coordinates. */
export function sourcePixel(lab: ArrayLike<number>, w: number, h: number, l: number): number {
  const xs: number[] = [], ys: number[] = [];
  for (let p = 0; p < w * h; p++) if (lab[p] === l) { xs.push(p % w); ys.push((p - (p % w)) / w); }
  if (!xs.length) return -1;
  const med = (a: number[]) => { const s = a.slice().sort((u, v) => u - v), n = s.length; return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]); };
  const mx = med(xs), my = med(ys);
  let best = -1, bd = Infinity;
  for (let k = 0; k < xs.length; k++) {
    const d = (xs[k] - mx) ** 2 + (ys[k] - my) ** 2;
    if (d < bd) { bd = d; best = ys[k] * w + xs[k]; }
  }
  return best;
}

/** Default number of averaging steps for label l: 2 (width + height) of its bounding box. */
export function autoIters(lab: ArrayLike<number>, w: number, h: number, l: number): number {
  let x0 = w, x1 = -1, y0 = h, y1 = -1;
  for (let p = 0; p < w * h; p++) if (lab[p] === l) { const x = p % w, y = (p - x) / w; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  return x1 < 0 ? 0 : 2 * ((x1 - x0 + 1) + (y1 - y0 + 1));
}

/** Heat of label l after `iters` steps of "add 1 at the source, then average over the 3 x 3
 *  neighbours inside the cell" (0 outside the cell). */
export function heatTarget(lab: ArrayLike<number>, w: number, h: number, l: number, iters: number, src = sourcePixel(lab, w, h, l)): Float64Array {
  const P: number[] = [];
  for (let p = 0; p < w * h; p++) if (lab[p] === l) P.push(p);
  let a = new Float64Array(w * h), b = new Float64Array(w * h);
  if (src < 0) return a;
  for (let it = 0; it < iters; it++) {
    a[src] += 1;
    for (const p of P) {
      const x = p % w, y = (p - x) / w;
      let s = 0, c = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const q = yy * w + xx;
        if (lab[q] !== l) continue;
        s += a[q]; c++;
      }
      b[p] = s / c;
    }
    [a, b] = [b, a];
  }
  return a;
}

/** Arrows of label l from its heat: central differences (neighbours outside the cell count
 *  as 0), normalised to length 1; 0 where the difference is 0. Written into vx, vy. */
export function flowFromHeat(lab: ArrayLike<number>, w: number, h: number, l: number, heat: Float64Array, vx: Float32Array, vy: Float32Array): void {
  const H = (x: number, y: number) => (x < 0 || y < 0 || x >= w || y >= h || lab[y * w + x] !== l ? 0 : heat[y * w + x]);
  for (let p = 0; p < w * h; p++) {
    if (lab[p] !== l) continue;
    const x = p % w, y = (p - x) / w;
    const gx = H(x + 1, y) - H(x - 1, y), gy = H(x, y + 1) - H(x, y - 1);
    const m = Math.hypot(gx, gy);
    vx[p] = m > 0 ? gx / m : 0; vy[p] = m > 0 ? gy / m : 0;
  }
}

export interface FlowTarget { vx: Float32Array; vy: Float32Array; heat: Float64Array; sources: number[]; iters: number[] }

/** Arrows of every label of a label image. `iters` = null: autoIters per label. The heat
 *  image holds each cell's heat divided by its maximum (for display). */
export function flowTargets(lab: ArrayLike<number>, w: number, h: number, iters: number | null = null): FlowTarget {
  let n = 0; for (let p = 0; p < w * h; p++) if (lab[p] > n) n = lab[p];
  const vx = new Float32Array(w * h), vy = new Float32Array(w * h), heat = new Float64Array(w * h);
  const sources: number[] = [], its: number[] = [];
  for (let l = 1; l <= n; l++) {
    const src = sourcePixel(lab, w, h, l);
    sources.push(src);
    if (src < 0) { its.push(0); continue; }
    const k = iters ?? autoIters(lab, w, h, l);
    its.push(k);
    const ht = heatTarget(lab, w, h, l, k, src);
    flowFromHeat(lab, w, h, l, ht, vx, vy);
    let mx = 0; for (let p = 0; p < w * h; p++) if (lab[p] === l && ht[p] > mx) mx = ht[p];
    for (let p = 0; p < w * h; p++) if (lab[p] === l) heat[p] = mx > 0 ? ht[p] / mx : 0;
  }
  return { vx, vy, heat, sources, iters: its };
}

/** Bilinear interpolation of a w x h image at (x, y) (pixel centres at integers, clamped). */
export function bilinear(a: ArrayLike<number>, w: number, h: number, x: number, y: number): number {
  x = Math.max(0, Math.min(w - 1, x)); y = Math.max(0, Math.min(h - 1, y));
  const x0 = Math.min(w - 2, Math.floor(x)), y0 = Math.min(h - 2, Math.floor(y));
  const fx = x - x0, fy = y - y0, p = y0 * w + x0;
  return (1 - fy) * ((1 - fx) * a[p] + fx * a[p + 1]) + fy * ((1 - fx) * a[p + w] + fx * a[p + w + 1]);
}

/** Euler path from (x, y): `steps` steps of length eta |v|; returns 2 (steps + 1) numbers x, y. */
export function tracePath(vx: Float32Array, vy: Float32Array, w: number, h: number, x: number, y: number, steps: number, eta = 1): Float32Array {
  const out = new Float32Array(2 * (steps + 1));
  out[0] = x; out[1] = y;
  for (let j = 1; j <= steps; j++) {
    const u = bilinear(vx, w, h, x, y), v = bilinear(vy, w, h, x, y);
    x = Math.max(0, Math.min(w - 1, x + eta * u)); y = Math.max(0, Math.min(h - 1, y + eta * v));
    out[2 * j] = x; out[2 * j + 1] = y;
  }
  return out;
}

/** End points after `steps` Euler steps from every foreground pixel (NaN elsewhere). Also
 *  keeps the positions after every step when `keepAll` (for the animation). */
export function followFlows(fg: ArrayLike<number>, vx: Float32Array, vy: Float32Array, w: number, h: number, steps: number, eta = 1,
  keepAll = false): { ex: Float32Array; ey: Float32Array; frames?: Float32Array[] } {
  const ex = new Float32Array(w * h).fill(NaN), ey = new Float32Array(w * h).fill(NaN);
  const idx: number[] = [];
  for (let p = 0; p < w * h; p++) if (fg[p]) { idx.push(p); ex[p] = p % w; ey[p] = (p - (p % w)) / w; }
  const frames: Float32Array[] = [];
  const snap = () => { const f = new Float32Array(2 * idx.length); idx.forEach((p, k) => { f[2 * k] = ex[p]; f[2 * k + 1] = ey[p]; }); frames.push(f); };
  if (keepAll) snap();
  for (let j = 0; j < steps; j++) {
    for (const p of idx) {
      const x = ex[p], y = ey[p];
      const u = bilinear(vx, w, h, x, y), v = bilinear(vy, w, h, x, y);
      ex[p] = Math.max(0, Math.min(w - 1, x + eta * u)); ey[p] = Math.max(0, Math.min(h - 1, y + eta * v));
    }
    if (keepAll) snap();
  }
  return keepAll ? { ex, ey, frames } : { ex, ey };
}

/** Histogram of rounded end points. */
export function endHistogram(ex: Float32Array, ey: Float32Array, w: number, h: number): Int32Array {
  const hist = new Int32Array(w * h);
  for (let p = 0; p < w * h; p++) {
    if (Number.isNaN(ex[p])) continue;
    const x = Math.round(ex[p]), y = Math.round(ey[p]);
    hist[y * w + x]++;
  }
  return hist;
}

/** Seeds = local maxima of the histogram in a 5 x 5 window with more than minSeed particles;
 *  each grown 5 times by 3 x 3 inside histogram > 2. Returns the seed pixels (most particles
 *  first) and the seed-region label image. */
export function findSinks(hist: Int32Array, w: number, h: number, minSeed = 10): { seeds: number[]; region: Int32Array } {
  const seeds: number[] = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x, v = hist[p];
    if (v <= minSeed) continue;
    let mx = 0;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < w && yy < h) mx = Math.max(mx, hist[yy * w + xx]);
    }
    if (v >= mx) seeds.push(p);
  }
  seeds.sort((a, b) => hist[b] - hist[a] || a - b);
  const region = new Int32Array(w * h);
  seeds.forEach((s, k) => {
    let pix = new Set<number>([s]);
    for (let it = 0; it < 5; it++) {
      const next = new Set<number>();
      for (const p of pix) {
        const x = p % w, y = (p - x) / w;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          const q = yy * w + xx;
          if (hist[q] > 2) next.add(q);
        }
      }
      pix = next;
    }
    for (const p of pix) region[p] = k + 1;
  });
  return { seeds, region };
}

/** Relabel 1..n in order of first appearance, dropping labels with fewer than minSize pixels. */
export function relabel(lab: Int32Array, minSize = 0): { labels: Int32Array; count: number; small: number } {
  let n = 0; for (const v of lab) if (v > n) n = v;
  const size = new Int32Array(n + 1);
  for (const v of lab) size[v]++;
  const map = new Int32Array(n + 1);
  let k = 0, small = 0;
  const out = new Int32Array(lab.length);
  for (let p = 0; p < lab.length; p++) {
    const v = lab[p];
    if (!v) continue;
    if (!map[v]) { if (size[v] < minSize) { map[v] = -1; small++; } else map[v] = ++k; }
    if (map[v] > 0) out[p] = map[v];
  }
  return { labels: out, count: k, small };
}

/** Flow error of every mask: mean over its pixels of |v_pred - v_mask|^2, where v_mask are
 *  the arrows recomputed from the mask alone (same heat steps). */
export function flowErrors(masks: Int32Array, w: number, h: number, vx: Float32Array, vy: Float32Array, iters: number | null = null): number[] {
  const T = flowTargets(masks, w, h, iters);
  let n = 0; for (const v of masks) if (v > n) n = v;
  const s = new Float64Array(n + 1), c = new Float64Array(n + 1);
  for (let p = 0; p < w * h; p++) {
    const l = masks[p];
    if (!l) continue;
    s[l] += (vx[p] - T.vx[p]) ** 2 + (vy[p] - T.vy[p]) ** 2; c[l]++;
  }
  return Array.from({ length: n }, (_, i) => (c[i + 1] ? s[i + 1] / c[i + 1] : 0));
}

export interface DecodeOptions { steps: number; eta?: number; minSeed?: number; minSize?: number; flowThreshold?: number; iters?: number | null }
export interface Decoded {
  ex: Float32Array; ey: Float32Array; hist: Int32Array; seeds: number[]; region: Int32Array;
  /** masks before the flow check (after the size filter) */
  raw: Int32Array; nRaw: number; small: number;
  errors: number[]; dropped: boolean[];
  /** final masks */
  masks: Int32Array; count: number;
}

/** The whole decoding: follow, histogram, sinks, masks, size filter, flow check. */
export function decode(fg: ArrayLike<number>, vx: Float32Array, vy: Float32Array, w: number, h: number, o: DecodeOptions): Decoded {
  const { ex, ey } = followFlows(fg, vx, vy, w, h, o.steps, o.eta ?? 1);
  const hist = endHistogram(ex, ey, w, h);
  const { seeds, region } = findSinks(hist, w, h, o.minSeed ?? 10);
  const lab0 = new Int32Array(w * h);
  for (let p = 0; p < w * h; p++) if (!Number.isNaN(ex[p])) lab0[p] = region[Math.round(ey[p]) * w + Math.round(ex[p])];
  const r = relabel(lab0, o.minSize ?? 10);
  const errors = flowErrors(r.labels, w, h, vx, vy, o.iters ?? null);
  const thr = o.flowThreshold ?? 0.4;
  const dropped = errors.map((e) => thr > 0 && e > thr);
  const keep = new Int32Array(w * h);
  for (let p = 0; p < w * h; p++) { const l = r.labels[p]; if (l && !dropped[l - 1]) keep[p] = l; }
  const f = relabel(keep);
  return { ex, ey, hist, seeds, region, raw: r.labels, nRaw: r.count, small: r.small, errors, dropped, masks: f.labels, count: f.count };
}

// ---- the integer grid of exercise B2 ------------------------------------------------------

/** Directions as [dRow, dCol]; index 0 = no movement. */
export const DIRS: [number, number][] = [[0, 0], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]];
export const GLYPH = ["·", "→", "↘", "↓", "↙", "←", "↖", "↑", "↗"];

/** The 5 x 5 field of exercise B2 (rows top to bottom). */
export const B2_FIELD: number[] = [
  "→←↓←←",
  "→↘↓↙←",
  "→→·←←",
  "↗↑↑↘↓",
  "↗↑↑→·",
].flatMap((r) => Array.from(r).map((c) => GLYPH.indexOf(c)));

/** Follow an integer field from `start` until a cell repeats; returns the visited cells and
 *  the cycle (one cell = a sink, two or more = an oscillation). */
export function gridFollow(field: ArrayLike<number>, rows: number, cols: number, start: number): { path: number[]; cycle: number[] } {
  const path = [start];
  const seen = new Map<number, number>([[start, 0]]);
  let p = start;
  for (;;) {
    const [dr, dc] = DIRS[field[p]];
    const r = Math.max(0, Math.min(rows - 1, Math.floor(p / cols) + dr)), c = Math.max(0, Math.min(cols - 1, (p % cols) + dc));
    const q = r * cols + c;
    if (seen.has(q)) return { path, cycle: path.slice(seen.get(q)!) };
    seen.set(q, path.length); path.push(q); p = q;
  }
}

/** Group every cell by the cycle its path ends in. Key = sorted cycle cells joined by ",". */
export function gridGroups(field: ArrayLike<number>, rows: number, cols: number): Map<string, number[]> {
  const g = new Map<string, number[]>();
  for (let p = 0; p < rows * cols; p++) {
    const k = gridFollow(field, rows, cols, p).cycle.slice().sort((a, b) => a - b).join(",");
    if (!g.has(k)) g.set(k, []);
    g.get(k)!.push(p);
  }
  return g;
}

// ---- synthetic scenes (48 x 48) ---------------------------------------------------------------

export const SIZE = 48;
export type ShapeId = "disc" | "C" | "rod" | "touching" | "dumbbell";
export const SHAPES: { id: ShapeId; label: string }[] = [
  { id: "disc", label: "disc" }, { id: "C", label: "C-shape" }, { id: "rod", label: "bent rod" },
  { id: "touching", label: "two touching cells" }, { id: "dumbbell", label: "dumbbell" },
];

/** Label image of a scene (1 or 2 cells). */
export function makeShape(id: ShapeId, n = SIZE): Int32Array {
  const lab = new Int32Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const p = y * n + x, X = x - 23.5, Y = y - 23.5, r = Math.hypot(X, Y);
    if (id === "disc") { if (r <= 14) lab[p] = 1; }
    else if (id === "C") {
      const a = Math.abs(Math.atan2(Y, X)) * 180 / Math.PI;
      if (r <= 18 && r >= 9 && a > 55) lab[p] = 1;
    } else if (id === "rod") {
      // a thick arc: centreline on a circle of radius 17 around (23.5, 40), from 200° to 340°
      const cy = 41, R = 19, ang = Math.atan2(y - cy, X) * 180 / Math.PI; // -180..180, upper half negative
      const d = Math.abs(Math.hypot(X, y - cy) - R);
      if (d <= 4.5 && ang <= -15 && ang >= -165) lab[p] = 1;
      // round caps
      for (const t of [-15, -165]) {
        const ex = 23.5 + R * Math.cos(t * Math.PI / 180), ey = cy + R * Math.sin(t * Math.PI / 180);
        if (Math.hypot(x - ex, y - ey) <= 4.5) lab[p] = 1;
      }
    } else if (id === "touching") {
      const d1 = Math.hypot(x - 13, y - 23.5), d2 = Math.hypot(x - 34, y - 23.5);
      if (d1 <= 12 || d2 <= 12) lab[p] = d1 <= d2 ? 1 : 2;
    } else if (id === "dumbbell") {
      const d1 = Math.hypot(x - 12, y - 23.5), d2 = Math.hypot(x - 35, y - 23.5);
      if (d1 <= 10 || d2 <= 10 || (Math.abs(Y) <= 2.5 && x > 12 && x < 35)) lab[p] = 1;
    }
  }
  return lab;
}

/** Labels that touch (8-neighbours) are merged into one: the "one shared field" error. */
export function mergeTouching(lab: Int32Array, w: number, h: number): Int32Array {
  let n = 0; for (const v of lab) if (v > n) n = v;
  const par = Array.from({ length: n + 1 }, (_, i) => i);
  const find = (a: number): number => { while (par[a] !== a) { par[a] = par[par[a]]; a = par[a]; } return a; };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const a = lab[y * w + x];
    if (!a) continue;
    for (const [dx, dy] of [[1, 0], [0, 1], [1, 1], [-1, 1]]) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      const b = lab[yy * w + xx];
      if (b && b !== a) par[find(a)] = find(b);
    }
  }
  const out = new Int32Array(lab.length);
  for (let p = 0; p < lab.length; p++) if (lab[p]) out[p] = find(lab[p]);
  return relabel(out).labels;
}

/** Seeded smooth random field with unit standard deviation (white noise, Gaussian blur σ). */
export function smoothNoise(w: number, h: number, seed: number, sigma = 1.5): Float32Array {
  let a = seed >>> 0;
  const unif = () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const z = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) { const u = 1 - unif(), v = unif(); z[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  const b = blur(z, w, h, sigma);
  let s = 0; for (const v of b) s += v * v;
  const sd = Math.sqrt(s / b.length) || 1;
  return b.map((v) => v / sd);
}

/** Separable Gaussian blur (σ ≤ 0: copy), borders replicated. */
export function blur(a: Float32Array, w: number, h: number, sigma: number): Float32Array {
  if (!(sigma > 0)) return a.slice();
  const r = Math.ceil(3 * sigma), k = new Float64Array(2 * r + 1);
  let s = 0; for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma)); s += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  const t = new Float32Array(w * h), o = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0; for (let i = -r; i <= r; i++) v += k[i + r] * a[y * w + Math.max(0, Math.min(w - 1, x + i))];
    t[y * w + x] = v;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0; for (let i = -r; i <= r; i++) v += k[i + r] * t[Math.max(0, Math.min(h - 1, y + i)) * w + x];
    o[y * w + x] = v;
  }
  return o;
}

export interface Errors { noise: number; blur: number; shared: boolean; debris: boolean }
export interface Prediction { fg: Uint8Array; vx: Float32Array; vy: Float32Array; target: FlowTarget; targetLab: Int32Array }

/** Debris: a small blob in the top-left corner (not a cell) with random arrows. */
export function debrisMask(w: number, h: number): Uint8Array {
  const m = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (((x - 6) / 5) ** 2 + ((y - 6) / 4) ** 2 <= 1) m[y * w + x] = 1;
  return m;
}

/** Simulated network output: the exact target of the (possibly merged) cells, plus smooth
 *  noise on both components, then a blur of the arrows; outside the foreground 0. */
export function predict(lab: Int32Array, w: number, h: number, e: Errors, iters: number | null = null, seed = 7): Prediction {
  const targetLab = e.shared ? mergeTouching(lab, w, h) : lab;
  const target = flowTargets(targetLab, w, h, iters);
  const fg = Uint8Array.from(lab, (v) => (v ? 1 : 0));
  let vx: Float32Array = target.vx.slice(), vy: Float32Array = target.vy.slice();
  const dm = e.debris ? debrisMask(w, h) : null;
  const nx = smoothNoise(w, h, seed), ny = smoothNoise(w, h, seed + 101);
  if (dm) for (let p = 0; p < w * h; p++) if (dm[p] && !fg[p]) { fg[p] = 1; vx[p] = nx[p]; vy[p] = ny[p]; }
  // noise turns the arrows (they keep length 1, as the target's)
  if (e.noise > 0) for (let p = 0; p < w * h; p++) if (fg[p]) {
    const u = vx[p] + e.noise * nx[p], v = vy[p] + e.noise * ny[p], m = Math.hypot(u, v);
    vx[p] = m > 0 ? u / m : 0; vy[p] = m > 0 ? v / m : 0;
  }
  if (e.blur > 0) { vx = blur(vx, w, h, e.blur); vy = blur(vy, w, h, e.blur); }
  for (let p = 0; p < w * h; p++) if (!fg[p]) { vx[p] = 0; vy[p] = 0; }
  return { fg, vx, vy, target, targetLab };
}
