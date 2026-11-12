// Mathematical morphology (no DOM). Images are flat arrays with their sizes passed
// explicitly: grey images any ArrayLike<number> (results Float32Array), binary masks
// Uint8Array of 0/1 (non-zero input counts as 1).
//
// Structuring elements (SE) are lists of offsets (dx, dy) relative to the origin.
// Definitions (the adjoint pair, so opening/closing are idempotent for any SE):
//   dilation  δ_B(f)(x) = max_{b ∈ B} f(x − b)
//   erosion   ε_B(f)(x) = min_{b ∈ B} f(x + b)
// Outside the image is neutral by default (ignored: +∞ for erosion, −∞ for dilation),
// so an object touching the border is not eroded from outside; pass `outside` to change it.
//
// Sections: structuring elements · grey/binary erosion, dilation, opening, closing, top-hats,
// gradient, rank filters, hit-or-miss · geodesic operators and reconstruction · regional
// extrema, h-maxima/h-minima, minima imposition · fill holes, clear border.

import { Fifo } from "./pqueue";
import { label2d, Conn2, offsets2d } from "./label";

type A = ArrayLike<number>;

export interface SE { dx: Int32Array; dy: Int32Array }

function seFrom(points: [number, number][]): SE {
  return { dx: Int32Array.from(points, p => p[0]), dy: Int32Array.from(points, p => p[1]) };
}

/** Disk of radius r: offsets with dx² + dy² ≤ r² (+ a small tolerance, so r = 1 is the 3×3 cross). */
export function seDisk(r: number): SE {
  const p: [number, number][] = [], R = Math.floor(r);
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) if (dx * dx + dy * dy <= r * r + 1e-9) p.push([dx, dy]);
  return seFrom(p);
}

/** (2r+1) × (2r+1) square. */
export function seSquare(r: number): SE {
  const p: [number, number][] = [];
  for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) p.push([dx, dy]);
  return seFrom(p);
}

/** Diamond |dx| + |dy| ≤ r (r = 1: the 4-neighbour cross). */
export function seDiamond(r: number): SE {
  const p: [number, number][] = [];
  for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) if (Math.abs(dx) + Math.abs(dy) <= r) p.push([dx, dy]);
  return seFrom(p);
}

/** Digital line segment of `length` pixels through the origin at `angle` radians (x right, y down). */
export function seLine(length: number, angle: number): SE {
  const set = new Set<string>(), p: [number, number][] = [], h = (length - 1) / 2;
  const steps = Math.max(1, Math.ceil(length * 2));
  for (let k = 0; k <= steps; k++) {
    const t = -h + (2 * h * k) / steps;
    const dx = Math.round(t * Math.cos(angle)), dy = Math.round(t * Math.sin(angle)), key = dx + "," + dy;
    if (!set.has(key)) { set.add(key); p.push([dx, dy]); }
  }
  return seFrom(p);
}

/** SE from a w × h 0/1 mask with the origin at (ox, oy) (default the centre). */
export function seFromMask(mask: A, w: number, h: number, ox = w >> 1, oy = h >> 1): SE {
  const p: [number, number][] = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[y * w + x]) p.push([x - ox, y - oy]);
  return seFrom(p);
}

/** Reflected SE B̌ = {−b}. */
export function seReflect(se: SE): SE {
  return { dx: se.dx.map(v => -v), dy: se.dy.map(v => -v) };
}

/** Bounding radius of an SE (max |offset| component). */
export function seRadius(se: SE): number {
  let r = 0;
  for (let i = 0; i < se.dx.length; i++) r = Math.max(r, Math.abs(se.dx[i]), Math.abs(se.dy[i]));
  return r;
}

// ---------------------------------------------------------------------------------------
// Grey-level operators

function extremum(f: A, w: number, h: number, se: SE, isMax: boolean, outside?: number): Float32Array {
  const out = new Float32Array(w * h), n = se.dx.length, s = isMax ? -1 : 1;   // dilation reads x − b
  const neutral = isMax ? -Infinity : Infinity, ov = outside ?? neutral;
  const ox = Int32Array.from(se.dx, v => s * v), oy = Int32Array.from(se.dy, v => s * v), lin = new Int32Array(n);
  let x0 = 0, x1 = w, y0 = 0, y1 = h;   // interior box where every offset stays inside
  for (let k = 0; k < n; k++) {
    lin[k] = oy[k] * w + ox[k];
    x0 = Math.max(x0, -ox[k]); x1 = Math.min(x1, w - ox[k]); y0 = Math.max(y0, -oy[k]); y1 = Math.min(y1, h - oy[k]);
  }
  for (let y = 0; y < h; y++) {
    const inY = y >= y0 && y < y1;
    for (let x = 0; x < w; x++) {
      const p = y * w + x;
      let best = neutral;
      if (inY && x >= x0 && x < x1) {
        if (isMax) for (let k = 0; k < n; k++) { const v = f[p + lin[k]]; if (v > best) best = v; }
        else for (let k = 0; k < n; k++) { const v = f[p + lin[k]]; if (v < best) best = v; }
      } else {
        for (let k = 0; k < n; k++) {
          const xx = x + ox[k], yy = y + oy[k];
          const v = xx < 0 || xx >= w || yy < 0 || yy >= h ? ov : f[yy * w + xx];
          if (isMax ? v > best : v < best) best = v;
        }
      }
      out[p] = n ? best : neutral;
    }
  }
  return out;
}

/** Grey erosion ε_B(f)(x) = min_{b∈B} f(x + b). */
export function erode(f: A, w: number, h: number, se: SE, outside?: number): Float32Array {
  return extremum(f, w, h, se, false, outside);
}

/** Grey dilation δ_B(f)(x) = max_{b∈B} f(x − b). */
export function dilate(f: A, w: number, h: number, se: SE, outside?: number): Float32Array {
  return extremum(f, w, h, se, true, outside);
}

/** Opening δ_B(ε_B(f)): removes bright details smaller than B. */
export function open(f: A, w: number, h: number, se: SE): Float32Array {
  return dilate(erode(f, w, h, se), w, h, se);
}

/** Closing ε_B(δ_B(f)): fills dark details smaller than B. */
export function close(f: A, w: number, h: number, se: SE): Float32Array {
  return erode(dilate(f, w, h, se), w, h, se);
}

/** White top-hat f − open(f): small bright structures on any background. */
export function whiteTopHat(f: A, w: number, h: number, se: SE): Float32Array {
  const o = open(f, w, h, se);
  for (let i = 0; i < o.length; i++) o[i] = f[i] - o[i];
  return o;
}

/** Black top-hat close(f) − f: small dark structures. */
export function blackTopHat(f: A, w: number, h: number, se: SE): Float32Array {
  const c = close(f, w, h, se);
  for (let i = 0; i < c.length; i++) c[i] -= f[i];
  return c;
}

/** Morphological gradient δ(f) − ε(f). */
export function morphGradient(f: A, w: number, h: number, se: SE): Float32Array {
  const d = dilate(f, w, h, se), e = erode(f, w, h, se);
  for (let i = 0; i < d.length; i++) d[i] -= e[i];
  return d;
}

/**
 * Rank filter: the value of rank `q` (0 = min, 0.5 = median, 1 = max) among the pixels
 * under the SE centred at x (only inside pixels count). Sorting per pixel: small SEs.
 */
export function rankFilter(f: A, w: number, h: number, se: SE, q: number): Float32Array {
  const out = new Float32Array(w * h), buf = new Float64Array(se.dx.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let m = 0;
    for (let k = 0; k < se.dx.length; k++) {
      const xx = x + se.dx[k], yy = y + se.dy[k];
      if (xx >= 0 && xx < w && yy >= 0 && yy < h) buf[m++] = f[yy * w + xx];
    }
    const s = buf.subarray(0, m).sort();
    out[y * w + x] = s[Math.round(q * (m - 1))];
  }
  return out;
}

/** Median filter over the SE (rank 0.5). */
export function median(f: A, w: number, h: number, se: SE): Float32Array {
  return rankFilter(f, w, h, se, 0.5);
}

// ---------------------------------------------------------------------------------------
// Binary versions (Uint8Array 0/1)

const toBin = (g: Float32Array): Uint8Array => { const b = new Uint8Array(g.length); for (let i = 0; i < g.length; i++) b[i] = g[i] > 0 ? 1 : 0; return b; };
const as01 = (m: A): Uint8Array => { const b = new Uint8Array(m.length); for (let i = 0; i < m.length; i++) b[i] = m[i] ? 1 : 0; return b; };

/** Binary erosion: pixels x with x + b in the object for every b ∈ B (outside: neutral, or 0 with `zeroOutside`). */
export function erodeBinary(m: A, w: number, h: number, se: SE, zeroOutside = false): Uint8Array {
  return toBin(erode(as01(m), w, h, se, zeroOutside ? 0 : undefined));
}
export function dilateBinary(m: A, w: number, h: number, se: SE): Uint8Array {
  return toBin(dilate(as01(m), w, h, se));
}
export function openBinary(m: A, w: number, h: number, se: SE): Uint8Array {
  return toBin(open(as01(m), w, h, se));
}
export function closeBinary(m: A, w: number, h: number, se: SE): Uint8Array {
  return toBin(close(as01(m), w, h, se));
}

/** Hit-or-miss: object pixels where `hit` fits in the object and `miss` fits in the background. */
export function hitOrMiss(m: A, w: number, h: number, hit: SE, miss: SE): Uint8Array {
  const a = erodeBinary(m, w, h, hit, true);
  const inv = new Uint8Array(m.length);
  for (let i = 0; i < m.length; i++) inv[i] = m[i] ? 0 : 1;
  const b = toBin(erode(inv, w, h, miss, 1));   // background continues outside the image
  for (let i = 0; i < a.length; i++) a[i] &= b[i];
  return a;
}

// ---------------------------------------------------------------------------------------
// Geodesic operators and reconstruction

/** One geodesic dilation step: min(δ(marker) over the 3×3 (8) or cross (4) neighbourhood, mask). */
export function geodesicDilate(marker: A, mask: A, w: number, h: number, conn: Conn2 = 8): Float32Array {
  const d = dilate(marker, w, h, conn === 8 ? seSquare(1) : seDiamond(1));
  for (let i = 0; i < d.length; i++) d[i] = Math.min(d[i], mask[i]);
  return d;
}

/** One geodesic erosion step: max(ε(marker), mask). */
export function geodesicErode(marker: A, mask: A, w: number, h: number, conn: Conn2 = 8): Float32Array {
  const e = erode(marker, w, h, conn === 8 ? seSquare(1) : seDiamond(1));
  for (let i = 0; i < e.length; i++) e[i] = Math.max(e[i], mask[i]);
  return e;
}

/**
 * Reconstruction by dilation of `marker` under `mask` (marker ≤ mask is enforced by a
 * pointwise min): iterate geodesic dilation until stable. Fast hybrid algorithm (Vincent
 * 1993): one raster and one anti-raster pass, then a FIFO propagation.
 */
export function reconstructByDilation(marker: A, mask: A, w: number, h: number, conn: Conn2 = 8): Float32Array {
  const n = w * h, J = new Float32Array(n), I = Float32Array.from(mask);
  for (let i = 0; i < n; i++) J[i] = Math.min(marker[i], I[i]);
  const nb = offsets2d(conn), m = nb.length;
  const NX = Int32Array.from(nb, o => o[0]), NY = Int32Array.from(nb, o => o[1]);
  // N+ = neighbours visited before in raster order, N− = after (as index lists into NX/NY)
  const plus = [...nb.keys()].filter(k => NY[k] < 0 || (NY[k] === 0 && NX[k] < 0));
  const minus = [...nb.keys()].filter(k => NY[k] > 0 || (NY[k] === 0 && NX[k] > 0));
  const PX = Int32Array.from(plus, k => NX[k]), PY = Int32Array.from(plus, k => NY[k]);
  const MX = Int32Array.from(minus, k => NX[k]), MY = Int32Array.from(minus, k => NY[k]), np = PX.length;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    let v = J[p];
    for (let k = 0; k < np; k++) {
      const xx = x + PX[k], yy = y + PY[k];
      if (xx >= 0 && xx < w && yy >= 0) { const u = J[yy * w + xx]; if (u > v) v = u; }
    }
    J[p] = v < I[p] ? v : I[p];
  }
  const q = new Fifo(1024);
  for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
    const p = y * w + x;
    let v = J[p];
    for (let k = 0; k < np; k++) {
      const xx = x + MX[k], yy = y + MY[k];
      if (xx >= 0 && xx < w && yy < h) { const u = J[yy * w + xx]; if (u > v) v = u; }
    }
    const jp = (J[p] = v < I[p] ? v : I[p]);
    for (let k = 0; k < np; k++) {
      const xx = x + MX[k], yy = y + MY[k];
      if (xx < 0 || xx >= w || yy >= h) continue;
      const r = yy * w + xx;
      if (J[r] < jp && J[r] < I[r]) { q.push(p); break; }
    }
  }
  while (!q.isEmpty()) {
    const p = q.shift(), x = p % w, y = (p - x) / w, jp = J[p];
    for (let k = 0; k < m; k++) {
      const xx = x + NX[k], yy = y + NY[k];
      if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
      const r = yy * w + xx;
      if (J[r] < jp && I[r] !== J[r]) { J[r] = jp < I[r] ? jp : I[r]; q.push(r); }
    }
  }
  return J;
}

/** Reconstruction by erosion of `marker` above `mask` (marker ≥ mask): the dual, via negation. */
export function reconstructByErosion(marker: A, mask: A, w: number, h: number, conn: Conn2 = 8): Float32Array {
  const n = w * h, a = new Float32Array(n), b = new Float32Array(n);
  for (let i = 0; i < n; i++) { a[i] = -marker[i]; b[i] = -mask[i]; }
  const r = reconstructByDilation(a, b, w, h, conn);
  for (let i = 0; i < n; i++) r[i] = -r[i];
  return r;
}

// ---------------------------------------------------------------------------------------
// Regional extrema and h-transforms

/**
 * Regional maxima: 0/1 mask of the connected plateaus (equal values, `conn`) with no
 * neighbour of higher value. Exact for real values (no ε).
 */
export function regionalMaxima(f: A, w: number, h: number, conn: Conn2 = 8): Uint8Array {
  const n = w * h, nb = offsets2d(conn);
  // label plateaus: components of equal value
  const seen = new Int32Array(n).fill(-1), out = new Uint8Array(n), q = new Fifo(256), members: number[] = [];
  for (let s = 0; s < n; s++) {
    if (seen[s] >= 0) continue;
    const v = f[s];
    let isMax = true;
    members.length = 0;
    seen[s] = s; q.push(s);
    while (!q.isEmpty()) {
      const p = q.shift(), x = p % w, y = (p - x) / w;
      members.push(p);
      for (const [dx, dy] of nb) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
        const r = yy * w + xx, fr = f[r];
        if (fr > v) isMax = false;
        else if (fr === v && seen[r] < 0) { seen[r] = s; q.push(r); }
      }
    }
    if (isMax) for (const p of members) out[p] = 1;
  }
  return out;
}

/** Regional minima (plateaus with no lower neighbour). */
export function regionalMinima(f: A, w: number, h: number, conn: Conn2 = 8): Uint8Array {
  const g = new Float64Array(w * h);
  for (let i = 0; i < g.length; i++) g[i] = -f[i];
  return regionalMaxima(g, w, h, conn);
}

/** h-maxima transform HMAX_h(f) = reconstruction by dilation of f − h under f: lowers every peak by up to h. */
export function hMaxTransform(f: A, w: number, h: number, hh: number, conn: Conn2 = 8): Float32Array {
  const m = new Float32Array(w * h);
  for (let i = 0; i < m.length; i++) m[i] = f[i] - hh;
  return reconstructByDilation(m, f, w, h, conn);
}

/** h-minima transform HMIN_h(f) = reconstruction by erosion of f + h above f: fills every basin shallower than h. */
export function hMinTransform(f: A, w: number, h: number, hh: number, conn: Conn2 = 8): Float32Array {
  const m = new Float32Array(w * h);
  for (let i = 0; i < m.length; i++) m[i] = f[i] + hh;
  return reconstructByErosion(m, f, w, h, conn);
}

/** h-maxima: 0/1 mask of the maxima with dynamic ≥ h (regional maxima of HMAX_h). */
export function hMaxima(f: A, w: number, h: number, hh: number, conn: Conn2 = 8): Uint8Array {
  return regionalMaxima(hMaxTransform(f, w, h, hh, conn), w, h, conn);
}

/** h-minima: 0/1 mask of the minima with depth ≥ h (regional minima of HMIN_h) — the usual watershed markers. */
export function hMinima(f: A, w: number, h: number, hh: number, conn: Conn2 = 8): Uint8Array {
  return regionalMinima(hMinTransform(f, w, h, hh, conn), w, h, conn);
}

/**
 * Minima imposition: modify f so its only regional minima are the marker pixels.
 * Marker pixels are set to min(f) − 1; every basin without a marker is filled up to its
 * spill level (reconstruction by erosion). The usual step before a marker watershed.
 */
export function imposeMinima(f: A, markers: A, w: number, h: number, conn: Conn2 = 8): Float32Array {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < f.length; i++) { if (f[i] < lo) lo = f[i]; if (f[i] > hi) hi = f[i]; }
  const low = lo - 1, high = hi + 1;
  const fm = new Float32Array(w * h), g = new Float32Array(w * h);
  for (let i = 0; i < fm.length; i++) {
    fm[i] = markers[i] ? low : high;
    g[i] = Math.min(f[i], fm[i]);
  }
  return reconstructByErosion(fm, g, w, h, conn);
}

// ---------------------------------------------------------------------------------------
// Binary utilities

/**
 * Fill holes: background components not connected to the image border become object.
 * `conn` is the connectivity of the BACKGROUND (4, the usual pair for an 8-connected object).
 */
export function fillHoles(m: A, w: number, h: number, conn: Conn2 = 4): Uint8Array {
  const bg = new Uint8Array(w * h);
  for (let i = 0; i < bg.length; i++) bg[i] = m[i] ? 0 : 1;
  const lab = label2d(bg, w, h, conn), touches = new Uint8Array(lab.count + 1);
  for (let x = 0; x < w; x++) { touches[lab.data[x]] = 1; touches[lab.data[(h - 1) * w + x]] = 1; }
  for (let y = 0; y < h; y++) { touches[lab.data[y * w]] = 1; touches[lab.data[y * w + w - 1]] = 1; }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = m[i] || !touches[lab.data[i]] ? 1 : 0;
  return out;
}

/** Remove the objects (components with connectivity `conn`) that touch the image border. */
export function clearBorder(m: A, w: number, h: number, conn: Conn2 = 8): Uint8Array {
  const lab = label2d(m, w, h, conn), touches = new Uint8Array(lab.count + 1);
  for (let x = 0; x < w; x++) { touches[lab.data[x]] = 1; touches[lab.data[(h - 1) * w + x]] = 1; }
  for (let y = 0; y < h; y++) { touches[lab.data[y * w]] = 1; touches[lab.data[y * w + w - 1]] = 1; }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = lab.data[i] && !touches[lab.data[i]] ? 1 : 0;
  return out;
}
