// Inpainting: filling the masked pixels of an image (owner: the edge-preserving smoothing day,
// unit inpainting/). No DOM: also used by the tests.
//
// Images are row-major Float64Arrays w × h; mask[p] = 1 marks a missing pixel (to be filled),
// 0 a known one. Neighbours are the 4-neighbours inside the image (no flux across the border).
//   harmonicJacobi(u, mask, w, h)        // one sweep: every hole pixel ← mean of its neighbours (old values)
//   harmonicGaussSeidel(u, mask, w, h)   // one sweep in place (new values used at once)
//   harmonicSolve(f, mask, w, h)         // Laplace's equation in the hole, by conjugate gradients
//   new TvInpaint(f, mask, w, h).run(n)  // minimise TV inside the hole (primal–dual iterations)
//   exemplarInpaint(f, mask, w, h, opts) // Criminisi-style patch copying, every step recorded
//
// The hole values of harmonic inpainting solve the sparse system L_ΩΩ u_Ω = −L_Ω∂ f_∂; Jacobi
// is the explicit heat step with dt = 1/4 run only inside the hole.

/** Neighbour indices of p (inside the image). */
function neighbours(p: number, w: number, h: number, out: number[]): number {
  const x = p % w, y = (p - x) / w;
  let n = 0;
  if (x > 0) out[n++] = p - 1;
  if (x < w - 1) out[n++] = p + 1;
  if (y > 0) out[n++] = p - w;
  if (y < h - 1) out[n++] = p + w;
  return n;
}

/** Initial guess: hole pixels set to `fill` (default the mean of the known pixels). */
export function initHole(f: ArrayLike<number>, mask: ArrayLike<number>, fill?: number): Float64Array {
  const u = Float64Array.from(f);
  if (fill === undefined) {
    let s = 0, n = 0;
    for (let i = 0; i < u.length; i++) if (!mask[i]) { s += u[i]; n++; }
    fill = n ? s / n : 0;
  }
  for (let i = 0; i < u.length; i++) if (mask[i]) u[i] = fill;
  return u;
}

/** One Jacobi sweep (returns a new array). */
export function harmonicJacobi(u: ArrayLike<number>, mask: ArrayLike<number>, w: number, h: number): Float64Array {
  const out = Float64Array.from(u), nb = [0, 0, 0, 0];
  for (let p = 0; p < w * h; p++) {
    if (!mask[p]) continue;
    const k = neighbours(p, w, h, nb);
    let s = 0;
    for (let i = 0; i < k; i++) s += u[nb[i]];
    out[p] = s / k;
  }
  return out;
}

/** One Gauss–Seidel sweep in raster order, in place. */
export function harmonicGaussSeidel(u: Float64Array, mask: ArrayLike<number>, w: number, h: number): Float64Array {
  const nb = [0, 0, 0, 0];
  for (let p = 0; p < w * h; p++) {
    if (!mask[p]) continue;
    const k = neighbours(p, w, h, nb);
    let s = 0;
    for (let i = 0; i < k; i++) s += u[nb[i]];
    u[p] = s / k;
  }
  return u;
}

/** Largest change of a hole pixel if one more Jacobi sweep were made (the residual / degree). */
export function harmonicResidual(u: ArrayLike<number>, mask: ArrayLike<number>, w: number, h: number): number {
  const nb = [0, 0, 0, 0];
  let r = 0;
  for (let p = 0; p < w * h; p++) {
    if (!mask[p]) continue;
    const k = neighbours(p, w, h, nb);
    let s = 0;
    for (let i = 0; i < k; i++) s += u[nb[i]];
    r = Math.max(r, Math.abs(s / k - u[p]));
  }
  return r;
}

/** Solve Laplace's equation in the hole with the known pixels as boundary values (conjugate
 *  gradients on the masked graph Laplacian). Returns the filled image, iterations and the final
 *  residual norm of the linear system. */
export function harmonicSolve(f: ArrayLike<number>, mask: ArrayLike<number>, w: number, h: number, tol = 1e-10, maxIter = 20000): { u: Float64Array; iters: number; residual: number } {
  const n = w * h, idx = new Int32Array(n).fill(-1), list: number[] = [];
  for (let p = 0; p < n; p++) if (mask[p]) { idx[p] = list.length; list.push(p); }
  const m = list.length, u = Float64Array.from(f);
  if (m === 0) return { u, iters: 0, residual: 0 };
  const deg = new Float64Array(m), b = new Float64Array(m), nb = [0, 0, 0, 0];
  for (let i = 0; i < m; i++) {
    const k = neighbours(list[i], w, h, nb);
    deg[i] = k;
    for (let j = 0; j < k; j++) if (!mask[nb[j]]) b[i] += f[nb[j]];
  }
  const A = (x: Float64Array, out: Float64Array) => {
    for (let i = 0; i < m; i++) {
      const k = neighbours(list[i], w, h, nb);
      let s = deg[i] * x[i];
      for (let j = 0; j < k; j++) { const q = idx[nb[j]]; if (q >= 0) s -= x[q]; }
      out[i] = s;
    }
  };
  // start from the mean of the boundary values
  const x = new Float64Array(m);
  let bs = 0, bn = 0;
  for (let p = 0; p < n; p++) if (!mask[p]) { bs += f[p]; bn++; }
  x.fill(bn ? bs / bn : 0);
  const r = new Float64Array(m), Ap = new Float64Array(m);
  A(x, Ap);
  for (let i = 0; i < m; i++) r[i] = b[i] - Ap[i];
  const p = Float64Array.from(r);
  let rr = 0; for (let i = 0; i < m; i++) rr += r[i] * r[i];
  const bnorm = Math.sqrt(b.reduce((s, v) => s + v * v, 0)) || 1;
  let it = 0;
  for (; it < maxIter && Math.sqrt(rr) > tol * bnorm; it++) {
    A(p, Ap);
    let pAp = 0; for (let i = 0; i < m; i++) pAp += p[i] * Ap[i];
    const a = rr / pAp;
    for (let i = 0; i < m; i++) { x[i] += a * p[i]; r[i] -= a * Ap[i]; }
    let rr2 = 0; for (let i = 0; i < m; i++) rr2 += r[i] * r[i];
    const beta = rr2 / rr;
    rr = rr2;
    for (let i = 0; i < m; i++) p[i] = r[i] + beta * p[i];
  }
  for (let i = 0; i < m; i++) u[list[i]] = x[i];
  // true residual
  A(x, Ap);
  let res = 0; for (let i = 0; i < m; i++) res += (b[i] - Ap[i]) ** 2;
  return { u, iters: it, residual: Math.sqrt(res) };
}

/** TV inpainting: minimise the isotropic total variation of u subject to u = f on the known
 *  pixels, by primal–dual iterations (Chambolle & Pock 2011; forward differences, Neumann
 *  border). Call run(n) repeatedly to animate. */
export class TvInpaint {
  readonly w: number; readonly h: number;
  u: Float64Array;
  iters = 0;
  private ub: Float64Array; private px: Float64Array; private py: Float64Array;
  private f: Float64Array; private mask: Uint8Array;
  constructor(f: ArrayLike<number>, mask: ArrayLike<number>, w: number, h: number, init?: ArrayLike<number>) {
    this.w = w; this.h = h;
    this.f = Float64Array.from(f); this.mask = Uint8Array.from(mask);
    this.u = init ? Float64Array.from(init) : initHole(f, mask);
    this.ub = Float64Array.from(this.u);
    this.px = new Float64Array(w * h); this.py = new Float64Array(w * h);
  }
  run(n: number): Float64Array {
    const { w, h, u, ub, px, py, f, mask } = this;
    const tau = 0.35, sig = 0.35, old = new Float64Array(w * h);
    for (let k = 0; k < n; k++) {
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const p = y * w + x;
        const gx = x < w - 1 ? ub[p + 1] - ub[p] : 0, gy = y < h - 1 ? ub[p + w] - ub[p] : 0;
        const qx = px[p] + sig * gx, qy = py[p] + sig * gy, m = Math.max(1, Math.hypot(qx, qy));
        px[p] = qx / m; py[p] = qy / m;
      }
      old.set(u);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const p = y * w + x;
        if (!mask[p]) { u[p] = f[p]; continue; }
        const div = (x < w - 1 ? px[p] : 0) - (x > 0 ? px[p - 1] : 0) + (y < h - 1 ? py[p] : 0) - (y > 0 ? py[p - w] : 0);
        u[p] += tau * div;
      }
      for (let p = 0; p < w * h; p++) ub[p] = 2 * u[p] - old[p];
      this.iters++;
    }
    return u;
  }
}

/** Run TV inpainting for `iters` iterations from the harmonic fill. */
export function tvInpaint(f: ArrayLike<number>, mask: ArrayLike<number>, w: number, h: number, iters = 2000): Float64Array {
  const t = new TvInpaint(f, mask, w, h, harmonicSolve(f, mask, w, h, 1e-6, 2000).u);
  return t.run(iters);
}

// ---------------------------------------------------------------------------------------
// Exemplar (patch) inpainting

export interface ExemplarStep {
  /** Centre of the target patch (on the fill front) and of the copied source patch. */
  tx: number; ty: number; sx: number; sy: number;
  priority: number; confidence: number; data: number;
  /** Pixels filled in this step. */
  filled: number[];
}
export interface ExemplarOptions {
  /** Patch half-size (patch = 2P+1); default 4 (9 × 9). */
  patch?: number;
  /** Search half-size around the target (source patches must be fully known); default 24. */
  search?: number;
  /** Use the data term (isophotes into the front); default true. */
  dataTerm?: boolean;
  /** Stop after this many steps (default: until the hole is filled). */
  maxSteps?: number;
}

/** Fill-front pixels: missing pixels with at least one known 4-neighbour. */
export function fillFront(mask: ArrayLike<number>, w: number, h: number): number[] {
  const out: number[] = [], nb = [0, 0, 0, 0];
  for (let p = 0; p < w * h; p++) {
    if (!mask[p]) continue;
    const k = neighbours(p, w, h, nb);
    for (let i = 0; i < k; i++) if (!mask[nb[i]]) { out.push(p); break; }
  }
  return out;
}

/** Exemplar inpainting (Criminisi, Pérez & Toyama 2004): repeatedly take the front pixel of
 *  highest priority P = C·D (C: confidence of the known pixels in its patch; D: strength of the
 *  isophote running into the front), find the fully known patch in the search window with the
 *  smallest squared difference over the target's known pixels, and copy its missing pixels.
 *  Returns the result, the fill order (step number per pixel, −1 = known) and the steps. */
export function exemplarInpaint(f: ArrayLike<number>, maskIn: ArrayLike<number>, w: number, h: number, o: ExemplarOptions = {}): { u: Float64Array; order: Int32Array; steps: ExemplarStep[] } {
  const P = o.patch ?? 4, S = o.search ?? 24, useData = o.dataTerm ?? true, maxSteps = o.maxSteps ?? Infinity;
  const n = w * h, u = Float64Array.from(f), mask = Uint8Array.from(maskIn);
  const C = new Float64Array(n), order = new Int32Array(n).fill(-1), steps: ExemplarStep[] = [];
  for (let p = 0; p < n; p++) { C[p] = mask[p] ? 0 : 1; if (mask[p]) u[p] = 0; }
  // source patches must be fully known in the ORIGINAL image
  const orig = Uint8Array.from(maskIn);
  const known = (x: number, y: number) => x >= 0 && y >= 0 && x < w && y < h && !mask[y * w + x];
  while (steps.length < maxSteps) {
    const front = fillFront(mask, w, h);
    if (!front.length) break;
    // priorities
    let best = -1, bestP = -Infinity, bestC = 0, bestD = 0;
    for (const p of front) {
      const x = p % w, y = (p - x) / w;
      let cs = 0, cnt = 0;
      for (let j = -P; j <= P; j++) for (let i = -P; i <= P; i++) {
        const xx = x + i, yy = y + j;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        cnt++; if (!mask[yy * w + xx]) cs += C[yy * w + xx];
      }
      const conf = cs / Math.max(1, cnt);
      let D = 0.001;
      if (useData) {
        // front normal from the mask (pointing out of the hole), isophote from the strongest
        // known gradient in the 3 × 3 neighbourhood
        let nx = 0, ny = 0;
        for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) { const m = known(x + i, y + j) ? 1 : 0; nx += i * m; ny += j * m; }
        const nl = Math.hypot(nx, ny) || 1; nx /= nl; ny /= nl;
        let gx = 0, gy = 0, gm = 0;
        for (let j = -2; j <= 2; j++) for (let i = -2; i <= 2; i++) {
          const xx = x + i, yy = y + j;
          if (!known(xx - 1, yy) || !known(xx + 1, yy) || !known(xx, yy - 1) || !known(xx, yy + 1)) continue;
          const ax = 0.5 * (u[yy * w + xx + 1] - u[yy * w + xx - 1]), ay = 0.5 * (u[(yy + 1) * w + xx] - u[(yy - 1) * w + xx]);
          const m = Math.hypot(ax, ay);
          if (m > gm) { gm = m; gx = ax; gy = ay; }
        }
        // isophote = gradient rotated by 90°; D = |isophote · n| / 255
        D += Math.abs(-gy * nx + gx * ny) / 255;
      }
      const pr = conf * D;
      if (pr > bestP) { bestP = pr; best = p; bestC = conf; bestD = D; }
    }
    const tx = best % w, ty = (best - tx) / w;
    // best source patch (fully known in the original image, inside the image)
    let sBest = Infinity, sx = -1, sy = -1;
    for (let cy = Math.max(P, ty - S); cy <= Math.min(h - 1 - P, ty + S); cy++) for (let cx = Math.max(P, tx - S); cx <= Math.min(w - 1 - P, tx + S); cx++) {
      let ok = true, ssd = 0;
      for (let j = -P; j <= P && ok; j++) for (let i = -P; i <= P; i++) {
        const q = (cy + j) * w + cx + i;
        if (orig[q]) { ok = false; break; }
        const xx = tx + i, yy = ty + j;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h || mask[yy * w + xx]) continue;
        const d = u[yy * w + xx] - u[q]; ssd += d * d;
        if (ssd >= sBest) { ok = false; break; }
      }
      if (ok && ssd < sBest) { sBest = ssd; sx = cx; sy = cy; }
    }
    const filled: number[] = [];
    if (sx < 0) {
      // no complete source patch: fall back to the mean of the known neighbours
      const nb = [0, 0, 0, 0], k = neighbours(best, w, h, nb);
      let s = 0, c = 0;
      for (let i = 0; i < k; i++) if (!mask[nb[i]]) { s += u[nb[i]]; c++; }
      u[best] = c ? s / c : 0; mask[best] = 0; C[best] = bestC; order[best] = steps.length; filled.push(best);
    } else {
      for (let j = -P; j <= P; j++) for (let i = -P; i <= P; i++) {
        const xx = tx + i, yy = ty + j;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const q = yy * w + xx;
        if (!mask[q]) continue;
        u[q] = u[(sy + j) * w + sx + i]; mask[q] = 0; C[q] = bestC; order[q] = steps.length; filled.push(q);
      }
    }
    steps.push({ tx, ty, sx, sy, priority: bestP, confidence: bestC, data: bestD, filled });
  }
  return { u, order, steps };
}

/** Masked sum of squared differences between a target patch with some missing pixels (NaN) and
 *  a candidate patch (same length). */
export function maskedSSD(target: ArrayLike<number>, cand: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < target.length; i++) if (Number.isFinite(target[i])) s += (target[i] - cand[i]) ** 2;
  return s;
}
