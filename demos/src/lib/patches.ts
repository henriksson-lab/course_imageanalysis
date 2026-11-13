// Patches, dictionaries and sparse coding (owner: the sparse representations day,
// dictionaries/ unit). No DOM; runs under node.
//
// A dictionary D holds `k` atoms of dimension d = p·p, atom j at D[j*d … j*d + d) (atom-major).
// Patches are stored the same way: patch i at Y[i*d …]. Codes keep at most `maxK` nonzeros
// per patch: idx[i*maxK + s] (−1 = unused) and val[i*maxK + s].
//
// - extractPatches / aggregatePatches: overlapping p × p patches and the weighted average that
//   puts them back (with the noisy image as a prior term: (λy + Σ Rᵀp) / (λ + count)).
// - dctDictionary(p, k): the overcomplete 2D DCT (√k frequencies per axis, Elad & Aharon 2006).
// - batchOmp: OMP for many patches with one precomputed Gram matrix (Rubinstein et al. 2008),
//   stopping at K atoms or when the residual energy falls below a target.
// - ksvdStep (Aharon, Elad & Bruckstein 2006), modStep (Engan et al. 1999), kmeansStep (k-means
//   via lib/kmeans.ts: one atom, coefficient 1).
// - ksvdDenoise, patchInpaint: restoration with a dictionary.
// - A same-page cache (globalThis) so a dictionary learnt by one demo can be used by another.
import { dctMatrix } from "./dct";
import { cholesky, choleskySolve } from "./linalg";
import type { Rng } from "./random";
import { sampleIndices } from "./random";
import { assignStep, updateStep } from "./kmeans";

export type Vec = Float64Array;

export interface Patches { Y: Vec; pos: Int32Array; n: number; d: number; p: number }

/** Positions 0, stride, 2·stride, …, and always the last one (size − p). */
export function patchStarts(size: number, p: number, stride: number): number[] {
  const out: number[] = [];
  for (let s = 0; s + p <= size; s += stride) out.push(s);
  if (out[out.length - 1] !== size - p) out.push(size - p);
  return out;
}

/** All p × p patches of a w × h image at the given stride (covering the borders). */
export function extractPatches(img: ArrayLike<number>, w: number, h: number, p: number, stride = 1): Patches {
  const xs = patchStarts(w, p, stride), ys = patchStarts(h, p, stride), d = p * p, n = xs.length * ys.length;
  const Y = new Float64Array(n * d), pos = new Int32Array(2 * n);
  let i = 0;
  for (const y0 of ys) for (const x0 of xs) {
    for (let v = 0; v < p; v++) for (let u = 0; u < p; u++) Y[i * d + v * p + u] = img[(y0 + v) * w + x0 + u];
    pos[2 * i] = x0; pos[2 * i + 1] = y0; i++;
  }
  return { Y, pos, n, d, p };
}

/** Average overlapping patches back into a w × h image: x = (λ·base + Σ Rᵀ patch) / (λ + count).
 *  `mask` (optional) restricts the base term to known pixels (inpainting). */
export function aggregatePatches(Yhat: ArrayLike<number>, pos: ArrayLike<number>, n: number, p: number, w: number, h: number,
  base?: ArrayLike<number>, lambda = 0, mask?: ArrayLike<number>): Vec {
  const acc = new Float64Array(w * h), cnt = new Float64Array(w * h), d = p * p;
  for (let i = 0; i < n; i++) {
    const x0 = pos[2 * i], y0 = pos[2 * i + 1];
    for (let v = 0; v < p; v++) for (let u = 0; u < p; u++) { const q = (y0 + v) * w + x0 + u; acc[q] += Yhat[i * d + v * p + u]; cnt[q]++; }
  }
  if (base && lambda > 0) for (let q = 0; q < w * h; q++) { const m = mask ? (mask[q] ? 1 : 0) : 1; acc[q] += lambda * m * base[q]; cnt[q] += lambda * m; }
  for (let q = 0; q < w * h; q++) acc[q] = cnt[q] > 0 ? acc[q] / cnt[q] : (base ? base[q] : 0);
  return acc;
}

/** Normalise every atom to unit norm (zero atoms are left alone). Returns the norms. */
export function normaliseAtoms(D: Vec, d: number): Vec {
  const k = D.length / d, nrm = new Float64Array(k);
  for (let j = 0; j < k; j++) {
    let s = 0; for (let t = 0; t < d; t++) s += D[j * d + t] ** 2;
    s = Math.sqrt(s); nrm[j] = s;
    if (s > 0) for (let t = 0; t < d; t++) D[j * d + t] /= s;
  }
  return nrm;
}

/** Overcomplete 2D DCT dictionary of k atoms (k a square, √k ≥ p) for p × p patches: 1D atoms
 *  cos(π·f·n/√k), mean removed for f > 0, normalised; 2D atoms are their outer products. */
export function dctDictionary(p: number, k: number): Vec {
  const m = Math.round(Math.sqrt(k));
  if (m === p) {   // the orthonormal DCT-II basis
    const C = dctMatrix(p), D = new Float64Array(p * p * p * p), d = p * p;
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) {
      const j = a * p + b;
      for (let v = 0; v < p; v++) for (let u = 0; u < p; u++) D[j * d + v * p + u] = C[a * p + v] * C[b * p + u];
    }
    return D;
  }
  const one = new Float64Array(m * p);
  for (let f = 0; f < m; f++) {
    let mean = 0;
    for (let t = 0; t < p; t++) { one[f * p + t] = Math.cos(Math.PI * f * t / m); mean += one[f * p + t]; }
    mean /= p;
    let s = 0;
    for (let t = 0; t < p; t++) { if (f > 0) one[f * p + t] -= mean; s += one[f * p + t] ** 2; }
    s = Math.sqrt(s) || 1;
    for (let t = 0; t < p; t++) one[f * p + t] /= s;
  }
  const d = p * p, D = new Float64Array(m * m * d);
  for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) {
    const j = a * m + b;
    for (let v = 0; v < p; v++) for (let u = 0; u < p; u++) D[j * d + v * p + u] = one[a * p + v] * one[b * p + u];
  }
  normaliseAtoms(D, d);
  return D;
}

/** k atoms = randomly chosen patches (the first atom is the constant patch), normalised. */
export function randomPatchDictionary(P: Patches, k: number, r: Rng): Vec {
  const d = P.d, D = new Float64Array(k * d);
  for (let t = 0; t < d; t++) D[t] = 1 / Math.sqrt(d);
  const idx = sampleIndices(r, P.n, k - 1);
  for (let j = 1; j < k; j++) {
    const i = idx[(j - 1) % idx.length];
    let mean = 0; for (let t = 0; t < d; t++) mean += P.Y[i * d + t] / d;
    for (let t = 0; t < d; t++) D[j * d + t] = P.Y[i * d + t] - mean + 1e-6 * Math.sin(7 * j + t);
  }
  normaliseAtoms(D, d);
  return D;
}

/** Gram matrix G = DᵀD (k × k). */
export function gram(D: ArrayLike<number>, d: number): Vec {
  const k = D.length / d, G = new Float64Array(k * k);
  for (let a = 0; a < k; a++) for (let b = a; b < k; b++) {
    let s = 0; for (let t = 0; t < d; t++) s += D[a * d + t] * D[b * d + t];
    G[a * k + b] = s; G[b * k + a] = s;
  }
  return G;
}

export interface Codes { n: number; maxK: number; idx: Int32Array; val: Vec; err: Vec }

/** Batch-OMP: for each patch, greedy selection with the residual correlations updated through the
 *  Gram matrix, stopping at K atoms or when ‖r‖² ≤ errTarget. `G` may be passed in (precomputed). */
export function batchOmp(D: ArrayLike<number>, d: number, Y: ArrayLike<number>, n: number, opts: { K: number; errTarget?: number; G?: Vec }): Codes {
  const k = D.length / d, K = Math.max(1, Math.min(opts.K, k, d)), G = opts.G ?? gram(D, d);
  const target = opts.errTarget ?? 0;
  const idx = new Int32Array(n * K).fill(-1), val = new Float64Array(n * K), err = new Float64Array(n);
  const a0 = new Float64Array(k), a = new Float64Array(k), Gs = new Float64Array(K * K), b = new Float64Array(K);
  const S: number[] = [];
  for (let i = 0; i < n; i++) {
    const yo = i * d;
    let yy = 0; for (let t = 0; t < d; t++) yy += Y[yo + t] ** 2;
    for (let j = 0; j < k; j++) { let s = 0; const o = j * d; for (let t = 0; t < d; t++) s += D[o + t] * Y[yo + t]; a0[j] = s; a[j] = s; }
    S.length = 0;
    let e2 = yy, gamma: Vec = new Float64Array(0);
    while (S.length < K && e2 > target) {
      let best = -1, bv = 1e-12;
      for (let j = 0; j < k; j++) { const v = Math.abs(a[j]); if (v > bv && !S.includes(j)) { bv = v; best = j; } }
      if (best < 0) break;
      S.push(best);
      const s = S.length;
      for (let p = 0; p < s; p++) { b[p] = a0[S[p]]; for (let q = 0; q < s; q++) Gs[p * s + q] = G[S[p] * k + S[q]]; }
      const L = cholesky(Gs.subarray(0, s * s), s);
      if (!L) { S.pop(); break; }
      gamma = choleskySolve(L, b.subarray(0, s), s);
      for (let j = 0; j < k; j++) { let v = a0[j]; for (let p = 0; p < s; p++) v -= G[j * k + S[p]] * gamma[p]; a[j] = v; }
      let yg = 0; for (let p = 0; p < s; p++) yg += a0[S[p]] * gamma[p];
      e2 = Math.max(0, yy - yg);
    }
    for (let p = 0; p < S.length; p++) { idx[i * K + p] = S[p]; val[i * K + p] = gamma[p]; }
    err[i] = e2;
  }
  return { n, maxK: K, idx, val, err };
}

/** Patches rebuilt from codes: Ŷ = D C. */
export function reconstruct(D: ArrayLike<number>, d: number, c: Codes): Vec {
  const Y = new Float64Array(c.n * d);
  for (let i = 0; i < c.n; i++) for (let s = 0; s < c.maxK; s++) {
    const j = c.idx[i * c.maxK + s]; if (j < 0) continue;
    const v = c.val[i * c.maxK + s], o = j * d;
    for (let t = 0; t < d; t++) Y[i * d + t] += v * D[o + t];
  }
  return Y;
}

/** Mean squared representation error per pixel, ‖Y − DC‖²_F / (n·d). */
export function representationError(D: ArrayLike<number>, d: number, Y: ArrayLike<number>, c: Codes): number {
  const R = reconstruct(D, d, c);
  let s = 0; for (let i = 0; i < R.length; i++) s += (Y[i] - R[i]) ** 2;
  return s / R.length;
}

/** How many patches use each atom. */
export function atomUsage(c: Codes, k: number): Int32Array {
  const u = new Int32Array(k);
  for (let i = 0; i < c.idx.length; i++) if (c.idx[i] >= 0) u[c.idx[i]]++;
  return u;
}

/** Leading singular pair of the d × m matrix E (columns = residual patches): returns the unit left
 *  vector u and σv (the new coefficients). Power iteration on EEᵀ started from `u0`. */
export function rank1(E: ArrayLike<number>, d: number, m: number, u0?: ArrayLike<number>, iters = 30): { u: Vec; sv: Vec } {
  let u: Vec = u0 ? Float64Array.from(u0) : Float64Array.from({ length: d }, (_, t) => 1 + 0.01 * t);
  const tmp = new Float64Array(m);
  for (let it = 0; it < iters; it++) {
    for (let c = 0; c < m; c++) { let s = 0; for (let t = 0; t < d; t++) s += E[c * d + t] * u[t]; tmp[c] = s; }
    const nu = new Float64Array(d);
    for (let c = 0; c < m; c++) { const v = tmp[c]; for (let t = 0; t < d; t++) nu[t] += E[c * d + t] * v; }
    let s = 0; for (let t = 0; t < d; t++) s += nu[t] * nu[t];
    s = Math.sqrt(s);
    if (s === 0) break;
    for (let t = 0; t < d; t++) nu[t] /= s;
    u = nu;
  }
  const sv = new Float64Array(m);
  for (let c = 0; c < m; c++) { let s = 0; for (let t = 0; t < d; t++) s += E[c * d + t] * u[t]; sv[c] = s; }
  return { u, sv };
}

/** One K-SVD dictionary update (after sparse coding): for each atom, the patches that use it,
 *  their residual without this atom, and the best rank-1 fit (new atom and its coefficients).
 *  An unused atom is replaced by the worst-represented patch. D and codes change in place. */
export function ksvdUpdate(D: Vec, d: number, Y: ArrayLike<number>, c: Codes): void {
  const k = D.length / d, K = c.maxK;
  const users: number[][] = Array.from({ length: k }, () => []);
  for (let i = 0; i < c.n; i++) for (let s = 0; s < K; s++) { const j = c.idx[i * K + s]; if (j >= 0) users[j].push(i * K + s); }
  // residuals R = Y − DC, maintained as atoms change
  const R = Float64Array.from(Y), Yh = reconstruct(D, d, c);
  for (let t = 0; t < R.length; t++) R[t] -= Yh[t];
  const replaced = new Set<number>();
  for (let j = 0; j < k; j++) {
    const us = users[j], m = us.length, o = j * d;
    if (!m) {
      // replace by the worst-represented patch (not yet used for a replacement)
      let worst = -1, we = -1;
      for (let i = 0; i < c.n; i++) {
        if (replaced.has(i)) continue;
        let e = 0; for (let t = 0; t < d; t++) e += R[i * d + t] ** 2;
        if (e > we) { we = e; worst = i; }
      }
      if (worst >= 0) {
        replaced.add(worst);
        let s = 0; for (let t = 0; t < d; t++) s += R[worst * d + t] ** 2;
        s = Math.sqrt(s) || 1;
        for (let t = 0; t < d; t++) D[o + t] = R[worst * d + t] / s;
      }
      continue;
    }
    const E = new Float64Array(m * d);
    for (let q = 0; q < m; q++) {
      const slot = us[q], i = Math.floor(slot / K), v = c.val[slot];
      for (let t = 0; t < d; t++) E[q * d + t] = R[i * d + t] + v * D[o + t];
    }
    const { u, sv } = rank1(E, d, m, D.subarray(o, o + d));
    for (let t = 0; t < d; t++) D[o + t] = u[t];
    for (let q = 0; q < m; q++) {
      const slot = us[q], i = Math.floor(slot / K);
      c.val[slot] = sv[q];
      for (let t = 0; t < d; t++) R[i * d + t] = E[q * d + t] - sv[q] * u[t];
    }
  }
}

/** One MOD update: D = Y Cᵀ (C Cᵀ + εI)⁻¹, atoms renormalised (codes rescaled to match). */
export function modUpdate(D: Vec, d: number, Y: ArrayLike<number>, c: Codes): void {
  const k = D.length / d, K = c.maxK;
  const CC = new Float64Array(k * k), YC = new Float64Array(k * d);
  for (let i = 0; i < c.n; i++) for (let s = 0; s < K; s++) {
    const a = c.idx[i * K + s]; if (a < 0) continue;
    const va = c.val[i * K + s];
    for (let t = 0; t < d; t++) YC[a * d + t] += va * Y[i * d + t];
    for (let s2 = 0; s2 < K; s2++) { const b = c.idx[i * K + s2]; if (b >= 0) CC[a * k + b] += va * c.val[i * K + s2]; }
  }
  let tr = 0; for (let a = 0; a < k; a++) tr += CC[a * k + a];
  const eps = 1e-6 * (tr / k || 1);
  for (let a = 0; a < k; a++) CC[a * k + a] += eps;
  const L = cholesky(CC, k);
  if (!L) return;
  const col = new Float64Array(k);
  for (let t = 0; t < d; t++) {
    for (let a = 0; a < k; a++) col[a] = YC[a * d + t];
    const sol = choleskySolve(L, col, k);
    for (let a = 0; a < k; a++) D[a * d + t] = sol[a];
  }
  const nrm = normaliseAtoms(D, d);
  for (let i = 0; i < c.idx.length; i++) if (c.idx[i] >= 0) c.val[i] *= nrm[c.idx[i]];
}

/** One k-means step on the patches (lib/kmeans.ts): assign each patch to its nearest centre
 *  (code = that atom with coefficient 1), then move each centre to the mean of its patches.
 *  Here the atoms are NOT normalised (they are the cluster means). Returns the codes. */
export function kmeansStep(D: Vec, d: number, Y: ArrayLike<number>, n: number): Codes {
  const k = D.length / d;
  const { assign } = assignStep(Y, n, d, D, k);
  D.set(updateStep(Y, n, d, assign, k, undefined, D).centres);   // empty clusters keep their centre
  const c: Codes = { n, maxK: 1, idx: Int32Array.from(assign), val: new Float64Array(n).fill(1), err: new Float64Array(n) };
  return c;
}

export type DictMethod = "ksvd" | "mod" | "kmeans";

/** One learning iteration: sparse coding (OMP with K atoms, or nearest centre for k-means) and the
 *  dictionary update. Returns the codes after the update and the representation error. */
export function learnStep(D: Vec, d: number, P: Patches, K: number, method: DictMethod): { codes: Codes; error: number } {
  if (method === "kmeans") {
    const codes = kmeansStep(D, d, P.Y, P.n);
    return { codes, error: representationError(D, d, P.Y, codes) };
  }
  const codes = batchOmp(D, d, P.Y, P.n, { K });
  if (method === "ksvd") ksvdUpdate(D, d, P.Y, codes); else modUpdate(D, d, P.Y, codes);
  return { codes, error: representationError(D, d, P.Y, codes) };
}

/** Principal components of the patches (uncentred, like a dictionary through 0): the top `k`
 *  eigenvectors of Σ y yᵀ, by power iteration with deflation (d ≤ 64). */
export function patchPCA(P: Patches, k: number): Vec {
  const d = P.d, C = new Float64Array(d * d);
  for (let i = 0; i < P.n; i++) for (let a = 0; a < d; a++) { const ya = P.Y[i * d + a]; for (let b = a; b < d; b++) C[a * d + b] += ya * P.Y[i * d + b]; }
  for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) C[a * d + b] = C[b * d + a];
  const out = new Float64Array(k * d);
  for (let j = 0; j < k; j++) {
    let v: Vec = Float64Array.from({ length: d }, (_, t) => Math.cos(t * (j + 1)) + 0.1);
    let lam = 0;
    for (let it = 0; it < 200; it++) {
      const w = new Float64Array(d);
      for (let a = 0; a < d; a++) { let s = 0; for (let b = 0; b < d; b++) s += C[a * d + b] * v[b]; w[a] = s; }
      let s = 0; for (let a = 0; a < d; a++) s += w[a] * w[a];
      s = Math.sqrt(s); if (!s) break;
      lam = s;
      for (let a = 0; a < d; a++) w[a] /= s;
      v = w;
    }
    out.set(v, j * d);
    for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a * d + b] -= lam * v[a] * v[b];
  }
  return out;
}

/** Error of the best k-term approximation in an orthonormal set of atoms (PCA: the first k). */
export function linearApproxError(B: ArrayLike<number>, d: number, P: Patches, k: number): number {
  let s = 0;
  for (let i = 0; i < P.n; i++) {
    let e = 0; for (let t = 0; t < d; t++) e += P.Y[i * d + t] ** 2;
    for (let j = 0; j < k; j++) { let c = 0; for (let t = 0; t < d; t++) c += B[j * d + t] * P.Y[i * d + t]; e -= c * c; }
    s += Math.max(0, e);
  }
  return s / (P.n * d);
}

/** K-SVD-style denoising with a fixed dictionary: code every patch (stride) with OMP until the
 *  residual energy ≤ d·(C·σ)², average the overlapping patches with weight λ on the noisy image. */
export function ksvdDenoise(img: ArrayLike<number>, w: number, h: number, D: ArrayLike<number>, p: number, sigma: number,
  opts: { stride?: number; C?: number; lambda?: number; maxK?: number } = {}): Vec {
  const d = p * p, P = extractPatches(img, w, h, p, opts.stride ?? 1), C = opts.C ?? 1.15;
  const codes = batchOmp(D, d, P.Y, P.n, { K: opts.maxK ?? Math.floor(d / 2), errTarget: d * (C * sigma) ** 2 });
  return aggregatePatches(reconstruct(D, d, codes), P.pos, P.n, p, w, h, img, opts.lambda ?? 0.5 / Math.max(sigma, 1e-6));
}

/** Patch inpainting: code each patch with OMP using only its known pixels (atoms masked and
 *  renormalised), fill the patch from the code, and average the overlapping patches; known pixels
 *  are kept. Patches with fewer than `minKnown` known pixels are skipped; pixels that no usable
 *  patch covers (inside holes larger than a patch) are filled in further passes from the border
 *  inwards, treating the pixels filled so far as known ("onion peeling"). */
export function patchInpaint(img: ArrayLike<number>, known: ArrayLike<number>, w: number, h: number, D: ArrayLike<number>, p: number,
  opts: { stride?: number; K?: number; errPerPixel?: number; minKnown?: number; passes?: number } = {}): Vec {
  const d = p * p, k = D.length / d, K = opts.K ?? 6, e2pp = opts.errPerPixel ?? 0, minKnown = opts.minKnown ?? Math.max(4, Math.floor(d / 8));
  let cur = Float64Array.from(img), kn = Uint8Array.from(known, v => (v ? 1 : 0));
  const Dm = new Float64Array(k * d), nrm = new Float64Array(k);
  for (let pass = 0; pass < (opts.passes ?? 8); pass++) {
    const P = extractPatches(cur, w, h, p, opts.stride ?? 1), M = extractPatches(kn, w, h, p, opts.stride ?? 1);
    const acc = new Float64Array(w * h), cnt = new Float64Array(w * h);
    for (let i = 0; i < P.n; i++) {
      const mk = M.Y.subarray(i * d, (i + 1) * d);
      let nk = 0; for (let t = 0; t < d; t++) nk += mk[t];
      if (nk < minKnown || nk === d) continue;
      for (let j = 0; j < k; j++) {
        let s = 0;
        for (let t = 0; t < d; t++) { const v = mk[t] * D[j * d + t]; Dm[j * d + t] = v; s += v * v; }
        s = Math.sqrt(s); nrm[j] = s;
        if (s > 1e-9) for (let t = 0; t < d; t++) Dm[j * d + t] /= s;
      }
      const y = new Float64Array(d);
      for (let t = 0; t < d; t++) y[t] = mk[t] * P.Y[i * d + t];
      const c = batchOmp(Dm, d, y, 1, { K: Math.min(K, Math.max(1, Math.floor(nk / 2))), errTarget: e2pp * nk });
      const x0 = P.pos[2 * i], y0 = P.pos[2 * i + 1];
      for (let t = 0; t < d; t++) {
        if (mk[t]) continue;
        let v = 0;
        for (let s2 = 0; s2 < c.maxK; s2++) { const j = c.idx[s2]; if (j >= 0 && nrm[j] > 1e-9) v += c.val[s2] / nrm[j] * D[j * d + t]; }
        const q = (y0 + Math.floor(t / p)) * w + x0 + (t % p);
        acc[q] += v; cnt[q]++;
      }
    }
    let missing = 0;
    const next = Float64Array.from(cur), nkn = Uint8Array.from(kn);
    for (let q = 0; q < w * h; q++) {
      if (kn[q]) continue;
      if (cnt[q] > 0) { next[q] = acc[q] / cnt[q]; nkn[q] = 1; } else missing++;
    }
    cur = next;
    if (!missing) break;
    kn = nkn;
  }
  for (let q = 0; q < w * h; q++) if (known[q]) cur[q] = img[q];
  return cur;
}

// ---- same-page cache (each demo bundle has its own copy of this module, so it lives on globalThis)

export interface CachedDict { D: Vec; d: number; p: number; k: number; label: string; source: string }

export function putDictionary(key: string, entry: CachedDict): void {
  const g = globalThis as unknown as { __qbPatchDicts?: Map<string, CachedDict>; dispatchEvent?: (e: Event) => boolean };
  g.__qbPatchDicts ??= new Map();
  g.__qbPatchDicts.set(key, entry);
  if (typeof CustomEvent !== "undefined" && g.dispatchEvent) g.dispatchEvent(new CustomEvent("qb-patchdict", { detail: key }));
}

export function getDictionary(key: string): CachedDict | undefined {
  const g = globalThis as unknown as { __qbPatchDicts?: Map<string, CachedDict> };
  return g.__qbPatchDicts?.get(key);
}
