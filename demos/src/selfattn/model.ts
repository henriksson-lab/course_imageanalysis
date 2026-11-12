// Model of the selfattn demo (no DOM; tested in demos/test/moderndl-test.ts).
//
// A 64 × 64 image is cut into an 8 × 8 grid of 8 × 8-pixel patches: 64 tokens. Each token
// has hand-made content features (brightness, contrast, edge orientation) and a position (its
// grid coordinates). The attention score of token i for token j is
//   s_ij = w_c · (−‖f_i − f_j‖²) + w_p · (−‖x_i − x_j‖²),
// a dot product of extended query and key vectors (see the unit's math slide), and the
// weights are a_ij = softmax_j(s_ij / T). The value of a token is its patch of pixels, so
// the output patch i = Σ_j a_ij · patch_j: one attention layer applied to the image.
// With content features = the raw patch pixels and w_p = 0 this is non-local means.
import { rng } from "../lib/ui";

export const S = 64, P = 8, G = S / P, NT = G * G;

export type Features = "hand" | "pixels";

/** The synthetic test image: a striped texture region, four bright round cells far apart, a long diagonal filament. */
export function synthImage(noise = 4, seed = 3): Float32Array {
  const img = new Float32Array(S * S), r = rng(seed);
  const cells = [[12, 50], [52, 52], [50, 12], [30, 30]];
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    let v = 40;
    // textured region (top left), stripes at 45°
    if (Math.hypot(x - 14, y - 14) < 11) v = 95 + 45 * Math.sin((x + y) * 0.9);
    // cells
    for (const [cx, cy] of cells) { const d = Math.hypot(x - cx, y - cy); if (d < 5) v = 210 - 4 * d; }
    // filament from (2, 40) to (40, 2)... a line x + y = 42, and a second one, x + y = 92
    for (const c of [42, 92]) { const d = Math.abs(x + y - c) / Math.SQRT2; if (d < 1.4 && x > 2 && y > 2 && x < 61 && y < 61) v = Math.max(v, 185); }
    img[y * S + x] = v + noise * r.norm();
  }
  return img;
}

/** Add seeded Gaussian noise. */
export function addNoise(img: Float32Array, sd: number, seed = 9): Float32Array {
  const r = rng(seed);
  return img.map(v => v + sd * r.norm());
}

/** Patch k of an image as a 64-vector (row-major inside the patch). */
export function patch(img: Float32Array, k: number): Float32Array {
  const gx = k % G, gy = (k - gx) / G, out = new Float32Array(P * P);
  for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) out[y * P + x] = img[(gy * P + y) * S + gx * P + x];
  return out;
}

/** Hand-made content features per token: brightness, contrast, edge orientation (cos 2θ, sin 2θ times coherence). Standardised over the tokens. */
export function handFeatures(img: Float32Array): Float64Array[] {
  const F: Float64Array[] = [];
  for (let k = 0; k < NT; k++) {
    const p = patch(img, k);
    let m = 0; for (const v of p) m += v; m /= p.length;
    let s = 0; for (const v of p) s += (v - m) ** 2; s = Math.sqrt(s / p.length);
    let jxx = 0, jyy = 0, jxy = 0;
    for (let y = 0; y < P - 1; y++) for (let x = 0; x < P - 1; x++) {
      const gx = p[y * P + x + 1] - p[y * P + x], gy = p[(y + 1) * P + x] - p[y * P + x];
      jxx += gx * gx; jyy += gy * gy; jxy += gx * gy;
    }
    const tr = jxx + jyy + 1e-9, c2 = (jxx - jyy) / tr, s2 = 2 * jxy / tr;
    F.push(Float64Array.from([m, s, c2, s2]));
  }
  // standardise each feature (orientation features: scale only)
  for (let d = 0; d < 4; d++) {
    let m = 0; for (const f of F) m += f[d]; m /= NT;
    let v = 0; for (const f of F) v += (f[d] - m) ** 2; const sd = Math.sqrt(v / NT) || 1;
    for (const f of F) f[d] = d < 2 ? (f[d] - m) / sd : f[d] / Math.max(sd, 0.3);
  }
  return F;
}

/** Raw patch pixels as features (for non-local means), scaled by 1/(8·σ) so that pure noise differences are about 1 per pixel... */
export function pixelFeatures(img: Float32Array, scale = 40): Float64Array[] {
  return Array.from({ length: NT }, (_, k) => Float64Array.from(patch(img, k), v => v / (scale * P)));
}

export interface AttnOptions { wc: number; wp: number; T: number; features: Features; perm?: Int32Array }

/** Grid position of token k (in patch units). With a permutation, token k sits at grid cell perm[k]. */
export const pos = (k: number): [number, number] => [k % G, Math.floor(k / G)];

/** NT × NT attention weights (row i = the weights of query i; each row sums to 1). */
export function attentionWeights(F: Float64Array[], o: AttnOptions): Float64Array {
  const A = new Float64Array(NT * NT), s = new Float64Array(NT);
  for (let i = 0; i < NT; i++) {
    const [xi, yi] = pos(o.perm ? o.perm[i] : i);
    let mx = -Infinity;
    for (let j = 0; j < NT; j++) {
      let dc = 0; const fi = F[i], fj = F[j];
      for (let d = 0; d < fi.length; d++) dc += (fi[d] - fj[d]) ** 2;
      const [xj, yj] = pos(o.perm ? o.perm[j] : j);
      const dp = (xi - xj) ** 2 + (yi - yj) ** 2;
      s[j] = (-o.wc * dc - o.wp * dp) / o.T;
      if (s[j] > mx) mx = s[j];
    }
    let z = 0;
    for (let j = 0; j < NT; j++) { s[j] = Math.exp(s[j] - mx); z += s[j]; }
    for (let j = 0; j < NT; j++) A[i * NT + j] = s[j] / z;
  }
  return A;
}

/** Output image: patch i (at its grid cell) = Σ_j a_ij · patch_j. Tokens are given by an image laid out as the grid. */
export function attend(img: Float32Array, A: Float64Array): Float32Array {
  const out = new Float32Array(S * S), patches = Array.from({ length: NT }, (_, k) => patch(img, k));
  for (let i = 0; i < NT; i++) {
    const acc = new Float64Array(P * P);
    for (let j = 0; j < NT; j++) { const a = A[i * NT + j]; if (a < 1e-9) continue; const pj = patches[j]; for (let q = 0; q < P * P; q++) acc[q] += a * pj[q]; }
    const gx = i % G, gy = (i - gx) / G;
    for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) out[(gy * P + y) * S + gx * P + x] = acc[y * P + x];
  }
  return out;
}

/** Shuffle the patches of an image: patch k goes to grid cell perm[k]. */
export function shufflePatches(img: Float32Array, perm: Int32Array): Float32Array {
  const out = new Float32Array(S * S);
  for (let k = 0; k < NT; k++) {
    const p = patch(img, k), t = perm[k], gx = t % G, gy = (t - gx) / G;
    for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) out[(gy * P + y) * S + gx * P + x] = p[y * P + x];
  }
  return out;
}

export function permutation(seed = 5): Int32Array {
  const r = rng(seed), p = Int32Array.from({ length: NT }, (_, i) => i);
  for (let i = NT - 1; i > 0; i--) { const j = Math.floor(r.unif() * (i + 1)); const t = p[i]; p[i] = p[j]; p[j] = t; }
  return p;
}

/** Features of an image for the chosen kind. */
export function features(img: Float32Array, kind: Features): Float64Array[] {
  return kind === "hand" ? handFeatures(img) : pixelFeatures(img);
}

/** The worked example: one query, keys and values, scores q·k, softmax(scores / T), output. */
export function tinyAttention(q: number[], K: number[][], v: number[], T = 1): { scores: number[]; weights: number[]; out: number } {
  const scores = K.map(k => k.reduce((s, kv, d) => s + kv * q[d], 0));
  const mx = Math.max(...scores), e = scores.map(s => Math.exp((s - mx) / T)), z = e.reduce((a, b) => a + b, 0);
  const weights = e.map(x => x / z);
  return { scores, weights, out: weights.reduce((s, w, j) => s + w * v[j], 0) };
}
