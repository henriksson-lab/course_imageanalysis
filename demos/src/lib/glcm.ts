// Grey-level co-occurrence matrices, Haralick features and local binary patterns (owner:
// measurements day, unit texture-features/). No DOM.
//
// Conventions:
//  - quantise(v) = floor((v − lo) / (hi − lo) · L), clamped to 0..L−1; levels are 0-based;
//  - offset (dx, dy) in pixels, y down. Angle θ (degrees, counter-clockwise as seen on screen)
//    and distance d in the chessboard sense (Haralick 1973): the direction (cos θ, −sin θ)
//    scaled so its larger component is 1, times d, rounded. 0° = (d, 0), 45° = (d, −d),
//    90° = (0, −d), 135° = (−d, −d); other angles are rounded to the nearest pixel;
//  - glcm counts ordered pairs (level at x, level at x + d) with both pixels inside the image
//    and, if given, inside the mask. symmetric adds the transpose (each pair counted both ways);
//  - entropies in bits (log2), except inside the two information measures of correlation,
//    which follow Haralick (1973) with natural logarithms;
//  - naming traps (flagged in the table): "homogeneity" here is Σ p / (1 + (i − j)²) (=
//    Haralick's inverse difference moment); "inverse difference" is Σ p / (1 + |i − j|);
//    "energy" is √ASM (some tools call ASM itself energy); sum variance uses the sum average
//    (Haralick's paper has the sum entropy there, a known misprint).
//  - LBP: the 8 neighbours clockwise from the top-left, bit = 1 if neighbour ≥ centre; the
//    code reads the bits as a binary number, first (top-left) bit most significant.

type A = ArrayLike<number>;

/** Quantise values to L levels 0..L−1 over [lo, hi]. */
export function quantise(values: A, L: number, lo: number, hi: number): Int32Array {
  const out = new Int32Array(values.length), s = hi > lo ? L / (hi - lo) : 0;
  for (let i = 0; i < values.length; i++) out[i] = Math.max(0, Math.min(L - 1, Math.floor((values[i] - lo) * s)));
  return out;
}

/** Pixel offset for distance d and angle θ in degrees (counter-clockwise on screen, y down). */
export function offsetFor(d: number, angleDeg: number): [number, number] {
  const t = angleDeg * Math.PI / 180, c = Math.cos(t), s = Math.sin(t), m = Math.max(Math.abs(c), Math.abs(s));
  const dx = Math.round(d * c / m), dy = -Math.round(d * s / m);
  return [dx === 0 ? 0 : dx, dy === 0 ? 0 : dy];
}

/** Co-occurrence counts P[i·L + j] for the offset (dx, dy); see the header. */
export function glcm(q: A, w: number, h: number, L: number, dx: number, dy: number,
  opts: { mask?: A; symmetric?: boolean } = {}): Float64Array {
  const P = new Float64Array(L * L), m = opts.mask;
  for (let y = 0; y < h; y++) {
    const y2 = y + dy;
    if (y2 < 0 || y2 >= h) continue;
    for (let x = 0; x < w; x++) {
      const x2 = x + dx;
      if (x2 < 0 || x2 >= w) continue;
      const a = y * w + x, b = y2 * w + x2;
      if (m && (!m[a] || !m[b])) continue;
      P[q[a] * L + q[b]]++;
    }
  }
  if (opts.symmetric) for (let i = 0; i < L; i++) for (let j = i; j < L; j++) {
    const s = P[i * L + j] + P[j * L + i];
    P[i * L + j] = s; P[j * L + i] = s;
  }
  return P;
}

/** Normalise counts to a joint probability (sum 1); all zero stays zero. */
export function normaliseGlcm(P: A): Float64Array {
  let s = 0;
  for (let i = 0; i < P.length; i++) s += P[i];
  return Float64Array.from(P, v => (s ? v / s : 0));
}

export interface Haralick {
  asm: number;
  energy: number;
  contrast: number;
  dissimilarity: number;
  homogeneity: number;
  inverseDifference: number;
  entropy: number;
  correlation: number;
  variance: number;
  sumAverage: number;
  sumVariance: number;
  sumEntropy: number;
  differenceVariance: number;
  differenceEntropy: number;
  imc1: number;
  imc2: number;
}

/** The main texture features as names, for tables (key, label, note). */
export const HARALICK_INFO: { key: keyof Haralick; label: string; note: string }[] = [
  { key: "contrast", label: "contrast", note: "Σ (i−j)² p" },
  { key: "homogeneity", label: "homogeneity", note: "Σ p / (1+(i−j)²); also called inverse difference moment" },
  { key: "asm", label: "ASM", note: "Σ p²; some tools call this energy" },
  { key: "energy", label: "energy", note: "√ASM" },
  { key: "entropy", label: "entropy", note: "−Σ p log₂ p (bits)" },
  { key: "correlation", label: "correlation", note: "Pearson r of a pixel and its offset partner" },
  { key: "dissimilarity", label: "dissimilarity", note: "Σ |i−j| p" },
  { key: "inverseDifference", label: "inverse difference", note: "Σ p / (1+|i−j|)" },
  { key: "variance", label: "variance", note: "Σ (i−μ)² p" },
  { key: "sumAverage", label: "sum average", note: "mean of i+j" },
  { key: "sumVariance", label: "sum variance", note: "variance of i+j" },
  { key: "sumEntropy", label: "sum entropy", note: "entropy of i+j" },
  { key: "differenceVariance", label: "difference variance", note: "variance of |i−j|" },
  { key: "differenceEntropy", label: "difference entropy", note: "entropy of |i−j|" },
  { key: "imc1", label: "info. corr. 1", note: "information measure of correlation 1" },
  { key: "imc2", label: "info. corr. 2", note: "information measure of correlation 2" },
];

const plog2 = (p: number) => (p > 0 ? p * Math.log2(p) : 0);
const plog = (p: number) => (p > 0 ? p * Math.log(p) : 0);

/** Haralick features of a normalised L×L co-occurrence matrix p. Correlation is NaN if σ = 0. */
export function haralick(p: A, L: number): Haralick {
  let asm = 0, contrast = 0, diss = 0, hom = 0, invd = 0, ent = 0, mi = 0, mj = 0;
  const px = new Float64Array(L), py = new Float64Array(L), psum = new Float64Array(2 * L - 1), pdiff = new Float64Array(L);
  for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) {
    const v = p[i * L + j];
    if (!v) continue;
    const d = i - j;
    asm += v * v; contrast += d * d * v; diss += Math.abs(d) * v;
    hom += v / (1 + d * d); invd += v / (1 + Math.abs(d)); ent -= plog2(v);
    mi += i * v; mj += j * v;
    px[i] += v; py[j] += v; psum[i + j] += v; pdiff[Math.abs(d)] += v;
  }
  let vi = 0, vj = 0, cov = 0;
  for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) {
    const v = p[i * L + j];
    if (!v) continue;
    vi += (i - mi) ** 2 * v; vj += (j - mj) ** 2 * v; cov += (i - mi) * (j - mj) * v;
  }
  const corr = vi > 1e-15 && vj > 1e-15 ? cov / Math.sqrt(vi * vj) : NaN;
  let sa = 0, se = 0, sv = 0;
  for (let k = 0; k < psum.length; k++) { sa += k * psum[k]; se -= plog2(psum[k]); }
  for (let k = 0; k < psum.length; k++) sv += (k - sa) ** 2 * psum[k];
  let dm = 0, dv = 0, de = 0;
  for (let k = 0; k < L; k++) { dm += k * pdiff[k]; de -= plog2(pdiff[k]); }
  for (let k = 0; k < L; k++) dv += (k - dm) ** 2 * pdiff[k];
  // information measures of correlation (natural logs)
  let hxy = 0, hxy1 = 0, hxy2 = 0, hx = 0, hy = 0;
  for (let i = 0; i < L; i++) { hx -= plog(px[i]); hy -= plog(py[i]); }
  for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) {
    const v = p[i * L + j], q = px[i] * py[j];
    hxy -= plog(v);
    if (q > 0) { if (v > 0) hxy1 -= v * Math.log(q); hxy2 -= q * Math.log(q); }
  }
  const mh = Math.max(hx, hy);
  return {
    asm, energy: Math.sqrt(asm), contrast, dissimilarity: diss, homogeneity: hom, inverseDifference: invd,
    entropy: ent, correlation: corr, variance: vi, sumAverage: sa, sumVariance: sv, sumEntropy: se,
    differenceVariance: dv, differenceEntropy: de,
    imc1: mh > 0 ? (hxy - hxy1) / mh : NaN,
    imc2: Math.sqrt(Math.max(0, 1 - Math.exp(-2 * (hxy2 - hxy)))),
  };
}

/** Quantise, count, normalise and compute features in one call. */
export function textureFeatures(img: A, w: number, h: number, o: { L: number; lo: number; hi: number; d: number; angle: number; mask?: A; symmetric?: boolean }):
  { q: Int32Array; P: Float64Array; p: Float64Array; f: Haralick; pairs: number; offset: [number, number] } {
  const q = quantise(img, o.L, o.lo, o.hi);
  const off = offsetFor(o.d, o.angle);
  const P = glcm(q, w, h, o.L, off[0], off[1], { mask: o.mask, symmetric: o.symmetric });
  let pairs = 0;
  for (const v of P) pairs += v;
  const p = normaliseGlcm(P);
  return { q, P, p, f: haralick(p, o.L), pairs, offset: off };
}

// ---------------------------------------------------------------------------------------
// Local binary patterns

/** Neighbour offsets clockwise from the top-left. */
export const LBP_OFFSETS: [number, number][] = [[-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0]];

/** LBP code of a 3×3 neighbourhood given as 9 values in raster order (centre = index 4). */
export function lbp3x3(v: A): number {
  const c = v[4];
  let code = 0;
  for (const [dx, dy] of LBP_OFFSETS) code = (code << 1) | (v[(dy + 1) * 3 + dx + 1] >= c ? 1 : 0);
  return code;
}

/** Rotate an 8-bit code by k positions (circular). */
export const rotateCode = (code: number, k: number): number => ((code << k) | (code >> (8 - k))) & 255;

/** Rotation-invariant code: the smallest of the 8 circular rotations. */
export function lbpRotInv(code: number): number {
  let m = code;
  for (let k = 1; k < 8; k++) m = Math.min(m, rotateCode(code, k));
  return m;
}

/** Number of 0/1 transitions around the circle (uniform patterns have ≤ 2). */
export function lbpTransitions(code: number): number {
  let t = 0;
  for (let k = 0; k < 8; k++) if (((code >> k) & 1) !== ((code >> ((k + 1) % 8)) & 1)) t++;
  return t;
}

/** LBP code of every interior pixel (border pixels −1). */
export function lbpImage(img: A, w: number, h: number): Int32Array {
  const out = new Int32Array(w * h).fill(-1), nb = new Float64Array(9);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    for (let k = 0; k < 9; k++) nb[k] = img[(y + Math.floor(k / 3) - 1) * w + x + (k % 3) - 1];
    out[y * w + x] = lbp3x3(nb);
  }
  return out;
}
