// Model of the genrestore demo (no DOM; tested in demos/test/moderndl-test.ts).
//
// 32 × 32 synthetic cells: a nucleus with 0–2 bright nucleoli, sometimes a protrusion, or a
// division (two nuclei). The PRIOR is a Gaussian mixture with one small isotropic component
// (sd s) per training image, so the posterior of a measurement is exact (lib/mixture.ts):
//   blur:      y = A x + n with A a Gaussian blur with mirrored borders, which is diagonal in
//              the orthonormal DCT basis (multipliers a_k = exp(−σ²|ω_k|²/2), ω = πk/32);
//   inpaint:   y = x + n on the measured pixels only (A diagonal in the pixel basis).
// Restorations: classical (Wiener with the mean power spectrum of the training images; for
// holes, smooth harmonic filling), the posterior mean, posterior samples, their disagreement.
// Measurements on a restoration: nucleus area, number of nuclei, number of nucleoli, PSNR.
import { dct2, idct2 } from "../lib/dct";
import { makeDiag, posteriorDiag, posteriorMean, posteriorSample, type DiagMixture, type DiagPosterior } from "../lib/mixture";
import { label2d } from "../lib/label";
import { rng } from "../lib/ui";

export const N = 32, D = N * N;
export const BG = 0.08, NUC = 0.5, NUCLEOLUS = 0.9;

export type CellKind = "plain" | "protrusion" | "dividing" | "three" | "ring";
export interface CellSpec { kind: CellKind; nucleoli: number; seed: number }

/** Nucleus templates: centre x, y, semi-axes a, b, angle. */
const TEMPLATES: number[][] = [[16, 16, 7.5, 6.4, 0.3], [16.5, 15.5, 7.8, 6.2, 1.2], [15.5, 16, 7.2, 6.8, 2.0], [16, 16.5, 8.0, 6.0, 2.7], [15.8, 15.6, 7.4, 6.5, 0.8], [16.2, 16.2, 7.6, 6.3, 1.7]];

/** Render one cell (values about 0..1), seeded. */
export function cell(spec: CellSpec): Float64Array {
  const r = rng(spec.seed), img = new Float64Array(D).fill(BG);
  const nuc: { x: number; y: number; a: number; b: number; t: number }[] = [];
  // the nucleus outline is one of six templates (the prior knows the outlines well; what the
  // measurement cannot show is inside: nucleoli, and small additions such as a protrusion)
  const tp = TEMPLATES[Math.floor(r.unif() * TEMPLATES.length)];
  const cx = tp[0], cy = tp[1];
  if (spec.kind === "dividing") {
    const ang = tp[4], d = 6.2 + 0.5 * r.unif();
    for (const sgn of [-1, 1]) nuc.push({ x: cx + sgn * d * Math.cos(ang), y: cy + sgn * d * Math.sin(ang), a: 4.6 + 0.6 * r.unif(), b: 4.2 + 0.5 * r.unif(), t: ang });
  } else nuc.push({ x: cx, y: cy, a: tp[2], b: tp[3], t: tp[4] });
  const inside = (e: typeof nuc[number], x: number, y: number, grow = 0) => {
    const c = Math.cos(e.t), s = Math.sin(e.t), u = c * (x - e.x) + s * (y - e.y), v = -s * (x - e.x) + c * (y - e.y);
    return (u / (e.a + grow)) ** 2 + (v / (e.b + grow)) ** 2 < 1;
  };
  // protrusion: a small blob on the nucleus boundary
  let prot: { x: number; y: number; r: number } | null = null;
  if (spec.kind === "protrusion") {
    const e = nuc[0], ang = 2 * Math.PI * r.unif();
    prot = { x: e.x + (e.a + 1.2) * Math.cos(ang), y: e.y + (e.b + 1.2) * Math.sin(ang), r: 2.6 + 0.6 * r.unif() };
  }
  // nucleoli positions inside the (first) nucleus
  const nn = spec.kind === "three" ? 3 : spec.kind === "dividing" ? 0 : spec.nucleoli;
  const spots: [number, number][] = [];
  for (let k = 0, tries = 0; k < nn && tries < 200; tries++) {
    const e = nuc[0], u = (2 * r.unif() - 1) * 0.55 * e.a, v = (2 * r.unif() - 1) * 0.55 * e.b;
    const c = Math.cos(e.t), s = Math.sin(e.t), x = e.x + c * u - s * v, y = e.y + s * u + c * v;
    if (spots.every(([px, py]) => Math.hypot(px - x, py - y) > 4.2)) spots.push([x, y]);
    else continue;
    k++;
  }
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let v = BG;
    for (const e of nuc) {
      if (inside(e, x, y)) {
        v = NUC;
        if (spec.kind === "ring" && inside(e, x, y, -3.2)) v = BG + 0.12;
      }
    }
    if (prot && Math.hypot(x - prot.x, y - prot.y) < prot.r) v = NUC;
    for (const [sx, sy] of spots) if (Math.hypot(x - sx, y - sy) < 1.7) v = NUCLEOLUS;
    img[y * N + x] = v;
  }
  // a little optical blur (3 × 3 binomial)
  const out = new Float64Array(D);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let s = 0, w = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const xx = Math.min(N - 1, Math.max(0, x + dx)), yy = Math.min(N - 1, Math.max(0, y + dy)), k = (2 - Math.abs(dx)) * (2 - Math.abs(dy));
      s += k * img[yy * N + xx]; w += k;
    }
    out[y * N + x] = s / w;
  }
  return out;
}

/** The training set: 300 cells; kinds in fixed proportions. */
export function trainingSet(): CellSpec[] {
  const out: CellSpec[] = [];
  for (let i = 0; i < 300; i++) {
    const kind: CellKind = i % 20 < 3 ? "protrusion" : i % 20 < 6 ? "dividing" : "plain";
    out.push({ kind, nucleoli: i % 3, seed: 1000 + i });
  }
  return out;
}

export type PriorId = "matched" | "nodiv" | "noprot";

export interface Prior { specs: CellSpec[]; images: Float64Array[]; pix: DiagMixture; dct: DiagMixture; power: Float64Array }

/** The prior: one component per training image (sd s), in the pixel basis and in the DCT basis. */
export function makePrior(id: PriorId, s = 0.02): Prior {
  const specs = trainingSet().filter(c => (id === "nodiv" ? c.kind !== "dividing" : id === "noprot" ? c.kind !== "protrusion" : true));
  const images = specs.map(cell);
  const w = new Array(images.length).fill(1);
  const pix = makeDiag(w, images, s * s);
  const C = images.map(im => dct2(im, N, N));
  const dct = makeDiag(w, C, s * s);
  const power = new Float64Array(D);
  for (const c of C) for (let k = 0; k < D; k++) power[k] += c[k] * c[k] / C.length;
  return { specs, images, pix, dct, power };
}

/** DCT multipliers of a Gaussian blur of width σ (mirrored borders). */
export function blurMultipliers(sigma: number): Float64Array {
  const a = new Float64Array(D);
  for (let ky = 0; ky < N; ky++) for (let kx = 0; kx < N; kx++) {
    const wx = Math.PI * kx / N, wy = Math.PI * ky / N;
    a[ky * N + kx] = Math.exp(-sigma * sigma * (wx * wx + wy * wy) / 2);
  }
  return a;
}

export type Mode = "blur" | "inpaint";

export interface Measurement { mode: Mode; a: Float64Array; y: Float64Array; noise: number; shown: Float64Array; mask?: Uint8Array }

/** Measure x: blur + noise (in the DCT basis), or the pixels outside the hole + noise. */
export function measure(x: Float64Array, mode: Mode, sigma: number, noise: number, hole: Uint8Array, seed = 77): Measurement {
  const r = rng(seed);
  if (mode === "blur") {
    const a = blurMultipliers(sigma), c = dct2(x, N, N);
    const y = c.map((v, k) => a[k] * v + noise * r.norm());
    return { mode, a, y, noise, shown: idct2(y, N, N) };
  }
  const a = Float64Array.from(hole, h => (h ? 0 : 1));
  const y = x.map((v, i) => (hole[i] ? 0 : v + noise * r.norm()));
  return { mode, a, y, noise, shown: y.map((v, i) => (hole[i] ? NaN : v)), mask: hole };
}

export interface Restoration { post: DiagPosterior; mean: Float64Array; samples: Float64Array[]; sd: Float64Array; classical: Float64Array }

export function restore(prior: Prior, m: Measurement, nSamples = 32, seed = 5): Restoration {
  const blur = m.mode === "blur", mix = blur ? prior.dct : prior.pix;
  const post = posteriorDiag(mix, m.y, m.a, Math.max(1e-8, m.noise * m.noise));
  const back = (v: Float64Array) => (blur ? idct2(v, N, N) : v);
  const mean = back(posteriorMean(post));
  const r = rng(seed), samples: Float64Array[] = [];
  for (let k = 0; k < nSamples; k++) samples.push(back(posteriorSample(post, r).x));
  const sd = new Float64Array(D);
  for (let i = 0; i < D; i++) {
    let s = 0, s2 = 0; for (const x of samples) { s += x[i]; s2 += x[i] * x[i]; }
    sd[i] = Math.sqrt(Math.max(0, s2 / nSamples - (s / nSamples) ** 2));
  }
  let classical: Float64Array;
  if (blur) {
    const n2 = m.noise * m.noise;
    classical = idct2(m.y.map((v, k) => (m.a[k] * prior.power[k] * v) / (m.a[k] * m.a[k] * prior.power[k] + n2 + 1e-12)), N, N);
  } else classical = harmonicFill(m.y, m.mask!);
  return { post, mean, samples, sd, classical };
}

/** Fill the hole smoothly: Jacobi iterations of the Laplace equation with the measured pixels fixed. */
export function harmonicFill(y: Float64Array, hole: Uint8Array, iters = 400): Float64Array {
  let x = Float64Array.from(y);
  let s = 0, c = 0; for (let i = 0; i < D; i++) if (!hole[i]) { s += y[i]; c++; }
  for (let i = 0; i < D; i++) if (hole[i]) x[i] = c ? s / c : 0;
  for (let it = 0; it < iters; it++) {
    const z = Float64Array.from(x);
    for (let yy = 0; yy < N; yy++) for (let xx = 0; xx < N; xx++) {
      const p = yy * N + xx;
      if (!hole[p]) continue;
      let t = 0, n = 0;
      if (xx > 0) { t += x[p - 1]; n++; } if (xx < N - 1) { t += x[p + 1]; n++; }
      if (yy > 0) { t += x[p - N]; n++; } if (yy < N - 1) { t += x[p + N]; n++; }
      z[p] = t / n;
    }
    x = z;
  }
  return x;
}

export interface CellMeasures { area: number; nuclei: number; nucleoli: number }

/** Nucleus = pixels > 0.3 (components of ≥ 6 pixels); nucleoli = pixels > 0.7 inside, components of ≥ 2 pixels. */
export function measureCell(x: Float64Array): CellMeasures {
  const nm = Uint8Array.from(x, v => (v > 0.3 ? 1 : 0));
  const L = label2d(nm, N, N, 4), size = new Int32Array(L.count + 1);
  for (const l of L.data) size[l]++;
  let nuclei = 0, area = 0;
  for (let k = 1; k <= L.count; k++) if (size[k] >= 6) { nuclei++; area += size[k]; }
  const om = Uint8Array.from(x, v => (v > 0.7 ? 1 : 0));
  const O = label2d(om, N, N, 4), os = new Int32Array(O.count + 1);
  for (const l of O.data) os[l]++;
  let nucleoli = 0; for (let k = 1; k <= O.count; k++) if (os[k] >= 2) nucleoli++;
  return { area, nuclei, nucleoli };
}

/** PSNR in dB with peak 1. */
export function psnr(a: Float64Array, b: Float64Array): number {
  let s = 0; for (let i = 0; i < D; i++) s += (a[i] - b[i]) ** 2;
  return 10 * Math.log10(1 / (s / D));
}
