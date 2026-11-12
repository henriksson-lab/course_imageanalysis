// Model of the leakage demo (no DOM). Twelve 48 × 48 synthetic images from 4 "days" (3 per
// day); pixel task: nucleus or background, with dim nuclei as the hard class and a per-day
// change of brightness, contrast and noise (scaled by `dayEffect`). Features per pixel: a
// small stack of Gaussian-smoothed intensities (σ = 1, 2, 4) and the gradient magnitude.
// Classifier: k nearest neighbours on a random subsample of training pixels, which can only
// remember, so it shows leakage clearly. Five ways to split into 4 folds: random pixels,
// random 8 × 8 blocks, blocks with a buffer, by image, by day. Optional "augment then split":
// every image is rotated by 90°, 180°, 270° before the split, and each copy counts as an image.
// "Nothing to learn": the labels are random smooth blobs unrelated to the image.
import { Grey, createGrey, gaussianBlur, gradient } from "../lib/imageops";
import { nuclei } from "../lib/synth";
import { rng, Rng, gauss, shuffle } from "../lib/random";

export const S = 48, DAYS = 4, PER_DAY = 3, BLOCK = 8, NF = 4;
export const NTRAIN = 2000, NTEST = 500;

export interface Settings {
  dayEffect: number;     // 0 = all days alike, 1 = typical, 2 = strong
  nothing: boolean;      // labels unrelated to the image
  augment: boolean;      // rotated copies made before the split
  k: number;             // neighbours
  buffer: number;        // buffer width (px) around test blocks
}

export const DEFAULTS: Settings = { dayEffect: 1, nothing: false, augment: false, k: 5, buffer: 6 };

export type Strategy = "pixels" | "blocks" | "buffer" | "image" | "day";
export const STRATEGIES: { id: Strategy; label: string }[] = [
  { id: "pixels", label: "random pixels" },
  { id: "blocks", label: "random blocks" },
  { id: "buffer", label: "blocks + buffer" },
  { id: "image", label: "by image" },
  { id: "day", label: "by day" },
];

export interface DataSet {
  n: number;                  // number of images (12, or 48 with augmentation)
  images: Grey[];             // for display (originals and rotated copies)
  labels: Uint8Array[];
  feats: Float32Array[];      // per image: NF values per pixel, pixel-major
  day: Int32Array;
  source: Int32Array;         // index of the original image (0..11)
}

function rot90(a: ArrayLike<number>, times: number, stride = 1): Float32Array {
  // rotate an S × S image (with `stride` values per pixel) by 90° counter-clockwise `times` times
  let cur = Float32Array.from(a);
  for (let t = 0; t < times; t++) {
    const out = new Float32Array(cur.length);
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++)
      for (let c = 0; c < stride; c++) out[((S - 1 - x) * S + y) * stride + c] = cur[(y * S + x) * stride + c];
    cur = out;
  }
  return cur;
}

/** The 12 base images with their day parameters applied. */
export function makeSet(s: Settings): DataSet {
  const dr = rng(5);
  const days = Array.from({ length: DAYS }, () => ({ off: gauss(dr, 0, 1), con: gauss(dr, 0, 1), noi: dr.unif() }));
  // make the day effect visible even with 4 days: spread the offsets
  [-1.2, 0.9, -0.3, 1.4].forEach((v, d) => (days[d].off = v));
  const images: Grey[] = [], labels: Uint8Array[] = [], feats: Float32Array[] = [], day: number[] = [], source: number[] = [];
  const blobRng = rng(77);
  for (let i = 0; i < DAYS * PER_DAY; i++) {
    const d = Math.floor(i / PER_DAY), r = rng(100 + i);
    const nu = nuclei({ width: S, height: S, count: 5, radius: [4, 7], touching: 0.2, foreground: 150, background: 60, texture: 12, noise: 0, blur: 1, seed: 300 + i });
    // dim nuclei: about one in three
    const dim = nu.ellipses.map(() => r.unif() < 0.35);
    const img = createGrey(S, S);
    const de = s.dayEffect, P = days[d];
    const contrast = Math.exp(0.35 * de * P.con), offset = 28 * de * P.off, noise = 10 * (1 + de * P.noi);
    for (let p = 0; p < S * S; p++) {
      const l = nu.labels[p];
      let v = nu.image.data[p];
      if (l && dim[l - 1]) v = 60 + 0.3 * (v - 60);
      img.data[p] = 60 + contrast * (v - 60) + offset + noise * r.norm();
    }
    let lab: Uint8Array;
    if (s.nothing) {
      const z = createGrey(S, S);
      for (let p = 0; p < S * S; p++) z.data[p] = blobRng.norm();
      const b = gaussianBlur(z, 3);
      const med = Float32Array.from(b.data).sort()[S * S >> 1];
      lab = Uint8Array.from(b.data, v => (v > med ? 1 : 0));
    } else lab = Uint8Array.from(nu.labels, v => (v ? 1 : 0));
    const f = features(img);
    const copies = s.augment ? 4 : 1;
    for (let c = 0; c < copies; c++) {
      images.push({ width: S, height: S, data: rot90(img.data, c) });
      labels.push(Uint8Array.from(rot90(lab, c)));
      feats.push(rot90(f, c, NF));
      day.push(d); source.push(i);
    }
  }
  return { n: images.length, images, labels, feats, day: Int32Array.from(day), source: Int32Array.from(source) };
}

/** Feature stack: Gaussian σ = 1, 2, 4 and gradient magnitude σ = 1, scaled to similar ranges. */
export function features(img: Grey): Float32Array {
  const g1 = gaussianBlur(img, 1), g2 = gaussianBlur(img, 2), g4 = gaussianBlur(img, 4), gm = gradient(img, 1).mag;
  const out = new Float32Array(S * S * NF);
  for (let p = 0; p < S * S; p++) {
    out[p * NF] = g1.data[p] / 40; out[p * NF + 1] = g2.data[p] / 40; out[p * NF + 2] = g4.data[p] / 40; out[p * NF + 3] = gm.data[p] / 15;
  }
  return out;
}

/** Fold of every pixel (−1 = excluded) for test fold `f`: 0 = train, 1 = test, 2 = buffer (unused). */
export interface Split { fold: Int8Array[]; nFolds: number }

/** Assign every pixel of every image to one of 4 folds. */
export function assignFolds(set: DataSet, strat: Strategy, seed = 1): Int8Array[] {
  const r = rng(seed), out: Int8Array[] = [];
  const nb = S / BLOCK;
  if (strat === "pixels") for (let i = 0; i < set.n; i++) out.push(Int8Array.from({ length: S * S }, () => Math.floor(4 * r.unif())));
  else if (strat === "blocks" || strat === "buffer") {
    for (let i = 0; i < set.n; i++) {
      const bf = Array.from({ length: nb * nb }, () => Math.floor(4 * r.unif()));
      out.push(Int8Array.from({ length: S * S }, (_, p) => bf[Math.floor((p / S) / BLOCK) * nb + Math.floor((p % S) / BLOCK)]));
    }
  } else if (strat === "image") {
    const order = shuffle(r, Array.from({ length: set.n }, (_, i) => i));
    const fi = new Int32Array(set.n);
    order.forEach((img, k) => (fi[img] = k % 4));
    for (let i = 0; i < set.n; i++) out.push(new Int8Array(S * S).fill(fi[i]));
  } else for (let i = 0; i < set.n; i++) out.push(new Int8Array(S * S).fill(set.day[i]));
  return out;
}

/** Role of each pixel when fold f is the test fold: 0 train, 1 test, 2 excluded (buffer). */
export function roles(set: DataSet, folds: Int8Array[], f: number, strat: Strategy, buffer: number): Uint8Array[] {
  return folds.map(fo => {
    const role = Uint8Array.from(fo, v => (v === f ? 1 : 0));
    if (strat === "buffer" && buffer > 0) {
      // exclude training pixels within `buffer` (chessboard distance) of a test pixel
      const near = new Uint8Array(S * S);
      for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
        if (role[y * S + x] !== 1) continue;
        // only block edges need checking, but the image is small
        for (let dy = -buffer; dy <= buffer; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= S) continue;
          for (let dx = -buffer; dx <= buffer; dx++) { const xx = x + dx; if (xx >= 0 && xx < S) near[yy * S + xx] = 1; }
        }
      }
      for (let p = 0; p < S * S; p++) if (role[p] === 0 && near[p]) role[p] = 2;
    }
    return role;
  });
}

function sample(r: Rng, pool: number[], n: number): number[] {
  if (pool.length <= n) return pool;
  const a = pool.slice();
  for (let i = 0; i < n; i++) { const j = i + Math.floor(r.unif() * (a.length - i)); const t = a[i]; a[i] = a[j]; a[j] = t; }
  return a.slice(0, n);
}

/** k-NN classifier on (image, pixel) ids. */
export interface Knn { X: Float32Array; y: Uint8Array; n: number; k: number }

export function trainKnn(set: DataSet, ids: number[], k: number): Knn {
  const X = new Float32Array(ids.length * NF), y = new Uint8Array(ids.length);
  ids.forEach((id, j) => {
    const i = Math.floor(id / (S * S)), p = id % (S * S);
    for (let c = 0; c < NF; c++) X[j * NF + c] = set.feats[i][p * NF + c];
    y[j] = set.labels[i][p];
  });
  return { X, y, n: ids.length, k };
}

/** Predict one feature vector (votes of the k nearest training pixels). */
export function predictKnn(m: Knn, f: ArrayLike<number>, o = 0): number {
  const k = m.k, bd = new Float64Array(k).fill(Infinity), bl = new Uint8Array(k);
  const f0 = f[o], f1 = f[o + 1], f2 = f[o + 2], f3 = f[o + 3], X = m.X;
  for (let j = 0; j < m.n; j++) {
    const a = X[j * 4] - f0, b = X[j * 4 + 1] - f1, c = X[j * 4 + 2] - f2, e = X[j * 4 + 3] - f3;
    const d = a * a + b * b + c * c + e * e;
    if (d >= bd[k - 1]) continue;
    let q = k - 1;
    while (q > 0 && bd[q - 1] > d) { bd[q] = bd[q - 1]; bl[q] = bl[q - 1]; q--; }
    bd[q] = d; bl[q] = m.y[j];
  }
  let s = 0;
  for (let q = 0; q < k; q++) s += bl[q];
  return 2 * s > k ? 1 : 0;
}

export interface StrategyResult { strategy: Strategy; label: string; folds: number[]; mean: number; min: number; max: number }

/** Accuracy of each fold for one strategy. */
export function evaluate(set: DataSet, strat: Strategy, s: Settings, seed = 1): number[] {
  const folds = assignFolds(set, strat, seed), r = rng(seed + 11), acc: number[] = [];
  for (let f = 0; f < 4; f++) {
    const ro = roles(set, folds, f, strat, s.buffer);
    const train: number[] = [], test: number[] = [];
    ro.forEach((role, i) => { for (let p = 0; p < S * S; p++) { if (role[p] === 0) train.push(i * S * S + p); else if (role[p] === 1) test.push(i * S * S + p); } });
    // the test set always consists of original images only (copies are training material)
    const testO = test.filter(id => set.n === DAYS * PER_DAY || Math.floor(id / (S * S)) % 4 === 0);
    const m = trainKnn(set, sample(r, train, NTRAIN), s.k);
    const ts = sample(r, testO.length ? testO : test, NTEST);
    let ok = 0;
    for (const id of ts) { const i = Math.floor(id / (S * S)), p = id % (S * S); if (predictKnn(m, set.feats[i], p * NF) === set.labels[i][p]) ok++; }
    acc.push(ts.length ? ok / ts.length : NaN);
  }
  return acc;
}

export function evaluateAll(set: DataSet, s: Settings, seed = 1): StrategyResult[] {
  return STRATEGIES.map(st => {
    const a = evaluate(set, st.id, s, seed);
    return { strategy: st.id, label: st.label, folds: a, mean: a.reduce((x, y) => x + y, 0) / a.length, min: Math.min(...a), max: Math.max(...a) };
  });
}

/** Prediction of a whole image when it is in the test fold, with roles for display. */
export function predictImage(set: DataSet, strat: Strategy, s: Settings, img: number, seed = 1): { pred: Uint8Array; roles: Uint8Array[]; acc: number } {
  const folds = assignFolds(set, strat, seed), r = rng(seed + 23);
  // the fold that holds most of this image's pixels
  const cnt = [0, 0, 0, 0];
  folds[img].forEach(v => cnt[v]++);
  const f = cnt.indexOf(Math.max(...cnt));
  const ro = roles(set, folds, f, strat, s.buffer);
  const train: number[] = [];
  ro.forEach((role, i) => { for (let p = 0; p < S * S; p++) if (role[p] === 0) train.push(i * S * S + p); });
  const m = trainKnn(set, sample(r, train, NTRAIN), s.k);
  const pred = new Uint8Array(S * S);
  let ok = 0, n = 0;
  for (let p = 0; p < S * S; p++) {
    pred[p] = predictKnn(m, set.feats[img], p * NF);
    if (ro[img][p] === 1) { n++; if (pred[p] === set.labels[img][p]) ok++; }
  }
  return { pred, roles: ro, acc: n ? ok / n : NaN };
}

export interface Preset { id: string; label: string; settings: Partial<Settings> }

export const PRESETS: Preset[] = [
  { id: "default", label: "random pixels look great", settings: {} },
  { id: "nothing", label: "nothing to learn", settings: { nothing: true } },
  { id: "buffer", label: "blocks need a buffer", settings: { buffer: 8 } },
  { id: "newday", label: "a new day: domain shift", settings: { dayEffect: 2 } },
  { id: "noday", label: "no day effect", settings: { dayEffect: 0 } },
  { id: "augment", label: "augment then split", settings: { augment: true } },
];
