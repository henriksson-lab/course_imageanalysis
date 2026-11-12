// Model of the leakage demo (no DOM). Twelve 48 × 48 synthetic images from 4 "days" (3 per
// day); pixel task: nucleus or background, with dim nuclei as the hard class and a per-day
// change of brightness, contrast and noise (scaled by `dayEffect`); a smooth background
// texture as in real cells. Features per pixel: a feature stack (Gaussian σ = 1, 2, 4, 8;
// gradient magnitude σ = 1, 2, 4; Laplacian of Gaussian σ = 2, 4), standardised with fixed
// constants. Classifier: k nearest neighbours over all training pixels (exact, with a k-d
// tree), which can only remember, so it shows leakage clearly. Five ways to split into 4 folds: random pixels,
// random 8 × 8 blocks, blocks with a buffer, by image, by day. Optional "augment then split":
// every image is rotated by 90°, 180°, 270° before the split, and each copy counts as an image.
// "Nothing to learn": the labels are random smooth blobs unrelated to the image.
import { Grey, createGrey, gaussianBlur, gradient, logFilter } from "../lib/imageops";
import { nuclei } from "../lib/synth";
import { rng, Rng, shuffle } from "../lib/random";

export const S = 48, DAYS = 4, PER_DAY = 3, BLOCK = 8, NF = 9;
export const NTEST = 200;

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
  tree?: Knn;                 // k-d tree over every pixel of every image (built once)
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

// Tuning knobs (exported for the tests and for calibrating the constants below).
export const P = { texture: 10, noise: 6, blob: 4, offset: 50, contrast: 0.5 };

/** The 12 base images with their day parameters applied (and rotated copies if augmenting). */
export function makeSet(s: Settings): DataSet {
  // day d: brightness offset, contrast (log factor) and extra noise, fixed per day
  const days = [{ off: -1.2, con: 0.6, noi: 0.2 }, { off: 0.9, con: -0.8, noi: 0.9 }, { off: -0.3, con: -0.2, noi: 0.5 }, { off: 1.4, con: 0.9, noi: 0.1 }];
  const images: Grey[] = [], labels: Uint8Array[] = [], feats: Float32Array[] = [], day: number[] = [], source: number[] = [];
  const blobRng = rng(77);
  for (let i = 0; i < DAYS * PER_DAY; i++) {
    const d = Math.floor(i / PER_DAY), r = rng(100 + i);
    const nu = nuclei({ width: S, height: S, count: 5, radius: [4, 7], touching: 0.2, foreground: 150, background: 60, texture: 12, noise: 0, blur: 1, seed: 300 + i });
    const dim = nu.ellipses.map(() => r.unif() < 0.35);       // dim nuclei: the hard class
    // smooth background texture (cytoplasm, autofluorescence) at three scales
    const tex = createGrey(S, S);
    for (const sg of [2, 4, 8]) {
      const z = createGrey(S, S);
      for (let p = 0; p < S * S; p++) z.data[p] = r.norm();
      const b = gaussianBlur(z, sg);
      for (let p = 0; p < S * S; p++) tex.data[p] += P.texture * 3.5 * sg * b.data[p] / 3;
    }
    const de = s.dayEffect, Q = days[d];
    const contrast = Math.exp(P.contrast * de * Q.con), offset = P.offset * de * Q.off, noise = P.noise * (1 + de * Q.noi);
    const img = createGrey(S, S);
    for (let p = 0; p < S * S; p++) {
      const l = nu.labels[p];
      let v = nu.image.data[p];
      if (l && dim[l - 1]) v = 60 + 0.3 * (v - 60);
      img.data[p] = 60 + contrast * (v + tex.data[p] - 60) + offset + noise * r.norm();
    }
    let lab: Uint8Array;
    if (s.nothing) {
      const z = createGrey(S, S);
      for (let p = 0; p < S * S; p++) z.data[p] = blobRng.norm();
      const b = gaussianBlur(z, P.blob);
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

// Standardisation constants (mean, SD of each feature over a reference set made once with
// the default settings): fixed numbers, so no statistics of the test images enter training.
export const FEAT_MEAN = [72.8, 72.8, 72.6, 71.4, 5.9, 4.55, 2.72, -0.08, -0.69];
export const FEAT_SD = [25.6, 22.2, 16.0, 9.73, 6.46, 4.37, 1.89, 10.1, 14.6];

/** Feature stack: Gaussian σ = 1, 2, 4, 8; gradient magnitude σ = 1, 2, 4; LoG σ = 2, 4. */
export function features(img: Grey): Float32Array {
  const maps = [1, 2, 4, 8].map(sg => gaussianBlur(img, sg)).concat([1, 2, 4].map(sg => gradient(img, sg).mag), [2, 4].map(sg => logFilter(img, sg, true)));
  const out = new Float32Array(S * S * NF);
  for (let p = 0; p < S * S; p++) for (let c = 0; c < NF; c++) out[p * NF + c] = (maps[c].data[p] - FEAT_MEAN[c]) / FEAT_SD[c];
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

/** k-NN classifier: training vectors in a k-d tree (exact search). */
export interface Knn { X: Float32Array; y: Uint8Array; n: number; idx: Int32Array; split: Int8Array }

/**
 * k-d tree over all pixels of all images (point j = image · S² + pixel). Training is then
 * just a mask: a query skips the points that are not allowed (test or buffer pixels).
 */
export function buildTree(set: DataSet): Knn {
  if (set.tree) return set.tree;
  const n = set.n * S * S, X = new Float32Array(n * NF), y = new Uint8Array(n);
  for (let i = 0; i < set.n; i++) { X.set(set.feats[i], i * S * S * NF); y.set(set.labels[i], i * S * S); }
  // implicit tree: node = range [lo, hi) of idx, median element at mid, split dimension stored at mid
  const idx = Int32Array.from({ length: n }, (_, j) => j), split = new Int8Array(n);
  const build = (lo: number, hi: number, depth: number) => {
    if (hi - lo <= 1) { if (hi > lo) split[lo] = -1; return; }
    // dimension of largest spread
    let bd = 0, bs = -1;
    for (let c = 0; c < NF; c++) {
      let mn = Infinity, mx = -Infinity;
      for (let q = lo; q < hi; q += Math.max(1, (hi - lo) >> 6)) { const v = X[idx[q] * NF + c]; if (v < mn) mn = v; if (v > mx) mx = v; }
      if (mx - mn > bs) { bs = mx - mn; bd = c; }
    }
    const mid = (lo + hi) >> 1;
    select(idx, X, bd, lo, hi - 1, mid);
    split[mid] = bd;
    build(lo, mid, depth + 1);
    build(mid + 1, hi, depth + 1);
  };
  build(0, n, 0);
  set.tree = { X, y, n, idx, split };
  return set.tree;
}

/** Quickselect: put the element of rank `kth` (by feature c) at position kth of idx[l..r]. */
function select(idx: Int32Array, X: Float32Array, c: number, l: number, r: number, kth: number): void {
  while (r > l) {
    const pv = X[idx[(l + r) >> 1] * NF + c];
    let i = l, j = r;
    while (i <= j) {
      while (X[idx[i] * NF + c] < pv) i++;
      while (X[idx[j] * NF + c] > pv) j--;
      if (i <= j) { const t = idx[i]; idx[i] = idx[j]; idx[j] = t; i++; j--; }
    }
    if (kth <= j) r = j; else if (kth >= i) l = i; else return;
  }
}

const STACK = new Int32Array(512), BOUND = new Float64Array(256);
/** Nodes visited per query at most (best-bin-first cut-off: near-exact, and fast enough to feel live). */
export let MAX_VISITS = 400;
export function setMaxVisits(v: number): void { MAX_VISITS = v; }
const QV = new Float64Array(NF);

/** Predict one feature vector f[o .. o+NF): majority of the k nearest allowed (training) pixels. */
export function predictKnn(m: Knn, allowed: Uint8Array, k: number, f: ArrayLike<number>, o = 0): number {
  const bd = new Float64Array(k).fill(Infinity), bl = new Uint8Array(k), X = m.X, idx = m.idx, sp = m.split, y = m.y;
  for (let c = 0; c < NF; c++) QV[c] = f[o + c];
  // explicit stack of (lo, hi, bound²): nodes to visit
  const st = STACK, bnd = BOUND;
  let top = 0, visits = 0;
  st[0] = 0; st[1] = m.n; bnd[0] = 0; top = 1;
  while (top > 0 && visits++ < MAX_VISITS) {
    top--;
    const lo = st[2 * top], hi = st[2 * top + 1];
    if (bnd[top] >= bd[k - 1] || hi <= lo) continue;
    const mid = (lo + hi) >> 1, j = idx[mid], jb = j * NF;
    let d = 0;
    for (let c = 0; c < NF; c++) { const a = X[jb + c] - QV[c]; d += a * a; }
    if (d < bd[k - 1] && allowed[j]) {
      let t = k - 1;
      while (t > 0 && bd[t - 1] > d) { bd[t] = bd[t - 1]; bl[t] = bl[t - 1]; t--; }
      bd[t] = d; bl[t] = y[j];
    }
    if (hi - lo === 1) continue;
    const c = sp[mid], diff = QV[c] - X[jb + c], d2 = diff * diff;
    // push the far side first (visited later), with its lower bound
    if (diff < 0) { st[2 * top] = mid + 1; st[2 * top + 1] = hi; bnd[top] = d2; top++; st[2 * top] = lo; st[2 * top + 1] = mid; bnd[top] = 0; top++; }
    else { st[2 * top] = lo; st[2 * top + 1] = mid; bnd[top] = d2; top++; st[2 * top] = mid + 1; st[2 * top + 1] = hi; bnd[top] = 0; top++; }
  }
  let s = 0;
  for (let t = 0; t < k; t++) s += bl[t];
  return 2 * s > k ? 1 : 0;
}

/** Accuracy of always predicting the more frequent class (over the original images). */
export function majority(set: DataSet): number {
  let pos = 0, n = 0;
  set.labels.forEach((l, i) => { if (set.n === DAYS * PER_DAY || i % 4 === 0) for (const v of l) { pos += v; n++; } });
  return Math.max(pos / n, 1 - pos / n);
}

export interface StrategyResult { strategy: Strategy; label: string; folds: number[]; mean: number; min: number; max: number }

/** Accuracy of each fold for one strategy. */
export function evaluate(set: DataSet, strat: Strategy, s: Settings, seed = 1): number[] {
  const folds = assignFolds(set, strat, seed), r = rng(seed + 11), acc: number[] = [], tree = buildTree(set);
  const allowed = new Uint8Array(set.n * S * S);
  for (let f = 0; f < 4; f++) {
    const ro = roles(set, folds, f, strat, s.buffer), test: number[] = [];
    ro.forEach((role, i) => {
      for (let p = 0; p < S * S; p++) {
        allowed[i * S * S + p] = role[p] === 0 ? 1 : 0;
        // the test set holds original images only (rotated copies are training material)
        if (role[p] === 1 && (set.n === DAYS * PER_DAY || i % 4 === 0)) test.push(i * S * S + p);
      }
    });
    const ts = sample(r, test, NTEST);
    let ok = 0;
    for (const id of ts) { const i = Math.floor(id / (S * S)), p = id % (S * S); if (predictKnn(tree, allowed, s.k, set.feats[i], p * NF) === set.labels[i][p]) ok++; }
    acc.push(ts.length ? ok / ts.length : NaN);
  }
  return acc;
}

export function evaluateAll(set: DataSet, s: Settings, seed = 1): StrategyResult[] {
  return STRATEGIES.map(st => summarise(st.id, evaluate(set, st.id, s, seed)));
}

export function summarise(strategy: Strategy, a: number[]): StrategyResult {
  return { strategy, label: STRATEGIES.find(q => q.id === strategy)!.label, folds: a, mean: a.reduce((x, y) => x + y, 0) / a.length, min: Math.min(...a), max: Math.max(...a) };
}

/** Prediction of a whole image (index into set) for the fold that holds most of it, with the roles of all pixels. */
export function predictImage(set: DataSet, strat: Strategy, s: Settings, img: number, seed = 1): { pred: Uint8Array; roles: Uint8Array[]; acc: number } {
  const folds = assignFolds(set, strat, seed), tree = buildTree(set);
  const cnt = [0, 0, 0, 0];
  folds[img].forEach(v => cnt[v]++);
  const f = cnt.indexOf(Math.max(...cnt));
  const ro = roles(set, folds, f, strat, s.buffer);
  const allowed = new Uint8Array(set.n * S * S);
  ro.forEach((role, i) => { for (let p = 0; p < S * S; p++) allowed[i * S * S + p] = role[p] === 0 ? 1 : 0; });
  const pred = new Uint8Array(S * S);
  let ok = 0, n = 0;
  for (let p = 0; p < S * S; p++) {
    pred[p] = predictKnn(tree, allowed, s.k, set.feats[img], p * NF);
    if (ro[img][p] === 1) { n++; if (pred[p] === set.labels[img][p]) ok++; }
  }
  return { pred, roles: ro, acc: n ? ok / n : NaN };
}

export interface Preset { id: string; label: string; settings: Partial<Settings>; show?: Strategy }

export const PRESETS: Preset[] = [
  { id: "default", label: "random pixels look great", settings: {}, show: "pixels" },
  { id: "nothing", label: "nothing to learn", settings: { nothing: true }, show: "pixels" },
  { id: "buffer", label: "blocks need a buffer", settings: { nothing: true }, show: "buffer" },
  { id: "newday", label: "a new day: domain shift", settings: { dayEffect: 2 }, show: "day" },
  { id: "noday", label: "no day effect", settings: { dayEffect: 0 }, show: "image" },
  { id: "augment", label: "augment then split", settings: { augment: true }, show: "image" },
];
