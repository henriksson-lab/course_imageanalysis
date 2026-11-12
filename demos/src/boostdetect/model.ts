// Training and detection for the boostdetect demo (no DOM): feature values of the pool on
// every patch (variance-normalised), AdaBoost, a cascade, and sliding-window detection
// with scaled features on a test scene.
import { integralImage, Table, TEMPLATES } from "../lib/integral";
import {
  FeatureMatrix, prepareFeatures, FeatureSpec, featurePool, scaleFeature, ScaledFeature, featureValue, windowStats,
  Stage, Stump, stumpPredict, strongScore,
} from "../lib/boost";
import { WIN, Dataset } from "./synth";

/** The feature pool: all templates, positions on a stride-2 grid, cell sizes 2..8 (several thousand). */
export const POOL_SIZES = [2, 4, 6, 8];
export const POOL_STRIDE = 3;
export function makePool(): FeatureSpec[] { return featurePool(WIN, POOL_STRIDE, POOL_SIZES); }

export interface PatchTables { T1: Table; sd: number }

export function patchTables(p: Float64Array): PatchTables {
  const T1 = integralImage(p, WIN, WIN), T2 = integralImage(p, WIN, WIN, "f64", v => v * v);
  return { T1, sd: windowStats(T1, T2, 0, 0, WIN).sd };
}

/** Feature-value matrix (features × examples) of a dataset, normalised by each patch's SD. */
export function featureMatrix(pool: FeatureSpec[], data: Dataset): { M: FeatureMatrix; tables: PatchTables[] } {
  const tables = data.patches.map(patchTables);
  const nEx = data.patches.length, nF = pool.length;
  const values = new Float32Array(nF * nEx);
  for (let f = 0; f < nF; f++) {
    const sf = scaleFeature(pool[f], 1);
    for (let i = 0; i < nEx; i++) values[f * nEx + i] = featureValue(tables[i].T1, sf, 0, 0, tables[i].sd);
  }
  return { M: prepareFeatures(values, nF, nEx), tables };
}

/** Value of pool feature f on a patch (for test examples). */
export function patchFeature(pool: FeatureSpec[], f: number, t: PatchTables): number {
  return featureValue(t.T1, scaleFeature(pool[f], 1), 0, 0, t.sd);
}

/** Error rate of sign(Σ α h) over a set of patches, for every number of rounds 1..T. */
export function errorCurve(pool: FeatureSpec[], stumps: Stump[], tables: PatchTables[], y: ArrayLike<number>): number[] {
  const score = new Float64Array(tables.length), out: number[] = [];
  for (const s of stumps) {
    let wrong = 0;
    for (let i = 0; i < tables.length; i++) {
      score[i] += s.alpha * stumpPredict(s, patchFeature(pool, s.feature, tables[i]));
      if ((score[i] >= 0 ? 1 : -1) !== y[i]) wrong++;
    }
    out.push(wrong / tables.length);
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Detection

export interface Window { x: number; y: number; size: number; score: number; rejectedAt: number; features: number }

export interface DetectResult {
  windows: number;
  /** windows[k] reached stage k (k = 0 means all windows). */
  reached: number[];
  totalFeatures: number;
  detections: Window[];
  merged: Window[];
  /** Per pixel of the scene (scale 1 only): the stage that rejected the window centred there (−1 = passed, NaN = none). */
  stageMap: Float64Array;
}

export const SCALES = [1, 1.25, 1.5, 2];

/**
 * Slide a window over every position and scale of the scene; features are scaled, not the
 * image. `stages` is the cascade; with `monolithic` all stumps are evaluated (one strong
 * classifier with threshold `monoThreshold`).
 */
export function detect(img: Float64Array, W: number, H: number, pool: FeatureSpec[], stages: Stage[],
  monolithic: { stumps: Stump[]; threshold: number } | null): DetectResult {
  const T1 = integralImage(img, W, H), T2 = integralImage(img, W, H, "f64", v => v * v);
  const reached = new Array(stages.length + 1).fill(0);
  const stageMap = new Float64Array(W * H).fill(NaN);
  let windows = 0, totalFeatures = 0;
  const detections: Window[] = [];
  for (const sc of SCALES) {
    const size = Math.round(WIN * sc), step = Math.max(2, Math.round(2 * sc));
    const cache = new Map<number, ScaledFeature>();
    const sfOf = (f: number) => { let s = cache.get(f); if (!s) { s = scaleFeature(pool[f], sc); cache.set(f, s); } return s; };
    for (let y = 0; y + size <= H; y += step)
      for (let x = 0; x + size <= W; x += step) {
        windows++;
        const { sd } = windowStats(T1, T2, x, y, size);
        const val = (f: number) => featureValue(T1, sfOf(f), x, y, sd);
        let rejectedAt = -1, feats = 0, score = 0;
        if (monolithic) {
          score = strongScore(monolithic.stumps, val);
          feats = monolithic.stumps.length;
          if (score < monolithic.threshold) rejectedAt = 0;
        } else {
          reached[0]++;
          for (let k = 0; k < stages.length; k++) {
            score = strongScore(stages[k].stumps, val);
            feats += stages[k].stumps.length;
            if (score < stages[k].threshold) { rejectedAt = k; break; }
            reached[k + 1]++;
          }
        }
        totalFeatures += feats;
        if (sc === 1) {
          const c = (y + (size >> 1)) * W + x + (size >> 1);
          for (let dy = 0; dy < step; dy++) for (let dx = 0; dx < step; dx++) stageMap[c + dy * W + dx] = rejectedAt;
        }
        if (rejectedAt < 0) detections.push({ x, y, size, score, rejectedAt, features: feats });
      }
  }
  if (monolithic) { reached[0] = windows; reached[1] = detections.length; }
  return { windows, reached, totalFeatures, detections, merged: mergeDetections(detections), stageMap };
}

/** Group overlapping detections (overlap of the smaller box > 40%) and average each group. */
export function mergeDetections(d: Window[]): Window[] {
  const groups: Window[][] = [];
  const overlap = (a: Window, b: Window) => {
    const ix = Math.max(0, Math.min(a.x + a.size, b.x + b.size) - Math.max(a.x, b.x));
    const iy = Math.max(0, Math.min(a.y + a.size, b.y + b.size) - Math.max(a.y, b.y));
    return ix * iy / Math.min(a.size, b.size) ** 2;
  };
  for (const w of d) {
    const g = groups.find(gr => gr.some(o => overlap(o, w) > 0.4));
    if (g) g.push(w); else groups.push([w]);
  }
  return groups.filter(g => g.length >= 2).map(g => ({
    x: g.reduce((a, w) => a + w.x, 0) / g.length, y: g.reduce((a, w) => a + w.y, 0) / g.length,
    size: g.reduce((a, w) => a + w.size, 0) / g.length, score: Math.max(...g.map(w => w.score)), rejectedAt: -1, features: 0,
  }));
}

/** Does a merged detection match a target box (overlap > 40% of the smaller)? */
export function matches(a: { x: number; y: number; size: number }, b: { x: number; y: number; size: number }): boolean {
  const ix = Math.max(0, Math.min(a.x + a.size, b.x + b.size) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.size, b.y + b.size) - Math.max(a.y, b.y));
  return ix * iy / Math.min(a.size, b.size) ** 2 > 0.4 && Math.max(a.size, b.size) / Math.min(a.size, b.size) < 1.6;
}

export { TEMPLATES };

// ---------------------------------------------------------------------------------------
// Cascade training with bootstrapped negatives (Viola & Jones): each stage is trained on
// the positives and on background windows that all earlier stages wrongly accepted.

import { rng } from "../lib/random";
import { trainStage, StageReport, runCascade } from "../lib/boost";
import { background, Target } from "./synth";

/** Resample the window [x, x+size)² of a scene to a 24 × 24 patch (area average). */
export function windowPatch(img: Float64Array, W: number, x: number, y: number, size: number): Float64Array {
  const out = new Float64Array(WIN * WIN), f = size / WIN;
  for (let j = 0; j < WIN; j++) for (let i = 0; i < WIN; i++) {
    let s = 0, n = 0;
    for (let yy = y + Math.floor(j * f); yy < y + Math.ceil((j + 1) * f); yy++)
      for (let xx = x + Math.floor(i * f); xx < x + Math.ceil((i + 1) * f); xx++) { s += img[yy * W + xx]; n++; }
    out[j * WIN + i] = s / n;
  }
  return out;
}

/**
 * Collect up to `count` background windows (from scenes without targets) that pass every
 * stage of the cascade so far. Returns the patches and the pass rate of the windows tried.
 */
export function mineNegatives(target: Target, pool: FeatureSpec[], stages: Stage[], count: number, seed: number,
  maxTries = 60000): { patches: Float64Array[]; tried: number; passRate: number } {
  const R = rng(seed), W = 256, patches: Float64Array[] = [];
  let tried = 0, scene: Float64Array | null = null, T1: Table | null = null, T2: Table | null = null;
  const caches = SCALES.map(() => new Map<number, ScaledFeature>());
  while (patches.length < count && tried < maxTries) {
    if (tried % 1500 === 0) {
      scene = background(W, W, R, target, 0.9);
      for (let i = 0; i < scene.length; i++) scene[i] = Math.max(0, Math.min(255, scene[i] + 7 * R.norm()));
      T1 = integralImage(scene, W, W); T2 = integralImage(scene, W, W, "f64", v => v * v);
    }
    tried++;
    const si = Math.floor(R.unif() * SCALES.length), sc = SCALES[si], size = Math.round(WIN * sc);
    const x = Math.floor(R.unif() * (W - size + 1)), y = Math.floor(R.unif() * (W - size + 1));
    const { sd } = windowStats(T1!, T2!, x, y, size);
    const cache = caches[si];
    const val = (f: number) => { let s = cache.get(f); if (!s) { s = scaleFeature(pool[f], sc); cache.set(f, s); } return featureValue(T1!, s, x, y, sd); };
    if (runCascade(stages, val).rejectedAt < 0) patches.push(windowPatch(scene!, W, x, y, size));
  }
  return { patches, tried, passRate: tried ? patches.length / tried : 0 };
}

export interface CascadeStep {
  stage: Stage | null;
  report: StageReport | null;
  /** Fraction of background windows that passed all earlier stages (measured while mining). */
  passBefore: number;
  negatives: number;
}

/**
 * Train the next stage of a cascade: mine negatives that pass the current stages, train a
 * boosted stage to detection ≥ dTarget and false positives ≤ fTarget on them.
 */
export function nextStage(target: Target, pool: FeatureSpec[], posTables: Float64Array[], stages: Stage[],
  dTarget: number, fTarget: number, maxStumps: number, nNeg: number, seed: number): CascadeStep {
  const mined = mineNegatives(target, pool, stages, nNeg, seed);
  if (mined.patches.length < 20) return { stage: null, report: null, passBefore: mined.passRate, negatives: mined.patches.length };
  const data: Dataset = { patches: [...posTables, ...mined.patches], y: Int8Array.from({ length: posTables.length + mined.patches.length }, (_, i) => (i < posTables.length ? 1 : -1)), nPos: posTables.length };
  const { M } = featureMatrix(pool, data);
  const mask = new Float64Array(data.y.length).fill(1);
  const { stage, report } = trainStage(M, data.y, mask, dTarget, fTarget, maxStumps);
  return { stage, report, passBefore: mined.passRate, negatives: mined.patches.length };
}
