// Active shape model search (Cootes, Taylor, Cooper & Graham 1995) on synthetic images
// (no DOM). The shape model is a PDM of GPA-aligned training shapes (lib/shapes.ts) in a
// model frame where a landmark's RMS distance from the centroid is 1; the image shape is
// X = T(x̄ + Φb), with T the similarity [[a, −b], [b, a]] x + t. One search iteration:
//   1. at every landmark, sample the image along the normal and find the best offset
//      (strongest edge, or the smallest Mahalanobis distance to the trained profile);
//   2. fit the pose T to the suggested points Y (Procrustes);
//   3. b = Φᵀ(T⁻¹(Y) − x̄);   4. clamp |b_k| ≤ 3√λ_k.
// Coarse to fine on a Gaussian pyramid: each level has its own profile model.
import { Grey, gaussianPyramid, sampleBilinear } from "../lib/imageops";
import { inverse } from "../lib/linalg";
import { rng } from "../lib/random";
import {
  Shape, Family, PDM, family, trainingSet, gpa, pdm, applySim, procrustes, invertFit, projectB, clampB,
  synthesise, renderShape, shapeNormals, nPoints, centroidSize, centre, sampleParams, RenderOptions,
} from "../lib/shapes";

export const IMG = 128;
/** Scale of the true and the training shapes in the image: pixels per model unit. */
export const SCALE = 26;

export interface Pose { a: number; b: number; tx: number; ty: number }

export const poseOf = (s: number, theta: number, tx: number, ty: number): Pose => ({ a: s * Math.cos(theta), b: s * Math.sin(theta), tx, ty });
export const poseScale = (p: Pose) => Math.hypot(p.a, p.b);
export const poseAngle = (p: Pose) => Math.atan2(p.b, p.a);

export function toImage(x: Shape, p: Pose): Shape { return applySim(x, p.a, p.b, p.tx, p.ty); }

/** A shape in the model frame: centred, RMS distance 1 (centroid size √n). */
export function toModelFrame(s: Shape): Shape {
  const c = centre(s), k = Math.sqrt(nPoints(s)) / centroidSize(c);
  return c.map(v => v * k);
}

export interface Profiles { mean: Float64Array[]; icov: Float64Array[] }

export interface AsmModel {
  fam: Family;
  pdm: PDM;
  /** Profile half-length k (profiles have 2k + 1 samples). */
  k: number;
  levels: number;
  profiles: Profiles[];   // one per pyramid level
}

export interface TrainOptions { family: string; count?: number; seed?: number; k?: number; levels?: number }

/** Derivative profile of 2k+1 samples centred at (x, y) along (nx, ny), normalised to Σ|g| = 1; also the raw derivative. */
export function sampleProfile(img: Grey, x: number, y: number, nx: number, ny: number, k: number): { g: Float64Array; raw: Float64Array } {
  const v = new Float64Array(2 * k + 3);
  for (let t = -k - 1; t <= k + 1; t++) v[t + k + 1] = sampleBilinear(img, x + t * nx, y + t * ny, "replicate");
  const raw = new Float64Array(2 * k + 1);
  let s = 0;
  for (let t = 0; t < 2 * k + 1; t++) { raw[t] = (v[t + 2] - v[t]) / 2; s += Math.abs(raw[t]); }
  const g = raw.map(d => d / (s + 1e-6));
  return { g, raw };
}

/** Train the shape model (GPA + PCA) and the profile models (per landmark and level) from rendered training images. */
export function train(o: TrainOptions): AsmModel {
  const fam = family(o.family), k = o.k ?? 3, levels = o.levels ?? 3;
  const ts = trainingSet(fam, { count: o.count ?? 40, seed: o.seed ?? 3, noise: 0.01 });
  const g = gpa(ts.shapes);
  const aligned = g.aligned.map(s => s.map(v => v * Math.sqrt(fam.n)));
  const model = pdm(aligned);
  const r = rng(17), n = fam.n, d = 2 * k + 1;
  const sums: { m: Float64Array; S: Float64Array; c: number }[][] = Array.from({ length: levels }, () => Array.from({ length: n }, () => ({ m: new Float64Array(d), S: new Float64Array(d * d), c: 0 })));
  aligned.forEach((x, j) => {
    const pose = poseOf(SCALE * (1 + 0.05 * r.norm()), 0.1 * r.norm(), IMG / 2 + 3 * r.norm(), IMG / 2 + 3 * r.norm());
    const X = toImage(x, pose);
    const img = renderShape(X, { width: IMG, height: IMG, blur: 1.2, noise: 0.03, seed: 100 + j });
    const pyr = gaussianPyramid(img, levels);
    const N = shapeNormals(X);
    for (let L = 0; L < levels; L++) {
      const f = 2 ** L;
      for (let i = 0; i < n; i++) {
        const { g: p } = sampleProfile(pyr[L], X[2 * i] / f, X[2 * i + 1] / f, N[2 * i], N[2 * i + 1], k);
        const acc = sums[L][i];
        acc.c++;
        for (let a = 0; a < d; a++) { acc.m[a] += p[a]; for (let b2 = 0; b2 < d; b2++) acc.S[a * d + b2] += p[a] * p[b2]; }
      }
    }
  });
  const profiles: Profiles[] = sums.map(lv => {
    const mean: Float64Array[] = [], icov: Float64Array[] = [];
    for (const acc of lv) {
      const m = acc.m.map(v => v / acc.c), S = new Float64Array(d * d);
      let tr = 0;
      for (let a = 0; a < d; a++) for (let b2 = 0; b2 < d; b2++) { S[a * d + b2] = acc.S[a * d + b2] / acc.c - m[a] * m[b2]; if (a === b2) tr += S[a * d + b2]; }
      for (let a = 0; a < d; a++) S[a * d + a] += 0.05 * tr / d + 1e-6;   // regularise
      mean.push(m); icov.push(inverse(S, d) ?? new Float64Array(d * d));
    }
    return { mean, icov };
  });
  return { fam, pdm: model, k, levels, profiles };
}

export interface SearchOptions {
  /** Number of shape modes used. */
  modes: number;
  /** Search half-length m (offsets −m..m, in pixels of the level). */
  m: number;
  match: "edge" | "mahal";
  levels: number;
  constrain: boolean;
  /** Iterations per level. */
  iterPerLevel: number;
}

export const SEARCH_DEFAULTS: SearchOptions = { modes: 3, m: 6, match: "mahal", levels: 1, constrain: true, iterPerLevel: 12 };

/** Cost of every offset −m..m at landmark i (lower is better). */
export function offsetCosts(M: AsmModel, img: Grey, L: number, X: Shape, N: Shape, i: number, o: SearchOptions): Float64Array {
  const f = 2 ** L, out = new Float64Array(2 * o.m + 1), d = 2 * M.k + 1;
  const pm = M.profiles[Math.min(L, M.levels - 1)];
  for (let t = -o.m; t <= o.m; t++) {
    const x = X[2 * i] / f + t * N[2 * i], y = X[2 * i + 1] / f + t * N[2 * i + 1];
    const { g, raw } = sampleProfile(img, x, y, N[2 * i], N[2 * i + 1], M.k);
    if (o.match === "edge") out[t + o.m] = -Math.abs(raw[M.k]);
    else {
      const mu = pm.mean[i], S = pm.icov[i];
      let c = 0;
      for (let a = 0; a < d; a++) { let r = 0; for (let b = 0; b < d; b++) r += S[a * d + b] * (g[b] - mu[b]); c += (g[a] - mu[a]) * r; }
      out[t + o.m] = c;
    }
  }
  return out;
}

export type Stage = "start" | "suggested" | "pose" | "project" | "clamp";

/** The state of one search, stepped sub-step by sub-step. */
export class Search {
  M: AsmModel;
  pyr: Grey[];
  o: SearchOptions;
  pose: Pose;
  b: Float64Array;          // shape parameters (all modes; only the first o.modes are used)
  bRaw: Float64Array;       // after projection, before clamping
  X: Shape;                 // current shape in the image
  Y: Shape | null = null;   // suggested points
  private X0: Shape;        // shape at the start of the iteration
  level: number;
  iter = 0;                 // iterations at this level
  total = 0;
  stage: Stage = "start";
  /** Mean landmark movement of the last iteration (pixels). */
  moved = NaN;

  constructor(M: AsmModel, img: Grey, start: Pose, o: SearchOptions) {
    this.M = M; this.o = o;
    this.pyr = gaussianPyramid(img, Math.max(1, o.levels));
    this.pose = { ...start };
    this.b = new Float64Array(M.pdm.values.length);
    this.bRaw = this.b.slice();
    this.level = Math.max(1, o.levels) - 1;
    this.X = toImage(this.modelShape(this.b), this.pose);
    this.X0 = this.X;
  }

  modelShape(b: Float64Array): Shape { return synthesise(this.M.pdm, b.subarray(0, this.o.modes)); }

  /** 1. Suggested points along the normals. */
  suggest(): void {
    this.X0 = this.X;
    const n = nPoints(this.X), N = shapeNormals(this.X), Y = new Float64Array(this.X.length), f = 2 ** this.level;
    for (let i = 0; i < n; i++) {
      const c = offsetCosts(this.M, this.pyr[this.level], this.level, this.X, N, i, this.o);
      // ties (a flat profile) keep the point where it is
      let best = this.o.m;
      for (let t = 0; t < c.length; t++) if (c[t] < c[best] - 1e-12) best = t;
      const off = (best - this.o.m) * f;
      Y[2 * i] = this.X[2 * i] + off * N[2 * i]; Y[2 * i + 1] = this.X[2 * i + 1] + off * N[2 * i + 1];
    }
    this.Y = Y; this.stage = "suggested";
  }

  /** 2. Pose: Procrustes of the current model shape onto Y. (Without the constraint: X = Y.) */
  fitPose(): void {
    if (!this.Y) return;
    if (!this.o.constrain) { this.finish(this.Y); return; }
    const f = procrustes(this.modelShape(this.b), this.Y);
    this.pose = { a: f.a, b: f.b, tx: f.tx, ty: f.ty };
    this.X = toImage(this.modelShape(this.b), this.pose);
    this.stage = "pose";
  }

  /** 3. Project Y into the model frame and onto the modes. */
  project(): void {
    if (!this.Y || !this.o.constrain) return;
    const fit = { a: this.pose.a, b: this.pose.b, tx: this.pose.tx, ty: this.pose.ty, reflect: false, s: 0, theta: 0, rss: 0, aligned: this.Y };
    const y = invertFit(fit, this.Y);
    const bb = projectB(this.M.pdm, y, this.o.modes);
    this.bRaw = new Float64Array(this.b.length); this.bRaw.set(bb);
    this.X = toImage(this.modelShape(this.bRaw), this.pose);
    this.stage = "project";
  }

  /** 4. Clamp to the plausible range; end of one iteration. */
  clamp(): void {
    if (!this.o.constrain) return;
    const c = clampB(this.M.pdm, this.bRaw.subarray(0, this.o.modes), 3);
    const b = new Float64Array(this.b.length); b.set(c);
    this.b = b;
    this.finish(toImage(this.modelShape(this.b), this.pose));
  }

  private finish(Xnew: Shape): void {
    let mv = 0;
    for (let i = 0; i < Xnew.length; i += 2) mv += Math.hypot(Xnew[i] - this.X0[i], Xnew[i + 1] - this.X0[i + 1]);
    this.moved = mv / nPoints(Xnew);
    this.X = Xnew; this.stage = "clamp";
    this.iter++; this.total++;
    if (this.level > 0 && this.iter >= this.o.iterPerLevel) { this.level--; this.iter = 0; }
  }

  /** The next sub-step. */
  next(): void {
    if (this.stage === "start" || this.stage === "clamp") this.suggest();
    else if (this.stage === "suggested") this.fitPose();
    else if (this.stage === "pose") this.project();
    else this.clamp();
  }

  /** One full iteration (from wherever it is to the end of the next clamp). */
  iterate(): void {
    do this.next(); while (this.stage !== "clamp");
  }

  done(): boolean { return this.level === 0 && this.iter >= this.o.iterPerLevel; }
}

// ---------------------------------------------------------------------------------------
// Test images

export interface Scene {
  /** True shape in the image. */
  truth: Shape;
  img: Grey;
  /** True pose of the model frame. */
  pose: Pose;
}

/** A test image of an unseen shape of the family (params from `seed`, or given). */
export function makeScene(fam: Family, o: { seed?: number; params?: number[]; noise?: number; clutter?: number; occluder?: RenderOptions["occluder"]; pose?: Pose }): Scene {
  const p = o.params ?? sampleParams(fam, rng(o.seed ?? 77));
  const x = toModelFrame(fam.make(p));
  const pose = o.pose ?? poseOf(SCALE, 0.08, IMG / 2 + 2, IMG / 2 + 3);
  const truth = toImage(x, pose);
  const img = renderShape(truth, { width: IMG, height: IMG, blur: 1.2, noise: o.noise ?? 0.03, clutter: o.clutter ?? 0, occluder: o.occluder ?? null, seed: 5 });
  return { truth, img, pose };
}

/** RMS distance between corresponding landmarks. */
export function rmsError(a: Shape, b: Shape): number {
  let s = 0;
  for (let i = 0; i < a.length; i += 2) s += (a[i] - b[i]) ** 2 + (a[i + 1] - b[i + 1]) ** 2;
  return Math.sqrt(s / nPoints(a));
}
