// Iterative closest point (ICP) in 2D, no DOM (owner: keypoints day, unit icp/).
//
// A transform is a similarity x' = a x − b y + tx, y' = b x + a y + ty: scale √(a² + b²),
// angle atan2(b, a) (with y down, a positive angle turns clockwise on screen). Rigid: a² + b² = 1;
// translation: a = 1, b = 0.
//  - fitPairs(P, Q, kind): the least-squares transform mapping P[i] onto Q[i], closed form:
//    centre both sets; S_dot = Σ p·q, S_cross = Σ (p_x q_y − p_y q_x), S_pp = Σ |p|² (centred);
//    rigid: θ = atan2(S_cross, S_dot); similarity: a = S_dot / S_pp, b = S_cross / S_pp;
//    then t = q̄ − M p̄.
//  - nearest(P, Q): for each point of P the nearest point of Q (brute force).
//  - icpStep: match every moved point to its nearest fixed point, keep the closest fraction
//    (trimmed ICP), and refit the transform from the ORIGINAL moving points to their matches.
//  - icp: repeat until the matching (and so the transform) no longer changes.
//  - makeScene: a fixed set and a moving set with a known true transform (spots or an outline),
//    with noise, missing and extra points.

import { rng } from "../lib/random";

export interface Pt { x: number; y: number }
export type Kind = "translation" | "rigid" | "similarity";
export interface Sim { a: number; b: number; tx: number; ty: number }

export const IDENTITY: Sim = { a: 1, b: 0, tx: 0, ty: 0 };
export const apply = (T: Sim, p: Pt): Pt => ({ x: T.a * p.x - T.b * p.y + T.tx, y: T.b * p.x + T.a * p.y + T.ty });
export const applyAll = (T: Sim, P: Pt[]): Pt[] => P.map(p => apply(T, p));
export const angleOf = (T: Sim): number => Math.atan2(T.b, T.a);
export const scaleOf = (T: Sim): number => Math.hypot(T.a, T.b);
export const translation = (tx: number, ty: number): Sim => ({ a: 1, b: 0, tx, ty });

/** T2 ∘ T1 (first T1, then T2). */
export function compose(T2: Sim, T1: Sim): Sim {
  const a = T2.a * T1.a - T2.b * T1.b, b = T2.b * T1.a + T2.a * T1.b;
  const t = apply(T2, { x: T1.tx, y: T1.ty });
  return { a, b, tx: t.x, ty: t.y };
}

export function invert(T: Sim): Sim {
  const d = T.a * T.a + T.b * T.b, a = T.a / d, b = -T.b / d;
  return { a, b, tx: -(a * T.tx - b * T.ty), ty: -(b * T.tx + a * T.ty) };
}

/** Turn by θ (radians) about the centre c, then shift by (sx, sy). */
export function rotationAbout(theta: number, c: Pt, sx = 0, sy = 0): Sim {
  const a = Math.cos(theta), b = Math.sin(theta);
  return { a, b, tx: c.x - (a * c.x - b * c.y) + sx, ty: c.y - (b * c.x + a * c.y) + sy };
}

export function centroid(P: Pt[]): Pt {
  let x = 0, y = 0;
  for (const p of P) { x += p.x; y += p.y; }
  return { x: x / P.length, y: y / P.length };
}

export interface PairFit { T: Sim; pBar: Pt; qBar: Pt; sDot: number; sCross: number; spp: number }

/** Least-squares transform of the given kind mapping P[i] onto Q[i] (closed form, see header). */
export function fitPairsDetail(P: Pt[], Q: Pt[], kind: Kind): PairFit {
  const pBar = centroid(P), qBar = centroid(Q);
  let sDot = 0, sCross = 0, spp = 0;
  for (let i = 0; i < P.length; i++) {
    const x = P[i].x - pBar.x, y = P[i].y - pBar.y, u = Q[i].x - qBar.x, v = Q[i].y - qBar.y;
    sDot += x * u + y * v; sCross += x * v - y * u; spp += x * x + y * y;
  }
  let a = 1, b = 0;
  if (kind === "rigid") {
    const h = Math.hypot(sDot, sCross);
    if (h > 0) { a = sDot / h; b = sCross / h; }
  } else if (kind === "similarity" && spp > 0) {
    a = sDot / spp; b = sCross / spp;
  }
  const T: Sim = { a, b, tx: qBar.x - (a * pBar.x - b * pBar.y), ty: qBar.y - (b * pBar.x + a * pBar.y) };
  return { T, pBar, qBar, sDot, sCross, spp };
}
export const fitPairs = (P: Pt[], Q: Pt[], kind: Kind): Sim => fitPairsDetail(P, Q, kind).T;

/** Sum of squared distances |T(P[i]) − Q[i]|². */
export function sse(T: Sim, P: Pt[], Q: Pt[]): number {
  let s = 0;
  for (let i = 0; i < P.length; i++) { const r = apply(T, P[i]); s += (r.x - Q[i].x) ** 2 + (r.y - Q[i].y) ** 2; }
  return s;
}

/** For each point of P, the index of and the distance to the nearest point of Q (brute force: |P| · |Q| distances). */
export function nearest(P: Pt[], Q: Pt[]): { idx: Int32Array; dist: Float64Array } {
  const idx = new Int32Array(P.length), dist = new Float64Array(P.length);
  for (let i = 0; i < P.length; i++) {
    let best = Infinity, bj = -1;
    for (let j = 0; j < Q.length; j++) {
      const d = (P[i].x - Q[j].x) ** 2 + (P[i].y - Q[j].y) ** 2;
      if (d < best) { best = d; bj = j; }
    }
    idx[i] = bj; dist[i] = Math.sqrt(best);
  }
  return { idx, dist };
}

export interface IcpOptions { kind: Kind; keep: number }

/** One ICP iteration from the transform T. */
export interface IcpStep {
  /** The transform before this step. */
  T: Sim;
  /** Nearest fixed point of every moved point, and its distance. */
  match: Int32Array;
  dist: Float64Array;
  /** The pairs used in the fit (the closest `keep` fraction). */
  kept: boolean[];
  /** RMS distance of the kept pairs before and after the fit. */
  rms: number;
  rmsAfter: number;
  /** The new transform. */
  next: Sim;
}

/** Number of pairs kept by trimmed ICP: round(keep · n), at least 2 (at most n). */
export const keepCount = (n: number, keep: number): number => Math.min(n, Math.max(2, Math.round(keep * n)));

export function icpStep(M: Pt[], F: Pt[], T: Sim, o: IcpOptions): IcpStep {
  const moved = applyAll(T, M);
  const { idx, dist } = nearest(moved, F);
  const order = Array.from(dist.keys()).sort((i, j) => dist[i] - dist[j] || i - j);
  const kept = new Array<boolean>(M.length).fill(false);
  for (const i of order.slice(0, keepCount(M.length, o.keep))) kept[i] = true;
  const P: Pt[] = [], Q: Pt[] = [];
  for (let i = 0; i < M.length; i++) if (kept[i]) { P.push(M[i]); Q.push(F[idx[i]]); }
  const next = fitPairs(P, Q, o.kind);
  return { T, match: idx, dist, kept, rms: Math.sqrt(sse(T, P, Q) / P.length), rmsAfter: Math.sqrt(sse(next, P, Q) / P.length), next };
}

export interface IcpResult { steps: IcpStep[]; T: Sim; converged: boolean }

/** ICP from T0 until the matching repeats (then the transform cannot change any more), at most maxIter steps. */
export function icp(M: Pt[], F: Pt[], o: IcpOptions & { T0?: Sim; maxIter?: number }): IcpResult {
  const steps: IcpStep[] = [];
  let T = o.T0 ?? IDENTITY;
  const maxIter = o.maxIter ?? 100;
  for (let k = 0; k < maxIter; k++) {
    const s = icpStep(M, F, T, o);
    steps.push(s);
    const prev = steps.length > 1 ? steps[steps.length - 2] : null;
    T = s.next;
    if (prev && sameMatching(prev, s)) return { steps, T, converged: true };
  }
  return { steps, T, converged: false };
}

function sameMatching(a: IcpStep, b: IcpStep): boolean {
  for (let i = 0; i < a.match.length; i++) if (a.match[i] !== b.match[i] || a.kept[i] !== b.kept[i]) return false;
  return true;
}

/** Mean distance between where T and the true transform put the given points. */
export function trueError(T: Sim, truth: Sim, P: Pt[]): number {
  let s = 0;
  for (const p of P) { const a = apply(T, p), b = apply(truth, p); s += Math.hypot(a.x - b.x, a.y - b.y); }
  return s / P.length;
}

// ---------------------------------------------------------------------------------------
// Test scenes, in a 400 × 400 field.

export type SceneKind = "spots" | "outline";
export interface SceneOptions {
  scene: SceneKind;
  /** Number of fixed points. */
  n: number;
  /** True rotation (degrees) about the field centre and shift (px) that map the moving set onto the fixed one. */
  angle: number;
  shift: [number, number];
  /** Position noise (px, standard deviation) of the moving points. */
  noise: number;
  /** Fraction of the fixed points without a partner, and as many extra moving points. */
  unpaired: number;
  seed: number;
}
export interface Scene {
  F: Pt[];
  M: Pt[];
  /** The true transform, moving → fixed. */
  truth: Sim;
  /** The partner in F of each moving point (spots), -2 on the outline (a real point without an exact partner), -1 for an extra point. */
  partner: Int32Array;
}

export const FIELD = 400;
export const CENTRE: Pt = { x: FIELD / 2, y: FIELD / 2 };

/** A blobby cell outline: radius as a function of the angle (no symmetry). */
export const outlineRadius = (phi: number): number => 120 * (1 + 0.22 * Math.cos(2 * phi + 0.4) + 0.12 * Math.sin(3 * phi) + 0.05 * Math.cos(5 * phi + 1));

/**
 * A scene. The random numbers are drawn in the same order whatever the angle and shift, so
 * changing those moves the same points.
 */
export function makeScene(o: SceneOptions): Scene {
  const r = rng(o.seed);
  const F: Pt[] = [];
  if (o.scene === "spots") {
    let tries = 0;
    while (F.length < o.n && tries < 20000) {
      tries++;
      const p = { x: 70 + 260 * r.unif(), y: 70 + 260 * r.unif() };
      if (F.every(q => Math.hypot(p.x - q.x, p.y - q.y) >= 30)) F.push(p);
    }
  } else {
    for (let i = 0; i < o.n; i++) {
      const phi = 2 * Math.PI * i / o.n, rad = outlineRadius(phi);
      F.push({ x: CENTRE.x + rad * Math.cos(phi), y: CENTRE.y + 0.8 * rad * Math.sin(phi) });
    }
  }
  const truth = rotationAbout(o.angle * Math.PI / 180, CENTRE, o.shift[0], o.shift[1]);
  const inv = invert(truth);
  // The moving set samples the same objects (spots: the same points; outline: every third
  // position of the fixed sampling, half-way between two fixed samples, so no exact partners).
  const src: { p: Pt; i: number }[] = [];
  if (o.scene === "spots") F.forEach((p, i) => src.push({ p, i }));
  else for (let k = 0; 3 * k < o.n; k++) {
    const phi = 2 * Math.PI * (3 * k + 0.5) / o.n, rad = outlineRadius(phi);
    src.push({ p: { x: CENTRE.x + rad * Math.cos(phi), y: CENTRE.y + 0.8 * rad * Math.sin(phi) }, i: 3 * k });
  }
  // Partial overlap: the moving field of view misses the leftmost `unpaired` fraction of the
  // objects and sees as many other objects beyond the right edge of the fixed set.
  const nMiss = Math.round(o.unpaired * src.length);
  const byX = src.map((s, k) => k).sort((u, v) => src[u].p.x - src[v].p.x);
  const missing = new Set(byX.slice(0, nMiss));
  const right = Math.max(...F.map(p => p.x)), width = Math.max(...F.map(p => p.x)) - Math.min(...F.map(p => p.x));
  const M: Pt[] = [], partner: number[] = [];
  src.forEach((s, k) => {
    const nx = r.norm(), ny = r.norm();
    if (missing.has(k)) return;
    const q = apply(inv, s.p);
    M.push({ x: q.x + o.noise * nx, y: q.y + o.noise * ny });
    partner.push(o.scene === "spots" ? s.i : -2);
  });
  const ys = F.map(p => p.y), y0 = Math.min(...ys), y1 = Math.max(...ys);
  for (let k = 0; k < nMiss; k++) {
    const p = { x: right + 15 + (width * nMiss / src.length) * r.unif(), y: y0 + (y1 - y0) * r.unif() };
    M.push(apply(inv, p)); partner.push(-1);
  }
  return { F, M, truth, partner: Int32Array.from(partner) };
}

// ---------------------------------------------------------------------------------------
// The demo's runs (shared with the test, so that the numbers on the pages are checked).

export const SCENE_N: Record<SceneKind, number> = { spots: 40, outline: 90 };
export const DEMO_SHIFT: [number, number] = [12, -8];

export interface DemoSettings {
  scene: SceneKind; kind: Kind;
  /** Rotation between the sets (degrees), noise (px), unpaired fraction, kept fraction. */
  angle: number; noise: number; unpaired: number; keep: number;
  /** Start: match the centroids, else translate by `startShift` (dragged). */
  centroids: boolean; startShift: [number, number];
}

export function demoScene(s: DemoSettings): Scene {
  return makeScene({ scene: s.scene, n: SCENE_N[s.scene], angle: s.angle, shift: DEMO_SHIFT, noise: s.noise, unpaired: s.unpaired, seed: 1 });
}

export function startTransform(s: DemoSettings, sc: Scene): Sim {
  if (!s.centroids) return translation(s.startShift[0], s.startShift[1]);
  const a = centroid(sc.M), b = centroid(sc.F);
  return translation(b.x - a.x, b.y - a.y);
}

/** The moving points that really lie on the objects of the fixed set (not the extras). */
export const realPoints = (sc: Scene): Pt[] => sc.M.filter((_, i) => sc.partner[i] !== -1);

export function demoRun(s: DemoSettings, sc: Scene = demoScene(s)): { sc: Scene; res: IcpResult; err: number[] } {
  const res = icp(sc.M, sc.F, { kind: s.kind, keep: s.keep, T0: startTransform(s, sc), maxIter: 100 });
  const real = realPoints(sc);
  return { sc, res, err: res.steps.map(st => trueError(st.T, sc.truth, real)) };
}

/** Final true error for every rotation between the sets in `angles` (the other settings fixed). */
export function sweep(s: DemoSettings, angles: number[]): number[] {
  return angles.map(angle => {
    const r = demoRun({ ...s, angle });
    return trueError(r.res.T, r.sc.truth, realPoints(r.sc));
  });
}
