// Model of the colocobjects demo (no DOM). A two-channel spot scene at physical scale
// (pixel 65 nm, NA 1.4, A at 520 nm, B at 670 nm, two cells as the mask): nA spots in A, a
// fraction p of them with a B partner at a true distance d (random direction), extra
// independent B spots; Gaussian PSF of σ = 0.21 λ/NA, Poisson noise on a background, camera
// offset; an optional chromatic shift of B. Spots are detected in both channels (LoG +
// local maxima + weighted centroid) and compared: nearest-neighbour or one-to-one distances,
// the fraction within dMax and its chance level, cross-L with a Monte Carlo envelope, and
// for paired spots the measured distance against the true one.
import { S, cellMask, spotPairScene, spotImage, addNoise, psfSigmaNm, randomPointsInMask, SpotScene } from "../lib/twochannel";
import { detectSpots, crossNearest, pairOneToOne, fractionWithin, chanceFraction, correctedFraction, crossK, crossLminusR, envelope, pearsonMasked, meanMeasuredDistance, objectRelations, Spot } from "../lib/coloc";
import { label2d } from "../lib/label";
import { rng } from "../lib/random";
import { median } from "../lib/stats";

export const PIXEL_NM = 65, NA = 1.4, LAMBDA_A = 520, LAMBDA_B = 670;
export const BACKGROUND = 5, OFFSET = 100;
export const sigmaA = psfSigmaNm(NA, LAMBDA_A) / PIXEL_NM, sigmaB = psfSigmaNm(NA, LAMBDA_B) / PIXEL_NM;

export interface ObjOptions {
  nA: number;
  pair: number;
  /** True A → B distance in nm. */
  dNm: number;
  extraB: number;
  photons: number;
  /** Chromatic shift of B in x, nm. */
  shiftNm: number;
  seed?: number;
}

export const DEFAULT_OBJ: ObjOptions = { nA: 40, pair: 0.6, dNm: 0, extraB: 20, photons: 500, shiftNm: 0, seed: 11 };

export interface ObjScene {
  o: ObjOptions;
  mask: Uint8Array;
  /** Mask area in px². */
  area: number;
  truth: SpotScene;
  a: Float64Array; b: Float64Array;
  detA: Spot[]; detB: Spot[];
}

let MASK: Uint8Array | null = null;
export function mask(): Uint8Array { return MASK ??= cellMask(); }

export function makeObjScene(o: ObjOptions): ObjScene {
  const m = mask();
  let area = 0;
  for (let i = 0; i < m.length; i++) area += m[i];
  const truth = spotPairScene({ w: S, h: S, mask: m, nA: o.nA, pairFraction: o.pair, distance: o.dNm / PIXEL_NM, extraB: o.extraB, seed: o.seed ?? 11 });
  const sh = o.shiftNm / PIXEL_NM;
  const ea = spotImage(truth.ax, truth.ay, S, S, sigmaA, o.photons), eb = spotImage(truth.bx, truth.by, S, S, sigmaB, o.photons, sh, 0);
  for (let i = 0; i < S * S; i++) { const bg = BACKGROUND * (0.3 + 0.7 * m[i]); ea[i] += bg; eb[i] += bg; }
  const a = addNoise(ea, (o.seed ?? 11) + 1, OFFSET), b = addNoise(eb, (o.seed ?? 11) + 2, OFFSET);
  const detA = detectSpots(a, S, S, sigmaA, 5, m), detB = detectSpots(b, S, S, sigmaB, 5, m);
  return { o, mask: m, area, truth, a, b, detA, detB };
}

export interface DistanceResult {
  /** Per detected A spot: distance (px) to its B partner (nearest, or one-to-one; ∞ if none). */
  dist: Float64Array;
  partner: Int32Array;
  fObs: number; fChance: number; fCorr: number;
  rhoB: number;
}

export function distances(sc: ObjScene, dMaxNm: number, oneToOne: boolean): DistanceResult {
  const ax = sc.detA.map(s => s.x), ay = sc.detA.map(s => s.y), bx = sc.detB.map(s => s.x), by = sc.detB.map(s => s.y);
  const dMax = dMaxNm / PIXEL_NM;
  const r = oneToOne ? pairOneToOne(ax, ay, bx, by, dMax) : (() => { const n = crossNearest(ax, ay, bx, by); return { partner: n.idx, dist: n.dist }; })();
  const rhoB = sc.detB.length / sc.area;
  const fObs = fractionWithin(r.dist, dMax), fChance = chanceFraction(rhoB, dMax);
  return { dist: r.dist, partner: r.partner, fObs, fChance, fCorr: correctedFraction(fObs, fChance), rhoB };
}

/** Pixel Pearson r inside the cells (for contrast with the object numbers). */
export const pixelR = (sc: ObjScene): number => pearsonMasked(sc.a, sc.b, sc.mask);

/** Cross L(r) − r in nm for radii in nm, observed and with 39 random relocations of B in the mask. */
export function crossL(sc: ObjScene, radiiNm: number[], nSim = 39, seed = 5): { obs: Float64Array; lo: Float64Array; hi: Float64Array } {
  const rad = radiiNm.map(r => r / PIXEL_NM);
  const ax = sc.detA.map(s => s.x), ay = sc.detA.map(s => s.y), bx = sc.detB.map(s => s.x), by = sc.detB.map(s => s.y);
  const toNm = (v: Float64Array) => v.map(x => x * PIXEL_NM);
  const obs = toNm(crossLminusR(crossK(ax, ay, bx, by, rad, sc.area), rad));
  const env = envelope(r => {
    const p = randomPointsInMask(bx.length, sc.mask, S, S, r, 2);
    return toNm(crossLminusR(crossK(ax, ay, p.x, p.y, rad, sc.area), rad));
  }, nSim, seed);
  return { obs, lo: env.lo, hi: env.hi };
}

/** For truly paired spots that were detected in both channels: measured distances (nm), the
 *  localisation errors per axis (nm, robust sd over detected spots, from the known truth), and
 *  the mean distance predicted from the true distance and those errors (Rice mean). */
export function pairedDistances(sc: ObjScene): { measured: number[]; sigA: number; sigB: number; predicted: number } {
  const tol = 1.5;
  const near = (det: Spot[], x: number, y: number) => {
    let best = -1, bd = tol;
    det.forEach((s, k) => { const d = Math.hypot(s.x - x, s.y - y); if (d < bd) { bd = d; best = k; } });
    return best;
  };
  const sh = sc.o.shiftNm / PIXEL_NM;
  const measured: number[] = [], errA: number[] = [], errB: number[] = [];
  sc.truth.ax.forEach((x, i) => {
    const ka = near(sc.detA, x, sc.truth.ay[i]);
    if (ka >= 0) errA.push(sc.detA[ka].x - x, sc.detA[ka].y - sc.truth.ay[i]);
    const j = sc.truth.partner[i];
    if (j < 0) return;
    const kb = near(sc.detB, sc.truth.bx[j] + sh, sc.truth.by[j]);
    if (kb >= 0) errB.push(sc.detB[kb].x - sc.truth.bx[j] - sh, sc.detB[kb].y - sc.truth.by[j]);
    if (ka >= 0 && kb >= 0) measured.push(PIXEL_NM * Math.hypot(sc.detA[ka].x - sc.detB[kb].x, sc.detA[ka].y - sc.detB[kb].y));
  });
  // robust per-axis sd of the localisation errors (1.4826 × median absolute error)
  const robust = (e: number[]) => (e.length ? PIXEL_NM * 1.4826 * median(e.map(Math.abs)) : NaN);
  const sigA = robust(errA), sigB = robust(errB);
  const dEff = Math.hypot(sc.o.dNm, 0);
  return { measured, sigA, sigB, predicted: meanMeasuredDistance(dEff, sigA, sigB) };
}

// ---------------------------------------------------------------------------------------
// Larger objects: discs in A and B, thresholded and labelled

export interface BlobOptions { rA: number; rB: number; dPx: number; pair: number; seed?: number }

export interface BlobScene { labA: Int32Array; nA: number; labB: Int32Array; nB: number; anyFrac: number; halfFrac: number; centroidFrac: number; rel: ReturnType<typeof objectRelations> }

/** 16 A discs of radius rA; a fraction `pair` with a B disc of radius rB at distance dPx; plus 6 extra B discs. */
export function makeBlobScene(o: BlobOptions): BlobScene {
  const r = rng(o.seed ?? 21), m = mask();
  const A = new Uint8Array(S * S), B = new Uint8Array(S * S);
  const disc = (img: Uint8Array, x: number, y: number, rad: number) => {
    for (let yy = Math.floor(y - rad); yy <= Math.ceil(y + rad); yy++) for (let xx = Math.floor(x - rad); xx <= Math.ceil(x + rad); xx++)
      if (xx >= 0 && yy >= 0 && xx < S && yy < S && (xx - x) ** 2 + (yy - y) ** 2 <= rad * rad) img[yy * S + xx] = 1;
  };
  const centres: [number, number][] = [];
  const free = (x: number, y: number, sep: number) => centres.every(([cx, cy]) => Math.hypot(cx - x, cy - y) > sep);
  const sep = 2 * Math.max(o.rA, o.rB) + o.dPx + 3;
  let guard = 0;
  while (centres.length < 16 && guard++ < 20000) {
    const p = randomPointsInMask(1, m, S, S, r, 8);
    if (free(p.x[0], p.y[0], sep)) centres.push([p.x[0], p.y[0]]);
  }
  for (const [x, y] of centres) {
    disc(A, x, y, o.rA);
    if (r.unif() < o.pair) { const th = 2 * Math.PI * r.unif(); disc(B, x + o.dPx * Math.cos(th), y + o.dPx * Math.sin(th), o.rB); }
  }
  for (let k = 0; k < 6; k++) { const p = randomPointsInMask(1, m, S, S, r, 8); if (free(p.x[0], p.y[0], sep)) disc(B, p.x[0], p.y[0], o.rB); }
  const la = label2d(A, S, S, 8), lb = label2d(B, S, S, 8);
  const rel = objectRelations(la.data, la.count, lb.data, S);
  const n = Math.max(1, rel.length);
  return {
    labA: la.data, nA: la.count, labB: lb.data, nB: lb.count, rel,
    anyFrac: rel.filter(q => q.overlapPx > 0).length / n,
    halfFrac: rel.filter(q => q.overlapPx > 0.5 * q.area).length / n,
    centroidFrac: rel.filter(q => q.centroidIn).length / n,
  };
}
