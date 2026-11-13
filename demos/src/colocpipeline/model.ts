// Model of the colocpipeline demo (no DOM): the two-channel forward model and the corrections
// in pipeline order. True spot positions (two cells, pixel 65 nm, NA 1.4, A at 520 nm, B at
// 670 nm) → per-channel Gaussian PSF → chromatic shift of B → bleed-through of A into B →
// shared uneven background + offset → Poisson noise. Corrections: subtract the background
// (white top-hat), compensate (B − ŝ A), register (shift from a two-colour bead image by
// cross-correlation with an upsampled peak, then a Fourier shift of B). Readouts per stage:
// Pearson r and Manders M1/M2 with Costes thresholds inside the cells, ICQ, and the object
// fraction within dMax (with its chance level).
import { S, cellMask, spotPairScene, spotImage, addNoise, psfSigmaNm, psfFwhmNm, unevenBackground, compensate, subtractBackground, registerShift, shiftImage, beadPositions, SpotScene } from "../lib/twochannel";
import { pearsonMasked, costes, manders, icq, detectSpots, crossNearest, fractionWithin, chanceFraction, gaussianOverlap } from "../lib/coloc";

export const PIXEL_NM = 65, NA = 1.4, LAMBDA_A = 520, LAMBDA_B = 670, OFFSET = 100, BASE_BG = 3;
export const sigmaA = psfSigmaNm(NA, LAMBDA_A) / PIXEL_NM, sigmaB = psfSigmaNm(NA, LAMBDA_B) / PIXEL_NM;
export const fwhmA = psfFwhmNm(NA, LAMBDA_A);

export interface PipeOptions {
  /** True A → B distance of paired spots (nm). */
  dNm: number;
  /** Shared uneven background amplitude (photons per pixel). */
  uneven: number;
  /** Bleed-through of A into B. */
  bleed: number;
  /** Lateral chromatic shift of B (nm). */
  shiftXNm: number; shiftYNm: number;
  photons: number;
  /** Corrections. */
  subtract: boolean; tophatR: number;
  compensate: boolean; sHat: number;
  register: boolean;
  nA?: number; pair?: number; extraB?: number; seed?: number;
}

export const DEFAULT_PIPE: PipeOptions = { dNm: 0, uneven: 0, bleed: 0, shiftXNm: 0, shiftYNm: 0, photons: 600, subtract: false, tophatR: 6, compensate: false, sHat: 0, register: false, nA: 60, pair: 0.5, extraB: 30, seed: 31 };

let MASK: Uint8Array | null = null;
export const mask = (): Uint8Array => (MASK ??= cellMask());

export interface Stage { a: Float64Array; b: Float64Array }
export interface Readout { r: number; m1: number; m2: number; tA: number; tB: number; costesOk: boolean; icq: number; fObs: number; fChance: number; nA: number; nB: number }

export interface PipeResult {
  truth: SpotScene;
  ideal: Stage; measured: Stage; corrected: Stage;
  /** Shift estimated from the beads (nm) when registration is on. */
  estShiftNm: [number, number] | null;
  read: { ideal: Readout; measured: Readout; corrected: Readout };
}

/** Readouts inside the cells for one stage; offset = the camera offset still in the data. */
export function readout(st: Stage, offset: number, dMaxNm = 130): Readout {
  const m = mask();
  const r = pearsonMasked(st.a, st.b, m);
  const c = costes(st.a, st.b, m, 60);
  let tA = c.tA, tB = c.tB;
  if (!c.ok) {
    // fallback: thresholds 3 noise-sd above the offset (as a careful user would set them)
    tA = offset + 3 * Math.sqrt(BASE_BG + 1); tB = tA;
  }
  const md = manders(st.a, st.b, tA, tB, m);
  const dA = detectSpots(st.a, S, S, sigmaA, 5, m), dB = detectSpots(st.b, S, S, sigmaB, 5, m);
  const nn = crossNearest(dA.map(s => s.x), dA.map(s => s.y), dB.map(s => s.x), dB.map(s => s.y));
  let area = 0;
  for (let i = 0; i < m.length; i++) area += m[i];
  const dMax = dMaxNm / PIXEL_NM;
  return { r, m1: md.m1, m2: md.m2, tA, tB, costesOk: c.ok, icq: icq(st.a, st.b, m).icq, fObs: fractionWithin(nn.dist, dMax), fChance: chanceFraction(dB.length / area, dMax), nA: dA.length, nB: dB.length };
}

const beadCache = new Map<string, [number, number]>();
/** Shift of B against A (px) measured on a two-colour bead image with the same optics. */
export function beadShift(shiftXNm: number, shiftYNm: number): [number, number] {
  const key = `${shiftXNm},${shiftYNm}`;
  const hit = beadCache.get(key);
  if (hit) return hit;
  const p = beadPositions(S, S, 5, 5, 3);
  const a = addNoise(spotImage(p.x, p.y, S, S, sigmaA, 5000), 41, OFFSET);
  const b = addNoise(spotImage(p.x, p.y, S, S, sigmaB, 5000, shiftXNm / PIXEL_NM, shiftYNm / PIXEL_NM), 42, OFFSET);
  const t = registerShift(a, b, S, S);
  const out: [number, number] = [t.tx, t.ty];
  beadCache.set(key, out);
  return out;
}

export function scene(o: PipeOptions): SpotScene {
  return spotPairScene({ w: S, h: S, mask: mask(), nA: o.nA ?? 60, pairFraction: o.pair ?? 0.5, distance: o.dNm / PIXEL_NM, extraB: o.extraB ?? 30, seed: o.seed ?? 31 });
}

export function runPipeline(o: PipeOptions, withIdealReadout = true): PipeResult {
  const t = scene(o);
  const m = mask();
  const ea = spotImage(t.ax, t.ay, S, S, sigmaA, o.photons);
  const ebTrue = spotImage(t.bx, t.by, S, S, sigmaB, o.photons);
  // ideal: the same spots, PSF, photons and noise, without any confound
  const base = (i: number) => BASE_BG * (0.3 + 0.7 * m[i]);
  const ideal: Stage = { a: addNoise(ea.map((v, i) => v + base(i)), (o.seed ?? 31) + 1, OFFSET), b: addNoise(ebTrue.map((v, i) => v + base(i)), (o.seed ?? 31) + 2, OFFSET) };
  const eb = (o.shiftXNm || o.shiftYNm) ? spotImage(t.bx, t.by, S, S, sigmaB, o.photons, o.shiftXNm / PIXEL_NM, o.shiftYNm / PIXEL_NM) : ebTrue;
  const g = unevenBackground(S, S, 9);
  const xa = new Float64Array(S * S), xb = new Float64Array(S * S);
  for (let i = 0; i < S * S; i++) {
    const bg = base(i) + o.uneven * g[i];
    xa[i] = ea[i] + bg;
    xb[i] = eb[i] + o.bleed * ea[i] + bg;
  }
  const measured: Stage = { a: addNoise(xa, (o.seed ?? 31) + 1, OFFSET), b: addNoise(xb, (o.seed ?? 31) + 2, OFFSET) };
  let ca = measured.a, cb = measured.b, off = OFFSET;
  if (o.subtract) { ca = subtractBackground(ca, S, S, o.tophatR); cb = subtractBackground(cb, S, S, o.tophatR); off = 0; }
  if (o.compensate) cb = compensate(ca, cb, o.sHat);
  let est: [number, number] | null = null;
  if (o.register) {
    est = beadShift(o.shiftXNm, o.shiftYNm);
    cb = shiftImage(cb, S, S, -est[0], -est[1]);
  }
  const corrected: Stage = { a: ca, b: cb };
  const empty: Readout = { r: NaN, m1: NaN, m2: NaN, tA: NaN, tB: NaN, costesOk: false, icq: NaN, fObs: NaN, fChance: NaN, nA: 0, nB: 0 };
  return {
    truth: t, ideal, measured, corrected,
    estShiftNm: est ? [est[0] * PIXEL_NM, est[1] * PIXEL_NM] : null,
    read: { ideal: withIdealReadout ? readout(ideal, OFFSET) : empty, measured: readout(measured, OFFSET), corrected: readout(corrected, off) },
  };
}

/** Resolution sweep: pixel r inside the cells (simulated, all spots paired, no confounds) and
 *  the object fraction within dMax (nearest neighbour) against the true pair distance, with
 *  the analytic Gaussian overlap for comparison. */
export function sweep(distancesNm: number[], dMaxNm = 130): { d: number[]; r: number[]; frac: number[]; analytic: (d: number) => number } {
  const r: number[] = [], frac: number[] = [], m = mask();
  for (const d of distancesNm) {
    const o = { ...DEFAULT_PIPE, dNm: d, pair: 1, extraB: 0 };
    const t = scene(o);
    const bg = (i: number) => BASE_BG * (0.3 + 0.7 * m[i]);
    const a = addNoise(spotImage(t.ax, t.ay, S, S, sigmaA, o.photons).map((v, i) => v + bg(i)), 32, OFFSET);
    const b = addNoise(spotImage(t.bx, t.by, S, S, sigmaB, o.photons).map((v, i) => v + bg(i)), 33, OFFSET);
    r.push(pearsonMasked(a, b, m));
    const dA = detectSpots(a, S, S, sigmaA, 5, m), dB = detectSpots(b, S, S, sigmaB, 5, m);
    const nn = crossNearest(dA.map(s => s.x), dA.map(s => s.y), dB.map(s => s.x), dB.map(s => s.y));
    frac.push(fractionWithin(nn.dist, dMaxNm / PIXEL_NM));
  }
  return { d: distancesNm, r, frac, analytic: (d: number) => gaussianOverlap(d / PIXEL_NM, sigmaA, sigmaB) };
}
