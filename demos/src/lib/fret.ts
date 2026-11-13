// FRET (owner: FRET day, plan/fret-day.md). No DOM.
//
// Sections: physics (efficiency curve, apparent distance, orientation factor, Förster radius
// from parametric spectra, the rate picture shared with the FLIM day) · sensitised emission
// (the three-image forward model, its correction, normalised indices, noise) · acceptor
// photobleaching (bleach series, corrections) · ratios (ratio of sums, mean of ratios,
// delta-method CV and bias, calibration to a concentration) · seeded scenes for the demos
// (a field of cells of four classes, a bleach experiment, a biosensor time-lapse).
//
// Images are flat Float32Arrays, row-major, width × height (as lib/unmix.ts maps).
// Notation: E efficiency, r distance, R0 Förster radius, κ² orientation factor; images
// IDD (donor excitation, donor emission), IDA (donor excitation, acceptor emission: the
// "FRET channel"), IAA (acceptor excitation, acceptor emission); bleed-through d,
// cross-excitation a, instrument factor G.
import { rng, poisson, gauss, Rng } from "./random";
import { Dye, emission } from "./unmix";
import { nuclei } from "./synth";

type A = ArrayLike<number>;

// =======================================================================================
// Physics

/** FRET efficiency E = 1 / (1 + (r/R0)^6). */
export const efficiency = (r: number, R0: number): number => 1 / (1 + (r / R0) ** 6);

/** Apparent distance r = R0 (1/E − 1)^(1/6) (∞ for E ≤ 0, 0 for E ≥ 1). */
export function distanceFromE(E: number, R0: number): number {
  if (!(E > 0)) return Infinity;
  if (E >= 1) return 0;
  return R0 * Math.pow(1 / E - 1, 1 / 6);
}

/** dE/dr = −(6/r) E (1 − E). */
export function dEdr(r: number, R0: number): number {
  const E = efficiency(r, R0);
  return -6 / r * E * (1 - E);
}

export type V3 = [number, number, number];
const dot3 = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const unit3 = (a: V3): V3 => { const n = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };

/** Orientation factor κ² = (μD·μA − 3 (μD·r̂)(μA·r̂))² for dipole directions μD, μA and the
 *  donor→acceptor direction r (all normalised here). 4 collinear, 1 parallel side by side,
 *  0 e.g. perpendicular dipoles both perpendicular to r. */
export function kappa2(muD: V3, muA: V3, r: V3): number {
  const d = unit3(muD), a = unit3(muA), e = unit3(r);
  const k = dot3(d, a) - 3 * dot3(d, e) * dot3(a, e);
  return k * k;
}

/** κ² for dipoles in the plane (angles in radians from the x axis; r along x). */
export const kappa2Planar = (thetaD: number, thetaA: number): number =>
  kappa2([Math.cos(thetaD), Math.sin(thetaD), 0], [Math.cos(thetaA), Math.sin(thetaA), 0], [1, 0, 0]);

/** Förster radius in nm from the practical formula R0 = 0.0211 (κ² n⁻⁴ QD J)^(1/6),
 *  J in M⁻¹ cm⁻¹ nm⁴. */
export const forsterRadius = (QD: number, k2: number, n: number, J: number): number =>
  0.0211 * Math.pow(Math.max(0, k2 * QD * J / n ** 4), 1 / 6);

/** An absorption (excitation) spectrum: a split Gaussian with peak `peak`, right standard
 *  deviation `width` and a long blue tail on the left (sd `width * tail`); relative, 1 at
 *  the peak. The mirror image of `emission` of lib/unmix.ts. */
export interface Absorber { peak: number; width: number; tail: number }
export function absorption(d: Absorber, l: number): number {
  const s = l > d.peak ? d.width : d.width * d.tail, t = (l - d.peak) / s;
  return Math.exp(-0.5 * t * t);
}

/** A fluorophore for the FRET demos: excitation shape, emission shape (a `Dye` of
 *  lib/unmix.ts), peak extinction coefficient ε (M⁻¹cm⁻¹) and quantum yield Q. */
export interface Fluor { name: string; ex: Absorber; em: Dye; eps: number; Q: number }

/** Example pair: a cyan donor and a yellow acceptor (shapes only resemble real proteins). */
export const CYAN_DONOR: Fluor = { name: "cyan donor", ex: { peak: 434, width: 14, tail: 1.6 }, em: { name: "donor", peak: 476, width: 13, tail: 2.0 }, eps: 32500, Q: 0.40 };
export const YELLOW_ACCEPTOR: Fluor = { name: "yellow acceptor", ex: { peak: 514, width: 11, tail: 2.2 }, em: { name: "acceptor", peak: 527, width: 11, tail: 2.0 }, eps: 83000, Q: 0.61 };

/** Overlap integral J = ∫ F_D(λ) ε_A(λ) λ⁴ dλ (M⁻¹ cm⁻¹ nm⁴), F_D normalised to unit area,
 *  ε_A = epsA × the absorption shape. Midpoint rule, `step` nm over 300..900 nm. */
export function overlapIntegral(donorEm: Dye, acc: Absorber, epsA: number, step = 0.5): number {
  let area = 0, J = 0;
  for (let l = 300 + step / 2; l < 900; l += step) {
    const f = emission(donorEm, l);
    area += f * step;
    J += f * epsA * absorption(acc, l) * l ** 4 * step;
  }
  return area > 0 ? J / area : 0;
}

/** Donor lifetime with an acceptor at efficiency E: τ_DA = τ_D (1 − E) (rate picture). */
export const lifetimeWithAcceptor = (tauD: number, E: number): number => tauD * (1 - E);

/** Efficiency from rates: E = kT / (kT + kD), kT = kD (R0/r)⁶. */
export function efficiencyFromRates(r: number, R0: number, tauD: number): { kD: number; kT: number; E: number; tauDA: number } {
  const kD = 1 / tauD, kT = kD * (R0 / r) ** 6;
  return { kD, kT, E: kT / (kT + kD), tauDA: 1 / (kD + kT) };
}

// =======================================================================================
// Sensitised emission (three images)

export interface ThreeCubeParams {
  ND: number; NA: number;   // donor and acceptor amounts
  E: number;                // efficiency of a paired donor
  fPaired: number;          // fraction of donors that have an acceptor
  d: number; a: number;     // bleed-through, cross-excitation
  G: number;                // sensitised acceptor signal per unit of lost donor signal
  cDD?: number; cAA?: number;  // brightness constants (default 1)
}
export interface Three { IDD: number; IDA: number; IAA: number; Fc: number }

/** Noise-free forward model: IDD = cDD ND (1 − f E); IAA = cAA NA;
 *  Fc = G cDD ND f E; IDA = d IDD + a IAA + Fc. */
export function threeCube(p: ThreeCubeParams): Three {
  const cDD = p.cDD ?? 1, cAA = p.cAA ?? 1, fe = p.fPaired * p.E;
  const IDD = cDD * p.ND * (1 - fe), IAA = cAA * p.NA, Fc = p.G * cDD * p.ND * fe;
  return { IDD, IAA, Fc, IDA: p.d * IDD + p.a * IAA + Fc };
}

/** Corrected sensitised emission Fc = IDA − d IDD − a IAA. */
export const correctFc = (IDA: number, IDD: number, IAA: number, d: number, a: number): number => IDA - d * IDD - a * IAA;

/** Apparent efficiency E_app = Fc / (Fc + G IDD) (= f E for the model above). */
export function eFret(Fc: number, IDD: number, G: number): number {
  const den = Fc + G * IDD;
  return den !== 0 ? Fc / den : NaN;
}

/** Symmetric index Fc / √(IDD IAA). */
export const nfretIndex = (Fc: number, IDD: number, IAA: number): number => (IDD > 0 && IAA > 0 ? Fc / Math.sqrt(IDD * IAA) : NaN);

/** Per-acceptor index Fc / IAA. */
export const perAcceptorIndex = (Fc: number, IAA: number): number => (IAA > 0 ? Fc / IAA : NaN);

/** Stoichiometry S = (G IDD + Fc) / (G IDD + Fc + IAA/k), k = acceptor-excitation factor. */
export function stoichiometry(Fc: number, IDD: number, IAA: number, G: number, k: number): number {
  const D = G * IDD + Fc;
  return D / (D + IAA / k);
}

/** Poisson variance of Fc from independent images: Var(IDA) + d² Var(IDD) + a² Var(IAA)
 *  with Var = mean (photon counts). */
export const fcVariance = (IDA: number, IDD: number, IAA: number, d: number, a: number): number => IDA + d * d * IDD + a * a * IAA;

/** Slope through the origin Σxy/Σx² over the pixels where mask ≠ 0 (all if no mask). */
export function slopeThroughOriginMasked(x: A, y: A, mask?: A): number {
  let sxy = 0, sxx = 0;
  for (let i = 0; i < x.length; i++) if (!mask || mask[i]) { sxy += x[i] * y[i]; sxx += x[i] * x[i]; }
  return sxx > 0 ? sxy / sxx : 0;
}

// =======================================================================================
// Acceptor photobleaching

/** E = 1 − pre/post (donor before and after bleaching the acceptor). */
export const eFromBleach = (pre: number, post: number): number => (post !== 0 ? 1 - pre / post : NaN);

/** Observed E when only a fraction b of acceptors is bleached: b E / (1 − (1 − b) E). */
export const eObservedIncomplete = (E: number, b: number): number => b * E / (1 - (1 - b) * E);

/** Inverse: E = E_obs / (b + (1 − b) E_obs). */
export const correctIncomplete = (Eobs: number, b: number): number => Eobs / (b + (1 - b) * Eobs);

/** Donor-bleaching correction: divide the post value by the control's remaining fraction. */
export const correctDonorBleach = (pre: number, post: number, controlPre: number, controlPost: number): number =>
  eFromBleach(pre, post * controlPre / controlPost);

/** Delta-method standard deviation of E = 1 − P/Q for Poisson P, Q: (P/Q) √(1/P + 1/Q). */
export const bleachSigmaE = (P: number, Q: number): number => (P / Q) * Math.sqrt(1 / P + 1 / Q);

/** Donor intensity per unit donor (relative) with a fraction `accLeft` of acceptors left,
 *  paired fraction f and efficiency E: 1 − f E accLeft. */
export const donorRelative = (E: number, f: number, accLeft: number): number => 1 - f * E * accLeft;

/** Time course of a bleach experiment in the bleached region: per pulse k = 0..n the
 *  acceptor fraction left (1 − perPulse)^k, the donor relative to the fully quenched-free
 *  donor (times the donor's own bleaching (1 − donorLoss)^k). */
export function bleachSeries(o: { E: number; f: number; pulses: number; perPulse: number; donorLoss: number }):
  { acc: number[]; donor: number[]; control: number[] } {
  const acc: number[] = [], donor: number[] = [], control: number[] = [];
  for (let k = 0; k <= o.pulses; k++) {
    const left = (1 - o.perPulse) ** k, dl = (1 - o.donorLoss) ** k;
    acc.push(left);
    donor.push(donorRelative(o.E, o.f, left) * dl);
    control.push(donorRelative(o.E, o.f, 1) * dl);
  }
  return { acc, donor, control };
}

// =======================================================================================
// Ratios

/** Σ a / Σ d over the mask (all pixels if none). */
export function ratioOfSums(a: A, d: A, mask?: A): number {
  let sa = 0, sd = 0;
  for (let i = 0; i < a.length; i++) if (!mask || mask[i]) { sa += a[i]; sd += d[i]; }
  return sd !== 0 ? sa / sd : NaN;
}

/** Mean of a/d over the mask, skipping pixels with d ≤ 0 (they have no finite ratio). */
export function meanOfRatios(a: A, d: A, mask?: A): number {
  let s = 0, n = 0;
  for (let i = 0; i < a.length; i++) if ((!mask || mask[i]) && d[i] > 0) { s += a[i] / d[i]; n++; }
  return n ? s / n : NaN;
}

/** Delta-method CV of A/D: √(varA/μA² + varD/μD² − 2 cov/(μA μD)); Poisson by default. */
export function ratioCVPredicted(muA: number, muD: number, varA = muA, varD = muD, cov = 0): number {
  return Math.sqrt(Math.max(0, varA / (muA * muA) + varD / (muD * muD) - 2 * cov / (muA * muD)));
}

/** Second-order mean of A/D: (μA/μD)(1 + varD/μD² − cov/(μA μD)). */
export function ratioMeanPredicted(muA: number, muD: number, varD = muD, cov = 0): number {
  return (muA / muD) * (1 + varD / (muD * muD) - cov / (muA * muD));
}

/** Concentration from a ratio: c = Kd (R − Rmin)/(Rmax − R) (NaN outside (Rmin, Rmax) is
 *  returned as 0 below and ∞ above). */
export function calibrate(R: number, Rmin: number, Rmax: number, Kd: number): number {
  if (R <= Rmin) return 0;
  if (R >= Rmax) return Infinity;
  return Kd * (R - Rmin) / (Rmax - R);
}

/** Ratio for a concentration: R = (Rmin Kd + Rmax c)/(Kd + c) (linear in fraction bound). */
export const calibrateInverse = (c: number, Rmin: number, Rmax: number, Kd: number): number => (Rmin * Kd + Rmax * c) / (Kd + c);

/** dc/dR = Kd (Rmax − Rmin)/(Rmax − R)². */
export const dcdR = (R: number, Rmin: number, Rmax: number, Kd: number): number => Kd * (Rmax - Rmin) / (Rmax - R) ** 2;

// =======================================================================================
// Small image helpers

/** Integer shift of a map (x right, y down), zero outside. */
export function shiftMap(a: A, w: number, h: number, dx: number, dy: number, fill = 0): Float32Array {
  const out = new Float32Array(w * h).fill(fill);
  for (let y = 0; y < h; y++) {
    const sy = y - dy;
    if (sy < 0 || sy >= h) continue;
    for (let x = 0; x < w; x++) { const sx = x - dx; if (sx >= 0 && sx < w) out[y * w + x] = a[sy * w + sx]; }
  }
  return out;
}

/** Sum of m × m blocks (binning); output ⌊w/m⌋ × ⌊h/m⌋. */
export function binSum(a: A, w: number, h: number, m: number): { data: Float32Array; w: number; h: number } {
  const W = Math.floor(w / m), H = Math.floor(h / m), out = new Float32Array(W * H);
  for (let y = 0; y < H * m; y++) for (let x = 0; x < W * m; x++) out[Math.floor(y / m) * W + Math.floor(x / m)] += a[y * w + x];
  return { data: out, w: W, h: H };
}

/** Nearest-neighbour upsampling of an m-binned map back to w × h (for display). */
export function unbin(a: A, W: number, H: number, m: number, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const X = Math.min(W - 1, Math.floor(x / m)), Y = Math.min(H - 1, Math.floor(y / m));
    out[y * w + x] = a[Y * W + X];
  }
  return out;
}

/** Poisson count, fast: exact (Knuth) below λ = 10, normal approximation rounded and
 *  clipped at 0 above (the error is far below what a demo can show). */
export function fastPoisson(r: Rng, lambda: number): number {
  if (!(lambda > 0)) return 0;
  if (lambda < 10) {
    const L = Math.exp(-lambda);
    let k = 0, p = 1;
    do { k++; p *= r.unif(); } while (p > L);
    return k - 1;
  }
  return Math.max(0, Math.round(lambda + Math.sqrt(lambda) * r.norm()));
}

/** Poisson counts of photons × expected + offset, seeded. */
export function poissonImage(expected: A, photons: number, offset: number, r: Rng): Float32Array {
  const out = new Float32Array(expected.length);
  for (let i = 0; i < out.length; i++) out[i] = poisson(r, Math.max(0, photons * expected[i])) + offset;
  return out;
}

// Cell geometry from synth.nuclei: label image plus a smooth "thickness" (a dome inside each
// ellipse, 0.35 at the rim and 1 at the centre) used to modulate every channel.
interface CellGeom { w: number; h: number; labels: Int32Array; count: number; thick: Float32Array; cx: number[]; cy: number[] }
function cells(w: number, h: number, count: number, radius: [number, number], seed: number): CellGeom {
  const nu = nuclei({ width: w, height: h, count, radius, touching: 0, elongation: 0.5, noise: 0, texture: 0, blur: 0, seed });
  const thick = new Float32Array(w * h);
  for (const [k, e] of nu.ellipses.entries()) {
    const c = Math.cos(e.angle), s = Math.sin(e.angle);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (nu.labels[y * w + x] !== k + 1) continue;
      const dx = x - e.x, dy = y - e.y, u = (dx * c + dy * s) / e.a, v = (-dx * s + dy * c) / e.b;
      const q = Math.min(1, u * u + v * v);
      thick[y * w + x] = 0.35 + 0.65 * Math.sqrt(1 - q);
    }
  }
  return { w, h, labels: nu.labels, count: nu.count, thick, cx: nu.ellipses.map(e => e.x), cy: nu.ellipses.map(e => e.y) };
}

// =======================================================================================
// Scene 1: a field of cells for the three-image correction

export type CellClass = "donor" | "acceptor" | "free" | "fusion" | "binding";
export const CLASS_NAMES: Record<CellClass, string> = { donor: "donor only", acceptor: "acceptor only", free: "free D + A (no FRET)", fusion: "D–A fusion", binding: "binding (half paired)" };

export interface FieldOptions {
  size?: number;          // image width = height (default 128)
  seed?: number;
  E?: number;             // efficiency of the fusion (default 0.3)
  Ebind?: number;         // efficiency of a bound pair in "binding" cells (default 0.4)
  fBind?: number;         // paired fraction in "binding" cells (default 0.5)
  extraAcceptor?: number; // free acceptor co-expressed in fusion cells, × the fusion amount (per cell × u in [0, 1])
  binding?: boolean;      // include binding cells (default true)
  d?: number; a?: number; G?: number;  // true instrument constants (default 0.5, 0.1, 2)
}
export interface Field {
  w: number; h: number;
  labels: Int32Array; count: number;
  cls: CellClass[];                    // per cell (index = label − 1)
  trueE: number[];                     // per cell: E of a paired donor (0 if none)
  trueF: number[];                     // per cell: paired fraction
  ND: Float32Array; NA: Float32Array;  // per pixel amounts (photon units at photons = 1)
  IDD: Float32Array; IDA: Float32Array; IAA: Float32Array;  // expected images (no offset)
  d: number; a: number; G: number;
}

/** A seeded field of cells, about two of each class, with per-cell expression levels. */
export function fretField(o: FieldOptions = {}): Field {
  const n = o.size ?? 128, seed = o.seed ?? 5, r = rng(seed * 31 + 7);
  const withBinding = o.binding ?? true;
  const geom = cells(n, n, withBinding ? 11 : 9, [9, 13], seed);
  const order: CellClass[] = withBinding
    ? ["donor", "acceptor", "free", "fusion", "binding", "fusion", "donor", "acceptor", "free", "fusion", "binding"]
    : ["donor", "acceptor", "free", "fusion", "fusion", "donor", "acceptor", "free", "fusion"];
  const d = o.d ?? 0.5, a = o.a ?? 0.1, G = o.G ?? 2, E = o.E ?? 0.3, Eb = o.Ebind ?? 0.4, fb = o.fBind ?? 0.5;
  const P = n * n, ND = new Float32Array(P), NA = new Float32Array(P);
  const cls: CellClass[] = [], trueE: number[] = [], trueF: number[] = [];
  const exD: number[] = [], exA: number[] = [];
  for (let k = 0; k < geom.count; k++) {
    const c = order[k % order.length];
    cls.push(c);
    const lvl = 0.5 + r.unif();              // expression level 0.5..1.5
    const u = r.unif();
    let nd = 0, na = 0, e = 0, f = 0;
    if (c === "donor") nd = lvl;
    else if (c === "acceptor") na = lvl;
    else if (c === "free") { nd = lvl; na = lvl * (0.7 + 0.6 * r.unif()); }
    else if (c === "fusion") { nd = lvl; na = lvl * (1 + (o.extraAcceptor ?? 0) * u); e = E; f = 1; }
    else { nd = lvl; na = lvl * (0.8 + 0.4 * r.unif()); e = Eb; f = fb; }
    exD.push(nd); exA.push(na); trueE.push(e); trueF.push(f);
  }
  const IDD = new Float32Array(P), IDA = new Float32Array(P), IAA = new Float32Array(P);
  for (let i = 0; i < P; i++) {
    const L = geom.labels[i];
    if (!L) continue;
    const k = L - 1, t = geom.thick[i];
    ND[i] = exD[k] * t; NA[i] = exA[k] * t;
    const v = threeCube({ ND: ND[i], NA: NA[i], E: trueE[k], fPaired: trueF[k], d, a, G });
    IDD[i] = v.IDD; IDA[i] = v.IDA; IAA[i] = v.IAA;
  }
  return { w: n, h: n, labels: geom.labels, count: geom.count, cls, trueE, trueF, ND, NA, IDD, IDA, IAA, d, a, G };
}

/** Noisy measured images of a field: Poisson(photons × expected) + offset, IDA optionally
 *  shifted by (shift, 0) px (misregistration). */
export function measureField(f: Field, o: { photons: number; offset: number; seed?: number; shift?: number; noise?: boolean }):
  { IDD: Float32Array; IDA: Float32Array; IAA: Float32Array } {
  const r = rng((o.seed ?? 1) * 7 + 3), noisy = o.noise ?? true;
  const img = (e: Float32Array) => noisy ? poissonImage(e, o.photons, o.offset, r) : Float32Array.from(e, v => o.photons * v + o.offset);
  const IDD = img(f.IDD), IAA = img(f.IAA);
  let IDA = img(f.IDA);
  if (o.shift) IDA = shiftMap(IDA, f.w, f.h, o.shift, 0, o.offset);
  return { IDD, IDA, IAA };
}

// =======================================================================================
// Scene 2: acceptor photobleaching

export interface BleachOptions {
  size?: number; seed?: number;
  E?: number; f?: number;          // efficiency and paired fraction in the FRET cells
  pulses?: number;                 // number of bleach pulses
  perPulse?: number;               // fraction of acceptors bleached per pulse in the region (default 0.3)
  donorLoss?: number;              // donor bleaching per pulse, whole field (default 0)
  rect?: [number, number, number, number];   // bleach region x0, y0, x1, y1 (pixels, exclusive end)
  photons?: number;                // donor photons at full thickness without FRET (default 600)
  drift?: [number, number];        // shift of the post images (px)
  photoconvert?: number;           // donor-channel signal per bleached acceptor (default 0)
  noise?: boolean;
}
export interface BleachScene {
  w: number; h: number; labels: Int32Array; count: number;
  fretCells: number[];             // labels of the FRET cells
  controlCell: number;             // label of the second FRET cell (outside the default region)
  accOnlyCell: number;             // label of an acceptor-only cell
  region: Uint8Array;              // bleach region ∩ cells
  control: Uint8Array;             // non-bleached part of the FRET cells (control region)
  donorPre: Float32Array; donorPost: Float32Array;
  accPre: Float32Array; accPost: Float32Array;
  series: { regionDonor: number[]; regionAcc: number[]; controlDonor: number[]; controlAcc: number[] };  // per pulse, mean expected
}

/** A seeded bleach experiment: two FRET cells (one partly inside the bleach region) and an
 *  acceptor-only control cell, bleached as a whole (its donor channel shows only noise,
 *  unless bleached acceptors photoconvert into a donor-coloured species). */
export function bleachScene(o: BleachOptions = {}): BleachScene {
  const n = o.size ?? 128, seed = o.seed ?? 4;
  const geom = cells(n, n, 3, [17, 22], seed);
  const E = o.E ?? 0.3, f = o.f ?? 1, pulses = o.pulses ?? 12, pp = o.perPulse ?? 0.3, dl = o.donorLoss ?? 0;
  const ph = o.photons ?? 600;
  // cell 1 and 2 FRET, cell 3 acceptor only
  const fretCells = [1, 2].filter(k => k <= geom.count), accOnly = geom.count >= 3 ? 3 : 0;
  let rect = o.rect;
  if (!rect) {   // default: a box over the right half of cell 1
    const cx = Math.round(geom.cx[0]), cy = Math.round(geom.cy[0]);
    rect = [cx - 2, cy - 14, cx + 18, cy + 14];
  }
  const P = n * n, region = new Uint8Array(P), control = new Uint8Array(P);
  const inRect = (x: number, y: number) => x >= rect![0] && x < rect![2] && y >= rect![1] && y < rect![3];
  const left = (1 - pp) ** pulses, dfac = (1 - dl) ** pulses;
  const donorPre = new Float32Array(P), donorPost = new Float32Array(P), accPre = new Float32Array(P), accPost = new Float32Array(P);
  const sums = { rd: new Float64Array(pulses + 1), ra: new Float64Array(pulses + 1), cd: new Float64Array(pulses + 1), ca: new Float64Array(pulses + 1) };
  let nr = 0, nc = 0;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const i = y * n + x, L = geom.labels[i];
    if (!L) continue;
    const t = geom.thick[i];
    const isFret = fretCells.includes(L), isAcc = L === accOnly;
    const inR = inRect(x, y) || isAcc;   // the acceptor-only control cell is bleached as a whole
    if (inR) region[i] = 1;
    if (isFret && !inR) control[i] = 1;
    const D = isFret ? t : 0, Aa = (isFret || isAcc) ? t : 0;
    const accLeft = inR ? left : 1;
    donorPre[i] = D * donorRelative(E, isFret ? f : 0, 1);
    donorPost[i] = D * donorRelative(E, isFret ? f : 0, accLeft) * dfac + (o.photoconvert ?? 0) * Aa * (1 - accLeft);
    accPre[i] = Aa; accPost[i] = Aa * accLeft;
    if (isFret) {
      for (let k = 0; k <= pulses; k++) {
        const lk = inR ? (1 - pp) ** k : 1, dk = (1 - dl) ** k;
        const dv = D * donorRelative(E, f, lk) * dk, av = Aa * lk;
        if (inR) { sums.rd[k] += dv; sums.ra[k] += av; } else { sums.cd[k] += dv; sums.ca[k] += av; }
      }
      if (inR) nr++; else nc++;
    }
  }
  const r = rng(seed * 13 + 1), noisy = o.noise ?? true;
  const meas = (e: Float32Array) => noisy ? poissonImage(e, ph, 0, r) : Float32Array.from(e, v => ph * v);
  const dPre = meas(donorPre), aPre = meas(accPre);
  let dPost = meas(donorPost), aPost = meas(accPost);
  const [sx, sy] = o.drift ?? [0, 0];
  if (sx || sy) { dPost = shiftMap(dPost, n, n, sx, sy); aPost = shiftMap(aPost, n, n, sx, sy); }
  const per = (s: Float64Array, m: number) => Array.from(s, v => (m ? ph * v / m : 0));
  return {
    w: n, h: n, labels: geom.labels, count: geom.count, fretCells, controlCell: fretCells[1] ?? 0, accOnlyCell: accOnly,
    region, control, donorPre: dPre, donorPost: dPost, accPre: aPre, accPost: aPost,
    series: { regionDonor: per(sums.rd, nr), regionAcc: per(sums.ra, nr), controlDonor: per(sums.cd, nc), controlAcc: per(sums.ca, nc) },
  };
}

// =======================================================================================
// Scene 3: a single-chain biosensor time-lapse

export interface SensorOptions {
  size?: number; frames?: number; seed?: number;
  Rmin?: number; Rmax?: number; Kd?: number;   // sensor (default 0.5, 2.5, 200 nM)
}
export interface SensorScene {
  w: number; h: number; frames: number;
  labels: Int32Array; count: number;
  thick: Float32Array;         // thickness × expression per pixel (multiplies both channels)
  conc: Float32Array[];        // per frame: concentration (nM) per pixel (0 outside cells)
  Rmin: number; Rmax: number; Kd: number;
  roles: string[];             // per cell
}

/** Three cells: a wave of high concentration crossing cell 1, a steady gradient in cell 2,
 *  a constant control cell 3. Thickness and expression differ per cell (the thick nucleus
 *  region is brighter in both channels). Concentrations only; channels from sensorFrame. */
export function sensorScene(o: SensorOptions = {}): SensorScene {
  const n = o.size ?? 128, T = o.frames ?? 40, seed = o.seed ?? 9;
  const geom = cells(n, n, 3, [20, 26], seed);
  const r = rng(seed + 5);
  const expr = [1.4, 0.7, 1.0].map(v => v * (0.9 + 0.2 * r.unif()));
  const thick = new Float32Array(n * n);
  for (let i = 0; i < n * n; i++) {
    const L = geom.labels[i];
    if (!L) continue;
    // a thick "nucleus" bump near the cell centre: 1.6× the dome
    const x = i % n, y = (i - x) / n, k = L - 1;
    const dn = Math.hypot(x - geom.cx[k] - 4, y - geom.cy[k] + 3);
    thick[i] = expr[k] * geom.thick[i] * (1 + 0.8 * Math.exp(-(dn * dn) / (2 * 7 * 7)));
  }
  const conc: Float32Array[] = [];
  for (let t = 0; t < T; t++) {
    const c = new Float32Array(n * n);
    for (let i = 0; i < n * n; i++) {
      const L = geom.labels[i];
      if (!L) continue;
      const x = i % n, y = (i - x) / n, k = L - 1;
      if (k === 0) {        // wave from left to right, passing between frames 8 and 32
        const front = geom.cx[0] - 30 + 60 * (t - 8) / 24;
        const u = x - front;
        c[i] = 80 + 2500 * Math.exp(-(u * u) / (2 * 8 * 8));
      } else if (k === 1) { // steady gradient top → bottom
        c[i] = 60 + 600 * Math.min(1, Math.max(0, (y - (geom.cy[1] - 25)) / 50));
      } else c[i] = 150;    // control
    }
    conc.push(c);
  }
  return { w: n, h: n, frames: T, labels: geom.labels, count: geom.count, thick, conc, Rmin: o.Rmin ?? 0.5, Rmax: o.Rmax ?? 2.5, Kd: o.Kd ?? 200, roles: ["wave", "gradient", "control"] };
}

export interface SensorFrameOptions {
  photons: number;       // donor photons per pixel at thickness 1
  offset: number;        // background added to both channels
  bleed?: number;        // donor bleed-through into the acceptor channel (adds bleed × D)
  accBleach?: number;    // acceptor loss per frame (fraction)
  shift?: number;        // acceptor channel shifted by this many px in x
  seed?: number; noise?: boolean;
}

/** Donor and acceptor channels of frame t: D = photons·thick, A = photons·thick·R(c)·
 *  (1 − accBleach)^t + bleed·D, Poisson + offset. */
export function sensorFrame(s: SensorScene, t: number, o: SensorFrameOptions): { D: Float32Array; A: Float32Array; Dexp: Float32Array; Aexp: Float32Array } {
  const P = s.w * s.h, Dexp = new Float32Array(P), Aexp = new Float32Array(P);
  const bl = (1 - (o.accBleach ?? 0)) ** t, c = s.conc[t];
  for (let i = 0; i < P; i++) {
    const th = s.thick[i];
    if (!th) continue;
    const R = calibrateInverse(c[i], s.Rmin, s.Rmax, s.Kd);
    Dexp[i] = o.photons * th;
    Aexp[i] = o.photons * th * R * bl + (o.bleed ?? 0) * Dexp[i];
  }
  const r = rng((o.seed ?? 1) * 1009 + t * 7919 + 17), noisy = o.noise ?? true;
  const D = new Float32Array(P), A0 = new Float32Array(P);
  for (let i = 0; i < P; i++) {
    D[i] = (noisy ? fastPoisson(r, Dexp[i] + o.offset) : Dexp[i] + o.offset);
    A0[i] = (noisy ? fastPoisson(r, Aexp[i] + o.offset) : Aexp[i] + o.offset);
  }
  const A = o.shift ? shiftMap(A0, s.w, s.h, o.shift, 0, o.offset) : A0;
  return { D, A, Dexp, Aexp };
}

// =======================================================================================
// Monte Carlo of a ratio (for rationoise)

export interface RatioMC { ratios: Float64Array; A: Float64Array; D: Float64Array }

/** n draws of (A, D): Poisson (or Gaussian with the same variance) counts with means
 *  muA, muD, each multiplied by a shared fluctuation (1 + flicker·z) when flicker > 0.
 *  `bin` pixels are summed per draw. */
export function ratioDraws(o: { muA: number; muD: number; n: number; flicker?: number; bin?: number; gaussian?: boolean; seed?: number }): RatioMC {
  const r = rng(o.seed ?? 1), n = o.n, m = Math.max(1, o.bin ?? 1), fl = o.flicker ?? 0;
  const A = new Float64Array(n), D = new Float64Array(n), ratios = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const g = fl > 0 ? Math.max(0, 1 + fl * r.norm()) : 1;
    const la = o.muA * m * g, ld = o.muD * m * g;
    const a = o.gaussian ? gauss(r, la, Math.sqrt(la)) : poisson(r, la);
    const d = o.gaussian ? gauss(r, ld, Math.sqrt(ld)) : poisson(r, ld);
    A[i] = a; D[i] = d; ratios[i] = d !== 0 ? a / d : NaN;
  }
  return { ratios, A, D };
}
