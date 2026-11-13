// Fluorescence decays for the FLIM day (no DOM). Times in ns, the laser period T in ns,
// a histogram of nBins bins of width Δ = T / nBins over one period; bin j covers
// [jΔ, (j+1)Δ) with centre (j + ½)Δ.
//
// - Multi-exponential models on a periodic window (closed-form sum over previous pulses),
//   integrated over each bin, so the bins of one normalised component sum to 1.
// - IRF: a Gaussian with a delay, or any measured array of bins. For the Gaussian the model is
//   exact (the exponentially modified Gaussian, wrapped round the period, integrated over each
//   bin); for an array it is the circular convolution of bins, IRF ⊛ decay (which shifts the
//   decay by half a bin compared with the continuous convolution).
// - Expected counts μ_j = N · (IRF ⊛ decay)_j + b; Poisson histograms; photon-by-photon TCSPC
//   simulation with IRF jitter, background, wrap-around and first-photon-only pile-up.
// - Quick estimators (mean arrival time, two gates), frequency-domain numbers (φ, m, τ_φ, τ_m),
//   amplitude ↔ photon fractions, τ̄_a and τ̄_f.
// - FRET on the phasor plot (the rate picture itself comes from lib/fret.ts, FRET day).
// - flimScene(kind, opts): x × y × t photon-count stacks (time-major: stack[j * P + p]) of the
//   unmixing day's cell, for the scenes of the flimcell demo, with ground truth.

import { rng, poisson, gauss, exponential, Rng } from "./random";
import { normCdf } from "./stats";
import { cellScene, blurMap } from "./unmix";
import { efficiency, distanceFromE, lifetimeWithAcceptor } from "./fret";

type A = ArrayLike<number>;

/** Time axis of a histogram: period T (ns) and number of bins. */
export interface TimeAxis { T: number; nBins: number }

export const binWidth = (ax: TimeAxis): number => ax.T / ax.nBins;
export function binCentres(ax: TimeAxis): Float64Array {
  const d = binWidth(ax);
  return Float64Array.from({ length: ax.nBins }, (_, j) => (j + 0.5) * d);
}

/** Normalised periodic decay density p_T(t) = e^{-t/τ} / (τ (1 − e^{-T/τ})) on [0, T). */
export function periodicDensity(t: number, tau: number, T: number): number {
  if (tau <= 0) return 0;
  return Math.exp(-t / tau) / (tau * (1 - Math.exp(-T / tau)));
}

/** Bin integrals of the normalised periodic exponential (sum = 1). τ ≤ 0: a spike in bin 0. */
export function expBins(tau: number, ax: TimeAxis): Float64Array {
  const n = ax.nBins, d = binWidth(ax), out = new Float64Array(n);
  if (!(tau > 1e-9)) { out[0] = 1; return out; }
  const norm = 1 - Math.exp(-ax.T / tau);
  let prev = 1;
  for (let j = 0; j < n; j++) {
    const next = Math.exp(-(j + 1) * d / tau);
    out[j] = (prev - next) / norm;
    prev = next;
  }
  return out;
}

/** Fraction of a single decay still to come at time t (no wrap): e^{-t/τ}. */
export const survival = (t: number, tau: number): number => Math.exp(-t / tau);

/** Fraction of the photons in one period that were excited by earlier pulses (steady state): e^{-T/τ}. */
export const wrappedFraction = (tau: number, T: number): number => Math.exp(-T / tau);

/** Gaussian IRF (sd σ, delay t0) integrated over each bin, wrapped round the period, sum 1.
 *  σ = 0: a spike in the bin that contains t0. */
export function gaussianIRF(sigma: number, t0: number, ax: TimeAxis): Float64Array {
  const n = ax.nBins, d = binWidth(ax), out = new Float64Array(n);
  const tw = ((t0 % ax.T) + ax.T) % ax.T;
  if (!(sigma > 1e-6)) { out[Math.min(n - 1, Math.floor(tw / d))] = 1; return out; }
  const K = Math.ceil(6 * sigma / ax.T) + 1;
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let k = -K; k <= K; k++) s += normCdf(((j + 1) * d + k * ax.T - tw) / sigma) - normCdf((j * d + k * ax.T - tw) / sigma);
    out[j] = s;
  }
  let tot = 0;
  for (const v of out) tot += v;
  for (let j = 0; j < n; j++) out[j] /= tot;
  return out;
}

/** Circular convolution (h ⊛ x)_j = Σ_i h_i x_{(j−i) mod n}, sparse over the non-zero h_i. */
export function convolveCircular(h: A, x: A, out?: Float64Array): Float64Array {
  const n = x.length, y = out ?? new Float64Array(n);
  y.fill(0);
  for (let i = 0; i < h.length; i++) {
    const hi = h[i];
    if (Math.abs(hi) < 1e-12) continue;
    for (let j = 0; j < n; j++) y[(j + i) % n] += hi * x[j];
  }
  return y;
}

/** A multi-exponential decay: lifetimes and photon fractions f (sum 1). */
export interface Decay { tau: number[]; f: number[] }

/** An instrument response: a Gaussian (sd σ, delay, ns), or a measured array of bins (sum 1). */
export type IRF = { sigma: number; delay: number } | ArrayLike<number>;
const isSpec = (h: IRF): h is { sigma: number; delay: number } => (h as { sigma?: number }).sigma !== undefined;

/** The IRF as bins (for display and for the discrete convolution). */
export const irfBins = (h: IRF, ax: TimeAxis): Float64Array => (isSpec(h) ? gaussianIRF(h.sigma, h.delay, ax) : Float64Array.from(h));

/** CDF of (Gaussian(t0, σ) + exponential(τ)) at t: the exponentially modified Gaussian.
 *  σ = 0: a delayed exponential; τ = 0: the Gaussian alone. */
export function emgCdf(t: number, t0: number, sigma: number, tau: number): number {
  if (!(tau > 1e-9)) return sigma > 1e-9 ? normCdf((t - t0) / sigma) : (t >= t0 ? 1 : 0);
  if (!(sigma > 1e-9)) return t > t0 ? 1 - Math.exp(-(t - t0) / tau) : 0;
  const z = (t - t0) / sigma;
  if (z < -12) return 0;
  const e = -(t - t0) / tau + sigma * sigma / (2 * tau * tau);
  const tail = z - sigma / tau < -12 ? 0 : Math.exp(e) * normCdf(z - sigma / tau);
  return Math.max(0, normCdf(z) - tail);
}

/** Exact bin integrals of one periodic exponential (τ) seen through a Gaussian IRF: the
 *  continuous convolution, summed over all earlier pulses, integrated over each bin (sum 1). */
export function expIrfBins(tau: number, sigma: number, t0: number, ax: TimeAxis): Float64Array {
  const n = ax.nBins, d = binWidth(ax), T = ax.T, out = new Float64Array(n);
  const tw = ((t0 % T) + T) % T;
  const edges = new Float64Array(n + 1);
  // pulses k = -1, 0, 1 explicitly; the tail of all earlier pulses (k >= 2) as a geometric series
  for (let k = -1; k <= 1; k++) {
    for (let j = 0; j <= n; j++) edges[j] = emgCdf(j * d + k * T, tw, sigma, tau);
    for (let j = 0; j < n; j++) out[j] += edges[j + 1] - edges[j];
  }
  if (tau > 1e-9) {
    const C = Math.exp(tw / tau + (sigma > 1e-9 ? sigma * sigma / (2 * tau * tau) : 0));
    const q = Math.exp(-2 * T / tau) / (1 - Math.exp(-T / tau));
    for (let j = 0; j < n; j++) out[j] += C * q * (Math.exp(-j * d / tau) - Math.exp(-(j + 1) * d / tau));
  }
  return out;
}

/** Bin probabilities of a multi-exponential decay through the IRF (sum 1): exact for a
 *  Gaussian IRF, the circular convolution of bins for a measured one. */
export function decayBins(dec: Decay, irf: IRF, ax: TimeAxis): Float64Array {
  const n = ax.nBins, out = new Float64Array(n);
  if (isSpec(irf)) {
    dec.tau.forEach((tau, k) => { if (dec.f[k] === 0) return; const e = expIrfBins(tau, irf.sigma, irf.delay, ax); for (let j = 0; j < n; j++) out[j] += dec.f[k] * e[j]; });
    return out;
  }
  dec.tau.forEach((tau, k) => { const e = expBins(tau, ax); for (let j = 0; j < n; j++) out[j] += dec.f[k] * e[j]; });
  return convolveCircular(irf, out);
}

/** Expected counts: N signal photons spread over the bins, plus background b per bin. */
export function expectedCounts(dec: Decay, irf: IRF, ax: TimeAxis, N: number, bPerBin = 0): Float64Array {
  const p = decayBins(dec, irf, ax);
  for (let j = 0; j < p.length; j++) p[j] = N * p[j] + bPerBin;
  return p;
}

/** A Poisson histogram with means mu. */
export function poissonHistogram(r: Rng, mu: A): Float64Array {
  return Float64Array.from({ length: mu.length }, (_, j) => poisson(r, mu[j]));
}

// ---------------------------------------------------------------------------------------
// Photon-by-photon TCSPC

export interface TcspcOptions {
  tau: number[]; f: number[];      // lifetimes and photon fractions of the fluorescence
  T: number;                       // pulse period (ns)
  irfSigma: number; irfDelay: number;
  bgFraction: number;              // fraction of detected photons that are background (uniform in time)
  rate: number;                    // fraction of pulses with ≥ 1 detected photon (pile-up when large)
}

/** Draw one fluorescence or background arrival time in [0, T). */
function arrival(r: Rng, o: TcspcOptions): number {
  if (r.unif() < o.bgFraction) return r.unif() * o.T;
  let u = r.unif(), k = 0;
  while (k < o.f.length - 1 && u >= o.f[k]) { u -= o.f[k]; k++; }
  const t = o.irfDelay + (o.irfSigma > 0 ? gauss(r, 0, o.irfSigma) : 0) + exponential(r, 1 / o.tau[k]);
  return ((t % o.T) + o.T) % o.T;
}

/** One detection event: only pulses with at least one photon are simulated. The number of
 *  photons in such a pulse is Poisson(λ) given ≥ 1, with λ = −ln(1 − rate); the electronics
 *  record the earliest. Returns the recorded time and how many photons were in the pulse. */
export function detectPulse(r: Rng, o: TcspcOptions): { t: number; k: number } {
  const lam = -Math.log(1 - Math.min(0.999, Math.max(1e-9, o.rate)));
  // zero-truncated Poisson by inversion
  const e = Math.exp(-lam);
  let u = r.unif() * (1 - e), k = 1, p = e * lam;
  while (u > p && k < 50) { u -= p; k++; p *= lam / k; }
  let t = Infinity;
  for (let i = 0; i < k; i++) t = Math.min(t, arrival(r, o));
  return { t, k };
}

/** Number of pulses until the next detection (geometric with success probability `rate`). */
export function pulsesUntilDetection(r: Rng, rate: number): number {
  if (rate >= 1) return 1;
  return 1 + Math.floor(Math.log(1 - r.unif()) / Math.log(1 - rate));
}

/** Histogram of `count` detections (photon by photon). */
export function simulateTcspc(r: Rng, o: TcspcOptions, ax: TimeAxis, count: number, into?: Float64Array): Float64Array {
  const h = into ?? new Float64Array(ax.nBins), d = binWidth(ax);
  for (let i = 0; i < count; i++) {
    const { t } = detectPulse(r, o);
    h[Math.min(ax.nBins - 1, Math.floor(t / d))]++;
  }
  return h;
}

// ---------------------------------------------------------------------------------------
// Quick estimators

/** Mean arrival time of a histogram (bin centres). */
export function meanArrival(y: A, ax: TimeAxis): number {
  const d = binWidth(ax);
  let s = 0, n = 0;
  for (let j = 0; j < y.length; j++) { s += y[j] * (j + 0.5) * d; n += y[j]; }
  return n > 0 ? s / n : NaN;
}

/** Sum of the bins whose centres lie in [a, b). */
export function gateSum(y: A, ax: TimeAxis, a: number, b: number): number {
  const d = binWidth(ax);
  let s = 0;
  for (let j = 0; j < y.length; j++) { const t = (j + 0.5) * d; if (t >= a && t < b) s += y[j]; }
  return s;
}

/** Two adjacent gates of (about) width w from t = a, snapped to whole bins so that both hold
 *  the same number of bins: τ = w' / ln(D1 / D2) with the snapped width w' (NaN if undefined). */
export function twoGateTau(y: A, ax: TimeAxis, a: number, w: number): { tau: number; d1: number; d2: number; a: number; w: number } {
  const d = binWidth(ax), j0 = Math.max(0, Math.round(a / d)), nw = Math.max(1, Math.round(w / d));
  let d1 = 0, d2 = 0;
  for (let j = j0; j < Math.min(y.length, j0 + nw); j++) d1 += y[j];
  for (let j = j0 + nw; j < Math.min(y.length, j0 + 2 * nw); j++) d2 += y[j];
  const ws = nw * d;
  const tau = d1 > 0 && d2 > 0 && d1 > d2 ? ws / Math.log(d1 / d2) : NaN;
  return { tau, d1, d2, a: j0 * d, w: ws };
}

/** Mean of the periodic exponential on [0, T): τ − T e^{-T/τ} / (1 − e^{-T/τ}). */
export function periodicMean(tau: number, T: number): number {
  const e = Math.exp(-T / tau);
  return tau - T * e / (1 - e);
}

/** Invert periodicMean by bisection (the mean-arrival estimate corrected for the window). */
export function tauFromPeriodicMean(m: number, T: number): number {
  if (!(m > 0)) return NaN;
  if (m >= T / 2) return Infinity;
  let lo = 1e-6, hi = 1e4;
  for (let i = 0; i < 80; i++) { const mid = Math.sqrt(lo * hi); if (periodicMean(mid, T) < m) lo = mid; else hi = mid; }
  return Math.sqrt(lo * hi);
}

// ---------------------------------------------------------------------------------------
// Frequency domain and fractions

/** Phasor of a multi-exponential decay at angular frequency ω (rad/ns): P = Σ f_k / (1 − iωτ_k). */
export function decayPhasor(dec: Decay, omega: number): { g: number; s: number } {
  let g = 0, s = 0;
  dec.tau.forEach((t, k) => { const w = omega * t, q = 1 + w * w; g += dec.f[k] / q; s += dec.f[k] * w / q; });
  return { g, s };
}

/** Phase φ (rad) and modulation m of a phasor; phase and modulation lifetimes. */
export function phaseModulation(g: number, s: number, omega: number): { phi: number; m: number; tauPhi: number; tauM: number } {
  const phi = Math.atan2(s, g), m = Math.hypot(g, s);
  return { phi, m, tauPhi: Math.tan(phi) / omega, tauM: m > 0 && m <= 1 ? Math.sqrt(1 / (m * m) - 1) / omega : NaN };
}

export const omegaOf = (fMHz: number): number => 2 * Math.PI * fMHz * 1e-3; // rad/ns

/** Photon fractions from amplitude fractions: f_k = α_k τ_k / Σ α_j τ_j. */
export function photonFractions(alpha: A, tau: A): number[] {
  let s = 0;
  for (let k = 0; k < tau.length; k++) s += alpha[k] * tau[k];
  return Array.from(tau, (t, k) => (s > 0 ? alpha[k] * t / s : 0));
}

/** Amplitude fractions from photon fractions: α_k = (f_k/τ_k) / Σ f_j/τ_j. */
export function amplitudeFractions(f: A, tau: A): number[] {
  let s = 0;
  for (let k = 0; k < tau.length; k++) s += f[k] / tau[k];
  return Array.from(tau, (t, k) => (s > 0 ? f[k] / t / s : 0));
}

/** Amplitude-weighted mean lifetime Σ α_k τ_k (α normalised). */
export function tauAmplitude(alpha: A, tau: A): number {
  let s = 0, a = 0;
  for (let k = 0; k < tau.length; k++) { s += alpha[k] * tau[k]; a += alpha[k]; }
  return s / a;
}

/** Intensity-weighted mean lifetime Σ f_k τ_k = Σ α_k τ_k² / Σ α_k τ_k. */
export function tauIntensity(alpha: A, tau: A): number {
  let s = 0, a = 0;
  for (let k = 0; k < tau.length; k++) { s += alpha[k] * tau[k] * tau[k]; a += alpha[k] * tau[k]; }
  return s / a;
}

// ---------------------------------------------------------------------------------------
// FRET: the rate picture comes from lib/fret.ts (owned by the FRET day); re-exported here
// under the names the FLIM demos use.

/** FRET efficiency at distance r for Förster distance R0: 1 / (1 + (r/R0)^6). */
export const fretEfficiency = efficiency;
/** Distance from efficiency: r = R0 (1/E − 1)^{1/6}. */
export const fretDistance = distanceFromE;
/** Donor lifetime with an acceptor at efficiency E: τ_DA = τ_D (1 − E). */
export { lifetimeWithAcceptor };
/** Efficiency from lifetimes: E = 1 − τ_DA / τ_D. */
export const efficiencyFromLifetimes = (tauDA: number, tauD: number): number => 1 - tauDA / tauD;

/** FRET trajectory on the phasor plot: the phasor of a donor population of which a fraction
 *  α (amplitude) interacts at efficiency E, plus a background photon fraction (phasor 0) and
 *  an extra species (photon fraction fx at phasor px). */
export function fretPhasor(tauD: number, E: number, alpha: number, omega: number, bgFrac = 0, fx = 0, px: { g: number; s: number } = { g: 0, s: 0 }): { g: number; s: number } {
  const tauDA = lifetimeWithAcceptor(tauD, E);
  const f = photonFractions([1 - alpha, alpha], [tauD, Math.max(tauDA, 1e-9)]);
  const P = decayPhasor({ tau: [tauD, Math.max(tauDA, 1e-9)], f }, omega);
  const w = 1 - bgFrac - fx;
  return { g: w * P.g + fx * px.g, s: w * P.s + fx * px.s };
}

// ---------------------------------------------------------------------------------------
// Synthetic FLIM images of the unmixing day's cell

export type SceneKind = "concentration" | "twodyes" | "quencher" | "mixture" | "background" | "fret" | "nadh";

export interface FlimSceneOptions {
  n?: number;            // image size (default 128)
  nBins?: number;        // time bins (default 64)
  T?: number;            // period, ns (default 12.5 = 80 MHz)
  photons?: number;      // expected photons in the brightest pixels (default 400)
  irfSigma?: number;     // ns (default 0.1)
  irfDelay?: number;     // ns (default 0.5)
  bgPerBin?: number;     // background counts per bin and pixel (default scene-dependent)
  seed?: number;
}

export interface FlimScene {
  kind: SceneKind;
  n: number; ax: TimeAxis; irf: Float64Array; irfSpec: { sigma: number; delay: number };
  /** counts, time-major: stack[j * P + p] */
  stack: Float32Array;
  /** total counts per pixel */
  intensity: Float32Array;
  /** ground truth: species lifetimes, expected photons per species and pixel, background per bin */
  tau: number[]; names: string[]; abundance: Float32Array[]; bgPerBin: number;
  /** ground-truth intensity-weighted lifetime per pixel (NaN outside the cell) */
  tauTrue: Float32Array;
  /** cell mask, compartment maps used by the scene */
  cell: Uint8Array; regions: Record<string, Float32Array>;
}

const smooth01 = (v: number): number => Math.max(0, Math.min(1, v));

/** Build the per-species expected photon maps of a scene (before noise). */
function sceneSpecies(kind: SceneKind, n: number): { tau: number[]; names: string[]; ab: Float32Array[]; bg: number; regions: Record<string, Float32Array>; cell: Uint8Array } {
  const c = cellScene(n, 3), P = n * n;
  const cellSoft = blurMap(Float32Array.from(c.cell), n, n, 1.2);
  const nucMask = Float32Array.from(c.nucleus, v => smooth01((v - 0.3) / 0.3));
  // structured concentration (varies about 10×): nucleus, mitochondria, membrane, a floor
  const conc = new Float32Array(P);
  let cmax = 0;
  for (let p = 0; p < P; p++) { conc[p] = cellSoft[p] * (0.1 + 0.55 * c.mito[p] + 0.45 * c.membrane[p] + 0.35 * c.nucleus[p] + 0.25 * c.auto[p]); cmax = Math.max(cmax, conc[p]); }
  for (let p = 0; p < P; p++) conc[p] /= cmax;
  const ab = (f: (p: number) => number) => Float32Array.from({ length: P }, (_, p) => f(p));
  const regions: Record<string, Float32Array> = { nucleus: nucMask, mito: c.mito, membrane: c.membrane, vesicles: c.vesicles };
  switch (kind) {
    case "concentration":
      return { tau: [2.5], names: ["dye (2.5 ns)"], ab: [conc], bg: 0.02, regions, cell: c.cell };
    case "twodyes":
      return { tau: [1.0, 3.0], names: ["dye A (1.0 ns)", "dye B (3.0 ns)"], ab: [ab(p => 0.8 * cellSoft[p] * nucMask[p]), ab(p => 0.8 * cellSoft[p] * (1 - nucMask[p]))], bg: 0.02, regions, cell: c.cell };
    case "quencher": {
      // the same dye; in the mitochondria a quencher shortens τ 2.5 → 1.2 ns and, with it, the brightness
      const q = c.mito;
      return { tau: [2.5, 1.2], names: ["free dye (2.5 ns)", "quenched (1.2 ns)"], ab: [ab(p => conc[p] * (1 - q[p])), ab(p => conc[p] * q[p] * 1.2 / 2.5)], bg: 0.02, regions, cell: c.cell };
    }
    case "mixture":
    case "background": {
      // two species, the long one's photon fraction varies smoothly across the cell
      const frac = ab(p => { const x = (p % n) / n, y = Math.floor(p / n) / n; return smooth01(0.15 + 0.75 * (0.65 * x + 0.35 * y) - 0.3 * nucMask[p]); });
      regions.fraction = frac;
      const I = kind === "background" ? conc : ab(p => cellSoft[p] * (0.55 + 0.45 * conc[p]));
      return { tau: [0.6, 3.5], names: ["short (0.6 ns)", "long (3.5 ns)"], ab: [ab(p => I[p] * (1 - frac[p])), ab(p => I[p] * frac[p])], bg: kind === "background" ? 0.25 : 0.02, regions, cell: c.cell };
    }
    case "fret": {
      // donor 2.6 ns; FRET state at E = 0.5 (1.3 ns). Interacting (amplitude) fraction:
      // 0.6 in the mitochondria, 0.25 at the membrane, 0.1 elsewhere, 0 in the nucleus.
      const tauD = 2.6, tauDA = 1.3;
      const alpha = ab(p => smooth01(0.1 * (1 - nucMask[p]) + 0.5 * c.mito[p] + 0.2 * c.membrane[p]));
      regions.alpha = alpha;
      const I = ab(p => cellSoft[p] * (0.45 + 0.55 * conc[p]));
      // photons ∝ amplitude × lifetime (the quenched donor is dimmer)
      return { tau: [tauD, tauDA], names: ["donor alone (2.6 ns)", "donor with acceptor (1.3 ns)"], ab: [ab(p => I[p] * (1 - alpha[p])), ab(p => I[p] * alpha[p] * tauDA / tauD)], bg: 0.02, regions, cell: c.cell };
    }
    case "nadh": {
      // free 0.4 ns, bound 2.5 ns; molecule fraction bound 0.6 in mitochondria, 0.25 in the
      // cytoplasm, 0.15 in the nucleus; a label (4.5 ns) in a few bright vesicles
      const bound = ab(p => smooth01(0.25 + 0.35 * c.mito[p] - 0.1 * nucMask[p]));
      regions.bound = bound;
      const I = ab(p => cellSoft[p] * (0.5 + 0.5 * conc[p]));
      const wf = (p: number) => I[p] * (1 - bound[p]) * 0.4, wb = (p: number) => I[p] * bound[p] * 2.5;
      const nrm = 1 / 1.2;
      return { tau: [0.4, 2.5, 4.5], names: ["free NADH (0.4 ns)", "bound NADH (2.5 ns)", "label (4.5 ns)"], ab: [ab(p => nrm * wf(p)), ab(p => nrm * wb(p)), ab(p => 0.9 * c.vesicles[p])], bg: 0.02, regions, cell: c.cell };
    }
  }
}

/** A seeded x × y × t photon-count stack of the scene, with ground truth. */
export function flimScene(kind: SceneKind, o: FlimSceneOptions = {}): FlimScene {
  const n = o.n ?? 128, P = n * n, ax: TimeAxis = { T: o.T ?? 12.5, nBins: o.nBins ?? 64 };
  const irfSpec = { sigma: o.irfSigma ?? 0.1, delay: o.irfDelay ?? 0.5 }, irf = gaussianIRF(irfSpec.sigma, irfSpec.delay, ax);
  const sp = sceneSpecies(kind, n);
  const photons = o.photons ?? 400;
  // scale so that the brightest pixels (99th percentile of the total) get `photons`
  const tot = new Float32Array(P);
  for (const a of sp.ab) for (let p = 0; p < P; p++) tot[p] += a[p];
  const sorted = Float32Array.from(tot).sort();
  const ref = sorted[Math.floor(0.99 * (P - 1))] || 1;
  const abundance = sp.ab.map(a => Float32Array.from(a, v => v * photons / ref));
  const bg = o.bgPerBin ?? sp.bg * photons / 400;
  const basis = sp.tau.map(t => decayBins({ tau: [t], f: [1] }, irfSpec, ax));
  const r = rng(o.seed ?? 11), nb = ax.nBins;
  const stack = new Float32Array(nb * P), intensity = new Float32Array(P), tauTrue = new Float32Array(P);
  const mu = new Float64Array(nb);
  for (let p = 0; p < P; p++) {
    mu.fill(bg);
    let s = 0, st = 0;
    for (let k = 0; k < basis.length; k++) {
      const a = abundance[k][p];
      s += a; st += a * sp.tau[k];
      if (a <= 0) continue;
      const b = basis[k];
      for (let j = 0; j < nb; j++) mu[j] += a * b[j];
    }
    tauTrue[p] = sp.cell[p] && s > 0 ? st / s : NaN;
    let tot2 = 0;
    for (let j = 0; j < nb; j++) { const v = poisson(r, mu[j]); stack[j * P + p] = v; tot2 += v; }
    intensity[p] = tot2;
  }
  return { kind, n, ax, irf, irfSpec, stack, intensity, tau: sp.tau, names: sp.names, abundance, bgPerBin: bg, tauTrue, cell: sp.cell, regions: sp.regions };
}

/** Sum each pixel's histogram over a (2r+1)² box (binning; edges use the pixels inside). */
export function binStack(stack: A, n: number, nBins: number, r: number): Float32Array {
  const P = n * n;
  if (r <= 0) return Float32Array.from(stack);
  const out = new Float32Array(nBins * P), tmp = new Float32Array(P);
  for (let j = 0; j < nBins; j++) {
    const off = j * P;
    for (let y = 0; y < n; y++) {
      let s = 0;
      for (let x = -r; x <= r; x++) if (x >= 0 && x < n) s += stack[off + y * n + x];
      for (let x = 0; x < n; x++) {
        tmp[y * n + x] = s;
        const a = x - r, b = x + r + 1;
        if (a >= 0) s -= stack[off + y * n + a];
        if (b < n) s += stack[off + y * n + b];
      }
    }
    for (let x = 0; x < n; x++) {
      let s = 0;
      for (let y = -r; y <= r; y++) if (y >= 0 && y < n) s += tmp[y * n + x];
      for (let y = 0; y < n; y++) {
        out[off + y * n + x] = s;
        const a = y - r, b = y + r + 1;
        if (a >= 0) s -= tmp[a * n + x];
        if (b < n) s += tmp[b * n + x];
      }
    }
  }
  return out;
}

/** One pixel's histogram out of a time-major stack. */
export function pixelHistogram(stack: A, P: number, nBins: number, p: number): Float64Array {
  return Float64Array.from({ length: nBins }, (_, j) => stack[j * P + p]);
}
