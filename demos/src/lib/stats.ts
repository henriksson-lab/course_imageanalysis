// Statistics for the course demos (owner: evaluation day; no DOM).
//
//  - descriptive: mean, variance (n − 1), sd, median, quantile, mad, pearson, linfit
//  - distributions: normal pdf/cdf/quantile, log-gamma, regularised incomplete beta,
//    Student t cdf/quantile (any df > 0)
//  - tests and intervals: one-sample / paired / Welch t-tests (two-sided p, 95 % CI),
//    z-test, Bland–Altman (bias, limits of agreement, proportional-bias slope)
//  - multiple testing: Bonferroni, Benjamini–Hochberg (step-up), family-wise error rate
//  - seeded simulators: two-level experiment (experiments × cells, independent / paired /
//    confounded designs) and a well plate (wells × features, row/column gradient, edge
//    wells, layouts)
//
// Random numbers come from lib/random.ts (`rng`, `gauss`); never Math.random.
import { Rng, gauss, shuffle } from "./random";

type A = ArrayLike<number>;

// ---------------------------------------------------------------------------------------
// Descriptive statistics

export function sum(x: A): number { let s = 0; for (let i = 0; i < x.length; i++) s += x[i]; return s; }

export function mean(x: A): number { return x.length ? sum(x) / x.length : NaN; }

/** Sample variance (divides by n − 1; NaN for n < 2). */
export function variance(x: A): number {
  const n = x.length;
  if (n < 2) return NaN;
  const m = mean(x);
  let s = 0;
  for (let i = 0; i < n; i++) s += (x[i] - m) ** 2;
  return s / (n - 1);
}

/** Sample standard deviation (n − 1). */
export const sd = (x: A): number => Math.sqrt(variance(x));

/** Standard error of the mean, sd / √n. */
export const sem = (x: A): number => sd(x) / Math.sqrt(x.length);

/** Quantile with linear interpolation between order statistics (type 7, as most software). */
export function quantile(x: A, q: number): number {
  const s = Float64Array.from(x).sort();
  const n = s.length;
  if (!n) return NaN;
  const h = (n - 1) * Math.min(1, Math.max(0, q)), lo = Math.floor(h), hi = Math.ceil(h);
  return s[lo] + (h - lo) * (s[hi] - s[lo]);
}

export const median = (x: A): number => quantile(x, 0.5);

/** Median absolute deviation from the median (unscaled; × 1.4826 estimates the SD of normal data). */
export function mad(x: A): number {
  const m = median(x);
  return median(Float64Array.from(x, v => Math.abs(v - m)));
}

/** Pearson correlation. */
export function pearson(x: A, y: A): number {
  const n = Math.min(x.length, y.length), mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { const a = x[i] - mx, b = y[i] - my; sxy += a * b; sxx += a * a; syy += b * b; }
  return sxy / Math.sqrt(sxx * syy);
}

/** Least-squares line y = intercept + slope · x. */
export function linfit(x: A, y: A): { slope: number; intercept: number } {
  const n = Math.min(x.length, y.length), mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; }
  const slope = sxx > 0 ? sxy / sxx : 0;
  return { slope, intercept: my - slope * mx };
}

// ---------------------------------------------------------------------------------------
// Distributions

export const normPdf = (z: number): number => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

/** erfc by a Chebyshev fit (Numerical Recipes erfcc; relative error < 1.2e-7). */
function erfc(x: number): number {
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}

/** Standard normal CDF Φ(z). */
export const normCdf = (z: number): number => 0.5 * erfc(-z / Math.SQRT2);

/** Standard normal quantile Φ⁻¹(p) (Acklam's rational approximation, one Newton step). */
export function normQuantile(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const pl = 0.02425;
  let x: number;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= 1 - pl) {
    const q = p - 0.5, r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const e = normCdf(x) - p;
  return x - e / Math.max(normPdf(x), 1e-300);
}

/** log Γ(x) for x > 0 (Lanczos, g = 7). */
export function logGamma(x: number): number {
  const g = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - logGamma(1 - x);
  x -= 1;
  let s = g[0];
  for (let i = 1; i < 9; i++) s += g[i] / (x + i);
  const t = x + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(s);
}

/** Continued fraction for the incomplete beta (Numerical Recipes betacf, modified Lentz). */
function betacf(a: number, b: number, x: number): number {
  const FPMIN = 1e-300, EPS = 1e-14;
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

/** Regularised incomplete beta I_x(a, b). */
export function incBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbt = logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x);
  const bt = Math.exp(lbt);
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}

/** Student t CDF with df degrees of freedom (df = Infinity: normal). */
export function tCdf(t: number, df: number): number {
  if (!Number.isFinite(df)) return normCdf(t);
  if (!Number.isFinite(t)) return t > 0 ? 1 : 0;
  const x = df / (df + t * t), tail = 0.5 * incBeta(x, df / 2, 0.5);
  return t > 0 ? 1 - tail : tail;
}

/** Student t quantile (inverse CDF), by bisection on tCdf. */
export function tQuantile(p: number, df: number): number {
  if (!Number.isFinite(df)) return normQuantile(p);
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p === 0.5) return 0;
  if (p < 0.5) return -tQuantile(1 - p, df);
  let lo = 0, hi = 1;
  while (tCdf(hi, df) < p) { lo = hi; hi *= 2; if (hi > 1e12) return hi; }
  for (let i = 0; i < 200 && hi - lo > 1e-12 * Math.max(1, hi); i++) {
    const m = 0.5 * (lo + hi);
    if (tCdf(m, df) < p) lo = m; else hi = m;
  }
  return 0.5 * (lo + hi);
}

/** Two-sided p-value of a t statistic. */
export const tTwoSidedP = (t: number, df: number): number => 2 * tCdf(-Math.abs(t), df);

/** Two-sided p-value of a z statistic. */
export const zTwoSidedP = (z: number): number => 2 * normCdf(-Math.abs(z));

// ---------------------------------------------------------------------------------------
// Tests and confidence intervals

/** Result of a test of a difference (or mean) against 0. */
export interface TestResult {
  estimate: number;      // mean (one-sample/paired) or difference of means (two-sample)
  se: number;            // its standard error
  t: number;             // estimate / se
  df: number;
  p: number;             // two-sided
  ci: [number, number];  // confidence interval (level given to the test, default 95 %)
  n: number;             // number of values (pairs, or total of both groups)
}

function result(estimate: number, se: number, df: number, n: number, level: number): TestResult {
  const t = estimate / se, q = tQuantile(1 - (1 - level) / 2, df);
  return { estimate, se, t, df, p: tTwoSidedP(t, df), ci: [estimate - q * se, estimate + q * se], n };
}

/** One-sample t-test of mean(x) = mu0. */
export function tTestOne(x: A, mu0 = 0, level = 0.95): TestResult {
  const n = x.length;
  return result(mean(x) - mu0, sd(x) / Math.sqrt(n), n - 1, n, level);
}

/** Paired t-test: one-sample test on the differences b[i] − a[i]. */
export function tTestPaired(a: A, b: A, level = 0.95): TestResult {
  const d = Float64Array.from({ length: Math.min(a.length, b.length) }, (_, i) => b[i] - a[i]);
  return tTestOne(d, 0, level);
}

/** Welch two-sample t-test of mean(b) − mean(a) (unequal variances, Welch–Satterthwaite df). */
export function tTestWelch(a: A, b: A, level = 0.95): TestResult {
  const na = a.length, nb = b.length, va = variance(a) / na, vb = variance(b) / nb;
  const se = Math.sqrt(va + vb);
  const df = (va + vb) ** 2 / (va * va / (na - 1) + vb * vb / (nb - 1));
  return result(mean(b) - mean(a), se, df, na + nb, level);
}

/** Two-sample t-test with pooled variance (equal variances), df = na + nb − 2. */
export function tTestPooled(a: A, b: A, level = 0.95): TestResult {
  const na = a.length, nb = b.length;
  const sp2 = ((na - 1) * variance(a) + (nb - 1) * variance(b)) / (na + nb - 2);
  return result(mean(b) - mean(a), Math.sqrt(sp2 * (1 / na + 1 / nb)), na + nb - 2, na + nb, level);
}

/** Bland–Altman analysis of a method y against a reference x (both per item). */
export interface BlandAltman {
  means: Float64Array;   // (x + y) / 2
  diffs: Float64Array;   // y − x
  bias: number;          // mean difference
  sd: number;            // SD of the differences
  lo: number; hi: number; // limits of agreement, bias ∓ k · sd (k = 1.96)
  slope: number;         // regression of diffs on means: proportional bias
  intercept: number;
  r: number;             // Pearson correlation of x and y (for comparison)
}

export function blandAltman(x: A, y: A, k = 1.96): BlandAltman {
  const n = Math.min(x.length, y.length);
  const means = Float64Array.from({ length: n }, (_, i) => (x[i] + y[i]) / 2);
  const diffs = Float64Array.from({ length: n }, (_, i) => y[i] - x[i]);
  const bias = mean(diffs), s = sd(diffs), fit = linfit(means, diffs);
  return { means, diffs, bias, sd: s, lo: bias - k * s, hi: bias + k * s, slope: fit.slope, intercept: fit.intercept, r: pearson(x, y) };
}

/** Standard error of a proportion p from n independent items, √(p(1 − p)/n). */
export const seProportion = (p: number, n: number): number => Math.sqrt(p * (1 - p) / n);

/** Design effect 1 + (m − 1)ρ for clusters of size m with intra-cluster correlation ρ. */
export const designEffect = (m: number, rho: number): number => 1 + (m - 1) * rho;

/** False-positive rate of a cells-as-n z-test at level alpha under the null (large-E limit). */
export const pseudoRepFPR = (m: number, rho: number, alpha = 0.05): number =>
  2 * normCdf(-normQuantile(1 - alpha / 2) / Math.sqrt(designEffect(m, rho)));

// ---------------------------------------------------------------------------------------
// Multiple testing

/** Bonferroni: reject p_i ≤ alpha / m. */
export function bonferroni(p: A, alpha = 0.05): boolean[] {
  const m = p.length;
  return Array.from({ length: m }, (_, i) => p[i] <= alpha / m);
}

/**
 * Benjamini–Hochberg step-up at FDR q: find the largest k with p_(k) ≤ k q / m and reject
 * the k smallest p-values. Returns the rejections and the p-value threshold p_(k) (0 if none).
 */
export function benjaminiHochberg(p: A, q = 0.05): { reject: boolean[]; threshold: number; k: number } {
  const m = p.length, idx = Array.from({ length: m }, (_, i) => i).sort((a, b) => p[a] - p[b]);
  let k = 0;
  for (let j = m; j >= 1; j--) if (p[idx[j - 1]] <= j * q / m) { k = j; break; }
  const reject = new Array<boolean>(m).fill(false);
  for (let j = 0; j < k; j++) reject[idx[j]] = true;
  return { reject, threshold: k ? p[idx[k - 1]] : 0, k };
}

/** P(at least one false hit) among m independent null tests at level alpha. */
export const familywiseError = (m: number, alpha = 0.05): number => 1 - (1 - alpha) ** m;

// ---------------------------------------------------------------------------------------
// Simulators

export type Design = "independent" | "paired" | "confounded";

export interface TwoLevelOptions {
  experiments: number;   // E per condition (independent, confounded) or E in total (paired)
  cells: number;         // m cells per experiment and condition
  sdBetween: number;     // SD of the experiment (day) effect u_e
  sdWithin: number;      // SD of the cells around their experiment mean
  effect: number;        // true treatment effect
  design: Design;
  batch?: number;        // confounded: shift of the second batch (which holds every treated experiment)
  mu?: number;           // grand mean (default 0)
}

/** One simulated experiment: its id, the condition (0 control, 1 treated), and the cell values. */
export interface ExperimentData { exp: number; cond: 0 | 1; values: Float64Array }

/**
 * Two-level data y = μ + effect·T + u_e (+ batch) + ε. Independent: 2E experiments, E per
 * condition, each with its own u_e. Paired: E experiments, both conditions in each, sharing
 * u_e. Confounded: like independent, but all controls in batch 1 and all treated in batch 2,
 * which is shifted by `batch`.
 */
export function simulateTwoLevel(r: Rng, o: TwoLevelOptions): ExperimentData[] {
  const out: ExperimentData[] = [], mu = o.mu ?? 0, E = o.experiments, m = o.cells;
  const cells = (centre: number) => Float64Array.from({ length: m }, () => gauss(r, centre, o.sdWithin));
  if (o.design === "paired") {
    for (let e = 0; e < E; e++) {
      const u = gauss(r, 0, o.sdBetween);
      out.push({ exp: e, cond: 0, values: cells(mu + u) });
      out.push({ exp: e, cond: 1, values: cells(mu + u + o.effect) });
    }
  } else {
    const shift = o.design === "confounded" ? (o.batch ?? 0) : 0;
    for (const c of [0, 1] as const)
      for (let e = 0; e < E; e++) {
        const u = gauss(r, 0, o.sdBetween);
        out.push({ exp: c * E + e, cond: c, values: cells(mu + u + c * (o.effect + shift)) });
      }
  }
  return out;
}

/** The three analyses of two-level data: cells as n, experiment means as n, and paired means. */
export function analyseTwoLevel(data: ExperimentData[], level = 0.95): { cells: TestResult; means: TestResult; paired: TestResult | null } {
  const all = (c: 0 | 1) => { const v: number[] = []; data.filter(d => d.cond === c).forEach(d => d.values.forEach(x => v.push(x))); return v; };
  const ms = (c: 0 | 1) => data.filter(d => d.cond === c).map(d => mean(d.values));
  const cells = tTestWelch(all(0), all(1), level);
  const means = tTestWelch(ms(0), ms(1), level);
  let paired: TestResult | null = null;
  const ids = [...new Set(data.map(d => d.exp))];
  if (ids.every(e => data.some(d => d.exp === e && d.cond === 0) && data.some(d => d.exp === e && d.cond === 1))) {
    const a = ids.map(e => mean(data.find(d => d.exp === e && d.cond === 0)!.values));
    const b = ids.map(e => mean(data.find(d => d.exp === e && d.cond === 1)!.values));
    paired = tTestPaired(a, b, level);
  }
  return { cells, means, paired };
}

export type PlateLayout = "left" | "random" | "blocked";

export interface PlateOptions {
  rows: number; cols: number;     // e.g. 16 × 24 or 8 × 12
  features: number;               // m features per well
  fracReal: number;               // fraction of features with a real effect
  effect: number;                 // effect size of the real ones, in units of the well SD
  rowGradient: number;            // shift per plate height (SD units), top to bottom
  colGradient: number;            // shift per plate width (SD units), left to right
  edge: number;                   // extra shift of the outer wells (SD units)
  layout: PlateLayout;            // treated on the left half / randomised / balanced per row
}

/** A simulated plate: condition per well, values[f][well], which features are real. */
export interface Plate { rows: number; cols: number; cond: Uint8Array; values: Float64Array[]; real: boolean[] }

/**
 * Wells × features; each feature gets its own sensitivity (random sign and size around 1) to
 * the shared plate gradients and edge effect, so position effects hit many features at once.
 * Layout "left": treated wells are the left half; "random": a random half; "blocked": half of
 * every row, randomised within the row (so every row and both halves hold both conditions).
 */
export function simulatePlate(r: Rng, o: PlateOptions): Plate {
  const n = o.rows * o.cols, cond = new Uint8Array(n);
  if (o.layout === "left") for (let i = 0; i < n; i++) cond[i] = (i % o.cols) < o.cols / 2 ? 1 : 0;
  else if (o.layout === "random") {
    const idx = shuffle(r, Array.from({ length: n }, (_, i) => i));
    for (let k = 0; k < n / 2; k++) cond[idx[k]] = 1;
  } else {
    for (let y = 0; y < o.rows; y++) {
      const idx = shuffle(r, Array.from({ length: o.cols }, (_, i) => i));
      for (let k = 0; k < o.cols / 2; k++) cond[y * o.cols + idx[k]] = 1;
    }
  }
  const nReal = Math.round(o.fracReal * o.features);
  const real = Array.from({ length: o.features }, (_, f) => f < nReal);
  const values: Float64Array[] = [];
  for (let f = 0; f < o.features; f++) {
    const sens = gauss(r, 1, 0.5) * (r.unif() < 0.5 ? -1 : 1);
    const v = new Float64Array(n);
    for (let y = 0; y < o.rows; y++) for (let x = 0; x < o.cols; x++) {
      const i = y * o.cols + x;
      const pos = o.rowGradient * (y / (o.rows - 1) - 0.5) + o.colGradient * (x / (o.cols - 1) - 0.5) +
        ((y === 0 || x === 0 || y === o.rows - 1 || x === o.cols - 1) ? o.edge : 0);
      v[i] = gauss(r, 0, 1) + sens * pos + (real[f] && cond[i] ? o.effect : 0);
    }
    values.push(v);
  }
  return { rows: o.rows, cols: o.cols, cond, values, real };
}
