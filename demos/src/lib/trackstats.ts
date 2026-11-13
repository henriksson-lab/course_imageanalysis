// What tracks measure (tracking day). No DOM. Per-track numbers (duration, net displacement,
// path length, speed, straightness, turning angles), the mean squared displacement (time
// averaged per track and over an ensemble; lags from the time stamps, so tracks with gaps work)
// and fits (line with offset, power law on log–log axes, diffusion + directed, confined plateau).
//
// A track is { t, x, y }: frame numbers (integers, increasing, gaps allowed) and positions.
// Multiply lags by the frame interval Δt in the caller.

export interface Track { t: number[]; x: number[]; y: number[] }

export interface TrackNumbers {
  duration: number;      // frames between first and last point
  net: number;           // distance first → last
  path: number;          // sum of step lengths
  meanStep: number;      // mean step length per frame (path / duration)
  straightness: number;  // net / path (1 = straight line)
}

export function trackNumbers(tr: Track): TrackNumbers {
  const n = tr.t.length;
  let path = 0;
  for (let i = 1; i < n; i++) path += Math.hypot(tr.x[i] - tr.x[i - 1], tr.y[i] - tr.y[i - 1]);
  const duration = n ? tr.t[n - 1] - tr.t[0] : 0, net = n ? Math.hypot(tr.x[n - 1] - tr.x[0], tr.y[n - 1] - tr.y[0]) : 0;
  return { duration, net, path, meanStep: duration ? path / duration : 0, straightness: path > 0 ? net / path : 1 };
}

/** Turning angles (radians, −π..π) between consecutive steps. */
export function turningAngles(tr: Track): number[] {
  const out: number[] = [];
  for (let i = 1; i + 1 < tr.t.length; i++) {
    const a1 = Math.atan2(tr.y[i] - tr.y[i - 1], tr.x[i] - tr.x[i - 1]);
    const a2 = Math.atan2(tr.y[i + 1] - tr.y[i], tr.x[i + 1] - tr.x[i]);
    let d = a2 - a1;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    out.push(d);
  }
  return out;
}

export interface MSD { lag: number[]; msd: number[]; n: number[] }

/** Sums of squared displacements and pair counts per lag 1..maxLag (lags from the time stamps). */
function msdSums(tr: Track, maxLag: number, sum: Float64Array, cnt: Float64Array, dims: 1 | 2): void {
  const n = tr.t.length;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const L = tr.t[j] - tr.t[i];
    if (L > maxLag) break;
    const dx = tr.x[j] - tr.x[i], dy = dims === 2 ? tr.y[j] - tr.y[i] : 0;
    sum[L] += dx * dx + dy * dy; cnt[L]++;
  }
}

function finishMSD(sum: Float64Array, cnt: Float64Array, maxLag: number): MSD {
  const out: MSD = { lag: [], msd: [], n: [] };
  for (let L = 1; L <= maxLag; L++) if (cnt[L] > 0) { out.lag.push(L); out.msd.push(sum[L] / cnt[L]); out.n.push(cnt[L]); }
  return out;
}

/** Time-averaged MSD of one track: for each lag, the mean over all pairs of points that far apart. */
export function msdTrack(tr: Track, maxLag: number, dims: 1 | 2 = 2): MSD {
  const sum = new Float64Array(maxLag + 1), cnt = new Float64Array(maxLag + 1);
  msdSums(tr, maxLag, sum, cnt, dims);
  return finishMSD(sum, cnt, maxLag);
}

/** Ensemble MSD: all pairs of all tracks pooled per lag. */
export function msdEnsemble(tracks: Track[], maxLag: number, dims: 1 | 2 = 2): MSD {
  const sum = new Float64Array(maxLag + 1), cnt = new Float64Array(maxLag + 1);
  for (const tr of tracks) msdSums(tr, maxLag, sum, cnt, dims);
  return finishMSD(sum, cnt, maxLag);
}

/** Least squares y = a + b x (or y = b x through the origin). */
export function lineFit(x: number[], y: number[], throughOrigin = false): { slope: number; offset: number } {
  const n = x.length;
  if (!n) return { slope: NaN, offset: NaN };
  if (throughOrigin) {
    let sxy = 0, sxx = 0;
    for (let i = 0; i < n; i++) { sxy += x[i] * y[i]; sxx += x[i] * x[i]; }
    return { slope: sxy / sxx, offset: 0 };
  }
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i]; }
  const d = n * sxx - sx * sx;
  if (Math.abs(d) < 1e-300) return { slope: 0, offset: sy / n };
  const slope = (n * sxy - sx * sy) / d;
  return { slope, offset: (sy - slope * sx) / n };
}

/** Power law MSD = K τ^α, fitted as a line on log–log axes (positive values only). */
export function powerFit(lag: number[], msd: number[]): { alpha: number; K: number } {
  const lx: number[] = [], ly: number[] = [];
  lag.forEach((L, i) => { if (L > 0 && msd[i] > 0) { lx.push(Math.log(L)); ly.push(Math.log(msd[i])); } });
  const f = lineFit(lx, ly);
  return { alpha: f.slope, K: Math.exp(f.offset) };
}

/** MSD = c + a τ + b τ² by least squares (diffusion + directed + offset); a = 2dD, b = v². */
export function quadFit(lag: number[], msd: number[]): { c: number; a: number; b: number } {
  // normal equations for [1, τ, τ²]
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], r = [0, 0, 0];
  lag.forEach((L, i) => {
    const f = [1, L, L * L];
    for (let p = 0; p < 3; p++) { r[p] += f[p] * msd[i]; for (let q = 0; q < 3; q++) A[p][q] += f[p] * f[q]; }
  });
  const x = solve3(A, r);
  return { c: x[0], a: x[1], b: x[2] };
}

function solve3(A: number[][], b: number[]): number[] {
  const M = A.map((row, i) => [...row, b[i]]), n = 3;
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    if (Math.abs(M[c][c]) < 1e-300) return [NaN, NaN, NaN];
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let j = c; j <= n; j++) M[r][j] -= f * M[c][j]; }
  }
  return M.map((row, i) => row[n] / M[i][i]);
}

/** Confined: MSD = c + L (1 − e^(−τ/τc)); τc by a grid search, L and c by least squares. */
export function plateauFit(lag: number[], msd: number[]): { plateau: number; tau: number; c: number } {
  let best = { plateau: NaN, tau: NaN, c: NaN }, bestErr = Infinity;
  const tmax = Math.max(...lag, 1);
  for (let k = 0; k <= 120; k++) {
    const tau = 0.2 * Math.pow(tmax * 25, k / 120);
    const g = lag.map(L => 1 - Math.exp(-L / tau));
    const f = lineFit(g, msd);
    let err = 0;
    lag.forEach((_, i) => { err += (f.offset + f.slope * g[i] - msd[i]) ** 2; });
    if (err < bestErr) { bestErr = err; best = { plateau: f.slope, tau, c: f.offset }; }
  }
  return best;
}

/** Diffusion coefficient from the first lags: slope / (2 dims Δt), and the offset. */
export function diffusionFromMSD(m: MSD, nFit: number, dims: 1 | 2, dt = 1): { D: number; offset: number } {
  const k = Math.max(2, Math.min(nFit, m.lag.length));
  const f = lineFit(m.lag.slice(0, k).map(L => L * dt), m.msd.slice(0, k));
  return { D: f.slope / (2 * dims), offset: f.offset };
}

/** Persistent random walk in 2D (speed s, persistence time P): MSD = 2 s² P (τ − P (1 − e^(−τ/P))). */
export const persistentMSD = (tau: number, s: number, P: number): number => 2 * s * s * P * (tau - P * (1 - Math.exp(-tau / P)));
