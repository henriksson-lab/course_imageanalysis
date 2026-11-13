// Model of the msdplot demo (no DOM): simulated 2D tracks (physical units: µm, s), their MSD
// (per track and ensemble), fits, per-track D, and the "speed trap" curve.
import { simTracks, MotionType } from "../lib/tracksim";
import * as TS from "../lib/trackstats";

export interface State {
  type: MotionType; n: number; N: number;
  D: number; v: number; R: number; alpha: number; P: number;
  sigma: number; dt: number;
  /** fit the first fitFrac of the lags shown */
  fitFrac: number;
  loglog: boolean;
  mode: "msd" | "speed";
  seed: number;
}
export const DEFAULT: State = { type: "free", n: 40, N: 200, D: 0.1, v: 0.5, R: 0.5, alpha: 0.5, P: 10, sigma: 0, dt: 0.1, fitFrac: 0.25, loglog: true, mode: "msd", seed: 1 };
export const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "free", label: "free diffusion: slope 1", s: { type: "free", loglog: true } },
  { id: "directed", label: "directed: slope 2 at long lags", s: { type: "directed", v: 0.5, D: 0.1, loglog: true } },
  { id: "confined", label: "confined: plateau at R²", s: { type: "confined", R: 0.5, D: 0.1, N: 400, loglog: false } },
  { id: "locerr", label: "localisation error: the offset", s: { type: "free", D: 0.01, dt: 0.05, sigma: 0.03, loglog: false, fitFrac: 0.1 } },
  { id: "short", label: "short tracks: D scatters", s: { type: "free", N: 20, n: 100, loglog: false } },
  { id: "speed", label: "the speed trap", s: { mode: "speed", type: "free", D: 0.1 } },
  { id: "tail", label: "noisy tail: fit only the start", s: { type: "free", n: 10, N: 100, loglog: false, fitFrac: 1 } },
  { id: "anomalous", label: "anomalous: α = 0.5", s: { type: "anomalous", alpha: 0.5, D: 0.1, N: 256, loglog: true } },
  { id: "persistent", label: "a persistent cell: from 2 to 1", s: { type: "persistent", v: 0.2, P: 10, dt: 1, N: 400, loglog: true, fitFrac: 1 } },
];

export const MAXLAG = 250, PER_TRACK_LAGS = 4;
export function simulate(s: State): TS.Track[] {
  return simTracks({ type: s.type, n: s.n, N: s.N, D: s.D, v: s.v, R: s.R, alpha: s.alpha, dt: s.dt, sigma: s.sigma, P: s.P, switchProb: 0.03, seed: s.seed });
}

export interface Fits {
  ens: TS.MSD;            // lags in frames
  maxLag: number;
  nFit: number;
  line: { slope: number; offset: number };   // MSD vs τ (s), first nFit lags
  D: number;              // slope / 4
  alpha: number; K: number;
  quad: { c: number; a: number; b: number };
  plateau: { plateau: number; tau: number; c: number };
  perTrackD: number[];
  perTrack: TS.MSD[];
}

export function analyse(s: State, tracks: TS.Track[]): Fits {
  const maxLag = Math.max(2, Math.min(MAXLAG, s.N - 1));
  const ens = TS.msdEnsemble(tracks, maxLag);
  const nFit = Math.max(2, Math.round(s.fitFrac * ens.lag.length));
  const tau = ens.lag.map(L => L * s.dt);
  const line = TS.lineFit(tau.slice(0, nFit), ens.msd.slice(0, nFit));
  const pw = TS.powerFit(tau.slice(0, nFit), ens.msd.slice(0, nFit));
  const quad = TS.quadFit(tau.slice(0, nFit), ens.msd.slice(0, nFit));
  const plateau = TS.plateauFit(tau.slice(0, nFit), ens.msd.slice(0, nFit));
  const perTrack = tracks.map(t => TS.msdTrack(t, maxLag));
  // per track: a line with offset through the first PER_TRACK_LAGS lags (the most precise ones)
  const perTrackD = perTrack.map(m => {
    const k = Math.min(PER_TRACK_LAGS, m.lag.length);
    return TS.lineFit(m.lag.slice(0, k).map(L => L * s.dt), m.msd.slice(0, k)).slope / 4;
  });
  return { ens, maxLag, nFit, line, D: line.slope / 4, alpha: pw.alpha, K: pw.K, quad, plateau, perTrackD, perTrack };
}

/** The speed trap: one long free track subsampled every k frames; mean step length / (k Δt0). */
export function speedCurve(s: State): { dt: number[]; speed: number[] } {
  const dt0 = 0.005, N = 8192;
  const [tr] = simTracks({ type: "free", n: 1, N, D: s.D, v: 0, R: 1, alpha: 1, dt: dt0, sigma: 0, seed: s.seed + 100 });
  // localisation noise is added per observed frame, after subsampling
  let seed = 12345;
  const rnd = () => { seed = (seed * 1103515245 + 12345) >>> 0; return seed / 4294967296; };
  const gaussN = () => Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd());
  const out = { dt: [] as number[], speed: [] as number[] };
  for (let k = 1; k <= 512; k *= 2) {
    let sum = 0, cnt = 0, px = NaN, py = NaN;
    for (let i = 0; i < N; i += k) {
      const x = tr.x[i] + s.sigma * gaussN(), y = tr.y[i] + s.sigma * gaussN();
      if (i > 0) { sum += Math.hypot(x - px, y - py); cnt++; }
      px = x; py = y;
    }
    out.dt.push(k * dt0); out.speed.push(sum / cnt / (k * dt0));
  }
  return out;
}

/** Expected apparent speed of a 2D random walk with localisation error σ: √(π (D Δt + σ²)) / Δt. */
export const apparentSpeed = (D: number, dt: number, sigma: number): number => Math.sqrt(Math.PI * (D * dt + sigma * sigma)) / dt;
