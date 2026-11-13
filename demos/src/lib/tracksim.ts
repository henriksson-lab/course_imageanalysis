// Seeded simulators with ground truth for the tracking day. No DOM.
//
// - simParticles: N particles in a W × H field (random walk, directed, nearly constant
//   velocity, waiting/running), entering and leaving at the border, localisation noise,
//   missed detections, stage drift. Output: detections per frame + the true links.
// - simCells: dividing cells (gamma cycle times, daughters on both sides, area halving and
//   regrowth, a bright frame before division), cells entering at the border, missed and false
//   detections, segmentation splits (one cell → two pieces for one frame). Output: detections,
//   the true links (link / gap / split) and the true lineage.
// - simTracks: single tracks for MSD analysis (free, directed, confined, anomalous = fractional
//   Brownian motion by circulant embedding, switching, persistent), in physical units.
import { rng, gauss, bernoulli, poisson, uniform, Rng } from "./random";
import { fft1d } from "./fft";
import type { Edge } from "./linking";
import type { Track } from "./trackstats";

export interface SimDet {
  t: number; x: number; y: number;
  area: number; bright: number;
  /** true object id (-1: false detection or the extra piece of a segmentation split) */
  pid: number;
  kind: "real" | "false" | "piece";
}

export interface ParticleOptions {
  /** mean number of particles in the field */
  n: number;
  T: number;
  W: number; H: number;
  motion: "rw" | "directed" | "mixed" | "cv" | "switching";
  /** diffusion coefficient (units² per time unit) and frame interval */
  D: number; dt: number;
  /** speed (units per time unit) for directed / cv / running particles */
  speed: number;
  /** cv: velocity change per frame (sd, units per time unit) */
  accel?: number;
  /** localisation noise (sd) */
  sigma: number;
  /** probability that a detection is missed */
  miss?: number;
  /** stage drift per frame (added to every detection) */
  drift?: [number, number];
  /** switching: probability per frame to switch between waiting and running */
  switchProb?: number;
  /** extra margin around the field (particles there are simulated but not seen) */
  margin?: number;
  seed: number;
}

export interface SimMovie {
  dets: SimDet[];
  truth: Edge[];
  T: number; W: number; H: number;
  /** cumulative drift added at each frame */
  drift: [number, number][];
  /** true positions per particle per frame (NaN when outside the field; without drift) */
  truePos?: Float64Array[];
}

/** Particles in an enlarged periodic box; only the inner field is observed. */
export function simParticles(o: ParticleOptions): SimMovie {
  const r = rng(o.seed), M = o.margin ?? Math.max(20, 6 * Math.sqrt(2 * o.D * o.dt) + 3 * o.speed * o.dt + 3 * Math.hypot(o.drift?.[0] ?? 0, o.drift?.[1] ?? 0));
  const BW = o.W + 2 * M, BH = o.H + 2 * M;
  const N = Math.max(1, Math.round(o.n * BW * BH / (o.W * o.H)));
  const s = Math.sqrt(2 * o.D * o.dt);
  const px = new Float64Array(N), py = new Float64Array(N), vx = new Float64Array(N), vy = new Float64Array(N), run = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    px[i] = uniform(r, -M, o.W + M); py[i] = uniform(r, -M, o.H + M);
    const a = uniform(r, 0, 2 * Math.PI);
    const directed = o.motion === "directed" || o.motion === "cv" || (o.motion === "mixed" && i % 2 === 0) || o.motion === "switching";
    vx[i] = directed ? o.speed * Math.cos(a) : 0; vy[i] = directed ? o.speed * Math.sin(a) : 0;
    run[i] = o.motion === "switching" ? (r.unif() < 0.5 ? 1 : 0) : 1;
  }
  const dets: SimDet[] = [], truth: Edge[] = [], drift: [number, number][] = [];
  const last = new Int32Array(N).fill(-1);   // last detection index while continuously visible
  const truePos: Float64Array[] = Array.from({ length: N }, () => new Float64Array(2 * o.T).fill(NaN));
  let dx = 0, dy = 0;
  for (let t = 0; t < o.T; t++) {
    if (t > 0) {
      dx += o.drift?.[0] ?? 0; dy += o.drift?.[1] ?? 0;
      for (let i = 0; i < N; i++) {
        if (o.motion === "cv" && o.accel) {
          vx[i] += gauss(r, 0, o.accel); vy[i] += gauss(r, 0, o.accel);
          const sp = Math.hypot(vx[i], vy[i]) || 1;   // keep the speed, let the direction wander
          vx[i] *= o.speed / sp; vy[i] *= o.speed / sp;
        }
        if (o.motion === "switching" && bernoulli(r, o.switchProb ?? 0.05)) {
          run[i] = 1 - run[i];
          if (run[i]) { const a = uniform(r, 0, 2 * Math.PI); vx[i] = o.speed * Math.cos(a); vy[i] = o.speed * Math.sin(a); }
        }
        let nx = px[i] + gauss(r, 0, s) + (run[i] ? vx[i] * o.dt : 0);
        let ny = py[i] + gauss(r, 0, s) + (run[i] ? vy[i] * o.dt : 0);
        // the box is periodic around the observed window (which moves with the drift)
        let wrapped = false;
        if (nx + dx < -M) { nx += BW; wrapped = true; } else if (nx + dx > o.W + M) { nx -= BW; wrapped = true; }
        if (ny + dy < -M) { ny += BH; wrapped = true; } else if (ny + dy > o.H + M) { ny -= BH; wrapped = true; }
        if (wrapped) last[i] = -1;
        px[i] = nx; py[i] = ny;
      }
    }
    drift.push([dx, dy]);
    for (let i = 0; i < N; i++) {
      const ox = px[i] + dx, oy = py[i] + dy;
      const inside = ox >= 0 && ox <= o.W && oy >= 0 && oy <= o.H;
      if (!inside) { last[i] = -1; continue; }
      truePos[i][2 * t] = px[i]; truePos[i][2 * t + 1] = py[i];
      if (o.miss && bernoulli(r, o.miss)) continue;
      const k = dets.length;
      dets.push({ t, x: ox + gauss(r, 0, o.sigma), y: oy + gauss(r, 0, o.sigma), area: 1, bright: 1, pid: i, kind: "real" });
      if (last[i] >= 0) truth.push({ a: last[i], b: k, kind: t - dets[last[i]].t > 1 ? "gap" : "link" });
      last[i] = k;
    }
  }
  return { dets, truth, T: o.T, W: o.W, H: o.H, drift, truePos };
}

/** Mean nearest-neighbour spacing of random points at density n: 0.5 / √n. */
export const meanSpacing = (density: number): number => 0.5 / Math.sqrt(density);
/** rms displacement per frame of a 2D random walk: √(4 D Δt). */
export const rmsStep = (D: number, dt: number): number => Math.sqrt(4 * D * dt);
/** Nearest-neighbour link error estimate a/(1+a), a = 4π n D Δt. */
export function nnErrorEstimate(density: number, D: number, dt: number): number {
  const a = 4 * Math.PI * density * D * dt;
  return a / (1 + a);
}

// ---------------------------------------------------------------------------------------
// Dividing cells

export interface CellOptions {
  /** founder cells at frame 0 */
  n0: number;
  T: number;
  W: number; H: number;
  /** diffusion per frame (units² / frame) */
  D: number;
  /** cycle time: minimum + gamma(shape k, mean − minimum), in frames */
  cycleMean: number; cycleMin: number; cycleShape: number;
  /** radius of a newborn cell */
  radius: number;
  miss: number;
  /** false detections per frame (Poisson mean) */
  falseRate: number;
  /** probability that a cell is segmented into two pieces in a frame */
  splitProb: number;
  /** new cells entering at the left border per frame (Poisson mean) */
  enterRate: number;
  /** daughters are pushed apart along a direction within ±spread of the x axis */
  spread?: number;
  seed: number;
}

export interface SimCell { id: number; parent: number; t0: number; t1: number; divides: boolean; entered: boolean; children: number[] }
export interface CellMovie extends SimMovie { cells: SimCell[] }

function gammaSample(r: Rng, k: number, mean: number): number {
  // sum of k exponentials (k integer ≥ 1), scaled to the mean
  const kk = Math.max(1, Math.round(k));
  let s = 0;
  for (let i = 0; i < kk; i++) s += -Math.log(1 - r.unif());
  return s * mean / kk;
}

export function simCells(o: CellOptions): CellMovie {
  const r = rng(o.seed);
  interface Live { id: number; x: number; y: number; born: number; div: number; a0: number }
  const cells: SimCell[] = [], live: Live[] = [];
  const cycle = () => o.cycleMin + gammaSample(r, o.cycleShape, Math.max(0.5, o.cycleMean - o.cycleMin));
  const spawn = (x: number, y: number, parent: number, t: number, phase: number, entered: boolean): Live => {
    const id = cells.length, c = cycle();
    cells.push({ id, parent, t0: t, t1: t, divides: false, entered, children: [] });
    if (parent >= 0) cells[parent].children.push(id);
    const L: Live = { id, x, y, born: t - phase * c, div: t - phase * c + c, a0: Math.PI * o.radius ** 2 };
    live.push(L);
    return L;
  };
  for (let i = 0; i < o.n0; i++) {
    const x = o.W * (i + 0.5) / o.n0 + uniform(r, -0.2, 0.2) * o.W / o.n0;
    spawn(x, uniform(r, 0.25 * o.H, 0.75 * o.H), -1, 0, r.unif(), false);
  }
  const dets: SimDet[] = [], truth: Edge[] = [];
  const lastDet = new Map<number, number>();       // cell id → its last detection
  const motherLast = new Map<number, number>();    // daughter id → mother's last detection
  const s = Math.sqrt(2 * o.D), spread = o.spread ?? Math.PI / 5;
  for (let t = 0; t < o.T; t++) {
    if (t > 0) {
      // move, divide, leave
      const now = [...live];
      live.length = 0;
      for (const c of now) {
        c.x += gauss(r, 0, s); c.y += gauss(r, 0, s);
        if (c.y < o.radius) c.y = 2 * o.radius - c.y;
        if (c.y > o.H - o.radius) c.y = 2 * (o.H - o.radius) - c.y;
        if (c.x < 0 || c.x > o.W) { cells[c.id].t1 = t - 1; continue; }   // left the field
        if (t >= c.div) {
          cells[c.id].divides = true; cells[c.id].t1 = t - 1;
          const a = uniform(r, -spread, spread) + (r.unif() < 0.5 ? 0 : Math.PI), d = 0.9 * o.radius;
          for (const sg of [1, -1]) {
            const L = spawn(c.x + sg * d * Math.cos(a), c.y + sg * d * Math.sin(a), c.id, t, 0, false);
            const md = lastDet.get(c.id);
            if (md !== undefined) motherLast.set(L.id, md);
          }
          continue;
        }
        live.push(c);
      }
      // cells do not overlap: push pairs closer than 1.8 radii apart (a few relaxation sweeps)
      const dmin = 1.8 * o.radius;
      for (let sweep = 0; sweep < 3; sweep++) for (let i = 0; i < live.length; i++) for (let j = i + 1; j < live.length; j++) {
        const a = live[i], b = live[j], ddx = b.x - a.x, ddy = b.y - a.y, dd = Math.hypot(ddx, ddy);
        if (dd >= dmin || dd === 0) continue;
        const push = (dmin - dd) / 2 / dd;
        a.x -= ddx * push; a.y -= ddy * push; b.x += ddx * push; b.y += ddy * push;
      }
      for (const c of live) c.y = Math.min(o.H - o.radius, Math.max(o.radius, c.y));
      const nEnter = poisson(r, o.enterRate);
      for (let k = 0; k < nEnter; k++) spawn(uniform(r, 0, 0.6 * o.radius), uniform(r, o.radius, o.H - o.radius), -1, t, r.unif(), true);
    }
    for (const c of live) {
      cells[c.id].t1 = t;
      const phase = Math.min(1, Math.max(0, (t - c.born) / Math.max(1, c.div - c.born)));
      const area = c.a0 * (1 + phase);
      const bright = c.div - t <= 1 ? 2 : 1;
      if (bernoulli(r, o.miss)) continue;
      const k = dets.length;
      if (o.splitProb > 0 && bernoulli(r, o.splitProb)) {
        // a fragment (35 % of the area) is cut off; the larger part keeps the cell's identity
        const a = uniform(r, 0, 2 * Math.PI), d = Math.sqrt(area / Math.PI);
        dets.push({ t, x: c.x - 0.3 * d * Math.cos(a), y: c.y - 0.3 * d * Math.sin(a), area: 0.65 * area, bright, pid: c.id, kind: "real" });
        dets.push({ t, x: c.x + 0.6 * d * Math.cos(a), y: c.y + 0.6 * d * Math.sin(a), area: 0.35 * area, bright, pid: -1, kind: "piece" });
      } else dets.push({ t, x: c.x, y: c.y, area, bright, pid: c.id, kind: "real" });
      const prev = lastDet.get(c.id);
      if (prev !== undefined) truth.push({ a: prev, b: k, kind: t - dets[prev].t > 1 ? "gap" : "link" });
      else if (motherLast.has(c.id)) truth.push({ a: motherLast.get(c.id)!, b: k, kind: "split" });
      lastDet.set(c.id, k);
    }
    const nf = poisson(r, o.falseRate);
    for (let k = 0; k < nf; k++) {
      dets.push({ t, x: uniform(r, 0, o.W), y: uniform(r, 0, o.H), area: Math.PI * o.radius ** 2 * uniform(r, 0.3, 1), bright: 1, pid: -1, kind: "false" });
    }
  }
  // detections ordered by frame already
  return { dets, truth, T: o.T, W: o.W, H: o.H, drift: Array.from({ length: o.T }, () => [0, 0] as [number, number]), cells };
}

// ---------------------------------------------------------------------------------------
// Single tracks for MSD analysis (physical units: D in µm²/s, v in µm/s, Δt in s)

export type MotionType = "free" | "directed" | "confined" | "anomalous" | "switching" | "persistent";
export interface TrackOptions {
  type: MotionType;
  /** number of tracks and points per track */
  n: number; N: number;
  D: number; v: number; R: number; alpha: number; dt: number;
  /** localisation error (sd per axis) */
  sigma: number;
  /** persistence time (s) for the persistent walk */
  P?: number;
  /** switching probability per frame (waiting ↔ running) */
  switchProb?: number;
  seed: number;
}

/** Fractional Gaussian noise of length N (unit variance, Hurst H = α/2) by circulant embedding. */
export function fgn(r: Rng, N: number, alpha: number): Float64Array {
  const H2 = alpha;   // 2H
  let M = 2;
  while (M < 2 * N) M *= 2;
  const half = M / 2;
  const gam = (k: number) => 0.5 * (Math.pow(Math.abs(k + 1), H2) - 2 * Math.pow(Math.abs(k), H2) + Math.pow(Math.abs(k - 1), H2));
  const re = new Float64Array(M), im = new Float64Array(M);
  for (let k = 0; k <= half; k++) re[k] = gam(k);
  for (let k = 1; k < half; k++) re[M - k] = gam(k);
  fft1d(re, im);
  const lam = Float64Array.from(re, v => Math.max(0, v));
  for (let k = 0; k < M; k++) {
    const s = Math.sqrt(lam[k] / M);
    re[k] = s * r.norm(); im[k] = s * r.norm();
  }
  fft1d(re, im);
  return re.slice(0, N);
}

export function simTracks(o: TrackOptions): Track[] {
  const r = rng(o.seed), out: Track[] = [];
  const s = Math.sqrt(2 * o.D * o.dt);
  for (let k = 0; k < o.n; k++) {
    const t: number[] = [], X: number[] = [], Y: number[] = [];
    let x = 0, y = 0;
    if (o.type === "confined") { const a = uniform(r, 0, 2 * Math.PI), q = o.R * Math.sqrt(r.unif()); x = q * Math.cos(a); y = q * Math.sin(a); }
    let fx: Float64Array | null = null, fy: Float64Array | null = null;
    if (o.type === "anomalous") { fx = fgn(r, o.N, o.alpha); fy = fgn(r, o.N, o.alpha); }
    // anomalous: MSD per axis = 2 D (τ in s)^α, so a step of one frame has variance 2 D Δt^α
    const sa = Math.sqrt(2 * o.D * Math.pow(o.dt, o.alpha));
    let theta = uniform(r, 0, 2 * Math.PI), running = r.unif() < 0.5;
    const dir = uniform(r, 0, 2 * Math.PI);
    for (let i = 0; i < o.N; i++) {
      if (i > 0) {
        switch (o.type) {
          case "free": x += gauss(r, 0, s); y += gauss(r, 0, s); break;
          case "directed": x += gauss(r, 0, s) + o.v * o.dt * Math.cos(dir); y += gauss(r, 0, s) + o.v * o.dt * Math.sin(dir); break;
          case "confined": {
            x += gauss(r, 0, s); y += gauss(r, 0, s);
            const q = Math.hypot(x, y);
            if (q > o.R) { const f = Math.max(0, 2 * o.R - q) / q; x *= f; y *= f; }
            break;
          }
          case "anomalous": x += sa * fx![i - 1]; y += sa * fy![i - 1]; break;
          case "switching":
            if (bernoulli(r, o.switchProb ?? 0.05)) { running = !running; theta = uniform(r, 0, 2 * Math.PI); }
            x += gauss(r, 0, s) + (running ? o.v * o.dt * Math.cos(theta) : 0);
            y += gauss(r, 0, s) + (running ? o.v * o.dt * Math.sin(theta) : 0);
            break;
          case "persistent": {
            const P = o.P ?? 10, sub = 10, h = o.dt / sub;
            for (let j = 0; j < sub; j++) {
              theta += gauss(r, 0, Math.sqrt(2 * h / P));
              x += o.v * h * Math.cos(theta); y += o.v * h * Math.sin(theta);
            }
            break;
          }
        }
      }
      t.push(i); X.push(x + gauss(r, 0, o.sigma)); Y.push(y + gauss(r, 0, o.sigma));
    }
    out.push({ t, x: X, y: Y });
  }
  return out;
}
