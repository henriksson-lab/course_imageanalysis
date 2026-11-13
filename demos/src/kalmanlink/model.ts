// Model of the kalmanlink demo (no DOM): the simulated scenes and the three trackers.
import { rng, gauss, bernoulli, uniform } from "../lib/random";
import * as LK from "../lib/linking";
import * as KF from "../lib/kalman";
import { simParticles, SimMovie, SimDet } from "../lib/tracksim";

export type Method = "nn" | "last" | "kalman";
export type Scene = "cross" | "random" | "turn" | "mixed";
export interface State { scene: Scene; n: number; speed: number; sigma: number; miss: number; Q: number; gate: number; radius: number; method: Method; t: number }
export const T = 50, W = 256, MAXMISS = 2;
export const DEFAULT: State = { scene: "cross", n: 12, speed: 6, sigma: 1.5, miss: 0, Q: 0.01, gate: 3.5, radius: 20, method: "kalman", t: 26 };
export const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "cross", label: "particles cross", s: { scene: "cross", n: 12, t: 26 } },
  { id: "stream", label: "a dense stream", s: { scene: "random", n: 60, speed: 6, radius: 12, t: 25 } },
  { id: "turn", label: "a sudden turn: the gate misses", s: { scene: "turn", n: 16, speed: 7, Q: 0.002, sigma: 1, gate: 3, t: 26 } },
  { id: "missed", label: "missed frames: the ellipse grows", s: { scene: "cross", n: 12, miss: 0.25, t: 26 } },
  { id: "mixed", label: "mixed: waiting + running", s: { scene: "mixed", n: 40, speed: 7, t: 25 } },
];
export const METHODS: [Method, string][] = [["nn", "nearest (last)"], ["last", "global (last)"], ["kalman", "global + Kalman"]];


/** Crossing pairs or a sudden turn: straight lines with noise and misses, built here. */
export function simLines(s: State, seed: number): SimMovie {
  const r = rng(seed), dets: SimDet[] = [], truth: LK.Edge[] = [];
  interface P { x0: number; y0: number; vx: number; vy: number }
  const ps: P[] = [];
  if (s.scene === "cross") {
    const pairs = Math.max(1, Math.round(s.n / 2));
    for (let k = 0; k < pairs; k++) {
      const cx = uniform(r, 50, W - 50), cy = uniform(r, 50, W - 50), phi = uniform(r, 0, 2 * Math.PI), dl = uniform(r, 0.25, 0.45);
      const tc = uniform(r, 18, 32);
      for (const sg of [1, -1]) {
        const vx = s.speed * Math.cos(phi + sg * dl), vy = s.speed * Math.sin(phi + sg * dl);
        ps.push({ x0: cx - vx * tc, y0: cy - vy * tc, vx, vy });
      }
    }
  } else {
    for (let k = 0; k < s.n; k++) {
      const phi = uniform(r, 0, 2 * Math.PI);
      ps.push({ x0: uniform(r, 20, W - 20) - s.speed * 25 * Math.cos(phi), y0: uniform(r, 20, W - 20) - s.speed * 25 * Math.sin(phi), vx: s.speed * Math.cos(phi), vy: s.speed * Math.sin(phi) });
    }
  }
  const last = new Int32Array(ps.length).fill(-1);
  for (let t = 0; t < T; t++) {
    ps.forEach((p, i) => {
      let x: number, y: number;
      if (s.scene === "turn" && t > 25) {
        const xt = p.x0 + 25 * p.vx, yt = p.y0 + 25 * p.vy;
        x = xt - (t - 25) * p.vy; y = yt + (t - 25) * p.vx;   // 90° turn
      } else { x = p.x0 + t * p.vx; y = p.y0 + t * p.vy; }
      if (x < 0 || x > W || y < 0 || y > W) { last[i] = -1; return; }
      if (bernoulli(r, s.miss)) return;
      const k = dets.length;
      dets.push({ t, x: x + gauss(r, 0, s.sigma), y: y + gauss(r, 0, s.sigma), area: 1, bright: 1, pid: i, kind: "real" });
      if (last[i] >= 0) truth.push({ a: last[i], b: k, kind: t - dets[last[i]].t > 1 ? "gap" : "link" });
      last[i] = k;
    });
  }
  return { dets, truth, T, W, H: W, drift: [] };
}

export function simulate(s: State): SimMovie {
  if (s.scene === "cross" || s.scene === "turn") return simLines(s, 5);
  return simParticles({ n: s.n, T, W, H: W, motion: s.scene === "mixed" ? "switching" : "cv", D: 0.05, dt: 1, speed: s.speed, accel: 0.06 * s.speed, sigma: s.sigma, miss: s.miss, switchProb: 0.08, seed: 9 });
}

export interface Track { id: number; last: number; missed: number; kf?: KF.KState; pred?: KF.KState }
export interface Result { edges: LK.Edge[]; trackOf: Int32Array; gates: { t: number; track: number; mean: [number, number]; cov: [[number, number], [number, number]] }[] }

export function track(mv: SimMovie, s: State, method: Method): Result {
  const frames = LK.byFrame(mv.dets, T), R = s.sigma * s.sigma;
  const model = KF.constantVelocity(2, 1, s.Q, R);
  const tracks: Track[] = [], edges: LK.Edge[] = [], trackOf = new Int32Array(mv.dets.length).fill(-1), gates: Result["gates"] = [];
  let active: Track[] = [];
  const newTrack = (k: number): Track => {
    const d = mv.dets[k], v0 = (1.5 * s.speed) ** 2;
    const tr: Track = { id: tracks.length, last: k, missed: 0, kf: { x: [d.x, 0, d.y, 0], P: [[R, 0, 0, 0], [0, v0, 0, 0], [0, 0, R, 0], [0, 0, 0, v0]] } };
    tracks.push(tr); trackOf[k] = tr.id;
    return tr;
  };
  for (const k of frames[0] ?? []) active.push(newTrack(k));
  for (let t = 1; t < T; t++) {
    const D = frames[t] ?? [], m = active.length, n = D.length;
    const C = new Float64Array(m * n).fill(Infinity), death = new Float64Array(m);
    const g2 = method === "kalman" ? s.gate * s.gate : s.radius * s.radius;
    active.forEach((tr, i) => {
      if (method === "kalman") {
        tr.pred = KF.predict(tr.kf!, model);
        const S0 = KF.innovation(tr.pred, [0, 0], model);
        const Si = S0.Sinv, px = tr.pred.x[0], py = tr.pred.x[2];
        gates.push({ t, track: tr.id, mean: [px, py], cov: [[S0.S[0][0], S0.S[0][1]], [S0.S[1][0], S0.S[1][1]]] });
        death[i] = g2 / 2 + S0.logdetS;
        D.forEach((k, j) => {
          const yx = mv.dets[k].x - px, yy = mv.dets[k].y - py;
          const d2 = yx * (Si[0][0] * yx + Si[0][1] * yy) + yy * (Si[1][0] * yx + Si[1][1] * yy);
          if (d2 <= g2) C[i * n + j] = d2 + S0.logdetS;
        });
      } else {
        const p = mv.dets[tr.last];
        gates.push({ t, track: tr.id, mean: [p.x, p.y], cov: [[g2, 0], [0, g2]] });
        death[i] = g2 / 2;
        D.forEach((k, j) => {
          const d2 = (mv.dets[k].x - p.x) ** 2 + (mv.dets[k].y - p.y) ** 2;
          if (d2 <= g2) C[i * n + j] = d2;
        });
      }
    });
    let links: { i: number; j: number }[];
    if (method === "nn") {
      // every track takes its nearest detection; a detection wanted by several goes to the closest
      const want = LK.nearestNeighbour(C, m, n), best = new Map<number, number>();
      want.forEach((j, i) => { if (j >= 0 && (!best.has(j) || C[i * n + j] < C[best.get(j)! * n + j])) best.set(j, i); });
      links = [...best].map(([j, i]) => ({ i, j }));
    } else links = LK.assignGlobal(C, m, n, death, g2 / 2).links;
    const linkedT = new Set<number>(), linkedD = new Set<number>(), next: Track[] = [];
    for (const { i, j } of links) {
      const tr = active[i], k = D[j];
      edges.push({ a: tr.last, b: k, kind: mv.dets[k].t - mv.dets[tr.last].t > 1 ? "gap" : "link" });
      trackOf[k] = tr.id; tr.last = k; tr.missed = 0;
      if (method === "kalman") tr.kf = KF.update(tr.pred!, [mv.dets[k].x, mv.dets[k].y], model).state;
      linkedT.add(i); linkedD.add(j); next.push(tr);
    }
    active.forEach((tr, i) => {
      if (linkedT.has(i)) return;
      tr.missed++;
      if (method === "kalman") tr.kf = tr.pred;
      if (tr.missed <= MAXMISS) next.push(tr);
    });
    D.forEach((k, j) => { if (!linkedD.has(j)) next.push(newTrack(k)); });
    active = next;
  }
  return { edges, trackOf, gates };
}

/** Identity switches: along each true particle, how often the assigned track changes. */
export function switches(mv: SimMovie, trackOf: Int32Array): number {
  let sw = 0;
  for (const e of mv.truth) if (trackOf[e.a] !== trackOf[e.b]) sw++;
  return sw;
}

