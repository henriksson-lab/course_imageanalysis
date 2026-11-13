// Model of the kalman1d demo (no DOM): the truth, the measurements and the filter run.
import * as KF from "../lib/kalman";
import { rng } from "../lib/random";

export const T = 40, TURN = 20;
export type Model = "rw" | "cv";
export interface State { model: Model; Q: number; R: number; turn: boolean; missed: number[]; k: number; phase: 0 | 1 }
export const DEFAULT: State = { model: "cv", Q: 0.05, R: 2, turn: true, missed: [], k: 1, phase: 0 };
export const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "meas", label: "trust the measurements (R small)", s: { model: "cv", Q: 0.05, R: 0.1, k: 25, phase: 1 } },
  { id: "model", label: "trust the model (Q small): lag at the turn", s: { model: "cv", Q: 0.001, R: 4, k: 30, phase: 1 } },
  { id: "rwlag", label: "random walk on a moving object: always behind", s: { model: "rw", Q: 0.1, R: 4, k: 15, phase: 1 } },
  { id: "cv", label: "constant velocity: locks on", s: { model: "cv", Q: 0.01, R: 4, k: 15, phase: 1, turn: false } },
  { id: "missed", label: "three missed frames: the band widens", s: { model: "cv", Q: 0.05, R: 2, missed: [12, 13, 14], k: 14, phase: 1, turn: false } },
  { id: "golden", label: "golden ratio (random walk, Q = R)", s: { model: "rw", Q: 1, R: 1, turn: false, k: 39, phase: 1 } },
];
export const C_PRED = "var(--qb-c1, #1f77b4)", C_MEAS = "var(--qb-c4, #ff7f0e)", C_EST = "#222", C_TRUE = "#999";

// fixed noise pattern, scaled by √R
export const NOISE = (() => { const r = rng(17); return Array.from({ length: T }, () => r.norm()); })();
export const truth = (t: number, turn: boolean) => (!turn || t <= TURN ? 2 + t : 2 + TURN - 0.8 * (t - TURN));

export interface StepRec { xPred: number; pPred: number; z: number; missed: boolean; x: number; p: number; K: number; v: number }
export function run(s: State): StepRec[] {
  const zs = Array.from({ length: T }, (_, t) => truth(t, s.turn) + Math.sqrt(s.R) * NOISE[t]);
  const mdl = s.model === "rw" ? KF.randomWalk(1, s.Q, s.R) : KF.constantVelocity(1, 1, s.Q, s.R);
  let st: KF.KState = s.model === "rw" ? { x: [zs[0]], P: [[s.R]] } : { x: [zs[0], 0], P: [[s.R, 0], [0, 4]] };
  const out: StepRec[] = [{ xPred: zs[0], pPred: s.R, z: zs[0], missed: false, x: zs[0], p: s.R, K: 1, v: 0 }];
  for (let t = 1; t < T; t++) {
    const pr = KF.predict(st, mdl);
    const missed = s.missed.includes(t);
    let K = 0;
    if (!missed) { const u = KF.update(pr, [zs[t]], mdl); st = u.state; K = u.K[0][0]; } else st = pr;
    out.push({ xPred: pr.x[0], pPred: pr.P[0][0], z: zs[t], missed, x: st.x[0], p: st.P[0][0], K, v: s.model === "cv" ? st.x[1] : 0 });
  }
  return out;
}

