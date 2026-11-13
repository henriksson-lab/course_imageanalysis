// Model of the linkmovie demo (no DOM): simulation, linking and the displacement/spacing ratio.
import * as LK from "../lib/linking";
import { simParticles, SimMovie, meanSpacing } from "../lib/tracksim";

export type Method = LK.LinkMethod;
export type Motion = "rw" | "directed" | "mixed";
export interface State { n: number; D: number; dt: number; sigma: number; speed: number; motion: Motion; method: Method; gate: number; bd: number; drift: boolean; correct: boolean; t: number; seed: number }
export const T = 60, W = 256;
export const GATE_OFF = 60;
export const DEFAULT: State = { n: 60, D: 4, dt: 1, sigma: 0.5, speed: 4, motion: "rw", method: "global", gate: 20, bd: 15, drift: false, correct: false, t: 30, seed: 1 };
export const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "easy", label: "sparse and slow: everything works", s: { n: 30, D: 2, method: "nn" } },
  { id: "half", label: "ratio 0.5: NN fails, global better", s: { n: 150, D: 6.8, method: "nn", gate: 25, bd: 15 } },
  { id: "one", label: "ratio 1: nobody can", s: { n: 150, D: 27, method: "global", gate: 40, bd: 30 } },
  { id: "drift", label: "drift", s: { n: 60, D: 2, drift: true, correct: false, gate: 10, bd: 7 } },
  { id: "blow", label: "b too low: tracks shatter", s: { n: 60, D: 4, bd: 3 } },
  { id: "bhigh", label: "b too high: wrong links", s: { n: 100, D: 4, gate: GATE_OFF, bd: 60 } },
];
export const METHODS: [Method, string][] = [["nn", "nearest"], ["greedy", "greedy"], ["global", "global"]];
export const MCOL: Record<Method, string> = { nn: "#d62728", greedy: "#ff7f0e", global: "#1f77b4" };

export const DRIFT: [number, number] = [6, 3];
export const ratioOf = (s: State, motion = s.motion) => {
  const msd = 4 * s.D * s.dt + (motion === "rw" ? 0 : motion === "directed" ? 1 : 0.5) * (s.speed * s.dt) ** 2
    + (s.drift && !s.correct ? DRIFT[0] ** 2 + DRIFT[1] ** 2 : 0);
  return Math.sqrt(msd) / meanSpacing(s.n / (W * W));
};

export function simulate(s: State): SimMovie {
  return simParticles({ n: s.n, T, W, H: W, motion: s.motion, D: s.D, dt: s.dt, speed: s.speed, sigma: s.sigma, drift: s.drift ? DRIFT : undefined, seed: s.seed });
}

export function link(mv: SimMovie, s: State, method: Method): LK.Edge[] {
  const frames = LK.byFrame(mv.dets, T);
  const corr = (k: number) => {
    const d = mv.dets[k];
    return s.correct ? { x: d.x - mv.drift[d.t][0], y: d.y - mv.drift[d.t][1] } : d;
  };
  return LK.linkFrames(mv.dets, frames, { method, gate: s.gate >= GATE_OFF ? Infinity : s.gate, b: s.bd * s.bd, posA: corr, posB: corr });
}

