// The optimiser race of the optimisers unit: plain gradient descent, momentum and Adam on
// loss surfaces over two weights (a, b), with learning-rate schedules, gradient noise and a
// scale factor on the loss. The same surfaces as code/training/src/optimisers/run.rs.
// No DOM, so the exercise numbers come from this code (demos/test/training-test.ts).
import { rng } from "../lib/ui";

export type SurfaceId = "valley" | "saddle" | "curved" | "bowl";
export interface Surface {
  id: SurfaceId; label: string;
  f(a: number, b: number): number;
  grad(a: number, b: number): [number, number];
  min: number;
  minima: [number, number][];
  start: [number, number];
}

export const SURFACES: Surface[] = [
  {
    id: "valley", label: "long valley", min: 0, minima: [[0, 0]], start: [-1.8, 0.8],
    f: (a, b) => 0.1 * a * a + 5 * b * b, grad: (a, b) => [0.2 * a, 10 * b],
  },
  {
    id: "saddle", label: "saddle", min: -0.25, minima: [[0, -1], [0, 1]], start: [-1.8, 0.001],
    f: (a, b) => 0.5 * a * a - 0.5 * b * b + 0.25 * b ** 4, grad: (a, b) => [a, -b + b ** 3],
  },
  {
    id: "curved", label: "curved valley", min: 0, minima: [[1, 1]], start: [-1.5, 1.5],
    f: (a, b) => 0.5 * (1 - a) ** 2 + 5 * (b - a * a) ** 2,
    grad: (a, b) => [-(1 - a) - 20 * a * (b - a * a), 10 * (b - a * a)],
  },
  {
    id: "bowl", label: "round bowl", min: 0, minima: [[0, 0]], start: [-1.8, 1.5],
    f: (a, b) => 0.5 * (a * a + b * b), grad: (a, b) => [a, b],
  },
];
export const surface = (id: string): Surface => SURFACES.find(s => s.id === id) ?? SURFACES[0];

export type OptId = "gd" | "momentum" | "adam";
export const OPTS: OptId[] = ["gd", "momentum", "adam"];
export const OPT_LABEL: Record<OptId, string> = { gd: "gradient descent", momentum: "momentum", adam: "Adam" };

export type Schedule = "constant" | "step" | "cosine" | "warmup";
export const SCHEDULE_LABEL: Record<Schedule, string> = { constant: "constant", step: "step decay", cosine: "cosine", warmup: "warm-up + cosine" };

/** Learning rate at step t (0-based) of T for the schedule: step decay divides by 10 at 50 % and
 *  75 % of the steps; cosine falls from η to 0; warm-up rises linearly over the first 10 %. */
export function lrAt(s: Schedule, eta: number, t: number, T: number): number {
  if (s === "constant") return eta;
  if (s === "step") return eta * (t < 0.5 * T ? 1 : t < 0.75 * T ? 0.1 : 0.01);
  const W = s === "warmup" ? Math.round(0.1 * T) : 0;
  if (t < W) return eta * (t + 1) / W;
  return 0.5 * eta * (1 + Math.cos(Math.PI * (t - W) / Math.max(1, T - W)));
}

export interface RaceOptions {
  eta: number; schedule: Schedule; steps: number;
  noise: number;        // sd of the noise added to each gradient component
  scale: number;        // the loss multiplied by this (gradients too)
  beta: number;         // momentum
  biasCorrection: boolean;
  seed: number;
}
export const DEFAULT_RACE: RaceOptions = { eta: 0.1, schedule: "constant", steps: 200, noise: 0, scale: 1, beta: 0.9, biasCorrection: true, seed: 1 };

/** An optimiser walking on a surface, one step at a time. */
export class Walker {
  a: number; b: number;
  v = [0, 0]; m = [0, 0]; s = [0, 0]; t = 0;
  path: [number, number][];
  losses: number[];
  r: ReturnType<typeof rng>;
  diverged = false;
  constructor(public surf: Surface, public opt: OptId, public o: RaceOptions, start?: [number, number]) {
    [this.a, this.b] = start ?? surf.start;
    this.path = [[this.a, this.b]];
    this.losses = [surf.f(this.a, this.b)];
    this.r = rng(o.seed);
  }
  /** Loss above the minimum, in the units of the (scaled) loss. */
  excess(): number { return this.o.scale * (this.losses[this.losses.length - 1] - this.surf.min); }
  step(): void {
    if (this.diverged || this.t >= this.o.steps) return;
    const o = this.o, g0 = this.surf.grad(this.a, this.b);
    const g = [o.scale * (g0[0] + o.noise * this.r.norm()), o.scale * (g0[1] + o.noise * this.r.norm())];
    const eta = lrAt(o.schedule, o.eta, this.t, o.steps);
    this.t++;
    const w = [this.a, this.b];
    for (let i = 0; i < 2; i++) {
      if (this.opt === "gd") w[i] -= eta * g[i];
      else if (this.opt === "momentum") { this.v[i] = o.beta * this.v[i] + g[i]; w[i] -= eta * this.v[i]; }
      else {
        this.m[i] = 0.9 * this.m[i] + 0.1 * g[i];
        this.s[i] = 0.999 * this.s[i] + 0.001 * g[i] * g[i];
        const c1 = o.biasCorrection ? 1 - 0.9 ** this.t : 1, c2 = o.biasCorrection ? 1 - 0.999 ** this.t : 1;
        w[i] -= eta * (this.m[i] / c1) / (Math.sqrt(this.s[i] / c2) + 1e-8);
      }
    }
    [this.a, this.b] = w;
    if (!Number.isFinite(this.a) || !Number.isFinite(this.b) || Math.abs(this.a) > 1e6 || Math.abs(this.b) > 1e6) {
      this.diverged = true;
      return;
    }
    this.path.push([this.a, this.b]);
    this.losses.push(this.surf.f(this.a, this.b));
  }
  run(n: number): this { for (let i = 0; i < n; i++) this.step(); return this; }
  /** First step from which the loss stays within tol (in unscaled units) of the minimum, or -1. */
  stepsTo(tol = 1e-3): number {
    if (this.diverged) return -1;
    let k = -1;
    for (let i = this.losses.length - 1; i >= 0; i--) { if (this.losses[i] - this.surf.min < tol) k = i; else break; }
    return k;
  }
}

/** Steps to within tol of the minimum for one optimiser and constant η (−1: not reached or diverged). */
export function stepsToConverge(sid: SurfaceId, opt: OptId, eta: number, steps = 2000, extra: Partial<RaceOptions> = {}): number {
  const w = new Walker(surface(sid), opt, { ...DEFAULT_RACE, eta, steps, ...extra }).run(steps);
  return w.stepsTo();
}
