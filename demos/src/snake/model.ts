// The snake simulation behind the `snake` demo (no DOM): settings, force fields from the
// image, one step of the explicit or semi-implicit update, energies. Uses lib/snake.ts.
import { Grey } from "../lib/imageops";
import {
  Contour, ForceField, ImplicitSolver, edgeMap, gridGradient, gvfFast, normaliseField, fieldMax, scaleField,
  implicitSolver, implicitStep, explicitStep, sampleField, balloonForce, addForces,
  internalEnergy, externalEnergy, resample, internalForce,
} from "../lib/snake";

export interface Settings {
  alpha: number;
  beta: number;
  tau: number;
  sigma: number;
  /** Weight of the image (edge) term. */
  w: number;
  n: number;
  rule: "explicit" | "implicit";
  force: "edge" | "gvf";
  /** Balloon force (positive inflates). */
  kappa: number;
  /** Resample to equal spacing every k steps (0 = never). */
  resample: number;
  /** GVF smoothness. */
  mu: number;
}

export const DEFAULTS: Settings = {
  alpha: 0.1, beta: 0.05, tau: 1, sigma: 2, w: 2, n: 40, rule: "implicit", force: "edge", kappa: 0, resample: 0, mu: 0.2,
};

export const GVF_COARSE = 250, GVF_FINE = 40;

export interface Fields { sigma: number; f: Grey; edge: ForceField; gvf: ForceField | null; mu: number }

/** Edge map f = |∇(G_σ*I)|² (max 1), its gradient, and (on demand) the GVF field scaled to the same maximum. */
export function makeFields(img: Grey, sigma: number, withGvf: boolean, mu: number, prev?: Fields | null): Fields {
  if (prev && prev.sigma === sigma && prev.mu === mu && (prev.gvf || !withGvf)) return prev;
  const f = edgeMap(img, sigma), edge = gridGradient(f);
  let g: ForceField | null = null;
  if (withGvf) {
    // GVF, normalised: direction from the diffused field, strength that of a typical edge force
    const raw = gvfFast(edge, mu, GVF_COARSE, GVF_FINE), m = fieldMax(edge);
    g = scaleField(normaliseField(raw, 1e-3 * fieldMax(raw)), 0.5 * m);
  }
  return { sigma, f, edge, gvf: g, mu };
}

export interface StepForces { ext: Contour; balloon: Contour | null; internal: Contour }

/** The forces at the current contour. */
export function forces(c: Contour, s: Settings, F: Fields): StepForces {
  const field = s.force === "gvf" && F.gvf ? F.gvf : F.edge;
  return {
    ext: sampleField(field, c, s.w),
    balloon: s.kappa !== 0 ? balloonForce(c, s.kappa) : null,
    internal: internalForce(c, s.alpha, s.beta),
  };
}

export class Sim {
  c: Contour;
  s: Settings;
  F: Fields;
  iter = 0;
  history: { int: number; ext: number }[] = [];
  private solver: ImplicitSolver | null = null;

  constructor(c: Contour, s: Settings, F: Fields) { this.c = c; this.s = s; this.F = F; this.record(); }

  energies(): { int: number; ext: number } {
    return { int: internalEnergy(this.c, this.s.alpha, this.s.beta).total, ext: externalEnergy(this.F.f, this.c, this.s.w) };
  }

  record(): void { this.history.push(this.energies()); if (this.history.length > 2000) this.history.shift(); }

  /** True when the contour has blown up (non-finite or far outside the image). */
  exploded(): boolean {
    for (let i = 0; i < this.c.x.length; i++) {
      const x = this.c.x[i], y = this.c.y[i];
      if (!Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > 1e4 || Math.abs(y) > 1e4) return true;
    }
    return false;
  }

  step(): void {
    if (this.exploded()) return;
    const s = this.s, fr = forces(this.c, s, this.F);
    const f = addForces(fr.ext, fr.balloon);
    if (s.rule === "explicit") this.c = explicitStep(this.c, s.alpha, s.beta, s.tau, f);
    else {
      const n = this.c.x.length, gamma = 1 / s.tau;
      if (!this.solver || this.solver.n !== n || this.solver.alpha !== s.alpha || this.solver.beta !== s.beta || this.solver.gamma !== gamma)
        this.solver = implicitSolver(n, s.alpha, s.beta, gamma);
      this.c = implicitStep(this.c, this.solver, f);
    }
    this.iter++;
    if (s.resample > 0 && this.iter % s.resample === 0 && !this.exploded()) this.c = resample(this.c);
    this.record();
  }

  run(k: number): void { for (let i = 0; i < k; i++) this.step(); }
}
