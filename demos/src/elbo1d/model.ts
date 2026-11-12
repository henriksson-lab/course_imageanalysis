// The smallest VAE that can be solved exactly (no DOM: the exercise numbers come from here).
// Prior z ~ N(0, 1). Decoder: x | z ~ N(w z, sx²) (one grey value x, decoded as w z with
// Gaussian pixel noise sx). Encoder cloud for one observed x: q(z) = N(mu, s²).
// Loss = reconstruction + beta · KL, with (dropping the constant ½ log 2π sx²)
//   reconstruction R = E_q[(x − w z)²] / (2 sx²) = ((x − w mu)² + w² s²) / (2 sx²)
//   KL(q ‖ prior)     = ½ (mu² + s² − 1 − ln s²).
// Best cloud: mu* = w x / (w² + beta sx²),  s*² = beta sx² / (beta sx² + w²)
// (for beta = 1 this is the exact posterior of z given x).
import { rng } from "../lib/ui";

export interface Params { x: number; w: number; sx: number; beta: number }
export const DEFAULT: Params = { x: 1.5, w: 1, sx: 0.5, beta: 1 };
export const N_DRAWS = 20;
/** Fixed standard-normal draws ε_k (seed 4): the reparameterised samples are z_k = mu + s ε_k. */
export const EPS: number[] = (() => { const r = rng(4); return Array.from({ length: N_DRAWS }, () => r.norm()); })();

export const recon = (p: Params, mu: number, s: number) => ((p.x - p.w * mu) ** 2 + p.w ** 2 * s * s) / (2 * p.sx ** 2);
export const kl = (mu: number, s: number) => 0.5 * (mu * mu + s * s - 1 - Math.log(s * s));
export const loss = (p: Params, mu: number, s: number) => recon(p, mu, s) + p.beta * kl(mu, s);
/** The same reconstruction term estimated from the N_DRAWS reparameterised samples. */
export const reconMC = (p: Params, mu: number, s: number) =>
  EPS.reduce((t, e) => t + (p.x - p.w * (mu + s * e)) ** 2, 0) / EPS.length / (2 * p.sx ** 2);
export function best(p: Params): { mu: number; s: number } {
  const b = Math.max(p.beta, 1e-6);
  return { mu: p.w * p.x / (p.w ** 2 + b * p.sx ** 2), s: Math.sqrt(b * p.sx ** 2 / (b * p.sx ** 2 + p.w ** 2)) };
}
