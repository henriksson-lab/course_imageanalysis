// Model of the mcpixel demo (no DOM): one pixel's integral in 2D, I = ∫ L_i(θ) cos θ dθ over the
// half-circle θ ∈ (−90°, 90°) of incoming directions, estimated by Monte Carlo.
import { TabulatedPdf } from "../lib/montecarlo";

export type ProfileId = "sky" | "window" | "lamp" | "lamps";
export type PdfId = "uniform" | "cosine" | "lamp" | "bad";
const D = Math.PI / 180;

export interface Lamp { a: number; b: number; L: number }   // angles in radians
export interface Profile { id: ProfileId; label: string; base: (th: number) => number; lamps: Lamp[] }

export const PROFILES: Profile[] = [
  { id: "sky", label: "overcast sky", base: th => (1 + 2 * Math.cos(th)) / 3, lamps: [] },
  { id: "window", label: "a window", base: () => 0.05, lamps: [{ a: 15 * D, b: 45 * D, L: 1 }] },
  { id: "lamp", label: "a tiny bright lamp", base: () => 0.05, lamps: [{ a: 39.75 * D, b: 40.25 * D, L: 300 }] },
  { id: "lamps", label: "two lamps", base: () => 0.05, lamps: [{ a: -51 * D, b: -49 * D, L: 20 }, { a: 29 * D, b: 31 * D, L: 20 }] },
];

/** Incoming radiance L_i(θ). */
export function radiance(p: Profile, th: number): number {
  let v = p.base(th);
  for (const l of p.lamps) if (th >= l.a && th <= l.b) v += l.L;
  return v;
}
/** The integrand f(θ) = L_i(θ) cos θ. */
export const integrand = (p: Profile, th: number): number => (Math.abs(th) <= Math.PI / 2 ? radiance(p, th) * Math.cos(th) : 0);

/** The exact integral: the base part by fine quadrature (Simpson), lamps exactly (∫cos = sin b − sin a). */
export function exactIntegral(p: Profile): number {
  const n = 20000, h = Math.PI / n;
  let s = 0;
  for (let i = 0; i <= n; i++) { const th = -Math.PI / 2 + i * h, w = i === 0 || i === n ? 1 : i % 2 ? 4 : 2; s += w * p.base(th) * Math.cos(th); }
  s *= h / 3;
  for (const l of p.lamps) s += l.L * (Math.sin(l.b) - Math.sin(l.a));
  return s;
}

export interface Sampler { sample: (u: number) => number; pdf: (th: number) => number; label: string }

const NB = 3600;   // 0.05° bins for the tabulated pdfs
export function makeSampler(id: PdfId, p: Profile): Sampler {
  if (id === "uniform") return { label: "uniform", sample: u => Math.PI * (u - 0.5), pdf: th => (Math.abs(th) <= Math.PI / 2 ? 1 / Math.PI : 0) };
  if (id === "cosine") return { label: "cosine", sample: u => Math.asin(2 * u - 1), pdf: th => (Math.abs(th) <= Math.PI / 2 ? Math.cos(th) / 2 : 0) };
  const w = new Float64Array(NB), h = Math.PI / NB;
  const lampTot = p.lamps.reduce((s, l) => s + (l.b - l.a), 0);
  for (let i = 0; i < NB; i++) {
    const a = -Math.PI / 2 + i * h, c = a + h / 2;
    const inLamp = p.lamps.some(l => c >= l.a && c <= l.b);
    if (id === "lamp") w[i] = lampTot > 0 ? 0.5 * (Math.sin(a + h) - Math.sin(a)) / 2 + (inLamp ? 0.5 * h / lampTot : 0) : (Math.sin(a + h) - Math.sin(a)) / 2;
    else w[i] = p.lamps.some(l => c >= l.a - 2 * D && c <= l.b + 2 * D) ? 0 : 1;   // a pdf that is zero at the lamps
  }
  if (id === "bad" && p.lamps.length === 0) for (let i = 0; i < NB; i++) { const c = -Math.PI / 2 + (i + 0.5) * h; w[i] = c > 30 * D ? 0 : 1; }
  const t = new TabulatedPdf(-Math.PI / 2, Math.PI / 2, w);
  return { label: id === "lamp" ? "towards the lamp" : "misses the lamp", sample: u => t.sample(u), pdf: th => t.pdf(th) };
}

/** Standard deviation of one sample f/p (exact by quadrature; Infinity if not finite). */
export function singleSigma(p: Profile, s: Sampler, I: number): number {
  const n = 36000, h = Math.PI / n;
  let m2 = 0, m1 = 0;
  for (let i = 0; i < n; i++) {
    const th = -Math.PI / 2 + (i + 0.5) * h, f = integrand(p, th), q = s.pdf(th);
    if (q > 0) { m2 += f * f / q * h; m1 += f * h; }
  }
  void I;
  return Math.sqrt(Math.max(0, m2 - m1 * m1));
}

/** One estimate from N samples; also the samples (angle, weight f/p) if keep. */
export function estimateN(p: Profile, s: Sampler, N: number, u: () => number, keep = false): { est: number; samples: [number, number][] } {
  let sum = 0;
  const samples: [number, number][] = [];
  for (let k = 0; k < N; k++) {
    const th = s.sample(u()), q = s.pdf(th), w = q > 0 ? integrand(p, th) / q : 0;
    sum += w;
    if (keep) samples.push([th, w]);
  }
  return { est: sum / N, samples };
}
