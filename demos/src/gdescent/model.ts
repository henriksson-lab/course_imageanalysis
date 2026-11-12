// Loss landscapes over two parameters (a, b) for the gdescent demo, each the MEAN over N = 40
// training examples, so a mini-batch gradient is a true (noisy, unbiased) estimate of the full
// gradient. No DOM, so the exercise numbers come from this code.
//   bowl:    linear regression ŷ = a·u + b·v (inputs u, v mildly correlated): an elongated bowl
//   valley:  the same with strongly correlated inputs: a long narrow valley
//   neuron:  one neuron p = sigmoid(a·x + b) on 40 points of two overlapping classes (cross-entropy)
//   wells:   a landscape with two minima, (a² − 1)² + 0.6 b² + 0.25 a, plus a zero-mean linear
//            term per example (so mini-batches are noisy here too)
import { rng } from "../lib/random";

export type LandscapeId = "bowl" | "valley" | "neuron" | "wells";
export const NEX = 40;

export interface Landscape {
  id: LandscapeId; label: string;
  /** Loss and gradient of example i at (a, b). */
  ex(i: number, a: number, b: number): [number, number, number];
  range: [number, number, number, number];   // a0, a1, b0, b1 shown
}

function regression(id: LandscapeId, label: string, rho: number, seed: number): Landscape {
  const r = rng(seed), u: number[] = [], v: number[] = [], y: number[] = [];
  for (let i = 0; i < NEX; i++) {
    const g1 = r.norm(), g2 = r.norm(), uu = g1, vv = rho * g1 + Math.sqrt(1 - rho * rho) * g2;
    u.push(uu); v.push(vv); y.push(1.0 * uu + 0.5 * vv + 0.3 * r.norm());
  }
  return {
    id, label, range: [-2, 3, -2.5, 2.5],
    ex: (i, a, b) => { const e = a * u[i] + b * v[i] - y[i]; return [e * e, 2 * e * u[i], 2 * e * v[i]]; },
  };
}

function neuron(): Landscape {
  const r = rng(31), x: number[] = [], c: number[] = [];
  for (let i = 0; i < NEX; i++) { const cls = i % 2; c.push(cls); x.push((cls ? 0.8 : -0.8) + 1.0 * r.norm()); }
  return {
    id: "neuron", label: "one neuron (classifier)", range: [-2, 5, -3.5, 3.5],
    ex: (i, a, b) => {
      const s = a * x[i] + b, p = 1 / (1 + Math.exp(-s));
      const L = Math.max(s, 0) + Math.log1p(Math.exp(-Math.abs(s))) - c[i] * s;
      return [L, (p - c[i]) * x[i], p - c[i]];
    },
  };
}

function wells(): Landscape {
  const r = rng(41), g: [number, number][] = [];
  for (let i = 0; i < NEX; i++) g.push([1.5 * r.norm(), 1.5 * r.norm()]);
  const m = [g.reduce((s, q) => s + q[0], 0) / NEX, g.reduce((s, q) => s + q[1], 0) / NEX];
  g.forEach(q => { q[0] -= m[0]; q[1] -= m[1]; });
  return {
    id: "wells", label: "two minima", range: [-2, 2, -2, 2],
    ex: (i, a, b) => [(a * a - 1) ** 2 + 0.6 * b * b + 0.25 * a + g[i][0] * a + g[i][1] * b, 4 * a * (a * a - 1) + 0.25 + g[i][0], 1.2 * b + g[i][1]],
  };
}

export const LANDSCAPES: Landscape[] = [regression("bowl", "elongated bowl", 0.5, 11), regression("valley", "narrow valley", 0.95, 12), neuron(), wells()];

/** Mean loss and gradient over the examples idx (default all). */
export function lossGrad(L: Landscape, a: number, b: number, idx?: number[]): [number, number, number] {
  let l = 0, ga = 0, gb = 0;
  const ii = idx ?? Array.from({ length: NEX }, (_, i) => i);
  for (const i of ii) { const [x, y, z] = L.ex(i, a, b); l += x; ga += y; gb += z; }
  return [l / ii.length, ga / ii.length, gb / ii.length];
}

export interface PathState { a: number; b: number; va: number; vb: number; pts: [number, number][]; losses: number[]; r: ReturnType<typeof rng> }

export function newPath(L: Landscape, a: number, b: number, seed: number): PathState {
  return { a, b, va: 0, vb: 0, pts: [[a, b]], losses: [lossGrad(L, a, b)[0]], r: rng(seed) };
}

/** One step: gradient on a random batch (batch ≥ NEX: full gradient), optional momentum β. */
export function stepPath(L: Landscape, p: PathState, eta: number, batch: number, beta: number): void {
  let idx: number[] | undefined;
  if (batch < NEX) {
    const perm = Array.from({ length: NEX }, (_, i) => i);
    for (let i = 0; i < batch; i++) { const j = i + Math.floor(p.r.unif() * (NEX - i)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
    idx = perm.slice(0, batch);
  }
  const [, ga, gb] = lossGrad(L, p.a, p.b, idx);
  p.va = beta * p.va + ga; p.vb = beta * p.vb + gb;
  p.a -= eta * p.va; p.b -= eta * p.vb;
  if (!Number.isFinite(p.a) || !Number.isFinite(p.b) || Math.abs(p.a) > 1e6 || Math.abs(p.b) > 1e6) { p.a = NaN; p.b = NaN; }
  p.pts.push([p.a, p.b]);
  p.losses.push(Number.isFinite(p.a) ? lossGrad(L, p.a, p.b)[0] : Infinity);
}

/** The minimum on a fine grid followed by full-batch gradient descent (for marking it). */
export function findMinima(L: Landscape): [number, number, number][] {
  const [a0, a1, b0, b1] = L.range, n = 60, out: [number, number, number][] = [];
  const f = (a: number, b: number) => lossGrad(L, a, b)[0];
  const G: number[] = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) G.push(f(a0 + (a1 - a0) * (i + 0.5) / n, b0 + (b1 - b0) * (j + 0.5) / n));
  for (let j = 1; j < n - 1; j++) for (let i = 1; i < n - 1; i++) {
    const v = G[j * n + i];
    let isMin = true;
    for (let dj = -1; dj <= 1 && isMin; dj++) for (let di = -1; di <= 1; di++) if ((di || dj) && G[(j + dj) * n + i + di] < v) { isMin = false; break; }
    if (!isMin) continue;
    let a = a0 + (a1 - a0) * (i + 0.5) / n, b = b0 + (b1 - b0) * (j + 0.5) / n;
    for (let k = 0; k < 5000; k++) { const [, ga, gb] = lossGrad(L, a, b); a -= 0.02 * ga; b -= 0.02 * gb; }
    if (!out.some(m => Math.hypot(m[0] - a, m[1] - b) < 0.05)) out.push([a, b, f(a, b)]);
  }
  return out;
}
