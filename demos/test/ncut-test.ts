// Tests for demos/src/ncut/spectral.ts against dense and brute-force references.
import { WGraph, fiedler, jacobiEigen, degrees, sweep, stoerWagner } from "../src/ncut/spectral";
import { rng } from "../src/lib/ui";

type Check = (ok: boolean, msg: string) => void;
type Close = (a: number, b: number, tol?: number) => boolean;

function randomGraph(r: ReturnType<typeof rng>, n: number, dens: number): WGraph {
  const G: WGraph = { n, p: [], q: [], w: [] };
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++)
    if (j === i + 1 || r.unif() < dens) { G.p.push(i); G.q.push(j); G.w.push(0.05 + r.unif()); }   // a path keeps it connected
  return G;
}

function cutOf(G: WGraph, A: ArrayLike<number>): number {
  let c = 0;
  for (let k = 0; k < G.p.length; k++) if (A[G.p[k]] !== A[G.q[k]]) c += G.w[k];
  return c;
}

export const tests: [string, (check: Check, close: Close) => void][] = [
  ["fiedler vs dense", (check, close) => {
    const r = rng(2024);
    for (let trial = 0; trial < 120; trial++) {
      const n = 3 + Math.floor(r.unif() * 28);
      const G = randomGraph(r, n, 0.1 + 0.5 * r.unif());
      const d = degrees(G);
      for (const normalized of [false, true]) {
        // dense reference: L or D^-1/2 L D^-1/2
        const M = new Float64Array(n * n);
        for (let i = 0; i < n; i++) M[i * n + i] = d[i];
        for (let k = 0; k < G.p.length; k++) { M[G.p[k] * n + G.q[k]] -= G.w[k]; M[G.q[k] * n + G.p[k]] -= G.w[k]; }
        if (normalized) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i * n + j] /= Math.sqrt(d[i] * d[j]);
        const ref = jacobiEigen(M, n);
        check(Math.abs(ref.values[0]) < 1e-9, `dense smallest eigenvalue is 0 (trial ${trial})`);
        const f = fiedler(G, normalized);
        check(close(f.mu, ref.values[1], 1e-7), `fiedler mu ${f.mu} vs dense ${ref.values[1]} (trial ${trial}, normalized ${normalized})`);
        // eigen-equation L x = mu x or L x = mu D x
        let res = 0, nx = 0;
        for (let i = 0; i < n; i++) {
          let lx = d[i] * f.x[i];
          for (let k = 0; k < G.p.length; k++) {
            if (G.p[k] === i) lx -= G.w[k] * f.x[G.q[k]];
            if (G.q[k] === i) lx -= G.w[k] * f.x[G.p[k]];
          }
          res += (lx - f.mu * (normalized ? d[i] : 1) * f.x[i]) ** 2; nx += f.x[i] ** 2;
        }
        check(Math.sqrt(res / nx) < 1e-6 * Math.max(1, f.mu), `eigen-equation residual (trial ${trial})`);
        // orthogonal to the trivial vector: sum x = 0, or sum d x = 0
        let s = 0; for (let i = 0; i < n; i++) s += (normalized ? d[i] : 1) * f.x[i];
        check(Math.abs(s) < 1e-7 * n * Math.max(...Array.from(d)), `orthogonal to the constant (trial ${trial})`);
      }
    }
  }],
  ["sweep vs brute force", (check, close) => {
    const r = rng(31);
    for (let trial = 0; trial < 150; trial++) {
      const n = 2 + Math.floor(r.unif() * 9);
      const G = randomGraph(r, n, 0.5);
      const x = Array.from({ length: n }, () => r.unif());
      const sw = sweep(G, x);
      const d = degrees(G);
      for (let k = 0; k < n - 1; k++) {
        const A = new Uint8Array(n);
        for (let j = 0; j <= k; j++) A[sw.order[j]] = 1;
        const c = cutOf(G, A);
        let va = 0; for (let i = 0; i < n; i++) if (A[i]) va += d[i];
        const tot = d.reduce((a, b) => a + b, 0);
        check(close(sw.cut[k], c, 1e-9), `sweep cut (trial ${trial})`);
        check(close(sw.ncut[k], c / va + c / (tot - va), 1e-9), `sweep ncut (trial ${trial})`);
      }
    }
  }],
  ["stoer-wagner vs brute force", (check, close) => {
    const r = rng(47);
    for (let trial = 0; trial < 200; trial++) {
      const n = 2 + Math.floor(r.unif() * 10);
      const G = randomGraph(r, n, 0.4);
      let best = Infinity;
      for (let m = 1; m < (1 << n) - 1; m++) best = Math.min(best, cutOf(G, Array.from({ length: n }, (_, i) => m >> i & 1)));
      const sw = stoerWagner(G);
      check(close(sw.value, best, 1e-9), `stoer-wagner ${sw.value} vs ${best} (trial ${trial})`);
      check(close(cutOf(G, sw.side), best, 1e-9), `stoer-wagner side (trial ${trial})`);
      const k = sw.side.reduce((a, b) => a + b, 0);
      check(k > 0 && k < n, `stoer-wagner side is a proper split (trial ${trial})`);
    }
  }],
  ["fiedler on a 32 x 32 image grid", (check) => {
    // the size used by the ncut demo; report the time and check convergence
    const N = 32, r = rng(8);
    const I = Array.from({ length: N * N }, (_, p) => (((p % N) - 16) ** 2 + (Math.floor(p / N) - 16) ** 2 < 80 ? 180 : 70) + 25 * r.norm());
    const G: WGraph = { n: N * N, p: [], q: [], w: [] };
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) for (let dy = 0; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      if ((dy === 0 && dx <= 0) || dx * dx + dy * dy > 4) continue;
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || xx >= N || yy >= N) continue;
      const a = y * N + x, b = yy * N + xx;
      G.p.push(a); G.q.push(b); G.w.push(Math.exp(-((I[a] - I[b]) ** 2) / (2 * 30 * 30)));
    }
    for (const normalized of [true, false]) {
      const t0 = Date.now();
      const f = fiedler(G, normalized);
      console.log(`  (32 x 32 grid, ${normalized ? "normalized" : "unnormalized"}: ${Date.now() - t0} ms, residual ${f.residual.toExponential(1)})`);
      check(f.residual < 1e-6, `32 x 32 fiedler converged (residual ${f.residual})`);
    }
  }],
];
