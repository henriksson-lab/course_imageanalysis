// Image data for tools/figs/inpainting/figs.py (run by it; prints JSON). Uses lib/inpaint.ts.
import { harmonicJacobi, harmonicSolve, TvInpaint, exemplarInpaint, initHole } from "../../../demos/src/lib/inpaint";
import { testCard } from "../../../demos/src/lib/pde";
import { rng } from "../../../demos/src/lib/random";
const r = (a: ArrayLike<number>) => Array.from(a, v => Math.round(v * 10) / 10);
const out: Record<string, unknown> = {};
const N = 128, card = testCard().truth;
const rect = (m: Uint8Array, w: number, x0: number, y0: number, x1: number, y1: number) => { for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) m[y * w + x] = 1; };
// Jacobi on a 16 × 16 hole (as the demo's "medium" hole)
{ const m = new Uint8Array(N * N); rect(m, N, 30, 14, 46, 30);
  let u: Float64Array = initHole(card, m, 0); const snaps: Record<string, number[]> = {};
  for (let k = 1; k <= 1000; k++) { u = harmonicJacobi(u, m, N, N); if (k === 10 || k === 100 || k === 1000) snaps["k" + k] = r(u); }
  const S = harmonicSolve(card, m, N, N);
  const err = (a: ArrayLike<number>) => { let e = 0, c = 0; for (let i = 0; i < N * N; i++) if (m[i]) { e = Math.max(e, Math.abs(a[i] - S.u[i])); c++; } return e; };
  out.jacobi = { snaps, solve: r(S.u), iters: S.iters, maxdiff: [10, 100, 1000].map(k => err(snaps["k" + k])) }; }
// harmonic vs TV on the large hole (corner of the square, two discs)
{ const m = new Uint8Array(N * N); rect(m, N, 24, 18, 56, 50);
  const H = harmonicSolve(card, m, N, N).u, T = new TvInpaint(card, m, N, N, H); T.run(3000);
  const e = (a: ArrayLike<number>) => { let s = 0, c = 0; for (let i = 0; i < N * N; i++) if (m[i]) { s += Math.abs(a[i] - card[i]); c++; } return s / c; };
  out.htv = { mask: Array.from(m), harm: r(H), tv: r(T.u), truth: r(card), err: [e(H), e(T.u)] }; }
// connectivity: bar width 10, gaps 6 and 16
{ const res: Record<string, unknown> = {};
  for (const gap of [6, 16]) { const w = 128, h = 64, t = new Float64Array(w * h).fill(200), m = new Uint8Array(w * h);
    for (let y = 27; y < 37; y++) for (let x = 0; x < w; x++) t[y * w + x] = 50;
    const x0 = 64 - Math.floor(gap / 2); rect(m, w, x0, 12, x0 + gap, 52);
    const T = new TvInpaint(t, m, w, h, harmonicSolve(t, m, w, h, 1e-6, 3000).u); T.run(3000);
    res["g" + gap] = { u: r(T.u), mask: Array.from(m), centre: T.u[32 * w + 64] }; }
  out.conn = res; }
// exemplar on a texture
function texture(): Float64Array { const t = new Float64Array(N * N), q = rng(3); for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const a = Math.hypot(((x % 8) + 8) % 8 - 3.5, ((y % 8) + 8) % 8 - 3.5); t[y * N + x] = 70 + (a < 2.5 ? 110 : 0) + 25 * Math.sin(2 * Math.PI * (x + y) / 16) + 4 * q.norm(); } return t; }
{ const t = texture(), m = new Uint8Array(N * N); rect(m, N, 48, 48, 76, 76);
  const E = exemplarInpaint(t, m, N, N, { patch: 4, search: 32 }), H = harmonicSolve(t, m, N, N).u;
  const e = (a: ArrayLike<number>) => { let s = 0, c = 0; for (let i = 0; i < N * N; i++) if (m[i]) { s += Math.abs(a[i] - t[i]); c++; } return s / c; };
  out.tex = { truth: r(t), mask: Array.from(m), ex: r(E.u), order: Array.from(E.order), steps: E.steps.length, harm: r(H), err: [e(E.u), e(H)] }; }
// exemplar invents a nucleus
{ const t = new Float64Array(N * N), q = rng(5); const nuc: [number, number, number][] = [[30, 30, 12], [80, 26, 10], [104, 70, 12], [36, 92, 11], [76, 104, 9]];
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { let v = 60 + 6 * q.norm(); for (const [cx, cy, rr] of nuc) { const d = Math.hypot(x - cx, y - cy); if (d < rr) v = 170 + 25 * Math.cos(d / 2) + 6 * q.norm(); else if (d < rr + 1.5) v = 120; } t[y * N + x] = v; }
  const m = new Uint8Array(N * N); rect(m, N, 52, 48, 80, 76);
  const E = exemplarInpaint(t, m, N, N, { patch: 4, search: 32 }), H = harmonicSolve(t, m, N, N).u;
  let bright = 0; for (let i = 0; i < N * N; i++) if (m[i] && E.u[i] > 120) bright++;
  out.nuc = { truth: r(t), mask: Array.from(m), ex: r(E.u), harm: r(H), bright, holeMax: Math.max(...Array.from(t).filter((_, i) => m[i])) }; }
console.log(JSON.stringify(out));
