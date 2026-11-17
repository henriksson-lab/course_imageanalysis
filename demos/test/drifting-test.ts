// Tests for the drifting demo (diffusion-models day, unit drifting/): the field against a direct
// evaluation of the formulas, the by-hand numbers of the exercise (B1, B2, B4), anti-symmetry,
// the drifting step, the stop-gradient loss of the generator, and the preset outcomes quoted on
// the pages. Run:
//   node_modules/.bin/esbuild demos/test/drifting-test.ts --bundle --platform=node --log-level=warning --outfile=<tmp>/t.js && node <tmp>/t.js
import * as M from "../src/drifting/model";
import { rng } from "../src/lib/ui";
import { forward } from "../src/lib/nn";

declare const process: { exit(code: number): never };
let fails = 0, passes = 0;
function check(name: string, ok: boolean, info = ""): void {
  if (ok) passes++; else { fails++; console.log("FAIL", name, info); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const onLine = (v: number[]) => Float64Array.from(v.flatMap(x => [x, 0]));

// ---- the field against the formulas V⁺ = Σ a (y − x), V⁻ = Σ b (x_k − x) ------------------------
{
  const r = rng(4), real = Float64Array.from({ length: 40 }, () => r.norm()), gen = Float64Array.from({ length: 24 }, () => 1.5 * r.norm());
  for (const tau of [0.1, 0.5, 2]) {
    const { v, plus, minus } = M.field(gen, real, tau);
    let ok = true;
    for (let i = 0; i < 12; i++) {
      const x = gen[2 * i], y = gen[2 * i + 1];
      let wp = 0, px = 0, py = 0, wm = 0, mx = 0, my = 0;
      for (let j = 0; j < 20; j++) { const w = Math.exp(-Math.hypot(real[2 * j] - x, real[2 * j + 1] - y) / tau); wp += w; px += w * (real[2 * j] - x); py += w * (real[2 * j + 1] - y); }
      for (let k = 0; k < 12; k++) { if (k === i) continue; const w = Math.exp(-Math.hypot(gen[2 * k] - x, gen[2 * k + 1] - y) / tau); wm += w; mx += w * (gen[2 * k] - x); my += w * (gen[2 * k + 1] - y); }
      if (!close(plus[2 * i], px / wp, 1e-9) || !close(plus[2 * i + 1], py / wp, 1e-9) || !close(minus[2 * i], mx / wm, 1e-9) || !close(minus[2 * i + 1], my / wm, 1e-9)) ok = false;
      if (!close(v[2 * i], px / wp - mx / wm, 1e-9) || !close(v[2 * i + 1], py / wp - my / wm, 1e-9)) ok = false;
    }
    check(`field = V⁺ − V⁻ by the formulas, τ = ${tau}`, ok);
  }
  const { v } = M.field(gen, real, 0.002);
  check("small τ: no underflow", Array.from(v).every(Number.isFinite));
}

// ---- by hand: B1 (V⁺ and τ), B2 (two points are pushed apart), B4 (coverage) ------------------
{
  const real = onLine([0, 2]);
  check("B1 τ = 1", close(M.vPlus(0.5, 0, real, 1)[0], 0.0379, 1e-2), String(M.vPlus(0.5, 0, real, 1)[0]));
  check("B1 τ = 0.25", close(M.vPlus(0.5, 0, real, 0.25)[0], -0.464, 1e-3), String(M.vPlus(0.5, 0, real, 0.25)[0]));
  check("B1 τ large → mean", close(M.vPlus(0.5, 0, real, 1e4)[0], 0.5, 1e-3));
  const a = M.field(onLine([0.5, 1.5]), real, 1).v;
  check("B2a", close(a[0], -0.962, 1e-3) && close(a[2], 0.962, 1e-3), `${a[0]} ${a[2]}`);
  const c2 = M.field(onLine([-1, 3]), real, 1).v;
  check("B2c", close(c2[0], -2.762, 1e-3), String(c2[0]));
  const d2 = M.field(onLine([0, 2]), real, 1).v;
  check("B2d: an exact copy of two points is not at rest", close(d2[0], -1.762, 1e-3), String(d2[0]));
  const d = new M.Drift(real, onLine([0.5, 1.5]), { tau: 1, eta: 0.5, views: 1, jitter: 0 });
  d.advance();
  check("B2b: one step with η = 0.5", close(d.x[0], 0.019, 2e-2) && close(d.x[2], 1.981, 1e-3), `${d.x[0]} ${d.x[2]}`);
  const rr = onLine([0, 1, 2, 3, 10]);
  check("B4a coverage", close(M.coverage(rr, onLine([2.5, 6]), 1), 0.6));
  check("B4b coverage", close(M.coverage(rr, onLine([2.5]), 1), 0.4));
}

// ---- anti-symmetry at a probe point: V_{p,q}(x) = −V_{q,p}(x); zero when p = q -----------------
{
  const r = rng(8), P = Float64Array.from({ length: 30 }, () => r.norm()), Q = Float64Array.from({ length: 30 }, () => r.norm() + 0.5);
  let ok = true, zero = true;
  for (let t = 0; t < 20; t++) {
    const x = 2 * r.norm(), y = 2 * r.norm();
    const vpq = [M.vPlus(x, y, P, 0.4)[0] - M.vPlus(x, y, Q, 0.4)[0], M.vPlus(x, y, P, 0.4)[1] - M.vPlus(x, y, Q, 0.4)[1]];
    const vqp = [M.vPlus(x, y, Q, 0.4)[0] - M.vPlus(x, y, P, 0.4)[0], M.vPlus(x, y, Q, 0.4)[1] - M.vPlus(x, y, P, 0.4)[1]];
    if (!close(vpq[0], -vqp[0]) || !close(vpq[1], -vqp[1])) ok = false;
    const vpp = M.vPlus(x, y, P, 0.4)[0] - M.vPlus(x, y, P, 0.4)[0];
    if (vpp !== 0) zero = false;
  }
  check("anti-symmetry at probe points", ok);
  check("V_{p,p} = 0 at probe points", zero);
  // the caveat: one point at the centre of a symmetric set feels no force, though q ≠ p
  const ring = Float64Array.from({ length: 16 }, (_, i) => i % 2 === 0 ? Math.cos(Math.PI * (i / 2) / 4) : Math.sin(Math.PI * ((i - 1) / 2) / 4));
  const v0 = M.field(onLine([0]), ring, 0.5).v;
  check("V = 0 at the centre of a ring (q ≠ p)", Math.hypot(v0[0], v0[1]) < 1e-12, `${v0[0]} ${v0[1]}`);
}

// ---- augmentation: views per point, the point moves by the mean field of its views -------------
{
  const real = M.makeData("moons"), x0 = M.initPoints("noise", 6);
  const d = new M.Drift(real, x0.slice(), { tau: 0.2, eta: 0.5, views: 4, jitter: 0.1 });
  const v = d.lastV.slice();
  d.advance();
  let ok = true;
  for (let i = 0; i < 12; i++) if (!close(d.x[i], x0[i] + 0.5 * v[i], 1e-12)) ok = false;
  check("drift step = x + η · (mean field of the views)", ok);
  check("views: A per point", d.views().length === 6 * 4 * 2);
}

// ---- generator: the stop-gradient loss is a regression on f + V; its value is mean |V|²/2 -------
{
  const real = M.makeData("moons"), g = new M.Generator(real, { tau: 0.2, batch: 16, lr: 0.003, hidden: 8 }, 3);
  // replay the generator's noise stream to recompute the field of the first batch
  const r = rng(3 + 100);
  for (let i = 0; i < 400; i++) { r.norm(); r.norm(); }
  const eps = Array.from({ length: 16 }, () => [r.norm(), r.norm()]);
  const gen = new Float64Array(32);
  eps.forEach((e, i) => { const f = forward(g.net, e).a; gen[2 * i] = f[f.length - 1][0]; gen[2 * i + 1] = f[f.length - 1][1]; });
  const pos = new Float64Array(32);
  for (let i = 0; i < 16; i++) { const j = Math.floor(r.unif() * 400); pos[2 * i] = real[2 * j]; pos[2 * i + 1] = real[2 * j + 1]; }
  const V = M.field(gen, pos, 0.2).v;
  let want = 0; for (let i = 0; i < 32; i++) want += V[i] * V[i] / 32;
  const loss = g.train();
  check("generator loss = mean |V|² / 2", close(loss, want, 1e-9), `${loss} ${want}`);
}

// ---- the presets quoted on the pages -----------------------------------------------------------
{
  const run = (id: string) => { const d = M.newDrift(M.presetConfig(id)); for (let s = 0; s < M.POINT_STEPS; s++) d.advance(); return d; };
  const drift = run("drift"), attract = run("attract"), few = run("few"), views = run("views"), match = M.newDrift(M.presetConfig("match"));
  check("drift: |V| 0.451 → 0.001, coverage 0.357 → 0.993", close(drift.normHist[0], 0.451, 2e-3) && drift.normHist[300] < 0.0015 && close(drift.covHist[0], 0.357, 2e-3) && close(drift.covHist[300], 0.993, 2e-3), `${drift.normHist[0]} ${drift.normHist[300]} ${drift.covHist[0]} ${drift.covHist[300]}`);
  check("match: |V| starts at 0.034", close(match.normHist[0], 0.034, 2e-2), String(match.normHist[0]));
  check("attraction only collapses: coverage 0.048", close(attract.covHist[300], 0.048, 2e-2), String(attract.covHist[300]));
  check("20 points: 0.092 without views, 0.313 with 8 views", close(few.covHist[300], 0.092, 2e-2) && close(views.covHist[300], 0.313, 2e-2), `${few.covHist[300]} ${views.covHist[300]}`);
}

console.log(`drifting: ${passes} passed, ${fails} failed`);
if (fails) process.exit(1);
