// Checks for the ICP unit (demos/src/icp/icp.ts). Run:
//   node_modules/.bin/esbuild demos/test/icp-test.ts --bundle --platform=node --log-level=warning | node
import * as I from "../src/icp/icp";
import { svd } from "../src/lib/linalg";
import { procrustes } from "../src/lib/shapes";
import { rng } from "../src/lib/random";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const pts = (xy: number[]): I.Pt[] => Array.from({ length: xy.length / 2 }, (_, i) => ({ x: xy[2 * i], y: xy[2 * i + 1] }));
const onX = (xs: number[]): I.Pt[] => xs.map(x => ({ x, y: 0 }));

// ---- the closed form for known pairs ----
function testClosedForm(): void {
  current = "closed form";
  // The by-hand example (exercise A2): p and its image under θ = 36.87° (cos 0.8, sin 0.6), t = (10, 5).
  const P = pts([2, 0, -2, 0, 0, 1, 0, -1]), Q = pts([11.6, 6.2, 8.4, 3.8, 9.4, 5.8, 10.6, 4.2]);
  const f = I.fitPairsDetail(P, Q, "rigid");
  check(close(f.sDot, 8, 1e-12) && close(f.sCross, 6, 1e-12) && close(f.spp, 10, 1e-12), `S_dot ${f.sDot}, S_cross ${f.sCross}, S_pp ${f.spp}`);
  check(close(f.T.a, 0.8, 1e-12) && close(f.T.b, 0.6, 1e-12) && close(f.T.tx, 10, 1e-12) && close(f.T.ty, 5, 1e-12), `rigid ${JSON.stringify(f.T)}`);
  check(close(I.sse(f.T, P, Q), 0, 1e-9), "exact pairs: zero residual");
  // the same pairs, moved: the centroids carry the translation (exercise A2 with uncentred points)
  check(close(f.qBar.x, 10, 1e-12) && close(f.qBar.y, 5, 1e-12), "q̄ = (10, 5)");

  // random sets: compare with lib/shapes procrustes and with the SVD of the cross-covariance
  const r = rng(3);
  for (let trial = 0; trial < 30; trial++) {
    const n = 3 + (trial % 8);
    const A = Array.from({ length: n }, () => ({ x: 100 * r.unif(), y: 100 * r.unif() }));
    const T = { a: 1.3 * Math.cos(trial), b: 1.3 * Math.sin(trial), tx: 20 * r.norm(), ty: 20 * r.norm() };
    const B = A.map(p => { const q = I.apply(T, p); return { x: q.x + 2 * r.norm(), y: q.y + 2 * r.norm() }; });
    const flat = (S: I.Pt[]) => Float64Array.from(S.flatMap(p => [p.x, p.y]));
    for (const kind of ["rigid", "similarity"] as I.Kind[]) {
      const mine = I.fitPairs(A, B, kind);
      const ref = procrustes(flat(A), flat(B), { scale: kind === "similarity" });
      check(close(mine.a, ref.a, 1e-9) && close(mine.b, ref.b, 1e-9) && close(mine.tx, ref.tx, 1e-7) && close(mine.ty, ref.ty, 1e-7), `${kind} = procrustes (trial ${trial})`);
    }
    // rigid: R = U diag(1, det UVᵀ) Vᵀ from C = Σ q pᵀ (centred) — the math slide
    const pb = I.centroid(A), qb = I.centroid(B);
    const C = new Float64Array(4);
    for (let i = 0; i < n; i++) {
      const p = [A[i].x - pb.x, A[i].y - pb.y], q = [B[i].x - qb.x, B[i].y - qb.y];
      C[0] += q[0] * p[0]; C[1] += q[0] * p[1]; C[2] += q[1] * p[0]; C[3] += q[1] * p[1];
    }
    const { U, V } = svd(C, 2, 2);
    const u = (i: number, j: number) => U[i * 2 + j], v = (i: number, j: number) => V[i * 2 + j];
    const detUV = (u(0, 0) * u(1, 1) - u(0, 1) * u(1, 0)) * (v(0, 0) * v(1, 1) - v(0, 1) * v(1, 0));
    const d = [1, Math.sign(detUV)];
    const R = (i: number, j: number) => u(i, 0) * d[0] * v(j, 0) + u(i, 1) * d[1] * v(j, 1);
    const mine = I.fitPairs(A, B, "rigid");
    check(close(R(0, 0), mine.a, 1e-9) && close(R(1, 0), mine.b, 1e-9) && close(R(0, 1), -mine.b, 1e-9), `SVD rotation = atan2 rotation (trial ${trial})`);
    // optimality: small perturbations of the angle or the translation do not lower the error
    const e0 = I.sse(mine, A, B);
    for (const dt of [-1e-3, 1e-3]) {
      const th = Math.atan2(mine.b, mine.a) + dt;
      const Tp = { a: Math.cos(th), b: Math.sin(th), tx: mine.tx, ty: mine.ty };
      // refit the translation for the perturbed angle: centroids
      const c = I.apply({ ...Tp, tx: 0, ty: 0 }, pb);
      Tp.tx = qb.x - c.x; Tp.ty = qb.y - c.y;
      check(I.sse(Tp, A, B) >= e0 - 1e-9, "perturbed angle is not better");
      check(I.sse({ ...mine, tx: mine.tx + 10 * dt }, A, B) >= e0 - 1e-9, "perturbed translation is not better");
    }
    // translation only: the difference of the centroids
    const t = I.fitPairs(A, B, "translation");
    check(t.a === 1 && t.b === 0 && close(t.tx, qb.x - pb.x, 1e-12) && close(t.ty, qb.y - pb.y, 1e-12), "translation = q̄ − p̄");
  }
  // compose / invert
  const T1 = { a: 0.9, b: 0.4, tx: 3, ty: -2 }, T2 = { a: 1.1, b: -0.2, tx: -5, ty: 7 }, p = { x: 4, y: 9 };
  const c = I.apply(I.compose(T2, T1), p), cc = I.apply(T2, I.apply(T1, p));
  check(close(c.x, cc.x, 1e-12) && close(c.y, cc.y, 1e-12), "compose");
  const back = I.apply(I.invert(T1), I.apply(T1, p));
  check(close(back.x, p.x, 1e-12) && close(back.y, p.y, 1e-12), "invert");
}

// ---- the by-hand ICP runs (exercise A3, A4) ----
function testByHand(): void {
  current = "by hand";
  // A3: P = {0, 4, 8}, Q = {3, 7, 11} on a line. From t = 0: stuck at t = 1/3. From t = 2: t = 3.
  const P = onX([0, 4, 8]), Q = onX([3, 7, 11]);
  for (const kind of ["translation", "rigid"] as I.Kind[]) {
    const r0 = I.icp(P, Q, { kind, keep: 1 });
    check(r0.converged && close(r0.T.tx, 1 / 3, 1e-12) && close(r0.T.ty, 0, 1e-12) && close(r0.T.b, 0, 1e-12), `${kind}: from 0 to 1/3, got ${JSON.stringify(r0.T)}`);
    check(Array.from(r0.steps[0].match).join() === "0,0,1", "first matching 0→3, 4→3, 8→7");
    check(close(r0.steps[0].rms, Math.sqrt((9 + 1 + 1) / 3), 1e-12), "first RMS √(11/3)");
    const r2 = I.icp(P, Q, { kind, keep: 1, T0: I.translation(2, 0) });
    check(close(r2.T.tx, 3, 1e-12) && close(r2.steps[r2.steps.length - 1].rmsAfter, 0, 1e-9), `${kind}: from 2 to 3, got ${r2.T.tx}`);
  }
  // A4: P = {0, 2, 4, 30}, Q = {0.5, 2.5, 4.5}. Plain ICP: t = −6, then −7.5 (stays). Trimmed (keep 3 of 4): t = 0.5.
  const P4 = onX([0, 2, 4, 30]), Q4 = onX([0.5, 2.5, 4.5]);
  const plain = I.icp(P4, Q4, { kind: "translation", keep: 1 });
  check(close(plain.steps[0].next.tx, -6, 1e-12) && close(plain.T.tx, -7.5, 1e-12), `plain: −6 then −7.5, got ${plain.steps.map(s => s.next.tx)}`);
  const trimmed = I.icp(P4, Q4, { kind: "translation", keep: 0.75 });
  check(close(trimmed.steps[0].next.tx, 0.5, 1e-12) && close(trimmed.T.tx, 0.5, 1e-12), `trimmed: 0.5, got ${trimmed.T.tx}`);
  check(I.keepCount(4, 0.75) === 3 && I.keepCount(10, 0.05) === 2 && I.keepCount(3, 1) === 3, "keepCount");
}

// ---- ICP never increases the error (untrimmed), and the scenes behave ----
function testMonotone(): void {
  current = "monotone";
  for (const scene of ["spots", "outline"] as I.SceneKind[]) {
    for (const angle of [10, 45, 120]) {
      const s = I.makeScene({ scene, n: 40, angle, shift: [25, -15], noise: 1, unpaired: 0.1, seed: 1 });
      for (const kind of ["translation", "rigid", "similarity"] as I.Kind[]) {
        const r = I.icp(s.M, s.F, { kind, keep: 1 });
        for (let k = 0; k < r.steps.length; k++) {
          const st = r.steps[k];
          check(st.rmsAfter <= st.rms + 1e-9, `${scene} ${angle}° ${kind} step ${k}: the fit does not increase the error`);
          if (k > 0) check(st.rms <= r.steps[k - 1].rmsAfter + 1e-9, `${scene} ${angle}° ${kind} step ${k}: rematching does not increase the error`);
        }
        check(r.converged, `${scene} ${angle}° ${kind}: converges (${r.steps.length} steps)`);
      }
    }
  }
  // a scene: the moving set maps onto its partners under the truth (no noise)
  const s = I.makeScene({ scene: "spots", n: 30, angle: 30, shift: [10, 5], noise: 0, unpaired: 0.2, seed: 2 });
  check(s.F.length === 30 && s.M.length === 30, `30 fixed, 24 + 6 moving (${s.F.length}, ${s.M.length})`);
  let maxd = 0;
  s.partner.forEach((j, i) => { if (j >= 0) { const q = I.apply(s.truth, s.M[i]); maxd = Math.max(maxd, Math.hypot(q.x - s.F[j].x, q.y - s.F[j].y)); } });
  check(maxd < 1e-9, `truth maps partners exactly (${maxd})`);
  check(Array.from(s.partner).filter(j => j === -1).length === 6, "6 extras");
  // the same seed with another angle moves the same points
  const s2 = I.makeScene({ scene: "spots", n: 30, angle: -50, shift: [10, 5], noise: 0, unpaired: 0.2, seed: 2 });
  check(s2.F.every((p, i) => p.x === s.F[i].x && p.y === s.F[i].y) && Array.from(s2.partner).join() === Array.from(s.partner).join(), "angle does not change the points");
}

// ---- the demo's claims ----
function testDemoClaims(): void {
  current = "demo";
  // The numbers quoted on icp/exercise.qmd and icp/lec.qmd (demo presets).
  const BASE: I.DemoSettings = { scene: "spots", kind: "rigid", angle: 10, noise: 1, unpaired: 0, keep: 1, centroids: false, startShift: [0, 0] };
  const run = (s: Partial<I.DemoSettings>) => {
    const r = I.demoRun({ ...BASE, ...s });
    const K = r.res.steps.length;
    return { moves: K - 1, rms0: r.res.steps[0].rms, rms: r.res.steps[K - 1].rmsAfter, err: r.err[K - 1], angle: I.angleOf(r.res.T) * 180 / Math.PI, conv: r.res.converged };
  };
  const near = (a: number, b: number, tol = 0.006) => Math.abs(a - b) <= tol;
  const close10 = run({});
  check(close10.moves === 3 && near(close10.rms0, 18.87) && near(close10.rms, 1.53) && near(close10.err, 0.16), `close: ${JSON.stringify(close10)}`);
  const SW = Array.from({ length: 73 }, (_, i) => -180 + 5 * i);
  const okAngles = (s: Partial<I.DemoSettings>) => { const e = I.sweep({ ...BASE, ...s }, SW); return SW.filter((_, i) => e[i] < 5).join(","); };
  check(okAngles({}) === "-10,-5,0,5,10,15", `sweep spots: ${okAngles({})}`);
  const a15 = run({ angle: 15 }), a20 = run({ angle: 20 });
  check(a15.moves === 5 && a15.err < 1 && Math.abs(a20.err - 37.57) < 0.01, `15° and 20°: ${JSON.stringify(a15)} ${JSON.stringify(a20)}`);
  const far = run({ angle: 40 });
  check(near(far.err, 70.93) && near(far.angle, 1.8, 0.06) && near(far.rms, 25.14), `40°: local minimum ${JSON.stringify(far)}`);
  const cOn = run({ startShift: [80, 0], centroids: true }), cOff = run({ startShift: [80, 0] });
  check(cOn.moves === 2 && near(cOn.err, 0.16) && cOff.moves === 5 && near(cOff.err, 33.03), `centroid start ${JSON.stringify(cOn)} / ${JSON.stringify(cOff)}`);
  const trims: [number, number][] = [[1, 21.86], [0.9, 10.72], [0.8, 3.82], [0.7, 0.18], [0.6, 0.49], [0.5, 0.60], [0.4, 17.08]];
  for (const [keep, e] of trims) { const r = run({ unpaired: 0.3, keep }); check(near(r.err, e), `overlap 30 %, keep ${keep}: ${r.err} (expected ${e})`); }
  check(run({ unpaired: 0.3, keep: 0.7 }).moves === 6, "trimmed: 6 moves");
  check(near(run({ unpaired: 0.3, centroids: true }).err, 87.93), "overlap + centroid start fails");
  const tr = run({ kind: "translation" });
  check(tr.moves === 1 && near(tr.rms, 18.24) && near(tr.err, 20.41), `translation only: ${JSON.stringify(tr)}`);
  check(near(run({ kind: "translation", angle: 0 }).err, 0.15), "translation at 0°");
  const ol = run({ scene: "outline", angle: 30 });
  check(ol.moves === 19 && near(ol.err, 3.48) && near(ol.angle, 28.2, 0.06), `outline: ${JSON.stringify(ol)}`);
  check(okAngles({ scene: "outline", angle: 30 }) === "-85,-80,-75,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,70", "sweep outline");
  const n3 = run({ noise: 3 }), n5 = run({ noise: 5 });
  check(near(n3.err, 0.48) && near(n3.rms, 4.58) && near(n5.err, 0.79) && near(n5.rms, 7.63), `noise: ${JSON.stringify(n3)} ${JSON.stringify(n5)}`);
}

testClosedForm();
testByHand();
testMonotone();
testDemoClaims();
console.log(`icp: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
