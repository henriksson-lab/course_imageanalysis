// Tests for lib/docking.ts and the dock2d demo (shape-matching day, unit fft-docking): the
// by-hand 1D example, FFT scores against brute force (dense and over the ligand's cells), the
// complex two-term score, the planted pose of the "key and lock" scene, the behaviour of the
// demo's presets, and the counts quoted on the 3D slides. Run alone:
//   node_modules/.bin/esbuild demos/test/dock2d-test.ts --bundle --platform=node --log-level=warning | node
import {
  layers, receptorGrid, scoreDirect, scoreSparse, score1dFFT, Docker, ligandGrid, scoreAt, complexScore,
  ligandCells, rotatePolyCW, centroid, orientations3d, fftFlops, topPoses, angleDiff, type Ligand,
} from "../src/lib/docking";
import { DEFAULTS, Settings, search, makeScene, toCell, TRUE_POSE, isTrueFit, ligandAt, ligandPoly } from "../src/dock2d/model";
import { PRESETS } from "../src/dock2d/presets";
import { rng } from "../src/lib/random";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log("FAIL: " + msg); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

// ---- by hand: a = (ρ, ρ, ρ, 1, 0, 0, 0, 0), ρ = −9; a two-cell ligand at d, d + 1 ----
{
  const rho = -9;
  const L = layers([1, 1, 1, 0, 0, 0, 0, 0], 8, 1, 1);
  const a = receptorGrid(L, rho);
  [rho, rho, rho, 1, 0, 0, 0, 0].forEach((v, i) => check(a[i] === v, `by-hand receptor a(${i}) = ${a[i]}, expected ${v}`));
  const b = [1, 1, 0, 0, 0, 0, 0, 0];
  const S = scoreDirect(a, b, 8, 1), F = score1dFFT(a, b);
  const expected = [-18, -18, -8, 1, 0, 0, 0, -9];   // S(7): the ligand wraps around onto cell 0
  expected.forEach((v, d) => {
    check(close(S[d], v), `by-hand direct S(${d}) = ${S[d]}, expected ${v}`);
    check(close(F[d], v, 1e-12), `by-hand FFT S(${d}) = ${F[d]}, expected ${v}`);
  });
  // the correlation theorem at k = 0 and k = 4: Σ S = A0·B0 = (−26)(2); Σ (−1)^d S(d) = A4·B4 = (−10)(0)
  check(close(S.reduce((s, v) => s + v, 0), -52), "Σ S(d) = A0 B0 = −52");
  check(close(S.reduce((s, v, d) => s + (d % 2 ? -v : v), 0), 0), "Σ (−1)^d S(d) = A4 B4 = 0");
}

// ---- FFT = brute force (dense and over the ligand cells) on a small random grid ----
{
  const r = rng(5), M = 16;
  const mask = Uint8Array.from({ length: M * M }, (_, i) => {
    const x = i % M, y = Math.floor(i / M);
    return (x - 8) ** 2 + (y - 9) ** 2 < 14 + 6 * r.unif() ? 1 : 0;
  });
  const R = layers(mask, M, M, 2), a = receptorGrid(R, -7);
  const lig: Ligand = { dx: Int32Array.from([0, 1, -1, 0, 2]), dy: Int32Array.from([0, 0, 0, 1, -1]) };
  const dense = scoreDirect(a, ligandGrid(lig, M), M, M), sparse = scoreSparse(a, M, lig);
  const dk = new Docker(a, M), fft = dk.map(lig);
  let worst = 0;
  for (let i = 0; i < M * M; i++) worst = Math.max(worst, Math.abs(dense[i] - fft[i]), Math.abs(dense[i] - sparse[i]));
  check(worst < 1e-9, `16² grid: FFT, dense and sparse scores agree (largest difference ${worst})`);
  check(dk.ffts === 3, `one receptor FFT + two per angle = 3, got ${dk.ffts}`);
  for (const [x, y] of [[3, 4], [8, 1], [15, 15]]) check(close(scoreAt(a, R, lig, x, y).score, fft[y * M + x], 1e-9), `scoreAt (${x}, ${y}) = map`);
  // layers: surface outside the core, no overlap; ρ only on the core
  let overlap = 0;
  for (let i = 0; i < M * M; i++) overlap += R.core[i] & R.surface[i];
  check(overlap === 0, "surface layer and core do not overlap");
}

// ---- the complex score: Re = Σ s_R s_L − w Σ c_R c_L ----
{
  const r = rng(9), M = 8, w = 9;
  const bin = () => Float64Array.from({ length: M * M }, () => (r.unif() < 0.3 ? 1 : 0));
  const sR = bin(), cR = bin(), sL = bin(), cL = bin();
  const C = complexScore(sR, cR, sL, cL, M, w);
  const ss = scoreDirect(sR, sL, M, M), cc = scoreDirect(cR, cL, M, M);
  let worst = 0;
  for (let i = 0; i < M * M; i++) worst = Math.max(worst, Math.abs(C[i] - (ss[i] - w * cc[i])));
  check(worst < 1e-9, `complex score = surface·surface − w core·core (largest difference ${worst})`);
}

// ---- geometry ----
{
  const p = rotatePolyCW([[1, 0]], 90);
  check(close(p[0][0], 0, 1e-12) && close(p[0][1], 1, 1e-12), "90° clockwise on screen (y down) takes (1, 0) to (0, 1)");
  const c = centroid(ligandPoly(false));
  check(Math.hypot(c[0], c[1]) < 1e-9, "the ligand is centred on its centroid");
  const L0 = ligandCells(ligandPoly(false), 0, 1), L90 = ligandCells(ligandPoly(false), 90, 1);
  check(Math.abs(L0.dx.length - L90.dx.length) <= 6, `cell count about the same after a turn (${L0.dx.length}, ${L90.dx.length})`);
  check(angleDiff(350, 10) === -20 && angleDiff(10, 350) === 20, "angleDiff wraps");
  const t = topPoses([{ angle: 0, x: 0, y: 0, score: 5 }, { angle: 5, x: 1, y: 0, score: 4 }, { angle: 90, x: 0, y: 0, score: 3 }], 5, 2, 20);
  check(t.length === 2 && t[1].angle === 90, "NMS drops the same fit at a neighbouring angle, keeps another angle");
}

// ---- the key-and-lock scene ----
const run = (p: Partial<Settings>) => search({ ...DEFAULTS, ...p });
const rankOfTrue = (p: Partial<Settings>) => { const r = run(p); return { r, rank: r.top.findIndex(q => isTrueFit(r.sc, q)) + 1 }; };
{
  const { r, rank } = rankOfTrue({});
  check(rank === 1, `key and lock: the true fit is the best pose (rank ${rank})`);
  check(r.dk.ffts === 1 + 2 * 72, `72 angles: ${r.dk.ffts} FFTs, expected 145`);
  check(r.top[0].score > 1.5 * r.top[1].score, `a clear winner: ${r.top[0].score} vs ${r.top[1].score}`);
  // the hand score at the true pose = the map value there
  const sc = makeScene(DEFAULTS), [gx, gy] = toCell(sc, TRUE_POSE.x, TRUE_POSE.y);
  const L = ligandAt(sc, DEFAULTS, TRUE_POSE.angle), hand = scoreAt(sc.a, sc.R, L, gx, gy);
  const map = new Docker(sc.a, sc.M).map(L);
  check(close(hand.score, map[gy * sc.M + gx]), "hand score = FFT map at the true pose");
  check(hand.clash === 0 && hand.contact > 40, `true pose: no clash, many contacts (${hand.contact})`);
}
// presets behave as their labels say
{
  const P = (id: string) => { const p = PRESETS.find(q => q.id === id); if (!p) throw new Error("no preset " + id); return p.s; };
  const mild = rankOfTrue(P("mild"));
  const best = mild.r.top[0], sa = scoreAt(mild.r.sc.a, mild.r.sc.R, ligandAt(mild.r.sc, { ...DEFAULTS, ...P("mild") }, best.angle), best.x, best.y);
  check(mild.rank !== 1 && sa.clash > 20, `ρ too mild: the best pose sinks in (rank of the fit ${mild.rank}, ${sa.clash} core cells overlapped)`);
  const thin = run(P("thin"));
  check(thin.results.every(q => q.best < 1e-9), "no surface layer: no positive score anywhere");
  check(rankOfTrue(P("coarse")).rank !== 1, "a coarse angle step misses the fit");
  check(rankOfTrue(P("bump")).rank === 0, "flexible bump: the strict score drops the fit from the top 5");
  check(rankOfTrue({ ...P("bump"), rho: -1 }).rank === 1, "flexible bump with a softer core (ρ = −1): the fit is back on top");
  check(rankOfTrue({ ...P("key") }).rank === 1, "preset key and lock finds the fit");
}

// ---- counts on the 3D slides ----
{
  const n15 = orientations3d(15), n6 = orientations3d(6);
  check(n15 > 4300 && n15 < 4500, `8π²/Δθ³ at 15°: ${n15.toFixed(0)} ≈ 4400`);
  check(n6 > 68000 && n6 < 70000, `8π²/Δθ³ at 6°: ${n6.toFixed(0)} ≈ 69 000`);
  const f = fftFlops(128 ** 3);
  check(f > 2e8 && f < 2.4e8, `one 128³ FFT: ${f.toExponential(2)} ≈ 2·10⁸ flops`);
  const arc = 20 * 10 * Math.PI / 180;
  check(arc > 3.4 && arc < 3.6, `10° at 20 Å: ${arc.toFixed(2)} Å`);
}

console.log(`dock2d: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
