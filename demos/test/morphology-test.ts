// Reference checks for the morphology day's demo code (morphgrid, granulo, morph1d, tophat,
// recon1d, reconstruct) and the lib/morph.ts + lib/distance.ts properties the units teach.
// Brute force or a second, independent route for every routine, on small random inputs. Run:
//   node_modules/.bin/esbuild demos/test/morphology-test.ts --bundle --platform=node --log-level=warning | node
import * as MO from "../src/lib/morph";
import * as DT from "../src/lib/distance";
import { rng } from "../src/lib/random";
import { Off, erodeGrid, dilateGrid, complement, reflect, chamferGrid, INF, openGrid, closeGrid } from "../src/morphgrid/ops";
import { applyStages, dualOf } from "../src/morphgrid/mode-binary";
import { openingFunction, granulometry, SEKind } from "../src/granulo/size";
import { erode1, dilate1, open1, close1, seHeights, median1 } from "../src/morph1d/m1d";
import { erodeDisc, dilateDisc, rowExtremum } from "../src/tophat/fastdisc";
import { iterates, hmax, hmin, regionalMax, imposeParts, reconstruct, TWELVE } from "../src/recon1d/r1d";
import { geodesicSteps, reconstructBinary, fillHolesRec, clearBorderRec } from "../src/reconstruct/binrec";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
const section: Record<string, number> = {};
function check(ok: boolean, msg: string): void {
  checks++;
  section[current] = (section[current] ?? 0) + 1;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const eq = (a: ArrayLike<number>, b: ArrayLike<number>) => a.length === b.length && Array.from(a).every((v, i) => (v ? 1 : 0) === (b[i] ? 1 : 0));
const eqv = (a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-6) => a.length === b.length && Array.from(a).every((v, i) => Math.abs(v - b[i]) <= tol || (v === b[i]));
const subset = (a: ArrayLike<number>, b: ArrayLike<number>) => Array.from(a).every((v, i) => !v || b[i]);
const R = rng(2026);
const randMask = (w: number, h: number, p = 0.5) => Uint8Array.from({ length: w * h }, () => (R.unif() < p ? 1 : 0));
function randSE(): Off[] {
  const s: Off[] = [];
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) if (R.unif() < 0.3) s.push([dx, dy]);
  if (!s.length) s.push([0, 0]);
  return s;
}
const libSE = (s: Off[]): MO.SE => ({ dx: Int32Array.from(s, o => o[0]), dy: Int32Array.from(s, o => o[1]) });

// ---------------------------------------------------------------------------------------
current = "morphgrid ops vs lib/morph";
for (let t = 0; t < 60; t++) {
  const w = 5 + Math.floor(R.unif() * 8), h = 5 + Math.floor(R.unif() * 8), A = randMask(w, h, 0.6), B = randSE();
  check(eq(erodeGrid(A, w, h, B, 0), MO.erodeBinary(A, w, h, libSE(B), true)), "erosion, outside 0");
  check(eq(erodeGrid(A, w, h, B, 1), MO.erodeBinary(A, w, h, libSE(B), false)), "erosion, outside 1 = neutral");
  check(eq(dilateGrid(A, w, h, B, 0), MO.dilateBinary(A, w, h, libSE(B))), "dilation");
  check(eq(openGrid(A, w, h, B, -1), MO.openBinary(A, w, h, libSE(B))), "opening, border ignored = lib");
  check(eq(closeGrid(A, w, h, B, -1), MO.closeBinary(A, w, h, libSE(B))), "closing, border ignored = lib");
}

current = "duality (A ⊖ B)ᶜ = Aᶜ ⊕ B̌";
for (let t = 0; t < 60; t++) {
  const w = 6 + Math.floor(R.unif() * 6), h = 6 + Math.floor(R.unif() * 6), A = randMask(w, h), B = randSE();
  for (const o of [0, 1, -1]) {
    const od = o < 0 ? -1 : 1 - o;
    check(eq(complement(erodeGrid(A, w, h, B, o)), dilateGrid(complement(A), w, h, reflect(B), od)), `erosion, outside ${o}`);
    check(eq(complement(dilateGrid(A, w, h, B, o)), erodeGrid(complement(A), w, h, reflect(B), od)), `dilation, outside ${o}`);
    check(eq(complement(openGrid(A, w, h, B, o)), closeGrid(complement(A), w, h, reflect(B), od)), `opening, outside ${o}`);
    // the demo's own dual expression equals the operation
    const stages = [{ kind: "erode" as const }, { kind: "dilate" as const }];
    check(eq(dualOf(stages, A, w, h, B, o), applyStages(stages, A, w, h, B, o)), "dualOf(opening)");
  }
}

current = "adjunction δ(A) ⊆ C ⇔ A ⊆ ε(C)";
for (let t = 0; t < 300; t++) {
  const w = 5, h = 5, A = randMask(w, h, 0.25), C = randMask(w, h, 0.7), B = randSE();
  // dilation clips at the border, so its adjoint erosion treats the outside as neutral (1)
  const lhs = subset(dilateGrid(A, w, h, B, 0), C), rhs = subset(A, erodeGrid(C, w, h, B, 1));
  check(lhs === rhs, "adjunction");
}

current = "opening/closing properties (lib)";
for (let t = 0; t < 60; t++) {
  const w = 10, h = 9, A = randMask(w, h, 0.55), B = libSE(randSE());
  const o = MO.openBinary(A, w, h, B), c = MO.closeBinary(A, w, h, B);
  check(subset(o, A) && subset(A, c), "A ∘ B ⊆ A ⊆ A • B");
  check(eq(MO.openBinary(o, w, h, B), o), "opening idempotent");
  check(eq(MO.closeBinary(c, w, h, B), c), "closing idempotent");
  const A2 = Uint8Array.from(A, (v, i) => v | (i % 7 === 0 ? 1 : 0));
  check(subset(o, MO.openBinary(A2, w, h, B)), "opening increasing");
}

current = "two lines make a square";
for (let t = 0; t < 30; t++) {
  const w = 9, h = 8, A = randMask(w, h, 0.2);
  const sq: Off[] = []; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) sq.push([dx, dy]);
  const lines = applyStages([{ kind: "dilate", se: [[-1, 0], [0, 0], [1, 0]] }, { kind: "dilate", se: [[0, -1], [0, 0], [0, 1]] }], A, w, h, [], 0);
  check(eq(lines, dilateGrid(A, w, h, sq, 0)), "row then column = square");
}

// ---------------------------------------------------------------------------------------
current = "distance transforms vs brute force";
for (let t = 0; t < 40; t++) {
  const w = 4 + Math.floor(R.unif() * 9), h = 4 + Math.floor(R.unif() * 9), A = randMask(w, h, 0.75);
  const bg: [number, number][] = [];
  for (let i = 0; i < w * h; i++) if (!A[i]) bg.push([i % w, Math.floor(i / w)]);
  const cb = chamferGrid(A, w, h, "cityblock"), ch = chamferGrid(A, w, h, "chessboard");
  const lcb = DT.chamfer(A, w, h, "cityblock"), lch = DT.chamfer(A, w, h, "chessboard"), e2 = DT.edt(A, w, h, { squared: true });
  for (let i = 0; i < w * h; i++) {
    const x = i % w, y = Math.floor(i / w);
    let c1 = Infinity, c2 = Infinity, c3 = Infinity;
    for (const [bx, by] of bg) { c1 = Math.min(c1, Math.abs(bx - x) + Math.abs(by - y)); c2 = Math.min(c2, Math.max(Math.abs(bx - x), Math.abs(by - y))); c3 = Math.min(c3, (bx - x) ** 2 + (by - y) ** 2); }
    if (!bg.length) { check(cb[i] === INF, "no background → ∞"); continue; }
    check(cb[i] === c1 && lcb[i] === c1, `city-block at ${x},${y}: ${cb[i]} vs ${c1}`);
    check(ch[i] === c2 && lch[i] === c2, `chessboard at ${x},${y}`);
    check(e2[i] === c3, `squared EDT at ${x},${y}`);
  }
}

current = "erosion by a disc = threshold of the EDT";
for (let t = 0; t < 20; t++) {
  const w = 30, h = 26, A = randMask(w, h, 0.85), d2 = DT.edt(A, w, h, { squared: true });
  for (const r of [1, 1.5, 2, 3, 4.2]) {
    const er = MO.erodeBinary(A, w, h, MO.seDisk(r));
    check(eq(er, Uint8Array.from(d2, v => (v > r * r + 1e-9 ? 1 : 0))), `r = ${r}`);
  }
  for (const r of [1, 2, 3]) {
    check(eq(MO.erodeBinary(A, w, h, MO.seDiamond(r)), Uint8Array.from(DT.chamfer(A, w, h, "cityblock"), v => (v > r ? 1 : 0))), `diamond r = ${r}`);
    check(eq(MO.erodeBinary(A, w, h, MO.seSquare(r)), Uint8Array.from(DT.chamfer(A, w, h, "chessboard"), v => (v > r ? 1 : 0))), `square r = ${r}`);
  }
}

// ---------------------------------------------------------------------------------------
current = "granulo opening function vs lib openings";
{
  const w = 48, h = 40, A = new Uint8Array(w * h);
  for (let k = 0; k < 9; k++) {
    const cx = R.unif() * w, cy = R.unif() * h, rr = 1 + R.unif() * 7;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if ((x - cx) ** 2 + (y - cy) ** 2 <= rr * rr) A[y * w + x] = 1;
  }
  for (let i = 0; i < w * h; i++) if (R.unif() < 0.03) A[i] = 1;
  const ses: [SEKind, (r: number) => MO.SE][] = [["disc", r => MO.seDisk(r)], ["square", r => MO.seSquare(r)], ["diamond", r => MO.seDiamond(r)], ["hline", r => MO.seLine(2 * r + 1, 0)]];
  for (const [kind, mk] of ses) {
    const of = openingFunction(A, w, h, kind, 12);
    for (let r = 0; r <= 8; r++) {
      const o = r === 0 ? A : MO.openBinary(A, w, h, mk(r));
      // union of the openings by B_r' for all r' ≥ r: the same as A ∘ B_r for the square,
      // diamond and line (each B_r' is a union of translates of B_r), slightly larger for
      // digital discs, which are not (the granulometry by discs of radius ≥ r)
      const u = Uint8Array.from(o);
      for (let q = Math.max(1, r); q <= 12; q++) { const oq = MO.openBinary(A, w, h, mk(q)); for (let i = 0; i < u.length; i++) u[i] |= oq[i]; }
      check(eq(Uint8Array.from(of, v => (v >= r ? 1 : 0)), u), `${kind} r = ${r}`);
      if (kind !== "disc") check(eq(u, o), `${kind}: union = opening, r = ${r}`);
    }
    const g = granulometry(of, 12);
    let n = 0; for (const v of A) n += v;
    check(g.area[0] === n && g.spectrum.reduce((s, v) => s + v, 0) === n, `${kind}: spectrum sums to |A|`);
  }
}
// 1D: runs of length 1, 2, 2, 4, 5 and line SEs (the exercise)
{
  const runs = [1, 2, 2, 4, 5], row: number[] = [0];
  for (const L of runs) { for (let k = 0; k < L; k++) row.push(1); row.push(0, 0); }
  const of = openingFunction(row, row.length, 1, "hline", 5), g = granulometry(of, 5);
  // a(r) with a line of length 2r + 1 keeps runs of length ≥ 2r + 1
  for (let r = 0; r <= 3; r++) check(g.area[r] === runs.filter(L => L >= 2 * r + 1).reduce((s, L) => s + L, 0), `1D runs, r = ${r}`);
}

// ---------------------------------------------------------------------------------------
current = "morph1d vs lib (flat) and non-flat properties";
for (let t = 0; t < 40; t++) {
  const n = 10 + Math.floor(R.unif() * 30), f = Float64Array.from({ length: n }, () => Math.round(R.unif() * 100));
  for (const k of [0, 1, 2, 4]) {
    const s = seHeights(k, "flat"), se = MO.seLine(2 * k + 1, 0);
    check(eqv(erode1(f, s), MO.erode(f, n, 1, se)), `erosion k = ${k}`);
    check(eqv(dilate1(f, s), MO.dilate(f, n, 1, se)), `dilation k = ${k}`);
    check(eqv(open1(f, s), MO.open(f, n, 1, se)), `opening k = ${k}`);
    check(eqv(close1(f, s), MO.close(f, n, 1, se)), `closing k = ${k}`);
    check(eqv(median1(f, k), MO.median(f, n, 1, se)), `median k = ${k}`);
    const p = seHeights(Math.max(1, k), "parabola"), o = open1(f, p), c = close1(f, p);
    check(Array.from(o).every((v, i) => v <= f[i] + 1e-9) && Array.from(c).every((v, i) => v >= f[i] - 1e-9), "non-flat: opening ≤ f ≤ closing");
    check(eqv(open1(o, p), o, 1e-9), "non-flat opening idempotent");
  }
}

current = "tophat fast disc vs lib";
for (let t = 0; t < 6; t++) {
  const w = 37, h = 29, f = Float32Array.from({ length: w * h }, () => Math.round(R.unif() * 255));
  for (const r of [1, 2, 3.5, 5, 9]) {
    check(eqv(erodeDisc(f, w, h, r), MO.erode(f, w, h, MO.seDisk(r))), `erode r = ${r}`);
    check(eqv(dilateDisc(f, w, h, r), MO.dilate(f, w, h, MO.seDisk(r))), `dilate r = ${r}`);
  }
  for (const hw of [1, 3, 7, 20]) check(eqv(rowExtremum(f, w, h, hw, false), MO.erode(f, w, h, MO.seLine(2 * hw + 1, 0))), `row min hw = ${hw}`);
}

// ---------------------------------------------------------------------------------------
current = "recon1d vs lib reconstruction";
for (let t = 0; t < 60; t++) {
  const n = 6 + Math.floor(R.unif() * 30), f = Int32Array.from({ length: n }, () => Math.floor(R.unif() * 10));
  const m = Int32Array.from(f, v => (R.unif() < 0.2 ? v : 0));
  check(eqv(reconstruct(m, f, true), MO.reconstructByDilation(m, f, n, 1, 8)), "by dilation");
  const m2 = Int32Array.from(f, v => (R.unif() < 0.2 ? v : 9));
  check(eqv(reconstruct(m2, f, false), MO.reconstructByErosion(m2, f, n, 1, 8)), "by erosion");
  for (const h of [1, 2, 3]) {
    check(eqv(hmax(f, h), MO.hMaxTransform(f, n, 1, h, 8)), `hmax h = ${h}`);
    check(eqv(hmin(f, h), MO.hMinTransform(f, n, 1, h, 8)), `hmin h = ${h}`);
    // brute force "fill every basin by h": HMIN(x) = min over y of max(f(y) + h, max f between x and y)
    const brute = Array.from(f, (_, x) => {
      let best = Infinity;
      for (let y = 0; y < n; y++) {
        let mx = -Infinity;
        for (let z = Math.min(x, y); z <= Math.max(x, y); z++) mx = Math.max(mx, f[z]);
        best = Math.min(best, Math.max(f[y] + h, mx));
      }
      return best;
    });
    check(eqv(hmin(f, h), brute), `hmin = brute-force fill, h = ${h}`);
  }
  check(eqv(regionalMax(f), MO.regionalMaxima(f, n, 1, 8)), "regional maxima");
  check(eqv(regionalMax(f, true), MO.regionalMinima(f, n, 1, 8)), "regional minima");
  const its = iterates(m, f, true);
  check(its.every((s, k) => k === 0 || Array.from(s).every((v, i) => v >= its[k - 1][i])), "iterates increase");
  // imposition: the only regional minima are the marker runs
  const mk = Uint8Array.from(f, () => (R.unif() < 0.1 ? 1 : 0));
  if (mk.some(v => v)) {
    const { fm, g } = imposeParts(f, mk), imp = reconstruct(fm, g, false);
    check(eq(regionalMax(imp, true), mk) || Array.from(regionalMax(imp, true)).every((v, i) => !v || mk[i]), "imposed minima only at markers");
    check(Array.from(mk).every((v, i) => !v || imp[i] === 0), "markers at 0");
  }
}
check(TWELVE.length === 12, "12-sample signal");

current = "reconstruct (binary) vs lib";
for (let t = 0; t < 30; t++) {
  const w = 20 + Math.floor(R.unif() * 10), h = 18, A = randMask(w, h, 0.55), M = Uint8Array.from(A, v => (v && R.unif() < 0.05 ? 1 : 0));
  check(eq(reconstructBinary(M, A, w, h, 8), MO.reconstructByDilation(M, A, w, h, 8)), "R_A(M), 8-connected");
  check(eq(reconstructBinary(M, A, w, h, 4), MO.reconstructByDilation(M, A, w, h, 4)), "R_A(M), 4-connected");
  check(eq(fillHolesRec(A, w, h), MO.fillHoles(A, w, h, 4)), "fill holes");
  check(eq(clearBorderRec(A, w, h), MO.clearBorder(A, w, h, 8)), "clear border");
  // replay: after k geodesic dilations the marker covers exactly the pixels with steps ≤ k
  const steps = geodesicSteps(M, A, w, h, 8);
  let cur: Float32Array = Float32Array.from(M, (v, i) => (v && A[i] ? 1 : 0));
  for (let k = 1; k <= 6; k++) {
    cur = MO.geodesicDilate(cur, A, w, h, 8);
    check(eq(cur, Uint8Array.from(steps, v => (v >= 0 && v <= k ? 1 : 0))), `replay k = ${k}`);
  }
}

console.log(Object.entries(section).map(([k, v]) => `  ${k}: ${v}`).join("\n"));
console.log(failures ? `${failures} of ${checks} checks FAILED` : `all ${checks} checks passed`);
if (failures) process.exit(1);
