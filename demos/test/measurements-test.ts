// Reference checks for the measurements day: lib/features.ts, lib/glcm.ts,
// lib/neighbours.ts, and the demo models. Each routine is compared with a brute-force or
// textbook version on small inputs. Run:
//   node_modules/.bin/esbuild demos/test/measurements-test.ts --bundle --platform=node --log-level=warning | node
import { rng } from "../src/lib/random";
import * as FE from "../src/lib/features";
import * as GL from "../src/lib/glcm";
import * as NB from "../src/lib/neighbours";
import { pearson, median } from "../src/lib/stats";
import { label2d } from "../src/lib/label";
import { makeCell, PATTERNS, makePlate } from "../src/intensityfeat/model";
import { makePatch, PATCHES } from "../src/glcm/model";
import { makePopulation } from "../src/neighbours/model";
import { makeChannels } from "../src/coloc/model";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0;
let current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

// ---------------------------------------------------------------------------------------
function testFeatures(): void {
  current = "features";
  const r = rng(3);
  // statistics against a naive version
  for (let t = 0; t < 20; t++) {
    const n = 1 + Math.floor(r.unif() * 30);
    const v = Float64Array.from({ length: n }, () => Math.round(r.unif() * 50));
    const s = FE.intensityStats(v);
    const sorted = [...v].sort((a, b) => a - b);
    const q = (p: number) => { const hh = (n - 1) * p, lo = Math.floor(hh); return sorted[lo] + (hh - lo) * (sorted[Math.ceil(hh)] - sorted[lo]); };
    const m = v.reduce((a, b) => a + b, 0) / n;
    check(close(s.integrated, m * n), "integrated = sum");
    check(close(s.sd, Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / n)), "sd over n");
    check(close(s.median, q(0.5)) && close(s.q1, q(0.25)) && close(s.q3, q(0.75)), "quartiles type 7");
    const dev = [...v].map(x => Math.abs(x - q(0.5))).sort((a, b) => a - b);
    const hh = (n - 1) / 2;
    check(close(s.mad, dev[Math.floor(hh)] + (hh - Math.floor(hh)) * (dev[Math.ceil(hh)] - dev[Math.floor(hh)])), "MAD");
    // offset: mean, median, quartiles shift by b; sd and MAD do not
    const b = 7, s2 = FE.intensityStats(v.map(x => x + b));
    check(close(s2.mean, s.mean + b) && close(s2.median, s.median + b) && close(s2.q1, s.q1 + b), "offset shifts location");
    check(close(s2.sd, s.sd) && close(s2.mad, s.mad), "offset leaves spread");
    check(close(s2.integrated, s.integrated + b * n), "offset adds b|Ω| to the integral");
  }
  // edge band against the definition
  const w = 12, h = 10;
  for (let t = 0; t < 10; t++) {
    const m = Uint8Array.from({ length: w * h }, () => (r.unif() < 0.6 ? 1 : 0));
    const e = FE.edgeBand(m, w, h);
    let ok = true;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const out = (xx: number, yy: number) => xx < 0 || yy < 0 || xx >= w || yy >= h || !m[yy * w + xx];
      const want = m[i] && (out(x - 1, y) || out(x + 1, y) || out(x, y - 1) || out(x, y + 1)) ? 1 : 0;
      if (want !== e[i]) ok = false;
    }
    check(ok, "edge band = pixels with a 4-neighbour outside");
  }
  // disc: uniform -> mass displacement 0, FracAtD sums to 1, MeanFrac = 1
  const W = 41, disc = new Uint8Array(W * W), uni = new Float64Array(W * W);
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) if ((x - 20) ** 2 + (y - 20) ** 2 <= 15 * 15) { disc[y * W + x] = 1; uni[y * W + x] = 5; }
  const px = FE.pixelsOf(disc);
  check(close(FE.massDisplacement(uni, px, W), 0, 1e-12), "uniform disc: mass displacement 0");
  for (const nb of [2, 3, 5, 8]) {
    const rd = FE.radialDistribution(uni, disc, W, W, nb, [20, 20]);
    check(close(rd.fracAtD.reduce((a, b) => a + b, 0), 1), "FracAtD sums to 1");
    check([...rd.meanFrac].every(v => close(v, 1)), "MeanFrac = 1 for a uniform object");
    check([...rd.radialCV].every(v => !(Math.abs(v) > 1e-12)), "RadialCV = 0 for a uniform object");
  }
  // bright spot off-centre: mass displacement > 0, towards the spot
  const spot = Float64Array.from(uni);
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) if ((x - 28) ** 2 + (y - 20) ** 2 <= 9) spot[y * W + x] += 50;
  const [wx] = FE.weightedCentroid(spot, px, W);
  check(wx > 20 && FE.massDisplacement(spot, px, W) > 1, "spot pulls the weighted centroid");
  // robust z: the controls have median 0
  const ctrl = Float64Array.from({ length: 9 }, () => 10 + 5 * r.norm());
  check(close(median(FE.robustZ(ctrl, ctrl)), 0, 1e-12), "robust z of the controls has median 0");
  const zc = FE.zScore(ctrl, ctrl);
  check(close(zc.reduce((a, b) => a + b, 0), 0, 1e-9), "z of the controls has mean 0");
  // aggregation
  const ag = FE.aggregate([1, 2, 3, 10, 20], [0, 0, 0, 1, 1], 2, "median");
  check(ag[0] === 2 && ag[1] === 15, "per-group median");
  // background ring excludes the object and stays within gap..gap+width
  const ring = FE.backgroundRing(disc, W, W, 2, 3);
  let okRing = true;
  for (let i = 0; i < W * W; i++) if (ring[i] && disc[i]) okRing = false;
  check(okRing && ring.reduce((a, b) => a + b, 0) > 0, "background ring outside the object");
}

// ---------------------------------------------------------------------------------------
function testGlcm(): void {
  current = "glcm";
  const r = rng(7);
  // constant image
  const c = new Int32Array(25).fill(2);
  const pc = GL.normaliseGlcm(GL.glcm(c, 5, 5, 4, 1, 0, { symmetric: true }));
  const fc = GL.haralick(pc, 4);
  check(close(fc.asm, 1) && fc.contrast === 0 && close(fc.homogeneity, 1) && fc.entropy === 0, "constant image: ASM 1, contrast 0");
  check(Number.isNaN(fc.correlation), "constant image: correlation undefined");
  for (let t = 0; t < 20; t++) {
    const w = 3 + Math.floor(r.unif() * 6), h = 3 + Math.floor(r.unif() * 6), L = 2 + Math.floor(r.unif() * 5);
    const q = Int32Array.from({ length: w * h }, () => Math.floor(r.unif() * L));
    const mask = Uint8Array.from({ length: w * h }, () => (r.unif() < 0.8 ? 1 : 0));
    for (const ang of [0, 45, 90, 135]) {
      const d = 1 + Math.floor(r.unif() * 2);
      const [dx, dy] = GL.offsetFor(d, ang);
      const P = GL.glcm(q, w, h, L, dx, dy, { mask });
      // brute force
      const B = new Float64Array(L * L);
      const xs: number[] = [], ys: number[] = [];
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const x2 = x + dx, y2 = y + dy;
        if (x2 < 0 || y2 < 0 || x2 >= w || y2 >= h || !mask[y * w + x] || !mask[y2 * w + x2]) continue;
        B[q[y * w + x] * L + q[y2 * w + x2]]++;
        xs.push(q[y * w + x]); ys.push(q[y2 * w + x2]);
      }
      check(P.every((v, i) => v === B[i]), "glcm = brute-force pair count");
      const S = GL.glcm(q, w, h, L, dx, dy, { mask, symmetric: true });
      let symOk = true;
      for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) if (S[i * L + j] !== S[j * L + i] || S[i * L + j] !== B[i * L + j] + B[j * L + i]) symOk = false;
      check(symOk, "symmetric glcm = P + Pᵀ");
      if (xs.length < 3) continue;
      const f = GL.haralick(GL.normaliseGlcm(P), L);
      const n = xs.length;
      check(close(f.contrast, xs.reduce((a, x, k) => a + (x - ys[k]) ** 2, 0) / n), "contrast = mean squared difference of pairs");
      const rr = pearson(xs, ys);
      if (Number.isFinite(rr) && Number.isFinite(f.correlation)) check(close(f.correlation, rr, 1e-9), "correlation = Pearson r of the pairs");
      let hom = 0;
      for (let k = 0; k < n; k++) hom += 1 / (1 + (xs[k] - ys[k]) ** 2);
      check(close(f.homogeneity, hom / n), "homogeneity = mean of 1/(1+(i−j)²)");
      check(close(f.energy * f.energy, f.asm), "energy = √ASM");
      check(f.asm <= 1 + 1e-12 && f.entropy >= 0, "ranges");
    }
  }
  // quantisation
  const qq = GL.quantise([0, 0.24, 0.25, 0.99, 1, 2, -1], 4, 0, 1);
  check([0, 0, 1, 3, 3, 3, 0].every((v, i) => v === qq[i]), "quantise and clamp");
  check(GL.offsetFor(2, 45).join() === "2,-2" && GL.offsetFor(1, 90).join() === "0,-1" && GL.offsetFor(1, 135).join() === "-1,-1", "offsets: 45° = up-right");
  // LBP
  const code = GL.lbp3x3([5, 9, 1, 4, 4, 6, 7, 2, 3]);
  // clockwise from top-left: 5 9 1 6 3 2 7 4 -> ≥4: 1 1 0 1 0 0 1 1 = 0b11010011 = 211
  check(code === 211, `LBP code 211, got ${code}`);
  check(GL.lbpRotInv(211) === Math.min(...Array.from({ length: 8 }, (_, k) => GL.rotateCode(211, k))), "rotation-invariant = min rotation");
  check(GL.lbpRotInv(GL.rotateCode(211, 3)) === GL.lbpRotInv(211), "rotation invariance");
  check(GL.lbpTransitions(0b00011100) === 2 && GL.lbpTransitions(0b01010101) === 8, "transitions");
  // demo patches build and have the right size; "same histogram" pair has identical sorted values
  for (const p of PATCHES) { const im = makePatch(p.id, {}); check(im.length === 64 * 64, "patch size " + p.id); }
  const a = makePatch("pairA", {}), b = makePatch("pairB", {});
  const sa = [...a].sort((x, y) => x - y), sb = [...b].sort((x, y) => x - y);
  check(sa.every((v, i) => v === sb[i]), "permuted pair has the same histogram");
}

// ---------------------------------------------------------------------------------------
function testNeighbours(): void {
  current = "neighbours";
  const r = rng(11);
  for (let t = 0; t < 15; t++) {
    const w = 14, h = 11;
    const lab = Int32Array.from({ length: w * h }, () => (r.unif() < 0.45 ? 1 + Math.floor(r.unif() * 5) : 0));
    const count = 5;
    for (const conn of [4, 8] as const) {
      const T = NB.touching(lab, w, h, count, conn);
      const off = conn === 4 ? [[1, 0], [-1, 0], [0, 1], [0, -1]] : [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
      const sets = Array.from({ length: count + 1 }, () => new Set<number>());
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (const [dx, dy] of off) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const a = lab[y * w + x], b = lab[yy * w + xx];
        if (a && b && a !== b) { sets[a].add(b); sets[b].add(a); }
      }
      check(sets.every((s, l) => [...s].sort((a, b) => a - b).join() === T.neighbours[l].join()), "adjacency = brute force");
      check([...T.touching].every((v, l) => v <= T.boundary[l]), "touching ≤ boundary");
    }
    // expansion: every pixel gets the label of a nearest object pixel
    const ex = NB.expandLabels(lab, w, h);
    let ok = true;
    for (let i = 0; i < w * h; i++) {
      if (lab[i]) { if (ex[i] !== lab[i]) ok = false; continue; }
      const x = i % w, y = Math.floor(i / w);
      let best = Infinity;
      for (let j = 0; j < w * h; j++) if (lab[j]) best = Math.min(best, (x - j % w) ** 2 + (y - Math.floor(j / w)) ** 2);
      let mine = Infinity;
      for (let j = 0; j < w * h; j++) if (lab[j] === ex[i]) mine = Math.min(mine, (x - j % w) ** 2 + (y - Math.floor(j / w)) ** 2);
      if (Math.abs(mine - best) > 1e-9) ok = false;
    }
    check(ok, "expandLabels = nearest object label");
    // within distance: brute force over all pixels
    for (const rr of [1, 1.5, 3]) {
      const wd = NB.withinDistance(lab, w, h, count, rr);
      for (let a = 1; a <= count; a++) for (let b = 1; b <= count; b++) {
        if (a === b) continue;
        let m = Infinity;
        for (let i = 0; i < w * h; i++) if (lab[i] === a) for (let j = 0; j < w * h; j++) if (lab[j] === b)
          m = Math.min(m, Math.hypot(i % w - j % w, Math.floor(i / w) - Math.floor(j / w)));
        check((m <= rr) === wd[a].includes(b), `within ${rr}: ${a}-${b}`);
      }
    }
  }
  // nearest neighbours
  const xs = [0, 3, 0, 10, 4], ys = [0, 0, 4, 10, 4];
  const nn = NB.nearestNeighbours(xs, ys);
  check(nn.first[0] === 1 && close(nn.firstDist[0], 3) && nn.second[0] === 2 && close(nn.secondDist[0], 4), "first/second closest");
  check(close(nn.angle[0], 90), "angle between neighbours");
  const ce = NB.clarkEvans([1, 1, 1, 1], 16);
  check(close(ce.expected, 1) && close(ce.R, 1), "Clark–Evans: 4 points in 16 px², d = 1 -> R = 1");
  // parents
  const w = 6, h = 4;
  const P = Int32Array.from([1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0]);
  const C = Int32Array.from([0, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0]);
  const pa = NB.assignParents(C, P, w, h, 3, 2, "overlap");
  check(pa.parent[1] === 1 && pa.parent[2] === 2 && pa.parent[3] === 0 && pa.nParents[2] === 2, "parents by overlap");
  check(pa.childCount[1] === 1 && pa.childCount[2] === 1, "child counts");
  // colocalisation
  const n = 200, A = Float64Array.from({ length: n }, () => r.unif() * 100);
  check(close(NB.pearsonMasked(A, A.map(v => 3 * v + 7)), 1), "Pearson of aA + b is 1");
  check(close(NB.pearsonMasked(A, A.map(v => -2 * v + 1)), -1), "Pearson of −2A + 1 is −1");
  const Bn = Float64Array.from({ length: n }, () => r.unif() * 100);
  check(close(NB.pearsonMasked(A, Bn), pearson(A, Bn)), "Pearson = lib/stats");
  const md = NB.manders(A, A, 0, 0);
  check(close(md.m1, 1) && close(md.m2, 1), "M1 = M2 = 1 when A = B");
  let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < n; i++) { ab += A[i] * Bn[i]; aa += A[i] ** 2; bb += Bn[i] ** 2; }
  check(close(NB.overlapCoefficient(A, Bn), ab / Math.sqrt(aa * bb)), "overlap coefficient");
  const reg = NB.orthoRegression(A, A.map(v => 2 * v + 5));
  check(close(reg.slope, 2) && close(reg.intercept, 5), "orthogonal regression on an exact line");
  // M1 by hand
  const m = NB.manders([10, 20, 0, 5], [0, 8, 9, 1], 4, 4);
  check(close(m.m1, 20 / 35) && close(m.m2, 8 / 17), "Manders with thresholds");
  const cs = NB.costes(A, A.map(v => v + 5 * r.norm()));
  check(cs.trace.length > 0 && cs.trace[cs.trace.length - 1].r <= 0, "Costes stops at r ≤ 0");
  const sc = NB.blockScramble(Float64Array.from({ length: 64 }, (_, i) => i), 8, 8, 4, rng(1));
  check([...sc].sort((a, b) => a - b).every((v, i) => v === i), "block scramble is a permutation");
}

// ---------------------------------------------------------------------------------------
function testModels(): void {
  current = "models";
  for (const p of PATTERNS) {
    const c = makeCell({ pattern: p.id });
    const s = FE.intensityStats(FE.valuesAt(c.clean, FE.pixelsOf(c.mask)));
    check(Math.abs(s.integrated - c.target) / c.target < 1e-6, `same integral for ${p.id}: ${s.integrated}`);
  }
  const pl = makePlate({});
  check(pl.values.length === 24, "plate 4 × 6");
  const pop = makePopulation({ pattern: "poisson", n: 50, seed: 1 });
  check(pop.count > 30, "population size " + pop.count);
  const L = label2d(pop.nuclei.map(v => (v ? 1 : 0)), pop.w, pop.h, 8);
  check(L.count === pop.count, "nuclei do not touch each other: " + L.count + " vs " + pop.count);
  const ch = makeChannels({});
  check(ch.a.length === ch.w * ch.h, "two channels");
}

testFeatures();
testGlcm();
testNeighbours();
testModels();
console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
