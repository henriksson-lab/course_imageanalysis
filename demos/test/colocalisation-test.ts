// Reference checks for the colocalisation day: lib/coloc.ts, lib/twochannel.ts and the demo
// models. Run:
//   node_modules/.bin/esbuild demos/test/colocalisation-test.ts --bundle --platform=node --log-level=warning | node
import { rng, gauss } from "../src/lib/random";
import * as C from "../src/lib/coloc";
import * as T from "../src/lib/twochannel";
import { pearson, sd } from "../src/lib/stats";
import { colocStats, startThresholds } from "../src/coloc/model";
import * as P from "../src/colocpipeline/model";
import * as O from "../src/colocobjects/model";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

// ---------------------------------------------------------------------------------------
// The original makeChannels of the built coloc demo (copied verbatim before the move), to
// check that pixelScene is bit-identical.
import { gaussianBlur, fromArray } from "../src/lib/imageops";
function makeChannelsOriginal(o: { f?: number; layout?: "mix" | "inside"; offset?: number; noise?: number; diffuse?: number; bleed?: number; shift?: number; seed?: number }) {
  const S = 128;
  function blobs(r: ReturnType<typeof rng>, inside: Uint8Array, n: number, rMin: number, rMax: number, centres?: [number, number, number][]) {
    const img = new Float64Array(S * S), list: [number, number, number][] = centres ? [...centres] : [];
    let guard = 0;
    while (!centres && list.length < n && guard++ < 5000) {
      const x = 4 + (S - 8) * r.unif(), y = 4 + (S - 8) * r.unif(), rad = rMin + (rMax - rMin) * r.unif();
      if (!inside[Math.round(y) * S + Math.round(x)]) continue;
      list.push([x, y, rad]);
    }
    for (const [x, y, rad] of list)
      for (let yy = Math.max(0, Math.floor(y - 3 * rad)); yy <= Math.min(S - 1, Math.ceil(y + 3 * rad)); yy++)
        for (let xx = Math.max(0, Math.floor(x - 3 * rad)); xx <= Math.min(S - 1, Math.ceil(x + 3 * rad)); xx++)
          img[yy * S + xx] += Math.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * rad * rad));
    return { img, list };
  }
  const r = rng(o.seed ?? 4), cells = T.cellMask();
  const f = o.f ?? 0.5, off = o.offset ?? 10, sdv = o.noise ?? 4, diff = o.diffuse ?? 0, bleed = o.bleed ?? 0, shift = Math.round(o.shift ?? 0);
  let A: Float64Array, Bs: Float64Array;
  if ((o.layout ?? "mix") === "inside") {
    const big = blobs(r, cells, 7, 7, 9), small: [number, number, number][] = [];
    big.list.slice(0, 4).forEach(([x, y]) => { for (let k = 0; k < 3; k++) small.push([x + 3 * r.norm(), y + 3 * r.norm(), 1.4]); });
    A = blobs(r, cells, 0, 0, 0, small).img.map(v => 160 * v);
    Bs = big.img.map(v => 110 * Math.min(1, v));
  } else {
    const shared = blobs(r, cells, 40, 1.3, 2.5), indep = blobs(r, cells, 40, 1.3, 2.5);
    A = shared.img.map(v => 150 * v);
    Bs = Float64Array.from(shared.img, (v, i) => 150 * (f * v + (1 - f) * indep.img[i]));
  }
  const blurA = gaussianBlur(fromArray(S, S, A), 1).data, blurB = gaussianBlur(fromArray(S, S, Bs), 1).data;
  const a = new Float64Array(S * S), b = new Float64Array(S * S);
  for (let i = 0; i < S * S; i++) a[i] = blurA[i] + diff * cells[i];
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const xs = Math.min(S - 1, Math.max(0, x - shift)), j = y * S + xs;
    b[y * S + x] = blurB[j] + diff * cells[j] + bleed * a[j];
  }
  const r2 = rng((o.seed ?? 4) + 101);
  for (let i = 0; i < S * S; i++) { a[i] = Math.max(0, a[i] + off + gauss(r2, 0, sdv)); b[i] = Math.max(0, b[i] + off + gauss(r2, 0, sdv)); }
  return { a, b };
}

function testScene(): void {
  current = "pixelScene";
  const opts = [{}, { f: 1 }, { f: 0 }, { f: 0, diffuse: 30 }, { layout: "inside" as const }, { f: 0, bleed: 0.4 }, { f: 1, shift: 3 }, { f: 0.25, noise: 15, offset: 50, seed: 9 }];
  for (const o of opts) {
    const p = T.pixelScene(o), q = makeChannelsOriginal(o);
    let same = true;
    for (let i = 0; i < p.a.length; i++) if (p.a[i] !== q.a[i] || p.b[i] !== q.b[i]) { same = false; break; }
    check(same, "bit-identical to the built makeChannels for " + JSON.stringify(o));
  }
  // built numbers quoted in the deck
  const cells = T.cellMask();
  const rp = C.pearsonMasked(T.pixelScene({ f: 1 }).a, T.pixelScene({ f: 1 }).b, cells);
  check(Math.abs(rp - 0.988) < 0.0005, "perfect r = 0.988: " + rp);
  const ind = T.pixelScene({ f: 0 });
  check(Math.abs(C.pearsonMasked(ind.a, ind.b, cells) + 0.08) < 0.005, "independent inside cells r = −0.08");
  const bg = T.pixelScene({ f: 0, diffuse: 30 });
  check(Math.abs(C.pearsonMasked(bg.a, bg.b) - 0.51) < 0.005, "whole image with diffuse r = 0.51");
  const cs = colocStats(T.pixelScene({ f: 1 }), cells, ...(() => { const s = startThresholds(T.pixelScene({ f: 1 }), cells); return [s.tA, s.tB] as [number, number]; })());
  check(Math.abs(cs.m1 - 0.940) < 0.0005 && Math.abs(cs.m2 - 0.939) < 0.0005, "perfect M1, M2 = 0.940, 0.939");
}

// ---------------------------------------------------------------------------------------
function testPixelCoefficients(): void {
  current = "pixel coefficients";
  const r = rng(1);
  for (let t = 0; t < 10; t++) {
    const n = 50, a = Float64Array.from({ length: n }, () => 10 * r.unif()), b = Float64Array.from(a, v => 3 * v + 7);
    check(close(C.pearsonMasked(a, b), 1, 1e-12), "Pearson of aA + b is 1");
    check(close(C.spearmanMasked(a, b), 1, 1e-12), "Spearman of an increasing function is 1");
    const c = Float64Array.from(a, v => v * v * v);
    check(close(C.spearmanMasked(a, c), 1, 1e-12), "Spearman of a monotone map is 1");
    const m = C.manders(a, a, 0, 0);
    check(m.m1 === 1 && m.m2 === 1, "M1 = M2 = 1 when A = B");
    const s = 2.5, bs = Float64Array.from(a, v => s * v), ta = 4;
    const ms = C.manders(a, bs, ta, s * ta);
    check(close(ms.m1, 1) && close(ms.m2, 1), "M1 = M2 = 1 for B = aA on matched thresholds");
    const x = Float64Array.from({ length: n }, () => r.unif() + 0.1), y = Float64Array.from({ length: n }, () => r.unif() + 0.1);
    const k = C.kCoefficients(x, y);
    check(close(C.overlapCoefficient(x, y) ** 2, k.k1 * k.k2, 1e-12), "overlap² = k1 k2");
    // Pearson as the PDM sum
    const d = C.pdm(x, y);
    const mx = x.reduce((p, q) => p + q) / n, my = y.reduce((p, q) => p + q) / n;
    const sx = Math.sqrt(x.reduce((p, q) => p + (q - mx) ** 2, 0) / n), sy = Math.sqrt(y.reduce((p, q) => p + (q - my) ** 2, 0) / n);
    check(close(d.reduce((p, q) => p + q) / (n * sx * sy), pearson(x, y), 1e-10), "r = Σ PDM / (n sA sB)");
    // Manders without own threshold against brute force
    const mn = C.mandersNoOwnThreshold(x, y, 0.5, 0.6);
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) { den += x[i]; if (y[i] > 0.6) num += x[i]; }
    check(close(mn.m1, num / den), "Manders without own threshold");
  }
  // ranks with ties
  const rk = C.ranks([3, 1, 3, 2]);
  check(rk[0] === 3.5 && rk[1] === 1 && rk[2] === 3.5 && rk[3] === 2, "average ranks with ties");
  // ICQ
  const a = Float64Array.from({ length: 2000 }, () => r.norm());
  check(close(C.icq(a, a).icq, 0.5), "ICQ of A = A is 0.5");
  const b = Float64Array.from({ length: 2000 }, () => r.norm());
  check(Math.abs(C.icq(a, b).icq) < 0.05, "ICQ of independent symmetric noise ≈ 0: " + C.icq(a, b).icq);
}

// ---------------------------------------------------------------------------------------
function testNullAndShift(): void {
  current = "shift curve and randomisation";
  // a blurred scene: pixel shuffle null narrower than block null
  const sc = T.pixelScene({ f: 0, blur: 2 });
  const p1 = C.randomisationTest(sc.a, sc.b, 128, 128, { block: 1, n: 100, seed: 3 });
  const p8 = C.randomisationTest(sc.a, sc.b, 128, 128, { block: 8, n: 100, seed: 3 });
  check(sd(p1.null) < sd(p8.null), `1-px shuffle null sd ${sd(p1.null)} < 8-px block null sd ${sd(p8.null)}`);
  // masked block scramble: blocks touching the mask are permuted, the rest stays; values preserved
  {
    const w = 16, h = 16, b = Float64Array.from({ length: w * h }, (_, i) => i), mask = new Uint8Array(w * h);
    for (let y = 2; y < 9; y++) for (let x = 3; x < 12; x++) mask[y * w + x] = 1;
    const out = C.blockScrambleMasked(b, w, h, 4, rng(5), mask);
    check([...out].sort((p, q) => p - q).every((v, i) => v === i), "masked scramble keeps the values");
    let untouched = true;
    for (let y = 12; y < 16; y++) for (let x = 0; x < 16; x++) if (out[y * w + x] !== b[y * w + x]) untouched = false;
    check(untouched, "blocks outside the mask stay");
    let moved = 0;
    for (let i = 0; i < w * h; i++) if (out[i] !== b[i]) moved++;
    check(moved > 0, "some blocks inside the mask move");
    // blocks stay intact: within a 4 × 4 block, values are consecutive in rows
    let intact = true;
    for (let by = 0; by < 4; by++) for (let bx = 0; bx < 4; bx++) {
      const v0 = out[(4 * by) * w + 4 * bx];
      for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) if (out[(4 * by + y) * w + 4 * bx + x] !== v0 + y * w + x) intact = false;
    }
    check(intact, "blocks are moved whole");
  }
  // permutation p
  check(close(C.permutationP([0.1, 0.2, 0.3], 0.25), 2 / 4), "permutation p with +1");
  // shift curve recovers an imposed shift
  for (const s of [2.4, -1.3]) {
    const sh = T.pixelScene({ f: 1, shiftX: s, noise: 1 });
    const c = C.shiftCurve(sh.a, sh.b, 128, 128, 6, sh.cells, "x");
    const pk = C.curvePeak(c.shifts, c.r);
    check(Math.abs(pk.at - s) < 0.15, `shift curve peak ${pk.at} for imposed ${s}`);
  }
  // shift curve peak by brute force at t = 0 equals Pearson over the mask
  const sc0 = T.pixelScene({});
  const c0 = C.shiftCurve(sc0.a, sc0.b, 128, 128, 0, sc0.cells);
  check(close(c0.r[0], C.pearsonMasked(sc0.a, sc0.b, sc0.cells), 1e-9), "shift 0 = Pearson");
}

// ---------------------------------------------------------------------------------------
function testObjects(): void {
  current = "objects";
  const r = rng(2);
  // chance fraction against simulated Poisson points on a torus-free big square
  const W = 400, rho = 0.002, d = 2;
  const nB = Math.round(rho * W * W);
  let hit = 0, tot = 0;
  for (let rep = 0; rep < 100; rep++) {
    const bx = Float64Array.from({ length: nB }, () => W * r.unif()), by = Float64Array.from({ length: nB }, () => W * r.unif());
    const ax = Float64Array.from({ length: 200 }, () => 20 + (W - 40) * r.unif()), ay = Float64Array.from({ length: 200 }, () => 20 + (W - 40) * r.unif());
    const nn = C.crossNearest(ax, ay, bx, by);
    hit += C.fractionWithin(nn.dist, d) * 200; tot += 200;
  }
  const fSim = hit / tot, fTh = C.chanceFraction(rho, d), se = Math.sqrt(fTh * (1 - fTh) / tot);
  check(Math.abs(fSim - fTh) < 3 * se, `chance fraction ${fSim} vs ${fTh} (se ${se})`);
  // cross-K of independent points ≈ π r² (points away from the border only for A)
  const bx = Float64Array.from({ length: 2000 }, () => W * r.unif()), by = Float64Array.from({ length: 2000 }, () => W * r.unif());
  const ax = Float64Array.from({ length: 300 }, () => 30 + (W - 60) * r.unif()), ay = Float64Array.from({ length: 300 }, () => 30 + (W - 60) * r.unif());
  const radii = [5, 10, 20];
  const K = C.crossK(ax, ay, bx, by, radii, W * W);
  radii.forEach((rr, k) => check(Math.abs(K[k] / (Math.PI * rr * rr) - 1) < 0.15, `K(${rr}) / πr² = ${K[k] / (Math.PI * rr * rr)}`));
  // one-to-one pairing against brute force on small sets
  for (let t = 0; t < 30; t++) {
    const n1 = 1 + Math.floor(5 * r.unif()), n2 = 1 + Math.floor(5 * r.unif());
    const px = Array.from({ length: n1 }, () => 10 * r.unif()), py = Array.from({ length: n1 }, () => 10 * r.unif());
    const qx = Array.from({ length: n2 }, () => 10 * r.unif()), qy = Array.from({ length: n2 }, () => 10 * r.unif());
    const dMax = 4;
    const res = C.pairOneToOne(px, py, qx, qy, dMax);
    const count = Array.from(res.partner).filter(j => j >= 0).length, total = Array.from(res.dist).filter(Number.isFinite).reduce((a, b) => a + b, 0);
    // brute force over all injective partial maps
    let bestC = -1, bestT = Infinity;
    const used = new Array(n2).fill(false);
    const rec = (i: number, c: number, tot: number) => {
      if (i === n1) { if (c > bestC || (c === bestC && tot < bestT - 1e-12)) { bestC = c; bestT = tot; } return; }
      rec(i + 1, c, tot);
      for (let j = 0; j < n2; j++) if (!used[j]) {
        const dd = Math.hypot(px[i] - qx[j], py[i] - qy[j]);
        if (dd > dMax) continue;
        used[j] = true; rec(i + 1, c + 1, tot + dd); used[j] = false;
      }
    };
    rec(0, 0, 0);
    check(count === bestC && close(total, bestT, 1e-9), `one-to-one: ${count}/${total} vs ${bestC}/${bestT}`);
  }
  // Gaussian overlap against numeric integration
  for (const [dl, s1, s2] of [[0, 1, 1], [1.5, 1, 1], [2, 1, 1.6], [0, 1.2, 2]]) {
    let ab = 0, aa = 0, bb = 0;
    const h = 0.05;
    for (let y = -12; y <= 12; y += h) for (let x = -12; x <= 14; x += h) {
      const g1 = Math.exp(-(x * x + y * y) / (2 * s1 * s1)) / (s1 * s1), g2 = Math.exp(-((x - dl) ** 2 + y * y) / (2 * s2 * s2)) / (s2 * s2);
      ab += g1 * g2; aa += g1 * g1; bb += g2 * g2;
    }
    check(close(C.gaussianOverlap(dl, s1, s2), ab / Math.sqrt(aa * bb), 1e-6), `Gaussian overlap ${dl},${s1},${s2}`);
  }
  // mean measured distance against simulation
  for (const [dd, s] of [[0, 20], [60, 20], [30, 10]]) {
    let sum = 0;
    const N = 40000;
    for (let k = 0; k < N; k++) sum += Math.hypot(dd + s * r.norm() - s * r.norm(), s * r.norm() - s * r.norm());
    check(Math.abs(sum / N - C.meanMeasuredDistance(dd, s, s)) < 0.5, `mean measured distance ${dd}, ${s}: ${sum / N} vs ${C.meanMeasuredDistance(dd, s, s)}`);
  }
  // spot detector finds well-separated bright spots
  const xs = [20.3, 40.7, 60.2, 80.5], ys = [30.6, 50.1, 70.4, 90.8];
  const img = T.addNoise(T.spotImage(xs, ys, 128, 128, 1.5, 3000), 7, 10);
  const sp = C.detectSpots(img, 128, 128, 1.5, 5);
  check(sp.length === 4, "4 spots detected: " + sp.length);
  for (const s of sp) {
    const j = xs.findIndex((x, k) => Math.hypot(x - s.x, ys[k] - s.y) < 0.3);
    check(j >= 0, `spot at ${s.x.toFixed(2)}, ${s.y.toFixed(2)} located within 0.3 px`);
  }
}

// ---------------------------------------------------------------------------------------
function testTwoChannel(): void {
  current = "twochannel";
  // spot image integrates to the photons
  const im = T.spotImage([64.3], [60.8], 128, 128, 1.3, 1000);
  check(close(im.reduce((a, b) => a + b, 0), 1000, 1e-6), "spot image sums to the photons");
  // compensation with the true s recovers B
  const r = rng(4);
  const a = Float64Array.from({ length: 100 }, () => r.unif() * 50), b = Float64Array.from({ length: 100 }, () => r.unif() * 50);
  const leak = Float64Array.from(b, (v, i) => v + 0.3 * a[i]);
  const rec = T.compensate(a, leak, 0.3);
  check(rec.every((v, i) => close(v, b[i], 1e-12)), "compensate with the true s recovers B");
  check(close(T.leakSlope(a, Float64Array.from(a, v => 0.3 * v)), 0.3), "leak slope through the origin");
  // registration recovers an imposed sub-pixel shift
  const bead = T.beadPositions(128, 128);
  const A = T.spotImage(bead.x, bead.y, 128, 128, 1.3, 5000);
  for (const [sx, sy] of [[1.4, -0.6], [-2.3, 0.8], [0.3, 0.2]]) {
    const B = T.spotImage(bead.x, bead.y, 128, 128, 1.3, 5000, sx, sy);
    const t = T.registerShift(T.addNoise(A, 1, 5), T.addNoise(B, 2, 5), 128, 128);
    check(Math.abs(t.tx - sx) < 0.1 && Math.abs(t.ty - sy) < 0.1, `registration ${t.tx}, ${t.ty} for ${sx}, ${sy}`);
  }
  // shiftImage moves the content by +s
  const sh = T.shiftImage(A, 128, 128, 2, 0);
  check(Math.abs(sh[Math.round(bead.y[12]) * 128 + Math.round(bead.x[12]) + 2] - A[Math.round(bead.y[12]) * 128 + Math.round(bead.x[12])]) < 1e-6 * 5000, "shiftImage by +2 px");
  // spot pair scene: partners at the right distance
  const scn = T.spotPairScene({ w: 128, h: 128, nA: 50, pairFraction: 0.6, distance: 2.5, extraB: 10, seed: 3 });
  let ok = true, nPaired = 0;
  scn.partner.forEach((j, i) => { if (j >= 0) { nPaired++; if (!close(Math.hypot(scn.bx[j] - scn.ax[i], scn.by[j] - scn.ay[i]), 2.5, 1e-9)) ok = false; } });
  check(ok && scn.bx.length === nPaired + 10, "partners at distance 2.5, plus 10 extra");
}

function testModels(): void {
  current = "demo models";
  const res = P.runPipeline(P.DEFAULT_PIPE);
  check(res.read.ideal.r === res.read.measured.r, "no confounds: measured = ideal");
  const reg = P.runPipeline({ ...P.DEFAULT_PIPE, shiftXNm: 100, register: true });
  check(Math.abs(reg.estShiftNm![0] - 100) < 3 && Math.abs(reg.estShiftNm![1]) < 3, "bead registration within 3 nm: " + reg.estShiftNm);
  check(Math.abs(reg.read.corrected.r - res.read.ideal.r) < 0.02, "registration restores r");
  const sp = O.makeObjScene({ ...O.DEFAULT_OBJ, pair: 0, extraB: 12 });
  const d = O.distances(sp, 130, false);
  check(Math.abs(d.fObs - d.fChance) < 0.05, `independent sparse spots near chance: ${d.fObs} vs ${d.fChance}`);
}

testScene();
testPixelCoefficients();
testNullAndShift();
testObjects();
testTwoChannel();
testModels();
console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
