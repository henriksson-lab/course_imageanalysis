// Data for the static figures of the deconvolution day (deconv/, regularise/, richardson-lucy/,
// denoise/), computed with the demos' own code so that every number matches the demos.
// Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/deconvolution/data.ts --bundle --platform=node --log-level=warning | node - <out.json>
// then: python3 -I tools/figs/deconvolution/figs.py <out.json> <repo root>
import * as fs from "fs";
import * as P from "../../../demos/src/lib/psf";
import * as D from "../../../demos/src/lib/deconv";
import { rmse } from "../../../demos/src/lib/metrics";
import { rng } from "../../../demos/src/lib/ui";
import { phantom, beads, render, makePSF, blur, simulate, addNoise, BEADS } from "../../../demos/src/deconvlab/scenes";
import { truthOf } from "../../../demos/src/regpath/profiles";
import * as M from "../../../demos/src/rlsteps/model";
import { testImage } from "../../../demos/src/denoisers/testimage";
import * as DN from "../../../demos/src/lib/denoise";

const out: Record<string, unknown> = {};
const arr = (a: ArrayLike<number>, d = 5) => Array.from(a, v => +v.toFixed(d));
const N = 128;

// ------------------------------------------------------------------ deconv/
function linear(scene: "phantom" | "beads", psf: { kind: "gauss" | "airy"; width: number }, noise: number, m: D.LinearMethod, used = 1, realistic = false, border: "periodic" | "padded" = "periodic") {
  const { truth, clean } = simulate(scene === "phantom" ? phantom(N) : beads(N), N, psf, realistic);
  const obs = noise === 0 ? clean.slice() : addNoise(clean, { kind: "gauss", level: noise });
  let x: Float64Array;
  if (border === "padded") {
    const Mx = 2 * N, pt = D.padTaper(obs, N, N, Mx, Mx);
    const ctx = D.gainContext(D.spectrum2d(P.toOrigin(makePSF(Mx, psf, 1, used), Mx, Mx), Mx, Mx), Mx, Mx);
    const xx = D.restoreSpectrum(D.spectrum2d(pt.data, Mx, Mx), m, ctx);
    x = new Float64Array(N * N);
    for (let y = 0; y < N; y++) for (let i = 0; i < N; i++) x[y * N + i] = xx[(y + pt.oy) * Mx + i + pt.ox];
  } else {
    const ctx = D.gainContext(D.spectrum2d(P.toOrigin(makePSF(N, psf, 1, used), N, N), N, N), N, N);
    x = D.restoreSpectrum(D.spectrum2d(obs, N, N), m, ctx);
  }
  return { truth, obs, x, rmse: rmse(x, truth), rmseObs: rmse(obs, truth) };
}
{
  const g2 = { kind: "gauss" as const, width: 2 }, airy = { kind: "airy" as const, width: 1.0 };
  const tr = linear("beads", airy, 0.01, { kind: "truncated", eps: 0.1 });
  const inv = linear("beads", airy, 0.01, { kind: "inverse" });
  out.truncated = { truth: arr(tr.truth), obs: arr(tr.obs), x: arr(tr.x), rmse: tr.rmse, rmseInv: inv.rmse };
  // Airy OTF along kx (cycles/px) for the gain-curve plot
  const H = D.spectrum2d(P.toOrigin(makePSF(N, airy), N, N), N, N);
  out.airyH = arr(Array.from({ length: N / 2 + 1 }, (_, k) => Math.hypot(H.re[k], H.im[k])));
  const tik: Record<string, unknown>[] = [];
  for (const lam of [1e-4, 0.033, 1]) {
    const r = linear("phantom", g2, 0.01, { kind: "tikhonov", lambda: lam, penalty: "roughness" });
    tik.push({ lambda: lam, x: arr(r.x), rmse: r.rmse });
  }
  const t0 = linear("phantom", g2, 0.01, { kind: "inverse" });
  out.tikhonov = { truth: arr(t0.truth), obs: arr(t0.obs), rmseObs: t0.rmseObs, runs: tik };
  const wp: Record<string, unknown>[] = [];
  for (const used of [0.8, 1, 1.3]) {
    const r = linear("beads", g2, 0.01, { kind: "wiener", snr: 85, model: "flat" }, used);
    const row = Array.from({ length: 40 }, (_, i) => r.x[104 * N + 4 + i]);
    wp.push({ used, row: arr(row), min: Math.min(...row) });
    if (used === 1) out.wrongpsfTruth = arr(Array.from({ length: 40 }, (_, i) => r.truth[104 * N + 4 + i]));
  }
  out.wrongpsf = wp;
  const bp = linear("beads", g2, 0.001, { kind: "wiener", snr: 1e5, model: "flat" }, 1, true, "periodic");
  const bq = linear("beads", g2, 0.001, { kind: "wiener", snr: 1e5, model: "flat" }, 1, true, "padded");
  out.border = { obs: arr(bp.obs), periodic: arr(bp.x), padded: arr(bq.x), rmseP: bp.rmse, rmseQ: bq.rmse };
}

// ------------------------------------------------------------------ regularise/
{
  const n = 256, truth = truthOf("stepramp"), B = D.blur1dOf(D.gaussian1d(n, 3)), r = rng(1);
  const e = Float64Array.from({ length: n }, () => r.norm());
  const y = D.blur1d(truth, B).map((v, i) => v + 0.02 * e[i]);
  const R2 = D.penaltySymbol1d(n, "roughness");
  const LAMS = Array.from({ length: 37 }, (_, i) => 10 ** (-7 + i * 0.25));
  const path = LAMS.map(lam => {
    const x = D.tikhonov1d(y, B, lam, R2), res = D.residual1d(x, y, B);
    return { lam, res: D.rms(res), err: D.rms(x.map((v, i) => v - truth[i])), rho: D.norm2(res), eta: Math.max(1e-12, D.norm2(D.penaltyOp1d(x, "roughness"))) };
  });
  const ic = D.lcurveCorner(path.map(q => q.rho), path.map(q => q.eta));
  const lamDisc = D.crossing(path.map(q => q.lam), path.map(q => q.res), 0.02);
  const xDisc = D.tikhonov1d(y, B, lamDisc, R2), rDisc = D.residual1d(xDisc, y, B);
  const est = (lam: number) => arr(D.tikhonov1d(y, B, lam, R2));
  const errOf = (lam: number) => D.rms(D.tikhonov1d(y, B, lam, R2).map((v, i) => v - truth[i]));
  out.regpath = {
    truth: arr(truth), y: arr(y), path, corner: ic, lamCorner: path[ic].lam, lamDisc,
    discPt: { rho: D.norm2(rDisc), eta: D.norm2(D.penaltyOp1d(xDisc, "roughness")), err: D.rms(xDisc.map((v, i) => v - truth[i])) },
    est: { small: est(1e-6), best: est(0.1), large: est(100) }, errs: { small: errOf(1e-6), best: errOf(0.1), large: errOf(100) },
  };
  const KS = Array.from(new Set(Array.from({ length: 41 }, (_, i) => Math.round(10 ** (i * 0.1)))));
  out.landweber = KS.map(k => {
    const x = D.applyFactors(y, B, D.landweberFactors(B.H2, k));
    return { k, err: D.rms(x.map((v, i) => v - truth[i])), res: D.rms(D.residual1d(x, y, B)) };
  });
}

// ------------------------------------------------------------------ richardson-lucy/
function rl2d(opts: { obj: "beads" | "phantom"; photons: number; bg?: number; bgModel?: boolean; used?: number; ks: number[]; noiseFree?: boolean }) {
  const truth = render(opts.obj === "beads" ? beads(N) : phantom(N), N), hiT = Math.max(...truth);
  const psfT = makePSF(N, { kind: "gauss", width: 2 });
  const bg = opts.bg ?? 0, b = opts.bgModel === false ? 0 : bg;
  const y = M.photonData(blur(truth, psfT, N), opts.photons, bg, 5, opts.noiseFree ?? false);
  const used = makePSF(N, { kind: "gauss", width: 2 }, 1, opts.used ?? 1), A = D.fftOperator2d(used, N, N);
  let x = D.flatStart(y, b);
  const curve: { k: number; err: number }[] = [], snaps: Record<number, number[]> = {};
  const kmax = Math.max(...opts.ks), keep = new Set(M.snapshotIters(kmax));
  const rec = (k: number) => { if (keep.has(k)) curve.push({ k, err: 100 * rmse(x.map(v => v / opts.photons), truth) / hiT }); if (opts.ks.includes(k)) snaps[k] = arr(x.map(v => v / opts.photons), 4); };
  rec(0);
  for (let k = 1; k <= kmax; k++) { x = D.rlStep(x, y, A, b); rec(k); }
  // best flat Wiener over 25 SNR values (as in the demo)
  const ctx = D.gainContext(D.spectrum2d(P.toOrigin(used, N, N), N, N), N, N), G = D.spectrum2d(y.map(v => v - bg), N, N);
  let wBest = Infinity, wX: Float64Array = new Float64Array(0), wSnr = 0;
  for (let i = 0; i < 25; i++) {
    const snr = 10 ** (i / 4), w = D.restoreSpectrum(G, { kind: "wiener", snr, model: "flat" }, ctx).map(v => v / opts.photons), e = rmse(w, truth);
    if (e < wBest) { wBest = e; wX = w; wSnr = snr; }
  }
  return { truth: arr(truth, 4), obs: arr(y.map(v => (v - bg) / opts.photons), 4), curve, snaps, wiener: { x: arr(wX, 4), err: 100 * wBest / hiT, snr: wSnr, negFrac: wX.filter(v => v < 0).length / wX.length } };
}
{
  const semi: Record<string, unknown> = {};
  for (const ph of [1000, 100, 20]) {
    const r = rl2d({ obj: "beads", photons: ph, ks: [1000] });
    const ib = r.curve.reduce((b, q, i) => (q.err < r.curve[b].err ? i : b), 0);
    semi[ph] = { curve: r.curve, bestK: r.curve[ib].k, bestErr: r.curve[ib].err };
  }
  out.semi = semi;
  const r100 = rl2d({ obj: "beads", photons: 100, ks: [8] });
  out.negative = { truth: r100.truth.slice(24 * N, 25 * N), obs: r100.obs.slice(24 * N, 25 * N), rl: r100.snaps[8].slice(24 * N, 25 * N), wiener: r100.wiener.x.slice(24 * N, 25 * N), wienerSnr: r100.wiener.snr, wienerNegFrac: r100.wiener.negFrac, rlMin: Math.min(...r100.snaps[8]) };
  // photons are counted: phantom, 100 photons, Wiener at a few SNRs (negatives) and RL k = 5
  {
    const r = rl2d({ obj: "phantom", photons: 100, ks: [5] });
    const truth = render(phantom(N), N), y = M.photonData(blur(truth, makePSF(N, { kind: "gauss", width: 2 }), N), 100, 0, 5);
    const ctx = D.gainContext(D.spectrum2d(P.toOrigin(makePSF(N, { kind: "gauss", width: 2 }), N, N), N, N), N, N), G = D.spectrum2d(y, N, N);
    const ws: Record<string, unknown>[] = [];
    for (const snr of [r.wiener.snr, 100, 1000]) {
      const w = D.restoreSpectrum(G, { kind: "wiener", snr, model: "flat" }, ctx).map(v => v / 100);
      ws.push({ snr, x: arr(w, 4), neg: w.filter(v => v < 0).length / w.length, min: Math.min(...w), err: 100 * rmse(w, truth) / Math.max(...truth) });
    }
    out.photons = { obs: r.obs, rl5: r.snaps[5], rlMin: Math.min(...r.snaps[5]), rlErr: r.curve.find(q => q.k === 5)!.err, wiener: ws };
  }
  const bgOff = rl2d({ obj: "phantom", photons: 100, bg: 30, bgModel: false, ks: [10, 100] });
  const bgOn = rl2d({ obj: "phantom", photons: 100, bg: 30, bgModel: true, ks: [10] });
  out.background = { truth: bgOff.truth, obsRaw: arr(bgOff.obs.map(v => v + 30 / 100), 4), off100: bgOff.snaps[100], on10: bgOn.snaps[10],
    offMin: Math.min(...bgOff.curve.map(q => q.err)), off100Err: bgOff.curve.find(q => q.k === 100)!.err, on10Err: bgOn.curve.find(q => q.k === 10)!.err };
  const mm: Record<string, unknown>[] = [];
  for (const used of [0.7, 1, 1.3]) {
    const r = rl2d({ obj: "beads", photons: 1000, used, ks: [100] }), img = r.snaps[100];
    const row = Array.from({ length: 40 }, (_, i) => img[104 * N + 4 + i]);
    let rim = Infinity; for (let yy = 104 - 8; yy <= 104 + 8; yy++) for (let xx = 24 - 8; xx <= 24 + 8; xx++) rim = Math.min(rim, img[yy * N + xx]);
    mm.push({ used, row: arr(row), fwhm: P.fwhm(img.slice(104 * N, 105 * N)), rim });
    if (used === 1) out.psfmismatchTruth = arr(Array.from({ length: 40 }, (_, i) => r.truth[104 * N + 4 + i]));
  }
  out.psfmismatch = mm;
  // blind: PSF FWHM along the iterations
  const truth = render(beads(N), N), psfT = makePSF(N, { kind: "gauss", width: 2 });
  const y = M.photonData(blur(truth, psfT, N), 1000, 0, 5, false);
  const sigmaOf = (hh: Float64Array) => { let s = 0; for (let yy = 0; yy < N; yy++) for (let xx = 0; xx < N; xx++) s += hh[yy * N + xx] * ((xx - N / 2) ** 2 + (yy - N / 2) ** 2); return Math.sqrt(s / 2); };
  const fw = (h: Float64Array) => P.fwhm(h.subarray((N / 2) * N, (N / 2 + 1) * N));
  const blind: Record<string, unknown> = { trueFwhm: fw(psfT) };
  for (const mode of ["none", "size", "gauss"]) {
    let h = makePSF(N, { kind: "gauss", width: 2 }, 1, 1.5), x = D.flatStart(y, 0);
    const c: [number, number][] = [[0, fw(h)]];
    for (let k = 1; k <= 300; k++) {
      ({ x, psf: h } = D.blindStep(x, h, y, N, N, { background: 0, supportRadius: mode === "size" ? 6 : undefined }));
      if (mode === "gauss") h = P.gaussianPSF(N, N, sigmaOf(h));
      c.push([k, fw(h)]);
    }
    blind[mode] = c;
  }
  out.blind = blind;
  // 3D
  const zs = 200, vol = M.volume3d(zs);
  const psfW = M.psf3d("widefield", zs), psfC = M.psf3d("confocal", zs);
  const AW = D.fftOperator3d(psfW, M.NX, M.NY, M.NZ), AC = D.fftOperator3d(psfC, M.NX, M.NY, M.NZ);
  const otfImg = (H: D.Spectrum) => {
    const o = new Float64Array(M.NX * M.NZ); let mx = 0;
    for (let kz = 0; kz < M.NZ; kz++) for (let kx = 0; kx < M.NX; kx++) {
      const i = (kz * M.NY) * M.NX + kx, v = Math.hypot(H.re[i], H.im[i]);
      o[((kz + M.NZ / 2) % M.NZ) * M.NX + (kx + M.NX / 2) % M.NX] = v; mx = Math.max(mx, v);
    }
    return arr(o.map(v => v / mx), 6);
  };
  const clean3 = AW.forward(vol), y3 = M.photonData(clean3, 1000, 0, 9);
  const x3 = D.richardsonLucy(y3, AW, 100);
  out.threeD = { NX: M.NX, NZ: M.NZ, zstep: zs, truth: arr(M.xz(vol), 4), obs: arr(M.xz(y3.map(v => v / 1000)), 4), rl: arr(M.xz(x3.map(v => v / 1000)), 4),
    psfW: arr(M.xz(psfW), 7), psfC: arr(M.xz(psfC), 7), otfW: otfImg(AW.H), otfC: otfImg(AC.H), NA: M.OPT3.NA, wavelength: M.OPT3.wavelength, px: M.PX3 };
}

// ------------------------------------------------------------------ denoise/
{
  const t = testImage(), r = rng(17);   // the demo's seed
  const noisy = t.map(v => v + 10 * r.norm());
  const g = DN.gaussianDenoise(noisy, N, N, 3), bl = DN.bilateral(noisy, N, N, 2, 30), nl = DN.nonLocalMeans(noisy, N, N, 2, 5, 20);
  out.methodnoise = { noisy: arr(noisy, 2), gauss: arr(DN.methodNoise(noisy, g), 2), bilateral: arr(DN.methodNoise(noisy, bl), 2), nlm: arr(DN.methodNoise(noisy, nl), 2) };
  // measurements: the measure field
  const psf = makePSF(N, { kind: "airy", width: 1.0 });
  const truth = render(M.measureScene(), N), clean = blur(truth, psf, N), y = M.photonData(clean, 1000, 0, 21), A = D.fftOperator2d(psf, N, N);
  const snaps: Record<number, number[]> = {}, reads: Record<number, M.Readout> = {};
  let x = D.flatStart(y, 0);
  for (let k = 1; k <= 1000; k++) { x = D.rlStep(x, y, A, 0); if (k === 10 || k === 100 || k === 1000) { const e = x.map(v => v / 1000); snaps[k] = arr(e, 4); reads[k] = M.measure(e, N, M.BASE, M.COUNT_THR); } }
  const obs = y.map(v => v / 1000);
  out.measure = { truth: arr(truth, 4), obs: arr(obs, 4), snaps, reads, readRaw: M.measure(obs, N, M.BASE, M.COUNT_THR) };
}

fs.writeFileSync(process.argv[2], JSON.stringify(out));
console.log("wrote", process.argv[2]);
