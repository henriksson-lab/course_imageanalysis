// Numbers and maps for the second batch of optical-flow figures (figs2.py), computed with
// the course's own demo code (demos/src/lib/flow.ts), with the same scenes, seeds and
// parameters as the lkflow and hsflow demos, so the figures match the demos and the slides.
//
// Bundle and run from the repository root:
//   node_modules/.bin/esbuild tools/figs/opticalflow/compute.ts --bundle --platform=node --outfile=<tmp>/compute.js
//   python3 -I -c "from PIL import Image; open('<tmp>/cell.raw','wb').write(Image.open('demos/src/fourier2d/cell256.png').convert('L').tobytes())"
//   node <tmp>/compute.js <tmp>/cell.raw <tmp>/flow.json
import { readFileSync, writeFileSync } from "fs";
import { Grey, cropGrey, sampleBilinear } from "../../../demos/src/lib/imageops";
import { remap, Remap, ssdNcc, mutualInformation } from "../../../demos/src/lib/deform";
import { rng } from "../../../demos/src/lib/random";
import {
  makePair, lkPyramid, lucasKanade, endpointError, ringScene, cellScene, compositePair,
  HornSchunck, warpResidual, interiorMean, Flow,
} from "../../../demos/src/lib/flow";

const [rawPath, outPath] = process.argv.slice(2);
const raw = readFileSync(rawPath);
const cell: Grey = { width: 256, height: 256, data: Float32Array.from(raw) };
const N = 128;
const crop = cropGrey(cell, 64, 64, N, N);
const arr = (a: ArrayLike<number>) => Array.from(a, v => Math.round(v * 1000) / 1000);
const out: Record<string, unknown> = {};

// lkflow (mean EPE 8 px from the border, as in its readout): cell scene, translation at 30°, noise 2, seed 11, σ = 1, ρ = 3, one level, one solve
const shiftPair = (s: number, noise: number) => makePair(crop, "translation", { s, angle: 30, r: 34 }, noise, 11);
out.shifts = [0.5, 1, 1.5, 2, 3, 4, 6, 8].map(s => {
  const p = shiftPair(s, 2);
  const one = endpointError(lkPyramid(p.I1, p.I2, { sigma: 1, rho: 3, levels: 1, iters: 1 }), p.truth, 8).mean;
  const pyr = endpointError(lkPyramid(p.I1, p.I2, { sigma: 1, rho: 3, levels: 4, iters: 3 }), p.truth, 8).mean;
  return { s, one, pyr };
});

// window size: noise 12 on the micrograph (1 px shift); motion boundary scene (1.5 px, noise 2)
const pn = shiftPair(1, 12);
const sc = cellScene(N, N, 5);
const pb = compositePair(sc.bg, sc.fg, sc.r, 1.5 * Math.cos(Math.PI / 6), -1.5 * Math.sin(Math.PI / 6), 2, 11);
// pixels within 6 px of the cell border
const c = (N - 1) / 2, near: number[] = [];
for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (Math.abs(Math.hypot(x - c, y - c) - sc.r) <= 6) near.push(y * N + x);
out.rho = [1, 1.5, 2, 3, 4, 5, 6, 8].map(rho => {
  const en = endpointError(lkPyramid(pn.I1, pn.I2, { sigma: 1, rho, levels: 1, iters: 1 }), pn.truth, 8).mean;
  const eb = endpointError(lkPyramid(pb.I1, pb.I2, { sigma: 1, rho, levels: 1, iters: 1 }), pb.truth, 0);
  let s = 0; for (const i of near) s += eb.map[i];
  return { rho, noise12: en, boundary: s / near.length };
});

// confidence: micrograph, 1.5 px shift, noise 2, ρ = 2 (the "membrane" preset)
{
  const p = shiftPair(1.5, 2);
  const r = lkPyramid(p.I1, p.I2, { sigma: 1, rho: 2, levels: 1, iters: 1 });
  const e = endpointError(r, p.truth, 0);
  const i = 114 * N + 85;
  out.conf = { I1: arr(p.I1.data), mu2: arr(r.mu2), u: arr(r.u), v: arr(r.v), epe: arr(e.map), tu: p.truth.u[0], tv: p.truth.v[0],
    membrane: { mu1: r.mu1[i], mu2: r.mu2[i], u: r.u[i], v: r.v[i] } };
}

// hsflow: ring scene, 1.5 px at 30°, noise 2, seed 11; LK σ = 1, ρ = 3; HS λ = 100
{
  const ring = ringScene(N, N, 3);
  const p = makePair(ring, "translation", { s: 1.5, angle: 30 }, 2, 11);
  const lk = lucasKanade(p.I1, p.I2, { sigma: 1, rho: 3 });
  const hs = new HornSchunck(p.I1, p.I2, { lambda: 100, sigma: 1 });
  hs.step(60);
  const at60 = { u: arr(hs.u), v: arr(hs.v), epe: arr(endpointError(hs, p.truth, 0).map), meanEpe: endpointError(hs, p.truth, 8).mean };
  const r60 = warpResidual(p.I1, p.I2, hs);
  const absr = Float32Array.from(r60.data, Math.abs);
  // mean |residual| in the flat centre (radius < 0.2 N) and overall (margin 8)
  let sc0 = 0, nc = 0, se = 0, ne = 0;
  for (let y = 8; y < N - 8; y++) for (let x = 8; x < N - 8; x++) {
    const i = y * N + x, d = Math.hypot(x - c, y - c);
    if (d < 0.2 * N) { sc0 += absr[i]; nc++; se += hs.u.length && Math.hypot(hs.u[i] - p.truth.u[i], hs.v[i] - p.truth.v[i]); ne++; }
  }
  hs.step(2000 - 60);
  out.ring = {
    I1: arr(p.I1.data), tu: p.truth.u[0], tv: p.truth.v[0],
    lk: { u: arr(lk.u), v: arr(lk.v), mu2: arr(lk.mu2), meanEpe: endpointError(lk, p.truth, 8).mean },
    hs2000: { u: arr(hs.u), v: arr(hs.v), meanEpe: endpointError(hs, p.truth, 8).mean },
    hs60: { ...at60, resid: arr(r60.data), residMean: interiorMean(absr, N, N, 8), residCentre: sc0 / nc, epeCentre: se / ne },
  };
}

// jointhist: fixed = 96² crop at (80, 80) of the micrograph (0..1); moving = remapped copy + noise 0.03
// (seed 9), shifted in x only; 16 bins; SSD (negated), NCC and MI over the overlap
{
  const JN = 96, X0 = 80, Y0 = 80, F = new Float64Array(JN * JN);
  for (let y = 0; y < JN; y++) for (let x = 0; x < JN; x++) F[y * JN + x] = cell.data[(Y0 + y) * 256 + X0 + x] / 255;
  const curves: Record<string, unknown> = {};
  for (const map of ["identity", "inverted", "nonmonotone"] as Remap[]) {
    const R = rng(9), M = new Float64Array(JN * JN);
    for (let i = 0; i < JN * JN; i++) M[i] = Math.max(0, Math.min(1, remap(F[i], map) + 0.03 * R.norm()));
    const xs: number[] = [], ss: number[] = [], nc: number[] = [], mi: number[] = [];
    for (let t = -84; t <= 84; t++) {
      const vals = new Float64Array(JN * JN), mask = new Uint8Array(JN * JN);
      for (let y = 0; y < JN; y++) for (let x = 0; x < JN; x++) {
        const mx = x - t; if (mx < 0 || mx > JN - 1) continue;
        mask[y * JN + x] = 1; vals[y * JN + x] = sampleBilinear({ width: JN, height: JN, data: M } as unknown as Grey, mx, y);
      }
      const a = ssdNcc(F, vals, mask), b = mutualInformation(F, vals, 16, mask);
      xs.push(t); ss.push(-a.ssd); nc.push(a.ncc); mi.push(b.mi);
    }
    curves[map] = { xs, ss, nc, mi };
  }
  out.jointhist = curves;
}

writeFileSync(outPath, JSON.stringify(out));
const brief = JSON.parse(JSON.stringify(out, (k, v) => (Array.isArray(v) && v.length > 100 ? `[${v.length}]` : v)));
console.log(JSON.stringify(brief, null, 1));
