// Data for the texture-features figures, from the glcm demo's own code (same patches, seeds,
// quantisation 0..255, symmetric GLCM). Run via figs.py; argument: output JSON path.
import * as fs from "fs";
import { makePatch, P } from "../../../demos/src/glcm/model";
import { textureFeatures, lbp3x3, lbpRotInv, lbpTransitions } from "../../../demos/src/lib/glcm";

const out: Record<string, unknown> = { P };
const tf = (img: Float64Array, L: number, d: number, angle = 0) => textureFeatures(img, P, P, { L, lo: 0, hi: 255, d, angle, symmetric: true });
// "same histogram" pair, 8 levels, d = 1
const A = makePatch("pairA", { seed: 1 }), B = makePatch("pairB", { seed: 1 });
const fa = tf(A, 8, 1), fb = tf(B, 8, 1);
out.pair = { A: Array.from(A), B: Array.from(B), pA: Array.from(fa.p), pB: Array.from(fb.p), fA: fa.f, fB: fb.f };
// correlation and contrast against d: smoothed noise sigma 4 (seed 1), white noise (seed 1)
const sm = makePatch("smooth", { seed: 1, corr: 4 }), wn = makePatch("noise", { seed: 1 });
const ds = Array.from({ length: 16 }, (_, k) => k + 1);
out.scale = { d: ds, smooth: Array.from(sm), noise: Array.from(wn),
  corrSmooth: ds.map(d => tf(sm, 8, d).f.correlation), corrNoise: ds.map(d => tf(wn, 8, d).f.correlation),
  conSmooth: ds.map(d => tf(sm, 8, d).f.contrast), conNoise: ds.map(d => tf(wn, 8, d).f.contrast) };
// stripes period 8: contrast and correlation against d
const st = makePatch("stripes", { period: 8, angle: 0 });
out.stripes = { img: Array.from(st), con: ds.map(d => tf(st, 8, d).f.contrast), corr: ds.map(d => tf(st, 8, d).f.correlation) };
// LBP worked example
const nb = [52, 70, 91, 60, 64, 88, 40, 66, 75];
const code = lbp3x3(nb);
out.lbp = { v: nb, code, rot: lbpRotInv(code), trans: lbpTransitions(code) };
fs.writeFileSync(process.argv[2], JSON.stringify(out));
