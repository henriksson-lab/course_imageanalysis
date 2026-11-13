// Data for the sfm/ slide figures, computed with the demos' own flatland code
// (demos/src/bundleadjust/model.ts, demos/src/lib/bundle.ts, demos/src/lib/flatland.ts) and the
// demo presets' settings. Writes sfm.json for figs.py. Build and run from the repo root:
//   node_modules/.bin/esbuild tools/figs/sfm/compute.ts --bundle --platform=node --outfile=<tmp>/f.js && node <tmp>/f.js <tmp>
import * as fs from "fs";
import { makeWorld, perturbed, gaugeFor, BAWorld } from "../../../demos/src/bundleadjust/model";
import { FlatBA, lmStep, pointCovariances } from "../../../demos/src/lib/bundle";
import { makeScene, ring, render1d, sweepPixel, noise1d, rayDir, P, Cam1D } from "../../../demos/src/lib/flatland";

const out = process.argv[2] ?? ".";
const res: Record<string, unknown> = {};
const r3 = (v: number) => +v.toFixed(4);
const camJ = (c: Cam1D) => ({ x: r3(c.x), y: r3(c.y), theta: r3(c.theta) });

function state(w: BAWorld, prob: FlatBA, th: Float64Array) {
  const cams = prob.cams(th), pts = prob.pts(th);
  const rays = w.obs.map(o => { const c = cams[o.cam], d = rayDir(c, o.u), p = pts[o.pt], L = Math.hypot(p[0] - c.x, p[1] - c.y) + 0.25; return [r3(c.x), r3(c.y), r3(c.x + L * d[0]), r3(c.y + L * d[1]), o.cam]; });
  return { cams: cams.map(camJ), pts: pts.map(p => p.map(r3)), rays };
}
function rms(prob: FlatBA, th: Float64Array): number { const r = prob.evaluate(th, false).r; return Math.sqrt(r.reduce((a, b) => a + b * b, 0) / r.length); }

// default world (10 cameras, 40 points); "large" preset: perturbation 0.6, seed 10; GN vs LM (μ starts at 1e-2)
{
  const w = makeWorld(), prob = new FlatBA(w.cams, w.pts.length, w.obs), fixed = gaugeFor(w);
  const truth = FlatBA.pack(w.cams, w.pts);
  res.truth = { ...state(w, prob, truth), outline: w.scene.polys[0].map(p => p.map(r3)) };
  for (const [name, s, seed] of [["large", 0.6, 10], ["small", 0.05, 5]] as [string, number, number][]) {
    const th0 = perturbed(w, s, fixed, seed);
    if (name === "large") res.start = state(w, prob, th0);
    const hist: Record<string, number[]> = {};
    for (const lm of [false, true]) {
      let th = Float64Array.from(th0), mu = 1e-2;
      const h = [rms(prob, th)];
      for (let k = 0; k < (lm ? 14 : 4); k++) { const st = lmStep(prob, th, { mu, lm, fixed }); th = st.theta; mu = lm ? st.mu : mu; h.push(rms(prob, th)); }
      hist[lm ? "LM" : "GN"] = h.map(v => +v.toPrecision(4));
      if (lm && name === "large") res.lmFinal = state(w, prob, th);
    }
    res[name + "Hist"] = hist;
    console.log(name, JSON.stringify(hist));
  }
  const cov = pointCovariances(prob, truth, fixed, 0.2);
  res.covRing = cov.map(c => (c ? c.map(v => +v.toExponential(4)) : null));
}
// "far" preset: 6 cameras on a 115° arc, radius 4
{
  const w = makeWorld({ nCams: 6, arc: 2.0, radius: 4 }), prob = new FlatBA(w.cams, w.pts.length, w.obs);
  const truth = FlatBA.pack(w.cams, w.pts);
  res.far = { ...state(w, prob, truth), cov: pointCovariances(prob, truth, gaugeFor(w), 0.2).map(c => (c ? c.map(v => +v.toExponential(4)) : null)) };
}
// plane sweep (flatland demo, 3 views, window ±3): textured pixel 60 and textureless pixel 66
{
  const OBJECT: P[] = [[-0.9, -0.15], [-0.2, -0.65], [0.75, -0.35], [0.85, 0.5], [-0.6, 0.55]];
  const DEPTHS = Array.from({ length: 90 }, (_, k) => 1.6 + k * 0.035);
  for (const [id, u] of [["textured", 60], ["textureless", 66]] as [string, number][]) {
    const scene = makeScene([OBJECT], (k, e) => (id === "textureless" && k === 0 && e === 1 ? () => 0.62 : null), 3);
    void noise1d;
    const cams = [-0.42, 0, 0.42].map(a => ring(1, 3.1, 150, 120, [0, -0.05], -Math.PI / 2 + a, 0)[0]);
    const imgs = cams.map(c => render1d(scene, c).values);
    const truthDepth = render1d(scene, cams[1]).depth;
    const sw = sweepPixel(cams, imgs, 1, u, DEPTHS, 3);
    const all: number[] = [];
    for (let p = 0; p < 120; p++) { const s = sweepPixel(cams, imgs, 1, p, DEPTHS, 3); all.push(s.best >= 0 ? DEPTHS[s.best] : NaN); }
    let bad = 0, n = 0;
    all.forEach((z, p) => { if (Number.isFinite(truthDepth[p])) { n++; if (!(Math.abs(z - truthDepth[p]) <= 0.1)) bad++; } });
    res["sweep_" + id] = { u, depths: DEPTHS.map(r3), cost: sw.cost.map(v => (Number.isFinite(v) ? r3(v) : null)), best: DEPTHS[sw.best], truthU: truthDepth[u],
      all: all.map(v => (Number.isFinite(v) ? r3(v) : null)), truth: Array.from(truthDepth, v => (Number.isFinite(v) ? r3(v) : null)), bad, n, img: Array.from(imgs[1], r3) };
    console.log(id, "pixel", u, "best", DEPTHS[sw.best].toFixed(2), "true", truthDepth[u].toFixed(2), `${bad} of ${n} off`);
  }
}
fs.writeFileSync(`${out}/sfm.json`, JSON.stringify(res));
