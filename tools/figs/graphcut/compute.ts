// Numbers and label maps for the graph-cut day's static figures, computed with the demos' own
// code (lib/graph.ts, lib/maxflow.ts, graphcut/model.ts, ncut/*). Run by figs.py, which
// bundles this file with esbuild for node and passes two arguments:
//   argv[2]  raw 256 x 256 grey bytes of demos/src/fourier2d/cell256.png
//   argv[3]  output JSON path
import { gridEdges, contrastWeights, solveSeg, thresholdLabels, gaussCosts, segEnergy, SegProblem } from "../../../demos/src/lib/graph";
import { edmondsKarpSteps, Edge } from "../../../demos/src/lib/maxflow";
import { PRESETS as MF } from "../../../demos/src/maxflow/graphs";
import * as GC from "../../../demos/src/graphcut/model";
import { POINT_PRESETS, pointGraph, imageGraph, diskImage, cellCrop } from "../../../demos/src/ncut/data";
import { fiedler, sweep, stoerWagner, WGraph } from "../../../demos/src/ncut/spectral";
declare const require: any, process: any;
const fs = require("fs");
const big: Uint8Array = fs.readFileSync(process.argv[2]);
const out: any = {};
const arr = (a: ArrayLike<number>) => Array.from(a, v => Math.round(v * 1000) / 1000);

// ---- maxflow: the textbook example, step by step
{
  const p = MF.find(q => q.id === "textbook")!;
  const r = edmondsKarpSteps(p.nodes.length, p.edges, 0, 1);
  out.maxflow = { nodes: p.nodes, edges: p.edges, steps: r.steps.map(s => ({ nodes: s.nodes, b: s.bottleneck, flow: s.flow, total: s.total })), flow: r.flow, total: r.total, S: r.S };
  // the back-edge example of the lecture slide (s a b t): s=0 t=1 a=2 b=3
  const E: Edge[] = [{ u: 0, v: 2, c: 1 }, { u: 0, v: 3, c: 1 }, { u: 2, v: 3, c: 1 }, { u: 2, v: 1, c: 1 }, { u: 3, v: 1, c: 1 }];
  out.backedge = edmondsKarpSteps(4, E, 0, 1).total;
}

// ---- seg-energy: the synthetic disk + arm, noise 50, fixed Gaussian model, no seeds
{
  const I = GC.addNoise(GC.synthImage(), 50);
  const truth = GC.synthTruth();
  const { D0, D1 } = gaussCosts(I, 80, 170, 50);
  const E = gridEdges(GC.N, GC.N, 8);
  const w = contrastWeights(I, E, Infinity);
  const res: any[] = [];
  for (const lambda of [0, 0.8, 5, 20]) {
    const P: SegProblem = { n: GC.N * GC.N, D0, D1, E, w, lambda };
    const L = lambda === 0 ? thresholdLabels(P) : solveSeg(P).L;
    let wrong = 0; for (let k = 0; k < L.length; k++) wrong += L[k] !== truth[k] ? 1 : 0;
    const en = segEnergy(P, L);
    res.push({ lambda, L: Array.from(L), wrong, data: en.data, boundary: en.boundary });
  }
  out.energy = { I: arr(I), res };
}

// ---- graphcut presets
{
  const cell = GC.downsample(big);
  const get = (id: string, patch: Partial<GC.Settings> = {}) => {
    const p = GC.PRESETS.find(q => q.id === id)!;
    const I = p.image === "synth" ? GC.addNoise(GC.synthImage(), p.noise) : GC.addNoise(cell, p.noise);
    const seeds = GC.rasterize(p.strokes);
    const st = { ...p.settings, ...patch };
    const r = GC.segment(I, seeds, st);
    const truth = GC.synthTruth();
    let wc = 0, wt = 0;
    for (let k = 0; k < truth.length; k++) { wc += r.cut.L[k] !== truth[k] ? 1 : 0; wt += r.thr[k] !== truth[k] ? 1 : 0; }
    return { id, settings: { ...st, sigma: Number.isFinite(st.sigma) ? st.sigma : null }, I: arr(I), seeds: Array.from(seeds), L: Array.from(r.cut.L), thr: Array.from(r.thr), wrongCut: wc, wrongThr: wt, info: r.info };
  };
  out.gc = {
    noise: get("noise"), shrink: get("shrink"), blocky4: get("blocky"), blocky8: get("blocky", { nbhd: 8 }),
    nucleus: get("nucleus"), badmodel: get("badmodel"),
  };
  // contrast weights on the micrograph: for each pixel the smaller of its right and down weights
  const E = gridEdges(GC.N, GC.N, 4);
  const res: any = {};
  for (const sigma of [10, 25]) {
    const w = contrastWeights(cell, E, sigma);
    const m = new Float64Array(GC.N * GC.N).fill(1);
    for (let k = 0; k < w.length; k++) { m[E.p[k]] = Math.min(m[E.p[k]], w[k]); }
    res[sigma] = arr(m);
  }
  out.contrast = { I: arr(cell), w: res };
}

// ---- ncut
{
  const two = POINT_PRESETS.find(p => p.id === "two")!;
  const pts = two.pts();
  const G = pointGraph(pts, two.sigma);
  const f = fiedler(G, true);
  const sw = sweep(G, f.x);
  out.two = { pts, x: arr(f.x), mu: f.mu, sweepNcut: arr(sw.ncut), order: sw.order, edges: G.p.map((p, k) => [p, G.q[k], G.w[k]]).filter(e => (e[2] as number) > 0.05) };
  const ol = POINT_PRESETS.find(p => p.id === "outlier")!;
  const pts2 = ol.pts();
  for (const sigma of [0.08, 0.15]) {
    const G2 = pointGraph(pts2, sigma);
    const f2 = fiedler(G2, true);
    const mc = stoerWagner(G2);
    const fu = fiedler(G2, false);
    const sideN = Array.from(f2.x, v => (v > 0 ? 1 : 0));
    const ncutOf = (side: ArrayLike<number>) => {
      let c = 0, va = 0, vb = 0;
      const d = new Float64Array(G2.n);
      for (let k = 0; k < G2.p.length; k++) { d[G2.p[k]] += G2.w[k]; d[G2.q[k]] += G2.w[k]; if (side[G2.p[k]] !== side[G2.q[k]]) c += G2.w[k]; }
      for (let i = 0; i < G2.n; i++) (side[i] ? (va += d[i]) : (vb += d[i]));
      return { cut: c, ncut: c / va + c / vb, nA: Array.from(side).filter(v => v).length };
    };
    out["outlier" + sigma] = { pts: pts2, x: arr(f2.x), xu: arr(fu.x), minCut: Array.from(mc.side), minCutVal: mc.value, ncutSide: sideN, minStats: ncutOf(mc.side), ncutStats: ncutOf(sideN) };
  }
  for (const [name, I] of [["disk", diskImage()], ["cell", cellCrop(big)]] as [string, Float64Array][]) {
    const G3 = imageGraph(I, 30);
    const f3 = fiedler(G3, true);
    out["img_" + name] = { I: arr(I), x: arr(f3.x), mu: f3.mu };
  }
  // the two triangles + pendant node of the slide (exercise B4); nodes 0..6
  const T: WGraph = { n: 7, p: [0, 0, 1, 3, 3, 4, 2, 5], q: [1, 2, 2, 4, 5, 5, 3, 6], w: [1, 1, 1, 1, 1, 1, 0.1, 0.05] };
  const ft = fiedler(T, true);
  out.tri = { x: arr(ft.x), mu: ft.mu, minCut: stoerWagner(T).value };
}
fs.writeFileSync(process.argv[3], JSON.stringify(out));
console.log("wrote", process.argv[3]);
