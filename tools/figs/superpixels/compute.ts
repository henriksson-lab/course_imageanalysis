// Data for the figures of superpixels/, computed as in the slic demo (lib/superpixels.ts,
// lib/mst.ts, lib/watershed.ts, lib/graph.ts). Run from the repository root:
//   python3 -I tools/figs/random-walker/figs.py <rawdir> --export-cell
//   node_modules/.bin/esbuild tools/figs/superpixels/compute.ts --bundle --platform=node --log-level=warning > <rawdir>/sp.js && node <rawdir>/sp.js <rawdir>
//   python3 -I tools/figs/superpixels/figs.py <rawdir>
import * as SP from "../../../demos/src/lib/superpixels";
import * as M from "../../../demos/src/lib/mst";
import { gradient, fromArray } from "../../../demos/src/lib/imageops";
import { srgbToLab } from "../../../demos/src/lib/colour";
import { watershed } from "../../../demos/src/lib/watershed";
import { gridEdges, contrastWeights, solveSeg } from "../../../demos/src/lib/graph";
import { colourScene, fhScene } from "../../../demos/src/fhseg/scenes";

declare const process: { argv: string[] };
declare function require(m: string): { readFileSync(p: string, e: string): string; writeFileSync(p: string, d: string): void };
const fs = require("fs");
const raw = process.argv[2];
const N = 256;
const out: Record<string, unknown> = {};
const A = (a: ArrayLike<number>) => Array.from(a);

const cs = colourScene(6);
const lab: Float64Array[] = [new Float64Array(N * N), new Float64Array(N * N), new Float64Array(N * N)];
for (let p = 0; p < N * N; p++) { const l = srgbToLab([cs.rgb![0][p] / 255, cs.rgb![1][p] / 255, cs.rgb![2][p] / 255]); for (let j = 0; j < 3; j++) lab[j][p] = l[j]; }
const csGrad = Float64Array.from(gradient(fromArray(N, N, cs.I), 1).mag.data);
const csG = M.gridDissim(lab, N, N, 8), csOrd = M.sortEdges(csG);
// the grey test scene (texture, faint step, ramp, spots) for the method comparison and the curves
const gs = fhScene(4), gch = [Float64Array.from(gs.I, v => v * 100 / 255)];
const gGrad = Float64Array.from(gradient(fromArray(N, N, gs.I), 1).mag.data);
const gG = M.gridDissim(gch, N, N, 8), gOrd = M.sortEdges(gG);
out.rgb = cs.rgb!.map(c => A(c).map(Math.round));
out.truth = A(cs.truth);

function fhK(K: number, G = csG, ord = csOrd): Int32Array {
  let lo = 1, hi = 100000, best = new Int32Array(N * N);
  const min = Math.max(5, Math.floor(N * N / K / 10));
  for (let it = 0; it < 18; it++) {
    const k = Math.sqrt(lo * hi), r = M.felzenszwalb(G, k, min, ord);
    best = r.labels;
    if (Math.abs(r.count - K) <= Math.max(2, 0.05 * K)) break;
    if (r.count > K) lo = k; else hi = k;
  }
  return best;
}
const wsK = (K: number, g = csGrad) => SP.relabel(watershed(g, N, N, SP.gridMarkers(N, N, K), { conn: 8 })).labels;

// iterations, grid and window
{
  const s = SP.slic(lab, N, N, { K: 100, m: 20, iters: 10, trace: true, connectivity: false });
  out.iters = { labels: [0, 1, 3, 10].map(i => A(s.traceLabels![i])), centres: [0, 1, 3, 10].map(i => s.traceCentres![i].map(c => [c[3], c[4]])), S: s.S, objective: s.objective };
  console.log("objective", s.objective.map(v => v.toFixed(0)).join(" "));
}
// compactness
{
  const res: Record<string, unknown> = {};
  for (const m of [5, 20, 80]) { const s = SP.slic(lab, N, N, { K: 200, m, iters: 10 }); res[m] = { labels: A(s.labels), count: s.count, br: SP.boundaryRecall(s.labels, cs.truth, N, N, 2), ue: SP.undersegmentationError(s.labels, cs.truth) }; }
  out.compact = res;
  console.log("compact", Object.entries(res).map(([k, v]) => `${k}: ${JSON.stringify({ ...(v as object), labels: undefined })}`).join(" | "));
}
// orphans
{
  const a = SP.slic(lab, N, N, { K: 200, m: 6, iters: 10, connectivity: false }), b = SP.slic(lab, N, N, { K: 200, m: 6, iters: 10 });
  let pieces = 0;
  { // count 4-connected pieces of a
    const seen = new Uint8Array(N * N);
    for (let s = 0; s < N * N; s++) { if (seen[s]) continue; pieces++; const st = [s]; seen[s] = 1; while (st.length) { const p = st.pop()!; const x = p % N; for (const q of [x > 0 ? p - 1 : -1, x < N - 1 ? p + 1 : -1, p - N, p + N]) if (q >= 0 && q < N * N && !seen[q] && a.labels[q] === a.labels[p]) { seen[q] = 1; st.push(q); } } }
  }
  out.orphans = { before: A(a.labels), after: A(b.labels), labelsBefore: a.count, pieces, after_count: b.count, orphanPx: b.orphans };
  console.log("orphans", a.count, "labels in", pieces, "pieces; after", b.count, "orphan px", b.orphans);
}
// three methods at about K = 200 on the grey test scene, and metrics against K
{
  const s = SP.slic(gch, N, N, { K: 200, m: 20, iters: 10 }).labels, f = fhK(200, gG, gOrd), w = wsK(200, gGrad);
  const met = (l: Int32Array) => [SP.boundaryRecall(l, gs.truth, N, N, 2), SP.undersegmentationError(l, gs.truth), SP.achievableAccuracy(l, gs.truth), Math.max(...l)];
  out.methods = { I: A(gs.I).map(Math.round), slic: A(s), fh: A(f), ws: A(w), metrics: { slic: met(s), fh: met(f), ws: met(w) } };
  console.log("methods", JSON.stringify((out.methods as { metrics: unknown }).metrics));
  const Ks = [25, 50, 100, 200, 400, 800], curves: Record<string, number[][]> = { slic: [], fh: [], ws: [] };
  for (const K of Ks) {
    for (const [name, l] of [["slic", SP.slic(gch, N, N, { K, m: 20, iters: 10 }).labels], ["fh", fhK(K, gG, gOrd)], ["ws", wsK(K, gGrad)]] as [string, Int32Array][]) curves[name].push([Math.max(...l), ...met(l).slice(0, 3)]);
  }
  out.curves = curves;
  console.log("curves", JSON.stringify(curves));
}
// graph cut on pixels and on superpixels (seeds as in the slic demo, colour scene)
{
  const seeds = new Int8Array(N * N);
  const disc = (cx: number, cy: number, r: number, v: number) => { for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) if (x * x + y * y <= r * r) { const xx = cx + x, yy = cy + y; if (xx >= 0 && yy >= 0 && xx < N && yy < N) seeds[yy * N + xx] = v; } };
  disc(75, 80, 14, 1); disc(10, 10, 6, -1); disc(128, 128, 6, -1); disc(250, 128, 6, -1); disc(75, 140, 5, -1); disc(240, 245, 6, -1);
  const sigma = 15;
  let t = performance.now();
  const E = gridEdges(N, N, 4);
  const pix = solveSeg({ n: N * N, D0: new Float64Array(N * N), D1: new Float64Array(N * N), E, w: contrastWeights(cs.I, E, sigma), lambda: 1, seeds }).L;
  const tPix = performance.now() - t;
  t = performance.now();
  const sp = SP.slic(lab, N, N, { K: 400, m: 20, iters: 10 }).labels;
  const tSlic = performance.now() - t;
  t = performance.now();
  const g = SP.spGraph(sp, N, N, cs.I);
  const sps = new Int8Array(g.n);
  for (let p = 0; p < N * N; p++) if (seeds[p]) sps[sp[p] - 1] = seeds[p];
  const w = new Float64Array(g.a.length);
  for (let e = 0; e < w.length; e++) w[e] = g.len[e] * Math.exp(-((g.mean[g.a[e]] - g.mean[g.b[e]]) ** 2) / (2 * sigma * sigma));
  const L = solveSeg({ n: g.n, D0: new Float64Array(g.n), D1: new Float64Array(g.n), E: { p: g.a, q: g.b, geo: new Float64Array(g.a.length).fill(1) }, w, lambda: 1, seeds: sps }).L;
  const tSp = performance.now() - t;
  const spL = Uint8Array.from(sp, l => L[l - 1]);
  let both = 0, onlyP = 0, onlyS = 0;
  for (let p = 0; p < N * N; p++) { if (pix[p] && spL[p]) both++; else if (pix[p]) onlyP++; else if (spL[p]) onlyS++; }
  out.gc = { pix: A(pix), sp: A(spL), spLabels: A(sp), seeds: A(seeds), tPix, tSlic, tSp, nodes: g.n, edges: g.a.length, pixEdges: E.p.length };
  console.log("gc", { tPix, tSlic, tSp, nodes: g.n, edges: g.a.length, both, onlyP, onlyS });
}
// thin structure lost: grey scene, K = 60 and 400
{
  const s = fhScene(4), ch = [Float64Array.from(s.I, v => v * 100 / 255)];
  const a = SP.slic(ch, N, N, { K: 60, m: 20, iters: 10 }), b = SP.slic(ch, N, N, { K: 400, m: 20, iters: 10 });
  out.thin = { I: A(s.I).map(Math.round), k60: A(a.labels), k400: A(b.labels), br60: SP.boundaryRecall(a.labels, s.truth, N, N, 2), br400: SP.boundaryRecall(b.labels, s.truth, N, N, 2), asa60: SP.achievableAccuracy(a.labels, s.truth), asa400: SP.achievableAccuracy(b.labels, s.truth) };
  console.log("thin", (out.thin as { br60: number }).br60, (out.thin as { br400: number }).br400, (out.thin as { asa60: number }).asa60, (out.thin as { asa400: number }).asa400);
}
// the micrograph with 300 superpixels and their RAG
{
  const cell = Float64Array.from(JSON.parse(fs.readFileSync(`${raw}/cell256.json`, "utf8")) as number[]);
  const s = SP.slic([Float64Array.from(cell, v => v * 100 / 255)], N, N, { K: 300, m: 20, iters: 10 });
  const g = SP.spGraph(s.labels, N, N, cell);
  out.cell = { I: A(cell).map(Math.round), labels: A(s.labels), count: s.count, cx: A(g.cx), cy: A(g.cy), a: A(g.a), b: A(g.b) };
  console.log("cell", s.count, "superpixels", g.a.length, "edges");
}
fs.writeFileSync(`${raw}/sp.json`, JSON.stringify(out));
