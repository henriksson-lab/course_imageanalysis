// Data for the figures of merge-trees/, computed with demos/src/lib/hierarchy.ts (as in the
// mergetree demo). Run from the repository root:
//   python3 -I tools/figs/random-walker/figs.py <rawdir> --export-cell
//   node_modules/.bin/esbuild tools/figs/merge-trees/compute.ts --bundle --platform=node --log-level=warning > <rawdir>/mt.js && node <rawdir>/mt.js <rawdir>
//   python3 -I tools/figs/merge-trees/figs.py <rawdir>
import * as H from "../../../demos/src/lib/hierarchy";
import { gradient, fromArray } from "../../../demos/src/lib/imageops";
import { hMinimaMarkers, watershed } from "../../../demos/src/lib/watershed";
import { iouTable, match } from "../../../demos/src/lib/eval";
import { nucleiScene, leakScene } from "../../../demos/src/fhseg/scenes";

declare const process: { argv: string[] };
declare function require(m: string): { readFileSync(p: string, e: string): string; writeFileSync(p: string, d: string): void };
const fs = require("fs");
const raw = process.argv[2];
const N = 256;
const out: Record<string, unknown> = {};

function basins(I: Float64Array, h: number) {
  const g = Float64Array.from(gradient(fromArray(N, N, I), 1.5).mag.data);
  const leaves = watershed(g, N, N, hMinimaMarkers(g, N, N, h, 8).data, { conn: 8 });
  return { g, leaves, R: H.rag(leaves, N, N, I, g) };
}
const cutRegions = (T: H.MergeTree, regions: number) => {
  const k = Math.max(0, Math.min(T.merges, T.n - regions));
  return H.cutAt(T, k === 0 ? -Infinity : T.level[T.n + k - 1]).labels;
};
const pix = (leaves: Int32Array, leafLab: Int32Array) => Int32Array.from(leaves, v => (v > 0 ? leafLab[v - 1] : 0));

// nuclei: dynamics hierarchy at three cuts, and the RAG of the basins
{
  const s = nucleiScene(6);
  const { leaves, R } = basins(s.I, 0.7);
  const T = H.attributeTree(R, "dynamics").tree;
  const cuts: Record<string, number[]> = {};
  const nt = Math.max(...s.truth);
  let info = "";
  for (const r of [60, 19, 8]) {
    const lab = pix(leaves, cutRegions(T, r));
    cuts[r] = Array.from(lab);
    info += ` ${r}: ${match(iouTable(s.truth, lab, nt, r), 0.5, "greedy").tp}/${nt}`;
  }
  const cx = new Float64Array(R.n), cy = new Float64Array(R.n);
  for (let p = 0; p < N * N; p++) { const l = leaves[p] - 1; if (l >= 0) { cx[l] += p % N; cy[l] += Math.floor(p / N); } }
  for (let i = 0; i < R.n; i++) { cx[i] /= R.size[i]; cy[i] /= R.size[i]; }
  out.nuclei = { I: Array.from(s.I, Math.round), leaves: Array.from(leaves), cuts, n: R.n, cx: Array.from(cx), cy: Array.from(cy), a: Array.from(R.a), b: Array.from(R.b), mean: Array.from(R.sum, (v, i) => v / R.size[i]) };
  console.log("nuclei leaves", R.n, "matched", info);
}
// micrograph: UCM (altitude) and two thresholds
{
  const cell = Float64Array.from(JSON.parse(fs.readFileSync(`${raw}/cell256.json`, "utf8")) as number[]);
  const { leaves, R } = basins(cell, 6);
  const T = H.attributeTree(R, "dynamics").tree;
  const rankT: H.MergeTree = { ...T, level: Float64Array.from(T.level, (_, v) => (v < T.n ? 0 : v - T.n + 1)) };
  const u = H.ucmImage(rankT, leaves, N, N);
  out.cell = { I: Array.from(cell, Math.round), ucm: Array.from(u), merges: T.merges, n: R.n, c40: Array.from(pix(leaves, cutRegions(T, 60))), c10: Array.from(pix(leaves, cutRegions(T, 20))) };
  console.log("cell leaves", R.n);
}
// leak scene: altitude vs mean difference at 2 regions
{
  const s = leakScene(5);
  const { leaves, R } = basins(s.I, 0.7);
  const Ta = H.attributeTree(R, "altitude").tree, Tm = H.meanDiffTree(R);
  out.leak = { I: Array.from(s.I, Math.round), alt: Array.from(pix(leaves, cutRegions(Ta, 2))), mean: Array.from(pix(leaves, cutRegions(Tm, 2))) };
}
fs.writeFileSync(`${raw}/mt.json`, JSON.stringify(out));
