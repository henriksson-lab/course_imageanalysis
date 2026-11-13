// Data for the extra slide figures of dtree/ (the tree playground's own data and tree code,
// so the numbers match the demo). Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/dtree/data.ts
// then python3 -I tools/figs/dtree/figs.py <dir>. Numbers are printed.
import * as fs from "fs";
import * as os from "os";
import * as FO from "../../../demos/src/lib/forest";
import { dataset, Dataset2D } from "../../../demos/src/lib/datasets2d";
import { diagonal } from "../../../demos/src/treeplay/data";

const TMP = process.env.FIGTMP || os.tmpdir();

function table(d: Dataset2D) {
  const T = FO.tableFromRows(d.points.map(p => [p.x, p.y]));
  return { B: FO.binTable(T, 4096), target: FO.classTarget(d.points.map(p => p.c), 2) };
}
function grow(d: Dataset2D, o: FO.TreeOptions): FO.Tree {
  const { B, target } = table(d);
  return FO.growTree(B, target, null, { criterion: "gini", minLeaf: 1, ...o });
}
/** Leaf rectangles [x0, x1, y0, y1, P(class 1), count] of a 2-feature tree on [0, 1]². */
function rects(t: FO.Tree): number[][] {
  const out: number[][] = [];
  const rec = (q: number, b: number[]) => {
    if (t.feature[q] < 0) { out.push([...b, t.value[q * t.k + 1], t.count[q]]); return; }
    const f = t.feature[q], th = t.threshold[q];
    const L = b.slice(), R = b.slice();
    if (f === 0) { L[1] = th; R[0] = th; } else { L[3] = th; R[2] = th; }
    rec(t.left[q], L); rec(t.right[q], R);
  };
  rec(0, [-0.05, 1.05, -0.05, 1.05]);
  return out;
}
/** Split lines [x0, x1, y0, y1] of every internal node. */
function lines(t: FO.Tree): number[][] {
  const out: number[][] = [];
  const rec = (q: number, b: number[]) => {
    if (t.feature[q] < 0) return;
    const f = t.feature[q], th = t.threshold[q];
    out.push(f === 0 ? [th, th, b[2], b[3]] : [b[0], b[1], th, th]);
    const L = b.slice(), R = b.slice();
    if (f === 0) { L[1] = th; R[0] = th; } else { L[3] = th; R[2] = th; }
    rec(t.left[q], L); rec(t.right[q], R);
  };
  rec(0, [-0.05, 1.05, -0.05, 1.05]);
  return out;
}
function errorOn(t: FO.Tree, d: Dataset2D): number {
  let e = 0;
  for (const p of d.points) { const v = FO.predictPoint(t, [p.x, p.y]); if ((v[1] > v[0] ? 1 : 0) !== p.c) e++; }
  return e / d.points.length;
}
const pts = (d: Dataset2D) => d.points.map(p => [p.x, p.y, p.c]);
const out: Record<string, unknown> = {};

// two blobs (nearest-neighbour picture)
const blobs = dataset("blobs", { seed: 1, n: 100 });
out.blobs = pts(blobs);

// XOR: root split scan and the tree after three splits
const xor = dataset("xor", { seed: 5, n: 120 }), xorTest = dataset("xor", { seed: 1005, n: 400 });
{
  const { B, target } = table(xor);
  const idx = Int32Array.from({ length: B.n }, (_, i) => i), scan: FO.SplitScan[] = [];
  const s = FO.bestSplit(B, idx, 0, B.n, target, null, [0, 1], { criterion: "gini", scan });
  out.xorScan = scan.map(q => ({ feature: q.feature, t: Array.from(q.thresholds), g: Array.from(q.gains) }));
  const t3 = grow(xor, { maxSplits: 3 });
  console.log(`XOR: best root decrease ${s!.gain.toFixed(3)}; after 3 splits training error ${Math.round(errorOn(t3, xor) * 120)}/120, test ${(100 * errorOn(t3, xorTest)).toFixed(1)} %`);
  out.xor = pts(xor); out.xorRects = rects(t3); out.xorLines = lines(t3);
}

// noisy overlap: error against depth, and the partitions at depth 3 and fully grown
const ov = dataset("overlap", { seed: 4, n: 200 }), ovTest = dataset("overlap", { seed: 1004, n: 1000 });
{
  const tr: number[] = [], te: number[] = [];
  for (let D = 1; D <= 12; D++) { const t = grow(ov, { maxDepth: D }); tr.push(errorOn(t, ov)); te.push(errorOn(t, ovTest)); }
  console.log("overlap train:", tr.map(v => (100 * v).toFixed(1)).join(" "));
  console.log("overlap test: ", te.map(v => (100 * v).toFixed(1)).join(" "));
  out.depthTrain = tr; out.depthTest = te;
  out.overlap = pts(ov);
  out.ovRects3 = rects(grow(ov, { maxDepth: 3 })); out.ovRects12 = rects(grow(ov, { maxDepth: 12 }));
}

// diagonal: the staircase of a fully grown tree
{
  const dg = diagonal(150, 2), t = grow(dg, { maxDepth: 12 });
  const sh = FO.treeShape(t);
  console.log(`diagonal: ${sh.leaves} leaves, depth ${sh.depth}, test error ${(100 * errorOn(t, diagonal(600, 1002))).toFixed(1)} %`);
  out.diag = pts(dg); out.diagRects = rects(t); out.diagLeaves = sh.leaves;
}
fs.writeFileSync(TMP + "/dtree.json", JSON.stringify(out));
console.log("wrote", TMP + "/dtree.json");
