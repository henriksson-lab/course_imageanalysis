// Data for the extra slide figures of forest/ (the forest playground's own data, settings
// and seed, so the numbers match the demo). Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/forest/data.ts
// then python3 -I tools/figs/forest/figs.py <dir>. Numbers are printed.
import * as fs from "fs";
import * as os from "os";
import * as FO from "../../../demos/src/lib/forest";
import { PRESETS, makeData, featureTable } from "../../../demos/src/forestplay/data";

const TMP = process.env.FIGTMP || os.tmpdir();
const RES = 120;
const gx: number[] = [], gy: number[] = [];
for (let j = 0; j < RES; j++) for (let i = 0; i < RES; i++) { gx.push((i + 0.5) / RES); gy.push((j + 0.5) / RES); }

function run(id: string) {
  const p = PRESETS.find(q => q.id === id)!;
  const d = makeData(p.data, false), t = makeData(p.data, true);
  const F = { extra: p.extra, noise: p.noise };
  const T = featureTable(d.points.map(q => q.x), d.points.map(q => q.y), F, 1);
  const y = d.points.map(q => q.c), k = Math.max(2, ...y.map(c => c + 1));
  const f = FO.trainForest(FO.binTable(T, 1024), FO.classTarget(y, k), null, {
    nTrees: p.B, mtry: p.m, maxDepth: p.depth, minLeaf: p.minLeaf, bootstrap: p.bootstrap, seed: 7,
  });
  const G = featureTable(gx, gy, F, 2);
  const P = FO.predictForest(f, G.X, G.n);
  const Tt = featureTable(t.points.map(q => q.x), t.points.map(q => q.y), F, 3), yt = t.points.map(q => q.c);
  const tc = FO.testErrorCurve(f, Tt.X, Tt.n, yt);
  const oc = p.bootstrap && p.B > 1 ? FO.oobErrorCurve(f, T.X, y) : null;
  // the first trees' own maps (leaf proportions of class 1)
  const trees = f.trees.slice(0, 3).map(tr => Array.from({ length: G.n }, (_, g) => Math.round(tr.value[FO.leafOfRow(tr, G.X, G.n, g) * k + 1] * 100) / 100));
  console.log(`${id}: test error ${(100 * tc[p.B - 1]).toFixed(1)} %` + (oc ? `, OOB ${(100 * oc[p.B - 1]).toFixed(1)} %` : ""));
  return {
    points: d.points.map(q => [q.x, q.y, q.c]), k, P: Array.from(P, v => Math.round(v * 1000) / 1000),
    test: Array.from(tc), oob: oc ? Array.from(oc, v => (Number.isFinite(v) ? v : null)) : null, trees,
    inbag: f.inbag.slice(0, 3).map(a => Array.from(a)),
  };
}

const out = { RES, single: run("single"), bagged: run("bagged"), boundary: run("boundary") };
// test error of single trees grown on bootstrap samples (the bagged forest's first trees)
fs.writeFileSync(TMP + "/forest.json", JSON.stringify(out));
console.log("wrote", TMP + "/forest.json");
