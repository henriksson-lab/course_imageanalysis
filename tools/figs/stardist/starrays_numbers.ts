// Readouts of the starrays demo in its presets (for the stardist/ lecture notes and exercise
// solutions): run the same model code as the demo and print what its readout shows.
//   node_modules/.bin/esbuild tools/figs/stardist/starrays_numbers.ts --bundle --platform=node --log-level=warning | node
import * as G from "../../../demos/src/starrays/geom";
import * as S from "../../../demos/src/starrays/scene";
import { SHAPES } from "../../../demos/src/starrays/shapes";
import { rasterise } from "../../../demos/src/lib/shapecanvas";

const W = 64, mask = (id: string) => rasterise(SHAPES.find(s => s.id === id)!.polys(), W, W);
const f2 = (v: number) => v.toFixed(2), f3 = (v: number) => v.toFixed(3);
console.log("== shape mode");
for (const [id, cx, cy, n] of [["rect", 32, 32, 8], ["rect", 32, 32, 16], ["rect", 32, 32, 32], ["L", 19.5, 12.5, 32], ["star", 32, 32, 5], ["star", 32, 32, 10], ["disc", 32, 32, 8], ["disc", 32, 32, 32], ["ellipse", 32, 32, 8], ["ellipse", 32, 32, 32]] as [string, number, number, number][]) {
  const m = mask(id), f = G.fitAt(m, W, W, cx, cy, n);
  let area = 0; for (const v of m) area += v;
  console.log(`${id} n=${n} centre (${cx},${cy}): shape ${area} px, polygon ${f.pix.length} px (triangle rule ${G.polyArea(f.rays).toFixed(1)}), IoU ${f3(f.iou)}, missed ${f.missed}, extra ${f.extra}` + (n <= 8 ? `, rays ${Array.from(f.rays, r => r.toFixed(1)).join(" ")}` : ""));
}
for (const id of ["disc", "ellipse", "rect", "L", "C", "ring", "rod", "star"]) for (const n of [8, 32]) {
  const b = G.bestCentre(mask(id), W, W, n)!;
  console.log(`best centre ${id} n=${n}: (${b.cx}, ${b.cy}) IoU ${f3(b.iou)}`);
}
console.log("== decode mode (defaults noise 0.3, contact 0.2, 32 rays)");
const sc = S.makeScene();
for (const [label, p, t, n, noise, contact] of [["default", 0.5, 0.3, 32, 0.3, 0.2], ["strong", 0.5, 0.02, 32, 0.3, 0.2], ["weak", 0.5, 0.8, 32, 0.3, 0.2], ["low", 0.3, 0.3, 32, 0.3, 0.2], ["high", 0.7, 0.3, 32, 0.3, 0.2],
  ["B4a", 0.5, 0.3, 32, 0.3, 0.2], ["B4b", 0.5, 0.5, 32, 0.3, 0.2], ["B4c", 0.3, 0.7, 32, 0.3, 0.2], ["n=8", 0.5, 0.3, 8, 0.3, 0.2], ["n=4", 0.5, 0.3, 4, 0.3, 0.2], ["noise 0", 0.5, 0.3, 32, 0, 0.2], ["noise 1", 0.5, 0.3, 32, 1, 0.2], ["contact 0", 0.5, 0.02, 32, 0.3, 0], ["contact 0.5", 0.5, 0.3, 32, 0.3, 0.5], ["noise 0.6 nms 0.7", 0.5, 0.7, 32, 0.6, 0.2]] as [string, number, number, number, number, number][]) {
  const sim = S.simulate(n, noise, contact), d = S.decode(sim, p, t);
  const s = d.scores;
  console.log(`${label}: prob ${p} nms ${t} n ${n} noise ${noise} contact ${contact}: ${d.above.length} candidates, ${d.kept.length} kept, IoUs ${d.nms.iouComputed}; TP ${s.tp} FP ${s.fp} FN ${s.fn} F1 ${f2(s.f1)}; missed ${d.missed.join(",")}`);
  if (label === "default" || label === "strong") d.nms.steps.forEach((st, k) => {
    const ious = st.removed.map(r => r.iou);
    const own = (i: number) => { const c = sim.cands[d.above[i]]; return sc.objects[Math.floor(c.cy) * S.SN + Math.floor(c.cx)]; };
    const nb = st.removed.filter(r => own(r.j) !== own(st.kept)).map(r => `${own(r.j)}:${f2(r.iou)}`);
    console.log(`   step ${k + 1}: kept obj ${own(st.kept)} prob ${f2(sim.cands[d.above[st.kept]].score)}, deleted ${st.removed.length}` + (ious.length ? ` IoU ${f2(Math.min(...ious))}-${f2(Math.max(...ious))}` : "") + (nb.length ? ` neighbours ${nb.join(" ")}` : ""));
  });
}

// ---- instancemaps, preset "cshape" (exercise E8): the star route on the mixed scene --------
import * as IM from "../../../demos/src/instancemaps/model";
import { iouTable, match, objectScores, splitsMerges } from "../../../demos/src/lib/eval";
{
  console.log("== instancemaps cshape (star route, threshold 0.6)");
  const sc = IM.makeScene("mixed"), T = IM.computeTargets(sc);
  const pr = IM.predict(sc, T, "star", { noise: 0, blur: 0, contact: 0, missed: 0 });
  for (const [n, t] of [[32, 0.3], [8, 0.3], [64, 0.3], [32, 0.1], [32, 0.6], [32, 0.9]]) {
    const res = IM.postprocess(pr, { thr: 0.6, h: 0.15, nRays: n, nmsT: t, steps: 40 });
    const tb = iouTable(sc.labels, res.labels), s = objectScores(match(tb, 0.5)), sm = splitsMerges(tb);
    console.log(`rays ${n} NMS ${t}: ${res.count} objects, TP ${s.tp} FP ${s.fp} FN ${s.fn} F1 ${s.f1.toFixed(2)}, splits ${sm.splits.length} merges ${sm.merges.length}`);
  }
}
