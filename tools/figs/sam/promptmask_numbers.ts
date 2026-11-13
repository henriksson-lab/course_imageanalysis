// Readouts of the toy demo promptmask for the presets used in sam/exercise.qmd (E7) and the
// lecture notes. Run:
//   node_modules/.bin/esbuild tools/figs/sam/promptmask_numbers.ts --bundle --platform=node --log-level=warning | node
import { makeScene, W } from "../../../demos/src/promptmask/scene";
import * as M from "../../../demos/src/promptmask/model";

const S = makeScene();
const E = M.encode(S.img, W, W);
console.log(`encoder ${E.ms.toFixed(0)} ms (node)`);
const show = (name: string, pts: M.Point[], box: M.Box | null) => {
  const d = M.decode(E, pts, box);
  const xy: [number, number] = pts.length ? [pts[0].x, pts[0].y] : [(box![0] + box![2]) / 2, (box![1] + box![3]) / 2];
  console.log(`${name}: decoder ${d.ms.toFixed(0)} ms; highest score = ${M.LEVELS[d.best]}`);
  for (const c of d.cands) {
    const t = [S.part, S.object, S.group].map((L) => M.trueIoU(c.mask, L, W, xy[0], xy[1]));
    console.log(`  ${c.level.padEnd(6)} area ${String(c.area).padStart(5)}  score ${c.score.toFixed(2)}  stability ${c.stability.toFixed(2)}  true IoU nucleolus ${t[0] < 0 ? "–" : t[0].toFixed(2)} nucleus ${t[1] < 0 ? "–" : t[1].toFixed(2)} group ${t[2] < 0 ? "–" : t[2].toFixed(2)}`);
  }
};
const sp = S.spots;
show("edge", [{ x: sp.edge[0], y: sp.edge[1], label: 1 }], null);
show("between", [{ x: sp.between[0], y: sp.between[1], label: 1 }], null);
show("negative", [{ x: sp.between[0], y: sp.between[1], label: 1 }, { x: sp.pairRight[0], y: sp.pairRight[1], label: 0 }], null);
show("box", [], S.box);
show("cluster nucleus", [{ x: sp.cluster[0], y: sp.cluster[1], label: 1 }], null);
show("cell nucleus", [{ x: sp.cellNucleus[0], y: sp.cellNucleus[1], label: 1 }], null);
show("cytoplasm", [{ x: sp.cytoplasm[0], y: sp.cytoplasm[1], label: 1 }], null);
console.log("spots", JSON.stringify(sp), "box", JSON.stringify(S.box));
for (const n of [8, 16]) {
  const A = M.autoCandidates(E, n);
  console.log(`automatic ${n}x${n}: decoder ${A.ms.toFixed(0)} ms`);
  for (const [k, f] of Object.entries({ default: M.AUTO_DEFAULT, strict: { score: 0.9, stability: 0.95, nms: 0.7 }, loose: { score: 0.5, stability: 0.6, nms: 0.95 } })) {
    const r = M.filterAuto(A.cands, A.points, f, S.object, W, W);
    const c = r.counts, s = r.scores!;
    const big = r.kept.filter((q) => q.area > 600).length;
    console.log(`  ${k.padEnd(7)} drops: score ${c.iou}, stability ${c.stability + c.empty}, nms ${c.nms} -> ${c.kept} masks (${big} larger than 600 px); TP ${s.tp} FP ${s.fp} FN ${s.fn} F1 ${s.f1.toFixed(2)}`);
  }
}
