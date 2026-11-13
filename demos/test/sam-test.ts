// Tests for the toy promptable segmenter of unit sam/ (demos/src/promptmask/): the scene and
// its truth, the encoder pieces, the decoder's three candidates on the preset prompts, negative
// clicks and boxes, the stability score as used in the exercise (B2, B6), and automatic masks.
// Run: node_modules/.bin/esbuild demos/test/sam-test.ts --bundle --platform=node --log-level=warning | node
import { makeScene, W } from "../src/promptmask/scene";
import * as M from "../src/promptmask/model";
import * as P from "../src/samnet/post";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

current = "scene";
const S = makeScene();
{
  const S2 = makeScene();
  check(S.img.every((v, i) => v === S2.img[i]), "fixed seed: identical images");
  check(S.img.every((v) => v >= 0 && v <= 255), "values in 0..255");
  check(S.nObjects === 11, "11 nuclei: " + S.nObjects);
  const ids = new Set<number>(); S.object.forEach((v) => { if (v) ids.add(v); });
  check(ids.size === 11, "every nucleus has truth pixels: " + ids.size);
  const groups = new Set<number>(); S.group.forEach((v) => { if (v) groups.add(v); });
  check(groups.size === 8, "8 groups (5 isolated, pair, cluster, cell): " + groups.size);
  // Nested truth: a nucleolus pixel is inside a nucleus, a nucleus pixel inside a group.
  let nested = true;
  for (let p = 0; p < W * W; p++) { if (S.part[p] && !S.object[p]) nested = false; if (S.object[p] && !S.group[p]) nested = false; }
  check(nested, "part ⊂ object ⊂ group");
  check(S.object[S.spots.pairRight[1] * W + S.spots.pairRight[0]] > 0, "the negative-click spot is inside a nucleus");
}

current = "encoder";
const E = M.encode(S.img, W, W);
{
  check(E.L.length === 3 && E.L.every((a) => a.length === W * W), "three smoothed copies");
  // Minimum filter against brute force on a small array.
  const a = Float32Array.from({ length: 7 * 5 }, (_, i) => Math.sin(i * 1.7) * 10);
  const m = M.minFilter(a, 7, 5, 1);
  let ok = true;
  for (let y = 0; y < 5; y++) for (let x = 0; x < 7; x++) {
    let b = Infinity;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const xx = x + dx, yy = y + dy; if (xx >= 0 && yy >= 0 && xx < 7 && yy < 5) b = Math.min(b, a[yy * 7 + xx]); }
    if (m[y * 7 + x] !== b) ok = false;
  }
  check(ok, "minFilter = brute-force 3 x 3 minimum");
  // Background below the nuclei.
  const [x, y] = S.spots.cluster, p = y * W + x;
  check(E.B[p] < E.L[0][p] - 80, `background under a nucleus is low: ${E.B[p].toFixed(0)} vs ${E.L[0][p].toFixed(0)}`);
}

current = "stability";
{
  // B2: 380 pixels with logit > 1, 430 with logit > -1 -> 0.884; 400 / 412 -> 0.971.
  const mk = (hi: number, lo: number) => { const l = new Float32Array(1000).fill(-5); l.fill(3, 0, hi); l.fill(0, hi, lo); return l; };
  check(close(M.stability(mk(380, 430), 1), 380 / 430, 1e-12), "380/430");
  check(close(M.stability(mk(400, 412), 1), 400 / 412, 1e-12), "400/412");
  check(M.stability(mk(380, 430), 1) < 0.95 && M.stability(mk(400, 412), 1) >= 0.95, "only the second passes 0.95");
  // B6: it is the IoU of the two nested masks, in [0, 1]; 1 when no logit lies in (−1, 1].
  const l = mk(380, 430);
  const a = P.threshold(l, 1), b = P.threshold(l, -1);
  check(close(P.maskIoU(a, b), M.stability(l, 1), 1e-12), "stability = IoU of the nested masks");
  check(M.stability(Float32Array.from([5, 5, -5, -3]), 1) === 1, "no logit near zero -> 1");
}

current = "keepPieces";
{
  // Two blobs of positive logits; a seed in the second keeps only that one; a seed 3 px off snaps.
  const w = 20, h = 10, l = new Float32Array(w * h).fill(-1);
  for (let y = 2; y < 6; y++) { for (let x = 2; x < 5; x++) l[y * w + x] = 2; for (let x = 12; x < 16; x++) l[y * w + x] = 2; }
  const m = M.keepPieces(l.slice(), w, h, [[13, 3]]);
  let a = 0; for (const v of m) a += v;
  check(a === 16 && m[3 * w + 3] === 0, "only the clicked piece: " + a);
  const s = M.keepPieces(l.slice(), w, h, [[8, 3]], 4);
  let a2 = 0; for (const v of s) a2 += v;
  check(a2 === 12, "a seed between pieces snaps to the nearer one: " + a2);
}

current = "decoder";
const iouAt = (c: M.CandidateMask, t: Int32Array, xy: [number, number]) => M.trueIoU(c.mask, t, W, xy[0], xy[1]);
{
  const edge = M.decode(E, [{ x: S.spots.edge[0], y: S.spots.edge[1], label: 1 }], null);
  check(edge.cands.length === 3, "three candidates");
  const [part, obj, grp] = edge.cands;
  check(part.area < obj.area && obj.area < grp.area, `edge: areas grow part < object < group: ${part.area} ${obj.area} ${grp.area}`);
  check(iouAt(obj, S.object, S.spots.edge) > 0.75, "edge: object candidate = the nucleus: " + iouAt(obj, S.object, S.spots.edge).toFixed(2));
  check(iouAt(grp, S.group, S.spots.edge) > 0.55, "edge: group candidate = the pair: " + iouAt(grp, S.group, S.spots.edge).toFixed(2));
  check(edge.best === 1, "edge: the highest score is the object candidate");
  edge.cands.forEach((c) => check(c.score >= 0 && c.score <= c.stability + 1e-12 && c.stability <= 1, "0 ≤ score (±2) ≤ stability (±1) ≤ 1"));

  const btw = M.decode(E, [{ x: S.spots.between[0], y: S.spots.between[1], label: 1 }], null);
  check(iouAt(btw.cands[1], S.object, S.spots.between) > 0.75, "between: object candidate is one of the two nuclei");
  check(iouAt(btw.cands[2], S.group, S.spots.between) > 0.55, "between: group candidate covers the pair");

  const neg = M.decode(E, [{ x: S.spots.between[0], y: S.spots.between[1], label: 1 }, { x: S.spots.pairRight[0], y: S.spots.pairRight[1], label: 0 }], null);
  const right = S.object[S.spots.pairRight[1] * W + S.spots.pairRight[0]];
  let inRight = 0; for (let p = 0; p < W * W; p++) if (neg.cands[2].mask[p] && S.object[p] === right) inRight++;
  let inRight0 = 0; for (let p = 0; p < W * W; p++) if (btw.cands[2].mask[p] && S.object[p] === right) inRight0++;
  check(neg.cands[2].area < btw.cands[2].area && inRight < 0.25 * inRight0, `negative click removes the neighbour from the group: ${inRight0} -> ${inRight} px`);
  check(neg.cands[1].mask[S.spots.pairRight[1] * W + S.spots.pairRight[0]] === 0, "the negative click itself is never in the mask");

  const bx = S.box;
  const box = M.decode(E, [], bx);
  let outside = 0;
  box.cands.forEach((c) => { for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) if (c.mask[y * W + x] && (x < bx[0] || x > bx[2] || y < bx[1] || y > bx[3])) outside++; });
  check(outside === 0, "box: nothing outside the box");
  const cxy: [number, number] = [(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2];
  check(iouAt(box.cands[box.best], S.object, cxy) > 0.65, "box: the best candidate is the nucleus: " + iouAt(box.cands[box.best], S.object, cxy).toFixed(2));

  const none = M.decode(E, [{ x: 10, y: 10, label: 0 }], null);
  check(none.cands.length === 0 && none.best === -1, "only a negative click: no candidates");
}

current = "automatic";
{
  const A = M.autoCandidates(E, 4);
  check(A.points === 16 && A.cands.length === 48, "4 x 4 grid -> 48 candidates");
  const A8 = M.autoCandidates(E, 8);
  const def = M.filterAuto(A8.cands, A8.points, M.AUTO_DEFAULT, S.object, W, W);
  const c = def.counts;
  check(c.iou + c.stability + c.empty + c.nms + c.kept === 192, "every candidate has one fate");
  check(def.scores !== null && def.scores.f1 > 0.45 && def.scores.f1 <= 1, "default filters: F1 " + def.scores?.f1.toFixed(3));
  const loose = M.filterAuto(A8.cands, A8.points, { score: 0.5, stability: 0.6, nms: 0.95 }, S.object, W, W);
  const strict = M.filterAuto(A8.cands, A8.points, { score: 0.9, stability: 0.95, nms: 0.7 }, S.object, W, W);
  check(loose.counts.kept > def.counts.kept && def.counts.kept > strict.counts.kept, `loose > default > strict: ${loose.counts.kept} ${def.counts.kept} ${strict.counts.kept}`);
  check(loose.scores!.fp > def.scores!.fp, "loose filters add false positives");
  console.log(`  automatic 8x8: default ${def.counts.kept} masks F1 ${def.scores!.f1.toFixed(2)}; strict ${strict.counts.kept} F1 ${strict.scores!.f1.toFixed(2)}; loose ${loose.counts.kept} F1 ${loose.scores!.f1.toFixed(2)}`);
}

console.log(`sam-test: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
