// Tests for the yolodecode demo (demos/src/yolodecode/model.ts) and the by-hand numbers of the
// yolo/ unit (lecture, exercise B1–B3, B7, B8, B10):
//  - output sizes (1344, 5376, 8400 cells; × 144 numbers with 80 classes);
//  - the decoding of one cell (exercise B1: box (80, 72)–(144, 120), IoU 0.79, p = 0.77), a
//    two-peaked distribution (mean 4.8), and agreement of the whole decoding with the real
//    network's TypeScript decoding (yolonet/post.ts decode) on the simulated output;
//  - greedy NMS with classes (B3: class-aware keeps b1, b3, b4; agnostic b1, b3; same at 0.2)
//    and the counterexample of B10 (agnostic keeps a box that class-aware removes);
//  - GIoU of disjoint boxes (−0.2, −0.11), the distribution-loss minimiser (0.7 / 0.3 → 2.3);
//  - the simulated network: deterministic, and the outcomes the presets and exercises quote.
// Run: node_modules/.bin/esbuild demos/test/yolo-test.ts --bundle --platform=node --log-level=warning | node
import * as M from "../src/yolodecode/model";
import * as P from "../src/yolonet/post";
import { type Box, boxIou } from "../src/lib/boxes";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-6) => Math.abs(a - b) <= tol;
const bx = (x0: number, y0: number, x1: number, y1: number): Box => ({ x0, y0, x1, y1 });

current = "sizes";
{
  check(M.outputSize(256, 80).cells === 1344 && M.outputSize(512, 80).cells === 5376 && M.outputSize(640, 80).cells === 8400, "cells 256/512/640");
  check(M.outputSize(512, 80).numbers === 5376 * 144, "numbers per cell 4 × 16 + 80 = 144");
  check(M.NCELLS === 5376 && M.GRIDS.map((g) => g.g).join() === "64,32,16", "demo grids");
  const c = M.cellInfo(M.cellIndex(1, 5, 7));
  check(c.stride === 16 && c.cx === 120 && c.cy === 88, "stride 16, row 5, col 7 → centre (120, 88)");
  check([8, 16, 32].map((s) => 15 * s).join() === "120,240,480", "reach 15 strides");
}

current = "decode one cell (B1)";
{
  const sim = M.simulate(M.makeScene("mixed")), a = M.cellIndex(1, 5, 7);
  M.setSideProbs(sim, a, 0, M.massAt([[2, 0.5], [3, 0.5]]));
  M.setSideProbs(sim, a, 1, M.massAt([[1, 1]]));
  M.setSideProbs(sim, a, 2, M.massAt([[1, 0.5], [2, 0.5]]));
  M.setSideProbs(sim, a, 3, M.massAt([[2, 1]]));
  M.setClassLogit(sim, a, 0, 1.2);
  const dec = M.decodeAll(sim, "dist"), d = Array.from(dec.dist.subarray(a * 4, a * 4 + 4));
  check(d.every((v, k) => close(v, [2.5, 1, 1.5, 2][k], 1e-6)), `distances ${d}`);
  const b = dec.boxes[a];
  check(close(b.x0, 80, 1e-4) && close(b.y0, 72, 1e-4) && close(b.x1, 144, 1e-4) && close(b.y1, 120, 1e-4), `box ${JSON.stringify(b)}`);
  const truth = bx(84, 70, 150, 122), v = boxIou(b, truth);
  check(close(v, 2880 / 3624, 1e-6) && Math.round(v * 100) === 79, `IoU ${v} (2880 / 3624)`);
  check(close(dec.prob[a * M.NC], 0.7685, 1e-4), "sigmoid(1.2) = 0.77");
  check(close(M.sigmoid(-0.5), 0.3775, 1e-4) && close(M.sigmoid(-3), 0.0474, 1e-4), "sigmoid −0.5, −3");
  check(sim.scene.objs[0].box.x0 === 84 && sim.scene.objs[0].box.y1 === 122, "the scene's B1 cell has the true box of B1");
  // two peaks
  M.setSideProbs(sim, a, 2, M.massAt([[2, 0.6], [9, 0.4]]));
  check(close(M.decodeAll(sim, "dist").dist[a * 4 + 2], 4.8, 1e-6), "0.6 at 2 and 0.4 at 9 → 4.8");
  // setBin keeps a distribution
  const p = M.setBin(M.massAt([[2, 0.5], [3, 0.5]]), 7, 0.2);
  check(close(p.reduce((s, x) => s + x, 0), 1) && close(p[7], 0.2) && close(p[2], 0.4), "setBin");
}

current = "same decoding as the real network's";
{
  // Copy the simulated distance logits into the real head's layout (144 per cell, 80 classes) and
  // decode with yolonet/post.ts: the boxes must agree with the demo's decoding.
  const sim = M.simulate(M.makeScene("rods")), dec = M.decodeAll(sim, "dist");
  const raw = new Float32Array(M.NCELLS * P.NO);
  for (let a = 0; a < M.NCELLS; a++) {
    for (let k = 0; k < 4 * M.BINS; k++) raw[a * P.NO + k] = sim.raw[a * M.NO + k];
    for (let c = 0; c < M.NC; c++) raw[a * P.NO + 4 * M.BINS + c] = sim.raw[a * M.NO + 4 * M.BINS + c];
  }
  const ref = P.decode(raw, 512);
  let e = 0, ep = 0;
  for (let a = 0; a < M.NCELLS; a++) {
    const b = dec.boxes[a], o = a * P.ND;
    e = Math.max(e, Math.abs(b.x0 - ref[o]), Math.abs(b.y0 - ref[o + 1]), Math.abs(b.x1 - ref[o + 2]), Math.abs(b.y1 - ref[o + 3]));
    for (let c = 0; c < M.NC; c++) ep = Math.max(ep, Math.abs(dec.prob[a * M.NC + c] - ref[o + 8 + c]));
  }
  check(e < 1e-3 && ep < 1e-6, `boxes ${e}, probabilities ${ep}`);
}

current = "NMS with classes (B3, B10)";
{
  const fake = (boxes: Box[]): M.Decoded => ({ head: "dist", boxes, dist: new Float32Array(0), prob: new Float32Array(0) });
  const B = [bx(10, 10, 50, 50), bx(14, 12, 54, 52), bx(40, 10, 80, 50), bx(12, 10, 52, 50), bx(44, 14, 84, 54), bx(100, 100, 120, 120)];
  const sc = [0.9, 0.8, 0.75, 0.6, 0.5, 0.2], cl = [0, 0, 0, 1, 0, 0];
  const iou = (i: number, j: number) => Math.round(boxIou(B[i], B[j]) * 100) / 100;
  check(iou(0, 1) === 0.75 && iou(0, 3) === 0.9 && iou(1, 3) === 0.82 && iou(2, 4) === 0.68 && iou(0, 2) === 0.14, "IoU table");
  check(close(boxIou(B[1], B[2]), 0.199, 1e-3), "b2–b3 0.199");
  const cands = sc.map((s, k) => ({ cell: k, cls: cl[k], score: s })).filter((c) => c.score > 0.25);
  const kept = (t: number, agn: boolean) => M.runNms(fake(B), cands, t, agn).keep.map((k) => "b" + (cands[k].cell + 1)).sort().join(" ");
  check(kept(0.5, false) === "b1 b3 b4", `class-aware 0.5: ${kept(0.5, false)}`);
  check(kept(0.5, true) === "b1 b3", `agnostic 0.5: ${kept(0.5, true)}`);
  check(kept(0.2, false) === "b1 b3 b4" && kept(0.2, true) === "b1 b3", "same at 0.2");
  // B10: c debris 0.9, a cell 0.8, b cell 0.7
  const C = [bx(0, 0, 40, 40), bx(10, 0, 50, 40), bx(20, 0, 60, 40)];
  check(close(boxIou(C[0], C[1]), 0.6) && close(boxIou(C[1], C[2]), 0.6) && close(boxIou(C[0], C[2]), 1 / 3), "B10 IoUs");
  const cc = [{ cell: 0, cls: 1, score: 0.9 }, { cell: 1, cls: 0, score: 0.8 }, { cell: 2, cls: 0, score: 0.7 }];
  const k2 = (agn: boolean) => M.runNms(fake(C), cc, 0.5, agn).keep.map((k) => "cab"[k]).sort().join("");
  check(k2(false) === "ac" && k2(true) === "bc", `B10: aware ${k2(false)}, agnostic ${k2(true)}`);
}

current = "GIoU and the distribution loss (B7, B8)";
{
  const A = bx(0, 0, 4, 4);
  check(close(M.giou(A, bx(6, 0, 10, 4)), -0.2) && close(M.giou(A, bx(5, 0, 9, 4)), -4 / 36), "GIoU −0.2, −0.111");
  check(close(M.giou(A, A), 1) && M.giou(A, bx(100, 100, 101, 101)) > -1, "GIoU range");
  let prev = -2;
  for (let x = 20; x >= 4; x -= 1) { const g = M.giou(A, bx(x, 0, x + 4, 4)); check(g >= prev - 1e-12, `GIoU rises as B approaches (x = ${x})`); prev = g; }
  // minimiser of −(0.7 log p2 + 0.3 log p3) on p2 + p3 = 1
  let best = 0, bl = Infinity;
  for (let p = 0.001; p < 1; p += 0.001) { const l = -(0.7 * Math.log(p) + 0.3 * Math.log(1 - p)); if (l < bl) { bl = l; best = p; } }
  check(close(best, 0.7, 1.5e-3) && close(2 * 0.7 + 3 * 0.3, 2.3, 1e-12), `minimiser ${best}`);
}

current = "simulated network";
{
  const run = (id: M.SceneId, head: M.Head, conf: number, t: number, agn: boolean, scales = [true, true, true]) => {
    const sc = M.makeScene(id), sim = M.simulate(sc), dec = M.decodeAll(sim, head);
    const c = M.candidates(dec, conf, scales), n = M.runNms(dec, c, t, agn);
    return { sim, dec, c, n, ev: M.evaluate(sc.objs, n.keep.map((k) => dec.boxes[c[k].cell]), n.keep.map((k) => c[k].cls)) };
  };
  const a1 = M.simulate(M.makeScene("mixed")), a2 = M.simulate(M.makeScene("mixed"));
  check(a1.raw.every((v, i) => v === a2.raw[i]), "deterministic");
  const im1 = M.makeScene("mixed").image.data, im2 = M.makeScene("mixed").image.data;
  check(im1.every((v, i) => v === im2[i]), "deterministic image");
  const aware = run("mixed", "dist", 0.25, 0.45, false), agn = run("mixed", "dist", 0.25, 0.45, true);
  check(aware.ev.tp === 13 && aware.ev.fp === 1 && aware.ev.fn === 0, `mixed class-aware ${JSON.stringify(aware.ev)}`);
  check(agn.ev.tp === 13 && agn.ev.fp === 0 && agn.ev.fn === 0, `mixed agnostic ${JSON.stringify(agn.ev)}`);
  const r45 = run("rods", "dist", 0.25, 0.45, false), r50 = run("rods", "dist", 0.25, 0.5, false);
  check(r45.ev.fn === 1 && r45.ev.fp === 0 && r50.ev.fn === 0 && r50.ev.fp === 0, `rods 0.45 ${JSON.stringify(r45.ev)}, 0.5 ${JSON.stringify(r50.ev)}`);
  const anc = run("mixed", "anchor", 0.25, 0.45, false);
  const never = anc.sim.bestAnchor.map((v, k) => (v < 0.5 ? k : -1)).filter((k) => k >= 0);
  check(never.join() === "8,9,10,11" && anc.ev.missed.join() === "8,9,10,11", `anchors never learn ${never}, missed ${anc.ev.missed}`);
  // the large cell: stride 8 cannot reach its sides, stride 32 finds it
  const lg = run("large", "dist", 0.25, 0.45, false), mid = M.cellAt(0, 256, 260);
  check(lg.sim.target[mid * 4] > 15 && lg.dec.dist[mid * 4] < 15, "large cell: stride-8 target beyond 15 bins, decoded below");
  check(lg.ev.fn === 0 && lg.n.keep.some((k) => M.cellInfo(lg.c[k].cell).stride === 32 && boxIou(lg.dec.boxes[lg.c[k].cell], lg.sim.scene.objs[0].box) > 0.8), "large cell found at stride 32");
  check(boxIou(lg.dec.boxes[mid], lg.sim.scene.objs[0].box) < 0.5, "the stride-8 box of the large cell is too small (IoU < 0.5)");
  // the small cell: only stride-8 centres inside its box
  const small = M.makeScene("mixed").objs[8];
  check(M.centresInside(small.box).join() === "4,0,0", "small cell 4/0/0");
  check(run("mixed", "dist", 0.25, 0.45, false, [false, true, true]).ev.missed.includes(8), "without stride 8 the small cell is missed");
  check(close(M.centreness(5, 3, 5, 3), 1) && M.centreness(1, 3, 9, 3) < 0.4, "centreness");
}

console.log(`yolo-test: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
