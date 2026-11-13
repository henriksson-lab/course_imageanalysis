// Tests for the flow-field decoding of the cellpose/ unit (demos/src/flowtrack/model.ts) and the
// numbers of its exercises (B1 heat steps, B2 grid, B3 flow check, B9 two opposite arrows).
// Run: node_modules/.bin/esbuild demos/test/cellpose-test.ts --bundle --platform=node --log-level=warning | node
import * as M from "../src/flowtrack/model";
import { iouTable, match, objectScores } from "../src/lib/eval";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;
const N = M.SIZE;

// ---- B1: 1D heat target on 5 pixels, source in the middle
current = "heat 1D";
{
  const lab = new Int32Array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]); // a 5 x 2 image, cell = top row
  check(M.sourcePixel(lab, 5, 2, 1) === 2, "source = middle pixel");
  const want = [[0, 1 / 3, 1 / 3, 1 / 3, 0], [1 / 6, 5 / 9, 2 / 3, 5 / 9, 1 / 6], [0.3611, 0.7963, 0.9259, 0.7963, 0.3611]];
  want.forEach((wv, i) => {
    const h = M.heatTarget(lab, 5, 2, 1, i + 1);
    check(wv.every((v, k) => close(h[k], v, 1e-4)), `step ${i + 1}: ${Array.from(h.slice(0, 5)).map((v) => v.toFixed(4)).join(" ")}`);
  });
  const vx = new Float32Array(10), vy = new Float32Array(10);
  M.flowFromHeat(lab, 5, 2, 1, M.heatTarget(lab, 5, 2, 1, 3), vx, vy);
  check(Array.from(vx.slice(0, 5)).join() === "1,1,0,-1,-1", `arrows → → sink ← ← (${Array.from(vx.slice(0, 5)).join()})`);
  // 11 pixels, 3 steps: the heat has reached pixels 2..8 only; the two end pixels have no arrow yet
  const lab11 = new Int32Array(22); lab11.fill(1, 0, 11);
  const h11 = M.heatTarget(lab11, 11, 2, 1, 3);
  check(h11[0] === 0 && h11[1] === 0 && h11[9] === 0 && h11[10] === 0 && h11[2] > 0 && h11[8] > 0, "11 pixels after 3 steps: pixels 0, 1, 9, 10 still at 0");
  const vx11 = new Float32Array(22), vy11 = new Float32Array(22);
  M.flowFromHeat(lab11, 11, 2, 1, h11, vx11, vy11);
  check(vx11[0] === 0 && vx11[10] === 0 && vx11[1] === 1 && vx11[9] === -1, "11 pixels: no arrow at the two ends yet");
}

// ---- B2: the 5 x 5 grid
current = "grid B2";
{
  const f = M.B2_FIELD;
  check(f.length === 25 && f.every((v) => v >= 0), "field parsed");
  const p = (r: number, c: number) => r * 5 + c;
  const a = M.gridFollow(f, 5, 5, p(0, 0)), b = M.gridFollow(f, 5, 5, p(4, 0)), c = M.gridFollow(f, 5, 5, p(3, 3));
  check(a.cycle.slice().sort().join() === [p(0, 0), p(0, 1)].join(), "(0,0) oscillates with (0,1)");
  check(b.path.join() === [p(4, 0), p(3, 1), p(2, 1), p(2, 2)].join() && b.cycle.join() === String(p(2, 2)), "(4,0) → (3,1) → (2,1) → (2,2)");
  check(c.path.join() === [p(3, 3), p(4, 4)].join() && c.cycle.join() === String(p(4, 4)), "(3,3) → (4,4)");
  const g = M.gridGroups(f, 5, 5);
  const sizes = [...g.values()].map((v) => v.length).sort((x, y) => y - x);
  check(sizes.join() === "19,4,2", `groups 19, 4, 2 (${sizes.join()})`);
  // fixing one arrow: (0,1) pointing right instead of left joins the oscillating pair to the centre
  const fixed = f.slice(); fixed[p(0, 1)] = M.GLYPH.indexOf("→");
  const g2 = [...M.gridGroups(fixed, 5, 5).values()].map((v) => v.length).sort((x, y) => y - x);
  check(g2.join() === "21,4", `(0,1) → right: groups 21, 4 (${g2.join()})`);
  const fixed2 = f.slice(); fixed2[p(0, 0)] = M.GLYPH.indexOf("↘");
  const g3 = [...M.gridGroups(fixed2, 5, 5).values()].map((v) => v.length).sort((x, y) => y - x);
  check(g3.join() === "21,4", `(0,0) → down-right: groups 21, 4 (${g3.join()})`);
}

// ---- B3: flow check arithmetic (the formula of flowErrors on four given pixels)
current = "flow check";
{
  const mse = (pr: number[][], mk: number[][]) => pr.reduce((s, v, i) => s + (v[0] - mk[i][0]) ** 2 + (v[1] - mk[i][1]) ** 2, 0) / pr.length;
  const mk = [[1, 0], [1, 0], [0, 1], [1, 0]];
  check(close(mse([[1, 0], [0.8, 0.6], [0, 1], [-1, 0]], mk), 1.1, 1e-12), "mean 1.1");
  check(close(mse([[1, 0], [0.8, 0.6], [0, 1], [0.6, 0.8]], mk), 0.3, 1e-12), "mean 0.3");
}

// ---- B9: two arrows pointing at each other, bilinear interpolation
current = "opposite arrows";
{
  const vx = new Float32Array([1, -1, 1, -1]), vy = new Float32Array(4); // 2 x 2: left column → , right column ←
  const p1 = M.tracePath(vx, vy, 2, 2, 0, 0, 6, 1);
  check(p1[2] === 1 && p1[4] === 0 && p1[12] === 0, "step 1 from a pixel centre: oscillates 0, 1, 0, …");
  const p2 = M.tracePath(vx, vy, 2, 2, 0, 0, 40, 0.5);
  check(close(p2[80], 0.5, 1e-6), `step 0.5: stops at x = 0.5 (${p2[80]})`);
  const p3 = M.tracePath(vx, vy, 2, 2, 0.1, 0, 60, 0.25);
  check(close(p3[120], 0.5, 1e-6), "step 0.25: converges to 0.5");
}

// ---- exact fields on the scenes: one sink per cell, paths end at the source
current = "exact fields";
for (const s of M.SHAPES) {
  const lab = M.makeShape(s.id);
  let n = 0; for (const v of lab) if (v > n) n = v;
  const T = M.flowTargets(lab, N, N);
  for (let l = 1; l <= n; l++) {
    const src = T.sources[l - 1];
    const H = M.heatTarget(lab, N, N, l, T.iters[l - 1], src);
    let mx = -1, arg = -1; for (let p = 0; p < N * N; p++) if (lab[p] === l && H[p] > mx) { mx = H[p]; arg = p; }
    check(arg === src, `${s.id} cell ${l}: heat maximal at the source`);
    let noWarmer = 0;
    for (let p = 0; p < N * N; p++) if (lab[p] === l && p !== src) {
      const x = p % N, y = (p - x) / N; let up = false;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const xx = x + dx, yy = y + dy; if (xx >= 0 && yy >= 0 && xx < N && yy < N && lab[yy * N + xx] === l && H[yy * N + xx] > H[p]) up = true; }
      if (!up) noWarmer++;
    }
    check(noWarmer === 0, `${s.id} cell ${l}: every pixel but the source has a warmer neighbour (${noWarmer} do not)`);
    let far = 0, cnt = 0;
    for (let p = 0; p < N * N; p++) if (lab[p] === l) {
      const path = M.tracePath(T.vx, T.vy, N, N, p % N, Math.floor(p / N), 200, 1);
      const ex = path[400], ey = path[401];
      if (Math.hypot(ex - (src % N), ey - Math.floor(src / N)) > 1.5) far++;
      cnt++;
    }
    check(far === 0, `${s.id} cell ${l}: every path ends within 1.5 px of the source (${far} of ${cnt} do not)`);
  }
  const fg = Uint8Array.from(lab, (v) => (v ? 1 : 0));
  const d = M.decode(fg, T.vx, T.vy, N, N, { steps: 100 });
  const sc = objectScores(match(iouTable(lab, d.masks), 0.5));
  check(d.count === n && sc.f1 === 1, `${s.id}: decoding recovers the ${n} cell(s) (count ${d.count}, F1 ${sc.f1})`);
  check(d.errors.every((e) => e < 1e-9), `${s.id}: flow error 0 for the exact field`);
}

// ---- simulated errors (the demo presets)
current = "presets";
{
  const run = (shape: M.ShapeId, e: Partial<M.Errors>, steps: number, thr: number) => {
    const lab = M.makeShape(shape);
    const pr = M.predict(lab, N, N, { noise: 0, blur: 0, shared: false, debris: false, ...e });
    const d = M.decode(pr.fg, pr.vx, pr.vy, N, N, { steps, flowThreshold: thr });
    return { d, sc: objectScores(match(iouTable(lab, d.masks), 0.5)) };
  };
  const shared = run("touching", { shared: true }, 100, 0.4);
  check(shared.d.count === 1 && shared.sc.fn === 1, "shared field: the two touching cells merge");
  const debris = run("disc", { debris: true }, 100, 0.4);
  check(debris.d.nRaw === 2 && debris.d.count === 1 && debris.d.dropped.filter(Boolean).length === 1, "debris: two masks, the flow check drops one");
  const noisy = run("C", { noise: 0.35 }, 100, 0);
  check(noisy.d.seeds.length === 2 && noisy.d.count === 2, `noise 0.35 on the C: an extra sink, 2 pieces (${noisy.d.count})`);
  const noisyChecked = run("C", { noise: 0.35 }, 100, 0.4);
  check(noisyChecked.d.count === 0, "noise 0.35 with the flow check: both pieces dropped");
  const few = run("C", {}, 10, 0.4);
  let a = 0; for (const v of few.d.masks) if (v) a++;
  check(a < 528 * 0.6 && few.sc.tp === 0, `10 steps on the C: the mask shrinks (${a} of 528 pixels)`);
  const enough = run("C", {}, 20, 0.4);
  check(enough.sc.tp === 1, "20 steps on the C: found");
}

console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
