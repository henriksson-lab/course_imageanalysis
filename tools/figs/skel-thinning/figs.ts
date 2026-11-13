// Static figures for skel-thinning/lec.qmd (SVG). Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/skel-thinning/figs.ts --bundle --platform=node --log-level=warning | node
// Uses the course's own skeleton library (demos/src/lib/skeleton.ts), so the pictures match the
// morphgrid demo. Numbers quoted on the slides are printed.
import { Svg, C, K, pixGrid, imageUp, hex } from "./svglib";
import {
  thin, thinSimple, zhangSuen, skeletonize, simpleTest, hitOrMissAny, rotations8,
  T_END, T_ISOLATED, T_TJUNCTION, T_YJUNCTION, RING, skeletonGraph,
} from "../../../demos/src/lib/skeleton";
import { growTree, renderTree } from "../../../demos/src/lib/trees";

const grid = (rows: string[]) => ({ w: rows[0].length, h: rows.length, m: Uint8Array.from(rows.join(""), c => (c === "1" ? 1 : 0)) });
const count = (m: ArrayLike<number>) => { let n = 0; for (let i = 0; i < m.length; i++) n += m[i] ? 1 : 0; return n; };

// ---------------------------------------------------------------- fig-why.svg
// Two bars of the same area but different length; and a synthetic neuron with its skeleton.
{
  const s = new Svg(640, 470);
  const bars = [{ L: 100, W: 3 }, { L: 50, W: 6 }];
  const px = 5;
  let y = 38;
  s.text(10, 22, "same area, different length", { size: 22 });
  for (const b of bars) {
    const w = b.L + 2, h = b.W + 2, m = new Uint8Array(w * h);
    for (let yy = 1; yy < 1 + b.W; yy++) for (let xx = 1; xx < 1 + b.L; xx++) m[yy * w + xx] = 1;
    const sk = skeletonize(m, w, h), g = skeletonGraph(sk, w, h);
    const rgb = new Uint8Array(3 * w * h);
    for (let i = 0; i < w * h; i++) { const c = hex(sk[i] ? K.obj : m[i] ? K.maskDark : "#ffffff"); rgb.set(c, 3 * i); }
    imageUp(s, 10, y, w * px, h * px, rgb, w, h, 1, false);
    s.text(10, y + h * px + 22, `${b.L} × ${b.W}: area ${b.L * b.W} px, skeleton ${count(sk)} px`, { size: 21 });
    console.log(`fig-why: bar ${b.L}x${b.W}: area ${b.L * b.W}, skeleton pixels ${count(sk)}, edges ${g.edges.length}`);
    y += h * px + 42;
  }
  // neuron
  const t = growTree({ kind: "neuron", seed: 7 }), { mask } = renderTree(t, { noise: 0, seed: 107 });
  const N = 256, sk = skeletonize(mask, N, N), g = skeletonGraph(sk, N, N);
  const rgb = new Uint8Array(3 * N * N);
  for (let i = 0; i < N * N; i++) rgb.set(hex(sk[i] ? "#111111" : mask[i] ? K.regular : "#ffffff"), 3 * i);
  const y0 = y + 14, D = 300, sc = D / N;
  imageUp(s, 10, y0, D, D, rgb, N, N, 2);
  const ends = g.nodes.filter(n => n.kind === "end"), junc = g.nodes.filter(n => n.kind === "junction");
  for (const n of junc) s.circle(10 + (n.x + 0.5) * sc, y0 + (n.y + 0.5) * sc, 4.5, K.junction, "#fff", 1.2);
  for (const n of ends) s.circle(10 + (n.x + 0.5) * sc, y0 + (n.y + 0.5) * sc, 4.5, K.end, "#fff", 1.2);
  const lx = 335;
  s.text(lx, y0 + 30, "a neuron mask (grey)", { size: 21 });
  s.text(lx, y0 + 56, "and its skeleton (dark)", { size: 21 });
  s.circle(lx + 8, y0 + 92, 7, K.end); s.text(lx + 24, y0 + 98, "end points", { size: 21 });
  s.circle(lx + 8, y0 + 124, 7, K.junction); s.text(lx + 24, y0 + 130, "branch points", { size: 21 });
  s.text(lx, y0 + 172, "area says little;", { size: 21 });
  s.text(lx, y0 + 198, "length and branching", { size: 21 });
  s.text(lx, y0 + 224, "describe the shape", { size: 21 });
  s.text(lx, y0 + 270, "synthetic neuron", { size: 19, color: "#666" });
  console.log(`fig-why: neuron mask ${count(mask)}, skeleton ${count(sk)}, ends ${ends.length}, junction nodes ${junc.length}`);
  s.H = y0 + D + 10;
  s.out("skel-thinning/fig-why.svg");
}

// ---------------------------------------------------------------- fig-classes.svg
// A small skeleton: hit-or-miss with the end-point, isolated, T- and Y-junction templates (all rotations).
{
  const G = grid([
    "000000000000000",
    "010000000100010",
    "000000000010100",
    "011111110001000",
    "000001000001000",
    "000001000001000",
    "000001000001000",
    "000001111111110",
    "000000000000000",
  ]);
  const ends = hitOrMissAny(G.m, G.w, G.h, rotations8(T_END));
  const iso = hitOrMissAny(G.m, G.w, G.h, [T_ISOLATED]);
  const tj = hitOrMissAny(G.m, G.w, G.h, rotations8(T_TJUNCTION));
  const yj = hitOrMissAny(G.m, G.w, G.h, rotations8(T_YJUNCTION));
  const cell = 36, s = new Svg(G.w * cell + 20, G.h * cell + 120);
  pixGrid(s, 10, 10, G.w, G.h, cell, (x, y) => {
    const i = y * G.w + x;
    if (!G.m[i]) return null;
    return iso[i] ? K.purple : ends[i] ? K.end : tj[i] || yj[i] ? K.junction : K.regular;
  }, (x, y) => {
    const i = y * G.w + x;
    return iso[i] ? { t: "I", color: "#fff" } : ends[i] ? { t: "E", color: "#fff" } : tj[i] ? { t: "T", color: "#fff" } : yj[i] ? { t: "Y", color: "#fff" } : null;
  }, 18);
  const ly = G.h * cell + 42;
  const items: [string, string][] = [[K.end, "E  end point"], [K.junction, "T, Y  junction"], [K.purple, "I  isolated"], [K.regular, "no template fits"]];
  items.forEach(([c, t], k) => { const lx = 10 + (k % 2) * 270, yy = ly + Math.floor(k / 2) * 30; s.rect(lx, yy - 16, 20, 20, c); s.text(lx + 28, yy, t, { size: 21 }); });
  s.H = ly + 50;
  console.log(`fig-classes: ends ${count(ends)}, isolated ${count(iso)}, T ${count(tj)}, Y ${count(yj)}`);
  s.out("skel-thinning/fig-classes.svg");
}

// ---------------------------------------------------------------- fig-simple.svg
// The four neighbourhoods of the "simple points" slide: object parts (letters), background parts (numbers).
{
  const cases: { name: string; rows: string[] }[] = [
    { name: "straight edge", rows: ["000", "010", "111"] },
    { name: "middle of a line", rows: ["000", "111", "000"] },
    { name: "interior", rows: ["111", "111", "111"] },
    { name: "isolated", rows: ["000", "010", "000"] },
  ];
  const cell = 46, pw = 3 * cell, colW = 300, rowH = 250;
  const s = new Svg(2 * colW, 2 * rowH);
  const PART = ["#2a3346", "#7b4ea3"], BGC = ["#1f77b4", "#ff7f0e"];
  cases.forEach((c, k) => {
    const x0 = (k % 2) * colW + (colW - pw) / 2, y0 = Math.floor(k / 2) * rowH + 34;
    const nb = RING.map(([dx, dy]) => (c.rows[1 + dy][1 + dx] === "1" ? 1 : 0));
    const st = simpleTest(nb);
    const fgOf = new Map<number, number>(), bgOf = new Map<number, number>();
    st.fg.forEach((comp, j) => comp.forEach(r => fgOf.set(r, j)));
    st.bg.forEach((comp, j) => comp.forEach(r => bgOf.set(r, j)));
    const ringAt = (x: number, y: number) => RING.findIndex(([dx, dy]) => dx === x - 1 && dy === y - 1);
    s.text(x0 + pw / 2, y0 - 12, c.name, { anchor: "middle", size: 22 });
    pixGrid(s, x0, y0, 3, 3, cell, (x, y) => {
      if (x === 1 && y === 1) return K.orange;
      const r = ringAt(x, y);
      return fgOf.has(r) ? PART[fgOf.get(r)! % 2] : null;
    }, (x, y) => {
      if (x === 1 && y === 1) return { t: "p", color: "#fff" };
      const r = ringAt(x, y);
      if (fgOf.has(r)) return { t: String.fromCharCode(97 + fgOf.get(r)!), color: "#fff" };
      if (bgOf.has(r)) return { t: String(bgOf.get(r)! + 1), color: BGC[bgOf.get(r)! % 2] };
      return null;
    }, 20);
    const v = `${st.fg.length} object, ${st.bg.length} background`;
    s.text(x0 + pw / 2, y0 + pw + 28, v, { anchor: "middle", size: 21 });
    s.text(x0 + pw / 2, y0 + pw + 54, st.simple ? "simple" : "not simple", { anchor: "middle", size: 22, weight: "bold", color: st.simple ? C.green : C.red });
    console.log(`fig-simple: ${c.name}: fg ${st.fg.length} bg ${st.bg.length} simple ${st.simple} (${st.reason})`);
  });
  s.out("skel-thinning/fig-simple.svg");
}

// ---------------------------------------------------------------- fig-endpoints.svg
// 3 × 7 rectangle: template thinning (ends kept) vs simple-point thinning without end points; the ring.
{
  const RECT = grid(["000000000", "011111110", "011111110", "011111110", "000000000"]);
  const RINGG = grid(["000000000", "001111100", "011111110", "011000110", "011000110", "011000110", "011111110", "001111100", "000000000"]);
  const keep = thin(RECT.m, RECT.w, RECT.h).skeleton;
  const noEnds = thinSimple(RECT.m, RECT.w, RECT.h, { keepEnds: false }).skeleton;
  const ring = thinSimple(RINGG.m, RINGG.w, RINGG.h, { keepEnds: false }).skeleton;
  const cell = 28, s = new Svg(620, 440);
  const draw = (G: { w: number; h: number; m: Uint8Array }, sk: Uint8Array, x0: number, y0: number, label: string, sub: string) => {
    s.text(x0, y0 - 36, label, { size: 21 });
    s.text(x0, y0 - 10, sub, { size: 20, color: "#555" });
    pixGrid(s, x0, y0, G.w, G.h, cell, (x, y) => { const i = y * G.w + x; return sk[i] ? K.obj : G.m[i] ? K.removed : null; });
  };
  draw(RECT, keep, 10, 64, "end points kept", `${count(keep)} pixels left`);
  draw(RECT, noEnds, 10, 284, "end points not kept", `${count(noEnds)} pixel left`);
  draw(RINGG, ring, 330, 64, "a ring, end points not kept", `${count(ring)} pixels: the loop stays`);
  s.rect(330, 360, 20, 20, K.obj); s.text(358, 376, "skeleton", { size: 20 });
  s.rect(460, 360, 20, 20, K.removed); s.text(488, 376, "removed", { size: 20 });
  console.log(`fig-endpoints: rect keep ends ${count(keep)}, no ends ${count(noEnds)}, ring ${count(ring)}`);
  s.out("skel-thinning/fig-endpoints.svg");
}

// ---------------------------------------------------------------- fig-zhangsuen.svg
// Zhang–Suen vs template thinning on the same small shape: artefacts at the ends, corners, diagonals.
{
  const w = 30, h = 22, m = new Uint8Array(w * h);
  // a thick horizontal bar, a thick vertical bar down from it, and a diagonal arm
  for (let y = 3; y < 9; y++) for (let x = 2; x < 22; x++) m[y * w + x] = 1;
  for (let y = 3; y < 20; y++) for (let x = 9; x < 14; x++) m[y * w + x] = 1;
  for (let k = 0; k < 9; k++) for (let d = -2; d <= 2; d++) { const x = 20 + k + (d > 0 ? d : 0), y = 8 + k - (d < 0 ? d : 0) + 0; if (x < w - 1 && y < h - 1) m[y * w + x] = 1; }
  const zs = zhangSuen(m, w, h).skeleton, tp = thin(m, w, h).skeleton;
  const cell = 9.5, s = new Svg(620, 300);
  const draw = (sk: Uint8Array, x0: number, label: string) => {
    s.text(x0, 24, label, { size: 22 });
    pixGrid(s, x0, 38, w, h, cell, (x, y) => { const i = y * w + x; return sk[i] ? K.obj : m[i] ? K.mask : null; });
  };
  draw(zs, 10, "Zhang–Suen (parallel)");
  draw(tp, 320, "template thinning");
  s.text(10, 290, `${count(zs)} vs ${count(tp)} skeleton pixels; grey: the shape`, { size: 20, color: "#555" });
  console.log(`fig-zhangsuen: ZS ${count(zs)}, templates ${count(tp)}`);
  s.out("skel-thinning/fig-zhangsuen.svg");
}
