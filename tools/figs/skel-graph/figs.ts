// Static figures for skel-graph/lec.qmd (SVG). Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/skel-graph/figs.ts --bundle --platform=node --log-level=warning | node
// Uses the skelgraph demo's own masks and analysis (demos/src/skelgraph/masks.ts), so the
// pictures and numbers match the demo presets. Numbers quoted on the slides are printed.
declare const process: any;
import { Svg, C, K, pixGrid, imageUp, hex } from "../skel-thinning/svglib";
import { makeMask, analyseMask, N } from "../../../demos/src/skelgraph/masks";
import { classifySkeleton, cyclomatic, chainCounts, lengthFromCounts, totalLength, SkelGraph } from "../../../demos/src/lib/skeleton";
import { topology } from "../../../demos/src/lib/skeleton";

const crop = (x0: number, y0: number, w: number, h: number, f: (i: number) => [number, number, number]) => {
  const rgb = new Uint8Array(3 * w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) rgb.set(f((y0 + y) * N + x0 + x), 3 * (y * w + x));
  return rgb;
};
const thicken = (a: Uint8Array): Uint8Array => {
  const o = a.slice();
  for (let y = 1; y < N - 1; y++) for (let x = 1; x < N - 1; x++) if (a[y * N + x]) { o[y * N + x + 1] = 1; o[(y + 1) * N + x] = 1; }
  return o;
};
/** Nodes of a graph drawn on a crop at scale f. */
function nodes(s: Svg, g: SkelGraph, ox: number, oy: number, x0: number, y0: number, f: number, r = 6) {
  for (const n of g.nodes) {
    if (n.kind !== "end" && n.kind !== "junction") continue;
    s.circle(ox + (n.x - x0 + 0.5) * f, oy + (n.y - y0 + 0.5) * f, r, n.kind === "end" ? K.end : K.junction, "#fff", 1.5);
  }
}
const count = (g: SkelGraph, k: string) => g.nodes.filter(n => n.kind === k).length;

// ---------------------------------------------------------------- fig-classify.svg
// A zoom into the Y's skeleton at its junction: pixel classes, and the junction cluster = one node.
{
  const MID = (process.env.CLS ?? "spurs") as "loop" | "spurs", at = MID === "loop" ? [66, 67] : [125, 128];
  const r = analyseMask(makeMask(MID, 30), false, 0, false, "weighted");
  const cls = classifySkeleton(r.skel, N, N);
  const J = r.graph.nodes.filter(n => n.kind === "junction").sort((a, b) => Math.hypot(a.x - at[0], a.y - at[1]) - Math.hypot(b.x - at[0], b.y - at[1]))[0];
  const cw = 15, ch = 13, cx = Math.round(J.x) - 7, cy = Math.round(J.y) - 6, cell = 30;
  const s = new Svg(cw * cell + 20, ch * cell + 100);
  const inCl = new Set(J.pixels);
  pixGrid(s, 10, 10, cw, ch, cell, (x, y) => {
    const i = (cy + y) * N + cx + x, c = cls[i];
    if (!c) return r.dt[i] > 0 ? K.mask : null;
    return c === 1 ? K.end : c === 3 ? K.junction : K.regular;
  }, (x, y) => {
    const i = (cy + y) * N + cx + x, xx = cx + x, yy = cy + y;
    if (!cls[i]) return null;
    let d = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) if ((dx || dy) && r.skel[(yy + dy) * N + xx + dx]) d++;
    return { t: String(d), color: "#fff" };
  }, 17);
  // outline the junction cluster
  const xs = J.pixels.map(p => p % N - cx), ys = J.pixels.map(p => Math.floor(p / N) - cy);
  const bx = Math.min(...xs), by = Math.min(...ys), bw = Math.max(...xs) - bx + 1, bh = Math.max(...ys) - by + 1;
  s.rect(10 + bx * cell - 4, 10 + by * cell - 4, bw * cell + 8, bh * cell + 8, "none", { stroke: K.orange, sw: 3, rx: 6 });
  const ly = ch * cell + 44;
  s.rect(10, ly - 16, 20, 20, K.regular); s.text(38, ly, "2 neighbours: regular", { size: 21 });
  s.rect(280, ly - 16, 20, 20, K.junction); s.text(308, ly, "≥ 3: junction", { size: 21 });
  s.text(10, ly + 32, `orange frame: ${J.pixels.length} junction pixels touch: one node`, { size: 21 });
  console.log(`fig-classify: junction cluster of ${J.pixels.length} pixels at (${J.x.toFixed(1)}, ${J.y.toFixed(1)})`);
  s.out("skel-graph/fig-classify.svg");
}

// ---------------------------------------------------------------- fig-loops.svg
{
  const m = makeMask("loop", 30), r = analyseMask(m, false, 0, false, "weighted"), g = r.graph, topo = topology(m, N, N);
  const x0 = 34, y0 = 34, w = 188, h = 188, f = 2.2;
  const sk = thicken(r.skel);
  const s = new Svg(Math.round(w * f) + 20, Math.round(h * f) + 90);
  imageUp(s, 10, 10, w * f, h * f, crop(x0, y0, w, h, i => (sk[i] ? hex(K.obj) : m[i] ? hex(K.mask) : [255, 255, 255])), w, h, 3);
  nodes(s, g, 10, 10, x0, y0, f, 7);
  const E = g.edges.length, V = g.nodes.length, Cc = g.components;
  s.text(10, h * f + 44, `E − V + C = ${E} − ${V} + ${Cc} = ${cyclomatic(g)} = holes (${topo.holes})`, { size: 22 });
  s.text(10, h * f + 74, "blue: end points, red: junctions", { size: 20, color: "#555" });
  console.log(`fig-loops: E ${E} V ${V} C ${Cc} cyc ${cyclomatic(g)} holes ${topo.holes}`);
  s.out("skel-graph/fig-loops.svg");
}

// ---------------------------------------------------------------- fig-staircase.svg
// A digital line of slope 1/2: axial and diagonal steps, versus the true line.
{
  const n = 10, cell = 38, W = n + 1, H = n / 2 + 1;
  const pix: [number, number][] = [];
  for (let x = 0; x <= n; x++) pix.push([x, H - 1 - Math.round(x / 2 - 1e-9)]);
  const s = new Svg(W * cell + 20, H * cell + 150);
  pixGrid(s, 10, 10, W, H, cell, (x, y) => (pix.some(p => p[0] === x && p[1] === y) ? K.mask : null));
  const P = (p: [number, number]) => [10 + (p[0] + 0.5) * cell, 10 + (p[1] + 0.5) * cell];
  let ne = 0, no = 0;
  for (let k = 1; k < pix.length; k++) {
    const a = P(pix[k - 1]), b = P(pix[k]), diag = pix[k][1] !== pix[k - 1][1];
    s.line(a[0], a[1], b[0], b[1], diag ? K.orange : K.end, 5);
    if (diag) no++; else ne++;
  }
  const a = P(pix[0]), b = P(pix[pix.length - 1]);
  s.line(a[0], a[1], b[0], b[1], "#222", 2, "7 5");
  const idx = pix.map(([x, y]) => y * 1000 + x), cc = chainCounts(idx, 1000);
  const truth = Math.hypot(n, n / 2);
  const ly = H * cell + 44;
  s.line(12, ly - 6, 42, ly - 6, K.end, 5); s.text(50, ly, `${ne} axial steps, 1 each`, { size: 20 });
  s.line(12, ly + 26, 42, ly + 26, K.orange, 5); s.text(50, ly + 32, `${no} diagonal steps, √2 each`, { size: 20 });
  s.line(12, ly + 58, 42, ly + 58, "#222", 2, "7 5"); s.text(50, ly + 64, `true line ${truth.toFixed(1)}; steps ${ne + no}; 1 and √2: ${lengthFromCounts(cc, "weighted").toFixed(1)}`, { size: 20 });
  console.log(`fig-staircase: ne ${cc.ne} no ${cc.no} nc ${cc.nc} truth ${truth.toFixed(2)} count ${lengthFromCounts(cc, "count")} weighted ${lengthFromCounts(cc, "weighted").toFixed(2)} corrected ${lengthFromCounts(cc, "corrected").toFixed(2)}`);
  s.out("skel-graph/fig-staircase.svg");
}

// ---------------------------------------------------------------- fig-spurs.svg, fig-prune.svg
{
  const m = makeMask("spurs", 30);
  const r0 = analyseMask(m, false, 0, false, "weighted");
  const x0 = 14, y0 = 36, w = 228, h = 210;
  const g0 = r0.graph;
  {
    const f = 2.0, sk = thicken(r0.skel);
    const s = new Svg(Math.round(w * f) + 20, Math.round(h * f) + 60);
    imageUp(s, 10, 10, w * f, h * f, crop(x0, y0, w, h, i => (sk[i] ? hex(K.obj) : m[i] ? hex(K.mask) : [255, 255, 255])), w, h, 3);
    nodes(s, g0, 10, 10, x0, y0, f, 6);
    s.text(10, h * f + 44, `${count(g0, "end")} end points, ${count(g0, "junction")} junctions, ${g0.edges.length} edges`, { size: 22 });
    console.log(`fig-spurs: ends ${count(g0, "end")} junctions ${count(g0, "junction")} edges ${g0.edges.length} length ${totalLength(g0, "weighted").toFixed(1)}`);
    s.out("skel-graph/fig-spurs.svg");
  }
  {
    const f = 1.2, s = new Svg(2 * Math.round(w * f) + 40, Math.round(h * f) + 130);
    [12, 24].forEach((k, j) => {
      const r = analyseMask(m, false, k, false, "weighted"), g = r.graph;
      const gone = thicken(Uint8Array.from(r0.skel, (v, i) => (v && !r.skel[i] ? 1 : 0))), kept = thicken(r.skel);
      const ox = 10 + j * (w * f + 20), nj = count(g, "junction");
      s.text(ox, 24, `k = ${k}: ${r.removed} spurs cut`, { size: 22 });
      imageUp(s, ox, 38, w * f, h * f, crop(x0, y0, w, h, i => (kept[i] ? hex(K.obj) : gone[i] ? hex(K.spur) : m[i] ? hex(K.mask) : [255, 255, 255])), w, h, 3);
      nodes(s, g, ox, 38, x0, y0, f, 5);
      s.text(ox, 38 + h * f + 30, `${count(g, "end")} ends, ${nj} junction${nj === 1 ? "" : "s"},`, { size: 22 });
      s.text(ox, 38 + h * f + 58, `${g.edges.length} edges`, { size: 22 });
      console.log(`fig-prune: k ${k} removed ${r.removed} ends ${count(g, "end")} junctions ${nj} edges ${g.edges.length} length ${totalLength(g, "weighted").toFixed(1)}`);
    });
    s.text(10, h * f + 124, "pink: pruned pixels", { size: 20, color: "#555" });
    s.out("skel-graph/fig-prune.svg");
  }
}
