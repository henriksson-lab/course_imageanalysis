// Static figures for skel-neurovessel/lec.qmd (SVG). Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/skel-neurovessel/figs.ts --bundle --platform=node --log-level=warning | node
// Uses the arbor demo's own pipeline (demos/src/arbor/pipeline.ts), so the pictures and numbers
// match the demo presets. Numbers quoted on the slides are printed.
import { Svg, C, K, imageUp, hex, viridis, magma } from "../skel-thinning/svglib";
import { generate, vesselness, segment, measure, truthMeasures, matchPoints, NEURON, VESSEL, SEG_N, SEG_V, N } from "../../../demos/src/arbor/pipeline";
import { pathLength, chord } from "../../../demos/src/lib/skeleton";
import { label2d } from "../../../demos/src/lib/label";

const thicken = (a: ArrayLike<number>): Uint8Array => {
  const o = Uint8Array.from(a as ArrayLike<number>, v => (v ? 1 : 0));
  for (let y = 0; y < N - 1; y++) for (let x = 0; x < N - 1; x++) if (a[y * N + x]) { o[y * N + x + 1] = 1; o[(y + 1) * N + x] = 1; }
  return o;
};
const rgbOf = (f: (i: number) => [number, number, number]) => { const o = new Uint8Array(3 * N * N); for (let i = 0; i < N * N; i++) o.set(f(i), 3 * i); return o; };
const greyRGB = (d: ArrayLike<number>, lo: number, hi: number) => rgbOf(i => { const v = Math.round(255 * Math.max(0, Math.min(1, (d[i] - lo) / (hi - lo)))); return [v, v, v]; });
const pieces = (m: Uint8Array) => label2d(m, N, N, 8).count;
const frac = (m: Uint8Array) => { let n = 0; for (const v of m) n += v; return n / m.length; };

const vgen = generate(VESSEL);

// ---------------------------------------------------------------- fig-threshold.svg
{
  const img = vgen.image.data;
  const hiT = 130, loT = 52;
  const mHi = Uint8Array.from(img, v => (v > hiT ? 1 : 0)), mLo = Uint8Array.from(img, v => (v > loT ? 1 : 0));
  const V = vesselness(vgen.image, SEG_V).V, seg = segment(V, SEG_V);
  let lo = Infinity, hi = -Infinity; for (const v of img) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const D = 230, gap = 26, s = new Svg(2 * D + gap + 20, 2 * D + 2 * 64 + 10);
  const panels: [Uint8Array, string, string][] = [
    [greyRGB(img, lo, hi), "image", ""],
    [rgbOf(i => (mHi[i] ? [255, 255, 255] : [20, 20, 28])), `threshold ${hiT}: gaps`, `${pieces(mHi)} pieces`],
    [rgbOf(i => (mLo[i] ? [255, 255, 255] : [20, 20, 28])), `threshold ${loT}: too fat`, `${(100 * frac(mLo)).toFixed(0)} % foreground`],
    [rgbOf(i => (seg.mask[i] ? [255, 255, 255] : [20, 20, 28])), "vesselness + hysteresis", `${pieces(seg.mask)} piece${pieces(seg.mask) === 1 ? "" : "s"}`],
  ];
  panels.forEach(([rgb, t, sub], k) => {
    const x = 10 + (k % 2) * (D + gap), y = 30 + Math.floor(k / 2) * (D + 64);
    s.text(x, y - 9, t, { size: 21 });
    imageUp(s, x, y, D, D, rgb, N, N, 2);
    if (sub) s.text(x, y + D + 24, sub, { size: 20, color: "#555" });
  });
  console.log(`fig-threshold: grey range ${lo.toFixed(0)}..${hi.toFixed(0)}; high ${hiT}: ${pieces(mHi)} pieces, ${(100 * frac(mHi)).toFixed(1)}%; low ${loT}: ${pieces(mLo)} pieces, ${(100 * frac(mLo)).toFixed(1)}%; vesselness mask ${pieces(seg.mask)} pieces ${(100 * frac(seg.mask)).toFixed(1)}%`);
  s.out("skel-neurovessel/fig-threshold.svg");
}

// ---------------------------------------------------------------- fig-scales.svg
{
  const one = { ...SEG_V, sMax: 1 };
  const s1 = segment(vesselness(vgen.image, one).V, one), s5 = segment(vesselness(vgen.image, SEG_V).V, SEG_V);
  const t = truthMeasures(vgen), m1 = measure(vgen, s1), m5 = measure(vgen, s5);
  const D = 280, s = new Svg(2 * D + 50, D + 100);
  [[s1, m1, "σ = 1 only"], [s5, m5, "σ = 1 … 5"]].forEach(([sg, m, title]: any, k) => {
    const x = 10 + k * (D + 30);
    s.text(x, 22, title, { size: 22 });
    imageUp(s, x, 34, D, D, rgbOf(i => (sg.skel[i] ? hex(K.orange) : sg.mask[i] ? [235, 235, 240] : [20, 20, 28])), N, N, 3);
    s.text(x, 34 + D + 26, `${m.branchPoints} branch points, ${m.loops} loop${m.loops === 1 ? "" : "s"}`, { size: 21 });
    s.text(x, 34 + D + 50, `length ${m.length.corrected.toFixed(0)} (truth ${t.length.toFixed(0)})`, { size: 21 });
  });
  console.log(`fig-scales: truth length ${t.length.toFixed(0)} bp ${t.branchPoints} loops ${t.loops}; one scale length ${m1.length.corrected.toFixed(0)} (${(100 * (m1.length.corrected / t.length - 1)).toFixed(0)}%) bp ${m1.branchPoints} loops ${m1.loops}; 1-5 length ${m5.length.corrected.toFixed(0)} bp ${m5.branchPoints} loops ${m5.loops}`);
  s.out("skel-neurovessel/fig-scales.svg");
}

// ---------------------------------------------------------------- fig-orders.svg
// A small dendritic tree: centrifugal order (from the soma) and Strahler order (from the tips).
{
  interface Br { id: string; from: [number, number]; to: [number, number]; kids: string[] }
  const B: Br[] = [
    { id: "r", from: [150, 300], to: [150, 225], kids: ["A", "B"] },
    { id: "A", from: [150, 225], to: [95, 160], kids: ["A1", "A2"] },
    { id: "A1", from: [95, 160], to: [35, 140], kids: [] },
    { id: "A2", from: [95, 160], to: [100, 95], kids: ["A2a", "A2b"] },
    { id: "A2a", from: [100, 95], to: [70, 35], kids: [] },
    { id: "A2b", from: [100, 95], to: [135, 40], kids: [] },
    { id: "B", from: [150, 225], to: [210, 150], kids: ["B1", "B2"] },
    { id: "B1", from: [210, 150], to: [265, 160], kids: [] },
    { id: "B2", from: [210, 150], to: [215, 90], kids: ["B2a", "B2b"] },
    { id: "B2a", from: [215, 90], to: [185, 35], kids: [] },
    { id: "B2b", from: [215, 90], to: [265, 45], kids: [] },
  ];
  const by = new Map(B.map(b => [b.id, b]));
  const cf = new Map<string, number>(), sh = new Map<string, number>();
  const down = (id: string, o: number) => { cf.set(id, o); for (const k of by.get(id)!.kids) down(k, o + 1); };
  const up = (id: string): number => {
    const b = by.get(id)!;
    if (!b.kids.length) { sh.set(id, 1); return 1; }
    const v = b.kids.map(up), mx = Math.max(...v), n = v.filter(x => x === mx).length;
    sh.set(id, n >= 2 ? mx + 1 : mx); return sh.get(id)!;
  };
  down("r", 1); up("r");
  const COL = ["", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"];
  const s = new Svg(640, 380);
  const draw = (ox: number, ord: Map<string, number>, title: string, sub: string) => {
    s.text(ox + 150, 24, title, { size: 22, anchor: "middle" });
    s.text(ox + 150, 366, sub, { size: 20, anchor: "middle", color: "#555" });
    for (const b of B) {
      const o = ord.get(b.id)!;
      s.line(ox + b.from[0], b.from[1] + 20, ox + b.to[0], b.to[1] + 20, COL[o], 3 + 1.5 * o);
    }
    for (const b of B) {
      const o = ord.get(b.id)!, mx = (b.from[0] + b.to[0]) / 2, my = (b.from[1] + b.to[1]) / 2 + 20;
      s.circle(ox + mx, my, 15, "#fff", COL[o], 2.5);
      s.text(ox + mx, my + 7, String(o), { size: 20, anchor: "middle", weight: "bold" });
    }
    s.circle(ox + 150, 320, 16, "#555");
  };
  draw(10, cf, "centrifugal", "+1 at each branch point");
  draw(330, sh, "Strahler", "tips 1; +1 if equals meet");
  console.log(`fig-orders: root centrifugal ${cf.get("r")}, Strahler ${sh.get("r")}; max centrifugal ${Math.max(...cf.values())}`);
  s.out("skel-neurovessel/fig-orders.svg");
}

// ---------------------------------------------------------------- fig-vesselmeasures.svg
// Vessel preset: skeleton coloured by radius (DT), and one edge with its path length and chord.
{
  const seg = segment(vesselness(vgen.image, SEG_V).V, SEG_V), m = measure(vgen, seg);
  const sk = thicken(thicken(seg.skel));
  let rmax = 0; for (let i = 0; i < N * N; i++) if (seg.skel[i]) rmax = Math.max(rmax, seg.dt[i]);
  // radius per thick pixel: from the nearest skeleton pixel (itself or left/up neighbour)
  const rad = (i: number) => { for (const o of [0, -1, -N, -N - 1, -2, -2 * N]) if (seg.skel[i + o]) return seg.dt[i + o]; return 0; };
  // the edge with the largest tortuosity among long edges
  const cand = seg.graph.edges.map(e => ({ e, L: pathLength(e.path, N, "corrected"), c: chord(e.path, N) })).filter(o => o.c >= 40).sort((a, b) => b.L / b.c - a.L / a.c);
  const E = cand[0];
  const D = 400, f = D / N, s = new Svg(D + 200, D + 70);
  imageUp(s, 10, 10, D, D, rgbOf(i => (sk[i] ? viridis(Math.min(1, rad(i) / 8)) : seg.mask[i] ? [228, 230, 236] : [255, 255, 255])), N, N, 2);
  const P = (p: number) => [10 + ((p % N) + 0.5) * f, 10 + (Math.floor(p / N) + 0.5) * f];
  let d = "";
  E.e.path.forEach((p, k) => { const [x, y] = P(p); d += (k ? "L" : "M") + x.toFixed(1) + "," + y.toFixed(1); });
  s.add(`<path d="${d}" fill="none" stroke="${K.orange}" stroke-width="5" stroke-opacity="0.85" stroke-linejoin="round"/>`);
  const a = P(E.e.path[0]), b = P(E.e.path[E.e.path.length - 1]);
  s.line(a[0], a[1], b[0], b[1], "#222", 2.5, "7 5");
  // colour bar
  const bx = D + 40, byy = 40, bh = 200;
  for (let k = 0; k < bh; k++) { const c = viridis(1 - k / bh); s.rect(bx, byy + k, 24, 1.2, `rgb(${c[0]},${c[1]},${c[2]})`); }
  s.rect(bx, byy, 24, bh, "none", { stroke: "#888" });
  s.text(bx, byy - 12, "radius (DT)", { size: 21 });
  for (const v of [0, 2, 4, 6, 8]) s.text(bx + 32, byy + bh - v / 8 * bh + 6, String(v), { size: 20 });
  s.line(bx, 300, bx + 30, 300, K.orange, 5); s.text(bx + 38, 306, "path L", { size: 21 });
  s.line(bx, 332, bx + 30, 332, "#222", 2.5, "7 5"); s.text(bx + 38, 338, "chord C", { size: 21 });
  s.text(bx, 372, `τ = L / C = ${(E.L / E.c).toFixed(2)}`, { size: 21 });
  s.text(10, D + 44, `L = ${E.L.toFixed(0)}, C = ${E.c.toFixed(0)}; mean radius ${m.radius.toFixed(2)} px`, { size: 21 });
  console.log(`fig-vesselmeasures: edge L ${E.L.toFixed(1)} C ${E.c.toFixed(1)} tau ${(E.L / E.c).toFixed(3)}; mean radius ${m.radius.toFixed(2)}, mean tort ${m.tortuosity.toFixed(3)}; rmax ${rmax.toFixed(1)}`);
  s.out("skel-neurovessel/fig-vesselmeasures.svg");
}

// ---------------------------------------------------------------- fig-truth.svg
// Noise preset (neuron, noise 14): missed and spurious branch points, without and with pruning.
{
  const gen = generate({ ...NEURON, noise: 14 }), V = vesselness(gen.image, SEG_N).V, t = truthMeasures(gen);
  const D = 260, s = new Svg(2 * D + 50, D + 130);
  [0, 8].forEach((k, j) => {
    const seg = segment(V, { ...SEG_N, k }), m = measure(gen, seg), mt = matchPoints(t.branchXY, m.branchXY, 8);
    const x0 = 10 + j * (D + 30), f = D / N, sk = thicken(seg.skel);
    s.text(x0, 22, k ? `pruned (k = ${k})` : "no pruning", { size: 22 });
    imageUp(s, x0, 34, D, D, rgbOf(i => (sk[i] ? hex(K.obj) : seg.mask[i] ? [222, 226, 236] : [255, 255, 255])), N, N, 2);
    for (const [x, y] of mt.missed) s.circle(x0 + (x + 0.5) * f, 34 + (y + 0.5) * f, 7, "none", K.junction, 2.5);
    for (const [x, y] of mt.spurious) s.rect(x0 + (x + 0.5) * f - 6, 34 + (y + 0.5) * f - 6, 12, 12, "none", { stroke: K.orange, sw: 2.5 });
    s.text(x0, 34 + D + 26, `${m.branchPoints} branch points, ${m.ends} ends`, { size: 21 });
    s.text(x0, 34 + D + 50, `${mt.missed.length} missed, ${mt.spurious.length} spurious`, { size: 21 });
    console.log(`fig-truth: k ${k}: bp ${m.branchPoints} ends ${m.ends} (truth ${t.branchPoints}, ${t.ends}); missed ${mt.missed.length} spurious ${mt.spurious.length}`);
  });
  const ly = D + 118;
  s.circle(18, ly - 6, 7, "none", K.junction, 2.5); s.text(32, ly, "missed (true, not found)", { size: 20 });
  s.rect(262, ly - 12, 12, 12, "none", { stroke: K.orange, sw: 2.5 }); s.text(282, ly, "spurious (found, not true)", { size: 20 });
  s.H = ly + 10;
  console.log(`fig-truth: truth bp ${t.branchPoints} ends ${t.ends}`);
  s.out("skel-neurovessel/fig-truth.svg");
}

// ---------------------------------------------------------------- fig-treefile.svg
// A traced tree as a list of points with parent links.
{
  const pts: [number, number, number, number][] = [ // x, y, r, parent
    [60, 250, 6, -1], [110, 215, 3, 0], [160, 185, 2.5, 1], [200, 130, 2, 2], [225, 70, 1.5, 3], [215, 200, 2, 2], [270, 215, 1.5, 5],
  ];
  const s = new Svg(640, 310);
  for (let i = 1; i < pts.length; i++) { const p = pts[i], q = pts[p[3]]; s.arrow(p[0], p[1], q[0], q[1], "#888", 2); }
  pts.forEach(([x, y, r], i) => {
    s.circle(x, y, 4 + 3 * r, i === 0 ? "#555" : K.mask, "#2a3346", 1.5);
    s.text(x + (i === 0 ? 0 : 16), y + (i === 0 ? 7 : -14), String(i + 1), { size: 21, anchor: i === 0 ? "middle" : "start", color: i === 0 ? "#fff" : C.fg, weight: "bold" });
  });
  s.text(20, 292, "arrows point to the parent", { size: 20, color: "#555" });
  const tx = 320, ty = 40, cw = [46, 60, 60, 56, 82];
  const head = ["id", "x", "y", "r", "parent"];
  let x = tx;
  head.forEach((h, k) => { s.text(x + cw[k] - 6, ty, h, { size: 21, anchor: "end", weight: "bold" }); x += cw[k]; });
  s.line(tx, ty + 8, tx + cw.reduce((a, b) => a + b), ty + 8, "#888", 1.2);
  pts.forEach(([px, py, r, par], i) => {
    const row = [String(i + 1), String(px), String(300 - py), r.toFixed(1), par < 0 ? "−1" : String(par + 1)];
    let xx = tx;
    row.forEach((v, k) => { s.text(xx + cw[k] - 6, ty + 36 + i * 33, v, { size: 21, anchor: "end" }); xx += cw[k]; });
  });
  s.out("skel-neurovessel/fig-treefile.svg");
}
