// Slide figures for bvh/ (second pass: pictures on the slides), computed with lib/bvh.ts and the
// bvhstep demo's scenes. Also writes tools/figs/bvh/cost.json for fig_cost.py.
// Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/bvh/figs.ts --bundle --platform=node --log-level=warning | node
//   python3 -I tools/figs/bvh/fig_cost.py
import * as B from "../../../demos/src/lib/bvh";
import { rng } from "../../../demos/src/lib/random";
import { makeScene, primBox, primHit, Prim } from "../../../demos/src/bvhstep/scene";
import { save, svg, txt, line, circ, C } from "../raycast/util";

const drawPrims = (prims: Prim[], X: (x: number) => number, Y: (y: number) => number, s: number, tested = new Set<number>(), hi = new Set<number>()) => prims.map((p, i) => {
  const t = tested.has(i), h = hi.has(i);
  const st = h ? `fill="${C.orange}" stroke="#000" stroke-width="2"` : t ? `fill="#9ecae1" stroke="#000" stroke-width="1.6"` : `fill="#c9d3dd" stroke="#5b6b7b" stroke-width="1"`;
  return p.kind === 0 ? circ(X(p.a[0]), Y(p.a[1]), Math.max(1.2, p.a[2] * s), st)
    : line(X(p.a[0]), Y(p.a[1]), X(p.a[2]), Y(p.a[3]), h ? `stroke="${C.orange}" stroke-width="4"` : t ? `stroke="#000" stroke-width="3"` : `stroke="#5b6b7b" stroke-width="2"`);
}).join("");

// ---------------------------------------------------------------- tests per ray against N (the demo's plot, uniform scene)
{
  const NS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], r = rng(99), out: { n: number; bvh: number; box: number; prim: number }[] = [];
  for (const n of NS) {
    const pr = makeScene(n, "uniform"), bv = B.buildBVH(pr.map(primBox), { method: "sah" });
    let sb = 0, sp = 0; const R = 1000;
    for (let i = 0; i < R; i++) {
      const a = 2 * Math.PI * r.unif(), o = [50 + 70 * Math.cos(a), 50 + 70 * Math.sin(a)], t = [100 * r.unif(), 100 * r.unif()];
      const L = Math.hypot(t[0] - o[0], t[1] - o[1]), d = [(t[0] - o[0]) / L, (t[1] - o[1]) / L];
      const b = B.traverseBVH(bv, o, d, (p, tm) => primHit(pr[p], o, d, tm)); sb += b.boxTests; sp += b.primTests;
    }
    out.push({ n, bvh: (sb + sp) / R, box: sb / R, prim: sp / R });
  }
  console.log(out.map(o => `${o.n}: ${o.bvh.toFixed(1)}`).join("  "));
  save("tools/figs/bvh/cost.json", JSON.stringify(out));
}

// ---------------------------------------------------------------- median vs SAH on the clustered scene (preset "sah")
{
  const prims = makeScene(256, "clustered"), boxes = prims.map(primBox), P = 250, s = P / 100;
  const X = (x: number) => x * s, Y = (y: number) => P - y * s;
  let b = "";
  ([["median", 0], ["sah", 1]] as [ "median" | "sah", number][]).forEach(([m, k]) => {
    const bvh = B.buildBVH(boxes, { method: m }), cost = B.bvhCost(bvh), x0 = k * (P + 30);
    console.log(m, "cost", cost.toFixed(1), B.bvhStats(bvh));
    let g = `<rect x="0" y="0" width="${P}" height="${P}" fill="#fcfcfd" stroke="#999"/>`;
    g += drawPrims(prims, X, Y, s);
    const cols = [C.black, C.blue, C.red];
    bvh.nodes.forEach(n => { if (n.depth < 1 || n.depth > 2) return; g += `<rect x="${X(n.min[0]).toFixed(1)}" y="${Y(n.max[1]).toFixed(1)}" width="${((n.max[0] - n.min[0]) * s).toFixed(1)}" height="${((n.max[1] - n.min[1]) * s).toFixed(1)}" fill="none" stroke="${cols[n.depth]}" stroke-width="${n.depth === 1 ? 3 : 2}" ${n.depth === 2 ? 'stroke-dasharray="6 3"' : ""}/>`; });
    b += `<g transform="translate(${x0},40)">${g}</g>`;
    b += txt(x0, 28, m === "sah" ? `SAH: cost ${cost.toFixed(0)}` : `median: cost ${cost.toFixed(0)}`, `font-weight="600"`);
  });
  b += txt(0, P + 74, "levels 1 (solid) and 2 (dashed)", `font-size="21"`);
  save("bvh/fig-split.svg", svg(2 * P + 30, P + 84, b, 22));
}

// ---------------------------------------------------------------- a grid in a stadium (preset "stadium")
{
  const prims = makeScene(1024, "stadium"), boxes = prims.map(primBox), c = 16, g = B.buildGrid(boxes, [0, 0], [100, 100], [c, c]);
  const o = [8, 10], aim = [54, 46], L = Math.hypot(aim[0] - o[0], aim[1] - o[1]), d = [(aim[0] - o[0]) / L, (aim[1] - o[1]) / L];
  const hit = (p: number, tm: number) => primHit(prims[p], o, d, tm);
  const res = B.traverseGrid(g, o, d, hit, Infinity, true);
  const bvh = B.buildBVH(boxes, { method: "sah" }), rb = B.traverseBVH(bvh, o, d, hit);
  console.log("stadium grid: cells", res.cells, "prim tests", res.primTests, "| BVH box", rb.boxTests, "prim", rb.primTests);
  const P = 440, s = P / 100, X = (x: number) => x * s, Y = (y: number) => P - y * s;
  let b = `<rect x="0" y="0" width="${P}" height="${P}" fill="#fcfcfd" stroke="#999"/>`;
  const tested = new Set<number>();
  res.trace.forEach(e => { if (e.kind === "cell") b += `<rect x="${X(100 * e.cell[0] / c)}" y="${Y(100 * (e.cell[1] + 1) / c)}" width="${100 / c * s}" height="${100 / c * s}" fill="${C.yellow}" fill-opacity="0.55"/>`; else if (e.kind === "prim") tested.add(e.id); });
  for (let i = 0; i <= c; i++) b += line(X(100 * i / c), 0, X(100 * i / c), P, `stroke="#000" stroke-opacity="0.18"`) + line(0, Y(100 * i / c), P, Y(100 * i / c), `stroke="#000" stroke-opacity="0.18"`);
  b += drawPrims(prims, X, Y, s, tested);
  b += line(X(o[0]), Y(o[1]), X(o[0] + res.t * d[0]), Y(o[1] + res.t * d[1]), `stroke="${C.orange}" stroke-width="3"`) + circ(X(o[0] + res.t * d[0]), Y(o[1] + res.t * d[1]), 5, `fill="${C.red}"`);
  b += circ(X(o[0]), Y(o[1]), 6, `fill="#fff" stroke="#000" stroke-width="2"`);
  b += txt(0, P + 30, `grid 16 × 16: ${res.cells} cells, ${res.primTests} object tests`, `font-size="21"`);
  b += txt(0, P + 58, `SAH BVH: ${rb.boxTests} box + ${rb.primTests} object tests`, `font-size="21"`);
  save("bvh/fig-grid.svg", svg(P, P + 68, b, 22));
}

// ---------------------------------------------------------------- objects or space: one small scene, two structures
{
  const prims: Prim[] = [
    { kind: 0, a: [14, 20, 7] }, { kind: 0, a: [30, 14, 6] }, { kind: 1, a: [10, 40, 42, 34] }, { kind: 0, a: [22, 70, 8] },
    { kind: 0, a: [70, 78, 6] }, { kind: 1, a: [58, 86, 88, 64] }, { kind: 0, a: [80, 26, 9] }, { kind: 0, a: [64, 14, 5] },
    { kind: 0, a: [50, 50, 9] },
  ];
  const boxes = prims.map(primBox), bvh = B.buildBVH(boxes, { method: "median", leafSize: 2 });
  const P = 250, s = P / 100, X = (x: number) => x * s, Y = (y: number) => P - y * s;
  const big = 8; // the disc in the middle: one leaf of the BVH, four cells of the grid
  let b = "";
  // left: BVH leaves (may overlap)
  let g = `<rect x="0" y="0" width="${P}" height="${P}" fill="#fcfcfd" stroke="#999"/>`;
  const cols = [C.black, C.blue, C.green, C.red, C.purple];
  bvh.nodes.forEach(n => { if (n.depth < 1) return; g += `<rect x="${X(n.min[0]).toFixed(1)}" y="${Y(n.max[1]).toFixed(1)}" width="${((n.max[0] - n.min[0]) * s).toFixed(1)}" height="${((n.max[1] - n.min[1]) * s).toFixed(1)}" fill="none" stroke="${cols[Math.min(n.depth, 4)]}" stroke-width="${n.depth === 1 ? 3 : 2}" ${n.depth > 1 ? 'stroke-dasharray="6 3"' : ""}/>`; });
  g += drawPrims(prims, X, Y, s, new Set(), new Set([big]));
  b += `<g transform="translate(0,40)">${g}</g>` + txt(0, 28, "BVH: groups objects", `font-weight="600"`);
  // right: 4 × 4 grid
  const c = 4, gr = B.buildGrid(boxes, [0, 0], [100, 100], [c, c]);
  g = `<rect x="0" y="0" width="${P}" height="${P}" fill="#fcfcfd" stroke="#999"/>`;
  gr.cells.forEach((list, lin) => { if (list.includes(big)) { const i = lin % c, j = Math.floor(lin / c); g += `<rect x="${X(100 * i / c)}" y="${Y(100 * (j + 1) / c)}" width="${100 / c * s}" height="${100 / c * s}" fill="${C.orange}" fill-opacity="0.25"/>`; } });
  for (let i = 0; i <= c; i++) g += line(X(100 * i / c), 0, X(100 * i / c), P, `stroke="#000" stroke-opacity="0.35"`) + line(0, Y(100 * i / c), P, Y(100 * i / c), `stroke="#000" stroke-opacity="0.35"`);
  g += drawPrims(prims, X, Y, s, new Set(), new Set([big]));
  const nCells = gr.cells.filter(l => l.includes(big)).length, empty = gr.cells.filter(l => l.length === 0).length;
  b += `<g transform="translate(${P + 30},40)">${g}</g>` + txt(P + 30, 28, "grid: cuts space", `font-weight="600"`);
  b += txt(0, P + 74, `orange object: 1 BVH leaf, but ${nCells} grid cells`, `font-size="21"`);
  console.log("objspace: big in", nCells, "cells; empty", empty);
  save("bvh/fig-objspace.svg", svg(2 * P + 30, P + 84, b, 22));
}
