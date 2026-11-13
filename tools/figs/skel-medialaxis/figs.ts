// Static figures for skel-medialaxis/lec.qmd (SVG). Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/skel-medialaxis/figs.ts --bundle --platform=node --log-level=warning | node
// Uses the medialaxis demo's own shapes and analysis (demos/src/medialaxis/shapes.ts), so the
// pictures and numbers match the demo presets. Numbers quoted on the slides are printed.
import { Svg, C, K, Panel, pixGrid, imageUp, hex } from "../skel-thinning/svglib";
import { SHAPES, BASE, State, analyseShape, N, ShapeId } from "../../../demos/src/medialaxis/shapes";
import { metricDist, Metric } from "../../../demos/src/lib/skeleton";

const st = (s: Partial<State>): State => ({ ...BASE, ...s });
const mix = (a: [number, number, number], b: [number, number, number], t: number): [number, number, number] =>
  [Math.round(a[0] + (b[0] - a[0]) * t), Math.round(a[1] + (b[1] - a[1]) * t), Math.round(a[2] + (b[2] - a[2]) * t)];

/** Crop of the 256 × 256 demo canvas, coloured per pixel. */
function crop(x0: number, y0: number, w: number, h: number, f: (i: number) => [number, number, number]) {
  const rgb = new Uint8Array(3 * w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) rgb.set(f((y0 + y) * N + x0 + x), 3 * (y * w + x));
  return rgb;
}
/** Thicken a set of pixels by one pixel (for visibility of a 1-pixel axis at small display scale). */
function thicken(a: Uint8Array): Uint8Array {
  const o = a.slice();
  for (let y = 1; y < N - 1; y++) for (let x = 1; x < N - 1; x++) if (a[y * N + x]) { o[y * N + x + 1] = 1; o[(y + 1) * N + x] = 1; }
  return o;
}

// ---------------------------------------------------------------- fig-grassfire.svg
{
  const A = analyseShape(SHAPES.dumbbell.make(), st({ shape: "dumbbell", method: "thinning" }));
  const x0 = 16, y0 = 76, w = 224, h = 104, f = 2.6;
  const ax = thicken(A.axis), step = 8;
  const front = new Uint8Array(N * N);
  for (let y = 1; y < N - 1; y++) for (let x = 1; x < N - 1; x++) {
    const i = y * N + x;
    if (!A.mask[i]) continue;
    const k = Math.floor(A.dt[i] / step);
    if (Math.floor(A.dt[i + 1] / step) !== k || Math.floor(A.dt[i + N] / step) !== k) front[i] = 1;
  }
  const rgb = crop(x0, y0, w, h, i => (ax[i] ? hex(K.axis) : front[i] ? hex(K.disc) : A.mask[i] ? mix(hex("#fde0c5"), hex("#fdae6b"), Math.min(1, A.dt[i] / A.maxDT)) : [255, 255, 255]));
  const s = new Svg(Math.round(w * f) + 20, Math.round(h * f) + 70);
  imageUp(s, 10, 10, w * f, h * f, rgb, w, h, 3);
  const ly = h * f + 44;
  s.line(14, ly - 6, 44, ly - 6, K.disc, 4); s.text(52, ly, `fire front every ${step} time steps`, { size: 21 });
  s.line(350, ly - 6, 380, ly - 6, K.axis, 4); s.text(388, ly, "where fronts meet", { size: 21 });
  console.log(`fig-grassfire: max DT ${A.maxDT.toFixed(2)}, ends ${A.ends}, branches ${A.branches}`);
  s.out("skel-medialaxis/fig-grassfire.svg");
}

// ---------------------------------------------------------------- fig-metrics.svg
{
  const r = 5, n = 2 * r + 1, cell = 19, s = new Svg(700, 310);
  const ms: [Metric, string, string][] = [["cityblock", "city-block", "diamond"], ["chessboard", "chessboard", "square"], ["euclidean", "Euclidean", "round disc"]];
  ms.forEach(([m, name, shape], k) => {
    const x0 = 20 + k * 230, y0 = 40;
    s.text(x0 + n * cell / 2, 26, name, { anchor: "middle", size: 22 });
    let cnt = 0;
    pixGrid(s, x0, y0, n, n, cell, (x, y) => {
      const d = metricDist(x - r, y - r, m);
      if (x === r && y === r) return K.axis;
      if (d <= r + 1e-9) { cnt++; return K.disc; }
      return null;
    });
    s.text(x0 + n * cell / 2, y0 + n * cell + 28, `a ${shape}`, { anchor: "middle", size: 21 });
    console.log(`fig-metrics: ${name} radius ${r}: ${cnt + 1} pixels`);
  });
  s.text(20, 300, `all pixels within distance ${r} of the centre (red)`, { size: 20, color: "#555" });
  s.out("skel-medialaxis/fig-metrics.svg");
}

// ---------------------------------------------------------------- fig-ridges.svg
{
  const A = analyseShape(SHAPES.rect.make(), st({ shape: "rect", method: "ridges", lambda: 8 }));
  const x0 = 34, y0 = 86, w = 188, h = 84, f = 2.0;
  const ax = thicken(A.axis);
  const rgb = crop(x0, y0, w, h, i => (ax[i] ? hex(K.axis) : A.mask[i] ? mix(hex("#eef2f8"), hex("#2a3346"), A.dt[i] / A.maxDT) : [255, 255, 255]));
  const s = new Svg(580, 470);
  imageUp(s, 10, 10, w * f, h * f, rgb, w, h, 3);
  // cross-section at x = 128
  const xc = 128;
  const X = 10 + (xc - x0 + 0.5) * f;
  s.line(X, 10, X, 10 + h * f, K.orange, 3, "8 5");
  s.text(X + 8, 30, "section", { size: 20, color: "#b85c00" });
  const ys: number[] = [], vs: number[] = [];
  let top = 0;
  for (let y = 88; y <= 168; y++) { ys.push(y); vs.push(A.dt[y * N + xc]); if (A.dt[y * N + xc] > top) top = A.dt[y * N + xc]; }
  const P = new Panel(s, 70, 230, 490, 170, [88, 168], [0, 40]);
  P.axes([92, 110, 128, 146, 164], [0, 10, 20, 30, 40], "y (pixels) along the section", "DT");
  P.line(ys, vs, K.orange, 3);
  P.hline(top, "#999", 1.5, "5 4");
  P.text(166, top + 2, `top ${top.toFixed(1)} = w / 2`, { size: 21, anchor: "end" });
  const lx = 10 + w * f + 14;
  s.text(lx, 40, "DT: dark =", { size: 21 });
  s.text(lx, 64, "far from the", { size: 21 });
  s.text(lx, 88, "boundary", { size: 21 });
  s.text(lx, 128, "red: ridge =", { size: 21, color: K.axis });
  s.text(lx, 152, "medial axis", { size: 21, color: K.axis });
  console.log(`fig-ridges: rect width ${164 - 92}, DT top at x=128: ${top}`);
  s.out("skel-medialaxis/fig-ridges.svg");
}

// ---------------------------------------------------------------- fig-thinaxis.svg
{
  const shape: ShapeId = "star";
  const T = analyseShape(SHAPES[shape].make(), st({ shape, method: "thinning" }));
  const M = analyseShape(SHAPES[shape].make(), st({ shape, method: "ridges", lambda: 0 }));
  const x0 = 14, y0 = 20, w = 230, h = 232, f = 1.3;
  const s = new Svg(2 * Math.round(w * f) + 40, Math.round(h * f) + 120);
  const panel = (A: typeof T, xx: number, title: string, l1: string, l2: string) => {
    const ax = thicken(A.axis);
    const rgb = crop(x0, y0, w, h, i => (ax[i] ? hex(K.axis) : A.mask[i] && !A.recon[i] ? hex(K.removed2) : A.mask[i] ? hex(K.mask) : [255, 255, 255]));
    s.text(xx, 22, title, { size: 22 });
    imageUp(s, xx, 34, w * f, h * f, rgb, w, h, 3);
    s.text(xx, 34 + h * f + 28, l1, { size: 21 });
    s.text(xx, 34 + h * f + 52, l2, { size: 21 });
  };
  panel(T, 10, "thinning", `${T.pieces} piece, ${T.ends} end points`, `discs miss ${T.missing} px (orange)`);
  panel(M, 30 + w * f, "maximal discs (Euclidean)", `${M.pieces} pieces`, "discs rebuild the star exactly");
  console.log(`fig-thinaxis: thinning pieces ${T.pieces} ends ${T.ends} missing ${T.missing} extra ${T.extra} axis ${T.nAxis}; maximal pieces ${M.pieces} missing ${M.missing} extra ${M.extra} axis ${M.nAxis}`);
  s.out("skel-medialaxis/fig-thinaxis.svg");
}

// ---------------------------------------------------------------- fig-bump.svg
{
  const a0 = analyseShape(SHAPES.ellipse.make(), st({ shape: "ellipse", method: "thinning", bump: 0, bumpPos: 0.2 }));
  const a1 = analyseShape(SHAPES.ellipse.make(), st({ shape: "ellipse", method: "thinning", bump: 3, bumpPos: 0.2 }));
  const x0 = 24, y0 = 68, w = 208, h = 124, f = 1.45;
  const s = new Svg(2 * Math.round(w * f) + 40, Math.round(h * f) + 90);
  const panel = (A: typeof a0, xx: number, title: string) => {
    const ax = thicken(A.axis);
    const rgb = crop(x0, y0, w, h, i => (ax[i] ? hex(K.axis) : A.mask[i] ? hex(K.mask) : [255, 255, 255]));
    s.text(xx, 22, title, { size: 22 });
    imageUp(s, xx, 34, w * f, h * f, rgb, w, h, 3);
    s.text(xx, 34 + h * f + 28, `${A.ends} end points, ${A.branches} branches`, { size: 21 });
  };
  panel(a0, 10, "smooth ellipse");
  panel(a1, 30 + w * f, "with a bump of radius 3 px");
  console.log(`fig-bump: no bump ends ${a0.ends} branches ${a0.branches}; bump ends ${a1.ends} branches ${a1.branches}`);
  s.out("skel-medialaxis/fig-bump.svg");
}
