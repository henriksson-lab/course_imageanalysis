// moments mode "moments" (unit regionprops/): region properties of the digitised shape.
// Overlays: centroid, bounding box, equivalent ellipse (semi-axes 2√λ1, 2√λ2), the two
// eigenvectors as arrows of length 2√λi. Table: area, centroid, central moments, S, λ1, λ2,
// orientation, eccentricity, elongation, three perimeter estimates against the exact
// outline, objects − holes = Euler number (and Gray's bit-quads). Plot: perimeter estimate /
// exact perimeter against the number of pixels across the shape.
import { htmlEl, checkbox } from "../lib/ui";
import { Plot } from "../lib/plot";
import { label2d } from "../lib/label";
import { rasterise, scalePoly, rotatePoly, ellipsePoly, SHAPE_PRESETS, Poly } from "../lib/shapecanvas";
import { eigSym2 } from "../lib/linalg";
import { registerMode, MomentsContext, MomentsModeSpec, Digitised } from "./registry";
import { regionProps, edgePerimeter, chainPerimeter, croftonPerimeter, bitQuads, countHoles } from "./regionprops";
import { exactMeasures, disc, rect } from "./shapes";
import { sline, spoly, scross, sarrow, table, f2, P2 } from "./draw";

const SWEEP = [8, 6, 4, 3, 2, 1.5, 1, 0.75, 0.5, 0.35, 0.25];
const C1 = "var(--qb-c2, #d62728)", C2 = "var(--qb-c3, #2ca02c)", CB = "var(--qb-c5, #9467bd)", CE = "var(--qb-c4, #ff7f0e)";

/** The exercise object (regionprops B1): 12 pixels in a 5 × 5 box, as one orthogonal
 *  polygon in pixel units (pixel (x, y) covers [x, x+1] × [y, y+1]), offset by one pixel. */
export const HAND_OUTLINE: [number, number][] = [[1, 1], [3, 1], [3, 2], [4, 2], [4, 3], [5, 3], [5, 4], [6, 4], [6, 5], [5, 5], [5, 6], [4, 6], [4, 5], [3, 5], [3, 4], [2, 4], [2, 3], [1, 3]];

registerMode({
  id: "moments",
  build(ctx: MomentsContext): MomentsModeSpec {
    const show = { box: true, ellipse: true, axes: true };
    htmlEl("div", "mo-lab", "show", ctx.controls);
    const ck = htmlEl("div", "mo-checks", "", ctx.controls);
    checkbox(ck, "box", show.box, v => { show.box = v; ctx.refresh(); });
    checkbox(ck, "ellipse", show.ellipse, v => { show.ellipse = v; ctx.refresh(); });
    checkbox(ck, "eigenvectors", show.axes, v => { show.axes = v; ctx.refresh(); });

    const tab = htmlEl("div", "", "", ctx.out);
    const plotBox = htmlEl("div", "", "", ctx.out);
    const plot = new Plot(plotBox, { width: 520, height: 250, x: [2, 250], xlog: true, y: [0.7, 1.45], xlabel: "pixels across the shape", ylabel: "estimate / exact", font: 17, xticks: [2, 5, 10, 20, 50, 100, 200], xfmt: v => String(v) });
    plot.hline(1, { color: "var(--qb-axis, #444)", width: 1.5 });
    plot.hline(4 / Math.PI, { color: "var(--qb-muted, #999)", width: 1.5, dash: "5 4", text: "4/π" });
    const lE = plot.line([], [], { color: C1, label: "edge count" });
    const lC = plot.line([], [], { color: C2, label: "chain code" });
    const lK = plot.line([], [], { color: CB, label: "Crofton" });
    const now = plot.vline(10, { color: "var(--qb-muted, #999)", width: 1.5, dash: "3 3" });
    plot.legend(undefined, "tr");
    let lastSweepKey = "";

    function sweep(polys: Poly[], exactP: number, exactA: number): void {
      const key = JSON.stringify(polys);
      if (key === lastSweepKey) return;
      lastSweepKey = key;
      const xs: number[] = [], e: number[] = [], c: number[] = [], k: number[] = [];
      for (const p of SWEEP) {
        const w = Math.ceil(64 / p), m = rasterise(polys.map(q => scalePoly(q, 1 / p, 1 / p, [0, 0])), w, w);
        if (!m.some(v => v)) continue;
        xs.push(Math.sqrt(4 * exactA / Math.PI) / p);
        e.push(edgePerimeter(m, w, w) * p / exactP); c.push(chainPerimeter(m, w, w) * p / exactP); k.push(croftonPerimeter(m, w, w) * p / exactP);
      }
      lE.set(xs, e); lC.set(xs, c); lK.set(xs, k);
    }

    function update(d: Digitised): void {
      const g = ctx.overlay;
      g.replaceChildren();
      const lab = label2d(d.mask, d.w, d.h, 8);
      const props = regionProps(lab.data, d.w, d.h, lab.count);
      const P = (x: number, y: number): P2 => [(x + 0.5) * d.px, (y + 0.5) * d.py];
      for (const r of props) {
        const c = P(r.cx, r.cy);
        if (show.box) spoly(g, [[r.x0 * d.px, r.y0 * d.py], [(r.x1 + 1) * d.px, r.y0 * d.py], [(r.x1 + 1) * d.px, (r.y1 + 1) * d.py], [r.x0 * d.px, (r.y1 + 1) * d.py]], "var(--qb-muted, #777)", 0.3, "none", "1 0.8");
        const a = 2 * Math.sqrt(r.l1) * d.px, b = 2 * Math.sqrt(r.l2) * d.px, v1 = r.v1, v2: P2 = [-v1[1], v1[0]];
        if (show.ellipse) spoly(g, Array.from({ length: 72 }, (_, i) => { const t = 2 * Math.PI * i / 72; return [c[0] + a * Math.cos(t) * v1[0] + b * Math.sin(t) * v2[0], c[1] + a * Math.cos(t) * v1[1] + b * Math.sin(t) * v2[1]] as P2; }), CE, 0.45);
        if (show.axes) { sarrow(g, c, [c[0] + a * v1[0], c[1] + a * v1[1]], C1); sarrow(g, c, [c[0] + b * v2[0], c[1] + b * v2[1]], C2); }
        scross(g, c, 1.2, "var(--qb-fg, #222)", 0.4);
      }
      const ex = exactMeasures(d.polys);
      const rows: (string[] | string)[] = [];
      const big = props.slice().sort((p, q) => q.area - p.area)[0];
      const holes = countHoles(d.mask, d.w, d.h, 8), q = bitQuads(d.mask, d.w, d.h);
      if (big) {
        const one = props.length === 1 && ex;
        const s = d.px;
        const exE = one ? eigSym2(ex!.s11, ex!.s12, ex!.s22) : null;
        rows.push(props.length > 1 ? `largest of ${props.length} objects (pixels of size ${s})` : `the object (pixels of size ${s})`);
        rows.push(["", "digitised", one ? "exact" : ""]);
        rows.push(["area m₀₀", big.area.toFixed(0), one ? f2(ex!.area / (s * s), 1) : ""]);
        rows.push(["centroid (x̄, ȳ)", `(${f2(big.cx)}, ${f2(big.cy)})`, one ? `(${f2(ex!.cx / s - 0.5)}, ${f2(ex!.cy / s - 0.5)})` : ""]);
        rows.push(["μ₂₀, μ₁₁, μ₀₂", `${big.mu20.toFixed(0)}, ${big.mu11.toFixed(0)}, ${big.mu02.toFixed(0)}`, ""]);
        rows.push(["S₁₁, S₁₂, S₂₂ (S = μ/m₀₀)", `${f2(big.s11)}, ${f2(big.s12)}, ${f2(big.s22)}`, one ? `${f2(ex!.s11 / s / s)}, ${f2(ex!.s12 / s / s)}, ${f2(ex!.s22 / s / s)}` : ""]);
        rows.push(["λ₁, λ₂", `${f2(big.l1)}, ${f2(big.l2)}`, exE ? `${f2(exE.l1 / s / s)}, ${f2(exE.l2 / s / s)}` : ""]);
        const deg = (t: number) => { let o = -t * 180 / Math.PI; if (o <= -90) o += 180; return o; };
        const iso = big.l1 - big.l2 < 1e-3 * big.l1;
        rows.push(["orientation θ (° ⟲)", iso ? "undefined" : f2(big.orientation, 1), exE && exE.l1 - exE.l2 > 1e-3 * exE.l1 ? f2(deg(exE.angle), 1) : ""]);
        rows.push(["eccentricity √(1 − λ₂/λ₁)", f2(big.eccentricity), exE ? f2(Math.sqrt(Math.max(0, 1 - exE.l2 / exE.l1))) : ""]);
        rows.push(["elongation √(λ₁/λ₂)", f2(big.elongation), exE ? f2(Math.sqrt(exE.l1 / exE.l2)) : ""]);
        const ratio = (v: number) => (one ? ` (× ${f2(v / (ex!.perimeter / s), 3)})` : "");
        rows.push(["perimeter: edge count", f2(big.perimEdges, 1) + ratio(big.perimEdges), one ? f2(ex!.perimeter / s, 1) : ""]);
        rows.push(["perimeter: chain code", f2(big.perimChain, 1) + ratio(big.perimChain), ""]);
        rows.push(["perimeter: Crofton", f2(big.perimCrofton, 1) + ratio(big.perimCrofton), ""]);
      } else rows.push("no object pixels");
      rows.push(["Euler: objects − holes", `${lab.count} − ${holes} = ${lab.count - holes}`, ""]);
      rows.push(["Euler: (Q₁ − Q₃ − 2Q_D)/4", `(${q.q1} − ${q.q3} − 2·${q.qd})/4 = ${(q.q1 - q.q3 - 2 * q.qd) / 4}`, ""]);
      table(tab, rows);
      if (ex) {
        now.set(Math.max(2, Math.min(250, Math.sqrt(4 * ex.area / Math.PI) / d.px)));
        if (d.final) sweep(d.polys, ex.perimeter, ex.area);
      }
    }

    const ell = (ang: number): Poly[] => [ellipsePoly(32, 32, 22, 10, 96, ang)];
    return {
      presets: [
        { id: "ellipse", label: "rotate the ellipse", shapes: () => ell(0.5) },
        { id: "square", label: "a square: equal eigenvalues", shapes: () => [rotatePoly(rect(18, 18, 46, 46), 0.35)] },
        { id: "L", label: "the L: the ellipse is not the shape", shapes: s => SHAPE_PRESETS.find(p => p.id === "L")!.make(s) },
        { id: "ring", label: "a ring: Euler number 0", shapes: () => [disc(32, 32, 22), disc(32, 32, 11)] },
        { id: "two", label: "two objects, one hole", shapes: () => [disc(22, 24, 15), disc(22, 24, 7), ellipsePoly(49, 46, 10, 6, 48, -0.6)] },
        { id: "disc", label: "a digital disc: 8r, not 2πr", shapes: () => [disc(32, 32, 20)], pixel: 2 },
        { id: "hand", label: "12 pixels (exercise)", shapes: () => [HAND_OUTLINE.map(([x, y]) => [8 * x, 8 * y] as [number, number])], pixel: 8 },
      ],
      update,
    };
  },
});
