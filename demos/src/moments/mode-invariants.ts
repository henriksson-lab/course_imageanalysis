// moments mode "invariants" (moments elective, unit moment-invariants/): which numbers survive
// which transformation. A reference copy of the shape is frozen with "set reference" (each
// preset sets one and then moves, scales, rotates or mirrors the shape). The reference is
// digitised on the same pixel grid as the current shape, so the pixel-size slider
// re-digitises both.
// Table: raw m, central μ, normalised η and Hu's φ1..φ7 for reference and current shape,
// with the relative change per row (green < 1 %, red ≥ 10 %). Bars: −sign(φ)·log10|φ| of
// both. Inset: both shapes after pose normalisation (centroid to the middle, area fixed,
// major axis horizontal, μ30 > 0), with a warning when the pose is ambiguous.
import { htmlEl, checkbox, button, svgEl } from "../lib/ui";
import { rasterise, scalePoly, rotatePoly, mirrorPoly, translatePoly, Poly } from "../lib/shapecanvas";
import { rawMoments, centralFromRaw, normalisedMoments, huInvariants, logFeature, pose, idx, Table } from "../lib/moments";
import { Plot } from "../lib/plot";
import { registerMode, MomentsContext, MomentsModeSpec, Digitised } from "./registry";
import { sci } from "./mode-higher";
import { polygonCoverageEvenOdd } from "../lib/moments";
import { rect } from "./shapes";

const N = 4;
const CREF = "var(--qb-c4, #ff7f0e)", CCUR = "var(--qb-c1, #1f77b4)";

interface Inv { m: Table; mu: Table; eta: Table; hu: number[]; mask: ArrayLike<number>; w: number; h: number }

/** The L of the presets: two arms of different length, so it has no symmetry at all. */
export const L_SHAPE: Poly = [[18, 14], [30, 14], [30, 38], [46, 38], [46, 50], [18, 50]];

function invOf(mask: ArrayLike<number>, w: number, h: number): Inv | null {
  const m = rawMoments(mask, w, h, N);
  if (!(m[0] > 0)) return null;
  const mu = centralFromRaw(m, N), eta = normalisedMoments(mu, N);
  return { m, mu, eta, hu: huInvariants(eta, N), mask, w, h };
}

/** Change between a and b relative to their size, or to a row scale for values near 0. */
function change(a: number, b: number, scale: number): number {
  const den = Math.max(Math.abs(a), Math.abs(b), 1e-3 * scale);
  return den > 0 ? Math.abs(a - b) / den : 0;
}

function changeColour(c: number): string {
  // green below 1 %, yellow around 3 %, red from 10 %
  const t = Math.min(1, Math.max(0, Math.log10(Math.max(c, 1e-6) / 0.01) / Math.log10(10)));
  const r = Math.round(60 + 180 * t), g = Math.round(170 - 110 * t), b = 70;
  return `rgba(${r},${g},${b},${c < 0.01 ? 0.28 : 0.35 + 0.3 * t})`;
}

registerMode({
  id: "invariants",
  build(ctx: MomentsContext): MomentsModeSpec {
    let refPolys: Poly[] | null = null;
    let last: Digitised | null = null;
    const st = { inset: true, aa: false };
    const bars = ctx.host.dataset.bars !== "off";
    const rowsOpt = ctx.host.dataset.rows ?? "all";

    htmlEl("div", "mo-lab", "reference", ctx.controls);
    const rr = htmlEl("div", "qb-presets", "", ctx.controls);
    button(rr, "set reference = current shape", () => { if (last) { refPolys = last.polys.map(p => p.map(q => [...q] as [number, number])); ctx.refresh(); } });
    const ck = htmlEl("div", "mo-checks", "", ctx.controls);
    const cI = checkbox(ck, "pose-normalised inset", st.inset, v => { st.inset = v; ctx.refresh(); });
    const cA = checkbox(ck, "grey edges (anti-aliased)", st.aa, v => { st.aa = v; ctx.refresh(); });

    const tabBox = htmlEl("div", "", "", ctx.out);
    const lower = htmlEl("div", "", "", ctx.out);
    lower.style.cssText = "display:flex;gap:.6em;align-items:flex-start;flex-wrap:wrap";
    const plotBox = htmlEl("div", "", "", lower);
    plotBox.style.cssText = "flex:1 1 260px;min-width:220px";
    const insetBox = htmlEl("div", "", "", lower);
    insetBox.style.cssText = "flex:0 0 150px;text-align:center;font-size:14px;color:var(--qb-muted)";
    let plot: Plot | null = null, bRef: ReturnType<Plot["bars"]> | null = null, bCur: ReturnType<Plot["bars"]> | null = null;
    if (bars) {
      plot = new Plot(plotBox, { width: 420, height: 190, x: [0.4, 7.6], y: [0, 12], xticks: [1, 2, 3, 4, 5, 6, 7], xfmt: v => `φ${"₀₁₂₃₄₅₆₇"[v]}`, ylabel: "−sign·log₁₀|φ|", font: 16, margin: { l: 58, b: 34 } });
      plot.hline(0, { color: "var(--qb-axis, #444)", width: 1 });
      bRef = plot.bars([], [], { color: CREF, barWidth: 0.36, label: "reference" });
      bCur = plot.bars([], [], { color: CCUR, barWidth: 0.36, label: "current" });
      const lg = htmlEl("div", "", "", plotBox);
      lg.style.cssText = "font-size:14px;text-align:center";
      lg.innerHTML = `<span style="color:${CREF}">■</span> reference &nbsp; <span style="color:${CCUR}">■</span> current`;
    }
    const insetCan = document.createElement("canvas");
    insetCan.width = insetCan.height = 96;
    insetCan.style.cssText = "width:144px;height:144px;image-rendering:pixelated;border:1px solid var(--qb-grid, #ddd)";
    const insetNote = htmlEl("div", "", "", insetBox);
    insetBox.prepend(insetCan);

    /** Digitise polygons on the current grid: binary (pixel centres) or the fraction of each
     *  pixel inside (4 × 4 samples). */
    function digitise(polys: Poly[], d: Digitised): Inv | null {
      const sc = polys.map(p => scalePoly(p, 1 / d.px, 1 / d.py, [0, 0]));
      return invOf(st.aa ? polygonCoverageEvenOdd(sc, d.w, d.h, 4) : rasterise(sc, d.w, d.h), d.w, d.h);
    }

    /** Draw the mask pose-normalised into the inset (nearest neighbour), colour per shape. */
    function drawInset(items: { v: Inv; col: [number, number, number] }[]): string {
      const S = insetCan.width, c2 = insetCan.getContext("2d")!, id = c2.createImageData(S, S);
      let note = "";
      for (const { v, col } of items) {
        const P = pose(v.m, N);
        if (P.ambiguous) note = "ambiguous pose: λ₁ ≈ λ₂ or μ₃₀ ≈ 0";
        // inset pixel (i, j) → normalised coords (u, w) with area 1 mapped to ~ (S/3)² pixels
        const k = S / 3, c = Math.cos(P.theta), s = Math.sin(P.theta), sc = Math.sqrt(P.area);
        for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) {
          const u = (i - S / 2 + 0.5) / k * sc, w = (j - S / 2 + 0.5) / k * sc;
          // rotate by +θ back into the image frame (y down): x = c u − s w, y = s u + c w
          const x = Math.round(P.cx + c * u - s * w), y = Math.round(P.cy + s * u + c * w);
          if (x < 0 || y < 0 || x >= v.w || y >= v.h || !(v.mask[y * v.w + x] >= 0.5)) continue;
          const o = 4 * (j * S + i);
          for (let q = 0; q < 3; q++) id.data[o + q] = id.data[o + 3] ? (id.data[o + q] + col[q]) / 2 : col[q];
          id.data[o + 3] = 200;
        }
      }
      c2.putImageData(id, 0, 0);
      return note;
    }

    function update(d: Digitised): void {
      last = d;
      ctx.overlay.replaceChildren();
      const cur = st.aa ? digitise(d.polys, d) : invOf(d.mask, d.w, d.h), ref = refPolys ? digitise(refPolys, d) : null;
      // reference outline on the canvas
      if (refPolys) for (const p of refPolys) {
        const path = svgEl("path", { d: "M" + p.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join("L") + "Z" }, ctx.overlay);
        path.style.cssText = `fill:none;stroke:${CREF};stroke-width:0.45;stroke-dasharray:1.2 0.8`;
      }
      tabBox.replaceChildren();
      if (!cur) { tabBox.textContent = "no object pixels"; return; }
      const t = htmlEl("table", "mo-tab", "", tabBox);
      const hr = htmlEl("tr", "h", "", t);
      for (const s of ["", "reference", "current", "change"]) htmlEl("td", "", s, hr);
      const sz = Math.sqrt((cur.mu[idx(N, 2, 0)] + cur.mu[idx(N, 0, 2)]) / cur.m[0]);
      const rows: [string, (v: Inv) => number, number][] = [];
      const add = (name: string, f: (v: Inv) => number, scale: number) => rows.push([name, f, scale]);
      const compact = rowsOpt === "compact";
      add("m₁₀", v => v.m[idx(N, 1, 0)], cur.m[0] * 32);
      if (!compact) add("m₂₀", v => v.m[idx(N, 2, 0)], cur.m[0] * 32 * 32);
      add("μ₂₀", v => v.mu[idx(N, 2, 0)], cur.m[0] * sz ** 2);
      if (!compact) add("μ₁₁", v => v.mu[idx(N, 1, 1)], cur.m[0] * sz ** 2);
      add("μ₃₀", v => v.mu[idx(N, 3, 0)], cur.m[0] * sz ** 3);
      add("η₂₀", v => v.eta[idx(N, 2, 0)], cur.hu[0]);
      if (!compact) add("η₁₁", v => v.eta[idx(N, 1, 1)], cur.hu[0]);
      add("η₃₀", v => v.eta[idx(N, 3, 0)], cur.hu[0] ** 1.5);
      const huScale = [1, 2, 3, 3, 6, 4, 6].map(k => cur.hu[0] ** k);
      for (let k = 0; k < 7; k++) if (!compact || k === 0 || k === 1 || k === 2 || k === 6) add(`φ${"₁₂₃₄₅₆₇"[k]}`, v => v.hu[k], huScale[k]);
      for (const [name, f, scale] of rows) {
        const tr = htmlEl("tr", "", "", t);
        htmlEl("td", "", name, tr);
        const a = ref ? f(ref) : NaN, b = f(cur);
        htmlEl("td", "v", ref ? sci(a) : "–", tr);
        htmlEl("td", "v", sci(b), tr);
        const c = htmlEl("td", "v", "", tr);
        if (ref) {
          const ch = change(a, b, scale);
          c.textContent = ch < 1e-4 ? "0 %" : ch >= 0.995 ? "≥ 100 %" : `${(100 * ch).toPrecision(2)} %`;
          c.style.background = changeColour(ch);
        }
      }
      if (bRef && bCur) {
        const xs = [1, 2, 3, 4, 5, 6, 7];
        bCur.set(xs.map(x => x + 0.19), cur.hu.map(logFeature));
        bRef.set(ref ? xs.map(x => x - 0.19) : [], ref ? ref.hu.map(logFeature) : []);
        const all = [...cur.hu, ...(ref ? ref.hu : [])].map(logFeature);
        plot!.setDomain(undefined, [Math.min(0, ...all) - 0.5, Math.max(...all) + 0.5]);
      }
      insetBox.style.display = st.inset ? "" : "none";
      if (st.inset) {
        const items: { v: Inv; col: [number, number, number] }[] = [];
        if (ref) items.push({ v: ref, col: [255, 127, 14] });
        items.push({ v: cur, col: [31, 119, 180] });
        const note = drawInset(items);
        insetNote.textContent = note || "pose-normalised (both)";
        insetNote.style.color = note ? "var(--qb-c2, #d62728)" : "";
      }
    }

    const base = (): Poly => L_SHAPE.map(p => [...p] as [number, number]);
    const setRef = (p: Poly[], aa = false) => () => { refPolys = p.map(q => q.map(v => [...v] as [number, number])); st.inset = true; cI.checked = true; st.aa = aa; cA.checked = aa; };
    return {
      presets: [
        { id: "translate", label: "translate: m changes, μ does not", shapes: () => [translatePoly(base(), 8, 6)], apply: setRef([base()]) },
        { id: "scale", label: "scale: μ changes, η does not", shapes: () => [scalePoly(base(), 1.25, 1.25, [32, 32])], apply: setRef([scalePoly(base(), 0.75, 0.75, [32, 32])]) },
        { id: "rotate", label: "rotate: η changes, φ does not", shapes: () => [rotatePoly(base(), 50 * Math.PI / 180)], apply: setRef([base()]) },
        { id: "mirror", label: "mirror: only φ₇ flips", shapes: () => [mirrorPoly(base(), "x")], apply: setRef([base()]) },
        { id: "small", label: "small shape: discretisation", shapes: () => [rotatePoly(scalePoly(base(), 0.3), 45 * Math.PI / 180)], apply: setRef([scalePoly(base(), 0.3)]) },
        { id: "square", label: "square: pose is ambiguous", shapes: () => [rotatePoly(rect(18, 18, 42, 42), 0.5)], apply: setRef([rect(18, 18, 42, 42)]) },
      ],
      update,
    };
  },
});
