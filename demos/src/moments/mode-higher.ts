// moments mode "higher" (moments elective, unit moments-basis/): moments of any order of
// the digitised shape, read as a 2D distribution.
// Table: the triangle of moments up to order N (data-order, default 4), raw m_pq or central
// μ_pq (toggle), divided by m00 (the moments of f/m00 as a density), each cell coloured by
// sign (red +, blue −) and size (|v|^{1/(p+q)} against the largest of its order). Below:
// mean, spread, skewness and kurtosis of the two marginals, and the effect of counting a
// pixel as a point or as a unit square (μ20 grows by exactly 1/12 per pixel).
// Overlays: the marginal projections along x (bars from the bottom edge) and y (bars from
// the right edge) with mean, ±1 sd and a skewness arrow; optionally the equivalent ellipse.
// "grey image": f = the mask blurred and multiplied by a left-to-right ramp, drawn over the
// mask; the same formulas, with f as weights.
import { htmlEl, checkbox, svgEl } from "../lib/ui";
import { mirrorPoly, scalePoly, ellipsePoly, SHAPE_PRESETS, Poly, Pt } from "../lib/shapecanvas";
import { rawMoments, centralFromRaw, idx, Table } from "../lib/moments";
import { gaussianBlur, fromArray } from "../lib/imageops";
import { eigSym2 } from "../lib/linalg";
import { diverging, textColorOn, RGB } from "../lib/colour";
import { registerMode, MomentsContext, MomentsModeSpec, Digitised } from "./registry";
import { sline, spoly, sarrow, scross, P2 } from "./draw";
import { rect } from "./shapes";

const CX = "var(--qb-c2, #d62728)", CY = "var(--qb-c3, #2ca02c)", CE = "var(--qb-c4, #ff7f0e)";
const SUB = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"];
const SUP: Record<string, string> = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };

/** 3 significant digits; powers of ten written as ×10ⁿ outside [0.01, 10⁴). */
export function sci(v: number, d = 3): string {
  if (!Number.isFinite(v)) return "–";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 0.01 && a < 1e4) return Number(v.toPrecision(d)).toString();
  const e = Math.floor(Math.log10(a)), m = v / 10 ** e;
  return `${m.toFixed(d - 1)}×10${String(e).split("").map(c => SUP[c]).join("")}`;
}

/** A pear: the union of a large disc (the bulb, left) and a smaller one (the neck, right),
 *  traced in polar coordinates around the bulb's centre; s = bulb radius. */
export function pearPoly(cx: number, cy: number, s: number, n = 160): Poly {
  const rB = 0.6 * s, b = 1.05 * s;   // neck disc radius and distance between the centres
  return Array.from({ length: n }, (_, i) => {
    const t = 2 * Math.PI * i / n, st = Math.sin(t), ct = Math.cos(t);
    const disc = rB * rB - b * b * st * st;
    const far = disc >= 0 ? b * ct + Math.sqrt(disc) : -Infinity;
    const r = Math.max(s, far);
    return [cx + r * ct, cy + r * st] as Pt;
  });
}

/** Cell colour of a moment: sign by hue, size by saturation (t in [0, 1]). */
function cellColour(sign: number, t: number): RGB {
  return diverging(0.5 + 0.5 * Math.sign(sign) * Math.min(1, Math.max(0, t)) * 0.92);
}

registerMode({
  id: "higher",
  build(ctx: MomentsContext): MomentsModeSpec {
    const N = Math.max(2, Math.min(6, Number(ctx.host.dataset.order ?? 4) || 4));
    const st = { central: true, square: false, grey: false, marg: true, ellipse: false };
    htmlEl("div", "mo-lab", "show", ctx.controls);
    const ck = htmlEl("div", "mo-checks", "", ctx.controls);
    const cC = checkbox(ck, "central μ (else raw m)", st.central, v => { st.central = v; ctx.refresh(); });
    const cG = checkbox(ck, "grey image", st.grey, v => { st.grey = v; ctx.refresh(); });
    const cS = checkbox(ck, "pixel = square", st.square, v => { st.square = v; ctx.refresh(); });
    const cM = checkbox(ck, "marginals", st.marg, v => { st.marg = v; ctx.refresh(); });
    const cE = checkbox(ck, "ellipse", st.ellipse, v => { st.ellipse = v; ctx.refresh(); });

    const head = htmlEl("div", "", "", ctx.out);
    head.style.cssText = "font-size:15px;color:var(--qb-muted)";
    const tabBox = htmlEl("div", "", "", ctx.out);
    const stats = htmlEl("div", "", "", ctx.out);

    // grey-image overlay (an <image> in canvas units, under the marginals)
    const greyCan = document.createElement("canvas");
    let greyImg: SVGImageElement | null = null;

    function greyOf(d: Digitised): Float64Array {
      const g = gaussianBlur(fromArray(d.w, d.h, d.mask), 1.2 / d.px);
      let x0 = d.w, x1 = 0;
      for (let y = 0; y < d.h; y++) for (let x = 0; x < d.w; x++) if (d.mask[y * d.w + x]) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); }
      const f = new Float64Array(d.w * d.h);
      for (let y = 0; y < d.h; y++) for (let x = 0; x < d.w; x++) {
        const r = x1 > x0 ? (x - x0) / (x1 - x0) : 0.5;
        f[y * d.w + x] = g.data[y * d.w + x] * (0.25 + 0.75 * Math.min(1, Math.max(0, r)));
      }
      return f;
    }

    function drawGrey(d: Digitised, f: Float64Array | null): void {
      if (!f) { greyImg?.remove(); greyImg = null; return; }
      greyCan.width = d.w; greyCan.height = d.h;
      const c2 = greyCan.getContext("2d")!, id = c2.createImageData(d.w, d.h);
      for (let i = 0; i < d.w * d.h; i++) {
        const v = Math.min(1, f[i]);
        if (v <= 0.002) continue;
        id.data[4 * i] = 30; id.data[4 * i + 1] = 30; id.data[4 * i + 2] = 40; id.data[4 * i + 3] = Math.round(235 * v);
      }
      c2.putImageData(id, 0, 0);
      greyImg = svgEl("image", { x: 0, y: 0, width: d.w * d.px, height: d.h * d.py, preserveAspectRatio: "none", href: greyCan.toDataURL() });
      greyImg.style.setProperty("image-rendering", "pixelated");
      ctx.overlay.appendChild(greyImg);
    }

    function table(T: Table, m00: number): void {
      tabBox.replaceChildren();
      const t = htmlEl("table", "mo-tab mo-tri", "", tabBox);
      t.style.cssText = "width:auto;border-collapse:separate;border-spacing:3px";
      const hr = htmlEl("tr", "", "", t);
      htmlEl("td", "", "", hr);
      for (let p = 0; p <= N; p++) { const c = htmlEl("td", "", `p = ${p}`, hr); c.style.cssText = "text-align:center;color:var(--qb-muted);border:none"; }
      // largest |v|^{1/k} per order k, for the colour scale
      const big = new Array(N + 1).fill(0);
      for (let p = 0; p <= N; p++) for (let q = 0; p + q <= N; q++) {
        const k = p + q, v = Math.abs(T[idx(N, p, q)] / m00);
        if (k > 0) big[k] = Math.max(big[k], v ** (1 / k));
      }
      for (let q = 0; q <= N; q++) {
        const tr = htmlEl("tr", "", "", t);
        const c0 = htmlEl("td", "", `q = ${q}`, tr);
        c0.style.cssText = "color:var(--qb-muted);border:none;white-space:nowrap";
        for (let p = 0; p <= N; p++) {
          const td = htmlEl("td", "v", "", tr);
          td.style.cssText = "border:none;text-align:center;min-width:4.2em;padding:.15em .3em;border-radius:3px";
          if (p + q > N) continue;
          const k = p + q, v = T[idx(N, p, q)] / m00;
          const tt = k === 0 ? 0.3 : big[k] > 0 ? (Math.abs(v) ** (1 / k)) / big[k] : 0;
          const col = cellColour(v, Math.abs(v) < 1e-9 * (big[k] ** k || 1) ? 0 : tt);
          td.style.background = `rgb(${col.map(c => Math.round(255 * c)).join(",")})`;
          td.style.color = textColorOn(col);
          td.textContent = Math.abs(v) < 1e-9 * (big[k] ** k || 1) ? "0" : sci(v);
          td.title = `${st.central ? "μ" : "m"}${SUB[p]}${SUB[q]} / m₀₀`;
        }
      }
    }

    function marginals(d: Digitised, f: ArrayLike<number>, mu: Table, m: Table): void {
      const g = ctx.overlay, m00 = m[0];
      const colX = new Float64Array(d.w), rowY = new Float64Array(d.h);
      for (let y = 0; y < d.h; y++) for (let x = 0; x < d.w; x++) { const v = f[y * d.w + x]; colX[x] += v; rowY[y] += v; }
      const mx = Math.max(...colX), my = Math.max(...rowY), H = 10, S = ctx.size;
      const xb = m[idx(N, 1, 0)] / m00, yb = m[idx(N, 0, 1)] / m00;
      const sx = Math.sqrt(mu[idx(N, 2, 0)] / m00), sy = Math.sqrt(mu[idx(N, 0, 2)] / m00);
      const gx = mu[idx(N, 3, 0)] / m00 / sx ** 3, gy = mu[idx(N, 0, 3)] / m00 / sy ** 3;
      const cx = (xb + 0.5) * d.px, cy = (yb + 0.5) * d.py;
      // x marginal: bars up from the bottom edge
      if (mx > 0) for (let x = 0; x < d.w; x++) if (colX[x] > 0) {
        const r = svgEl("rect", { x: x * d.px, y: S - H * colX[x] / mx, width: d.px, height: H * colX[x] / mx }, g);
        r.style.setProperty("fill", CX); r.style.setProperty("fill-opacity", "0.35");
      }
      if (my > 0) for (let y = 0; y < d.h; y++) if (rowY[y] > 0) {
        const r = svgEl("rect", { x: S - H * rowY[y] / my, y: y * d.py, width: H * rowY[y] / my, height: d.py }, g);
        r.style.setProperty("fill", CY); r.style.setProperty("fill-opacity", "0.35");
      }
      // mean and ±1 sd
      sline(g, [cx, S - H - 1.5], [cx, S], CX, 0.45);
      sline(g, [(xb - sx + 0.5) * d.px, S - H - 0.8], [(xb + sx + 0.5) * d.px, S - H - 0.8], CX, 0.45);
      sline(g, [S - H - 1.5, cy], [S, cy], CY, 0.45);
      sline(g, [S - H - 0.8, (yb - sy + 0.5) * d.py], [S - H - 0.8, (yb + sy + 0.5) * d.py], CY, 0.45);
      // skewness arrows: from the mean towards the long tail, length |γ|·sd
      if (Math.abs(gx) > 0.02) sarrow(g, [cx, S - H - 2.6], [cx + gx * sx * d.px, S - H - 2.6], CX, 0.55);
      if (Math.abs(gy) > 0.02) sarrow(g, [S - H - 2.6, cy], [S - H - 2.6, cy + gy * sy * d.py], CY, 0.55);
    }

    function update(d: Digitised): void {
      ctx.overlay.replaceChildren();
      const f: Float64Array = st.grey ? greyOf(d) : Float64Array.from(d.mask);
      drawGrey(d, st.grey ? f : null);
      const m = rawMoments(f, d.w, d.h, N, { pixel: st.square ? "square" : "point" });
      const m00 = m[0];
      if (!(m00 > 0)) { tabBox.textContent = "no object pixels"; stats.replaceChildren(); head.textContent = ""; return; }
      const mu = centralFromRaw(m, N);
      const dPt = centralFromRaw(rawMoments(f, d.w, d.h, 2), 2)[idx(2, 2, 0)];
      const dSq = centralFromRaw(rawMoments(f, d.w, d.h, 2, { pixel: "square" }), 2)[idx(2, 2, 0)];
      head.innerHTML = `${st.central ? "central moments μ<sub>pq</sub>" : "raw moments m<sub>pq</sub>"} ÷ m₀₀, up to order ${N} (pixel units; x right, y down${st.grey ? "; grey f" : ""}). m₀₀ = ${sci(m00, 4)}`;
      table(st.central ? mu : m, m00);
      const xb = m[idx(N, 1, 0)] / m00, yb = m[idx(N, 0, 1)] / m00;
      const v20 = mu[idx(N, 2, 0)] / m00, v02 = mu[idx(N, 0, 2)] / m00;
      const gx = mu[idx(N, 3, 0)] / m00 / v20 ** 1.5, gy = mu[idx(N, 0, 3)] / m00 / v02 ** 1.5;
      const kx = N >= 4 ? mu[idx(N, 4, 0)] / m00 / v20 ** 2 : NaN, ky = N >= 4 ? mu[idx(N, 0, 4)] / m00 / v02 ** 2 : NaN;
      stats.replaceChildren();
      const t = htmlEl("table", "mo-tab", "", stats);
      const row = (a: string, b: string, c: string) => { const tr = htmlEl("tr", "", "", t); htmlEl("td", "", a, tr); htmlEl("td", "v", b, tr); htmlEl("td", "v", c, tr); };
      const hr = htmlEl("tr", "h", "", t); htmlEl("td", "", "marginal", hr); htmlEl("td", "", "along x", hr); htmlEl("td", "", "along y", hr);
      row("mean", xb.toFixed(2), yb.toFixed(2));
      row("sd = √(μ₂₀/m₀₀), √(μ₀₂/m₀₀)", Math.sqrt(v20).toFixed(2), Math.sqrt(v02).toFixed(2));
      row("skewness μ₃₀/(m₀₀ sd³)", gx.toFixed(3), gy.toFixed(3));
      if (N >= 4) row("kurtosis μ₄₀/(m₀₀ sd⁴)", kx.toFixed(3), ky.toFixed(3));
      row("μ₂₀/m₀₀: point → square", `${(dPt / m00).toFixed(3)} → ${(dSq / m00).toFixed(3)}`, `+${((dSq - dPt) / m00).toFixed(4)}`);
      if (st.marg) marginals(d, f, mu, m);
      if (st.ellipse) {
        const e = eigSym2(v20, mu[idx(N, 1, 1)] / m00, v02), a = 2 * Math.sqrt(Math.max(0, e.l1)), b = 2 * Math.sqrt(Math.max(0, e.l2)), v1 = e.v1, v2: P2 = [-v1[1], v1[0]];
        const c: P2 = [(xb + 0.5) * d.px, (yb + 0.5) * d.py];
        spoly(ctx.overlay, Array.from({ length: 72 }, (_, i) => { const th = 2 * Math.PI * i / 72; return [c[0] + d.px * (a * Math.cos(th) * v1[0] + b * Math.sin(th) * v2[0]), c[1] + d.py * (a * Math.cos(th) * v1[1] + b * Math.sin(th) * v2[1])] as P2; }), CE, 0.5);
      }
      scross(ctx.overlay, [(xb + 0.5) * d.px, (yb + 0.5) * d.py], 1.2, "var(--qb-fg, #222)", 0.4);
    }

    const set = (o: Partial<typeof st>) => () => {
      Object.assign(st, { central: true, square: false, grey: false, marg: true, ellipse: false }, o);
      cC.checked = st.central; cS.checked = st.square; cG.checked = st.grey; cM.checked = st.marg; cE.checked = st.ellipse;
    };
    // square of side a and disc of radius r = a/√3: the same variance a²/12 = r²/4
    const a = 30, r = a / Math.sqrt(3);
    return {
      presets: [
        { id: "pear", label: "a pear: the sign of μ₃₀", shapes: () => [pearPoly(24, 30, 12)], apply: set({}) },
        { id: "mirror", label: "its mirror image", shapes: () => [mirrorPoly(pearPoly(24, 30, 12), "x", [32, 30])], apply: set({}) },
        { id: "square", label: "square: kurtosis", shapes: () => [rect(32 - a / 2, 30 - a / 2, 32 + a / 2, 30 + a / 2)], apply: set({ ellipse: true }) },
        { id: "disc", label: "disc with the same S", shapes: () => [ellipsePoly(32, 30, r, r, 120)], apply: set({ ellipse: true }) },
        { id: "L", label: "L: the ellipse is not enough", shapes: s => SHAPE_PRESETS.find(p => p.id === "L")!.make(s), apply: set({ ellipse: true }) },
        { id: "grey", label: "grey vs binary", shapes: () => [scalePoly(ellipsePoly(30, 30, 18, 12, 96, 0.3), 1)], apply: set({ grey: true }) },
        { id: "pixel", label: "pixel: point or square", shapes: () => [rect(28, 26, 36, 34)], pixel: 8, apply: set({ square: true, marg: false }) },
      ],
      update,
    };
  },
});
