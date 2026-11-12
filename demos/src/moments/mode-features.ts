// moments mode "features" (unit shape-features/): size and shape features beyond moments of
// the largest object. Overlays: convex hull (pixel corners or centres), the calipers at an
// angle θ with the width plot w(θ), min and max Feret, the largest inscribed circle, the
// smallest enclosing circle, the circle of equal area, and a Fourier outline with k
// coefficients. Table: digitised value and exact value (from the outline) side by side.
// Units: pixels or µm (0.25 µm per canvas unit), square or anisotropic pixels (rows twice as
// far apart); in µm the mask is first resampled to square pixels. "mask error" dilates or
// erodes the mask by 1–2 pixels before measuring.
import { htmlEl, button, checkbox, slider } from "../lib/ui";
import { Plot } from "../lib/plot";
import { label2d } from "../lib/label";
import { seDisk, dilateBinary, erodeBinary } from "../lib/morph";
import { convexHull, shoelaceArea, rotatingCalipers, feretAt, minEnclosingCircle, widthProfile, Pt } from "../lib/convexhull";
import { registerMode, MomentsContext, MomentsModeSpec, Digitised } from "./registry";
import { shapeFeatures, outline, fourierOutline } from "./features";
import { exactMeasures, disc, crescent, twoDiscs, rod, rect, star } from "./shapes";
import { sline, spoly, scircle, scross, table, f2, P2 } from "./draw";

const UM_PER_UNIT = 0.25;
const COL = { hull: "var(--qb-c1, #1f77b4)", cal: "var(--qb-c4, #ff7f0e)", fer: "var(--qb-c2, #d62728)", ins: "var(--qb-c3, #2ca02c)", enc: "var(--qb-c5, #9467bd)", eq: "var(--qb-muted, #777)", fou: "#e377c2" };

/** By-hand objects of the exercise, as orthogonal polygons in pixel units (pixel size 8). */
export const HAND_HULL: [number, number][] = [[1, 1], [3, 1], [3, 3], [5, 3], [5, 2], [7, 2], [7, 5], [5, 5], [5, 7], [3, 7], [3, 5], [1, 5]];

registerMode({
  id: "features",
  build(ctx: MomentsContext): MomentsModeSpec {
    const st = { hullMode: "corners" as "corners" | "centres", theta: 30, k: 4, um: false, aniso: false, err: 0 };
    const show = { hull: true, calipers: true, feret: true, inscribed: false, enclosing: false, equal: false, fourier: false };

    htmlEl("div", "mo-lab", "convex hull of the pixel …", ctx.controls);
    const hr = htmlEl("div", "qb-presets", "", ctx.controls);
    const hc = button(hr, "corners", () => { st.hullMode = "corners"; sync(); ctx.refresh(); });
    const hn = button(hr, "centres", () => { st.hullMode = "centres"; sync(); ctx.refresh(); });
    const ck = htmlEl("div", "mo-checks", "", ctx.controls);
    const boxes: Record<keyof typeof show, HTMLInputElement> = {} as never;
    const names: [keyof typeof show, string][] = [["hull", "hull"], ["calipers", "calipers at θ"], ["feret", "min/max Feret"], ["inscribed", "inscribed ○"], ["enclosing", "enclosing ○"], ["equal", "equal-area ○"], ["fourier", "Fourier outline"]];
    for (const [key, label] of names) boxes[key] = checkbox(ck, label, show[key], v => { show[key] = v; ctx.refresh(); });
    const th = slider(ctx.controls, "θ", 0, 179, 1, st.theta, v => { st.theta = v; ctx.refresh(); }, v => `${v}°`);
    const ks = slider(ctx.controls, "k", 1, 24, 1, st.k, v => { st.k = v; ctx.refresh(); });
    const ur = htmlEl("div", "qb-presets", "", ctx.controls);
    const bpx = button(ur, "px", () => { st.um = false; sync(); ctx.refresh(); });
    const bum = button(ur, "µm", () => { st.um = true; sync(); ctx.refresh(); });
    const ban = checkbox(ur, "rows 2× apart", false, v => { st.aniso = v; ctx.setAnisotropy(v ? 2 : 1); });
    const es = slider(ctx.controls, "mask error", -2, 2, 1, 0, v => { st.err = v; ctx.refresh(); }, v => (v > 0 ? `+${v}` : `${v}`));
    function sync(): void {
      hc.classList.toggle("active", st.hullMode === "corners"); hn.classList.toggle("active", st.hullMode === "centres");
      bpx.classList.toggle("active", !st.um); bum.classList.toggle("active", st.um);
    }
    sync();

    const tab = htmlEl("div", "", "", ctx.out);
    const plotBox = htmlEl("div", "", "", ctx.out);
    const plot = new Plot(plotBox, { width: 520, height: 200, x: [0, 180], y: [0, 10], xlabel: "θ (degrees)", ylabel: "w(θ)", font: 17, xticks: [0, 45, 90, 135, 180] });
    const wl = plot.line([], [], { color: COL.cal });
    const wNow = plot.vline(30, { color: COL.cal, width: 1.5, dash: "3 3" });
    const wMin = plot.scatter([], [], { color: COL.fer, r: 5 });

    function set(opts: Partial<typeof st> & { show?: Partial<typeof show> }): void {
      Object.assign(st, opts);
      if (opts.show) for (const k of Object.keys(show) as (keyof typeof show)[]) { show[k] = opts.show[k] ?? false; boxes[k].checked = show[k]; }
      th.set(st.theta); ks.set(st.k); es.set(st.err); ban.checked = st.aniso; sync();
      ctx.setAnisotropy(st.aniso ? 2 : 1);
    }

    function transformMask(d: Digitised): Digitised {
      if (!st.err) return d;
      const se = seDisk(Math.abs(st.err));
      const m = st.err > 0 ? dilateBinary(d.mask, d.w, d.h, se) : erodeBinary(d.mask, d.w, d.h, se);
      return { ...d, mask: m };
    }

    function update(d: Digitised): void {
      const g = ctx.overlay;
      g.replaceChildren();
      // largest 8-connected object only
      const lab = label2d(d.mask, d.w, d.h, 8);
      const sizes = new Int32Array(lab.count + 1);
      for (const v of lab.data) sizes[v]++;
      let best = 0;
      for (let l = 1; l <= lab.count; l++) if (sizes[l] > sizes[best] || best === 0) best = l;
      let mask = Uint8Array.from(lab.data, v => (v === best && best ? 1 : 0)), w = d.w, h = d.h, sx = d.px, sy = d.py;
      // physical units: resample to square pixels first
      if (st.um && d.py !== d.px) {
        const a = Math.round(d.py / d.px), m2 = new Uint8Array(w * h * a);
        for (let y = 0; y < h * a; y++) for (let x = 0; x < w; x++) m2[y * w + x] = mask[Math.floor(y / a) * w + x];
        mask = m2; h = h * a; sy = d.px;
      }
      const f = best ? shapeFeatures(mask, w, h, st.hullMode) : null;
      const C = (p: Pt | P2): P2 => [p[0] * sx, p[1] * sy];
      const L = st.um ? sx * UM_PER_UNIT : 1, unit = st.um ? " µm" : " px";
      const len = (v: number) => f2(v * L, 2), area = (v: number) => f2(v * L * L, 1);
      if (!f) { table(tab, ["no object pixels"]); wl.set([], []); return; }
      // overlays
      if (show.hull) spoly(g, f.hull.map(C), COL.hull, 0.4);
      const thr = st.theta * Math.PI / 180, ang = -thr;   // ⟲ on screen = negative angle with y down
      const ux = Math.cos(ang), uy = Math.sin(ang), vx = -uy, vy = ux;
      const supp = (t: number, col: string, wd = 0.4, dash?: string) => sline(g, C([t * ux - 200 * vx, t * uy - 200 * vy]), C([t * ux + 200 * vx, t * uy + 200 * vy]), col, wd, dash);
      const fa = feretAt(f.hull, ang);
      if (show.calipers) {
        const lo = fa.lo[0] * ux + fa.lo[1] * uy, hi = fa.hi[0] * ux + fa.hi[1] * uy;
        supp(lo, COL.cal, 0.45); supp(hi, COL.cal, 0.45);
      }
      const cal = f.calipers;
      if (show.feret) {
        sline(g, C(cal.maxPair[0]), C(cal.maxPair[1]), COL.fer, 0.5);
        const mx = Math.cos(cal.minAngle), my = Math.sin(cal.minAngle);
        let lo = Infinity, hi = -Infinity;
        for (const p of f.hull) { const t = p[0] * mx + p[1] * my; lo = Math.min(lo, t); hi = Math.max(hi, t); }
        const nx = -my, ny = mx;
        for (const t of [lo, hi]) sline(g, C([t * mx - 200 * nx, t * my - 200 * ny]), C([t * mx + 200 * nx, t * my + 200 * ny]), COL.fer, 0.35, "1.2 0.8");
      }
      const ell = (c: Pt | P2, r: number, col: string, dash?: string) => {
        if (Math.abs(sx - sy) < 1e-9) scircle(g, C(c), r * sx, col, 0.45, dash);
        else spoly(g, Array.from({ length: 64 }, (_, i) => C([c[0] + r * Math.cos(2 * Math.PI * i / 64), c[1] + r * Math.sin(2 * Math.PI * i / 64)])), col, 0.45, "none", dash);
      };
      const ctr: P2 = [f.region.cx + 0.5, f.region.cy + 0.5];
      if (show.inscribed) { ell(f.inscribedAt, f.inscribedR, COL.ins); scross(g, C(f.inscribedAt), 0.8, COL.ins, 0.35); }
      if (show.enclosing) ell([f.enclosing.x, f.enclosing.y], f.enclosing.r, COL.enc);
      if (show.equal) ell(ctr, Math.sqrt(f.area / Math.PI), COL.eq, "1.5 1");
      if (show.fourier) {
        const fo = fourierOutline(outline(mask, w, h, 128), st.k);
        spoly(g, fo.smooth.map(C), COL.fou, 0.55);
      }
      // exact values (outline), in the current units; not for anisotropic pixels in px
      const ex = exactMeasures(d.polys);
      const exOk = !!ex && lab.count === 1 && !(d.py !== d.px && !st.um) && !st.err;
      const s = st.um ? UM_PER_UNIT : 1 / d.px;     // canvas units → current units
      let exHull: Pt[] = [], exCal = null, exMec = null;
      if (exOk) {
        exHull = convexHull(d.polys.flat() as Pt[]);
        exCal = rotatingCalipers(exHull);
        exMec = minEnclosingCircle(exHull, 7);
      }
      const exA = exOk ? ex!.area * s * s : NaN, exP = exOk ? ex!.perimeter * s : NaN, exHA = exOk ? Math.abs(shoelaceArea(exHull)) * s * s : NaN;
      const E = (v: number, d2 = 2) => (exOk && Number.isFinite(v) ? f2(v, d2) : "");
      const A = f.area, P = f.region.perimCrofton;
      const rows: (string[] | string)[] = [
        `largest object · ${st.um ? `µm (${f2(sx * UM_PER_UNIT, 3)} µm per pixel)` : "pixels"} · hull of pixel ${st.hullMode}`,
        ["", "digitised", exOk ? "exact" : ""],
        ["area A", area(A) + (st.um ? " µm²" : ""), E(exA, 1)],
        ["perimeter P (Crofton)", len(P) + unit, E(exP)],
        ["equivalent diameter √(4A/π)", len(Math.sqrt(4 * A / Math.PI)), E(Math.sqrt(4 * exA / Math.PI))],
        ["convex hull area", area(f.hullArea), E(exHA, 1)],
        ["solidity A / hull", f2(f.solidity, 3), E(exA / exHA, 3)],
        ["form factor 4πA/P²: edge count", f2(f.ffEdges, 3), E(4 * Math.PI * exA / (exP * exP), 3)],
        ["… chain code", f2(f.ffChain, 3), ""],
        ["… Crofton", f2(f.ffCrofton, 3), ""],
        ["compactness 2π·mean r²/A", f2(f.compactMoments, 3), exOk ? f2(2 * Math.PI * (ex!.s11 + ex!.s22) / ex!.area, 3) : ""],
        ["max Feret", len(cal.maxDiameter), exCal ? E(exCal.maxDiameter * s) : ""],
        ["min Feret", len(cal.minWidth), exCal ? E(exCal.minWidth * s) : ""],
        [`w(θ) at θ = ${st.theta}°`, len(fa.width), exOk ? E(feretAt(exHull, ang).width * s) : ""],
        ["max inscribed radius", len(f.inscribedR), ""],
        ["mean radius (mean DT)", len(f.meanR), ""],
        ["enclosing circle radius", len(f.enclosing.r), exMec ? E(exMec.r * s) : ""],
      ];
      table(tab, rows);
      const prof = widthProfile(f.hull, 180);
      const xs = Array.from(prof.theta, t => (180 - t * 180 / Math.PI) % 180), ys = Array.from(prof.width, v => v * L);
      const order = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b]);
      wl.set(order.map(i => xs[i]), order.map(i => ys[i]));
      plot.setDomain([0, 180], [0, Math.max(...ys) * 1.1]);
      wNow.set(st.theta);
      const minDeg = ((180 - cal.minAngle * 180 / Math.PI) % 180 + 180) % 180, maxDeg = ((180 - cal.maxAngle * 180 / Math.PI) % 180 + 180) % 180;
      wMin.set([minDeg, maxDeg], [cal.minWidth * L, cal.maxDiameter * L]);
    }

    const discArea = Math.PI * 16 * 16;
    return {
      presets: [
        { id: "disc", label: "disc: form factor 0.62 by edge count", shapes: () => [disc(32, 32, 22)], apply: () => set({ err: 0, aniso: false, um: false, show: { hull: true, feret: true } }) },
        { id: "small", label: "small disc: form factor above 1", shapes: () => [disc(32, 32, 3.2)], pixel: 1, apply: () => set({ err: 0, aniso: false, um: false, show: { hull: true } }) },
        { id: "star", label: "star vs disc: same area, solidity tells", shapes: () => { const p = star(32, 33, 26, 11, 5); const a = Math.abs(exactMeasures([p])!.area); return [p.map(([x, y]) => [32 + (x - 32) * Math.sqrt(discArea / a), 33 + (y - 33) * Math.sqrt(discArea / a)] as [number, number])]; }, apply: () => set({ err: 0, aniso: false, um: false, show: { hull: true, equal: true } }) },
        { id: "pair", label: "two touching nuclei: solidity drops", shapes: () => [twoDiscs(32, 32, 13, 22)], apply: () => set({ err: 0, aniso: false, um: false, show: { hull: true } }) },
        { id: "rod", label: "rod: min Feret ≈ 2 × inscribed radius", shapes: () => [rod(32, 32, 34, 5, 0.5)], apply: () => set({ err: 0, aniso: false, um: false, theta: 30, show: { feret: true, inscribed: true, calipers: true } }) },
        { id: "crescent", label: "crescent", shapes: () => [crescent(30, 32, 20, 14)], apply: () => set({ err: 0, aniso: false, um: false, show: { hull: true, inscribed: true } }) },
        { id: "aniso", label: "anisotropic pixels: the circle that is not", shapes: () => [disc(32, 32, 20)], pixel: 2, apply: () => set({ err: 0, aniso: true, um: false, show: { hull: true, feret: true } }) },
        { id: "fourier", label: "k Fourier coefficients", shapes: () => [star(32, 33, 26, 12, 5)], apply: () => set({ err: 0, aniso: false, um: false, k: 3, show: { fourier: true } }) },
        { id: "hand", label: "6 × 6 hull (exercise)", shapes: () => [HAND_HULL.map(([x, y]) => [8 * x, 8 * y] as [number, number])], pixel: 8, apply: () => set({ err: 0, aniso: false, um: false, hullMode: "corners", show: { hull: true, feret: true } }) },
        { id: "square", label: "4 × 4 square (exercise)", shapes: () => [rect(16, 16, 48, 48)], pixel: 8, apply: () => set({ err: 0, aniso: false, um: false, show: { hull: true } }) },
      ],
      update, transformMask,
    };
  },
});

