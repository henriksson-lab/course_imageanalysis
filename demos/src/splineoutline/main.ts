// Smooth outlines from pixels (splines day, unit 3). A synthetic object is rasterised to a
// 64 × 64 grid; its outer boundary (the pixel edges, a staircase) is fitted with a closed
// cubic smoothing spline (m control points, penalty λ on second differences of the control
// points). From the fitted curve: length (an integral of the speed) and curvature (a formula
// of the derivatives), shown as a comb or colour and as curvature against arc length.
// Readouts compare the length by edge count, by chain code, by the spline and the true
// length. Toggle: truncated Fourier descriptors with K terms on the same outline.
// <div data-demo="splineoutline" data-preset="disc|small|star|neck|fourier|rough"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { rasterise, ellipsePoly, starPoly, Poly } from "../lib/shapecanvas";
import { fitClosedSpline, sampleCurve, arcLength, totalCurvature, Pt } from "../lib/spline";
import { diverging } from "../lib/colour";
import { fft1d } from "../lib/fft";
import { traceBoundary } from "./boundary";

const NAME = "splineoutline";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; }
.qb-demo[data-demo="${NAME}"] .so-main { flex: 0 1 500px; min-width: 260px; }
.qb-demo[data-demo="${NAME}"] .so-side { flex: 1 1 380px; max-width: 600px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .so-side .qb-plot { flex: none; max-width: none; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 18px; white-space: pre; }
.qb-demo[data-demo="${NAME}"] .so-main .qb-axes { display: none; }
.qb-demo[data-demo="${NAME}"] .so-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .so-row { display: flex; flex-wrap: wrap; gap: .2em .9em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .so-main { flex: 0 0 500px; }
.reveal .qb-demo[data-demo="${NAME}"] .so-side { flex: 0 0 600px; gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .04em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { padding: .2em .45em; font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .so-lab, .reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const G = 64;
type ShapeId = "disc" | "small" | "ellipse" | "kidney" | "star" | "neck" | "square";
const SHAPES: { id: ShapeId; label: string; polys: () => Poly[]; r?: number }[] = [
  { id: "disc", label: "disc r = 10", polys: () => [ellipsePoly(32, 32, 10, 10, 400)], r: 10 },
  { id: "small", label: "disc r = 3", polys: () => [ellipsePoly(32, 32, 3, 3, 200)], r: 3 },
  { id: "ellipse", label: "ellipse", polys: () => [ellipsePoly(32, 32, 24, 12, 400, 0.4)] },
  { id: "kidney", label: "kidney", polys: () => [polar(t => 18 + 3 * Math.cos(2 * t) - 7 * Math.exp(-(((t - Math.PI / 2 + Math.PI) % (2 * Math.PI) - Math.PI) ** 2) / 0.35))] },
  { id: "star", label: "star", polys: () => [starPoly(32, 32, 25, 12, 5)] },
  { id: "neck", label: "two cells, neck", polys: () => [ellipsePoly(20, 32, 13, 13, 300), ellipsePoly(44, 32, 13, 13, 300)] },
  { id: "square", label: "square", polys: () => [[[14, 14], [50, 14], [50, 50], [14, 50]]] },
];
function polar(r: (t: number) => number, n = 400): Poly {
  return Array.from({ length: n }, (_, i) => { const t = 2 * Math.PI * i / n; return [32 + r(t) * Math.cos(t), 32 + r(t) * Math.sin(t)] as [number, number]; });
}

interface State { shape: ShapeId; logLam: number; m: number; comb: boolean; colour: boolean; fourier: boolean; K: number }
const DEFAULT: State = { shape: "disc", logLam: 0.5, m: 40, comb: false, colour: true, fourier: false, K: 8 };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "disc", label: "disc: three perimeters", s: { shape: "disc" } },
  { id: "small", label: "small disc (r = 3)", s: { shape: "small", m: 12 } },
  { id: "star", label: "star: corners vs λ", s: { shape: "star", logLam: 0.5, m: 80 } },
  { id: "neck", label: "neck between two cells", s: { shape: "neck", m: 60, comb: true } },
  { id: "fourier", label: "Fourier rings at a corner", s: { shape: "square", m: 80, logLam: -1, fourier: true, K: 8 } },
  { id: "rough", label: "too little smoothing", s: { shape: "disc", logLam: -3, m: 120 } },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...DEFAULT, ...p0.s };

  const main = plotFigure(host, "", "so-main");
  const plot = new Plot(main, { width: 500, height: 500, x: [0, G], y: [G, 0], xticks: [], yticks: [], margin: { l: 6, r: 6, t: 6, b: 6 }, grid: false });
  const side = htmlEl("div", "so-side", "", host);
  htmlEl("div", "so-lab", "presets", side);
  const pbtns = presets(side, PRESETS.map(p => ({ label: p.label, apply: () => load(p.s) })));
  htmlEl("div", "so-lab", "object", side);
  const shBtns = presets(side, SHAPES.map(s => ({ label: s.label, apply: () => { st.shape = s.id; draw(); } })));
  const lamS = slider(side, "log₁₀ λ (smoothing)", -3, 3, 0.05, st.logLam, v => { st.logLam = v; draw(); }, v => v.toFixed(2));
  const mS = slider(side, "control points m", 6, 120, 1, st.m, v => { st.m = v; draw(); });
  const row = htmlEl("div", "so-row", "", side);
  const cks = [
    checkbox(row, "curvature colour", st.colour, v => { st.colour = v; draw(); }),
    checkbox(row, "comb", st.comb, v => { st.comb = v; draw(); }),
    checkbox(row, "Fourier descriptors", st.fourier, v => { st.fourier = v; draw(); }),
  ];
  const kS = slider(side, "Fourier terms K", 1, 40, 1, st.K, v => { st.K = v; draw(); });
  const ro = htmlEl("div", "qb-readout", "", side);
  const kfig = plotFigure(side, "");
  const kplot = new Plot(kfig, { width: 600, height: 190, x: [0, 1], y: [-0.3, 0.6], xlabel: "arc length s", ylabel: "κ", margin: { b: 50, l: 92 } });
  button(side, "reset", () => load(p0.s));

  const under = svgEl("g", {}, plot.layer);

  function load(s: Partial<State>): void {
    Object.assign(st, DEFAULT, s);
    lamS.set(st.logLam); mS.set(st.m); kS.set(st.K);
    [st.colour, st.comb, st.fourier].forEach((v, i) => (cks[i].checked = v));
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i].s === s));
    draw();
  }

  function draw(): void {
    const sh = SHAPES.find(s => s.id === st.shape)!;
    shBtns.forEach((b, i) => b.classList.toggle("active", SHAPES[i].id === st.shape));
    const polys = sh.polys(), mask = rasterise(polys, G, G, "nonzero"), B = traceBoundary(mask, G, G);
    // boundary in the rasteriser's frame (pixel x covers [x, x + 1])
    const pts: Pt[] = B.points.map(p => [p[0] + 0.5, p[1] + 0.5]);
    const corners: Pt[] = B.corners.map(p => [p[0] + 0.5, p[1] + 0.5]);
    const m = Math.min(st.m, pts.length);
    const fit = fitClosedSpline(pts, m, Math.pow(10, st.logLam));
    const S = sampleCurve(fit.curve, 10);
    const len = arcLength(fit.curve, 4), tk = totalCurvature(fit.curve, 4), sgn = tk < 0 ? -1 : 1;
    // (scales are read after the zoom below)
    plot.clear(); under.replaceChildren();
    // zoom to the object (square box, 4 pixels of margin)
    let bx0 = G, bx1 = 0, by0 = G, by1 = 0;
    for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) if (B.mask[y * G + x]) { bx0 = Math.min(bx0, x); bx1 = Math.max(bx1, x + 1); by0 = Math.min(by0, y); by1 = Math.max(by1, y + 1); }
    const half = Math.max(bx1 - bx0, by1 - by0) / 2 + 4, cx = (bx0 + bx1) / 2, cy = (by0 + by1) / 2;
    plot.setDomain([cx - half, cx + half], [cy + half, cy - half]);
    plot.heatmap(B.mask, G, G, { map: t => (t > 0.5 ? [0.78, 0.85, 0.95] : [1, 1, 1]), lo: 0, hi: 1, maxCells: 0, x0: 0, x1: G, y0: G, y1: 0 });
    plot.layer.appendChild(under);
    // staircase
    const path = (P: Pt[], style: string, close = true) => {
      const e = svgEl("path", { d: P.map((p, i) => (i ? "L" : "M") + plot.x(p[0]).toFixed(1) + "," + plot.y(p[1]).toFixed(1)).join("") + (close ? "Z" : ""), fill: "none" }, under);
      e.style.cssText = style;
    };
    path(corners, "stroke:var(--qb-muted,#888);stroke-width:1.5");
    // true outline
    for (const seg of outerSegments(polys)) path(seg, "stroke:var(--qb-c3,#2ca02c);stroke-width:1.5;stroke-dasharray:4 4", false);
    // curvature colour / comb
    const kap = S.map(s => sgn * s.kappa);
    const kmaxAbs = Math.max(0.05, ...kap.map(Math.abs).sort((a, b) => a - b).slice(0, Math.floor(0.98 * kap.length)));
    if (st.comb) {
      const sc = 4 / kmaxAbs, tips: Pt[] = [];
      S.forEach((s, i) => {
        const L = Math.hypot(...s.d1) || 1;
        // outward normal for a clockwise-on-screen curve (y down): (d1y, −d1x) × sign
        const nx = sgn * s.d1[1] / L, ny = -sgn * s.d1[0] / L, k = kap[i];
        const tip: Pt = [s.pt[0] + sc * k * nx, s.pt[1] + sc * k * ny];
        tips.push(tip);
        if (i % 2 === 0) path([s.pt, tip], "stroke:#e377c2;stroke-width:1.2;opacity:.8", false);
      });
      path(tips, "stroke:#e377c2;stroke-width:1.8");
    }
    if (st.colour) {
      path(S.map(s => s.pt), "stroke:var(--qb-fg,#222);stroke-width:7");
      for (let i = 0; i + 1 < S.length; i++) {
        const c = diverging(0.5 + 0.5 * Math.max(-1, Math.min(1, kap[i] / kmaxAbs)));
        path([S[i].pt, S[i + 1].pt], `stroke:rgb(${c.map(v => Math.round(255 * v)).join(",")});stroke-width:5;stroke-linecap:round`, false);
      }
    } else path(S.map(s => s.pt), "stroke:var(--qb-c1,#1f77b4);stroke-width:4");
    // Fourier descriptors
    let fLen = NaN;
    if (st.fourier) {
      const F = fourierOutline(pts, st.K);
      path(F, "stroke:var(--qb-c4,#ff7f0e);stroke-width:3");
      fLen = 0;
      for (let i = 0; i < F.length; i++) { const a = F[i], b = F[(i + 1) % F.length]; fLen += Math.hypot(b[0] - a[0], b[1] - a[1]); }
    }
    // curvature against arc length
    const sArr: number[] = [0];
    for (let i = 1; i < S.length; i++) sArr.push(sArr[i - 1] + Math.hypot(S[i].pt[0] - S[i - 1].pt[0], S[i].pt[1] - S[i - 1].pt[1]));
    const kl = Math.min(...kap), kh = Math.max(...kap), pad = 0.1 * (kh - kl + 0.05);
    kplot.clear();
    kplot.setDomain([0, sArr[sArr.length - 1]], [Math.min(-0.05, kl - pad), kh + 2.5 * pad]);
    kplot.hline(0, { color: "var(--qb-muted,#888)", width: 1.5, dash: "" });
    if (sh.r) kplot.hline(1 / sh.r, { color: "var(--qb-c3,#2ca02c)", text: `1/r = ${(1 / sh.r).toFixed(3)}` });
    kplot.line(sArr, kap, { color: "var(--qb-c1,#1f77b4)", width: 3 });
    const tl = trueLength(polys);
    ro.textContent =
      `length  edge count  ${B.edges.toFixed(1).padStart(7)}\n` +
      `        chain code  ${B.chain.toFixed(1).padStart(7)}\n` +
      `        spline      ${len.toFixed(1).padStart(7)}\n` +
      (st.fourier ? `        Fourier K=${String(st.K).padEnd(2)}  ${fLen.toFixed(1).padStart(7)}\n` : "") +
      `        true        ${tl.toFixed(1).padStart(7)}\n` +
      `∮κ ds = ${(tk * sgn / (2 * Math.PI)).toFixed(3)} × 2π   (m = ${m}, ${pts.length} edges)`;
  }

  load(p0.s);
}

/** Truncated Fourier descriptors: resample the closed polygon to 256 points by arc length,
 *  keep the frequencies −K … K of x + i y, transform back. */
function fourierOutline(P: Pt[], K: number): Pt[] {
  const N = 256, n = P.length, L = [0];
  for (let i = 0; i < n; i++) { const a = P[i], b = P[(i + 1) % n]; L.push(L[i] + Math.hypot(b[0] - a[0], b[1] - a[1])); }
  const re = new Float64Array(N), im = new Float64Array(N);
  let j = 0;
  for (let k = 0; k < N; k++) {
    const s = L[n] * k / N;
    while (j < n - 1 && L[j + 1] < s) j++;
    const a = P[j], b = P[(j + 1) % n], f = (s - L[j]) / (L[j + 1] - L[j] || 1);
    re[k] = a[0] + f * (b[0] - a[0]); im[k] = a[1] + f * (b[1] - a[1]);
  }
  fft1d(re, im);
  for (let k = K + 1; k < N - K; k++) { re[k] = 0; im[k] = 0; }
  fft1d(re, im, true);
  const sc = 1 / N;   // fft1d is unnormalised in both directions
  return Array.from({ length: N }, (_, k) => [re[k] * sc, im[k] * sc] as Pt);
}

/** Parts of the polygon outlines that are not inside another polygon (the union's outline). */
function outerSegments(polys: Poly[]): Pt[][] {
  const out: Pt[][] = [];
  polys.forEach((p, k) => {
    let cur: Pt[] = [];
    for (let i = 0; i <= p.length; i++) {
      const q = p[i % p.length];
      if (polys.some((o, l) => l !== k && pointIn(o, q[0], q[1]))) { if (cur.length > 1) out.push(cur); cur = []; }
      else cur.push([q[0], q[1]]);
    }
    if (cur.length > 1) out.push(cur);
  });
  return out;
}
function pointIn(p: Poly, x: number, y: number): boolean {
  let c = false;
  for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
    const [xi, yi] = p[i], [xj, yj] = p[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
  }
  return c;
}

/** Length of the outline of the union of the polygons: each edge counts where it lies outside
 *  every other polygon (sampled). */
function trueLength(polys: Poly[]): number {
  const inside = (p: Poly, x: number, y: number) => {
    let c = false;
    for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
      const [xi, yi] = p[i], [xj, yj] = p[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
    }
    return c;
  };
  let L = 0;
  polys.forEach((p, k) => {
    for (let i = 0; i < p.length; i++) {
      const a = p[i], b = p[(i + 1) % p.length], seg = Math.hypot(b[0] - a[0], b[1] - a[1]), S = 8;
      for (let s = 0; s < S; s++) {
        const t = (s + 0.5) / S, x = a[0] + t * (b[0] - a[0]), y = a[1] + t * (b[1] - a[1]);
        if (!polys.some((q, l) => l !== k && inside(q, x, y))) L += seg / S;
      }
    }
  });
  return L;
}

autoMount(NAME, mount);
