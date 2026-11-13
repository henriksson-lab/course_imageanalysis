// Why ℓ1 makes zeros: the geometry of the lasso in two unknowns. Left: the coefficient plane
// with the data misfit ½(x − b)ᵀQ(x − b) as ellipses around the least-squares point b (a large
// drag handle); Q = AᵀA = [[1, ρ], [ρ, 1]] with ρ the correlation of the two columns (ρ = 0:
// round contours, an orthonormal A). The penalty ball through the solution (ℓ2 circle, ℓ1
// diamond, ℓ½ star, ℓ0: the two axes), the misfit contour that touches it, the solution and its
// path as λ runs from 0 to λ_max. Right: tab "transfer curves" (the scalar problems c(w) for the
// four penalties at the current λ) or tab "path" (a 6-coefficient problem with a fixed seed:
// coefficients against λ for ℓ1 and ℓ2 side by side, λ_max marked, current λ as a line).
// Conventions (lib/sparse.ts): ℓ1 λ|x|, ℓ2 (λ/2)x², ℓ0 (λ²/2)·count (threshold λ), ℓ½ λ√|x|.
// Auto-mounts on <div data-demo="lassogeom">. Optional: data-preset="…", data-tab="curves|path".
import { htmlEl, svgEl, slider, presets, button, autoMount, draggable, svgPoint } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { penalised2d, lassoScalar, misfit2, penaltyValue, PenaltyKind, Sym2, lassoPath, ridge, gaussianMatrix } from "../lib/sparse";
import { rng } from "../lib/random";

const NAME = "lassogeom";
const PENS: { p: PenaltyKind; label: string; col: string }[] = [
  { p: 2, label: "ℓ2", col: seriesColor(0) },
  { p: 1, label: "ℓ1", col: seriesColor(1) },
  { p: 0.5, label: "ℓ½", col: seriesColor(4) },
  { p: 0, label: "ℓ0", col: seriesColor(3) },
];
const colOf = (p: PenaltyKind) => PENS.find(q => q.p === p)!.col;
const labOf = (p: PenaltyKind) => PENS.find(q => q.p === p)!.label;

interface State { b: [number, number]; rho: number; p: PenaltyKind; lambda: number; tab: "curves" | "path" }
const DEF: State = { b: [3, 1], rho: 0, p: 1, lambda: 1, tab: "curves" };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "soft", label: "round contours: ℓ1 = soft threshold", s: { b: [3, 1], rho: 0, p: 1, lambda: 1, tab: "curves" } },
  { id: "corner", label: "the corner wins", s: { b: [3, 1], rho: 0.3, p: 1, lambda: 2, tab: "curves" } },
  { id: "corr", label: "correlated columns: ℓ1 picks one", s: { b: [2.6, -0.6], rho: 0.9, p: 1, lambda: 1, tab: "curves" } },
  { id: "ridge", label: "ℓ2 never reaches the axis", s: { b: [3, 1], rho: 0.3, p: 2, lambda: 2, tab: "curves" } },
  { id: "max", label: "λ ≥ λ_max: all zero", s: { b: [3, 1], rho: 0, p: 1, lambda: 3.2, tab: "curves" } },
  { id: "half", label: "p < 1: pointier, but two valleys", s: { b: [2, 1.8], rho: 0.6, p: 0.5, lambda: 1.1, tab: "curves" } },
  { id: "path", label: "the path: one at a time", s: { b: [3, 1], rho: 0, p: 1, lambda: 1.5, tab: "path" } },
];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; }
.qb-demo[data-demo="${NAME}"] .lg-plane { flex: 0 1 470px; min-width: 260px; }
.qb-demo[data-demo="${NAME}"] .lg-plane svg { touch-action: none; }
.qb-demo[data-demo="${NAME}"] .lg-right { flex: 1 1 330px; min-width: 0; max-width: 520px; display: flex; flex-direction: column; gap: .2em; }
.qb-demo[data-demo="${NAME}"] .lg-row { display: flex; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .lg-row > div { flex: 1 1 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 320px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .lg-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre; }
/* pages: plane | curves + readout | panel in one row of ~1075 px; plane capped by the window height */
.qb-demo[data-demo="${NAME}"] .lg-plane { flex: 0 1 390px; max-width: calc(100vh - 240px); }
.qb-demo[data-demo="${NAME}"] .lg-right { flex: 0 1 330px; max-width: 330px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; gap: .25em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-btn { font-size: 14px; padding: .12em .5em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-presets { gap: .25em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; padding: .35em .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .lg-plane { max-width: none; }
.reveal .qb-demo[data-demo="${NAME}"] .lg-right { flex-basis: 330px; max-width: 360px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; gap: .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .lg-plane { flex-basis: 440px; }
.reveal .qb-demo[data-demo="${NAME}"] .lg-right { flex-basis: 330px; max-width: 360px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex-basis: 300px; max-width: 320px; gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .lg-lab { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .25em .5em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

// ---- the 6-coefficient path problem (fixed seed), scaled so that λ_max = 4 ----
const PM = 24, PN = 6;
const PATH = (() => {
  const r = rng(2026), A = gaussianMatrix(PM, PN, r);
  // correlate columns 0/1 and 3/4 a little
  for (let i = 0; i < PM; i++) { A[i * PN + 1] = 0.6 * A[i * PN] + 0.8 * A[i * PN + 1]; A[i * PN + 4] = 0.5 * A[i * PN + 3] + 0.87 * A[i * PN + 4]; }
  const xt = [2.5, 0, -1.5, 1, 0, 0.5], y = new Float64Array(PM);
  for (let i = 0; i < PM; i++) { let s = 0.15 * r.norm(); for (let j = 0; j < PN; j++) s += A[i * PN + j] * xt[j]; y[i] = s; }
  let lm = 0;
  for (let j = 0; j < PN; j++) { let s = 0; for (let i = 0; i < PM; i++) s += A[i * PN + j] * y[i]; lm = Math.max(lm, Math.abs(s)); }
  const k = 4 / lm; for (let i = 0; i < PM; i++) y[i] *= k;
  const lams = Float64Array.from({ length: 100 }, (_, i) => 4 * Math.pow(10, -2.5 * i / 99) * 1.2);
  const l1 = lassoPath(A, PM, PN, y, lams), l2 = Array.from(lams, l => ridge(A, PM, PN, y, l));
  return { lams, l1, l2, lmax: 4 };
})();

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const init: State = { ...DEF, b: [...DEF.b] as [number, number] };
  if (host.dataset.tab === "path") init.tab = "path";
  if (p0) Object.assign(init, p0.s, { b: [...(p0.s.b ?? DEF.b)] });
  const st: State = { ...init, b: [...init.b] as [number, number] };

  // ---- plane ----
  const VB = 470, D = 4.3;
  const box = htmlEl("div", "lg-plane", "", host);
  const svg = svgEl("svg", { viewBox: `0 0 ${VB} ${VB}`, width: "100%", role: "img", "aria-label": "coefficient plane" }, box);
  const sx = (x: number) => VB / 2 + x * (VB / 2 - 12) / D, sy = (y: number) => VB / 2 - y * (VB / 2 - 12) / D;
  const toW = (ev: PointerEvent): [number, number] => { const [px, py] = svgPoint(svg, ev); return [(px - VB / 2) * D / (VB / 2 - 12), -(py - VB / 2) * D / (VB / 2 - 12)]; };
  const gGrid = svgEl("g", {}, svg), gEll = svgEl("g", {}, svg), gBall = svgEl("g", {}, svg), gPath = svgEl("g", {}, svg), gTop = svgEl("g", {}, svg);
  for (let v = -4; v <= 4; v++) {
    svgEl("line", { x1: sx(v), x2: sx(v), y1: sy(-D), y2: sy(D), class: "grid" }, gGrid);
    svgEl("line", { x1: sx(-D), x2: sx(D), y1: sy(v), y2: sy(v), class: "grid" }, gGrid);
  }
  svgEl("line", { x1: sx(-D), x2: sx(D), y1: sy(0), y2: sy(0), class: "axis", "stroke-width": 1.5 }, gGrid);
  svgEl("line", { x1: sx(0), x2: sx(0), y1: sy(-D), y2: sy(D), class: "axis", "stroke-width": 1.5 }, gGrid);
  const t1 = svgEl("text", { x: sx(D) - 4, y: sy(0) - 8, "text-anchor": "end", "font-size": 20 }, gGrid); t1.textContent = "x₁";
  const t2 = svgEl("text", { x: sx(0) + 8, y: sy(D) + 18, "font-size": 20 }, gGrid); t2.textContent = "x₂";
  const lsDot = svgEl("circle", { r: 13, fill: "#fff", stroke: "var(--qb-fg)", "stroke-width": 3, class: "handle" }, gTop);
  const lsLab = svgEl("text", { "font-size": 18 }, gTop); lsLab.textContent = "least squares";
  const solDot = svgEl("circle", { r: 9, stroke: "#fff", "stroke-width": 2 }, gTop);
  draggable(lsDot as unknown as SVGElement, toW, (x, y) => {
    st.b = [Math.max(-4, Math.min(4, Math.round(x * 10) / 10)), Math.max(-4, Math.min(4, Math.round(y * 10) / 10))];
    changed(); draw();
  });

  // ---- right side ----
  const right = htmlEl("div", "lg-right", "", host);
  const rightBox = htmlEl("div", "", "", right);
  const ro = htmlEl("div", "qb-readout", "", right);

  // ---- panel ----
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "lg-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEF, p.s, { b: [...(p.s.b ?? DEF.b)] }); sync(); draw(); } })));
  htmlEl("div", "lg-lab", "penalty", panel);
  const pRow = htmlEl("div", "qb-presets", "", panel);
  const pBtns = PENS.map(q => button(pRow, q.label, () => { st.p = q.p; changed(); sync(); draw(); }));
  const lamS = slider(panel, "λ", 0, 5, 0.05, st.lambda, v => { st.lambda = v; changed(); draw(); });
  const rhoS = slider(panel, "correlation ρ", -0.9, 0.9, 0.05, st.rho, v => { st.rho = v; changed(); draw(); });
  const tRow = htmlEl("div", "qb-presets", "", panel);
  const tBtns = [button(tRow, "transfer curves", () => { st.tab = "curves"; sync(); draw(); }), button(tRow, "path, 6 coefficients", () => { st.tab = "path"; sync(); draw(); })];
  button(panel, "reset", () => { Object.assign(st, init, { b: [...init.b] }); sync(); draw(); presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p0)); });
  htmlEl("div", "qb-note", "Drag the least-squares point. Same λ, same threshold: ℓ0 uses the weight λ²/2 per nonzero, so with round contours it is the hard threshold at λ.", panel);
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");

  function changed(): void { presetBtns.forEach(b => b.classList.remove("active")); }
  function sync(): void {
    lamS.set(st.lambda); rhoS.set(st.rho);
    pBtns.forEach((b, i) => b.classList.toggle("active", PENS[i].p === st.p));
    tBtns.forEach((b, i) => b.classList.toggle("active", ["curves", "path"][i] === st.tab));
  }

  const Q = (): Sym2 => [1, st.rho, 1];
  function ellipsePts(c: number): string {
    const r = st.rho, s = Math.sqrt(2 * Math.max(c, 0)), a = s / Math.sqrt(1 + r), bb = s / Math.sqrt(1 - r);
    const pts: string[] = [];
    for (let k = 0; k <= 96; k++) {
      const t = 2 * Math.PI * k / 96, u = a * Math.cos(t), v = bb * Math.sin(t);
      const x = st.b[0] + (u + v) / Math.SQRT2, y = st.b[1] + (u - v) / Math.SQRT2;
      pts.push(`${sx(x).toFixed(1)},${sy(y).toFixed(1)}`);
    }
    return pts.join(" ");
  }
  function ballPts(x: [number, number], p: PenaltyKind): string {
    const pts: string[] = [];
    if (p === 2) {
      const R = Math.hypot(x[0], x[1]);
      for (let k = 0; k <= 96; k++) { const t = 2 * Math.PI * k / 96; pts.push(`${sx(R * Math.cos(t)).toFixed(1)},${sy(R * Math.sin(t)).toFixed(1)}`); }
    } else if (p === 1) {
      const R = Math.abs(x[0]) + Math.abs(x[1]);
      for (const [u, v] of [[R, 0], [0, R], [-R, 0], [0, -R], [R, 0]]) pts.push(`${sx(u)},${sy(v)}`);
    } else {
      const s = (Math.sqrt(Math.abs(x[0])) + Math.sqrt(Math.abs(x[1]))) ** 2;
      for (const [qx, qy] of [[1, 1], [-1, 1], [-1, -1], [1, -1]]) for (let k = 0; k <= 40; k++) {
        const t = qx * qy > 0 ? k / 40 : 1 - k / 40;
        pts.push(`${sx(qx * s * t * t).toFixed(1)},${sy(qy * s * (1 - t) * (1 - t)).toFixed(1)}`);
      }
    }
    return pts.join(" ");
  }
  const lamMax1 = () => { const q = Q(); return Math.max(Math.abs(q[0] * st.b[0] + q[1] * st.b[1]), Math.abs(q[1] * st.b[0] + q[2] * st.b[1])); };

  let plots: { curves?: Plot; p1?: Plot; p2?: Plot; vl?: ReturnType<Plot["vline"]>[] } = {};
  let shownTab = "";

  function draw(): void {
    const q = Q(), x = penalised2d(q, st.b, st.lambda, st.p), col = colOf(st.p);
    // ellipses: a few levels + the touching one
    gEll.replaceChildren();
    const f0 = misfit2(q, st.b, x);
    for (const c of [0.25, 1, 2.25, 4, 6.25, 9]) svgEl("polygon", { points: ellipsePts(c), fill: "none", stroke: "var(--qb-muted)", "stroke-width": 1.2, opacity: 0.6 }, gEll);
    if (f0 > 1e-9) svgEl("polygon", { points: ellipsePts(f0), fill: "none", stroke: "var(--qb-fg)", "stroke-width": 2.5 }, gEll);
    // ball
    gBall.replaceChildren();
    const nz = (Math.abs(x[0]) > 1e-9 ? 1 : 0) + (Math.abs(x[1]) > 1e-9 ? 1 : 0);
    if (st.p === 0) {
      for (const [a, b2, c2, d2] of [[-D, 0, D, 0], [0, -D, 0, D]]) svgEl("line", { x1: sx(a), y1: sy(b2), x2: sx(c2), y2: sy(d2), stroke: col, "stroke-width": 6, opacity: 0.35 }, gBall);
      svgEl("circle", { cx: sx(0), cy: sy(0), r: 7, fill: col, opacity: 0.6 }, gBall);
    } else if (nz > 0) {
      svgEl("polygon", { points: ballPts(x, st.p), fill: col, "fill-opacity": 0.15, stroke: col, "stroke-width": 3 }, gBall);
    }
    // path λ: 0 → λ_max (ℓ1) or to 5·λ_max (others)
    gPath.replaceChildren();
    const lm = lamMax1(), top = st.p === 1 ? lm : st.p === 2 ? 40 : 1.6 * lm + 0.5;
    const pts: [number, number][] = [];
    for (let k = 0; k <= 160; k++) { const l = st.p === 2 ? Math.pow(10, -2 + 3.6 * k / 160) : top * k / 160; pts.push(penalised2d(q, st.b, l, st.p)); }
    if (st.p === 1 || st.p === 2) {
      svgEl("polyline", { points: pts.map(p => `${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(" "), fill: "none", stroke: col, "stroke-width": 2.5, "stroke-dasharray": "6 4" }, gPath);
    } else for (const p of pts) svgEl("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 2.5, fill: col }, gPath);
    // dots
    lsDot.setAttribute("cx", String(sx(st.b[0]))); lsDot.setAttribute("cy", String(sy(st.b[1])));
    lsLab.setAttribute("x", String(Math.min(sx(st.b[0]) + 16, VB - 120))); lsLab.setAttribute("y", String(sy(st.b[1]) - 16));
    solDot.setAttribute("cx", String(sx(x[0]))); solDot.setAttribute("cy", String(sy(x[1]))); solDot.setAttribute("fill", col);

    // right side
    if (shownTab !== st.tab) {
      rightBox.replaceChildren(); plots = {}; shownTab = st.tab;
      if (st.tab === "curves") {
        plots.curves = new Plot(rightBox, { width: 420, height: 400, x: [-4, 4], y: [-4, 4], xlabel: "w (data coefficient)", ylabel: "c (estimate)", xticks: [-4, -2, 0, 2, 4], yticks: [-4, -2, 0, 2, 4] });
      } else {
        const mk = (title: string) => new Plot(htmlEl("div", "", "", rightBox), { width: 420, height: title === "ℓ2 (ridge)" ? 225 : 200, x: [0.012, 4.8], y: [-2, 3], xlog: true, xlabel: title === "ℓ2 (ridge)" ? "λ (log scale)" : "", ylabel: title, xticks: [0.1, 1, 4], yticks: [-2, 0, 2], font: 22, margin: { t: 6, b: title === "ℓ2 (ridge)" ? 56 : 30 } });
        plots.p1 = mk("ℓ1 (lasso)"); plots.p2 = mk("ℓ2 (ridge)");
        for (let j = 0; j < PN; j++) {
          plots.p1.line(PATH.lams, PATH.l1.map(v => v[j]), { color: seriesColor(j), width: 2.5 });
          plots.p2.line(PATH.lams, PATH.l2.map(v => v[j]), { color: seriesColor(j), width: 2.5 });
        }
        plots.p1.vline(PATH.lmax, { color: "var(--qb-muted)", dash: "5 4", width: 1.5, text: "λmax" });
        plots.vl = [];
      }
    }
    if (st.tab === "curves" && plots.curves) {
      const pc = plots.curves, ws = Float64Array.from({ length: 401 }, (_, i) => -4 + 8 * i / 400);
      pc.clear();
      pc.line([-4, 4], [-4, 4], { color: "var(--qb-muted)", width: 1.5, dash: "4 4" });
      for (const pp of PENS) if (pp.p !== st.p) pc.line(ws, Float64Array.from(ws, w => lassoScalar(w, st.lambda, pp.p)), { color: pp.col, width: 2, label: pp.label, opacity: 0.75 });
      pc.line(ws, Float64Array.from(ws, w => lassoScalar(w, st.lambda, st.p)), { color: col, width: 5, label: labOf(st.p) });
      pc.legend(undefined, "tl");
    } else if (plots.p1 && plots.p2) {
      plots.vl?.forEach(m => m.remove());
      const l = Math.max(st.lambda, 0.013);
      plots.vl = [plots.p1.vline(l, { color: "var(--qb-fg)", width: 2 }), plots.p2.vline(l, { color: "var(--qb-fg)", width: 2 })];
    }

    // readout
    const fmt = (v: number) => (Math.abs(v) < 5e-10 ? "0" : v.toFixed(2));
    const pen = penaltyValue(x, st.lambda, st.p);
    let pathLine = "";
    if (st.tab === "path") {
      const k = PATH.lams.reduce((bi, l, i) => (Math.abs(Math.log(l / Math.max(st.lambda, 0.013))) < Math.abs(Math.log(PATH.lams[bi] / Math.max(st.lambda, 0.013))) ? i : bi), 0);
      pathLine = `\n6-coef. problem at λ: ${PATH.l1[k].filter(v => v !== 0).length} of 6 nonzero (ℓ1), ${PATH.l2[k].filter(v => Math.abs(v) > 1e-12).length} (ℓ2)`;
    }
    ro.textContent = `${labOf(st.p)}, λ = ${st.lambda.toFixed(2)}, ρ = ${st.rho.toFixed(2)}\n` +
      `least squares b = (${fmt(st.b[0])}, ${fmt(st.b[1])})\n` +
      `solution x = (${fmt(x[0])}, ${fmt(x[1])})   zeros: ${2 - nz}\n` +
      `misfit ${f0.toFixed(3)}   penalty ${pen.toFixed(3)}\n` +
      `λ_max (ℓ1) = ${lm.toFixed(2)}` + pathLine;
  }

  sync();
  draw();
}

autoMount(NAME, mount);
