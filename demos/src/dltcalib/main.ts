// Camera calibration (lib/camera.ts, lib/bundle.ts). A hidden camera (f = 820 px, principal
// point (335, 245), 640 × 480 pixels) photographs a known target with seeded pixel noise and
// optional radial distortion. "3D target": two orthogonal checkerboard walls (72 corners);
// click corners to leave them out or take them in; the DLT (with or without Hartley
// normalisation) gives P, split into K and the pose. "Flat board": an 8 × 6 board in 1–5
// poses; homographies → K (Zhang); with 1 view there are too few equations, and views that are
// all nearly parallel leave the focal length undetermined. Both: refine the reprojection
// error by Levenberg–Marquardt (with or without k1, k2). Shows estimated vs true K, the RMS
// error, the residuals as arrows (magnified), and the singular values of the linear system.
//
// <div data-demo="dltcalib" data-preset="six|noise|unnormalised|coplanar|distortion|planar|parallel"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { project, V2, Camera } from "../lib/camera";
import { Plot } from "../lib/plot";
import { injectStyle } from "../triangulate/flatview";
import { IMG_W, IMG_H, K_TRUE, makeTarget3D, makePlanar, Target3D, PlanarViews, CalibProblem, pack, camOf, fixedMask, rmsOf, dltEstimate, zhangEstimate, lmStep } from "./model";

const NAME = "dltcalib";
const SEL = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${SEL} { flex-wrap: nowrap; gap: .8em; }
${SEL} .dc-main { flex: 0 1 520px; min-width: 280px; }
${SEL} .dc-main svg.dc-img { width: 100%; display: block; background: #f3f3f3; border-radius: 4px; cursor: pointer; }
${SEL} .dc-side { flex: 0 1 280px; min-width: 220px; display: flex; flex-direction: column; gap: .3em; }
${SEL} .qb-panel { flex: 1 1 260px; max-width: 330px; gap: .3em; }
${SEL} .qb-readout { font-size: 15px; }
${SEL} table.dc-k { border-collapse: collapse; font-size: 16px; font-family: ui-monospace, monospace; }
${SEL} table.dc-k td, ${SEL} table.dc-k th { padding: .05em .45em; text-align: right; border-bottom: 1px solid var(--qb-grid); }
${SEL} .dc-row { display: flex; flex-wrap: wrap; gap: .25em; align-items: center; }
${SEL} .dc-cap { font-size: 16px; color: var(--qb-muted); }
.reveal ${SEL} { font-size: 18px; }
.reveal ${SEL} .dc-main { flex: 0 0 500px; }
.reveal ${SEL} .dc-side { flex: 0 0 270px; }
.reveal ${SEL} .qb-panel { gap: .15em; }
.reveal ${SEL} .qb-btn { font-size: 16px; padding: .06em .35em; }
.reveal ${SEL} .qb-readout { font-size: 16px; padding: .2em .4em; }
.reveal ${SEL} table.dc-k { font-size: 17px; }
@media (max-width: 860px) { ${SEL} { flex-wrap: wrap; } }
`;

type Mode = "3d" | "planar";
interface State { mode: Mode; sigma: number; k1: number; normalise: boolean; modelDist: boolean; nViews: number; parallel: boolean; view: number; mag: number; sel: number[] | null }
const DEFAULT: State = { mode: "3d", sigma: 0.5, k1: 0, normalise: true, modelDist: false, nViews: 3, parallel: false, view: 0, mag: 30, sel: null };
interface Preset { id: string; label: string; s: Partial<State> }
const SIX = [0, 10, 26, 36, 59, 67];
const PRESETS: Preset[] = [
  { id: "six", label: "6 points, no noise: exact", s: { sigma: 0, sel: SIX } },
  { id: "noise", label: "noise 0.5 px, all 72 points", s: {} },
  { id: "unnormalised", label: "unnormalised DLT", s: { normalise: false } },
  { id: "coplanar", label: "all points on one plane: degenerate", s: { sel: Array.from({ length: 36 }, (_, i) => 36 + i) } },
  { id: "distortion", label: "distortion ignored: residual pattern", s: { k1: -0.3, sigma: 0.3 } },
  { id: "planar", label: "flat board: 1, 2, 3 views", s: { mode: "planar", nViews: 3 } },
  { id: "parallel", label: "parallel views: f undetermined", s: { mode: "planar", nViews: 3, parallel: true } },
];

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: State = { ...DEFAULT, ...(p0?.s ?? {}) };

  const main = htmlEl("div", "dc-main", "", host);
  const modeRow = htmlEl("div", "dc-row", "", main);
  const mBtns = [button(modeRow, "3D target (DLT)", () => { st.mode = "3d"; rebuild(); }), button(modeRow, "flat board (homographies)", () => { st.mode = "planar"; rebuild(); })];
  const viewRow = htmlEl("span", "dc-row", "", modeRow);
  const svg = svgEl("svg", { viewBox: `-10 -10 ${IMG_W + 20} ${IMG_H + 20}`, class: "dc-img" }, main);
  const g = svgEl("g", {}, svg);
  const cap = htmlEl("div", "dc-cap", "", main);
  const side = htmlEl("div", "dc-side", "", host);
  const table = htmlEl("table", "dc-k", "", side);
  const svPlot = new Plot(side, { width: 280, height: 205, x: [0.5, 12.5], y: [1e-4, 10], ylog: true, xlabel: "singular values of A", font: 16, xticks: [], margin: { l: 56, r: 8, t: 8, b: 44 } });
  const svBars = svPlot.bars([], [], { color: "var(--qb-c1)", barWidth: 0.7, base: 1e-6 });
  const ro = htmlEl("div", "qb-readout", "", side);

  const panel = htmlEl("div", "qb-panel", "", host);
  const pBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); rebuild(); } })));
  const sS = slider(panel, "noise σ (px)", 0, 2, 0.1, st.sigma, v => { st.sigma = v; rebuild(); });
  const kS = slider(panel, "distortion k1", -0.4, 0.2, 0.05, st.k1, v => { st.k1 = v; rebuild(); });
  const vS = slider(panel, "views (flat board)", 1, 5, 1, st.nViews, v => { st.nViews = v; st.view = Math.min(st.view, v - 1); rebuild(); });
  const normC = checkbox(panel, "normalise first (Hartley)", st.normalise, v => { st.normalise = v; linear(); });
  const parC = checkbox(panel, "nearly parallel views", st.parallel, v => { st.parallel = v; rebuild(); });
  const distC = checkbox(panel, "refine k1, k2 too", st.modelDist, v => { st.modelDist = v; draw(); });
  const stepRow = htmlEl("div", "dc-row", "", panel);
  button(stepRow, "linear", () => linear());
  button(stepRow, "refine: LM step", () => refine(1));
  button(stepRow, "refine ×20", () => refine(20));
  button(stepRow, "use all points", () => { st.sel = null; rebuild(); });
  const magS = slider(panel, "arrows ×", 5, 100, 5, st.mag, v => { st.mag = v; draw(); });
  button(panel, "reset", () => { Object.assign(st, DEFAULT, p0?.s ?? {}); pBtns.forEach(b => b.classList.remove("active")); sync(); rebuild(); });
  function sync(): void { sS.set(st.sigma); kS.set(st.k1); vS.set(st.nViews); normC.checked = st.normalise; parC.checked = st.parallel; distC.checked = st.modelDist; magS.set(st.mag); }

  let t3: Target3D, pv: PlanarViews, use: boolean[][] = [], theta: Float64Array | null = null, mu = 1e-3, steps = 0, S: Float64Array = new Float64Array(0), note = "", condNum = NaN;
  function rebuild(): void {
    if (st.mode === "3d") {
      t3 = makeTarget3D(st.sigma, st.k1);
      use = [t3.X.map((_, i) => (st.sel ? st.sel.includes(i) : true))];
    } else {
      pv = makePlanar(st.nViews, st.sigma, st.k1, st.parallel ? 0.04 : 1);
      use = pv.obs.map(o => o.map(() => true));
    }
    linear();
  }
  function problem(): CalibProblem { return st.mode === "3d" ? new CalibProblem(t3.X, [t3.obs], use) : new CalibProblem(pv.X, pv.obs, use); }
  function linear(): void {
    steps = 0; mu = 1e-3; note = ""; condNum = NaN;
    if (st.mode === "3d") {
      const e = dltEstimate(t3, use[0], st.normalise);
      if (!e) { theta = null; S = new Float64Array(0); note = "need at least 6 points"; draw(); return; }
      S = e.S; condNum = e.cond;
      theta = pack(e.K, 0, 0, [{ R: e.R, t: e.t }]);
      theta[4] = e.K[1];
      if (S[10] < 1e-6 * S[0]) note = "two (or more) null directions: degenerate";
    } else {
      const z = zhangEstimate(pv);
      S = z.S;
      if (!z.K || !z.poses) { theta = null; note = z.rows < 5 ? `${z.rows} equations for 5 unknowns: not enough views` : "no valid K (ω not positive definite)"; draw(); return; }
      theta = pack(z.K, 0, 0, z.poses);
      theta[4] = z.K[1];
      if (z.rows === 5) note = "2 views: zero skew assumed";
    }
    draw();
  }
  function refine(n: number): void {
    if (!theta) return;
    const prob = problem(), nv = st.mode === "3d" ? 1 : st.nViews, fixed = fixedMask(nv, st.modelDist);
    if (steps === 0) theta[4] = 0;
    for (let k = 0; k < n; k++) { const s = lmStep(prob, theta, { mu, lm: true, fixed }); theta = s.theta; mu = s.mu; steps++; }
    draw();
  }

  svg.addEventListener("pointerdown", ev => {
    if (st.mode !== "3d") return;
    const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    const q = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    let best = -1, bd = 18;
    t3.obs.forEach((o, i) => { const d = Math.hypot(o[0] - q.x, o[1] - q.y); if (d < bd) { bd = d; best = i; } });
    if (best < 0) return;
    use[0][best] = !use[0][best];
    st.sel = use[0].map((u, i) => (u ? i : -1)).filter(i => i >= 0);
    linear();
  });

  function drawGrid(cam: Camera, X: ArrayLike<number>[], nx: number, ny: number, offset: number): void {
    // lines between neighbouring corners, curved by the true distortion (sampled)
    const seg = (a: ArrayLike<number>, b: ArrayLike<number>) => {
      let d = "";
      for (let k = 0; k <= 8; k++) {
        const t = k / 8, P = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t], q = project(cam, P);
        d += (k ? "L" : "M") + q.u.toFixed(1) + "," + q.v.toFixed(1);
      }
      const p = svgEl("path", { d, fill: "none", "stroke-width": 1.2 }, g);
      p.style.stroke = "#b8b8b8";
    };
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const k = offset + j * nx + i;
      if (i + 1 < nx) seg(X[k], X[k + 1]);
      if (j + 1 < ny) seg(X[k], X[k + nx]);
    }
  }

  function draw(): void {
    mBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.mode === "3d")));
    viewRow.replaceChildren();
    if (st.mode === "planar") for (let v = 0; v < st.nViews; v++) { const b = button(viewRow, `view ${v + 1}`, () => { st.view = v; draw(); }); b.classList.toggle("active", v === st.view); }
    g.replaceChildren();
    const frame = svgEl("rect", { x: 0, y: 0, width: IMG_W, height: IMG_H, fill: "#fff", "stroke-width": 1.5 }, g);
    frame.style.stroke = "#888";
    const v = st.mode === "3d" ? 0 : st.view;
    const X = st.mode === "3d" ? t3.X : pv.X, obs = st.mode === "3d" ? t3.obs : pv.obs[v], trueCam = st.mode === "3d" ? t3.cam : pv.cams[v];
    if (st.mode === "3d") { drawGrid(trueCam, X, 6, 6, 0); drawGrid(trueCam, X, 6, 6, 36); } else drawGrid(trueCam, X, 8, 6, 0);
    const pp = svgEl("circle", { cx: K_TRUE[2], cy: K_TRUE[5], r: 5, fill: "none", "stroke-width": 2 }, g); pp.style.stroke = "#2ca02c";
    const est = theta ? camOf(theta, v) : null;
    const ar = st.mag;
    obs.forEach((o, i) => {
      const on = use[v][i];
      const c = svgEl("circle", { cx: o[0], cy: o[1], r: 5, "stroke-width": 2 }, g);
      c.style.fill = on ? "#1f77b4" : "#fff"; c.style.stroke = on ? "#1f77b4" : "#999";
      if (est && on) {
        const q = project(est, X[i]), ex = o[0] + ar * (q.u - o[0]), ey = o[1] + ar * (q.v - o[1]);
        const l = svgEl("line", { x1: o[0], y1: o[1], x2: ex, y2: ey, "stroke-width": 2.2, "stroke-linecap": "round" }, g);
        l.style.stroke = "#e8402a";
      }
    });
    if (est) { const pe = svgEl("path", { d: `M${est.K[2] - 8},${est.K[5]}h16M${est.K[2]},${est.K[5] - 8}v16`, "stroke-width": 2.5 }, g); pe.style.stroke = "#e8402a"; }
    cap.textContent = (st.mode === "3d" ? "click a corner to leave it out / take it in · " : `view ${v + 1} of ${st.nViews} · `) +
      `red: residual × ${ar} (estimate − observed) · green circle: true principal point, red cross: estimated`;

    // K table
    const Ke = est?.K, kE = theta ? [theta[5], theta[6]] : [NaN, NaN];
    const rows: [string, number, number][] = [["fx", K_TRUE[0], Ke ? Ke[0] : NaN], ["fy", K_TRUE[4], Ke ? Ke[4] : NaN], ["cx", K_TRUE[2], Ke ? Ke[2] : NaN], ["cy", K_TRUE[5], Ke ? Ke[5] : NaN], ["skew", 0, Ke ? Ke[1] : NaN], ["k1", st.k1, kE[0]]];
    table.replaceChildren();
    const hr = htmlEl("tr", "", "", table);
    ["", "true", "estimate", "diff"].forEach(h => htmlEl("th", "", h, hr));
    for (const [name, tv, ev] of rows) {
      const tr = htmlEl("tr", "", "", table);
      htmlEl("td", "", name, tr); htmlEl("td", "", tv.toFixed(name === "k1" ? 2 : 1), tr);
      htmlEl("td", "", Number.isFinite(ev) ? ev.toFixed(name === "k1" ? 3 : 1) : "–", tr);
      htmlEl("td", "", Number.isFinite(ev) ? (ev - tv).toFixed(name === "k1" ? 3 : 1) : "", tr);
    }
    // singular values
    const n = S.length;
    svPlot.setDomain([0.5, Math.max(n, 1) + 0.5], [1e-6 * (S[0] || 1), 3 * (S[0] || 1)]);
    svBars.set(Array.from({ length: n }, (_, k) => k + 1), Array.from(S, s => Math.max(s, 1e-7 * (S[0] || 1))));
    svPlot.setLabels({ xlabel: st.mode === "3d" ? "the 12 singular values of A" : "singular values (equations on ω)" });
    // readout
    let rms = NaN;
    if (theta) rms = rmsOf(problem().residuals(theta));
    const used = use.reduce((a, u) => a + u.filter(Boolean).length, 0);
    ro.textContent = `${used} points used · ${steps ? `LM: ${steps} steps` : "linear estimate"}\n` +
      `RMS reprojection error: ${Number.isFinite(rms) ? rms.toFixed(3) + " px" : "–"}` +
      (Number.isFinite(condNum) ? `\ncondition s1/s11: ${condNum.toExponential(2)}` : "") + (note ? `\n${note}` : "");
    void ({} as V2);
  }

  sync();
  if (p0) pBtns[PRESETS.indexOf(p0)].classList.add("active");
  rebuild();
}

autoMount(NAME, mount);
