// The snake's internal forces as a linear map: a closed polygon of n points evolves under
// −A x alone (no image), with the explicit step x ← (I − τA) x or the semi-implicit step
// (A + γI) x_new = γ x, γ = 1/τ. A is circulant, so its eigenvectors are the Fourier modes of
// the contour: the plots show the eigenvalues λ_k, the factor each mode is multiplied by per
// step, and the contour's own amplitude per mode (initial grey, current colour).
// Auto-mounts on <div data-demo="snakespectrum">. Options: data-preset="<id>",
// data-view="contour|matrix" (what the left panel shows first).
import { htmlEl, slider, presets, button, autoMount, svgEl, svgPoint, Slider } from "../lib/ui";
import { Plot, seriesColor, heatmapGrid } from "../lib/plot";
import { diverging } from "../lib/colour";
import { rng } from "../lib/random";
import { Contour, circleContour, cloneContour, snakeMatrix, snakeEigenvalue, amplification, explicitStep, implicitSolver, implicitStep, modeAmplitudes, lambdaMax, resample, ImplicitSolver } from "../lib/snake";

const NAME = "snakespectrum";

type ShapeId = "noisy" | "square" | "star";
interface St { shape: ShapeId; n: number; alpha: number; beta: number; tau: number; rule: "explicit" | "implicit" }
interface Preset { id: string; label: string; s: St }

const PRESETS: Preset[] = [
  { id: "alpha", label: "α only: shrinks to a point", s: { shape: "noisy", n: 16, alpha: 0.2, beta: 0, tau: 1, rule: "implicit" } },
  { id: "beta", label: "β only: smooths, keeps its size longer", s: { shape: "noisy", n: 16, alpha: 0, beta: 0.2, tau: 1, rule: "implicit" } },
  { id: "limit", label: "explicit at the limit", s: { shape: "noisy", n: 16, alpha: 0.25, beta: 0, tau: 1.95, rule: "explicit" } },
  { id: "beyond", label: "explicit just beyond: explodes", s: { shape: "noisy", n: 16, alpha: 0.25, beta: 0, tau: 2.1, rule: "explicit" } },
  { id: "huge", label: "semi-implicit, huge τ: still stable", s: { shape: "noisy", n: 16, alpha: 0.25, beta: 0, tau: 50, rule: "implicit" } },
  { id: "small", label: "n = 6: the matrix by hand", s: { shape: "noisy", n: 6, alpha: 1, beta: 0, tau: 0.2, rule: "explicit" } },
];

/** Start shapes (radius about 1, y down). */
function makeShape(id: ShapeId, n: number): Contour {
  if (id === "noisy") {
    const c = circleContour(0, 0, 1, n), r = rng(5);
    for (let i = 0; i < n; i++) { c.x[i] += 0.12 * r.norm(); c.y[i] += 0.12 * r.norm(); }
    return c;
  }
  if (id === "square") {
    const sq = { x: Float64Array.from([-0.85, 0.85, 0.85, -0.85]), y: Float64Array.from([-0.85, -0.85, 0.85, 0.85]) };
    return resample(sq, n);
  }
  const m = 10, x = new Float64Array(m), y = new Float64Array(m);
  for (let i = 0; i < m; i++) { const t = -Math.PI / 2 + 2 * Math.PI * i / m, r = i % 2 ? 0.45 : 1.15; x[i] = r * Math.cos(t); y[i] = r * Math.sin(t); }
  return resample({ x, y }, n);
}

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] .ss-left { flex: 0 1 380px; min-width: 240px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .ss-left svg.ss-contour { width: 100%; aspect-ratio: 1; touch-action: none; background: var(--qb-panel); border-radius: 4px; }
[data-demo="${NAME}"] .ss-cap { font-size: 17px; text-align: center; }
[data-demo="${NAME}"] .ss-plots { flex: 0 1 360px; min-width: 240px; display: flex; flex-direction: column; gap: .1em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 330px; gap: .35em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .ss-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
[data-demo="${NAME}"] .ss-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 16px; gap: .5em; }
.reveal .qb-demo[data-demo="${NAME}"] .ss-left { flex-basis: 360px; }
.reveal .qb-demo[data-demo="${NAME}"] .ss-plots { flex-basis: 340px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function onScreen(el: HTMLElement): boolean {
  if (el.closest("section[aria-hidden='true'], section.past, section.future")) return false;
  const r = el.getBoundingClientRect();
  return r.width > 0 && r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth;
}

const f2 = (v: number) => (Number.isFinite(v) ? (Math.abs(v) >= 1e4 ? v.toExponential(1) : v.toFixed(2)) : "∞");

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: St = { ...p0.s };
  let view: "contour" | "matrix" = host.dataset.view === "matrix" ? "matrix" : "contour";
  let start = makeShape(st.shape, st.n), c = cloneContour(start), iter = 0, playing = false;
  let solver: ImplicitSolver | null = null;

  // ---- layout ----
  const left = htmlEl("div", "ss-left", "", host);
  const cap = htmlEl("div", "ss-cap", "", left);
  const R = 1.9;
  const svg = svgEl("svg", { viewBox: `${-R} ${-R} ${2 * R} ${2 * R}`, class: "ss-contour" }, left);
  const gStart = svgEl("g", {}, svg), gCur = svgEl("g", {}, svg);
  const matBox = htmlEl("div", "", "", left);

  const plots = htmlEl("div", "ss-plots", "", host);
  const pl = { width: 380, height: 175, font: 18, margin: { t: 30, b: 34, l: 62 } };
  const pLam = new Plot(plots, { ...pl, x: [-0.5, 8.5], y: [0, 1], title: "eigenvalue λₖ of A", xticks: 6 });
  const pAmp = new Plot(plots, { ...pl, x: [-0.5, 8.5], y: [-1.5, 1.5], title: "factor per step (mode k)", xticks: 6 });
  const pMode = new Plot(plots, { ...pl, height: 200, margin: { t: 30, b: 50, l: 62 }, x: [0.5, 8.5], y: [1e-4, 2], ylog: true, title: "amplitude of mode k in the contour", xlabel: "mode k (wiggles around the curve)", xticks: 6 });
  const mLam = pLam.bars([], [], { color: seriesColor(0) });
  const mLim = pLam.hline(1, { color: seriesColor(1), text: "2/τ" });
  const ampStyle: { color: string; colors?: string[] } = { color: seriesColor(2) };
  const mAmp = pAmp.bars([], [], ampStyle);
  pAmp.hline(1, { color: "var(--qb-muted, #777)", width: 1.5 }); pAmp.hline(-1, { color: "var(--qb-muted, #777)", width: 1.5 });
  pAmp.hline(0, { color: "var(--qb-axis, #444)", width: 1, dash: "" });
  const mInit = pMode.bars([], [], { color: "var(--qb-grid, #ccc)", fillOpacity: 1, base: 1e-4 });
  const mCur = pMode.scatter([], [], { color: seriesColor(1), r: 6 });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ss-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const ctl = htmlEl("div", "ss-row", "", panel);
  const playBtn = button(ctl, "▶ play", () => setPlaying(!playing));
  button(ctl, "step", () => { setPlaying(false); step(1); });
  button(ctl, "+10", () => { setPlaying(false); step(10); });
  button(ctl, "reset", () => restart());
  const sl: Record<string, Slider> = {};
  sl.alpha = slider(panel, "α", 0, 1, 0.01, st.alpha, v => set({ alpha: v }));
  sl.beta = slider(panel, "β", 0, 1, 0.01, st.beta, v => set({ beta: v }));
  // τ on a log scale from 0.05 to 100
  const tOf = (u: number) => Number((0.05 * 2000 ** (u / 100)).toPrecision(2)), uOf = (t: number) => 100 * Math.log(t / 0.05) / Math.log(2000);
  sl.tau = slider(panel, "τ", 0, 100, 0.5, uOf(st.tau), u => set({ tau: tOf(u) }), u => String(Number((0.05 * 2000 ** (u / 100)).toPrecision(3))));
  sl.n = slider(panel, "n", 4, 32, 1, st.n, v => { st.n = v; start = resample(start, v); restart(); });
  const row = htmlEl("div", "ss-row", "", panel);
  const ruleBtns = (["explicit", "implicit"] as const).map(r => button(row, r === "explicit" ? "explicit" : "semi-implicit", () => set({ rule: r })));
  const row2 = htmlEl("div", "ss-row", "", panel);
  const shapeBtns = (["noisy", "square", "star"] as const).map(s => button(row2, s === "noisy" ? "noisy circle" : s, () => { st.shape = s; start = makeShape(s, st.n); restart(); }));
  const row3 = htmlEl("div", "ss-row", "", panel);
  const viewBtns = (["contour", "matrix"] as const).map(v => button(row3, v === "contour" ? "show contour" : "show matrix A", () => { view = v; draw(); }));
  const ro = htmlEl("div", "qb-readout", "", panel);

  function set(p: Partial<St>): void {
    Object.assign(st, p);
    presetBtns.forEach(b => b.classList.remove("active"));
    draw();
  }
  function load(p: Preset): void {
    Object.assign(st, p.s);
    start = makeShape(st.shape, st.n);
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    restart();
  }
  function restart(): void { setPlaying(false); c = cloneContour(start); iter = 0; draw(); }

  function step(k: number): void {
    for (let i = 0; i < k; i++) {
      if (exploded()) break;
      if (st.rule === "explicit") c = explicitStep(c, st.alpha, st.beta, st.tau, null);
      else {
        const g = 1 / st.tau;
        if (!solver || solver.n !== st.n || solver.alpha !== st.alpha || solver.beta !== st.beta || solver.gamma !== g) solver = implicitSolver(st.n, st.alpha, st.beta, g);
        c = implicitStep(c, solver, null);
      }
      iter++;
    }
    draw();
  }
  const exploded = () => { for (let i = 0; i < c.x.length; i++) if (!(Math.abs(c.x[i]) < 1e6 && Math.abs(c.y[i]) < 1e6)) return true; return false; };

  function setPlaying(v: boolean): void { playing = v; playBtn.textContent = v ? "❚❚ pause" : "▶ play"; if (v) requestAnimationFrame(tick); }
  let last = 0;
  function tick(t: number): void {
    if (!playing) return;
    if (!onScreen(host)) { setPlaying(false); return; }
    if (t - last > 120) { last = t; step(1); if (exploded()) { setPlaying(false); return; } }
    requestAnimationFrame(tick);
  }

  // ---- drawing ----
  const path = (q: Contour) => {
    let d = "";
    for (let i = 0; i < q.x.length; i++) d += (i ? "L" : "M") + q.x[i].toFixed(4) + "," + q.y[i].toFixed(4);
    return d + "Z";
  };
  function drawContour(): void {
    gStart.replaceChildren(); gCur.replaceChildren();
    svgEl("path", { d: path(start), fill: "none", stroke: "var(--qb-muted, #999)", "stroke-width": 0.02, "stroke-dasharray": "0.06 0.05" }, gStart);
    svgEl("circle", { cx: 0, cy: 0, r: 0.025, fill: "var(--qb-muted, #999)" }, gStart);
    if (exploded()) return;
    const big = c.x.some((v, i) => Math.abs(v) > R || Math.abs(c.y[i]) > R);
    svgEl("path", { d: path(c), fill: "none", stroke: seriesColor(0), "stroke-width": 0.03, "stroke-linejoin": "round" }, gCur);
    if (big) { const t = svgEl("text", { x: 0, y: -R + 0.2, "text-anchor": "middle", "font-size": 0.15 }, gCur); t.textContent = "(partly outside the view)"; }
    for (let i = 0; i < c.x.length; i++) {
      const h = svgEl("circle", { cx: c.x[i], cy: c.y[i], r: 0.065, fill: i === 0 ? seriesColor(3) : seriesColor(0), stroke: "var(--qb-bg, #fff)", "stroke-width": 0.015 }, gCur);
      h.style.cursor = "grab";
      h.addEventListener("pointerdown", ev => {
        ev.preventDefault(); ev.stopPropagation(); setPlaying(false);
        // listeners on window: draw() replaces the dragged circle while it moves
        const mv = (e: PointerEvent) => { const [x, y] = svgPoint(svg, e); c.x[i] = Math.max(-R, Math.min(R, x)); c.y[i] = Math.max(-R, Math.min(R, y)); start = cloneContour(c); iter = 0; presetBtns.forEach(b => b.classList.remove("active")); draw(); };
        const upf = () => { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", upf); window.removeEventListener("pointercancel", upf); };
        window.addEventListener("pointermove", mv); window.addEventListener("pointerup", upf); window.addEventListener("pointercancel", upf);
      });
    }
  }

  function drawMatrix(): void {
    matBox.replaceChildren();
    const n = st.n, A = snakeMatrix(n, st.alpha, st.beta);
    let m = 0;
    for (const v of A) m = Math.max(m, Math.abs(v));
    const cell = Math.min(46, 360 / n);
    const hm = heatmapGrid(matBox, A, n, n, { map: diverging, lo: -m || -1, hi: m || 1, cell, numbers: n <= 12, gridLines: true, fmt: v => (Math.abs(v) < 1e-12 ? "0" : Number(v.toFixed(2)).toString()) });
    hm.svg.style.maxWidth = "380px";
  }

  function draw(): void {
    const n = st.n, K = Math.floor(n / 2);
    sl.alpha.set(st.alpha); sl.beta.set(st.beta); sl.tau.set(uOf(st.tau)); sl.n.set(st.n);
    ruleBtns.forEach((b, i) => b.classList.toggle("active", ["explicit", "implicit"][i] === st.rule));
    shapeBtns.forEach((b, i) => b.classList.toggle("active", ["noisy", "square", "star"][i] === st.shape));
    viewBtns.forEach((b, i) => b.classList.toggle("active", ["contour", "matrix"][i] === view));
    svg.style.display = view === "contour" ? "" : "none";
    matBox.style.display = view === "matrix" ? "" : "none";
    if (view === "contour") drawContour(); else drawMatrix();
    cap.textContent = view === "contour" ? `step ${iter}${exploded() ? ": exploded" : ""} (dashed: start; drag the points)` : `A for n = ${n}, α = ${st.alpha}, β = ${st.beta}`;

    const ks = Array.from({ length: K + 1 }, (_, k) => k);
    const lam = ks.map(k => snakeEigenvalue(k, n, st.alpha, st.beta)), lm = lambdaMax(n, st.alpha, st.beta);
    const lhi = Math.max(lm, st.rule === "explicit" ? 2 / st.tau : 0, 1e-3) * 1.3;
    const stp = K <= 8 ? 1 : K <= 16 ? 2 : 4, tk = ks.filter(k => k % stp === 0);
    pLam.opts.xticks = tk; pAmp.opts.xticks = tk; pMode.opts.xticks = tk.filter(k => k > 0);
    pLam.setDomain([-0.5, K + 0.5], [0, lhi]);
    mLam.set(ks, lam);
    if (st.rule === "explicit" && 2 / st.tau <= lhi) { mLim.style.opacity = 1; mLim.set(2 / st.tau); } else { mLim.style.opacity = 0; mLim.set(-1); }
    const amp = lam.map(l => amplification(l, st.tau, st.rule));
    const lo = Math.min(-1.3, ...amp.map(a => a * 1.1)), hi = 1.3;
    pAmp.setDomain([-0.5, K + 0.5], [Math.max(lo, -4), hi]);
    mAmp.set(ks, amp.map(a => Math.max(a, -4)));
    ampStyle.colors = amp.map(a => (Math.abs(a) > 1 + 1e-9 ? "var(--qb-c2, #d62728)" : "var(--qb-c3, #2ca02c)"));
    mAmp.render();
    const a0 = modeAmplitudes(start), a1 = exploded() ? a0.map(() => NaN) : modeAmplitudes(c);
    const kk = ks.slice(1), init = kk.map(k => a0[k]), cur = kk.map(k => a1[k]);
    const fin = [...init, ...cur].filter(v => Number.isFinite(v) && v > 0);
    const ymax = Math.max(2, ...fin) * 1.5;
    pMode.setDomain([0.5, K + 0.5], [1e-4, ymax]);
    mInit.set(kk, init.map(v => Math.max(v, 1e-4)));
    mCur.set(kk, cur.map(v => (v > 1e-4 ? v : NaN)));

    const kmax = lam.indexOf(Math.max(...lam));
    ro.textContent = [
      `λ_max = 4α + 16β = ${f2(lm)}  (mode ${kmax})`,
      st.rule === "explicit" ? `τ·λ_max = ${f2(st.tau * lm)}: ${st.tau * lm <= 2 ? "stable" : "unstable (> 2)"}` : `γ = 1/τ = ${f2(1 / st.tau)}: stable for any τ`,
      `factor of mode ${kmax}: ${f2(amplification(lm, st.tau, st.rule))}`,
      `factor of mode 1:  ${f2(amplification(lam[1] ?? 0, st.tau, st.rule))}`,
    ].join("\n");
  }

  load(p0);
  if (host.dataset.view === "matrix") { view = "matrix"; draw(); }
}

autoMount(NAME, mount);
