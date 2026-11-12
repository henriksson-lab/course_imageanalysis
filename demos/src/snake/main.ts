// Active contours (snakes): a closed polygon pulled by internal forces (elasticity α,
// rigidity β) and by the image (edge force, GVF, balloon). Drag on the image to place a
// circle (or draw a closed curve), then play or step. Overlays: the forces on every point
// (internal blue, external red), the external force field, and the energy per iteration.
// Optional side panel: the matrix A (or A + γI) of the semi-implicit step.
// Auto-mounts on <div data-demo="snake">. Options:
//   data-preset="<id>"          start preset (see PRESETS)
//   data-forces="edge|all"      edge: edge force only (unit 1); all: + GVF and balloon (unit 2)
//   data-matrix="on"            show the matrix panel
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, svgPoint, Slider } from "../lib/ui";
import { Grey, drawGrey, cloneGrey } from "../lib/imageops";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { Plot, seriesColor, heatmapGrid } from "../lib/plot";
import { diverging } from "../lib/colour";
import { Contour, circleContour, resample, snakeMatrix, lambdaMax, cloneContour, sampleGrid } from "../lib/snake";
import { N, ImageId, cleanImage, addNoise } from "./images";
import { Settings, DEFAULTS, Sim, makeFields, Fields, forces } from "./model";

const NAME = "snake";

interface Preset {
  id: string; label: string; image: ImageId; noise: number;
  start: { cx: number; cy: number; r: number };
  s: Partial<Settings>;
  all?: boolean;     // only with data-forces="all"
  edgeOnly?: boolean; // only without
}

const PRESETS: Preset[] = [
  { id: "disk", label: "fits the disk", image: "disk", noise: 0.05, start: { cx: 63.5, cy: 63.5, r: 45 }, s: {} },
  { id: "alpha", label: "α too large: shrinks away", image: "disk", noise: 0.05, start: { cx: 63.5, cy: 63.5, r: 45 }, s: { alpha: 1 }, edgeOnly: true },
  { id: "beta", label: "β too large: rounds the square", image: "square", noise: 0.05, start: { cx: 63.5, cy: 63.5, r: 50 }, s: { beta: 2 }, edgeOnly: true },
  { id: "sigma", label: "σ too small: stuck on noise", image: "disk", noise: 0.15, start: { cx: 63.5, cy: 63.5, r: 45 }, s: { sigma: 0.5 }, edgeOnly: true },
  { id: "far", label: "too far: does not move", image: "disk", noise: 0.05, start: { cx: 63.5, cy: 63.5, r: 58 }, s: { alpha: 0.01 } },
  { id: "tau", label: "τ too large: explodes", image: "disk", noise: 0.05, start: { cx: 63.5, cy: 63.5, r: 45 }, s: { rule: "explicit", tau: 2 } },
  { id: "u", label: "concavity: cannot enter the U", image: "u", noise: 0.05, start: { cx: 63.5, cy: 65.5, r: 56 }, s: { n: 100, alpha: 0.03, beta: 0.01, resample: 10 } },
  { id: "ugvf", label: "U: GVF enters the concavity", image: "u", noise: 0.05, start: { cx: 63.5, cy: 65.5, r: 56 }, s: { n: 100, alpha: 0.03, beta: 0.01, resample: 10, force: "gvf" }, all: true },
  { id: "balloon", label: "balloon from inside", image: "disk", noise: 0.05, start: { cx: 63.5, cy: 63.5, r: 8 }, s: { kappa: 0.3 }, all: true },
  { id: "leak", label: "balloon leaks through the weak edge", image: "weak", noise: 0.05, start: { cx: 63.5, cy: 63.5, r: 8 }, s: { kappa: 0.6 }, all: true },
  { id: "fargvf", label: "far start: GVF captures it", image: "disk", noise: 0.05, start: { cx: 63.5, cy: 63.5, r: 58 }, s: { alpha: 0.01, force: "gvf" }, all: true },
  { id: "blobs", label: "two blobs: one snake bridges them", image: "blobs", noise: 0.05, start: { cx: 63.5, cy: 63.5, r: 58 }, s: { n: 50 } },
  { id: "cell", label: "micrograph: nucleus", image: "cell", noise: 0, start: { cx: 74, cy: 57, r: 44 }, s: { sigma: 3, n: 60 } },
];

const IMAGES: { id: ImageId; label: string }[] = [
  { id: "disk", label: "disk" }, { id: "square", label: "square" }, { id: "u", label: "U" },
  { id: "blobs", label: "two blobs" }, { id: "weak", label: "weak edge" }, { id: "cell", label: "micrograph" },
];

const STYLE = `
[data-demo="${NAME}"] { gap: .9em; }
[data-demo="${NAME}"] .sn-fig { flex: 0 1 430px; min-width: 260px; margin: 0; }
[data-demo="${NAME}"] .sn-fig figcaption { font-size: 17px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] .sn-stage { position: relative; width: 100%; aspect-ratio: 1; }
[data-demo="${NAME}"] .sn-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .sn-stage svg { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; cursor: crosshair; }
[data-demo="${NAME}"] .sn-mid { flex: 0 1 300px; min-width: 220px; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 360px; gap: .35em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .sn-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .sn-credit { font-size: 12px; color: var(--qb-muted); }
[data-demo="${NAME}"] .sn-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
[data-demo="${NAME}"] .sn-matcap { font-size: 15px; text-align: center; }
.reveal [data-demo="${NAME}"] { font-size: 16px; gap: .6em; }
.reveal [data-demo="${NAME}"] .sn-fig { flex-basis: 420px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .15em; max-width: 380px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 13px; }
.reveal [data-demo="${NAME}"] .sn-mid { flex-basis: 270px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

/** Is the element on screen (and not on a hidden reveal.js slide)? */
function onScreen(el: HTMLElement): boolean {
  if (el.closest("section[aria-hidden='true'], section.past, section.future")) return false;
  const r = el.getBoundingClientRect();
  return r.width > 0 && r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth;
}

const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "∞");

function mount(host: HTMLElement): void {
  injectStyle();
  const all = host.dataset.forces === "all";
  const showMatrix = host.dataset.matrix === "on";
  const visible = PRESETS.filter(p => (all ? !p.edgeOnly : !p.all));
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? visible[0];

  // ---- state ----
  const st = { image: p0.image as ImageId, noise: p0.noise, s: { ...DEFAULTS, ...p0.s } as Settings, start: circleContour(p0.start.cx, p0.start.cy, p0.start.r, p0.s.n ?? DEFAULTS.n), view: "image" as "image" | "edge", place: "circle" as "circle" | "draw", arrows: true, field: false };
  let img: Grey = cleanImage("disk");
  let cellImg: Grey | null = null;
  let F: Fields | null = null;
  let sim: Sim | null = null;
  let playing = false;

  // ---- layout ----
  const fig = htmlEl("figure", "sn-fig", "", host);
  const cap = htmlEl("figcaption", "", "", fig);
  const stage = htmlEl("div", "sn-stage", "", fig);
  const cv = htmlEl("canvas", "", "", stage);
  cv.width = N; cv.height = N;
  const svg = svgEl("svg", { viewBox: `-0.5 -0.5 ${N} ${N}` }, stage);
  const gField = svgEl("g", {}, svg), gStart = svgEl("g", {}, svg), gSnake = svgEl("g", {}, svg), gForce = svgEl("g", {}, svg), gDraw = svgEl("g", {}, svg);
  const credit = htmlEl("div", "sn-credit", CELL_CREDIT, fig);

  const mid = htmlEl("div", "sn-mid", "", host);
  const ePlot = new Plot(mid, { width: 360, height: 230, x: [0, 50], y: [-1, 1], xlabel: "iteration", ylabel: "energy", font: 18 });
  const lInt = ePlot.line([], [], { color: seriesColor(0), label: "internal" });
  const lExt = ePlot.line([], [], { color: seriesColor(1), label: "image" });
  const lTot = ePlot.line([], [], { color: "var(--qb-fg, #222)", label: "total", width: 2.5 });
  ePlot.legend(undefined, "tr");
  const ro = htmlEl("div", "qb-readout", "", mid);
  let matBox: HTMLElement | null = null, matCap: HTMLElement | null = null;
  if (showMatrix) {
    matCap = htmlEl("div", "sn-matcap", "", mid);
    matBox = htmlEl("div", "", "", mid);
  }

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "sn-lab", "presets", panel);
  const presetBtns = presets(panel, visible.map(p => ({ label: p.label, apply: () => load(p) })));
  const ctl = htmlEl("div", "sn-row", "", panel);
  const playBtn = button(ctl, "▶ play", () => setPlaying(!playing));
  button(ctl, "step", () => { setPlaying(false); doSteps(1); });
  button(ctl, "+10", () => { setPlaying(false); doSteps(10); });
  button(ctl, "reset", () => restart());
  const sl: Record<string, Slider> = {};
  sl.alpha = slider(panel, "α elasticity", 0, 2, 0.01, st.s.alpha, v => setS({ alpha: v }));
  sl.beta = slider(panel, "β rigidity", 0, 3, 0.01, st.s.beta, v => setS({ beta: v }));
  sl.tau = slider(panel, "τ step", 0.05, 4, 0.05, st.s.tau, v => setS({ tau: v }));
  sl.sigma = slider(panel, "σ smoothing", 0.5, 6, 0.5, st.s.sigma, v => setS({ sigma: v }));
  sl.w = slider(panel, "w image", 0, 8, 0.1, st.s.w, v => setS({ w: v }));
  sl.n = slider(panel, "n points", 8, 160, 1, st.s.n, v => { st.s.n = v; st.start = resample(st.start, v); restart(); });
  if (all) sl.kappa = slider(panel, "κ balloon", -1, 1.5, 0.05, st.s.kappa, v => setS({ kappa: v }));
  sl.resample = slider(panel, "resample every", 0, 50, 1, st.s.resample, v => setS({ resample: v }), v => (v === 0 ? "off" : String(v)));
  sl.noise = slider(panel, "noise", 0, 0.3, 0.01, st.noise, v => { st.noise = v; makeImage(); });
  const row1 = htmlEl("div", "sn-row", "", panel);
  const ruleBtns = (["explicit", "implicit"] as const).map(r => button(row1, r === "explicit" ? "explicit" : "semi-implicit", () => setS({ rule: r })));
  const forceBtns = all ? (["edge", "gvf"] as const).map(f => button(row1, f === "edge" ? "edge force" : "GVF", () => setS({ force: f }))) : [];
  const row2 = htmlEl("div", "sn-row", "", panel);
  const imgSel = htmlEl("select", "qb-btn", "", row2);
  for (const im of IMAGES) { const o = htmlEl("option", "", im.label, imgSel); o.value = im.id; }
  imgSel.addEventListener("change", () => { st.image = imgSel.value as ImageId; imgSel.blur(); makeImage(); });
  const viewBtns = (["image", "edge"] as const).map(v => button(row2, v === "image" ? "image" : "edge map", () => { st.view = v; drawImage(); sync(); }));
  const row3 = htmlEl("div", "sn-row", "", panel);
  const placeBtns = (["circle", "draw"] as const).map(m => button(row3, m === "circle" ? "drag: circle" : "drag: draw", () => { st.place = m; sync(); }));
  checkbox(row3, "forces", st.arrows, v => { st.arrows = v; drawOverlay(); });
  checkbox(row3, "field", st.field, v => { st.field = v; drawField(); });
  const note = htmlEl("div", "qb-note", "Drag on the image to place a new start (a circle, or draw a closed curve). Blue arrows: internal force, red: image force (+ balloon), both × 10.", panel);
  void note;

  function setS(p: Partial<Settings>): void {
    Object.assign(st.s, p);
    if (sim) sim.s = st.s;
    if (p.sigma !== undefined || p.force !== undefined) refreshFields();
    presetBtns.forEach(b => b.classList.remove("active"));
    sync(); drawOverlay(); drawField(); drawMatrix(); readout();
  }

  function sync(): void {
    for (const k of Object.keys(sl)) if (k !== "noise") sl[k].set((st.s as unknown as Record<string, number>)[k]);
    sl.noise.set(st.noise);
    ruleBtns.forEach((b, i) => b.classList.toggle("active", ["explicit", "implicit"][i] === st.s.rule));
    forceBtns.forEach((b, i) => b.classList.toggle("active", ["edge", "gvf"][i] === st.s.force));
    viewBtns.forEach((b, i) => b.classList.toggle("active", ["image", "edge"][i] === st.view));
    placeBtns.forEach((b, i) => b.classList.toggle("active", ["circle", "draw"][i] === st.place));
    imgSel.value = st.image;
    playBtn.textContent = playing ? "❚❚ pause" : "▶ play";
  }

  function load(p: Preset): void {
    setPlaying(false);
    st.image = p.image; st.noise = p.noise; st.s = { ...DEFAULTS, ...p.s };
    st.start = circleContour(p.start.cx, p.start.cy, p.start.r, st.s.n);
    presetBtns.forEach((b, i) => b.classList.toggle("active", visible[i] === p));
    makeImage();
  }

  // ---- image and fields ----
  function makeImage(): void {
    if (st.image === "cell" && !cellImg) {
      loadCell(N).then(g => { cellImg = { width: g.width, height: g.height, data: g.data.map(v => v / 255) }; makeImage(); });
      return;
    }
    const base = st.image === "cell" ? cloneGrey(cellImg!) : cleanImage(st.image);
    img = st.noise > 0 ? addNoise(base, st.noise) : base;
    credit.style.visibility = st.image === "cell" ? "visible" : "hidden";
    F = null;
    refreshFields();
    restart();
  }

  function refreshFields(): void {
    F = makeFields(img, st.s.sigma, st.s.force === "gvf", st.s.mu, F);
    if (sim) sim.F = F;
    drawImage(); drawField();
  }

  function drawImage(): void {
    if (st.view === "edge" && F) drawGrey(cv, F.f, { mode: "clip", lo: 0, hi: 1 });
    else drawGrey(cv, img, { mode: "clip", lo: 0, hi: 1 });
  }

  function restart(): void {
    setPlaying(false);
    if (!F) return;
    sim = new Sim(cloneContour(st.start), st.s, F);
    drawStart(); drawOverlay(); drawMatrix(); readout(); drawPlot(); sync();
  }

  // ---- drawing ----
  const pathOf = (c: Contour) => {
    let d = "";
    for (let i = 0; i < c.x.length; i++) {
      if (!Number.isFinite(c.x[i]) || !Number.isFinite(c.y[i])) return "";
      d += (i ? "L" : "M") + c.x[i].toFixed(2) + "," + c.y[i].toFixed(2);
    }
    return d + "Z";
  };
  function arrow(g: SVGGElement, x: number, y: number, dx: number, dy: number, col: string, w = 0.6): void {
    const L = Math.hypot(dx, dy);
    if (!(L > 0.3) || !Number.isFinite(L)) return;
    const ux = dx / L, uy = dy / L, hl = Math.min(L, 2.2), x1 = x + dx, y1 = y + dy, bx = x1 - ux * hl, by = y1 - uy * hl;
    svgEl("line", { x1: x, y1: y, x2: bx, y2: by, stroke: col, "stroke-width": w }, g);
    svgEl("polygon", { points: `${x1},${y1} ${bx - uy * hl * 0.45},${by + ux * hl * 0.45} ${bx + uy * hl * 0.45},${by - ux * hl * 0.45}`, fill: col }, g);
  }

  function drawStart(): void {
    gStart.replaceChildren();
    svgEl("path", { d: pathOf(st.start), fill: "none", stroke: "#7fd3ff", "stroke-width": 0.6, "stroke-dasharray": "2 1.5", opacity: 0.9 }, gStart);
  }

  function drawOverlay(): void {
    gSnake.replaceChildren(); gForce.replaceChildren();
    if (!sim) return;
    const c = sim.c;
    if (sim.exploded()) { cap.textContent = `iteration ${sim.iter}: the snake has exploded`; return; }
    svgEl("path", { d: pathOf(c), fill: "none", stroke: "#ffd400", "stroke-width": 1.1, "stroke-linejoin": "round" }, gSnake);
    for (let i = 0; i < c.x.length; i++) svgEl("circle", { cx: c.x[i], cy: c.y[i], r: c.x.length > 80 ? 0.7 : 1, fill: "#ff8c00" }, gSnake);
    if (st.arrows && F) {
      const fr = forces(c, st.s, F), k = 10;
      for (let i = 0; i < c.x.length; i++) {
        arrow(gForce, c.x[i], c.y[i], k * fr.internal.x[i], k * fr.internal.y[i], "#3fa9ff");
        const ex = fr.ext.x[i] + (fr.balloon ? fr.balloon.x[i] : 0), ey = fr.ext.y[i] + (fr.balloon ? fr.balloon.y[i] : 0);
        arrow(gForce, c.x[i], c.y[i], k * ex, k * ey, "#ff3b3b");
      }
    }
    cap.textContent = `iteration ${sim.iter}, n = ${c.x.length}, ${st.s.rule === "explicit" ? "explicit" : "semi-implicit"}`;
  }

  function drawField(): void {
    gField.replaceChildren();
    if (!st.field || !F) return;
    const fld = st.s.force === "gvf" && F.gvf ? F.gvf : F.edge;
    let m = 0;
    for (let i = 0; i < fld.fx.length; i++) m = Math.max(m, Math.hypot(fld.fx[i], fld.fy[i]));
    const step = 6;
    for (let y = step / 2; y < N; y += step) for (let x = step / 2; x < N; x += step) {
      const fx = sampleGrid(fld.fx, N, N, x, y), fy = sampleGrid(fld.fy, N, N, x, y), L = Math.hypot(fx, fy);
      if (L < 0.02 * m) { svgEl("circle", { cx: x, cy: y, r: 0.35, fill: "#9be7a0" }, gField); continue; }
      const k = 4.6 * Math.sqrt(L / m) / L;
      arrow(gField, x - fx * k / 2, y - fy * k / 2, fx * k, fy * k, "#5ee36a", 0.45);
    }
  }

  function readout(): void {
    if (!sim) return;
    const e = sim.history[sim.history.length - 1], lm = lambdaMax(st.s.n, st.s.alpha, st.s.beta);
    const lines = [
      `E internal ${fmt(e.int, 1).padStart(9)}`,
      `E image    ${fmt(e.ext, 1).padStart(9)}`,
      `E total    ${fmt(e.int + e.ext, 1).padStart(9)}`,
      `λmax = 4α+16β = ${fmt(lm, 2)}`,
      st.s.rule === "explicit" ? `τ·λmax = ${fmt(st.s.tau * lm, 2)}  (${st.s.tau * lm < 2 ? "stable: < 2" : "unstable: > 2"})` : `semi-implicit: stable for any τ`,
    ];
    ro.textContent = lines.join("\n");
  }

  function drawPlot(): void {
    if (!sim) return;
    const H = sim.history, n0 = sim.iter - H.length + 1;
    const xs = H.map((_, i) => n0 + i), a = H.map(h => h.int), b = H.map(h => h.ext), t = H.map(h => h.int + h.ext);
    const fin = [...a, ...b, ...t].filter(Number.isFinite).filter(v => Math.abs(v) < 1e6);
    let lo = Math.min(0, ...fin), hi = Math.max(1, ...fin);
    const pad = 0.08 * (hi - lo); lo -= pad; hi += pad;
    ePlot.setDomain([Math.max(0, n0), Math.max(50, sim.iter)], [lo, hi]);
    lInt.set(xs, a); lExt.set(xs, b); lTot.set(xs, t);
  }

  function drawMatrix(): void {
    if (!matBox || !matCap) return;
    const n = st.s.n, imp = st.s.rule === "implicit", gamma = 1 / st.s.tau;
    const A = snakeMatrix(n, st.s.alpha, st.s.beta);
    if (imp) for (let i = 0; i < n; i++) A[i * n + i] += gamma;
    let m = 0;
    for (const v of A) m = Math.max(m, Math.abs(v));
    matBox.replaceChildren();
    const cell = Math.max(2, Math.min(40, 300 / n));
    const hm = heatmapGrid(matBox, A, n, n, { map: diverging, lo: -m, hi: m, cell, numbers: n <= 12, gridLines: n <= 40, fmt: v => (Math.abs(v) < 1e-12 ? "0" : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2).replace(/\.?0+$/, "")) });
    hm.svg.style.maxWidth = "300px";
    // outline the cyclic corner entries (the wrap-around of the closed contour)
    const off = 4, corner = (r: number, c: number) => svgEl("rect", { x: off + c * cell, y: off + r * cell, width: cell, height: cell, fill: "none", stroke: "#ff8c00", "stroke-width": Math.max(1, cell / 10) }, hm.svg);
    if (n >= 5) { corner(0, n - 2); corner(0, n - 1); corner(1, n - 1); corner(n - 2, 0); corner(n - 1, 0); corner(n - 1, 1); }
    matCap.textContent = imp ? `A + γI (n = ${n}, γ = 1/τ = ${fmt(gamma, 2)}); orange: wrap-around` : `A (n = ${n}); orange: wrap-around entries`;
  }

  // ---- running ----
  let frame = 0;
  function doSteps(k: number): void {
    if (!sim) return;
    for (let i = 0; i < k; i++) sim.step();
    drawOverlay(); readout(); drawPlot();
  }
  function setPlaying(v: boolean): void {
    playing = v;
    playBtn.textContent = playing ? "❚❚ pause" : "▶ play";
    if (playing) requestAnimationFrame(tick);
  }
  function tick(): void {
    if (!playing || !sim) return;
    if (!onScreen(host)) { setPlaying(false); return; }
    for (let i = 0; i < 2; i++) sim.step();
    frame++;
    drawOverlay();
    if (frame % 3 === 0) { readout(); drawPlot(); }
    if (sim.exploded()) { readout(); drawPlot(); setPlaying(false); return; }
    requestAnimationFrame(tick);
  }

  // ---- placing a start contour ----
  let drag: { x0: number; y0: number; pts: [number, number][] } | null = null;
  svg.addEventListener("pointerdown", ev => {
    ev.preventDefault();
    svg.setPointerCapture(ev.pointerId);
    setPlaying(false);
    const [x, y] = svgPoint(svg, ev);
    drag = { x0: x, y0: y, pts: [[x, y]] };
  });
  svg.addEventListener("pointermove", ev => {
    if (!drag) return;
    const [x, y] = svgPoint(svg, ev);
    drag.pts.push([x, y]);
    gDraw.replaceChildren();
    if (st.place === "circle") {
      const r = Math.hypot(x - drag.x0, y - drag.y0);
      svgEl("circle", { cx: drag.x0, cy: drag.y0, r, fill: "none", stroke: "#7fd3ff", "stroke-width": 0.8 }, gDraw);
    } else svgEl("path", { d: "M" + drag.pts.map(p => p[0].toFixed(1) + "," + p[1].toFixed(1)).join("L"), fill: "none", stroke: "#7fd3ff", "stroke-width": 0.8 }, gDraw);
  });
  const up = (ev: PointerEvent) => {
    if (!drag) return;
    const [x, y] = svgPoint(svg, ev);
    gDraw.replaceChildren();
    if (st.place === "circle") {
      const r = Math.max(3, Math.hypot(x - drag.x0, y - drag.y0));
      st.start = circleContour(drag.x0, drag.y0, r, st.s.n);
    } else if (drag.pts.length > 4) {
      st.start = resample({ x: Float64Array.from(drag.pts, p => p[0]), y: Float64Array.from(drag.pts, p => p[1]) }, st.s.n);
    }
    drag = null;
    presetBtns.forEach(b => b.classList.remove("active"));
    restart();
  };
  svg.addEventListener("pointerup", up);
  svg.addEventListener("pointercancel", () => { drag = null; gDraw.replaceChildren(); });

  sync();
  load(p0);
}

autoMount(NAME, mount);
