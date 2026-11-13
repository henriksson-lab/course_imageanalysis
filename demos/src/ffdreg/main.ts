// Non-rigid registration with a B-spline free-form deformation. Fixed image: a 128² crop of the
// micrograph; moving image: the fixed one deformed by a seeded smooth random field (known), with
// its contrast optionally changed (inverted, a non-monotone "second stain") and noise. Drag
// control points to deform by hand, or press "optimise": gradient descent on similarity
// (SSD / NCC / MI) + λ·bending, optionally coarse to fine (grid twice as coarse first).
// Tabs: checkerboard of fixed and registered, difference, moving (with the deformed grid),
// registered, the Jacobian determinant (folding in black). Plot: cost against iteration.
//
// <div data-demo="ffdreg" data-preset="drag|ssd|fold|inverted|coarse|blob" data-run></div>  (data-run: start optimising at once)
import { autoMount, htmlEl, slider, presets, button, checkbox, svgEl, svgPoint } from "../lib/ui";
import { Grey, cropGrey, fromFunction } from "../lib/imageops";
import {
  FFDGrid, makeGrid, ffdDisplacement, knotPos, jacobianDeterminant, randomGrid, deformForTruth, warpMoving,
  FFDRegistration, Measure, remap, Remap, REMAP_NAMES, displacementError, similarityCost, bendingEnergy,
} from "../lib/deform";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { rng } from "../lib/random";
import { Plot } from "../lib/plot";
import { diverging } from "../lib/colour";
import { ImgView } from "../canny/imgview";
import { injectCss } from "../lkflow/flowui";

const NAME = "ffdreg", S = `.qb-demo[data-demo="${NAME}"]`;
const CSS = `
${S} { gap: .9em; flex-wrap: wrap; }
${S} .fr-main { flex: 0 1 400px; min-width: 240px; }
${S} .fr-main .ev-svg { pointer-events: auto; }
${S} .fr-mid { flex: 0 1 300px; min-width: 200px; display: flex; flex-direction: column; gap: .3em; }
${S} .qb-panel { flex: 1 1 300px; max-width: 380px; gap: .3em; }
${S} .fr-tabs { display: flex; flex-wrap: wrap; gap: .25em; margin-bottom: .3em; }
${S} .fr-tabs .qb-btn { padding: .12em .45em; }
${S} .qb-readout { font-size: 15px; }
${S} .fr-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
${S} .fr-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
.reveal ${S} { font-size: 18px; flex-wrap: nowrap; }
.reveal ${S} .fr-main { flex: 0 0 390px; }
.reveal ${S} .fr-mid { flex: 0 0 300px; }
.reveal ${S} .qb-panel { flex: 1 1 380px; gap: .15em; }
.reveal ${S} .qb-btn { font-size: 16px; padding: .06em .38em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .4em; }
.reveal ${S} .fr-lab { font-size: 16px; }
.reveal ${S} .ev-credit { display: none; }
@media (max-width: 860px) { ${S} { flex-wrap: wrap; } }
`;

type Tab = "checker" | "diff" | "moving" | "registered" | "jacobian";
const TABS: [Tab, string][] = [["checker", "checkerboard"], ["diff", "difference"], ["moving", "moving + grid"], ["registered", "registered"], ["jacobian", "Jacobian det."]];
const MEASURES: [Measure, string][] = [["ssd", "SSD"], ["ncc", "NCC"], ["mi", "mutual information"]];
const MAPS: Remap[] = ["identity", "inverted", "nonmonotone"];
const SPACINGS = [6, 8, 12, 16, 24, 32];
interface State { scene: "warp" | "blob"; map: Remap; noise: number; measure: Measure; lambda: number; spacing: number; coarse: boolean; tab: Tab }
const DEFAULT: State = { scene: "warp", map: "identity", noise: 0.03, measure: "ssd", lambda: 0, spacing: 16, coarse: true, tab: "moving" };
const PRESETS: { id: string; label: string; s: Partial<State>; run?: boolean; drag?: boolean }[] = [
  { id: "drag", label: "drag one control point: a local bump", s: { tab: "moving" }, drag: true },
  { id: "ssd", label: "optimise with SSD", s: { measure: "ssd", tab: "checker" }, run: true },
  { id: "fold", label: "fine grid, λ = 0: folding", s: { spacing: 6, noise: 0.25, coarse: false, tab: "jacobian" }, run: true },
  { id: "inverted", label: "inverted contrast: SSD fails, MI works", s: { map: "inverted", measure: "ssd", tab: "checker" }, run: true },
  { id: "coarse", label: "coarse to fine", s: { spacing: 8, lambda: 10, coarse: true, tab: "checker" }, run: true },
  { id: "blob", label: "too free: it explains away a growing blob", s: { scene: "blob", spacing: 8, tab: "jacobian" }, run: true },
];
const N = 128;
const LAMS = [0, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000];

function mount(host: HTMLElement): void {
  injectCss(NAME, CSS);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(p0?.s ?? {}) });
  const st = start();

  const main = htmlEl("div", "fr-main", "", host);
  const tabs = htmlEl("div", "fr-tabs", "", main);
  const tabBtns = TABS.map(([t, l]) => button(tabs, l, () => { st.tab = t; draw(); }));
  const view = new ImgView(main, "");
  htmlEl("div", "ev-credit", CELL_CREDIT, main);

  const mid = htmlEl("div", "fr-mid", "", host);
  htmlEl("div", "fr-lab", "cost = similarity + λ·bending", mid);
  const cplot = new Plot(mid, { width: 300, height: 230, x: [0, 50], y: [0, 1], xlabel: "iteration", font: 18, margin: { l: 58, r: 10, t: 10, b: 58 } });
  const cline = cplot.line([], [], { color: "var(--qb-c1)", width: 3 });
  const ro = htmlEl("div", "qb-readout", "", mid);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "fr-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => applyPreset(p) })));
  htmlEl("div", "fr-lab", "similarity measure", panel);
  const measBtns = presets(panel, MEASURES.map(([m, l]) => ({ label: l, apply: () => { st.measure = m; stop(); draw(); } })));
  htmlEl("div", "fr-lab", "moving image contrast", panel);
  const mapBtns = presets(panel, MAPS.map(m => ({ label: REMAP_NAMES[m], apply: () => { st.map = m; makeImages(); } })));
  const lamS = slider(panel, "λ (bending)", 0, LAMS.length - 1, 1, LAMS.indexOf(st.lambda), k => { st.lambda = LAMS[k]; stop(); draw(); }, k => String(LAMS[k]));
  const spS = slider(panel, "grid spacing (px)", 0, SPACINGS.length - 1, 1, SPACINGS.indexOf(st.spacing), k => { st.spacing = SPACINGS[k]; resetGrid(); }, k => String(SPACINGS[k]));
  const noS = slider(panel, "noise", 0, 0.3, 0.01, st.noise, v => { st.noise = v; makeImages(); });
  const coC = checkbox(panel, "coarse to fine (start with a grid twice as coarse)", st.coarse, v => { st.coarse = v; });
  const row = htmlEl("div", "fr-row", "", panel);
  const runB = button(row, "optimise", () => { if (timer) stop(); else run(); });
  button(row, "1 step", () => { stop(); ensureReg(); stepOnce(); draw(); });
  button(row, "zero grid", () => { stop(); resetGrid(); });
  button(row, "reset", () => { stop(); Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); makeImages(); });
  function sync(): void { lamS.set(LAMS.indexOf(st.lambda)); spS.set(SPACINGS.indexOf(st.spacing)); noS.set(st.noise); coC.checked = st.coarse; }
  function applyPreset(p: typeof PRESETS[number]): void {
    stop(); Object.assign(st, DEFAULT, p.s); sync(); makeImages();
    if (p.drag && grid) {
      const k = Math.round((64 - grid.x0) / grid.spacing), l = Math.round((64 - grid.y0) / grid.spacing);
      grid.cx[l * grid.nx + k] = 9; refresh();
    }
    if (p.run) run();
  }

  // ---- images ----
  let cell: Grey | null = null, F: Grey | null = null, M: Grey | null = null;
  type Field = { u: Float32Array<ArrayBufferLike>; v: Float32Array<ArrayBufferLike> };
  let truth: Field = { u: new Float32Array(N * N), v: new Float32Array(N * N) };
  loadCell().then(g => { cell = g; if (p0?.drag || (p0?.run && host.dataset.run !== undefined)) applyPreset(p0); else makeImages(); });
  function makeImages(): void {
    if (!cell) return;
    stop();
    const c = cropGrey(cell, 64, 64, N, N);
    let f: Grey = { ...c, data: c.data.map(v => v / 255) };
    if (st.scene === "warp") {
      truth = ffdDisplacement(randomGrid(N, N, 32, 3, 7), N, N);
      M = deformForTruth(f, truth.u, truth.v);
    } else {
      // a bright blob that grows between the two images; no deformation of anything else
      const blob = (r: number) => fromFunction(N, N, (x, y) => { const d = Math.hypot(x - 92, y - 40); return Math.max(f.data[y * N + x], 1 / (1 + Math.exp((d - r) / 0.8))); });
      truth = { u: new Float32Array(N * N), v: new Float32Array(N * N) };
      M = blob(13); f = blob(8);
    }
    F = f;
    const R = rng(3);
    M = { ...M, data: M.data.map(v => Math.max(0, Math.min(1, remap(v, st.map) + st.noise * R.norm()))) };
    resetGrid();
  }

  // ---- the grid, registration and drawing ----
  let grid: FFDGrid | null = null, reg: FFDRegistration | null = null, disp: Field = { u: new Float32Array(N * N), v: new Float32Array(N * N) }, W: Grey | null = null;
  let history: number[] = [];
  function resetGrid(): void { stop(); grid = makeGrid(N, N, st.spacing); reg = null; history = []; refresh(); }
  function refresh(): void {
    if (!grid || !M || !F) return;
    disp = ffdDisplacement(grid, N, N);
    W = warpMoving(M, disp.u, disp.v);
    draw();
  }
  function ensureReg(): void {
    if (reg || !F || !M || !grid) return;
    const zero = grid.cx.every(v => v === 0) && grid.cy.every(v => v === 0) && grid.spacing === st.spacing;
    reg = new FFDRegistration(F, M, { measure: st.measure, lambda: st.lambda, spacing: st.spacing, coarse: st.coarse && zero && st.spacing * 2 <= 64 ? 2 : 1, blur: 1 },
      zero && st.coarse ? undefined : grid);
  }
  function stepOnce(): boolean {
    if (!reg) return false;
    reg.o.measure = st.measure; reg.o.lambda = st.lambda;
    const more = reg.iterate();
    grid = reg.grid; history.push(reg.current!.cost);
    disp = { u: reg.current!.u, v: reg.current!.v };
    W = warpMoving(M!, disp.u, disp.v);
    return more;
  }
  let timer = 0;
  function stop(): void { if (timer) { cancelAnimationFrame(timer); timer = 0; } runB.textContent = "optimise"; reg = null; }
  function run(): void {
    ensureReg();
    if (!reg) return;
    runB.textContent = "pause";
    let n = 0;
    const tick = () => {
      if (!host.isConnected) { stop(); return; }
      const more = stepOnce(); n++;
      draw();
      if (!more || n >= 300) { stop(); return; }
      timer = requestAnimationFrame(tick);
    };
    timer = requestAnimationFrame(tick);
  }

  // drag control points (pointer on the overlay)
  let dragK = -1;
  view.svg.style.touchAction = "none";
  view.svg.addEventListener("pointerdown", ev => {
    if (!grid) return;
    const [x, y] = svgPoint(view.svg, ev);
    let best = -1, bd = 36;
    for (let l = 0; l < grid.ny; l++) for (let k = 0; k < grid.nx; k++) {
      const i = l * grid.nx + k, [kx, ky] = knotPos(grid, k, l), d = (kx + grid.cx[i] + 0.5 - x) ** 2 + (ky + grid.cy[i] + 0.5 - y) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    if (best < 0) return;
    ev.preventDefault(); stop();
    dragK = best; view.svg.setPointerCapture(ev.pointerId);
  });
  view.svg.addEventListener("pointermove", ev => {
    if (dragK < 0 || !grid) return;
    const [x, y] = svgPoint(view.svg, ev), k = dragK % grid.nx, l = Math.floor(dragK / grid.nx), [kx, ky] = knotPos(grid, k, l);
    grid.cx[dragK] = Math.max(-30, Math.min(30, x - 0.5 - kx)); grid.cy[dragK] = Math.max(-30, Math.min(30, y - 0.5 - ky));
    refresh();
  });
  const endDrag = () => { dragK = -1; };
  view.svg.addEventListener("pointerup", endDrag);
  view.svg.addEventListener("pointercancel", endDrag);

  function draw(): void {
    if (!F || !M || !W || !grid) return;
    tabBtns.forEach((b, i) => b.classList.toggle("active", TABS[i][0] === st.tab));
    measBtns.forEach((b, i) => b.classList.toggle("active", MEASURES[i][0] === st.measure));
    mapBtns.forEach((b, i) => b.classList.toggle("active", MAPS[i] === st.map));
    const J = jacobianDeterminant(disp.u, disp.v, N, N);
    let folds = 0, jmin = Infinity, jmax = -Infinity;
    for (const j of J) { if (j <= 0) folds++; jmin = Math.min(jmin, j); jmax = Math.max(jmax, j); }
    const px = new Uint8ClampedArray(4 * N * N);
    const put = (i: number, r: number, g: number, b: number) => { px[4 * i] = r; px[4 * i + 1] = g; px[4 * i + 2] = b; px[4 * i + 3] = 255; };
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const i = y * N + x, f = 255 * F.data[i], w = 255 * W.data[i];
      switch (st.tab) {
        case "checker": { const c = ((x >> 4) + (y >> 4)) & 1 ? w : f; put(i, c, c, c); break; }
        case "diff": { const d = 128 + 255 * (W.data[i] - F.data[i]); put(i, d, d, d); break; }
        case "moving": { const m = 255 * M.data[i]; put(i, m, m, m); break; }
        case "registered": put(i, w, w, w); break;
        case "jacobian": {
          if (J[i] <= 0) { put(i, 0, 0, 0); break; }
          const c = diverging(Math.max(0, Math.min(1, 0.5 + (J[i] - 1) / 1.2)));
          put(i, 255 * c[0], 255 * c[1], 255 * c[2]); break;
        }
      }
    }
    view.rgba(px, N, N);
    view.setCaption({ checker: "checkerboard: fixed / registered moving", diff: "registered − fixed (grey = 0)", moving: "moving image, the grid it is sampled on", registered: "moving image after the deformation", jacobian: "Jacobian det. (red grows, blue shrinks, black folds)" }[st.tab]);
    drawGrid();
    const sim = similarityCost(F.data, W.data, st.measure, 24), be = bendingEnergy(grid, N, N).value;
    const err = displacementError(disp.u, disp.v, truth.u, truth.v, N, N);
    if (history.length > 1) {
      const xs = history.map((_, i) => i), lo = Math.min(...history), hi = Math.max(...history);
      cplot.setDomain([0, Math.max(10, history.length - 1)], [lo - 0.05 * (hi - lo || 1), hi + 0.05 * (hi - lo || 1)]);
      cline.set(xs, history);
    } else cline.set([], []);
    ro.textContent = `${MEASURES.find(m => m[0] === st.measure)![1]}: ${st.measure === "ssd" ? "mean sq. diff." : st.measure === "ncc" ? "NCC" : "MI (bits)"} = ${sim.score.toFixed(3)}\n` +
      `bending ${be.toExponential(1)}, iterations ${reg ? reg.iterations : history.length}\n` +
      `Jacobian ${jmin.toFixed(2)} … ${jmax.toFixed(2)}, folded ${folds} px\n` +
      `error vs true field: ${err.toFixed(2)} px` + (grid.spacing !== st.spacing ? `\n(grid now ${grid.spacing} px, refining)` : "");
  }

  function drawGrid(): void {
    view.clear();
    if (!grid) return;
    if (st.tab === "moving") {
      // deformed sampling grid: points x + d(x) along lines every 16 px
      const g = svgEl("g", {}, view.over);
      for (let k = 8; k < N; k += 16) {
        let a = "", b = "";
        for (let t = 0; t < N; t += 2) {
          const i1 = t * N + k, i2 = k * N + t;
          a += `${a ? "L" : "M"}${(k + disp.u[i1] + 0.5).toFixed(2)},${(t + disp.v[i1] + 0.5).toFixed(2)}`;
          b += `${b ? "L" : "M"}${(t + disp.u[i2] + 0.5).toFixed(2)},${(k + disp.v[i2] + 0.5).toFixed(2)}`;
        }
        for (const d of [a, b]) view.add("path", { d, fill: "none", stroke: "#22d3ee", "stroke-width": 1.4, opacity: 0.85 }, g);
      }
    }
    // control points (handles at knot + coefficient), inside the image area
    for (let l = 0; l < grid.ny; l++) for (let k = 0; k < grid.nx; k++) {
      const i = l * grid.nx + k, [kx, ky] = knotPos(grid, k, l);
      if (kx < 0 || ky < 0 || kx > N - 1 || ky > N - 1) continue;
      const x = kx + grid.cx[i] + 0.5, y = ky + grid.cy[i] + 0.5, moved = grid.cx[i] || grid.cy[i];
      if (moved) view.add("line", { x1: kx + 0.5, y1: ky + 0.5, x2: x, y2: y, stroke: "#ffd400", "stroke-width": 1.5 });
      view.add("circle", { cx: x, cy: y, r: Math.max(0.9, Math.min(2.4, grid.spacing / 6)), fill: moved ? "#ffd400" : "rgba(255,212,0,.35)", stroke: "#7a5c00", "stroke-width": 1 });
    }
  }

}

autoMount(NAME, mount);
