// Active shape model search on a synthetic image: place the mean shape (move square,
// rotate/scale ring), then step through one iteration sub-step by sub-step (suggest points
// along the normals → fit the pose → project onto the modes → clamp), or play. Beside the
// image: the matching cost along the normal at the selected landmark (click a landmark) and
// the shape parameters b against their ±3√λ limits. Optionally a snake (lib/snake.ts) from
// the same start, for comparison.
// Auto-mounts on <div data-demo="asm">. Options: data-preset="<id>".
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, svgPoint, Slider } from "../lib/ui";
import { drawGrey } from "../lib/imageops";
import { Plot, seriesColor } from "../lib/plot";
import { Shape, centroid, shapeNormals, nPoints } from "../lib/shapes";
import { Contour, edgeMap, gridGradient, implicitSolver, implicitStep, sampleField, resample } from "../lib/snake";
import { AsmModel, Search, SearchOptions, SEARCH_DEFAULTS, Pose, Scene, train, makeScene, poseOf, poseScale, poseAngle, offsetCosts, rmsError, toImage, IMG, SCALE } from "./model";

const NAME = "asm";

interface SceneOpts { family: string; noise: number; clutter: number; occlude: boolean; outside: boolean }
interface Preset { id: string; label: string; scene: Partial<SceneOpts>; start: [number, number, number, number]; o: Partial<SearchOptions>; snake?: boolean }

// start = [dx, dy, rotation (degrees), scale factor] relative to the true pose
const PRESETS: Preset[] = [
  { id: "clean", label: "clean image, good start", scene: {}, start: [10, -4, 10, 0.95], o: {} },
  { id: "noise", label: "noise: constraint saves it", scene: { noise: 0.25 }, start: [10, -4, 10, 0.95], o: {} },
  { id: "occlusion", label: "occlusion: model fills in", scene: { occlude: true }, start: [10, -4, 10, 0.95], o: {} },
  { id: "far", label: "far start: single level fails, pyramid works", scene: {}, start: [-22, 12, -20, 0.9], o: { levels: 1 } },
  { id: "free", label: "no shape constraint: points scatter", scene: { noise: 0.25 }, start: [10, -4, 10, 0.95], o: { constrain: false } },
  { id: "outside", label: "shape outside the training set", scene: { outside: true }, start: [4, -2, 5, 0.95], o: {}, snake: true },
  { id: "hand", label: "hands: coarse to fine", scene: { family: "hand" }, start: [8, -4, 8, 0.95], o: { levels: 3 } },
];
const SCENE0: SceneOpts = { family: "nucleus", noise: 0.03, clutter: 0, occlude: false, outside: false };
const OUTSIDE: Record<string, number[]> = { nucleus: [0.5, 0.55, 0.35], hand: [2.2, 0.6, 1.0] };

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] .as-fig { flex: 0 1 420px; min-width: 260px; margin: 0; }
[data-demo="${NAME}"] .as-fig figcaption { font-size: 16px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] .as-stage { position: relative; width: 100%; aspect-ratio: 1; }
[data-demo="${NAME}"] .as-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .as-stage svg { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; }
[data-demo="${NAME}"] .as-mid { flex: 0 1 300px; min-width: 220px; display: flex; flex-direction: column; gap: .2em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 370px; gap: .3em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .as-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
[data-demo="${NAME}"] .as-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .qb-btn:disabled { opacity: .4; cursor: default; }
[data-demo="${NAME}"] .qb-btn.next { border-color: var(--qb-accent); border-width: 2px; }
.reveal [data-demo="${NAME}"] { font-size: 16px; gap: .5em; }
.reveal [data-demo="${NAME}"] .as-fig { flex-basis: 410px; }
.reveal [data-demo="${NAME}"] .as-mid { flex-basis: 280px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .12em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 13px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
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

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = { sc: { ...SCENE0 } as SceneOpts, o: { ...SEARCH_DEFAULTS } as SearchOptions, k: 3, snake: false, truth: false, sel: 0 };
  let M: AsmModel | null = null, Mkey = "";
  let scene: Scene;
  let start: Pose;
  let S: Search | null = null;
  let snakeC: Contour | null = null;
  let playing = false;

  // ---- layout ----
  const fig = htmlEl("figure", "as-fig", "", host);
  const cap = htmlEl("figcaption", "", "", fig);
  const stage = htmlEl("div", "as-stage", "", fig);
  const cv = htmlEl("canvas", "", "", stage);
  const svg = svgEl("svg", { viewBox: `-0.5 -0.5 ${IMG} ${IMG}` }, stage);
  const gTruth = svgEl("g", {}, svg), gNorm = svgEl("g", {}, svg), gSnake = svgEl("g", {}, svg), gShape = svgEl("g", {}, svg), gHand = svgEl("g", {}, svg);

  const mid = htmlEl("div", "as-mid", "", host);
  const pCost = new Plot(mid, { width: 330, height: 215, x: [-6, 6], y: [0, 1], xlabel: "offset along the normal (px)", ylabel: "cost", title: "landmark 1", font: 17, margin: { t: 30 } });
  const mCost = pCost.line([], [], { color: seriesColor(0) });
  const mCostDots = pCost.scatter([], [], { color: seriesColor(0), r: 4 });
  const mBest = pCost.vline(0, { color: seriesColor(1) });
  const pB = new Plot(mid, { width: 330, height: 215, x: [0.4, 5.6], y: [-4, 4], xlabel: "mode k", ylabel: "b_k / √λ_k", title: "shape parameters", font: 17, xticks: [1, 2, 3, 4, 5], yticks: [-3, 0, 3], margin: { t: 30 } });
  const bStyle: { color: string; colors?: string[] } = { color: seriesColor(0) };
  const mB = pB.bars([], [], bStyle);
  const mBraw = pB.scatter([], [], { color: seriesColor(1), r: 5 });
  pB.hline(3, { color: seriesColor(1), width: 1.5 }); pB.hline(-3, { color: seriesColor(1), width: 1.5 });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "as-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "as-lab", "one iteration, step by step", panel);
  const subRow = htmlEl("div", "as-row", "", panel);
  const subBtns = [
    button(subRow, "1 suggest", () => sub()),
    button(subRow, "2 pose", () => sub()),
    button(subRow, "3 project", () => sub()),
    button(subRow, "4 clamp", () => sub()),
  ];
  const ctl = htmlEl("div", "as-row", "", panel);
  button(ctl, "iteration", () => { setPlaying(false); if (S) { S.iterate(); afterStep(); } });
  const playBtn = button(ctl, "▶ play", () => setPlaying(!playing));
  button(ctl, "reset", () => restart());
  const sl: Record<string, Slider> = {};
  sl.noise = slider(panel, "noise", 0, 0.4, 0.01, st.sc.noise, v => { st.sc.noise = v; clearActive(); newScene(); });
  sl.modes = slider(panel, "modes", 1, 5, 1, st.o.modes, v => { st.o.modes = v; clearActive(); restart(); });
  sl.k = slider(panel, "profile k", 1, 6, 1, st.k, v => { st.k = v; clearActive(); retrain(); });
  sl.m = slider(panel, "search m", 2, 12, 1, st.o.m, v => { st.o.m = v; clearActive(); restart(); });
  sl.levels = slider(panel, "pyramid levels", 1, 3, 1, st.o.levels, v => { st.o.levels = v; clearActive(); restart(); });
  const mrow = htmlEl("div", "as-row", "", panel);
  const matchBtns = (["edge", "mahal"] as const).map(m => button(mrow, m === "edge" ? "strongest edge" : "profile model", () => { st.o.match = m; clearActive(); restart(); }));
  const famBtns = ["nucleus", "hand"].map(f => button(mrow, f, () => { st.sc.family = f; clearActive(); retrain(); }));
  const crow = htmlEl("div", "as-row", "", panel);
  const conBox = checkbox(crow, "shape constraint", st.o.constrain, v => { st.o.constrain = v; clearActive(); restart(); });
  const snBox = checkbox(crow, "snake too", st.snake, v => { st.snake = v; runSnake(); draw(); });
  const trBox = checkbox(crow, "truth", st.truth, v => { st.truth = v; draw(); });
  const crow2 = htmlEl("div", "as-row", "", panel);
  const occBox = checkbox(crow2, "occluder", st.sc.occlude, v => { st.sc.occlude = v; clearActive(); newScene(); });
  const clBox = checkbox(crow2, "clutter", st.sc.clutter > 0, v => { st.sc.clutter = v ? 6 : 0; clearActive(); newScene(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Before the first step, drag the square (move) or the ring (rotate and scale) to place the mean shape. Click a landmark to see its cost curve.", panel);

  function clearActive(): void { presetBtns.forEach(b => b.classList.remove("active")); }

  function load(p: Preset): void {
    setPlaying(false);
    st.sc = { ...SCENE0, ...p.scene }; st.o = { ...SEARCH_DEFAULTS, ...p.o }; st.snake = !!p.snake; st.sel = 0;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    retrain(false);
    const P = scene.pose, [dx, dy, rot, sc] = p.start;
    start = poseOf(poseScale(P) * sc, poseAngle(P) + rot * Math.PI / 180, P.tx + dx, P.ty + dy);
    restart();
  }

  function retrain(re = true): void {
    const key = `${st.sc.family}/${st.k}`;
    if (!M || key !== Mkey) { M = train({ family: st.sc.family, k: st.k, levels: 3 }); Mkey = key; }
    newScene(re);
  }

  function newScene(re = true): void {
    const fam = M!.fam;
    scene = makeScene(fam, { noise: st.sc.noise, clutter: st.sc.clutter, occluder: st.sc.occlude ? (fam.id === "hand" ? [10, 46, 118, 40, 9] : [20, 52, 110, 62, 9]) : null, params: st.sc.outside ? OUTSIDE[fam.id] : undefined });
    drawGrey(cv, scene.img, { mode: "clip", lo: 0, hi: 1 });
    if (!start) start = poseOf(SCALE * 0.95, 0.25, IMG / 2 + 10, IMG / 2 - 4);
    if (re) restart();
  }

  function restart(): void {
    setPlaying(false);
    if (!M) return;
    S = new Search(M, scene.img, start, { ...st.o });
    runSnake();
    draw();
  }

  function runSnake(): void {
    snakeC = null;
    if (!st.snake || !M) return;
    const X0 = toImage(M.pdm.mean, start), n = 64;
    let c = resample({ x: Float64Array.from({ length: nPoints(X0) }, (_, i) => X0[2 * i]), y: Float64Array.from({ length: nPoints(X0) }, (_, i) => X0[2 * i + 1]) }, n);
    const f = edgeMap(scene.img, 1.5), F = gridGradient(f), sol = implicitSolver(n, 0.05, 0.02, 1);
    for (let it = 0; it < 300; it++) c = implicitStep(c, sol, sampleField(F, c, 2));
    snakeC = c;
  }

  function sub(): void { setPlaying(false); if (S) { S.next(); afterStep(); } }
  function afterStep(): void { draw(); }

  function setPlaying(v: boolean): void { playing = v; playBtn.textContent = v ? "❚❚ pause" : "▶ play"; if (v) requestAnimationFrame(tick); }
  let last = 0;
  function tick(t: number): void {
    if (!playing || !S) return;
    if (!onScreen(host)) { setPlaying(false); return; }
    if (t - last > 140) { last = t; S.iterate(); draw(); if (S.done()) { setPlaying(false); return; } }
    requestAnimationFrame(tick);
  }

  // ---- drawing ----
  const pathOf = (s: Shape) => { let d = ""; for (let i = 0; i < s.length; i += 2) d += (i ? "L" : "M") + s[i].toFixed(2) + "," + s[i + 1].toFixed(2); return d + "Z"; };

  function draw(): void {
    if (!S || !M) return;
    gTruth.replaceChildren(); gNorm.replaceChildren(); gSnake.replaceChildren(); gShape.replaceChildren(); gHand.replaceChildren();
    if (st.truth) svgEl("path", { d: pathOf(scene.truth), fill: "none", stroke: "#ffffff", "stroke-width": 0.6, "stroke-dasharray": "1.5 1.2", opacity: 0.9 }, gTruth);
    const X = S.X, n = nPoints(X), N = shapeNormals(X), f = 2 ** S.level, L = st.o.m * f;
    for (let i = 0; i < n; i++) {
      const sel = i === st.sel;
      svgEl("line", { x1: X[2 * i] - L * N[2 * i], y1: X[2 * i + 1] - L * N[2 * i + 1], x2: X[2 * i] + L * N[2 * i], y2: X[2 * i + 1] + L * N[2 * i + 1], stroke: sel ? "#ffd400" : "#7fd3ff", "stroke-width": sel ? 0.8 : 0.35, opacity: sel ? 1 : 0.7 }, gNorm);
    }
    if (snakeC) {
      let d = ""; for (let i = 0; i < snakeC.x.length; i++) d += (i ? "L" : "M") + snakeC.x[i].toFixed(2) + "," + snakeC.y[i].toFixed(2);
      svgEl("path", { d: d + "Z", fill: "none", stroke: "#5ee36a", "stroke-width": 0.9 }, gSnake);
    }
    svgEl("path", { d: pathOf(X), fill: "none", stroke: "#3fa9ff", "stroke-width": 1.1, "stroke-linejoin": "round" }, gShape);
    for (let i = 0; i < n; i++) {
      const c = svgEl("circle", { cx: X[2 * i], cy: X[2 * i + 1], r: i === st.sel ? 1.8 : 1.2, fill: i === st.sel ? "#ffd400" : "#3fa9ff", stroke: "#000", "stroke-width": 0.2 }, gShape);
      c.style.cursor = "pointer";
      c.addEventListener("pointerdown", ev => { ev.stopPropagation(); st.sel = i; draw(); });
    }
    if (S.Y && S.stage !== "clamp" && S.stage !== "start") {
      const Y = S.Y;
      for (let i = 0; i < n; i++) svgEl("circle", { cx: Y[2 * i], cy: Y[2 * i + 1], r: 1.2, fill: "#ff3b3b" }, gShape);
    }
    if (S.total === 0 && S.stage === "start") drawHandles();

    // cost curve at the selected landmark
    const costs = offsetCosts(M, S.pyr[S.level], S.level, X, N, st.sel, S.o);
    const offs = Array.from(costs, (_, t) => (t - S!.o.m) * f);
    let lo = Infinity, hi = -Infinity, bi = 0;
    costs.forEach((v, t) => { lo = Math.min(lo, v); hi = Math.max(hi, v); if (v < costs[bi]) bi = t; });
    const pad = 0.1 * (hi - lo || 1);
    pCost.setDomain([-S.o.m * f - 0.5, S.o.m * f + 0.5], [lo - pad, hi + pad]);
    pCost.setLabels({ title: `landmark ${st.sel + 1}: ${S.o.match === "edge" ? "−|edge|" : "Mahalanobis"}` });
    mCost.set(offs, costs); mCostDots.set(offs, costs); mBest.set(offs[bi]);

    // b bars
    const K = st.o.modes, sd = Array.from({ length: K }, (_, k) => Math.sqrt(M!.pdm.values[k]));
    const bs = Array.from({ length: K }, (_, k) => S!.b[k] / sd[k]), braw = Array.from({ length: K }, (_, k) => S!.bRaw[k] / sd[k]);
    const ext = Math.max(4, ...braw.map(Math.abs)) * 1.1;
    pB.setDomain([0.4, K + 0.6], [-ext, ext]);
    bStyle.colors = bs.map(v => (Math.abs(v) >= 3 - 1e-9 ? "var(--qb-c2, #d62728)" : "var(--qb-c1, #1f77b4)"));
    mB.set(bs.map((_, k) => k + 1), bs);
    mBraw.set(braw.map((_, k) => k + 1), S.stage === "project" || S.stage === "clamp" ? braw : braw.map(() => NaN));

    // buttons and readout
    const order = ["start", "suggested", "pose", "project", "clamp"], nextIdx = (order.indexOf(S.stage) % 4);
    subBtns.forEach((b, i) => { b.classList.toggle("next", i === (S!.stage === "clamp" || S!.stage === "start" ? 0 : nextIdx)); b.disabled = !st.o.constrain && i >= 2; });
    sl.noise.set(st.sc.noise); sl.modes.set(st.o.modes); sl.k.set(st.k); sl.m.set(st.o.m); sl.levels.set(st.o.levels);
    matchBtns.forEach((b, i) => b.classList.toggle("active", ["edge", "mahal"][i] === st.o.match));
    famBtns.forEach((b, i) => b.classList.toggle("active", ["nucleus", "hand"][i] === st.sc.family));
    conBox.checked = st.o.constrain; snBox.checked = st.snake; trBox.checked = st.truth; occBox.checked = st.sc.occlude; clBox.checked = st.sc.clutter > 0;
    const stageName: Record<string, string> = { start: "start: mean shape", suggested: "suggested points (red)", pose: "pose fitted", project: "projected onto the modes", clamp: "clamped" };
    cap.textContent = `level ${S.level} (${IMG >> S.level} px), iteration ${S.total}: ${stageName[S.stage]}`;
    ro.textContent = [
      `RMS error to the truth: ${rmsError(X, scene.truth).toFixed(2)} px`,
      `pose: scale ${poseScale(S.pose).toFixed(1)}, angle ${(poseAngle(S.pose) * 180 / Math.PI).toFixed(1)}°`,
      `last move: ${Number.isFinite(S.moved) ? S.moved.toFixed(2) + " px" : "–"}`,
      snakeC ? `snake (green): ${outlineDist(snakeC).toFixed(2)} px from the outline` : "",
    ].filter(Boolean).join("\n");
  }

  /** RMS distance from snake points to the true outline (polygon). */
  function outlineDist(c: Contour): number {
    const T = scene.truth, n = nPoints(T);
    let s = 0;
    for (let i = 0; i < c.x.length; i++) {
      let best = Infinity;
      for (let j = 0; j < n; j++) {
        const k = (j + 1) % n, ax = T[2 * j], ay = T[2 * j + 1], bx = T[2 * k], by = T[2 * k + 1];
        const vx = bx - ax, vy = by - ay, t = Math.max(0, Math.min(1, ((c.x[i] - ax) * vx + (c.y[i] - ay) * vy) / (vx * vx + vy * vy || 1)));
        best = Math.min(best, Math.hypot(c.x[i] - ax - t * vx, c.y[i] - ay - t * vy));
      }
      s += best * best;
    }
    return Math.sqrt(s / c.x.length);
  }

  // ---- placing the start pose ----
  function drawHandles(): void {
    if (!S || !M) return;
    const X = S.X, [cx, cy] = centroid(X), r = 0.8 * poseScale(start);
    const hx = cx + r * Math.cos(poseAngle(start)), hy = cy + r * Math.sin(poseAngle(start));
    svgEl("line", { x1: cx, y1: cy, x2: hx, y2: hy, stroke: "#ffd400", "stroke-width": 0.5 }, gHand);
    const mv = svgEl("rect", { x: cx - 2.5, y: cy - 2.5, width: 5, height: 5, fill: "rgba(0,0,0,.3)", stroke: "#ffd400", "stroke-width": 0.8 }, gHand);
    const rh = svgEl("circle", { cx: hx, cy: hy, r: 2.8, fill: "rgba(0,0,0,.3)", stroke: "#ffd400", "stroke-width": 0.8 }, gHand);
    const drag = (el: SVGElement, onMove: (x: number, y: number) => void) => {
      el.style.cursor = "grab";
      el.addEventListener("pointerdown", ev => {
        ev.preventDefault(); ev.stopPropagation();
        const mvf = (e: PointerEvent) => { const [x, y] = svgPoint(svg, e); onMove(x, y); clearActive(); restart(); };
        const up = () => { window.removeEventListener("pointermove", mvf); window.removeEventListener("pointerup", up); window.removeEventListener("pointercancel", up); };
        window.addEventListener("pointermove", mvf); window.addEventListener("pointerup", up); window.addEventListener("pointercancel", up);
      });
    };
    // the model frame's centroid is 0, so the shape centroid is the translation
    drag(mv, (x, y) => { start = { ...start, tx: x, ty: y }; });
    drag(rh, (x, y) => {
      const s = Math.max(5, Math.hypot(x - start.tx, y - start.ty) / 0.8), th = Math.atan2(y - start.ty, x - start.tx);
      start = poseOf(s, th, start.tx, start.ty);
    });
  }

  load(p0);
}

autoMount(NAME, mount);
