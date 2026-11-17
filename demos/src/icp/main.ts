// Iterative closest point, step by step. Blue: the fixed point set; orange: the moving set,
// drawn where the current transform puts it. "1. match" joins every orange point to its
// nearest blue point (grey; red dashed = left out by trimming); "2. move" solves for the best
// transform of the chosen kind on those pairs and moves the set. Drag the square handle to
// choose the start. The left plot shows the RMS distance of the matched pairs and the true
// error per iteration; the right plot the final true error for every rotation between the sets
// (where the start converges).
//
// <div data-demo="icp" data-preset="close|far|centroids|overlap|trimmed|translation|outline"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl, draggable, svgPoint } from "../lib/ui";
import { Plot } from "../lib/plot";
import * as I from "./icp";

const NAME = "icp";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .6em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .icp-svg { flex: 0 1 420px; min-width: 260px; }
.qb-demo[data-demo="${NAME}"] .icp-svg svg { width: 100%; display: block; background: var(--qb-panel); border-radius: 6px; touch-action: none; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 240px; max-width: 290px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .icp-plots { flex: 0 1 250px; min-width: 200px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .icp-cap { font-size: 15px; text-align: center; margin-bottom: -.3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-line; }
.qb-demo[data-demo="${NAME}"] .icp-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
.qb-demo[data-demo="${NAME}"] .icp-lab { font-size: 14px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .qb-btn:disabled { opacity: .45; cursor: default; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; flex-wrap: nowrap; gap: .6em; }
.reveal .qb-demo[data-demo="${NAME}"] .icp-svg { flex: 0 0 500px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 340px; max-width: 340px; gap: .25em; }
.reveal .qb-demo[data-demo="${NAME}"] .icp-plots { flex: 0 0 280px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .05em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] .icp-lab { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const BLUE = "#1f77b4", ORANGE = "#ff7f0e", RED = "#d62728", GREEN = "#2ca02c", GREY = "#8a9099";
const SWEEP = Array.from({ length: 73 }, (_, i) => -180 + 5 * i);
const OK = 5;   // px: "converged" in the sweep

const BASE: I.DemoSettings = { scene: "spots", kind: "rigid", angle: 10, noise: 1, unpaired: 0, keep: 1, centroids: false, startShift: [0, 0] };
interface Preset { id: string; label: string; s: Partial<I.DemoSettings> }
const PRESETS: Preset[] = [
  { id: "close", label: "close start", s: {} },
  { id: "far", label: "turned 40°", s: { angle: 40 } },
  { id: "centroids", label: "far shift: centroids", s: { startShift: [80, 0], centroids: true } },
  { id: "overlap", label: "partial overlap", s: { unpaired: 0.3 } },
  { id: "trimmed", label: "trimmed: keep 70 %", s: { unpaired: 0.3, keep: 0.7 } },
  { id: "translation", label: "translation only", s: { kind: "translation" } },
  { id: "outline", label: "outline", s: { scene: "outline", angle: 30 } },
];
const KINDS: I.Kind[] = ["translation", "rigid", "similarity"];

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const st: I.DemoSettings = { ...BASE, ...(p0?.s ?? {}) };
  st.startShift = [...st.startShift];
  let sc = I.demoScene(st), run = I.demoRun(st, sc), half = 0, showTruth = false;

  // ---- figure ----
  const wrap = htmlEl("div", "icp-svg", "", host);
  const svg = svgEl("svg", { viewBox: "-10 -10 470 420", role: "img", "aria-label": "fixed points (blue) and moving points (orange) with their nearest-neighbour matches" }, wrap);
  const gOutline = svgEl("g", {}, svg), gTruth = svgEl("g", {}, svg), gLines = svgEl("g", {}, svg), gFixed = svgEl("g", {}, svg), gMoving = svgEl("g", {}, svg), gHandle = svgEl("g", {}, svg);
  const label = svgEl("text", { x: 0, y: 12, "font-size": 20 }, svg);
  const handle = svgEl("rect", { width: 22, height: 22, rx: 3, class: "handle" }, gHandle);
  handle.style.stroke = ORANGE;
  let dragFrom: { x: number; y: number; sx: number; sy: number } | null = null;
  handle.addEventListener("pointerdown", ev => {
    const [x, y] = svgPoint(svg, ev);
    const T0 = I.startTransform(st, sc);
    dragFrom = { x, y, sx: T0.tx, sy: T0.ty };
  });
  draggable(handle, ev => svgPoint(svg, ev), (x, y) => {
    if (!dragFrom) return;
    st.centroids = false; centroidBox.checked = false;
    st.startShift = [dragFrom.sx + x - dragFrom.x, dragFrom.sy + y - dragFrom.y];
    recompute(false);
  });
  handle.addEventListener("pointerup", () => { if (dragFrom) { dragFrom = null; presetBtns.forEach(b => b.classList.remove("active")); updateSweep(); } });

  // ---- panel ----
  const panel = htmlEl("div", "qb-panel", "", host);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => {
    Object.assign(st, BASE, p.s); st.startShift = [...(p.s.startShift ?? [0, 0])]; sync(); recompute(true);
  } })));
  const sceneRow = htmlEl("div", "icp-row", "", panel);
  htmlEl("span", "icp-lab", "points", sceneRow);
  const sceneBtns = (["spots", "outline"] as I.SceneKind[]).map(k => button(sceneRow, k, () => { st.scene = k; changed(true); }));
  const kindRow = htmlEl("div", "icp-row", "", panel);
  htmlEl("span", "icp-lab", "model", kindRow);
  const kindBtns = KINDS.map(k => button(kindRow, k, () => { st.kind = k; changed(true); }));
  const angS = slider(panel, "rotation between sets", -180, 180, 1, st.angle, v => { st.angle = v; changed(true); }, v => v + "°");
  const noiS = slider(panel, "noise (px)", 0, 5, 0.5, st.noise, v => { st.noise = v; changed(true); });
  const unpS = slider(panel, "unpaired", 0, 40, 5, st.unpaired * 100, v => { st.unpaired = v / 100; changed(true); }, v => v + " %");
  const keepS = slider(panel, "keep (trim)", 40, 100, 5, st.keep * 100, v => { st.keep = v / 100; changed(true); }, v => v + " %");
  const centroidBox = checkbox(panel, "start: match the centroids", st.centroids, v => { st.centroids = v; changed(true); });
  checkbox(panel, "show the true position", showTruth, v => { showTruth = v; draw(); });
  const ctl = htmlEl("div", "icp-row", "", panel);
  const stepBtn = button(ctl, "1. match", () => { if (half < 2 * run.res.steps.length - 1) { half++; draw(); } });
  button(ctl, "run to the end", () => { half = 2 * run.res.steps.length - 1; draw(); });
  button(ctl, "back to the start", () => { half = 0; draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  function sync(): void {
    angS.set(st.angle); noiS.set(st.noise); unpS.set(Math.round(st.unpaired * 100)); keepS.set(Math.round(st.keep * 100));
    centroidBox.checked = st.centroids;
  }
  function changed(fromControl: boolean): void { if (fromControl) presetBtns.forEach(b => b.classList.remove("active")); recompute(true); }

  // ---- plots ----
  const plots = htmlEl("div", "icp-plots", "", host);
  htmlEl("div", "icp-cap", "per iteration", plots);
  const conv = new Plot(plots, { width: 300, height: 200, x: [0, 10], y: [0.1, 300], ylog: true, xlabel: "iteration", ylabel: "px", font: 16, yticks: [0.1, 1, 10, 100] });
  const rmsLine = conv.line([], [], { color: BLUE, width: 2.5, label: "RMS of pairs" });
  const errLine = conv.line([], [], { color: ORANGE, width: 2.5, dash: "6 4", label: "true error" });
  const curDot = conv.scatter([], [], { color: BLUE, r: 5 });
  conv.legend(undefined, "bl");
  htmlEl("div", "icp-cap", "final true error vs rotation", plots);
  const sw = new Plot(plots, { width: 300, height: 200, x: [-180, 180], y: [0.1, 300], ylog: true, xlabel: "rotation between sets (°)", ylabel: "px", font: 16, xticks: [-180, -90, 0, 90, 180], yticks: [0.1, 1, 10, 100] });
  sw.line([-180, 180], [OK, OK], { color: GREY, width: 1.5, dash: "4 4" });
  const swDots = sw.scatter([], [], { r: 4 });
  const swCur = sw.line([], [], { color: "var(--qb-fg)", width: 2 });

  function recompute(resetSweep: boolean): void {
    sc = I.demoScene(st); run = I.demoRun(st, sc); half = 0;
    if (resetSweep) updateSweep();
    draw();
  }
  function updateSweep(): void {
    const e = I.sweep(st, SWEEP);
    swDots.set(SWEEP, e.map(v => Math.max(0.1, Math.min(300, v))), e.map(v => (v < OK ? GREEN : RED)));
  }

  const circle = (g: SVGGElement, p: I.Pt, r: number, fill: string, stroke: string, w = 2.5, opacity = 1) => {
    const c = svgEl("circle", { cx: p.x.toFixed(1), cy: p.y.toFixed(1), r, "stroke-width": w }, g);
    c.style.fill = fill; c.style.stroke = stroke; c.style.opacity = String(opacity);
  };

  function draw(): void {
    sceneBtns.forEach((b, i) => b.classList.toggle("active", ["spots", "outline"][i] === st.scene));
    kindBtns.forEach((b, i) => b.classList.toggle("active", KINDS[i] === st.kind));
    const steps = run.res.steps, K = steps.length;
    const j = Math.min(K - 1, Math.floor(half / 2)), matched = half % 2 === 1, step = steps[j];
    const T = step.T, moved = I.applyAll(T, sc.M);

    gOutline.replaceChildren(); gTruth.replaceChildren(); gLines.replaceChildren(); gFixed.replaceChildren(); gMoving.replaceChildren();
    if (st.scene === "outline") {
      const pl = svgEl("polygon", { points: sc.F.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" "), fill: "none", "stroke-width": 2 }, gOutline);
      pl.style.stroke = BLUE; pl.style.opacity = "0.35";
    }
    if (showTruth) for (const p of I.applyAll(sc.truth, sc.M)) circle(gTruth, p, 7, "none", ORANGE, 1.5, 0.45);
    if (matched) {
      moved.forEach((p, i) => {
        const q = sc.F[step.match[i]];
        const l = svgEl("line", { x1: p.x.toFixed(1), y1: p.y.toFixed(1), x2: q.x.toFixed(1), y2: q.y.toFixed(1), "stroke-width": step.kept[i] ? 2.5 : 2 }, gLines);
        l.style.stroke = step.kept[i] ? "#555" : RED;
        if (!step.kept[i]) l.setAttribute("stroke-dasharray", "5 4");
      });
    }
    const rf = st.scene === "outline" ? 4 : 6;
    for (const p of sc.F) circle(gFixed, p, rf, BLUE, BLUE, 1);
    for (const p of moved) circle(gMoving, p, 6, "var(--qb-bg)", ORANGE, 3);
    const c = I.centroid(moved);
    handle.setAttribute("x", (c.x - 11).toFixed(1)); handle.setAttribute("y", (c.y - 11).toFixed(1));
    label.textContent = matched ? `iteration ${j + 1}: nearest neighbours` : j === 0 ? "start" : `after iteration ${j}`;

    // the button: match, then move
    const atEnd = half >= 2 * K - 1;
    stepBtn.textContent = matched ? "2. move" : "1. match";
    stepBtn.disabled = atEnd;

    // readout
    const nKeep = step.kept.filter(Boolean).length;
    const deg = (I.angleOf(T) * 180 / Math.PI).toFixed(1);
    const lines = [
      `RMS of the ${nKeep} kept pairs: ${step.rms.toFixed(2)} px` + (matched ? ` → ${step.rmsAfter.toFixed(2)} after the move` : ""),
      `angle found ${deg}°, true ${st.angle}°` + (st.kind === "similarity" ? `, scale ${I.scaleOf(T).toFixed(3)}` : ""),
      `true error ${run.err[j].toFixed(2)} px`,
      atEnd ? (run.res.converged ? `converged after ${K - 1} moves: the matching no longer changes` : `stopped after ${K} iterations`) : "",
    ];
    ro.textContent = lines.filter(Boolean).join("\n");

    // plots
    const xs = Array.from({ length: j + 1 }, (_, k) => k);
    const clip = (v: number) => Math.max(0.1, Math.min(300, v));
    conv.setDomain([0, Math.max(5, K - 1)]);
    rmsLine.set(xs, xs.map(k => clip(steps[k].rms)));
    errLine.set(xs, xs.map(k => clip(run.err[k])));
    curDot.set([j], [clip(step.rms)]);
    swCur.set([st.angle, st.angle], [0.1, 300]);
  }

  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  recompute(true);
}

autoMount(NAME, mount);
