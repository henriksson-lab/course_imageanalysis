// Boundary metrics: a fixed true outline (grey) and a predicted outline (blue) through 12
// draggable control points, with an optional thin spike and a stray fragment. Whiskers join
// boundary points to the nearest point of the other outline, coloured by length; the longest
// is the Hausdorff distance. Readouts: IoU, Dice, Hausdorff, HD95, average surface distance
// (ASD), and a histogram of all boundary distances. The size slider scales the object while
// the pixel errors stay the same.
// Auto-mounts on <div data-demo="contourdist">. Options: data-preset="free|dilate|spike|fragment|small".
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, svgPoint, draggable, setAttrs as setA } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { viridis, toCss } from "../lib/colour";
import * as M from "./model";

const NAME = "contourdist";
const STYLE = `
[data-demo="${NAME}"] { gap: 1em; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] .cd-view { flex: 0 1 430px; min-width: 260px; }
[data-demo="${NAME}"] .cd-view svg { width: 100%; aspect-ratio: 1; background: var(--qb-panel); border-radius: 4px; touch-action: none; }
[data-demo="${NAME}"] .cd-right { flex: 1 1 300px; max-width: 420px; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .cd-num { display: grid; grid-template-columns: repeat(5, auto); gap: .1em .9em; font-size: 15px; justify-content: start; }
.reveal [data-demo="${NAME}"] .cd-num { font-size: 16px; gap: .1em .5em; }
.reveal [data-demo="${NAME}"] .cd-num b { font-size: 18px; }
[data-demo="${NAME}"] .cd-num b { font-size: 22px; font-family: ui-monospace, monospace; font-weight: 600; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 260px; gap: .4em; }
[data-demo="${NAME}"] .cd-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .cd-view { flex-basis: 420px; }
.reveal [data-demo="${NAME}"] .cd-right { flex-basis: 300px; max-width: 340px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex-basis: 250px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

// lib/plot.ts numbers its clip paths per bundle ("qb-clip-1", …), so two demos on one page
// share ids and clip each other's plots. Give every plot of this demo a unique clip id.
let clipCount = 0;
function uniqueClip<T extends { svg: SVGSVGElement; layer: SVGGElement }>(p: T): T {
  const cp = p.svg.querySelector("clipPath");
  if (cp) { cp.id = `${NAME}-clip-${++clipCount}`; p.layer.setAttribute("clip-path", `url(#${cp.id})`); }
  return p;
}

const pathOf = (c: M.Pt[]) => "M" + c.map(p => p[0].toFixed(3) + "," + p[1].toFixed(3)).join("L") + "Z";

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = M.PRESETS.find(p => p.id === host.dataset.preset) ?? M.PRESETS[0];
  let st: M.Settings = p0.make();

  const view = htmlEl("figure", "cd-view", "", host);
  htmlEl("figcaption", "", "truth (grey) and prediction (blue); drag the dots", view);
  const svg = svgEl("svg", {}, view);
  const gDyn = svgEl("g", {}, svg), gHandles = svgEl("g", {}, svg);
  const ctrlH = Array.from({ length: M.NCTRL }, (_, k) => {
    const h = svgEl("circle", { fill: "#fff", stroke: "#1f77b4" }, gHandles);
    draggable(h, ev => svgPoint(svg, ev), (x, y) => { st.pred.offsets[k] = M.offsetFor(st.R, k, [x, y]); clearActive(); draw(); });
    return h;
  });
  const spikeH = svgEl("rect", { fill: "#fff", stroke: "#d62728" }, gHandles);
  draggable(spikeH, ev => svgPoint(svg, ev), (x, y) => {
    const phi = Math.atan2(y, x), ff = M.frame(st.R, phi);
    st.pred.spike.phi = phi; st.pred.spike.len = Math.max(1, (x - ff.p[0]) * ff.n[0] + (y - ff.p[1]) * ff.n[1]);
    clearActive(); draw();
  });
  const fragH = svgEl("circle", { fill: "none", stroke: "#d62728" }, gHandles);
  fragH.style.pointerEvents = "all";
  draggable(fragH, ev => svgPoint(svg, ev), (x, y) => { st.pred.fragment.u = x / st.R; st.pred.fragment.v = y / st.R; clearActive(); draw(); });
  const right = htmlEl("div", "cd-right", "", host);
  const nums = htmlEl("div", "cd-num", "", right);
  const vals: Record<string, HTMLElement> = {};
  for (const k of ["IoU", "Dice", "Hausdorff", "HD95", "ASD"]) { htmlEl("span", "", k, nums); }
  for (const k of ["IoU", "Dice", "Hausdorff", "HD95", "ASD"]) vals[k] = htmlEl("b", "", "", nums);
  htmlEl("div", "qb-note", "Hausdorff, HD95 and ASD in pixels.", right);
  const histHost = htmlEl("div", "", "", right);
  const slide = !!host.closest(".reveal");
  const hist = new Plot(histHost, { x: [0, 5], y: [0, 1], xlabel: "boundary distance (px)", ylabel: "share of points", width: slide ? 420 : 560, height: slide ? 360 : 300, font: slide ? 26 : 18 });
  uniqueClip(hist);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "cd-lab", "presets", panel);
  const pbtn = presets(panel, M.PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const sizeS = slider(panel, "object radius (px)", 5, 50, 1, st.R, v => { st.R = v; draw(); });
  const spikeC = checkbox(panel, "thin spike", st.pred.spike.on, v => { st.pred.spike.on = v; draw(); });
  const fragC = checkbox(panel, "stray fragment", st.pred.fragment.on, v => { st.pred.fragment.on = v; draw(); });
  button(panel, "reset", () => load(p0));
  htmlEl("div", "qb-note", "Whiskers join boundary points to the nearest point of the other outline (colour = length). The red whisker is the Hausdorff distance.", panel);

  function clearActive(): void { pbtn.forEach(b => b.classList.remove("active")); }

  function load(p: M.Preset): void {
    st = p.make();
    sizeS.set(st.R); spikeC.checked = st.pred.spike.on; fragC.checked = st.pred.fragment.on;
    pbtn.forEach((b, i) => b.classList.toggle("active", M.PRESETS[i] === p));
    draw();
  }


  function draw(): void {
    const res = M.evaluate(st), R = st.R, half = 2.05 * R + 3;
    svg.setAttribute("viewBox", `${-half} ${-half} ${2 * half} ${2 * half}`);
    gDyn.replaceChildren();
    const u = (2 * half) / 430;   // one screen pixel in world units (approx.)
    // pixel grid
    const gstep = half > 40 ? 5 : 1, g = svgEl("g", {}, gDyn);
    for (let v = Math.ceil(-half / gstep) * gstep; v <= half; v += gstep) {
      svgEl("line", { x1: v, x2: v, y1: -half, y2: half, stroke: "var(--qb-grid)", "stroke-width": 0.8 * u }, g);
      svgEl("line", { y1: v, y2: v, x1: -half, x2: half, stroke: "var(--qb-grid)", "stroke-width": 0.8 * u }, g);
    }
    const sb = svgEl("text", { x: -half + 2 * u, y: half - 4 * u, "font-size": 15 * u }, gDyn);
    sb.textContent = `grid: ${gstep} px`;
    sb.style.fill = "var(--qb-muted)";
    // regions
    svgEl("path", { d: pathOf(res.truth), fill: "#888", "fill-opacity": 0.18, stroke: "#666", "stroke-width": 3 * u }, gDyn);
    for (const c of res.pred) svgEl("path", { d: pathOf(c), fill: "#1f77b4", "fill-opacity": 0.12, stroke: "#1f77b4", "stroke-width": 2.2 * u }, gDyn);
    // whiskers
    const dmax = Math.max(res.dist.hausdorff, 1e-6);
    const whisk = (from: M.Pt[], to: M.Pt[][], every: number) => {
      for (let k = 0; k < from.length; k += every) {
        const p = from[k], q = M.nearestOn(p, to);
        if (q.d < 0.05) continue;
        svgEl("line", { x1: p[0], y1: p[1], x2: q.q[0], y2: q.q[1], stroke: toCss(viridis(0.15 + 0.8 * q.d / dmax)), "stroke-width": 1.6 * u }, gDyn);
      }
    };
    const tr = M.resample(res.truth, Math.max(0.5, R / 12)), pr = res.pred.flatMap(c => M.resample(c, Math.max(0.5, R / 12)));
    whisk(tr, res.pred, 1);
    whisk(pr, [res.truth], 1);
    // Hausdorff whisker: the farthest sampled boundary point (same samples as the metric)
    const step = st.R < 15 ? 0.2 : 0.5;
    const sa = M.resample(res.truth, step), sp = res.pred.flatMap(c => M.resample(c, step));
    let best = { d: -1, p: [0, 0] as M.Pt, to: res.pred };
    res.dist.dAB.forEach((v, i) => { if (v > best.d) best = { d: v, p: sa[i], to: res.pred }; });
    res.dist.dBA.forEach((v, i) => { if (v > best.d) best = { d: v, p: sp[i], to: [res.truth] }; });
    const bq = M.nearestOn(best.p, best.to).q;
    if (best.d > 0.05) {
      svgEl("line", { x1: best.p[0], y1: best.p[1], x2: bq[0], y2: bq[1], stroke: "#d62728", "stroke-width": 3.5 * u }, gDyn);
      const mx = (best.p[0] + bq[0]) / 2, right = mx > 0;
      const t = svgEl("text", { x: mx + (right ? -10 : 6) * u, y: (best.p[1] + bq[1]) / 2 - 8 * u, "text-anchor": right ? "end" : "start", "font-size": 18 * u, "font-weight": 700, class: "qb-halo" }, gDyn);
      t.textContent = `H = ${res.dist.hausdorff.toFixed(1)} px`;
      t.style.fill = "#d62728";
    }
    // handles
    M.controlPoints(R, st.pred).forEach((c, k) => setA(ctrlH[k], { cx: c[0], cy: c[1], r: 7 * u, "stroke-width": 2.5 * u }));
    const f = M.frame(R, st.pred.spike.phi), L = st.pred.spike.len;
    setA(spikeH, { x: f.p[0] + L * f.n[0] - 6 * u, y: f.p[1] + L * f.n[1] - 6 * u, width: 12 * u, height: 12 * u, "stroke-width": 2.5 * u });
    spikeH.style.display = st.pred.spike.on ? "" : "none";
    setA(fragH, { cx: st.pred.fragment.u * R, cy: st.pred.fragment.v * R, r: 9 * u, "stroke-width": 2.5 * u, "stroke-dasharray": `${4 * u} ${3 * u}` });
    fragH.style.display = st.pred.fragment.on ? "" : "none";
    // numbers and histogram
    vals["IoU"].textContent = res.iou.toFixed(3);
    vals["Dice"].textContent = res.dice.toFixed(3);
    vals["Hausdorff"].textContent = res.dist.hausdorff.toFixed(2);
    vals["HD95"].textContent = res.dist.hd95.toFixed(2);
    vals["ASD"].textContent = res.dist.asd.toFixed(2);
    const d = [...res.dist.dAB, ...res.dist.dBA], top = Math.max(2, Math.ceil(res.dist.hausdorff + 0.5));
    const nb = 20, w = top / nb, counts = new Array(nb).fill(0);
    for (const v of d) counts[Math.min(nb - 1, Math.floor(v / w))]++;
    const sh = counts.map(c => c / d.length);
    hist.clear();
    hist.setDomain([0, top], [0, Math.max(0.2, Math.ceil(Math.max(...sh) * 10) / 10)]);
    hist.histogram(Array.from({ length: nb + 1 }, (_, i) => i * w), sh, { color: seriesColor(0) });
    hist.vline(res.dist.asd, { text: "ASD", color: seriesColor(2) });
    hist.vline(res.dist.hausdorff, { text: "H", color: seriesColor(1) });
  }

  load(p0);
}

autoMount(NAME, mount);
