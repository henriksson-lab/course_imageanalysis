// B-spline basis functions on a knot line (splines day, unit 2). Degree 0–3; the knots are
// draggable along the axis (drop one onto another to make a double or triple knot); the sum
// of all basis functions (dashed) is 1 inside the valid range. The lower panel steps through
// the Cox–de Boor tower of the selected function: level q shows the degree-q functions that
// it is built from, and the two ramps that blend them into the next level. "Boxes" overlays
// the uniform kernel β^p (a box convolved with itself p times) centred on the selected
// function: on uniform knots it is the same bump.
// <div data-demo="bsplinebasis" data-preset="uniform|deg0|clamped|double|triple|uneven"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { Plot, plotFigure, seriesColor } from "../lib/plot";
import { basisN, coxDeBoorTower, bsplineKernel, knotDomain } from "../lib/spline";

const NAME = "bsplinebasis";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; }
.qb-demo[data-demo="${NAME}"] .bb-plots { flex: 0 1 640px; min-width: 0; max-width: calc((100vh - 150px) * 1.3); display: flex; flex-direction: column; gap: .2em; }
.qb-demo[data-demo="${NAME}"].qb-wide { flex-wrap: nowrap; }
.qb-demo[data-demo="${NAME}"] .bb-plots .qb-plot { flex: none; max-width: none; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 330px; min-width: 260px; gap: .3em; font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-btn { font-size: 14px; padding: .15em .5em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] .bb-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .bb-plots { flex: 0 0 690px; max-width: none; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 400px; gap: .2em; font-size: inherit; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .05em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { padding: .2em .45em; font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .bb-lab, .reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

interface State { knots: number[]; p: number; sel: number; level: number; boxes: boolean }
const U = Array.from({ length: 11 }, (_, i) => i);
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "uniform", label: "uniform cubic", s: { knots: U, p: 3, sel: 3, level: 3 } },
  { id: "deg0", label: "degree 0 → 3: boxes", s: { knots: U, p: 0, sel: 3, level: 0, boxes: true } },
  { id: "clamped", label: "clamped ends", s: { knots: [0, 0, 0, 0, 2, 4, 6, 8, 10, 10, 10, 10], p: 3, sel: 0, level: 3 } },
  { id: "double", label: "double knot: kink in the slope", s: { knots: [0, 1, 2, 3, 4, 5, 5, 7, 8, 9, 10], p: 3, sel: 3, level: 3 } },
  { id: "triple", label: "triple knot: corner", s: { knots: [0, 1, 2, 3, 5, 5, 5, 7, 8, 9, 10], p: 3, sel: 3, level: 3 } },
  { id: "uneven", label: "uneven knots", s: { knots: [0, 0.5, 1.5, 2, 4, 4.5, 6, 8.5, 9, 9.5, 10], p: 3, sel: 3, level: 3 } },
];
const DEFAULT: State = { knots: U, p: 3, sel: 3, level: 3, boxes: false };
const SNAP = 0.18;

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...DEFAULT, ...p0.s, knots: [] };

  const col = htmlEl("div", "bb-plots", "", host);
  const f1 = plotFigure(col, "");
  const plot = new Plot(f1, { width: 690, height: 330, x: [-0.4, 10.4], y: [-0.2, 1.12], ylabel: "N_i,p(t)", xticks: U, yticks: [0, 0.5, 1], margin: { b: 34 } });
  const f2 = plotFigure(col, "");
  const tower = new Plot(f2, { width: 690, height: 200, x: [-0.4, 10.4], y: [-0.05, 1.1], xlabel: "t", ylabel: "level q", xticks: U, yticks: [0, 1] });
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "bb-lab", "presets", panel);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p.s) })));
  htmlEl("div", "bb-lab", "degree", panel);
  const degRow = htmlEl("div", "qb-presets", "", panel);
  const degBtns = [0, 1, 2, 3].map(d => button(degRow, "degree " + d, () => { st.p = d; st.level = d; fix(); draw(); }));
  const selS = slider(panel, "function i", 0, 9, 1, st.sel, v => { st.sel = v; draw(); });
  const levS = slider(panel, "tower level q", 0, 3, 1, st.level, v => { st.level = Math.min(v, st.p); draw(); });
  checkbox(panel, "boxes: β^p = box convolved p times", st.boxes, v => { st.boxes = v; draw(); });
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "uniform knots", () => { st.knots = U.slice(); draw(); });
  button(row, "reset", () => load(p0.s));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Drag the knots (triangles) along the axis; drop one onto a neighbour to double it.", panel);

  const knotG = svgEl("g", {}, plot.over);

  function load(s: Partial<State>): void {
    Object.assign(st, DEFAULT, s);
    st.knots = (s.knots ?? U).slice();
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i].s === s));
    fix(); draw();
  }
  function fix(): void {
    const nb = st.knots.length - st.p - 1;
    st.sel = Math.max(0, Math.min(st.sel, nb - 1));
    st.level = Math.min(st.level, st.p);
    selS.input.max = String(nb - 1); selS.set(st.sel);
    levS.input.max = String(st.p); levS.set(st.level);
  }

  function draw(): void {
    const K = st.knots, p = st.p, nb = K.length - p - 1;
    fix();
    degBtns.forEach((b, i) => b.classList.toggle("active", i === p));
    plot.clear(); tower.clear();
    const [a, b] = knotDomain(K, p);
    plot.band([a, b], [-0.2, -0.2], [1.12, 1.12], { color: "var(--qb-c3, #2ca02c)", fillOpacity: 0.07 });
    const xs: number[] = [];
    for (let k = 0; k <= 520; k++) xs.push(-0.2 + 10.4 * k / 520);
    // evaluate exactly at the knots too, so corners show
    for (const k of K) xs.push(k, k - 1e-9);
    xs.sort((u, v) => u - v);
    const sum = new Array(xs.length).fill(0);
    for (let i = 0; i < nb; i++) {
      const ys = xs.map(t => basisN(i, p, K, t));
      ys.forEach((v, j) => (sum[j] += v));
      plot.line(xs, ys, { color: seriesColor(i), width: i === st.sel ? 5 : 2.2, opacity: i === st.sel ? 1 : 0.75 });
    }
    plot.line(xs, sum, { color: "var(--qb-fg, #222)", width: 2.5, dash: "7 5" });
    plot.text(a + 0.05, 1.07, "valid range: sum = 1", { size: 18, color: "var(--qb-c3, #2ca02c)" });
    if (st.boxes) {
      const c = (K[st.sel] + K[st.sel + p + 1]) / 2;
      plot.line(xs, xs.map(t => bsplineKernel(p, t - c)), { color: "var(--qb-fg, #222)", width: 3.5, dash: "2 5" });
    }
    // knot handles
    knotG.replaceChildren();
    K.forEach((k, idx) => {
      const mult = K.filter(v => v === k).length, rank = K.slice(0, idx).filter(v => v === k).length;
      const x = plot.x(k), y = plot.y(-0.1) + rank * 0 - rank * 13;
      const tri = svgEl("path", { d: `M${x},${y - 11}L${x - 10},${y + 7}L${x + 10},${y + 7}Z` }, knotG);
      tri.style.cssText = `fill:${mult > 1 ? "var(--qb-c2,#d62728)" : "var(--qb-fg,#222)"};stroke:var(--qb-bg,#fff);stroke-width:1.5;cursor:ew-resize;touch-action:none`;
      tri.addEventListener("pointerdown", ev => {
        ev.preventDefault();
        tri.setPointerCapture(ev.pointerId);
        const move = (e: PointerEvent) => {
          let [t] = plot.toData(e);
          const lo = idx > 0 ? K[idx - 1] : -0.2, hi = idx < K.length - 1 ? K[idx + 1] : 10.2;
          t = Math.min(hi, Math.max(lo, t));
          if (t - lo < SNAP) t = lo; else if (hi - t < SNAP) t = hi;
          K[idx] = Math.round(t * 100) / 100;
          pbtns.forEach(bb => bb.classList.remove("active"));
          drawCurvesOnly();
        };
        const up = () => { tri.removeEventListener("pointermove", move); tri.removeEventListener("pointerup", up); draw(); };
        tri.addEventListener("pointermove", move);
        tri.addEventListener("pointerup", up);
      });
    });
    drawTower(xs);
    // readout
    const groups: string[] = [];
    for (let j = 0; j < K.length;) {
      let m = 1;
      while (j + m < K.length && K[j + m] === K[j]) m++;
      if (m > 1 && K[j] > a && K[j] < b) groups.push(`knot ${K[j]} ×${m}: C${sup(p - m)}`);
      j += m;
    }
    ro.textContent = `degree ${p}: ${nb} functions, C${sup(p - 1)} at simple knots\n` +
      `N${sub(st.sel)},${p}: support [${K[st.sel]}, ${K[st.sel + p + 1]}]` +
      (groups.length ? "\n" + groups.join("\n") : "") +
      `\nvalid range [${a}, ${b}]`;
  }
  // light redraw while a knot is dragged (handles stay, so the pointer capture survives)
  function drawCurvesOnly(): void {
    const K = st.knots, p = st.p, nb = K.length - p - 1;
    plot.clear();
    const [a, b] = knotDomain(K, p);
    plot.band([a, b], [-0.2, -0.2], [1.12, 1.12], { color: "var(--qb-c3, #2ca02c)", fillOpacity: 0.07 });
    const xs: number[] = [];
    for (let k = 0; k <= 400; k++) xs.push(-0.2 + 10.4 * k / 400);
    for (const k of K) xs.push(k);
    xs.sort((u, v) => u - v);
    const sum = new Array(xs.length).fill(0);
    for (let i = 0; i < nb; i++) {
      const ys = xs.map(t => basisN(i, p, K, t));
      ys.forEach((v, j) => (sum[j] += v));
      plot.line(xs, ys, { color: seriesColor(i), width: i === st.sel ? 5 : 2.2 });
    }
    plot.line(xs, sum, { color: "var(--qb-fg, #222)", width: 2.5, dash: "7 5" });
    const tris = knotG.querySelectorAll("path");
    K.forEach((k, idx) => {
      const x = plot.x(k), rank = K.slice(0, idx).filter(v => v === k).length, y = plot.y(-0.1) - rank * 13;
      tris[idx]?.setAttribute("d", `M${x},${y - 11}L${x - 10},${y + 7}L${x + 10},${y + 7}Z`);
    });
  }

  function drawTower(xs: number[]): void {
    const K = st.knots, i = st.sel, q = st.level;
    // functions of level q that N_{i,p} is built from: N_{i+j,q}, j = 0 … p − q
    for (let j = 0; j <= st.p - q; j++) {
      const ys = xs.map(t => coxDeBoorTower(i + j, q, K, t)[q][0]);
      tower.line(xs, ys, { color: seriesColor(i + j), width: 3.5 });
    }
    if (q < st.p) {
      // ramps of the next level for the first pair: (t − t_k)/(t_{k+q+1} − t_k) and (t_{k+q+2} − t)/(…)
      for (let j = 0; j <= st.p - q - 1; j++) {
        const k = i + j, d1 = K[k + q + 1] - K[k], d2 = K[k + q + 2] - K[k + 1];
        if (d1 > 0) tower.line([K[k], K[k + q + 1]], [0, 1], { color: seriesColor(k), width: 1.8, dash: "5 4" });
        if (d2 > 0) tower.line([K[k + 1], K[k + q + 2]], [1, 0], { color: seriesColor(k + 1), width: 1.8, dash: "5 4" });
      }
    }
    if (st.boxes) {
      const c = (K[i] + K[i + q + 1]) / 2;
      tower.line(xs, xs.map(t => bsplineKernel(q, t - c)), { color: "var(--qb-fg, #222)", width: 3, dash: "2 5" });
    }
    tower.text(-0.3, 1.0, q < st.p ? `level ${q}: N${sub(i)}…N${sub(i + st.p - q)} (degree ${q}); dashed ramps blend neighbours into level ${q + 1}` : `level ${q} = the selected N${sub(i)},${st.p}`, { size: 17 });
  }

  load(p0.s);
}

const SUB = "₀₁₂₃₄₅₆₇₈₉", SUP: Record<string, string> = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³" };
const sub = (n: number) => String(n).split("").map(c => SUB[Number(c)] ?? c).join("");
const sup = (n: number) => (n < 0 ? "⁻¹ (jump)" : String(n).split("").map(c => SUP[c] ?? c).join(""));

autoMount(NAME, mount);
