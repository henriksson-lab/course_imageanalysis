// medialaxis: the medial axis of a shape as the ridges of its distance transform (skeleton
// day, unit skel-medialaxis). A polygon shape (drag the round handles) is rasterised at
// 256 × 256. Left: the distance transform DT as colour with iso-lines every 8 steps, and the
// axis on top (red), by one of three methods: centres of maximal discs (= ridges of DT),
// Lantuéjoul's formula (erosions and openings), or thinning (with DT on it as radius). The
// grassfire slider t burns the shape from the boundary: the front is the iso-line DT = t, and
// the axis pixels light up when the fronts meet there. Click near the axis to draw its
// maximal disc. Right: the union of all axis discs, with missing pixels in red. A bump on the
// boundary (size, position) shows how a tiny change adds a whole branch.
//
// <div data-demo="medialaxis" data-preset="rect|bump|ring|star|metric|fire"
//      data-shape="rect|ellipse|L|star|dumbbell|ring" data-metric="cityblock|chessboard|euclidean"
//      data-method="ridges|lantuejoul|thinning"></div>
// The λ slider keeps only the axis pixels whose two nearest boundary points are at least λ
// apart (removes the small branches that boundary pixels create; maximal-disc method only).
import { autoMount, htmlEl, slider, presets, button, svgEl, draggable, svgPoint } from "../lib/ui";
import { viridis } from "../lib/colour";
import { Metric } from "../lib/skeleton";
import { ImgView, injectStyle } from "../canny/imgview";
import { choice, CHOICE_CSS } from "./choice";
import { Method, ShapeId, Shape, SHAPES, State, Preset, BASE, PRESETS, analyseShape } from "./shapes";

const NAME = "medialaxis";
const N = 256;
const STYLE = CHOICE_CSS + `
[data-demo="${NAME}"] { gap: .9em; flex-wrap: nowrap; align-items: flex-start; }
[data-demo="${NAME}"] .ma-main { flex: 0 1 430px; min-width: 260px; }
[data-demo="${NAME}"] .ma-mid { flex: 0 1 250px; min-width: 180px; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 360px; gap: .35em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
[data-demo="${NAME}"] .ma-legend { font-size: 14px; color: var(--qb-muted); }
[data-demo="${NAME}"] .ev-svg .ma-handle { pointer-events: all; cursor: grab; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .ma-main { flex: 0 0 440px; }
.reveal [data-demo="${NAME}"] .ma-mid { flex: 0 0 250px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 1 1 360px; gap: .2em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .06em .38em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .2em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { font-size: 14px; line-height: 1.2; }
@media (max-width: 860px) { [data-demo="${NAME}"] { flex-wrap: wrap; } }
`;

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const ds = host.dataset;
  const p0 = PRESETS.find(p => p.id === ds.preset);
  const start = (): State => ({
    ...BASE, ...(p0?.s ?? {}),
    ...(ds.shape && ds.shape in SHAPES ? { shape: ds.shape as ShapeId } : {}),
    ...(ds.metric ? { metric: ds.metric as Metric } : {}),
    ...(ds.method ? { method: ds.method as Method } : {}),
  });
  const st: State = start();
  let shape: Shape = SHAPES[st.shape].make();

  // ---- layout ----
  const main = htmlEl("div", "ma-main", "", host);
  const view = new ImgView(main, "distance transform DT and the axis (red)");
  const legend = htmlEl("div", "ma-legend", "drag the handles · click near the axis for its disc", main);
  const mid = htmlEl("div", "ma-mid", "", host);
  const rview = new ImgView(mid, "union of the axis discs");
  htmlEl("div", "ma-legend", "grey: reconstructed · red: missing", mid);
  const ro = htmlEl("div", "qb-readout", "", mid);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ev-lab", "examples", panel);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const shapeC = choice(panel, "shape", (Object.keys(SHAPES) as ShapeId[]).map(k => [k, SHAPES[k].label] as [ShapeId, string]), st.shape, v => { st.shape = v; shape = SHAPES[v].make(); st.pick = null; recompute(); });
  const methodC = choice<Method>(panel, "axis", [["ridges", "maximal discs"], ["lantuejoul", "Lantuéjoul"], ["thinning", "thinning"]], st.method, v => { st.method = v; recompute(); });
  const metricC = choice<Metric>(panel, "metric", [["cityblock", "city-block"], ["chessboard", "chessboard"], ["euclidean", "Euclidean"]], st.metric, v => { st.metric = v; recompute(); });
  const tS = slider(panel, "grassfire t", 0, 99, 1, st.t, v => { st.t = v; draw(); }, v => (v >= 99 ? "off" : String(v)));
  const lS = slider(panel, "λ (axis pruning)", 0, 24, 1, st.lambda, v => { st.lambda = v; recompute(); }, v => (v ? `${v} px` : "off"));
  const bS = slider(panel, "bump size", 0, 14, 1, st.bump, v => { st.bump = v; recompute(); });
  const bpS = slider(panel, "bump position", 0, 1, 0.01, st.bumpPos, v => { st.bumpPos = v; recompute(); });
  const note = htmlEl("div", "qb-note", "", panel);
  button(panel, "⟲ reset", () => { Object.assign(st, start()); shape = SHAPES[st.shape].make(); pbtns.forEach(b => b.classList.remove("active")); sync(); recompute(); });

  function sync(): void { shapeC.set(st.shape); methodC.set(st.method); metricC.set(st.metric); tS.set(st.t); lS.set(st.lambda); bS.set(st.bump); bpS.set(st.bumpPos); }
  function load(p: Preset): void { Object.assign(st, BASE, p.s); shape = SHAPES[st.shape].make(); sync(); recompute(); }

  // ---- computation ----
  let mask: Uint8Array = new Uint8Array(N * N), dt: Float32Array = new Float32Array(N * N), axis: Uint8Array = new Uint8Array(N * N), radius: Float32Array = new Float32Array(N * N);
  let recon: Uint8Array = new Uint8Array(N * N), maxDT = 1, info = "";
  const fmtR = (v: number) => (Math.abs(v - Math.round(v)) < 1e-6 ? String(Math.round(v)) : v.toFixed(2));
  let pending = false;
  function recompute(): void {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; compute(); draw(); });
  }
  function compute(): void {
    const r = analyseShape(shape, st);
    mask = r.mask; dt = r.dt; axis = r.axis; radius = r.radius; recon = r.recon; maxDT = r.maxDT; info = r.info; note.textContent = r.note;
  }
  const fmt = (v: number) => (Math.abs(v - Math.round(v)) < 1e-6 ? String(Math.round(v)) : v.toFixed(2));

  // ---- drawing ----
  function draw(): void {
    const px = new Uint8ClampedArray(N * N * 4), burn = st.t < 99, step = 8;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const i = y * N + x;
      let c: [number, number, number];
      if (!mask[i]) c = [0.13, 0.13, 0.15];
      else {
        const d = dt[i];
        c = viridis(Math.min(1, d / maxDT));
        const iso = (j: number) => mask[j] && Math.floor(dt[j] / step) !== Math.floor(d / step);
        if ((x + 1 < N && iso(i + 1)) || (y + 1 < N && iso(i + N))) c = c.map(v => v * 0.55) as [number, number, number];
        if (burn && d <= st.t) c = c.map(v => 0.25 + 0.2 * v) as [number, number, number];
        if (burn && d > st.t - 1 && d <= st.t) c = [1, 0.55, 0.1];
      }
      if (axis[i] && (!burn || dt[i] <= st.t)) c = [0.9, 0.1, 0.12];
      else if (axis[i] && burn) c = [0.95, 0.6, 0.6];
      px[4 * i] = 255 * c[0]; px[4 * i + 1] = 255 * c[1]; px[4 * i + 2] = 255 * c[2]; px[4 * i + 3] = 255;
    }
    view.rgba(px, N, N);
    const rp = new Uint8ClampedArray(N * N * 4);
    for (let i = 0; i < N * N; i++) {
      const c = mask[i] && recon[i] ? [150, 150, 150] : mask[i] ? [228, 26, 28] : recon[i] ? [255, 150, 0] : [245, 245, 245];
      if (axis[i]) { c[0] = 40; c[1] = 40; c[2] = 40; }
      rp.set([c[0], c[1], c[2], 255], 4 * i);
    }
    rview.rgba(rp, N, N);
    ro.textContent = info + (burn ? `\ngrassfire t = ${st.t}: ${countBurnt()} % burnt` : "");
    overlay();
  }
  function countBurnt(): string {
    let a = 0, b = 0;
    for (let i = 0; i < N * N; i++) if (mask[i]) { a++; if (dt[i] <= st.t) b++; }
    return (100 * b / Math.max(1, a)).toFixed(0);
  }

  function overlay(): void {
    view.clear();
    // the maximal disc of the picked axis pixel
    if (st.pick) {
      const [x, y] = st.pick, i = y * N + x;
      if (axis[i]) {
        const r = radius[i], cx = x + 0.5, cy = y + 0.5;
        const m: Metric = st.method === "lantuejoul" ? (st.metric === "cityblock" ? "cityblock" : "chessboard") : st.metric;
        const col = "#fff";
        if (m === "euclidean") view.add("circle", { cx, cy, r, fill: "none", stroke: col, "stroke-width": 2.5 });
        else if (m === "chessboard") view.add("rect", { x: cx - r + 0.5, y: cy - r + 0.5, width: 2 * r - 1, height: 2 * r - 1, fill: "none", stroke: col, "stroke-width": 2.5 });
        else { const R = r - 0.5; view.add("polygon", { points: `${cx},${cy - R} ${cx + R},${cy} ${cx},${cy + R} ${cx - R},${cy}`, fill: "none", stroke: col, "stroke-width": 2.5 }); }
        view.add("circle", { cx, cy, r: 2.2, fill: col, stroke: "#000", "stroke-width": 1 });
        legend.textContent = `disc at (${x}, ${y}): radius ${fmtR(r)}`;
      } else legend.textContent = "drag the handles · click near the axis for its disc";
    } else legend.textContent = "drag the handles · click near the axis for its disc";
    // vertex handles
    shape.polys.forEach((p, k) => {
      view.add("polygon", { points: p.map(q => q.join(",")).join(" "), fill: "none", stroke: "#fff", "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0.7 });
      p.forEach((q, j) => {
        const hnd = svgEl("circle", { cx: q[0], cy: q[1], r: 6, class: "ma-handle", fill: "rgba(255,255,255,0.85)", stroke: "#222", "stroke-width": 1.5, "vector-effect": "non-scaling-stroke" }, view.over);
        hnd.addEventListener("pointerdown", ev => ev.stopPropagation());
        draggable(hnd, ev => svgPoint(view.svg, ev), (nx, ny) => {
          shape.polys[k][j] = [Math.max(4, Math.min(N - 4, nx)), Math.max(4, Math.min(N - 4, ny))];
          hnd.setAttribute("cx", String(shape.polys[k][j][0])); hnd.setAttribute("cy", String(shape.polys[k][j][1]));
          recomputeLight();
        });
      });
    });
  }
  // while dragging: recompute without rebuilding the handles (they hold the pointer capture)
  let dragPending = false;
  function recomputeLight(): void {
    if (dragPending) return;
    dragPending = true;
    requestAnimationFrame(() => {
      dragPending = false;
      compute();
      const keep = [...view.over.querySelectorAll(".ma-handle")];
      draw();
      // draw() rebuilt the overlay: put the original (captured) handles back
      view.over.querySelectorAll(".ma-handle").forEach(h => h.remove());
      keep.forEach(h => view.over.appendChild(h));
    });
  }
  view.onPick((x, y) => {
    // nearest axis pixel within 8 px
    let best: [number, number] | null = null, bd = 64;
    for (let yy = Math.max(0, y - 8); yy <= Math.min(N - 1, y + 8); yy++) for (let xx = Math.max(0, x - 8); xx <= Math.min(N - 1, x + 8); xx++) {
      if (!axis[yy * N + xx]) continue;
      const d = (xx - x) ** 2 + (yy - y) ** 2;
      if (d < bd) { bd = d; best = [xx, yy]; }
    }
    st.pick = best;
    overlay();
  }, false);

  if (p0) pbtns[PRESETS.indexOf(p0)]?.classList.add("active");
  compute(); draw();
}

autoMount(NAME, mount);
