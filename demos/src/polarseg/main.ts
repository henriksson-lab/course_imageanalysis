// Closed boundaries by DP in polar coordinates. Left: the image with a draggable centre,
// some of the rays, the allowed annulus and the resulting contour. Middle: the unwrapped
// polar image (angle across, radius down) with the optimal path, or the cost, or the
// cumulative table M. The path has one radius per angle, may change by at most δ per
// angle step (penalty α per pixel of change), and must close (|r(0°) − r(last)| ≤ δ):
// ignored ("open"), by unrolling three times and keeping the middle ("unroll"), or exactly
// by trying every start radius ("exact").
// Auto-mounts on <div data-demo="polarseg">. Options: data-preset="done|nosmooth|open|badcentre|polarity|cell",
// data-image="nucleus|bean|cell", data-view="polar|cost|table".
import { htmlEl, slider, presets, button, autoMount, svgEl, draggable, svgPoint } from "../lib/ui";
import { Grey, drawGrey, gaussianBlur } from "../lib/imageops";
import { drawScalarImage, viridis, diverging, finiteRange } from "../lib/colour";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { Closure } from "../lib/dp";
import { Polarity, PolarSettings, PolarResult, segmentPolar, nucleusImage, beanImage, NUCLEUS, BEAN, SIZE, polygonCentroid } from "./model";

const NAME = "polarseg";
const KS = [24, 36, 60, 90, 120, 180, 360];
const R = 110;
type ImageId = "nucleus" | "bean" | "cell";
type View = "polar" | "cost" | "table";

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] .ps-left { flex: 0 1 330px; min-width: 220px; }
[data-demo="${NAME}"] .ps-mid { flex: 1 1 340px; min-width: 240px; max-width: 470px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .ps-box { position: relative; width: 100%; }
[data-demo="${NAME}"] .ps-box canvas { width: 100%; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .ps-left .ps-box canvas { aspect-ratio: 1; }
[data-demo="${NAME}"] .ps-mid .ps-box canvas { aspect-ratio: 1.35; }
[data-demo="${NAME}"] .ps-box svg { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; }
[data-demo="${NAME}"] .ps-left svg { touch-action: none; cursor: crosshair; }
[data-demo="${NAME}"] .ps-axis { font-size: 15px; color: var(--qb-muted); display: flex; justify-content: space-between; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 330px; gap: .35em; }
[data-demo="${NAME}"] .ps-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
[data-demo="${NAME}"] .ps-credit { font-size: 12px; color: var(--qb-muted); }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .ps-left { flex-basis: 360px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .2em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.reveal [data-demo="${NAME}"] .ps-lab { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface State extends PolarSettings { image: ImageId; view: View }

const BASE: State = {
  image: "nucleus", view: "polar", cx: NUCLEUS.cx, cy: NUCLEUS.cy, K: 120, R, rMin: 5, rMax: 105,
  delta: 1, alpha: 0.1, polarity: "bright-dark", closure: "exact", sigma: 1.5,
};
const PRESETS: Record<string, { label: string; s: Partial<State> }> = {
  done: { label: "nucleus: done", s: {} },
  nosmooth: { label: "no smoothness", s: { delta: 8, alpha: 0 } },
  open: { label: "open path: jump at 0°", s: { closure: "open", delta: 2, alpha: 0 } },
  badcentre: { label: "bad centre", s: { image: "bean", cx: 72, cy: 142, rMin: 3, rMax: 105 } },
  polarity: { label: "wrong polarity", s: { polarity: "dark-bright" } },
  cell: { label: "micrograph nucleus", s: { image: "cell", cx: 150, cy: 116, polarity: "line", sigma: 2, rMin: 30, rMax: 85, delta: 2, alpha: 0.05 } },
};

/** Centre, polarity and annulus that suit each image (applied when the image is chosen). */
const IMG_DEF: Record<ImageId, Partial<State>> = {
  nucleus: { cx: NUCLEUS.cx, cy: NUCLEUS.cy, polarity: "bright-dark", rMin: 5, rMax: 105 },
  bean: { cx: BEAN.cx, cy: BEAN.cy, polarity: "bright-dark", rMin: 3, rMax: 105 },
  cell: { cx: 150, cy: 116, polarity: "line", rMin: 30, rMax: 85 },
};

function mount(host: HTMLElement): void {
  injectStyle();
  const startKey = host.dataset.preset && PRESETS[host.dataset.preset] ? host.dataset.preset : "done";
  const init = (): State => {
    const im = host.dataset.image as ImageId | undefined;
    const s0: State = { ...BASE, ...PRESETS[startKey].s, ...(im && IMG_DEF[im] ? { image: im, ...IMG_DEF[im] } : {}) };
    if (host.dataset.view) s0.view = host.dataset.view as View;
    return s0;
  };
  let st = init();

  // ---- layout
  const left = htmlEl("figure", "ps-left", "", host);
  htmlEl("figcaption", "", "image: drag the centre (or click)", left);
  const lbox = htmlEl("div", "ps-box", "", left);
  const imgCanvas = htmlEl("canvas", "", "", lbox);
  const lsvg = svgEl("svg", { viewBox: `0 0 ${SIZE} ${SIZE}`, preserveAspectRatio: "none" }, lbox);
  const gRays = svgEl("g", {}, lsvg);
  const annIn = svgEl("circle", { fill: "none", stroke: "#4dd", "stroke-width": 1.2, "stroke-dasharray": "4 3" }, lsvg);
  const annOut = svgEl("circle", { fill: "none", stroke: "#4dd", "stroke-width": 1.2, "stroke-dasharray": "4 3" }, lsvg);
  const contour = svgEl("polyline", { fill: "none", stroke: "#ff7f0e", "stroke-width": 2.4, "stroke-linejoin": "round" }, lsvg);
  const seam = svgEl("line", { stroke: "#d62728", "stroke-width": 3, "stroke-dasharray": "4 3" }, lsvg);
  const start0 = svgEl("circle", { r: 4, fill: "#d62728" }, lsvg);
  const handle = svgEl("circle", { r: 8, fill: "rgba(255,255,255,.85)", stroke: "#d62728", "stroke-width": 3 }, lsvg);
  const credit = htmlEl("div", "ps-credit", CELL_CREDIT, left);

  const mid = htmlEl("figure", "ps-mid", "", host);
  const cap = htmlEl("figcaption", "", "", mid);
  const mbox = htmlEl("div", "ps-box", "", mid);
  const polCanvas = htmlEl("canvas", "", "", mbox);
  const msvg = svgEl("svg", { preserveAspectRatio: "none" }, mbox);
  const band = svgEl("g", {}, msvg);
  const ppath = svgEl("polyline", { fill: "none", stroke: "#ff7f0e", "stroke-width": 2.6, "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round" }, msvg);
  const axis = htmlEl("div", "ps-axis", "", mid);
  htmlEl("span", "", "θ = 0°", axis); htmlEl("span", "", "angle →   (radius ↓, centre at the top)", axis); htmlEl("span", "", "360°", axis);
  const viewBtns = presets(mid, [
    { label: "polar image", apply: () => { st.view = "polar"; drawMid(); } },
    { label: "cost", apply: () => { st.view = "cost"; drawMid(); } },
    { label: "table M", apply: () => { st.view = "table"; drawMid(); } },
  ]);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ps-lab", "image", panel);
  const imgBtns = presets(panel, [
    { label: "nucleus", apply: () => setImage("nucleus") },
    { label: "bean", apply: () => setImage("bean") },
    { label: "micrograph", apply: () => setImage("cell") },
  ]);
  const sDelta = slider(panel, "δ (max step)", 0, 8, 1, st.delta, v => { st.delta = v; schedule(); });
  const sAlpha = slider(panel, "α (penalty)", 0, 1, 0.05, st.alpha, v => { st.alpha = v; schedule(); });
  const sMin = slider(panel, "r min", 0, R - 1, 1, st.rMin, v => { st.rMin = Math.min(v, st.rMax - 1); schedule(); });
  const sMax = slider(panel, "r max", 1, R - 1, 1, st.rMax, v => { st.rMax = Math.max(v, st.rMin + 1); schedule(); });
  const sK = slider(panel, "angles K", 0, KS.length - 1, 1, KS.indexOf(st.K), v => { st.K = KS[v]; schedule(); }, v => String(KS[v]));
  htmlEl("div", "ps-lab", "edge polarity (inside → outside)", panel);
  const POLS: [Polarity, string][] = [["bright-dark", "bright→dark"], ["dark-bright", "dark→bright"], ["either", "either"], ["line", "dark line"]];
  const polBtns = presets(panel, POLS.map(([p, l]) => ({ label: l, apply: () => { st.polarity = p; schedule(); } })));
  htmlEl("div", "ps-lab", "closure", panel);
  const CLS: [Closure, string][] = [["open", "open"], ["unroll", "unroll ×3"], ["exact", "exact"]];
  const clBtns = presets(panel, CLS.map(([c, l]) => ({ label: l, apply: () => { st.closure = c; schedule(); } })));
  button(panel, "centre → centroid of the contour", () => { if (last) { const [x, y] = polygonCentroid(last.contour); st.cx = Math.round(x); st.cy = Math.round(y); schedule(); } });
  htmlEl("div", "ps-lab", "presets", panel);
  const presetBtns = presets(panel, Object.entries(PRESETS).map(([k, p]) => ({ label: p.label, apply: () => applyPreset(k) })));
  button(panel, "reset", () => { presetBtns.forEach(b => b.classList.remove("active")); st = init(); syncControls(); loadImage(); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  // ---- images
  const cache = new Map<ImageId, { raw: Grey; smooth: Grey }>();
  let cur: { raw: Grey; smooth: Grey } | null = null;
  async function getImage(id: ImageId): Promise<{ raw: Grey; smooth: Grey }> {
    let e = cache.get(id);
    if (!e) {
      const raw = id === "nucleus" ? nucleusImage() : id === "bean" ? beanImage() : await loadCell();
      e = { raw, smooth: gaussianBlur(raw, id === "cell" ? 2 : 1.5) };
      cache.set(id, e);
    }
    return e;
  }
  function loadImage(): void {
    const id = st.image;
    getImage(id).then(e => { if (st.image !== id) return; cur = e; drawGrey(imgCanvas, e.raw, { mode: "clip", lo: 0, hi: 255 }); compute(); });
  }
  function setImage(id: ImageId): void {
    st.image = id;
    Object.assign(st, IMG_DEF[id]);
    if (id === "cell") st.delta = Math.max(2, st.delta);
    syncControls(); loadImage();
  }
  function applyPreset(k: string): void {
    st = { ...BASE, ...PRESETS[k].s, view: st.view };
    syncControls(); loadImage();
  }
  function syncControls(): void {
    sDelta.set(st.delta); sAlpha.set(st.alpha); sMin.set(st.rMin); sMax.set(st.rMax); sK.set(KS.indexOf(st.K));
    polBtns.forEach((b, i) => b.classList.toggle("active", POLS[i][0] === st.polarity));
    clBtns.forEach((b, i) => b.classList.toggle("active", CLS[i][0] === st.closure));
    imgBtns.forEach((b, i) => b.classList.toggle("active", (["nucleus", "bean", "cell"] as ImageId[])[i] === st.image));
    viewBtns.forEach((b, i) => b.classList.toggle("active", (["polar", "cost", "table"] as View[])[i] === st.view));
    credit.style.display = st.image === "cell" ? "" : "none";
  }

  // ---- centre handle
  const toImg = (ev: PointerEvent): [number, number] => {
    const [x, y] = svgPoint(lsvg, ev);
    return [Math.max(0, Math.min(SIZE - 1, Math.round(x))), Math.max(0, Math.min(SIZE - 1, Math.round(y)))];
  };
  // while the centre is dragged, the exact closure (R runs) is replaced by the 3-run unroll
  // heuristic so the contour follows the pointer; the exact result is computed on release
  let dragging = false;
  draggable(handle, toImg, (x, y) => { dragging = true; st.cx = x; st.cy = y; presetBtns.forEach(b => b.classList.remove("active")); schedule(); });
  handle.addEventListener("pointerup", () => { if (dragging) { dragging = false; schedule(); } });
  handle.addEventListener("pointercancel", () => { if (dragging) { dragging = false; schedule(); } });
  lsvg.addEventListener("pointerdown", ev => {
    if (ev.target === handle) return;
    const [x, y] = toImg(ev);
    st.cx = x; st.cy = y; presetBtns.forEach(b => b.classList.remove("active")); schedule();
  });

  // ---- computation and drawing
  let last: PolarResult | null = null, pending = false;
  function schedule(): void {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; compute(); });
  }
  function compute(): void {
    if (!cur) return;
    last = segmentPolar(cur.smooth, dragging && st.closure === "exact" ? { ...st, closure: "unroll" } : st);
    drawLeft(); drawMid(); readout();
  }

  function drawLeft(): void {
    if (!last) return;
    const { cx, cy, K } = st;
    handle.setAttribute("cx", String(cx)); handle.setAttribute("cy", String(cy));
    annIn.setAttribute("cx", String(cx)); annIn.setAttribute("cy", String(cy)); annIn.setAttribute("r", String(Math.max(0.5, st.rMin)));
    annOut.setAttribute("cx", String(cx)); annOut.setAttribute("cy", String(cy)); annOut.setAttribute("r", String(st.rMax));
    gRays.replaceChildren();
    const every = Math.max(1, Math.round(K / 24));
    for (let k = 0; k < K; k += every) {
      const a = 2 * Math.PI * k / K;
      svgEl("line", { x1: cx, y1: cy, x2: cx + st.rMax * Math.cos(a), y2: cy + st.rMax * Math.sin(a), stroke: k === 0 ? "#d62728" : "rgba(80,220,220,.55)", "stroke-width": k === 0 ? 1.6 : 0.8 }, gRays);
    }
    const pts = last.contour;
    const closes = last.dp.closes;
    contour.setAttribute("points", [...pts, ...(closes ? [pts[0]] : [])].map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" "));
    seam.style.display = closes ? "none" : "";
    const [x0, y0] = pts[0], [x1, y1] = pts[pts.length - 1];
    seam.setAttribute("x1", String(x1)); seam.setAttribute("y1", String(y1)); seam.setAttribute("x2", String(x0)); seam.setAttribute("y2", String(y0));
    start0.setAttribute("cx", String(x0)); start0.setAttribute("cy", String(y0));
  }

  function drawMid(): void {
    if (!last) return;
    const K = st.K;
    viewBtns.forEach((b, i) => b.classList.toggle("active", (["polar", "cost", "table"] as View[])[i] === st.view));
    msvg.setAttribute("viewBox", `0 0 ${K} ${R}`);
    if (st.view === "polar") {
      drawScalarImage(polCanvas, last.polar.data, K, R, { lo: 0, hi: 255 });
      cap.textContent = "unwrapped image P(θ, r)";
    } else {
      const src = st.view === "cost" ? last.cost : last.dp.M, T = new Float64Array(K * R);
      for (let k = 0; k < K; k++) for (let i = 0; i < R; i++) {
        const v = src[k * R + i];
        T[i * K + k] = st.view === "table" && (i < st.rMin || i > st.rMax) ? NaN : v;
      }
      if (st.view === "cost") {
        drawScalarImage(polCanvas, T, K, R, { map: diverging, lo: -1, hi: 1 });
        cap.textContent = "cost C(θ, r): blue = cheap (an edge), red = expensive";
      } else {
        const [lo, hi] = finiteRange(T);
        drawScalarImage(polCanvas, T, K, R, { map: viridis, lo, hi, nan: [40, 40, 40, 255] });
        cap.textContent = st.closure === "exact" ? "table M of the best start radius (dark = low)" : "cumulative table M (dark = low)";
      }
    }
    band.replaceChildren();
    for (const r of [st.rMin, st.rMax]) svgEl("line", { x1: 0, x2: K, y1: r + 0.5, y2: r + 0.5, stroke: "#4dd", "stroke-width": 1.5, "stroke-dasharray": "5 4", "vector-effect": "non-scaling-stroke" }, band);
    ppath.setAttribute("points", Array.from(last.dp.path, (r, k) => `${k + 0.5},${r + 0.5}`).join(" "));
  }

  function readout(): void {
    if (!last) return;
    const d = last.dp;
    ro.textContent = [
      `gap |r(0°) − r(last)|: ${d.gap} px`,
      d.closes ? "  closes (gap ≤ δ)" : "  does NOT close (gap > δ)",
      `path cost: ${d.closedCost.toFixed(2)}`,
      `operations: ${d.ops.toLocaleString("en-US").replace(/,/g, " ")}`,
      `time: ${last.ms.toFixed(0)} ms`,
    ].join("\n");
  }

  syncControls();
  loadImage();
}

autoMount(NAME, mount);
