// Fast marching vs Dijkstra on the pixel grid. Left: the arrival time T of a front that
// starts at the seeds and moves with speed 1/P (|∇T| = P), computed by fast marching
// (lib/shortestpath.ts: Dijkstra's order with the upwind update). Right: the same with
// Dijkstra on the 4- or 8-neighbour grid (edge cost = length × mean P). Colour = time,
// white lines = fronts at equal time steps. Click a target: the minimal path by gradient
// descent on T (orange) and Dijkstra's pixel path (blue).
// Auto-mounts on <div data-demo="fastmarch">. Options: data-preset="flat|diag|two|neurite|edge".
import { htmlEl, slider, presets, button, autoMount, svgEl, svgPoint } from "../lib/ui";
import { Grey, drawGrey, gradient } from "../lib/imageops";
import { drawScalarImage, viridis } from "../lib/colour";
import { fastMarching, seedNeighbourhood, dijkstraGrid, descendPath, tracePath, pathLength, polylineLength, Conn } from "../lib/shortestpath";
import { getImage, ImageId, IMAGE_LABELS, CELL_CREDIT, neuron, nucleusPoint } from "../livewire/images";
import { liveWireFeatures, SIZE } from "../livewire/model";

const NAME = "fastmarch";
type CostId = "flat" | "edges" | "ridge";
type Pt = [number, number];

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] figure { margin: 0; flex: 1 1 300px; min-width: 220px; max-width: 400px; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] .fm-box { position: relative; width: 100%; }
[data-demo="${NAME}"] .fm-box canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .fm-box svg { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; cursor: crosshair; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 330px; gap: .35em; }
[data-demo="${NAME}"] .fm-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
[data-demo="${NAME}"] .fm-credit { font-size: 12px; color: var(--qb-muted); }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] figure { flex-basis: 360px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .2em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .fm-lab, .reveal [data-demo="${NAME}"] .qb-note { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface State { image: ImageId; cost: CostId; conn: Conn; k: number; seeds: Pt[]; target: Pt | null; show: "time" | "image" }
const BASE: State = { image: "nucleus", cost: "flat", conn: 4, k: 20, seeds: [[128, 128]], target: null, show: "time" };
const PRESETS: Record<string, { label: string; s: () => Partial<State> }> = {
  flat: { label: "flat cost: circles, not diamonds", s: () => ({ cost: "flat", conn: 4, seeds: [[128, 128]], target: null }) },
  diag: { label: "Dijkstra 4-nbr vs fast marching", s: () => ({ cost: "flat", conn: 4, seeds: [[30, 225]], target: [225, 30] }) },
  two: { label: "two seeds: where fronts meet", s: () => ({ cost: "flat", conn: 8, seeds: [[70, 90], [190, 170]], target: null }) },
  neurite: { label: "minimal path along a neurite", s: () => ({ image: "neurite", cost: "ridge", conn: 8, seeds: [rp(neuron().tips[0])], target: rp(neuron().tips[5]), show: "image" }) },
  edge: { label: "edge cost: front slows at the nucleus", s: () => ({ image: "nucleus", cost: "edges", conn: 8, k: 8, seeds: [nucleusPoint(180).map((v, i) => v + (i === 0 ? 25 : 0)) as Pt], target: [230, 200] }) },
};
const rp = (p: Pt): Pt => [Math.round(p[0]), Math.round(p[1])];

function mount(host: HTMLElement): void {
  injectStyle();
  const startKey = host.dataset.preset && PRESETS[host.dataset.preset] ? host.dataset.preset : "flat";
  const init = (): State => ({ ...BASE, ...PRESETS[startKey].s() });
  let st = init();

  const mkFig = (title: string) => {
    const fig = htmlEl("figure", "", "", host);
    const cap = htmlEl("figcaption", "", title, fig);
    const box = htmlEl("div", "fm-box", "", fig);
    const canvas = htmlEl("canvas", "", "", box);
    const svg = svgEl("svg", { viewBox: `0 0 ${SIZE} ${SIZE}`, preserveAspectRatio: "none" }, box);
    const g = svgEl("g", {}, svg);
    return { fig, cap, canvas, svg, g };
  };
  const L = mkFig("fast marching: arrival time T"), R = mkFig("");
  const credit = htmlEl("div", "fm-credit", CELL_CREDIT, L.fig);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "fm-lab", "click on an image places", panel);
  let clickMode: "seed" | "target" = "target";
  const modeBtns = presets(panel, [
    { label: "target", apply: () => { clickMode = "target"; } },
    { label: "add seed", apply: () => { clickMode = "seed"; } },
  ]);
  modeBtns[0].classList.add("active");
  button(panel, "clear seeds", () => { st.seeds = []; st.target = null; compute(); });
  htmlEl("div", "fm-lab", "image", panel);
  const imgBtns = presets(panel, IMAGE_LABELS.map(([id, l]) => ({ label: l, apply: () => { st.image = id; load(); } })));
  htmlEl("div", "fm-lab", "cost P (time per pixel)", panel);
  const COSTS: [CostId, string][] = [["flat", "P = 1"], ["edges", "slow at edges"], ["ridge", "fast on bright lines"]];
  const costBtns = presets(panel, COSTS.map(([c, l]) => ({ label: l, apply: () => { st.cost = c; compute(); } })));
  const sK = slider(panel, "edge contrast k", 5, 60, 1, st.k, v => { st.k = v; compute(); });
  htmlEl("div", "fm-lab", "Dijkstra neighbours", panel);
  const connBtns = presets(panel, [
    { label: "4 neighbours", apply: () => { st.conn = 4; compute(); } },
    { label: "8 neighbours", apply: () => { st.conn = 8; compute(); } },
  ]);
  const showBtns = presets(panel, [
    { label: "show time", apply: () => { st.show = "time"; draw(); } },
    { label: "show image", apply: () => { st.show = "image"; draw(); } },
  ]);
  htmlEl("div", "fm-lab", "presets", panel);
  const presetBtns = presets(panel, Object.entries(PRESETS).map(([k, p]) => ({ label: p.label, apply: () => { st = { ...BASE, ...p.s() }; load(); } })));
  button(panel, "reset", () => { presetBtns.forEach(b => b.classList.remove("active")); st = init(); load(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "White lines: fronts at equal time steps. Orange: gradient descent on T; blue: Dijkstra's pixel path.", panel);

  let img: Grey | null = null;
  let P = new Float64Array(SIZE * SIZE);
  let T: Float64Array | null = null, D: Float64Array | null = null, pred: Int32Array | null = null, msF = 0, msD = 0;
  let featCache: { id: ImageId; f: ReturnType<typeof liveWireFeatures> } | null = null;
  let gradCache: { id: ImageId; mag: Float32Array } | null = null;

  function sync(): void {
    sK.set(st.k);
    imgBtns.forEach((b, i) => b.classList.toggle("active", IMAGE_LABELS[i][0] === st.image));
    costBtns.forEach((b, i) => b.classList.toggle("active", COSTS[i][0] === st.cost));
    connBtns.forEach((b, i) => b.classList.toggle("active", (i ? 8 : 4) === st.conn));
    showBtns.forEach((b, i) => b.classList.toggle("active", (i ? "image" : "time") === st.show));
    credit.style.display = st.image === "cell" ? "" : "none";
    R.cap.textContent = `Dijkstra, ${st.conn} neighbours`;
  }
  let token = 0;
  async function load(): Promise<void> {
    const my = ++token;
    const g = await getImage(st.image);
    if (my !== token) return;
    img = g; compute();
  }

  function costMap(): void {
    const n = SIZE * SIZE;
    if (st.cost === "flat" || !img) { P.fill(1); return; }
    if (st.cost === "edges") {
      if (!gradCache || gradCache.id !== st.image) gradCache = { id: st.image, mag: gradient(img, 1.5).mag.data };
      const m = gradCache.mag;
      for (let i = 0; i < n; i++) P[i] = 1 + (m[i] / st.k) ** 2;   // P = 1/g with the edge-stopping g = 1/(1 + (|∇I|/k)²)
      return;
    }
    if (!featCache || featCache.id !== st.image) featCache = { id: st.image, f: liveWireFeatures(img, 1) };
    const fR = featCache.f.fR;
    for (let i = 0; i < n; i++) P[i] = 0.05 + fR[i];
  }

  function compute(): void {
    sync();
    if (!img) return;
    costMap();
    const seeds = st.seeds.map(([x, y]) => y * SIZE + x);
    if (!seeds.length) { T = D = null; pred = null; draw(); return; }
    let t0 = performance.now();
    T = fastMarching(SIZE, SIZE, P, seedNeighbourhood(SIZE, SIZE, P, seeds)).T;
    msF = performance.now() - t0;
    t0 = performance.now();
    const res = dijkstraGrid({ width: SIZE, height: SIZE, conn: st.conn, edge: (p, q, len) => len * 0.5 * (P[p] + P[q]), sources: seeds });
    msD = performance.now() - t0;
    D = res.dist; pred = res.pred;
    draw();
  }

  function paint(canvas: HTMLCanvasElement, V: Float64Array | null, mx: number): void {
    if (!V || st.show === "image" || !img) { if (img) drawGrey(canvas, img, { mode: "clip", lo: 0, hi: 255 }); return; }
    drawScalarImage(canvas, V, SIZE, SIZE, { map: viridis, lo: 0, hi: mx });
    const step = niceStep(mx / 12), c2 = canvas.getContext("2d")!, id = c2.getImageData(0, 0, SIZE, SIZE);
    for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
      const i = y * SIZE + x, b = Math.floor(V[i] / step);
      if ((x + 1 < SIZE && Math.floor(V[i + 1] / step) !== b) || (y + 1 < SIZE && Math.floor(V[i + SIZE] / step) !== b))
        id.data[4 * i] = id.data[4 * i + 1] = id.data[4 * i + 2] = 255;
    }
    c2.putImageData(id, 0, 0);
  }
  const niceStep = (v: number) => { const p = 10 ** Math.floor(Math.log10(Math.max(v, 1e-9))), m = v / p; return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p; };

  function draw(): void {
    let mx = 0;
    if (T) for (let i = 0; i < T.length; i++) if (T[i] < Infinity && T[i] > mx) mx = T[i];
    if (D) for (let i = 0; i < D.length; i++) if (D[i] < Infinity && D[i] > mx) mx = D[i];
    paint(L.canvas, T, mx); paint(R.canvas, D, mx);
    const lines: string[] = [];
    let fmPath: Pt[] = [], djPath: number[] = [];
    if (st.target && T && D && pred) {
      const ti = st.target[1] * SIZE + st.target[0];
      fmPath = descendPath(T, SIZE, SIZE, st.target);
      djPath = tracePath(pred, ti, D);
      lines.push(`at the target: T ${T[ti].toFixed(1)}, d ${D[ti].toFixed(1)}`);
      if (st.cost === "flat") {
        const e = Math.min(...st.seeds.map(s => Math.hypot(s[0] - st.target![0], s[1] - st.target![1])));
        lines.push(`straight line: ${e.toFixed(1)}`);
      }
      lines.push(`path length: FM ${polylineLength(fmPath).toFixed(1)}, Dijkstra ${pathLength(djPath, SIZE).toFixed(1)}`);
    } else lines.push(st.seeds.length ? "click to place a target" : "add a seed");
    lines.push(`time: FM ${msF.toFixed(0)} ms, Dijkstra ${msD.toFixed(0)} ms`);
    ro.textContent = lines.join("\n");
    for (const F of [L, R]) {
      F.g.replaceChildren();
      const pts = (a: Pt[]) => a.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
      if (djPath.length) svgEl("polyline", { points: pts(djPath.map(p => [p % SIZE, Math.floor(p / SIZE)] as Pt)), fill: "none", stroke: "#1f77b4", "stroke-width": 2.2, "stroke-dasharray": F === L ? "5 3" : "" }, F.g);
      if (fmPath.length && F === L) svgEl("polyline", { points: pts(fmPath), fill: "none", stroke: "#ff7f0e", "stroke-width": 2.4 }, F.g);
      for (const s of st.seeds) svgEl("circle", { cx: s[0], cy: s[1], r: 5, fill: "#d62728", stroke: "#fff", "stroke-width": 1.5 }, F.g);
      if (st.target) svgEl("rect", { x: st.target[0] - 5, y: st.target[1] - 5, width: 10, height: 10, fill: "#fff", stroke: "#222", "stroke-width": 2 }, F.g);
    }
  }

  for (const F of [L, R]) F.svg.addEventListener("pointerdown", ev => {
    ev.preventDefault();
    const [x, y] = svgPoint(F.svg, ev);
    const p: Pt = [Math.max(0, Math.min(SIZE - 1, Math.round(x))), Math.max(0, Math.min(SIZE - 1, Math.round(y)))];
    presetBtns.forEach(b => b.classList.remove("active"));
    if (clickMode === "seed") { st.seeds.push(p); compute(); }
    else { st.target = p; draw(); }
  });

  load();
}

autoMount(NAME, mount);
