// distws: splitting touching objects with the distance transform (watershed day, unit
// ws-distance). Panels: the mask (discs and ellipses: drag the centre handle to move, the
// rim handle to resize; or a thresholded synthetic nuclei image), its distance transform
// (colour, iso-lines every few pixels, peaks = h-maxima as white dots), and the result of a
// watershed of −DT inside the mask (one colour per object). Options: metric (city-block,
// chessboard, chamfer 3-4, Euclidean), h, flood −DT or the image gradient (markers from the
// DT), fill holes. With exactly two discs, a profile along the line through the centres shows
// the two peaks, the neck, and the depth compared with h and with r − √(r² − d²/4).
// Options: data-preset (split | merge | ellipse | three | holes | nuclei).
import { autoMount, htmlEl, slider, presets, button, checkbox, svgEl, svgPoint } from "../lib/ui";
import { Grey } from "../lib/imageops";
import { nuclei } from "../lib/synth";
import { labelColor, viridis } from "../lib/colour";
import { Shape, shapesMask, splitMask, withHoles, fillHoles, profile, neckOf, neckDepth, Metric, FloodMode, DistRun } from "./geom";

const NAME = "distws";
const N = 180;
const disc = (x: number, y: number, r: number): Shape => ({ x, y, a: r, b: r, angle: 0 });

interface Preset { id: string; label: string; shapes?: Shape[]; nuclei?: boolean; h: number; metric: Metric; holes?: number; fill?: boolean; mode?: FloodMode }
const PRESETS: Preset[] = [
  { id: "split", label: "two discs: split", shapes: [disc(70, 90, 30), disc(110, 90, 30)], h: 2, metric: "euclidean" },
  { id: "merge", label: "move them together until they merge", shapes: [disc(73, 90, 30), disc(107, 90, 30)], h: 3, metric: "euclidean" },
  { id: "ellipse", label: "an ellipse is oversplit", shapes: [{ x: 90, y: 90, a: 55, b: 22, angle: 0.3 }], h: 0, metric: "euclidean" },
  { id: "three", label: "three in a row", shapes: [disc(50, 90, 24), disc(90, 90, 24), disc(130, 90, 24)], h: 1, metric: "euclidean" },
  { id: "holes", label: "noisy mask with holes", shapes: [disc(65, 90, 30), disc(115, 90, 30)], h: 0, metric: "euclidean", holes: 12, fill: false },
  { id: "nuclei", label: "nuclei image", nuclei: true, h: 2, metric: "euclidean" },
];
const METRICS: { id: Metric; label: string }[] = [
  { id: "cityblock", label: "city-block" }, { id: "chessboard", label: "chessboard" }, { id: "3-4", label: "chamfer 3-4" }, { id: "euclidean", label: "Euclidean" },
];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .6em 1em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .dw-left { flex: 1 1 600px; min-width: 0; display: flex; flex-direction: column; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .dw-imgs { display: flex; gap: .5em; width: 100%; }
.qb-demo[data-demo="${NAME}"] .dw-fig { flex: 1 1 0; min-width: 0; margin: 0; position: relative; }
.qb-demo[data-demo="${NAME}"] .dw-fig figcaption { font-size: 17px; text-align: center; margin-bottom: .15em; }
.qb-demo[data-demo="${NAME}"] .dw-box { position: relative; width: 100%; aspect-ratio: 1; }
.qb-demo[data-demo="${NAME}"] .dw-box canvas { position: absolute; inset: 0; width: 100%; height: 100%; image-rendering: pixelated; border-radius: 3px; outline: 1px solid var(--qb-grid, #ccc); }
.qb-demo[data-demo="${NAME}"] .dw-box svg { position: absolute; inset: 0; width: 100%; height: 100%; }
.qb-demo[data-demo="${NAME}"] .dw-prof svg { width: 100%; display: block; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 330px; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .dw-lab { font-size: 15px; color: var(--qb-muted); }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .08em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex-basis: 320px; gap: .25em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = { shapes: [] as Shape[], useNuclei: false, thr: 115, h: 2, metric: "euclidean" as Metric, mode: "dt" as FloodMode, holes: 0, fill: false };
  let nuc: { image: Grey; count: number } | null = null, run: DistRun | null = null, mask: Uint8Array = new Uint8Array(N * N);

  const left = htmlEl("div", "dw-left", "", host);
  const imgs = htmlEl("div", "dw-imgs", "", left);
  const fig = (title: string) => {
    const f = htmlEl("figure", "dw-fig", "", imgs);
    htmlEl("figcaption", "", title, f);
    const box = htmlEl("div", "dw-box", "", f);
    const c = htmlEl("canvas", "", "", box);
    c.width = N; c.height = N;
    const svg = svgEl("svg", { viewBox: `0 0 ${N} ${N}` }, box);
    return { c, ctx: c.getContext("2d")!, svg };
  };
  const pMask = fig("mask"), pDt = fig("distance transform"), pRes = fig("watershed of −DT");
  const prof = htmlEl("div", "dw-prof", "", left);
  const profSvg = svgEl("svg", { viewBox: "0 0 820 150", role: "img", "aria-label": "distance transform along the line between the two centres" }, prof);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "dw-lab", "metric", panel);
  const metBtns = presets(panel, METRICS.map(m => ({ label: m.label, apply: () => { st.metric = m.id; recompute(); } })));
  const hS = slider(panel, "h", 0, 10, 0.1, 2, v => { st.h = v; recompute(); });
  const modeBtns = presets(panel, [
    { label: "flood −DT", apply: () => { st.mode = "dt"; recompute(); } },
    { label: "flood gradient, markers from DT", apply: () => { st.mode = "gradient"; recompute(); } },
  ]);
  const fillC = checkbox(panel, "fill holes first", false, v => { st.fill = v; recompute(); });
  const tS = slider(panel, "threshold", 60, 200, 1, 115, v => { st.thr = v; recompute(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  ro.style.whiteSpace = "pre-line";
  htmlEl("div", "dw-lab", "examples", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  button(htmlEl("div", "qb-presets", "", panel), "reset", () => load(p0));
  const note = htmlEl("div", "qb-note", "", panel);

  function load(p: Preset): void {
    st.shapes = (p.shapes ?? []).map(s => ({ ...s }));
    st.useNuclei = !!p.nuclei;
    st.h = p.h; hS.set(p.h);
    st.metric = p.metric; st.holes = p.holes ?? 0; st.fill = p.fill ?? false; fillC.checked = st.fill;
    st.mode = p.mode ?? "dt";
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    recompute();
  }

  let pending = false;
  function recompute(): void {
    if (st.useNuclei) {
      nuc ??= nuclei({ width: N, height: N, count: 15, radius: [10, 15], touching: 0.6, seed: 4, noise: 6 });
      mask = Uint8Array.from(nuc.image.data, v => (v > st.thr ? 1 : 0));
    } else {
      mask = shapesMask(N, N, st.shapes);
      if (st.holes) mask = withHoles(mask, N, N, st.holes);
    }
    if (st.fill) mask = fillHoles(mask, N, N);
    run = splitMask(mask, N, N, st.metric, st.h, st.mode, st.useNuclei && nuc ? nuc.image : undefined);
    draw();
  }
  const schedule = () => { if (pending) return; pending = true; requestAnimationFrame(() => { pending = false; recompute(); }); };

  function draw(): void {
    if (!run) return;
    metBtns.forEach((b, i) => b.classList.toggle("active", METRICS[i].id === st.metric));
    modeBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.mode === "dt")));
    (tS.input.closest("label") as HTMLElement).style.display = st.useNuclei ? "" : "none";
    const n = N * N, dt = run.dt;
    let dmax = 0;
    for (const v of dt) if (v > dmax && Number.isFinite(v)) dmax = v;
    const a = pMask.ctx.createImageData(N, N), b = pDt.ctx.createImageData(N, N), c = pRes.ctx.createImageData(N, N);
    const step = dmax > 24 ? 4 : 2, band = (i: number) => Math.floor(dt[i] / step);
    for (let i = 0; i < n; i++) {
      const j = 4 * i, m = mask[i], x = i % N;
      let g = m ? 40 : 255;
      if (st.useNuclei && nuc) { const v = Math.max(0, Math.min(255, nuc.image.data[i])); g = m ? 0.45 * v : 0.3 * v + 170; }
      a.data[j] = a.data[j + 1] = a.data[j + 2] = g; a.data[j + 3] = 255;
      if (m) {
        const col = viridis(dmax > 0 ? dt[i] / dmax : 0);
        let k = 1;
        if ((x + 1 < N && mask[i + 1] && band(i + 1) !== band(i)) || (i + N < n && mask[i + N] && band(i + N) !== band(i))) k = 0.55;
        b.data[j] = 255 * col[0] * k; b.data[j + 1] = 255 * col[1] * k; b.data[j + 2] = 255 * col[2] * k;
      } else b.data[j] = b.data[j + 1] = b.data[j + 2] = 255;
      b.data[j + 3] = 255;
      const L = run.labels[i];
      if (L > 0) { const col = labelColor(L); c.data[j] = 255 * col[0]; c.data[j + 1] = 255 * col[1]; c.data[j + 2] = 255 * col[2]; }
      else { c.data[j] = c.data[j + 1] = c.data[j + 2] = m ? 30 : 255; }
      c.data[j + 3] = 255;
    }
    // object boundaries in dark on the result
    for (let i = 0; i < n; i++) {
      const L = run.labels[i], x = i % N;
      if (L > 0 && ((x + 1 < N && run.labels[i + 1] > 0 && run.labels[i + 1] !== L) || (i + N < n && run.labels[i + N] > 0 && run.labels[i + N] !== L))) {
        const j = 4 * i; c.data[j] = c.data[j + 1] = c.data[j + 2] = 20;
      }
    }
    pMask.ctx.putImageData(a, 0, 0); pDt.ctx.putImageData(b, 0, 0); pRes.ctx.putImageData(c, 0, 0);
    // peaks
    pDt.svg.replaceChildren();
    for (const p of run.peakCentres) svgEl("circle", { cx: p.x + 0.5, cy: p.y + 0.5, r: 3.2, fill: "#fff", stroke: "#111", "stroke-width": 1.2 }, pDt.svg);
    drawHandles();
    drawProfile();
    const comp = run.nComponents;
    ro.textContent = `components of the mask: ${comp}\npeaks (markers): ${run.peakCentres.length} · objects: ${run.nObjects}` +
      (st.useNuclei && nuc ? `\ntrue number of nuclei: ${nuc.count}` : "") + `\nDT + markers + flooding: ${run.ms.toFixed(0)} ms`;
    note.textContent = st.useNuclei
      ? "Thresholded synthetic nuclei. Touching nuclei form one component; the peaks of the DT give one marker per nucleus if h is right."
      : "Drag a centre (●) to move a shape, a rim handle (○) to resize it. White dots: the peaks (h-maxima of the DT) = one marker each.";
  }

  function drawHandles(): void {
    const svg = pMask.svg;
    svg.replaceChildren();
    if (st.useNuclei) return;
    st.shapes.forEach((s, k) => {
      const rx = s.x + s.a * Math.cos(s.angle), ry = s.y + s.a * Math.sin(s.angle);
      svgEl("line", { x1: s.x, y1: s.y, x2: rx, y2: ry, stroke: "#ff7f0e", "stroke-width": 1, "stroke-dasharray": "3 2" }, svg);
      const c = svgEl("circle", { cx: s.x, cy: s.y, r: 5, fill: "#ff7f0e", stroke: "#fff", "stroke-width": 1.5 }, svg);
      const r = svgEl("circle", { cx: rx, cy: ry, r: 5, fill: "#fff", stroke: "#ff7f0e", "stroke-width": 2 }, svg);
      for (const el of [c, r]) { el.style.cursor = "grab"; el.style.touchAction = "none"; }
      const drag = (el: SVGCircleElement, f: (x: number, y: number) => void) => {
        el.addEventListener("pointerdown", ev => {
          ev.preventDefault(); ev.stopPropagation();
          el.setPointerCapture(ev.pointerId);
          const move = (e: PointerEvent) => { const [x, y] = svgPoint(svg, e); f(x, y); schedule(); };
          const up = () => { el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); el.removeEventListener("pointercancel", up); };
          el.addEventListener("pointermove", move); el.addEventListener("pointerup", up); el.addEventListener("pointercancel", up);
        });
      };
      drag(c, (x, y) => { const sh = st.shapes[k]; sh.x = Math.max(0, Math.min(N, x)); sh.y = Math.max(0, Math.min(N, y)); });
      drag(r, (x, y) => {
        const sh = st.shapes[k], d = Math.max(4, Math.min(80, Math.hypot(x - sh.x, y - sh.y))), q = sh.b / sh.a;
        sh.a = d; sh.b = d * q;
      });
    });
  }

  function drawProfile(): void {
    const g = profSvg;
    g.replaceChildren();
    const two = !st.useNuclei && st.shapes.length === 2 && st.shapes.every(s => s.a === s.b);
    prof.style.display = two ? "" : "none";
    if (!two || !run) return;
    const [s1, s2] = st.shapes, d = Math.hypot(s2.x - s1.x, s2.y - s1.y);
    if (d < 1) return;
    const ux = (s2.x - s1.x) / d, uy = (s2.y - s1.y) / d;
    const x0 = s1.x - (s1.a + 4) * ux, y0 = s1.y - (s1.a + 4) * uy, x1 = s2.x + (s2.a + 4) * ux, y1 = s2.y + (s2.a + 4) * uy;
    const p = profile(run.dt, N, N, x0, y0, x1, y1), L = p[p.length - 1].t;
    let vmax = 1;
    for (const q of p) if (q.v > vmax) vmax = q.v;
    const W = 820, H = 150, ml = 46, mr = 350, mt = 14, mb = 26;
    const X = (t: number) => ml + (W - ml - mr) * t / L, Y = (v: number) => mt + (H - mt - mb) * (1 - v / (vmax * 1.08));
    for (const v of [0, Math.round(vmax / 2), Math.round(vmax)]) {
      svgEl("line", { x1: ml, x2: W - mr, y1: Y(v), y2: Y(v), stroke: "var(--qb-grid, #e3e3e3)" }, g);
      const t = svgEl("text", { x: ml - 6, y: Y(v) + 6, "text-anchor": "end" }, g); t.textContent = String(v); t.style.fontSize = "16px";
    }
    let path = "";
    p.forEach((q, i) => { path += `${i ? "L" : "M"}${X(q.t).toFixed(1)},${Y(q.v).toFixed(1)}`; });
    svgEl("path", { d: path, fill: "none", stroke: "#1f77b4", "stroke-width": 2.5 }, g);
    const nk = neckOf(p), depth = Math.min(nk.p1, nk.p2) - nk.neck;
    svgEl("circle", { cx: X(nk.t1), cy: Y(nk.p1), r: 5, fill: "#fff", stroke: "#111", "stroke-width": 1.5 }, g);
    svgEl("circle", { cx: X(nk.t2), cy: Y(nk.p2), r: 5, fill: "#fff", stroke: "#111", "stroke-width": 1.5 }, g);
    svgEl("circle", { cx: X(nk.tn), cy: Y(nk.neck), r: 5, fill: "#d62728" }, g);
    svgEl("line", { x1: X(nk.tn), x2: X(nk.tn), y1: Y(nk.neck), y2: Y(Math.min(nk.p1, nk.p2)), stroke: "#d62728", "stroke-width": 2 }, g);
    svgEl("line", { x1: X(nk.tn) - 14, x2: X(nk.tn) + 14, y1: Y(Math.min(nk.p1, nk.p2) - st.h), y2: Y(Math.min(nk.p1, nk.p2) - st.h), stroke: "#ff7f0e", "stroke-width": 3 }, g);
    const lab = svgEl("text", { x: ml + 4, y: H - 6 }, g); lab.textContent = "DT along the line through the two centres"; lab.style.fontSize = "15px"; lab.style.fill = "var(--qb-muted, #777)";
    const tx = W - mr + 16, eq = s1.a === s2.a ? s1.a : null;
    const lines = [
      `peaks ${nk.p1.toFixed(1)}, ${nk.p2.toFixed(1)} · neck ${nk.neck.toFixed(1)}`,
      `depth (red) ${depth.toFixed(2)} vs h (orange) ${st.h.toFixed(1)}`,
      eq !== null ? `r = ${eq.toFixed(1)}, d = ${d.toFixed(1)}: r − √(r² − d²/4) = ${d <= 2 * eq ? neckDepth(eq, d).toFixed(2) : "–"}` : "unequal radii",
      depth > st.h ? "depth > h: two markers → split" : "depth ≤ h: one marker → one object",
    ];
    lines.forEach((s, i) => { const t = svgEl("text", { x: tx, y: 30 + i * 28 }, g); t.textContent = s; t.style.fontSize = "15.5px"; if (i === 3) t.style.fontWeight = "700"; });
  }

  load(p0);
}

autoMount(NAME, mount);
