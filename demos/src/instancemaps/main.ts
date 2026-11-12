// From masks to objects (modern deep learning day, unit instances/). A synthetic cell image
// with ground truth; the "network output" is SIMULATED: the exact training target of the
// chosen route, degraded by error sliders (noise, blur, contact bias, missed dim cells).
// Routes: semantic mask + connected components; boundary class + labelling of the interiors;
// distance map + h-maxima + watershed; star-convex polygons + NMS; flows followed to their
// sinks. Panels: input, predicted map, resulting objects with merges (M) and splits (S)
// marked, and the evaluation day's readouts (lib/eval.ts): F1 and matching accuracy at τ = 0.5.
// Click the map to see one pixel's polygon (star) or path (flows). Animate: particles move to
// their sinks (flows), the flood rises (distance).
// Auto-mounts on <div data-demo="instancemaps">. Options: data-preset="semantic|gap|dist|
// elong|cshape|flows|noisyflows|contact|crowded", data-route.
import { htmlEl, slider, presets, button, autoMount, svgEl, svgPoint, type Slider } from "../lib/ui";
import { toRGBA } from "../lib/imageops";
import { scalarToRGBA, viridis, labelLUT, hsvToRgb } from "../lib/colour";
import { iouTable, match, objectScores, matchingAccuracy, splitsMerges } from "../lib/eval";
import * as M from "./model";

const NAME = "instancemaps";
const N = M.N;
const ROUTES: M.Route[] = ["semantic", "boundary", "distance", "star", "flows"];
const ROUTE_LABEL: Record<M.Route, string> = { semantic: "semantic", boundary: "boundary class", distance: "distance", star: "star-convex", flows: "flows" };

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .im-left { flex: 1 1 640px; min-width: 300px; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .im-panels { display: flex; gap: .5em; }
[data-demo="${NAME}"] figure { margin: 0; flex: 1 1 0; min-width: 0; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin: 0 0 .15em; white-space: nowrap; }
[data-demo="${NAME}"] .im-stage { position: relative; width: 100%; aspect-ratio: 1; }
[data-demo="${NAME}"] .im-stage canvas, [data-demo="${NAME}"] .im-stage svg { position: absolute; inset: 0; width: 100%; height: 100%; border-radius: 3px; }
[data-demo="${NAME}"] .im-stage canvas { image-rendering: pixelated; }
[data-demo="${NAME}"] .im-stage svg { touch-action: none; overflow: visible; }
[data-demo="${NAME}"] .im-map svg { cursor: crosshair; }
[data-demo="${NAME}"] .im-sim { font-size: 14px; color: var(--qb-muted); text-align: center; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 340px; gap: .3em; }
[data-demo="${NAME}"] .im-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre; line-height: 1.3; }
[data-demo="${NAME}"] .im-hide { display: none !important; }
[data-demo="${NAME}"] svg text { font-size: 9px; font-weight: 700; paint-order: stroke; stroke: #000; stroke-width: 2px; }
.reveal [data-demo="${NAME}"] { font-size: 17px; gap: .6em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .08em .4em; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .15em; flex-basis: 350px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 16px; padding: .3em .5em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface State { scene: M.SceneId; route: M.Route; err: M.Errors; post: M.Post; click: number }
interface Preset { id: string; label: string; s: Partial<Omit<State, "err" | "post">> & { err?: Partial<M.Errors>; post?: Partial<M.Post> } }

const E0: M.Errors = { noise: 0, blur: 0, contact: 0, missed: 0 };
const P0: M.Post = { thr: 0.5, h: 0.15, nRays: 32, nmsT: 0.3, steps: 40 };
const THR0: Record<M.Route, number> = { semantic: 0.5, boundary: 0.5, distance: 0.1, star: 0.6, flows: 0.5 };
const PRESETS: Preset[] = [
  { id: "semantic", label: "semantic mask merges", s: { scene: "mixed", route: "semantic" } },
  { id: "gap", label: "boundary: gaps merge", s: { scene: "mixed", route: "boundary", err: { contact: 0.5 } } },
  { id: "dist", label: "distance separates", s: { scene: "mixed", route: "distance", err: { contact: 0.5 } } },
  { id: "elong", label: "elongated: oversplit", s: { scene: "mixed", route: "distance", err: { noise: 0.2 } } },
  { id: "cshape", label: "C-shape: no polygon fits", s: { scene: "mixed", route: "star", click: 65 * N + 113 } },
  { id: "flows", label: "flows follow the bend", s: { scene: "mixed", route: "flows" } },
  { id: "noisyflows", label: "noisy flows: many sinks", s: { scene: "mixed", route: "flows", err: { noise: 0.7 } } },
  { id: "contact", label: "contact bias: merges", s: { scene: "mixed", route: "distance", err: { contact: 1 } } },
  { id: "crowded", label: "crowded tissue", s: { scene: "crowded", route: "distance", err: { noise: 0.1, blur: 1 } } },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { scene: "mixed", route: "semantic", err: { ...E0 }, post: { ...P0 }, click: -1 };
  const cache = new Map<M.SceneId, { sc: M.Scene; T: M.Targets }>();
  let sc: M.Scene, T: M.Targets, pred: M.Prediction, res: M.Result;
  let anim = 0, animT = -1;

  // ---- layout ----
  const left = htmlEl("div", "im-left", "", host);
  const panels = htmlEl("div", "im-panels", "", left);
  const mk = (cap: string, cls = "") => {
    const f = htmlEl("figure", cls, "", panels);
    const c = htmlEl("figcaption", "", cap, f);
    const stg = htmlEl("div", "im-stage", "", f);
    const cv = htmlEl("canvas", "", "", stg); cv.width = N; cv.height = N;
    const svg = svgEl("svg", { viewBox: `0 0 ${N} ${N}` }, stg);
    return { cap: c, cv, svg };
  };
  const pIn = mk("input image"), pMap = mk("predicted map", "im-map"), pRes = mk("objects");
  htmlEl("div", "im-sim", "simulated network output: the exact target map plus the errors you set", left);
  const ro = htmlEl("div", "qb-readout", "", left);
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "im-lab", "presets", panel);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => applyPreset(p) })));
  htmlEl("div", "im-lab", "what the network predicts", panel);
  const rRow = htmlEl("div", "qb-presets", "", panel);
  const rBtns = ROUTES.map(r => button(rRow, ROUTE_LABEL[r], () => { st.route = r; st.post.thr = THR0[r]; thrS.set(st.post.thr); st.click = -1; run(); }));
  htmlEl("div", "im-lab", "network errors", panel);
  const noiseS = slider(panel, "noise", 0, 1, 0.05, 0, v => { st.err.noise = v; run(); });
  const blurS = slider(panel, "blur", 0, 4, 0.25, 0, v => { st.err.blur = v; run(); });
  const contS = slider(panel, "contact bias", 0, 1, 0.05, 0, v => { st.err.contact = v; run(); });
  const missS = slider(panel, "missed dim cells", 0, 1, 0.05, 0, v => { st.err.missed = v; run(); });
  htmlEl("div", "im-lab", "post-processing", panel);
  const thrS = slider(panel, "threshold", 0.05, 0.95, 0.05, 0.5, v => { st.post.thr = v; run(); });
  const hS = slider(panel, "h (markers)", 0.02, 0.5, 0.01, P0.h, v => { st.post.h = v; run(); });
  const raysS = slider(panel, "rays n", 4, 64, 1, P0.nRays, v => { st.post.nRays = v; run(); }, v => String(nearestRays(v)));
  const nmsS = slider(panel, "NMS IoU t", 0.05, 0.9, 0.05, P0.nmsT, v => { st.post.nmsT = v; run(); });
  const stepS = slider(panel, "flow steps", 0, 100, 1, P0.steps, v => { st.post.steps = v; run(); });
  const btnRow = htmlEl("div", "qb-presets", "", panel);
  const animBtn = button(btnRow, "animate", () => startAnim());
  button(btnRow, "reset", () => applyPreset(p0));
  const sceneBtns = (["mixed", "crowded"] as M.SceneId[]).map(s => button(btnRow, s === "mixed" ? "mixed cells" : "crowded", () => { st.scene = s; st.click = -1; run(); }));
  const note = htmlEl("div", "qb-note", "", panel);

  const nearestRays = (v: number) => [4, 8, 16, 32, 64].reduce((a, b) => (Math.abs(b - v) < Math.abs(a - v) ? b : a));

  pMap.svg.addEventListener("pointerdown", ev => {
    const [x, y] = svgPoint(pMap.svg, ev);
    st.click = Math.max(0, Math.min(N - 1, Math.floor(y))) * N + Math.max(0, Math.min(N - 1, Math.floor(x)));
    drawMapOverlay();
  });

  function applyPreset(p: Preset): void {
    stopAnim();
    st.scene = p.s.scene ?? "mixed"; st.route = p.s.route ?? "semantic"; st.click = p.s.click ?? -1;
    st.err = { ...E0, ...(p.s.err ?? {}) };
    st.post = { ...P0, thr: THR0[st.route], ...(p.s.post ?? {}) };
    noiseS.set(st.err.noise); blurS.set(st.err.blur); contS.set(st.err.contact); missS.set(st.err.missed);
    thrS.set(st.post.thr); hS.set(st.post.h); raysS.set(st.post.nRays); nmsS.set(st.post.nmsT); stepS.set(st.post.steps);
    run();
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
  }

  function run(): void {
    stopAnim();
    let c = cache.get(st.scene);
    if (!c) { const s = M.makeScene(st.scene); c = { sc: s, T: M.computeTargets(s) }; cache.set(st.scene, c); }
    sc = c.sc; T = c.T;
    st.post.nRays = nearestRays(raysS.get());
    pred = M.predict(sc, T, st.route, st.err);
    res = M.postprocess(pred, st.post);
    rBtns.forEach((b, i) => b.classList.toggle("active", ROUTES[i] === st.route));
    sceneBtns.forEach((b, i) => b.classList.toggle("active", ["mixed", "crowded"][i] === st.scene));
    const show = (s: Slider, on: boolean) => (s.input.parentElement as HTMLElement).classList.toggle("im-hide", !on);
    show(hS, st.route === "distance"); show(raysS, st.route === "star"); show(nmsS, st.route === "star"); show(stepS, st.route === "flows");
    animBtn.classList.toggle("im-hide", st.route !== "flows" && st.route !== "distance");
    draw();
  }

  // ---- drawing ----
  function put(cv: HTMLCanvasElement, rgba: Uint8ClampedArray): void {
    cv.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(rgba), N, N), 0, 0);
  }
  function drawInput(): void {
    put(pIn.cv, toRGBA(sc.image, { lo: 0, hi: 230 }));
    pIn.svg.innerHTML = "";
  }
  function mapRGBA(): Uint8ClampedArray {
    const out = new Uint8ClampedArray(4 * N * N);
    if (st.route === "semantic") return scalarToRGBA(pred.fg, { lo: 0, hi: 1 });
    if (st.route === "boundary") {
      for (let i = 0; i < N * N; i++) {
        const b = pred.pBnd![i], it = pred.pInt![i];
        out[4 * i] = 255 * Math.min(1, b + 0.1 * it); out[4 * i + 1] = 255 * it * 0.85; out[4 * i + 2] = 255 * b; out[4 * i + 3] = 255;
      }
      return out;
    }
    if (st.route === "distance" || st.route === "star") return scalarToRGBA(pred.D!, { map: viridis, lo: 0, hi: 1 });
    for (let i = 0; i < N * N; i++) {
      const on = pred.fg[i] > st.post.thr ? 1 : 0.12;
      const h = (Math.atan2(pred.vy![i], pred.vx![i]) / (2 * Math.PI) + 1) % 1;
      const [r, g, b] = hsvToRgb([h * 360, 0.85, on]);
      out[4 * i] = 255 * r; out[4 * i + 1] = 255 * g; out[4 * i + 2] = 255 * b; out[4 * i + 3] = 255;
    }
    return out;
  }
  function drawMap(): void {
    put(pMap.cv, mapRGBA());
    pMap.cap.textContent = "predicted map: " + ({ semantic: "P(cell)", boundary: "interior / boundary", distance: "distance to border", star: "objectness", flows: "flow direction" } as Record<M.Route, string>)[st.route];
    drawMapOverlay();
  }
  function drawMapOverlay(): void {
    const svg = pMap.svg;
    svg.innerHTML = "";
    if (st.route === "flows") {
      const g = svgEl("g", { stroke: "#fff", "stroke-width": 0.45, opacity: 0.85 }, svg);
      let d = "";
      for (let y = 3; y < N; y += 6) for (let x = 3; x < N; x += 6) {
        const p = y * N + x;
        if (pred.fg[p] <= st.post.thr) continue;
        const u = pred.vx![p] * 2.6, v = pred.vy![p] * 2.6;
        d += `M${x + 0.5 - u / 2},${y + 0.5 - v / 2}l${u},${v}`;
        const hx = x + 0.5 + u / 2, hy = y + 0.5 + v / 2, a = Math.atan2(v, u);
        d += `M${hx},${hy}l${-1.1 * Math.cos(a - 0.5)},${-1.1 * Math.sin(a - 0.5)}M${hx},${hy}l${-1.1 * Math.cos(a + 0.5)},${-1.1 * Math.sin(a + 0.5)}`;
      }
      svgEl("path", { d, fill: "none" }, g);
      if (st.click >= 0) {
        let x = st.click % N, y = Math.floor(st.click / N), d2 = `M${x + 0.5},${y + 0.5}`;
        const samp = (a: Float32Array, xx: number, yy: number) => {
          const xi = Math.max(0, Math.min(N - 2, Math.floor(xx))), yi = Math.max(0, Math.min(N - 2, Math.floor(yy)));
          const fx = Math.max(0, Math.min(1, xx - xi)), fy = Math.max(0, Math.min(1, yy - yi)), p = yi * N + xi;
          return (1 - fy) * ((1 - fx) * a[p] + fx * a[p + 1]) + fy * ((1 - fx) * a[p + N] + fx * a[p + N + 1]);
        };
        for (let j = 0; j < st.post.steps; j++) { x += samp(pred.vx!, x, y); y += samp(pred.vy!, x, y); d2 += `L${x + 0.5},${y + 0.5}`; }
        svgEl("path", { d: d2, fill: "none", stroke: "#fff", "stroke-width": 1.2 }, svg);
        svgEl("circle", { cx: x + 0.5, cy: y + 0.5, r: 1.6, fill: "#ff3b30" }, svg);
      }
    }
    if (st.route === "star" && st.click >= 0 && pred.D![st.click] > 0) {
      const n = st.post.nRays, step = M.MAX_RAYS / n, p = st.click, x0 = p % N + 0.5, y0 = Math.floor(p / N) + 0.5;
      let d = "", poly = "";
      for (let k = 0; k < n; k++) {
        const r = pred.rays![p * M.MAX_RAYS + k * step], [dx, dy] = M.rayDir(k, n);
        d += `M${x0},${y0}l${r * dx},${r * dy}`;
        poly += `${x0 + r * dx},${y0 + r * dy} `;
      }
      svgEl("path", { d, stroke: "#fff", "stroke-width": 0.35, fill: "none" }, svg);
      svgEl("polygon", { points: poly, stroke: "#ff3b30", "stroke-width": 1, fill: "none" }, svg);
      svgEl("circle", { cx: x0, cy: y0, r: 1.2, fill: "#ff3b30" }, svg);
    }
  }

  function drawResult(water = Infinity, particles?: { x: Float32Array; y: Float32Array }): void {
    const lut = labelLUT(Math.max(2, res.count + 1), [0, 0, 0], 255);
    const out = new Uint8ClampedArray(4 * N * N);
    const base = toRGBA(sc.image, { lo: 0, hi: 230 });
    for (let i = 0; i < N * N; i++) {
      const l = res.labels[i];
      const show = l > 0 && (!res.level || !(res.level[i] > water));
      for (let c = 0; c < 3; c++) out[4 * i + c] = show ? 0.75 * lut[4 * (l % 256) + c] + 0.25 * base[4 * i + c] : 0.35 * base[4 * i + c];
      out[4 * i + 3] = 255;
    }
    // truth outlines
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const p = y * N + x, l = sc.labels[p];
      if (l && (x === N - 1 || y === N - 1 || sc.labels[p + 1] !== l || sc.labels[p + N] !== l)) { out[4 * p] = 255; out[4 * p + 1] = 255; out[4 * p + 2] = 255; }
    }
    if (particles) for (let p = 0; p < N * N; p++) {
      const x = particles.x[p], y = particles.y[p];
      if (!Number.isFinite(x)) continue;
      const q = Math.round(y) * N + Math.round(x);
      out[4 * q] = 255; out[4 * q + 1] = 255; out[4 * q + 2] = 255;
    }
    put(pRes.cv, out);
  }

  function evaluate(): string {
    const tb = iouTable(sc.labels, res.labels), m = match(tb, 0.5, "optimal"), s = objectScores(m), sm = splitsMerges(tb);
    const svg = pRes.svg;
    svg.innerHTML = "";
    const cT = cents(sc.labels, sc.n), cP = cents(res.labels, res.count);
    for (const mg of sm.merges) { const c = cP[mg.p]; if (c) label(svg, c[0], c[1], "M", "#ff3b30"); }
    for (const sp of sm.splits) { const c = cT[sp.t]; if (c) label(svg, c[0], c[1], "S", "#ffd23f"); }
    return `true cells ${sc.n}   objects found ${res.count}\n` +
      `TP ${m.tp}   FP ${m.fp}   FN ${m.fn}   (IoU ≥ 0.5)\n` +
      `F1 ${s.f1.toFixed(2)}   matching accuracy ${matchingAccuracy(m).toFixed(2)}\n` +
      `merges (M) ${sm.merges.length}   splits (S) ${sm.splits.length}`;
  }
  /** Centroid of each label 1..n (index label − 1). */
  function cents(lab: Int32Array, n: number): [number, number][] {
    const sx = new Float64Array(n + 1), sy = new Float64Array(n + 1), c = new Float64Array(n + 1);
    for (let p = 0; p < N * N; p++) { const l = lab[p]; if (l) { sx[l] += p % N; sy[l] += Math.floor(p / N); c[l]++; } }
    return Array.from({ length: n }, (_, k) => [sx[k + 1] / (c[k + 1] || 1), sy[k + 1] / (c[k + 1] || 1)] as [number, number]);
  }
  function label(svg: SVGSVGElement, x: number, y: number, t: string, col: string): void {
    const e = svgEl("text", { x: x + 0.5, y: y + 3.5, "text-anchor": "middle", fill: col }, svg); e.textContent = t;
  }

  function draw(): void {
    drawInput(); drawMap(); drawResult();
    ro.textContent = evaluate();
    note.textContent = {
      semantic: "Connected components of P(cell) > threshold.",
      boundary: "Interiors (the most likely class) are labelled, then grown back over the boundary pixels.",
      distance: "Mask = distance > threshold; markers = h-maxima; watershed of −distance inside the mask.",
      star: "Every pixel with objectness > threshold proposes its polygon; NMS keeps the best. Click the map to see a pixel's rays.",
      flows: "Every cell pixel follows the arrows for the given number of steps; pixels that end in the same sink form one object. Click the map to follow one pixel.",
    }[st.route];
  }

  // ---- animation ----
  function stopAnim(): void { if (anim) cancelAnimationFrame(anim); anim = 0; animT = -1; }
  function startAnim(): void {
    stopAnim();
    if (st.route === "flows") {
      const mask = Uint8Array.from(pred.fg, v => (v > st.post.thr ? 1 : 0));
      let j = 0;
      const samp = (a: Float32Array, xx: number, yy: number) => {
        const xi = Math.max(0, Math.min(N - 2, Math.floor(xx))), yi = Math.max(0, Math.min(N - 2, Math.floor(yy)));
        const fx = Math.max(0, Math.min(1, xx - xi)), fy = Math.max(0, Math.min(1, yy - yi)), p = yi * N + xi;
        return (1 - fy) * ((1 - fx) * a[p] + fx * a[p + 1]) + fy * ((1 - fx) * a[p + N] + fx * a[p + N + 1]);
      };
      const tick = () => {
        const e = M.flowEnds(mask, pred.vx!, pred.vy!, j, samp);
        drawResult(Infinity, e);
        j++;
        if (j <= st.post.steps) anim = requestAnimationFrame(() => setTimeout(tick, 40));
        else { anim = 0; drawResult(); }
      };
      tick();
    } else if (st.route === "distance" && res.level) {
      let lo = Infinity, hi = -Infinity;
      for (const v of res.level) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      let L = lo;
      const tick = () => {
        drawResult(L);
        L += (hi - lo) / 40;
        if (L <= hi + 1e-9) anim = requestAnimationFrame(() => setTimeout(tick, 40));
        else { anim = 0; drawResult(); }
      };
      tick();
    }
  }

  applyPreset(p0);
}

autoMount(NAME, mount);
