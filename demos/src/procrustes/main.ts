// Procrustes alignment of two landmark sets, step by step (centre → scale → rotate), with
// the closed-form 2D solution sR = [[a, −b], [b, a]]; and generalised Procrustes analysis
// (GPA) of a training set. Points are draggable; the orange set also has a move handle
// (square, at its centroid) and a rotate/scale handle (ring).
// Auto-mounts on <div data-demo="procrustes">. Options: data-preset="<id>", data-mode="pair|gpa".
import { htmlEl, presets, button, checkbox, autoMount, svgEl } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { rng } from "../lib/random";
import { Shape, procrustes, centroid, family, trainingSet, normalise, gpaStep, sumSq, nPoints, applySim, translateShape } from "../lib/shapes";

const NAME = "procrustes";

// The by-hand triangle and its image under a = 1.2, b = 0.9 (s = 1.5, θ = 36.87°), t = (−1, −1.5)
const TRI = Float64Array.from([0, 0, 2, 0, 0, 1]);
const LSHAPE = Float64Array.from([0, 0, 2, 0, 2, 0.6, 0.6, 0.6, 0.6, 2, 0, 2]);
function leaf12(seed: number, k = 0): Shape {
  const f = family("leaf"), ts = trainingSet(f, { count: k + 1, seed, noise: 0 });
  const s = ts.shapes[k], out = new Float64Array(24);
  for (let i = 0; i < 12; i++) { out[2 * i] = s[4 * i]; out[2 * i + 1] = -s[4 * i + 1]; }   // y up for this demo
  return out;
}

interface Preset { id: string; label: string; mode: "pair" | "gpa"; make?: () => { p: Shape; q: Shape }; reflection?: boolean; scale?: boolean }

function jitter(s: Shape, sd: number, seed: number): Shape {
  const r = rng(seed);
  return s.map(v => v + sd * r.norm());
}

const PRESETS: Preset[] = [
  { id: "exact", label: "pure rotation recovers exactly", mode: "pair", make: () => ({ p: translateShape(TRI, 0.6, 0.5), q: applySim(TRI, 1.2, 0.9, -1, -1.5) }) },
  { id: "noise", label: "noise: best fit is not exact", mode: "pair", make: () => {
    const base = translateShape(leaf12(4).map(v => v * 1.3), -1.6, 0);
    return { p: base, q: jitter(applySim(translateShape(base, 1.6, 0), 0.75, -0.55, 1.2, 0.1), 0.08, 3) };
  } },
  { id: "mirror", label: "mirror image: reflection or not", mode: "pair", make: () => {
    const L = translateShape(LSHAPE, -2.4, -1);
    const m = LSHAPE.map((v, i) => (i % 2 === 0 ? -v : v));
    return { p: L, q: applySim(m, 0.9, 0.3, 2.0, -0.6) };
  } },
  { id: "mislabel", label: "one landmark mislabelled", mode: "pair", make: () => {
    const base = translateShape(leaf12(4).map(v => v * 1.3), -1.6, 0);
    const q = applySim(translateShape(base, 1.6, 0), 0.75, -0.55, 1.2, 0.1);
    // landmark 3 is placed where landmark 9 is (a wrong correspondence)
    q[6] = q[18]; q[7] = q[19];
    return { p: base, q };
  } },
  { id: "rigid", label: "rigid: no scaling allowed", mode: "pair", scale: false, make: () => ({ p: TRI.slice(), q: applySim(TRI, 1.2, 0.9, -1, -1.5) }) },
  { id: "gpa", label: "GPA on a set", mode: "gpa" },
];

const STYLE = `
[data-demo="${NAME}"] { gap: .9em; }
[data-demo="${NAME}"] .pr-plot { flex: 0 1 600px; min-width: 280px; }
[data-demo="${NAME}"] .pr-plot svg { touch-action: none; }
[data-demo="${NAME}"] .pr-cap { font-size: 17px; text-align: center; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 420px; gap: .4em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; }
[data-demo="${NAME}"] .pr-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
[data-demo="${NAME}"] .pr-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .pr-plot { flex-basis: 640px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .5em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const f3 = (v: number) => (Math.abs(v) < 5e-4 ? "0.000" : v.toFixed(3));
const deg = (r: number) => (r * 180 / Math.PI).toFixed(1) + "°";

function rotScaleAbout(s: Shape, cx: number, cy: number, k: number, th: number): Shape {
  const a = k * Math.cos(th), b = k * Math.sin(th), out = new Float64Array(s.length);
  for (let i = 0; i < s.length; i += 2) {
    const x = s[i] - cx, y = s[i + 1] - cy;
    out[i] = cx + a * x - b * y; out[i + 1] = cy + b * x + a * y;
  }
  return out;
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS.find(p => p.mode === host.dataset.mode) ?? PRESETS[0];
  const st = { reflection: false, scale: true, stage: 0 };
  let P: Shape = TRI.slice(), Q: Shape = TRI.slice(), P0: Shape = P.slice();
  let mode: "pair" | "gpa" = "pair";
  // GPA state
  let raw: Shape[] = [], al: Shape[] = [], mean: Shape | null = null, ref: Shape | null = null, gIter = 0, gChange = NaN;
  let anim = 0;

  // ---- layout ----
  const box = htmlEl("div", "pr-plot", "", host);
  const cap = htmlEl("div", "pr-cap", "", box);
  const plot = new Plot(box, { width: 640, height: 470, x: [-3.2, 3.2], y: [-2.3, 2.3], equal: true, xticks: 7, yticks: 5 });
  const g = svgEl("g", {}, plot.over);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "pr-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "pr-lab", "align the orange set onto the blue one", panel);
  const row = htmlEl("div", "pr-row", "", panel);
  const stepBtns = [
    button(row, "1 centre", () => doStage(1)),
    button(row, "2 scale", () => doStage(2)),
    button(row, "3 rotate", () => doStage(3)),
    button(row, "all", () => doStage(3, true)),
    button(row, "undo all", () => { P = P0.slice(); st.stage = 0; draw(); }),
  ];
  const gpaRow = htmlEl("div", "pr-row", "", panel);
  const gpaBtns = [
    button(gpaRow, "GPA step", () => gpaNext()),
    button(gpaRow, "start over", () => gpaReset()),
  ];
  const opts = htmlEl("div", "pr-row", "", panel);
  const reflBox = checkbox(opts, "allow reflection", st.reflection, v => { st.reflection = v; draw(); });
  const scaleBox = checkbox(opts, "allow scaling", st.scale, v => { st.scale = v; draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Drag any point. The orange square moves the whole orange set, the ring rotates and scales it.", panel);

  function load(p: Preset): void {
    mode = p.mode;
    st.reflection = !!p.reflection; st.scale = p.scale !== false; st.stage = 0;
    reflBox.checked = st.reflection; scaleBox.checked = st.scale;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    if (mode === "pair") { const pq = p.make!(); P = pq.p; Q = pq.q; P0 = P.slice(); }
    else gpaReset();
    draw();
  }

  // ---- pair: steps ----
  function fit() { return procrustes(P, Q, { reflection: st.reflection, scale: st.scale }); }

  function animateTo(target: Shape, after: () => void): void {
    const from = P.slice(), t0 = performance.now(), id = ++anim;
    const frame = (t: number) => {
      if (id !== anim) return;
      const u = Math.min(1, (t - t0) / 500), e = u * u * (3 - 2 * u);
      P = from.map((v, i) => v + e * (target[i] - v));
      draw();
      if (u < 1) requestAnimationFrame(frame); else { P = target; after(); draw(); }
    };
    requestAnimationFrame(frame);
  }

  /** Apply the stages up to `k` that have not been applied yet (one at a time, animated). */
  function doStage(k: number, chain = false): void {
    if (mode !== "pair") return;
    const next = st.stage + 1;
    if (next > k) return;
    const [px, py] = centroid(P), [qx, qy] = centroid(Q), f = fit();
    let target: Shape;
    if (next === 1) target = translateShape(P, qx - px, qy - py);
    else if (next === 2) target = rotScaleAbout(P, px, py, st.scale ? f.s : 1, 0);
    else {
      // the rotation (and the reflection, if better) that remains after centring and scaling
      const g2 = procrustes(P, Q, { reflection: st.reflection, scale: false });
      target = g2.aligned;
    }
    animateTo(target, () => { st.stage = next; if (chain && next < k) setTimeout(() => doStage(k, true), 120); });
  }

  // ---- GPA ----
  function gpaReset(): void {
    const ts = trainingSet(family("leaf"), { count: 10, seed: 21, noise: 0.02, rot: 0.6, logScale: 0.25, shift: 0.7 });
    raw = ts.shapes.map(s => { const o = new Float64Array(24); for (let i = 0; i < 12; i++) { o[2 * i] = s[4 * i] * 0.9; o[2 * i + 1] = -s[4 * i + 1] * 0.9; } return o; });
    al = []; mean = null; ref = null; gIter = 0; gChange = NaN;
    draw();
  }
  function gpaNext(): void {
    if (mode !== "gpa") return;
    if (!mean) { al = raw.map(normalise); ref = al[0].slice(); mean = ref.slice(); gIter = 0; gChange = NaN; }
    else { const r = gpaStep(al, mean, ref!); al = r.aligned; gChange = r.change; mean = r.mean; gIter++; }
    draw();
  }

  // ---- drawing ----
  const X = (v: number) => plot.x(v), Y = (v: number) => plot.y(v);
  function poly(s: Shape, col: string, w: number, dash = "", op = 1, fill = "none"): void {
    let d = "";
    for (let i = 0; i < s.length; i += 2) d += (i ? "L" : "M") + X(s[i]).toFixed(1) + "," + Y(s[i + 1]).toFixed(1);
    svgEl("path", { d: d + "Z", fill, "fill-opacity": 0.12, stroke: col, "stroke-width": w, "stroke-dasharray": dash, opacity: op, "stroke-linejoin": "round" }, g);
  }
  function dragPoint(el: SVGElement, onMove: (x: number, y: number) => void): void {
    el.style.cursor = "grab";
    el.addEventListener("pointerdown", ev => {
      ev.preventDefault(); anim++;
      // listeners on window: draw() replaces the dragged element while it moves
      const mv = (e: PointerEvent) => { const [x, y] = plot.toData(e); onMove(x, y); };
      const up = () => { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); window.removeEventListener("pointercancel", up); draw(); };
      window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up); window.addEventListener("pointercancel", up);
    });
  }

  function drawPair(): void {
    const f = fit();
    // residual lines from each orange point to its blue partner
    for (let i = 0; i < P.length; i += 2) svgEl("line", { x1: X(P[i]), y1: Y(P[i + 1]), x2: X(Q[i]), y2: Y(Q[i + 1]), stroke: "var(--qb-muted, #999)", "stroke-width": 1.5, "stroke-dasharray": "4 3" }, g);
    poly(Q, seriesColor(0), 3, "", 1, seriesColor(0));
    poly(P, seriesColor(3), 3, "", 1, seriesColor(3));
    const n = nPoints(P);
    for (let i = 0; i < n; i++) {
      const cq = svgEl("circle", { cx: X(Q[2 * i]), cy: Y(Q[2 * i + 1]), r: 9, fill: seriesColor(0), stroke: "var(--qb-bg, #fff)", "stroke-width": 2 }, g);
      dragPoint(cq, (x, y) => { Q[2 * i] = x; Q[2 * i + 1] = y; st.stage = 0; P0 = P.slice(); clearActive(); draw(); });
      const cp = svgEl("circle", { cx: X(P[2 * i]), cy: Y(P[2 * i + 1]), r: 9, fill: seriesColor(3), stroke: "var(--qb-bg, #fff)", "stroke-width": 2 }, g);
      dragPoint(cp, (x, y) => { P[2 * i] = x; P[2 * i + 1] = y; st.stage = 0; P0 = P.slice(); clearActive(); draw(); });
      if (n <= 12) {
        const t = svgEl("text", { x: X(P[2 * i]) + 10, y: Y(P[2 * i + 1]) - 9, "font-size": 15, class: "qb-halo" }, g); t.textContent = String(i + 1);
        const t2 = svgEl("text", { x: X(Q[2 * i]) + 10, y: Y(Q[2 * i + 1]) - 9, "font-size": 15, class: "qb-halo" }, g); t2.textContent = String(i + 1);
      }
    }
    // whole-set handles for the orange set
    const [cx, cy] = centroid(P);
    let rad = 0;
    for (let i = 0; i < P.length; i += 2) rad = Math.max(rad, Math.hypot(P[i] - cx, P[i + 1] - cy));
    const hx = cx + rad * 0.75, hy = cy;
    svgEl("line", { x1: X(cx), y1: Y(cy), x2: X(hx), y2: Y(hy), stroke: seriesColor(3), "stroke-width": 1.5, opacity: 0.6 }, g);
    const mv = svgEl("rect", { x: X(cx) - 9, y: Y(cy) - 9, width: 18, height: 18, fill: "var(--qb-bg, #fff)", stroke: seriesColor(3), "stroke-width": 3 }, g);
    let base: Shape | null = null;
    dragPoint(mv, (x, y) => { const [ax, ay] = centroid(P); P = translateShape(P, x - ax, y - ay); st.stage = 0; P0 = P.slice(); clearActive(); draw(); });
    const rh = svgEl("circle", { cx: X(hx), cy: Y(hy), r: 10, fill: "var(--qb-bg, #fff)", stroke: seriesColor(3), "stroke-width": 3 }, g);
    rh.addEventListener("pointerdown", () => { base = P.slice(); });
    dragPoint(rh, (x, y) => {
      if (!base) return;
      const k = Math.hypot(x - cx, y - cy) / (rad * 0.75), th = Math.atan2(y - cy, x - cx);
      P = rotScaleAbout(base, cx, cy, Math.max(0.1, k), th);
      st.stage = 0; P0 = P.slice(); clearActive(); draw();
    });

    const now = sumSq(P, Q);
    const M = f.reflect ? `[${f3(-f.a)} ${f3(-f.b)}; ${f3(-f.b)} ${f3(f.a)}]` : `[${f3(f.a)} ${f3(-f.b)}; ${f3(f.b)} ${f3(f.a)}]`;
    ro.textContent = [
      `best fit of orange onto blue:`,
      `a = ${f3(f.a)}   b = ${f3(f.b)}`,
      `scale √(a²+b²) = ${f3(f.s)}`,
      `angle atan2(b, a) = ${deg(f.theta)}${f.reflect ? "  (+ mirror)" : ""}`,
      `sR = ${M}`,
      `residual after the fit: ${f3(f.rss)}`,
      `residual now  Σ|p−q|²: ${f3(now)}`,
    ].join("\n");
    const stages = ["as drawn", "centred", "centred + scaled", "aligned"];
    cap.textContent = `orange onto blue: ${stages[st.stage]}`;
  }

  function drawGpa(): void {
    const n = 12;
    if (!mean) {
      raw.forEach((s, j) => poly(s, j === 0 ? seriesColor(0) : "var(--qb-muted, #888)", j === 0 ? 2.5 : 1.8, "", 0.9));
      cap.textContent = "10 training leaves (12 landmarks each), raw; blue: shape 1";
      ro.textContent = "Press GPA step: centre every shape,\nscale it to size 1, take shape 1 as\nthe first mean.";
      return;
    }
    const k = 1.25 * Math.sqrt(n);                         // display scale for size-1 shapes
    const disp = (s: Shape) => s.map(v => v * k);
    al.forEach(s => poly(disp(s), "var(--qb-muted, #888)", 1.6, "", 0.8));
    // landmark scatter: all shapes' landmark 1 (tip) etc. as small dots
    for (const s of al) for (let i = 0; i < n; i++) svgEl("circle", { cx: X(s[2 * i] * k), cy: Y(s[2 * i + 1] * k), r: 3, fill: seriesColor(i % 5) }, g);
    poly(disp(mean), "var(--qb-fg, #222)", 4);
    const resid = al.reduce((t, s) => t + sumSq(s, mean!), 0);
    cap.textContent = gIter === 0 ? "centred and scaled to size 1; black: mean = shape 1" : `after ${gIter} GPA iteration${gIter > 1 ? "s" : ""}; black: the mean shape`;
    ro.textContent = [
      `iteration ${gIter}`,
      `change of the mean: ${Number.isFinite(gChange) ? gChange.toExponential(2) : "–"}`,
      `Σ residual to the mean: ${resid.toFixed(4)}`,
      `(shapes drawn ×${k.toFixed(1)}: size-1 shapes are small)`,
    ].join("\n");
  }

  function clearActive(): void { presetBtns.forEach(b => b.classList.remove("active")); }

  function draw(): void {
    g.replaceChildren();
    stepBtns.forEach(b => (b.style.display = mode === "pair" ? "" : "none"));
    gpaBtns.forEach(b => (b.style.display = mode === "gpa" ? "" : "none"));
    reflBox.parentElement!.style.display = scaleBox.parentElement!.style.display = mode === "pair" ? "" : "none";
    stepBtns.forEach((b, i) => b.classList.toggle("active", i < 3 && st.stage === i + 1));
    if (mode === "pair") drawPair(); else drawGpa();
  }

  load(p0);
}

autoMount(NAME, mount);
