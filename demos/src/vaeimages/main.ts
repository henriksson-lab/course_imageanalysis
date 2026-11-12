// Variational autoencoders on images: encode, decode, walk, sample. Two tabs, both with fixed,
// pre-trained weights (no training in the browser), the same network shape (784 → 192 → 2 → 192 → 784):
//
// "Digits": a 2D VAE trained on MNIST (train_mnist.py, int8 weights). Adapted from the digits
//   tab of the vaegen demo of the 3MB00A course.
// "Cell shapes": a 2D VAE trained on SYNTHETIC cell masks (train_shapes.py): round, elongated,
//   dividing, with a protrusion. Includes three anomalies (ring, two cells, square) as examples.
//
// Left: the latent space with the encoder means of 1000 test images, coloured by class, and a
// draggable z (decoded on the right). Draw an image (or load an example) and Encode: the cross
// is the cloud centre, the ellipse ±2 sd. Noise can be added to the encoder input (denoising);
// the reconstruction error is shown (anomaly detection). The strip shows a walk z → z′, draws
// from the encoder's cloud, or draws from the prior N(0, I).
//
// <div data-demo="vaeimages"
//      data-tabs="digits shapes"   which tabs to offer (the first one is shown first)
//      data-seed="1"></div>
import { autoMount, button, checkbox, htmlEl, presets, rng, scale, slider, svgEl, svgPoint, setAttrs, draggable } from "../lib/ui";
import { MNIST_POINTS } from "./mnist_weights";
import { SHAPE_POINTS, SHAPE_KINDS, SHAPE_EXAMPLES } from "./shape_weights";
import { DIGIT_EXAMPLES } from "./digit_examples";
import { digitVae, shapeVae, preprocessDigit, preprocessShape, fromB64, mse, type ImageVae, type V2 } from "./imagevae";

const NAME = "vaeimages";
const DIGIT_COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];
const SHAPE_COL = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"];
const L = 4.2;            // latent plot shows [-L, L]²
const LS = 440;           // latent plot size (px)
const STRIP = 7;

const CSS = `
[data-demo="${NAME}"].qb-demo { display: block; }
[data-demo="${NAME}"] .vg-tabs { display: flex; gap: .4em; margin-bottom: 8px; }
[data-demo="${NAME}"] .vg-tabs .qb-btn { font-size: 18px; padding: .2em .9em; }
[data-demo="${NAME}"] .vg-pane { display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-start; }
[data-demo="${NAME}"] .vg-left { flex: 0 0 ${LS}px; width: ${LS}px; max-width: 100%; }
[data-demo="${NAME}"] .vg-right { flex: 1 1 520px; max-width: 680px; min-width: 0; display: flex; flex-direction: column; gap: 6px; }
[data-demo="${NAME}"] .vg-row { display: flex; flex-wrap: wrap; gap: .35em; align-items: center; }
[data-demo="${NAME}"] .vg-lab { font-size: 17px; color: var(--qb-muted); }
[data-demo="${NAME}"] .qb-btn { font-size: 17px; padding: .14em .5em; }
[data-demo="${NAME}"] .qb-slider { font-size: 17px; flex: 1 1 240px; grid-template-columns: auto 1fr 2.6em; }
[data-demo="${NAME}"] .qb-check { font-size: 17px; }
[data-demo="${NAME}"] .qb-note { font-size: 17px; min-height: 1.3em; }
[data-demo="${NAME}"] svg text { font-size: 18px; }
[data-demo="${NAME}"] svg .ttl { font-weight: 600; }
[data-demo="${NAME}"] svg .mut { fill: var(--qb-muted); }
[data-demo="${NAME}"] svg .frame { fill: none; stroke: var(--qb-grid); stroke-width: 1.5; }
[data-demo="${NAME}"] svg .halo { stroke: #fff; stroke-width: 4; paint-order: stroke; stroke-linejoin: round; }
[data-demo="${NAME}"] .vg-imgs { display: flex; gap: 14px; align-items: flex-start; flex-wrap: wrap; }
[data-demo="${NAME}"] .vg-imgs figure { margin: 0; display: flex; flex-direction: column; gap: 2px; }
[data-demo="${NAME}"] .vg-imgs figcaption { font-size: 17px; color: var(--qb-muted); }
[data-demo="${NAME}"] canvas.vg-big { width: 180px; height: 180px; border: 1.5px solid var(--qb-grid); border-radius: 6px; image-rendering: pixelated; }
[data-demo="${NAME}"] canvas.vg-draw { width: 180px; height: 180px; border: 2px dashed #9aa; border-radius: 6px; touch-action: none; cursor: crosshair; background: #fff; }
[data-demo="${NAME}"] canvas.vg-small { width: 84px; height: 84px; border: 1px solid var(--qb-grid); border-radius: 4px; image-rendering: pixelated; }
[data-demo="${NAME}"] .vg-strip { display: flex; gap: 6px; }
[data-demo="${NAME}"] .vg-strip canvas { width: 62px; height: 62px; border: 1px solid var(--qb-grid); border-radius: 4px; image-rendering: pixelated; }
.reveal [data-demo="${NAME}"].qb-demo { margin: 0; }
.reveal [data-demo="${NAME}"] .vg-pane { flex-wrap: nowrap; }
`;

function injectCss(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = CSS;
  document.head.appendChild(s);
}

const fmt = (z: V2) => `(${z[0].toFixed(2)}, ${z[1].toFixed(2)})`;
const lerp = (a: V2, b: V2, t: number): V2 => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
function place(g: SVGGraphicsElement, x: number, y: number): void { g.setAttribute("transform", `translate(${x.toFixed(1)},${y.toFixed(1)})`); }

/** Paint a 28×28 image (0 = white, 1 = black) into a canvas of the same size. */
function paint(cv: HTMLCanvasElement, img: ArrayLike<number>): void {
  const ctx = cv.getContext("2d")!;
  const id = ctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = Math.round(255 * (1 - Math.max(0, Math.min(1, img[i]))));
    id.data[4 * i] = id.data[4 * i + 1] = id.data[4 * i + 2] = v; id.data[4 * i + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}
function canvas28(parent: Element, cls: string): HTMLCanvasElement {
  const c = htmlEl("canvas", cls, "", parent);
  c.width = 28; c.height = 28;
  return c;
}

interface Model {
  vae: ImageVae;
  points: [number, number, number][];
  classNames: string[];        // label drawn at each class centre
  colours: string[];
  examples: { label: string; img: string }[];
  preprocess: (ink: Float32Array, D: number) => Float32Array | null;
  brush: number;
  what: string;                // "a digit", "a cell"
  presets: (centre: V2[]) => { label: string; z: V2; z2?: V2 }[];
  start: (centre: V2[]) => [V2, V2];
}

const MODELS: Record<string, Model> = {
  digits: {
    vae: digitVae, points: MNIST_POINTS, classNames: Array.from({ length: 10 }, (_, d) => String(d)), colours: DIGIT_COL,
    examples: DIGIT_EXAMPLES, preprocess: preprocessDigit, brush: 20, what: "a digit",
    start: c => [[...c[3]], [...c[8]]],
    presets: c => [
      { label: "a 1", z: c[1] }, { label: "a 7", z: c[7] },
      { label: "walk 1 → 7", z: c[1], z2: c[7] }, { label: "walk 0 → 1", z: c[0], z2: c[1] },
      { label: "far outside", z: [3.3, -3.3] },
    ],
  },
  shapes: {
    vae: shapeVae, points: SHAPE_POINTS, classNames: SHAPE_KINDS, colours: SHAPE_COL,
    examples: SHAPE_EXAMPLES, preprocess: preprocessShape, brush: 34, what: "a cell (fill it in)",
    start: c => [[...c[0]], [...c[1]]],
    presets: c => [
      { label: "round", z: c[0] }, { label: "elongated", z: c[1] }, { label: "dividing", z: c[2] },
      { label: "walk round → dividing", z: c[0], z2: c[2] },
      { label: "far outside", z: [3.3, 3.3] },
    ],
  },
};

function mountModel(pane: HTMLElement, M: Model, seed0: number): void {
  const nCls = M.classNames.length;
  const centre: V2[] = Array.from({ length: nCls }, (_, d) => {
    const p = M.points.filter(q => q[2] === d);
    const med = (k: 0 | 1) => p.map(q => q[k]).sort((a, b) => a - b)[Math.floor(p.length / 2)];
    return [med(0), med(1)];
  });
  let [z, z2] = M.start(centre);
  let walk = false, showGrid = false;
  let code: { mu: V2; sd: V2 } | null = null;
  let clean: Float32Array | null = null;      // the encoder input before noise
  let noise = 0, exIdx = -1, fromExample = false;
  let strip: "none" | "cloud" | "walk" | "prior" = "none";
  let seed = seed0;

  // ---- left: latent space ----
  const left = htmlEl("div", "vg-left", "", pane);
  const clampL = (v: number) => Math.max(-L + 0.05, Math.min(L - 0.05, v));
  const svg = svgEl("svg", { viewBox: `0 0 ${LS} ${LS}`, role: "img", "aria-label": "Latent space" }, left);
  const sx = scale(-L, L, 8, LS - 8), sy = scale(-L, L, LS - 8, 8);
  svgEl("rect", { x: 1, y: 1, width: LS - 2, height: LS - 2, rx: 6, class: "frame" }, svg);
  const under = svgEl("g", {}, svg);
  svgEl("line", { x1: sx(0), x2: sx(0), y1: 8, y2: LS - 8, class: "grid" }, svg);
  svgEl("line", { y1: sy(0), y2: sy(0), x1: 8, x2: LS - 8, class: "grid" }, svg);
  const priorRing = svgEl("circle", { cx: sx(0), cy: sy(0), r: sx(2) - sx(0), fill: "none", stroke: "#888", "stroke-width": 1.5, "stroke-dasharray": "5 5" }, svg);
  const pts = svgEl("g", {}, svg);
  const over = svgEl("g", {}, svg);
  const handles = svgEl("g", {}, svg);
  svgEl("text", { x: 14, y: LS - 14, class: "mut halo" }, svg).textContent = "z₁ →";
  svgEl("text", { x: 14, y: 26, class: "mut halo" }, svg).textContent = "↑ z₂";
  const ringLab = svgEl("text", { x: sx(0) + 4, y: sy(2) - 6, class: "mut halo", style: "font-size:15px" }, svg);
  ringLab.textContent = "prior: radius 2";
  const toWorld = (ev: { clientX: number; clientY: number }): V2 => { const [x, y] = svgPoint(svg, ev); return [clampL(sx.invert(x)), clampL(sy.invert(y))]; };
  const handle = (stroke: string, r: number, text: string) => {
    const g = svgEl("g", {}, handles);
    svgEl("circle", { r: r + 10, fill: "transparent" }, g);
    svgEl("circle", { r, fill: "#fff", stroke, "stroke-width": 3.5 }, g);
    const t = svgEl("text", { x: r + 4, y: -r, class: "ttl halo" }, g);
    t.textContent = text;
    return g;
  };
  const gridImg = svgEl("image", { x: sx(-4), y: sy(4), width: sx(4) - sx(-4), height: sy(-4) - sy(4), preserveAspectRatio: "none" }, under);
  for (const [a, b, d] of M.points)
    svgEl("circle", { cx: sx(a).toFixed(1), cy: sy(b).toFixed(1), r: 2.2, fill: M.colours[d], "fill-opacity": 0.45 }, pts);
  centre.forEach((c, d) => {
    const t = svgEl("text", { x: sx(c[0]), y: sy(c[1]) + 8, "text-anchor": "middle", class: "ttl halo", fill: M.colours[d], style: nCls > 4 ? "font-size:24px" : "font-size:19px" }, pts);
    t.textContent = M.classNames[d];
  });
  const walkLine = svgEl("line", { stroke: "#333", "stroke-width": 2, "stroke-dasharray": "6 4" }, over);
  const stripDots = svgEl("g", {}, over);
  const cloud = svgEl("ellipse", { fill: "#f4a261", "fill-opacity": 0.3, stroke: "#c4621a", "stroke-width": 2.5 }, over);
  const codeMark = svgEl("path", { d: "M-8,-8L8,8M-8,8L8,-8", stroke: "#c4621a", "stroke-width": 3.5 }, over);
  const h1 = handle("#111", 11, "z");
  const h2 = handle("#c4621a", 9, "z′");

  // ---- right ----
  const right = htmlEl("div", "vg-right", "", pane);
  const imgs = htmlEl("div", "vg-imgs", "", right);
  const f1 = htmlEl("figure", "", "", imgs);
  const drawCap = htmlEl("figcaption", "", `draw ${M.what} here`, f1);
  const drawCv = htmlEl("canvas", "vg-draw", "", f1);
  const D = 196;
  drawCv.width = D; drawCv.height = D;
  const f2 = htmlEl("figure", "", "", imgs);
  htmlEl("figcaption", "", "decoded from z", f2);
  const bigCv = canvas28(f2, "vg-big");
  const f3 = htmlEl("figure", "", "", imgs);
  htmlEl("figcaption", "", "encoder input", f3);
  const inCv = canvas28(f3, "vg-small");
  const zLab = htmlEl("figcaption", "", "", f3);
  const errLab = htmlEl("figcaption", "", "", f3);
  const stripLab = htmlEl("div", "vg-lab", "", right);
  const stripRow = htmlEl("div", "vg-strip", "", right);
  const stripCv = Array.from({ length: STRIP }, () => canvas28(stripRow, ""));

  const r1 = htmlEl("div", "vg-row", "", right);
  button(r1, "Clear", () => { clearDraw(); code = null; clean = null; fromExample = false; if (strip === "cloud") strip = "none"; draw(); });
  button(r1, "Encode", () => doEncode());
  button(r1, "example ▸", () => { exIdx = (exIdx + 1) % M.examples.length; loadExample(exIdx); });
  const sNoise = slider(r1, "noise on input", 0, 0.6, 0.05, 0, v => { noise = v; if (clean) encodeImage(clean); }, v => v.toFixed(2));
  const r1b = htmlEl("div", "vg-row", "", right);
  button(r1b, "sample from the cloud", () => {
    if (!code) { note.textContent = "Encode an image first (draw one, or press “example ▸”)."; return; }
    seed++; strip = "cloud"; walk = false; walkBox.checked = false; draw();
  });
  button(r1b, "sample from the prior", () => { seed++; strip = "prior"; walk = false; walkBox.checked = false; draw(); });
  const walkBox = checkbox(r1b, "walk z → z′", false, v => { walk = v; strip = v ? "walk" : "none"; draw(); });
  checkbox(r1b, "decoded grid", false, v => { showGrid = v; draw(); });
  const r2 = htmlEl("div", "vg-row", "", right);
  htmlEl("span", "vg-lab", "Presets:", r2);
  presets(r2, [
    ...M.presets(centre).map(p => ({ label: p.label, apply: () => {
      z = [...p.z];
      if (p.z2) { z2 = [...p.z2]; walk = true; walkBox.checked = true; strip = "walk"; }
      else if (strip === "walk") { strip = "none"; walk = false; walkBox.checked = false; }
      draw();
    } })),
    { label: "Reset", apply: () => {
      [z, z2] = M.start(centre); walk = false; walkBox.checked = false; strip = "none"; code = null; clean = null;
      noise = 0; sNoise.set(0); exIdx = -1; seed = seed0; clearDraw(); draw();
    } },
  ]);
  const note = htmlEl("div", "qb-note", "", right);

  // ---- drawing ----
  const dctx = drawCv.getContext("2d")!;
  function clearDraw(): void {
    dctx.fillStyle = "#fff"; dctx.fillRect(0, 0, D, D); paint(inCv, new Float32Array(784));
    zLab.textContent = ""; errLab.textContent = ""; drawCap.textContent = `draw ${M.what} here`;
  }
  clearDraw();
  let last: V2 | null = null;
  const pos = (ev: PointerEvent): V2 => {
    const b = drawCv.getBoundingClientRect();
    return [(ev.clientX - b.left) * D / b.width, (ev.clientY - b.top) * D / b.height];
  };
  drawCv.addEventListener("pointerdown", ev => {
    ev.preventDefault();
    try { drawCv.setPointerCapture(ev.pointerId); } catch { /* synthetic events */ }
    if (fromExample) { clearDraw(); fromExample = false; }
    last = pos(ev); stroke(last, last);
  });
  drawCv.addEventListener("pointermove", ev => { if (!last) return; const p = pos(ev); stroke(last, p); last = p; });
  const end = () => { last = null; };
  drawCv.addEventListener("pointerup", end);
  drawCv.addEventListener("pointercancel", end);
  function stroke(a: V2, b: V2): void {
    Object.assign(dctx, { strokeStyle: "#000", lineWidth: M.brush, lineCap: "round", lineJoin: "round", shadowColor: "#000", shadowBlur: 3 });
    dctx.beginPath(); dctx.moveTo(a[0], a[1]); dctx.lineTo(b[0], b[1]); dctx.stroke();
  }
  function loadExample(k: number): void {
    const img = fromB64(M.examples[k].img);
    const tmp = document.createElement("canvas"); tmp.width = tmp.height = 28; paint(tmp, img);
    dctx.imageSmoothingEnabled = false;
    dctx.drawImage(tmp, 0, 0, D, D);
    drawCap.textContent = `example ${k + 1}/${M.examples.length}: ${M.examples[k].label}`;
    fromExample = true;
    encodeImage(img);
  }
  function doEncode(): void {
    if (fromExample && clean) { encodeImage(clean); return; }
    const px = dctx.getImageData(0, 0, D, D).data, ink = new Float32Array(D * D);
    for (let i = 0; i < D * D; i++) ink[i] = 1 - px[4 * i] / 255;
    const img = M.preprocess(ink, D);
    if (!img) { note.textContent = "Nothing drawn yet: draw in the dashed box first."; return; }
    encodeImage(img);
  }
  /** Encode `img` (with noise added if the slider is above 0) and show where it lands. */
  function encodeImage(img: Float32Array): void {
    clean = img;
    let inp = img;
    if (noise > 0) {
      const r = rng(1000 + seed0);
      inp = img.map(v => Math.max(0, Math.min(1, v + noise * r.norm())));
    }
    paint(inCv, inp);
    code = M.vae.encode(inp);
    z = [clampL(code.mu[0]), clampL(code.mu[1])];
    zLab.textContent = `→ ${fmt(code.mu)}`;
    errLab.textContent = `error ${mse(img, M.vae.decode(code.mu)).toFixed(4)}`;
    strip = walk ? "walk" : strip === "cloud" ? "cloud" : strip === "prior" ? "prior" : "none";
    draw();
  }

  draggable(h1, toWorld, (x, y) => { z = [x, y]; draw(); });
  draggable(h2, toWorld, (x, y) => { z2 = [x, y]; draw(); });
  svg.addEventListener("click", ev => {
    if (h1.contains(ev.target as Node) || h2.contains(ev.target as Node)) return;
    z = toWorld(ev); draw();
  });

  let gridDone = false;
  function renderGrid(): void {
    const n = 16, c = document.createElement("canvas");
    c.width = c.height = 28 * n;
    const ctx = c.getContext("2d")!;
    const tmp = document.createElement("canvas"); tmp.width = tmp.height = 28;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      paint(tmp, M.vae.decode([-4 + 8 * (j + 0.5) / n, 4 - 8 * (i + 0.5) / n]));
      ctx.drawImage(tmp, 28 * j, 28 * i);
    }
    gridImg.setAttribute("href", c.toDataURL());
    gridDone = true;
  }

  function draw(): void {
    if (showGrid && !gridDone) renderGrid();
    gridImg.style.display = showGrid ? "" : "none";
    pts.style.opacity = showGrid ? "0.25" : "1";
    place(h1, sx(z[0]), sy(z[1]));
    h2.style.display = walkLine.style.display = walk ? "" : "none";
    place(h2, sx(z2[0]), sy(z2[1]));
    setAttrs(walkLine, { x1: sx(z[0]), y1: sy(z[1]), x2: sx(z2[0]), y2: sy(z2[1]) });
    stripDots.replaceChildren();
    cloud.style.display = codeMark.style.display = code ? "" : "none";
    if (code) {
      place(codeMark, sx(code.mu[0]), sy(code.mu[1]));
      setAttrs(cloud, { cx: sx(code.mu[0]), cy: sy(code.mu[1]),
        rx: Math.max(3, sx(2 * code.sd[0]) - sx(0)).toFixed(1), ry: Math.max(3, sx(2 * code.sd[1]) - sx(0)).toFixed(1) });
    }
    paint(bigCv, M.vae.decode(z));
    let zs: V2[] = [];
    if (strip === "walk" && walk) {
      zs = Array.from({ length: STRIP }, (_, s) => lerp(z, z2, s / (STRIP - 1)));
      zs.forEach(p => svgEl("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 4.5, fill: "#333" }, stripDots));
      stripLab.textContent = `Walk: ${STRIP} points from z ${fmt(z)} to z′ ${fmt(z2)}, decoded`;
    } else if (strip === "cloud" && code) {
      const r = rng(seed), c = code;
      zs = Array.from({ length: STRIP }, () => [c.mu[0] + c.sd[0] * r.norm(), c.mu[1] + c.sd[1] * r.norm()] as V2);
      zs.forEach(p => svgEl("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 3, fill: "#c4621a" }, stripDots));
      stripLab.textContent = `${STRIP} draws z = μ + σε from the encoder's cloud (σ = ${c.sd[0].toFixed(2)}, ${c.sd[1].toFixed(2)}), decoded`;
    } else if (strip === "prior") {
      const r = rng(seed);
      zs = Array.from({ length: STRIP }, () => [r.norm(), r.norm()] as V2);
      zs.forEach(p => svgEl("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 4, fill: "#2a6f97", stroke: "#fff", "stroke-width": 1 }, stripDots));
      stripLab.textContent = `${STRIP} new images: z drawn from the prior N(0, I), decoded`;
    } else stripLab.textContent = "Strip: tick “walk”, or press a “sample” button.";
    stripCv.forEach((cv, k) => paint(cv, zs[k] ? M.vae.decode(zs[k]) : new Float32Array(784)));
    note.textContent = `z = ${fmt(z)}. Drag z or click in the map. Error = mean squared pixel error of the reconstruction.`;
  }
  draw();
}

// ============================================================================================
function mount(host: HTMLElement): void {
  injectCss();
  const tabs = (host.dataset.tabs ?? "digits shapes").split(/\s+/).filter(t => t in MODELS);
  if (!tabs.length) tabs.push("digits");
  const seed = Math.max(1, Math.round(Number(host.dataset.seed ?? 1) || 1));
  const bar = htmlEl("div", "vg-tabs", "", host);
  const panes = new Map<string, HTMLElement>();
  const show = (t: string) => {
    if (!panes.has(t)) {
      const p = htmlEl("div", "vg-pane", "", host);
      panes.set(t, p);
      mountModel(p, MODELS[t], seed);
    }
    panes.forEach((p, k) => { p.style.display = k === t ? "" : "none"; });
  };
  if (tabs.length > 1) {
    const btns = presets(bar, tabs.map(t => ({ label: t === "digits" ? "Digits (MNIST)" : "Cell shapes (synthetic)", apply: () => show(t) })));
    btns[0].classList.add("active");
  } else bar.style.display = "none";
  show(tabs[0]);
}

autoMount(NAME, mount);
