// Frozen embeddings + a linear probe (modern deep learning day, unit attention/): the random-
// forest day's paint–train–look loop, on features that a small network learned WITHOUT
// labels (self-supervised, train_ssl.py), next to raw pixel patches and the filter stack.
// Modes: "similar": click a pixel → cosine similarity of every pixel's embedding to it;
// "paint": paint nucleus / background strokes → a logistic-regression probe trained live →
// probability map (Dice against the truth on the synthetic image); "prompt": a click →
// similarity above a threshold → the connected piece containing the click (the idea of a
// promptable segmenter, as a toy). "Another microscope" sliders change the image AFTER the
// probe was trained: the probe is kept, the embeddings are recomputed.
// Auto-mounts on <div data-demo="embedprobe">. Options: data-preset="raw|learned|few|stack|prompt|shift",
// data-kind="raw|stack|learned" (overrides the preset's embedding at start), data-image="cell".
import { htmlEl, slider, presets, button, autoMount, svgEl, svgPoint } from "../lib/ui";
import { scalarToRGBA, viridis } from "../lib/colour";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { paintSegment } from "../lib/paint";
import { probeScene, W } from "./scene";
import * as M from "./model";

const NAME = "embedprobe";
const G = M.G;
const KINDS: M.Kind[] = ["raw", "stack", "learned"];
const KIND_LABEL: Record<M.Kind, string> = { raw: "raw 9×9 patch", stack: "filter stack", learned: "learned (self-sup.)" };

const STYLE = `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] figure { margin: 0; flex: 0 1 330px; min-width: 200px; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin: 0 0 .15em; }
[data-demo="${NAME}"] .ep-stage { position: relative; width: 100%; aspect-ratio: 1; }
[data-demo="${NAME}"] .ep-stage canvas, [data-demo="${NAME}"] .ep-stage svg { position: absolute; inset: 0; width: 100%; height: 100%; border-radius: 3px; }
[data-demo="${NAME}"] .ep-stage canvas { image-rendering: pixelated; }
[data-demo="${NAME}"] .ep-stage svg { touch-action: none; cursor: crosshair; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 290px; max-width: 380px; gap: .3em; }
[data-demo="${NAME}"] .ep-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre; line-height: 1.3; }
[data-demo="${NAME}"] .ep-credit { font-size: 12px; color: var(--qb-muted); }
[data-demo="${NAME}"] .qb-btn.nuc.active { background: #ff7f0e; border-color: #ff7f0e; }
[data-demo="${NAME}"] .qb-btn.bgd.active { background: #1f77b4; border-color: #1f77b4; }
.reveal [data-demo="${NAME}"] { font-size: 17px; gap: .6em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .15em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] figure { flex: 0 0 300px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 1 1 500px; max-width: 540px; }
.reveal [data-demo="${NAME}"] .ep-lab { display: none; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; line-height: 1.2; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

type Mode = "similar" | "paint" | "prompt";
type ImgId = "synth" | "cell";
type Stroke = { v: number; pts: [number, number][] };
interface State { img: ImgId; kind: M.Kind; mode: Mode; click: number; thr: number; shift: M.Shift; brush: number; strokes: Stroke[] }
const NOSHIFT: M.Shift = { gain: 1, blur: 0, noise: 0 };

// strokes on the synthetic image (seed 900001), in image pixels
const FEW: Stroke[] = [
  { v: 1, pts: [[0, 0]] }, { v: 1, pts: [[0, 0]] }, { v: 2, pts: [[0, 0]] }, { v: 2, pts: [[0, 0]] }, { v: 2, pts: [[0, 0]] },
];

interface Preset { id: string; label: string; s: Partial<State> }

function mount(host: HTMLElement): void {
  injectStyle();
  const scene = probeScene(900001);
  const synth = Float32Array.from(scene.image, v => Math.round(v));
  const gtruth = M.gridTruth(scene.truth);
  // the few strokes: inside the three largest nuclei (one bright, one dim) and three background spots incl. a debris spot
  const ell = [...scene.ellipses].map((e, i) => ({ e, lev: meanIn(i + 1) })).sort((a, b) => b.lev - a.lev);
  function meanIn(l: number): number {
    let s = 0, c = 0;
    const lab = scene.truth;
    for (let i = 0; i < W * W; i++) if (lab[i] && inside(scene.ellipses[l - 1], i)) { s += synth[i]; c++; }
    return c ? s / c : 0;
  }
  function inside(e: { x: number; y: number; a: number; b: number; angle: number }, i: number): boolean {
    const x = i % W - e.x, y = Math.floor(i / W) - e.y, c = Math.cos(e.angle), s = Math.sin(e.angle);
    return ((c * x + s * y) / e.a) ** 2 + ((-s * x + c * y) / e.b) ** 2 < 1;
  }
  const bright = ell[0].e, dim = ell[ell.length - 1].e;
  const bgPts = findBackground();
  FEW[0] = { v: 1, pts: [[bright.x - 3, bright.y], [bright.x + 3, bright.y]] };
  FEW[1] = { v: 1, pts: [[dim.x - 2, dim.y], [dim.x + 2, dim.y]] };
  FEW[2] = { v: 2, pts: [[bgPts[0][0] - 4, bgPts[0][1]], [bgPts[0][0] + 4, bgPts[0][1]]] };
  FEW[3] = { v: 2, pts: [[bgPts[1][0] - 4, bgPts[1][1]], [bgPts[1][0] + 4, bgPts[1][1]]] };
  FEW[4] = { v: 2, pts: [[bgPts[2][0] - 4, bgPts[2][1]], [bgPts[2][0] + 4, bgPts[2][1]]] };
  function findBackground(): [number, number][] {
    // three background points far from nuclei, spread over the image
    const out: [number, number][] = [];
    for (const [x, y] of [[20, 20], [108, 20], [20, 108], [108, 108], [64, 64], [64, 20], [20, 64], [108, 64], [64, 108]]) {
      let ok = true;
      for (let dy = -8; dy <= 8 && ok; dy++) for (let dx = -8; dx <= 8 && ok; dx++) if (scene.truth[(y + dy) * W + x + dx]) ok = false;
      if (ok) out.push([x, y]);
      if (out.length === 3) break;
    }
    while (out.length < 3) out.push([4, 4 + 10 * out.length]);
    return out;
  }
  const brightQ = Math.round(bright.y / 2) * G + Math.round(bright.x / 2);

  const PRESETS: Preset[] = [
    { id: "raw", label: "raw pixels: similarity = brightness", s: { kind: "raw", mode: "similar", click: brightQ } },
    { id: "learned", label: "learned features: similarity", s: { kind: "learned", mode: "similar", click: brightQ } },
    { id: "few", label: "five strokes, learned features", s: { kind: "learned", mode: "paint", strokes: FEW } },
    { id: "stack", label: "five strokes, filter stack", s: { kind: "stack", mode: "paint", strokes: FEW } },
    { id: "prompt", label: "a click → a mask", s: { kind: "stack", mode: "prompt", click: brightQ, thr: 0.5 } },
    { id: "shift", label: "new microscope: probe kept", s: { kind: "learned", mode: "paint", strokes: FEW, shift: { gain: 0.5, blur: 1.5, noise: 8 } } },
  ];
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  let first = true;
  const st: State = { img: "synth", kind: "learned", mode: "similar", click: brightQ, thr: 0.6, shift: { ...NOSHIFT }, brush: 2, strokes: [] };
  let cellImg: Float32Array | null = null;
  const embCache = new Map<string, M.Embedding>();

  const mk = (cap: string) => {
    const f = htmlEl("figure", "", "", host);
    const c = htmlEl("figcaption", "", cap, f);
    const stg = htmlEl("div", "ep-stage", "", f);
    const cv = htmlEl("canvas", "", "", stg);
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${W}` }, stg);
    return { f, c, cv, svg };
  };
  const pIn = mk("image"), pOut = mk("result");
  pIn.cv.width = W; pIn.cv.height = W; pOut.cv.width = G; pOut.cv.height = G;
  const credit = htmlEl("div", "ep-credit", "", pIn.f);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ep-lab", "presets", panel);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => applyPreset(p) })));
  htmlEl("div", "ep-lab", "image · embedding", panel);
  const r1 = htmlEl("div", "qb-presets", "", panel);
  const imgBtns = (["synth", "cell"] as ImgId[]).map(k => button(r1, k === "synth" ? "synthetic" : "micrograph", () => setImg(k)));
  const r2 = htmlEl("div", "qb-presets", "", panel);
  const kindBtns = KINDS.map(k => button(r2, KIND_LABEL[k], () => { st.kind = k; run(); }));
  htmlEl("div", "ep-lab", "mode", panel);
  const r3 = htmlEl("div", "qb-presets", "", panel);
  const modeBtns = (["similar", "paint", "prompt"] as Mode[]).map(m => button(r3, m === "similar" ? "click: similarity" : m === "paint" ? "paint + probe" : "click: prompt", () => { st.mode = m; run(); }));
  const r4 = htmlEl("div", "qb-presets", "", panel);
  let brushVal = 1;
  const nucB = button(r4, "nucleus", () => { brushVal = 1; brushBtns(); }, "nuc");
  const bgB = button(r4, "background", () => { brushVal = 2; brushBtns(); }, "bgd");
  button(r4, "clear", () => { st.strokes = []; run(); });
  const brushBtns = () => { nucB.classList.toggle("active", brushVal === 1); bgB.classList.toggle("active", brushVal === 2); };
  brushBtns();
  const thrS = slider(panel, "prompt threshold", 0, 1, 0.02, st.thr, v => { st.thr = v; draw(); });
  htmlEl("div", "ep-lab", "another microscope (probe kept)", panel);
  const gS = slider(panel, "brightness ×", 0.3, 2, 0.05, 1, v => { st.shift.gain = v; run(); });
  const bS = slider(panel, "blur σ", 0, 3, 0.25, 0, v => { st.shift.blur = v; run(); });
  const nS = slider(panel, "noise", 0, 30, 1, 0, v => { st.shift.noise = v; run(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "The learned encoder is tiny (7 k weights) and was trained without labels on synthetic images and this micrograph. Its output is shown on a 64 × 64 grid.", panel);
  button(panel, "reset", () => applyPreset(p0));

  // painting / clicking
  let drawing: Stroke | null = null;
  pIn.svg.addEventListener("pointerdown", ev => {
    const [x, y] = svgPoint(pIn.svg, ev);
    if (st.mode === "paint") {
      pIn.svg.setPointerCapture(ev.pointerId);
      drawing = { v: brushVal, pts: [[x, y]] }; st.strokes.push(drawing); drawStrokes();
    } else {
      st.click = Math.max(0, Math.min(G - 1, Math.floor(y / 2))) * G + Math.max(0, Math.min(G - 1, Math.floor(x / 2)));
      draw();
    }
  });
  pIn.svg.addEventListener("pointermove", ev => { if (drawing) { drawing.pts.push(svgPoint(pIn.svg, ev)); drawStrokes(); } });
  const end = () => { if (drawing) { drawing = null; draw(); } };
  pIn.svg.addEventListener("pointerup", end); pIn.svg.addEventListener("pointercancel", end);

  function setImg(k: ImgId): void {
    st.img = k;
    if (k === "cell" && !cellImg) { loadCell(W).then(g => { cellImg = Float32Array.from(g.data, v => Math.round(v)); run(); }).catch(() => { st.img = "synth"; run(); }); return; }
    run();
  }
  function applyPreset(p: Preset): void {
    Object.assign(st, { img: "synth", kind: "learned", mode: "similar", click: brightQ, thr: 0.6, shift: { ...NOSHIFT }, strokes: [] }, p.s);
    if (first && KINDS.includes(host.dataset.kind as M.Kind)) st.kind = host.dataset.kind as M.Kind;
    if (first && host.dataset.image === "cell") { st.img = "cell"; st.click = 64 * G / 2 + G / 2; }
    first = false;
    st.strokes = st.strokes.map(s => ({ v: s.v, pts: s.pts.map(q => [q[0], q[1]] as [number, number]) }));
    st.shift = { ...st.shift };
    thrS.set(st.thr); gS.set(st.shift.gain); bS.set(st.shift.blur); nS.set(st.shift.noise);
    setImg(st.img);
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
  }

  const baseImg = () => (st.img === "cell" && cellImg ? cellImg : synth);
  function emb(img: Float32Array, shiftKey: string): M.Embedding {
    const key = `${st.img}|${st.kind}|${shiftKey}`;
    let e = embCache.get(key);
    if (!e) { e = M.embedding(img, st.kind); if (embCache.size > 12) embCache.clear(); embCache.set(key, e); }
    return e;
  }
  let shownImg: Float32Array = synth, E0: M.Embedding, E: M.Embedding;
  function run(): void {
    const b = baseImg();
    const sk = `${st.shift.gain}|${st.shift.blur}|${st.shift.noise}`;
    shownImg = sk === "1|0|0" ? b : M.shifted(b, st.shift);
    E0 = emb(b, "1|0|0");
    E = emb(shownImg, sk);
    imgBtns.forEach((x, i) => x.classList.toggle("active", ["synth", "cell"][i] === st.img));
    kindBtns.forEach((x, i) => x.classList.toggle("active", KINDS[i] === st.kind));
    modeBtns.forEach((x, i) => x.classList.toggle("active", ["similar", "paint", "prompt"][i] === st.mode));
    r4.style.display = st.mode === "paint" ? "" : "none";
    (thrS.input.parentElement as HTMLElement).style.display = st.mode === "prompt" ? "" : "none";
    credit.textContent = st.img === "cell" ? CELL_CREDIT : "synthetic: nuclei of any brightness, bright debris, fibres";
    draw();
  }

  function labelledPositions(): { pos: number[]; y: number[] } {
    const map = new Uint8Array(W * W);
    for (const s of st.strokes) for (let k = 0; k < s.pts.length; k++) {
      const a = s.pts[Math.max(0, k - 1)], b = s.pts[k];
      paintSegment(map, W, W, a[0], a[1], b[0], b[1], st.brush, s.v);
    }
    const pos: number[] = [], y: number[] = [];
    for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) {
      const v = map[2 * gy * W + 2 * gx];
      if (v) { pos.push(gy * G + gx); y.push(v === 1 ? 1 : 0); }
    }
    return { pos, y };
  }

  function drawStrokes(): void {
    const svg = pIn.svg;
    svg.innerHTML = "";
    if (st.mode === "paint") for (const s of st.strokes) {
      svgEl("polyline", { points: s.pts.map(p => p.join(",")).join(" "), fill: "none", stroke: s.v === 1 ? "#ff7f0e" : "#1f77b4", "stroke-width": 2 * st.brush + 1, "stroke-linecap": "round", "stroke-linejoin": "round", opacity: 0.9 }, svg);
    }
    else if (st.click >= 0) {
      const x = (st.click % G) * 2 + 1, y = Math.floor(st.click / G) * 2 + 1;
      svgEl("circle", { cx: x, cy: y, r: 3, fill: "none", stroke: "#ff3b30", "stroke-width": 1.5 }, svg);
    }
  }

  function draw(): void {
    const id = new ImageData(W, W);
    for (let i = 0; i < W * W; i++) { const v = shownImg[i]; id.data[4 * i] = v; id.data[4 * i + 1] = v; id.data[4 * i + 2] = v; id.data[4 * i + 3] = 255; }
    pIn.cv.getContext("2d")!.putImageData(id, 0, 0);
    drawStrokes();
    let out: Float32Array, lo = 0, hi = 1, text = "", cap = "";
    const truthOK = st.img === "synth";
    if (st.mode === "similar") {
      out = M.similarity(E, st.click); lo = -1; hi = 1;
      cap = "similarity to the clicked pixel";
      if (truthOK) {
        let sn = 0, cn = 0, sb = 0, cb = 0;
        for (let p = 0; p < G * G; p++) if (gtruth[p]) { sn += out[p]; cn++; } else { sb += out[p]; cb++; }
        text = `mean similarity: nuclei ${(sn / cn).toFixed(2)}, elsewhere ${(sb / cb).toFixed(2)}`;
      }
    } else if (st.mode === "paint") {
      const { pos, y } = labelledPositions();
      const probe = M.trainProbe(E0, pos, y);
      out = pos.length ? M.applyProbe(E, probe) : new Float32Array(G * G).fill(0.5);
      cap = "probe: P(nucleus)";
      text = `${pos.length} labelled grid pixels (${y.filter(v => v).length} nucleus)` + (st.shift.gain !== 1 || st.shift.blur || st.shift.noise ? "\nprobe trained on the original image" : "");
      if (truthOK && pos.length) text += `\nDice against the truth: ${M.dice(Array.from(out, v => (v > 0.5 ? 1 : 0)), gtruth).toFixed(2)}`;
    } else {
      const sim = M.similarity(E, st.click), m = M.promptMask(sim, st.click, st.thr);
      out = Float32Array.from(m); cap = "prompt → mask";
      const area = m.reduce((a, b) => a + b, 0);
      text = `mask: ${area} grid pixels`;
      if (truthOK) {
        // Dice against the true nucleus under the click
        const lab = M.gridTruth(scene.truth);
        text += gtruth[st.click] ? `\nDice against that nucleus: ${M.dice(m, nucleusAt(st.click)).toFixed(2)}` : "\n(click is not on a nucleus)";
        void lab;
      }
    }
    pOut.c.textContent = cap;
    const rgba = scalarToRGBA(out, { map: viridis, lo, hi });
    pOut.cv.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(rgba), G, G), 0, 0);
    ro.textContent = `${KIND_LABEL[st.kind]} (${E.d}-D)\n` + text;
  }

  function nucleusAt(q: number): Uint8Array {
    const x = (q % G) * 2, y = Math.floor(q / G) * 2;
    const k = scene.ellipses.findIndex(e => inside(e, y * W + x));
    const out = new Uint8Array(G * G);
    if (k < 0) return out;
    for (let p = 0; p < G * G; p++) if (inside(scene.ellipses[k], 2 * Math.floor(p / G) * W + 2 * (p % G))) out[p] = 1;
    return out;
  }

  applyPreset(p0);
}

autoMount(NAME, mount);
