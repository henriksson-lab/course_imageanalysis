// cnnviewer: a small trained CNN (conv 3×3×8 → pool → conv 3×3×16 → pool → dense → softmax,
// about 9,000 weights; trained offline by train_cnn.py) looking at a 28 × 28 input. Draw or
// pick an input, shift or rotate it, and see the layer-1 kernels, the feature maps of both
// layers, and the class probabilities. Click a layer-2 map to see the 9 test patches that excite
// it most. "Occlusion map": cover each part of the input in turn and watch the probability of
// the predicted class. Weight sets: digits with / without rotation augmentation, training
// snapshots (epochs 0, 1, 3, 10), and cell masks (4 classes).
//
// <div data-demo="cnnviewer" data-weights="digits" data-example="12"></div>
import { autoMount, button, htmlEl, presets as presetRow, slider, svgEl } from "../lib/ui";
import { drawGrey } from "../lib/imageops";
import { bytesFromB64 } from "../lib/q8";
import { runCnn, transform, occlusion, getCnn, type CnnRun } from "./net";
import { CNN_INFO, TOP_PATCHES, SHAPE_EXAMPLES } from "./cnn_weights";
import { DIGIT_EXAMPLES } from "../vaeimages/digit_examples";

const NAME = "cnnviewer";
const STYLE = `
[data-demo="${NAME}"] { gap: .9em; align-items: flex-start; }
[data-demo="${NAME}"] .cv-col { display: flex; flex-direction: column; gap: .35em; min-width: 0; }
[data-demo="${NAME}"] .cv-in { flex: 0 1 250px; }
[data-demo="${NAME}"] .cv-maps { flex: 1 1 380px; }
[data-demo="${NAME}"] .cv-out { flex: 0 1 270px; }
[data-demo="${NAME}"] .cv-pad { width: 100%; aspect-ratio: 1; image-rendering: pixelated; border: 2px solid #999; touch-action: none; cursor: crosshair; background: #000; }
[data-demo="${NAME}"] .cv-grid { display: grid; gap: 3px; }
[data-demo="${NAME}"] .cv-grid canvas { width: 100%; aspect-ratio: 1; image-rendering: pixelated; border: 1px solid var(--qb-grid); }
[data-demo="${NAME}"] .cv-grid canvas.sel { outline: 3px solid #d62728; }
[data-demo="${NAME}"] .cv-h { font-size: 16px; font-weight: 600; }
[data-demo="${NAME}"] .qb-note { font-size: 14px; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .4em; }
.reveal [data-demo="${NAME}"] .cv-in { flex-basis: 220px; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const SETS: { id: string; label: string; kind: "digits" | "shapes" }[] = [
  { id: "digits", label: "digits (10 epochs)", kind: "digits" },
  { id: "digitsAug", label: "digits + rotation augmentation", kind: "digits" },
  { id: "digitsE0", label: "epoch 0 (random)", kind: "digits" },
  { id: "digitsE1", label: "epoch 1", kind: "digits" },
  { id: "digitsE3", label: "epoch 3", kind: "digits" },
  { id: "shapes", label: "cell masks", kind: "shapes" },
];
const SHAPE_NAMES = ["round", "elongated", "dividing", "protrusion"];
const toImg = (b64: string) => Float32Array.from(bytesFromB64(b64), v => v / 255);

function mount(host: HTMLElement): void {
  injectStyle();
  let set = SETS.find(s => s.id === host.dataset.weights) ?? SETS[0];
  let base: Float32Array = toImg(set.kind === "shapes" ? SHAPE_EXAMPLES[0].img : DIGIT_EXAMPLES[Number(host.dataset.example ?? 12)].img);
  let rot = 0, dx = 0, dy = 0, selMap = -1, occ: { map: Float32Array; n: number; k: number } | null = null;

  // ---- input column ----
  const cin = htmlEl("div", "cv-col cv-in", "", host);
  htmlEl("div", "cv-h", "input (draw here)", cin);
  const pad = htmlEl("canvas", "cv-pad", "", cin);
  pad.width = 28; pad.height = 28;
  const exRow = htmlEl("div", "qb-presets", "", cin);
  const sRot = slider(cin, "rotate", -180, 180, 5, 0, v => { rot = v; occ = null; draw(); }, v => `${v}°`);
  const sDx = slider(cin, "shift x", -6, 6, 1, 0, v => { dx = v; occ = null; draw(); });
  const sDy = slider(cin, "shift y", -6, 6, 1, 0, v => { dy = v; occ = null; draw(); });
  const r2 = htmlEl("div", "qb-presets", "", cin);
  button(r2, "clear", () => { base = new Float32Array(784); reset(); });
  button(r2, "occlusion map", () => { const x = current(), r = runCnn(x, set.id), k = argmax(r.probs); occ = { ...occlusion(x, set.id, k), k }; draw(); });
  const credit = htmlEl("div", "qb-note", "", cin);

  // ---- maps column ----
  const cm = htmlEl("div", "cv-col cv-maps", "", host);
  htmlEl("div", "cv-h", "layer 1: 8 learned 3×3 kernels and their feature maps (28 × 28)", cm);
  const k1 = htmlEl("div", "cv-grid", "", cm); k1.style.gridTemplateColumns = "repeat(8, 1fr)";
  const m1 = htmlEl("div", "cv-grid", "", cm); m1.style.gridTemplateColumns = "repeat(8, 1fr)";
  htmlEl("div", "cv-h", "layer 2: 16 feature maps (14 × 14); click one", cm);
  const m2 = htmlEl("div", "cv-grid", "", cm); m2.style.gridTemplateColumns = "repeat(8, 1fr)";
  const patchHead = htmlEl("div", "cv-h", "", cm);
  const pg = htmlEl("div", "cv-grid", "", cm); pg.style.gridTemplateColumns = "repeat(9, 1fr)";

  // ---- output column ----
  const co = htmlEl("div", "cv-col cv-out", "", host);
  htmlEl("div", "cv-h", "class probabilities (softmax)", co);
  const bars = svgEl("svg", { viewBox: "0 0 270 300", width: "100%", role: "img", "aria-label": "Class probabilities" }, co);
  const setRow = htmlEl("div", "qb-presets", "", co);
  const setBtns = SETS.map(s => button(setRow, s.label, () => { const kindChanged = s.kind !== set.kind; set = s; if (kindChanged) { base = toImg(s.kind === "shapes" ? SHAPE_EXAMPLES[0].img : DIGIT_EXAMPLES[12].img); reset(); } else { occ = null; draw(); } }));
  const readout = htmlEl("div", "qb-readout", "", co);
  const presetBtns = presetRow(co, [
    { label: "a 6 rotated to a 9", apply: () => preset("digits", DIGIT_EXAMPLES[12].img, 180, 0, 0) },
    { label: "shifted digit: maps shift, prediction stays", apply: () => preset("digits", DIGIT_EXAMPLES[12].img, 0, 4, 0) },
    { label: "rotation 30°, no augmentation", apply: () => preset("digits", DIGIT_EXAMPLES[12].img, 30, 0, 0) },
    { label: "rotation 30°, with augmentation", apply: () => preset("digitsAug", DIGIT_EXAMPLES[12].img, 30, 0, 0) },
    { label: "a protrusion cell", apply: () => preset("shapes", SHAPE_EXAMPLES[3].img, 0, 0, 0) },
    { label: "what is the network looking at?", apply: () => { preset("digits", DIGIT_EXAMPLES[8].img, 0, 0, 0); const x = current(); occ = { ...occlusion(x, set.id, 4), k: 4 }; draw(); } },
  ]);
  void presetBtns;

  function preset(id: string, img: string, r: number, x: number, y: number): void {
    set = SETS.find(s => s.id === id)!; base = toImg(img); rot = r; dx = x; dy = y; occ = null; selMap = -1;
    sRot.set(r); sDx.set(x); sDy.set(y); buildExamples(); draw();
  }
  function reset(): void { rot = dx = dy = 0; sRot.set(0); sDx.set(0); sDy.set(0); occ = null; buildExamples(); draw(); }
  const argmax = (p: ArrayLike<number>) => { let k = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[k]) k = i; return k; };
  const current = () => transform(base, rot, dx, dy);

  function buildExamples(): void {
    exRow.replaceChildren();
    if (set.kind === "digits") [0, 2, 4, 6, 8, 10, 12, 14, 16, 18].forEach(i => button(exRow, DIGIT_EXAMPLES[i].label.replace("a test ", ""), () => { base = toImg(DIGIT_EXAMPLES[i].img); reset(); }));
    else SHAPE_EXAMPLES.slice(0, 4).forEach(s => button(exRow, SHAPE_NAMES[s.c], () => { base = toImg(s.img); reset(); }));
    credit.textContent = set.kind === "digits" ? "Digits: MNIST (LeCun, Cortes & Burges), CC BY-SA 3.0. Draw with the mouse; the drawing is not centred." : "Synthetic cell masks (as in the VAE day). Draw with the mouse.";
  }

  // drawing on the 28 × 28 pad: soft round brush, ink added (rotation/shift reset)
  let drawing = false;
  const paint = (ev: PointerEvent) => {
    const r = pad.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * 28, y = (ev.clientY - r.top) / r.height * 28;
    if (rot || dx || dy) { base = current(); rot = dx = dy = 0; sRot.set(0); sDx.set(0); sDy.set(0); }
    for (let j = Math.floor(y - 3); j <= y + 3; j++) for (let i = Math.floor(x - 3); i <= x + 3; i++) {
      if (i < 0 || i >= 28 || j < 0 || j >= 28) continue;
      const d2 = (i + 0.5 - x) ** 2 + (j + 0.5 - y) ** 2;
      base[j * 28 + i] = Math.min(1, base[j * 28 + i] + Math.exp(-d2 / 1.4));
    }
    occ = null; draw();
  };
  pad.addEventListener("pointerdown", ev => { ev.preventDefault(); drawing = true; pad.setPointerCapture(ev.pointerId); paint(ev); });
  pad.addEventListener("pointermove", ev => { if (drawing) paint(ev); });
  pad.addEventListener("pointerup", () => { drawing = false; });

  function mapCanvas(parent: HTMLElement, data: ArrayLike<number>, w: number, h: number, lo: number, hi: number): HTMLCanvasElement {
    const c = htmlEl("canvas", "", "", parent);
    drawGrey(c, { width: w, height: h, data: Float32Array.from(data) }, { mode: "clip", lo, hi });
    return c;
  }

  function draw(): void {
    setBtns.forEach((b, i) => b.classList.toggle("active", SETS[i] === set));
    const x = current(), r: CnnRun = runCnn(x, set.id), net = getCnn(set.id);
    // input (with occlusion overlay)
    const ctx = pad.getContext("2d")!, id = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) { const v = Math.round(255 * x[i]); id.data[4 * i] = id.data[4 * i + 1] = id.data[4 * i + 2] = v; id.data[4 * i + 3] = 255; }
    if (occ) {
      const p0 = r.probs[occ.k];
      for (let y = 0; y < 28; y++) for (let xx = 0; xx < 28; xx++) {
        // the drop in probability when the square centred near this pixel is covered
        const i = Math.min(occ.n - 1, Math.max(0, Math.round((y - 2.5) / 2))), j = Math.min(occ.n - 1, Math.max(0, Math.round((xx - 2.5) / 2)));
        const drop = Math.max(0, p0 - occ.map[i * occ.n + j]) / Math.max(1e-6, p0), q = 4 * (y * 28 + xx);
        id.data[q] = Math.min(255, id.data[q] * 0.5 + 255 * drop); id.data[q + 1] = id.data[q + 1] * 0.5; id.data[q + 2] = id.data[q + 2] * 0.5;
      }
    }
    ctx.putImageData(id, 0, 0);
    // layer 1 kernels and maps
    k1.replaceChildren(); m1.replaceChildren(); m2.replaceChildren(); pg.replaceChildren();
    for (let o = 0; o < 8; o++) {
      const w = net.c1.W.slice(9 * o, 9 * o + 9), m = Math.max(...Array.from(w, Math.abs));
      mapCanvas(k1, w, 3, 3, -m, m);
    }
    let mx1 = 1e-6; for (const v of r.a1.data) mx1 = Math.max(mx1, v);
    for (let o = 0; o < 8; o++) mapCanvas(m1, r.a1.data.subarray(o * 784, (o + 1) * 784), 28, 28, 0, mx1);
    let mx2 = 1e-6; for (const v of r.a2.data) mx2 = Math.max(mx2, v);
    for (let o = 0; o < 16; o++) {
      const c = mapCanvas(m2, r.a2.data.subarray(o * 196, (o + 1) * 196), 14, 14, 0, mx2);
      c.style.cursor = "pointer";
      if (o === selMap) c.classList.add("sel");
      c.addEventListener("click", () => { selMap = o === selMap ? -1 : o; draw(); });
    }
    if (selMap >= 0 && set.id === "digits") {
      patchHead.textContent = `the 9 test patches (8 × 8 = its receptive field) that excite layer-2 map ${selMap + 1} most`;
      TOP_PATCHES[selMap].forEach(p => mapCanvas(pg, Array.from(bytesFromB64(p), v => v / 255), 8, 8, 0, 1));
    } else patchHead.textContent = selMap >= 0 ? "(top patches are stored for the \"digits\" weights only)" : "";
    // bars
    bars.replaceChildren();
    const K = r.probs.length, bh = 270 / K, best = argmax(r.probs);
    for (let k = 0; k < K; k++) {
      const y = 6 + k * bh, w = 150 * r.probs[k];
      const lab = svgEl("text", { x: 4, y: y + bh * 0.7, style: `font-size:${Math.min(18, bh * 0.75)}px` }, bars);
      lab.textContent = set.kind === "digits" ? String(k) : SHAPE_NAMES[k];
      svgEl("rect", { x: 100, y: y + 2, width: Math.max(0.5, w), height: bh - 4, fill: k === best ? "#d62728" : "#1f77b4" }, bars);
      const v = svgEl("text", { x: 104 + w, y: y + bh * 0.7, style: `font-size:${Math.min(17, bh * 0.7)}px` }, bars);
      v.textContent = r.probs[k] >= 0.005 ? r.probs[k].toFixed(2) : "";
    }
    const acc = set.id === "shapes" ? CNN_INFO["shapes acc"] : set.id.startsWith("digitsE") ? CNN_INFO[set.id + " acc"] : CNN_INFO[set.id + " acc rot 0"];
    readout.textContent = `prediction: ${set.kind === "digits" ? best : SHAPE_NAMES[best]} (p = ${r.probs[best].toFixed(2)})\ntest accuracy of these weights: ${(100 * acc).toFixed(1)} %` +
      (set.id === "digits" || set.id === "digitsAug" ? `\n(on test digits rotated by 45°: ${(100 * CNN_INFO[set.id + " acc rot 45"]).toFixed(1)} %)` : "") +
      (occ ? `\nocclusion: red = covering this part lowers p(${set.kind === "digits" ? occ.k : SHAPE_NAMES[occ.k]})` : "");
  }

  buildExamples();
  draw();
}

autoMount(NAME, mount);
