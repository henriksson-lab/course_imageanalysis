// convlayer: one convolutional layer = a bank of 8 kernels (3 × 3), each giving one feature map
// (correlation + bias, then ReLU). Bank "hand-made": Sobel, Laplacian, Gaussian, diagonal and
// line detectors. Bank "learned": layer 1 of the small CNN of cnnviewer (trained on digits).
// Click a kernel to edit it. Shift the input: the maps shift with it (equivariance).
// "2×2 max pool" shows the maps at half size.
//
// <div data-demo="convlayer" data-input="digit" data-bank="learned"></div>
//   data-input  cell | digit | mask      data-bank  hand | learned      data-pool  1
import { autoMount, button, checkbox, htmlEl, presets as presetRow, releaseFocus } from "../lib/ui";
import { createGrey, drawGrey, type Grey } from "../lib/imageops";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { conv2d, relu, maxPool, tensor, shift as shiftT, convFromQ8, type ConvLayer } from "../lib/cnn";
import { bytesFromB64 } from "../lib/q8";
import { CNN_SETS, SHAPE_EXAMPLES } from "../cnnviewer/cnn_weights";
import { DIGIT_EXAMPLES } from "../vaeimages/digit_examples";

const NAME = "convlayer";
const STYLE = `
[data-demo="${NAME}"] { gap: 1em; align-items: flex-start; }
[data-demo="${NAME}"] .cl-in { flex: 0 1 270px; display: flex; flex-direction: column; gap: .35em; min-width: 200px; }
[data-demo="${NAME}"] .cl-in canvas { width: 100%; aspect-ratio: 1; image-rendering: pixelated; border: 1px solid var(--qb-grid); }
[data-demo="${NAME}"] .cl-bank { flex: 1 1 560px; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .5em .7em; min-width: 0; }
[data-demo="${NAME}"] .cl-unit { display: flex; flex-direction: column; align-items: center; gap: 3px; cursor: pointer; border-radius: 6px; padding: 3px; }
[data-demo="${NAME}"] .cl-unit.sel { background: var(--qb-panel); outline: 2px solid var(--qb-accent); }
[data-demo="${NAME}"] .cl-k { display: grid; grid-template-columns: repeat(3, 1fr); width: 78px; height: 78px; border: 1px solid #999; }
[data-demo="${NAME}"] .cl-k div { display: flex; align-items: center; justify-content: center; font-size: 13px; font-family: ui-monospace, monospace; }
[data-demo="${NAME}"] .cl-unit canvas { width: 100%; max-width: 140px; aspect-ratio: 1; image-rendering: pixelated; border: 1px solid var(--qb-grid); }
[data-demo="${NAME}"] .cl-name { font-size: 15px; text-align: center; }
[data-demo="${NAME}"] .cl-edit { display: grid; grid-template-columns: repeat(3, 4.2em); gap: 3px; }
[data-demo="${NAME}"] .cl-edit input { font: inherit; font-size: 16px; width: 100%; box-sizing: border-box; text-align: center; }
[data-demo="${NAME}"] .qb-note { font-size: 14px; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .cl-k { width: 62px; height: 62px; }
.reveal [data-demo="${NAME}"] .cl-k div { font-size: 11px; }
.reveal [data-demo="${NAME}"] .cl-unit canvas { max-width: 108px; }
.reveal [data-demo="${NAME}"] .cl-in { flex-basis: 250px; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

interface K { name: string; w: number[]; b: number }
const HAND: K[] = [
  { name: "Sobel x", w: [-1, 0, 1, -2, 0, 2, -1, 0, 1], b: 0 },
  { name: "Sobel y", w: [-1, -2, -1, 0, 0, 0, 1, 2, 1], b: 0 },
  { name: "diagonal ↗", w: [0, 1, 2, -1, 0, 1, -2, -1, 0], b: 0 },
  { name: "diagonal ↘", w: [-2, -1, 0, -1, 0, 1, 0, 1, 2], b: 0 },
  { name: "Laplacian", w: [0, 1, 0, 1, -4, 1, 0, 1, 0], b: 0 },
  { name: "Gaussian", w: [1, 2, 1, 2, 4, 2, 1, 2, 1].map(v => v / 16), b: 0 },
  { name: "horizontal line", w: [-1, -1, -1, 2, 2, 2, -1, -1, -1], b: 0 },
  { name: "vertical line", w: [-1, 2, -1, -1, 2, -1, -1, 2, -1], b: 0 },
];
function learned(): K[] {
  const L = convFromQ8(CNN_SETS.digits.c1, 3);
  return Array.from({ length: 8 }, (_, o) => ({ name: `learned ${o + 1}`, w: Array.from(L.W.slice(9 * o, 9 * o + 9), v => Math.round(v * 100) / 100), b: Math.round(L.b[o] * 100) / 100 }));
}

function grey28(b64: string): Grey {
  const by = bytesFromB64(b64), g = createGrey(28, 28);
  for (let i = 0; i < 784; i++) g.data[i] = by[i] / 255;
  return g;
}

function mount(host: HTMLElement): void {
  injectStyle();
  let inputKind = host.dataset.input ?? "cell", bankKind = host.dataset.bank ?? "hand", pool = host.dataset.pool === "1";
  let bank: K[] = (bankKind === "learned" ? learned() : HAND).map(k => ({ ...k, w: [...k.w] }));
  let sel = 0, dx = 0, dy = 0, cell: Grey | null = null;
  const digit = grey28(DIGIT_EXAMPLES[12].img), mask = grey28(SHAPE_EXAMPLES[3].img);

  const left = htmlEl("div", "cl-in", "", host);
  htmlEl("div", "cl-name", "input", left);
  const inCanvas = htmlEl("canvas", "", "", left);
  const inRow = htmlEl("div", "qb-presets", "", left);
  const inBtns = ["cell", "digit", "mask"].map(k => button(inRow, k === "cell" ? "micrograph" : k === "digit" ? "digit" : "cell mask", () => { inputKind = k; dx = dy = 0; draw(); }));
  const shRow = htmlEl("div", "qb-presets", "", left);
  const step = () => (inputKind === "cell" ? 8 : 3);
  button(shRow, "shift →", () => { dx += step(); draw(); });
  button(shRow, "shift ↓", () => { dy += step(); draw(); });
  button(shRow, "no shift", () => { dx = dy = 0; draw(); });
  const cPool = checkbox(left, "2×2 max pool", pool, v => { pool = v; draw(); });
  const bankRow = htmlEl("div", "qb-presets", "", left);
  const bankBtns = ["hand", "learned"].map(k => button(bankRow, k === "hand" ? "hand-made kernels" : "learned kernels", () => { bankKind = k; bank = (k === "learned" ? learned() : HAND).map(q => ({ ...q, w: [...q.w] })); draw(); }));
  htmlEl("div", "cl-name", "edit the selected kernel:", left);
  const edit = htmlEl("div", "cl-edit", "", left);
  const inputs = Array.from({ length: 9 }, (_, i) => {
    const e = htmlEl("input", "", "", edit);
    e.type = "number"; e.step = "0.1";
    releaseFocus(e);
    e.addEventListener("change", () => { const v = Number(e.value); if (Number.isFinite(v)) { bank[sel].w[i] = v; draw(); } });
    return e;
  });
  const credit = htmlEl("div", "qb-note", "", left);
  const grid = htmlEl("div", "cl-bank", "", host);
  const presetBtns = presetRow(left, [
    { label: "learned filters look like edge detectors", apply: () => { inputKind = "digit"; bankKind = "learned"; bank = learned(); pool = false; dx = dy = 0; draw(); } },
    { label: "shift the input", apply: () => { inputKind = "cell"; bankKind = "hand"; bank = HAND.map(q => ({ ...q, w: [...q.w] })); pool = false; dx = 24; dy = 0; draw(); } },
    { label: "pooling", apply: () => { inputKind = "digit"; bankKind = "hand"; bank = HAND.map(q => ({ ...q, w: [...q.w] })); pool = true; dx = dy = 0; draw(); } },
  ]);
  void presetBtns;

  function currentInput(): Grey | null {
    if (inputKind === "digit") return digit;
    if (inputKind === "mask") return mask;
    return cell;
  }

  function draw(): void {
    inBtns.forEach((b, i) => b.classList.toggle("active", ["cell", "digit", "mask"][i] === inputKind));
    bankBtns.forEach((b, i) => b.classList.toggle("active", ["hand", "learned"][i] === bankKind));
    cPool.checked = pool;
    credit.textContent = inputKind === "cell" ? CELL_CREDIT : inputKind === "digit" ? "MNIST test digit (LeCun, Cortes & Burges), CC BY-SA 3.0" : "synthetic cell mask";
    const g = currentInput();
    if (!g) return;
    const x = shiftT(tensor(1, g.height, g.width, Float32Array.from(g.data)), dx, dy, 0);
    drawGrey(inCanvas, { width: g.width, height: g.height, data: x.data }, { mode: "clip", lo: 0, hi: 1 });
    const L: ConvLayer = { W: Float32Array.from(bank.flatMap(k => k.w)), b: Float32Array.from(bank.map(k => k.b)), cin: 1, cout: 8, k: 3 };
    let maps = relu(conv2d(x, L));
    if (pool) maps = maxPool(maps);
    grid.replaceChildren();
    bank.forEach((k, o) => {
      const u = htmlEl("div", "cl-unit" + (o === sel ? " sel" : ""), "", grid);
      u.addEventListener("click", () => { sel = o; draw(); });
      const kg = htmlEl("div", "cl-k", "", u);
      const m = Math.max(1e-9, ...k.w.map(Math.abs));
      k.w.forEach(v => {
        const c = htmlEl("div", "", (Math.round(v * 100) / 100).toString().replace(/^(-?)0\./, "$1."), kg);
        const t = v / m;
        c.style.background = t >= 0 ? `rgb(${255 - 30 * t},${255 - 130 * t},${255 - 220 * t})` : `rgb(${255 + 210 * t},${255 + 120 * t},${255 + 30 * t})`;
      });
      const hw = maps.h * maps.w, data = maps.data.slice(o * hw, (o + 1) * hw);
      let mx = 0; for (const v of data) mx = Math.max(mx, v);
      const cv = htmlEl("canvas", "", "", u);
      drawGrey(cv, { width: maps.w, height: maps.h, data }, { mode: "clip", lo: 0, hi: mx || 1 });
      htmlEl("div", "cl-name", `${k.name}${k.b ? ` (b ${k.b})` : ""}  max ${mx.toFixed(2)}`, u);
    });
    bank[sel].w.forEach((v, i) => { inputs[i].value = String(v); });
  }

  loadCell(128).then(img => { cell = { width: img.width, height: img.height, data: Float32Array.from(img.data, v => v / 255) }; draw(); });
  draw();
}

autoMount(NAME, mount);
