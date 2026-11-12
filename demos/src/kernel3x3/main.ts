// 3 x 3 kernel filter. An input image (the micrograph of fourier2d, or a synthetic test
// pattern, optionally with seeded noise), an editable 3 x 3 kernel, and the filtered
// output, updated live. Presets, "divide by the sum", apply 1-5 times, border rule,
// three ways to show negative outputs, and a click on either image shows the
// weighted sum for that pixel with numbers.
// Auto-mounts on <div data-demo="kernel3x3">. Optional: data-image="cell|test",
// data-noise="0" (noise SD in grey levels), data-preset="gauss" (a preset id from conv.ts),
// data-times="1", data-border="replicate|zero", data-preblur="1" (Gaussian x 2 on the input first).
import { htmlEl, slider, presets, checkbox, button, autoMount, rng } from "../lib/ui";
import { PRESETS, Border, Display, kernelSum, divisor, neighbourhood, filterRepeat, toGrey } from "./conv";
// @ts-ignore -- esbuild inlines the image as a data URL
import cellUrl from "../fourier2d/cell256.png";

const NAME = "kernel3x3";
const N = 256;
const CREDIT = "Image: transmission electron micrograph of a mesenchymal stem cell, Robert M. Hunt, CC BY 3.0, via Wikimedia Commons";
const SEED = 7;
const GAUSS = [1, 2, 1, 2, 4, 2, 1, 2, 1];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { flex-direction: column; align-items: stretch; gap: .5em; }
.qb-demo[data-demo="${NAME}"] .k3-top { display: flex; gap: 14px; align-items: flex-start; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; flex: 0 1 282px; min-width: 200px; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 18px; text-align: center; line-height: 1.2; margin: 0 0 .25em; }
.qb-demo[data-demo="${NAME}"] figcaption small { color: var(--qb-muted); font-size: 18px; }
.qb-demo[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; cursor: crosshair; touch-action: none; image-rendering: pixelated; }
.qb-demo[data-demo="${NAME}"] .k3-kern { flex: 0 1 312px; display: flex; flex-direction: column; gap: .3em; align-items: center; }
.qb-demo[data-demo="${NAME}"] .k3-grid { display: grid; grid-template-columns: repeat(3, 100px); gap: 5px; }
.qb-demo[data-demo="${NAME}"] .k3-cell { display: flex; align-items: stretch; gap: 2px; background: var(--qb-panel); border-radius: 6px; padding: 3px; }
.qb-demo[data-demo="${NAME}"] .k3-cell input { font: inherit; font-size: 22px; font-family: ui-monospace, monospace; width: 100%; box-sizing: border-box; text-align: center; border: 1px solid #bbb; border-radius: 4px; padding: 2px 0; background: var(--qb-bg); color: var(--qb-fg); }
.qb-demo[data-demo="${NAME}"] .k3-cell input.neg { color: var(--qb-c2); }
.qb-demo[data-demo="${NAME}"] .k3-cell input.zero { color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .k3-cell .qb-btn { flex: 0 0 22px; padding: 0; font-size: 18px; line-height: 1.2; }
.qb-demo[data-demo="${NAME}"] .k3-sum { font-size: 18px; text-align: center; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 200px; gap: .45em; min-width: 190px; }
.qb-demo[data-demo="${NAME}"] .k3-lab { font-size: 18px; color: var(--qb-muted); margin-bottom: -.25em; }
.qb-demo[data-demo="${NAME}"] .k3-presets { display: flex; gap: .5em; align-items: center; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] .k3-bottom { display: flex; gap: 1em; align-items: flex-start; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] .k3-bottom .qb-readout { flex: 0 0 auto; font-size: 18px; line-height: 1.25; margin: 0; white-space: pre; }
.qb-demo[data-demo="${NAME}"] .k3-side { flex: 1 1 300px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .k3-credit { font-size: 13px; color: var(--qb-muted); }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 20px; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 20px; margin: 0 0 .25em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .k3-cell .qb-btn { font-size: 18px; padding: 0; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .k3-bottom .qb-readout { font-size: 18px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

/** Decode the inlined PNG into grey levels 0..255. */
function loadCell(): Promise<Float64Array> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = N; c.height = N;
      const ctx = c.getContext("2d")!;
      ctx.drawImage(img, 0, 0, N, N);
      const d = ctx.getImageData(0, 0, N, N).data;
      const out = new Float64Array(N * N);
      for (let k = 0; k < N * N; k++) out[k] = d[4 * k];
      resolve(out);
    };
    img.onerror = reject;
    img.src = cellUrl;
  });
}

/** A synthetic test image: a "cell" with a nucleus, thin lines, letters, a checkerboard. */
function testImage(): Float64Array {
  const c = document.createElement("canvas");
  c.width = N; c.height = N;
  const ctx = c.getContext("2d")!;
  const grey = (v: number) => `rgb(${v},${v},${v})`;
  ctx.fillStyle = grey(50); ctx.fillRect(0, 0, N, N);
  // cell with a nucleus (top left)
  ctx.fillStyle = grey(190); ctx.beginPath(); ctx.arc(72, 72, 54, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = grey(110); ctx.beginPath(); ctx.arc(80, 66, 22, 0, 2 * Math.PI); ctx.fill();
  // thin lines, 1, 2 and 4 pixels wide (top right)
  ctx.fillStyle = grey(230);
  [[150, 1], [162, 2], [176, 4]].forEach(([x, w]) => ctx.fillRect(x, 16, w, 100));
  [[196, 1], [212, 2], [228, 4]].forEach(([y, h]) => ctx.fillRect(196, y - 180, 50, h));
  // letters (bottom left), pixel-sharp: draw then threshold
  ctx.fillStyle = grey(230);
  ctx.font = "bold 58px sans-serif"; ctx.textBaseline = "alphabetic";
  ctx.fillText("ABC", 10, 200);
  // a filled square and a checkerboard (bottom right)
  ctx.fillStyle = grey(150); ctx.fillRect(12, 214, 110, 30);
  for (let r = 0; r < 12; r++) for (let q = 0; q < 12; q++) {
    ctx.fillStyle = grey((r + q) % 2 ? 30 : 220);
    ctx.fillRect(146 + 8 * q, 146 + 8 * r, 8, 8);
  }
  const d = ctx.getImageData(0, 0, N, N).data;
  const out = new Float64Array(N * N);
  for (let k = 0; k < N * N; k++) out[k] = d[4 * k];
  return out;
}

/** Add seeded Gaussian noise and round to whole grey levels 0..255. */
function addNoise(img: Float64Array, sd: number): Float64Array {
  const r = rng(SEED);
  const out = new Float64Array(img.length);
  for (let k = 0; k < img.length; k++) {
    const v = sd > 0 ? img[k] + sd * r.norm() : img[k];
    out[k] = Math.min(255, Math.max(0, Math.round(v)));
  }
  return out;
}

function mkFigure(parent: Element, title: string, sub: string): { fig: HTMLElement; c: HTMLCanvasElement; sub: HTMLElement } {
  const fig = htmlEl("figure", "", "", parent);
  const cap = htmlEl("figcaption", "", title + " ", fig);
  const s = htmlEl("small", "", sub, cap);
  const c = htmlEl("canvas", "", "", fig);
  c.width = N; c.height = N;
  return { fig, c, sub: s };
}

function paint(c: HTMLCanvasElement, grey: ArrayLike<number>, sel: { row: number; col: number }): void {
  const ctx = c.getContext("2d")!;
  const id = ctx.createImageData(N, N);
  for (let k = 0; k < N * N; k++) {
    const g = grey[k];
    id.data[4 * k] = id.data[4 * k + 1] = id.data[4 * k + 2] = g; id.data[4 * k + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
  // marker: the 3 x 3 neighbourhood of the selected pixel, and a ring around it
  const x = sel.col - 1, y = sel.row - 1;
  ctx.lineWidth = 3; ctx.strokeStyle = "#000";
  ctx.strokeRect(x - 0.5, y - 0.5, 4, 4);
  ctx.beginPath(); ctx.arc(sel.col + 0.5, sel.row + 0.5, 11, 0, 2 * Math.PI); ctx.stroke();
  ctx.lineWidth = 1.5; ctx.strokeStyle = "#ffd400";
  ctx.strokeRect(x - 0.5, y - 0.5, 4, 4);
  ctx.beginPath(); ctx.arc(sel.col + 0.5, sel.row + 0.5, 11, 0, 2 * Math.PI); ctx.stroke();
}

/** Short number for the readout: integers as is, else 1-2 decimals. */
function fmt(v: number, dec = 1): string {
  if (Math.abs(v - Math.round(v)) < 1e-9) return String(Math.round(v));
  return v.toFixed(dec);
}

function pad(s: string, n: number): string { return s.length >= n ? s : " ".repeat(n - s.length) + s; }

function mount(host: HTMLElement): void {
  injectStyle();
  const top = htmlEl("div", "k3-top", "", host);
  const fIn = mkFigure(top, "input", "");
  const kern = htmlEl("div", "k3-kern", "", top);
  const fOut = mkFigure(top, "output", "");
  const panel = htmlEl("div", "qb-panel", "", top);
  const presetRow = htmlEl("div", "k3-presets", "", host);
  const bottom = htmlEl("div", "k3-bottom", "", host);
  const ro = htmlEl("div", "qb-readout k3-ro", "", bottom);
  const side = htmlEl("div", "k3-side", "", bottom);
  const note = htmlEl("div", "qb-note", "", side);
  const credit = htmlEl("div", "k3-credit", CREDIT, side);

  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS.find(p => p.id === "gauss")!;
  const st = {
    image: (host.dataset.image === "test" ? "test" : "cell") as "cell" | "test",
    noise: Number(host.dataset.noise ?? 0),
    k: p0.k.slice(),
    normalise: p0.normalise,
    times: Math.max(1, Math.min(5, Number(host.dataset.times ?? 1))),
    border: (host.dataset.border === "zero" ? "zero" : "replicate") as Border,
    display: "clip" as Display,
    autoDisplay: true,
    preblur: host.dataset.preblur === "1",
    sel: { row: 128, col: 128 },
  };

  // ---- kernel editor ----
  htmlEl("div", "k3-sum", "kernel: 3 × 3 weights", kern);
  const grid = htmlEl("div", "k3-grid", "", kern);
  const inputs: HTMLInputElement[] = [];
  for (let q = 0; q < 9; q++) {
    const cell = htmlEl("div", "k3-cell", "", grid);
    const inp = htmlEl("input", "", "", cell);
    inp.type = "number"; inp.step = "1";
    inp.setAttribute("aria-label", `weight ${"abcdefghi"[q]}`);
    inp.addEventListener("input", () => {
      const v = parseFloat(inp.value);
      if (!Number.isFinite(v)) return;
      st.k[q] = v; changed(false);
    });
    inp.addEventListener("change", () => { syncInputs(); inp.blur(); });
    inp.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === "Escape") inp.blur(); });
    inputs.push(inp);
    const minus = button(cell, "−", () => { st.k[q] = Math.round((st.k[q] - 1) * 1e6) / 1e6; changed(true); });
    cell.insertBefore(minus, inp);
    button(cell, "+", () => { st.k[q] = Math.round((st.k[q] + 1) * 1e6) / 1e6; changed(true); });
  }
  const sumEl = htmlEl("div", "k3-sum", "", kern);
  const normBox = checkbox(kern, "divide by the sum", st.normalise, on => { st.normalise = on; changed(false); });
  const timesS = slider(kern, "apply ×", 1, 5, 1, st.times, v => { st.times = v; schedule(); });

  // ---- side panel ----
  htmlEl("div", "k3-lab", "image", panel);
  const imgBtns = presets(panel, [
    { label: "cell", apply: () => { st.image = "cell"; makeInput(); } },
    { label: "test pattern", apply: () => { st.image = "test"; makeInput(); } },
  ]);
  const noiseS = slider(panel, "noise", 0, 60, 5, st.noise, v => { st.noise = v; makeInput(); });
  htmlEl("div", "k3-lab", "show the output", panel);
  const dispItems: { label: string; mode: Display }[] = [
    { label: "as is", mode: "clip" }, { label: "|value|", mode: "abs" }, { label: "0 = grey", mode: "mid" },
  ];
  const dispBtns = presets(panel, dispItems.map(d => ({ label: d.label, apply: () => { st.display = d.mode; st.autoDisplay = false; schedule(); } })));
  htmlEl("div", "k3-lab", "outside the image", panel);
  const borderBtns = presets(panel, [
    { label: "copy edge", apply: () => { st.border = "replicate"; schedule(); } },
    { label: "zeros", apply: () => { st.border = "zero"; schedule(); } },
  ]);
  const preBox = checkbox(panel, "blur input first (Gaussian × 2)", st.preblur, on => { st.preblur = on; schedule(); });

  // ---- presets ----
  const presetBtns = presets(presetRow, PRESETS.map(p => ({
    label: p.label, apply: () => {
      st.k = p.k.slice(); st.normalise = p.normalise; st.autoDisplay = true; normBox.checked = p.normalise;
      changed(true);
    },
  })));
  button(presetRow, "reset", () => {
    st.k = p0.k.slice(); st.normalise = p0.normalise; normBox.checked = p0.normalise; st.autoDisplay = true;
    st.times = Math.max(1, Math.min(5, Number(host.dataset.times ?? 1))); timesS.set(st.times);
    st.border = host.dataset.border === "zero" ? "zero" : "replicate";
    st.preblur = host.dataset.preblur === "1"; preBox.checked = st.preblur;
    st.sel = { row: 128, col: 128 };
    changed(true);
  });

  function syncInputs(): void {
    st.k.forEach((v, q) => {
      if (document.activeElement !== inputs[q]) inputs[q].value = String(v);
      inputs[q].classList.toggle("neg", v < 0);
      inputs[q].classList.toggle("zero", v === 0);
    });
  }

  /** The kernel or the normalise box changed. `all` also rewrites the input boxes. */
  function changed(all: boolean): void {
    if (all) inputs.forEach((inp, q) => { inp.value = String(st.k[q]); });
    syncInputs();
    schedule();
  }

  // ---- clicks on either image select a pixel ----
  for (const c of [fIn.c, fOut.c]) {
    c.addEventListener("pointerdown", ev => {
      const r = c.getBoundingClientRect();
      const col = Math.floor((ev.clientX - r.left) / r.width * N);
      const row = Math.floor((ev.clientY - r.top) / r.height * N);
      st.sel = { row: Math.min(N - 1, Math.max(0, row)), col: Math.min(N - 1, Math.max(0, col)) };
      schedule();
    });
  }

  // ---- images ----
  let cellClean: Float64Array | null = null;
  let input: Float64Array = new Float64Array(N * N);
  let testClean: Float64Array | null = null;

  function makeInput(): void {
    if (st.image === "cell" && !cellClean) {
      loadCell().then(v => { cellClean = v; makeInput(); });
      return;
    }
    if (st.image === "test" && !testClean) testClean = testImage();
    input = addNoise(st.image === "cell" ? cellClean! : testClean!, st.noise);
    schedule();
  }

  let pending = false;
  function schedule(): void {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; draw(); });
  }

  function draw(): void {
    const s = kernelSum(st.k);
    const div = divisor(st.k, st.normalise);
    if (st.autoDisplay) st.display = Math.abs(s) < 1e-9 ? "mid" : "clip";
    const base = st.preblur ? filterRepeat(input, N, N, GAUSS, 16, st.border, 2) : input;
    const out = filterRepeat(base, N, N, st.k, div, st.border, st.times);
    const { grey, lo, hi } = toGrey(out, st.display);

    paint(fIn.c, st.preblur ? toGrey(base, "clip").grey : input, st.sel);
    paint(fOut.c, grey, st.sel);
    fIn.sub.textContent = (st.image === "cell" ? (st.noise ? `(noise ${st.noise}` : "(micrograph")
      : (st.noise ? `(test, noise ${st.noise}` : "(test pattern")) + (st.preblur ? ", blurred)" : ")");
    fOut.sub.textContent = st.display === "clip" ? "(as is)" : st.display === "abs" ? "(|value|, stretched)" : "(0 = grey, stretched)";
    credit.style.visibility = st.image === "cell" ? "visible" : "hidden";

    // sum and divisor
    const sTxt = fmt(s, 2);
    sumEl.textContent = Math.abs(s) < 1e-9 ? "sum = 0: only changes survive"
      : div !== 1 ? `sum = ${sTxt}: output ÷ ${sTxt}`
      : Math.abs(s - 1) < 1e-9 ? "sum = 1: brightness kept" : `sum = ${sTxt} (not divided)`;
    normBox.parentElement!.style.opacity = Math.abs(s) < 1e-9 || Math.abs(s - 1) < 1e-9 ? "0.45" : "1";

    // highlight buttons
    const pi = PRESETS.findIndex(p => p.k.every((v, q) => v === st.k[q]) && (p.normalise === st.normalise || Math.abs(kernelSum(p.k)) < 1e-9 || kernelSum(p.k) === 1));
    presetBtns.forEach((b, i) => b.classList.toggle("active", i === pi));
    imgBtns.forEach((b, i) => b.classList.toggle("active", i === (st.image === "cell" ? 0 : 1)));
    dispBtns.forEach((b, i) => b.classList.toggle("active", dispItems[i].mode === st.display));
    borderBtns.forEach((b, i) => b.classList.toggle("active", i === (st.border === "replicate" ? 0 : 1)));
    noiseS.set(st.noise); timesS.set(st.times);

    // the weighted sum at the selected pixel, for the LAST pass
    const { row, col } = st.sel;
    const prev = st.times > 1 ? filterRepeat(base, N, N, st.k, div, st.border, st.times - 1) : base;
    const nb = neighbourhood(prev, N, N, row, col, st.border);
    const ws = nb.reduce((a, v, q) => a + v * st.k[q], 0);
    const val = ws / div;
    const g = grey[row * N + col];
    const iw = Math.max(4, ...nb.map(v => fmt(v).length + 1));
    const kw = Math.max(3, ...st.k.map(v => fmt(v, 2).length + 1));
    const pw = Math.max(5, ...nb.map((v, q) => fmt(v * st.k[q]).length + 2));
    const lines: string[] = [];
    lines.push(`pixel row ${row + 1}, column ${col + 1}` + (st.times > 1 ? ` (pass ${st.times} of ${st.times})` : "")
      + (st.preblur ? ", input blurred first" : "") + ":");
    const head = "input".padEnd(3 * iw + 4) + "weights".padEnd(3 * kw + 4) + "products";
    lines.push(head);
    for (let r = 0; r < 3; r++) {
      const a = [0, 1, 2].map(q => pad(fmt(nb[3 * r + q]), iw)).join("");
      const b = [0, 1, 2].map(q => pad(fmt(st.k[3 * r + q], 2), kw)).join("");
      const c = [0, 1, 2].map(q => pad(fmt(nb[3 * r + q] * st.k[3 * r + q]), pw)).join("");
      lines.push(a + (r === 1 ? "  × " : "    ") + b + (r === 1 ? "  → " : "    ") + c);
    }
    let last = `sum of products = ${fmt(ws)}`;
    if (div !== 1) last += `, ÷ ${fmt(div, 2)} = ${fmt(val, 2)}`;
    last += `   → grey ${g}`;
    lines.push(last);
    lines.push(`whole output: from ${fmt(lo, 0)} to ${fmt(hi, 0)}`);
    ro.textContent = lines.join("\n");

    note.textContent = (st.display === "clip"
      ? "As is: values below 0 show black, above 255 white. "
      : st.display === "abs"
        ? "|value|: the size of the response, sign dropped; the largest is white. "
        : "0 = grey: negative darker, positive brighter; the largest size is black or white. ")
      + "Edge kernels (sum 0) give negative numbers, so they need one of the last two. "
      + "Click either image to see the weighted sum at that pixel. Type a weight and press Enter.";
  }

  // initial state
  inputs.forEach((inp, q) => { inp.value = String(st.k[q]); });
  syncInputs();
  makeInput();
}

autoMount(NAME, mount);
