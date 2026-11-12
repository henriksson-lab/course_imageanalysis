// Blur and downsample in linear light vs on gamma-encoded values. The same Gaussian blur
// (or block-average downsampling) is applied twice: directly on the encoded sRGB values
// ("naive"), and as decode -> blur -> encode ("linear light"). A profile along a chosen
// row compares the two. Toggle: show the linear values directly (too dark: why we encode).
// Auto-mounts on <div data-demo="gammablur">. Options: data-image (stripes|checker|lights|
// lines), data-op (blur|down2|down4), data-sigma, data-preset (stripes|seam|lights|down).
import { htmlEl, slider, presets, button, checkbox, autoMount, rng } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { srgbToLinear, linearToSrgb } from "../lib/colour";
import { createGrey, gaussianBlur, Grey } from "../lib/imageops";

const NAME = "gammablur";
const N = 128;
const STYLE = `
[data-demo="${NAME}"] { gap: .7em; }
[data-demo="${NAME}"] .gb-imgs { flex: 1 1 640px; display: flex; flex-wrap: wrap; gap: .5em; min-width: 0; max-width: 760px; }
[data-demo="${NAME}"] .gb-imgs figure { margin: 0; flex: 1 1 180px; max-width: 240px; min-width: 120px; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin-bottom: .15em; }
[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; cursor: row-resize; touch-action: none; }
[data-demo="${NAME}"] .gb-imgs figure.gb-prof { flex: 1 1 100%; max-width: 740px; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 330px; gap: .4em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .gb-row { display: flex; flex-wrap: wrap; gap: .3em; }
[data-demo="${NAME}"] .gb-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .gb-imgs figure { max-width: 205px; }
.reveal [data-demo="${NAME}"] .gb-imgs figure.gb-prof { max-width: 630px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
`;

type Img = [Float32Array, Float32Array, Float32Array]; // encoded sRGB in [0, 1], N x N
type ImageId = "stripes" | "checker" | "lights" | "lines";
type Op = "blur" | "down2" | "down4";

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const newImg = (): Img => [new Float32Array(N * N), new Float32Array(N * N), new Float32Array(N * N)];

/** Test images, as encoded sRGB values in [0, 1]. */
export function makeImage(id: ImageId): Img {
  const I = newImg();
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const p = y * N + x;
    let c = [0, 0, 0];
    if (id === "stripes") {
      // left: 1-pixel black/white stripes; right: uniform 128 (top) and 188 (bottom)
      if (x < 64) c = x % 2 ? [1, 1, 1] : [0, 0, 0];
      else { const v = (y < 64 ? 128 : 188) / 255; c = [v, v, v]; }
    } else if (id === "checker") {
      const red = (Math.floor(x / 32) + Math.floor(y / 32)) % 2 === 0;
      c = red ? [1, 0, 0] : [0, 1, 0];
    } else if (id === "lines") {
      const lines = [[10, 1], [22, 1], [34, 2], [48, 3]];
      const on = lines.some(([x0, w]) => x >= x0 && x < x0 + w) || (x >= 64 && (x + y) % 6 === 0) ;
      c = on ? [1, 1, 1] : [0, 0, 0];
      if (y > 100 && x < 64) c = x % 4 < 2 ? [1, 0.85, 0.3] : [0.05, 0.05, 0.25];
    }
    I[0][p] = c[0]; I[1][p] = c[1]; I[2][p] = c[2];
  }
  if (id === "lights") {
    // seeded small bright lights on a dark night sky, some saturated (clipped at 1)
    const r = rng(17), lin = [new Float32Array(N * N).fill(0.002), new Float32Array(N * N).fill(0.003), new Float32Array(N * N).fill(0.008)];
    const cols = [[1, 0.75, 0.35], [1, 0.3, 0.2], [0.4, 0.8, 1], [1, 1, 1], [0.3, 1, 0.4]];
    for (let k = 0; k < 40; k++) {
      const cx = 4 + r.unif() * (N - 8), cy = 4 + r.unif() * (N - 8), s = 0.6 + 0.9 * r.unif(), a = 1 + 5 * r.unif(), col = cols[k % cols.length];
      for (let y = Math.floor(cy - 5); y <= cy + 5; y++) for (let x = Math.floor(cx - 5); x <= cx + 5; x++) {
        if (x < 0 || y < 0 || x >= N || y >= N) continue;
        const g = a * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s));
        for (let c = 0; c < 3; c++) lin[c][y * N + x] += g * col[c];
      }
    }
    for (let c = 0; c < 3; c++) for (let p = 0; p < N * N; p++) I[c][p] = linearToSrgb(Math.min(1, lin[c][p]));
  }
  return I;
}

/** The operation on one channel (values in whatever domain they come). */
function op(ch: Float32Array, o: Op, sigma: number): Float32Array {
  if (o === "blur") {
    const g: Grey = createGrey(N, N);
    g.data.set(ch);
    return Float32Array.from(gaussianBlur(g, sigma, "mirror").data);
  }
  // block average, shown at the original size (each block one colour)
  const k = o === "down2" ? 2 : 4, out = new Float32Array(N * N);
  for (let by = 0; by < N; by += k) for (let bx = 0; bx < N; bx += k) {
    let s = 0;
    for (let y = by; y < by + k; y++) for (let x = bx; x < bx + k; x++) s += ch[y * N + x];
    s /= k * k;
    for (let y = by; y < by + k; y++) for (let x = bx; x < bx + k; x++) out[y * N + x] = s;
  }
  return out;
}

/** Naive (on encoded values) and correct (in linear light) results, both encoded. */
export function process(I: Img, o: Op, sigma: number): { naive: Img; linear: Img } {
  const naive = I.map(ch => op(ch, o, sigma)) as Img;
  const linear = I.map(ch => {
    const l = op(Float32Array.from(ch, srgbToLinear), o, sigma);
    return Float32Array.from(l, v => linearToSrgb(Math.min(1, Math.max(0, v))));
  }) as Img;
  return { naive, linear };
}

/** Mean luminance Y (linear light) of an encoded image. */
export function meanLuminance(I: Img): number {
  let s = 0;
  for (let p = 0; p < N * N; p++) s += 0.2126 * srgbToLinear(I[0][p]) + 0.7152 * srgbToLinear(I[1][p]) + 0.0722 * srgbToLinear(I[2][p]);
  return s / (N * N);
}

autoMount(NAME, host => {
  injectStyle();
  const st = {
    image: (["stripes", "checker", "lights", "lines"].includes(host.dataset.image ?? "") ? host.dataset.image : "stripes") as ImageId,
    op: (["blur", "down2", "down4"].includes(host.dataset.op ?? "") ? host.dataset.op : "blur") as Op,
    sigma: Number(host.dataset.sigma ?? 2) || 2, row: 40, showLinear: false, channel: 3,
  };
  const imgs = htmlEl("div", "gb-imgs", "", host);
  const mk = (cap: string) => { const f = htmlEl("figure", "", "", imgs); htmlEl("figcaption", "", cap, f); const c = htmlEl("canvas", "", "", f); c.width = N; c.height = N; return c; };
  const cIn = mk("input"), cNaive = mk("on encoded values"), cLin = mk("in linear light");
  const profBox = plotFigure(imgs, "", "gb-prof");
  const prof = new Plot(profBox, { width: 740, height: 210, x: [0, N - 1], y: [0, 255], xlabel: "x (pixels) along the marked row", ylabel: "value", yticks: [0, 64, 128, 188, 255], margin: { b: 52 } });
  const lIn = prof.line([], [], { color: "#aaa", width: 2, label: "input" });
  const lNaive = prof.line([], [], { color: "#d62728", width: 3, dash: "7 4", label: "on encoded values" });
  const lLin = prof.line([], [], { color: "#1f77b4", width: 3, label: "in linear light" });
  prof.legend(undefined, "tr");

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "gb-lab", "Image", panel);
  const imRow = htmlEl("div", "gb-row", "", panel);
  const IM: Record<ImageId, string> = { stripes: "stripes + greys", checker: "red/green checker", lights: "lights at night", lines: "thin lines" };
  const imBtns = (Object.keys(IM) as ImageId[]).map(id => { const b = button(imRow, IM[id], () => { st.image = id; update(); }); b.dataset.id = id; return b; });
  htmlEl("div", "gb-lab", "Operation", panel);
  const opRow = htmlEl("div", "gb-row", "", panel);
  const OPS: Record<Op, string> = { blur: "Gaussian blur", down2: "downsample ×2", down4: "downsample ×4" };
  const opBtns = (Object.keys(OPS) as Op[]).map(id => { const b = button(opRow, OPS[id], () => { st.op = id; update(); }); b.dataset.id = id; return b; });
  const sS = slider(panel, "σ (blur)", 0.5, 6, 0.5, st.sigma, v => { st.sigma = v; update(); });
  htmlEl("div", "gb-lab", "Profile shows", panel);
  const chRow = htmlEl("div", "gb-row", "", panel);
  const chBtns = ["R′", "G′", "B′", "luminance"].map((t, k) => { const b = button(chRow, t, () => { st.channel = k; update(); }); return b; });
  const linC = checkbox(panel, "show linear values directly", false, v => { st.showLinear = v; update(); });
  const read = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Click or drag on an image to choose the row of the profile. Values are 8-bit display codes; \"luminance\" is the encoded value of the luminance Y. A projector may not follow sRGB: judge 128 vs 188 on your own screen.", panel);
  const P: Record<string, Partial<typeof st>> = {
    stripes: { image: "stripes", op: "blur", sigma: 2, row: 40, channel: 3, showLinear: false },
    seam: { image: "checker", op: "blur", sigma: 3, row: 16, channel: 3, showLinear: false },
    lights: { image: "lights", op: "blur", sigma: 3, row: 64, channel: 3, showLinear: false },
    down: { image: "lines", op: "down4", sigma: 2, row: 40, channel: 3, showLinear: false },
  };
  const list: [string, string][] = [["stripes", "black + white = 128 or 188?"], ["seam", "red meets green: the dark seam"], ["lights", "blurred lights lose their glow"], ["down", "downsampling darkens fine detail"]];
  const pb = presets(panel, [...list.map(([id, label]) => ({ label, apply: () => { Object.assign(st, P[id]); sync(); update(); } })),
    { label: "reset", apply: () => { Object.assign(st, P.stripes); sync(); update(); } }]);
  function sync(): void { sS.set(st.sigma); linC.checked = st.showLinear; }

  const draw = (c: HTMLCanvasElement, I: Img) => {
    const ctx = c.getContext("2d")!, id = ctx.createImageData(N, N);
    for (let p = 0; p < N * N; p++) {
      for (let k = 0; k < 3; k++) id.data[4 * p + k] = 255 * (st.showLinear ? srgbToLinear(I[k][p]) : I[k][p]);
      id.data[4 * p + 3] = 255;
    }
    // row marker
    for (let x = 0; x < N; x += 2) { const p = st.row * N + x; id.data[4 * p] = 255; id.data[4 * p + 1] = 200; id.data[4 * p + 2] = 0; }
    ctx.putImageData(id, 0, 0);
  };
  for (const c of [cIn, cNaive, cLin]) {
    const set = (ev: PointerEvent) => { const r = c.getBoundingClientRect(); st.row = Math.max(0, Math.min(N - 1, Math.floor((ev.clientY - r.top) / r.height * N))); update(); };
    c.addEventListener("pointerdown", ev => { ev.preventDefault(); c.setPointerCapture(ev.pointerId); set(ev); const mv = (e: PointerEvent) => set(e); c.addEventListener("pointermove", mv); c.addEventListener("pointerup", () => c.removeEventListener("pointermove", mv), { once: true }); });
  }

  function profile(I: Img): number[] {
    const out: number[] = [];
    for (let x = 0; x < N; x++) {
      const p = st.row * N + x;
      if (st.channel < 3) out.push(255 * I[st.channel][p]);
      else out.push(255 * linearToSrgb(0.2126 * srgbToLinear(I[0][p]) + 0.7152 * srgbToLinear(I[1][p]) + 0.0722 * srgbToLinear(I[2][p])));
    }
    return out;
  }

  function update(): void {
    imBtns.forEach(b => b.classList.toggle("active", b.dataset.id === st.image));
    opBtns.forEach(b => b.classList.toggle("active", b.dataset.id === st.op));
    chBtns.forEach((b, k) => b.classList.toggle("active", k === st.channel));
    sS.input.disabled = st.op !== "blur";
    const I = makeImage(st.image), { naive, linear } = process(I, st.op, st.sigma);
    draw(cIn, I); draw(cNaive, naive); draw(cLin, linear);
    const xs = Array.from({ length: N }, (_, i) => i);
    lIn.set(xs, profile(I)); lNaive.set(xs, profile(naive)); lLin.set(xs, profile(linear));
    const yIn = meanLuminance(I), yN = meanLuminance(naive), yL = meanLuminance(linear);
    const lines = [`mean luminance Y (light):`, `  input            ${yIn.toFixed(4)}`, `  on encoded       ${yN.toFixed(4)}  (${(100 * (yN / yIn - 1)).toFixed(1)} %)`, `  in linear light  ${yL.toFixed(4)}  (${(100 * (yL / yIn - 1)).toFixed(1)} %)`];
    if (st.image === "stripes") {
      // centre of the stripe area, row st.row
      let a = 0, b = 0; for (let x = 16; x < 48; x++) { a += naive[0][st.row * N + x]; b += linear[0][st.row * N + x]; }
      lines.push(`stripes after the operation:`, `  on encoded ${Math.round(255 * a / 32)}, in linear light ${Math.round(255 * b / 32)}`);
    }
    read.textContent = lines.join("\n");
  }
  const pre = host.dataset.preset;
  if (pre && P[pre]) { Object.assign(st, P[pre]); sync(); pb[list.findIndex(l => l[0] === pre)]?.classList.add("active"); }
  update();
});
