// tophat: grey-level morphology on 256 × 256 images (morphology day, unit morph-grey).
// Three panels: the input (with a draggable profile line), the result of the chosen
// operation with a flat disc of radius r, and a threshold overlay. The threshold can be
// applied to the input or to the result; for the synthetic image the true spots are known,
// so the readout counts found and false detections. The profile under the line is plotted
// below (as in the 1D demo morph1d).
// Options: data-preset (fails | flatten | small | large | granules | gradient), data-profile ("0" hides it).
import { autoMount, htmlEl, slider, presets, button, checkbox, svgEl, svgPoint, draggable } from "../lib/ui";
import { Grey, createGrey, drawGrey, sobel, fromArray } from "../lib/imageops";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { nuclei } from "../lib/synth";
import { label2d } from "../lib/label";
import { Plot } from "../lib/plot";
import { erodeDisc, dilateDisc, openDisc, closeDisc } from "./fastdisc";
import { spotsOnSlope, SpotImage } from "./images";

const NAME = "tophat";
type ImgId = "spots" | "cell" | "nuclei";
type OpId = "erode" | "dilate" | "open" | "close" | "wth" | "bth" | "grad" | "sobel";
const OPS: { id: OpId; label: string }[] = [
  { id: "erode", label: "erosion" }, { id: "dilate", label: "dilation" }, { id: "open", label: "opening" }, { id: "close", label: "closing" },
  { id: "wth", label: "white top-hat" }, { id: "bth", label: "black top-hat" }, { id: "grad", label: "gradient" }, { id: "sobel", label: "Sobel" },
];
const IMGS: { id: ImgId; label: string }[] = [
  { id: "spots", label: "spots on a slope" }, { id: "cell", label: "cell (micrograph)" }, { id: "nuclei", label: "nuclei" },
];
interface Preset { id: string; label: string; img: ImgId; op: OpId; r: number; on: "input" | "result"; t: number }
const PRESETS: Preset[] = [
  { id: "fails", label: "one threshold fails", img: "spots", op: "wth", r: 6, on: "input", t: 150 },
  { id: "flatten", label: "top-hat flattens the background", img: "spots", op: "wth", r: 6, on: "result", t: 28 },
  { id: "small", label: "SE too small: spots shrink", img: "spots", op: "wth", r: 2, on: "result", t: 28 },
  { id: "large", label: "SE too large: background leaks in", img: "spots", op: "wth", r: 20, on: "result", t: 28 },
  { id: "granules", label: "granules in the cell", img: "cell", op: "bth", r: 4, on: "result", t: 80 },
  { id: "gradient", label: "morphological gradient vs Sobel", img: "nuclei", op: "grad", r: 1, on: "result", t: 60 },
];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; }
.qb-demo[data-demo="${NAME}"] .th-main { flex: 1 1 760px; min-width: 0; max-width: 800px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .th-row { display: flex; gap: .6em; }
.qb-demo[data-demo="${NAME}"] .th-fig { flex: 1 1 0; min-width: 0; margin: 0; position: relative; }
.qb-demo[data-demo="${NAME}"] .th-fig figcaption { font-size: 16px; text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.qb-demo[data-demo="${NAME}"] .th-fig canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; }
.qb-demo[data-demo="${NAME}"] .th-fig svg.th-over { position: absolute; left: 0; bottom: 0; width: 100%; aspect-ratio: 1; touch-action: none; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 320px; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .th-lab { font-size: 15px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .th-credit { font-size: 12px; color: var(--qb-muted); }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const SZ = 256;

function sample(img: Grey, x: number, y: number): number {
  const xi = Math.max(0, Math.min(img.width - 1, Math.round(x))), yi = Math.max(0, Math.min(img.height - 1, Math.round(y)));
  return img.data[yi * img.width + xi];
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[1];
  const showProfile = host.dataset.profile !== "0";
  const st = { img: p0.img, op: p0.op, r: p0.r, on: p0.on, t: p0.t, line: [24, 70, 232, 190] as number[] };
  let synth: SpotImage | null = null, cell: Grey | null = null, nuc: Grey | null = null;
  let input: Grey = createGrey(SZ, SZ), result: Grey = createGrey(SZ, SZ), bgEst: Grey | null = null;

  const main = htmlEl("div", "th-main", "", host);
  const row = htmlEl("div", "th-row", "", main);
  const fig = (cap: string) => { const f = htmlEl("figure", "th-fig", "", row); const c = htmlEl("figcaption", "", cap, f); const cv = htmlEl("canvas", "", "", f); cv.width = SZ; cv.height = SZ; return { f, c, cv }; };
  const A = fig("input"), B = fig("result"), C = fig("threshold");
  const over = svgEl("svg", { class: "th-over", viewBox: `0 0 ${SZ} ${SZ}` }, A.f);
  const pl = svgEl("line", { stroke: "#ff7f0e", "stroke-width": 2.5 }, over);
  const h1 = svgEl("circle", { r: 8, fill: "#fff", stroke: "#ff7f0e", "stroke-width": 3 }, over);
  const h2 = svgEl("circle", { r: 8, fill: "#fff", stroke: "#ff7f0e", "stroke-width": 3 }, over);
  const credit = htmlEl("div", "th-credit", "", main);
  let prof: Plot | null = null, lIn: ReturnType<Plot["line"]> | null = null, lBg: ReturnType<Plot["line"]> | null = null, lRes: ReturnType<Plot["line"]> | null = null, lT: ReturnType<Plot["hline"]> | null = null;
  const profBox = htmlEl("div", "", "", main);
  if (showProfile) {
    prof = new Plot(profBox, { width: 800, height: 210, x: [0, 1], y: [0, 255], xlabel: "position along the orange line (pixels)", ylabel: "grey", ariaLabel: "profile" });
    lIn = prof.line([], [], { color: "var(--qb-fg, #222)", width: 2.5, label: "input" });
    lBg = prof.line([], [], { color: "#ff7f0e", width: 3, label: "opening / closing" });
    lRes = prof.line([], [], { color: "#2ca02c", width: 2.5, label: "result" });
    lT = prof.hline(0, { color: "#d62728", width: 1.5 });
  } else { over.style.display = "none"; }

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "th-lab", "image", panel);
  const imgBtns = presets(panel, IMGS.map(m => ({ label: m.label, apply: () => { st.img = m.id; recompute(); } })));
  htmlEl("div", "th-lab", "operation (flat disc of radius r)", panel);
  const opBtns = presets(panel, OPS.map(m => ({ label: m.label, apply: () => { st.op = m.id; recompute(); } })));
  const rS = slider(panel, "r", 1, 20, 1, st.r, v => { st.r = v; recompute(); });
  htmlEl("div", "th-lab", "threshold applied to the", panel);
  const onBtns = presets(panel, [
    { label: "input", apply: () => { st.on = "input"; draw(); } },
    { label: "result", apply: () => { st.on = "result"; draw(); } },
  ]);
  const tS = slider(panel, "t", 0, 255, 1, st.t, v => { st.t = v; draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  ro.style.whiteSpace = "pre-line";
  htmlEl("div", "th-lab", "examples", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  button(panel, "reset", () => load(p0));

  async function getInput(): Promise<Grey> {
    if (st.img === "spots") { synth ??= spotsOnSlope(); return synth.image; }
    if (st.img === "nuclei") { nuc ??= nuclei({ seed: 3, count: 22, gradient: 0 }).image; return nuc; }
    if (!cell) { try { cell = await loadCell(); } catch { cell = createGrey(SZ, SZ, 128); } }
    return cell;
  }

  async function load(p: Preset): Promise<void> {
    st.img = p.img; st.op = p.op; st.r = p.r; st.on = p.on; st.t = p.t;
    rS.set(p.r); tS.set(p.t);
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    await recompute();
  }

  async function recompute(): Promise<void> {
    input = await getInput();
    const f = input.data, r = st.r;
    let out: Float32Array;
    bgEst = null;
    switch (st.op) {
      case "erode": out = erodeDisc(f, SZ, SZ, r); break;
      case "dilate": out = dilateDisc(f, SZ, SZ, r); break;
      case "open": out = openDisc(f, SZ, SZ, r); break;
      case "close": out = closeDisc(f, SZ, SZ, r); break;
      case "wth": { const o = openDisc(f, SZ, SZ, r); bgEst = fromArray(SZ, SZ, o); out = Float32Array.from(f, (v, i) => v - o[i]); break; }
      case "bth": { const c = closeDisc(f, SZ, SZ, r); bgEst = fromArray(SZ, SZ, c); out = Float32Array.from(f, (v, i) => c[i] - v); break; }
      case "grad": { const d = dilateDisc(f, SZ, SZ, r), e = erodeDisc(f, SZ, SZ, r); out = Float32Array.from(d, (v, i) => v - e[i]); break; }
      default: out = Float32Array.from(sobel(input).mag.data, v => v / 4); break;
    }
    result = fromArray(SZ, SZ, out);
    imgBtns.forEach((b, i) => b.classList.toggle("active", IMGS[i].id === st.img));
    opBtns.forEach((b, i) => b.classList.toggle("active", OPS[i].id === st.op));
    credit.textContent = st.img === "cell" ? CELL_CREDIT : st.img === "spots" ? "Synthetic: 40 bright spots and 25 dark specks on a sloping, vignetted background, noise σ = 4." : "Synthetic nuclei.";
    draw();
  }

  function draw(): void {
    onBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.on === "input")));
    drawGrey(A.cv, input, { mode: "clip", lo: 0, hi: 255 });
    const absolute = st.op === "erode" || st.op === "dilate" || st.op === "open" || st.op === "close";
    drawGrey(B.cv, result, absolute ? { mode: "clip", lo: 0, hi: 255 } : { mode: "clip", lo: 0, hi: Math.max(40, maxOf(result.data)) });
    B.c.textContent = OPS.find(o => o.id === st.op)!.label + (st.op === "sobel" ? " magnitude / 4" : `, r = ${st.r}`);
    // threshold overlay
    const src = st.on === "input" ? input : result, mask = new Uint8Array(SZ * SZ);
    for (let i = 0; i < mask.length; i++) mask[i] = src.data[i] > st.t ? 1 : 0;
    const id = new ImageData(SZ, SZ);
    for (let i = 0; i < mask.length; i++) {
      const g = Math.max(0, Math.min(255, input.data[i])) * 0.8 + 30;
      if (mask[i]) { id.data[4 * i] = 255; id.data[4 * i + 1] = 120; id.data[4 * i + 2] = 20; }
      else { id.data[4 * i] = g; id.data[4 * i + 1] = g; id.data[4 * i + 2] = g; }
      id.data[4 * i + 3] = 255;
    }
    C.cv.getContext("2d")!.putImageData(id, 0, 0);
    C.c.textContent = `${st.on} > ${st.t}`;
    // count detections
    const lab = label2d(mask, SZ, SZ, 8);
    let txt = `${lab.count} objects above the threshold`;
    if (st.img === "spots" && synth) {
      const targets = st.op === "bth" && st.on === "result" ? synth.dark : synth.bright;
      const hitLabels = new Set<number>();
      let found = 0;
      for (const p of targets) {
        let l = 0;
        for (let dy = -1; dy <= 1 && !l; dy++) for (let dx = -1; dx <= 1 && !l; dx++) {
          const x = Math.round(p.x) + dx, y = Math.round(p.y) + dy;
          if (x >= 0 && y >= 0 && x < SZ && y < SZ) l = lab.data[y * SZ + x];
        }
        if (l) { found++; hitLabels.add(l); }
      }
      const kind = targets === synth.dark ? "dark specks" : "bright spots";
      txt += `\n${kind} found: ${found} of ${targets.length}\nfalse objects: ${lab.count - hitLabels.size}`;
    }
    ro.textContent = txt;
    // profile
    const [x0, y0, x1, y1] = st.line;
    setLine();
    if (prof && lIn && lBg && lRes && lT) {
      const L = Math.max(1, Math.round(Math.hypot(x1 - x0, y1 - y0))), xs: number[] = [], a: number[] = [], b: number[] = [], c: number[] = [];
      for (let k = 0; k <= L; k++) {
        const x = x0 + (x1 - x0) * k / L, y = y0 + (y1 - y0) * k / L;
        xs.push(k); a.push(sample(input, x, y)); c.push(sample(result, x, y)); if (bgEst) b.push(sample(bgEst, x, y));
      }
      prof.setDomain([0, L], [Math.min(0, ...c), 255]);
      lIn.set(xs, a); lBg.set(bgEst ? xs : [], b); lRes.set(xs, c);
      lT.set(st.t);
      prof.legend([{ label: "input", color: "var(--qb-fg, #222)" }, ...(bgEst ? [{ label: st.op === "wth" ? "opening" : "closing", color: "#ff7f0e" }] : []), { label: "result", color: "#2ca02c" }, { label: "threshold", color: "#d62728" }], "tr");
    }
  }

  function setLine(): void {
    const [x0, y0, x1, y1] = st.line;
    pl.setAttribute("x1", String(x0)); pl.setAttribute("y1", String(y0)); pl.setAttribute("x2", String(x1)); pl.setAttribute("y2", String(y1));
    h1.setAttribute("cx", String(x0)); h1.setAttribute("cy", String(y0)); h2.setAttribute("cx", String(x1)); h2.setAttribute("cy", String(y1));
  }
  const clamp = (v: number) => Math.max(0, Math.min(SZ - 1, v));
  draggable(h1, ev => svgPoint(over, ev), (x, y) => { st.line[0] = clamp(x); st.line[1] = clamp(y); draw(); });
  draggable(h2, ev => svgPoint(over, ev), (x, y) => { st.line[2] = clamp(x); st.line[3] = clamp(y); draw(); });

  void load(p0);
}

function maxOf(a: ArrayLike<number>): number { let m = -Infinity; for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i]; return m; }

autoMount(NAME, mount);
