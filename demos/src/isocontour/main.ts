// Iso-contours of a real image: marching squares (lib/marching.ts) on a 128 × 128 image, the
// course micrograph or a synthetic image (two smooth blobs that merge as c drops, and two
// discs joined by a thin neck). Slider/histogram for the iso-value c, Gaussian blur σ before
// extraction, added noise. Readouts: number of closed contours, total length, the area they
// enclose (shoelace) and the number of pixels above c. The image is padded with a dark border,
// so every contour closes.
// Auto-mounts on <div data-demo="isocontour">. Optional: data-preset="blobs|noise|blur|merge|cell".
import { htmlEl, slider, presets, button, autoMount, svgEl, checkbox } from "../lib/ui";
import { Grey, fromFunction, gaussianBlur, drawGrey } from "../lib/imageops";
import { addGaussianNoise } from "../lib/synth";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { marchingSquares, padGrid, contourLength, lineArea, lineLength } from "../lib/marching";
import { histC } from "./histc";

const NAME = "isocontour";
const N = 128;

interface Preset { id: string; label: string; image: "synth" | "cell"; c: number; sigma: number; noise: number }
const PRESETS: Preset[] = [
  { id: "blobs", label: "test image", image: "synth", c: 120, sigma: 0, noise: 0 },
  { id: "noise", label: "noise: many tiny contours", image: "synth", c: 120, sigma: 0, noise: 25 },
  { id: "blur", label: "blur first", image: "synth", c: 120, sigma: 2, noise: 25 },
  { id: "merge", label: "two blobs merge as c drops", image: "synth", c: 140, sigma: 0, noise: 0 },
  { id: "cell", label: "micrograph", image: "cell", c: 100, sigma: 1, noise: 0 },
];

/** The synthetic image: two Gaussian blobs (saddle value between them ≈ 128) and two discs
 *  joined by a 3-pixel neck, background 30. */
export function synthImage(): Grey {
  const disc = (x: number, y: number, cx: number, cy: number, r: number) => 0.5 * (1 + Math.tanh((r - Math.hypot(x - cx, y - cy)) / 0.8));
  const neck = (x: number, y: number) => (x > 40 && x < 92 ? 0.5 * (1 + Math.tanh((1.5 - Math.abs(y - 98)) / 0.6)) : 0);
  return fromFunction(N, N, (x, y) => {
    const g = 175 * (Math.exp(-((x - 42) ** 2 + (y - 42) ** 2) / (2 * 11 ** 2)) + Math.exp(-((x - 82) ** 2 + (y - 46) ** 2) / (2 * 11 ** 2)));
    const d = Math.max(disc(x, y, 34, 98, 12), disc(x, y, 96, 98, 13), neck(x, y));
    return 30 + g + 170 * d;
  });
}

const STYLE = `
[data-demo="${NAME}"] { gap: 1em; }
[data-demo="${NAME}"] .ic-img { flex: 0 1 430px; min-width: 240px; position: relative; margin: 0; }
[data-demo="${NAME}"] .ic-img canvas, [data-demo="${NAME}"] .ic-img svg { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .ic-img svg { position: absolute; left: 0; top: 0; }
[data-demo="${NAME}"] .ic-credit { font-size: 12px; color: var(--qb-muted); }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 320px; max-width: 440px; gap: .35em; }
[data-demo="${NAME}"] .ic-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .ic-img { flex-basis: 420px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = { image: p0.image, c: p0.c, sigma: p0.sigma, noise: p0.noise, contours: true };

  const fig = htmlEl("figure", "ic-img", "", host);
  const cv = htmlEl("canvas", "", "", fig);
  cv.width = N; cv.height = N;
  const svg = svgEl("svg", { viewBox: `0 0 ${N} ${N}` }, fig);
  const path = svgEl("path", { fill: "none", stroke: "#ffd400", "stroke-width": 2.2, "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round" }, svg);
  const credit = htmlEl("div", "ic-credit", CELL_CREDIT, host.querySelector(".ic-img")!);
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "ic-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const H = histC(panel, { lo: 0, hi: 255, c: st.c, onChange: v => { st.c = Math.round(v); cS.set(st.c); run(); }, fmt: v => v.toFixed(0) });
  const cS = slider(panel, "iso-value c", 0, 255, 1, st.c, v => { st.c = v; H.setC(v); run(); });
  const sS = slider(panel, "blur σ", 0, 4, 0.25, st.sigma, v => { st.sigma = v; makeImage(); });
  const nS = slider(panel, "noise", 0, 50, 1, st.noise, v => { st.noise = v; makeImage(); });
  checkbox(panel, "show contours", st.contours, v => { st.contours = v; run(); });
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Yellow: the contours at level c (marching squares, linear interpolation, asymptotic decider). Area: the shoelace formula on the closed contours; holes count negative.", panel);

  function load(p: Preset): void {
    st.image = p.image; st.c = p.c; st.sigma = p.sigma; st.noise = p.noise;
    cS.set(st.c); sS.set(st.sigma); nS.set(st.noise); H.setC(st.c);
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    makeImage();
  }

  let cell: Grey | null = null;
  const synth = synthImage();
  let img: Grey = synth;
  function makeImage(): void {
    if (st.image === "cell" && !cell) { loadCell(N).then(g => { cell = g; makeImage(); }); return; }
    let g = st.image === "cell" ? cell! : synth;
    if (st.noise > 0) g = addGaussianNoise(g, st.noise, 11);
    if (st.sigma > 0) g = gaussianBlur(g, st.sigma);
    img = g;
    credit.style.visibility = st.image === "cell" ? "visible" : "hidden";
    drawGrey(cv, img, { mode: "clip", lo: 0, hi: 255 });
    H.setValues(img.data, 0, 255);
    run();
  }

  function run(): void {
    const t0 = performance.now();
    const P = padGrid(img.data, N, N, -1);
    const C = marchingSquares(P.values, P.w, P.h, st.c, { saddle: "decider" });
    const ms = performance.now() - t0;
    let d = "";
    if (st.contours) for (const L of C.lines) {
      d += L.ids.map((id, i) => `${i ? "L" : "M"}${(C.points[2 * id] - 1 + 0.5).toFixed(2)},${(C.points[2 * id + 1] - 1 + 0.5).toFixed(2)}`).join("") + (L.closed ? "Z" : "");
    }
    path.setAttribute("d", d);
    let above = 0;
    for (let i = 0; i < img.data.length; i++) if (img.data[i] > st.c) above++;
    const areas = C.lines.map(L => lineArea(C, L));
    const outer = areas.filter(a => a > 0).length, holes = areas.filter(a => a < 0).length;
    const area = areas.reduce((a, b) => a + b, 0);
    const longest = Math.max(0, ...C.lines.map(L => lineLength(C, L)));
    ro.textContent = [
      `contours        ${String(C.lines.length).padStart(7)}  (${outer} outer, ${holes} holes)`,
      `total length    ${contourLength(C).toFixed(1).padStart(7)} px  (longest ${longest.toFixed(1)})`,
      `enclosed area   ${area.toFixed(1).padStart(7)} px²  (shoelace)`,
      `pixels above c  ${String(above).padStart(7)}`,
      `${ms.toFixed(1)} ms`,
    ].join("\n");
  }

  load(p0);
}

autoMount(NAME, mount);
