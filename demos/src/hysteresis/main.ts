// Hysteresis thresholding. Two thresholds t_low < t_high: pixels above t_high are seeds
// (red); pixels above t_low are kept (orange) if they are connected (4 or 8) to a seed,
// and rejected (blue) otherwise. Top: a 1D profile (a fading structure plus noise) with
// both thresholds as draggable lines. Bottom: a 128 × 128 image of faint, fading
// filaments with noise (synthetic, seeded), or the edge strength (Sobel) of the course
// micrograph, next to the two single thresholds for comparison.
// Auto-mounts on <div data-demo="hysteresis">. Options: data-preset="high|low|hyst|conn|edges".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { Grey } from "../lib/imageops";
import { hysteresis, hysteresis1d } from "../lib/threshold";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { injectStyle, RGB } from "../histthresh/scenes";
import { N, fadingFilaments, profile1d, edgeStrength } from "./data";

const NAME = "hysteresis";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] figure { margin: 0; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin: 0 0 .2em; }
[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .hy-main { flex: 0 1 300px; min-width: 220px; }
[data-demo="${NAME}"] .hy-mid { flex: 1 1 380px; min-width: 260px; max-width: 460px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .hy-mid svg { touch-action: none; cursor: ns-resize; }
[data-demo="${NAME}"] .hy-row { display: flex; gap: .6em; }
[data-demo="${NAME}"] .hy-row figure { flex: 1 1 0; min-width: 0; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 270px; gap: .35em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; }
[data-demo="${NAME}"] .hy-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .hy-key span { display: inline-block; width: .9em; height: .9em; border-radius: 2px; margin: 0 .25em 0 .5em; vertical-align: -1px; }
[data-demo="${NAME}"] .hy-credit { font-size: 12px; color: var(--qb-muted); }
.reveal [data-demo="${NAME}"] { font-size: 16px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .1em .4em; }
`;

const SEED: RGB = [214, 39, 40], KEPT: RGB = [255, 160, 30], REJ: RGB = [60, 120, 230];

interface State { source: "filaments" | "edges"; lo: number; hi: number; conn: 4 | 8 }
const PRESETS: { id: string; label: string; s: State }[] = [
  { id: "high", label: "high alone breaks the filament", s: { source: "filaments", lo: 105, hi: 105, conn: 8 } },
  { id: "low", label: "low alone keeps noise", s: { source: "filaments", lo: 62, hi: 62, conn: 8 } },
  { id: "hyst", label: "hysteresis keeps both ends", s: { source: "filaments", lo: 62, hi: 105, conn: 8 } },
  { id: "conn", label: "4- vs 8-connectivity", s: { source: "filaments", lo: 62, hi: 105, conn: 4 } },
  { id: "edges", label: "edge strength of the micrograph", s: { source: "edges", lo: 120, hi: 220, conn: 8 } },
];

function paint(c: HTMLCanvasElement, img: Grey, layers: { m: ArrayLike<number>; rgb: RGB }[]): void {
  c.width = img.width; c.height = img.height;
  const ctx = c.getContext("2d")!, id = ctx.createImageData(img.width, img.height);
  for (let i = 0; i < img.data.length; i++) {
    const g = 0.6 * img.data[i];
    let col: RGB = [g, g, g];
    for (const L of layers) if (L.m[i]) col = L.rgb;
    id.data[4 * i] = col[0]; id.data[4 * i + 1] = col[1]; id.data[4 * i + 2] = col[2]; id.data[4 * i + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[2];
  const st: State = { ...p0.s };

  const main = htmlEl("figure", "hy-main", "", host);
  htmlEl("figcaption", "", "hysteresis", main);
  const cv = htmlEl("canvas", "", "", main);
  const key = htmlEl("div", "hy-key qb-note", "", main);
  key.innerHTML = `<span style="background:rgb(${SEED})"></span>seed &gt; t<sub>high</sub><span style="background:rgb(${KEPT})"></span>kept<span style="background:rgb(${REJ})"></span>rejected`;
  const credit = htmlEl("div", "hy-credit", CELL_CREDIT, main);
  const mid = htmlEl("div", "hy-mid", "", host);
  const plot = new Plot(mid, { width: 460, height: 230, x: [0, 199], y: [0, 230], xlabel: "position along a 1D profile", ylabel: "intensity", xticks: [0, 50, 100, 150] });
  const row = htmlEl("div", "hy-row", "", mid);
  const fh = htmlEl("figure", "", "", row); htmlEl("figcaption", "", "single threshold t_high", fh);
  const cvH = htmlEl("canvas", "", "", fh);
  const fl = htmlEl("figure", "", "", row); htmlEl("figcaption", "", "single threshold t_low", fl);
  const cvL = htmlEl("canvas", "", "", fl);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "hy-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const loS = slider(panel, "t_low", 0, 254, 1, st.lo, v => { st.lo = v; if (st.hi < v) st.hi = v; run(); });
  const hiS = slider(panel, "t_high", 0, 254, 1, st.hi, v => { st.hi = v; if (st.lo > v) st.lo = v; run(); });
  htmlEl("div", "hy-lab", "connectivity", panel);
  const cRow = htmlEl("div", "qb-presets", "", panel);
  const cBtns = ([4, 8] as const).map(k => button(cRow, `${k}-connected`, () => { st.conn = k; run(); }));
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);

  function load(p: typeof PRESETS[number]): void {
    Object.assign(st, p.s);
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    makeImage();
  }

  const fil = fadingFilaments(4), prof = profile1d(2);
  let edges: Grey | null = null, img = fil;
  function makeImage(): void {
    if (st.source === "edges" && !edges) {
      loadCell(N).then(c => {
        edges = edgeStrength(c);
        makeImage();
      });
      return;
    }
    img = st.source === "edges" ? edges! : fil;
    credit.style.display = st.source === "edges" ? "" : "none";
    run();
  }

  function run(): void {
    loS.set(st.lo); hiS.set(st.hi);
    cBtns.forEach((b, i) => b.classList.toggle("active", [4, 8][i] === st.conn));
    const h = hysteresis(img.data, N, N, st.lo, st.hi, st.conn);
    const rej = Uint8Array.from(h.lowMask, (v, i) => (v && !h.mask[i] ? 1 : 0));
    paint(cv, img, [{ m: rej, rgb: REJ }, { m: h.mask, rgb: KEPT }, { m: h.seeds, rgb: SEED }]);
    paint(cvH, img, [{ m: h.seeds, rgb: SEED }]);
    paint(cvL, img, [{ m: h.lowMask, rgb: KEPT }]);
    let nk = 0, ns = 0, nr = 0, nl = 0;
    for (let i = 0; i < N * N; i++) { nk += h.mask[i]; ns += h.seeds[i]; nr += rej[i]; nl += h.lowMask[i]; }
    ro.textContent = `seeds (> t_high): ${ns} px\nfinal mask: ${nk} px\nrejected: ${nr} of ${nl} px > t_low`;
    // 1D
    const xs = Array.from({ length: prof.length }, (_, i) => i), keep = hysteresis1d(prof, st.lo, st.hi);
    plot.clear();
    plot.line(xs, prof, { color: "var(--qb-muted, #888)", width: 1.5 });
    const cols = xs.map(i => (prof[i] > st.hi ? `rgb(${SEED})` : keep[i] ? `rgb(${KEPT})` : prof[i] > st.lo ? `rgb(${REJ})` : "var(--qb-grid, #ccc)"));
    plot.scatter(xs, prof, { colors: cols, r: 3.6, stroke: "none" });
    plot.hline(st.hi, { color: `rgb(${SEED})`, width: 3, dash: "", text: "t_high" });
    plot.hline(st.lo, { color: `rgb(${REJ})`, width: 3, dash: "", text: "t_low" });
  }

  // drag the nearer threshold line on the 1D plot
  const svg = plot.svg;
  let which: "lo" | "hi" | null = null;
  const setY = (ev: PointerEvent) => {
    const y = Math.max(0, Math.min(254, Math.round(plot.toData(ev)[1])));
    if (!which) which = Math.abs(y - st.lo) <= Math.abs(y - st.hi) ? "lo" : "hi";
    if (which === "lo") { st.lo = Math.min(y, st.hi); } else { st.hi = Math.max(y, st.lo); }
    presetBtns.forEach(b => b.classList.remove("active"));
    run();
  };
  svg.addEventListener("pointerdown", ev => { ev.preventDefault(); svg.setPointerCapture(ev.pointerId); which = null; setY(ev); });
  svg.addEventListener("pointermove", ev => { if (which) setY(ev); });
  svg.addEventListener("pointerup", () => { which = null; });
  svg.addEventListener("pointercancel", () => { which = null; });

  load(p0);
}

autoMount(NAME, mount);
