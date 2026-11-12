// watershed: watershed segmentation of a 256 × 256 image (watershed day, unit ws-gradient).
// Three panels: the input (click to place markers), the landscape that is flooded (gradient
// magnitude of the smoothed image, the smoothed intensity, or its inverse) with the markers,
// and the result (regions or boundaries over the input). Markers: every regional minimum,
// the h-minima, or clicked markers (each click a new object, or more of the same object, or a
// background brush). The flood-level slider replays the flooding: only pixels flooded at or
// below that water level are coloured. Images: a synthetic test image (blobs, two touching
// pairs, a cell whose dark outline has a gap) with adjustable noise, and the course micrograph.
// Options: data-preset (raw | smooth | hmin | clicks | leak | cell).
import { autoMount, htmlEl, slider, presets, button } from "../lib/ui";
import { Grey, createGrey } from "../lib/imageops";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { labelColor, RGB } from "../lib/colour";
import { boundaries } from "../lib/watershed";
import { N, synth, CLICKS_OBJECTS, CLICKS_LEAK } from "./images";
import { Land, MarkerMode, Click, landscape, runWatershed, WsRun } from "./pipeline";

const NAME = "watershed";
type Img = "synth" | "cell";
type ClickMode = "new" | "same" | "background";
interface Preset { id: string; label: string; img: Img; land: Land; sigma: number; h: number; noise: number; mode: MarkerMode; clicks?: [number, number, number][]; show?: "regions" | "boundaries" }
const PRESETS: Preset[] = [
  { id: "raw", label: "raw gradient: oversegmentation", img: "synth", land: "gradient", sigma: 1, h: 0, noise: 8, mode: "minima" },
  { id: "smooth", label: "smoothing alone", img: "synth", land: "gradient", sigma: 4, h: 0, noise: 8, mode: "minima" },
  { id: "hmin", label: "h-minima", img: "synth", land: "gradient", sigma: 1.5, h: 2.5, noise: 8, mode: "hmin" },
  { id: "clicks", label: "one click per object: marker-controlled", img: "synth", land: "gradient", sigma: 1.5, h: 0, noise: 8, mode: "clicked", clicks: CLICKS_OBJECTS },
  { id: "leak", label: "edge with a gap: leakage", img: "synth", land: "gradient", sigma: 1.5, h: 0, noise: 8, mode: "clicked", clicks: CLICKS_LEAK },
  { id: "cell", label: "micrograph: nucleus + background clicks", img: "cell", land: "gradient", sigma: 2, h: 0, noise: 0, mode: "clicked", clicks: [[160, 150, 2], [95, 60, 1], [165, 40, 1], [230, 120, 1], [215, 240, 1], [110, 230, 1], [80, 150, 1]], show: "boundaries" },
];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { flex-direction: column; gap: .5em; }
.qb-demo[data-demo="${NAME}"] .ws-imgs { display: flex; flex-wrap: wrap; gap: .6em; width: 100%; }
.qb-demo[data-demo="${NAME}"] .ws-fig { flex: 1 1 230px; max-width: 340px; min-width: 0; margin: 0; }
.qb-demo[data-demo="${NAME}"] .ws-fig figcaption { font-size: 17px; text-align: center; margin-bottom: .15em; }
.qb-demo[data-demo="${NAME}"] .ws-fig canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; touch-action: none; border-radius: 3px; }
.qb-demo[data-demo="${NAME}"] .ws-fig canvas.click { cursor: crosshair; }
.qb-demo[data-demo="${NAME}"] .ws-ctl { display: flex; flex-wrap: wrap; gap: .5em 1.4em; width: 100%; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .ws-col { display: flex; flex-direction: column; gap: .35em; flex: 1 1 250px; min-width: 0; }
.qb-demo[data-demo="${NAME}"] .ws-lab { font-size: 15px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .ws-credit { font-size: 12px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .ws-row { display: flex; flex-wrap: wrap; gap: .35em .8em; align-items: center; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 16px; gap: .3em; }
.reveal .qb-demo[data-demo="${NAME}"] .ws-fig { flex-basis: 250px; max-width: 262px; }
.reveal .qb-demo[data-demo="${NAME}"] .ws-imgs { flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .08em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; }
.reveal .qb-demo[data-demo="${NAME}"] .ws-col { gap: .2em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const BG_RGB: RGB = [0.55, 0.55, 0.6];
const colourOf = (l: number): RGB => (l === 1 ? BG_RGB : labelColor(l));

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = {
    img: "synth" as Img, land: "gradient" as Land, sigma: 1, h: 0, noise: 8, mode: "minima" as MarkerMode,
    clicks: [] as Click[], clickMode: "new" as ClickMode, level: Infinity, show: "regions" as "regions" | "boundaries",
  };
  let input: Grey = createGrey(N, N, 128), cell: Grey | null = null, land: Float32Array = new Float32Array(N * N), res: WsRun | null = null;
  let landMax = 1, landKey = "", inputKey = "";

  const imgs = htmlEl("div", "ws-imgs", "", host);
  const fig = (title: string) => {
    const f = htmlEl("figure", "ws-fig", "", imgs);
    htmlEl("figcaption", "", title, f);
    const c = htmlEl("canvas", "", "", f);
    c.width = N; c.height = N;
    return { f, c, ctx: c.getContext("2d")! };
  };
  const pIn = fig("input (click: marker)"), pLand = fig("landscape + markers"), pRes = fig("watershed");
  pIn.c.classList.add("click"); pRes.c.classList.add("click");
  const credit = htmlEl("div", "ws-credit", "", host);

  const ctl = htmlEl("div", "ws-ctl", "", host);
  const col1 = htmlEl("div", "ws-col", "", ctl), col2 = htmlEl("div", "ws-col", "", ctl), col3 = htmlEl("div", "ws-col", "", ctl);
  const imgBtns = presets(col1, [
    { label: "test image", apply: () => { st.img = "synth"; st.clicks = []; void recompute(); } },
    { label: "micrograph", apply: () => { st.img = "cell"; st.clicks = []; void recompute(); } },
  ]);
  const LANDS: { id: Land; label: string }[] = [{ id: "gradient", label: "gradient ‖∇‖" }, { id: "intensity", label: "intensity" }, { id: "inverted", label: "inverted" }];
  const landBtns = presets(col1, LANDS.map(l => ({ label: l.label, apply: () => { st.land = l.id; void recompute(); } })));
  const sS = slider(col1, "σ (smoothing)", 0.5, 6, 0.5, 1, v => { st.sigma = v; void recompute(); });
  const nS = slider(col1, "noise sd", 0, 30, 1, 8, v => { st.noise = v; void recompute(); });

  const MODES: { id: MarkerMode; label: string }[] = [{ id: "minima", label: "all minima" }, { id: "hmin", label: "h-minima" }, { id: "clicked", label: "clicked" }];
  htmlEl("div", "ws-lab", "markers", col2);
  const modeBtns = presets(col2, MODES.map(m => ({ label: m.label, apply: () => { st.mode = m.id; void recompute(); } })));
  const hS = slider(col2, "h", 0, 10, 0.1, 0, v => { st.h = v; if (st.mode !== "hmin") st.mode = "hmin"; void recompute(); });
  const cmRow = htmlEl("div", "ws-row", "", col2);
  htmlEl("span", "ws-lab", "a click adds", cmRow);
  const CMODES: { id: ClickMode; label: string }[] = [{ id: "new", label: "new object" }, { id: "same", label: "same object" }, { id: "background", label: "background" }];
  const cmBtns = presets(cmRow, CMODES.map(m => ({ label: m.label, apply: () => { st.clickMode = m.id; draw(); } })));
  button(cmRow, "clear", () => { st.clicks = []; void recompute(); });

  const lS = slider(col3, "flood level", 0, 30, 0.1, 30, v => { st.level = v; draw(); });
  const showBtns = presets(col3, [
    { label: "regions", apply: () => { st.show = "regions"; draw(); } },
    { label: "boundaries", apply: () => { st.show = "boundaries"; draw(); } },
  ]);
  const ro = htmlEl("div", "qb-readout", "", col3);
  ro.style.whiteSpace = "pre-line";

  const prow = htmlEl("div", "ws-row", "", host);
  htmlEl("span", "ws-lab", "examples", prow);
  const presetBtns = presets(prow, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  button(prow, "reset", () => load(p0));
  const note = htmlEl("div", "qb-note", "", host);

  async function load(p: Preset): Promise<void> {
    st.img = p.img; st.land = p.land; st.sigma = p.sigma; st.h = p.h; st.noise = p.noise; st.mode = p.mode;
    st.clicks = (p.clicks ?? []).map(([x, y, label]) => ({ x, y, label, r: label === 1 ? 5 : 3 }));
    st.show = p.show ?? "regions";
    st.level = Infinity;
    sS.set(st.sigma); hS.set(st.h); nS.set(st.noise);
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    await recompute();
  }

  async function getInput(): Promise<Grey> {
    if (st.img === "synth") {
      const k = "synth" + st.noise;
      if (k !== inputKey) { input = synth(st.noise); inputKey = k; }
      return input;
    }
    if (!cell) { try { cell = await loadCell(); } catch { cell = createGrey(N, N, 128); } }
    inputKey = "cell";
    input = cell;
    return input;
  }

  async function recompute(): Promise<void> {
    const img = await getInput();
    const k = `${inputKey}|${st.land}|${st.sigma}`;
    if (k !== landKey) {
      land = landscape(img, st.land, st.sigma);
      landMax = 0;
      for (const v of land) if (v > landMax) landMax = v;
      landKey = k;
      const hmax = st.land === "gradient" ? Math.max(5, Math.ceil(landMax / 2)) : 80;
      hS.input.max = String(hmax); hS.input.step = st.land === "gradient" ? "0.1" : "1";
      lS.input.max = String(Math.ceil(landMax * 10) / 10); lS.input.step = landMax > 50 ? "1" : "0.1";
    }
    if (st.mode === "clicked" && !st.clicks.length) res = null;
    else res = runWatershed(land, N, N, st.mode, st.h, st.clicks);
    draw();
  }

  function draw(): void {
    imgBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.img === "synth")));
    landBtns.forEach((b, i) => b.classList.toggle("active", LANDS[i].id === st.land));
    modeBtns.forEach((b, i) => b.classList.toggle("active", MODES[i].id === st.mode));
    cmBtns.forEach((b, i) => b.classList.toggle("active", CMODES[i].id === st.clickMode));
    showBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.show === "regions")));
    (nS.input.closest("label") as HTMLElement).style.opacity = st.img === "synth" ? "1" : "0.4";
    (hS.input.closest("label") as HTMLElement).style.opacity = st.mode === "hmin" ? "1" : "0.4";
    const lev = Math.min(st.level, landMax);
    lS.set(Math.round(lev * 10) / 10);
    credit.textContent = st.img === "cell" ? CELL_CREDIT : `Synthetic test image (seeded), noise sd ${st.noise}.`;
    const n = N * N;
    // input with clicked markers
    const a = pIn.ctx.createImageData(N, N), b = pLand.ctx.createImageData(N, N), c = pRes.ctx.createImageData(N, N);
    const mk = res?.markers;
    let lo = Infinity, hi = -Infinity;
    for (const v of land) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const gam = st.land === "gradient" ? 0.6 : 1;
    const bnd = res ? boundaries(res.labels, N, N) : null;
    for (let i = 0; i < n; i++) {
      const v = Math.max(0, Math.min(255, input.data[i]));
      const j = 4 * i;
      a.data[j] = a.data[j + 1] = a.data[j + 2] = v; a.data[j + 3] = 255;
      if (st.mode === "clicked" && mk && mk[i]) { const col = colourOf(mk[i]); a.data[j] = 255 * col[0]; a.data[j + 1] = 255 * col[1]; a.data[j + 2] = 255 * col[2]; }
      const t = hi > lo ? Math.pow((land[i] - lo) / (hi - lo), gam) : 0, g = 255 * (1 - t);
      b.data[j] = b.data[j + 1] = b.data[j + 2] = g; b.data[j + 3] = 255;
      if (mk && mk[i]) {
        const col: RGB = st.mode === "clicked" ? colourOf(mk[i]) : [0.85, 0.1, 0.1], al = st.mode === "hmin" ? 0.45 : 1;
        b.data[j] = g * (1 - al) + 255 * col[0] * al; b.data[j + 1] = g * (1 - al) + 255 * col[1] * al; b.data[j + 2] = g * (1 - al) + 255 * col[2] * al;
      }
      c.data[j] = c.data[j + 1] = c.data[j + 2] = v; c.data[j + 3] = 255;
      if (res) {
        const L = res.labels[i], flooded = L > 0 && !(res.level[i] > lev);
        if (flooded && st.show === "regions") {
          const col = colourOf(L), al = L === 1 ? 0.35 : 0.6;
          c.data[j] = v * (1 - al) + 255 * col[0] * al; c.data[j + 1] = v * (1 - al) + 255 * col[1] * al; c.data[j + 2] = v * (1 - al) + 255 * col[2] * al;
        } else if (flooded && bnd && bnd[i]) { c.data[j] = 255; c.data[j + 1] = 40; c.data[j + 2] = 40; }
      }
    }
    pIn.ctx.putImageData(a, 0, 0); pLand.ctx.putImageData(b, 0, 0); pRes.ctx.putImageData(c, 0, 0);
    if (!res) {
      ro.textContent = "no markers yet";
      note.textContent = "Clicked markers: click in the input or result image (\"new object\" for each object, \"background\" and drag for the background).";
      return;
    }
    const objs = st.mode === "clicked" ? `${res.nRegions - (res.markers.includes(1) ? 1 : 0)} objects + background` : `${res.nRegions}`;
    ro.textContent = `regional minima: ${res.nMinima}\nmarkers: ${res.nMarkers} · regions: ${objs}\nmarkers + flooding: ${res.ms.toFixed(0)} ms`;
    const notes: Record<MarkerMode, string> = {
      minima: "Every regional minimum of the landscape (red) starts a basin: noise makes many minima, so many regions.",
      hmin: `h-minima: every basin is filled by h = ${st.h} first; only minima deeper than h start a basin.`,
      clicked: "Markers (coloured) are the only sources; the flooding only decides where the boundaries go. Grey = background marker.",
    };
    note.textContent = notes[st.mode] + (lev < landMax ? ` Replay: only pixels flooded at water level ≤ ${lev.toFixed(1)} are coloured.` : "");
  }

  // clicks
  let painting = false;
  const toPix = (cv: HTMLCanvasElement, ev: PointerEvent): [number, number] => {
    const r = cv.getBoundingClientRect();
    return [Math.floor((ev.clientX - r.left) / r.width * N), Math.floor((ev.clientY - r.top) / r.height * N)];
  };
  const addClick = (x: number, y: number) => {
    if (x < 0 || y < 0 || x >= N || y >= N) return;
    if (st.mode !== "clicked") { st.mode = "clicked"; st.clicks = []; }
    const objLabels = st.clicks.map(c => c.label).filter(l => l > 1);
    const last = objLabels.length ? Math.max(...objLabels) : 1;
    const label = st.clickMode === "background" ? 1 : st.clickMode === "same" && last > 1 ? last : last + 1;
    st.clicks.push({ x, y, label, r: label === 1 ? 5 : 3 });
  };
  for (const cv of [pIn.c, pRes.c]) {
    cv.addEventListener("pointerdown", ev => {
      ev.preventDefault();
      const [x, y] = toPix(cv, ev);
      addClick(x, y);
      if (st.clickMode === "background") { painting = true; cv.setPointerCapture(ev.pointerId); draw(); }
      else void recompute();
    });
    cv.addEventListener("pointermove", ev => {
      if (!painting) return;
      const [x, y] = toPix(cv, ev), last = st.clicks[st.clicks.length - 1];
      if (last && Math.hypot(last.x - x, last.y - y) < 4) return;
      addClick(x, y);
      draw();
    });
    const end = () => { if (painting) { painting = false; void recompute(); } };
    cv.addEventListener("pointerup", end);
    cv.addEventListener("pointercancel", end);
  }

  void load(p0);
}

autoMount(NAME, mount);
