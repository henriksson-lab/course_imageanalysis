// Colour maps for scalar data, and channel merges. A grey-level image (a smooth height
// field, a fluorescence-like image, a signed difference) through a chosen colour map, with
// the colour bar and its lightness profile L*(t) underneath: jet's L* goes up and down
// (false edges), viridis/cividis/magma rise monotonically, the diverging map has its
// lightest point at zero. Display range sliders; simulate a dichromat (protan/deutan/
// tritan, rank-2 projections in linear RGB) on everything shown. Mode "merge": two
// fluorescence channels merged additively in red/green or magenta/green.
// Auto-mounts on <div data-demo="colourmap">. Options: data-mode (map|merge),
// data-image (height|fluor|signed), data-map, data-preset (jet|viridis|merge|signed).
import { htmlEl, slider, presets, button, autoMount, rng } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { ColorMap, grey, viridis, inferno, diverging, lightnessProfile } from "../lib/colour";
import { jet, rainbow, cividis, magma, simulateDichromatBytes, Dichromacy } from "../colourcloud/colourx";

const NAME = "colourmap";
const N = 160;
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] .cmap-img { flex: 0 1 330px; min-width: 200px; margin: 0; }
[data-demo="${NAME}"] .cmap-img canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 3px; }
[data-demo="${NAME}"] .cmap-mid { flex: 1 1 330px; min-width: 240px; max-width: 420px; display: flex; flex-direction: column; gap: .2em; }
[data-demo="${NAME}"] canvas.cmap-bar { width: 100%; height: 34px; display: block; border: 1px solid #888; box-sizing: border-box; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 250px; max-width: 420px; gap: .35em; }
[data-demo="${NAME}"] .cmap-row { display: flex; flex-wrap: wrap; gap: .3em; }
[data-demo="${NAME}"] .cmap-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .42em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
`;

export const MAPS: Record<string, { label: string; map: ColorMap }> = {
  grey: { label: "grey", map: grey }, viridis: { label: "viridis", map: viridis }, cividis: { label: "cividis", map: cividis },
  magma: { label: "magma", map: magma }, inferno: { label: "inferno", map: inferno }, jet: { label: "jet (rainbow)", map: jet },
  rainbow: { label: "hue circle", map: rainbow }, diverging: { label: "diverging", map: diverging },
};
type ImageId = "height" | "fluor" | "signed";
type Sim = "none" | Dichromacy;
type Pair = "rg" | "mg";

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

/** Scalar test images (seeded). height: smooth bumps on a ramp, 0..1; fluor: nuclei-like
 *  blobs with noise, 0..1; signed: a difference image with values in [-3, 7]. */
export function scalarImage(id: ImageId): Float32Array {
  const v = new Float32Array(N * N), r = rng(id === "height" ? 3 : id === "fluor" ? 8 : 12);
  const bumps = Array.from({ length: id === "fluor" ? 14 : 6 }, () => [r.unif() * N, r.unif() * N, (id === "fluor" ? 6 : 18) + 10 * r.unif(), 0.4 + 0.6 * r.unif()]);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let s = 0;
    if (id === "height") s = 0.25 * x / N + 0.15 * y / N;
    for (const [cx, cy, w, a] of bumps) s += a * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w * w)) * (id === "height" ? 0.6 : 1);
    if (id === "signed") s = 0;
    v[y * N + x] = s;
  }
  if (id === "fluor") for (let p = 0; p < N * N; p++) v[p] = Math.max(0, Math.min(1, 0.05 + 0.8 * v[p] + 0.04 * r.norm()));
  if (id === "signed") for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
    v[y * N + x] = 7 * Math.exp(-((x - 55) ** 2 + (y - 60) ** 2) / 600) - 3 * Math.exp(-((x - 110) ** 2 + (y - 105) ** 2) / 500) + 0.6 * Math.sin(x / 9) * Math.cos(y / 13);
  if (id === "height") { let mx = 0; for (const s of v) mx = Math.max(mx, s); for (let p = 0; p < N * N; p++) v[p] /= mx; }
  return v;
}

/** Second channel for the merge: thin filaments (seeded curves), 0..1. */
function filaments(): Float32Array {
  const v = new Float32Array(N * N), r = rng(31);
  for (let k = 0; k < 9; k++) {
    let x = r.unif() * N, y = r.unif() * N, a = r.unif() * 2 * Math.PI;
    for (let s = 0; s < 260; s++) {
      a += 0.08 * r.norm(); x += Math.cos(a); y += Math.sin(a);
      for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
        const xi = Math.round(x) + dx, yi = Math.round(y) + dy;
        if (xi >= 0 && yi >= 0 && xi < N && yi < N) v[yi * N + xi] = Math.max(v[yi * N + xi], 0.8 * Math.exp(-(dx * dx + dy * dy) / 2));
      }
    }
  }
  return v;
}

autoMount(NAME, host => {
  injectStyle();
  const st = {
    mode: (host.dataset.mode === "merge" ? "merge" : "map") as "map" | "merge",
    image: (["height", "fluor", "signed"].includes(host.dataset.image ?? "") ? host.dataset.image : "height") as ImageId,
    map: host.dataset.map && MAPS[host.dataset.map] ? host.dataset.map : "jet",
    lo: 0, hi: 1, sim: "none" as Sim, pair: "rg" as Pair,
  };
  const fig = htmlEl("figure", "cmap-img", "", host);
  const cap = htmlEl("figcaption", "", "", fig);
  const cv = htmlEl("canvas", "", "", fig); cv.width = N; cv.height = N;
  const mid = htmlEl("div", "cmap-mid", "", host);
  const barCap = htmlEl("div", "cmap-lab", "colour bar", mid);
  const bar = htmlEl("canvas", "cmap-bar", "", mid); bar.width = 256; bar.height = 1;
  const lf = plotFigure(mid, "");
  const lp = new Plot(lf, { width: 420, height: 250, x: [0, 1], y: [0, 100], xlabel: "position in the map", ylabel: "lightness L*", yticks: [0, 25, 50, 75, 100], xticks: [0, 0.2, 0.4, 0.6, 0.8, 1] });
  const lline = lp.line([], [], { color: "#222", width: 3 });
  const read = htmlEl("div", "qb-readout", "", mid);

  const panel = htmlEl("div", "qb-panel", "", host);
  const modeRow = htmlEl("div", "cmap-row", "", panel);
  const modeBtns = (["map", "merge"] as const).map(m => { const b = button(modeRow, m === "map" ? "one channel + map" : "merge two channels", () => { st.mode = m; update(); }); b.dataset.id = m; return b; });
  const mapBox = htmlEl("div", "", "", panel); mapBox.style.display = "contents";
  htmlEl("div", "cmap-lab", "Image", mapBox);
  const imRow = htmlEl("div", "cmap-row", "", mapBox);
  const IMS: Record<ImageId, string> = { height: "height field", fluor: "fluorescence", signed: "signed difference" };
  const imBtns = (Object.keys(IMS) as ImageId[]).map(id => { const b = button(imRow, IMS[id], () => { setImage(id); update(); }); b.dataset.id = id; return b; });
  htmlEl("div", "cmap-lab", "Colour map", mapBox);
  const mRow = htmlEl("div", "cmap-row", "", mapBox);
  const mBtns = Object.keys(MAPS).map(id => { const b = button(mRow, MAPS[id].label, () => { st.map = id; update(); }); b.dataset.id = id; return b; });
  const loS = slider(mapBox, "display min", -8, 8, 0.1, st.lo, v => { st.lo = v; update(); });
  const hiS = slider(mapBox, "display max", -8, 8, 0.1, st.hi, v => { st.hi = v; update(); });
  const mergeBox = htmlEl("div", "", "", panel); mergeBox.style.display = "contents";
  htmlEl("div", "cmap-lab", "Channel colours (nuclei + filaments)", mergeBox);
  const pRow = htmlEl("div", "cmap-row", "", mergeBox);
  const pBtns = (["rg", "mg"] as Pair[]).map(p => { const b = button(pRow, p === "rg" ? "red + green" : "magenta + green", () => { st.pair = p; update(); }); b.dataset.id = p; return b; });
  htmlEl("div", "cmap-lab", "Simulate colour vision", panel);
  const sRow = htmlEl("div", "cmap-row", "", panel);
  const SIMS: Record<Sim, string> = { none: "normal", protan: "protanope", deutan: "deuteranope", tritan: "tritanope" };
  const sBtns = (Object.keys(SIMS) as Sim[]).map(s => { const b = button(sRow, SIMS[s], () => { st.sim = s; update(); }); b.dataset.id = s; return b; });
  htmlEl("div", "qb-note", "Dichromat simulation: a rank-2 projection in linear RGB (the missing cone's signal replaced by a combination of the other two). The L* curve is that of the map itself.", panel);
  const P: Record<string, Partial<typeof st>> = {
    jet: { mode: "map", image: "height", map: "jet", lo: 0, hi: 1, sim: "none" },
    viridis: { mode: "map", image: "height", map: "viridis", lo: 0, hi: 1, sim: "none" },
    merge: { mode: "merge", pair: "rg", sim: "deutan" },
    signed: { mode: "map", image: "signed", map: "viridis", lo: -3, hi: 7, sim: "none" },
  };
  const list: [string, string][] = [["jet", "jet invents edges"], ["viridis", "viridis is monotone"], ["merge", "red/green merge for a deuteranope"], ["signed", "signed data needs a diverging map"]];
  const pb = presets(fig, [...list.map(([id, label]) => ({ label, apply: () => { Object.assign(st, P[id]); loS.set(st.lo); hiS.set(st.hi); update(); } })),
    { label: "reset", apply: () => { Object.assign(st, P.jet); loS.set(st.lo); hiS.set(st.hi); update(); } }]);

  function setImage(id: ImageId): void {
    st.image = id;
    if (id === "signed") { st.lo = -3; st.hi = 7; } else { st.lo = 0; st.hi = 1; }
    loS.set(st.lo); hiS.set(st.hi);
  }
  const cache = new Map<string, Float32Array>();
  const img = (id: ImageId) => { let v = cache.get(id); if (!v) { v = scalarImage(id); cache.set(id, v); } return v; };
  let fil: Float32Array | null = null;

  function update(): void {
    modeBtns.forEach(b => b.classList.toggle("active", b.dataset.id === st.mode));
    imBtns.forEach(b => b.classList.toggle("active", b.dataset.id === st.image));
    mBtns.forEach(b => b.classList.toggle("active", b.dataset.id === st.map));
    pBtns.forEach(b => b.classList.toggle("active", b.dataset.id === st.pair));
    sBtns.forEach(b => b.classList.toggle("active", b.dataset.id === st.sim));
    mapBox.querySelectorAll<HTMLElement>(":scope > *").forEach(e => e.style.display = st.mode === "map" ? "" : "none");
    mergeBox.querySelectorAll<HTMLElement>(":scope > *").forEach(e => e.style.display = st.mode === "merge" ? "" : "none");
    const ctx = cv.getContext("2d")!, id = ctx.createImageData(N, N), d = id.data;
    const map = MAPS[st.map].map;
    if (st.mode === "map") {
      const v = img(st.image), span = st.hi - st.lo || 1e-9;
      const lut = new Uint8ClampedArray(3 * 256);
      for (let i = 0; i < 256; i++) { const c = map(i / 255); lut[3 * i] = 255 * c[0]; lut[3 * i + 1] = 255 * c[1]; lut[3 * i + 2] = 255 * c[2]; }
      for (let p = 0; p < N * N; p++) {
        const i = Math.max(0, Math.min(255, Math.round(255 * (v[p] - st.lo) / span)));
        d[4 * p] = lut[3 * i]; d[4 * p + 1] = lut[3 * i + 1]; d[4 * p + 2] = lut[3 * i + 2]; d[4 * p + 3] = 255;
      }
      cap.textContent = `${IMS[st.image]} (synthetic), ${MAPS[st.map].label}`;
    } else {
      const a = img("fluor"); fil ??= filaments();
      const ca = st.pair === "rg" ? [1, 0, 0] : [1, 0, 1], cb = [0, 1, 0];
      for (let p = 0; p < N * N; p++) {
        for (let k = 0; k < 3; k++) d[4 * p + k] = 255 * Math.min(1, a[p] * ca[k] + fil[p] * cb[k]);
        d[4 * p + 3] = 255;
      }
      cap.textContent = `merge (synthetic): nuclei in ${st.pair === "rg" ? "red" : "magenta"}, filaments in green`;
    }
    if (st.sim !== "none") simulateDichromatBytes(d, st.sim);
    ctx.putImageData(id, 0, 0);
    // colour bar and L* profile
    const bctx = bar.getContext("2d")!, bid = bctx.createImageData(256, 1);
    for (let i = 0; i < 256; i++) { const c = map(i / 255); bid.data[4 * i] = 255 * c[0]; bid.data[4 * i + 1] = 255 * c[1]; bid.data[4 * i + 2] = 255 * c[2]; bid.data[4 * i + 3] = 255; }
    if (st.sim !== "none") simulateDichromatBytes(bid.data, st.sim);
    bctx.putImageData(bid, 0, 0);
    barCap.textContent = st.mode === "map" ? `colour bar: ${st.lo.toFixed(1)} … ${st.hi.toFixed(1)}` : "colour bar of the map (not used in a merge)";
    const prof = lightnessProfile(map, 101), xs = prof.map((_, i) => i / 100);
    lline.set(xs, prof);
    let ups = 0, downs = 0;
    for (let i = 1; i < prof.length; i++) { if (prof[i] > prof[i - 1] + 0.05) ups++; else if (prof[i] < prof[i - 1] - 0.05) downs++; }
    const mono = downs === 0 || ups === 0;
    const iMax = prof.indexOf(Math.max(...prof));
    read.textContent = [`L* from ${prof[0].toFixed(1)} to ${prof[100].toFixed(1)}, max ${prof[iMax].toFixed(1)} at t = ${(iMax / 100).toFixed(2)}`,
      mono ? "L* is monotone" : "L* goes up and down: not monotone"].join("\n");
    for (const e of [barCap, bar, lf, read]) e.style.display = st.mode === "map" ? "" : "none";
  }
  const pre = host.dataset.preset;
  if (pre && P[pre]) { Object.assign(st, P[pre]); loS.set(st.lo); hiS.set(st.hi); pb[list.findIndex(l => l[0] === pre)]?.classList.add("active"); }
  else if (st.image === "signed") setImage("signed");
  update();
});
