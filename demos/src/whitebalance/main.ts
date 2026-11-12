// White balance as a linear map. A chart of 24 patches with known reflectance spectra is
// rendered under a lamp (blackbody 2000–10000 K, fluorescent tube, narrow-band RGB LED,
// white LED) and corrected: estimate the lamp's white (grey world, white patch, click a
// grey card) and scale by a diagonal matrix in a chosen basis (encoded sRGB, linear RGB,
// cones = von Kries, Bradford), or fit a full 3 x 3 matrix to the chart. Readout: ΔE*ab per
// patch against the chart under daylight (D65). Model in wb.ts.
// Auto-mounts on <div data-demo="whitebalance">. Options: data-preset (tungsten|greyworld|
// forest|highlight|encoded|led).
import { htmlEl, slider, presets, button, checkbox, autoMount } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { linearToSrgb, RGB } from "../lib/colour";
import * as SP from "../lib/spectra";
import { runWB, illuminant, chartImage, ChartKind, Illum, WbObserver, Estimate, Space, NP, COLS, CELL, GAP } from "./wb";

const NAME = "whitebalance";
const STYLE = `
[data-demo="${NAME}"] { gap: .7em; }
[data-demo="${NAME}"] .wb-left { flex: 1 1 560px; max-width: 680px; min-width: 0; display: flex; flex-wrap: wrap; gap: .4em .6em; }
[data-demo="${NAME}"] .wb-left figure { margin: 0; flex: 1 1 200px; min-width: 150px; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin-bottom: .1em; }
[data-demo="${NAME}"] .wb-spd figcaption { font-size: 14px; color: var(--qb-muted); }
[data-demo="${NAME}"] canvas { width: 100%; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] canvas.wb-click { cursor: pointer; }
[data-demo="${NAME}"] .wb-left figure.wb-spd { flex: 1 1 100%; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 420px; gap: .35em; }
[data-demo="${NAME}"] .wb-row { display: flex; flex-wrap: wrap; gap: .3em; }
[data-demo="${NAME}"] .wb-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .wb-left { max-width: 620px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .42em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

autoMount(NAME, host => {
  injectStyle();
  const st = { kind: "chart" as ChartKind, illum: "blackbody" as Illum, T: 2800, obs: "eye" as WbObserver, how: "card" as Estimate, space: "lms" as Space, card: 20, highlight: false };

  const left = htmlEl("div", "wb-left", "", host);
  const fig = (cap: string) => { const f = htmlEl("figure", "", "", left); htmlEl("figcaption", "", cap, f); return htmlEl("canvas", "", "", f); };
  const cCap = fig("as captured"), cCor = fig("corrected"), cRef = fig("under daylight (truth)");
  cCap.classList.add("wb-click");
  const spdFig = plotFigure(left, "lamp spectrum (orange) and daylight (grey, dashed), each scaled to peak 1", "wb-spd");
  const spd = new Plot(spdFig, { width: 660, height: 150, x: [375, 735], y: [0, 1.1], xlabel: "wavelength (nm)", ylabel: "lamp", yticks: [0, 1], xticks: [400, 450, 500, 550, 600, 650, 700], margin: { b: 50 } });
  const spdD65 = spd.line([], [], { color: "#999", width: 2, dash: "4 4" });
  const spdLine = spd.line([], [], { color: "#ff7f0e", width: 3 });

  const panel = htmlEl("div", "qb-panel", "", host);
  const row = (lab: string) => { htmlEl("div", "wb-lab", lab, panel); return htmlEl("div", "wb-row", "", panel); };
  const group = <T extends string>(parent: HTMLElement, items: Record<T, string>, get: () => T, set: (v: T) => void) => {
    const btns = (Object.keys(items) as T[]).map(k => { const b = button(parent, items[k], () => { set(k); update(); }); b.dataset.id = k; return b; });
    return () => btns.forEach(b => b.classList.toggle("active", b.dataset.id === get()));
  };
  const r1 = row("Scene and lamp");
  const gKind = group(r1, { chart: "colour chart", forest: "forest" } as Record<ChartKind, string>, () => st.kind, v => { st.kind = v; st.card = v === "forest" ? 23 : 20; });
  const r2 = htmlEl("div", "wb-row", "", panel);
  const gIll = group(r2, { d65: "daylight", blackbody: "blackbody", fluorescent: "fluorescent", rgbled: "RGB LED", whiteled: "white LED" } as Record<Illum, string>, () => st.illum, v => { st.illum = v; });
  const tS = slider(panel, "T (K)", 2000, 10000, 100, st.T, v => { st.T = v; update(); });
  const r3 = row("Estimate the white");
  const gHow = group(r3, { none: "none", greyworld: "grey world", whitepatch: "white patch", card: "grey card (click)", lstsq: "3×3 fit to chart" } as Record<Estimate, string>, () => st.how, v => { st.how = v; });
  const r4 = row("Diagonal scaling applied to");
  const gSpace = group(r4, { encoded: "encoded sRGB", rgb: "linear RGB", lms: "cones (von Kries)", bradford: "Bradford" } as Record<Space, string>, () => st.space, v => { st.space = v; });
  const r5 = htmlEl("div", "wb-row", "", panel);
  const gObs = group(r5, { eye: "observer: eye", camera: "observer: camera" } as Record<WbObserver, string>, () => st.obs, v => { st.obs = v; });
  const hiC = checkbox(panel, "specular highlight (clipped)", st.highlight, v => { st.highlight = v; update(); });
  const read = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Click a patch of the captured chart to use it as the grey card. Exposure is matched (same mean luminance as the truth), so ΔE*ab measures the colour error. ΔE ≈ 2 is just noticeable.", panel);
  const P: Record<string, Partial<typeof st>> = {
    tungsten: { kind: "chart", illum: "blackbody", T: 2800, how: "none", space: "lms", highlight: false, obs: "eye", card: 20 },
    greyworld: { kind: "chart", illum: "blackbody", T: 2800, how: "greyworld", space: "lms", highlight: false, obs: "eye", card: 20 },
    forest: { kind: "forest", illum: "blackbody", T: 4000, how: "greyworld", space: "lms", highlight: false, obs: "eye", card: 23 },
    highlight: { kind: "chart", illum: "blackbody", T: 2800, how: "whitepatch", space: "lms", highlight: true, obs: "eye", card: 20 },
    encoded: { kind: "chart", illum: "blackbody", T: 2800, how: "card", space: "encoded", highlight: false, obs: "eye", card: 20 },
    led: { kind: "chart", illum: "rgbled", T: 2800, how: "card", space: "lms", highlight: false, obs: "eye", card: 20 },
  };
  const list: [string, string][] = [["tungsten", "tungsten 2800 K"], ["greyworld", "grey world works"], ["forest", "grey world in a forest"], ["highlight", "white patch fooled by a highlight"], ["encoded", "diagonal on encoded values"], ["led", "narrow-band LED: no diagonal fixes it"]];
  const pb = presets(left, [...list.map(([id, label]) => ({ label, apply: () => { Object.assign(st, P[id]); tS.set(st.T); hiC.checked = st.highlight; update(); } })),
    { label: "reset", apply: () => { Object.assign(st, P.tungsten, { how: "card" }); tS.set(st.T); hiC.checked = st.highlight; update(); } }]);

  cCap.addEventListener("pointerdown", ev => {
    const r = cCap.getBoundingClientRect(), w = cCap.width, h = cCap.height;
    const x = (ev.clientX - r.left) / r.width * w, y = (ev.clientY - r.top) / r.height * h;
    const cx = Math.floor((x - GAP) / (CELL + GAP)), cy = Math.floor((y - GAP) / (CELL + GAP));
    if (cx < 0 || cy < 0 || cx >= COLS || cy * COLS + cx >= NP) return;
    st.card = cy * COLS + cx; st.how = "card"; update();
  });

  function draw(c: HTMLCanvasElement, cols: RGB[], mark = -1, hi = false, white: RGB = [1, 1, 1]): void {
    const img = chartImage(cols.map(v => v.map(x => Math.min(1, Math.max(0, x))) as RGB), hi, white);
    c.width = img.w; c.height = img.h;
    const ctx = c.getContext("2d")!, id = ctx.createImageData(img.w, img.h);
    for (let p = 0; p < img.w * img.h; p++) {
      const gap = img.patch[p] < 0;
      id.data[4 * p] = gap ? 40 : 255 * linearToSrgb(img.r[p]); id.data[4 * p + 1] = gap ? 40 : 255 * linearToSrgb(img.g[p]); id.data[4 * p + 2] = gap ? 40 : 255 * linearToSrgb(img.b[p]); id.data[4 * p + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    if (mark >= 0) {
      const x0 = GAP + (mark % COLS) * (CELL + GAP), y0 = GAP + Math.floor(mark / COLS) * (CELL + GAP);
      ctx.strokeStyle = "#ffcc00"; ctx.lineWidth = 2; ctx.strokeRect(x0 - 1, y0 - 1, CELL + 2, CELL + 2);
    }
  }

  function update(): void {
    gKind(); gIll(); gHow(); gSpace(); gObs();
    tS.input.disabled = st.illum !== "blackbody";
    const r = runWB(st);
    const hiWhite: RGB = [1, 1, 1];
    draw(cCap, r.captured, st.how === "card" ? st.card : -1, st.highlight, hiWhite);
    draw(cCor, r.corrected);
    draw(cRef, r.reference);
    const E = illuminant(st.illum, st.T), m = Math.max(...E), md = Math.max(...SP.D65_SPD), wl = Array.from(SP.WL);
    spdLine.set(wl, Array.from(E, v => v / m)); spdD65.set(wl, Array.from(SP.D65_SPD, v => v / md));
    const mean = r.dE.reduce((s, v) => s + v, 0) / r.dE.length, worst = r.dE.indexOf(Math.max(...r.dE));
    const lines = [
      `estimated white (linear): ${st.how === "none" || st.how === "lstsq" ? "–" : r.white.map(v => v.toFixed(3)).join(", ")}`,
      `mean ΔE*ab vs truth: ${mean.toFixed(2)}`,
      `worst patch: ${worst + 1}, ΔE*ab ${r.dE[worst].toFixed(2)}`,
    ];
    read.textContent = lines.join("\n");
  }
  const pre = host.dataset.preset;
  if (pre && P[pre]) { Object.assign(st, P[pre]); tS.set(st.T); hiC.checked = st.highlight; pb[list.findIndex(l => l[0] === pre)]?.classList.add("active"); }
  update();
});
