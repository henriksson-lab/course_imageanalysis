// Colour deconvolution of brightfield stains (moved here from the colour day). A synthetic
// slide rendered by Beer–Lambert from known concentration maps (seeded): H&E (hematoxylin
// nuclei, eosin cytoplasm/stroma) or H-DAB (some nuclei DAB-positive), with detector noise
// and 8-bit rounding. The analysis: optical density od = −log10(I/I0) per channel, then
// c = S⁻¹ od with the stain vectors as the columns of S (third column: DAB, or the cross
// product of the first two as a "residual" channel). Panels: the RGB image; the pixels'
// OD normalised to sum one in a triangle (R, G, B corners) where the stain vectors are
// draggable; the three concentration images (estimated or true); the crosstalk matrix
// T = S⁻¹ S_true (what each estimated map contains of each true stain) and κ(S).
// Auto-mounts on <div data-demo="stainsep">. Options: data-mode (he|hdab), data-preset
// (ruifrok|true|wrongE|residual|parallel|io|macenko|dab).
import { htmlEl, slider, presets, checkbox, svgEl, draggable, svgPoint, autoMount } from "../lib/ui";
import { heatmapGrid } from "../lib/plot";
import { mapFromAnchors } from "../lib/colour";
import { inv3, matMul } from "../lib/linalg";
import { histogram } from "../lib/histogram";
import { otsu } from "../lib/threshold";
import { label2d, componentSizes } from "../lib/label";
import * as U from "../lib/unmix";
import { injectStyle, uniqueClip, canvasFigure, fmt, COMMON_CSS } from "../lsunmix/common";

const NAME = "stainsep";
const CSS = COMMON_CSS + `
[data-demo="${NAME}"] { gap: .6em; align-items: flex-start; }
[data-demo="${NAME}"] .ss-left { flex: 1 1 380px; min-width: 260px; max-width: 420px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .ss-top { display: flex; gap: .4em; align-items: flex-start; }
[data-demo="${NAME}"] .ss-top .um-fig { flex: 0 0 58%; }
[data-demo="${NAME}"] .ss-maps { display: flex; gap: .3em; }
[data-demo="${NAME}"] .ss-maps .um-fig { flex: 1 1 0; min-width: 0; }
[data-demo="${NAME}"] .ss-mid { flex: 1 1 300px; min-width: 220px; max-width: 360px; display: flex; flex-direction: column; gap: .2em; }
[data-demo="${NAME}"] .ss-mid svg.ss-tri { touch-action: none; }
[data-demo="${NAME}"] .ss-T { max-width: 250px; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 330px; gap: .3em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; flex-wrap: nowrap; }
.reveal [data-demo="${NAME}"] .ss-left { flex: 0 0 390px; }
.reveal [data-demo="${NAME}"] .ss-mid { flex: 0 0 330px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 0 0 340px; max-width: 340px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .08em .35em; }
.reveal [data-demo="${NAME}"] .um-readout { font-size: 14px; }
.qb-demo[data-demo="${NAME}"] .ss-pre { white-space: pre; font-size: 13px; }
.reveal [data-demo="${NAME}"] .qb-check, .reveal [data-demo="${NAME}"] .qb-slider { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .15em; }
`;

import { V3, N, P, I0_TRUE, TRUE_V, RUIF, WRONG_E, WRONG_DAB, PARALLEL_E, PARALLEL_DAB } from "./model";
const SCOL = ["#3b2f8f", "#d6337f", "#8a5a1e", "#777777"];
const NAMES = ["H", "E", "DAB"];
const mapH = mapFromAnchors(["#ffffff", "#a99fd9", "#3b2f8f", "#120a40"]);
const mapE = mapFromAnchors(["#ffffff", "#f3a6c8", "#d6337f", "#6b0a37"]);
const mapD = mapFromAnchors(["#ffffff", "#d9b38a", "#8a5a1e", "#3a2208"]);
const mapR = mapFromAnchors(["#2166ac", "#ffffff", "#b2182b"]);

interface St { V: V3[]; third: "dab" | "cross"; i0: number; density: number; noise: number; truth: boolean; otsu: boolean }

function mount(host: HTMLElement): void {
  injectStyle(NAME, CSS);
  const hdab = host.dataset.mode === "hdab";
  const sc = U.stainScene(N, 7, hdab ? "hdab" : "he");
  const Strue = U.stainMatrix(TRUE_V[0], TRUE_V[1], TRUE_V[2]);
  const st: St = { V: RUIF.map(v => [...v] as V3), third: hdab ? "dab" : "cross", i0: 242, density: 1, noise: 2, truth: false, otsu: hdab };
  // slots: 0 = H, 1 = E (H&E) or DAB (H-DAB), 2 = the third column
  const slotStain = hdab ? [0, 2, 1] : [0, 1, 2];   // which true stain each slot means

  // ---- layout ----
  const left = htmlEl("div", "ss-left", "", host);
  const top = htmlEl("div", "ss-top", "", left);
  const rgbF = canvasFigure(top, hdab ? "H-DAB slide (synthetic)" : "H&E slide (synthetic)", N, N);
  const tNote = htmlEl("div", "qb-readout um-readout ss-pre", "", top);
  const mapsRow = htmlEl("div", "ss-maps", "", left);
  const maps = [0, 1, 2].map(k => canvasFigure(mapsRow, "", N, N));
  const mid = htmlEl("div", "ss-mid", "", host);
  const TW = 330, TH = 300, V: [number, number][] = [[24, 275], [165, 32], [306, 275]];
  const tri = svgEl("svg", { viewBox: `0 0 ${TW} ${TH}`, width: "100%", class: "qb-plot-svg ss-tri", role: "img", "aria-label": "Optical densities normalised to sum one, with the stain vectors" }, mid);
  svgEl("polygon", { points: V.map(v => v.join(",")).join(" "), fill: "none", stroke: "#999", "stroke-width": 1.5 }, tri);
  ["OD red", "OD green", "OD blue"].forEach((s, i) => { const t = svgEl("text", { x: V[i][0] + (i === 0 ? -6 : i === 2 ? 6 : 0), y: V[i][1] + (i === 1 ? -10 : 22), "text-anchor": i === 0 ? "start" : i === 2 ? "end" : "middle", "font-size": 17 }, tri); t.textContent = s; });
  const triPts = svgEl("g", {}, tri), triLines = svgEl("g", {}, tri), triH = svgEl("g", {}, tri);
  const tBox = htmlEl("div", "ss-T", "", mid);
  const panel = htmlEl("div", "qb-panel", "", host);

  const toTri = (v: ArrayLike<number>): [number, number] => {
    const s = v[0] + v[1] + v[2] || 1, a = v[0] / s, b = v[1] / s, c = v[2] / s;
    return [a * V[0][0] + b * V[1][0] + c * V[2][0], a * V[0][1] + b * V[1][1] + c * V[2][1]];
  };
  const fromTri = (x: number, y: number): V3 => {
    const [x1, y1] = V[0], [x2, y2] = V[1], [x3, y3] = V[2], d = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
    let a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / d, b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / d;
    a = Math.max(0.005, a); b = Math.max(0.005, b);
    let c = 1 - a - b;
    if (c < 0.005) { const s = a + b; a = a / s * 0.995; b = b / s * 0.995; c = 0.005; }
    return U.unit3([a, b, c]);
  };

  // ---- presets and controls ----
  const presetDefs: { id: string; label: string; s: Partial<St>; macenko?: boolean }[] = [
    { id: "ruifrok", label: "Ruifrok defaults", s: { V: RUIF } },
    { id: "true", label: "the true vectors: perfect separation", s: { V: TRUE_V } },
    { id: "wrongE", label: hdab ? "wrong DAB vector: crosstalk" : "wrong eosin vector: crosstalk", s: { V: hdab ? [TRUE_V[0], TRUE_V[1], WRONG_DAB] : [TRUE_V[0], WRONG_E, TRUE_V[2]] } },
    ...(hdab ? [] : [{ id: "residual", label: "two stains, residual from the cross product", s: { V: TRUE_V, third: "cross" as const } }]),
    { id: "parallel", label: "nearly parallel stains: noise explodes", s: { V: hdab ? [TRUE_V[0], TRUE_V[1], PARALLEL_DAB] : [TRUE_V[0], PARALLEL_E, TRUE_V[2]] } },
    { id: "io", label: "wrong I₀ / saturated pixels", s: { V: TRUE_V, i0: 205, density: 2.6 } },
    { id: "macenko", label: "estimate from the image (Macenko)", s: { V: RUIF }, macenko: true },
    ...(hdab ? [{ id: "dab", label: "count positive nuclei: threshold the DAB map", s: { V: TRUE_V, otsu: true } }] : []),
  ];
  const pbtn = presets(panel, [...presetDefs.map(p => ({ label: p.label, apply: () => applyPreset(p) })), { label: "reset", apply: () => applyPreset({ s: { V: RUIF } }) }]);
  const thirdC = hdab ? null : checkbox(panel, "third column = cross product (residual)", st.third === "cross", v => { st.third = v ? "cross" : "dab"; manual(); });
  const i0S = slider(panel, "I₀ assumed", 180, 255, 1, st.i0, v => { st.i0 = v; manual(); });
  const dS = slider(panel, "stain density", 0.5, 3, 0.1, st.density, v => { st.density = v; manual(true); });
  const nS = slider(panel, "noise (grey levels)", 0, 8, 0.5, st.noise, v => { st.noise = v; manual(true); });
  const truthC = checkbox(panel, "show true concentrations", st.truth, v => { st.truth = v; draw(); });
  const read = htmlEl("div", "qb-readout um-readout", "", mid);
  const note = htmlEl("div", "qb-note", "Drag the stain vectors (circles) in the triangle. The triangle shows each pixel's optical densities divided by their sum.", panel);
  void note;
  function manual(rerender = false): void { pbtn.forEach(b => b.classList.remove("active")); if (rerender) render(); draw(); }

  // ---- image ----
  let img: { R: Float32Array; G: Float32Array; B: Float32Array } = { R: new Float32Array(P), G: new Float32Array(P), B: new Float32Array(P) };
  let scaled: U.StainScene = sc;
  function render(): void {
    scaled = { ...sc, conc: sc.conc.map(c => c.map(v => v * st.density)) as [Float32Array, Float32Array, Float32Array] };
    img = U.renderBrightfield(scaled, Strue, I0_TRUE, { noise: st.noise, seed: 5 });
  }
  const Sest = (): Float64Array => st.third === "cross" ? U.stainMatrix(st.V[0], st.V[1]) : U.stainMatrix(st.V[0], st.V[1], st.V[2]);
  // in H&E the slots are H, E, (DAB | residual); in H-DAB the slots are H, E, DAB: V[2] is DAB.

  const handles: SVGCircleElement[] = [];
  for (let k = 0; k < 3; k++) {
    const h = svgEl("circle", { r: 11, fill: SCOL[k], stroke: "#fff", "stroke-width": 3 }, triH);
    const t = svgEl("title", {}, h); t.textContent = NAMES[k];
    handles.push(h);
    draggable(h, ev => svgPoint(tri, ev), (x, y) => { st.V[k] = fromTri(x, y); manual(); });
  }
  const cross = svgEl("rect", { width: 14, height: 14, fill: "#fff", stroke: SCOL[3], "stroke-width": 3 }, triH);

  function draw(): void {
    // RGB
    const ctx = rgbF.canvas.getContext("2d")!, id = ctx.createImageData(N, N);
    for (let p = 0; p < P; p++) { id.data[4 * p] = img.R[p]; id.data[4 * p + 1] = img.G[p]; id.data[4 * p + 2] = img.B[p]; id.data[4 * p + 3] = 255; }
    ctx.putImageData(id, 0, 0);
    const S = Sest(), I0: V3 = [st.i0, st.i0 * I0_TRUE[1] / I0_TRUE[0], st.i0 * I0_TRUE[2] / I0_TRUE[0]];
    const res = U.colourDeconvolve(img.R, img.G, img.B, I0, S, 1);
    // triangle: pixels with total OD > 0.15
    triPts.replaceChildren();
    let d = "";
    if (res) for (let p = 0; p < P; p += 3) {
      const o = [res.od[0][p], res.od[1][p], res.od[2][p]];
      if (o[0] + o[1] + o[2] < 0.25 || Math.min(...o) < -0.05) continue;
      const t = toTri(o);
      d += `M${t[0].toFixed(1)} ${t[1].toFixed(1)}h0`;
    }
    svgEl("path", { d, stroke: "#666", "stroke-width": 2.6, "stroke-linecap": "round", "stroke-opacity": 0.3, fill: "none" }, triPts);
    triLines.replaceChildren();
    TRUE_V.forEach((v, k) => { if (!hdab && k === 2) return; const t = toTri(v); svgEl("circle", { cx: t[0], cy: t[1], r: 6, fill: "none", stroke: SCOL[k], "stroke-width": 2, "stroke-dasharray": "3 2" }, triLines); });
    const used = st.third === "cross" ? 2 : 3;
    handles.forEach((h, k) => {
      h.style.display = k < used ? "" : "none";
      const t = toTri(st.V[k]); h.setAttribute("cx", t[0].toFixed(1)); h.setAttribute("cy", t[1].toFixed(1));
    });
    cross.style.display = st.third === "cross" ? "" : "none";
    if (st.third === "cross") { const t = toTri([Math.abs(S[2]), Math.abs(S[5]), Math.abs(S[8])]); cross.setAttribute("x", (t[0] - 7).toFixed(1)); cross.setAttribute("y", (t[1] - 7).toFixed(1)); }
    // concentration maps
    const labels = st.third === "cross" ? ["H", "E", "residual"] : ["H", "E", "DAB"];
    const cmaps = st.third === "cross" ? [mapH, mapE, mapR] : [mapH, mapE, mapD];
    const showIdx = hdab ? [0, 2, 1] : [0, 1, 2];     // H-DAB: show H, DAB, E
    showIdx.forEach((k, slot) => {
      const f = maps[slot], ctx2 = f.canvas.getContext("2d")!, id2 = ctx2.createImageData(N, N);
      const resid = labels[k] === "residual";
      const v = st.truth ? (resid ? new Float32Array(P) : scaled.conc[k]) : res ? res.c[k] : new Float32Array(P);
      const map = cmaps[k];
      for (let p = 0; p < P; p++) {
        const t = resid ? 0.5 + v[p] / 0.3 : v[p] / (1.2 * st.density);
        const c = map(Math.max(0, Math.min(1, t)));
        id2.data[4 * p] = 255 * c[0]; id2.data[4 * p + 1] = 255 * c[1]; id2.data[4 * p + 2] = 255 * c[2]; id2.data[4 * p + 3] = 255;
      }
      ctx2.putImageData(id2, 0, 0);
      f.cap.textContent = (st.truth ? "true " : "") + labels[k];
    });
    // crosstalk matrix T = S⁻¹ S_true (rows: estimated maps, columns: true stains)
    const Si = inv3(S);
    tBox.replaceChildren();
    const lines: string[] = [];
    if (Si) {
      const T = matMul(Si, Strue, 3, 3, 3);
      const rows = hdab ? [0, 2, 1] : [0, 1, 2], cols = hdab ? [0, 2, 1] : [0, 1];   // H&E: no DAB on the slide
      const Tshow = new Float64Array(rows.length * cols.length);
      rows.forEach((r, i) => cols.forEach((c, j) => { Tshow[i * cols.length + j] = T[r * 3 + c]; }));
      const g = heatmapGrid(tBox, Tshow, rows.length, cols.length, { map: mapR, lo: -1, hi: 1, numbers: true, fmt: v => fmt(v, 2), rowLabels: rows.map(r => "est. " + labels[r]), colLabels: cols.map(c => ["H", "E", "DAB"][c]), font: 15, cell: 52, title: "map contains (true →)" });
      uniqueClip(g.svg, NAME);
      lines.push(`κ(S) = ${fmt(U.condition(S, 3, 3), 1)}`);
    } else lines.push("S is singular: no inverse");
    // noise in the background of the H map (lumen pixels: no stain)
    if (res) {
      let n = 0, m = 0, m2 = 0, neg = 0;
      for (let p = 0; p < P; p++) if (scaled.conc[0][p] + scaled.conc[1][p] + scaled.conc[2][p] < 0.02) { n++; m += res.c[0][p]; m2 += res.c[0][p] ** 2; }
      for (let p = 0; p < P; p++) if (res.c[0][p] < -0.05 || res.c[1][p] < -0.05) neg++;
      if (n) lines.push(`H map in unstained areas: mean ${fmt(m / n, 3)}, sd ${fmt(Math.sqrt(Math.max(0, m2 / n - (m / n) ** 2)), 3)}`);
      if (neg) lines.push(`pixels with a concentration < −0.05: ${fmt(100 * neg / P, 1)} %`);
      let sat = 0;
      for (let p = 0; p < P; p++) if (img.R[p] <= 1 || img.G[p] <= 1 || img.B[p] <= 1) sat++;
      if (sat) lines.push(`saturated (I ≤ 1) pixels: ${sat}`);
      if (hdab && st.otsu) {
        const dab = res.c[2], h = histogram(dab, { bins: 128 }), t = otsu(h.counts);
        const thr = h.lo + (t.t + 1) * h.binWidth;
        const mask = new Uint8Array(P);
        for (let p = 0; p < P; p++) mask[p] = dab[p] > thr ? 1 : 0;
        const L = label2d(mask, N, N, 8), sizes = componentSizes(L.data, L.count);
        let cnt = 0;
        for (let i = 1; i <= L.count; i++) if (sizes[i] >= 15) cnt++;
        lines.push(`Otsu on the DAB map: threshold ${fmt(thr, 2)}`);
        lines.push(`positive nuclei found: ${cnt} (true: ${sc.positive.length} of ${sc.nucleusCount})`);
      }
    }
    read.textContent = lines.join("\n");
    tNote.textContent = st.third === "cross" ? "S = [H  E  H×E]" : "S = [H  E  DAB]";
    tNote.textContent += "\n(unit columns)\n" + [0, 1, 2].map(r => [0, 1, 2].map(c => fmt(S[r * 3 + c], 2).padStart(5)).join(" ")).join("\n");
    void slotStain;
  }

  function applyPreset(p: { s: Partial<St>; macenko?: boolean }): void {
    Object.assign(st, { V: RUIF, third: hdab ? "dab" : "cross", i0: 242, density: 1, noise: 2, otsu: hdab }, p.s);
    st.V = st.V.map(v => [...v] as V3);
    thirdC && (thirdC.checked = st.third === "cross"); i0S.set(st.i0); dS.set(st.density); nS.set(st.noise);
    render();
    if (p.macenko) {
      const I0: V3 = [st.i0, st.i0 * I0_TRUE[1] / I0_TRUE[0], st.i0 * I0_TRUE[2] / I0_TRUE[0]];
      const r0 = U.colourDeconvolve(img.R, img.G, img.B, I0, U.stainMatrix(RUIF[0], RUIF[1]), 1);
      const m = r0 ? U.macenko(r0.od, 0.15, 1) : null;
      if (m) {
        if (hdab) {
          // H-DAB: the two dominant stains are H and DAB
          st.V[0] = m.v1; st.V[2] = m.v2;
        } else { st.V[0] = m.v1; st.V[1] = m.v2; }
      }
    }
    draw();
  }
  render();
  const pre = presetDefs.findIndex(d => d.id === host.dataset.preset);
  if (pre >= 0) { applyPreset(presetDefs[pre]); pbtn[pre].classList.add("active"); }
  else applyPreset({ s: {} });
}

autoMount(NAME, mount);
