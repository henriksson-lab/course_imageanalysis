// Colour conversions as 3 x 3 matrices on an image: the three output channels shown as grey
// images, and the result recombined. Modes (data-mode):
//  - "ycbcr":  R'G'B' -> Y'CbCr (JPEG, full range: a matrix plus an offset). Subsample the
//              chroma (Cb, Cr) or the luma (Y') by 1, 2, 4 or 8 and rebuild the RGB image;
//              the mean colour difference ΔE*ab to the original measures the damage.
//  - "grey":   greyscale as a weighted sum of R, G, B, on linear or on encoded values
//              (luminance, luma, channel mean, one channel); the contrast of the two classes.
//  - "matrix": any 3 x 3 matrix on linear RGB (editable): sRGB -> XYZ, P3 -> sRGB, ...;
//              the share of pixels pushed outside [0, 1] (out of gamut).
// Auto-mounts on <div data-demo="colourmatrix">. Options: data-mode, data-image (objects|
// chart|isolum|he), data-preset.
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { srgbToLinear, linearToSrgb, linearRgbToXyz, xyzToLab, RGB, M_RGB_TO_XYZ } from "../lib/colour";
import { rgbToXyzMatrix, P3_DISPLAY, SRGB_DISPLAY, inv3, mat3mul } from "../lib/spectra";
import { Scene, SceneId, SCENE_LABELS, makeScene } from "../colourcloud/scenes";
import { ycbcrRoundTrip, meanDeltaE } from "./ops";

const NAME = "colourmatrix";
const STYLE = `
[data-demo="${NAME}"] { gap: .7em; }
[data-demo="${NAME}"] .cm-imgs { flex: 1 1 600px; display: grid; grid-template-columns: repeat(3, 1fr); gap: .4em .6em; max-width: 700px; min-width: 0; }
[data-demo="${NAME}"] .cm-imgs figure { margin: 0; min-width: 0; }
[data-demo="${NAME}"] .cm-imgs figure.cm-big { grid-row: span 1; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin-bottom: .1em; white-space: nowrap; overflow: hidden; }
[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 280px; max-width: 420px; gap: .4em; }
[data-demo="${NAME}"] .cm-mat { display: grid; grid-template-columns: repeat(3, 4.2em) 1.2em 4.2em; gap: 3px; align-items: center; font-size: 15px; }
[data-demo="${NAME}"] .cm-mat input { width: 100%; font: inherit; font-family: ui-monospace, monospace; font-size: 14px; padding: 1px 2px; box-sizing: border-box; }
[data-demo="${NAME}"] .cm-row { display: flex; flex-wrap: wrap; gap: .3em; }
[data-demo="${NAME}"] .cm-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .cm-pre { grid-column: 1 / -1; }
[data-demo="${NAME}"] .cm-credit { font-size: 13px; color: var(--qb-muted); grid-column: 1 / -1; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .cm-imgs { max-width: 600px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .45em; }
.reveal [data-demo="${NAME}"] .qb-note, .reveal [data-demo="${NAME}"] .cm-pre { grid-column: 1 / -1; }
[data-demo="${NAME}"] .cm-credit { display: none; }
`;

type Mode = "ycbcr" | "grey" | "matrix";

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const MATS: Record<string, { label: string; M: number[][] }> = {
  identity: { label: "identity", M: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] },
  xyz: { label: "sRGB → XYZ", M: M_RGB_TO_XYZ.map(r => r.map(v => Math.round(v * 1e4) / 1e4)) },
  p3: { label: "P3 → sRGB", M: mat3mul(inv3(rgbToXyzMatrix(SRGB_DISPLAY).M), rgbToXyzMatrix(P3_DISPLAY).M).map(r => r.map(v => Math.round(v * 1e4) / 1e4)) },
  swap: { label: "swap R and G", M: [[0, 1, 0], [1, 0, 0], [0, 0, 1]] },
  sat: { label: "more saturation", M: [[1.6, -0.5, -0.1], [-0.2, 1.4, -0.2], [-0.1, -0.5, 1.6]] },
};

autoMount(NAME, host => {
  injectStyle();
  const mode0 = (["ycbcr", "grey", "matrix"].includes(host.dataset.mode ?? "") ? host.dataset.mode : "ycbcr") as Mode;
  const st = {
    mode: mode0, image: ((host.dataset.image ?? "objects") in SCENE_LABELS ? host.dataset.image : "objects") as SceneId,
    k: 1, target: "chroma" as "chroma" | "luma",
    wts: [0.2126, 0.7152, 0.0722], domain: "linear" as "linear" | "encoded",
    M: MATS.xyz.M.map(r => [...r]), off: [0, 0, 0],
  };
  const cache = new Map<string, Scene>();
  const scene = () => { let s = cache.get(st.image); if (!s) { s = makeScene(st.image, 128); cache.set(st.image, s); } return s; };

  const imgs = htmlEl("div", "cm-imgs", "", host);
  const fig = (cap: string) => { const f = htmlEl("figure", "", "", imgs); const c = htmlEl("figcaption", "", cap, f); const cv = htmlEl("canvas", "", "", f); return { cap: c, cv }; };
  const fIn = fig("input"), fOut = fig("result");
  const fBlank = fig(""); fBlank.cv.style.visibility = "hidden";
  const fc = [fig("channel 1"), fig("channel 2"), fig("channel 3")];
  const credit = htmlEl("div", "cm-credit", "", imgs);

  const panel = htmlEl("div", "qb-panel", "", host);
  const modeRow = htmlEl("div", "cm-row", "", panel);
  const MODES: Record<Mode, string> = { ycbcr: "Y′CbCr", grey: "greyscale", matrix: "any matrix" };
  const modeBtns = (Object.keys(MODES) as Mode[]).map(m => { const b = button(modeRow, MODES[m], () => { st.mode = m; update(); }); b.dataset.id = m; return b; });
  htmlEl("div", "cm-lab", "Image", panel);
  const imRow = htmlEl("div", "cm-row", "", panel);
  const imBtns = (Object.keys(SCENE_LABELS) as SceneId[]).map(id => { const b = button(imRow, SCENE_LABELS[id], () => { st.image = id; update(); }); b.dataset.id = id; return b; });
  // y'cbcr controls
  const yBox = htmlEl("div", "", "", panel); yBox.style.display = "contents";
  const kS = slider(yBox, "subsample by", 0, 3, 1, 0, v => { st.k = 2 ** v; update(); }, v => String(2 ** v));
  const tRow = htmlEl("div", "cm-row", "", yBox);
  const tBtns = (["chroma", "luma"] as const).map(t => { const b = button(tRow, t === "chroma" ? "subsample Cb, Cr" : "subsample Y′", () => { st.target = t; update(); }); b.dataset.id = t; return b; });
  // grey controls
  const gBox = htmlEl("div", "", "", panel); gBox.style.display = "contents";
  const wS = ["R", "G", "B"].map((c, k) => slider(gBox, "weight " + c, 0, 1, 0.01, st.wts[k], v => { st.wts[k] = v; update(); }));
  const dRow = htmlEl("div", "cm-row", "", gBox);
  const dBtns = (["linear", "encoded"] as const).map(d => { const b = button(dRow, d === "linear" ? "on linear RGB" : "on encoded R′G′B′", () => { st.domain = d; update(); }); b.dataset.id = d; return b; });
  // matrix controls
  const mBox = htmlEl("div", "", "", panel); mBox.style.display = "contents";
  const matGrid = htmlEl("div", "cm-mat", "", mBox);
  const cells: HTMLInputElement[] = [];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const inp = htmlEl("input", "", "", matGrid); inp.type = "number"; inp.step = "0.05";
      inp.addEventListener("change", () => { const v = Number(inp.value); if (Number.isFinite(v)) { st.M[r][c] = v; update(); } inp.blur(); });
      cells.push(inp);
    }
    htmlEl("span", "", r === 1 ? "+" : "", matGrid);
    const o = htmlEl("input", "", "", matGrid); o.type = "number"; o.step = "0.05";
    o.addEventListener("change", () => { const v = Number(o.value); if (Number.isFinite(v)) { st.off[r] = v; update(); } o.blur(); });
    cells.push(o);
  }
  const matRow = htmlEl("div", "cm-row", "", mBox);
  for (const [id, m] of Object.entries(MATS)) button(matRow, m.label, () => { st.M = m.M.map(r => [...r]); st.off = [0, 0, 0]; update(); }).dataset.id = id;
  const read = htmlEl("div", "qb-readout", "", panel);
  const note = htmlEl("div", "qb-note", "", panel);

  const P: Record<string, () => void> = {
    ycc: () => { st.mode = "ycbcr"; st.image = "objects"; st.k = 1; st.target = "chroma"; },
    chroma4: () => { st.mode = "ycbcr"; st.image = "objects"; st.k = 4; st.target = "chroma"; },
    luma4: () => { st.mode = "ycbcr"; st.image = "objects"; st.k = 4; st.target = "luma"; },
    isolum: () => { st.mode = "grey"; st.image = "isolum"; st.wts = [0.2126, 0.7152, 0.0722]; st.domain = "linear"; },
    he: () => { st.mode = "grey"; st.image = "he"; st.wts = [0, 1, 0]; st.domain = "encoded"; },
    xyz: () => { st.mode = "matrix"; st.image = "objects"; st.M = MATS.xyz.M.map(r => [...r]); st.off = [0, 0, 0]; },
    p3: () => { st.mode = "matrix"; st.image = "objects"; st.M = MATS.p3.M.map(r => [...r]); st.off = [0, 0, 0]; },
  };
  const all: [string, string, Mode][] = [["ycc", "Y′CbCr channels", "ycbcr"], ["chroma4", "subsample chroma by 4: invisible", "ycbcr"], ["luma4", "subsample luma by 4: obvious", "ycbcr"],
    ["isolum", "the isoluminant test", "grey"], ["he", "stained tissue: best single channel", "grey"], ["xyz", "sRGB → XYZ", "matrix"], ["p3", "P3 → sRGB: out of gamut", "matrix"]];
  const list = all.filter(l => host.dataset.mode ? l[2] === mode0 : true);
  const greyPresets = [["luminance", [0.2126, 0.7152, 0.0722], "linear"], ["luma", [0.299, 0.587, 0.114], "encoded"], ["mean", [1 / 3, 1 / 3, 1 / 3], "encoded"], ["G only", [0, 1, 0], "encoded"]] as const;
  const gpRow = htmlEl("div", "cm-row", "", gBox);
  for (const [lab, w, d] of greyPresets) button(gpRow, lab, () => { st.wts = [...w]; st.domain = d; update(); });
  const preBox = htmlEl("div", "cm-pre", "", imgs);
  const pb = presets(preBox, [...list.map(([id, label]) => ({ label, apply: () => { P[id](); update(); } })),
    { label: "reset", apply: () => { P[mode0 === "ycbcr" ? "ycc" : mode0 === "grey" ? "isolum" : "xyz"](); update(); } }]);

  const drawRGB = (c: HTMLCanvasElement, R: ArrayLike<number>, G: ArrayLike<number>, B: ArrayLike<number>, w: number, h: number, scale = 255) => {
    c.width = w; c.height = h;
    const ctx = c.getContext("2d")!, id = ctx.createImageData(w, h);
    for (let p = 0; p < w * h; p++) { id.data[4 * p] = 255 * R[p] / scale; id.data[4 * p + 1] = 255 * G[p] / scale; id.data[4 * p + 2] = 255 * B[p] / scale; id.data[4 * p + 3] = 255; }
    ctx.putImageData(id, 0, 0);
  };
  const drawGrey = (c: HTMLCanvasElement, V: ArrayLike<number>, w: number, h: number, lo: number, hi: number) => drawRGB(c, Array.from(V, v => 255 * (v - lo) / (hi - lo)), Array.from(V, v => 255 * (v - lo) / (hi - lo)), Array.from(V, v => 255 * (v - lo) / (hi - lo)), w, h);

  function classContrast(s: Scene, V: ArrayLike<number>): string {
    if (!s.truth || !s.classes) return "";
    const target = st.image === "he" ? 1 : st.image === "isolum" ? 1 : 1;
    let a = 0, na = 0, b = 0, nb = 0;
    for (let p = 0; p < V.length; p++) { if (s.truth[p] === target) { a += V[p]; na++; } else if (st.image !== "objects" || s.truth[p] === 0) { b += V[p]; nb++; } }
    return `mean grey: ${s.classes[target]} ${(a / na).toFixed(1)}, ${st.image === "objects" ? "table" : s.classes[0]} ${(b / nb).toFixed(1)}`;
  }

  function update(): void {
    modeBtns.forEach(b => b.classList.toggle("active", b.dataset.id === st.mode));
    imBtns.forEach(b => b.classList.toggle("active", b.dataset.id === st.image));
    yBox.querySelectorAll<HTMLElement>(":scope > *").forEach(e => e.style.display = st.mode === "ycbcr" ? "" : "none");
    gBox.querySelectorAll<HTMLElement>(":scope > *").forEach(e => e.style.display = st.mode === "grey" ? "" : "none");
    mBox.querySelectorAll<HTMLElement>(":scope > *").forEach(e => e.style.display = st.mode === "matrix" ? "" : "none");
    tBtns.forEach(b => b.classList.toggle("active", b.dataset.id === st.target));
    dBtns.forEach(b => b.classList.toggle("active", b.dataset.id === st.domain));
    kS.set(Math.log2(st.k)); st.wts.forEach((v, k) => wS[k].set(v));
    for (let r = 0; r < 3; r++) { for (let c = 0; c < 3; c++) cells[4 * r + c].value = String(st.M[r][c]); cells[4 * r + 3].value = String(st.off[r]); }
    const s = scene(), w = s.w, h = s.h, n = w * h;
    credit.textContent = s.credit;
    const eR = Float32Array.from(s.r, v => 255 * linearToSrgb(Math.min(1, v))), eG = Float32Array.from(s.g, v => 255 * linearToSrgb(Math.min(1, v))), eB = Float32Array.from(s.b, v => 255 * linearToSrgb(Math.min(1, v)));
    drawRGB(fIn.cv, eR, eG, eB, w, h);
    const lines: string[] = [];
    fBlank.cap.textContent = "";
    if (st.mode === "ycbcr") {
      const rt = ycbcrRoundTrip(eR, eG, eB, w, h, st.k, st.target);
      fc[0].cap.textContent = "Y′ (luma)" + (st.target === "luma" && st.k > 1 ? ` ÷${st.k}` : "");
      fc[1].cap.textContent = "Cb" + (st.target === "chroma" && st.k > 1 ? ` ÷${st.k}` : "");
      fc[2].cap.textContent = "Cr" + (st.target === "chroma" && st.k > 1 ? ` ÷${st.k}` : "");
      drawGrey(fc[0].cv, rt.Y, w, h, 0, 255); drawGrey(fc[1].cv, rt.Cb, w, h, 0, 255); drawGrey(fc[2].cv, rt.Cr, w, h, 0, 255);
      drawRGB(fOut.cv, rt.R, rt.G, rt.B, w, h);
      fOut.cap.textContent = "rebuilt R′G′B′";
      const bytes = st.k === 1 ? 3 : 1 + 2 / (st.k * st.k), bytesL = st.k === 1 ? 3 : 2 + 1 / (st.k * st.k);
      lines.push("Y′ = 0.299 R′ + 0.587 G′ + 0.114 B′");
      lines.push("Cb = 128 + 0.564 (B′ − Y′)");
      lines.push("Cr = 128 + 0.713 (R′ − Y′)");
      lines.push(`bytes per pixel: ${(st.target === "chroma" ? bytes : bytesL).toFixed(3)} (was 3)`);
      lines.push(`mean ΔE*ab rebuilt vs input: ${meanDeltaE([rt.R, rt.G, rt.B], [eR, eG, eB]).toFixed(2)}`);
      note.textContent = "Cb and Cr are shown with 128 as mid-grey. Subsampling by k keeps one value per k × k block.";
    } else if (st.mode === "grey") {
      const sum = st.wts[0] + st.wts[1] + st.wts[2] || 1, wn = st.wts.map(v => v / sum);
      const V = new Float32Array(n);
      for (let p = 0; p < n; p++) {
        if (st.domain === "linear") V[p] = 255 * linearToSrgb(Math.min(1, wn[0] * s.r[p] + wn[1] * s.g[p] + wn[2] * s.b[p]));
        else V[p] = wn[0] * eR[p] + wn[1] * eG[p] + wn[2] * eB[p];
      }
      ["R′", "G′", "B′"].forEach((c, k) => fc[k].cap.textContent = `${c} (weight ${wn[k].toFixed(3)})`);
      drawGrey(fc[0].cv, eR, w, h, 0, 255); drawGrey(fc[1].cv, eG, w, h, 0, 255); drawGrey(fc[2].cv, eB, w, h, 0, 255);
      drawGrey(fOut.cv, V, w, h, 0, 255);
      fOut.cap.textContent = "greyscale";
      lines.push(`weights (normalised): ${wn.map(v => v.toFixed(3)).join(", ")}`);
      lines.push(`applied to ${st.domain === "linear" ? "linear RGB, then encoded" : "encoded R′G′B′"}`);
      const cc = classContrast(s, V); if (cc) lines.push(cc);
      note.textContent = "The weights are normalised to sum to 1. On linear RGB the result is a light quantity (luminance for 0.2126, 0.7152, 0.0722) and is encoded for display.";
    } else {
      const Ch = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
      let outside = 0;
      for (let p = 0; p < n; p++) {
        const l = [s.r[p], s.g[p], s.b[p]];
        let out = false;
        for (let r = 0; r < 3; r++) { const v = st.M[r][0] * l[0] + st.M[r][1] * l[1] + st.M[r][2] * l[2] + st.off[r]; Ch[r][p] = v; if (v < -1e-4 || v > 1 + 1e-4) out = true; }
        if (out) outside++;
      }
      fc.forEach((f, k) => f.cap.textContent = `output ${k + 1}`);
      for (let k = 0; k < 3; k++) drawGrey(fc[k].cv, Ch[k].map(v => 255 * linearToSrgb(Math.min(1, Math.max(0, v)))), w, h, 0, 255);
      drawRGB(fOut.cv, Ch[0].map(v => 255 * linearToSrgb(Math.min(1, Math.max(0, v)))), Ch[1].map(v => 255 * linearToSrgb(Math.min(1, Math.max(0, v)))), Ch[2].map(v => 255 * linearToSrgb(Math.min(1, Math.max(0, v)))), w, h);
      fOut.cap.textContent = "outputs shown as RGB";
      // mark out-of-gamut pixels in the blank slot
      const mask = new Float32Array(n);
      for (let p = 0; p < n; p++) mask[p] = Ch.some(c => c[p] < -1e-4 || c[p] > 1 + 1e-4) ? 255 : 0;
      fBlank.cv.style.visibility = "visible"; drawGrey(fBlank.cv, mask, w, h, 0, 255); fBlank.cap.textContent = "outside [0, 1] (white)";
      lines.push("applied to linear RGB in [0, 1]");
      lines.push(`pixels outside [0, 1]: ${outside} of ${n} (${(100 * outside / n).toFixed(1)} %)`);
      lines.push(`row sums: ${st.M.map(r => (r[0] + r[1] + r[2]).toFixed(4)).join(", ")}`);
      note.textContent = "Edit the cells (Enter). Rows of sRGB → XYZ: X, Y (luminance), Z; each row sum is that coordinate of white. P3 → sRGB treats the image as P3 values and shows them on an sRGB screen.";
    }
    if (st.mode !== "matrix") fBlank.cv.style.visibility = "hidden";
    read.textContent = lines.join("\n");
  }
  const pre = host.dataset.preset;
  if (pre && P[pre]) { P[pre](); pb[list.findIndex(l => l[0] === pre)]?.classList.add("active"); }
  else if (mode0 === "grey") P.isolum();
  else if (mode0 === "matrix") P.xyz();
  update();
});
