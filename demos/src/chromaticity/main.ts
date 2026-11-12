// The CIE xy chromaticity diagram: the spectral locus (monochromatic light, with wavelength
// ticks) filled with approximate colours, gamut triangles of sRGB / Display P3 / Rec. 2020
// and the D65 white point.
//  - mode "mix": two lights A and B (drag them; brightness sliders) and their additive
//    mixture: it moves along the straight line between them, but not uniformly in the weight.
//  - mode "display": drag the three primaries and the white point of a display; its
//    RGB -> XYZ matrix and the inverse are shown live; a test colour T is in the gamut iff its
//    barycentric coordinates (= its linear RGB, up to scale) are all >= 0.
// Auto-mounts on <div data-demo="chromaticity">. Options: data-mode (mix|display),
// data-preset (mix|dim|srgbp3|laser|impossible).
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl } from "../lib/ui";
import { Plot } from "../lib/plot";
import { toCss } from "../lib/colour";
import * as SP from "../lib/spectra";
import { uniqueClip } from "../colourcloud/plotfix";

const NAME = "chromaticity";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; }
[data-demo="${NAME}"] .ch-plot { flex: 1 1 470px; min-width: 300px; max-width: 540px; }
[data-demo="${NAME}"] .ch-plot svg { touch-action: none; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 420px; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; white-space: pre; }
[data-demo="${NAME}"] .ch-row { display: flex; flex-wrap: wrap; gap: .3em .8em; align-items: center; }
[data-demo="${NAME}"] .ch-warn { color: var(--qb-c2); font-weight: 600; min-height: 1.3em; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.reveal [data-demo="${NAME}"] .ch-plot { flex-basis: 440px; max-width: 440px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
`;
type XY = [number, number];

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const LOCUS = SP.spectralLocus(380, 700, 1);
const LOCUS_POLY: XY[] = LOCUS.map(p => [p.x, p.y]);

/** Is (x, y) inside the region bounded by the spectral locus and the purple line? */
export function insideLocus(x: number, y: number, tol = 0.004): boolean {
  let inside = false, dmin = Infinity;
  for (let i = 0, j = LOCUS_POLY.length - 1; i < LOCUS_POLY.length; j = i++) {
    const [xi, yi] = LOCUS_POLY[i], [xj, yj] = LOCUS_POLY[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
    const ex = xj - xi, ey = yj - yi, L2 = ex * ex + ey * ey;
    const t = L2 > 0 ? Math.max(0, Math.min(1, ((x - xi) * ex + (y - yi) * ey) / L2)) : 0;
    dmin = Math.min(dmin, Math.hypot(x - xi - t * ex, y - yi - t * ey));
  }
  return inside || dmin <= tol;
}

/** Barycentric coordinates of p in the triangle (a, b, c). */
export function barycentric(p: XY, a: XY, b: XY, c: XY): [number, number, number] {
  const d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
  const l1 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / d;
  const l2 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / d;
  return [l1, l2, 1 - l1 - l2];
}

const locusAt = (l: number): XY => SP.xyzToXy(SP.cmf(l));

/** An image (data URL) of the diagram's fill: each (x, y) inside the locus in its colour at
 *  the highest brightness the sRGB screen can show (clipped outside the sRGB triangle). */
function fillImage(nx: number, ny: number, x1: number, y1: number): string {
  const c = document.createElement("canvas");
  c.width = nx; c.height = ny;
  const ctx = c.getContext("2d")!, id = ctx.createImageData(nx, ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const x = (i + 0.5) / nx * x1, y = (1 - (j + 0.5) / ny) * y1, p = 4 * (j * nx + i);
    if (!insideLocus(x, y, 0)) continue;
    const lin = SP.xyzToLinearSrgbCal(SP.xyYToXyz(x, y, 1)).map(v => Math.max(0, v));
    const m = Math.max(...lin, 1e-9);
    const e = lin.map(v => 255 * (v / m <= 0.0031308 ? 12.92 * v / m : 1.055 * (v / m) ** (1 / 2.4) - 0.055));
    id.data[p] = e[0]; id.data[p + 1] = e[1]; id.data[p + 2] = e[2]; id.data[p + 3] = 235;
  }
  ctx.putImageData(id, 0, 0);
  return c.toDataURL();
}

autoMount(NAME, host => {
  injectStyle();
  const st = {
    mode: (host.dataset.mode === "display" ? "display" : "mix") as "mix" | "display",
    A: locusAt(470), B: locusAt(580), YA: 0.3, YB: 0.6, w: 0.5,
    disp: { r: [...SP.SRGB_DISPLAY.r] as XY, g: [...SP.SRGB_DISPLAY.g] as XY, b: [...SP.SRGB_DISPLAY.b] as XY, w: [...SP.D65_XY] as XY },
    T: [0.25, 0.55] as XY,
    tri: { srgb: true, p3: false, rec2020: false },
  };

  const box = htmlEl("div", "ch-plot", "", host);
  const plot = new Plot(box, { width: 560, height: 600, x: [-0.06, 0.8], y: [0, 0.9], equal: true, xlabel: "x", ylabel: "y", xticks: [0, 0.2, 0.4, 0.6, 0.8], yticks: [0, 0.2, 0.4, 0.6, 0.8] });
  uniqueClip(plot.svg, NAME);
  // fill image (computed once)
  const X1 = 0.8, Y1 = 0.9;
  plot.mark(g => {
    const im = svgEl("image", { x: plot.x(0), y: plot.y(Y1), width: plot.x(X1) - plot.x(0), height: plot.y(0) - plot.y(Y1), preserveAspectRatio: "none" }, g);
    im.setAttribute("href", fillImage(200, 225, X1, Y1));
  }, []);
  // locus outline, purple line, ticks
  plot.mark(g => {
    const pts = LOCUS.map(p => `${plot.x(p.x)},${plot.y(p.y)}`).join(" ");
    svgEl("polygon", { points: pts, fill: "none", stroke: "#333", "stroke-width": 2 }, g);
    for (const l of [460, 470, 480, 490, 500, 510, 520, 540, 560, 580, 600, 620, 700]) {
      const [x, y] = locusAt(l);
      // tick direction: radially away from the white point
      let nx = x - 0.3127, ny = y - 0.329; const n = Math.hypot(nx, ny) || 1; nx /= n; ny /= n;
      const px = plot.x(x), py = plot.y(y), qx = plot.x(x + 0.018 * nx), qy = plot.y(y + 0.018 * ny);
      svgEl("line", { x1: px, y1: py, x2: qx, y2: qy, stroke: "#333", "stroke-width": 2 }, g);
      const t = svgEl("text", { x: plot.x(x + 0.045 * nx), y: plot.y(y + 0.045 * ny) + 6, "text-anchor": "middle", "font-size": 16 }, g);
      t.textContent = String(l);
    }
  }, [], {}, "line", plot.over);
  const triMarks = {
    srgb: plot.line([], [], { color: "#222", width: 2.5, label: "sRGB" }),
    p3: plot.line([], [], { color: "#d62728", width: 2.5, dash: "8 4", label: "Display P3" }),
    rec2020: plot.line([], [], { color: "#1f77b4", width: 2.5, dash: "3 4", label: "Rec. 2020" }),
  };
  const custom = plot.line([], [], { color: "#ff7f0e", width: 3.5 });
  const mixLine = plot.line([], [], { color: "#111", width: 2, dash: "4 4" });
  const layer = svgEl("g", {}, plot.over);

  const panel = htmlEl("div", "qb-panel", "", host);
  const modeRow = htmlEl("div", "ch-row", "", panel);
  const mixBtn = button(modeRow, "mix two lights", () => { st.mode = "mix"; update(); });
  const dispBtn = button(modeRow, "design a display", () => { st.mode = "display"; update(); });
  const triRow = htmlEl("div", "ch-row", "", panel);
  const triC = {
    srgb: checkbox(triRow, "sRGB", st.tri.srgb, v => { st.tri.srgb = v; update(); }),
    p3: checkbox(triRow, "P3", st.tri.p3, v => { st.tri.p3 = v; update(); }),
    rec2020: checkbox(triRow, "Rec. 2020", st.tri.rec2020, v => { st.tri.rec2020 = v; update(); }),
  };
  const mixBox = htmlEl("div", "", "", panel);
  mixBox.style.display = "contents";
  const yaS = slider(mixBox, "brightness A", 0.05, 1, 0.05, st.YA, v => { st.YA = v; update(); });
  const ybS = slider(mixBox, "brightness B", 0.05, 1, 0.05, st.YB, v => { st.YB = v; update(); });
  const wS = slider(mixBox, "share of B", 0, 1, 0.01, st.w, v => { st.w = v; update(); });
  const warn = htmlEl("div", "ch-warn", "", panel);
  const read = htmlEl("div", "qb-readout", "", panel);
  const note = htmlEl("div", "qb-note", "", panel);
  const P: Record<string, () => void> = {
    mix: () => { st.mode = "mix"; st.A = locusAt(470); st.B = locusAt(580); st.YA = 0.3; st.YB = 0.6; st.w = 0.5; },
    dim: () => { st.mode = "mix"; st.A = locusAt(470); st.B = locusAt(580); st.YA = 0.1; st.YB = 1; st.w = 0.5; },
    srgbp3: () => { st.mode = "display"; st.tri = { srgb: true, p3: true, rec2020: false }; st.disp = { r: [...SP.SRGB_DISPLAY.r], g: [...SP.SRGB_DISPLAY.g], b: [...SP.SRGB_DISPLAY.b], w: [...SP.D65_XY] } as typeof st.disp; st.T = [0.21, 0.71]; },
    laser: () => { st.mode = "display"; st.tri = { srgb: true, p3: false, rec2020: true }; st.disp = { r: locusAt(635), g: locusAt(532), b: locusAt(465), w: [...SP.D65_XY] as XY }; st.T = [0.21, 0.71]; },
    impossible: () => { st.mode = "display"; st.tri = { srgb: true, p3: false, rec2020: false }; st.disp = { r: [0.7, 0.3], g: [0.05, 0.9], b: [0.15, 0.06], w: [...SP.D65_XY] as XY }; st.T = [0.25, 0.55]; },
  };
  const list: [string, string][] = [["mix", "mixing two lights"], ["dim", "a dim blue and a bright yellow"], ["srgbp3", "sRGB vs P3"], ["laser", "a laser display"], ["impossible", "a primary outside the locus"]];
  const pb = presets(box, [...list.map(([id, label]) => ({ label, apply: () => { P[id](); sync(); update(); } })),
    { label: "reset", apply: () => { P[st.mode === "mix" ? "mix" : "srgbp3"](); st.tri = { srgb: true, p3: false, rec2020: false }; sync(); update(); } }]);
  function sync(): void { yaS.set(st.YA); ybS.set(st.YB); wS.set(st.w); triC.srgb.checked = st.tri.srgb; triC.p3.checked = st.tri.p3; triC.rec2020.checked = st.tri.rec2020; }

  const toData = (ev: PointerEvent): XY => {
    const [x, y] = plot.toData(ev);
    return [Math.max(0, Math.min(0.8, x)), Math.max(0, Math.min(0.9, y))];
  };
  function handle(p: XY, label: string, col: string, onMove: (q: XY) => void): void {
    const g = svgEl("g", {}, layer), cx = plot.x(p[0]), cy = plot.y(p[1]);
    svgEl("circle", { cx, cy, r: 11, fill: "#fff", stroke: col, "stroke-width": 4 }, g);
    const t = svgEl("text", { x: cx + 14, y: cy - 10, "font-size": 20, "font-weight": 700, class: "qb-halo" }, g);
    t.textContent = label;
    g.style.cursor = "grab";
    g.addEventListener("pointerdown", ev => {
      ev.preventDefault(); ev.stopPropagation();
      plot.svg.setPointerCapture(ev.pointerId);
      const move = (e: PointerEvent) => { onMove(toData(e)); update(); };
      const up = () => { plot.svg.removeEventListener("pointermove", move); plot.svg.removeEventListener("pointerup", up); };
      plot.svg.addEventListener("pointermove", move);
      plot.svg.addEventListener("pointerup", up);
    });
  }
  const dot = (p: XY, col: string, r = 7) => svgEl("circle", { cx: plot.x(p[0]), cy: plot.y(p[1]), r, fill: col, stroke: "#fff", "stroke-width": 2 }, layer);

  const tri = (d: { r: XY; g: XY; b: XY }): [number[], number[]] => [[d.r[0], d.g[0], d.b[0], d.r[0]], [d.r[1], d.g[1], d.b[1], d.r[1]]];
  const fm = (M: number[][]) => M.map(r => r.map(v => v.toFixed(3).padStart(7)).join(" ")).join("\n");

  function update(): void {
    mixBtn.classList.toggle("active", st.mode === "mix");
    dispBtn.classList.toggle("active", st.mode === "display");
    mixBox.querySelectorAll("label").forEach(l => (l as HTMLElement).style.display = st.mode === "mix" ? "" : "none");
    triMarks.srgb.set(...(st.tri.srgb ? tri(SP.SRGB_DISPLAY) : [[], []] as [number[], number[]]));
    triMarks.p3.set(...(st.tri.p3 ? tri(SP.P3_DISPLAY) : [[], []] as [number[], number[]]));
    triMarks.rec2020.set(...(st.tri.rec2020 ? tri(SP.REC2020_DISPLAY) : [[], []] as [number[], number[]]));
    plot.legend([...(st.tri.srgb ? [{ label: "sRGB", color: "#222" }] : []), ...(st.tri.p3 ? [{ label: "Display P3", color: "#d62728", dash: "8 4" }] : []),
      ...(st.tri.rec2020 ? [{ label: "Rec. 2020", color: "#1f77b4", dash: "3 4" }] : []), ...(st.mode === "display" ? [{ label: "your display", color: "#ff7f0e" }] : [])], "tr");
    layer.replaceChildren();
    dot(SP.D65_XY, "#000", 5);
    const lines: string[] = [];
    let warning = "";
    if (st.mode === "mix") {
      custom.set([], []);
      const XA = SP.xyYToXyz(st.A[0], st.A[1], st.YA), XB = SP.xyYToXyz(st.B[0], st.B[1], st.YB);
      const XM = [0, 1, 2].map(c => (1 - st.w) * XA[c] + st.w * XB[c]) as SP.V3;
      const m = SP.xyzToXy(XM), sA = XA[0] + XA[1] + XA[2], sB = XB[0] + XB[1] + XB[2];
      const t = st.w * sB / ((1 - st.w) * sA + st.w * sB);
      mixLine.set([st.A[0], st.B[0]], [st.A[1], st.B[1]]);
      dot(m, "#111", 9);
      const mt = svgEl("text", { x: plot.x(m[0]) + 12, y: plot.y(m[1]) + 22, "font-size": 20, "font-weight": 700, class: "qb-halo" }, layer);
      mt.textContent = "mix";
      handle(st.A, "A", "#1f77b4", q => { st.A = q; });
      handle(st.B, "B", "#ff7f0e", q => { st.B = q; });
      const f = (p: XY) => `(${p[0].toFixed(3)}, ${p[1].toFixed(3)})`;
      lines.push(`A  xy ${f(st.A)}  Y ${st.YA.toFixed(2)}  X+Y+Z ${sA.toFixed(2)}`);
      lines.push(`B  xy ${f(st.B)}  Y ${st.YB.toFixed(2)}  X+Y+Z ${sB.toFixed(2)}`);
      lines.push(`mix = ${(1 - st.w).toFixed(2)}·A + ${st.w.toFixed(2)}·B (in XYZ)`);
      lines.push(`mix xy ${f(m)}  Y ${XM[1].toFixed(3)}`);
      lines.push(`position on the line A→B: ${t.toFixed(3)}`);
      lines.push(`  = w(X+Y+Z)_B / ((1−w)(X+Y+Z)_A + w(X+Y+Z)_B)`);
      const outs = [["A", st.A], ["B", st.B]].filter(([, p]) => !insideLocus((p as XY)[0], (p as XY)[1])).map(([n]) => n);
      if (outs.length) warning = `${outs.join(" and ")} outside the locus: not a real light`;
      note.textContent = "Drag A and B. The mixture always lies on the straight line between them; where on the line depends on how much light each contributes (X+Y+Z), not on the slider alone.";
    } else {
      mixLine.set([], []);
      const d = st.disp;
      custom.set(...tri(d));
      handle(d.r, "R", "#d62728", q => { d.r = q; });
      handle(d.g, "G", "#2ca02c", q => { d.g = q; });
      handle(d.b, "B", "#1f77b4", q => { d.b = q; });
      handle(d.w, "W", "#555", q => { d.w = q; });
      handle(st.T, "T", "#111", q => { st.T = q; });
      const { M, scales } = SP.rgbToXyzMatrix({ name: "", ...d });
      const Mi = SP.inv3(M);
      const bc = barycentric(st.T, d.r, d.g, d.b);
      lines.push("RGB → XYZ (columns: primaries, white = R=G=B=1)");
      lines.push(fm(M));
      lines.push(`scales of the primaries: ${scales.map(v => v.toFixed(3)).join(", ")}`);
      lines.push("XYZ → RGB (inverse)");
      lines.push(fm(Mi));
      lines.push(`T: barycentric (${bc.map(v => v.toFixed(2)).join(", ")})`);
      lines.push(`   ${bc.every(v => v >= -1e-9) ? "inside the gamut" : "outside: needs a negative primary"}`);
      lines.push(`T on this display: rgb ∝ (${SP.mat3v(Mi, SP.xyYToXyz(st.T[0], st.T[1], 0.3)).map(v => v.toFixed(2)).join(", ")})`);
      const outs = (["r", "g", "b"] as const).filter(k => !insideLocus(d[k][0], d[k][1])).map(k => k.toUpperCase());
      if (outs.length) warning = `primary ${outs.join(", ")} outside the locus: no light has that chromaticity`;
      else if (scales.some(v => v <= 0)) warning = "white point outside the triangle: needs a negative primary";
      note.textContent = "Drag R, G, B (primaries), W (white point) and T (a test colour). The display can show exactly the colours inside its triangle.";
    }
    warn.textContent = warning;
    read.textContent = lines.join("\n");
  }
  void toCss;

  const pre = host.dataset.preset;
  if (pre && P[pre]) { P[pre](); sync(); pb[list.findIndex(l => l[0] === pre)]?.classList.add("active"); }
  else if (st.mode === "display") P.srgbp3();
  update();
});
