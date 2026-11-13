// Rectangle (Haar-like) features from an integral image.
//  data-mode="templates" (default): drag a feature template over the image (drag inside to
//    move it, the round handle to resize it). White rectangles count +, black −; the dots
//    are the table entries read, numbered, with their coefficients. Readout: the response,
//    the lookups against the pixels a direct sum would add, and optionally the response at
//    every position (diverging colours).
//  data-mode="stack": the box group of a pixel feature stack (box mean, box SD,
//    centre − surround, edge x, edge y at scales 1..16) as thumbnails; paint pixels of two
//    classes and see any two features as a scatter plot, with one threshold.
// Options: data-mode, data-image="face|cell|cells", data-preset.
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, svgPoint } from "../lib/ui";
import { Plot } from "../lib/plot";
import { drawScalarImage, diverging } from "../lib/colour";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import {
  integralImage, Table, TEMPLATES, templateById, templateCorners, evalCorners, templatePixels, responseMap,
  boxFeature, BOX_FEATURES, BoxFeature,
} from "../lib/integral";
import { testScene } from "../boostdetect/synth";
import { N, SCALES, ImageId, faceScene, TPreset, T_PRESETS, SPreset, S_PRESETS, dab as dabL, paintStrokes, separation, bestSeparation } from "./model";

const NAME = "haarexplore";

const STYLE = `
[data-demo="${NAME}"] figure { margin: 0; position: relative; min-width: 0; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin: 0 0 .15em; }
[data-demo="${NAME}"] canvas { width: 100%; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .hx-main { flex: 0 1 340px; min-width: 220px; }
[data-demo="${NAME}"] .hx-main svg.hx-over { position: absolute; left: 0; width: 100%; touch-action: none; }
[data-demo="${NAME}"] .hx-side { flex: 0 1 340px; min-width: 200px; display: flex; flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 280px; gap: .4em; }
[data-demo="${NAME}"] .hx-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .hx-credit { font-size: 12px; color: var(--qb-muted); }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; }
[data-demo="${NAME}"] .hx-thumbs { display: grid; grid-template-columns: auto repeat(5, minmax(0, 1fr)); gap: 2px; align-items: center; font-size: 13px; }
[data-demo="${NAME}"] .hx-thumbs canvas { cursor: pointer; border: 2px solid transparent; }
[data-demo="${NAME}"] .hx-thumbs canvas.selx { border-color: #d62728; }
[data-demo="${NAME}"] .hx-thumbs canvas.sely { border-color: #2ca02c; }
[data-demo="${NAME}"] .hx-thumbs span { text-align: center; color: var(--qb-muted); }
[data-demo="${NAME}"] .qb-btn.cla.active { background: #ff7f0e; border-color: #ff7f0e; }
[data-demo="${NAME}"] .qb-btn.clb.active { background: #1f77b4; border-color: #1f77b4; }
.reveal [data-demo="${NAME}"] { font-size: 17px; gap: .7em; }
.reveal [data-demo="${NAME}"] .hx-main { flex-basis: 360px; }
.reveal [data-demo="${NAME}"] .hx-side { flex-basis: 360px; }
.reveal [data-demo="${NAME}"][data-mode="stack"] .hx-main { flex-basis: 320px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex-basis: 300px; gap: .25em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; }
.reveal [data-demo="${NAME}"] .qb-note, .reveal [data-demo="${NAME}"] .hx-credit, .reveal [data-demo="${NAME}"] .hx-lab { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const d = host.dataset;
  const mode = d.mode === "stack" ? "stack" : "templates";
  const images: Partial<Record<ImageId, Float64Array>> = {};
  const getImage = (id: ImageId, then: (img: Float64Array) => void): void => {
    if (images[id]) { then(images[id]!); return; }
    if (id === "cell") { loadCell(N).then(g => { images.cell = Float64Array.from(g.data); then(images.cell); }); return; }
    images[id] = id === "face" ? faceScene() : testScene("cells", 5).img;
    then(images[id]!);
  };
  if (mode === "templates") mountTemplates(host, d, getImage); else mountStack(host, d, getImage);
}

// ======================================================================================

function mountTemplates(host: HTMLElement, d: DOMStringMap, getImage: (id: ImageId, then: (img: Float64Array) => void) => void): void {
  const p0 = T_PRESETS.find(p => p.id === d.preset) ?? { ...T_PRESETS[0], image: (d.image as ImageId) ?? "face" };
  const st = { ...p0 };
  const main = htmlEl("figure", "hx-main", "", host);
  const cap = htmlEl("figcaption", "", "drag the template; round handle: resize", main);
  const cv = htmlEl("canvas", "", "", main);
  const over = svgEl("svg", { viewBox: `0 0 ${N} ${N}`, class: "hx-over" }, main);
  const credit = htmlEl("div", "hx-credit", CELL_CREDIT, main);
  const side = htmlEl("div", "hx-side", "", host);
  const mapFig = htmlEl("figure", "", "", side);
  const mapCap = htmlEl("figcaption", "", "", mapFig);
  const mcv = htmlEl("canvas", "", "", mapFig);
  const mapOver = svgEl("svg", { viewBox: `0 0 ${N} ${N}`, class: "hx-over" }, mapFig);
  mapOver.style.cssText = "position:absolute;left:0;width:100%;pointer-events:none";
  const ro = htmlEl("div", "qb-readout", "", side);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "hx-lab", "presets", panel);
  const pbtns = presets(panel, T_PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "hx-lab", "image", panel);
  const imRow = htmlEl("div", "qb-presets", "", panel);
  const IMS: ImageId[] = ["face", "cell", "cells"];
  const imBtns = IMS.map(id => button(imRow, id === "face" ? "cartoon face" : id === "cell" ? "micrograph" : "synthetic cells", () => { st.image = id; update(); }));
  htmlEl("div", "hx-lab", "template", panel);
  const tRow = htmlEl("div", "qb-presets", "", panel);
  const SHORT: Record<string, string> = { "edge-x": "edge |", "edge-y": "edge —", "line-x": "line |", "line-y": "line —", checker: "diagonal", centre: "centre − surround" };
  const tBtns = TEMPLATES.map(t => button(tRow, SHORT[t.id] ?? t.label, () => { st.tpl = t.id; clampTpl(); update(); }));
  const mapC = checkbox(panel, "response everywhere", st.map, v => { st.map = v; update(); });
  button(panel, "reset", () => load(p0 as TPreset));
  htmlEl("div", "qb-note", "Response = Σ (sign × table entry) over the numbered corners. Shared corners of neighbouring rectangles are read once.", panel);

  let img: Float64Array | null = null, T: Table | null = null, mapKey = "";

  function load(p: TPreset): void {
    Object.assign(st, p);
    pbtns.forEach((b, i) => b.classList.toggle("active", T_PRESETS[i] === p));
    update();
  }
  function clampTpl(): void {
    const t = templateById(st.tpl);
    st.sx = Math.max(1, Math.min(st.sx, Math.floor(N / t.cols))); st.sy = Math.max(1, Math.min(st.sy, Math.floor(N / t.rows)));
    st.x = Math.max(0, Math.min(st.x, N - t.cols * st.sx)); st.y = Math.max(0, Math.min(st.y, N - t.rows * st.sy));
  }

  // dragging
  let drag: { kind: "move" | "size"; dx: number; dy: number } | null = null;
  over.style.touchAction = "none";
  over.addEventListener("pointerdown", ev => {
    const [px, py] = svgPoint(over, ev);
    const t = templateById(st.tpl), W = t.cols * st.sx, H = t.rows * st.sy;
    const hx = st.x + W, hy = st.y + H;
    if (Math.hypot(px - hx, py - hy) < 12) drag = { kind: "size", dx: 0, dy: 0 };
    else if (px >= st.x && px <= st.x + W && py >= st.y && py <= st.y + H) drag = { kind: "move", dx: px - st.x, dy: py - st.y };
    else { drag = { kind: "move", dx: W / 2, dy: H / 2 }; st.x = Math.round(px - W / 2); st.y = Math.round(py - H / 2); clampTpl(); drawOverlay(); }
    ev.preventDefault();
    over.setPointerCapture(ev.pointerId);
    pbtns.forEach(b => b.classList.remove("active"));
  });
  over.addEventListener("pointermove", ev => {
    if (!drag) return;
    const [px, py] = svgPoint(over, ev);
    const t = templateById(st.tpl);
    if (drag.kind === "move") { st.x = Math.round(px - drag.dx); st.y = Math.round(py - drag.dy); }
    else { st.sx = Math.max(1, Math.round((px - st.x) / t.cols)); st.sy = Math.max(1, Math.round((py - st.y) / t.rows)); }
    clampTpl();
    drawOverlay();
  });
  const up = () => { if (drag) { const k = drag.kind; drag = null; if (k === "size") update(); else drawOverlay(); } };
  over.addEventListener("pointerup", up);
  over.addEventListener("pointercancel", up);

  function update(): void {
    imBtns.forEach((b, i) => b.classList.toggle("active", IMS[i] === st.image));
    tBtns.forEach((b, i) => b.classList.toggle("active", TEMPLATES[i].id === st.tpl));
    mapC.checked = st.map;
    credit.style.display = st.image === "cell" ? "" : "none";
    getImage(st.image, im => {
      if (img !== im) { img = im; T = integralImage(im, N, N); mapKey = ""; }
      drawScalarImage(cv, im, N, N, { lo: 0, hi: 255 });
      requestAnimationFrame(() => { over.style.top = cv.offsetTop + "px"; mapOver.style.top = mcv.offsetTop + "px"; });
      over.style.top = cv.offsetTop + "px";
      drawMap();
      drawOverlay();
    });
  }

  function drawMap(): void {
    mapFig.style.display = st.map ? "" : "none";
    if (!st.map || !T) return;
    const key = `${st.image}:${st.tpl}:${st.sx}:${st.sy}`;
    if (key === mapKey) return;
    mapKey = key;
    const t = templateById(st.tpl);
    const area = st.sx * st.sy;
    const m = responseMap(T, t, st.sx, st.sy, area);
    let mx = 0;
    for (const v of m) if (Number.isFinite(v)) mx = Math.max(mx, Math.abs(v));
    drawScalarImage(mcv, m, N, N, { map: diverging, lo: -mx, hi: mx, nan: [235, 235, 235, 255] });
    mapCap.textContent = `response at every centre (red +, blue −; ±${mx.toFixed(0)})`;
    mapOver.style.top = mcv.offsetTop + "px";
  }

  function drawOverlay(): void {
    if (!T || !img) return;
    over.replaceChildren();
    const t = templateById(st.tpl);
    const W = t.cols * st.sx, H = t.rows * st.sy;
    for (const r of [...t.rects].sort((a, b) => b.w * b.h - a.w * a.h)) {
      const pos = r.weight > 0;
      const e = svgEl("rect", { x: st.x + r.x * st.sx, y: st.y + r.y * st.sy, width: r.w * st.sx, height: r.h * st.sy }, over);
      e.style.cssText = `fill:${pos ? "#fff" : "#000"};fill-opacity:.45;stroke:${pos ? "#fff" : "#000"};stroke-width:1.5`;
    }
    const cs = templateCorners(t, st.x, st.y, st.sx, st.sy);
    cs.forEach((c, k) => {
      const dot = svgEl("circle", { cx: c.x, cy: c.y, r: 4.5 }, over);
      dot.style.cssText = `fill:${c.c > 0 ? "#2ca02c" : "#d62728"};stroke:#fff;stroke-width:1.2`;
      const tx = svgEl("text", { x: c.x + 5, y: c.y - 4, "font-size": 11, class: "qb-halo" }, over);
      tx.textContent = String(k + 1);
    });
    const hdl = svgEl("circle", { cx: st.x + W, cy: st.y + H, r: 7 }, over);
    hdl.style.cssText = "fill:#ffcc00;stroke:#333;stroke-width:1.5;cursor:nwse-resize";
    // marker on the map
    mapOver.replaceChildren();
    if (st.map) {
      const mk = svgEl("rect", { x: st.x, y: st.y, width: W, height: H }, mapOver);
      mk.style.cssText = "fill:none;stroke:#222;stroke-width:1.5;stroke-dasharray:4 3";
    }
    const val = evalCorners(T, cs);
    const area = st.sx * st.sy;
    const coef = (c: number) => (Number.isInteger(c) ? String(c) : c.toFixed(3));
    ro.textContent = `${t.label}: ${W} × ${H} px at (${st.x}, ${st.y})\n` +
      `response Σ c·S = ${val.toFixed(0)}\n` +
      `  per pixel of one cell: ${(val / area).toFixed(1)}\n` +
      `lookups: ${cs.length}   (coefficients ${cs.map(c => coef(c.c)).join(", ")})\n` +
      `pixels a direct sum adds: ${templatePixels(t, st.sx, st.sy)}`;
    cap.textContent = "drag the template; yellow handle: resize";
  }

  load(p0 as TPreset);
}

// ======================================================================================

function mountStack(host: HTMLElement, d: DOMStringMap, getImage: (id: ImageId, then: (img: Float64Array) => void) => void): void {
  const p0 = S_PRESETS.find(p => p.id === d.preset) ?? S_PRESETS[0];
  const st = { image: p0.image, fx: p0.fx, fy: p0.fy, brush: 1 as 0 | 1 | 2, thr: NaN, probe: [140, 120] as [number, number] };
  const labels = new Uint8Array(N * N);
  const main = htmlEl("figure", "hx-main", "", host);
  htmlEl("figcaption", "", "paint pixels of class A and B", main);
  const cv = htmlEl("canvas", "", "", main);
  cv.style.cursor = "crosshair"; cv.style.touchAction = "none";
  const credit = htmlEl("div", "hx-credit", CELL_CREDIT, main);
  const thumbs = htmlEl("div", "hx-thumbs", "", main);
  const side = htmlEl("div", "hx-side", "", host);
  const scat = new Plot(side, { width: 400, height: 330, x: [0, 1], y: [0, 1], xlabel: "x feature", ylabel: "y feature" });
  const bars = new Plot(side, { width: 400, height: 190, x: [-0.5, 24.5], y: [-3, 3], ylabel: "z-score", xticks: [2, 7, 12, 17, 22], xfmt: v => BOX_FEATURES[Math.round((v - 2) / 5)]?.id ?? "" });
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "hx-lab", "presets", panel);
  const pbtns = presets(panel, S_PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "hx-lab", "image", panel);
  const imRow = htmlEl("div", "qb-presets", "", panel);
  const IMS: ImageId[] = ["cell", "cells", "face"];
  const imBtns = IMS.map(id => button(imRow, id === "face" ? "cartoon face" : id === "cell" ? "micrograph" : "synthetic cells", () => { st.image = id; labels.fill(0); update(); }));
  htmlEl("div", "hx-lab", "mouse on the image", panel);
  const bRow = htmlEl("div", "qb-presets", "", panel);
  const bBtns = [
    button(bRow, "inspect", () => { st.brush = 0; sync(); }),
    button(bRow, "class A", () => { st.brush = 1; sync(); }, "cla"),
    button(bRow, "class B", () => { st.brush = 2; sync(); }, "clb"),
    button(bRow, "clear", () => { labels.fill(0); drawAll(); }),
  ];
  htmlEl("div", "hx-lab", "click a thumbnail: x feature (red); with \"y\" ticked: y feature (green)", panel);
  const yC = checkbox(panel, "y", false, () => undefined);
  const thrS = slider(panel, "threshold on x", 0, 100, 1, 50, v => { st.thr = v; drawScatter(); }, v => `${v} %`);
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Rows: feature type; columns: scale 1, 2, 4, 8, 16 px. Every thumbnail costs a few table lookups per pixel, whatever the scale. Bottom plot: the feature vector of the inspected pixel (z-scores over the image).", panel);

  let img: Float64Array | null = null;
  let maps: Float64Array[] = [], mstats: { m: number; s: number; lo: number; hi: number }[] = [];
  const thumbCanvases: HTMLCanvasElement[] = [];

  function load(p: SPreset): void {
    st.image = p.image; st.fx = p.fx; st.fy = p.fy; st.thr = 50;
    labels.fill(0);
    paintStrokes(labels, p.a, 1); paintStrokes(labels, p.b, 2);
    thrS.set(50);
    pbtns.forEach((b, i) => b.classList.toggle("active", S_PRESETS[i] === p));
    update();
  }
  const dab = (x: number, y: number, lab: number) => dabL(labels, x, y, lab);
  function sync(): void {
    imBtns.forEach((b, i) => b.classList.toggle("active", IMS[i] === st.image));
    bBtns.forEach((b, i) => b.classList.toggle("active", i < 3 && i === st.brush));
    credit.style.display = st.image === "cell" ? "" : "none";
    thumbCanvases.forEach((c, i) => { c.classList.toggle("selx", i === st.fx); c.classList.toggle("sely", i === st.fy); });
  }

  // painting / inspecting
  let last: [number, number] | null = null;
  const toPix = (ev: PointerEvent): [number, number] => {
    const r = cv.getBoundingClientRect();
    return [(ev.clientX - r.left) / r.width * N, (ev.clientY - r.top) / r.height * N];
  };
  cv.addEventListener("pointerdown", ev => {
    ev.preventDefault();
    const p = toPix(ev);
    if (st.brush === 0) { st.probe = [Math.floor(p[0]), Math.floor(p[1])]; drawImage(); drawBars(); return; }
    cv.setPointerCapture(ev.pointerId);
    last = p; dab(...p, st.brush); pbtns.forEach(b => b.classList.remove("active")); drawImage();
  });
  cv.addEventListener("pointermove", ev => {
    if (!last) return;
    const q = toPix(ev);
    const n = Math.max(1, Math.ceil(Math.hypot(q[0] - last[0], q[1] - last[1])));
    for (let j = 0; j <= n; j++) dab(last[0] + (q[0] - last[0]) * j / n, last[1] + (q[1] - last[1]) * j / n, st.brush);
    last = q; drawImage();
  });
  const up = () => { if (last) { last = null; drawScatter(); } };
  cv.addEventListener("pointerup", up);
  cv.addEventListener("pointercancel", up);

  function update(): void {
    sync();
    getImage(st.image, im => {
      if (im !== img) {
        img = im;
        const T1 = integralImage(im, N, N), T2 = integralImage(im, N, N, "f64", v => v * v);
        maps = []; mstats = [];
        for (const f of BOX_FEATURES) for (const s of SCALES) {
          const m = boxFeature(im, N, N, f.id as BoxFeature, s, T1, T2);
          let a = 0, q = 0, lo = Infinity, hi = -Infinity;
          for (const v of m) { a += v; q += v * v; if (v < lo) lo = v; if (v > hi) hi = v; }
          a /= m.length;
          maps.push(m); mstats.push({ m: a, s: Math.sqrt(Math.max(1e-12, q / m.length - a * a)), lo, hi });
        }
        buildThumbs();
      }
      drawAll();
    });
  }

  function buildThumbs(): void {
    thumbs.replaceChildren(); thumbCanvases.length = 0;
    htmlEl("span", "", "", thumbs);
    for (const s of SCALES) htmlEl("span", "", `${s} px`, thumbs);
    BOX_FEATURES.forEach((f, fi) => {
      htmlEl("span", "", f.label, thumbs).style.textAlign = "right";
      SCALES.forEach((_, si) => {
        const i = fi * 5 + si, c = htmlEl("canvas", "", "", thumbs);
        const m = maps[i], ms = mstats[i];
        const signed = f.id === "dob" || f.id.startsWith("haar");
        const lim = 3 * ms.s + Math.abs(ms.m);
        drawScalarImage(c, downsample(m), 64, 64, signed ? { map: diverging, lo: -lim, hi: lim } : { lo: ms.lo, hi: ms.hi });
        c.title = `${f.label}, scale ${SCALES[si]}`;
        c.addEventListener("click", () => { if (yC.checked) st.fy = i; else st.fx = i; sync(); drawScatter(); });
        thumbCanvases.push(c);
      });
    });
    sync();
  }

  function downsample(m: Float64Array): Float64Array {
    const o = new Float64Array(64 * 64);
    for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) {
      let s = 0;
      for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) s += m[(4 * y + j) * N + 4 * x + i];
      o[y * 64 + x] = s / 16;
    }
    return o;
  }

  function drawImage(): void {
    if (!img) return;
    const ctx = cv.getContext("2d")!;
    cv.width = N; cv.height = N;
    const id = ctx.createImageData(N, N);
    for (let i = 0; i < N * N; i++) {
      let r = img[i], g = img[i], b = img[i];
      if (labels[i] === 1) { r = 255; g = 127; b = 14; }
      if (labels[i] === 2) { r = 31; g = 119; b = 180; }
      id.data[4 * i] = r; id.data[4 * i + 1] = g; id.data[4 * i + 2] = b; id.data[4 * i + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    ctx.strokeStyle = "#ffcc00"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(st.probe[0] + 0.5, st.probe[1] + 0.5, 5, 0, 2 * Math.PI); ctx.stroke();
  }

  const fname = (i: number) => `${BOX_FEATURES[Math.floor(i / 5)].label}, ${SCALES[i % 5]} px`;

  function drawScatter(): void {
    if (!maps.length) return;
    const X = maps[st.fx], Y = maps[st.fy];
    const xs: number[] = [], ys: number[] = [], cols: string[] = [];
    const idx: number[] = [];
    for (let i = 0; i < N * N; i++) if (labels[i]) idx.push(i);
    // at most 600 dots (every k-th painted pixel)
    const step = Math.max(1, Math.ceil(idx.length / 600));
    for (let k = 0; k < idx.length; k += step) { const i = idx[k]; xs.push(X[i]); ys.push(Y[i]); cols.push(labels[i] === 1 ? "#ff7f0e" : "#1f77b4"); }
    const sx = mstats[st.fx], sy = mstats[st.fy];
    scat.clear();
    scat.setLabels({ xlabel: "x: " + fname(st.fx), ylabel: "y: " + fname(st.fy) });
    scat.setDomain([sx.lo, sx.hi === sx.lo ? sx.lo + 1 : sx.hi], [sy.lo, sy.hi === sy.lo ? sy.lo + 1 : sy.hi]);
    scat.scatter(xs, ys, { colors: cols, r: 3.5, fillOpacity: 0.7, stroke: "none" });
    const thr = sx.lo + (sx.hi - sx.lo) * (Number.isFinite(st.thr) ? st.thr : 50) / 100;
    scat.vline(thr, { color: "#333", width: 2 });
    const { aL, aR, bL, bR, share } = separation(X, labels, thr);
    ro.textContent = `x: ${fname(st.fx)}\ny: ${fname(st.fy)}\n` +
      `threshold x = ${thr.toFixed(1)}\n` +
      `A (orange): ${aL} left, ${aR} right\nB (blue):   ${bL} left, ${bR} right\n` +
      (aL + aR + bL + bR ? `this threshold separates ${(100 * share).toFixed(0)} %\nbest threshold on x:   ${(100 * bestSeparation(X, labels).share).toFixed(0)} %` : "paint some pixels");
    drawImage();
  }

  function drawBars(): void {
    if (!maps.length) return;
    const p = st.probe[1] * N + st.probe[0];
    const z = maps.map((m, i) => (m[p] - mstats[i].m) / mstats[i].s);
    bars.clear();
    bars.bars(z.map((_, i) => i), z.map(v => Math.max(-3, Math.min(3, v))), { colors: z.map((_, i) => ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][Math.floor(i / 5)]), barWidth: 0.8 });
    bars.hline(0, { color: "#888", width: 1, dash: "" });
  }

  function drawAll(): void { drawImage(); drawScatter(); drawBars(); }

  load(p0);
}

autoMount(NAME, mount);
