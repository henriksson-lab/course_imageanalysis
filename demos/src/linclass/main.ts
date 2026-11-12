// A linear classifier f(x) = w·x + b in 2D. Points of two classes (blue = −1, red = +1); the
// line f = 0 with its normal vector w drawn as an arrow from a point on the line. Drag the
// arrow's tip to turn w, drag its foot to shift the line (b = −w·foot); "× 2" / "÷ 2" scale
// (w, b) together (the line does not move, f doubles or halves, distances stay). The half-planes
// are shaded by the sign of f; each point can show its signed distance f/‖w‖. Buttons: the
// nearest-class-mean line (w = μ₊ − μ₋ through the midpoint), the 1-NN boundary (shaded
// regions), and one perceptron step (the first misclassified point pulls the line). Readout:
// f, ‖w‖, number of errors, smallest y·f/‖w‖ (the margin of the line).
//
// Auto-mounts on <div data-demo="linclass">. Options:
//   data-preset  one of the ids in geom.ts (byhand, many, mean, outlier, xor, ring, objects)
//   data-presets comma-separated preset ids to offer (default all)
//   data-axes    "name 1,name 2": axis names (default from the preset, else x₁, x₂)
//   data-labels  "1": show the signed distances from the start
import { htmlEl, svgEl, button, presets, checkbox, autoMount, svgPoint } from "../lib/ui";
import { pointPlayground } from "../lib/playground";
import { QUALITATIVE, hexToRgb } from "../lib/colour";
import type { LPoint } from "../lib/datasets2d";
import { worldOf } from "../svm2d/field";
import { LIN_PRESETS, LinPreset, Line, f, norm, signedDistance, errors, minMargin, nearestMeanLine, nnClass, perceptronStep, ySign } from "./geom";

const NAME = "linclass";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; align-items: flex-start; }
[data-demo="${NAME}"] .lc-left { flex: 0 1 470px; min-width: 260px; }
[data-demo="${NAME}"] .lc-left .qb-plot { max-width: none; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 420px; gap: .4em; }
[data-demo="${NAME}"] .lc-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; overflow-x: auto; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .lc-left { flex-basis: 460px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const fmt = (v: number, d = 2) => (Math.abs(v) < 5e-13 ? 0 : v).toFixed(d);
const signed = (v: number, d = 2) => (v >= 0 ? "+" : "−") + fmt(Math.abs(v), d);

function mount(host: HTMLElement): void {
  injectStyle();
  const offered = (host.dataset.presets ?? "").split(",").map(s => s.trim()).filter(Boolean);
  const list = LIN_PRESETS.filter(p => !offered.length || offered.includes(p.id));
  const p0 = list.find(p => p.id === host.dataset.preset) ?? list[0];
  const axesAttr = host.dataset.axes?.split(",").map(s => s.trim());
  const colors = [QUALITATIVE[0], QUALITATIVE[1]];
  let line: Line = { w: [...p0.line.w], b: p0.line.b };
  let foot: [number, number] = [0, 0];
  let preset: LinPreset = p0, showD = host.dataset.labels === "1", showNN = false, showMean = false, lastPerc = -1;

  const left = htmlEl("div", "lc-left", "", host);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "lc-lab", "presets", panel);
  const presetBtns = presets(panel, list.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "lc-lab", "add points of class", panel);
  const classBox = htmlEl("div", "", "", panel);
  const pg = pointPlayground(left, { k: 2, colors, names: ["class −1", "class +1"], controlsParent: classBox, resolution: 96, r: 9, onChange: () => { lastPerc = -1; update(); } });
  const W = worldOf(pg, p0.range[0], p0.range[1]);

  htmlEl("div", "lc-lab", "the line", panel);
  const row1 = htmlEl("div", "qb-presets", "", panel);
  button(row1, "nearest class mean", () => { const m = nearestMeanLine(W.points()); line = { w: m.w, b: m.b }; showMean = true; placeFoot(); update(); });
  button(row1, "perceptron step", () => {
    const r = perceptronStep(line, W.points(), lastPerc + 1);
    if (r.index >= 0) { line = r.line; lastPerc = r.index; placeFoot(); }
    else lastPerc = -1;
    update(r.index < 0 ? "no misclassified point: the perceptron stops" : "");
  });
  const row2 = htmlEl("div", "qb-presets", "", panel);
  button(row2, "(w, b) × 2", () => { line = { w: [2 * line.w[0], 2 * line.w[1]], b: 2 * line.b }; update(); });
  button(row2, "(w, b) ÷ 2", () => { line = { w: [line.w[0] / 2, line.w[1] / 2], b: line.b / 2 }; update(); });
  button(row2, "reset line", () => { line = { w: [...preset.line.w], b: preset.line.b }; showMean = false; placeFoot(); update(); });
  const dCb = checkbox(panel, "show signed distances f/‖w‖", showD, v => { showD = v; update(); });
  const nnCb = checkbox(panel, "shade the 1-NN regions instead", showNN, v => { showNN = v; update(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  const note = htmlEl("div", "qb-note", "", panel);
  htmlEl("div", "qb-note", "Drag the arrow's tip to turn w, its foot to move the line; click to add points.", panel);

  // ---- drawing groups
  const gLine = svgEl("g", {}, pg.overlay);
  const gTop = svgEl("g", {}, pg.top);
  const gHandles = svgEl("g", {}, pg.top);

  /** Foot of the arrow: the point of the line nearest to the centre of the view. */
  function placeFoot(): void {
    const c = (W.lo + W.hi) / 2, n2 = line.w[0] ** 2 + line.w[1] ** 2 || 1, v = f(line, c, c) / n2;
    foot = [c - v * line.w[0], c - v * line.w[1]];
  }

  function arrowScale(): number { return 0.22 * (W.hi - W.lo) / (norm(line) || 1); }

  function handle(cx: number, cy: number, col: string, onMove: (x: number, y: number) => void): void {
    const h = svgEl("circle", { cx, cy, r: 13, class: "handle" }, gHandles);
    h.style.setProperty("stroke", col);
    h.style.cursor = "grab";
    h.style.touchAction = "none";
    h.addEventListener("pointerdown", ev => {
      ev.preventDefault(); ev.stopPropagation();
      h.setPointerCapture(ev.pointerId);
      const move = (e: PointerEvent) => {
        e.stopPropagation();
        const [sx, sy] = svgPoint(pg.svg, e);
        onMove(W.fromU(pg.sx.invert(sx)), W.fromU(pg.sy.invert(sy)));
      };
      const up = () => { h.removeEventListener("pointermove", move); h.removeEventListener("pointerup", up); h.removeEventListener("pointercancel", up); update(); };
      h.addEventListener("pointermove", move);
      h.addEventListener("pointerup", up);
      h.addEventListener("pointercancel", up);
    });
  }

  let raf = 0;
  function update(msg = ""): void {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => draw(msg));
  }

  function draw(msg: string): void {
    const pts = W.points();
    // background
    const c0 = hexToRgb(colors[0]).map(v => 255 * v), c1 = hexToRgb(colors[1]).map(v => 255 * v);
    pg.setBackgroundColour((u, v) => {
      const x = W.fromU(u), y = W.fromU(v);
      const pos = showNN && pts.length ? nnClass(pts, x, y) === 1 : f(line, x, y) > 0;
      const c = pos ? c1 : c0;
      return [c[0], c[1], c[2], 0.2];
    });
    gLine.replaceChildren(); gTop.replaceChildren(); gHandles.replaceChildren();
    // the line: clip f = 0 to the square
    const L = W.lo, H = W.hi, segs: [number, number][] = [];
    const [a, b] = line.w, c = line.b;
    if (Math.abs(b) > 1e-12) for (const x of [L, H]) { const y = -(a * x + c) / b; if (y >= L - 1e-9 && y <= H + 1e-9) segs.push([x, y]); }
    if (Math.abs(a) > 1e-12) for (const y of [L, H]) { const x = -(b * y + c) / a; if (x >= L - 1e-9 && x <= H + 1e-9) segs.push([x, y]); }
    if (segs.length >= 2) {
      const l = svgEl("line", { x1: W.X(segs[0][0]), y1: W.Y(segs[0][1]), x2: W.X(segs[1][0]), y2: W.Y(segs[1][1]) }, gLine);
      l.style.setProperty("stroke", "var(--qb-fg, #222)"); l.style.setProperty("stroke-width", "3");
    }
    // nearest means
    if (showMean && pts.length) {
      const m = nearestMeanLine(pts);
      const s = svgEl("line", { x1: W.X(m.mMinus[0]), y1: W.Y(m.mMinus[1]), x2: W.X(m.mPlus[0]), y2: W.Y(m.mPlus[1]) }, gTop);
      s.style.setProperty("stroke", "var(--qb-fg, #222)"); s.style.setProperty("stroke-dasharray", "5 4"); s.style.setProperty("stroke-width", "2");
      [[m.mMinus, colors[0]], [m.mPlus, colors[1]]].forEach(([mm, col]) => {
        const [mx, my] = mm as [number, number];
        const t = svgEl("path", { d: `M${W.X(mx) - 11},${W.Y(my)}h22M${W.X(mx)},${W.Y(my) - 11}v22` }, gTop);
        t.style.setProperty("stroke", col as string); t.style.setProperty("stroke-width", "6");
        const t2 = svgEl("path", { d: `M${W.X(mx) - 9},${W.Y(my)}h18M${W.X(mx)},${W.Y(my) - 9}v18` }, gTop);
        t2.style.setProperty("stroke", "#fff"); t2.style.setProperty("stroke-width", "2");
      });
    }
    // distances
    if (showD) {
      const lab = svgEl("g", { class: "qb-halo" }, gTop);
      for (const p of pts) {
        const t = svgEl("text", { x: W.X(p.x) + 12, y: W.Y(p.y) - 10 }, lab);
        t.textContent = signed(signedDistance(line, p.x, p.y));
        t.style.fontSize = "19px";
        if (ySign(p.c) * f(line, p.x, p.y) <= 0) t.style.setProperty("font-weight", "700");
      }
    }
    // wrong points ringed
    for (const p of pts) if (ySign(p.c) * f(line, p.x, p.y) <= 0) {
      const r = svgEl("circle", { cx: W.X(p.x), cy: W.Y(p.y), r: 14, fill: "none" }, gTop);
      r.style.setProperty("stroke", "var(--qb-fg, #222)"); r.style.setProperty("stroke-width", "2.5"); r.style.setProperty("stroke-dasharray", "4 3");
    }
    if (lastPerc >= 0 && pts[lastPerc]) {
      const p = pts[lastPerc];
      const r = svgEl("circle", { cx: W.X(p.x), cy: W.Y(p.y), r: 18, fill: "none" }, gTop);
      r.style.setProperty("stroke", "#e6a700"); r.style.setProperty("stroke-width", "4");
    }
    // arrow w from the foot
    const s = arrowScale(), tip: [number, number] = [foot[0] + s * line.w[0], foot[1] + s * line.w[1]];
    const X0 = W.X(foot[0]), Y0 = W.Y(foot[1]), X1 = W.X(tip[0]), Y1 = W.Y(tip[1]);
    const ang = Math.atan2(Y1 - Y0, X1 - X0);
    const ar = svgEl("path", { d: `M${X0},${Y0}L${X1},${Y1}M${X1 - 16 * Math.cos(ang - 0.4)},${Y1 - 16 * Math.sin(ang - 0.4)}L${X1},${Y1}L${X1 - 16 * Math.cos(ang + 0.4)},${Y1 - 16 * Math.sin(ang + 0.4)}`, fill: "none" }, gHandles);
    ar.style.setProperty("stroke", "var(--qb-fg, #222)"); ar.style.setProperty("stroke-width", "3.5");
    const wl = svgEl("text", { x: X1 + 10 * Math.cos(ang) - 6, y: Y1 + 14 * Math.sin(ang) + 6, class: "qb-halo" }, gHandles);
    wl.textContent = "w"; wl.style.fontSize = "22px"; wl.style.setProperty("font-weight", "700"); wl.style.setProperty("font-style", "italic");
    handle(X0, Y0, "var(--qb-fg, #222)", (x, y) => { foot = [x, y]; line = { w: line.w, b: -(line.w[0] * x + line.w[1] * y) }; draw(""); });
    handle(X1, Y1, "#e6a700", (x, y) => {
      const nw = norm(line), dx = x - foot[0], dy = y - foot[1], dl = Math.hypot(dx, dy);
      if (dl < 1e-6) return;
      const w: [number, number] = [nw * dx / dl, nw * dy / dl];
      line = { w, b: -(w[0] * foot[0] + w[1] * foot[1]) };
      draw("");
    });
    // readout
    const e = errors(line, pts), mm = minMargin(line, pts);
    let txt = `f(x) = ${fmt(line.w[0])} x₁ ${line.w[1] >= 0 ? "+" : "−"} ${fmt(Math.abs(line.w[1]))} x₂ ${line.b >= 0 ? "+" : "−"} ${fmt(Math.abs(line.b))}\n‖w‖ = ${fmt(norm(line))}\nerrors: ${e} of ${pts.length}\nsmallest y·f/‖w‖: ${Number.isFinite(mm) ? signed(mm) : "–"}`;
    if (pts.length <= 6) txt += "\n" + pts.map(p => `(${fmt(p.x, 1)}, ${fmt(p.y, 1)}) ${p.c ? "+1" : "−1"}: f = ${signed(f(line, p.x, p.y))}, d = ${signed(signedDistance(line, p.x, p.y))}`).join("\n");
    ro.textContent = txt;
    note.textContent = msg || preset.note;
  }

  function load(p: LinPreset): void {
    preset = p; showMean = false; lastPerc = -1;
    const ax = axesAttr && axesAttr.length === 2 ? axesAttr : p.axes ?? ["x₁", "x₂"];
    W.setRange(p.range[0], p.range[1], ax[0], ax[1]);
    line = { w: [...p.line.w], b: p.line.b };
    placeFoot();
    presetBtns.forEach((b, i) => b.classList.toggle("active", list[i] === p));
    W.set(p.points() as LPoint[]);
    update();
  }
  dCb.checked = showD; nnCb.checked = showNN;
  load(p0);
}

autoMount(NAME, mount);
