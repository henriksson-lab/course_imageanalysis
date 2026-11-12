// Marching squares by hand, made interactive. A small grid of corner values (grey squares
// with numbers); the iso-value c is a slider. Each cell shows its case index (0..15) and its
// line segments; crossing points sit on the cell edges by linear interpolation (or at the
// midpoints). Saddle cells (cases 5 and 10) follow a rule: separate, join, or the asymptotic
// decider, which compares the saddle value f* of the bilinear interpolant with c. "Show the
// interpolant" shades every cell with the bilinear interpolant and draws its exact level curve
// (dashed), which the straight segments approximate. Click a corner to flip it across c, drag
// it up/down to change its value; click a cell to see its arithmetic. Uses lib/marching.ts.
// Auto-mounts on <div data-demo="marchsquares">. Optional: data-preset="bump|all16|join|separate|midpoint|exercise".
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, svgPoint } from "../lib/ui";
import { marchingSquares, squareCase, saddleValue, isSquareSaddle, edgeT, SaddleRule, EdgeInterp, SQ_EDGES } from "../lib/marching";

const NAME = "marchsquares";
const S = 540, PAD = 34;
const VMAX = 10;

interface Preset { id: string; label: string; n: number; values: number[]; c: number; interp?: EdgeInterp; saddle?: SaddleRule; interpolant?: boolean; select?: [number, number] }

function bump(n: number): number[] {
  const v: number[] = [];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++)
    v.push(Math.round(10 * (0.5 + 9 * Math.exp(-((x - 3.3) ** 2 + (y - 3.6) ** 2) / 5.5))) / 10);
  return v;
}
const ALL16_BITS = [
  [1, 1, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 0, 0],
  [1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 1], [0, 1, 1, 1, 1, 1]];
const all16 = (): number[] => ALL16_BITS.flatMap((row, y) => row.map((b, x) => (b ? 7 : 3) + (((x * 7 + y * 3) % 5) - 2) * 0.4));

const PRESETS: Preset[] = [
  { id: "bump", label: "one bump", n: 8, values: bump(8), c: 5, select: [2, 2] },
  { id: "all16", label: "all 16 cases", n: 6, values: all16(), c: 5 },
  { id: "join", label: "saddle: the decider joins", n: 4, c: 5, saddle: "decider", interpolant: true, select: [1, 1], values: [2, 3, 4, 2, 3, 8, 4, 3, 2, 3, 7, 3, 1, 2, 3, 2] },
  { id: "separate", label: "saddle: the decider separates", n: 4, c: 5, saddle: "decider", interpolant: true, select: [1, 1], values: [2, 3, 4, 2, 3, 6, 4, 3, 2, 3, 6.5, 3, 1, 2, 3, 2] },
  { id: "midpoint", label: "midpoint vs linear", n: 8, values: bump(8), c: 3, interp: "midpoint", interpolant: true },
  { id: "exercise", label: "exercise grid", n: 4, c: 5, values: [1, 3, 6, 2, 2, 7, 4, 3, 6, 4, 8, 4, 2, 3, 6, 1], select: [0, 1] },
];

const STYLE = `
[data-demo="${NAME}"] { gap: 1em; }
[data-demo="${NAME}"] .ms-plot { flex: 1 1 440px; min-width: 0; max-width: 560px; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 400px; gap: .4em; }
[data-demo="${NAME}"] .ms-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; min-height: 9.5em; }
[data-demo="${NAME}"] svg text { font-size: 18px; }
[data-demo="${NAME}"] svg .val { font-family: ui-monospace, monospace; font-weight: 600; pointer-events: none; }
[data-demo="${NAME}"] svg .case { fill: var(--qb-muted); pointer-events: none; }
[data-demo="${NAME}"] svg .corner { cursor: ns-resize; }
[data-demo="${NAME}"] svg .cell { cursor: pointer; }
.reveal [data-demo="${NAME}"] { font-size: 19px; }
.reveal [data-demo="${NAME}"] .ms-plot { max-width: 520px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .45em; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .25em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const f2 = (v: number) => (Math.round(v * 100) / 100).toString();
const f1 = (v: number) => v.toFixed(1);

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = {
    n: p0.n, values: [...p0.values], c: p0.c, interp: (p0.interp ?? "linear") as EdgeInterp,
    saddle: (p0.saddle ?? "decider") as SaddleRule, interpolant: !!p0.interpolant, cases: true,
    sel: (p0.select ?? null) as [number, number] | null,
  };

  const plot = htmlEl("div", "ms-plot", "", host);
  const svg = svgEl("svg", { viewBox: `0 0 ${S} ${S}`, width: "100%" }, plot);
  const gImg = svgEl("g", {}, svg), gGrid = svgEl("g", {}, svg), gCells = svgEl("g", {}, svg);
  const gExact = svgEl("g", {}, svg), gSeg = svgEl("g", {}, svg), gCorner = svgEl("g", {}, svg);
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "ms-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const cS = slider(panel, "iso-value c", 0, VMAX, 0.1, st.c, v => { st.c = v; draw(); });
  htmlEl("div", "ms-lab", "crossing points · saddle rule", panel);
  const row1 = htmlEl("div", "qb-presets", "", panel);
  const interpBtns = (["linear", "midpoint"] as EdgeInterp[]).map(m => button(row1, m, () => { st.interp = m; draw(); }));
  const row2 = htmlEl("div", "qb-presets", "", panel);
  const ruleBtns = (["separate", "join", "decider"] as SaddleRule[]).map(m => button(row2, m === "decider" ? "asymptotic decider" : m, () => { st.saddle = m; draw(); }));
  const chkI = checkbox(panel, "show the bilinear interpolant", st.interpolant, v => { st.interpolant = v; draw(); });
  const chkC = checkbox(panel, "show case numbers", st.cases, v => { st.cases = v; draw(); });
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Orange corners are above c. Click a corner to flip it across c, or drag it up/down. Click a cell for its arithmetic.", panel);

  function load(p: Preset): void {
    st.n = p.n; st.values = [...p.values]; st.c = p.c; st.interp = p.interp ?? "linear"; st.saddle = p.saddle ?? "decider";
    st.interpolant = !!p.interpolant; st.sel = p.select ?? null;
    cS.set(st.c); chkI.checked = st.interpolant; chkC.checked = st.cases;
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    draw();
  }

  // geometry: corner (x, y) at pixel (PAD + x * h, PAD + y * h)
  const h = () => (S - 2 * PAD) / (st.n - 1);
  const px = (x: number) => PAD + x * h(), py = (y: number) => PAD + y * h();
  const val = (x: number, y: number) => st.values[y * st.n + x];
  const cellValues = (x: number, y: number) => [val(x, y), val(x + 1, y), val(x + 1, y + 1), val(x, y + 1)];

  // ---- corner interaction ----
  let drag: { k: number; y0: number; v0: number; moved: boolean } | null = null;
  svg.addEventListener("pointermove", ev => {
    if (!drag) return;
    const [, y] = svgPoint(svg, ev);
    if (Math.abs(y - drag.y0) > 3) drag.moved = true;
    if (!drag.moved) return;
    st.values[drag.k] = Math.max(0, Math.min(VMAX, Math.round((drag.v0 - (y - drag.y0) / 25) * 10) / 10));
    draw();
  });
  const endDrag = () => {
    if (!drag) return;
    if (!drag.moved) {
      const v = st.values[drag.k];
      let w = Math.round((2 * st.c - v) * 10) / 10;
      if (w === v || (w > st.c) === (v > st.c)) w = v > st.c ? st.c - 1 : st.c + 1;
      st.values[drag.k] = Math.max(0, Math.min(VMAX, w));
    }
    drag = null; presetBtns.forEach(b => b.classList.remove("active"));
    draw();
  };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);

  function draw(): void {
    interpBtns.forEach((b, i) => b.classList.toggle("active", ["linear", "midpoint"][i] === st.interp));
    ruleBtns.forEach((b, i) => b.classList.toggle("active", ["separate", "join", "decider"][i] === st.saddle));
    const n = st.n, H = h(), c = st.c;
    for (const g of [gImg, gGrid, gCells, gExact, gSeg, gCorner]) g.replaceChildren();

    // background: bilinear interpolant (fine image) or plain
    if (st.interpolant) {
      const R = 24 * (n - 1), cv = document.createElement("canvas");
      cv.width = R; cv.height = R;
      const ctx = cv.getContext("2d")!, id = ctx.createImageData(R, R);
      for (let j = 0; j < R; j++) for (let i = 0; i < R; i++) {
        const X = (i + 0.5) / 24, Y = (j + 0.5) / 24, x0 = Math.min(n - 2, Math.floor(X)), y0 = Math.min(n - 2, Math.floor(Y));
        const fx = X - x0, fy = Y - y0, v = cellValues(x0, y0);
        const b = v[0] * (1 - fx) * (1 - fy) + v[1] * fx * (1 - fy) + v[2] * fx * fy + v[3] * (1 - fx) * fy;
        const g = 60 + 175 * b / VMAX, k = 4 * (j * R + i);
        const above = b > c;
        id.data[k] = above ? Math.min(255, g + 25) : g; id.data[k + 1] = above ? g : g; id.data[k + 2] = above ? g - 20 : g + 10; id.data[k + 3] = 255;
      }
      ctx.putImageData(id, 0, 0);
      svgEl("image", { href: cv.toDataURL(), x: PAD, y: PAD, width: S - 2 * PAD, height: S - 2 * PAD, preserveAspectRatio: "none" }, gImg);
      // exact level curve: marching squares on a fine sampling of the interpolant
      const F = 12, m = F * (n - 1) + 1, fine = new Float64Array(m * m);
      for (let j = 0; j < m; j++) for (let i = 0; i < m; i++) {
        const X = i / F, Y = j / F, x0 = Math.min(n - 2, Math.floor(X)), y0 = Math.min(n - 2, Math.floor(Y));
        const fx = X - x0, fy = Y - y0, v = cellValues(x0, y0);
        fine[j * m + i] = v[0] * (1 - fx) * (1 - fy) + v[1] * fx * (1 - fy) + v[2] * fx * fy + v[3] * (1 - fx) * fy;
      }
      const C = marchingSquares(fine, m, m, c, { saddle: "decider" });
      let d = "";
      for (let s = 0; s < C.segments.length; s += 2) {
        const a = C.segments[s], b = C.segments[s + 1];
        d += `M${px(C.points[2 * a] / F).toFixed(1)},${py(C.points[2 * a + 1] / F).toFixed(1)}L${px(C.points[2 * b] / F).toFixed(1)},${py(C.points[2 * b + 1] / F).toFixed(1)}`;
      }
      svgEl("path", { d, fill: "none", stroke: "var(--qb-c2)", "stroke-width": 2.5, "stroke-dasharray": "6 4" }, gExact);
    } else {
      svgEl("rect", { x: PAD, y: PAD, width: S - 2 * PAD, height: S - 2 * PAD, fill: "var(--qb-panel)" }, gImg);
    }
    // grid lines
    for (let k = 0; k < n; k++) {
      svgEl("line", { x1: px(k), y1: py(0), x2: px(k), y2: py(n - 1), stroke: "#999", "stroke-width": 1.2 }, gGrid);
      svgEl("line", { x1: px(0), y1: py(k), x2: px(n - 1), y2: py(k), stroke: "#999", "stroke-width": 1.2 }, gGrid);
    }
    // cells: click target, case index, selection
    for (let y = 0; y < n - 1; y++) for (let x = 0; x < n - 1; x++) {
      const v = cellValues(x, y), cs = squareCase(v, c), sel = st.sel && st.sel[0] === x && st.sel[1] === y;
      const r = svgEl("rect", { x: px(x), y: py(y), width: H, height: H, fill: "transparent", class: "cell", stroke: sel ? "var(--qb-c5)" : "none", "stroke-width": sel ? 5 : 0 }, gCells);
      r.addEventListener("click", () => { st.sel = [x, y]; draw(); });
      if (st.cases) {
        const t = svgEl("text", { x: px(x) + H / 2, y: py(y) + H / 2 + 6, "text-anchor": "middle", class: "case" }, gCells);
        t.textContent = String(cs);
        if (isSquareSaddle(cs)) { t.setAttribute("style", "fill: var(--qb-c5); font-weight: 700"); }
      }
    }
    // segments (thick) and crossing points
    const C = marchingSquares(st.values, n, n, c, { saddle: st.saddle, interp: st.interp });
    let d = "";
    for (let s = 0; s < C.segments.length; s += 2) {
      const a = C.segments[s], b = C.segments[s + 1];
      d += `M${px(C.points[2 * a]).toFixed(1)},${py(C.points[2 * a + 1]).toFixed(1)}L${px(C.points[2 * b]).toFixed(1)},${py(C.points[2 * b + 1]).toFixed(1)}`;
    }
    svgEl("path", { d, fill: "none", stroke: "var(--qb-c1)", "stroke-width": 5, "stroke-linecap": "round" }, gSeg);
    for (let i = 0; i < C.points.length; i += 2)
      svgEl("circle", { cx: px(C.points[i]), cy: py(C.points[i + 1]), r: 5, fill: "var(--qb-c1)" }, gSeg);
    // corners with values
    const rad = Math.min(19, H * 0.3);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const k = y * n + x, v = st.values[k], above = v > c;
      const g = svgEl("g", { class: "corner" }, gCorner);
      svgEl("circle", { cx: px(x), cy: py(y), r: rad, fill: above ? "#ff7f0e" : "#fff", stroke: "#444", "stroke-width": 1.5 }, g);
      const t = svgEl("text", { x: px(x), y: py(y) + 6, "text-anchor": "middle", class: "val" }, g);
      t.textContent = f1(v);
      if (above) t.setAttribute("style", "fill: #fff");
      g.style.touchAction = "none";
      g.addEventListener("pointerdown", ev => {
        ev.preventDefault();
        svg.setPointerCapture(ev.pointerId);
        drag = { k, y0: svgPoint(svg, ev)[1], v0: v, moved: false };
      });
    }
    readout(C.lines.length);
  }

  function readout(nLines: number): void {
    const c = st.c;
    const lines: string[] = [];
    if (st.sel && st.sel[0] < st.n - 1 && st.sel[1] < st.n - 1) {
      const [x, y] = st.sel, v = cellValues(x, y), bits = v.map(a => (a > c ? 1 : 0));
      const cs = squareCase(v, c);
      lines.push(`cell (${x}, ${y})   corners 0 1 2 3 = ${v.map(f1).join(", ")}`);
      lines.push(`bits ${bits.join(" ")} → case ${bits.map((b, k) => b ? 1 << k : 0).filter(a => a).join(" + ") || "0"} = ${cs}`);
      for (let e = 0; e < 4; e++) {
        const [a, b] = SQ_EDGES[e];
        if (bits[a] === bits[b]) continue;
        const t = edgeT(v[a], v[b], c, st.interp);
        lines.push(`edge ${a}–${b}: t = (${f1(c)} − ${f1(v[a])}) / (${f1(v[b])} − ${f1(v[a])}) = ${st.interp === "midpoint" ? "0.5 (midpoint)" : f2(t)}`);
      }
      if (isSquareSaddle(cs)) {
        const f = saddleValue(v);
        lines.push(`saddle: f* = (${f1(v[0])}·${f1(v[2])} − ${f1(v[1])}·${f1(v[3])}) / (${f1(v[0])} + ${f1(v[2])} − ${f1(v[1])} − ${f1(v[3])})`);
        lines.push(`        = ${f2(f)} ${f > c ? ">" : "≤"} c: decider ${f > c ? "joins" : "separates"} the above corners`);
      }
    } else lines.push("click a cell");
    lines.push(`contour pieces: ${nLines}`);
    ro.textContent = lines.join("\n");
  }

  load(p0);
}

autoMount(NAME, mount);
