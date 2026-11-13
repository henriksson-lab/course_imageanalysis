// regionfilter: the measurement table and a scatter plot of two properties, linked to the
// label image (labelling day, unit label-measure/; the course's feature-table + scatter
// shell). Pipeline: threshold → opening → fill holes → label (4 or 8) → measure → border
// rule (keep, remove objects touching the border, or Gundersen's counting frame) → filter
// by a rectangle brushed in the scatter plot → count. Click an object or a table row to
// select it; drag in the scatter plot to keep only the objects inside the rectangle.
//
// Auto-mounts on <div data-demo="regionfilter">. Options: data-preset (id below).
import { autoMount, htmlEl, button, slider, checkbox } from "../lib/ui";
import { Grey } from "../lib/imageops";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { Plot } from "../lib/plot";
import { labelColor, toCss } from "../lib/colour";
import { mixedPopulation, runPipeline, PROPS, Row, BorderRule, PipelineOptions } from "./pipeline";

const NAME = "regionfilter";
type PropId = typeof PROPS[number]["id"];
interface Brush { x0: number; x1: number; y0: number; y1: number }
interface Preset { id: string; label: string; input: "synth" | "cell"; opts: Partial<PipelineOptions>; x: PropId; y: PropId; brush?: Brush | null }

const BASE: PipelineOptions = { threshold: 100, dark: false, openRadius: 0, fill: true, conn: 8, border: "keep", frameMargin: 20 };
const PRESETS: Preset[] = [
  { id: "start", label: "the pipeline", input: "synth", opts: {}, x: "area", y: "eccentricity", brush: null },
  { id: "debris", label: "debris: filter by area", input: "synth", opts: {}, x: "area", y: "solidity", brush: { x0: 40, x1: 1e9, y0: -1e9, y1: 1e9 } },
  { id: "rods", label: "rods vs round: eccentricity", input: "synth", opts: {}, x: "area", y: "eccentricity", brush: { x0: 40, x1: 1e9, y0: -1e9, y1: 0.93 } },
  { id: "pairs", label: "touching nuclei: solidity flags them", input: "synth", opts: {}, x: "area", y: "solidity", brush: { x0: 40, x1: 1e9, y0: 0.91, y1: 1e9 } },
  { id: "frame", label: "border objects: counting frame", input: "synth", opts: { border: "frame" }, x: "area", y: "solidity", brush: { x0: 40, x1: 1e9, y0: -1e9, y1: 1e9 } },
  { id: "open", label: "opening removes the specks", input: "synth", opts: { openRadius: 2 }, x: "area", y: "solidity", brush: null },
  { id: "conn", label: "connectivity changes the count", input: "cell", opts: { threshold: 95, dark: true, fill: false, conn: 4 }, x: "area", y: "solidity", brush: null },
  { id: "cell", label: "the micrograph", input: "cell", opts: { threshold: 95, dark: true, openRadius: 1, fill: true, conn: 8 }, x: "area", y: "solidity", brush: { x0: 30, x1: 1e9, y0: -1e9, y1: 1e9 } },
];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; flex-wrap: wrap; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .rf-img { flex: 0 1 340px; min-width: 220px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .rf-img canvas { width: 100%; image-rendering: pixelated; cursor: pointer; border: 1px solid var(--qb-grid); }
.qb-demo[data-demo="${NAME}"] .rf-mid { flex: 1 1 330px; min-width: 0; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .rf-right { flex: 1 1 340px; max-width: 480px; min-width: 0; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .rf-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .rf-tab { max-height: 228px; overflow-y: auto; border: 1px solid var(--qb-grid); border-radius: 4px; }
.qb-demo[data-demo="${NAME}"] .rf-tab table { border-collapse: collapse; width: 100%; font-size: 14px; }
.qb-demo[data-demo="${NAME}"] .rf-tab th { position: sticky; top: 0; background: var(--qb-panel); cursor: pointer; font-weight: 600; padding: .1em .3em; text-align: right; white-space: nowrap; }
.qb-demo[data-demo="${NAME}"] .rf-tab td { white-space: nowrap; padding: .03em .3em; text-align: right; font-family: ui-monospace, monospace; border-bottom: 1px solid var(--qb-grid); }
.qb-demo[data-demo="${NAME}"] .rf-tab tr.out td { color: var(--qb-muted); text-decoration: line-through; }
.qb-demo[data-demo="${NAME}"] .rf-tab tr.sel td { background: #fff3b0; }
.qb-demo[data-demo="${NAME}"] .rf-tab tr { cursor: pointer; }
.qb-demo[data-demo="${NAME}"] .rf-sw { display: inline-block; width: .8em; height: .8em; border-radius: 2px; vertical-align: -.05em; }
.qb-demo[data-demo="${NAME}"] .qb-slider { font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 14px; padding: .12em .45em; }
.qb-demo[data-demo="${NAME}"] .rf-count { font-size: 18px; font-weight: 600; }
.qb-demo[data-demo="${NAME}"] .rf-credit { font-size: 12px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .rf-ex .qb-btn { font-size: 13px; padding: .1em .45em; }
.qb-demo[data-demo="${NAME}"] .rf-ex .qb-presets { gap: .25em; }
.qb-demo[data-demo="${NAME}"] .rf-ex { display: flex; flex-direction: column; gap: .3em; padding-bottom: .2em; border-bottom: 1px solid var(--qb-grid); }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .rf-tab { max-height: 200px; }
.reveal .qb-demo[data-demo="${NAME}"] { flex-wrap: nowrap; }
@media (max-width: 1000px) { .qb-demo[data-demo="${NAME}"] { flex-wrap: wrap; } }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const propOf = (id: PropId) => PROPS.find(p => p.id === id)!;
const val = (r: Row, id: PropId) => r[id] as number;

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const synth = mixedPopulation(3, 256);
  let cell: Grey | null = null;
  let input: "synth" | "cell" = "synth";
  const o: PipelineOptions = { ...BASE };
  let xProp: PropId = "area", yProp: PropId = "eccentricity", brush: Brush | null = null;
  let sel = -1, sortBy: PropId | "label" = "label", sortDesc = false;
  let res: ReturnType<typeof runPipeline> | null = null;

  // ---- left: image ----
  const left = htmlEl("div", "rf-img", "", host);
  const can = htmlEl("canvas", "", "", left);
  can.width = 256; can.height = 256;
  const irow = htmlEl("div", "qb-presets", "", left);
  const bS = button(irow, "synthetic", () => setInput("synth"));
  const bC = button(irow, "micrograph", () => setInput("cell"));
  const showMask = checkbox(irow, "mask only", false, () => draw());
  const credit = htmlEl("div", "rf-credit", "", left);

  // ---- middle: controls, readout, table ----
  // pages: the examples on top of this column (beside the image); slides keep them under the plot
  const onSlide = !!host.closest(".reveal");
  const mid = htmlEl("div", onSlide ? "rf-mid" : "rf-mid qb-panel", "", host);
  const exBox = onSlide ? null : htmlEl("div", "rf-ex", "", mid);
  const thr = slider(mid, "threshold", 0, 255, 1, o.threshold, v => { o.threshold = v; run(); });
  const opn = slider(mid, "open radius", 0, 4, 1, o.openRadius, v => { o.openRadius = v; run(); });
  const crow = htmlEl("div", "qb-presets", "", mid);
  const fillBox = checkbox(crow, "fill holes", o.fill, v => { o.fill = v; run(); });
  const c8 = button(crow, "8-conn.", () => { o.conn = 8; run(); });
  const c4 = button(crow, "4-conn.", () => { o.conn = 4; run(); });
  const brow = htmlEl("div", "qb-presets", "", mid);
  htmlEl("span", "rf-lab", "border:", brow);
  const RULES: [BorderRule, string][] = [["keep", "keep"], ["remove", "remove touching"], ["frame", "counting frame"]];
  const rb = RULES.map(([r, l]) => button(brow, l, () => { o.border = r; run(); }));
  const count = htmlEl("div", "rf-count", "", mid);
  const tabBox = htmlEl("div", "rf-tab", "", mid);

  // ---- right: scatter ----
  const right = htmlEl("div", "rf-right", "", host);
  const arow = htmlEl("div", "qb-presets", "", right);
  const bx = button(arow, "", () => { xProp = PROPS[(PROPS.findIndex(p => p.id === xProp) + 1) % PROPS.length].id; brush = null; refresh(); });
  const by = button(arow, "", () => { yProp = PROPS[(PROPS.findIndex(p => p.id === yProp) + 1) % PROPS.length].id; brush = null; refresh(); });
  button(arow, "clear filter", () => { brush = null; refresh(); });
  const plot = new Plot(right, { width: 420, height: 330, x: [0, 1], y: [0, 1], xlabel: "x", ylabel: "y", font: 17 });
  const dots = plot.scatter([], [], { r: 5.5 });
  const selDot = plot.scatter([], [], { r: 9, color: "none", stroke: "var(--qb-fg, #222)" });
  const brushRect = plot.mark((g, b: Brush | null) => {
    if (!b) return;
    const [xa, xb] = plot.x.domain, [ya, yb] = plot.y.domain;
    const X0 = plot.x(Math.max(Math.min(xa, xb), b.x0)), X1 = plot.x(Math.min(Math.max(xa, xb), b.x1));
    const Y0 = plot.y(Math.max(Math.min(ya, yb), b.y0)), Y1 = plot.y(Math.min(Math.max(ya, yb), b.y1));
    const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    r.setAttribute("x", String(Math.min(X0, X1))); r.setAttribute("y", String(Math.min(Y0, Y1)));
    r.setAttribute("width", String(Math.abs(X1 - X0))); r.setAttribute("height", String(Math.abs(Y1 - Y0)));
    r.style.cssText = "fill: var(--qb-c1, #1f77b4); fill-opacity: .1; stroke: var(--qb-c1, #1f77b4); stroke-width: 2; stroke-dasharray: 6 4";
    g.appendChild(r);
  }, [null as Brush | null]);
  htmlEl("div", "qb-note", "Drag a rectangle in the plot: only the objects inside are kept.", right);
  htmlEl("div", "rf-lab", "examples", exBox ?? right);
  const prow = htmlEl("div", "qb-presets", "", exBox ?? right);
  const pb = PRESETS.map(p => button(prow, p.label, () => load(p)));
  button(prow, "⟲ reset", () => load(p0));

  // brushing
  let drag: [number, number] | null = null;
  plot.svg.style.touchAction = "none";
  plot.svg.addEventListener("pointerdown", ev => { ev.preventDefault(); plot.svg.setPointerCapture(ev.pointerId); drag = plot.toData(ev); });
  plot.svg.addEventListener("pointermove", ev => {
    if (!drag) return;
    const q = plot.toData(ev);
    brush = { x0: Math.min(drag[0], q[0]), x1: Math.max(drag[0], q[0]), y0: Math.min(drag[1], q[1]), y1: Math.max(drag[1], q[1]) };
    brushRect.set(brush); refresh(false);
  });
  const up = (ev: PointerEvent) => {
    if (!drag) return;
    const q = plot.toData(ev);
    if (Math.hypot(q[0] - drag[0], q[1] - drag[1]) < 1e-9) brush = null;
    drag = null; refresh();
  };
  plot.svg.addEventListener("pointerup", up);
  plot.svg.addEventListener("pointercancel", () => { drag = null; });

  can.addEventListener("pointerdown", ev => {
    if (!res) return;
    const r = can.getBoundingClientRect();
    const x = Math.floor((ev.clientX - r.left) / r.width * 256), y = Math.floor((ev.clientY - r.top) / r.height * 256);
    if (x < 0 || y < 0 || x >= 256 || y >= 256) return;
    sel = res.labels[y * 256 + x] || -1;
    refresh();
  });

  const inBrush = (r: Row) => !brush || (val(r, xProp) >= brush.x0 && val(r, xProp) <= brush.x1 && val(r, yProp) >= brush.y0 && val(r, yProp) <= brush.y1);
  const kept = (r: Row) => r.counted && inBrush(r);

  function setInput(i: "synth" | "cell"): void {
    input = i;
    if (i === "cell" && !cell) { loadCell(256).then(g => { cell = g; run(); }); }
    run();
  }

  function run(): void {
    bS.classList.toggle("active", input === "synth"); bC.classList.toggle("active", input === "cell");
    c8.classList.toggle("active", o.conn === 8); c4.classList.toggle("active", o.conn === 4);
    rb.forEach((b, k) => b.classList.toggle("active", RULES[k][0] === o.border));
    credit.textContent = input === "cell" ? CELL_CREDIT + ". Objects: dark regions." : "Synthetic (seed 3): round nuclei, rods, specks, three touching pairs.";
    const img = input === "synth" ? synth.image : cell;
    if (!img) { count.textContent = "loading …"; return; }
    res = runPipeline(img, { ...o, dark: input === "cell" ? true : o.dark });
    if (sel > res.count) sel = -1;
    refresh();
  }

  function refresh(full = true): void {
    if (!res) return;
    bx.textContent = `x: ${propOf(xProp).label} ▸`; by.textContent = `y: ${propOf(yProp).label} ▸`;
    const rows = res.rows, keptRows = rows.filter(kept);
    const mean = keptRows.length ? keptRows.reduce((s, r) => s + r.area, 0) / keptRows.length : 0;
    count.textContent = `count: ${keptRows.length}  (labelled ${res.count})  ·  mean area ${mean.toFixed(1)}`;
    // scatter
    const xs = rows.map(r => val(r, xProp)), ys = rows.map(r => val(r, yProp));
    if (full) {
      const ext = (v: number[]) => { const a = Math.min(...v, 0), b = Math.max(...v, 1e-9); const pad = (b - a) * 0.06 || 1; return [a === 0 ? 0 : a - pad, b + pad] as [number, number]; };
      plot.setLabels({ xlabel: propOf(xProp).label, ylabel: propOf(yProp).label });
      plot.setDomain(rows.length ? ext(xs) : [0, 1], rows.length ? ext(ys) : [0, 1]);
    }
    dots.set(xs, ys, rows.map(r => (kept(r) ? toCss(labelColor(r.label)) : "var(--qb-grid, #ccc)")));
    const s = rows.find(r => r.label === sel);
    selDot.set(s ? [val(s, xProp)] : [], s ? [val(s, yProp)] : []);
    brushRect.set(brush);
    if (full) table(rows);
    draw();
  }

  function table(rows: Row[]): void {
    const cols: { id: PropId | "label"; label: string; fmt: (r: Row) => string }[] = [
      { id: "label", label: "#", fmt: r => String(r.label) },
      ...PROPS.filter(p => ["area", "eccentricity", "solidity", "formFactor"].includes(p.id)).map(p => ({ id: p.id, label: p.label.replace("eccentricity", "ecc.").replace("form factor", "form f."), fmt: (r: Row) => p.fmt(val(r, p.id)) })),
    ];
    const sorted = rows.slice().sort((a, b) => { const d = (sortBy === "label" ? a.label - b.label : val(a, sortBy) - val(b, sortBy)); return sortDesc ? -d : d; });
    tabBox.replaceChildren();
    const t = htmlEl("table", "", "", tabBox), hr = htmlEl("tr", "", "", t);
    for (const c of cols) {
      const th = htmlEl("th", "", c.label + (sortBy === c.id ? (sortDesc ? " ▼" : " ▲") : ""), hr);
      th.addEventListener("click", () => { if (sortBy === c.id) sortDesc = !sortDesc; else { sortBy = c.id; sortDesc = c.id !== "label"; } table(rows); });
    }
    for (const r of sorted) {
      const tr = htmlEl("tr", (kept(r) ? "" : "out") + (r.label === sel ? " sel" : ""), "", t);
      cols.forEach((c, i) => {
        const td = htmlEl("td", "", "", tr);
        if (i === 0) { const sw = htmlEl("span", "rf-sw", "", td); sw.style.background = toCss(labelColor(r.label)); td.append(" " + c.fmt(r)); }
        else td.textContent = c.fmt(r);
      });
      tr.addEventListener("click", () => { sel = r.label; refresh(); });
    }
    const selRow = tabBox.querySelector("tr.sel") as HTMLElement | null;
    if (selRow) selRow.scrollIntoView({ block: "nearest" });
  }

  function draw(): void {
    if (!res) return;
    const img = input === "synth" ? synth.image : cell!;
    const ctx = can.getContext("2d")!, id = ctx.createImageData(256, 256);
    const keep = new Uint8Array(res.count + 1);
    for (const r of res.rows) keep[r.label] = kept(r) ? 1 : 2;
    const cols = Array.from({ length: res.count + 1 }, (_, l) => labelColor(l).map(v => v * 255));
    for (let i = 0; i < 256 * 256; i++) {
      const g = Math.max(0, Math.min(255, img.data[i])), l = res.labels[i];
      let rgb = showMask.checked ? [255, 255, 255] : [g, g, g];
      if (showMask.checked && res.mask[i] && !l) rgb = [60, 60, 60];
      if (l) {
        const c = keep[l] === 1 ? cols[l] : [150, 150, 150], a = showMask.checked ? 1 : keep[l] === 1 ? 0.55 : 0.35;
        rgb = rgb.map((v, k) => v * (1 - a) + c[k] * a);
        if (l === sel) {
          const x = i % 256, y = (i - x) / 256;
          const edge = x === 0 || y === 0 || x === 255 || y === 255 || res.labels[i - 1] !== l || res.labels[i + 1] !== l || res.labels[i - 256] !== l || res.labels[i + 256] !== l;
          if (edge) rgb = [255, 220, 0];
        }
      }
      id.data[4 * i] = rgb[0]; id.data[4 * i + 1] = rgb[1]; id.data[4 * i + 2] = rgb[2]; id.data[4 * i + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    if (o.border === "frame") {
      const m = o.frameMargin ?? 20, x1 = 256 - m, y1 = 256 - m;
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#2ca02c"; ctx.beginPath(); ctx.moveTo(m, y1); ctx.lineTo(m, m); ctx.lineTo(x1, m); ctx.stroke();
      ctx.strokeStyle = "#d62728"; ctx.beginPath(); ctx.moveTo(x1, 0); ctx.lineTo(x1, y1); ctx.lineTo(0, y1); ctx.stroke();
    }
  }

  function load(p: Preset): void {
    Object.assign(o, BASE, p.opts);
    thr.set(o.threshold); opn.set(o.openRadius); fillBox.checked = o.fill;
    xProp = p.x; yProp = p.y; brush = p.brush ?? null; sel = -1;
    pb.forEach((b, k) => b.classList.toggle("active", PRESETS[k] === p));
    setInput(p.input);
  }
  load(p0);
}

autoMount(NAME, mount);
