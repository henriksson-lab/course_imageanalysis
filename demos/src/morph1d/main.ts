// morph1d: grey-level morphology of a 1D profile (morphology day, unit morph-grey). The
// structuring element is a bar of width w (flat) or a parabola cap (non-flat) that rolls
// UNDER the profile (opening: push it up as far as it goes; the top of the bar traces the
// opening) or OVER it (closing). "position" moves it, "sweep" animates it, and the trace it
// leaves behind is drawn. Output traces: erosion, dilation, opening, closing, white and black
// top-hat, morphological gradient, median, and |derivative| for comparison. Drag the round
// handles to change the profile. One profile is a line across the course micrograph.
// Options: data-preset (narrow | slope | noise | gradient | cell).
import { autoMount, htmlEl, slider, presets, checkbox, button, rng, draggable } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { Profile, seHeights, erode1, dilate1, open1, close1, median1, absDerivative, fromPoints } from "./m1d";

const NAME = "morph1d";
const N = 120;

type TraceId = "erode" | "dilate" | "open" | "close" | "wth" | "bth" | "grad" | "median" | "deriv";
const TRACES: { id: TraceId; label: string; color: string; dash?: string }[] = [
  { id: "erode", label: "erosion", color: "#1f77b4" },
  { id: "dilate", label: "dilation", color: "#d62728" },
  { id: "open", label: "opening", color: "#ff7f0e" },
  { id: "close", label: "closing", color: "#9467bd" },
  { id: "wth", label: "white top-hat", color: "#2ca02c" },
  { id: "bth", label: "black top-hat", color: "#8c564b" },
  { id: "grad", label: "gradient", color: "#e377c2" },
  { id: "median", label: "median", color: "#17becf" },
  { id: "deriv", label: "|derivative|", color: "#555", dash: "6 4" },
];

interface Preset {
  id: string; label: string; pts?: [number, number][]; noise?: "saltpepper" | "gauss"; cell?: boolean;
  w: number; shape: "flat" | "parabola"; roll: "under" | "over"; traces: TraceId[];
}
const PRESETS: Preset[] = [
  {
    id: "narrow", label: "narrow peak goes, wide peak stays", w: 11, shape: "flat", roll: "under", traces: ["open"],
    pts: [[0, 60], [24, 60], [28, 200], [32, 60], [62, 60], [72, 190], [94, 190], [104, 60], [119, 60]],
  },
  {
    id: "slope", label: "peaks on a slope: top-hat", w: 15, shape: "flat", roll: "under", traces: ["open", "wth"],
    pts: [[0, 40], [18, 58], [21, 120], [24, 64], [48, 88], [51, 170], [54, 94], [80, 120], [83, 160], [86, 126], [104, 144], [107, 220], [110, 150], [119, 160]],
  },
  {
    id: "noise", label: "noise: min, median, max", w: 5, shape: "flat", roll: "under", traces: ["erode", "median", "dilate"], noise: "saltpepper",
    pts: [[0, 70], [58, 70], [62, 180], [119, 180]],
  },
  {
    id: "gradient", label: "gradient vs derivative", w: 3, shape: "flat", roll: "under", traces: ["grad", "deriv"],
    pts: [[0, 50], [38, 50], [40, 200], [70, 200], [100, 60], [119, 60]],
  },
  { id: "cell", label: "a line across the cell", w: 9, shape: "flat", roll: "over", traces: ["close", "bth"], cell: true },
];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: 1em; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] .m1-plot { flex: 1 1 600px; min-width: 0; max-width: 780px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 340px; gap: .45em; }
.qb-demo[data-demo="${NAME}"] .m1-lab { font-size: 15px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .m1-traces { display: grid; grid-template-columns: 1fr 1fr; gap: .1em .6em; }
.qb-demo[data-demo="${NAME}"] .m1-traces .qb-check { font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .m1-credit { font-size: 12px; color: var(--qb-muted); }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .15em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .m1-plot { max-width: 780px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function noiseFor(kind: Preset["noise"], seed: number): Float64Array {
  const r = rng(seed), out = new Float64Array(N);
  if (kind === "saltpepper") for (let x = 0; x < N; x++) { const u = r.unif(); out[x] = u < 0.07 ? 120 : u < 0.14 ? -120 : 0; }
  if (kind === "gauss") for (let x = 0; x < N; x++) out[x] = 6 * r.norm();
  return out;
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = {
    pts: [] as [number, number][], noise: noiseFor(undefined, 0), cellRow: null as Float64Array | null, useCell: false,
    w: 11, shape: "flat" as "flat" | "parabola", roll: "under" as "under" | "over", pos: 30, swept: -1,
    on: new Set<TraceId>(["open"]),
  };
  let f: Profile = new Float64Array(N);
  let timer: number | null = null;
  let ero: Profile = new Float64Array(N), dil: Profile = new Float64Array(N), s = seHeights(0, "flat");

  const plotBox = htmlEl("div", "m1-plot", "", host);
  const plot = new Plot(plotBox, { width: 780, height: 430, x: [0, N - 1], y: [0, 255], xlabel: "position x", ylabel: "grey level", ariaLabel: "profile and morphological traces" });
  const credit = htmlEl("div", "m1-credit", "", plotBox);
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "m1-lab", "structuring element", panel);
  const wS = slider(panel, "width w", 1, 41, 2, 11, v => { st.w = v; draw(); });
  const shapeBtns = presets(panel, [
    { label: "flat bar", apply: () => { st.shape = "flat"; draw(); } },
    { label: "parabola (non-flat)", apply: () => { st.shape = "parabola"; draw(); } },
  ]);
  const rollBtns = presets(panel, [
    { label: "roll under (opening)", apply: () => { st.roll = "under"; st.swept = -1; draw(); } },
    { label: "roll over (closing)", apply: () => { st.roll = "over"; st.swept = -1; draw(); } },
  ]);
  const posS = slider(panel, "position", 0, N - 1, 1, 30, v => { st.pos = v; draw(); });
  const sweepRow = htmlEl("div", "qb-presets", "", panel);
  const sweepBtn = button(sweepRow, "sweep", () => sweep());
  button(sweepRow, "clear trace", () => { stop(); st.swept = -1; draw(); });
  htmlEl("div", "m1-lab", "show", panel);
  const tbox = htmlEl("div", "m1-traces", "", panel);
  const checks = new Map<TraceId, HTMLInputElement>();
  for (const t of TRACES) {
    const c = checkbox(tbox, t.label, st.on.has(t.id), v => { if (v) st.on.add(t.id); else st.on.delete(t.id); draw(); });
    (c.parentElement as HTMLElement).style.color = t.color;
    checks.set(t.id, c);
  }
  htmlEl("div", "m1-lab", "examples", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  button(panel, "reset", () => load(p0));

  const lines = new Map<TraceId, ReturnType<Plot["line"]>>();
  const fLine = plot.line([], [], { color: "var(--qb-fg, #222)", width: 3.5 });
  for (const t of TRACES) lines.set(t.id, plot.line([], [], { color: t.color, width: 3, dash: t.dash, label: t.label }));
  const trace = plot.line([], [], { color: "#b35300", width: 5, opacity: 0.55 });
  const bar = plot.mark((g: SVGGElement) => drawBar(g), [] as [], {});
  const hg = document.createElementNS("http://www.w3.org/2000/svg", "g");
  plot.over.appendChild(hg);
  let handleEls: SVGCircleElement[] = [];

  function stop(): void { if (timer !== null) { clearInterval(timer); timer = null; } sweepBtn.classList.remove("active"); }
  function sweep(): void {
    if (timer !== null) { stop(); return; }
    st.pos = 0; st.swept = -1;
    sweepBtn.classList.add("active");
    timer = window.setInterval(() => {
      if (!host.isConnected || st.pos >= N - 1) { stop(); return; }
      st.pos++; posS.set(st.pos); draw();
    }, 45);
    draw();
  }

  async function load(p: Preset): Promise<void> {
    stop();
    st.useCell = !!p.cell;
    st.pts = (p.pts ?? []).map(q => [q[0], q[1]] as [number, number]);
    st.noise = noiseFor(p.noise, 11);
    st.w = p.w; wS.set(p.w); st.shape = p.shape; st.roll = p.roll;
    st.on = new Set(p.traces);
    for (const [id, c] of checks) c.checked = st.on.has(id);
    st.pos = p.roll === "under" ? 30 : 50; posS.set(st.pos); st.swept = -1;
    handleEls = []; hg.replaceChildren();
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    credit.textContent = p.cell ? "Row 128 of the micrograph, every second pixel. " + CELL_CREDIT : "";
    if (p.cell && !st.cellRow) {
      try {
        const img = await loadCell();
        st.cellRow = Float64Array.from({ length: N }, (_, i) => img.data[128 * img.width + 8 + 2 * i]);
      } catch { st.cellRow = new Float64Array(N).fill(128); }
    }
    draw();
  }

  function profile(): Profile {
    if (st.useCell && st.cellRow) return st.cellRow.slice();
    const base = fromPoints(st.pts, N);
    for (let x = 0; x < N; x++) base[x] = Math.max(0, Math.min(255, base[x] + st.noise[x]));
    return base;
  }

  function draw(): void {
    shapeBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.shape === "flat")));
    rollBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.roll === "under")));
    f = profile();
    const k = (st.w - 1) / 2;
    s = seHeights(k, st.shape);
    ero = erode1(f, s); dil = dilate1(f, s);
    const op = dilate1(ero, s), cl = erode1(dil, s);
    const xs = Float64Array.from({ length: N }, (_, i) => i);
    const vals: Record<TraceId, ArrayLike<number>> = {
      erode: ero, dilate: dil, open: op, close: cl,
      wth: f.map((v, i) => v - op[i]), bth: cl.map((v, i) => v - f[i]),
      grad: dil.map((v, i) => v - ero[i]), median: median1(f, k), deriv: absDerivative(f),
    };
    fLine.set(xs, f);
    for (const t of TRACES) lines.get(t.id)!.set(st.on.has(t.id) ? xs : [], st.on.has(t.id) ? vals[t.id] : []);
    // the trace left behind by the rolling SE (positions 0..pos)
    st.swept = Math.max(st.swept, st.pos);
    const T = new Float64Array(N).fill(NaN);
    for (let c = 0; c <= st.pos; c++) for (let b = -k; b <= k; b++) {
      const x = c + b;
      if (x < 0 || x >= N) continue;
      const v = st.roll === "under" ? ero[c] + s[b + k] : dil[c] - s[b + k];
      T[x] = Number.isNaN(T[x]) ? v : st.roll === "under" ? Math.max(T[x], v) : Math.min(T[x], v);
    }
    trace.set(xs, T);
    bar.render();
    if (handleEls.length !== (st.useCell ? 0 : st.pts.length)) makeHandles();
    handleEls.forEach((c, j) => { c.setAttribute("cx", String(plot.x(st.pts[j][0]))); c.setAttribute("cy", String(plot.y(st.pts[j][1]))); });
    plot.legend(TRACES.filter(t => st.on.has(t.id)).map(t => ({ label: t.label, color: t.color, dash: t.dash })), "tl");
  }

  function drawBar(g: SVGGElement): void {
    const k = (st.w - 1) / 2, c = st.pos, under = st.roll === "under";
    if (s.length !== st.w) return;
    const base = under ? ero[c] : dil[c];
    let d = "";
    for (let b = -k; b <= k; b++) {
      const x = c + b;
      if (x < 0 || x >= N) continue;
      const y = under ? base + s[b + k] : base - s[b + k];
      d += (d ? "L" : "M") + plot.x(x).toFixed(1) + "," + plot.y(y).toFixed(1);
    }
    // the SE as a thick bar (or cap) with a band on the side away from the profile
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", d);
    p.setAttribute("style", "fill:none;stroke:#ff7f0e;stroke-width:9;stroke-linecap:round;opacity:.9");
    g.appendChild(p);
    const o = document.createElementNS("http://www.w3.org/2000/svg", "line");
    const yb = plot.y(under ? base : base);
    o.setAttribute("x1", String(plot.x(c))); o.setAttribute("x2", String(plot.x(c)));
    o.setAttribute("y1", String(yb)); o.setAttribute("y2", String(under ? plot.box.b : plot.box.t));
    o.setAttribute("style", "stroke:#ff7f0e;stroke-width:2;stroke-dasharray:4 4");
    g.appendChild(o);
  }

  function makeHandles(): void {
    hg.replaceChildren();
    handleEls = [];
    if (st.useCell) return;
    st.pts.forEach((_q, j) => {
      const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("r", "9");
      handleEls.push(c);
      hg.appendChild(c);
      c.setAttribute("class", "handle");
      c.setAttribute("style", "fill:#fff;stroke:#222;stroke-width:2.5;cursor:grab");
      draggable(c as unknown as SVGElement, ev => plot.toData(ev), (_x, y) => { st.pts[j][1] = Math.round(Math.max(0, Math.min(255, y))); st.swept = -1; draw(); });
    });
  }

  load(p0);
}

autoMount(NAME, mount);
