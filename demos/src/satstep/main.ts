// Prefix sums and the summed-area table, step by step.
//  data-mode="1d": a row of values (drag the bars), its running total as a staircase, two
//    markers a, b; the range sum is the shaded bars and the rise of the staircase. "smooth"
//    replaces bars and staircase by a curve f and its antiderivative F.
//  data-mode="2d": an editable small image and its zero-padded table. Step fills one table
//    cell (pixel + left + up − up-left, highlighted); drag on the image to choose a
//    rectangle: its four corner entries light up with signs + − − + and the four-term sum
//    is written out, with a running count of additions "direct vs table".
// Optional: data-preset (1d: "exercise|ramp|spike"; 2d: "exercise|row|border|single").
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, svgPoint } from "../lib/ui";
import { Plot } from "../lib/plot";
import { prefixSum, integralImage, at, rectSum } from "../lib/integral";

const NAME = "satstep";
const STYLE = `
[data-demo="${NAME}"] .ss-main { flex: 1 1 560px; min-width: 0; max-width: 760px; display: flex; flex-direction: column; gap: .2em; }
[data-demo="${NAME}"] .ss-grids { width: 100%; touch-action: none; }
[data-demo="${NAME}"] .qb-panel { flex: 0 1 300px; }
[data-demo="${NAME}"] .ss-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.3em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; }
[data-demo="${NAME}"] .ss-cell { cursor: pointer; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .ss-main { flex-basis: 720px; max-width: 760px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .3em; flex-basis: 330px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 16px; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const C_PLUS = "#2ca02c", C_MINUS = "#d62728", C_RECT = "#1f77b4", C_FOCUS = "#ff7f0e";

// ======================================================================================
// 1D

const ROW_PRESETS: { id: string; label: string; v: number[]; a: number; b: number }[] = [
  { id: "exercise", label: "exercise row", v: [3, 1, 4, 1, 5, 9, 2, 6], a: 3, b: 7 },
  { id: "ramp", label: "ramp (16)", v: [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8], a: 2, b: 13 },
  { id: "spike", label: "one spike", v: [1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1, 1], a: 4, b: 9 },
];

function mount1d(host: HTMLElement): void {
  const p0 = ROW_PRESETS.find(p => p.id === host.dataset.preset) ?? ROW_PRESETS[0];
  const st = { v: [...p0.v], a: p0.a, b: p0.b, smooth: false, tries: 0 };
  const main = htmlEl("div", "ss-main", "", host);
  const top = new Plot(main, { width: 700, height: 250, x: [0.5, 8.5], y: [0, 10], ylabel: "value f", xticks: 8 });
  const bot = new Plot(main, { width: 700, height: 270, x: [0.5, 8.5], y: [0, 40], xlabel: "position i", ylabel: "running total S" });
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ss-lab", "presets", panel);
  const pbtns = presets(panel, ROW_PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const aS = slider(panel, "a", 1, 8, 1, st.a, v => { st.a = Math.min(v, st.b); aS.set(st.a); draw(); });
  const bS = slider(panel, "b", 1, 8, 1, st.b, v => { st.b = Math.max(v, st.a); bS.set(st.b); draw(); });
  checkbox(panel, "smooth (f and its antiderivative F)", false, v => { st.smooth = v; draw(); });
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Drag up or down on the bars to change the values (0–9).", panel);

  function load(p: typeof ROW_PRESETS[0]): void {
    st.v = [...p.v]; st.a = p.a; st.b = p.b;
    pbtns.forEach((b, i) => b.classList.toggle("active", ROW_PRESETS[i] === p));
    draw();
  }

  // drag bars
  let dragging = false;
  const setBar = (ev: PointerEvent) => {
    const [x, y] = top.toData(ev);
    const i = Math.round(x) - 1;
    if (i < 0 || i >= st.v.length) return;
    st.v[i] = Math.max(0, Math.min(9, Math.round(y)));
    pbtns.forEach(b => b.classList.remove("active"));
    draw();
  };
  top.svg.style.touchAction = "none";
  top.svg.addEventListener("pointerdown", ev => { ev.preventDefault(); dragging = true; top.svg.setPointerCapture(ev.pointerId); setBar(ev); });
  top.svg.addEventListener("pointermove", ev => { if (dragging) setBar(ev); });
  top.svg.addEventListener("pointerup", () => { dragging = false; });

  // smooth f: cosine interpolation through the values at integer positions
  const fSmooth = (x: number) => {
    const n = st.v.length, xi = Math.min(n, Math.max(1, x));
    const i = Math.floor(xi), t = xi - i;
    const a = st.v[i - 1], b = st.v[Math.min(n - 1, i)];
    return a + (b - a) * (1 - Math.cos(Math.PI * t)) / 2;
  };

  function draw(): void {
    const n = st.v.length;
    aS.input.max = bS.input.max = String(n);
    if (st.b > n) st.b = n;
    if (st.a > st.b) st.a = st.b;
    aS.set(st.a); bS.set(st.b);
    const P = prefixSum(st.v);
    const ymax = Math.max(10, Math.ceil(P[n] / 10) * 10);
    top.clear(); bot.clear();
    top.setDomain([0.5, n + 0.5], [0, 10]);
    bot.setDomain([0.5 - 1, n + 0.5], [0, ymax]);
    const xs = st.v.map((_, i) => i + 1);
    const inR = (i: number) => i + 1 >= st.a && i + 1 <= st.b;
    let range: number, direct: string;
    if (!st.smooth) {
      top.bars(xs, st.v, { colors: st.v.map((_, i) => (inR(i) ? C_RECT : "#bbb")), barWidth: 0.82 });
      st.v.forEach((v, i) => top.text(i + 1, v + 0.3, String(v), { anchor: "middle", size: 18 }));
      // staircase: S[i] holds from i to i+1 (S[0] = 0 at position 0)
      const sx = [0, ...xs], sy = Array.from(P);
      bot.step(sx, sy, { color: "#555", width: 2.5 }, "mid");
      bot.scatter(sx, sy, { color: "#555", r: 5 });
      sx.forEach((x, i) => bot.text(x, sy[i] + ymax * 0.035, String(sy[i]), { anchor: "middle", size: 16 }));
      const lo = P[st.a - 1], hi = P[st.b];
      bot.band([st.a - 1, st.b], [lo, lo], [hi, hi], { color: C_RECT, fillOpacity: 0.15 });
      bot.arrow(st.b + 0.3, lo, st.b + 0.3, hi, { color: C_RECT, width: 3 });
      bot.scatter([st.a - 1], [lo], { color: C_MINUS, r: 8 });
      bot.scatter([st.b], [hi], { color: C_PLUS, r: 8 });
      range = hi - lo;
      direct = `${Math.max(0, st.b - st.a)} additions`;
      ro.textContent = `sum of f(${st.a}) … f(${st.b})\n` +
        `direct:  ${st.v.slice(st.a - 1, st.b).join(" + ")} = ${range}\n` +
        `         (${direct})\n` +
        `table:   S(${st.b}) − S(${st.a - 1}) = ${hi} − ${lo} = ${range}\n` +
        `         (1 subtraction)`;
    } else {
      top.band(Array.from({ length: 101 }, (_, k) => st.a - 0.5 + (st.b - st.a + 1) * k / 100), new Array(101).fill(0),
        Array.from({ length: 101 }, (_, k) => fSmooth(st.a - 0.5 + (st.b - st.a + 1) * k / 100)), { color: C_RECT, fillOpacity: 0.3 });
      top.fn(fSmooth, { color: "#333", width: 3 }, 300);
      // F(x) = ∫_{0.5}^{x} f
      const Fx: number[] = [], Xs: number[] = [];
      let acc = 0;
      const steps = 400, x0 = 0.5, x1 = n + 0.5, dx = (x1 - x0) / steps;
      for (let k = 0; k <= steps; k++) {
        const x = x0 + k * dx;
        if (k > 0) acc += 0.5 * (fSmooth(x - dx) + fSmooth(x)) * dx;
        Xs.push(x); Fx.push(acc);
      }
      const Fat = (x: number) => Fx[Math.round((x - x0) / dx)];
      bot.line(Xs, Fx, { color: "#333", width: 3 });
      const A = st.a - 0.5, Bb = st.b + 0.5, FA = Fat(A), FB = Fat(Bb);
      bot.band([A, Bb], [FA, FA], [FB, FB], { color: C_RECT, fillOpacity: 0.15 });
      bot.arrow(Bb + 0.3, FA, Bb + 0.3, FB, { color: C_RECT, width: 3 });
      bot.scatter([A], [FA], { color: C_MINUS, r: 8 });
      bot.scatter([Bb], [FB], { color: C_PLUS, r: 8 });
      range = FB - FA;
      ro.textContent = `∫ f from ${A} to ${Bb}\n= area under the curve\n= F(${Bb}) − F(${A})\n= ${FB.toFixed(2)} − ${FA.toFixed(2)} = ${range.toFixed(2)}`;
    }
  }
  load(p0);
}

// ======================================================================================
// 2D

interface Preset2d { id: string; label: string; n: number; m: number; img: number[]; rect: [number, number, number, number] | null; filled: boolean }
const EX = [
  2, 0, 1, 3, 1,
  1, 4, 2, 0, 2,
  3, 1, 5, 2, 0,
  0, 2, 1, 4, 3,
  1, 3, 0, 2, 5,
];
// rect: [x0, y0, x1, y1] inclusive 1-based pixel indices (column, row)
const PRESETS_2D: Preset2d[] = [
  { id: "exercise", label: "exercise image", n: 5, m: 5, img: EX, rect: null, filled: false },
  { id: "row", label: "1 × 8 row = 1D", n: 8, m: 1, img: [3, 1, 4, 1, 5, 9, 2, 6], rect: [4, 1, 8, 1], filled: true },
  { id: "border", label: "rectangle at the border", n: 5, m: 5, img: EX, rect: [1, 1, 3, 2], filled: true },
  { id: "single", label: "one pixel, four lookups", n: 5, m: 5, img: EX, rect: [3, 3, 3, 3], filled: true },
  { id: "inner", label: "inner 3 × 3", n: 5, m: 5, img: EX, rect: [2, 2, 4, 4], filled: true },
];

function mount2d(host: HTMLElement): void {
  const p0 = PRESETS_2D.find(p => p.id === host.dataset.preset) ?? PRESETS_2D[0];
  const st = {
    n: p0.n, m: p0.m, img: [...p0.img], filledTo: 0, rect: p0.rect as Preset2d["rect"],
    edit: false, direct: 0, table: 0, tries: 0,
  };
  const main = htmlEl("div", "ss-main", "", host);
  const svg = svgEl("svg", { class: "ss-grids", role: "img", "aria-label": "image and summed-area table" }, main);
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ss-lab", "presets", panel);
  const pbtns = presets(panel, PRESETS_2D.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "ss-lab", "fill the table", panel);
  const row1 = htmlEl("div", "qb-presets", "", panel);
  button(row1, "step", () => { if (st.filledTo < st.n * st.m) st.filledTo++; draw(); });
  button(row1, "fill all", () => { st.filledTo = st.n * st.m; draw(); });
  button(row1, "empty", () => { st.filledTo = 0; st.rect = null; draw(); });
  htmlEl("div", "ss-lab", "mouse on the image", panel);
  const row2 = htmlEl("div", "qb-presets", "", panel);
  const modeBtns = [
    button(row2, "choose rectangle", () => { st.edit = false; sync(); }),
    button(row2, "edit pixels (±1)", () => { st.edit = true; sync(); }),
  ];
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Choose rectangle: drag across the image. Edit: click the upper half of a pixel for +1, the lower half for −1.", panel);

  function sync(): void { modeBtns.forEach((b, i) => b.classList.toggle("active", (i === 1) === st.edit)); }
  function load(p: Preset2d): void {
    st.n = p.n; st.m = p.m; st.img = [...p.img]; st.rect = p.rect; st.filledTo = p.filled ? p.n * p.m : 0;
    st.direct = 0; st.table = 0; st.tries = 0; st.edit = false;
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS_2D[i] === p));
    if (st.rect) count();
    sync(); draw();
  }
  function count(): void {
    if (!st.rect) return;
    const [x0, y0, x1, y1] = st.rect;
    st.direct += (x1 - x0 + 1) * (y1 - y0 + 1) - 1; st.table += 3; st.tries++;
  }

  const CS = 50, GAP = 60, FONT = 22;
  let geo = { ix: 0, iy: 0, tx: 0, ty: 0 };

  // pointer on the image
  let dragFrom: [number, number] | null = null;
  const cellAt = (ev: PointerEvent): [number, number] | null => {
    const [px, py] = svgPoint(svg, ev);
    const x = Math.floor((px - geo.ix) / CS), y = Math.floor((py - geo.iy) / CS);
    if (x < 0 || y < 0 || x >= st.n || y >= st.m) return null;
    return [x, y];
  };
  svg.style.touchAction = "none";
  svg.addEventListener("pointerdown", ev => {
    const c = cellAt(ev);
    if (!c) return;
    ev.preventDefault();
    pbtns.forEach(b => b.classList.remove("active"));
    if (st.edit) {
      const [, py] = svgPoint(svg, ev);
      const upper = (py - geo.iy) / CS - c[1] < 0.5;
      const i = c[1] * st.n + c[0];
      st.img[i] = Math.max(0, Math.min(9, st.img[i] + (upper ? 1 : -1)));
      draw();
      return;
    }
    svg.setPointerCapture(ev.pointerId);
    dragFrom = c;
    st.rect = [c[0] + 1, c[1] + 1, c[0] + 1, c[1] + 1];
    st.filledTo = st.n * st.m;
    draw();
  });
  svg.addEventListener("pointermove", ev => {
    if (!dragFrom) return;
    const [px, py] = svgPoint(svg, ev);
    const x = Math.max(0, Math.min(st.n - 1, Math.floor((px - geo.ix) / CS)));
    const y = Math.max(0, Math.min(st.m - 1, Math.floor((py - geo.iy) / CS)));
    st.rect = [Math.min(dragFrom[0], x) + 1, Math.min(dragFrom[1], y) + 1, Math.max(dragFrom[0], x) + 1, Math.max(dragFrom[1], y) + 1];
    draw();
  });
  const up = () => { if (dragFrom) { dragFrom = null; count(); draw(); } };
  svg.addEventListener("pointerup", up);
  svg.addEventListener("pointercancel", up);

  function text(g: SVGElement, x: number, y: number, s: string, attrs: Record<string, string | number> = {}): void {
    const { fill, ...rest } = attrs;
    const t = svgEl("text", { x, y, "text-anchor": "middle", "dominant-baseline": "central", "font-size": FONT, ...rest }, g);
    if (fill) t.style.fill = String(fill);
    t.textContent = s;
  }

  function draw(): void {
    const { n, m } = st;
    const T = integralImage(st.img, n, m);
    const lab = 26;
    geo = { ix: lab, iy: 34 + lab, tx: lab + n * CS + GAP + lab, ty: 34 + lab };
    const W = geo.tx + (n + 1) * CS + 6, H = geo.ty + (m + 1) * CS + 6;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.replaceChildren();
    const g = svgEl("g", {}, svg);
    text(g, geo.ix + n * CS / 2, 16, "image I", { "font-weight": 600 });
    text(g, geo.tx + (n + 1) * CS / 2, 16, "table S", { "font-weight": 600 });
    // focus of the last filled step
    const focus = st.filledTo > 0 && st.filledTo <= n * m && !st.rect ? st.filledTo - 1 : -1;
    const fx = focus % n, fy = Math.floor(focus / n);
    const inRect = (x: number, y: number) => !!st.rect && x + 1 >= st.rect[0] && x + 1 <= st.rect[2] && y + 1 >= st.rect[1] && y + 1 <= st.rect[3];
    // image cells
    for (let y = 0; y < m; y++) for (let x = 0; x < n; x++) {
      const v = st.img[y * n + x];
      const shade = 255 - Math.round(v * 14);
      const r = svgEl("rect", { x: geo.ix + x * CS, y: geo.iy + y * CS, width: CS, height: CS, fill: `rgb(${shade},${shade},${shade})`, stroke: "#999", "stroke-width": 1, class: "ss-cell" }, g);
      if (inRect(x, y)) r.setAttribute("fill", "#cfe3f3");
      text(g, geo.ix + (x + 0.5) * CS, geo.iy + (y + 0.5) * CS, String(v));
    }
    for (let x = 0; x < n; x++) text(g, geo.ix + (x + 0.5) * CS, geo.iy - 14, String(x + 1), { "font-size": 15, fill: "#777" });
    for (let y = 0; y < m; y++) text(g, geo.ix - 13, geo.iy + (y + 0.5) * CS, String(y + 1), { "font-size": 15, fill: "#777" });
    if (st.rect) {
      const [x0, y0, x1, y1] = st.rect;
      svgEl("rect", { x: geo.ix + (x0 - 1) * CS, y: geo.iy + (y0 - 1) * CS, width: (x1 - x0 + 1) * CS, height: (y1 - y0 + 1) * CS, fill: "none", stroke: C_RECT, "stroke-width": 4 }, g);
    }
    if (focus >= 0) svgEl("rect", { x: geo.ix + fx * CS + 2, y: geo.iy + fy * CS + 2, width: CS - 4, height: CS - 4, fill: "none", stroke: C_FOCUS, "stroke-width": 4 }, g);
    // table cells
    const marks = new Map<string, { col: string; sign: string }>();
    if (focus >= 0) {
      marks.set(`${fx + 1},${fy + 1}`, { col: C_FOCUS, sign: "=" });
      marks.set(`${fx},${fy + 1}`, { col: C_PLUS, sign: "+" });
      marks.set(`${fx + 1},${fy}`, { col: C_PLUS, sign: "+" });
      marks.set(`${fx},${fy}`, { col: C_MINUS, sign: "−" });
    }
    if (st.rect) {
      const [x0, y0, x1, y1] = st.rect;
      marks.set(`${x1},${y1}`, { col: C_PLUS, sign: "+" });
      marks.set(`${x1},${y0 - 1}`, { col: C_MINUS, sign: "−" });
      marks.set(`${x0 - 1},${y1}`, { col: C_MINUS, sign: "−" });
      marks.set(`${x0 - 1},${y0 - 1}`, { col: C_PLUS, sign: "+" });
    }
    for (let y = 0; y <= m; y++) for (let x = 0; x <= n; x++) {
      const pad = x === 0 || y === 0;
      const filled = pad || ((y - 1) * n + (x - 1)) < st.filledTo;
      svgEl("rect", { x: geo.tx + x * CS, y: geo.ty + y * CS, width: CS, height: CS, fill: pad ? "#eceff3" : "#fff", stroke: "#999", "stroke-width": 1 }, g);
      const mk = marks.get(`${x},${y}`);
      if (mk) {
        svgEl("rect", { x: geo.tx + x * CS + 2, y: geo.ty + y * CS + 2, width: CS - 4, height: CS - 4, fill: mk.col, "fill-opacity": 0.18, stroke: mk.col, "stroke-width": 4 }, g);
        text(g, geo.tx + x * CS + 9, geo.ty + y * CS + 11, mk.sign, { "font-size": 17, fill: mk.col, "font-weight": 700 });
      }
      if (filled) text(g, geo.tx + (x + 0.5) * CS, geo.ty + (y + 0.5) * CS + 2, String(at(T, x, y)), { "font-size": at(T, x, y) >= 100 ? 18 : FONT, fill: pad ? "#888" : "#222" });
    }
    for (let x = 0; x <= n; x++) text(g, geo.tx + (x + 0.5) * CS, geo.ty - 14, String(x), { "font-size": 15, fill: "#777" });
    for (let y = 0; y <= m; y++) text(g, geo.tx - 13, geo.ty + (y + 0.5) * CS, String(y), { "font-size": 15, fill: "#777" });

    // readout
    let s = "";
    if (st.rect) {
      const [x0, y0, x1, y1] = st.rect;
      const D = at(T, x1, y1), Bv = at(T, x1, y0 - 1), Cv = at(T, x0 - 1, y1), A = at(T, x0 - 1, y0 - 1);
      const sum = rectSum(T, x0 - 1, y0 - 1, x1, y1);
      const px = (x1 - x0 + 1) * (y1 - y0 + 1);
      s = `rectangle x ${x0}–${x1}, y ${y0}–${y1} (${px} px)\n` +
        `S(${x1},${y1}) − S(${x1},${y0 - 1}) − S(${x0 - 1},${y1}) + S(${x0 - 1},${y0 - 1})\n` +
        `= ${D} − ${Bv} − ${Cv} + ${A} = ${sum}\n` +
        `additions: direct ${px - 1}, table 3\n` +
        `all ${st.tries} rectangles so far:\n  direct ${st.direct}, table ${st.table}`;
    } else if (focus >= 0) {
      const v = st.img[focus], L = at(T, fx, fy + 1), U = at(T, fx + 1, fy), UL = at(T, fx, fy);
      s = `S(${fx + 1},${fy + 1}) = I(${fx + 1},${fy + 1}) + S(${fx},${fy + 1}) + S(${fx + 1},${fy}) − S(${fx},${fy})\n` +
        `= ${v} + ${L} + ${U} − ${UL} = ${at(T, fx + 1, fy + 1)}\n` +
        `cells filled: ${st.filledTo} of ${n * m}`;
    } else s = `cells filled: ${st.filledTo} of ${n * m}\nPress step, or drag a rectangle on the image.`;
    ro.textContent = s;
  }
  load(p0);
}

function mount(host: HTMLElement): void {
  injectStyle();
  if ((host.dataset.mode ?? "2d") === "1d") mount1d(host); else mount2d(host);
}

autoMount(NAME, mount);
