// Course overview: all days as a dependency graph (data-demo="coursegraph", on overview.qmd).
// Parts are columns on a wide screen and horizontal bands otherwise; hard prerequisite edges solid,
// soft ("see also") dashed. Click a day, or one of its unit marks, to highlight its transitive hard
// prerequisites and list a learning path in topological order with links to the unit pages.
// Search box over day titles, unit titles and topics. Ctrl + wheel or the buttons zoom; drag pans.
// Data: days.json, generated from days.yml by tools/days_json.py (make sidebar).
// Options: data-base="" (prefix of the unit links, relative to the page).
import { autoMount, htmlEl, svgEl, button, checkbox } from "../lib/ui";
import raw from "./days.json";
import { CourseData, buildCourse, learningPath, dayEdges, dependents, search, resolve, Course } from "./graph";

const NAME = "coursegraph";
const P = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${P} { display: block; font-size: 15px; }
${P} .cg-bar { display: flex; flex-wrap: wrap; gap: .5em .8em; align-items: center; margin-bottom: .5em; }
${P} .cg-search { position: relative; flex: 1 1 18em; max-width: 30em; }
${P} .cg-search input { width: 100%; font: inherit; padding: .35em .6em; border: 1px solid #aaa; border-radius: 6px; box-sizing: border-box; }
${P} .cg-results { position: absolute; z-index: 20; left: 0; right: 0; top: 100%; margin: 2px 0 0; padding: .2em 0;
  list-style: none; background: #fff; border: 1px solid #bbb; border-radius: 6px; box-shadow: 0 4px 14px rgba(0,0,0,.15);
  max-height: 22em; overflow-y: auto; }
${P} .cg-results:empty { display: none; }
${P} .cg-results li { padding: .3em .7em; cursor: pointer; line-height: 1.3; }
${P} .cg-results li:hover, ${P} .cg-results li.cg-active { background: #eef3fb; }
${P} .cg-results .cg-where { font-size: .82em; color: #666; display: block; }
${P} .cg-results mark { background: #ffe9a8; padding: 0; }
${P} .cg-legend { display: flex; gap: 1em; align-items: center; font-size: .85em; color: #555; flex-wrap: wrap; }
${P} .cg-legend svg { display: inline-block; vertical-align: middle; }
${P} .cg-zoom { display: flex; gap: .3em; }
${P} .cg-zoom .qb-btn { min-width: 2.2em; }
${P} .cg-view { position: relative; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; background: #fff; }
${P} .cg-view svg { width: 100%; height: auto; cursor: grab; }
${P} .cg-view svg.cg-panning { cursor: grabbing; }
${P} .cg-hint { position: absolute; right: .6em; bottom: .3em; font-size: .78em; color: #888; pointer-events: none; }
${P} .cg-day { cursor: pointer; }
${P} .cg-day rect.cg-box { stroke-width: 1.5; }
${P} .cg-day text { font-size: 12.5px; pointer-events: none; }
${P} .cg-day text.cg-num { font-weight: 700; }
${P} .cg-day:hover rect.cg-box { stroke-width: 3; }
${P} .cg-day.cg-dim { opacity: .3; }
${P} .cg-day.cg-hit rect.cg-box { stroke: #e8590c !important; stroke-width: 4; }
${P} .cg-day.cg-sel rect.cg-box { stroke: #111 !important; stroke-width: 3.5; }
${P} .cg-day.cg-planned rect.cg-box { stroke-dasharray: 5 3; }
${P} .cg-pip { cursor: pointer; }
${P} .cg-partlabel { font-size: 13px; font-weight: 700; }
${P} .cg-panel { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.2fr); gap: 1.2em; margin-top: .8em; }
@media (max-width: 760px) { ${P} .cg-panel { grid-template-columns: 1fr; } }
${P} .cg-card { background: #f6f7f9; border-radius: 8px; padding: .7em 1em; line-height: 1.4; }
${P} .cg-card h4 { margin: 0 0 .4em; font-size: 1.05em; }
${P} .cg-card h5 { margin: .7em 0 .25em; font-size: .9em; text-transform: uppercase; letter-spacing: .03em; color: #555; }
${P} .cg-card ol, ${P} .cg-card ul { margin: .2em 0; padding-left: 1.5em; }
${P} .cg-card li { margin: .15em 0; }
${P} .cg-link { background: none; border: none; padding: 0; font: inherit; color: #1f5fa8; cursor: pointer; text-decoration: underline dotted; text-align: left; }
${P} .cg-link:hover { color: #0b3d75; }
${P} .cg-small { font-size: .85em; color: #555; }
${P} .cg-chip { display: inline-block; font-size: .78em; font-weight: 700; color: #fff; border-radius: 4px; padding: 0 .4em; margin-right: .35em; vertical-align: 1px; }
${P} .cg-then { font-weight: 600; }
@media print { ${P} { display: none; } }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

/** Split a title into at most `maxLines` lines of about `width` characters. */
function wrap(text: string, width: number, maxLines: number): string[] {
  const words = text.split(/\s+/), lines: string[] = [];
  let cur = "";
  for (const w of words) {
    if (cur && (cur + " " + w).length > width) { lines.push(cur); cur = w; } else cur = cur ? cur + " " + w : w;
  }
  if (cur) lines.push(cur);
  if (lines.length > maxLines) {
    const keep = lines.slice(0, maxLines);
    keep[maxLines - 1] = keep[maxLines - 1].replace(/\s*\S*$/, "") + "…";
    return keep;
  }
  return lines;
}

interface Box { x: number; y: number; w: number; h: number; col: number; row: number }
interface Layout { W: number; H: number; boxes: Box[]; columns: boolean; labels: { x: number; y: number; w: number; part: number; lines: string[] }[] }

const BOX_H = 52, ROW = 66;

function layout(c: Course, W: number): Layout {
  const parts = c.data.parts, days = c.data.days;
  const boxes: Box[] = new Array(days.length);
  const labels: Layout["labels"] = [];
  const columns = W / parts.length >= 170;   // columns only when a box gets ~145 px
  if (columns) {
    const colW = W / parts.length, gap = 26, top = 50;
    let H = 0;
    parts.forEach((p, pi) => {
      labels.push({ x: pi * colW + gap / 2, y: 6, w: colW - gap, part: pi, lines: wrap(`${p.id}. ${p.title}`, Math.floor((colW - gap) / 7.4), 2) });
      days.map((d, di) => [d, di] as const).filter(([d]) => d.part === pi).forEach(([, di], r) => {
        boxes[di] = { x: pi * colW + gap / 2, y: top + r * ROW, w: colW - gap, h: BOX_H, col: pi, row: r };
        H = Math.max(H, top + r * ROW + BOX_H + 14);
      });
    });
    return { W, H, boxes, columns, labels };
  }
  const k = Math.max(2, Math.min(9, Math.floor(W / 150)));
  const pitch = W / k, gap = 22;
  let y = 4, row = 0;
  parts.forEach((p, pi) => {
    labels.push({ x: 6, y, w: W - 12, part: pi, lines: [`Part ${p.id}: ${p.title}`] });
    y += 26;
    const ds = days.map((d, di) => [d, di] as const).filter(([d]) => d.part === pi);
    ds.forEach(([, di], j) => {
      const r = Math.floor(j / k), cc = j % k;
      boxes[di] = { x: cc * pitch + gap / 2, y: y + r * ROW, w: pitch - gap, h: BOX_H, col: cc, row: row + r };
    });
    const rows = Math.ceil(ds.length / k);
    y += rows * ROW + 8;
    row += rows;
  });
  return { W, H: y, boxes, columns: false, labels };
}

/** Curve from prerequisite box a to dependent box b (ends at b's border, where the arrow goes). */
function edgePath(L: Layout, a: Box, b: Box): string {
  const f = (v: number) => v.toFixed(1);
  if (L.columns) {
    if (b.col !== a.col) {
      const fwd = b.col > a.col;
      const x0 = fwd ? a.x + a.w : a.x, x1 = fwd ? b.x : b.x + b.w;
      const y0 = a.y + a.h / 2, y1 = b.y + b.h / 2, dx = (x1 - x0) * 0.5;
      return `M${f(x0)},${f(y0)} C${f(x0 + dx)},${f(y0)} ${f(x1 - dx)},${f(y1)} ${f(x1)},${f(y1)}`;
    }
    const x = a.x + a.w, y0 = a.y + a.h / 2 + (b.y > a.y ? 8 : -8), y1 = b.y + b.h / 2 + (b.y > a.y ? -8 : 8);
    const bulge = Math.min(12, 5 + 2 * Math.abs(b.row - a.row));
    return `M${f(x)},${f(y0)} C${f(x + bulge)},${f(y0)} ${f(x + bulge)},${f(y1)} ${f(x)},${f(y1)}`;
  }
  const ax = a.x + a.w / 2, bx = b.x + b.w / 2;
  if (b.row > a.row) {
    const y0 = a.y + a.h, y1 = b.y, dy = (y1 - y0) * 0.5;
    return `M${f(ax)},${f(y0)} C${f(ax)},${f(y0 + dy)} ${f(bx)},${f(y1 - dy)} ${f(bx)},${f(y1)}`;
  }
  if (b.row < a.row) {
    const y0 = a.y, y1 = b.y + b.h, dy = (y1 - y0) * 0.5;
    return `M${f(ax)},${f(y0)} C${f(ax)},${f(y0 + dy)} ${f(bx)},${f(y1 - dy)} ${f(bx)},${f(y1)}`;
  }
  const lift = Math.min(12, 4 + Math.abs(bx - ax) / 60);
  return `M${f(ax)},${f(a.y)} C${f(ax)},${f(a.y - lift)} ${f(bx)},${f(b.y - lift)} ${f(bx)},${f(b.y)}`;
}

let instances = 0;

function mount(host: HTMLElement): void {
  injectStyle();
  const c = buildCourse(raw as CourseData);
  const days = c.data.days, parts = c.data.parts;
  const base = host.dataset.base ?? "";
  const uid = `cg${++instances}`;
  const edges = dayEdges(c);
  const unitLink = (u: number, page = "lec") => `${base}${c.units[u].folder}/${page}.html`;

  // ---- toolbar
  const bar = htmlEl("div", "cg-bar", "", host);
  const sbox = htmlEl("div", "cg-search", "", bar);
  const input = htmlEl("input", "", "", sbox);
  Object.assign(input, { type: "search", placeholder: "Search a topic, e.g. Fourier, watershed, Hessian…" });
  input.setAttribute("aria-label", "Search days, units and topics");
  const results = htmlEl("ul", "cg-results", "", sbox);
  results.setAttribute("role", "listbox");
  const zoomBox = htmlEl("div", "cg-zoom", "", bar);
  button(zoomBox, "+", () => zoomBy(1.4)).title = "Zoom in";
  button(zoomBox, "−", () => zoomBy(1 / 1.4)).title = "Zoom out";
  button(zoomBox, "Fit", () => { view = { x: 0, y: 0, s: 1 }; applyView(); }).title = "Show the whole course";
  let showSoft = false;
  checkbox(bar, "show “see also” links", false, v => { showSoft = v; drawEdges(); });
  button(bar, "Clear", () => select(null));
  const legend = htmlEl("div", "cg-legend", "", host);
  legend.innerHTML =
    `<span><svg width="34" height="10"><line x1="1" y1="5" x2="33" y2="5" stroke="#333" stroke-width="2"/></svg> needs (hard prerequisite)</span>` +
    `<span><svg width="34" height="10"><line x1="1" y1="5" x2="33" y2="5" stroke="#333" stroke-width="2" stroke-dasharray="5 3"/></svg> see also (soft)</span>` +
    `<span><svg width="18" height="12"><rect x="1" y="1" width="16" height="10" rx="3" fill="#2a7f8f"/></svg> needed for the selection</span>` +
    `<span>small bars in a box = its units</span>`;
  legend.style.marginBottom = ".4em";

  // ---- graph
  const viewDiv = htmlEl("div", "cg-view", "", host);
  const svg = svgEl("svg", { role: "img" }, viewDiv);
  svg.setAttribute("aria-label", "Dependency graph of all days of the course; the list below the graph has the same content.");
  htmlEl("div", "cg-hint", "click a day or a unit bar · Ctrl + wheel zooms · drag pans", viewDiv);
  const defs = svgEl("defs", {}, svg);
  for (const [id, col] of [["base", "#9a9a9a"], ["hi", "#222"], ["soft", "#e8590c"]] as const) {
    const m = svgEl("marker", { id: `${uid}-${id}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
    svgEl("path", { d: "M0,1 L9,5 L0,9 z", fill: col }, m);
  }
  const gBands = svgEl("g", {}, svg);
  const gEdges = svgEl("g", { fill: "none" }, svg);
  const gDays = svgEl("g", {}, svg);
  const gHi = svgEl("g", { fill: "none", "pointer-events": "none" }, svg);

  // ---- panel
  const panel = htmlEl("div", "cg-panel", "", host);
  const left = htmlEl("div", "cg-card", "", panel);
  const right = htmlEl("div", "cg-card", "", panel);
  right.setAttribute("aria-live", "polite");

  let L: Layout = layout(c, 1000);
  let view = { x: 0, y: 0, s: 1 };
  let sel: { day: number; unit?: number } | null = null;
  let hits = new Set<number>();
  const dayG: SVGGElement[] = [];
  const pips: SVGRectElement[][] = [];

  function applyView(): void {
    const w = L.W / view.s, h = L.H / view.s;
    view.x = Math.max(0, Math.min(L.W - w, view.x));
    view.y = Math.max(0, Math.min(L.H - h, view.y));
    svg.setAttribute("viewBox", `${view.x} ${view.y} ${w} ${h}`);
    viewDiv.style.touchAction = view.s > 1 ? "none" : "pan-y";   // unzoomed: a touch scrolls the page
  }
  function zoomBy(f: number, cx = L.W / 2 / view.s + view.x, cy = L.H / 2 / view.s + view.y): void {
    const s = Math.max(1, Math.min(6, view.s * f));
    view.x = cx - (cx - view.x) * view.s / s;
    view.y = cy - (cy - view.y) * view.s / s;
    view.s = s;
    applyView();
  }

  function draw(): void {
    gBands.replaceChildren(); gDays.replaceChildren();
    dayG.length = 0; pips.length = 0;
    svg.setAttribute("width", String(L.W)); svg.setAttribute("height", String(L.H));
    // part labels / bands
    for (const lab of L.labels) {
      const p = parts[lab.part];
      if (L.columns) {
        svgEl("rect", { x: lab.x - 8, y: 2, width: lab.w + 16, height: L.H - 4, rx: 8, fill: p.tint, opacity: 0.55 }, gBands);
        lab.lines.forEach((ln, i) => {
          const t = svgEl("text", { x: lab.x + lab.w / 2, y: lab.y + 13 + i * 15, "text-anchor": "middle", class: "cg-partlabel", fill: p.colour }, gBands);
          t.textContent = ln;
        });
      } else {
        svgEl("rect", { x: 0, y: lab.y, width: L.W, height: 22, rx: 4, fill: p.tint }, gBands);
        svgEl("rect", { x: 0, y: lab.y, width: 5, height: 22, fill: p.colour }, gBands);
        const t = svgEl("text", { x: lab.x + 8, y: lab.y + 16, class: "cg-partlabel", fill: p.colour }, gBands);
        t.textContent = lab.lines[0];
      }
    }
    days.forEach((d, di) => {
      const b = L.boxes[di], p = parts[d.part];
      const g = svgEl("g", { class: "cg-day" + (d.planned ? " cg-planned" : ""), tabindex: -1 }, gDays);
      svgEl("title", {}, g).textContent = `Day ${d.number}: ${d.title}${d.planned ? " (in preparation)" : ""}\n` +
        d.units.map((u, k) => `${k + 1}. ${u.title}`).join("\n");
      svgEl("rect", { class: "cg-box", x: b.x, y: b.y, width: b.w, height: b.h, rx: 6, fill: p.tint, stroke: p.colour }, g);
      const chars = Math.floor((b.w - 34) / 6.6);
      const lines = wrap(d.title, chars, 2);
      const num = svgEl("text", { class: "cg-num", x: b.x + 7, y: b.y + 17, fill: p.colour }, g);
      num.textContent = String(d.number);
      lines.forEach((ln, i) => {
        const t = svgEl("text", { class: "cg-title", x: b.x + 28, y: b.y + 17 + i * 14.5, fill: "#222" }, g);
        t.textContent = ln;
      });
      const n = d.units.length, pw = Math.min(16, (b.w - 14 - (n - 1) * 3) / n);
      pips.push(c.dayUnits[di].map((u, k) => {
        const r = svgEl("rect", { class: "cg-pip", x: b.x + 7 + k * (pw + 3), y: b.y + b.h - 10, width: pw, height: 6, rx: 1.5, fill: p.colour, opacity: 0.45 }, g);
        svgEl("title", {}, r).textContent = `${c.units[u].title} (unit ${k + 1} of ${n})`;
        r.dataset.unit = String(u);
        return r;
      }));
      g.dataset.day = String(di);
      dayG.push(g);
    });
    drawEdges();
    paint();
    applyView();
  }

  function drawEdges(): void {
    gEdges.replaceChildren(); gHi.replaceChildren();
    for (const e of edges) {
      if (!e.hard && !showSoft) continue;
      svgEl("path", {
        d: edgePath(L, L.boxes[e.from], L.boxes[e.to]), stroke: e.hard ? "#9a9a9a" : "#c4c4c4",
        "stroke-width": e.hard ? 1.2 : 1, "stroke-dasharray": e.hard ? "" : "4 3",
        opacity: sel ? 0.25 : 0.7, "marker-end": `url(#${uid}-base)`,
      }, gEdges);
    }
    if (!sel) return;
    // hard edges actually used by the selection's prerequisites (unit level, between days)
    const lp = learningPath(c, sel);
    const nodes = new Set([...lp.need, ...lp.targets]);
    const used = new Set<string>();
    for (const u of nodes) for (const p of c.prereqs[u]) {
      const a = c.units[p].day, b = c.units[u].day;
      if (a !== b && nodes.has(p)) used.add(a + ">" + b);
    }
    for (const key of used) {
      const [a, b] = key.split(">").map(Number);
      svgEl("path", { d: edgePath(L, L.boxes[a], L.boxes[b]), stroke: "#222", "stroke-width": 2, opacity: 0.8, "marker-end": `url(#${uid}-hi)` }, gHi);
    }
    // the selection's own "see also" links
    const softs = new Set<number>();
    const us = sel.unit !== undefined ? [sel.unit] : c.dayUnits[sel.day];
    for (const u of us) for (const s of c.soft[u]) softs.add(c.units[s].day);
    softs.delete(sel.day);
    for (const a of softs)
      svgEl("path", { d: edgePath(L, L.boxes[a], L.boxes[sel.day]), stroke: "#e8590c", "stroke-width": 1.8, "stroke-dasharray": "5 3", opacity: 0.85, "marker-end": `url(#${uid}-soft)` }, gHi);
  }

  function paint(): void {
    const lp = sel ? learningPath(c, sel) : null;
    days.forEach((d, di) => {
      const g = dayG[di], p = parts[d.part];
      const isSel = !!sel && sel.day === di;
      const needed = lp ? c.dayUnits[di].some(u => lp.need.has(u)) : false;
      const filled = isSel || needed;
      g.classList.toggle("cg-sel", isSel);
      g.classList.toggle("cg-dim", !!sel && !filled);
      g.classList.toggle("cg-hit", hits.has(di));
      const box = g.querySelector("rect.cg-box")!;
      box.setAttribute("fill", filled ? p.colour : p.tint);
      g.querySelectorAll("text").forEach(t => t.setAttribute("fill", filled ? "#fff" : t.classList.contains("cg-num") ? p.colour : "#222"));
      pips[di].forEach(r => {
        const u = Number(r.dataset.unit);
        const target = !!lp && lp.targets.includes(u);
        const need = !!lp && lp.need.has(u);
        r.setAttribute("fill", target ? "#ffb347" : filled ? "#fff" : p.colour);
        r.setAttribute("opacity", target || need ? "1" : filled ? "0.35" : "0.45");
      });
    });
  }

  // ---- panel text
  const dayLabel = (di: number) => `Day ${days[di].number}: ${days[di].title}`;
  function chip(parent: HTMLElement, di: number): void {
    const p = parts[days[di].part];
    const s = htmlEl("span", "cg-chip", p.id, parent);
    s.style.background = p.colour;
    s.title = `Part ${p.id}: ${p.title}`;
  }
  function dayButton(parent: HTMLElement, di: number, text = dayLabel(di)): HTMLButtonElement {
    const b = htmlEl("button", "cg-link", text, parent);
    b.type = "button";
    b.addEventListener("click", () => { select({ day: di }); b.blur(); });
    return b;
  }
  function unitLinks(parent: HTMLElement, u: number, withPages = true): void {
    const un = c.units[u];
    if (un.planned) { parent.append(`${un.title} (in preparation)`); return; }
    const a = htmlEl("a", "", un.title, parent);
    a.href = unitLink(u);
    if (withPages) {
      const s = htmlEl("span", "cg-small", "", parent);
      s.append(" (");
      const q = htmlEl("a", "", "quiz", s); q.href = unitLink(u, "quiz");
      s.append(", ");
      const x = htmlEl("a", "", "exercise", s); x.href = unitLink(u, "exercise");
      s.append(")");
    }
  }
  function refButton(parent: HTMLElement, ref: string): void {
    const u = resolve(c, ref);
    if (ref.endsWith("/")) {
      const b = htmlEl("button", "cg-link", c.units[u].title, parent);
      b.type = "button";
      b.addEventListener("click", () => { select({ day: c.units[u].day, unit: u }); b.blur(); });
      parent.append(` (day ${days[c.units[u].day].number})`);
    } else dayButton(parent, c.units[u].day);
  }
  function list(parent: HTMLElement, items: ((li: HTMLElement) => void)[], ordered = false): void {
    const l = htmlEl(ordered ? "ol" : "ul", "", "", parent);
    for (const f of items) f(htmlEl("li", "", "", l));
  }

  function renderPanel(): void {
    left.replaceChildren(); right.replaceChildren();
    if (!sel) {
      htmlEl("h4", "", "Pick a topic", left);
      htmlEl("p", "", "Click a day in the graph, or one of the small unit bars inside a box, or search for a topic above. " +
        "The graph then shows everything that topic builds on, and the list on the right gives an order to learn it in.", left);
      htmlEl("p", "cg-small", `${days.length} days in ${parts.length} parts, ${c.units.length} units. Each box is a day; ` +
        `an arrow goes from a day to a later day that needs it.`, left);
      htmlEl("h4", "", "Learning path", right);
      htmlEl("p", "", "Nothing selected yet. Following the course from the top always works: every day comes after what it needs.", right);
      return;
    }
    const di = sel.day, d = days[di];
    const h = htmlEl("h4", "", "", left);
    chip(h, di);
    h.append(dayLabel(di) + (d.planned ? " (in preparation)" : ""));
    const toList = htmlEl("a", "cg-small", "in the list below", left);
    toList.href = `#day-${d.id}`;
    htmlEl("h5", "", "Units of this day", left);
    list(left, c.dayUnits[di].map(u => (li: HTMLElement) => {
      if (sel!.unit === u) li.style.fontWeight = "700";
      const b = htmlEl("button", "cg-link", "path", li);
      b.type = "button"; b.title = "Show the learning path to this unit";
      b.addEventListener("click", () => { select({ day: di, unit: u }); b.blur(); });
      li.append(" · ");
      unitLinks(li, u);
    }), true);
    if (d.owns.length) {
      htmlEl("h5", "", "Topics introduced here", left);
      list(left, d.owns.map(o => (li: HTMLElement) => { li.textContent = o; }));
    }
    const unitRefs = sel.unit !== undefined ? c.units[sel.unit].hard : d.units.flatMap(u => u.hard);
    if (d.hard.length || unitRefs.length) {
      htmlEl("h5", "", "Needs directly", left);
      list(left, [...d.hard, ...unitRefs].map(r => (li: HTMLElement) => refButton(li, r)));
    }
    const soft = [...d.soft, ...(sel.unit !== undefined ? c.units[sel.unit].soft : d.units.flatMap(u => u.soft))];
    if (soft.length) {
      htmlEl("h5", "", "See also", left);
      list(left, [...new Set(soft)].map(r => (li: HTMLElement) => refButton(li, r)));
    }
    const later = dependents(c, di);
    if (later.length) {
      htmlEl("h5", "", "Needed later by", left);
      const p = htmlEl("p", "", "", left);
      later.forEach((x, i) => { if (i) p.append(", "); dayButton(p, x); });
    }

    // learning path
    const lp = learningPath(c, sel);
    const what = sel.unit !== undefined ? `“${c.units[sel.unit].title}”` : dayLabel(di);
    htmlEl("h4", "", "Learning path to " + what, right);
    if (!lp.steps.length) {
      htmlEl("p", "", "Nothing earlier is needed: you can start here.", right);
    } else {
      const nd = new Set(lp.steps.map(s => s.day)).size;
      htmlEl("p", "cg-small", `Before it: ${lp.need.size} unit${lp.need.size === 1 ? "" : "s"} from ${nd} day${nd === 1 ? "" : "s"}, ` +
        `in an order that respects every prerequisite.`, right);
      list(right, lp.steps.map(s => (li: HTMLElement) => {
        chip(li, s.day);
        dayButton(li, s.day);
        const n = c.dayUnits[s.day].length;
        li.append(htmlEl("span", "cg-small", s.all ? ` · all ${n} units` : ` · unit${s.units.length > 1 ? "s" : ""} ${s.units.map(u => c.units[u].k + 1).join(", ")} of ${n}`));
        list(li, s.units.map(u => (l2: HTMLElement) => unitLinks(l2, u, false)), false);
      }), true);
    }
    const then = htmlEl("p", "cg-then", "Then: ", right);
    if (sel.unit !== undefined) unitLinks(then, sel.unit);
    else {
      then.append(dayLabel(di) + " — start with ");
      unitLinks(then, c.dayUnits[di][0], false);
    }
  }

  function select(s: { day: number; unit?: number } | null): void {
    sel = s;
    drawEdges(); paint(); renderPanel();
    try {
      const id = !s ? "" : s.unit !== undefined ? c.units[s.unit].folder + "/" : days[s.day].id;
      history.replaceState(null, "", id ? `#path:${id}` : location.pathname + location.search);
    } catch { /* file:// or sandbox */ }
  }

  // ---- search
  let active = 0, current: ReturnType<typeof search> = [];
  const esc = (s: string) => s.replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]!));
  function marked(text: string, words: string[]): string {
    let out = esc(text);
    for (const w of words) if (w) out = out.replace(new RegExp(esc(w).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi"), m => `<mark>${m}</mark>`);
    return out;
  }
  function choose(i: number): void {
    const hItem = current[i];
    if (!hItem) return;
    results.replaceChildren();
    select(hItem.unit !== undefined ? { day: hItem.day, unit: hItem.unit } : { day: hItem.day });
    input.blur();
  }
  function runSearch(): void {
    const q = input.value.trim();
    current = search(c, q, 40);
    hits = new Set(current.map(x => x.day));
    paint();
    results.replaceChildren();
    if (!q) return;
    const words = q.toLowerCase().split(/\s+/);
    if (!current.length) {
      const li = htmlEl("li", "", "No day, unit or topic matches.", results);
      li.style.cursor = "default";
      return;
    }
    active = 0;
    current.slice(0, 14).forEach((x, i) => {
      const li = htmlEl("li", i === 0 ? "cg-active" : "", "", results);
      li.setAttribute("role", "option");
      li.innerHTML = marked(x.text, words) + `<span class="cg-where">${x.field === "topic" ? "topic of " : x.field === "unit" ? "unit of " : ""}${esc(dayLabel(x.day))}</span>`;
      li.addEventListener("pointerdown", ev => { ev.preventDefault(); choose(i); });
    });
  }
  input.addEventListener("input", runSearch);
  input.addEventListener("focus", runSearch);
  input.addEventListener("blur", () => { results.replaceChildren(); });
  input.addEventListener("keydown", ev => {
    const n = results.children.length;
    if (ev.key === "Enter") { ev.preventDefault(); choose(active); }
    else if ((ev.key === "ArrowDown" || ev.key === "ArrowUp") && n) {
      ev.preventDefault();
      results.children[active]?.classList.remove("cg-active");
      active = (active + (ev.key === "ArrowDown" ? 1 : n - 1)) % n;
      results.children[active].classList.add("cg-active");
      results.children[active].scrollIntoView({ block: "nearest" });
    } else if (ev.key === "Escape") { input.value = ""; runSearch(); input.blur(); }
  });

  // ---- pointer: click selects, drag pans, ctrl + wheel zooms
  const toSvg = (ev: { clientX: number; clientY: number }): [number, number] => {
    const r = svg.getBoundingClientRect();
    return [view.x + (ev.clientX - r.left) / r.width * L.W / view.s, view.y + (ev.clientY - r.top) / r.height * L.H / view.s];
  };
  let drag: { x: number; y: number; vx: number; vy: number; moved: boolean; id: number } | null = null;
  svg.addEventListener("pointerdown", ev => {
    drag = { x: ev.clientX, y: ev.clientY, vx: view.x, vy: view.y, moved: false, id: ev.pointerId };
  });
  svg.addEventListener("pointermove", ev => {
    if (!drag) return;
    const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) > 5) {
      drag.moved = true;
      svg.classList.add("cg-panning");
      try { svg.setPointerCapture(drag.id); } catch { /* ignore */ }
    }
    if (drag.moved) {
      const r = svg.getBoundingClientRect();
      view.x = drag.vx - dx / r.width * L.W / view.s;
      view.y = drag.vy - dy / r.height * L.H / view.s;
      applyView();
    }
  });
  const endDrag = (ev: PointerEvent) => {
    if (!drag) return;
    const wasDrag = drag.moved;
    drag = null;
    svg.classList.remove("cg-panning");
    if (wasDrag || ev.type === "pointercancel") return;
    const t = ev.target as Element;
    const pip = t.closest?.(".cg-pip") as SVGElement | null;
    const g = t.closest?.(".cg-day") as SVGElement | null;
    if (pip) { const u = Number(pip.dataset.unit); select({ day: c.units[u].day, unit: u }); }
    else if (g) select({ day: Number(g.dataset.day) });
  };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);
  svg.addEventListener("wheel", ev => {
    if (!ev.ctrlKey && !ev.metaKey) return;
    ev.preventDefault();
    const [x, y] = toSvg(ev);
    zoomBy(Math.exp(-ev.deltaY * 0.0025), x, y);
  }, { passive: false });

  // ---- size: lay out at the container's width, 1 svg unit = 1 css pixel
  let lastW = 0;
  const relayout = () => {
    const W = Math.round(viewDiv.clientWidth || host.clientWidth || 1000);
    if (Math.abs(W - lastW) < 8) return;
    lastW = W;
    L = layout(c, W);
    view = { x: 0, y: 0, s: 1 };
    draw();
  };
  relayout();
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(relayout).observe(viewDiv);

  // initial selection from the address (#path:<day id> or #path:<folder>/)
  const m = /^#path:(.+)$/.exec(decodeURIComponent(location.hash));
  let initial: { day: number; unit?: number } | null = null;
  if (m) {
    const ref = m[1];
    if (ref.endsWith("/") && c.unitIndex.has(ref.slice(0, -1))) { const u = c.unitIndex.get(ref.slice(0, -1))!; initial = { day: c.units[u].day, unit: u }; }
    else if (c.dayIndex.has(ref)) initial = { day: c.dayIndex.get(ref)! };
  }
  sel = initial;
  drawEdges(); paint(); renderPanel();
}

autoMount(NAME, mount);
