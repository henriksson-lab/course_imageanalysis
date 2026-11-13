// ray2d: a 2D ray / reflection / refraction sandbox (ray-tracing day, unit whitted/). A light
// source (one ray, a point source, a parallel beam or a converging beam; drag its handle and its
// aiming handle) and objects of glass, water or oil (flat interface, slab, ball, biconvex lens,
// prism, half-disc, a coverslip stack, a mirror; drag their centre handles; click to select, then
// set the refractive index and the rotation). Rays are traced with Snell's law; with Fresnel on,
// every boundary splits the ray into a reflected and a refracted part and the opacity shows the
// power each carries. Click a hit point: incidence and refraction angles, n1 sin θ1 = n2 sin θ2,
// the reflectance R. Dispersion: three colours with a Cauchy index. Depth cap 12, power cut-off 0.1 %.
// Options: data-preset (snell, critical, guide, grazing, ball, lens, prism, oil, water).
import { htmlEl, svgEl, slider, button, presets, checkbox, autoMount, svgPoint } from "../lib/ui";
import * as R2 from "../lib/rays2d";
import { State, Obj, SrcType, PRESETS, regionOf, sceneOf, sourceRays, axisCrossings } from "./presets";

const NAME = "ray2d";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${S} { gap: .7em; }
${S} .r2-c1 { flex: 1 1 700px; min-width: 300px; max-width: 760px; display: flex; flex-direction: column; gap: .3em; }
${S} .qb-panel { flex: 1 1 300px; max-width: 390px; gap: .35em; }
${S} svg.r2-svg { width: 100%; display: block; background: #fcfcfd; border: 1px solid var(--qb-grid); border-radius: 4px; touch-action: none; }
${S} .r2-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
${S} .qb-readout { font-family: system-ui, sans-serif; white-space: pre; line-height: 1.3; }
${S} .r2-sel { font-weight: 600; }
.reveal ${S} { flex-wrap: nowrap; font-size: 18px; gap: .5em; }
.reveal ${S} .r2-c1 { flex: 0 0 720px; }
.reveal ${S} .qb-panel { flex: 0 0 400px; gap: .2em; padding: .3em .45em; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .02em .35em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .4em; }
.reveal ${S} .qb-presets { gap: .2em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style"); s.id = "qb-style-" + NAME; s.textContent = STYLE; document.head.appendChild(s);
}

// world (mm): x ∈ [0, 40], y ∈ [0, 25], y up
const WX = 40, WY = 25, VW = 760, VH = VW * WY / WX;
const sx = (x: number) => x / WX * VW, sy = (y: number) => VH - y / WY * VH;

const WAVES: { nm: number; col: string }[] = [{ nm: 650, col: "#e3261b" }, { nm: 550, col: "#1aa33a" }, { nm: 450, col: "#2a54e8" }];
const INDEX_PRESETS: [string, number][] = [["air", 1], ["water", 1.33], ["oil", 1.518], ["glass", 1.52], ["diamond", 2.42]];

function mount(host: HTMLElement): void {
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  let st = p0.s();

  const c1 = htmlEl("div", "r2-c1", "", host);
  const svg = svgEl("svg", { viewBox: `0 0 ${VW} ${VH}`, class: "r2-svg" }, c1);
  const gObj = svgEl("g", {}, svg), gRay = svgEl("g", {}, svg), gHit = svgEl("g", {}, svg), gH = svgEl("g", {}, svg);
  const ro = htmlEl("div", "qb-readout", "", c1);

  const panel = htmlEl("div", "qb-panel", "", host);
  const pb = presets(panel, [...PRESETS.map(p => ({ label: p.label, apply: () => { st = p.s(); sync(); } })), { label: "reset", apply: () => { st = p0.s(); sync(); pb.forEach(b => b.classList.toggle("active", b === pb[PRESETS.indexOf(p0)])); } }]);
  const srow = htmlEl("div", "r2-row", "", panel);
  const srcBtns = (["ray", "point", "beam", "focus"] as SrcType[]).map(t => ({ t, b: button(srow, t === "ray" ? "one ray" : t === "point" ? "point" : t === "beam" ? "beam" : "converging", () => { st.src.type = t; if (t !== "ray" && st.src.count < 3) st.src.count = 9; sync(); }) }));
  const cntS = slider(panel, "number of rays", 1, 41, 1, st.src.count, v => { st.src.count = v; draw(); });
  const crow = htmlEl("div", "r2-row", "", panel);
  const frC = checkbox(crow, "Fresnel split", st.fresnel, v => { st.fresnel = v; draw(); });
  const dsC = checkbox(crow, "dispersion", st.dispersion, v => { st.dispersion = v; draw(); });
  const selL = htmlEl("div", "r2-sel", "", panel);
  const nS = slider(panel, "refractive index", 1, 2.42, 0.001, 1.5, v => { const o = st.objs[st.sel]; if (o) { o.n = v; draw(); } }, v => v.toFixed(3));
  const irow = htmlEl("div", "r2-row", "", panel);
  for (const [lab, n] of INDEX_PRESETS) button(irow, `${lab} ${n}`, () => { const o = st.objs[st.sel]; if (o) { o.n = n; nS.set(n); draw(); } });
  const aS = slider(panel, "rotation (°)", -180, 180, 1, 0, v => { const o = st.objs[st.sel]; if (o) { o.angle = v; draw(); } });
  htmlEl("div", "qb-note", "Drag the source (S), its aim (◆) and the objects (+). Click a hit point (●).", panel);

  function sync(): void {
    cntS.set(st.src.count); frC.checked = st.fresnel; dsC.checked = st.dispersion;
    const o = st.objs[st.sel];
    if (o) { nS.set(o.n); aS.set(o.angle); }
    draw();
  }

  let hits: R2.HitInfo[] = [];
  function draw(): void {
    srcBtns.forEach(({ t, b }) => b.classList.toggle("active", t === st.src.type));
    cntS.input.disabled = st.src.type === "ray";
    gObj.replaceChildren(); gRay.replaceChildren(); gHit.replaceChildren(); gH.replaceChildren();
    const sc = sceneOf(st);
    // objects
    st.objs.forEach((o, k) => {
      const shade = Math.max(0, Math.min(1, (o.n - 1) / 1.4));
      const fill = `rgba(${Math.round(120 - 60 * shade)},${Math.round(170 - 50 * shade)},${Math.round(230 - 40 * shade)},${0.18 + 0.35 * shade})`;
      if (o.type === "mirror") {
        const a = o.angle * Math.PI / 180, c = Math.cos(a) * o.a / 2, s = Math.sin(a) * o.a / 2;
        svgEl("line", { x1: sx(o.x - c), y1: sy(o.y - s), x2: sx(o.x + c), y2: sy(o.y + s), stroke: "#555", "stroke-width": 6 }, gObj);
      } else {
        const r = regionOf(o)!;
        // rasterise the region outline by sampling (convex): polygon of boundary points
        const path = outline(r);
        const pe = svgEl("path", { d: path, fill, stroke: k === st.sel ? "#1f77b4" : "#6a8aa8", "stroke-width": k === st.sel ? 2.5 : 1.5 }, gObj);
        pe.addEventListener("pointerdown", () => { if (st.sel !== k) { st.sel = k; sync(); } });
      }
      const lab = svgEl("text", { x: sx(Math.max(3, Math.min(WX - 3, o.x))), y: sy(o.y) - 14, "text-anchor": "middle", "font-size": 21 }, gObj);
      lab.textContent = o.type === "mirror" ? "mirror" : `n = ${o.n.toFixed(2)}`; lab.classList.add("qb-halo");
    });
    if (st.axis) svgEl("line", { x1: 0, x2: VW, y1: sy(st.src.y), y2: sy(st.src.y), stroke: "#999", "stroke-dasharray": "6 5", "stroke-width": 1 }, gObj);
    if (st.src.type === "focus") { const F = sx(st.src.x + st.src.focus); svgEl("line", { x1: F, x2: F, y1: sy(st.src.y) - 14, y2: sy(st.src.y) + 14, stroke: "#000", "stroke-width": 2 }, gObj); const t = svgEl("text", { x: F, y: sy(st.src.y) + 36, "text-anchor": "middle", "font-size": 21 }, gObj); t.textContent = "nominal focus"; t.classList.add("qb-halo"); }
    // rays
    hits = [];
    const waves = st.dispersion ? WAVES : [{ nm: 550, col: "#e08a00" }];
    const finals: R2.Seg[] = [];
    for (const r of sourceRays(st)) for (const w of waves) {
      const res = R2.trace2d(sc, r.o, r.d, 1, { fresnel: st.fresnel, maxDepth: 12, minPower: 1e-3, nm: w.nm, maxLen: 200 });
      for (const sgm of res.segs) {
        svgEl("line", { x1: sx(sgm.x0), y1: sy(sgm.y0), x2: sx(sgm.x1), y2: sy(sgm.y1), stroke: w.col, "stroke-width": 2, "stroke-opacity": Math.min(1, 0.06 + 0.94 * sgm.power), "stroke-linecap": "round" }, gRay);
        finals.push(sgm);
      }
      hits.push(...res.hits);
    }
    hits.forEach((h, i) => {
      if (h.power < 0.02 || i > 400) return;
      const c = svgEl("circle", { cx: sx(h.x), cy: sy(h.y), r: i === st.hit ? 7 : 3.5, fill: i === st.hit ? "#ffcc00" : "#444", "fill-opacity": i === st.hit ? 1 : 0.6, stroke: "#000", "stroke-width": i === st.hit ? 2 : 0 }, gHit);
      c.style.cursor = "pointer";
      c.addEventListener("pointerdown", ev => { ev.stopPropagation(); st.hit = i; draw(); });
    });
    // handles
    const s = st.src, a = s.angle * Math.PI / 180, aimR = 3.2;
    const sh = handle(s.x, s.y, "S", 13, "#fff");
    drag(sh, (x, y) => { s.x = x; s.y = y; });
    const ah = svgEl("rect", { x: sx(s.x + aimR * Math.cos(a)) - 8, y: sy(s.y + aimR * Math.sin(a)) - 8, width: 16, height: 16, transform: `rotate(45 ${sx(s.x + aimR * Math.cos(a))} ${sy(s.y + aimR * Math.sin(a))})`, fill: "#ffcc00", stroke: "#000", "stroke-width": 1.5 }, gH);
    drag(ah, (x, y) => { s.angle = Math.round(Math.atan2(y - s.y, x - s.x) * 180 / Math.PI); });
    st.objs.forEach((o, k) => {
      if (o.type === "layer") return;
      const h = handle(o.x, o.y, "+", 11, k === st.sel ? "#cfe3f7" : "#fff");
      drag(h, (x, y) => { o.x = x; o.y = y; }, () => { st.sel = k; sync(); });
    });
    // readouts
    const o = st.objs[st.sel];
    selL.textContent = o ? `selected: ${o.name}` : "";
    const L: string[] = [];
    const h = hits[st.hit];
    if (h) {
      if (h.mirror) L.push(`mirror: θ1 = ${h.theta1.toFixed(1)}° = θ reflected; R = 1`);
      else {
        L.push(`n1 = ${h.n1.toFixed(3)} → n2 = ${h.n2.toFixed(3)}${st.dispersion ? ` (${h.nm} nm)` : ""}, power in ${(100 * h.power).toFixed(1)} %`);
        L.push(h.theta2 === null
          ? `θ1 = ${h.theta1.toFixed(1)}° > critical ${(Math.asin(h.n2 / h.n1) * 180 / Math.PI).toFixed(1)}°: total internal reflection`
          : `θ1 = ${h.theta1.toFixed(1)}°, θ2 = ${h.theta2.toFixed(1)}°;  n1 sin θ1 = ${(h.n1 * Math.sin(h.theta1 * Math.PI / 180)).toFixed(3)} = n2 sin θ2 = ${(h.n2 * Math.sin(h.theta2 * Math.PI / 180)).toFixed(3)}`);
        L.push(`reflected R = ${(100 * h.R).toFixed(1)} %, transmitted ${(100 * (1 - h.R)).toFixed(1)} %`);
      }
    } else L.push("Click a hit point (●) to read its angles.");
    if (st.axis) {
      // where do the outgoing rays cross the axis y = source y?
      const xs = axisCrossings(finals, s.y);
      if (xs.length) L.push(`outgoing rays cross the axis at x = ${Math.min(...xs).toFixed(2)} … ${Math.max(...xs).toFixed(2)} mm`);
    }
    ro.textContent = L.join("\n");
  }

  function outline(r: R2.Region): string {
    // boundary of a convex region: march around its centroid and find the boundary along each direction
    const pts: [number, number][] = [];
    let cx = 0, cy = 0, n = 0;
    for (let x = -10; x <= 50; x += 0.25) for (let y = -10; y <= 35; y += 0.25) if (R2.inside(r, x, y)) { cx += x; cy += y; n++; }
    if (!n) return "";
    cx /= n; cy /= n;
    for (let k = 0; k < 360; k++) {
      const a = k * Math.PI / 180, d: R2.P2 = [Math.cos(a), Math.sin(a)];
      const sc: R2.Scene2D = { regions: [r], mirrors: [], nAmbient: 1 };
      const b = R2.nearestBoundary(sc, [cx, cy], d, 0);
      const t = b ? Math.min(b.t, 300) : 300;
      pts.push([cx + t * d[0], cy + t * d[1]]);
    }
    return "M" + pts.map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join("L") + "Z";
  }

  function handle(x: number, y: number, txt: string, r: number, fill: string): SVGGElement {
    const g = svgEl("g", {}, gH);
    svgEl("circle", { cx: sx(x), cy: sy(y), r, fill, stroke: "#000", "stroke-width": 1.5 }, g);
    const t = svgEl("text", { x: sx(x), y: sy(y) + 7, "text-anchor": "middle", "font-size": 20, "font-weight": 700 }, g); t.textContent = txt;
    t.style.pointerEvents = "none";
    return g;
  }
  function drag(el: SVGElement, onMove: (x: number, y: number) => void, onDown?: () => void): void {
    el.style.cursor = "grab"; el.style.touchAction = "none";
    el.addEventListener("pointerdown", ev => {
      ev.preventDefault(); ev.stopPropagation();
      onDown?.();
      const move = (e: PointerEvent) => {
        const [px, py] = svgPoint(svg, e);
        onMove(Math.round(Math.max(0, Math.min(WX, px / VW * WX)) * 10) / 10, Math.round(Math.max(0, Math.min(WY, (VH - py) / VH * WY)) * 10) / 10);
        st.hit = -1; draw();
      };
      const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
      window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
    });
  }

  pb[PRESETS.indexOf(p0)].classList.add("active");
  sync();
}

autoMount(NAME, host => { injectStyle(); mount(host); });
