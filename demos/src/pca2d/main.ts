// PCA of a cloud of two-pixel images. Each dot is an image (pixel 1 across, pixel 2 up); drag
// any dot to change the data. Shown: the mean (+), the 2-SD covariance ellipse, PC1 and PC2
// (arrows of length 2√λ), and a line along a unit direction u (drag the blue handle). Red
// segments join each image to its reconstruction from one score along u; the side plot shows
// the variance of the scores and the reconstruction error for every angle of u, and the bars
// the share of the variance on each PC (a two-bar scree plot). Untick "centre" to put the line
// through (0, 0) instead of the mean.
//
// <div data-demo="pca2d" data-data="pixels|four|round|unrelated|outlier" data-angle="0"
//      data-centre="1"></div>
import { autoMount, button, checkbox, htmlEl, presets, svgEl, scale, draggable, svgPoint, setAttrs } from "../lib/ui";
import { makeData, pca, unit, origin, score, decode, along, varAlong, wrapAngle, KINDS, type Kind, type V2, type Pca } from "./data";

const NAME = "pca2d";
const STYLE = `
[data-demo="${NAME}"] { gap: 1em; }
[data-demo="${NAME}"] .pc-plot { flex: 0 1 470px; min-width: 0; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 360px; max-width: 520px; gap: .45em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre; overflow-x: auto; }
[data-demo="${NAME}"] svg text { font-size: 18px; }
[data-demo="${NAME}"] .pc-sw { display: flex; gap: 14px; align-items: flex-end; }
[data-demo="${NAME}"] .pc-sw figure { margin: 0; text-align: center; font-size: 15px; color: var(--qb-muted); }
[data-demo="${NAME}"] .pc-checks { display: flex; flex-wrap: wrap; gap: .2em 1em; }
.reveal [data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal [data-demo="${NAME}"] .pc-plot { flex: 0 0 450px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 1 1 560px; max-width: 640px; gap: .25em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .3em .5em; }
.reveal [data-demo="${NAME}"] .qb-panel > svg { max-height: 130px; }
.reveal [data-demo="${NAME}"] .qb-note { font-size: 16px; }
.reveal [data-demo="${NAME}"] .pc-sw { display: none; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const S = 470, LO = -0.1, HI = 1.15;
const BLUE = "#1f77b4", RED = "#d62728", PC1C = "#e07b1a", PC2C = "#7b4fa0";
const grey = (v: number) => { const g = Math.round(255 * Math.max(0, Math.min(1, v))); return `rgb(${g},${g},${g})`; };
const deg = (r: number) => r * 180 / Math.PI;
const f3 = (v: number) => v.toFixed(3), f4 = (v: number) => v.toFixed(4);

function mount(host: HTMLElement): void {
  injectStyle();
  const kind0 = (KINDS as string[]).includes(host.dataset.data ?? "") ? host.dataset.data as Kind : "pixels";
  const ang0 = (Number(host.dataset.angle ?? 0) || 0) * Math.PI / 180;
  const centre0 = host.dataset.centre !== "0";
  let P: V2[] = [], pc: Pca = pca([[0, 0], [1, 1]]), th = ang0, sel = 0, centred = centre0, showPC = true;

  // ---- the point cloud ----
  const plotDiv = htmlEl("div", "pc-plot", "", host);
  const svg = svgEl("svg", { viewBox: `0 0 ${S} ${S}`, width: "100%", role: "img", "aria-label": "Two-pixel images as points, with their principal components" }, plotDiv);
  const sx = scale(LO, HI, 40, S - 8), sy = scale(LO, HI, S - 40, 8);
  for (const v of [0, 0.5, 1]) {
    svgEl("line", { x1: sx(v), x2: sx(v), y1: sy(LO), y2: sy(HI), class: v === 0 ? "axis" : "grid" }, svg);
    svgEl("line", { y1: sy(v), y2: sy(v), x1: sx(LO), x2: sx(HI), class: v === 0 ? "axis" : "grid" }, svg);
    svgEl("text", { x: sx(v), y: S - 16, "text-anchor": "middle" }, svg).textContent = String(v);
    svgEl("text", { x: 30, y: sy(v) + 6, "text-anchor": "end" }, svg).textContent = String(v);
  }
  svgEl("text", { x: S - 10, y: S - 46, "text-anchor": "end", class: "halo" }, svg).textContent = "pixel 1 →";
  svgEl("text", { x: 46, y: 26 }, svg).textContent = "↑ pixel 2";
  const ellipse = svgEl("ellipse", { fill: PC1C, "fill-opacity": 0.07, stroke: PC1C, "stroke-width": 1.5, "stroke-dasharray": "5 4" }, svg);
  const gErr = svgEl("g", {}, svg);
  const line = svgEl("line", { stroke: BLUE, "stroke-width": 3 }, svg);
  const gProj = svgEl("g", {}, svg);
  const mkArrow = (col: string, lab: string) => {
    const l = svgEl("line", { stroke: col, "stroke-width": 4, "stroke-linecap": "round" }, svg);
    const h = svgEl("polygon", { fill: col }, svg);
    const t = svgEl("text", { class: "halo", fill: col, "font-weight": "700" }, svg);
    t.textContent = lab;
    return { l, h, t };
  };
  const ar1 = mkArrow(PC1C, "PC1"), ar2 = mkArrow(PC2C, "PC2");
  const meanMark = svgEl("path", { stroke: "#000", "stroke-width": 3, fill: "none" }, svg);
  const gPts = svgEl("g", {}, svg);
  const selMark = svgEl("circle", { r: 10, fill: "none", stroke: "#000", "stroke-width": 2.5, "pointer-events": "none" }, svg);
  const handle = svgEl("circle", { r: 12, class: "handle", stroke: BLUE }, svg);
  const handleLab = svgEl("text", { class: "halo", fill: BLUE, "font-weight": "600" }, svg);
  handleLab.textContent = "u";

  // ---- side panel ----
  const panel = htmlEl("div", "qb-panel", "", host);
  const EW = 400, EH = 150;
  const esvg = svgEl("svg", { viewBox: `0 0 ${EW} ${EH}`, width: "100%", role: "img", "aria-label": "Variance and error against the angle of u" }, panel);
  const ex = scale(-90, 90, 30, EW - 14);
  let ey = scale(0, 1, EH - 30, 26);
  const vPath = svgEl("path", { fill: "none", stroke: BLUE, "stroke-width": 2.5 }, esvg);
  const rPath = svgEl("path", { fill: "none", stroke: RED, "stroke-width": 2.5 }, esvg);
  const vMark = svgEl("circle", { r: 6, fill: BLUE }, esvg);
  const rMark = svgEl("circle", { r: 6, fill: RED }, esvg);
  const pcTick = svgEl("line", { stroke: PC1C, "stroke-width": 2, "stroke-dasharray": "4 3" }, esvg);
  for (const a of [-90, -45, 0, 45, 90]) svgEl("text", { x: ex(a), y: EH - 6, "text-anchor": "middle", style: "font-size:17px" }, esvg).textContent = `${a}°`;
  const eTitle = svgEl("text", { x: 4, y: 18, style: "font-size:17px" }, esvg);
  eTitle.innerHTML = `<tspan fill="${BLUE}">variance along u</tspan> and <tspan fill="${RED}">error</tspan> vs angle of u`;

  const BW = 400, BH = 56;
  const bsvg = svgEl("svg", { viewBox: `0 0 ${BW} ${BH}`, width: "100%", role: "img", "aria-label": "Share of the variance on each principal component" }, panel);
  const bars = [0, 1].map(k => {
    const y = 4 + k * 26;
    svgEl("text", { x: 0, y: y + 17, style: "font-size:16px", fill: k ? PC2C : PC1C, "font-weight": "700" }, bsvg).textContent = `PC${k + 1}`;
    const r = svgEl("rect", { x: 44, y, height: 20, fill: k ? PC2C : PC1C, "fill-opacity": 0.8 }, bsvg);
    const t = svgEl("text", { y: y + 16, style: "font-size:16px" }, bsvg);
    return { r, t };
  });

  const sw = htmlEl("div", "pc-sw", "", panel);
  const swatch = (cap: string) => {
    const f = htmlEl("figure", "", "", sw);
    const s2 = svgEl("svg", { viewBox: "0 0 84 42", width: 84, height: 42 }, f);
    const r1 = svgEl("rect", { x: 1, y: 1, width: 40, height: 40, stroke: "#999" }, s2);
    const r2 = svgEl("rect", { x: 43, y: 1, width: 40, height: 40, stroke: "#999" }, s2);
    htmlEl("figcaption", "", cap, f);
    return (p: V2) => { r1.setAttribute("fill", grey(p[0])); r2.setAttribute("fill", grey(p[1])); };
  };
  const swX = swatch("image x"), swR = swatch("rebuilt from 1 score");
  const out = htmlEl("div", "qb-readout", "", panel);
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "u = PC1", () => { th = pc.theta; draw(); });
  button(row, "u = PC2", () => { th = wrapAngle(pc.theta + Math.PI / 2); draw(); });
  const checks = htmlEl("div", "pc-checks", "", panel);
  const cbC = checkbox(checks, "centre (line through the mean)", centred, v => { centred = v; draw(); });
  const cbP = checkbox(checks, "show PCs and ellipse", true, v => { showPC = v; draw(); });
  htmlEl("div", "qb-note", "Drag u (blue handle) or any dot. Click a dot to select it. Red: what is lost by keeping one score.", panel);
  presets(panel, [
    { label: "neighbouring pixels", apply: () => setData("pixels", 0, true) },
    { label: "four images", apply: () => setData("four", 0, true) },
    { label: "round cloud", apply: () => setData("round", 0, true) },
    { label: "unrelated pixels", apply: () => setData("unrelated", 45 * Math.PI / 180, true) },
    { label: "one outlier", apply: () => { setData("outlier", 0, true); th = pc.theta; draw(); } },
    { label: "no centring", apply: () => { setData("unrelated", 0, false); th = pc.theta; draw(); } },
    { label: "Reset", apply: () => setData(kind0, ang0, centre0) },
  ]);

  const toWorld = (ev: { clientX: number; clientY: number }): [number, number] => { const [x, y] = svgPoint(svg, ev); return [sx.invert(x), sy.invert(y)]; };
  const clamp = (v: number) => Math.max(LO, Math.min(HI, v));

  function setData(k: Kind, angle: number, c: boolean): void {
    P = makeData(k).map(p => [p[0], p[1]] as V2);
    th = angle; centred = c; cbC.checked = c; showPC = true; cbP.checked = true;
    sel = k === "outlier" ? P.length - 1 : 0;
    gPts.replaceChildren();
    const r = P.length > 60 ? 5 : 7;
    P.forEach((p, i) => {
      const dot = svgEl("circle", { r, fill: "#555", "fill-opacity": 0.75 }, gPts);
      dot.addEventListener("pointerdown", () => { sel = i; draw(); });
      draggable(dot, toWorld, (x, y) => { P[i] = [clamp(x), clamp(y)]; sel = i; draw(); });
    });
    draw();
  }

  draggable(handle, toWorld, (x, y) => {
    const o = origin(pc, centred);
    if (Math.hypot(x - o[0], y - o[1]) < 1e-6) return;
    th = wrapAngle(Math.atan2(y - o[1], x - o[0])); draw();
  });

  function arrow(a: ReturnType<typeof mkArrow>, m: V2, d: V2, len: number): void {
    const tip: V2 = [m[0] + len * d[0], m[1] + len * d[1]];
    const X = sx(tip[0]), Y = sy(tip[1]);
    const ang = Math.atan2(-(d[1]), d[0]);            // screen angle (y down)
    const back = 14, wing = 7, cx = Math.cos(ang), cy = Math.sin(ang);
    setAttrs(a.l, { x1: sx(m[0]).toFixed(1), y1: sy(m[1]).toFixed(1), x2: (X - back * cx).toFixed(1), y2: (Y - back * cy).toFixed(1) });
    a.h.setAttribute("points", [[X, Y], [X - back * cx - wing * cy, Y - back * cy + wing * cx], [X - back * cx + wing * cy, Y - back * cy - wing * cx]].map(q => q.map(v => v.toFixed(1)).join(",")).join(" "));
    setAttrs(a.t, { x: (X + 8).toFixed(1), y: (Y - 8).toFixed(1) });
    for (const e of [a.l, a.h, a.t]) (e as SVGElement).style.display = showPC ? "" : "none";
  }

  function draw(): void {
    pc = pca(P);
    const o = origin(pc, centred), u = unit(th);
    // dots
    const dots = gPts.children;
    P.forEach((p, i) => setAttrs(dots[i], { cx: sx(p[0]).toFixed(1), cy: sy(p[1]).toFixed(1) }));
    // mean, ellipse, PCs
    const mx = sx(pc.m[0]), my = sy(pc.m[1]);
    meanMark.setAttribute("d", `M${mx - 9},${my}H${mx + 9}M${mx},${my - 9}V${my + 9}`);
    const k = (HI - LO) / (S - 48);                   // world units per screen pixel (same in x and y)
    setAttrs(ellipse, {
      cx: mx.toFixed(1), cy: my.toFixed(1),
      rx: (2 * Math.sqrt(pc.l1) / k).toFixed(1), ry: (2 * Math.sqrt(Math.max(0, pc.l2)) / k).toFixed(1),
      transform: `rotate(${(-deg(pc.theta)).toFixed(2)} ${mx.toFixed(1)} ${my.toFixed(1)})`,
    });
    ellipse.style.display = showPC ? "" : "none";
    arrow(ar1, pc.m, unit(pc.theta), Math.max(0.03, 2 * Math.sqrt(pc.l1)));
    arrow(ar2, pc.m, unit(pc.theta + Math.PI / 2), Math.max(0.03, 2 * Math.sqrt(Math.max(0, pc.l2))));
    // the line along u, projections and errors
    setAttrs(line, { x1: sx(o[0] - 2 * u[0]).toFixed(1), y1: sy(o[1] - 2 * u[1]).toFixed(1), x2: sx(o[0] + 2 * u[0]).toFixed(1), y2: sy(o[1] + 2 * u[1]).toFixed(1) });
    const hp: V2 = [o[0] + (centred ? 0.42 : 0.9) * u[0], o[1] + (centred ? 0.42 : 0.9) * u[1]];
    setAttrs(handle, { cx: sx(hp[0]).toFixed(1), cy: sy(hp[1]).toFixed(1) });
    setAttrs(handleLab, { x: (sx(hp[0]) + 14).toFixed(1), y: (sy(hp[1]) - 12).toFixed(1) });
    gErr.replaceChildren(); gProj.replaceChildren();
    for (const p of P) {
      const q = decode(score(p, o, th), o, th);
      svgEl("line", { x1: sx(p[0]).toFixed(1), y1: sy(p[1]).toFixed(1), x2: sx(q[0]).toFixed(1), y2: sy(q[1]).toFixed(1), stroke: RED, "stroke-width": 1.6 }, gErr);
      svgEl("circle", { cx: sx(q[0]).toFixed(1), cy: sy(q[1]).toFixed(1), r: 2.5, fill: BLUE }, gProj);
    }
    const p = P[sel], s = score(p, o, th), q = decode(s, o, th);
    setAttrs(selMark, { cx: sx(p[0]).toFixed(1), cy: sy(p[1]).toFixed(1) });
    swX(p); swR(q);
    // variance/error against angle (along the current line: through the mean or the origin)
    const angs = Array.from({ length: 181 }, (_, i) => i - 90);
    const curves = angs.map(a => along(P, o, a * Math.PI / 180));
    const top = Math.max(1e-9, ...curves.map(c => Math.max(c.meanSq, c.err))) * 1.08;
    ey = scale(0, top, EH - 30, 26);
    vPath.setAttribute("d", angs.map((a, i) => `${i ? "L" : "M"}${ex(a).toFixed(1)},${ey(curves[i].meanSq).toFixed(1)}`).join(""));
    rPath.setAttribute("d", angs.map((a, i) => `${i ? "L" : "M"}${ex(a).toFixed(1)},${ey(curves[i].err).toFixed(1)}`).join(""));
    const cur = along(P, o, th);
    setAttrs(vMark, { cx: ex(deg(th)).toFixed(1), cy: ey(cur.meanSq).toFixed(1) });
    setAttrs(rMark, { cx: ex(deg(th)).toFixed(1), cy: ey(cur.err).toFixed(1) });
    setAttrs(pcTick, { x1: ex(deg(pc.theta)).toFixed(1), x2: ex(deg(pc.theta)).toFixed(1), y1: 24, y2: EH - 28 });
    // scree bars
    const tot = pc.l1 + pc.l2, share = [pc.l1 / tot, Math.max(0, pc.l2) / tot];
    bars.forEach((b, i) => {
      const w = 190 * (Number.isFinite(share[i]) ? share[i] : 0);
      b.r.setAttribute("width", w.toFixed(1));
      b.t.setAttribute("x", (44 + w + 6).toFixed(1));
      b.t.textContent = `${(100 * share[i]).toFixed(1)} %, λ${i + 1} = ${f4(i ? pc.l2 : pc.l1)}`;
    });
    const u1 = unit(pc.theta);
    out.textContent =
      `mean = (${f3(pc.m[0])}, ${f3(pc.m[1])})     n = ${P.length}\n` +
      `S = [${f4(pc.a)}  ${f4(pc.b)}; ${f4(pc.b)}  ${f4(pc.c)}]\n` +
      `PC1 = (${f3(u1[0])}, ${f3(u1[1])}), angle ${deg(pc.theta).toFixed(1)}°\n` +
      `u   = (${f3(u[0])}, ${f3(u[1])}), angle ${deg(th).toFixed(1)}°` + (centred ? "" : ", through (0, 0)") + `\n` +
      (centred ? `variance along u ${f4(cur.meanSq)}   (uᵀSu = ${f4(varAlong(pc, th))})\n`
               : `mean squared score ${f4(cur.meanSq)}\n`) +
      (centred ? `error ${f4(cur.err)}   PCA best ${f4(Math.max(0, pc.l2))}   keep nothing ${f4(tot)}\n`
               : `error ${f4(cur.err)}   PCA best (centred) ${f4(Math.max(0, pc.l2))}\n`) +
      `x = (${f3(p[0])}, ${f3(p[1])}) → score ${f3(s)} → x̂ = (${f3(q[0])}, ${f3(q[1])})`;
  }

  setData(kind0, ang0, centre0);
}

autoMount(NAME, mount);
