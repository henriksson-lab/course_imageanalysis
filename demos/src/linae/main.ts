// PCA as a linear autoencoder, and what a nonlinear autoencoder adds.
// Each point is a two-pixel image (p1, p2). A linear autoencoder with one number keeps the
// score along a unit direction u through the mean: encode s = u·(x − mean), decode
// x̂ = mean + s u. Drag the direction; the red segments are the reconstruction errors and the
// small plot shows the mean squared error for every angle. "PCA" jumps to the best angle.
// Dataset "spot" (images of a moving spot) lies on a curve: "train nonlinear autoencoder"
// fits a 2 → 16 → 1 → 16 → 2 ReLU network live and draws its decoder curve.
//
// <div data-demo="linae" data-data="pixels|spot" data-angle="0"></div>
import { autoMount, button, checkbox, htmlEl, presets, svgEl, scale, draggable, svgPoint } from "../lib/ui";
import type { Mlp, Adam } from "../lib/nn";
import { makeData, pca, linEncode, linDecode, linError, makeAE, aeTrain, aeError, aeDecode, aeCode, type V2 } from "./data";

const NAME = "linae";
const STYLE = `
[data-demo="${NAME}"] { gap: 1em; }
[data-demo="${NAME}"] .la-plot { flex: 0 1 470px; min-width: 0; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 360px; max-width: 520px; gap: .45em; }
[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre; }
[data-demo="${NAME}"] svg text { font-size: 18px; }
[data-demo="${NAME}"] .la-sw { display: flex; gap: 14px; align-items: flex-end; }
[data-demo="${NAME}"] .la-sw figure { margin: 0; text-align: center; font-size: 15px; color: var(--qb-muted); }
.reveal [data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal [data-demo="${NAME}"] .la-plot { flex: 0 0 440px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 1 1 560px; max-width: 620px; gap: .3em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .3em .5em; }
.reveal [data-demo="${NAME}"] .qb-panel > svg { max-height: 120px; }
.reveal [data-demo="${NAME}"] .qb-note { font-size: 16px; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const S = 470, LO = -0.1, HI = 1.15, AE_STEPS = 1500;
const grey = (v: number) => { const g = Math.round(255 * Math.max(0, Math.min(1, v))); return `rgb(${g},${g},${g})`; };

function mount(host: HTMLElement): void {
  injectStyle();
  const kind0 = host.dataset.data === "spot" ? "spot" : "pixels";
  const ang0 = Number(host.dataset.angle ?? 0) * Math.PI / 180 || 0;
  let kind: "pixels" | "spot" = kind0, P = makeData(kind), pc = pca(P), th = ang0, sel = 0;
  let ae: { net: Mlp; opt: Adam } | null = null, aeSteps = 0, timer = 0, showAE = true;

  const plotDiv = htmlEl("div", "la-plot", "", host);
  const svg = svgEl("svg", { viewBox: `0 0 ${S} ${S}`, width: "100%", role: "img", "aria-label": "Two-pixel images as points" }, plotDiv);
  const sx = scale(LO, HI, 40, S - 8), sy = scale(LO, HI, S - 40, 8);
  for (const v of [0, 0.5, 1]) {
    svgEl("line", { x1: sx(v), x2: sx(v), y1: sy(LO), y2: sy(HI), class: "grid" }, svg);
    svgEl("line", { y1: sy(v), y2: sy(v), x1: sx(LO), x2: sx(HI), class: "grid" }, svg);
    svgEl("text", { x: sx(v), y: S - 16, "text-anchor": "middle" }, svg).textContent = String(v);
    svgEl("text", { x: 30, y: sy(v) + 6, "text-anchor": "end" }, svg).textContent = String(v);
  }
  svgEl("text", { x: S - 10, y: S - 46, "text-anchor": "end", class: "halo" }, svg).textContent = "pixel 1 →";
  svgEl("text", { x: 46, y: 26 }, svg).textContent = "↑ pixel 2";
  const gErr = svgEl("g", {}, svg);
  const line = svgEl("line", { stroke: "#1f77b4", "stroke-width": 3 }, svg);
  const aePath = svgEl("path", { fill: "none", stroke: "#2ca02c", "stroke-width": 4 }, svg);
  const gPts = svgEl("g", {}, svg);
  const selMark = svgEl("circle", { r: 9, fill: "none", stroke: "#000", "stroke-width": 2.5 }, svg);
  const handle = svgEl("circle", { r: 12, class: "handle", stroke: "#1f77b4" }, svg);
  const handleLab = svgEl("text", { class: "halo", fill: "#1f77b4", "font-weight": "600" }, svg);
  handleLab.textContent = "u";

  const panel = htmlEl("div", "qb-panel", "", host);
  // error vs angle plot
  const EW = 400, EH = 170;
  const esvg = svgEl("svg", { viewBox: `0 0 ${EW} ${EH}`, width: "100%" }, panel);
  const ex = scale(-90, 90, 30, EW - 24);
  let ey = scale(0, 1, EH - 30, 10);
  const ePath = svgEl("path", { fill: "none", stroke: "#1f77b4", "stroke-width": 2.5 }, esvg);
  const eMark = svgEl("circle", { r: 6, fill: "#d62728" }, esvg);
  const eAE = svgEl("line", { stroke: "#2ca02c", "stroke-width": 2.5, "stroke-dasharray": "6 4", x1: ex(-90), x2: ex(90) }, esvg);
  for (const a of [-90, -45, 0, 45, 90]) svgEl("text", { x: ex(a), y: EH - 6, "text-anchor": "middle", style: "font-size:19px" }, esvg).textContent = `${a}°`;
  svgEl("text", { x: 4, y: 20, style: "font-size:19px" }, esvg).textContent = "error vs angle of u";
  const eMax = svgEl("text", { x: 46, y: 18 + 0, "text-anchor": "end", style: "font-size:14px" }, esvg);

  const sw = htmlEl("div", "la-sw", "", panel);
  const swatch = (cap: string) => {
    const f = htmlEl("figure", "", "", sw);
    const s2 = svgEl("svg", { viewBox: "0 0 84 42", width: 84, height: 42 }, f);
    const r1 = svgEl("rect", { x: 1, y: 1, width: 40, height: 40, stroke: "#999" }, s2);
    const r2 = svgEl("rect", { x: 43, y: 1, width: 40, height: 40, stroke: "#999" }, s2);
    const c = htmlEl("figcaption", "", cap, f);
    return { set: (p: V2) => { r1.setAttribute("fill", grey(p[0])); r2.setAttribute("fill", grey(p[1])); }, cap: c };
  };
  const swX = swatch("image x"), swL = swatch("linear x̂"), swA = swatch("autoenc. x̂");
  const out = htmlEl("div", "qb-readout", "", panel);
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "PCA", () => { th = pc.theta; draw(); });
  const aeBtn = button(row, `train nonlinear autoencoder (${AE_STEPS} steps)`, () => trainAE());
  const cbAE = checkbox(panel, "show the autoencoder", true, v => { showAE = v; draw(); });
  htmlEl("div", "qb-note", "Drag u (blue handle). Click a point to see it as a two-pixel image. Red: what is lost by keeping one number.", panel);
  presets(panel, [
    { label: "neighbouring pixels", apply: () => setData("pixels", 0) },
    { label: "moving spot", apply: () => setData("spot", 0) },
    { label: "spot: PCA", apply: () => { setData("spot", 0); th = pc.theta; draw(); } },
    { label: "Reset", apply: () => setData(kind0, ang0) },
  ]);

  function setData(k: "pixels" | "spot", angle: number): void {
    cancelAnimationFrame(timer); timer = 0;
    kind = k; P = makeData(k); pc = pca(P); th = angle; ae = null; aeSteps = 0; sel = k === "spot" ? 20 : 0;
    showAE = true; cbAE.checked = true;
    gPts.replaceChildren();
    P.forEach((p, i) => {
      const c = svgEl("circle", { cx: sx(p[0]).toFixed(1), cy: sy(p[1]).toFixed(1), r: 4.5, fill: "#555", "fill-opacity": 0.75 }, gPts);
      c.addEventListener("click", () => { sel = i; draw(); });
    });
    // error curve
    const angs = Array.from({ length: 181 }, (_, i) => i - 90);
    const errs = angs.map(a => linError(P, a * Math.PI / 180));
    const top = Math.max(...errs) * 1.1;
    ey = scale(0, top, EH - 32, 30);
    ePath.setAttribute("d", angs.map((a, i) => `${i ? "L" : "M"}${ex(a).toFixed(1)},${ey(errs[i]).toFixed(1)}`).join(""));
    eMax.textContent = "";
    draw();
  }
  function trainAE(): void {
    if (!ae) { ae = makeAE(); aeSteps = 0; }
    cancelAnimationFrame(timer);
    const target = aeSteps + AE_STEPS;
    const tick = () => { const n = Math.min(30, target - aeSteps); aeTrain(ae!.net, ae!.opt, P, n); aeSteps += n; timer = aeSteps < target ? requestAnimationFrame(tick) : 0; draw(); };
    timer = requestAnimationFrame(tick);
    draw();
  }

  const toWorld = (ev: { clientX: number; clientY: number }): [number, number] => { const [x, y] = svgPoint(svg, ev); return [sx.invert(x), sy.invert(y)]; };
  draggable(handle, toWorld, (x, y) => {
    let a = Math.atan2(y - pc.m[1], x - pc.m[0]);
    if (a > Math.PI / 2) a -= Math.PI; if (a < -Math.PI / 2) a += Math.PI;   // a line has no arrow: keep u in (−90°, 90°]
    th = a; draw();
  });

  function draw(): void {
    const m = pc.m, u: V2 = [Math.cos(th), Math.sin(th)];
    setAttrs2(line, { x1: sx(m[0] - 2 * u[0]), y1: sy(m[1] - 2 * u[1]), x2: sx(m[0] + 2 * u[0]), y2: sy(m[1] + 2 * u[1]) });
    const hp: V2 = [m[0] + 0.42 * u[0], m[1] + 0.42 * u[1]];
    setAttrs2(handle, { cx: sx(hp[0]), cy: sy(hp[1]) });
    setAttrs2(handleLab, { x: sx(hp[0]) + 14, y: sy(hp[1]) - 12 });
    gErr.replaceChildren();
    const useAE = ae && showAE;
    for (const p of P) {
      const q = useAE ? (aeDecodeOf(p)) : linDecode(linEncode(p, m, th), m, th);
      svgEl("line", { x1: sx(p[0]).toFixed(1), y1: sy(p[1]).toFixed(1), x2: sx(q[0]).toFixed(1), y2: sy(q[1]).toFixed(1), stroke: "#d62728", "stroke-width": 1.6 }, gErr);
    }
    line.style.opacity = useAE ? "0.35" : "1";
    if (ae && showAE) {
      const codes = P.map(p => aeCode(ae!.net, p));
      const lo = Math.min(...codes), hi = Math.max(...codes);
      const zs = Array.from({ length: 121 }, (_, i) => lo + (hi - lo) * i / 120);
      aePath.setAttribute("d", zs.map((z, i) => { const q = aeDecode(ae!.net, z); return `${i ? "L" : "M"}${sx(q[0]).toFixed(1)},${sy(q[1]).toFixed(1)}`; }).join(""));
      aePath.style.display = "";
    } else aePath.style.display = "none";
    const p = P[sel];
    setAttrs2(selMark, { cx: sx(p[0]), cy: sy(p[1]) });
    const s = linEncode(p, m, th), q = linDecode(s, m, th);
    swX.set(p); swL.set(q);
    if (ae) swA.set(aeDecodeOf(p)); else swA.set([1, 1]);
    swA.cap.textContent = ae ? "autoenc. x̂" : "(not trained)";
    const deg = th * 180 / Math.PI;
    setAttrs2(eMark, { cx: ex(deg), cy: ey(linError(P, th)) });
    const eA = ae ? aeError(ae.net, P) : NaN;
    eAE.style.display = ae ? "" : "none";
    if (ae) { eAE.setAttribute("y1", String(ey(eA))); eAE.setAttribute("y2", String(ey(eA))); }
    const L = linError(P, th), tot = pc.l1 + pc.l2;
    out.textContent =
      `u = (${u[0].toFixed(3)}, ${u[1].toFixed(3)}), angle ${deg.toFixed(1)}°\n` +
      `image x = (${p[0].toFixed(3)}, ${p[1].toFixed(3)})\n` +
      `score ${s.toFixed(3)} → x̂ = (${q[0].toFixed(3)}, ${q[1].toFixed(3)})\n` +
      `linear error   ${L.toFixed(4)}\n` +
      `PCA best       ${pc.l2.toFixed(4)}\n` +
      `keep nothing   ${tot.toFixed(4)}\n` +
      (ae ? `autoencoder    ${eA.toFixed(4)} (${aeSteps} steps)` : `autoencoder    not trained`);
    aeBtn.disabled = timer !== 0;
  }
  const aeDecodeOf = (p: V2): V2 => aeDecode(ae!.net, aeCode(ae!.net, p));
  setData(kind, th);
}

function setAttrs2(e: Element, a: Record<string, number>): void { for (const [k, v] of Object.entries(a)) e.setAttribute(k, v.toFixed(1)); }

autoMount(NAME, mount);
