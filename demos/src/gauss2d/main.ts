// One 2D Gaussian, or a mixture of 2–3 (data-mode="mixture"), with draggable handles: the centre
// (circle) moves the mean; the two square handles are the ends of the 1σ half-axes, which rotate
// and stretch the ellipse (axes = eigenvectors of Σ, half-lengths √λ). Drawn: the 1, 2 and 3σ
// ellipses, 300 seeded samples (each sample is μ + Q Λ^½ z with a fixed z, so they move with
// the handles), and two probe points P and Q with their Euclidean and Mahalanobis distances
// and the density. The constraint menu snaps Σ to spherical / diagonal / full. Mixture: weight
// sliders, the density or the most probable component as background, samples coloured by the
// component that produced them ("hide colours": what an algorithm sees), and the
// responsibilities of P.
//
// <div data-demo="gauss2d" data-preset="circle|cigar|maha|hidden|tied|curved" data-mode="single|mixture"></div>
import { autoMount, button, checkbox, htmlEl, presets, slider, svgEl, draggable, svgPoint, scale, type Slider } from "../lib/ui";
import { QUALITATIVE, hexToRgb } from "../lib/colour";
import { rng } from "../lib/random";
import { eigSym2 } from "../lib/linalg";

const NAME = "gauss2d";
const STYLE = `
[data-demo="${NAME}"] { gap: 1em; flex-wrap: nowrap; }
[data-demo="${NAME}"] .g2-plot { flex: 0 1 480px; min-width: 260px; }
[data-demo="${NAME}"] .g2-plot svg { touch-action: none; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 380px; max-width: 520px; gap: .4em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
[data-demo="${NAME}"] .g2-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
[data-demo="${NAME}"] .g2-bars { display: flex; height: 26px; border-radius: 4px; overflow: hidden; border: 1px solid #bbb; }
[data-demo="${NAME}"] .g2-bars span { display: grid; place-items: center; color: #fff; font-size: 14px; white-space: nowrap; overflow: hidden; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .g2-plot { flex: 0 0 500px; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 1 1 600px; max-width: 640px; gap: .25em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .12em .5em; }
.reveal [data-demo="${NAME}"] .qb-note { display: none; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; padding: .25em .5em; }
@media (max-width: 760px) { [data-demo="${NAME}"] { flex-wrap: wrap; } }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

type Constraint = "spherical" | "diagonal" | "full";
interface Comp { mx: number; my: number; s1: number; s2: number; th: number; w: number }
interface Preset { id: string; label: string; comps: Comp[]; P: [number, number]; Q: [number, number]; cons: Constraint; bg: Bg; showQ: boolean }
type Bg = "density" | "label" | "none";

const L = 5, S = 480, NS = 300;
const COLS = QUALITATIVE.slice(0, 3);
const deg = Math.PI / 180;
const MAHA = Math.atan2(1, 2);   // Σ = [[5, 2], [2, 2]]: λ = 6, 1; v1 = (2, 1)/√5

const PRESETS: Preset[] = [
  { id: "circle", label: "circle", comps: [{ mx: 0, my: 0, s1: 1.5, s2: 1.5, th: 0, w: 1 }], P: [2, 1], Q: [-1, 2], cons: "spherical", bg: "density", showQ: false },
  { id: "cigar", label: "tilted cigar", comps: [{ mx: 0, my: 0, s1: 2.5, s2: 0.6, th: 30 * deg, w: 1 }], P: [2, 1], Q: [-1, 2], cons: "full", bg: "density", showQ: false },
  { id: "maha", label: "same Euclidean distance, different Mahalanobis", comps: [{ mx: 0, my: 0, s1: Math.sqrt(6), s2: 1, th: MAHA, w: 1 }], P: [2, 1], Q: [-1, 2], cons: "full", bg: "density", showQ: true },
  { id: "hidden", label: "two overlapping components: the hidden labels", comps: [{ mx: -1, my: -0.5, s1: 1.6, s2: 0.8, th: 20 * deg, w: 0.6 }, { mx: 1.4, my: 1, s1: 1.2, s2: 0.7, th: -40 * deg, w: 0.4 }], P: [0.3, 0.3], Q: [-1, 2], cons: "full", bg: "density", showQ: false },
  { id: "tied", label: "same covariance: straight boundary", comps: [{ mx: -1.6, my: -0.8, s1: 1.4, s2: 0.7, th: 35 * deg, w: 0.5 }, { mx: 1.6, my: 0.8, s1: 1.4, s2: 0.7, th: 35 * deg, w: 0.5 }], P: [0, 0], Q: [-1, 2], cons: "full", bg: "label", showQ: false },
  { id: "curved", label: "different covariances: curved boundary", comps: [{ mx: -1.2, my: 0, s1: 0.6, s2: 0.6, th: 0, w: 0.5 }, { mx: 1.2, my: 0, s1: 2.2, s2: 1.4, th: 0, w: 0.5 }], P: [0, 0], Q: [-1, 2], cons: "full", bg: "label", showQ: false },
];

const f2 = (v: number) => v.toFixed(2), f3 = (v: number) => v.toFixed(3);

/** Σ of a component from its axes. */
function sigma(c: Comp): [number, number, number] {
  const cs = Math.cos(c.th), sn = Math.sin(c.th), a = c.s1 * c.s1, b = c.s2 * c.s2;
  return [a * cs * cs + b * sn * sn, (a - b) * cs * sn, a * sn * sn + b * cs * cs];
}
function maha2(c: Comp, x: number, y: number): number {
  const cs = Math.cos(c.th), sn = Math.sin(c.th), dx = x - c.mx, dy = y - c.my;
  const u = cs * dx + sn * dy, v = -sn * dx + cs * dy;
  return (u / c.s1) ** 2 + (v / c.s2) ** 2;
}
const pdf = (c: Comp, x: number, y: number) => Math.exp(-0.5 * maha2(c, x, y)) / (2 * Math.PI * c.s1 * c.s2);

let mounted = 0;

function mount(host: HTMLElement): void {
  injectStyle();
  const mixMode = host.dataset.mode === "mixture";
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? (mixMode ? PRESETS[3] : PRESETS[1]);
  let comps: Comp[] = [], P: [number, number] = [0, 0], Q: [number, number] = [0, 0], cons: Constraint = "full", bg: Bg = "density";
  let showQ = false, hide = false, seed = 1;

  // ---- SVG ----
  const plotDiv = htmlEl("div", "g2-plot", "", host);
  const svg = svgEl("svg", { viewBox: `0 0 ${S} ${S}`, width: "100%", role: "img", "aria-label": "A 2D Gaussian with its covariance ellipses" }, plotDiv);
  const sx = scale(-L, L, 8, S - 8), sy = scale(-L, L, S - 8, 8), k = (S - 16) / (2 * L);
  const bgImg = svgEl("image", { x: 8, y: 8, width: S - 16, height: S - 16, preserveAspectRatio: "none" }, svg);
  for (let v = -4; v <= 4; v++) {
    svgEl("line", { x1: sx(v), x2: sx(v), y1: sy(-L), y2: sy(L), class: v === 0 ? "axis" : "grid", "stroke-width": v === 0 ? 1.2 : 1 }, svg);
    svgEl("line", { y1: sy(v), y2: sy(v), x1: sx(-L), x2: sx(L), class: v === 0 ? "axis" : "grid", "stroke-width": v === 0 ? 1.2 : 1 }, svg);
  }
  for (const v of [-4, -2, 2, 4]) {
    svgEl("text", { x: sx(v), y: sy(0) + 20, "text-anchor": "middle", style: "font-size:16px", fill: "#666" }, svg).textContent = String(v);
    svgEl("text", { x: sx(0) - 6, y: sy(v) + 5, "text-anchor": "end", style: "font-size:16px", fill: "#666" }, svg).textContent = String(v);
  }
  svgEl("rect", { x: 8, y: 8, width: S - 16, height: S - 16, fill: "none", stroke: "#444", "stroke-width": 1.5 }, svg);
  const clipId = "g2clip" + (++mounted);
  svgEl("rect", { x: 8, y: 8, width: S - 16, height: S - 16 }, svgEl("clipPath", { id: clipId }, svgEl("defs", {}, svg)));
  const gSamples = svgEl("g", { "clip-path": `url(#${clipId})` }, svg);
  const gEll = svgEl("g", { "clip-path": `url(#${clipId})` }, svg);
  const gProbe = svgEl("g", {}, svg);
  const gHandles = svgEl("g", {}, svg);
  const toWorld = (ev: PointerEvent): [number, number] => { const [x, y] = svgPoint(svg, ev); return [Math.max(-L, Math.min(L, sx.invert(x))), Math.max(-L, Math.min(L, sy.invert(y)))]; };

  // ---- panel ----
  const panel = htmlEl("div", "qb-panel", "", host);
  const ro = htmlEl("div", "qb-readout", "", panel);
  const barLab = htmlEl("div", "g2-lab", "responsibilities of P (which component produced it?)", panel);
  const bars = htmlEl("div", "g2-bars", "", panel);
  const wDiv = htmlEl("div", "", "", panel);
  let wSliders: Slider[] = [];
  const row = htmlEl("div", "qb-presets", "", panel);
  htmlEl("span", "g2-lab", "Σ:", row);
  const consBtns = (["spherical", "diagonal", "full"] as Constraint[]).map(c => button(row, c, () => { cons = c; comps.forEach(snap); draw(); }));
  const row2 = htmlEl("div", "qb-presets", "", panel);
  htmlEl("span", "g2-lab", "background:", row2);
  const bgBtns = (["density", "label", "none"] as Bg[]).map(b => button(row2, b === "label" ? "most probable component" : b, () => { bg = b; draw(); }));
  const row3 = htmlEl("div", "qb-presets", "", panel);
  button(row3, "resample", () => { seed++; makeSamples(); draw(); });
  const addBtn = button(row3, "add component", () => { if (comps.length < 3) { comps.push({ mx: 2.5, my: -2.5, s1: 0.8, s2: 0.8, th: 0, w: 0.25 }); snap(comps[comps.length - 1]); rebuild(); } });
  const remBtn = button(row3, "remove component", () => { if (comps.length > 1) { comps.pop(); rebuild(); } });
  const hideCb = checkbox(panel, "hide colours (what the algorithm sees)", false, v => { hide = v; draw(); });
  const qCb = checkbox(panel, "show probe Q", false, v => { showQ = v; draw(); });
  htmlEl("div", "qb-note", "Drag the circle (mean) and the squares (ends of the 1σ half-axes). Drag the probes P and Q.", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  button(panel, "Reset", () => load(p0)).style.alignSelf = "flex-start";

  // ---- samples: fixed z and u per sample ----
  let Z: Float64Array = new Float64Array(0), U: Float64Array = new Float64Array(0);
  function makeSamples(): void {
    const r = rng(seed);
    Z = new Float64Array(2 * NS); U = new Float64Array(NS);
    for (let i = 0; i < NS; i++) { Z[2 * i] = r.norm(); Z[2 * i + 1] = r.norm(); U[i] = r.unif(); }
  }

  function snap(c: Comp): void {
    if (cons === "spherical") { const s = Math.sqrt(c.s1 * c.s2); c.s1 = s; c.s2 = s; c.th = 0; }
    else if (cons === "diagonal") {
      // keep the axis that is closest to horizontal as axis 1
      const cs = Math.abs(Math.cos(c.th));
      if (cs < Math.SQRT1_2) { const t = c.s1; c.s1 = c.s2; c.s2 = t; }
      c.th = 0;
    }
  }

  function load(p: Preset): void {
    comps = p.comps.map(c => ({ ...c })); P = [...p.P]; Q = [...p.Q]; cons = p.cons; bg = p.bg; showQ = p.showQ; qCb.checked = showQ;
    hide = false; hideCb.checked = false; seed = 1; makeSamples();
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    rebuild();
  }

  function rebuild(): void {
    // weight sliders
    wDiv.replaceChildren();
    wSliders = comps.length > 1 ? comps.map((c, j) => slider(wDiv, `weight ${j + 1}`, 0.05, 1, 0.05, c.w, v => { c.w = v; draw(); }, () => f2(c.w / comps.reduce((a, q) => a + q.w, 0)))) : [];
    wSliders.forEach((s, j) => (s.input.parentElement as HTMLElement).style.setProperty("color", COLS[j]));
    // handles
    gHandles.replaceChildren();
    comps.forEach((c, j) => {
      const col = COLS[j];
      const a1 = svgEl("line", { "stroke-width": 3, stroke: col }, gHandles);
      const a2 = svgEl("line", { "stroke-width": 3, stroke: col, "stroke-dasharray": "6 4" }, gHandles);
      const h0 = svgEl("circle", { r: 12, class: "handle", stroke: col }, gHandles);
      const h1 = svgEl("rect", { width: 20, height: 20, rx: 3, fill: col, stroke: "#fff", "stroke-width": 2 }, gHandles);
      const h2 = svgEl("rect", { width: 18, height: 18, rx: 3, fill: "#fff", stroke: col, "stroke-width": 3 }, gHandles);
      draggable(h0, toWorld, (x, y) => { c.mx = x; c.my = y; draw(); });
      draggable(h1, toWorld, (x, y) => {
        const dx = x - c.mx, dy = y - c.my, len = Math.max(0.15, Math.hypot(dx, dy));
        if (cons === "spherical") { c.s1 = len; c.s2 = len; }
        else if (cons === "diagonal") { c.th = 0; c.s1 = Math.max(0.15, Math.abs(dx)); }
        else { c.th = Math.atan2(dy, dx); c.s1 = len; }
        draw();
      });
      draggable(h2, toWorld, (x, y) => {
        const dx = x - c.mx, dy = y - c.my;
        if (cons === "spherical") { const len = Math.max(0.15, Math.hypot(dx, dy)); c.s1 = len; c.s2 = len; }
        else if (cons === "diagonal") { c.th = 0; c.s2 = Math.max(0.15, Math.abs(dy)); }
        else c.s2 = Math.max(0.15, Math.abs(-Math.sin(c.th) * dx + Math.cos(c.th) * dy));
        draw();
      });
      (c as Comp & { el?: SVGElement[] }).el = [a1, a2, h0, h1, h2];
    });
    // probes
    gProbe.replaceChildren();
    const mkProbe = (lab: string, get: () => [number, number], set: (p: [number, number]) => void) => {
      const g = svgEl("g", {}, gProbe);
      svgEl("path", { d: "M-9,-9L9,9M-9,9L9,-9", stroke: "#000", "stroke-width": 4 }, g);
      svgEl("circle", { r: 14, fill: "transparent" }, g);
      const t = svgEl("text", { x: 10, y: -10, "font-weight": 700, style: "font-size:20px", class: "qb-halo" }, g);
      t.textContent = lab;
      draggable(g as unknown as SVGElement, toWorld, (x, y) => { set([x, y]); draw(); });
      return { g, get };
    };
    probes = [mkProbe("P", () => P, p => { P = p; }), mkProbe("Q", () => Q, p => { Q = p; })];
    draw();
  }
  let probes: { g: SVGGElement; get: () => [number, number] }[] = [];

  // ---- drawing ----
  const RES = 96, canvas = document.createElement("canvas");
  canvas.width = RES; canvas.height = RES;
  function drawBackground(W: number[]): void {
    if (bg === "none") { bgImg.removeAttribute("href"); return; }
    const ctx = canvas.getContext("2d")!, id = ctx.createImageData(RES, RES);
    const vals = new Float64Array(RES * RES), lab = new Int32Array(RES * RES), conf = new Float64Array(RES * RES);
    let mx = 0;
    for (let j = 0; j < RES; j++) for (let i = 0; i < RES; i++) {
      const x = -L + (i + 0.5) * 2 * L / RES, y = L - (j + 0.5) * 2 * L / RES;
      let s = 0, b = 0, bv = -1;
      comps.forEach((c, q) => { const v = W[q] * pdf(c, x, y); s += v; if (v > bv) { bv = v; b = q; } });
      vals[j * RES + i] = s; lab[j * RES + i] = b; conf[j * RES + i] = s > 0 ? bv / s : 1;
      if (s > mx) mx = s;
    }
    const rgb = COLS.map(h => hexToRgb(h).map(v => 255 * v));
    for (let p = 0; p < RES * RES; p++) {
      if (bg === "density") {
        const t = Math.pow(vals[p] / (mx || 1), 0.6);
        id.data[4 * p] = 255 - 200 * t; id.data[4 * p + 1] = 255 - 120 * t; id.data[4 * p + 2] = 255 - 30 * t; id.data[4 * p + 3] = 255;
      } else {
        const c = rgb[lab[p]], a = 0.15 + 0.35 * (comps.length > 1 ? (conf[p] - 1 / comps.length) / (1 - 1 / comps.length) : 1);
        id.data[4 * p] = 255 + (c[0] - 255) * a; id.data[4 * p + 1] = 255 + (c[1] - 255) * a; id.data[4 * p + 2] = 255 + (c[2] - 255) * a; id.data[4 * p + 3] = 255;
      }
    }
    ctx.putImageData(id, 0, 0);
    bgImg.setAttribute("href", canvas.toDataURL());
  }

  function draw(): void {
    const tot = comps.reduce((a, c) => a + c.w, 0), W = comps.map(c => c.w / tot);
    drawBackground(W);
    // samples
    gSamples.replaceChildren();
    const cum = W.map((_, j) => W.slice(0, j + 1).reduce((a, b) => a + b, 0));
    for (let i = 0; i < NS; i++) {
      let j = cum.findIndex(v => U[i] < v);
      if (j < 0) j = comps.length - 1;
      const c = comps[j], cs = Math.cos(c.th), sn = Math.sin(c.th);
      const u = c.s1 * Z[2 * i], v = c.s2 * Z[2 * i + 1];
      const x = c.mx + cs * u - sn * v, y = c.my + sn * u + cs * v;
      if (Math.abs(x) > L || Math.abs(y) > L) continue;
      svgEl("circle", { cx: sx(x).toFixed(1), cy: sy(y).toFixed(1), r: 3.2, fill: hide || comps.length === 1 ? "#555" : COLS[j], "fill-opacity": 0.6 }, gSamples);
    }
    // ellipses and handles
    gEll.replaceChildren();
    comps.forEach((c, j) => {
      for (const n of [1, 2, 3]) {
        svgEl("ellipse", {
          cx: sx(c.mx).toFixed(1), cy: sy(c.my).toFixed(1), rx: (n * c.s1 * k).toFixed(1), ry: (n * c.s2 * k).toFixed(1),
          transform: `rotate(${(-c.th / deg).toFixed(2)} ${sx(c.mx).toFixed(1)} ${sy(c.my).toFixed(1)})`,
          fill: "none", stroke: COLS[j], "stroke-width": n === 1 ? 3 : 1.6, "stroke-dasharray": n === 1 ? "" : "6 5", opacity: n === 1 ? 1 : 0.8,
        }, gEll);
      }
      const [a1, a2, h0, h1, h2] = (c as Comp & { el: SVGElement[] }).el;
      const e1 = [c.mx + c.s1 * Math.cos(c.th), c.my + c.s1 * Math.sin(c.th)], e2 = [c.mx - c.s2 * Math.sin(c.th), c.my + c.s2 * Math.cos(c.th)];
      for (const [ln, e] of [[a1, e1], [a2, e2]] as [SVGElement, number[]][]) {
        ln.setAttribute("x1", sx(c.mx).toFixed(1)); ln.setAttribute("y1", sy(c.my).toFixed(1));
        ln.setAttribute("x2", sx(e[0]).toFixed(1)); ln.setAttribute("y2", sy(e[1]).toFixed(1));
      }
      h0.setAttribute("cx", sx(c.mx).toFixed(1)); h0.setAttribute("cy", sy(c.my).toFixed(1));
      h1.setAttribute("x", (sx(e1[0]) - 10).toFixed(1)); h1.setAttribute("y", (sy(e1[1]) - 10).toFixed(1));
      h2.setAttribute("x", (sx(e2[0]) - 9).toFixed(1)); h2.setAttribute("y", (sy(e2[1]) - 9).toFixed(1));
    });
    probes.forEach((p, i) => {
      const [x, y] = p.get();
      p.g.setAttribute("transform", `translate(${sx(x).toFixed(1)},${sy(y).toFixed(1)})`);
      p.g.style.display = i === 1 && !showQ ? "none" : "";
    });
    // readout
    const lines: string[] = [];
    comps.forEach((c, j) => {
      const [a, b, d] = sigma(c), e = eigSym2(a, b, d);
      const pre = comps.length > 1 ? `${j + 1}: π = ${f2(W[j])}  ` : "";
      lines.push(`${pre}μ = (${f2(c.mx)}, ${f2(c.my)})   Σ = [${f2(a)}  ${f2(b)}; ${f2(b)}  ${f2(d)}]`);
      lines.push(`${comps.length > 1 ? "   " : ""}λ1 = ${f2(e.l1)}, λ2 = ${f2(e.l2)}   v1 = (${f2(e.v1[0])}, ${f2(e.v1[1])}), angle ${(e.angle / deg).toFixed(1)}°`);
    });
    const probeLine = (lab: string, [x, y]: [number, number]) => {
      if (comps.length === 1) {
        const c = comps[0];
        return `${lab} = (${f2(x)}, ${f2(y)}): Euclidean ${f2(Math.hypot(x - c.mx, y - c.my))}, Mahalanobis ${f2(Math.sqrt(maha2(c, x, y)))}, p = ${f3(pdf(c, x, y))}`;
      }
      const p = comps.reduce((s, c, j) => s + W[j] * pdf(c, x, y), 0);
      return `${lab} = (${f2(x)}, ${f2(y)}): p(${lab}) = ${f3(p)}, d_M to ` + comps.map((c, j) => `${j + 1}: ${f2(Math.sqrt(maha2(c, x, y)))}`).join(", ");
    };
    lines.push(probeLine("P", P));
    if (showQ) lines.push(probeLine("Q", Q));
    ro.textContent = lines.join("\n");
    // responsibilities of P
    const mix = comps.length > 1;
    bars.style.display = mix ? "" : "none"; barLab.style.display = mix ? "" : "none";
    if (mix) {
      const v = comps.map((c, j) => W[j] * pdf(c, P[0], P[1])), s = v.reduce((a, b) => a + b, 0) || 1;
      bars.replaceChildren();
      v.forEach((x, j) => {
        const sp = htmlEl("span", "", `${(100 * x / s).toFixed(0)} %`, bars);
        sp.style.width = `${(100 * x / s).toFixed(2)}%`; sp.style.background = COLS[j];
      });
    }
    consBtns.forEach((b, i) => b.classList.toggle("active", ["spherical", "diagonal", "full"][i] === cons));
    bgBtns.forEach((b, i) => b.classList.toggle("active", ["density", "label", "none"][i] === bg));
    addBtn.disabled = comps.length >= 3; remBtn.disabled = comps.length <= 1;
    hideCb.parentElement!.style.display = mix ? "" : "none";
  }

  load(p0);
}

autoMount(NAME, mount);
