// Matching outlines with Fourier descriptors (shape-matching day, unit fourier-descriptors/).
// Outline A (blue) and outline B (red): B is A turned, scaled, started m samples later,
// mirrored and noisy, or a different shape. Panels: A and B with their reconstruction from
// |k| ≤ K (epicycles animate it), B with A moved onto it at the best match; the magnitudes
// |c_k| of both; the phase of c_k(B) relative to c_k(A) (constant: rotation; ramp: new start);
// the match score over the starting point m (one FFT) with the recovered angle and scale.
// Options: data-preset="<id>", data-presets="id,id,…" (which preset buttons), data-controls
// (names of the sliders/checkboxes to show, e.g. "K,angle,start").
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { Pt, Coeffs, coef, evalSeries, epicycles, sampleMean } from "../lib/contour";
import { compute, DEFAULTS, PRESETS, Preset, Settings, Result, SHAPES, ShapeId, NUCLEI, misfit } from "./model";

const NAME = "fdmatch";
const BLUE = "#1f77b4", RED = "#d62728", GREY = "#9a9a9a";

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; }
.qb-demo[data-demo="${NAME}"] figure { margin: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 15px; text-align: center; line-height: 1.15; margin-bottom: .1em; }
.qb-demo[data-demo="${NAME}"] .fd-figs { flex: 1 1 500px; max-width: 760px; min-width: 300px; display: grid; grid-template-columns: repeat(3, 1fr); gap: .3em .6em; align-items: start; }
.qb-demo[data-demo="${NAME}"] .fd-figs svg { width: 100%; height: auto; }
.qb-demo[data-demo="${NAME}"] .fd-out svg { max-height: min(250px, calc(50vh - 120px)); }
.qb-demo[data-demo="${NAME}"] .qb-plot { max-width: none; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 270px; min-width: 240px; max-width: 330px; gap: .3em; font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-btn { font-size: 13px; padding: .1em .45em; }
.qb-demo[data-demo="${NAME}"] .qb-panel .qb-presets { gap: .25em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 13px; white-space: pre; overflow-x: auto; flex-shrink: 0; }
.qb-demo[data-demo="${NAME}"] .fd-lab { font-size: 13px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .fd-row { display: flex; flex-wrap: wrap; gap: 0 .8em; }
.qb-demo[data-demo="${NAME}"] .fd-sel select { font: inherit; font-size: 14px; }
.qb-demo[data-demo="${NAME}"] .fd-off { display: none !important; }
.qb-demo[data-demo="${NAME}"] .qb-slider { gap: .35em; }
@media (max-width: 640px) { .qb-demo[data-demo="${NAME}"] .fd-figs { grid-template-columns: 1fr 1fr; } }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .fd-figs { flex: 1 1 700px; max-width: 760px; }
.reveal .qb-demo[data-demo="${NAME}"] .fd-out svg { max-height: 235px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { font-size: 15px; max-width: 330px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
.reveal .slides .qb-demo[data-demo="${NAME}"] figcaption { font-size: 17px; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const f2 = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "–");
const DEG = 180 / Math.PI;

/** An outline panel: an SVG 0..S × 0..S with layers. */
interface OutPanel { svg: SVGSVGElement; mask: SVGPathElement; path: SVGPathElement; rec: SVGPathElement; epi: SVGGElement; dots: SVGGElement; extra: SVGPathElement }
const S = 240;

function outPanel(parent: Element, caption: string, col: string): OutPanel {
  const fig = htmlEl("figure", "fd-out", "", parent);
  htmlEl("figcaption", "", caption, fig);
  const svg = svgEl("svg", { viewBox: `0 0 ${S} ${S}` }, fig);
  svgEl("rect", { x: 0, y: 0, width: S, height: S, rx: 6, fill: "var(--qb-panel, #f6f7f9)" }, svg);
  const mask = svgEl("path", { fill: "#d9d9d9" }, svg);
  const path = svgEl("path", { fill: "none", stroke: GREY, "stroke-width": 1.5 }, svg);
  const extra = svgEl("path", { fill: "none", stroke: BLUE, "stroke-width": 2.5, "stroke-dasharray": "6 4" }, svg);
  const rec = svgEl("path", { fill: "none", stroke: col, "stroke-width": 3 }, svg);
  const epi = svgEl("g", {}, svg);
  const dots = svgEl("g", {}, svg);
  return { svg, mask, path, rec, epi, dots, extra };
}

const pathD = (pts: Pt[], close = true) => pts.length ? "M" + pts.map(p => p[0].toFixed(2) + " " + p[1].toFixed(2)).join("L") + (close ? "Z" : "") : "";

function mount(host: HTMLElement): void {
  injectStyle();
  const ids = host.dataset.presets?.split(",").map(x => x.trim());
  const LIST = ids ? PRESETS.filter(p => ids.includes(p.id)) : PRESETS;
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? LIST[0];
  const st: Settings = { ...DEFAULTS };
  let animate = true, undo = false, t = 0, last = 0, res: Result | null = null;

  // ---- layout ----
  const figs = htmlEl("div", "fd-figs", "", host);
  const pA = outPanel(figs, "A, rebuilt from |k| ≤ K", BLUE);
  const pB = outPanel(figs, "B, rebuilt from |k| ≤ K", RED);
  const pO = outPanel(figs, "B, and A moved onto it", RED);
  const fM = plotFigure(figs, "sizes |c_k| (A blue, B red)");
  const magP = new Plot(fM, { width: 300, height: 230, x: [-10.5, 10.5], y: [1e-4, 2], ylog: true, xlabel: "k", yticks: [1e-4, 1e-3, 1e-2, 0.1, 1], margin: { l: 58 }, font: 17, grid: true });
  const fP = plotFigure(figs, "phase of c_k(B) rel. to c_k(A)");
  const phP = new Plot(fP, { width: 300, height: 230, x: [-10.5, 10.5], y: [-180, 180], xlabel: "k", yticks: [-180, -90, 0, 90, 180], margin: { l: 58 }, font: 17 });
  const fS = plotFigure(figs, "score over the start m");
  const scP = new Plot(fS, { width: 300, height: 230, x: [0, 64], y: [0, 1], xlabel: "start m", margin: { l: 52 }, font: 17 });
  const panel = htmlEl("div", "qb-panel", "", host);

  // ---- controls ----
  htmlEl("div", "fd-lab", "presets", panel);
  const presetBtns = presets(panel, LIST.map(p => ({ label: p.label, apply: () => load(p) })));
  const selRow = htmlEl("div", "fd-row fd-sel", "", panel);
  const aIds: ShapeId[] = ["leaf", "hand", "nucleus", "ellipse", "square", "shellA"];
  const aSel = select(selRow, "A", aIds.map(id => [id, SHAPES[id].label.split(" (")[0]]), v => { st.mode = "pair"; st.a = v as ShapeId; run(); });
  const bIds: (ShapeId | "same")[] = ["same", "leaf", "hand", "nucleus", "shellB"];
  const bSel = select(selRow, "B", bIds.map(id => [id, id === "same" ? "copy of A" : SHAPES[id].label.split(" (")[0]]), v => { st.mode = "pair"; st.b = v as ShapeId | "same"; run(); });
  const nSel = select(selRow, "N", [4, 8, 16, 32, 64, 128].map(n => [String(n), String(n)]), v => { st.n = Number(v); run(); });
  const kS = slider(panel, "K", 1, 64, 1, st.K, v => { st.K = v; run(); });
  const angS = slider(panel, "angle α", -180, 180, 5, st.angle, v => { st.angle = v; run(); }, v => v + "°");
  const scS = slider(panel, "scale s", 0.5, 2, 0.05, st.scale, v => { st.scale = v; run(); });
  const mS = slider(panel, "start m", 0, 127, 1, st.start, v => { st.start = Math.min(v, st.n - 1); run(); });
  const noS = slider(panel, "noise", 0, 0.05, 0.005, st.noise, v => { st.noise = v; run(); }, v => (v ? v.toFixed(3) : "off"));
  const pxS = slider(panel, "pixel size", 6, 30, 1, st.px, v => { st.px = v; run(); }, v => v + " px");
  const nucA = slider(panel, "nucleus A", 1, NUCLEI.length, 1, st.nucA, v => { st.nucA = v; run(); });
  const nucB = slider(panel, "nucleus B", 1, NUCLEI.length, 1, st.nucB, v => { st.nucB = v; run(); });
  const row1 = htmlEl("div", "fd-row", "", panel);
  const pixC = checkbox(row1, "pixel mask, traced", st.pixels, v => { st.pixels = v; run(); });
  const arcC = checkbox(row1, "arc-length resampling", st.arclen, v => { st.arclen = v; run(); });
  const mirC = checkbox(row1, "mirror B", st.mirrorB, v => { st.mirrorB = v; run(); });
  const allC = checkbox(row1, "allow mirror", st.allowMirror, v => { st.allowMirror = v; run(); });
  const magC = checkbox(row1, "magnitudes only", st.magOnly, v => { st.magOnly = v; run(); });
  const undoC = checkbox(row1, "phase: undo best start", false, v => { undo = v; draw(); });
  const aniC = checkbox(row1, "epicycles", true, v => { animate = v; if (v) requestAnimationFrame(tick); else drawEpi(); });
  button(panel, "reset", () => load(p0));
  const ro = htmlEl("div", "qb-readout", "", panel);
  hideControls(panel, host.dataset.controls);
  htmlEl("div", "qb-note", "Angles are clockwise on screen. Big dot: sample 0, with an arrow towards sample 1. Score: |q(m)| / √(Σ|c^A_k|² Σ|c^B_k|²) over k ≠ 0, 1 for a perfect fit; misfit √(1 − score²): the rms distance left after the best fit, relative to the size of B. Nuclei: synthetic kidney-shaped outlines, rasterised to pixel masks and traced.", panel);

  // ---- drawing helpers ----
  let fitA = (p: Pt): Pt => p, fitB = (p: Pt): Pt => p;
  function fitter(centre: Pt, scale: number): (p: Pt) => Pt {
    return p => [S / 2 + scale * (p[0] - centre[0]), S / 2 + scale * (p[1] - centre[1])];
  }
  function radius(pts: Pt[], c: Pt): number {
    let r = 0;
    for (const p of pts) r = Math.max(r, Math.hypot(p[0] - c[0], p[1] - c[1]));
    return r || 1;
  }
  function maskPath(m: { data: Uint8Array; w: number } | undefined, f: (p: Pt) => Pt): string {
    if (!m) return "";
    const parts: string[] = [];
    const [x0, y0] = f([0, 0]), [x1] = f([1, 0]), s = x1 - x0;
    for (let i = 0; i < m.data.length; i++) if (m.data[i]) {
      const x = i % m.w, y = (i - x) / m.w;
      parts.push(`M${(x0 + (x - 0.5) * s).toFixed(1)} ${(y0 + (y - 0.5) * s).toFixed(1)}h${s.toFixed(2)}v${s.toFixed(2)}h${(-s).toFixed(2)}z`);
    }
    return parts.join("");
  }
  function sampleDots(g: SVGGElement, z: Pt[], f: (p: Pt) => Pt, col: string, small: boolean): void {
    const r = small ? 2 : 3.2;
    if (z.length <= 256) for (let i = 1; i < z.length; i++) { const [x, y] = f(z[i]); svgEl("circle", { cx: x, cy: y, r, fill: col }, g); }
    // sample 0 and an arrow towards sample 1
    const [x, y] = f(z[0]), [x1, y1] = f(z[1 % z.length]), d = Math.hypot(x1 - x, y1 - y) || 1;
    const L = 16, ex = x + L * (x1 - x) / d, ey = y + L * (y1 - y) / d;
    svgEl("line", { x1: x, y1: y, x2: ex, y2: ey, stroke: "#000", "stroke-width": 2.2 }, g);
    const ang = Math.atan2(ey - y, ex - x), hx = (a: number) => ex - 7 * Math.cos(ang + a), hy = (a: number) => ey - 7 * Math.sin(ang + a);
    svgEl("path", { d: `M${ex} ${ey}L${hx(0.45)} ${hy(0.45)}L${hx(-0.45)} ${hy(-0.45)}Z`, fill: "#000" }, g);
    svgEl("circle", { cx: x, cy: y, r: 6, fill: col, stroke: "#000", "stroke-width": 2 }, g);
  }

  function stems(p: Plot, xs: number[], ys: number[], col: string, base: number): void {
    p.mark(g => {
      xs.forEach((x, i) => {
        if (!Number.isFinite(ys[i])) return;
        const X = p.x(x), Y = p.y(ys[i]), B = p.y(base);
        svgEl("line", { x1: X, x2: X, y1: B, y2: Y, stroke: col, "stroke-width": 2 }, g);
        svgEl("circle", { cx: X, cy: Y, r: 4, fill: col }, g);
      });
    }, []);
  }

  function drawEpi(): void {
    if (!res) return;
    const K = effK();
    for (const [pn, c, f, col] of [[pA, res.cA, fitA, BLUE], [pB, res.cB, fitB, RED]] as [OutPanel, Coeffs, (p: Pt) => Pt, string][]) {
      pn.epi.replaceChildren();
      if (!animate) continue;
      const chain = epicycles(c, K, t), sc = f([1, 0])[0] - f([0, 0])[0];
      for (const e of chain) {
        if (e.k === 0) continue;
        const [cx, cy] = f(e.centre), [tx, ty] = f(e.tip);
        if (e.r * sc > 1.5) svgEl("circle", { cx, cy, r: e.r * sc, fill: "none", stroke: "#555", "stroke-width": 1, opacity: 0.6 }, pn.epi);
        svgEl("line", { x1: cx, y1: cy, x2: tx, y2: ty, stroke: "#333", "stroke-width": 1.5 }, pn.epi);
      }
      const [px, py] = f(evalSeries(c, K, t));
      svgEl("circle", { cx: px, cy: py, r: 5, fill: col, stroke: "#fff", "stroke-width": 1.5 }, pn.epi);
    }
  }

  const effK = () => Math.max(1, Math.min(st.K, Math.floor(st.n / 2)));

  function draw(): void {
    if (!res) return;
    const r = res, K = effK(), n = st.n;
    // common scale for A and B
    const cA = sampleMean(r.A.z), cB = sampleMean(r.B.z);
    const R = Math.max(radius(r.A.path, cA), radius(r.B.path, cB), radius(r.A.z, cA), radius(r.B.z, cB));
    const sc = 0.4 * S / R;
    fitA = fitter(cA, sc); fitB = fitter(cB, sc);
    for (const [pn, s, c, f, col] of [[pA, r.A, r.cA, fitA, BLUE], [pB, r.B, r.cB, fitB, RED]] as const) {
      pn.mask.setAttribute("d", maskPath(s.mask, f));
      pn.path.setAttribute("d", pathD(s.path.map(f)));
      const M = Math.max(256, 8 * n);
      pn.rec.setAttribute("d", pathD(Array.from({ length: M }, (_, i) => f(evalSeries(c, K, i / M)))));
      pn.dots.replaceChildren();
      sampleDots(pn.dots, s.z, f, col, n > 32);
      pn.extra.setAttribute("d", "");
    }
    // overlay: B and A moved onto it
    const m = r.match, rB = radius(r.B.z, cB), fo = fitter(cB, 0.4 * S / Math.max(rB, radius(m.aligned, cB)));
    pO.mask.setAttribute("d", ""); pO.path.setAttribute("d", "");
    pO.rec.setAttribute("d", st.magOnly ? "" : pathD(r.B.z.map(fo)));
    pO.extra.setAttribute("d", st.magOnly ? "" : pathD(m.aligned.map(fo)));
    pO.dots.replaceChildren(); pO.epi.replaceChildren();
    if (!st.magOnly) { sampleDots(pO.dots, m.aligned, fo, BLUE, true); sampleDots(pO.dots, r.B.z, fo, RED, true); }
    else { const t0 = svgEl("text", { x: S / 2, y: S / 2, "text-anchor": "middle", "font-size": 16 }, pO.svg); t0.textContent = "magnitudes only: no pose"; pO.dots.appendChild(t0); }

    // magnitudes and phases
    const ks: number[] = [];
    for (let k = -K; k <= K; k++) if (k !== 0) ks.push(k);
    const big = (c: Coeffs) => Math.max(...ks.map(k => Math.hypot(...coef(c, k)))) || 1;
    const bA = big(r.cA), bB = big(r.cB);
    const magA = ks.map(k => Math.max(1e-4, Math.hypot(...coef(r.cA, k)) / bA)), magB = ks.map(k => Math.max(1e-4, Math.hypot(...coef(r.cB, k)) / bB));
    magP.clear(); phP.clear(); scP.clear();
    magP.setDomain([-K - 0.6, K + 0.6]); phP.setDomain([-K - 0.6, K + 0.6]);
    stems(magP, ks.map(k => k - 0.15), magA, BLUE, 1e-4);
    stems(magP, ks.map(k => k + 0.15), magB, RED, 1e-4);
    const ph = ks.map(k => {
      const [a, b0] = coef(r.cA, k), [p, q] = coef(r.cB, k), b = undo && m.mirrored ? -b0 : b0;
      if (Math.hypot(a, b) < 1e-3 * bA || Math.hypot(p, q) < 1e-3 * bB) return NaN;
      let d = Math.atan2(a * q - b * p, a * p + b * q);           // arg(c_B · conj(c_A))
      if (undo) { d -= 2 * Math.PI * k * m.m / n; d = Math.atan2(Math.sin(d), Math.cos(d)); }
      return d * DEG;
    });
    phP.hline(0, { color: GREY, width: 1 });
    stems(phP, ks, ph, "#6a3d9a", 0);
    // scores over m
    let lo = Math.min(...Array.from(m.scores), ...(m.scoresMirror ? Array.from(m.scoresMirror) : [1]));
    lo = Math.max(0, Math.floor((lo - 0.02) * 20) / 20);
    scP.setDomain([0, n - 1 || 1], [lo, 1]);
    const ms = Array.from({ length: n }, (_, i) => i);
    if (m.scoresMirror) scP.line(ms, m.scoresMirror, { color: GREY, width: 2, dash: "5 4", label: "mirror image of A" });
    scP.line(ms, m.scores, { color: "#6a3d9a", width: 2.5, label: "A" });
    if (n <= 64) scP.scatter(ms, m.scores, { color: "#6a3d9a", r: 2.5 });
    if (!st.magOnly) {
      const peak = m.mirrored ? m.scoresMirror![m.m] : m.scores[m.m];
      scP.scatter([m.m], [peak], { color: m.mirrored ? GREY : "#6a3d9a", r: 7, stroke: "#000" });
      if (st.mode === "pair" && !st.pixels) scP.vline(st.start, { color: "#2ca02c", width: 2, dash: "4 3" });
    }
    if (m.scoresMirror) scP.legend(undefined, "bl");

    // readout
    const lines: string[] = [];
    if (st.mode === "nuclei") lines.push(`nucleus ${st.nucA} (A) and nucleus ${st.nucB} (B), N = ${n}`);
    else lines.push(`N = ${n}, K = ${K}` + (st.pixels ? `, traced: ${r.A.path.length} / ${r.B.path.length} boundary pixels` : ""));
    const show = (k: number) => {
      const [a, b] = coef(r.cA, k), [p, q] = coef(r.cB, k);
      const c = (x: number, y: number) => `${f2(x, 3)}${y < 0 ? " − " : " + "}${f2(Math.abs(y), 3)}i`;
      return `c_${k < 0 ? "−" + -k : k}: A ${c(a, b)}  B ${c(p, q)}`;
    };
    if (n <= 8) for (const k of [1, -1]) lines.push(show(k));
    lines.push(`magnitudes: distance ${r.magDist < 1e-9 ? "0.000" : f2(r.magDist, 3)}` + (st.magOnly ? (r.magDist < 0.02 ? "  → same shape?" : "  → different") : ""));
    if (!st.magOnly) {
      lines.push(`best start m = ${m.m}${st.mode === "pair" && !st.pixels ? ` (true ${st.start})` : ""}`);
      lines.push(`angle ${f2(m.angle * DEG, 1)}°, scale ${f2(m.scale, 3)}${m.mirrored ? ", MIRRORED" : ""}`);
      lines.push(`score ${f2(m.score, 4)}, misfit ${f2(100 * misfit(m.score), 1)} %`);
    }
    if (r.pairs) {
      lines.push("misfit of every pair (%):");
      lines.push("     " + NUCLEI.map((_, j) => String(j + 1).padStart(5)).join(""));
      for (let i = 1; i <= NUCLEI.length; i++) {
        let row = String(i).padStart(3) + "  ";
        for (let j = 1; j <= NUCLEI.length; j++) {
          const pr = r.pairs.find(q => (q.i === i && q.j === j) || (q.i === j && q.j === i));
          row += (pr ? f2(100 * pr.misfit, 1) : "·").padStart(5);
        }
        lines.push(row);
      }
    }
    ro.textContent = lines.join("\n");
    drawEpi();
  }

  function syncSliders(): void {
    kS.set(st.K); angS.set(st.angle); scS.set(st.scale); mS.set(st.start); noS.set(st.noise); pxS.set(st.px);
    nucA.set(st.nucA); nucB.set(st.nucB);
    kS.input.max = String(Math.max(1, Math.floor(st.n / 2)));
    mS.input.max = String(st.n - 1);
    pixC.checked = st.pixels; arcC.checked = st.arclen; mirC.checked = st.mirrorB; allC.checked = st.allowMirror; magC.checked = st.magOnly;
    nSel.value = String(st.n);
    aSel.value = st.mode === "pair" ? st.a : ""; bSel.value = st.mode === "pair" ? st.b : "";
    undoC.checked = undo;
    const nuc = st.mode === "nuclei", off = (el: { input: HTMLInputElement }, v: boolean) => el.input.closest(".qb-slider")!.classList.toggle("fd-off", v);
    off(nucA, !nuc); off(nucB, !nuc);
    for (const el of [angS, scS, mS, noS]) off(el, nuc);
    off(pxS, !st.pixels || nuc);
    for (const c of [pixC, mirC]) c.closest(".qb-check")!.classList.toggle("fd-off", nuc);
  }

  function run(): void {
    if (st.mode === "nuclei") st.pixels = true;
    st.start = Math.min(st.start, st.n - 1);
    st.K = Math.min(st.K, Math.max(1, Math.floor(st.n / 2)));
    res = compute(st);
    syncSliders();
    draw();
  }

  function load(p: Preset): void {
    Object.assign(st, DEFAULTS, p.s);
    presetBtns.forEach((b, i) => b.classList.toggle("active", LIST[i] === p));
    t = 0;
    run();
  }

  function tick(now: number): void {
    if (!animate || !host.isConnected) return;
    const dt = last ? Math.min(0.1, (now - last) / 1000) : 0;
    last = now;
    t = (t + dt / 8) % 1;                // one turn in 8 s
    drawEpi();
    requestAnimationFrame(tick);
  }

  load(p0);
  aniC.checked = animate;
  requestAnimationFrame(tick);
}

/** A labelled drop-down list that gives the keyboard back after a choice. */
function select(parent: Element, label: string, options: [string, string][], onChange: (v: string) => void): HTMLSelectElement {
  const lab = htmlEl("label", "fd-selbox", "", parent);
  htmlEl("span", "", label + " ", lab);
  const sel = htmlEl("select", "", "", lab);
  for (const [v, t] of options) { const o = htmlEl("option", "", t, sel); o.value = v; }
  sel.addEventListener("change", () => { onChange(sel.value); sel.blur(); });
  return sel;
}

/** Show only the sliders and checkboxes whose names start with one of the given names. */
function hideControls(panel: HTMLElement, spec: string | undefined): void {
  if (!spec) return;
  const keep = spec.split(",").map(x => x.trim().toLowerCase());
  panel.querySelectorAll<HTMLElement>(".qb-slider, .qb-check").forEach(el => {
    const name = (el.querySelector(".qb-slider-name")?.textContent ?? el.textContent ?? "").trim().toLowerCase();
    if (!keep.some(k => name.startsWith(k))) el.style.display = "none";
  });
}


autoMount(NAME, mount);
