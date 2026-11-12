// Point–line duality behind the Hough transform. Left: the image plane (x, y with the origin
// at the centre, y up) with up to 8 draggable points; click on empty space to add one.
// Right: parameter space. In the normal parameterisation every point is the sinusoid
// rho = x cos(theta) + y sin(theta); in slope–intercept form (y = m x + c) it is the straight
// line c = y - m x. Small grey dots: where the curves of two points cross (the line through
// that pair). When all points are collinear, the common crossing is ringed. Click anywhere in
// parameter space to draw that line in the image. "Accumulator" bins the votes with
// steps dtheta, drho and shows the counts behind the curves.
//
// <div data-demo="houghdual" data-preset="three|two|vertical|twolines|noisy|bins"
//      data-param="normal|slope" data-acc="0|1"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl, draggable, rng } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { inferno } from "../lib/colour";
import { sinusoid, lineThrough } from "../lib/hough";
import { lineTLS } from "../lib/ransac";
import { injectStyle } from "../houghlines/view";

const NAME = "houghdual";
const STYLE = `
[data-demo="${NAME}"] { gap: .8em; flex-wrap: nowrap; }
[data-demo="${NAME}"] .hd-fig { flex: 0 1 380px; min-width: 200px; margin: 0; }
[data-demo="${NAME}"] .hd-fig.par { flex: 0 1 470px; }
[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin: 0 0 .1em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 240px; max-width: 300px; gap: .4em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre-wrap; }
[data-demo="${NAME}"] .hd-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .hd-fig { flex-basis: 360px; }
.reveal [data-demo="${NAME}"] .hd-fig.par { flex-basis: 450px; }
.reveal [data-demo="${NAME}"] .qb-panel { gap: .25em; flex-basis: 270px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .1em .45em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; padding: .25em .45em; }
@media (max-width: 860px) { [data-demo="${NAME}"] { flex-wrap: wrap; } }
`;

type Param = "normal" | "slope";
interface State { pts: [number, number][]; param: Param; acc: boolean; dTheta: number; dRho: number; sel: [number, number] | null }
interface Preset { id: string; label: string; s: Partial<State> }

const noisy = (): [number, number][] => {
  const r = rng(7), out: [number, number][] = [];
  for (let i = 0; i < 6; i++) { const t = -3.5 + 1.4 * i; out.push([+(t * 0.8 + 0.5 + 0.18 * r.norm()).toFixed(2), +(t * 0.6 - 1 + 0.18 * r.norm()).toFixed(2)]); }
  return out;
};
const PRESETS: Preset[] = [
  { id: "three", label: "three collinear points", s: { pts: [[2, 0], [1, 1], [0, 2]] } },
  { id: "two", label: "two points", s: { pts: [[-3, -1], [2, 2]] } },
  { id: "vertical", label: "towards vertical", s: { pts: [[0.6, -3], [1, 0], [1.4, 3]], param: "slope" } },
  { id: "twolines", label: "two lines", s: { pts: [[-4, -2], [-1, -1], [2, 0], [-3, 3], [0, 1], [3, -1]] } },
  { id: "noisy", label: "noisy collinear", s: { pts: noisy() } },
  { id: "bins", label: "coarse bins", s: { pts: noisy(), acc: true, dTheta: 15, dRho: 1.5 } },
];
const DEFAULT: State = { pts: [[2, 0], [1, 1], [0, 2]], param: "normal", acc: false, dTheta: 10, dRho: 1, sel: null };
const L = 5, RMAX = 7.5, MMAX = 4, CMAX = 10;
const deg = (r: number) => r * 180 / Math.PI;

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => {
    const s: State = { ...DEFAULT, ...(p0?.s ?? {}) };
    if (host.dataset.param === "slope") s.param = "slope";
    if (host.dataset.acc === "1") s.acc = true;
    s.pts = s.pts.map(p => [p[0], p[1]] as [number, number]);
    return s;
  };
  const st = start();

  // ---- layout ----
  const imgFig = htmlEl("figure", "hd-fig", "", host);
  htmlEl("figcaption", "", "image plane: drag points, click to add", imgFig);
  const img = new Plot(imgFig, { width: 400, height: 400, x: [-L, L], y: [-L, L], equal: true, xlabel: "x", ylabel: "y", margin: { l: 58, b: 58, r: 10, t: 10 }, xticks: [-4, -2, 0, 2, 4], yticks: [-4, -2, 0, 2, 4] });
  const parFig = htmlEl("figure", "hd-fig par", "", host);
  const parCap = htmlEl("figcaption", "", "", parFig);
  const par = new Plot(parFig, { width: 480, height: 400, x: [0, 180], y: [-RMAX, RMAX], xlabel: "θ (degrees)", ylabel: "ρ", margin: { l: 62, b: 58, r: 12, t: 10 }, xticks: [0, 45, 90, 135, 180] });
  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "hd-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "hd-lab", "parameter space", panel);
  const parBtns = presets(panel, [
    { label: "normal (θ, ρ)", apply: () => { st.param = "normal"; st.sel = null; draw(); } },
    { label: "slope–intercept (m, c)", apply: () => { st.param = "slope"; st.sel = null; draw(); } },
  ]);
  const accC = checkbox(panel, "accumulator (binned votes)", st.acc, v => { st.acc = v; draw(); });
  const dtS = slider(panel, "Δθ", 1, 30, 1, st.dTheta, v => { st.dTheta = v; draw(); }, v => v + "°");
  const drS = slider(panel, "Δρ", 0.25, 3, 0.25, st.dRho, v => { st.dRho = v; draw(); }, v => v.toFixed(2));
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "remove last", () => { st.pts.pop(); st.sel = null; draw(); });
  button(row, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  function load(p: Preset): void {
    Object.assign(st, DEFAULT, p.s);
    st.pts = st.pts.map(q => [q[0], q[1]] as [number, number]);
    sync(); draw();
  }
  function sync(): void { accC.checked = st.acc; dtS.set(st.dTheta); drS.set(st.dRho); }

  // ---- interaction ----
  img.svg.style.touchAction = "none";
  img.svg.addEventListener("pointerdown", ev => {
    if ((ev.target as Element).classList.contains("hd-handle")) return;
    if (st.pts.length >= 8) return;
    const [x, y] = img.toData(ev);
    if (Math.abs(x) > L || Math.abs(y) > L) return;
    st.pts.push([+x.toFixed(2), +y.toFixed(2)]);
    draw();
  });
  par.svg.addEventListener("pointerdown", ev => {
    const [a, b] = par.toData(ev);
    st.sel = [a, b];
    draw();
  });
  par.svg.style.cursor = "crosshair";

  // persistent handles (marks are redrawn, handles must survive a drag)
  const handleLayer = svgEl("g", {}, img.svg);
  let handles: SVGCircleElement[] = [];
  function syncHandles(): void {
    while (handles.length > st.pts.length) handles.pop()!.remove();
    while (handles.length < st.pts.length) {
      const k = handles.length;
      const h = svgEl("circle", { r: 11, class: "hd-handle" }, handleLayer);
      h.style.strokeWidth = "3"; h.style.fill = "#fff";
      draggable(h, ev => img.toData(ev), (x, y) => {
        st.pts[k] = [+Math.max(-L, Math.min(L, x)).toFixed(2), +Math.max(-L, Math.min(L, y)).toFixed(2)];
        draw();
      });
      h.addEventListener("contextmenu", e => { e.preventDefault(); st.pts.splice(k, 1); st.sel = null; draw(); });
      handles.push(h);
    }
    handles.forEach((h, k) => {
      h.setAttribute("cx", String(img.x(st.pts[k][0]))); h.setAttribute("cy", String(img.y(st.pts[k][1])));
      h.style.stroke = colour(k);
    });
  }
  const colour = (k: number) => seriesColor(k);

  // ---- drawing ----
  function draw(): void {
    parBtns.forEach((b, i) => b.classList.toggle("active", (i === 0) === (st.param === "normal")));
    img.clear(); par.clear();
    const normal = st.param === "normal";
    if (normal) { par.opts.xticks = [0, 45, 90, 135, 180]; par.setDomain([0, 180], [-RMAX, RMAX]); par.setLabels({ xlabel: "θ (degrees)", ylabel: "ρ" }); }
    else { par.opts.xticks = [-4, -2, 0, 2, 4]; par.setDomain([-MMAX, MMAX], [-CMAX, CMAX]); par.setLabels({ xlabel: "slope m", ylabel: "intercept c" }); }
    parCap.textContent = normal ? "parameter space: ρ = x cos θ + y sin θ" : "parameter space: c = y − m x";
    const n = st.pts.length;

    // accumulator behind the curves
    let accMax = 0, accBest: [number, number] | null = null;
    if (st.acc && normal && n) {
      const nT = Math.round(180 / st.dTheta), nR = Math.ceil(RMAX / st.dRho) * 2;
      const V = new Float64Array(nT * nR);
      for (const [x, y] of st.pts) for (let j = 0; j < nT; j++) {
        const th = (j + 0.5) * st.dTheta * Math.PI / 180, rho = sinusoid(x, y, th);
        const k = Math.floor(rho / st.dRho + nR / 2);
        if (k >= 0 && k < nR) V[k * nT + j]++;
      }
      for (let i = 0; i < V.length; i++) if (V[i] > accMax) { accMax = V[i]; accBest = [i % nT, Math.floor(i / nT)]; }
      par.heatmap(V, nT, nR, { map: t => inferno(0.12 + 0.88 * t), lo: 0, hi: Math.max(n, 2), x0: 0, x1: nT * st.dTheta, y0: -nR / 2 * st.dRho, y1: nR / 2 * st.dRho, flipY: false, numbers: nT * nR <= 300, gridLines: nT * nR <= 300 });
      if (accBest) {
        const [j, k] = accBest;
        par.mark(g => {
          const r = svgEl("rect", { x: par.x(j * st.dTheta), y: par.y((k + 1 - nR / 2) * st.dRho), width: par.x((j + 1) * st.dTheta) - par.x(j * st.dTheta), height: par.y((k - nR / 2) * st.dRho) - par.y((k + 1 - nR / 2) * st.dRho) }, g);
          r.style.fill = "none"; r.style.stroke = "#3ddc84"; r.style.strokeWidth = "3";
        }, []);
      }
    }

    // one curve per point
    st.pts.forEach(([x, y], k) => {
      if (normal) par.fn(t => sinusoid(x, y, t * Math.PI / 180), { color: colour(k), width: 3 }, 240);
      else par.fn(m => y - m * x, { color: colour(k), width: 3 }, 2);
    });
    // pairwise crossings
    const cross: [number, number][] = [];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const [x1, y1] = st.pts[i], [x2, y2] = st.pts[j];
      if (normal) { const l = lineThrough(x1, y1, x2, y2); cross.push([deg(l.theta), l.rho]); }
      else if (Math.abs(x2 - x1) > 1e-9) { const m = (y2 - y1) / (x2 - x1); cross.push([m, y1 - m * x1]); }
    }
    par.scatter(cross.map(c => c[0]), cross.map(c => c[1]), { color: "#444", r: 4 });

    // collinear?
    let common: { theta: number; rho: number } | null = null, resid = Infinity;
    if (n >= 2) {
      const fit = lineTLS(st.pts.map(([x, y]) => ({ x, y })))!;
      resid = Math.max(...st.pts.map(([x, y]) => Math.abs(sinusoid(x, y, fit.theta) - fit.rho)));
      if (resid < 0.03 || n === 2) common = fit;
    }
    if (common) {
      if (normal) ring(deg(common.theta), common.rho);
      else if (Math.abs(Math.sin(common.theta)) > 1e-6) {
        const m = -Math.cos(common.theta) / Math.sin(common.theta), c = common.rho / Math.sin(common.theta);
        ring(m, c);
      }
      drawLine(common.theta, common.rho, "#3ddc84", "", 4);
    }
    // selected parameter -> line in the image
    if (st.sel) {
      let th: number, rh: number;
      if (normal) { th = st.sel[0] * Math.PI / 180; rh = st.sel[1]; }
      else { const [m, c] = st.sel; th = Math.atan2(1, -m); rh = c * Math.sin(th); }
      drawLine(th, rh, "#e8402a", "7 5", 3);
      par.scatter([st.sel[0]], [st.sel[1]], { color: "#e8402a", r: 6 });
    }
    syncHandles();

    // readout
    const lines: string[] = [];
    st.pts.forEach(([x, y], k) => {
      const A = Math.hypot(x, y), ph = deg(Math.atan2(y, x));
      if (n <= 4) lines.push(normal ? `P${k + 1} (${x.toFixed(1)}, ${y.toFixed(1)}): ${A.toFixed(2)} cos(θ − ${ph.toFixed(0)}°)` : `P${k + 1} (${x.toFixed(1)}, ${y.toFixed(1)}): c = ${y.toFixed(1)} − ${x.toFixed(1)}·m`);
    });
    if (n >= 3) lines.push(common ? "all curves meet in one point" : `not collinear (off by ${resid.toFixed(2)})`);
    if (common) {
      if (normal) lines.push(`line: θ = ${deg(common.theta).toFixed(1)}°, ρ = ${common.rho.toFixed(2)}`);
      else {
        const s = Math.sin(common.theta);
        lines.push(Math.abs(s) < 1e-3 ? "line is vertical: m = ∞" : `line: m = ${(-Math.cos(common.theta) / s).toFixed(2)}, c = ${(common.rho / s).toFixed(2)}`);
      }
    }
    if (st.acc && normal && accMax) lines.push(`highest bin: ${accMax} of ${n} votes`);
    if (st.acc && !normal) lines.push("(accumulator: normal form only)");
    if (st.sel) lines.push(normal ? `red line: θ = ${st.sel[0].toFixed(0)}°, ρ = ${st.sel[1].toFixed(2)}` : `red line: m = ${st.sel[0].toFixed(2)}, c = ${st.sel[1].toFixed(2)}`);
    ro.textContent = lines.join("\n");
  }

  function ring(a: number, b: number): void {
    par.mark(g => {
      const c = svgEl("circle", { cx: par.x(a), cy: par.y(b), r: 13 }, g);
      c.style.fill = "none"; c.style.stroke = "#3ddc84"; c.style.strokeWidth = "4";
    }, []);
  }

  function drawLine(theta: number, rho: number, color: string, dash: string, width: number): void {
    // the visible part of x cos + y sin = rho inside the plot box
    const c = Math.cos(theta), s = Math.sin(theta), fx = rho * c, fy = rho * s, D = 3 * L;
    img.line([fx + D * s, fx - D * s], [fy - D * c, fy + D * c], { color, width, dash: dash || undefined });
  }

  draw();
}

autoMount(NAME, mount);
