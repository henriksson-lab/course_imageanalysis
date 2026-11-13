// The aperture problem. A pattern (a bar, a grating, a corner, a texture, a flat patch, the
// micrograph) moves with a velocity you set by dragging the green dot in velocity space; you
// see it through a hole (circle, or a tall slit for the barber pole). Left: the moving pattern
// in the hole, with the motion each pixel can see (its normal flow, yellow). Right: velocity
// space (u, v): every pixel's constraint Ix u + Iy v + It = 0 is a line; the true velocity
// (green) lies on all of them, the least-squares point (red) is where the lines best meet.
//
// <div data-demo="aperture" data-preset="bar|barber|corner|texture|flat|noise"></div>
import { autoMount, htmlEl, slider, presets, button, checkbox, svgEl } from "../lib/ui";
import { Grey, sampleBilinear, cropGrey } from "../lib/imageops";
import { texture, solveLK } from "../lib/flow";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { rng } from "../lib/random";
import { Plot } from "../lib/plot";
import { injectCss } from "../lkflow/flowui";

const NAME = "aperture", S = `.qb-demo[data-demo="${NAME}"]`;
const CSS = `
${S} { gap: .9em; flex-wrap: wrap; align-items: flex-start; }
${S} .ap-left { flex: 0 1 330px; min-width: 200px; max-width: calc(100vh - 280px); }
${S} .ap-left canvas { width: 100%; aspect-ratio: 1; display: block; border-radius: 4px; }
${S} .ap-left .ap-wrap { position: relative; }
${S} .ap-left svg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
${S} .ap-cap { font-size: 17px; text-align: center; margin-bottom: .2em; }
${S} .ap-right { flex: 0 1 340px; min-width: 200px; max-width: calc(100vh - 250px); touch-action: none; }
${S} .ap-right svg { cursor: crosshair; }
${S} .qb-panel { flex: 0 1 300px; min-width: 240px; gap: .3em; font-size: 15px; }
${S} .qb-panel .qb-btn { font-size: 14px; padding: .15em .5em; }
${S} .qb-readout { font-size: 15px; }
${S} .ap-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
${S} .ap-credit { font-size: 12px; color: var(--qb-muted); }
.reveal ${S} { font-size: 18px; flex-wrap: nowrap; }
.reveal ${S} .ap-left { flex: 0 0 360px; max-width: none; }
.reveal ${S} .ap-right { flex: 0 0 380px; max-width: none; }
.reveal ${S} .ap-cap { font-size: 18px; }
.reveal ${S} .qb-panel { flex: 1 1 330px; gap: .2em; font-size: inherit; }
.reveal ${S} .qb-btn { font-size: 16px; padding: .06em .38em; }
.reveal ${S} .qb-readout { font-size: 18px; }
.reveal ${S} .ap-lab { font-size: 16px; }
.reveal ${S} .ap-credit { display: none; }
@media (max-width: 860px) { ${S} { flex-wrap: wrap; } }
`;

type Pattern = "bar" | "grating" | "corner" | "texture" | "flat" | "cell";
const PATTERNS: [Pattern, string][] = [["bar", "bar"], ["grating", "grating"], ["corner", "corner"], ["texture", "texture"], ["flat", "flat patch"], ["cell", "micrograph"]];
interface State { pattern: Pattern; u: number; v: number; radius: number; slit: boolean; noise: number; nohole: boolean; playing: boolean }
const DEFAULT: State = { pattern: "bar", u: 1.5, v: -1, radius: 24, slit: false, noise: 0, nohole: false, playing: true };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "bar", label: "bar through a hole: only the normal part", s: { pattern: "bar", u: 1.5, v: -1 } },
  { id: "barber", label: "the barber pole", s: { pattern: "grating", slit: true, u: 1.5, v: 0 } },
  { id: "corner", label: "a corner: the lines cross", s: { pattern: "corner", u: 1, v: 1 } },
  { id: "texture", label: "texture: many lines agree", s: { pattern: "texture", u: -1, v: -1.5 } },
  { id: "flat", label: "flat patch: no lines", s: { pattern: "flat", u: 1, v: 0.5 } },
  { id: "noise", label: "noise: lines scatter", s: { pattern: "corner", u: 1, v: 1, noise: 6 } },
];
const HALF = 48;          // pattern window: [−48, 48]² pixels
const LIM = 3;            // velocity space ±3 px/frame

function smooth(t: number): number { return 1 / (1 + Math.exp(-t)); }

function mount(host: HTMLElement): void {
  injectCss(NAME, CSS);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(p0?.s ?? {}) });
  const st = start();

  const left = htmlEl("div", "ap-left", "", host);
  const cap = htmlEl("div", "ap-cap", "", left);
  const wrap = htmlEl("div", "ap-wrap", "", left);
  const C = 192, canvas = htmlEl("canvas", "", "", wrap);
  canvas.width = C; canvas.height = C;
  const over = svgEl("svg", { viewBox: `${-HALF} ${-HALF} ${2 * HALF} ${2 * HALF}` }, wrap);
  const credit = htmlEl("div", "ap-credit", CELL_CREDIT, left);

  const right = htmlEl("div", "ap-right", "", host);
  htmlEl("div", "ap-cap", "velocity space: drag the green dot", right);
  const plot = new Plot(right, { width: 380, height: 380, x: [-LIM, LIM], y: [-LIM, LIM], equal: true, xlabel: "u (px/frame, right)", ylabel: "v (px/frame, down)",
    yfmt: v => (Math.abs(v) < 1e-9 ? "0" : String(-v).replace("-", "−")), font: 18, margin: { l: 64, r: 10, t: 8, b: 60 } });
  const lines = plot.mark<[[number, number, number][]]>((g, ls) => {
    for (const [a, b, c] of ls) {
      // a u + b v + c = 0, plotted with y = −v
      let pts: [number, number][];
      if (Math.abs(b) > Math.abs(a)) pts = [-LIM * 2, LIM * 2].map(u => [u, (a * u + c) / b] as [number, number]);
      else pts = [-LIM * 2, LIM * 2].map(y => [-(b * -y + c) / a, y] as [number, number]);
      const l = svgEl("line", { x1: plot.x(pts[0][0]), y1: plot.y(pts[0][1]), x2: plot.x(pts[1][0]), y2: plot.y(pts[1][1]) }, g);
      l.style.stroke = "var(--qb-c1)"; l.style.strokeWidth = "1.6"; l.style.opacity = "0.55";
    }
  }, [[]]);
  const nfDots = plot.scatter([], [], { color: "#e0a800", r: 4 });
  const lsDot = plot.scatter([0], [0], { color: "var(--qb-c2)", r: 8 });
  const trueDot = plot.scatter([0], [0], { color: "var(--qb-c3)", r: 11 });
  plot.legend([{ label: "true (drag)", color: "var(--qb-c3)", kind: "dot" }, { label: "least squares", color: "var(--qb-c2)", kind: "dot" }, { label: "normal flows", color: "#e0a800", kind: "dot" }], "tl");
  let dragging = false;
  const setV = (ev: PointerEvent) => {
    const [x, y] = plot.toData(ev);
    st.u = Math.round(Math.max(-LIM, Math.min(LIM, x)) * 20) / 20; st.v = Math.round(Math.max(-LIM, Math.min(LIM, -y)) * 20) / 20;
    solve(); drawPlot();
  };
  plot.svg.addEventListener("pointerdown", ev => { ev.preventDefault(); dragging = true; plot.svg.setPointerCapture(ev.pointerId); setV(ev); });
  plot.svg.addEventListener("pointermove", ev => { if (dragging) setV(ev); });
  plot.svg.addEventListener("pointerup", () => { dragging = false; });
  plot.svg.addEventListener("pointercancel", () => { dragging = false; });

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ap-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); solve(); drawPlot(); } })));
  htmlEl("div", "ap-lab", "pattern", panel);
  const patBtns = presets(panel, PATTERNS.map(([k, l]) => ({ label: l, apply: () => { st.pattern = k; solve(); drawPlot(); } })));
  const radS = slider(panel, "hole radius (px)", 4, 40, 1, st.radius, v => { st.radius = v; solve(); drawPlot(); });
  const noiseS = slider(panel, "noise", 0, 15, 1, st.noise, v => { st.noise = v; solve(); drawPlot(); });
  const slitC = checkbox(panel, "tall slit instead of a circle", st.slit, v => { st.slit = v; solve(); drawPlot(); });
  const holeC = checkbox(panel, "remove the hole (show everything)", st.nohole, v => { st.nohole = v; });
  const row = htmlEl("div", "qb-presets", "", panel);
  const playB = button(row, "pause", () => { st.playing = !st.playing; playB.textContent = st.playing ? "pause" : "play"; });
  button(row, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); solve(); drawPlot(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  function sync(): void { radS.set(st.radius); noiseS.set(st.noise); slitC.checked = st.slit; holeC.checked = st.nohole; playB.textContent = st.playing ? "pause" : "play"; }

  // ---- the pattern P(x, y), x, y in [−48, 48] ----
  const tex = texture(97, 97, 21, 2.5, 45);
  let cell: Grey | null = null;
  loadCell().then(g => { cell = cropGrey(g, 60, 70, 97, 97); if (st.pattern === "cell") { solve(); drawPlot(); } });
  function P(x: number, y: number): number {
    switch (st.pattern) {
      case "bar": { // bar of width 12 along the direction 60° (CCW on screen), smooth edges
        const nx = Math.sin(Math.PI / 3), ny = Math.cos(Math.PI / 3), d = x * nx + y * ny;
        return 50 + 150 * smooth((d + 6) / 1.2) * smooth((6 - d) / 1.2);
      }
      case "grating": { const d = (x - y) / Math.SQRT2; return 128 + 80 * Math.sin(2 * Math.PI * d / 14); }
      case "corner": return 50 + 150 * smooth((x + 4) / 1.2) * smooth((4 - y) / 1.2);
      case "texture": return sampleBilinear(tex, x + HALF, y + HALF, "wrap");
      case "flat": return 128;
      case "cell": return cell ? sampleBilinear(cell, x + HALF, y + HALF, "replicate") : 128;
    }
  }
  const inHole = (x: number, y: number) => st.slit ? Math.abs(x) <= 8 && Math.abs(y) <= 40 : x * x + y * y <= st.radius * st.radius;

  // ---- constraints of the pixels in the hole (frames at t = ∓1/2 around the drawn instant) ----
  let cons: [number, number, number][] = [], est: [number, number] = [0, 0], nflow: { x: number; y: number; u: number; v: number }[] = [];
  function solve(): void {
    const R = rng(5), I = (x: number, y: number, t: number) => P(x - st.u * t, y - st.v * t);
    cons = []; nflow = [];
    let a = 0, b = 0, c = 0, p = 0, q = 0;
    for (let y = -HALF + 2; y <= HALF - 2; y += 2) for (let x = -HALF + 2; x <= HALF - 2; x += 2) {
      if (!inHole(x, y)) continue;
      const n = () => st.noise * R.norm();
      // frame 1 at t = −1/2, frame 2 at t = +1/2; gradients from the mean of the two frames
      const f1 = [I(x - 1, y, -0.5) + n(), I(x + 1, y, -0.5) + n(), I(x, y - 1, -0.5) + n(), I(x, y + 1, -0.5) + n(), I(x, y, -0.5) + n()];
      const f2 = [I(x - 1, y, 0.5) + n(), I(x + 1, y, 0.5) + n(), I(x, y - 1, 0.5) + n(), I(x, y + 1, 0.5) + n(), I(x, y, 0.5) + n()];
      const Ix = ((f1[1] - f1[0]) + (f2[1] - f2[0])) / 4, Iy = ((f1[3] - f1[2]) + (f2[3] - f2[2])) / 4, It = f2[4] - f1[4];
      a += Ix * Ix; b += Ix * Iy; c += Iy * Iy; p += Ix * It; q += Iy * It;
      const g2 = Ix * Ix + Iy * Iy;
      if (g2 > 4) {
        if ((x + y) % 4 === 0 || st.pattern !== "texture") cons.push([Ix, Iy, It]);
        if (x % 8 === 0 && y % 8 === 0) nflow.push({ x, y, u: -It * Ix / g2, v: -It * Iy / g2 });
      }
    }
    const r = solveLK(a, b, c, p, q, 1e-3);
    est = [r[0], r[1]];
    const kind = r[2] < 1 ? "no gradients: nothing to see" : r[3] < 0.02 * r[2] ? "lines parallel: normal part only" : "lines cross: motion determined";
    ro.textContent = `true      (${st.u.toFixed(2)}, ${st.v.toFixed(2)})\nleast sq. (${est[0].toFixed(2)}, ${est[1].toFixed(2)})\n${cons.length} lines; ${kind}`;
    if (cons.length > 160) cons = cons.filter((_, i) => i % Math.ceil(cons.length / 160) === 0);
  }

  function drawPlot(): void {
    patBtns.forEach((bt, i) => bt.classList.toggle("active", PATTERNS[i][0] === st.pattern));
    credit.style.display = st.pattern === "cell" ? "" : "none";
    lines.set(cons);
    nfDots.set(nflow.map(n => n.u), nflow.map(n => -n.v));
    lsDot.set([est[0]], [-est[1]]);
    trueDot.set([st.u], [-st.v]);
  }

  // ---- animation of the pattern in the hole ----
  const ctx = canvas.getContext("2d")!, img = ctx.createImageData(C, C);
  let t = 0, last = 0, raf = 0;
  function frame(ts: number): void {
    if (!host.isConnected) return;
    raf = requestAnimationFrame(frame);
    if (ts - last < 33) return;
    last = ts;
    if (st.playing) t += 0.25;
    const T = 24, tt = (t % T) - T / 2;   // loop: the pattern sweeps back and forth through 24 frames
    const k = 2 * HALF / C, R = rng(Math.floor(t * 4) + 1);
    for (let j = 0; j < C; j++) for (let i = 0; i < C; i++) {
      const x = -HALF + (i + 0.5) * k, y = -HALF + (j + 0.5) * k;
      const inside = inHole(x, y);
      let val = P(x - st.u * tt, y - st.v * tt);
      if (st.noise) val += st.noise * R.norm();
      const o = 4 * (j * C + i);
      if (inside) { img.data[o] = img.data[o + 1] = img.data[o + 2] = val; }
      else if (st.nohole) { img.data[o] = img.data[o + 1] = img.data[o + 2] = 0.7 * val; }
      else { img.data[o] = 70; img.data[o + 1] = 74; img.data[o + 2] = 82; }
      img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    drawOverlay();
  }
  function drawOverlay(): void {
    over.replaceChildren();
    const arrow = (x: number, y: number, u: number, v: number, col: string, w: number) => {
      const s = 4, x1 = x + s * u, y1 = y + s * v, L = Math.hypot(x1 - x, y1 - y);
      if (L < 0.5) return;
      const ux = (x1 - x) / L, uy = (y1 - y) / L, hl = Math.min(3, L * 0.5);
      const l = svgEl("line", { x1: x, y1: y, x2: x1 - ux * hl * 0.6, y2: y1 - uy * hl * 0.6, "stroke-width": w, "stroke-linecap": "round" }, over);
      l.style.stroke = col;
      svgEl("polygon", { points: `${x1},${y1} ${x1 - ux * hl - uy * hl * 0.5},${y1 - uy * hl + ux * hl * 0.5} ${x1 - ux * hl + uy * hl * 0.5},${y1 - uy * hl - ux * hl * 0.5}`, fill: col }, over);
    };
    for (const n of nflow) arrow(n.x, n.y, n.u, n.v, "#ffd400", 1.1);
    if (st.nohole) arrow(0, 0, st.u, st.v, "#22c55e", 1.8);
    if (st.slit) svgEl("rect", { x: -8, y: -40, width: 16, height: 80, fill: "none", stroke: "#ffd400", "stroke-width": 0.6, "stroke-dasharray": "2 1.5" }, over);
    else svgEl("circle", { cx: 0, cy: 0, r: st.radius, fill: "none", stroke: "#ffd400", "stroke-width": 0.6, "stroke-dasharray": "2 1.5" }, over);
    cap.textContent = st.nohole ? "without the hole: the true motion (green)" : "through the hole: what each pixel sees (yellow)";
  }
  sync(); solve(); drawPlot();
  raf = requestAnimationFrame(frame);
  void raf;
}

autoMount(NAME, mount);
