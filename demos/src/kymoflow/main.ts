// Motion in 1D. A profile (the by-hand bump, a ramp, a smooth bump, moving spots, a line
// through the micrograph) moves over time. Top: frame 1 and frame 2, with Ix and It at a
// pixel you click. Middle: the per-pixel estimate u = −It/Ix (holes where Ix ≈ 0) and the
// window least-squares estimate −ΣIxIt/ΣIx². Right: the kymograph (x across, time down): a
// moving feature draws a line whose slope is its speed; drag the two handles to read it.
//
// <div data-demo="kymoflow" data-preset="byhand|ramp|bump|fast|bleach|cross|transport" data-mode="kymo"></div>
import { autoMount, htmlEl, slider, presets, button, svgEl, svgPoint, draggable } from "../lib/ui";
import { Grey, sampleBilinear } from "../lib/imageops";
import { flow1d, windowLS1d } from "../lib/flow";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { rng } from "../lib/random";
import { Plot, seriesColor } from "../lib/plot";
import { injectCss } from "../lkflow/flowui";

const NAME = "kymoflow", S = `.qb-demo[data-demo="${NAME}"]`;
const CSS = `
${S} { gap: .9em; flex-wrap: wrap; }
${S} .ky-plots { flex: 0 1 440px; min-width: 260px; display: flex; flex-direction: column; gap: .2em; }
${S} .ky-kymo { flex: 0 1 290px; min-width: 180px; }
${S} .ky-kymo .ky-wrap { position: relative; }
${S} .ky-kymo canvas { width: 100%; height: 100%; display: block; image-rendering: pixelated; border-radius: 3px; object-fit: fill; max-height: none; }
${S} .ky-kymo svg.ky-over { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; }
${S} .ky-cap { font-size: 17px; text-align: center; margin-bottom: .15em; }
${S} .qb-panel { flex: 1 1 280px; max-width: 340px; gap: .3em; }
${S} .qb-readout { font-size: 15px; }
${S} .ky-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
${S} .ky-credit { font-size: 12px; color: var(--qb-muted); }
/* pages: plots | kymograph | panel in one row of ~1075 px, compact panel */
${S} .ky-plots { flex: 0 1 400px; max-width: calc((100vh - 230px) * 0.92); }
${S} .ky-kymo { flex: 0 1 250px; max-width: calc((100vh - 260px) * 0.83); }
${S} .qb-panel { flex: 1 1 260px; gap: .2em; font-size: 15px; }
${S} .qb-panel .qb-btn { font-size: 13.5px; padding: .1em .45em; }
${S} .qb-panel .qb-presets { gap: .25em; }
${S} .qb-panel .qb-slider { gap: .3em; }
${S} .qb-panel .qb-readout { font-size: 13.5px; padding: .3em .5em; }
.reveal ${S} .ky-plots, .reveal ${S} .ky-kymo { max-width: none; }
.reveal ${S} .ky-kymo canvas { max-height: none; }
.reveal ${S} .qb-panel { font-size: inherit; }
.reveal ${S} .qb-panel .qb-presets { gap: .35em; }
.reveal ${S} { font-size: 18px; flex-wrap: nowrap; }
.reveal ${S} .ky-plots { flex: 0 0 400px; }
.reveal ${S} .ky-kymo { flex: 0 0 230px; }
.reveal ${S} .ky-cap { font-size: 18px; }
.reveal ${S} .qb-panel { flex: 0 0 450px; max-width: 450px; gap: .12em; }
.reveal ${S} .qb-btn, .reveal ${S} .qb-panel .qb-btn { font-size: 16px; padding: .06em .38em; }
.reveal ${S} .qb-readout, .reveal ${S} .qb-panel .qb-readout { font-size: 18px; padding: .2em .4em; }
.reveal ${S} .ky-lab { font-size: 16px; }
.reveal ${S} .ky-credit { display: none; }
@media (max-width: 860px) { ${S} { flex-wrap: wrap; } }
`;

type Profile = "byhand" | "ramp" | "bump" | "spots2" | "transport" | "cell";
const PROFILES: [Profile, string][] = [["byhand", "by-hand bump"], ["ramp", "ramp"], ["bump", "smooth bump"], ["spots2", "two spots"], ["transport", "vesicles"], ["cell", "micrograph line"]];
interface State { profile: Profile; speed: number; noise: number; win: number; bleach: number; dt: number; px: number }
const DEFAULT: State = { profile: "bump", speed: 1, noise: 0, win: 3, bleach: 0, dt: 1, px: 36 };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "byhand", label: "the by-hand bump", s: { profile: "byhand", speed: 1, win: 2, px: 3 } },
  { id: "ramp", label: "a ramp: exact everywhere", s: { profile: "ramp", speed: 1.5, px: 60 } },
  { id: "bump", label: "a bump: pixels disagree, the window is right", s: { profile: "bump", speed: 1, win: 6, px: 36 } },
  { id: "fast", label: "too fast: the estimate collapses", s: { profile: "bump", speed: 6, win: 6, px: 36 } },
  { id: "bleach", label: "bleaching breaks constancy", s: { profile: "bump", speed: 1, win: 6, bleach: 0.1, px: 36 } },
  { id: "cross", label: "two spots, two speeds", s: { profile: "spots2", speed: 1, win: 3, px: 30 } },
  { id: "transport", label: "transport: read the slopes", s: { profile: "transport", speed: 1, noise: 4, win: 3, px: 30 } },
];
const T = 40;   // frames in the kymograph

function mount(host: HTMLElement): void {
  injectCss(NAME, CSS);
  const kymoOnly = host.dataset.mode === "kymo";
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(p0?.s ?? {}) });
  const st = start();

  const plots = htmlEl("div", "ky-plots", "", host);
  const pf = new Plot(plots, { width: 440, height: 210, x: [0, 127], y: [0, 255], xlabel: "", ylabel: "grey value", font: 18, margin: { l: 62, r: 12, t: 10, b: 30 } });
  const f1Line = pf.line([], [], { color: seriesColor(0), width: 3, label: "frame 1" });
  const f2Line = pf.line([], [], { color: seriesColor(1), width: 3, dash: "7 4", label: "frame 2" });
  const pxMark1 = pf.vline(0, { color: "#e0a800", width: 2.5 });
  pf.legend(undefined, "tr");
  const pu = new Plot(plots, { width: 440, height: 220, x: [0, 127], y: [-1, 3], xlabel: "x (pixels)", ylabel: "u (px/frame)", font: 18, margin: { l: 62, r: 12, t: 10, b: 58 } });
  const truth = pu.hline(1, { color: "var(--qb-c3)", width: 2, dash: "6 4" });
  const perPx = pu.scatter([], [], { color: seriesColor(0), r: 4, label: "−It/Ix per pixel" });
  const winLine = pu.line([], [], { color: seriesColor(1), width: 3, label: "window LS" });
  const pxMark2 = pu.vline(0, { color: "#e0a800", width: 2.5 });
  pu.legend(undefined, "tr");
  for (const p of [pf, pu]) {
    p.svg.style.touchAction = "none"; p.svg.style.cursor = "crosshair";
    let down = false;
    const pick = (ev: PointerEvent) => { const [x] = p.toData(ev); st.px = Math.max(1, Math.min(L - 2, Math.round(x))); draw(); };
    p.svg.addEventListener("pointerdown", ev => { ev.preventDefault(); down = true; p.svg.setPointerCapture(ev.pointerId); pick(ev); });
    p.svg.addEventListener("pointermove", ev => { if (down) pick(ev); });
    p.svg.addEventListener("pointerup", () => { down = false; });
  }

  const ky = htmlEl("div", "ky-kymo", "", host);
  htmlEl("div", "ky-cap", "kymograph: x →, time ↓", ky);
  const kwrap = htmlEl("div", "ky-wrap", "", ky);
  const kc = htmlEl("canvas", "", "", kwrap);
  const kover = svgEl("svg", { class: "ky-over", preserveAspectRatio: "none" }, kwrap);
  const credit = htmlEl("div", "ky-credit", CELL_CREDIT, ky);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ky-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); rebuild(); } })));
  htmlEl("div", "ky-lab", "profile", panel);
  const profBtns = presets(panel, PROFILES.map(([k, l]) => ({ label: l, apply: () => { st.profile = k; st.px = k === "byhand" ? 3 : 36; rebuild(); } })));
  const spS = slider(panel, "speed (px/frame)", -6, 6, 0.25, st.speed, v => { st.speed = v; rebuild(); });
  const noS = slider(panel, "noise", 0, 20, 1, st.noise, v => { st.noise = v; rebuild(); });
  const wS = slider(panel, "window half-width", 0, 15, 1, st.win, v => { st.win = v; draw(); });
  const bS = slider(panel, "bleaching per frame", 0, 0.3, 0.01, st.bleach, v => { st.bleach = v; rebuild(); });
  const dS = slider(panel, "frames apart", 1, 4, 1, st.dt, v => { st.dt = v; draw(); });
  button(panel, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); rebuild(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  function sync(): void { spS.set(st.speed); noS.set(st.noise); wS.set(st.win); bS.set(st.bleach); dS.set(st.dt); }
  if (kymoOnly) { plots.style.display = "none"; }

  // ---- profiles ----
  let cellRow: Grey | null = null;
  loadCell().then(g => { cellRow = g; if (st.profile === "cell") rebuild(); });
  let L = 128, frames: Float64Array[] = [], speeds: number[] = [];
  const gauss = (x: number, c: number, s: number, a: number) => a * Math.exp(-((x - c) ** 2) / (2 * s * s));
  function profileAt(x: number, t: number): number {
    const s = st.speed;
    switch (st.profile) {
      case "byhand": { const c = 3 + s * t; const k = Math.round(x - c); return Math.abs(x - c - k) < 1e-9 ? ([1, 4, 1][k + 1] ?? 0) : bump3(x - c); }
      case "ramp": return 128 + 1.6 * (x - s * t - 64);
      case "bump": return 30 + gauss(x - s * t, 36, 4, 160);
      case "spots2": return 30 + gauss(x - s * t, 30, 2.5, 140) + gauss(x + 0.75 * s * t, 98, 2.5, 140);
      case "transport": {
        let v = 30;
        for (const [c, k] of [[10, 1.5], [40, 0.5], [70, 2.5], [110, -1]]) v += gauss(x - k * s * t, c, 2.2, 120);
        return v;
      }
      case "cell": return cellRow ? sampleBilinear(cellRow, 64 + x - s * t, 150, "replicate") : 128;
    }
  }
  /** Linear interpolation of the by-hand samples between integer positions. */
  function bump3(d: number): number { const a = Math.floor(d), f = d - a, at = (k: number) => [1, 4, 1][k + 1] ?? 0; return (1 - f) * at(a) + f * at(a + 1); }
  function rebuild(): void {
    L = st.profile === "byhand" ? 16 : 128;
    speeds = st.profile === "spots2" ? [st.speed, -0.75 * st.speed] : st.profile === "transport" ? [1.5, 0.5, 2.5, -1].map(k => k * st.speed) : [st.speed];
    const R = rng(3);
    frames = [];
    for (let t = 0; t < T; t++) {
      const f = new Float64Array(L), keep = (1 - st.bleach) ** t;
      for (let x = 0; x < L; x++) f[x] = keep * profileAt(x, t) + (st.noise ? st.noise * R.norm() : 0);
      frames.push(f);
    }
    drawKymo();
    draw();
  }

  // ---- kymograph with a slope ruler ----
  const handles: [number, number][] = [[0, 0], [0, 0]];
  const ruler = svgEl("line", { "stroke-width": 3, "vector-effect": "non-scaling-stroke" }, kover);
  ruler.style.stroke = "#ffd400";
  const hEls = handles.map(() => { const c = svgEl("ellipse", { rx: 1, ry: 1, class: "handle", "vector-effect": "non-scaling-stroke" }, kover); c.style.stroke = "#ffd400"; c.style.fill = "rgba(255,212,0,.35)"; c.style.strokeWidth = "3"; return c; });
  hEls.forEach((h, k) => draggable(h, ev => svgPoint(kover, ev), (x, y) => { handles[k] = [Math.max(0, Math.min(L, x)), Math.max(0, Math.min(T, y))]; drawRuler(); }));
  function resetRuler(): void {
    const s = speeds[0] ?? 0, x0 = st.profile === "byhand" ? 3.5 : st.profile === "ramp" ? 64 : st.profile === "spots2" || st.profile === "transport" ? 30.5 : 36.5;
    const t1 = Math.min(T, Math.max(4, Math.abs(s) > 0.01 ? Math.min(T, (s > 0 ? L - x0 - 2 : x0 - 2) / Math.abs(s)) : T));
    handles[0] = [x0, 0.5]; handles[1] = [x0 + s * (t1 - 0.5) * 0.6, 0.5 + (t1 - 0.5) * 0.6];
    if (st.profile === "transport") handles[1] = [x0 + 6, 20.5];
  }
  function drawKymo(): void {
    kc.width = L; kc.height = T;
    kwrap.style.aspectRatio = "5 / 6";
    kover.setAttribute("viewBox", `0 0 ${L} ${T}`);
    const ctx = kc.getContext("2d")!, id = ctx.createImageData(L, T);
    let lo = Infinity, hi = -Infinity;
    for (const f of frames) for (const v of f) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    for (let t = 0; t < T; t++) for (let x = 0; x < L; x++) {
      const g = 255 * (frames[t][x] - lo) / (hi - lo || 1), o = 4 * (t * L + x);
      id.data[o] = id.data[o + 1] = id.data[o + 2] = g; id.data[o + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    resetRuler(); drawRuler();
  }
  let rulerText = "";
  function drawRuler(): void {
    const [a, b] = handles, rx = L / 40;
    ruler.setAttribute("x1", String(a[0])); ruler.setAttribute("y1", String(a[1])); ruler.setAttribute("x2", String(b[0])); ruler.setAttribute("y2", String(b[1]));
    hEls.forEach((h, k) => { h.setAttribute("cx", String(handles[k][0])); h.setAttribute("cy", String(handles[k][1])); h.setAttribute("rx", String(1.6 * rx)); h.setAttribute("ry", String(1.6 * rx * (T / L) * (5 / 6))); });
    const dt = b[1] - a[1];
    rulerText = Math.abs(dt) > 0.05 ? `ruler slope: ${((b[0] - a[0]) / dt).toFixed(2)} px/frame` : "ruler: drag the handles";
    readout();
  }

  // ---- 1D flow ----
  let est = flow1d([], []), ls = windowLS1d([], [], 0);
  function draw(): void {
    profBtns.forEach((b, i) => b.classList.toggle("active", PROFILES[i][0] === st.profile));
    credit.style.display = st.profile === "cell" ? "" : "none";
    const f1 = frames[0], f2 = frames[Math.min(T - 1, st.dt)], xs = Array.from({ length: L }, (_, i) => i);
    let lo = Infinity, hi = -Infinity;
    for (const v of f1) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    for (const v of f2) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const pad = 0.08 * (hi - lo || 1);
    pf.setDomain([0, L - 1], [lo - pad, hi + pad]);
    f1Line.set(xs, f1); f2Line.set(xs, f2);
    let gmax = 0; for (let x = 1; x < L - 1; x++) gmax = Math.max(gmax, Math.abs(f1[x + 1] - f1[x - 1]) / 2);
    // hide estimates from slopes that are tiny compared with the profile's steepest slope (tails, noise)
    est = flow1d(f1, f2, st.profile === "byhand" ? 1e-9 : Math.max(0.05 * gmax, 0.5 * st.noise));
    ls = windowLS1d(est.ix, est.it, st.win, st.profile === "byhand" ? 1e-9 : 0.02 * gmax * gmax * (2 * st.win + 1));
    const u = Array.from(est.u, v => v / st.dt), w = Array.from(ls, v => v / st.dt);
    const sp = speeds.length ? speeds : [0];
    const umin = Math.min(-1, ...sp) - 0.5, umax = Math.max(1, ...sp) + 1;
    pu.setDomain([0, L - 1], [umin, umax]);
    perPx.set(xs, u.map(v => (v < umin || v > umax ? NaN : v)));
    winLine.set(xs, w);
    if (speeds.length === 1) truth.set(speeds[0]);
    truth.g.style.display = speeds.length === 1 ? "" : "none";
    pxMark1.set(st.px); pxMark2.set(st.px);
    readout();
  }
  function readout(): void {
    if (!est.ix.length) return;
    const x = st.px, fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : "undefined");
    ro.textContent = (kymoOnly ? "" : `pixel x = ${x}: Ix = ${est.ix[x].toFixed(2)}, It = ${est.it[x].toFixed(2)}\n−It/Ix = ${fmt(est.u[x] / st.dt)}, window LS = ${fmt(ls[x] / st.dt)}\n`) +
      `true speed${speeds.length > 1 ? "s" : ""}: ${speeds.map(s => s.toFixed(2)).join(", ")}\n${rulerText}`;
  }

  rebuild();
}

autoMount(NAME, mount);
