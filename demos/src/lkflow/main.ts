// Lucas–Kanade optical flow on a frame pair with known motion. Frame 1 is a 128² crop of the
// micrograph (or a synthetic scene); frame 2 is frame 1 moved by a chosen flow (translation,
// rotation, zoom, shear, swirl, a disc moving over a still background) plus seeded noise.
// Tabs: the frames (and a flicker), the estimated flow (colour wheel or arrows), the
// confidence μ2 (smaller eigenvalue of J), the endpoint error against the truth, and the
// pyramid levels. Click a pixel: its window's constraint lines in velocity space, the
// least-squares point, the true velocity and the uncertainty ellipse (∝ J⁻¹).
//
// <div data-demo="lkflow" data-preset="small|large|pyramid|rotation|membrane|flat|boundary|noise"
//      data-tab="flow|arrows|conf|epe|frames|flicker|pyramid"></div>
import { autoMount, htmlEl, slider, presets, button } from "../lib/ui";
import { Grey, cropGrey } from "../lib/imageops";
import {
  FieldKind, makePair, FramePair, lkPyramid, LKResult, Flow, flowDerivatives, warpByFlow, endpointError, flowToRGBA,
  ringScene, cellScene, compositePair, maxSpeed,
} from "../lib/flow";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { Plot } from "../lib/plot";
import { viridis, inferno } from "../lib/colour";
import { ImgView } from "../canny/imgview";
import { injectCss, flowWheel, drawQuiver, WHEEL_CSS, fmtSpeed } from "./flowui";

const NAME = "lkflow", S = `.qb-demo[data-demo="${NAME}"]`;
const CSS = `
${S} { gap: .9em; flex-wrap: wrap; }
${S} .lk-main { flex: 0 1 400px; min-width: 240px; }
${S} .lk-mid { flex: 0 1 330px; min-width: 220px; display: flex; flex-direction: column; gap: .3em; }
${S} .qb-panel { flex: 1 1 300px; max-width: 380px; gap: .3em; }
${S} .lk-tabs { display: flex; flex-wrap: wrap; gap: .25em; margin-bottom: .3em; }
${S} .lk-tabs .qb-btn { padding: .12em .45em; }
${S} .qb-readout { font-size: 15px; }
${S} .lk-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
/* pages: image | velocity plot | panel in one row of ~1075 px, compact panel */
${S} .lk-main { flex: 0 1 370px; max-width: calc(100vh - 290px); }
${S} .lk-mid { flex: 0 1 290px; min-width: 200px; }
${S} .qb-panel { flex: 1 1 280px; max-width: 340px; gap: .2em; font-size: 15px; }
${S} .qb-panel .qb-btn, ${S} .lk-tabs .qb-btn { font-size: 13.5px; padding: .1em .45em; }
${S} .qb-panel .qb-presets { gap: .25em; }
${S} .qb-readout { font-size: 13.5px; padding: .3em .5em; }
.reveal ${S} .lk-main { max-width: none; }
.reveal ${S} .lk-mid { min-width: 220px; }
.reveal ${S} .qb-panel { max-width: 380px; font-size: inherit; }
.reveal ${S} .qb-panel .qb-presets { gap: .35em; }
.reveal ${S} .qb-panel .qb-btn, .reveal ${S} .lk-tabs .qb-btn { font-size: 16px; padding: .06em .38em; }
.reveal ${S} .qb-readout.qb-readout { font-size: 18px; padding: .2em .4em; }
.reveal ${S} { font-size: 18px; flex-wrap: nowrap; }
.reveal ${S} .lk-main { flex: 0 0 390px; }
.reveal ${S} .lk-mid { flex: 0 0 320px; }
.reveal ${S} .qb-panel { flex: 1 1 360px; gap: .15em; }
.reveal ${S} .qb-btn { font-size: 16px; padding: .06em .38em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .4em; }
.reveal ${S} .qb-slider { gap: .3em; }
.reveal ${S} .lk-lab { font-size: 16px; }
.reveal ${S} .ev-credit { display: none; }
@media (max-width: 860px) { ${S} { flex-wrap: wrap; } }
${WHEEL_CSS(S)}
`;

type Scene = "cell" | "ring" | "boundary";
type Tab = "frames" | "flicker" | "flow" | "arrows" | "conf" | "epe" | "pyramid";
const TABS: [Tab, string][] = [["frames", "frame 1"], ["flicker", "flicker 1↔2"], ["flow", "flow (colour)"], ["arrows", "arrows"], ["conf", "confidence μ₂"], ["epe", "error (EPE)"], ["pyramid", "pyramid"]];
const MOTIONS: [FieldKind, string][] = [["translation", "shift"], ["rotation", "rotation"], ["zoom", "zoom"], ["shear", "shear"], ["swirl", "swirl"], ["regions", "disc moves"]];
interface State { scene: Scene; motion: FieldKind; size: number; sigma: number; rho: number; noise: number; conf: number; iters: number; levels: number; tab: Tab; px: number; py: number }
const DEFAULT: State = { scene: "cell", motion: "translation", size: 1, sigma: 1, rho: 3, noise: 2, conf: 0, iters: 1, levels: 1, tab: "flow", px: 80, py: 40 };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "small", label: "small shift: works", s: { size: 1, tab: "epe" } },
  { id: "large", label: "8 px shift: fails", s: { size: 8, tab: "epe" } },
  { id: "pyramid", label: "the pyramid rescues it", s: { size: 8, levels: 4, iters: 3, tab: "pyramid" } },
  { id: "rotation", label: "rotation", s: { motion: "rotation", size: 3, levels: 2, iters: 2, tab: "arrows" } },
  { id: "membrane", label: "a membrane: aperture problem", s: { size: 1.5, motion: "translation", tab: "conf", px: 85, py: 114, rho: 2 } },
  { id: "flat", label: "flat: nothing to see", s: { scene: "ring", size: 1.5, tab: "flow", px: 64, py: 64, conf: 20 } },
  { id: "boundary", label: "a motion boundary", s: { scene: "boundary", motion: "regions", size: 1.5, tab: "epe", rho: 4, px: 98, py: 64 } },
  { id: "noise", label: "noise: enlarge the window", s: { noise: 12, rho: 1.5, tab: "epe" } },
];
const CONF_MAX = 400;
const confOf = (t: number) => (t <= 0 ? 0 : Math.round(CONF_MAX ** (t / 100) * 10) / 10);
const tOfConf = (c: number) => (c <= 0 ? 0 : 100 * Math.log(c) / Math.log(CONF_MAX));

function mount(host: HTMLElement): void {
  injectCss(NAME, CSS);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const tab0 = TABS.find(t => t[0] === host.dataset.tab)?.[0];
  const start = (): State => ({ ...DEFAULT, ...(p0?.s ?? {}), ...(tab0 ? { tab: tab0 } : {}) });
  const st = start();
  const N = 128;

  // ---- layout ----
  const main = htmlEl("div", "lk-main", "", host);
  const tabs = htmlEl("div", "lk-tabs", "", main);
  const tabBtns = TABS.map(([t, l]) => button(tabs, l, () => { st.tab = t; draw(); }));
  const view = new ImgView(main, "");
  const credit = htmlEl("div", "ev-credit", CELL_CREDIT, main);
  const mid = htmlEl("div", "lk-mid", "", host);
  htmlEl("div", "lk-lab", "velocity space of the clicked pixel's window", mid);
  const vplot = new Plot(mid, { width: 330, height: 300, x: [-3, 3], y: [-3, 3], equal: true, xlabel: "u (px/frame)", ylabel: "v (px/frame)", yfmt: v => (Math.abs(v) < 1e-9 ? "0" : (-v).toFixed(Math.abs(v) < 2 && v % 1 ? 1 : 0).replace("-", "−")), font: 18, margin: { l: 62, r: 10, t: 8, b: 60 } });
  const linesMark = vplot.mark<[[number, number, number, number, number][]]>((g, ls) => {
    for (const [a, b, c, op, wd] of ls) {
      // a u + b v + c = 0, with v drawn upward-negative: plot y = −v so "down" is down on screen
      const [x0, x1] = vplot.x.domain, pts: [number, number][] = [];
      if (Math.abs(b) > Math.abs(a)) { for (const x of [x0, x1]) pts.push([x, -(-(a * x + c) / b)]); }
      else { const [y0, y1] = vplot.y.domain; for (const y of [y0, y1]) { const v = -y; pts.push([-(b * v + c) / a, y]); } }
      const l = document.createElementNS("http://www.w3.org/2000/svg", "line");
      l.setAttribute("x1", String(vplot.x(pts[0][0]))); l.setAttribute("y1", String(vplot.y(pts[0][1])));
      l.setAttribute("x2", String(vplot.x(pts[1][0]))); l.setAttribute("y2", String(vplot.y(pts[1][1])));
      l.style.stroke = "var(--qb-c1)"; l.style.strokeWidth = String(wd); l.style.opacity = String(op);
      g.appendChild(l);
    }
  }, [[]]);
  const ell = vplot.ellipse([0, 0], [[0.01, 0], [0, 0.01]], 2, { color: "var(--qb-c2)", fill: "var(--qb-c2)", fillOpacity: 0.15, width: 2 });
  const truthDot = vplot.scatter([0], [0], { color: "var(--qb-c3)", r: 8 });
  const estDot = vplot.scatter([0], [0], { color: "var(--qb-c2)", r: 7 });
  vplot.legend([{ label: "true", color: "var(--qb-c3)", kind: "dot" }, { label: "estimate", color: "var(--qb-c2)", kind: "dot" }, { label: "constraints", color: "var(--qb-c1)" }], "tl");
  const pro = htmlEl("div", "qb-readout", "", mid);
  const wheel = flowWheel(mid, 72);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "lk-lab", "presets", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); sync(); rebuild(); } })));
  htmlEl("div", "lk-lab", "motion (frame 1 → frame 2)", panel);
  const motionBtns = presets(panel, MOTIONS.map(([k, l]) => ({ label: l, apply: () => { st.motion = k; if (k === "regions" && st.scene === "ring") st.scene = "cell"; rebuild(); } })));
  htmlEl("div", "lk-lab", "scene", panel);
  const sceneBtns = presets(panel, [
    { label: "micrograph", apply: () => { st.scene = "cell"; rebuild(); } },
    { label: "ring + flat", apply: () => { st.scene = "ring"; rebuild(); } },
    { label: "cell on substrate", apply: () => { st.scene = "boundary"; st.motion = "regions"; rebuild(); } },
  ]);
  const sizeS = slider(panel, "largest motion (px)", 0, 10, 0.25, st.size, v => { st.size = v; rebuild(); });
  const noiseS = slider(panel, "noise", 0, 20, 1, st.noise, v => { st.noise = v; rebuild(); });
  const sigS = slider(panel, "σ (gradient)", 0, 3, 0.25, st.sigma, v => { st.sigma = v; compute(); });
  const rhoS = slider(panel, "window ρ", 0.5, 10, 0.5, st.rho, v => { st.rho = v; compute(); });
  const itS = slider(panel, "warping iterations", 1, 5, 1, st.iters, v => { st.iters = v; compute(); });
  const levS = slider(panel, "pyramid levels", 1, 4, 1, st.levels, v => { st.levels = v; compute(); });
  const confS = slider(panel, "confidence ≥", 0, 100, 1, tOfConf(st.conf), t => { st.conf = confOf(t); draw(); }, t => String(confOf(t)));
  button(panel, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); rebuild(); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  function sync(): void {
    sizeS.set(st.size); noiseS.set(st.noise); sigS.set(st.sigma); rhoS.set(st.rho); itS.set(st.iters); levS.set(st.levels); confS.set(tOfConf(st.conf));
  }

  view.onPick((x, y) => { st.px = x; st.py = y; drawPixel(); drawOverlay(); });

  // ---- data ----
  let cell: Grey | null = null, pair: FramePair | null = null, res: (LKResult & { perLevel: Flow[] }) | null = null, ms = 0;
  let deriv: ReturnType<typeof flowDerivatives> | null = null;
  function base(): Grey | null {
    if (!cell) { loadCell().then(g => { cell = cropGrey(g, 64, 64, N, N); rebuild(); }); return null; }
    return cell;
  }
  let ringBase: Grey | null = null, sceneCache: ReturnType<typeof cellScene> | null = null;
  function rebuild(): void {
    const m = st.size, kind = st.motion;
    if (st.scene === "boundary") {
      sceneCache ??= cellScene(N, N, 5);
      pair = compositePair(sceneCache.bg, sceneCache.fg, sceneCache.r, m * Math.cos(Math.PI / 6), -m * Math.sin(Math.PI / 6), st.noise, 11);
    } else {
      const b = st.scene === "ring" ? (ringBase ??= ringScene(N, N, 3)) : base();
      if (!b) return;
      const p = kind === "translation" || kind === "regions" ? { s: m, angle: 30, r: 34 } : kind === "swirl" ? { s: m, r: 30 } : { s: m / 64 };
      pair = makePair(b, kind, p, st.noise, 11);
    }
    compute();
  }
  let pending = 0;
  function compute(): void {
    if (pending) return;
    pending = requestAnimationFrame(() => { pending = 0; computeNow(); });
  }
  function computeNow(): void {
    if (!pair) return;
    const t0 = performance.now();
    res = lkPyramid(pair.I1, pair.I2, { sigma: st.sigma, rho: st.rho, levels: st.levels, iters: st.iters });
    ms = performance.now() - t0;
    // constraint lines around the final estimate (finest level)
    deriv = flowDerivatives(pair.I1, warpByFlow(pair.I2, res.u, res.v), st.sigma);
    draw();
  }

  // ---- drawing ----
  let flick = 0, flickTimer = 0;
  function draw(): void {
    if (!pair || !res) return;
    tabBtns.forEach((b, i) => b.classList.toggle("active", TABS[i][0] === st.tab));
    motionBtns.forEach((b, i) => b.classList.toggle("active", MOTIONS[i][0] === st.motion));
    sceneBtns.forEach((b, i) => b.classList.toggle("active", ["cell", "ring", "boundary"][i] === st.scene));
    credit.style.display = st.scene === "cell" ? "" : "none";
    if (st.tab === "flicker") { if (!flickTimer) flickTimer = window.setInterval(() => { flick ^= 1; if (st.tab === "flicker" && host.isConnected) draw(); else stopFlicker(); }, 600); }
    else stopFlicker();
    const vmax = Math.max(0.5, maxSpeed(pair.truth, 4));
    const mask = st.conf > 0 ? Uint8Array.from(res.mu2, m => (m >= st.conf ? 1 : 0)) : undefined;
    const epe = endpointError(res, pair.truth, 0);
    wheel.el.style.display = st.tab === "flow" || st.tab === "pyramid" ? "" : "none";
    wheel.setScale(vmax);
    switch (st.tab) {
      case "frames": view.grey(pair.I1, { mode: "clip", lo: 0, hi: 255 }); view.setCaption("frame 1"); break;
      case "flicker": view.grey(flick ? pair.I2 : pair.I1, { mode: "clip", lo: 0, hi: 255 }); view.setCaption(flick ? "frame 2" : "frame 1"); break;
      case "flow": view.rgba(flowToRGBA(res, vmax, mask), N, N); view.setCaption(mask ? "estimated flow (grey: confidence too low)" : "estimated flow (colour wheel)"); break;
      case "arrows": view.grey(pair.I1, { mode: "clip", lo: -60, hi: 330 }); view.setCaption("estimated flow (black) and truth (green)"); break;
      case "conf": {
        const lg = Float32Array.from(res.mu2, v => Math.log10(1 + v));
        view.map(lg, N, N, viridis, 0, Math.log10(1 + CONF_MAX)); view.setCaption("confidence: smaller eigenvalue μ₂ of J (log)"); break;
      }
      case "epe": view.map(epe.map, N, N, inferno, 0, Math.max(0.5, vmax)); view.setCaption(`endpoint error (black 0 … yellow ${fmtSpeed(Math.max(0.5, vmax))} px)`); break;
      case "pyramid": {
        const px = new Uint8ClampedArray(4 * N * N).fill(255), L = res.perLevel.length;
        res.perLevel.forEach((f, l) => {
          const ox = (l % 2) * 64, oy = Math.floor(l / 2) * 64, rgba = flowToRGBA(f, vmax);
          for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) {
            const sx = Math.min(f.w - 1, Math.floor(x * f.w / 64)), sy = Math.min(f.h - 1, Math.floor(y * f.h / 64)), s = 4 * (sy * f.w + sx), d = 4 * ((oy + y) * N + ox + x);
            px[d] = rgba[s]; px[d + 1] = rgba[s + 1]; px[d + 2] = rgba[s + 2];
          }
        });
        view.rgba(px, N, N); view.setCaption(L > 1 ? "flow per level, each in its own pixels" : "one level only (raise 'pyramid levels')"); break;
      }
    }
    drawOverlay();
    drawPixel();
    let ce = 0, cn = 0;
    if (mask) for (let y = 8; y < N - 8; y++) for (let x = 8; x < N - 8; x++) { const i = y * N + x; if (mask[i]) { ce += epe.map[i]; cn++; } }
    const epeIn = endpointError(res, pair.truth, 8).mean;
    ro.textContent = `mean EPE ${epeIn.toFixed(2)} px` + (mask ? `\nconfident pixels: ${cn ? (ce / cn).toFixed(2) : "–"} px (${Math.round(100 * cn / ((N - 16) ** 2))} %)` : "") + `\n${ms.toFixed(0)} ms`;
  }
  function stopFlicker(): void { if (flickTimer) { clearInterval(flickTimer); flickTimer = 0; } }

  function drawOverlay(): void {
    if (!pair || !res) return;
    view.clear();
    if (st.tab === "arrows") {
      const sc = 6 / Math.max(1, maxSpeed(pair.truth, 4));
      drawQuiver(view, pair.truth, 12, sc, "#18a558", 2.2);
      drawQuiver(view, res, 12, sc, "#111", 1.6);
    }
    if (st.tab === "pyramid") {
      res.perLevel.forEach((f, l) => {
        const t = view.add("text", { x: (l % 2) * 64 + 3, y: Math.floor(l / 2) * 64 + 10, "font-size": 6, class: "qb-halo" });
        t.textContent = `level ${l}: median ${fmtSpeed(medianSpeed(f, Math.max(2, 8 >> l)))} px`;
      });
      return;
    }
    const r = st.rho * 2;
    view.add("circle", { cx: st.px + 0.5, cy: st.py + 0.5, r: Math.max(1, r), fill: "none", stroke: "#ffd400", "stroke-width": 2, "stroke-dasharray": "4 3" });
    view.box(st.px, st.py, 0, "#ffd400", 2.5);
  }

  function drawPixel(): void {
    if (!pair || !res || !deriv) return;
    const i = st.py * N + st.px, u0 = res.u[i], v0 = res.v[i], ut = pair.truth.u[i], vt = pair.truth.v[i];
    const R = Math.max(2, 1.4 * Math.max(Math.abs(ut), Math.abs(vt), Math.abs(u0), Math.abs(v0)) + 1);
    const lim = Math.min(12, R);
    vplot.setDomain([-lim, lim], [-lim, lim]);
    // constraint lines of the window (Gaussian weights), linearised around the estimate
    const rad = Math.ceil(2 * st.rho), stepPx = Math.max(1, Math.round(rad / 4));
    const ls: [number, number, number, number, number][] = [];
    let gmax = 0;
    const cand: [number, number, number, number][] = [];
    for (let dy = -rad; dy <= rad; dy += stepPx) for (let dx = -rad; dx <= rad; dx += stepPx) {
      const x = st.px + dx, y = st.py + dy;
      if (x < 0 || y < 0 || x >= N || y >= N || dx * dx + dy * dy > rad * rad) continue;
      const j = y * N + x, a = deriv.ix.data[j], b = deriv.iy.data[j], c = deriv.it.data[j] - a * u0 - b * v0;
      const wgt = Math.exp(-(dx * dx + dy * dy) / (2 * st.rho * st.rho)), g = Math.hypot(a, b);
      if (g < 1e-6) continue;
      gmax = Math.max(gmax, g * wgt);
      cand.push([a, b, c, g * wgt]);
    }
    for (const [a, b, c, s] of cand) if (s > 0.08 * gmax) ls.push([a, b, c, Math.min(0.9, 0.15 + 0.75 * s / gmax), 1 + 2 * s / gmax]);
    linesMark.set(ls);
    truthDot.set([ut], [-vt]);
    estDot.set([u0], [-v0]);
    const a = res.jxx.data[i], b = res.jxy.data[i], c = res.jyy.data[i], det = a * c - b * b;
    if (res.mu2[i] > 1e-3 * Math.max(1e-9, res.mu1[i]) && det > 0) {
      // uncertainty ∝ J⁻¹: residual variance / effective number of pixels in the window (4πρ²)
      let s2 = 0, sw = 0;
      for (const [aa, bb, cc, ww] of cand) { const rr = aa * u0 + bb * v0 + cc; s2 += ww * rr * rr; sw += ww; }
      const sig2 = Math.max(1, (sw ? s2 / sw : 0) + st.noise * st.noise) / (4 * Math.PI * st.rho * st.rho);
      const inv = [c / det * sig2, -b / det * sig2, a / det * sig2];
      ell.set([u0, -v0], [[inv[0], -inv[1]], [-inv[1], inv[2]]]);
      ell.g.style.display = "";
    } else ell.g.style.display = "none";
    const kind = res.mu1[i] < 3 ? "flat (both μ small)" : res.mu2[i] < 0.1 * res.mu1[i] ? "edge (μ₂ ≪ μ₁)" : "corner/texture";
    pro.textContent = `pixel (${st.px}, ${st.py}): ${kind}\nμ₁ = ${res.mu1[i].toFixed(1)}, μ₂ = ${res.mu2[i].toFixed(1)}\nestimate (${u0.toFixed(2)}, ${v0.toFixed(2)})\ntrue     (${ut.toFixed(2)}, ${vt.toFixed(2)})\nEPE ${Math.hypot(u0 - ut, v0 - vt).toFixed(2)} px`;
  }

  rebuild();
}

/** Median speed over the pixels at least `margin` from the border (robust to border artefacts). */
function medianSpeed(f: Flow, margin: number): number {
  const v: number[] = [];
  for (let y = margin; y < f.h - margin; y++) for (let x = margin; x < f.w - margin; x++) v.push(Math.hypot(f.u[y * f.w + x], f.v[y * f.w + x]));
  if (!v.length) return 0;
  v.sort((a, b) => a - b);
  return v[v.length >> 1];
}

autoMount(NAME, mount);
