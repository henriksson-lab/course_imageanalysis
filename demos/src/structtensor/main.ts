// The structure tensor: orientation, coherence and corners. Images: synthetic textures
// (stripes with an angle and a frequency, rotating stripes, a "fingerprint", two orientations
// meeting at a boundary, a checkerboard with a square and a triangle, a single square, smoothed
// isotropic noise) or the micrograph. J = G_ρ * (∇L ∇Lᵀ), with the gradient at scale σ and the
// window ρ. Tabs: the image with an ellipse per 16-px cell (axes from J), the orientation map
// (hue = orientation of the structure, saturation = coherence, brightness = energy), and the
// Harris response with the detected corners. Click a pixel: the window, its gradient vectors
// as arrows with their mean (which cancels on a line), the gradient scatter with the ellipse of
// J (the PCA picture), J with numbers, μ1, μ2, orientation, coherence and the verdict.
// "Average gradients instead" shows the naive orientation from the mean gradient.
//
// <div data-demo="structtensor" data-preset="edge|corner|flat|cancel|rho0|rho|rotating|checker|cell"
//      data-presets="edge,corner,flat"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { Plot, seriesColor } from "../lib/plot";
import { Grey, gaussianBlur } from "../lib/imageops";
import { structureTensor, StructureTensor, harris, localMaxima2D } from "../lib/edges";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { addGaussianNoise } from "../lib/synth";
import { ImgView, hueRGBA, colourWheel, injectStyle } from "../canny/imgview";
import { makeTexture } from "./scene";

const NAME = "structtensor";
const STYLE = `
[data-demo="${NAME}"] { gap: .9em; flex-wrap: nowrap; }
[data-demo="${NAME}"] .st-main { flex: 0 1 420px; min-width: 240px; }
[data-demo="${NAME}"] .st-mid { flex: 0 1 290px; min-width: 220px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 340px; gap: .3em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
[data-demo="${NAME}"] .st-tabs { display: flex; flex-wrap: wrap; gap: .25em; margin-bottom: .3em; }
[data-demo="${NAME}"] .st-tabs .qb-btn { padding: .12em .45em; }
[data-demo="${NAME}"] .st-row { display: flex; gap: .4em; align-items: flex-start; }
[data-demo="${NAME}"] .st-row > * { flex: 1 1 0; min-width: 0; }
[data-demo="${NAME}"] .st-key { display: flex; align-items: center; gap: .4em; font-size: 14px; color: var(--qb-muted); }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .st-main { flex: 0 0 410px; }
.reveal [data-demo="${NAME}"] .st-mid { flex: 0 0 300px; gap: .15em; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 1 1 320px; gap: .15em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .08em .4em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; padding: .2em .45em; }
.reveal [data-demo="${NAME}"] .ev-credit, .reveal [data-demo="${NAME}"] .st-key { display: none; }
@media (max-width: 860px) { [data-demo="${NAME}"] { flex-wrap: wrap; } }
`;

type Tab = "ellipses" | "orient" | "corners";
const TABS: [Tab, string][] = [["ellipses", "image + ellipses"], ["orient", "orientation"], ["corners", "corners (Harris)"]];
const IMAGES: [string, string][] = [["stripes", "stripes"], ["rotating", "rotating"], ["fingerprint", "fingerprint"], ["boundary", "two textures"], ["checker", "corners"], ["square", "square"], ["noise", "noise"], ["cell", "micrograph"]];
interface State { image: string; angle: number; freq: number; noise: number; sigma: number; rho: number; k: number; thr: number; naive: boolean; tab: Tab; px: number; py: number }
interface Preset { id: string; label: string; s: Partial<State> }
const DEFAULT: State = { image: "square", angle: 30, freq: 0.12, noise: 0, sigma: 1, rho: 3, k: 0.05, thr: 0.1, naive: false, tab: "ellipses", px: 32, py: 64 };
const PRESETS: Preset[] = [
  { id: "edge", label: "edge: one big eigenvalue", s: { image: "square", px: 32, py: 64 } },
  { id: "corner", label: "corner: two", s: { image: "square", px: 32, py: 32 } },
  { id: "flat", label: "flat: none", s: { image: "square", px: 64, py: 64 } },
  { id: "cancel", label: "averaging gradients cancels", s: { image: "stripes", naive: true, rho: 4, tab: "orient", px: 64, py: 64 } },
  { id: "rho0", label: "ρ = 0: every pixel an edge", s: { image: "square", rho: 0, tab: "corners", px: 32, py: 32 } },
  { id: "rho", label: "ρ too small or too large", s: { image: "boundary", noise: 25, rho: 1, tab: "orient", px: 60, py: 64 } },
  { id: "rotating", label: "rotating stripes: hue follows", s: { image: "rotating", freq: 0.15, tab: "orient", px: 64, py: 64 } },
  { id: "checker", label: "corners of a checkerboard", s: { image: "checker", tab: "corners", rho: 2, px: 32, py: 32 } },
  { id: "cell", label: "micrograph", s: { image: "cell", sigma: 1.5, rho: 4, tab: "orient", px: 50, py: 30 } },
];
const CELL = 16;

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(p0?.s ?? {}) });
  const st: State = start();

  const main = htmlEl("div", "st-main", "", host);
  const tabsDiv = htmlEl("div", "st-tabs", "", main);
  const tabBtns = TABS.map(([t, l]) => button(tabsDiv, l, () => { st.tab = t; draw(); }));
  const view = new ImgView(main, "");
  const key = htmlEl("div", "st-key", "", main);
  colourWheel(key, 60, true);
  htmlEl("span", "", "hue = orientation of the structure (0° and 180° alike), saturation = coherence, brightness = energy", key);
  const credit = htmlEl("div", "ev-credit", CELL_CREDIT, main);

  const mid = htmlEl("div", "st-mid", "", host);
  const row = htmlEl("div", "st-row", "", mid);
  const win = new ImgView(row, "window");
  const sp = new Plot(row, { width: 200, height: 200, x: [-1, 1], y: [-1, 1], equal: true, font: 15, xticks: [], yticks: [], margin: { l: 8, r: 8, t: 26, b: 8 }, title: "gradient cloud" });
  const cloud = sp.scatter([], [], { r: 3, color: seriesColor(0), opacity: 0.7 });
  const ell = sp.mark<[number, number, number, number]>((g, a, b, c, m) => {
    // ellipse of J scaled to the plot: semi-axes √μ along the eigenvectors (1.5 "standard deviations")
    const tr = a + c, d = Math.sqrt(((a - c) / 2) ** 2 + b * b), mu1 = tr / 2 + d, mu2 = Math.max(0, tr / 2 - d), th = 0.5 * Math.atan2(2 * b, a - c);
    const pts: string[] = [];
    for (let k = 0; k <= 72; k++) {
      const t = 2 * Math.PI * k / 72, u = 1.5 * Math.sqrt(mu1) * Math.cos(t), v = 1.5 * Math.sqrt(mu2) * Math.sin(t);
      const x = u * Math.cos(th) - v * Math.sin(th), y = u * Math.sin(th) + v * Math.cos(th);
      pts.push(`${sp.x(Math.max(-m, Math.min(m, x))).toFixed(1)},${sp.y(Math.max(-m, Math.min(m, -y))).toFixed(1)}`);
    }
    const p = svgEl("polygon", { points: pts.join(" ") }, g);
    p.style.fill = seriesColor(3); p.style.fillOpacity = "0.12"; p.style.stroke = seriesColor(3); p.style.strokeWidth = "2.5";
  }, [0, 0, 0, 1]);
  const meanArrow = sp.arrow(0, 0, 0, 0, { color: seriesColor(1), width: 3 });
  const ro = htmlEl("div", "qb-readout", "", mid);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ev-lab", "presets", panel);
  const only = host.dataset.presets ? host.dataset.presets.split(/[\s,]+/) : null;
  const shown = PRESETS.filter(p => !only || only.includes(p.id));
  const presetBtns = presets(panel, shown.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "ev-lab", "image", panel);
  const imgBtns = presets(panel, IMAGES.map(([id, l]) => ({ label: l, apply: () => { st.image = id; st.px = 64; st.py = 64; makeImage(); } })));
  const stripeDiv = htmlEl("div", "", "", panel);
  const angS = slider(stripeDiv, "angle", 0, 180, 5, st.angle, v => { st.angle = v; makeImage(); }, v => v + "°");
  const freqS = slider(stripeDiv, "frequency", 0.04, 0.3, 0.01, st.freq, v => { st.freq = v; makeImage(); });
  const sigS = slider(panel, "σ (gradient)", 0.5, 4, 0.25, st.sigma, v => { st.sigma = v; compute(); });
  const rhoS = slider(panel, "ρ (window)", 0, 12, 0.5, st.rho, v => { st.rho = v; compute(); });
  const cornerDiv = htmlEl("div", "", "", panel);
  const kS = slider(cornerDiv, "Harris k", 0.02, 0.2, 0.01, st.k, v => { st.k = v; draw(); });
  const thrS = slider(cornerDiv, "corner threshold", 0, 0.5, 0.01, st.thr, v => { st.thr = v; draw(); }, v => v.toFixed(2) + "·max");
  const noiseS = slider(panel, "noise", 0, 40, 1, st.noise, v => { st.noise = v; makeImage(); });
  const naiveC = checkbox(panel, "average gradients instead", st.naive, v => { st.naive = v; compute(); });
  button(panel, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); makeImage(); });

  function sync(): void {
    angS.set(st.angle); freqS.set(st.freq); sigS.set(st.sigma); rhoS.set(st.rho); kS.set(st.k); thrS.set(st.thr); noiseS.set(st.noise); naiveC.checked = st.naive;
  }
  function load(p: Preset): void { Object.assign(st, start(), p.s); sync(); makeImage(); }
  const pick = (x: number, y: number) => { st.px = Math.round(x * 128 / N); st.py = Math.round(y * 128 / N); draw(); };
  view.onPick(pick);

  let N = 128, img: Grey | null = null, cell: Grey | null = null;
  let J: StructureTensor | null = null, mgx: Grey | null = null, mgy: Grey | null = null;
  function makeImage(): void {
    if (st.image === "cell") {
      if (!cell) { loadCell(256).then(g => { cell = g; makeImage(); }); return; }
      N = 256; img = st.noise > 0 ? addGaussianNoise(cell, st.noise, 1) : cell;
    } else { N = 128; img = makeTexture(st.image, N, st.noise, 1, st.angle, st.freq); }
    compute();
  }
  function compute(): void {
    if (!img) return;
    J = structureTensor(img, st.sigma, st.rho);
    mgx = gaussianBlur(J.gx, st.rho); mgy = gaussianBlur(J.gy, st.rho);
    draw();
  }

  function draw(): void {
    if (!img || !J || !mgx || !mgy) return;
    tabBtns.forEach((b, i) => b.classList.toggle("active", TABS[i][0] === st.tab));
    imgBtns.forEach((b, i) => b.classList.toggle("active", IMAGES[i][0] === st.image));
    stripeDiv.style.display = st.image === "stripes" || st.image === "boundary" ? "" : "none";
    cornerDiv.style.display = st.tab === "corners" ? "" : "none";
    key.style.display = st.tab === "orient" ? "" : "none";
    credit.style.display = st.image === "cell" ? "" : "none";
    const n = N * N, Jt = J;
    view.clear();
    let cap = "";
    if (st.tab === "ellipses") {
      view.grey(img, { mode: "auto" });
      let mmax = 0; for (let y = CELL / 2; y < N; y += CELL) for (let x = CELL / 2; x < N; x += CELL) mmax = Math.max(mmax, Jt.mu1.data[y * N + x]);
      const R = 0.46 * CELL;
      for (let y = CELL / 2; y < N; y += CELL) for (let x = CELL / 2; x < N; x += CELL) {
        const i = y * N + x, a = R * Math.sqrt(Jt.mu1.data[i] / (mmax || 1)), b = R * Math.sqrt(Jt.mu2.data[i] / (mmax || 1));
        if (a < 0.4) { view.add("circle", { cx: x, cy: y, r: 0.6, fill: "#ffd400" }); continue; }
        view.add("ellipse", { cx: x, cy: y, rx: Math.max(a, 0.3), ry: Math.max(b, 0.3), fill: "none", stroke: "#ffd400", "stroke-width": 2, transform: `rotate(${(Jt.angle.data[i] * 180 / Math.PI).toFixed(1)} ${x} ${y})` });
      }
      cap = "image, one ellipse of J per 16-px cell";
    } else if (st.tab === "orient") {
      const ang = new Float32Array(n), val = new Float32Array(n), sat = new Float32Array(n);
      let emax = 0;
      for (let i = 0; i < n; i++) {
        if (st.naive) {
          const gx = mgx.data[i], gy = mgy.data[i];
          ang[i] = Math.atan2(-gy, gx) + Math.PI / 2; val[i] = Math.hypot(gx, gy); sat[i] = 1;
        } else {
          ang[i] = -Jt.angle.data[i] + Math.PI / 2; val[i] = Math.sqrt(Jt.mu1.data[i] + Jt.mu2.data[i]); sat[i] = Jt.coherence.data[i];
        }
        emax = Math.max(emax, val[i]);
      }
      view.rgba(hueRGBA(ang, val, st.naive ? emax : 0.8 * emax, sat, true), N, N);
      cap = st.naive ? "orientation from the AVERAGE gradient (naive)" : "orientation (hue), coherence (saturation)";
    } else {
      const R = harris(Jt, st.k);
      view.grey(R, { mode: "mid" });
      let rmax = 0; for (const v of R.data) rmax = Math.max(rmax, v);
      const cs = localMaxima2D(R, st.thr * rmax, Math.max(2, Math.round(2 * st.rho)), 2).slice(0, 200);
      for (const c of cs) {
        const s = 2.5 * N / 128;
        view.add("line", { x1: c.x + 0.5 - s, y1: c.y + 0.5 - s, x2: c.x + 0.5 + s, y2: c.y + 0.5 + s, stroke: "#ff3b30", "stroke-width": 3 });
        view.add("line", { x1: c.x + 0.5 - s, y1: c.y + 0.5 + s, x2: c.x + 0.5 + s, y2: c.y + 0.5 - s, stroke: "#ff3b30", "stroke-width": 3 });
      }
      cap = `Harris response det J − k·(tr J)² (0 = grey): ${cs.length} corners`;
    }
    view.setCaption(cap);
    // window around the selected pixel
    const X = Math.min(N - 1, Math.round(st.px * N / 128)), Y = Math.min(N - 1, Math.round(st.py * N / 128)), i0 = Y * N + X;
    const W = Math.max(2, Math.ceil(2 * Math.max(st.rho, 0.5)));
    view.add("rect", { x: X - W, y: Y - W, width: 2 * W + 1, height: 2 * W + 1, fill: "none", stroke: "#22d3ee", "stroke-width": 2 });
    view.box(X, Y, 0, "#ffd400", 2);
    const P = 2 * W + 1, patch = new Float32Array(P * P);
    for (let v = 0; v < P; v++) for (let u = 0; u < P; u++) {
      const x = Math.max(0, Math.min(N - 1, X - W + u)), y = Math.max(0, Math.min(N - 1, Y - W + v));
      patch[v * P + u] = img.data[y * N + x];
    }
    win.grey({ width: P, height: P, data: patch }, { mode: "auto" });
    win.clear();
    const stepA = Math.max(1, Math.ceil(P / 9)), gxs: number[] = [], gys: number[] = [];
    let gmax = 0;
    for (let v = 0; v < P; v++) for (let u = 0; u < P; u++) {
      const x = X - W + u, y = Y - W + v;
      if (x < 0 || y < 0 || x >= N || y >= N || (u - W) ** 2 + (v - W) ** 2 > W * W) continue;
      const j = y * N + x; gxs.push(Jt.gx.data[j]); gys.push(Jt.gy.data[j]); gmax = Math.max(gmax, Math.hypot(Jt.gx.data[j], Jt.gy.data[j]));
    }
    for (let v = 0; v < P; v += stepA) for (let u = 0; u < P; u += stepA) {
      const x = X - W + u, y = Y - W + v;
      if (x < 0 || y < 0 || x >= N || y >= N || (u - W) ** 2 + (v - W) ** 2 > W * W) continue;
      const j = y * N + x, L = 0.9 * stepA / (gmax || 1), gx = Jt.gx.data[j] * L, gy = Jt.gy.data[j] * L;
      if (Math.hypot(gx, gy) > 0.15) win.arrow(u + 0.5 - gx / 2, v + 0.5 - gy / 2, u + 0.5 + gx / 2, v + 0.5 + gy / 2, "#ffd400", 2);
    }
    const mx = mgx.data[i0], my = mgy.data[i0], Lm = 0.45 * P / (gmax || 1);
    if (Math.hypot(mx, my) * Lm > 0.3) win.arrow(W + 0.5, W + 0.5, W + 0.5 + mx * Lm, W + 0.5 + my * Lm, "#ff3b30", 3.5);
    // gradient cloud (y up in the plot: plot −gy so it matches the screen)
    const m = 1.15 * (gmax || 1);
    sp.setDomain([-m, m], [-m, m]);
    cloud.set(gxs, gys.map(v => -v));
    ell.set(Jt.jxx.data[i0], Jt.jxy.data[i0], Jt.jyy.data[i0], m);
    meanArrow.set(0, 0, mx, -my);
    // readout
    const a = Jt.jxx.data[i0], b = Jt.jxy.data[i0], c = Jt.jyy.data[i0], mu1 = Jt.mu1.data[i0], mu2 = Jt.mu2.data[i0];
    let m1max = 0; for (const v of Jt.mu1.data) m1max = Math.max(m1max, v);
    const verdict = mu1 < 0.02 * m1max ? "flat" : mu2 > 0.2 * mu1 ? "corner (two large eigenvalues)" : "edge / oriented (one large)";
    const ori = ((Math.round(-Jt.angle.data[i0] * 180 / Math.PI + 90) % 180) + 180) % 180;
    const f = (v: number) => (Math.abs(v) < 0.005 ? "0" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
    ro.textContent =
      `pixel (${X}, ${Y}), σ = ${st.sigma}, ρ = ${st.rho}\nJ = [${f(a)}  ${f(b)}; ${f(b)}  ${f(c)}]\nμ1 = ${f(mu1)}, μ2 = ${f(mu2)}\n` +
      `orientation ${ori.toFixed(0)}°, coherence ${Jt.coherence.data[i0].toFixed(2)}\nmean gradient (red) ${f(Math.hypot(mx, my))} vs √μ1 = ${f(Math.sqrt(mu1))}\n→ ${verdict}`;
  }

  sync();
  if (p0 && shown.includes(p0)) presetBtns[shown.indexOf(p0)].classList.add("active");
  makeImage();
}

autoMount(NAME, mount);
