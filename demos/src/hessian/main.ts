// The Hessian per pixel: blobs, ridges and saddles. Images: an editable quadratic
// f = ax² + bxy + cy², a test scene (bright blob, thin and thick bright lines, a dark line, a
// saddle, an edge), curved lines, the micrograph. The Hessian of the σ-smoothed image,
// normalised by σ², gives two eigenvalues |λ1| ≤ |λ2| per pixel. Tabs: the image, the
// class map (bright/dark blob, bright/dark ridge, saddle, flat), λ1, λ2, trace (= the LoG),
// determinant, Frangi vesselness, best scale. Right: the (λ1, λ2) plane with the class regions
// and a thinned scatter of all pixels. Click a pixel: glyph (eigenvector axes, length ∝ |λ|,
// red = negative curvature), the matrix, eigenvalues and verdict. Multi-scale: σ from 1 to
// σmax (8 levels), each pixel takes the scale with the largest |λ2| (vesselness: the largest
// vesselness).
//
// Modes and images are registries (modes.ts): data-mode="hessian" (default); the skeleton
// day registers a vesselness mode there.
// <div data-demo="hessian" data-mode="hessian" data-preset="hill|ridge|saddle|edge|trace|vessel|thick|multi|cell"
//      data-image="quadratic|scene|lines|cell" data-presets="hill,ridge,saddle"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { Plot } from "../lib/plot";
import { Grey } from "../lib/imageops";
import { hessian, HessianResult, classifyHessianPixel, HessClass, HESS_CLASS_NAMES, frangiValue, geomScales } from "../lib/edges";
import { CELL_CREDIT } from "../lib/cell";
import { inferno, viridis, scalarToRGBA } from "../lib/colour";
import { ImgView, injectStyle } from "../canny/imgview";
import { TABS, Tab, HessState, BASE_STATE, IMAGES, MODES, HessPreset } from "./modes";

const NAME = "hessian";
const STYLE = `
[data-demo="${NAME}"] { gap: .9em; flex-wrap: nowrap; }
[data-demo="${NAME}"] .hs-main { flex: 0 1 420px; min-width: 240px; }
[data-demo="${NAME}"] .hs-mid { flex: 0 1 300px; min-width: 220px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 340px; gap: .3em; }
[data-demo="${NAME}"] .qb-readout { font-size: 15px; white-space: pre-wrap; }
[data-demo="${NAME}"] .hs-tabs { display: flex; flex-wrap: wrap; gap: .25em; margin-bottom: .3em; }
[data-demo="${NAME}"] .hs-tabs .qb-btn { padding: .12em .45em; }
[data-demo="${NAME}"] .hs-legend { display: flex; flex-wrap: wrap; gap: .1em .7em; font-size: 14px; }
[data-demo="${NAME}"] .hs-legend span::before { content: ""; display: inline-block; width: .8em; height: .8em; margin-right: .25em; background: var(--c); border-radius: 2px; vertical-align: -1px; }
[data-demo="${NAME}"] .hs-busy { font-size: 14px; color: var(--qb-muted); min-height: 1.1em; }
.reveal [data-demo="${NAME}"] { font-size: 18px; }
.reveal [data-demo="${NAME}"] .hs-main { flex: 0 0 410px; }
.reveal [data-demo="${NAME}"] .hs-mid { flex: 0 0 300px; gap: .15em; }
.reveal [data-demo="${NAME}"] .qb-panel { flex: 1 1 320px; gap: .15em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .08em .4em; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .2em .45em; }
.reveal [data-demo="${NAME}"] .ev-credit { display: none; }
@media (max-width: 860px) { [data-demo="${NAME}"] { flex-wrap: wrap; } }
`;

const CLASS_COLOURS: [number, number, number][] = [[150, 150, 150], [232, 64, 42], [43, 108, 176], [255, 159, 28], [34, 167, 201], [155, 89, 182]];
const css = (c: [number, number, number]) => `rgb(${c.join(",")})`;
const LEVELS = 8;

/** "−x² + 0.5xy" style text of a x² + b xy + c y². */
function poly(a: number, b: number, c: number): string {
  const terms: string[] = [];
  for (const [k, v] of [[a, "x²"], [b, "xy"], [c, "y²"]] as [number, string][]) {
    if (Math.abs(k) < 1e-9) continue;
    const mag = Math.abs(k) === 1 ? "" : String(+Math.abs(k).toFixed(2));
    terms.push((k < 0 ? "−" : terms.length ? "+" : "") + mag + v);
  }
  return terms.length ? terms.join(" ").replace(/^\+/, "") : "0";
}

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const mode = MODES.get(host.dataset.mode ?? "hessian") ?? MODES.get("hessian")!;
  const p0 = mode.presets.find(p => p.id === host.dataset.preset);
  const start = (): HessState => ({ ...BASE_STATE, mode: mode.id, ...mode.defaults, ...(host.dataset.image && IMAGES.has(host.dataset.image) ? { image: host.dataset.image } : {}), ...(p0?.s ?? {}) });
  const st: HessState = start();

  // ---- layout ----
  const main = htmlEl("div", "hs-main", "", host);
  const tabsDiv = htmlEl("div", "hs-tabs", "", main);
  const tabBtns = TABS.map(([t, l]) => button(tabsDiv, l, () => { st.tab = t; draw(); }));
  const view = new ImgView(main, "");
  const legend = htmlEl("div", "hs-legend", "", main);
  HESS_CLASS_NAMES.forEach((n, k) => { const s = htmlEl("span", "", n, legend); s.style.setProperty("--c", css(CLASS_COLOURS[k])); });
  const credit = htmlEl("div", "ev-credit", CELL_CREDIT, main);
  const mid = htmlEl("div", "hs-mid", "", host);
  const sp = new Plot(mid, { width: 300, height: 300, x: [-1, 1], y: [-1, 1], equal: true, font: 16, xlabel: "λ1 (smaller |λ|)", ylabel: "λ2", margin: { l: 50, r: 6, t: 6, b: 46 }, xticks: 3, yticks: 3 });
  const regions = sp.mark<[number, number]>((g, m, t) => {
    const L = (x1: number, y1: number, x2: number, y2: number, dash: string, col = "#999") => {
      const l = svgEl("line", { x1: sp.x(x1), y1: sp.y(y1), x2: sp.x(x2), y2: sp.y(y2) }, g);
      l.style.stroke = col; l.style.strokeWidth = "1.5"; l.style.strokeDasharray = dash;
    };
    const band = svgEl("rect", { x: sp.x(-m), y: sp.y(t), width: sp.x(m) - sp.x(-m), height: sp.y(-t) - sp.y(t) }, g);
    band.style.fill = "#999"; band.style.opacity = "0.15";
    L(-m, -m, m, m, "2 3"); L(-m, m, m, -m, "2 3");
    L(-0.5 * m, -m, 0, 0, "6 4", "#666"); L(0.5 * m, -m, 0, 0, "6 4", "#666"); L(-0.5 * m, m, 0, 0, "6 4", "#666"); L(0.5 * m, m, 0, 0, "6 4", "#666");
    const T = (x: number, y: number, s: string, c: [number, number, number]) => {
      const e = svgEl("text", { x: sp.x(x), y: sp.y(y), "text-anchor": "middle", class: "qb-halo" }, g);
      e.textContent = s; e.style.fontSize = "13px"; e.style.fill = css(c); e.style.fontWeight = "700";
    };
    T(0, -0.93 * m, "bright ridge", CLASS_COLOURS[3]); T(0, 0.88 * m, "dark ridge", CLASS_COLOURS[4]);
    T(-0.62 * m, -0.75 * m, "bright blob", CLASS_COLOURS[1]); T(0.62 * m, 0.8 * m, "dark blob", CLASS_COLOURS[2]);
    T(0.66 * m, -0.75 * m, "saddle", CLASS_COLOURS[5]); T(-0.66 * m, 0.8 * m, "saddle", CLASS_COLOURS[5]);
  }, [1, 0.1]);
  const scat = sp.scatter([], [], { r: 2.2, opacity: 0.6 });
  const selDot = sp.scatter([], [], { r: 8, color: "none", stroke: "#000" });
  const ro = htmlEl("div", "qb-readout", "", mid);
  const busy = htmlEl("div", "hs-busy", "", mid);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ev-lab", "presets", panel);
  const only = host.dataset.presets ? host.dataset.presets.split(/[\s,]+/) : null;
  const shown = mode.presets.filter(p => !only || only.includes(p.id));
  const presetBtns = presets(panel, shown.map(p => ({ label: p.label, apply: () => load(p) })));
  htmlEl("div", "ev-lab", "image", panel);
  const imgIds = mode.images.filter(id => IMAGES.has(id));
  const imgBtns = presets(panel, imgIds.map(id => ({ label: IMAGES.get(id)!.label, apply: () => { st.image = id; st.px = 64; st.py = 64; makeImage(); } })));
  const quadDiv = htmlEl("div", "", "", panel);
  const qa = slider(quadDiv, "a", -1, 1, 0.1, st.qa, v => { st.qa = v; makeImage(); });
  const qb = slider(quadDiv, "b", -1, 1, 0.1, st.qb, v => { st.qb = v; makeImage(); });
  const qc = slider(quadDiv, "c", -1, 1, 0.1, st.qc, v => { st.qc = v; makeImage(); });
  const sigS = slider(panel, "σ", 1, 8, 0.25, st.sigma, v => { st.sigma = v; compute(); });
  const multiC = checkbox(panel, "multi-scale σ = 1 … σmax", st.multi, v => { st.multi = v; compute(); });
  const smaxDiv = htmlEl("div", "", "", panel);
  const smaxS = slider(smaxDiv, "σmax", 2, 12, 0.5, st.sigmaMax, v => { st.sigmaMax = v; if (st.multi) compute(); });
  const tS = slider(panel, "flat below |λ2| =", 0, 20, 0.25, st.t, v => { st.t = v; draw(); });
  const vesDiv = htmlEl("div", "", "", panel);
  vesDiv.style.display = "flex"; vesDiv.style.flexDirection = "column"; vesDiv.style.gap = "inherit";
  const betaS = slider(vesDiv, "β", 0.1, 2, 0.05, st.beta, v => { st.beta = v; draw(); });
  const cS = slider(vesDiv, "c", 1, 60, 1, st.c, v => { st.c = v; draw(); });
  const brightC = checkbox(vesDiv, "bright lines (untick: dark)", st.bright, v => { st.bright = v; draw(); });
  const noiseS = slider(panel, "noise", 0, 30, 1, st.noise, v => { st.noise = v; makeImage(); });
  button(panel, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); makeImage(); });

  function sync(): void {
    qa.set(st.qa); qb.set(st.qb); qc.set(st.qc); sigS.set(st.sigma); multiC.checked = st.multi; smaxS.set(st.sigmaMax);
    tS.set(st.t); betaS.set(st.beta); cS.set(st.c); noiseS.set(st.noise); brightC.checked = st.bright;
  }
  function load(p: HessPreset): void { Object.assign(st, start(), p.s); sync(); makeImage(); }
  view.onPick((x, y) => { st.px = Math.round(x * 128 / N); st.py = Math.round(y * 128 / N); draw(); });

  // ---- computation ----
  let N = 128, img: Grey | null = null;
  let Hs: HessianResult[] = [], sigmas: number[] = [];
  let pick: Int32Array = new Int32Array(0);       // per pixel: index of the scale used for the eigenvalues
  async function makeImage(): Promise<void> {
    const im = IMAGES.get(st.image) ?? IMAGES.get("scene")!;
    N = im.size;
    busy.textContent = "loading …";
    img = await im.make(N, st);
    compute();
  }
  function compute(): void {
    if (!img) return;
    busy.textContent = "computing …";
    setTimeout(() => {
      const t0 = performance.now();
      sigmas = st.multi ? geomScales(1, st.sigmaMax, LEVELS) : [st.sigma];
      const im = IMAGES.get(st.image);
      Hs = sigmas.map(s => {
        const H = hessian(img!, s, false), k = im?.units ? im.units(N, s) : s * s;
        for (const g of [H.xx, H.xy, H.yy, H.l1, H.l2]) for (let i = 0; i < g.data.length; i++) g.data[i] *= k;
        return H;
      });
      pick = new Int32Array(N * N);
      if (Hs.length > 1) for (let i = 0; i < N * N; i++) {
        let b = 0; for (let k = 1; k < Hs.length; k++) if (Math.abs(Hs[k].l2.data[i]) > Math.abs(Hs[b].l2.data[i])) b = k;
        pick[i] = b;
      }
      busy.textContent = `${sigmas.length} scale${sigmas.length > 1 ? "s" : ""}, ${N} × ${N}: ${Math.round(performance.now() - t0)} ms`;
      draw();
    }, 10);
  }

  // ---- drawing ----
  function draw(): void {
    if (!img || !Hs.length) return;
    tabBtns.forEach((b, i) => b.classList.toggle("active", TABS[i][0] === st.tab));
    imgBtns.forEach((b, i) => b.classList.toggle("active", imgIds[i] === st.image));
    quadDiv.style.display = st.image === "quadratic" ? "" : "none";
    credit.style.display = IMAGES.get(st.image)?.credit ? "" : "none";
    legend.style.visibility = st.tab === "classes" ? "visible" : "hidden";
    smaxDiv.style.display = st.multi ? "" : "none"; sigS.input.disabled = st.multi;
    vesDiv.style.display = st.tab === "vessel" || st.tab === "scale" ? "flex" : "none";
    const n = N * N, l1 = new Float32Array(n), l2 = new Float32Array(n), cls = new Uint8Array(n), ves = new Float32Array(n), best = new Int32Array(n);
    let gmin = Infinity, gmax = -Infinity;
    for (let i = 0; i < n; i++) { gmin = Math.min(gmin, img.data[i]); gmax = Math.max(gmax, img.data[i]); }
    for (let i = 0; i < n; i++) {
      const H = Hs[pick[i]];
      l1[i] = H.l1.data[i]; l2[i] = H.l2.data[i];
      cls[i] = classifyHessianPixel(l1[i], l2[i], st.t);
      let vb = 0, kb = 0;
      for (let k = 0; k < Hs.length; k++) { const v = frangiValue(Hs[k].l1.data[i], Hs[k].l2.data[i], st.beta, st.c, st.bright); if (v > vb) { vb = v; kb = k; } }
      ves[i] = vb; best[i] = kb;
    }
    const px = new Uint8ClampedArray(4 * n);
    const grey = (i: number) => 255 * (img!.data[i] - gmin) / (gmax - gmin || 1);
    const signed = (a: ArrayLike<number>, cap?: number) => {
      let m = cap ?? 0; if (!cap) for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(a[i]));
      for (let i = 0; i < n; i++) { const v = 128 + 127 * Math.max(-1, Math.min(1, a[i] / (m || 1))); px[4 * i] = px[4 * i + 1] = px[4 * i + 2] = v; px[4 * i + 3] = 255; }
    };
    let cap = "";
    switch (st.tab) {
      case "image": {
        const contour = st.image === "quadratic";
        for (let i = 0; i < n; i++) {
          let g = grey(i);
          if (contour) { const x = i % N; const a = Math.floor(img.data[i] / 12), b = x + 1 < N ? Math.floor(img.data[i + 1] / 12) : a, c = i + N < n ? Math.floor(img.data[i + N] / 12) : a; if (a !== b || a !== c) g = 0.45 * g; }
          px[4 * i] = px[4 * i + 1] = px[4 * i + 2] = g; px[4 * i + 3] = 255;
        }
        cap = st.image === "quadratic" ? `f = ${poly(st.qa, st.qb, st.qc)} (with contours)` : "image"; break;
      }
      case "classes":
        for (let i = 0; i < n; i++) {
          const c = CLASS_COLOURS[cls[i]], g = grey(i), w = cls[i] === HessClass.Flat ? 0 : 0.7;
          px[4 * i] = w * c[0] + (1 - w) * g * (w ? 1 : 0.6); px[4 * i + 1] = w * c[1] + (1 - w) * g * (w ? 1 : 0.6); px[4 * i + 2] = w * c[2] + (1 - w) * g * (w ? 1 : 0.6); px[4 * i + 3] = 255;
        }
        cap = "classes from (λ1, λ2)"; break;
      case "l1": signed(l1); cap = "λ1 (the smaller curvature), 0 = grey"; break;
      case "l2": signed(l2); cap = "λ2 (the larger curvature), 0 = grey"; break;
      case "trace": { const tr = l1.map((v, i) => v + l2[i]); signed(tr); cap = "trace λ1 + λ2 = σ²·LoG"; break; }
      case "det": { const d = l1.map((v, i) => v * l2[i]); signed(d); cap = "determinant λ1·λ2 (blobs only)"; break; }
      case "vessel": { let m = 0; for (const v of ves) m = Math.max(m, v); px.set(scalarToRGBA(ves, { map: inferno, lo: 0, hi: m || 1 })); cap = `vesselness (${st.bright ? "bright" : "dark"} lines)`; break; }
      case "scale": {
        let m = 0; for (const v of ves) m = Math.max(m, v);
        const sc = scalarToRGBA(Float32Array.from(best, (k, i) => (ves[i] > 0.1 * m ? k : NaN)), { map: viridis, lo: 0, hi: Math.max(1, sigmas.length - 1), nan: [0, 0, 0, 255] });
        px.set(sc); cap = sigmas.length > 1 ? `best σ for vesselness (dark blue ${sigmas[0].toFixed(1)} → yellow ${sigmas[sigmas.length - 1].toFixed(1)})` : "best σ: tick multi-scale"; break;
      }
    }
    view.rgba(px, N, N); view.setCaption(cap);
    // glyph
    const X = Math.min(N - 1, Math.round(st.px * N / 128)), Y = Math.min(N - 1, Math.round(st.py * N / 128)), i = Y * N + X;
    const H = Hs[pick[i]], a = H.angle1.data[i], e1 = l1[i], e2 = l2[i], mx = Math.max(Math.abs(e1), Math.abs(e2)) || 1;
    view.clear();
    const Lg = 10 * N / 128;
    const axis = (ang: number, lam: number) => {
      const len = Lg * Math.abs(lam) / mx, dx = Math.cos(ang) * len, dy = Math.sin(ang) * len;
      view.add("line", { x1: X + 0.5 - dx, y1: Y + 0.5 - dy, x2: X + 0.5 + dx, y2: Y + 0.5 + dy, stroke: lam < 0 ? "#ff3b30" : "#3b82f6", "stroke-width": 5, "stroke-linecap": "round" });
    };
    axis(a, e1); axis(a + Math.PI / 2, e2);
    view.add("circle", { cx: X + 0.5, cy: Y + 0.5, r: 1.5 * N / 128, fill: "#ffd400", stroke: "#000", "stroke-width": 1 });
    // scatter (pixels away from the border, where the filters are not affected by it)
    const mg = Math.min(Math.floor(N / 4), Math.ceil(2.5 * sigmas[sigmas.length - 1])), inner: number[] = [];
    for (let y = mg; y < N - mg; y++) for (let x = mg; x < N - mg; x++) inner.push(y * N + x);
    const all = Float32Array.from(inner, j => Math.abs(l2[j])).sort();
    const m = Math.max(st.t * 1.5, all[Math.floor(0.995 * (all.length - 1))] * 1.1, 1e-6);
    sp.setDomain([-m, m], [-m, m]);
    regions.set(m, st.t);
    const stride = Math.max(1, Math.floor(inner.length / 2500)), xs: number[] = [], ys: number[] = [], cols: string[] = [];
    for (let q = 0; q < inner.length; q += stride) { const j = inner[q]; xs.push(Math.max(-m, Math.min(m, l1[j]))); ys.push(Math.max(-m, Math.min(m, l2[j]))); cols.push(css(CLASS_COLOURS[cls[j]])); }
    scat.set(xs, ys, cols);
    selDot.set([Math.max(-m, Math.min(m, e1))], [Math.max(-m, Math.min(m, e2))]);
    const f = (v: number) => (Math.abs(v) < 0.005 ? "0" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
    ro.textContent =
      `pixel (${X}, ${Y}), σ = ${sigmas[pick[i]].toFixed(2)}${st.image === "quadratic" ? " (units of f)" : " (×σ²)"}\n` +
      `H = [${f(H.xx.data[i])}  ${f(H.xy.data[i])}; ${f(H.xy.data[i])}  ${f(H.yy.data[i])}]\n` +
      `λ1 = ${f(e1)}, λ2 = ${f(e2)}\ntrace = ${f(e1 + e2)}, det = ${f(e1 * e2)}\n` +
      `→ ${HESS_CLASS_NAMES[cls[i]]}; vesselness ${ves[i].toFixed(2)}` + (sigmas.length > 1 ? ` (best σ ${sigmas[best[i]].toFixed(1)})` : "");
  }

  sync();
  if (p0 && shown.includes(p0)) presetBtns[shown.indexOf(p0)].classList.add("active");
  makeImage();
}

autoMount(NAME, mount);
