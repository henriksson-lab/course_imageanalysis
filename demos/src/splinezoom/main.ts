// Image interpolation beyond bilinear (splines day, unit 4). A 128 × 128 image (the
// micrograph or a test pattern) and an interpolator: nearest, bilinear, cubic convolution
// (Keys, a = −½), cubic B-spline with the prefilter, or without it. Operations: zoom into a
// 16 × 16 patch ×8 (click the image to move the patch), or rotate by 36° n times (each
// rotation resamples the previous result), with the difference to a reference and its RMS.
// Below: a profile through the result (with the samples), and the frequency responses of
// the interpolators.
// <div data-demo="splinezoom" data-preset="impulse|noprefilter|rotate|zone|edge|labels"></div>
import { autoMount, htmlEl, slider, presets, button } from "../lib/ui";
import { Plot, plotFigure } from "../lib/plot";
import { Grey, createGrey, fromFunction, drawGrey } from "../lib/imageops";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { zonePlate, siemensStar, stepEdge } from "../lib/synth";
import { SplineInterp, INTERP_LABEL, sampler, rotateWith, interpResponse } from "../lib/splineimage";

const NAME = "splinezoom";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .7em; }
.qb-demo[data-demo="${NAME}"] .sz-left { flex: 1 1 620px; max-width: 700px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .sz-imgs { display: flex; gap: .5em; }
.qb-demo[data-demo="${NAME}"] .sz-imgs figure { margin: 0; flex: 1 1 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] figcaption { font-size: 17px; text-align: center; margin-bottom: .1em; }
.qb-demo[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
.qb-demo[data-demo="${NAME}"] .sz-orig canvas { cursor: crosshair; }
.qb-demo[data-demo="${NAME}"] .sz-plots { display: flex; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .sz-plots .qb-plot { flex: 1 1 0; max-width: none; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 1 380px; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 18px; white-space: pre-wrap; }
.qb-demo[data-demo="${NAME}"] .sz-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .sz-credit { font-size: 12px; color: var(--qb-muted); }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] figcaption { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .sz-left { flex: 0 0 650px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 440px; gap: .2em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .04em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { padding: .2em .45em; font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .sz-lab, .reveal .qb-demo[data-demo="${NAME}"] .qb-note { display: none; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

const N = 128, PATCH = 16, ZOOM = 8;
type ImgId = "cell" | "zone" | "star" | "edge" | "impulse" | "labels";
type Op = "zoom" | "rotate";
interface State { img: ImgId; op: Op; interp: SplineInterp; n: number; px: number; py: number }
const DEFAULT: State = { img: "cell", op: "zoom", interp: "bspline", n: 10, px: 56, py: 56 };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "impulse", label: "impulse: who interpolates?", s: { img: "impulse", op: "zoom", interp: "bspline", px: 56, py: 56 } },
  { id: "noprefilter", label: "prefilter off: blur", s: { img: "impulse", op: "zoom", interp: "bspline-nopre", px: 56, py: 56 } },
  { id: "rotate", label: "rotate 10 × 36°", s: { img: "cell", op: "rotate", interp: "bilinear", n: 10 } },
  { id: "zone", label: "zone plate: aliasing", s: { img: "zone", op: "zoom", interp: "nearest", px: 104, py: 56 } },
  { id: "edge", label: "edge: overshoot of the cubics", s: { img: "edge", op: "zoom", interp: "keys", px: 56, py: 56 } },
  { id: "labels", label: "label image: why nearest", s: { img: "labels", op: "zoom", interp: "bspline", px: 40, py: 46 } },
];
const INTERPS: SplineInterp[] = ["nearest", "bilinear", "keys", "bspline", "bspline-nopre"];
const COLS: Record<SplineInterp, string> = { nearest: "#7f7f7f", bilinear: "#2ca02c", keys: "#ff7f0e", bspline: "#1f77b4", "bspline-nopre": "#9467bd" };
const LABEL_VALUES = [0, 80, 160, 240];

function makeImage(id: ImgId, cell: Grey | null): Grey {
  switch (id) {
    case "cell": return cell ?? createGrey(N, N, 128);
    case "zone": return zonePlate(N, 0, 255);
    case "star": return siemensStar(N, 18, 0, 255);
    case "edge": return stepEdge(N, N, 0.35, 50, 200, 0);
    case "impulse": { const g = createGrey(N, N, 0); g.data[64 * N + 64] = 255; return g; }
    case "labels": return fromFunction(N, N, (x, y) => {
      if ((x - 44) ** 2 + (y - 50) ** 2 < 15 ** 2) return LABEL_VALUES[1];
      if (x > 58 && x < 100 && y > 30 && y < 72) return LABEL_VALUES[3];
      if (y > 80 && y < 110 && x > 20 && x < 90) return LABEL_VALUES[2];
      return 0;
    });
  }
}

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st: State = { ...DEFAULT, ...p0.s };
  let cell: Grey | null = null;

  const left = htmlEl("div", "sz-left", "", host);
  const imgs = htmlEl("div", "sz-imgs", "", left);
  const fo = htmlEl("figure", "sz-orig", "", imgs);
  const capO = htmlEl("figcaption", "", "original (click: patch)", fo);
  const cvO = htmlEl("canvas", "", "", fo);
  const fr = htmlEl("figure", "", "", imgs);
  const capR = htmlEl("figcaption", "", "result", fr);
  const cvR = htmlEl("canvas", "", "", fr);
  const fd = htmlEl("figure", "", "", imgs);
  const capD = htmlEl("figcaption", "", "difference", fd);
  const cvD = htmlEl("canvas", "", "", fd);
  const plots = htmlEl("div", "sz-plots", "", left);
  const fpr = plotFigure(plots, "");
  const prof = new Plot(fpr, { width: 340, height: 200, x: [0, 16], y: [-40, 280], xlabel: "x (pixels)", margin: { l: 52 } });
  const ffr = plotFigure(plots, "");
  const freq = new Plot(ffr, { width: 340, height: 200, x: [0, 2 * Math.PI], y: [-0.15, 1.2], xlabel: "frequency ω", xticks: [0, Math.PI, 2 * Math.PI], xfmt: v => (v === 0 ? "0" : v < 4 ? "π" : "2π"), yticks: [0, 0.5, 1], margin: { l: 52 } });
  const credit = htmlEl("div", "sz-credit", CELL_CREDIT, left);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "sz-lab", "presets", panel);
  const pbtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p.s) })));
  htmlEl("div", "sz-lab", "image", panel);
  const imgBtns = presets(panel, (["cell", "zone", "star", "edge", "impulse", "labels"] as ImgId[]).map(i => ({ label: i === "cell" ? "micrograph" : i === "zone" ? "zone plate" : i === "star" ? "Siemens star" : i === "labels" ? "label image" : i, apply: () => { st.img = i; run(); } })));
  htmlEl("div", "sz-lab", "interpolator", panel);
  const intBtns = presets(panel, INTERPS.map(m => ({ label: INTERP_LABEL[m], apply: () => { st.interp = m; run(); } })));
  intBtns.forEach((b, i) => { b.style.boxShadow = `inset 5px 0 0 ${COLS[INTERPS[i]]}`; });   // colour key of the curves
  htmlEl("div", "sz-lab", "operation", panel);
  const opRow = htmlEl("div", "qb-presets", "", panel);
  const opBtns = (["zoom", "rotate"] as Op[]).map(o => button(opRow, o === "zoom" ? "zoom ×8" : "rotate 36° n times", () => { st.op = o; run(); }));
  let timer = 0;
  const nS = slider(panel, "n rotations", 1, 20, 1, st.n, v => { st.n = v; st.op = "rotate"; clearTimeout(timer); timer = window.setTimeout(run, 150); });
  button(panel, "reset", () => load(p0.s));
  const ro = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Click the original to move the zoom patch.", panel);

  for (const c of [cvO, cvR, cvD]) { c.width = N; c.height = N; }
  cvO.addEventListener("click", ev => {
    const r = cvO.getBoundingClientRect();
    st.px = Math.max(0, Math.min(N - PATCH, Math.round((ev.clientX - r.left) / r.width * N - PATCH / 2)));
    st.py = Math.max(0, Math.min(N - PATCH, Math.round((ev.clientY - r.top) / r.height * N - PATCH / 2)));
    st.op = "zoom";
    pbtns.forEach(b => b.classList.remove("active"));
    run();
  });

  // frequency responses (static apart from the highlight)
  freq.vline(Math.PI, { color: "var(--qb-muted, #888)", width: 1.5, text: "Nyquist" });
  freq.line([0, Math.PI, Math.PI, 2 * Math.PI], [1, 1, 0, 0], { color: "var(--qb-fg,#222)", width: 1.5, dash: "4 4" });
  const fm = INTERPS.map(m => {
    const xs = Array.from({ length: 200 }, (_, k) => 2 * Math.PI * k / 199);
    return freq.line(xs, xs.map(w => interpResponse(m, w)), { color: COLS[m], width: 2 });
  });

  function load(s: Partial<State>): void {
    Object.assign(st, DEFAULT, s);
    nS.set(st.n);
    pbtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i].s === s));
    run();
  }

  function run(): void {
    if (st.img === "cell" && !cell) { loadCell(N).then(g => { cell = g; run(); }); }
    credit.style.visibility = st.img === "cell" ? "visible" : "hidden";
    imgBtns.forEach((b, i) => b.classList.toggle("active", ["cell", "zone", "star", "edge", "impulse", "labels"][i] === st.img));
    intBtns.forEach((b, i) => b.classList.toggle("active", INTERPS[i] === st.interp));
    opBtns.forEach((b, i) => b.classList.toggle("active", ["zoom", "rotate"][i] === st.op));
    fm.forEach((m, i) => { m.style.width = INTERPS[i] === st.interp ? 5 : 2; m.render(); });
    const img = makeImage(st.img, cell);
    const disp = { mode: "clip" as const, lo: 0, hi: 255 };
    drawGrey(cvO, img, disp);
    const ctx = cvO.getContext("2d")!;
    fd.style.visibility = st.op === "rotate" ? "visible" : "hidden";
    prof.clear();
    const t0 = performance.now();
    if (st.op === "zoom") {
      ctx.strokeStyle = "#ff3b30"; ctx.lineWidth = 1.5;
      ctx.strokeRect(st.px - 0.5, st.py - 0.5, PATCH + 1, PATCH + 1);
      const s = sampler(img, st.interp), out = createGrey(N, N);
      for (let v = 0; v < N; v++) for (let u = 0; u < N; u++) out.data[v * N + u] = s(st.px + (u + 0.5) / ZOOM - 0.5, st.py + (v + 0.5) / ZOOM - 0.5);
      drawGrey(cvR, out, disp);
      capR.textContent = `${INTERP_LABEL[st.interp]}, ×8`;
      // profile along the middle row of the patch
      const yy = st.py + PATCH / 2, xs: number[] = [], ys: number[] = [];
      for (let k = 0; k <= 15 * 16; k++) { const x = k / 16; xs.push(x); ys.push(s(st.px + x, yy)); }
      const sx = Array.from({ length: 16 }, (_, k) => k), sy = sx.map(k => img.data[yy * N + st.px + k]);
      const lo = Math.min(0, ...ys), hi = Math.max(255, ...ys);
      prof.setDomain([0, 15], [lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)]);
      prof.line(xs, ys, { color: COLS[st.interp], width: 3 });
      prof.scatter(sx, sy, { color: "var(--qb-fg,#222)", r: 4.5 });
      const ms = Math.min(...out.data), Ms = Math.max(...out.data);
      let off = 0;
      if (st.img === "labels") for (const v of out.data) if (!LABEL_VALUES.some(l => Math.abs(v - l) < 0.5)) off++;
      // value halfway between the two middle samples of the profile
      const half = s(st.px + 7.5, yy);
      ro.textContent = `range of the result ${ms.toFixed(1)} … ${Ms.toFixed(1)}` + (Ms > 255.5 || ms < -0.5 ? "  (overshoot)" : "") +
        `\nat the samples: ${s(st.px + 8, yy).toFixed(1)} (sample ${img.data[yy * N + st.px + 8].toFixed(1)})\nhalfway (x + ½): ${half.toFixed(1)}` +
        (st.img === "labels" ? `\nnot a label value: ${(100 * off / (N * N)).toFixed(1)} % of the pixels` : "") +
        `\ntime ${(performance.now() - t0).toFixed(0)} ms`;
    } else {
      let cur = img;
      const step = 36 * Math.PI / 180;
      for (let k = 0; k < st.n; k++) cur = rotateWith(cur, step, st.interp);
      const total = st.n * 36;
      const ref = total % 360 === 0 ? img : rotateWith(img, total * Math.PI / 180, "bspline");
      drawGrey(cvR, cur, disp);
      capR.textContent = `${st.n} × 36° = ${total}°`;
      capD.textContent = total % 360 === 0 ? "minus the original" : "minus one rotation";
      const diff = createGrey(N, N);
      let ss = 0, cnt = 0;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const i = y * N + x, d = cur.data[i] - ref.data[i];
        diff.data[i] = d;
        if ((x - 63.5) ** 2 + (y - 63.5) ** 2 < (0.42 * N) ** 2) { ss += d * d; cnt++; }
      }
      drawGrey(cvD, diff, { mode: "mid" });
      const xs = Array.from({ length: N }, (_, k) => k);
      prof.setDomain([0, N - 1], [-40, 280]);
      prof.line(xs, xs.map(x => ref.data[64 * N + x]), { color: "var(--qb-muted,#999)", width: 2, dash: "4 3" });
      prof.line(xs, xs.map(x => cur.data[64 * N + x]), { color: COLS[st.interp], width: 2.5 });
      ro.textContent = `RMS difference ${Math.sqrt(ss / cnt).toFixed(2)} grey levels\n(inside the central disc)\ntime ${(performance.now() - t0).toFixed(0)} ms`;
    }
    capO.textContent = st.op === "zoom" ? "original (click: patch)" : "original";
  }

  load(p0.s);
}

autoMount(NAME, mount);
