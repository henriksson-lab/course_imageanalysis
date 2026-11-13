// Filling holes. An image with known truth (test card, micrograph, a dot texture, a broken dark
// bar, a field of synthetic nuclei) and a mask of missing pixels (presets, or paint your own
// with the mouse on the left panel in "paint" mode). Methods:
//  - harmonic (diffusion): Jacobi or Gauss–Seidel sweeps (play, with a counter) or a direct solve;
//  - TV: primal–dual iterations that minimise the total variation inside the hole (started
//    from the diffusion fill);
//  - exemplar: patch by patch, highest priority first, copying the best-matching known patch
//    (the target patch and an arrow from its source are drawn; the fill order is shown).
// Right panel: the error inside the hole (|result − truth|), or the fill order for exemplar.
//
// <div data-demo="inpaint" data-preset="column|large|jacobi|tvedge|barjoin|barbreak|texture|nucleus|text"></div>
import { autoMount, htmlEl, button, slider, presets as presetRow } from "../lib/ui";
import { CELL_CREDIT } from "../lib/cell";
import { harmonicJacobi, harmonicGaussSeidel, harmonicSolve, TvInpaint, exemplarInpaint, initHole, ExemplarStep } from "../lib/inpaint";
import { testCard } from "../lib/pde";
import { ImgView, injectStyle } from "../canny/imgview";
import { choice, cell128, PS_CSS, animator } from "../heatflow/common";
import { rng } from "../lib/random";
import { scalarToRGBA, viridis } from "../lib/colour";

const NAME = "inpaint";
const STYLE = PS_CSS + `
.qb-demo[data-demo="${NAME}"] { gap: .8em; flex-wrap: wrap; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .ip-main { flex: 1 1 520px; min-width: 0; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .ip-row { display: flex; gap: .7em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .ip-row > * { flex: 1 1 0; min-width: 0; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 300px; max-width: 460px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre-wrap; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .06em .38em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 345px; gap: .15em; }
.reveal .qb-demo[data-demo="${NAME}"] .ip-main { flex: 0 0 770px; }
.reveal .qb-demo[data-demo="${NAME}"] .ip-row > * { max-width: 370px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 16px; padding: .2em .45em; }
@media (max-width: 860px) { .qb-demo[data-demo="${NAME}"] { flex-wrap: wrap; } .qb-demo[data-demo="${NAME}"] .ip-row { flex-wrap: wrap; } }
`;

type Img = "card" | "cell" | "texture" | "bar" | "nuclei";
type MaskKind = "column" | "scratch" | "text" | "small" | "large" | "medium" | "bargap" | "texhole" | "nucleushole" | "none";
type Method = "jacobi" | "gs" | "tv" | "exemplar";
interface State { image: Img; mask: MaskKind; method: Method; gap: number; patch: number; run: number | "solve" }
const DEFAULT: State = { image: "card", mask: "large", method: "jacobi", gap: 6, patch: 4, run: 0 };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "column", label: "dead column: diffusion is enough", s: { image: "cell", mask: "column", method: "gs", run: "solve" } },
  { id: "large", label: "large hole: diffusion blurs", s: { image: "card", mask: "large", method: "gs", run: "solve" } },
  { id: "jacobi", label: "Jacobi needs ~w² sweeps", s: { image: "card", mask: "medium", method: "jacobi", run: 0 } },
  { id: "tvedge", label: "TV continues edges", s: { image: "card", mask: "large", method: "tv", run: 1500 } },
  { id: "barjoin", label: "bar: gap < width connects", s: { image: "bar", mask: "bargap", gap: 6, method: "tv", run: 2500 } },
  { id: "barbreak", label: "bar: gap > width breaks", s: { image: "bar", mask: "bargap", gap: 16, method: "tv", run: 2500 } },
  { id: "texture", label: "texture: exemplar wins", s: { image: "texture", mask: "texhole", method: "exemplar", run: 9999 } },
  { id: "nucleus", label: "exemplar invents a nucleus", s: { image: "nuclei", mask: "nucleushole", method: "exemplar", run: 9999 } },
  { id: "text", label: "text over the micrograph", s: { image: "cell", mask: "text", method: "gs", run: "solve" } },
];
const BAR_W = 10;

function makeImage(kind: Img, cell: Float64Array | null): { w: number; h: number; truth: Float64Array } {
  if (kind === "cell" && cell) return { w: 128, h: 128, truth: cell };
  if (kind === "bar") {
    const w = 128, h = 64, t = new Float64Array(w * h).fill(200);
    for (let y = 27; y < 27 + BAR_W; y++) for (let x = 0; x < w; x++) t[y * w + x] = 50;
    return { w, h, truth: t };
  }
  if (kind === "texture") {
    const w = 128, h = 128, t = new Float64Array(w * h), r = rng(3);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const a = Math.hypot(((x % 8) + 8) % 8 - 3.5, ((y % 8) + 8) % 8 - 3.5);
      t[y * w + x] = 70 + (a < 2.5 ? 110 : 0) + 25 * Math.sin(2 * Math.PI * (x + y) / 16) + 4 * r.norm();
    }
    return { w, h, truth: t };
  }
  if (kind === "nuclei") {
    const w = 128, h = 128, t = new Float64Array(w * h), r = rng(5);
    const nuc: [number, number, number][] = [[30, 30, 12], [80, 26, 10], [104, 70, 12], [36, 92, 11], [76, 104, 9]];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let v = 60 + 6 * r.norm();
      for (const [cx, cy, rr] of nuc) { const d = Math.hypot(x - cx, y - cy); if (d < rr) v = 170 + 25 * Math.cos(d / 2) + 6 * r.norm(); else if (d < rr + 1.5) v = 120; }
      t[y * w + x] = v;
    }
    return { w, h, truth: t };
  }
  return { w: 128, h: 128, truth: testCard().truth };
}

function makeMask(kind: MaskKind, w: number, h: number, gap: number): Uint8Array {
  const m = new Uint8Array(w * h);
  const rect = (x0: number, y0: number, x1: number, y1: number) => { for (let y = Math.max(0, y0); y < Math.min(h, y1); y++) for (let x = Math.max(0, x0); x < Math.min(w, x1); x++) m[y * w + x] = 1; };
  switch (kind) {
    case "column": rect(70, 0, 71, h); break;
    case "scratch": for (let t = 0; t < 1; t += 0.002) { const x = Math.round(10 + 108 * t), y = Math.round(20 + 90 * t + 8 * Math.sin(6 * t)); rect(x, y, x + 2, y + 2); } break;
    case "small": rect(28, 28, 40, 40); break;
    case "medium": rect(30, 14, 46, 30); break;
    case "large": rect(24, 18, 56, 50); break;
    case "bargap": rect(64 - Math.floor(gap / 2), 12, 64 - Math.floor(gap / 2) + gap, 52); break;
    case "texhole": rect(48, 48, 76, 76); break;
    case "nucleushole": rect(52, 48, 80, 76); break;
    case "text": textMask(m, w, h); break;
  }
  return m;
}

/** "10 µm" and a scale bar burnt into the bottom right, rendered with the canvas font. */
function textMask(m: Uint8Array, w: number, h: number): void {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const g = c.getContext("2d");
  if (!g) return;
  g.fillStyle = "#fff"; g.font = "bold 15px sans-serif"; g.textBaseline = "alphabetic";
  g.fillText("10 µm", 72, 108);
  g.fillRect(70, 114, 50, 4);
  const d = g.getImageData(0, 0, w, h).data;
  for (let i = 0; i < w * h; i++) if (d[4 * i + 3] > 60) m[i] = 1;
}

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(p0?.s ?? {}) });
  const st: State = start();

  const main = htmlEl("div", "ip-main", "", host);
  const row = htmlEl("div", "ip-row", "", main);
  const viewR = new ImgView(row, "result"), viewE = new ImgView(row, "error in the hole");
  const ro = htmlEl("div", "qb-readout", "", main);
  const credit = htmlEl("div", "ps-credit", CELL_CREDIT, main);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "ps-lab", "presets", panel);
  const presetBtns = presetRow(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const setImg = choice(panel, "image", [["card", "card"], ["cell", "micrograph"], ["texture", "texture"], ["bar", "bar"], ["nuclei", "nuclei"]], id => { st.image = id as Img; st.mask = defaultMask(st.image); setup(); });
  const setMask = choice(panel, "hole", [["column", "column"], ["scratch", "scratch"], ["text", "text"], ["small", "small"], ["large", "large"], ["none", "paint"]], id => { st.mask = id as MaskKind; setup(); });
  const gapS = slider(panel, "bar gap", 1, 24, 1, st.gap, v => { st.gap = v; if (st.image === "bar") { st.mask = "bargap"; setup(); } });
  const setMethod = choice(panel, "method", [["jacobi", "Jacobi"], ["gs", "Gauss–Seidel"], ["tv", "TV"], ["exemplar", "exemplar"]], id => { st.method = id as Method; resetMethod(); draw(); });
  const patchS = slider(panel, "exemplar patch", 2, 7, 1, st.patch, v => { st.patch = v; if (st.method === "exemplar") { resetMethod(); draw(); } }, v => `${2 * v + 1}`);
  const ctl = htmlEl("div", "qb-presets", "", panel);
  button(ctl, "step", () => { anim.stop(); playB.textContent = "play"; advance(1); draw(); });
  const playB = button(ctl, "play", () => { if (anim.running()) { anim.stop(); playB.textContent = "play"; } else { anim.start(); playB.textContent = "pause"; } });
  button(ctl, "solve / finish", () => { anim.stop(); playB.textContent = "play"; finish(); draw(); });
  button(ctl, "restart", () => { anim.stop(); playB.textContent = "play"; resetMethod(); draw(); });
  button(panel, "reset all", () => { anim.stop(); playB.textContent = "play"; Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); setup(true); });

  const anim = animator(() => {
    advance(st.method === "jacobi" || st.method === "gs" ? 10 : st.method === "tv" ? 40 : 1);
    draw();
    if (st.method === "exemplar" && exStep >= (ex?.steps.length ?? 0)) { playB.textContent = "play"; return false; }
    if (iters > 20000) { playB.textContent = "play"; return false; }
    return true;
  });

  // ---- state ----
  let cellImg: Float64Array | null = null;
  let W = 128, H = 128, truth: Float64Array = new Float64Array(W * H), mask: Uint8Array = new Uint8Array(W * H);
  let u: Float64Array = new Float64Array(0), iters = 0, tvState: TvInpaint | null = null;
  let ex: { u: Float64Array; order: Int32Array; steps: ExemplarStep[] } | null = null, exStep = 0, solved = false;

  function defaultMask(img: Img): MaskKind { return img === "bar" ? "bargap" : img === "texture" ? "texhole" : img === "nuclei" ? "nucleushole" : img === "cell" ? "column" : "large"; }
  function setup(runPreset = false): void {
    anim.stop(); playB.textContent = "play";
    if (st.image === "cell" && !cellImg) { cell128().then(g => { cellImg = Float64Array.from(g.data); setup(runPreset); }); return; }
    const I = makeImage(st.image, cellImg); W = I.w; H = I.h; truth = I.truth;
    if (st.mask !== "none") mask = makeMask(st.mask, W, H, st.gap);
    else if (mask.length !== W * H) mask = new Uint8Array(W * H);
    sync();
    resetMethod();
    if (runPreset) { if (st.run === "solve") finish(); else if (st.run) advance(st.run); }
    draw();
  }
  function resetMethod(): void {
    iters = 0; solved = false; exStep = 0; ex = null; tvState = null;
    u = initHole(truth, mask, 0);
    let s = 0, n = 0;
    for (let i = 0; i < W * H; i++) if (!mask[i]) { s += truth[i]; n++; }
    const mean = n ? s / n : 0;
    if (st.method === "tv") { u = mask.some(v => v) ? harmonicSolve(truth, mask, W, H, 1e-6, 3000).u : initHole(truth, mask, mean); tvState = new TvInpaint(truth, mask, W, H, u); }
    if (st.method === "exemplar") ex = exemplarInpaint(truth, mask, W, H, { patch: st.patch, search: 32 });
  }
  function advance(n: number): void {
    if (st.method === "jacobi") for (let k = 0; k < n; k++) { u = harmonicJacobi(u, mask, W, H); iters++; }
    else if (st.method === "gs") for (let k = 0; k < n; k++) { harmonicGaussSeidel(u, mask, W, H); iters++; }
    else if (st.method === "tv" && tvState) { u = Float64Array.from(tvState.run(n)); iters += n; }
    else if (ex) exStep = Math.min(ex.steps.length, exStep + n);
  }
  function finish(): void {
    if (st.method === "jacobi" || st.method === "gs") { const r = harmonicSolve(truth, mask, W, H); u = r.u; solved = true; iters = r.iters; }
    else if (st.method === "tv") advance(Math.max(0, 3000 - iters));
    else if (ex) exStep = ex.steps.length;
  }
  function load(p: Preset): void { Object.assign(st, DEFAULT, p.s); setup(true); }
  function sync(): void { setImg(st.image); setMask(st.mask); gapS.set(st.gap); setMethod(st.method); patchS.set(st.patch); }

  // paint your own hole
  viewR.onPick((x, y) => {
    if (st.mask !== "none") return;
    for (let j = -2; j <= 2; j++) for (let i = -2; i <= 2; i++) if (i * i + j * j <= 5) { const xx = x + i, yy = y + j; if (xx >= 0 && yy >= 0 && xx < W && yy < H) mask[yy * W + xx] = 1; }
    resetMethod(); draw();
  });

  function current(): Float64Array {
    if (st.method !== "exemplar" || !ex) return u;
    const out = Float64Array.from(truth);
    for (let p = 0; p < W * H; p++) if (mask[p]) out[p] = ex.order[p] >= 0 && ex.order[p] < exStep ? ex.u[p] : NaN;
    return out;
  }
  function draw(): void {
    sync();
    gapS.input.parentElement!.style.display = st.image === "bar" ? "" : "none";
    patchS.input.parentElement!.style.display = st.method === "exemplar" ? "" : "none";
    credit.style.display = st.image === "cell" ? "" : "none";
    const cur = current();
    // result: grey, unfilled hole pixels in magenta
    const px = new Uint8ClampedArray(4 * W * H);
    let err = 0, cnt = 0, nHole = 0;
    for (let p = 0; p < W * H; p++) {
      const v = cur[p];
      if (Number.isNaN(v)) { px[4 * p] = 220; px[4 * p + 1] = 60; px[4 * p + 2] = 200; }
      else { const g = Math.max(0, Math.min(255, v)); px[4 * p] = px[4 * p + 1] = px[4 * p + 2] = g; }
      px[4 * p + 3] = 255;
      if (mask[p]) { nHole++; if (!Number.isNaN(v)) { err += Math.abs(v - truth[p]); cnt++; } }
    }
    viewR.rgba(px, W, H);
    viewR.clear();
    // outline the hole
    for (let p = 0; p < W * H; p++) if (mask[p]) {
      const x = p % W, y = (p - x) / W;
      if (x === 0 || !mask[p - 1]) viewR.add("line", { x1: x, y1: y, x2: x, y2: y + 1, stroke: "#22d3ee", "stroke-width": 1.5 });
      if (x === W - 1 || !mask[p + 1]) viewR.add("line", { x1: x + 1, y1: y, x2: x + 1, y2: y + 1, stroke: "#22d3ee", "stroke-width": 1.5 });
      if (y === 0 || !mask[p - W]) viewR.add("line", { x1: x, y1: y, x2: x + 1, y2: y, stroke: "#22d3ee", "stroke-width": 1.5 });
      if (y === H - 1 || !mask[p + W]) viewR.add("line", { x1: x, y1: y + 1, x2: x + 1, y2: y + 1, stroke: "#22d3ee", "stroke-width": 1.5 });
    }
    let info = "";
    if (st.method === "exemplar" && ex) {
      const P = st.patch;
      if (exStep > 0 && exStep <= ex.steps.length) {
        const s = ex.steps[exStep - 1];
        viewR.box(s.tx, s.ty, P, "#ffd400", 2);
        if (s.sx >= 0) { viewR.box(s.sx, s.sy, P, "#ff9f1c", 2); viewR.arrow(s.sx + 0.5, s.sy + 0.5, s.tx + 0.5, s.ty + 0.5, "#ff9f1c", 2.5); }
        info = `step ${exStep} of ${ex.steps.length}: target (${s.tx}, ${s.ty}), copied from (${s.sx}, ${s.sy}); priority = confidence ${s.confidence.toFixed(2)} × data ${s.data.toFixed(3)}`;
      } else info = `${ex.steps.length} patches will fill the hole; press step or play`;
      // fill order map
      const ord = new Float64Array(W * H).fill(NaN);
      for (let p = 0; p < W * H; p++) if (mask[p] && ex.order[p] < exStep) ord[p] = ex.order[p];
      const rgba = scalarToRGBA(ord, { map: viridis, lo: 0, hi: Math.max(1, ex.steps.length - 1) });
      for (let p = 0; p < W * H; p++) if (!mask[p]) { const g = truth[p] * 0.35; rgba[4 * p] = rgba[4 * p + 1] = rgba[4 * p + 2] = g; rgba[4 * p + 3] = 255; } else if (Number.isNaN(ord[p])) { rgba[4 * p] = 30; rgba[4 * p + 1] = 30; rgba[4 * p + 2] = 30; rgba[4 * p + 3] = 255; }
      viewE.rgba(rgba, W, H);
      viewE.setCaption("fill order (dark blue first, yellow last)");
    } else {
      const e = new Uint8ClampedArray(4 * W * H);
      for (let p = 0; p < W * H; p++) {
        const g = mask[p] ? Math.min(255, 4 * Math.abs(cur[p] - truth[p])) : 0;
        e[4 * p] = g; e[4 * p + 1] = mask[p] ? g * 0.6 : 0; e[4 * p + 2] = mask[p] ? 40 : 0; e[4 * p + 3] = 255;
        if (!mask[p]) { const q = truth[p] * 0.25; e[4 * p] = e[4 * p + 1] = e[4 * p + 2] = q; }
      }
      viewE.rgba(e, W, H);
      viewE.setCaption("error in the hole |result − truth| (bright = large)");
      info = st.method === "tv" ? `TV: ${iters} iterations` : solved ? `direct solve (conjugate gradients: ${iters} iterations)` : `${st.method === "jacobi" ? "Jacobi" : "Gauss–Seidel"}: ${iters} sweeps`;
    }
    viewR.setCaption(st.mask === "none" ? "paint a hole (drag), then pick a method" : "result (cyan: the hole)");
    ro.textContent = `${nHole} missing pixels · ${info}\nmean |error| in the hole: ${cnt ? (err / cnt).toFixed(1) : "–"} grey levels` +
      (st.image === "bar" ? `   ·   bar width ${BAR_W}, gap ${st.gap}: centre of the gap ${cur[32 * W + 64].toFixed(0)} (bar 50, background 200)` : "");
  }

  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  setup(true);
}

autoMount(NAME, mount);
