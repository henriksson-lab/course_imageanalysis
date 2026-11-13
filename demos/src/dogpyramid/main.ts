// The DoG pyramid, step by step. A 192 × 192 synthetic image (discs and squares of three
// sizes on a gentle ramp, a long bar, seeded noise) or a crop of the micrograph. Left: the
// octave pyramid, one row per octave (DoGs, or the blurred images), built level by level.
// Middle: the selected DoG level, large (blue negative, red positive, 0 = white), with the
// candidates of that level by stage: grey = unstable (refinement left the cube), blue x = too
// weak (contrast), orange = on an edge (tr²/det too large), green circle = kept (radius 2σ).
// Click a candidate: its 3 × 3 × 3 neighbourhood, the sub-pixel offset, the refined (x, y, σ),
// the Hessian test and the verdict.
//
// <div data-demo="dogpyramid" data-preset="octave|bar|edge|noise|s1|refine" data-image="synthetic|cell"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { Grey, cropGrey } from "../lib/imageops";
import { diverging, scalarToRGBA } from "../lib/colour";
import { rng } from "../lib/random";
import * as K from "../lib/keypoints";
import { loadCell, CELL_CREDIT } from "../lib/cell";
import { ImgView, injectStyle } from "../canny/imgview";
import { makeScene } from "./scene";

const NAME = "dogpyramid";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .8em; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] .dp-pyr { flex: 0 1 300px; min-width: 220px; display: flex; flex-direction: column; gap: .25em; }
.qb-demo[data-demo="${NAME}"] .dp-row { display: flex; gap: 3px; align-items: center; }
.qb-demo[data-demo="${NAME}"] .dp-row span { font-size: 14px; width: 2.6em; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .dp-row canvas { image-rendering: pixelated; cursor: pointer; border: 2px solid transparent; border-radius: 3px; background: #ddd; }
.qb-demo[data-demo="${NAME}"] .dp-row canvas.sel { border-color: #ffb000; }
.qb-demo[data-demo="${NAME}"] .dp-main { flex: 0 1 380px; min-width: 240px; display: flex; flex-direction: column; gap: .2em; }
.qb-demo[data-demo="${NAME}"] .dp-inset svg { width: 100%; display: block; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 340px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .dp-key { font-size: 14px; color: var(--qb-muted); }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .dp-pyr { flex: 0 0 300px; }
.reveal .qb-demo[data-demo="${NAME}"] .dp-main { flex: 0 0 360px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 420px; max-width: 420px; gap: .12em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .05em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .2em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .dp-row span { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] .dp-key { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] .ev-credit { display: none; }
@media (max-width: 900px) { .qb-demo[data-demo="${NAME}"] { flex-wrap: wrap; } }
`;

interface State { image: "synthetic" | "cell"; S: number; contrast: number; r: number; up: boolean; noise: number; showGauss: boolean; so: number; sl: number; built: number; sel: number }
const DEFAULT: State = { image: "synthetic", S: 3, contrast: 3, r: 10, up: false, noise: 2, showGauss: false, so: 0, sl: 1, built: -1, sel: -1 };
interface Preset { id: string; label: string; s: Partial<State>; pick?: "bar" | "refine" | "strong" }
const PRESETS: Preset[] = [
  { id: "octave", label: "one octave, three levels", s: { built: 5, so: 0, sl: 1 }, pick: "strong" },
  { id: "bar", label: "the bar gives edge candidates", s: { r: 51, so: 0, sl: 1 }, pick: "bar" },
  { id: "edge", label: "edge test removes them", s: { r: 10, so: 0, sl: 1 }, pick: "bar" },
  { id: "noise", label: "contrast threshold vs noise", s: { noise: 6, contrast: 1, so: 0, sl: 1 } },
  { id: "s1", label: "S = 1: blobs between levels are missed", s: { S: 1, so: 0, sl: 1 } },
  { id: "refine", label: "refinement: sub-pixel offset", s: { so: 0, sl: 1 }, pick: "refine" },
];
const STAGE_COLOR: Record<K.Stage, string> = { unstable: "#9aa0a6", contrast: "#5b8ff9", edge: "#ff9f1c", kept: "#2ca02c" };
const STAGE_TEXT: Record<K.Stage, string> = { unstable: "unstable: left the cube", contrast: "too weak (|D̂| < threshold)", edge: "on an edge: rejected", kept: "kept" };

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, ...(host.dataset.image === "cell" ? { image: "cell" as const, noise: 0 } : {}), ...(p0?.s ?? {}) });
  const st: State = start();

  const pyrDiv = htmlEl("div", "dp-pyr", "", host);
  const pyrHead = htmlEl("div", "dp-key", "", pyrDiv);
  const rowsDiv = htmlEl("div", "", "", pyrDiv);
  const stepRow = htmlEl("div", "qb-presets", "", pyrDiv);
  button(stepRow, "restart", () => { st.built = 1; st.sel = -1; draw(); });
  button(stepRow, "step", () => { st.built = st.built < 0 ? total() : Math.min(total(), st.built + 1); draw(); });
  button(stepRow, "all", () => { st.built = -1; draw(); });
  const gC = checkbox(pyrDiv, "show blurred images", st.showGauss, v => { st.showGauss = v; draw(); });
  const credit = htmlEl("div", "ev-credit", CELL_CREDIT, pyrDiv);

  const main = htmlEl("div", "dp-main", "", host);
  const view = new ImgView(main, "");
  const insetDiv = htmlEl("div", "dp-inset", "", main);
  const inset = svgEl("svg", { viewBox: "0 0 352 130" }, insetDiv);

  const panel = htmlEl("div", "qb-panel", "", host);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, p.s); pickWhat = p.pick ?? null; sync(); makeImage(); } })));
  const imgBtns = presets(panel, [
    { label: "test image (synthetic)", apply: () => { st.image = "synthetic"; st.noise = 2; sync(); makeImage(); } },
    { label: "micrograph", apply: () => { st.image = "cell"; st.noise = 0; sync(); makeImage(); } },
  ]);
  const sS = slider(panel, "steps per octave S", 1, 5, 1, st.S, v => { st.S = v; st.sl = Math.min(st.sl, v); st.built = -1; st.sel = -1; run(); });
  const cS = slider(panel, "contrast threshold", 0, 12, 0.5, st.contrast, v => { st.contrast = v; run(); });
  const rS = slider(panel, "edge ratio r", 1, 51, 1, st.r, v => { st.r = v; run(); }, v => v > 50 ? "off" : String(v));
  const nS = slider(panel, "noise", 0, 12, 0.5, st.noise, v => { st.noise = v; makeImage(); });
  const uC = checkbox(panel, "upsample the first octave 2×", st.up, v => { st.up = v; st.so = 0; st.sel = -1; run(); });
  button(panel, "reset", () => { Object.assign(st, start()); pickWhat = null; presetBtns.forEach(b => b.classList.remove("active")); sync(); makeImage(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  const key = htmlEl("div", "dp-key", "", panel);
  key.innerHTML = `<b style="color:${STAGE_COLOR.kept}">○</b> kept &nbsp; <b style="color:${STAGE_COLOR.edge}">○</b> edge &nbsp; <b style="color:${STAGE_COLOR.contrast}">×</b> weak &nbsp; <b style="color:${STAGE_COLOR.unstable}">×</b> unstable`;

  let pickWhat: Preset["pick"] | null = p0?.pick ?? null;
  function sync(): void { sS.set(st.S); cS.set(st.contrast); rS.set(st.r); nS.set(st.noise); uC.checked = st.up; gC.checked = st.showGauss; }

  let cell: Grey | null = null, img: Grey | null = null, res: K.DoGResult | null = null;
  function makeImage(): void {
    if (st.image === "cell") {
      if (!cell) { loadCell(256).then(g => { cell = cropGrey(g, 40, 30, 192, 192); makeImage(); }); return; }
      img = cell;
      if (st.noise > 0) { const r = rng(4); img = { width: 192, height: 192, data: Float32Array.from(cell.data, v => v + st.noise * r.norm()) }; }
    } else img = makeScene(192, st.noise);
    run();
  }
  function run(): void {
    if (!img) return;
    res = K.detectDoG(img, { S: st.S, contrast: st.contrast, edgeR: st.r > 50 ? 1e9 : st.r, upsample: st.up });
    st.so = Math.min(st.so, res.pyr.octaves.length - 1);
    if (pickWhat) { pick(pickWhat); pickWhat = null; }
    else if (st.sel >= res.records.length) st.sel = -1;
    draw();
  }
  function pick(what: NonNullable<Preset["pick"]>): void {
    const R = res!.records;
    let best = -1, score = -Infinity;
    R.forEach((rec, i) => {
      const c = rec.cand;
      if (what !== "refine" && (c.octave !== st.so || c.level !== st.sl)) return;
      let s = -Infinity;
      if (what === "bar") s = rec.stage === "edge" || (st.r > 50 && Math.abs(c.iy * res!.pyr.octaves[c.octave].step - 62) < 4 && rec.edge.ratio > 12.1) ? -Math.abs(c.ix * res!.pyr.octaves[c.octave].step - 96) : -Infinity;
      else if (what === "refine") s = rec.stage === "kept" ? -Math.abs(Math.hypot(rec.refined.offset[0], rec.refined.offset[1]) - 0.4) : -Infinity;
      else s = rec.stage === "kept" ? Math.abs(rec.refined.value) : -Infinity;
      if (s > score) { score = s; best = i; }
    });
    st.sel = best;
    if (best >= 0) { st.so = R[best].cand.octave; st.sl = R[best].cand.level; }
  }
  const total = () => res ? res.pyr.octaves.reduce((a, o) => a + o.dog.length, 0) : 0;
  const dogIndex = (o: number, s: number) => { let k = 0; for (let q = 0; q < o; q++) k += res!.pyr.octaves[q].dog.length; return k + s; };
  const builtOK = (o: number, s: number) => st.built < 0 || dogIndex(o, s) < st.built;

  view.onPick((x, y) => {
    if (!res) return;
    let best = -1, bd = 3.5;
    res.records.forEach((rec, i) => {
      const c = rec.cand;
      if (c.octave !== st.so || c.level !== st.sl || !builtOK(c.octave, c.level + 1)) return;
      const d = Math.hypot(c.ix - x, c.iy - y);
      if (d < bd) { bd = d; best = i; }
    });
    st.sel = best; draw();
  }, false);

  function thumb(parent: Element, g: Grey, size: number, sel: boolean, signed: boolean, onClick: () => void, visible: boolean): void {
    const c = htmlEl("canvas", sel ? "sel" : "", "", parent);
    c.width = g.width; c.height = g.height; c.style.width = c.style.height = size + "px";
    if (visible) {
      let m = 0; for (const v of g.data) m = Math.max(m, Math.abs(v));
      const px = signed ? scalarToRGBA(g.data, { map: diverging, lo: -m, hi: m }) : scalarToRGBA(g.data, { lo: 0, hi: 255 });
      const id = new ImageData(g.width, g.height); id.data.set(px); c.getContext("2d")!.putImageData(id, 0, 0);
    }
    c.addEventListener("click", onClick);
  }

  function draw(): void {
    if (!res || !img) return;
    imgBtns.forEach((b, i) => b.classList.toggle("active", (i === 1) === (st.image === "cell")));
    credit.style.display = st.image === "cell" ? "" : "none";
    const P = res.pyr;
    pyrHead.textContent = st.showGauss ? `blurred images: ${P.S + 3} per octave` : `DoGs: ${P.S + 2} per octave (extrema on levels 1–${P.S})`;
    rowsDiv.replaceChildren();
    const per = st.showGauss ? P.S + 3 : P.S + 2, size = Math.min(54, Math.floor(250 / per) - 4);
    P.octaves.forEach((oc, o) => {
      const row = htmlEl("div", "dp-row", "", rowsDiv);
      htmlEl("span", "", `${oc.dog[0].width}²`, row);
      const list = st.showGauss ? oc.gauss : oc.dog;
      list.forEach((g, s) => thumb(row, g, size, !st.showGauss && o === st.so && s === st.sl, !st.showGauss, () => {
        st.so = o; st.sl = st.showGauss ? Math.max(0, Math.min(s, P.S + 1)) : s; st.sel = -1; draw();
      }, builtOK(o, st.showGauss ? Math.max(0, s - 1) : s)));
    });
    // the selected level, large
    const oc = P.octaves[st.so], D = oc.dog[Math.min(st.sl, oc.dog.length - 1)];
    let m = 0; for (const v of D.data) m = Math.max(m, Math.abs(v));
    if (builtOK(st.so, st.sl)) view.map(D.data, D.width, D.height, diverging, -m, m);
    else view.rgba(new Uint8ClampedArray(4 * D.width * D.height).fill(220), D.width, D.height);
    const sAbs = K.levelSigma(P.sigma0, P.S, st.sl) * oc.step;
    view.setCaption(`octave ${st.so} (${D.width}²), DoG level ${st.sl}: σ = ${sAbs.toFixed(2)} px`);
    view.clear();
    const counts: Record<string, number> = { cand: 0, unstable: 0, contrast: 0, edge: 0, kept: 0 };
    const sc = D.width / 128;     // marker size in this level's pixels
    res.records.forEach((rec, i) => {
      const c = rec.cand;
      if (!builtOK(c.octave, c.level + 1)) return;
      counts.cand++; counts[rec.stage]++;
      if (c.octave !== st.so || c.level !== st.sl) return;
      const col = STAGE_COLOR[rec.stage], x = c.ix + 0.5, y = c.iy + 0.5, w = i === st.sel ? 3.5 : 2;
      if (rec.stage === "kept" || rec.stage === "edge") {
        const rr = Math.max(1.5 * sc, 2 * rec.refined.sigma / oc.step);
        view.add("circle", { cx: rec.refined.x / oc.step + 0.5, cy: rec.refined.y / oc.step + 0.5, r: rr, fill: "none", stroke: col, "stroke-width": w });
      } else {
        const h = 1.5 * sc;
        view.add("path", { d: `M${x - h},${y - h}L${x + h},${y + h}M${x - h},${y + h}L${x + h},${y - h}`, stroke: col, "stroke-width": w });
      }
      if (i === st.sel) view.box(c.ix, c.iy, 1, "#ffd400", 2.5);
    });
    drawInset();
    ro.textContent = `candidates: ${counts.cand} (unstable ${counts.unstable})\nweak ${counts.contrast} · edge ${counts.edge} · kept ${counts.kept}` + selText();
  }

  function selText(): string {
    if (st.sel < 0 || !res) return "\nclick a candidate in the large view";
    const rec = res.records[st.sel], f = rec.refined, c = rec.cand;
    const off = f.offset.map(v => (v >= 0 ? "+" : "") + v.toFixed(2)).join(", ");
    return `\nsample (${c.ix}, ${c.iy}), level ${c.level}: D = ${c.value.toFixed(2)}` +
      (f.moves ? ` (moved ${f.moves}×)` : "") +
      `\noffset ${off}\n→ (${f.x.toFixed(2)}, ${f.y.toFixed(2)}) px, σ ${f.sigma.toFixed(2)}, D̂ ${f.value.toFixed(2)}` +
      `\ntr²/det ${Number.isFinite(rec.edge.ratio) ? rec.edge.ratio.toFixed(1) : "∞ (det ≤ 0)"} (limit ${st.r > 50 ? "off" : rec.edge.limit.toFixed(1)})\n→ ${STAGE_TEXT[rec.stage]}`;
  }

  function drawInset(): void {
    inset.replaceChildren();
    if (st.sel < 0 || !res) { insetDiv.style.display = "none"; return; }
    insetDiv.style.display = "";
    const rec = res.records[st.sel], c = rec.cand, oc = res.pyr.octaves[c.octave], w = oc.dog[0].width;
    const C = 36, gap = 10, x0 = 4;
    let m = 0;
    for (let ds = -1; ds <= 1; ds++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) m = Math.max(m, Math.abs(oc.dog[c.level + ds].data[(c.iy + dy) * w + c.ix + dx]));
    for (let ds = -1; ds <= 1; ds++) {
      const gx = x0 + (ds + 1) * (3 * C + gap);
      const lab = svgEl("text", { x: gx + 1.5 * C, y: 17, "text-anchor": "middle", "font-size": 16 }, inset);
      lab.textContent = `level ${c.level + ds}`;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const v = oc.dog[c.level + ds].data[(c.iy + dy) * w + c.ix + dx], col = diverging(0.5 + 0.5 * v / (m || 1));
        const r = svgEl("rect", { x: gx + (dx + 1) * C, y: 24 + (dy + 1) * C, width: C - 2, height: C - 2, rx: 3 }, inset);
        r.style.fill = `rgb(${col.map(u => Math.round(255 * u)).join(",")})`;
        if (!ds && !dx && !dy) { r.style.stroke = "#ffd400"; r.style.strokeWidth = "4"; }
        const t = svgEl("text", { x: gx + (dx + 1) * C + C / 2 - 1, y: 24 + (dy + 1) * C + C / 2 + 5, "text-anchor": "middle", "font-size": 15 }, inset);
        t.textContent = Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1);
      }
    }
  }

  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  makeImage();
}

autoMount(NAME, mount);
