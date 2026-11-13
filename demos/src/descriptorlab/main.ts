// Describing a keypoint. View 1 (synthetic scene, repetitive scene or the micrograph) with DoG
// keypoints; click one. Its window is resampled in the keypoint frame (16 × 16 samples: the
// gradient arrows, the 4 × 4 cells and the Gaussian weight), or, in binary mode, a smoothed
// 32 × 32 patch with point pairs coloured by their bit. The descriptor: 4 × 4 "roses" of 8
// petals (plus the 128 numbers), or the 256 bits as a 16 × 16 grid, or the normalised patch.
// Below: the distances from this descriptor to every descriptor of view 2, sorted, with the
// true partner (known transform) marked. View 2 is view 1 rotated, zoomed, with a lighting
// change, blur, noise, and an optional error in the keypoint frame.
//
// <div data-demo="descriptorlab" data-preset="nearest|norot|rot|gain|gamma|shift|binary|grid|blur"
//      data-desc="hist|binary|patch" data-scene="scene|grid|cell"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { Plot } from "../lib/plot";
import { Grey } from "../lib/imageops";
import { rng } from "../lib/random";
import * as K from "../lib/keypoints";
import * as D from "../lib/descriptors";
import { CELL_CREDIT } from "../lib/cell";
import { ImgView, injectStyle } from "../canny/imgview";
import { getScene, makeViews, detect, describe, describeKps, drawKp, scheduler, SceneKind, PATTERN, GREEN, RED, YELLOW, CYAN, COMMON_CSS, ViewPairResult, DescKind, DescOptions, Features } from "../repeatability/kpcommon";

const NAME = "descriptorlab";
const STYLE = COMMON_CSS + `
.qb-demo[data-demo="${NAME}"] { gap: .7em; flex-wrap: wrap; }
.qb-demo[data-demo="${NAME}"] .dl-a { flex: 0 1 470px; min-width: 300px; display: flex; flex-direction: column; gap: .2em; }
.qb-demo[data-demo="${NAME}"] .dl-pair { display: flex; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .dl-pair .ev-fig { flex: 1 1 0; }
.qb-demo[data-demo="${NAME}"] .dl-b { flex: 0 1 250px; min-width: 190px; display: flex; flex-direction: column; gap: .2em; }
.qb-demo[data-demo="${NAME}"] .dl-b svg { width: 100%; display: block; }
.qb-demo[data-demo="${NAME}"] .dl-cap { font-size: 16px; text-align: center; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 360px; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; }
.qb-demo[data-demo="${NAME}"] .dl-rowa { display: flex; flex-wrap: wrap; gap: .3em .8em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] .dl-a { flex: 0 0 440px; }
.reveal .qb-demo[data-demo="${NAME}"] .dl-b { flex: 0 0 230px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-panel { flex: 0 0 390px; max-width: 390px; gap: .08em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .03em .35em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 15px; padding: .2em .4em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-slider { gap: .3em; }
.reveal .qb-demo[data-demo="${NAME}"] .dl-cap { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .ev-credit { display: none; }
@media (max-width: 900px) { .qb-demo[data-demo="${NAME}"] { flex-wrap: wrap; } }
`;

interface State {
  scene: SceneKind; desc: DescKind; angle: number; zoom: number; gain: number; offset: number; gamma: number;
  blur: number; noise: number; frameErr: number; rotate: boolean; normalise: boolean; clip: boolean; soft: boolean; root: boolean;
  metric: D.Metric; sel: number;
}
const DEFAULT: State = { scene: "scene", desc: "hist", angle: 25, zoom: 1.2, gain: 1, offset: 0, gamma: 1, blur: 0, noise: 2, frameErr: 0, rotate: true, normalise: true, clip: true, soft: true, root: false, metric: "l2", sel: -1 };
interface Preset { id: string; label: string; s: Partial<State> }
const PRESETS: Preset[] = [
  { id: "nearest", label: "the true partner is nearest", s: {} },
  { id: "norot", label: "rotate 90° without θ: lost", s: { angle: 90, zoom: 1, rotate: false } },
  { id: "rot", label: "with θ: found again", s: { angle: 90, zoom: 1 } },
  { id: "gain", label: "gain ×3: normalising fixes it", s: { gain: 3, normalise: false, clip: false } },
  { id: "gamma", label: "gamma 0.5: clipping helps", s: { gamma: 0.5, clip: false } },
  { id: "shift", label: "a 2-px frame error: patch vs histogram", s: { desc: "patch", frameErr: 2, angle: 0, zoom: 1 } },
  { id: "binary", label: "binary: 256 bits, Hamming", s: { desc: "binary", metric: "hamming" } },
  { id: "grid", label: "repetitive grid: equally near impostors", s: { scene: "grid", angle: 10, zoom: 1 } },
  { id: "blur", label: "blur", s: { blur: 2 } },
];
const N_KP = 150;

function mount(host: HTMLElement): void {
  injectStyle(NAME, STYLE);
  const p0 = PRESETS.find(p => p.id === host.dataset.preset);
  const base = (): Partial<State> => ({
    ...(host.dataset.scene ? { scene: host.dataset.scene as SceneKind } : {}),
    ...(host.dataset.desc ? { desc: host.dataset.desc as DescKind, metric: host.dataset.desc === "binary" ? "hamming" as D.Metric : "l2" as D.Metric } : {}),
  });
  const start = (): State => ({ ...DEFAULT, ...base(), ...(p0?.s ?? {}) });
  const st: State = start();

  const colA = htmlEl("div", "dl-a", "", host);
  const pair = htmlEl("div", "dl-pair", "", colA);
  const view1 = new ImgView(pair, "view 1: click a keypoint"), view2 = new ImgView(pair, "view 2");
  const strip = new Plot(colA, { width: 440, height: 150, x: [0, 1], y: [0, 1], font: 16, xlabel: "keypoints of view 2, nearest first", ylabel: "distance", margin: { l: 56, r: 8, t: 8, b: 44 }, xticks: 4, yticks: 3 });
  const stripMark = strip.mark<[number[], number, number]>((g, ds, partner, n2) => {
    const top = Math.min(ds.length, 60);
    for (let k = 0; k < top; k++) {
      const c = svgEl("circle", { cx: strip.x(k), cy: strip.y(ds[k]), r: k === partner ? 7 : 4 }, g);
      c.style.fill = k === partner ? GREEN : k < 2 ? "#444" : "#9aa0a6";
    }
    void n2;
  }, [[], -1, 0]);
  const ro = htmlEl("div", "qb-readout", "", colA);
  const rowA = htmlEl("div", "dl-rowa", "", colA);
  const credit = htmlEl("div", "ev-credit", CELL_CREDIT, colA);
  const busy = htmlEl("div", "kp-busy", "", colA);

  const colB = htmlEl("div", "dl-b", "", host);
  const capP = htmlEl("div", "dl-cap", "", colB);
  const patchSvg = svgEl("svg", { viewBox: "0 0 240 240" }, colB);
  const capD = htmlEl("div", "dl-cap", "", colB);
  const descSvg = svgEl("svg", { viewBox: "0 0 240 240" }, colB);

  const panel = htmlEl("div", "qb-panel", "", host);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => { Object.assign(st, DEFAULT, base(), p.s); selPos = null; sync(); load(); } })));
  const descBtns = presets(panel, [
    { label: "gradient histogram", apply: () => { st.desc = "hist"; if (st.metric === "hamming") st.metric = "l2"; recompute(); } },
    { label: "binary", apply: () => { st.desc = "binary"; st.metric = "hamming"; recompute(); } },
    { label: "patch (NCC)", apply: () => { st.desc = "patch"; st.metric = "l2"; recompute(); } },
  ]);
  const metBtns = presets(rowA, (["l2", "l1", "chi2", "hamming"] as D.Metric[]).map(m => ({ label: m === "chi2" ? "χ²" : m === "hamming" ? "Hamming" : m.toUpperCase(), apply: () => { if ((m === "hamming") !== (st.desc === "binary")) return; st.metric = m; draw(); } })));
  const sceneBtns = presets(rowA, [
    { label: "scene", apply: () => { st.scene = "scene"; selPos = null; load(); } },
    { label: "repetitive", apply: () => { st.scene = "grid"; selPos = null; load(); } },
    { label: "micrograph", apply: () => { st.scene = "cell"; selPos = null; load(); } },
  ]);
  const aS = slider(panel, "rotation°", -180, 180, 5, st.angle, v => { st.angle = v; recompute(); });
  const zS = slider(panel, "zoom", 0.6, 1.8, 0.05, st.zoom, v => { st.zoom = v; recompute(); });
  const gS = slider(panel, "gain", 0.3, 3, 0.1, st.gain, v => { st.gain = v; recompute(); });
  const oS = slider(panel, "offset", -60, 60, 5, st.offset, v => { st.offset = v; recompute(); });
  const gmS = slider(panel, "gamma", 0.4, 2.5, 0.05, st.gamma, v => { st.gamma = v; recompute(); });
  const bS = slider(panel, "blur σ", 0, 4, 0.25, st.blur, v => { st.blur = v; recompute(); });
  const nS = slider(panel, "noise", 0, 20, 1, st.noise, v => { st.noise = v; recompute(); });
  const fS = slider(panel, "frame error (px)", 0, 3, 0.25, st.frameErr, v => { st.frameErr = v; recompute(); });
  const checks = htmlEl("div", "qb-presets", "", panel);
  const rC = checkbox(checks, "rotate to θ", st.rotate, v => { st.rotate = v; recompute(); });
  const nC = checkbox(checks, "normalise", st.normalise, v => { st.normalise = v; recompute(); });
  const cC = checkbox(checks, "clip 0.2", st.clip, v => { st.clip = v; recompute(); });
  const sC = checkbox(checks, "soft binning", st.soft, v => { st.soft = v; recompute(); });
  const rtC = checkbox(checks, "root", st.root, v => { st.root = v; recompute(); });
  button(panel, "reset", () => { Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); load(); });

  function sync(): void {
    aS.set(st.angle); zS.set(st.zoom); gS.set(st.gain); oS.set(st.offset); gmS.set(st.gamma); bS.set(st.blur); nS.set(st.noise); fS.set(st.frameErr);
    rC.checked = st.rotate; nC.checked = st.normalise; cC.checked = st.clip; sC.checked = st.soft; rtC.checked = st.root;
  }

  let scene: Grey | null = null, views: ViewPairResult | null = null;
  let f1: Features | null = null, f2: Features | null = null;
  let k1: K.Keypoint[] = [], k2: K.Keypoint[] = [], D1: D.Desc[] = [], D2: D.Desc[] = [], partner: Int32Array = new Int32Array(0);
  const run = scheduler(on => { busy.textContent = on ? "computing…" : ""; });
  function load(): void { getScene(st.scene).then(g => { scene = g; recompute(); }); }

  function recompute(): void {
    run(() => {
      if (!scene) return;
      views = makeViews(st.scene, scene, { zoom: st.zoom, angle: st.angle }, { gain: st.gain, offset: st.offset, gamma: st.gamma, blur: st.blur, noise: st.noise });
      const a = detect(views.v1, "dog", N_KP), b = detect(views.v2, "dog", N_KP);
      f1 = a; f2 = b;
      const o: DescOptions = { rotate: st.rotate, normalise: st.normalise, clip: st.clip ? 0.2 : 0, soft: st.soft, root: st.root };
      const r1 = describe(a, st.desc, o), r2 = describe(b, st.desc, o);
      k1 = r1.kps; D1 = r1.desc; k2 = r2.kps; D2 = r2.desc;
      if (st.frameErr > 0) {
        // frame error: describe each keypoint of view 2 from a position moved by frameErr px (seeded direction)
        const r = rng(9);
        D2 = describeKps(b, k2.map(k => { const t = 2 * Math.PI * r.unif(); return { ...k, x: k.x + st.frameErr * Math.cos(t), y: k.y + st.frameErr * Math.sin(t) }; }), st.desc, o);
      }
      partner = new Int32Array(k1.length).fill(-1);
      k1.forEach((k, i) => {
        const [x, y] = views!.map12(k.x, k.y);
        let best = -1, bd = Infinity;
        k2.forEach((q, j) => {
          const d = Math.hypot(q.x - x, q.y - y);
          if (d > 3) return;
          const want = k.theta + st.angle * Math.PI / 180, dt = st.rotate ? Math.abs(Math.atan2(Math.sin(q.theta - want), Math.cos(q.theta - want))) : 0;
          if (d + 5 * dt < bd) { bd = d + 5 * dt; best = j; }
        });
        partner[i] = best;
      });
      if (!target.has(st.scene)) target.set(st.scene, defaultTarget(a));
      const [tx, ty] = selPos ?? target.get(st.scene)!;
      let best = -1, bd = Infinity;
      k1.forEach((k, i) => { const d = Math.hypot(k.x - tx, k.y - ty) + (partner[i] >= 0 ? 0 : 5); if (d < bd) { bd = d; best = i; } });
      st.sel = best;
      draw();
    });
  }
  view1.onPick((x, y) => {
    let best = -1, bd = 10;
    k1.forEach((k, i) => { const d = Math.hypot(k.x - x, k.y - y); if (d < bd) { bd = d; best = i; } });
    if (best >= 0) { st.sel = best; selPos = [k1[best].x, k1[best].y]; draw(); }
  }, false);

  // the default keypoint: the strongest one (away from the border) whose nearest descriptor is its true partner
  // under the default view 2 (25°, zoom 1.2, gradient histograms), so that every preset starts from the same point
  const target = new Map<SceneKind, [number, number]>();
  let selPos: [number, number] | null = null;
  function defaultTarget(a: Features): [number, number] {
    const v = makeViews(st.scene, scene!, { zoom: DEFAULT.zoom, angle: DEFAULT.angle }, { noise: DEFAULT.noise });
    const b = detect(v.v2, "dog", N_KP), r1 = describe(a, "hist"), r2 = describe(b, "hist");
    let best: [number, number] = [128, 128], bv = -1;
    const grid = st.scene === "grid";
    r1.kps.forEach((k, i) => {
      if (k.x < 40 || k.y < 40 || k.x > 216 || k.y > 216 || k.response <= bv) return;
      if (grid) { if (Math.hypot(k.x - 128, k.y - 128) < 40) { bv = k.response; best = [k.x, k.y]; } return; }
      if (k.x > 175 && k.y > 175) return;                         // not in the scene's repetitive patch
      let jn = -1, dn = Infinity, d2 = Infinity;
      r2.desc.forEach((d, j) => { const e = D.l2(r1.desc[i], d); if (e < dn) { d2 = dn; dn = e; jn = j; } else if (e < d2) d2 = e; });
      const [x, y] = v.map12(k.x, k.y);
      if (jn >= 0 && dn < 0.7 * d2 && Math.hypot(r2.kps[jn].x - x, r2.kps[jn].y - y) < 2) { bv = k.response; best = [k.x, k.y]; }
    });
    return best;
  }

  function frameOutline(view: ImgView, kp: K.Keypoint, color: string, width: number): void {
    const half = (st.desc === "binary" ? 12 : 12) * kp.sigma / 2, t = st.rotate ? kp.theta : 0;
    const ux = Math.cos(t), uy = -Math.sin(t), vx = Math.sin(t), vy = Math.cos(t);
    const pts = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => `${kp.x + 0.5 + half * (a * ux + b * vx)},${kp.y + 0.5 + half * (a * uy + b * vy)}`);
    view.add("polygon", { points: pts.join(" "), fill: "none", stroke: color, "stroke-width": width });
    view.add("line", { x1: kp.x + 0.5, y1: kp.y + 0.5, x2: kp.x + 0.5 + half * ux, y2: kp.y + 0.5 + half * uy, stroke: color, "stroke-width": width });
  }

  function draw(): void {
    if (!views || !f1 || !f2) return;
    descBtns.forEach((b, i) => b.classList.toggle("active", ["hist", "binary", "patch"][i] === st.desc));
    metBtns.forEach((b, i) => { const m = (["l2", "l1", "chi2", "hamming"] as D.Metric[])[i]; b.classList.toggle("active", m === st.metric); b.disabled = (m === "hamming") !== (st.desc === "binary"); });
    sceneBtns.forEach((b, i) => b.classList.toggle("active", ["scene", "grid", "cell"][i] === st.scene));
    credit.style.display = st.scene === "cell" ? "" : "none";
    view1.grey(views.v1, { mode: "clip", lo: 0, hi: 255 }); view2.grey(views.v2, { mode: "auto" });
    view1.clear(); view2.clear();
    for (const k of k1) view1.add("circle", { cx: k.x + 0.5, cy: k.y + 0.5, r: 2.2, fill: "rgba(255,255,255,.75)", stroke: "rgba(0,0,0,.6)", "stroke-width": 0.8 });
    for (const k of k2) view2.add("circle", { cx: k.x + 0.5, cy: k.y + 0.5, r: 2.2, fill: "rgba(255,255,255,.75)", stroke: "rgba(0,0,0,.6)", "stroke-width": 0.8 });
    const i = st.sel;
    if (i < 0 || i >= k1.length) { ro.textContent = "click a keypoint in view 1"; return; }
    const f = D.distanceFn(st.metric), a = D1[i];
    const ds = D2.map((b, j) => ({ j, d: f(a, b) })).sort((p, q) => p.d - q.d);
    const pj = partner[i], rank = ds.findIndex(e => e.j === pj);
    frameOutline(view1, k1[i], YELLOW, 2.5);
    if (pj >= 0) frameOutline(view2, k2[pj], GREEN, 3);
    if (ds.length) { const n1 = k2[ds[0].j]; drawKp(view2, n1, ds[0].j === pj ? GREEN : RED, { width: 3, r: 2 * n1.sigma }); }
    if (ds.length > 1) drawKp(view2, k2[ds[1].j], CYAN, { width: 2, r: 2 * k2[ds[1].j].sigma });
    // distance strip
    const top = Math.min(ds.length, 60), mx = Math.max(...ds.slice(0, top).map(e => e.d), 1e-9);
    strip.setDomain([-1, top], [0, mx * 1.08]);
    stripMark.set(ds.map(e => e.d), rank, ds.length);
    // patch and descriptor
    drawPatch(i); drawDescriptor(a);
    const d1 = ds[0]?.d ?? NaN, d2 = ds[1]?.d ?? NaN;
    let ok = 0, tot = 0;
    k1.forEach((_, q) => { const pp = partner[q]; if (pp < 0) return; tot++; let jn = -1, dn = Infinity; D2.forEach((d, j) => { const e = f(D1[q], d); if (e < dn) { dn = e; jn = j; } }); if (jn === pp) ok++; });
    ro.textContent = `keypoint ${i}: σ = ${k1[i].sigma.toFixed(2)} px, θ = ${(k1[i].theta * 180 / Math.PI).toFixed(0)}°\n` +
      (pj < 0 ? "no true partner in view 2" : `true partner: rank ${rank + 1} of ${ds.length}, d = ${ds[rank].d.toFixed(3)}`) +
      `\nnearest d1 = ${d1.toFixed(3)}, second d2 = ${d2.toFixed(3)}\nd1/d2 = ${(d1 / d2).toFixed(2)}` + (pj >= 0 ? (rank === 0 ? "  → nearest is right" : "  → nearest is WRONG") : "") +
      `\nall keypoints: nearest = partner for ${ok} of ${tot} (${tot ? Math.round(100 * ok / tot) : 0} %)`;
  }

  function drawPatch(i: number): void {
    patchSvg.replaceChildren();
    const kp = k1[i], pyr = f1!.pyr;
    if (st.desc === "binary") {
      const { patch } = D.binaryPatch(pyr, kp, PATTERN, { steer: st.rotate });
      const S = patch.width, c = 240 / S, bits = D.binaryFromPatch(patch, PATTERN);
      greyCells(patchSvg, patch, c);
      for (let k = 0; k < 48; k++) {
        const P = PATTERN.pairs, h = (S - 1) / 2;
        const l = svgEl("line", { x1: (P[4 * k] + h + 0.5) * c, y1: (P[4 * k + 1] + h + 0.5) * c, x2: (P[4 * k + 2] + h + 0.5) * c, y2: (P[4 * k + 3] + h + 0.5) * c, "stroke-width": 2.2 }, patchSvg);
        l.style.stroke = D.getBit(bits, k) ? GREEN : RED; l.style.opacity = "0.85";
      }
      capP.textContent = "smoothed patch, 48 of 256 pairs (green: bit 1)";
      return;
    }
    const det = D.gradHistDetail(pyr, kp, { rotate: st.rotate, soft: st.soft });
    const n = det.n, c = 240 / n;
    greyCells(patchSvg, det.patch, c);
    if (st.desc === "patch") { capP.textContent = `window in the keypoint frame (${n} × ${n})`; return; }
    // cells, Gaussian weight, gradient arrows
    svgEl("circle", { cx: 120, cy: 120, r: 120, fill: "rgba(255,212,0,.10)", stroke: "rgba(255,212,0,.8)", "stroke-width": 1.5, "stroke-dasharray": "4 4" }, patchSvg);
    for (let k = 1; k < 4; k++) {
      for (const [x1, y1, x2, y2] of [[k * 60, 0, k * 60, 240], [0, k * 60, 240, k * 60]]) { const l = svgEl("line", { x1, y1, x2, y2, "stroke-width": 1.5 }, patchSvg); l.style.stroke = "#22d3ee"; }
    }
    let mm = 0; for (const v of det.mag) mm = Math.max(mm, v);
    for (let j = 0; j < n; j++) for (let q = 0; q < n; q++) {
      const k = j * n + q, m = det.mag[k] / (mm || 1);
      if (m < 0.08) continue;
      const cx = (q + 0.5) * c, cy = (j + 0.5) * c, L = 0.95 * c * m, a = det.angle[k];
      const l = svgEl("line", { x1: cx, y1: cy, x2: cx + L * Math.cos(a), y2: cy - L * Math.sin(a), "stroke-width": 2, "stroke-linecap": "round" }, patchSvg);
      l.style.stroke = "#ffd400";
    }
    capP.textContent = `window in the frame: ${n} × ${n}, 4 × 4 cells`;
  }

  function greyCells(svg: SVGSVGElement, g: Grey, c: number): void {
    let lo = Infinity, hi = -Infinity; for (const v of g.data) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    for (let y = 0; y < g.height; y++) for (let x = 0; x < g.width; x++) {
      const v = Math.round(255 * (g.data[y * g.width + x] - lo) / ((hi - lo) || 1));
      const r = svgEl("rect", { x: x * c, y: y * c, width: c + 0.4, height: c + 0.4 }, svg);
      r.style.fill = `rgb(${v},${v},${v})`;
    }
  }

  function drawDescriptor(a: D.Desc): void {
    descSvg.replaceChildren();
    if (st.desc === "binary") {
      const b = a as Uint32Array, c = 15;
      for (let k = 0; k < 256; k++) {
        const r = svgEl("rect", { x: (k % 16) * c, y: Math.floor(k / 16) * c, width: c - 1, height: c - 1 }, descSvg);
        r.style.fill = D.getBit(b, k) ? "#111" : "#f2f2f2"; r.style.stroke = "#ccc";
      }
      capD.textContent = "256 bits = 32 bytes";
      return;
    }
    if (st.desc === "patch") {
      const v = a as Float32Array, n = 16, c = 15;
      let m = 0; for (const x of v) m = Math.max(m, Math.abs(x));
      for (let k = 0; k < n * n; k++) {
        const t = Math.round(127.5 + 127.5 * v[k] / (m || 1));
        const r = svgEl("rect", { x: (k % n) * c, y: Math.floor(k / n) * c, width: c, height: c }, descSvg);
        r.style.fill = `rgb(${t},${t},${t})`;
      }
      capD.textContent = "256 numbers: mean 0, length 1";
      return;
    }
    const v = a as Float32Array;
    let m = 0; for (const x of v) m = Math.max(m, x);
    for (let cj = 0; cj < 4; cj++) for (let ci = 0; ci < 4; ci++) {
      const cx = ci * 60 + 30, cy = cj * 60 + 30;
      const bg = svgEl("rect", { x: ci * 60 + 1, y: cj * 60 + 1, width: 58, height: 58, rx: 4 }, descSvg);
      bg.style.fill = "#f4f6f8"; bg.style.stroke = "#d0d4d8";
      for (let b = 0; b < 8; b++) {
        const val = v[(cj * 4 + ci) * 8 + b] / (m || 1), L = 27 * val, ang = b * Math.PI / 4, w = 0.33;
        if (L < 0.5) continue;
        const p = (r: number, t: number) => `${(cx + r * Math.cos(t)).toFixed(1)},${(cy - r * Math.sin(t)).toFixed(1)}`;
        const path = svgEl("path", { d: `M${cx},${cy}L${p(L, ang - w)}L${p(L, ang + w)}Z` }, descSvg);
        path.style.fill = "#1f77b4"; path.style.opacity = "0.85";
      }
    }
    capD.textContent = "4 × 4 cells × 8 directions = 128";
  }

  sync();
  if (p0) presetBtns[PRESETS.indexOf(p0)].classList.add("active");
  load();
}

autoMount(NAME, mount);
