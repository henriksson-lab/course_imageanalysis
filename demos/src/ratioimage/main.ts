// Ratio imaging with a single-chain biosensor. A seeded time-lapse (40 frames, 128 × 128) of
// three cells with different thickness and expression: a wave of high concentration crosses
// cell 1, cell 2 holds a steady gradient, cell 3 is a constant control. Donor channel D and
// acceptor channel A (both under donor excitation); the sensor's ratio R = A/D goes from
// Rmin = 0.5 (empty) to Rmax = 2.5 (saturated), Kd = 200 nM. Tiles: the two channels.
// Middle: the ratio map (plain, intensity-modulated, or the donor intensity alone), two
// regions (click to place the active one). Right: the ratio over time in both regions
// (ratio of sums solid, mean of pixel ratios dashed) and the calibration curve c(R) with the
// region-1 value and its error bar. Controls: frame + play, photons, background offset and
// its subtraction, mask threshold, binning before dividing, bleed-through, channel shift,
// acceptor bleaching.
//
// <div data-demo="ratioimage" data-preset="lies|cancels|nobg|nomask|imd|few|shift|bleach|calib"></div>
import { autoMount, htmlEl, slider, presets, checkbox, button, svgEl } from "../lib/ui";
import { Plot } from "../lib/plot";
import { viridis, ColorMap } from "../lib/colour";
import * as F from "../lib/fret";
import { Tile, tileCss, injectStyle, fmt, debounce, filterPresets, hideControl } from "../threecube/view";

const NAME = "ratioimage";
const Q = `.qb-demo[data-demo="${NAME}"]`;
const CSS = tileCss(NAME) + `
${Q} { gap: .6em; align-items: flex-start; }
${Q} .ri-tiles { flex: 0 1 160px; min-width: 110px; display: flex; flex-direction: column; gap: .25em; }
${Q} .ri-map { flex: 1 1 280px; min-width: 200px; max-width: 340px; display: flex; flex-direction: column; gap: .25em; }
${Q} .ri-plots { flex: 1 1 300px; min-width: 220px; max-width: 360px; display: flex; flex-direction: column; gap: .1em; }
${Q} .qb-panel { flex: 1 1 260px; max-width: 340px; }
${Q} .ri-row { display: flex; flex-wrap: wrap; gap: .25em; }
.reveal ${Q} { flex-wrap: nowrap; }
.reveal ${Q} .ri-tiles { flex: 0 0 150px; }
.reveal ${Q} .ri-map { flex: 0 0 280px; }
.reveal ${Q} .ri-plots { flex: 0 0 300px; }
.reveal ${Q} .qb-panel { flex: 0 0 330px; }
`;

type Display = "ratio" | "imd" | "intensity";
interface State {
  frame: number; photons: number; offset: number; subtract: boolean; mask: number; bin: number;
  bleed: number; shift: number; bleach: number; display: Display;
  roi: [[number, number], [number, number]]; active: 0 | 1;
}
const DEFAULT: State = { frame: 20, photons: 100, offset: 40, subtract: true, mask: 0.4, bin: 1, bleed: 0, shift: 0, bleach: 0, display: "ratio", roi: [[0, 0], [0, 0]], active: 0 };
const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "lies", label: "intensity lies: thick cells look active", s: { display: "intensity" } },
  { id: "cancels", label: "the ratio cancels thickness", s: { display: "ratio" } },
  { id: "nobg", label: "forgot the background", s: { subtract: false } },
  { id: "nomask", label: "no mask: the background explodes", s: { mask: 0 } },
  { id: "imd", label: "intensity-modulated display", s: { mask: 0, display: "imd" } },
  { id: "few", label: "few photons: bin before dividing", s: { photons: 6 } },
  { id: "shift", label: "one channel shifted by a pixel", s: { shift: 1 } },
  { id: "bleach", label: "acceptor bleaches: the ratio drifts", s: { bleach: 0.01 } },
  { id: "calib", label: "calibrate: near R_max the error explodes", s: { photons: 30, frame: 20 } },
];
const RMAP_LO = Math.log(0.3), RMAP_HI = Math.log(3.3), RAD = 5;

function mount(host: HTMLElement): void {
  injectStyle(NAME, CSS);
  const scene = F.sensorScene();
  const W = scene.w, H = scene.h, T = scene.frames;
  const cellC: [number, number][] = [];
  { const sx = [0, 0, 0], sy = [0, 0, 0], n = [0, 0, 0];
    for (let i = 0; i < W * H; i++) { const L = scene.labels[i]; if (!L) continue; sx[L - 1] += i % W; sy[L - 1] += Math.floor(i / W); n[L - 1]++; }
    for (let k = 0; k < 3; k++) cellC.push([Math.round(sx[k] / n[k]), Math.round(sy[k] / n[k])]); }
  const defRoi = (): [[number, number], [number, number]] => [[cellC[0][0], cellC[0][1]], [cellC[2][0], cellC[2][1]]];
  const pre = PRESETS.find(p => p.id === host.dataset.preset);
  const start = (): State => ({ ...DEFAULT, roi: defRoi(), ...(pre?.s ?? {}) });
  const st: State = start();

  // ---- layout ----
  const tiles = htmlEl("div", "ri-tiles", "", host);
  const tD = new Tile(tiles, "donor channel D"), tA = new Tile(tiles, "acceptor channel A");
  const mapBox = htmlEl("div", "ri-map", "", host);
  const dispRow = htmlEl("div", "ri-row", "", mapBox);
  const dispBtns = (["ratio", "imd", "intensity"] as Display[]).map(d => button(dispRow, d === "ratio" ? "ratio" : d === "imd" ? "ratio × intensity" : "intensity", () => { st.display = d; off(); draw(); }));
  const tR = new Tile(mapBox, "");
  const roiRow = htmlEl("div", "ri-row", "", mapBox);
  const roiBtns = [0, 1].map(k => button(roiRow, `place region ${k + 1}`, () => { st.active = k as 0 | 1; roiBtns.forEach((b, j) => b.classList.toggle("active", j === k)); }));
  roiBtns[0].classList.add("active");
  tR.onDrag((x, y) => { st.roi[st.active] = [Math.floor(x), Math.floor(y)]; series(); draw(); }, (x, y) => { st.roi[st.active] = [Math.floor(x), Math.floor(y)]; series(); draw(); });
  const plots = htmlEl("div", "ri-plots", "", host);
  const RV = !!host.closest(".reveal");
  const tp = new Plot(plots, { width: RV ? 300 : 340, height: RV ? 200 : 230, x: [0, T - 1], y: [0, 4.6], xlabel: "frame", ylabel: "ratio A/D", yticks: [0, 1, 2, 3, 4], margin: { l: 56, r: 10, t: 8, b: 50 } });
  const COL = ["#d62728", "#1f77b4"];
  const ls = [0, 1].map(k => tp.line([], [], { color: COL[k], width: 3.5, label: `region ${k + 1}` }));
  const lm = [0, 1].map(k => tp.line([], [], { color: COL[k], width: 2.5, dash: "5 4" }));
  const vf = tp.vline(0, { color: "#555", width: 2 });
  tp.legend([{ label: "region 1", color: COL[0] }, { label: "region 2", color: COL[1] }, { label: "mean of ratios", color: "#555", dash: "5 4" }], "tr");
  const cp = new Plot(plots, { width: RV ? 300 : 340, height: RV ? 200 : 230, x: [0.4, 2.6], y: [0, 4000], xlabel: "ratio R", ylabel: "c (nM)", yticks: [0, 1000, 2000, 3000, 4000], margin: { l: 84, r: 10, t: 8, b: 50 } });
  cp.fn(R => F.calibrate(R, scene.Rmin, scene.Rmax, scene.Kd), { color: "#555", width: 3 }, 400);
  const errX = cp.line([], [], { color: COL[0], width: 4 }), errY = cp.line([], [], { color: COL[0], width: 4 });
  const cpt = cp.scatter([], [], { color: COL[0], r: 7 });

  const panel = htmlEl("div", "qb-panel", "", host);
  const PL = filterPresets(host, PRESETS);
  const presetBtns = presets(panel, PL.map(p => ({ label: p.label, apply: () => { stop(); Object.assign(st, DEFAULT, { roi: defRoi() }, p.s); sync(); regenerate(); } })));
  const frow = htmlEl("div", "ri-row", "", panel);
  const fS = slider(frow, "frame", 0, T - 1, 1, st.frame, v => { st.frame = v; draw(); });
  const playB = button(frow, "play", () => (timer ? stop() : play()));
  const regen = debounce(() => regenerate(), 150);
  const phS = slider(panel, "photons", 2, 400, 1, st.photons, v => { st.photons = v; off(); regen(); });
  const ofS = slider(panel, "background offset", 0, 100, 1, st.offset, v => { st.offset = v; off(); regen(); });
  const subC = checkbox(panel, "subtract background (cell-free corner)", st.subtract, v => { st.subtract = v; off(); series(); draw(); });
  const mS = slider(panel, "mask threshold", 0, 1, 0.05, st.mask, v => { st.mask = v; off(); draw(); });
  const binRow = htmlEl("div", "ri-row", "", panel);
  htmlEl("span", "", "bin before dividing:", binRow);
  const binBtns = [1, 2, 4].map(m => button(binRow, `${m}×${m}`, () => { st.bin = m; off(); series(); draw(); }));
  const blS = slider(panel, "bleed-through d", 0, 0.6, 0.01, st.bleed, v => { st.bleed = v; off(); regen(); });
  const shS = slider(panel, "shift A (px)", 0, 3, 1, st.shift, v => { st.shift = v; off(); regen(); });
  const bcS = slider(panel, "acceptor bleach / frame", 0, 0.03, 0.001, st.bleach, v => { st.bleach = v; off(); regen(); });
  button(panel, "reset", () => { stop(); Object.assign(st, start()); presetBtns.forEach(b => b.classList.remove("active")); sync(); regenerate(); });
  const ro = htmlEl("div", "qb-readout", "", panel);
  const off = () => presetBtns.forEach(b => b.classList.remove("active"));
  if (host.dataset.compact !== undefined) [phS, ofS, blS, shS, bcS].forEach(hideControl);
  function sync(): void {
    fS.set(st.frame); phS.set(st.photons); ofS.set(st.offset); subC.checked = st.subtract; mS.set(st.mask);
    blS.set(st.bleed); shS.set(st.shift); bcS.set(st.bleach);
  }
  let timer = 0;
  function play(): void { playB.textContent = "pause"; timer = window.setInterval(() => { st.frame = (st.frame + 1) % T; fS.set(st.frame); draw(); }, 160); }
  function stop(): void { if (timer) clearInterval(timer); timer = 0; playB.textContent = "play"; }

  // ---- data ----
  let frames: { D: Float32Array; A: Float32Array }[] = [];
  function regenerate(): void {
    frames = [];
    for (let t = 0; t < T; t++) {
      const f = F.sensorFrame(scene, t, { photons: st.photons, offset: st.offset, bleed: st.bleed, accBleach: st.bleach, shift: st.shift, seed: 1 });
      frames.push({ D: f.D, A: f.A });
    }
    series(); draw();
  }
  const bgOf = (v: Float32Array) => { let s = 0; for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) s += v[y * W + x]; return s / 64; };
  const inRoi = (k: number, x: number, y: number) => (x - st.roi[k][0]) ** 2 + (y - st.roi[k][1]) ** 2 <= RAD * RAD;
  function roiStats(t: number, k: number): { ros: number; mor: number; sA: number; sD: number; rawA: number; rawD: number; c: number } {
    const { D, A } = frames[t];
    const bD = st.subtract ? bgOf(D) : 0, bA = st.subtract ? bgOf(A) : 0, m = st.bin;
    let sA = 0, sD = 0, rawA = 0, rawD = 0, c = 0, n = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (inRoi(k, x, y)) { const i = y * W + x; sA += A[i] - bA; sD += D[i] - bD; rawA += A[i]; rawD += D[i]; c += scene.conc[t][i]; n++; }
    // mean of ratios over the (binned) pixels whose centre lies in the region
    let s = 0, nr = 0;
    const Wb = Math.floor(W / m), Hb = Math.floor(H / m);
    for (let Y = 0; Y < Hb; Y++) for (let X = 0; X < Wb; X++) {
      const cx = X * m + (m - 1) / 2, cy = Y * m + (m - 1) / 2;
      if (!inRoi(k, cx, cy)) continue;
      let a = 0, d = 0;
      for (let yy = 0; yy < m; yy++) for (let xx = 0; xx < m; xx++) { const i = (Y * m + yy) * W + X * m + xx; a += A[i] - bA; d += D[i] - bD; }
      if (d > 0) { s += a / d; nr++; }
    }
    return { ros: sD !== 0 ? sA / sD : NaN, mor: nr ? s / nr : NaN, sA, sD, rawA, rawD, c: n ? c / n : NaN };
  }
  let ser: { ros: number[]; mor: number[] }[] = [];
  function series(): void {
    ser = [0, 1].map(k => { const ros: number[] = [], mor: number[] = []; for (let t = 0; t < T; t++) { const r = roiStats(t, k); ros.push(r.ros); mor.push(r.mor); } return { ros, mor }; });
  }

  // ---- drawing ----
  function draw(): void {
    const { D, A } = frames[st.frame];
    const bD = st.subtract ? bgOf(D) : 0, bA = st.subtract ? bgOf(A) : 0, m = st.bin;
    const hi = 2.6 * st.photons + st.offset;
    tD.grey(D, W, H, 0, hi); tA.grey(A, W, H, 0, hi * 1.3);
    const Ab = Float32Array.from(A, v => v - bA), Db = Float32Array.from(D, v => v - bD);
    const a = F.binSum(Ab, W, H, m), d = F.binSum(Db, W, H, m);
    const Wb = a.w, Hb = a.h, R = new Float32Array(Wb * Hb), I = new Float32Array(Wb * Hb);
    const thr = st.mask * st.photons * m * m;
    for (let i = 0; i < R.length; i++) {
      I[i] = a.data[i] + d.data[i];
      R[i] = st.mask > 0 && I[i] < thr ? NaN : a.data[i] / d.data[i];
    }
    const px = new Uint8ClampedArray(4 * W * H), map: ColorMap = viridis, iScale = 1 / (2.8 * st.photons * m * m);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const j = Math.min(Hb - 1, Math.floor(y / m)) * Wb + Math.min(Wb - 1, Math.floor(x / m)), o = 4 * (y * W + x);
      let c: [number, number, number] = [0.16, 0.16, 0.16];
      if (st.display === "intensity") { const g = Math.min(1, Math.max(0, Db[y * W + x] / (2.6 * st.photons))); c = [g, g, g]; }
      else {
        const r = R[j];
        if (Number.isFinite(r) && r > 0) {
          c = map(Math.min(1, Math.max(0, (Math.log(r) - RMAP_LO) / (RMAP_HI - RMAP_LO))));
          if (st.display === "imd") { const v = Math.min(1, Math.max(0, I[j] * iScale)); c = [c[0] * v, c[1] * v, c[2] * v]; }
        } else if (Number.isFinite(r) || r === Infinity || r === -Infinity) c = r > 0 ? map(1) : map(0);
        else if (!(st.mask > 0 && I[j] < thr)) c = map(0);
      }
      px[o] = 255 * c[0]; px[o + 1] = 255 * c[1]; px[o + 2] = 255 * c[2]; px[o + 3] = 255;
    }
    tR.rgba(px, W, H);
    tR.setCaption(st.display === "intensity" ? "donor intensity" : `ratio A/D, colour 0.3 … 3.3 (log)${st.display === "imd" ? " × intensity" : ""}`);
    tR.clear();
    st.roi.forEach((p, k) => { tR.add("circle", { cx: p[0] + 0.5, cy: p[1] + 0.5, r: RAD, fill: "none", stroke: COL[k] === "#1f77b4" ? "#4fa3ff" : "#ff5a5a", "stroke-width": 3 }); tR.label(p[0] + 0.5, p[1] - RAD - 4, String(k + 1), 9, COL[k]); });
    dispBtns.forEach((b, i) => b.classList.toggle("active", ["ratio", "imd", "intensity"][i] === st.display));
    binBtns.forEach((b, i) => b.classList.toggle("active", [1, 2, 4][i] === st.bin));
    // time plot
    const ts = Array.from({ length: T }, (_, i) => i);
    const clip = (v: number[]) => v.map(x => (Number.isFinite(x) ? Math.max(-1, Math.min(5, x)) : NaN));
    ser.forEach((s, k) => { ls[k].set(ts, clip(s.ros)); lm[k].set(ts, clip(s.mor)); });
    vf.set(st.frame);
    // calibration of region 1
    const r1 = roiStats(st.frame, 0), r2 = roiStats(st.frame, 1);
    const sdR = Math.abs(r1.ros) * Math.sqrt(r1.rawA / (r1.sA * r1.sA) + r1.rawD / (r1.sD * r1.sD));
    const c = F.calibrate(r1.ros, scene.Rmin, scene.Rmax, scene.Kd);
    const cLo = F.calibrate(r1.ros - sdR, scene.Rmin, scene.Rmax, scene.Kd), cHi = F.calibrate(r1.ros + sdR, scene.Rmin, scene.Rmax, scene.Kd);
    const cl = (v: number) => Math.min(4000, v);
    cpt.set([r1.ros], [cl(c)]);
    errX.set([r1.ros - sdR, r1.ros + sdR], [cl(c), cl(c)]);
    errY.set([r1.ros, r1.ros], [cl(cLo), cl(cHi)]);
    ro.textContent = [
      `frame ${st.frame}`,
      `R₁ = ${fmt(r1.ros, 3)} ± ${fmt(sdR, 3)}`,
      `   mean of ratios ${fmt(r1.mor, 3)}`,
      `c₁ = ${fmt(c, 0)} nM (${fmt(cLo, 0)} … ${fmt(cHi, 0)})`,
      `   true ${fmt(r1.c, 0)} nM`,
      `R₂ = ${fmt(r2.ros, 3)}, mean of ratios ${fmt(r2.mor, 3)}`,
    ].join("\n");
    void svgEl;
  }

  sync();
  regenerate();
  const pi = PL.findIndex(p => p.id === host.dataset.preset);
  if (pi >= 0) presetBtns[pi].classList.add("active");
}

autoMount(NAME, mount);
