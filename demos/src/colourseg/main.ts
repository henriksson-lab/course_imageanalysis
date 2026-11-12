// Colour segmentation: choose a colour space in which "the red objects" become a simple
// region. Image (click: reference colour) | 2D histogram of a chromatic pair (log counts;
// selected pixels in orange) | the result. Methods: per-channel RGB thresholds; a hue
// interval (may wrap through 0°) with saturation and value floors; a ball of radius ΔE
// around a clicked colour (ΔE76, ΔE2000, or plain RGB distance); Otsu on one channel;
// k-means in RGB or Lab (step by step). Dice against the synthetic scene's truth.
// Auto-mounts on <div data-demo="colourseg">. Options: data-preset (rgb|hue|wrap|speckle|
// ball|kmeans), data-method.
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { linearToSrgb } from "../lib/colour";
import { Scene, SceneId, makeScene, withNoise } from "../colourcloud/scenes";
import * as SG from "./seg";

const NAME = "colourseg";
const STYLE = `
[data-demo="${NAME}"] { gap: .7em; }
[data-demo="${NAME}"] .cs-imgs { flex: 1 1 600px; display: grid; grid-template-columns: repeat(3, 1fr); gap: .5em; max-width: 720px; min-width: 0; }
[data-demo="${NAME}"] figure { margin: 0; min-width: 0; }
[data-demo="${NAME}"] figcaption { font-size: 16px; text-align: center; margin-bottom: .1em; white-space: nowrap; overflow: hidden; }
[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] canvas.cs-pick { cursor: crosshair; }
[data-demo="${NAME}"] .cs-axes { font-size: 14px; color: var(--qb-muted); text-align: center; }
[data-demo="${NAME}"] .cs-credit { font-size: 13px; color: var(--qb-muted); grid-column: 1 / -1; }
[data-demo="${NAME}"] .cs-pre { grid-column: 1 / -1; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 270px; max-width: 420px; gap: .35em; }
[data-demo="${NAME}"] .cs-row { display: flex; flex-wrap: wrap; gap: .3em; }
[data-demo="${NAME}"] .cs-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .cs-imgs { max-width: 640px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .1em .42em; }
.reveal [data-demo="${NAME}"] .qb-note, .reveal [data-demo="${NAME}"] .cs-credit { display: none; }
`;

type Method = "rgb" | "hsv" | "ball" | "otsu" | "kmeans";

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

autoMount(NAME, host => {
  injectStyle();
  const st = {
    image: "objects" as SceneId, noise: 0.01, method: (host.dataset.method ?? "hsv") as Method,
    rMin: 120, gMax: 90, bMax: 90,
    hue: 0, hw: 20, sMin: 0.4, vMin: 0.1,
    ref: -1, radius: 25, metric: "de76" as SG.Metric,
    ch: "a" as SG.OtsuChannel, above: true,
    k: 3, kspace: "rgb" as "rgb" | "lab", kSteps: 0,
  };
  const imgs = htmlEl("div", "cs-imgs", "", host);
  const fig = (cap: string) => { const f = htmlEl("figure", "", "", imgs); const c = htmlEl("figcaption", "", cap, f); const cv = htmlEl("canvas", "", "", f); return { f, c, cv }; };
  const fImg = fig("image (click: reference)"), fHist = fig("2D histogram"), fRes = fig("result");
  fImg.cv.classList.add("cs-pick");
  const axes = htmlEl("div", "cs-axes", "", fHist.f);
  const credit = htmlEl("div", "cs-credit", "", imgs);
  const preBox = htmlEl("div", "cs-pre", "", imgs);

  const panel = htmlEl("div", "qb-panel", "", host);
  const imRow = htmlEl("div", "cs-row", "", panel);
  const IMS: Partial<Record<SceneId, string>> = { objects: "objects in a shadow", he: "stained tissue", chart: "all hues" };
  const imBtns = (Object.keys(IMS) as SceneId[]).map(id => { const b = button(imRow, IMS[id]!, () => { st.image = id; st.ref = -1; st.kSteps = 0; update(); }); b.dataset.id = id; return b; });
  const noiseS = slider(panel, "noise", 0, 0.06, 0.005, st.noise, v => { st.noise = v; update(); });
  htmlEl("div", "cs-lab", "Method", panel);
  const mRow = htmlEl("div", "cs-row", "", panel);
  const MS: Record<Method, string> = { rgb: "RGB thresholds", hsv: "hue + floors", ball: "ΔE ball", otsu: "Otsu on a channel", kmeans: "k-means" };
  const mBtns = (Object.keys(MS) as Method[]).map(m => { const b = button(mRow, MS[m], () => { st.method = m; update(); }); b.dataset.id = m; return b; });
  const box = (m: Method) => { const d = htmlEl("div", "", "", panel); d.dataset.m = m; return d; };
  const bRgb = box("rgb");
  const sR = slider(bRgb, "R′ ≥", 0, 255, 1, st.rMin, v => { st.rMin = v; update(); });
  const sG = slider(bRgb, "G′ ≤", 0, 255, 1, st.gMax, v => { st.gMax = v; update(); });
  const sB = slider(bRgb, "B′ ≤", 0, 255, 1, st.bMax, v => { st.bMax = v; update(); });
  const bHsv = box("hsv");
  const sH = slider(bHsv, "hue centre", 0, 359, 1, st.hue, v => { st.hue = v; update(); }, v => v + "°");
  const sHW = slider(bHsv, "± width", 1, 90, 1, st.hw, v => { st.hw = v; update(); }, v => v + "°");
  const sS = slider(bHsv, "S ≥", 0, 1, 0.02, st.sMin, v => { st.sMin = v; update(); });
  const sV = slider(bHsv, "V ≥", 0, 1, 0.02, st.vMin, v => { st.vMin = v; update(); });
  const bBall = box("ball");
  const sRad = slider(bBall, "radius", 1, 100, 1, st.radius, v => { st.radius = v; update(); });
  const metRow = htmlEl("div", "cs-row", "", bBall);
  const metBtns = (["de76", "de2000", "rgb"] as SG.Metric[]).map(m => { const b = button(metRow, m === "de76" ? "ΔE76 (Lab)" : m === "de2000" ? "ΔE2000" : "RGB distance", () => { st.metric = m; update(); }); b.dataset.id = m; return b; });
  const bOtsu = box("otsu");
  const chRow = htmlEl("div", "cs-row", "", bOtsu);
  const chBtns = (["R", "G", "B", "S", "L", "a", "b"] as SG.OtsuChannel[]).map(c => { const b = button(chRow, c === "a" ? "a*" : c === "b" ? "b*" : c === "L" ? "L*" : c === "S" ? "S" : c + "′", () => { st.ch = c; update(); }); b.dataset.id = c; return b; });
  const abRow = htmlEl("div", "cs-row", "", bOtsu);
  const abBtns = [true, false].map(a => { const b = button(abRow, a ? "keep above" : "keep below", () => { st.above = a; update(); }); b.dataset.id = String(a); return b; });
  const bK = box("kmeans");
  const sK = slider(bK, "k", 2, 5, 1, st.k, v => { st.k = v; st.kSteps = 0; update(); });
  const ksRow = htmlEl("div", "cs-row", "", bK);
  const ksBtns = (["rgb", "lab"] as const).map(s => { const b = button(ksRow, s === "rgb" ? "in RGB" : "in Lab", () => { st.kspace = s; st.kSteps = 0; update(); }); b.dataset.id = s; return b; });
  const kRow = htmlEl("div", "cs-row", "", bK);
  button(kRow, "step", () => { st.kSteps++; update(); });
  button(kRow, "run", () => { st.kSteps = 100; update(); });
  button(kRow, "restart", () => { st.kSteps = 0; update(); });
  const read = htmlEl("div", "qb-readout", "", panel);
  htmlEl("div", "qb-note", "Dice compares the result with the true objects of the synthetic scene (red apple; nuclei). In the histogram, orange marks the selected pixels.", panel);

  const P: Record<string, Partial<typeof st>> = {
    rgb: { image: "objects", noise: 0.01, method: "rgb", rMin: 120, gMax: 90, bMax: 90 },
    hue: { image: "objects", noise: 0.01, method: "hsv", hue: 0, hw: 20, sMin: 0.4, vMin: 0.1 },
    wrap: { image: "chart", noise: 0, method: "hsv", hue: 0, hw: 20, sMin: 0.4, vMin: 0.1 },
    speckle: { image: "objects", noise: 0.03, method: "hsv", hue: 0, hw: 20, sMin: 0, vMin: 0 },
    ball: { image: "objects", noise: 0.01, method: "ball", ref: -1, radius: 25, metric: "de76" },
    kmeans: { image: "objects", noise: 0.01, method: "kmeans", k: 4, kspace: "rgb", kSteps: 100 },
  };
  const list: [string, string][] = [["rgb", "R threshold fails in the shadow"], ["hue", "hue survives the shadow"], ["wrap", "red wraps around 0°"], ["speckle", "grey and dark pixels speckle"], ["ball", "ΔE ball in Lab vs RGB distance"], ["kmeans", "k-means in RGB vs in Lab"]];
  const pb = presets(preBox, [...list.map(([id, label]) => ({ label, apply: () => { Object.assign(st, P[id]); sync(); update(); } })),
    { label: "reset", apply: () => { Object.assign(st, P.hue); sync(); update(); } }]);
  function sync(): void { sR.set(st.rMin); sG.set(st.gMax); sB.set(st.bMax); sH.set(st.hue); sHW.set(st.hw); sS.set(st.sMin); sV.set(st.vMin); sRad.set(st.radius); sK.set(st.k); noiseS.set(st.noise); }

  // ---- data ----
  let curKey = "", sc: Scene, px: SG.Pixels, dv: SG.Derived;
  function data(): void {
    const key = st.image + "|" + st.noise;
    if (key === curKey) return;
    curKey = key;
    sc = withNoise(makeScene(st.image, 128), st.noise);
    const n = sc.w * sc.h, enc = (a: Float32Array) => Float32Array.from(a, v => linearToSrgb(Math.min(1, Math.max(0, v))));
    px = { r: enc(sc.r), g: enc(sc.g), b: enc(sc.b), n };
    dv = SG.derive(px);
  }
  const defaultRef = (): number => {
    // a lit pixel of the red apple / a nucleus / the chart's red
    if (st.image === "objects") return 50 * 128 + 40;
    if (st.image === "he" && sc.truth) { for (let p = 0; p < sc.truth.length; p++) if (sc.truth[p] === 1) return p; }
    return 64 * 128 + 2;
  };

  fImg.cv.addEventListener("pointerdown", ev => {
    const r = fImg.cv.getBoundingClientRect();
    const x = Math.floor((ev.clientX - r.left) / r.width * 128), y = Math.floor((ev.clientY - r.top) / r.height * 128);
    if (x < 0 || y < 0 || x >= 128 || y >= 128) return;
    st.ref = y * 128 + x;
    if (st.method === "hsv") { const h = dv.hsv[3 * st.ref]; if (Number.isFinite(h)) { st.hue = Math.round(h) % 360; sH.set(st.hue); } }
    else st.method = "ball";
    update();
  });

  function drawImage(c: HTMLCanvasElement, dim: Uint8Array | null, overlay?: (p: number) => [number, number, number] | null): void {
    c.width = 128; c.height = 128;
    const ctx = c.getContext("2d")!, id = ctx.createImageData(128, 128);
    for (let p = 0; p < px.n; p++) {
      let col: [number, number, number] = [255 * px.r[p], 255 * px.g[p], 255 * px.b[p]];
      const o = overlay?.(p);
      if (o) col = o;
      else if (dim && !dim[p]) col = col.map(v => 0.25 * v + 25) as [number, number, number];
      id.data[4 * p] = col[0]; id.data[4 * p + 1] = col[1]; id.data[4 * p + 2] = col[2]; id.data[4 * p + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
  }

  /** 2D histogram (log) of all pixels in grey, selected pixels in orange. */
  function drawHist(mask: Uint8Array): void {
    const B = 128, c = fHist.cv; c.width = B; c.height = B;
    const all = new Float64Array(B * B), sel = new Float64Array(B * B);
    let xy: (p: number) => [number, number];
    if (st.method === "hsv") { xy = p => [Number.isFinite(dv.hsv[3 * p]) ? dv.hsv[3 * p] / 360 : -1, dv.hsv[3 * p + 1]]; axes.textContent = "x: hue 0–360°, y: saturation 0–1 (up)"; }
    else if (st.method === "rgb") { xy = p => { const s = px.r[p] + px.g[p] + px.b[p]; return s > 0 ? [px.r[p] / s, px.g[p] / s] : [-1, -1]; }; axes.textContent = "x: r = R′/(R′+G′+B′), y: g (up)"; }
    else { xy = p => [(dv.lab[3 * p + 1] + 100) / 200, (dv.lab[3 * p + 2] + 100) / 200]; axes.textContent = "x: a* (−100…100), y: b* (up)"; }
    for (let p = 0; p < px.n; p++) {
      const [u, v] = xy(p);
      if (!(u >= 0 && u <= 1 && v >= 0 && v <= 1)) continue;
      const i = Math.min(B - 1, Math.floor(u * B)) + B * (B - 1 - Math.min(B - 1, Math.floor(v * B)));
      all[i]++; if (mask[p]) sel[i]++;
    }
    let mx = 0; for (const v of all) mx = Math.max(mx, v);
    const ctx = c.getContext("2d")!, id = ctx.createImageData(B, B), lm = Math.log(1 + mx);
    for (let i = 0; i < B * B; i++) {
      const t = all[i] ? Math.log(1 + all[i]) / lm : 0;
      if (sel[i] > 0) { id.data[4 * i] = 255; id.data[4 * i + 1] = 140 - 60 * t; id.data[4 * i + 2] = 0; }
      else { const g = all[i] ? 175 - 150 * t : 255; id.data[4 * i] = g; id.data[4 * i + 1] = g; id.data[4 * i + 2] = g; }
      id.data[4 * i + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    // region outlines
    ctx.strokeStyle = "#1f77b4"; ctx.lineWidth = 1;
    if (st.method === "hsv") {
      const y = B * (1 - st.sMin);
      for (const [a, b] of wrapIntervals(st.hue - st.hw, st.hue + st.hw)) ctx.strokeRect(a / 360 * B, 0, (b - a) / 360 * B, y);
    } else if (st.method === "ball" && st.ref >= 0 && st.metric === "de76") {
      ctx.beginPath(); ctx.arc((dv.lab[3 * st.ref + 1] + 100) / 200 * B, B * (1 - (dv.lab[3 * st.ref + 2] + 100) / 200), st.radius / 200 * B, 0, 2 * Math.PI); ctx.stroke();
    }
  }
  const wrapIntervals = (a: number, b: number): [number, number][] => a < 0 ? [[0, b], [360 + a, 360]] : b > 360 ? [[a, 360], [0, b - 360]] : [[a, b]];

  function update(): void {
    data();
    if (st.ref < 0) st.ref = defaultRef();
    imBtns.forEach(b => b.classList.toggle("active", b.dataset.id === st.image));
    mBtns.forEach(b => b.classList.toggle("active", b.dataset.id === st.method));
    for (const d of panel.querySelectorAll<HTMLElement>("[data-m]")) d.style.display = d.dataset.m === st.method ? "" : "none";
    metBtns.forEach(b => b.classList.toggle("active", b.dataset.id === st.metric));
    chBtns.forEach(b => b.classList.toggle("active", b.dataset.id === st.ch));
    abBtns.forEach(b => b.classList.toggle("active", b.dataset.id === String(st.above)));
    ksBtns.forEach(b => b.classList.toggle("active", b.dataset.id === st.kspace));
    credit.textContent = sc.credit;
    let mask: Uint8Array, lines: string[] = [], overlay: ((p: number) => [number, number, number] | null) | undefined;
    if (st.method === "rgb") { mask = SG.rgbRule(px, st.rMin, st.gMax, st.bMax); lines.push(`R′ ≥ ${st.rMin}, G′ ≤ ${st.gMax}, B′ ≤ ${st.bMax}`); }
    else if (st.method === "hsv") {
      mask = SG.hueRule(dv, px.n, st.hue, st.hw, st.sMin, st.vMin);
      const iv = wrapIntervals(st.hue - st.hw, st.hue + st.hw).map(([a, b]) => `${a.toFixed(0)}–${b.toFixed(0)}°`).join(" ∪ ");
      lines.push(`hue in ${iv}`, `S ≥ ${st.sMin.toFixed(2)}, V ≥ ${st.vMin.toFixed(2)}`);
    } else if (st.method === "ball") {
      mask = SG.ballRule(px, dv, st.ref, st.radius, st.metric);
      const L = [0, 1, 2].map(k => dv.lab[3 * st.ref + k].toFixed(1)).join(", ");
      lines.push(`reference Lab (${L})`, `sRGB ${[px.r, px.g, px.b].map(a => Math.round(255 * a[st.ref])).join(", ")}`);
    } else if (st.method === "otsu") {
      const v = SG.channelValues(px, dv, st.ch), t = SG.otsu(v);
      mask = Uint8Array.from(v, x => (st.above ? x >= t : x < t) ? 1 : 0);
      lines.push(`Otsu threshold on ${st.ch}: ${t.toFixed(st.ch === "S" ? 3 : 1)}`);
    } else {
      const pts = new Float32Array(3 * px.n);
      for (let p = 0; p < px.n; p++) {
        if (st.kspace === "rgb") { pts[3 * p] = 255 * px.r[p]; pts[3 * p + 1] = 255 * px.g[p]; pts[3 * p + 2] = 255 * px.b[p]; }
        else pts.set(dv.lab.subarray(3 * p, 3 * p + 3), 3 * p);
      }
      const km = SG.kmeans(pts, px.n, st.k, Math.max(1, st.kSteps), 5);
      // cluster colours = mean encoded colour of their pixels
      const mean = Array.from({ length: st.k }, () => [0, 0, 0, 0]);
      for (let p = 0; p < px.n; p++) { const m = mean[km.labels[p]]; m[0] += px.r[p]; m[1] += px.g[p]; m[2] += px.b[p]; m[3]++; }
      overlay = p => { const m = mean[km.labels[p]]; return [255 * m[0] / m[3], 255 * m[1] / m[3], 255 * m[2] / m[3]]; };
      // the cluster that best matches the target class
      let bestJ = 0, bestD = -1;
      for (let j = 0; j < st.k; j++) { const m = Uint8Array.from(km.labels, l => l === j ? 1 : 0); const d = sc.truth ? SG.diceClass(m, sc.truth, 1) : 0; if (d > bestD) { bestD = d; bestJ = j; } }
      mask = Uint8Array.from(km.labels, l => l === bestJ ? 1 : 0);
      lines.push(`iterations: ${km.iterations}${km.iterations < Math.max(1, st.kSteps) ? " (converged)" : ""}`, `cluster sizes: ${mean.map(m => m[3]).join(", ")}`);
    }
    drawImage(fImg.cv, null, p => {
      const x0 = st.ref % 128, y0 = Math.floor(st.ref / 128), x = p % 128, y = Math.floor(p / 128);
      return (x === x0 && Math.abs(y - y0) >= 2 && Math.abs(y - y0) <= 5) || (y === y0 && Math.abs(x - x0) >= 2 && Math.abs(x - x0) <= 5) ? [255, 255, 255] : null;
    });
    drawImage(fRes.cv, mask, overlay);
    fRes.c.textContent = st.method === "kmeans" ? "clusters (mean colours)" : "selected pixels";
    drawHist(mask);
    const nSel = mask.reduce((s, v) => s + v, 0);
    lines.push(`selected: ${nSel} pixels`);
    if (sc.truth && st.image !== "chart") {
      const name = st.image === "objects" ? "red apple" : "nuclei";
      lines.push(`Dice vs ${name}: ${SG.diceClass(mask, sc.truth, 1).toFixed(3)}${st.method === "kmeans" ? " (best cluster)" : ""}`);
    }
    read.textContent = lines.join("\n");
  }
  const pre = host.dataset.preset;
  if (pre && P[pre]) { Object.assign(st, P[pre]); sync(); pb[list.findIndex(l => l[0] === pre)]?.classList.add("active"); }
  update();
});
