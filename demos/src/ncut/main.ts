// Spectral partitioning and normalized cuts (Shi & Malik 2000). A graph on draggable points
// (Gaussian affinity, width sigma) or on the pixels of a 32 x 32 image (intensity affinity
// sigma, neighbours within 2 pixels). Computes the Fiedler vector, the second-smallest
// eigenvector of L x = mu D x (normalized) or L x = mu x (unnormalized), shows it as
// colour, and splits the nodes at a threshold. Plots: the sorted vector, and Ncut and the
// cut value for every split along it. Compare with the minimum cut without balancing
// (Stoer-Wagner for points; for the image, the smallest cut along the sweep).
// Auto-mounts on <div data-demo="ncut">. Optional: data-preset="two|three|outlier|rings|long|disk|cell".
import { htmlEl, slider, presets, button, checkbox, autoMount, svgEl, svgPoint, draggable } from "../lib/ui";
import { fiedler, sweep, stoerWagner, WGraph, Fiedler, Sweep } from "./spectral";
import { POINT_PRESETS, pointGraph, imageGraph, diskImage, cellCrop, IMG, Pt } from "./data";
// @ts-ignore -- esbuild inlines the image as a data URL
import cellUrl from "../fourier2d/cell256.png";

const NAME = "ncut";
const CREDIT = "Micrograph: Robert M. Hunt, CC BY 3.0, via Wikimedia Commons";
const S = 420;            // main view size
const PW = 330, PH = 190; // plot size

const STYLE = `
[data-demo="${NAME}"] { gap: .9em; }
[data-demo="${NAME}"] .nc-main { flex: 0 1 410px; min-width: 240px; }
[data-demo="${NAME}"] .nc-main canvas { width: 100%; aspect-ratio: 1; display: block; image-rendering: pixelated; border-radius: 3px; }
[data-demo="${NAME}"] .nc-cap { font-size: 16px; text-align: center; margin-bottom: .2em; }
[data-demo="${NAME}"] .nc-mid { flex: 0 1 330px; min-width: 220px; display: flex; flex-direction: column; gap: .3em; }
[data-demo="${NAME}"] .qb-readout { font-size: 14px; white-space: pre; }
[data-demo="${NAME}"] .qb-panel { flex: 1 1 240px; max-width: 320px; gap: .4em; }
[data-demo="${NAME}"] .nc-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.25em; }
[data-demo="${NAME}"] .nc-credit { font-size: 12px; color: var(--qb-muted); }
[data-demo="${NAME}"] svg text { font-size: 15px; }
.reveal [data-demo="${NAME}"] { font-size: 17px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; }
.reveal [data-demo="${NAME}"] .qb-note { font-size: 14px; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .15em .5em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function loadBig(): Promise<Float64Array> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = 256; c.height = 256;
      const ctx = c.getContext("2d")!;
      ctx.drawImage(img, 0, 0, 256, 256);
      const d = ctx.getImageData(0, 0, 256, 256).data;
      resolve(Float64Array.from({ length: 256 * 256 }, (_, k) => d[4 * k]));
    };
    img.onerror = reject;
    img.src = cellUrl;
  });
}

/** Diverging colour: -1 blue, 0 white, +1 orange. */
function divColor(v: number): [number, number, number] {
  const t = Math.max(-1, Math.min(1, v));
  const end = t < 0 ? [31, 119, 180] : [255, 127, 14];
  const a = Math.abs(t);
  return [0, 1, 2].map(i => Math.round(255 + (end[i] - 255) * a)) as [number, number, number];
}
const css = (c: [number, number, number]) => `rgb(${c[0]},${c[1]},${c[2]})`;
const sig3 = (v: number) => (v === 0 ? "0" : Math.abs(v) >= 0.01 && Math.abs(v) < 1000 ? v.toPrecision(3) : v.toExponential(2));

type Mode = "points" | "image";
interface AnyPreset { id: string; label: string; mode: Mode; sigma: number }
const PRESETS: AnyPreset[] = [
  ...POINT_PRESETS.map(p => ({ id: p.id, label: p.label, mode: "points" as Mode, sigma: p.sigma })),
  { id: "disk", label: "image: shapes", mode: "image", sigma: 30 },
  { id: "cell", label: "image: micrograph", mode: "image", sigma: 30 },
];

function mount(host: HTMLElement): void {
  injectStyle();
  const p0 = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS[0];
  const st = {
    preset: p0, mode: p0.mode, sigma: p0.sigma, normalized: true, thr: 0, showMin: false, showImage: false,
    pts: [] as Pt[], img: new Float64Array(IMG * IMG) as Float64Array,
  };

  // ---- layout ----
  const main = htmlEl("div", "nc-main", "", host);
  const mainCap = htmlEl("div", "nc-cap", "", main);
  const svg = svgEl("svg", { viewBox: `0 0 ${S} ${S}`, width: "100%" }, main);
  const cv = htmlEl("canvas", "", "", main);
  cv.width = S; cv.height = S;
  const credit = htmlEl("div", "nc-credit", CREDIT, main);
  const mid = htmlEl("div", "nc-mid", "", host);
  const plot1 = svgEl("svg", { viewBox: `0 0 ${PW} ${PH}`, width: "100%" }, mid);
  const plot2 = svgEl("svg", { viewBox: `0 0 ${PW} ${PH}`, width: "100%" }, mid);
  const ro = htmlEl("div", "qb-readout", "", mid);
  const panel = htmlEl("div", "qb-panel", "", host);

  htmlEl("div", "nc-lab", "data", panel);
  const presetBtns = presets(panel, PRESETS.map(p => ({ label: p.label, apply: () => load(p) })));
  const sigP = slider(panel, "σ", 0.02, 0.3, 0.01, 0.08, v => { st.sigma = v; recompute(true); });
  const sigI = slider(panel, "σ", 5, 100, 5, 30, v => { st.sigma = v; recompute(true); });
  const normBox = checkbox(panel, "normalized: L x = μ D x", true, v => { st.normalized = v; recompute(true); });
  const thrS = slider(panel, "threshold", -1, 1, 0.01, 0, v => { st.thr = v; draw(); }, v => v.toFixed(2));
  const row = htmlEl("div", "qb-presets", "", panel);
  button(row, "best Ncut", () => { setBest(); draw(); });
  button(row, "threshold 0", () => { st.thr = 0; thrS.set(0); draw(); });
  const minBox = checkbox(panel, "show the minimum cut instead", false, v => { st.showMin = v; draw(); });
  const imgBox = checkbox(panel, "show the image", false, v => { st.showImage = v; draw(); });
  button(panel, "reset", () => load(p0));
  const note = htmlEl("div", "qb-note", "", panel);

  let bigCell: Float64Array | null = null;
  function load(p: AnyPreset): void {
    if (p.id === "cell" && !bigCell) { loadBig().then(b => { bigCell = b; load(p); }); return; }
    st.preset = p; st.mode = p.mode; st.sigma = p.sigma;
    if (p.mode === "points") st.pts = POINT_PRESETS.find(q => q.id === p.id)!.pts();
    else st.img = p.id === "cell" ? cellCrop(bigCell!) : diskImage();
    st.normalized = true; normBox.checked = true;
    st.showMin = false; minBox.checked = false; st.showImage = false; imgBox.checked = false;
    (p.mode === "points" ? sigP : sigI).set(p.sigma);
    sigP.input.parentElement!.style.display = p.mode === "points" ? "" : "none";
    sigI.input.parentElement!.style.display = p.mode === "image" ? "" : "none";
    imgBox.parentElement!.style.display = p.mode === "image" ? "" : "none";
    svg.style.display = p.mode === "points" ? "" : "none";
    cv.style.display = p.mode === "image" ? "" : "none";
    credit.style.display = p.id === "cell" ? "" : "none";
    presetBtns.forEach((b, i) => b.classList.toggle("active", PRESETS[i] === p));
    recompute(true);
  }

  // ---- computation ----
  let G: WGraph = { n: 0, p: [], q: [], w: [] };
  let F: Fiedler = { x: new Float64Array(0), mu: 0, residual: 0 };
  let SW: Sweep | null = null;
  let minSide: Uint8Array = new Uint8Array(0), minVal = 0;
  let pending = false;
  /** zero = true: the eigenvector changed, so start again from threshold 0 (its sign). */
  function recompute(zero = false): void {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      G = st.mode === "points" ? pointGraph(st.pts, st.sigma) : imageGraph(st.img, st.sigma);
      F = fiedler(G, st.normalized);
      SW = sweep(G, F.x);
      if (st.mode === "points") { const m = stoerWagner(G); minSide = m.side; minVal = m.value; }
      else {
        let kc = 0;
        SW.cut.forEach((v, k) => { if (v < SW!.cut[kc]) kc = k; });
        minSide = new Uint8Array(G.n);
        for (let j = 0; j <= kc; j++) minSide[SW.order[j]] = 1;
        minVal = SW.cut[kc];
      }
      if (zero) { st.thr = 0; thrS.set(0); }
      draw();
    });
  }

  /** Threshold halfway between the two values at the split with the smallest Ncut. */
  function setBest(): void {
    if (!SW || G.n < 2) return;
    let kb = 0;
    SW.ncut.forEach((v, k) => { if (v < SW!.ncut[kb]) kb = k; });
    const t = (F.x[SW.order[kb]] + F.x[SW.order[kb + 1]]) / 2;
    st.thr = Math.round(t * 100) / 100;
    // rounding must not move the split: fall back to the exact midpoint
    if (!(F.x[SW.order[kb]] < st.thr && st.thr <= F.x[SW.order[kb + 1]])) st.thr = t;
    thrS.set(st.thr);
  }

  // ---- drawing ----
  const toS = (v: number) => 10 + v * (S - 20);
  function draw(): void {
    if (!SW) return;
    const n = G.n, x = F.x;
    const inA = Uint8Array.from(x, v => (v < st.thr ? 1 : 0));
    const side = st.showMin ? minSide : inA;
    const kA = side.reduce((a, b) => a + b, 0);

    if (st.mode === "points") {
      svg.replaceChildren();
      let wmax = 0;
      for (const w of G.w) wmax = Math.max(wmax, w);
      for (let k = 0; k < G.p.length; k++) {
        const a = G.p[k], b = G.q[k], w = G.w[k] / (wmax || 1);
        const cut = side[a] !== side[b];
        if (w < 0.02 && !(cut && w > 1e-4)) continue;
        svgEl("line", {
          x1: toS(st.pts[a][0]), y1: toS(st.pts[a][1]), x2: toS(st.pts[b][0]), y2: toS(st.pts[b][1]),
          stroke: cut ? "#d62728" : "#888", "stroke-width": cut ? 1 + 2.5 * Math.sqrt(w) : 1.2, opacity: cut ? Math.max(0.35, Math.sqrt(w)) : 0.15 + 0.5 * w,
        }, svg);
      }
      st.pts.forEach((p, i) => {
        const fill = st.showMin ? (side[i] ? "#1f77b4" : "#ff7f0e") : css(divColor(x[i]));
        const c = svgEl("circle", { cx: toS(p[0]), cy: toS(p[1]), r: 8, fill, stroke: side[i] ? "#0b3c66" : "#8a3d00", "stroke-width": 2.2 }, svg);
        draggable(c, ev => svgPoint(svg, ev), (sx, sy) => {
          st.pts[i] = [Math.min(1, Math.max(0, (sx - 10) / (S - 20))), Math.min(1, Math.max(0, (sy - 10) / (S - 20)))];
          presetBtns.forEach(b => b.classList.remove("active"));
          recompute();
        });
      });
    } else {
      const ctx = cv.getContext("2d")!;
      const cs = S / IMG;
      for (let p = 0; p < n; p++) {
        const g = st.img[p];
        const col: [number, number, number] = st.showImage ? [g, g, g] : st.showMin ? (side[p] ? [31, 119, 180] : [255, 127, 14]) : divColor(x[p]);
        ctx.fillStyle = css(col.map(Math.round) as [number, number, number]);
        ctx.fillRect((p % IMG) * cs, Math.floor(p / IMG) * cs, cs + 0.5, cs + 0.5);
      }
      ctx.strokeStyle = st.showImage ? "#ffd400" : "#000"; ctx.lineWidth = 3;
      ctx.beginPath();
      for (let p = 0; p < n; p++) {
        const px = p % IMG, py = Math.floor(p / IMG);
        if (px + 1 < IMG && side[p] !== side[p + 1]) { ctx.moveTo((px + 1) * cs, py * cs); ctx.lineTo((px + 1) * cs, (py + 1) * cs); }
        if (py + 1 < IMG && side[p] !== side[p + IMG]) { ctx.moveTo(px * cs, (py + 1) * cs); ctx.lineTo((px + 1) * cs, (py + 1) * cs); }
      }
      ctx.stroke();
    }
    mainCap.textContent = st.showMin
      ? (st.mode === "points" ? "minimum cut, no balance" : "smallest cut along the sweep")
      : "Fiedler vector and the split";

    // plot 1: sorted Fiedler vector with the threshold
    plot1.replaceChildren();
    const L = 38, R = PW - 8, T = 22, B = PH - 26;
    const xr = (k: number) => L + (n > 1 ? k / (n - 1) : 0) * (R - L);
    const yv = (v: number) => B - (v + 1) / 2 * (B - T);
    svgEl("text", { x: L, y: 15 }, plot1).textContent = `sorted Fiedler vector (μ = ${sig3(F.mu)})`;
    for (const v of [-1, 0, 1]) {
      svgEl("line", { x1: L, x2: R, y1: yv(v), y2: yv(v), class: "grid" }, plot1);
      svgEl("text", { x: L - 5, y: yv(v) + 5, "text-anchor": "end" }, plot1).textContent = String(v);
    }
    const r = n > 200 ? 1.6 : 3.2;
    SW.order.forEach((i, k) => svgEl("circle", { cx: xr(k), cy: yv(x[i]), r, fill: css(divColor(x[i])), stroke: "#555", "stroke-width": n > 200 ? 0 : 0.6 }, plot1));
    svgEl("line", { x1: L, x2: R, y1: yv(st.thr), y2: yv(st.thr), stroke: "#000", "stroke-width": 1.5, "stroke-dasharray": "5 4" }, plot1);
    svgEl("text", { x: (L + R) / 2, y: PH - 6, "text-anchor": "middle" }, plot1).textContent = "nodes, sorted by their value";

    // plot 2: Ncut and cut along the sweep
    plot2.replaceChildren();
    const kSplit = inA.reduce((a, b) => a + b, 0);   // split after kSplit nodes
    const nc = SW.ncut, ct = SW.cut;
    const cmax = Math.max(...Array.from(ct)) || 1;
    const yn = (v: number) => B - Math.min(1, v / 2) * (B - T);
    const xs = (k: number) => L + (n > 2 ? k / (n - 2) : 0) * (R - L);
    svgEl("text", { x: L, y: 15, fill: "#9467bd" }, plot2).textContent = "Ncut";
    svgEl("text", { x: L + 50, y: 15, fill: "#2ca02c" }, plot2).textContent = "cut (scaled to its max)";
    for (const v of [0, 1, 2]) {
      svgEl("line", { x1: L, x2: R, y1: yn(v), y2: yn(v), class: "grid" }, plot2);
      svgEl("text", { x: L - 5, y: yn(v) + 5, "text-anchor": "end" }, plot2).textContent = String(v);
    }
    const poly = (vals: ArrayLike<number>, f: (v: number) => number, color: string) =>
      svgEl("polyline", { points: Array.from(vals, (v, k) => `${xs(k)},${f(v)}`).join(" "), fill: "none", stroke: color, "stroke-width": 2 }, plot2);
    poly(ct, v => B - v / cmax * (B - T) / 2, "#2ca02c");
    poly(nc, yn, "#9467bd");
    let kb = 0;
    nc.forEach((v, k) => { if (v < nc[kb]) kb = k; });
    svgEl("circle", { cx: xs(kb), cy: yn(nc[kb]), r: 5, fill: "#9467bd" }, plot2);
    if (kSplit >= 1 && kSplit <= n - 1) svgEl("line", { x1: xs(kSplit - 1), x2: xs(kSplit - 1), y1: T, y2: B, stroke: "#000", "stroke-width": 1.5, "stroke-dasharray": "5 4" }, plot2);
    svgEl("text", { x: (L + R) / 2, y: PH - 6, "text-anchor": "middle" }, plot2).textContent = "k = nodes below the split";

    // readout
    const k = kSplit;
    const lines = [`${st.normalized ? "L x = μ D x" : "L x = μ x"}: μ₂ = ${sig3(F.mu)}`];
    if (k >= 1 && k <= n - 1) {
      lines.push(`split: ${k} | ${n - k} nodes`,
        `cut ${sig3(ct[k - 1])}, vol ${sig3(SW.volA[k - 1])} | ${sig3(SW.volB[k - 1])}`,
        `Ncut ${sig3(nc[k - 1])}`, `best Ncut ${sig3(nc[kb])} at k = ${kb + 1}`);
    } else lines.push("no split: threshold outside");
    const mk = minSide.reduce((a, b) => a + b, 0);
    lines.push(`min cut${st.mode === "points" ? "" : " on sweep"} ${sig3(minVal)}: ${mk} | ${n - mk}`);
    ro.textContent = lines.join("\n");
    note.textContent = st.mode === "points"
      ? `Affinity w = exp(−d²/2σ²) between every pair of points; drag the points. Red lines: the edges that are cut. ${st.showMin ? `Showing the minimum cut: ${kA} | ${n - kA}.` : ""}`
      : "Pixel graph: neighbours within 2 pixels, w = exp(−ΔI²/2σ²)·exp(−d²/2·1.5²). Black line: the split.";
  }

  load(p0);
}

autoMount(NAME, mount);
