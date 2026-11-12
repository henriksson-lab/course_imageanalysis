// Static figures for the thresholding units (written into the unit folders).
import { Grey, boxFilter } from "/data/henriksson/github/course/imageanalysis/demos/src/lib/imageops";
import { intHistogram, histogram } from "/data/henriksson/github/course/imageanalysis/demos/src/lib/histogram";
import { rng, shuffle } from "/data/henriksson/github/course/imageanalysis/demos/src/lib/random";
import * as TH from "/data/henriksson/github/course/imageanalysis/demos/src/lib/threshold";
import * as HM from "/data/henriksson/github/course/imageanalysis/demos/src/histthresh/model";
import * as LM from "/data/henriksson/github/course/imageanalysis/demos/src/localthresh/model";
import { rocModel } from "/data/henriksson/github/course/imageanalysis/demos/src/threshroc/model";
import * as HY from "/data/henriksson/github/course/imageanalysis/demos/src/hysteresis/data";
import { pngGrey } from "/data/henriksson/github/course/imageanalysis/demos/test/png";
declare const require: any;
const fs = require("fs");
const R = "/data/henriksson/github/course/imageanalysis/";
const C = { fg: "#1b2a3a", grid: "#e3e3e3", c1: "#1f77b4", c2: "#d62728", c3: "#2ca02c", c4: "#ff7f0e", c5: "#9467bd", muted: "#888" };

function u8(img: Grey | ArrayLike<number>): Uint8Array { const d = "data" in (img as any) ? (img as Grey).data : (img as ArrayLike<number>); return Uint8Array.from(d as ArrayLike<number>, v => Math.max(0, Math.min(255, Math.round(v)))); }
function writePng(path: string, imgs: ArrayLike<number>[], cols: number, gap = 6, n = 256) {
  const rows = Math.ceil(imgs.length / cols), w = cols * (n + gap) - gap, h = rows * (n + gap) - gap, g = new Uint8Array(w * h).fill(255);
  imgs.forEach((im, k) => { const ox = (k % cols) * (n + gap), oy = Math.floor(k / cols) * (n + gap); for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) g[(oy + y) * w + ox + x] = im[y * n + x]; });
  fs.writeFileSync(R + path, pngGrey(g, w, h));
}

/** Tiny SVG plotter: panels with axes. */
class Svg {
  parts: string[] = [];
  constructor(public W: number, public H: number) {}
  add(s: string) { this.parts.push(s); }
  out(path: string) { fs.writeFileSync(R + path, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${this.W} ${this.H}" width="${this.W}" height="${this.H}" font-family="Helvetica, Arial, sans-serif" font-size="19" fill="${C.fg}">\n<rect width="${this.W}" height="${this.H}" fill="#fff"/>\n${this.parts.join("\n")}\n</svg>\n`); }
}
class Panel {
  constructor(public s: Svg, public x0: number, public y0: number, public w: number, public h: number, public xd: [number, number], public yd: [number, number]) {}
  X = (v: number) => this.x0 + (v - this.xd[0]) / (this.xd[1] - this.xd[0]) * this.w;
  Y = (v: number) => this.y0 + this.h - (v - this.yd[0]) / (this.yd[1] - this.yd[0]) * this.h;
  axes(xt: number[], yt: number[], xl = "", yl = "", title = "", fmt = (v: number) => String(v)) {
    const s = this.s;
    for (const v of xt) s.add(`<line x1="${this.X(v)}" x2="${this.X(v)}" y1="${this.y0}" y2="${this.y0 + this.h}" stroke="${C.grid}"/><text x="${this.X(v)}" y="${this.y0 + this.h + 22}" text-anchor="middle">${fmt(v)}</text>`);
    for (const v of yt) s.add(`<line y1="${this.Y(v)}" y2="${this.Y(v)}" x1="${this.x0}" x2="${this.x0 + this.w}" stroke="${C.grid}"/><text x="${this.x0 - 8}" y="${this.Y(v) + 6}" text-anchor="end">${fmt(v)}</text>`);
    s.add(`<path d="M${this.x0},${this.y0}V${this.y0 + this.h}H${this.x0 + this.w}" fill="none" stroke="#444" stroke-width="1.5"/>`);
    if (xl) s.add(`<text x="${this.x0 + this.w / 2}" y="${this.y0 + this.h + 46}" text-anchor="middle">${xl}</text>`);
    if (yl) s.add(`<text transform="rotate(-90 ${this.x0 - 52} ${this.y0 + this.h / 2})" x="${this.x0 - 52}" y="${this.y0 + this.h / 2}" text-anchor="middle">${yl}</text>`);
    if (title) s.add(`<text x="${this.x0 + this.w / 2}" y="${this.y0 - 10}" text-anchor="middle" font-weight="bold">${title}</text>`);
  }
  bars(edges: ArrayLike<number>, counts: ArrayLike<number>, color: string, op = 0.7) {
    let d = "";
    for (let i = 0; i < counts.length; i++) { const a = this.X(edges[i]), b = this.X(edges[i + 1]), y = this.Y(Math.min(counts[i], this.yd[1])); d += `M${a.toFixed(1)},${this.Y(0).toFixed(1)}V${y.toFixed(1)}H${b.toFixed(1)}V${this.Y(0).toFixed(1)}Z`; }
    this.s.add(`<path d="${d}" fill="${color}" fill-opacity="${op}" stroke="${color}" stroke-width="0.6"/>`);
  }
  line(xs: ArrayLike<number>, ys: ArrayLike<number>, color: string, w = 3, dash = "") {
    let d = ""; for (let i = 0; i < xs.length; i++) d += (i ? "L" : "M") + this.X(xs[i]).toFixed(1) + "," + this.Y(ys[i]).toFixed(1);
    this.s.add(`<path d="${d}" fill="none" stroke="${color}" stroke-width="${w}" ${dash ? `stroke-dasharray="${dash}"` : ""} stroke-linejoin="round"/>`);
  }
  area(xs: ArrayLike<number>, ys: ArrayLike<number>, color: string, op = 0.35) {
    let d = `M${this.X(xs[0])},${this.Y(0)}`; for (let i = 0; i < xs.length; i++) d += `L${this.X(xs[i]).toFixed(1)},${this.Y(ys[i]).toFixed(1)}`; d += `L${this.X(xs[xs.length - 1])},${this.Y(0)}Z`;
    this.s.add(`<path d="${d}" fill="${color}" fill-opacity="${op}" stroke="none"/>`);
  }
  vline(x: number, color: string, text = "", w = 3, dash = "") { this.s.add(`<line x1="${this.X(x)}" x2="${this.X(x)}" y1="${this.y0}" y2="${this.y0 + this.h}" stroke="${color}" stroke-width="${w}" ${dash ? `stroke-dasharray="${dash}"` : ""}/>` + (text ? `<text x="${this.X(x) + 6}" y="${this.y0 + 20}" fill="${color}">${text}</text>` : "")); }
  hline(y: number, color: string, text = "", w = 3) { this.s.add(`<line y1="${this.Y(y)}" y2="${this.Y(y)}" x1="${this.x0}" x2="${this.x0 + this.w}" stroke="${color}" stroke-width="${w}"/>` + (text ? `<text x="${this.x0 + this.w - 4}" y="${this.Y(y) - 6}" text-anchor="end" fill="${color}">${text}</text>` : "")); }
  dot(x: number, y: number, color: string, r = 7) { this.s.add(`<circle cx="${this.X(x)}" cy="${this.Y(y)}" r="${r}" fill="${color}" stroke="#fff" stroke-width="1.5"/>`); }
  text(x: number, y: number, t: string, color = C.fg, anchor = "start", size = 19) { this.s.add(`<text x="${this.X(x)}" y="${this.Y(y)}" fill="${color}" text-anchor="${anchor}" font-size="${size}">${t}</text>`); }
}

// ---------------- unit 1: histogram/
const truth = HM.blobTruth();
const base = { source: "blobs" as const, contrast: 100, noise: 8, median: false, blur: 0 };
const clean = HM.makeImage(base, truth, null);
const mask120 = Uint8Array.from(clean.data, v => (v > 120 ? 255 : 0));
writePng("histogram/fig-blobs.png", [u8(clean), mask120], 2, 8);
const sh = Float32Array.from(clean.data); shuffle(rng(3), sh);
writePng("histogram/fig-shuffle.png", [u8(clean), u8(sh)], 2, 8);
{ // histogram of the clean image with t
  const s = new Svg(560, 330), h = intHistogram(clean.data, 0, 255);
  const p = new Panel(s, 80, 20, 450, 240, [0, 255], [0, 2400]);
  p.axes([0, 50, 100, 150, 200, 250], [0, 1000, 2000], "grey level", "pixels");
  p.bars(Array.from({ length: 257 }, (_, i) => i - 0.5), h.counts, C.c1);
  p.vline(120, C.c4, "t = 120");
  p.text(60, 2200, "background", C.fg, "middle"); p.text(160, 900, "objects", C.fg, "middle");
  s.out("histogram/fig-hist.svg");
}
{ // bins: same image at 8, 32, 256 bins
  const img = HM.makeImage({ ...base, contrast: 60, noise: 14 }, truth, null);
  const s = new Svg(1050, 300);
  [8, 32, 256].forEach((B, k) => {
    const h = histogram(img.data, { bins: B, min: 0, max: 256 }), w = 256 / B;
    const mx = Math.max(...h.counts);
    const p = new Panel(s, 70 + k * 340, 40, 270, 190, [0, 256], [0, mx * 1.08]);
    p.axes([0, 100, 200], [], "grey level", k === 0 ? "pixels per bin" : "", `${B} bins`);
    p.bars(Array.from({ length: B + 1 }, (_, i) => i * w), h.counts, C.c1);
  });
  s.out("histogram/fig-bins.svg");
}
{ // noise fills the valley; a 3×3 mean restores it
  const noisy = HM.makeImage({ ...base, noise: 35 }, truth, null);
  const sm = boxFilter(noisy, 1);
  const s = new Svg(1000, 310);
  ([[noisy, "noise σ = 35"], [sm, "after a 3 × 3 mean"]] as const).forEach(([img, title], k) => {
    const h = intHistogram(Uint8Array.from(img.data, v => Math.round(v)), 0, 255);
    const p = new Panel(s, 90 + k * 480, 40, 400, 200, [0, 255], [0, 1500]);
    p.axes([0, 50, 100, 150, 200, 250], [0, 500, 1000, 1500], "grey level", k === 0 ? "pixels" : "", title);
    p.bars(Array.from({ length: 257 }, (_, i) => i - 0.5), h.counts, k ? C.c3 : C.c1);
  });
  s.out("histogram/fig-noise.svg");
}

// ---------------- unit 2: otsu/
{ // the worked example split at t = 3
  const h = [2, 5, 3, 1, 1, 2, 4, 2], st = TH.splitStats(h, 3);
  const s = new Svg(560, 330), p = new Panel(s, 80, 30, 450, 230, [-0.6, 7.6], [0, 7]);
  p.axes([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 4, 6], "grey level i", "count hᵢ");
  h.forEach((c, i) => { p.bars([i - 0.4, i + 0.4], [c], i <= 3 ? C.c1 : C.c4, 0.85); p.text(i, c + 0.15, String(c), C.fg, "middle"); });
  p.vline(3.5, C.fg, "", 3); p.text(3.6, 5.6, "t = 3", C.fg, "start", 18);
  p.dot(st.mu0, 0.3, C.c1, 9); p.dot(st.mu1, 0.3, C.c4, 9);
  p.text(1.5, 6.4, `class 0: ω₀ = 0.55, μ₀ = 1.27`, C.c1, "middle", 15); p.text(5.7, 6.4, `class 1: ω₁ = 0.45, μ₁ = 5.78`, C.c4, "middle", 15);
  s.out("otsu/fig-split.svg");
}
{ // Bayes picture: two weighted Gaussians and the error tails
  const [m0, m1, sd, w1] = [50, 150, 20, 0.1];
  const xs = Array.from({ length: 251 }, (_, i) => i), f0 = xs.map(x => (1 - w1) * TH.normalPdf(x, m0, sd)), f1 = xs.map(x => w1 * TH.normalPdf(x, m1, sd));
  const tb = TH.gaussianCrossings(m0, sd, m1, sd, w1)[0];
  const s = new Svg(620, 330), p = new Panel(s, 90, 20, 500, 240, [0, 250], [0, 0.019]);
  p.axes([0, 50, 100, 150, 200, 250], [], "grey level x", "ω f(x)");
  p.area(xs.filter(x => x > tb), f0.filter((_, i) => xs[i] > tb), C.c1, 0.45);
  p.area(xs.filter(x => x <= tb), f1.filter((_, i) => xs[i] <= tb), C.c4, 0.55);
  p.line(xs, f0, C.c1); p.line(xs, f1, C.c4);
  p.vline(tb, C.c2, "", 3); p.text(tb + 4, 0.0165, `crossing ${tb.toFixed(1)}`, C.c2, "start", 17);
  p.vline(100, C.muted, "", 2, "6 4"); p.text(97, 0.012, "midpoint 100", C.muted, "end", 17);
  p.text(50, 0.0185, "background ω₀f₀ (90 %)", C.c1, "middle", 17); p.text(175, 0.0035, "object ω₁f₁ (10 %)", C.c4, "middle", 17);
  s.out("otsu/fig-bayes.svg");
}

// ---------------- unit 3: local-threshold/
{
  const lt = LM.discTruth(), BASE: LM.Settings = { source: "discs", method: "global", r: 25, k: 0.3, C: 10, bg: 30, light: 1.0, vignette: 0.3, noise: 6 };
  const img = LM.makeImage(BASE, lt, null), g = LM.segment(img, BASE), sv = LM.segment(img, { ...BASE, method: "sauvola" });
  writePng("local-threshold/fig-gradient.png", [u8(img), Uint8Array.from(g.mask, v => (v ? 0 : 255)), Uint8Array.from(sv.mask, v => (v ? 0 : 255))], 3, 8);
  // profile along row 128 with global and Sauvola thresholds
  const s = new Svg(620, 320), p = new Panel(s, 80, 20, 510, 230, [0, 255], [0, 255]);
  p.axes([0, 50, 100, 150, 200, 250], [0, 100, 200], "x along row 128", "grey level");
  const xs = Array.from({ length: 256 }, (_, i) => i), row = 128;
  p.line(xs, xs.map(x => img.data[row * 256 + x]), "#555", 1.3);
  p.line(xs, xs.map(x => g.T.data[row * 256 + x]), C.c2, 3, "8 5");
  p.line(xs, xs.map(x => sv.T.data[row * 256 + x]), C.c4, 3.5);
  p.text(4, 138, "global Otsu", C.c2, "start", 17); p.text(110, 50, "Sauvola T(x)", C.c4, "middle", 17);
  s.out("local-threshold/fig-profile.svg");
}
{ // 1D hysteresis
  const prof = HY.profile1d(2), keep = TH.hysteresis1d(prof, 62, 105);
  const s = new Svg(620, 300), p = new Panel(s, 80, 20, 510, 210, [0, 199], [0, 220]);
  p.axes([0, 50, 100, 150], [0, 100, 200], "position", "intensity");
  p.line(Array.from(prof, (_, i) => i), prof, "#aaa", 1.5);
  prof.forEach((v, i) => p.dot(i, v, v > 105 ? C.c2 : keep[i] ? C.c4 : v > 62 ? C.c1 : "#ccc", 3.6));
  p.hline(105, C.c2, "t_high"); p.hline(62, C.c1, "t_low");
  s.out("local-threshold/fig-hyst1d.svg");
}

// ---------------- unit 4: seg-eval/
{
  const M = rocModel({ fraction: 0.1, contrast: 60, noise: 15, light: 0, smooth: false });
  const s = new Svg(760, 340);
  const a = new Panel(s, 80, 30, 260, 240, [0, 1], [0, 1]), b = new Panel(s, 470, 30, 260, 240, [0, 1], [0, 1]);
  a.axes([0, 0.5, 1], [0, 0.5, 1], "FPR", "TPR (recall)", "ROC"); b.axes([0, 0.5, 1], [0, 0.5, 1], "recall", "precision", "precision–recall");
  a.line([0, 1], [0, 1], "#bbb", 1.5, "5 5");
  a.line(M.pts.map(q => q.fpr), M.pts.map(q => q.tpr), C.c1); b.line(M.pts.map(q => q.tpr), M.pts.map(q => q.precision), C.c1);
  for (const [t, col] of [[M.marks.otsu, C.c5], [M.marks.dice, C.c3]] as const) { const q = M.at(t); a.dot(q.fpr, q.tpr, col, 8); b.dot(q.tpr, q.precision, col, 8); }
  a.text(0.32, 0.75, "Otsu", C.c5, "start", 17); a.text(0.08, 0.6, "max Dice", C.c3, "start", 17);
  s.out("seg-eval/fig-roc.svg");
  writePng("seg-eval/fig-truth.png", [u8(M.img), Uint8Array.from(M.truth, v => (v ? 255 : 0))], 2, 8);
}
console.log("figures written");
