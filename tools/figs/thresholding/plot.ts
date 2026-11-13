// Shared helpers for the static figures of the thresholding day (histogram/, otsu/,
// local-threshold/, seg-eval/): a tiny SVG plotter and PNG writers. Same style as figs.ts.
import { Grey } from "/data/henriksson/github/course/imageanalysis/demos/src/lib/imageops";
import { pngGrey } from "/data/henriksson/github/course/imageanalysis/demos/test/png";
declare const require: any;
const fs = require("fs");
const zlib = require("zlib");
export const R = "/data/henriksson/github/course/imageanalysis/";
export const C = { fg: "#1b2a3a", grid: "#e3e3e3", c1: "#1f77b4", c2: "#d62728", c3: "#2ca02c", c4: "#ff7f0e", c5: "#9467bd", muted: "#888" };
export type RGB = [number, number, number];
export const TPc: RGB = [44, 160, 44], FPc: RGB = [214, 39, 40], FNc: RGB = [31, 119, 220];

export function u8(img: Grey | ArrayLike<number>): Uint8Array { const d = "data" in (img as any) ? (img as Grey).data : (img as ArrayLike<number>); return Uint8Array.from(d as ArrayLike<number>, v => Math.max(0, Math.min(255, Math.round(v)))); }

/** Tile n × n grey images into a row/grid with white gaps. */
export function writePng(path: string, imgs: ArrayLike<number>[], cols: number, gap = 8, n = 256) {
  const rows = Math.ceil(imgs.length / cols), w = cols * (n + gap) - gap, h = rows * (n + gap) - gap, g = new Uint8Array(w * h).fill(255);
  imgs.forEach((im, k) => { const ox = (k % cols) * (n + gap), oy = Math.floor(k / cols) * (n + gap); for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) g[(oy + y) * w + ox + x] = im[y * n + x]; });
  fs.writeFileSync(R + path, pngGrey(g, w, h));
}

const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(b: Uint8Array): number { let c = 0xffffffff; for (const x of b) c = CRC[(c ^ x) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }
function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length), dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}
/** RGB PNG, rgb = w*h*3 bytes. */
export function pngRgb(rgb: Uint8Array, w: number, h: number): Uint8Array {
  const ihdr = new Uint8Array(13), dv = new DataView(ihdr.buffer);
  dv.setUint32(0, w); dv.setUint32(4, h); ihdr[8] = 8; ihdr[9] = 2;
  const raw = new Uint8Array((3 * w + 1) * h);
  for (let y = 0; y < h; y++) raw.set(rgb.subarray(y * 3 * w, (y + 1) * 3 * w), y * (3 * w + 1) + 1);
  const idat = new Uint8Array(zlib.deflateSync(raw));
  const parts = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", idat), chunk("IEND", new Uint8Array(0))];
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}
/** Grey image (n × n) with coloured mask layers blended in, as an RGB array. */
export function overlay(img: ArrayLike<number>, layers: { mask: ArrayLike<number>; rgb: RGB; alpha?: number }[], n = 256): Uint8Array {
  const out = new Uint8Array(n * n * 3);
  for (let i = 0; i < n * n; i++) {
    const g = Math.max(0, Math.min(255, img[i])); let r = g, gg = g, b = g;
    for (const L of layers) if (L.mask[i]) { const a = L.alpha ?? 0.5; r = (1 - a) * r + a * L.rgb[0]; gg = (1 - a) * gg + a * L.rgb[1]; b = (1 - a) * b + a * L.rgb[2]; }
    out[3 * i] = r; out[3 * i + 1] = gg; out[3 * i + 2] = b;
  }
  return out;
}
/** Grey → RGB. */
export const greyRgb = (img: ArrayLike<number>): Uint8Array => { const o = new Uint8Array(img.length * 3); for (let i = 0; i < img.length; i++) o[3 * i] = o[3 * i + 1] = o[3 * i + 2] = Math.max(0, Math.min(255, Math.round(img[i]))); return o; };
/** Tile n × n RGB images. */
export function writeRgbTiles(path: string, imgs: Uint8Array[], cols: number, gap = 8, n = 256) {
  const rows = Math.ceil(imgs.length / cols), w = cols * (n + gap) - gap, h = rows * (n + gap) - gap, g = new Uint8Array(w * h * 3).fill(255);
  imgs.forEach((im, k) => { const ox = (k % cols) * (n + gap), oy = Math.floor(k / cols) * (n + gap); for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) for (let c = 0; c < 3; c++) g[3 * ((oy + y) * w + ox + x) + c] = im[3 * (y * n + x) + c]; });
  fs.writeFileSync(R + path, pngRgb(g, w, h));
}

/** Tiny SVG plotter: panels with axes. */
export class Svg {
  parts: string[] = [];
  constructor(public W: number, public H: number) {}
  add(s: string) { this.parts.push(s); }
  text(x: number, y: number, t: string, color = C.fg, anchor = "start", size = 19, extra = "") { this.add(`<text x="${x}" y="${y}" fill="${color}" text-anchor="${anchor}" font-size="${size}" ${extra}>${t}</text>`); }
  out(path: string) { fs.writeFileSync(R + path, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${this.W} ${this.H}" width="${this.W}" height="${this.H}" font-family="Helvetica, Arial, sans-serif" font-size="19" fill="${C.fg}">\n<rect width="${this.W}" height="${this.H}" fill="#fff"/>\n${this.parts.join("\n")}\n</svg>\n`); }
}
export class Panel {
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
  rect(x0: number, x1: number, y0: number, y1: number, color: string, op = 1, stroke = "none") { const a = this.X(x0), b = this.X(x1), c = this.Y(y1), d = this.Y(y0); this.s.add(`<rect x="${Math.min(a, b).toFixed(1)}" y="${Math.min(c, d).toFixed(1)}" width="${Math.abs(b - a).toFixed(1)}" height="${Math.abs(d - c).toFixed(1)}" fill="${color}" fill-opacity="${op}" stroke="${stroke}"/>`); }
  line(xs: ArrayLike<number>, ys: ArrayLike<number>, color: string, w = 3, dash = "") {
    let d = ""; for (let i = 0; i < xs.length; i++) d += (i ? "L" : "M") + this.X(xs[i]).toFixed(1) + "," + this.Y(ys[i]).toFixed(1);
    this.s.add(`<path d="${d}" fill="none" stroke="${color}" stroke-width="${w}" ${dash ? `stroke-dasharray="${dash}"` : ""} stroke-linejoin="round"/>`);
  }
  area(xs: ArrayLike<number>, ys: ArrayLike<number>, color: string, op = 0.35) {
    let d = `M${this.X(xs[0])},${this.Y(0)}`; for (let i = 0; i < xs.length; i++) d += `L${this.X(xs[i]).toFixed(1)},${this.Y(ys[i]).toFixed(1)}`; d += `L${this.X(xs[xs.length - 1])},${this.Y(0)}Z`;
    this.s.add(`<path d="${d}" fill="${color}" fill-opacity="${op}" stroke="none"/>`);
  }
  vline(x: number, color: string, text = "", w = 3, dash = "") { this.s.add(`<line x1="${this.X(x)}" x2="${this.X(x)}" y1="${this.y0}" y2="${this.y0 + this.h}" stroke="${color}" stroke-width="${w}" ${dash ? `stroke-dasharray="${dash}"` : ""}/>` + (text ? `<text x="${this.X(x) + 6}" y="${this.y0 + 20}" fill="${color}">${text}</text>` : "")); }
  hline(y: number, color: string, text = "", w = 3, dash = "") { this.s.add(`<line y1="${this.Y(y)}" y2="${this.Y(y)}" x1="${this.x0}" x2="${this.x0 + this.w}" stroke="${color}" stroke-width="${w}" ${dash ? `stroke-dasharray="${dash}"` : ""}/>` + (text ? `<text x="${this.x0 + this.w - 4}" y="${this.Y(y) - 6}" text-anchor="end" fill="${color}">${text}</text>` : "")); }
  dot(x: number, y: number, color: string, r = 7) { this.s.add(`<circle cx="${this.X(x)}" cy="${this.Y(y)}" r="${r}" fill="${color}" stroke="#fff" stroke-width="1.5"/>`); }
  arrow(x1: number, y1: number, x2: number, y2: number, color: string, w = 2.5) {
    const a = this.X(x1), b = this.Y(y1), c = this.X(x2), d = this.Y(y2), ang = Math.atan2(d - b, c - a), L = 11;
    this.s.add(`<line x1="${a}" y1="${b}" x2="${c}" y2="${d}" stroke="${color}" stroke-width="${w}"/><path d="M${c},${d}L${c - L * Math.cos(ang - 0.4)},${d - L * Math.sin(ang - 0.4)}L${c - L * Math.cos(ang + 0.4)},${d - L * Math.sin(ang + 0.4)}Z" fill="${color}"/>`);
  }
  text(x: number, y: number, t: string, color = C.fg, anchor = "start", size = 19) { this.s.add(`<text x="${this.X(x)}" y="${this.Y(y)}" fill="${color}" text-anchor="${anchor}" font-size="${size}">${t}</text>`); }
}

/** Build and run a figure script: node_modules/.bin/esbuild <ts> --bundle --platform=node --outfile=<tmp>.js && node <tmp>.js */
