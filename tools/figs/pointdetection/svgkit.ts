// Small SVG helpers shared by the point-detection day's figure scripts
// (tools/figs/pointdetection/figs.ts keeps its own copy; tools/figs/spot-*/figs.ts import this).
// Colours: Okabe–Ito (colour-blind safe). Matched / false / missed also differ by marker shape.
import * as fs from "fs";
import * as zlib from "zlib";

export const C = {
  tp: "#009E73", fp: "#D55E00", fn: "#F0E442", fnDark: "#B8860B", blue: "#0072B2", sky: "#56B4E9",
  orange: "#E69F00", purple: "#CC79A7", grey: "#777", fg: "#222",
};
export const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
export const f1 = (v: number) => v.toFixed(1);
export function svg(w: number, h: number, body: string): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" font-family="Helvetica, Arial, sans-serif" font-size="20">\n<rect width="${w}" height="${h}" fill="#fff"/>\n${body}\n</svg>\n`;
}
export type TextOpt = { size?: number; anchor?: string; fill?: string; weight?: string; italic?: boolean };
export const text = (x: number, y: number, s: string, o: TextOpt = {}) =>
  `<text x="${f1(x)}" y="${f1(y)}" font-size="${o.size ?? 20}" text-anchor="${o.anchor ?? "middle"}" fill="${o.fill ?? C.fg}"${o.weight ? ` font-weight="${o.weight}"` : ""}${o.italic ? ` font-style="italic"` : ""}>${esc(s)}</text>`;
export const line = (x1: number, y1: number, x2: number, y2: number, st = C.fg, w = 2, dash = "") =>
  `<line x1="${f1(x1)}" y1="${f1(y1)}" x2="${f1(x2)}" y2="${f1(y2)}" stroke="${st}" stroke-width="${w}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
export const circle = (x: number, y: number, r: number, o: { fill?: string; stroke?: string; w?: number; dash?: string } = {}) =>
  `<circle cx="${f1(x)}" cy="${f1(y)}" r="${f1(r)}" fill="${o.fill ?? "none"}" stroke="${o.stroke ?? "none"}" stroke-width="${o.w ?? 2}"${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}/>`;
export const rect = (x: number, y: number, w: number, h: number, o: { fill?: string; stroke?: string; w?: number; opacity?: number; dash?: string } = {}) =>
  `<rect x="${f1(x)}" y="${f1(y)}" width="${f1(w)}" height="${f1(h)}" fill="${o.fill ?? "none"}" stroke="${o.stroke ?? "none"}" stroke-width="${o.w ?? 2}"${o.opacity !== undefined ? ` fill-opacity="${o.opacity}"` : ""}${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}/>`;
export const path = (pts: [number, number][], o: { stroke?: string; w?: number; dash?: string; fill?: string; opacity?: number } = {}) =>
  `<path d="M${pts.map(p => f1(p[0]) + "," + f1(p[1])).join("L")}${o.fill ? "Z" : ""}" fill="${o.fill ?? "none"}"${o.opacity !== undefined ? ` fill-opacity="${o.opacity}"` : ""} stroke="${o.stroke ?? C.fg}" stroke-width="${o.w ?? 2.5}" stroke-linejoin="round"${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}/>`;
/** A cross marker (false detections). */
export const cross = (x: number, y: number, r: number, st: string, w = 2.5) => line(x - r, y - r, x + r, y + r, st, w) + line(x - r, y + r, x + r, y - r, st, w);
export const arrow = (x1: number, y1: number, x2: number, y2: number, st = C.fg, w = 2.5) => {
  const a = Math.atan2(y2 - y1, x2 - x1), L = 11;
  return line(x1, y1, x2, y2, st, w) + path([[x2 - L * Math.cos(a - 0.4), y2 - L * Math.sin(a - 0.4)], [x2, y2], [x2 - L * Math.cos(a + 0.4), y2 - L * Math.sin(a + 0.4)]], { stroke: st, w });
};

export function axes(x0: number, y0: number, w: number, h: number, xd: [number, number], yd: [number, number], xt: number[], yt: number[], xl: string, yl: string,
  fmt = (v: number) => String(v), logx = false, logy = false, fs = 18) {
  const tx = (v: number) => (logx ? Math.log10(v) : v), ty = (v: number) => (logy ? Math.log10(v) : v);
  const X = (v: number) => x0 + (tx(v) - tx(xd[0])) / (tx(xd[1]) - tx(xd[0])) * w, Y = (v: number) => y0 + h - (ty(v) - ty(yd[0])) / (ty(yd[1]) - ty(yd[0])) * h;
  let s = "";
  for (const t of xt) s += line(X(t), y0, X(t), y0 + h, "#e3e3e3", 1) + line(X(t), y0 + h, X(t), y0 + h + 6, C.fg, 1.5) + text(X(t), y0 + h + 25, fmt(t), { size: fs });
  for (const t of yt) s += line(x0, Y(t), x0 + w, Y(t), "#e3e3e3", 1) + line(x0 - 6, Y(t), x0, Y(t), C.fg, 1.5) + text(x0 - 10, Y(t) + 6, fmt(t), { size: fs, anchor: "end" });
  s += path([[x0, y0], [x0, y0 + h], [x0 + w, y0 + h]], { w: 1.5 });
  if (xl) s += text(x0 + w / 2, y0 + h + 50, xl);
  if (yl) s += `<text x="${x0 - 55}" y="${y0 + h / 2}" font-size="20" text-anchor="middle" transform="rotate(-90 ${x0 - 55} ${y0 + h / 2})">${esc(yl)}</text>`;
  return { X, Y, s };
}

// ---- PNG (grey, 8 bit) embedded as a data URI, drawn pixelated
const crcTable = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(b: Buffer): number { let c = 0xffffffff; for (const v of b) c = crcTable[(c ^ v) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }
function chunk(type: string, data: Buffer): Buffer {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type, "ascii"), data]), crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}
/** 8-bit greyscale PNG of w × h values in 0..255. */
export function pngGrey(v: Uint8Array, w: number, h: number): Buffer {
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 0; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  const raw = Buffer.alloc((w + 1) * h);
  for (let y = 0; y < h; y++) { raw[y * (w + 1)] = 0; for (let x = 0; x < w; x++) raw[y * (w + 1) + 1 + x] = v[y * w + x]; }
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", zlib.deflateSync(raw, { level: 9 })), chunk("IEND", Buffer.alloc(0))]);
}
/** Grey image (crop x0..x0+cw, y0..y0+ch of a w-wide array) as an embedded pixelated <image>. lo/hi: display range. */
export function greyImage(data: ArrayLike<number>, w: number, cx: number, cy: number, cw: number, ch: number, X: number, Y: number, px: number, lo?: number, hi?: number): string {
  let a = Infinity, b = -Infinity;
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) { const v = data[(cy + y) * w + cx + x]; a = Math.min(a, v); b = Math.max(b, v); }
  lo = lo ?? a; hi = hi ?? b;
  const u = new Uint8Array(cw * ch);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) u[y * cw + x] = Math.round(255 * Math.max(0, Math.min(1, (data[(cy + y) * w + cx + x] - lo) / (hi - lo || 1))));
  const uri = "data:image/png;base64," + pngGrey(u, cw, ch).toString("base64");
  return `<image x="${f1(X)}" y="${f1(Y)}" width="${f1(cw * px)}" height="${f1(ch * px)}" preserveAspectRatio="none" style="image-rendering:pixelated" image-rendering="optimizeSpeed" href="${uri}" xlink:href="${uri}"/>` +
    rect(X, Y, cw * px, ch * px, { stroke: "#555", w: 1 });
}
export const out = (p: string, s: string) => { fs.writeFileSync(p, s); console.log("wrote", p, (s.length / 1024).toFixed(0) + " KB"); };
/** Normal CDF. */
export function Phi(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x)), d = 0.3989423 * Math.exp(-x * x / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x > 0 ? 1 - p : p;
}
