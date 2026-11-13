// Shared helpers for the ray-tracing day's slide figures (raycast/, whitted/, bvh/, pathtrace/):
// a tiny PNG writer, SVG snippets, nearest-neighbour upscaling and Okabe–Ito colours.
declare const require: (m: string) => any;
const fs = require("fs"), zlib = require("zlib");

const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
const crc32 = (b: Uint8Array) => { let c = 0xffffffff; for (const x of b) c = CRC[(c ^ x) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length), dv = new DataView(out.buffer);
  dv.setUint32(0, data.length); for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8); dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}
export function pngRGB(rgb: Uint8Array, w: number, h: number): Uint8Array {
  const ihdr = new Uint8Array(13), dv = new DataView(ihdr.buffer);
  dv.setUint32(0, w); dv.setUint32(4, h); ihdr[8] = 8; ihdr[9] = 2;
  const raw = new Uint8Array((3 * w + 1) * h);
  for (let y = 0; y < h; y++) raw.set(rgb.subarray(3 * y * w, 3 * (y + 1) * w), y * (3 * w + 1) + 1);
  const parts = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", new Uint8Array(zlib.deflateSync(raw))), chunk("IEND", new Uint8Array(0))];
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0)); let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}
/** Nearest-neighbour upscale of an RGB8 image by an integer factor. */
export function upscale(rgb: Uint8Array, w: number, h: number, k: number): Uint8Array {
  const out = new Uint8Array(3 * w * k * h * k);
  for (let y = 0; y < h * k; y++) for (let x = 0; x < w * k; x++) for (let c = 0; c < 3; c++) out[3 * (y * w * k + x) + c] = rgb[3 * (Math.floor(y / k) * w + Math.floor(x / k)) + c];
  return out;
}
/** An <image> element with the PNG inlined (so the SVG is self-contained). */
export const imgEl = (x: number, y: number, w: number, h: number, png: Uint8Array, pixelated = true) =>
  `<image x="${x}" y="${y}" width="${w}" height="${h}" ${pixelated ? 'style="image-rendering:pixelated"' : ""} href="data:image/png;base64,${Buffer.from(png).toString("base64")}"/>`;
declare const Buffer: any;

export const save = (path: string, s: string | Uint8Array) => { fs.writeFileSync(path, s); console.log("wrote", path); };
export const svg = (w: number, h: number, body: string, fs = 20) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" font-family="system-ui, Helvetica, Arial, sans-serif" font-size="${fs}">\n<rect width="${w}" height="${h}" fill="#fff"/>\n${body}\n</svg>\n`;
export const f1 = (v: number) => v.toFixed(1);
export const txt = (x: number, y: number, s: string, o: string = "") => `<text x="${f1(x)}" y="${f1(y)}" ${o}>${s}</text>`;
export const line = (x1: number, y1: number, x2: number, y2: number, o: string) => `<line x1="${f1(x1)}" y1="${f1(y1)}" x2="${f1(x2)}" y2="${f1(y2)}" ${o}/>`;
export const circ = (x: number, y: number, r: number, o: string) => `<circle cx="${f1(x)}" cy="${f1(y)}" r="${f1(r)}" ${o}/>`;
export const poly = (pts: number[][], o: string) => `<polygon points="${pts.map(p => `${f1(p[0])},${f1(p[1])}`).join(" ")}" ${o}/>`;
export const arrowDefs = `<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs>`;
// Okabe–Ito
export const C = { orange: "#E69F00", sky: "#56B4E9", green: "#009E73", yellow: "#F0E442", blue: "#0072B2", red: "#D55E00", purple: "#CC79A7", grey: "#777777", black: "#000000" };
