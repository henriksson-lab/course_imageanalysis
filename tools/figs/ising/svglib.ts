// Small SVG/PNG helpers shared by the Ising-day figure scripts (ising/, mcmc/, mrf-denoise/, gmrf/).
// Each unit script is run as
//   node_modules/.bin/esbuild tools/figs/<unit>/figs.ts --bundle --platform=node --log-level=warning | node
// from the repository root; outputs go into the unit folders as fig-*.svg.
declare const require: any;
const fs = require("fs");
const zlib = require("zlib");

export const R = "/data/henriksson/github/course/imageanalysis/";
/** Colours: tab10, as in the demos. plus = +1 spin / object, minus = -1 spin. */
export const C = {
  fg: "#222", axis: "#444", grid: "#e3e3e3", muted: "#888",
  plus: "#ff7f0e", minus: "#1f77b4", blue: "#1f77b4", orange: "#ff7f0e", green: "#2ca02c", red: "#d62728", purple: "#9467bd", grey: "#999",
};
export const FONT = "system-ui, sans-serif";

export const fmt = (v: number, d = 2) => {
  const s = v.toFixed(d);
  return s.startsWith("-") ? "−" + s.slice(1) : s;
};

export class Svg {
  parts: string[] = [];
  constructor(public W: number, public H: number) {}
  add(s: string) { this.parts.push(s); return this; }
  text(x: number, y: number, t: string, o: { size?: number; anchor?: string; color?: string; weight?: string; italic?: boolean } = {}) {
    this.add(`<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" font-size="${o.size ?? 18}" text-anchor="${o.anchor ?? "start"}" fill="${o.color ?? C.fg}"${o.weight ? ` font-weight="${o.weight}"` : ""}${o.italic ? ` font-style="italic"` : ""}>${t}</text>`);
    return this;
  }
  line(x1: number, y1: number, x2: number, y2: number, color = C.axis, w = 2, dash = "") {
    this.add(`<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${color}" stroke-width="${w}"${dash ? ` stroke-dasharray="${dash}"` : ""} stroke-linecap="round"/>`);
    return this;
  }
  rect(x: number, y: number, w: number, h: number, fill: string, o: { stroke?: string; sw?: number; rx?: number } = {}) {
    this.add(`<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" fill="${fill}"${o.stroke ? ` stroke="${o.stroke}" stroke-width="${o.sw ?? 1}"` : ""}${o.rx ? ` rx="${o.rx}"` : ""}/>`);
    return this;
  }
  circle(x: number, y: number, r: number, fill: string, stroke = "none", sw = 1) {
    this.add(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`);
    return this;
  }
  arrow(x1: number, y1: number, x2: number, y2: number, color = C.axis, w = 2) {
    const a = Math.atan2(y2 - y1, x2 - x1), L = 11;
    const p1 = [x2 - L * Math.cos(a - 0.4), y2 - L * Math.sin(a - 0.4)], p2 = [x2 - L * Math.cos(a + 0.4), y2 - L * Math.sin(a + 0.4)];
    this.line(x1, y1, x2 - 0.6 * L * Math.cos(a), y2 - 0.6 * L * Math.sin(a), color, w);
    this.add(`<path d="M${x2.toFixed(1)},${y2.toFixed(1)}L${p1[0].toFixed(1)},${p1[1].toFixed(1)}L${p2[0].toFixed(1)},${p2[1].toFixed(1)}Z" fill="${color}"/>`);
    return this;
  }
  /** Embed an RGB image (w x h, 3 bytes per pixel) at (x, y) scaled to dw x dh, pixelated. */
  image(x: number, y: number, dw: number, dh: number, rgb: Uint8Array, w: number, h: number, border = true) {
    const b64 = Buffer.from(pngRGB(rgb, w, h)).toString("base64");
    if (border) this.rect(x - 1, y - 1, dw + 2, dh + 2, "none", { stroke: "#bbb" });
    this.add(`<image x="${x}" y="${y}" width="${dw}" height="${dh}" style="image-rendering:pixelated" preserveAspectRatio="none" href="data:image/png;base64,${b64}"/>`);
    return this;
  }
  out(path: string) {
    fs.writeFileSync(R + path, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${this.W} ${this.H}" font-family="${FONT}">\n<rect width="${this.W}" height="${this.H}" fill="#fff"/>\n${this.parts.join("\n")}\n</svg>\n`);
    console.log("wrote", path);
  }
}

/** A plot panel with linear (or log10 x) axes. */
export class Panel {
  constructor(public s: Svg, public x0: number, public y0: number, public w: number, public h: number,
    public xd: [number, number], public yd: [number, number], public logx = false) {}
  private tx = (v: number) => (this.logx ? Math.log10(v) : v);
  X = (v: number) => this.x0 + (this.tx(v) - this.tx(this.xd[0])) / (this.tx(this.xd[1]) - this.tx(this.xd[0])) * this.w;
  Y = (v: number) => this.y0 + this.h - (v - this.yd[0]) / (this.yd[1] - this.yd[0]) * this.h;
  axes(xt: number[], yt: number[], xl = "", yl = "", xf = (v: number) => String(v).replace("-", "−"), yf = (v: number) => String(v).replace("-", "−")) {
    const s = this.s;
    for (const v of xt) { s.line(this.X(v), this.y0, this.X(v), this.y0 + this.h, C.grid, 1); s.text(this.X(v), this.y0 + this.h + 22, xf(v), { anchor: "middle", size: 17, color: C.axis }); }
    for (const v of yt) { s.line(this.x0, this.Y(v), this.x0 + this.w, this.Y(v), C.grid, 1); s.text(this.x0 - 8, this.Y(v) + 6, yf(v), { anchor: "end", size: 17, color: C.axis }); }
    s.add(`<path d="M${this.x0},${this.y0}V${this.y0 + this.h}H${this.x0 + this.w}" fill="none" stroke="${C.axis}" stroke-width="1.5"/>`);
    if (xl) s.text(this.x0 + this.w / 2, this.y0 + this.h + 48, xl, { anchor: "middle", size: 18 });
    if (yl) s.add(`<text transform="rotate(-90 ${this.x0 - 50} ${this.y0 + this.h / 2})" x="${this.x0 - 50}" y="${this.y0 + this.h / 2}" text-anchor="middle" font-size="18" fill="${C.fg}">${yl}</text>`);
  }
  line(xs: ArrayLike<number>, ys: ArrayLike<number>, color: string, w = 3, dash = "") {
    let d = "";
    for (let i = 0; i < xs.length; i++) d += (i ? "L" : "M") + this.X(xs[i]).toFixed(1) + "," + this.Y(ys[i]).toFixed(1);
    this.s.add(`<path d="${d}" fill="none" stroke="${color}" stroke-width="${w}"${dash ? ` stroke-dasharray="${dash}"` : ""} stroke-linejoin="round"/>`);
  }
  steps(xs: ArrayLike<number>, ys: ArrayLike<number>, color: string, w = 3) {
    let d = "";
    for (let i = 0; i < xs.length; i++) {
      const xa = this.X(xs[i] - 0.5), xb = this.X(xs[i] + 0.5), y = this.Y(ys[i]);
      d += (i ? "L" : "M") + xa.toFixed(1) + "," + y.toFixed(1) + "L" + xb.toFixed(1) + "," + y.toFixed(1);
    }
    this.s.add(`<path d="${d}" fill="none" stroke="${color}" stroke-width="${w}" stroke-linejoin="round"/>`);
  }
  dot(x: number, y: number, color: string, r = 6) { this.s.circle(this.X(x), this.Y(y), r, color, "#fff", 1.5); }
  hline(y: number, color: string, w = 2, dash = "") { this.s.line(this.x0, this.Y(y), this.x0 + this.w, this.Y(y), color, w, dash); }
  vline(x: number, color: string, w = 2, dash = "") { this.s.line(this.X(x), this.y0, this.X(x), this.y0 + this.h, color, w, dash); }
  text(x: number, y: number, t: string, o: { size?: number; anchor?: string; color?: string; weight?: string } = {}) { this.s.text(this.X(x), this.Y(y), t, o); }
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
/** RGB PNG writer (3 bytes per pixel, row-major). */
export function pngRGB(rgb: Uint8Array, w: number, h: number): Uint8Array {
  const ihdr = new Uint8Array(13), dv = new DataView(ihdr.buffer);
  dv.setUint32(0, w); dv.setUint32(4, h); ihdr[8] = 8; ihdr[9] = 2;
  const raw = new Uint8Array((3 * w + 1) * h);
  for (let y = 0; y < h; y++) raw.set(rgb.subarray(3 * y * w, 3 * (y + 1) * w), y * (3 * w + 1) + 1);
  const idat = new Uint8Array(zlib.deflateSync(raw));
  const parts = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", idat), chunk("IEND", new Uint8Array(0))];
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

/** Map each pixel through f to an RGB triple. */
export function toRGB(n: number, f: (p: number) => [number, number, number]): Uint8Array {
  const out = new Uint8Array(3 * n);
  for (let p = 0; p < n; p++) { const c = f(p); out[3 * p] = c[0]; out[3 * p + 1] = c[1]; out[3 * p + 2] = c[2]; }
  return out;
}
export const SPIN_RGB = (v: number): [number, number, number] => (v > 0 ? [255, 127, 14] : [31, 119, 180]);

/** Draw a small grid of spins (+1 orange, -1 blue) as rounded squares; returns cell centre function. */
export function spinGrid(s: Svg, x0: number, y0: number, cols: number, rows: number, spins: ArrayLike<number>, cell = 44, gap = 4, labels = false) {
  const cx = (i: number) => x0 + i * (cell + gap) + cell / 2, cy = (j: number) => y0 + j * (cell + gap) + cell / 2;
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
    const v = spins[j * cols + i];
    s.rect(x0 + i * (cell + gap), y0 + j * (cell + gap), cell, cell, v > 0 ? C.plus : C.minus, { rx: 5 });
    if (labels) s.text(cx(i), cy(j) + 7, v > 0 ? "+1" : "−1", { anchor: "middle", size: 19, color: "#fff", weight: "bold" });
  }
  return { cx, cy };
}
