// Shared SVG helpers for the slide figures of the VAE day (autoencoder/, vae/, vae-gen/).
// Same style as demos/test/vae-figs.ts (the figures already on these decks).
declare const require: (m: string) => any;
const fs = require("fs");
import { pngGrey, tile } from "../../../demos/test/png";

export const INK = "#1b2a3a", BLUE = "#2a6f97", RED = "#bc4749", GREEN = "#6a994e", ORANGE = "#e07b1a", GREY = "#8d99ae", DARKOR = "#c4621a";
export const DIGIT_COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];

export function writer(root: string) {
  return (file: string, s: string | Uint8Array) => { fs.writeFileSync(`${root}/${file}`, s); console.log("wrote", file); };
}
export const svgOpen = (w: number, h: number, fs = 19) =>
  `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" font-family="Helvetica, Arial, sans-serif" font-size="${fs}" fill="${INK}">\n<rect width="${w}" height="${h}" fill="#fff"/>\n` +
  `<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${INK}"/></marker>` +
  `<marker id="arR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${RED}"/></marker>` +
  `<marker id="arB" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${BLUE}"/></marker></defs>\n`;
export const sc = (d0: number, d1: number, r0: number, r1: number) => (v: number) => r0 + (v - d0) * (r1 - r0) / (d1 - d0);
export const path = (pts: [number, number][]) => pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");
export const f = (v: number) => v.toFixed(1);
/** 28×28 images (ink 0..1, dark on white) tiled into one PNG data URI. */
export function pngURI(imgs: ArrayLike<number>[], cols: number, gap = 2, up = 4) {
  const t = tile(imgs, cols, gap, up);
  let bin = ""; for (const b of pngGrey(t.grey, t.w, t.h)) bin += String.fromCharCode(b);
  return { uri: "data:image/png;base64," + btoa(bin), w: t.w, h: t.h };
}
/** An image of any size (values 0..1 = ink, dark on white), upscaled, as a data URI. */
export function pngAny(v: ArrayLike<number>, w: number, h: number, up: number, invert = true) {
  const W = w * up, H = h * up, g = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const a = Math.max(0, Math.min(1, v[Math.floor(y / up) * w + Math.floor(x / up)]));
    g[y * W + x] = Math.round(255 * (invert ? 1 - a : a));
  }
  let bin = ""; for (const b of pngGrey(g, W, H)) bin += String.fromCharCode(b);
  return "data:image/png;base64," + btoa(bin);
}
export const img = (x: number, y: number, w: number, h: number, uri: string, frame = true) =>
  `<image x="${x}" y="${y}" width="${w}" height="${h}" href="${uri}" style="image-rendering:pixelated"/>` +
  (frame ? `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="none" stroke="${INK}" stroke-width="2"/>` : "") + "\n";
export const npdf = (z: number, m: number, s: number) => Math.exp(-0.5 * ((z - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
