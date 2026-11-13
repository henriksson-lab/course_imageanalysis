// Shared SVG helpers for the evaluation-day figures (same style as figs.ts).
export const ROOT = "/data/henriksson/github/course/imageanalysis/";
export const C = { tp: "#2ca02c", fp: "#d62728", fn: "#ff7f0e", blue: "#1f77b4", grey: "#777", purple: "#9467bd", fg: "#222", light: "#e3e3e3" };
export const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
export function svg(w: number, h: number, body: string): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" font-family="Helvetica, Arial, sans-serif" font-size="20">\n<rect width="${w}" height="${h}" fill="#fff"/>\n${body}\n</svg>\n`;
}
export const text = (x: number, y: number, s: string, o: { size?: number; anchor?: string; fill?: string; weight?: string; italic?: boolean } = {}) =>
  `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" font-size="${o.size ?? 20}" text-anchor="${o.anchor ?? "middle"}" fill="${o.fill ?? C.fg}"${o.weight ? ` font-weight="${o.weight}"` : ""}${o.italic ? ` font-style="italic"` : ""}>${esc(s)}</text>`;
export const line = (x1: number, y1: number, x2: number, y2: number, st = C.fg, w = 2, dash = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${st}" stroke-width="${w}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
type Sty = { fill?: string; stroke?: string; w?: number; op?: number; dash?: string };
export const circle = (x: number, y: number, r: number, o: Sty = {}) => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="${o.fill ?? "none"}" fill-opacity="${o.op ?? 1}" stroke="${o.stroke ?? "none"}" stroke-width="${o.w ?? 2}"${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}/>`;
export const ellipse = (x: number, y: number, a: number, b: number, o: Sty = {}) => `<ellipse cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" rx="${a.toFixed(1)}" ry="${b.toFixed(1)}" fill="${o.fill ?? "none"}" fill-opacity="${o.op ?? 1}" stroke="${o.stroke ?? "none"}" stroke-width="${o.w ?? 2}"${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}/>`;
export const rect = (x: number, y: number, w: number, h: number, o: Sty & { rx?: number } = {}) => `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}"${o.rx ? ` rx="${o.rx}"` : ""} fill="${o.fill ?? "none"}" fill-opacity="${o.op ?? 1}" stroke="${o.stroke ?? "none"}" stroke-width="${o.w ?? 2}"${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}/>`;
export const path = (pts: [number, number][], o: { stroke?: string; w?: number; fill?: string; op?: number; close?: boolean; dash?: string } = {}) => `<path d="M${pts.map(p => p[0].toFixed(1) + "," + p[1].toFixed(1)).join("L")}${o.close ? "Z" : ""}" fill="${o.fill ?? "none"}" fill-opacity="${o.op ?? 1}" stroke="${o.stroke ?? C.fg}" stroke-width="${o.w ?? 2.5}" stroke-linejoin="round"${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}/>`;
export const arrow = (x1: number, y1: number, x2: number, y2: number, st = C.fg, w = 2.5, head = 11) => {
  const a = Math.atan2(y2 - y1, x2 - x1), l = head;
  const p1: [number, number] = [x2 - l * Math.cos(a - 0.4), y2 - l * Math.sin(a - 0.4)], p2: [number, number] = [x2 - l * Math.cos(a + 0.4), y2 - l * Math.sin(a + 0.4)];
  return line(x1, y1, x2 - 0.7 * l * Math.cos(a), y2 - 0.7 * l * Math.sin(a), st, w) + path([p1, [x2, y2], p2], { fill: st, stroke: st, w: 1, close: true });
};
/** Axes box: returns map functions and the SVG of axes with ticks. */
export function axes(x0: number, y0: number, w: number, h: number, xd: [number, number], yd: [number, number], xt: number[], yt: number[], xl: string, yl: string, fmt = (v: number) => String(v), o: { logx?: boolean } = {}) {
  const tx = (v: number) => (o.logx ? Math.log10(v) : v);
  const X = (v: number) => x0 + (tx(v) - tx(xd[0])) / (tx(xd[1]) - tx(xd[0])) * w, Y = (v: number) => y0 + h - (v - yd[0]) / (yd[1] - yd[0]) * h;
  let s = "";
  for (const t of xt) { s += line(X(t), y0, X(t), y0 + h, C.light, 1) + line(X(t), y0 + h, X(t), y0 + h + 6, C.fg, 1.5) + text(X(t), y0 + h + 25, fmt(t), { size: 18 }); }
  for (const t of yt) { s += line(x0, Y(t), x0 + w, Y(t), C.light, 1) + line(x0 - 6, Y(t), x0, Y(t), C.fg, 1.5) + text(x0 - 10, Y(t) + 6, fmt(t), { size: 18, anchor: "end" }); }
  s += path([[x0, y0], [x0, y0 + h], [x0 + w, y0 + h]], { w: 1.5 });
  if (xl) s += text(x0 + w / 2, y0 + h + 52, xl);
  if (yl) s += `<text x="${x0 - 55}" y="${y0 + h / 2}" font-size="20" text-anchor="middle" transform="rotate(-90 ${x0 - 55} ${y0 + h / 2})">${esc(yl)}</text>`;
  return { X, Y, s };
}
/** Pixel counts of shapes and their pairwise intersections on a fine grid. */
export function areas(fs: ((x: number, y: number) => boolean)[], box: [number, number, number, number], step = 0.25) {
  const n = fs.length, inter = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let y = box[1]; y < box[3]; y += step) for (let x = box[0]; x < box[2]; x += step) {
    const v = fs.map(f => f(x, y));
    for (let i = 0; i < n; i++) if (v[i]) for (let j = 0; j < n; j++) if (v[j]) inter[i][j]++;
  }
  return inter;
}
export const iouOf = (A: number[][], i: number, j: number) => A[i][j] / (A[i][i] + A[j][j] - A[i][j]);
export const inDisc = (cx: number, cy: number, r: number) => (x: number, y: number) => (x - cx) ** 2 + (y - cy) ** 2 <= r * r;
