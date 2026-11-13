// Extra slide figures for local-threshold/lec.qmd (the first set is in tools/figs/thresholding/figs.ts).
// Run: node_modules/.bin/esbuild tools/figs/local-threshold/figs.ts --bundle --platform=node --outfile=/tmp/x.js && node /tmp/x.js
import * as LM from "/data/henriksson/github/course/imageanalysis/demos/src/localthresh/model";
import { confusion, dice } from "/data/henriksson/github/course/imageanalysis/demos/src/lib/metrics";
import { Svg, Panel, C, u8, writePng } from "../thresholding/plot";

const truth = LM.discTruth();
const BASE: LM.Settings = { source: "discs", method: "global", r: 25, k: 0.3, C: 10, bg: 30, light: 1.0, vignette: 0.3, noise: 6 };
const img = LM.makeImage(BASE, truth, null);
const bw = (m: Uint8Array) => Uint8Array.from(m, v => (v ? 0 : 255));
const report = (name: string, s: LM.Settings) => { const g = LM.segment(img, s), c = confusion(g.mask, truth); console.log(name, g.info, "Dice", dice(c).toFixed(3), "FP", c.fp, "FN", c.fn); return g; };

{ // remedy 1: image, smooth background, ratio, mask
  const B = LM.smoothBackground(img, BASE.bg);
  const ratio = Float64Array.from(img.data, (v, i) => 128 * v / Math.max(1, B.data[i]));
  const g = report("divide", { ...BASE, method: "divide" });
  writePng("local-threshold/fig-divide.png", [u8(img), u8(B), u8(ratio), bw(g.mask)], 2, 8);
}
{ // Niblack vs Sauvola, both 51 × 51
  const n = report("niblack", { ...BASE, method: "niblack", r: 25, k: -0.2 }), s = report("sauvola", { ...BASE, method: "sauvola", r: 25, k: 0.3 });
  writePng("local-threshold/fig-niblack.png", [bw(n.mask), bw(s.mask)], 2, 8);
}
{ // window size: mean − C with 11 × 11, 51 × 51, 241 × 241
  const ms = [5, 25, 120].map(r => report(`meanC r=${r}`, { ...BASE, method: "meanC", r, C: 10 }).mask);
  writePng("local-threshold/fig-window.png", ms.map(bw), 3, 8);
}
{ // the by-hand profile: mean − C, window 3, C = 10, replicated borders; global t = 125
  const I = [100, 110, 80, 130, 140, 150, 160, 130, 180, 190], n = I.length;
  const m = I.map((_, i) => (I[Math.max(0, i - 1)] + I[i] + I[Math.min(n - 1, i + 1)]) / 3), T = m.map(v => v - 10);
  console.log("profile T", T.map(v => v.toFixed(1)).join(" "), "objects", I.map((v, i) => (v < T[i] ? i + 1 : 0)).filter(Boolean));
  const S = new Svg(480, 360), p = new Panel(S, 80, 34, 385, 240, [0.4, 10.6], [60, 200]);
  p.axes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [60, 100, 140, 180], "position", "grey level");
  const pos = I.map((_, i) => i + 1);
  p.hline(125, C.c2, "", 2.5, "8 5");
  p.line(pos, T, C.c4, 3.5);
  I.forEach((v, i) => p.dot(i + 1, v, v < T[i] ? C.fg : "#bbb", 9));
  I.forEach((v, i) => { if (v < 125) S.add(`<circle cx="${p.X(i + 1)}" cy="${p.Y(v)}" r="14" fill="none" stroke="${C.c2}" stroke-width="2.5"/>`); });
  S.text(p.X(10.5), p.Y(125) + 24, "global t = 125", C.c2, "end", 19);
  S.text(p.X(5.4), p.Y(T[4]) - 34, "T = mean − 10", C.c4, "end", 19);
  S.text(80, 22, "● object, mean − C", C.fg, "start", 19); S.text(290, 22, "○ object, global t", C.c2, "start", 19);
  S.out("local-threshold/fig-meanc.svg");
}
{ // running sum along a row: one pixel enters, one leaves
  const S = new Svg(640, 250), w = 52, x0 = 20, y0 = 70, vals = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
  vals.forEach((v, i) => {
    const inOld = i >= 2 && i <= 6, inNew = i >= 3 && i <= 7;
    const fill = i === 2 ? "#cfe0f2" : i === 7 ? "#ffd9b0" : inOld && inNew ? "#eee" : "#fff";
    S.add(`<rect x="${x0 + i * w}" y="${y0}" width="${w}" height="${w}" fill="${fill}" stroke="#444" stroke-width="1.5"/>`);
    S.text(x0 + i * w + w / 2, y0 + w / 2 + 7, String(v), C.fg, "middle", 20);
  });
  const br = (a: number, b: number, y: number, col: string, lab: string, up: boolean) => {
    const xa = x0 + a * w + 3, xb = x0 + (b + 1) * w - 3, d = up ? -10 : 10;
    S.add(`<path d="M${xa},${y + d}V${y}H${xb}V${y + d}" fill="none" stroke="${col}" stroke-width="3"/>`);
    S.text((xa + xb) / 2, up ? y - 22 : y + 30, lab, col, "middle", 19);
  };
  br(2, 6, y0 - 12, C.c1, "window at x: S(x) = 4+1+5+9+2 = 21", true);
  br(3, 7, y0 + w + 12, C.c4, "at x + 1: S = 21 − 4 + 6 = 23", false);
  S.text(x0 + 2 * w + w / 2, y0 + w + 82, "leaves", C.c1, "middle", 18);
  S.text(x0 + 7 * w + w / 2, y0 + w + 82, "enters", C.c4, "middle", 18);
  S.out("local-threshold/fig-runsum.svg");
}
{ // 4- and 8-neighbourhoods, and a diagonal line
  const S = new Svg(640, 260), c = 40;
  const grid = (ox: number, oy: number, on: (x: number, y: number) => string | null, title: string) => {
    for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) { const f = on(x, y) ?? "#fff"; S.add(`<rect x="${ox + x * c}" y="${oy + y * c}" width="${c}" height="${c}" fill="${f}" stroke="#666"/>`); }
    S.text(ox + 1.5 * c, oy + 3 * c + 30, title, C.fg, "middle", 19);
  };
  grid(20, 40, (x, y) => (x === 1 && y === 1 ? C.fg : (x === 1 || y === 1) ? C.c4 : null), "4 neighbours");
  grid(200, 40, (x, y) => (x === 1 && y === 1 ? C.fg : C.c4), "8 neighbours");
  // diagonal line on a 5 × 5 grid
  const ox = 400, oy = 20, cc = 36;
  for (let y = 0; y < 5; y++) for (let x = 0; x < 5; x++) S.add(`<rect x="${ox + x * cc}" y="${oy + y * cc}" width="${cc}" height="${cc}" fill="${x === y ? C.c1 : "#fff"}" stroke="#666"/>`);
  S.text(ox + 2.5 * cc, oy + 5 * cc + 26, "diagonal line: 5 pieces", C.fg, "middle", 18);
  S.text(ox + 2.5 * cc, oy + 5 * cc + 50, "with 4, one with 8", C.fg, "middle", 18);
  S.out("local-threshold/fig-conn.svg");
}
console.log("local-threshold figures written");
