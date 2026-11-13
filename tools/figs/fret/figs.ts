// Figures for the FRET day (SVG). Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/fret/figs.ts --bundle --platform=node --log-level=warning | node
// Writes into fret-physics/, sensitised-emission/, acceptor-bleach/, ratiometric/.
import * as fs from "fs";
import * as F from "../../../demos/src/lib/fret";
import { emission } from "../../../demos/src/lib/unmix";

const C = { blue: "#1f77b4", red: "#d62728", green: "#2ca02c", grey: "#777", purple: "#7b3294", fg: "#222", donor: "#1b9fc0", acc: "#d39e00" };
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
const svg = (w: number, h: number, body: string) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" font-family="Helvetica, Arial, sans-serif" font-size="20">\n<rect width="${w}" height="${h}" fill="#fff"/>\n${body}\n</svg>\n`;
const text = (x: number, y: number, s: string, o: { size?: number; anchor?: string; fill?: string; weight?: string } = {}) =>
  `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" font-size="${o.size ?? 20}" text-anchor="${o.anchor ?? "middle"}" fill="${o.fill ?? C.fg}"${o.weight ? ` font-weight="${o.weight}"` : ""}>${esc(s)}</text>`;
const line = (x1: number, y1: number, x2: number, y2: number, st = C.fg, w = 2, dash = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${st}" stroke-width="${w}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
const path = (pts: [number, number][], o: { stroke?: string; w?: number; fill?: string; op?: number; close?: boolean; dash?: string } = {}) => `<path d="M${pts.map(p => p[0].toFixed(1) + "," + p[1].toFixed(1)).join("L")}${o.close ? "Z" : ""}" fill="${o.fill ?? "none"}" fill-opacity="${o.op ?? 1}" stroke="${o.stroke ?? C.fg}" stroke-width="${o.w ?? 2.5}" stroke-linejoin="round"${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}/>`;
const rect = (x: number, y: number, w: number, h: number, fill: string, op = 1, stroke = "none") => `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" fill="${fill}" fill-opacity="${op}" stroke="${stroke}" stroke-width="2"/>`;
const circle = (x: number, y: number, r: number, fill: string, stroke = "none") => `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="2"/>`;
function axes(x0: number, y0: number, w: number, h: number, xd: [number, number], yd: [number, number], xt: number[], yt: number[], xl: string, yl: string, fmt = (v: number) => String(v)) {
  const X = (v: number) => x0 + (v - xd[0]) / (xd[1] - xd[0]) * w, Y = (v: number) => y0 + h - (v - yd[0]) / (yd[1] - yd[0]) * h;
  let s = "";
  for (const t of xt) s += line(X(t), y0, X(t), y0 + h, "#e3e3e3", 1) + line(X(t), y0 + h, X(t), y0 + h + 6, C.fg, 1.5) + text(X(t), y0 + h + 25, fmt(t), { size: 18 });
  for (const t of yt) s += line(x0, Y(t), x0 + w, Y(t), "#e3e3e3", 1) + line(x0 - 6, Y(t), x0, Y(t), C.fg, 1.5) + text(x0 - 10, Y(t) + 6, fmt(t), { size: 18, anchor: "end" });
  s += path([[x0, y0], [x0, y0 + h], [x0 + w, y0 + h]], { w: 1.5 });
  s += text(x0 + w / 2, y0 + h + 52, xl);
  s += `<text x="${x0 - 55}" y="${y0 + h / 2}" font-size="20" text-anchor="middle" transform="rotate(-90 ${x0 - 55} ${y0 + h / 2})">${esc(yl)}</text>`;
  return { X, Y, s };
}
const out = (p: string, s: string) => { fs.writeFileSync(p, s); console.log("wrote", p); };

// 1. efficiency curve with the window
{
  const R0 = 5, a = axes(90, 30, 620, 300, [0, 12], [0, 1], [0, 2, 4, 6, 8, 10, 12], [0, 0.25, 0.5, 0.75, 1], "distance r (nm), R₀ = 5 nm", "efficiency E");
  let s = a.s + rect(a.X(2.5), 30, a.X(7.5) - a.X(2.5), 300, C.purple, 0.12);
  s += text(a.X(6.3), a.Y(0.9), "sensitive window 0.5–1.5 R₀", { fill: C.purple, size: 18 });
  const pts: [number, number][] = [];
  for (let r = 0.05; r <= 12; r += 0.05) pts.push([a.X(r), a.Y(F.efficiency(r, R0))]);
  s += path(pts, { stroke: C.blue, w: 4 });
  for (const k of [0.5, 1, 1.5, 2]) {
    const r = k * R0, E = F.efficiency(r, R0);
    s += circle(a.X(r), a.Y(E), 7, C.red) + text(a.X(r) + 10, a.Y(E) - 12, `${k} R₀: ${E.toFixed(3)}`, { anchor: "start", size: 18, fill: C.red });
    console.log(`E at ${k} R0 = ${E}`);
  }
  // a fluorescent protein to scale above the axis (≈ 2.4 × 4.2 nm)
  s += rect(a.X(0.3), a.Y(0.32), a.X(2.7) - a.X(0.3), a.Y(0) - a.Y(0.14), "#3fc1d9", 0.6, "#333") + text(a.X(1.5), a.Y(0.36), "one protein, to scale", { size: 16, anchor: "start" });
  out("fret-physics/fig-efficiency.svg", svg(760, 400, s));
}
// 2. spectra and overlap
{
  const D = F.CYAN_DONOR, A = F.YELLOW_ACCEPTOR;
  const a = axes(90, 30, 620, 260, [380, 620], [0, 1.05], [400, 450, 500, 550, 600], [0, 0.5, 1], "wavelength λ (nm)", "relative");
  let s = a.s;
  const L: number[] = []; for (let l = 380; l <= 620; l += 1) L.push(l);
  const ov: [number, number][] = L.map(l => [a.X(l), a.Y(Math.min(emission(D.em, l), F.absorption(A.ex, l)))]);
  s += path([[a.X(380), a.Y(0)], ...ov, [a.X(620), a.Y(0)]], { fill: C.purple, op: 0.35, stroke: "none", close: true });
  const curve = (f: (l: number) => number, col: string, dash = "") => path(L.map(l => [a.X(l), a.Y(f(l))] as [number, number]), { stroke: col, w: 3.5, dash });
  s += curve(l => F.absorption(D.ex, l), C.donor, "8 6") + curve(l => emission(D.em, l), C.donor) + curve(l => F.absorption(A.ex, l), C.acc, "8 6") + curve(l => emission(A.em, l), C.acc);
  s += text(a.X(430), a.Y(1.0) - 8, "donor ex", { fill: C.donor, size: 18 }) + text(a.X(470), a.Y(1.0) - 8, "donor em", { fill: C.donor, size: 18, anchor: "end" });
  s += text(a.X(520), a.Y(1.0) - 8, "acceptor ex", { fill: C.acc, size: 18, anchor: "end" }) + text(a.X(535), a.Y(1.0) - 8, "acceptor em", { fill: C.acc, size: 18, anchor: "start" });
  s += text(a.X(500), a.Y(0.15), "overlap", { fill: C.purple, size: 18 });
  const J = F.overlapIntegral(D.em, A.ex, A.eps);
  console.log("J", J, "R0", F.forsterRadius(D.Q, 2 / 3, 1.4, J));
  out("fret-physics/fig-spectra.svg", svg(760, 360, s));
}
// 3. three images: excitation × emission grid
{
  let s = "";
  const x0 = 250, y0 = 70, cw = 300, ch = 120;
  s += text(x0 + cw / 2, 50, "donor filter", { weight: "bold" }) + text(x0 + 1.5 * cw, 50, "acceptor filter", { weight: "bold" });
  s += text(x0 - 20, y0 + ch / 2 + 6, "donor laser", { anchor: "end", weight: "bold" }) + text(x0 - 20, y0 + 1.5 * ch + 6, "acceptor laser", { anchor: "end", weight: "bold" });
  const cell = (i: number, j: number, fill: string, l1: string, l2: string, l3 = "") => {
    let t = rect(x0 + j * cw + 4, y0 + i * ch + 4, cw - 8, ch - 8, fill, 0.25, "#555");
    t += text(x0 + (j + 0.5) * cw, y0 + i * ch + 38, l1, { weight: "bold", size: 22 }) + text(x0 + (j + 0.5) * cw, y0 + i * ch + 68, l2, { size: 18 });
    if (l3) t += text(x0 + (j + 0.5) * cw, y0 + i * ch + 94, l3, { size: 18 });
    return t;
  };
  s += cell(0, 0, C.donor, "IDD", "donor emission (quenched)");
  s += cell(0, 1, C.purple, "IDA: the FRET channel", "d·IDD (bleed-through)", "+ a·IAA (cross-exc.) + Fc");
  s += cell(1, 0, "#ccc", "(not used)", "acceptor does not emit here");
  s += cell(1, 1, C.acc, "IAA", "acceptor, excited directly");
  out("sensitised-emission/fig-threecube.svg", svg(880, 330, s));
}
// 4. bleach time course
{
  const ser = F.bleachSeries({ E: 0.25, f: 1, pulses: 12, perPulse: 0.3, donorLoss: 0 });
  const a = axes(90, 30, 560, 270, [0, 12], [0, 1.5], [0, 2, 4, 6, 8, 10, 12], [0, 0.5, 1, 1.5], "bleach pulse", "relative to before");
  let s = a.s;
  const rel = (v: number[]) => v.map(x => x / v[0]);
  const curve = (v: number[], col: string, dash = "") => path(v.map((y, k) => [a.X(k), a.Y(y)] as [number, number]), { stroke: col, w: 4, dash });
  s += curve(rel(ser.donor), C.donor) + curve(ser.acc, C.acc) + curve(rel(ser.control), C.grey, "8 6");
  s += text(a.X(12) - 4, a.Y(rel(ser.donor)[12]) - 12, `donor: × ${(ser.donor[12] / ser.donor[0]).toFixed(3)}`, { anchor: "end", fill: C.donor, size: 18 });
  s += text(a.X(1.5), a.Y(0.25), "acceptor in the region", { anchor: "start", fill: C.acc, size: 18 });
  s += text(a.X(12) - 4, a.Y(1) + 26, "donor outside the region", { anchor: "end", fill: C.grey, size: 18 });
  console.log("bleach: donor after/before", ser.donor[12] / ser.donor[0], "acc left", ser.acc[12], "E_obs", 1 - ser.donor[0] / ser.donor[12]);
  out("acceptor-bleach/fig-bleach-course.svg", svg(700, 370, s));
}
// 5. calibration curve
{
  const a = axes(100, 30, 560, 280, [0.5, 2.5], [0, 4000], [0.5, 1, 1.5, 2, 2.5], [0, 1000, 2000, 3000, 4000], "ratio R", "concentration c (nM)");
  let s = a.s;
  const pts: [number, number][] = [];
  for (let R = 0.5; R < 2.45; R += 0.005) pts.push([a.X(R), a.Y(Math.min(4000, F.calibrate(R, 0.5, 2.5, 200)))]);
  s += path(pts, { stroke: C.blue, w: 4 });
  for (const R of [1.5, 2.3]) {
    const c = F.calibrate(R, 0.5, 2.5, 200), lo = F.calibrate(R - 0.05, 0.5, 2.5, 200), hi = F.calibrate(R + 0.05, 0.5, 2.5, 200);
    s += rect(a.X(R - 0.05), a.Y(Math.min(4000, hi)), a.X(R + 0.05) - a.X(R - 0.05), a.Y(lo) - a.Y(Math.min(4000, hi)), C.red, 0.2);
    s += circle(a.X(R), a.Y(c), 7, C.red) + text(a.X(R) - 12, a.Y(c) - 10, `R = ${R}: ${c.toFixed(0)} nM (${lo.toFixed(0)}–${hi.toFixed(0)})`, { anchor: "end", size: 18, fill: C.red });
    console.log("calib", R, c, lo, hi);
  }
  s += line(a.X(2.5), 30, a.X(2.5), 310, C.grey, 2, "6 5") + text(a.X(2.5) - 6, 50, "R_max", { anchor: "end", size: 18, fill: C.grey });
  out("ratiometric/fig-calibration.svg", svg(720, 390, s));
}
// 6. histograms of a ratio: bright and dim
{
  const draw = (muA: number, muD: number, x0: number, title: string) => {
    const mc = F.ratioDraws({ muA, muD, n: 100000, seed: 7 });
    const t = muA / muD, lo = 0, hi = 4 * t, bins = 60, w = (hi - lo) / bins, cnt = new Array(bins).fill(0);
    let s = 0, n = 0;
    for (const r of mc.ratios) { if (!Number.isFinite(r)) continue; s += r; n++; const b = Math.floor((r - lo) / w); if (b >= 0 && b < bins) cnt[b]++; }
    const mx = Math.max(...cnt);
    const a = axes(x0, 50, 300, 230, [lo, hi], [0, 1.05], [0, t, 2 * t, 3 * t, 4 * t], [], "A/D", "", v => v.toFixed(2));
    let o = a.s + text(x0 + 150, 30, title, { size: 18, weight: "bold" });
    cnt.forEach((c, i) => { o += rect(a.X(lo + i * w), a.Y(c / mx), a.X(lo + (i + 1) * w) - a.X(lo + i * w), a.Y(0) - a.Y(c / mx), C.blue, 0.5); });
    o += line(a.X(t), 50, a.X(t), 280, C.green, 4) + line(a.X(s / n), 50, a.X(s / n), 280, C.red, 3, "7 5");
    console.log("hist", muA, muD, "mean", s / n, "bias %", 100 * (s / n / t - 1), "CV pred", F.ratioCVPredicted(muA, muD));
    return o;
  };
  out("ratiometric/fig-ratio-hist.svg", svg(820, 360, draw(400, 1000, 70, "A ≈ 400, D ≈ 1000 photons") + draw(20, 5, 480, "A ≈ 20, D ≈ 5 photons") + text(410, 350, "green: true ratio · red dashes: mean of the pixel ratios", { size: 17, fill: C.grey })));
}
