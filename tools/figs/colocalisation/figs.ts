// Static figures for the colocalisation day (print fallbacks of the demos). Run from the repo
// root:
//   node_modules/.bin/esbuild tools/figs/colocalisation/figs.ts --bundle --platform=node --log-level=warning | node
// Writes coloc-null/fig-shiftcurve.svg, coloc-null/fig-nulls.svg,
// coloc-objects/fig-chance.svg, coloc-confounds/fig-overlap.svg.
import { writeFileSync } from "fs";
import { pixelScene } from "../../../demos/src/lib/twochannel";
import { shiftCurve, randomisationTest, chanceFraction, gaussianOverlap } from "../../../demos/src/lib/coloc";

type Series = { x: number[]; y: number[]; color: string; dash?: string; label: string; bars?: boolean; fill?: string };

function svgPlot(o: { w?: number; h?: number; x: [number, number]; y: [number, number]; xlabel: string; ylabel: string; series: Series[]; vlines?: { x: number; color: string; label: string }[]; xticks: number[]; yticks: number[]; legendLeft?: boolean; band?: { x0: number; x1: number; color: string; label: string } }): string {
  const W = o.w ?? 640, H = o.h ?? 360, L = 72, R = 16, T = 16, B = 58;
  const sx = (v: number) => L + (v - o.x[0]) / (o.x[1] - o.x[0]) * (W - L - R);
  const sy = (v: number) => H - B - (v - o.y[0]) / (o.y[1] - o.y[0]) * (H - T - B);
  const f = (v: number) => v.toFixed(1);
  let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="system-ui, sans-serif" font-size="18">\n<rect width="${W}" height="${H}" fill="white"/>\n`;
  if (o.band) s += `<rect x="${f(sx(o.band.x0))}" y="${T}" width="${f(sx(o.band.x1) - sx(o.band.x0))}" height="${H - T - B}" fill="${o.band.color}" opacity="0.18"/>\n`;
  for (const t of o.xticks) s += `<line x1="${f(sx(t))}" x2="${f(sx(t))}" y1="${T}" y2="${H - B}" stroke="#e3e3e3"/><text x="${f(sx(t))}" y="${H - B + 22}" text-anchor="middle">${t}</text>\n`;
  for (const t of o.yticks) s += `<line x1="${L}" x2="${W - R}" y1="${f(sy(t))}" y2="${f(sy(t))}" stroke="#e3e3e3"/><text x="${L - 8}" y="${f(sy(t) + 6)}" text-anchor="end">${t}</text>\n`;
  s += `<line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="#444"/><line x1="${L}" x2="${L}" y1="${T}" y2="${H - B}" stroke="#444"/>\n`;
  s += `<text x="${(L + W - R) / 2}" y="${H - 10}" text-anchor="middle">${o.xlabel}</text><text transform="translate(18 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${o.ylabel}</text>\n`;
  for (const se of o.series) {
    if (se.bars) {
      const bw = se.x.length > 1 ? se.x[1] - se.x[0] : 1;
      se.x.forEach((x, i) => { if (se.y[i] > 0) s += `<rect x="${f(sx(x))}" y="${f(sy(se.y[i]))}" width="${f(sx(x + bw) - sx(x))}" height="${f(sy(0) - sy(se.y[i]))}" fill="${se.fill ?? se.color}" opacity="0.55"/>`; });
      s += "\n";
    } else s += `<polyline fill="none" stroke="${se.color}" stroke-width="3" ${se.dash ? `stroke-dasharray="${se.dash}"` : ""} points="${se.x.map((x, i) => `${f(sx(x))},${f(sy(se.y[i]))}`).join(" ")}"/>\n`;
  }
  for (const v of o.vlines ?? []) s += `<line x1="${f(sx(v.x))}" x2="${f(sx(v.x))}" y1="${T}" y2="${H - B}" stroke="${v.color}" stroke-width="3"/><text x="${f(sx(v.x) + 6)}" y="${T + 20}" fill="${v.color}">${v.label}</text>\n`;
  // legend
  const items = [...o.series.map(se => ({ label: se.label, color: se.fill ?? se.color, dash: se.dash, box: !!se.bars })), ...(o.band ? [{ label: o.band.label, color: o.band.color, dash: undefined, box: true }] : [])];
  items.forEach((it, k) => {
    const y = T + 14 + 24 * k, x = o.legendLeft ? L + 12 : W - R - 250;
    s += it.box ? `<rect x="${x}" y="${y - 10}" width="26" height="14" fill="${it.color}" opacity="0.55"/>` : `<line x1="${x}" x2="${x + 26}" y1="${y - 3}" y2="${y - 3}" stroke="${it.color}" stroke-width="3" ${it.dash ? `stroke-dasharray="${it.dash}"` : ""}/>`;
    s += `<text x="${x + 34}" y="${y + 3}">${it.label}</text>\n`;
  });
  return s + "</svg>\n";
}

// 1. shift curves: small colocalised structures vs large-scale correlation
{
  const a = pixelScene({ f: 1 }), b = pixelScene({ f: 0, diffuse: 30 });
  const ca = shiftCurve(a.a, a.b, 128, 128, 20, a.cells), cb = shiftCurve(b.a, b.b, 128, 128, 20);
  writeFileSync("coloc-null/fig-shiftcurve.svg", svgPlot({
    x: [-20, 20], y: [-0.1, 1], xlabel: "shift of B (px)", ylabel: "Pearson r", xticks: [-20, -10, 0, 10, 20], yticks: [0, 0.5, 1],
    band: { x0: -1.18, x1: 1.18, color: "#ff7f0e", label: "blur FWHM" },
    series: [{ x: ca.shifts, y: ca.r, color: "#1f77b4", label: "colocalised spots" }, { x: cb.shifts, y: cb.r, color: "#d62728", dash: "8 5", label: "cell outline only" }],
  }));
}
// 2. null distributions: 1-px shuffle vs 8-px blocks (independent channels, blur 2, seed 15)
{
  const s = pixelScene({ f: 0, blur: 2, seed: 15 });
  const t1 = randomisationTest(s.a, s.b, 128, 128, { block: 1, n: 200, seed: 78, mask: s.cells });
  const t8 = randomisationTest(s.a, s.b, 128, 128, { block: 8, n: 200, seed: 77, mask: s.cells });
  const lo = -0.25, hi = 0.25, nb = 50, bw = (hi - lo) / nb;
  const h = (v: Float64Array) => { const c = new Array(nb).fill(0); v.forEach(x => { const k = Math.floor((x - lo) / bw); if (k >= 0 && k < nb) c[k]++; }); return c; };
  const xs = Array.from({ length: nb }, (_, k) => lo + k * bw);
  writeFileSync("coloc-null/fig-nulls.svg", svgPlot({
    x: [lo, hi], y: [0, 70], xlabel: "Pearson r with B shuffled (200 times)", ylabel: "count", xticks: [-0.2, -0.1, 0, 0.1, 0.2], yticks: [0, 20, 40, 60],
    series: [{ x: xs, y: h(t1.null), color: "#888", fill: "#888", bars: true, label: "1-px shuffle" }, { x: xs, y: h(t8.null), color: "#1f77b4", fill: "#1f77b4", bars: true, label: "8-px blocks" }],
    vlines: [{ x: t8.observed, color: "#d62728", label: "observed" }], legendLeft: true,
  }));
}
// 3. chance level of the object fraction
{
  const d = Array.from({ length: 101 }, (_, k) => k * 0.06);
  writeFileSync("coloc-objects/fig-chance.svg", svgPlot({
    x: [0, 6], y: [0, 1], xlabel: "distance d (px)", ylabel: "chance of a B spot within d", xticks: [0, 1, 2, 3, 4, 5, 6], yticks: [0, 0.25, 0.5, 0.75, 1],
    series: [{ x: d, y: d.map(v => chanceFraction(0.02, v)), color: "#d62728", label: "dense: ρ = 0.02 px⁻²" }, { x: d, y: d.map(v => chanceFraction(0.002, v)), color: "#1f77b4", label: "sparse: ρ = 0.002 px⁻²" }],
    vlines: [{ x: 2, color: "#444", label: "d = 2" }],
  }));
}
// 4. normalised overlap of two Gaussian spots against their distance (NA 1.4)
{
  const d = Array.from({ length: 101 }, (_, k) => k * 5);
  const sA = 0.21 * 520 / 1.4, sB = 0.21 * 670 / 1.4, fwhm = 2.3548 * sA;
  writeFileSync("coloc-confounds/fig-overlap.svg", svgPlot({
    x: [0, 500], y: [0, 1], xlabel: "distance between the two spots (nm)", ylabel: "normalised overlap", xticks: [0, 100, 200, 300, 400, 500], yticks: [0, 0.25, 0.5, 0.75, 1],
    band: { x0: 0, x1: 0.7 * fwhm, color: "#ff7f0e", label: "< 0.7 FWHM" },
    series: [{ x: d, y: d.map(v => gaussianOverlap(v, sA, sA)), color: "#1f77b4", label: "both 520 nm" }, { x: d, y: d.map(v => gaussianOverlap(v, sA, sB)), color: "#d62728", dash: "8 5", label: "520 and 670 nm" }],
    vlines: [{ x: fwhm, color: "#444", label: "FWHM" }],
  }));
}
console.log("figures written");
