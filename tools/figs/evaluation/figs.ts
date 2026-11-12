// Figures for the evaluation day (SVG). Run: run.sh figs.ts. Numbers are printed so the text can quote them.
import * as fs from "fs";
import * as OM from "/data/henriksson/github/course/imageanalysis/demos/src/objmatch/model";
import * as CD from "/data/henriksson/github/course/imageanalysis/demos/src/contourdist/model";
import * as FSm from "/data/henriksson/github/course/imageanalysis/demos/src/featurescan/model";
import * as EV from "/data/henriksson/github/course/imageanalysis/demos/src/lib/eval";
import * as ST from "/data/henriksson/github/course/imageanalysis/demos/src/lib/stats";
import { rng } from "/data/henriksson/github/course/imageanalysis/demos/src/lib/random";
const ROOT = "/data/henriksson/github/course/imageanalysis/";
const C = { tp: "#2ca02c", fp: "#d62728", fn: "#ff7f0e", blue: "#1f77b4", grey: "#777", purple: "#9467bd", fg: "#222" };
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
function svg(w: number, h: number, body: string): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" font-family="Helvetica, Arial, sans-serif" font-size="20">\n<rect width="${w}" height="${h}" fill="#fff"/>\n${body}\n</svg>\n`;
}
const text = (x: number, y: number, s: string, o: { size?: number; anchor?: string; fill?: string; weight?: string } = {}) =>
  `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" font-size="${o.size ?? 20}" text-anchor="${o.anchor ?? "middle"}" fill="${o.fill ?? C.fg}"${o.weight ? ` font-weight="${o.weight}"` : ""}>${esc(s)}</text>`;
const line = (x1: number, y1: number, x2: number, y2: number, st = C.fg, w = 2, dash = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${st}" stroke-width="${w}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
const circle = (x: number, y: number, r: number, o: { fill?: string; stroke?: string; w?: number; op?: number; dash?: string } = {}) => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="${o.fill ?? "none"}" fill-opacity="${o.op ?? 1}" stroke="${o.stroke ?? "none"}" stroke-width="${o.w ?? 2}"${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}/>`;
const ellipse = (x: number, y: number, a: number, b: number, o: { fill?: string; stroke?: string; w?: number; op?: number; dash?: string } = {}) => `<ellipse cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" rx="${a.toFixed(1)}" ry="${b.toFixed(1)}" fill="${o.fill ?? "none"}" fill-opacity="${o.op ?? 1}" stroke="${o.stroke ?? "none"}" stroke-width="${o.w ?? 2}"${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}/>`;
const path = (pts: [number, number][], o: { stroke?: string; w?: number; fill?: string; op?: number; close?: boolean; dash?: string } = {}) => `<path d="M${pts.map(p => p[0].toFixed(1) + "," + p[1].toFixed(1)).join("L")}${o.close ? "Z" : ""}" fill="${o.fill ?? "none"}" fill-opacity="${o.op ?? 1}" stroke="${o.stroke ?? C.fg}" stroke-width="${o.w ?? 2.5}" stroke-linejoin="round"${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}/>`;

/** Axes box: returns map functions and the SVG of axes with ticks. */
function axes(x0: number, y0: number, w: number, h: number, xd: [number, number], yd: [number, number], xt: number[], yt: number[], xl: string, yl: string, fmt = (v: number) => String(v)) {
  const X = (v: number) => x0 + (v - xd[0]) / (xd[1] - xd[0]) * w, Y = (v: number) => y0 + h - (v - yd[0]) / (yd[1] - yd[0]) * h;
  let s = "";
  for (const t of xt) { s += line(X(t), y0, X(t), y0 + h, "#e3e3e3", 1) + line(X(t), y0 + h, X(t), y0 + h + 6, C.fg, 1.5) + text(X(t), y0 + h + 25, fmt(t), { size: 18 }); }
  for (const t of yt) { s += line(x0, Y(t), x0 + w, Y(t), "#e3e3e3", 1) + line(x0 - 6, Y(t), x0, Y(t), C.fg, 1.5) + text(x0 - 10, Y(t) + 6, fmt(t), { size: 18, anchor: "end" }); }
  s += path([[x0, y0], [x0, y0 + h], [x0 + w, y0 + h]], { w: 1.5 });
  s += text(x0 + w / 2, y0 + h + 52, xl);
  s += `<text x="${x0 - 52}" y="${y0 + h / 2}" font-size="20" text-anchor="middle" transform="rotate(-90 ${x0 - 52} ${y0 + h / 2})">${esc(yl)}</text>`;
  return { X, Y, s };
}

// grid IoU / Dice of shapes given by predicates
function areas(fs: ((x: number, y: number) => boolean)[], box: [number, number, number, number], step = 0.25) {
  const n = fs.length, inter = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let y = box[1]; y < box[3]; y += step) for (let x = box[0]; x < box[2]; x += step) {
    const v = fs.map(f => f(x, y));
    for (let i = 0; i < n; i++) if (v[i]) for (let j = 0; j < n; j++) if (v[j]) inter[i][j]++;
  }
  return inter;
}
const inDisc = (cx: number, cy: number, r: number) => (x: number, y: number) => (x - cx) ** 2 + (y - cy) ** 2 <= r * r;
const inEll = (cx: number, cy: number, a: number, b: number) => (x: number, y: number) => ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1;

// ---------------------------------------------------------------- object-eval/fig-iou.svg
{
  const discIoU = (r: number, d: number) => { const A = 2 * r * r * Math.acos(d / (2 * r)) - (d / 2) * Math.sqrt(4 * r * r - d * d); return A / (2 * Math.PI * r * r - A); };
  const R = 60; let body = "";
  [0.9, 0.7, 0.5, 0.3].forEach((iou, k) => {
    let lo = 0, hi = 2 * R; for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (discIoU(R, m) > iou) lo = m; else hi = m; }
    const cx = 110 + k * 230, cy = 105;
    body += circle(cx - lo / 2, cy, R, { fill: "#999", op: 0.25, stroke: "#555", w: 3 });
    body += circle(cx + lo / 2, cy, R, { fill: C.blue, op: 0.2, stroke: C.blue, w: 3 });
    body += text(cx, cy + R + 38, `IoU ${iou.toFixed(1)}`, { weight: "bold" });
    body += text(cx, cy + R + 64, `Dice ${(2 * iou / (1 + iou)).toFixed(2)}, shift ${(lo / R * 100).toFixed(0)} % of r`, { size: 17 });
  });
  fs.writeFileSync(ROOT + "object-eval/fig-iou.svg", svg(920, 240, body));
}
// ---------------------------------------------------------------- object-eval/fig-merge.svg
{
  const t1 = inDisc(-17, 0, 20), t2 = inDisc(17, 0, 20), p = (x: number, y: number) => (Math.abs(x) <= 17 ? Math.abs(y) <= 20 : (Math.abs(x) - 17) ** 2 + y * y <= 400);
  const tu = (x: number, y: number) => t1(x, y) || t2(x, y);
  const A = areas([tu, p], [-45, -25, 45, 25], 0.1);
  const dice = 2 * A[0][1] / (A[0][0] + A[1][1]);
  console.log("fig-merge Dice", dice.toFixed(3));
  const a1 = areas([t1, p], [-45, -25, 45, 25], 0.1); const iou1 = a1[0][1] / (a1[0][0] + a1[1][1] - a1[0][1]);
  console.log("fig-merge IoU(T1,P)", iou1.toFixed(3));
  const s = 3.2; let body = "";
  const L = (x: number) => 190 + s * x, R2 = (x: number) => 610 + s * x, Yc = 110;
  body += text(190, 30, "truth: 2 nuclei", { weight: "bold" }) + text(610, 30, "prediction: 1 object", { weight: "bold" });
  body += circle(L(-17), Yc, 20 * s, { fill: "#4c9be8", op: 0.35, stroke: "#1f77b4", w: 3 }) + circle(L(17), Yc, 20 * s, { fill: "#e88b4c", op: 0.35, stroke: "#c85a00", w: 3 });
  body += circle(R2(-17), Yc, 20 * s, { stroke: "#999", w: 2, dash: "6 4" }) + circle(R2(17), Yc, 20 * s, { stroke: "#999", w: 2, dash: "6 4" });
  body += `<path d="M${R2(-17)},${Yc - 20 * s}H${R2(17)}A${20 * s},${20 * s} 0 0 1 ${R2(17)},${Yc + 20 * s}H${R2(-17)}A${20 * s},${20 * s} 0 0 1 ${R2(-17)},${Yc - 20 * s}Z" fill="${C.tp}" fill-opacity="0.3" stroke="${C.tp}" stroke-width="3"/>`;
  body += text(400, 215, `pixel Dice ${dice.toFixed(2)}   ·   objects: 1 instead of 2   ·   IoU with each nucleus ${iou1.toFixed(2)}`, { size: 19 });
  fs.writeFileSync(ROOT + "object-eval/fig-merge.svg", svg(820, 235, body));
}
// ---------------------------------------------------------------- object-eval/fig-greedy.svg
{
  const g1 = inDisc(0, 0, 20), g2 = inDisc(34, 0, 15), p1 = inEll(15, 0, 30, 17), p2 = inDisc(-9, 0, 13);
  const box: [number, number, number, number] = [-30, -25, 55, 25];
  const A = areas([g1, g2, p1, p2], box, 0.1);
  const iou = (i: number, j: number) => A[i][j] / (A[i][i] + A[j][j] - A[i][j]);
  const T = [[iou(0, 2), iou(0, 3)], [iou(1, 2), iou(1, 3)]];
  console.log("fig-greedy table", T.map(r => r.map(v => v.toFixed(2))));
  const tb = EV.tableFromIou(T);
  console.log(" greedy TP", EV.matchGreedy(tb, 0.3).tp, "optimal TP", EV.matchOptimal(tb, 0.3).tp);
  const s = 4, X = (x: number) => 150 + s * x, Y = (y: number) => 120 + s * y;
  let body = "";
  body += circle(X(0), Y(0), 20 * s, { stroke: "#555", w: 3 }) + circle(X(34), Y(0), 15 * s, { stroke: "#555", w: 3 });
  body += ellipse(X(15), Y(0), 30 * s, 17 * s, { fill: C.blue, op: 0.18, stroke: C.blue, w: 3, dash: "8 5" });
  body += circle(X(-9), Y(0), 13 * s, { fill: C.purple, op: 0.25, stroke: C.purple, w: 3 });
  body += text(X(0), Y(-23), "G1", { weight: "bold" }) + text(X(37), Y(-17), "G2", { weight: "bold" });
  body += text(X(25), Y(13), "P1", { fill: C.blue, weight: "bold" }) + text(X(-9), Y(4), "P2", { fill: C.purple, weight: "bold" });
  // table
  const tx = 520, ty = 50, cw = 80, ch = 44;
  body += text(tx + cw * 1.5, ty - 12, "P1", { fill: C.blue, weight: "bold" }) + text(tx + cw * 2.5, ty - 12, "P2", { fill: C.purple, weight: "bold" });
  ["G1", "G2"].forEach((g, i) => {
    body += text(tx + cw * 0.6, ty + ch * (i + 0.5) + 7, g, { weight: "bold" });
    T[i].forEach((v, j) => { body += `<rect x="${tx + cw * (j + 1)}" y="${ty + ch * i}" width="${cw - 4}" height="${ch - 4}" fill="${v >= 0.3 ? "#cfe8cf" : "#f2f2f2"}"/>` + text(tx + cw * (j + 1.5) - 2, ty + ch * (i + 0.5) + 5, v.toFixed(2)); });
  });
  body += text(tx + 110, ty + 130, "τ = 0.3:", { size: 18, anchor: "start" });
  body += text(tx, ty + 160, "greedy: G1–P1 first → 1 match", { size: 18, anchor: "start" });
  body += text(tx, ty + 188, "optimal: G1–P2, G2–P1 → 2 matches", { size: 18, anchor: "start" });
  fs.writeFileSync(ROOT + "object-eval/fig-greedy.svg", svg(860, 250, body));
}
// ---------------------------------------------------------------- object-eval/fig-pr.svg
{
  const pr = EV.prFromScores([5, 4, 3, 2, 1], [1, 1, 0, 1, 0], 4);
  const ax = axes(80, 20, 380, 280, [0, 1], [0, 1.05], [0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1], "recall", "precision", v => String(v));
  let body = ax.s;
  // interpolated area
  const R = [0, ...pr.recall], I = [pr.interp[0], ...pr.interp];
  const area: [number, number][] = [[ax.X(0), ax.Y(0)]];
  for (let k = 1; k < R.length; k++) { area.push([ax.X(R[k - 1]), ax.Y(I[k])], [ax.X(R[k]), ax.Y(I[k])]); }
  area.push([ax.X(R[R.length - 1]), ax.Y(0)]);
  body += path(area, { fill: C.blue, op: 0.15, stroke: "none", close: true });
  body += path(area.slice(1, -1), { stroke: C.blue, w: 3 });
  const tags = ["TP", "TP", "FP", "TP", "FP"];
  for (let k = 0; k < 5; k++) { body += circle(ax.X(pr.recall[k]), ax.Y(pr.precision[k]), 8, { fill: tags[k] === "TP" ? C.tp : C.fp, stroke: "#fff", w: 1.5 }); body += text(ax.X(pr.recall[k]) + (k === 2 ? -14 : 14), ax.Y(pr.precision[k]) + (k === 4 ? 26 : -12), `${k + 1}`, { size: 17, anchor: k === 2 ? "end" : "start" }); }
  body += text(270, 260, `AP = ${pr.ap.toFixed(4)} (shaded area)`, { size: 19 });
  fs.writeFileSync(ROOT + "object-eval/fig-pr.svg", svg(500, 370, body));
}
// ---------------------------------------------------------------- object-eval/fig-f1tau.svg (from the objmatch model)
{
  const sc = OM.makeScene(1);
  const curve = (id: string) => { const p = OM.PRESETS.find(q => q.id === id)!; const o = OM.generate(sc, p.settings); const lab = OM.rasterise(o); return OM.evaluate(sc.truth.labels, sc.truth.count, lab, o.length, o.map(q => q.score), 0.5, "optimal"); };
  const a = curve("sloppy"), b = curve("dice");
  console.log("fig-f1tau sloppy F1@0.5", a.f1[OM.TAUS.indexOf(0.5)].toFixed(3), "@0.8", a.f1[OM.TAUS.indexOf(0.8)].toFixed(3), "dice", a.dice.toFixed(3), "| merges F1@0.5", b.f1[OM.TAUS.indexOf(0.5)].toFixed(3), "@0.8", b.f1[OM.TAUS.indexOf(0.8)].toFixed(3), "dice", b.dice.toFixed(3));
  const ax = axes(80, 20, 480, 280, [0.1, 0.95], [0, 1.02], [0.1, 0.3, 0.5, 0.7, 0.9], [0, 0.2, 0.4, 0.6, 0.8, 1], "IoU threshold τ", "object F1");
  let body = ax.s;
  body += path(OM.TAUS.map((t, i) => [ax.X(t), ax.Y(a.f1[i])] as [number, number]), { stroke: C.blue, w: 3.5 });
  body += path(OM.TAUS.map((t, i) => [ax.X(t), ax.Y(b.f1[i])] as [number, number]), { stroke: C.fp, w: 3.5 });
  body += text(ax.X(0.12), ax.Y(0.08), "", {});
  body += line(600, 60, 640, 60, C.blue, 3.5) + text(650, 66, "sloppy outlines, no merges", { size: 18, anchor: "start" });
  body += line(600, 95, 640, 95, C.fp, 3.5) + text(650, 101, "tight outlines, pairs merged", { size: 18, anchor: "start" });
  body += text(650, 140, `pixel Dice: ${a.dice.toFixed(2)} and ${b.dice.toFixed(2)}`, { size: 18, anchor: "start" });
  fs.writeFileSync(ROOT + "object-eval/fig-f1tau.svg", svg(920, 370, body));
}
// ---------------------------------------------------------------- agreement/fig-boundary.svg (contourdist model)
{
  let body = "";
  const draw = (cx: number, id: string, title: string) => {
    const p = CD.PRESETS.find(q => q.id === id)!.make(), r = CD.evaluate(p), s = 2.2;
    body += text(cx, 30, title, { weight: "bold" });
    body += path(r.truth.map(q => [cx + s * q[0], 150 + s * q[1]] as [number, number]), { stroke: "#666", w: 3, close: true, fill: "#999", op: 0.15 });
    for (const c of r.pred) body += path(c.map(q => [cx + s * q[0], 150 + s * q[1]] as [number, number]), { stroke: C.blue, w: 2.5, close: true });
    body += text(cx, 268, `IoU ${r.iou.toFixed(3)}`, { size: 18 }) + text(cx, 292, `Hausdorff ${r.dist.hausdorff.toFixed(1)} px · ASD ${r.dist.asd.toFixed(2)} px`, { size: 18 });
    console.log("fig-boundary", id, "IoU", r.iou.toFixed(4), "Dice", r.dice.toFixed(4), "H", r.dist.hausdorff.toFixed(2), "HD95", r.dist.hd95.toFixed(2), "ASD", r.dist.asd.toFixed(3));
  };
  draw(160, "dilate", "1 px too large everywhere");
  draw(480, "spike", "one thin spike");
  draw(800, "fragment", "a stray fragment");
  fs.writeFileSync(ROOT + "agreement/fig-boundary.svg", svg(960, 305, body));
}
// ---------------------------------------------------------------- agreement/fig-ba.svg (by-hand data)
{
  const man = [20, 35, 50, 65, 80, 100], au = [19, 33, 46, 60, 72, 88], ba = ST.blandAltman(man, au);
  const a1 = axes(80, 20, 300, 280, [0, 110], [0, 110], [0, 50, 100], [0, 50, 100], "manual count", "automated count");
  let body = a1.s + path([[a1.X(0), a1.Y(0)], [a1.X(110), a1.Y(110)]], { stroke: "#999", w: 2, dash: "6 4" });
  man.forEach((m, i) => (body += circle(a1.X(m), a1.Y(au[i]), 7, { fill: C.blue })));
  body += text(a1.X(30), a1.Y(100), `r = ${ba.r.toFixed(4)}`, { size: 19 });
  const a2 = axes(560, 20, 330, 280, [0, 110], [-16, 6], [0, 50, 100], [-15, -10, -5, 0, 5], "mean of the two", "auto − manual");
  body += a2.s + line(a2.X(0), a2.Y(0), a2.X(110), a2.Y(0), "#999", 1.5);
  body += line(a2.X(0), a2.Y(ba.bias), a2.X(110), a2.Y(ba.bias), C.fp, 2.5) + text(a2.X(108), a2.Y(ba.bias) - 8, `bias ${ba.bias.toFixed(2)}`, { size: 17, anchor: "end", fill: C.fp });
  for (const v of [ba.lo, ba.hi]) body += line(a2.X(0), a2.Y(v), a2.X(110), a2.Y(v), C.fn, 2, "6 4");
  body += text(a2.X(108), a2.Y(ba.hi) - 8, `+1.96 SD ${ba.hi.toFixed(1)}`, { size: 17, anchor: "end", fill: C.fn });
  body += text(a2.X(108), a2.Y(ba.lo) + 22, `−1.96 SD ${ba.lo.toFixed(1)}`, { size: 17, anchor: "end", fill: C.fn });
  ba.means.forEach((m, i) => (body += circle(a2.X(m), a2.Y(ba.diffs[i]), 7, { fill: C.blue })));
  body += line(a2.X(10), a2.Y(ba.intercept + ba.slope * 10), a2.X(105), a2.Y(ba.intercept + ba.slope * 105), C.tp, 2, "3 4");
  fs.writeFileSync(ROOT + "agreement/fig-ba.svg", svg(920, 370, body));
}
// ---------------------------------------------------------------- leakage/fig-splits.svg
{
  const r = rng(4), n = 16, cs = 12; let body = "";
  const grid = (x0: number, title: string, role: (x: number, y: number) => number) => {
    body += text(x0 + n * cs / 2, 28, title, { weight: "bold", size: 19 });
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const v = role(x, y), col = v === 1 ? C.fp : v === 2 ? "#bbb" : C.blue;
      body += `<rect x="${x0 + x * cs}" y="${45 + y * cs}" width="${cs - 1}" height="${cs - 1}" fill="${col}" fill-opacity="${v === 2 ? 0.9 : 0.75}"/>`;
    }
  };
  const rp = Array.from({ length: n * n }, () => (r.unif() < 0.25 ? 1 : 0));
  grid(20, "random pixels", (x, y) => rp[y * n + x]);
  const bl = Array.from({ length: 16 }, () => (r.unif() < 0.25 ? 1 : 0)); bl[5] = 1; bl[10] = 0; bl[3] = 0;
  grid(250, "random blocks", (x, y) => bl[Math.floor(y / 4) * 4 + Math.floor(x / 4)]);
  grid(480, "blocks + buffer", (x, y) => { const b = bl[Math.floor(y / 4) * 4 + Math.floor(x / 4)]; if (b) return 1; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const xx = x + dx, yy = y + dy; if (xx >= 0 && yy >= 0 && xx < n && yy < n && bl[Math.floor(yy / 4) * 4 + Math.floor(xx / 4)]) return 2; } return 0; });
  // image split: 4 small images
  body += text(800, 28, "by image", { weight: "bold", size: 19 });
  for (let k = 0; k < 4; k++) body += `<rect x="${710 + (k % 2) * 98}" y="${45 + Math.floor(k / 2) * 98}" width="92" height="92" fill="${k === 1 ? C.fp : C.blue}" fill-opacity="0.75"/>`;
  body += `<rect x="20" y="255" width="18" height="18" fill="${C.fp}" fill-opacity="0.75"/>` + text(46, 270, "test", { size: 18, anchor: "start" });
  body += `<rect x="110" y="255" width="18" height="18" fill="${C.blue}" fill-opacity="0.75"/>` + text(136, 270, "train", { size: 18, anchor: "start" });
  body += `<rect x="210" y="255" width="18" height="18" fill="#bbb"/>` + text(236, 270, "buffer (not used)", { size: 18, anchor: "start" });
  fs.writeFileSync(ROOT + "leakage/fig-splits.svg", svg(920, 290, body));
}
// ---------------------------------------------------------------- replicates/fig-null.svg
{
  // two conditions, 4 experiments each, experiment means with SD 1; observed difference 1.6
  const r = rng(12), N = 20000, d: number[] = [];
  for (let i = 0; i < N; i++) { let a = 0, b = 0; for (let k = 0; k < 4; k++) { a += r.norm(); b += r.norm(); } d.push((b - a) / 4); }
  const obs = 1.6, p = d.filter(v => Math.abs(v) >= obs).length / N;
  const se = Math.sqrt(2 / 4), pz = ST.zTwoSidedP(obs / se);
  console.log("fig-null p sim", p.toFixed(4), "SE", se.toFixed(3), "z p", pz.toFixed(4), "share beyond +1.6", (d.filter(v => v >= obs).length / N).toFixed(4));
  const nb = 40, lo = -3, hi = 3, w = (hi - lo) / nb, c = new Array(nb).fill(0);
  d.forEach(v => { const k = Math.floor((v - lo) / w); if (k >= 0 && k < nb) c[k]++; });
  const ax = axes(80, 20, 560, 260, [lo, hi], [0, Math.max(...c) * 1.1], [-3, -2, -1, 0, 1, 2, 3], [], "difference of the means (treated − control)", "how often");
  let body = ax.s;
  c.forEach((v, k) => { const x0 = lo + k * w, out = Math.abs(x0 + w / 2) >= obs; body += `<rect x="${ax.X(x0).toFixed(1)}" y="${ax.Y(v).toFixed(1)}" width="${(ax.X(x0 + w) - ax.X(x0) - 1).toFixed(1)}" height="${(ax.Y(0) - ax.Y(v)).toFixed(1)}" fill="${out ? C.fp : C.blue}" fill-opacity="0.7"/>`; });
  body += line(ax.X(obs), ax.Y(0), ax.X(obs), 30, C.fg, 3) + text(ax.X(obs) + 8, 50, `observed ${obs}`, { size: 18, anchor: "start" });
  body += text(ax.X(-2.9), 50, `chance alone: ${(100 * p).toFixed(1)} % as far out`, { size: 18, anchor: "start" });
  fs.writeFileSync(ROOT + "replicates/fig-null.svg", svg(680, 350, body));
}
// ---------------------------------------------------------------- replicates/fig-pvals.svg (featurescan model)
{
  let body = "";
  const draw = (x0: number, s: Partial<FSm.Settings>, title: string) => {
    const set = { ...FSm.DEFAULTS, ...s }, a = FSm.analyse(FSm.makePlate(set));
    const nb = 20, c = new Array(nb).fill(0); a.p.forEach(v => c[Math.min(nb - 1, Math.floor(v * nb))]++);
    const ax = axes(x0, 40, 330, 230, [0, 1], [0, 80], [0, 0.5, 1], [0, 20, 40, 60, 80], "p-value", "features");
    body += ax.s + text(x0 + 165, 25, title, { weight: "bold", size: 19 });
    c.forEach((v, k) => (body += `<rect x="${ax.X(k / nb).toFixed(1)}" y="${ax.Y(Math.min(v, 80)).toFixed(1)}" width="${(ax.X((k + 1) / nb) - ax.X(k / nb) - 1).toFixed(1)}" height="${(ax.Y(0) - ax.Y(Math.min(v, 80))).toFixed(1)}" fill="${C.blue}" fill-opacity="0.7"/>`));
    body += line(ax.X(0), ax.Y(300 / nb), ax.X(1), ax.Y(300 / nb), C.fp, 2, "6 4");
    console.log("fig-pvals", title, "first bin", c[0], "raw", a.raw, "false", a.falseRaw, "bonf", a.bonf, "BH", a.bh, "falseBH", a.falseBH, "real", a.nReal);
  };
  draw(90, {}, "300 features, none real");
  draw(560, { fracReal: 0.1 }, "300 features, 30 real");
  fs.writeFileSync(ROOT + "replicates/fig-pvals.svg", svg(940, 340, body));
}
