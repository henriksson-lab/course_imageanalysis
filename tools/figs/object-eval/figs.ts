// Extra figures for object-eval/ (slides). Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/object-eval/figs.ts
// then python3 -I tools/figs/object-eval/scene.py <dir>/objeval-scene.json. Numbers are printed so the text can quote them.
import * as fs from "fs";
import * as os from "os";
import * as OM from "../../../demos/src/objmatch/model";
import * as EV from "../../../demos/src/lib/eval";
import { ROOT, C, svg, text, line, circle, rect, path, axes, areas, iouOf, inDisc } from "../evaluation/svglib";
const TMP = process.env.FIGTMP || os.tmpdir();

// ------------------------------------------------ scene of the objmatch demo, preset "a typical result" (for scene.py)
{
  const sc = OM.makeScene(1), p = OM.PRESETS[0];
  const pr = OM.predict(sc, p.settings);
  const ev = OM.evaluate(sc.truth.labels, sc.truth.count, pr.labels, pr.count, pr.scores, p.tau, p.method);
  console.log("scene: TP", ev.match.tp, "FP", ev.match.fp, "FN", ev.match.fn, "F1", ev.scores.f1.toFixed(3), "Dice", ev.dice.toFixed(3), "merges", ev.merges.length, "splits", ev.splits.length);
  fs.writeFileSync(TMP + "/objeval-scene.json", JSON.stringify({
    W: OM.W, image: Array.from(sc.image.data, v => Math.round(v)), truth: Array.from(sc.truth.labels), pred: Array.from(pr.labels),
    matchT: Array.from(ev.match.matchT), matchP: Array.from(ev.match.matchP), merges: ev.merges.map(m => m.p),
    tp: ev.match.tp, fp: ev.match.fp, fn: ev.match.fn, f1: ev.scores.f1, dice: ev.dice,
  }));
}

// ------------------------------------------------ fig-errors.svg: split, merge, miss, speck (2 × 2)
{
  const s = 2.4; let body = "";
  const panel = (cx: number, cy: number, title: string, foot: string, foot2: string) => {
    body += text(cx, cy - 75, title, { weight: "bold" }) + text(cx, cy + 82, foot, { size: 18 }) + text(cx, cy + 106, foot2, { size: 19, weight: "bold" });
  };
  // split: truth disc r 20; two halves with a 2-px gap
  {
    const g = inDisc(0, 0, 20), h1 = (x: number, y: number) => g(x, y) && x <= -1, h2 = (x: number, y: number) => g(x, y) && x >= 1;
    const A = areas([g, h1, h2], [-21, -21, 21, 21], 0.1), i1 = iouOf(A, 0, 1), i2 = iouOf(A, 0, 2);
    console.log("errors split IoU", i1.toFixed(3), i2.toFixed(3));
    const cx = 140, cy = 100, r = 20 * s;
    body += `<path d="M${cx - s},${cy - Math.sqrt(r * r - s * s)}A${r},${r} 0 0 0 ${cx - s},${cy + Math.sqrt(r * r - s * s)}Z" fill="${C.fp}" fill-opacity="0.3" stroke="${C.fp}" stroke-width="3"/>`;
    body += `<path d="M${cx + s},${cy - Math.sqrt(r * r - s * s)}A${r},${r} 0 0 1 ${cx + s},${cy + Math.sqrt(r * r - s * s)}Z" fill="${C.fp}" fill-opacity="0.3" stroke="${C.fp}" stroke-width="3"/>`;
    body += circle(cx, cy, r, { stroke: C.fn, w: 3, dash: "8 5" });
    panel(cx, cy, "split", `IoU ${((i1 + i2) / 2).toFixed(2)} with each half`, "1 FN + 2 FP");
  }
  // merge: two discs r 13, one capsule covering both
  {
    const t1 = inDisc(-12, 0, 13), t2 = inDisc(12, 0, 13), p = (x: number, y: number) => (Math.abs(x) <= 12 ? Math.abs(y) <= 13 : (Math.abs(x) - 12) ** 2 + y * y <= 169);
    const A = areas([t1, t2, p], [-26, -14, 26, 14], 0.1), i1 = iouOf(A, 0, 2);
    console.log("errors merge IoU", i1.toFixed(3));
    const cx = 410, cy = 100, r = 13 * s;
    body += `<path d="M${cx - 12 * s},${cy - r}H${cx + 12 * s}A${r},${r} 0 0 1 ${cx + 12 * s},${cy + r}H${cx - 12 * s}A${r},${r} 0 0 1 ${cx - 12 * s},${cy - r}Z" fill="${C.fp}" fill-opacity="0.3" stroke="${C.fp}" stroke-width="3"/>`;
    body += circle(cx - 12 * s, cy, r, { stroke: C.fn, w: 3, dash: "8 5" }) + circle(cx + 12 * s, cy, r, { stroke: C.fn, w: 3, dash: "8 5" });
    panel(cx, cy, "merge", `IoU ${i1.toFixed(2)} with each nucleus`, "2 FN + 1 FP");
  }
  {
    const cx = 140, cy = 335;
    body += circle(cx, cy, 15 * s, { fill: "#bbb", op: 0.35, stroke: C.fn, w: 3, dash: "8 5" });
    panel(cx, cy, "missed", "dim nucleus, no prediction", "1 FN");
  }
  {
    const cx = 410, cy = 335;
    body += circle(cx, cy, 3 * s, { fill: C.fp, op: 0.4, stroke: C.fp, w: 3 });
    panel(cx, cy, "speck", "prediction, no nucleus", "1 FP");
  }
  body += circle(30, 488, 10, { stroke: C.fn, w: 3, dash: "5 3" }) + text(48, 495, "true object", { size: 18, anchor: "start" });
  body += circle(185, 488, 10, { fill: C.fp, op: 0.3, stroke: C.fp, w: 3 }) + text(203, 495, "prediction; counts at τ = 0.5", { size: 18, anchor: "start" });
  fs.writeFileSync(ROOT + "object-eval/fig-errors.svg", svg(550, 510, body));
}

// ------------------------------------------------ fig-ioushift.svg: IoU of a disc and its shifted copy
{
  const discIoU = (r: number, d: number) => { if (d >= 2 * r) return 0; const A = 2 * r * r * Math.acos(d / (2 * r)) - (d / 2) * Math.sqrt(4 * r * r - d * d); return A / (2 * Math.PI * r * r - A); };
  console.log("iou r3 d1", discIoU(3, 1).toFixed(3), "r20 d1", discIoU(20, 1).toFixed(3), "approx r3", (1 - 4 / (Math.PI * 3)).toFixed(3), "approx r20", (1 - 4 / (Math.PI * 20)).toFixed(3));
  const ax = axes(80, 15, 220, 200, [0, 6], [0, 1], [0, 2, 4, 6], [0, 0.5, 1], "shift d (pixels)", "IoU");
  let body = ax.s;
  const cols: [number, string][] = [[3, C.fp], [8, C.purple], [20, C.blue]];
  for (const [r, col] of cols) {
    const ds = Array.from({ length: 121 }, (_, i) => i * 0.05);
    body += path(ds.map(d => [ax.X(d), ax.Y(discIoU(r, d))] as [number, number]), { stroke: col, w: 3.5 });
    body += path(ds.filter(d => 1 - 4 * d / (Math.PI * r) > 0).map(d => [ax.X(d), ax.Y(1 - 4 * d / (Math.PI * r))] as [number, number]), { stroke: col, w: 2, dash: "6 5" });
  }
  body += circle(ax.X(1), ax.Y(discIoU(3, 1)), 7, { fill: C.fp }) + text(ax.X(1) + 12, ax.Y(discIoU(3, 1)) + 22, discIoU(3, 1).toFixed(2), { size: 18, anchor: "start", fill: C.fp });
  body += circle(ax.X(1), ax.Y(discIoU(20, 1)), 7, { fill: C.blue }) + text(ax.X(1) + 10, ax.Y(discIoU(20, 1)) - 12, discIoU(20, 1).toFixed(2), { size: 18, anchor: "start", fill: C.blue });
  [["r = 20 px", C.blue], ["r = 8 px", C.purple], ["r = 3 px", C.fp]].forEach(([l, c], i) => { body += line(318, 40 + 30 * i, 348, 40 + 30 * i, c, 3.5) + text(354, 46 + 30 * i, l, { size: 19, anchor: "start" }); });
  body += line(318, 130, 348, 130, C.grey, 2, "6 5") + text(354, 136, "approx.", { size: 19, anchor: "start" });
  fs.writeFileSync(ROOT + "object-eval/fig-ioushift.svg", svg(450, 280, body));
}

// ------------------------------------------------ fig-border.svg: objects touching the image border
{
  let body = "";
  const x0 = 20, y0 = 15, w = 500, h = 300;
  const objs: [number, number, number][] = [[130, 110, 34], [260, 210, 38], [380, 95, 32], [440, 235, 30], [28, 250, 34], [520, 150, 36], [270, 18, 30], [150, 315, 28]];
  body += `<defs><clipPath id="img"><rect x="${x0}" y="${y0}" width="${w}" height="${h}"/></clipPath></defs>`;
  body += rect(x0, y0, w, h, { fill: "#f4f4f4" });
  let inside = 0, cut = 0;
  for (const [x, y, r] of objs) {
    const touches = x - r < x0 || x + r > x0 + w || y - r < y0 || y + r > y0 + h;
    if (touches) cut++; else inside++;
    body += `<g clip-path="url(#img)">${circle(x, y, r, touches ? { fill: "#bbb", op: 0.6, stroke: C.grey, w: 2.5, dash: "6 4" } : { fill: C.blue, op: 0.3, stroke: C.blue, w: 3 })}</g>`;
  }
  body += rect(x0, y0, w, h, { stroke: C.fg, w: 3 });
  body += circle(x0 + 14, 352, 12, { fill: C.blue, op: 0.3, stroke: C.blue, w: 3 }) + text(x0 + 36, 359, `inside: counted (${inside})`, { size: 20, anchor: "start" });
  body += circle(x0 + 14, 392, 12, { fill: "#bbb", op: 0.6, stroke: C.grey, w: 2.5, dash: "6 4" }) + text(x0 + 36, 399, `touching the border (${cut}): left out`, { size: 20, anchor: "start" });
  body += text(x0 + 36, 426, "of the truth and the prediction alike", { size: 20, anchor: "start" });
  fs.writeFileSync(ROOT + "object-eval/fig-border.svg", svg(540, 440, body));
}
