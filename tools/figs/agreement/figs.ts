// Extra figures for agreement/ (slides). Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/agreement/figs.ts
// then python3 -I tools/figs/agreement/annotators.py <dir>/agreement-scene.json. Numbers are printed so the text can quote them.
import * as fs from "fs";
import * as os from "os";
import * as CD from "../../../demos/src/contourdist/model";
import * as OM from "../../../demos/src/objmatch/model";
import * as EV from "../../../demos/src/lib/eval";
import * as ST from "../../../demos/src/lib/stats";
import { rng } from "../../../demos/src/lib/random";
import { ROOT, C, svg, text, line, circle, path, axes } from "../evaluation/svglib";
const TMP = process.env.FIGTMP || os.tmpdir();

// ------------------------------------------------ fig-whiskers.svg: nearest-point distances between two outlines
{
  const free = CD.PRESETS.find(p => p.id === "free")!.make();
  const set: CD.Settings = { R: 40, pred: { ...free.pred, offsets: free.pred.offsets.map(([a, b]) => [2 * a, 2 * b] as CD.Pt), spike: { on: true, phi: 0.6, len: 14 } as any } };
  const r = CD.evaluate(set);
  console.log("whiskers: H", r.dist.hausdorff.toFixed(2), "HD95", r.dist.hd95.toFixed(2), "ASD", r.dist.asd.toFixed(2), "IoU", r.iou.toFixed(3));
  const s = 4.2, cx = 225, cy = 250, P = (q: CD.Pt): [number, number] => [cx + s * q[0], cy - s * q[1]];
  let body = path(r.truth.map(P), { stroke: "#555", w: 3, close: true, fill: "#999", op: 0.15 });
  for (const c of r.pred) body += path(c.map(P), { stroke: C.blue, w: 3, close: true });
  // whiskers from evenly spaced points of the prediction to the truth
  const pts = CD.resample(r.pred[0], 5);
  let worst = { d: 0, p: [0, 0] as CD.Pt, q: [0, 0] as CD.Pt };
  for (const p of CD.resample(r.pred[0], 0.25)) { const n = CD.nearestOn(p, [r.truth]); if (n.d > worst.d) worst = { d: n.d, p, q: n.q }; }
  for (const p of pts) { const n = CD.nearestOn(p, [r.truth]); if (n.d > 0.6) { const a = P(p), b = P(n.q); body += line(a[0], a[1], b[0], b[1], C.fn, 2.5); } }
  const a = P(worst.p), b = P(worst.q);
  body += line(a[0], a[1], b[0], b[1], C.fp, 4) + circle(a[0], a[1], 5, { fill: C.fp });
  body += text(a[0] - 10, a[1] - 14, `Hausdorff ${r.dist.hausdorff.toFixed(1)} px`, { size: 20, anchor: "end", fill: C.fp, weight: "bold" });
  body += line(20, 480, 55, 480, "#555", 3) + text(63, 487, "true outline", { size: 19, anchor: "start" });
  body += line(200, 480, 235, 480, C.blue, 3) + text(243, 487, "prediction", { size: 19, anchor: "start" });
  body += line(20, 512, 55, 512, C.fn, 2.5) + text(63, 519, `distance to the other outline; mean (ASD) ${r.dist.asd.toFixed(1)} px`, { size: 19, anchor: "start" });
  fs.writeFileSync(ROOT + "agreement/fig-whiskers.svg", svg(500, 535, body));
}

// ------------------------------------------------ fig-corr.svg: r = 1 but 20 % off, vs unbiased scatter
{
  const r = rng(21), n = 15;
  const man = Array.from({ length: n }, (_, i) => Math.round(20 + 130 * i / (n - 1) + 4 * r.norm()));
  const a = man.map(m => 0.8 * m), b = man.map(m => Math.round(m + 8 * r.norm()));
  const ra = ST.pearson(man, a), rb = ST.pearson(man, b);
  const ea = ST.mean(man.map((m, i) => Math.abs(a[i] - m))), eb = ST.mean(man.map((m, i) => Math.abs(b[i] - m)));
  console.log("corr: A r", ra.toFixed(4), "mean |err|", ea.toFixed(1), "| B r", rb.toFixed(4), "mean |err|", eb.toFixed(1), "bias", ST.mean(man.map((m, i) => b[i] - m)).toFixed(1));
  let body = "";
  const panel = (x0: number, y: number[], col: string, title: string, l1: string, l2: string) => {
    const ax = axes(x0, 45, 220, 220, [0, 170], [0, 170], [0, 50, 100, 150], [0, 50, 100, 150], "manual count", x0 < 200 ? "automated count" : "");
    body += ax.s + text(x0 + 110, 25, title, { weight: "bold" });
    body += line(ax.X(0), ax.Y(0), ax.X(170), ax.Y(170), "#999", 2, "6 4");
    man.forEach((m, i) => (body += circle(ax.X(m), ax.Y(y[i]), 6, { fill: col, op: 0.85 })));
    body += text(ax.X(168), ax.Y(32), l1, { size: 19, anchor: "end" }) + text(ax.X(168), ax.Y(10), l2, { size: 19, anchor: "end" });
  };
  panel(80, a, C.fp, "always 80 %", `r = ${ra.toFixed(2)}`, `off by ${ea.toFixed(0)} on average`);
  panel(380, b, C.blue, "unbiased, noisy", `r = ${rb.toFixed(2)}`, `off by ${eb.toFixed(0)} on average`);
  body += line(150, 345, 185, 345, "#999", 2, "6 4") + text(193, 351, "identity line: perfect agreement", { size: 19, anchor: "start" });
  fs.writeFileSync(ROOT + "agreement/fig-corr.svg", svg(620, 365, body));
}

// ------------------------------------------------ fig-calib.svg: Bland–Altman before and after removing a stable bias
{
  const r = rng(5), n = 30;
  const make = (k: number) => { const man = Array.from({ length: k }, () => Math.round(20 + 120 * r.unif())); return { man, au: man.map(m => Math.round(m - 10 + 4 * r.norm())) }; };
  const cal = make(20), test = make(n);
  const shift = -ST.mean(cal.man.map((m, i) => cal.au[i] - m));
  const before = ST.blandAltman(test.man, test.au), after = ST.blandAltman(test.man, test.au.map(v => v + shift));
  console.log("calib: shift", shift.toFixed(2), "before bias", before.bias.toFixed(2), "limits", before.lo.toFixed(1), before.hi.toFixed(1), "| after bias", after.bias.toFixed(2), "limits", after.lo.toFixed(1), after.hi.toFixed(1));
  let body = "";
  const panel = (x0: number, ba: ReturnType<typeof ST.blandAltman>, title: string) => {
    const ax = axes(x0, 45, 230, 230, [0, 160], [-25, 15], [0, 50, 100, 150], [-20, -10, 0, 10], "mean of the two", x0 < 200 ? "auto − manual" : "");
    body += ax.s + text(x0 + 115, 25, title, { weight: "bold" });
    body += line(ax.X(0), ax.Y(0), ax.X(160), ax.Y(0), "#999", 1.5);
    body += line(ax.X(0), ax.Y(ba.bias), ax.X(160), ax.Y(ba.bias), C.fp, 2.5);
    for (const v of [ba.lo, ba.hi]) body += line(ax.X(0), ax.Y(v), ax.X(160), ax.Y(v), C.fn, 2, "6 4");
    ba.means.forEach((m, i) => (body += circle(ax.X(m), ax.Y(ba.diffs[i]), 5.5, { fill: C.blue, op: 0.85 })));
    body += text(x0 + 115, 360, `bias ${ba.bias.toFixed(1)}`, { size: 19, fill: C.fp }) + text(x0 + 115, 385, `limits ${ba.lo.toFixed(1)} to ${ba.hi.toFixed(1)}`, { size: 19, fill: "#b35900" });
  };
  panel(80, before, "before");
  panel(390, after, "after calibration");
  fs.writeFileSync(ROOT + "agreement/fig-calib.svg", svg(640, 400, body));
}

// ------------------------------------------------ fig-arearatio.svg: area ratio of a disc drawn 1 px too large
{
  const f = (r: number) => (r + 1) ** 2 / r ** 2;
  console.log("arearatio", [3, 10, 15].map(r => f(r).toFixed(2)).join(" "));
  const ax = axes(80, 20, 330, 240, [2, 20], [1, 2.3], [2, 5, 10, 15, 20], [1, 1.5, 2], "object radius r (pixels)", "measured / true area");
  let body = ax.s + line(ax.X(2), ax.Y(1), ax.X(20), ax.Y(1), "#999", 1.5);
  body += path(Array.from({ length: 181 }, (_, i) => 2 + i * 0.1).map(r => [ax.X(r), ax.Y(f(r))] as [number, number]), { stroke: C.blue, w: 3.5 });
  for (const r of [3, 10, 15]) body += circle(ax.X(r), ax.Y(f(r)), 7, { fill: C.fp }) + text(ax.X(r) + 10, ax.Y(f(r)) - 12, f(r).toFixed(2), { size: 19, anchor: "start", fill: C.fp });
  fs.writeFileSync(ROOT + "agreement/fig-arearatio.svg", svg(440, 330, body));
}

// ------------------------------------------------ fig-sen.svg: standard error of a recall of 0.9
{
  const se = (n: number) => ST.seProportion(0.9, n);
  console.log("SE n=30", se(30).toFixed(3), "n=300", se(300).toFixed(3));
  const ax = axes(85, 20, 330, 240, [10, 1000], [0, 0.1], [10, 30, 100, 300, 1000], [0, 0.05, 0.1], "independent objects n", "SE of the recall", v => String(v), { logx: true });
  let body = ax.s;
  body += path(Array.from({ length: 101 }, (_, i) => 10 ** (1 + 2 * i / 100)).map(n => [ax.X(n), ax.Y(se(n))] as [number, number]), { stroke: C.blue, w: 3.5 });
  for (const n of [30, 300]) body += circle(ax.X(n), ax.Y(se(n)), 7, { fill: C.fp }) + text(ax.X(n) + 10, ax.Y(se(n)) - 12, se(n).toFixed(3), { size: 19, anchor: "start", fill: C.fp });
  fs.writeFileSync(ROOT + "agreement/fig-sen.svg", svg(440, 330, body));
}

// ------------------------------------------------ scene for annotators.py: two annotators of the objmatch scene
{
  const sc = OM.makeScene(1);
  const T = EV.iouTable(sc.truth.labels, sc.annotatorB.labels, sc.truth.count, sc.annotatorB.count);
  const sB = EV.objectScores(EV.match(T, 0.5, "optimal")), sB8 = EV.objectScores(EV.match(T, 0.8, "optimal"));
  console.log("annotators: A", sc.truth.count, "B", sc.annotatorB.count, "F1@0.5", sB.f1.toFixed(3), "F1@0.8", sB8.f1.toFixed(3));
  fs.writeFileSync(TMP + "/agreement-scene.json", JSON.stringify({ W: OM.W, image: Array.from(sc.image.data, v => Math.round(v)), a: Array.from(sc.truth.labels), b: Array.from(sc.annotatorB.labels), nA: sc.truth.count, nB: sc.annotatorB.count, f1: sB.f1, f18: sB8.f1 }));
}
