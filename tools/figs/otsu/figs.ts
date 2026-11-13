// Extra slide figures for otsu/lec.qmd (the first set is in tools/figs/thresholding/figs.ts).
// Run: node_modules/.bin/esbuild tools/figs/otsu/figs.ts --bundle --platform=node --outfile=/tmp/x.js && node /tmp/x.js
import { intHistogram } from "/data/henriksson/github/course/imageanalysis/demos/src/lib/histogram";
import * as TH from "/data/henriksson/github/course/imageanalysis/demos/src/lib/threshold";
import * as HM from "/data/henriksson/github/course/imageanalysis/demos/src/histthresh/model";
import { mixtureModel } from "/data/henriksson/github/course/imageanalysis/demos/src/mixture/model";
import { Svg, Panel, C } from "../thresholding/plot";

const f2 = (v: number) => v.toFixed(2);
const edges256 = Array.from({ length: 257 }, (_, i) => i - 0.5);
const xs256 = Array.from({ length: 256 }, (_, i) => i);

{ // tight classes vs wide classes: the same histogram split at a bad t and at Otsu's t
  const img = HM.makeImage({ source: "blobs", contrast: 100, noise: 20, median: false, blur: 0 }, HM.blobTruth(), null);
  const h = intHistogram(img.data, 0, 255).counts, o = TH.otsu(h);
  const S = new Svg(600, 600), mx = Math.max(...Array.from(h));
  [75, o.t].forEach((t, k) => {
    const st = TH.splitStats(h, t);
    console.log(`split t=${t}: w0 ${f2(st.w0)} mu0 ${f2(st.mu0)} mu1 ${f2(st.mu1)} sd0 ${f2(Math.sqrt(st.var0))} sd1 ${f2(Math.sqrt(st.var1))} varW ${st.varW.toFixed(0)} varB ${st.varB.toFixed(0)} varT ${st.varT.toFixed(0)}`);
    const p = new Panel(S, 80, 40 + k * 290, 490, 170, [0, 255], [0, mx * 1.5]);
    p.axes([0, 50, 100, 150, 200, 250], [], k ? "grey level" : "", "pixels");
    p.bars(edges256.slice(0, t + 2), Array.from(h).slice(0, t + 1), C.c1, 0.75);
    p.bars(edges256.slice(t + 1), Array.from(h).slice(t + 1), C.c4, 0.75);
    p.vline(t + 0.5, C.fg, "", 2.5);
    const y = mx * 1.3;
    p.line([st.mu0 - Math.sqrt(st.var0), st.mu0 + Math.sqrt(st.var0)], [y, y], C.c1, 4); p.dot(st.mu0, y, C.c1, 8);
    p.line([st.mu1 - Math.sqrt(st.var1), st.mu1 + Math.sqrt(st.var1)], [y, y], C.c4, 4); p.dot(st.mu1, y, C.c4, 8);
    S.text(80, 28 + k * 290, `${k ? "Otsu" : "too low"}: t = ${t}:  within ${st.varW.toFixed(0)} + between ${st.varB.toFixed(0)}`, C.fg, "start", 19);
  });
  S.text(575, 590, "dot = class mean, bar = ± one standard deviation", C.muted, "end", 18);
  S.out("otsu/fig-tight.svg");
}

{ // the worked example: within + between = total at every t
  const h = [2, 5, 3, 1, 1, 2, 4, 2], o = TH.otsu(h);
  const S = new Svg(520, 380), p = new Panel(S, 80, 50, 420, 250, [-0.6, 6.6], [0, 6.5]);
  p.axes([0, 1, 2, 3, 4, 5, 6], [0, 2, 4, 6], "threshold t", "variance");
  for (let t = 0; t <= 6; t++) {
    const st = TH.splitStats(h, t);
    p.rect(t - 0.36, t + 0.36, 0, st.varW, C.c1, 0.8);
    p.rect(t - 0.36, t + 0.36, st.varW, st.varW + st.varB, C.c4, t === o.t ? 1 : 0.55);
  }
  p.hline(TH.splitStats(h, 3).varT, C.fg, "", 1.5, "6 4");
  S.text(80, 22, "■", C.c1, "start", 20); S.text(102, 22, "within", C.fg, "start", 19);
  S.text(200, 22, "■", C.c4, "start", 20); S.text(222, 22, "between", C.fg, "start", 19);
  p.text(3, 6.05, "t* = 3", C.fg, "middle", 19);
  S.text(500, 44, "total 5.81", C.fg, "end", 17);
  console.log("worked example otsu t", o.t);
  S.out("otsu/fig-stack.svg");
}

{ // isodata on the worked example (converges to Otsu's split) and on three groups (stuck)
  const S = new Svg(600, 600);
  ([[[2, 5, 3, 1, 1, 2, 4, 2], 5.5, "the worked example"], [[9, 0, 0, 8, 7, 0, 0, 5], 6.5, "three groups"]] as const).forEach(([hh, t0, title], k) => {
    const h = Array.from(hh), iso = TH.isodata(h, t0), o = TH.otsu(h);
    const tr = iso.trace.filter((v, j) => j === 0 || Math.abs(v - iso.trace[j - 1]) > 1e-9);
    console.log(title, "isodata trace", iso.trace.map(f2).join(" → "), "otsu t", o.t);
    const ymax = Math.max(...h) * 1.75, p = new Panel(S, 80, 50 + k * 290, 490, 170, [-0.6, 7.6], [0, ymax]);
    p.axes([0, 1, 2, 3, 4, 5, 6, 7], [], k ? "grey level i" : "", "count");
    const fin = Math.floor(iso.value);
    h.forEach((c, i) => p.bars([i - 0.38, i + 0.38], [c], i <= fin ? C.c1 : C.c4, 0.85));
    const base = Math.max(...h) * 1.12, step = (ymax - base) / (tr.length + 0.3);
    tr.forEach((v, j) => {
      const y = ymax - step * (j + 0.6);
      if (j) p.arrow(tr[j - 1], y + step, v, y, C.fg, 2);
      p.dot(v, y, j === tr.length - 1 ? C.c2 : C.fg, 6);
    });
    p.text(tr[0] + 0.2, ymax - step * 0.6 - step * 0.15, `start ${t0}`, C.fg, "start", 18);
    p.text(tr[tr.length - 1] - 0.2, ymax - step * (tr.length - 0.4) - step * 0.15, `stops at ${f2(tr[tr.length - 1])}`, C.c2, "end", 18);
    if (k) { p.vline(o.t + 0.5, C.c5, "", 2.5, "7 4"); p.text(o.t + 0.6, ymax * 0.5, "Otsu's best split", C.c5, "start", 18); }
    S.text(80, 32 + k * 290, title, C.fg, "start", 19, 'font-weight="bold"');
  });
  S.out("otsu/fig-isodata.svg");
}

{ // posterior probability of "object" for the 90 % / 10 % example and for equal weights
  const [m0, m1, sd] = [50, 150, 20], xs = Array.from({ length: 251 }, (_, i) => i);
  const post = (w1: number) => xs.map(x => { const a = w1 * TH.normalPdf(x, m1, sd), b = (1 - w1) * TH.normalPdf(x, m0, sd); return a / (a + b); });
  const t10 = TH.gaussianCrossings(m0, sd, m1, sd, 0.1)[0];
  const S = new Svg(560, 360), p = new Panel(S, 90, 20, 440, 250, [0, 250], [0, 1]);
  p.axes([0, 50, 100, 150, 200, 250], [0, 0.5, 1], "grey level x", "P(object | x)");
  p.hline(0.5, C.muted, "", 1.5, "5 4");
  p.line(xs, post(0.5), C.muted, 3, "9 5");
  p.line(xs, post(0.1), C.c4, 3.5);
  p.dot(100, 0.5, C.muted, 7); p.dot(t10, 0.5, C.c4, 8);
  p.text(t10 + 6, 0.38, `10 % objects: ${t10.toFixed(1)}`, C.c4, "start", 19);
  p.text(96, 0.62, "50 %: 100", C.muted, "end", 19);
  p.text(6, 0.08, "background", C.c1, "start", 19); p.text(244, 0.9, "object", C.c4, "end", 19);
  S.out("otsu/fig-posterior.svg");
}

{ // Kittler–Illingworth: the two fitted normal curves at a bad split and at the best split
  const P = { mu0: 80, mu1: 150, s0: 12, s1: 40, w1: 0.4, n: 128 };
  const M = mixtureModel(P), h = M.hist, ki = TH.kittler(h), o = TH.otsu(h), N = Array.from(h).reduce((a, b) => a + b, 0);
  console.log("KI preset 'unequal variances': kittler", ki.t, "otsu", o.t, "J(ki)", f2(ki.curve[ki.t]), "J(otsu)", f2(ki.curve[o.t]));
  const S = new Svg(600, 600), mx = Math.max(...Array.from(h));
  [o.t, ki.t].forEach((t, k) => {
    const st = TH.splitStats(h, t), s0 = Math.sqrt(st.var0), s1 = Math.sqrt(st.var1);
    const p = new Panel(S, 80, 50 + k * 290, 490, 170, [0, 255], [0, mx * 1.15]);
    p.axes([0, 50, 100, 150, 200, 250], [], k ? "grey level" : "", "pixels");
    p.bars(edges256, h, "#aab4bd", 0.7);
    p.line(xs256, xs256.map(x => N * st.w0 * TH.normalPdf(x, st.mu0, s0)), C.c1, 3);
    p.line(xs256, xs256.map(x => N * st.w1 * TH.normalPdf(x, st.mu1, s1)), C.c4, 3);
    p.vline(t + 0.5, C.fg, "", 2.5);
    S.text(80, 32 + k * 290, `${k ? "best fit (minimum error)" : "Otsu's split"}: t = ${t}, J = ${f2(ki.curve[t])}`, C.fg, "start", 19, 'font-weight="bold"');
  });
  S.out("otsu/fig-kifit.svg");
}

{ // bias: Otsu on the two-Gaussian model, 10 % and 3 % objects, against the crossing
  const [m0, m1, sd] = [50, 150, 20];
  const S = new Svg(600, 600);
  [0.1, 0.03].forEach((w1, k) => {
        const di = xs256.map(x => (1 - w1) * TH.normalPdf(x, m0, sd) + w1 * TH.normalPdf(x, m1, sd)), o = TH.otsu(di), tO = o.t + 0.5, tB = TH.gaussianCrossings(m0, sd, m1, sd, w1)[0];
    console.log(`w1=${w1}: otsu ${tO.toFixed(1)} crossing ${tB.toFixed(1)} err otsu ${(100 * TH.thresholdError(tO, m0, sd, m1, sd, w1)).toFixed(2)}% err crossing ${(100 * TH.thresholdError(tB, m0, sd, m1, sd, w1)).toFixed(2)}%`);
    const xs = xs256.slice(0, 251), f0 = xs.map(x => Math.min(0.0075, (1 - w1) * TH.normalPdf(x, m0, sd))), f1 = xs.map(x => w1 * TH.normalPdf(x, m1, sd));
    const p = new Panel(S, 80, 50 + k * 290, 490, 170, [0, 250], [0, 0.0075]);
    p.axes([0, 50, 100, 150, 200, 250], [], k ? "grey level x" : "", "ω f(x)");
    p.area(xs, f0, C.c1, 0.25); p.area(xs, f1, C.c4, 0.35);
    p.line(xs, f0, C.c1, 2.5); p.line(xs, f1, C.c4, 2.5);
    p.vline(tO, C.c5, "", 3, "8 5"); p.vline(tB, C.c2, "", 3);
    const halo = 'stroke="#fff" stroke-width="5" paint-order="stroke" font-weight="bold"';
    S.text(p.X(tO) + (k ? 6 : -6), p.Y(0.0062), `Otsu ${tO.toFixed(1)}`, C.c5, k ? "start" : "end", 19, halo);
    S.text(p.X(tB) + 6, p.Y(k ? 0.0045 : 0.0062), `crossing ${tB.toFixed(1)}`, C.c2, "start", 19, halo);
    S.text(80, 32 + k * 290, `${Math.round(w1 * 100)} % objects (background clipped)`, C.fg, "start", 19, 'font-weight="bold"');
  });
  S.out("otsu/fig-bias.svg");
}

{ // triangle: a unimodal histogram with a long tail (mixture preset "small object (3 %)")
  const M = mixtureModel({ mu0: 60, mu1: 100, s0: 12, s1: 20, w1: 0.03, n: 128 }), h = M.hist, tr = TH.triangle(h), o = TH.otsu(h);
  console.log("triangle t", tr.t, "peak", tr.peak, "end", tr.end, "otsu", o.t);
  const hp = h[tr.peak], S = new Svg(560, 360), p = new Panel(S, 90, 20, 440, 250, [0, 200], [0, hp * 1.08]);
  p.axes([0, 50, 100, 150, 200], [], "grey level", "pixels");
  p.bars(edges256, h, C.c1, 0.7);
  p.line([tr.peak, tr.end], [hp, 0], C.fg, 2.5);
  // perpendicular foot (in the scaled coordinates used by the method)
  const sx = 1 / Math.abs(tr.end - tr.peak), sy = 1 / hp, ax = (tr.end - tr.peak) * sx, ay = -hp * sy, L2 = ax * ax + ay * ay;
  const px = (tr.t - tr.peak) * sx, py = (h[tr.t] - hp) * sy, u = (px * ax + py * ay) / L2;
  const fx = tr.peak + u * ax / sx, fy = hp + u * ay / sy;
  p.line([tr.t, fx], [h[tr.t], fy], C.c2, 3);
  p.vline(tr.t + 0.5, C.c2, "", 2, "6 4");
  p.text(tr.t + 4, hp * 0.75, `triangle t = ${tr.t}`, C.c2, "start", 19);
  p.vline(o.t + 0.5, C.c5, "", 2, "6 4");
  p.text(4, hp * 0.92, `Otsu ${o.t}`, C.c5, "start", 19);
  p.text(198, hp * 0.4, "3 % objects:", C.fg, "end", 19); p.text(198, hp * 0.3, "a tail, no peak", C.fg, "end", 19);
  S.out("otsu/fig-triangle.svg");
}
console.log("otsu figures written");
