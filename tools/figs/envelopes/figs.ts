// Figures for envelopes/ (SVG). Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/envelopes/figs.ts --bundle --platform=node --log-level=warning | node
import { svg, text, line, rect, circle, path, out, C, patternPanel, curvePanel } from "../point-patterns/kit";
import * as PP from "../../../demos/src/lib/pointpattern";
import { rng } from "../../../demos/src/lib/random";
import { makePattern, DEFAULT_SPEC, makeWindow, Spec } from "../../../demos/src/csrplay/source";
import { RADII, curve, reference, nullCurves, PRECOMP_M, PRECOMP_PW39, PRECOMP_PW19, PRECOMP_MAD, DEFAULT_COND, simulateConditions, conditionTests } from "../../../demos/src/envelopetest/model";

const D = "envelopes/";
const sq = makeWindow("square");
const pat = (s: Partial<Spec>) => makePattern({ ...DEFAULT_SPEC, ...s });
const weakSpec: Partial<Spec> = { model: "thomas", k: 75, mu: 2, sigma: 3, seed: 1 };
const weak = pat(weakSpec);
const idxAll = PP.rangeIdx(RADII, [1, 50]);

// ---- null histogram at one r
{
  const sims = nullCurves(weak.x.length, sq, 99, "L", 31 + 5), obs = curve(weak, sq, "L"), k = 5;
  const vals = sims.map(c => c[k]), o = obs[k], rk = PP.mcRank(o, vals);
  const lo = -3, hi = 7, bw = 0.5, nb = (hi - lo) / bw, cnt = new Array(nb).fill(0);
  for (const v of vals) cnt[Math.min(nb - 1, Math.max(0, Math.floor((v - lo) / bw)))]++;
  let b = curvePanel(90, 20, 560, 300, [lo, hi], [0, 50], [-2, 0, 2, 4, 6], [0, 10, 20, 30, 40, 50], "L(5) − 5  (µm)", "simulations", []);
  const X = (v: number) => 90 + (v - lo) / (hi - lo) * 560, Y = (c: number) => 320 - c / 50 * 300;
  cnt.forEach((c, i) => { if (c) b += rect(X(lo + i * bw) + 1, Y(c), 560 / nb - 2, 320 - Y(c), { fill: C.grey, opacity: 0.8 }); });
  b += line(X(o), 20, X(o), 320, C.fp, 4) + text(X(o) - 8, 50, `observed ${o.toFixed(2)}`, { fill: C.fp, anchor: "end", weight: "bold" });
  b += text(370, 395, "grey: 99 CSR simulations (same n, same window)", { size: 19 }) + text(370, 420, `rank of the observed: ${rk.rank} of 100, p = ${rk.pUpper.toFixed(2)}`, { size: 19 });
  out(D + "fig-null-hist.svg", svg(700, 440, b));
  console.log("null hist: obs", o.toFixed(3), "rank", rk.rank, "p upper", rk.pUpper, "n", weak.x.length, "sim range", Math.min(...vals).toFixed(2), Math.max(...vals).toFixed(2));
}

// ---- pointwise envelope of the weak cluster pattern
{
  const sims = nullCurves(weak.x.length, sq, 39, "L", 31 + 5), obs = curve(weak, sq, "L"), env = PP.pointwiseEnvelope(sims);
  let b = patternPanel(30, 40, 260, sq, weak, { r: 2.6, title: `weak clusters, n = ${weak.x.length}` });
  b += curvePanel(390, 20, 480, 290, [0, 50], [-4, 8], [0, 10, 20, 30, 40, 50], [-4, 0, 4, 8], "r (µm)", "L(r) − r  (µm)",
    [{ xs: RADII, ys: RADII.map(() => 0), color: "#000", dash: "7 5", w: 2 }, { xs: RADII, ys: obs, color: C.blue, label: "observed" }],
    { bands: [{ xs: RADII, lo: env.lo, hi: env.hi, color: C.grey, opacity: 0.35, label: "min–max of 39 CSR" }], legendAt: [870, 25] });
  out(D + "fig-pointwise.svg", svg(900, 380, b));
  const esc = PP.escapes(obs, env, idxAll);
  console.log("pointwise: escapes", esc.length, RADII[esc[0]], RADII[esc[esc.length - 1]]);
}

// ---- 20 null patterns: which escape the pointwise envelope?
{
  const r = rng(2024), n = 150;
  let b = "", nEsc = 0, nMad = 0;
  for (let t = 0; t < 20; t++) {
    const obs = curve(PP.binomialProcess(r, n, sq), sq, "L"), sims = Array.from({ length: 39 }, () => curve(PP.binomialProcess(r, n, sq), sq, "L"));
    const env = PP.pointwiseEnvelope(sims), esc = PP.escapes(obs, env, idxAll), mad = PP.madTest(obs, sims, reference("L"), idxAll);
    if (esc.length) nEsc++;
    if (mad.p <= 0.05) nMad++;
    const x0 = 20 + (t % 5) * 210, y0 = 20 + Math.floor(t / 5) * 150, w = 190, h = 120;
    const X = (v: number) => x0 + v / 50 * w, Y = (v: number) => y0 + h / 2 - v / 4 * (h / 2);
    const up = idxAll.map(k => [X(RADII[k]), Y(env.hi[k])] as [number, number]), dn = idxAll.map(k => [X(RADII[k]), Y(env.lo[k])] as [number, number]).reverse();
    b += rect(x0, y0, w, h, { fill: esc.length ? "#fde8e4" : "#fff", stroke: "#999", w: 1 }) + path([...up, ...dn], { fill: C.grey, opacity: 0.35, stroke: "none", w: 0 });
    b += line(x0, Y(0), x0 + w, Y(0), "#000", 1, "4 3") + path(idxAll.map(k => [X(RADII[k]), Math.max(y0, Math.min(y0 + h, Y(obs[k])))] as [number, number]), { stroke: esc.length ? C.fp : C.blue, w: 2.2 });
    for (const k of esc) b += circle(X(RADII[k]), Math.max(y0, Math.min(y0 + h, Y(obs[k]))), 3, { fill: C.fp });
  }
  b += text(530, 635, `20 random (CSR) patterns, each against its own 39 simulations (L − r, 50 radii):`, { size: 19 }) + text(530, 662, `${nEsc} leave the pointwise band somewhere (red); the MAD test rejects ${nMad}`, { size: 19 });
  out(D + "fig-escapes.svg", svg(1070, 680, b));
  console.log("escapes:", nEsc, "of 20; MAD rejects", nMad);
}

// ---- rejection rate against the number of radii
{
  const b = curvePanel(90, 20, 520, 300, [0, 52], [0, 0.8], [0, 10, 20, 30, 40, 50], [0, 0.05, 0.2, 0.4, 0.6, 0.8], "number of radii tested", "share of CSR patterns rejected",
    [{ xs: PRECOMP_M, ys: PRECOMP_PW19, color: C.orange, label: "pointwise, 19 sims" }, { xs: PRECOMP_M, ys: PRECOMP_PW39, color: C.fp, label: "pointwise, 39 sims" }, { xs: PRECOMP_M, ys: PRECOMP_MAD, color: C.tp, label: "MAD test, 39 sims" }],
    { hlines: [{ y: 0.05 }], legendAt: [605, 200] });
  out(D + "fig-rate.svg", svg(640, 400, b));
}

// ---- MAD: global vs pointwise envelope on the trap pattern
{
  const p = pat({ model: "csr", seed: 6 }), sims = nullCurves(p.x.length, sq, 199, "L", 31 * 6 + 5).slice(0, 39), obs = curve(p, sq, "L");
  const env = PP.pointwiseEnvelope(sims), mad = PP.madTest(obs, sims, reference("L"), idxAll), esc = PP.escapes(obs, env, idxAll);
  let kmax = idxAll[0];
  for (const k of idxAll) if (Math.abs(obs[k]) > Math.abs(obs[kmax])) kmax = k;
  let b = curvePanel(90, 20, 620, 330, [0, 50], [-4, 4], [0, 10, 20, 30, 40, 50], [-4, -2, 0, 2, 4], "r (µm)", "L(r) − r  (µm)",
    [{ xs: RADII, ys: RADII.map(() => 0), color: "#000", dash: "7 5", w: 2 }, { xs: RADII, ys: obs, color: C.blue, label: "observed (a CSR pattern)" }],
    { bands: [{ xs: RADII, lo: mad.lo, hi: mad.hi, color: C.tp, opacity: 0.18, label: `global (MAD), ±${mad.Tcrit.toFixed(2)}` }, { xs: RADII, lo: env.lo, hi: env.hi, color: C.grey, opacity: 0.4, label: "pointwise, 39 sims" }], legendAt: [420, 25] });
  const X = (v: number) => 90 + v / 50 * 620, Y = (v: number) => 20 + (4 - v) / 8 * 330;
  for (const k of esc) b += circle(X(RADII[k]), Y(obs[k]), 4.5, { fill: C.fp });
  b += line(X(RADII[kmax]), Y(0), X(RADII[kmax]), Y(obs[kmax]), C.fp, 3) + text(X(RADII[kmax]) - 8, Y(obs[kmax] / 2), `T = ${mad.T0.toFixed(2)}`, { anchor: "end", fill: C.fp, weight: "bold" });
  out(D + "fig-mad.svg", svg(740, 430, b));
  console.log("mad fig: T", mad.T0.toFixed(2), "Tcrit", mad.Tcrit.toFixed(2), "rank", mad.rank, "p", mad.p, "escapes", esc.length);
}

// ---- SuperPlot of per-image summaries
{
  const o = { ...DEFAULT_COND }, { images } = simulateConditions(o), t = conditionTests(images, o.experiments);
  const vals = images.map(s => s.value), lo = Math.floor(Math.min(...vals)) - 1, hi = Math.ceil(Math.max(...vals)) + 1;
  let b = curvePanel(110, 20, 420, 330, [-0.6, 1.6], [lo, hi], [], [0, 2, 4, 6, 8, 10].filter(v => v >= lo && v <= hi), "", "L(10) − 10 per image (µm)", []);
  const X = (v: number) => 110 + (v + 0.6) / 2.2 * 420, Y = (v: number) => 20 + (hi - v) / (hi - lo) * 330, cols = [C.blue, C.fp, C.tp, C.orange];
  images.forEach((s, i) => { const j = ((i * 0.6180339) % 1 - 0.5) * 0.35; b += circle(X(s.cond + j), Y(s.value), 6, { fill: cols[s.exp], stroke: "#fff", w: 1 }); });
  for (const c of [0, 1]) t.means[c].forEach((m, e) => { b += circle(X(c + (e - 1.5) * 0.08), Y(m), 13, { fill: cols[e], stroke: "#000", w: 2 }); });
  b += text(X(0), 385, "control", { size: 20 }) + text(X(1), 385, "treated", { size: 20 });
  b += text(560, 80, "small dots: 3 images per dish", { anchor: "start", size: 19 }) + text(560, 108, "big dots: 4 experiment means", { anchor: "start", size: 19 });
  b += text(560, 170, `images as n (12 vs 12): p = ${t.imagesP.toFixed(3)}`, { anchor: "start", size: 19, fill: C.fp });
  b += text(560, 198, `experiments as n (4 vs 4): p = ${t.expP.toFixed(2)}`, { anchor: "start", size: 19, fill: C.tp });
  b += text(560, 250, "no true treatment effect", { anchor: "start", size: 19, weight: "bold" });
  out(D + "fig-superplot.svg", svg(900, 400, b));
  console.log("superplot p images", t.imagesP.toFixed(4), "exp", t.expP.toFixed(4));
}
