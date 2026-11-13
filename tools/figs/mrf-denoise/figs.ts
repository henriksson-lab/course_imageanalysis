// Static figures for mrf-denoise/lec.qmd. Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/mrf-denoise/figs.ts --bundle --platform=node --log-level=warning | node
// Uses the mrfdenoise demo's own model (same image, noise seed and methods), so the numbers
// match the demo and the slide text.
import { Svg, Panel, C, fmt, toRGB } from "../ising/svglib";
import { N, shapesTruth, makeProblem, Run, wrong, energyOf, DEFAULT_RUN, Noise, Problem } from "../../../demos/src/mrfdenoise/model";

const flip: Noise = { kind: "flip", eps: 0.2, sd: 0.6, seed: 1 };
const truth = shapesTruth();
/** Error colours of the demo: grey correct object, white correct background, red false object, blue missed object. */
const errRGB = (P: Problem, s: ArrayLike<number>) => toRGB(N * N, p => {
  const e = s[p] > 0 ? 1 : 0, t = P.truth[p];
  if (e === t) return t ? [150, 150, 150] : [250, 250, 250];
  return t ? [31, 119, 180] : [214, 39, 40];
});
const binRGB = (s: ArrayLike<number>) => toRGB(N * N, p => (s[p] > 0 ? [30, 30, 30] : [250, 250, 250]));
const legend = (s: Svg, x: number, y: number) => {
  const items: [string, string][] = [["#969696", "object, correct"], [C.red, "false object"], [C.blue, "missed object"]];
  items.forEach(([c, l], i) => { const xx = x + i * 205; s.rect(xx, y - 15, 18, 18, c); s.text(xx + 26, y, l, { size: 19 }); });
};

// ---------------------------------------------------------------------------------------
// fig-mpm.svg: lambda = 1.2: exact MAP vs MPM (Gibbs at T = 1, 300 sweeps) and the marginal p(object).
{
  const P = makeProblem(truth, flip, 1.2);
  const g = new Run("gibbs", P, DEFAULT_RUN); g.finish();
  const wMap = wrong(P, P.map), wMpm = wrong(P, g.estimate);
  const s = new Svg(540, 580);
  const D = 250;
  s.image(10, 10, D, D, errRGB(P, P.map), N, N);
  s.image(280, 10, D, D, errRGB(P, g.estimate), N, N);
  const mg = g.marginal!;
  s.image(10, 320, D, D, toRGB(N * N, p => { const v = Math.round(250 - 220 * mg[p]); return [v, v, v]; }), N, N);
  s.text(135, 292, `MAP: ${wMap} wrong`, { anchor: "middle", size: 22 });
  s.text(405, 292, `MPM: ${wMpm} wrong`, { anchor: "middle", size: 22 });
  const items: [string, string][] = [["#969696", "object, correct"], [C.red, "false object"], [C.blue, "missed object"]];
  items.forEach(([c, l], i) => { s.rect(285, 345 + i * 36, 20, 20, c); s.text(315, 362 + i * 36, l, { size: 21 }); });
  s.text(285, 500, "left: p(object)", { size: 21 });
  s.text(285, 528, "white 0, black 1", { size: 21, color: C.axis });
  s.out("mrf-denoise/fig-mpm.svg");
  console.log("mpm lambda 1.2: MAP wrong", wMap, "MPM wrong", wMpm);
}

// ---------------------------------------------------------------------------------------
// fig-energy-curves.svg: lambda = 0.8: energy against sweeps for ICM, slow annealing, quench; exact min-cut energy.
{
  const P = makeProblem(truth, flip, 0.8);
  const icm = new Run("icm", P); icm.finish();
  const slow = new Run("anneal", P, { ...DEFAULT_RUN, coolSweeps: 1000 }); slow.finish();
  const quench = new Run("anneal", P, { ...DEFAULT_RUN, coolSweeps: 10 }); quench.finish();
  const s = new Svg(760, 400);
  const Pn = new Panel(s, 100, 20, 390, 290, [1, 2000], [5100, 5800], true);
  Pn.axes([1, 10, 100, 1000], [5100, 5300, 5500, 5700], "sweeps (log scale)", "");
  s.add(`<text transform="rotate(-90 22 165)" x="22" y="165" text-anchor="middle" font-size="18" fill="${C.fg}">posterior energy</text>`);
  const curve = (r: Run, col: string, dash = "") => {
    const ks: number[] = [], es: number[] = [];
    r.hist.k.forEach((k, i) => { if (k >= 1) { ks.push(k); es.push(Math.min(5800, r.hist.E[i])); } });
    Pn.line(ks, es, col, 3, dash);
  };
  Pn.hline(P.Emin, C.fg, 2, "6 4");
  curve(quench, C.purple, "8 4"); curve(slow, C.orange); curve(icm, C.blue);
  const rows: [string, string, Run | null][] = [["ICM", C.blue, icm], ["quench (10)", C.purple, quench], ["slow (1000)", "#c05f00", slow], ["exact (min cut)", C.fg, null]];
  rows.forEach(([l, c, r], i) => {
    const y = 60 + i * 52;
    s.text(505, y, l, { size: 19, color: c });
    const E = r ? r.energy : P.Emin;
    s.text(505, y + 24, `${E.toFixed(1)}, ${wrong(P, r ? r.estimate : P.map)} wrong`, { size: 17, color: C.axis });
  });
  s.out("mrf-denoise/fig-energy-curves.svg");
  console.log("energies: icm", icm.energy, icm.sweeps, wrong(P, icm.estimate), "slow", slow.energy, wrong(P, slow.estimate), "quench", quench.energy, "exact", P.Emin, wrong(P, P.map));
}

// ---------------------------------------------------------------------------------------
// fig-notbest.svg: lambda = 1.6: exact MAP, ICM, mean field (errors against the truth).
{
  const P = makeProblem(truth, flip, 1.6);
  const icm = new Run("icm", P); icm.finish();
  const mf = new Run("meanfield", P); mf.finish();
  const s = new Svg(830, 340);
  const D = 250;
  const cols: [string, ArrayLike<number>][] = [["exact MAP", P.map], ["ICM", icm.estimate], ["mean field", mf.estimate]];
  cols.forEach(([l, st], i) => {
    s.image(10 + 280 * i, 10, D, D, errRGB(P, st), N, N);
    s.text(135 + 280 * i, 290, `${l}: ${wrong(P, st)} wrong`, { anchor: "middle", size: 21 });
    console.log("notbest", l, wrong(P, st), energyOf(P, st).total.toFixed(1));
  });
  legend(s, 20, 330);
  s.out("mrf-denoise/fig-notbest.svg");
}

// ---------------------------------------------------------------------------------------
// fig-lambda.svg: the exact MAP for three values of lambda.
{
  const s = new Svg(830, 300);
  const D = 250;
  [0.3, 0.8, 3].forEach((lam, i) => {
    const P = makeProblem(truth, flip, lam);
    s.image(10 + 280 * i, 10, D, D, binRGB(P.map), N, N);
    s.text(135 + 280 * i, 292, `λ = ${lam}: ${wrong(P, P.map)} wrong`, { anchor: "middle", size: 21 });
    console.log("lambda", lam, wrong(P, P.map));
  });
  s.out("mrf-denoise/fig-lambda.svg");
}

// ---------------------------------------------------------------------------------------
// fig-icm-order.svg: ICM on y = (1, 0, 1), eps = 0.2, lambda = 2, in two visiting orders.
{
  const c = Math.log(4), lam = 2, y = [1, 0, 1];
  const E = (L: number[]) => L.reduce((a, l, i) => a + (l !== y[i] ? c : 0), 0) + lam * ((L[0] !== L[1] ? 1 : 0) + (L[1] !== L[2] ? 1 : 0));
  const icmOrder = (order: number[]) => {
    const L = y.slice(), states: { L: number[]; at: number }[] = [{ L: L.slice(), at: -1 }];
    for (const p of order) {
      const cost = (v: number) => (v !== y[p] ? c : 0) + lam * ((p > 0 && L[p - 1] !== v ? 1 : 0) + (p < 2 && L[p + 1] !== v ? 1 : 0));
      L[p] = cost(1) < cost(0) ? 1 : cost(0) < cost(1) ? 0 : L[p];
      states.push({ L: L.slice(), at: p });
    }
    return states;
  };
  const s = new Svg(800, 290);
  const cell = 34;
  const rowsSpec: [string, number[], number][] = [["left to right", [0, 1, 2], 50], ["middle first", [1, 0, 2], 190]];
  for (const [lab, order, y0] of rowsSpec) {
    const st = icmOrder(order);
    s.text(10, y0 - 14, lab, { size: 20, weight: "bold" });
    st.forEach((q, k) => {
      const x0 = 10 + k * 190;
      q.L.forEach((v, i) => {
        s.rect(x0 + i * (cell + 4), y0, cell, cell, v ? "#1e1e1e" : "#fafafa", { stroke: i === q.at ? C.orange : "#999", sw: i === q.at ? 4 : 1.5 });
      });
      s.text(x0 + 1.5 * cell + 4, y0 + cell + 26, k === 0 ? `y, E = ${fmt(E(q.L))}` : `E = ${fmt(E(q.L))}`, { anchor: "middle", size: 18, color: C.axis });
      if (k > 0) s.arrow(x0 - 70, y0 + cell / 2, x0 - 14, y0 + cell / 2, C.axis, 2);
    });
    const fin = st[st.length - 1].L;
    console.log("icm", lab, fin.join(""), E(fin).toFixed(2));
  }
  s.text(10, 285, "orange frame: the pixel just visited. Black = 1, white = 0.", { size: 17, color: C.axis });
  s.out("mrf-denoise/fig-icm-order.svg");
}

// ---------------------------------------------------------------------------------------
// fig-mincut3.svg: the three pixels y = (1, 0, 1) as an s-t graph; the minimum cut gives 111.
{
  const c = Math.log(4), lam = 2;
  const s = new Svg(470, 360);
  const px = [70, 220, 370], py = 175, S = [220, 35], T = [220, 320];
  // source side = label 1: the edge source -> p costs D_p(0) (paid if p ends on the 0 side), p -> sink costs D_p(1).
  const y = [1, 0, 1];
  px.forEach((x, i) => {
    const d0 = y[i] ? c : 0, d1 = y[i] ? 0 : c;
    if (d0 > 0) { s.line(S[0], S[1] + 24, x, py - 28, C.orange, 3.5); s.text((S[0] + x) / 2 + (i === 0 ? -62 : 14), (S[1] + py) / 2, fmt(d0), { size: 22, color: "#c05f00" }); }
    if (d1 > 0) { s.line(x, py + 28, T[0], T[1] - 24, C.blue, 3.5, "8 5"); s.text(x + 12, py + 80, fmt(d1), { size: 22, color: C.blue }); }
  });
  for (let i = 0; i < 2; i++) { s.line(px[i] + 28, py, px[i + 1] - 28, py, C.fg, 3.5); s.text((px[i] + px[i + 1]) / 2, py - 12, `λ = ${lam}`, { anchor: "middle", size: 21 }); }
  px.forEach((x, i) => { s.circle(x, py, 28, y[i] ? "#1e1e1e" : "#fafafa", C.fg, 2.5); s.text(x, py + 8, `${y[i]}`, { anchor: "middle", size: 22, color: y[i] ? "#fff" : C.fg, weight: "bold" }); });
  s.circle(S[0], S[1], 24, C.orange); s.text(S[0], S[1] + 8, "1", { anchor: "middle", size: 22, color: "#fff", weight: "bold" });
  s.circle(T[0], T[1], 24, C.blue); s.text(T[0], T[1] + 8, "0", { anchor: "middle", size: 22, color: "#fff", weight: "bold" });
  s.add(`<path d="M30,228 Q220,250 410,228" fill="none" stroke="${C.red}" stroke-width="3.5" stroke-dasharray="3 6"/>`);
  s.text(290, 272, `min cut ${fmt(c)}`, { size: 21, color: C.red });
  s.text(290, 298, "→ labels 111", { size: 21, color: C.red });
  s.text(10, 40, "source = label 1", { size: 20, color: "#c05f00" });
  s.text(10, 326, "sink = label 0", { size: 20, color: C.blue });
  s.out("mrf-denoise/fig-mincut3.svg");
}
