// Static figures for gmrf/lec.qmd. Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/gmrf/figs.ts --bundle --platform=node --log-level=warning | node
// Uses lib/smooth.ts and the mrf1d demo's model (same profile, noise seed and solvers).
import { Svg, Panel, C, fmt } from "../ising/svglib";
import { gmrf1d, tv1d, huber, gmrfResponse, denoise1d, samplePrior1d } from "../../../demos/src/lib/smooth";
import { mixedProfile, noisy } from "../../../demos/src/mrf1d/model";
import { rng } from "../../../demos/src/lib/random";

const OR = "#c05f00";   // orange for text

// ---------------------------------------------------------------------------------------
// fig-penalties.svg: rho(d) for quadratic, TV and Huber (delta = 1).
{
  const s = new Svg(470, 380);
  const P = new Panel(s, 80, 20, 370, 280, [-3, 3], [0, 4.5]);
  P.axes([-3, -2, -1, 0, 1, 2, 3], [0, 1, 2, 3, 4], "difference d = x_p − x_q", "penalty ρ(d)");
  const ds: number[] = []; for (let k = 0; k <= 240; k++) ds.push(-3 + 6 * k / 240);
  P.line(ds, ds.map(d => Math.min(4.5, d * d / 2)), C.blue, 3.5);
  P.line(ds, ds.map(d => Math.abs(d)), C.orange, 3.5, "9 5");
  P.line(ds, ds.map(d => huber(d, 1)), C.green, 3.5, "3 4");
  P.text(-0.95, 4.1, "quadratic d²/2", { size: 19, color: C.blue });
  P.text(-0.95, 3.6, "TV |d|", { size: 19, color: OR });
  P.text(-0.95, 3.1, "Huber, δ = 1", { size: 19, color: "#1e7a1e" });
  s.out("gmrf/fig-penalties.svg");
}

/** Stem plot of small vectors: data (grey squares) and estimates (coloured dots joined by lines). */
function stems(path: string, y: number[], ests: [string, Float64Array, string][], ymax: number, W = 520) {
  const s = new Svg(W, 380);
  const n = y.length;
  const P = new Panel(s, 70, 20, W - 100, 260, [0.5, n + 0.5], [0, ymax]);
  P.axes(y.map((_, i) => i + 1), Array.from({ length: Math.floor(ymax) + 1 }, (_, i) => i), "pixel", "value");
  y.forEach((v, i) => { s.rect(P.X(i + 1) - 9, P.Y(v) - 9, 18, 18, C.grey); });
  ests.forEach(([, x, col], k) => {
    P.line(x.map((_, i) => i + 1), x, col, 3, k ? "8 5" : "");
    const right = col === C.blue;
    x.forEach((v, i) => { P.dot(i + 1, v, col, 7); P.text(i + 1 + (right ? 0.12 : -0.12), v + (right ? -0.25 : 0.1), fmt(v), { size: 18, anchor: right ? "start" : "end", color: col === C.orange ? OR : col }); });
  });
  // legend
  let lx = 80;
  s.rect(lx, 352, 16, 16, C.grey); s.text(lx + 24, 367, "data y", { size: 19 }); lx += 120;
  for (const [l, , col] of ests) { s.circle(lx + 8, 360, 7, col); s.text(lx + 22, 367, l, { size: 19, color: col === C.orange ? OR : col }); lx += 60 + 11 * l.length; }
  s.out(path);
}

// ---------------------------------------------------------------------------------------
// fig-hand-quad.svg: y = (0, 3, 0), quadratic, lambda = 1 -> (0.75, 1.5, 0.75).
{
  const x = gmrf1d([0, 3, 0], 1);
  stems("gmrf/fig-hand-quad.svg", [0, 3, 0], [["quadratic, λ = 1", x, C.blue]], 3.3, 460);
  console.log("hand quad", Array.from(x));
}

// ---------------------------------------------------------------------------------------
// fig-hand-tv.svg: y = (0, 0, 3, 3), lambda = 1: TV (0.5, 0.5, 2.5, 2.5) vs quadratic (ramp).
{
  const t = tv1d([0, 0, 3, 3], 1), q = gmrf1d([0, 0, 3, 3], 1);
  stems("gmrf/fig-hand-tv.svg", [0, 0, 3, 3], [["TV", t, C.orange], ["quadratic", q, C.blue]], 3.3, 520);
  console.log("hand tv", Array.from(t), "quad", Array.from(q));
}

// ---------------------------------------------------------------------------------------
// fig-gain.svg: gain of quadratic smoothing against frequency, lambda = 1 and 10.
{
  const s = new Svg(580, 380);
  const P = new Panel(s, 80, 20, 470, 280, [0, 0.5], [0, 1]);
  P.axes([0, 0.1, 0.2, 0.3, 0.4, 0.5], [0, 0.25, 0.5, 0.75, 1], "frequency (cycles per sample)", "gain");
  const fs: number[] = []; for (let k = 0; k <= 250; k++) fs.push(0.5 * k / 250);
  for (const [lam, col, dash] of [[1, C.blue, ""], [10, C.orange, "9 5"]] as [number, string, string][]) {
    P.line(fs, fs.map(f => gmrfResponse(2 * Math.PI * f, lam)), col, 3.5, dash);
    const g = gmrfResponse(Math.PI, lam);
    P.dot(0.5, g, col);
    P.text(0.49, g + 0.05, `1/${Math.round(1 / g)}`, { anchor: "end", size: 19, color: col === C.orange ? OR : col });
  }
  const g05 = gmrfResponse(2 * Math.PI * 0.05, 10);
  P.dot(0.05, g05, C.orange);
  P.text(0.062, g05 + 0.03, fmt(g05), { size: 19, color: OR });
  P.text(0.16, 0.2, "λ = 10", { size: 19, color: OR });
  P.text(0.2, 0.72, "λ = 1", { size: 19, color: C.blue });
  s.out("gmrf/fig-gain.svg");
  console.log("gain pi", gmrfResponse(Math.PI, 1), gmrfResponse(Math.PI, 10), "0.05:", g05);
}

// ---------------------------------------------------------------------------------------
// fig-jumps.svg: three ways to climb by 3, with their quadratic and TV costs.
{
  const s = new Svg(900, 300);
  const cases: [string, number][] = [["one jump of 3", 1], ["three steps of 1", 3], ["thirty steps of 0.1", 30]];
  cases.forEach(([lab, k], i) => {
    const P = new Panel(s, 30 + i * 300, 30, 250, 150, [0, 40], [0, 3]);
    s.line(P.x0, P.Y(0), P.x0 + P.w, P.Y(0), C.grid, 1); s.line(P.x0, P.Y(3), P.x0 + P.w, P.Y(3), C.grid, 1);
    const xs: number[] = [], ys: number[] = [];
    const a = 5, b = 35, w = (b - a) / k;
    xs.push(0); ys.push(0);
    for (let j = 0; j < k; j++) { const x = k === 1 ? 20 : a + j * w + w / 2; xs.push(x, x); ys.push(3 * j / k, 3 * (j + 1) / k); }
    xs.push(40); ys.push(3);
    P.line(xs, ys, C.fg, 3);
    const q = k * (3 / k) ** 2 / 2, tv = 3;
    s.text(P.x0 + 125, 22, lab, { anchor: "middle", size: 20 });
    s.text(P.x0 + 125, 225, `quadratic: ${fmt(q)}`, { anchor: "middle", size: 20, color: C.blue });
    s.text(P.x0 + 125, 255, `TV: ${fmt(tv, 0)}`, { anchor: "middle", size: 20, color: OR });
  });
  s.out("gmrf/fig-jumps.svg");
}

// ---------------------------------------------------------------------------------------
// fig-tv-price.svg: the mrf1d profile (seed 1, sd 0.3): TV staircases the ramp (lambda = 3)
// and shrinks the narrow peak (lambda = 2).
{
  const truth = mixedProfile(), y = noisy(truth, 0.3, 1);
  const t3 = denoise1d(y, "tv", 3), t2 = denoise1d(y, "tv", 2);
  const s = new Svg(980, 380);
  const panel = (x0: number, w: number, view: [number, number], est: Float64Array, lab: string) => {
    const P = new Panel(s, x0, 20, w, 270, [view[0], view[1]], [-1, 3]);
    P.axes(view[0] === 60 ? [60, 80, 100, 120, 140, 160] : [140, 160, 180, 199], [-1, 0, 1, 2, 3], "position", x0 < 200 ? "value" : "");
    for (let i = view[0]; i <= view[1]; i++) s.circle(P.X(i), P.Y(Math.max(-1, Math.min(3, y[i]))), 2.8, C.grey);
    const idx: number[] = []; for (let i = view[0]; i <= view[1]; i++) idx.push(i);
    P.line(idx, idx.map(i => truth[i]), C.fg, 2, "5 4");
    P.steps(idx, idx.map(i => est[i]), C.orange, 3.5);
    P.text(view[0] + 0.03 * (view[1] - view[0]), 2.75, lab, { size: 19, color: OR });
    return P;
  };
  panel(80, 500, [60, 160], t3, "TV, λ = 3: the ramp becomes steps");
  const P2 = panel(660, 300, [140, 199], t2, "TV, λ = 2");
  let pk = 0; for (let i = 166; i < 170; i++) pk = Math.max(pk, t2[i]);
  P2.text(171, pk + 0.15, `peak ${fmt(pk)}`, { size: 19, color: OR });
  s.text(80, 370, "grey: noisy data · dashed: truth · orange: TV estimate", { size: 18, color: C.axis });
  let steps = 0; for (let i = 80; i < 140; i++) if (Math.abs(t3[i + 1] - t3[i]) > 1e-6) steps++;
  s.out("gmrf/fig-tv-price.svg");
  console.log("peak lambda 2", pk, "jumps on the ramp (80..140) at lambda 3:", steps);
}

// ---------------------------------------------------------------------------------------
// fig-prior-samples.svg: draws from the quadratic and TV priors with the same step variance
// (lambda = 1, lambda_TV = sqrt 2; seeds as in the mrf1d "prior samples" tab), and the step densities.
{
  const n = 200, lam = 1, lamTV = Math.sqrt(2 * lam);
  const cols = [C.blue, C.orange, C.green];
  const s = new Svg(1030, 330);
  const q = [0, 1, 2].map(k => samplePrior1d(rng(100 + k), n, "quadratic", lam));
  const t = [0, 1, 2].map(k => samplePrior1d(rng(200 + k), n, "tv", lamTV));
  let ext = 0; for (const d of [...q, ...t]) for (const v of d) ext = Math.max(ext, Math.abs(v));
  ext = Math.ceil(ext / 5) * 5;
  const idx = Array.from({ length: n }, (_, i) => i);
  const walk = (x0: number, draws: Float64Array[], lab: string) => {
    const P = new Panel(s, x0, 30, 300, 220, [0, n - 1], [-ext, ext]);
    P.axes([0, 100, 199], [-ext, 0, ext], "position", x0 < 100 ? "value" : "");
    draws.forEach((d, k) => P.line(idx, d, cols[k], 2));
    s.text(x0 + 150, 20, lab, { anchor: "middle", size: 19 });
  };
  walk(70, q, "quadratic prior: normal steps");
  walk(420, t, "TV prior: Laplace steps");
  const P = new Panel(s, 790, 30, 190, 220, [-3, 3], [0, 1]);
  P.axes([-3, 0, 3], [0, 0.5, 1], "step d", "");
  const ds: number[] = []; for (let k = 0; k <= 200; k++) ds.push(-3 + 6 * k / 200);
  P.line(ds, ds.map(d => Math.exp(-d * d / 2) / Math.sqrt(2 * Math.PI)), C.blue, 3);
  P.line(ds, ds.map(d => lamTV / 2 * Math.exp(-lamTV * Math.abs(d))), C.orange, 3, "8 4");
  s.text(885, 20, "step density", { anchor: "middle", size: 19 });
  s.text(885, 318, "solid: normal · dashed: Laplace", { size: 16, anchor: "middle", color: C.axis });
  s.out("gmrf/fig-prior-samples.svg");
}
