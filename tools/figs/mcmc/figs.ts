// Static figures for mcmc/lec.qmd. Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/mcmc/figs.ts --bundle --platform=node --log-level=warning | node
// Simulations use the demos' own code (lib/mrf.ts) with the demos' seeds, so the numbers
// match the isingenum and isinglattice demos.
import { Svg, Panel, C, fmt, toRGB, SPIN_RGB } from "../ising/svglib";
import {
  lattice, metropolisSweep, magnetisation, energy, randomSpins, conditionalPlus, acceptance,
  localField, onsagerM, TC,
} from "../../../demos/src/lib/mrf";
import { rng, randInt } from "../../../demos/src/lib/random";

// ---------------------------------------------------------------------------------------
// fig-local-rule.svg: neighbours (+, +, +, -) vote u = 2; p(+1 | neighbours) against u for T = 1, 2, 4.
{
  const s = new Svg(470, 560);
  // neighbourhood diagram
  const cx = 110, cy = 100, c = 52, g = 6;
  const cellAt = (dx: number, dy: number, v: number | null, t: string) => {
    const x = cx + dx * (c + g) - c / 2, y = cy + dy * (c + g) - c / 2;
    s.rect(x, y, c, c, v === null ? "#eee" : v > 0 ? C.plus : C.minus, { rx: 6, stroke: v === null ? "#999" : undefined, sw: 2 });
    s.text(x + c / 2, y + c / 2 + 9, t, { anchor: "middle", size: 26, color: v === null ? C.fg : "#fff", weight: "bold" });
  };
  cellAt(0, -1, 1, "+"); cellAt(-1, 0, 1, "+"); cellAt(1, 0, 1, "+"); cellAt(0, 1, -1, "−"); cellAt(0, 0, null, "?");
  s.text(215, 95, "u = 1 + 1 + 1 − 1 = 2", { size: 21 });
  s.text(215, 125, "(J = 1, h = 0)", { size: 19, color: C.axis });
  // logistic curves
  const P = new Panel(s, 90, 220, 360, 260, [-4, 4], [0, 1]);
  P.axes([-4, -2, 0, 2, 4], [0, 0.5, 1], "neighbours' vote u", "p(pixel = +1)");
  const us: number[] = []; for (let k = 0; k <= 160; k++) us.push(-4 + 8 * k / 160);
  const cols: [number, string, string][] = [[1, C.blue, ""], [2, C.orange, "8 4"], [4, C.purple, "3 4"]];
  for (const [T, col, dash] of cols) P.line(us, us.map(u => conditionalPlus(u, T)), col, 3.5, dash);
  P.vline(2, C.grey, 1.5, "4 4");
  const dash = (T: number) => (T === 1 ? "" : T === 2 ? "8 4" : "3 4");
  const lab: [number, string, number][] = [[1, C.blue, 0.02], [2, C.orange, -0.05], [4, C.purple, -0.06]];
  for (const [T, col, dy] of lab) {
    const p = conditionalPlus(2, T);
    P.dot(2, p, col);
    const ly = 0.42 - 0.1 * [1, 2, 4].indexOf(T);
    P.line([1.0, 1.5], [ly + 0.02, ly + 0.02], col, 3.5, dash(T));
    P.text(1.65, ly, `T = ${T}: ${fmt(p)}`, { size: 18, color: col === C.orange ? "#c05f00" : col });
    console.log("p(+|u=2) T", T, p.toFixed(3));
  }
  s.out("mcmc/fig-local-rule.svg");
}

// ---------------------------------------------------------------------------------------
// fig-acceptance.svg: probability that a proposed flip happens, against Delta E.
{
  const s = new Svg(620, 380);
  const P = new Panel(s, 80, 20, 510, 280, [-8, 8], [0, 1]);
  P.axes([-8, -4, 0, 4, 8], [0, 0.25, 0.5, 0.75, 1], "energy change ΔE of the flip", "probability of flipping");
  const xs: number[] = []; for (let k = 0; k <= 320; k++) xs.push(-8 + 16 * k / 320);
  P.line(xs, xs.map(d => acceptance(d, 2)), C.orange, 3.5);
  P.line(xs, xs.map(d => acceptance(d, 0.5)), C.blue, 3.5, "3 4");
  P.line(xs, xs.map(d => 1 / (1 + Math.exp(d / 2))), C.purple, 3, "10 5");
  const a2 = acceptance(4, 2), a05 = acceptance(4, 0.5);
  P.dot(4, a2, C.orange); P.dot(4, a05, C.blue);
  P.text(2.0, 0.76, "uphill by ΔE = 4:", { size: 18 });
  P.text(2.0, 0.67, `T = 2: e⁻² = ${fmt(a2)}`, { size: 18, color: "#c05f00" });
  P.text(2.0, 0.58, `T = 0.5: e⁻⁸ = ${a05.toFixed(4)}`, { size: 18, color: C.blue });
  const leg: [string, string, string, number][] = [["Metropolis T = 2", C.orange, "", 0.42], ["Metropolis T = 0.5", C.blue, "3 4", 0.33], ["Gibbs T = 2", C.purple, "10 5", 0.24]];
  for (const [l, col, d, y] of leg) { P.line([-7.7, -6.6], [y + 0.02, y + 0.02], col, 3.5, d); P.text(-6.4, y, l, { size: 18, color: col === C.orange ? "#c05f00" : col }); }
  P.text(0.3, 0.9, "downhill: always", { size: 17, color: C.axis });
  s.out("mcmc/fig-acceptance.svg");
  console.log("accept dE=4", a2, a05);
}

// ---------------------------------------------------------------------------------------
// fig-twostate.svg: the two-state Metropolis chain and its convergence to (0.73, 0.27).
{
  const s = new Svg(470, 560);
  const q = Math.exp(-1);
  // diagram
  const ax = 150, bx = 340, y = 140, r = 42;
  s.circle(ax, y, r, "#fff", C.orange, 4); s.circle(bx, y, r, "#fff", C.blue, 4);
  s.text(ax, y - 2, "a", { anchor: "middle", size: 26, weight: "bold" }); s.text(ax, y + 22, "E = 0", { anchor: "middle", size: 16, color: C.axis });
  s.text(bx, y - 2, "b", { anchor: "middle", size: 26, weight: "bold" }); s.text(bx, y + 22, "E = 1", { anchor: "middle", size: 16, color: C.axis });
  s.add(`<path d="M${ax + 30},${y - 32} Q${(ax + bx) / 2},${y - 95} ${bx - 34},${y - 30}" fill="none" stroke="${C.fg}" stroke-width="2.5"/>`);
  s.add(`<path d="M${bx - 34},${y - 30} l-4,-14 l-9,10 Z" fill="${C.fg}"/>`);
  s.text((ax + bx) / 2, y - 72, fmt(q), { anchor: "middle", size: 21 });
  s.add(`<path d="M${bx - 30},${y + 32} Q${(ax + bx) / 2},${y + 95} ${ax + 34},${y + 30}" fill="none" stroke="${C.fg}" stroke-width="2.5"/>`);
  s.add(`<path d="M${ax + 34},${y + 30} l4,14 l9,-10 Z" fill="${C.fg}"/>`);
  s.text((ax + bx) / 2, y + 88, "1", { anchor: "middle", size: 21 });
  s.add(`<path d="M${ax - 28},${y - 30} C${ax - 80},${y - 100} ${ax + 10},${y - 110} ${ax - 2},${y - 44}" fill="none" stroke="${C.fg}" stroke-width="2.5"/>`);
  s.text(ax - 40, y - 92, fmt(1 - q), { anchor: "middle", size: 21 });
  s.text(400, 40, "T = 1", { anchor: "middle", size: 18, color: C.axis });
  // convergence: exact P(a) after k steps from a and from b, and the running fraction of one run
  const P = new Panel(s, 90, 270, 350, 210, [0, 12], [0, 1]);
  P.axes([0, 4, 8, 12], [0, 0.27, 0.5, 0.73, 1], "steps", "probability of being in a");
  const ks: number[] = [], fa: number[] = [], fb: number[] = [];
  let pa = 1, pb = 0;
  for (let k = 0; k <= 12; k++) { ks.push(k); fa.push(pa); fb.push(pb); const na = pa * (1 - q) + (1 - pa) * 1; pa = na; const nb = pb * (1 - q) + (1 - pb); pb = nb; }
  P.hline(1 / (1 + q), C.grey, 2, "6 4");
  P.line(ks, fa, C.orange, 3); P.line(ks, fb, C.blue, 3, "8 4");
  for (let k = 0; k <= 12; k += 1) { P.dot(ks[k], fa[k], C.orange, 5); P.dot(ks[k], fb[k], C.blue, 5); }
  P.text(11.8, 0.82, `π(a) = ${fmt(1 / (1 + q))}`, { anchor: "end", size: 18 });
  P.text(1.2, 0.12, "start in b", { size: 18, color: C.blue });
  P.text(1.0, 0.93, "start in a", { size: 18, color: "#c05f00" });
  s.out("mcmc/fig-twostate.svg");
  console.log("pi(a)", 1 / (1 + q), "after 12", fa[12], fb[12]);
}

// ---------------------------------------------------------------------------------------
// fig-burnin.svg: (left) E per spin from two starts, 128 x 128 periodic, T = 2.5 (seed 1);
// (right) the 3 x 3 periodic lattice at T = 1, Metropolis, from all +1 (the isingenum "stuck" preset).
{
  const s = new Svg(1010, 360);
  const T = 2.5, K = 150;
  const run = (start: "plus" | "random") => {
    const L = lattice(128, 128, true), r = rng(1);
    const sp = start === "plus" ? new Int8Array(L.n).fill(1) : randomSpins(L.n, r);
    const e = [energy(L, sp, 1, 0).total / L.n];
    for (let k = 1; k <= K; k++) { metropolisSweep(L, sp, 1, 0, T, r); e.push(energy(L, sp, 1, 0).total / L.n); }
    return e;
  };
  const eR = run("random"), eP = run("plus"), ks = eR.map((_, k) => k);
  const P = new Panel(s, 90, 20, 360, 250, [0, K], [-2, 0]);
  P.s.rect(P.X(0), P.Y(0), P.X(40) - P.X(0), P.Y(-2) - P.Y(0), "#f1f1f1");
  P.axes([0, 50, 100, 150], [-2, -1.5, -1, -0.5, 0], "sweeps", "energy per spin");
  P.line(ks, eR, C.blue, 2.5); P.line(ks, eP, C.orange, 2.5);
  P.text(20, -0.25, "burn-in", { anchor: "middle", size: 18, color: C.axis });
  P.text(K, -0.6, "from random spins", { anchor: "end", size: 18, color: C.blue });
  P.text(K, -1.75, "from all +1", { anchor: "end", size: 18, color: "#c05f00" });
  console.log("burnin E/n at 40:", eR[40].toFixed(3), eP[40].toFixed(3), "at 150:", eR[K].toFixed(3), eP[K].toFixed(3));
  // 3 x 3 chain
  const L = lattice(3, 3, true), r = rng(1), sp = new Int8Array(9).fill(1);
  const N = 100000, every = 100, ms: number[] = [], st: number[] = [];
  let minM = 1, sumPos = 0;
  for (let i = 1; i <= N; i++) {
    const p = randInt(r, 9), u = localField(L, sp, p, 1, 0);
    if (r.unif() < acceptance(2 * sp[p] * u, 1)) sp[p] = -sp[p];
    const m = magnetisation(sp); minM = Math.min(minM, m); if (m > 0) sumPos++;
    if (i % every === 0) { ms.push(m); st.push(i); }
  }
  const Q = new Panel(s, 600, 20, 350, 250, [0, N], [-1, 1]);
  Q.s.rect(Q.X(0), Q.Y(0), Q.w, Q.Y(-1) - Q.Y(0), "#eef3f9");
  Q.axes([0, 50000, 100000], [-1, -0.5, 0, 0.5, 1], "steps (one site each)", "magnetisation m", v => (v === 0 ? "0" : v === 50000 ? "50 000" : "100 000"));
  Q.line(st, ms, C.orange, 1.2);
  Q.text(N / 2, -0.6, "never visited", { anchor: "middle", size: 18, color: C.blue });
  Q.text(N / 2, -0.78, "(as probable as m > 0)", { anchor: "middle", size: 17, color: C.axis });
  s.text(270, 350, `128 × 128, T = ${T}`, { anchor: "middle", size: 18, color: C.axis });
  s.text(775, 350, "3 × 3, T = 1, Metropolis", { anchor: "middle", size: 18, color: C.axis });
  s.out("mcmc/fig-burnin.svg");
  console.log("3x3 stuck: min m", minM, "fraction m>0", sumPos / N);
}

// ---------------------------------------------------------------------------------------
// fig-onsager.svg: |m| against T, Onsager's curve and the demo's T-scan (64 x 64, seed 2).
{
  const SCAN_T = Array.from({ length: 31 }, (_, i) => Math.round((4 - 0.1 * i) * 10) / 10);
  const L = lattice(64, 64, true), r = rng(2), sp = randomSpins(L.n, r);
  const dots: [number, number][] = [];
  for (const T of SCAN_T) {
    let acc = 0;
    for (let k = 1; k <= 300; k++) { metropolisSweep(L, sp, 1, 0, T, r); if (k > 100) acc += Math.abs(magnetisation(sp)); }
    dots.push([T, acc / 200]);
  }
  const s = new Svg(620, 370);
  const P = new Panel(s, 80, 20, 510, 270, [1, 4], [0, 1]);
  P.axes([1, 1.5, 2, 2.5, 3, 3.5, 4], [0, 0.25, 0.5, 0.75, 1], "temperature T", "magnetisation |m|");
  const Ts: number[] = []; for (let k = 0; k <= 600; k++) Ts.push(1 + 3 * k / 600);
  P.vline(TC, C.grey, 2, "6 4");
  P.text(TC + 0.05, 0.93, `T_c = ${TC.toFixed(2)}`, { size: 18, color: C.axis });
  P.line(Ts, Ts.map(T => onsagerM(T)), C.fg, 3);
  for (const [T, m] of dots) P.dot(T, m, C.orange, 5.5);
  const d2 = dots.find(d => Math.abs(d[0] - 2) < 1e-9)!;
  P.text(1.1, 0.55, `T = 2: ${d2[1].toFixed(3)}`, { size: 18, color: "#c05f00" });
  P.text(1.1, 0.47, `(exact ${onsagerM(2).toFixed(3)})`, { size: 18 });
  s.line(P.X(1.62), P.Y(0.6), P.X(1.97), P.Y(d2[1] - 0.02), C.grey, 1.5);
  P.text(3.95, 0.5, "line: Onsager (infinite lattice)", { anchor: "end", size: 18 });
  P.text(3.95, 0.42, "dots: 64 × 64 simulation", { anchor: "end", size: 18, color: "#c05f00" });
  s.out("mcmc/fig-onsager.svg");
  console.log("scan T=2", d2, "T=3", dots.find(d => d[0] === 3), "T=4", dots[0]);
}

// ---------------------------------------------------------------------------------------
// fig-slowing.svg: |m| against sweeps from all +1, 128 x 128 (seed 1, as the table on the slide).
{
  const temps: [number, string, string][] = [[4, C.purple, "3 4"], [3, C.blue, "8 4"], [2.5, C.green, ""], [2.3, C.orange, ""], [2.27, C.red, ""]];
  const s = new Svg(800, 380);
  const P = new Panel(s, 80, 20, 480, 280, [1, 5000], [0, 1], true);
  P.axes([1, 10, 100, 1000, 5000], [0, 0.1, 0.25, 0.5, 0.75, 1], "sweeps (log scale)", "magnetisation |m|", v => String(v), v => String(v));
  P.hline(0.1, C.grey, 2, "6 4");
  temps.forEach(([T, col, dash], idx) => {
    const L = lattice(128, 128, true), sp = new Int8Array(L.n).fill(1), r = rng(1);
    const ks: number[] = [], ms: number[] = []; let cross = -1;
    for (let k = 1; k <= 5000; k++) {
      metropolisSweep(L, sp, 1, 0, T, r);
      const m = Math.abs(magnetisation(sp));
      ks.push(k); ms.push(m);
      if (cross < 0 && m < 0.1) { cross = k; if (k > 50) { /* keep running a bit for the picture */ } }
      if (cross > 0 && k > cross * 1.3 + 5) break;
    }
    P.line(ks, ms, col, 2.5, dash);
    P.dot(cross, 0.1, col, 6);
    const ly = 60 + idx * 34;
    s.line(590, ly - 6, 625, ly - 6, col, 3, dash);
    s.text(635, ly, `T = ${T}`, { size: 18, color: col === C.orange ? "#c05f00" : col });
    s.text(790, ly, String(cross), { size: 18, anchor: "end", color: col === C.orange ? "#c05f00" : col });
    console.log("slowing", T, cross);
  });
  s.text(690, 30, "sweeps to |m| &lt; 0.1", { size: 17, anchor: "middle", color: C.axis });
  s.out("mcmc/fig-slowing.svg");
}

// ---------------------------------------------------------------------------------------
// fig-anneal.svg: 128 x 128 from random spins (seed 1): quench to T = 0.5 vs cooling 3.5 -> 0.5
// over 300 sweeps (the isinglattice presets "quench" and "slow cooling"), after 1000 sweeps.
{
  const runs: { label: string; s: Int8Array; m: number; e: number }[] = [];
  for (const mode of ["quench", "slow"] as const) {
    const L = lattice(128, 128, true), r = rng(1), sp = randomSpins(L.n, r);
    let T = mode === "quench" ? 0.5 : 3.5;
    const rate = (0.5 / 3.5) ** (1 / 300);
    let cooling = mode === "slow";
    for (let k = 1; k <= 1000; k++) {
      if (cooling) { T = Math.max(0.5, T * rate); if (T <= 0.5 + 1e-9) { T = 0.5; cooling = false; } }
      metropolisSweep(L, sp, 1, 0, T, r);
    }
    runs.push({ label: mode, s: sp, m: magnetisation(sp), e: energy(L, sp, 1, 0).total / L.n });
  }
  const s = new Svg(540, 380);
  runs.forEach((ru, i) => {
    const x = 15 + i * 270;
    s.image(x, 15, 240, 240, toRGB(128 * 128, p => SPIN_RGB(ru.s[p])), 128, 128);
    s.text(x + 120, 286, i === 0 ? "quench to T = 0.5" : "cool 3.5 → 0.5", { anchor: "middle", size: 22 });
    s.text(x + 120, 314, i === 0 ? "" : "over 300 sweeps", { anchor: "middle", size: 22 });
    s.text(x + 120, 346, `m = ${fmt(ru.m)}`, { anchor: "middle", size: 21, color: C.axis });
    s.text(x + 120, 372, `E per spin ${fmt(ru.e)}`, { anchor: "middle", size: 21, color: C.axis });
    console.log("anneal", ru.label, ru.m, ru.e);
  });
  s.out("mcmc/fig-anneal.svg");
}
