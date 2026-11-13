// Static figures for ising/lec.qmd. Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/ising/figs.ts --bundle --platform=node --log-level=warning | node
import { Svg, Panel, C, fmt } from "./svglib";
import { lattice, energy, enumerate, stateSpins } from "../../../demos/src/lib/mrf";

// ---------------------------------------------------------------------------------------
// fig-bonds.svg: a 4 x 4 state with its 24 neighbour pairs, agreeing (green) and disagreeing (red, dashed).
{
  const st = [1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1];
  const cell = 52, gap = 26, x0 = 20, y0 = 20, n = 4;
  const s = new Svg(560, 360);
  const pos = (i: number) => i * (cell + gap);
  let agree = 0, dis = 0;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const v = st[j * n + i];
    if (i + 1 < n) {
      const ok = v === st[j * n + i + 1]; ok ? agree++ : dis++;
      const y = y0 + pos(j) + cell / 2;
      s.line(x0 + pos(i) + cell + 3, y, x0 + pos(i + 1) - 3, y, ok ? C.green : C.red, 6, ok ? "" : "5 4");
    }
    if (j + 1 < n) {
      const ok = v === st[(j + 1) * n + i]; ok ? agree++ : dis++;
      const x = x0 + pos(i) + cell / 2;
      s.line(x, y0 + pos(j) + cell + 3, x, y0 + pos(j + 1) - 3, ok ? C.green : C.red, 6, ok ? "" : "5 4");
    }
  }
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const v = st[j * n + i];
    s.rect(x0 + pos(i), y0 + pos(j), cell, cell, v > 0 ? C.plus : C.minus, { rx: 6 });
    s.text(x0 + pos(i) + cell / 2, y0 + pos(j) + cell / 2 + 7, v > 0 ? "+" : "−", { anchor: "middle", size: 24, color: "#fff", weight: "bold" });
  }
  const E = energy(lattice(4, 4, false), st, 1, 0).total;
  if (E !== -(agree - dis)) throw new Error("bond count");
  const tx = 345;
  s.line(tx, 60, tx + 44, 60, C.green, 6); s.text(tx + 54, 67, `agree: ${agree}`, { size: 20 });
  s.text(tx + 54, 94, `each −J`, { size: 20, color: C.axis });
  s.line(tx, 140, tx + 44, 140, C.red, 6, "5 4"); s.text(tx + 54, 147, `disagree: ${dis}`, { size: 20 });
  s.text(tx + 54, 174, `each +J`, { size: 20, color: C.axis });
  s.text(tx, 232, `J = 1, h = 0:`, { size: 20 });
  s.text(tx, 262, `E = −${agree} + ${dis} = ${fmt(E, 0)}`, { size: 20, weight: "bold" });
  s.text(tx, 300, `lowest possible: −24`, { size: 20, color: C.axis });
  s.out("ising/fig-bonds.svg");
  console.log("bonds: agree", agree, "disagree", dis, "E", E);
}

// ---------------------------------------------------------------------------------------
// fig-levels-T.svg: probability of the three energy levels of the 2 x 2 lattice (J = 1, h = 0, free) against T.
{
  const L = lattice(2, 2, false);
  const Ts: number[] = []; for (let k = 0; k <= 200; k++) Ts.push(10 ** (-0.7 + 2.7 * k / 200));
  const lv: Record<string, number[]> = { "-4": [], "0": [], "4": [] };
  for (const T of Ts) { const en = enumerate(L, 1, 0, T); for (const l of en.levels) lv[String(l.E)].push(l.p); }
  const at = (T: number) => enumerate(L, 1, 0, T).levels;
  const a1 = at(1), a4 = at(4);
  console.log("levels T=1", a1.map(l => [l.E, l.count, l.p.toFixed(3)]), "T=4", a4.map(l => [l.E, l.p.toFixed(3)]));
  const s = new Svg(620, 400);
  const P = new Panel(s, 80, 20, 510, 300, [0.2, 100], [0, 1], true);
  P.axes([0.2, 0.5, 1, 2, 5, 10, 20, 50, 100], [0, 0.25, 0.5, 0.75, 1], "temperature T (log scale)", "probability of the level");
  P.line(Ts, lv["-4"], C.orange, 3.5);
  P.line(Ts, lv["0"], C.purple, 3.5, "10 5");
  P.line(Ts, lv["4"], C.blue, 3.5, "3 4");
  const e1 = a1.find(l => l.E === -4)!.p, e4 = a4.find(l => l.E === -4)!.p;
  P.dot(1, e1, C.orange); P.dot(4, e4, C.orange);
  P.text(0.88, e1 - 0.04, fmt(e1, 2), { anchor: "end", size: 18, color: C.fg });
  P.text(4, e4 + 0.05, fmt(e4, 2), { anchor: "start", size: 18, color: C.fg });
  P.text(1.35, 0.93, "aligned, E = −4 (2 states)", { color: "#c05f00", size: 18 });
  P.text(95, 0.79, "E = 0 (12 states)", { anchor: "end", color: C.purple, size: 18 });
  P.text(95, 0.025, "checkerboards, E = +4", { anchor: "end", color: C.blue, size: 18 });
  s.out("ising/fig-levels-T.svg");
}

// ---------------------------------------------------------------------------------------
// fig-states-3x3.svg: all 512 states of the 3 x 3 periodic lattice, sorted by energy (J = 1, h = 0).
{
  const L = lattice(3, 3, true);
  const states = Array.from({ length: 512 }, (_, k) => ({ k, E: energy(L, stateSpins(k, 9), 1, 0).total }));
  states.sort((a, b) => a.E - b.E || a.k - b.k);
  const cols = 32, cell = 6, tile = 3 * cell, gapx = 4, gapy = 4;
  const s = new Svg(cols * (tile + gapx) - gapx + 4, 16 * (tile + gapy) - gapy + 4);
  let d = { plus: "", minus: "" };
  states.forEach((st, idx) => {
    const sp = stateSpins(st.k, 9), ox = 2 + (idx % cols) * (tile + gapx), oy = 2 + Math.floor(idx / cols) * (tile + gapy);
    for (let p = 0; p < 9; p++) {
      const r = `M${ox + (p % 3) * cell},${oy + Math.floor(p / 3) * cell}h${cell}v${cell}h${-cell}Z`;
      if (sp[p] > 0) d.plus += r; else d.minus += r;
    }
  });
  s.add(`<path d="${d.plus}" fill="${C.plus}"/>`); s.add(`<path d="${d.minus}" fill="${C.minus}"/>`);
  s.out("ising/fig-states-3x3.svg");
  const lev = new Map<number, number>(); for (const st of states) lev.set(st.E, (lev.get(st.E) ?? 0) + 1);
  console.log("3x3 periodic levels", [...lev.entries()]);
}

// ---------------------------------------------------------------------------------------
// fig-transfer.svg: summing an open chain of 4 spins from left to right (J = 1, h = 0, T = 1).
{
  const J = 1, T = 1, a = Math.exp(J / T), b = Math.exp(-J / T);
  const n = 4, w = 220, x0 = 90, yP = 70, yM = 230;
  const s = new Svg(n * w - w + 2 * x0 + 90, 330);
  let vp = 1, vm = 1;
  const vals: [number, number][] = [[vp, vm]];
  for (let i = 1; i < n; i++) { const np = vp * a + vm * b, nm = vp * b + vm * a; vp = np; vm = nm; vals.push([vp, vm]); }
  for (let i = 0; i + 1 < n; i++) {
    const xa = x0 + i * w + 34, xb = x0 + (i + 1) * w - 34;
    s.arrow(xa, yP, xb, yP, C.green, 3); s.arrow(xa, yM, xb, yM, C.green, 3);
    s.arrow(xa, yP + 14, xb, yM - 14, C.red, 2.5); s.arrow(xa, yM - 14, xb, yP + 14, C.red, 2.5);
    if (i === 0) {
      s.text((xa + xb) / 2, yP - 12, `× e = ${fmt(a)}`, { anchor: "middle", size: 18, color: "#1e7a1e" });
      s.text((xa + xb) / 2, yM + 30, `× e = ${fmt(a)}`, { anchor: "middle", size: 18, color: "#1e7a1e" });
      s.text((xa + xb) / 2 + 30, (yP + yM) / 2 - 8, `× 1/e`, { anchor: "start", size: 18, color: C.red });
      s.text((xa + xb) / 2 + 30, (yP + yM) / 2 + 16, `= ${fmt(b)}`, { anchor: "start", size: 18, color: C.red });
    }
  }
  for (let i = 0; i < n; i++) {
    const x = x0 + i * w;
    s.rect(x - 34, yP - 22, 68, 44, C.plus, { rx: 6 }); s.text(x, yP + 7, fmt(vals[i][0]), { anchor: "middle", size: 18, color: "#fff", weight: "bold" });
    s.rect(x - 34, yM - 22, 68, 44, C.minus, { rx: 6 }); s.text(x, yM + 7, fmt(vals[i][1]), { anchor: "middle", size: 18, color: "#fff", weight: "bold" });
    s.text(x, 300, `pixel ${i + 1}`, { anchor: "middle", size: 18, color: C.axis });
  }
  s.text(10, yP + 7, "+1", { size: 20, color: "#c05f00", weight: "bold" });
  s.text(10, yM + 7, "−1", { size: 20, color: C.blue, weight: "bold" });
  const Z = vals[n - 1][0] + vals[n - 1][1];
  const xe = x0 + (n - 1) * w + 52;
  s.text(xe, (yP + yM) / 2 - 6, "Z =", { size: 20 });
  s.text(xe, (yP + yM) / 2 + 20, fmt(Z), { size: 20, weight: "bold" });
  s.out("ising/fig-transfer.svg");
  // check against listing all 16 states
  let Zb = 0; for (let k = 0; k < 16; k++) { const sp = stateSpins(k, 4); let e = 0; for (let i = 0; i < 3; i++) e -= J * sp[i] * sp[i + 1]; Zb += Math.exp(-e / T); }
  console.log("chain Z", Z, "brute", Zb, vals);
}

// ---------------------------------------------------------------------------------------
// fig-spins-labels.svg: the same 3 x 3 state as spins and as labels; disagreeing pairs marked.
{
  const lab = [1, 1, 0, 1, 1, 0, 1, 0, 0];
  const n = 3, cell = 56, gap = 10;
  const s = new Svg(720, 310);
  const drawGrid = (x0: number, spins: boolean) => {
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const v = lab[j * n + i];
      s.rect(x0 + i * (cell + gap), 40 + j * (cell + gap), cell, cell, v ? C.plus : C.minus, { rx: 6 });
      s.text(x0 + i * (cell + gap) + cell / 2, 40 + j * (cell + gap) + cell / 2 + 8, spins ? (v ? "+1" : "−1") : String(v), { anchor: "middle", size: 22, color: "#fff", weight: "bold" });
    }
    // disagreeing pairs: thick black bar in the gap
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const v = lab[j * n + i];
      if (i + 1 < n && v !== lab[j * n + i + 1]) { const x = x0 + i * (cell + gap) + cell + gap / 2; s.line(x, 40 + j * (cell + gap) - 2, x, 40 + j * (cell + gap) + cell + 2, C.fg, 5); }
      if (j + 1 < n && v !== lab[(j + 1) * n + i]) { const y = 40 + j * (cell + gap) + cell + gap / 2; s.line(x0 + i * (cell + gap) - 2, y, x0 + i * (cell + gap) + cell + 2, y, C.fg, 5); }
    }
  };
  let dis = 0, pairs = 0;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    if (i + 1 < n) { pairs++; if (lab[j * n + i] !== lab[j * n + i + 1]) dis++; }
    if (j + 1 < n) { pairs++; if (lab[j * n + i] !== lab[(j + 1) * n + i]) dis++; }
  }
  const agree = pairs - dis;
  drawGrid(20, true); drawGrid(360, false);
  s.text(20 + 99, 26, "spins s = ±1", { anchor: "middle", size: 22 });
  s.text(360 + 99, 26, "labels L = 0 / 1", { anchor: "middle", size: 22 });
  s.text(280, 140, "s = 2L − 1", { anchor: "middle", size: 21, color: C.axis });
  s.arrow(240, 150, 325, 150, C.axis, 2);
  s.text(10, 268, `Ising (J = 1): −(${agree} − ${dis}) = ${fmt(-(agree - dis), 0)}`, { size: 22 });
  s.text(372, 268, `smoothness (λ = 2): 2 · ${dis} = ${2 * dis}`, { size: 22 });
  s.text(360, 302, `difference ${2 * dis + (agree - dis)} = J × (${pairs} pairs) for every state`, { anchor: "middle", size: 21, color: C.axis });
  s.out("ising/fig-spins-labels.svg");
}
