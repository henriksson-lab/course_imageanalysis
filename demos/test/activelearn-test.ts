// Tests of the few-labels unit's active-learning demo (demos/src/activelearn/model.ts): the
// Newton fit reaches a zero gradient, the selection rules do what they say, and the numbers
// quoted on the pages. Run:
//   node_modules/.bin/esbuild demos/test/activelearn-test.ts --bundle --platform=node --log-level=warning | node
import * as A from "../src/activelearn/model";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;
const pool = A.makePool(500, 1), test = A.makePool(600, 2);

current = "pool";
{
  check(pool.length === 500 && pool.filter(p => p.rare).length === 30, "500 cells, 30 of the rare subtype");
  check(pool.filter(p => p.rare).every(p => p.c === 1), "the rare subtype is class 1");
}

current = "fit";
{
  const idx = A.startSet(pool, 1).concat([10, 20, 30, 40, 50, 60]);
  const w = A.fit(pool, idx, undefined, 40);
  // gradient of the penalised loss at the fitted weights ≈ 0
  const g = new Float64Array(A.NF);
  for (const i of idx) { const f = A.phi(pool[i]), e = A.probPt(w, pool[i]) - pool[i].c; for (let a = 0; a < A.NF; a++) g[a] += e * f[a]; }
  for (let a = 1; a < A.NF; a++) g[a] += 2 * A.LAMBDA * w[a];
  check(Math.max(...Array.from(g, Math.abs)) < 1e-6, `gradient at the fit: max |g| = ${Math.max(...Array.from(g, Math.abs))}`);
}

current = "rules";
{
  const L = new A.Labeller(pool, A.startSet(pool, 1), 51);
  const u = L.propose("uncertain");
  const unl = pool.map((_, j) => j).filter(j => !L.isLab[j]);
  check(unl.every(j => Math.abs(L.p[j] - 0.5) >= Math.abs(L.p[u] - 0.5)), "most uncertain: p closest to 0.5");
  const d = L.propose("diverse");
  check(unl.every(j => L.dmin[j] <= L.dmin[d]), "most different: farthest from the labelled cells");
  const m = L.propose("mistakes");
  check((L.p[m] > 0.5 ? 1 : 0) !== pool[m].c, "the model's mistakes: a cell it gets wrong");
  check(L.propose("random") === L.propose("random"), "the random proposal stays until it is labelled");
}

current = "quoted numbers";
{
  const run = (rule: A.Rule, n: number) => { const L = new A.Labeller(pool, A.startSet(pool, 1), 51); while (L.labelled.length < n) L.step(rule); return { L, s: A.score(L.w, test) }; };
  const u = run("uncertain", 40), b = run("both", 40), r = run("random", 40);
  check(close(u.s.acc, 0.91, 0.002) && u.s.rare === 0, `uncertain, 40 labels: ${u.s.acc}, rare ${u.s.rare}; quoted 91 %, 0 %`);
  check(close(b.s.acc, 0.95, 0.002) && close(b.s.rare, 0.81, 0.01), `uncertain + different, 40: ${b.s.acc}, rare ${b.s.rare}; quoted 95 %, 81 %`);
  check(close(r.s.acc, 0.82, 0.002), `random, 40: ${r.s.acc}; quoted 82 %`);
  check(u.L.labelled.filter(j => pool[j].rare).length === 0, "uncertainty sampling labels no rare cell in 40");
  const all = new A.Labeller(pool, pool.map((_, i) => i));
  check(close(A.score(all.w, test).acc, 0.973, 0.002), `all 500 labelled: ${A.score(all.w, test).acc}, quoted 97.3 %`);
}

console.log(`activelearn: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
