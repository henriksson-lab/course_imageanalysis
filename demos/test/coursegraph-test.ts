// Checks of the course overview graph (demos/src/coursegraph/graph.ts) on the real days.json and on
// random DAGs. Run:
//   node_modules/.bin/esbuild demos/test/coursegraph-test.ts --bundle --platform=node --log-level=warning | node
// - closure() equals reachability from a dense Warshall transitive closure;
// - topoOrder() puts every prerequisite before its user, contains each node once, and follows the
//   tie-break order (on the course data: the course order itself, which respects all prerequisites);
// - learningPath() of every day and unit: the steps cover exactly the closure, include all units of
//   every hard prerequisite day, and a day's prerequisites are a subset of any later unit's;
// - known cases from days.yml (day 1 needs nothing; hessian/ needs all of day 1 and the first two
//   units of day 2, not the structure tensor); dayEdges() and search().
import raw from "../src/coursegraph/days.json";
import { CourseData, buildCourse, closure, topoOrder, learningPath, dayEdges, search, resolve } from "../src/coursegraph/graph";
import { rng } from "../src/lib/random";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}

/** Dense transitive closure (Warshall): R[i][j] = j is a (transitive) prerequisite of i. */
function warshall(prereqs: number[][]): Uint8Array[] {
  const n = prereqs.length;
  const R = Array.from({ length: n }, () => new Uint8Array(n));
  prereqs.forEach((ps, i) => { for (const p of ps) R[i][p] = 1; });
  for (let k = 0; k < n; k++) for (let i = 0; i < n; i++) if (R[i][k]) {
    const Ri = R[i], Rk = R[k];
    for (let j = 0; j < n; j++) if (Rk[j]) Ri[j] = 1;
  }
  return R;
}

function checkTopo(prereqs: number[][], nodes: Set<number>, order: number[], what: string): void {
  check(order.length === nodes.size && new Set(order).size === order.length && order.every(n => nodes.has(n)),
    `${what}: order is a permutation of the set`);
  const at = new Map(order.map((n, i) => [n, i]));
  for (const n of order) for (const p of prereqs[n]) if (nodes.has(p))
    check(at.get(p)! < at.get(n)!, `${what}: ${p} before ${n}`);
}

function testRandom(): void {
  current = "random DAGs";
  const r = rng(13);
  for (let trial = 0; trial < 200; trial++) {
    const n = 2 + Math.floor(r.unif() * 40);
    // random DAG: edges go from a later to an earlier node of a random permutation
    const perm = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(r.unif() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
    const dens = 0.05 + 0.3 * r.unif();
    const prereqs: number[][] = Array.from({ length: n }, () => []);
    for (let a = 0; a < n; a++) for (let b = 0; b < a; b++) if (r.unif() < dens) prereqs[perm[a]].push(perm[b]);
    const R = warshall(prereqs);
    // tie-break by node index, which is NOT a topological order here
    const all = new Set(prereqs.keys());
    checkTopo(prereqs, all, topoOrder(prereqs, all), `trial ${trial} all`);
    for (let t = 0; t < n; t++) {
      const cl = closure(prereqs, [t]);
      const ref = new Set<number>();
      for (let j = 0; j < n; j++) if (R[t][j]) ref.add(j);
      check(cl.size === ref.size && [...ref].every(j => cl.has(j)), `trial ${trial} closure of ${t}`);
      checkTopo(prereqs, cl, topoOrder(prereqs, cl), `trial ${trial} path to ${t}`);
    }
    // ties: among ready nodes the smallest key goes first, so with no edges the order is sorted by key
    const free: number[][] = Array.from({ length: n }, () => []);
    const key = (i: number) => (i * 7919) % n;
    const o = topoOrder(free, all, key);
    check(o.every((x, i) => i === 0 || key(o[i - 1]) <= key(x)), `trial ${trial} tie-break`);
  }
  // a cycle is reported
  let threw = false;
  try { topoOrder([[1], [0]], [0, 1]); } catch { threw = true; }
  check(threw, "cycle throws");
}

function testCourse(): void {
  current = "days.json";
  const c = buildCourse(raw as CourseData);
  const U = c.units.length, D = c.data.days.length;
  console.log(`  course: ${c.data.parts.length} parts, ${D} days, ${U} units`);
  check(D >= 50 && U >= 200, "size");
  // course order is a topological order: every prerequisite comes earlier
  for (let u = 0; u < U; u++) for (const p of c.prereqs[u]) check(p < u, `course order: ${c.units[p].folder} before ${c.units[u].folder}`);
  const R = warshall(c.prereqs);
  for (let u = 0; u < U; u++) {
    const lp = learningPath(c, { unit: u });
    const ref = new Set<number>();
    for (let j = 0; j < U; j++) if (R[u][j]) ref.add(j);
    check(lp.need.size === ref.size && [...ref].every(j => lp.need.has(j)), `closure of ${c.units[u].folder}`);
    const flat = lp.steps.flatMap(s => s.units);
    checkTopo(c.prereqs, lp.need, flat, `path to ${c.units[u].folder}`);
    check(flat.every((x, i) => i === 0 || flat[i - 1] < x), `path to ${c.units[u].folder} follows the course order`);
    check(lp.steps.every((s, i) => i === 0 || lp.steps[i - 1].day !== s.day), `steps grouped by day (${c.units[u].folder})`);
    // the previous unit of the day is needed, and all earlier units of the day with it
    for (let k = 0; k < c.units[u].k; k++) check(lp.need.has(c.dayUnits[c.units[u].day][k]), `${c.units[u].folder} needs earlier units of its day`);
    // every hard prerequisite of the day (and the unit) is in, with what it needs
    const d = c.data.days[c.units[u].day];
    for (const r of [...d.hard, ...c.units[u].hard]) {
      const t = resolve(c, r);
      check(lp.need.has(t), `${c.units[u].folder} needs ${r}`);
      if (!r.endsWith("/")) for (const x of c.dayUnits[c.units[t].day]) check(lp.need.has(x), `${c.units[u].folder} needs all of day ${r}`);
      for (const x of closure(c.prereqs, [t])) check(lp.need.has(x), `${c.units[u].folder}: transitive via ${r}`);
    }
  }
  for (let di = 0; di < D; di++) {
    const lp = learningPath(c, { day: di });
    const last = c.dayUnits[di][c.dayUnits[di].length - 1];
    const ofLast = closure(c.prereqs, [last]);
    for (const x of c.dayUnits[di]) ofLast.delete(x);
    check(lp.need.size === ofLast.size && [...ofLast].every(x => lp.need.has(x)), `day ${c.data.days[di].id} = closure of its last unit minus the day`);
    check(lp.steps.every(s => s.all === (s.units.length === c.dayUnits[s.day].length)), `day ${c.data.days[di].id}: 'all' flags`);
  }
  // known cases
  const day = (id: string) => c.dayIndex.get(id)!, unit = (f: string) => c.unitIndex.get(f)!;
  check(learningPath(c, { day: day("images") }).steps.length === 0, "day 1 needs nothing");
  const hes = learningPath(c, { unit: unit("hessian") });
  const names = [...hes.need].map(u => c.units[u].folder).sort();
  check(JSON.stringify(names) === JSON.stringify(["canny", "fourier", "images", "kernels", "pca", "scale-space"]), "hessian/ path: " + names.join(","));
  check(!hes.need.has(unit("structure-tensor")), "hessian/ does not need the structure tensor");
  const edgesPath = learningPath(c, { day: day("edges") });
  check(edgesPath.steps.length === 1 && edgesPath.steps[0].all && edgesPath.steps[0].day === day("images"), "edges day needs all of day 1");

  current = "dayEdges";
  const E = dayEdges(c);
  check(new Set(E.map(e => e.from + ">" + e.to)).size === E.length, "no duplicate day edges");
  for (const e of E) if (e.hard) check(e.from < e.to, `hard day edge ${e.from}->${e.to} points forward`);
  for (let u = 0; u < U; u++) for (const p of c.prereqs[u]) {
    const a = c.units[p].day, b = c.units[u].day;
    if (a !== b) check(E.some(e => e.hard && e.from === a && e.to === b), `unit edge ${c.units[p].folder}->${c.units[u].folder} has a day edge`);
  }

  current = "search";
  const s1 = search(c, "watershed");
  check(s1.length > 0 && s1[0].field === "day" && s1[0].day === day("watershed"), "search watershed: day first");
  const s2 = search(c, "STRUCTURE tensor");
  check(s2.some(h => h.unit === unit("structure-tensor")), "search is case-insensitive and finds units");
  check(search(c, "convolution border").some(h => h.field === "topic" && h.day === day("images")), "search finds topics (owns)");
  check(search(c, "   ").length === 0, "empty query");
  check(search(c, "zzqx-nothing").length === 0, "no match");
}

testRandom();
testCourse();
console.log(`coursegraph: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
