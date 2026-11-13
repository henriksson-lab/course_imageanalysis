// Tests for the maxflow crate (demos/wasm/maxflow/, built by `make wasm` into
// demos/dist/wasm/maxflow/) against the TypeScript reference lib/maxflow.ts (Dinic) and
// lib/graph.ts solveSeg(): the same maximum flow and the same source side (the smallest minimum
// cut) on random graphs, a valid flow on every edge, and the same segmentation and energy on
// grid problems like the graphcut demo's. Through the wrappers of demos/src/graphcut/wasmapi.ts. Run:
//   node_modules/.bin/esbuild demos/test/maxflow-wasm-test.ts --bundle --platform=node --log-level=warning | node
// Skips with a message when the crate is not built (WASM_REQUIRED=1, as in CI, makes that a failure).
import { FlowGraph } from "../src/lib/maxflow";
import { gridEdges, contrastWeights, solveSeg, gaussCosts, SegProblem, Nbhd } from "../src/lib/graph";
import { rng } from "../src/lib/ui";
import { WasmFlowGraph, WasmGridCut, MaxflowGlue } from "../src/graphcut/wasmapi";

declare const require: (m: string) => any;
declare const process: { exit(code: number): never; cwd(): string; env: Record<string, string | undefined> };
const fs = require("fs"), path = require("path"), url = require("url");

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol = 1e-7) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

async function main(): Promise<void> {
  const dir = path.join(process.cwd(), "demos/dist/wasm/maxflow");
  const wasmFile = path.join(dir, "maxflow_bg.wasm"), glueFile = path.join(dir, "maxflow.js");
  if (!fs.existsSync(wasmFile) || !fs.existsSync(glueFile)) {
    if (process.env.WASM_REQUIRED) { check(false, "demos/dist/wasm/maxflow not built (run make wasm)"); return; }
    console.log("SKIP maxflow crate: demos/dist/wasm/maxflow/ not built (run make wasm)");
    return;
  }
  const g: MaxflowGlue & { default: (o: object) => Promise<unknown> } = await import(url.pathToFileURL(glueFile).href);
  await g.default({ module_or_path: new Uint8Array(fs.readFileSync(wasmFile)) });

  // ---- Random graphs: FlowGraph API -----------------------------------------------------------
  current = "random graphs";
  const r = rng(7);
  for (let trial = 0; trial < 300; trial++) {
    const n = 2 + Math.floor(r.unif() * (trial < 200 ? 10 : 60));
    const m = Math.floor(r.unif() * n * 4);
    const ts = new FlowGraph(n, m), wa = new WasmFlowGraph(g, n, m);
    const edges: [number, number, number, number][] = [];
    const integer = trial % 2 === 0;   // integer capacities give ties: a test of the "smallest cut"
    for (let k = 0; k < m; k++) {
      const u = Math.floor(r.unif() * n), v = Math.floor(r.unif() * n);
      if (u === v) continue;
      const c = integer ? Math.floor(r.unif() * 5) : r.unif() * 10;
      const cr = r.unif() < 0.3 ? (integer ? Math.floor(r.unif() * 5) : r.unif() * 10) : 0;
      edges.push([u, v, c, cr]);
      const e1 = ts.addEdge(u, v, c, cr), e2 = wa.addEdge(u, v, c, cr);
      check(e1 === e2, `edge ids ${e1} ${e2}`);
    }
    const s = 0, t = n - 1;
    const f1 = ts.maxflow(s, t), f2 = wa.maxflow(s, t);
    check(close(f1, f2), `trial ${trial}: max flow ${f1} vs ${f2}`);
    const S1 = ts.sourceSide(s), S2 = wa.sourceSide(s);
    check(S1.length === S2.length && S1.every((v, i) => v === S2[i]), `trial ${trial}: source side ${S1.join("")} vs ${S2.join("")}`);
    // the Wasm flow is a valid flow: within capacities, conserved, of value f2, saturating the cut
    const net = new Float64Array(n);
    let cutCap = 0;
    edges.forEach(([u, v, c, cr], k) => {
      const fe = wa.flow(2 * k);
      check(fe <= c + 1e-9 && -fe <= cr + 1e-9, `trial ${trial}: flow ${fe} within [-${cr}, ${c}]`);
      net[u] -= fe; net[v] += fe;
      if (S2[u] && !S2[v]) cutCap += c;
      if (S2[v] && !S2[u]) cutCap += cr;
    });
    for (let p = 0; p < n; p++) if (p !== s && p !== t) check(Math.abs(net[p]) < 1e-7, `trial ${trial}: conservation at ${p}: ${net[p]}`);
    check(close(net[t], f2) && close(cutCap, f2), `trial ${trial}: flow into t ${net[t]}, cut ${cutCap}, max flow ${f2}`);
    wa.free();
  }

  // ---- Grid segmentation: GridCut against solveSeg -----------------------------------------------
  current = "grid cut";
  for (const [side, nbhd, lambda, sigma, noise] of [
    [12, 4, 1, Infinity, 1], [20, 8, 0.5, 1.5, 0.8], [33, 8, 2, Infinity, 1.2], [64, 8, 1.5, Infinity, 0.6], [64, 4, 6, 1, 0.9],
  ] as [number, Nbhd, number, number, number][]) {
    const n = side * side, rr = rng(side + nbhd);
    const I = new Float64Array(n);
    for (let y = 0; y < side; y++) for (let x = 0; x < side; x++)
      I[y * side + x] = ((x - side / 2) ** 2 + (y - side / 2) ** 2 < (side / 3) ** 2 ? 3 : 0) + noise * rr.norm();
    const { D0, D1 } = gaussCosts(I, 0, 3, 1);
    const seeds = new Int8Array(n);
    seeds[(side >> 1) * side + (side >> 1)] = 1; seeds[0] = -1; seeds[n - 1] = -1;
    const E = gridEdges(side, side, nbhd);
    const P: SegProblem = { n, D0, D1, E, w: contrastWeights(I, E, sigma), lambda, seeds };
    const ref = solveSeg(P);
    const gc = new WasmGridCut(g, n, E);
    for (let rep = 0; rep < 2; rep++) {   // the second solve reuses the graph
      const { flow: f, L } = gc.solve(P);
      check(close(f, ref.flow), `${side}² ${nbhd}-nbhd λ=${lambda}: flow ${f} vs ${ref.flow}`);
      check(L.every((v, i) => v === ref.L[i]), `${side}² ${nbhd}-nbhd λ=${lambda}: labels differ`);
      check(close(f, ref.energy.total, 1e-6), `${side}² flow ${f} = energy ${ref.energy.total}`);
    }
    // a changed problem on the same graph (other seeds and lambda)
    seeds.fill(0); seeds[1] = -1;
    const P2 = { ...P, seeds, lambda: lambda / 2 };
    const ref2 = solveSeg(P2);
    const r2 = gc.solve(P2);
    check(close(r2.flow, ref2.flow) && r2.L.every((v, i) => v === ref2.L[i]), `${side}² re-solve`);
    gc.free();
  }

  // ---- For information: speed on a 256 x 256 problem ---------------------------------------------
  current = "speed";
  {
    const side = 256, n = side * side, rr = rng(3);
    const I = new Float64Array(n);
    for (let y = 0; y < side; y++) for (let x = 0; x < side; x++)
      I[y * side + x] = ((x - 128) ** 2 + (y - 128) ** 2 < 80 ** 2 ? 3 : 0) + rr.norm();
    const { D0, D1 } = gaussCosts(I, 0, 3, 1);
    const E = gridEdges(side, side, 8), w = contrastWeights(I, E, Infinity);
    const seeds = new Int8Array(n);
    const t0 = performance.now();
    const ref = solveSeg({ n, D0, D1, E, w, lambda: 1.5, seeds });
    const t1 = performance.now();
    const gc = new WasmGridCut(g, n, E);
    const t2 = performance.now();
    const r = gc.solve({ n, D0, D1, E, w, lambda: 1.5, seeds });
    const t3 = performance.now();
    check(close(r.flow, ref.flow) && r.L.every((v, i) => v === ref.L[i]), "256² result");
    console.log(`  256² 8-nbhd: TypeScript solveSeg ${(t1 - t0).toFixed(0)} ms, Wasm GridCut build ${(t2 - t1).toFixed(0)} ms + solve ${(t3 - t2).toFixed(0)} ms`);
    gc.free();
  }
}

main().then(() => {
  console.log(`${checks - failures}/${checks} checks passed`);
  if (failures) process.exit(1);
}, (e) => { console.log("ERROR", e); process.exit(1); });
