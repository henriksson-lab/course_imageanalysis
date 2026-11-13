// TypeScript face of the maxflow crate (demos/wasm/maxflow/): the same API as lib/maxflow.ts
// FlowGraph and a Solver for model.ts, over the crate's handle functions. Takes the glue module
// as a parameter, so node tests can pass one they loaded themselves (wasmcut.ts passes the
// inlined one in the browser).
import type { SegProblem, GridEdges } from "../lib/graph";
import type { Solver } from "./model";

/** The functions of demos/dist/wasm/maxflow/maxflow.js used here. */
export interface MaxflowGlue {
  graph_new(n: number, pairs: number): number;
  graph_free(h: number): void;
  graph_add_edge(h: number, u: number, v: number, c: number, cRev: number): number;
  graph_maxflow(h: number, s: number, t: number): number;
  graph_source_side(h: number, s: number): Uint8Array;
  graph_flow(h: number, e: number): number;
  grid_new(n: number, ep: Int32Array, eq: Int32Array): number;
  grid_free(h: number): void;
  grid_solve(h: number, d0: Float64Array, d1: Float64Array, seeds: Int8Array, w: Float64Array, lambda: number): number;
  grid_labels(h: number): Uint8Array;
}

/** lib/maxflow.ts FlowGraph (addEdge, maxflow, sourceSide, flow), solved in Wasm. Call free()
 *  when done: the graph lives in Wasm memory. */
export class WasmFlowGraph {
  private h: number;
  constructor(private g: MaxflowGlue, readonly n: number, pairs = 16) { this.h = g.graph_new(n, pairs); }
  addEdge(u: number, v: number, c: number, cRev = 0): number { return this.g.graph_add_edge(this.h, u, v, c, cRev); }
  maxflow(s: number, t: number): number { return this.g.graph_maxflow(this.h, s, t); }
  sourceSide(s: number): Uint8Array { return this.g.graph_source_side(this.h, s); }
  flow(e: number): number { return this.g.graph_flow(this.h, e); }
  free(): void { this.g.graph_free(this.h); }
}

/** Graph-cut segmentation on one neighbour graph (lib/graph.ts solveSeg, graph built once). */
export class WasmGridCut {
  private h: number;
  constructor(private g: MaxflowGlue, readonly n: number, E: GridEdges) { this.h = g.grid_new(n, E.p, E.q); }
  /** Minimum energy (= solveSeg's flow) and labels (1 = object). */
  solve(P: SegProblem): { L: Uint8Array; flow: number } {
    const f64 = (a: ArrayLike<number>) => (a instanceof Float64Array ? a : Float64Array.from(a));
    const flow = this.g.grid_solve(this.h, f64(P.D0), f64(P.D1), P.seeds ?? new Int8Array(P.n), f64(P.w), P.lambda);
    return { L: this.g.grid_labels(this.h), flow };
  }
  free(): void { this.g.grid_free(this.h); }
}

/** A Solver that keeps one WasmGridCut per neighbour graph (GridEdges object). */
export function gridSolver(g: MaxflowGlue): Solver {
  const grids = new Map<GridEdges, WasmGridCut>();
  return (P: SegProblem) => {
    let gc = grids.get(P.E);
    if (!gc) { gc = new WasmGridCut(g, P.n, P.E); grids.set(P.E, gc); }
    return gc.solve(P);
  };
}
