// The max flow of the graphcut demo in Rust/WebAssembly (demos/wasm/maxflow/, Boykov-Kolmogorov):
// a Solver for model.ts with the same results as lib/graph.ts solveSeg (checked by
// demos/test/maxflow-wasm-test.ts). The module is small (about 35 kB), so it is inlined in the
// demo bundle and works from file:// too.
import * as glue from "../../dist/wasm/maxflow/maxflow.js";
import wasm from "../../dist/wasm/maxflow/maxflow_bg.wasm";
import { instantiateInline } from "../lib/wasm";
import { gridSolver } from "./wasmapi";
import type { Solver } from "./model";

let ready: Promise<Solver> | null = null;

/** Instantiate the module once per page and return the Wasm solver. */
export function wasmSolver(): Promise<Solver> {
  if (!ready) ready = instantiateInline(glue, wasm).then(() => gridSolver(glue));
  return ready;
}
