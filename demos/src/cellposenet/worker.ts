// The Cellpose Wasm module in Web Workers. The demo bundle (dist/cellposenet.js) is also the
// worker script: loaded by `new Worker(...)`, main.ts sees no `document` and calls
// workerMain(). Each worker holds one network (weights sent once) and runs tiles, decoding
// and paths on request, so the page stays responsive and tiles run in parallel.
// Pool: one request per worker at a time, a queue for the rest.
import * as glue from "../../dist/wasm/cellpose/cellpose.js";
import wasmBytes from "../../dist/wasm/cellpose/cellpose_bg.wasm";
import { instantiateInline } from "../lib/wasm";

export type Req =
  | { type: "init"; weights: Uint8Array }
  | { type: "tile"; input: Float32Array; h: number; w: number }
  | { type: "masks"; dy: Float32Array; dx: Float32Array; cellprob: Float32Array; h: number; w: number;
      niter: number; cellprobThreshold: number; flowThreshold: number; minSize: number }
  | { type: "finish"; masksRaw: Uint32Array; errors: Float32Array; h: number; w: number; flowThreshold: number; minSize: number }
  | { type: "paths"; dy: Float32Array; dx: Float32Array; cellprob: Float32Array; h: number; w: number;
      cellprobThreshold: number; starts: Uint32Array; niter: number; stride: number };

export interface MasksOut {
  masks: Uint32Array; masksRaw: Uint32Array; errors: Float32Array; endY: Float32Array; endX: Float32Array;
  hist: Uint32Array; seeds: Uint32Array; nFg: number; ms: number;
}

// ---- worker side --------------------------------------------------------------------------------

export function workerMain(): void {
  const scope = self as unknown as { onmessage: ((e: MessageEvent) => void) | null; postMessage(m: unknown, t?: Transferable[]): void };
  const ready = instantiateInline(glue, wasmBytes);
  let loadedNet = false;
  scope.onmessage = async (e: MessageEvent) => {
    const { id, req } = e.data as { id: number; req: Req };
    try {
      await ready;
      const t0 = performance.now();
      if (req.type === "init") {
        const params = glue.load_network(req.weights);
        loadedNet = true;
        scope.postMessage({ id, res: { params, ms: performance.now() - t0 } });
      } else if (req.type === "tile") {
        if (!loadedNet) throw new Error("network not loaded");
        const y = glue.run_tile(req.input, req.h, req.w);
        const style = glue.last_style();
        scope.postMessage({ id, res: { y, style, ms: performance.now() - t0 } }, [y.buffer]);
      } else if (req.type === "masks") {
        glue.compute_masks(req.dy, req.dx, req.cellprob, req.h, req.w, req.niter, req.cellprobThreshold,
          req.flowThreshold, req.minSize, 0.4);
        const out: MasksOut = {
          masks: glue.last_masks(), masksRaw: glue.last_masks_raw(), errors: glue.last_flow_errors(), endY: glue.last_end_y(),
          endX: glue.last_end_x(), hist: glue.last_sink_histogram(), seeds: glue.last_seeds(), nFg: glue.last_n_foreground(), ms: 0,
        };
        out.ms = performance.now() - t0;
        scope.postMessage({ id, res: out }, [out.masks.buffer, out.masksRaw.buffer, out.endY.buffer, out.endX.buffer, out.hist.buffer]);
      } else if (req.type === "finish") {
        const masks = glue.finish_masks(req.masksRaw, req.errors, req.h, req.w, req.flowThreshold, req.minSize);
        scope.postMessage({ id, res: { masks, ms: performance.now() - t0 } }, [masks.buffer]);
      } else if (req.type === "paths") {
        const p = glue.trace_paths(req.dy, req.dx, req.cellprob, req.h, req.w, req.cellprobThreshold, req.starts, req.niter, req.stride);
        scope.postMessage({ id, res: { paths: p, ms: performance.now() - t0 } }, [p.buffer]);
      }
    } catch (err) {
      scope.postMessage({ id, error: String((err as Error)?.message ?? err) });
    }
  };
}

// ---- page side ----------------------------------------------------------------------------------

interface Job { req: Req; transfer: Transferable[]; resolve: (v: any) => void; reject: (e: Error) => void }

export class Pool {
  private workers: Worker[] = [];
  private idle: Worker[] = [];
  private queue: Job[] = [];
  private pending = new Map<number, Job>();
  private nextId = 1;

  constructor(scriptUrl: string, n: number) {
    for (let i = 0; i < n; i++) {
      const wk = new Worker(scriptUrl);
      wk.onmessage = (e: MessageEvent) => {
        const { id, res, error } = e.data as { id: number; res?: unknown; error?: string };
        const job = this.pending.get(id);
        this.pending.delete(id);
        if (!this.idle.includes(wk)) this.idle.push(wk);
        if (job) { if (error) job.reject(new Error(error)); else job.resolve(res); }
        this.pump();
      };
      wk.onerror = (e: ErrorEvent) => {
        for (const [id, job] of this.pending) { job.reject(new Error(e.message || "worker failed")); this.pending.delete(id); }
      };
      this.workers.push(wk);
      this.idle.push(wk);
    }
  }

  get size(): number { return this.workers.length; }

  private pump(): void {
    while (this.idle.length && this.queue.length) {
      const wk = this.idle.pop()!, job = this.queue.shift()!, id = this.nextId++;
      this.pending.set(id, job);
      wk.postMessage({ id, req: job.req }, job.transfer);
    }
  }

  /** Run on any free worker. */
  call<T>(req: Req, transfer: Transferable[] = []): Promise<T> {
    return new Promise<T>((resolve, reject) => { this.queue.push({ req, transfer, resolve, reject }); this.pump(); });
  }

  /** Run on every worker (e.g. load the weights; call it while no other jobs run); waits for all. */
  broadcast<T>(req: Req): Promise<T[]> {
    return Promise.all(this.workers.map((wk) => new Promise<T>((resolve, reject) => {
      const id = this.nextId++;
      this.idle = this.idle.filter((x) => x !== wk);
      this.pending.set(id, { req, transfer: [], resolve, reject });
      wk.postMessage({ id, req });
    })));
  }

  terminate(): void { this.workers.forEach((w) => w.terminate()); this.workers = []; this.idle = []; }
}
