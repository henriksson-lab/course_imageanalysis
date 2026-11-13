// The network side of the stardistnet demo, run in a Web Worker so the page stays responsive
// while the network runs (about 1-2 s for 512 x 512 in WebAssembly on a laptop CPU). The worker
// is the same bundle (demos/dist/stardistnet.js) started with `new Worker(...)`; main.ts calls
// runWorker() when there is no document. Messages: protocol.ts.
import * as glue from "../../dist/wasm/stardist/stardist.js";
import { loadWasm, loadModelFile } from "../lib/wasm";
import type { Req, Res } from "./protocol";

export const MODEL_DIR = "stardist/2D_versatile_fluo/";

interface WorkerScope {
  postMessage(m: Res, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<Req>) => void) | null;
}

export function runWorker(): void {
  const ctx = self as unknown as WorkerScope;
  const post = (m: Res, transfer: Transferable[] = []) => ctx.postMessage(m, transfer);
  let loaded = false;
  let engine = 1;
  let last: { prob: Float32Array; dist: Float32Array; ow: number; oh: number; w: number; h: number } | null = null;
  const now = () => performance.now();

  async function handle(m: Req): Promise<void> {
    switch (m.type) {
      case "init": {
        const t0 = now();
        engine = m.engine;
        let wasmBytes = 0;
        await loadWasm(glue, "stardist", { base: m.base, onProgress: (l, t) => { wasmBytes = l; post({ type: "progress", what: "module", loaded: l, total: t }); } });
        const dec = new TextDecoder();
        const [cfg, thr, w] = await Promise.all([
          loadModelFile(MODEL_DIR + "config.json", { base: m.base }),
          loadModelFile(MODEL_DIR + "thresholds.json", { base: m.base }),
          loadModelFile(MODEL_DIR + "weights_best.h5", { base: m.base, onProgress: (l, t) => post({ type: "progress", what: "weights", loaded: l, total: t }) }),
        ]);
        const t1 = now();
        glue.load(dec.decode(cfg), dec.decode(thr), w);
        loaded = true;
        post({ type: "ready", wasmBytes, weightBytes: w.length, fetchMs: t1 - t0, buildMs: now() - t1,
          nRays: glue.n_rays(), grid: glue.grid(), probThresh: glue.default_prob_thresh(), nmsThresh: glue.default_nms_thresh() });
        return;
      }
      case "predict": {
        if (!loaded) throw new Error("network not loaded");
        const t0 = now();
        const x = glue.normalize_percentile(new Float32Array(m.grey), m.pmin, m.pmax);
        glue.predict(x, m.w, m.h, engine);
        const ms = now() - t0;
        const ow = glue.out_width(), oh = glue.out_height();
        last = { prob: glue.prob(), dist: glue.dist(), ow, oh, w: m.w, h: m.h };
        // Copies for the page (the worker keeps its own for decoding).
        const prob = last.prob.slice(), dist = last.dist.slice();
        post({ type: "predicted", id: m.id, ms, ow, oh, prob: prob.buffer, dist: dist.buffer }, [prob.buffer, dist.buffer]);
        return;
      }
      case "decode": {
        if (!loaded || !last) throw new Error("no prediction to decode");
        const t0 = now();
        glue.instances(last.prob, last.dist, last.ow, last.oh, last.w, last.h, glue.n_rays(), glue.grid(), m.probThresh, m.nmsThresh);
        const points = glue.dec_points(), probs = glue.dec_probs(), coords = glue.dec_coords(), labels = glue.dec_labels();
        const res: Res = { type: "decoded", id: m.id, ms: now() - t0, candidates: glue.dec_candidates(), count: glue.dec_count(),
          points: points.buffer as ArrayBuffer, probs: probs.buffer as ArrayBuffer, coords: coords.buffer as ArrayBuffer, labels: labels.buffer as ArrayBuffer };
        post(res, [res.points, res.probs, res.coords, res.labels]);
        return;
      }
    }
  }

  ctx.onmessage = (e) => {
    const m = e.data;
    handle(m).catch((err) => post({ type: "error", id: "id" in m ? m.id : 0, message: (err as Error)?.message ?? String(err) }));
  };
}
