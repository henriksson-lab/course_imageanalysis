// The network side of the samnet demo, run in a Web Worker so the page stays responsive while
// the image encoder works (tens of seconds on a CPU in WebAssembly). The worker is the same
// bundle (demos/dist/samnet.js) started with `new Worker(...)`; main.ts calls runWorker() when
// there is no document. Messages: see the Req/Res types in protocol.ts.
import * as glue from "../../dist/wasm/sam/sam.js";
import { loadWasm, loadModelFile } from "../lib/wasm";
import * as P from "./post";
import type { Req, Res, CandidateMsg } from "./protocol";

export const WEIGHTS_KEY = "sam2/sam2.1_hiera_tiny.safetensors";

interface WorkerScope {
  postMessage(m: Res, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<Req>) => void) | null;
}

export function runWorker(): void {
  const ctx = self as unknown as WorkerScope;
  const post = (m: Res, transfer: Transferable[] = []) => ctx.postMessage(m, transfer);
  let loaded = false;
  let size = { w: 0, h: 0 };
  let amgRun = 0; // id of the automatic-mask run that may continue (0: none)
  const now = () => performance.now();

  async function handle(m: Req): Promise<void> {
    switch (m.type) {
      case "init": {
        const t0 = now();
        let wasmBytes = 0, weightBytes = 0;
        await loadWasm(glue, "sam", { base: m.base, onProgress: (l, t) => { wasmBytes = l; post({ type: "progress", what: "module", loaded: l, total: t }); } });
        const w = await loadModelFile(WEIGHTS_KEY, { base: m.base, onProgress: (l, t) => { weightBytes = l; post({ type: "progress", what: "weights", loaded: l, total: t }); } });
        weightBytes = w.length;
        const t1 = now();
        glue.load(w);
        loaded = true;
        post({ type: "ready", wasmBytes, weightBytes, fetchMs: t1 - t0, buildMs: now() - t1 });
        return;
      }
      case "encode": {
        if (!loaded) throw new Error("network not loaded");
        amgRun = 0;
        const t0 = now();
        glue.encode(new Uint8Array(m.rgb), m.w, m.h);
        const ms = now() - t0;
        size = { w: m.w, h: m.h };
        const emb = glue.embedding();
        const pca = P.embeddingRGB(emb, 256, 64);
        post({ type: "encoded", id: m.id, ms, pca: pca.buffer as ArrayBuffer }, [pca.buffer as ArrayBuffer]);
        return;
      }
      case "decode": {
        if (!loaded) throw new Error("network not loaded");
        const t0 = now();
        const logits = glue.decode(new Float32Array(m.points), new Int32Array(m.labels), new Float32Array(m.box ?? []),
          m.maskInput ? new Float32Array(m.maskInput) : new Float32Array(0), m.multimask);
        const ms = now() - t0;
        const iou = glue.last_iou(), obj = glue.last_object_score(), [, masks, tokens] = glue.last_shape();
        const res: Res = { type: "decoded", id: m.id, ms, masks, tokens,
          logits: logits.buffer as ArrayBuffer, iou: Array.from(iou), objectScore: obj[0] };
        post(res, [logits.buffer as ArrayBuffer]);
        return;
      }
      case "amg": {
        if (!loaded) throw new Error("network not loaded");
        const run = m.id;
        amgRun = run;
        const { w, h } = size;
        const grid = P.pointGrid(m.pointsPerSide).map(([x, y]) => [x * w, y * h] as [number, number]);
        const batch = Math.max(1, m.batch);
        const t0 = now();
        let decodeMs = 0, postMs = 0;
        for (let s = 0; s < grid.length; s += batch) {
          if (amgRun !== run) { post({ type: "amgCancelled", id: run }); return; }
          const pts = grid.slice(s, s + batch);
          const t1 = now();
          const logits = glue.decode_points(new Float32Array(pts.flat()));
          const t2 = now();
          const [count, masks] = glue.last_shape();
          const cands = P.batchCandidates(logits, glue.last_iou(), count, masks, pts, s, w, h, m.stabilityOffset);
          const t3 = now();
          decodeMs += t2 - t1; postMs += t3 - t2;
          const msgs: CandidateMsg[] = cands.map((c) => ({ point: c.point, output: c.output, px: c.px, py: c.py,
            iou: c.iou, stability: c.stability, box: c.box, area: c.area, mask: c.mask ? (c.mask.buffer as ArrayBuffer) : null }));
          post({ type: "amgBatch", id: run, done: Math.min(grid.length, s + batch), total: grid.length, candidates: msgs,
            elapsedMs: now() - t0 }, msgs.flatMap((c) => (c.mask ? [c.mask] : [])));
          // Let a cancel message in between batches.
          await new Promise((r) => setTimeout(r, 0));
        }
        if (amgRun === run) post({ type: "amgDone", id: run, ms: now() - t0, decodeMs, postMs, points: grid.length });
        amgRun = 0;
        return;
      }
      case "cancel": {
        amgRun = 0;
        return;
      }
    }
  }

  // One request at a time, in order (a cancel goes straight through).
  let queue: Promise<void> = Promise.resolve();
  ctx.onmessage = (e) => {
    const m = e.data;
    if (m.type === "cancel") { amgRun = 0; return; }
    queue = queue.then(() => handle(m)).catch((err: unknown) => {
      post({ type: "error", id: "id" in m ? m.id : 0, message: err instanceof Error ? err.message : String(err) });
    });
  };
}
