// The network side of the yolonet demo, in a Web Worker so the page stays responsive while the
// network runs (about a second at 512 x 512 on a laptop CPU). The worker is the same bundle
// (demos/dist/yolonet.js) started with `new Worker(...)`; main.ts calls runWorker() when there
// is no document. Messages: protocol.ts.
import * as glue from "../../dist/wasm/yolo/yolo.js";
import { loadWasm, loadModelFile } from "../lib/wasm";
import type { Req, Res } from "./protocol";

export const WEIGHTS_KEY = "yolo/yolo11n-coco.safetensors";

interface WorkerScope {
  postMessage(m: Res, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<Req>) => void) | null;
}

export function runWorker(): void {
  const ctx = self as unknown as WorkerScope;
  const post = (m: Res, transfer: Transferable[] = []) => ctx.postMessage(m, transfer);
  let loaded = false;
  const now = () => performance.now();

  async function handle(m: Req): Promise<void> {
    if (m.type === "init") {
      const t0 = now();
      let wasmBytes = 0;
      await loadWasm(glue, "yolo", { base: m.base, onProgress: (l, t) => { wasmBytes = l; post({ type: "progress", what: "module", loaded: l, total: t }); } });
      const w = await loadModelFile(WEIGHTS_KEY, { base: m.base, onProgress: (l, t) => post({ type: "progress", what: "weights", loaded: l, total: t }) });
      const t1 = now();
      const report = glue.load(w);
      loaded = true;
      post({ type: "ready", wasmBytes, weightBytes: w.length, fetchMs: t1 - t0, buildMs: now() - t1, report });
      return;
    }
    if (!loaded) throw new Error("network not loaded");
    const t0 = now();
    const raw = glue.run(new Float32Array(m.input), m.size);
    const t1 = now();
    const decoded = glue.decode(raw, m.size);
    const t2 = now();
    post({ type: "result", id: m.id, size: m.size, raw: raw.buffer as ArrayBuffer, decoded: decoded.buffer as ArrayBuffer,
      netMs: t1 - t0, decodeMs: t2 - t1 }, [raw.buffer as ArrayBuffer, decoded.buffer as ArrayBuffer]);
  }

  ctx.onmessage = (e) => {
    handle(e.data).catch((err) => post({ type: "error", message: (err as Error)?.message ?? String(err) }));
  };
}
