// Messages between the stardistnet page (main.ts) and its worker (worker.ts).

export type Req =
  | { type: "init"; base: string; engine: number }
  /** grey: Float32Array buffer (raw intensities, w*h, row-major); normalised in the worker. */
  | { type: "predict"; id: number; grey: ArrayBuffer; w: number; h: number; pmin: number; pmax: number }
  /** Decoding of the last prediction with these thresholds. */
  | { type: "decode"; id: number; probThresh: number; nmsThresh: number };

export type Res =
  | { type: "progress"; what: "module" | "weights"; loaded: number; total: number }
  | { type: "ready"; wasmBytes: number; weightBytes: number; fetchMs: number; buildMs: number;
      nRays: number; grid: number; probThresh: number; nmsThresh: number }
  /** prob: Float32 (oh*ow); dist: Float32 (oh*ow*nRays, rays last). */
  | { type: "predicted"; id: number; ms: number; ow: number; oh: number; prob: ArrayBuffer; dist: ArrayBuffer }
  /** points: Float32 (count*2, y x); coords: Float32 (count*2*nRays: ys then xs);
   *  labels: Uint32 (h*w). Objects sorted by decreasing probability. */
  | { type: "decoded"; id: number; ms: number; candidates: number; count: number; points: ArrayBuffer;
      probs: ArrayBuffer; coords: ArrayBuffer; labels: ArrayBuffer }
  | { type: "error"; id: number; message: string };
