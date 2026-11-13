// Messages between the yolonet page (main.ts) and its Web Worker (worker.ts).
export type Req =
  | { type: "init"; base: string }
  /** input: 3 x size x size float32 (CHW, 0..1), transferred. */
  | { type: "run"; id: number; size: number; input: ArrayBuffer };

export type Res =
  | { type: "progress"; what: "module" | "weights"; loaded: number; total: number }
  | { type: "ready"; wasmBytes: number; weightBytes: number; fetchMs: number; buildMs: number; report: string }
  /** raw: A x 144, decoded: A x 88 (see post.ts); both float32, transferred. */
  | { type: "result"; id: number; size: number; raw: ArrayBuffer; decoded: ArrayBuffer; netMs: number; decodeMs: number }
  | { type: "error"; message: string };
