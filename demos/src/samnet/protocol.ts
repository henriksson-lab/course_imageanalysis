// Messages between the samnet page (main.ts) and its worker (worker.ts).
import type { Box } from "./post";

export type Req =
  | { type: "init"; base: string }
  | { type: "encode"; id: number; rgb: ArrayBuffer; w: number; h: number }
  | { type: "decode"; id: number; points: number[]; labels: number[]; box: number[] | null; maskInput: ArrayBuffer | null; multimask: boolean }
  | { type: "amg"; id: number; pointsPerSide: number; batch: number; stabilityOffset: number }
  | { type: "cancel" };

/** A candidate of automatic mask generation as sent from the worker (mask: w*h bytes or null). */
export interface CandidateMsg {
  point: number; output: number; px: number; py: number;
  iou: number; stability: number; box: Box | null; area: number; mask: ArrayBuffer | null;
}

export type Res =
  | { type: "progress"; what: "module" | "weights"; loaded: number; total: number }
  | { type: "ready"; wasmBytes: number; weightBytes: number; fetchMs: number; buildMs: number }
  | { type: "encoded"; id: number; ms: number; pca: ArrayBuffer }
  | { type: "decoded"; id: number; ms: number; masks: number; tokens: number; logits: ArrayBuffer; iou: number[]; objectScore: number }
  | { type: "amgBatch"; id: number; done: number; total: number; candidates: CandidateMsg[]; elapsedMs: number }
  | { type: "amgDone"; id: number; ms: number; decodeMs: number; postMs: number; points: number }
  | { type: "amgCancelled"; id: number }
  | { type: "error"; id: number; message: string };
