# cellpose: Cellpose 3 in the browser (Rust → WebAssembly)

Used by the demo `demos/src/cellposenet/` (`data-demo="cellposenet"`); tested by
`demos/test/cellpose-wasm-test.ts`. Built by `make wasm` into `demos/dist/wasm/cellpose/`.

## What runs in Wasm

Everything that is numeric and heavy, as plain functions over flat arrays:

- **The network**: the Cellpose 3 U-net (`cyto3` and `nuclei`, 6.6 M parameters):
  `load_network(bytes)` (safetensors, F16/BF16/F32, PyTorch names of `resnet_torch.py`) and
  `run_tile(input, h, w)`: (2, h, w) → (3, h, w) = flow dY, flow dX, cell probability (logit),
  plus `last_style()` (the 256-long style vector). Convolutions are matrix products with the
  batch norm + ReLU folded into the input packing; the 4×8 block kernel uses 128-bit SIMD
  (`simd128`, enabled per function, so no RUSTFLAGS are needed).
- **The decoding** (cellpose 3.1 `dynamics.py`, `metrics.flow_error`,
  `utils.fill_holes_and_remove_small_masks`, CPU path, same label order):
  `compute_masks(...)` (Euler tracking of every pixel above the probability threshold, histogram
  of end points, seeds, masks, flow-error check, holes, small masks) with the intermediate
  results through `last_masks_raw`, `last_flow_errors`, `last_end_y/x`, `last_sink_histogram`,
  `last_seeds`; `finish_masks` (only the flow check and clean-up, for a new flow threshold or
  minimum size), `trace_paths` (paths of chosen pixels), `masks_to_flows`, `flow_errors`.

TypeScript (`demos/src/cellposenet/pipeline.ts`) does the cheap parts around the network,
following cellpose 3.1: channel choice, 1–99 percentile normalisation, rescaling by
diam_mean / diameter (bilinear), padding, 224 × 224 tiles with 10 % overlap, tapered
averaging, the size model (style → diameter). The tiles run in parallel Web Workers (the demo
bundle is its own worker script; the 133 kB module is inlined in it).

## Why not the cellpose-rs crate directly

[cellpose-rs](https://github.com/henriksson-lab/cellpose-rs) (checked at 4e428d0) implements
**Cellpose-SAM / CP-DINO only** (a ViT-L encoder, ~300 M parameters, 1.2 GB checkpoint, about
600 MB even in f16): too large for the course site (GitHub Pages, ~1 GB in total) and far too
slow on a CPU in Wasm (hundreds of GFLOP per 256² tile). It also depends on candle (git fork),
rayon, reqwest, hf-hub, indicatif and file I/O, and uses `std::time::Instant` in its dynamics
(which panics on wasm32-unknown-unknown). So this crate re-implements the Cellpose 3 U-net and
the 2D decoding in ~1400 lines without dependencies (except wasm-bindgen), and is checked
against Python Cellpose 3.1.1.2 and against cellpose-rs's `dynamics::compute_masks` (below).
cellpose-rs was not modified.

## Weights (tools/models/manifest.json)

| file | from | stored as | bytes |
|---|---|---|---|
| `cellpose/cyto3.safetensors` | cellpose.org `cyto3` (26.6 MB, f32) | **bf16** | 13,255,938 |
| `cellpose/nuclei.safetensors` | cellpose.org `nucleitorch_0` | f16 | 13,255,534 |
| `cellpose/size_cyto3.json` | `size_cyto3.npy` | JSON | 11,142 |

cyto3 cannot be stored in f16: its batch-norm running variances reach 3.3 × 10⁶, beyond f16's
range (65504), so they become `inf` and the network outputs a constant (0 masks). bf16 keeps
f32's range; with bf16 weights Python Cellpose finds the same masks as with f32 (AP@0.5 =
1.000 on cellpose-img02, 0.995 on cellpose-img00). The nuclei model is fine in f16 (identical
masks on bbbc039-A02_s1, AP 0.997 on dsb-fluo-dense). The nuclei checkpoint has no
`diam_mean`; the demo uses 17 (Cellpose's value for this model).

## Agreement with Python Cellpose 3.1.1.2 and cellpose-rs

Python reference runs used the same bf16/f16 weights (loaded into Cellpose 3.1.1.2's `CPnet`).

- Network, one 224 × 224 tile: max |difference| of flows 8 × 10⁻⁶ and cell probability
  5 × 10⁻⁵ (cyto3), 1.4 × 10⁻⁴ / 8.8 × 10⁻⁴ (nuclei); style vector 2 × 10⁻⁷ / 4 × 10⁻⁶.
- Decoding on Python's flows: labels identical to Python at 100 % of the pixels
  (cellpose-img02 full image, 467 × 349, 102 masks; dsb-fluo-small, 33 masks).
- Whole route in the test (PNG → channels → normalisation → rescaling → tiles → Wasm network →
  Wasm decoding) on two crops: labels identical to Python at 100.00 % of the pixels (30 and 14
  masks, one crop with rescaling 17/13).
- cellpose-rs `dynamics::compute_masks` on the same flows (it follows the newer Cellpose 4
  dynamics and numbers labels differently): same number of masks, the same objects at 99.84–100 %
  of the pixels.

## Benchmark (headless Chrome 104, Intel Xeon Gold 6138 @ 2.0 GHz, local server)

| | |
|---|---|
| Wasm module | 132,604 bytes (62 kB gzipped), inlined in `dist/cellposenet.js` (225 kB, 99 kB gzipped) |
| Weights | 13.3 MB per model (bf16 / f16), downloaded once, then from the Cache API |
| Load | download 0.2 s (local server, cold cache); 0.1–0.2 s for the workers to parse and pack the weights |
| One 224 × 224 tile | 2.1 s (31 GFLOP, ~15 GFLOP/s on one core); node 20: 2.1 s |
| **512 × 512 image** (cellpose-img00, cyto3, diameter 30 → 9 tiles) | network **18.7 s on 1 worker, 6.3 s on 4 workers**; decoding 1.4 s (2.4 s on the first run, before the JIT has optimised it) |
| 467 × 349 (cellpose-img02, 6 tiles) | network 4.5 s on 4 workers; decoding 0.55–1.0 s; 102 cells (= Python) |
| 696 × 520 nuclei (bbbc039, diameter 29 → rescaled ×0.59, 6 tiles) | network 4.4 s on 4 workers; decoding 1.1 s; F1 0.92 vs ground truth |

Notes: `opt-level = 3` instead of the usual `"s"`: with `"s"` a tile takes 3.5 s instead of
2.1 s (the module is 5 kB smaller). The demo uses min(4, cores − 1) workers (`data-workers`
overrides; each worker holds ~26 MB of f32 weights plus up to ~100 MB of activations). A
flow-threshold or minimum-size change only re-runs `finish_masks` (milliseconds); a new
probability threshold or number of steps re-runs the tracking (about 0.5–1.5 s); a new image,
channel choice or diameter re-runs the network.

Possible speed-ups not done: relaxed-SIMD fused multiply–add (not in every browser), Winograd
3 × 3 convolutions (~2× fewer multiplications), WebGPU.
