# `sam`: SAM 2.1 (Hiera tiny) in WebAssembly

The whole network runs in the browser: image encoder, prompt encoder and mask decoder of
SAM 2.1 Hiera tiny, from the henriksson-lab crate **sam2-rs** (Candle, CPU backend), compiled to
`wasm32-unknown-unknown`. Used by the demo `demos/src/samnet/` (in a Web Worker); tested by
`demos/test/sam-wasm-test.ts`.

- Build: `make wasm` (or `demos/wasm/build.sh sam`) → `demos/dist/wasm/sam/{sam.js, sam_bg.wasm}`.
  The module is **fetched** (`loadWasm(glue, "sam")`), not inlined.
- Weights: `make models` → `demos/dist/models/sam2/sam2.1_hiera_tiny.safetensors` (manifest entry
  `sam2.1-hiera-tiny`: the official checkpoint, converted by sam2-rs `tools/convert_checkpoint.py`,
  stored as f16; Candle reads it into f32).
- Two dependencies are patched copies, because the published sources do not compile or do not
  run on wasm32: see `vendor/README.md`.

## API (JavaScript, from the wasm-bindgen glue)

Free functions over one network per module instance (no exported classes):

| Function | Does |
|---|---|
| `load(weights: Uint8Array)` | build the network from the safetensors bytes |
| `encode(rgb: Uint8Array, w, h)` | image encoder (resize to 1024², normalise, Hiera + FPN, skip projections), **once per image**; result cached |
| `embedding(): Float32Array` | the cached image embedding, 256 × 64 × 64 (channel-major) |
| `decode(points, labels, box, maskInput, multimask): Float32Array` | prompt encoder + mask decoder, **per click**: masks × 256 × 256 logits (3 with `multimask`). Points/box in image pixels; labels 1/0; `box` empty or x0,y0,x1,y1; `maskInput` empty or a 256² logit plane |
| `decode_points(points): Float32Array` | a batch of single positive points (automatic mask generation): count × 3 × 256 × 256 |
| `last_iou()`, `last_object_score()`, `last_shape()` | scores of the last decode, and [count, masks, prompt tokens] |

Restoring masks to the image (bilinear, half-pixel centres, threshold 0), the stability score,
the filters and box NMS of automatic mask generation are in TypeScript
(`demos/src/samnet/post.ts`), so the demo can show and the test can check each step.

Natively (`cargo build`, rlib), `sam::Sam` exposes the same steps (`from_safetensors`,
`encode_rgb8`, `decode_prompts`, `decode_point_batch`); its output equals sam2-rs's
`Sam2ImagePredictor::predict_cached_prompts` exactly (max |logit difference| 0, all three test
prompts).

## Benchmark (2026-11-13)

512 × 512 sample image (`images/stardist-img2d.png`), one CPU core, no SIMD
(the build uses `wasm32-unknown-unknown` without `+simd128`, as `demos/wasm/build.sh` does).
Machine: 40-core Linux server; a laptop will be in the same range or slower. Release
profile `opt-level = 3` (not the usual `"s"`): 7 % faster encoder (25.1 vs 27.0 s) for 0.5 MB
more module.

| | Headless Chrome (worker) | node 20 (test) | native sam2-rs, 8 threads |
|---|---:|---:|---:|
| Module `sam_bg.wasm` | 2.56 MB (2 557 650 B) | | |
| Weights (f16 safetensors) | 78.0 MB (77 980 836 B) | | |
| Download + cache, first load (local server) | 1.7 s | | |
| Build network from weights | 0.4 s | 0.47 s | 0.1–0.2 s |
| Image encoder (once per image) | 26.6 s | 24.5 s | 2.5 s |
| Prompt encoder + mask decoder (per click, 3 masks) | 0.41–0.43 s | 0.34 s | 0.06 s |
| Automatic masks, 8 × 8 grid (64 points × 3 candidates) | 21.7 s (decoder 21.1 s) | 21.6 s | 5.4 s |

The decoder costs about 0.33 s per prompt point in Wasm, so a 16 × 16 grid takes about 90 s;
the demo defaults to 8 × 8. Agreement with native (test): predicted IoU within 0.01,
restored logits within 0.05 at probe pixels, mask areas within 0.5 %, and the same 5 automatic
masks (8 × 8 grid, default thresholds).

## Faster later (not done)

- **SIMD** (`-C target-feature=+simd128`, all current browsers): needs (a) the Wasm build to pass
  that flag (`demos/wasm/build.sh` or a `.cargo/config.toml` at the repository root) and (b) a fix in
  candle-core at the pinned revision, which then fails to compile (`CurrentCpuF16`/`CurrentCpuBF16`
  missing for the simd128 path). Expected gain: a few times on the matrix products.
- **Threads** (rayon on Web Workers) need cross-origin isolation headers, which GitHub Pages
  cannot set.
- WebGPU through a GPU backend.
